#!/usr/bin/env python3
"""Independently verify the immutable PGIR-014 campaign freeze.

This verifier uses only the Python standard library.  It does not import the
builder, semantic implementation, or supervisor identity code.  A successful
exit means the frozen bytes, Git bindings, canonical identities, dependency
graph, typed rejection receipt, and fail-closed no-go decision all replayed.
It never turns a verified no-go artifact into execution authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FREEZE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FREEZE_DIR.parents[3]
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"
TASK_BOARD = REPOSITORY_ROOT / "docs/architecture/proof_grounded_ir_learning.todo.md"
TASK_IDENTITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/task-identity@1"

REQUIRED_BINDING_NAMES = {
    "compiler",
    "corpus",
    "decompiler",
    "example_contracts",
    "gap_matrix",
    "lineage",
    "policy",
    "rights",
    "schema_registry",
    "source_snapshots",
    "split",
    "tokenizer_policy",
}


class FreezeVerificationError(ValueError):
    """Raised when any identity, reference, or fail-closed gate drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeVerificationError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise FreezeVerificationError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise FreezeVerificationError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def dag_json_cid(value: Any) -> str:
    digest = hashlib.sha256(canonical_bytes(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def raw_cid(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode(
        "ascii"
    ).rstrip("=").lower()


def supervisor_identity(namespace: str, value: Any) -> str:
    return dag_json_cid({"namespace": namespace, "value": value})


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or absent JSON file: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                FreezeVerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                FreezeVerificationError(f"non-finite number {raw!r} in {path}")
            ),
        )
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    validate_value(value)
    return value


def verify_projection_identity(
    value: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> None:
    require(cid_field in value, f"missing {cid_field}")
    projection = dict(value)
    claimed_cid = projection.pop(cid_field)
    claimed_sha = projection.pop(sha_field) if sha_field else None
    require(claimed_cid == dag_json_cid(projection), f"{cid_field} does not match projection")
    if sha_field:
        expected_sha = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
        require(claimed_sha == expected_sha, f"{sha_field} does not match projection")


def git(*args: str, cwd: Path) -> str:
    process = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode:
        raise FreezeVerificationError(
            f"git {' '.join(args)} failed in {cwd}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git SHA-1 object identity.


def verify_file_binding(
    binding: Mapping[str, Any], *, source_tree: str, datasets_commit: str
) -> None:
    expected = {"path", "repository", "raw_cid", "sha256", "size_bytes"}
    require(expected.issubset(binding), f"incomplete file binding: {binding}")
    relative = str(binding["path"])
    normalized = relative.replace("\\", "/")
    require(
        normalized == relative
        and normalized
        and not normalized.startswith("/")
        and ".." not in Path(normalized).parts,
        f"unsafe bound path: {relative!r}",
    )
    path = REPOSITORY_ROOT / normalized
    require(path.is_file() and not path.is_symlink(), f"bound file is absent or unsafe: {relative}")
    resolved = path.resolve(strict=True)
    require(
        os.path.commonpath((str(REPOSITORY_ROOT.resolve()), str(resolved)))
        == str(REPOSITORY_ROOT.resolve()),
        f"bound file escapes repository root: {relative}",
    )
    data = path.read_bytes()
    require(binding["size_bytes"] == len(data), f"size drift: {relative}")
    require(
        binding["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(),
        f"SHA-256 drift: {relative}",
    )
    require(binding["raw_cid"] == raw_cid(data), f"raw CID drift: {relative}")
    if "git_blob" not in binding:
        require(binding["repository"] == "pgir-freeze", f"missing Git blob binding: {relative}")
        return
    require(binding["git_blob"] == git_blob_id(data), f"Git blob digest drift: {relative}")
    repository = binding["repository"]
    if repository == "ipfs_datasets_py":
        cwd = DATASETS_ROOT
        commit = datasets_commit
        object_path = relative.removeprefix("ipfs_datasets_py/")
    elif repository == "ipfs_accelerate_py":
        cwd = REPOSITORY_ROOT
        commit = source_tree
        object_path = relative
    else:
        raise FreezeVerificationError(f"unsupported Git repository {repository!r}")
    committed_blob = git("rev-parse", f"{commit}:{object_path}", cwd=cwd)
    require(committed_blob == binding["git_blob"], f"commit/blob mismatch: {relative}")


def all_file_bindings(value: Any) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if {"path", "repository", "raw_cid", "sha256", "size_bytes"}.issubset(item):
                existing = result.get(item["path"])
                require(existing in (None, item), f"conflicting file bindings for {item['path']}")
                result[item["path"]] = item
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return result


def verify_nested_binding_identities(value: Any, path: str = "$") -> int:
    count = 0
    if isinstance(value, dict):
        if "binding_cid" in value:
            projection = dict(value)
            claimed = projection.pop("binding_cid")
            require(claimed == dag_json_cid(projection), f"binding CID drift at {path}")
            count += 1
        for key, child in value.items():
            count += verify_nested_binding_identities(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            count += verify_nested_binding_identities(child, f"{path}[{index}]")
    return count


def normalize_identity_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_identity_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text).rstrip("/")


def split_csv(value: str) -> list[str]:
    return [
        item
        for raw in str(value or "").split(",")
        if (item := raw.strip()) and item.lower() not in {"none", "n/a"}
    ]


def task_identity(task: Mapping[str, Any], *, semantic_key: str = "") -> dict[str, str]:
    metadata = dict(task.get("metadata") or {})
    semantic = {
        key: value
        for key, value in {
            "title": normalize_identity_text(task.get("title")),
            "outputs": sorted(
                {
                    normalize_identity_path(item)
                    for item in task.get("outputs", [])
                    if normalize_identity_path(item)
                }
            ),
            "acceptance": [
                normalize_identity_text(item)
                for item in split_csv(str(task.get("acceptance") or metadata.get("acceptance criteria") or ""))
                if normalize_identity_text(item)
            ],
            "evidence": sorted(
                {
                    normalize_identity_text(item)
                    for item in split_csv(str(metadata.get("missing evidence") or ""))
                    if normalize_identity_text(item)
                }
            ),
            "goal": normalize_identity_text(
                metadata.get("goal id")
                or metadata.get("goal packet key")
                or metadata.get("goal")
            ),
            "semantic_hint": normalize_identity_text(
                semantic_key
                or metadata.get("semantic key")
                or metadata.get("bundle key")
                or metadata.get("work scope")
                or metadata.get("fingerprint")
            ),
        }.items()
        if value
    }
    require(bool(semantic), f"task {task.get('task_id')} has no semantic identity")
    material = {"schema": TASK_IDENTITY_SCHEMA, "semantic": semantic}
    fingerprint = hashlib.sha256(canonical_bytes(material)).hexdigest()
    return {
        "canonical_task_key": f"task/v1/{fingerprint}",
        "canonical_task_cid": dag_json_cid(material),
        "semantic_fingerprint": fingerprint,
    }


def parse_task_board() -> dict[str, dict[str, Any]]:
    require(TASK_BOARD.is_file() and not TASK_BOARD.is_symlink(), "protected task board is absent")
    tasks: dict[str, dict[str, Any]] = {}
    current_id = ""
    current_title = ""
    block: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, block
        if not current_id:
            return
        metadata: dict[str, str] = {}
        for line in block:
            stripped = line.strip()
            if stripped.startswith("- ") and ":" in stripped:
                key, value = stripped[2:].split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        task = {
            "task_id": current_id,
            "title": current_title,
            "depends_on": split_csv(metadata.get("depends on", "")),
            "outputs": split_csv(metadata.get("outputs", "")),
            "acceptance": metadata.get("acceptance", ""),
            "metadata": metadata,
        }
        task.update(task_identity(task))
        require(current_id not in tasks, f"duplicate task {current_id}")
        tasks[current_id] = task
        current_id = ""
        current_title = ""
        block = []

    for line in TASK_BOARD.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            if line.startswith("## PGIR-"):
                header = line[3:].strip()
                current_id, _, current_title = header.partition(" ")
            continue
        if current_id:
            block.append(line)
    flush()
    return tasks


def descendants(tasks: Mapping[str, Mapping[str, Any]], ancestor: str) -> set[str]:
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for task_id, task in tasks.items():
            dependencies = set(task.get("depends_on") or ())
            if task_id != ancestor and task_id not in result and (
                ancestor in dependencies or dependencies.intersection(result)
            ):
                result.add(task_id)
                changed = True
    return result


def verify_task_revisions(
    revisions: Mapping[str, Any], root: Mapping[str, Any]
) -> None:
    verify_projection_identity(
        revisions, cid_field="revision_set_cid", sha_field="revision_set_sha256"
    )
    require(revisions["campaign_input_root_cid"] == root["root_cid"], "revision/root mismatch")
    tasks = parse_task_board()
    expected_descendants = descendants(tasks, "PGIR-014")
    records = revisions.get("revisions")
    require(isinstance(records, list), "revision records must be a list")
    require(len(records) == revisions["descendant_task_count"] == 26, "descendant count drift")
    require({item["task_id"] for item in records} == expected_descendants, "descendant set drift")
    require(revisions["lease_eligible_count"] == 0, "no-go revision set enabled a lease")
    semantic_key = f"pgir-campaign-input-root@1:{root['root_cid']}"
    graph_tasks: list[dict[str, Any]] = []
    for item in records:
        task = tasks[item["task_id"]]
        current = task_identity(task)
        revised = task_identity(task, semantic_key=semantic_key)
        require(item["current_task_cid"] == current["canonical_task_cid"], f"current CID drift for {item['task_id']}")
        require(item["revised_task_cid"] == revised["canonical_task_cid"], f"revised CID drift for {item['task_id']}")
        require(item["revised_task_key"] == revised["canonical_task_key"], f"revised key drift for {item['task_id']}")
        require(item["semantic_fingerprint"] == revised["semantic_fingerprint"], f"fingerprint drift for {item['task_id']}")
        require(item["input_binding"]["semantic_key"] == semantic_key, f"semantic key drift for {item['task_id']}")
        require(item["input_binding"]["decision"] == "no_go", f"decision drift for {item['task_id']}")
        require(item["lease_eligible"] is False, f"no-go task became lease eligible: {item['task_id']}")
        require(item["depends_on"] == task["depends_on"], f"dependency drift for {item['task_id']}")
        graph_tasks.append(
            {
                "task_id": item["task_id"],
                "task_cid": item["revised_task_cid"],
                "depends_on": item["depends_on"],
            }
        )
    graph_projection = {
        "schema": "PGIRDescendantTaskGraph@1",
        "root_task_id": "PGIR-014",
        "tasks": graph_tasks,
    }
    require(
        revisions["candidate_graph_cid"] == dag_json_cid(graph_projection),
        "candidate descendant graph CID drift",
    )


def verify_plan_admission(
    receipt: Mapping[str, Any], root: Mapping[str, Any], revisions: Mapping[str, Any]
) -> None:
    expected_top = {
        "schema",
        "compiler_version",
        "requirement_id",
        "request_id",
        "candidate_plan_id",
        "candidate_graph_id",
        "repository_tree_id",
        "verdict",
        "admitted",
        "semantic_roots",
        "intent_result_id",
        "legal_result_ids",
        "legal_permission_ids",
        "security_decision_ids",
        "security_grant_ids",
        "checked_dependency_ids",
        "checked_assumption_ids",
        "generated_formula_ids",
        "proof_result_ids",
        "checked_validation_ids",
        "cve_security_evidence_ids",
        "rejection_reasons",
        "reason_codes",
        "counterexamples",
        "local_replan_action_ids",
        "closure_id",
        "permissions_are_grants",
        "generated_formulas_are_proofs",
        "authorizes_execution",
        "receipt_id",
    }
    require(set(receipt) == expected_top, "plan-admission receipt has an open or incomplete shape")
    require(
        receipt["schema"]
        == "ipfs_accelerate_py/agent-supervisor/plan-admission-receipt@1",
        "wrong plan-admission schema",
    )
    projection = dict(receipt)
    claimed = projection.pop("receipt_id")
    require(
        claimed == supervisor_identity("plan-admission-receipt", projection),
        "plan-admission receipt identity drift",
    )
    require(receipt["verdict"] == "rejected" and receipt["admitted"] is False, "no-go plan was admitted")
    require(receipt["authorizes_execution"] is False, "plan receipt claims execution authority")
    require(receipt["permissions_are_grants"] is False, "legal permissions became grants")
    require(receipt["generated_formulas_are_proofs"] is False, "generated formulas became proofs")
    require(receipt["candidate_plan_id"] == revisions["revision_set_cid"], "candidate plan mismatch")
    require(receipt["candidate_graph_id"] == revisions["candidate_graph_cid"], "candidate graph mismatch")
    require(receipt["repository_tree_id"] == root["repository"]["source_tree_id"], "receipt tree mismatch")
    require(receipt["semantic_roots"]["campaign"] == root["root_cid"], "receipt root mismatch")
    rejections = receipt["rejection_reasons"]
    require(isinstance(rejections, list) and rejections, "rejected receipt lacks reasons")
    for rejection in rejections:
        projection = dict(rejection)
        rejection_id = projection.pop("rejection_id")
        require(
            rejection_id == supervisor_identity("plan-admission-rejection", projection),
            "plan rejection identity drift",
        )
    require(
        receipt["reason_codes"] == sorted({item["code"] for item in rejections}),
        "plan rejection reason projection drift",
    )


def verify_manifest(manifest: Mapping[str, Any], root: Mapping[str, Any]) -> None:
    verify_projection_identity(manifest, cid_field="manifest_cid")
    require(manifest["campaign_input_root_cid"] == root["root_cid"], "manifest/root mismatch")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "manifest artifacts must be an object")
    require(manifest["artifact_count"] == len(artifacts), "manifest artifact count drift")
    for name, binding in artifacts.items():
        require(
            name == Path(name).name and not Path(name).is_absolute(),
            f"unsafe bundle artifact name: {name}",
        )
        path = FREEZE_DIR / name
        require(path.is_file() and not path.is_symlink(), f"bundle artifact absent: {name}")
        data = path.read_bytes()
        require(binding["size_bytes"] == len(data), f"bundle size drift: {name}")
        require(binding["raw_cid"] == raw_cid(data), f"bundle raw CID drift: {name}")
        require(
            binding["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest(),
            f"bundle SHA-256 drift: {name}",
        )


def verify_source_self_identities(root: Mapping[str, Any]) -> None:
    source_manifest = strict_json(
        REPOSITORY_ROOT
        / "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json"
    )
    projection = dict(source_manifest)
    claimed = projection.pop("manifest_cid")
    require(claimed == dag_json_cid(projection), "source revision manifest CID drift")

    gap = strict_json(
        DATASETS_ROOT / "docs/architecture/proof_grounded_ir_learning/gap_matrix.json"
    )
    projection = dict(gap)
    claimed_cid = projection.pop("matrix_cid")
    claimed_sha = projection.pop("matrix_sha256")
    require(claimed_cid == dag_json_cid(projection), "gap matrix CID drift")
    require(
        claimed_sha == hashlib.sha256(canonical_bytes(projection)).hexdigest(),
        "gap matrix SHA-256 drift",
    )
    require(root["bindings"]["gap_matrix"]["matrix_cid"] == claimed_cid, "root/gap mismatch")


def verify_semantic_gates(root: Mapping[str, Any]) -> None:
    corpus_dir = DATASETS_ROOT / "data/ir_learning/corpora"
    split_dir = DATASETS_ROOT / "data/ir_learning/splits"
    corpus = strict_json(corpus_dir / "corpus_root.json")
    rights = strict_json(corpus_dir / "rights_manifest.json")
    reconciliation = strict_json(corpus_dir / "reconciliation_receipt.json")
    split = strict_json(split_dir / "split_root.json")
    leakage = strict_json(split_dir / "leakage_report.json")
    require(corpus["source_count"] == rights["source_count"] == reconciliation["source_count"] == 7173, "source count does not reconcile")
    require(corpus["derived_count"] == reconciliation["derived_count"] == 38690, "derived count does not reconcile")
    require(corpus["patent_source_groups"] == reconciliation["patent_source_groups"] == 2174, "patent groups do not reconcile")
    require(corpus["training_admitted_rows"] == rights["training_admitted_rows"] == 0, "training rows unexpectedly admitted")
    require(len(rights["admitted_source_record_ids"]) == 0, "rights manifest admits source rows")
    require(len(rights["quarantined_source_record_ids"]) == 7173, "rights quarantine count drift")
    require(corpus["materialized"] is False, "no-go corpus unexpectedly materialized")
    require(split["leakage_passed"] is True, "split root fails leakage")
    require(leakage == {"blocked_operations": [], "passed": True, "violations": []}, "leakage receipt drift")
    insufficient = sorted(
        name for name, item in split["holdouts"].items() if item["status"] == "insufficient"
    )
    qualification = root["qualification"]
    require(qualification["decision"] == "no_go", "campaign no-go decision drift")
    require(qualification["lease_barrier"] == "closed", "campaign lease barrier opened")
    require(qualification["descendant_execution_authorized"] is False, "descendant execution was authorized")
    require(qualification["training_task_eligible_count"] == 0, "training task became eligible")
    require(qualification["training_admitted_rows"] == 0, "root claims admitted training rows")
    require(qualification["rights_quarantined_rows"] == 7173, "root rights count drift")
    require(qualification["insufficient_holdouts"] == insufficient, "root holdout projection drift")


def verify_all() -> dict[str, Any]:
    tokenizer = strict_json(FREEZE_DIR / "tokenizer_policy.json")
    root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    revisions = strict_json(FREEZE_DIR / "descendant_task_revisions.json")
    admission = strict_json(FREEZE_DIR / "plan_admission_receipt.json")
    verification = strict_json(FREEZE_DIR / "verification_receipt.json")
    manifest = strict_json(FREEZE_DIR / "manifest.v3.json")
    result = strict_json(FREEZE_DIR / "result.v3.json")

    verify_projection_identity(tokenizer, cid_field="policy_cid", sha_field="policy_sha256")
    require(tokenizer["training_policy"]["authorized"] is False, "tokenizer policy authorizes training")
    verify_projection_identity(root, cid_field="root_cid", sha_field="root_sha256")
    expected_root_keys = {
        "schema",
        "interface",
        "contract_version",
        "task_id",
        "objective_id",
        "objective_revision",
        "repository",
        "bindings",
        "referential_integrity",
        "qualification",
        "canonicalization",
        "supersession",
        "root_sha256",
        "root_cid",
    }
    require(set(root) == expected_root_keys, "campaign root has an open or incomplete shape")
    require(root["schema"] == "IRCampaignInputRoot@1", "wrong campaign root schema")
    require(set(root["bindings"]) == REQUIRED_BINDING_NAMES, "required binding domain drift")
    require(
        root["referential_integrity"]["required_binding_names"]
        == sorted(REQUIRED_BINDING_NAMES),
        "required binding projection drift",
    )
    require(root["referential_integrity"]["unresolved_identities"] == [], "unresolved identities remain")
    require(root["referential_integrity"]["hidden_labels_accessed"] is False, "hidden-label access claimed")
    require(root["referential_integrity"]["source_or_split_mutated"] is False, "source/split mutation claimed")
    require(root["supersession"]["immutable"] is True, "root is not immutable")
    require(root["supersession"]["replacement_policy"] == "supersede_never_overwrite", "unsafe root replacement policy")
    require(root["bindings"]["tokenizer_policy"]["policy_cid"] == tokenizer["policy_cid"], "root/tokenizer mismatch")

    nested_count = verify_nested_binding_identities(root)
    require(nested_count == len(REQUIRED_BINDING_NAMES), "nested binding identity count drift")
    source_tree = root["repository"]["source_tree_id"]
    datasets_commit = root["repository"]["datasets_commit"]
    require(re.fullmatch(r"[0-9a-f]{40}", source_tree) is not None, "invalid source tree commit")
    require(re.fullmatch(r"[0-9a-f]{40}", datasets_commit) is not None, "invalid datasets commit")
    git("cat-file", "-e", f"{source_tree}^{{commit}}", cwd=REPOSITORY_ROOT)
    git("cat-file", "-e", f"{datasets_commit}^{{commit}}", cwd=DATASETS_ROOT)
    file_bindings = all_file_bindings(root)
    require(len(file_bindings) >= 40, "campaign root binds too few exact source files")
    for binding in file_bindings.values():
        verify_file_binding(binding, source_tree=source_tree, datasets_commit=datasets_commit)
    expected_tree = git(
        "rev-parse", f"{datasets_commit}:ipfs_datasets_py/logic/ir_core", cwd=DATASETS_ROOT
    )
    require(root["bindings"]["schema_registry"]["tree_oid"] == expected_tree, "schema tree drift")

    verify_source_self_identities(root)
    verify_semantic_gates(root)
    verify_task_revisions(revisions, root)
    verify_plan_admission(admission, root, revisions)
    verify_projection_identity(verification, cid_field="receipt_cid")
    require(verification["campaign_input_root_cid"] == root["root_cid"], "verification/root mismatch")
    require(verification["revision_set_cid"] == revisions["revision_set_cid"], "verification/revision mismatch")
    require(verification["plan_admission_receipt_id"] == admission["receipt_id"], "verification/admission mismatch")
    require(verification["campaign_decision"] == "no_go", "verification changed campaign decision")
    require(verification["all_integrity_checks_passed"] is True, "persisted verification failed")
    require(verification["authorizes_execution"] is False, "verification claims execution authority")
    require(all(item["status"] == "passed" for item in verification["checks"]), "persisted check failed")
    verify_manifest(manifest, root)
    require(manifest["supersedes_manifest_cid"] == "baguqeerafyxup44ij426ipllqqbo6voszvz4uviil5dm5lokai6cfihrsaha", "manifest supersession drift")
    require(manifest["revision_set_cid"] == revisions["revision_set_cid"], "manifest/revision mismatch")
    require(manifest["plan_admission_receipt_id"] == admission["receipt_id"], "manifest/admission mismatch")
    require(manifest["verification_receipt_cid"] == verification["receipt_cid"], "manifest/verification mismatch")
    verify_projection_identity(result, cid_field="result_cid", sha_field="result_sha256")
    require(result["result_identity"] == "RESULT(PGIR-014)", "wrong task result identity")
    require(result["supersedes_result_cid"] == "baguqeeravwtoxdkhmlg4khg7wg5vqtiv6vq5byd2ecy27bkcjyvkjjoo4q3q", "result supersession drift")
    require(result["campaign_input_root_cid"] == root["root_cid"], "result/root mismatch")
    require(result["manifest_cid"] == manifest["manifest_cid"], "result/manifest mismatch")
    require(result["decision"] == "no_go" and result["disposition"] == "frozen_no_go", "result decision drift")
    require(result["descendant_execution_authorized"] is False, "result authorizes descendants")
    require(result["training_task_eligible_count"] == 0, "result enables training")
    require(result["unresolved_identities"] == [], "result carries unresolved identities")

    return {
        "schema": "PGIRFreezeIndependentVerificationOutcome@1",
        "verified": True,
        "campaign_decision": "no_go",
        "authorizes_execution": False,
        "campaign_input_root_cid": root["root_cid"],
        "manifest_cid": manifest["manifest_cid"],
        "revision_set_cid": revisions["revision_set_cid"],
        "plan_admission_receipt_id": admission["receipt_id"],
        "file_binding_count": len(file_bindings),
        "binding_domain_count": nested_count,
        "descendant_task_count": revisions["descendant_task_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="suppress successful output")
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)
    try:
        outcome = verify_all()
    except (FreezeVerificationError, KeyError, TypeError, OSError, ValueError) as exc:
        print(f"PGIR-014 freeze verification failed: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        if args.json:
            print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
        else:
            print(
                "PGIR-014 freeze verified: "
                f"{outcome['campaign_input_root_cid']} "
                "(campaign decision: no_go; execution authority: false)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
