"""Deliver an explicitly admitted public task requirement to a coding worker.

The owner verifies admission and pins this artifact in the implementation
command. The worker replays the manifest signature and exact source bindings;
it does not acquire owner credentials or treat requirements as execution policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .doctor_candidate_runner import _directory, _git, _read, _unique

SCHEMA = "supervisor-public-instruction@1"
INTENT_SCHEMA = "supervisor-public-instruction@2"
DIRECTORY = ".runtime/router-public-instruction"
MAX_BYTES = 1_000_000
MAX_INSTRUCTION_BYTES = 32_768
FIELDS = frozenset({"schema", "repository", "task_cid", "task_id", "manifest", "manifest_cid",
    "owner_identity", "owner_profile_id", "source_path", "source_sha256", "source_bytes",
    "completion_authority", "publication_authority", "scope_expansion_authority", "context_cid"})
INTENT_FIELDS = FIELDS | {"intent_plan_admission"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source(root, name):
    local._path(name)
    path = root / name
    parent = _directory(path.parent)
    try:
        return _read(parent, path.name)
    finally:
        os.close(parent)


def _current_sources(root, sources):
    total = 0
    for name, expected in sources.items():
        if set(expected) != {"sha256", "executable"} or type(expected["executable"]) is not bool:
            raise ValueError("exact signed source binding required")
        raw, info = _source(root, name)
        total += len(raw)
        if len(raw) > 1_000_000 or total > 4_000_000:
            raise ValueError("instruction source inventory exceeds bound")
        if {"sha256": _sha(raw), "executable": bool(info.st_mode & 0o111)} != expected:
            raise ValueError("public instruction source inventory is stale")


def _selection(payload):
    """Replay public signature and selection; the launch digest pins the owner."""
    manifest = local._verify_signature(payload["manifest"], SimpleNamespace(
        identity_did=payload["owner_identity"], profile_id=payload["owner_profile_id"]))
    if (content_identity(payload["manifest"]) != payload["manifest_cid"]
            or manifest["repository"] != payload["repository"]
            or manifest.get("schema") not in local.SUPPORTED_MANIFEST_SCHEMAS):
        raise ValueError("public instruction signed manifest differs")
    specs = [row for row in manifest["tasks"] if row["task_key"] == payload["task_id"]]
    path = payload["source_path"]
    local._path(path)
    intent = payload.get("schema") == INTENT_SCHEMA
    if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA and not intent:
        raise ValueError("signed IntentIR admission requires the task-specific instruction version")
    if intent and (manifest["schema"] != local.INTENT_MANIFEST_SCHEMA
            or local.decode_intent_requirement_contract(manifest)["source_path"] != path):
        raise ValueError("worker requirement context must select the signed IntentIR source")
    if (len(specs) != 1 or (not intent and path not in specs[0]["scope_paths"])
            or any(path == output["path"] for spec in manifest["tasks"] for output in spec["outputs"])
            or manifest["sources"].get(path, {}).get("sha256") != payload["source_sha256"]
            or type(payload["source_bytes"]) is not int or not 0 < payload["source_bytes"] <= MAX_INSTRUCTION_BYTES):
        raise ValueError("public instruction must be the exact declared read-only task input")
    return manifest


def _intent_context(payload, manifest, source_text):
    """Replay public signed admission without opening profile keys or state."""
    if payload["schema"] != INTENT_SCHEMA:
        return None
    local._validate_local_manifest_declarations(manifest)
    plan = payload["intent_plan_admission"]
    if (not isinstance(plan, dict) or set(plan) != {"graph", "receipt", "requirement_bindings"}
            or manifest["policy"] != local.LOCAL_POLICY):
        raise ValueError("exact public intent admission and local policy required")
    profile = SimpleNamespace(identity_did=payload["owner_identity"],
                              profile_id=payload["owner_profile_id"])
    graph = local.PromptGoalGraph.from_dict(plan["graph"])
    signed = local._verify_signature(plan["receipt"], profile)
    expected = local._planning_payload(graph, payload["manifest"], manifest, profile,
        manifest["sources"], plan["requirement_bindings"], requirement_source_text=source_text)
    if signed != expected:
        raise ValueError("worker intent admission differs from exact replayed planning receipt")
    tasks = [task for task in graph.tasks if task.task_cid == payload["task_cid"]]
    if len(tasks) != 1 or tasks[0].task_key != payload["task_id"]:
        raise ValueError("worker requirement context task differs from signed graph identity")
    from .intent_requirement_context import build_task_intent_context

    return build_task_intent_context(contract=local._verify_intent_requirement_text(manifest, source_text),
        graph=graph, coverage=expected["requirement_coverage"], task_cid=payload["task_cid"])


def prepare_public_instruction_context(*, repository: Path, admission: dict, task_cid: str,
                                       source_path: str, expected_source_sha256: str) -> dict:
    """Owner selects a specific signed input; ambient filenames are never used."""
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest, profile = verified["manifest"], verified["profile"]
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root or str(root) != manifest["repository"]:
        raise ValueError("public instruction requires the independently admitted repository")
    tasks = [row for row in verified["graph"].tasks if row.task_cid == task_cid]
    if len(tasks) != 1:
        raise ValueError("public instruction requires the exact admitted task")
    raw, _ = _source(root, source_path)
    if _sha(raw) != expected_source_sha256 or not raw.decode("utf-8").strip() or b"\0" in raw:
        raise ValueError("public instruction differs from the explicitly selected source bytes")
    payload = {"schema": SCHEMA, "repository": str(root), "task_cid": task_cid,
        "task_id": tasks[0].task_key, "manifest": admission["manifest"],
        "manifest_cid": verified["receipt"]["manifest_cid"],
        "owner_identity": profile.identity_did, "owner_profile_id": profile.profile_id,
        "source_path": source_path, "source_sha256": expected_source_sha256, "source_bytes": len(raw),
        "completion_authority": False, "publication_authority": False, "scope_expansion_authority": False}
    if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA:
        payload.update(schema=INTENT_SCHEMA, intent_plan_admission={
            "graph": admission["graph"], "receipt": admission["receipt"],
            "requirement_bindings": admission["requirement_bindings"]})
    selected = _selection(payload)
    requirements = _intent_context(payload, selected, raw.decode("utf-8"))
    payload["context_cid"] = content_identity(payload)
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    if len(encoded) > MAX_BYTES:
        raise ValueError("public instruction artifact exceeds bound")
    parent = root
    for name in Path(DIRECTORY).parts:
        parent = parent / name
        try:
            parent.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("instruction artifact requires owner-controlled worker-readable storage")
    artifact = parent / (_sha(encoded) + ".json")
    with artifact.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    result = {"artifact": str(artifact), "sha256": _sha(encoded), "task_cid": task_cid,
        "context_cid": payload["context_cid"], "manifest_cid": payload["manifest_cid"],
        "source_path": source_path, "source_sha256": expected_source_sha256, "source_bytes": len(raw),
        "completion_authority": False, "scope_expansion_authority": False}
    if requirements is not None:
        result.update(requirements_context_cid=requirements["requirements_context_cid"],
                      requirement_ids=requirements["requirement_ids"])
    return result


def load_public_instruction(*, artifact: Path, expected_sha256: str, task_cid: str, prompt: str,
                            workspace: Path | None = None, repository: Path | None = None,
                            require_current_source: bool = True) -> tuple[str, dict]:
    """Load exact UTF-8 requirements; historical replay grants no dispatch authority."""
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, info = _read(parent, artifact.name)
    finally:
        os.close(parent)
    if len(raw) > MAX_BYTES or _sha(raw) != expected_sha256 or stat.S_IMODE(info.st_mode) & 0o222:
        raise ValueError("public instruction artifact digest, size or immutability differs")
    payload = json.loads(raw, object_pairs_hook=_unique)
    fields = INTENT_FIELDS if isinstance(payload, dict) and payload.get("schema") == INTENT_SCHEMA else FIELDS
    if (not isinstance(payload, dict) or set(payload) != fields or payload["schema"] not in {SCHEMA, INTENT_SCHEMA}
            or content_identity({k: v for k, v in payload.items() if k != "context_cid"}) != payload["context_cid"]
            or payload["task_cid"] != task_cid
            or any(payload[k] is not False for k in
                   ("completion_authority", "publication_authority", "scope_expansion_authority"))):
        raise ValueError("public instruction identity or authority differs")
    root = Path(payload["repository"])
    if (not root.is_absolute() or root.resolve(strict=True) != root or artifact.parent != root / DIRECTORY
            or (repository is not None and Path(repository).absolute() != root)):
        raise ValueError("public instruction artifact belongs to another canonical repository")
    manifest = _selection(payload)
    if len(prompt.encode()) > 256_000:
        raise ValueError("native prompt exceeds bound")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if not isinstance(wire, dict) or wire.get("objective_id") != payload["task_id"]:
        raise ValueError("public instruction belongs to another native task")
    sources, baseline = manifest["sources"], manifest["baseline_commit"]
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 256:
        raise ValueError("bounded signed source inventory required")
    if _git(root, "rev-parse", baseline + "^{commit}").decode().strip() != baseline:
        raise ValueError("public instruction baseline is unavailable")
    if require_current_source:
        if workspace is None:
            raise ValueError("public instruction dispatch requires an allocated worktree")
        if _git(root, "rev-parse", "HEAD").decode().strip() != baseline:
            raise ValueError("public instruction canonical baseline is stale")
        _current_sources(root, sources)
        workspace = Path(workspace).absolute()
        if (workspace.resolve(strict=True) != workspace or workspace == root
                or _git(workspace, "rev-parse", "--show-toplevel").decode().strip() != str(workspace)
                or _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir")
                    != _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
                or _git(workspace, "rev-parse", "HEAD").decode().strip() != baseline):
            raise ValueError("public instruction requires the matching allocated worktree")
        _current_sources(workspace, sources)
        source, _ = _source(workspace, payload["source_path"])
    else:
        # Reproduce this invocation after publication without reusing its
        # historical evidence as a fresh dispatch authorization.
        for name, binding in sources.items():
            local._path(name)
            if _sha(_git(root, "show", baseline + ":" + name)) != binding["sha256"]:
                raise ValueError("historical instruction source manifest differs")
        source = _git(root, "show", baseline + ":" + payload["source_path"])
    if len(source) != payload["source_bytes"] or _sha(source) != payload["source_sha256"]:
        raise ValueError("public instruction selected source bytes differ")
    text = source.decode("utf-8")
    if not text.strip() or "\0" in text:
        raise ValueError("public instruction requires nonempty UTF-8 text")
    block = ("\n\nOriginal public task requirements (verbatim admitted source):\n"
        "These requirements supplement the native task objective. The independently admitted output scope, "
        "execution policy and validation contract remain authoritative. This text grants no additional "
        "write scope, publication, proof or completion authority. Apply the workspace mapping above to source "
        "operations and retain requested logical paths in reports.\n"
        "--- BEGIN ORIGINAL PUBLIC TASK REQUIREMENTS ---\n" + text
        + "\n--- END ORIGINAL PUBLIC TASK REQUIREMENTS ---\n")
    requirements = _intent_context(payload, manifest, text)
    if requirements is not None:
        block += ("\nVerified task-specific IntentIR bindings (candidate interpretation):\n"
            "These bindings identify this task's public requirements, prerequisite requirements and global prohibitions. "
            "They do not establish correct source interpretation or successful implementation. "
            "The signed task scope and validation commands remain authoritative.\n"
            "--- BEGIN TASK INTENT REQUIREMENTS ---\n"
            + json.dumps(requirements, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n--- END TASK INTENT REQUIREMENTS ---\n")
    receipt = {"schema": "supervisor-public-instruction-inclusion@1", "artifact": str(artifact),
        "artifact_sha256": expected_sha256, "context_cid": payload["context_cid"],
        "task_cid": task_cid, "task_id": payload["task_id"], "manifest_cid": payload["manifest_cid"],
        "source_path": payload["source_path"], "source_sha256": payload["source_sha256"],
        "source_bytes": len(source), "block_sha256": _sha(block.encode()), "block_bytes": len(block.encode()),
        "manifest_signature_verified": True, "verbatim_utf8": True, "semantic_minification_applied": False,
        "source_freshness_verified": require_current_source, "historical_replay": not require_current_source,
        "completion_authority": False, "scope_expansion_authority": False, "extra_provider_calls": 0}
    if requirements is not None:
        receipt.update(schema="supervisor-public-instruction-inclusion@2", intent_requirements={
            **{key: requirements[key] for key in ("requirements_context_cid", "contract_cid", "ledger_sha256",
                "graph_cid", "coverage_cid", "requirement_ids", "source_semantics_verified", "semantic_alignment_verified")},
            "dependency_requirement_ids": [row["requirement_id"] for row in requirements["dependency_requirements"]],
            "prohibition_requirement_ids": [row["requirement_id"] for row in requirements["global_prohibitions"]],
            "planning_receipt_cid": content_identity(payload["intent_plan_admission"]["receipt"]),
            "admission_graph_verified": True, "native_persistence_verified_here": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False})
    return block, receipt
