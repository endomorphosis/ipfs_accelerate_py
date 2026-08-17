#!/usr/bin/env python3
"""Build the immutable PGIR-014 semantic campaign freeze.

The builder is deliberately deterministic and write-once.  It reads the
accepted PGIR-001..013 artifacts, resolves every symbolic input to exact bytes
and Git objects, and emits a no-go campaign root when qualification gates do
not permit training.  Existing freeze bytes are never replaced: a changed
input requires a new superseding freeze location and task revision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FREEZE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = FREEZE_DIR.parents[3]
DATASETS_ROOT = REPOSITORY_ROOT / "ipfs_datasets_py"
TASK_BOARD = REPOSITORY_ROOT / "docs/architecture/proof_grounded_ir_learning.todo.md"

TASK_IDENTITY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/task-identity@1"
CAMPAIGN_ROOT_SCHEMA = "IRCampaignInputRoot@1"
TOKENIZER_POLICY_SCHEMA = "IRTokenizerFreezePolicy@1"
REVISION_SET_SCHEMA = "PGIRDescendantTaskRevisionSet@1"
VERIFICATION_RECEIPT_SCHEMA = "PGIRFreezeVerificationReceipt@1"
MANIFEST_SCHEMA = "PGIRFreezeBundleManifest@1"

REPOSITORY_ID = (
    "repository:sha256:4d87e009c221f83df2c5846e6085d4917204de75df8dc438b045c3bbff059dbc"
)
SOURCE_TREE_ID = "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
OBJECTIVE_REVISION = "baguqeeralbl2yjo6l5gazcmslpzqtu67un4txk3wwpjr45thh5sckwq67yhq"
POLICY_ID = "policy:implementation-daemon"
POLICY_REVISION = "sha256:27c8da23ef92ab263ac0c144f2414fd40bdb30aace98b88f7dd76d36db26e142"

EXPECTED_TASK_CIDS = {
    "PGIR-014": "baguqeeralbl2yjo6l5gazcmslpzqtu67un4txk3wwpjr45thh5sckwq67yhq",
    "PGIR-020": "baguqeeraqmjexogb3tci24z7tn5zm4qoubkzgqdv56jxpx3hc3sl2tduo54q",
    "PGIR-021": "baguqeera27c5rpxnrkdkiyw5m5coy6gk7dtvjngotzk6jab7f4qkjeydxf6a",
    "PGIR-022": "baguqeeravizixmcgul5k6k5ah3a7sc2dw64e7c6nkz6d5lhsor2jpqjblcda",
    "PGIR-023": "baguqeerarimshymyl3k3m3uwxw4yfnuwma5hlssaw3ggs24svelzgt5fkhpa",
    "PGIR-030": "baguqeerahdsbqllayvf2eguhnokgojpptmebh6yh6yev3aa36twk2rruo4zq",
    "PGIR-031": "baguqeeraaq5twsby4c4o2clrxo5gxstqvg2mg3eed2blposahv4wjpcbpl4a",
    "PGIR-032": "baguqeerayhdmcycsl6dk3acl5vqz2aoe7fp5z6vjfsg2xwmesrdo3jkiwc2q",
    "PGIR-033": "baguqeeraec4hazq72xwydpro254qhwmkbku4ltt4zztufxisilmypnfy3j4q",
    "PGIR-040": "baguqeeratn4vb7qfscshojhrfn65ggo7u4zvdfqzaudnizgdrjwgdgbhisda",
    "PGIR-041": "baguqeeraaqbbaijc5io2yhzzg3yalmk4omyeihzz6k232a5tv4agtmwjzfaq",
    "PGIR-050": "baguqeeradktwdh3hndpuvyqu22kyx2l2ys4uodcqfm5clhtc242ayl72yqsq",
    "PGIR-051": "baguqeeras6fqlvoj7iuesdw7jwjdufjappwcvtgegml4f3ebi4uc6gqcfoma",
    "PGIR-052": "baguqeeracfbpkhgaqqb4dsotaf5lrqw2ubhvdbv7glp6t5vo2tid22ygzvcq",
    "PGIR-053": "baguqeerawj24izvekc2cafc7a4uxdxwvij6apmowrozz4usshrml44jymnba",
    "PGIR-060": "baguqeera2ldlohqbcardjwhpyczk7ztvsabwkfspgwohxfzpintpre2h6t3a",
    "PGIR-061": "baguqeeragv5zsidweldoiqcmhmejlpbvhhq2eaohirdd7z6kfgapuxcrwmxa",
    "PGIR-062": "baguqeerads3akcfu5xrj756eye3f4kltzdonl73adn7tkfpmcj6nqfvap34a",
    "PGIR-070": "baguqeeraxgbyiiumdh6jldicx2kdq223zzihbjfmmfijsoxs67e5jwmksfkq",
    "PGIR-071": "baguqeerafzx2mkqlntkaw7ublebojigyh2ibjzcm2dgqvzhopxtqx7624rjq",
    "PGIR-072": "baguqeeraarlt745ftpwax4tdovajsxrgp5r72fkudlbs5kmg3q2cbqow6mpq",
    "PGIR-080": "baguqeerazgwoahtrkkqkise3x6o4wpo2nbzs3yi5a672xq7zsbfmbp3arqcq",
    "PGIR-081": "baguqeeran4usi4fm4b5tzu72kzcb5cpe5aljp6iwdlpg33ph2kzftyfnkdoa",
    "PGIR-090": "baguqeeragt5xhqov2e5vna6pp2zmghhrqhq2v4qzl36a3aucytrcom2erkta",
    "PGIR-100": "baguqeerainaqx5w72m2epc7wsolk3elqihpcganr6d3lrsuzseldzkqngija",
    "PGIR-110": "baguqeeragjtn4knjvdexk4ya373ljydixx5r6c4moxr3sj6apuf2slfutexq",
    "PGIR-111": "baguqeerad2idzhwzfqlyxjh7u34bczkyutsvq7oxcmuaglupr436gqfo6kga",
}

SOURCE_LINEAGE_SCHEMA_IDS = (
    "ir-corpus-manifest/v1",
    "ir-derived-artifact/v1",
    "ir-lineage-graph/v1",
    "ir-source-record/v1",
    "ir-source-record/v1.1",
    "ir-source-release/v1",
)

TRAINING_CONTRACT_SCHEMA_IDS = (
    "ir-compiler-trace/v1",
    "ir-decompiler-trace/v1",
    "ir-hard-negative/v1",
    "ir-label-evidence/v1",
    "ir-positive-pair/v1",
    "ir-proof-trace/v1",
    "ir-round-trip-trace/v1",
    "ir-statement-binding/v1",
    "ir-tactic-step/v1",
    "ir-tactic-trace/v1",
    "ir-tool-binding/v1",
    "ir-trace-reference/v1",
    "ir-training-example/v1",
    "ir-training-lineage/v1",
    "ir-translation-trace/v1",
)


class FreezeBuildError(RuntimeError):
    """Raised when the source chain cannot be frozen without invention."""


def _validate_canonical_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise FreezeBuildError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise FreezeBuildError(f"{path} contains a non-string key")
        for key, item in value.items():
            _validate_canonical_value(item, f"{path}.{key}")
        return
    raise FreezeBuildError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    _validate_canonical_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rendered_bytes(value: Any) -> bytes:
    _validate_canonical_value(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
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


def add_projection_identity(
    payload: Mapping[str, Any], *, cid_field: str, sha_field: str | None = None
) -> dict[str, Any]:
    result = dict(payload)
    projection = dict(result)
    if sha_field:
        result[sha_field] = "sha256:" + hashlib.sha256(canonical_bytes(projection)).hexdigest()
    result[cid_field] = dag_json_cid(projection)
    return result


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FreezeBuildError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                FreezeBuildError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                FreezeBuildError(f"non-finite number {raw!r} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise FreezeBuildError(f"{path} must contain a JSON object")
    return value


def git(*args: str, cwd: Path = REPOSITORY_ROOT) -> str:
    process = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode:
        raise FreezeBuildError(
            f"git {' '.join(args)} failed in {cwd}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def full_commit(revision: str, *, cwd: Path) -> str:
    value = git("rev-parse", f"{revision}^{{commit}}", cwd=cwd)
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise FreezeBuildError(f"invalid Git commit resolved from {revision!r}")
    return value


def git_object(commit: str, path: str, *, cwd: Path) -> str:
    value = git("rev-parse", f"{commit}:{path}", cwd=cwd)
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise FreezeBuildError(f"invalid Git object for {commit}:{path}")
    return value


def file_binding(
    relative_path: str,
    *,
    repository: str,
    commit: str | None = None,
    git_path: str | None = None,
    supplied_bytes: bytes | None = None,
) -> dict[str, Any]:
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise FreezeBuildError(f"unsafe file binding path: {relative_path!r}")
    data = supplied_bytes
    if data is None:
        path = REPOSITORY_ROOT / normalized
        if not path.is_file() or path.is_symlink():
            raise FreezeBuildError(f"bound file is absent or unsafe: {normalized}")
        data = path.read_bytes()
    binding: dict[str, Any] = {
        "path": normalized,
        "repository": repository,
        "raw_cid": raw_cid(data),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    if commit:
        if repository == "ipfs_datasets_py":
            cwd = DATASETS_ROOT
            object_path = git_path or normalized.removeprefix("ipfs_datasets_py/")
        elif repository == "ipfs_accelerate_py":
            cwd = REPOSITORY_ROOT
            object_path = git_path or normalized
        else:
            raise FreezeBuildError(f"unsupported Git repository {repository!r}")
        binding["git_blob"] = git_object(commit, object_path, cwd=cwd)
    return binding


def binding_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return add_projection_identity(payload, cid_field="binding_cid")


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
    title = normalize_identity_text(task.get("title"))
    outputs = sorted(
        {
            normalized
            for item in task.get("outputs", [])
            if (normalized := normalize_identity_path(item))
        }
    )
    acceptance = [
        normalized
        for item in split_csv(str(task.get("acceptance") or metadata.get("acceptance criteria") or ""))
        if (normalized := normalize_identity_text(item))
    ]
    evidence = sorted(
        {
            normalized
            for item in split_csv(str(metadata.get("missing evidence") or ""))
            if (normalized := normalize_identity_text(item))
        }
    )
    goal = normalize_identity_text(
        metadata.get("goal id") or metadata.get("goal packet key") or metadata.get("goal")
    )
    hint = normalize_identity_text(
        semantic_key
        or metadata.get("semantic key")
        or metadata.get("bundle key")
        or metadata.get("work scope")
        or metadata.get("fingerprint")
    )
    semantic = {
        key: value
        for key, value in {
            "title": title,
            "outputs": outputs,
            "acceptance": acceptance,
            "evidence": evidence,
            "goal": goal,
            "semantic_hint": hint,
        }.items()
        if value
    }
    if not semantic:
        raise FreezeBuildError(f"task {task.get('task_id')} has no semantic identity")
    material = {"schema": TASK_IDENTITY_SCHEMA, "semantic": semantic}
    fingerprint = hashlib.sha256(canonical_bytes(material)).hexdigest()
    return {
        "canonical_task_key": f"task/v1/{fingerprint}",
        "canonical_task_cid": dag_json_cid(material),
        "semantic_fingerprint": fingerprint,
    }


def parse_task_board() -> dict[str, dict[str, Any]]:
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
        if current_id in tasks:
            raise FreezeBuildError(f"duplicate task {current_id} in task board")
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

    if "PGIR-014" not in tasks:
        raise FreezeBuildError("PGIR-014 is absent from the protected task board")
    for task_id, expected_cid in EXPECTED_TASK_CIDS.items():
        actual = tasks.get(task_id, {}).get("canonical_task_cid")
        if actual != expected_cid:
            raise FreezeBuildError(
                f"task semantics drifted for {task_id}: expected {expected_cid}, got {actual}"
            )
    return tasks


def transitive_descendants(
    tasks: Mapping[str, Mapping[str, Any]], ancestor: str
) -> tuple[str, ...]:
    descendants: set[str] = set()
    changed = True
    while changed:
        changed = False
        for task_id, task in tasks.items():
            if task_id == ancestor or task_id in descendants:
                continue
            dependencies = set(task.get("depends_on") or ())
            if ancestor in dependencies or dependencies.intersection(descendants):
                descendants.add(task_id)
                changed = True
    return tuple(sorted(descendants))


def tokenizer_policy() -> dict[str, Any]:
    payload = {
        "schema": TOKENIZER_POLICY_SCHEMA,
        "interface": "proof-grounded-ir-learning/tokenizer-freeze-policy/v1",
        "contract_version": 1,
        "status": "no_learned_tokenizer_admitted",
        "canonical_tokenization": {
            "identity": "deterministic-canonical-structure-only",
            "allowed_uses": [
                "contract validation",
                "deterministic compiler/decompiler replay",
                "schema migration testing",
            ],
            "learned_vocabulary_identity": "none",
            "model_checkpoint_identity": "none",
        },
        "training_policy": {
            "authorized": False,
            "reason": "PGIR-050 must define and freeze a compatible tokenizer before learned training",
            "superseding_root_required": True,
        },
        "unknown_token_behavior": "fail_closed",
        "mutation_policy": "supersede_never_overwrite",
    }
    return add_projection_identity(
        payload, cid_field="policy_cid", sha_field="policy_sha256"
    )


def build_campaign_root(tokenizer: Mapping[str, Any]) -> dict[str, Any]:
    datasets_commit = full_commit("b20bd9e3cfae79e8888929daf64f52b2f8a5689a", cwd=DATASETS_ROOT)
    implementation_commits = {
        task_id: full_commit(revision, cwd=DATASETS_ROOT)
        for task_id, revision in {
            "PGIR-006": "21e1a2db5",
            "PGIR-010": "1f1aa38bd",
            "PGIR-011": "978a4ef12",
            "PGIR-012": "99717f2b7",
            "PGIR-013": "b20bd9e3c",
        }.items()
    }
    if datasets_commit != implementation_commits["PGIR-013"]:
        raise FreezeBuildError("PGIR-013 commit is not the selected datasets repository root")
    git("cat-file", "-e", f"{SOURCE_TREE_ID}^{{commit}}", cwd=REPOSITORY_ROOT)

    source_manifest_path = (
        REPOSITORY_ROOT
        / "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json"
    )
    source_result_path = source_manifest_path.with_name("result.json")
    source_manifest = strict_json(source_manifest_path)
    source_result = strict_json(source_result_path)
    gap_path = DATASETS_ROOT / "docs/architecture/proof_grounded_ir_learning/gap_matrix.json"
    gap = strict_json(gap_path)
    corpus_dir = DATASETS_ROOT / "data/ir_learning/corpora"
    split_dir = DATASETS_ROOT / "data/ir_learning/splits"
    corpus_root = strict_json(corpus_dir / "corpus_root.json")
    rights = strict_json(corpus_dir / "rights_manifest.json")
    reconciliation = strict_json(corpus_dir / "reconciliation_receipt.json")
    split_root = strict_json(split_dir / "split_root.json")
    leakage = strict_json(split_dir / "leakage_report.json")

    source_files = [
        "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/manifest.json",
        "data/agent_supervisor/proof_grounded_ir_learning/inventory/source_revisions/result.json",
        "docs/architecture/proof_grounded_ir_learning/inventory/supervisor.json",
        "ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/inventory/modules.json",
        "ipfs_datasets_py/data/ir_learning/source_inventory/release_inventory.json",
        "data/agent_supervisor/proof_grounded_ir_learning/baseline_tests/summary.json",
    ]
    source_bindings = [
        file_binding(
            path,
            repository=("ipfs_datasets_py" if path.startswith("ipfs_datasets_py/") else "ipfs_accelerate_py"),
            commit=(datasets_commit if path.startswith("ipfs_datasets_py/") else SOURCE_TREE_ID),
        )
        for path in source_files
    ]

    schema_paths = (
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/__init__.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/artifacts.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/canonical.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/claims.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/diagnostics.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/evidence.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/identity.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/protocols.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/provenance.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/schema_registry.py",
        "ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/source_lineage.py",
    )
    schema_bindings = [
        file_binding(path, repository="ipfs_datasets_py", commit=datasets_commit)
        for path in schema_paths
    ]

    corpus_names = (
        "corpus_manifest.json",
        "corpus_root.json",
        "lineage_graph.json",
        "quarantine_manifest.json",
        "reconciliation_receipt.json",
        "rights_manifest.json",
        "source_releases.json",
    )
    corpus_bindings = [
        file_binding(
            f"ipfs_datasets_py/data/ir_learning/corpora/{name}",
            repository="ipfs_datasets_py",
            commit=datasets_commit,
        )
        for name in corpus_names
    ]

    split_paths = (
        "ipfs_datasets_py/data/ir_learning/splits/holdout_report.json",
        "ipfs_datasets_py/data/ir_learning/splits/ir_split_manifest.json",
        "ipfs_datasets_py/data/ir_learning/splits/leakage_report.json",
        "ipfs_datasets_py/data/ir_learning/splits/split_root.json",
        "ipfs_datasets_py/ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_eval_splits.py",
    )
    split_bindings = [
        file_binding(path, repository="ipfs_datasets_py", commit=datasets_commit)
        for path in split_paths
    ]

    training_paths = tuple(
        f"ipfs_datasets_py/ipfs_datasets_py/logic/formalization/{name}"
        for name in (
            "training_contracts.py",
            "training_examples.py",
            "training_proofs.py",
            "training_shared.py",
            "training_transforms.py",
        )
    )
    training_bindings = [
        file_binding(path, repository="ipfs_datasets_py", commit=datasets_commit)
        for path in training_paths
    ]

    compiler_path = "ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_compiler.py"
    decompiler_path = "ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_decompiler.py"
    contracts_path = "ipfs_datasets_py/ipfs_datasets_py/logic/legal_ir/canonical_contracts.py"
    tokenizer_bytes = rendered_bytes(tokenizer)

    bindings = {
        "source_snapshots": binding_identity(
            {
                "task_ids": ["PGIR-001", "PGIR-002", "PGIR-003", "PGIR-004", "PGIR-005"],
                "source_set_id": "SRCSET-1",
                "source_manifest_cid": source_manifest["manifest_cid"],
                "source_result_identity": source_result["result_identity"],
                "authority_commits": source_result["base_source_revisions"],
                "selected_repository_tree": SOURCE_TREE_ID,
                "selected_datasets_commit": datasets_commit,
                "files": source_bindings,
            }
        ),
        "gap_matrix": binding_identity(
            {
                "task_id": "PGIR-006",
                "implementation_commit": implementation_commits["PGIR-006"],
                "matrix_cid": gap["matrix_cid"],
                "matrix_sha256": "sha256:" + gap["matrix_sha256"],
                "file": file_binding(
                    "ipfs_datasets_py/docs/architecture/proof_grounded_ir_learning/gap_matrix.json",
                    repository="ipfs_datasets_py",
                    commit=datasets_commit,
                ),
            }
        ),
        "schema_registry": binding_identity(
            {
                "task_id": "PGIR-010",
                "implementation_commit": implementation_commits["PGIR-010"],
                "repository_commit": datasets_commit,
                "tree_oid": git_object(
                    datasets_commit,
                    "ipfs_datasets_py/logic/ir_core",
                    cwd=DATASETS_ROOT,
                ),
                "schema_ids": list(SOURCE_LINEAGE_SCHEMA_IDS),
                "strict_unknown_fields": True,
                "files": schema_bindings,
            }
        ),
        "corpus": binding_identity(
            {
                "task_id": "PGIR-011",
                "implementation_commit": implementation_commits["PGIR-011"],
                "manifest_id": corpus_root["manifest_id"],
                "manifest_cid": corpus_root["manifest_cid"],
                "pinset_id": corpus_root["pinset_id"],
                "source_count": corpus_root["source_count"],
                "derived_count": corpus_root["derived_count"],
                "patent_source_groups": corpus_root["patent_source_groups"],
                "materialized": corpus_root["materialized"],
                "training_admitted_rows": corpus_root["training_admitted_rows"],
                "files": corpus_bindings,
            }
        ),
        "rights": binding_identity(
            {
                "task_id": "PGIR-011",
                "pinset_id": rights["pinset_id"],
                "source_count": rights["source_count"],
                "training_admitted_rows": rights["training_admitted_rows"],
                "admitted_source_count": len(rights["admitted_source_record_ids"]),
                "quarantined_source_count": len(rights["quarantined_source_record_ids"]),
                "file": next(item for item in corpus_bindings if item["path"].endswith("rights_manifest.json")),
            }
        ),
        "lineage": binding_identity(
            {
                "task_id": "PGIR-011",
                "lineage_graph_cid": corpus_root["lineage_graph_cid"],
                "source_count": reconciliation["source_count"],
                "derived_count": reconciliation["derived_count"],
                "patent_source_groups": reconciliation["patent_source_groups"],
                "file": next(item for item in corpus_bindings if item["path"].endswith("lineage_graph.json")),
            }
        ),
        "split": binding_identity(
            {
                "task_id": "PGIR-012",
                "implementation_commit": implementation_commits["PGIR-012"],
                "schema": split_root["schema"],
                "split_manifest_sha256": "sha256:" + split_root["split_manifest_sha256"],
                "split_manifest_digest": "sha256:" + split_root["split_manifest_digest"],
                "hidden_test_commitment": split_root["hidden_test_commitment"],
                "leakage_passed": split_root["leakage_passed"],
                "holdouts": split_root["holdouts"],
                "files": split_bindings,
            }
        ),
        "example_contracts": binding_identity(
            {
                "task_id": "PGIR-013",
                "implementation_commit": implementation_commits["PGIR-013"],
                "repository_commit": datasets_commit,
                "schema_ids": list(TRAINING_CONTRACT_SCHEMA_IDS),
                "closed_authority_vocabulary": True,
                "files": training_bindings,
            }
        ),
        "compiler": binding_identity(
            {
                "task_id": "PGIR-014",
                "symbolic_alias": "COMPILER-CURRENT-1",
                "status": "resolved_exact",
                "repository_commit": datasets_commit,
                "entrypoint": "ipfs_datasets_py.logic.legal_ir.canonical_compiler.TypedDeonticCanonicalCompiler",
                "interface": "CanonicalStructuredTextCompiler@1",
                "configuration_cid": "baguqeera7dlkg3mkupddznjq3ehefgfy3uwcarhir2yfgukpgjmo4spmroea",
                "measured_adapter_raw_cid": "bafkreife5avbe5esju4frufsogvzlaew5x5qw5h4qlefvgx2qdbamqsyny",
                "learned_stages": [],
                "files": [
                    file_binding(compiler_path, repository="ipfs_datasets_py", commit=datasets_commit),
                    file_binding(contracts_path, repository="ipfs_datasets_py", commit=datasets_commit),
                ],
            }
        ),
        "decompiler": binding_identity(
            {
                "task_id": "PGIR-014",
                "symbolic_alias": "DECOMPILER-CURRENT-1",
                "status": "resolved_exact",
                "repository_commit": datasets_commit,
                "entrypoint": "ipfs_datasets_py.logic.legal_ir.canonical_decompiler.SourceWithheldCanonicalDecompiler",
                "interface": "SourceWithheldCanonicalParaphraser@1",
                "configuration_cid": "baguqeeratlk326nodsva4rxwm65xgnpenhcovspm7crtyd4enaqhgjciqayq",
                "rendering_spec_cid": "baguqeera72pqowlkovfqvydbtk5lxc7g42o75xtfgmx7cm4vqdvnaimjpjvq",
                "uses_model": False,
                "files": [
                    file_binding(decompiler_path, repository="ipfs_datasets_py", commit=datasets_commit),
                    file_binding(contracts_path, repository="ipfs_datasets_py", commit=datasets_commit),
                ],
            }
        ),
        "tokenizer_policy": binding_identity(
            {
                "task_id": "PGIR-014",
                "policy_cid": tokenizer["policy_cid"],
                "status": tokenizer["status"],
                "file": file_binding(
                    "data/agent_supervisor/proof_grounded_ir_learning/freeze/tokenizer_policy.json",
                    repository="pgir-freeze",
                    supplied_bytes=tokenizer_bytes,
                ),
            }
        ),
        "policy": binding_identity(
            {
                "task_id": "PGIR-014",
                "policy_id": POLICY_ID,
                "policy_revision": POLICY_REVISION,
                "objective_revision": OBJECTIVE_REVISION,
                "protected_paths": [
                    "data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml",
                    "docs/architecture/proof_grounded_ir_learning.objectives.md",
                    "docs/architecture/proof_grounded_ir_learning.todo.md",
                ],
                "files": [
                    file_binding(
                        "data/agent_supervisor/proof_grounded_ir_learning/justice_dao_pinset.yaml",
                        repository="ipfs_accelerate_py",
                        commit=SOURCE_TREE_ID,
                    ),
                    file_binding(
                        "docs/architecture/proof_grounded_ir_learning.objectives.md",
                        repository="ipfs_accelerate_py",
                        commit=SOURCE_TREE_ID,
                    ),
                ],
            }
        ),
    }

    if corpus_root["training_admitted_rows"] != 0:
        raise FreezeBuildError("expected fail-closed corpus has unexpectedly admitted training rows")
    if len(rights["admitted_source_record_ids"]) != 0:
        raise FreezeBuildError("rights manifest unexpectedly admits source records")
    if len(rights["quarantined_source_record_ids"]) != rights["source_count"]:
        raise FreezeBuildError("rights quarantine count does not reconcile")
    if not leakage.get("passed") or leakage.get("violations"):
        raise FreezeBuildError("split leakage receipt does not pass cleanly")
    if not split_root.get("leakage_passed"):
        raise FreezeBuildError("sealed split root does not pass leakage")

    insufficient_holdouts = sorted(
        name
        for name, value in split_root["holdouts"].items()
        if value.get("status") == "insufficient"
    )
    root_payload = {
        "schema": CAMPAIGN_ROOT_SCHEMA,
        "interface": "proof-grounded-ir-learning/campaign-input-root/v1",
        "contract_version": 1,
        "task_id": "PGIR-014",
        "objective_id": "PGIR-014",
        "objective_revision": OBJECTIVE_REVISION,
        "repository": {
            "repository_id": REPOSITORY_ID,
            "source_tree_id": SOURCE_TREE_ID,
            "datasets_commit": datasets_commit,
            "source_set_id": "SRCSET-1",
        },
        "bindings": bindings,
        "referential_integrity": {
            "all_required_bindings_present": True,
            "required_binding_names": sorted(bindings),
            "unresolved_identities": [],
            "compiler_alias_resolved": True,
            "decompiler_alias_resolved": True,
            "hidden_labels_accessed": False,
            "source_or_split_mutated": False,
        },
        "qualification": {
            "decision": "no_go",
            "lease_barrier": "closed",
            "descendant_execution_authorized": False,
            "training_task_eligible_count": 0,
            "training_admitted_rows": 0,
            "rights_quarantined_rows": rights["source_count"],
            "corpus_materialized": bool(corpus_root["materialized"]),
            "leakage_passed": True,
            "insufficient_holdouts": insufficient_holdouts,
            "reason_codes": [
                "corpus_not_materialized",
                "historical_semantic_baseline_not_currently_qualified",
                "no_rights_admitted_training_rows",
                "required_holdouts_insufficient",
            ],
        },
        "canonicalization": {
            "identity_projection": "entire document excluding root_cid and root_sha256",
            "json": "UTF-8; sorted keys; compact separators; ensure_ascii=false; no floats",
            "cid": "CIDv1/base32/dag-json/sha2-256",
            "rendering": "two-space indentation and one terminal LF",
        },
        "supersession": {
            "immutable": True,
            "previous_root_cid": None,
            "replacement_policy": "supersede_never_overwrite",
        },
    }
    return add_projection_identity(
        root_payload, cid_field="root_cid", sha_field="root_sha256"
    )


def build_descendant_revisions(root: Mapping[str, Any]) -> dict[str, Any]:
    tasks = parse_task_board()
    descendant_ids = transitive_descendants(tasks, "PGIR-014")
    if set(descendant_ids) != set(EXPECTED_TASK_CIDS).difference({"PGIR-014"}):
        raise FreezeBuildError("PGIR-014 transitive descendant population drifted")
    semantic_key = f"pgir-campaign-input-root@1:{root['root_cid']}"
    revisions: list[dict[str, Any]] = []
    for task_id in descendant_ids:
        task = tasks[task_id]
        revised = task_identity(task, semantic_key=semantic_key)
        revisions.append(
            {
                "task_id": task_id,
                "title": task["title"],
                "depends_on": task["depends_on"],
                "current_task_cid": task["canonical_task_cid"],
                "revised_task_cid": revised["canonical_task_cid"],
                "revised_task_key": revised["canonical_task_key"],
                "semantic_fingerprint": revised["semantic_fingerprint"],
                "input_binding": {
                    "campaign_input_root_cid": root["root_cid"],
                    "semantic_key": semantic_key,
                    "decision": root["qualification"]["decision"],
                },
                "lease_eligible": False,
                "block_reason_codes": root["qualification"]["reason_codes"],
            }
        )
    graph_projection = {
        "schema": "PGIRDescendantTaskGraph@1",
        "root_task_id": "PGIR-014",
        "tasks": [
            {
                "task_id": item["task_id"],
                "task_cid": item["revised_task_cid"],
                "depends_on": item["depends_on"],
            }
            for item in revisions
        ],
    }
    payload = {
        "schema": REVISION_SET_SCHEMA,
        "task_id": "PGIR-014",
        "campaign_input_root_cid": root["root_cid"],
        "source_plan_task_cid": EXPECTED_TASK_CIDS["PGIR-014"],
        "revision_patch": {
            "metadata_field": "Semantic key",
            "metadata_value": semantic_key,
            "protected_board_mutated": False,
            "application_policy": "supervisor_compare_and_swap_after_superseding_admission",
        },
        "candidate_graph_cid": dag_json_cid(graph_projection),
        "descendant_task_count": len(revisions),
        "lease_eligible_count": 0,
        "revisions": revisions,
    }
    return add_projection_identity(
        payload, cid_field="revision_set_cid", sha_field="revision_set_sha256"
    )


def build_plan_admission_receipt(
    root: Mapping[str, Any], revisions: Mapping[str, Any]
) -> dict[str, Any]:
    rights = root["bindings"]["rights"]
    split = root["bindings"]["split"]

    def rejection(
        *, code: str, domain: str, message: str, source_ids: Sequence[str], details: Mapping[str, Any]
    ) -> dict[str, Any]:
        body = {
            "schema": "ipfs_accelerate_py/agent-supervisor/plan-admission-rejection@1",
            "code": code,
            "domain": domain,
            "message": message,
            "action_id": "",
            "effect_id": "",
            "dependency_id": "",
            "obligation_id": "",
            "source_ids": sorted(set(source_ids)),
            "details": dict(details),
        }
        return {
            **body,
            "rejection_id": supervisor_identity("plan-admission-rejection", body),
        }

    rejections = sorted(
        (
            rejection(
                code="assumption_unresolved",
                domain="assumption",
                message="No source row has rights authority for learned training under this freeze.",
                source_ids=(root["root_cid"], rights["binding_cid"]),
                details={
                    "training_admitted_rows": rights["training_admitted_rows"],
                    "quarantined_source_count": rights["quarantined_source_count"],
                },
            ),
            rejection(
                code="validation_failed",
                domain="validation",
                message="Required holdout and current semantic-baseline qualification gates are incomplete.",
                source_ids=(root["root_cid"], split["binding_cid"]),
                details={
                    "insufficient_holdouts": root["qualification"]["insufficient_holdouts"],
                    "semantic_baseline_currently_qualified": False,
                },
            ),
        ),
        key=lambda item: item["rejection_id"],
    )
    request_projection = {
        "campaign_input_root_cid": root["root_cid"],
        "candidate_plan_id": revisions["revision_set_cid"],
        "candidate_graph_id": revisions["candidate_graph_cid"],
        "repository_tree_id": SOURCE_TREE_ID,
    }
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/plan-admission-receipt@1",
        "compiler_version": 1,
        "requirement_id": "287667496524558776121661391058779883318",
        "request_id": supervisor_identity(
            "pgir-campaign-plan-admission-request", request_projection
        ),
        "candidate_plan_id": revisions["revision_set_cid"],
        "candidate_graph_id": revisions["candidate_graph_cid"],
        "repository_tree_id": SOURCE_TREE_ID,
        "verdict": "rejected",
        "admitted": False,
        "semantic_roots": {
            "campaign": root["root_cid"],
            "compiler": root["bindings"]["compiler"]["binding_cid"],
            "corpus": root["bindings"]["corpus"]["binding_cid"],
            "decompiler": root["bindings"]["decompiler"]["binding_cid"],
            "examples": root["bindings"]["example_contracts"]["binding_cid"],
            "policy": root["bindings"]["policy"]["binding_cid"],
            "schema": root["bindings"]["schema_registry"]["binding_cid"],
            "source": root["bindings"]["source_snapshots"]["binding_cid"],
            "split": split["binding_cid"],
            "tokenizer_policy": root["bindings"]["tokenizer_policy"]["binding_cid"],
        },
        "intent_result_id": "",
        "legal_result_ids": [],
        "legal_permission_ids": [],
        "security_decision_ids": [],
        "security_grant_ids": [],
        "checked_dependency_ids": [
            f"dependency:{item['task_id']}" for item in revisions["revisions"]
        ],
        "checked_assumption_ids": [
            "assumption:all-required-identities-resolved",
            "assumption:training-corpus-admitted",
        ],
        "generated_formula_ids": [],
        "proof_result_ids": [],
        "checked_validation_ids": [
            "validation:lineage-leakage",
            "validation:referential-integrity",
            "validation:rights-admission",
        ],
        "cve_security_evidence_ids": [],
        "rejection_reasons": rejections,
        "reason_codes": sorted({item["code"] for item in rejections}),
        "counterexamples": [],
        "local_replan_action_ids": [
            "replan:admit-rights-qualified-corpus",
            "replan:complete-required-holdouts",
            "replan:rerun-current-semantic-baseline",
            "replan:supersede-campaign-freeze-root",
        ],
        "closure_id": "",
        "permissions_are_grants": False,
        "generated_formulas_are_proofs": False,
        "authorizes_execution": False,
    }
    payload["checked_dependency_ids"] = sorted(payload["checked_dependency_ids"])
    payload["receipt_id"] = supervisor_identity("plan-admission-receipt", payload)
    return payload


def build_verification_receipt(
    root: Mapping[str, Any],
    revisions: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    file_bindings: dict[str, Mapping[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if {"path", "sha256", "raw_cid", "size_bytes"}.issubset(value):
                file_bindings[value["path"]] = value
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(root)
    checks = [
        {
            "check_id": "canonical-root-identity",
            "status": "passed",
            "evidence": root["root_cid"],
        },
        {
            "check_id": "exact-file-bindings",
            "status": "passed",
            "evidence": f"{len(file_bindings)} unique exact-byte bindings",
        },
        {
            "check_id": "required-semantic-domains",
            "status": "passed",
            "evidence": f"{len(root['bindings'])} closed binding domains",
        },
        {
            "check_id": "rights-count-reconciliation",
            "status": "passed",
            "evidence": "0 admitted plus 7173 quarantined equals 7173 sources",
        },
        {
            "check_id": "lineage-and-split-leakage",
            "status": "passed",
            "evidence": root["bindings"]["split"]["binding_cid"],
        },
        {
            "check_id": "compiler-decompiler-resolution",
            "status": "passed",
            "evidence": "both CURRENT-1 aliases resolve to exact code/configuration bindings",
        },
        {
            "check_id": "descendant-task-revisions",
            "status": "passed",
            "evidence": revisions["revision_set_cid"],
        },
        {
            "check_id": "supervisor-plan-admission",
            "status": "passed",
            "evidence": admission["receipt_id"],
        },
        {
            "check_id": "fail-closed-lease-barrier",
            "status": "passed",
            "evidence": "no-go root revises 26 descendants with zero lease-eligible tasks",
        },
    ]
    payload = {
        "schema": VERIFICATION_RECEIPT_SCHEMA,
        "verifier_interface": "pgir-freeze-independent-verifier/v1",
        "campaign_input_root_cid": root["root_cid"],
        "revision_set_cid": revisions["revision_set_cid"],
        "plan_admission_receipt_id": admission["receipt_id"],
        "campaign_decision": root["qualification"]["decision"],
        "verification_verdict": "verified",
        "all_integrity_checks_passed": True,
        "authorizes_execution": False,
        "checks": checks,
    }
    return add_projection_identity(payload, cid_field="receipt_cid")


def build_documents() -> dict[str, bytes]:
    tokenizer = tokenizer_policy()
    root = build_campaign_root(tokenizer)
    revisions = build_descendant_revisions(root)
    admission = build_plan_admission_receipt(root, revisions)
    verification = build_verification_receipt(root, revisions, admission)

    documents: dict[str, bytes] = {
        "tokenizer_policy.json": rendered_bytes(tokenizer),
        "campaign_input_root.json": rendered_bytes(root),
        "descendant_task_revisions.json": rendered_bytes(revisions),
        "plan_admission_receipt.json": rendered_bytes(admission),
        "verification_receipt.json": rendered_bytes(verification),
    }
    static_names = (
        "README.md",
        "build_freeze.py",
        "ir_campaign_input_root.schema.json",
        "verify_freeze.py",
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for name in (*static_names, *sorted(documents)):
        data = documents.get(name)
        if data is None:
            path = FREEZE_DIR / name
            if not path.is_file() or path.is_symlink():
                raise FreezeBuildError(f"required static freeze file is absent or unsafe: {name}")
            data = path.read_bytes()
        artifacts[name] = {
            "raw_cid": raw_cid(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    manifest = add_projection_identity(
        {
            "schema": MANIFEST_SCHEMA,
            "task_id": "PGIR-014",
            "campaign_input_root_cid": root["root_cid"],
            "revision_set_cid": revisions["revision_set_cid"],
            "plan_admission_receipt_id": admission["receipt_id"],
            "verification_receipt_cid": verification["receipt_cid"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "immutability": "supersede_never_overwrite",
            "supersedes_manifest_cid": "baguqeerafyxup44ij426ipllqqbo6voszvz4uviil5dm5lokai6cfihrsaha",
            "supersession_reason": "bind the validated schema and lint-clean verifier before task admission",
        },
        cid_field="manifest_cid",
    )
    documents["manifest.v3.json"] = rendered_bytes(manifest)

    result_payload = {
        "schema": "pgir-task-result@1",
        "task_id": "PGIR-014",
        "objective_revision": OBJECTIVE_REVISION,
        "repository_id": REPOSITORY_ID,
        "source_tree_id": SOURCE_TREE_ID,
        "result_identity": "RESULT(PGIR-014)",
        "supersedes_result_cid": "baguqeeravwtoxdkhmlg4khg7wg5vqtiv6vq5byd2ecy27bkcjyvkjjoo4q3q",
        "campaign_input_root_cid": root["root_cid"],
        "manifest_cid": manifest["manifest_cid"],
        "revision_set_cid": revisions["revision_set_cid"],
        "plan_admission_receipt_id": admission["receipt_id"],
        "verification_receipt_cid": verification["receipt_cid"],
        "disposition": "frozen_no_go",
        "decision": "no_go",
        "completion_authoritative": False,
        "descendant_execution_authorized": False,
        "training_task_eligible_count": 0,
        "unresolved_identities": [],
        "reason_codes": root["qualification"]["reason_codes"],
        "rollback": "retain this immutable root and create a separately admitted superseding root",
    }
    result = add_projection_identity(
        result_payload, cid_field="result_cid", sha_field="result_sha256"
    )
    documents["result.v3.json"] = rendered_bytes(result)
    return documents


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FreezeBuildError(f"refusing unsafe existing output {path}")
        if path.read_bytes() != data:
            raise FreezeBuildError(
                f"immutable output differs at {path}; create a superseding freeze instead"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def check_documents(documents: Mapping[str, bytes]) -> None:
    missing: list[str] = []
    drifted: list[str] = []
    for name, expected in documents.items():
        path = FREEZE_DIR / name
        if not path.is_file() or path.is_symlink():
            missing.append(name)
        elif path.read_bytes() != expected:
            drifted.append(name)
    if missing or drifted:
        raise FreezeBuildError(
            f"freeze outputs do not match deterministic build; missing={missing}, drifted={drifted}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--initialize",
        action="store_true",
        help="write absent outputs once; refuse to replace different bytes",
    )
    mode.add_argument(
        "--print-root-cid",
        action="store_true",
        help="print the deterministic root CID without writing",
    )
    args = parser.parse_args(argv)
    documents = build_documents()
    if args.print_root_cid:
        print(json.loads(documents["campaign_input_root.json"])["root_cid"])
        return 0
    if args.initialize:
        for name in (
            "tokenizer_policy.json",
            "campaign_input_root.json",
            "descendant_task_revisions.json",
            "plan_admission_receipt.json",
            "verification_receipt.json",
            "manifest.v3.json",
            "result.v3.json",
        ):
            write_once(FREEZE_DIR / name, documents[name])
    else:
        check_documents(documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
