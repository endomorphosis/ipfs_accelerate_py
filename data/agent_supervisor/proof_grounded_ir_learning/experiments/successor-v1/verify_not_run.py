#!/usr/bin/env python3
"""Independently verify the PGIR-206 successor typed ``not_run`` package.

This verifier is deliberately independent of ``build_not_run.py``.  It uses
only the Python standard library, reads repository state without mutating it,
and invokes only local, read-only Git plumbing.  It never probes hardware,
opens a network connection, imports the builder, trains, evaluates, proves,
or writes an artifact.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_DIR = Path(__file__).absolute().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[4]
FREEZE_DIR = PACKAGE_DIR.parents[1] / "freeze" / "successor-v1"
BOARD_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "architecture"
    / "proof_grounded_ir_learning"
    / "successor.todo.md"
)
PACKAGE_PREFIX = (
    "data/agent_supervisor/proof_grounded_ir_learning/experiments/successor-v1"
)

TASK_ID = "PGIR-206"
TASK_TITLE = "Re-run R1-R6 on the superseding freeze"
TASK_CID = "baguqeerafze3wxxiomo4rhuxaguuk35d4tj4lrp2jtuj6jcinxnokczuctca"
TASK_KEY = "task/v1/2e49bb5ee8731dc89e9701a9456fa3e4d3c5c5fa4ce89f24486ddae50b3414c4"
TASK_FINGERPRINT = "2e49bb5ee8731dc89e9701a9456fa3e4d3c5c5fa4ce89f24486ddae50b3414c4"
REVISED_TASK_CID = "baguqeera57xb6lh6ra2f3x26wzi44pcymf42iqwhxsigg6pw6lnwzrtjwmoq"
REVISED_TASK_KEY = "task/v1/efee1f2cfe88345ddf5eb651ce3c586179a442c7bc906379f6f2db6cc669b31d"
OBJECTIVE_ID = "PGIR-G110"

PGIR205_ROOT_CID = "baguqeerajvu2dvjjxe4l6dibujguiedwhdahseziqpz7xhhp724jlxhlxz4q"
PGIR205_ROOT_SHA256 = "sha256:4d69a1d529b938bf0d01a24d44107638c079132883f3fb9ceffeb895dcebbe79"
PGIR205_RESULT_CID = "baguqeerarcuvejfqyjsfbqdtel67wu2lhjec3oh27enlqbxsvgyfbytsgd2q"
PGIR205_RESULT_SHA256 = "sha256:88a95224b0c26450c07322fdfb534b3a482db8faf91ab806f2a9b050e27230f5"
PGIR205_MANIFEST_CID = "baguqeeraw4k4u62ddeg2cvpn6qjqn3nwd2mdzrczc373zgtt67hbkgkrlbaq"
PGIR205_REVISION_SET_CID = "baguqeeraukisqjimylkyns2gr7af5imn4pmkv45gnja52c5vug4shc7lj7sq"
PGIR205_PLAN_RECEIPT_CID = "baguqeeranh57nf23krpy5qwgrivpnp2z7o2a5gcdzulvm73ekbb6upk6rwda"
PGIR205_VERIFICATION_CID = "baguqeeraxjwl7huyq2d5py3itpji5epx5om5a65f53vathq3jcmkfvxrr5sa"
PGIR205_PORTABILITY_CID = "baguqeera2ua47f7yht3oijt5vg4nzjsgnd7d75w5tdfw52fg6hj6v4tbmqfa"

PGIR205_FOREST = (
    {
        "role": "implementation",
        "commit": "011b7fd2bf15b380089944d2487989220a343338",
        "tree": "947e573d8d5ad299c8348e5a5bd507439bed4427",
        "parents": ["20ef9e48d59b505b04e6236a9a31aaba287c36fd"],
        "subject": "PGIR-205: Issue a superseding campaign input freeze",
    },
    {
        "role": "merge",
        "commit": "b4cd376393002c5a7daccdd8cac5c744ac6bf8aa",
        "tree": "947e573d8d5ad299c8348e5a5bd507439bed4427",
        "parents": [
            "20ef9e48d59b505b04e6236a9a31aaba287c36fd",
            "011b7fd2bf15b380089944d2487989220a343338",
        ],
        "subject": (
            "Merge commit '011b7fd2bf15b380089944d2487989220a343338' "
            "into agent/pgir-successor-current-supervisor-20260825"
        ),
    },
    {
        "role": "completion",
        "commit": "f2e8d607de2396f15681ec2b6edb546b68c9952a",
        "tree": "24ddf9e4b9a02811c2505c6a47311add63f76d60",
        "parents": ["b4cd376393002c5a7daccdd8cac5c744ac6bf8aa"],
        "subject": "PGIR-205: mark todo completed",
    },
)

REASON_CODES = (
    "no_rights_admitted_training_rows",
    "corpus_not_materialized",
    "required_holdouts_insufficient",
    "tokenizer_not_admitted",
    "historical_semantic_baseline_not_currently_qualified",
    "integrated_evidence_does_not_authorize_execution",
    "portability_no_go",
)
FAILED_HOLDOUTS = (
    "compiler",
    "cross_reference",
    "domain",
    "exception",
    "length",
    "lineage",
    "notation",
    "premise",
    "proof_library",
    "publication",
    "rare_operator",
    "time",
    "type",
)
ARM_DEFINITIONS = (
    ("R1", "deterministic_compiler_decompiler", False, (0,), ()),
    ("R2", "token_cross_entropy_only", True, (32, 33, 34), ("token_cross_entropy",)),
    (
        "R3",
        "token_cross_entropy_plus_cosine",
        True,
        (32, 33, 34),
        ("token_cross_entropy", "normalized_cosine"),
    ),
    ("R4", "supervised_contrastive", True, (32, 33, 34), ("supervised_contrastive",)),
    (
        "R5",
        "full_multi_task",
        True,
        (32, 33, 34),
        (
            "token_cross_entropy",
            "normalized_cosine",
            "supervised_contrastive",
            "cycle",
            "structural",
            "relation",
            "semantic",
            "source_span",
            "calibration",
            "regularization",
        ),
    ),
    (
        "R6",
        "proof_grounded_curriculum",
        True,
        (32, 33, 34),
        ("full_multi_task", "nondifferentiable_proof_curriculum"),
    ),
)
METRIC_DEFINITIONS = (
    ("token_cross_entropy", "nats_per_canonical_token", "minimize"),
    ("latent_separation", "score", "maximize"),
    ("retrieval_recall", "rate", "maximize"),
    ("structural_equivalence", "rate", "maximize"),
    ("semantic_equivalence", "rate", "maximize"),
    ("proof_replay_rate", "rate", "maximize"),
    ("readability_score", "score", "maximize"),
    ("calibration_error", "error", "minimize"),
    ("ood_acceptance", "rate", "maximize"),
    ("latency", "milliseconds", "minimize"),
    ("resource_cost", "milli_resource_units", "minimize"),
    ("target_attainment", "boolean", "maximize"),
)

EXPECTED_FILES = (
    "README.md",
    "arms.json",
    "build_not_run.py",
    "campaign.json",
    "comparison.json",
    "heldouts.json",
    "manifest.json",
    "metrics.json",
    "receipts/admission.json",
    "receipts/checkpoint.json",
    "receipts/evaluation.json",
    "receipts/proof.json",
    "receipts/reducer_cas.json",
    "receipts/resource.json",
    "receipts/training.json",
    "result.json",
    "seeds.json",
    "verify_not_run.py",
)
MANIFEST_INPUTS = tuple(
    name for name in EXPECTED_FILES if name not in {"manifest.json", "result.json"}
)
JSON_FILES = tuple(name for name in EXPECTED_FILES if name.endswith(".json"))
SEALS = {
    "arms.json": ("arm_set_cid", "arm_set_sha256"),
    "campaign.json": ("campaign_cid", "campaign_sha256"),
    "comparison.json": ("comparison_cid", "comparison_sha256"),
    "heldouts.json": ("heldout_cid", "heldout_sha256"),
    "manifest.json": ("manifest_cid", "manifest_sha256"),
    "metrics.json": ("metrics_cid", "metrics_sha256"),
    "receipts/admission.json": ("receipt_cid", "receipt_sha256"),
    "receipts/checkpoint.json": ("receipt_cid", "receipt_sha256"),
    "receipts/evaluation.json": ("receipt_cid", "receipt_sha256"),
    "receipts/proof.json": ("receipt_cid", "receipt_sha256"),
    "receipts/reducer_cas.json": ("receipt_cid", "receipt_sha256"),
    "receipts/resource.json": ("receipt_cid", "receipt_sha256"),
    "receipts/training.json": ("receipt_cid", "receipt_sha256"),
    "result.json": ("result_cid", "result_sha256"),
    "seeds.json": ("seed_cid", "seed_sha256"),
}

FREEZE_MANIFEST_FILES = (
    "README.md",
    "build_campaign_freeze.py",
    "campaign_input_root.json",
    "descendant_task_revisions.json",
    "ir_campaign_input_root.schema.json",
    "pgir_211_baseline_verifier_run.json",
    "plan_admission_receipt.json",
    "portability_receipt.json",
    "verification_receipt.json",
    "verify_campaign_freeze.py",
)
FREEZE_CRITICAL_FILES = (
    "campaign_input_root.json",
    "descendant_task_revisions.json",
    "manifest.json",
    "plan_admission_receipt.json",
    "portability_receipt.json",
    "result.json",
    "verification_receipt.json",
)


class VerificationError(ValueError):
    """Raised when an artifact, identity, or zero-effect claim drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def validate_value(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise VerificationError(f"{path} contains a float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{path} has a non-string key")
        for key, item in value.items():
            validate_value(item, f"{path}.{key}")
        return
    raise VerificationError(f"{path} contains unsupported {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def render_json(value: Any) -> bytes:
    validate_value(value)
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def cid(codec: int, data: bytes) -> str:
    def unsigned_varint(value: int) -> bytes:
        encoded = bytearray()
        while value >= 0x80:
            encoded.append((value & 0x7F) | 0x80)
            value >>= 7
        encoded.append(value)
        return bytes(encoded)

    digest = hashlib.sha256(data).digest()
    encoded = b"\x01" + unsigned_varint(codec) + b"\x12\x20" + digest
    return "b" + base64.b32encode(encoded).decode("ascii").rstrip("=").lower()


def dag_json_cid(value: Any) -> str:
    return cid(0x0129, canonical_bytes(value))


def raw_cid(data: bytes) -> str:
    return cid(0x55, data)


def strict_json(path: Path, *, exact_render: bool = True) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise VerificationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_float=lambda raw: (_ for _ in ()).throw(
                VerificationError(f"float {raw!r} in {path}")
            ),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                VerificationError(f"non-finite value {raw!r} in {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot strictly load {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    validate_value(value, str(path))
    if exact_render:
        require(data == render_json(value), f"{path} is not exact canonical rendered JSON")
    return value


def projection(payload: Mapping[str, Any], cid_field: str, sha_field: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {cid_field, sha_field}
    }


def verify_sealed(
    name: str,
    payload: Mapping[str, Any],
    expected_projection: Mapping[str, Any],
) -> None:
    cid_field, sha_field = SEALS[name]
    require(
        set(payload) == set(expected_projection) | {cid_field, sha_field},
        f"{name} exact top-level schema drift",
    )
    actual_projection = projection(payload, cid_field, sha_field)
    require(actual_projection == dict(expected_projection), f"{name} canonical projection drift")
    require(payload[cid_field] == dag_json_cid(actual_projection), f"{name} CID does not replay")
    require(
        payload[sha_field] == sha256(canonical_bytes(actual_projection)),
        f"{name} projection SHA-256 does not replay",
    )


def expected_task_binding() -> dict[str, Any]:
    return {
        "current_task_cid": TASK_CID,
        "current_task_key": TASK_KEY,
        "objective_id": OBJECTIVE_ID,
        "parent_goal": OBJECTIVE_ID,
        "subgoal": "controlled-comparisons-v2",
        "task_id": TASK_ID,
        "title": TASK_TITLE,
    }


def expected_input_binding() -> dict[str, Any]:
    return {
        "campaign_input_root_cid": PGIR205_ROOT_CID,
        "campaign_input_root_sha256": PGIR205_ROOT_SHA256,
        "completion_authoritative": False,
        "descendant_execution_authorized": False,
        "manifest_cid": PGIR205_MANIFEST_CID,
        "pgir_205_forest": [dict(row) for row in PGIR205_FOREST],
        "plan_admission_receipt_cid": PGIR205_PLAN_RECEIPT_CID,
        "portability_receipt_cid": PGIR205_PORTABILITY_CID,
        "result_cid": PGIR205_RESULT_CID,
        "result_identity": "RESULT(PGIR-205)",
        "result_sha256": PGIR205_RESULT_SHA256,
        "revision_set_cid": PGIR205_REVISION_SET_CID,
        "verification_receipt_cid": PGIR205_VERIFICATION_CID,
    }


def expected_zero_effects() -> dict[str, Any]:
    return {
        "checkpoint_created": False,
        "evaluation_invoked": False,
        "gpu_probe_performed": False,
        "hidden_labels_opened": False,
        "network_accessed": False,
        "optimizer_steps": 0,
        "proof_invoked": False,
        "promotion_attempted": False,
        "reducer_cas_attempted": False,
        "resource_lease_acquired": False,
        "resource_lease_requested": False,
        "training_started": False,
        "weights_created": False,
    }


def expected_experiment_keys() -> list[dict[str, Any]]:
    return [
        {"arm_id": arm_id, "experiment_key": f"{arm_id}/seed-{seed}", "seed": seed}
        for arm_id, _kind, _learned, seeds, _losses in ARM_DEFINITIONS
        for seed in seeds
    ]


def verify_source_policy() -> None:
    require(sys.flags.no_site == 1, "verifier must run with raw Python -S")
    source_path = PACKAGE_DIR / "verify_not_run.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    forbidden_roots = {
        "aiohttp",
        "ftplib",
        "http",
        "requests",
        "socket",
        "ssl",
        "telnetlib",
        "urllib",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    require(
        not any(name.split(".", 1)[0] in forbidden_roots for name in imported),
        "verifier imports a network-capable module",
    )
    require(
        not any(name.endswith("build_not_run") for name in imported),
        "verifier imports build_not_run.py",
    )
    require(
        "build_not_run" not in sys.modules,
        "builder unexpectedly present in verifier module graph",
    )


def verify_population() -> None:
    package_stat = PACKAGE_DIR.lstat()
    require(stat.S_ISDIR(package_stat.st_mode), "package root is not a real directory")
    require(not PACKAGE_DIR.is_symlink(), "package root is a symlink")
    found_files: set[str] = set()
    found_dirs: set[str] = set()

    def walk(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(PACKAGE_DIR).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), f"package entry is a symlink: {relative}")
                if stat.S_ISDIR(mode):
                    found_dirs.add(relative)
                    walk(path)
                else:
                    require(stat.S_ISREG(mode), f"package entry is not a regular file: {relative}")
                    found_files.add(relative)

    walk(PACKAGE_DIR)
    require(found_dirs == {"receipts"}, f"unexpected package directories: {sorted(found_dirs)}")
    require(found_files == set(EXPECTED_FILES), "exact 18-file package population drift")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def verify_active_task_identity(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    text = BOARD_PATH.read_text(encoding="utf-8")
    heading = f"## {TASK_ID} "
    starts = [index for index, line in enumerate(text.splitlines()) if line.startswith(heading)]
    require(len(starts) == 1, "board must contain exactly one PGIR-206 heading")
    lines = text.splitlines()
    start = starts[0]
    title = lines[start][len(heading) :].strip()
    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        block.append(line)
    fields: dict[str, str] = {}
    for line in block:
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            normalized = key.strip().casefold()
            require(normalized not in fields, f"duplicate board field {key!r}")
            fields[normalized] = value.strip()
    require(title == TASK_TITLE, "PGIR-206 board title drift")
    require(fields.get("goal id") == OBJECTIVE_ID, "PGIR-206 goal ID drift")
    require(
        fields.get("outputs") == PACKAGE_PREFIX + "/",
        "PGIR-206 declared output drift",
    )
    acceptance = fields.get("acceptance", "")
    require(bool(acceptance), "PGIR-206 acceptance is absent")
    material = {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-identity@1",
        "semantic": {
            "acceptance": [normalize_text(acceptance)],
            "goal": normalize_text(OBJECTIVE_ID),
            "outputs": [PACKAGE_PREFIX],
            "title": normalize_text(title),
        },
    }
    fingerprint = hashlib.sha256(canonical_bytes(material)).hexdigest()
    require(fingerprint == TASK_FINGERPRINT, "active task semantic fingerprint drift")
    require("task/v1/" + fingerprint == TASK_KEY, "active task key drift")
    require(dag_json_cid(material) == TASK_CID, "active task CID drift")
    for name, payload in payloads.items():
        binding = payload.get("task_binding")
        if binding is not None:
            require(binding == expected_task_binding(), f"{name} does not bind the active task")


def verify_freeze_inputs() -> None:
    root = strict_json(FREEZE_DIR / "campaign_input_root.json")
    result = strict_json(FREEZE_DIR / "result.json")
    manifest = strict_json(FREEZE_DIR / "manifest.json")
    revisions = strict_json(FREEZE_DIR / "descendant_task_revisions.json")
    admission = strict_json(FREEZE_DIR / "plan_admission_receipt.json")
    verification = strict_json(FREEZE_DIR / "verification_receipt.json")
    portability = strict_json(FREEZE_DIR / "portability_receipt.json")

    for payload, cid_field, sha_field, label in (
        (root, "root_cid", "root_sha256", "PGIR-205 root"),
        (result, "result_cid", "result_sha256", "PGIR-205 result"),
        (revisions, "revision_set_cid", "revision_set_sha256", "PGIR-205 revisions"),
    ):
        projected = projection(payload, cid_field, sha_field)
        require(payload[cid_field] == dag_json_cid(projected), f"{label} CID does not replay")
        require(
            payload[sha_field] == sha256(canonical_bytes(projected)),
            f"{label} SHA does not replay",
        )
    for payload, cid_field, label in (
        (manifest, "manifest_cid", "PGIR-205 manifest"),
        (verification, "receipt_cid", "PGIR-205 verification"),
    ):
        projected = {key: value for key, value in payload.items() if key != cid_field}
        require(payload[cid_field] == dag_json_cid(projected), f"{label} CID does not replay")
    portability_projection = {
        key: value
        for key, value in portability.items()
        if key not in {"receipt_cid", "receipt_sha256"}
    }
    require(
        portability["receipt_cid"] == dag_json_cid(portability_projection),
        "PGIR-205 portability CID does not replay",
    )
    require(
        portability["receipt_sha256"] == sha256(canonical_bytes(portability_projection)),
        "PGIR-205 portability SHA does not replay",
    )
    admission_projection = {
        key: value for key, value in admission.items() if key != "receipt_id"
    }
    require(
        admission["receipt_id"]
        == dag_json_cid(
            {"namespace": "plan-admission-receipt", "value": admission_projection}
        ),
        "PGIR-205 plan-admission receipt identity does not replay",
    )

    require(root["root_cid"] == PGIR205_ROOT_CID, "PGIR-205 root CID changed")
    require(root["root_sha256"] == PGIR205_ROOT_SHA256, "PGIR-205 root SHA changed")
    require(root.get("qualification", {}).get("decision") == "no_go", "freeze is not no_go")
    require(
        root.get("qualification", {}).get("descendant_execution_authorized") is False,
        "freeze authorizes descendant execution",
    )
    require(
        root.get("qualification", {}).get("lease_barrier") == "closed",
        "freeze lease barrier is not closed",
    )
    require(result["result_cid"] == PGIR205_RESULT_CID, "PGIR-205 result CID changed")
    require(result["result_sha256"] == PGIR205_RESULT_SHA256, "PGIR-205 result SHA changed")
    require(result.get("result_identity") == "RESULT(PGIR-205)", "PGIR-205 result identity drift")
    require(result.get("decision") == "no_go", "PGIR-205 result decision drift")
    require(result.get("completion_authoritative") is False, "PGIR-205 result claims authority")
    require(
        result.get("training_task_eligible_count") == 0,
        "PGIR-205 result has an eligible training task",
    )
    require(
        result.get("reason_codes") == list(REASON_CODES),
        "PGIR-205 fail-closed reason codes drift",
    )
    require(
        result.get("descendant_execution_authorized") is False,
        "PGIR-205 result authorizes execution",
    )
    require(result.get("manifest_cid") == PGIR205_MANIFEST_CID, "result/manifest binding drift")
    require(manifest["manifest_cid"] == PGIR205_MANIFEST_CID, "PGIR-205 manifest CID changed")
    require(
        revisions["revision_set_cid"] == PGIR205_REVISION_SET_CID,
        "PGIR-205 revision set CID changed",
    )
    require(admission["receipt_id"] == PGIR205_PLAN_RECEIPT_CID, "PGIR-205 admission CID changed")
    require(admission.get("admitted") is False, "PGIR-205 admission unexpectedly passed")
    require(admission.get("authorizes_execution") is False, "PGIR-205 admission grants authority")
    require(admission.get("verdict") == "rejected", "PGIR-205 admission verdict drift")
    require(
        verification["receipt_cid"] == PGIR205_VERIFICATION_CID,
        "PGIR-205 verification CID changed",
    )
    require(
        verification.get("authorizes_execution") is False,
        "PGIR-205 verification grants authority",
    )
    require(
        portability["receipt_cid"] == PGIR205_PORTABILITY_CID,
        "PGIR-205 portability CID changed",
    )
    require(portability.get("status") == "portability_no_go", "portability status drift")
    require(
        portability.get("pgir_205_execution_authorized") is False,
        "portability grants PGIR-205 execution authority",
    )

    require(manifest.get("artifact_count") == 10, "PGIR-205 manifest artifact count drift")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict), "PGIR-205 manifest artifacts malformed")
    require(set(artifacts) == set(FREEZE_MANIFEST_FILES), "PGIR-205 manifest population drift")
    for name in FREEZE_MANIFEST_FILES:
        entry = artifacts[name]
        require(
            isinstance(entry, dict) and set(entry) == {"raw_cid", "sha256", "size_bytes"},
            f"PGIR-205 manifest schema drift for {name}",
        )
        path = FREEZE_DIR / name
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) and not path.is_symlink(), f"PGIR-205 input is not regular: {name}")
        data = path.read_bytes()
        require(entry["raw_cid"] == raw_cid(data), f"PGIR-205 raw CID drift for {name}")
        require(entry["sha256"] == sha256(data), f"PGIR-205 raw SHA drift for {name}")
        require(entry["size_bytes"] == len(data), f"PGIR-205 byte length drift for {name}")

    revision_rows = [row for row in revisions.get("revisions", []) if row.get("task_id") == TASK_ID]
    require(len(revision_rows) == 1, "PGIR-205 revision set lacks unique PGIR-206 row")
    revision = revision_rows[0]
    require(revision.get("current_task_cid") == TASK_CID, "active task CID/revision mismatch")
    require(revision.get("current_task_key") == TASK_KEY, "active task key/revision mismatch")
    require(revision.get("revised_task_cid") == REVISED_TASK_CID, "inert revised task CID drift")
    require(revision.get("revised_task_key") == REVISED_TASK_KEY, "inert revised task key drift")
    require(revision.get("lease_eligible") is False, "inert revision is lease eligible")
    require(
        revision.get("input_binding")
        == {
            "campaign_input_root_cid": PGIR205_ROOT_CID,
            "decision": "no_go",
            "semantic_key": f"pgir-campaign-input-root@2:{PGIR205_ROOT_CID}",
        },
        "inert task revision input binding drift",
    )
    require(
        revisions.get("revision_patch", {}).get("protected_board_mutated") is False,
        "inert revision claims protected-board application",
    )
    require(TASK_CID != REVISED_TASK_CID and TASK_KEY != REVISED_TASK_KEY, "active/inert identity collapsed")


def verify_artifacts(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    task = expected_task_binding()
    inputs = expected_input_binding()
    effects = expected_zero_effects()
    keys = expected_experiment_keys()
    require(len(keys) == 16, "experiment key population is not 16")

    heldouts = payloads["heldouts.json"]
    verify_sealed(
        "heldouts.json",
        heldouts,
        {
            "campaign_input_root_cid": PGIR205_ROOT_CID,
            "failed_holdout_count": 13,
            "hidden_labels_opened": False,
            "hidden_test_commitment": "sha256:6a801b51b980e666b4010da9a679ad8e11a233078f988165565481b98d4b9ded",
            "holdouts": [
                {"count": 0, "holdout_id": item, "status": "permanent_no_go"}
                for item in FAILED_HOLDOUTS
            ],
            "identical_across_arms": True,
            "leakage_passed": True,
            "result_identity": "RESULT(PGIR-202)",
            "schema": "PGIRExperimentHeldouts@2",
            "split_binding_cid": "baguqeera456gtyfkktybzrjbujmp2mqy6i2bins372xgdsgvaahdcvpxy7aq",
            "status": "not_run",
            "task_binding": task,
        },
    )

    seeds = payloads["seeds.json"]
    verify_sealed(
        "seeds.json",
        seeds,
        {
            "best_test_selection": False,
            "deterministic_arm_seeds": [0],
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "hidden_test_tuning": False,
            "learned_arm_seeds": [32, 33, 34],
            "schema": "PGIRExperimentSeedPolicy@2",
            "status": "not_run",
            "task_binding": task,
        },
    )

    expected_arms = [
        {
            "arm_id": arm_id,
            "disposition": "not_run",
            "execution_authorized": False,
            "kind": kind,
            "learned": learned,
            "loss_components": list(losses),
            "proof_in_gradient_path": False,
            "reason_code": "admission_closed",
            "seeds": list(arm_seeds),
            "training_started": False,
        }
        for arm_id, kind, learned, arm_seeds, losses in ARM_DEFINITIONS
    ]
    expected_dispositions = [
        {
            "arm_id": key["arm_id"],
            "disposition": "not_run",
            "execution_authorized": False,
            "experiment_key": key["experiment_key"],
            "reason_code": "admission_closed",
            "seed": key["seed"],
        }
        for key in keys
    ]
    arms = payloads["arms.json"]
    verify_sealed(
        "arms.json",
        arms,
        {
            "arm_count": 6,
            "arms": expected_arms,
            "campaign_input_root_cid": PGIR205_ROOT_CID,
            "disposition_count": 16,
            "dispositions": expected_dispositions,
            "experiment_key_count": 16,
            "schema": "PGIRExperimentArmSet@2",
            "status": "not_run",
            "task_binding": task,
        },
    )

    campaign = payloads["campaign.json"]
    verify_sealed(
        "campaign.json",
        campaign,
        {
            "arm_set_cid": arms["arm_set_cid"],
            "authorizes_execution": False,
            "decision": "no_go",
            "disposition": "typed_not_run",
            "execution_status": "not_run",
            "experiment_key_count": 16,
            "heldout_cid": heldouts["heldout_cid"],
            "input_binding": inputs,
            "lease_eligible": False,
            "observed_effects": effects,
            "reason_codes": list(REASON_CODES),
            "schema": "PGIRControlledCampaign@2",
            "seed_cid": seeds["seed_cid"],
            "task_binding": task,
            "training_admitted_rows": 0,
        },
    )

    definitions = [
        {"direction": direction, "metric_id": metric_id, "unit": unit}
        for metric_id, unit, direction in METRIC_DEFINITIONS
    ]
    cells = [
        {
            "arm_id": key["arm_id"],
            "confidence_interval": None,
            "denominator": 0,
            "experiment_key": key["experiment_key"],
            "metric_id": metric_id,
            "missing_as_zero": False,
            "numerator": None,
            "reason_code": "admission_closed",
            "sample_count": 0,
            "seed": key["seed"],
            "status": "not_run",
            "target_attained": None,
            "unit": unit,
            "value": None,
        }
        for key in keys
        for metric_id, unit, _direction in METRIC_DEFINITIONS
    ]
    require(len(cells) == 192, "metric cell population is not 192")
    metrics = payloads["metrics.json"]
    verify_sealed(
        "metrics.json",
        metrics,
        {
            "campaign_cid": campaign["campaign_cid"],
            "cell_count": 192,
            "cells": cells,
            "metric_count": 12,
            "metric_definitions": definitions,
            "schema": "PGIRExperimentMetricMatrix@2",
            "status": "not_run",
            "task_binding": task,
        },
    )

    arm_ids = [item[0] for item in ARM_DEFINITIONS]
    pairs = [
        {
            "effect": None,
            "left_arm_id": left,
            "paired_sample_count": 0,
            "pair_id": f"{left}__{right}",
            "reason_code": "admission_closed",
            "right_arm_id": right,
            "status": "not_run",
            "uncertainty": None,
            "winner": None,
        }
        for left_index, left in enumerate(arm_ids)
        for right in arm_ids[left_index + 1 :]
    ]
    require(len(pairs) == 15, "unordered arm-pair population is not 15")
    comparison = payloads["comparison.json"]
    verify_sealed(
        "comparison.json",
        comparison,
        {
            "campaign_cid": campaign["campaign_cid"],
            "metric_ids": [item[0] for item in METRIC_DEFINITIONS],
            "metrics_cid": metrics["metrics_cid"],
            "no_winner": True,
            "pair_count": 15,
            "pairs": pairs,
            "schema": "PGIRExperimentComparison@2",
            "status": "not_run",
            "task_binding": task,
        },
    )

    admission = payloads["receipts/admission.json"]
    verify_sealed(
        "receipts/admission.json",
        admission,
        {
            "admitted": False,
            "authorizes_execution": False,
            "campaign_cid": campaign["campaign_cid"],
            "checked_gates": [
                {"gate_id": gate, "passed": False}
                for gate in (
                    "rights",
                    "corpus",
                    "holdouts",
                    "tokenizer",
                    "current_baseline",
                    "integrated_evidence",
                    "portability",
                )
            ],
            "decision": "rejected",
            "execution_status": "not_run",
            "input_plan_admission_receipt_cid": PGIR205_PLAN_RECEIPT_CID,
            "reason_codes": list(REASON_CODES),
            "schema": "PGIRExperimentAdmissionReceipt@2",
            "task_binding": task,
        },
    )

    training = payloads["receipts/training.json"]
    verify_sealed(
        "receipts/training.json",
        training,
        {
            "admission_receipt_cid": admission["receipt_cid"],
            "batch_count": 0,
            "campaign_cid": campaign["campaign_cid"],
            "data_rows_read": 0,
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "gpu_probe_performed": False,
            "optimizer_steps": 0,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentTrainingReceipt@2",
            "status": "not_run",
            "task_binding": task,
            "training_started": False,
        },
    )

    checkpoint = payloads["receipts/checkpoint.json"]
    verify_sealed(
        "receipts/checkpoint.json",
        checkpoint,
        {
            "campaign_cid": campaign["campaign_cid"],
            "checkpoint_count": 0,
            "checkpoint_paths": [],
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentCheckpointReceipt@2",
            "shared_checkpoint_write": False,
            "status": "not_run",
            "task_binding": task,
            "training_receipt_cid": training["receipt_cid"],
            "weight_artifacts": [],
            "weights_created": False,
        },
    )

    proof = payloads["receipts/proof.json"]
    verify_sealed(
        "receipts/proof.json",
        proof,
        {
            "attempt_count": 0,
            "authority_granted": False,
            "campaign_cid": campaign["campaign_cid"],
            "checked_proof_count": 0,
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "hidden_labels_opened": False,
            "kernel_check_count": 0,
            "nondifferentiable": True,
            "proof_invoked": False,
            "proof_results": [],
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentProofReceipt@2",
            "status": "not_run",
            "task_binding": task,
            "timeout_as_falsehood": False,
            "training_receipt_cid": training["receipt_cid"],
        },
    )

    evaluation = payloads["receipts/evaluation.json"]
    verify_sealed(
        "receipts/evaluation.json",
        evaluation,
        {
            "best_test_selection": False,
            "campaign_cid": campaign["campaign_cid"],
            "evaluation_invoked": False,
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "hidden_labels_opened": False,
            "hidden_test_access": False,
            "hidden_test_selection": False,
            "hidden_test_tuning": False,
            "metric_cell_count": 192,
            "measured_cell_count": 0,
            "metrics_cid": metrics["metrics_cid"],
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentEvaluationReceipt@2",
            "status": "not_run",
            "target_attainment_claim_count": 0,
            "task_binding": task,
        },
    )

    resource = payloads["receipts/resource.json"]
    verify_sealed(
        "receipts/resource.json",
        resource,
        {
            "bounded_exhaustion": {
                "kind": "admission_closed",
                "reason_codes": list(REASON_CODES),
            },
            "campaign_cid": campaign["campaign_cid"],
            "experiment_key_count": 16,
            "experiment_keys": keys,
            "gpu_probe_performed": False,
            "gpu_seconds_used": 0,
            "lease_acquired": False,
            "lease_id": None,
            "lease_requested": False,
            "measured_cost": None,
            "network_accessed": False,
            "proof_seconds_used": 0,
            "provider_call_count": 0,
            "reason_code": "admission_closed",
            "resource_class": "gpu-large",
            "schema": "PGIRExperimentResourceReceipt@2",
            "status": "not_run",
            "task_binding": task,
            "token_count": 0,
        },
    )

    reducer = payloads["receipts/reducer_cas.json"]
    verify_sealed(
        "receipts/reducer_cas.json",
        reducer,
        {
            "campaign_cid": campaign["campaign_cid"],
            "candidate_pointer": None,
            "compare_and_swap_attempted": False,
            "expected_pointer": None,
            "observed_pointer_after": None,
            "observed_pointer_before": None,
            "pointer_unchanged": True,
            "reason_code": "admission_closed",
            "schema": "PGIRExperimentReducerCASReceipt@2",
            "status": "not_run",
            "task_binding": task,
            "winner": None,
        },
    )

    receipt_names = tuple(name for name in JSON_FILES if name.startswith("receipts/"))
    require(len(receipt_names) == 7, "zero-effect receipt population is not seven")
    require(
        len({payloads[name]["receipt_cid"] for name in receipt_names}) == 7,
        "zero-effect receipt identities are not unique",
    )

    object_fields = {
        "arms.json": ("arm_set_cid", "arm_set_sha256"),
        "campaign.json": ("campaign_cid", "campaign_sha256"),
        "comparison.json": ("comparison_cid", "comparison_sha256"),
        "heldouts.json": ("heldout_cid", "heldout_sha256"),
        "metrics.json": ("metrics_cid", "metrics_sha256"),
        "seeds.json": ("seed_cid", "seed_sha256"),
    }
    manifest_entries: dict[str, Any] = {}
    for name in MANIFEST_INPUTS:
        path = PACKAGE_DIR / name
        data = path.read_bytes()
        if name in payloads:
            cid_field, sha_field = object_fields.get(name, ("receipt_cid", "receipt_sha256"))
            object_cid = payloads[name][cid_field]
            object_sha = payloads[name][sha_field]
        else:
            object_cid = None
            object_sha = None
        manifest_entries[name] = {
            "object_cid": object_cid,
            "object_sha256": object_sha,
            "raw_cid": raw_cid(data),
            "sha256": sha256(data),
            "size_bytes": len(data),
        }
    manifest = payloads["manifest.json"]
    verify_sealed(
        "manifest.json",
        manifest,
        {
            "artifact_count": 16,
            "artifacts": manifest_entries,
            "campaign_cid": campaign["campaign_cid"],
            "decision": "no_go",
            "execution_status": "not_run",
            "immutability": "supersede_never_overwrite",
            "json_artifact_count": 13,
            "schema": "PGIRExperimentNotRunManifest@2",
            "task_binding": task,
        },
    )

    receipt_cids = {
        name.removeprefix("receipts/").removesuffix(".json"): payloads[name]["receipt_cid"]
        for name in payloads
        if name.startswith("receipts/")
    }
    result = payloads["result.json"]
    expected_result = {
        "arm_count": 6,
        "arm_set_cid": arms["arm_set_cid"],
        "campaign_cid": campaign["campaign_cid"],
        "checkpoint_count": 0,
        "comparison_cid": comparison["comparison_cid"],
        "completion_authoritative": False,
        "decision": "no_go",
        "descendant_execution_authorized": False,
        "disposition": "typed_not_run",
        "execution_authorized": False,
        "execution_status": "not_run",
        "experiment_key_count": 16,
        "heldout_cid": heldouts["heldout_cid"],
        "input_binding": inputs,
        "manifest_cid": manifest["manifest_cid"],
        "measured_cell_count": 0,
        "metric_cell_count": 192,
        "metric_count": 12,
        "metrics_cid": metrics["metrics_cid"],
        "observed_effects": effects,
        "pair_count": 15,
        "reason_codes": list(REASON_CODES),
        "receipt_cids": receipt_cids,
        "result_identity": "RESULT(PGIR-206)",
        "schema": "pgir-task-result@1",
        "seed_cid": seeds["seed_cid"],
        "task_binding": task,
        "training_admitted_rows": 0,
    }
    verify_sealed("result.json", result, expected_result)


def git_bytes(*args: str, allow_one: bool = False) -> bytes:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(REPOSITORY_ROOT), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    allowed = {0, 1} if allow_one else {0}
    require(
        completed.returncode in allowed,
        f"read-only git {' '.join(args)} failed: {completed.stderr.decode('utf-8', 'replace').strip()}",
    )
    return completed.stdout


def git_text(*args: str, allow_one: bool = False) -> str:
    return git_bytes(*args, allow_one=allow_one).decode("utf-8").strip()


def verify_git_forest() -> None:
    for row in PGIR205_FOREST:
        commit = row["commit"]
        require(git_text("rev-parse", f"{commit}^{{commit}}") == commit, f"missing {row['role']} commit")
        require(git_text("show", "-s", "--format=%T", commit) == row["tree"], f"{row['role']} tree drift")
        require(
            git_text("show", "-s", "--format=%P", commit).split() == row["parents"],
            f"{row['role']} parents drift",
        )
        require(git_text("show", "-s", "--format=%s", commit) == row["subject"], f"{row['role']} subject drift")
    ancestry = (
        (PGIR205_FOREST[0]["parents"][0], PGIR205_FOREST[0]["commit"]),
        (PGIR205_FOREST[0]["commit"], PGIR205_FOREST[1]["commit"]),
        (PGIR205_FOREST[1]["commit"], PGIR205_FOREST[2]["commit"]),
        (PGIR205_FOREST[2]["commit"], "HEAD"),
    )
    for ancestor, descendant in ancestry:
        result = git_bytes("merge-base", "--is-ancestor", ancestor, descendant, allow_one=True)
        del result
        # ``git_bytes`` accepts rc=1 for a false ancestry query, so replay it as
        # the exact merge base and require the ancestor itself.
        require(
            git_text("merge-base", ancestor, descendant) == ancestor,
            f"Git ancestry failed for {ancestor} -> {descendant}",
        )

    implementation = PGIR205_FOREST[0]["commit"]
    for name in FREEZE_CRITICAL_FILES:
        relative = f"data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/{name}"
        expected_blob = git_text("rev-parse", f"{implementation}:{relative}")
        actual_blob = git_text("hash-object", "--", relative)
        require(actual_blob == expected_blob, f"current PGIR-205 bytes drifted for {name}")


def verify_git_index() -> None:
    raw = git_bytes("ls-files", "--stage", "-z", "--", PACKAGE_PREFIX)
    entries: dict[str, list[tuple[str, str, int]]] = {}
    pattern = re.compile(rb"^(\d+) ([0-9a-f]+) ([0-3])\t(.*)$")
    for record in raw.split(b"\0"):
        if not record:
            continue
        match = pattern.match(record)
        require(match is not None, "malformed Git index entry")
        assert match is not None
        mode = match.group(1).decode("ascii")
        blob = match.group(2).decode("ascii")
        stage = int(match.group(3))
        path = match.group(4).decode("utf-8")
        entries.setdefault(path, []).append((mode, blob, stage))
    expected_paths = {f"{PACKAGE_PREFIX}/{name}" for name in EXPECTED_FILES}
    require(set(entries) == expected_paths, "Git index package population is not exactly 18 files")
    for path in sorted(expected_paths):
        rows = entries[path]
        require(len(rows) == 1, f"multiple Git index records for {path}")
        mode, blob, stage = rows[0]
        require(stage == 0, f"non-stage-0 Git index entry for {path}")
        require(mode == "100644", f"non-regular Git blob mode for {path}: {mode}")
        file_path = REPOSITORY_ROOT / path
        filesystem_mode = file_path.lstat().st_mode
        require(stat.S_ISREG(filesystem_mode) and not file_path.is_symlink(), f"non-regular worktree file: {path}")
        require(git_text("hash-object", "--", path) == blob, f"index/worktree blob mismatch for {path}")


def verify() -> dict[str, Any]:
    verify_source_policy()
    verify_population()
    verify_freeze_inputs()
    payloads = {name: strict_json(PACKAGE_DIR / name) for name in JSON_FILES}
    verify_active_task_identity(payloads)
    verify_artifacts(payloads)
    verify_git_forest()
    verify_git_index()
    result = payloads["result.json"]
    return {
        "arm_count": 6,
        "decision": "no_go",
        "execution_authorized": False,
        "execution_status": "not_run",
        "experiment_key_count": 16,
        "file_count": 18,
        "metric_cell_count": 192,
        "pair_count": 15,
        "receipt_count": 7,
        "result_cid": result["result_cid"],
        "task_id": TASK_ID,
        "verified": True,
    }


def main() -> int:
    try:
        summary = verify()
    except (VerificationError, OSError, UnicodeError, ValueError) as exc:
        print(f"PGIR-206 verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
