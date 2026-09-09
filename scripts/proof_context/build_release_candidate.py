#!/usr/bin/env python3
"""Build and verify the immutable PCCE v0.1-rc1 evidence bundle.

The rc1 evidence cut is deliberately frozen at an outer Git commit.  The
builder reads admitted inputs from that commit rather than from the mutable
working tree, emits deterministic JSON, and builds twice before publishing.
Missing or negative evidence is preserved as a release ``NO-GO``; it is never
silently repaired or upgraded by this assembly task.

``--check`` is read-only.  It reconstructs rc1 from the frozen evidence cut,
compares every byte, validates the manifest bindings, and scans the bounded
bundle for credential material or hidden benchmark bodies.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
ACCELERATOR_ROOT = SCRIPT_PATH.parents[2]
OUTER_ROOT = SCRIPT_PATH.parents[4]

TASK_ID = "PCCE-081"
BOARD_NAMESPACE = "proof-carrying-context-engine-v0.1"
RELEASE_CANDIDATE = "v0.1-rc1"
RELEASE_RELATIVE_PATH = Path("artifacts/proof_carrying_context_engine/release/v0.1-rc1")
SOURCE_SNAPSHOT_COMMIT = "c5bdde1482a9afab7c5827a52bdf4f7b1d63f090"
SOURCE_SNAPSHOT_TREE = "bb7849ac3bc92ad80e0bc7dff9d3b421d08ce872"
TASK_BOARD_PATH = "artifacts/proof_carrying_context_engine/control/task_board.json"
ARTIFACT_ROOT = "artifacts/proof_carrying_context_engine"
RECEIPT_ROOT = f"{ARTIFACT_ROOT}/receipts"
SCHEMA_PREFIX = "lift_coding.proof-carrying-context-engine"

SOURCE_GITLINK_TREES = {
    "Mcp-Plus-Plus": "3a57e33053d6007cb99cbe265c6608954d9cea7c",
    "external/ipfs_accelerate": "ebd9b42ba48a71ca908e86a9467e581446be5930",
    "external/ipfs_datasets": "839980ccaba593b146213928bc5e8bf95e7a1fd9",
    "external/ipfs_kit": "b4413c1ad92b85df2e110c0185d2449e56fc6497",
}

EXPECTED_MISSING_PREDECESSORS: tuple[str, ...] = ()

# Paths are outer-repository evidence.  Source files that live behind gitlinks
# are bound transitively by their producing receipts and source commit/tree.
REQUIRED_INPUTS = (
    ("control", f"{ARTIFACT_ROOT}/control/task_board.json", "PCCE-000"),
    (
        "control",
        f"{ARTIFACT_ROOT}/control/task_dependency_graph.json",
        "PCCE-000",
    ),
    ("source", f"{ARTIFACT_ROOT}/inventory/ipfs_accelerate.json", "PCCE-003"),
    ("source", f"{ARTIFACT_ROOT}/inventory/ipfs_datasets.json", "PCCE-001"),
    ("source", f"{ARTIFACT_ROOT}/inventory/ipfs_kit.json", "PCCE-002"),
    ("source", f"{ARTIFACT_ROOT}/inventory/mcp_plus_plus.json", "PCCE-004"),
    (
        "contracts",
        f"{ARTIFACT_ROOT}/contracts/canonical_ownership.json",
        "PCCE-005",
    ),
    (
        "contracts",
        f"{ARTIFACT_ROOT}/contracts/compatibility_matrix.json",
        "PCCE-007",
    ),
    ("contracts", f"{ARTIFACT_ROOT}/contracts/epic_a_gate.json", "PCCE-011"),
    ("runtime", f"{ARTIFACT_ROOT}/runtime/runtime_api.json", "PCCE-020"),
    ("adapters", f"{ARTIFACT_ROOT}/adapters/conformance.json", "PCCE-035"),
    ("cli", f"{ARTIFACT_ROOT}/cli/command_manifest.json", "PCCE-044"),
    ("cli", f"{ARTIFACT_ROOT}/cli/output_schema.json", "PCCE-043"),
    (
        "environment",
        f"{ARTIFACT_ROOT}/environment/artifact_hashes.json",
        "PCCE-053",
    ),
    (
        "environment",
        f"{ARTIFACT_ROOT}/environment/dependency_locks.json",
        "PCCE-053",
    ),
    (
        "environment",
        f"{ARTIFACT_ROOT}/environment/sbom.spdx.json",
        "PCCE-053",
    ),
    (
        "environment",
        f"{ARTIFACT_ROOT}/environment/manifest.json",
        "PCCE-053",
    ),
    (
        "installation",
        f"{ARTIFACT_ROOT}/installation/qualification.json",
        "PCCE-056",
    ),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/thresholds.json", "PCCE-060"),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/raw_results.jsonl", "PCCE-067"),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/run_manifest.json", "PCCE-067"),
    (
        "benchmark",
        f"{ARTIFACT_ROOT}/benchmark/execution_receipt.json",
        "PCCE-067",
    ),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/metrics.json", "PCCE-068"),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/qualification.json", "PCCE-068"),
    ("benchmark", f"{ARTIFACT_ROOT}/benchmark/report.md", "PCCE-068"),
    (
        "self_hosting",
        f"{ARTIFACT_ROOT}/benchmark/self_hosting/attempts.jsonl",
        "PCCE-079",
    ),
    (
        "self_hosting",
        f"{ARTIFACT_ROOT}/benchmark/self_hosting/manifest.json",
        "PCCE-079",
    ),
    (
        "self_hosting",
        f"{ARTIFACT_ROOT}/benchmark/self_hosting/qualification.json",
        "PCCE-079",
    ),
    (
        "self_hosting",
        f"{ARTIFACT_ROOT}/benchmark/self_hosting/report.md",
        "PCCE-079",
    ),
    ("security", f"{ARTIFACT_ROOT}/security/threat_model.json", "PCCE-070"),
    ("security", f"{ARTIFACT_ROOT}/security/findings.json", "PCCE-076"),
    ("security", f"{ARTIFACT_ROOT}/security/qualification.json", "PCCE-076"),
    ("security", f"{ARTIFACT_ROOT}/security/report.md", "PCCE-076"),
    ("ci", f"{ARTIFACT_ROOT}/ci/required_jobs.json", "PCCE-080"),
)

OUTPUT_FILES = (
    "VERIFY.md",
    "known_limitations.json",
    "qualification.json",
    "release_manifest.json",
    "rollback.json",
    "verification.json",
)

FORBIDDEN_JSON_KEYS = {
    "answer_key",
    "api_key",
    "credentials",
    "expected_patch",
    "hidden_answer",
    "hidden_answers",
    "private_key",
    "provider_token",
}

FORBIDDEN_BYTE_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{24,}"),
    re.compile(rb"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{24,}"),
)


class ReleaseEvidenceError(RuntimeError):
    """Raised when rc1 cannot be assembled or verified without overclaiming."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_cid_v1_from_sha256(sha256: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ReleaseEvidenceError(f"invalid SHA-256 for raw CID: {sha256!r}")
    payload = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    cid = "b" + base64.b32encode(payload).decode("ascii").lower().rstrip("=")
    _verify_raw_cid_v1(cid, sha256)
    return cid


def _verify_raw_cid_v1(cid: str, sha256: str) -> None:
    if not cid.startswith("b") or cid != cid.lower():
        raise ReleaseEvidenceError(f"noncanonical raw CIDv1: {cid!r}")
    encoded = cid[1:].upper()
    encoded += "=" * ((8 - len(encoded) % 8) % 8)
    try:
        payload = base64.b32decode(encoded, casefold=False)
    except (ValueError, binascii.Error) as exc:
        raise ReleaseEvidenceError(f"undecodable raw CIDv1: {cid!r}") from exc
    expected = b"\x01\x55\x12\x20" + bytes.fromhex(sha256)
    if payload != expected:
        raise ReleaseEvidenceError(f"raw CIDv1 does not bind {sha256}: {cid}")


def _descriptor(path: str, value: bytes) -> dict[str, Any]:
    sha256 = _sha256_bytes(value)
    return {
        "path": path,
        "sha256": sha256,
        "cid_v1_raw": _raw_cid_v1_from_sha256(sha256),
        "size": len(value),
    }


def _git(*args: str, allow_missing: bool = False) -> bytes | None:
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(OUTER_ROOT), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return completed.stdout
    if allow_missing:
        return None
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    raise ReleaseEvidenceError(f"git {' '.join(args)} failed: {stderr}")


def _verify_snapshot_identity() -> None:
    tree = _git("rev-parse", f"{SOURCE_SNAPSHOT_COMMIT}^{{tree}}")
    if tree is None or tree.decode("ascii").strip() != SOURCE_SNAPSHOT_TREE:
        raise ReleaseEvidenceError("the frozen outer evidence-cut tree is unavailable or changed")


def _snapshot_bytes(path: str) -> bytes | None:
    object_name = f"{SOURCE_SNAPSHOT_COMMIT}:{path}"
    exists = _git("cat-file", "-e", object_name, allow_missing=True)
    if exists is None:
        return None
    return _git("show", object_name)


def _snapshot_json(path: str) -> dict[str, Any]:
    value = _snapshot_bytes(path)
    if value is None:
        raise ReleaseEvidenceError(f"required frozen JSON is missing: {path}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceError(f"malformed frozen JSON {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ReleaseEvidenceError(f"frozen JSON is not an object: {path}")
    return parsed


def _source_gitlinks() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(SOURCE_GITLINK_TREES):
        raw = _git("ls-tree", SOURCE_SNAPSHOT_COMMIT, "--", path)
        if raw is None:
            raise ReleaseEvidenceError(f"cannot inspect frozen gitlink: {path}")
        parts = raw.decode("ascii").strip().split(maxsplit=3)
        if len(parts) != 4 or parts[0:2] != ["160000", "commit"]:
            raise ReleaseEvidenceError(f"frozen path is not one gitlink: {path}")
        object_id, observed_path = parts[2], parts[3]
        if observed_path != path or re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
            raise ReleaseEvidenceError(f"malformed frozen gitlink: {path}")
        result.append(
            {
                "path": path,
                "commit": object_id,
                "tree": SOURCE_GITLINK_TREES[path],
                "tree_observation": "verified from the local submodule object at rc1 authoring",
            }
        )
    return result


def _task_board() -> dict[str, Any]:
    board = _snapshot_json(TASK_BOARD_PATH)
    if board.get("board_namespace") != BOARD_NAMESPACE or board.get("task_count") != 67:
        raise ReleaseEvidenceError("frozen task board namespace/count drifted")
    tasks = board.get("tasks")
    order = board.get("topological_order")
    if not isinstance(tasks, list) or not isinstance(order, list):
        raise ReleaseEvidenceError("frozen task board has no task list/topological order")
    if len(tasks) != 67 or len(order) != 67 or len(set(order)) != 67:
        raise ReleaseEvidenceError("frozen task board does not contain 67 unique tasks")
    if TASK_ID not in order:
        raise ReleaseEvidenceError(f"{TASK_ID} is absent from frozen topological order")
    return board


def _predecessor_receipts(board: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    tasks = {item.get("task_id"): item for item in board["tasks"]}
    order = board["topological_order"]
    predecessor_ids = order[: order.index(TASK_ID)]
    if len(predecessor_ids) != 64:
        raise ReleaseEvidenceError("rc1 expected exactly 64 predecessor tasks")
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in predecessor_ids:
        task = tasks.get(task_id)
        if not isinstance(task, dict):
            raise ReleaseEvidenceError(f"task descriptor missing for {task_id}")
        path = f"{RECEIPT_ROOT}/{task_id}.json"
        value = _snapshot_bytes(path)
        base = {
            "task_id": task_id,
            "canonical_task_cid": task.get("canonical_task_cid"),
            "canonical_task_key": task.get("canonical_task_key"),
            "receipt_path": path,
        }
        if value is None:
            missing.append(task_id)
            records.append({**base, "receipt_status": "missing-at-evidence-cut"})
            continue
        try:
            receipt = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ReleaseEvidenceError(f"malformed predecessor receipt {path}: {exc}") from exc
        if not isinstance(receipt, dict) or receipt.get("task_id") != task_id:
            raise ReleaseEvidenceError(f"predecessor receipt identity mismatch: {path}")
        descriptor = _descriptor(path, value)
        records.append(
            {
                **base,
                "receipt_status": receipt.get("status", "untyped"),
                "artifact_identity": receipt.get("artifact_identity"),
                "completion_mode": receipt.get("completion_mode"),
                "receipt_sha256": descriptor["sha256"],
                "receipt_cid_v1_raw": descriptor["cid_v1_raw"],
                "receipt_size": descriptor["size"],
            }
        )
    if tuple(missing) != EXPECTED_MISSING_PREDECESSORS:
        raise ReleaseEvidenceError(
            "frozen predecessor absence set drifted: "
            f"{missing!r} != {list(EXPECTED_MISSING_PREDECESSORS)!r}"
        )
    return records, missing


def _required_input_records() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for category, path, producer_task in REQUIRED_INPUTS:
        value = _snapshot_bytes(path)
        base = {
            "category": category,
            "path": path,
            "producer_task": producer_task,
        }
        if value is None:
            missing.append(path)
            records.append({**base, "status": "missing-at-evidence-cut"})
        else:
            records.append({**base, "status": "present", **_descriptor(path, value)})
    return records, missing


def _package_evidence() -> tuple[list[dict[str, Any]], list[str]]:
    path = f"{ARTIFACT_ROOT}/environment/artifact_hashes.json"
    hashes = _snapshot_json(path)
    artifacts = hashes.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 8:
        raise ReleaseEvidenceError("exactly four wheel/sdist package pairs are required")
    pairs: dict[str, set[str]] = {}
    unavailable: list[str] = []
    records: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise ReleaseEvidenceError("package artifact descriptor is not an object")
        distribution = str(item.get("distribution", ""))
        kind = str(item.get("kind", ""))
        filename = str(item.get("filename", ""))
        sha256 = str(item.get("sha256", ""))
        cid = str(item.get("cid_v1_raw", ""))
        if kind not in {"wheel", "sdist"} or not distribution or not filename:
            raise ReleaseEvidenceError(f"invalid package descriptor: {item!r}")
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ReleaseEvidenceError(f"invalid package hash: {filename}")
        _verify_raw_cid_v1(cid, sha256)
        pairs.setdefault(distribution, set()).add(kind)
        bytes_available = item.get("bytes_available") is True
        bytes_verified = item.get("bytes_verified") is True
        if not bytes_available or not bytes_verified:
            unavailable.append(filename)
        records.append(
            {
                "distribution": distribution,
                "version": item.get("version"),
                "kind": kind,
                "filename": filename,
                "sha256": sha256,
                "cid_v1_raw": cid,
                "size": item.get("size"),
                "source_commit": item.get("source_commit"),
                "descriptor_bytes_available": bytes_available,
                "descriptor_bytes_verified": bytes_verified,
                "bytes_copied_into_rc1": False,
                "hash_evidence": "admitted-PCCE-053-descriptor",
                "signature_evidence": "unavailable-not-observed",
                "qualification_credit": False,
            }
        )
    if len(pairs) != 4 or any(kinds != {"wheel", "sdist"} for kinds in pairs.values()):
        raise ReleaseEvidenceError("package evidence is not four complete wheel/sdist pairs")
    records.sort(key=lambda item: (str(item["distribution"]), str(item["kind"])))
    return records, sorted(unavailable)


def _receipt(task_id: str) -> dict[str, Any]:
    return _snapshot_json(f"{RECEIPT_ROOT}/{task_id}.json")


def _snapshot_descriptor(path: str) -> dict[str, Any]:
    value = _snapshot_bytes(path)
    if value is None:
        raise ReleaseEvidenceError(f"required frozen artifact is missing: {path}")
    return _descriptor(path, value)


def _contract_evidence() -> dict[str, Any]:
    schema_receipt = _receipt("PCCE-006")
    vector_receipt = _receipt("PCCE-007")
    package_receipt = _receipt("PCCE-057")
    schema_digests = schema_receipt.get("evidence", {}).get("schema_digests")
    packaged = package_receipt.get("evidence", {})
    if not isinstance(schema_digests, dict) or len(schema_digests) != 18:
        raise ReleaseEvidenceError("PCCE-006 must bind 17 schemas plus its schema test")
    packaged_schema_cids = packaged.get("schema_cids")
    if not isinstance(packaged_schema_cids, dict) or len(packaged_schema_cids) != 17:
        raise ReleaseEvidenceError("PCCE-057 must bind all 17 packaged schema CIDs")
    return {
        "schema_source": {
            "task_id": "PCCE-006",
            "commit": schema_receipt.get("bound_commit"),
            "tree": schema_receipt.get("bound_tree"),
            "schema_count": 17,
            "schema_digests": schema_digests,
        },
        "canonical_vectors": {
            "task_id": "PCCE-007",
            "sha256": vector_receipt.get("evidence", {}).get("vector_sha256"),
            "packaged": packaged.get("canonical_vectors"),
        },
        "data_only_contract_package": {
            "task_id": "PCCE-057",
            "runtime_authority": False,
            "contract_bundle": packaged.get("contract_bundle"),
            "schema_cids": packaged_schema_cids,
            "artifacts": packaged.get("artifacts"),
        },
    }


def _corpus_evidence() -> dict[str, Any]:
    freeze = _receipt("PCCE-060").get("evidence", {})
    typed = _receipt("PCCE-061").get("evidence", {})
    files = typed.get("nested_source", {}).get("files", {})
    visible_path = "benchmarks/proof_context/corpus/typed_structured/visible_manifest.json"
    evaluator_path = "benchmarks/proof_context/corpus/typed_structured/evaluator_manifest.json"
    visible = files.get(visible_path)
    evaluator = files.get(evaluator_path)
    if not isinstance(visible, dict) or not isinstance(evaluator, dict):
        raise ReleaseEvidenceError("typed corpus visible/evaluator identities are absent")
    return {
        "corpus_manifest": freeze.get("corpus_manifest"),
        "thresholds": freeze.get("thresholds"),
        "visible_partition_identity": {
            "path": visible_path,
            "sha256": visible.get("sha256"),
            "raw_cid_v1": visible.get("raw_cid_v1"),
            "structured_cid": visible.get("structured_cid"),
        },
        "sealed_evaluator_partition_identity": {
            "path": evaluator_path,
            "sha256": evaluator.get("sha256"),
            "raw_cid_v1": evaluator.get("raw_cid_v1"),
            "structured_cid": evaluator.get("structured_cid"),
            "body_included": False,
        },
        "isolation_claim_at_evidence_cut": typed.get("partition_isolation"),
    }


def _example_and_harness_evidence() -> dict[str, Any]:
    harness = _receipt("PCCE-045")
    example = _receipt("PCCE-055")
    self_hosting = _snapshot_json(f"{ARTIFACT_ROOT}/benchmark/self_hosting/qualification.json")
    self_hosting_receipt = _receipt("PCCE-079")
    return {
        "packaged_self_hosting_harness": {
            "task_id": "PCCE-045",
            "artifact_identity": harness.get("artifact_identity"),
            "authority": harness.get("evidence", {}).get("authority"),
            "evidence_schema": harness.get("evidence", {}).get("evidence_schema"),
            "surface": harness.get("evidence", {}).get("surface"),
        },
        "bounded_self_hosting_disposition": {
            "task_id": "PCCE-079",
            "status": "present-and-bound-no-go",
            "decision": self_hosting.get("decision"),
            "qualification_status": self_hosting.get("status"),
            "qualification_cap": self_hosting.get("qualification_cap"),
            "release_qualified": False,
            "qualification_credit": False,
            "qualification": _snapshot_descriptor(
                f"{ARTIFACT_ROOT}/benchmark/self_hosting/qualification.json"
            ),
            "receipt_artifact_identity": self_hosting_receipt.get("artifact_identity"),
        },
        "example": {
            "task_id": "PCCE-055",
            "artifact_identity": example.get("artifact_identity"),
            "nested_repository": example.get("evidence", {}).get("nested_repository"),
            "walkthrough": example.get("evidence", {}).get("walkthrough"),
            "acceptance": example.get("evidence", {}).get("acceptance"),
            "proof_reuse": example.get("evidence", {}).get("proof_reuse"),
            "seal": example.get("evidence", {}).get("seal"),
        },
    }


def _blockers(
    *, missing_predecessors: list[str], missing_inputs: list[str], unavailable: list[str]
) -> list[dict[str, Any]]:
    benchmark_path = f"{ARTIFACT_ROOT}/benchmark/qualification.json"
    benchmark = _snapshot_json(benchmark_path)
    benchmark_decision = benchmark.get("decision", {})
    security_path = f"{ARTIFACT_ROOT}/security/qualification.json"
    security = _snapshot_json(security_path)
    self_hosting_path = f"{ARTIFACT_ROOT}/benchmark/self_hosting/qualification.json"
    self_hosting = _snapshot_json(self_hosting_path)
    ci_path = f"{ARTIFACT_ROOT}/ci/required_jobs.json"
    ci = _snapshot_json(ci_path)
    ci_qualification = ci.get("qualification", {})
    blockers: list[dict[str, Any]] = [
        {
            "id": "installability-no-go",
            "source": f"{ARTIFACT_ROOT}/installation/qualification.json",
            "observed_decision": "NO-GO",
            "waiver": None,
        },
        {
            "id": "package-byte-set-incomplete",
            "unavailable_artifacts": unavailable,
            "rc1_contains_archive_bytes": False,
            "waiver": None,
        },
        {
            "id": "package-signature-evidence-unavailable",
            "signed_package_claim": False,
            "signature_results_observed": 0,
            "waiver": None,
        },
        {
            "id": "bounded-self-hosting-qualification-no-go",
            "source_task": "PCCE-079",
            "source": self_hosting_path,
            "source_cid_v1_raw": _snapshot_descriptor(self_hosting_path)["cid_v1_raw"],
            "observed_decision": self_hosting.get("decision"),
            "qualification_status": self_hosting.get("status"),
            "waiver": None,
        },
        {
            "id": "benchmark-qualification-no-go",
            "source_tasks": ["PCCE-067", "PCCE-068"],
            "source": benchmark_path,
            "source_cid_v1_raw": _snapshot_descriptor(benchmark_path)["cid_v1_raw"],
            "observed_decision": benchmark_decision.get("status"),
            "release_qualified": benchmark_decision.get("release_qualified"),
            "provider_or_model_runs_claimed": False,
            "waiver": None,
        },
        {
            "id": "security-qualification-no-go",
            "source_task": "PCCE-076",
            "source": security_path,
            "source_cid_v1_raw": _snapshot_descriptor(security_path)["cid_v1_raw"],
            "observed_decision": security.get("decision"),
            "release_qualified": security.get("qualification", {}).get("release_qualified"),
            "security_release_claim": False,
            "waiver": None,
        },
        {
            "id": "current-head-ci-qualification-no-go",
            "source_task": "PCCE-080",
            "source": ci_path,
            "source_cid_v1_raw": _snapshot_descriptor(ci_path)["cid_v1_raw"],
            "observed_decision": ci_qualification.get("decision"),
            "release_qualified": ci_qualification.get("release_qualified"),
            "external_ci_authority_available": ci_qualification.get(
                "external_ci_authority_available"
            ),
            "ci_run_claimed": False,
            "waiver": None,
        },
    ]
    if missing_predecessors:
        blockers.insert(
            1,
            {
                "id": "predecessor-receipts-incomplete",
                "missing_task_ids": missing_predecessors,
                "waiver": None,
            },
        )
    if missing_inputs:
        blockers.insert(
            1,
            {
                "id": "required-release-inputs-missing",
                "missing_paths": missing_inputs,
                "waiver": None,
            },
        )
    return blockers


def _known_limitations(blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.release-limitations@1",
        "release_candidate": RELEASE_CANDIDATE,
        "decision": "NO-GO",
        "limitations": [
            {
                "id": blocker["id"],
                "release_effect": "blocks-promotion",
                "waived": False,
            }
            for blocker in blockers
        ],
        "claims_not_made": [
            "current-head CI passed",
            "live provider or model benchmark completed",
            "package signatures verified",
            "production sandbox resistance established",
            "release package archive set is complete",
            "security or benchmark qualification passed",
            "time-separated self-hosting qualification passed",
        ],
        "qualification_level_assignment": "deferred-to-PCCE-082",
    }


def _qualification(blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.release-candidate-qualification@1",
        "release_candidate": RELEASE_CANDIDATE,
        "decision": "NO-GO",
        "release_qualified": False,
        "promotable": False,
        "blocker_ids": [item["id"] for item in blockers],
        "waivers": [],
        "qualification_level": None,
        "qualification_level_owner": "PCCE-082",
        "basis": "frozen-evidence-cut-only",
    }


def _rollback() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.release-rollback@1",
        "release_candidate": RELEASE_CANDIDATE,
        "published_externally": False,
        "tag_created": False,
        "rollback_steps": [
            "retain rc1 manifest, receipt, and NO-GO evidence",
            "withdraw or ignore the unpromoted rc1 directory identity",
            "repair only through the owning predecessor tasks",
            "assemble v0.1-rc2; never overwrite rc1 with different bytes",
        ],
        "destructive_cleanup_required": False,
        "next_candidate_on_evidence_change": "v0.1-rc2",
    }


def _verification_record() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}.release-verification@1",
        "release_candidate": RELEASE_CANDIDATE,
        "source_snapshot": {
            "commit": SOURCE_SNAPSHOT_COMMIT,
            "tree": SOURCE_SNAPSHOT_TREE,
        },
        "deterministic_two_build": {
            "required": True,
            "performed_by_build_command": True,
            "comparison": "exact relative-path set and byte equality",
        },
        "checks": [
            "frozen commit/tree availability",
            "all manifest input SHA-256 and raw CID bindings",
            "exact bundle path set and byte-for-byte reconstruction",
            "decision is fail-closed while blockers are nonempty",
            "bounded credential-pattern and hidden-body-key scan",
        ],
        "scan": {
            "scope": "all six files in the bounded rc1 directory",
            "credential_or_hidden_body_matches": 0,
            "result": "passed-by-build-and-check-command",
        },
    }


def _verify_instructions() -> bytes:
    text = """# PCCE v0.1-rc1 verification

This directory is an immutable, unpromoted **NO-GO** evidence bundle. It is
not a signed release and does not claim a provider run, a CI run, benchmark
qualification, security qualification, or package publication.

From the outer repository root, run:

```sh
python external/ipfs_accelerate/scripts/proof_context/build_release_candidate.py --check artifacts/proof_carrying_context_engine/release/v0.1-rc1
python -m json.tool artifacts/proof_carrying_context_engine/release/v0.1-rc1/release_manifest.json
python -m json.tool artifacts/proof_carrying_context_engine/release/v0.1-rc1/qualification.json
```

The first command reconstructs every byte from the frozen Git evidence cut,
checks all transitive SHA-256/raw-CID bindings, and repeats the secret/hidden
body scan. A different evidence population requires `v0.1-rc2`; rc1 must not
be overwritten.
"""
    return text.encode("utf-8")


def _manifest(
    *,
    supporting: dict[str, bytes],
    board: dict[str, Any],
    predecessors: list[dict[str, Any]],
    missing_predecessors: list[str],
    inputs: list[dict[str, Any]],
    missing_inputs: list[str],
    packages: list[dict[str, Any]],
    unavailable_packages: list[str],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    task = next(item for item in board["tasks"] if item.get("task_id") == TASK_ID)
    installation_path = f"{ARTIFACT_ROOT}/installation/qualification.json"
    installation = _snapshot_json(installation_path)
    benchmark_path = f"{ARTIFACT_ROOT}/benchmark/qualification.json"
    benchmark = _snapshot_json(benchmark_path)
    security_path = f"{ARTIFACT_ROOT}/security/qualification.json"
    security = _snapshot_json(security_path)
    ci_path = f"{ARTIFACT_ROOT}/ci/required_jobs.json"
    ci = _snapshot_json(ci_path)
    self_hosting_path = f"{ARTIFACT_ROOT}/benchmark/self_hosting/qualification.json"
    self_hosting = _snapshot_json(self_hosting_path)
    environment = _snapshot_json(f"{ARTIFACT_ROOT}/environment/manifest.json")
    script_bytes = SCRIPT_PATH.read_bytes()
    return {
        "schema": f"{SCHEMA_PREFIX}.release-manifest@1",
        "release_candidate": RELEASE_CANDIDATE,
        "board_namespace": BOARD_NAMESPACE,
        "task_authority": {
            "task_id": TASK_ID,
            "canonical_task_cid": task.get("canonical_task_cid"),
            "canonical_task_key": task.get("canonical_task_key"),
            "database_mutated_by_builder": False,
        },
        "decision": {
            "status": "NO-GO",
            "release_qualified": False,
            "promotable": False,
            "blocker_count": len(blockers),
            "waiver_count": 0,
        },
        "evidence_cut": {
            "outer_commit": SOURCE_SNAPSHOT_COMMIT,
            "outer_tree": SOURCE_SNAPSHOT_TREE,
            "gitlinks": _source_gitlinks(),
            "historical_not_current_head_claim": True,
        },
        "builder": {
            **_descriptor(
                "external/ipfs_accelerate/scripts/proof_context/build_release_candidate.py",
                script_bytes,
            ),
            "deterministic_json": "UTF-8, sorted keys, two-space indentation, LF",
            "network_access_required": False,
        },
        "predecessors": {
            "required_count": len(predecessors),
            "present_receipt_count": len(predecessors) - len(missing_predecessors),
            "missing_receipt_count": len(missing_predecessors),
            "records": predecessors,
        },
        "source_and_input_artifacts": {
            "required_count": len(inputs),
            "present_count": len(inputs) - len(missing_inputs),
            "missing_count": len(missing_inputs),
            "records": inputs,
        },
        "packages": {
            "distribution_count": 4,
            "wheel_sdist_descriptor_count": len(packages),
            "archive_bytes_copied_count": 0,
            "unavailable_descriptor_bytes": unavailable_packages,
            "signature_results_observed": 0,
            "records": packages,
        },
        "environment": {
            "manifest_path": f"{ARTIFACT_ROOT}/environment/manifest.json",
            "environment_id": environment.get("environment_id"),
            "artifact_clean_install_status": environment.get("artifact_clean_install_status"),
            "artifact_byte_availability_status": environment.get(
                "artifact_byte_availability_status"
            ),
            "dependency_locks_bound": True,
            "sbom_bound": True,
        },
        "contracts": _contract_evidence(),
        "corpus": _corpus_evidence(),
        "harness_and_example": _example_and_harness_evidence(),
        "qualification_inputs": {
            "installation": {
                "status": "present-and-bound-no-go",
                "decision": installation.get("decision"),
                "release_qualified": installation.get("release_qualified"),
                "waivers": installation.get("waivers"),
                "binding": _snapshot_descriptor(installation_path),
            },
            "benchmark": {
                "status": "present-and-bound-no-go",
                "decision": benchmark.get("decision", {}).get("status"),
                "decision_kind": benchmark.get("decision", {}).get("decision_kind"),
                "release_qualified": benchmark.get("decision", {}).get("release_qualified"),
                "binding": _snapshot_descriptor(benchmark_path),
            },
            "security": {
                "status": "present-and-bound-no-go",
                "decision": security.get("decision"),
                "release_qualified": security.get("qualification", {}).get("release_qualified"),
                "binding": _snapshot_descriptor(security_path),
            },
            "current_head_ci": {
                "status": "present-and-bound-no-go",
                "decision": ci.get("qualification", {}).get("decision"),
                "release_qualified": ci.get("qualification", {}).get("release_qualified"),
                "external_ci_authority_available": ci.get("qualification", {}).get(
                    "external_ci_authority_available"
                ),
                "binding": _snapshot_descriptor(ci_path),
            },
            "longitudinal_self_hosting": {
                "status": "present-and-bound-no-go",
                "decision": self_hosting.get("decision"),
                "qualification_status": self_hosting.get("status"),
                "qualification_cap": self_hosting.get("qualification_cap"),
                "release_qualified": False,
                "binding": _snapshot_descriptor(self_hosting_path),
            },
        },
        "blockers": blockers,
        "bundle_files": [
            _descriptor(f"{RELEASE_RELATIVE_PATH}/{name}", value)
            for name, value in sorted(supporting.items())
        ],
        "identity_policy": {
            "manifest_identity": "CIDv1(raw,sha2-256) of exact release_manifest.json bytes",
            "bundle_identity": "manifest identity plus its transitively bound bundle_files",
            "manifest_identity_recorded_by": f"{RECEIPT_ROOT}/{TASK_ID}.json",
        },
        "external_publication_performed": False,
        "release_tag_created": False,
    }


def _assemble(directory: Path) -> None:
    _verify_snapshot_identity()
    board = _task_board()
    predecessors, missing_predecessors = _predecessor_receipts(board)
    inputs, missing_inputs = _required_input_records()
    packages, unavailable_packages = _package_evidence()
    blockers = _blockers(
        missing_predecessors=missing_predecessors,
        missing_inputs=missing_inputs,
        unavailable=unavailable_packages,
    )
    supporting = {
        "VERIFY.md": _verify_instructions(),
        "known_limitations.json": _json_bytes(_known_limitations(blockers)),
        "qualification.json": _json_bytes(_qualification(blockers)),
        "rollback.json": _json_bytes(_rollback()),
        "verification.json": _json_bytes(_verification_record()),
    }
    manifest = _manifest(
        supporting=supporting,
        board=board,
        predecessors=predecessors,
        missing_predecessors=missing_predecessors,
        inputs=inputs,
        missing_inputs=missing_inputs,
        packages=packages,
        unavailable_packages=unavailable_packages,
        blockers=blockers,
    )
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in {**supporting, "release_manifest.json": _json_bytes(manifest)}.items():
        (directory / name).write_bytes(value)


def _relative_file_bytes(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise ReleaseEvidenceError(f"release directory does not exist: {directory}")
    found: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ReleaseEvidenceError(f"release bundle must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            found[relative] = path.read_bytes()
        elif path != directory and not path.is_dir():
            raise ReleaseEvidenceError(f"unsupported release entry: {path}")
    if tuple(sorted(found)) != OUTPUT_FILES:
        raise ReleaseEvidenceError(
            f"release path set differs: {sorted(found)!r} != {list(OUTPUT_FILES)!r}"
        )
    return found


def _walk_json_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json_keys(child)


def _scan_bundle(directory: Path) -> None:
    files = _relative_file_bytes(directory)
    findings: list[str] = []
    for relative, value in files.items():
        for pattern in FORBIDDEN_BYTE_PATTERNS:
            if pattern.search(value):
                findings.append(f"{relative}:credential-pattern")
        if relative.endswith(".json"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ReleaseEvidenceError(f"malformed bundle JSON {relative}: {exc}") from exc
            forbidden = sorted(set(_walk_json_keys(parsed)) & FORBIDDEN_JSON_KEYS)
            findings.extend(f"{relative}:forbidden-key:{key}" for key in forbidden)
    if findings:
        raise ReleaseEvidenceError(f"secret/hidden-body scan failed: {findings!r}")


def _validate_manifest(directory: Path) -> dict[str, Any]:
    files = _relative_file_bytes(directory)
    try:
        manifest = json.loads(files["release_manifest.json"])
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceError(f"malformed release manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReleaseEvidenceError("release manifest is not a JSON object")
    expected_schema = f"{SCHEMA_PREFIX}.release-manifest@1"
    if manifest.get("schema") != expected_schema:
        raise ReleaseEvidenceError("release manifest schema is unsupported")
    decision = manifest.get("decision")
    blockers = manifest.get("blockers")
    if not isinstance(decision, dict) or not isinstance(blockers, list) or not blockers:
        raise ReleaseEvidenceError("release manifest must retain its blockers")
    if decision != {
        "status": "NO-GO",
        "release_qualified": False,
        "promotable": False,
        "blocker_count": len(blockers),
        "waiver_count": 0,
    }:
        raise ReleaseEvidenceError("release decision is not the exact fail-closed disposition")
    descriptors = manifest.get("bundle_files")
    if not isinstance(descriptors, list) or len(descriptors) != len(files) - 1:
        raise ReleaseEvidenceError("manifest bundle-file descriptors are incomplete")
    expected_paths = {
        f"{RELEASE_RELATIVE_PATH}/{name}" for name in files if name != "release_manifest.json"
    }
    observed_paths: set[str] = set()
    for item in descriptors:
        if not isinstance(item, dict):
            raise ReleaseEvidenceError("bundle-file descriptor is not an object")
        path = str(item.get("path", ""))
        prefix = f"{RELEASE_RELATIVE_PATH}/"
        if not path.startswith(prefix):
            raise ReleaseEvidenceError(f"bundle-file path escaped rc1: {path!r}")
        relative = path.removeprefix(prefix)
        if relative not in files or relative == "release_manifest.json":
            raise ReleaseEvidenceError(f"unexpected bundle-file descriptor: {path!r}")
        expected = _descriptor(path, files[relative])
        if item != expected:
            raise ReleaseEvidenceError(f"bundle-file digest mismatch: {path}")
        observed_paths.add(path)
    if observed_paths != expected_paths:
        raise ReleaseEvidenceError("manifest bundle-file path set is incomplete")
    for section in ("predecessors", "source_and_input_artifacts"):
        records = manifest.get(section, {}).get("records", [])
        for record in records:
            sha256 = record.get("receipt_sha256", record.get("sha256"))
            cid = record.get("receipt_cid_v1_raw", record.get("cid_v1_raw"))
            if sha256 is not None or cid is not None:
                if not isinstance(sha256, str) or not isinstance(cid, str):
                    raise ReleaseEvidenceError(f"partial digest binding in {section}")
                _verify_raw_cid_v1(cid, sha256)
    return manifest


def _build(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pcce081-build-a-") as first_tmp:
        with tempfile.TemporaryDirectory(prefix="pcce081-build-b-") as second_tmp:
            first = Path(first_tmp) / RELEASE_CANDIDATE
            second = Path(second_tmp) / RELEASE_CANDIDATE
            _assemble(first)
            _assemble(second)
            first_files = _relative_file_bytes(first)
            second_files = _relative_file_bytes(second)
            if first_files != second_files:
                raise ReleaseEvidenceError("two deterministic rc1 builds differ")
            _scan_bundle(first)
            _validate_manifest(first)
            if output.exists():
                existing = _relative_file_bytes(output)
                if existing != first_files:
                    raise ReleaseEvidenceError(
                        "rc1 already exists with different bytes; build rc2 instead of overwriting"
                    )
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(first, output)
    return _check(output)


def _check(output: Path) -> dict[str, Any]:
    actual = _relative_file_bytes(output)
    with tempfile.TemporaryDirectory(prefix="pcce081-check-") as temporary:
        expected_dir = Path(temporary) / RELEASE_CANDIDATE
        _assemble(expected_dir)
        expected = _relative_file_bytes(expected_dir)
    if actual != expected:
        changed = sorted(
            name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)
        )
        raise ReleaseEvidenceError(f"rc1 differs from deterministic reconstruction: {changed}")
    _scan_bundle(output)
    manifest = _validate_manifest(output)
    manifest_bytes = actual["release_manifest.json"]
    sha256 = _sha256_bytes(manifest_bytes)
    return {
        "task_id": TASK_ID,
        "release_candidate": RELEASE_CANDIDATE,
        "decision": manifest["decision"]["status"],
        "release_qualified": manifest["decision"]["release_qualified"],
        "file_count": len(actual),
        "release_manifest_sha256": sha256,
        "release_manifest_cid_v1_raw": _raw_cid_v1_from_sha256(sha256),
        "deterministic_reconstruction": "passed",
        "secret_and_hidden_body_scan": "passed",
    }


def _resolve_output(value: str | None) -> Path:
    path = Path(value) if value else RELEASE_RELATIVE_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--build",
        nargs="?",
        const=str(RELEASE_RELATIVE_PATH),
        metavar="DIRECTORY",
        help="build rc1 twice and publish only exact deterministic bytes",
    )
    action.add_argument(
        "--check",
        nargs="?",
        const=str(RELEASE_RELATIVE_PATH),
        metavar="DIRECTORY",
        help="read-only reconstruction and verification of an existing rc1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.build is not None:
            result = _build(_resolve_output(args.build))
        else:
            result = _check(_resolve_output(args.check))
    except ReleaseEvidenceError as exc:
        print(f"PCCE-081 release evidence error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
