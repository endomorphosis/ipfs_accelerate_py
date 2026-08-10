"""ASE3-000 current-main convergence and historical-state isolation tests."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.validation import (
    prompt_v3_convergence as convergence_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.prompt_v3_convergence import (
    ACCEPTANCE_CHILD_CHANGED_PATHS,
    ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
    ARTIFACT_FILENAMES,
    BOARD_NAMESPACE,
    DEFAULT_ARTIFACT_ROOT,
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
    FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
    FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,
    FAILED_VALIDATION_EVENT_019_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
    FALSE_COMPLETION_RECOVERY_FILENAME,
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME,
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
    MANIFEST_FILENAME,
    MAX_EVIDENCE_SNAPSHOT_BYTES,
    MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES,
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME,
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME,
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
    NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
    OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES,
    OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS,
    OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
    OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
    POST_WAVE3_RESIDUAL_FILENAME,
    PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    PROTECTED_RUNTIME_ACTIVATION_AUTHORIZATION_SCHEMA,
    PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME,
    PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_RELATIVE_PATH,
    PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_SCHEMA,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
    SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
    ConvergenceManifest,
    CurrentMainBaseline,
    RescueDispositionReport,
    canonical_operator_acceptance_review_bytes,
    load_operator_acceptance_receipt,
    validate_acceptance_child_transition,
    validate_ase3_019_accepted_control_plane,
    validate_convergence_artifacts,
    validate_git_generation_provenance,
    validate_operator_acceptance_signature,
    validate_operator_repair_acceptance_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "config"
    / "agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
)
TASKBOARD_PATH = REPO_ROOT / PROMPT_V3_TASKBOARD_RELATIVE_PATH
VALIDATOR_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "prompt_v3_convergence.py"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _rebind_component_digest(root: Path, filename: str) -> None:
    manifest_path = root / MANIFEST_FILENAME
    manifest = _load(manifest_path)
    components = manifest["components"]
    assert isinstance(components, dict)
    components[filename] = "sha256:" + hashlib.sha256(
        (root / filename).read_bytes()
    ).hexdigest()
    _write(manifest_path, manifest)


def _recompute_event_id(event: dict[str, object]) -> str:
    body = dict(event)
    body.pop("event_id", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _base58btc_encode(raw: bytes) -> str:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    accumulator = int.from_bytes(raw, "big")
    encoded = ""
    while accumulator:
        accumulator, remainder = divmod(accumulator, 58)
        encoded = alphabet[remainder] + encoded
    leading_zeroes = len(raw) - len(raw.lstrip(b"\x00"))
    return ("1" * leading_zeroes) + (encoded or "1")


def _reviewer_identity(private_key: Ed25519PrivateKey) -> str:
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return "did:key:z" + _base58btc_encode(b"\xed\x01" + public)


def _sign_operator_receipt(
    payload: dict[str, object],
    private_key: Ed25519PrivateKey,
) -> None:
    review = payload["review"]
    assert isinstance(review, dict)
    review["signature"] = ""
    signature = private_key.sign(canonical_operator_acceptance_review_bytes(payload))
    review["signature"] = (
        "ed25519:"
        + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    )


def _review_authority(reviewer: str) -> dict[str, object]:
    return {
        "reviewer_identity": reviewer,
        "reviewer_provider": "local_operator",
        "profile_id": "local-operator-profile",
        "profile_content_id": "sha256:" + ("1" * 64),
        "lifecycle_anchor_id": "2" * 64,
        "lifecycle_anchor_digest": "sha256:" + ("3" * 64),
        "lifecycle_generation": 1,
        "lifecycle_witness_path": (
            convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
        ),
        "lifecycle_witness_sha256": "sha256:" + ("4" * 64),
        "lifecycle_witness_id": "sha256:" + ("5" * 64),
        "lifecycle_witness_nonce": "operator-witness-nonce",
        "lifecycle_root_pin_path": (
            convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ),
        "lifecycle_root_pin_sha256": "sha256:" + ("6" * 64),
        "lifecycle_root_identity_did": reviewer,
        "fallback_authorization_id": "sha256:" + ("7" * 64),
        "fallback_authorization_sha256": "sha256:" + ("8" * 64),
        "lifecycle_witness_observed_at_ms": 1_786_215_600_000,
        "lifecycle_witness_expires_at_ms": 1_786_222_800_000,
        "fallback_authorized_at_ms": 1_786_217_400_000,
    }


def _receipt_review_authority(
    authority: dict[str, object],
) -> dict[str, object]:
    return {
        field: authority[field]
        for field in convergence_module._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
    }


def _operator_repair_receipt_027(
) -> tuple[dict[str, object], str, dict[str, object]]:
    private_key = Ed25519PrivateKey.generate()
    reviewer = _reviewer_identity(private_key)
    authority = _review_authority(reviewer)
    final_values = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-027"
    ]
    generations = json.loads(json.dumps(final_values["generations"]))
    final_blobs = dict(final_values["final_blobs"])
    contracts = convergence_module._ACCEPTANCE_TASK_CONTRACTS["ASE3-027"]
    parent_head = "d32415e4308a8462e96b4d04f807338f0a2d8b53"
    parent_tree = "87191ce65498a637c7b9500d72d434cadb8efbef"
    created_at = "2026-08-08T20:00:00Z"
    payload: dict[str, object] = {
        "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "task": {
            "task_id": "ASE3-027",
            "canonical_task_cid": contracts["canonical_task_cid"],
            "goal_id": contracts["goal_id"],
            "repairs_task": contracts["repairs_task"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "recovery": {
            "artifact": "false_completion_recovery_20260808.json",
            "pointer": "false_completions/ASE3-018",
            "historical_completion_authority": False,
            "branch_local_completion_authority": False,
            "repair_required": True,
        },
        "implementation": {
            "generations": generations,
            "final_blobs": final_blobs,
        },
        "acceptance_parent": {
            "head": parent_head,
            "tree": parent_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(
                convergence_module._sequential_future_artifacts_after("A032")
            ),
            "task_statuses": convergence_module._sequential_task_statuses_after(
                "A032"
            ),
            "reload_gate_status": "blocked",
        },
        "validation": {
            "command": convergence_module._FALSE_COMPLETION_REPAIR_TASKS[
                "ASE3-027"
            ]["validation"],
            "exit_code": 0,
            "passed": True,
            "passed_count": 174,
            "failed_count": 0,
            "validated_head": parent_head,
            "validated_tree": parent_tree,
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-027-repair",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._REPAIR_ACCEPTANCE_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, reviewer, authority


def _operator_repair_receipt_023(
) -> tuple[dict[str, object], str, dict[str, object]]:
    private_key = Ed25519PrivateKey.generate()
    reviewer = _reviewer_identity(private_key)
    authority = _review_authority(reviewer)
    final_values = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-023"
    ]
    generations = json.loads(json.dumps(final_values["generations"]))
    final_blobs = dict(final_values["final_blobs"])
    contracts = convergence_module._ACCEPTANCE_TASK_CONTRACTS["ASE3-023"]
    parent_head = "a43b2ce74816ac9226f6319b92425d0b002b6be6"
    parent_tree = "bbb94ffe87c3b582e40b1052ba5b9dc1ca8b4c40"
    created_at = "2026-08-09T22:00:00Z"
    payload: dict[str, object] = {
        "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "task": {
            "task_id": "ASE3-023",
            "canonical_task_cid": contracts["canonical_task_cid"],
            "goal_id": contracts["goal_id"],
            "repairs_task": contracts["repairs_task"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "recovery": {
            "artifact": "false_completion_recovery_20260808.json",
            "pointer": "false_completions/ASE3-006",
            "historical_completion_authority": False,
            "branch_local_completion_authority": False,
            "repair_required": True,
        },
        "implementation": {
            "generations": generations,
            "final_blobs": final_blobs,
        },
        "acceptance_parent": {
            "head": parent_head,
            "tree": parent_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(
                convergence_module._sequential_future_artifacts_after("A032")
            ),
            "task_statuses": convergence_module._sequential_task_statuses_after(
                "A032"
            ),
            "reload_gate_status": "blocked",
        },
        "validation": {
            "command": convergence_module._FALSE_COMPLETION_REPAIR_TASKS[
                "ASE3-023"
            ]["validation"],
            "exit_code": 0,
            "passed": True,
            "passed_count": 110,
            "failed_count": 0,
            "validated_head": parent_head,
            "validated_tree": parent_tree,
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-023-repair",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._REPAIR_ACCEPTANCE_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, reviewer, authority


def _minimal_operator_receipt(task_id: str) -> dict[str, object]:
    expected = convergence_module._ACCEPTANCE_TASK_CONTRACTS[task_id]
    if task_id == "ASE3-019":
        fields = convergence_module._ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS
    elif task_id == "ASE3-030":
        fields = convergence_module._HERMETIC_IDENTITY_ACCEPTANCE_REQUIRED_FIELDS
    else:
        fields = convergence_module._OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS
    payload: dict[str, object] = {field: {} for field in fields}
    payload.update(
        {
            "schema": expected["schema"],
            "created_at": "2026-08-08T20:00:00Z",
            "board_namespace": BOARD_NAMESPACE,
            "task": {"task_id": task_id},
        }
    )
    return payload


def _hermetic_acceptance_receipt_fixture(
    *,
    signing_key: Ed25519PrivateKey | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    private_key = signing_key or Ed25519PrivateKey.generate()
    authority = _review_authority(_reviewer_identity(private_key))
    paths = sorted(convergence_module._HERMETIC_REQUIRED_MODULE_MEMBER_MAP.values())
    blobs = {path: f"{index + 1:040x}" for index, path in enumerate(paths)}
    raw = {
        path: "sha256:" + hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in paths
    }
    members = {
        path: {
            "git_blob": blobs[path],
            "raw_sha256": raw[path],
            "archive_member_sha256": raw[path],
        }
        for path in paths
    }
    origins = {
        module: {
            "member_path": path,
            "origin": f"capsule://sealed/{path}",
        }
        for module, path in sorted(
            convergence_module._HERMETIC_REQUIRED_MODULE_MEMBER_MAP.items()
        )
    }
    parent_head = "1" * 40
    parent_tree = "2" * 40
    manifest = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "control-plane-dependency-manifest@1"
        ),
        "source_head": parent_head,
        "source_tree": parent_tree,
        "member_paths": paths,
        "module_names": list(origins),
        "cid_profile": "cidv1-base32-lower-raw+dag-json-sha2-256",
    }
    manifest_sha = convergence_module._canonical_sha256(manifest)
    archive_sha = "sha256:" + ("a" * 64)
    archive_root = convergence_module._canonical_sha256(
        {"member_paths": paths, "members": members}
    )
    descriptor_sha = "sha256:" + ("b" * 64)
    capsule = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor.sealed-control-plane-capsule@1"
        ),
        "manifest_sha256": manifest_sha,
        "archive_sha256": archive_sha,
        "archive_root_sha256": archive_root,
        "sealed_descriptor_sha256": descriptor_sha,
        "member_count": len(paths),
    }
    capsule_sha = convergence_module._canonical_sha256(capsule)
    generation = {
        "role": "hermetic-control-plane",
        "source_commit": "3" * 40,
        "source_parent": "4" * 40,
        "source_tree": "5" * 40,
        "replay_commit": "6" * 40,
        "replay_parent": "7" * 40,
        "replay_tree": "8" * 40,
        "integrated_commit": "9" * 40,
        "integrated_parent": "a" * 40,
        "integrated_tree": "b" * 40,
        "source_patch_sha256": "sha256:" + ("c" * 64),
        "replay_patch_sha256": "sha256:" + ("c" * 64),
        "integrated_patch_sha256": "sha256:" + ("c" * 64),
        "changed_paths": ["ipfs_accelerate_py/llm_router.py"],
    }
    probe_command = list(convergence_module._HERMETIC_HOSTILE_PROBE_ARGV)
    suite_report = "sha256:" + ("d" * 64)
    frozen: dict[str, object] = {
        "ready": True,
        "generations": [generation],
        "final_blobs": blobs,
        "final_raw_sha256": raw,
        "member_paths": paths,
        "module_origins": origins,
        "manifest_sha256": manifest_sha,
        "capsule_sha256": capsule_sha,
        "archive_sha256": archive_sha,
        "archive_root_sha256": archive_root,
        "sealed_descriptor_sha256": descriptor_sha,
        "probe_command": probe_command,
        "suite_passed_count": 37,
        "suite_report_sha256": suite_report,
    }
    contract = convergence_module._ACCEPTANCE_TASK_CONTRACTS["ASE3-030"]
    created_at = "2026-08-08T20:00:00Z"
    payload: dict[str, object] = {
        "schema": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "task": {
            "task_id": "ASE3-030",
            "canonical_task_cid": contract["canonical_task_cid"],
            "goal_id": contract["goal_id"],
            "repairs_task": contract["repairs_task"],
            "todo_contract_sha256": contract["todo_contract_sha256"],
            "completed_contract_sha256": contract["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "acceptance_parent": {
            "head": parent_head,
            "tree": parent_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(
                convergence_module._sequential_future_artifacts_after("A019")
            ),
            "task_statuses": convergence_module._sequential_task_statuses_after(
                "A019"
            ),
            "reload_gate_status": "blocked",
        },
        "provenance": {
            "generations": [generation],
            "final_blobs": blobs,
            "final_raw_sha256": raw,
        },
        "closure": {
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "capsule": capsule,
            "capsule_sha256": capsule_sha,
            "archive": {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "deterministic-control-plane-archive@1"
                ),
                "format": "zip-stored-sorted-v1",
                "sha256": archive_sha,
                "root_sha256": archive_root,
                "member_paths": paths,
            },
            "members": members,
            "module_origins": origins,
            "cid_vectors": json.loads(
                json.dumps(convergence_module._HERMETIC_CID_VECTORS)
            ),
        },
        "probe": {
            "command": probe_command,
            "environment": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": None},
            "exit_code": 0,
            "isolated": True,
            "user_site_enabled": False,
            "pythonpath_present": False,
            "multiformats_imported": False,
            "repository_or_candidate_imported": False,
            "sealed_descriptor_only": True,
            "all_modules_imported": True,
            "all_module_origins_verified": True,
            "raw_cid_minted": True,
            "raw_cid_validated": True,
            "dag_json_cid_minted": True,
            "dag_json_cid_validated": True,
            "scheduler_or_provider_effect_started": False,
            "stdout_sha256": "sha256:" + ("e" * 64),
            "stderr_sha256": "sha256:" + ("f" * 64),
        },
        "suite": {
            "command": convergence_module._PROGRAM_EXPANSION_TASKS["ASE3-030"][
                "validation"
            ],
            "exit_code": 0,
            "passed": True,
            "passed_count": 37,
            "failed_count": 0,
            "validated_head": parent_head,
            "validated_tree": parent_tree,
            "report_sha256": suite_report,
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-030-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._HERMETIC_ACCEPTANCE_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, frozen, authority


def _reload_receipt_fixture(
    *,
    signing_key: Ed25519PrivateKey | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    private_key = signing_key or Ed25519PrivateKey.generate()
    authority = _review_authority(_reviewer_identity(private_key))
    parent_head = "1" * 40
    parent_tree = "2" * 40
    scheduler_blob = "3" * 40
    daemon_blob = "4" * 40
    receipt_bindings = {
        filename: "sha256:" + hashlib.sha256(filename.encode()).hexdigest()
        for filename in convergence_module.SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES
    }
    stopped: dict[str, object] = {
        "generation_number": 7,
        "head": parent_head,
        "tree": parent_tree,
        "scheduler_path": convergence_module._RELOAD_SCHEDULER_PATH,
        "scheduler_blob": scheduler_blob,
        "scheduler_raw_sha256": "sha256:" + ("5" * 64),
        "daemon_path": convergence_module._RELOAD_DAEMON_PATH,
        "daemon_blob": daemon_blob,
        "daemon_raw_sha256": "sha256:" + ("6" * 64),
        "observed_owned_processes": 0,
        "observed_scoped_provider_containers": 0,
        "observed_inflight_attempts": 0,
    }
    stopped_id = convergence_module._canonical_sha256(stopped)
    stopped = {"generation_id": stopped_id, **stopped}
    target_identity = {
        "source_head": parent_head,
        "source_tree": parent_tree,
        "generation_number": 8,
        "scheduler_blob": scheduler_blob,
        "daemon_blob": daemon_blob,
    }
    target_id = convergence_module._canonical_sha256(target_identity)
    frozen: dict[str, object] = {
        "ready": True,
        "stopped_generation_id": stopped_id,
        "stopped_generation_number": 7,
        "target_generation_id": target_id,
        "scheduler_blob": scheduler_blob,
        "scheduler_raw_sha256": "sha256:" + ("5" * 64),
        "daemon_blob": daemon_blob,
        "daemon_raw_sha256": "sha256:" + ("6" * 64),
    }
    created_at = "2026-08-08T20:00:02Z"
    payload: dict[str, object] = {
        "schema": convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "task": dict(convergence_module._RELOAD_TASK_CONTRACT),
        "acceptance_parent": {
            "head": parent_head,
            "tree": parent_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "acceptance_receipts": receipt_bindings,
            "task_statuses": convergence_module._sequential_task_statuses_after(
                "A023/027"
            ),
        },
        "incident": {
            "attempt2_incident": SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME,
            "attempt2_incident_sha256": (
                convergence_module._ASE3_019_ATTEMPT2_INCIDENT_SHA256
            ),
            "operator_salvage_receipt": OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
            "operator_salvage_receipt_sha256": receipt_bindings[
                OPERATOR_SALVAGE_RECEIPT_019_FILENAME
            ],
            "accepted_control_plane_sha256": convergence_module._canonical_sha256(
                convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE
            ),
        },
        "stopped_generation": stopped,
        "authorization": {
            "source_head": parent_head,
            "source_tree": parent_tree,
            "stopped_generation_id": stopped_id,
            "target_generation_id": target_id,
            "target_generation_number": 8,
            "target_scheduler_blob": scheduler_blob,
            "target_daemon_blob": daemon_blob,
            "lease_namespace": BOARD_NAMESPACE,
            "lease_state_at_authorization": "unclaimed",
            "required_cas_transition": "unclaimed_to_reserved",
            "single_winner_required": True,
            "launch_only_after_l_validates": True,
            "post_launch_birth_receipt_required": True,
            "post_launch_birth_receipt_schema": (
                convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
            ),
            "attempt_counters_unchanged": True,
            "queue_history_unchanged": True,
            "legacy_refill_unchanged": True,
            "runtime_effect_started": False,
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-022-reload-preparation",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._RELOAD_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, frozen, authority


def _standard_sign(
    private_key: Ed25519PrivateKey,
    payload: dict[str, object],
) -> str:
    return base64.b64encode(
        private_key.sign(convergence_module._canonical_json_bytes(payload))
    ).decode("ascii")


def _content_id(payload: dict[str, object]) -> str:
    return convergence_module._canonical_sha256(payload)


def _root_pin_payload(
    *,
    root_identity_did: str,
    base_head: str,
    base_tree: str,
    pinned_at_ms: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "base_head": base_head,
        "base_tree": base_tree,
        "root_identity_did": root_identity_did,
        "pinned_at_ms": pinned_at_ms,
    }
    payload["pin_id"] = _content_id(payload)
    return payload


def _lifecycle_witness_payload(
    *,
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    base_head: str,
    base_tree: str,
    observed_at_ms: int,
) -> tuple[dict[str, object], dict[str, object]]:
    root_did = _reviewer_identity(root_key)
    active_did = _reviewer_identity(active_key)
    profile_path = "local-profile/profile.json"
    anchor_id = hashlib.sha256(profile_path.encode("utf-8")).hexdigest()
    profile: dict[str, object] = {
        "schema": convergence_module.LOCAL_DEV_PROFILE_V5_SCHEMA,
        "repository_cid": "repository:acceptance-fixture",
        "baseline_commit": base_head,
        "capabilities": ["edit", "isolated_worktree", "read", "test"],
        "created_at": observed_at_ms / 1000,
        "profile_id": "local-operator-profile-fixture",
        "identity_did": active_did,
        "revoked": False,
        "lifecycle_generation": 1,
        "lifecycle_anchor_id": anchor_id,
        "lifecycle_root_path": "local-profile-lifecycle-root",
        "effect_bounds": ["edit", "isolated_worktree", "test"],
        "budget_cid": "budget:acceptance-fixture",
        "resource_cid": "resource:acceptance-fixture",
        "route_id": convergence_module._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE[
            "route_id"
        ],
        "reviewer_identity": active_did,
        "reviewer_provider": "local_operator",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_reasoning_effort": "high",
    }
    profile_content_id = _content_id(profile)
    profile_signature = _standard_sign(active_key, profile)

    did_state_unsigned: dict[str, object] = {
        "schema": convergence_module.LOCAL_PROFILE_DID_STATE_V1_SCHEMA,
        "identity_did": active_did,
        "status": "active",
        "profile_path": profile_path,
        "profile_id": profile["profile_id"],
        "profile_content_id": profile_content_id,
        "anchor_id": anchor_id,
        "generation": 1,
        "previous_identity_did": "",
        "updated_at_ns": observed_at_ms * 1_000_000,
        "root_identity_did": root_did,
    }
    did_state: dict[str, object] = {
        **did_state_unsigned,
        "root_signature": _standard_sign(root_key, did_state_unsigned),
    }
    did_state["state_id"] = _content_id(did_state)
    did_state_digest = _content_id(did_state)

    anchor_unsigned: dict[str, object] = {
        "schema": convergence_module.LOCAL_PROFILE_LIFECYCLE_ANCHOR_V3_SCHEMA,
        "anchor_id": anchor_id,
        "generation": 1,
        "status": "active",
        "repository_cid": profile["repository_cid"],
        "profile_id": profile["profile_id"],
        "profile_content_id": profile_content_id,
        "identity_did": active_did,
        "did_state_id": did_state["state_id"],
        "did_status": "active",
        "previous_profile_id": "",
        "previous_profile_content_id": "",
        "previous_identity_did": "",
        "previous_anchor_digest": "",
        "updated_at_ns": observed_at_ms * 1_000_000,
        "root_identity_did": root_did,
    }
    anchor: dict[str, object] = {
        **anchor_unsigned,
        "root_signature": _standard_sign(root_key, anchor_unsigned),
    }
    anchor_digest = _content_id(anchor)

    registry_unsigned: dict[str, object] = {
        "schema": convergence_module.LOCAL_PROFILE_ROOT_REGISTRY_V2_SCHEMA,
        "profile_path": did_state["profile_path"],
        "lifecycle_root": profile["lifecycle_root_path"],
        "root_identity_did": root_did,
    }
    registry = {**registry_unsigned, "registry_id": _content_id(registry_unsigned)}
    body: dict[str, object] = {
        "schema": convergence_module.LOCAL_PROFILE_LIFECYCLE_WITNESS_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "base_head": base_head,
        "base_tree": base_tree,
        "observed_at_ms": observed_at_ms,
        "expires_at_ms": observed_at_ms + 600_000,
        "nonce": "acceptance-lifecycle-witness-nonce",
        "profile": profile,
        "profile_content_id": profile_content_id,
        "profile_signature": profile_signature,
        "anchor": anchor,
        "anchor_digest": anchor_digest,
        "registry": registry,
        "did_state": did_state,
        "did_state_digest": did_state_digest,
        "root_identity_did": root_did,
    }
    active_signature = _standard_sign(active_key, body)
    root_signed = {**body, "active_key_signature": active_signature}
    witness: dict[str, object] = {
        **root_signed,
        "root_signature": _standard_sign(root_key, root_signed),
    }
    witness["witness_id"] = _content_id(witness)
    final_values = {
        "reviewer_identity": active_did,
        "profile_id": profile["profile_id"],
        "profile_content_id": profile_content_id,
        "lifecycle_anchor_id": anchor_id,
        "lifecycle_anchor_digest": anchor_digest,
        "lifecycle_generation": 1,
    }
    return witness, final_values


def _fallback_authorization_v2_payload(
    *,
    active_key: Ed25519PrivateKey,
    witness: dict[str, object],
    witness_sha256: str,
    root_pin: dict[str, object],
    root_pin_sha256: str,
    source_head: str,
    source_tree: str,
    authorized_at_ms: int,
) -> dict[str, object]:
    profile = witness["profile"]
    anchor = witness["anchor"]
    assert isinstance(profile, dict)
    assert isinstance(anchor, dict)
    v1 = _load(DEFAULT_ARTIFACT_ROOT / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME)
    reviewer: dict[str, object] = {
        "identity": profile["identity_did"],
        "provider": "local_operator",
        "profile_id": profile["profile_id"],
        "profile_content_id": witness["profile_content_id"],
        "lifecycle_anchor_id": anchor["anchor_id"],
        "generation": profile["lifecycle_generation"],
        "witness_path": (
            convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
        ),
        "witness_sha256": witness_sha256,
    }
    authority_bounds: dict[str, object] = {
        "repository_cid": profile["repository_cid"],
        "baseline_commit": profile["baseline_commit"],
        "effects": profile["effect_bounds"],
        "budget_cid": profile["budget_cid"],
        "resource_cid": profile["resource_cid"],
        "authority_cid": witness["profile_content_id"],
    }
    source = dict(v1["authorization_source"])
    source["source_head"] = source_head
    source["source_tree"] = source_tree
    review_payload: dict[str, object] = {
        "schema": convergence_module.PROVIDER_FALLBACK_POLICY_REVIEW_V2_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "authorization_source": {
            field: source[field] for field in ("kind", "source_head", "source_tree")
        },
        "route": v1["route"],
        "authority_bounds": authority_bounds,
        "reviewer": reviewer,
        "lifecycle_root_identity_did": root_pin["root_identity_did"],
        "lifecycle_witness_nonce": witness["nonce"],
        "lifecycle_root_pin_path": (
            convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ),
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
        "fallback_implementer_identity": "codex",
    }
    reviewer["signature"] = _standard_sign(active_key, review_payload)
    return {
        "schema": convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "authorization_source": source,
        "route": v1["route"],
        "ownership_contract": {
            "canonical_route_plan_owner": "ipfs_accelerate_py.llm_router",
            "typed_fallback_decision_owner": "ipfs_accelerate_py.llm_router",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed": False,
        },
        "bootstrap_route_guarantees": {
            "explicit_codex_review_conflict_denied": True,
        },
        "reviewer": reviewer,
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_pin["root_identity_did"],
        "lifecycle_witness_nonce": witness["nonce"],
        "lifecycle_root_pin_path": (
            convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ),
        "lifecycle_root_pin_sha256": root_pin_sha256,
        "authorized_at_ms": authorized_at_ms,
    }


def _transition_lifecycle_kwargs(repository: Path) -> dict[str, object]:
    root_path = repository / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    witness_path = repository / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    authorization_path = (
        repository / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    root_payload = _load(root_path)
    witness = _load(witness_path)
    profile = witness["profile"]
    assert isinstance(profile, dict)
    return {
        "lifecycle_root_pin_raw": root_path.read_bytes(),
        "lifecycle_witness_raw": witness_path.read_bytes(),
        "fallback_authorization_raw": authorization_path.read_bytes(),
        "expected_root_identity_did": root_payload["root_identity_did"],
        "expected_final_values": {
            "reviewer_identity": profile["identity_did"],
            "profile_id": profile["profile_id"],
            "profile_content_id": witness["profile_content_id"],
            "lifecycle_anchor_id": profile["lifecycle_anchor_id"],
            "lifecycle_anchor_digest": witness["anchor_digest"],
            "lifecycle_generation": profile["lifecycle_generation"],
        },
    }


_TRANSITION_AUTHORITY_RELATIVE_PATHS = (
    convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH,
    convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH,
    convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
    (
        "data/agent_supervisor/prompt_only_self_improvement_v3/"
        f"convergence/{MANIFEST_FILENAME}"
    ),
)


def _validate_transition_repository(repository: Path) -> tuple[str, ...]:
    artifact_root = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    report = validate_convergence_artifacts(
        artifact_root,
        repo_root=repository,
        check_repository=True,
        taskboard_path=repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    )
    return report.errors


def _initialize_transition_repository(
    tmp_path: Path,
    *,
    preparation_manifest_updates: dict[str, object] | None = None,
    q_manifest_updates: dict[str, object] | None = None,
    root_pin_extra_path: bool = False,
    preparation_extra_path: bool = False,
) -> tuple[Path, str, str]:
    repository = tmp_path / "transition-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Acceptance Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "acceptance@example.invalid"],
        cwd=repository,
        check=True,
    )
    board_path = repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    board_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASKBOARD_PATH, board_path)
    manifest_path = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
        / MANIFEST_FILENAME
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACT_FILENAMES:
        shutil.copy2(DEFAULT_ARTIFACT_ROOT / filename, manifest_path.parent / filename)
    shutil.copy2(DEFAULT_ARTIFACT_ROOT / MANIFEST_FILENAME, manifest_path)
    manifest_path.chmod(0o644)
    if q_manifest_updates is not None:
        q_manifest = _load(manifest_path)
        q_manifest.update(q_manifest_updates)
        _write(manifest_path, q_manifest)
        manifest_path.chmod(0o644)

    # Q is the lifecycle base.  R pins the fixed root, then P adds the witness
    # and authorization that bind R's exact commit and tree.
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "lifecycle base"],
        cwd=repository,
        check=True,
    )
    lifecycle_base_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle_base_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle_base_time_ms = (
        int(
            subprocess.run(
                ["git", "show", "-s", "--format=%ct", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        * 1000
    )
    root_key = Ed25519PrivateKey.generate()
    active_key = Ed25519PrivateKey.generate()
    root_pin_path = (
        repository
        / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    root_pin = _root_pin_payload(
        root_identity_did=_reviewer_identity(root_key),
        base_head=lifecycle_base_head,
        base_tree=lifecycle_base_tree,
        pinned_at_ms=lifecycle_base_time_ms,
    )
    _write(root_pin_path, root_pin)
    root_pin_path.chmod(0o644)
    if root_pin_extra_path:
        (repository / "unexpected-root-pin-path.txt").write_text(
            "unexpected\n",
            encoding="utf-8",
        )
    subprocess.run(
        ["git", "add", "." if root_pin_extra_path else str(root_pin_path)],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "pin lifecycle root"],
        cwd=repository,
        check=True,
    )
    root_pin_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_pin_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_pin_time_ms = (
        int(
            subprocess.run(
                ["git", "show", "-s", "--format=%ct", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        * 1000
    )
    witness, _ = _lifecycle_witness_payload(
        root_key=root_key,
        active_key=active_key,
        base_head=root_pin_head,
        base_tree=root_pin_tree,
        observed_at_ms=root_pin_time_ms,
    )
    witness_path = (
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    _write(witness_path, witness)
    witness_path.chmod(0o644)
    authorization = _fallback_authorization_v2_payload(
        active_key=active_key,
        witness=witness,
        witness_sha256=(
            "sha256:" + hashlib.sha256(witness_path.read_bytes()).hexdigest()
        ),
        root_pin=root_pin,
        root_pin_sha256=(
            "sha256:" + hashlib.sha256(root_pin_path.read_bytes()).hexdigest()
        ),
        source_head=root_pin_head,
        source_tree=root_pin_tree,
        authorized_at_ms=root_pin_time_ms + 1,
    )
    authorization_path = (
        repository
        / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    _write(authorization_path, authorization)
    authorization_path.chmod(0o644)
    _rebind_component_digest(
        manifest_path.parent,
        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
    )
    if preparation_manifest_updates is not None:
        preparation_manifest = _load(manifest_path)
        preparation_manifest.update(preparation_manifest_updates)
        _write(manifest_path, preparation_manifest)
    if preparation_extra_path:
        (repository / "unexpected-preparation-path.txt").write_text(
            "unexpected\n",
            encoding="utf-8",
        )
    manifest_path.chmod(0o644)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "preparation"],
        cwd=repository,
        check=True,
    )
    preparation_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    preparation_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    board_path.write_bytes(
        convergence_module._obsolete_status_only_acceptance_board(
            board_path.read_bytes()
        )
    )
    for relative_path in OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS:
        receipt_path = repository / relative_path
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("{}\n", encoding="utf-8")
    manifest = _load(manifest_path)
    manifest["schema"] = ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
    manifest["created_at"] = "2026-08-08T20:00:01Z"
    manifest["acceptance"] = {
        "phase": "operator_acceptance",
        "preparation_head": preparation_head,
        "preparation_tree": preparation_tree,
        "receipts": {
            filename: "sha256:" + (str(index + 1) * 64)
            for index, filename in enumerate(OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES)
        },
        "tasks": {
            task_id: {
                "canonical_task_cid": expected["canonical_task_cid"],
                "todo_contract_sha256": expected["todo_contract_sha256"],
                "completed_contract_sha256": expected["completed_contract_sha256"],
            }
            for task_id, expected in convergence_module._ACCEPTANCE_TASK_CONTRACTS.items()
        },
        "reload_gate_completed": False,
    }
    _write(manifest_path, manifest)
    manifest_path.chmod(0o644)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "acceptance"],
        cwd=repository,
        check=True,
    )
    return repository, preparation_head, preparation_tree


def _append_reload_transition(
    repository: Path,
    *,
    extra_path: bool = False,
    extra_board_prose: bool = False,
    wrong_acceptance_parent: bool = False,
    executable_receipt: bool = False,
    reopen_acceptance_task: str | None = None,
    include_birth_receipt: bool = False,
) -> tuple[str, str, str]:
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    acceptance_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    board_path = repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    board_path.write_bytes(
        convergence_module._status_only_reload_board(board_path.read_bytes())
    )
    if extra_board_prose:
        board_path.write_bytes(board_path.read_bytes() + b"\nforged reload prose\n")
    if reopen_acceptance_task is not None:
        board_text = board_path.read_text(encoding="utf-8")
        task_start = board_text.index(f"## {reopen_acceptance_task} ")
        status_start = board_text.index("- Status: completed\n", task_start)
        board_path.write_text(
            board_text[:status_start]
            + "- Status: todo\n"
            + board_text[status_start + len("- Status: completed\n") :],
            encoding="utf-8",
        )
    receipt_path = (
        repository / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    )
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt_path.chmod(0o755 if executable_receipt else 0o644)
    if include_birth_receipt:
        birth_path = (
            repository
            / convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH
        )
        birth_path.write_text("{}\n", encoding="utf-8")
        birth_path.chmod(0o644)
    receipt_sha = "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    manifest_path = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
        / MANIFEST_FILENAME
    )
    manifest = _load(manifest_path)
    manifest["schema"] = convergence_module.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
    manifest["created_at"] = "2026-08-08T20:00:03Z"
    manifest["reload"] = {
        "phase": "provider_attempt_daemon_reload",
        "acceptance_head": (
            "0" * 40 if wrong_acceptance_parent else acceptance_head
        ),
        "acceptance_tree": acceptance_tree,
        "receipt": {
            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: receipt_sha,
        },
        "task": dict(convergence_module._RELOAD_TASK_CONTRACT),
        "accepted_task_statuses": {
            task_id: "completed"
            for task_id in convergence_module._ACCEPTANCE_TASK_CONTRACTS
        },
        "reload_gate_completed": True,
        "launch_authorization_only": True,
        "post_launch_birth_receipt_required": True,
        "post_launch_birth_receipt_schema": (
            convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
        ),
    }
    _write(manifest_path, manifest)
    manifest_path.chmod(0o644)
    if extra_path:
        (repository / "unexpected-reload-path.txt").write_text(
            "unexpected\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "reload authorization"],
        cwd=repository,
        check=True,
    )
    reload_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return reload_head, acceptance_head, acceptance_tree


def _portable_recovery_repository(
    tmp_path: Path,
    *,
    include_failed_candidate_parent: bool = False,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    portable = tmp_path / "portable-repository"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(REPO_ROOT), str(portable)],
        check=True,
        capture_output=True,
        text=True,
    )
    taskboard = portable / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    taskboard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASKBOARD_PATH, taskboard)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    failed = recovery["failed_attempt"]
    launch = incident["launch"]
    baseline = _load(root / "current_main_baseline.json")
    seed = baseline["integration_seed"]
    assert isinstance(failed, dict)
    assert isinstance(launch, dict)
    assert isinstance(seed, dict)
    command = [
        "git",
        "-c",
        "user.name=Portable Validation",
        "-c",
        "user.email=portable@example.invalid",
        "commit-tree",
        str(seed["tree"]),
        "-p",
        str(launch["launch_head"]),
    ]
    if include_failed_candidate_parent:
        command.extend(("-p", str(failed["implementation_commit"])))
    command.extend(("-m", "portable recovery descendant"))
    descendant = subprocess.run(
        command,
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "symbolic-ref", "HEAD", "refs/heads/portable-descendant"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "update-ref", "HEAD", descendant],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    return root, portable, taskboard


def test_checked_in_convergence_packet_is_valid_on_integration_checkout() -> None:
    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        repo_root=REPO_ROOT,
        check_repository=True,
    )

    assert report.valid is True, report.errors
    assert report.errors == ()
    assert set(report.checked_artifacts) == {*ARTIFACT_FILENAMES, MANIFEST_FILENAME}
    assert report.integration_seed_commit == "7d70a558e0f54a16a04b3a145fe3d43360cac4c5"


def test_rescue_population_is_complete_and_every_item_has_a_disposition() -> None:
    payload = _load(DEFAULT_ARTIFACT_ROOT / "rescue_artifact_dispositions.json")
    baseline = CurrentMainBaseline.from_dict(
        _load(DEFAULT_ARTIFACT_ROOT / "current_main_baseline.json")
    )
    report = RescueDispositionReport.from_dict(payload)

    assert report.validate(baseline) == ()
    assert len(report.commits) == 36
    assert len(report.files) == 35
    assert {item.disposition for item in (*report.commits, *report.files)} <= {
        "port",
        "rewrite",
        "superseded",
        "discard",
    }
    assert all(
        item.target_tasks
        for item in (*report.commits, *report.files)
        if item.disposition in {"port", "rewrite"}
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "false_completions.ASE3-006",
            "repair_task",
            "ASE3-027",
            "false_completions.ASE3-006",
        ),
        (
            "false_completions.ASE3-018",
            "repair_strict_shard",
            2,
            "false_completions.ASE3-018",
        ),
        (
            "failed_attempt",
            "merge_dispatched",
            True,
            "failed_attempt.merge_dispatched",
        ),
        (
            "disposition",
            "attempt_counter_mutation_authorized",
            True,
            "disposition.attempt_counter_mutation_authorized",
        ),
    ),
)
def test_false_completion_recovery_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_RECOVERY_FILENAME
    payload = _load(path)
    target: object = payload
    for component in section.split("."):
        assert isinstance(target, dict)
        target = target[component]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, FALSE_COMPLETION_RECOVERY_FILENAME)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("filename", "section", "field", "replacement", "error_fragment"),
    (
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            "",
            "task_id",
            "ASE3-018",
            "false_completion_merge_receipt.ASE3-006.task_id",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
            "merge_result.integration_commit_proof",
            "passed",
            False,
            "false_completion_merge_receipt.ASE3-018.integration_commit_proof.passed",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            "merge_result",
            "returncode",
            False,
            "false_completion_merge_receipt.ASE3-006.merge_result.returncode",
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
            "merge_result.todo_update_result.protected_board_postcondition",
            "trusted",
            False,
            (
                "false_completion_merge_receipt.ASE3-018."
                "protected_board_postcondition.trusted"
            ),
        ),
        (
            FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
            (
                "merge_result.todo_update_result.protected_board_postcondition."
                "release_proof"
            ),
            "clean",
            False,
            (
                "false_completion_merge_receipt.ASE3-006."
                "protected_board_postcondition.release_proof.clean"
            ),
        ),
        (
            FAILED_VALIDATION_EVENT_019_FILENAME,
            "",
            "rescue_branch",
            "rescue/forged",
            "failed_validation_event.ASE3-019.event_id",
        ),
        (
            FAILED_VALIDATION_EVENT_019_FILENAME,
            "",
            "merge_dispatched",
            True,
            "failed_validation_event.ASE3-019.merge_dispatched",
        ),
    ),
)
def test_recovery_snapshot_tampering_fails_after_manifest_rebind(
    tmp_path: Path,
    filename: str,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / filename
    payload = _load(path)
    target: object = payload
    for component in filter(None, section.split(".")):
        assert isinstance(target, dict)
        target = target[component]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, filename)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "attempt_accounting",
            "attempt_restoration_authorized",
            True,
            "attempt_restoration_authorized",
        ),
        (
            "terminal_failure",
            "primary_provider_effect_dispatched",
            True,
            "primary_provider_effect_dispatched",
        ),
        (
            "terminal_failure",
            "implementation_runner_dispatched",
            False,
            "implementation_runner_dispatched",
        ),
        (
            "control_plane_provenance",
            "accepted_control_plane_required_for_salvage",
            False,
            "accepted_control_plane_required_for_salvage",
        ),
        (
            "operator_salvage_gate",
            "accepted_control_plane_required",
            False,
            "accepted_control_plane_required",
        ),
        (
            "operator_salvage_gate",
            "required_receipt_fields",
            [
                "schema",
                "created_at",
                "board_namespace",
                "task",
                "incident",
                "authority",
                "source_candidate",
                "salvage_base",
                "implementation",
                "merge",
                "validation",
                "review",
                "denials",
            ],
            "required_receipt_fields",
        ),
        (
            "task",
            "board_status",
            "completed",
            "task.board_status",
        ),
    ),
)
def test_attempt2_incident_tampering_fails_after_manifest_rebind(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    target = payload[section]
    assert isinstance(target, dict)
    target[field] = replacement
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_attempt2_event_semantics_fail_even_after_identity_and_manifest_rebind(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    events = payload["events"]
    event_ids = payload["event_ids"]
    assert isinstance(events, dict)
    assert isinstance(event_ids, list)
    finished = events["implementation_finished"]
    assert isinstance(finished, dict)
    finished["provider_dispatched"] = False
    finished["event_id"] = _recompute_event_id(finished)
    event_ids[2] = finished["event_id"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "event_snapshot.provider_dispatched" in error for error in report.errors
    )


@pytest.mark.parametrize(
    ("event_name", "event_index", "field", "replacement", "error_fragment"),
    (
        (
            "prior_attempt_seeded",
            0,
            "applied",
            False,
            "events.prior_attempt_seeded.applied",
        ),
        (
            "implementation_started",
            1,
            "branch",
            "implementation/forged",
            "events.implementation_started.branch",
        ),
        (
            "implementation_shutdown_reconciled",
            3,
            "reconciled",
            False,
            "events.implementation_shutdown_reconciled.reconciled",
        ),
    ),
)
def test_attempt2_event_chain_semantics_fail_after_event_id_rebind(
    tmp_path: Path,
    event_name: str,
    event_index: int,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    events = payload["events"]
    event_ids = payload["event_ids"]
    assert isinstance(events, dict)
    assert isinstance(event_ids, list)
    event = events[event_name]
    assert isinstance(event, dict)
    event[field] = replacement
    event["event_id"] = _recompute_event_id(event)
    event_ids[event_index] = event["event_id"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_attempt2_event_bundle_order_is_exact_after_manifest_rebind(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME
    payload = _load(path)
    event_order = payload["event_order"]
    assert isinstance(event_order, list)
    event_order[0], event_order[1] = event_order[1], event_order[0]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("event_snapshot.event_order" in error for error in report.errors)


def test_attempt2_log_tampering_fails_after_manifest_rebind(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "agent implementation route binding fields are invalid",
            "forged terminal success",
            1,
        ),
        encoding="utf-8",
    )
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("log_snapshot" in error for error in report.errors)


def test_attempt2_log_uses_a_dedicated_eight_kibibyte_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME
    path.write_bytes(b"x" * (8 * 1024 + 1))

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("8192-byte evidence snapshot bound" in error for error in report.errors)


def test_recovery_snapshot_symlink_is_rejected_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    path.unlink()
    path.symlink_to(
        DEFAULT_ARTIFACT_ROOT / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    )

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(path.name in error for error in report.errors)


def test_evidence_snapshot_hardlink_is_rejected_before_parsing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    backing = root / "hardlink-backing.json"
    shutil.copy2(path, backing)
    path.unlink()
    os.link(backing, path)

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("single-link evidence file" in error for error in report.errors)


def test_evidence_snapshot_size_bound_fails_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME
    path.write_bytes(b" " * (MAX_EVIDENCE_SNAPSHOT_BYTES + 1))

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("evidence snapshot bound" in error for error in report.errors)


def test_evidence_snapshot_descriptor_instability_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    real_fstat = os.fstat
    regular_fstat_calls = 0

    def unstable_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal regular_fstat_calls
        observed = real_fstat(descriptor)
        if not convergence_module.stat.S_ISREG(observed.st_mode):
            return observed
        regular_fstat_calls += 1
        if regular_fstat_calls != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(convergence_module.os, "fstat", unstable_fstat)
    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any("changed during bounded read" in error for error in report.errors)


def test_recovery_shard_fields_name_the_repair_and_retry_tasks() -> None:
    recovery = _load(DEFAULT_ARTIFACT_ROOT / FALSE_COMPLETION_RECOVERY_FILENAME)
    completions = recovery["false_completions"]
    failed = recovery["failed_attempt"]
    assert isinstance(completions, dict)
    assert isinstance(failed, dict)
    assert completions["ASE3-006"]["repair_strict_shard"] == 2
    assert completions["ASE3-018"]["repair_strict_shard"] == 0
    assert failed["retry_strict_shard"] == 1
    assert all("strict_shard" not in item for item in completions.values())
    assert "strict_shard" not in failed


def test_component_tampering_fails_closed_before_repository_checks(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    baseline_path = root / "current_main_baseline.json"
    baseline = _load(baseline_path)
    original = baseline["original_checkout"]
    assert isinstance(original, dict)
    original["dirty_entry_count"] = 0
    _write(baseline_path, baseline)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("dirty_entry_count" in error for error in report.errors)
    assert any("digest mismatch" in error for error in report.errors)


def test_rebound_historical_state_still_cannot_claim_v3_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "historical_state_contradictions.json"
    payload = _load(path)
    payload["authority"] = "completion-authority"
    payload["v3_completion_credit"] = True
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("authority: must be evidence-only" in error for error in report.errors)
    assert any("v3_completion_credit: must be false" in error for error in report.errors)


def test_rebound_post_wave3_residual_mapping_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / POST_WAVE3_RESIDUAL_FILENAME
    payload = _load(path)
    residuals = payload["residuals"]
    assert isinstance(residuals, list)
    record = next(
        item
        for item in residuals
        if isinstance(item, dict)
        and item.get("gap_id") == "trusted-context-canonical-composition"
    )
    record["target_task"] = "ASE3-019"
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(
        "trusted-context-canonical-composition.target_task: expected ASE3-018"
        in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "provider_incident",
            "attempt_consumed",
            True,
            "provider_incident.attempt_consumed: expected False",
        ),
        (
            "provider_incident",
            "fallback_dispatched",
            True,
            "provider_incident.fallback_dispatched: expected False",
        ),
        (
            "disposition",
            "completion_authority",
            True,
            "disposition.completion_authority: expected False",
        ),
        (
            "disposition",
            "gate_task",
            "ASE3-009",
            "disposition.gate_task: expected 'ASE3-008'",
        ),
    ),
)
def test_rebound_post_wave3_authority_and_provider_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / POST_WAVE3_RESIDUAL_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "authorization_source",
            "source_head",
            "0" * 40,
            "authorization_source.source_head: expected",
        ),
        (
            "authorization_source",
            "prospective_only",
            False,
            "authorization_source.prospective_only: expected True",
        ),
        (
            "route",
            "route_id",
            "global-ambient-route",
            (
                "route.route_id: expected 'agent-supervisor-prompt-v3-grok45-"
                "terra56-high-auth-or-hard-quota-v1'"
            ),
        ),
        (
            "route",
            "fallback_reasoning_effort",
            "medium",
            "route.fallback_reasoning_effort: expected 'high'",
        ),
        (
            "route",
            "allowed_trigger_classes",
            ["grok_authentication_unavailable", "rate_limit"],
            "route.allowed_trigger_classes: expected",
        ),
        (
            "ownership_contract",
            "canonical_route_plan_owner",
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
            (
                "ownership_contract.canonical_route_plan_owner: expected "
                "'ipfs_accelerate_py.llm_router'"
            ),
        ),
        (
            "ownership_contract",
            "typed_fallback_decision_owner",
            "implementation_daemon",
            (
                "ownership_contract.typed_fallback_decision_owner: expected "
                "'ipfs_accelerate_py.llm_router'"
            ),
        ),
        (
            "ownership_contract",
            "route_plan_and_decision_exports_required_before_bootstrap_dispatch",
            False,
            (
                "route_plan_and_decision_exports_required_before_bootstrap_dispatch: "
                "expected True"
            ),
        ),
        (
            "ownership_contract",
            "route_authority_binding_fields",
            ["board_namespace", "authorization_artifact_sha256"],
            "ownership_contract.route_authority_binding_fields: expected",
        ),
        (
            "ownership_contract",
            "verified_authority_binding_must_reach_terminal_outcome_and_daemon_accounting",
            False,
            (
                "verified_authority_binding_must_reach_terminal_outcome_and_daemon_accounting: "
                "expected True"
            ),
        ),
        (
            "ownership_contract",
            "ambient_six_field_route_profile_alone_authorizes_fallback",
            True,
            (
                "ambient_six_field_route_profile_alone_authorizes_fallback: expected "
                "False"
            ),
        ),
        (
            "ownership_contract",
            "runner_role",
            "route_policy_and_failure_classifier",
            (
                "ownership_contract.runner_role: expected "
                "'isolation_process_effect_and_terminal_outcome_emitter'"
            ),
        ),
        (
            "ownership_contract",
            "daemon_role",
            "provider_failure_reclassification",
            "ownership_contract.daemon_role: expected 'task_retry_accounting_only'",
        ),
        (
            "ownership_contract",
            "scheduler_role",
            "route_policy_owner",
            "ownership_contract.scheduler_role: expected 'route_profile_input_only'",
        ),
        (
            "ownership_contract",
            "duplicate_route_policy_or_failure_classification_outside_router_allowed",
            True,
            (
                "duplicate_route_policy_or_failure_classification_outside_router_allowed: "
                "expected False"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "fallback_dispatch_scope",
            "once_per_host_forever",
            (
                "bootstrap_route_guarantees.fallback_dispatch_scope: expected "
                "'once_per_runner_same_daemon_attempt'"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "direct_auth_signal_allowlist",
            ["not signed in", "not authenticated", "forbidden"],
            "bootstrap_route_guarantees.direct_auth_signal_allowlist: expected",
        ),
        (
            "bootstrap_route_guarantees",
            "ambiguous_direct_auth_signals_denied",
            ["401", "403"],
            "ambiguous_direct_auth_signals_denied: expected",
        ),
        (
            "bootstrap_route_guarantees",
            "ambiguous_signal_may_continue_only_as_independently_confirmed_hard_quota",
            False,
            (
                "ambiguous_signal_may_continue_only_as_independently_confirmed_hard_quota: "
                "expected True"
            ),
        ),
        (
            "bootstrap_route_guarantees",
            "hard_quota_independent_confirmation_required",
            False,
            "hard_quota_independent_confirmation_required: expected True",
        ),
        (
            "bootstrap_route_guarantees",
            "explicit_codex_review_conflict_denied",
            False,
            "explicit_codex_review_conflict_denied: expected True",
        ),
        (
            "bootstrap_route_guarantees",
            "durable_cross_process_restart_reservation_present",
            True,
            "durable_cross_process_restart_reservation_present: expected False",
        ),
        (
            "bootstrap_route_guarantees",
            "full_signed_field_equality_present",
            True,
            "full_signed_field_equality_present: expected False",
        ),
        (
            "ase3_019_completion_requirements",
            "durable_cross_process_restart_once_only_cas_required",
            False,
            "durable_cross_process_restart_once_only_cas_required: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "auth_signal_policy_expansion_requires_signed_typed_policy",
            False,
            "auth_signal_policy_expansion_requires_signed_typed_policy: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "canonical_route_plan_and_typed_decision_must_remain_router_owned",
            False,
            (
                "canonical_route_plan_and_typed_decision_must_remain_router_owned: "
                "expected True"
            ),
        ),
        (
            "ase3_019_completion_requirements",
            "provider_capacity_attempt_restoration_must_remain_denied",
            False,
            (
                "provider_capacity_attempt_restoration_must_remain_denied: expected "
                "True"
            ),
        ),
        (
            "ase3_019_completion_requirements",
            "signed_reviewer_identity_and_provider_required",
            False,
            "signed_reviewer_identity_and_provider_required: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "fallback_implementer_and_reviewer_must_differ",
            False,
            "fallback_implementer_and_reviewer_must_differ: expected True",
        ),
        (
            "ase3_019_completion_requirements",
            "signed_equality_fields",
            ["invocation", "task", "prompt", "scope", "budget", "authority"],
            "ase3_019_completion_requirements.signed_equality_fields: expected",
        ),
        (
            "external_docker_boundary",
            "image_id",
            "sha256:" + "0" * 64,
            "external_docker_boundary.image_id: expected",
        ),
        (
            "external_docker_boundary",
            "workspace_is_only_writable_bind_mount",
            False,
            "workspace_is_only_writable_bind_mount: expected True",
        ),
        (
            "denials",
            "arbitrary_error_fallback_allowed",
            True,
            "denials.arbitrary_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "rate_limit_fallback_allowed",
            True,
            "denials.rate_limit_fallback_allowed: expected False",
        ),
        (
            "denials",
            "transport_error_fallback_allowed",
            True,
            "denials.transport_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "invalid_request_fallback_allowed",
            True,
            "denials.invalid_request_fallback_allowed: expected False",
        ),
        (
            "denials",
            "unknown_error_fallback_allowed",
            True,
            "denials.unknown_error_fallback_allowed: expected False",
        ),
        (
            "denials",
            "post_effect_fallback_allowed",
            True,
            "denials.post_effect_fallback_allowed: expected False",
        ),
        (
            "denials",
            "workspace_changed_before_fallback_allowed",
            True,
            "workspace_changed_before_fallback_allowed: expected False",
        ),
        (
            "denials",
            "attempt_counter_mutation_authorized",
            True,
            "attempt_counter_mutation_authorized: expected False",
        ),
        (
            "denials",
            "provider_capacity_attempt_restoration_allowed",
            True,
            "provider_capacity_attempt_restoration_allowed: expected False",
        ),
        (
            "denials",
            "legacy_objective_refill_authorized",
            True,
            "legacy_objective_refill_authorized: expected False",
        ),
        (
            "denials",
            "legacy_codebase_refill_authorized",
            True,
            "legacy_codebase_refill_authorized: expected False",
        ),
        (
            "historical_evidence",
            "post_wave3_residual_report_is_immutable",
            False,
            "post_wave3_residual_report_is_immutable: expected True",
        ),
    ),
)
def test_rebound_provider_fallback_authorization_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_ase3_019_cannot_downgrade_terra_reasoning_or_auth_fallback(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "exactly one concurrent or restarted worker automatically admits a "
        "matching pre-effect Codex `gpt-5.6-terra` fallback at `high` reasoning"
    )
    replacement = (
        "exactly one concurrent or restarted worker requires reauthentication "
        "before a Codex `gpt-5.6-terra` fallback at `medium` reasoning"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.acceptance: exact automatic "
        "auth/quota fallback contract required"
    ) in report.errors


def test_ase3_019_cannot_move_route_policy_outside_llm_router(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "Export an immutable canonical implementation route plan and typed "
        "fallback decision from `ipfs_accelerate_py.llm_router` as the sole "
        "provider-policy source"
    )
    replacement = (
        "Let the runner and daemon independently choose implementation routes "
        "and fallback decisions"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.effects: exact automatic "
        "auth/quota fallback contract required"
    ) in report.errors


def test_ase3_019_must_name_llm_router_and_its_dedicated_route_test(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "- Outputs: ipfs_accelerate_py/llm_router.py, "
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py"
    )
    replacement = (
        "- Outputs: "
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert (
        "provider_fallback_task_contract.ASE3-019.outputs: exact "
        "llm_router-owned route surface required"
    ) in report.errors


@pytest.mark.parametrize(
    ("needle", "replacement", "error_fragment"),
    (
        (
            "- Repairs task: ASE3-006\n",
            "- Repairs task: ASE3-018\n",
            "ASE3-023.repairs_task",
        ),
        (
            (
                "- Is schedulable: true\n- Review only: false\n- Priority: P0\n"
                "- Track: ambient-inference-production-repair\n"
            ),
            (
                "- Is schedulable: false\n- Review only: false\n- Priority: P0\n"
                "- Track: ambient-inference-production-repair\n"
            ),
            "ASE3-027.is_schedulable",
        ),
        (
            "- Depends on: ASE3-006, ASE3-018, ASE3-019, ASE3-023, ASE3-027\n",
            "- Depends on: ASE3-006, ASE3-018, ASE3-019\n",
            "ASE3-022.depends_on",
        ),
        (
            (
                "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
                "and once-only fallback\n"
            ),
            "## ASE3-019 Changed identity\n",
            "provider_fallback_task_contract.ASE3-019.title",
        ),
        (
            (
                "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
                "and once-only fallback\n\n- Status: todo\n"
            ),
            (
                "## ASE3-019 Seal signed provider authority, authentication lifecycle, "
                "and once-only fallback\n\n- Status: completed\n"
            ),
            "provider_fallback_task_contract.ASE3-019.contract_sha256",
        ),
        (
            "Configured-board production launch consumes the compiled active plan",
            "Configured-board production launch may ignore the compiled active plan",
            "false_completion_repair_tasks.ASE3-023.contract_sha256",
        ),
        (
            "call the existing canonical target, state/run, profile, objective/task-source",
            "optionally bypass the canonical target, state/run, profile, objective/task-source",
            "false_completion_repair_tasks.ASE3-027.contract_sha256",
        ),
    ),
)
def test_false_completion_repair_task_contract_fails_closed(
    tmp_path: Path,
    needle: str,
    replacement: str,
    error_fragment: str,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_reload_gate_rejects_a_removed_blocked_reason(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "- Blocked reason: provider-attempt daemon reload boundary not yet accepted\n"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, "", 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "ASE3-022.blocked_reason: expected 'provider-attempt daemon reload "
        "boundary not yet accepted'" in error
        for error in report.errors
    )


def test_reload_gate_rejects_a_removed_ase3_021_dependency(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019, ASE3-022, "
        "ASE3-024, ASE3-025\n"
    )
    replacement = (
        "- Depends on: ASE3-004, ASE3-006, ASE3-007, ASE3-019, ASE3-024, "
        "ASE3-025\n"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert "provider_attempt_reload_gate.ASE3-021.depends_on: missing ASE3-022" in (
        report.errors
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    (
        (
            "goal",
            "- Goal id: ASE3-G055\n- Outputs: ",
            "ASE3-022.goal_id: must be absent",
        ),
        (
            "outputs",
            "- Outputs: data/forged-reload-receipt.json",
            "ASE3-022.outputs: expected only",
        ),
        (
            "predicted",
            "- Predicted files: data/forged-reload-receipt.json",
            "ASE3-022.predicted_files: expected only",
        ),
    ),
)
def test_reload_gate_rejects_goal_enrollment_and_receipt_redirects(
    tmp_path: Path,
    field: str,
    replacement: str,
    error_fragment: str,
) -> None:
    taskboard_path = tmp_path / f"prompt-v3-{field}.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    if field == "goal":
        needle = f"- Outputs: {PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        replacement += PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    elif field == "outputs":
        needle = f"- Outputs: {PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
    else:
        needle = (
            "- Predicted files: "
            f"{PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH}"
        )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_reload_gate_completion_requires_atomic_l_transition(
    tmp_path: Path,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "## ASE3-022 Accept the provider-attempt daemon reload boundary\n\n"
        "- Status: blocked\n"
    )
    replacement = (
        "## ASE3-022 Accept the provider-attempt daemon reload boundary\n\n"
        "- Status: completed\n"
    )
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "provider_attempt_reload_gate.ASE3-022.status: expected blocked" in error
        for error in report.errors
    )


def test_reload_receipt_path_is_forbidden_before_atomic_l_transition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    receipt.symlink_to(tmp_path / "missing-reload-receipt-target.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "provider_attempt_reload_gate.receipt: forbidden before reload phase" in error
        for error in report.errors
    )


@pytest.mark.parametrize("receipt_kind", ("regular", "dangling-symlink"))
def legacy_operator_salvage_receipt_path_is_reserved_during_c1(
    tmp_path: Path,
    receipt_kind: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / OPERATOR_SALVAGE_RECEIPT_019_FILENAME
    if receipt_kind == "regular":
        receipt.write_text("{}\n", encoding="utf-8")
    else:
        receipt.symlink_to(tmp_path / "missing-salvage-receipt-target.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        OPERATOR_SALVAGE_RECEIPT_019_FILENAME in error
        and "partial population forbidden" in error
        for error in report.errors
    )


def test_reload_gate_c1_operator_salvage_contract_is_exact(tmp_path: Path) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "mandatory accepted-control-plane provenance"
    assert text.count(needle) == 1
    taskboard_path.write_text(
        text.replace(needle, "optional ambient control-plane provenance", 1),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert any(
        "ASE3-022.contract_sha256" in error for error in report.errors
    )


def test_all_future_acceptance_and_reload_paths_are_protected_and_absent() -> None:
    config = _load(CONFIG_PATH)
    protected_paths = config["protected_paths"]
    assert isinstance(protected_paths, list)
    expected = {
        *convergence_module.SEQUENTIAL_RESERVED_ARTIFACT_INTRODUCTION_PHASE,
        convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,
    }
    assert expected <= set(protected_paths)
    assert all(
        not (REPO_ROOT / relative_path).exists()
        and not (REPO_ROOT / relative_path).is_symlink()
        for relative_path in expected
    )
    assert protected_paths == list(convergence_module._PROTECTED_PATHS)


@pytest.mark.parametrize("artifact_kind", ("regular", "dangling-symlink"))
def test_dormant_phase_rejects_regular_or_symlink_future_receipt_presence(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_FILENAME
    if artifact_kind == "regular":
        receipt.write_text("{}\n", encoding="utf-8")
    else:
        receipt.symlink_to(tmp_path / "missing-native-acceptance-receipt.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "no exact sequential phase population matches" in error
        for error in report.errors
    )


def test_all_present_protected_files_have_stable_authority_and_git_modes() -> None:
    assert convergence_module._validate_protected_file_authority(
        repo_root=REPO_ROOT,
        phase="preparation",
    ) == []


def test_protected_authority_rejects_mode_and_git_symlink_for_any_protected_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_relative = convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH.as_posix()
    target = (REPO_ROOT / target_relative).absolute()
    real_reader = convergence_module._read_regular_snapshot
    real_git = convergence_module._git

    def unsafe_mode_reader(
        path: Path,
        *,
        maximum_bytes: int = MAX_EVIDENCE_SNAPSHOT_BYTES,
    ) -> SimpleNamespace:
        snapshot = real_reader(Path(path), maximum_bytes=maximum_bytes)
        if snapshot.path != target:
            return snapshot
        return SimpleNamespace(
            raw=snapshot.raw,
            path=snapshot.path,
            uid=snapshot.uid,
            mode=snapshot.mode | 0o020,
        )

    def forged_git_mode(
        repo_root: Path,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        if args == ("ls-tree", "HEAD", "--", target_relative):
            return subprocess.CompletedProcess(
                ["git", *args],
                0,
                f"120000 blob {'0' * 40}\t{target_relative}\n",
                "",
            )
        return real_git(repo_root, *args)

    monkeypatch.setattr(
        convergence_module,
        "_read_regular_snapshot",
        unsafe_mode_reader,
    )
    monkeypatch.setattr(convergence_module, "_git", forged_git_mode)

    errors = convergence_module._validate_protected_file_authority(
        repo_root=REPO_ROOT,
        phase="preparation",
    )

    assert any("group-or-other writable" in error for error in errors)
    assert any("exact regular Git mode required" in error for error in errors)


@pytest.mark.parametrize(
    "filename",
    OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES,
)
@pytest.mark.parametrize("receipt_kind", ("regular", "dangling-symlink"))
def legacy_every_premature_operator_acceptance_receipt_fails_preparation(
    tmp_path: Path,
    filename: str,
    receipt_kind: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt = root / filename
    if receipt_kind == "regular":
        receipt.write_text("{}\n", encoding="utf-8")
    else:
        receipt.symlink_to(tmp_path / "missing-acceptance-receipt.json")

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        filename in error and "partial population forbidden" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    "task_id",
    ("ASE3-019", "ASE3-030", "ASE3-023", "ASE3-027"),
)
def legacy_preparation_rejects_each_premature_completed_status(
    tmp_path: Path,
    task_id: str,
) -> None:
    taskboard_path = tmp_path / "prompt-v3.todo.md"
    tasks = convergence_module._load_taskboard_metadata(TASKBOARD_PATH)
    title = tasks[task_id][convergence_module._TASK_TITLE_KEY]
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = f"## {task_id} {title}\n\n- Status: todo\n"
    replacement = f"## {task_id} {title}\n\n- Status: completed\n"
    assert text.count(needle) == 1
    taskboard_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard_path,
    )

    assert report.valid is False
    assert f"operator_acceptance.phase.{task_id}.status" in "\n".join(report.errors)


def test_completed_contract_hashes_are_exact_status_only_variants() -> None:
    before = convergence_module._load_taskboard_metadata(TASKBOARD_PATH)
    after_raw = convergence_module._obsolete_status_only_acceptance_board(
        TASKBOARD_PATH.read_bytes()
    )
    after = convergence_module._parse_taskboard_metadata(after_raw.decode("utf-8"))
    for task_id, expected in convergence_module._ACCEPTANCE_TASK_CONTRACTS.items():
        before_task = before[task_id]
        after_task = after[task_id]
        assert before_task["status"] == "todo"
        assert after_task["status"] == "completed"
        assert {
            key: value for key, value in before_task.items() if key != "status"
        } == {key: value for key, value in after_task.items() if key != "status"}
        assert (
            convergence_module._task_contract_sha256(before_task)
            == expected["todo_contract_sha256"]
        )
        assert (
            convergence_module._task_contract_sha256(after_task)
            == expected["completed_contract_sha256"]
        )
        assert (
            convergence_module._canonical_task_cid_from_metadata(after_task)
            == expected["canonical_task_cid"]
        )


def test_acceptance_receipt_loader_is_bounded_single_link_and_duplicate_safe(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME
    payload = {
        key: {}
        for key in convergence_module._OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS
    }
    payload.update(
        {
            "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
            "created_at": "2026-08-08T20:00:00Z",
            "board_namespace": BOARD_NAMESPACE,
            "task": {"task_id": "ASE3-023"},
        }
    )
    _write(receipt_path, payload)
    snapshot = load_operator_acceptance_receipt(
        receipt_path,
        task_id="ASE3-023",
    )
    assert snapshot.filename == OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME
    assert snapshot.sha256.startswith("sha256:")

    duplicate = json.dumps(payload).replace(
        '"schema":',
        '"schema": "duplicate", "schema":',
        1,
    )
    receipt_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key: schema"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-023")

    receipt_path.write_bytes(b" " * (MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES + 1))
    with pytest.raises(ValueError, match="evidence snapshot bound"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-023")

    backing = tmp_path / "receipt-backing.json"
    _write(backing, payload)
    receipt_path.unlink()
    os.link(backing, receipt_path)
    with pytest.raises(ValueError, match="single-link evidence file"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-023")


def test_acceptance_receipt_loader_rejects_symlink_and_descriptor_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    receipt_path.symlink_to(target)
    with pytest.raises(ValueError, match="regular nonsymlink"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-027")

    receipt_path.unlink()
    payload = {
        key: {}
        for key in convergence_module._OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS
    }
    payload.update(
        {
            "schema": OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
            "task": {"task_id": "ASE3-027"},
        }
    )
    _write(receipt_path, payload)
    real_fstat = os.fstat
    regular_calls = 0

    def growing_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal regular_calls
        observed = real_fstat(descriptor)
        if not convergence_module.stat.S_ISREG(observed.st_mode):
            return observed
        regular_calls += 1
        if regular_calls != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(convergence_module.os, "fstat", growing_fstat)
    with pytest.raises(ValueError, match="changed during bounded read"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-027")


def test_acceptance_receipt_loader_rejects_path_swap_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME
    _write(receipt_path, _minimal_operator_receipt("ASE3-023"))
    replacement = tmp_path / "replacement.json"
    _write(replacement, _minimal_operator_receipt("ASE3-023"))
    real_read = os.read
    swapped = False

    def swapping_read(descriptor: int, amount: int) -> bytes:
        nonlocal swapped
        result = real_read(descriptor, amount)
        if result and not swapped:
            swapped = True
            receipt_path.unlink()
            replacement.rename(receipt_path)
        return result

    monkeypatch.setattr(convergence_module.os, "read", swapping_read)
    with pytest.raises(ValueError, match="changed during bounded read"):
        load_operator_acceptance_receipt(receipt_path, task_id="ASE3-023")


def test_review_signature_covers_the_entire_receipt_except_its_signature() -> None:
    private_key = Ed25519PrivateKey.generate()
    reviewer = _reviewer_identity(private_key)
    authority = _review_authority(reviewer)
    created_at = "2026-08-08T20:00:00Z"
    payload: dict[str, object] = {
        "schema": "test-receipt@1",
        "created_at": created_at,
        "bound_value": {"must_remain": "exact"},
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:implementer",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
    }
    _sign_operator_receipt(payload, private_key)
    assert validate_operator_acceptance_signature(
        payload,
        expected_authority=authority,
    ) == ()

    bound_value = payload["bound_value"]
    assert isinstance(bound_value, dict)
    bound_value["must_remain"] = "forged"
    assert any(
        "cryptographic verification failed" in error
        for error in validate_operator_acceptance_signature(
            payload,
            expected_authority=authority,
        )
    )


@pytest.mark.parametrize("receipt_kind", ("A", "L"))
@pytest.mark.parametrize(
    ("signed_at", "valid"),
    (
        ("2026-08-08T18:59:59Z", False),
        ("2026-08-08T19:00:00Z", True),
        ("2026-08-08T21:00:00Z", True),
        ("2026-08-08T21:00:01Z", False),
    ),
)
def test_sequential_acceptance_and_l_receipt_times_are_inside_root_signed_witness_interval(
    receipt_kind: str,
    signed_at: str,
    valid: bool,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    if receipt_kind == "A":
        payload, frozen, authority = _hermetic_acceptance_receipt_fixture(
            signing_key=private_key
        )
    else:
        payload, frozen, authority = _reload_receipt_fixture(
            signing_key=private_key
        )
    if signed_at == "2026-08-08T19:00:00Z":
        authority["fallback_authorized_at_ms"] = authority[
            "lifecycle_witness_observed_at_ms"
        ]
    payload["created_at"] = signed_at
    review = payload["review"]
    assert isinstance(review, dict)
    review["signed_at"] = signed_at
    _sign_operator_receipt(payload, private_key)

    if receipt_kind == "A":
        errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
            payload,
            lifecycle_authority=authority,
            frozen_values=frozen,
        )
    else:
        errors = convergence_module.validate_provider_attempt_reload_receipt(
            payload,
            lifecycle_authority=authority,
            frozen_values=frozen,
            accepted_control_plane=(
                convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE
            ),
        )

    if valid:
        assert errors == ()
    else:
        assert any("signed_at: outside witness validity" in error for error in errors)


@pytest.mark.parametrize("receipt_kind", ("A", "L"))
def test_sequential_acceptance_and_l_receipts_cannot_predate_fallback_authorization(
    receipt_kind: str,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    if receipt_kind == "A":
        payload, frozen, authority = _hermetic_acceptance_receipt_fixture(
            signing_key=private_key
        )
    else:
        payload, frozen, authority = _reload_receipt_fixture(
            signing_key=private_key
        )
    authority["fallback_authorized_at_ms"] = 1_786_219_203_000

    if receipt_kind == "A":
        errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
            payload,
            lifecycle_authority=authority,
            frozen_values=frozen,
        )
    else:
        errors = convergence_module.validate_provider_attempt_reload_receipt(
            payload,
            lifecycle_authority=authority,
            frozen_values=frozen,
            accepted_control_plane=(
                convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE
            ),
        )

    assert any("signed_at: predates fallback authorization" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    (
        ("reviewer_provider", "codex", "Codex/OpenAI review is denied"),
        ("reviewer_provider", "openai", "Codex/OpenAI review is denied"),
        ("lifecycle_witness_nonce", "forged", "authority.lifecycle_witness_nonce"),
        ("implementer_identity", "__reviewer__", "self-review is denied"),
    ),
)
def test_review_policy_denies_codex_openai_revoked_and_self_review(
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    payload, reviewer, authority = _operator_repair_receipt_027()
    review = payload["review"]
    assert isinstance(review, dict)
    review[field] = reviewer if replacement == "__reviewer__" else replacement
    errors = validate_operator_acceptance_signature(
        payload,
        expected_authority=authority,
    )
    assert any(error_fragment in error for error in errors)


def test_ase3_027_final_blob_freeze_is_sealed_and_validates_implementation(
) -> None:
    final_values = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-027"
    ]
    assert final_values["ready"] is True
    assert final_values["pending"] is None
    assert final_values["validation_passed_count"] == 174
    assert len(final_values["generations"]) == 2
    assert len(final_values["final_blobs"]) == 5

    payload, _, authority = _operator_repair_receipt_027()
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-027",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert not any(
        "final product values are not populated" in error for error in errors
    )
    # Implementation topology and final blobs are sealed; remaining errors (if
    # any) must not be the pre-freeze sentinel gate.
    impl_errors = [
        error
        for error in errors
        if "implementation" in error and "final product values" in error
    ]
    assert impl_errors == []

    recovery = payload["recovery"]
    assert isinstance(recovery, dict)
    recovery["ambient_override"] = True
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-027",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert any(
        "ASE3-027.recovery: exact key population required" in error
        for error in errors
    )


def test_ase3_027_final_blob_tamper_fails_closed_after_freeze() -> None:
    payload, _, authority = _operator_repair_receipt_027()
    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    blobs = dict(implementation["final_blobs"])
    first_path = next(iter(blobs))
    blobs[first_path] = "0" * 40
    implementation["final_blobs"] = blobs
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-027",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert any("final_blobs" in error for error in errors)


def test_ase3_023_final_blob_freeze_is_sealed_and_validates_implementation(
) -> None:
    final_values = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-023"
    ]
    assert final_values["ready"] is True
    assert final_values["pending"] is None
    assert final_values["validation_passed_count"] == 110
    assert len(final_values["generations"]) == 3
    assert len(final_values["final_blobs"]) == 7
    assert [generation["role"] for generation in final_values["generations"]] == [
        "product-salvage",
        "capsule-identity",
        "recovery-barrier",
    ]

    payload, _, authority = _operator_repair_receipt_023()
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-023",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert not any(
        "final product values are not populated" in error for error in errors
    )
    impl_errors = [
        error
        for error in errors
        if "implementation" in error and "final product values" in error
    ]
    assert impl_errors == []

    recovery = payload["recovery"]
    assert isinstance(recovery, dict)
    recovery["ambient_override"] = True
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-023",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert any(
        "ASE3-023.recovery: exact key population required" in error
        for error in errors
    )


def test_ase3_023_final_blob_tamper_fails_closed_after_freeze() -> None:
    payload, _, authority = _operator_repair_receipt_023()
    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    blobs = dict(implementation["final_blobs"])
    first_path = next(iter(blobs))
    blobs[first_path] = "0" * 40
    implementation["final_blobs"] = blobs
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-023",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert any("final_blobs" in error for error in errors)



def test_product_generation_v1_triples_are_sealed_for_pre_q_products() -> None:
    values = convergence_module._PRODUCT_GENERATION_FINAL_VALUES
    assert values["ASE3-019"]["ready"] is False
    for task_id, expected_count in (
        ("ASE3-023", 3),
        ("ASE3-027", 2),
        ("ASE3-030", 2),
        ("ASE3-031", 1),
        ("ASE3-032", 1),
    ):
        final = values[task_id]
        assert final["ready"] is True
        assert final["pending"] is None
        assert final["schema"].endswith("prompt-v3-product-generation@1")
        assert len(final["generations"]) == expected_count
        for generation in final["generations"]:
            assert generation["source_commit"] != generation["replay_commit"]
            assert (
                generation["source_patch_sha256"]
                == generation["replay_patch_sha256"]
                == generation["integrated_patch_sha256"]
            )
            # Source/replay remain non-ancestors; integrated is on main.
            assert (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(REPO_ROOT),
                        "merge-base",
                        "--is-ancestor",
                        generation["source_commit"],
                        "HEAD",
                    ],
                    check=False,
                ).returncode
                != 0
            )
            assert (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(REPO_ROOT),
                        "merge-base",
                        "--is-ancestor",
                        generation["replay_commit"],
                        "HEAD",
                    ],
                    check=False,
                ).returncode
                != 0
            )
            assert (
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(REPO_ROOT),
                        "merge-base",
                        "--is-ancestor",
                        generation["integrated_commit"],
                        "HEAD",
                    ],
                    check=False,
                ).returncode
                == 0
            )


def test_ase3_019_source_and_salvage_freeze_is_sealed() -> None:
    final_values = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-019"
    ]
    assert final_values["ready"] is True
    assert final_values["pending"] is None
    assert final_values["validation_passed_count"] == 160
    source = final_values["source_candidate"]
    assert source["source_commit"] == (
        convergence_module._ASE3_019_ATTEMPT2_PRIOR_SEED["source_commit"]
    )
    assert source["source_tree"] == (
        convergence_module._ASE3_019_ATTEMPT2_PRIOR_SEED["source_tree"]
    )
    salvage = final_values["salvage_base"]
    assert salvage["branch"] == "agent/prompt-self-improvement-v3"
    # Salvage tip is a main-reachable integrated product commit.
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                salvage["head"],
                "HEAD",
            ],
            check=False,
        ).returncode
        == 0
    )
    tree = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{salvage['head']}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tree == salvage["tree"]
    # Source remains available for incident reconstruction.
    assert (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "cat-file",
                "-e",
                f"{source['source_commit']}^{{commit}}",
            ],
            check=False,
        ).returncode
        == 0
    )


def test_ase3_027_generation_rejects_wrong_patch_path_and_topology() -> None:
    expected = convergence_module._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[
        "ASE3-027"
    ]["generations"][0]
    generation = {
        key: list(value) if key == "changed_paths" else value
        for key, value in expected.items()
    }
    assert validate_git_generation_provenance(
        repo_root=REPO_ROOT,
        generation=generation,
        acceptance_parent_head="d32415e4308a8462e96b4d04f807338f0a2d8b53",
    ) == ()

    generation["binary_full_index_patch_sha256"] = "sha256:" + ("0" * 64)
    generation["changed_paths"] = ["forged.py"]
    errors = validate_git_generation_provenance(
        repo_root=REPO_ROOT,
        generation=generation,
        acceptance_parent_head="d32415e4308a8462e96b4d04f807338f0a2d8b53",
    )
    assert any("patch" in error for error in errors)
    assert any("changed_paths" in error for error in errors)


def test_ase3_030_signed_hermetic_receipt_schema_is_exact() -> None:
    payload, frozen, authority = _hermetic_acceptance_receipt_fixture()

    assert convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
    ) == ()


def test_ase3_030_self_consistent_receipt_cannot_select_its_own_provenance() -> None:
    payload, _, authority = _hermetic_acceptance_receipt_fixture()

    errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
    )

    # After freeze, fixture-selected generations/blobs must not match sealed pins.
    assert errors
    assert not any(
        convergence_module._FINAL_VALUE_PENDING_030 in error for error in errors
    )
    assert any(
        "generations" in error or "final_blobs" in error or "frozen" in error
        for error in errors
    )


def test_ase3_030_031_032_acceptance_final_values_are_sealed() -> None:
    hermetic = convergence_module._HERMETIC_IDENTITY_FINAL_VALUES
    assert hermetic["ready"] is True
    assert hermetic["pending"] is None
    assert hermetic["suite_passed_count"] == 108
    assert len(hermetic["generations"]) == 2
    assert len(hermetic["final_blobs"]) == 7
    assert len(hermetic["final_raw_sha256"]) == 7
    assert hermetic["manifest_sha256"].startswith("sha256:")
    assert hermetic["capsule_sha256"].startswith("sha256:")
    assert hermetic["archive_sha256"].startswith("sha256:")

    native = convergence_module._NATIVE_DEPENDENCY_ACCEPTANCE_FINAL_VALUES
    assert native["ready"] is True
    assert native["pending"] is None
    assert native["passed_count"] == 46
    assert native["report_sha256"].startswith("sha256:")

    duckdb = convergence_module._DUCKDB_POLICY_ACCEPTANCE_FINAL_VALUES
    assert duckdb["ready"] is True
    assert duckdb["pending"] is None
    assert duckdb["passed_count"] == 51
    assert duckdb["report_sha256"].startswith("sha256:")


def test_ase3_030_generation_reconstructs_source_replay_and_integrated_git(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "hermetic-generation"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Hermetic Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "hermetic@example.invalid"],
        cwd=repository,
        check=True,
    )
    path = repository / "ipfs_accelerate_py/llm_router.py"
    path.parent.mkdir(parents=True)
    path.write_text("old = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "base"],
        cwd=repository,
        check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path.write_text("old = False\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    changed_tree = subprocess.run(
        ["git", "write-tree"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def commit_tree(message: str, parent: str) -> str:
        return subprocess.run(
            ["git", "commit-tree", changed_tree, "-p", parent, "-m", message],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    source = commit_tree("source", base)
    replay = commit_tree("replay", base)
    integrated = commit_tree("integrated", base)
    acceptance = subprocess.run(
        [
            "git",
            "commit-tree",
            changed_tree,
            "-p",
            integrated,
            "-m",
            "acceptance parent",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = convergence_module._git_diff_patch(repository, base, source)
    assert patch.returncode == 0
    patch_sha = "sha256:" + hashlib.sha256(patch.stdout).hexdigest()
    generation = {
        "role": "hermetic-control-plane",
        "source_commit": source,
        "source_parent": base,
        "source_tree": changed_tree,
        "replay_commit": replay,
        "replay_parent": base,
        "replay_tree": changed_tree,
        "integrated_commit": integrated,
        "integrated_parent": base,
        "integrated_tree": changed_tree,
        "source_patch_sha256": patch_sha,
        "replay_patch_sha256": patch_sha,
        "integrated_patch_sha256": patch_sha,
        "changed_paths": ["ipfs_accelerate_py/llm_router.py"],
    }

    assert convergence_module.validate_hermetic_generation_provenance(
        repo_root=repository,
        generation=generation,
        acceptance_parent_head=acceptance,
    ) == ()

    path.write_text("old = False\nextra = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    mismatched_tree = subprocess.run(
        ["git", "write-tree"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mismatched_replay = subprocess.run(
        [
            "git",
            "commit-tree",
            mismatched_tree,
            "-p",
            base,
            "-m",
            "mismatched replay",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mismatched_patch = convergence_module._git_diff_patch(
        repository,
        base,
        mismatched_replay,
    )
    assert mismatched_patch.returncode == 0
    mismatched_generation = dict(generation)
    mismatched_generation.update(
        {
            "replay_commit": mismatched_replay,
            "replay_tree": mismatched_tree,
            "replay_patch_sha256": "sha256:"
            + hashlib.sha256(mismatched_patch.stdout).hexdigest(),
        }
    )
    mismatch_errors = convergence_module.validate_hermetic_generation_provenance(
        repo_root=repository,
        generation=mismatched_generation,
        acceptance_parent_head=acceptance,
    )
    assert any("patch digests must be identical" in error for error in mismatch_errors)
    assert any(
        "full-index binary patches must be byte-identical" in error
        for error in mismatch_errors
    )

    generation["replay_patch_sha256"] = "sha256:" + ("0" * 64)
    errors = convergence_module.validate_hermetic_generation_provenance(
        repo_root=repository,
        generation=generation,
        acceptance_parent_head=acceptance,
    )
    assert any("replay_patch" in error for error in errors)


def test_ase3_030_closure_reconstructs_full_git_blob_and_raw_map() -> None:
    payload, frozen, _ = _hermetic_acceptance_receipt_fixture()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    paths = list(frozen["member_paths"])
    blobs: dict[str, str] = {}
    raw_map: dict[str, str] = {}
    members: dict[str, dict[str, str]] = {}
    for path in paths:
        blob = subprocess.run(
            ["git", "rev-parse", f"{head}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        raw = subprocess.run(
            ["git", "show", f"{head}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        raw_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
        blobs[path] = blob
        raw_map[path] = raw_sha
        members[path] = {
            "git_blob": blob,
            "raw_sha256": raw_sha,
            "archive_member_sha256": raw_sha,
        }
    closure = payload["closure"]
    assert isinstance(closure, dict)
    manifest = closure["manifest"]
    archive = closure["archive"]
    capsule = closure["capsule"]
    assert isinstance(manifest, dict)
    assert isinstance(archive, dict)
    assert isinstance(capsule, dict)
    manifest["source_head"] = head
    manifest["source_tree"] = tree
    manifest_sha = convergence_module._canonical_sha256(manifest)
    archive_root = convergence_module._canonical_sha256(
        {"member_paths": paths, "members": members}
    )
    archive["root_sha256"] = archive_root
    capsule["manifest_sha256"] = manifest_sha
    capsule["archive_root_sha256"] = archive_root
    closure["manifest_sha256"] = manifest_sha
    closure["members"] = members
    closure["capsule_sha256"] = convergence_module._canonical_sha256(capsule)
    frozen.update(
        {
            "final_blobs": blobs,
            "final_raw_sha256": raw_map,
            "manifest_sha256": manifest_sha,
            "archive_root_sha256": archive_root,
            "capsule_sha256": closure["capsule_sha256"],
        }
    )

    assert convergence_module._validate_hermetic_closure(
        payload=closure,
        acceptance_parent_head=head,
        acceptance_parent_tree=tree,
        final_values=frozen,
        repo_root=REPO_ROOT,
    ) == []
    first_path = paths[0]
    members[first_path]["raw_sha256"] = "sha256:" + ("0" * 64)
    errors = convergence_module._validate_hermetic_closure(
        payload=closure,
        acceptance_parent_head=head,
        acceptance_parent_tree=tree,
        final_values=frozen,
        repo_root=REPO_ROOT,
    )
    assert any(f"members.{first_path}.raw_sha256: Git mismatch" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    (
        ("missing", "exact reviewed dependency closure required"),
        ("extra", "exact reviewed dependency closure required"),
        ("shadow", "reviewed_map"),
    ),
)
def test_ase3_030_reviewed_inventory_cannot_be_changed_with_frozen_maps(
    mutation: str,
    error_fragment: str,
) -> None:
    payload, frozen, _ = _hermetic_acceptance_receipt_fixture()
    closure = payload["closure"]
    assert isinstance(closure, dict)
    members = closure["members"]
    origins = closure["module_origins"]
    manifest = closure["manifest"]
    archive = closure["archive"]
    capsule = closure["capsule"]
    assert isinstance(members, dict)
    assert isinstance(origins, dict)
    assert isinstance(manifest, dict)
    assert isinstance(archive, dict)
    assert isinstance(capsule, dict)
    member_paths = manifest["member_paths"]
    assert isinstance(member_paths, list)
    final_blobs = frozen["final_blobs"]
    final_raw = frozen["final_raw_sha256"]
    assert isinstance(final_blobs, dict)
    assert isinstance(final_raw, dict)

    if mutation == "missing":
        module_name, member_path = next(
            reversed(convergence_module._HERMETIC_REQUIRED_MODULE_MEMBER_MAP.items())
        )
        origins.pop(module_name)
        members.pop(member_path)
        final_blobs.pop(member_path)
        final_raw.pop(member_path)
        member_paths.remove(member_path)
    elif mutation == "extra":
        module_name = "sitecustomize"
        member_path = "sitecustomize.py"
        digest = "sha256:" + hashlib.sha256(member_path.encode()).hexdigest()
        origins[module_name] = {
            "member_path": member_path,
            "origin": f"capsule://sealed/{member_path}",
        }
        members[member_path] = {
            "git_blob": "f" * 40,
            "raw_sha256": digest,
            "archive_member_sha256": digest,
        }
        final_blobs[member_path] = "f" * 40
        final_raw[member_path] = digest
        member_paths.append(member_path)
        member_paths.sort()
    else:
        origin_values = list(origins.values())
        assert all(isinstance(value, dict) for value in origin_values)
        first = origin_values[0]
        second = origin_values[1]
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        second["member_path"] = first["member_path"]
        second["origin"] = first["origin"]

    manifest["module_names"] = list(origins)
    archive["member_paths"] = member_paths
    archive_root = convergence_module._canonical_sha256(
        {"member_paths": member_paths, "members": members}
    )
    archive["root_sha256"] = archive_root
    manifest_sha = convergence_module._canonical_sha256(manifest)
    closure["manifest_sha256"] = manifest_sha
    capsule["manifest_sha256"] = manifest_sha
    capsule["archive_root_sha256"] = archive_root
    capsule["member_count"] = len(member_paths)
    closure["capsule_sha256"] = convergence_module._canonical_sha256(capsule)
    frozen.update(
        {
            "member_paths": member_paths,
            "module_origins": origins,
            "manifest_sha256": manifest_sha,
            "archive_root_sha256": archive_root,
            "capsule_sha256": closure["capsule_sha256"],
        }
    )

    errors = convergence_module._validate_hermetic_closure(
        payload=closure,
        acceptance_parent_head="1" * 40,
        acceptance_parent_tree="2" * 40,
        final_values=frozen,
        repo_root=None,
    )

    assert any(error_fragment in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        ("probe", "exit_code", False, "probe.contract.exit_code"),
        ("probe", "isolated", 1, "probe.contract.isolated"),
        (
            "probe",
            "scheduler_or_provider_effect_started",
            0,
            "scheduler_or_provider_effect_started",
        ),
    ),
)
def test_ase3_030_false_zero_and_bool_values_fail_closed(
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    payload, frozen, authority = _hermetic_acceptance_receipt_fixture()
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = replacement

    errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
    )

    assert any(error_fragment in error for error in errors)


def test_ase3_030_probe_argv_is_independent_of_self_consistent_frozen_values() -> None:
    payload, frozen, authority = _hermetic_acceptance_receipt_fixture()
    probe = payload["probe"]
    assert isinstance(probe, dict)
    bad_argv = list(convergence_module._HERMETIC_HOSTILE_PROBE_ARGV)
    bad_argv[0] = "/usr/bin/python"
    probe["command"] = bad_argv
    frozen["probe_command"] = bad_argv

    errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
    )

    assert any("independently frozen hostile-probe argv mismatch" in error for error in errors)
    assert any("fixed portable interpreter token required" in error for error in errors)
    assert any("absolute or machine-local argv value forbidden" in error for error in errors)


def test_ase3_030_extra_archive_member_and_module_shadow_fail_closed() -> None:
    payload, frozen, authority = _hermetic_acceptance_receipt_fixture()
    closure = payload["closure"]
    assert isinstance(closure, dict)
    archive = closure["archive"]
    origins = closure["module_origins"]
    assert isinstance(archive, dict)
    assert isinstance(origins, dict)
    member_paths = archive["member_paths"]
    assert isinstance(member_paths, list)
    member_paths.append("sitecustomize.py")
    origin_values = list(origins.values())
    assert all(isinstance(value, dict) for value in origin_values)
    first = origin_values[0]
    second = origin_values[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["member_path"] = first["member_path"]
    second["origin"] = first["origin"]

    errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
    )

    assert any("archive.contract.member_paths" in error for error in errors)
    assert any("shadowed member origin forbidden" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("closure", "capsule_root"),
        ("provenance", "convergence_manifest_digest"),
        ("acceptance_parent", "acceptance_head"),
    ),
)
def test_ase3_030_machine_local_circular_and_self_bound_fields_are_forbidden(
    section: str,
    field: str,
) -> None:
    payload, frozen, authority = _hermetic_acceptance_receipt_fixture()
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = "sha256:" + ("0" * 64)

    errors = convergence_module.validate_hermetic_identity_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
    )

    assert any(f"{section}.{field}: forbidden receipt authority field" in error for error in errors)


def test_reload_receipt_authorizes_old_plus_one_without_claiming_birth() -> None:
    payload, frozen, authority = _reload_receipt_fixture()

    assert convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
        frozen_values=frozen,
    ) == ()
    authorization = payload["authorization"]
    assert isinstance(authorization, dict)
    assert authorization["lease_state_at_authorization"] == "unclaimed"
    assert authorization["runtime_effect_started"] is False
    assert authorization["launch_only_after_l_validates"] is True
    assert authorization["post_launch_birth_receipt_required"] is True


def test_reload_control_plane_identity_comes_from_a_not_reload_frozen_values() -> None:
    payload, frozen, authority = _reload_receipt_fixture()
    incident = payload["incident"]
    assert isinstance(incident, dict)
    self_asserted = "sha256:" + ("0" * 64)
    incident["accepted_control_plane_sha256"] = self_asserted
    frozen["accepted_control_plane_sha256"] = self_asserted

    errors = convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
    )

    assert any("accepted_control_plane_sha256" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "stopped_generation",
            "observed_owned_processes",
            False,
            "observed_owned_processes",
        ),
        (
            "authorization",
            "single_winner_required",
            1,
            "single_winner_required",
        ),
        (
            "authorization",
            "runtime_effect_started",
            True,
            "runtime_effect_started",
        ),
    ),
)
def test_reload_receipt_false_zero_bool_and_false_open_claims_are_rejected(
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    payload, frozen, authority = _reload_receipt_fixture()
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = replacement

    errors = convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
    )

    assert any(error_fragment in error for error in errors)


def test_reload_receipt_cannot_bind_l_or_convergence_manifest() -> None:
    payload, frozen, authority = _reload_receipt_fixture()
    parent = payload["acceptance_parent"]
    assert isinstance(parent, dict)
    parent["reload_head"] = "1" * 40
    payload["convergence_manifest_sha256"] = "sha256:" + ("0" * 64)

    errors = convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
    )

    assert any("reload_head: forbidden receipt authority field" in error for error in errors)
    assert any("convergence_manifest_sha256: forbidden" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("authorization", "lease_acquired", True),
        ("authorization", "completed", True),
        ("authorization", "birth_receipt", {"ran": True}),
        ("top", "post_launch_effect", {"started": True}),
    ),
)
def test_l_receipt_cannot_contain_birth_effect_or_completion_claims(
    section: str,
    field: str,
    replacement: object,
) -> None:
    payload, frozen, authority = _reload_receipt_fixture()
    if section == "top":
        payload[field] = replacement
    else:
        block = payload[section]
        assert isinstance(block, dict)
        block[field] = replacement

    errors = convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        frozen_values=frozen,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
    )

    assert any("exact key population required" in error for error in errors)


def test_reload_final_value_sentinel_is_fail_closed() -> None:
    payload, _, authority = _reload_receipt_fixture()

    errors = convergence_module.validate_provider_attempt_reload_receipt(
        payload,
        lifecycle_authority=authority,
        accepted_control_plane=convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE,
    )

    assert any(
        convergence_module._FINAL_VALUE_PENDING_RELOAD in error for error in errors
    )


def test_reload_receipt_loader_rejects_unsafe_mode_and_hardlink(
    tmp_path: Path,
) -> None:
    path = tmp_path / PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME
    payload, _, _ = _reload_receipt_fixture()
    _write(path, payload)
    path.chmod(0o664)
    with pytest.raises(ValueError, match="group-or-other writable"):
        convergence_module.load_provider_attempt_reload_receipt(path)

    path.chmod(0o600)
    backing = tmp_path / "reload-backing.json"
    path.rename(backing)
    os.link(backing, path)
    with pytest.raises(ValueError, match="single-link evidence file"):
        convergence_module.load_provider_attempt_reload_receipt(path)


def test_ase3_019_control_plane_and_counter_effects_are_exact() -> None:
    control_plane = json.loads(
        json.dumps(convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE)
    )
    assert validate_ase3_019_accepted_control_plane(control_plane) == ()
    public_api = control_plane["public_api"]
    assert public_api == {
        "route_plan_type": "AgentImplementationRoutePlan",
        "fallback_decision_type": "AgentImplementationFallbackDecision",
        "capacity_projection_api": "project_agent_implementation_route_capacity",
        "control_plane_pin_type": "AgentImplementationControlPlanePin",
        "sealed_control_plane_type": "AgentImplementationSealedControlPlane",
        "source_generation_api": (
            "agent_implementation_control_plane_source_generation"
        ),
        "materialize_api": (
            "materialize_agent_implementation_control_plane_capsule"
        ),
        "build_pin_api": "build_agent_implementation_control_plane_pin",
        "seal_api": "seal_agent_implementation_control_plane_capsule",
        "verify_sealed_api": (
            "verify_agent_implementation_sealed_control_plane"
        ),
        "pin_schema": (
            "ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2"
        ),
        "manifest_schema": (
            "ipfs_accelerate_py.agent_supervisor.materialized-control-plane@1"
        ),
        "terminal_outcome_field": "accepted_control_plane",
    }
    assert all(control_plane["portable_acceptance_evidence"].values())
    assert "runner_path" not in control_plane
    assert "capsule_root" not in control_plane
    assert "executable_path" not in control_plane

    control_plane["canonical_route_owner"] = "implementation_daemon"
    control_plane["fallback_reasoning_effort"] = "medium"
    control_plane["attempt_counter_mutation_authorized"] = True
    control_plane["provider_capacity_attempt_restoration_allowed"] = True
    public_api["control_plane_pin_type"] = "CallerControlPlaneDTO"
    control_plane["runner_path"] = "/proc/self/fd/42"
    errors = validate_ase3_019_accepted_control_plane(control_plane)
    assert any("canonical_route_owner" in error for error in errors)
    assert any("fallback_reasoning_effort" in error for error in errors)
    assert any("attempt_counter_mutation_authorized" in error for error in errors)
    assert any(
        "provider_capacity_attempt_restoration_allowed" in error for error in errors
    )
    assert any("control_plane_pin_type" in error for error in errors)
    assert any("runner_path" in error for error in errors)


@pytest.mark.parametrize("field", ("exit_code", "failed_count"))
def test_acceptance_validation_rejects_false_as_integer_zero(field: str) -> None:
    payload, _, _ = _operator_repair_receipt_027()
    validation = payload["validation"]
    parent = payload["acceptance_parent"]
    assert isinstance(validation, dict)
    assert isinstance(parent, dict)
    validation[field] = False

    errors = convergence_module._validate_acceptance_validation(
        payload=validation,
        task_id="ASE3-027",
        acceptance_parent=parent,
    )

    assert any(
        f".{field}: expected integer in inclusive range 0..0" in error
        for error in errors
    )


def legacy_acceptance_manifest_at_2_binds_exact_receipts_tasks_and_parent(
    tmp_path: Path,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    manifest_path = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
        / MANIFEST_FILENAME
    )
    manifest = ConvergenceManifest.from_dict(_load(manifest_path))
    baseline = CurrentMainBaseline.from_dict(
        _load(DEFAULT_ARTIFACT_ROOT / "current_main_baseline.json")
    )
    assert manifest.validate(baseline) == ()

    payload = _load(manifest_path)
    acceptance = payload["acceptance"]
    assert isinstance(acceptance, dict)
    acceptance["reload_gate_completed"] = True
    errors = ConvergenceManifest.from_dict(payload).validate(baseline)
    assert any("reload_gate_completed" in error for error in errors)

    payload = _load(manifest_path)
    payload["unauthorized_top_level"] = True
    errors = ConvergenceManifest.from_dict(payload).validate(baseline)
    assert "convergence_manifest: exact @2 top-level population required" in errors


def legacy_acceptance_packet_rejects_a_manifest_receipt_digest_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    receipt_digests: dict[str, str] = {}
    for task_id, expected in convergence_module._ACCEPTANCE_TASK_CONTRACTS.items():
        filename = str(expected["filename"])
        path = root / filename
        _write(path, _minimal_operator_receipt(task_id))
        receipt_digests[filename] = "sha256:" + hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    receipt_digests[OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME] = (
        "sha256:" + ("0" * 64)
    )
    manifest_payload = _load(root / MANIFEST_FILENAME)
    manifest_payload["schema"] = ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
    manifest_payload["acceptance"] = {
        "phase": "operator_acceptance",
        "preparation_head": "d32415e4308a8462e96b4d04f807338f0a2d8b53",
        "preparation_tree": "87191ce65498a637c7b9500d72d434cadb8efbef",
        "receipts": receipt_digests,
        "tasks": {},
        "reload_gate_completed": False,
    }
    errors, checked = convergence_module._validate_operator_acceptance_packet(
        artifact_root=root,
        manifest=ConvergenceManifest.from_dict(manifest_payload),
        repo_root=None,
    )
    assert set(checked) == {
        *OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES,
        convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_FILENAME,
        convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_FILENAME,
    }
    assert any(
        f"receipts.{OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME}: digest mismatch"
        in error
        for error in errors
    )


def legacy_acceptance_transition_is_one_direct_child_with_exact_six_paths(
    tmp_path: Path,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert set(ACCEPTANCE_CHILD_CHANGED_PATHS) == set(
        subprocess.run(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                preparation_head,
                acceptance_head,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    root_pin_head = subprocess.run(
        ["git", "rev-parse", f"{preparation_head}^"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle_base_head = subprocess.run(
        ["git", "rev-parse", f"{root_pin_head}^"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert subprocess.run(
        [
            "git",
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            lifecycle_base_head,
            root_pin_head,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines() == [
        convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    ]
    assert set(
        subprocess.run(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                root_pin_head,
                preparation_head,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    ) == {
        convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH,
        convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
        (
            "data/agent_supervisor/prompt_only_self_improvement_v3/"
            f"convergence/{MANIFEST_FILENAME}"
        ),
    }
    assert validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        **_transition_lifecycle_kwargs(repository),
    ) == ()


@pytest.mark.parametrize(
    ("options", "error_fragment"),
    (
        (
            {"q_manifest_updates": {"schema": "forged-final-prep@1"}},
            ".Q.manifest.schema: exact @1 required",
        ),
        ({"root_pin_extra_path": True}, ".R.direct_child.changed_paths"),
        (
            {"preparation_extra_path": True},
            ".P019.direct_child.changed_paths",
        ),
        (
            {"p019_manifest_updates": {"self_consistent_extra": True}},
            ".P019.manifest_transformation: exact key population required",
        ),
    ),
)
def test_sequential_q_r_p019_reject_invalid_final_prep_and_extra_diffs(
    tmp_path: Path,
    options: dict[str, object],
    error_fragment: str,
) -> None:
    repository, heads = _sequential_transition_repository(tmp_path, **options)
    errors = convergence_module.validate_protected_acceptance_sequence(
        repo_root=repository,
        phase_heads=heads,
    )

    assert any(error_fragment in error for error in errors)


def legacy_reload_transition_is_one_direct_child_with_exact_three_paths(
    tmp_path: Path,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    reload_head, acceptance_head, acceptance_tree = _append_reload_transition(
        repository
    )
    observed = subprocess.run(
        [
            "git",
            "diff",
            *convergence_module._DETERMINISTIC_GIT_DIFF_FLAGS,
            "--name-only",
            acceptance_head,
            reload_head,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert observed == sorted(convergence_module.RELOAD_CHILD_CHANGED_PATHS)
    consumed = {
        relative_path: (repository / relative_path).read_bytes()
        for relative_path in convergence_module.RELOAD_CHILD_CHANGED_PATHS
    }
    assert convergence_module.validate_reload_child_transition(
        repo_root=repository,
        reload_head=reload_head,
        acceptance_head=acceptance_head,
        acceptance_tree=acceptance_tree,
        consumed_reload_blobs=consumed,
    ) == ()


@pytest.mark.parametrize(
    ("options", "error_fragment"),
    (
        ({"extra_path": True}, "changed_paths"),
        ({"extra_board_prose": True}, "only ASE3-022"),
        ({"wrong_acceptance_parent": True}, "parent mismatch"),
        ({"executable_receipt": True}, "exact 100644 Git mode"),
        ({"reopen_acceptance_task": "ASE3-023"}, "only ASE3-022"),
        ({"include_birth_receipt": True}, "post-L birth receipt must be absent"),
    ),
)
def legacy_reload_transition_rejects_extra_paths_statuses_parent_and_modes(
    tmp_path: Path,
    options: dict[str, object],
    error_fragment: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    reload_head, acceptance_head, acceptance_tree = _append_reload_transition(
        repository,
        **options,
    )

    errors = convergence_module.validate_reload_child_transition(
        repo_root=repository,
        reload_head=reload_head,
        acceptance_head=acceptance_head,
        acceptance_tree=acceptance_tree,
    )

    assert any(error_fragment in error for error in errors)


def legacy_reload_transition_rejects_dirty_consumed_receipt_blob(
    tmp_path: Path,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    reload_head, acceptance_head, acceptance_tree = _append_reload_transition(
        repository
    )
    consumed = {
        relative_path: (repository / relative_path).read_bytes()
        for relative_path in convergence_module.RELOAD_CHILD_CHANGED_PATHS
    }
    consumed[PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH] += b"forged"

    errors = convergence_module.validate_reload_child_transition(
        repo_root=repository,
        reload_head=reload_head,
        acceptance_head=acceptance_head,
        acceptance_tree=acceptance_tree,
        consumed_reload_blobs=consumed,
    )

    assert any("does not match reload HEAD" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    (
        ("launch_authorization_only", False, "launch_authorization_only"),
        (
            "post_launch_birth_receipt_required",
            0,
            "post_launch_birth_receipt_required",
        ),
        (
            "post_launch_birth_receipt_schema",
            "self-asserted-birth@1",
            "post_launch_birth_receipt_schema",
        ),
    ),
)
def legacy_reload_manifest_rejects_false_open_birth_claims(
    tmp_path: Path,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    _append_reload_transition(repository)
    artifact_root = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    payload = _load(artifact_root / MANIFEST_FILENAME)
    baseline = CurrentMainBaseline.from_dict(
        _load(artifact_root / "current_main_baseline.json")
    )
    assert ConvergenceManifest.from_dict(payload).validate(baseline) == ()
    reload_binding = payload["reload"]
    assert isinstance(reload_binding, dict)
    reload_binding[field] = replacement

    errors = ConvergenceManifest.from_dict(payload).validate(baseline)

    assert any(error_fragment in error for error in errors)


def legacy_top_level_at3_dispatches_a_validation_before_l_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    _, acceptance_head, _ = _append_reload_transition(repository)
    artifact_root = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    calls: list[str] = []

    def acceptance_dispatch(**kwargs: object) -> tuple[list[str], tuple[str, ...]]:
        calls.append("acceptance")
        assert kwargs["acceptance_head_override"] == acceptance_head
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ConvergenceManifest)
        assert (
            manifest.payload["schema"]
            == convergence_module.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
        )
        return [], ("acceptance-dispatch",)

    def reload_dispatch(**kwargs: object) -> tuple[list[str], tuple[str, ...]]:
        calls.append("reload")
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ConvergenceManifest)
        assert (
            manifest.payload["schema"]
            == convergence_module.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
        )
        return [], ("reload-dispatch",)

    monkeypatch.setattr(
        convergence_module,
        "_validate_operator_acceptance_packet",
        acceptance_dispatch,
    )
    monkeypatch.setattr(
        convergence_module,
        "_validate_provider_attempt_reload_packet",
        reload_dispatch,
    )

    report = validate_convergence_artifacts(
        artifact_root,
        repo_root=repository,
        check_repository=False,
        taskboard_path=repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    )

    assert calls == ["acceptance", "reload"]
    assert "acceptance-dispatch" in report.checked_artifacts
    assert "reload-dispatch" in report.checked_artifacts


def legacy_reload_packet_derives_control_plane_from_committed_signed_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    artifact_root = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    private_key = Ed25519PrivateKey.generate()
    signed_authority = _review_authority(_reviewer_identity(private_key))
    salvage_payload = _minimal_operator_receipt("ASE3-019")
    salvage_payload["accepted_control_plane"] = json.loads(
        json.dumps(convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE)
    )
    salvage_payload["review"] = {
        **_receipt_review_authority(signed_authority),
        "implementer_identity": "codex:ase3-019-product",
        "implementer_provider": "codex",
        "algorithm": "Ed25519",
        "signed_at": salvage_payload["created_at"],
        "signature": "",
    }
    _sign_operator_receipt(salvage_payload, private_key)
    salvage_path = artifact_root / OPERATOR_SALVAGE_RECEIPT_019_FILENAME
    _write(salvage_path, salvage_payload)
    manifest_path = artifact_root / MANIFEST_FILENAME
    manifest_payload = _load(manifest_path)
    acceptance_binding = manifest_payload["acceptance"]
    assert isinstance(acceptance_binding, dict)
    receipt_bindings = acceptance_binding["receipts"]
    assert isinstance(receipt_bindings, dict)
    receipt_bindings[OPERATOR_SALVAGE_RECEIPT_019_FILENAME] = (
        "sha256:" + hashlib.sha256(salvage_path.read_bytes()).hexdigest()
    )
    _write(manifest_path, manifest_payload)
    subprocess.run(
        ["git", "add", str(salvage_path), str(manifest_path)],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", "-q"],
        cwd=repository,
        check=True,
    )
    _append_reload_transition(repository)

    reload_payload, _, _ = _reload_receipt_fixture()
    reload_raw = convergence_module._canonical_json_bytes(reload_payload)
    reload_snapshot = convergence_module.OperatorAcceptanceReceiptSnapshot(
        filename=PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME,
        payload=reload_payload,
        sha256="sha256:" + hashlib.sha256(reload_raw).hexdigest(),
        raw=reload_raw,
    )
    captured_control_planes: list[Mapping[str, object] | None] = []
    signature_calls: list[Mapping[str, object]] = []
    real_signature_validator = validate_operator_acceptance_signature

    def signed_a_spy(
        payload: Mapping[str, object],
        *,
        expected_authority: Mapping[str, object] | None = None,
    ) -> tuple[str, ...]:
        signature_calls.append(payload)
        return real_signature_validator(
            payload,
            expected_authority=signed_authority,
        )

    def reload_receipt_spy(
        payload: Mapping[str, object],
        **kwargs: object,
    ) -> tuple[str, ...]:
        del payload
        candidate = kwargs.get("accepted_control_plane")
        captured_control_planes.append(
            candidate if isinstance(candidate, Mapping) else None
        )
        return ()

    monkeypatch.setattr(
        convergence_module,
        "load_provider_attempt_reload_receipt",
        lambda *args, **kwargs: reload_snapshot,
    )
    monkeypatch.setattr(
        convergence_module,
        "validate_operator_acceptance_signature",
        signed_a_spy,
    )
    monkeypatch.setattr(
        convergence_module,
        "validate_provider_attempt_reload_receipt",
        reload_receipt_spy,
    )

    reload_manifest_raw = manifest_path.read_bytes()
    reload_manifest = ConvergenceManifest.from_dict(_load(manifest_path))
    fallback_path = (
        artifact_root / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
    )
    errors, _ = convergence_module._validate_provider_attempt_reload_packet(
        artifact_root=artifact_root,
        manifest=reload_manifest,
        repo_root=repository,
        fallback_authorization=(
            convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
                _load(fallback_path)
            )
        ),
        fallback_authorization_raw=fallback_path.read_bytes(),
        manifest_raw=reload_manifest_raw,
        taskboard_raw=(repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH).read_bytes(),
    )

    assert signature_calls == [salvage_payload]
    assert captured_control_planes == [
        convergence_module._ASE3_019_ACCEPTED_CONTROL_PLANE
    ]
    assert not any("committed A receipt raw digest mismatch" in error for error in errors)

    reload_manifest.payload["acceptance"]["receipts"][
        OPERATOR_SALVAGE_RECEIPT_019_FILENAME
    ] = "sha256:" + ("0" * 64)
    forged_errors, _ = convergence_module._validate_provider_attempt_reload_packet(
        artifact_root=artifact_root,
        manifest=reload_manifest,
        repo_root=repository,
        fallback_authorization=(
            convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
                _load(fallback_path)
            )
        ),
        fallback_authorization_raw=fallback_path.read_bytes(),
        manifest_raw=reload_manifest_raw,
        taskboard_raw=(repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH).read_bytes(),
    )
    assert any(
        "committed A receipt raw digest mismatch" in error
        for error in forged_errors
    )


def test_lifecycle_root_witness_and_authorization_v2_are_portably_valid(
    tmp_path: Path,
) -> None:
    repository, preparation_head, _ = _initialize_transition_repository(tmp_path)
    root_path = (
        repository
        / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    witness_path = (
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    authorization_path = (
        repository
        / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    root_snapshot = convergence_module.load_local_profile_lifecycle_root_pin(
        root_path
    )
    witness_snapshot = convergence_module.load_local_operator_lifecycle_witness(
        witness_path
    )
    authorization = convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
        _load(authorization_path)
    )
    root_head = subprocess.run(
        ["git", "rev-parse", f"{preparation_head}^"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_tree = subprocess.run(
        ["git", "rev-parse", f"{root_head}^{{tree}}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_time_ms = (
        int(
            subprocess.run(
                ["git", "show", "-s", "--format=%ct", root_head],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        * 1000
    )
    lifecycle_kwargs = _transition_lifecycle_kwargs(repository)
    final_values = lifecycle_kwargs["expected_final_values"]
    assert isinstance(final_values, dict)
    assert convergence_module.validate_local_profile_lifecycle_root_pin(
        root_snapshot.payload,
        expected_root_identity_did=root_snapshot.root_identity_did,
    ) == ()
    assert convergence_module.validate_local_operator_lifecycle_witness(
        witness_snapshot.payload,
        root_identity_did=root_snapshot.root_identity_did,
        expected_base_head=root_head,
        expected_base_tree=root_tree,
        reference_time_ms=authorization.payload["authorized_at_ms"],
        earliest_observed_at_ms=root_time_ms,
        expected_final_values=final_values,
    ) == ()
    assert authorization.validate(
        lifecycle_witness=witness_snapshot,
        root_pin=root_snapshot,
        expected_source_head=root_head,
        expected_source_tree=root_tree,
        expected_final_values=final_values,
    ) == ()

    authorization_sha256 = "sha256:" + hashlib.sha256(
        authorization_path.read_bytes()
    ).hexdigest()
    review_authority = authorization.acceptance_review_authority(
        raw_sha256=authorization_sha256,
        lifecycle_witness=witness_snapshot,
        root_pin=root_snapshot,
    )
    assert review_authority["lifecycle_witness_observed_at_ms"] == (
        witness_snapshot.payload["observed_at_ms"]
    )
    assert review_authority["lifecycle_witness_expires_at_ms"] == (
        witness_snapshot.payload["expires_at_ms"]
    )
    assert review_authority["fallback_authorized_at_ms"] == (
        authorization.payload["authorized_at_ms"]
    )
    source = authorization.payload["authorization_source"]
    reviewer = authorization.payload["reviewer"]
    bounds = authorization.payload["authority_bounds"]
    assert isinstance(source, dict)
    assert isinstance(reviewer, dict)
    assert isinstance(bounds, dict)
    identity_material = {
        "schema": convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "artifact_path": (
            convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
        ),
        "artifact_sha256": authorization_sha256,
        "authorization_kind": source["kind"],
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "reviewer_identity": reviewer["identity"],
        "reviewer_provider": reviewer["provider"],
        "reviewer_signature": reviewer["signature"],
        "reviewer_profile_id": reviewer["profile_id"],
        "reviewer_profile_content_id": reviewer["profile_content_id"],
        "reviewer_lifecycle_anchor_id": reviewer["lifecycle_anchor_id"],
        "reviewer_lifecycle_generation": reviewer["generation"],
        "reviewer_witness_path": reviewer["witness_path"],
        "reviewer_witness_sha256": reviewer["witness_sha256"],
        "lifecycle_root_identity_did": authorization.payload[
            "lifecycle_root_identity_did"
        ],
        "lifecycle_witness_nonce": authorization.payload[
            "lifecycle_witness_nonce"
        ],
        "lifecycle_root_pin_path": authorization.payload[
            "lifecycle_root_pin_path"
        ],
        "lifecycle_root_pin_sha256": authorization.payload[
            "lifecycle_root_pin_sha256"
        ],
        "authorized_at_ms": authorization.payload["authorized_at_ms"],
        "fallback_implementer_identity": authorization.payload[
            "fallback_implementer_identity"
        ],
        "authority_bounds": bounds,
        "authorization_id": "",
    }
    expected_authorization_id = "sha256:" + hashlib.sha256(
        json.dumps(
            identity_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert authorization.authorization_id(
        raw_sha256=authorization_sha256
    ) == expected_authorization_id


def legacy_transition_authority_snapshots_are_mode_safe_and_git_exact(
    tmp_path: Path,
) -> None:
    previous_umask = os.umask(0o002)
    try:
        repository, _, _ = _initialize_transition_repository(tmp_path)
    finally:
        os.umask(previous_umask)
    for relative_path in _TRANSITION_AUTHORITY_RELATIVE_PATHS:
        path = repository / relative_path
        metadata = path.lstat()
        assert convergence_module.stat.S_ISREG(metadata.st_mode)
        assert metadata.st_nlink == 1
        assert metadata.st_uid in {0, os.geteuid()}
        assert convergence_module.stat.S_IMODE(metadata.st_mode) & 0o022 == 0
        snapshot = convergence_module._read_regular_snapshot(path)
        convergence_module._require_authority_file_snapshot(
            snapshot,
            repository_root=repository,
            expected_relative_path=relative_path,
        )
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative_path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        assert snapshot.raw == committed


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
@pytest.mark.parametrize("unsafe_mode", (0o664, 0o646))
def legacy_transition_authority_rejects_group_or_other_write_permission(
    tmp_path: Path,
    relative_path: str,
    unsafe_mode: int,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    path = repository / relative_path
    path.chmod(unsafe_mode)

    errors = _validate_transition_repository(repository)

    assert any(
        path.name in error and "authority file is group-or-other writable" in error
        for error in errors
    )


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_wrong_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    target = (repository / relative_path).absolute()
    real_reader = convergence_module._read_regular_snapshot

    def wrong_owner_snapshot(
        path: Path,
        *,
        maximum_bytes: int = MAX_EVIDENCE_SNAPSHOT_BYTES,
    ) -> SimpleNamespace:
        snapshot = real_reader(Path(path), maximum_bytes=maximum_bytes)
        if snapshot.path != target:
            return snapshot
        return SimpleNamespace(
            raw=snapshot.raw,
            path=snapshot.path,
            uid=max(1, os.geteuid() + 1),
            mode=snapshot.mode,
        )

    monkeypatch.setattr(
        convergence_module,
        "_read_regular_snapshot",
        wrong_owner_snapshot,
    )

    errors = _validate_transition_repository(repository)

    assert any(
        target.name in error and "authority file owner mismatch" in error
        for error in errors
    )


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_final_symlink(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    path = repository / relative_path
    target = tmp_path / f"external-{path.name}"
    shutil.copy2(path, target)
    path.unlink()
    path.symlink_to(target)

    errors = _validate_transition_repository(repository)

    assert any(
        path.name in error and "expected a regular nonsymlink file" in error
        for error in errors
    )


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_relocated_lexical_path(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    relocated = tmp_path / "relocated" / Path(relative_path).name
    relocated.parent.mkdir()
    shutil.copy2(repository / relative_path, relocated)
    relocated.chmod(0o644)
    snapshot = convergence_module._read_regular_snapshot(relocated)

    with pytest.raises(ValueError, match="must use its lexical repository path"):
        convergence_module._require_authority_file_snapshot(
            snapshot,
            repository_root=repository,
            expected_relative_path=relative_path,
        )


def legacy_transition_authority_rejects_symlinked_repository_parent(
    tmp_path: Path,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    alias = tmp_path / "repository-alias"
    alias.symlink_to(repository, target_is_directory=True)

    errors = _validate_transition_repository(alias)

    assert any("path contains a symlink or non-directory" in error for error in errors)


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_hardlink(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    path = repository / relative_path
    backing = tmp_path / f"hardlink-{path.name}"
    shutil.copy2(path, backing)
    path.unlink()
    os.link(backing, path)

    errors = _validate_transition_repository(repository)

    assert any(
        path.name in error and "expected a single-link evidence file" in error
        for error in errors
    )


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_path_swap_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    path = repository / relative_path
    replacement = tmp_path / f"replacement-{path.name}"
    shutil.copy2(path, replacement)
    target_inode = path.stat().st_ino
    real_read = os.read
    real_fstat = os.fstat
    swapped = False

    def swapping_read(descriptor: int, amount: int) -> bytes:
        nonlocal swapped
        result = real_read(descriptor, amount)
        if (
            result
            and not swapped
            and real_fstat(descriptor).st_ino == target_inode
        ):
            swapped = True
            path.unlink()
            replacement.rename(path)
        return result

    monkeypatch.setattr(convergence_module.os, "read", swapping_read)

    errors = _validate_transition_repository(repository)

    assert swapped is True
    assert any(
        path.name in error and "changed during bounded read" in error
        for error in errors
    )


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def legacy_transition_authority_rejects_descriptor_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    path = repository / relative_path
    target_inode = path.stat().st_ino
    real_fstat = os.fstat
    target_fstat_calls = 0

    def growing_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal target_fstat_calls
        observed = real_fstat(descriptor)
        if observed.st_ino != target_inode or not convergence_module.stat.S_ISREG(
            observed.st_mode
        ):
            return observed
        target_fstat_calls += 1
        if target_fstat_calls != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(convergence_module.os, "fstat", growing_fstat)

    errors = _validate_transition_repository(repository)

    assert target_fstat_calls >= 2
    assert any(
        path.name in error and "changed during bounded read" in error
        for error in errors
    )


def legacy_acceptance_transition_rejects_dirty_working_manifest_snapshot(
    tmp_path: Path,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    manifest_path = repository / _TRANSITION_AUTHORITY_RELATIVE_PATHS[-1]
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    consumed_blobs = {
        relative_path: (repository / relative_path).read_bytes()
        for relative_path in ACCEPTANCE_CHILD_CHANGED_PATHS
    }
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        consumed_acceptance_blobs=consumed_blobs,
        **_transition_lifecycle_kwargs(repository),
    )

    assert any(
        f"consumed_blobs.{_TRANSITION_AUTHORITY_RELATIVE_PATHS[-1]}: "
        "does not match acceptance HEAD" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("artifact", "field", "error_fragment"),
    (
        ("root", "pinned_at_ms", "pinned_at_ms: expected integer"),
        ("witness", "observed_at_ms", "observed_at_ms: expected integer"),
        ("witness", "expires_at_ms", "expires_at_ms: expected integer"),
        ("profile", "created_at", "created_at: expected positive finite"),
        ("profile", "lifecycle_generation", "lifecycle_generation: expected integer"),
        ("anchor", "generation", "anchor.generation: expected integer"),
        ("anchor", "updated_at_ns", "anchor.updated_at_ns: expected integer"),
        ("did_state", "generation", "did_state.generation: expected integer"),
        ("did_state", "updated_at_ns", "did_state.updated_at_ns: expected integer"),
        ("authorization", "authorized_at_ms", "authorized_at_ms: expected integer"),
        ("reviewer", "generation", "reviewer.generation: expected integer"),
    ),
)
def test_lifecycle_numeric_fields_reject_boolean_values(
    tmp_path: Path,
    artifact: str,
    field: str,
    error_fragment: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    root_path = (
        repository
        / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    witness_path = (
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    authorization_path = (
        repository
        / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    root_snapshot = convergence_module.load_local_profile_lifecycle_root_pin(
        root_path
    )
    witness_snapshot = convergence_module.load_local_operator_lifecycle_witness(
        witness_path
    )
    final_values = _transition_lifecycle_kwargs(repository)["expected_final_values"]
    assert isinstance(final_values, dict)

    if artifact == "root":
        payload = json.loads(json.dumps(root_snapshot.payload))
        payload[field] = False
        errors = convergence_module.validate_local_profile_lifecycle_root_pin(
            payload,
            expected_root_identity_did=root_snapshot.root_identity_did,
        )
    elif artifact in {"witness", "profile", "anchor", "did_state"}:
        payload = json.loads(json.dumps(witness_snapshot.payload))
        target = payload if artifact == "witness" else payload[artifact]
        assert isinstance(target, dict)
        target[field] = False
        errors = convergence_module.validate_local_operator_lifecycle_witness(
            payload,
            root_identity_did=root_snapshot.root_identity_did,
            expected_final_values=final_values,
        )
    else:
        payload = _load(authorization_path)
        target = payload if artifact == "authorization" else payload["reviewer"]
        assert isinstance(target, dict)
        target[field] = False
        errors = convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
            payload
        ).validate(
            lifecycle_witness=witness_snapshot,
            root_pin=root_snapshot,
            expected_final_values=final_values,
        )
    assert any(error_fragment in error for error in errors)


def test_lifecycle_witness_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    witness_path = (
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    raw = witness_path.read_text(encoding="utf-8")
    witness_path.write_text(
        '{"schema":"duplicate-must-fail",' + raw.lstrip()[1:],
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key: schema"):
        convergence_module.load_local_operator_lifecycle_witness(witness_path)


@pytest.mark.parametrize(
    "signature_field",
    ("active_key_signature", "root_signature"),
)
def test_lifecycle_witness_requires_both_ed25519_signatures(
    tmp_path: Path,
    signature_field: str,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    root_snapshot = convergence_module.load_local_profile_lifecycle_root_pin(
        repository
        / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    witness_snapshot = convergence_module.load_local_operator_lifecycle_witness(
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    payload = json.loads(json.dumps(witness_snapshot.payload))
    payload[signature_field] = base64.b64encode(b"\0" * 64).decode("ascii")
    final_values = _transition_lifecycle_kwargs(repository)["expected_final_values"]
    assert isinstance(final_values, dict)

    errors = convergence_module.validate_local_operator_lifecycle_witness(
        payload,
        root_identity_did=root_snapshot.root_identity_did,
        expected_final_values=final_values,
    )

    assert any(
        f"{signature_field}: cryptographic verification failed" in error
        for error in errors
    )


def test_authorization_v2_rejects_witness_drift_even_with_well_typed_digest(
    tmp_path: Path,
) -> None:
    repository, preparation_head, _ = _initialize_transition_repository(tmp_path)
    root_snapshot = convergence_module.load_local_profile_lifecycle_root_pin(
        repository
        / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    witness_snapshot = convergence_module.load_local_operator_lifecycle_witness(
        repository
        / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    authorization_path = (
        repository
        / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    payload = _load(authorization_path)
    reviewer = payload["reviewer"]
    assert isinstance(reviewer, dict)
    reviewer["witness_sha256"] = "sha256:" + ("0" * 64)
    root_head = subprocess.run(
        ["git", "rev-parse", f"{preparation_head}^"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root_tree = subprocess.run(
        ["git", "rev-parse", f"{root_head}^{{tree}}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    final_values = _transition_lifecycle_kwargs(repository)["expected_final_values"]
    assert isinstance(final_values, dict)

    errors = convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
        payload
    ).validate(
        lifecycle_witness=witness_snapshot,
        root_pin=root_snapshot,
        expected_source_head=root_head,
        expected_source_tree=root_tree,
        expected_final_values=final_values,
    )

    assert any(
        "reviewer.witness_sha256: witness equality mismatch" in error
        for error in errors
    )
    assert any("reviewer.signature: cryptographic verification failed" in error for error in errors)


@pytest.mark.parametrize(
    ("relative_path", "error_fragment"),
    (
        (
            convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH,
            "root_pin: consumed bytes do not match P",
        ),
        (
            convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH,
            "witness: consumed bytes do not match P",
        ),
        (
            convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH,
            "fallback_authorization: consumed bytes do not match P",
        ),
    ),
)
def legacy_acceptance_transition_rejects_dirty_lifecycle_bytes(
    tmp_path: Path,
    relative_path: str,
    error_fragment: str,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle_kwargs = _transition_lifecycle_kwargs(repository)
    dirty_path = repository / relative_path
    dirty_path.write_bytes(dirty_path.read_bytes() + b"\n")
    if relative_path == convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH:
        lifecycle_kwargs["lifecycle_root_pin_raw"] = dirty_path.read_bytes()
    elif relative_path == convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH:
        lifecycle_kwargs["lifecycle_witness_raw"] = dirty_path.read_bytes()
    else:
        lifecycle_kwargs["fallback_authorization_raw"] = dirty_path.read_bytes()

    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        **lifecycle_kwargs,
    )

    assert any(error_fragment in error for error in errors)


@pytest.mark.parametrize(
    ("preparation_updates", "error_fragment"),
    (
        ({"goal_id": "ASE3-G999"}, "goal_id: expected ASE3-G010"),
        ({"unauthorized_top_level": True}, "exact @1 top-level population required"),
    ),
)
def legacy_acceptance_transition_rejects_semantically_invalid_preparation_manifest(
    tmp_path: Path,
    preparation_updates: dict[str, object],
    error_fragment: str,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(
            tmp_path,
            preparation_manifest_updates=preparation_updates,
        )
    )
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        **_transition_lifecycle_kwargs(repository),
    )

    assert any(error_fragment in error for error in errors)


def legacy_acceptance_transition_rejects_dirty_consumed_receipt(tmp_path: Path) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_relative_path = OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS[1]
    (repository / dirty_relative_path).write_text(
        '{"dirty_worktree_value":true}\n',
        encoding="utf-8",
    )
    consumed_blobs = {
        relative_path: (repository / relative_path).read_bytes()
        for relative_path in ACCEPTANCE_CHILD_CHANGED_PATHS
    }

    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        consumed_acceptance_blobs=consumed_blobs,
        **_transition_lifecycle_kwargs(repository),
    )

    assert any(
        f"consumed_blobs.{dirty_relative_path}: does not match acceptance HEAD"
        in error
        for error in errors
    )


def legacy_acceptance_transition_rejects_unauthorized_manifest_drift(
    tmp_path: Path,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    manifest_path = (
        repository
        / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
        / MANIFEST_FILENAME
    )
    manifest = _load(manifest_path)
    components = manifest["components"]
    assert isinstance(components, dict)
    components["current_main_baseline.json"] = "sha256:" + ("0" * 64)
    _write(manifest_path, manifest)
    subprocess.run(["git", "add", str(manifest_path)], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "--amend", "--no-edit"],
        cwd=repository,
        check=True,
    )
    acceptance_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=acceptance_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        **_transition_lifecycle_kwargs(repository),
    )

    assert any(
        "manifest_transformation.components.current_main_baseline.json" in error
        for error in errors
    )


def legacy_acceptance_transition_rejects_extra_path_and_board_prose(
    tmp_path: Path,
) -> None:
    repository, preparation_head, preparation_tree = (
        _initialize_transition_repository(tmp_path)
    )
    board = repository / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    board.write_text(
        board.read_text(encoding="utf-8") + "\nunauthorized acceptance prose\n",
        encoding="utf-8",
    )
    extra = repository / "unexpected.txt"
    extra.write_text("unexpected\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "forged acceptance successor"],
        cwd=repository,
        check=True,
    )
    forged_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    errors = validate_acceptance_child_transition(
        repo_root=repository,
        acceptance_head=forged_head,
        preparation_head=preparation_head,
        preparation_tree=preparation_tree,
        **_transition_lifecycle_kwargs(repository),
    )
    assert any("exact direct single parent" in error for error in errors)
    assert any("changed_paths" in error for error in errors)
    assert any("taskboard" in error for error in errors)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "current_main_baseline.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            '  "board_namespace":',
            '  "schema": "duplicate-must-fail",\n  "board_namespace":',
            1,
        ),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("duplicate JSON key: schema" in error for error in report.errors)


def test_rebound_recorded_tree_must_match_the_git_object(tmp_path: Path) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "current_main_baseline.json"
    payload = _load(path)
    upstream = payload["upstream_main"]
    assert isinstance(upstream, dict)
    upstream["tree"] = "0" * 40
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(
        root,
        repo_root=REPO_ROOT,
        check_repository=True,
    )

    assert report.valid is False
    assert "repository_binding.upstream_main.tree: Git identity mismatch" in report.errors


@pytest.mark.parametrize(
    ("field_path", "value", "error_fragment"),
    (
        (("goal_id",), "ASE3-G999", "goal_id: expected ASE3-G010"),
        (("created_at",), "not-a-timestamp", "created_at: expected UTC timestamp"),
        (
            ("integration_seed_commit",),
            "0" * 40,
            "integration_seed_commit: baseline mismatch",
        ),
        (
            ("integration_seed_tree",),
            "0" * 40,
            "integration_seed_tree: baseline mismatch",
        ),
        (
            ("population", "rescue_commits"),
            35,
            "population.rescue_commits: expected 36",
        ),
        (
            ("population", "rescue_changed_paths"),
            34,
            "population.rescue_changed_paths: expected 35",
        ),
        (("population", "v2_tasks"), 7, "population.v2_tasks: expected 8"),
        (
            ("population", "historical_contradictions"),
            4,
            "population.historical_contradictions: expected 5",
        ),
        (
            ("population", "v3_seed_tasks"),
            14,
            "population.v3_seed_tasks: expected 15",
        ),
        (
            ("population", "v3_seed_goals"),
            8,
            "population.v3_seed_goals: expected 9",
        ),
        (
            ("completion_rules", "historical_status_or_receipt_satisfies_v3"),
            True,
            "historical_status_or_receipt_satisfies_v3: expected False",
        ),
        (
            ("completion_rules", "branch_local_commit_satisfies_v3"),
            True,
            "branch_local_commit_satisfies_v3: expected False",
        ),
        (
            ("completion_rules", "queue_drain_satisfies_goal_completion"),
            True,
            "queue_drain_satisfies_goal_completion: expected False",
        ),
        (
            ("completion_rules", "current_tree_acceptance_required"),
            False,
            "current_tree_acceptance_required: expected True",
        ),
        (
            ("completion_rules", "forced_residual_scan_required"),
            False,
            "forced_residual_scan_required: expected True",
        ),
        (
            ("downstream_rules", "required_ancestor"),
            "0" * 40,
            "downstream_rules.required_ancestor: expected",
        ),
        (
            ("downstream_rules", "merge_target_branch"),
            "other",
            "downstream_rules.merge_target_branch: expected",
        ),
        (
            ("downstream_rules", "rescue_disposition_required_before_use"),
            False,
            "rescue_disposition_required_before_use: expected True",
        ),
        (
            ("downstream_rules", "fresh_validation_receipt_required_per_task"),
            False,
            "fresh_validation_receipt_required_per_task: expected True",
        ),
        (
            ("downstream_rules", "protected_source_checkout_may_be_modified"),
            True,
            "protected_source_checkout_may_be_modified: expected False",
        ),
    ),
)
def test_rebound_manifest_fields_fail_closed(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / MANIFEST_FILENAME
    payload = _load(path)
    block: dict[str, object] = payload
    for field in field_path[:-1]:
        child = block[field]
        assert isinstance(child, dict)
        block = child
    block[field_path[-1]] = value
    _write(path, payload)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "extra_key"),
    (
        ("population", "unreviewed_count"),
        ("completion_rules", "soft_completion_allowed"),
        ("downstream_rules", "unreviewed_effect_allowed"),
    ),
)
def test_manifest_policy_and_count_objects_reject_extra_keys(
    tmp_path: Path,
    section: str,
    extra_key: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / MANIFEST_FILENAME
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[extra_key] = True
    _write(path, payload)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(f"convergence_manifest.{section}: population mismatch" in error for error in report.errors)


@pytest.mark.parametrize(
    ("section", "field", "value", "error_fragment"),
    (
        (
            "worktree",
            "isolated_from_source_checkout",
            False,
            "isolated_from_source_checkout: must be true",
        ),
        ("worktree", "branch", "other", "worktree.branch: must equal"),
        (
            "protected_source_checkout",
            "modified_by_bootstrap",
            True,
            "modified_by_bootstrap: must be false",
        ),
        (
            "state_namespace",
            "fresh_for_board",
            False,
            "fresh_for_board: must be true",
        ),
        (
            "state_namespace",
            "historical_import_allowed",
            True,
            "historical_import_allowed: must be false",
        ),
        (
            "downstream_binding",
            "changed_revision_requires_fresh_validation",
            False,
            "changed_revision_requires_fresh_validation: must be true",
        ),
    ),
)
def test_rebound_critical_worktree_receipt_fields_fail_closed(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    error_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "clean_integration_worktree_receipt.json"
    payload = _load(path)
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(error_fragment in error for error in report.errors)


def test_rebound_rescue_disposition_rejects_unknown_target_task(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    files = payload["files"]
    assert isinstance(files, list)
    first_rewrite = next(
        item
        for item in files
        if isinstance(item, dict) and item.get("disposition") == "rewrite"
    )
    first_rewrite["target_tasks"] = ["ASE3-999"]
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any("target_tasks: unknown task 'ASE3-999'" in error for error in report.errors)


@pytest.mark.parametrize("field", ("merge_base", "rescue_head", "current_seed"))
def test_rebound_rescue_top_level_identities_match_the_baseline(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    payload[field] = "0" * 40
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(
        f"rescue_artifact_dispositions.{field}: baseline mismatch" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("population", "mutation", "expected_fragment"),
    (
        ("commits", "replace-with-garbage", "commits[0]: expected object"),
        ("files", "replace-with-garbage", "files[0]: expected object"),
        ("commits", "append-extra-object", "commits: expected 36, got 37"),
        ("files", "append-extra-object", "files: expected 35, got 36"),
    ),
)
def test_rescue_populations_reject_non_objects_and_extra_elements(
    tmp_path: Path,
    population: str,
    mutation: str,
    expected_fragment: str,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    path = root / "rescue_artifact_dispositions.json"
    payload = _load(path)
    entries = payload[population]
    assert isinstance(entries, list)
    if mutation == "replace-with-garbage":
        entries[0] = "not-an-object"
    else:
        first = entries[0]
        assert isinstance(first, dict)
        entries.append(dict(first))
    _write(path, payload)
    _rebind_component_digest(root, path.name)

    report = validate_convergence_artifacts(root, check_repository=False)

    assert report.valid is False
    assert any(expected_fragment in error for error in report.errors)


def test_repository_validation_is_portable_to_an_alternate_descendant_worktree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)

    baseline_path = root / "current_main_baseline.json"
    baseline = _load(baseline_path)
    original = baseline["original_checkout"]
    seed = baseline["integration_seed"]
    assert isinstance(original, dict)
    assert isinstance(seed, dict)
    original["path"] = "/historical/source/checkout"
    _write(baseline_path, baseline)
    _rebind_component_digest(root, baseline_path.name)

    receipt_path = root / "clean_integration_worktree_receipt.json"
    receipt = _load(receipt_path)
    source = receipt["protected_source_checkout"]
    worktree = receipt["worktree"]
    assert isinstance(source, dict)
    assert isinstance(worktree, dict)
    source["path"] = original["path"]
    worktree["path"] = "/historical/integration/worktree"
    _write(receipt_path, receipt)
    _rebind_component_digest(root, receipt_path.name)

    portable = tmp_path / "portable-repository"
    subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            str(REPO_ROOT),
            str(portable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    portable_taskboard_path = portable / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    portable_taskboard_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TASKBOARD_PATH, portable_taskboard_path)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    launch = incident["launch"]
    assert isinstance(launch, dict)
    seed_tree = str(seed["tree"])
    descendant = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Portable Validation",
            "-c",
            "user.email=portable@example.invalid",
            "commit-tree",
                seed_tree,
                "-p",
                str(launch["launch_head"]),
            "-m",
            "portable descendant",
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "symbolic-ref", "HEAD", "refs/heads/portable-descendant"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "update-ref", "HEAD", descendant],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )

    assert subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "portable-descendant"
    assert Path(str(worktree["path"])).resolve() != portable.resolve()
    assert not Path(str(source["path"])).exists()

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=portable_taskboard_path,
    )

    assert report.valid is True, report.errors
    assert report.errors == ()


def test_recovery_requires_the_failed_candidate_rescue_ref(tmp_path: Path) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    failed = recovery["failed_attempt"]
    assert isinstance(failed, dict)
    rescue_branch = str(failed["rescue_branch"])
    for reference in (
        f"refs/heads/{rescue_branch}",
        f"refs/remotes/origin/{rescue_branch}",
    ):
        subprocess.run(
            ["git", "update-ref", "-d", reference],
            cwd=portable,
            check=True,
            capture_output=True,
            text=True,
        )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("ASE3-019.rescue_branch" in error for error in report.errors)


def test_recovery_requires_the_exact_attempt2_branch_ref(tmp_path: Path) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    incident = _load(root / SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME)
    prior_seed = incident["prior_attempt_seed"]
    assert isinstance(prior_seed, dict)
    branch = str(prior_seed["attempt_2_branch"])
    for reference in (
        f"refs/heads/{branch}",
        f"refs/remotes/origin/{branch}",
    ):
        subprocess.run(
            ["git", "update-ref", "-d", reference],
            cwd=portable,
            check=True,
            capture_output=True,
            text=True,
        )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("attempt_2_branch: exact ref unavailable" in error for error in report.errors)


def test_recovery_rejects_conflicting_exact_named_rescue_refs(
    tmp_path: Path,
) -> None:
    root, portable, taskboard = _portable_recovery_repository(tmp_path)
    recovery = _load(root / FALSE_COMPLETION_RECOVERY_FILENAME)
    source = recovery["source"]
    failed = recovery["failed_attempt"]
    assert isinstance(source, dict)
    assert isinstance(failed, dict)
    rescue_branch = str(failed["rescue_branch"])
    subprocess.run(
        [
            "git",
            "update-ref",
            f"refs/remotes/origin/{rescue_branch}",
            str(failed["implementation_commit"]),
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "update-ref",
            f"refs/heads/{rescue_branch}",
            str(source["recovery_parent_head"]),
        ],
        cwd=portable,
        check=True,
        capture_output=True,
        text=True,
    )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any("exact named refs disagree" in error for error in report.errors)


def test_recovery_rejects_a_head_containing_the_failed_candidate(
    tmp_path: Path,
) -> None:
    root, portable, taskboard = _portable_recovery_repository(
        tmp_path,
        include_failed_candidate_parent=True,
    )

    report = validate_convergence_artifacts(
        root,
        repo_root=portable,
        check_repository=True,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        "ASE3-019.merge_dispatched: candidate is an ancestor of HEAD" in error
        for error in report.errors
    )


def test_scheduler_config_loads_and_binds_the_v3_board_structurally() -> None:
    board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)

    assert board.board_namespace == BOARD_NAMESPACE
    assert board.task_prefix == "ASE3-"
    assert board.max_lanes == 3
    assert board.strict_task_sharding is True
    assert board.merge_target_branch == "agent/prompt-self-improvement-v3"
    assert board.validator_path.endswith("prompt_v3_convergence.py")
    for filename in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME):
        relative = (
            "data/agent_supervisor/prompt_only_self_improvement_v3/convergence/"
            + filename
        )
        assert relative in board.protected_paths


def test_program_expansion_projection_is_exact_and_dormant() -> None:
    config = _load(CONFIG_PATH)
    initial = config["initial_projection"]
    groups = config["task_groups"]
    dependencies = config["task_dependencies"]
    contract_layering = config["neutral_contract_layering"]
    identity_acceptance = config["protected_identity_acceptance"]
    native_authorization = config["protected_native_dependency_launch_authorization"]
    native_acceptance = config["protected_native_dependency_acceptance"]
    duckdb_acceptance = config["protected_duckdb_connection_policy_acceptance"]
    native_duckdb_sequence = config[
        "protected_native_duckdb_acceptance_sequence"
    ]
    activation = config["protected_runtime_activation"]
    refill = config["refill_policy"]
    monitor = config["monitor_policy"]
    assert isinstance(initial, dict)
    assert isinstance(groups, dict)
    assert isinstance(dependencies, dict)
    assert isinstance(contract_layering, dict)
    assert isinstance(identity_acceptance, dict)
    assert isinstance(native_authorization, dict)
    assert isinstance(native_acceptance, dict)
    assert isinstance(duckdb_acceptance, dict)
    assert isinstance(native_duckdb_sequence, dict)
    assert isinstance(activation, dict)
    assert isinstance(refill, dict)
    assert isinstance(monitor, dict)

    canonical = initial["canonical_task_ids"]
    assert initial["task_count"] == 30
    assert isinstance(canonical, list)
    assert len(canonical) == len(set(canonical)) == 30
    assert initial["noncanonical_transition_task_ids"] == ["ASE3-022"]
    assert set(dependencies) == set(canonical)
    assert dependencies["ASE3-008"] == ["ASE3-006", "ASE3-020", "ASE3-021"]
    assert dependencies["ASE3-013"] == ["ASE3-008", "ASE3-012", "ASE3-026"]
    assert dependencies["ASE3-031"] == ["ASE3-030"]
    assert dependencies["ASE3-032"] == ["ASE3-031"]
    assert dependencies["ASE3-033"] == ["ASE3-000"]
    assert {
        task_id
        for task_ids in groups.values()
        for task_id in task_ids
    } == set(canonical)
    assert config["acceptance_prerequisites"] == {
        "ASE3-023": ["ASE3-030", "ASE3-031", "ASE3-032"],
        "ASE3-022": ["ASE3-030", "ASE3-031", "ASE3-032"],
    }
    assert contract_layering == convergence_module._CONTRACT_LAYERING_POLICY
    assert (
        convergence_module._mapping_contract_sha256(contract_layering)
        == convergence_module._CONTRACT_LAYERING_POLICY_CONFIG_SHA256
    )
    accepted_inventory = contract_layering["accepted_tree_inventory"]
    assert isinstance(accepted_inventory, dict)
    assert accepted_inventory["source_task_id"] == "ASE3-023"
    assert accepted_inventory["roadmap_fixed_edge_or_importer_count_allowed"] is False
    assert contract_layering["ambient_effect_registry_allowed"] is False
    assert contract_layering["neutral_import_time_io_allowed"] is False
    assert contract_layering["capsule_security_critical_paths"] == [
        "ipfs_accelerate_py/agent_supervisor/entrypoints/contracts.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/execution_plan.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py",
        "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py",
    ]
    assert contract_layering["daemon_runner_authorization_baseline"] == {
        "test_path": (
            "test/api/test_agent_supervisor_implementation_daemon_runner.py"
        ),
        "stale_test_name": (
            "test_daemon_resolves_relative_worktree_root_for_runner_workspace"
        ),
        "ambient_only_route_fixture_allowed": False,
        "canonical_signed_accepted_route_authorization_and_binding_required": True,
        "accepted_public_artifact_mode": "0400",
        "owned_regular_nonsymlink_required": True,
        "private_signer_material_secure_mode_required": True,
        "complete_file_must_pass": True,
        "test_deselection_allowed": False,
        "route_verifier_bypass_allowed": False,
        "route_weakening_allowed": False,
    }
    assert contract_layering["downstream_task_id"] == "ASE3-028"
    assert contract_layering["downstream_requires_accepted_ase3_029"] is True
    assert identity_acceptance == {
        "task_id": "ASE3-030",
        "status": "reserved",
        "receipt_path": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        "receipt_schema": HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
        "artifact_phase": "A",
        "sequence_phase": "A030",
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
    }
    assert native_authorization == {
        "task_id": "ASE3-031",
        "status": "reserved",
        "authorization_path": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
        "authorization_schema": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
        "artifact_phase": "P",
        "sequence_phase": "P031",
        "signed_by_accepted_local_profile_required": True,
        "accepted_authorization_id_exact_match_required": True,
        "inspection_evidence_is_authority": False,
        "authorization_may_claim_launch_effect": False,
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-031"],
        "required_before_runtime_effects": ["ASE3-023"],
    }
    assert native_acceptance == {
        "task_id": "ASE3-031",
        "status": "reserved",
        "receipt_path": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        "receipt_schema": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
        "artifact_phase": "A",
        "sequence_phase": "A031",
        "authorization_path": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
        "accepted_authorization_id_exact_match_required": True,
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
    }
    assert duckdb_acceptance == {
        "task_id": "ASE3-032",
        "status": "reserved",
        "receipt_path": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        "receipt_schema": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
        "artifact_phase": "A",
        "sequence_phase": "A032",
        "requires_prior_acceptance_tasks": ["ASE3-030", "ASE3-031"],
        "strict_validator_and_manifest_binding_required": True,
        "required_before_task_acceptance": ["ASE3-023", "ASE3-022"],
    }
    assert native_duckdb_sequence == (
        convergence_module._NATIVE_DUCKDB_ACCEPTANCE_SEQUENCE
    )
    protected_paths = config["protected_paths"]
    assert isinstance(protected_paths, list)
    for path in (
        HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
        NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
        DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
    ):
        assert protected_paths.count(path) == 1
    assert activation == {
        "task_id": "ASE3-026",
        "status": "blocked",
        "receipt_path": (
            "data/agent_supervisor/prompt_only_self_improvement_v3/"
            "convergence/protected_runtime_activation_receipt.json"
        ),
        "receipt_schema": PROTECTED_RUNTIME_ACTIVATION_AUTHORIZATION_SCHEMA,
        "receipt_phase": "pre_effect_authorization",
        "authorization_may_claim_activation_effect": False,
        "post_activation_observation_receipt_path": (
            PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_RECEIPT_RELATIVE_PATH
        ),
        "post_activation_observation_receipt_schema": (
            PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_SCHEMA
        ),
        "post_activation_observation_required_for_completion": True,
        "post_activation_required_observations": [
            "lifecycle_process_birth",
            "lifecycle_lease_fence_heartbeat_and_cursor",
            "monitor_process_birth",
            "monitor_lease_fence_heartbeat_and_cursor",
            "refill_append_recompile_dispatch_or_adoption",
        ],
        "one_generation_cas_lease_required": True,
        "operator_review_required": True,
        "strict_validator_and_manifest_binding_required": True,
    }
    assert config["strict_task_sharding"] is True
    assert config["objective_refill_enabled"] is False
    assert config["codebase_refill_enabled"] is False
    assert refill["enable_after_task"] == "ASE3-026"
    assert refill["prompt_program_refill_enabled"] is False
    assert refill["saga_cursor_states"] == [
        "EVALUATING",
        "APPEND_RESERVED",
        "APPENDED",
        "PLAN_INVALIDATED",
        "RECOMPILED",
        "DISPATCHED",
        "ADOPTED",
    ]
    assert refill["saga_terminal_states"] == ["DISPATCHED", "ADOPTED"]
    assert refill["saga_terminal_states_are_alternatives"] is True
    assert refill["saga_cursor_durable"] is True
    assert refill["monitor_phase_deadlines_required"] is True
    assert monitor["enabled"] is False
    assert monitor["detached"] is True
    assert monitor["activation_task_id"] == "ASE3-026"
    assert monitor["durable_guardian"] == "ReviewedHostNamespaceReconciler"
    assert monitor["guardian_scope"] == "host_namespace"
    assert monitor["guardian_review_required"] is True
    assert monitor["semantic_progress_source"] == "configured_board_scheduler"
    assert monitor["running_join_fields"] == [
        "lifecycle_process_birth",
        "lifecycle_lease",
        "lifecycle_fence",
        "lifecycle_heartbeat",
        "lifecycle_event_cursor",
        "monitor_process_birth",
        "monitor_lease",
        "monitor_fence",
        "monitor_heartbeat",
        "monitor_event_cursor",
    ]
    assert monitor["running_requires_joined_lifecycle_monitor_evidence"] is True
    assert monitor["immutable_history_and_cursor_vectors_required"] is True
    assert monitor["unknown_outcome_effect_replay_authorized"] is False
    assert monitor["canary_task_id"] == "ASE3-013"
    assert monitor["canary_observation_seconds"] == 900
    assert monitor["post_recovery_continuous_health_seconds"] == 900
    assert monitor["continuous_health_required"] is True
    assert monitor["monotonic_elapsed_receipt_required"] is True
    assert monitor["prompt_may_override_observation_window"] is False


def test_ase3_029_daemon_runner_repair_scope_is_exact() -> None:
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )
    task = tasks["ASE3-029"]
    outputs = convergence_module._taskboard_csv(task, "outputs")
    predicted = convergence_module._taskboard_csv(task, "predicted files")
    validation = str(task["validation"])
    daemon_runner_test = (
        "test/api/test_agent_supervisor_implementation_daemon_runner.py"
    )
    supervisor_runner_test = (
        "test/api/test_agent_supervisor_implementation_supervisor_runner.py"
    )

    assert daemon_runner_test in outputs
    assert daemon_runner_test in predicted
    assert validation.split().count(daemon_runner_test) == 1
    assert supervisor_runner_test not in outputs
    assert supervisor_runner_test not in predicted
    assert validation.split().count(supervisor_runner_test) == 1


@pytest.mark.parametrize(
    "task_id",
    (
        "ASE3-024",
        "ASE3-025",
        "ASE3-028",
        "ASE3-029",
        "ASE3-030",
        "ASE3-031",
        "ASE3-032",
    ),
)
def test_program_expansion_task_identity_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = f"## {task_id} "
    assert text.count(needle) == 1
    taskboard.write_text(
        text.replace(needle, f"## {task_id} Tampered ", 1),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        f"program_plan_expansion.{task_id}.title" in error
        or f"program_plan_expansion.{task_id}.canonical_task_cid" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    "task_id",
    (
        "ASE3-008",
        "ASE3-009",
        "ASE3-010",
        "ASE3-011",
        "ASE3-012",
        "ASE3-013",
        "ASE3-014",
        "ASE3-020",
        "ASE3-021",
        "ASE3-024",
        "ASE3-025",
        "ASE3-028",
        "ASE3-029",
        "ASE3-030",
        "ASE3-031",
        "ASE3-032",
    ),
)
def test_required_program_task_cannot_complete_without_evidence(
    tmp_path: Path,
    task_id: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = f"## {task_id} "
    start = text.index(needle)
    status_start = text.index("- Status: todo\n", start)
    next_task = text.find("\n## ASE3-", start + len(needle))
    assert next_task == -1 or status_start < next_task
    taskboard.write_text(
        text[:status_start]
        + "- Status: completed\n"
        + text[status_start + len("- Status: todo\n") :],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        f"program_plan_expansion.{task_id}.status" in error
        or f"program_plan_expansion.{task_id}.contract_sha256" in error
        for error in report.errors
    )


@pytest.mark.parametrize(
    ("task_id", "needle", "replacement"),
    (
        (
            "ASE3-024",
            "`llm_router` owns planning-provider route and final admission",
            "the prompt broker owns planning-provider route and final admission",
        ),
        (
            "ASE3-025",
            "DuckDB owns the authoritative program revision",
            "Markdown owns the authoritative program revision",
        ),
        (
            "ASE3-028",
            (
                "both callers may normalize bounded non-authoritative inputs and "
                "execute one typed router decision"
            ),
            "both callers may independently rerank the router decision",
        ),
        (
            "ASE3-029",
            (
                "Concrete lower `control.provider_attempt_store`, "
                "`control.profile_authority`, and `control.plan_execution_store` "
                "services alone own the displaced CAS persistence"
            ),
            "Entrypoints alone retain every concrete effect implementation",
        ),
        (
            "ASE3-030",
            "without importing `multiformats` or mutable repository/candidate code",
            "by importing user-site `multiformats` and mutable candidate code",
        ),
        (
            "ASE3-031",
            "Inspection is evidence only and must never mint launch authority.",
            "Inspection mints launch authority.",
        ),
        (
            "ASE3-032",
            "insert `lock_configuration=true` last",
            "leave configuration unlocked",
        ),
    ),
)
def test_program_expansion_critical_policy_is_sealed(
    tmp_path: Path,
    task_id: str,
    needle: str,
    replacement: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    assert text.count(needle) == 1
    taskboard.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert (
        f"program_plan_expansion.{task_id}.contract_sha256: "
        "exact metadata/prose required"
    ) in report.errors


def test_amended_task_identity_and_activation_dependency_are_pinned(
    tmp_path: Path,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "- Depends on: ASE3-005, ASE3-008, ASE3-026\n"
    assert text.count(needle) == 1
    taskboard.write_text(
        text.replace(needle, "- Depends on: ASE3-005, ASE3-008\n", 1),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert (
        "program_plan_expansion.ASE3-009.depends_on: exact expansion required"
        in report.errors
    )


@pytest.mark.parametrize(
    ("task_id", "needle", "replacement"),
    (
        (
            "ASE3-008",
            "- Depends on: ASE3-006, ASE3-020, ASE3-021",
            "- Depends on: ASE3-006, ASE3-020",
        ),
        (
            "ASE3-013",
            "- Depends on: ASE3-008, ASE3-012, ASE3-026",
            "- Depends on: ASE3-008, ASE3-012",
        ),
    ),
)
def test_monitor_strategy_direct_dependency_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
    needle: str,
    replacement: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    start = text.index(f"## {task_id} ")
    end = text.find("\n## ASE3-", start + 1)
    if end < 0:
        end = len(text)
    block = text[start:end]
    assert block.count(needle) == 1
    taskboard.write_text(
        text[:start] + block.replace(needle, replacement, 1) + text[end:],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert (
        f"program_plan_expansion.{task_id}.depends_on: exact expansion required"
        in report.errors
    )


@pytest.mark.parametrize(
    ("task_id", "needle", "replacement"),
    (
        (
            "ASE3-031",
            "- Depends on: ASE3-030",
            "- Depends on: ASE3-019",
        ),
        (
            "ASE3-032",
            "- Depends on: ASE3-031",
            "- Depends on: ASE3-030",
        ),
    ),
)
def test_native_duckdb_direct_dependency_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
    needle: str,
    replacement: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    start = text.index(f"## {task_id} ")
    end = text.find("\n## ASE3-", start + 1)
    assert end > start
    block = text[start:end]
    assert block.count(needle) == 1
    taskboard.write_text(
        text[:start] + block.replace(needle, replacement, 1) + text[end:],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert (
        f"program_plan_expansion.{task_id}.depends_on: exact population required"
        in report.errors
    )


@pytest.mark.parametrize("task_id", ("ASE3-019", "ASE3-022", "ASE3-023", "ASE3-027"))
def test_operator_protected_task_block_byte_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    start = text.index(f"## {task_id} ")
    end = text.find("\n## ASE3-", start + 1)
    assert end > start
    block = text[start:end]
    assert "\n\n" in block
    mutated_block = block.replace("\n\n", "\n \n", 1)
    taskboard.write_text(
        text[:start] + mutated_block + text[end:],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert (
        f"protected_task_block_bytes.{task_id}: protected task block bytes changed"
        in report.errors
    )


def test_operator_protected_task_block_seal_allows_only_exact_phase_statuses() -> None:
    boards = {"Q": TASKBOARD_PATH.read_bytes()}
    for phase in convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES[1:]:
        boards[phase] = convergence_module._status_only_sequential_phase_board(
            boards[convergence_module.SEQUENTIAL_PHASE_PARENT[phase]],
            phase,
        )

    for phase, raw in boards.items():
        text = raw.decode("utf-8")
        assert convergence_module._validate_protected_task_block_bytes(
            text,
            phase=phase,
        ) == []

        task_start = text.index("## ASE3-019 ")
        task_end = text.index("\n## ASE3-", task_start + 1)
        task_block = text[task_start:task_end]
        prose_mutation = (
            text[:task_start]
            + task_block.replace(
                "- Completion: manual",
                "- Completion: forged",
                1,
            )
            + text[task_end:]
        )
        assert any(
            "protected task block bytes changed" in error
            for error in convergence_module._validate_protected_task_block_bytes(
                prose_mutation,
                phase=phase,
            )
        )

        expected_status = convergence_module._sequential_task_statuses_after(phase)[
            "ASE3-019"
        ]
        wrong_status = "todo" if expected_status == "completed" else "completed"
        status_mutation = (
            text[:task_start]
            + task_block.replace(
                f"- Status: {expected_status}",
                f"- Status: {wrong_status}",
                1,
            )
            + text[task_end:]
        )
        assert any(
            f"ASE3-019.status: exact {phase} phase status required" in error
            for error in convergence_module._validate_protected_task_block_bytes(
                status_mutation,
                phase=phase,
            )
        )


@pytest.mark.parametrize(
    ("task_id", "needle", "replacement"),
    (
        (
            "ASE3-008",
            "ReviewedHostNamespaceReconciler",
            "ClientOwnedMonitorGuardian",
        ),
        (
            "ASE3-013",
            "The 900-second clock begins only after the final recovery",
            "The 900-second clock begins before recovery",
        ),
        (
            "ASE3-012",
            "no prompt-product launch-reachable path can call raw `duckdb.connect`",
            "prompt-product launch paths may call raw `duckdb.connect`",
        ),
        (
            "ASE3-020",
            "persist UNKNOWN, prohibit replay",
            "replay an unknown effect",
        ),
        (
            "ASE3-020",
            "Every prompt-product-reachable run-registry and runtime-history connection",
            "Only selected run-registry connections",
        ),
        (
            "ASE3-021",
            (
                "EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→"
                "RECOMPILED→DISPATCHED/ADOPTED"
            ),
            "EVALUATING→APPENDED→DISPATCHED",
        ),
        (
            "ASE3-026",
            "authorization_effect_observed: false",
            "authorization_effect_observed: true",
        ),
        (
            "ASE3-025",
            "every generated-board/planning-reachable connection",
            "selected generated-board connections",
        ),
        (
            "ASE3-031",
            "authorization_may_claim_launch_effect: false",
            "authorization_may_claim_launch_effect: true",
        ),
        (
            "ASE3-032",
            (
                "persistent-catalog seal covering "
                "schemas/tables/views/sequences/macros/custom types/indexes"
            ),
            "table-only catalog inventory",
        ),
    ),
)
def test_sealed_program_task_contract_mutation_fails_closed(
    tmp_path: Path,
    task_id: str,
    needle: str,
    replacement: str,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    start = text.index(f"## {task_id} ")
    end = text.find("\n## ASE3-", start + 1)
    if end < 0:
        end = len(text)
    block = text[start:end]
    assert needle in block
    taskboard.write_text(
        text[:start] + block.replace(needle, replacement, 1) + text[end:],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        f"program_plan_expansion.{task_id}.contract_sha256" in error
        for error in report.errors
    )


def test_protected_runtime_activation_stays_blocked_without_strict_receipt(
    tmp_path: Path,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "## ASE3-026 Authorize, activate, and observe the durable refill and "
        "autonomous monitor runtime\n\n- Status: blocked\n"
    )
    assert text.count(needle) == 1
    taskboard.write_text(
        text.replace(
            needle,
            needle.removesuffix("- Status: blocked\n") + "- Status: completed\n",
            1,
        ),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert any(
        "program_plan_expansion.ASE3-026" in error for error in report.errors
    )


def legacy_unvalidated_protected_runtime_activation_receipt_is_reserved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    (root / PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "ASE3-026.receipt: present without strict validation" in error
        for error in report.errors
    )


def legacy_partial_hermetic_identity_acceptance_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "convergence"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, root)
    (root / HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        root,
        check_repository=False,
        taskboard_path=TASKBOARD_PATH,
    )

    assert report.valid is False
    assert any(
        "partial population forbidden" in error
        for error in report.errors
    )


def legacy_hermetic_identity_completion_requires_reserved_receipt_binding(
    tmp_path: Path,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    task_start = text.index(
        "## ASE3-030 Seal hermetic control-plane identity dependency closure\n"
    )
    status_start = text.index("- Status: todo\n", task_start)
    taskboard.write_text(
        text[:status_start]
        + "- Status: completed\n"
        + text[status_start + len("- Status: todo\n") :],
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert report.valid is False
    assert (
        "program_plan_expansion.ASE3-030.status: completion requires atomic acceptance"
    ) in report.errors


def test_scheduler_dependency_projection_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    dependencies = config["task_dependencies"]
    assert isinstance(dependencies, dict)
    dependencies["ASE3-025"] = ["ASE3-004", "ASE3-023"]
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.task_dependencies.ASE3-025: "
        "taskboard mismatch"
    ) in errors


@pytest.mark.parametrize("task_id", ("ASE3-023", "ASE3-022"))
def legacy_hermetic_identity_acceptance_prerequisite_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    prerequisites = config["acceptance_prerequisites"]
    assert isinstance(prerequisites, dict)
    prerequisites[task_id] = []
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.task_dependencies.ASE3-025: "
        "taskboard mismatch"
    ) in errors


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        (
            "accepted_tree_inventory",
            "roadmap_fixed_edge_or_importer_count_allowed",
            True,
        ),
        ("accepted_tree_inventory", "analyzer_implementation_sha256_required", False),
        ("", "ambient_effect_registry_allowed", True),
        ("", "neutral_import_time_io_allowed", 0),
        (
            "",
            "capsule_security_critical_paths",
            ["ipfs_accelerate_py/agent_supervisor/entrypoints/local_profile.py"],
        ),
        (
            "daemon_runner_authorization_baseline",
            "ambient_only_route_fixture_allowed",
            True,
        ),
        (
            "daemon_runner_authorization_baseline",
            "accepted_public_artifact_mode",
            "0644",
        ),
        (
            "daemon_runner_authorization_baseline",
            "test_deselection_allowed",
            True,
        ),
        ("protected_route_invariants", "capacity_projection_dispatch_authorized", True),
        ("scheduler_authorization_baseline", "group_or_other_writable_allowed", True),
        ("", "downstream_requires_accepted_ase3_029", False),
        ("", "task_contract_sha256", "sha256:" + "0" * 64),
    ),
)
def test_contract_layering_scheduler_policy_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    policy = config["neutral_contract_layering"]
    assert isinstance(policy, dict)
    target = policy
    if section:
        nested = policy[section]
        assert isinstance(nested, dict)
        target = nested
    target[field] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(
        "program_scheduler_projection.neutral_contract_layering: exact "
        "content-bound lower-effect contract required" in error
        or "program_scheduler_projection.neutral_contract_layering.contract_sha256"
        in error
        or "program_scheduler_projection.neutral_contract_layering."
        "task_contract_sha256: taskboard mismatch" in error
        for error in errors
    ), errors


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error_fragment"),
    (
        (
            "protected_runtime_activation",
            "authorization_may_claim_activation_effect",
            True,
            "program_scheduler_projection.protected_runtime_activation",
        ),
        (
            "protected_runtime_activation",
            "post_activation_observation_required_for_completion",
            False,
            "program_scheduler_projection.protected_runtime_activation",
        ),
        (
            "refill_policy",
            "saga_cursor_states",
            ["EVALUATING", "APPENDED", "DISPATCHED"],
            "program_scheduler_projection.refill_policy.saga_cursor_states",
        ),
        (
            "refill_policy",
            "monitor_phase_deadlines_required",
            False,
            "program_scheduler_projection.refill_policy.monitor_phase_deadlines_required",
        ),
        (
            "refill_policy",
            "saga_terminal_states_are_alternatives",
            False,
            (
                "program_scheduler_projection.refill_policy."
                "saga_terminal_states_are_alternatives"
            ),
        ),
        (
            "monitor_policy",
            "durable_guardian",
            "ClientOwnedMonitorGuardian",
            "program_scheduler_projection.monitor_policy.durable_guardian",
        ),
        (
            "monitor_policy",
            "running_join_fields",
            ["lifecycle_process_birth", "monitor_process_birth"],
            "program_scheduler_projection.monitor_policy.running_join_fields",
        ),
        (
            "monitor_policy",
            "unknown_outcome_effect_replay_authorized",
            True,
            (
                "program_scheduler_projection.monitor_policy."
                "unknown_outcome_effect_replay_authorized"
            ),
        ),
        (
            "monitor_policy",
            "post_recovery_continuous_health_seconds",
            899,
            (
                "program_scheduler_projection.monitor_policy."
                "post_recovery_continuous_health_seconds"
            ),
        ),
        (
            "",
            "objective_refill_enabled",
            True,
            "program_scheduler_projection.objective_refill_enabled",
        ),
        (
            "monitor_policy",
            "enabled",
            True,
            "program_scheduler_projection.monitor_policy.enabled",
        ),
    ),
)
def test_monitor_strategy_scheduler_contract_mutation_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    error_fragment: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    target: dict[str, object] = config
    if section:
        nested = config[section]
        assert isinstance(nested, dict)
        target = nested
    target[field] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(error_fragment in error for error in errors), errors


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("authorization_may_claim_activation_effect", False),
        ("post_activation_observation_required_for_completion", True),
        ("one_generation_cas_lease_required", True),
        ("operator_review_required", True),
        ("strict_validator_and_manifest_binding_required", True),
    ),
)
@pytest.mark.parametrize("replacement_kind", ("integer", "string", "null"))
def test_protected_activation_boolean_json_type_aliases_fail_closed(
    tmp_path: Path,
    field: str,
    expected: bool,
    replacement_kind: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    activation = config["protected_runtime_activation"]
    assert isinstance(activation, dict)
    assert type(activation[field]) is bool
    assert activation[field] is expected
    replacement: object
    if replacement_kind == "integer":
        replacement = int(expected)
    elif replacement_kind == "string":
        replacement = str(expected).lower()
    else:
        assert replacement_kind == "null"
        replacement = None
    assert type(replacement) is not bool
    activation[field] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        f"program_scheduler_projection.protected_runtime_activation.{field}: "
        f"expected exact JSON boolean {str(expected).lower()}"
    ) in errors
    assert any(
        "protected_runtime_activation.contract_sha256: exact parsed gate"
        in error
        for error in errors
    )


@pytest.mark.parametrize("mutation", ("missing_key", "extra_key"))
def test_protected_activation_exact_key_population_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    activation = config["protected_runtime_activation"]
    assert isinstance(activation, dict)
    if mutation == "missing_key":
        activation.pop("operator_review_required")
    else:
        assert mutation == "extra_key"
        activation["unsealed_extra"] = False
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.protected_runtime_activation.keys: exact "
        "population required"
    ) in errors
    assert any(
        "protected_runtime_activation.contract_sha256: exact parsed gate"
        in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("goal_id", "needle", "replacement"),
    (
        (
            "ASE3-G050",
            "durable EVALUATING to APPEND_RESERVED",
            "process-local EVALUATING to APPEND_RESERVED",
        ),
        (
            "ASE3-G055",
            "immutable nonforking history and monotonic cursor vectors",
            "mutable history and optional cursor vectors",
        ),
        (
            "ASE3-G060",
            "ReviewedHostNamespaceReconciler",
            "ClientOwnedMonitorGuardian",
        ),
        (
            "ASE3-G080",
            "after the final recovery",
            "before the final recovery",
        ),
    ),
)
def test_monitor_strategy_objective_contract_mutation_fails_closed(
    tmp_path: Path,
    goal_id: str,
    needle: str,
    replacement: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    objectives_path = tmp_path / convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH
    text = objectives_path.read_text(encoding="utf-8")
    start = text.index(f"## {goal_id} ")
    end = text.find("\n## ASE3-G", start + 1)
    if end < 0:
        end = len(text)
    block = text[start:end]
    assert needle in block
    objectives_path.write_text(
        text[:start] + block.replace(needle, replacement, 1) + text[end:],
        encoding="utf-8",
    )
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        f"program_scheduler_projection.objectives.{goal_id}.contract_sha256: "
        "exact monitor-strategy goal contract required"
    ) in errors


def test_contract_layering_objective_contract_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    objectives_path = tmp_path / convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH
    text = objectives_path.read_text(encoding="utf-8")
    needle = (
        "concrete CAS/profile/plan-store effects live only in the three declared "
        "lower control services"
    )
    assert text.count(needle) == 1
    objectives_path.write_text(
        text.replace(
            needle,
            "entrypoints retain every concrete effect implementation",
            1,
        ),
        encoding="utf-8",
    )
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.objectives.ASE3-G020.contract_sha256: "
        "exact contract-layering goal contract required"
    ) in errors


@pytest.mark.parametrize("task_id", ("ASE3-023", "ASE3-022"))
def test_native_duckdb_acceptance_prerequisite_tampering_fails_closed(
    tmp_path: Path,
    task_id: str,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    prerequisites = config["acceptance_prerequisites"]
    assert isinstance(prerequisites, dict)
    prerequisites[task_id] = ["ASE3-030", "ASE3-031"]
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.acceptance_prerequisites: exact "
        "ASE3-030/031/032 fail-closed acceptance join required"
    ) in errors


@pytest.mark.parametrize("replacement", (30.0, True))
def test_program_projection_task_count_requires_exact_integer(
    tmp_path: Path,
    replacement: object,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    initial = config["initial_projection"]
    assert isinstance(initial, dict)
    initial["task_count"] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.initial_projection.task_count: "
        "expected exact integer 30"
    ) in errors


def test_ase3_033_protected_transition_roadmap_contract_is_exact_and_dormant() -> None:
    config = _load(CONFIG_PATH)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )
    policy = config["protected_transition_construction"]
    assert isinstance(policy, dict)

    assert type(config["initial_projection"]["task_count"]) is int
    assert config["initial_projection"]["task_count"] == 30
    assert config["task_dependencies"]["ASE3-033"] == ["ASE3-000"]
    assert policy["ordinary_dependencies"] == ["ASE3-000"]
    assert policy["pre_q_integration_freeze_prerequisites"] == list(
        convergence_module._TRANSITION_CONSTRUCTION_PRE_Q_REVIEWS
    )
    assert policy["prerequisite_acceptance_status_required"] is False
    assert policy["required_before_phases"] == list(
        convergence_module._TRANSITION_CONSTRUCTION_REQUIRED_PHASES
    )
    assert policy["product_outputs"] == list(
        convergence_module._TRANSITION_CONSTRUCTION_OUTPUTS
    )
    assert policy["public_apis"] == list(
        convergence_module._TRANSITION_CONSTRUCTION_PUBLIC_APIS
    )
    assert policy["required_transition_tests"] == list(
        convergence_module._TRANSITION_CONSTRUCTION_REQUIRED_TESTS
    )
    assert convergence_module._mapping_contract_sha256(policy) == (
        convergence_module._TRANSITION_CONSTRUCTION_POLICY_SHA256
    )
    plan_text = (
        REPO_ROOT / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    ).read_text(encoding="utf-8")
    assert convergence_module._normalized_markdown_section_contract_sha256(
        plan_text,
        section_heading=(
            convergence_module._TRANSITION_CONSTRUCTION_PLAN_SECTION_HEADING
        ),
        containing_heading=(
            convergence_module._TRANSITION_CONSTRUCTION_PLAN_CONTAINING_HEADING
        ),
        end_heading=(
            convergence_module._TRANSITION_CONSTRUCTION_PLAN_SECTION_END_HEADING
        ),
    ) == convergence_module._TRANSITION_CONSTRUCTION_PLAN_SECTION_CONTRACT_SHA256

    assert policy["q_inventory_contract"] == {
        "path": convergence_module._TRANSITION_Q_INVENTORY_RELATIVE_PATH,
        "allowed_authority": [
            "pre_existing_lifecycle_root_identity_did",
            "stable_policy",
        ],
        "forbidden": [
            "future_pin_sentinel",
            "reviewer_identity",
            "profile",
            "profile_content",
            "lifecycle_anchor",
            "signature",
            "time_observation",
            "test_observation",
            "capsule_output",
            "generation",
        ],
    }
    generation = policy["product_generation_contract"]
    assert isinstance(generation, dict)
    assert generation["roles"] == ["source", "replay", "integrated"]
    assert generation["source_and_replay_are_q_ancestors"] is False
    assert generation["integrated_is_q_ancestor"] is True
    assert generation["generation_order"] == [
        "ASE3-019",
        "ASE3-030",
        "ASE3-031",
        "ASE3-032",
        "ASE3-023",
        "ASE3-027",
    ]
    assert generation["ordered_commit_counts"] == {
        "ASE3-019": 2,
        "ASE3-030": 2,
        "ASE3-031": 1,
        "ASE3-032": 1,
        "ASE3-023": 3,
        "ASE3-027": 2,
    }
    assert generation["required_generation_fields"] == [
        "task_id",
        "ordered_commit_count",
        "source",
        "replay",
        "integrated",
        "final_commits",
    ]
    assert generation["required_role_fields"] == [
        "base_commit",
        "ordered_commits",
        "final_commit",
    ]
    assert generation["required_commit_fields"] == [
        "ordinal",
        "commit",
        "parent",
        "tree",
        "changed_paths",
        "files",
        "canonical_patch_base64",
        "canonical_patch_sha256",
    ]
    assert generation["required_file_fields"] == [
        "path",
        "mode",
        "raw",
        "blob",
    ]
    assert generation["each_ordered_commit_has_exactly_one_parent"] is True
    assert generation["exact_parent_chain_required"] is True
    assert generation["first_commit_parent_equals_role_base"] is True
    assert generation["later_commit_parent_equals_previous_commit"] is True
    assert generation["final_commit_equals_last_ordered_commit"] is True
    assert generation["final_commit_map_contract"] == {
        "exact_task_keys": [
            "ASE3-019",
            "ASE3-030",
            "ASE3-031",
            "ASE3-032",
            "ASE3-023",
            "ASE3-027",
        ],
        "exact_role_keys": ["source", "replay", "integrated"],
        "values_equal_declared_role_final_commits": True,
        "declared_role_final_commits_equal_last_ordered_commits": True,
    }
    assert generation["reconstruct_every_ordered_commit_patch_stream"] is True
    assert generation["patch_reconstruction_operands"] == ["parent", "commit"]
    assert generation["canonical_diff_argv"] == [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--binary",
        "--full-index",
    ]
    assert generation["canonical_patch_encoding"] == "standard-base64"
    assert generation["patch_stream_comparison_key"] == ["task_id", "ordinal"]
    assert generation[
        "source_replay_integrated_patch_streams_byte_identical"
    ] is True
    assert generation["preserve_exact_reviewed_file_modes"] is True
    assert generation["reviewed_mode_examples"] == {
        "ipfs_accelerate_py/llm_router.py": "100755"
    }
    assert generation[
        "cross_task_or_cross_commit_evidence_substitution_allowed"
    ] is False
    portable = generation["portable_verification_contract"]
    assert portable == {
        "fresh_clone_plus_prune_must_verify": True,
        "ambient_local_refs_or_worktree_only_objects_allowed": False,
        "keep_alive_refs_required": False,
        "bundle_artifact_required": False,
        "durable_object_carrier": {
            "kind": "sealed_product_generation_record",
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "prompt-v3-product-generation@1"
            ),
        },
        "source_and_replay_verify_from_sealed_record_only": True,
        "source_and_replay_git_object_reachability_required": False,
        "integrated_commits_must_be_reachable_q_ancestors": True,
        "ase3_033_tooling_must_be_present_in_q_parent_tree": True,
        "sealed_record_fields_sufficient_for_source_replay": [
            "ordinal",
            "commit",
            "parent",
            "tree",
            "changed_paths",
            "files",
            "canonical_patch_base64",
            "canonical_patch_sha256",
        ],
        "git_diff_reconstruction": {
            "required_when_parent_and_commit_objects_present": True,
            "portable_fallback": (
                "sealed_canonical_patch_base64_and_file_raw_blob_mode"
            ),
            "ambient_ref_carrier_forbidden": True,
        },
    }
    authority = policy["phase_authority_contract"]
    assert isinstance(authority, dict)
    assert authority["a019_binds_actual_provider_authorization_v2_digest"] is True
    assert authority["p031_consumer"] == "A031-acceptance-preload-only"
    assert authority["p031_runtime_reuse_allowed"] is False
    assert authority["p031_attempt_ledger"] == {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "prompt-v3-p031-attempt-ledger@1"
        ),
        "maximum_attempts": 3,
        "append_only": True,
        "signed_failure_evidence_required": True,
        "fresh_nonce_and_expiry_per_attempt": True,
        "effect_started_authorization_terminal": True,
        "failed_attempt_overwrite_or_erasure_allowed": False,
        "task_status_before_successful_A031": "todo",
    }
    assert authority["reload_v2_prior_authorization_nullable"] is True
    assert authority["reload_v2_native_authorization_bindings"] == [
        "target_generation",
        "accepted_A031_pin",
        "accepted_A032_gate",
        "accepted_A023_gate",
    ]
    assert authority["reload_same_generation_recovery"] == {
        "bounded": True,
        "requires_dead_and_effectless_observation": True,
        "adopts_consumed_generation_authorization": True,
        "provider_replay_allowed": False,
    }
    signing = policy["signing_adapter_contract"]
    assert signing == {
        "api": "sign_prompt_v3_operator_artifact",
        "receipt_payload_encoding": "strict-ascii-only",
        "canonical_bytes_compatible_with": (
            "entrypoints.local_profile.sign_profile_binding"
        ),
        "receipt_mapping_round_trips_to_identical_sign_profile_binding_canonical_bytes": True,
        "sign_profile_binding_signature_encoding": "standard-base64",
        "signature_transcode": [
            "strict-standard-base64-decode",
            "require-ed25519-64-byte-signature",
            "base64url-without-padding-encode",
        ],
        "public_key_verification_required": True,
        "expected_repository_bindings": [
            "repository_id",
            "repository_head",
            "repository_tree",
        ],
        "expected_authority_bindings": [
            "profile_id",
            "profile_generation",
            "authority_id",
            "artifact_sha256",
        ],
        "post_sign_active_profile_and_generation_reload_required": True,
        "post_sign_reload_must_equal_pre_sign_profile_and_generation": True,
        "rotation_or_revocation_race_denied": True,
        "ambient_signing_key_environment_variable": (
            "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY"
        ),
        "ambient_signing_key_environment_allowed": False,
        "root_pin_builder": {
            "api": "build_prompt_v3_root_pin",
            "public_artifact_only": True,
            "exact_inputs": [
                "pre_existing_lifecycle_root_identity_did",
                "stable_policy",
                "repository_id",
                "repository_head",
                "repository_tree",
            ],
            "exact_output_fields": [
                "schema",
                "root_identity_did",
                "stable_policy_sha256",
                "repository_id",
                "repository_head",
                "repository_tree",
                "canonical_sha256",
            ],
            "private_key_profile_content_or_future_generation_allowed": False,
            "canonical_ascii_bytes_required": True,
            "expected_repository_and_root_authority_verification_required": True,
            "safe_dirfd_projection_required": True,
            "public_artifact_mode": "0400",
        },
    }
    safety = policy["builder_safety"]
    assert isinstance(safety, dict)
    assert safety["alternate_index_required"] is True
    assert safety["working_index_or_auto_commit_paths_allowed"] is False
    assert safety["single_parent_required"] is True
    assert safety["publication"] == {
        "target_ref_resolved_once": True,
        "canonical_lease_held_across_validation_and_publication": True,
        "worktree_holder_discovery": (
            "sanitized-git-worktree-list-porcelain-z"
        ),
        "unheld_target_ref": {
            "method": "git-update-ref-cas",
            "expected_old_oid_required": True,
            "rescue_ref_created_by_cas_before_publish": True,
        },
        "checked_out_exact_target_ref": {
            "method": "sanitized-git-merge-ff-only",
            "exact_argv": [
                "git",
                "-c",
                "core.hooksPath=<owned-empty-directory>",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "tag.gpgSign=false",
                "merge",
                "--ff-only",
                "--no-edit",
                "<candidate-commit>",
            ],
            "same_canonical_lease_required": True,
            "symbolic_head_must_equal_target_ref": True,
            "hooks_allowed": False,
            "commit_or_tag_signing_allowed": False,
            "candidate_must_fast_forward_expected_old_oid": True,
            "required_pre_and_post_state": [
                "target_ref",
                "symbolic_HEAD",
                "ref_oid",
                "HEAD_oid",
                "tree_oid",
                "index_tree_oid",
                "worktree_status_porcelain_v2_z",
            ],
            "pre_state_must_be_exact_and_clean": True,
            "post_ref_head_tree_index_must_equal_candidate": True,
            "post_worktree_must_be_clean": True,
        },
        "detached_only_rejection_of_real_checked_out_target_allowed": False,
        "failure_recovery": {
            "rescue_ref_required": True,
            "target_ref_restore_by_checked_cas": True,
            "checked_out_target_restore_from_verified_rescue_under_same_lease": True,
            "exact_pre_state_reverification_required": True,
            "rescue_ref_retained_on_unrecovered_failure": True,
        },
    }
    assert safety["gitignore_correction"] == {
        "path": ".gitignore",
        "replace_exact_line": "core",
        "with_exact_line": "/core",
        "preserve_root_core_dump_ignore": True,
        "nested_agent_supervisor_core_must_not_be_ignored": True,
        "source_mode": "100755",
        "target_mode": "100644",
    }
    assert safety["required_suite_commands"] == [
        (
            "python ipfs_accelerate_py/agent_supervisor/validation/"
            "prompt_v3_convergence.py --check-all"
        ),
        *(tasks[task_id]["validation"] for task_id in (
            "ASE3-019",
            "ASE3-030",
            "ASE3-031",
            "ASE3-032",
            "ASE3-023",
            "ASE3-027",
            "ASE3-033",
        )),
    ]
    filesystem = safety["filesystem"]
    assert filesystem == {
        "dirfd_relative_io_required": True,
        "o_nofollow_required": True,
        "single_link_required": True,
        "owner_required": True,
        "group_or_other_writable_allowed": False,
        "public_artifact_mode": "0400",
        "private_key_mode": "0600",
        "builder_created_q_and_protected_artifact_mode": "100644",
        "pre_existing_or_reviewed_tracked_modes_preserved": True,
        "runtime_chmod_removes_group_other_write": True,
        "directory_group_or_other_writable_allowed": False,
        "fsync_and_atomic_rename_required": True,
    }
    activation = policy["activation_contract"]
    assert activation == {
        "post_l_birth_required": True,
        "configured_target_ref_resolved_once": True,
        "rev_parse_head_argument_count": 1,
        "activation_before_post_l_birth_allowed": False,
    }
    launch_authority = policy["runtime_launch_authority_contract"]
    assert launch_authority == {
        "loader_api": "load_verified_prompt_v3_runtime_launch_authority",
        "full_committed_chain_required": [
            "Q",
            "R",
            "P019",
            "A019",
            "A030",
            "P031",
            "A031",
            "A032",
            "A023/027",
            "L",
        ],
        "required_acceptance_joins": ["A031", "A032", "ASE3-023"],
        "returns_strict_dto_only_after_full_verification": True,
        "outer_consumer": "ASE3-023-scheduler-plan-revision-store",
        "sealed_child_binds": [
            "dto_sha256",
            "chain_head",
            "chain_tree",
            "A031_pin_id",
            "A032_receipt_id",
            "A023_receipt_id",
            "L_authorization_id",
            "target_generation",
        ],
        "sealed_child_may_import_convergence_validator": False,
        "a031_acceptance_only_operations": [
            "consume_prompt_v3_a031_authorization",
            "append_prompt_v3_a031_failure_attempt",
            "reauthorize_prompt_v3_p031_attempt",
        ],
        "runtime_l_consumption_owner": "ASE3-023-PlanRevisionStore",
    }
    assert policy["q_status_transition"] == {
        "task_id": "ASE3-033",
        "from": "todo",
        "to": "completed",
        "other_status_changes_allowed": False,
        "changed_paths": list(convergence_module._TRANSITION_Q_CHANGED_PATHS),
    }
    sequence = config["protected_native_duckdb_acceptance_sequence"]
    assert sequence["phases"][0] == {
        "phase": "Q",
        "parent_phase": None,
        "task_ids": ["ASE3-033"],
        "changed_paths": list(convergence_module._TRANSITION_Q_CHANGED_PATHS),
    }

    task = tasks["ASE3-033"]
    expected = convergence_module._PROGRAM_EXPANSION_TASKS["ASE3-033"]
    assert task["status"] == "todo"
    assert convergence_module._taskboard_csv(task, "depends on") == ("ASE3-000",)
    assert convergence_module._task_contract_sha256(task) == expected[
        "contract_sha256"
    ]
    assert convergence_module._canonical_task_cid_from_metadata(task) == expected[
        "canonical_task_cid"
    ]
    assert config["objective_refill_enabled"] is False
    assert config["codebase_refill_enabled"] is False
    assert config["monitor_policy"]["enabled"] is False
    # Tooling may exist in Q's parent while ASE3-033 remains todo; only the Q
    # inventory stays reserved until the Q status transition.
    for relative_path in convergence_module._TRANSITION_CONSTRUCTION_RESERVED_PATHS:
        assert not (REPO_ROOT / relative_path).exists()
    for relative_path in convergence_module._TRANSITION_CONSTRUCTION_OUTPUTS[:5]:
        assert (REPO_ROOT / relative_path).is_file()


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("q_inventory_contract", "allowed_authority", ["reviewer_identity"]),
        ("product_generation_contract", "source_and_replay_are_q_ancestors", True),
        ("product_generation_contract", "integrated_is_q_ancestor", False),
        (
            "product_generation_contract",
            "source_replay_integrated_patch_streams_byte_identical",
            False,
        ),
        ("product_generation_contract", "preserve_exact_reviewed_file_modes", False),
        (
            "product_generation_contract",
            "portable_verification_contract",
            {
                "fresh_clone_plus_prune_must_verify": False,
                "ambient_local_refs_or_worktree_only_objects_allowed": True,
                "keep_alive_refs_required": True,
                "bundle_artifact_required": True,
                "durable_object_carrier": {
                    "kind": "ambient-local-refs",
                    "schema": "forbidden",
                },
                "source_and_replay_verify_from_sealed_record_only": False,
                "source_and_replay_git_object_reachability_required": True,
                "integrated_commits_must_be_reachable_q_ancestors": False,
                "ase3_033_tooling_must_be_present_in_q_parent_tree": False,
                "sealed_record_fields_sufficient_for_source_replay": [],
                "git_diff_reconstruction": {
                    "required_when_parent_and_commit_objects_present": False,
                    "portable_fallback": "ambient-local-refs",
                    "ambient_ref_carrier_forbidden": False,
                },
            },
        ),
        ("phase_authority_contract", "p031_runtime_reuse_allowed", True),
        (
            "signing_adapter_contract",
            "post_sign_active_profile_and_generation_reload_required",
            False,
        ),
        (
            "signing_adapter_contract",
            "ambient_signing_key_environment_allowed",
            True,
        ),
        ("builder_safety", "publication", {"method": "detached-only"}),
        ("activation_contract", "activation_before_post_l_birth_allowed", True),
    ),
)
def test_ase3_033_transition_policy_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    policy = config["protected_transition_construction"]
    assert isinstance(policy, dict)
    nested = policy[section]
    assert isinstance(nested, dict)
    nested[field] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(
        "protected_transition_construction.contract_sha256: exact protected "
        "transition construction policy required" in error
        for error in errors
    ), errors


@pytest.mark.parametrize(
    ("task_id", "expected_count"),
    (
        ("ASE3-019", 2),
        ("ASE3-030", 2),
        ("ASE3-031", 1),
        ("ASE3-032", 1),
        ("ASE3-023", 3),
        ("ASE3-027", 2),
    ),
)
def test_ase3_033_every_generation_ordered_commit_count_is_sealed(
    tmp_path: Path,
    task_id: str,
    expected_count: int,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config = _load(config_path)
    policy = config["protected_transition_construction"]
    assert isinstance(policy, dict)
    generation = policy["product_generation_contract"]
    assert isinstance(generation, dict)
    counts = generation["ordered_commit_counts"]
    assert isinstance(counts, dict)
    assert counts[task_id] == expected_count
    counts[task_id] = expected_count + 1
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(
        "protected_transition_construction.contract_sha256: exact protected "
        "transition construction policy required" in error
        for error in errors
    ), errors


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        (
            "Source and clean replay\ncommits remain nonancestors of Q",
            "Source and clean replay\ncommits are ancestors of Q",
        ),
        (
            "requires them to equal the verified\npre-sign profile and generation, "
            "and denies any rotation or revocation race.",
            "permits them to differ from the verified\npre-sign profile and generation, "
            "and allows any rotation or revocation race.",
        ),
        (
            "If the exact target ref is checked out,\nthe builder must not reject "
            "it as detached-only",
            "If the exact target ref is checked out,\nthe builder must reject it "
            "as detached-only",
        ),
        (
            "Portable verification must survive a normal fresh clone plus pruning without\n"
            "ambient local refs, worktree-only objects, keep-alive ref carriers, or bundle\n"
            "artifacts.",
            "Portable verification may rely on ambient local refs, worktree-only objects, "
            "keep-alive ref carriers, or bundle artifacts after clone and pruning.",
        ),
    ),
)
def test_ase3_033_plan_section_hash_rejects_contradictory_semantics(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    plan_text = plan_path.read_text(encoding="utf-8")
    assert plan_text.count(original) == 1
    plan_path.write_text(
        plan_text.replace(original, replacement, 1),
        encoding="utf-8",
    )
    plan_path.chmod(0o600)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.plan.ASE3-033.contract_sha256: exact "
        "normalized protected transition construction section required"
    ) in errors


def test_hermetic_identity_protected_acceptance_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    acceptance = config["protected_identity_acceptance"]
    assert isinstance(acceptance, dict)
    acceptance["receipt_schema"] = "forged@1"
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.protected_identity_acceptance: exact "
        "reserved ASE3-030 receipt contract required"
    ) in errors


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        (
            "protected_native_dependency_launch_authorization",
            "authorization_schema",
            "forged@1",
        ),
        (
            "protected_native_dependency_launch_authorization",
            "inspection_evidence_is_authority",
            0,
        ),
        (
            "protected_native_dependency_launch_authorization",
            "authorization_may_claim_launch_effect",
            True,
        ),
        (
            "protected_native_dependency_launch_authorization",
            "required_before_runtime_effects",
            [],
        ),
        (
            "protected_native_dependency_launch_authorization",
            "sequence_phase",
            "P019",
        ),
        (
            "protected_native_dependency_acceptance",
            "artifact_phase",
            "P",
        ),
        (
            "protected_native_dependency_acceptance",
            "strict_validator_and_manifest_binding_required",
            False,
        ),
        (
            "protected_duckdb_connection_policy_acceptance",
            "requires_prior_acceptance_tasks",
            ["ASE3-031"],
        ),
        (
            "protected_duckdb_connection_policy_acceptance",
            "receipt_schema",
            "forged@1",
        ),
    ),
)
def test_native_duckdb_gate_tampering_fails_closed(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    gate = config[section]
    assert isinstance(gate, dict)
    gate[field] = replacement
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(
        f"program_scheduler_projection.{section}: exact typed reserved gate required"
        in error
        or f"program_scheduler_projection.{section}.contract_sha256" in error
        for error in errors
    )


def test_native_duckdb_acceptance_phase_reordering_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    sequence = config["protected_native_duckdb_acceptance_sequence"]
    assert isinstance(sequence, dict)
    phases = sequence["phases"]
    assert isinstance(phases, list)
    phases[6], phases[7] = phases[7], phases[6]
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection."
        "protected_native_duckdb_acceptance_sequence: exact sequential phase "
        "DAG required"
    ) in errors


def test_scheduler_phase_records_seal_each_direct_child_predicate() -> None:
    config = _load(CONFIG_PATH)
    sequence = config["protected_native_duckdb_acceptance_sequence"]
    assert isinstance(sequence, dict)
    phase_records = sequence["phases"]
    assert isinstance(phase_records, list)
    assert [record["phase"] for record in phase_records] == list(
        convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES
    )
    for record in phase_records:
        phase = record["phase"]
        if phase == "Q":
            assert record["parent_phase"] is None
            assert record["task_ids"] == ["ASE3-033"]
            assert record["changed_paths"] == list(
                convergence_module._TRANSITION_Q_CHANGED_PATHS
            )
            continue
        assert record["parent_phase"] == convergence_module.SEQUENTIAL_PHASE_PARENT[
            phase
        ]
        assert record["task_ids"] == list(
            convergence_module.SEQUENTIAL_PHASE_STATUS_TRANSITIONS[phase]
        )
        assert record["changed_paths"] == list(
            convergence_module.SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
        )
    assert sequence["pre_effect_authorization_only_phases"] == ["P031", "L"]
    assert sequence["runtime_effect_receipt_phases"] == ["A031", "A023/027"]
    assert sequence["post_launch_birth_receipt_forbidden_through_phase"] == "L"
    assert sequence["post_launch_birth_receipt_schema"] == (
        convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
    )


@pytest.mark.parametrize(
    ("task_id", "dependency"),
    (
        ("ASE3-031", "ASE3-032"),
        ("ASE3-027", "ASE3-023"),
    ),
)
def test_native_duckdb_same_or_later_phase_dependency_fails_closed(
    tmp_path: Path,
    task_id: str,
    dependency: str,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    dependencies = config["task_dependencies"]
    assert isinstance(dependencies, dict)
    dependencies[task_id] = [dependency]
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection."
        "protected_native_duckdb_acceptance_sequence.phase_dependency_dag: "
        f"{task_id} depends on {dependency} without a strictly earlier committed "
        "acceptance phase"
    ) in errors


def test_reserved_native_duckdb_protected_path_removal_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    protected_paths = config["protected_paths"]
    assert isinstance(protected_paths, list)
    protected_paths.remove(NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH)
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.protected_paths: all reserved native "
        "DuckDB acceptance paths must be unique and protected"
    ) in errors


def test_signed_canary_observation_policy_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)
    config = _load(CONFIG_PATH)
    monitor = config["monitor_policy"]
    assert isinstance(monitor, dict)
    monitor["canary_observation_seconds"] = 30
    _write(config_path, config)
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.monitor_policy.canary_observation_seconds: "
        "expected 900"
    ) in errors


def test_plan_canary_observation_policy_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    assert "`monitor_policy.canary_observation_seconds: 900`" in text
    plan_path.write_text(
        text.replace("`monitor_policy.canary_observation_seconds: 900`", "30 seconds"),
        encoding="utf-8",
    )
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.plan.canary_observation_seconds: exact "
        "signed 900-second policy required"
    ) in errors


@pytest.mark.parametrize(
    "needle",
    (
        "ReviewedHostNamespaceReconciler",
        (
            "EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→RECOMPILED→"
            "DISPATCHED/ADOPTED"
        ),
        "900 uninterrupted healthy seconds after its final injected recovery",
    ),
)
def test_plan_monitor_strategy_contract_tampering_fails_closed(
    tmp_path: Path,
    needle: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    assert needle in text
    plan_path.write_text(
        text.replace(needle, "forged-monitor-strategy-contract"),
        encoding="utf-8",
    )
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert any(
        "program_scheduler_projection.plan.monitor_strategy: missing" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("needle", "replacement"),
    (
        (
            "and cannot claim a birth, heartbeat,",
            "and may claim a birth, heartbeat,",
        ),
        (
            "CAS/lease winner activate the exact old+1 generation.",
            (
                "CAS/lease winner may activate any generation, including exact "
                "old+1 generation."
            ),
        ),
    ),
)
def test_ase3_026_plan_semantic_contradiction_retaining_tokens_fails_closed(
    tmp_path: Path,
    needle: str,
    replacement: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    assert text.count(needle) == 1
    mutated = text.replace(needle, replacement, 1)
    for retained_token in (
        "authorization_effect_observed: false",
        PROTECTED_RUNTIME_ACTIVATION_AUTHORIZATION_SCHEMA,
        PROTECTED_RUNTIME_POST_ACTIVATION_OBSERVATION_SCHEMA,
        "old+1",
    ):
        assert retained_token in mutated
    plan_path.write_text(mutated, encoding="utf-8")
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.plan.ASE3-026.contract_sha256: exact "
        "normalized protected activation section required"
    ) in errors


@pytest.mark.parametrize(
    ("needle", "field"),
    (
        (
            HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            "hermetic_identity_acceptance_receipt",
        ),
        (
            HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_SCHEMA,
            "hermetic_identity_acceptance_schema",
        ),
        (
            "25fedf091dad928dad1f83c9f81a54c2d401eabe",
            "native_dependency_reviewed_commit",
        ),
        (
            NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
            "native_dependency_launch_authorization",
        ),
        (
            NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
            "native_dependency_launch_authorization_schema",
        ),
        (
            NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            "native_dependency_acceptance_receipt",
        ),
        (
            NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
            "native_dependency_acceptance_schema",
        ),
        (
            DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH,
            "duckdb_connection_policy_acceptance_receipt",
        ),
        (
            DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
            "duckdb_connection_policy_acceptance_schema",
        ),
        (
            (
                "Q→R(root pin)→P019(witness+provider auth@2+manifest)→A019→"
                "A030→P031(native auth+manifest)→A031→A032→A023/027→"
                "L(ASE3-022 reload authorization)"
            ),
            "native_duckdb_acceptance_sequence",
        ),
    ),
)
def test_plan_native_duckdb_acceptance_contract_tampering_fails_closed(
    tmp_path: Path,
    needle: str,
    field: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    assert needle in text
    plan_path.write_text(text.replace(needle, "forged", 1), encoding="utf-8")
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        f"program_scheduler_projection.plan.{field}: exact protected join required"
        in errors
    )


def test_native_duckdb_plan_semantic_contradiction_fails_full_section_seal(
    tmp_path: Path,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    needle = (
        "The\nauthorization is pre-launch authority only: it cannot claim that preload,"
    )
    replacement = (
        "The\nauthorization is pre-launch authority and proves that preload occurred;"
    )
    assert text.count(needle) == 1
    plan_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.plan.ASE3-031-032.contract_sha256: exact "
        "normalized native DuckDB gate section required"
    ) in errors


@pytest.mark.parametrize(
    ("needle", "replacement"),
    (
        (
            (
                "a count\ncopied from a rehearsal, moving repair worktree, or older "
                "prospective tree must\nfail validation"
            ),
            "a roadmap-fixed count is authoritative and may bypass inventory",
        ),
        (
            "canonical signed lifecycle-bound authorization@2",
            "unsigned provider fallback authorization@1",
        ),
        (
            "ambient registries, service locators,",
            "ambient registries and service locators are permitted,",
        ),
        ("`dispatch_authorized=False`", "`dispatch_authorized=True`"),
    ),
)
def test_contract_layering_plan_semantic_tampering_fails_full_section_seal(
    tmp_path: Path,
    needle: str,
    replacement: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    start = text.index(convergence_module._CONTRACT_LAYERING_PLAN_SECTION_HEADING)
    end = text.index(convergence_module._CONTRACT_LAYERING_PLAN_SECTION_END_HEADING)
    block = text[start:end]
    assert block.count(needle) == 1
    plan_path.write_text(
        text[:start] + block.replace(needle, replacement, 1) + text[end:],
        encoding="utf-8",
    )
    tasks = convergence_module._parse_taskboard_metadata(
        TASKBOARD_PATH.read_text(encoding="utf-8")
    )

    errors = convergence_module._validate_program_scheduler_projection(
        repo_root=tmp_path,
        tasks=tasks,
    )

    assert (
        "program_scheduler_projection.plan.ASE3-029.contract_sha256: exact "
        "normalized content-bound contract-layering section required"
    ) in errors


@pytest.mark.parametrize(
    ("projection_name", "needle", "replacement"),
    (
        (
            "audit_finding",
            (
                "the final accepted ASE3-023 tree must produce a content-bound exact "
                "AST\n  inventory of every runtime/todo-daemon import of entrypoints; "
                "a roadmap-fixed\n  count from any prospective tree is stale evidence"
            ),
            (
                "the prospective ASE3-019/023 tree has a fixed exact AST inventory "
                "of\n  eleven runtime/todo-daemon imports of entrypoints; that "
                "roadmap-fixed\n  count is authoritative evidence"
            ),
        ),
        (
            "wave_ordering",
            "concrete CAS/profile/plan-store implementations to lower control services",
            "concrete CAS/profile/plan-store implementations back into entrypoints",
        ),
        (
            "verification_gates",
            "no ambient registry exists, and all provider policy remains in `llm_router`;",
            (
                "an ambient registry supplies lower effects, and all provider policy "
                "remains in `llm_router`;"
            ),
        ),
        (
            "wave_ordering",
            "ASE3-028\nremains strictly later and focuses only",
            "ASE3-028\nruns before ASE3-029 and focuses only",
        ),
    ),
)
def test_contract_layering_normative_plan_projection_tampering_fails_full_validation(
    tmp_path: Path,
    projection_name: str,
    needle: str,
    replacement: str,
) -> None:
    for relative in (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
        convergence_module.PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    ):
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    plan_path = tmp_path / convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH
    text = plan_path.read_text(encoding="utf-8")
    assert text.count(needle) == 1
    plan_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        repo_root=tmp_path,
        check_repository=False,
    )

    assert report.valid is False
    assert (
        "program_scheduler_projection.plan.ASE3-029."
        f"{projection_name}.contract_sha256: exact normalized ASE3-029 normative "
        "plan projection required"
    ) in report.errors


def test_canary_task_cannot_shorten_signed_observation_window(
    tmp_path: Path,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = "signed config `monitor_policy.canary_observation_seconds: 900` window"
    assert text.count(needle) == 1
    taskboard.write_text(
        text.replace(needle, "caller-selected 30-second policy window", 1),
        encoding="utf-8",
    )

    report = validate_convergence_artifacts(
        DEFAULT_ARTIFACT_ROOT,
        check_repository=False,
        taskboard_path=taskboard,
    )

    assert (
        "program_plan_expansion.ASE3-013.contract_sha256: exact amended "
        "metadata/prose required"
    ) in report.errors


def test_check_all_cli_emits_the_sealed_preflight_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.validation.prompt_v3_convergence",
            "--check-all",
            "--repo-root",
            str(REPO_ROOT),
            "--artifacts-root",
            str(DEFAULT_ARTIFACT_ROOT),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_check_all_direct_file_entrypoint_matches_scheduler_execution() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--check-all",
            "--repo-root",
            str(REPO_ROOT),
            "--artifacts-root",
            str(DEFAULT_ARTIFACT_ROOT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert payload["valid"] is True
    assert payload["errors"] == []


def _json_clone(value: object) -> object:
    return json.loads(json.dumps(value))


def _sequential_parent_fixture(phase: str) -> dict[str, object]:
    return {
        "head": hashlib.sha1((phase + "-head").encode()).hexdigest(),
        "tree": hashlib.sha1((phase + "-tree").encode()).hexdigest(),
        "branch": "agent/prompt-self-improvement-v3",
        "phase": phase,
        "manifest_schema": (
            convergence_module.CONVERGENCE_MANIFEST_SCHEMA
            if phase in {"Q", "R", "P019"}
            else ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
        ),
        "prior_artifacts": {
            path: "sha256:" + hashlib.sha256(path.encode()).hexdigest()
            for path in convergence_module._sequential_artifacts_after(phase)
        },
        "future_artifact_paths_absent": list(
            convergence_module._sequential_future_artifacts_after(phase)
        ),
        "task_statuses": convergence_module._sequential_task_statuses_after(
            phase
        ),
        "reload_gate_status": "blocked",
    }


def _native_task_fixture(task_id: str, *, authorization_only: bool) -> dict[str, object]:
    contract = convergence_module._SEQUENTIAL_TASK_CONTRACTS[task_id]
    return {
        "task_id": task_id,
        "canonical_task_cid": contract["canonical_task_cid"],
        "todo_contract_sha256": contract["todo_contract_sha256"],
        "completed_contract_sha256": contract["completed_contract_sha256"],
        "status_before": "todo",
        "status_after": "todo" if authorization_only else "completed",
    }


def _native_p031_fixture(
    *,
    signing_key: Ed25519PrivateKey | None = None,
) -> tuple[dict[str, object], bytes, dict[str, object], Ed25519PrivateKey]:
    private_key = signing_key or Ed25519PrivateKey.generate()
    authority = _review_authority(_reviewer_identity(private_key))
    created_at = "2026-08-08T20:00:00Z"
    payload: dict[str, object] = {
        "schema": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "phase": "P031",
        "task": _native_task_fixture("ASE3-031", authorization_only=True),
        "acceptance_parent": _sequential_parent_fixture("A030"),
        "authorization_id": "sha256:" + ("0" * 64),
        "product": _json_clone(convergence_module._ASE3_031_PRODUCT_IDENTITY),
        "native_pin": _json_clone(
            convergence_module._ASE3_031_REVIEWED_DEPENDENCY_PIN
        ),
        "host_abi_trust_boundary": _json_clone(
            convergence_module._ASE3_031_HOST_ABI_TRUST_BOUNDARY
        ),
        "claims": dict(convergence_module._NATIVE_DEPENDENCY_AUTHORIZATION_CLAIMS),
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-031-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._NATIVE_DEPENDENCY_AUTHORIZATION_DENIALS),
    }
    payload["authorization_id"] = (
        convergence_module.native_dependency_launch_authorization_id(payload)
    )
    _sign_operator_receipt(payload, private_key)
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return payload, raw, authority, private_key


def _native_a031_fixture(
    *,
    signing_key: Ed25519PrivateKey | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    bytes,
    dict[str, object],
]:
    p031, p031_raw, authority, private_key = _native_p031_fixture(
        signing_key=signing_key
    )
    created_at = "2026-08-08T20:00:01Z"
    final_values = {
        "ready": True,
        "passed_count": 17,
        "report_sha256": "sha256:" + ("9" * 64),
    }
    payload: dict[str, object] = {
        "schema": NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "phase": "A031",
        "task": _native_task_fixture("ASE3-031", authorization_only=False),
        "acceptance_parent": _sequential_parent_fixture("P031"),
        "launch_authorization": {
            "path": NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH,
            "sha256": "sha256:" + hashlib.sha256(p031_raw).hexdigest(),
            "authorization_id": p031["authorization_id"],
            "phase": "P031",
        },
        "product": _json_clone(convergence_module._ASE3_031_PRODUCT_IDENTITY),
        "native_pin": _json_clone(
            convergence_module._ASE3_031_REVIEWED_DEPENDENCY_PIN
        ),
        "sealed_descriptor": {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "native-dependency-descriptor@1"
            ),
            "descriptor": 3,
            "st_dev": 1,
            "st_ino": 2,
            "st_mode": 0o100500,
            "st_uid": os.geteuid(),
            "st_nlink": 1,
            "size_bytes": convergence_module._ASE3_031_REVIEWED_DEPENDENCY_PIN[
                "size_bytes"
            ],
            "payload_sha256": convergence_module._ASE3_031_REVIEWED_DEPENDENCY_PIN[
                "payload_sha256"
            ],
            "seals": 15,
        },
        "process_terminal": {
            "terminal_sentinel_set_before_native_module_creation": True,
            "native_module_creation_started": True,
            "partial_initialization_retry_denied": True,
            "second_preload_attempt_denied": True,
            "terminal_returncode": 0,
        },
        "preload_evidence": {
            "launch_schema": (
                "ipfs_accelerate_py.agent_supervisor.native-dependency-launch@1"
            ),
            "accepted_authorization_id": p031["authorization_id"],
            "sealed_fd_verified_before_module_creation": True,
            "module_name": "_duckdb",
            "public_alias": "duckdb",
            "distribution_version": "1.5.2",
            "engine_version": "v1.5.2",
            "query_42_result": 42,
            "parent_environment_sanitized_before_exec": True,
            "forbidden_parent_environment_names": [
                "GLIBC_TUNABLES",
                "LD_AUDIT",
                "LD_DEBUG",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "PYTHONHOME",
                "PYTHONPATH",
            ],
            "child_observed_forbidden_environment_names": [],
            "python_side_environment_rejection_triggered": False,
            "runtime_effect_started_at": created_at,
            "runtime_effect_started_after_authorization": True,
        },
        "host_abi_trust_boundary": _json_clone(
            convergence_module._ASE3_031_HOST_ABI_TRUST_BOUNDARY
        ),
        "suite": {
            "command": convergence_module._PROGRAM_EXPANSION_TASKS["ASE3-031"][
                "validation"
            ],
            "exit_code": 0,
            "passed": True,
            "passed_count": final_values["passed_count"],
            "failed_count": 0,
            "validated_head": _sequential_parent_fixture("P031")["head"],
            "validated_tree": _sequential_parent_fixture("P031")["tree"],
            "report_sha256": final_values["report_sha256"],
            "required_test_functions": list(
                convergence_module._ASE3_031_REQUIRED_TEST_FUNCTIONS
            ),
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-031-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._NATIVE_DEPENDENCY_ACCEPTANCE_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, p031, p031_raw, {"authority": authority, "final": final_values}


def _duckdb_a032_fixture(
    *,
    signing_key: Ed25519PrivateKey | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    private_key = signing_key or Ed25519PrivateKey.generate()
    authority = _review_authority(_reviewer_identity(private_key))
    created_at = "2026-08-08T20:00:02Z"
    final_values = {
        "ready": True,
        "passed_count": 23,
        "report_sha256": "sha256:" + ("a" * 64),
    }
    parent = _sequential_parent_fixture("A031")
    payload: dict[str, object] = {
        "schema": DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "phase": "A032",
        "task": _native_task_fixture("ASE3-032", authorization_only=False),
        "acceptance_parent": parent,
        "product": _json_clone(convergence_module._ASE3_032_PRODUCT_IDENTITY),
        "connection_birth_policy": {
            "settings_in_connect_call": _json_clone(
                convergence_module._ASE3_032_CONNECTION_POLICY_SETTINGS
            ),
            "tuning_bounds": _json_clone(
                convergence_module._ASE3_032_CONNECTION_TUNING_BOUNDS
            ),
            "lock_configuration_last": True,
            "returned_connection_exact_bool_tuple": [
                False,
                False,
                False,
                False,
                True,
            ],
            "close_on_verification_failure": True,
            "caller_override_or_coercion_allowed": False,
        },
        "connection_sites": _json_clone(
            convergence_module._ASE3_032_CONNECTION_SITE_COUNTS
        ),
        "external_byte_boundary": dict(
            convergence_module._DUCKDB_EXTERNAL_BYTE_BOUNDARY
        ),
        "catalog_seal": {
            "path_independent_full_persistent_catalog_equality": True,
            "inventories": [
                "databases",
                "schemas",
                "tables",
                "views",
                "sequences",
                "macros_and_functions",
                "types",
                "indexes",
                "constraints",
                "columns",
            ],
            "foreign_catalog_cases_rejected": _json_clone(
                convergence_module._ASE3_032_FOREIGN_CATALOG_CASES
            ),
            "source_bytes_unchanged_on_rejection": True,
            "temporary_files_cleaned_on_rejection": True,
        },
        "legacy_migration": {
            field: True
            for field in convergence_module._DUCKDB_LEGACY_MIGRATION_REQUIRED_FIELDS
        },
        "compaction": {
            "attach_count": 0,
            "source_read_only": True,
            "target_policy_initialized": True,
            "partial_copy_failure_preserves_authoritative_store": True,
            "atomic_replace_failure_preserves_authoritative_store": True,
            "foreign_catalog_rejection_preserves_source_bytes": True,
            "temporary_files_cleaned": True,
        },
        "suite": {
            "command": convergence_module._PROGRAM_EXPANSION_TASKS["ASE3-032"][
                "validation"
            ],
            "exit_code": 0,
            "passed": True,
            "passed_count": final_values["passed_count"],
            "failed_count": 0,
            "validated_head": parent["head"],
            "validated_tree": parent["tree"],
            "report_sha256": final_values["report_sha256"],
            "required_test_functions": list(
                convergence_module._ASE3_032_REQUIRED_TEST_FUNCTIONS
            ),
        },
        "review": {
            **_receipt_review_authority(authority),
            "implementer_identity": "codex:ase3-032-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence_module._DUCKDB_POLICY_ACCEPTANCE_DENIALS),
    }
    _sign_operator_receipt(payload, private_key)
    return payload, final_values, authority


def _git_bound_p031_fixture(
    *,
    parent_head: str,
    parent_tree: str,
) -> tuple[dict[str, object], dict[str, object]]:
    private_key = Ed25519PrivateKey.generate()
    payload, _, authority, _ = _native_p031_fixture(signing_key=private_key)
    parent = payload["acceptance_parent"]
    assert isinstance(parent, dict)
    parent["head"] = parent_head
    parent["tree"] = parent_tree
    payload["authorization_id"] = (
        convergence_module.native_dependency_launch_authorization_id(payload)
    )
    _sign_operator_receipt(payload, private_key)
    return payload, authority


def _git_bound_a032_fixture(
    *,
    parent_head: str,
    parent_tree: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    private_key = Ed25519PrivateKey.generate()
    payload, final_values, authority = _duckdb_a032_fixture(
        signing_key=private_key
    )
    parent = payload["acceptance_parent"]
    suite = payload["suite"]
    assert isinstance(parent, dict)
    assert isinstance(suite, dict)
    parent["head"] = parent_head
    parent["tree"] = parent_tree
    suite["validated_head"] = parent_head
    suite["validated_tree"] = parent_tree
    _sign_operator_receipt(payload, private_key)
    return payload, final_values, authority


def test_p031_a031_and_a032_signed_schemas_are_exact() -> None:
    product_parent_head = convergence_module._ASE3_032_PRODUCT_IDENTITY["commit"]
    product_parent_tree = convergence_module._ASE3_032_PRODUCT_IDENTITY["tree"]
    signing_key = Ed25519PrivateKey.generate()
    a031, p031, _, context = _native_a031_fixture(signing_key=signing_key)
    p031_parent = p031["acceptance_parent"]
    assert isinstance(p031_parent, dict)
    p031_parent["head"] = product_parent_head
    p031_parent["tree"] = product_parent_tree
    p031["authorization_id"] = (
        convergence_module.native_dependency_launch_authorization_id(p031)
    )
    _sign_operator_receipt(p031, signing_key)
    p031_raw = _sequential_json_bytes(p031)
    a031_parent = a031["acceptance_parent"]
    a031_launch = a031["launch_authorization"]
    a031_preload = a031["preload_evidence"]
    a031_suite = a031["suite"]
    assert isinstance(a031_parent, dict)
    assert isinstance(a031_launch, dict)
    assert isinstance(a031_preload, dict)
    assert isinstance(a031_suite, dict)
    a031_parent["head"] = product_parent_head
    a031_parent["tree"] = product_parent_tree
    a031_launch["sha256"] = "sha256:" + hashlib.sha256(p031_raw).hexdigest()
    a031_launch["authorization_id"] = p031["authorization_id"]
    a031_preload["accepted_authorization_id"] = p031["authorization_id"]
    a031_suite["validated_head"] = product_parent_head
    a031_suite["validated_tree"] = product_parent_tree
    _sign_operator_receipt(a031, signing_key)
    assert convergence_module.validate_native_dependency_launch_authorization(
        p031,
        repo_root=REPO_ROOT,
        lifecycle_authority=context["authority"],
    ) == ()
    assert convergence_module.validate_native_dependency_acceptance_receipt(
        a031,
        launch_authorization=p031,
        launch_authorization_raw=p031_raw,
        repo_root=REPO_ROOT,
        lifecycle_authority=context["authority"],
        final_values=context["final"],
    ) == ()

    a032_key = Ed25519PrivateKey.generate()
    a032, final_values, authority = _duckdb_a032_fixture(
        signing_key=a032_key
    )
    a032_parent = a032["acceptance_parent"]
    a032_suite = a032["suite"]
    assert isinstance(a032_parent, dict)
    assert isinstance(a032_suite, dict)
    a032_parent["head"] = product_parent_head
    a032_parent["tree"] = product_parent_tree
    a032_suite["validated_head"] = product_parent_head
    a032_suite["validated_tree"] = product_parent_tree
    _sign_operator_receipt(a032, a032_key)
    assert convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
        a032,
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
        final_values=final_values,
    ) == ()


def test_a031_and_a032_product_identities_reconstruct_exact_git_bytes() -> None:
    acceptance_parent_head = convergence_module._ASE3_032_PRODUCT_IDENTITY["commit"]
    acceptance_parent_tree = convergence_module._ASE3_032_PRODUCT_IDENTITY["tree"]
    for task_id, product, required_tests, required_sites in (
        (
            "ASE3-031",
            convergence_module._ASE3_031_PRODUCT_IDENTITY,
            convergence_module._ASE3_031_REQUIRED_TEST_FUNCTIONS,
            None,
        ),
        (
            "ASE3-032",
            convergence_module._ASE3_032_PRODUCT_IDENTITY,
            convergence_module._ASE3_032_REQUIRED_TEST_FUNCTIONS,
            convergence_module._ASE3_032_CONNECTION_SITE_COUNTS,
        ),
    ):
        assert convergence_module._validate_frozen_product_identity(
            actual=product,
            expected=product,
            prefix=f"product.{task_id}",
            repo_root=REPO_ROOT,
            acceptance_parent_head=acceptance_parent_head,
            acceptance_parent_tree=acceptance_parent_tree,
            require_acceptance_parent=True,
            required_test_functions=required_tests,
            required_connection_sites=required_sites,
        ) == []


def test_a031_and_a032_required_suite_functions_match_audited_git_files() -> None:
    for product, test_path, required_functions in (
        (
            convergence_module._ASE3_031_PRODUCT_IDENTITY,
            "test/api/test_agent_supervisor_native_dependency_pin.py",
            convergence_module._ASE3_031_REQUIRED_TEST_FUNCTIONS,
        ),
        (
            convergence_module._ASE3_032_PRODUCT_IDENTITY,
            "test/api/test_agent_supervisor_duckdb_connection_policy.py",
            convergence_module._ASE3_032_REQUIRED_TEST_FUNCTIONS,
        ),
    ):
        source = subprocess.run(
            ["git", "show", f"{product['commit']}:{test_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        observed = [
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        assert observed == required_functions


def _reviewed_product_descendant_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "reviewed-product-descendant"
    subprocess.run(
        ["git", "clone", "-q", "--shared", str(REPO_ROOT), str(repo)],
        check=True,
    )
    _sequential_git(
        repo,
        "checkout",
        "-q",
        "--detach",
        convergence_module._ASE3_032_PRODUCT_IDENTITY["commit"],
    )
    _sequential_git(repo, "config", "user.name", "Product Shadow Audit")
    _sequential_git(
        repo,
        "config",
        "user.email",
        "product-shadow@example.invalid",
    )
    return repo


@pytest.mark.parametrize("mutation", ("bytes", "mode"))
def test_rehashed_reviewed_product_descendants_cannot_shadow_any_frozen_path(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = _reviewed_product_descendant_repository(tmp_path)
    paths_031 = tuple(
        convergence_module._ASE3_031_PRODUCT_IDENTITY["changed_paths"]
    )
    paths_032 = tuple(
        convergence_module._ASE3_032_PRODUCT_IDENTITY["changed_paths"]
    )
    all_paths = (*paths_031, *paths_032)
    if mutation == "bytes":
        for relative_path in all_paths:
            path = repo / relative_path
            path.write_bytes(path.read_bytes() + b"\n# rehashed descendant shadow\n")
        _sequential_git(repo, "add", "--", *all_paths)
    else:
        for relative_path in all_paths:
            expected_mode = convergence_module._FROZEN_PRODUCT_GIT_IDENTITIES[
                (
                    convergence_module._ASE3_031_PRODUCT_IDENTITY["commit"]
                    if relative_path in paths_031
                    else convergence_module._ASE3_032_PRODUCT_IDENTITY["commit"]
                )
            ][relative_path]["mode"]
            _sequential_git(
                repo,
                "update-index",
                "--chmod=-x" if expected_mode == "100755" else "--chmod=+x",
                "--",
                relative_path,
            )
    _sequential_git(repo, "commit", "-q", "-m", f"shadow reviewed {mutation}")
    descendant_head = _sequential_git(repo, "rev-parse", "HEAD")
    descendant_tree = _sequential_git(repo, "rev-parse", "HEAD^{tree}")

    p031, p031_authority = _git_bound_p031_fixture(
        parent_head=descendant_head,
        parent_tree=descendant_tree,
    )
    p031_errors = (
        convergence_module.validate_native_dependency_launch_authorization(
            p031,
            repo_root=repo,
            lifecycle_authority=p031_authority,
        )
    )
    a032, final_values, a032_authority = _git_bound_a032_fixture(
        parent_head=descendant_head,
        parent_tree=descendant_tree,
    )
    a032_errors = (
        convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
            a032,
            repo_root=repo,
            lifecycle_authority=a032_authority,
            final_values=final_values,
        )
    )

    for relative_path in paths_031:
        assert any(
            f"acceptance_parent_tree.{relative_path}" in error
            for error in p031_errors
        )
        raw_errors = [
            error
            for error in p031_errors
            if f"acceptance_parent_raw_sha256.{relative_path}" in error
        ]
        assert bool(raw_errors) is (mutation == "bytes")
    for relative_path in paths_032:
        assert any(
            f"acceptance_parent_tree.{relative_path}" in error
            for error in a032_errors
        )
        raw_errors = [
            error
            for error in a032_errors
            if f"acceptance_parent_raw_sha256.{relative_path}" in error
        ]
        assert bool(raw_errors) is (mutation == "bytes")
    assert not any(".test_ast:" in error for error in (*p031_errors, *a032_errors))
    assert not any(".site_ast:" in error for error in a032_errors)


def test_rehashed_reviewed_test_body_substitution_fails_ast_equality(
    tmp_path: Path,
) -> None:
    repo = _reviewed_product_descendant_repository(tmp_path)
    path_031 = "test/api/test_agent_supervisor_native_dependency_pin.py"
    path_032 = "test/api/test_agent_supervisor_duckdb_connection_policy.py"
    replacements = {
        path_031: (
            b"assert stat.S_IMODE(source.stat().st_mode) == 0o775",
            b"assert stat.S_IMODE(source.stat().st_mode) == 0o700",
        ),
        path_032: (b"assert len(observed) == 1", b"assert len(observed) == 2"),
    }
    for relative_path, (needle, replacement) in replacements.items():
        path = repo / relative_path
        raw = path.read_bytes()
        assert raw.count(needle) >= 1
        path.write_bytes(raw.replace(needle, replacement, 1))
    _sequential_git(repo, "add", "--", path_031, path_032)
    _sequential_git(repo, "commit", "-q", "-m", "substitute reviewed test bodies")
    head = _sequential_git(repo, "rev-parse", "HEAD")
    tree = _sequential_git(repo, "rev-parse", "HEAD^{tree}")

    p031, p031_authority = _git_bound_p031_fixture(
        parent_head=head,
        parent_tree=tree,
    )
    p031_errors = (
        convergence_module.validate_native_dependency_launch_authorization(
            p031,
            repo_root=repo,
            lifecycle_authority=p031_authority,
        )
    )
    a032, final_values, a032_authority = _git_bound_a032_fixture(
        parent_head=head,
        parent_tree=tree,
    )
    a032_errors = (
        convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
            a032,
            repo_root=repo,
            lifecycle_authority=a032_authority,
            final_values=final_values,
        )
    )
    assert any(".test_ast: exact frozen reviewed AST required" in error for error in p031_errors)
    assert any(".test_ast: exact frozen reviewed AST required" in error for error in a032_errors)
    assert not any(".site_ast:" in error for error in a032_errors)


def test_rehashed_a032_runtime_site_counter_substitution_fails_ast_proof(
    tmp_path: Path,
) -> None:
    repo = _reviewed_product_descendant_repository(tmp_path)
    relative_path = "test/api/test_agent_supervisor_duckdb_connection_policy.py"
    path = repo / relative_path
    raw = path.read_bytes()
    needle = b'"merge_queue.initialize": 1,'
    assert raw.count(needle) == 1
    path.write_bytes(raw.replace(needle, b'"merge_queue.initialize": 2,', 1))
    _sequential_git(repo, "add", "--", relative_path)
    _sequential_git(repo, "commit", "-q", "-m", "substitute reviewed site count")
    head = _sequential_git(repo, "rev-parse", "HEAD")
    tree = _sequential_git(repo, "rev-parse", "HEAD^{tree}")
    payload, final_values, authority = _git_bound_a032_fixture(
        parent_head=head,
        parent_tree=tree,
    )
    errors = convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
        payload,
        repo_root=repo,
        lifecycle_authority=authority,
        final_values=final_values,
    )
    assert any(".test_ast: exact frozen reviewed AST required" in error for error in errors)
    assert any(".site_ast: exact reviewed connection-site Counter required" in error for error in errors)


def test_rehashed_product_receipts_cannot_rebind_parent_head_to_another_tree() -> None:
    head = convergence_module._ASE3_032_PRODUCT_IDENTITY["commit"]
    wrong_tree = "0" * 40
    p031, p031_authority = _git_bound_p031_fixture(
        parent_head=head,
        parent_tree=wrong_tree,
    )
    p031_errors = (
        convergence_module.validate_native_dependency_launch_authorization(
            p031,
            repo_root=REPO_ROOT,
            lifecycle_authority=p031_authority,
        )
    )
    a032, final_values, a032_authority = _git_bound_a032_fixture(
        parent_head=head,
        parent_tree=wrong_tree,
    )
    a032_errors = (
        convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
            a032,
            repo_root=REPO_ROOT,
            lifecycle_authority=a032_authority,
            final_values=final_values,
        )
    )
    assert any("acceptance_parent_tree: head/tree identity mismatch" in error for error in p031_errors)
    assert any("acceptance_parent_tree: head/tree identity mismatch" in error for error in a032_errors)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("effect", "claims.runtime_effect_started: expected False"),
        ("extra", "exact key population required"),
        ("time", "outside witness validity"),
    ),
)
def test_p031_is_signed_prelaunch_authority_and_cannot_claim_effect(
    mutation: str,
    expected: str,
) -> None:
    payload, _, authority, private_key = _native_p031_fixture()
    if mutation == "effect":
        claims = payload["claims"]
        assert isinstance(claims, dict)
        claims["runtime_effect_started"] = True
    elif mutation == "extra":
        payload["extra"] = "self-consistent-extension"
    else:
        payload["created_at"] = "2027-08-08T20:00:00Z"
        review = payload["review"]
        assert isinstance(review, dict)
        review["signed_at"] = payload["created_at"]
    payload["authorization_id"] = (
        convergence_module.native_dependency_launch_authorization_id(payload)
    )
    _sign_operator_receipt(payload, private_key)

    errors = convergence_module.validate_native_dependency_launch_authorization(
        payload,
        lifecycle_authority=authority,
    )

    assert any(expected in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    (
        (
            "parent_environment_sanitized_before_exec",
            False,
            "preload_evidence.parent_environment_sanitized_before_exec: expected True",
        ),
        (
            "child_observed_forbidden_environment_names",
            ["LD_PRELOAD"],
            "preload_evidence.child_observed_forbidden_environment_names: exact population required",
        ),
        (
            "sealed_fd_verified_before_module_creation",
            False,
            "preload_evidence.sealed_fd_verified_before_module_creation: expected True",
        ),
    ),
)
def test_a031_rejects_hostile_environment_and_pre_seal_effect_confusion(
    field: str,
    value: object,
    expected: str,
) -> None:
    a031, p031, p031_raw, context = _native_a031_fixture()
    preload = a031["preload_evidence"]
    assert isinstance(preload, dict)
    preload[field] = value

    errors = convergence_module.validate_native_dependency_acceptance_receipt(
        a031,
        launch_authorization=p031,
        launch_authorization_raw=p031_raw,
        lifecycle_authority=context["authority"],
        final_values=context["final"],
    )

    assert any(expected in error for error in errors)


def test_a031_requires_exact_signed_p031_bytes_and_terminal_single_attempt() -> None:
    a031, p031, p031_raw, context = _native_a031_fixture()
    terminal = a031["process_terminal"]
    assert isinstance(terminal, dict)
    terminal["second_preload_attempt_denied"] = False
    errors = convergence_module.validate_native_dependency_acceptance_receipt(
        a031,
        launch_authorization=p031,
        launch_authorization_raw=p031_raw + b" ",
        lifecycle_authority=context["authority"],
        final_values=context["final"],
    )
    assert any("launch_authorization.sha256: expected" in error for error in errors)
    assert any(
        "process_terminal.second_preload_attempt_denied: expected True" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("connection_sites", "merge_queue.initialize", 0),
        ("catalog_seal", "foreign_catalog_cases_rejected", []),
        (
            "legacy_migration",
            "transactional_two_step_add_default_then_set_not_null",
            False,
        ),
        ("compaction", "attach_count", 1),
    ),
)
def test_a032_rejects_site_catalog_migration_and_compaction_weakening(
    section: str,
    field: str,
    value: object,
) -> None:
    payload, final_values, authority = _duckdb_a032_fixture()
    block = payload[section]
    assert isinstance(block, dict)
    block[field] = value

    errors = convergence_module.validate_duckdb_connection_policy_acceptance_receipt(
        payload,
        lifecycle_authority=authority,
        final_values=final_values,
    )

    assert any(
        f"{section}.{field}: expected" in error
        or f"{section}.{field}: exact population required" in error
        for error in errors
    )


@pytest.mark.parametrize("kind", ("regular", "symlink"))
def test_native_phase_loader_requires_a_regular_no_follow_receipt(
    tmp_path: Path,
    kind: str,
) -> None:
    payload, raw, _, _ = _native_p031_fixture()
    path = tmp_path / NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_FILENAME
    if kind == "regular":
        path.write_bytes(raw)
        path.chmod(0o600)
        snapshot = convergence_module.load_native_dependency_launch_authorization(
            path
        )
        assert snapshot.payload == payload
    else:
        path.symlink_to(tmp_path / "missing-native-authorization.json")
        with pytest.raises((OSError, ValueError)):
            convergence_module.load_native_dependency_launch_authorization(path)


def _sequential_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result.stdout.strip()


def _sequential_write(repo: Path, relative_path: str, raw: bytes) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o644)


def _sequential_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sequential_commit(repo: Path, phase: str, paths: tuple[str, ...]) -> str:
    _sequential_git(repo, "add", "--", *paths)
    _sequential_git(repo, "commit", "-q", "-m", f"protected {phase}")
    return _sequential_git(repo, "rev-parse", "HEAD")


def _phase_parent_payload(
    repo: Path,
    *,
    phase: str,
    head: str,
    prior_artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "phase": phase,
        "head": head,
        "tree": _sequential_git(repo, "rev-parse", f"{head}^{{tree}}"),
        "branch": "protected/sequential-acceptance-fixture",
        "manifest_schema": (
            convergence_module.CONVERGENCE_MANIFEST_SCHEMA
            if phase in {"Q", "R", "P019"}
            else ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
        ),
        "prior_artifacts": dict(prior_artifacts),
    }


def _minimal_sequential_native_receipt(
    *,
    fields: tuple[str, ...],
    schema: str,
    phase: str,
    parent: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {field: {} for field in fields}
    payload.update(
        {
            "schema": schema,
            "created_at": "2026-08-08T20:00:02Z",
            "board_namespace": BOARD_NAMESPACE,
            "phase": phase,
            "acceptance_parent": dict(parent),
        }
    )
    return payload


def _top_level_sequential_phase_repository(
    tmp_path: Path,
    *,
    phase_times: Mapping[str, str] | None = None,
    a031_effect_time: str | None = None,
) -> tuple[Path, dict[str, str], dict[str, object]]:
    """Build real Q→…→L Git/manifest/lifecycle bytes for top-level dispatch."""

    repo = tmp_path / "top-level-sequential-acceptance"
    subprocess.run(
        ["git", "clone", "-q", "--shared", str(REPO_ROOT), str(repo)],
        check=True,
    )
    _sequential_git(repo, "config", "user.name", "Sequential Top-Level Audit")
    _sequential_git(
        repo,
        "config",
        "user.email",
        "sequential-top-level@example.invalid",
    )
    roadmap_paths = (
        PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
        convergence_module.PROMPT_V3_PLAN_RELATIVE_PATH,
        convergence_module.PROMPT_V3_OBJECTIVES_RELATIVE_PATH,
        PROMPT_V3_TASKBOARD_RELATIVE_PATH,
        Path("ipfs_accelerate_py/agent_supervisor/validation/prompt_v3_convergence.py"),
        Path("test/api/test_agent_supervisor_prompt_v3_convergence.py"),
    )
    for relative_path in roadmap_paths:
        source = REPO_ROOT / relative_path
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _sequential_git(repo, "add", "--", *(str(path) for path in roadmap_paths))
    _sequential_git(repo, "commit", "-q", "-m", "integrate ASE3-033 tooling")
    _sequential_git(repo, "commit", "-q", "--allow-empty", "-m", "protected Q")
    heads = {"Q": _sequential_git(repo, "rev-parse", "HEAD")}
    q_tree = _sequential_git(repo, "rev-parse", "HEAD^{tree}")

    root_key = Ed25519PrivateKey.generate()
    active_key = Ed25519PrivateKey.generate()
    pinned_at_ms = 1_786_215_600_000
    root_did = _reviewer_identity(root_key)
    root_pin = _root_pin_payload(
        root_identity_did=root_did,
        base_head=heads["Q"],
        base_tree=q_tree,
        pinned_at_ms=pinned_at_ms,
    )
    root_pin_path = (
        repo / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    _write(root_pin_path, root_pin)
    heads["R"] = _sequential_commit(
        repo,
        "R",
        convergence_module.Q_TO_R_CHANGED_PATHS,
    )
    r_tree = _sequential_git(repo, "rev-parse", "HEAD^{tree}")

    witness, reviewer_final_values = _lifecycle_witness_payload(
        root_key=root_key,
        active_key=active_key,
        base_head=heads["R"],
        base_tree=r_tree,
        observed_at_ms=pinned_at_ms + 1_000,
    )
    witness_path = (
        repo / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    )
    _write(witness_path, witness)
    authorization = _fallback_authorization_v2_payload(
        active_key=active_key,
        witness=witness,
        witness_sha256=(
            "sha256:" + hashlib.sha256(witness_path.read_bytes()).hexdigest()
        ),
        root_pin=root_pin,
        root_pin_sha256=(
            "sha256:" + hashlib.sha256(root_pin_path.read_bytes()).hexdigest()
        ),
        source_head=heads["R"],
        source_tree=r_tree,
        authorized_at_ms=pinned_at_ms + 2_000,
    )
    authorization_path = (
        repo
        / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    _write(authorization_path, authorization)
    artifact_root = authorization_path.parent
    _rebind_component_digest(
        artifact_root,
        PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
    )
    heads["P019"] = _sequential_commit(
        repo,
        "P019",
        convergence_module.R_TO_P019_CHANGED_PATHS,
    )

    root_pin_snapshot = convergence_module.load_local_profile_lifecycle_root_pin(
        root_pin_path
    )
    witness_snapshot = convergence_module.load_local_operator_lifecycle_witness(
        witness_path
    )
    authorization_raw = authorization_path.read_bytes()
    lifecycle_authority = (
        convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
            authorization
        ).acceptance_review_authority(
            raw_sha256=(
                "sha256:" + hashlib.sha256(authorization_raw).hexdigest()
            ),
            lifecycle_witness=witness_snapshot,
            root_pin=root_pin_snapshot,
        )
    )
    resolved_phase_times = {
        "A019": "2026-08-08T19:00:03Z",
        "A030": "2026-08-08T19:00:04Z",
        "P031": "2026-08-08T19:00:05Z",
        "A031": "2026-08-08T19:00:06Z",
        "A032": "2026-08-08T19:00:07Z",
        "A023": "2026-08-08T19:00:08Z",
        "A027": "2026-08-08T19:00:08Z",
        "L": "2026-08-08T19:00:09Z",
    }
    if phase_times is not None:
        resolved_phase_times.update(phase_times)

    def sign_phase_receipt(
        payload: dict[str, object],
        *,
        created_at: str,
        implementer: str,
    ) -> None:
        payload["created_at"] = created_at
        payload["review"] = {
            **_receipt_review_authority(dict(lifecycle_authority)),
            "implementer_identity": implementer,
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        }
        _sign_operator_receipt(payload, active_key)

    board_path = PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    manifest_path = convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH

    def prior_artifacts(parent_phase: str) -> dict[str, str]:
        return {
            relative_path: (
                "sha256:" + hashlib.sha256((repo / relative_path).read_bytes()).hexdigest()
            )
            for relative_path in convergence_module._sequential_artifacts_after(
                parent_phase
            )
        }

    for phase in convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES[3:]:
        parent_phase = convergence_module.SEQUENTIAL_PHASE_PARENT[phase]
        parent_head = heads[parent_phase]
        parent_tree = _sequential_git(
            repo,
            "rev-parse",
            f"{parent_head}^{{tree}}",
        )
        parent_manifest_raw = (repo / manifest_path).read_bytes()
        parent_manifest = json.loads(parent_manifest_raw)
        changed_paths = convergence_module.SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
        phase_artifacts: dict[str, bytes] = {}
        parent_payload = _phase_parent_payload(
            repo,
            phase=parent_phase,
            head=parent_head,
            prior_artifacts=prior_artifacts(parent_phase),
        )

        if phase == "A019":
            payload = _minimal_operator_receipt("ASE3-019")
            payload["merge"] = {
                "acceptance_parent_head": parent_head,
                "acceptance_parent_tree": parent_tree,
            }
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["A019"],
                implementer="codex:ase3-019-product",
            )
            phase_artifacts[
                f"{convergence_module._CONVERGENCE_RELATIVE_ROOT}/"
                f"{OPERATOR_SALVAGE_RECEIPT_019_FILENAME}"
            ] = _sequential_json_bytes(payload)
        elif phase == "A030":
            payload = _minimal_operator_receipt("ASE3-030")
            payload["acceptance_parent"] = parent_payload
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["A030"],
                implementer="codex:ase3-030-product",
            )
            phase_artifacts[HERMETIC_IDENTITY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                _sequential_json_bytes(payload)
            )
        elif phase == "P031":
            payload = _minimal_sequential_native_receipt(
                fields=(
                    convergence_module._NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_REQUIRED_FIELDS
                ),
                schema=NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
                phase=phase,
                parent=parent_payload,
            )
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["P031"],
                implementer="codex:ase3-031-authorization",
            )
            phase_artifacts[NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH] = (
                _sequential_json_bytes(payload)
            )
        elif phase == "A031":
            payload = _minimal_sequential_native_receipt(
                fields=convergence_module._NATIVE_DEPENDENCY_ACCEPTANCE_REQUIRED_FIELDS,
                schema=NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_SCHEMA,
                phase=phase,
                parent=parent_payload,
            )
            payload["preload_evidence"] = {
                "runtime_effect_started_at": (
                    a031_effect_time or resolved_phase_times["A031"]
                )
            }
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["A031"],
                implementer="codex:ase3-031-product",
            )
            phase_artifacts[NATIVE_DEPENDENCY_ACCEPTANCE_RECEIPT_RELATIVE_PATH] = (
                _sequential_json_bytes(payload)
            )
        elif phase == "A032":
            payload = _minimal_sequential_native_receipt(
                fields=convergence_module._DUCKDB_POLICY_ACCEPTANCE_REQUIRED_FIELDS,
                schema=DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_SCHEMA,
                phase=phase,
                parent=parent_payload,
            )
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["A032"],
                implementer="codex:ase3-032-product",
            )
            phase_artifacts[
                DUCKDB_CONNECTION_POLICY_ACCEPTANCE_RECEIPT_RELATIVE_PATH
            ] = _sequential_json_bytes(payload)
        elif phase == "A023/027":
            for task_id in ("ASE3-023", "ASE3-027"):
                payload = _minimal_operator_receipt(task_id)
                payload["acceptance_parent"] = parent_payload
                sign_phase_receipt(
                    payload,
                    created_at=resolved_phase_times[
                        "A023" if task_id == "ASE3-023" else "A027"
                    ],
                    implementer=f"codex:{task_id.lower()}-repair",
                )
                filename = convergence_module._ACCEPTANCE_TASK_CONTRACTS[task_id][
                    "filename"
                ]
                phase_artifacts[
                    f"{convergence_module._CONVERGENCE_RELATIVE_ROOT}/{filename}"
                ] = _sequential_json_bytes(payload)
        else:
            acceptance_receipts = {
                Path(relative_path).name: (
                    "sha256:"
                    + hashlib.sha256((repo / relative_path).read_bytes()).hexdigest()
                )
                for relative_path in convergence_module._sequential_artifacts_after(
                    "A023/027"
                )
                if Path(relative_path).name
                in convergence_module.SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES
            }
            payload = {
                field: {}
                for field in convergence_module._RELOAD_RECEIPT_REQUIRED_FIELDS
            }
            payload.update(
                {
                    "schema": (
                        convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA
                    ),
                    "created_at": "2026-08-08T20:00:03Z",
                    "board_namespace": BOARD_NAMESPACE,
                    "acceptance_parent": {
                        **parent_payload,
                        "acceptance_receipts": acceptance_receipts,
                    },
                    "authorization": {
                        "target_generation_id": "sha256:" + ("b" * 64),
                        "target_generation_number": 8,
                    },
                }
            )
            sign_phase_receipt(
                payload,
                created_at=resolved_phase_times["L"],
                implementer="codex:ase3-022-reload-preparation",
            )
            phase_artifacts[
                PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
            ] = _sequential_json_bytes(payload)

        artifact_hashes: dict[str, str] = {}
        for relative_path, raw in phase_artifacts.items():
            _sequential_write(repo, relative_path, raw)
            artifact_hashes[relative_path] = (
                "sha256:" + hashlib.sha256(raw).hexdigest()
            )
        if board_path in changed_paths:
            _sequential_write(
                repo,
                board_path,
                convergence_module._status_only_sequential_phase_board(
                    (repo / board_path).read_bytes(),
                    phase,
                ),
            )

        child_manifest = dict(parent_manifest)
        child_manifest["created_at"] = "2026-08-08T20:00:02Z"
        if phase == "L":
            child_manifest["schema"] = (
                convergence_module.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
            )
            child_manifest["reload"] = {
                "phase": "provider_attempt_daemon_reload",
                "acceptance_head": parent_head,
                "acceptance_tree": parent_tree,
                "receipt": {
                    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: (
                        artifact_hashes[
                            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
                        ]
                    )
                },
                "task": dict(convergence_module._RELOAD_TASK_CONTRACT),
                "accepted_task_statuses": {
                    task_id: "completed"
                    for task_id in convergence_module.SEQUENTIAL_ACCEPTANCE_TASK_IDS
                },
                "reload_gate_completed": True,
                "launch_authorization_only": True,
                "post_launch_birth_receipt_required": True,
                "post_launch_birth_receipt_schema": (
                    convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
                ),
            }
        else:
            child_manifest["schema"] = ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
            child_manifest["acceptance"] = {
                "phase": phase,
                "parent_phase": parent_phase,
                "parent_head": parent_head,
                "parent_tree": parent_tree,
                "parent_manifest_sha256": (
                    "sha256:" + hashlib.sha256(parent_manifest_raw).hexdigest()
                ),
                "artifacts": artifact_hashes,
                "task_statuses": (
                    convergence_module._sequential_task_statuses_after(phase)
                ),
                "reload_gate_status": "blocked",
                "pre_launch_authorization_only": phase == "P031",
                "runtime_effect_claimed": (
                    convergence_module.SEQUENTIAL_PHASE_RUNTIME_EFFECT_CLAIMS[phase]
                ),
            }
        _sequential_write(
            repo,
            manifest_path,
            _sequential_json_bytes(child_manifest),
        )
        heads[phase] = _sequential_commit(repo, phase, changed_paths)

    return repo, heads, {
        "root_identity_did": root_did,
        "reviewer_final_values": reviewer_final_values,
        "root_key": root_key,
        "active_key": active_key,
        "lifecycle_authority": dict(lifecycle_authority),
        "root_pin": root_pin,
        "pinned_at_ms": pinned_at_ms,
    }


def _sequential_transition_repository(
    tmp_path: Path,
    *,
    q_manifest_updates: Mapping[str, object] | None = None,
    p019_manifest_updates: Mapping[str, object] | None = None,
    root_pin_extra_path: bool = False,
    preparation_extra_path: bool = False,
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "sequential-acceptance"
    repo.mkdir()
    _sequential_git(repo, "init", "-q")
    _sequential_git(repo, "config", "user.name", "Sequential Audit")
    _sequential_git(repo, "config", "user.email", "sequential@example.invalid")

    board_path = PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    manifest_path = convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH
    authorization_path = (
        convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
    )
    q_authorization = _sequential_json_bytes(
        {"schema": convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_SCHEMA}
    )
    _sequential_write(repo, board_path, TASKBOARD_PATH.read_bytes())
    _sequential_write(repo, authorization_path, q_authorization)
    q_manifest: dict[str, object] = {
        "schema": convergence_module.CONVERGENCE_MANIFEST_SCHEMA,
        "components": {
            PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME: (
                "sha256:" + hashlib.sha256(q_authorization).hexdigest()
            )
        },
    }
    if q_manifest_updates is not None:
        q_manifest.update(q_manifest_updates)
    _sequential_write(repo, manifest_path, _sequential_json_bytes(q_manifest))
    heads = {
        "Q": _sequential_commit(
            repo,
            "Q",
            (board_path, authorization_path, manifest_path),
        )
    }

    root_path = convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    _sequential_write(repo, root_path, b'{"phase":"R"}\n')
    r_paths = list(convergence_module.Q_TO_R_CHANGED_PATHS)
    if root_pin_extra_path:
        extra_path = "unexpected-root-pin-path.txt"
        _sequential_write(repo, extra_path, b"unexpected\n")
        r_paths.append(extra_path)
    heads["R"] = _sequential_commit(repo, "R", tuple(r_paths))

    witness_path = convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
    p019_authorization = _sequential_json_bytes(
        {
            "schema": (
                convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA
            )
        }
    )
    _sequential_write(repo, witness_path, b'{"phase":"P019"}\n')
    _sequential_write(repo, authorization_path, p019_authorization)
    p019_manifest = json.loads((repo / manifest_path).read_text())
    p019_components = p019_manifest["components"]
    assert isinstance(p019_components, dict)
    p019_components[PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME] = (
        "sha256:" + hashlib.sha256(p019_authorization).hexdigest()
    )
    if p019_manifest_updates is not None:
        p019_manifest.update(p019_manifest_updates)
    _sequential_write(repo, manifest_path, _sequential_json_bytes(p019_manifest))
    p019_paths = list(convergence_module.R_TO_P019_CHANGED_PATHS)
    if preparation_extra_path:
        extra_path = "unexpected-preparation-path.txt"
        _sequential_write(repo, extra_path, b"unexpected\n")
        p019_paths.append(extra_path)
    heads["P019"] = _sequential_commit(repo, "P019", tuple(p019_paths))

    for phase in convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES[3:]:
        parent_phase = convergence_module.SEQUENTIAL_PHASE_PARENT[phase]
        parent_head = heads[parent_phase]
        parent_tree = _sequential_git(repo, "rev-parse", f"{parent_head}^{{tree}}")
        parent_manifest_raw = (repo / manifest_path).read_bytes()
        parent_manifest = json.loads(parent_manifest_raw)
        changed_paths = convergence_module.SEQUENTIAL_PHASE_CHANGED_PATHS[phase]
        artifact_hashes: dict[str, str] = {}
        for relative_path in changed_paths:
            if relative_path in {manifest_path, board_path}:
                continue
            if (
                phase == "L"
                and relative_path
                == convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
            ):
                raw = _sequential_json_bytes(
                    {
                        "phase": phase,
                        "artifact": relative_path,
                        "acceptance_parent": {
                            "head": parent_head,
                            "tree": parent_tree,
                        },
                    }
                )
            else:
                raw = _sequential_json_bytes(
                    {"phase": phase, "artifact": relative_path}
                )
            _sequential_write(repo, relative_path, raw)
            artifact_hashes[relative_path] = (
                "sha256:" + hashlib.sha256(raw).hexdigest()
            )
        if board_path in changed_paths:
            board_raw = (repo / board_path).read_bytes()
            _sequential_write(
                repo,
                board_path,
                convergence_module._status_only_sequential_phase_board(
                    board_raw,
                    phase,
                ),
            )

        child_manifest = dict(parent_manifest)
        child_manifest["created_at"] = "2026-08-08T20:00:00Z"
        if phase == "L":
            child_manifest["schema"] = (
                convergence_module.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
            )
            child_manifest["reload"] = {
                "phase": "provider_attempt_daemon_reload",
                "acceptance_head": parent_head,
                "acceptance_tree": parent_tree,
                "receipt": {
                    PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: (
                        artifact_hashes[
                            PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
                        ]
                    )
                },
                "task": dict(convergence_module._RELOAD_TASK_CONTRACT),
                "accepted_task_statuses": {
                    task_id: "completed"
                    for task_id in convergence_module.SEQUENTIAL_ACCEPTANCE_TASK_IDS
                },
                "reload_gate_completed": True,
                "launch_authorization_only": True,
                "post_launch_birth_receipt_required": True,
                "post_launch_birth_receipt_schema": (
                    convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
                ),
            }
        else:
            child_manifest["schema"] = ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA
            child_manifest["acceptance"] = {
                "phase": phase,
                "parent_phase": parent_phase,
                "parent_head": parent_head,
                "parent_tree": parent_tree,
                "parent_manifest_sha256": (
                    "sha256:" + hashlib.sha256(parent_manifest_raw).hexdigest()
                ),
                "artifacts": artifact_hashes,
                "task_statuses": (
                    convergence_module._sequential_task_statuses_after(phase)
                ),
                "reload_gate_status": "blocked",
                "pre_launch_authorization_only": phase == "P031",
                "runtime_effect_claimed": (
                    convergence_module.SEQUENTIAL_PHASE_RUNTIME_EFFECT_CLAIMS[phase]
                ),
            }
        _sequential_write(repo, manifest_path, _sequential_json_bytes(child_manifest))
        heads[phase] = _sequential_commit(repo, phase, changed_paths)
    return repo, heads


def _secure_phase_files(repo: Path) -> None:
    for relative_path in convergence_module._PROTECTED_PATHS:
        path = repo / relative_path
        if path.exists() and not path.is_symlink():
            path.chmod(path.stat().st_mode & 0o755)


def _direct_phase_packet_errors(
    repo: Path,
    *,
    phase: str,
    root_identity_did: str,
) -> list[str]:
    artifact_root = (
        repo / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    manifest_raw = (artifact_root / MANIFEST_FILENAME).read_bytes()
    authorization_raw = (
        artifact_root / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
    ).read_bytes()
    errors, _ = convergence_module._validate_sequential_phase_packet(
        phase=phase,
        artifact_root=artifact_root,
        manifest=ConvergenceManifest.from_dict(json.loads(manifest_raw)),
        repo_root=repo,
        fallback_authorization=(
            convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
                json.loads(authorization_raw)
            )
        ),
        fallback_authorization_raw=authorization_raw,
        manifest_raw=manifest_raw,
        taskboard_raw=(repo / PROMPT_V3_TASKBOARD_RELATIVE_PATH).read_bytes(),
        expected_root_identity_did=root_identity_did,
    )
    return errors


def test_every_sequential_phase_validates_through_real_top_level_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    reviewer_final_values = context["reviewer_final_values"]
    assert isinstance(reviewer_final_values, dict)
    monkeypatch.setattr(
        convergence_module,
        "_ACCEPTANCE_REVIEWER_FINAL_VALUES",
        reviewer_final_values,
    )

    real_root_validator = (
        convergence_module.validate_local_profile_lifecycle_root_pin
    )

    def validate_fixture_root_pin(
        payload: Mapping[str, object],
        **kwargs: object,
    ) -> tuple[str, ...]:
        kwargs["expected_root_identity_did"] = context["root_identity_did"]
        return real_root_validator(payload, **kwargs)

    monkeypatch.setattr(
        convergence_module,
        "validate_local_profile_lifecycle_root_pin",
        validate_fixture_root_pin,
    )
    for validator_name in (
        "validate_operator_salvage_receipt_019",
        "validate_hermetic_identity_acceptance_receipt",
        "validate_native_dependency_launch_authorization",
        "validate_native_dependency_acceptance_receipt",
        "validate_duckdb_connection_policy_acceptance_receipt",
        "validate_operator_repair_acceptance_receipt",
        "validate_provider_attempt_reload_receipt",
    ):
        monkeypatch.setattr(
            convergence_module,
            validator_name,
            lambda *args, **kwargs: (),
        )

    packet_phases: list[str] = []
    real_phase_packet_validator = convergence_module._validate_sequential_phase_packet

    def validate_tracked_phase_packet(
        **kwargs: object,
    ) -> tuple[list[str], tuple[str, ...]]:
        packet_phases.append(str(kwargs["phase"]))
        return real_phase_packet_validator(**kwargs)

    monkeypatch.setattr(
        convergence_module,
        "_validate_sequential_phase_packet",
        validate_tracked_phase_packet,
    )

    artifact_root = (
        repo / "data/agent_supervisor/prompt_only_self_improvement_v3/convergence"
    )
    for phase in convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES:
        _sequential_git(repo, "reset", "--hard", heads[phase])
        _secure_phase_files(repo)
        report = validate_convergence_artifacts(
            artifact_root,
            repo_root=repo,
            check_repository=True,
            taskboard_path=repo / PROMPT_V3_TASKBOARD_RELATIVE_PATH,
        )
        assert report.valid is True, (phase, report.errors)
        assert report.errors == ()
    assert packet_phases == list(convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES[1:])

    _sequential_git(repo, "reset", "--hard", heads["A032"])
    _secure_phase_files(repo)
    board = repo / PROMPT_V3_TASKBOARD_RELATIVE_PATH
    board_text = board.read_text(encoding="utf-8")
    task_start = board_text.index("## ASE3-019 ")
    task_end = board_text.index("\n## ASE3-", task_start + 1)
    task_block = board_text[task_start:task_end]
    board.write_text(
        board_text[:task_start]
        + task_block.replace("- Completion: manual", "- Completion: forged", 1)
        + board_text[task_end:],
        encoding="utf-8",
    )
    block_report = validate_convergence_artifacts(
        artifact_root,
        repo_root=repo,
        check_repository=True,
        taskboard_path=board,
    )
    assert block_report.valid is False
    assert any(
        "protected_task_block_bytes.ASE3-019: protected task block bytes changed"
        in error
        for error in block_report.errors
    )

    _sequential_git(repo, "reset", "--hard", heads["A032"])
    _secure_phase_files(repo)
    board_text = board.read_text(encoding="utf-8")
    task_start = board_text.index("## ASE3-023 ")
    status_start = board_text.index("- Status: todo", task_start)
    board.write_text(
        board_text[:status_start]
        + "- Status: completed"
        + board_text[status_start + len("- Status: todo") :],
        encoding="utf-8",
    )
    status_report = validate_convergence_artifacts(
        artifact_root,
        repo_root=repo,
        check_repository=True,
        taskboard_path=board,
    )
    assert status_report.valid is False
    assert any(
        "operator_acceptance.phase.A032.task_statuses: exact phase order required"
        in error
        for error in status_report.errors
    )
    assert any(
        "protected_task_block_bytes.ASE3-023.status: exact A032 phase status required"
        in error
        for error in status_report.errors
    )

    _sequential_git(repo, "reset", "--hard", heads["Q"])
    _secure_phase_files(repo)
    q_manifest_raw = (artifact_root / MANIFEST_FILENAME).read_bytes()
    q_authorization_raw = (
        artifact_root / PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME
    ).read_bytes()
    q_errors, q_checked = real_phase_packet_validator(
        phase="Q",
        artifact_root=artifact_root,
        manifest=ConvergenceManifest.from_dict(json.loads(q_manifest_raw)),
        repo_root=repo,
        fallback_authorization=(
            convergence_module.ProviderFallbackPolicyAuthorization.from_dict(
                json.loads(q_authorization_raw)
            )
        ),
        fallback_authorization_raw=q_authorization_raw,
        manifest_raw=q_manifest_raw,
        taskboard_raw=(repo / PROMPT_V3_TASKBOARD_RELATIVE_PATH).read_bytes(),
    )
    assert q_errors == []
    assert q_checked == ()


def _patch_sequential_receipt_semantics(
    monkeypatch: pytest.MonkeyPatch,
    context: Mapping[str, object],
) -> None:
    reviewer_final_values = context["reviewer_final_values"]
    assert isinstance(reviewer_final_values, dict)
    monkeypatch.setattr(
        convergence_module,
        "_ACCEPTANCE_REVIEWER_FINAL_VALUES",
        reviewer_final_values,
    )
    for validator_name in (
        "validate_operator_salvage_receipt_019",
        "validate_hermetic_identity_acceptance_receipt",
        "validate_native_dependency_launch_authorization",
        "validate_native_dependency_acceptance_receipt",
        "validate_duckdb_connection_policy_acceptance_receipt",
        "validate_operator_repair_acceptance_receipt",
        "validate_provider_attempt_reload_receipt",
    ):
        monkeypatch.setattr(
            convergence_module,
            validator_name,
            lambda *args, **kwargs: (),
        )


@pytest.mark.parametrize(
    ("phase_times", "a031_effect_time", "expected_edge"),
    (
        (
            {"A030": "2026-08-08T19:00:02Z"},
            None,
            "A019_to_A030",
        ),
        (
            {"P031": "2026-08-08T19:00:03Z"},
            None,
            "A030_to_P031",
        ),
        (
            {"A031": "2026-08-08T19:00:04Z"},
            None,
            "P031_to_A031",
        ),
        (
            {},
            "2026-08-08T19:00:04Z",
            "P031_to_A031",
        ),
        (
            {"A032": "2026-08-08T19:00:05Z"},
            None,
            "A031_to_A032",
        ),
        (
            {"A023": "2026-08-08T19:00:06Z"},
            None,
            "A032_to_A023",
        ),
        (
            {"A027": "2026-08-08T19:00:06Z"},
            None,
            "A032_to_A027",
        ),
        (
            {"A023": "2026-08-08T19:00:10Z"},
            None,
            "A023_to_L",
        ),
        (
            {"A027": "2026-08-08T19:00:10Z"},
            None,
            "A027_to_L",
        ),
    ),
)
def test_sequential_signed_receipt_chronology_rejects_every_inverted_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase_times: Mapping[str, str],
    a031_effect_time: str | None,
    expected_edge: str,
) -> None:
    repo, _, context = _top_level_sequential_phase_repository(
        tmp_path,
        phase_times=phase_times,
        a031_effect_time=a031_effect_time,
    )
    _patch_sequential_receipt_semantics(monkeypatch, context)
    _secure_phase_files(repo)

    errors = _direct_phase_packet_errors(
        repo,
        phase="L",
        root_identity_did=str(context["root_identity_did"]),
    )

    assert any(
        f"protected_acceptance.chronology.{expected_edge}:" in error
        for error in errors
    ), errors
    assert not any("cryptographic verification failed" in error for error in errors)
    assert not any("committed_bytes:" in error for error in errors)


@pytest.mark.parametrize(
    "phase_times",
    (
        {
            "A019": "2026-08-08T19:00:02Z",
            "A030": "2026-08-08T19:00:02Z",
            "P031": "2026-08-08T19:00:02Z",
            "A031": "2026-08-08T19:00:02Z",
            "A032": "2026-08-08T19:00:02Z",
            "A023": "2026-08-08T19:00:02Z",
            "A027": "2026-08-08T19:00:02Z",
            "L": "2026-08-08T19:00:02Z",
        },
        {
            "A023": "2026-08-08T19:00:08Z",
            "A027": "2026-08-08T19:00:07Z",
            "L": "2026-08-08T19:00:09Z",
        },
    ),
)
def test_sequential_signed_receipt_chronology_allows_equal_and_unordered_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase_times: Mapping[str, str],
) -> None:
    repo, _, context = _top_level_sequential_phase_repository(
        tmp_path,
        phase_times=phase_times,
        a031_effect_time=phase_times.get("A031"),
    )
    _patch_sequential_receipt_semantics(monkeypatch, context)
    _secure_phase_files(repo)

    errors = _direct_phase_packet_errors(
        repo,
        phase="L",
        root_identity_did=str(context["root_identity_did"]),
    )

    assert errors == []


@pytest.mark.parametrize(
    "invalid_time",
    (True, 0, "2026-02-30T19:00:04Z"),
)
def test_sequential_signed_receipt_chronology_rejects_coerced_or_noncalendar_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_time: object,
) -> None:
    repo, _, context = _top_level_sequential_phase_repository(
        tmp_path,
        phase_times={"A030": invalid_time},  # type: ignore[dict-item]
    )
    _patch_sequential_receipt_semantics(monkeypatch, context)
    _secure_phase_files(repo)

    errors = _direct_phase_packet_errors(
        repo,
        phase="L",
        root_identity_did=str(context["root_identity_did"]),
    )

    assert any(
        "protected_acceptance.chronology.A030.created_at: "
        "valid non-coerced UTC calendar timestamp required" in error
        for error in errors
    )


def _commit_post_l_birth_receipt(
    repo: Path,
    *,
    heads: Mapping[str, str],
    context: Mapping[str, object],
    created_at: str,
    mutation: str | None = None,
) -> tuple[dict[str, object], bytes, str]:
    reload_head = heads["L"]
    reload_tree = _sequential_git(repo, "rev-parse", f"{reload_head}^{{tree}}")
    reload_path = (
        repo / convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    )
    reload_raw = reload_path.read_bytes()
    reload_payload = json.loads(reload_raw)
    authorization = reload_payload["authorization"]
    assert isinstance(authorization, dict)
    active_key = context["active_key"]
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(active_key, Ed25519PrivateKey)
    assert isinstance(lifecycle_authority, dict)
    payload: dict[str, object] = {
        "schema": (
            convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
        ),
        "created_at": created_at,
        "board_namespace": BOARD_NAMESPACE,
        "phase": "post-L",
        "reload_authorization": {
            "path": (
                convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
            ),
            "sha256": "sha256:" + hashlib.sha256(reload_raw).hexdigest(),
            "head": reload_head,
            "tree": reload_tree,
            "phase": "L",
        },
        "generation": {
            "generation_id": authorization["target_generation_id"],
            "generation_number": authorization["target_generation_number"],
        },
        "process_birth": {
            "effect_started_at": created_at,
            "process_started_at": created_at,
            "runtime_effect_started": True,
        },
        "review": {
            **_receipt_review_authority(lifecycle_authority),
            "implementer_identity": "codex:provider-attempt-generation-birth",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(
            convergence_module._PROVIDER_ATTEMPT_GENERATION_BIRTH_DENIALS
        ),
    }
    if mutation == "reload_binding":
        reload_binding = payload["reload_authorization"]
        assert isinstance(reload_binding, dict)
        reload_binding["sha256"] = "sha256:" + ("0" * 64)
    elif mutation == "generation":
        generation = payload["generation"]
        assert isinstance(generation, dict)
        generation["generation_id"] = "sha256:" + ("1" * 64)
    elif mutation == "extra":
        payload["self_consistent_extra"] = True
    _sign_operator_receipt(payload, active_key)
    if mutation == "wrong_parent":
        _sequential_git(
            repo,
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "unaccepted post-L parent",
        )
    raw = _sequential_json_bytes(payload)
    _sequential_write(
        repo,
        convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,
        raw,
    )
    birth_head = _sequential_commit(
        repo,
        "post-L-birth",
        (
            convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH,
        ),
    )
    return payload, raw, birth_head


def _amend_l_with_resigned_reload_parent(
    repo: Path,
    *,
    heads: Mapping[str, str],
    context: Mapping[str, object],
    parent_head: str,
    parent_tree: str,
) -> dict[str, str]:
    """Forge a self-consistent L digest/signature around the wrong authority."""

    reload_path = (
        repo / convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    )
    reload_payload = json.loads(reload_path.read_bytes())
    acceptance_parent = reload_payload["acceptance_parent"]
    assert isinstance(acceptance_parent, dict)
    acceptance_parent["head"] = parent_head
    acceptance_parent["tree"] = parent_tree
    active_key = context["active_key"]
    assert isinstance(active_key, Ed25519PrivateKey)
    _sign_operator_receipt(reload_payload, active_key)
    reload_raw = _sequential_json_bytes(reload_payload)
    _sequential_write(
        repo,
        convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
        reload_raw,
    )

    manifest_path = repo / convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_bytes())
    reload_manifest = manifest["reload"]
    assert isinstance(reload_manifest, dict)
    receipt_binding = reload_manifest["receipt"]
    assert isinstance(receipt_binding, dict)
    receipt_binding[PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME] = (
        "sha256:" + hashlib.sha256(reload_raw).hexdigest()
    )
    _sequential_write(
        repo,
        convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH,
        _sequential_json_bytes(manifest),
    )
    _sequential_git(
        repo,
        "add",
        "--",
        convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
        convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH,
    )
    _sequential_git(repo, "commit", "-q", "--amend", "--no-edit")
    amended_heads = dict(heads)
    amended_heads["L"] = _sequential_git(repo, "rev-parse", "HEAD")
    return amended_heads


@pytest.mark.parametrize(
    ("created_at", "expected_error"),
    (
        ("2026-08-08T19:00:09Z", None),
        (
            "2026-08-08T19:00:08Z",
            "chronology: post-L birth predates signed L authority",
        ),
    ),
)
def test_post_l_birth_chronology_allows_equality_and_rejects_resigned_inversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    created_at: str,
    expected_error: str | None,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=heads,
        context=context,
        created_at=created_at,
    )
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)

    errors = convergence_module.validate_provider_attempt_generation_birth_receipt(
        payload,
        birth_receipt_raw=raw,
        birth_head=birth_head,
        phase_heads=heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )

    if expected_error is None:
        assert errors == ()
    else:
        assert any(expected_error in error for error in errors)
        assert not any("cryptographic verification failed" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("wrong_parent", "birth_head: exact direct L child required"),
        ("reload_binding", "reload_authorization.sha256: expected"),
        ("generation", "generation.generation_id: expected"),
        ("extra", "exact key population required"),
        ("raw", "committed_bytes: exact Git birth bytes required"),
    ),
)
def test_post_l_birth_rejects_replayed_or_self_consistent_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=heads,
        context=context,
        created_at="2026-08-08T19:00:09Z",
        mutation=None if mutation == "raw" else mutation,
    )
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)

    errors = convergence_module.validate_provider_attempt_generation_birth_receipt(
        payload,
        birth_receipt_raw=raw + (b" " if mutation == "raw" else b""),
        birth_head=birth_head,
        phase_heads=heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )

    assert any(expected_error in error for error in errors), errors


@pytest.mark.parametrize("wrong_parent", ("Q", "A023/027-tree"))
def test_post_l_birth_and_sequence_reject_resigned_l_parent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_parent: str,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    if wrong_parent == "Q":
        parent_head = heads["Q"]
        parent_tree = _sequential_git(repo, "rev-parse", f"{parent_head}^{{tree}}")
    else:
        parent_head = heads["A023/027"]
        parent_tree = _sequential_git(repo, "rev-parse", f"{heads['Q']}^{{tree}}")
    amended_heads = _amend_l_with_resigned_reload_parent(
        repo,
        heads=heads,
        context=context,
        parent_head=parent_head,
        parent_tree=parent_tree,
    )
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=amended_heads,
        context=context,
        created_at="2026-08-08T19:00:09Z",
    )

    sequence_errors = convergence_module.validate_protected_acceptance_sequence(
        repo_root=repo,
        phase_heads=amended_heads,
        through_phase="L",
    )
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)
    birth_errors = (
        convergence_module.validate_provider_attempt_generation_birth_receipt(
            payload,
            birth_receipt_raw=raw,
            birth_head=birth_head,
            phase_heads=amended_heads,
            repo_root=repo,
            lifecycle_authority=lifecycle_authority,
        )
    )

    expected = (
        "protected_acceptance.L.reload_receipt.acceptance_parent: "
        "exact A023/027 head/tree required"
    )
    assert expected in sequence_errors
    assert any(expected in error for error in birth_errors)
    assert not any("cryptographic verification failed" in error for error in birth_errors)


def test_post_l_birth_rejects_git_replacement_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=heads,
        context=context,
        created_at="2026-08-08T19:00:09Z",
    )
    _sequential_git(repo, "replace", heads["L"], heads["Q"])
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)

    errors = convergence_module.validate_provider_attempt_generation_birth_receipt(
        payload,
        birth_receipt_raw=raw,
        birth_head=birth_head,
        phase_heads=heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )

    assert any("Git replacement objects are forbidden" in error for error in errors)


def test_post_l_birth_rejects_live_l_receipt_byte_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=heads,
        context=context,
        created_at="2026-08-08T19:00:09Z",
    )
    reload_path = (
        repo / convergence_module.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
    )
    reload_path.write_bytes(reload_path.read_bytes() + b" ")
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)

    errors = convergence_module.validate_provider_attempt_generation_birth_receipt(
        payload,
        birth_receipt_raw=raw,
        birth_head=birth_head,
        phase_heads=heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )

    assert any(
        "reload_authorization.live_receipt: exact committed L bytes required"
        in error
        for error in errors
    )


def test_post_l_birth_propagates_full_l_packet_semantic_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    _patch_sequential_receipt_semantics(monkeypatch, context)
    semantic_sentinel = "committed L semantic sentinel"
    monkeypatch.setattr(
        convergence_module,
        "validate_provider_attempt_reload_receipt",
        lambda *args, **kwargs: (semantic_sentinel,),
    )
    payload, raw, birth_head = _commit_post_l_birth_receipt(
        repo,
        heads=heads,
        context=context,
        created_at="2026-08-08T19:00:09Z",
    )
    lifecycle_authority = context["lifecycle_authority"]
    assert isinstance(lifecycle_authority, dict)

    errors = convergence_module.validate_provider_attempt_generation_birth_receipt(
        payload,
        birth_receipt_raw=raw,
        birth_head=birth_head,
        phase_heads=heads,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )

    assert any(
        f"validated_L_packet.{semantic_sentinel}" in error for error in errors
    )


@pytest.mark.parametrize(
    ("mutation", "expected_fragments"),
    (
        ("root_other_q", ("exact Q phase head required", "exact Q phase tree required")),
        ("witness_auth_other_r", ("signed base mismatch", "authorization_source")),
        ("witness_predates_root", ("observed_at_ms: predates root-pin commit",)),
        ("authorization_predates_witness", ("outside witness",)),
    ),
)
def test_sequential_lifecycle_rejects_replayed_phase_parents_and_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_fragments: tuple[str, ...],
) -> None:
    repo, heads, context = _top_level_sequential_phase_repository(tmp_path)
    reviewer_final_values = context["reviewer_final_values"]
    root_key = context["root_key"]
    active_key = context["active_key"]
    root_pin = context["root_pin"]
    pinned_at_ms = context["pinned_at_ms"]
    root_identity_did = context["root_identity_did"]
    assert isinstance(reviewer_final_values, dict)
    assert isinstance(root_key, Ed25519PrivateKey)
    assert isinstance(active_key, Ed25519PrivateKey)
    assert isinstance(root_pin, dict)
    assert isinstance(pinned_at_ms, int)
    assert isinstance(root_identity_did, str)
    monkeypatch.setattr(
        convergence_module,
        "_ACCEPTANCE_REVIEWER_FINAL_VALUES",
        reviewer_final_values,
    )

    root_pin_path = (
        repo / convergence_module.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    )
    if mutation == "root_other_q":
        _sequential_git(repo, "reset", "--hard", heads["R"])
        payload = _load(root_pin_path)
        other_head = _sequential_git(repo, "rev-parse", f"{heads['Q']}^^")
        payload["base_head"] = other_head
        payload["base_tree"] = _sequential_git(
            repo,
            "rev-parse",
            f"{other_head}^{{tree}}",
        )
        payload_without_id = dict(payload)
        payload_without_id.pop("pin_id")
        payload["pin_id"] = convergence_module._canonical_sha256(payload_without_id)
        _write(root_pin_path, payload)
        _sequential_git(repo, "add", "--", str(root_pin_path))
        _sequential_git(repo, "commit", "-q", "--amend", "--no-edit")
        phase = "R"
    else:
        _sequential_git(repo, "reset", "--hard", heads["P019"])
        if mutation == "witness_auth_other_r":
            source_head = heads["Q"]
            source_tree = _sequential_git(
                repo,
                "rev-parse",
                f"{source_head}^{{tree}}",
            )
            observed_at_ms = pinned_at_ms + 1_000
            authorized_at_ms = pinned_at_ms + 2_000
        elif mutation == "witness_predates_root":
            source_head = heads["R"]
            source_tree = _sequential_git(
                repo,
                "rev-parse",
                f"{source_head}^{{tree}}",
            )
            observed_at_ms = pinned_at_ms - 1_000
            authorized_at_ms = pinned_at_ms - 500
        else:
            source_head = heads["R"]
            source_tree = _sequential_git(
                repo,
                "rev-parse",
                f"{source_head}^{{tree}}",
            )
            observed_at_ms = pinned_at_ms + 1_000
            authorized_at_ms = observed_at_ms - 1

        witness, _ = _lifecycle_witness_payload(
            root_key=root_key,
            active_key=active_key,
            base_head=source_head,
            base_tree=source_tree,
            observed_at_ms=observed_at_ms,
        )
        witness_path = (
            repo
            / convergence_module.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
        )
        _write(witness_path, witness)
        authorization = _fallback_authorization_v2_payload(
            active_key=active_key,
            witness=witness,
            witness_sha256=(
                "sha256:" + hashlib.sha256(witness_path.read_bytes()).hexdigest()
            ),
            root_pin=root_pin,
            root_pin_sha256=(
                "sha256:" + hashlib.sha256(root_pin_path.read_bytes()).hexdigest()
            ),
            source_head=source_head,
            source_tree=source_tree,
            authorized_at_ms=authorized_at_ms,
        )
        authorization_path = (
            repo
            / convergence_module.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
        )
        _write(authorization_path, authorization)
        _rebind_component_digest(
            authorization_path.parent,
            PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME,
        )
        _sequential_git(
            repo,
            "add",
            "--",
            *convergence_module.R_TO_P019_CHANGED_PATHS,
        )
        _sequential_git(repo, "commit", "-q", "--amend", "--no-edit")
        phase = "P019"

    _secure_phase_files(repo)
    errors = _direct_phase_packet_errors(
        repo,
        phase=phase,
        root_identity_did=root_identity_did,
    )
    for expected in expected_fragments:
        assert any(expected in error for error in errors), errors


def test_protected_acceptance_sequence_is_exactly_contiguous_and_direct(
    tmp_path: Path,
) -> None:
    repo, heads = _sequential_transition_repository(tmp_path)

    assert convergence_module.validate_protected_acceptance_sequence(
        repo_root=repo,
        phase_heads=heads,
    ) == ()
    for phase in convergence_module.SEQUENTIAL_ACCEPTANCE_PHASES[1:]:
        parent_phase = convergence_module.SEQUENTIAL_PHASE_PARENT[phase]
        parent_tree = _sequential_git(
            repo,
            "rev-parse",
            f"{heads[parent_phase]}^{{tree}}",
        )
        assert convergence_module.validate_sequential_acceptance_child_transition(
            repo_root=repo,
            phase=phase,
            child_head=heads[phase],
            parent_head=heads[parent_phase],
            parent_tree=parent_tree,
        ) == ()


def test_sequential_chain_rejects_missing_phase_and_wrong_direct_parent(
    tmp_path: Path,
) -> None:
    repo, heads = _sequential_transition_repository(tmp_path)
    missing = dict(heads)
    del missing["P031"]
    errors = convergence_module.validate_protected_acceptance_sequence(
        repo_root=repo,
        phase_heads=missing,
    )
    assert any("exact contiguous population required" in error for error in errors)

    wrong_parent = heads["P031"]
    wrong_tree = _sequential_git(repo, "rev-parse", f"{wrong_parent}^{{tree}}")
    errors = convergence_module.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase="A032",
        child_head=heads["A032"],
        parent_head=wrong_parent,
        parent_tree=wrong_tree,
    )
    assert any("exact direct single parent required" in error for error in errors)


def test_phase_status_projection_rejects_premature_later_completion(
    tmp_path: Path,
) -> None:
    repo, heads = _sequential_transition_repository(tmp_path)
    board_path = PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
    p031_board = subprocess.run(
        ["git", "show", f"{heads['P031']}:{board_path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    text = p031_board.decode()
    marker = "## ASE3-032"
    before, after = text.split(marker, 1)
    after = after.replace("- Status: todo", "- Status: completed", 1)
    premature = (before + marker + after).encode()

    with pytest.raises(ValueError, match="non-phase status transition"):
        convergence_module._status_only_sequential_phase_board(
            premature,
            "A031",
        )


def test_phase_manifest_rejects_prelaunch_and_reload_effect_claim_confusion(
    tmp_path: Path,
) -> None:
    repo, heads = _sequential_transition_repository(tmp_path)
    manifest_path = convergence_module._CONVERGENCE_MANIFEST_RELATIVE_PATH
    p031_raw = subprocess.run(
        ["git", "show", f"{heads['P031']}:{manifest_path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    p031 = json.loads(p031_raw)
    p031["acceptance"]["runtime_effect_claimed"] = True
    baseline = CurrentMainBaseline.from_dict(
        _load(DEFAULT_ARTIFACT_ROOT / "current_main_baseline.json")
    )
    errors = ConvergenceManifest.from_dict(p031).validate(baseline)
    assert any("runtime_effect_claimed: phase mismatch" in error for error in errors)

    l_raw = subprocess.run(
        ["git", "show", f"{heads['L']}:{manifest_path}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    l_manifest = json.loads(l_raw)
    l_manifest["reload"]["post_launch_birth_receipt_required"] = False
    errors = ConvergenceManifest.from_dict(l_manifest).validate(baseline)
    assert any("post_launch_birth_receipt_required" in error for error in errors)


def test_l_authorization_child_cannot_embed_post_launch_birth_receipt(
    tmp_path: Path,
) -> None:
    repo, heads = _sequential_transition_repository(tmp_path)
    birth_path = (
        convergence_module.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_RELATIVE_PATH
    )
    _sequential_write(repo, birth_path, b'{"post_launch_effect":true}\n')
    _sequential_git(repo, "add", "--", birth_path)
    _sequential_git(repo, "commit", "-q", "--amend", "--no-edit")
    heads["L"] = _sequential_git(repo, "rev-parse", "HEAD")

    errors = convergence_module.validate_protected_acceptance_sequence(
        repo_root=repo,
        phase_heads=heads,
    )

    assert any("post-L birth receipt is forbidden" in error for error in errors)


def test_obsolete_atomic_fan_in_entrypoint_is_fail_closed() -> None:
    errors = validate_acceptance_child_transition(
        repo_root=REPO_ROOT,
        acceptance_head="1" * 40,
        preparation_head="2" * 40,
        preparation_tree="3" * 40,
    )
    assert errors == (
        (
            "operator_acceptance.transition: obsolete atomic fan-in forbidden; "
            "use validate_sequential_acceptance_child_transition"
        ),
    )

    reload_errors, checked = (
        convergence_module._validate_provider_attempt_reload_packet(
            artifact_root=DEFAULT_ARTIFACT_ROOT,
            manifest=object(),
            repo_root=None,
            fallback_authorization=object(),
            fallback_authorization_raw=b"",
            manifest_raw=b"",
            taskboard_raw=b"",
        )
    )
    assert reload_errors == [
        (
            "provider_attempt_reload.packet: obsolete direct A-to-L validation "
            "forbidden; use the sequential L phase packet"
        )
    ]
    assert checked == ()
