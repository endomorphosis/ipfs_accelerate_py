"""ASE3-000 current-main convergence and historical-state isolation tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
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
    FAILED_PRE_DISPATCH_EVENT_019_ATTEMPT_2_FILENAME,
    FAILED_PRE_DISPATCH_LOG_019_ATTEMPT_2_FILENAME,
    FAILED_VALIDATION_EVENT_019_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_006_FILENAME,
    FALSE_COMPLETION_MERGE_RECEIPT_018_FILENAME,
    FALSE_COMPLETION_RECOVERY_FILENAME,
    MANIFEST_FILENAME,
    MAX_EVIDENCE_SNAPSHOT_BYTES,
    MAX_OPERATOR_ACCEPTANCE_RECEIPT_BYTES,
    OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME,
    OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES,
    OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS,
    OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
    OPERATOR_SALVAGE_RECEIPT_019_FILENAME,
    POST_WAVE3_RESIDUAL_FILENAME,
    PROMPT_V3_SCHEDULER_CONFIG_RELATIVE_PATH,
    PROMPT_V3_TASKBOARD_RELATIVE_PATH,
    PROTECTED_RUNTIME_ACTIVATION_RECEIPT_FILENAME,
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
            "manifest_schema": convergence_module.CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS),
            "task_statuses": {
                "ASE3-019": "todo",
                "ASE3-023": "todo",
                "ASE3-027": "todo",
            },
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
            **authority,
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


def _minimal_operator_receipt(task_id: str) -> dict[str, object]:
    expected = convergence_module._ACCEPTANCE_TASK_CONTRACTS[task_id]
    fields = (
        convergence_module._ASE3_019_OPERATOR_SALVAGE_REQUIRED_FIELDS
        if task_id == "ASE3-019"
        else convergence_module._OPERATOR_REPAIR_ACCEPTANCE_REQUIRED_FIELDS
    )
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
    subprocess.run(["git", "add", str(root_pin_path)], cwd=repository, check=True)
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
        convergence_module._status_only_acceptance_board(board_path.read_bytes())
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


def test_reload_gate_completion_requires_future_receipt_authority(
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
        "ASE3-022.status: completion requires a strict reload receipt validator "
        "and convergence-manifest binding" in error
        for error in report.errors
    )


def test_reload_receipt_path_is_reserved_until_strictly_validated(
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
        "receipt: present without a strict validator and convergence-manifest binding"
        in error
        for error in report.errors
    )


@pytest.mark.parametrize("receipt_kind", ("regular", "dangling-symlink"))
def test_operator_salvage_receipt_path_is_reserved_during_c1(
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
        and "present without a strict validator" in error
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
        *OPERATOR_ACCEPTANCE_RECEIPT_RELATIVE_PATHS,
        PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH,
    }
    assert expected <= set(protected_paths)
    assert all(not (REPO_ROOT / relative_path).exists() for relative_path in expected)


@pytest.mark.parametrize(
    "filename",
    OPERATOR_ACCEPTANCE_RECEIPT_FILENAMES,
)
@pytest.mark.parametrize("receipt_kind", ("regular", "dangling-symlink"))
def test_every_premature_operator_acceptance_receipt_fails_preparation(
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
        filename in error and "present without a strict validator" in error
        for error in report.errors
    )
    assert any("partial population forbidden" in error for error in report.errors)


@pytest.mark.parametrize("task_id", ("ASE3-019", "ASE3-023", "ASE3-027"))
def test_preparation_rejects_each_premature_completed_status(
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
    after_raw = convergence_module._status_only_acceptance_board(
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
            **authority,
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


def test_ase3_027_receipt_reconstructs_exact_source_and_integrated_provenance(
) -> None:
    payload, _, authority = _operator_repair_receipt_027()

    assert validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-027",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    ) == ()

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
    recovery.pop("ambient_override")

    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    generations = implementation["generations"]
    assert isinstance(generations, list)
    first_generation = generations[0]
    assert isinstance(first_generation, dict)
    first_generation["integrated_tree"] = "0" * 40
    errors = validate_operator_repair_acceptance_receipt(
        payload,
        task_id="ASE3-027",
        repo_root=REPO_ROOT,
        lifecycle_authority=authority,
    )
    assert any("integrated_tree" in error for error in errors)


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


def test_acceptance_manifest_at_2_binds_exact_receipts_tasks_and_parent(
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


def test_acceptance_packet_rejects_a_manifest_receipt_digest_mismatch(
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


def test_acceptance_transition_is_one_direct_child_with_exact_five_paths(
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


def test_transition_authority_snapshots_are_mode_safe_and_git_exact(
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
def test_transition_authority_rejects_group_or_other_write_permission(
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
def test_transition_authority_rejects_wrong_owner(
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
def test_transition_authority_rejects_final_symlink(
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
def test_transition_authority_rejects_relocated_lexical_path(
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


def test_transition_authority_rejects_symlinked_repository_parent(
    tmp_path: Path,
) -> None:
    repository, _, _ = _initialize_transition_repository(tmp_path)
    alias = tmp_path / "repository-alias"
    alias.symlink_to(repository, target_is_directory=True)

    errors = _validate_transition_repository(alias)

    assert any("path contains a symlink or non-directory" in error for error in errors)


@pytest.mark.parametrize("relative_path", _TRANSITION_AUTHORITY_RELATIVE_PATHS)
def test_transition_authority_rejects_hardlink(
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
def test_transition_authority_rejects_path_swap_during_read(
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
def test_transition_authority_rejects_descriptor_growth(
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


def test_acceptance_transition_rejects_dirty_working_manifest_snapshot(
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
def test_acceptance_transition_rejects_dirty_lifecycle_bytes(
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
def test_acceptance_transition_rejects_semantically_invalid_preparation_manifest(
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


def test_acceptance_transition_rejects_dirty_consumed_receipt(tmp_path: Path) -> None:
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


def test_acceptance_transition_rejects_unauthorized_manifest_drift(
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


def test_acceptance_transition_rejects_extra_path_and_board_prose(
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
    assert any("direct single-parent child" in error for error in errors)
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
    activation = config["protected_runtime_activation"]
    refill = config["refill_policy"]
    monitor = config["monitor_policy"]
    assert isinstance(initial, dict)
    assert isinstance(groups, dict)
    assert isinstance(dependencies, dict)
    assert isinstance(activation, dict)
    assert isinstance(refill, dict)
    assert isinstance(monitor, dict)

    canonical = initial["canonical_task_ids"]
    assert initial["task_count"] == 25
    assert isinstance(canonical, list)
    assert len(canonical) == len(set(canonical)) == 25
    assert initial["noncanonical_transition_task_ids"] == ["ASE3-022"]
    assert set(dependencies) == set(canonical)
    assert {
        task_id
        for task_ids in groups.values()
        for task_id in task_ids
    } == set(canonical)
    assert activation == {
        "task_id": "ASE3-026",
        "status": "blocked",
        "receipt_path": (
            "data/agent_supervisor/prompt_only_self_improvement_v3/"
            "convergence/protected_runtime_activation_receipt.json"
        ),
        "operator_review_required": True,
        "strict_validator_and_manifest_binding_required": True,
    }
    assert config["strict_task_sharding"] is True
    assert config["objective_refill_enabled"] is False
    assert config["codebase_refill_enabled"] is False
    assert refill["enable_after_task"] == "ASE3-026"
    assert refill["prompt_program_refill_enabled"] is False
    assert monitor["enabled"] is False
    assert monitor["detached"] is True
    assert monitor["activation_task_id"] == "ASE3-026"


@pytest.mark.parametrize("task_id", ("ASE3-024", "ASE3-025", "ASE3-028"))
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
                "all provider selection, authorization, freshness, failure "
                "classification, and fallback allow/deny decisions remain solely in "
                "`ipfs_accelerate_py.llm_router`"
            ),
            (
                "provider selection and fallback decisions may be delegated to lower "
                "layers"
            ),
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


def test_protected_runtime_activation_stays_blocked_without_strict_receipt(
    tmp_path: Path,
) -> None:
    taskboard = tmp_path / "prompt-v3.todo.md"
    text = TASKBOARD_PATH.read_text(encoding="utf-8")
    needle = (
        "## ASE3-026 Activate and reload the durable refill and autonomous "
        "monitor runtime\n\n- Status: blocked\n"
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


def test_unvalidated_protected_runtime_activation_receipt_is_reserved(
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
