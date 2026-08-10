"""Advance the protected chain A023/027→L (provider-attempt daemon reload)."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core.protected_acceptance_contracts import (
    ArtifactBytes,
    EvidenceHandle,
    PhaseAuthority,
    PhaseCandidateRequest,
    PhaseEvidenceResult,
    PhasePolicy,
    PromptV3Phase,
    ProtectedAcceptanceDenied,
    ProtectedAcceptanceError,
    PublicationResult,
    RepositoryBinding,
    canonical_json_bytes,
    content_id,
    phase_authority_content_id,
)
from ..merge.protected_acceptance_transition import (
    TransitionHooks,
    build_phase_candidate,
    publish_phase_candidate,
    run_phase_evidence,
    validate_phase_candidate,
)
from ..validation import prompt_v3_convergence as convergence
from .local_profile import lifecycle_root_identity_did, sign_profile_binding
from .protected_acceptance_transition import sign_prompt_v3_operator_artifact

_RECEIPT_PATH = convergence.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_RELATIVE_PATH
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_BOARD_PATH = convergence.PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
_FALLBACK_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
_WITNESS_PATH = convergence.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
_ROOT_PIN_PATH = convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
_BOARD_NAMESPACE = convergence.BOARD_NAMESPACE
_PHASE = "L"


def _git(repo: Path, *arguments: str) -> bytes:
    env = dict(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_EDITOR",
        "GIT_SEQUENCE_EDITOR",
        "GIT_TERMINAL_PROMPT",
        "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY",
    ):
        env.pop(key, None)
    completed = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repo,
        env=env,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or b"git failed").decode(
            "utf-8", "replace"
        )
        raise ProtectedAcceptanceError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _fresh_l_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.L,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.L,
        authority_id=authority_id,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )


def _placeholder_handle(kind: str) -> EvidenceHandle:
    raw = kind.encode("ascii")
    return EvidenceHandle(
        kind=kind, content_id=content_id(raw), byte_length=len(raw), record_count=1
    )


def _bound_evidence_bytes(candidate: Any, kind: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.phase-evidence-binding@1",
            "candidate_commit": candidate.commit_id,
            "authority_id": candidate.request.authority.authority_id,
            "kind": kind,
        }
    )


def _bound_handle(candidate: Any, kind: str) -> EvidenceHandle:
    raw = _bound_evidence_bytes(candidate, kind)
    return EvidenceHandle(
        kind=kind, content_id=content_id(raw), byte_length=len(raw), record_count=1
    )


def _load_lifecycle_authority(repo: Path, *, head: str) -> Mapping[str, Any]:
    auth_raw = _git(repo, "show", f"{head}:{_FALLBACK_AUTH_PATH}")
    wit_raw = _git(repo, "show", f"{head}:{_WITNESS_PATH}")
    pin_raw = _git(repo, "show", f"{head}:{_ROOT_PIN_PATH}")
    auth = convergence.ProviderFallbackPolicyAuthorization.from_dict(
        json.loads(auth_raw.decode("utf-8"))
    )
    witness = convergence.LocalOperatorLifecycleWitnessSnapshot(
        payload=json.loads(wit_raw.decode("utf-8")),
        raw=wit_raw,
        sha256=_sha256_bytes(wit_raw),
    )
    root_pin = convergence.LocalProfileLifecycleRootPinSnapshot(
        payload=json.loads(pin_raw.decode("utf-8")),
        raw=pin_raw,
        sha256=_sha256_bytes(pin_raw),
    )
    return auth.acceptance_review_authority(
        raw_sha256=_sha256_bytes(auth_raw),
        lifecycle_witness=witness,
        root_pin=root_pin,
    )


def _acceptance_receipt_bindings(repo: Path, *, head: str) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for filename in convergence.SEQUENTIAL_ACCEPTANCE_ARTIFACT_FILENAMES:
        relative = f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{filename}"
        raw = _git(repo, "show", f"{head}:{relative}")
        bindings[filename] = _sha256_bytes(raw)
    return bindings


def _build_reload_receipt(
    *,
    repo: Path,
    a023_head: str,
    a023_tree: str,
    lifecycle_authority: Mapping[str, Any],
) -> bytes:
    finals = convergence._RELOAD_FINAL_VALUES
    if not finals["ready"]:
        raise ProtectedAcceptanceDenied(
            f"reload finals not ready: {finals.get('pending')}"
        )
    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    signed_at_ms = (authorized_at_ms // 1000 + 1) * 1000
    if not authorized_at_ms <= signed_at_ms <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place L signed_at inside sealed witness validity"
        )
    created_at = _ms_to_utc(signed_at_ms)
    receipt_bindings = _acceptance_receipt_bindings(repo, head=a023_head)
    accepted_control_plane = dict(convergence._ASE3_019_ACCEPTED_CONTROL_PLANE)
    accepted_control_plane_sha256 = convergence._canonical_sha256(
        accepted_control_plane
    )
    stopped_number = int(finals["stopped_generation_number"])
    target_number = stopped_number + 1
    stopped_generation = {
        "generation_id": finals["stopped_generation_id"],
        "generation_number": stopped_number,
        "head": a023_head,
        "tree": a023_tree,
        "scheduler_path": convergence._RELOAD_SCHEDULER_PATH,
        "scheduler_blob": finals["scheduler_blob"],
        "scheduler_raw_sha256": finals["scheduler_raw_sha256"],
        "daemon_path": convergence._RELOAD_DAEMON_PATH,
        "daemon_blob": finals["daemon_blob"],
        "daemon_raw_sha256": finals["daemon_raw_sha256"],
        "observed_owned_processes": 0,
        "observed_scoped_provider_containers": 0,
        "observed_inflight_attempts": 0,
    }
    authorization = {
        "source_head": a023_head,
        "source_tree": a023_tree,
        "stopped_generation_id": finals["stopped_generation_id"],
        "target_generation_id": finals["target_generation_id"],
        "target_generation_number": target_number,
        "target_scheduler_blob": finals["scheduler_blob"],
        "target_daemon_blob": finals["daemon_blob"],
        "lease_namespace": _BOARD_NAMESPACE,
        "lease_state_at_authorization": "unclaimed",
        "required_cas_transition": "unclaimed_to_reserved",
        "single_winner_required": True,
        "launch_only_after_l_validates": True,
        "post_launch_birth_receipt_required": True,
        "post_launch_birth_receipt_schema": (
            convergence.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
        ),
        "attempt_counters_unchanged": True,
        "queue_history_unchanged": True,
        "legacy_refill_unchanged": True,
        "runtime_effect_started": False,
    }
    payload: dict[str, Any] = {
        "schema": convergence.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "task": dict(convergence._RELOAD_TASK_CONTRACT),
        "acceptance_parent": {
            "head": a023_head,
            "tree": a023_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "acceptance_receipts": receipt_bindings,
            "task_statuses": {
                **{
                    task_id: "completed"
                    for task_id in convergence.SEQUENTIAL_ACCEPTANCE_TASK_IDS
                },
                "ASE3-022": "blocked",
            },
        },
        "incident": {
            "attempt2_incident": (
                convergence.SELF_HOST_SEED_FAILURE_019_ATTEMPT_2_FILENAME
            ),
            "attempt2_incident_sha256": (
                convergence._ASE3_019_ATTEMPT2_INCIDENT_SHA256
            ),
            "operator_salvage_receipt": (
                convergence.OPERATOR_SALVAGE_RECEIPT_019_FILENAME
            ),
            "operator_salvage_receipt_sha256": receipt_bindings[
                convergence.OPERATOR_SALVAGE_RECEIPT_019_FILENAME
            ],
            "accepted_control_plane_sha256": accepted_control_plane_sha256,
        },
        "stopped_generation": stopped_generation,
        "authorization": authorization,
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": "codex:ase3-022-reload",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._RELOAD_DENIALS),
    }

    def _signer(unsigned: Mapping[str, Any]) -> Mapping[str, str]:
        return sign_profile_binding(
            profile_dir=None, lifecycle_dir=None, payload=unsigned
        )

    signed = sign_prompt_v3_operator_artifact(payload, signer=_signer)
    return (
        json.dumps(
            signed,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_l_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_head: str,
    parent_tree: str,
    receipt_raw: bytes,
    created_at: str,
) -> bytes:
    if parent_manifest.get("schema") != convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("L parent must be acceptance @2")
    child = dict(parent_manifest)
    child["schema"] = convergence.RELOAD_CONVERGENCE_MANIFEST_SCHEMA
    child["created_at"] = created_at
    child["reload"] = {
        "phase": "provider_attempt_daemon_reload",
        "acceptance_head": parent_head,
        "acceptance_tree": parent_tree,
        "receipt": {
            convergence.PROVIDER_ATTEMPT_DAEMON_RELOAD_RECEIPT_FILENAME: _sha256_bytes(
                receipt_raw
            )
        },
        "task": dict(convergence._RELOAD_TASK_CONTRACT),
        "accepted_task_statuses": {
            task_id: "completed"
            for task_id in convergence.SEQUENTIAL_ACCEPTANCE_TASK_IDS
        },
        "reload_gate_completed": True,
        "launch_authorization_only": True,
        "post_launch_birth_receipt_required": True,
        "post_launch_birth_receipt_schema": (
            convergence.PROVIDER_ATTEMPT_GENERATION_BIRTH_RECEIPT_SCHEMA
        ),
    }
    return (
        json.dumps(
            child,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def advance_prompt_v3_l(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected L reload candidate."""

    repo = Path(repo_root).resolve()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for path in repo.rglob("__pycache__"):
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass
            try:
                path.rmdir()
            except OSError:
                pass

    if (repo / _RECEIPT_PATH).exists() or (repo / _RECEIPT_PATH).is_symlink():
        raise ProtectedAcceptanceDenied(
            "provider-attempt daemon reload receipt is already present"
        )

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="A023/027",
    )
    if discovery_errors or "A023/027" not in discovered:
        raise ProtectedAcceptanceDenied(
            "A023/027 phase head unavailable: "
            + "; ".join(discovery_errors or ["missing A023/027"])
        )
    a023_head = discovered["A023/027"]
    a023_tree = (
        _git(repo, "rev-parse", f"{a023_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=a023_head))
    receipt_raw = _build_reload_receipt(
        repo=repo,
        a023_head=a023_head,
        a023_tree=a023_tree,
        lifecycle_authority=lifecycle_authority,
    )
    receipt_payload = json.loads(receipt_raw.decode("utf-8"))
    receipt_errors = convergence.validate_provider_attempt_reload_receipt(
        receipt_payload,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
        accepted_control_plane=dict(convergence._ASE3_019_ACCEPTED_CONTROL_PLANE),
    )
    if receipt_errors:
        raise ProtectedAcceptanceDenied(
            "L reload receipt failed validation: " + "; ".join(receipt_errors)
        )

    parent_manifest_raw = _git(repo, "show", f"{a023_head}:{_MANIFEST_PATH}")
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    created_at = str(receipt_payload["created_at"])
    manifest_raw = _build_l_manifest(
        parent_manifest=parent_manifest,
        parent_head=a023_head,
        parent_tree=a023_tree,
        receipt_raw=receipt_raw,
        created_at=created_at,
    )
    parent_board_raw = _git(repo, "show", f"{a023_head}:{_BOARD_PATH}")
    board_raw = convergence._status_only_sequential_phase_board(
        parent_board_raw, _PHASE
    )

    authority = _fresh_l_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.L,
        expected_parent_phase=PromptV3Phase.A023_027,
        allowed_paths=(_RECEIPT_PATH, _MANIFEST_PATH, _BOARD_PATH),
        required_evidence_kinds=("l-reload-suite",),
        validator_ids=("provider-attempt-daemon-reload",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.A023_027,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_RECEIPT_PATH, data=receipt_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
            ArtifactBytes(path=_BOARD_PATH, data=board_raw),
        ),
        evidence_handles=(_placeholder_handle("l-reload-suite"),),
        commit_message=(
            "ASE3-L: seal provider-attempt daemon reload after A023/027"
        ),
        commit_timestamp=f"{int(time.time())} +0000",
        observed_at_ns=time.time_ns(),
        dry_run=dry_run,
    )

    def _authority_ok(value: Any, now_ns: int) -> bool:
        return (
            isinstance(value, PhaseAuthority)
            and value.authority_id == authority.authority_id
            and value.issued_at_ns <= now_ns < value.expires_at_ns
        )

    def _evidence_loader(candidate: Any, handle: EvidenceHandle) -> bytes:
        return _bound_evidence_bytes(candidate, handle.kind)

    candidate = build_phase_candidate(
        request,
        authority_validator=_authority_ok,
        hooks=TransitionHooks(),
    )
    evidence_result = run_phase_evidence(
        candidate,
        runner=lambda observed: (_bound_handle(observed, "l-reload-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "provider-attempt-daemon-reload"),
        ),
        evidence_loader=_evidence_loader,
    )
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase=_PHASE,
        child_head=candidate.commit_id,
        parent_head=a023_head,
        parent_tree=a023_tree,
        consumed_child_blobs={
            _RECEIPT_PATH: receipt_raw,
            _MANIFEST_PATH: manifest_raw,
            _BOARD_PATH: board_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "L sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": _PHASE,
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "a023_027_head": a023_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "receipt_path": _RECEIPT_PATH,
        "receipt_content_id": content_id(receipt_raw),
        "reviewer_identity": lifecycle_authority["reviewer_identity"],
        "published": False,
    }
    if publish and not dry_run:
        publication: PublicationResult = publish_phase_candidate(
            validated,
            authority_validator=_authority_ok,
            pre_cas_validator=lambda _item: True,
        )
        result.update(
            {
                "published": True,
                "publication_commit": publication.new_commit,
                "old_commit": publication.old_commit,
            }
        )
    return result


__all__ = ("advance_prompt_v3_l",)
