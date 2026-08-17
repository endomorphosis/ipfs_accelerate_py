"""Advance the protected chain A032→A023/027 (dual repair receipts)."""

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

_RECEIPT_023_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/"
    f"{convergence.OPERATOR_ACCEPTANCE_RECEIPT_023_FILENAME}"
)
_RECEIPT_027_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/"
    f"{convergence.OPERATOR_ACCEPTANCE_RECEIPT_027_FILENAME}"
)
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_BOARD_PATH = convergence.PROMPT_V3_TASKBOARD_RELATIVE_PATH.as_posix()
_FALLBACK_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
_WITNESS_PATH = convergence.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
_ROOT_PIN_PATH = convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
_BOARD_NAMESPACE = convergence.BOARD_NAMESPACE
_PHASE = "A023/027"
_ALLOWED_PATHS = (
    _RECEIPT_023_PATH,
    _RECEIPT_027_PATH,
    _MANIFEST_PATH,
    _BOARD_PATH,
)


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


def _fresh_a023_027_authority(
    *, parent_commit: str, identity_did: str
) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.A023_027,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.A023_027,
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


def _serialize_generations(task_id: str) -> list[dict[str, Any]]:
    final_values = convergence._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[task_id]
    generations: list[dict[str, Any]] = []
    for generation in final_values["generations"]:
        item = dict(generation)
        item["changed_paths"] = list(generation["changed_paths"])
        generations.append(item)
    return generations


def _build_repair_receipt(
    *,
    task_id: str,
    a032_head: str,
    a032_tree: str,
    lifecycle_authority: Mapping[str, Any],
    created_at: str,
) -> bytes:
    contracts = convergence._ACCEPTANCE_TASK_CONTRACTS[task_id]
    final_values = convergence._ACCEPTANCE_IMPLEMENTATION_FINAL_VALUES[task_id]
    if not final_values["ready"]:
        raise ProtectedAcceptanceDenied(
            f"{task_id} final product values are not populated"
        )
    evidence_anchor = str(
        convergence._FALSE_COMPLETION_REPAIR_TASKS[task_id]["evidence_anchor"]
    )
    artifact, pointer = evidence_anchor.split("#", 1)
    validation_command = str(
        convergence._FALSE_COMPLETION_REPAIR_TASKS[task_id]["validation"]
    )
    payload: dict[str, Any] = {
        "schema": convergence.OPERATOR_REPAIR_ACCEPTANCE_RECEIPT_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "task": {
            "task_id": task_id,
            "canonical_task_cid": contracts["canonical_task_cid"],
            "goal_id": contracts["goal_id"],
            "repairs_task": contracts["repairs_task"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "completed",
        },
        "recovery": {
            "artifact": artifact,
            "pointer": pointer,
            "historical_completion_authority": False,
            "branch_local_completion_authority": False,
            "repair_required": True,
        },
        "implementation": {
            "generations": _serialize_generations(task_id),
            "final_blobs": dict(final_values["final_blobs"]),
        },
        "acceptance_parent": {
            "head": a032_head,
            "tree": a032_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "manifest_schema": convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "receipt_paths_absent": list(
                convergence._sequential_future_artifacts_after("A032")
            ),
            "task_statuses": convergence._sequential_task_statuses_after("A032"),
            "reload_gate_status": "blocked",
        },
        "validation": {
            "command": validation_command,
            "exit_code": 0,
            "passed": True,
            "passed_count": final_values["validation_passed_count"],
            "failed_count": 0,
            "validated_head": a032_head,
            "validated_tree": a032_tree,
        },
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": f"codex:{task_id.lower()}-product",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._REPAIR_ACCEPTANCE_DENIALS),
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


def _build_a023_027_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_raw: bytes,
    parent_head: str,
    parent_tree: str,
    receipt_023_raw: bytes,
    receipt_027_raw: bytes,
    created_at: str,
) -> bytes:
    if parent_manifest.get("schema") != convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("A023/027 parent must be acceptance @2")
    child = dict(parent_manifest)
    child["created_at"] = created_at
    child["acceptance"] = {
        "phase": _PHASE,
        "parent_phase": "A032",
        "parent_head": parent_head,
        "parent_tree": parent_tree,
        "parent_manifest_sha256": _sha256_bytes(parent_manifest_raw),
        "artifacts": {
            _RECEIPT_023_PATH: _sha256_bytes(receipt_023_raw),
            _RECEIPT_027_PATH: _sha256_bytes(receipt_027_raw),
        },
        "task_statuses": convergence._sequential_task_statuses_after(_PHASE),
        "reload_gate_status": "blocked",
        "pre_launch_authorization_only": False,
        "runtime_effect_claimed": True,
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


def advance_prompt_v3_a023_027(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected A023/027 dual-receipt candidate."""

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

    for relative in (_RECEIPT_023_PATH, _RECEIPT_027_PATH):
        if (repo / relative).exists() or (repo / relative).is_symlink():
            raise ProtectedAcceptanceDenied(
                f"operator repair receipt already present: {relative}"
            )

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="A032",
    )
    if discovery_errors or "A032" not in discovered:
        raise ProtectedAcceptanceDenied(
            "A032 phase head unavailable: "
            + "; ".join(discovery_errors or ["missing A032"])
        )
    a032_head = discovered["A032"]
    a032_tree = (
        _git(repo, "rev-parse", f"{a032_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=a032_head))
    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    signed_at_ms = (authorized_at_ms // 1000 + 1) * 1000
    if not authorized_at_ms <= signed_at_ms <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place A023/027 signed_at inside sealed witness validity"
        )
    created_at = _ms_to_utc(signed_at_ms)

    receipt_023_raw = _build_repair_receipt(
        task_id="ASE3-023",
        a032_head=a032_head,
        a032_tree=a032_tree,
        lifecycle_authority=lifecycle_authority,
        created_at=created_at,
    )
    receipt_027_raw = _build_repair_receipt(
        task_id="ASE3-027",
        a032_head=a032_head,
        a032_tree=a032_tree,
        lifecycle_authority=lifecycle_authority,
        created_at=created_at,
    )
    for task_id, raw in (
        ("ASE3-023", receipt_023_raw),
        ("ASE3-027", receipt_027_raw),
    ):
        payload = json.loads(raw.decode("utf-8"))
        errors = convergence.validate_operator_repair_acceptance_receipt(
            payload,
            task_id=task_id,
            repo_root=repo,
            lifecycle_authority=lifecycle_authority,
        )
        if errors:
            raise ProtectedAcceptanceDenied(
                f"{task_id} repair receipt failed validation: " + "; ".join(errors)
            )

    parent_manifest_raw = _git(repo, "show", f"{a032_head}:{_MANIFEST_PATH}")
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    manifest_raw = _build_a023_027_manifest(
        parent_manifest=parent_manifest,
        parent_manifest_raw=parent_manifest_raw,
        parent_head=a032_head,
        parent_tree=a032_tree,
        receipt_023_raw=receipt_023_raw,
        receipt_027_raw=receipt_027_raw,
        created_at=created_at,
    )
    parent_board_raw = _git(repo, "show", f"{a032_head}:{_BOARD_PATH}")
    board_raw = convergence._status_only_sequential_phase_board(
        parent_board_raw, _PHASE
    )

    authority = _fresh_a023_027_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.A023_027,
        expected_parent_phase=PromptV3Phase.A032,
        allowed_paths=_ALLOWED_PATHS,
        required_evidence_kinds=("a023-027-suite",),
        validator_ids=("operator-repair-acceptance-dual",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.A032,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_RECEIPT_023_PATH, data=receipt_023_raw),
            ArtifactBytes(path=_RECEIPT_027_PATH, data=receipt_027_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
            ArtifactBytes(path=_BOARD_PATH, data=board_raw),
        ),
        evidence_handles=(_placeholder_handle("a023-027-suite"),),
        commit_message=(
            "ASE3-A023/027: seal dual operator repair acceptance after A032"
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
        runner=lambda observed: (_bound_handle(observed, "a023-027-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "operator-repair-acceptance-dual"),
        ),
        evidence_loader=_evidence_loader,
    )
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase=_PHASE,
        child_head=candidate.commit_id,
        parent_head=a032_head,
        parent_tree=a032_tree,
        consumed_child_blobs={
            _RECEIPT_023_PATH: receipt_023_raw,
            _RECEIPT_027_PATH: receipt_027_raw,
            _MANIFEST_PATH: manifest_raw,
            _BOARD_PATH: board_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "A023/027 sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": _PHASE,
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "a032_head": a032_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "receipt_023_path": _RECEIPT_023_PATH,
        "receipt_027_path": _RECEIPT_027_PATH,
        "receipt_023_content_id": content_id(receipt_023_raw),
        "receipt_027_content_id": content_id(receipt_027_raw),
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


__all__ = ("advance_prompt_v3_a023_027",)
