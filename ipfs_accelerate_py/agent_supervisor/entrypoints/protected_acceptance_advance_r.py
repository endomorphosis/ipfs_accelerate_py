"""Advance the protected chain Q→R by publishing the lifecycle root pin."""

from __future__ import annotations

import os
import secrets
import subprocess
import time
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
    SignedArtifactRequest,
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
from .local_profile import lifecycle_root_identity_did
from .protected_acceptance_q_readiness import Q_INVENTORY_RELATIVE_PATH
from .protected_acceptance_transition import build_prompt_v3_root_pin

_ROOT_PIN_PATH = convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
_BOARD_NAMESPACE = convergence.BOARD_NAMESPACE


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


def _fresh_r_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.R,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.R,
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


def advance_prompt_v3_r(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected R (root-pin) candidate."""

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

    inventory_path = repo / Q_INVENTORY_RELATIVE_PATH
    if not inventory_path.is_file():
        raise ProtectedAcceptanceDenied("Q inventory is required before R")
    root_pin_path = repo / _ROOT_PIN_PATH
    if root_pin_path.exists() or root_pin_path.is_symlink():
        raise ProtectedAcceptanceDenied("lifecycle root pin is already present")

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    parent_tree = (
        _git(repo, "rev-parse", f"{parent_commit}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    authority = _fresh_r_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    pin_request = SignedArtifactRequest(
        phase=PromptV3Phase.R,
        authority=authority,
        body={
            "board_namespace": _BOARD_NAMESPACE,
            "base_head": parent_commit,
            "base_tree": parent_tree,
            "root_identity_did": root_did,
            "pinned_at_ms": int(time.time() * 1000),
        },
    )
    root_pin_bytes = build_prompt_v3_root_pin(pin_request)
    # Validate against convergence contract with the known root DID and Q head.
    pin_payload = __import__("json").loads(root_pin_bytes.decode("utf-8"))
    errors = convergence.validate_local_profile_lifecycle_root_pin(
        pin_payload,
        expected_root_identity_did=root_did,
        expected_base_head=parent_commit,
        expected_base_tree=parent_tree,
    )
    if errors:
        raise ProtectedAcceptanceDenied(
            "root pin failed validation: " + "; ".join(errors)
        )

    policy = PhasePolicy(
        phase=PromptV3Phase.R,
        expected_parent_phase=PromptV3Phase.Q,
        allowed_paths=(_ROOT_PIN_PATH,),
        required_evidence_kinds=("root-pin-suite",),
        validator_ids=("local-profile-lifecycle-root-pin",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.Q,
        authority=authority,
        artifacts=(ArtifactBytes(path=_ROOT_PIN_PATH, data=root_pin_bytes),),
        evidence_handles=(_placeholder_handle("root-pin-suite"),),
        commit_message="ASE3-R: pin local-profile lifecycle root after Q",
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
        runner=lambda observed: (_bound_handle(observed, "root-pin-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "local-profile-lifecycle-root-pin"),
        ),
        evidence_loader=_evidence_loader,
    )
    result: dict[str, Any] = {
        "phase": "R",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "root_pin_path": _ROOT_PIN_PATH,
        "root_pin_content_id": content_id(root_pin_bytes),
        "root_identity_did": root_did,
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


__all__ = ("advance_prompt_v3_r",)
