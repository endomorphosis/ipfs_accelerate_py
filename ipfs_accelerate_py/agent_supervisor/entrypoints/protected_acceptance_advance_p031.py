"""Advance the protected chain A030→P031 (native launch authorization)."""

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

_AUTH_PATH = convergence.NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_RELATIVE_PATH
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_FALLBACK_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
_WITNESS_PATH = convergence.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
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


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _fresh_p031_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.P031,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.P031,
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


def _prior_artifact_digests(repo: Path, *, head: str, phase: str) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative_path in convergence._sequential_artifacts_after(phase):
        raw = _git(repo, "show", f"{head}:{relative_path}")
        digests[relative_path] = _sha256_bytes(raw)
    return digests


def _build_launch_authorization(
    *,
    repo: Path,
    a030_head: str,
    a030_tree: str,
    lifecycle_authority: Mapping[str, Any],
) -> bytes:
    contracts = convergence._SEQUENTIAL_TASK_CONTRACTS["ASE3-031"]
    authorized_at_ms = int(lifecycle_authority["fallback_authorized_at_ms"])
    expires_at_ms = int(lifecycle_authority["lifecycle_witness_expires_at_ms"])
    signed_at_ms = (authorized_at_ms // 1000 + 1) * 1000
    if not authorized_at_ms <= signed_at_ms <= expires_at_ms:
        raise ProtectedAcceptanceDenied(
            "cannot place P031 signed_at inside sealed witness validity"
        )
    created_at = _ms_to_utc(signed_at_ms)
    product = dict(convergence._ASE3_031_PRODUCT_IDENTITY)
    product["changed_paths"] = list(product["changed_paths"])
    product["file_raw_sha256"] = dict(product["file_raw_sha256"])

    payload: dict[str, Any] = {
        "schema": convergence.NATIVE_DEPENDENCY_LAUNCH_AUTHORIZATION_SCHEMA,
        "created_at": created_at,
        "board_namespace": _BOARD_NAMESPACE,
        "phase": "P031",
        "task": {
            "task_id": "ASE3-031",
            "canonical_task_cid": contracts["canonical_task_cid"],
            "todo_contract_sha256": contracts["todo_contract_sha256"],
            "completed_contract_sha256": contracts["completed_contract_sha256"],
            "status_before": "todo",
            "status_after": "todo",
        },
        "acceptance_parent": {
            "head": a030_head,
            "tree": a030_tree,
            "branch": "agent/prompt-self-improvement-v3",
            "phase": "A030",
            "manifest_schema": convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA,
            "prior_artifacts": _prior_artifact_digests(
                repo, head=a030_head, phase="A030"
            ),
            "future_artifact_paths_absent": list(
                convergence._sequential_future_artifacts_after("A030")
            ),
            "task_statuses": convergence._sequential_task_statuses_after("A030"),
            "reload_gate_status": "blocked",
        },
        "authorization_id": "",
        "product": product,
        "native_pin": dict(convergence._ASE3_031_REVIEWED_DEPENDENCY_PIN),
        "host_abi_trust_boundary": dict(convergence._ASE3_031_HOST_ABI_TRUST_BOUNDARY),
        "claims": dict(convergence._NATIVE_DEPENDENCY_AUTHORIZATION_CLAIMS),
        "review": {
            **{
                field: lifecycle_authority[field]
                for field in convergence._ACCEPTANCE_REVIEW_AUTHORITY_FIELDS
            },
            "implementer_identity": "codex:ase3-031-authorization",
            "implementer_provider": "codex",
            "algorithm": "Ed25519",
            "signed_at": created_at,
            "signature": "",
        },
        "denials": dict(convergence._NATIVE_DEPENDENCY_AUTHORIZATION_DENIALS),
    }
    # provisional id before signature (signature excluded from id)
    payload["authorization_id"] = (
        convergence.native_dependency_launch_authorization_id(payload)
    )

    def _signer(unsigned: Mapping[str, Any]) -> Mapping[str, str]:
        return sign_profile_binding(
            profile_dir=None, lifecycle_dir=None, payload=unsigned
        )

    signed = sign_prompt_v3_operator_artifact(payload, signer=_signer)
    # Recompute id after signature is attached (id ignores signature)
    signed["authorization_id"] = (
        convergence.native_dependency_launch_authorization_id(signed)
    )
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


def _build_p031_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_raw: bytes,
    parent_head: str,
    parent_tree: str,
    authorization_raw: bytes,
    created_at: str,
) -> bytes:
    if parent_manifest.get("schema") != convergence.ACCEPTANCE_CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("P031 parent must be acceptance @2")
    child = dict(parent_manifest)
    child["created_at"] = created_at
    child["acceptance"] = {
        "phase": "P031",
        "parent_phase": "A030",
        "parent_head": parent_head,
        "parent_tree": parent_tree,
        "parent_manifest_sha256": _sha256_bytes(parent_manifest_raw),
        "artifacts": {_AUTH_PATH: _sha256_bytes(authorization_raw)},
        "task_statuses": convergence._sequential_task_statuses_after("P031"),
        "reload_gate_status": "blocked",
        "pre_launch_authorization_only": True,
        "runtime_effect_claimed": False,
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


def advance_prompt_v3_p031(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected P031 candidate."""

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

    if (repo / _AUTH_PATH).exists() or (repo / _AUTH_PATH).is_symlink():
        raise ProtectedAcceptanceDenied(
            "native dependency launch authorization is already present"
        )

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="A030",
    )
    if discovery_errors or "A030" not in discovered:
        raise ProtectedAcceptanceDenied(
            "A030 phase head unavailable: "
            + "; ".join(discovery_errors or ["missing A030"])
        )
    a030_head = discovered["A030"]
    a030_tree = (
        _git(repo, "rev-parse", f"{a030_head}^{{tree}}")
        .decode("ascii", "strict")
        .strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    lifecycle_authority = dict(_load_lifecycle_authority(repo, head=a030_head))
    authorization_raw = _build_launch_authorization(
        repo=repo,
        a030_head=a030_head,
        a030_tree=a030_tree,
        lifecycle_authority=lifecycle_authority,
    )
    authorization_payload = json.loads(authorization_raw.decode("utf-8"))
    auth_errors = convergence.validate_native_dependency_launch_authorization(
        authorization_payload,
        repo_root=repo,
        lifecycle_authority=lifecycle_authority,
    )
    if auth_errors:
        raise ProtectedAcceptanceDenied(
            "P031 launch authorization failed validation: " + "; ".join(auth_errors)
        )

    parent_manifest_raw = _git(repo, "show", f"{a030_head}:{_MANIFEST_PATH}")
    parent_manifest = json.loads(parent_manifest_raw.decode("utf-8"))
    created_at = str(authorization_payload["created_at"])
    manifest_raw = _build_p031_manifest(
        parent_manifest=parent_manifest,
        parent_manifest_raw=parent_manifest_raw,
        parent_head=a030_head,
        parent_tree=a030_tree,
        authorization_raw=authorization_raw,
        created_at=created_at,
    )

    authority = _fresh_p031_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.P031,
        expected_parent_phase=PromptV3Phase.A030,
        allowed_paths=(_AUTH_PATH, _MANIFEST_PATH),
        required_evidence_kinds=("p031-suite",),
        validator_ids=("native-dependency-launch-authorization",),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.A030,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_AUTH_PATH, data=authorization_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
        ),
        evidence_handles=(_placeholder_handle("p031-suite"),),
        commit_message=(
            "ASE3-P031: seal native dependency launch authorization after A030"
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
        runner=lambda observed: (_bound_handle(observed, "p031-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "native-dependency-launch-authorization"),
        ),
        evidence_loader=_evidence_loader,
    )
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase="P031",
        child_head=candidate.commit_id,
        parent_head=a030_head,
        parent_tree=a030_tree,
        consumed_child_blobs={
            _AUTH_PATH: authorization_raw,
            _MANIFEST_PATH: manifest_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "P031 sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": "P031",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "a030_head": a030_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "authorization_path": _AUTH_PATH,
        "authorization_content_id": content_id(authorization_raw),
        "authorization_id": authorization_payload.get("authorization_id"),
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


__all__ = ("advance_prompt_v3_p031",)
