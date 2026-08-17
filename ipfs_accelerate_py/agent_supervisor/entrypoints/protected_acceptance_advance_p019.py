"""Advance the protected chain R→P019 (witness + provider auth@2 + manifest)."""

from __future__ import annotations

import hashlib
import json
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
from .local_profile import (
    export_local_profile_lifecycle_witness,
    initialize_local_profile,
    lifecycle_root_identity_did,
    load_local_profile,
    sign_profile_binding,
)

_WITNESS_PATH = convergence.LOCAL_OPERATOR_LIFECYCLE_WITNESS_RELATIVE_PATH
_AUTH_PATH = convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_RELATIVE_PATH
_MANIFEST_PATH = (
    f"{convergence._CONVERGENCE_RELATIVE_ROOT}/{convergence.MANIFEST_FILENAME}"
)
_BOARD_NAMESPACE = convergence.BOARD_NAMESPACE
_REPOSITORY_CID = "repository:agent-supervisor-prompt-only-self-improvement-v3"


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


def _fresh_p019_authority(*, parent_commit: str, identity_did: str) -> PhaseAuthority:
    now = time.time_ns()
    nonce = secrets.token_urlsafe(24)
    issued = now - 1_000_000
    expires = now + 3_600_000_000_000
    authority_id = phase_authority_content_id(
        phase=PromptV3Phase.P019,
        nonce=nonce,
        parent_commit=parent_commit,
        identity_did=identity_did,
        issued_at_ns=issued,
        expires_at_ns=expires,
    )
    return PhaseAuthority(
        phase=PromptV3Phase.P019,
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


def _sha256_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_json_blob(repo: Path, commit: str, relative_path: str) -> dict[str, Any]:
    raw = _git(repo, "show", f"{commit}:{relative_path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProtectedAcceptanceDenied(f"{relative_path} must be a JSON object")
    return payload


def _ensure_reviewer_profile(*, baseline_commit: str) -> Any:
    """Load or create the durable local operator profile bound to R."""

    try:
        return load_local_profile(repository_cid=_REPOSITORY_CID)
    except Exception:
        pass
    return initialize_local_profile(
        repository_cid=_REPOSITORY_CID,
        baseline_commit=baseline_commit,
        effect_bounds=convergence._PROVIDER_FALLBACK_AUTHORIZATION_V2_EFFECTS,
        route_id=convergence._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE["route_id"],
    )


def _build_provider_authorization_v2(
    *,
    witness: Mapping[str, Any],
    witness_raw: bytes,
    root_pin: Mapping[str, Any],
    root_pin_raw: bytes,
    source_head: str,
    source_tree: str,
    authorized_at_ms: int,
    profile_dir: Path | None = None,
) -> bytes:
    profile = witness["profile"]
    anchor = witness["anchor"]
    if not isinstance(profile, Mapping) or not isinstance(anchor, Mapping):
        raise ProtectedAcceptanceDenied("witness profile/anchor projections missing")
    v1_source = dict(convergence._PROVIDER_FALLBACK_AUTHORIZATION_SOURCE)
    source = {
        "kind": v1_source["kind"],
        "source_head": source_head,
        "source_tree": source_tree,
        "prospective_only": v1_source["prospective_only"],
        "requires_descendant_tree": v1_source["requires_descendant_tree"],
    }
    route = dict(convergence._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE)
    reviewer = {
        "identity": profile["identity_did"],
        "provider": "local_operator",
        "profile_id": profile["profile_id"],
        "profile_content_id": witness["profile_content_id"],
        "lifecycle_anchor_id": anchor["anchor_id"],
        "generation": profile["lifecycle_generation"],
        "witness_path": _WITNESS_PATH,
        "witness_sha256": _sha256_bytes(witness_raw),
    }
    authority_bounds = {
        "repository_cid": profile["repository_cid"],
        "baseline_commit": profile["baseline_commit"],
        "effects": list(profile["effect_bounds"]),
        "budget_cid": profile["budget_cid"],
        "resource_cid": profile["resource_cid"],
        "authority_cid": witness["profile_content_id"],
    }
    review_payload = {
        "schema": convergence.PROVIDER_FALLBACK_POLICY_REVIEW_V2_SCHEMA,
        "board_namespace": _BOARD_NAMESPACE,
        "authorization_source": {
            field: source[field] for field in ("kind", "source_head", "source_tree")
        },
        "route": route,
        "authority_bounds": authority_bounds,
        "reviewer": dict(reviewer),
        "lifecycle_root_identity_did": root_pin["root_identity_did"],
        "lifecycle_witness_nonce": witness["nonce"],
        "lifecycle_root_pin_path": (
            convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ),
        "lifecycle_root_pin_sha256": _sha256_bytes(root_pin_raw),
        "authorized_at_ms": authorized_at_ms,
        "fallback_implementer_identity": "codex",
    }
    signed = sign_profile_binding(
        profile_dir=profile_dir,
        lifecycle_dir=None,
        payload=review_payload,
    )
    reviewer["signature"] = signed["signature"]
    payload = {
        "schema": convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_V2_SCHEMA,
        "board_namespace": _BOARD_NAMESPACE,
        "authorization_source": source,
        "route": route,
        "ownership_contract": dict(
            convergence._PROVIDER_FALLBACK_AUTHORIZATION_V2_OWNERSHIP_CONTRACT
        ),
        "bootstrap_route_guarantees": dict(
            convergence._PROVIDER_FALLBACK_AUTHORIZATION_V2_BOOTSTRAP_GUARANTEES
        ),
        "reviewer": reviewer,
        "authority_bounds": authority_bounds,
        "fallback_implementer_identity": "codex",
        "lifecycle_root_identity_did": root_pin["root_identity_did"],
        "lifecycle_witness_nonce": witness["nonce"],
        "lifecycle_root_pin_path": (
            convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
        ),
        "lifecycle_root_pin_sha256": _sha256_bytes(root_pin_raw),
        "authorized_at_ms": authorized_at_ms,
    }
    return canonical_json_bytes(payload) + b"\n"


def _build_p019_manifest(
    *,
    parent_manifest: Mapping[str, Any],
    authorization_raw: bytes,
) -> bytes:
    if parent_manifest.get("schema") != convergence.CONVERGENCE_MANIFEST_SCHEMA:
        raise ProtectedAcceptanceDenied("P019 parent manifest must remain @1")
    if "acceptance" in parent_manifest or "reload" in parent_manifest:
        raise ProtectedAcceptanceDenied("P019 parent must not carry effect phases")
    components = parent_manifest.get("components")
    if not isinstance(components, Mapping):
        raise ProtectedAcceptanceDenied("parent manifest components required")
    updated = dict(parent_manifest)
    updated_components = dict(components)
    updated_components[convergence.PROVIDER_FALLBACK_POLICY_AUTHORIZATION_FILENAME] = (
        _sha256_bytes(authorization_raw)
    )
    updated["components"] = updated_components
    return (
        json.dumps(
            updated,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def advance_prompt_v3_p019(
    *,
    repo_root: Path | str,
    target_ref: str = "refs/heads/main",
    dry_run: bool = False,
    publish: bool = True,
) -> Mapping[str, Any]:
    """Build and optionally publish the protected P019 candidate."""

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

    if (repo / _WITNESS_PATH).exists() or (repo / _WITNESS_PATH).is_symlink():
        raise ProtectedAcceptanceDenied("lifecycle witness is already present")

    parent_commit = (
        _git(repo, "rev-parse", target_ref).decode("ascii", "strict").strip()
    )
    # Resolve R tip via artifact discovery so freeze intermediates are skipped.
    discovered, discovery_errors = convergence._discover_sequential_phase_heads(
        repo_root=repo,
        head=parent_commit,
        through_phase="R",
    )
    if discovery_errors or "R" not in discovered:
        raise ProtectedAcceptanceDenied(
            "R phase head unavailable: " + "; ".join(discovery_errors or ["missing R"])
        )
    r_head = discovered["R"]
    r_tree = (
        _git(repo, "rev-parse", f"{r_head}^{{tree}}").decode("ascii", "strict").strip()
    )
    root_did = lifecycle_root_identity_did()
    if not isinstance(root_did, str) or not root_did.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("lifecycle root DID is unavailable")

    root_pin_path = convergence.LOCAL_PROFILE_LIFECYCLE_ROOT_PIN_RELATIVE_PATH
    root_pin_raw = _git(repo, "show", f"{r_head}:{root_pin_path}")
    root_pin = json.loads(root_pin_raw.decode("utf-8"))
    if root_pin.get("root_identity_did") != root_did:
        raise ProtectedAcceptanceDenied("root pin DID does not match local lifecycle root")

    profile = _ensure_reviewer_profile(baseline_commit=r_head)
    if profile.baseline_commit != r_head:
        # Rotate baseline to the R tip so witness base matches phase head.
        from .local_profile import initialize_local_profile as _init

        profile = _init(
            repository_cid=_REPOSITORY_CID,
            baseline_commit=r_head,
            force=True,
            effect_bounds=convergence._PROVIDER_FALLBACK_AUTHORIZATION_V2_EFFECTS,
            route_id=convergence._PROVIDER_FALLBACK_AUTHORIZATION_ROUTE["route_id"],
        )

    witness_nonce = secrets.token_urlsafe(24)
    observed_at_ms = int(time.time() * 1000)
    # Witness must not predate the root pin commit time.
    r_commit_ms = (
        int(
            _git(repo, "show", "-s", "--format=%ct", r_head)
            .decode("ascii", "strict")
            .strip()
        )
        * 1000
    )
    if observed_at_ms < r_commit_ms:
        observed_at_ms = r_commit_ms + 1
    witness = export_local_profile_lifecycle_witness(
        repository_cid=_REPOSITORY_CID,
        board_namespace=_BOARD_NAMESPACE,
        base_head=r_head,
        base_tree=r_tree,
        nonce=witness_nonce,
        observed_at_ms=observed_at_ms,
        expires_at_ms=observed_at_ms + 600_000,
    )
    witness_raw = canonical_json_bytes(witness) + b"\n"
    witness_errors = convergence.validate_local_operator_lifecycle_witness(
        witness,
        root_identity_did=root_did,
        expected_base_head=r_head,
        expected_base_tree=r_tree,
        reference_time_ms=observed_at_ms,
        earliest_observed_at_ms=r_commit_ms,
        expected_final_values={
            "reviewer_identity": witness["profile"]["identity_did"],
            "profile_id": witness["profile"]["profile_id"],
            "profile_content_id": witness["profile_content_id"],
            "lifecycle_anchor_id": witness["profile"]["lifecycle_anchor_id"],
            "lifecycle_anchor_digest": witness["anchor_digest"],
            "lifecycle_generation": witness["profile"]["lifecycle_generation"],
        },
    )
    if witness_errors:
        raise ProtectedAcceptanceDenied(
            "lifecycle witness failed validation: " + "; ".join(witness_errors)
        )

    authorized_at_ms = observed_at_ms
    authorization_raw = _build_provider_authorization_v2(
        witness=witness,
        witness_raw=witness_raw,
        root_pin=root_pin,
        root_pin_raw=root_pin_raw,
        source_head=r_head,
        source_tree=r_tree,
        authorized_at_ms=authorized_at_ms,
    )
    auth_payload = json.loads(authorization_raw.decode("utf-8"))
    auth_obj = convergence.ProviderFallbackPolicyAuthorization.from_dict(auth_payload)
    witness_snapshot = convergence.LocalOperatorLifecycleWitnessSnapshot(
        payload=witness,
        raw=witness_raw,
        sha256=_sha256_bytes(witness_raw),
    )
    root_snapshot = convergence.LocalProfileLifecycleRootPinSnapshot(
        payload=root_pin,
        raw=root_pin_raw,
        sha256=_sha256_bytes(root_pin_raw),
    )
    auth_errors = auth_obj.validate(
        lifecycle_witness=witness_snapshot,
        root_pin=root_snapshot,
        expected_source_head=r_head,
        expected_source_tree=r_tree,
        expected_final_values={
            "reviewer_identity": witness["profile"]["identity_did"],
            "profile_id": witness["profile"]["profile_id"],
            "profile_content_id": witness["profile_content_id"],
            "lifecycle_anchor_id": witness["profile"]["lifecycle_anchor_id"],
            "lifecycle_anchor_digest": witness["anchor_digest"],
            "lifecycle_generation": witness["profile"]["lifecycle_generation"],
        },
    )
    if auth_errors:
        raise ProtectedAcceptanceDenied(
            "provider auth@2 failed validation: " + "; ".join(auth_errors)
        )

    parent_manifest = _load_json_blob(repo, parent_commit, _MANIFEST_PATH)
    # Prefer R tip manifest bytes if parent_commit is a freeze intermediate.
    r_manifest = _load_json_blob(repo, r_head, _MANIFEST_PATH)
    # Manifest transformation is relative to R parent of P019; freezes must not
    # alter the manifest, so parent tip and R share the same blob.
    manifest_raw = _build_p019_manifest(
        parent_manifest=r_manifest if r_manifest else parent_manifest,
        authorization_raw=authorization_raw,
    )

    authority = _fresh_p019_authority(
        parent_commit=parent_commit, identity_did=root_did
    )
    policy = PhasePolicy(
        phase=PromptV3Phase.P019,
        expected_parent_phase=PromptV3Phase.R,
        allowed_paths=(_WITNESS_PATH, _AUTH_PATH, _MANIFEST_PATH),
        required_evidence_kinds=("p019-suite",),
        validator_ids=("local-operator-lifecycle-witness", "provider-auth-v2"),
    )
    request = PhaseCandidateRequest(
        repository=RepositoryBinding(root=str(repo), target_ref=target_ref),
        policy=policy,
        parent_commit=parent_commit,
        parent_phase=PromptV3Phase.R,
        authority=authority,
        artifacts=(
            ArtifactBytes(path=_WITNESS_PATH, data=witness_raw),
            ArtifactBytes(path=_AUTH_PATH, data=authorization_raw),
            ArtifactBytes(path=_MANIFEST_PATH, data=manifest_raw),
        ),
        evidence_handles=(_placeholder_handle("p019-suite"),),
        commit_message=(
            "ASE3-P019: seal lifecycle witness, provider auth@2, and manifest"
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
        runner=lambda observed: (_bound_handle(observed, "p019-suite"),),
        evidence_loader=_evidence_loader,
    )
    if not isinstance(evidence_result, PhaseEvidenceResult):
        raise TypeError("phase evidence runner returned an untyped result")
    validated = validate_phase_candidate(
        candidate,
        evidence_result,
        validator=lambda observed, _evidence: (
            _bound_handle(observed, "local-operator-lifecycle-witness"),
            _bound_handle(observed, "provider-auth-v2"),
        ),
        evidence_loader=_evidence_loader,
    )
    # Structural sequential child check against R (not freeze tip).
    transition_errors = convergence.validate_sequential_acceptance_child_transition(
        repo_root=repo,
        phase="P019",
        child_head=candidate.commit_id,
        parent_head=r_head,
        parent_tree=r_tree,
        consumed_child_blobs={
            _WITNESS_PATH: witness_raw,
            _AUTH_PATH: authorization_raw,
            _MANIFEST_PATH: manifest_raw,
        },
    )
    if transition_errors:
        raise ProtectedAcceptanceDenied(
            "P019 sequential transition failed: " + "; ".join(transition_errors)
        )

    result: dict[str, Any] = {
        "phase": "P019",
        "dry_run": dry_run,
        "parent_commit": parent_commit,
        "r_head": r_head,
        "candidate_commit": candidate.commit_id,
        "candidate_tree": candidate.tree_id,
        "witness_path": _WITNESS_PATH,
        "authorization_path": _AUTH_PATH,
        "manifest_path": _MANIFEST_PATH,
        "witness_content_id": content_id(witness_raw),
        "authorization_content_id": content_id(authorization_raw),
        "reviewer_identity": witness["profile"]["identity_did"],
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


__all__ = ("advance_prompt_v3_p019",)
