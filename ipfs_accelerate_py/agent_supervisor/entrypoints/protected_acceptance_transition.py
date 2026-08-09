"""Public composition facade for prompt-v3 protected transitions."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, Mapping

from ..core.protected_acceptance_contracts import (
    ArtifactSigner,
    CandidatePlan,
    EvidenceHandle,
    PhaseCandidateRequest,
    PhaseEvidenceResult,
    ProductProvenance,
    ProductProvenanceInspector,
    ProductProvenanceRequest,
    PromptV3Phase,
    ProtectedAcceptanceDenied,
    ProtectedAcceptanceError,
    PublicationResult,
    QuiescenceObservation,
    QuiescenceObserver,
    QuiescenceRequest,
    RejectionResult,
    RuntimeAuthorityLoader,
    RuntimeAuthorityValidator,
    RuntimeLaunchAuthorityRequest,
    SignedArtifactRequest,
    ValidatedCandidate,
    VerifiedRuntimeLaunchAuthority,
    canonical_json_bytes,
    content_id,
)
from ..merge.protected_acceptance_transition import (
    TransitionHooks,
    build_phase_candidate,
    publish_phase_candidate,
    reject_phase_candidate,
    run_phase_evidence,
    validate_phase_candidate,
)

ROOT_PIN_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.local-profile-lifecycle-root-pin@1"
)
REVIEWER_INITIALIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.prompt-v3-reviewer-initialization@1"
)
PROVIDER_AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-policy-authorization@2"
)
RELOAD_AUTHORIZATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.provider-attempt-daemon-reload@2"
)
_NO_TRANSITION_HOOKS = TransitionHooks()


def _exact_body(
    body: Mapping[str, Any], fields: tuple[str, ...], context: str
) -> dict[str, Any]:
    if not isinstance(body, Mapping) or set(body) != set(fields):
        raise ProtectedAcceptanceDenied(f"{context} requires exact fields")
    normalized = dict(body)
    try:
        canonical_json_bytes(normalized).decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise ProtectedAcceptanceDenied(f"{context} must be strict ASCII") from exc
    return normalized


def freeze_prompt_v3_product_provenance(
    request: ProductProvenanceRequest,
    *,
    inspector: ProductProvenanceInspector,
) -> ProductProvenance:
    """Freeze one source/replay/integrated generation via an injected inspector."""

    if not isinstance(request, ProductProvenanceRequest) or not callable(inspector):
        raise TypeError("provenance freeze requires typed request and inspector")
    result = inspector(request)
    if not isinstance(result, ProductProvenance) or result.task_id != request.task_id:
        raise ProtectedAcceptanceDenied(
            "provenance inspector returned an unbound product"
        )
    expected_commits = (
        request.source_commit,
        request.replay_commit,
        request.integrated_commit,
    )
    records = (result.source, result.replay, result.integrated)
    if tuple(record.commit for record in records) != expected_commits:
        raise ProtectedAcceptanceDenied(
            "provenance inspector substituted a requested generation"
        )
    if any(
        tuple(item.path for item in record.files) != request.product_paths
        for record in records
    ):
        raise ProtectedAcceptanceDenied(
            "provenance inspector substituted the requested product paths"
        )
    expected_evidence = (
        request.source_test_evidence,
        request.replay_test_evidence,
        request.integrated_test_evidence,
    )
    if tuple(record.test_evidence for record in records) != expected_evidence:
        raise ProtectedAcceptanceDenied(
            "provenance inspector substituted requested test evidence"
        )
    return result


def build_prompt_v3_root_pin(request: SignedArtifactRequest) -> bytes:
    """Build the exact validator-compatible R root-pin bytes."""

    if (
        not isinstance(request, SignedArtifactRequest)
        or request.phase is not PromptV3Phase.R
    ):
        raise ProtectedAcceptanceDenied("root pin requires fresh R authority")
    body = _exact_body(
        request.body,
        (
            "board_namespace",
            "base_head",
            "base_tree",
            "root_identity_did",
            "pinned_at_ms",
        ),
        "root pin",
    )
    if body["base_head"] != request.authority.parent_commit:
        raise ProtectedAcceptanceDenied("root pin must name its exact Q parent")
    unsigned = {"schema": ROOT_PIN_SCHEMA, **body}
    unsigned["pin_id"] = content_id(canonical_json_bytes(unsigned))
    return canonical_json_bytes(unsigned) + b"\n"


def _phase_artifact(
    request: SignedArtifactRequest,
    *,
    phase: PromptV3Phase,
    schema: str,
) -> bytes:
    if not isinstance(request, SignedArtifactRequest) or request.phase is not phase:
        raise ProtectedAcceptanceDenied(
            f"artifact requires fresh {phase.value} authority"
        )
    body = dict(request.body)
    if any(
        key in body for key in ("status", "branch", "path", "raw_key", "environment")
    ):
        raise ProtectedAcceptanceDenied("artifact body contains an authority override")
    envelope = {
        "schema": schema,
        "phase": phase.value,
        "parent_commit": request.authority.parent_commit,
        "phase_authority_id": request.authority.authority_id,
        "body": body,
    }
    return canonical_json_bytes(envelope) + b"\n"


def initialize_prompt_v3_reviewer_after_root_pin(
    request: SignedArtifactRequest,
) -> bytes:
    if "root_pin_content_id" not in request.body:
        raise ProtectedAcceptanceDenied(
            "reviewer initialization requires the actual R root pin"
        )
    return _phase_artifact(
        request,
        phase=PromptV3Phase.P019,
        schema=REVIEWER_INITIALIZATION_SCHEMA,
    )


def build_prompt_v3_provider_authorization(request: SignedArtifactRequest) -> bytes:
    required = {
        "root_pin_content_id",
        "reviewer_profile_content_id",
        "lifecycle_witness_content_id",
    }
    if not required.issubset(request.body):
        raise ProtectedAcceptanceDenied(
            "provider authorization lacks post-R reviewer bindings"
        )
    return _phase_artifact(
        request,
        phase=PromptV3Phase.P019,
        schema=PROVIDER_AUTHORIZATION_SCHEMA,
    )


def canonical_prompt_v3_review_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical full receipt bytes excluding only ``review.signature``."""

    if not isinstance(payload, Mapping):
        raise TypeError("review payload must be a mapping")
    review = payload.get("review")
    if not isinstance(review, Mapping) or type(review.get("signature")) is not str:
        raise ProtectedAcceptanceError("review.signature is required")
    unsigned = dict(payload)
    unsigned_review = dict(review)
    unsigned_review.pop("signature")
    unsigned["review"] = unsigned_review
    rendered = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    try:
        rendered.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProtectedAcceptanceDenied(
            "local-profile acceptance signing is restricted to strict ASCII"
        ) from exc
    return rendered


def _default_local_profile_signer(
    payload: Mapping[str, Any], expected_repository_cid: str
) -> Mapping[str, str]:
    if os.environ.get("AGENT_SUPERVISOR_LOCAL_PROFILE_KEY") is not None:
        raise ProtectedAcceptanceDenied(
            "raw local-profile key environment is forbidden"
        )
    from .local_profile import load_local_profile, sign_profile_binding

    # Load before and after signing to close rotation/revocation races.  The
    # default secured lifecycle/key-file locations are the only accepted
    # authority source; this facade exposes no profile/key path override.
    before = load_local_profile(repository_cid=expected_repository_cid)
    result = sign_profile_binding(profile_dir=None, lifecycle_dir=None, payload=payload)
    after = load_local_profile(repository_cid=before.repository_cid)
    if (
        before.profile_id != after.profile_id
        or before.identity_did != after.identity_did
        or before.lifecycle_generation != after.lifecycle_generation
        or result.get("profile_id") != before.profile_id
        or result.get("identity") != before.identity_did
    ):
        raise ProtectedAcceptanceDenied(
            "local profile rotated or was revoked while signing"
        )
    return result


def sign_prompt_v3_operator_artifact(
    payload: Mapping[str, Any],
    *,
    signer: ArtifactSigner | None = None,
) -> dict[str, Any]:
    """Sign receipt-minus-signature and transcode standard Base64 exactly once."""

    unsigned_bytes = canonical_prompt_v3_review_bytes(payload)
    unsigned = json.loads(unsigned_bytes.decode("ascii"))
    if signer is None:
        expected_repository_cid = payload.get("repository_cid")
        if type(expected_repository_cid) is not str or not expected_repository_cid:
            raise ProtectedAcceptanceDenied(
                "artifact must bind the expected repository before local-profile signing"
            )
        adapter = lambda value: _default_local_profile_signer(
            value, expected_repository_cid
        )
    else:
        adapter = signer
    if not callable(adapter):
        raise TypeError("operator signer must implement ArtifactSigner")
    signed = adapter(unsigned)
    if not isinstance(signed, Mapping) or set(signed) != {
        "identity",
        "signature",
        "profile_id",
    }:
        raise ProtectedAcceptanceDenied("operator signer returned unsupported fields")
    signature_token = signed.get("signature")
    if type(signature_token) is not str:
        raise ProtectedAcceptanceDenied("operator signer returned an invalid signature")
    try:
        signature = base64.b64decode(signature_token.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ProtectedAcceptanceDenied(
            "operator signer signature is not standard Base64"
        ) from exc
    if (
        len(signature) != 64
        or base64.b64encode(signature).decode("ascii") != signature_token
    ):
        raise ProtectedAcceptanceDenied(
            "operator signer signature is not canonical Ed25519 bytes"
        )
    review = payload["review"]
    identity = signed["identity"]
    if type(identity) is not str or not identity.startswith("did:key:z"):
        raise ProtectedAcceptanceDenied("operator signer identity is invalid")
    expected_identity = review.get("identity_did") or review.get("reviewer_identity")
    if expected_identity is not None and expected_identity != identity:
        raise ProtectedAcceptanceDenied(
            "operator signer is not the authorized reviewer"
        )
    result = dict(payload)
    result_review = dict(review)
    result_review["signature"] = "ed25519:" + base64.urlsafe_b64encode(
        signature
    ).decode("ascii").rstrip("=")
    result["review"] = result_review
    return result


def build_prompt_v3_phase_candidate(
    request: PhaseCandidateRequest,
    *,
    authority_validator: Callable[[Any, int], bool],
    hooks: TransitionHooks = _NO_TRANSITION_HOOKS,
) -> CandidatePlan:
    return build_phase_candidate(
        request, authority_validator=authority_validator, hooks=hooks
    )


def run_prompt_v3_phase_evidence(
    candidate: CandidatePlan,
    *,
    runner: Callable[[CandidatePlan], tuple[EvidenceHandle, ...]],
    evidence_loader: Callable[[CandidatePlan, EvidenceHandle], bytes],
) -> PhaseEvidenceResult:
    return run_phase_evidence(candidate, runner, evidence_loader)


def validate_prompt_v3_phase_candidate(
    candidate: CandidatePlan,
    evidence: PhaseEvidenceResult,
    *,
    validator: Callable[
        [CandidatePlan, PhaseEvidenceResult], tuple[EvidenceHandle, ...]
    ],
    evidence_loader: Callable[[CandidatePlan, EvidenceHandle], bytes],
) -> ValidatedCandidate:
    return validate_phase_candidate(candidate, evidence, validator, evidence_loader)


def publish_prompt_v3_phase_candidate(
    validated: ValidatedCandidate,
    *,
    authority_validator: Callable[[Any, int], bool],
    pre_cas_validator: Callable[[ValidatedCandidate], bool],
    hooks: TransitionHooks = _NO_TRANSITION_HOOKS,
) -> PublicationResult:
    return publish_phase_candidate(
        validated,
        authority_validator=authority_validator,
        pre_cas_validator=pre_cas_validator,
        hooks=hooks,
    )


def reject_prompt_v3_phase_candidate(candidate: CandidatePlan) -> RejectionResult:
    return reject_phase_candidate(candidate)


def observe_prompt_v3_quiescence(
    request: QuiescenceRequest,
    *,
    observer: QuiescenceObserver,
) -> QuiescenceObservation:
    if not isinstance(request, QuiescenceRequest) or not callable(observer):
        raise TypeError("quiescence requires typed request and observer")
    result = observer(request)
    if (
        not isinstance(result, QuiescenceObservation)
        or result.generation != request.generation
        or result.terminal_lane_ids != request.required_lane_ids
        or result.fenced is not True
    ):
        raise ProtectedAcceptanceDenied(
            "old generation is not exactly terminal and fenced"
        )
    return result


def build_prompt_v3_reload_authorization(request: SignedArtifactRequest) -> bytes:
    if (
        "prior_authorization_id" not in request.body
        or request.body["prior_authorization_id"]
        not in (
            None,
            "",
        )
        and not str(request.body["prior_authorization_id"]).startswith("sha256:")
    ):
        raise ProtectedAcceptanceDenied(
            "reload prior authorization must be canonical nullable content ID"
        )
    required = {
        "runtime_native_authorization_id",
        "target_generation",
        "accepted_a031_id",
        "accepted_a032_id",
        "accepted_a023_027_id",
    }
    if not required.issubset(request.body):
        raise ProtectedAcceptanceDenied(
            "reload lacks accepted gates or fresh generation authority"
        )
    if request.body["runtime_native_authorization_id"] == request.body.get(
        "p031_authorization_id"
    ):
        raise ProtectedAcceptanceDenied(
            "P031 authority cannot authorize post-L runtime"
        )
    return _phase_artifact(
        request,
        phase=PromptV3Phase.L,
        schema=RELOAD_AUTHORIZATION_SCHEMA,
    )


def load_verified_prompt_v3_runtime_launch_authority(
    request: RuntimeLaunchAuthorityRequest,
    *,
    loader: RuntimeAuthorityLoader,
    validator: RuntimeAuthorityValidator,
) -> VerifiedRuntimeLaunchAuthority:
    """Narrow hook for a later full-chain loader/validator composition."""

    if (
        not isinstance(request, RuntimeLaunchAuthorityRequest)
        or not callable(loader)
        or not callable(validator)
    ):
        raise TypeError(
            "runtime authority load requires typed request, loader, and validator"
        )
    loaded = loader(request)
    if not isinstance(loaded, Mapping):
        raise ProtectedAcceptanceDenied(
            "runtime authority loader returned an untyped value"
        )
    verified = validator(request, loaded)
    if not isinstance(verified, VerifiedRuntimeLaunchAuthority):
        raise ProtectedAcceptanceDenied(
            "runtime authority validator returned an untyped value"
        )
    if (
        verified.l_commit != request.l_commit
        or verified.l_tree != request.expected_l_tree
        or verified.l_raw_content_id != request.expected_l_raw_content_id
        or verified.target_generation != request.target_generation
    ):
        raise ProtectedAcceptanceDenied(
            "runtime launch authority drifted from the exact L binding"
        )
    return verified


__all__ = (
    "build_prompt_v3_phase_candidate",
    "build_prompt_v3_provider_authorization",
    "build_prompt_v3_reload_authorization",
    "build_prompt_v3_root_pin",
    "canonical_prompt_v3_review_bytes",
    "freeze_prompt_v3_product_provenance",
    "initialize_prompt_v3_reviewer_after_root_pin",
    "load_verified_prompt_v3_runtime_launch_authority",
    "observe_prompt_v3_quiescence",
    "publish_prompt_v3_phase_candidate",
    "reject_prompt_v3_phase_candidate",
    "run_prompt_v3_phase_evidence",
    "sign_prompt_v3_operator_artifact",
    "validate_prompt_v3_phase_candidate",
)
