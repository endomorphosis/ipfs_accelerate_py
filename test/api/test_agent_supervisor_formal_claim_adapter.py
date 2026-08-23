"""FACP-014: supervisor legacy assurance records → FCA envelope adapter."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ProofStatus,
)
from ipfs_accelerate_py.agent_supervisor.assurance.formal_claim_adapter import (
    ADAPTER_SCHEMA,
    DIMENSION_ORDER,
    Authority,
    EnvelopeAdaptation,
    EvidenceEnvelope,
    Freshness,
    Origin,
    Proof,
    TypedIncompatibility,
    VOCAB_SCHEMA,
    adapt_assurance_level,
    adapt_capability_evidence,
    adapt_database_repair_assurance_level,
    adapt_evidence_tier,
    adapt_execution_permit,
    adapt_generic_claim_mapping,
    adapt_legacy_record,
    adapt_proof_cache_entry,
    adapt_proof_receipt,
    adapt_proof_status,
    adapt_provider_capability_evidence,
    adapt_stale_or_unknown_marker,
    adapt_untrusted_draft_cache_entry,
    project_envelope_to_assurance_level,
    project_envelope_to_proof_status,
)
from ipfs_accelerate_py.agent_supervisor.proof import database_repair_evidence as repair_evidence
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
    DraftCacheKey,
    ProofCacheEntry,
    ProofCacheKey,
    UntrustedDraftCacheEntry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)

# Closed weakest defaults (absent dimensions stay unchecked/absent/weak).
_WEAKEST = {
    "origin": "absent",
    "integrity": "unchecked",
    "authority": "unchecked",
    "policy": "unchecked",
    "proof": "none",
    "freshness": "stale",
    "effect": "not_started",
    "environment": "hermetic",
    "review": "unreviewed",
}


def _assert_only_informed(result: EnvelopeAdaptation, *dims: str) -> None:
    assert isinstance(result, EnvelopeAdaptation)
    assert result.unsafe_promotion is False
    assert set(result.informed_dimensions) == set(dims)
    envelope = result.envelope.to_dict()
    for name in DIMENSION_ORDER:
        if name in dims:
            continue
        assert envelope[name] == _WEAKEST[name], (
            f"dimension {name} must remain weakest; got {envelope[name]!r}"
        )


def _budget() -> ResourceBudget:
    return ResourceBudget()


def _kernel_evidence(obligation_id: str = "obl:1") -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:checked",
        subject_id=obligation_id,
        verifier_id="kernel:lean",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
    )


def _receipt(**changes: Any) -> ProofReceipt:
    values: dict[str, Any] = {
        "obligation_id": "obl:1",
        "plan_id": "plan:1",
        "attempt_id": "attempt:1",
        "repository_id": "repository:sha256:demo",
        "repository_tree_id": "tree:abc123",
        "ast_scope_ids": ("scope:mod.fn",),
        "premise_ids": (),
        "translator_id": "translator:t",
        "solver_id": "solver:s",
        "kernel_id": "kernel:lean",
        "toolchain_id": "toolchain:t",
        "policy_id": "policy:p",
        "resource_budget": _budget(),
        "verdict": ProofVerdict.INCONCLUSIVE,
        "evidence": (),
        "freshness": EvidenceFreshness.CURRENT,
        "provider_claimed_assurance": AssuranceLevel.UNVERIFIED,
    }
    values.update(changes)
    return ProofReceipt(**values)


# ---------------------------------------------------------------------------
# AssuranceLevel ladders (proof only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "proof"),
    [
        (AssuranceLevel.UNVERIFIED, "none"),
        (AssuranceLevel.NONE, "none"),
        (AssuranceLevel.CANDIDATE, "candidate"),
        (AssuranceLevel.SOLVER_CHECKED, "candidate"),
        (AssuranceLevel.SOLVER_VERIFIED, "candidate"),
        (AssuranceLevel.KERNEL_VERIFIED, "verified"),
        (AssuranceLevel.ATTESTED, "verified"),
        ("unverified", "none"),
        ("kernel_verified", "verified"),
    ],
)
def test_assurance_level_maps_proof_only(level: Any, proof: str) -> None:
    result = adapt_assurance_level(level)
    _assert_only_informed(result, "proof")
    assert result.envelope.proof.value == proof
    # Attestation must not invent authority.valid.
    assert result.envelope.authority is Authority.UNCHECKED


@pytest.mark.parametrize(
    ("level", "proof"),
    [
        (repair_evidence.AssuranceLevel.NONE, "none"),
        (repair_evidence.AssuranceLevel.HEURISTIC, "candidate"),
        (repair_evidence.AssuranceLevel.VALIDATED, "candidate"),
        (repair_evidence.AssuranceLevel.SOLVER_CHECKED, "candidate"),
        (repair_evidence.AssuranceLevel.KERNEL_VERIFIED, "verified"),
        (repair_evidence.AssuranceLevel.ATTESTED, "verified"),
        ("heuristic", "candidate"),
    ],
)
def test_database_repair_assurance_level_never_fills_foreign_dims(
    level: Any, proof: str
) -> None:
    result = adapt_database_repair_assurance_level(level)
    _assert_only_informed(result, "proof")
    assert result.envelope.proof.value == proof
    assert result.envelope.origin is Origin.ABSENT


def test_unknown_assurance_level_is_typed_incompatibility() -> None:
    result = adapt_assurance_level("not-a-level")
    assert isinstance(result, TypedIncompatibility)
    assert result.code == "unknown_assurance_level"


# ---------------------------------------------------------------------------
# ProofStatus (proof + optional freshness)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ProofStatus.UNPROVED, {"proof": "none"}),
        (ProofStatus.CANDIDATE, {"proof": "candidate"}),
        (ProofStatus.SOLVER_CHECKED, {"proof": "candidate"}),
        (ProofStatus.KERNEL_VERIFIED, {"proof": "verified"}),
        (ProofStatus.VALIDATED_REFUTED, {"proof": "refuted"}),
        (ProofStatus.INCONCLUSIVE, {"proof": "unknown"}),
        (ProofStatus.UNSUPPORTED, {"proof": "verifier_unavailable"}),
        (ProofStatus.ERROR, {"proof": "unknown"}),
        (ProofStatus.STALE, {"proof": "unknown", "freshness": "stale"}),
    ],
)
def test_proof_status_seed_map(status: ProofStatus, expected: dict[str, str]) -> None:
    result = adapt_proof_status(status)
    assert isinstance(result, EnvelopeAdaptation)
    assert set(result.informed_dimensions) == set(expected)
    for key, value in expected.items():
        assert result.envelope.to_dict()[key] == value
    for name in DIMENSION_ORDER:
        if name in expected:
            continue
        assert result.envelope.to_dict()[name] == _WEAKEST[name]


# ---------------------------------------------------------------------------
# EvidenceTier (proof only; bare tier is not a receipt)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "proof"),
    [
        ("query_fact", "none"),
        ("graphrag_fact", "none"),
        ("observation", "candidate"),
        ("solver_candidate", "candidate"),
        ("kernel_proof", "candidate"),
        ("cryptographic_attestation", "candidate"),
    ],
)
def test_evidence_tier_maps_conservatively(tier: str, proof: str) -> None:
    result = adapt_evidence_tier(tier)
    _assert_only_informed(result, "proof")
    assert result.envelope.proof.value == proof
    if tier in {"kernel_proof", "cryptographic_attestation"}:
        assert any("does not mint proof.verified" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Proof receipts and caches
# ---------------------------------------------------------------------------


def test_proof_receipt_uses_authoritative_assurance_not_provider_claim() -> None:
    receipt = _receipt(
        verdict=ProofVerdict.PROVED,
        evidence=(_kernel_evidence(),),
        kernel_receipt_id="kr:1",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        freshness=EvidenceFreshness.CURRENT,
    )
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    result = adapt_proof_receipt(receipt)
    assert isinstance(result, EnvelopeAdaptation)
    assert result.envelope.proof is Proof.VERIFIED
    assert result.envelope.freshness is Freshness.CURRENT
    assert result.envelope.authority is Authority.UNCHECKED
    assert "provider_claimed_assurance is ignored" in " ".join(result.notes)


def test_stale_proof_receipt_forces_freshness_stale() -> None:
    receipt = _receipt(
        verdict=ProofVerdict.PROVED,
        evidence=(_kernel_evidence(),),
        kernel_receipt_id="kr:1",
        freshness=EvidenceFreshness.STALE,
    )
    result = adapt_proof_receipt(receipt)
    assert isinstance(result, EnvelopeAdaptation)
    assert result.envelope.freshness is Freshness.STALE
    # Stale receipt must not keep proof.verified claim of currency.
    assert "freshness" in result.informed_dimensions


def test_proof_cache_entry_expired_or_incomplete_is_stale() -> None:
    receipt = _receipt(
        verdict=ProofVerdict.PROVED,
        evidence=(_kernel_evidence(),),
        kernel_receipt_id="kr:1",
        freshness=EvidenceFreshness.CURRENT,
    )
    key = ProofCacheKey(
        obligation="obl:1",
        premises=(),
        translator="translator:t",
        solver="solver:s",
        kernel="kernel:lean",
        toolchain="toolchain:t",
        theorem_registry="registry:t",
        policy="policy:p",
        resource_budget="budget:1",
        candidate_tree="tree:abc123",
    )
    live = ProofCacheEntry.create(
        key, receipt, created_at_ms=1_000, expires_at_ms=5_000, complete=True
    )
    adapted = adapt_proof_cache_entry(live, now_ms=2_000)
    assert isinstance(adapted, EnvelopeAdaptation)
    assert adapted.envelope.freshness is Freshness.CURRENT

    expired = adapt_proof_cache_entry(live, now_ms=9_000)
    assert isinstance(expired, EnvelopeAdaptation)
    assert expired.envelope.freshness is Freshness.STALE

    incomplete = ProofCacheEntry.create(
        key, receipt, created_at_ms=1_000, expires_at_ms=5_000, complete=False
    )
    adapted_incomplete = adapt_proof_cache_entry(incomplete, now_ms=2_000)
    assert isinstance(adapted_incomplete, EnvelopeAdaptation)
    assert adapted_incomplete.envelope.freshness is Freshness.STALE


def test_untrusted_draft_cache_cannot_mint_verified_current() -> None:
    key = DraftCacheKey(
        goal_digest="goal:1",
        repository_tree_digest="tree:1",
        vocabulary_digest="vocab:1",
        compiler_digest="compiler:1",
        model_route_digest="route:1",
        model_version="model:1",
        assumptions_digest="assumptions:1",
        bounds_digest="bounds:1",
        policy_digest="policy:1",
    )
    entry = UntrustedDraftCacheEntry.create(
        key,
        {"draft_id": "draft:1", "body": "candidate text"},
        created_at_ms=1,
        expires_at_ms=100,
    )
    result = adapt_untrusted_draft_cache_entry(entry)
    assert isinstance(result, EnvelopeAdaptation)
    assert result.envelope.proof is Proof.CANDIDATE
    assert result.envelope.freshness is Freshness.STALE
    assert result.envelope.origin is Origin.DECLARED
    assert result.envelope.proof is not Proof.VERIFIED


# ---------------------------------------------------------------------------
# Execution permits and capability records
# ---------------------------------------------------------------------------


def test_execution_permit_maps_authority_only() -> None:
    current = adapt_execution_permit(
        {
            "legacy_kind": "ExecutionPermit",
            "issued_at_ms": 1_000,
            "expires_at_ms": 5_000,
            "admission_receipt_id": "admission:1",
            "policy_id": "policy:implementation-daemon",
        },
        now_ms=2_000,
    )
    _assert_only_informed(current, "authority")
    assert current.envelope.authority is Authority.VALID
    assert current.envelope.policy.value == "unchecked"
    assert current.envelope.effect.value == "not_started"

    expired = adapt_execution_permit(
        {
            "issued_at_ms": 1_000,
            "expires_at_ms": 5_000,
            "admission_receipt_id": "admission:1",
        },
        now_ms=9_000,
    )
    assert isinstance(expired, EnvelopeAdaptation)
    assert expired.envelope.authority is Authority.EXPIRED
    assert expired.envelope.freshness is Freshness.STALE
    assert set(expired.informed_dimensions) == {"authority", "freshness"}


def test_capability_records_are_discovery_only() -> None:
    provider = adapt_provider_capability_evidence(
        {
            "provider_id": "grok",
            "ready": True,
            "observed_capability_cid": "baguqeera" + "a" * 50,
        }
    )
    _assert_only_informed(provider, "origin")
    assert provider.envelope.origin is Origin.DECLARED
    assert provider.envelope.environment.value == "hermetic"
    assert provider.envelope.proof is Proof.NONE

    bundle = adapt_capability_evidence(
        {
            "legacy_kind": "CapabilityEvidence",
            "attempt_cid": "baguqeera" + "b" * 50,
            "providers": {
                "grok": {"provider_id": "grok", "ready": True},
                "codex": {"provider_id": "codex", "ready": False},
            },
        }
    )
    _assert_only_informed(bundle, "origin")
    assert bundle.envelope.origin is Origin.DECLARED
    assert "live_observed" not in bundle.envelope.to_dict().values()


# ---------------------------------------------------------------------------
# Stale/unknown markers and generic claim fields
# ---------------------------------------------------------------------------


def test_stale_and_unknown_markers() -> None:
    stale = adapt_stale_or_unknown_marker("stale")
    assert isinstance(stale, EnvelopeAdaptation)
    assert stale.envelope.freshness is Freshness.STALE
    assert stale.envelope.proof is Proof.UNKNOWN

    unknown = adapt_stale_or_unknown_marker(EvidenceFreshness.UNKNOWN)
    assert isinstance(unknown, EnvelopeAdaptation)
    assert unknown.envelope.freshness is Freshness.STALE


def test_generic_forbidden_fields_are_conservative() -> None:
    success = adapt_generic_claim_mapping({"success": True})
    _assert_only_informed(success, "effect")
    assert success.envelope.effect.value == "started"

    verified = adapt_generic_claim_mapping({"verified": True, "proven": True})
    _assert_only_informed(verified, "proof")
    assert verified.envelope.proof is Proof.CANDIDATE

    available = adapt_generic_claim_mapping({"available": True, "supported": True})
    assert isinstance(available, EnvelopeAdaptation)
    assert available.informed_dimensions == ()
    assert available.envelope.to_dict() == EvidenceEnvelope.weakest().to_dict()


# ---------------------------------------------------------------------------
# Dispatch + typed incompatibility
# ---------------------------------------------------------------------------


def test_adapt_legacy_record_dispatch_covers_supported_kinds() -> None:
    assert isinstance(
        adapt_legacy_record(AssuranceLevel.CANDIDATE), EnvelopeAdaptation
    )
    assert isinstance(adapt_legacy_record(ProofStatus.STALE), EnvelopeAdaptation)
    assert isinstance(
        adapt_legacy_record("observation", legacy_kind_hint="EvidenceTier"),
        EnvelopeAdaptation,
    )
    assert isinstance(
        adapt_legacy_record(
            {"issued_at_ms": 1, "expires_at_ms": 2},
            legacy_kind_hint="ExecutionPermit",
            now_ms=1,
        ),
        EnvelopeAdaptation,
    )
    unsupported = adapt_legacy_record(object())
    assert isinstance(unsupported, TypedIncompatibility)
    assert unsupported.code == "unsupported_legacy_record"


def test_adaptation_payload_carries_vocab_identity() -> None:
    result = adapt_assurance_level(AssuranceLevel.CANDIDATE)
    payload = result.to_dict()
    assert payload["schema"] == ADAPTER_SCHEMA
    assert payload["vocab_schema"] == VOCAB_SCHEMA
    assert payload["unsafe_promotion"] is False
    assert set(payload["envelope"]) == set(DIMENSION_ORDER)


# ---------------------------------------------------------------------------
# Reverse projection refuses information-losing promotion
# ---------------------------------------------------------------------------


def test_reverse_projection_assurance_level_round_trip_when_proof_only() -> None:
    forward = adapt_assurance_level(AssuranceLevel.KERNEL_VERIFIED)
    assert isinstance(forward, EnvelopeAdaptation)
    projected = project_envelope_to_assurance_level(forward.envelope)
    assert projected is AssuranceLevel.KERNEL_VERIFIED

    # proof.verified must not promote to ATTESTED.
    attested_forward = adapt_assurance_level(AssuranceLevel.ATTESTED)
    assert isinstance(attested_forward, EnvelopeAdaptation)
    projected_attested = project_envelope_to_assurance_level(attested_forward.envelope)
    assert projected_attested is AssuranceLevel.KERNEL_VERIFIED


def test_reverse_projection_refuses_when_non_proof_dimensions_set() -> None:
    rich = EvidenceEnvelope.weakest().with_updates(
        proof=Proof.VERIFIED,
        authority=Authority.VALID,
        freshness=Freshness.CURRENT,
    )
    refused = project_envelope_to_assurance_level(rich)
    assert isinstance(refused, TypedIncompatibility)
    assert refused.code == "information_losing_reverse_projection"
    assert refused.unsafe_promotion is True


def test_reverse_projection_proof_status_uses_informed_dimensions() -> None:
    stale = adapt_proof_status(ProofStatus.STALE)
    assert isinstance(stale, EnvelopeAdaptation)
    assert (
        project_envelope_to_proof_status(
            stale.envelope, informed_dimensions=stale.informed_dimensions
        )
        is ProofStatus.STALE
    )

    inconclusive = adapt_proof_status(ProofStatus.INCONCLUSIVE)
    assert isinstance(inconclusive, EnvelopeAdaptation)
    assert (
        project_envelope_to_proof_status(
            inconclusive.envelope,
            informed_dimensions=inconclusive.informed_dimensions,
        )
        is ProofStatus.INCONCLUSIVE
    )

    rich = EvidenceEnvelope.weakest().with_updates(
        proof=Proof.CANDIDATE,
        origin=Origin.LIVE_OBSERVED,
    )
    refused = project_envelope_to_proof_status(rich)
    assert isinstance(refused, TypedIncompatibility)
    assert refused.code == "information_losing_reverse_projection"


def test_weakest_envelope_has_unchecked_absent_defaults() -> None:
    weakest = EvidenceEnvelope.weakest().to_dict()
    assert weakest == _WEAKEST
    assert weakest["integrity"] == "unchecked"
    assert weakest["authority"] == "unchecked"
    assert weakest["origin"] == "absent"
