"""CBP-025: typed claim/evidence semantics and lifecycle tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    CACHE_LOOKUP_HIT,
    CACHE_LOOKUP_MISS,
    CLAIM_CATALOG_VERSION,
    CODE_CLAIM_RECORD_INTERFACE,
    CODE_CLAIM_RECORD_SCHEMA,
    ClaimFamily,
    ClaimStatus,
    CodeClaimContractError,
    CodeClaimRecord,
    EvidenceTier,
    InvalidationSelector,
    InvalidationSelectorKind,
    apply_cache_lookup,
    build_invalidation_selectors,
    build_open_claim,
    cache_miss_status,
    claim_from_implementation_evidence,
    claim_from_obligation,
    claim_from_query_fact,
    claim_from_receipt,
    evaluate_invalidation,
    evidence_kind_to_tier,
    mark_claim_stale,
    max_assurance_for_tiers,
    reject_natural_language_claim,
    resolve_claim_family,
    tiers_can_independently_mint_kernel,
)
from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    ImplementationEvidenceKind,
    ImplementationResultEvidence,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


def _obligation(**changes: object) -> CodeProofObligation:
    values = {
        "repository_id": "repository:sha256:demo",
        "repository_tree_id": "tree:abc123",
        "ast_scope_ids": ("scope:mod.fn",),
        "statement": "∀x. P(x)",
        "template_id": "lease-uniqueness-and-fencing",
        "template_version": "1.0.0",
        "template_semantic_hash": "sha256:template",
        "premise_ids": ("premise:a",),
        "invariant_class": "security.lease",
        "task_id": "CBP-025",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
    }
    values.update(changes)
    return CodeProofObligation(**values)  # type: ignore[arg-type]


def _kernel(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:checked-lean-source",
        subject_id=obligation_id,
        verifier_id="kernel:lean-4.19",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
    )


def _solver_counterexample(obligation_id: str) -> ProofEvidence:
    return ProofEvidence(
        kind=EvidenceKind.SOLVER_RESULT,
        authority=EvidenceAuthority.SOLVER,
        verdict=EvidenceVerdict.REJECTED,
        artifact_id="artifact:counterexample",
        subject_id=obligation_id,
        verifier_id="solver:z3@4.13",
        freshness=EvidenceFreshness.CURRENT,
        independent=True,
        simulated=False,
        metadata={"counterexample_verified": True},
    )


def _receipt(
    obligation: CodeProofObligation,
    evidence: tuple[ProofEvidence, ...],
    **changes: object,
) -> ProofReceipt:
    values = {
        "obligation_id": obligation.obligation_id,
        "plan_id": "plan:demo",
        "attempt_id": "attempt:one",
        "repository_id": obligation.repository_id,
        "repository_tree_id": obligation.repository_tree_id,
        "ast_scope_ids": obligation.ast_scope_ids,
        "premise_ids": obligation.premise_ids,
        "translator_id": "translator:python-to-lean@1",
        "solver_id": "solver:z3@4.13",
        "kernel_id": "kernel:lean-4.19",
        "toolchain_id": "toolchain:nix-lock-sha256",
        "policy_id": "policy:formal-v1",
        "resource_budget": ResourceBudget(wall_time_ms=1000),
        "verdict": ProofVerdict.PROVED,
        "evidence": evidence,
        "provider_id": "provider:hammer",
        "provider_claimed_assurance": AssuranceLevel.ATTESTED,
        "freshness": EvidenceFreshness.CURRENT,
        "theorem_registry_id": "registry:reviewed-v3",
    }
    values.update(changes)
    return ProofReceipt(**values)  # type: ignore[arg-type]


def _open_claim(**changes: object) -> CodeClaimRecord:
    values = {
        "property_id": "property:lease-uniqueness-and-fencing",
        "claim_family": ClaimFamily.SECURITY_PROPERTY,
        "repository_id": "repository:sha256:demo",
        "repository_tree_id": "tree:abc123",
        "scope_ids": ("scope:mod.fn",),
        "premise_ids": ("premise:a",),
        "assumption_ids": ("assumption:env",),
        "producer_id": "producer:cbp",
        "toolchain_id": "toolchain:nix-lock-sha256",
        "policy_id": "policy:formal-v1",
        "catalog_version": CLAIM_CATALOG_VERSION,
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
        "statement": "lease uniqueness under fencing tokens",
        "template_id": "lease-uniqueness-and-fencing",
    }
    values.update(changes)
    return build_open_claim(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Canonical identity / round-trip
# ---------------------------------------------------------------------------


def test_code_claim_record_canonical_round_trip_and_content_identity() -> None:
    first = _open_claim(
        scope_ids=("scope:z", "scope:a"),
        premise_ids=("premise:b", "premise:a"),
    )
    second = _open_claim(
        scope_ids=("scope:a", "scope:z"),
        premise_ids=("premise:a", "premise:b"),
    )
    assert first.claim_id == second.claim_id
    assert first.claim_id.startswith("baguqeera")
    assert first.to_dict()["schema"] == CODE_CLAIM_RECORD_SCHEMA
    assert first.to_dict()["interface"] == CODE_CLAIM_RECORD_INTERFACE
    assert first.status is ClaimStatus.OPEN

    payload = first.to_record()
    assert payload["claim_id"] == first.claim_id
    assert "claim_id" not in first.to_dict()  # identity is non-recursive

    restored = CodeClaimRecord.from_dict(first.to_dict())
    assert restored.claim_id == first.claim_id
    assert restored.scope_ids == ("scope:a", "scope:z")
    assert restored.premise_ids == ("premise:a", "premise:b")
    assert restored.claim_family is ClaimFamily.SECURITY_PROPERTY
    assert restored.required_assurance is AssuranceLevel.KERNEL_VERIFIED

    # Mismatched claimed identity fails closed.
    bad = first.to_dict()
    bad["claim_id"] = "baguqeera_forged"
    with pytest.raises(CodeClaimContractError, match="content identity"):
        CodeClaimRecord.from_dict(bad)


def test_record_binds_required_fields_and_invalidation_selectors() -> None:
    claim = _open_claim(
        obligation_id="obligation:demo",
        assumption_ids=("assumption:env", "assumption:bounds"),
    )
    payload = claim.to_dict()
    for key in (
        "property_id",
        "obligation_id",
        "claim_family",
        "repository_id",
        "repository_tree_id",
        "scope_ids",
        "premise_ids",
        "assumption_ids",
        "producer_id",
        "toolchain_id",
        "policy_id",
        "catalog_version",
        "evidence_ids",
        "required_assurance",
        "invalidation_selectors",
    ):
        assert key in payload

    kinds = {s.kind for s in claim.invalidation_selectors}
    assert InvalidationSelectorKind.REPOSITORY_TREE in kinds
    assert InvalidationSelectorKind.TOOLCHAIN in kinds
    assert InvalidationSelectorKind.POLICY in kinds
    assert InvalidationSelectorKind.CATALOG in kinds
    assert InvalidationSelectorKind.PROPERTY in kinds
    assert InvalidationSelectorKind.PREMISE_SET in kinds
    assert InvalidationSelectorKind.ASSUMPTION_SET in kinds


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_statuses_are_distinct() -> None:
    values = {status.value for status in ClaimStatus}
    assert values == {
        "unknown",
        "open",
        "satisfied",
        "refuted",
        "unsupported",
        "not_measured",
        "stale",
    }
    assert ClaimStatus.REFUTED.is_refutation
    assert not ClaimStatus.STALE.is_refutation
    assert not ClaimStatus.OPEN.is_refutation
    assert ClaimStatus.SATISFIED.terminal
    assert not ClaimStatus.OPEN.terminal


def test_cache_miss_is_never_refutation() -> None:
    claim = _open_claim()
    assert cache_miss_status(previously=ClaimStatus.UNKNOWN) is ClaimStatus.OPEN
    assert cache_miss_status(previously=ClaimStatus.SATISFIED) is ClaimStatus.OPEN
    assert cache_miss_status(previously=ClaimStatus.REFUTED) is ClaimStatus.OPEN
    assert cache_miss_status(previously=ClaimStatus.STALE) is ClaimStatus.STALE
    assert cache_miss_status(previously=ClaimStatus.UNSUPPORTED) is ClaimStatus.UNSUPPORTED

    missed = apply_cache_lookup(claim, outcome=CACHE_LOOKUP_MISS)
    assert missed.cache_lookup == CACHE_LOOKUP_MISS
    assert missed.status is ClaimStatus.OPEN
    assert missed.status is not ClaimStatus.REFUTED

    # Explicit construction of miss+refuted fails closed.
    with pytest.raises(CodeClaimContractError, match="cache miss"):
        claim.with_updates(
            status=ClaimStatus.REFUTED,
            cache_lookup=CACHE_LOOKUP_MISS,
        )


def test_stale_evidence_transitions_and_evaluation() -> None:
    claim = _open_claim()
    stale = mark_claim_stale(claim, reason_code="stale_or_unknown_evidence")
    assert stale.status is ClaimStatus.STALE
    assert stale.derived_assurance is AssuranceLevel.UNVERIFIED
    assert stale.status is not ClaimStatus.REFUTED

    # Tree drift invalidates.
    drifted = evaluate_invalidation(claim, current_tree_id="tree:other")
    assert drifted.status is ClaimStatus.STALE

    # Matching tree leaves claim open.
    same = evaluate_invalidation(claim, current_tree_id=claim.repository_tree_id)
    assert same.status is ClaimStatus.OPEN
    assert same.claim_id == claim.claim_id


def test_not_measured_and_unsupported_are_distinct_from_refuted() -> None:
    not_measured = _open_claim().with_updates(status=ClaimStatus.NOT_MEASURED)
    assert not_measured.status is ClaimStatus.NOT_MEASURED
    assert not not_measured.status.is_refutation

    unsupported = _open_claim(
        template_id="unsupported-proof-fail-closed",
        claim_family=ClaimFamily.UNSUPPORTED,
        property_id="property:unsupported-proof-fail-closed",
        obligation_id="obligation:unsupported-gate",
    ).with_updates(status=ClaimStatus.UNSUPPORTED)
    assert unsupported.status is ClaimStatus.UNSUPPORTED
    assert unsupported.status is not ClaimStatus.REFUTED


# ---------------------------------------------------------------------------
# Evidence tiers and assurance ceilings
# ---------------------------------------------------------------------------


def test_evidence_tiers_and_assurance_ceilings() -> None:
    assert EvidenceTier.QUERY_FACT.max_assurance is AssuranceLevel.UNVERIFIED
    assert EvidenceTier.GRAPHRAG_FACT.max_assurance is AssuranceLevel.UNVERIFIED
    assert EvidenceTier.OBSERVATION.max_assurance is AssuranceLevel.CANDIDATE
    assert EvidenceTier.SOLVER_CANDIDATE.max_assurance is AssuranceLevel.SOLVER_CHECKED
    assert EvidenceTier.KERNEL_PROOF.max_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert (
        EvidenceTier.CRYPTOGRAPHIC_ATTESTATION.max_assurance is AssuranceLevel.ATTESTED
    )

    assert not EvidenceTier.QUERY_FACT.can_mint_kernel_assurance
    assert not EvidenceTier.OBSERVATION.can_mint_kernel_assurance
    assert EvidenceTier.KERNEL_PROOF.can_mint_kernel_assurance

    assert max_assurance_for_tiers(
        (EvidenceTier.QUERY_FACT, EvidenceTier.OBSERVATION)
    ) is AssuranceLevel.CANDIDATE
    assert not tiers_can_independently_mint_kernel(
        (EvidenceTier.QUERY_FACT, EvidenceTier.OBSERVATION, EvidenceTier.SOLVER_CANDIDATE)
    )
    assert tiers_can_independently_mint_kernel((EvidenceTier.KERNEL_PROOF,))

    assert evidence_kind_to_tier(EvidenceKind.TEST_RESULT) is EvidenceTier.OBSERVATION
    assert evidence_kind_to_tier(EvidenceKind.SMT_CANDIDATE) is EvidenceTier.SOLVER_CANDIDATE
    assert evidence_kind_to_tier(EvidenceKind.KERNEL_VERIFICATION) is EvidenceTier.KERNEL_PROOF


def test_query_and_observation_cannot_mint_kernel_assurance() -> None:
    query = claim_from_query_fact(
        fact_id="fact:import-edge",
        repository_id="repository:sha256:demo",
        repository_tree_id="tree:abc123",
        property_id="property:dependency-reachability",
        claim_family=ClaimFamily.DEPENDENCY_REACHABILITY,
        graphrag=True,
    )
    assert query.evidence_tiers == (EvidenceTier.GRAPHRAG_FACT,)
    assert query.derived_assurance is AssuranceLevel.UNVERIFIED
    assert query.derived_assurance is not AssuranceLevel.KERNEL_VERIFIED
    assert not query.satisfies_required_assurance()

    with pytest.raises(CodeClaimContractError, match="cannot independently mint"):
        query.with_updates(derived_assurance=AssuranceLevel.KERNEL_VERIFIED)

    obs = claim_from_implementation_evidence(
        ImplementationResultEvidence(
            kind=ImplementationEvidenceKind.TEST,
            repository_tree_id="tree:abc123",
            repository_id="repository:sha256:demo",
            subject="test_lease_fence",
            passed=True,
            producer_id="pytest",
            scope_ids=("scope:mod.fn",),
        ),
        property_id="property:lease-uniqueness-and-fencing",
        claim_family=ClaimFamily.SECURITY_PROPERTY,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    assert obs.evidence_tiers == (EvidenceTier.OBSERVATION,)
    assert obs.derived_assurance is AssuranceLevel.CANDIDATE
    assert obs.status is ClaimStatus.OPEN
    with pytest.raises(CodeClaimContractError, match="cannot independently mint"):
        obs.with_updates(derived_assurance=AssuranceLevel.KERNEL_VERIFIED)


def test_natural_language_claims_fail_closed() -> None:
    with pytest.raises(CodeClaimContractError, match="natural-language"):
        CodeClaimRecord(
            claim_family=ClaimFamily.UNSUPPORTED,
            status=ClaimStatus.OPEN,
            statement="the code is probably fine under all conditions",
            metadata={"natural_language": True},
        )

    with pytest.raises(CodeClaimContractError, match="natural-language"):
        CodeClaimRecord(
            claim_family=ClaimFamily.UNSUPPORTED,
            status=ClaimStatus.UNKNOWN,
            statement="freeform prose without reviewed bindings",
        )

    with pytest.raises(CodeClaimContractError, match="natural-language"):
        reject_natural_language_claim("anything goes")

    # Reviewed property binding is admitted.
    ok = build_open_claim(
        property_id="property:dag-acyclicity",
        claim_family=ClaimFamily.BEHAVIORAL_INVARIANT,
        repository_id="repository:sha256:demo",
        repository_tree_id="tree:abc123",
        statement="dag remains acyclic after edge update",
    )
    assert ok.property_id == "property:dag-acyclicity"


# ---------------------------------------------------------------------------
# Adapters over existing contracts
# ---------------------------------------------------------------------------


def test_claim_from_obligation_and_kernel_receipt_satisfied() -> None:
    obligation = _obligation()
    open_claim = claim_from_obligation(
        obligation,
        property_id="property:lease-uniqueness-and-fencing",
        producer_id="producer:cbp",
        toolchain_id="toolchain:nix-lock-sha256",
        policy_id="policy:formal-v1",
    )
    assert open_claim.status is ClaimStatus.OPEN
    assert open_claim.obligation_id == obligation.obligation_id
    assert open_claim.claim_family is ClaimFamily.SECURITY_PROPERTY
    assert open_claim.repository_tree_id == obligation.repository_tree_id
    assert open_claim.premise_ids == obligation.premise_ids

    kernel = _kernel(obligation.obligation_id)
    receipt = _receipt(obligation, (kernel,))
    settled = claim_from_receipt(
        receipt,
        prior=open_claim,
        property_id=open_claim.property_id,
    )
    assert settled.status is ClaimStatus.SATISFIED
    assert settled.derived_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert settled.receipt_id == receipt.receipt_id
    assert EvidenceTier.KERNEL_PROOF in settled.evidence_tiers
    assert settled.satisfies_required_assurance()

    # Cache hit re-derives the same way.
    hit = apply_cache_lookup(open_claim, outcome=CACHE_LOOKUP_HIT, receipt=receipt)
    assert hit.cache_lookup == CACHE_LOOKUP_HIT
    assert hit.status is ClaimStatus.SATISFIED


def test_claim_from_receipt_refuted_on_counterexample() -> None:
    obligation = _obligation()
    open_claim = claim_from_obligation(
        obligation,
        property_id="property:lease-uniqueness-and-fencing",
    )
    cex = _solver_counterexample(obligation.obligation_id)
    receipt = _receipt(
        obligation,
        (cex,),
        verdict=ProofVerdict.DISPROVED,
    )
    refuted = claim_from_receipt(receipt, prior=open_claim)
    assert refuted.status is ClaimStatus.REFUTED
    assert refuted.status is not ClaimStatus.STALE
    assert refuted.status is not ClaimStatus.OPEN


def test_stale_receipt_projects_stale_status() -> None:
    obligation = _obligation()
    open_claim = claim_from_obligation(
        obligation,
        property_id="property:lease-uniqueness-and-fencing",
    )
    kernel = _kernel(obligation.obligation_id)
    # Stale kernel evidence item.
    stale_kernel = ProofEvidence(
        kind=EvidenceKind.KERNEL_VERIFICATION,
        authority=EvidenceAuthority.KERNEL,
        verdict=EvidenceVerdict.ACCEPTED,
        artifact_id="artifact:checked-lean-source",
        subject_id=obligation.obligation_id,
        verifier_id="kernel:lean-4.19",
        freshness=EvidenceFreshness.STALE,
        independent=True,
        simulated=False,
    )
    receipt = _receipt(
        obligation,
        (stale_kernel,),
        freshness=EvidenceFreshness.STALE,
    )
    stale_claim = claim_from_receipt(receipt, prior=open_claim)
    assert stale_claim.status is ClaimStatus.STALE
    assert stale_claim.derived_assurance is AssuranceLevel.UNVERIFIED
    # Presence of only stale kernel does not refute.
    assert stale_claim.status is not ClaimStatus.REFUTED
    _ = kernel  # silence linters if unused after edit paths


def test_claim_families_resolve_from_templates() -> None:
    assert (
        resolve_claim_family(template_id="lease-uniqueness-and-fencing")
        is ClaimFamily.SECURITY_PROPERTY
    )
    assert (
        resolve_claim_family(template_id="projection-equivalence")
        is ClaimFamily.SEMANTIC_EQUIVALENCE
    )
    assert (
        resolve_claim_family(template_id="merge-idempotence")
        is ClaimFamily.SUPERVISOR_LIFECYCLE
    )
    assert (
        resolve_claim_family(invariant_class="srt.structural")
        is ClaimFamily.SRT_STRUCTURAL
    )
    assert resolve_claim_family() is ClaimFamily.UNSUPPORTED


def test_build_invalidation_selectors_stable() -> None:
    a = build_invalidation_selectors(
        repository_tree_id="tree:1",
        scope_ids=("b", "a"),
        premise_ids=("p2", "p1"),
        toolchain_id="tc",
        policy_id="pol",
        catalog_version="1",
        property_id="property:x",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    b = build_invalidation_selectors(
        repository_tree_id="tree:1",
        scope_ids=("a", "b"),
        premise_ids=("p1", "p2"),
        toolchain_id="tc",
        policy_id="pol",
        catalog_version="1",
        property_id="property:x",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    assert a == b
    assert all(isinstance(s, InvalidationSelector) for s in a)


def test_interface_constant() -> None:
    assert CODE_CLAIM_RECORD_INTERFACE == "CodeClaimRecord@1"
    claim = _open_claim()
    assert claim.interface == CODE_CLAIM_RECORD_INTERFACE
    assert claim.to_dict()["interface"] == CODE_CLAIM_RECORD_INTERFACE


def test_all_claim_families_enumerable() -> None:
    families = {f.value for f in ClaimFamily}
    assert "dependency_reachability" in families
    assert "api_contract" in families
    assert "behavioral_invariant" in families
    assert "security_property" in families
    assert "semantic_equivalence" in families
    assert "supervisor_lifecycle" in families
    assert "srt_structural" in families
    assert "unsupported" in families


def test_query_fact_without_binding_fails_when_unsupported() -> None:
    with pytest.raises(CodeClaimContractError):
        claim_from_query_fact(
            fact_id="fact:x",
            repository_id="repository:sha256:demo",
            repository_tree_id="tree:abc123",
            claim_family=ClaimFamily.UNSUPPORTED,
        )
