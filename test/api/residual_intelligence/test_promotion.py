from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    RiskClass,
    TrainingAvailability,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.promotion import (
    AMORTIZATION_FIELDS,
    AUTONOMY_BOUNDS,
    EFFICIENCY_BOUNDS,
    HARD_GATES,
    ExpertPromotionGate,
    PromotionAction,
    PromotionAuthorization,
    PromotionEvidence,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.rights import (
    LeakageAudit,
    SourceRight,
    TrainingCorpusAdmission,
    TransformationRight,
)


def admitted_corpus() -> TrainingCorpusAdmission:
    audit = LeakageAudit(
        split_root="split:current",
        grouping_policy_id="groups:v1",
        train_group_count=4,
        development_group_count=1,
        holdout_group_count=2,
        adversarial_group_count=2,
        cross_partition_group_count=0,
        duplicate_example_count=0,
        hidden_test_commitment="hidden:commitment",
        passed=True,
    )
    source = "source:owned"
    return TrainingCorpusAdmission(
        source_identities=(source,),
        source_rights={source: SourceRight.FIRST_PARTY_OWNED.value},
        transformation_rights={source: TransformationRight.TRAINING_AND_DERIVATIVES_PERMITTED.value},
        privacy_classification=PrivacyClass.REPOSITORY_PRIVATE,
        tenant_scope="repository:one",
        data_retention_policy="retain:approved",
        corpus_root="corpus:current",
        split_root="split:current",
        holdout_roots=("holdout:current",),
        deduplication_policy="dedupe:v1",
        leakage_audit=audit,
        tokenizer_identity="tokenizer:v1",
        compiler_identity="compiler:v1",
        label_producers=("validator:v1",),
        negative_example_policy="negatives:v1",
        adversarial_partition="adversarial:current",
        environment="environment:v1",
        admission_decision=TrainingAvailability.ADMITTED,
        reason_codes=(),
    )


def evidence(
    *, risk: RiskClass = RiskClass.R3, accepted: int = 100,
    true_accepted: int = 100, critical: int = 0, **changes: object,
) -> PromotionEvidence:
    admission = admitted_corpus()
    values: dict[str, object] = {
        "gates": {name: True for name in HARD_GATES},
        "gate_evidence": {name: f"evidence:{name}:current" for name in HARD_GATES},
        "precision_ppm": (true_accepted * 1_000_000) // accepted,
        "accepted_count": accepted,
        "true_accepted_count": true_accepted,
        "critical_false_accepts": critical,
        "efficiency": dict(EFFICIENCY_BOUNDS),
        "autonomy": dict(AUTONOMY_BOUNDS),
        "amortization": {
            "training_evaluation_cost": 100,
            "per_use_saving": 10,
            "expected_break_even_uses": 10,
            "observed_uses": 10,
            "observed_savings": 100,
        },
        "risk": risk,
        "cas_identity": "cas:operator:1",
        "expert_identity": "expert:new",
        "admission": admission,
        "admission_id": admission.admission_id,
        "split_root": admission.split_root,
        "leakage_audit_id": admission.leakage_audit.audit_id,
    }
    values.update(changes)
    return PromotionEvidence(**values)  # type: ignore[arg-type]


def authorization(
    action: PromotionAction, *, subject: str, expected: str, generation: int,
    cas: str = "cas:operator:1",
) -> PromotionAuthorization:
    return PromotionAuthorization(
        authority_identity="operator:release",
        action=action,
        subject_identity=subject,
        expected_current_identity=expected,
        expected_generation=generation,
        cas_identity=cas,
    )


def test_conjunctive_gates_and_r4_remain_proposals() -> None:
    gate = ExpertPromotionGate()
    accepted = gate.decide(evidence())
    assert accepted.promoted is False  # eligibility itself cannot mutate a route
    assert accepted.reason_codes == ()
    r5 = gate.decide(evidence(risk=RiskClass.R5))
    assert "r4_r5_proposal_only" in r5.reason_codes
    failed = gate.decide(evidence(true_accepted=98, critical=1))
    assert "precision_below_99" in failed.reason_codes
    assert "critical_false_accept" in failed.reason_codes


def test_evidence_rejects_compensation_missing_denominators_and_unbound_lineage() -> None:
    with pytest.raises(ResidualIntelligenceError, match="exactly"):
        evidence(gates={name: True for name in HARD_GATES if name != "privacy"})
    with pytest.raises(ResidualIntelligenceError, match="independent evidence"):
        evidence(gate_evidence={name: "evidence:shared" for name in HARD_GATES})
    with pytest.raises(ResidualIntelligenceError, match="denominator"):
        evidence(amortization={
            "training_evaluation_cost": 100, "per_use_saving": 10,
            "expected_break_even_uses": 9, "observed_uses": 100, "observed_savings": 1_000,
        })
    with pytest.raises(ResidualIntelligenceError, match="split_root mismatch"):
        evidence(split_root="split:stale")
    assert AMORTIZATION_FIELDS == {
        "training_evaluation_cost", "per_use_saving", "expected_break_even_uses",
        "observed_uses", "observed_savings",
    }


def test_authorized_cas_promotion_rejects_stale_and_untrusted_writers() -> None:
    gate = ExpertPromotionGate(initial_identity="expert:old", trusted_authorities=("operator:release",))
    candidate = evidence()
    stale = gate.promote(candidate, authorization=authorization(
        PromotionAction.PROMOTE, subject=candidate.expert_identity, expected="expert:old", generation=99,
    ))
    assert stale.reason_codes == ("cas_generation_mismatch",)
    untrusted = gate.promote(candidate, authorization=PromotionAuthorization(
        authority_identity="operator:other", action=PromotionAction.PROMOTE,
        subject_identity=candidate.expert_identity, expected_current_identity="expert:old",
        expected_generation=0, cas_identity=candidate.cas_identity,
    ))
    assert untrusted.reason_codes == ("authorization_untrusted",)
    promoted = gate.promote(candidate, authorization=authorization(
        PromotionAction.PROMOTE, subject=candidate.expert_identity, expected="expert:old", generation=0,
    ))
    assert promoted.promoted is True
    assert promoted.previous_identity == "expert:old"
    assert gate.current_head().current_identity == "expert:new"


def test_exact_fenced_authorized_rollback_restores_only_prior_route() -> None:
    gate = ExpertPromotionGate(initial_identity="expert:old", trusted_authorities=("operator:release",))
    candidate = evidence()
    gate.promote(candidate, authorization=authorization(
        PromotionAction.PROMOTE, subject="expert:new", expected="expert:old", generation=0,
    ))
    rejected = gate.rollback(
        from_identity="expert:new", to_identity="expert:other", cas_identity="cas:operator:rollback", fence_drained=True,
    )
    assert rejected.reason_codes == ("rollback_target_not_exact_prior",)
    undrained = gate.rollback(
        from_identity="expert:new", to_identity="expert:old", cas_identity="cas:operator:rollback",
        authorization=authorization(PromotionAction.ROLLBACK, subject="expert:new", expected="expert:new", generation=1, cas="cas:operator:rollback"),
    )
    assert undrained.reason_codes == ("fenced_work_not_drained",)
    rollback = gate.rollback(
        from_identity="expert:new", to_identity="expert:old", cas_identity="cas:operator:rollback",
        authorization=authorization(PromotionAction.ROLLBACK, subject="expert:new", expected="expert:new", generation=1, cas="cas:operator:rollback"),
        fence_drained=True,
    )
    assert rollback.rolled_back is True
    assert rollback.to_dict()["promoted"] is False
    assert gate.current_head().current_identity == "expert:old"
