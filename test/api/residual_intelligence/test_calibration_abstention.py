from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.abstention import (
    CLOSED_DISPOSITIONS,
    AbstentionDecision,
    SelectivePredictionPolicy,
    SelectivePredictionRequest,
    selectively_predict,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.calibration import (
    CALIBRATION_GROUP_AXES,
    CalibrationEvidence,
    CalibrationGroup,
    CalibrationThresholdBinding,
    ThresholdChangeOrigin,
    apply_threshold_cas,
    rollback_threshold_binding,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.splits import SplitPartition

from .helpers import admission

ADMISSION_ID = "admission:fixture-current"
SPLIT_ROOT = "split:fixture-current"
HOLDOUT_ROOT = "holdout:fixture"
EVALUATION_ID = "evaluation:fixture-current"
THRESHOLD = 800_000


def group(**overrides: object) -> CalibrationGroup:
    payload: dict[str, object] = {
        "family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "repository": "ipfs_accelerate_py",
        "language": "python",
        "framework": "pytest",
        "risk": RiskClass.R2,
        "model": "fixture-linear@1",
        "quantization": "none",
        "hardware": "cpu-standard",
        "context_tier": "evidence",
    }
    payload.update(overrides)
    return CalibrationGroup(**payload)  # type: ignore[arg-type]


def evidence(
    target: CalibrationGroup | None = None,
    *,
    n_examples: int = 10,
    accept_count: int = 8,
    abstain_count: int = 2,
    reject_input_count: int = 0,
    ood_count: int = 0,
    capability_unavailable_count: int = 0,
    validation_required_count: int = 0,
    false_accept_count: int = 0,
    critical_false_accept_count: int = 0,
    adversarial: bool = True,
    thresholds: tuple[int, ...] = (THRESHOLD,),
    **overrides: object,
) -> CalibrationEvidence:
    total = (
        accept_count
        + abstain_count
        + reject_input_count
        + ood_count
        + capability_unavailable_count
        + validation_required_count
    )
    if n_examples != total:
        n_examples = total
    precision = (
        0
        if accept_count == 0
        else ((accept_count - false_accept_count) * 1_000_000) // accept_count
    )
    payload: dict[str, object] = {
        "group": target or group(),
        "admission_id": ADMISSION_ID,
        "admission_decision": TrainingAvailability.ADMITTED,
        "split_root": SPLIT_ROOT,
        "holdout_root": HOLDOUT_ROOT,
        "evaluation_identity": EVALUATION_ID,
        "example_identities": tuple(f"example:{index}" for index in range(n_examples)),
        "adversarial_example_identities": (("adversarial:1",) if adversarial else ()),
        "evaluated_threshold_candidates": thresholds,
        "accept_count": accept_count,
        "abstain_count": abstain_count,
        "reject_input_count": reject_input_count,
        "ood_count": ood_count,
        "capability_unavailable_count": capability_unavailable_count,
        "validation_required_count": validation_required_count,
        "false_accept_count": false_accept_count,
        "critical_false_accept_count": critical_false_accept_count,
        "precision_ppm": precision,
        "abstention_rate_ppm": (abstain_count * 1_000_000) // n_examples,
    }
    payload.update(overrides)
    return CalibrationEvidence(**payload)  # type: ignore[arg-type]


def binding(
    target: CalibrationGroup | None = None,
    *,
    record: CalibrationEvidence | None = None,
    accept_threshold_ppm: int = THRESHOLD,
    previous_binding_id: str = "",
    rollback_threshold_ppm: int = THRESHOLD,
    cas_identity: str = "cas:operator:fixture",
) -> CalibrationThresholdBinding:
    item = record or evidence(target)
    subject = target or item.group
    return CalibrationThresholdBinding(
        group_key=subject.group_key,
        accept_threshold_ppm=accept_threshold_ppm,
        evidence_id=item.evidence_id,
        cas_identity=cas_identity,
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        previous_binding_id=previous_binding_id,
        rollback_threshold_ppm=rollback_threshold_ppm,
    )


def policy(
    *records: CalibrationEvidence,
    bindings: tuple[CalibrationThresholdBinding, ...] | None = None,
    ood_signals_binding: bool = False,
    **overrides: object,
) -> SelectivePredictionPolicy:
    held = records or (evidence(),)
    payload: dict[str, object] = {
        "current_admission_id": ADMISSION_ID,
        "current_split_root": SPLIT_ROOT,
        "current_holdout_root": HOLDOUT_ROOT,
        "current_evaluation_identity": EVALUATION_ID,
        "evidence": held,
        "bindings": (
            bindings
            if bindings is not None
            else tuple(binding(item.group, record=item) for item in held)
        ),
        "ood_signals_binding": ood_signals_binding,
    }
    payload.update(overrides)
    return SelectivePredictionPolicy(**payload)  # type: ignore[arg-type]


def request_for(
    target: CalibrationGroup | None = None, **overrides: object
) -> SelectivePredictionRequest:
    payload: dict[str, object] = {"group": target or group(), "score_ppm": 900_000}
    payload.update(overrides)
    return SelectivePredictionRequest(**payload)  # type: ignore[arg-type]


def test_group_key_covers_all_nine_axes_and_isolates_groups() -> None:
    assert CALIBRATION_GROUP_AXES == (
        "family",
        "repository",
        "language",
        "framework",
        "risk",
        "model",
        "quantization",
        "hardware",
        "context_tier",
    )
    baseline = group()
    rebuilt = CalibrationGroup.from_dict(baseline.to_dict())
    assert rebuilt == baseline
    assert rebuilt.group_key == baseline.group_key
    assert set(baseline.axis_payload()) == set(CALIBRATION_GROUP_AXES)
    neighbors = (
        group(family=ResidualTaskFamily.TEST_SELECTION),
        group(repository="other-repo"),
        group(language="rust"),
        group(framework="cargo"),
        group(risk=RiskClass.R3),
        group(model="other-model@1"),
        group(quantization="int8"),
        group(hardware="gpu-live"),
        group(context_tier="invariant"),
    )
    keys = {baseline.group_key, *(item.group_key for item in neighbors)}
    assert len(keys) == 10


def test_current_held_out_evidence_round_trip_and_admission_binding() -> None:
    record, _examples = admission()
    held = evidence(
        admission_id=record.admission_id,
        split_root=record.split_root,
        holdout_root=record.holdout_roots[0],
    )
    held.validate_against_admission(record)
    rebuilt = CalibrationEvidence.from_dict(held.to_dict())
    assert rebuilt == held
    assert rebuilt.partition is SplitPartition.HELD_OUT
    assert rebuilt.is_current(
        admission_id=record.admission_id,
        split_root=record.split_root,
        evaluation_identity=EVALUATION_ID,
        holdout_root=record.holdout_roots[0],
    )


def test_calibration_rows_reject_non_held_out_and_unadmitted_sources() -> None:
    with pytest.raises(ResidualIntelligenceError, match="held-out"):
        evidence(partition=SplitPartition.TRAIN)
    with pytest.raises(ResidualIntelligenceError, match="admitted"):
        evidence(admission_decision=TrainingAvailability.TRAINING_UNAVAILABLE)
    record, _examples = admission(admitted=False)
    with pytest.raises(ResidualIntelligenceError, match="admitted"):
        evidence().validate_against_admission(record)


def test_calibration_evidence_rejects_private_bodies_and_hidden_tests() -> None:
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        evidence(example_identities=("example:1", "hidden_test_body:secret"))
    with pytest.raises(ResidualIntelligenceError, match="hidden-test"):
        evidence(hidden_test_bodies_accessed=True)


def test_closed_dispositions_cover_accept_abstain_reject_ood_capability_validation() -> None:
    assert tuple(item.value for item in CLOSED_DISPOSITIONS) == (
        "ACCEPT",
        "ABSTAIN",
        "REJECT_INPUT",
        "OUT_OF_DISTRIBUTION",
        "CAPABILITY_UNAVAILABLE",
        "VALIDATION_REQUIRED",
    )
    target = group()
    record = evidence(target)
    current = policy(record, ood_signals_binding=True)
    cases = (
        (request_for(target, score_ppm=900_000), ExpertDisposition.ACCEPT),
        (request_for(target, score_ppm=100_000), ExpertDisposition.ABSTAIN),
        (request_for(target, input_valid=False), ExpertDisposition.REJECT_INPUT),
        (request_for(target, out_of_distribution=True), ExpertDisposition.OUT_OF_DISTRIBUTION),
        (
            request_for(target, capability_available=False),
            ExpertDisposition.CAPABILITY_UNAVAILABLE,
        ),
        (
            request_for(target, validation_satisfied=False),
            ExpertDisposition.VALIDATION_REQUIRED,
        ),
    )
    observed = []
    for item, expected in cases:
        decision = selectively_predict(current, item)
        assert decision.disposition is expected
        assert decision.candidate_only is True
        observed.append(decision.disposition)
    missing = group(hardware="tpu-unavailable")
    absent = selectively_predict(current, request_for(missing))
    assert absent.disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
    assert absent.reason_codes == ("missing_calibration_group",)
    assert set(observed) == set(CLOSED_DISPOSITIONS)


def test_no_global_threshold_and_group_isolation() -> None:
    first = group()
    second = group(repository="other-repo")
    low = evidence(first, thresholds=(700_000, 900_000))
    high = evidence(second, thresholds=(700_000, 900_000))
    current = policy(
        low,
        high,
        bindings=(
            binding(first, record=low, accept_threshold_ppm=700_000),
            binding(second, record=high, accept_threshold_ppm=900_000),
        ),
    )
    score = request_for(first, score_ppm=800_000)
    assert selectively_predict(current, score).disposition is ExpertDisposition.ACCEPT
    isolated = request_for(second, score_ppm=800_000)
    assert selectively_predict(current, isolated).disposition is ExpertDisposition.ABSTAIN
    payload = current.to_dict()
    payload["global_threshold_ppm"] = 500_000
    with pytest.raises(UnknownFieldError, match="global threshold"):
        SelectivePredictionPolicy.from_dict(payload)


def test_stale_evidence_cannot_accept() -> None:
    record = evidence()
    current = policy(record, current_split_root="split:stale-other")
    decision = selectively_predict(current, request_for(score_ppm=999_000))
    assert decision.disposition is ExpertDisposition.ABSTAIN
    assert "current_evidence_required" in decision.reason_codes


def test_r4_r5_remain_proposal_tier_regardless_of_score() -> None:
    for risk in (RiskClass.R4, RiskClass.R5):
        target = group(risk=risk)
        record = evidence(target)
        current = policy(record)
        decision = selectively_predict(current, request_for(target, score_ppm=1_000_000))
        assert decision.disposition is ExpertDisposition.VALIDATION_REQUIRED
        assert decision.proposal_tier is True
        assert decision.abstained is False
        assert "r4_r5_proposal_tier" in decision.reason_codes
        with pytest.raises(ResidualIntelligenceError, match="R4/R5"):
            AbstentionDecision(
                disposition=ExpertDisposition.ACCEPT,
                group_key=target.group_key,
                risk_class=risk,
                score_ppm=1_000_000,
                group_threshold_bound=True,
                group_threshold_ppm=THRESHOLD,
                reason_codes=("group_threshold_met",),
                evidence_id=record.evidence_id,
            )


def test_self_threshold_rejection_and_authorized_cas_rollback() -> None:
    target = group()
    record = evidence(target, thresholds=(THRESHOLD, 600_000))
    current = policy(record)
    with pytest.raises(ResidualIntelligenceError, match="self-threshold"):
        SelectivePredictionRequest(
            group=target,
            score_ppm=900_000,
            model_proposed_threshold_ppm=100_000,
        )
    with pytest.raises(ResidualIntelligenceError, match="self-threshold"):
        current.apply_threshold_cas(
            group=target,
            proposed_threshold_ppm=600_000,
            origin=ThresholdChangeOrigin.MODEL_SELF,
            cas_identity="cas:model",
            expected_binding_id=current.bindings[0].binding_id,
        )
    with pytest.raises(ResidualIntelligenceError, match="self-threshold"):
        CalibrationThresholdBinding(
            group_key=target.group_key,
            accept_threshold_ppm=100_000,
            evidence_id=record.evidence_id,
            cas_identity="cas:model",
            origin=ThresholdChangeOrigin.MODEL_SELF,
        )
    updated = current.apply_threshold_cas(
        group=target,
        proposed_threshold_ppm=600_000,
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        cas_identity="cas:operator:update",
        expected_binding_id=current.bindings[0].binding_id,
    )
    assert updated.bindings[0].accept_threshold_ppm == 600_000
    assert updated.bindings[0].rollback_threshold_ppm == THRESHOLD
    with pytest.raises(ResidualIntelligenceError, match="compare-and-swap"):
        updated.apply_threshold_cas(
            group=target,
            proposed_threshold_ppm=THRESHOLD,
            origin=ThresholdChangeOrigin.OPERATOR_CAS,
            cas_identity="cas:operator:stale",
            expected_binding_id=current.bindings[0].binding_id,
        )
    restored = updated.rollback_threshold(group=target, cas_identity="cas:operator:rollback")
    assert restored.bindings[0].accept_threshold_ppm == THRESHOLD
    genesis = apply_threshold_cas(
        None,
        group=target,
        evidence=record,
        proposed_threshold_ppm=THRESHOLD,
        origin=ThresholdChangeOrigin.OPERATOR_CAS,
        cas_identity="cas:operator:genesis",
        admission_id=ADMISSION_ID,
        split_root=SPLIT_ROOT,
        evaluation_identity=EVALUATION_ID,
        holdout_root=HOLDOUT_ROOT,
    )
    with pytest.raises(ResidualIntelligenceError, match="genesis"):
        rollback_threshold_binding(genesis, cas_identity="cas:operator:rollback")


def test_threshold_cas_requires_adversarial_evaluation_and_zero_critical_false_accepts() -> None:
    target = group()
    no_adversarial = evidence(target, adversarial=False, thresholds=(THRESHOLD, 500_000))
    current = policy(no_adversarial)
    with pytest.raises(ResidualIntelligenceError, match="adversarial"):
        current.apply_threshold_cas(
            group=target,
            proposed_threshold_ppm=500_000,
            origin=ThresholdChangeOrigin.OPERATOR_CAS,
            cas_identity="cas:operator",
            expected_binding_id=current.bindings[0].binding_id,
        )
    poisoned = evidence(
        target,
        accept_count=8,
        false_accept_count=1,
        critical_false_accept_count=1,
        thresholds=(THRESHOLD, 500_000),
    )
    blocked = policy(poisoned)
    decision = selectively_predict(blocked, request_for(target, score_ppm=999_000))
    assert decision.disposition is ExpertDisposition.ABSTAIN
    assert "critical_false_accept" in decision.reason_codes
    with pytest.raises(ResidualIntelligenceError, match="critical false accepts"):
        blocked.apply_threshold_cas(
            group=target,
            proposed_threshold_ppm=500_000,
            origin=ThresholdChangeOrigin.OPERATOR_CAS,
            cas_identity="cas:operator",
            expected_binding_id=blocked.bindings[0].binding_id,
        )


def test_critical_boundary_and_missing_group_threshold_abstain() -> None:
    target = group()
    record = evidence(target)
    current = policy(record, bindings=())
    assert (
        selectively_predict(current, request_for(target)).disposition is ExpertDisposition.ABSTAIN
    )
    bound = policy(record)
    critical = selectively_predict(
        bound, request_for(target, score_ppm=999_000, critical_boundary=True)
    )
    assert critical.disposition is ExpertDisposition.ABSTAIN
    assert "critical_boundary_abstention" in critical.reason_codes


def test_policy_and_decision_round_trip_and_unknown_fields() -> None:
    current = policy()
    rebuilt = SelectivePredictionPolicy.from_dict(current.to_dict())
    assert rebuilt == current
    decision = current.decide(request_for())
    assert AbstentionDecision.from_dict(decision.to_dict()) == decision
    payload = decision.to_dict()
    payload["promotion"] = True
    with pytest.raises(UnknownFieldError):
        AbstentionDecision.from_dict(payload)
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        AbstentionDecision(
            disposition=ExpertDisposition.ABSTAIN,
            group_key=group().group_key,
            risk_class=RiskClass.R2,
            score_ppm=1,
            group_threshold_bound=True,
            group_threshold_ppm=THRESHOLD,
            reason_codes=("below_group_threshold",),
            candidate_only=False,
        )
