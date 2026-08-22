from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.calibration import (
    CalibrationGroup,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.inventory import (
    ResidualFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.ood import (
    BOUNDARY_AXES,
    CANDIDATE_ONLY_AUTHORITY,
    REASON_ADVISORY_ONLY,
    REASON_CALIBRATION_ABSENCE,
    REASON_CONTEXT_INCOMPLETE,
    REASON_DISAGREEMENT,
    REASON_FEATURE_RANGE,
    REASON_IN_BOUNDARY,
    REASON_MISSING_DETECTION,
    REASON_SAFETY_NOT_ESTABLISHED,
    REASON_UNKNOWN_OPERATION,
    REASON_UNKNOWN_REPOSITORY,
    REASON_UNKNOWN_SCHEMA,
    REASON_UNSEEN_AUTHORITY,
    REASON_UNSEEN_EFFECT,
    BoundaryAxis,
    BoundaryContract,
    FeatureRange,
    OODAssessment,
    OODObservation,
    OODSignal,
    OODSignalKind,
    ReferenceDistribution,
    assess_out_of_distribution,
    observation_from_task_input,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import (
    ResidualTaskInput,
)

from .helpers import admission

SCHEMA = "failure-signature@1"
OPERATION = "classify_failure"
REPOSITORY = "ipfs_accelerate_py"
EFFECT = "candidate_failure_class"
CAPABILITY = "cpu-standard"
CONTEXT_FIELD = "context:evidence"
EXIT_RANGE = FeatureRange(name="exit_code", minimum=0, maximum=2, observed_count=10)
SIGNATURE_RANGE = FeatureRange(
    name="failure_signature",
    allowed_values=("missing-edge",),
    observed_count=10,
)


def group(**overrides: object) -> CalibrationGroup:
    payload: dict[str, object] = {
        "family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "repository": REPOSITORY,
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


def family_boundary() -> ResidualFamilyBoundary:
    return ResidualFamilyBoundary(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        input_semantics="validated failure signature plus bounded dependency references",
        output_semantics="one failure class and one bounded action candidate",
        risk_class=RiskClass.R2,
        authority_class=CANDIDATE_ONLY_AUTHORITY,
        validation_contract="failure-attribution-validator@1",
        error_behavior="invalid output or failed validation escalates",
        abstention_behavior="unknown signatures abstain",
    )


def reference(
    target: CalibrationGroup | None = None,
    **overrides: object,
) -> ReferenceDistribution:
    record, _examples = admission()
    subject = target or group()
    payload: dict[str, object] = {
        "group": subject,
        "admission_id": record.admission_id,
        "admission_decision": TrainingAvailability.ADMITTED,
        "allowed_families": (ResidualTaskFamily.FAILURE_ATTRIBUTION,),
        "allowed_schemas": (SCHEMA,),
        "allowed_operations": (OPERATION,),
        "allowed_repositories": (REPOSITORY,),
        "allowed_effects": (EFFECT,),
        "allowed_authorities": (CANDIDATE_ONLY_AUTHORITY,),
        "allowed_capabilities": (CAPABILITY,),
        "required_context_fields": (CONTEXT_FIELD,),
        "feature_ranges": (EXIT_RANGE, SIGNATURE_RANGE),
        "example_identities": ("example:1", "example:2"),
        "statistic_identities": ("stat:exit_code", "stat:failure_signature"),
        "compact_statistics": {"n_examples": 2, "feature_count": 2},
        "family_distance_threshold_ppm": 0,
    }
    payload.update(overrides)
    return ReferenceDistribution(**payload)  # type: ignore[arg-type]


def contract(
    target: CalibrationGroup | None = None,
    **overrides: object,
) -> BoundaryContract:
    subject = target or group()
    payload: dict[str, object] = {
        "family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "schema": SCHEMA,
        "effects": (EFFECT,),
        "authority_class": CANDIDATE_ONLY_AUTHORITY,
        "repository": REPOSITORY,
        "calibration_group_key": subject.group_key,
        "capabilities": (CAPABILITY,),
        "required_context_fields": (CONTEXT_FIELD,),
        "risk_ceiling": subject.risk,
    }
    payload.update(overrides)
    return BoundaryContract(**payload)  # type: ignore[arg-type]


def observation(
    target: CalibrationGroup | None = None,
    **overrides: object,
) -> OODObservation:
    subject = target or group()
    payload: dict[str, object] = {
        "risk_class": subject.risk,
        "family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "schema": SCHEMA,
        "operation": OPERATION,
        "repository": REPOSITORY,
        "effects": (EFFECT,),
        "authority_class": CANDIDATE_ONLY_AUTHORITY,
        "features": {"exit_code": 1, "failure_signature": "missing-edge"},
        "calibration_group_key": subject.group_key,
        "context_fields": (CONTEXT_FIELD,),
        "capabilities": (CAPABILITY,),
        "capability_available": True,
        "disagreement": False,
        "family_distance_ppm": 0,
        "detection_available": True,
        "context_complete": True,
    }
    payload.update(overrides)
    return OODObservation(**payload)  # type: ignore[arg-type]


def assess(
    item: OODObservation | None = None,
    *,
    target: CalibrationGroup | None = None,
    policy_admits_ood: bool = False,
    **overrides: object,
) -> OODAssessment:
    subject = target or group()
    record, _examples = admission()
    return assess_out_of_distribution(
        item or observation(subject, **overrides),
        reference=reference(subject),
        boundary=contract(subject),
        admission=record,
        policy_admits_ood=policy_admits_ood,
    )


def kinds(result: OODAssessment) -> set[OODSignalKind]:
    return {item.kind for item in result.signals}


def failed_axes(result: OODAssessment) -> set[BoundaryAxis]:
    return {item.axis for item in result.boundary_findings if not item.passed}


def test_known_in_boundary_fixtures_remain_eligible() -> None:
    result = assess()
    assert result.in_boundary_eligible is True
    assert result.advisory_ood is False
    assert result.conservative_abstain is False
    assert result.bound_ood is False
    assert result.safety_established is False
    assert result.forced_disposition is None
    assert result.candidate_only is True
    assert result.reason_codes == (REASON_IN_BOUNDARY,)
    assert tuple(item.axis for item in result.boundary_findings) == BOUNDARY_AXES
    assert all(item.passed for item in result.boundary_findings)
    rebuilt = OODAssessment.from_dict(result.to_dict())
    assert rebuilt == result


def test_feature_range_outside_reference_is_advisory_ood() -> None:
    result = assess(features={"exit_code": 99, "failure_signature": "missing-edge"})
    assert result.in_boundary_eligible is False
    assert result.advisory_ood is True
    assert result.bound_ood is False
    assert result.conservative_abstain is False
    assert result.forced_disposition is None
    assert REASON_FEATURE_RANGE in result.reason_codes
    assert REASON_ADVISORY_ONLY in result.reason_codes
    assert OODSignalKind.FEATURE_RANGE in kinds(result)
    bound = assess(
        features={"exit_code": 99, "failure_signature": "missing-edge"},
        policy_admits_ood=True,
    )
    assert bound.bound_ood is True
    assert bound.forced_disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
    unknown_feature = assess(features={"exit_code": 1, "novel_metric": 4})
    assert OODSignalKind.FEATURE_RANGE in kinds(unknown_feature)
    token = assess(features={"exit_code": 1, "failure_signature": "unseen-token"})
    assert OODSignalKind.FEATURE_RANGE in kinds(token)


def test_unknown_schema_operation_repository() -> None:
    result = assess(
        schema="totally-unknown-schema@1",
        operation="novel_unbounded_rewrite",
        repository="unseen-repository",
    )
    assert result.in_boundary_eligible is False
    assert failed_axes(result) >= {BoundaryAxis.SCHEMA, BoundaryAxis.REPOSITORY}
    assert result.finding(BoundaryAxis.SCHEMA).unknown_or_missing is True
    assert result.finding(BoundaryAxis.REPOSITORY).unknown_or_missing is True
    assert kinds(result) >= {
        OODSignalKind.UNKNOWN_SCHEMA,
        OODSignalKind.UNKNOWN_OPERATION,
        OODSignalKind.UNKNOWN_REPOSITORY,
    }
    assert REASON_UNKNOWN_SCHEMA in result.reason_codes
    assert REASON_UNKNOWN_OPERATION in result.reason_codes
    assert REASON_UNKNOWN_REPOSITORY in result.reason_codes
    assert result.conservative_abstain is False
    high = assess(
        observation(
            group(risk=RiskClass.R4),
            risk_class=RiskClass.R4,
            schema="totally-unknown-schema@1",
            operation="novel_unbounded_rewrite",
            repository="unseen-repository",
        ),
        target=group(risk=RiskClass.R4),
    )
    assert high.conservative_abstain is True
    assert high.forced_disposition is ExpertDisposition.ABSTAIN


def test_unseen_effects_authority() -> None:
    result = assess(
        effects=("unseen_promote_checkpoint",),
        authority_class="authorized",
    )
    assert failed_axes(result) >= {BoundaryAxis.EFFECT, BoundaryAxis.AUTHORITY}
    assert kinds(result) >= {OODSignalKind.UNSEEN_EFFECT, OODSignalKind.UNSEEN_AUTHORITY}
    assert REASON_UNSEEN_EFFECT in result.reason_codes
    assert REASON_UNSEEN_AUTHORITY in result.reason_codes
    assert result.finding(BoundaryAxis.AUTHORITY).conservative_abstain is True
    assert result.conservative_abstain is True
    assert result.forced_disposition is ExpertDisposition.ABSTAIN
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        contract(authority_class="authorized")
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        reference(allowed_authorities=("authorized",))


def test_disagreement_is_advisory_ood() -> None:
    result = assess(
        disagreement=True,
        disagreement_identities=("expert:local", "expert:remote-shadow"),
    )
    assert result.advisory_ood is True
    assert result.in_boundary_eligible is False
    assert result.conservative_abstain is False
    assert result.forced_disposition is None
    assert OODSignalKind.DISAGREEMENT in kinds(result)
    assert REASON_DISAGREEMENT in result.reason_codes
    assert all(item.passed for item in result.boundary_findings)
    bound = assess(
        disagreement=True,
        disagreement_identities=("expert:local", "expert:remote-shadow"),
        policy_admits_ood=True,
    )
    assert bound.forced_disposition is ExpertDisposition.OUT_OF_DISTRIBUTION


def test_calibration_absence() -> None:
    result = assess(calibration_group_key="")
    assert result.finding(BoundaryAxis.CALIBRATION).passed is False
    assert result.finding(BoundaryAxis.CALIBRATION).unknown_or_missing is True
    assert OODSignalKind.CALIBRATION_ABSENCE in kinds(result)
    assert REASON_CALIBRATION_ABSENCE in result.reason_codes
    assert result.in_boundary_eligible is False
    assert result.conservative_abstain is False
    high = assess(
        observation(
            group(risk=RiskClass.R4),
            risk_class=RiskClass.R4,
            calibration_group_key="",
        ),
        target=group(risk=RiskClass.R4),
    )
    assert high.conservative_abstain is True
    assert high.finding(BoundaryAxis.CALIBRATION).conservative_abstain is True
    assert high.forced_disposition is ExpertDisposition.ABSTAIN
    assert "high_risk_missing_group" in high.reason_codes


def test_context_incomplete() -> None:
    result = assess(context_fields=(), context_complete=False)
    assert result.finding(BoundaryAxis.CONTEXT).passed is False
    assert result.finding(BoundaryAxis.CONTEXT).unknown_or_missing is True
    assert OODSignalKind.CONTEXT_INCOMPLETE in kinds(result)
    assert REASON_CONTEXT_INCOMPLETE in result.reason_codes
    assert result.conservative_abstain is False
    high = assess(
        observation(
            group(risk=RiskClass.R5),
            risk_class=RiskClass.R5,
            context_fields=(),
            context_complete=False,
        ),
        target=group(risk=RiskClass.R5),
    )
    assert high.conservative_abstain is True
    assert high.finding(BoundaryAxis.CONTEXT).conservative_abstain is True
    assert high.forced_disposition is ExpertDisposition.ABSTAIN
    assert "high_risk_incomplete_context" in high.reason_codes


def test_conservative_high_risk_unknown_or_missing_independently_abstains() -> None:
    target = group(risk=RiskClass.R4)
    result = assess(
        observation(
            target,
            risk_class=RiskClass.R4,
            family=None,
            calibration_group_key="",
            context_fields=(),
            context_complete=False,
        ),
        target=target,
        policy_admits_ood=False,
    )
    assert result.policy_admits_ood is False
    assert result.bound_ood is False
    assert result.conservative_abstain is True
    assert result.forced_disposition is ExpertDisposition.ABSTAIN
    assert failed_axes(result) >= {
        BoundaryAxis.FAMILY,
        BoundaryAxis.CALIBRATION,
        BoundaryAxis.CONTEXT,
    }
    assert result.finding(BoundaryAxis.FAMILY).conservative_abstain is True
    assert result.finding(BoundaryAxis.CALIBRATION).conservative_abstain is True
    assert result.finding(BoundaryAxis.CONTEXT).conservative_abstain is True
    low = assess(family=None, calibration_group_key="", context_complete=False, context_fields=())
    assert low.conservative_abstain is False
    assert low.forced_disposition is None
    assert low.advisory_ood is True


def test_missing_ood_detection_never_establishes_safety() -> None:
    record, _examples = admission()
    target = group()
    missing = assess_out_of_distribution(
        observation(target, detection_available=False),
        reference=reference(target),
        boundary=contract(target),
        admission=record,
    )
    assert missing.detection_available is False
    assert missing.safety_established is False
    assert missing.in_boundary_eligible is False
    assert OODSignalKind.MISSING_DETECTION in kinds(missing)
    assert REASON_MISSING_DETECTION in missing.reason_codes
    assert REASON_SAFETY_NOT_ESTABLISHED in missing.reason_codes
    assert missing.forced_disposition is None
    high = assess_out_of_distribution(
        observation(group(risk=RiskClass.R4), risk_class=RiskClass.R4, detection_available=False),
        reference=reference(group(risk=RiskClass.R4)),
        boundary=contract(group(risk=RiskClass.R4)),
        admission=record,
    )
    assert high.safety_established is False
    assert high.conservative_abstain is True
    assert high.forced_disposition is ExpertDisposition.ABSTAIN
    absent = assess_out_of_distribution(observation(target, risk_class=RiskClass.R2))
    assert absent.safety_established is False
    assert absent.in_boundary_eligible is False
    assert OODSignalKind.MISSING_DETECTION in kinds(absent)
    eligible = assess()
    assert eligible.in_boundary_eligible is True
    assert eligible.safety_established is False


def test_boundary_checks_are_independent_and_never_short_circuit() -> None:
    result = assess(
        family=None,
        schema="unknown-schema@1",
        effects=("unseen_effect",),
        authority_class="promotion",
        repository="other-repo",
        calibration_group_key="",
        capabilities=("tpu-unavailable",),
        capability_available=True,
        context_fields=(),
        context_complete=False,
        operation="novel_op",
        features={"exit_code": 99},
        disagreement=True,
        disagreement_identities=("a", "b"),
    )
    assert tuple(item.axis for item in result.boundary_findings) == BOUNDARY_AXES
    assert failed_axes(result) == set(BOUNDARY_AXES)
    assert kinds(result) >= {
        OODSignalKind.FEATURE_RANGE,
        OODSignalKind.UNKNOWN_SCHEMA,
        OODSignalKind.UNKNOWN_OPERATION,
        OODSignalKind.UNKNOWN_REPOSITORY,
        OODSignalKind.UNSEEN_EFFECT,
        OODSignalKind.UNSEEN_AUTHORITY,
        OODSignalKind.DISAGREEMENT,
        OODSignalKind.CALIBRATION_ABSENCE,
        OODSignalKind.CONTEXT_INCOMPLETE,
        OODSignalKind.CAPABILITY_UNAVAILABLE,
        OODSignalKind.FAMILY_BOUNDARY,
    }


def test_ood_is_advisory_unless_policy_admits() -> None:
    advisory = assess(features={"exit_code": 99, "failure_signature": "missing-edge"})
    assert advisory.advisory_ood is True
    assert advisory.policy_admits_ood is False
    assert advisory.bound_ood is False
    assert advisory.forced_disposition is None
    admitted = assess(
        features={"exit_code": 99, "failure_signature": "missing-edge"},
        policy_admits_ood=True,
    )
    assert admitted.bound_ood is True
    assert admitted.forced_disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
    with pytest.raises(ResidualIntelligenceError, match="advisory"):
        OODSignal(
            kind=OODSignalKind.FEATURE_RANGE,
            reason_code=REASON_FEATURE_RANGE,
            advisory=False,
        )


def test_capability_unavailable_forces_closed_disposition() -> None:
    result = assess(capability_available=False, capabilities=())
    assert result.forced_disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    assert result.finding(BoundaryAxis.CAPABILITY).passed is False
    assert result.in_boundary_eligible is False


def test_reference_distribution_requires_admitted_corpus() -> None:
    record, _examples = admission(admitted=False)
    with pytest.raises(ResidualIntelligenceError, match="admitted"):
        reference(admission_id=record.admission_id, admission_decision=record.admission_decision)
    admitted, _examples = admission()
    current = reference(admission_id=admitted.admission_id)
    current.validate_against_admission(admitted)
    with pytest.raises(ResidualIntelligenceError, match="admitted"):
        current.validate_against_admission(record)


def test_compact_statistics_reject_private_source() -> None:
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        reference(example_identities=("example:1", "source_text:recoverable"))
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        reference(statistic_identities=("hidden_test_body:secret",))
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        reference(compact_statistics={"source_text": 1})
    with pytest.raises(ResidualIntelligenceError, match="private body"):
        FeatureRange(name="prompt_text", minimum=0, maximum=10)
    with pytest.raises(ResidualIntelligenceError, match="credential-shaped"):
        observation(features={"api_key": 1})


def test_signal_contract_and_assessment_round_trip_reject_unknown_fields() -> None:
    signal = OODSignal(
        kind=OODSignalKind.FEATURE_RANGE,
        reason_code=REASON_FEATURE_RANGE,
        evidence_identities=("exit_code",),
    )
    assert OODSignal.from_dict(signal.to_dict()) == signal
    payload = signal.to_dict()
    payload["global_threshold_ppm"] = 1
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        OODSignal.from_dict(payload)
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        OODSignal(
            kind=OODSignalKind.FEATURE_RANGE,
            reason_code=REASON_FEATURE_RANGE,
            candidate_only=False,
        )
    envelope = contract()
    assert BoundaryContract.from_dict(envelope.to_dict()) == envelope
    dist = reference()
    assert ReferenceDistribution.from_dict(dist.to_dict()) == dist
    probe = observation()
    assert OODObservation.from_dict(probe.to_dict()) == probe
    result = assess()
    assert result.to_dict()["safety_established"] is False
    forged = result.to_dict()
    forged["safety_established"] = True
    with pytest.raises(ResidualIntelligenceError, match="establish safety"):
        OODAssessment.from_dict(forged)
    forged_accept = result.to_dict()
    forged_accept["in_boundary_eligible"] = False
    forged_accept["advisory_ood"] = True
    forged_accept["reason_codes"] = [REASON_ADVISORY_ONLY]
    forged_accept["forced_disposition"] = ExpertDisposition.ACCEPT.value
    with pytest.raises(ResidualIntelligenceError, match="cannot ACCEPT"):
        OODAssessment.from_dict(forged_accept)


def test_from_family_boundary_and_task_input_projection() -> None:
    target = group()
    envelope = BoundaryContract.from_family_boundary(
        family_boundary(),
        schema=SCHEMA,
        effects=(EFFECT,),
        repository=REPOSITORY,
        calibration_group_key=target.group_key,
        capabilities=(CAPABILITY,),
        required_context_fields=(CONTEXT_FIELD,),
    )
    assert envelope.family is ResidualTaskFamily.FAILURE_ATTRIBUTION
    assert envelope.authority_class == CANDIDATE_ONLY_AUTHORITY
    assert envelope.conservative_high_risk is False
    task = ResidualTaskInput(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        question_id="question:failure:1",
        repository_state_cid=REPOSITORY,
        objective_cid="objective:vrif",
        task_cid="task:VRIF-012",
        policy_cid="policy:residual-v1",
        context_capsule_cid=CONTEXT_FIELD,
        compact_features={"exit_code": 1, "failure_signature": "missing-edge"},
        allowed_outputs=("FAILURE_ATTRIBUTION", "ABSTAIN"),
        risk_class=RiskClass.R2,
        validation_policy="validator:failure-attribution@1",
        token_budget=256,
    )
    probe = observation_from_task_input(
        task,
        schema=SCHEMA,
        operation=OPERATION,
        effects=(EFFECT,),
        calibration_group_key=target.group_key,
        capabilities=(CAPABILITY,),
        context_fields=(CONTEXT_FIELD,),
    )
    record, _examples = admission()
    result = assess_out_of_distribution(
        probe,
        reference=reference(target),
        boundary=envelope,
        admission=record,
    )
    assert result.in_boundary_eligible is True
    assert result.safety_established is False


def test_inconsistent_boundary_and_reference_are_rejected() -> None:
    record, _examples = admission()
    with pytest.raises(ResidualIntelligenceError, match="inconsistent"):
        assess_out_of_distribution(
            observation(),
            reference=reference(),
            boundary=contract(repository="other-repo"),
            admission=record,
        )
