from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.ood import (
    BoundaryCheck,
    BoundaryContract,
    OODAssessment,
    OODSignal,
    OODSignalKind,
    assess_out_of_distribution,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput

from .helpers import admission


def task_input(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    risk: RiskClass = RiskClass.R2,
    features: Mapping[str, Any] | None = None,
) -> ResidualTaskInput:
    return ResidualTaskInput(
        task_family=family,
        question_id="question:failure:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-012",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=dict(features or {"exit_code": 1, "failure_signature": "missing-edge"}),
        allowed_outputs=("FAILURE_ATTRIBUTION", "ABSTAIN"),
        risk_class=risk,
        validation_policy="validator:failure-attribution@1",
        token_budget=256,
    )


def boundary_contract(**overrides: Any) -> BoundaryContract:
    payload: dict[str, Any] = {
        "allowed_task_families": (ResidualTaskFamily.FAILURE_ATTRIBUTION.value,),
        "allowed_schemas": ("failure-signature@1",),
        "allowed_operations": ("classify_failure",),
        "allowed_repository_families": ("ipfs_accelerate_py",),
        "allowed_effects": ("typed_candidate",),
        "allowed_authorities": ("candidate_only",),
        "known_calibration_groups": ("failure:python:R2:fixture",),
        "required_capabilities": ("cpu",),
        "required_context_references": ("capsule:bounded:1",),
        "feature_ranges": {"exit_code": {"minimum": 0, "maximum": 10}},
    }
    payload.update(overrides)
    return BoundaryContract(**payload)


def in_boundary_observation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "failure-signature@1",
        "operation": "classify_failure",
        "repository_family": "ipfs_accelerate_py",
        "effects": ("typed_candidate",),
        "authorities": ("candidate_only",),
        "calibration_group": "failure:python:R2:fixture",
        "available_capabilities": ("cpu",),
        "context_reference_ids": ("capsule:bounded:1",),
        "context_complete": True,
        "disagreement": False,
        "ood_detection_ran": True,
    }
    payload.update(overrides)
    return payload


def assess(
    *,
    risk: RiskClass = RiskClass.R2,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    features: Mapping[str, Any] | None = None,
    contract: BoundaryContract | None = None,
    observation: Mapping[str, Any] | None = None,
    reference_distribution: Mapping[str, Any] | None = None,
    corpus_admission: Any = None,
) -> OODAssessment:
    return assess_out_of_distribution(
        task_input(family=family, risk=risk, features=features),
        contract or boundary_contract(),
        observation=in_boundary_observation() if observation is None else observation,
        reference_distribution=reference_distribution,
        corpus_admission=corpus_admission,
    )


def test_known_in_boundary_fixtures_remain_eligible() -> None:
    assessment = assess()
    assert assessment.eligible is True
    assert assessment.in_boundary is True
    assert assessment.conservative_abstain is False
    assert assessment.ood_detected is False
    assert assessment.ood_detection_ran is True
    assert assessment.safety_established is True
    assert assessment.disposition is ExpertDisposition.ACCEPT
    assert assessment.candidate_only is True
    assert assessment.family_in_boundary is True
    assert assessment.schema_in_boundary is True
    assert assessment.effect_in_boundary is True
    assert assessment.authority_in_boundary is True
    assert assessment.repository_in_boundary is True
    assert assessment.calibration_in_boundary is True
    assert assessment.capability_in_boundary is True
    assert assessment.context_in_boundary is True
    assert assessment.signals == ()
    assert assessment.boundary_violations == ()


def test_high_risk_in_boundary_fixture_stays_eligible_as_validation_required() -> None:
    assessment = assess(risk=RiskClass.R4)
    assert assessment.eligible is True
    assert assessment.in_boundary is True
    assert assessment.conservative_abstain is False
    assert assessment.disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert assessment.safety_established is True


def test_feature_range_is_advisory_unless_policy_admits() -> None:
    advisory = assess(features={"exit_code": 99, "failure_signature": "missing-edge"})
    assert advisory.ood_detected is True
    assert {item.kind for item in advisory.signals} == {OODSignalKind.FEATURE_RANGE}
    assert all(item.advisory is True for item in advisory.signals)
    assert advisory.in_boundary is True
    assert advisory.eligible is True
    assert advisory.conservative_abstain is False
    assert advisory.disposition is ExpertDisposition.ACCEPT
    assert "feature_range" in advisory.reason_codes

    admitted = assess_out_of_distribution(
        task_input(features={"exit_code": 99, "failure_signature": "missing-edge"}),
        boundary_contract(admit_ood_policy=True),
        observation=in_boundary_observation(),
    )
    assert admitted.eligible is False
    assert admitted.disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
    assert all(item.advisory is False for item in admitted.signals)
    assert "ood_policy_admitted" in admitted.reason_codes


def test_unknown_schema_operation_repository_are_independent_hard_gates() -> None:
    schema = assess(observation=in_boundary_observation(schema="unknown-schema@9"))
    assert schema.schema_in_boundary is False
    assert schema.repository_in_boundary is True
    assert schema.eligible is False
    assert schema.disposition is ExpertDisposition.REJECT_INPUT
    assert any(item.kind is OODSignalKind.UNKNOWN_SCHEMA for item in schema.signals)
    assert any(item.check is BoundaryCheck.SCHEMA for item in schema.boundary_violations)

    operation = assess(observation=in_boundary_observation(operation="novel_unbounded_rewrite"))
    assert operation.schema_in_boundary is False
    assert any(item.kind is OODSignalKind.UNKNOWN_OPERATION for item in operation.signals)

    repository = assess(observation=in_boundary_observation(repository_family="foreign-tree"))
    assert repository.repository_in_boundary is False
    assert repository.schema_in_boundary is True
    assert any(item.kind is OODSignalKind.UNKNOWN_REPOSITORY for item in repository.signals)
    assert any(item.check is BoundaryCheck.REPOSITORY for item in repository.boundary_violations)


def test_unseen_effects_and_authority_are_independent_hard_gates() -> None:
    effects = assess(observation=in_boundary_observation(effects=("autonomous_mutation",)))
    assert effects.effect_in_boundary is False
    assert effects.authority_in_boundary is True
    assert effects.eligible is False
    assert any(item.kind is OODSignalKind.UNSEEN_EFFECT for item in effects.signals)
    assert any(item.check is BoundaryCheck.EFFECT for item in effects.boundary_violations)

    authority = assess(
        observation=in_boundary_observation(authorities=("model_created_completion",))
    )
    assert authority.authority_in_boundary is False
    assert authority.effect_in_boundary is True
    assert any(item.kind is OODSignalKind.UNSEEN_AUTHORITY for item in authority.signals)
    assert any(item.check is BoundaryCheck.AUTHORITY for item in authority.boundary_violations)


def test_disagreement_is_advisory_ood_and_does_not_force_low_risk_abstention() -> None:
    assessment = assess(
        observation=in_boundary_observation(
            disagreement=True,
            disagreement_sources=("teacher:remote-strong", "local:linear"),
        )
    )
    assert assessment.ood_detected is True
    assert {item.kind for item in assessment.signals} == {OODSignalKind.DISAGREEMENT}
    assert all(item.advisory is True for item in assessment.signals)
    assert assessment.in_boundary is True
    assert assessment.eligible is True
    assert assessment.conservative_abstain is False
    assert "teacher:remote-strong" in assessment.evidence_references

    high_risk = assess(
        risk=RiskClass.R4,
        observation=in_boundary_observation(
            disagreement=True,
            disagreement_sources=("teacher:remote-strong",),
        ),
    )
    assert high_risk.conservative_abstain is False
    assert high_risk.eligible is True
    assert high_risk.disposition is ExpertDisposition.VALIDATION_REQUIRED


def test_calibration_absence_is_advisory_ood_and_independent_boundary() -> None:
    missing = assess(observation=in_boundary_observation(calibration_group=""))
    assert missing.calibration_in_boundary is False
    assert missing.eligible is False
    assert any(item.kind is OODSignalKind.CALIBRATION_ABSENCE for item in missing.signals)
    assert any(item.check is BoundaryCheck.CALIBRATION for item in missing.boundary_violations)

    unknown = assess(observation=in_boundary_observation(calibration_group="never-fitted:R9"))
    assert unknown.calibration_in_boundary is False
    assert "calibration_absence" in unknown.reason_codes


def test_context_incomplete_is_advisory_ood_and_independent_boundary() -> None:
    incomplete = assess(observation=in_boundary_observation(context_complete=False))
    assert incomplete.context_in_boundary is False
    assert incomplete.eligible is False
    assert any(item.kind is OODSignalKind.CONTEXT_INCOMPLETE for item in incomplete.signals)

    missing_ref = assess(
        observation=in_boundary_observation(context_reference_ids=("capsule:other",))
    )
    assert missing_ref.context_in_boundary is False
    assert any(item.subject == "capsule:bounded:1" for item in missing_ref.boundary_violations)


def test_conservative_high_risk_unknown_or_missing_independently_abstains() -> None:
    missing_group = assess(
        risk=RiskClass.R4,
        observation=in_boundary_observation(calibration_group=""),
    )
    assert missing_group.conservative_abstain is True
    assert missing_group.eligible is False
    assert missing_group.disposition is ExpertDisposition.ABSTAIN
    assert "conservative_high_risk" in missing_group.reason_codes

    unknown_family = assess(
        risk=RiskClass.R5,
        family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
    )
    assert unknown_family.family_in_boundary is False
    assert unknown_family.conservative_abstain is True
    assert unknown_family.disposition is ExpertDisposition.ABSTAIN

    incomplete_context = assess(
        risk=RiskClass.R4,
        observation=in_boundary_observation(ood_detection_ran=False, context_complete=False),
    )
    assert incomplete_context.ood_detection_ran is False
    assert incomplete_context.signals == ()
    assert incomplete_context.conservative_abstain is True
    assert incomplete_context.disposition is ExpertDisposition.ABSTAIN
    assert incomplete_context.safety_established is False


def test_missing_ood_detection_never_establishes_safety() -> None:
    assessment = assess(observation=in_boundary_observation(ood_detection_ran=False))
    assert assessment.ood_detection_ran is False
    assert assessment.safety_established is False
    assert assessment.eligible is True
    assert assessment.in_boundary is True
    assert assessment.signals == ()
    assert "ood_detection_missing" in assessment.reason_codes

    with pytest.raises(ResidualIntelligenceError, match="never establishes safety"):
        OODAssessment(
            input_id="input:fixture",
            contract_id="contract:fixture",
            signals=(),
            boundary_violations=(),
            disposition=ExpertDisposition.ACCEPT,
            eligible=True,
            in_boundary=True,
            family_in_boundary=True,
            schema_in_boundary=True,
            effect_in_boundary=True,
            authority_in_boundary=True,
            repository_in_boundary=True,
            calibration_in_boundary=True,
            capability_in_boundary=True,
            context_in_boundary=True,
            conservative_abstain=False,
            ood_detected=False,
            ood_detection_ran=False,
            safety_established=True,
            reason_codes=(),
            evidence_references=(),
        )


def test_independent_boundary_checks_run_when_ood_detection_is_absent() -> None:
    assessment = assess(
        observation=in_boundary_observation(
            ood_detection_ran=False,
            schema="unknown-schema@9",
            effects=("unseen_effect",),
        )
    )
    assert assessment.ood_detection_ran is False
    assert assessment.signals == ()
    assert assessment.schema_in_boundary is False
    assert assessment.effect_in_boundary is False
    assert assessment.eligible is False
    assert {item.check for item in assessment.boundary_violations} >= {
        BoundaryCheck.SCHEMA,
        BoundaryCheck.EFFECT,
    }


def test_capability_unavailable_is_an_independent_hard_gate() -> None:
    assessment = assess(observation=in_boundary_observation(available_capabilities=()))
    assert assessment.capability_in_boundary is False
    assert assessment.eligible is False
    assert assessment.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    assert assessment.conservative_abstain is False
    assert any(item.check is BoundaryCheck.CAPABILITY for item in assessment.boundary_violations)


def test_family_distance_is_advisory_ood_signal() -> None:
    assessment = assess(observation=in_boundary_observation(family_distance_ppm=800_000))
    assert assessment.ood_detected is True
    assert assessment.eligible is True
    assert {item.kind for item in assessment.signals} == {OODSignalKind.FAMILY_DISTANCE}
    assert assessment.signals[0].score_ppm == 800_000
    assert assessment.signals[0].advisory is True


def test_reference_distribution_requires_admitted_corpus() -> None:
    stats = {
        "feature_ranges": {"exit_code": {"minimum": 0, "maximum": 10}},
        "statistic_identity": "stats:fixture-v1",
        "example_count": 4,
        "held_out": True,
    }
    with pytest.raises(ResidualIntelligenceError, match="admitted TrainingCorpusAdmission"):
        assess(reference_distribution=stats)

    record, _examples = admission(admitted=False)
    with pytest.raises(ResidualIntelligenceError, match="admitted TrainingCorpusAdmission"):
        assess(reference_distribution=stats, corpus_admission=record)

    admitted, _examples = admission()
    assessment = assess(reference_distribution=stats, corpus_admission=admitted)
    assert assessment.eligible is True
    assert assessment.safety_established is True


def test_compact_statistics_cannot_contain_recoverable_private_source() -> None:
    admitted, _examples = admission()
    with pytest.raises(ResidualIntelligenceError, match="recoverable private source"):
        assess(
            reference_distribution={
                "statistic_identity": "stats:fixture-v1",
                "examples": ["def recover_private_source():\n    return secret"],
            },
            corpus_admission=admitted,
        )
    with pytest.raises(ResidualIntelligenceError, match="recoverable private source"):
        assess(
            reference_distribution={
                "statistic_identity": "x" * 300,
            },
            corpus_admission=admitted,
        )


def test_unknown_observation_fields_and_contract_fields_rejected() -> None:
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        assess(observation=in_boundary_observation(promotion=True))
    payload = boundary_contract().to_dict()
    payload["model_created_permission"] = True
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        BoundaryContract.from_dict(payload)


def test_ood_signal_and_assessment_round_trip() -> None:
    contract = boundary_contract()
    rebuilt_contract = BoundaryContract.from_dict(contract.to_dict())
    assert rebuilt_contract == contract
    assert rebuilt_contract.contract_id == contract.contract_id

    assessment = assess(features={"exit_code": 99, "failure_signature": "missing-edge"})
    rebuilt = OODAssessment.from_dict(assessment.to_dict())
    assert rebuilt == assessment
    assert rebuilt.assessment_id == assessment.assessment_id
    assert OODSignal.from_dict(assessment.signals[0].to_dict()) == assessment.signals[0]


def test_candidate_only_cannot_be_lowered_on_assessment() -> None:
    payload = assess().to_dict(include_id=False)
    payload["candidate_only"] = False
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        OODAssessment.from_dict(payload)


def test_independent_checks_record_every_violation_without_short_circuit() -> None:
    assessment = assess(
        family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
        observation=in_boundary_observation(
            schema="unknown-schema@9",
            operation="unknown_operation",
            repository_family="foreign",
            effects=("unseen_effect",),
            authorities=("unseen_authority",),
            calibration_group="",
            available_capabilities=(),
            context_complete=False,
        ),
    )
    assert assessment.in_boundary is False
    assert assessment.family_in_boundary is False
    assert assessment.schema_in_boundary is False
    assert assessment.effect_in_boundary is False
    assert assessment.authority_in_boundary is False
    assert assessment.repository_in_boundary is False
    assert assessment.calibration_in_boundary is False
    assert assessment.capability_in_boundary is False
    assert assessment.context_in_boundary is False
    assert {item.check for item in assessment.boundary_violations} == set(BoundaryCheck)
    assert assessment.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
