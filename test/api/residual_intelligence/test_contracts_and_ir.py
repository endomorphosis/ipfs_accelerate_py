from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    AuthorityViolationError,
    PrerequisiteFinding,
    PrerequisiteStatus,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TypedBlocker,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import (
    ResidualIntelligenceIR,
    ResidualTaskInput,
    ResidualTaskOutput,
)


def task_input(*, risk: RiskClass = RiskClass.R2) -> ResidualTaskInput:
    return ResidualTaskInput(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        question_id="question:failure:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-001",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features={"exit_code": 1, "failure_signature": "missing-edge"},
        allowed_outputs=("FAILURE_ATTRIBUTION", "ABSTAIN"),
        risk_class=risk,
        validation_policy="validator:failure-attribution@1",
        token_budget=256,
    )


def task_output(*, candidate_only: bool = True) -> ResidualTaskOutput:
    return ResidualTaskOutput(
        output_class="FAILURE_ATTRIBUTION",
        structured_payload={
            "failure_class": "missing_dependency_edge",
            "recommended_action": "expand_context_reference",
            "reference_ids": ["dependency:1"],
        },
        confidence_or_score=990_000,
        calibration_group="failure:python:R2:fixture",
        abstained=False,
        reason_codes=(),
        evidence_references=("failure-signature:1",),
        candidate_only=candidate_only,
    )


def test_contract_round_trips_and_canonical_identity() -> None:
    original = task_input()
    rebuilt = ResidualTaskInput.from_dict(original.to_dict())
    assert rebuilt == original
    assert rebuilt.input_id == original.input_id

    output = task_output()
    rebuilt_output = ResidualTaskOutput.from_dict(output.to_dict())
    assert rebuilt_output == output
    assert ResidualIntelligenceIR(original, output).ir_id


def test_unknown_field_rejection() -> None:
    payload = task_input().to_dict()
    payload["model_created_permission"] = True
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualTaskInput.from_dict(payload)


def test_model_created_authority_and_completion_rejected() -> None:
    with pytest.raises(AuthorityViolationError):
        ResidualTaskOutput(
            output_class="FAILURE_ATTRIBUTION",
            structured_payload={"nested": {"completed": True}},
            confidence_or_score=1,
            calibration_group="group:1",
            abstained=False,
            reason_codes=(),
            evidence_references=(),
        )


def test_candidate_only_cannot_be_lowered() -> None:
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        task_output(candidate_only=False)


def test_high_risk_candidate_requires_validation_or_abstention() -> None:
    with pytest.raises(ResidualIntelligenceError, match="R4/R5"):
        ResidualIntelligenceIR(task_input(risk=RiskClass.R4), task_output())

    validated = ResidualTaskOutput(
        output_class="FAILURE_ATTRIBUTION",
        structured_payload={"failure_class": "unknown"},
        confidence_or_score=100,
        calibration_group="failure:R4",
        abstained=False,
        reason_codes=("VALIDATION_REQUIRED",),
        evidence_references=(),
    )
    assert ResidualIntelligenceIR(task_input(risk=RiskClass.R4), validated).ir_id


def test_secret_shaped_compact_feature_is_rejected() -> None:
    with pytest.raises(ResidualIntelligenceError, match="credential-shaped"):
        ResidualTaskInput(
            **{
                **task_input().to_dict(include_id=False),
                "compact_features": {"api_key": "not-for-training"},
            }
        )


def test_prerequisite_and_typed_blocker_round_trip() -> None:
    finding = PrerequisiteFinding(
        name="OptionalCompiler",
        status=PrerequisiteStatus.MISSING,
        source_revision="revision:fixture",
        source_paths=("path/to/optional.py",),
        evidence_paths=("test/optional.py",),
        schema_versions=(),
        environment_id="environment:fixture",
        caveats=("do_not_recreate",),
        required=False,
    )
    blocker = TypedBlocker(
        blocker_code="optional_compiler_unavailable",
        task_ids=("VRIF-016",),
        prerequisite_ids=(finding.finding_id,),
        continuation="continue independent tasks and expose a narrow interface",
        retryable=True,
    )
    assert PrerequisiteFinding.from_dict(finding.to_dict()) == finding
    assert TypedBlocker.from_dict(blocker.to_dict()) == blocker
    assert finding.blocks_required_work is False


def test_typed_blocker_rejects_non_boolean_retryability() -> None:
    with pytest.raises(ResidualIntelligenceError, match="retryable must be boolean"):
        TypedBlocker(
            blocker_code="bad",
            task_ids=("VRIF-016",),
            prerequisite_ids=(),
            continuation="stop this integration only",
            retryable=1,  # type: ignore[arg-type]
        )
