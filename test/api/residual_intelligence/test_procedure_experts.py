from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    EffectClass,
    HoleType,
    ProcedureHole,
    ProviderClass,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.procedure_experts import (
    NOMINATED_RULE_PREFIX,
    REASON_AUTHORITY_MUTATION_FORBIDDEN,
    REASON_COMPILER_INACTIVE,
    REASON_HOLE_UNDECLARED,
    REASON_PRECONDITIONS_UNSATISFIED,
    REASON_PROCEDURE_ROOT_MISMATCH,
    REASON_REPEATED_HOLE_RULE_NOMINATION,
    REASON_VALIDATOR_DECIDES,
    ProcedureHoleCapability,
    ProcedureHoleExpertAdapter,
    ProcedureHoleResolution,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DecodeStatus,
)

from .test_structured_specialist import (
    encoded_hole,
    expert,
    hole_features,
    request_for,
    task_input,
)

PROCEDURE_ROOT = "procedure:root:1"


def declared_hole(*, hole_id: str = "hole:bind-arg-1") -> ProcedureHole:
    return ProcedureHole(
        hole_id=hole_id,
        hole_type=HoleType.SELECT_ONE_OF_ALLOWED_SYMBOLS,
        input_schema_ref="schema:hole-input:1",
        output_schema_ref="schema:hole-output:1",
        allowed_provider_classes=(
            ProviderClass.DECLARATIVE_RULE,
            ProviderClass.LOCAL_SMALL_MODEL,
        ),
        context_budget_bytes=4096,
        authority_requirement_ids=(),
        effect_classes=(EffectClass.OBSERVE,),
        validation_observation_ids=("validator:procedure-hole@1",),
        fallback_step_id="step:fallback",
        maximum_attempts=3,
    )


def adapter(
    *,
    compiler_available: bool = True,
    capability: ProcedureHoleCapability | None = None,
) -> ProcedureHoleExpertAdapter:
    snapshot = capability or ProcedureHoleCapability.current(
        overlay={
            "parse_and_validate": compiler_available,
            "deterministic_invoke": compiler_available,
        }
    )
    return ProcedureHoleExpertAdapter(
        specialist=expert(compiler_available=compiler_available),
        procedure_root=PROCEDURE_ROOT,
        declared_holes=(declared_hole(),),
        capability=snapshot,
    )


def test_current_compiler_capability_is_read_only_and_inactive_when_missing() -> None:
    live = ProcedureHoleCapability.current()
    assert live.available is True
    assert live.parse_and_validate is True
    assert live.deterministic_invoke is True
    assert live.synthesize is False
    assert live.promote is False
    assert live.modify_policy is False
    inactive = ProcedureHoleCapability.current(
        overlay={"parse_and_validate": False, "deterministic_invoke": True}
    )
    assert inactive.available is False
    assert inactive.inactive_reason == REASON_COMPILER_INACTIVE
    with pytest.raises(ResidualIntelligenceError, match=REASON_AUTHORITY_MUTATION_FORBIDDEN):
        ProcedureHoleCapability(
            parse_and_validate=True,
            deterministic_invoke=True,
            synthesize=True,
        )


def test_adapter_rejects_undeclared_or_mismatched_roots() -> None:
    bound = adapter()
    undeclared = bound.resolve(
        request_for(task_input(features=hole_features(hole_id="hole:other")))
    )
    assert undeclared.disposition is ExpertDisposition.REJECT_INPUT
    assert REASON_HOLE_UNDECLARED in undeclared.reason_codes
    assert undeclared.candidate_only is True
    mismatched = bound.resolve(
        request_for(task_input(features=hole_features(procedure_root="procedure:other")))
    )
    assert mismatched.disposition is ExpertDisposition.REJECT_INPUT
    assert REASON_PROCEDURE_ROOT_MISMATCH in mismatched.reason_codes


def test_inactive_adapter_when_compiler_unavailable() -> None:
    bound = adapter(compiler_available=False)
    result = bound.resolve(request_for(task_input(), raw_output=encoded_hole()))
    assert result.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    assert REASON_COMPILER_INACTIVE in result.reason_codes
    assert result.specialist_prediction is None
    assert result.candidate_only is True


def test_preconditions_must_hold_before_a_candidate_is_emitted() -> None:
    result = adapter().resolve(
        request_for(
            task_input(features=hole_features(procedure_preconditions_satisfied=False)),
            raw_output=encoded_hole(),
        )
    )
    assert result.disposition is ExpertDisposition.ABSTAIN
    assert REASON_PRECONDITIONS_UNSATISFIED in result.reason_codes
    assert result.operator_id == ""


def test_typed_hole_candidate_stays_proposal_until_validator_decides() -> None:
    accepted = adapter().resolve(request_for(task_input(), raw_output=encoded_hole()))
    assert accepted.hole_id == "hole:bind-arg-1"
    assert accepted.procedure_root == PROCEDURE_ROOT
    assert accepted.operator_id == "bind_argument"
    assert accepted.argument_reference_ids == ("arg:0",)
    assert accepted.candidate_only is True
    assert REASON_VALIDATOR_DECIDES in accepted.reason_codes
    assert accepted.independent_validation is not None
    assert accepted.independent_validation.accepted is True
    assert accepted.specialist_prediction is not None
    assert accepted.specialist_prediction.decode_result.status is DecodeStatus.VALID
    rejected = adapter().resolve(
        request_for(task_input(), raw_output=encoded_hole(), accepted=False)
    )
    assert rejected.disposition is ExpertDisposition.REJECT_INPUT
    assert REASON_VALIDATOR_DECIDES in rejected.reason_codes


def test_repeated_hole_nominates_a_deterministic_rule_without_compiler_mutation() -> None:
    bound = adapter()
    first = bound.resolve(request_for(task_input(), raw_output=encoded_hole()), prior_attempts=0)
    assert first.nominated_rule == ""
    repeated = bound.resolve(
        request_for(task_input(), raw_output=encoded_hole()),
        prior_attempts=2,
    )
    assert repeated.nominated_rule == f"{NOMINATED_RULE_PREFIX}hole:bind-arg-1"
    assert REASON_REPEATED_HOLE_RULE_NOMINATION in repeated.reason_codes
    assert repeated.candidate_only is True
    with pytest.raises(ResidualIntelligenceError, match=REASON_AUTHORITY_MUTATION_FORBIDDEN):
        bound.synthesize()
    with pytest.raises(ResidualIntelligenceError, match=REASON_AUTHORITY_MUTATION_FORBIDDEN):
        bound.promote()
    with pytest.raises(ResidualIntelligenceError, match=REASON_AUTHORITY_MUTATION_FORBIDDEN):
        bound.modify_policy()


def test_capability_and_resolution_round_trip() -> None:
    live = ProcedureHoleCapability.current()
    restored = ProcedureHoleCapability.from_dict(json.loads(json.dumps(live.to_dict())))
    assert restored == live
    resolution = adapter().resolve(request_for(task_input(), raw_output=encoded_hole()))
    payload = resolution.to_dict()
    assert payload["candidate_only"] is True
    assert payload["schema"].endswith("procedure-hole-resolution@1")
