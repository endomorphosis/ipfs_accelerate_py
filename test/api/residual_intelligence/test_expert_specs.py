from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    DEFAULT_EXPERT_SPECS,
    SMALLEST_FORM_ORDER,
    ExpertClass,
    ModelSizePolicy,
    ResidualExpertSpec,
    expert_spec_for,
    family_spec_for,
    form_requires_quality_delta,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import grammar_for
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import (
    DEFAULT_TASK_FAMILY_SPECS,
    ClosedSchema,
    ResidualTaskFamilySpec,
)


def task_input(
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    **overrides: object,
) -> ResidualTaskInput:
    spec = family_spec_for(family)
    payload: dict[str, object] = {
        "task_family": family,
        "question_id": "question:fixture:1",
        "repository_state_cid": "repo:tree:abc",
        "objective_cid": "objective:vrif",
        "task_cid": "task:VRIF-010",
        "policy_cid": "policy:residual-v1",
        "context_capsule_cid": "capsule:bounded:1",
        "compact_features": {},
        "allowed_outputs": spec.output_classes,
        "risk_class": spec.allowed_risk_classes[0],
        "validation_policy": spec.validation_contract,
        "token_budget": spec.token_budget,
    }
    payload.update(overrides)
    return ResidualTaskInput(**payload)  # type: ignore[arg-type]


def test_all_closed_families_have_exact_specs_and_boundaries() -> None:
    assert set(DEFAULT_TASK_FAMILY_SPECS) == set(ResidualTaskFamily)
    assert set(DEFAULT_EXPERT_SPECS) == set(ResidualTaskFamily)
    assert tuple(member.value for member in ExpertClass) == ("A", "B", "C", "D", "E")
    used_classes = {item.expert_class for item in DEFAULT_TASK_FAMILY_SPECS.values()}
    assert used_classes == set(ExpertClass)
    for family in ResidualTaskFamily:
        family_spec = family_spec_for(family)
        expert = expert_spec_for(family)
        grammar = grammar_for(family)
        assert family_spec.family_boundary_id == expert.family_boundary_id
        assert family_spec.boundary().boundary_id == expert.boundary().boundary_id
        assert family_spec.grammar_id == grammar.grammar_id == expert.grammar_id
        assert family_spec.validator_required is True
        assert family_spec.prose_default is False
        assert family_spec.candidate_only is True
        assert family_spec.authority_class == "candidate_only"
        assert expert.validator_required is True
        assert expert.prose_default is False
        assert family_spec.validation_contract
        assert family_spec.error_behavior
        assert family_spec.abstention_behavior


def test_prompt_similarity_cannot_override_family_boundary() -> None:
    family = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    other = family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    assert family.family_boundary_id != other.family_boundary_id
    payload = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).to_dict()
    payload["family_boundary_id"] = other.family_boundary_id
    with pytest.raises(ResidualIntelligenceError, match="boundary"):
        ResidualExpertSpec.from_dict(payload)


def test_class_a_through_e_and_unbounded_abstention() -> None:
    assert family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION).expert_class is ExpertClass.A
    assert family_spec_for(ResidualTaskFamily.EVIDENCE_RANKING).expert_class is ExpertClass.B
    assert family_spec_for(ResidualTaskFamily.TEST_SELECTION).expert_class is ExpertClass.C
    assert family_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION).expert_class is ExpertClass.D
    unbounded = expert_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert unbounded.expert_class is ExpertClass.E
    assert unbounded.always_abstain is True
    assert unbounded.form is ModelSizePolicy.HUMAN_REVIEW
    assert unbounded.allowed_forms == (ModelSizePolicy.HUMAN_REVIEW,)
    with pytest.raises(ResidualIntelligenceError, match="class E|size policy"):
        unbounded.admit_form(ModelSizePolicy.LINEAR_LOGISTIC, routing_changing_quality_delta=True)


def test_smallest_form_order_is_the_declared_cascade() -> None:
    assert SMALLEST_FORM_ORDER == (
        ModelSizePolicy.EXACT_LOOKUP,
        ModelSizePolicy.DECLARATIVE_RULE,
        ModelSizePolicy.LINEAR_LOGISTIC,
        ModelSizePolicy.SMALL_RANKER_ENCODER,
        ModelSizePolicy.CONSTRAINED_STRUCTURED_DECODER,
        ModelSizePolicy.PARAMETER_EFFICIENT_ADAPTER,
        ModelSizePolicy.QUANTIZED_LOCAL_GENERAL,
        ModelSizePolicy.REMOTE_STANDARD,
        ModelSizePolicy.REMOTE_STRONG,
        ModelSizePolicy.HUMAN_REVIEW,
    )
    assert tuple(ModelSizePolicy) == SMALLEST_FORM_ORDER
    classification = expert_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    assert classification.form is ModelSizePolicy.EXACT_LOOKUP
    assert classification.smallest_form is ModelSizePolicy.EXACT_LOOKUP
    classification.admit_form(ModelSizePolicy.EXACT_LOOKUP)
    classification.admit_form(ModelSizePolicy.HUMAN_REVIEW)


def test_larger_form_needs_routing_changing_quality_delta() -> None:
    spec = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert form_requires_quality_delta(spec.smallest_form, ModelSizePolicy.LINEAR_LOGISTIC)
    with pytest.raises(ResidualIntelligenceError, match="routing-changing quality delta"):
        spec.admit_form(ModelSizePolicy.LINEAR_LOGISTIC)
    advanced = spec.with_form(
        ModelSizePolicy.LINEAR_LOGISTIC, routing_changing_quality_delta=True
    )
    assert advanced.form is ModelSizePolicy.LINEAR_LOGISTIC
    assert advanced.family_boundary_id == spec.family_boundary_id
    with pytest.raises(ResidualIntelligenceError, match="size policy|remote"):
        spec.admit_form(ModelSizePolicy.REMOTE_STANDARD, routing_changing_quality_delta=True)
    remote_ok = expert_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    assert remote_ok.admit_form(
        ModelSizePolicy.REMOTE_STANDARD, routing_changing_quality_delta=True
    ) is ModelSizePolicy.REMOTE_STANDARD


def test_closed_schemas_reject_unknown_fields_and_prose() -> None:
    spec = family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        spec.input_schema.validate({"arbitrary_prose": "explain the task"})
    with pytest.raises(ResidualIntelligenceError, match="prose"):
        ClosedSchema(name="task-classification-prose@1", fields=("explanation",))
    payload = spec.input_schema.to_dict()
    payload["allow_prose"] = True
    with pytest.raises(ResidualIntelligenceError, match="allow_prose"):
        ClosedSchema.from_dict(payload)
    payload = spec.to_dict()
    payload["prose_default"] = True
    with pytest.raises(ResidualIntelligenceError, match="prose_default"):
        ResidualTaskFamilySpec.from_dict(payload)


def test_risk_ceiling_rejects_unsupported_family_pairs() -> None:
    classification = family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    classification.admit_risk(RiskClass.R2)
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk pair"):
        classification.admit_risk(RiskClass.R5)
    sketch = family_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk pair"):
        sketch.admit_risk(RiskClass.R0)
    unbounded = family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    unbounded.admit_risk(RiskClass.R5)
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk pair"):
        unbounded.admit_risk(RiskClass.R2)
    expert = expert_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    payload = expert.to_dict()
    payload["risk_ceiling"] = RiskClass.R5.value
    with pytest.raises(ResidualIntelligenceError, match="risk ceiling"):
        ResidualExpertSpec.from_dict(payload)


def test_validator_is_required_and_bound_to_the_input() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    payload = spec.to_dict()
    payload["validator_required"] = False
    with pytest.raises(ResidualIntelligenceError, match="validator_required"):
        ResidualTaskFamilySpec.from_dict(payload)
    admitted = spec.admit_input(task_input())
    assert admitted.validation_policy == spec.validation_contract
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        spec.admit_input(task_input(compact_features={"explanation": "long prose"}))
    with pytest.raises(ResidualIntelligenceError, match="required validator"):
        spec.admit_input(task_input(validation_policy="optional-or-missing-validator"))
    with pytest.raises(ResidualIntelligenceError, match="token budget"):
        spec.admit_input(task_input(token_budget=spec.token_budget + 1))


def test_no_prose_default_and_candidate_only_cannot_be_lowered() -> None:
    expert = expert_spec_for(ResidualTaskFamily.LEMMA_SUGGESTION)
    payload = expert.to_dict()
    payload["candidate_only"] = False
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        ResidualExpertSpec.from_dict(payload)
    payload = expert.to_dict()
    payload["authority_class"] = "autonomous"
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        ResidualExpertSpec.from_dict(payload)
    payload = expert.to_dict()
    payload["prose_default"] = True
    with pytest.raises(ResidualIntelligenceError, match="prose_default"):
        ResidualExpertSpec.from_dict(payload)


def test_family_and_expert_specs_round_trip_canonically() -> None:
    family = family_spec_for(ResidualTaskFamily.PROCEDURE_HOLE_FILLING)
    rebuilt_family = ResidualTaskFamilySpec.from_dict(family.to_dict())
    assert rebuilt_family == family
    assert rebuilt_family.spec_id == family.spec_id
    expert = expert_spec_for(ResidualTaskFamily.PROCEDURE_HOLE_FILLING)
    rebuilt_expert = ResidualExpertSpec.from_dict(expert.to_dict())
    assert rebuilt_expert == expert
    assert rebuilt_expert.spec_id == expert.spec_id


def test_unknown_spec_fields_are_rejected() -> None:
    payload = family_spec_for(ResidualTaskFamily.CONTEXT_SUFFICIENCY).to_dict()
    payload["examples"] = [{"prompt": "similar text"}]
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualTaskFamilySpec.from_dict(payload)
    expert_payload = expert_spec_for(ResidualTaskFamily.CONTEXT_SUFFICIENCY).to_dict()
    expert_payload["promotion"] = True
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualExpertSpec.from_dict(expert_payload)


def test_closed_output_schema_matches_grammar_and_rejects_arbitrary_payloads() -> None:
    spec = family_spec_for(ResidualTaskFamily.RETRY_OR_ESCALATE)
    grammar = grammar_for(ResidualTaskFamily.RETRY_OR_ESCALATE)
    assert spec.output_schema.fields == grammar.payload_fields
    spec.output_schema.validate({"decision": "retry"})
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        spec.output_schema.validate({"decision": "retry", "markdown": "please retry"})
    with pytest.raises(ResidualIntelligenceError, match="closed enumeration"):
        spec.output_schema.validate({"decision": "invent-a-new-policy"})


def test_privacy_route_cannot_weaken_or_enable_unauthorized_remote() -> None:
    expert = expert_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)
    assert expert.privacy_class is PrivacyClass.REPOSITORY_PRIVATE
    assert expert.remote_route_permitted is False
    payload = expert.to_dict()
    payload["privacy_class"] = PrivacyClass.PUBLIC.value
    with pytest.raises(ResidualIntelligenceError, match="privacy"):
        ResidualExpertSpec.from_dict(payload)
    payload = expert.to_dict()
    payload["remote_route_permitted"] = True
    with pytest.raises(ResidualIntelligenceError, match="remote"):
        ResidualExpertSpec.from_dict(payload)
