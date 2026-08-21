from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    BEYOND_E_FORMS,
    EXPERT_CLASS_FORMS,
    EXPERT_SPECS,
    REASON_LARGER_FORM,
    SMALLEST_FORM_ORDER,
    ExpertClass,
    ModelSizePolicy,
    ResidualExpertSpec,
    ResidualTaskFamilySpec,
    assert_exact_family_boundary,
    assert_family_risk_admitted,
    assert_no_prose_default,
    assert_validator_required,
    default_expert_specs,
    expert_class_rank,
    expert_spec_for,
    experts_of_class,
    family_spec_for,
    reject_similarity_grouping,
    smallest_expert_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.inventory import (
    ResidualFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import (
    ResidualTaskInput,
    ResidualTaskOutput,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    grammar_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import (
    EXPERT_CLASS_VALUES,
    FAMILY_SPECS,
    REASON_NOVEL_UNBOUNDED,
    REASON_PROMPT_SIMILARITY,
    REASON_PROSE_DEFAULT,
    REASON_RISK_CEILING,
    REASON_VALIDATOR_REQUIRED,
    family_boundary_for,
)

from .helpers import admission


def task_input(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    risk: RiskClass = RiskClass.R2,
    features: dict[str, object] | None = None,
    token_budget: int = 256,
) -> ResidualTaskInput:
    return ResidualTaskInput(
        task_family=family,
        question_id="question:expert-spec:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-010",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=features if features is not None else {"exit_code": 1},
        allowed_outputs=(family.value, "ABSTAIN"),
        risk_class=risk,
        validation_policy="validator:expert-spec@1",
        token_budget=token_budget,
    )


def accepted_output(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    payload: dict[str, object] | None = None,
    abstained: bool = False,
    reason_codes: tuple[str, ...] = (),
    evidence: tuple[str, ...] = ("validator:failure-attribution@1",),
) -> ResidualTaskOutput:
    grammar = grammar_for(family)
    output_class = grammar.abstention_output_class if abstained else grammar.output_classes[0]
    return ResidualTaskOutput(
        output_class=output_class,
        structured_payload={} if abstained else payload or {
            "failure_class": "missing_dependency_edge",
            "recommended_action": "expand_context_reference",
            "reference_ids": ["dependency:1"],
        },
        confidence_or_score=900_000,
        calibration_group="failure:python:R2:fixture",
        abstained=abstained,
        reason_codes=reason_codes if reason_codes or not abstained else ("unknown_signature",),
        evidence_references=() if abstained else evidence,
        candidate_only=True,
    )


def test_every_taxonomy_family_has_exact_semantic_boundary() -> None:
    assert set(FAMILY_SPECS) == set(ResidualTaskFamily)
    assert len(default_expert_specs()) == len(ResidualTaskFamily)
    for family in ResidualTaskFamily:
        spec = family_spec_for(family)
        boundary = family_boundary_for(family)
        assert spec.task_family is family
        assert spec.as_boundary() == boundary
        assert spec.boundary_id == boundary.boundary_id
        assert spec.authority_class == "candidate_only"
        assert spec.validation_contract
        assert spec.error_behavior
        assert spec.abstention_behavior
        assert spec.input_semantics
        assert spec.output_semantics
        assert spec.validator_required is True
        assert spec.emit_prose_by_default is False
        assert spec.evaluation_corpus_admission_id == ""
        assert spec.training_corpus_admission_id == ""
        expert = smallest_expert_for(family)
        assert expert.family_boundary_id == spec.boundary_id
        assert expert.family_spec_id == spec.family_spec_id
        assert_exact_family_boundary(spec, boundary)
        assert_validator_required(spec)
        assert_no_prose_default(expert)


def test_prompt_similarity_cannot_override_family_boundary() -> None:
    left = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    right = family_spec_for(ResidualTaskFamily.RETRY_OR_ESCALATE)
    with pytest.raises(ResidualIntelligenceError, match="exact semantic boundary"):
        assert_exact_family_boundary(left, right)
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROMPT_SIMILARITY):
        reject_similarity_grouping("prompt_similarity")
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROMPT_SIMILARITY):
        reject_similarity_grouping("embedding-similarity")
    foreign = ResidualFamilyBoundary(
        task_family=ResidualTaskFamily.FAILURE_ATTRIBUTION,
        input_semantics="prompt-similar failure text",
        output_semantics="one failure class and one bounded action candidate",
        risk_class=RiskClass.R2,
        authority_class="candidate_only",
        validation_contract="failure-attribution-validator@1",
        error_behavior="invalid output or failed validation escalates",
        abstention_behavior="unknown signatures abstain",
    )
    with pytest.raises(ResidualIntelligenceError, match="exact semantic boundary"):
        assert_exact_family_boundary(left, foreign)


def test_expert_classes_are_a_through_e_in_smallest_form_order() -> None:
    assert tuple(item.value for item in ExpertClass) == EXPERT_CLASS_VALUES
    assert SMALLEST_FORM_ORDER == (
        ExpertClass.A,
        ExpertClass.B,
        ExpertClass.C,
        ExpertClass.D,
        ExpertClass.E,
    )
    assert [expert_class_rank(item) for item in SMALLEST_FORM_ORDER] == [0, 1, 2, 3, 4]
    assert EXPERT_CLASS_FORMS[ExpertClass.A] == "exact_lookup"
    assert EXPERT_CLASS_FORMS[ExpertClass.E] == "structured_decoder"
    assert BEYOND_E_FORMS[-1] == "remote_strong"
    for family_spec in FAMILY_SPECS.values():
        assert family_spec.allowed_expert_classes == EXPERT_CLASS_VALUES[
            : len(family_spec.allowed_expert_classes)
        ]
        assert family_spec.allowed_expert_classes[0] == "A"
    for form in ExpertClass:
        assert experts_of_class(form)
        assert all(item.expert_class is form for item in experts_of_class(form))


def test_larger_form_requires_routing_changing_quality_delta() -> None:
    family = family_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)
    policy = ModelSizePolicy.for_family(family)
    assert policy.smallest_form is ExpertClass.A
    assert policy.maximum_form is ExpertClass.E
    policy.admit(ExpertClass.A)
    with pytest.raises(ResidualIntelligenceError, match=REASON_LARGER_FORM):
        policy.admit(ExpertClass.E)
    with pytest.raises(ResidualIntelligenceError, match=REASON_LARGER_FORM):
        policy.admit(ExpertClass.A, beyond_e_form="remote_strong")
    admitted = ModelSizePolicy.for_family(
        family,
        quality_delta_evidence_cid="evidence:held-out-delta:1",
        routing_changing_quality_delta_ppm=10_000,
        adapter_permitted=True,
        remote_permitted=True,
    )
    assert admitted.admit(
        ExpertClass.E,
        quality_delta_ppm=10_000,
        evidence_cid="evidence:held-out-delta:1",
    ) is ExpertClass.E
    admitted.admit(
        ExpertClass.E,
        quality_delta_ppm=10_000,
        evidence_cid="evidence:held-out-delta:1",
        beyond_e_form="parameter_efficient_adapter",
    )
    with pytest.raises(ResidualIntelligenceError, match=REASON_LARGER_FORM):
        admitted.admit(
            ExpertClass.E,
            quality_delta_ppm=1,
            evidence_cid="evidence:held-out-delta:1",
        )
    classification = family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    with pytest.raises(ResidualIntelligenceError, match="maximum"):
        ModelSizePolicy.for_family(classification).admit(ExpertClass.E)


def test_closed_schemas_reject_unknown_fields() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    payload = spec.to_dict()
    payload["prompt_similarity"] = 0.99
    with pytest.raises(UnknownFieldError):
        ResidualTaskFamilySpec.from_dict(payload)
    payload = spec.to_dict()
    payload["examples"] = [{"input": "prose"}]
    with pytest.raises(UnknownFieldError):
        ResidualTaskFamilySpec.from_dict(payload)
    expert = smallest_expert_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    expert_payload = expert.to_dict()
    expert_payload["few_shot"] = True
    with pytest.raises(UnknownFieldError):
        ResidualExpertSpec.from_dict(expert_payload)
    reasons = spec.validate_task_input(
        task_input(features={"exit_code": 1, "arbitrary_shell": "rm -rf"})
    )
    assert "unknown_input_field" in reasons
    with pytest.raises(UnknownFieldError):
        spec.input_schema.reject_unknown(
            {"exit_code": 1, "explanation": "long prose"}, noun="compact_features"
        )


def test_unsupported_family_risk_pairs_reject() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert spec.admits_risk(RiskClass.R2)
    assert spec.admits_risk(RiskClass.R0)
    with pytest.raises(ResidualIntelligenceError, match=REASON_RISK_CEILING):
        assert_family_risk_admitted(ResidualTaskFamily.FAILURE_ATTRIBUTION, RiskClass.R3)
    with pytest.raises(ResidualIntelligenceError, match=REASON_RISK_CEILING):
        assert_family_risk_admitted(ResidualTaskFamily.TASK_CLASSIFICATION, RiskClass.R5)
    with pytest.raises(ResidualIntelligenceError, match=REASON_RISK_CEILING):
        assert_family_risk_admitted(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING, RiskClass.R0)
    disposition, reasons = smallest_expert_for(
        ResidualTaskFamily.FAILURE_ATTRIBUTION
    ).evaluate_input(task_input(risk=RiskClass.R4))
    assert disposition is ExpertDisposition.REJECT_INPUT
    assert REASON_RISK_CEILING in reasons


def test_validator_is_required_and_non_abstained_output_needs_evidence() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    expert = smallest_expert_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert_validator_required(spec)
    assert_validator_required(expert)
    with pytest.raises(ResidualIntelligenceError, match=REASON_VALIDATOR_REQUIRED):
        spec.validate_task_output(
            accepted_output(evidence=())
        )
    spec.validate_task_output(accepted_output())
    disposition, reasons = expert.evaluate_input(task_input())
    assert disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert REASON_VALIDATOR_REQUIRED in reasons
    with pytest.raises(ResidualIntelligenceError, match="validator"):
        replace(spec, validator_required=False)
    with pytest.raises(ResidualIntelligenceError, match="non-empty"):
        replace(spec, validation_contract="")


def test_prose_is_not_the_default_output() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    expert = smallest_expert_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert spec.emit_prose_by_default is False
    assert expert.emit_prose_by_default is False
    assert spec.prose_token_budget == 0
    assert expert.prose_token_budget == 0
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROSE_DEFAULT):
        replace(spec, emit_prose_by_default=True)
    with pytest.raises(ResidualIntelligenceError, match="prose_token_budget"):
        replace(expert, prose_token_budget=128)
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROSE_DEFAULT):
        spec.validate_task_output(
            accepted_output(payload={
                "failure_class": "missing_dependency_edge",
                "recommended_action": "expand_context_reference",
                "prose": "because it looks right",
            })
        )


def test_canonical_round_trip_and_grammar_limits() -> None:
    for family in ResidualTaskFamily:
        spec = family_spec_for(family)
        rebuilt = ResidualTaskFamilySpec.from_dict(spec.to_dict())
        assert rebuilt == spec
        assert rebuilt.family_spec_id == spec.family_spec_id
        grammar = grammar_for(family)
        assert spec.maximum_output_bytes == grammar.maximum_output_bytes
        assert set(spec.output_schema.fields) == set(grammar.payload_fields)
        expert = smallest_expert_for(family)
        assert ResidualExpertSpec.from_dict(expert.to_dict()) == expert
        assert expert.grammar_id == grammar.grammar_id
        assert expert.input_token_limit == spec.input_token_limit
        assert expert.output_token_limit == spec.output_token_limit
        assert expert.candidate_only is True
        assert expert.implementation_form == EXPERT_CLASS_FORMS[ExpertClass.A]


def test_novel_unbounded_reasoning_always_abstains() -> None:
    spec = family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    expert = smallest_expert_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert spec.always_abstain is True
    assert spec.allowed_expert_classes == ("A",)
    assert spec.privacy_route_policy == "human_review_only"
    assert spec.risk_floor is RiskClass.R5
    assert spec.risk_ceiling is RiskClass.R5
    disposition, reasons = expert.evaluate_input(
        task_input(
            family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            risk=RiskClass.R5,
            features={"reason_code": "outside_taxonomy"},
        )
    )
    assert disposition is ExpertDisposition.ABSTAIN
    assert REASON_NOVEL_UNBOUNDED in reasons
    spec.validate_task_output(
        accepted_output(
            family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            abstained=True,
            reason_codes=("novel_unbounded_reasoning",),
        )
    )
    with pytest.raises(ResidualIntelligenceError, match=REASON_NOVEL_UNBOUNDED):
        spec.validate_task_output(
            ResidualTaskOutput(
                output_class="ABSTAIN",
                structured_payload={},
                confidence_or_score=1,
                calibration_group="novel:unbounded",
                abstained=False,
                reason_codes=(),
                evidence_references=("validator:none",),
                candidate_only=True,
            )
        )


def test_r4_r5_remain_proposal_tier() -> None:
    sketch = family_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)
    lemma = family_spec_for(ResidualTaskFamily.LEMMA_SUGGESTION)
    unbounded = family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert sketch.proposal_tier is True
    assert lemma.proposal_tier is True
    assert unbounded.proposal_tier is True
    assert family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).proposal_tier is False
    payload = {
        "files": ["ipfs_accelerate_py/mod.py"],
        "symbol_ids": ["mod.fn"],
        "operations": ["replace_span"],
        "maximum_changed_lines": 8,
        "validation_ids": ["validator:patch@1"],
    }
    with pytest.raises(ResidualIntelligenceError, match="validation-required"):
        sketch.validate_task_output(
            ResidualTaskOutput(
                output_class="PATCH_SKETCH",
                structured_payload=payload,
                confidence_or_score=1,
                calibration_group="patch:python:R4:fixture",
                abstained=False,
                reason_codes=(),
                evidence_references=("validator:patch@1",),
                candidate_only=True,
            )
        )
    sketch.validate_task_output(
        ResidualTaskOutput(
            output_class="PATCH_SKETCH",
            structured_payload=payload,
            confidence_or_score=1,
            calibration_group="patch:python:R4:fixture",
            abstained=False,
            reason_codes=("VALIDATION_REQUIRED",),
            evidence_references=("validator:patch@1",),
            candidate_only=True,
        )
    )


def test_dataset_reference_requires_admitted_corpus() -> None:
    record, _examples = admission()
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    with pytest.raises(ResidualIntelligenceError, match="no dataset reference"):
        spec.bind_corpus_admission(record)
    bound = replace(spec, evaluation_corpus_admission_id=record.admission_id)
    bound.bind_corpus_admission(record)
    expert = replace(
        smallest_expert_for(ResidualTaskFamily.FAILURE_ATTRIBUTION),
        evaluation_corpus_admission_id=record.admission_id,
    )
    expert.bind_corpus_admission(record)
    unavailable, _examples = admission(admitted=False)
    blocked = replace(spec, evaluation_corpus_admission_id=unavailable.admission_id)
    with pytest.raises(ResidualIntelligenceError, match="admitted"):
        blocked.bind_corpus_admission(unavailable)


def test_class_mismatch_and_capability_gates() -> None:
    expert = smallest_expert_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    disposition, reasons = expert.evaluate_input(
        task_input(family=ResidualTaskFamily.TASK_CLASSIFICATION)
    )
    assert disposition is ExpertDisposition.REJECT_INPUT
    assert "task_family_mismatch" in reasons
    unavailable, reasons = expert.evaluate_input(task_input(), capability_available=False)
    assert unavailable is ExpertDisposition.CAPABILITY_UNAVAILABLE
    larger = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION, ExpertClass.C)
    blocked, larger_reasons = larger.evaluate_input(task_input())
    assert blocked is ExpertDisposition.REJECT_INPUT
    assert REASON_LARGER_FORM in larger_reasons
    oversized, token_reasons = expert.evaluate_input(task_input(token_budget=30_000))
    assert oversized is ExpertDisposition.REJECT_INPUT
    assert "token_limit_exceeded" in token_reasons
    assert larger.expert_id in {item.expert_id for item in EXPERT_SPECS.values()}
