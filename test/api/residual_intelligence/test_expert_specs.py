from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    DEFAULT_EXPERT_SPECS,
    EXPERT_CLASS_FORMS,
    SMALLEST_FORM_ORDER,
    ExpertClass,
    ModelSizePolicy,
    ResidualExpertSpec,
    assert_family_risk_allowed,
    expert_spec_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.inventory import (
    ResidualFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import (
    ResidualTaskInput,
    ResidualTaskOutput,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DEFAULT_GRAMMARS,
    grammar_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import (
    AUTHORITY_CLASS,
    DEFAULT_FAMILY_SPECS,
    ResidualTaskFamilySpec,
    family_spec_for,
)

from .helpers import admission


def failure_input(**overrides: object) -> ResidualTaskInput:
    payload: dict[str, object] = {
        "task_family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "question_id": "question:failure:1",
        "repository_state_cid": "repo:tree:abc",
        "objective_cid": "objective:vrif",
        "task_cid": "task:VRIF-010",
        "policy_cid": "policy:residual-v1",
        "context_capsule_cid": "capsule:bounded:1",
        "compact_features": {"exit_code": 1, "failure_signature": "missing-edge"},
        "allowed_outputs": ("FAILURE_ATTRIBUTION", "ABSTAIN"),
        "risk_class": RiskClass.R2,
        "validation_policy": "validator:failure-attribution@1",
        "token_budget": 256,
    }
    payload.update(overrides)
    return ResidualTaskInput(**payload)  # type: ignore[arg-type]


def test_every_taxonomy_family_has_exact_semantic_boundary() -> None:
    assert set(DEFAULT_FAMILY_SPECS) == set(ResidualTaskFamily)
    assert set(DEFAULT_EXPERT_SPECS) == set(ResidualTaskFamily)
    assert set(DEFAULT_GRAMMARS) == set(ResidualTaskFamily)
    seen_boundaries: set[str] = set()
    for family in ResidualTaskFamily:
        spec = family_spec_for(family)
        expert = expert_spec_for(family)
        grammar = grammar_for(family)
        assert spec.task_family is family
        assert spec.authority_class == AUTHORITY_CLASS
        assert spec.validator_required is True
        assert spec.validation_contract
        assert spec.error_behavior
        assert spec.abstention_behavior
        assert spec.input_semantics
        assert spec.output_semantics
        assert spec.grammar_id == grammar.grammar_id
        assert spec.allowed_outputs == grammar.output_classes
        assert spec.max_output_bytes == grammar.maximum_output_bytes
        boundary = spec.to_boundary()
        assert isinstance(boundary, ResidualFamilyBoundary)
        assert boundary.task_family is family
        assert boundary.boundary_id not in seen_boundaries
        seen_boundaries.add(boundary.boundary_id)
        assert expert.family_spec_id == spec.family_spec_id
        assert expert.grammar_id == grammar.grammar_id
        rebuilt = ResidualTaskFamilySpec.from_dict(spec.to_dict())
        assert rebuilt == spec
        assert rebuilt.family_spec_id == spec.family_spec_id


def test_prompt_similarity_cannot_override_family_boundary() -> None:
    first = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    second = family_spec_for(ResidualTaskFamily.RETRY_OR_ESCALATE)
    assert first.family_spec_id != second.family_spec_id
    assert first.to_boundary().boundary_id != second.to_boundary().boundary_id
    with pytest.raises(ResidualIntelligenceError, match="authority_class"):
        ResidualTaskFamilySpec.from_dict(
            {**first.to_dict(include_id=False), "authority_class": "authoritative"}
        )


def test_expert_classes_are_a_through_e() -> None:
    assert tuple(item.value for item in ExpertClass) == ("A", "B", "C", "D", "E")
    assert SMALLEST_FORM_ORDER == (
        ExpertClass.A,
        ExpertClass.B,
        ExpertClass.C,
        ExpertClass.D,
        ExpertClass.E,
    )
    assert EXPERT_CLASS_FORMS == {
        ExpertClass.A: "exact_lookup",
        ExpertClass.B: "declarative_rule",
        ExpertClass.C: "linear_logistic",
        ExpertClass.D: "ranker_encoder",
        ExpertClass.E: "constrained_structured_decoder",
    }
    observed = {item.expert_class for item in DEFAULT_EXPERT_SPECS.values()}
    assert observed == set(ExpertClass)
    assert tuple(ModelSizePolicy) == (
        ModelSizePolicy.NONE,
        ModelSizePolicy.LINEAR,
        ModelSizePolicy.SMALL_RANKER,
        ModelSizePolicy.STRUCTURED_SPECIALIST,
        ModelSizePolicy.PARAMETER_EFFICIENT_ADAPTER,
        ModelSizePolicy.QUANTIZED_LOCAL_GENERAL,
        ModelSizePolicy.REMOTE_STANDARD,
        ModelSizePolicy.REMOTE_STRONG,
    )


def test_smallest_form_order_requires_routing_changing_quality_delta() -> None:
    preferred = expert_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    assert preferred.expert_class is ExpertClass.A
    assert preferred.form == "exact_lookup"
    assert preferred.model_size_policy is ModelSizePolicy.NONE
    with pytest.raises(ResidualIntelligenceError, match="routing-changing quality delta"):
        expert_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION, ExpertClass.C)
    larger = expert_spec_for(
        ResidualTaskFamily.TASK_CLASSIFICATION,
        ExpertClass.C,
        routing_changing_quality_delta_ppm=50_000,
        held_out_evidence_current=True,
    )
    assert larger.expert_class is ExpertClass.C
    assert larger.form == "linear_logistic"
    assert larger.model_size_policy is ModelSizePolicy.LINEAR
    with pytest.raises(ResidualIntelligenceError, match="model size"):
        expert_spec_for(
            ResidualTaskFamily.TASK_CLASSIFICATION,
            ExpertClass.D,
            routing_changing_quality_delta_ppm=50_000,
            held_out_evidence_current=True,
        )
    with pytest.raises(ResidualIntelligenceError, match="expert class"):
        expert_spec_for(
            ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            ExpertClass.B,
            routing_changing_quality_delta_ppm=50_000,
            held_out_evidence_current=True,
        )


def test_closed_schemas_reject_unknown_fields() -> None:
    spec = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    payload = spec.to_dict()
    payload["prompt_embedding"] = "similar-looking-family"
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualTaskFamilySpec.from_dict(payload)
    expert = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    expert_payload = expert.to_dict()
    expert_payload["arbitrary_shell"] = "rm -rf elsewhere"
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualExpertSpec.from_dict(expert_payload)
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        expert.validate_input(
            failure_input(compact_features={"exit_code": 1, "prose_rationale": "because"})
        )
    rebuilt = ResidualExpertSpec.from_dict(expert.to_dict())
    assert rebuilt == expert
    assert rebuilt.expert_spec_id == expert.expert_spec_id


def test_unsupported_family_risk_pairs_are_rejected() -> None:
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk"):
        assert_family_risk_allowed(ResidualTaskFamily.FAILURE_ATTRIBUTION, RiskClass.R5)
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk"):
        assert_family_risk_allowed(ResidualTaskFamily.PROOF_SELECTION, RiskClass.R1)
    with pytest.raises(ResidualIntelligenceError, match="unsupported family-risk"):
        expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).validate_input(
            failure_input(risk_class=RiskClass.R4)
        )
    allowed = assert_family_risk_allowed(ResidualTaskFamily.FAILURE_ATTRIBUTION, RiskClass.R2)
    assert allowed.risk_ceiling is RiskClass.R2
    proof = family_spec_for(ResidualTaskFamily.PROOF_SELECTION)
    assert proof.risk_floor is RiskClass.R4
    assert proof.risk_ceiling is RiskClass.R5
    proof.assert_risk_allowed(RiskClass.R5)


def test_validator_is_required_and_prose_is_not_default() -> None:
    for spec in DEFAULT_FAMILY_SPECS.values():
        assert spec.validator_required is True
        assert spec.emits_prose_by_default is False
        assert spec.input_schema.allow_prose is False
        assert spec.output_schema.allow_prose is False
        assert spec.evaluation_dataset_reference == ""
        assert spec.training_dataset_reference == ""
        assert "examples" not in spec.to_dict()
    expert = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert expert.validator_required is True
    assert expert.candidate_only is True
    assert expert.emits_prose_by_default is False
    with pytest.raises(ResidualIntelligenceError, match="validator"):
        replace(expert, validator_required=False)
    with pytest.raises(ResidualIntelligenceError, match="prose"):
        replace(expert, emits_prose_by_default=True)
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        replace(expert, candidate_only=False)


def test_dataset_reference_requires_admitted_training_corpus() -> None:
    record, _examples = admission(admitted=True)
    with pytest.raises(ResidualIntelligenceError, match="admitted TrainingCorpusAdmission"):
        expert_spec_for(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            evaluation_dataset_reference=record.corpus_root,
        )
    bound = expert_spec_for(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        evaluation_admission=record,
        evaluation_dataset_reference=record.corpus_root,
    )
    assert bound.evaluation_dataset_reference == record.corpus_root
    withheld, _examples = admission(admitted=False)
    with pytest.raises(ResidualIntelligenceError, match="admitted TrainingCorpusAdmission"):
        expert_spec_for(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            evaluation_admission=withheld,
            evaluation_dataset_reference=withheld.corpus_root,
        )


def test_novel_unbounded_reasoning_always_abstains() -> None:
    spec = family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    expert = expert_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert spec.allowed_outputs == ("ABSTAIN",)
    assert spec.allowed_expert_classes == (ExpertClass.A,)
    assert spec.model_size_ceiling is ModelSizePolicy.NONE
    assert "novel_unbounded" in spec.abstention_codes
    output = ResidualTaskOutput(
        output_class="ABSTAIN",
        structured_payload={},
        confidence_or_score=0,
        calibration_group="novel:R5",
        abstained=True,
        reason_codes=("novel_unbounded",),
        evidence_references=(),
    )
    expert.validate_output(output)
    with pytest.raises(ResidualIntelligenceError, match="outside the closed expert schema"):
        expert.validate_output(
            ResidualTaskOutput(
                output_class="FAILURE_ATTRIBUTION",
                structured_payload={"failure_class": "unknown"},
                confidence_or_score=1,
                calibration_group="novel:R5",
                abstained=False,
                reason_codes=("VALIDATION_REQUIRED",),
                evidence_references=(),
            )
        )


def test_high_risk_outputs_remain_validation_required() -> None:
    expert = expert_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION)
    assert expert.proposal_tier is True
    payload = {
        "files": ["ipfs_accelerate_py/module.py"],
        "symbol_ids": ["symbol:1"],
        "operations": ["replace_span"],
        "maximum_changed_lines": 8,
        "validation_ids": ["validator:1"],
    }
    with pytest.raises(ResidualIntelligenceError, match="R4/R5"):
        expert.validate_output(
            ResidualTaskOutput(
                output_class="PATCH_SKETCH",
                structured_payload=payload,
                confidence_or_score=100,
                calibration_group="patch:R5",
                abstained=False,
                reason_codes=(),
                evidence_references=(),
            )
        )
    expert.validate_output(
        ResidualTaskOutput(
            output_class="PATCH_SKETCH",
            structured_payload=payload,
            confidence_or_score=100,
            calibration_group="patch:R5",
            abstained=False,
            reason_codes=("VALIDATION_REQUIRED",),
            evidence_references=("validator:patch-sketch@1",),
        )
    )


def test_family_input_round_trip_uses_closed_validator_and_limits() -> None:
    expert = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    task_input = failure_input()
    expert.validate_input(task_input)
    with pytest.raises(ResidualIntelligenceError, match="validation_policy"):
        expert.validate_input(failure_input(validation_policy="validator:other@1"))
    with pytest.raises(ResidualIntelligenceError, match="token_budget"):
        expert.validate_input(failure_input(token_budget=100_000))
