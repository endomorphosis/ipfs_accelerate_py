from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.expert_specs import (
    DEFAULT_MODEL_SIZE_POLICY,
    EXPERT_CLASS_FORMS,
    EXPERT_SPECS,
    MIN_ROUTING_CHANGING_DELTA_PPM,
    SMALLEST_FORM_ORDER,
    ExpertClass,
    ModelSizePolicy,
    ResidualExpertSpec,
    admit_expert_class,
    all_expert_specs,
    expert_spec_for,
    expert_specs_for_family,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.inventory import ResidualFamilyBoundary
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.structured_decoding import (
    DEFAULT_GRAMMARS,
    grammar_for,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import (
    ABSTAIN_OUTPUT_CLASS,
    CANDIDATE_ONLY_AUTHORITY,
    ERROR_INVALID_OUTPUT,
    FAMILY_SPECS,
    REASON_PROSE_DEFAULT,
    REASON_UNSUPPORTED_FAMILY_RISK,
    REASON_VALIDATOR_REQUIRED,
    ResidualTaskFamilySpec,
    all_family_specs,
    family_spec_for,
    reject_unsupported_family_risk,
)

from .helpers import admission


def failure_input(**overrides: object) -> ResidualTaskInput:
    payload: dict[str, object] = {
        "task_family": ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "question_id": "question:failure:spec",
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


def test_every_taxonomy_family_has_explicit_semantics_and_gates() -> None:
    specs = all_family_specs()
    assert len(specs) == len(ResidualTaskFamily)
    assert set(FAMILY_SPECS) == set(ResidualTaskFamily)
    assert set(DEFAULT_GRAMMARS) == set(ResidualTaskFamily)
    seen_semantics: set[tuple[str, str]] = set()
    for spec in specs:
        assert spec.authority_class == CANDIDATE_ONLY_AUTHORITY
        assert spec.candidate_only is True
        assert spec.error_behavior == ERROR_INVALID_OUTPUT
        assert spec.independent_validator_required is True
        assert spec.validator_identity.startswith("validator:")
        assert spec.emit_prose_by_default is False
        assert ABSTAIN_OUTPUT_CLASS in spec.output_classes
        assert spec.abstention_behavior
        assert spec.input_semantics
        assert spec.output_semantics
        pair = (spec.input_semantics, spec.output_semantics)
        assert pair not in seen_semantics
        seen_semantics.add(pair)
        grammar = grammar_for(spec.task_family)
        assert spec.output_classes == grammar.output_classes
        assert spec.maximum_output_bytes == grammar.maximum_output_bytes
        boundary = spec.to_family_boundary()
        assert isinstance(boundary, ResidualFamilyBoundary)
        assert boundary.task_family is spec.task_family
        assert boundary.validation_contract == spec.validation_contract
        assert boundary.error_behavior == spec.error_behavior
        assert boundary.abstention_behavior == spec.abstention_behavior


def test_exact_family_boundary_rejects_prompt_similarity_grouping() -> None:
    left = family_spec_for(ResidualTaskFamily.TASK_CLASSIFICATION)
    right = family_spec_for(ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION)
    assert left.semantic_kind == right.semantic_kind
    assert left.to_family_boundary().boundary_id != right.to_family_boundary().boundary_id
    assert left.input_semantics != right.input_semantics
    assert left.output_semantics != right.output_semantics
    mismatched = failure_input(task_family=ResidualTaskFamily.TASK_CLASSIFICATION)
    with pytest.raises(ResidualIntelligenceError, match="task_family_mismatch"):
        family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).validate_task_input(mismatched)


def test_class_a_through_e_exist_in_smallest_form_order() -> None:
    assert SMALLEST_FORM_ORDER == (
        ExpertClass.A,
        ExpertClass.B,
        ExpertClass.C,
        ExpertClass.D,
        ExpertClass.E,
    )
    assert tuple(item.value for item in ExpertClass) == ("A", "B", "C", "D", "E")
    assert set(EXPERT_CLASS_FORMS) == set(ExpertClass)
    assert EXPERT_CLASS_FORMS[ExpertClass.A] == ("exact_lookup",)
    assert "verified_procedure" in EXPERT_CLASS_FORMS[ExpertClass.B]
    assert EXPERT_CLASS_FORMS[ExpertClass.C] == ("linear_logistic",)
    assert EXPERT_CLASS_FORMS[ExpertClass.D] == ("small_ranker",)
    assert EXPERT_CLASS_FORMS[ExpertClass.E] == ("constrained_structured_decoder",)
    policy = DEFAULT_MODEL_SIZE_POLICY
    assert policy.form_order == SMALLEST_FORM_ORDER
    assert policy.allow_global_bigger_is_better is False
    assert policy.allow_skip_smaller_form is False
    rebuilt = ModelSizePolicy.from_dict(policy.to_dict())
    assert rebuilt.policy_id == policy.policy_id


def test_closed_schemas_reject_unknown_fields_and_examples() -> None:
    family = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    family_payload = family.to_dict()
    family_payload["prompt_embedding"] = [0.1, 0.2]
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualTaskFamilySpec.from_dict(family_payload)
    family_payload = family.to_dict()
    family_payload["examples"] = [{"input": "prose"}]
    with pytest.raises(UnknownFieldError, match="specifications_carry_no_examples"):
        ResidualTaskFamilySpec.from_dict(family_payload)
    expert = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    expert_payload = expert.to_dict()
    expert_payload["training_examples"] = ["secret-body"]
    with pytest.raises(UnknownFieldError, match="specifications_carry_no_examples"):
        ResidualExpertSpec.from_dict(expert_payload)
    family.validate_task_input(failure_input())
    with pytest.raises(ResidualIntelligenceError, match="unknown_compact_feature"):
        family.validate_task_input(failure_input(compact_features={"exit_code": 1, "prose": "no"}))


def test_risk_ceiling_rejects_unsupported_family_risk_pairs() -> None:
    family = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert family.allows_risk(RiskClass.R2)
    family.reject_unsupported_risk(RiskClass.R3)
    with pytest.raises(ResidualIntelligenceError, match=REASON_UNSUPPORTED_FAMILY_RISK):
        family.reject_unsupported_risk(RiskClass.R5)
    with pytest.raises(ResidualIntelligenceError, match=REASON_UNSUPPORTED_FAMILY_RISK):
        reject_unsupported_family_risk(
            ResidualTaskFamily.PATCH_SKETCH_GENERATION,
            RiskClass.R1,
        )
    with pytest.raises(ResidualIntelligenceError, match=REASON_UNSUPPORTED_FAMILY_RISK):
        family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING).reject_unsupported_risk(
            RiskClass.R0
        )
    with pytest.raises(ResidualIntelligenceError, match=REASON_UNSUPPORTED_FAMILY_RISK):
        family.validate_task_input(failure_input(risk_class=RiskClass.R5))


def test_validator_required_and_no_prose_default() -> None:
    family = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    payload = family.to_dict(include_id=False)
    payload["independent_validator_required"] = False
    with pytest.raises(ResidualIntelligenceError, match=REASON_VALIDATOR_REQUIRED):
        ResidualTaskFamilySpec.from_dict(payload)
    payload = family.to_dict(include_id=False)
    payload["emit_prose_by_default"] = True
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROSE_DEFAULT):
        ResidualTaskFamilySpec.from_dict(payload)
    expert_payload = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).to_dict(include_id=False)
    expert_payload["independent_validator_required"] = False
    with pytest.raises(ResidualIntelligenceError, match=REASON_VALIDATOR_REQUIRED):
        ResidualExpertSpec.from_dict(expert_payload)
    expert_payload = expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).to_dict(include_id=False)
    expert_payload["emit_prose_by_default"] = True
    with pytest.raises(ResidualIntelligenceError, match=REASON_PROSE_DEFAULT):
        ResidualExpertSpec.from_dict(expert_payload)


def test_larger_form_requires_routing_changing_quality_delta() -> None:
    smallest = admit_expert_class(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        ExpertClass.A,
        risk=RiskClass.R2,
    )
    assert smallest.expert_class is ExpertClass.A
    with pytest.raises(ResidualIntelligenceError, match="routing-changing quality delta"):
        admit_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.B,
            risk=RiskClass.R2,
            evidence_current=True,
        )
    with pytest.raises(ResidualIntelligenceError, match="current held-out evidence"):
        admit_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.B,
            risk=RiskClass.R2,
            quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
            routing_changing=True,
            evidence_current=False,
        )
    admitted, _examples = admission(admitted=True)
    larger = admit_expert_class(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        ExpertClass.B,
        risk=RiskClass.R2,
        quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
        routing_changing=True,
        evidence_current=True,
        admission=admitted,
    )
    assert larger.expert_class is ExpertClass.B
    assert larger.forms == EXPERT_CLASS_FORMS[ExpertClass.B]
    with pytest.raises(ResidualIntelligenceError, match="smallest-form-order"):
        admit_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.C,
            risk=RiskClass.R2,
            quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
            routing_changing=True,
            evidence_current=True,
            admission=admitted,
        )
    with pytest.raises(ResidualIntelligenceError, match="unsupported_family_class"):
        admit_expert_class(
            ResidualTaskFamily.FAILURE_ATTRIBUTION,
            ExpertClass.E,
            risk=RiskClass.R2,
            quality_delta_ppm=MIN_ROUTING_CHANGING_DELTA_PPM,
            routing_changing=True,
            evidence_current=True,
            compared_class=ExpertClass.D,
            admission=admitted,
        )


def test_structured_family_can_advance_one_class_with_held_out_delta() -> None:
    admitted, _examples = admission(admitted=True)
    spec = admit_expert_class(
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        ExpertClass.E,
        risk=RiskClass.R4,
        quality_delta_ppm=25_000,
        routing_changing=True,
        evidence_current=True,
        compared_class=ExpertClass.D,
        admission=admitted,
    )
    assert spec.expert_class is ExpertClass.E
    assert spec.risk_ceiling is RiskClass.R5
    assert spec.maximum_output_bytes == grammar_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION).maximum_output_bytes
    blocked, _examples = admission(admitted=False)
    with pytest.raises(ResidualIntelligenceError, match="training_unavailable"):
        admit_expert_class(
            ResidualTaskFamily.PATCH_SKETCH_GENERATION,
            ExpertClass.B,
            risk=RiskClass.R4,
            quality_delta_ppm=25_000,
            routing_changing=True,
            evidence_current=True,
            admission=blocked,
        )


def test_every_eligible_family_class_has_a_closed_expert_spec() -> None:
    specs = all_expert_specs()
    assert specs
    assert {item.expert_class for item in specs} == set(ExpertClass)
    families_with_e = {
        item.task_family for item in specs if item.expert_class is ExpertClass.E
    }
    assert ResidualTaskFamily.PROCEDURE_HOLE_FILLING in families_with_e
    assert ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING not in families_with_e
    for spec in specs:
        rebuilt = ResidualExpertSpec.from_dict(spec.to_dict())
        assert rebuilt == spec
        assert rebuilt.spec_id == spec.spec_id
        assert spec.candidate_only is True
        assert spec.independent_validator_required is True
        assert spec.emit_prose_by_default is False
        assert spec.abstention_output_class == ABSTAIN_OUTPUT_CLASS
        assert spec.error_behavior == ERROR_INVALID_OUTPUT
        assert spec.grammar().task_family is spec.task_family
        assert spec.family_boundary().authority_class == CANDIDATE_ONLY_AUTHORITY
        assert "provider_free" in spec.runtime_requirements
        assert spec.evaluation_corpus_admission_id == ""
    assert expert_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION).expert_class is ExpertClass.A
    assert [item.expert_class for item in expert_specs_for_family(ResidualTaskFamily.EVIDENCE_RANKING)] == [
        ExpertClass.A,
        ExpertClass.B,
        ExpertClass.C,
        ExpertClass.D,
    ]


def test_unbounded_family_always_abstains_at_class_a() -> None:
    family = family_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert family.always_abstain is True
    assert family.output_classes == (ABSTAIN_OUTPUT_CLASS,)
    assert family.eligible_expert_classes == ("A",)
    spec = expert_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING)
    assert spec.expert_class is ExpertClass.A
    assert spec.always_abstain is True
    with pytest.raises(ResidualIntelligenceError, match="unsupported_family_class"):
        expert_spec_for(ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING, ExpertClass.E)
    allowed = ResidualTaskInput(
        task_family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
        question_id="question:unbounded:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-010",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features={"unbounded_reason": "open-ended-request"},
        allowed_outputs=("ABSTAIN",),
        risk_class=RiskClass.R5,
        validation_policy="validator:novel-unbounded-reasoning@1",
        token_budget=64,
    )
    family.validate_task_input(allowed)
    with pytest.raises(ResidualIntelligenceError, match="output_class_outside_family_grammar"):
        family.validate_task_input(
            ResidualTaskInput(
                task_family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
                question_id="question:unbounded:2",
                repository_state_cid="repo:tree:abc",
                objective_cid="objective:vrif",
                task_cid="task:VRIF-010",
                policy_cid="policy:residual-v1",
                context_capsule_cid="capsule:bounded:1",
                compact_features={"unbounded_reason": "open-ended-request"},
                allowed_outputs=("NOVEL_UNBOUNDED_REASONING", "ABSTAIN"),
                risk_class=RiskClass.R5,
                validation_policy="validator:novel-unbounded-reasoning@1",
                token_budget=64,
            )
        )


def test_privacy_route_keeps_proof_witness_local() -> None:
    proof = family_spec_for(ResidualTaskFamily.PROOF_SELECTION)
    assert proof.privacy_route_policy == "local_only"
    assert proof.privacy_route_permits(provider_authorized=False, local_execution=True) is True
    assert proof.privacy_route_permits(provider_authorized=True, local_execution=False) is False
    ordinary = family_spec_for(ResidualTaskFamily.FAILURE_ATTRIBUTION)
    assert ordinary.privacy_route_permits(provider_authorized=True, local_execution=False) is True
    assert ordinary.privacy_route_permits(provider_authorized=False, local_execution=False) is False


def test_r4_r5_remain_candidate_only_and_validator_bound() -> None:
    sketch = expert_spec_for(ResidualTaskFamily.PATCH_SKETCH_GENERATION, ExpertClass.A)
    assert sketch.risk_ceiling is RiskClass.R5
    assert sketch.candidate_only is True
    assert sketch.independent_validator_required is True
    lemma = expert_spec_for(ResidualTaskFamily.LEMMA_SUGGESTION)
    assert lemma.privacy_route_policy == "local_only"
    assert lemma.always_abstain is False


def test_global_bigger_is_better_policy_is_rejected() -> None:
    with pytest.raises(ResidualIntelligenceError, match="bigger-is-better"):
        ModelSizePolicy(allow_global_bigger_is_better=True)
    with pytest.raises(ResidualIntelligenceError, match="smallest-form-order"):
        ModelSizePolicy(allow_skip_smaller_form=True)


def test_family_and_expert_spec_round_trip_and_registry_coverage() -> None:
    for family in ResidualTaskFamily:
        original = family_spec_for(family)
        rebuilt = ResidualTaskFamilySpec.from_dict(original.to_dict())
        assert rebuilt == original
        assert rebuilt.spec_id == original.spec_id
        for spec in expert_specs_for_family(family):
            assert (family, spec.expert_class) in EXPERT_SPECS
            assert spec.family_spec_id == original.spec_id
            assert spec.family_boundary_id == original.to_family_boundary().boundary_id
