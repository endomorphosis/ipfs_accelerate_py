from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.cascade import (
    CASCADE_ORDER,
    CASCADE_POLICY_VERSION,
    DETERMINISTIC_STAGES,
    REASON_EXACT_CACHE,
    REASON_HUMAN_FALLBACK,
    REASON_PRIVACY,
    REASON_PROVIDER_HEALTH,
    REASON_SAFE_FALLBACK,
    REASON_SIMULATION,
    REASON_VALIDATION,
    REMOTE_STAGES,
    CascadeCandidate,
    CascadeHardRejection,
    CascadeStage,
    ResidualCascade,
    ResidualCascadeWalk,
    stage_is_deterministic,
    stage_is_learned,
    stage_is_remote,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    UnknownFieldError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.residual_ir import ResidualTaskInput
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.router import (
    DEFAULT_ROUTER,
    ResidualExpertRouter,
    ResidualRouteDecision,
    ResidualRouteRequest,
    route,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.task_families import (
    family_spec_for,
)

LOCAL_HARDWARE = (
    "cpu-small-hermetic",
    "cpu-small-batch",
    "cpu-medium-batch",
    "cpu-gpu-optional-bounded",
)
REMOTE_HARDWARE = (*LOCAL_HARDWARE, "provider-standard", "provider-strong", "human-reviewer")


def task_input(
    *,
    family: ResidualTaskFamily = ResidualTaskFamily.FAILURE_ATTRIBUTION,
    risk: RiskClass = RiskClass.R2,
    features: dict[str, object] | None = None,
    allowed: tuple[str, ...] | None = None,
    token_budget: int = 256,
) -> ResidualTaskInput:
    family_spec = family_spec_for(family)
    if features is None:
        defaults: dict[ResidualTaskFamily, dict[str, object]] = {
            ResidualTaskFamily.FAILURE_ATTRIBUTION: {
                "exit_code": 1,
                "failure_signature": "missing-edge",
            },
            ResidualTaskFamily.PROOF_SELECTION: {"obligation_id": "obligation:1"},
            ResidualTaskFamily.PATCH_SKETCH_GENERATION: {"allowed_paths": ["src/mod.py"]},
            ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING: {
                "unbounded_reason": "left-closed-taxonomy"
            },
            ResidualTaskFamily.RETRY_OR_ESCALATE: {"attempt_disposition": "ABSTAIN"},
        }
        features = defaults.get(family, {"label_candidates": ["a"]})
    outputs = allowed if allowed is not None else family_spec.output_classes
    return ResidualTaskInput(
        task_family=family,
        question_id="question:router:1",
        repository_state_cid="repo:tree:abc",
        objective_cid="objective:vrif",
        task_cid="task:VRIF-013",
        policy_cid="policy:residual-v1",
        context_capsule_cid="capsule:bounded:1",
        compact_features=features,
        allowed_outputs=outputs,
        risk_class=risk,
        validation_policy=family_spec.validation_contract,
        token_budget=token_budget,
    )


def route_request(**overrides: object) -> ResidualRouteRequest:
    payload: dict[str, object] = {
        "task_input": task_input(),
        "privacy_class": PrivacyClass.REPOSITORY_PRIVATE,
        "hardware_available": LOCAL_HARDWARE,
        "validation_available": True,
    }
    payload.update(overrides)
    return ResidualRouteRequest(**payload)  # type: ignore[arg-type]


def decide(**overrides: object) -> ResidualRouteDecision:
    return route(route_request(**overrides))


def rejection_for(decision: ResidualRouteDecision, stage: CascadeStage) -> CascadeHardRejection:
    for item in decision.hard_rejections:
        if item.stage is stage:
            return item
    raise AssertionError(f"missing hard rejection for {stage.value}")


def candidate_for(decision: ResidualRouteDecision, stage: CascadeStage) -> CascadeCandidate:
    for item in decision.candidates:
        if item.stage is stage:
            return item
    raise AssertionError(f"missing candidate for {stage.value}")


def test_cascade_order_is_deterministic_and_procedure_first() -> None:
    assert CASCADE_ORDER == (
        CascadeStage.EXACT_CACHE,
        CascadeStage.VERIFIED_PROCEDURE,
        CascadeStage.DETERMINISTIC_RULE,
        CascadeStage.LOCAL_LINEAR_EXPERT,
        CascadeStage.LOCAL_RANKER,
        CascadeStage.LOCAL_STRUCTURED_SPECIALIST,
        CascadeStage.LOCAL_GENERAL_MODEL,
        CascadeStage.REMOTE_STANDARD_MODEL,
        CascadeStage.REMOTE_STRONG_MODEL,
        CascadeStage.HUMAN_REVIEW,
    )
    assert tuple(item.value for item in CASCADE_ORDER[:3]) == (
        "exact_cache",
        "verified_procedure",
        "deterministic_rule",
    )
    cascade = ResidualCascade()
    rebuilt = ResidualCascade.from_dict(cascade.to_dict())
    assert rebuilt.cascade_id == cascade.cascade_id
    assert rebuilt.stages == CASCADE_ORDER
    assert cascade.policy_version == CASCADE_POLICY_VERSION


def test_exact_cache_precedes_procedure_and_learned_routes() -> None:
    decision = decide(
        cache_hit=True,
        cache_identity="cache:failure:1",
        procedure_available=True,
        procedure_preconditions_satisfied=True,
        procedure_root="procedure:failure-attribution@1",
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
        local_linear_available=True,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.selected_stage is CascadeStage.EXACT_CACHE
    assert decision.disposition is ExpertDisposition.ACCEPT
    assert REASON_EXACT_CACHE in decision.reason_codes
    assert candidate_for(decision, CascadeStage.VERIFIED_PROCEDURE).stage is CascadeStage.VERIFIED_PROCEDURE
    assert candidate_for(decision, CascadeStage.DETERMINISTIC_RULE)
    assert decision.fallback_stage is CascadeStage.HUMAN_REVIEW
    assert decision.candidate_only is True


def test_procedure_precedes_deterministic_rule_when_cache_misses() -> None:
    decision = decide(
        procedure_available=True,
        procedure_preconditions_satisfied=True,
        procedure_root="procedure:failure-attribution@1",
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
        local_linear_available=True,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.selected_stage is CascadeStage.VERIFIED_PROCEDURE
    assert "verified_procedure" in decision.reason_codes
    assert "procedure:failure-attribution@1" in decision.evidence_references
    assert rejection_for(decision, CascadeStage.EXACT_CACHE).reason_codes[0] == "cache_miss"


def test_procedure_precondition_failure_does_not_skip_recording() -> None:
    decision = decide(
        procedure_available=True,
        procedure_preconditions_satisfied=False,
        procedure_root="procedure:failure-attribution@1",
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
    )
    rejected = rejection_for(decision, CascadeStage.VERIFIED_PROCEDURE)
    assert "procedure_precondition_failure" in rejected.reason_codes
    assert decision.selected_stage is CascadeStage.DETERMINISTIC_RULE


def test_deterministic_rule_precedes_local_specialists() -> None:
    decision = decide(
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
        local_linear_available=True,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.selected_stage is CascadeStage.DETERMINISTIC_RULE
    assert decision.disposition is ExpertDisposition.ACCEPT
    assert candidate_for(decision, CascadeStage.LOCAL_LINEAR_EXPERT)
    assert all(stage_is_deterministic(item) for item in DETERMINISTIC_STAGES)


def test_local_linear_precedes_remote_when_deterministic_stages_fail() -> None:
    decision = decide(
        local_linear_available=True,
        remote_standard_available=True,
        remote_strong_available=True,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=4096),
    )
    assert decision.selected_stage is CascadeStage.LOCAL_LINEAR_EXPERT
    assert decision.disposition is ExpertDisposition.ACCEPT
    assert candidate_for(decision, CascadeStage.REMOTE_STANDARD_MODEL)
    assert candidate_for(decision, CascadeStage.REMOTE_STRONG_MODEL)


def test_remote_standard_precedes_remote_strong() -> None:
    decision = decide(
        remote_standard_available=True,
        remote_strong_available=True,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=4096),
    )
    assert decision.selected_stage is CascadeStage.REMOTE_STANDARD_MODEL
    assert candidate_for(decision, CascadeStage.REMOTE_STRONG_MODEL)
    assert stage_is_remote(decision.selected_stage)


def test_human_fallback_when_no_earlier_stage_is_eligible() -> None:
    decision = decide()
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert decision.disposition is ExpertDisposition.ABSTAIN
    assert REASON_HUMAN_FALLBACK in decision.reason_codes
    assert REASON_SAFE_FALLBACK in decision.reason_codes
    assert decision.fallback_stage is CascadeStage.HUMAN_REVIEW
    assert {item.stage for item in decision.candidates} == {CascadeStage.HUMAN_REVIEW}
    assert len(decision.hard_rejections) == 9


def test_human_fallback_remains_reachable_when_marked_unavailable() -> None:
    decision = decide(human_review_available=False)
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert candidate_for(decision, CascadeStage.HUMAN_REVIEW)


def test_all_candidates_and_hard_rejections_cover_the_cascade() -> None:
    decision = decide(
        cache_hit=True,
        cache_identity="cache:failure:1",
        local_linear_available=True,
        expected_decision_value_microunits=1_000_000,
    )
    recorded = [item.stage for item in decision.candidates] + [
        item.stage for item in decision.hard_rejections
    ]
    assert len(recorded) == len(CASCADE_ORDER)
    assert set(recorded) == set(CASCADE_ORDER)
    assert all(item.candidate_only is True for item in decision.candidates)
    assert decision.walk_id
    walk = ResidualCascadeWalk.from_dict(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/residual-cascade-walk@1",
            "policy_version": decision.cascade_policy_version,
            "family": decision.family.value,
            "risk_class": decision.risk_class.value,
            "candidates": [item.to_dict() for item in decision.candidates],
            "hard_rejections": [item.to_dict() for item in decision.hard_rejections],
            "selected_stage": decision.selected_stage.value,
            "fallback_stage": "human_review",
            "candidate_only": True,
        }
    )
    assert walk.selected_stage is CascadeStage.EXACT_CACHE


def test_unsupported_family_risk_is_a_hard_rejection() -> None:
    decision = decide(task_input=task_input(risk=RiskClass.R5))
    assert decision.disposition is ExpertDisposition.REJECT_INPUT
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert "risk_ceiling_exceeded" in decision.reason_codes
    for stage in CASCADE_ORDER[:-1]:
        assert "risk_ceiling_exceeded" in rejection_for(decision, stage).reason_codes


def test_always_abstain_family_never_selects_a_specialist() -> None:
    decision = decide(
        task_input=task_input(
            family=ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
            risk=RiskClass.R5,
        ),
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        cache_hit=True,
        cache_identity="cache:unbounded:1",
        local_linear_available=True,
        remote_standard_available=True,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert "always_abstain_family" in rejection_for(decision, CascadeStage.EXACT_CACHE).reason_codes
    assert decision.risk_class is RiskClass.R5
    assert decision.disposition is not ExpertDisposition.ACCEPT


def test_privacy_route_rejects_unauthorized_remote() -> None:
    decision = decide(
        remote_standard_available=True,
        remote_strong_available=True,
        provider_authorized=False,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=4096),
    )
    remote = rejection_for(decision, CascadeStage.REMOTE_STANDARD_MODEL)
    assert REASON_PRIVACY in remote.reason_codes or "private_to_unauthorized_provider" in remote.reason_codes
    assert "provider_unauthorized" in remote.reason_codes
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW


def test_local_only_privacy_never_selects_a_remote_route() -> None:
    decision = decide(
        task_input=task_input(
            family=ResidualTaskFamily.PROOF_SELECTION,
            risk=RiskClass.R4,
            token_budget=4096,
        ),
        privacy_class=PrivacyClass.PROOF_WITNESS,
        local_ranker_available=True,
        remote_standard_available=True,
        remote_strong_available=True,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.privacy_class is PrivacyClass.PROOF_WITNESS
    for stage in REMOTE_STAGES:
        reasons = rejection_for(decision, stage).reason_codes
        assert REASON_PRIVACY in reasons
    assert decision.selected_stage is CascadeStage.LOCAL_RANKER
    assert decision.disposition is ExpertDisposition.VALIDATION_REQUIRED


def test_hardware_unavailable_rejects_local_learned_stages() -> None:
    decision = decide(
        local_linear_available=True,
        hardware_available=("cpu-small-hermetic",),
        expected_decision_value_microunits=1_000_000,
    )
    rejected = rejection_for(decision, CascadeStage.LOCAL_LINEAR_EXPERT)
    assert "hardware_unavailable" in rejected.reason_codes
    assert rejected.constraint == "hardware"
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW


def test_provider_unhealthy_rejects_remote_routes() -> None:
    decision = decide(
        remote_standard_available=True,
        provider_authorized=True,
        provider_healthy=False,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=4096),
    )
    rejected = rejection_for(decision, CascadeStage.REMOTE_STANDARD_MODEL)
    assert REASON_PROVIDER_HEALTH in rejected.reason_codes
    assert rejected.constraint == "provider"
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW


def test_simulation_is_never_a_live_route() -> None:
    decision = decide(
        simulated=True,
        cache_hit=True,
        cache_identity="cache:failure:1",
        procedure_available=True,
        procedure_preconditions_satisfied=True,
        procedure_root="procedure:failure-attribution@1",
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
        local_linear_available=True,
        remote_standard_available=True,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=4096),
    )
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert decision.disposition is ExpertDisposition.ABSTAIN
    for stage in CASCADE_ORDER[:-1]:
        assert REASON_SIMULATION in rejection_for(decision, stage).reason_codes
    assert all(item.stage is CascadeStage.HUMAN_REVIEW for item in decision.candidates)


def test_capability_inferred_from_importability_is_rejected() -> None:
    decision = decide(
        local_linear_available=True,
        capability_inferred_from_importability=True,
        expected_decision_value_microunits=1_000_000,
    )
    rejected = rejection_for(decision, CascadeStage.LOCAL_LINEAR_EXPERT)
    assert "capability_inferred_from_importability" in rejected.reason_codes
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW


def test_absent_validation_rejects_producing_stages() -> None:
    decision = decide(
        cache_hit=True,
        cache_identity="cache:failure:1",
        validation_available=False,
    )
    rejected = rejection_for(decision, CascadeStage.EXACT_CACHE)
    assert REASON_VALIDATION in rejected.reason_codes
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert decision.validation_required is True
    assert decision.validator_identity.startswith("validator:")


def test_r4_r5_preserves_validation_required() -> None:
    decision = decide(
        task_input=task_input(
            family=ResidualTaskFamily.PATCH_SKETCH_GENERATION,
            risk=RiskClass.R4,
            token_budget=1024,
        ),
        privacy_class=PrivacyClass.REPOSITORY_PRIVATE,
        local_structured_available=True,
        hardware_available=LOCAL_HARDWARE,
        expected_decision_value_microunits=1_000_000,
    )
    assert decision.selected_stage is CascadeStage.LOCAL_STRUCTURED_SPECIALIST
    assert decision.disposition is ExpertDisposition.VALIDATION_REQUIRED
    assert decision.validation_required is True
    assert "VALIDATION_REQUIRED" in decision.reason_codes
    assert decision.candidate_only is True


def test_budget_and_expected_decision_value_filter_learned_routes() -> None:
    cheap_blocked = decide(
        local_linear_available=True,
        expected_decision_value_microunits=10,
        hardware_available=LOCAL_HARDWARE,
    )
    assert "expected_decision_value_insufficient" in rejection_for(
        cheap_blocked, CascadeStage.LOCAL_LINEAR_EXPERT
    ).reason_codes
    budget_blocked = decide(
        local_linear_available=True,
        expected_decision_value_microunits=1_000_000,
        cost_budget_microunits=1,
        hardware_available=LOCAL_HARDWARE,
    )
    assert "token_or_cost_budget_exceeded" in rejection_for(
        budget_blocked, CascadeStage.LOCAL_LINEAR_EXPERT
    ).reason_codes
    token_blocked = decide(
        remote_standard_available=True,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        hardware_available=REMOTE_HARDWARE,
        expected_decision_value_microunits=1_000_000,
        task_input=task_input(token_budget=16),
    )
    assert "token_or_cost_budget_exceeded" in rejection_for(
        token_blocked, CascadeStage.REMOTE_STANDARD_MODEL
    ).reason_codes


def test_candidate_evidence_is_required_for_cache_and_procedure() -> None:
    cache = decide(cache_hit=True)
    assert "candidate_evidence_required" in rejection_for(
        cache, CascadeStage.EXACT_CACHE
    ).reason_codes
    procedure = decide(
        procedure_available=True,
        procedure_preconditions_satisfied=True,
    )
    assert "candidate_evidence_required" in rejection_for(
        procedure, CascadeStage.VERIFIED_PROCEDURE
    ).reason_codes


def test_ood_conservative_abstain_blocks_learned_not_deterministic() -> None:
    decision = decide(
        deterministic_rule_available=True,
        rule_identity="rule:missing-edge",
        local_linear_available=True,
        remote_standard_available=True,
        ood_conservative_abstain=True,
        expected_decision_value_microunits=1_000_000,
        hardware_available=REMOTE_HARDWARE,
        provider_authorized=True,
        provider_healthy=True,
        inference_policy_permits_remote=True,
        task_input=task_input(token_budget=4096),
    )
    assert decision.selected_stage is CascadeStage.DETERMINISTIC_RULE
    assert "ood_conservative_abstain" in rejection_for(
        decision, CascadeStage.LOCAL_LINEAR_EXPERT
    ).reason_codes
    assert stage_is_learned(CascadeStage.LOCAL_LINEAR_EXPERT)


def test_classification_family_rejects_ranker_and_structured_classes() -> None:
    decision = decide(
        local_ranker_available=True,
        local_structured_available=True,
        expected_decision_value_microunits=1_000_000,
        hardware_available=LOCAL_HARDWARE,
    )
    assert "unsupported_family_class" in rejection_for(
        decision, CascadeStage.LOCAL_RANKER
    ).reason_codes
    assert "unsupported_family_class" in rejection_for(
        decision, CascadeStage.LOCAL_STRUCTURED_SPECIALIST
    ).reason_codes


def test_local_execution_unavailable_is_capability_unavailable() -> None:
    decision = decide(
        cache_hit=True,
        cache_identity="cache:failure:1",
        local_execution_available=False,
    )
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert decision.disposition is ExpertDisposition.CAPABILITY_UNAVAILABLE
    assert "local_execution_unavailable" in rejection_for(
        decision, CascadeStage.EXACT_CACHE
    ).reason_codes


def test_route_decision_and_request_round_trip() -> None:
    request = route_request(
        cache_hit=True,
        cache_identity="cache:failure:1",
        evidence_references=("evidence:1",),
        calibration_group_key="failure:python:R2:fixture",
    )
    rebuilt_request = ResidualRouteRequest.from_dict(request.to_dict())
    assert rebuilt_request.request_id == request.request_id
    decision = DEFAULT_ROUTER.route(request)
    rebuilt = ResidualRouteDecision.from_dict(decision.to_dict())
    assert rebuilt == decision
    assert rebuilt.decision_id == decision.decision_id
    router = ResidualExpertRouter()
    assert ResidualExpertRouter.from_dict(router.to_dict()).router_id == router.router_id


def test_unknown_fields_and_candidate_only_are_rejected() -> None:
    request = route_request(cache_hit=True, cache_identity="cache:failure:1")
    payload = request.to_dict()
    payload["shadow_live"] = True
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualRouteRequest.from_dict(payload)
    decision = route(request)
    decision_payload = decision.to_dict()
    decision_payload["promotion"] = True
    with pytest.raises(UnknownFieldError, match="unknown fields"):
        ResidualRouteDecision.from_dict(decision_payload)
    decision_payload = decision.to_dict()
    decision_payload["candidate_only"] = False
    with pytest.raises(ResidualIntelligenceError, match="candidate_only"):
        ResidualRouteDecision.from_dict(decision_payload)
    decision_payload = decision.to_dict()
    decision_payload["validation_required"] = False
    with pytest.raises(ResidualIntelligenceError, match="required validation"):
        ResidualRouteDecision.from_dict(decision_payload)
    decision_payload = decision.to_dict()
    decision_payload["fallback_stage"] = "remote_strong_model"
    with pytest.raises(ResidualIntelligenceError, match="human_review"):
        ResidualRouteDecision.from_dict(decision_payload)


def test_cascade_rejects_reordered_stage_tables() -> None:
    payload = ResidualCascade().to_dict()
    payload["stages"] = list(reversed(payload["stages"]))
    with pytest.raises(ResidualIntelligenceError, match="admitted production order"):
        ResidualCascade.from_dict(payload)


def test_family_mismatch_is_recorded_as_reject_input() -> None:
    mismatched = task_input(
        family=ResidualTaskFamily.TASK_CLASSIFICATION,
        features={"exit_code": 1, "failure_signature": "missing-edge"},
    )
    decision = decide(task_input=mismatched, cache_hit=True, cache_identity="cache:1")
    assert decision.disposition is ExpertDisposition.REJECT_INPUT
    assert decision.selected_stage is CascadeStage.HUMAN_REVIEW
    assert "family_out_of_bound" in decision.reason_codes
    assert rejection_for(decision, CascadeStage.EXACT_CACHE).constraint in {"family", "input"}
