"""ASEH-020: canonical nine-stage receipt-to-human decision ladder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    CANONICAL_ROUTING_AUTHORITY,
    DETERMINISTIC_LADDER_STAGES,
    HUMAN_LADDER_STAGE,
    MODEL_LADDER_STAGES,
    MODEL_ROUTING_INTERFACE,
    RECEIPT_TO_HUMAN_LADDER,
    ConfidenceClass,
    DeterministicEvidenceState,
    LadderDecision,
    LadderEvidence,
    LadderStage,
    ModelRoutingPolicy,
    RiskClass,
    RoutingInputs,
    StageAction,
    StageRunReason,
    StageSkipReason,
    evaluate_ladder_for_routing_inputs,
    evaluate_receipt_to_human_ladder,
    ladder_evidence_from_routing_inputs,
    ladder_stage_to_model_route,
    model_routing_descriptor,
    route_model,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    ModelRoute as VerificationModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    CANONICAL_LADDER_AUTHORITY,
    RECEIPT_TO_HUMAN_LADDER as MODEL_ROUTE_LADDER,
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePolicy,
    PriorRepairAttempt,
    RiskLevel,
    decide_model_route,
    default_inventory,
    evaluate_ladder_for_model_route,
    evaluate_receipt_to_human_ladder as model_route_evaluate_ladder,
    ladder_evidence_from_verification,
    policy_cid_for,
    select_required_route,
)
from ipfs_accelerate_py.agent_supervisor.verification import model_route as model_route_module


ORDERED_STAGE_VALUES = (
    "exact_current_authoritative_cached_receipt",
    "ast_symbol_dependency_and_impact_analysis",
    "schema_type_static_lint_and_contract_checks",
    "selected_tests",
    "incremental_smt_or_theorem_prover",
    "local_small_specialist_model",
    "local_or_remote_medium_model",
    "remote_strong_or_frontier_model",
    "human_decision",
)


def _evidence(**overrides: object) -> LadderEvidence:
    payload: dict[str, Any] = {
        "cached_receipt": DeterministicEvidenceState.UNAVAILABLE.value,
        "ast_symbol_impact": DeterministicEvidenceState.UNAVAILABLE.value,
        "schema_type_static": DeterministicEvidenceState.UNAVAILABLE.value,
        "selected_tests": DeterministicEvidenceState.UNAVAILABLE.value,
        "incremental_prover": DeterministicEvidenceState.UNAVAILABLE.value,
        "small_model_required": False,
        "medium_model_required": False,
        "frontier_model_required": False,
        "small_model_available": False,
        "medium_model_available": False,
        "frontier_model_available": False,
        "human_review_required": False,
    }
    payload.update(overrides)
    return LadderEvidence.from_dict(payload)


def _routing_inputs(**overrides: object) -> RoutingInputs:
    payload: dict[str, Any] = {
        "context_tokens": 512,
        "lowest_confidence": ConfidenceClass.EXACT.value,
        "risk": RiskClass.LOW.value,
        "dependency_cone_size": 2,
        "unresolved_obligations": 0,
        "prior_repair_failures": 0,
        "available_proofs": 1,
        "prior_route_failed": False,
    }
    payload.update(overrides)
    return RoutingInputs.from_dict(payload)


def _facts(**kwargs: Any) -> ModelRouteFacts:
    base: dict[str, Any] = {
        "context_token_estimate": 2_048,
        "analysis_kind": AnalysisKind.LOCALIZED_EXACT,
        "opaque_dependency_count": 0,
        "risk_level": RiskLevel.LOW,
        "dependency_cone_size": 2,
        "changed_file_count": 1,
        "counterexample_quality": CounterexampleQuality.MINIMIZED,
        "exact_contract_available": True,
        "environment_reproducible": True,
    }
    base.update(kwargs)
    return ModelRouteFacts(**base)


def _assert_full_ordered_traversal(decision: LadderDecision) -> None:
    assert tuple(item.stage for item in decision.stages) == ORDERED_STAGE_VALUES
    assert len(decision.stages) == 9
    resolved = [item for item in decision.stages if item.resolved]
    assert len(resolved) == 1
    winner = resolved[0]
    assert winner.action == StageAction.RUN.value
    assert winner.stage == decision.selected_stage
    seen = False
    for item in decision.stages:
        if item.resolved:
            seen = True
            continue
        if seen:
            assert item.action == StageAction.SKIP.value
            assert item.reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
        else:
            assert item.reason in {
                *(reason.value for reason in StageRunReason),
                *(reason.value for reason in StageSkipReason),
            }
            assert item.reason != StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value


def test_canonical_authority_is_existing_model_routing_interface() -> None:
    assert CANONICAL_ROUTING_AUTHORITY == MODEL_ROUTING_INTERFACE
    assert CANONICAL_LADDER_AUTHORITY == CANONICAL_ROUTING_AUTHORITY
    assert model_route_module.CANONICAL_LADDER_AUTHORITY == MODEL_ROUTING_INTERFACE
    descriptor = model_routing_descriptor()
    assert descriptor["canonical_authority"] == MODEL_ROUTING_INTERFACE
    assert descriptor["receipt_to_human_ladder"] == list(ORDERED_STAGE_VALUES)


def test_model_route_consumes_the_same_ladder_and_is_not_a_second_authority() -> None:
    assert MODEL_ROUTE_LADDER is RECEIPT_TO_HUMAN_LADDER
    assert model_route_evaluate_ladder is evaluate_receipt_to_human_ladder
    assert model_route_module.RECEIPT_TO_HUMAN_LADDER is RECEIPT_TO_HUMAN_LADDER
    source = model_route_module.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    for stage in ORDERED_STAGE_VALUES:
        if stage == "selected_tests":
            continue
        assert f'"{stage}"' not in text
        assert f"'{stage}'" not in text
    assert "from ..semantic_state.routing import" in text
    assert "RECEIPT_TO_HUMAN_LADDER" in text


def test_ladder_has_exact_nine_stages_in_required_order() -> None:
    assert len(RECEIPT_TO_HUMAN_LADDER) == 9
    assert tuple(stage.value for stage in RECEIPT_TO_HUMAN_LADDER) == ORDERED_STAGE_VALUES
    assert tuple(stage.value for stage in DETERMINISTIC_LADDER_STAGES) == ORDERED_STAGE_VALUES[:5]
    assert tuple(stage.value for stage in MODEL_LADDER_STAGES) == ORDERED_STAGE_VALUES[5:8]
    assert HUMAN_LADDER_STAGE is LadderStage.HUMAN_DECISION
    assert [stage.value for stage in LadderStage] == list(ORDERED_STAGE_VALUES)


@pytest.mark.parametrize(
    ("field", "stage"),
    (
        ("cached_receipt", LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT),
        ("ast_symbol_impact", LadderStage.AST_SYMBOL_DEPENDENCY_AND_IMPACT_ANALYSIS),
        ("schema_type_static", LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS),
        ("selected_tests", LadderStage.SELECTED_TESTS),
        ("incremental_prover", LadderStage.INCREMENTAL_SMT_OR_THEOREM_PROVER),
    ),
)
def test_deterministic_evidence_resolves_before_any_model_stage(
    field: str, stage: LadderStage
) -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            **{
                field: DeterministicEvidenceState.RESOLVES.value,
                "small_model_required": True,
                "small_model_available": True,
                "medium_model_available": True,
                "frontier_model_available": True,
                "frontier_model_required": False,
            }
        )
    )
    _assert_full_ordered_traversal(decision)
    assert decision.selected_stage == stage.value
    assert decision.selected_route == ModelRoute.DETERMINISTIC_ONLY.value
    assert decision.resolved_by_deterministic_evidence is True
    run = [item for item in decision.stages if item.stage == stage.value][0]
    assert run.action == StageAction.RUN.value
    assert run.reason == StageRunReason.ELIGIBLE_DETERMINISTIC_EVIDENCE.value
    for model_stage in MODEL_LADDER_STAGES:
        record = [item for item in decision.stages if item.stage == model_stage.value][0]
        assert record.action == StageAction.SKIP.value
        assert record.reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
        assert record.resolved is False


def test_earlier_deterministic_stage_wins_when_later_stages_also_resolve() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            cached_receipt=DeterministicEvidenceState.RESOLVES.value,
            ast_symbol_impact=DeterministicEvidenceState.RESOLVES.value,
            schema_type_static=DeterministicEvidenceState.RESOLVES.value,
            selected_tests=DeterministicEvidenceState.RESOLVES.value,
            incremental_prover=DeterministicEvidenceState.RESOLVES.value,
            frontier_model_required=True,
            frontier_model_available=True,
        )
    )
    assert (
        decision.selected_stage
        == LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT.value
    )
    assert decision.resolved_by_deterministic_evidence is True
    skipped_after = [
        item.stage
        for item in decision.stages
        if item.reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
    ]
    assert skipped_after == list(ORDERED_STAGE_VALUES[1:])


def test_unresolved_deterministic_stage_is_run_then_later_stage_may_resolve() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            cached_receipt=DeterministicEvidenceState.UNRESOLVED.value,
            ast_symbol_impact=DeterministicEvidenceState.UNAVAILABLE.value,
            schema_type_static=DeterministicEvidenceState.RESOLVES.value,
        )
    )
    cache = decision.stages[0]
    assert cache.action == StageAction.RUN.value
    assert cache.reason == StageRunReason.UNRESOLVED_DETERMINISTIC_EVIDENCE.value
    assert cache.resolved is False
    assert (
        decision.selected_stage
        == LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS.value
    )


def test_frontier_availability_cannot_skip_selected_tests_or_prover() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            selected_tests=DeterministicEvidenceState.UNAVAILABLE.value,
            incremental_prover=DeterministicEvidenceState.RESOLVES.value,
            frontier_model_available=True,
            medium_model_available=True,
            small_model_available=True,
        )
    )
    tests = decision.stages[3]
    prover = decision.stages[4]
    frontier = decision.stages[7]
    assert tests.action == StageAction.SKIP.value
    assert tests.reason == StageSkipReason.EVIDENCE_UNAVAILABLE.value
    assert prover.action == StageAction.RUN.value
    assert prover.resolved is True
    assert frontier.reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
    assert decision.selected_route == ModelRoute.DETERMINISTIC_ONLY.value


def test_unavailable_required_small_model_does_not_escalate_to_available_frontier() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            small_model_required=True,
            small_model_available=False,
            medium_model_available=True,
            frontier_model_available=True,
        )
    )
    _assert_full_ordered_traversal(decision)
    small = decision.stages[5]
    medium = decision.stages[6]
    frontier = decision.stages[7]
    human = decision.stages[8]
    assert small.action == StageAction.SKIP.value
    assert small.reason == StageSkipReason.REQUIRED_TIER_UNAVAILABLE.value
    assert medium.reason == StageSkipReason.NOT_SMALLEST_ADEQUATE_EXECUTOR.value
    assert frontier.reason == StageSkipReason.NOT_SMALLEST_ADEQUATE_EXECUTOR.value
    assert human.action == StageAction.RUN.value
    assert human.reason == StageRunReason.HUMAN_DECISION_REQUIRED.value
    assert decision.selected_route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert decision.resolved_by_deterministic_evidence is False


def test_smallest_adequate_model_runs_only_after_deterministic_stages() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            small_model_required=True,
            small_model_available=True,
            frontier_model_available=True,
        )
    )
    for record in decision.stages[:5]:
        assert record.action == StageAction.SKIP.value
        assert record.reason == StageSkipReason.EVIDENCE_UNAVAILABLE.value
    small = decision.stages[5]
    assert small.action == StageAction.RUN.value
    assert small.reason == StageRunReason.SMALLEST_ADEQUATE_EXECUTOR.value
    assert decision.selected_route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert decision.stages[6].reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value
    assert decision.stages[7].reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value


def test_human_gate_skips_models_but_still_traverses_deterministic_stages() -> None:
    decision = evaluate_receipt_to_human_ladder(
        _evidence(
            cached_receipt=DeterministicEvidenceState.INELIGIBLE.value,
            ast_symbol_impact=DeterministicEvidenceState.INELIGIBLE.value,
            schema_type_static=DeterministicEvidenceState.INELIGIBLE.value,
            selected_tests=DeterministicEvidenceState.INELIGIBLE.value,
            incremental_prover=DeterministicEvidenceState.INELIGIBLE.value,
            small_model_required=True,
            small_model_available=True,
            frontier_model_available=True,
            human_review_required=True,
        )
    )
    _assert_full_ordered_traversal(decision)
    for record in decision.stages[:5]:
        assert record.reason == StageSkipReason.EVIDENCE_INELIGIBLE.value
    for record in decision.stages[5:8]:
        assert record.reason == StageSkipReason.HUMAN_GATE_PREEMPTS_MODEL.value
    assert decision.selected_stage == LadderStage.HUMAN_DECISION.value
    assert decision.selected_route == ModelRoute.HUMAN_REVIEW_REQUIRED.value


def test_empty_evidence_fail_closes_to_human_without_availability_fallback() -> None:
    decision = evaluate_receipt_to_human_ladder(LadderEvidence())
    _assert_full_ordered_traversal(decision)
    assert decision.selected_route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    for record in decision.stages[5:8]:
        assert record.reason == StageSkipReason.MODEL_NOT_REQUIRED.value
        assert record.action == StageAction.SKIP.value


def test_two_model_tiers_required_fails_closed() -> None:
    with pytest.raises(HarnessError, match="at most one model tier"):
        _evidence(small_model_required=True, frontier_model_required=True)


def test_unknown_ladder_evidence_fields_fail_closed() -> None:
    payload = _evidence().to_dict()
    payload["prefer_frontier_because_available"] = True
    with pytest.raises(HarnessError, match="fields must be exactly"):
        LadderEvidence.from_dict(payload)
    payload = _evidence().to_dict()
    payload["skip_to_available_model"] = True
    with pytest.raises(HarnessError):
        LadderEvidence.from_dict(payload)


def test_ladder_decision_round_trip_is_closed_and_deterministic() -> None:
    first = evaluate_receipt_to_human_ladder(
        _evidence(selected_tests=DeterministicEvidenceState.RESOLVES.value)
    )
    second = evaluate_receipt_to_human_ladder(first.evidence.to_dict())
    restored = LadderDecision.from_dict(first.to_dict())
    assert first.to_dict() == second.to_dict() == restored.to_dict()
    assert first.selected_route == ModelRoute.DETERMINISTIC_ONLY.value


def test_route_model_always_traverses_the_canonical_ladder() -> None:
    cases = (
        _routing_inputs(),
        _routing_inputs(risk=RiskClass.HIGH.value),
        _routing_inputs(
            context_tokens=8_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.MEDIUM.value,
            available_proofs=0,
        ),
        _routing_inputs(
            context_tokens=50_000,
            lowest_confidence=ConfidenceClass.HEURISTIC.value,
            risk=RiskClass.MEDIUM.value,
            dependency_cone_size=40,
            unresolved_obligations=2,
            available_proofs=0,
        ),
    )
    for inputs in cases:
        decision = route_model(inputs)
        ladder = evaluate_ladder_for_routing_inputs(
            inputs, ModelRoutingPolicy.default()
        )
        _assert_full_ordered_traversal(ladder)
        assert ladder.selected_route == decision.route
        if decision.route == ModelRoute.DETERMINISTIC_ONLY.value:
            assert ladder.resolved_by_deterministic_evidence is True
            assert any(
                item.reason == StageRunReason.ELIGIBLE_DETERMINISTIC_EVIDENCE.value
                for item in ladder.stages
            )


def test_proof_covered_routing_inputs_resolve_on_cached_receipt_before_models() -> None:
    inputs = _routing_inputs(
        unresolved_obligations=1,
        available_proofs=1,
        lowest_confidence=ConfidenceClass.EXACT.value,
        risk=RiskClass.LOW.value,
    )
    decision = route_model(inputs)
    evidence = ladder_evidence_from_routing_inputs(
        inputs, ModelRoutingPolicy.default()
    )
    ladder = evaluate_receipt_to_human_ladder(evidence)
    assert decision.route == ModelRoute.DETERMINISTIC_ONLY.value
    assert (
        ladder.selected_stage
        == LadderStage.EXACT_CURRENT_AUTHORITATIVE_CACHED_RECEIPT.value
    )
    assert evidence.frontier_model_available is False
    assert evidence.small_model_required is False


def test_normally_no_large_model_examples_resolve_deterministically() -> None:
    examples = {
        "exact_cache_freshness": "cached_receipt",
        "dependency_cone_selection": "ast_symbol_impact",
        "schema_validity": "schema_type_static",
        "type_validity": "schema_type_static",
        "known_test_selection": "selected_tests",
        "proof_receipt_identity_matching": "incremental_prover",
        "unchanged_documentation_only_path": "ast_symbol_impact",
        "stale_tree_detection": "cached_receipt",
        "current_lease_and_fence_validity": "cached_receipt",
        "policy_pointer_cas_status": "schema_type_static",
        "exact_deterministic_migration": "schema_type_static",
        "duplicate_retry_without_new_evidence": "cached_receipt",
    }
    for _name, field in examples.items():
        decision = evaluate_receipt_to_human_ladder(
            _evidence(
                **{
                    field: DeterministicEvidenceState.RESOLVES.value,
                    "frontier_model_available": True,
                    "medium_model_available": True,
                    "small_model_available": True,
                }
            )
        )
        assert decision.selected_route == ModelRoute.DETERMINISTIC_ONLY.value
        assert decision.resolved_by_deterministic_evidence is True
        for model_stage in MODEL_LADDER_STAGES:
            record = [item for item in decision.stages if item.stage == model_stage.value][0]
            assert record.reason == StageSkipReason.RESOLVED_BY_PRIOR_STAGE.value


def test_verification_adapter_agrees_with_canonical_ladder_for_mechanical_work() -> None:
    policy = ModelRoutePolicy(policy_cid=policy_cid_for("aseh-020-mechanical"))
    facts = _facts(
        analysis_kind=AnalysisKind.MECHANICAL_FORMATTING,
        counterexample_quality=CounterexampleQuality.NONE,
    )
    inventory = default_inventory()
    decision = decide_model_route(
        facts, available_models=inventory, policy=policy
    )
    ladder = evaluate_ladder_for_model_route(
        facts, available_models=inventory, policy=policy
    )
    _assert_full_ordered_traversal(ladder)
    assert decision.route is VerificationModelRoute.DETERMINISTIC_ONLY
    assert ladder.selected_route == ModelRoute.DETERMINISTIC_ONLY.value
    assert (
        ladder.selected_stage
        == LadderStage.SCHEMA_TYPE_STATIC_LINT_AND_CONTRACT_CHECKS.value
    )
    assert ladder.resolved_by_deterministic_evidence is True


def test_verification_unavailable_small_tier_matches_ladder_human_without_upgrade() -> None:
    policy = ModelRoutePolicy(policy_cid=policy_cid_for("aseh-020-no-upgrade"))
    facts = _facts(
        analysis_kind=AnalysisKind.LOCALIZED_EXACT,
        counterexample_quality=CounterexampleQuality.GOOD,
    )
    inventory = default_inventory(
        include_deterministic=True, small=False, medium=True, frontier=True
    )
    decision = decide_model_route(
        facts, available_models=inventory, policy=policy
    )
    required, _reasons = select_required_route(facts, (), policy)
    evidence = ladder_evidence_from_verification(
        facts, required_route=required, inventory=inventory, policy=policy
    )
    ladder = evaluate_receipt_to_human_ladder(evidence)
    assert required is VerificationModelRoute.SMALL_LOCAL_MODEL
    assert decision.route is VerificationModelRoute.HUMAN_REVIEW_REQUIRED
    assert ladder.selected_route == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    assert evidence.small_model_required is True
    assert evidence.small_model_available is False
    assert evidence.frontier_model_available is True
    assert ladder.stages[5].reason == StageSkipReason.REQUIRED_TIER_UNAVAILABLE.value
    assert ladder.stages[7].reason == StageSkipReason.NOT_SMALLEST_ADEQUATE_EXECUTOR.value


def test_verification_failed_smaller_route_still_walks_every_stage() -> None:
    policy = ModelRoutePolicy(policy_cid=policy_cid_for("aseh-020-failed-small"))
    facts = _facts(
        analysis_kind=AnalysisKind.LOCALIZED_EXACT,
        counterexample_quality=CounterexampleQuality.MINIMIZED,
    )
    prior = [PriorRepairAttempt(route=VerificationModelRoute.SMALL_LOCAL_MODEL, failed=True)]
    decision = decide_model_route(
        facts,
        prior_attempts=prior,
        available_models=default_inventory(),
        policy=policy,
    )
    ladder = evaluate_ladder_for_model_route(
        facts,
        prior_attempts=prior,
        available_models=default_inventory(),
        policy=policy,
    )
    _assert_full_ordered_traversal(ladder)
    assert decision.route.value == ladder.selected_route
    assert decision.route is not VerificationModelRoute.SMALL_LOCAL_MODEL
    assert ladder.stages[5].action == StageAction.SKIP.value


def test_stage_to_route_mapping_is_closed() -> None:
    assert {
        ladder_stage_to_model_route(stage) for stage in DETERMINISTIC_LADDER_STAGES
    } == {ModelRoute.DETERMINISTIC_ONLY.value}
    assert (
        ladder_stage_to_model_route(LadderStage.LOCAL_SMALL_SPECIALIST_MODEL)
        == ModelRoute.SMALL_LOCAL_MODEL.value
    )
    assert (
        ladder_stage_to_model_route(LadderStage.LOCAL_OR_REMOTE_MEDIUM_MODEL)
        == ModelRoute.MEDIUM_MODEL.value
    )
    assert (
        ladder_stage_to_model_route(LadderStage.REMOTE_STRONG_OR_FRONTIER_MODEL)
        == ModelRoute.FRONTIER_MODEL.value
    )
    assert (
        ladder_stage_to_model_route(LadderStage.HUMAN_DECISION)
        == ModelRoute.HUMAN_REVIEW_REQUIRED.value
    )
    with pytest.raises(HarnessError, match="unsupported ladder stage"):
        ladder_stage_to_model_route("skip_because_frontier_available")
