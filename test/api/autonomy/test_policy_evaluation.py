from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    MetaAction,
    PolicyObservation,
    RoutePolicyCandidate,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.policy_evaluation import (
    INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,
    PROTECTED_POLICY_AXES,
    ROUTE_POLICY_EVALUATION_INTERFACE,
    ComparisonEvidence,
    EvaluationDisposition,
    EvaluationPartition,
    LoggedDecision,
    PairingKind,
    PolicyEvaluationError,
    RoutePolicyEvaluation,
    evaluate_route_policy,
    field_is_forbidden,
    route_policy_evaluation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _candidate(**overrides: Any) -> RoutePolicyCandidate:
    values: dict[str, Any] = {
        "parent_policy_id": "policy-baseline",
        "policy_version": "route-v1",
        "allowed_actions": (
            MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
            MetaAction.RUN_SELECTED_TEST,
            MetaAction.CALL_REMOTE_STRONG_MODEL,
        ),
        "feature_names": ("risk_rank", "context_sufficient", "token_cost"),
        "integer_weights": {"risk_rank": -2, "context_sufficient": 5, "token_cost": -1},
        "training_observation_ids": ("observation-seed",),
        "held_out_evaluation_ids": ("evaluation-1",),
        "safety_gate_receipt_ids": ("safety-1",),
        "selection_reason": "linear_score",
    }
    values.update(overrides)
    return RoutePolicyCandidate(**values)


def _observation(
    *,
    episode_id: str,
    route_policy_id: str,
    selected_action: MetaAction = MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
    terminal_status: TerminalStatus = TerminalStatus.SUCCEEDED,
    evidence_gain_bp: int = 8_000,
    action_propensity_bp: int = 10_000,
    accepted_criterion_ids: tuple[str, ...] = ("AC-1",),
    cost_micros: int = 100,
    latency_ms: int = 40,
    safety_violation: bool = False,
) -> PolicyObservation:
    return PolicyObservation(
        episode_id=episode_id,
        route_policy_id=route_policy_id,
        selected_action=selected_action,
        selection_reason_codes=("linear_score", "shadow_only_enforced"),
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=terminal_status,
        action_propensity_bp=action_propensity_bp,
        accepted_criterion_ids=accepted_criterion_ids,
        evidence_gain_bp=evidence_gain_bp,
        cost_micros=cost_micros,
        latency_ms=latency_ms,
        safety_violation=safety_violation,
    )


def _decision(
    observation: PolicyObservation,
    *,
    frozen_input_ids: tuple[str, ...] = ("tree-1", "objective-rev-1", "seed-1"),
    pairing_kind: PairingKind = PairingKind.COLD,
    partition: EvaluationPartition = EvaluationPartition.HELD_OUT,
) -> LoggedDecision:
    return LoggedDecision.from_observation(
        observation,
        frozen_input_ids=frozen_input_ids,
        pairing_kind=pairing_kind,
        partition=partition,
    )


def _pair(
    candidate: RoutePolicyCandidate,
    *,
    pair_id: str,
    pairing_kind: PairingKind = PairingKind.COLD,
    frozen_input_ids: tuple[str, ...] = ("tree-1", "objective-rev-1", "seed-1"),
    baseline_gain: int = 4_000,
    candidate_gain: int = 8_000,
    baseline_cost: int = 400,
    candidate_cost: int = 100,
    baseline_latency: int = 80,
    candidate_latency: int = 40,
    baseline_action: MetaAction = MetaAction.CALL_REMOTE_STRONG_MODEL,
    candidate_action: MetaAction = MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
    baseline_accepted: tuple[str, ...] = ("AC-1",),
    candidate_accepted: tuple[str, ...] = ("AC-1", "AC-2"),
    baseline_propensity: int = 5_000,
    candidate_propensity: int = 10_000,
    candidate_safety: bool = False,
    baseline_status: TerminalStatus = TerminalStatus.SUCCEEDED,
    candidate_status: TerminalStatus = TerminalStatus.SUCCEEDED,
) -> ComparisonEvidence:
    baseline = _observation(
        episode_id=f"{pair_id}-baseline",
        route_policy_id=candidate.parent_policy_id,
        selected_action=baseline_action,
        evidence_gain_bp=baseline_gain,
        cost_micros=baseline_cost,
        latency_ms=baseline_latency,
        accepted_criterion_ids=baseline_accepted,
        action_propensity_bp=baseline_propensity,
        terminal_status=baseline_status,
    )
    contrast = _observation(
        episode_id=f"{pair_id}-candidate",
        route_policy_id=candidate.candidate_id,
        selected_action=candidate_action,
        evidence_gain_bp=candidate_gain,
        cost_micros=candidate_cost,
        latency_ms=candidate_latency,
        accepted_criterion_ids=candidate_accepted,
        action_propensity_bp=candidate_propensity,
        safety_violation=candidate_safety,
        terminal_status=candidate_status,
    )
    return ComparisonEvidence(
        pair_id=pair_id,
        baseline=_decision(baseline, frozen_input_ids=frozen_input_ids, pairing_kind=pairing_kind),
        candidate=_decision(contrast, frozen_input_ids=frozen_input_ids, pairing_kind=pairing_kind),
    )


def _paired_corpus(candidate: RoutePolicyCandidate) -> tuple[ComparisonEvidence, ...]:
    return (
        _pair(candidate, pair_id="pair-cold", pairing_kind=PairingKind.COLD),
        _pair(
            candidate,
            pair_id="pair-warm",
            pairing_kind=PairingKind.WARM,
            baseline_cost=200,
            candidate_cost=50,
            baseline_latency=20,
            candidate_latency=10,
        ),
    )


def test_interface_is_versioned_and_evaluation_cannot_mutate_production() -> None:
    evaluator = route_policy_evaluation()
    assert ROUTE_POLICY_EVALUATION_INTERFACE == "RoutePolicyEvaluation@1"
    assert evaluator.INTERFACE == ROUTE_POLICY_EVALUATION_INTERFACE
    assert evaluator.shadow_only is True
    assert evaluator.live_routing_effect is False
    assert evaluator.production_exploration is False
    assert evaluator.production_policy_mutated is False
    assert evaluator.affects_production_acceptance is False
    assert dict(evaluator.policy_axis_changes) == {axis: False for axis in PROTECTED_POLICY_AXES}
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.promote("operator-said-yes")
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.apply_production_policy()
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.apply_live_route()
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.evaluate(_candidate(), production=True)
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.evaluate(_candidate(), live=True)
    with pytest.raises(PolicyEvaluationError, match="mutate production policy"):
        evaluator.evaluate(_candidate(), mutate_production_policy=True)
    empty = evaluator.evaluate(_candidate())
    assert empty.reason_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert empty.improvement_claimed is False
    assert empty.production_policy_mutated is False


def test_missing_comparison_returns_exactly_insufficient_counterfactual_evidence() -> None:
    candidate = _candidate()
    logged = _decision(
        _observation(
            episode_id="held-out-1",
            route_policy_id=candidate.candidate_id,
            action_propensity_bp=10_000,
            evidence_gain_bp=9_000,
        )
    )
    result = evaluate_route_policy(
        candidate,
        observations=(logged,),
        evaluation_id="evaluation-1",
    )
    assert result.disposition is EvaluationDisposition.INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.disposition.value == INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.evaluation_code == INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.reason_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.blocker_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert result.paired_delta_bp == 0
    assert result.ips_value_bp == 0
    assert result.production_policy_mutated is False
    assert result.comparison_supported is False


def test_missing_propensity_returns_exactly_insufficient_even_when_candidate_looks_better() -> None:
    candidate = _candidate()
    frozen = ("tree-1", "objective-rev-1", "seed-missing-propensity")
    baseline = LoggedDecision(
        decision_id="baseline-missing-propensity",
        episode_id="episode-baseline",
        selected_action=MetaAction.CALL_REMOTE_STRONG_MODEL,
        policy_id=candidate.parent_policy_id,
        frozen_input_ids=frozen,
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
        propensity_bp=None,
        accepted_criterion_ids=("AC-1",),
        evidence_gain_bp=1_000,
        cost_micros=900,
        latency_ms=200,
    )
    contrast = LoggedDecision(
        decision_id="candidate-missing-propensity",
        episode_id="episode-candidate",
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        policy_id=candidate.candidate_id,
        frozen_input_ids=frozen,
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
        propensity_bp=10_000,
        accepted_criterion_ids=("AC-1", "AC-2"),
        evidence_gain_bp=9_000,
        cost_micros=10,
        latency_ms=5,
    )
    result = RoutePolicyEvaluation().evaluate(
        candidate,
        comparisons=(
            ComparisonEvidence(pair_id="pair-missing-propensity", baseline=baseline, candidate=contrast),
        ),
        evaluation_id="evaluation-1",
    )
    assert result.disposition.value == INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.reason_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.blocker_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert result.paired_delta_bp == 0
    assert result.propensity_supported is False
    assert result.comparison_supported is True


def test_zero_propensity_is_treated_as_missing_propensity_evidence() -> None:
    candidate = _candidate()
    frozen = ("tree-1", "objective-rev-1", "seed-zero")
    baseline = LoggedDecision(
        decision_id="baseline-zero-propensity",
        episode_id="episode-zero",
        selected_action=MetaAction.RUN_SELECTED_TEST,
        policy_id=candidate.parent_policy_id,
        frozen_input_ids=frozen,
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
        propensity_bp=0,
        evidence_gain_bp=2_000,
    )
    contrast = LoggedDecision(
        decision_id="candidate-zero-propensity",
        episode_id="episode-zero-c",
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        policy_id=candidate.candidate_id,
        frozen_input_ids=frozen,
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
        propensity_bp=10_000,
        evidence_gain_bp=8_000,
    )
    result = evaluate_route_policy(
        candidate,
        comparisons=(ComparisonEvidence(pair_id="pair-zero", baseline=baseline, candidate=contrast),),
    )
    assert result.reason_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False


def test_unbound_comparison_is_insufficient_counterfactual_evidence() -> None:
    candidate = _candidate()
    foreign = _candidate(parent_policy_id="other-baseline", policy_version="route-foreign")
    result = evaluate_route_policy(
        candidate,
        comparisons=_paired_corpus(foreign),
        evaluation_id="evaluation-1",
    )
    assert result.disposition.value == INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.reason_codes == (INSUFFICIENT_COUNTERFACTUAL_EVIDENCE,)
    assert result.version_bound is False
    assert result.comparison_supported is False
    assert result.improvement_claimed is False


def test_holdout_training_overlap_is_promotion_ineligible() -> None:
    seed = _observation(
        episode_id="training-leak",
        route_policy_id="policy-baseline",
        evidence_gain_bp=4_000,
    )
    candidate = _candidate(training_observation_ids=(seed.observation_id,))
    leaked = ComparisonEvidence(
        pair_id="pair-leak",
        baseline=_decision(seed),
        candidate=_decision(
            _observation(
                episode_id="training-leak-c",
                route_policy_id=candidate.candidate_id,
                evidence_gain_bp=9_000,
                accepted_criterion_ids=("AC-1", "AC-2"),
            )
        ),
    )
    result = evaluate_route_policy(candidate, comparisons=(leaked,), evaluation_id="evaluation-1")
    assert result.disposition is EvaluationDisposition.HOLDOUT_TRAINING_OVERLAP
    assert result.holdout_separated is False
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert "holdout_training_overlap" in result.blocker_codes


def test_cold_warm_pairing_and_safety_quality_floors_on_complete_evidence() -> None:
    candidate = _candidate()
    result = evaluate_route_policy(
        candidate,
        comparisons=_paired_corpus(candidate),
        evaluation_id="evaluation-1",
    )
    assert result.disposition is EvaluationDisposition.EVALUATED
    assert result.holdout_separated is True
    assert result.propensity_supported is True
    assert result.comparison_supported is True
    assert result.version_bound is True
    assert result.cold_warm_paired is True
    assert result.safety_floor_passed is True
    assert result.quality_floor_passed is True
    assert result.improvement_claimed is True
    assert result.promotion_eligible is True
    assert result.paired_delta_bp > 0
    assert result.accepted_criterion_delta > 0
    assert result.cost_delta_micros < 0
    assert result.production_policy_mutated is False
    assert result.shadow_only is True
    assert "cold_warm_paired" in result.reason_codes
    assert "improvement_supported" in result.reason_codes


def test_cost_savings_without_cold_warm_pairing_cannot_be_the_improvement_claim() -> None:
    candidate = _candidate()
    cold_only = _pair(
        candidate,
        pair_id="pair-cold-only",
        pairing_kind=PairingKind.COLD,
        baseline_gain=8_000,
        candidate_gain=8_000,
        baseline_accepted=("AC-1",),
        candidate_accepted=("AC-1",),
        baseline_cost=8_000,
        candidate_cost=1,
        baseline_latency=10,
        candidate_latency=10,
        baseline_status=TerminalStatus.SUCCEEDED,
        candidate_status=TerminalStatus.SUCCEEDED,
    )
    result = evaluate_route_policy(candidate, comparisons=(cold_only,), evaluation_id="evaluation-1")
    assert result.disposition is EvaluationDisposition.EVALUATED
    assert result.cold_warm_paired is False
    assert result.cost_delta_micros == 0
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert "no_improvement_claim" in result.reason_codes


def test_safety_floor_failure_blocks_improvement_and_promotion() -> None:
    candidate = _candidate()
    unsafe = _pair(
        candidate,
        pair_id="pair-unsafe",
        candidate_safety=True,
        candidate_gain=9_000,
        candidate_accepted=("AC-1", "AC-2", "AC-3"),
    )
    result = evaluate_route_policy(candidate, comparisons=(unsafe,), evaluation_id="evaluation-1")
    assert result.disposition is EvaluationDisposition.SAFETY_FLOOR_FAILED
    assert result.safety_floor_passed is False
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert result.safety_violation_count == 1
    assert result.production_policy_mutated is False


def test_quality_floor_failure_blocks_improvement_and_promotion() -> None:
    candidate = _candidate()
    worse = _pair(
        candidate,
        pair_id="pair-worse",
        baseline_gain=9_000,
        candidate_gain=1_000,
        baseline_accepted=("AC-1", "AC-2"),
        candidate_accepted=("AC-1",),
        candidate_status=TerminalStatus.FAILED,
        baseline_status=TerminalStatus.SUCCEEDED,
    )
    result = evaluate_route_policy(candidate, comparisons=(worse,), evaluation_id="evaluation-1")
    assert result.disposition is EvaluationDisposition.QUALITY_FLOOR_FAILED
    assert result.quality_floor_passed is False
    assert result.safety_floor_passed is True
    assert result.improvement_claimed is False
    assert result.promotion_eligible is False
    assert result.accepted_criterion_delta < 0


def test_exact_version_binding_is_required_for_comparison_support() -> None:
    candidate = _candidate()
    result = evaluate_route_policy(
        candidate,
        comparisons=_paired_corpus(candidate),
        evaluation_id="evaluation-other",
    )
    assert result.disposition.value == INSUFFICIENT_COUNTERFACTUAL_EVIDENCE
    assert result.version_bound is False
    assert result.comparison_supported is False
    assert result.improvement_claimed is False


def test_ips_uses_logged_propensity_and_does_not_invent_off_policy_reward() -> None:
    candidate = _candidate()
    matching = _pair(
        candidate,
        pair_id="pair-match",
        baseline_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        candidate_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        baseline_propensity=2_500,
        baseline_gain=8_000,
        candidate_gain=8_000,
        baseline_accepted=("AC-1",),
        candidate_accepted=("AC-1",),
    )
    switched = _pair(
        candidate,
        pair_id="pair-switch",
        frozen_input_ids=("tree-1", "objective-rev-1", "seed-1"),
        baseline_action=MetaAction.CALL_REMOTE_STRONG_MODEL,
        candidate_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        baseline_propensity=2_500,
        baseline_gain=1_000,
        candidate_gain=8_000,
        baseline_accepted=("AC-1",),
        candidate_accepted=("AC-1", "AC-2"),
        pairing_kind=PairingKind.WARM,
    )
    result = evaluate_route_policy(
        candidate,
        comparisons=(matching, switched),
        evaluation_id="evaluation-1",
    )
    assert result.propensity_supported is True
    assert result.ips_value_bp != 0
    # Only the matched logged action contributes to IPS; the switched pair is
    # zero-weighted rather than filled with the candidate's observed reward.
    assert result.ips_value_bp == 16_000
    assert result.paired_delta_bp != 0
    assert result.cold_warm_paired is True


def test_result_is_frozen_content_addressed_and_non_authorizing() -> None:
    candidate = _candidate()
    result = evaluate_route_policy(
        candidate,
        comparisons=_paired_corpus(candidate),
        evaluation_id="evaluation-1",
    )
    with pytest.raises(FrozenInstanceError):
        result.production_policy_mutated = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.improvement_claimed = True  # type: ignore[misc]
    encoded = result.to_dict()
    assert encoded["production_policy_mutated"] is False
    assert encoded["live_routing_effect"] is False
    assert encoded["shadow_only"] is True
    assert all(value is False for value in encoded["policy_axis_changes"].values())
    replay_id = content_identity({key: value for key, value in encoded.items() if key != "result_id"})
    assert replay_id == result.result_id
    assert result.candidate_id == candidate.candidate_id
    assert result.policy_version == candidate.policy_version


def test_comparison_requires_identical_frozen_inputs_and_distinct_policies() -> None:
    candidate = _candidate()
    baseline = _decision(
        _observation(episode_id="a", route_policy_id=candidate.parent_policy_id),
        frozen_input_ids=("tree-a",),
    )
    other_tree = _decision(
        _observation(episode_id="b", route_policy_id=candidate.candidate_id),
        frozen_input_ids=("tree-b",),
    )
    with pytest.raises(PolicyEvaluationError, match="identical frozen input"):
        ComparisonEvidence(pair_id="mismatch", baseline=baseline, candidate=other_tree)
    same_policy = _decision(
        _observation(episode_id="c", route_policy_id=candidate.parent_policy_id),
        frozen_input_ids=("tree-a",),
    )
    with pytest.raises(PolicyEvaluationError, match="distinct baseline and candidate"):
        ComparisonEvidence(pair_id="same-policy", baseline=baseline, candidate=same_policy)


def test_floats_forbidden_fields_and_self_promotion_are_rejected() -> None:
    assert field_is_forbidden("raw_prompt")
    assert field_is_forbidden("private_reasoning")
    candidate = _candidate()
    with pytest.raises(PolicyEvaluationError, match="float"):
        LoggedDecision.from_dict(
            {
                "decision_id": "float-1",
                "episode_id": "episode-float",
                "selected_action": MetaAction.RUN_LOCAL_STATIC_ANALYSIS.value,
                "policy_id": candidate.parent_policy_id,
                "frozen_input_ids": ("tree-1",),
                "feature_ids": ("risk_rank",),
                "terminal_status": TerminalStatus.SUCCEEDED.value,
                "evidence_gain_bp": 1.5,
            }
        )
    with pytest.raises(PolicyEvaluationError, match="forbidden"):
        LoggedDecision.from_dict(
            {
                "decision_id": "secret-1",
                "episode_id": "episode-secret",
                "selected_action": MetaAction.RUN_LOCAL_STATIC_ANALYSIS.value,
                "policy_id": candidate.parent_policy_id,
                "frozen_input_ids": ("tree-1",),
                "feature_ids": ("risk_rank",),
                "terminal_status": TerminalStatus.SUCCEEDED.value,
                "raw_prompt": "leak",
            }
        )
    payload = candidate.to_dict()
    payload["external_authorization_id"] = "self-promote"
    with pytest.raises((PolicyEvaluationError, Exception), match="authoriz"):
        evaluate_route_policy(payload, comparisons=_paired_corpus(candidate))


def test_logged_decision_and_comparison_round_trip_preserve_identity() -> None:
    candidate = _candidate()
    pair = _pair(candidate, pair_id="pair-round-trip")
    replayed_decision = LoggedDecision.from_dict(pair.baseline.to_dict())
    replayed_pair = ComparisonEvidence.from_dict(pair.to_dict())
    assert replayed_decision.content_id == pair.baseline.content_id
    assert replayed_pair.content_id == pair.content_id
    assert replayed_pair.pairing_kind is PairingKind.COLD
    assert replayed_pair.frozen_input_ids == pair.frozen_input_ids


def test_pending_decisions_cannot_score_offline_evaluation() -> None:
    candidate = _candidate()
    pending = LoggedDecision(
        decision_id="pending-1",
        episode_id="episode-pending",
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        policy_id=candidate.parent_policy_id,
        frozen_input_ids=("tree-1", "objective-rev-1"),
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.PENDING,
        propensity_bp=10_000,
        evidence_gain_bp=0,
    )
    contrast = LoggedDecision(
        decision_id="pending-c",
        episode_id="episode-pending-c",
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        policy_id=candidate.candidate_id,
        frozen_input_ids=("tree-1", "objective-rev-1"),
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
        propensity_bp=10_000,
        evidence_gain_bp=8_000,
    )
    with pytest.raises(PolicyEvaluationError, match="pending"):
        evaluate_route_policy(
            candidate,
            comparisons=(ComparisonEvidence(pair_id="pair-pending", baseline=pending, candidate=contrast),),
        )
