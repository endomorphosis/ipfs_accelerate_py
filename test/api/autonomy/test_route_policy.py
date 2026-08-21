from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    AutonomyContractError,
    CancellationBehavior,
    MetaAction,
    PolicyObservation,
    PrivacyClass,
    ResolutionAction,
    ResolutionCandidate,
    ResolutionEvidenceKind,
    RiskClass,
    RoutePolicyCandidate,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.route_policy import (
    PROTECTED_POLICY_AXES,
    ROUTE_POLICY_CANDIDATE_INTERFACE,
    SHADOW_ROUTE_POLICY_INTERFACE,
    SelectionDisposition,
    SelectionMode,
    ShadowRoutePolicy,
    ShadowRoutePolicyError,
    field_is_forbidden,
    shadow_route_policy,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _candidate(**overrides: Any) -> RoutePolicyCandidate:
    values: dict[str, Any] = {
        "parent_policy_id": "policy-1",
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


def _features(
    *,
    risk_rank: int | Fraction = 2,
    context_sufficient: int | Fraction = 1,
    token_cost: int | Fraction = 0,
) -> dict[str, int | Fraction]:
    return {
        "risk_rank": risk_rank,
        "context_sufficient": context_sufficient,
        "token_cost": token_cost,
    }


def _features_by_action(**token_costs: int) -> dict[MetaAction, dict[str, int | Fraction]]:
    defaults = {
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS: 0,
        MetaAction.RUN_SELECTED_TEST: 20,
        MetaAction.CALL_REMOTE_STRONG_MODEL: 1_000,
    }
    defaults.update(token_costs)
    return {
        action: _features(token_cost=cost) for action, cost in defaults.items()
    }


def _observation(
    *,
    episode_id: str = "episode-1",
    route_policy_id: str,
    selected_action: MetaAction = MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
    terminal_status: TerminalStatus = TerminalStatus.SUCCEEDED,
    evidence_gain_bp: int = 8_000,
    safety_violation: bool = False,
) -> PolicyObservation:
    return PolicyObservation(
        episode_id=episode_id,
        route_policy_id=route_policy_id,
        selected_action=selected_action,
        selection_reason_codes=("linear_score", "shadow_only_enforced"),
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=terminal_status,
        action_propensity_bp=10_000,
        evidence_gain_bp=evidence_gain_bp,
        safety_violation=safety_violation,
    )


def _action(kind: MetaAction, **overrides: object) -> ResolutionAction:
    remote = kind in {MetaAction.CALL_REMOTE_STANDARD_MODEL, MetaAction.CALL_REMOTE_STRONG_MODEL}
    values: dict[str, object] = {
        "action": kind,
        "precondition_ids": ("tree-current",),
        "expected_evidence_kind": ResolutionEvidenceKind.STATIC_ANALYSIS,
        "expected_uncertainty_reduction_bp": 8_000,
        "token_cost": 0,
        "latency_cost_ms": 100,
        "provider_cost_micros": 0,
        "resource_cost_units": 1,
        "invalidation_cost_units": 1,
        "privacy_cost_units": 1 if remote else 0,
        "privacy_class": PrivacyClass.PUBLIC if remote else PrivacyClass.LOCAL_ONLY,
        "risk_class": RiskClass.R1_READ_ONLY,
        "cancellation_behavior": CancellationBehavior.COOPERATIVE,
        "cacheable": True,
        "authority_class": AuthorityClass.VERIFIED,
        "can_change_decision": True,
        "accepted_as_authority": True,
    }
    values.update(overrides)
    return ResolutionAction(**values)


def test_interfaces_are_versioned_and_policy_is_shadow_only() -> None:
    policy = ShadowRoutePolicy(_candidate())
    assert SHADOW_ROUTE_POLICY_INTERFACE == "ShadowRoutePolicy@1"
    assert ROUTE_POLICY_CANDIDATE_INTERFACE == "RoutePolicyCandidate@1"
    assert policy.INTERFACE == SHADOW_ROUTE_POLICY_INTERFACE
    assert policy.CANDIDATE_INTERFACE == ROUTE_POLICY_CANDIDATE_INTERFACE
    assert policy.candidate.shadow_only is True
    assert policy.shadow_only is True
    assert policy.live_routing_effect is False
    assert policy.production_exploration is False
    assert policy.affects_production_acceptance is False
    assert dict(policy.policy_axis_changes) == {axis: False for axis in PROTECTED_POLICY_AXES}


def test_linear_score_is_deterministic_for_integer_and_rational_features() -> None:
    policy = shadow_route_policy(_candidate())
    integer_features = _features_by_action()
    rational_features = {
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS: _features(
            risk_rank=Fraction(4, 2),
            context_sufficient=(1, 1),
            token_cost={"numerator": 0, "denominator": 1},
        ),
        MetaAction.RUN_SELECTED_TEST: _features(token_cost=Fraction(20)),
        MetaAction.CALL_REMOTE_STRONG_MODEL: _features(token_cost=1_000),
    }
    first = policy.score(integer_features)
    second = policy.score(rational_features)
    third = policy.score(
        {
            MetaAction.CALL_REMOTE_STRONG_MODEL: integer_features[MetaAction.CALL_REMOTE_STRONG_MODEL],
            MetaAction.RUN_SELECTED_TEST: integer_features[MetaAction.RUN_SELECTED_TEST],
            MetaAction.RUN_LOCAL_STATIC_ANALYSIS: integer_features[MetaAction.RUN_LOCAL_STATIC_ANALYSIS],
        }
    )
    assert [item.score_id for item in first] == [item.score_id for item in second]
    assert [item.score_id for item in first] == [item.score_id for item in third]
    by_action = {item.action: item for item in first}
    #  -2*2 + 5*1 + -1*0 = 1  vs  -2*2 + 5*1 + -1*1000 = -999
    assert by_action[MetaAction.RUN_LOCAL_STATIC_ANALYSIS].linear_score == Fraction(1)
    assert by_action[MetaAction.CALL_REMOTE_STRONG_MODEL].linear_score == Fraction(-999)
    assert by_action[MetaAction.RUN_LOCAL_STATIC_ANALYSIS].total_bp > by_action[
        MetaAction.CALL_REMOTE_STRONG_MODEL
    ].total_bp


def test_floats_and_bools_are_rejected_as_features() -> None:
    policy = ShadowRoutePolicy(_candidate())
    with pytest.raises(ShadowRoutePolicyError, match="float"):
        policy.score({MetaAction.RUN_LOCAL_STATIC_ANALYSIS: _features(token_cost=1.5)})  # type: ignore[arg-type]
    with pytest.raises(ShadowRoutePolicyError, match="integer or rational"):
        policy.score({MetaAction.RUN_LOCAL_STATIC_ANALYSIS: _features(token_cost=True)})  # type: ignore[arg-type]


def test_undeclared_or_forbidden_features_are_rejected() -> None:
    policy = ShadowRoutePolicy(_candidate())
    assert field_is_forbidden("raw_prompt")
    assert field_is_forbidden("production_explore")
    with pytest.raises(ShadowRoutePolicyError, match="undeclared"):
        policy.score(
            {MetaAction.RUN_LOCAL_STATIC_ANALYSIS: {**_features(), "unexpected": 1}}
        )
    with pytest.raises(ShadowRoutePolicyError, match="forbidden"):
        policy.score(
            {MetaAction.RUN_LOCAL_STATIC_ANALYSIS: {**_features(), "raw_prompt": 1}}
        )
    with pytest.raises(ShadowRoutePolicyError, match="forbidden"):
        ShadowRoutePolicy(
            _candidate(
                feature_names=("production_explore", "risk_rank"),
                integer_weights={"production_explore": 1, "risk_rank": 0},
            )
        )


def test_selects_only_policy_admitted_actions_and_ignores_higher_unadmitted_score() -> None:
    policy = ShadowRoutePolicy(_candidate())
    features = _features_by_action()
    features[MetaAction.REQUEST_HUMAN_DECISION] = _features(
        risk_rank=0, context_sufficient=100, token_cost=0
    )
    selection = policy.select(features, episode_id="episode-closed")
    assert selection.disposition is SelectionDisposition.SELECTED
    assert selection.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    rejected = {item.action: item for item in selection.scores}[MetaAction.REQUEST_HUMAN_DECISION]
    assert rejected.admissible is False
    assert "action_not_policy_admitted" in rejected.reason_codes
    assert selection.selected_action in policy.allowed_actions()
    assert MetaAction.REQUEST_HUMAN_DECISION not in policy.allowed_actions()


def test_current_admission_intersection_is_a_hard_filter() -> None:
    policy = ShadowRoutePolicy(_candidate())
    selection = policy.select(
        _features_by_action(),
        episode_id="episode-admitted",
        admitted_actions=(MetaAction.RUN_SELECTED_TEST,),
    )
    assert selection.selected_action is MetaAction.RUN_SELECTED_TEST
    by_action = {item.action: item for item in selection.scores}
    assert by_action[MetaAction.RUN_LOCAL_STATIC_ANALYSIS].admissible is False
    assert "action_not_currently_admitted" in by_action[MetaAction.RUN_LOCAL_STATIC_ANALYSIS].reason_codes


def test_no_admitted_action_abstains_without_an_observation() -> None:
    policy = ShadowRoutePolicy(_candidate())
    selection = policy.select(
        {MetaAction.QUARANTINE_TASK: _features()},
        episode_id="episode-empty",
        admitted_actions=(MetaAction.QUARANTINE_TASK,),
    )
    assert selection.disposition is SelectionDisposition.ABSTAINED
    assert selection.selected_action is None
    assert selection.observation is None
    assert "no_policy_admitted_action" in selection.reason_codes
    assert selection.live_routing_effect is False


def test_linear_ucb_bonus_is_shadow_only_and_cannot_admit_new_actions() -> None:
    policy = ShadowRoutePolicy(_candidate())
    features = _features_by_action()
    greedy = policy.select(features, episode_id="episode-greedy", mode=SelectionMode.LINEAR_SCORE)
    ucb = policy.select(features, episode_id="episode-ucb", mode=SelectionMode.LINEAR_UCB)
    assert greedy.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    assert ucb.selected_action in policy.allowed_actions()
    assert ucb.shadow_only is True
    assert ucb.production_exploration is False
    assert ucb.live_routing_effect is False
    assert "linear_ucb" in ucb.reason_codes
    human = {item.action: item for item in ucb.scores}.get(MetaAction.REQUEST_HUMAN_DECISION)
    assert human is None or human.admissible is False
    by_action = {item.action: item for item in ucb.scores}
    assert by_action[MetaAction.CALL_REMOTE_STRONG_MODEL].ucb_bonus_bp > 0
    assert by_action[MetaAction.RUN_LOCAL_STATIC_ANALYSIS].admissible is True


def test_ucb_bonus_shrinks_after_shadow_observation_of_that_action() -> None:
    policy = ShadowRoutePolicy(_candidate())
    features = _features(
        risk_rank=2, context_sufficient=1, token_cost=1_000
    )
    before = {
        item.action: item
        for item in policy.score(
            {MetaAction.CALL_REMOTE_STRONG_MODEL: features},
            mode=SelectionMode.LINEAR_UCB,
        )
    }[MetaAction.CALL_REMOTE_STRONG_MODEL]
    observed = policy.observe(
        _observation(
            route_policy_id=policy.candidate.candidate_id,
            selected_action=MetaAction.CALL_REMOTE_STRONG_MODEL,
            terminal_status=TerminalStatus.FAILED,
            evidence_gain_bp=0,
        ),
        features,
    )
    after = {
        item.action: item
        for item in observed.score(
            {MetaAction.CALL_REMOTE_STRONG_MODEL: features},
            mode=SelectionMode.LINEAR_UCB,
        )
    }[MetaAction.CALL_REMOTE_STRONG_MODEL]
    assert after.observation_count == before.observation_count + 1
    assert after.ucb_bonus_bp < before.ucb_bonus_bp
    assert observed.candidate.candidate_id == policy.candidate.candidate_id
    assert observed.live_routing_effect is False


def test_logged_observation_records_reasons_and_positive_propensity() -> None:
    policy = ShadowRoutePolicy(_candidate())
    selection = policy.select(_features_by_action(), episode_id="episode-log")
    assert selection.observation is not None
    assert selection.observation.action_propensity_bp == 10_000
    assert selection.observation.route_policy_id == policy.candidate.candidate_id
    assert "shadow_only_enforced" in selection.observation.selection_reason_codes
    assert "closed_action_set" in selection.observation.selection_reason_codes
    assert tuple(selection.observation.feature_ids) == policy.candidate.feature_names
    replay = PolicyObservation.from_dict(selection.observation.to_dict())
    assert replay.observation_id == selection.observation.observation_id


def test_tie_break_is_stable_under_allowed_action_order() -> None:
    candidate = _candidate(
        allowed_actions=(MetaAction.RUN_SELECTED_TEST, MetaAction.RUN_LOCAL_STATIC_ANALYSIS),
        integer_weights={"risk_rank": 0, "context_sufficient": 0, "token_cost": 0},
    )
    policy = ShadowRoutePolicy(candidate)
    features = {
        MetaAction.RUN_LOCAL_STATIC_ANALYSIS: _features(token_cost=0),
        MetaAction.RUN_SELECTED_TEST: _features(token_cost=0),
    }
    one = policy.select(features, episode_id="episode-tie-a")
    two = policy.select(
        {
            MetaAction.RUN_SELECTED_TEST: features[MetaAction.RUN_SELECTED_TEST],
            MetaAction.RUN_LOCAL_STATIC_ANALYSIS: features[MetaAction.RUN_LOCAL_STATIC_ANALYSIS],
        },
        episode_id="episode-tie-b",
    )
    assert one.selected_action is MetaAction.RUN_SELECTED_TEST
    assert two.selected_action is MetaAction.RUN_SELECTED_TEST
    assert "tie_broken_by_allowed_action_order" in one.reason_codes


def test_production_or_live_flags_fail_closed() -> None:
    policy = ShadowRoutePolicy(_candidate())
    features = _features_by_action()
    with pytest.raises(ShadowRoutePolicyError, match="production exploration"):
        policy.score(features, production=True)
    with pytest.raises(ShadowRoutePolicyError, match="live routing effect"):
        policy.select(features, episode_id="episode-live", live=True)
    with pytest.raises(ShadowRoutePolicyError, match="live routing effect"):
        policy.select(features, episode_id="episode-explore", explore_production=True)
    with pytest.raises(ShadowRoutePolicyError, match="own promotion"):
        policy.promote("operator-said-yes")
    with pytest.raises(ShadowRoutePolicyError, match="live routing effect"):
        policy.apply_live_route()


def test_authority_and_privacy_ceilings_cannot_increase() -> None:
    policy = ShadowRoutePolicy(_candidate())
    strong = _action(
        MetaAction.CALL_REMOTE_STRONG_MODEL,
        authority_class=AuthorityClass.OPERATOR_REQUIRED,
        accepted_as_authority=False,
        privacy_class=PrivacyClass.PUBLIC,
        expected_evidence_kind=ResolutionEvidenceKind.MODEL_ADVICE,
        token_cost=10,
    )
    static = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    features = {
        strong: _features(token_cost=0, context_sufficient=9),
        static: _features(token_cost=50),
    }
    selection = policy.select(
        features,
        episode_id="episode-authority",
        authority_ceiling=AuthorityClass.VERIFIED,
    )
    assert selection.selected_action is MetaAction.RUN_LOCAL_STATIC_ANALYSIS
    rejected = {item.action: item for item in selection.scores}[MetaAction.CALL_REMOTE_STRONG_MODEL]
    assert rejected.admissible is False
    assert "authority_increase_denied" in rejected.reason_codes

    private_remote = _action(
        MetaAction.CALL_REMOTE_STRONG_MODEL,
        privacy_class=PrivacyClass.SENSITIVE,
        expected_evidence_kind=ResolutionEvidenceKind.MODEL_ADVICE,
    )
    blocked = policy.score(
        {private_remote: _features(token_cost=0)},
        privacy_ceiling=PrivacyClass.PUBLIC,
    )
    assert blocked[0].admissible is False
    assert "privacy_policy_increase_denied" in blocked[0].reason_codes


def test_inadmissible_resolution_candidate_is_not_chosen() -> None:
    policy = ShadowRoutePolicy(_candidate())
    action = _action(MetaAction.RUN_LOCAL_STATIC_ANALYSIS)
    candidate = ResolutionCandidate(
        question_id="question-1",
        resolution_action=action,
        expected_decision_value=100,
        admissible=False,
        policy_id="policy-1",
    )
    selection = policy.select(
        {candidate: _features(), MetaAction.RUN_SELECTED_TEST: _features(token_cost=20)},
        episode_id="episode-inadmissible",
    )
    assert selection.selected_action is MetaAction.RUN_SELECTED_TEST
    rejected = {item.action: item for item in selection.scores}[MetaAction.RUN_LOCAL_STATIC_ANALYSIS]
    assert rejected.admissible is False
    assert "candidate_not_admissible" in rejected.reason_codes


def test_fit_updates_integer_weights_without_expanding_actions_or_authority() -> None:
    policy = ShadowRoutePolicy(_candidate())
    success = _observation(
        episode_id="episode-success",
        route_policy_id=policy.candidate.candidate_id,
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        evidence_gain_bp=9_000,
    )
    failure = _observation(
        episode_id="episode-failure",
        route_policy_id=policy.candidate.candidate_id,
        selected_action=MetaAction.CALL_REMOTE_STRONG_MODEL,
        terminal_status=TerminalStatus.FAILED,
        evidence_gain_bp=0,
    )
    fitted = policy.fit(
        (success, failure),
        features_by_observation={
            success.observation_id: _features(token_cost=0),
            failure.observation_id: _features(token_cost=1_000),
        },
        held_out_evaluation_ids=("held-out-fit-1",),
        safety_gate_receipt_ids=("safety-fit-1",),
        policy_version="route-v2",
    )
    assert fitted.candidate.shadow_only is True
    assert fitted.candidate.external_authorization_id == ""
    assert fitted.candidate.parent_policy_id == policy.candidate.candidate_id
    assert set(fitted.candidate.allowed_actions) <= set(policy.candidate.allowed_actions)
    assert fitted.candidate.feature_names == policy.candidate.feature_names
    assert fitted.live_routing_effect is False
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in fitted.candidate.integer_weights.values())
    with pytest.raises(ShadowRoutePolicyError, match="already policy-admitted"):
        policy.propose(
            (success,),
            features_by_observation={success.observation_id: _features()},
            held_out_evaluation_ids=("held-out-fit-2",),
            safety_gate_receipt_ids=("safety-fit-2",),
            policy_version="route-v-expand",
            allowed_actions=(
                MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
                MetaAction.REQUEST_HUMAN_DECISION,
            ),
        )


def test_fit_rejects_unadmitted_training_actions_and_safety_success() -> None:
    policy = ShadowRoutePolicy(_candidate())
    leaked = _observation(
        route_policy_id=policy.candidate.candidate_id,
        selected_action=MetaAction.QUARANTINE_TASK,
    )
    with pytest.raises(ShadowRoutePolicyError, match="already policy-admitted"):
        policy.propose(
            (leaked,),
            features_by_observation={leaked.observation_id: _features()},
            held_out_evaluation_ids=("held-out-1",),
            safety_gate_receipt_ids=("safety-1",),
            policy_version="route-v-bad",
        )
    unsafe = _observation(
        route_policy_id=policy.candidate.candidate_id,
        safety_violation=True,
    )
    with pytest.raises(ShadowRoutePolicyError, match="safety violation"):
        policy.propose(
            (unsafe,),
            features_by_observation={unsafe.observation_id: _features()},
            held_out_evaluation_ids=("held-out-1",),
            safety_gate_receipt_ids=("safety-1",),
            policy_version="route-v-unsafe",
        )


def test_exact_versioned_rollback_restores_candidate_and_ucb_counts() -> None:
    original = ShadowRoutePolicy(_candidate())
    observed = original.observe(
        _observation(
            route_policy_id=original.candidate.candidate_id,
            selected_action=MetaAction.RUN_SELECTED_TEST,
        ),
        _features(token_cost=20),
    )
    success = _observation(
        route_policy_id=observed.candidate.candidate_id,
        selected_action=MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
    )
    fitted = observed.fit(
        (success,),
        features_by_observation={success.observation_id: _features()},
        held_out_evaluation_ids=("held-out-rollback",),
        safety_gate_receipt_ids=("safety-rollback",),
        policy_version="route-v2",
    )
    assert fitted.candidate.candidate_id != original.candidate.candidate_id
    assert original.candidate.candidate_id in fitted.lineage_ids
    assert fitted.rollback_vectors
    assert fitted.rollback_vectors[0]["to_candidate_id"] == observed.candidate.candidate_id
    restored = fitted.rollback(original.candidate.candidate_id)
    assert restored.candidate.candidate_id == original.candidate.candidate_id
    assert restored.candidate.to_dict() == original.candidate.to_dict()
    assert restored.ucb_state.state_id == observed.ucb_state.state_id
    with pytest.raises(ShadowRoutePolicyError, match="exact prior"):
        fitted.rollback("missing-candidate")
    with pytest.raises(ShadowRoutePolicyError, match="available to roll back"):
        original.rollback()


def test_rollback_without_target_restores_the_immediate_parent_snapshot() -> None:
    original = ShadowRoutePolicy(_candidate())
    success = _observation(route_policy_id=original.candidate.candidate_id)
    child = original.fit(
        (success,),
        features_by_observation={success.observation_id: _features()},
        held_out_evaluation_ids=("held-out-parent",),
        safety_gate_receipt_ids=("safety-parent",),
        policy_version="route-v2",
    )
    grandchild = child.fit(
        (success,),
        features_by_observation={success.observation_id: _features()},
        held_out_evaluation_ids=("held-out-child",),
        safety_gate_receipt_ids=("safety-child",),
        policy_version="route-v3",
    )
    restored = grandchild.rollback()
    assert restored.candidate.candidate_id == child.candidate.candidate_id
    assert original.candidate.candidate_id in restored.lineage_ids
    assert grandchild.candidate.candidate_id not in restored.lineage_ids


def test_candidate_cannot_self_promote_or_leave_shadow() -> None:
    with pytest.raises(AutonomyContractError, match="self|authorize"):
        _candidate(external_authorization_id="candidate-says-yes")
    with pytest.raises(AutonomyContractError, match="shadow-only"):
        _candidate(shadow_only=False)
    policy = ShadowRoutePolicy(_candidate())
    payload = policy.candidate.to_dict()
    payload["shadow_only"] = False
    with pytest.raises((AutonomyContractError, ShadowRoutePolicyError), match="shadow-only"):
        ShadowRoutePolicy(payload)


def test_selection_records_are_frozen_content_addressed_and_non_authorizing() -> None:
    policy = ShadowRoutePolicy(_candidate())
    selection = policy.select(_features_by_action(), episode_id="episode-frozen")
    with pytest.raises(FrozenInstanceError):
        selection.live_routing_effect = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        selection.shadow_only = False  # type: ignore[misc]
    encoded = selection.to_dict()
    assert encoded["live_routing_effect"] is False
    assert encoded["production_exploration"] is False
    assert encoded["affects_production_acceptance"] is False
    assert encoded["shadow_only"] is True
    assert all(value is False for value in encoded["policy_axis_changes"].values())
    replay_id = content_identity(
        {key: value for key, value in encoded.items() if key != "selection_id"}
    )
    assert replay_id == selection.selection_id
    assert policy.to_dict()["live_routing_effect"] is False


def test_observe_rejects_pending_and_foreign_or_unadmitted_actions() -> None:
    policy = ShadowRoutePolicy(_candidate())
    pending = _observation(
        route_policy_id=policy.candidate.candidate_id,
        terminal_status=TerminalStatus.PENDING,
    )
    with pytest.raises(ShadowRoutePolicyError, match="pending"):
        policy.observe(pending, _features())
    foreign = _observation(route_policy_id="other-policy")
    with pytest.raises(ShadowRoutePolicyError, match="not bound"):
        policy.observe(foreign, _features())
    leaked = PolicyObservation(
        episode_id="episode-leak",
        route_policy_id=policy.candidate.candidate_id,
        selected_action=MetaAction.QUARANTINE_TASK,
        selection_reason_codes=("linear_score",),
        feature_ids=("risk_rank", "context_sufficient", "token_cost"),
        terminal_status=TerminalStatus.SUCCEEDED,
    )
    with pytest.raises(ShadowRoutePolicyError, match="already policy-admitted"):
        policy.observe(leaked, _features())
