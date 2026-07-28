from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.adaptive_goal_refiner import (
    AdaptiveGoalRefinementError,
    AdaptiveGoalRefiner,
    AdaptiveRefinementCandidate,
    AdaptiveRefinementRequest,
    RefinementDecision,
    RefinementDeltaQualityReport,
    RefinementProducerKind,
    RefinementSignal,
    RefinementSignalKind,
    RefinementValueEstimate,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    Actor,
    ActorKind,
    FormalWorkPlan,
    Goal,
    PlanTask,
)
from ipfs_accelerate_py.agent_supervisor.objectives.goal_refinement_verification import (
    FrozenRefinementContext,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
    ObjectiveRefinementEventTracker,
    ObjectiveRefinementPollDecision,
)


def _plan(*, child: bool = False) -> FormalWorkPlan:
    goals = [
        Goal(
            goal_id="goal:root",
            owner_actor_id="actor:supervisor",
            satisfaction_formula_id="formula:root",
            source_ids=("objective:root",),
        )
    ]
    if child:
        goals.append(
            Goal(
                goal_id="goal:child",
                owner_actor_id="actor:supervisor",
                satisfaction_formula_id="formula:child",
                source_ids=("event:changed",),
            )
        )
    return FormalWorkPlan(
        vocabulary_profile_id="event-refinement-test",
        vocabulary_version=1,
        actors=(Actor("actor:supervisor", ActorKind.SUPERVISOR),),
        goals=tuple(goals),
        subgoals=(),
        tasks=(
            PlanTask(
                "task:root",
                "goal:root",
                actor_ids=("actor:supervisor",),
            ),
        ),
        events=(),
        fluents=(),
        preconditions=(),
        effects=(),
        norms=(),
        temporal_constraints=(),
        evidence_requirements=(),
        source_ids=("objective:root",),
        repository_tree_id="tree:event-refinement",
    )


def _signal(
    kind: RefinementSignalKind = RefinementSignalKind.COUNTEREXAMPLE,
    *,
    revision: str = "revision:1",
    observed_at: int = 1,
) -> RefinementSignal:
    return RefinementSignal(
        kind=kind,
        subject_id="goal:root",
        evidence_revision=revision,
        observed_at=observed_at,
        failure_signature=(
            "pytest::event-refinement"
            if kind is RefinementSignalKind.REPEATED_FAILURE
            else ""
        ),
        details={"source": "test"},
    )


def _request(signal: RefinementSignal | None = None) -> AdaptiveRefinementRequest:
    plan = _plan()
    root = plan.goals[0]
    return AdaptiveRefinementRequest(
        plan=plan,
        root_goal_id=root.goal_id,
        root_goal_content_id=root.content_id,
        assumption_ids=("assumption:frozen",),
        signals=(signal or _signal(),),
        cycle_id="cycle:event-refinement",
    )


def _candidate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
    return AdaptiveRefinementCandidate(
        plan=_plan(child=True),
        root_goal_id=request.root_goal_id,
        root_goal_content_id=request.root_goal_content_id,
        assumption_ids=request.assumption_ids,
        changed_goal_ids=("goal:child",),
        signal_kind=request.signals[0].kind,
        producer_id="provider:test",
        producer_kind=RefinementProducerKind.LANGUAGE_MODEL,
    )


@dataclass(frozen=True)
class _Verification:
    frozen_context: FrozenRefinementContext
    candidate_plan_id: str
    verified: bool = True
    content_id: str = "verification:independent"
    reason: str = ""


def _verify(
    candidate: AdaptiveRefinementCandidate,
    request: AdaptiveRefinementRequest,
) -> _Verification:
    return _Verification(
        frozen_context=request.frozen_context,
        candidate_plan_id=candidate.plan.content_id,
    )


@pytest.mark.parametrize(
    "kind",
    (
        RefinementSignalKind.COUNTEREXAMPLE,
        RefinementSignalKind.STALE_EVIDENCE,
        RefinementSignalKind.UNCOVERED_CRITERION,
        RefinementSignalKind.CAPABILITY_CHANGE,
        RefinementSignalKind.INTERFACE_CHANGE,
        RefinementSignalKind.SCOPE_CONFLICT,
        RefinementSignalKind.RESOURCE_INFEASIBLE,
        RefinementSignalKind.UNCERTAINTY_CHANGE,
        RefinementSignalKind.OPERATOR_REVISION,
    ),
)
def test_only_reviewed_semantic_event_families_reach_generation(
    kind: RefinementSignalKind,
) -> None:
    result = AdaptiveGoalRefiner(_candidate, _verify, clock=lambda: 10).refine(
        _request(_signal(kind))
    )
    assert result.admitted
    assert result.receipt.value_estimate.information_gain_millionths > 0
    with pytest.raises(AdaptiveGoalRefinementError, match="kind is unsupported"):
        _signal("heartbeat")  # type: ignore[arg-type]


def test_information_gain_and_downstream_cost_gate_precedes_model_call() -> None:
    calls = {"generator": 0}

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    def low_value(request: AdaptiveRefinementRequest) -> RefinementValueEstimate:
        return RefinementValueEstimate(
            information_gain_millionths=100_000,
            expected_downstream_cost_millionths=100_000,
            affected_subject_ids=("goal:root",),
            signal_ids=tuple(item.evidence_id for item in request.signals),
            rationale_codes=("no_positive_net_value",),
        )

    result = AdaptiveGoalRefiner(
        generate,
        _verify,
        value_estimator=low_value,
        clock=lambda: 10,
    ).refine(_request())
    assert result.decision is RefinementDecision.INSUFFICIENT_INFORMATION_GAIN
    assert not result.model_called
    assert calls == {"generator": 0}


def test_quality_lint_rejection_prevents_verification_and_commit() -> None:
    calls = {"verifier": 0}

    def reject(
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> RefinementDeltaQualityReport:
        return RefinementDeltaQualityReport(
            previous_plan_id=request.plan.content_id,
            candidate_plan_id=candidate.plan.content_id,
            root_goal_content_id=request.root_goal_content_id,
            assumption_ids=request.assumption_ids,
            changed_goal_ids=candidate.changed_goal_ids,
            accepted=False,
            debt_codes=("uncovered_acceptance",),
            linter_id="test-linter@1",
        )

    def verify(
        candidate: AdaptiveRefinementCandidate,
        request: AdaptiveRefinementRequest,
    ) -> _Verification:
        calls["verifier"] += 1
        return _verify(candidate, request)

    result = AdaptiveGoalRefiner(
        _candidate,
        verify,
        quality_linter=reject,
        clock=lambda: 10,
    ).refine(_request())
    assert result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert result.admitted_plan is None
    assert calls == {"verifier": 0}


def test_unchanged_poll_after_restart_has_no_model_call_or_objective_write(
    tmp_path,
) -> None:
    calls = {"generator": 0, "writer": 0}

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    def commit(_plan: FormalWorkPlan, _receipt) -> None:
        calls["writer"] += 1

    state_path = tmp_path / "event-state.json"
    request = _request()
    first = ObjectiveRefinementEventTracker(
        AdaptiveGoalRefiner(generate, _verify, clock=lambda: 10),
        state_path,
        objective_committer=commit,
    ).poll(request)
    before = state_path.read_bytes()

    unchanged_delivery = replace(
        request,
        signals=(replace(request.signals[0], observed_at=999, occurrence_count=9),),
        cycle_id="cycle:later-poll",
    )
    restarted = ObjectiveRefinementEventTracker(
        AdaptiveGoalRefiner(generate, _verify, clock=lambda: 20),
        state_path,
        objective_committer=commit,
    ).poll(unchanged_delivery)

    assert first.decision is ObjectiveRefinementPollDecision.DELTA_COMMITTED
    assert restarted.decision is ObjectiveRefinementPollDecision.NO_SEMANTIC_CHANGE
    assert not restarted.model_called
    assert not restarted.objective_written
    assert calls == {"generator": 1, "writer": 1}
    assert state_path.read_bytes() == before


def test_changed_event_reopens_poll_but_frozen_context_cannot_drift(
    tmp_path,
) -> None:
    calls = {"generator": 0}

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    tracker = ObjectiveRefinementEventTracker(
        AdaptiveGoalRefiner(generate, _verify, clock=lambda: 10),
        tmp_path / "events.json",
    )
    request = _request()
    tracker.poll(request)
    changed = tracker.poll(
        replace(
            request,
            signals=(_signal(revision="revision:2"),),
            cycle_id="cycle:changed",
        )
    )
    assert changed.decision is ObjectiveRefinementPollDecision.REFINEMENT_EVALUATED
    assert changed.model_called
    assert calls == {"generator": 2}

    with pytest.raises(AdaptiveGoalRefinementError, match="frozen root"):
        tracker.poll(
            replace(
                request,
                assumption_ids=("assumption:operator-mutated",),
                signals=(_signal(revision="revision:3"),),
            )
        )


def test_multiple_events_in_one_semantic_slot_do_not_create_poll_churn(
    tmp_path,
) -> None:
    calls = {"generator": 0}

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    tracker = ObjectiveRefinementEventTracker(
        AdaptiveGoalRefiner(generate, _verify, clock=lambda: 10),
        tmp_path / "events.json",
    )
    request = replace(
        _request(),
        signals=(
            _signal(revision="counterexample:a"),
            _signal(revision="counterexample:b"),
        ),
    )
    first = tracker.poll(request)
    unchanged = tracker.poll(
        replace(
            request,
            signals=tuple(
                replace(item, observed_at=1000 + index)
                for index, item in enumerate(request.signals)
            ),
        )
    )
    assert first.model_called
    assert unchanged.decision is ObjectiveRefinementPollDecision.NO_SEMANTIC_CHANGE
    assert not unchanged.model_called
    assert calls == {"generator": 1}
