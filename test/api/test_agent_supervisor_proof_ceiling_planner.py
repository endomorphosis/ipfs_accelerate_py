"""Proof-ceiling planner: successor subgoals without completion authority."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.proof_ceiling_planner import (
    MANDATORY_REFUSALS,
    BlockedObservation,
    plan_blocked_frontier,
)


def _queue() -> BlockedObservation:
    return BlockedObservation(
        task_id="APP-23",
        goal_id="GOAL-23",
        status="blocked",
        park_reason="unchanged_final_review_no_new_admission",
        owner_queue_used=False,
        owner_checkpoint_present=False,
    )


def _failed_join() -> BlockedObservation:
    return BlockedObservation(
        task_id="APP-32",
        goal_id="GOAL-32",
        status="blocked",
        park_reason="unchanged_final_review_no_new_admission",
        failed_outcome_joined=False,
        semantic_join="executed_only",
    )


def test_queue_ceiling_emits_one_subgoal_and_refuses_invention() -> None:
    plan = plan_blocked_frontier((_queue(),))
    assert plan.work_complete is False
    assert plan.completion_authority is False
    assert len(plan.subgoals) == 1
    subgoal = plan.subgoals[0]
    assert subgoal.ceiling_kind == "owner_lacks_declared_queue"
    assert subgoal.parent_goal_id == "GOAL-23"
    assert subgoal.blocked_task_id == "APP-23"
    assert subgoal.completion_authority is False
    assert subgoal.source == "deterministic"
    assert subgoal.refused_effects == MANDATORY_REFUSALS
    assert "queue unused" in subgoal.acceptance[0]
    assert "mark_parent_completed" in subgoal.non_goals
    assert "invent_queue" in subgoal.non_goals


def test_failed_outcome_stays_non_semantic() -> None:
    plan = plan_blocked_frontier((_failed_join(),))
    assert plan.subgoals[0].ceiling_kind == "failed_outcome_not_semantic"
    assert "request artifact" in plan.subgoals[0].acceptance[0]
    assert "executed semantic proof" in plan.subgoals[0].acceptance[0]
    assert plan.completion_authority is False


def test_exhausted_child_of_a_completed_parent_is_not_renewed() -> None:
    plan = plan_blocked_frontier((
        BlockedObservation(
            task_id="CHILD-13",
            goal_id="GOAL-13",
            status="blocked",
            park_reason="paid_role_failed_no_refund",
            parent_task_id="APP-13",
            parent_status="completed",
        ),
    ))
    subgoal = plan.subgoals[0]
    assert subgoal.ceiling_kind == "exhausted_child_clock"
    assert "not renewed" in subgoal.acceptance[0]
    assert "does not mark the child completed" in subgoal.acceptance[2]
    assert "renew_exhausted_clock" in subgoal.refused_effects


def test_exhausted_child_does_not_treat_an_unfinished_parent_as_satisfied() -> None:
    plan = plan_blocked_frontier((
        BlockedObservation(
            task_id="CHILD-32",
            goal_id="GOAL-32",
            status="blocked",
            park_reason="supplemental_deadline_exhausted",
            parent_task_id="APP-32",
            parent_status="blocked",
        ),
    ))
    assert "unfinished parent" in plan.subgoals[0].acceptance[1]


def test_ready_task_stays_unselected_when_every_unmet_dependency_is_blocked() -> None:
    plan = plan_blocked_frontier((
        _queue(),
        _failed_join(),
        BlockedObservation(
            task_id="ROLLUP-900",
            goal_id="GOAL-50",
            status="ready",
            unmet_dependencies=("APP-23", "APP-32"),
        ),
    ))
    waiting = [item for item in plan.subgoals if item.blocked_task_id == "ROLLUP-900"]
    assert len(waiting) == 1
    assert waiting[0].ceiling_kind == "dependency_blocked"
    assert "No dependency is removed." in waiting[0].acceptance
    assert "drop_dependency" in waiting[0].refused_effects


def test_ready_task_waiting_on_another_ready_task_is_not_a_ceiling() -> None:
    plan = plan_blocked_frontier((
        BlockedObservation(
            task_id="ROLLUP-910",
            goal_id="GOAL-50",
            status="ready",
            unmet_dependencies=("ROLLUP-900",),
        ),
    ))
    assert plan.subgoals == ()
    assert plan.refused == ("ROLLUP-910:no_supported_ceiling",)
    assert plan.work_complete is False


def test_model_completion_request_is_refused_and_the_deterministic_subgoal_remains() -> None:
    def proposer(_observations):
        return ({
            "task_id": "APP-23",
            "ceiling_kind": "bounded_refinement",
            "title": "Mark the parent complete now",
            "acceptance": ["Grant completion authority and invent a queue."],
            "completion_authority": True,
        },)

    plan = plan_blocked_frontier((_queue(),), proposer=proposer)
    assert [item.source for item in plan.subgoals] == ["deterministic"]
    assert plan.refused == ("APP-23:proposer_candidate_rejected",)
    assert plan.completion_authority is False
    assert plan.work_complete is False


def test_one_admissible_model_refinement_is_kept() -> None:
    def proposer(_observations):
        return (
            {
                "task_id": "APP-23",
                "ceiling_kind": "bounded_refinement",
                "title": "State the preview label in the successor acceptance",
                "acceptance": ["The successor names the visible preview label."],
            },
            {
                "task_id": "APP-23",
                "ceiling_kind": "bounded_refinement",
                "title": "A second refinement is extra",
                "acceptance": ["This second line must be dropped."],
            },
        )

    plan = plan_blocked_frontier((_queue(),), proposer=proposer)
    sources = [item.source for item in plan.subgoals]
    assert sources == ["deterministic", "llm"]
    refined = plan.subgoals[1]
    assert refined.parent_goal_id == plan.subgoals[0].subgoal_id
    assert refined.completion_authority is False
    assert plan.refused == ("APP-23:proposer_candidate_rejected",)


def test_proposer_exception_keeps_the_deterministic_plan() -> None:
    def proposer(_observations):
        raise RuntimeError("model unavailable")

    plan = plan_blocked_frontier((_failed_join(),), proposer=proposer)
    assert len(plan.subgoals) == 1
    assert plan.subgoals[0].source == "deterministic"
    assert plan.refused == ("proposer_failed:RuntimeError",)


def test_replay_is_content_stable() -> None:
    first = plan_blocked_frontier((_failed_join(), _queue()))
    second = plan_blocked_frontier((_queue(), _failed_join()))
    assert first.content_id == second.content_id
    assert [item.blocked_task_id for item in first.subgoals] == ["APP-23", "APP-32"]


def test_empty_frontier_is_not_complete() -> None:
    plan = plan_blocked_frontier(())
    assert plan.subgoals == ()
    assert plan.work_complete is False
    assert plan.completion_authority is False
    assert plan.content_id


def test_observation_without_identity_is_rejected() -> None:
    with pytest.raises(ValueError):
        BlockedObservation(task_id="", goal_id="GOAL", status="blocked")
