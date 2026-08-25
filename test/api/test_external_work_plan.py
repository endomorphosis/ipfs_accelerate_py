"""EAAEF-071: bounded task decompositions over ExternalGoalContract."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import (
    ExternalWorkPlan,
    WorkPlanError,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)


WRITE_SCOPE = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py"


def _goal(**overrides):
    payload = {
        "objective_id": "EAAEF-G020",
        "desired_outcomes": ("normalize export", "preserve identities"),
        "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
        "write_scope": (WRITE_SCOPE,),
        "authority_ceiling": "preview_only",
        "verification_requirements": ("focused pytest",),
        "proof_requirements": ("content identity",),
        "review_requirements": ("independent supervisor",),
        "completion_evidence": ("test receipt", "patch identity"),
        "timeout_seconds": 7200,
        "cpu_millicores": 4000,
        "ram_mib": 8192,
    }
    payload.update(overrides)
    return ExternalGoalContract.compile(payload)


def _task(**overrides):
    payload = {
        "task_id": "task-a",
        "covers": ("normalize export", "preserve identities"),
        "write_scope": (WRITE_SCOPE,),
        "depends_on": (),
        "timeout_seconds": 600,
        "cpu_millicores": 1000,
        "ram_mib": 1024,
    }
    payload.update(overrides)
    return payload


def test_decompose_covers_goal_outcomes() -> None:
    plan = ExternalWorkPlan.decompose(_goal(), (_task(),))
    assert isinstance(plan.formal_plan, FormalWorkPlan)
    assert {outcome for task in plan.tasks for outcome in task.covers} >= set(
        plan.goal.desired_outcomes
    )
    assert plan.formal_plan.plan_id.startswith("b")
    assert plan.content_id.startswith("b")
    clone = ExternalWorkPlan.decompose(_goal(), (_task(),))
    assert clone.content_id == plan.content_id
    assert clone.formal_plan.plan_id == plan.formal_plan.plan_id


def test_rejects_cycles_and_duplicate_task_ids() -> None:
    with pytest.raises(WorkPlanError, match="acyclic"):
        ExternalWorkPlan.decompose(
            _goal(),
            (
                _task(task_id="a", covers=("normalize export",), depends_on=("b",)),
                _task(task_id="b", covers=("preserve identities",), depends_on=("a",)),
            ),
        )
    with pytest.raises(WorkPlanError, match="duplicate task ids"):
        ExternalWorkPlan.decompose(_goal(), (_task(), _task()))


def test_rejects_write_scope_outside_goal() -> None:
    with pytest.raises(WorkPlanError, match="write-scope"):
        ExternalWorkPlan.decompose(
            _goal(),
            (_task(write_scope=("secrets/passwd",)),),
        )


def test_rejects_missing_coverage() -> None:
    with pytest.raises(WorkPlanError, match="coverage"):
        ExternalWorkPlan.decompose(
            _goal(),
            (_task(covers=("normalize export",)),),
        )


def test_rejects_contradictions_resources_and_duplicate_semantics() -> None:
    with pytest.raises(WorkPlanError, match="contradiction"):
        ExternalWorkPlan.decompose(
            _goal(),
            (
                _task(
                    covers=("normalize export", "preserve identities", "self_approve"),
                ),
            ),
        )
    with pytest.raises(WorkPlanError, match="resource"):
        ExternalWorkPlan.decompose(_goal(), (_task(cpu_millicores=9000),))
    with pytest.raises(WorkPlanError, match="proof"):
        ExternalWorkPlan.decompose(_goal(), (_task(proof_feasible=False),))
    with pytest.raises(WorkPlanError, match="duplicate"):
        ExternalWorkPlan.decompose(
            _goal(),
            (
                _task(task_id="left", covers=("normalize export",)),
                _task(task_id="right", covers=("preserve identities",)),
            ),
        )
