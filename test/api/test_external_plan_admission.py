"""EAAEF-073: deterministic scoring and single-plan admission."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import (
    ExternalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission import (
    SCORE_AXES,
    PlanAdmissionError,
    PlanCandidate,
    admit_plan,
    logic_gate,
    rank_plans,
    score_plan,
)


WRITE_A = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py"
WRITE_B = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/claude.py"


def _goal(**overrides):
    payload = {
        "objective_id": "EAAEF-G080",
        "desired_outcomes": ("normalize export", "preserve identities"),
        "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
        "write_scope": (WRITE_A, WRITE_B),
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


def _task(task_id: str, covers, write_scope, **overrides):
    payload = {
        "task_id": task_id,
        "covers": covers,
        "write_scope": write_scope,
        "depends_on": (),
        "timeout_seconds": 600,
        "cpu_millicores": 1000,
        "ram_mib": 1024,
    }
    payload.update(overrides)
    return payload


def _sequential_plan() -> ExternalWorkPlan:
    return ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task(
                "task-b",
                ("preserve identities",),
                (WRITE_B,),
                depends_on=("task-a",),
            ),
        ),
    )


def _parallel_plan() -> ExternalWorkPlan:
    return ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task("task-b", ("preserve identities",), (WRITE_B,)),
        ),
    )


def test_admits_exactly_one_deterministic_plan() -> None:
    sequential = _sequential_plan()
    parallel = _parallel_plan()
    first = admit_plan((sequential, parallel))
    second = admit_plan((parallel, sequential))
    assert first.admitted.content_id == parallel.content_id
    assert second.admitted.content_id == first.admitted.content_id
    assert first.content_id == second.content_id
    assert len(first.ranked) == 2
    assert {item.content_id for item in first.ranked} == {
        sequential.content_id,
        parallel.content_id,
    }
    assert first.to_dict()["verdict"] == "admitted"
    assert first.admitted_id == first.ranked[0].candidate_id


def test_rejects_empty_candidates() -> None:
    with pytest.raises(PlanAdmissionError, match="candidates"):
        admit_plan(())
    with pytest.raises(PlanAdmissionError, match="candidates"):
        rank_plans([])
    with pytest.raises(PlanAdmissionError, match="candidates"):
        logic_gate(())


def test_scores_named_axes_as_integers() -> None:
    scored = score_plan(_parallel_plan())
    assert tuple(scored.components) == SCORE_AXES
    assert SCORE_AXES == (
        "critical_path",
        "safe_width",
        "model_proof_cost",
        "resources",
        "merge_risk",
        "uncertainty",
        "prior_success",
        "cache_locality",
    )
    for name in SCORE_AXES:
        assert type(scored.components[name]) is int
        assert type(scored.axis_scores[name]) is int
        assert scored.components[name] >= 0
        assert scored.axis_scores[name] >= 0
    assert type(scored.total) is int
    assert scored.components["safe_width"] == 2
    sequential = score_plan(_sequential_plan())
    assert sequential.components["safe_width"] == 1
    assert sequential.components["critical_path"] > scored.components["critical_path"]


def test_prefers_shorter_critical_path_and_wider_safe_width() -> None:
    sequential = score_plan(_sequential_plan())
    parallel = score_plan(_parallel_plan())
    assert parallel.components["critical_path"] < sequential.components["critical_path"]
    assert parallel.components["safe_width"] > sequential.components["safe_width"]
    winner = logic_gate((sequential, parallel))
    assert winner.content_id == parallel.content_id


def test_tie_breaks_by_candidate_id() -> None:
    plan = _parallel_plan()
    ranked = rank_plans(
        (
            PlanCandidate(plan=plan, candidate_id="zeta"),
            PlanCandidate(plan=plan, candidate_id="alpha"),
        )
    )
    assert [item.candidate_id for item in ranked] == ["alpha", "zeta"]
    admitted = admit_plan(
        (
            PlanCandidate(plan=plan, candidate_id="zeta"),
            PlanCandidate(plan=plan, candidate_id="alpha"),
        )
    )
    assert admitted.admitted_id == "alpha"
    assert admitted.admitted.content_id == plan.content_id


def test_prior_success_and_cache_locality_can_win() -> None:
    sequential = _sequential_plan()
    parallel = _parallel_plan()
    default = admit_plan((sequential, parallel))
    assert default.admitted.content_id == parallel.content_id
    boosted = admit_plan(
        (
            PlanCandidate(plan=sequential, candidate_id="seq", prior_success=50),
            PlanCandidate(plan=parallel, candidate_id="par"),
        )
    )
    assert boosted.admitted_id == "seq"
    cached = admit_plan(
        (
            PlanCandidate(plan=sequential, candidate_id="seq"),
            PlanCandidate(plan=parallel, candidate_id="par"),
        ),
        cache_keys=(WRITE_A, "normalize export"),
        history={"seq": 40},
    )
    assert cached.admitted_id == "seq"
    scored = score_plan(
        PlanCandidate(plan=sequential, candidate_id="seq"),
        cache_keys=(WRITE_A,),
        history={"seq": 3},
    )
    assert scored.components["prior_success"] == 3
    assert scored.components["cache_locality"] == 1
