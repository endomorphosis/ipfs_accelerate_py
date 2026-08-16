"""Focused FrontierOptimizer@1 checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.frontier_optimizer import (
    OptimizerError,
    optimize_frontier,
    score_task,
)


def test_small_set_matches_exhaustive_and_is_deterministic() -> None:
    tasks = [
        {"task_id": "A", "completion_value": 5, "resource_cost": 2},
        {"task_id": "B", "completion_value": 4, "resource_cost": 2},
        {"task_id": "C", "completion_value": 3, "resource_cost": 1},
    ]
    first = optimize_frontier(tasks, capacity=3, conflicts=(("A", "B"),))
    second = optimize_frontier(list(reversed(tasks)), capacity=3, conflicts=(("B", "A"),))
    assert first == second
    assert first["algorithm"] == "exact-bounded"
    assert first["selected"] == ("A", "C")


def test_tie_break_is_task_id_and_floats_are_rejected() -> None:
    tasks = [
        {"task_id": "B", "completion_value": 2, "resource_cost": 1},
        {"task_id": "A", "completion_value": 2, "resource_cost": 1},
    ]
    result = optimize_frontier(tasks, capacity=1)
    assert result["selected"] == ("A",)
    with pytest.raises(OptimizerError, match="int"):
        score_task({"completion_value": 1.5})
