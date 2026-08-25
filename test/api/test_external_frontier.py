"""EAAEF-082: conflict-free frontier under deps and CPU budget."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierError,
    FrontierTask,
    select_frontier,
)


def test_selects_disjoint_ready_tasks_under_budget() -> None:
    tasks = (
        FrontierTask("a", (), ("a.py",), ("write-a",), 1000),
        FrontierTask("b", (), ("b.py",), ("write-b",), 1000),
        FrontierTask("c", ("a",), ("c.py",), ("write-c",), 1000),
    )
    frontier = select_frontier(tasks, cpu_budget=3000)
    assert frontier["task_ids"] == ["a", "b"]
    frontier2 = select_frontier(tasks, cpu_budget=3000, completed_ids=("a",))
    assert "c" in frontier2["task_ids"]
    assert "a" not in frontier2["task_ids"]


def test_overlapping_writes_serialize() -> None:
    tasks = (
        FrontierTask("a", (), ("shared.py",), ("write",), 100),
        FrontierTask("b", (), ("shared.py",), ("write",), 100),
    )
    frontier = select_frontier(tasks, cpu_budget=1000)
    assert frontier["task_ids"] == ["a"]


def test_cpu_budget_must_be_positive() -> None:
    with pytest.raises(FrontierError):
        select_frontier((), cpu_budget=0)
