"""EAAEF-085: overlapping writes/effects serialize; disjoint work may run in parallel."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.external_conflict_graph import (
    ConflictGraph,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierError,
    FrontierTask,
    select_frontier,
)


def test_overlapping_writes_or_same_effects_conflict() -> None:
    overlapping_writes = (
        FrontierTask("write-a", (), ("shared.py",), ("edit",), 100),
        FrontierTask("write-b", (), ("shared.py",), ("format",), 100),
    )
    derived_writes = ConflictGraph.derive(
        overlapping_writes[0].as_scope(),
        overlapping_writes[1].as_scope(),
    )
    assert derived_writes.conflicts is True
    frontier_writes = select_frontier(overlapping_writes, cpu_budget=1000)
    assert frontier_writes["task_ids"] == ["write-a"]

    same_effects = (
        FrontierTask("fx-a", (), ("a.py",), ("merge",), 100),
        FrontierTask("fx-b", (), ("b.py",), ("merge",), 100),
    )
    derived_effects = ConflictGraph.derive(
        same_effects[0].as_scope(),
        same_effects[1].as_scope(),
    )
    assert derived_effects.conflicts is True
    frontier_effects = select_frontier(same_effects, cpu_budget=1000)
    assert frontier_effects["task_ids"] == ["fx-a"]


def test_distinct_effects_and_writes_can_be_parallel() -> None:
    tasks = (
        FrontierTask("left", (), ("left.py",), ("edit-left",), 1000),
        FrontierTask("right", (), ("right.py",), ("edit-right",), 1000),
    )
    derived = ConflictGraph.derive(tasks[0].as_scope(), tasks[1].as_scope())
    assert derived.conflicts is False
    frontier = select_frontier(tasks, cpu_budget=3000)
    assert frontier["task_ids"] == ["left", "right"]
    assert frontier["cpu_used"] == 2000


def test_cpu_budget_and_completed_deps_are_required() -> None:
    tasks = (
        FrontierTask("root", (), ("root.py",), ("edit-root",), 1500),
        FrontierTask("child", ("root",), ("child.py",), ("edit-child",), 500),
        FrontierTask("other", (), ("other.py",), ("edit-other",), 1500),
    )
    tight = select_frontier(tasks, cpu_budget=1500)
    assert "child" not in tight["task_ids"]
    assert tight["cpu_used"] <= 1500
    assert set(tight["task_ids"]) <= {"other", "root"}
    assert len(tight["task_ids"]) == 1
    after_root = select_frontier(tasks, cpu_budget=2000, completed_ids=("root",))
    assert "root" not in after_root["task_ids"]
    assert "child" in after_root["task_ids"]
    assert after_root["cpu_used"] <= 2000
    with pytest.raises(FrontierError, match="cpu_budget"):
        select_frontier(tasks, cpu_budget=0)
