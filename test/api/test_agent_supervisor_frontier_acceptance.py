"""W6 frontier integration acceptance."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.planning.conflict_free_frontier import (
    READINESS_PREDICATES,
)
from ipfs_accelerate_py.agent_supervisor.planning.frontier_integration import plan_frontier


def _ready(task_id: str, **fields: object) -> dict:
    payload = {name: True for name in READINESS_PREDICATES}
    payload.update(fields)
    payload["task_id"] = task_id
    payload.setdefault("completion_value", 1)
    payload.setdefault("resource_cost", 1)
    return payload


def test_identical_snapshot_replays_and_does_not_dispatch() -> None:
    tasks = [_ready("A", completion_value=3), _ready("B", completion_value=2)]
    first = plan_frontier(tasks, capacity=1, conflicts=())
    second = plan_frontier(list(reversed(tasks)), capacity=1, conflicts=())
    assert first["selected"] == second["selected"] == ("A",)
    assert first["reserved"] is False
    assert first["dispatched"] is False
    assert first["rejected"] == second["rejected"]
