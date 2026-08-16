"""W6 integration: readiness, optimizer, and proposal-only transforms."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.planning.conflict_free_frontier import (
    construct_frontier,
)
from ipfs_accelerate_py.agent_supervisor.planning.frontier_optimizer import (
    optimize_frontier,
)
from ipfs_accelerate_py.agent_supervisor.planning.parallel_plan_compiler import (
    propose_plan_transform,
)


def plan_frontier(
    tasks: Sequence[Mapping[str, Any]],
    *,
    capacity: int,
    conflicts: Sequence[tuple[str, str]] = (),
    transform: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    constructed = construct_frontier(tasks, conflicts=conflicts)
    selected_tasks = [
        task for task in tasks if str(task.get("task_id")) in constructed["candidates"]
    ]
    optimized = optimize_frontier(
        selected_tasks, capacity=capacity, conflicts=conflicts
    )
    proposal = propose_plan_transform(transform) if transform else None
    return MappingProxyType(
        {
            "candidates": constructed["candidates"],
            "rejected": constructed["rejected"],
            "selected": optimized["selected"],
            "scores": optimized["scores"],
            "proposal": proposal,
            "reserved": False,
            "dispatched": False,
        }
    )
