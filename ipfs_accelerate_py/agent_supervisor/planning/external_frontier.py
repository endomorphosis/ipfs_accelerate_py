"""Resource-aware conflict-free frontier selection (EAAEF-082)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .external_conflict_graph import ConflictGraph


FRONTIER_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-frontier@1"


class FrontierError(ValueError):
    """Frontier selection failed closed."""


@dataclass(frozen=True)
class FrontierTask:
    task_id: str
    depends_on: tuple[str, ...]
    write_scope: tuple[str, ...]
    effect_scope: tuple[str, ...]
    cpu_millicores: int
    completed: bool = False

    def as_scope(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "write_scope": list(self.write_scope),
            "effect_scope": list(self.effect_scope),
        }


def select_frontier(
    tasks: Sequence[FrontierTask],
    *,
    cpu_budget: int,
    completed_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Deterministic ready antichain under deps, conflicts, and CPU budget."""

    if cpu_budget <= 0:
        raise FrontierError("cpu_budget must be positive")
    done = set(completed_ids) | {task.task_id for task in tasks if task.completed}
    ready: list[FrontierTask] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        if task.task_id in done:
            continue
        if any(dep not in done for dep in task.depends_on):
            continue
        ready.append(task)
    selected: list[str] = []
    used = 0
    for task in ready:
        if used + int(task.cpu_millicores) > cpu_budget:
            continue
        conflicted = False
        for other_id in selected:
            other = next(item for item in tasks if item.task_id == other_id)
            derived = ConflictGraph.derive(task.as_scope(), other.as_scope())
            if derived.conflicts:
                conflicted = True
                break
        if conflicted:
            continue
        selected.append(task.task_id)
        used += int(task.cpu_millicores)
    return MappingProxyType(
        {
            "schema": FRONTIER_SCHEMA,
            "task_ids": selected,
            "cpu_used": used,
            "cpu_budget": int(cpu_budget),
        }
    )
