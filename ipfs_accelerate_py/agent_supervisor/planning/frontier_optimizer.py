"""FrontierOptimizer@1 — bounded conflict-free antichain selection."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA = "lgswf/frontier-optimizer@1"
MAX_SCORE = 2**31 - 1
EXACT_BOUND = 8


class OptimizerError(ValueError):
    """Optimizer rejected a non-integer or infeasible score input."""


def _checked(value: int, name: str) -> int:
    if type(value) is not int:
        raise OptimizerError(f"{name} must be an int, not a float or bool")
    if value < -MAX_SCORE or value > MAX_SCORE:
        raise OptimizerError(f"{name} overflow/bounds")
    return value


def score_task(task: Mapping[str, Any]) -> int:
    positive = (
        _checked(task.get("completion_value", 0), "completion_value")
        + _checked(task.get("critical_path_reduction", 0), "critical_path_reduction")
        + _checked(task.get("downstream_unlock", 0), "downstream_unlock")
        + _checked(task.get("priority", 0), "priority")
        + _checked(task.get("age_fairness", 0), "age_fairness")
        + _checked(task.get("locality", 0), "locality")
    )
    negative = (
        _checked(task.get("resource_cost", 0), "resource_cost")
        + _checked(task.get("provider_cost", 0), "provider_cost")
        + _checked(task.get("proof_cost", 0), "proof_cost")
        + _checked(task.get("conflict_uncertainty", 0), "conflict_uncertainty")
        + _checked(task.get("retry_risk", 0), "retry_risk")
        + _checked(task.get("merge_congestion", 0), "merge_congestion")
    )
    return _checked(positive - negative, "score")


def _feasible(selected: Sequence[Mapping[str, Any]], capacity: int) -> bool:
    return sum(int(task.get("resource_cost", 0)) for task in selected) <= capacity


def optimize_frontier(
    tasks: Sequence[Mapping[str, Any]],
    *,
    capacity: int,
    conflicts: Sequence[tuple[str, str]] = (),
) -> Mapping[str, Any]:
    ranked = sorted(tasks, key=lambda task: (-score_task(task), str(task.get("task_id") or "")))
    blocked = {tuple(sorted(pair)) for pair in conflicts}

    def conflicts_with(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        pair = tuple(sorted((str(left.get("task_id")), str(right.get("task_id")))))
        return pair in blocked

    if len(ranked) <= EXACT_BOUND:
        best: tuple[Mapping[str, Any], ...] = ()
        best_score = None
        limit = 1 << len(ranked)
        for mask in range(limit):
            chosen = [ranked[index] for index in range(len(ranked)) if mask & (1 << index)]
            if any(
                conflicts_with(chosen[i], chosen[j])
                for i in range(len(chosen))
                for j in range(i + 1, len(chosen))
            ):
                continue
            if not _feasible(chosen, capacity):
                continue
            total = sum(score_task(task) for task in chosen)
            ids = tuple(str(task.get("task_id")) for task in chosen)
            if best_score is None or total > best_score or (total == best_score and ids < tuple(str(t.get("task_id")) for t in best)):
                best = tuple(chosen)
                best_score = total
        selected = best
        algorithm = "exact-bounded"
    else:
        selected_list: list[Mapping[str, Any]] = []
        for task in ranked:
            if any(conflicts_with(task, other) for other in selected_list):
                continue
            trial = selected_list + [task]
            if _feasible(trial, capacity):
                selected_list = trial
        selected = tuple(selected_list)
        algorithm = "greedy-local"

    return MappingProxyType(
        {
            "schema": SCHEMA,
            "algorithm": algorithm,
            "selected": tuple(str(task.get("task_id")) for task in selected),
            "scores": MappingProxyType(
                {str(task.get("task_id")): score_task(task) for task in tasks}
            ),
        }
    )
