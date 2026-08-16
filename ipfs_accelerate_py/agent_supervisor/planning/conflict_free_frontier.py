"""ConflictFreeParallelFrontierPlanner@1 — readiness gates and candidate set."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA = "lgswf/conflict-free-frontier@1"

READINESS_PREDICATES = (
    "active_plan_member",
    "legal_lifecycle",
    "predecessors_satisfied",
    "current_binding",
    "fresh_capsules_or_raw_source",
    "contracts_obligations_resolvable",
    "scope_admitted",
    "no_active_conflicting_writer",
    "resources_reservable",
    "completion_policy_known",
    "not_blocked_superseded_quarantined",
    "no_human_hold",
)


class FrontierError(ValueError):
    """A frontier construction input was invalid."""


def evaluate_readiness(task: Mapping[str, Any]) -> Mapping[str, Any]:
    reasons = []
    for predicate in READINESS_PREDICATES:
        if not task.get(predicate, False):
            reasons.append(predicate)
    return MappingProxyType(
        {
            "task_id": str(task.get("task_id") or ""),
            "ready": not reasons,
            "reasons": tuple(reasons),
        }
    )


def construct_frontier(
    tasks: Sequence[Mapping[str, Any]],
    *,
    conflicts: Sequence[tuple[str, str]] = (),
) -> Mapping[str, Any]:
    evaluated = [dict(evaluate_readiness(task)) for task in tasks]
    evaluated.sort(key=lambda item: item["task_id"])
    candidates = [item["task_id"] for item in evaluated if item["ready"]]
    rejected = {
        item["task_id"]: item["reasons"] for item in evaluated if not item["ready"]
    }
    blocked = set()
    pair_set = {tuple(sorted(pair)) for pair in conflicts}
    for left, right in pair_set:
        if left in candidates and right in candidates:
            # Keep the stable lower task ID; reject the other for conflict.
            loser = max(left, right)
            blocked.add(loser)
            rejected.setdefault(loser, ())
            rejected[loser] = tuple(rejected[loser]) + ("conflict",)
    selected = tuple(task_id for task_id in candidates if task_id not in blocked)
    return MappingProxyType(
        {
            "schema": SCHEMA,
            "candidates": selected,
            "rejected": MappingProxyType(rejected),
            "predicates": READINESS_PREDICATES,
        }
    )
