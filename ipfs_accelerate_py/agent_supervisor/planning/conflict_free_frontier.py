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
    result = MappingProxyType(
        {
            "task_id": str(task.get("task_id") or ""),
            "ready": not reasons,
            "reasons": tuple(reasons),
        }
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        task_id = str(result.get("task_id") or "readiness")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="task_readiness",
            record_ref=task_id,
            subject_kind="task_id",
            subject_ref=task_id,
        )
    except Exception:
        pass
    return result


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
    frontier = MappingProxyType(
        {
            "schema": SCHEMA,
            "candidates": selected,
            "rejected": MappingProxyType(rejected),
            "predicates": READINESS_PREDICATES,
        }
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            compose_semantic_work,
            mirror_work_record,
        )

        first = str((frontier.get("candidates") or ("frontier",))[0] or "frontier")
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="conflict_free_frontier",
            record_ref=first,
            subject_kind="task_id",
            subject_ref=first,
        )
        compose_semantic_work(subject_kind="task_id", subject_ref=first)
    except Exception:
        pass
    return frontier
