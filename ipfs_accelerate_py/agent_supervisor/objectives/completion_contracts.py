"""GoalCompletionContract@1 and TaskCompletionContract@1 evaluators."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

GOAL_SCHEMA = "lgswf/goal-completion-contract@1"
TASK_SCHEMA = "lgswf/task-completion-contract@1"

TASK_GATES = (
    "worker",
    "validation",
    "proof",
    "merge",
    "canonical_refresh",
    "supervisor_acceptance",
)

GOAL_CONJUNCTION = (
    "observable_state",
    "semantic_properties",
    "accepted_children",
    "tests_proofs",
    "resolved_counterexamples",
    "resolved_gaps",
    "accepted_tree_root",
    "human_approval",
)


class CompletionContractError(ValueError):
    """A completion contract evaluation was rejected."""


def evaluate_task_completion(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("self_approved") or record.get("worker_completes_task"):
        raise CompletionContractError("worker/model self-approval is forbidden")
    missing = [gate for gate in TASK_GATES if not record.get(gate)]
    return MappingProxyType(
        {
            "schema": TASK_SCHEMA,
            "accepted": not missing,
            "missing_gates": tuple(missing),
            "authority": "supervisor",
        }
    )


def evaluate_goal_completion(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if record.get("task_count_completion"):
        raise CompletionContractError("task-count completion is forbidden")
    if record.get("stale_evidence"):
        raise CompletionContractError("stale evidence cannot complete a goal")
    missing = [item for item in GOAL_CONJUNCTION if not record.get(item)]
    return MappingProxyType(
        {
            "schema": GOAL_SCHEMA,
            "accepted": not missing,
            "missing": tuple(missing),
            "authority": "supervisor",
        }
    )


def completion_truth_table() -> tuple[dict[str, Any], ...]:
    return (
        {"kind": "task", "gates": TASK_GATES, "self_approval": False},
        {"kind": "goal", "conjunction": GOAL_CONJUNCTION, "task_count": False},
    )
