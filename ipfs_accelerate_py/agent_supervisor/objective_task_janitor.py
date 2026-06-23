"""Reconcile supervisor todo strategy with the objective heap."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .objective_graph import ObjectiveGoal, objective_heap_schedule
from .todo_daemon.implementation_daemon import PortalTask

ACTIVE_GOAL_STATUSES = {"active", "todo", "open"}
OPEN_TASK_STATUSES = {"todo", "ready", "in_progress"}
JANITOR_RECEIPT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.objective_task_janitor.v1"
DEFAULT_MISSION_TERMS = (
    "ai virtual desktop",
    "desktop peer",
    "desktop offload",
    "end-to-end validation",
    "hallucinate app",
    "launch readiness",
    "launch replay",
    "meta glasses",
    "mobile phone",
    "offload",
    "phone",
    "playwright",
    "production",
    "swissknife",
    "virtual desktop",
)
GOAL_METADATA_KEYS = (
    "goal id",
    "goal ids",
    "goal packet goals",
    "graph parents",
)
CODEBASE_SCAN_BACKLOG_TITLE_PREFIXES = (
    "review swallowed exception path",
    "resolve code annotation",
)
CODEBASE_SCAN_BACKLOG_MARKERS = (
    "codebase scan filed this finding",
    "codebase refill scan",
)
GUARDRAIL_REPAIR_MARKERS = (
    "dependency guardrail",
    "generated dirty",
    "reconciliation guardrail",
    "retry budget",
    "retry-budget",
)


@dataclass(frozen=True)
class ObjectiveTaskJanitorReceipt:
    task_id: str
    action: str
    retired_task_reason: str
    goal_ids: list[str]
    title: str
    priority: str
    track: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique(values: Sequence[str]) -> list[str]:
    return [item for item in dict.fromkeys(str(value).strip() for value in values) if item]


def _split_terms(value: str) -> list[str]:
    terms = []
    for raw in re.split(r"[,;]", str(value or "")):
        term = " ".join(raw.strip().split())
        if term and term.lower() not in {"none", "n/a"}:
            terms.append(term)
    return _unique(terms)


def _task_goal_ids(task: PortalTask) -> list[str]:
    goal_ids: list[str] = []
    for key in GOAL_METADATA_KEYS:
        goal_ids.extend(_split_terms(task.metadata.get(key, "")))
    return _unique(goal_ids)


def _task_haystack(task: PortalTask) -> str:
    return " ".join(
        [
            task.task_id,
            task.title,
            task.track,
            task.priority,
            task.completion,
            task.acceptance,
            *task.outputs,
            *task.validation,
            *task.depends_on,
            *task.metadata.values(),
        ]
    ).lower()


def _goal_haystack(goal: ObjectiveGoal) -> str:
    return " ".join([goal.goal_id, goal.title, *goal.fields.values()]).lower()


def _matches_any_term(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def _is_generated_objective_task(task: PortalTask) -> bool:
    if any(task.metadata.get(key) for key in ("goal id", "missing evidence", "goal packet", "bundle shard")):
        return True
    return "objective scan" in task.title.lower() or "objective gap" in task.acceptance.lower()


def _is_guardrail_repair_task(task: PortalTask) -> bool:
    haystack = _task_haystack(task)
    return any(marker in haystack for marker in GUARDRAIL_REPAIR_MARKERS)


def _is_codebase_scan_backlog_task(task: PortalTask) -> bool:
    title = task.title.lower().strip()
    haystack = _task_haystack(task)
    return title.startswith(CODEBASE_SCAN_BACKLOG_TITLE_PREFIXES) or any(
        marker in haystack for marker in CODEBASE_SCAN_BACKLOG_MARKERS
    )


def _critical_goal_ids(goals: Sequence[ObjectiveGoal], mission_terms: Sequence[str]) -> set[str]:
    critical: set[str] = set()
    for goal in goals:
        if goal.status not in ACTIVE_GOAL_STATUSES:
            continue
        fields = goal.fields
        track = str(fields.get("track") or "").strip().lower()
        priority = str(fields.get("priority") or "").strip().upper()
        bundle = str(fields.get("bundle") or "").strip().lower()
        haystack = _goal_haystack(goal)
        if track == "launch" or _matches_any_term(haystack, mission_terms):
            critical.add(goal.goal_id)
            continue
        if priority == "P0" and bundle.startswith("objective/ops/"):
            critical.add(goal.goal_id)
    return critical


def _janitor_owned_task_ids(receipts: Sequence[Mapping[str, Any]], action: str) -> set[str]:
    return {
        str(receipt.get("task_id") or "")
        for receipt in receipts
        if str(receipt.get("schema") or "") == JANITOR_RECEIPT_SCHEMA
        and str(receipt.get("action") or "") == action
        and str(receipt.get("task_id") or "")
    }


def reconcile_objective_task_strategy(
    *,
    goals: Sequence[ObjectiveGoal],
    tasks: Sequence[PortalTask],
    strategy: Mapping[str, Any],
    now: str,
    mission_terms: Sequence[str] = DEFAULT_MISSION_TERMS,
    max_blocked_tasks: int = 50,
    max_deprioritized_tasks: int = 50,
    max_reopened_goals: int = 12,
) -> dict[str, Any]:
    """Return a strategy update that keeps todo work aligned with active goals."""

    goals_by_id = {goal.goal_id: goal for goal in goals if goal.goal_id}
    active_goal_ids = {goal.goal_id for goal in goals if goal.status in ACTIVE_GOAL_STATUSES}
    scheduled_goal_ids = [
        record.goal_id
        for record in objective_heap_schedule(goals)
        if record.goal_id in active_goal_ids
    ]
    critical_goal_ids = _critical_goal_ids(goals, mission_terms)
    open_tasks = [task for task in tasks if task.status in OPEN_TASK_STATUSES]
    open_goal_ids: set[str] = set()
    block_receipts: list[ObjectiveTaskJanitorReceipt] = []
    deprioritize_receipts: list[ObjectiveTaskJanitorReceipt] = []

    for task in open_tasks:
        goal_ids = _task_goal_ids(task)
        task_goal_set = set(goal_ids)
        if task_goal_set & active_goal_ids:
            open_goal_ids.update(task_goal_set & active_goal_ids)
            continue

        goal_known = [goal_id for goal_id in goal_ids if goal_id in goals_by_id]
        goal_missing = [goal_id for goal_id in goal_ids if goal_id not in goals_by_id]
        task_text = _task_haystack(task)
        mission_aligned = task.priority == "P0" or _matches_any_term(task_text, mission_terms)
        if goal_known or goal_missing:
            reason = "goal_not_active"
            if goal_known and all(goals_by_id[goal_id].status == "completed" for goal_id in goal_known):
                reason = "goal_completed"
            elif goal_missing and not goal_known:
                reason = "orphaned_goal_reference"
            block_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="block",
                    retired_task_reason=reason,
                    goal_ids=goal_ids,
                    title=task.title,
                    priority=task.priority,
                    track=task.track,
                )
            )
            continue

        if _is_generated_objective_task(task) and not mission_aligned:
            deprioritize_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="deprioritize",
                    retired_task_reason="off_mission_objective_generated_task",
                    goal_ids=goal_ids,
                    title=task.title,
                    priority=task.priority,
                    track=task.track,
                )
            )
            continue

        if (
            _is_codebase_scan_backlog_task(task)
            and not mission_aligned
            and not _is_guardrail_repair_task(task)
        ):
            deprioritize_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="deprioritize",
                    retired_task_reason="off_mission_codebase_scan_task",
                    goal_ids=goal_ids,
                    title=task.title,
                    priority=task.priority,
                    track=task.track,
                )
            )

    block_receipts = block_receipts[: max(0, int(max_blocked_tasks))]
    deprioritize_receipts = deprioritize_receipts[: max(0, int(max_deprioritized_tasks))]
    blocked_task_ids = [receipt.task_id for receipt in block_receipts]
    deprioritized_task_ids = [receipt.task_id for receipt in deprioritize_receipts]
    reopened_goal_ids = [
        goal_id
        for goal_id in scheduled_goal_ids
        if goal_id in critical_goal_ids and goal_id not in open_goal_ids
    ][: max(0, int(max_reopened_goals))]

    previous_receipts = [
        receipt
        for receipt in strategy.get("objective_task_janitor_receipts", [])
        if isinstance(receipt, Mapping)
    ] if isinstance(strategy.get("objective_task_janitor_receipts"), list) else []
    previously_blocked = _janitor_owned_task_ids(previous_receipts, "block")
    previously_deprioritized = _janitor_owned_task_ids(previous_receipts, "deprioritize")

    existing_blocked = _unique(
        [str(item) for item in strategy.get("blocked_tasks", [])]
        if isinstance(strategy.get("blocked_tasks"), list)
        else []
    )
    existing_deprioritized = _unique(
        [str(item) for item in strategy.get("deprioritized_tasks", [])]
        if isinstance(strategy.get("deprioritized_tasks"), list)
        else []
    )
    next_blocked = _unique(
        [task_id for task_id in existing_blocked if task_id not in previously_blocked] + blocked_task_ids
    )
    next_deprioritized = _unique(
        [task_id for task_id in existing_deprioritized if task_id not in previously_deprioritized]
        + deprioritized_task_ids
    )

    receipt_payloads = []
    for receipt in [*block_receipts, *deprioritize_receipts]:
        payload = receipt.to_dict()
        payload["schema"] = JANITOR_RECEIPT_SCHEMA
        payload["recorded_at"] = now
        receipt_payloads.append(payload)

    updated_strategy = dict(strategy)
    updated_strategy["blocked_tasks"] = next_blocked
    updated_strategy["deprioritized_tasks"] = next_deprioritized
    updated_strategy["objective_task_janitor_receipts"] = receipt_payloads
    updated_strategy["heap_goal_retirement_receipt"] = [
        payload for payload in receipt_payloads if payload.get("action") == "block"
    ]
    updated_strategy["objective_task_janitor_reopen_goal_ids"] = reopened_goal_ids
    updated_strategy["objective_task_janitor_force_goal_ids"] = reopened_goal_ids
    updated_strategy["objective_task_janitor_critical_goal_ids"] = sorted(critical_goal_ids)
    updated_strategy["objective_task_janitor_mission_terms"] = list(mission_terms)
    updated_strategy["objective_task_janitor_active_goal_ids"] = sorted(active_goal_ids)
    updated_strategy["objective_task_janitor_heap_schedule_goal_ids"] = scheduled_goal_ids
    updated_strategy["objective_task_janitor_last_run_at"] = now
    updated_strategy["objective_task_janitor_last_run_summary"] = {
        "blocked_count": len(blocked_task_ids),
        "deprioritized_count": len(deprioritized_task_ids),
        "reopened_goal_count": len(reopened_goal_ids),
        "active_goal_count": len(active_goal_ids),
        "scheduled_goal_count": len(scheduled_goal_ids),
    }

    comparable_keys = (
        "blocked_tasks",
        "deprioritized_tasks",
        "objective_task_janitor_receipts",
        "heap_goal_retirement_receipt",
        "objective_task_janitor_reopen_goal_ids",
        "objective_task_janitor_force_goal_ids",
        "objective_task_janitor_mission_terms",
    )
    changed = any(strategy.get(key) != updated_strategy.get(key) for key in comparable_keys)
    return {
        "changed": changed,
        "strategy": updated_strategy,
        "blocked_task_ids": blocked_task_ids,
        "deprioritized_task_ids": deprioritized_task_ids,
        "reopened_goal_ids": reopened_goal_ids,
        "receipts": receipt_payloads,
        "active_goal_ids": sorted(active_goal_ids),
        "scheduled_goal_ids": scheduled_goal_ids,
        "critical_goal_ids": sorted(critical_goal_ids),
        "open_goal_ids": sorted(open_goal_ids),
    }
