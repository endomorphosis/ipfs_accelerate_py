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
LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE = "launch Playwright validation gate"
LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND = (
    "(test ! -f swissknife/package.json || npm --prefix swissknife run test:e2e:meta-glasses) && "
    "(test ! -f hallucinate_app/package.json || "
    "npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts)"
)
LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS = (
    LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE.lower(),
    "playwright launch replay",
    "test:e2e:meta-glasses",
    "meta-glasses-virtual-os.spec.ts",
    "multimodal-control-surface.spec.ts",
)
DEFAULT_MISSION_TERMS = (
    "ai virtual desktop",
    "bluetooth",
    "camera",
    "captouch",
    "control plane",
    "desktop peer",
    "desktop offload",
    "end-to-end validation",
    "hallucinate app",
    "headphones",
    "ipfs",
    "launch playwright validation gate",
    "launch readiness",
    "launch replay",
    "libp2p",
    "mcp server",
    "mcp++",
    "meta glasses",
    "meta wearables dat",
    "microphone",
    "mobile phone",
    "neural band",
    "offload",
    "peer offload",
    "phone",
    "playwright",
    "production",
    "swissknife applications",
    "swissknife",
    "virtual desktop",
    "wifi",
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
WORKTREE_CLEANUP_BACKLOG_MARKERS = (
    "backlogged worktree",
    "backlogged worktree merge",
    "dirty backlogged worktrees",
    "preflight-conflicting backlogged worktree",
    "unsupported_status",
    "worktree reconciliation",
)
GUARDRAIL_REPAIR_MARKERS = (
    "dependency guardrail",
    "generated dirty",
    "reconciliation guardrail",
    "retry budget",
    "retry-budget",
)
DYNAMIC_GOAL_REGISTRATION_VALUES = {"active", "dynamic", "registered"}
COMPLETED_TASK_STATUSES = {"complete", "completed", "done", "succeeded"}


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


def _goal_requires_launch_playwright_gate(goal: ObjectiveGoal) -> bool:
    haystack = _goal_haystack(goal)
    return any(marker in haystack for marker in LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS)


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


def _is_mission_critical_codebase_scan_task(
    task: PortalTask,
    task_text: str,
    mission_terms: Sequence[str],
) -> bool:
    priority = str(task.priority or "").strip().upper()
    if priority == "P0":
        return True
    return priority == "P1" and _matches_any_term(task_text, mission_terms)


def _is_worktree_cleanup_backlog_task(task: PortalTask) -> bool:
    haystack = _task_haystack(task)
    return any(marker in haystack for marker in WORKTREE_CLEANUP_BACKLOG_MARKERS)


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


def registered_goal_ids_from_bundle_index(payload: Mapping[str, Any]) -> list[str]:
    """Return active dynamic goal IDs declared by an authoritative bundle index."""

    bundles = payload.get("bundles")
    if not isinstance(bundles, Mapping):
        return []

    registered: list[str] = []
    for bundle_key, raw_bundle in bundles.items():
        if not isinstance(raw_bundle, Mapping):
            continue
        tasks = raw_bundle.get("tasks")
        if not isinstance(tasks, list):
            continue
        for raw_task in tasks:
            if not isinstance(raw_task, Mapping):
                continue
            registration = str(raw_task.get("goal_registration") or "").strip().lower()
            candidate_kind = str(raw_task.get("candidate_kind") or "").strip().lower()
            if registration not in DYNAMIC_GOAL_REGISTRATION_VALUES and candidate_kind != "codebase_scan":
                continue
            status = str(raw_task.get("status") or "todo").strip().lower().replace("-", "_")
            if status in COMPLETED_TASK_STATUSES:
                continue
            registered.append(str(bundle_key))
            for key in ("goal_id", "subgoal_id", "parent_goal_id"):
                value = str(raw_task.get(key) or "").strip()
                if value:
                    registered.append(value)
            for key in ("parent_goal_ids", "goal_packet_goal_ids"):
                value = raw_task.get(key)
                if isinstance(value, (list, tuple)):
                    registered.extend(str(item) for item in value)
                elif isinstance(value, str):
                    registered.extend(_split_terms(value))
    return _unique(registered)


def reconcile_objective_task_strategy(
    *,
    goals: Sequence[ObjectiveGoal],
    tasks: Sequence[PortalTask],
    strategy: Mapping[str, Any],
    now: str,
    mission_terms: Sequence[str] = DEFAULT_MISSION_TERMS,
    registered_goal_ids: Sequence[str] = (),
    max_blocked_tasks: int = 50,
    max_deprioritized_tasks: int = 50,
    max_reopened_goals: int = 12,
) -> dict[str, Any]:
    """Return a strategy update that keeps todo work aligned with active goals."""

    goals_by_id = {goal.goal_id: goal for goal in goals if goal.goal_id}
    tasks_by_id = {task.task_id: task for task in tasks if task.task_id}
    heap_active_goal_ids = {goal.goal_id for goal in goals if goal.status in ACTIVE_GOAL_STATUSES}
    dynamic_goal_ids = set(_unique(registered_goal_ids))
    active_goal_ids = heap_active_goal_ids | dynamic_goal_ids
    scheduled_goal_ids = [
        record.goal_id
        for record in objective_heap_schedule(goals)
        if record.goal_id in active_goal_ids
    ]
    critical_goal_ids = _critical_goal_ids(goals, mission_terms)
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

    unblock_receipts: list[ObjectiveTaskJanitorReceipt] = []
    remove_receipts: list[ObjectiveTaskJanitorReceipt] = []
    for task_id in previously_blocked:
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        goal_ids = _task_goal_ids(task)
        active_task_goal_ids = [goal_id for goal_id in goal_ids if goal_id in active_goal_ids]
        known_goal_ids = [goal_id for goal_id in goal_ids if goal_id in goals_by_id]
        if active_task_goal_ids:
            unblock_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="unblock",
                    retired_task_reason="referenced_goal_is_active",
                    goal_ids=goal_ids,
                    title=task.title,
                    priority=task.priority,
                    track=task.track,
                )
            )
            continue
        if known_goal_ids and all(goals_by_id[goal_id].status == "completed" for goal_id in known_goal_ids):
            remove_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="remove",
                    retired_task_reason="referenced_goal_completed",
                    goal_ids=goal_ids,
                    title=task.title,
                    priority=task.priority,
                    track=task.track,
                )
            )

    unblocked_task_ids = [receipt.task_id for receipt in unblock_receipts]
    removed_task_ids = [receipt.task_id for receipt in remove_receipts]
    janitor_unblocked = set(unblocked_task_ids)
    strategy_blocked = set(existing_blocked)
    open_tasks = [
        task
        for task in tasks
        if task.status in OPEN_TASK_STATUSES
        and (task.task_id not in strategy_blocked or task.task_id in janitor_unblocked)
    ]
    open_tasks.extend(
        task
        for task_id in unblocked_task_ids
        for task in [tasks_by_id.get(task_id)]
        if task is not None and task.status not in OPEN_TASK_STATUSES
    )
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
        codebase_scan_task = _is_codebase_scan_backlog_task(task)
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
            codebase_scan_task
            and not _is_mission_critical_codebase_scan_task(task, task_text, mission_terms)
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
            continue

        if _is_worktree_cleanup_backlog_task(task) and not mission_aligned:
            deprioritize_receipts.append(
                ObjectiveTaskJanitorReceipt(
                    task_id=task.task_id,
                    action="deprioritize",
                    retired_task_reason="off_mission_worktree_cleanup_task",
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
    validation_gate_goal_ids = [
        goal_id
        for goal_id in scheduled_goal_ids
        if goal_id in critical_goal_ids
        and goal_id in goals_by_id
        and _goal_requires_launch_playwright_gate(goals_by_id[goal_id])
    ]

    next_blocked = _unique(
        [
            task_id
            for task_id in existing_blocked
            if task_id not in previously_blocked and task_id not in removed_task_ids and task_id not in unblocked_task_ids
        ]
        + blocked_task_ids
    )
    next_deprioritized = _unique(
        [task_id for task_id in existing_deprioritized if task_id not in previously_deprioritized]
        + deprioritized_task_ids
    )

    receipt_payloads = []
    for receipt in [*unblock_receipts, *remove_receipts, *block_receipts, *deprioritize_receipts]:
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
    updated_strategy["objective_task_janitor_validation_gate_goal_ids"] = validation_gate_goal_ids
    updated_strategy["objective_task_janitor_launch_playwright_validation_gate"] = {
        "evidence_term": LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE,
        "goal_ids": validation_gate_goal_ids,
        "validation_command": LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND,
        "active": bool(validation_gate_goal_ids),
    }
    updated_strategy["objective_task_janitor_critical_goal_ids"] = sorted(critical_goal_ids)
    updated_strategy["objective_task_janitor_mission_terms"] = list(mission_terms)
    updated_strategy["objective_task_janitor_active_goal_ids"] = sorted(active_goal_ids)
    updated_strategy["objective_task_janitor_heap_active_goal_ids"] = sorted(heap_active_goal_ids)
    updated_strategy["objective_task_janitor_registered_goal_ids"] = sorted(dynamic_goal_ids)
    updated_strategy["objective_task_janitor_heap_schedule_goal_ids"] = scheduled_goal_ids
    updated_strategy["objective_task_janitor_last_run_at"] = now
    updated_strategy["objective_task_janitor_last_run_summary"] = {
        "blocked_count": len(blocked_task_ids),
        "deprioritized_count": len(deprioritized_task_ids),
        "unblocked_count": len(unblocked_task_ids),
        "removed_count": len(removed_task_ids),
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
        "objective_task_janitor_validation_gate_goal_ids",
        "objective_task_janitor_launch_playwright_validation_gate",
        "objective_task_janitor_mission_terms",
        "objective_task_janitor_active_goal_ids",
        "objective_task_janitor_heap_active_goal_ids",
        "objective_task_janitor_registered_goal_ids",
    )
    changed = any(strategy.get(key) != updated_strategy.get(key) for key in comparable_keys)
    return {
        "changed": changed,
        "strategy": updated_strategy,
        "blocked_task_ids": blocked_task_ids,
        "deprioritized_task_ids": deprioritized_task_ids,
        "unblocked_task_ids": unblocked_task_ids,
        "removed_task_ids": removed_task_ids,
        "reopened_goal_ids": reopened_goal_ids,
        "receipts": receipt_payloads,
        "active_goal_ids": sorted(active_goal_ids),
        "heap_active_goal_ids": sorted(heap_active_goal_ids),
        "registered_goal_ids": sorted(dynamic_goal_ids),
        "scheduled_goal_ids": scheduled_goal_ids,
        "critical_goal_ids": sorted(critical_goal_ids),
        "open_goal_ids": sorted(open_goal_ids),
        "validation_gate_goal_ids": validation_gate_goal_ids,
    }
