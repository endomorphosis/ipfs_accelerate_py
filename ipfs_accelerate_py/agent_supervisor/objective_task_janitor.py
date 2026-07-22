"""Reconcile supervisor todo strategy with the objective heap."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .objective_graph import ObjectiveGoal, objective_heap_schedule
from .todo_daemon.implementation_daemon import PortalTask

# Compatibility export; lifecycle decisions below use ObjectiveGoal's
# canonical state helpers so reopened and verified_complete are not lost.
ACTIVE_GOAL_STATUSES = {"active", "todo", "open", "reopened"}
OPEN_TASK_STATUSES = {"todo", "ready", "in_progress"}
JANITOR_RECEIPT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.objective_task_janitor.v1"
JANITOR_COMPLETION_GATE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_task_janitor.completion_gate.v1"
)
JANITOR_GOAL_REOPEN_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_task_janitor.goal_reopening.v1"
)
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
JANITOR_BLOCKED_REASON_MARKER = "objective-task janitor"


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
        if not goal.is_schedulable:
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


def _is_materialized_janitor_block(task: PortalTask) -> bool:
    return (
        task.status == "blocked"
        and JANITOR_BLOCKED_REASON_MARKER in task.metadata.get("blocked reason", "").lower()
    )


def _goal_descendants(goals: Sequence[ObjectiveGoal]) -> dict[str, list[str]]:
    """Return every descendant for each goal, tolerating malformed cycles."""

    children: dict[str, list[str]] = {goal.goal_id: [] for goal in goals if goal.goal_id}
    for goal in goals:
        for parent_id in goal.parent_goal_ids:
            if parent_id in children and goal.goal_id not in children[parent_id]:
                children[parent_id].append(goal.goal_id)

    descendants: dict[str, list[str]] = {}
    for goal_id in children:
        pending = list(children[goal_id])
        seen = {goal_id}
        result: list[str] = []
        while pending:
            child_id = pending.pop(0)
            if child_id in seen:
                continue
            seen.add(child_id)
            result.append(child_id)
            pending.extend(children.get(child_id, ()))
        descendants[goal_id] = result
    return descendants


def _json_safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    serializer = getattr(value, "to_dict", None)
    if callable(serializer):
        payload = serializer()
        if isinstance(payload, Mapping):
            return {str(key): item for key, item in payload.items()}
    raise TypeError("contradictions must be mappings or expose to_dict()")


def _stable_contradiction_id(payload: Mapping[str, Any]) -> str:
    """Return the source identity used to make reopening replay-safe."""

    for key in ("contradiction_id", "fingerprint", "receipt_cid", "source_receipt_cid"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    source = payload.get("source_receipt")
    if isinstance(source, Mapping):
        for key in ("receipt_cid", "cid", "fingerprint", "id"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _payload_string_list(payload: Mapping[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            values.extend(_split_terms(value))
        elif isinstance(value, (list, tuple, set, frozenset)):
            values.extend(str(item) for item in value if str(item).strip())
    return _unique(values)


def _contradiction_goal_ids(payload: Mapping[str, Any]) -> list[str]:
    return _payload_string_list(
        payload,
        "goal_id",
        "goal_ids",
        "mapped_goal_id",
        "mapped_goal_ids",
        "affected_goal_id",
        "affected_goal_ids",
    )


def _contradiction_scheduled_task_ids(payload: Mapping[str, Any]) -> list[str]:
    values = _payload_string_list(
        payload,
        "newly_scheduled_task_ids",
        "scheduled_task_ids",
        "work_task_ids",
        "task_ids",
    )
    raw_work = payload.get("newly_scheduled_work") or payload.get("scheduled_work")
    if isinstance(raw_work, list):
        for item in raw_work:
            if isinstance(item, Mapping):
                task_id = str(item.get("task_id") or item.get("id") or "").strip()
                if task_id:
                    values.append(task_id)
            elif str(item).strip():
                values.append(str(item).strip())
    return _unique(values)


def _completion_gate_receipt(
    goal: ObjectiveGoal,
    *,
    decision: Any,
    descendants: Sequence[str],
    goals_by_id: Mapping[str, ObjectiveGoal],
    completion_decisions: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a persisted completion decision instead of trusting status.

    The decision and gate shapes are intentionally checked at this boundary.
    A claimed top-level ``passed`` flag cannot conceal a failed check, missing
    evidence snapshot, or a descendant which has since become inconclusive or
    reopened.
    """

    payload = dict(decision) if isinstance(decision, Mapping) else {}
    raw_gate = payload.get("completion_gate")
    gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else {}
    raw_checks = gate.get("checks")
    checks = [dict(item) for item in raw_checks if isinstance(item, Mapping)] if isinstance(raw_checks, list) else []
    evaluated_evidence = gate.get("evaluated_evidence")
    evidence = dict(evaluated_evidence) if isinstance(evaluated_evidence, Mapping) else {}
    reasons: list[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not payload:
        reject("completion_gate_missing")
    else:
        for reason_code in payload.get("reason_codes", ()):
            if str(reason_code):
                reject(str(reason_code))
        state = str(payload.get("state") or payload.get("next_state") or "").strip().lower()
        if payload.get("verified") is not True or state != "verified_complete":
            reject("completion_decision_not_verified")
        if not gate:
            reject("completion_gate_missing")
        else:
            gate_reason_codes = [str(item) for item in gate.get("reason_codes", ()) if str(item)]
            for reason_code in gate_reason_codes:
                reject(reason_code)
            if gate.get("passed") is not True:
                reject("completion_gate_failed")
            if not checks:
                reject("completion_gate_checks_missing")
            else:
                failed_checks = [check for check in checks if check.get("passed") is not True]
                for check in failed_checks:
                    reason_code = str(check.get("reason_code") or "").strip()
                    if reason_code:
                        reject(reason_code)
                if failed_checks:
                    reject("completion_gate_check_failed")
            if not evidence:
                reject("completion_gate_evidence_missing")

    descendant_receipts: list[dict[str, Any]] = []
    for child_id in descendants:
        child = goals_by_id.get(child_id)
        if child is None:
            continue
        child_state = child.lifecycle_state_value
        child_decision = completion_decisions.get(child_id)
        child_payload = dict(child_decision) if isinstance(child_decision, Mapping) else {}
        child_gate = child_payload.get("completion_gate")
        child_gate_payload = dict(child_gate) if isinstance(child_gate, Mapping) else {}
        child_checks = child_gate_payload.get("checks")
        child_checks_valid = bool(
            isinstance(child_checks, list)
            and child_checks
            and all(isinstance(check, Mapping) and check.get("passed") is True for check in child_checks)
        )
        child_evidence = child_gate_payload.get("evaluated_evidence")
        # A heap may contain verified descendants written before completion
        # decision artifacts existed. Preserve that compatibility only when
        # no child decision is supplied; a supplied decision is authoritative
        # and must itself contain a passing gate.
        child_passed = bool(
            child_state == "verified_complete"
            and (
                not child_payload
                or (
                    child_payload.get("verified") is True
                    and child_gate_payload.get("passed") is True
                    and child_checks_valid
                    and isinstance(child_evidence, Mapping)
                    and bool(child_evidence)
                )
            )
        )
        child_receipt = {
            "goal_id": child_id,
            "state": child_state,
            "passed": child_passed,
            "reason_codes": list(child_gate_payload.get("reason_codes") or ()),
            "evaluated_evidence": child_evidence if isinstance(child_evidence, Mapping) else {},
        }
        descendant_receipts.append(child_receipt)
        if not child_passed:
            reject(
                "descendant_reopened"
                if child_state == "reopened"
                else (
                    "descendant_analysis_inconclusive"
                    if child_state == "analysis_inconclusive"
                    else "descendant_completion_unverified"
                )
            )

    return {
        "schema": JANITOR_COMPLETION_GATE_RECEIPT_SCHEMA,
        "goal_id": goal.goal_id,
        "passed": not reasons,
        "reason_codes": reasons,
        "state": goal.lifecycle_state_value,
        "decision": payload,
        "evaluated_evidence": evidence,
        "descendants": descendant_receipts,
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
    completion_decisions: Mapping[str, Any] | None = None,
    contradictions: Sequence[Any] = (),
    max_blocked_tasks: int = 50,
    max_deprioritized_tasks: int = 50,
    max_reopened_goals: int = 12,
) -> dict[str, Any]:
    """Return a strategy update that keeps todo work aligned with active goals."""

    goals_by_id = {goal.goal_id: goal for goal in goals if goal.goal_id}
    supplied_completion_decisions = completion_decisions
    if supplied_completion_decisions is None:
        strategy_decisions = strategy.get("objective_completion_decisions")
        supplied_completion_decisions = (
            strategy_decisions if isinstance(strategy_decisions, Mapping) else {}
        )

    prior_reopening_receipts = [
        dict(receipt)
        for receipt in strategy.get("objective_task_janitor_goal_reopening_receipts", ())
        if isinstance(receipt, Mapping)
    ] if isinstance(
        strategy.get("objective_task_janitor_goal_reopening_receipts"), list
    ) else []
    contradiction_payloads = [_json_safe_mapping(item) for item in contradictions]
    goal_reopening_decisions: dict[str, Any] = {}
    if contradiction_payloads:
        # The lifecycle module owns contradiction classification, relevance,
        # propagation, and receipt idempotency.  The janitor consumes those
        # decisions to update scheduling strategy without maintaining a
        # second, subtly different reopening state machine.
        from .goal_completion import reconcile_goal_reopenings

        goal_records = []
        for goal in goals:
            dependency_ids: list[str] = []
            for key in ("depends_on", "dependencies", "dependency_goal_ids", "prerequisites"):
                dependency_ids.extend(_split_terms(goal.fields.get(key, "")))
            goal_records.append(
                {
                    "goal_id": goal.goal_id,
                    "state": goal.lifecycle_state_value,
                    "parent_goal_ids": goal.parent_goal_ids,
                    "depends_on": _unique(dependency_ids),
                    "acceptance_criteria": _split_terms(
                        goal.fields.get("acceptance", goal.fields.get("acceptance_criteria", ""))
                    ),
                }
            )

        historical_completion_receipts: list[dict[str, Any]] = []
        for goal_id, decision in supplied_completion_decisions.items():
            if isinstance(decision, Mapping):
                historical_receipt = dict(decision)
                historical_receipt.setdefault("goal_id", str(goal_id))
                historical_completion_receipts.append(historical_receipt)
        raw_history = strategy.get("objective_goal_completion_receipts", ())
        if isinstance(raw_history, list):
            historical_completion_receipts.extend(
                dict(receipt) for receipt in raw_history if isinstance(receipt, Mapping)
            )

        goal_reopening_decisions = reconcile_goal_reopenings(
            goal_records,
            contradictions,
            historical_completion_receipts=historical_completion_receipts,
            existing_reopen_receipts=prior_reopening_receipts,
            now=now,
        )

    serialized_reopening_decisions = {
        goal_id: (
            decision.to_dict()
            if callable(getattr(decision, "to_dict", None))
            else dict(decision)
        )
        for goal_id, decision in sorted(goal_reopening_decisions.items())
    }
    prior_reopening_decisions = strategy.get(
        "objective_task_janitor_goal_reopening_decisions", {}
    )
    persisted_reopening_decisions = (
        {
            str(goal_id): dict(payload)
            for goal_id, payload in prior_reopening_decisions.items()
            if isinstance(payload, Mapping)
        }
        if isinstance(prior_reopening_decisions, Mapping)
        else {}
    )
    for goal_id, payload in serialized_reopening_decisions.items():
        if payload.get("idempotent") is not True or goal_id not in persisted_reopening_decisions:
            persisted_reopening_decisions[goal_id] = payload
    contradiction_reopened_goal_ids = [
        goal_id
        for goal_id, payload in serialized_reopening_decisions.items()
        if payload.get("reopened") is True and payload.get("idempotent") is not True
    ]
    raw_pending_reopen_goal_ids = strategy.get(
        "objective_task_janitor_pending_reopen_goal_ids"
    )
    if isinstance(raw_pending_reopen_goal_ids, list):
        prior_pending_reopen_goal_ids = _unique(
            [str(goal_id) for goal_id in raw_pending_reopen_goal_ids]
        )
    else:
        # Compatibility for receipts written before pending materialization
        # was tracked separately from the append-only audit ledger.
        prior_pending_reopen_goal_ids = _unique(
            [str(receipt.get("goal_id") or "") for receipt in prior_reopening_receipts]
        )
    effective_contradiction_reopened_goal_ids = _unique(
        [
            goal_id
            for goal_id, payload in serialized_reopening_decisions.items()
            if str(payload.get("state") or "").strip().lower() == "reopened"
        ]
        + [
            goal_id
            for goal_id in prior_pending_reopen_goal_ids
            if goal_id in goals_by_id
            and goals_by_id[goal_id].lifecycle_state_value
            in {"provisionally_complete", "verified_complete"}
        ]
    )
    pending_reopen_goal_ids = _unique(
        [*prior_pending_reopen_goal_ids, *contradiction_reopened_goal_ids]
    )
    pending_reopen_goal_ids = [
        goal_id
        for goal_id in pending_reopen_goal_ids
        if goal_id in goals_by_id
        and goals_by_id[goal_id].lifecycle_state_value
        in {"provisionally_complete", "verified_complete"}
    ]
    recalculated_goal_ids = sorted(
        goal_id
        for goal_id, payload in serialized_reopening_decisions.items()
        if payload.get("contradiction_ids") or payload.get("reason_codes")
    )
    new_reopening_receipts = [
        dict(payload["reopening_receipt"])
        for payload in serialized_reopening_decisions.values()
        if payload.get("reopened") is True
        and payload.get("idempotent") is not True
        and isinstance(payload.get("reopening_receipt"), Mapping)
        and payload.get("reopening_receipt")
    ]
    def reopening_receipt_key(receipt: Mapping[str, Any]) -> str:
        receipt_id = str(receipt.get("receipt_id") or "").strip()
        if receipt_id:
            return "receipt:" + receipt_id
        contradiction_ids = receipt.get("contradiction_ids", ())
        if isinstance(contradiction_ids, str):
            contradiction_ids = [contradiction_ids]
        identity = {
            "goal_id": str(receipt.get("goal_id") or ""),
            "contradiction_ids": sorted(str(item) for item in contradiction_ids or ()),
        }
        return "fallback:" + _stable_contradiction_id(identity)

    seen_reopening_receipts = {
        reopening_receipt_key(receipt) for receipt in prior_reopening_receipts
    }
    persisted_reopening_receipts = list(prior_reopening_receipts)
    for receipt in new_reopening_receipts:
        receipt.setdefault("schema", JANITOR_GOAL_REOPEN_RECEIPT_SCHEMA)
        key = reopening_receipt_key(receipt)
        if key not in seen_reopening_receipts:
            persisted_reopening_receipts.append(receipt)
            seen_reopening_receipts.add(key)

    newly_scheduled_task_ids = _unique(
        [
            task_id
            for goal_id in contradiction_reopened_goal_ids
            for task_id in _contradiction_scheduled_task_ids(
                serialized_reopening_decisions.get(goal_id, {})
            )
        ]
        + [
            task_id
            for payload in contradiction_payloads
            if set(_contradiction_goal_ids(payload)) & set(contradiction_reopened_goal_ids)
            for task_id in _contradiction_scheduled_task_ids(payload)
        ]
    )
    raw_persisted_scheduled_tasks = strategy.get(
        "objective_task_janitor_newly_scheduled_task_ids", ()
    )
    persisted_scheduled_task_ids = _unique(
        (
            [str(task_id) for task_id in raw_persisted_scheduled_tasks]
            if isinstance(raw_persisted_scheduled_tasks, (list, tuple))
            else []
        )
        + newly_scheduled_task_ids
    )
    descendants_by_goal = _goal_descendants(goals)
    completion_gate_receipts = {
        goal_id: _completion_gate_receipt(
            goal,
            decision=supplied_completion_decisions.get(goal_id),
            descendants=descendants_by_goal.get(goal_id, ()),
            goals_by_id=goals_by_id,
            completion_decisions=supplied_completion_decisions,
        )
        for goal_id, goal in goals_by_id.items()
        if goal.lifecycle_state_value == "verified_complete"
        and goal_id in supplied_completion_decisions
    }

    def completion_proven(goal_id: str) -> bool:
        receipt = completion_gate_receipts.get(goal_id)
        # Legacy objective heaps have no persisted decision artifact. Their
        # established verified status remains compatible. Once a decision is
        # supplied, however, it is authoritative and evaluated fail closed.
        return receipt is None or receipt.get("passed") is True

    def completion_failure_reason(goal_ids: Sequence[str]) -> str:
        reason_codes = [
            str(code)
            for goal_id in goal_ids
            for code in completion_gate_receipts.get(goal_id, {}).get("reason_codes", ())
            if str(code)
        ]
        return reason_codes[0] if reason_codes else "completion_gate_missing"

    tasks_by_id = {task.task_id: task for task in tasks if task.task_id}
    contradiction_reopened_set = set(effective_contradiction_reopened_goal_ids)
    effective_goals = [
        ObjectiveGoal(
            goal.goal_id,
            goal.title,
            {**goal.fields, "status": "reopened"},
        )
        if goal.goal_id in contradiction_reopened_set
        else goal
        for goal in goals
    ]
    heap_active_goal_ids = {goal.goal_id for goal in effective_goals if goal.is_schedulable}
    dynamic_goal_ids = set(_unique(registered_goal_ids))
    active_goal_ids = heap_active_goal_ids | dynamic_goal_ids
    scheduled_goal_ids = [
        record.goal_id
        for record in objective_heap_schedule(effective_goals)
        if record.goal_id in active_goal_ids
    ]
    critical_goal_ids = _critical_goal_ids(effective_goals, mission_terms)
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
    retained_block_receipts: list[ObjectiveTaskJanitorReceipt] = []
    materialized_janitor_blocks = {
        task.task_id for task in tasks if task.task_id and _is_materialized_janitor_block(task)
    }
    for task_id in sorted(previously_blocked | materialized_janitor_blocks):
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
        if known_goal_ids and all(
            goals_by_id[goal_id].lifecycle_state_value == "verified_complete"
            and completion_proven(goal_id)
            for goal_id in known_goal_ids
        ):
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
            continue
        reason = "goal_not_active"
        if known_goal_ids and all(
            goals_by_id[goal_id].lifecycle_state_value == "verified_complete"
            for goal_id in known_goal_ids
        ):
            reason = completion_failure_reason(known_goal_ids)
        missing_goal_ids = [goal_id for goal_id in goal_ids if goal_id not in goals_by_id]
        if missing_goal_ids and len(missing_goal_ids) == len(goal_ids):
            reason = "orphaned_goal_reference"
        retained_block_receipts.append(
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
    block_receipts: list[ObjectiveTaskJanitorReceipt] = list(retained_block_receipts)
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
            if goal_known and all(
                goals_by_id[goal_id].lifecycle_state_value == "verified_complete"
                and completion_proven(goal_id)
                for goal_id in goal_known
            ):
                reason = "goal_completed"
            elif goal_known and all(
                goals_by_id[goal_id].lifecycle_state_value == "verified_complete"
                for goal_id in goal_known
            ):
                reason = completion_failure_reason(goal_known)
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
    missing_work_goal_ids = [
        goal_id
        for goal_id in scheduled_goal_ids
        if goal_id in critical_goal_ids and goal_id not in open_goal_ids
    ][: max(0, int(max_reopened_goals))]
    reopened_goal_ids = _unique(
        [*effective_contradiction_reopened_goal_ids, *missing_work_goal_ids]
    )
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
    updated_strategy["objective_task_janitor_completion_gate_receipts"] = [
        completion_gate_receipts[goal_id] for goal_id in sorted(completion_gate_receipts)
    ]
    updated_strategy["objective_task_janitor_completion_gate_passed_goal_ids"] = sorted(
        goal_id for goal_id, receipt in completion_gate_receipts.items() if receipt["passed"]
    )
    updated_strategy["objective_task_janitor_completion_gate_failed_goal_ids"] = sorted(
        goal_id for goal_id, receipt in completion_gate_receipts.items() if not receipt["passed"]
    )
    updated_strategy["objective_task_janitor_goal_reopening_decisions"] = (
        persisted_reopening_decisions
    )
    updated_strategy["objective_task_janitor_goal_reopening_receipts"] = (
        persisted_reopening_receipts
    )
    updated_strategy["objective_task_janitor_contradiction_reopened_goal_ids"] = (
        effective_contradiction_reopened_goal_ids
    )
    updated_strategy["objective_task_janitor_effective_reopened_goal_ids"] = (
        effective_contradiction_reopened_goal_ids
    )
    updated_strategy["objective_task_janitor_pending_reopen_goal_ids"] = (
        pending_reopen_goal_ids
    )
    updated_strategy["objective_task_janitor_recalculated_goal_ids"] = recalculated_goal_ids
    updated_strategy["objective_task_janitor_newly_scheduled_task_ids"] = (
        persisted_scheduled_task_ids
    )
    updated_strategy["objective_task_janitor_last_run_at"] = now
    updated_strategy["objective_task_janitor_last_run_summary"] = {
        "blocked_count": len(blocked_task_ids),
        "deprioritized_count": len(deprioritized_task_ids),
        "unblocked_count": len(unblocked_task_ids),
        "removed_count": len(removed_task_ids),
        "reopened_goal_count": len(reopened_goal_ids),
        "active_goal_count": len(active_goal_ids),
        "scheduled_goal_count": len(scheduled_goal_ids),
        "completion_gate_passed_count": sum(
            1 for receipt in completion_gate_receipts.values() if receipt["passed"]
        ),
        "completion_gate_failed_count": sum(
            1 for receipt in completion_gate_receipts.values() if not receipt["passed"]
        ),
        "contradiction_reopened_goal_count": len(contradiction_reopened_goal_ids),
        "recalculated_goal_count": len(recalculated_goal_ids),
        "newly_scheduled_task_count": len(newly_scheduled_task_ids),
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
        "objective_task_janitor_completion_gate_receipts",
        "objective_task_janitor_completion_gate_passed_goal_ids",
        "objective_task_janitor_completion_gate_failed_goal_ids",
        "objective_task_janitor_goal_reopening_decisions",
        "objective_task_janitor_goal_reopening_receipts",
        "objective_task_janitor_contradiction_reopened_goal_ids",
        "objective_task_janitor_effective_reopened_goal_ids",
        "objective_task_janitor_pending_reopen_goal_ids",
        "objective_task_janitor_recalculated_goal_ids",
        "objective_task_janitor_newly_scheduled_task_ids",
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
        "contradiction_reopened_goal_ids": contradiction_reopened_goal_ids,
        "effective_reopened_goal_ids": effective_contradiction_reopened_goal_ids,
        "recalculated_goal_ids": recalculated_goal_ids,
        "newly_scheduled_task_ids": newly_scheduled_task_ids,
        "goal_reopening_decisions": serialized_reopening_decisions,
        "goal_reopening_receipts": persisted_reopening_receipts,
        "receipts": receipt_payloads,
        "active_goal_ids": sorted(active_goal_ids),
        "heap_active_goal_ids": sorted(heap_active_goal_ids),
        "registered_goal_ids": sorted(dynamic_goal_ids),
        "scheduled_goal_ids": scheduled_goal_ids,
        "critical_goal_ids": sorted(critical_goal_ids),
        "open_goal_ids": sorted(open_goal_ids),
        "validation_gate_goal_ids": validation_gate_goal_ids,
        "completion_gate_receipts": updated_strategy[
            "objective_task_janitor_completion_gate_receipts"
        ],
        "completion_gate_passed_goal_ids": updated_strategy[
            "objective_task_janitor_completion_gate_passed_goal_ids"
        ],
        "completion_gate_failed_goal_ids": updated_strategy[
            "objective_task_janitor_completion_gate_failed_goal_ids"
        ],
    }
