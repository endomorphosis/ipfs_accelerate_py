"""Authoritative, event-derived scheduler state and throughput metrics.

The supervisor has several durable state stores, but its JSONL lifecycle events
are the common interchange format between them.  This module reduces those
events into one immutable snapshot used by both schedulers and operators.

The reducer deliberately accepts legacy event shapes.  Unknown events are
retained in the source count but otherwise ignored, missing dimensions receive
stable sentinel values, and incomplete timing pairs never produce negative or
invented durations.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .event_log import read_jsonl_event_sources


SCHEDULER_SNAPSHOT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.scheduler-snapshot@1"
SCHEDULER_PHASES = (
    "ready",
    "active",
    "idle",
    "blocked",
    "validation",
    "merge",
    "resolver",
)
UNKNOWN_IDENTITY = "unknown"

# Keep these strings local to the event reducer instead of requiring event
# readers to deserialize the typed receipt.  Event logs are a compatibility
# boundary and may contain receipts produced by newer supervisor versions.
REFILL_SCAN_TERMINAL_REASONS = (
    "generated",
    "exhausted",
    "duplicate_only",
    "threshold_satisfied",
    "cooldown",
    "disabled",
    "partial",
    "failed",
    "timed_out",
)
REFILL_SCAN_SKIPPED_REASONS = frozenset(
    {"threshold_satisfied", "cooldown", "disabled"}
)
REFILL_SCAN_FAILED_REASONS = frozenset({"failed", "timed_out"})
REFILL_SCAN_SUCCESS_REASONS = frozenset(
    {"generated", "exhausted", "duplicate_only"}
)

_REFILL_SCAN_EVENT_TYPES = frozenset(
    {
        "refill_scan_receipt",
        "scan_receipt",
        "objective_refill_receipt",
        "codebase_refill_receipt",
        # Historical result-bearing events are retained during migration.
        "objective_refill_scan",
        "codebase_refill_scan",
        "objective_refill_failed",
        "codebase_refill_failed",
        "objective_refill_timeout",
        "codebase_refill_timeout",
    }
)

_READY_EVENTS = frozenset(
    {
        "queued", "task_queued", "task_registered", "task_discovered",
        "task_ready", "ready", "lease_released", "lease_expired",
    }
)
_ACTIVE_EVENTS = frozenset(
    {
        "task_selected", "implementation_started", "implementation_resumed",
        "implementing", "worker_started", "lane_started",
    }
)
_IDLE_EVENTS = frozenset(
    {
        "idle", "lane_idle", "daemon_no_tasks", "worker_idle",
        "task_completed", "task_succeeded", "completed",
    }
)
_BLOCKED_EVENTS = frozenset(
    {
        "blocked", "task_blocked", "lane_blocked", "task_quarantined",
        "merge_quarantined", "dependency_blocked",
    }
)
_VALIDATION_START_EVENTS = frozenset(
    {"validation_started", "validating", "validation_stage_started"}
)
_VALIDATION_END_EVENTS = frozenset(
    {"validation_finished", "validation_completed", "validation_stage_finished"}
)
_MERGE_QUEUE_EVENTS = frozenset(
    {"merge_candidate_enqueued", "merge_enqueued", "merge_queued", "merge_queue"}
)
_MERGE_START_EVENTS = frozenset(
    {"merge_started", "merge_reconciliation_started", "merging"}
)
_MERGE_END_EVENTS = frozenset(
    {"merge_finished", "merge_reconciled", "merge_completed"}
)
_RESOLVER_EVENTS = frozenset(
    {
        "llm_merge_resolver_invoked", "merge_resolver_started",
        "resolver_started", "resolving",
    }
)
_COMPLETION_EVENTS = frozenset(
    {"task_completed", "task_succeeded", "completed", "merge_completed"}
)


def _now_iso(now: datetime | str | None = None) -> str:
    if isinstance(now, str):
        parsed = _parse_timestamp(now)
        return (parsed or datetime.now(timezone.utc)).isoformat()
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_time(event: Mapping[str, Any]) -> datetime | None:
    for key in ("timestamp", "occurred_at", "event_at", "created_at", "updated_at"):
        parsed = _parse_timestamp(event.get(key))
        if parsed is not None:
            return parsed
    for key in ("timestamp_ms", "occurred_at_ms", "created_at_ms", "updated_at_ms", "registered_at_ms"):
        value = event.get(key)
        if value not in (None, ""):
            try:
                return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
            except (OSError, OverflowError, TypeError, ValueError):
                pass
    return None


def _seconds(start: datetime | None, finish: datetime | None) -> float:
    if start is None or finish is None or finish < start:
        return 0.0
    return (finish - start).total_seconds()


def _first_text(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def normalize_metric_identity(
    event: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return the five mandatory canonical metric dimensions.

    Profile-G CIDs are preferred.  Display identifiers remain useful aliases,
    but they never replace an available canonical identifier.
    """

    raw_event = dict(event or {})
    raw_defaults = dict(defaults or {})
    event_identity = raw_event.get("identity")
    default_identity = raw_defaults.get("identity")
    sources = [
        dict(event_identity) if isinstance(event_identity, Mapping) else {},
        raw_event,
        dict(default_identity) if isinstance(default_identity, Mapping) else {},
        raw_defaults,
    ]
    goal = _first_text(
        sources,
        ("goal_cid", "canonical_goal_cid", "canonical_goal_id", "goal_id", "goal"),
    ) or UNKNOWN_IDENTITY
    subgoal = _first_text(
        sources,
        ("subgoal_cid", "canonical_subgoal_cid", "canonical_subgoal_id", "subgoal_id", "subgoal"),
    ) or UNKNOWN_IDENTITY
    task = _first_text(
        sources,
        (
            "task_cid", "canonical_task_cid", "canonical_task_id",
            "canonical_task_key", "task_id", "task",
        ),
    ) or UNKNOWN_IDENTITY
    lane = _first_text(
        sources,
        ("lane_id", "canonical_lane_id", "parallel_lane", "bundle_key", "state_prefix", "lane"),
    ) or UNKNOWN_IDENTITY
    provider = _first_text(
        sources,
        (
            "provider_id", "canonical_provider_id", "effective_provider_name",
            "provider_identity", "provider", "claimant_did", "worker_id",
        ),
    ) or UNKNOWN_IDENTITY
    return {
        "goal_cid": goal,
        "subgoal_cid": subgoal,
        "task_cid": task,
        "canonical_task_cid": task,
        "lane_id": lane,
        "provider_id": provider,
        # Explicit aliases make the canonical nature discoverable to API
        # clients without forcing older clients to rename their dimensions.
        "canonical_goal_id": goal,
        "canonical_subgoal_id": subgoal,
        "canonical_task_id": task,
        "canonical_lane_id": lane,
        "canonical_provider_id": provider,
    }


def _identity_key(identity: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        identity["goal_cid"],
        identity["subgoal_cid"],
        identity["task_cid"],
        identity["lane_id"],
        identity["provider_id"],
    )


def _metric_defaults(identity: Mapping[str, str]) -> dict[str, Any]:
    return {
        **dict(identity),
        "queue_wait_seconds": 0.0,
        "implementation_duration_seconds": 0.0,
        "validation_duration_seconds": 0.0,
        "merge_wait_seconds": 0.0,
        "implementation_attempts": 0,
        "merge_attempts": 0,
        "conflicts": 0,
        "retries": 0,
        "completions": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "conflict_rate": 0.0,
        "retry_rate": 0.0,
    }


def _number(value: Any) -> float:
    if isinstance(value, bool) or value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _usage(event: Mapping[str, Any]) -> tuple[int, float]:
    usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
    total = _number(event.get("total_tokens")) or _number(usage.get("total_tokens"))
    if not total:
        direct = _number(event.get("tokens")) or _number(event.get("token_count"))
        if direct:
            total = direct
        else:
            prompt = (
                _number(event.get("input_tokens"))
                or _number(event.get("prompt_tokens"))
                or _number(usage.get("input_tokens"))
                or _number(usage.get("prompt_tokens"))
            )
            completion = (
                _number(event.get("output_tokens"))
                or _number(event.get("completion_tokens"))
                or _number(usage.get("output_tokens"))
                or _number(usage.get("completion_tokens"))
            )
            total = prompt + completion
    cost = (
        _number(event.get("cost_usd"))
        or _number(event.get("estimated_cost_usd"))
        or _number(event.get("cost"))
        or _number(usage.get("cost_usd"))
        or _number(usage.get("cost"))
    )
    return max(0, int(total)), max(0.0, cost)


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type") or event.get("event_type") or event.get("event") or "").strip().lower()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _scan_receipt_projection(
    event: Mapping[str, Any], kind: str
) -> dict[str, Any] | None:
    """Extract one compact refill receipt projection from an event.

    The canonical event format puts the compact projection in ``scan_receipt``.
    ``receipt`` and ``scan_result`` are accepted for forward and migration
    compatibility.  Full generated items and per-file details are deliberately
    not copied into the scheduler snapshot.
    """

    nested: Mapping[str, Any] = {}
    for key in ("scan_receipt", "receipt", "scan_result", "receipt_projection"):
        candidate = event.get(key)
        if isinstance(candidate, Mapping):
            nested = candidate
            break

    raw_reason = nested.get("terminal_reason", nested.get("reason"))
    if raw_reason in (None, ""):
        raw_reason = event.get("terminal_reason", event.get("reason"))

    reason = str(getattr(raw_reason, "value", raw_reason) or "").strip().lower()
    reason = reason.replace("-", "_").replace(" ", "_")
    if not reason:
        if kind.endswith("_timeout"):
            reason = "timed_out"
        elif kind.endswith("_failed"):
            reason = "failed"
        elif kind in {"objective_refill_scan", "codebase_refill_scan"}:
            # Older supervisors emitted a *_scan event only after materializing
            # work.  This inference preserves their historical counters without
            # interpreting an arbitrary empty result as exhaustion.
            reason = "generated"

    scan_shape = any(
        key in nested or key in event
        for key in ("scan_mode", "analyzer_version", "candidate_funnel", "scan_kind")
    )
    is_scan_event = (
        kind in _REFILL_SCAN_EVENT_TYPES
        or "refill_scan" in kind
        or bool(event.get("scan_receipt_cid"))
        or bool(event.get("receipt_cid") and scan_shape)
        or bool(nested and scan_shape)
    )
    if not is_scan_event or reason not in REFILL_SCAN_TERMINAL_REASONS:
        return None

    def first(*keys: str, default: Any = "") -> Any:
        for source in (nested, event):
            for key in keys:
                value = source.get(key)
                if value not in (None, ""):
                    return value
        return default

    metadata = _mapping(nested.get("metadata"))
    funnel: Mapping[str, Any] = {}
    for candidate in (
        nested.get("candidate_funnel"),
        event.get("candidate_funnel"),
        metadata.get("candidate_funnel"),
    ):
        if isinstance(candidate, Mapping):
            funnel = candidate
            break

    # Some v1 analyzers put the bounded funnel counters directly in metadata.
    # Accept numeric scalar fields, while excluding timing/version identifiers
    # which are not candidate counts.
    if not funnel and metadata:
        ignored = {
            "timeout_seconds", "duration_seconds", "schema_version",
            "contract_version", "version", "current_open", "task_count",
        }
        funnel = {
            str(key): value
            for key, value in metadata.items()
            if key not in ignored
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
        }

    generated_count = int(max(0.0, _number(first("generated_count", default=0))))
    if not generated_count:
        items = nested.get("items", nested.get("findings"))
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
            generated_count = len(items)

    projection = {
        "receipt_cid": str(first("receipt_cid", "scan_receipt_cid", default="")),
        "scan_kind": str(first("scan_kind", "refill_kind", "source", default="")),
        "terminal_reason": reason,
        "scan_mode": str(first("scan_mode", "mode", default="")),
        "analyzer_version": str(first("analyzer_version", default="")),
        "repository_id": str(first("repository_id", "repository_identity", default="")),
        "tree_id": str(first("tree_id", "tree_identity", default="")),
        "started_at": str(first("started_at", default="")),
        "finished_at": str(first("finished_at", default="")),
        "duration_seconds": max(0.0, _number(first("duration_seconds", default=0))),
        "generated_count": generated_count,
        "safe_for_completion_reasoning": bool(
            first("safe_for_completion_reasoning", default=False)
        ),
        "health": str(first("health", "scan_health", default="")),
        "freshness": first("freshness", default=""),
        "artifact_path": str(first("artifact_path", "details_artifact_path", default="")),
        "artifact_cid": str(first("artifact_cid", "details_artifact_cid", default="")),
        "candidate_funnel": {
            str(key): int(max(0.0, _number(value)))
            for key, value in funnel.items()
            if not isinstance(value, bool) and isinstance(value, (int, float))
        },
    }
    if not projection["scan_kind"]:
        projection["scan_kind"] = "objective" if kind.startswith("objective_") else (
            "codebase" if kind.startswith("codebase_") else "unknown"
        )
    if not projection["finished_at"]:
        projection["finished_at"] = str(
            event.get("timestamp") or event.get("occurred_at") or ""
        )
    return projection


def _empty_scan_metrics() -> dict[str, Any]:
    by_reason = {reason: 0 for reason in REFILL_SCAN_TERMINAL_REASONS}
    return {
        "attempts": 0,
        "attempted": 0,
        "receipts": 0,
        "receipt_count": 0,
        "legacy_event_count": 0,
        "successful": 0,
        "skipped": 0,
        "failed_total": 0,
        **by_reason,
        "generated_count": 0,
        "by_terminal_reason": by_reason,
        "outcome_counts": by_reason,
        "by_scan_kind": {},
        "candidate_funnel": {},
        "latest_attempted_scan": None,
        "latest_successful_scan": None,
        "latest_attempt": None,
        "latest_successful": None,
    }


def _reduce_scan_metrics(
    events: Sequence[tuple[int, dict[str, Any], datetime | None]],
) -> dict[str, Any]:
    metrics = _empty_scan_metrics()
    scan_events: list[tuple[int, dict[str, Any], datetime | None, dict[str, Any]]] = []
    for index, event, occurred in events:
        projection = _scan_receipt_projection(event, _event_type(event))
        if projection is not None:
            scan_events.append((index, event, occurred, projection))

    # During migration a failed/timeout (and, in a few older paths, generated)
    # attempt first emitted a legacy event and then persisted its canonical
    # receipt event.  Correlate only the immediately preceding scan event with
    # the same kind/reason in a narrow time window.  This avoids counting the
    # compatibility event twice while retaining genuinely historical events
    # which have no receipt counterpart.
    superseded_legacy_positions: set[int] = set()
    for position, (_index, _event, occurred, projection) in enumerate(scan_events):
        if not projection["receipt_cid"] or position == 0:
            continue
        previous_position = position - 1
        _previous_index, _previous_event, previous_occurred, previous = scan_events[
            previous_position
        ]
        if previous["receipt_cid"]:
            continue
        if (
            previous["scan_kind"] != projection["scan_kind"]
            or previous["terminal_reason"] != projection["terminal_reason"]
        ):
            continue
        if occurred is None or previous_occurred is None:
            continue
        elapsed = (occurred - previous_occurred).total_seconds()
        if 0.0 <= elapsed <= 30.0:
            superseded_legacy_positions.add(previous_position)

    seen_receipts: set[str] = set()
    for position, (_index, _event, _occurred, projection) in enumerate(scan_events):
        if position in superseded_legacy_positions:
            continue
        receipt_cid = projection["receipt_cid"]
        if receipt_cid and receipt_cid in seen_receipts:
            continue
        if receipt_cid:
            seen_receipts.add(receipt_cid)

        reason = projection["terminal_reason"]
        metrics["attempts"] += 1
        metrics["attempted"] += 1
        if receipt_cid:
            metrics["receipts"] += 1
            metrics["receipt_count"] += 1
        else:
            metrics["legacy_event_count"] += 1
        metrics[reason] += 1
        metrics["by_terminal_reason"][reason] += 1
        if reason in REFILL_SCAN_SKIPPED_REASONS:
            metrics["skipped"] += 1
        if reason in REFILL_SCAN_FAILED_REASONS:
            metrics["failed_total"] += 1
        if reason in REFILL_SCAN_SUCCESS_REASONS:
            metrics["successful"] += 1
        metrics["generated_count"] += projection["generated_count"]

        kind = projection["scan_kind"]
        kind_counts = metrics["by_scan_kind"].setdefault(kind, {
            "attempts": 0,
            "skipped": 0,
            "failed_total": 0,
            **{reason_name: 0 for reason_name in REFILL_SCAN_TERMINAL_REASONS},
        })
        kind_counts["attempts"] += 1
        kind_counts[reason] += 1
        if reason in REFILL_SCAN_SKIPPED_REASONS:
            kind_counts["skipped"] += 1
        if reason in REFILL_SCAN_FAILED_REASONS:
            kind_counts["failed_total"] += 1
        for name, count in projection["candidate_funnel"].items():
            metrics["candidate_funnel"][name] = (
                metrics["candidate_funnel"].get(name, 0) + count
            )

        metrics["latest_attempted_scan"] = projection
        metrics["latest_attempt"] = projection
        if reason in REFILL_SCAN_SUCCESS_REASONS:
            metrics["latest_successful_scan"] = projection
            metrics["latest_successful"] = projection
    return metrics


def _explicit_phase(event: Mapping[str, Any]) -> str:
    phase = str(event.get("phase") or event.get("active_phase") or "").strip().lower()
    if "resolver" in phase or phase in {"conflict_repair", "resolving"}:
        return "resolver"
    if "validat" in phase or phase in {"test", "testing"}:
        return "validation"
    if "merge" in phase or phase in {"handoff", "integrating"}:
        return "merge"
    if phase in {"implementing", "implementation", "active", "running", "selected"}:
        return "active"
    if phase in SCHEDULER_PHASES:
        return phase
    state = str(event.get("state") or event.get("scheduler_state") or "").strip().lower()
    if state in {"accepted", "running", "implementing"}:
        return "active"
    if state in {"released", "expired", "pending", "registered", "queued"}:
        return "ready"
    if state in {"complete", "completed", "succeeded", "done"}:
        return "idle"
    if state in SCHEDULER_PHASES:
        return state
    return ""


def _phase_for_event(event: Mapping[str, Any], kind: str) -> str:
    explicit = _explicit_phase(event)
    if explicit:
        return explicit
    if kind == "daemon_pass":
        if event.get("active_task_id"):
            return "active"
        if int(_number(event.get("blocked_count"))) > 0 and int(_number(event.get("ready_count"))) == 0:
            return "blocked"
        if int(_number(event.get("ready_count"))) > 0:
            return "ready"
        return "idle"
    if kind in _RESOLVER_EVENTS or "resolver" in kind:
        return "resolver"
    if kind in _VALIDATION_START_EVENTS or kind.startswith("validation_") and "finished" not in kind:
        return "validation"
    if kind in _MERGE_QUEUE_EVENTS or kind in _MERGE_START_EVENTS:
        return "merge"
    if kind in _BLOCKED_EVENTS or "quarantin" in kind:
        return "blocked"
    if kind in _ACTIVE_EVENTS:
        return "active"
    if kind in _READY_EVENTS:
        return "ready"
    if kind in _IDLE_EVENTS or kind in _COMPLETION_EVENTS:
        return "idle"
    return ""


def _validation_interval(event: Mapping[str, Any]) -> tuple[float, datetime | None]:
    validation = event.get("validation_result")
    if not isinstance(validation, Mapping):
        validation = event.get("validation") if isinstance(event.get("validation"), Mapping) else {}
    results = validation.get("results") if isinstance(validation, Mapping) else []
    duration = 0.0
    earliest: datetime | None = None
    latest: datetime | None = None
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
        for result in results:
            if not isinstance(result, Mapping):
                continue
            started = _parse_timestamp(result.get("started_at") or result.get("start_time"))
            finished = _parse_timestamp(result.get("finished_at") or result.get("end_time"))
            duration += _seconds(started, finished)
            if started is not None and (earliest is None or started < earliest):
                earliest = started
            if finished is not None and (latest is None or finished > latest):
                latest = finished
    if not duration and isinstance(validation, Mapping):
        duration = max(
            0.0,
            _number(validation.get("duration_seconds"))
            or _seconds(
                _parse_timestamp(validation.get("started_at")),
                _parse_timestamp(validation.get("finished_at")),
            ),
        )
        earliest = earliest or _parse_timestamp(validation.get("started_at"))
    return duration, earliest


def _is_failed(event: Mapping[str, Any]) -> bool:
    if event.get("passed") is False or event.get("merged") is False or event.get("success") is False:
        return True
    returncode = event.get("returncode")
    return returncode not in (None, "", 0, "0")


def _is_conflict(event: Mapping[str, Any], kind: str) -> bool:
    if "conflict" in kind:
        return True
    values = " ".join(
        str(event.get(key) or "")
        for key in ("reason", "error", "failure_class", "merge_error", "code")
    ).lower()
    return "conflict" in values


@dataclass
class _Accumulator:
    identity: dict[str, str]
    metrics: dict[str, Any]
    phase: str = "idle"
    status: str = "unknown"
    last_event_type: str = ""
    last_event_at: str = ""
    display_task_id: str = ""
    queued_at: datetime | None = None
    implementation_started_at: datetime | None = None
    implementation_finished_at: datetime | None = None
    validation_started_at: datetime | None = None
    merge_queued_at: datetime | None = None
    merge_started_at: datetime | None = None
    merge_inflight: bool = False
    completed: bool = False


@dataclass(frozen=True)
class SchedulerSnapshot(Mapping[str, Any]):
    """Immutable public snapshot with mapping compatibility."""

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # JSON round-tripping prevents callers from mutating nested reducer
        # state shared with scheduler decisions.
        return json.loads(json.dumps(dict(self.payload), sort_keys=True))

    def to_payload(self) -> dict[str, Any]:
        return self.to_dict()

    @property
    def snapshot_id(self) -> str:
        return str(self.payload.get("snapshot_id") or "")

    @property
    def phases(self) -> Mapping[str, Any]:
        value = self.payload.get("phases")
        return value if isinstance(value, Mapping) else {}

    @property
    def metrics(self) -> Sequence[Mapping[str, Any]]:
        value = self.payload.get("metrics")
        return value if isinstance(value, Sequence) else ()

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


def scheduler_snapshot(
    events: Iterable[Mapping[str, Any]],
    *,
    now: datetime | str | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> SchedulerSnapshot:
    """Reduce supervisor events into authoritative phase and metric state."""

    raw_events = [dict(event) for event in events if isinstance(event, Mapping)]
    unique: list[tuple[int, dict[str, Any], datetime | None]] = []
    event_ids: set[str] = set()
    for index, event in enumerate(raw_events):
        event_id = str(event.get("event_id") or "")
        if event_id and event_id in event_ids:
            continue
        if event_id:
            event_ids.add(event_id)
        unique.append((index, event, _event_time(event)))
    unique.sort(key=lambda item: (item[2] is None, item[2] or datetime.max.replace(tzinfo=timezone.utc), item[0]))
    scan_metrics = _reduce_scan_metrics(unique)

    accumulators: dict[tuple[str, str, str, str, str], _Accumulator] = {}
    inherited_by_task: dict[str, dict[str, str]] = {}
    inherited_by_lane: dict[str, dict[str, str]] = {}

    for _index, event, occurred in unique:
        kind = _event_type(event)
        if kind in _REFILL_SCAN_EVENT_TYPES or _scan_receipt_projection(event, kind) is not None:
            # A repository scan is supervisor-level evidence, not a scheduler
            # task.  It contributes to scan_metrics but must not manufacture an
            # ``unknown`` idle task or alter a real task's current phase.
            continue
        task_alias = str(
            event.get("task_cid") or event.get("canonical_task_cid")
            or event.get("canonical_task_key") or event.get("task_id") or ""
        )
        lane_alias = str(
            event.get("lane_id") or event.get("parallel_lane")
            or event.get("bundle_key") or event.get("state_prefix") or ""
        )
        inherited = inherited_by_task.get(task_alias) or inherited_by_lane.get(lane_alias) or {}
        identity = normalize_metric_identity(event, {**dict(defaults or {}), **inherited})
        if task_alias:
            inherited_by_task[task_alias] = identity
        if lane_alias:
            inherited_by_lane[lane_alias] = identity
        key = _identity_key(identity)
        current = accumulators.get(key)
        if current is None:
            current = _Accumulator(identity=identity, metrics=_metric_defaults(identity))
            accumulators[key] = current

        current.last_event_type = kind
        if occurred is not None:
            current.last_event_at = occurred.isoformat()
        current.display_task_id = str(event.get("task_id") or current.display_task_id)
        tokens, cost = _usage(event)
        current.metrics["tokens"] += tokens
        current.metrics["cost_usd"] += cost

        phase = _phase_for_event(event, kind)
        if phase:
            current.phase = phase
            current.status = str(event.get("status") or event.get("state") or phase)

        if kind in _READY_EVENTS or kind == "task_selected":
            if current.queued_at is None:
                current.queued_at = occurred

        if kind == "implementation_started":
            current.metrics["implementation_attempts"] += 1
            attempt = int(_number(event.get("attempt")))
            if attempt > 1:
                current.metrics["retries"] += 1
            current.implementation_started_at = occurred
            if current.queued_at is not None:
                current.metrics["queue_wait_seconds"] += _seconds(current.queued_at, occurred)
                current.queued_at = None
        elif "retry" in kind and kind not in {"retry_exhausted"}:
            current.metrics["retries"] += 1

        if kind in _VALIDATION_START_EVENTS or phase == "validation":
            if current.validation_started_at is None:
                current.validation_started_at = occurred
            if current.implementation_started_at is not None:
                current.metrics["implementation_duration_seconds"] += _seconds(
                    current.implementation_started_at, occurred
                )
                current.implementation_started_at = None

        if kind in _VALIDATION_END_EVENTS:
            current.metrics["validation_duration_seconds"] += _seconds(
                current.validation_started_at, occurred
            )
            current.validation_started_at = None
            if not _is_failed(event):
                current.phase = "active"

        if kind == "implementation_finished":
            validation_duration, validation_start = _validation_interval(event)
            if validation_duration:
                current.metrics["validation_duration_seconds"] += validation_duration
            elif current.validation_started_at is not None:
                current.metrics["validation_duration_seconds"] += _seconds(
                    current.validation_started_at, occurred
                )
            if current.implementation_started_at is not None:
                implementation_finish = validation_start or occurred
                current.metrics["implementation_duration_seconds"] += _seconds(
                    current.implementation_started_at, implementation_finish
                )
            current.implementation_started_at = None
            current.validation_started_at = None
            current.implementation_finished_at = occurred
            if _is_failed(event):
                current.phase = "blocked"
                current.status = "failed"
            else:
                current.phase = "merge" if event.get("merge_pending") else "idle"
                current.status = "implemented"

        if kind in _MERGE_QUEUE_EVENTS:
            current.merge_queued_at = occurred
            current.phase = "merge"

        if kind in _MERGE_START_EVENTS:
            if not current.merge_inflight:
                current.metrics["merge_attempts"] += 1
            current.merge_inflight = True
            current.merge_started_at = occurred or _parse_timestamp(event.get("started_at"))
            wait_start = current.merge_queued_at or current.implementation_finished_at
            current.metrics["merge_wait_seconds"] += _seconds(wait_start, current.merge_started_at)
            current.merge_queued_at = None
            current.phase = "merge"

        if kind in _MERGE_END_EVENTS:
            explicit_start = _parse_timestamp(event.get("started_at"))
            merge_start = current.merge_started_at or explicit_start
            if not current.merge_inflight:
                current.metrics["merge_attempts"] += 1
                wait_start = current.merge_queued_at or current.implementation_finished_at
                current.metrics["merge_wait_seconds"] += _seconds(wait_start, merge_start or occurred)
            current.merge_started_at = None
            current.merge_queued_at = None
            current.merge_inflight = False
            if _is_conflict(event, kind):
                current.metrics["conflicts"] += 1
            if _is_failed(event):
                current.phase = "blocked"
                current.status = "merge_failed"
            else:
                current.phase = "idle"
                current.status = "merged"
                if not current.completed:
                    current.metrics["completions"] += 1
                    current.completed = True

        if _is_conflict(event, kind) and kind not in _MERGE_END_EVENTS:
            current.metrics["conflicts"] += 1

        if kind in _COMPLETION_EVENTS and not current.completed:
            current.metrics["completions"] += 1
            current.completed = True
            current.phase = "idle"
            current.status = "completed"

        if kind in _RESOLVER_EVENTS or phase == "resolver":
            current.phase = "resolver"

        if kind in _BLOCKED_EVENTS:
            current.phase = "blocked"
            current.status = "blocked"

    rows: list[dict[str, Any]] = []
    task_states: list[dict[str, Any]] = []
    phase_items: dict[str, list[dict[str, Any]]] = {phase: [] for phase in SCHEDULER_PHASES}
    for key in sorted(accumulators):
        current = accumulators[key]
        metrics = current.metrics
        metrics["conflict_rate"] = (
            metrics["conflicts"] / metrics["merge_attempts"]
            if metrics["merge_attempts"] else 0.0
        )
        metrics["retry_rate"] = (
            metrics["retries"] / metrics["implementation_attempts"]
            if metrics["implementation_attempts"] else 0.0
        )
        metrics["completion_count"] = metrics["completions"]
        metrics["total_tokens"] = metrics["tokens"]
        metrics["total_cost_usd"] = metrics["cost_usd"]
        rows.append(dict(metrics))
        state = {
            **current.identity,
            "task_id": current.display_task_id,
            "phase": current.phase,
            "status": current.status,
            "last_event_type": current.last_event_type,
            "last_event_at": current.last_event_at,
        }
        task_states.append(state)
        phase_items[current.phase].append(state)

    dimensions_all = normalize_metric_identity({}, {
        "goal_cid": "all", "subgoal_cid": "all", "task_cid": "all",
        "lane_id": "all", "provider_id": "all",
    })
    totals = _metric_defaults(dimensions_all)
    for row in rows:
        for name in (
            "queue_wait_seconds", "implementation_duration_seconds",
            "validation_duration_seconds", "merge_wait_seconds", "cost_usd",
        ):
            totals[name] += float(row[name])
        for name in (
            "implementation_attempts", "merge_attempts", "conflicts", "retries",
            "completions", "tokens",
        ):
            totals[name] += int(row[name])
    totals["conflict_rate"] = (
        totals["conflicts"] / totals["merge_attempts"] if totals["merge_attempts"] else 0.0
    )
    totals["retry_rate"] = (
        totals["retries"] / totals["implementation_attempts"]
        if totals["implementation_attempts"] else 0.0
    )
    totals["completion_count"] = totals["completions"]
    totals["total_tokens"] = totals["tokens"]
    totals["total_cost_usd"] = totals["cost_usd"]
    # Additive flat aliases let existing totals-only consumers adopt the new
    # receipt metrics without changing how they access the snapshot.  The
    # structured ``scan_metrics`` block below remains the authoritative view.
    totals.update({
        "scan_attempts": scan_metrics["attempts"],
        "scan_receipts": scan_metrics["receipts"],
        "scan_successful": scan_metrics["successful"],
        "scan_skipped": scan_metrics["skipped"],
        "scan_failed_total": scan_metrics["failed_total"],
        "scan_generated_count": scan_metrics["generated_count"],
        "refill_scan_attempts": scan_metrics["attempts"],
        "refill_scan_receipts": scan_metrics["receipts"],
        "refill_scan_successful": scan_metrics["successful"],
        "refill_scan_skipped": scan_metrics["skipped"],
        "refill_scan_failed_total": scan_metrics["failed_total"],
        "refill_scan_generated_count": scan_metrics["generated_count"],
    })
    for reason in REFILL_SCAN_TERMINAL_REASONS:
        totals[f"scan_{reason}"] = scan_metrics[reason]
        totals[f"refill_scan_{reason}"] = scan_metrics[reason]

    phases = {
        phase: {"count": len(phase_items[phase]), "items": phase_items[phase]}
        for phase in SCHEDULER_PHASES
    }
    phase_counts = {phase: phases[phase]["count"] for phase in SCHEDULER_PHASES}
    generated_at = _now_iso(now)
    fingerprint_material = {
        "schema": SCHEDULER_SNAPSHOT_SCHEMA,
        "events": [event for _index, event, _occurred in unique],
        "phase_counts": phase_counts,
        "metrics": rows,
        "scan_metrics": scan_metrics,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": SCHEDULER_SNAPSHOT_SCHEMA,
        "generated_at": generated_at,
        "snapshot_id": snapshot_id,
        "authoritative": True,
        "source": "event_log",
        "source_event_count": len(unique),
        "phases": phases,
        "phase_counts": phase_counts,
        "counts": dict(phase_counts),
        "task_states": task_states,
        "metrics": rows,
        "totals": totals,
        "scan_metrics": scan_metrics,
    }
    return SchedulerSnapshot(payload)


def build_scheduler_snapshot(
    events: Iterable[Mapping[str, Any]],
    *,
    now: datetime | str | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> SchedulerSnapshot:
    return scheduler_snapshot(events, now=now, defaults=defaults)


derive_scheduler_snapshot = build_scheduler_snapshot
scheduler_metrics_snapshot = build_scheduler_snapshot


def build_scheduler_snapshot_from_paths(
    paths: Iterable[Path | str],
    *,
    now: datetime | str | None = None,
    defaults: Mapping[str, Any] | None = None,
    include_rotated: bool = True,
) -> SchedulerSnapshot:
    events = read_jsonl_event_sources(paths, include_rotated=include_rotated)
    return scheduler_snapshot(events, now=now, defaults=defaults)


def write_scheduler_snapshot(path: Path | str, snapshot: SchedulerSnapshot | Mapping[str, Any]) -> Path:
    """Atomically publish a scheduler snapshot for operator readers."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.to_dict() if isinstance(snapshot, SchedulerSnapshot) else dict(snapshot)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


publish_scheduler_snapshot = write_scheduler_snapshot


def read_scheduler_snapshot(path: Path | str) -> SchedulerSnapshot | None:
    """Read a published snapshot, returning ``None`` for partial/invalid files."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEDULER_SNAPSHOT_SCHEMA:
        return None
    return SchedulerSnapshot(payload)


def scheduler_state_events(
    tasks: Iterable[Mapping[str, Any]],
    *,
    lanes: Iterable[Mapping[str, Any]] = (),
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Project lease/lane state into lifecycle events for the shared reducer."""

    occurred_at = timestamp or _now_iso()
    events: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        state = str(task.get("state") or task.get("lease_state") or "ready").lower()
        if state in {"released", "expired", "pending", "registered"}:
            state = "ready"
        elif state in {"complete", "completed", "succeeded"}:
            state = "idle"
        elif state == "accepted":
            state = "active"
        if state not in SCHEDULER_PHASES:
            state = "ready"
        events.append({**task, "type": "scheduler_state", "timestamp": occurred_at, "phase": state})
    for raw in lanes:
        lane = dict(raw)
        phase = str(lane.get("phase") or lane.get("active_phase") or lane.get("state") or "active").lower()
        events.append({**lane, "type": "scheduler_lane_state", "timestamp": occurred_at, "phase": phase})
    return events


def ready_task_cids(snapshot: SchedulerSnapshot | Mapping[str, Any]) -> tuple[str, ...]:
    """Return canonical ready task identities in published snapshot order."""

    payload = snapshot.payload if isinstance(snapshot, SchedulerSnapshot) else snapshot
    phases = payload.get("phases") if isinstance(payload, Mapping) else {}
    ready = phases.get("ready") if isinstance(phases, Mapping) else {}
    items = ready.get("items") if isinstance(ready, Mapping) else []
    return tuple(
        str(item.get("task_cid") or item.get("canonical_task_cid") or "")
        for item in items
        if isinstance(item, Mapping) and (item.get("task_cid") or item.get("canonical_task_cid"))
    )


__all__ = [
    "REFILL_SCAN_FAILED_REASONS",
    "REFILL_SCAN_SKIPPED_REASONS",
    "REFILL_SCAN_SUCCESS_REASONS",
    "REFILL_SCAN_TERMINAL_REASONS",
    "SCHEDULER_PHASES",
    "SCHEDULER_SNAPSHOT_SCHEMA",
    "SchedulerSnapshot",
    "build_scheduler_snapshot",
    "build_scheduler_snapshot_from_paths",
    "derive_scheduler_snapshot",
    "normalize_metric_identity",
    "publish_scheduler_snapshot",
    "read_scheduler_snapshot",
    "ready_task_cids",
    "scheduler_metrics_snapshot",
    "scheduler_snapshot",
    "scheduler_state_events",
    "write_scheduler_snapshot",
]
