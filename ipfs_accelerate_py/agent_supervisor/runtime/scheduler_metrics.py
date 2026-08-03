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


SCHEDULER_SNAPSHOT_SCHEMA_VERSION = 2
SCHEDULER_SNAPSHOT_SCHEMA = "ipfs_accelerate_py.agent_supervisor.scheduler-snapshot@2"
LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS = frozenset(
    {"ipfs_accelerate_py.agent_supervisor.scheduler-snapshot@1"}
)
GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION = 1
GOAL_COMPLETION_DIAGNOSTICS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.goal-completion-diagnostics@1"
)
PROOF_ROLLOUT_QUERY_SCHEMA_VERSION = 1
PROOF_ROLLOUT_QUERY_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-rollout-query@1"
)
RESOURCE_ADMISSION_METRICS_SCHEMA_VERSION = 1
RESOURCE_ADMISSION_METRICS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.resource-admission-metrics@1"
)
RESOURCE_ADMISSION_EVENT_TYPES = frozenset(
    {
        "adaptive_resource_snapshot",
        "adaptive_resources_observed",
        "resource_admission_observed",
        "resource_admission_snapshot",
        "resource_schedule_observed",
        "resource_schedule_snapshot",
        "scheduler_resource_snapshot",
    }
)
RESOURCE_ADMISSION_STAGES = (
    "analysis",
    "inference",
    "proof",
    "validation",
    "merge",
    "persistence",
    "execution",
)
MAX_PROOF_ROLLOUT_QUERY_ROWS = 128
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
_EXPLICIT_TASK_COMPLETION_EVENTS = frozenset(
    {"task_completed", "task_succeeded"}
)
_SCHEDULER_STATE_PROJECTION_EVENT = "scheduler_state_projection"
_SCHEDULER_STATE_PROJECTION_SCOPE = "authoritative_current"

_RESOURCE_PROJECTION_KEYS = (
    "resource_admission",
    "adaptive_resources",
    "resource_schedule",
    "resource_schedule_snapshot",
)
_RESOURCE_CAPACITY_INTEGER_FIELDS = (
    "configured_limit",
    "effective_limit",
    "active",
    "queued",
    "available",
    "pressure_percent",
    "queue_depth",
    "merge_age_ms",
    "provider_available_slots",
    "active_leases",
    "recovery_samples",
    "observed_at_ms",
)
_RESOURCE_METRIC_INTEGER_FIELDS = (
    "scheduled",
    "admitted",
    "backpressured",
    "completed",
    "accepted",
    "cancelled",
    "leases_acquired",
    "leases_released",
    "lease_transitions",
    "recovery_events",
    "contraction_events",
    "active_leases",
    "total_duration_ms",
    "admission_ratio_millionths",
    "acceptance_throughput_per_million_ms",
)


def _resource_stage(value: Any) -> str:
    raw = str(getattr(value, "value", value) or "").strip().lower()
    raw = raw.replace("/", "_").replace(" ", "_").replace("-", "_")
    aliases = {
        "analyze": "analysis",
        "analysis_pipeline": "analysis",
        "model": "inference",
        "llm": "inference",
        "provider": "inference",
        "solve": "proof",
        "solver": "proof",
        "validate": "validation",
        "acceptance": "validation",
        "git": "merge",
        "git_merge": "merge",
        "merging": "merge",
        "persist": "persistence",
        "artifact": "persistence",
        "scheduler": "execution",
    }
    return aliases.get(raw, raw) if raw else "execution"


def _resource_integer(value: Any, default: int = 0) -> int:
    """Return one non-negative integer without leaking floats into artifacts."""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text.startswith("+") and text[1:].isdigit())):
            return max(0, int(text))
    return default


def _resource_reason(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _resource_reasons(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = value
    else:
        values = ()
    return tuple(sorted({_resource_reason(item) for item in values if _resource_reason(item)}))


def _resource_reason_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        reason: count
        for reason, count in sorted(
            (
                (_resource_reason(raw_reason), _resource_integer(raw_count))
                for raw_reason, raw_count in value.items()
            )
        )
        if reason and count
    }


def _resource_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resource_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for stage, raw in value.items():
            if isinstance(raw, Mapping):
                rows.append({"stage": stage, **dict(raw)})
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _declares_resource_admission(value: Mapping[str, Any]) -> bool:
    kind = _event_type(value)
    resource_metric_shape = (
        "stages" in value
        and any(
            key in value
            for key in (
                "active_lease_count",
                "backpressure_reasons",
                "backpressure_reason_counts",
                "leases_acquired",
                "leases_released",
            )
        )
    )
    return (
        kind in RESOURCE_ADMISSION_EVENT_TYPES
        or value.get("schema") == RESOURCE_ADMISSION_METRICS_SCHEMA
        or any(key in value for key in _RESOURCE_PROJECTION_KEYS)
        or resource_metric_shape
        or any(
            key in value
            for key in (
                "effective_slots",
                "stage_capacities",
                "stage_metrics",
                "adaptive_metrics",
                "backpressure_reason_counts",
            )
        )
    )


def _resource_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in _RESOURCE_PROJECTION_KEYS:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            return nested
    return value


def _resource_signal(
    payload: Mapping[str, Any],
    host: Mapping[str, Any],
    signals: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: int = 0,
) -> int:
    for source in (signals, payload, host):
        for name in names:
            if name in source:
                return _resource_integer(source.get(name), default)
    return default


def _resource_admission_projection(
    value: Mapping[str, Any] | None,
    *,
    occurred: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate and canonicalize one operator-facing resource observation."""

    if not isinstance(value, Mapping):
        return None
    payload = _resource_payload(value)
    if not isinstance(payload, Mapping):
        return None
    # A declared wrapper must carry at least one schedule/resource field. This
    # avoids accepting a type-only event as authoritative zero-capacity data.
    meaningful = {
        "observed_at_ms",
        "configured_max_lanes",
        "effective_slots",
        "available_slots",
        "admitted_count",
        "backpressured_count",
        "active_lease_count",
        "host",
        "providers",
        "decisions",
        "stage_capacities",
        "stage_metrics",
        "stages",
        "adaptive_metrics",
        "signals",
        "backpressure_reasons",
        "backpressure_reason_counts",
    }
    if not meaningful.intersection(payload):
        return None

    host = _resource_mapping(payload.get("host"))
    signals_input = _resource_mapping(payload.get("signals"))
    providers = _resource_rows(payload.get("providers"))
    decisions = _resource_rows(payload.get("decisions"))

    observed_at_ms = _resource_integer(payload.get("observed_at_ms"))
    if not observed_at_ms:
        observed_at_ms = _resource_integer(signals_input.get("observed_at_ms"))
    if not observed_at_ms and occurred is not None:
        observed_at_ms = max(0, int(occurred.timestamp() * 1000))

    capacity_rows = _resource_rows(
        payload.get("stage_capacities")
        or payload.get("capacities_by_stage")
    )
    raw_adaptive_metrics = _resource_mapping(payload.get("adaptive_metrics"))
    metric_rows = _resource_rows(
        payload.get("stage_metrics")
        or payload.get("stages")
        or raw_adaptive_metrics.get("stages")
        or payload.get("metrics_by_stage")
    )

    capacities_by_stage: dict[str, dict[str, Any]] = {}
    for row in capacity_rows:
        stage = _resource_stage(row.get("stage"))
        normalized = {
            "stage": stage,
            **{
                name: _resource_integer(row.get(name))
                for name in _RESOURCE_CAPACITY_INTEGER_FIELDS
            },
        }
        reason = _resource_reason(row.get("reason"))
        if reason:
            normalized["reason"] = reason
        hysteresis_state = _resource_reason(row.get("hysteresis_state"))
        if hysteresis_state:
            normalized["hysteresis_state"] = hysteresis_state
        signal_limits = {
            str(name): _resource_integer(limit)
            for name, limit in sorted(
                _resource_mapping(row.get("signal_limits")).items()
            )
        }
        if signal_limits:
            normalized["signal_limits"] = signal_limits
        reason_counts = _resource_reason_counts(
            row.get("backpressure_reason_counts")
            or (
                row.get("backpressure_reasons")
                if isinstance(row.get("backpressure_reasons"), Mapping)
                else None
            )
            or row.get("reason_counts")
        )
        if reason_counts:
            normalized["backpressure_reason_counts"] = reason_counts
        capacities_by_stage[stage] = normalized

    metrics_by_stage: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        stage = _resource_stage(row.get("stage"))
        normalized = {
            "stage": stage,
            **{
                name: _resource_integer(row.get(name))
                for name in _RESOURCE_METRIC_INTEGER_FIELDS
            },
        }
        reason_counts = _resource_reason_counts(
            row.get("backpressure_reason_counts")
            or (
                row.get("backpressure_reasons")
                if isinstance(row.get("backpressure_reasons"), Mapping)
                else None
            )
            or row.get("reason_counts")
        )
        if reason_counts:
            normalized["backpressure_reason_counts"] = reason_counts
        metrics_by_stage[stage] = normalized

    # Canonical persisted projections carry a combined lookup in addition to
    # the two ordered row collections. Retain its per-stage reason histogram
    # when validating a read-back so write/read is lossless.
    for row in _resource_rows(payload.get("by_stage")):
        stage = _resource_stage(row.get("stage"))
        reason_counts = _resource_reason_counts(
            row.get("backpressure_reason_counts") or row.get("reason_counts")
        )
        if not reason_counts:
            continue
        if stage in capacities_by_stage:
            capacities_by_stage[stage]["backpressure_reason_counts"] = reason_counts
        elif stage in metrics_by_stage:
            metrics_by_stage[stage]["backpressure_reason_counts"] = reason_counts

    # A live schedule's explicit empty histogram is authoritative.  Falling
    # through to cumulative adaptive metrics here makes resolved historical
    # pressure look like current backpressure after the queue drains.
    if "backpressure_reason_counts" in payload:
        aggregate_reason_counts = _resource_reason_counts(
            payload.get("backpressure_reason_counts")
        )
    elif "backpressure_counts" in payload:
        aggregate_reason_counts = _resource_reason_counts(
            payload.get("backpressure_counts")
        )
    elif "backpressure_reasons" in payload:
        aggregate_reason_counts = _resource_reason_counts(
            payload.get("backpressure_reasons")
        )
    elif isinstance(raw_adaptive_metrics.get("backpressure_reasons"), Mapping):
        aggregate_reason_counts = _resource_reason_counts(
            raw_adaptive_metrics.get("backpressure_reasons")
        )
    else:
        aggregate_reason_counts = _resource_reason_counts(
            payload.get("reason_counts")
        )
    decision_reason_counts: dict[str, int] = {}
    decision_stage_reason_counts: dict[str, dict[str, int]] = {}
    backpressured_decisions = 0
    for decision in decisions:
        admitted = bool(decision.get("admitted") or decision.get("allowed"))
        if admitted:
            continue
        backpressured_decisions += 1
        reasons = _resource_reasons(
            decision.get("reasons") or decision.get("backpressure_reasons")
            or decision.get("reason")
        )
        stage = _resource_stage(decision.get("stage"))
        stage_counts = decision_stage_reason_counts.setdefault(stage, {})
        for reason in reasons:
            decision_reason_counts[reason] = decision_reason_counts.get(reason, 0) + 1
            stage_counts[reason] = stage_counts.get(reason, 0) + 1
    if decision_reason_counts:
        aggregate_reason_counts = decision_reason_counts

    listed_reasons = _resource_reasons(payload.get("backpressure_reasons"))
    if not aggregate_reason_counts:
        aggregate_reason_counts = {reason: 1 for reason in listed_reasons}
    reasons = tuple(sorted({*listed_reasons, *aggregate_reason_counts}))
    for stage, reason_counts in decision_stage_reason_counts.items():
        target = capacities_by_stage.get(stage) or metrics_by_stage.get(stage)
        if target is not None:
            target["backpressure_reason_counts"] = dict(
                sorted(reason_counts.items())
            )

    all_stages = sorted(
        {
            *capacities_by_stage,
            *metrics_by_stage,
            *decision_stage_reason_counts,
        },
        key=lambda stage: (
            RESOURCE_ADMISSION_STAGES.index(stage)
            if stage in RESOURCE_ADMISSION_STAGES
            else len(RESOURCE_ADMISSION_STAGES),
            stage,
        ),
    )
    by_stage: dict[str, dict[str, Any]] = {}
    for stage in all_stages:
        capacity = capacities_by_stage.get(stage, {})
        metrics = metrics_by_stage.get(stage, {})
        stage_counts = _resource_reason_counts(
            capacity.get("backpressure_reason_counts")
            or metrics.get("backpressure_reason_counts")
        )
        if stage in decision_stage_reason_counts:
            stage_counts = dict(sorted(decision_stage_reason_counts[stage].items()))
        by_stage[stage] = {
            "stage": stage,
            **{
                name: _resource_integer(capacity.get(name))
                for name in _RESOURCE_CAPACITY_INTEGER_FIELDS
            },
            **{
                name: _resource_integer(metrics.get(name))
                for name in _RESOURCE_METRIC_INTEGER_FIELDS
            },
            "backpressure_reason_counts": stage_counts,
        }
        if capacity.get("reason"):
            by_stage[stage]["reason"] = capacity["reason"]
        if capacity.get("hysteresis_state"):
            by_stage[stage]["hysteresis_state"] = capacity["hysteresis_state"]
        if capacity.get("signal_limits"):
            by_stage[stage]["signal_limits"] = dict(capacity["signal_limits"])

    provider_available_slots = _resource_signal(
        payload,
        host,
        signals_input,
        ("provider_available_slots", "available_provider_capacity"),
    )
    if not provider_available_slots:
        provider_available_slots = sum(
            _resource_integer(provider.get("available_concurrency"))
            for provider in providers
            if provider.get("healthy", True)
            and not _resource_integer(provider.get("retry_after_ms"))
        )

    queue_depth = _resource_signal(
        payload, host, signals_input, ("queue_depth", "queued_count", "ready_count")
    )
    if not queue_depth:
        queue_depth = sum(
            _resource_integer(row.get("queued"))
            for row in capacities_by_stage.values()
        )
    merge_age_ms = _resource_signal(
        payload,
        host,
        signals_input,
        ("merge_age_ms", "oldest_merge_age_ms", "merge_queue_age_ms"),
    )
    if not merge_age_ms:
        merge_age_ms = max(
            (
                _resource_integer(row.get("merge_age_ms"))
                for row in capacities_by_stage.values()
            ),
            default=0,
        )
    active_lease_count = _resource_signal(
        payload,
        host,
        signals_input,
        ("active_lease_count", "active_leases", "lease_count"),
    )
    raw_active_leases = payload.get("active_leases")
    if (
        not active_lease_count
        and isinstance(raw_active_leases, Sequence)
        and not isinstance(raw_active_leases, (str, bytes, bytearray))
    ):
        active_lease_count = len(raw_active_leases)
    if not active_lease_count:
        active_lease_count = _resource_integer(host.get("active_workers"))

    signals = {
        "cpu_percent": _resource_signal(
            payload, host, signals_input, ("cpu_percent", "cpu_usage_percent")
        ),
        "memory_percent": _resource_signal(
            payload, host, signals_input, ("memory_percent", "memory_usage_percent")
        ),
        "memory_available_bytes": _resource_signal(
            payload, host, signals_input,
            ("memory_available_bytes", "available_memory_bytes"),
        ),
        "gpu_memory_percent": _resource_signal(
            payload, host, signals_input,
            ("gpu_memory_percent", "gpu_memory_usage_percent"),
        ),
        "gpu_memory_available_bytes": _resource_signal(
            payload, host, signals_input,
            ("gpu_memory_available_bytes", "available_gpu_memory_bytes"),
        ),
        "disk_percent": _resource_signal(
            payload, host, signals_input, ("disk_percent", "disk_usage_percent")
        ),
        "disk_available_bytes": _resource_signal(
            payload, host, signals_input,
            ("disk_available_bytes", "available_disk_bytes"),
        ),
        "provider_available_slots": provider_available_slots,
        "queue_depth": queue_depth,
        "merge_age_ms": merge_age_ms,
        "active_lease_count": active_lease_count,
    }
    configured = _resource_integer(
        payload.get("configured_max_lanes"),
        _resource_integer(_resource_mapping(payload.get("policy")).get("max_lanes")),
    )
    live_backpressure_declared = any(
        name in payload
        for name in (
            "backpressured_count",
            "decisions",
            "backpressure_reason_counts",
            "backpressure_counts",
            "backpressure_reasons",
        )
    )
    if "backpressured_count" in payload:
        backpressured_count = _resource_integer(
            payload.get("backpressured_count")
        )
    elif "decisions" in payload:
        backpressured_count = backpressured_decisions
    elif live_backpressure_declared:
        backpressured_count = sum(aggregate_reason_counts.values())
    else:
        backpressured_count = 0
    if not live_backpressure_declared:
        backpressured_count = sum(
            _resource_integer(row.get("backpressured"))
            for row in metrics_by_stage.values()
        )

    stage_capacities = [capacities_by_stage[stage] for stage in all_stages if stage in capacities_by_stage]
    stage_metrics = [metrics_by_stage[stage] for stage in all_stages if stage in metrics_by_stage]
    return {
        "schema": RESOURCE_ADMISSION_METRICS_SCHEMA,
        "schema_version": RESOURCE_ADMISSION_METRICS_SCHEMA_VERSION,
        "observed": True,
        "observed_at_ms": observed_at_ms,
        "configured_max_lanes": configured,
        "effective_slots": _resource_integer(payload.get("effective_slots")),
        "available_slots": _resource_integer(payload.get("available_slots")),
        "admitted_count": _resource_integer(payload.get("admitted_count")),
        "backpressured_count": backpressured_count,
        "active_lease_count": active_lease_count,
        "queue_depth": queue_depth,
        "merge_age_ms": merge_age_ms,
        "backpressure_reasons": list(reasons),
        "backpressure_reason_counts": dict(sorted(aggregate_reason_counts.items())),
        "signals": signals,
        "stage_capacities": stage_capacities,
        "stage_metrics": stage_metrics,
        "by_stage": by_stage,
    }


def project_resource_admission_metrics(
    observations: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return the latest valid deterministic resource-admission projection.

    A direct resource schedule mapping is accepted for integrations. For event
    streams, timestamped observations supersede undated legacy observations;
    source order breaks exact timestamp ties.
    """

    if isinstance(observations, Mapping):
        values = [dict(observations)]
    else:
        values = [
            dict(value) for value in observations if isinstance(value, Mapping)
        ]
    candidates: list[tuple[int, Mapping[str, Any], datetime | None]] = []
    for index, value in enumerate(values):
        if _declares_resource_admission(value):
            candidates.append((index, value, _event_time(value)))
    candidates.sort(
        key=lambda item: (
            item[2] is not None,
            item[2] or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        )
    )
    latest: dict[str, Any] | None = None
    for _index, value, occurred in candidates:
        projected = _resource_admission_projection(value, occurred=occurred)
        if projected is None:
            raise ValueError(
                "resource admission observation has an invalid projection"
            )
        latest = projected
    return latest


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
    """Return canonical scheduler and proof metric dimensions.

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
    tree = _first_text(
        sources,
        (
            "repository_tree_id", "tree_id", "canonical_tree_id",
            "candidate_tree_id", "git_tree_id",
        ),
    ) or UNKNOWN_IDENTITY
    template = _first_text(
        sources,
        (
            "template_id", "canonical_template_id",
            "obligation_template_id", "template",
        ),
    ) or UNKNOWN_IDENTITY
    resource_class = _first_text(
        sources,
        (
            "resource_class", "canonical_resource_class",
            "worker_class", "resource_pool",
        ),
    ) or UNKNOWN_IDENTITY
    return {
        "goal_cid": goal,
        "subgoal_cid": subgoal,
        "task_cid": task,
        "canonical_task_cid": task,
        "lane_id": lane,
        "provider_id": provider,
        "repository_tree_id": tree,
        "tree_id": tree,
        "template_id": template,
        "resource_class": resource_class,
        # Explicit aliases make the canonical nature discoverable to API
        # clients without forcing older clients to rename their dimensions.
        "canonical_goal_id": goal,
        "canonical_subgoal_id": subgoal,
        "canonical_task_id": task,
        "canonical_lane_id": lane,
        "canonical_provider_id": provider,
        "canonical_tree_id": tree,
        "canonical_template_id": template,
        "canonical_resource_class": resource_class,
    }


def _identity_key(identity: Mapping[str, str]) -> tuple[str, ...]:
    return (
        identity["goal_cid"],
        identity["subgoal_cid"],
        identity["task_cid"],
        identity["lane_id"],
        identity["provider_id"],
        identity["repository_tree_id"],
        identity["template_id"],
        identity["resource_class"],
    )


def _task_state_key(
    identity: Mapping[str, str],
    identity_key: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return the logical identity used by current scheduler-state gauges."""

    canonical_task_cid = identity["task_cid"]
    if canonical_task_cid != UNKNOWN_IDENTITY:
        return ("task", canonical_task_cid)
    return ("identity", *(identity_key or _identity_key(identity)))


def _metric_defaults(identity: Mapping[str, str]) -> dict[str, Any]:
    return {
        **dict(identity),
        "queue_wait_seconds": 0.0,
        "implementation_duration_seconds": 0.0,
        "validation_duration_seconds": 0.0,
        "merge_wait_seconds": 0.0,
        "queue_latency_seconds": 0.0,
        "solver_latency_seconds": 0.0,
        "kernel_latency_seconds": 0.0,
        "model_latency_seconds": 0.0,
        "validation_latency_seconds": 0.0,
        "merge_latency_seconds": 0.0,
        "cancellation_latency_seconds": 0.0,
        "cache_latency_seconds": 0.0,
        "queue_latency_ms": 0,
        "solver_latency_ms": 0,
        "kernel_latency_ms": 0,
        "model_latency_ms": 0,
        "validation_latency_ms": 0,
        "merge_latency_ms": 0,
        "cancellation_latency_ms": 0,
        "cache_latency_ms": 0,
        "cancellations": 0,
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


def _string_list(value: Any) -> list[str]:
    """Return unique, non-empty strings from a compatibility field."""

    if value in (None, ""):
        return []
    values = value if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            text = str(
                item.get("acceptance_criterion")
                or item.get("criterion")
                or item.get("receipt_cid")
                or item.get("reason_code")
                or item.get("reason")
                or json.dumps(dict(item), sort_keys=True, default=str)
            ).strip()
        else:
            text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _diagnostic_list(value: Any) -> list[Any]:
    """Preserve structured diagnostics while deduplicating JSON values."""

    if value in (None, ""):
        return []
    values = value if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ) else [value]
    result: list[Any] = []
    seen: set[str] = set()
    for item in values:
        if item in (None, ""):
            continue
        try:
            normalized = json.loads(json.dumps(item, sort_keys=True, default=str))
        except (TypeError, ValueError):
            normalized = str(item)
        key = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _proof_rollout_projection(value: Any) -> dict[str, Any] | None:
    """Return a validated public proof-rollout projection.

    Supervisor status, daemon events, and scheduler snapshots wrap the same
    projection under slightly different compatibility keys.  Every read is
    revalidated at this boundary so an event cannot smuggle a transcript,
    witness, or provider payload into the operator snapshot.
    """

    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        return None

    candidate: Any = value
    for key in (
        "proof_rollout",
        "proof_rollout_status",
        "proof_rollout_diagnostics",
    ):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break
    if not isinstance(candidate, Mapping):
        return None

    # Keep the import local.  The policy module imports proof metrics and the
    # latter exposes scheduler compatibility aliases at module load time.
    from ..proof.formal_verification_policy import ProofRolloutStatus

    try:
        return ProofRolloutStatus(candidate).to_dict()
    except (TypeError, ValueError):
        return None


def _declares_proof_rollout(value: Any) -> bool:
    """Return whether a record claims to carry a rollout projection."""

    if not isinstance(value, Mapping):
        return False
    if str(value.get("schema") or "").endswith("proof-rollout-status@1"):
        return True
    return any(
        key in value
        for key in (
            "proof_rollout",
            "proof_rollout_status",
            "proof_rollout_diagnostics",
        )
    )


def proof_rollout_diagnostics(value: Any) -> dict[str, Any] | None:
    """Expose the validated bounded rollout view from a status or snapshot.

    A detached copy is returned so query clients cannot mutate a scheduler
    snapshot in place.  Invalid or absent projections are reported as
    unavailable rather than being converted into an optimistic shadow mode.
    """

    if isinstance(value, (Path, str)):
        loaded = read_proof_rollout_status(value)
        if loaded is None:
            return None
        value = loaded
    projection = _proof_rollout_projection(value)
    return (
        json.loads(json.dumps(projection, sort_keys=True))
        if projection is not None
        else None
    )


def query_proof_rollout_status(
    value: Any,
    *,
    record_types: Iterable[str] = (),
    limit: int = 50,
) -> dict[str, Any]:
    """Build a bounded, read-only query artifact from rollout diagnostics.

    ``record_types`` selects any of ``protected_scope``, ``capability``,
    ``active_plan``, ``selection``, ``decision``, ``assurance``, ``failure``,
    ``fallback``, ``override``, and ``transition``.  An empty selection means
    all record types.  The underlying proof verdict remains in the decision
    rows even when a matching override permitted the gate.
    """

    if isinstance(value, (Path, str)):
        loaded = read_proof_rollout_status(value)
        if loaded is None:
            raise ValueError("a valid proof rollout status is required")
        value = loaded
    projection = _proof_rollout_projection(value)
    if projection is None:
        raise ValueError("a valid proof rollout status is required")

    supported = (
        "protected_scope",
        "capability",
        "active_plan",
        "selection",
        "decision",
        "assurance",
        "failure",
        "fallback",
        "override",
        "transition",
    )
    raw_record_types = (
        (record_types,) if isinstance(record_types, str) else record_types
    )
    requested = {
        str(item or "").strip().lower()
        for item in raw_record_types
        if str(item or "").strip()
    }
    unknown = requested - set(supported)
    if unknown:
        raise ValueError(
            "unsupported proof rollout query record types: "
            + ", ".join(sorted(unknown))
        )
    selected = requested or set(supported)
    try:
        row_limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("proof rollout query limit must be an integer") from exc
    if isinstance(limit, bool) or row_limit < 1:
        raise ValueError("proof rollout query limit must be positive")
    row_limit = min(row_limit, MAX_PROOF_ROLLOUT_QUERY_ROWS)

    rows: list[dict[str, Any]] = []

    def append(record_type: str, record: Mapping[str, Any]) -> None:
        if record_type in selected:
            rows.append(
                {
                    "record_type": record_type,
                    "record": json.loads(json.dumps(dict(record), sort_keys=True)),
                }
            )

    for scope in projection["protected_scopes"]:
        append("protected_scope", {"scope": scope})
    for record in projection["capability_health"]:
        append("capability", record)
    for record in projection["active_plans"]:
        append("active_plan", record)
    for record in projection["selections"]:
        append("selection", record)
    for record in projection["decisions"]:
        append("decision", record)
    for assurance, count in sorted(projection["assurance_counts"].items()):
        append("assurance", {"assurance": assurance, "count": count})
    for record in projection["failures"]:
        append("failure", record)
    for fallback in projection["fallbacks"]:
        append("fallback", {"validation": fallback})
    for record in projection["overrides"]:
        append("override", record)
    for record in projection["transitions"]:
        append("transition", record)

    total_rows = len(rows)
    bounded_rows = rows[:row_limit]
    material = {
        "schema": PROOF_ROLLOUT_QUERY_SCHEMA,
        "schema_version": PROOF_ROLLOUT_QUERY_SCHEMA_VERSION,
        "status_snapshot_id": projection["snapshot_id"],
        "policy_id": projection["policy_id"],
        "rollout_mode": projection["rollout_mode"],
        "blocking": projection["blocking"],
        "mode_authority": projection["mode_authority"],
        "provider_health_can_change_mode": False,
        "record_types": sorted(selected),
        "rows": bounded_rows,
        "row_count": len(bounded_rows),
        "total_row_count": total_rows,
        "truncated": total_rows > len(bounded_rows),
        "limit": row_limit,
    }
    query_id = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**material, "query_id": query_id}


def write_proof_rollout_query(
    path: Path | str,
    status: Any,
    *,
    record_types: Iterable[str] = (),
    limit: int = 50,
) -> Path:
    """Atomically publish a portable bounded rollout query artifact."""

    artifact = query_proof_rollout_status(
        status, record_types=record_types, limit=limit
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _goal_lifecycle_state(value: Any) -> str:
    """Normalize lifecycle spelling without upgrading legacy completion.

    In particular, the historical ``completed`` label is only provisional:
    old task/board status did not prove that the completion gate passed.
    """

    raw = str(getattr(value, "value", value) or "").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "": "active",
        "open": "active",
        "ready": "active",
        "in_progress": "active",
        "complete": "provisionally_complete",
        "completed": "provisionally_complete",
        "done": "provisionally_complete",
        "provisional": "provisionally_complete",
        "verified": "verified_complete",
        "reopen": "reopened",
        "inconclusive": "analysis_inconclusive",
    }
    return aliases.get(raw, raw)


def _completion_candidates(
    value: Any,
    *,
    fallback_goal_id: str = "",
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield completion decisions from current and rollout-era containers."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _completion_candidates(item, fallback_goal_id=fallback_goal_id)
        return
    if not isinstance(value, Mapping):
        return

    goal_id = str(
        value.get("goal_id")
        or value.get("objective_goal_id")
        or value.get("goal_cid")
        or fallback_goal_id
        or ""
    ).strip()

    # Objective daemon and supervisor manifests use a goal-id keyed mapping.
    for key in (
        "objective_completion_decisions",
        "goal_completion_decisions",
        "completion_decisions",
    ):
        decisions = value.get(key)
        if isinstance(decisions, Mapping):
            for decision_goal_id, decision in decisions.items():
                yield from _completion_candidates(
                    decision, fallback_goal_id=str(decision_goal_id or "")
                )

    for key in (
        "objective_goal_migrations",
        "goal_migrations",
        "migration_records",
    ):
        migrations = value.get(key)
        if isinstance(migrations, Mapping):
            for migration_goal_id, record in migrations.items():
                yield from _completion_candidates(
                    record, fallback_goal_id=str(migration_goal_id or "")
                )
        elif isinstance(migrations, Sequence) and not isinstance(
            migrations, (str, bytes, bytearray)
        ):
            yield from _completion_candidates(migrations)

    # Migration records intentionally retain both the original decision and
    # a compact diagnostics view.  Merge those layers once, with the outer
    # lifecycle state authoritative, so recursion cannot later overwrite rich
    # evidence with a sparse wrapper event.
    nested_decision = value.get("completion_decision")
    nested_diagnostics = value.get("diagnostics")
    if goal_id and (
        isinstance(nested_decision, Mapping)
        or isinstance(nested_diagnostics, Mapping)
    ):
        merged: dict[str, Any] = {}
        if isinstance(nested_decision, Mapping):
            merged.update(nested_decision)
        if isinstance(nested_diagnostics, Mapping):
            merged.update(nested_diagnostics)
        for key in (
            "goal_id", "goal_cid", "legacy_state", "lifecycle_state",
            "state", "next_state", "status", "verified", "confidence",
            "reason_codes", "actionable_reasons", "reopen_reasons",
            "reopening_reasons", "stale_evidence", "observed_at",
            "evaluated_at", "timestamp",
        ):
            if key in value:
                merged[key] = value[key]
        yield goal_id, merged
        return

    # A previously projected diagnostics artifact is also a supported reader
    # input, which makes status/manifest republishing idempotent.
    for key in ("goal_completion_diagnostics", "goal_completion"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            goals = nested.get("goals")
            if isinstance(goals, Sequence) and not isinstance(
                goals, (str, bytes, bytearray)
            ):
                yield from _completion_candidates(goals)
            elif nested is not value:
                is_single_diagnostic = any(
                    marker in nested
                    for marker in (
                        "goal_id", "lifecycle_state", "state", "next_state",
                        "completion_gate", "uncovered_criteria", "missing_criteria",
                    )
                )
                if is_single_diagnostic:
                    yield from _completion_candidates(
                        nested, fallback_goal_id=goal_id
                    )
                else:
                    # Runner status payloads use a compact goal-id keyed map.
                    for diagnostic_goal_id, diagnostic in nested.items():
                        if isinstance(diagnostic, Mapping):
                            yield from _completion_candidates(
                                diagnostic,
                                fallback_goal_id=str(diagnostic_goal_id or ""),
                            )
        elif isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            yield from _completion_candidates(nested)

    for key in ("completion_decision", "decision"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            yield from _completion_candidates(
                nested, fallback_goal_id=fallback_goal_id
            )

    # Refill receipts retain daemon output under metadata during rollout.
    for key in ("metadata", "payload", "result", "scan_receipt", "receipt"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and nested is not value:
            yield from _completion_candidates(
                nested, fallback_goal_id=fallback_goal_id
            )

    kind = _event_type(value)
    has_completion_shape = any(
        key in value
        for key in (
            "lifecycle_state",
            "next_state",
            "previous_state",
            "completion_gate",
            "missing_criteria",
            "invalid_criteria",
            "uncovered_criteria",
            "reopen_reasons",
        )
    )
    has_inline_state = any(
        key in value for key in ("lifecycle_state", "state", "next_state", "status")
    )
    has_goal_event_shape = bool(
        goal_id
        and has_inline_state
        and "goal" in kind
        and any(term in kind for term in ("completion", "state", "reopen", "migration"))
    )
    if goal_id and (has_completion_shape or has_goal_event_shape):
        yield goal_id, value


def _completion_diagnostic(
    goal_id: str,
    decision: Mapping[str, Any],
    *,
    observed_at: str = "",
) -> dict[str, Any]:
    gate = _mapping(decision.get("completion_gate", decision.get("gate")))
    evaluated = _mapping(gate.get("evaluated_evidence"))
    coverage = _mapping(
        decision.get("coverage") or evaluated.get("coverage")
    )

    lifecycle_state = _goal_lifecycle_state(
        decision.get("lifecycle_state")
        or decision.get("state")
        or decision.get("next_state")
        or decision.get("status")
    )
    uncovered = _string_list(
        decision.get("uncovered_criteria")
        or decision.get("missing_criteria")
    )
    for criterion in _string_list(decision.get("invalid_criteria")):
        if criterion not in uncovered:
            uncovered.append(criterion)
    for check in gate.get("checks", ()) if isinstance(gate.get("checks"), Sequence) else ():
        if not isinstance(check, Mapping) or str(check.get("name") or "") != "mandatory_coverage":
            continue
        evidence = _mapping(check.get("evidence"))
        for criterion in _string_list(evidence.get("missing_criteria")) + _string_list(
            evidence.get("unverified_criteria")
        ):
            if criterion not in uncovered:
                uncovered.append(criterion)

    reason_codes = _string_list(
        decision.get("reason_codes")
        or gate.get("reason_codes")
        or gate.get("fail_reason_codes")
    )
    actionable = _string_list(
        decision.get("actionable_reasons") or gate.get("actionable_reasons")
    )
    stale_evidence = _diagnostic_list(decision.get("stale_evidence"))
    if not stale_evidence:
        for code in reason_codes:
            if "stale" in code.lower() or "freshness" in code.lower():
                stale_evidence.append(code)
        results = decision.get("evidence_results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                codes = _string_list(result.get("reason_codes"))
                if any("stale" in code.lower() or "freshness" in code.lower() for code in codes):
                    evidence = _mapping(result.get("evidence"))
                    identity = str(
                        evidence.get("receipt_cid")
                        or evidence.get("provenance_cid")
                        or evidence.get("acceptance_criterion")
                        or next((code for code in codes if "stale" in code.lower()), "stale_evidence")
                    ).strip()
                    if identity and identity not in stale_evidence:
                        stale_evidence.append(identity)

    analyzer_health = _mapping(
        decision.get("analyzer_health") or evaluated.get("analyzer_health")
    )
    exhaustion_quorum = _mapping(
        decision.get("exhaustion_quorum") or evaluated.get("exhaustion_quorum")
    )
    reopen_reasons = _string_list(
        decision.get("reopen_reasons")
        or decision.get("reopening_reasons")
        or decision.get("contradictions")
    )
    if lifecycle_state == "reopened" and not reopen_reasons:
        reopen_reasons = actionable or reason_codes or ["verification_invalidated"]

    confidence_value = decision.get("confidence")
    if confidence_value in (None, ""):
        confidence_value = coverage.get("confidence")
    confidence: float | None = None
    if confidence_value not in (None, "") and not isinstance(confidence_value, bool):
        try:
            confidence = min(1.0, max(0.0, float(confidence_value)))
        except (TypeError, ValueError):
            confidence = None

    return {
        "schema_version": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION,
        "goal_id": goal_id,
        "goal_cid": str(decision.get("goal_cid") or goal_id),
        "lifecycle_state": lifecycle_state,
        "status": lifecycle_state,
        "confidence": confidence,
        "confidence_reported": confidence is not None,
        "uncovered_criteria": uncovered,
        "stale_evidence": stale_evidence,
        "evidence_stale": bool(stale_evidence),
        "analyzer_health": dict(analyzer_health),
        "exhaustion_quorum": dict(exhaustion_quorum),
        "reopen_reasons": reopen_reasons,
        "reason_codes": reason_codes,
        "actionable_reasons": actionable,
        "completion_gate_passed": gate.get("passed") if "passed" in gate else None,
        "observed_at": str(
            decision.get("observed_at")
            or decision.get("evaluated_at")
            or evaluated.get("evaluated_at")
            or observed_at
            or ""
        ),
    }


def project_goal_completion_diagnostics(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a compact, fail-closed operator view of goal completion.

    The function is intentionally a projection, not another completion
    evaluator.  Missing confidence, health, quorum, or evidence stays missing
    and can therefore never be mistaken for positive proof.
    """

    values: Any = [records] if isinstance(records, Mapping) else records
    by_goal: dict[str, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        event_at = _event_time(raw)
        observed_at = event_at.isoformat() if event_at is not None else ""
        for goal_id, decision in _completion_candidates(raw):
            by_goal[goal_id] = _completion_diagnostic(
                goal_id, decision, observed_at=observed_at
            )

    goals = [by_goal[goal_id] for goal_id in sorted(by_goal)]
    state_counts: dict[str, int] = {}
    for goal in goals:
        state = str(goal["lifecycle_state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    analyzer_unhealthy = sum(
        1
        for goal in goals
        if goal["analyzer_health"]
        and str(goal["analyzer_health"].get("status") or goal["analyzer_health"].get("health") or "").lower()
        not in {"healthy", "ok", "passing"}
    )
    quorum_satisfied = sum(
        1
        for goal in goals
        if goal["exhaustion_quorum"].get("satisfied") is True
        or goal["exhaustion_quorum"].get("quorum_met") is True
    )
    return {
        "schema": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA,
        "schema_version": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at": _now_iso(now),
        "diagnostics_available": bool(goals),
        "goal_count": len(goals),
        "goals": goals,
        "by_goal_id": {goal["goal_id"]: goal for goal in goals},
        "state_counts": state_counts,
        "uncovered_criteria_count": sum(len(goal["uncovered_criteria"]) for goal in goals),
        "stale_evidence_goal_count": sum(bool(goal["stale_evidence"]) for goal in goals),
        "unknown_confidence_goal_count": sum(goal["confidence"] is None for goal in goals),
        "unhealthy_analyzer_goal_count": analyzer_unhealthy,
        "exhaustion_quorum_satisfied_goal_count": quorum_satisfied,
        "reopened_goal_count": state_counts.get("reopened", 0),
    }


goal_completion_diagnostics = project_goal_completion_diagnostics


def _legacy_goal_completion_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Synthesize explicit unknowns for identities found in a v1 snapshot."""

    goal_ids: set[str] = set()
    for collection_name in ("task_states", "metrics"):
        collection = payload.get(collection_name)
        if not isinstance(collection, Sequence) or isinstance(
            collection, (str, bytes, bytearray)
        ):
            continue
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            goal_id = str(
                item.get("goal_cid") or item.get("canonical_goal_id") or ""
            ).strip()
            if goal_id and goal_id not in {UNKNOWN_IDENTITY, "all"}:
                goal_ids.add(goal_id)
    goals = [
        {
            "schema_version": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION,
            "goal_id": goal_id,
            "goal_cid": goal_id,
            "lifecycle_state": "unknown",
            "status": "unknown",
            "confidence": None,
            "confidence_reported": False,
            "uncovered_criteria": [],
            "stale_evidence": [],
            "evidence_stale": False,
            "analyzer_health": {},
            "exhaustion_quorum": {},
            "reopen_reasons": [],
            "reason_codes": ["legacy_diagnostics_unavailable"],
            "actionable_reasons": [
                "Re-evaluate this goal with the versioned completion gate."
            ],
            "completion_gate_passed": None,
            "observed_at": "",
        }
        for goal_id in sorted(goal_ids)
    ]
    generated_at = str(payload.get("generated_at") or _now_iso())
    return {
        "schema": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA,
        "schema_version": GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "diagnostics_available": False,
        "compatibility_status": "unavailable_in_v1_source",
        "goal_count": len(goals),
        "goals": goals,
        "by_goal_id": {goal["goal_id"]: goal for goal in goals},
        "state_counts": ({"unknown": len(goals)} if goals else {}),
        "uncovered_criteria_count": 0,
        "stale_evidence_goal_count": 0,
        "unknown_confidence_goal_count": len(goals),
        "unhealthy_analyzer_goal_count": 0,
        "exhaustion_quorum_satisfied_goal_count": 0,
        "reopened_goal_count": 0,
    }


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
    last_event_sequence: int = -1


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

    @property
    def proof_rollout(self) -> Mapping[str, Any] | None:
        """Validated proof-rollout diagnostics carried by this snapshot."""

        return _proof_rollout_projection(self.payload)

    @property
    def resource_admission(self) -> Mapping[str, Any] | None:
        """Validated adaptive resource admission and backpressure metrics."""

        value = self.payload.get("resource_admission")
        return value if isinstance(value, Mapping) else None

    @property
    def adaptive_resources(self) -> Mapping[str, Any] | None:
        """Compatibility alias for :attr:`resource_admission`."""

        return self.resource_admission

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
    # Undated legacy events cannot supersede a timestamped current projection.
    # Treat them as the oldest evidence while preserving their source order.
    unique.sort(
        key=lambda item: (
            item[2] is not None,
            item[2] or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        )
    )
    explicit_completion_identities: set[str] = set()
    explicit_completion_aggregate_aliases: set[str] = set()
    for _index, event, _occurred in unique:
        if _event_type(event) not in _EXPLICIT_TASK_COMPLETION_EVENTS:
            continue
        sources = (
            (
                dict(event["identity"])
                if isinstance(event.get("identity"), Mapping)
                else {}
            ),
            event,
        )
        completion_identity = _first_text(
            sources,
            (
                "canonical_task_cid",
                "canonical_task_id",
                "canonical_task_key",
            ),
        ) or _first_text(sources, ("task_cid", "task_id", "task"))
        if not completion_identity:
            continue
        explicit_completion_identities.add(completion_identity)
        # Bundle supervisors retain the aggregate task CID in ``task_cid``
        # while an explicit member completion carries its own canonical CID.
        # Remember that relationship so an earlier aggregate merge terminal
        # is not counted as an additional completed task.
        for source in sources:
            aggregate_identity = str(source.get("task_cid") or "")
            if aggregate_identity and aggregate_identity != completion_identity:
                explicit_completion_aggregate_aliases.add(aggregate_identity)
    legacy_completion_identities: set[str] = set()
    scan_metrics = _reduce_scan_metrics(unique)
    completion_diagnostics = project_goal_completion_diagnostics(
        [event for _index, event, _occurred in unique], now=now
    )
    resource_admission = project_resource_admission_metrics(
        [event for _index, event, _occurred in unique]
    )
    rollout_projection: dict[str, Any] | None = None
    for _index, event, _occurred in unique:
        candidate = _proof_rollout_projection(event)
        if candidate is None and _declares_proof_rollout(event):
            raise ValueError(
                "event contains an invalid proof rollout status projection"
            )
        if candidate is not None:
            rollout_projection = candidate
    diagnostics_by_goal: dict[str, Mapping[str, Any]] = {}
    for goal_id, diagnostic in completion_diagnostics["by_goal_id"].items():
        diagnostics_by_goal[str(goal_id)] = diagnostic
        goal_cid = str(diagnostic.get("goal_cid") or "")
        if goal_cid:
            diagnostics_by_goal[goal_cid] = diagnostic

    accumulators: dict[tuple[str, ...], _Accumulator] = {}
    inherited_by_task: dict[str, dict[str, str]] = {}
    inherited_by_lane: dict[str, dict[str, str]] = {}
    latest_projection: tuple[int, str] | None = None
    for event_sequence, (_index, event, _occurred) in enumerate(unique):
        if (
            _event_type(event) == _SCHEDULER_STATE_PROJECTION_EVENT
            and str(event.get("scheduler_projection_scope") or "")
            == _SCHEDULER_STATE_PROJECTION_SCOPE
        ):
            latest_projection = (
                event_sequence,
                str(event.get("scheduler_projection_id") or ""),
            )
    authoritative_state_keys: set[tuple[str, ...]] = set()

    for event_sequence, (_index, event, occurred) in enumerate(unique):
        kind = _event_type(event)
        if kind == _SCHEDULER_STATE_PROJECTION_EVENT:
            # This envelope declares the complete current task set. It is
            # supervisor evidence and must never manufacture an anonymous
            # scheduler task or contribute to cumulative task metrics.
            continue
        if kind in _REFILL_SCAN_EVENT_TYPES or _scan_receipt_projection(event, kind) is not None:
            # A repository scan is supervisor-level evidence, not a scheduler
            # task.  It contributes to scan_metrics but must not manufacture an
            # ``unknown`` idle task or alter a real task's current phase.
            continue
        if _declares_resource_admission(event) and (
            kind in RESOURCE_ADMISSION_EVENT_TYPES
            or not any(
                event.get(key)
                for key in (
                    "task_cid",
                    "canonical_task_cid",
                    "canonical_task_key",
                    "task_id",
                    "lane_id",
                    "parallel_lane",
                    "bundle_key",
                    "state_prefix",
                )
            )
        ):
            # Resource observations are supervisor-wide gauges, not lifecycle
            # state. They must not create an anonymous scheduler task.
            continue
        if (
            _proof_rollout_projection(event) is not None
            and not any(
                event.get(key)
                for key in (
                    "task_cid",
                    "canonical_task_cid",
                    "canonical_task_key",
                    "task_id",
                    "lane_id",
                    "parallel_lane",
                    "bundle_key",
                    "state_prefix",
                )
            )
        ):
            # A supervisor heartbeat containing only rollout diagnostics is
            # status evidence, not a scheduler task.  It must not create an
            # ``unknown`` idle row.
            continue
        task_alias = str(
            event.get("task_cid") or event.get("canonical_task_cid")
            or event.get("canonical_task_key") or event.get("task_id") or ""
        )
        lane_alias = str(
            event.get("lane_id") or event.get("parallel_lane")
            or event.get("bundle_key") or event.get("state_prefix") or ""
        )
        if not task_alias and not lane_alias and next(
            _completion_candidates(event), None
        ) is not None:
            # A goal lifecycle decision is supervisor state, not a scheduler
            # task, so it must not manufacture an ``unknown`` task row.
            continue
        inherited = inherited_by_task.get(task_alias) or inherited_by_lane.get(lane_alias) or {}
        identity = normalize_metric_identity(event, {**dict(defaults or {}), **inherited})
        if task_alias:
            inherited_by_task[task_alias] = identity
        if lane_alias:
            inherited_by_lane[lane_alias] = identity
        key = _identity_key(identity)
        if (
            latest_projection is not None
            and event_sequence > latest_projection[0]
            and str(event.get("scheduler_projection_id") or "")
            == latest_projection[1]
            and str(event.get("scheduler_projection_scope") or "")
            == _SCHEDULER_STATE_PROJECTION_SCOPE
            and kind in {"scheduler_state", "scheduler_lane_state"}
        ):
            authoritative_state_keys.add(_task_state_key(identity, key))
        current = accumulators.get(key)
        if current is None:
            current = _Accumulator(identity=identity, metrics=_metric_defaults(identity))
            accumulators[key] = current
        completed_before_event = current.completed

        current.last_event_type = kind
        current.last_event_sequence = event_sequence
        if occurred is not None:
            current.last_event_at = occurred.isoformat()
        current.display_task_id = str(event.get("task_id") or current.display_task_id)
        tokens, cost = _usage(event)
        current.metrics["tokens"] += tokens
        current.metrics["cost_usd"] += cost
        # Proof/cache/resource integrations may publish already-measured
        # latencies.  Consume only explicit non-negative measurements here;
        # absent timestamp pairs must not produce invented durations.
        for category in (
            "queue",
            "solver",
            "kernel",
            "model",
            "validation",
            "merge",
            "cancellation",
            "cache",
        ):
            milliseconds_name = f"{category}_latency_ms"
            seconds_name = f"{category}_latency_seconds"
            milliseconds = _number(event.get(milliseconds_name))
            seconds = _number(event.get(seconds_name))
            if milliseconds:
                current.metrics[milliseconds_name] += int(milliseconds)
                current.metrics[seconds_name] += milliseconds / 1000.0
            elif seconds:
                current.metrics[seconds_name] += seconds
                current.metrics[milliseconds_name] += int(round(seconds * 1000.0))
        if "cancel" in kind:
            current.metrics["cancellations"] += 1

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

        if (
            current.completed
            and not completed_before_event
            and kind not in _EXPLICIT_TASK_COMPLETION_EVENTS
        ):
            event_identity = (
                dict(event["identity"])
                if isinstance(event.get("identity"), Mapping)
                else {}
            )
            legacy_identity = _first_text(
                (event_identity, event),
                (
                    "canonical_task_cid",
                    "canonical_task_id",
                    "canonical_task_key",
                ),
            ) or identity["task_cid"]
            if legacy_identity != UNKNOWN_IDENTITY:
                legacy_completion_identities.add(legacy_identity)

        if kind in _RESOLVER_EVENTS or phase == "resolver":
            current.phase = "resolver"

        if kind in _BLOCKED_EVENTS:
            current.phase = "blocked"
            current.status = "blocked"

    rows: list[dict[str, Any]] = []
    state_candidates: dict[
        tuple[str, ...],
        tuple[int, dict[str, Any]],
    ] = {}
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
        # General scheduler lifecycle timings are authoritative aliases for
        # the corresponding proof-wide categories when no specialized sample
        # was emitted.
        if not metrics["queue_latency_ms"]:
            metrics["queue_latency_seconds"] = metrics["queue_wait_seconds"]
            metrics["queue_latency_ms"] = int(
                round(metrics["queue_wait_seconds"] * 1000.0)
            )
        if not metrics["validation_latency_ms"]:
            metrics["validation_latency_seconds"] = metrics[
                "validation_duration_seconds"
            ]
            metrics["validation_latency_ms"] = int(
                round(metrics["validation_duration_seconds"] * 1000.0)
            )
        if not metrics["merge_latency_ms"]:
            metrics["merge_latency_seconds"] = metrics["merge_wait_seconds"]
            metrics["merge_latency_ms"] = int(
                round(metrics["merge_wait_seconds"] * 1000.0)
            )
        rows.append(dict(metrics))
        state = {
            **current.identity,
            "task_id": current.display_task_id,
            "phase": current.phase,
            "status": current.status,
            "last_event_type": current.last_event_type,
            "last_event_at": current.last_event_at,
        }
        goal_diagnostic = diagnostics_by_goal.get(current.identity["goal_cid"])
        if goal_diagnostic is not None:
            # Additive nested data leaves all v1 task-state keys intact.
            state["goal_completion"] = dict(goal_diagnostic)
        state_key = _task_state_key(current.identity, key)
        if latest_projection is not None and state_key not in authoritative_state_keys:
            continue
        previous = state_candidates.get(state_key)
        candidate = (current.last_event_sequence, state)
        if previous is None or candidate[0] > previous[0]:
            state_candidates[state_key] = candidate

    # Metrics remain dimensioned by provider/tree/template/resource class, but
    # current scheduler phase is a task-level gauge.  Select the latest state
    # for each canonical task so a terminal lease projection supersedes active
    # history emitted under an earlier provider or lane identity.
    task_states = sorted(
        (candidate[1] for candidate in state_candidates.values()),
        key=_identity_key,
    )
    phase_items: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in SCHEDULER_PHASES
    }
    for state in task_states:
        phase_items[state["phase"]].append(state)

    dimensions_all = normalize_metric_identity({}, {
        "goal_cid": "all", "subgoal_cid": "all", "task_cid": "all",
        "lane_id": "all", "provider_id": "all", "repository_tree_id": "all",
        "template_id": "all", "resource_class": "all",
    })
    totals = _metric_defaults(dimensions_all)
    for row in rows:
        for name in (
            "queue_wait_seconds", "implementation_duration_seconds",
            "validation_duration_seconds", "merge_wait_seconds", "cost_usd",
            "queue_latency_seconds", "solver_latency_seconds",
            "kernel_latency_seconds", "model_latency_seconds",
            "validation_latency_seconds", "merge_latency_seconds",
            "cancellation_latency_seconds", "cache_latency_seconds",
        ):
            totals[name] += float(row[name])
        for name in (
            "implementation_attempts", "merge_attempts", "conflicts", "retries",
            "completions", "tokens", "cancellations",
            "queue_latency_ms", "solver_latency_ms", "kernel_latency_ms",
            "model_latency_ms", "validation_latency_ms", "merge_latency_ms",
            "cancellation_latency_ms", "cache_latency_ms",
        ):
            totals[name] += int(row[name])
    totals["conflict_rate"] = (
        totals["conflicts"] / totals["merge_attempts"] if totals["merge_attempts"] else 0.0
    )
    totals["retry_rate"] = (
        totals["retries"] / totals["implementation_attempts"]
        if totals["implementation_attempts"] else 0.0
    )
    # ``completions`` retains the dimensioned lifecycle throughput used by
    # existing dashboards.  Bundle logs can repeat one terminal task in more
    # than one lane, or group several member terminals under one aggregate
    # task CID, so the operator-facing count is instead the exact number of
    # unique explicit terminal task identities when that evidence exists.
    # Distinct legacy terminals remain part of a mixed event stream, while a
    # bundle-level legacy terminal linked to explicit member receipts does not
    # become a phantom extra completion.
    totals["completion_count"] = (
        len(explicit_completion_identities)
        + len(
            legacy_completion_identities
            - explicit_completion_identities
            - explicit_completion_aggregate_aliases
        )
        if explicit_completion_identities
        else totals["completions"]
    )
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
        "goal_completion_diagnostics": {
            key: value
            for key, value in completion_diagnostics.items()
            if key != "generated_at"
        },
        "proof_rollout": rollout_projection,
        "resource_admission": resource_admission,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": SCHEDULER_SNAPSHOT_SCHEMA,
        "schema_version": SCHEDULER_SNAPSHOT_SCHEMA_VERSION,
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
        # The short alias is convenient for status pages; the explicit name
        # is used by persisted manifests.  Both are the same versioned view.
        "goal_completion": completion_diagnostics,
        "goal_completion_diagnostics": completion_diagnostics,
        "proof_rollout": rollout_projection,
        "proof_rollout_diagnostics": rollout_projection,
        "resource_admission": resource_admission,
        "adaptive_resources": resource_admission,
    }
    if rollout_projection is not None:
        capabilities = rollout_projection["capability_health"]
        payload.update(
            {
                "proof_policy_id": rollout_projection["policy_id"],
                "proof_rollout_mode": rollout_projection["rollout_mode"],
                "proof_rollout_blocking": rollout_projection["blocking"],
                "proof_capability_healthy": bool(capabilities)
                and all(bool(item["healthy"]) for item in capabilities),
                "proof_active_plan_count": len(
                    rollout_projection["active_plans"]
                ),
                "proof_override_count": len(rollout_projection["overrides"]),
                "proof_failure_count": len(rollout_projection["failures"]),
            }
        )
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


def write_proof_rollout_status(path: Path | str, status: Any) -> Path:
    """Atomically publish a validated, bounded rollout status artifact."""

    projection = _proof_rollout_projection(status)
    if projection is None:
        raise ValueError("a valid proof rollout status is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def read_proof_rollout_status(path: Path | str) -> ProofRolloutStatus | None:
    """Read a rollout status, rejecting malformed or private projections."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    projection = _proof_rollout_projection(payload)
    if projection is None:
        return None
    return ProofRolloutStatus(projection)


publish_proof_rollout_status = write_proof_rollout_status
query_proof_rollout_diagnostics = query_proof_rollout_status


def read_scheduler_snapshot(path: Path | str) -> SchedulerSnapshot | None:
    """Read current or v1 snapshots, returning ``None`` for invalid files.

    Compatibility defaults are additive.  The original schema identifier is
    preserved so automation can still distinguish an artifact produced by an
    older supervisor during a rolling upgrade.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") not in {
        SCHEDULER_SNAPSHOT_SCHEMA,
        *LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS,
    }:
        return None
    schema = str(payload.get("schema") or "")
    payload.setdefault(
        "schema_version",
        1 if schema in LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS else SCHEDULER_SNAPSHOT_SCHEMA_VERSION,
    )
    empty_diagnostics = (
        _legacy_goal_completion_diagnostics(payload)
        if schema in LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS
        else project_goal_completion_diagnostics((), now=payload.get("generated_at"))
    )
    diagnostics = payload.get("goal_completion_diagnostics")
    if not isinstance(diagnostics, Mapping):
        alias = payload.get("goal_completion")
        diagnostics = alias if isinstance(alias, Mapping) else empty_diagnostics
    payload.setdefault("goal_completion", diagnostics)
    payload.setdefault("goal_completion_diagnostics", diagnostics)

    raw_resource_admission = payload.get("resource_admission")
    if raw_resource_admission is None:
        raw_resource_admission = payload.get("adaptive_resources")
    resource_admission = (
        _resource_admission_projection(raw_resource_admission)
        if raw_resource_admission is not None
        else None
    )
    if raw_resource_admission is not None and resource_admission is None:
        return None
    payload["resource_admission"] = resource_admission
    payload["adaptive_resources"] = resource_admission

    raw_rollout = payload.get("proof_rollout")
    if raw_rollout is None:
        raw_rollout = payload.get("proof_rollout_diagnostics")
    rollout = (
        _proof_rollout_projection(raw_rollout)
        if raw_rollout is not None
        else None
    )
    if raw_rollout is not None and rollout is None:
        # A malformed rollout section is security-relevant.  Do not return a
        # snapshot that appears healthy after silently dropping it.
        return None
    if rollout is None and any(
        key in payload
        for key in (
            "proof_policy_id",
            "proof_rollout_mode",
            "proof_rollout_blocking",
        )
    ):
        # Flat compatibility summaries are never sufficient to establish
        # policy identity or mode authority on their own.
        return None
    payload["proof_rollout"] = rollout
    payload["proof_rollout_diagnostics"] = rollout
    if rollout is not None:
        capabilities = rollout["capability_health"]
        payload.update(
            {
                "proof_policy_id": rollout["policy_id"],
                "proof_rollout_mode": rollout["rollout_mode"],
                "proof_rollout_blocking": rollout["blocking"],
                "proof_capability_healthy": bool(capabilities)
                and all(bool(item["healthy"]) for item in capabilities),
                "proof_active_plan_count": len(rollout["active_plans"]),
                "proof_override_count": len(rollout["overrides"]),
                "proof_failure_count": len(rollout["failures"]),
            }
        )
    return SchedulerSnapshot(payload)


def scheduler_state_events(
    tasks: Iterable[Mapping[str, Any]],
    *,
    lanes: Iterable[Mapping[str, Any]] = (),
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Project lease/lane state into lifecycle events for the shared reducer."""

    occurred_at = timestamp or _now_iso()
    projection_id = occurred_at
    projection_metadata = {
        "scheduler_projection_id": projection_id,
        "scheduler_projection_scope": _SCHEDULER_STATE_PROJECTION_SCOPE,
    }
    events: list[dict[str, Any]] = [
        {
            "type": _SCHEDULER_STATE_PROJECTION_EVENT,
            "timestamp": occurred_at,
            **projection_metadata,
        }
    ]
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
        events.append(
            {
                **task,
                "type": "scheduler_state",
                "timestamp": occurred_at,
                "phase": state,
                **projection_metadata,
            }
        )
    for raw in lanes:
        lane = dict(raw)
        phase = str(lane.get("phase") or lane.get("active_phase") or lane.get("state") or "active").lower()
        events.append(
            {
                **lane,
                "type": "scheduler_lane_state",
                "timestamp": occurred_at,
                "phase": phase,
                **projection_metadata,
            }
        )
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


# ---------------------------------------------------------------------------
# ASI-169 event-derived usage-governance metrics
# ---------------------------------------------------------------------------

SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID = (
    "requirement:supervisor-usage-metrics.v1"
)
SUPERVISOR_USAGE_METRICS_SCHEMA_VERSION = 1
SUPERVISOR_USAGE_METRICS_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.usage-governance-metrics@1"
)
MAX_USAGE_METRIC_SERIES = 4_096
MAX_USAGE_PROVIDER_LABELS = 64
MAX_USAGE_DEPLOYMENT_LABELS = 128
MAX_USAGE_STAGE_LABELS = 32

USAGE_GOVERNANCE_EVENT_TYPES = frozenset(
    {
        "usage_reservation",
        "usage_reservation_denied",
        "usage_reservation_expired",
        "usage_settlement",
        "usage_correction",
        "usage_reset",
        "usage_wait",
        "usage_reroute",
        "usage_fallback",
        "usage_estimate",
        "usage_headroom",
        "usage_fairness",
        "usage_starvation",
        "usage_herd",
        "usage_ledger_health",
        "endpoint_usage_event",
        "provider_usage_event",
        "supervisor_usage_event",
    }
)

_USAGE_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "request",
        "request_id",
        "credential",
        "credential_id",
        "credential_pseudonym",
        "tenant",
        "tenant_id",
        "alias",
        "model",
        "model_id",
        "model_string",
        "model_alias",
        "endpoint",
        "endpoint_uri",
        "endpoint_url",
        "url",
        "account",
        "account_pseudonym",
        "user",
        "session",
        "prompt",
        "media",
        "output",
    }
)

_USAGE_METRIC_LABELS: dict[str, tuple[str, ...]] = {
    "usage_estimate_error_ratio_sum": ("provider", "deployment", "stage", "dimension"),
    "usage_estimate_error_ratio_count": (
        "provider",
        "deployment",
        "stage",
        "dimension",
    ),
    "usage_headroom_band": ("provider", "deployment", "stage", "dimension", "band"),
    "usage_denials_total": ("provider", "deployment", "stage", "reason"),
    "usage_waits_total": ("provider", "deployment", "stage", "reason"),
    "usage_reroutes_total": ("provider", "deployment", "stage", "reason"),
    "usage_fairness_total": ("provider", "deployment", "stage", "state"),
    "usage_starvation_total": ("provider", "deployment", "stage", "reason"),
    "usage_resets_total": ("provider", "deployment", "stage", "reason"),
    "usage_herd_total": ("provider", "deployment", "stage", "reason"),
    "usage_fallbacks_total": ("provider", "deployment", "stage", "reason"),
    "usage_settlements_total": ("provider", "deployment", "stage", "state"),
    "usage_corrections_total": ("provider", "deployment", "stage", "reason"),
    "usage_ledger_health": ("state",),
}

_USAGE_BOUNDED_VALUES: dict[str, frozenset[str]] = {
    "reason": frozenset(
        {
            "ok",
            "limit_exhausted",
            "capacity_unavailable",
            "policy_denied",
            "stale_snapshot",
            "reservation_conflict",
            "cooling_down",
            "timeout",
            "cancelled",
            "fallback",
            "reroute",
            "wait",
            "correction",
            "reset",
            "herd",
            "starvation",
            "fairness",
            "store_unhealthy",
            "unknown",
            "other",
        }
    ),
    "dimension": frozenset(
        {
            "requests",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cost_micros",
            "concurrency",
            "other",
        }
    ),
    "band": frozenset(
        {
            "unknown",
            "exhausted",
            "critical",
            "low",
            "medium",
            "high",
            "unlimited",
            "other",
        }
    ),
    "state": frozenset(
        {
            "healthy",
            "unhealthy",
            "ready",
            "exhausted",
            "stale",
            "unknown",
            "settled",
            "corrected",
            "other",
        }
    ),
    "stage": frozenset(
        {
            "planning",
            "analysis",
            "proof",
            "rescue",
            "validation",
            "implementation",
            "merge",
            "batch",
            "other",
        }
    ),
}


def forbidden_usage_metric_label_keys() -> frozenset[str]:
    return frozenset(_USAGE_FORBIDDEN_LABEL_KEYS)


def _usage_label_token(
    value: Any,
    *,
    allowed: frozenset[str] | None = None,
    default: str = "other",
) -> str:
    if value is None:
        return default
    text = str(value).strip().casefold()
    if not text:
        return default
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._:-" else "-" for ch in text
    ).strip("-")[:64]
    if not cleaned:
        return default
    if allowed is not None and cleaned not in allowed:
        return default if default in allowed else "other"
    return cleaned


def _usage_provider_label(value: Any) -> str:
    text = _usage_label_token(value, default="other")
    if text.startswith("provider:") or text == "other":
        return text
    return f"provider:{text}" if text != "other" else "other"


def _usage_deployment_label(value: Any) -> str:
    text = _usage_label_token(value, default="other")
    if text.startswith("deployment:") or text == "other":
        return text
    return f"deployment:{text}" if text != "other" else "other"


def _usage_event_kind(event: Mapping[str, Any]) -> str:
    for key in ("type", "kind", "event_type", "usage_event_kind"):
        value = event.get(key)
        if value:
            return str(value).strip().casefold()
    return ""


def _declares_usage_governance(event: Mapping[str, Any]) -> bool:
    kind = _usage_event_kind(event)
    if kind in USAGE_GOVERNANCE_EVENT_TYPES:
        return True
    if event.get("usage_governance") is True:
        return True
    if isinstance(event.get("usage"), Mapping):
        return True
    if any(
        key in event
        for key in (
            "headroom_band",
            "estimate_error",
            "reservation_denied",
            "ledger_health",
            "fairness_state",
            "starvation",
            "reset_herd",
        )
    ):
        return True
    return False


def _usage_identity(event: Mapping[str, Any]) -> dict[str, str]:
    usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
    provider = (
        event.get("provider")
        or event.get("provider_id")
        or usage.get("provider")
    )
    deployment = (
        event.get("deployment")
        or event.get("deployment_id")
        or usage.get("deployment")
    )
    stage = (
        event.get("stage")
        or event.get("supervisor_stage")
        or usage.get("stage")
    )
    return {
        "provider": _usage_provider_label(provider),
        "deployment": _usage_deployment_label(deployment),
        "stage": _usage_label_token(
            stage, allowed=_USAGE_BOUNDED_VALUES["stage"], default="other"
        ),
    }


def _usage_series_key(
    name: str, labels: Mapping[str, str]
) -> tuple[str, tuple[tuple[str, str], ...]]:
    expected = _USAGE_METRIC_LABELS[name]
    for key in labels:
        if str(key).casefold() in _USAGE_FORBIDDEN_LABEL_KEYS:
            raise ValueError(f"forbidden metric label: {key}")
    if set(labels) != set(expected):
        raise ValueError("metric labels must exactly match the metric contract")
    ordered = tuple((key, str(labels[key])) for key in expected)
    return name, ordered


def project_usage_governance_metrics(
    events: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Derive low-cardinality usage-governance metrics from endpoint events.

    Metrics are operational evidence only.  Labels are bounded to provider,
    deployment, stage, state, reason, dimension, and headroom band — never
    request, credential, tenant, prompt, media, output, model alias, or
    endpoint URL cardinality.
    """

    if isinstance(events, Mapping):
        values = [dict(events)]
    else:
        values = [dict(value) for value in events if isinstance(value, Mapping)]

    series: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    providers: set[str] = set()
    deployments: set[str] = set()
    stages: set[str] = set()

    def bump(name: str, labels: Mapping[str, str], amount: float = 1.0) -> None:
        if name not in _USAGE_METRIC_LABELS:
            return
        if amount < 0:
            return
        key = _usage_series_key(name, labels)
        if key[1][0][0] == "provider":
            providers.add(key[1][0][1])
        for label_name, label_value in key[1]:
            if label_name == "deployment":
                deployments.add(label_value)
            if label_name == "stage":
                stages.add(label_value)
        if key not in series and len(series) >= MAX_USAGE_METRIC_SERIES:
            return
        series[key] = series.get(key, 0.0) + float(amount)

    ledger_healthy = True
    for event in values:
        if not _declares_usage_governance(event):
            continue
        identity = _usage_identity(event)
        # Reject forbidden high-cardinality fields if present as metric labels.
        for forbidden in _USAGE_FORBIDDEN_LABEL_KEYS:
            if forbidden in event and forbidden in {
                "request_id",
                "credential_pseudonym",
                "tenant_id",
                "endpoint_url",
                "prompt",
            }:
                # Ignore payload presence; never promote to labels.
                pass
        kind = _usage_event_kind(event)
        reason = _usage_label_token(
            (event.get("reason_codes") or [event.get("reason") or "ok"])[0]
            if isinstance(event.get("reason_codes"), (list, tuple))
            else event.get("reason") or "ok",
            allowed=_USAGE_BOUNDED_VALUES["reason"],
            default="other",
        )
        dimension = _usage_label_token(
            event.get("dimension") or "total_tokens",
            allowed=_USAGE_BOUNDED_VALUES["dimension"],
            default="other",
        )
        if kind in {
            "usage_reservation_denied",
            "reservation_denied",
        } or event.get("reservation_denied"):
            bump(
                "usage_denials_total",
                {
                    **identity,
                    "reason": reason if reason != "ok" else "limit_exhausted",
                },
            )
        if kind in {"usage_wait", "wait"} or event.get("wait"):
            bump("usage_waits_total", {**identity, "reason": reason})
        if kind in {"usage_reroute", "reroute"} or event.get("reroute"):
            bump("usage_reroutes_total", {**identity, "reason": reason})
        if kind in {"usage_fallback", "fallback"} or event.get("fallback"):
            bump("usage_fallbacks_total", {**identity, "reason": reason})
        if kind in {"usage_reset", "reset"} or event.get("reset"):
            bump("usage_resets_total", {**identity, "reason": "reset"})
        if kind in {"usage_herd", "reset_herd"} or event.get("reset_herd"):
            bump("usage_herd_total", {**identity, "reason": "herd"})
        if kind in {"usage_starvation", "starvation"} or event.get("starvation"):
            bump("usage_starvation_total", {**identity, "reason": "starvation"})
        if kind in {"usage_fairness", "fairness"} or event.get("fairness_state"):
            state = _usage_label_token(
                event.get("fairness_state") or event.get("state") or "ready",
                allowed=_USAGE_BOUNDED_VALUES["state"],
                default="other",
            )
            bump("usage_fairness_total", {**identity, "state": state})
        if kind in {"usage_settlement", "settlement"} or event.get("settlement"):
            bump(
                "usage_settlements_total",
                {**identity, "state": "settled"},
            )
        if kind in {"usage_correction", "correction"} or event.get("correction"):
            bump("usage_corrections_total", {**identity, "reason": "correction"})
        if kind in {"usage_estimate", "estimate"} or "estimate_error" in event:
            raw_estimated = event.get("estimated", event.get("estimate"))
            raw_actual = event.get("actual", event.get("observed"))
            if raw_estimated not in (None, "") and raw_actual not in (None, ""):
                estimated = _number(raw_estimated)
                actual = _number(raw_actual)
                denom = max(1.0, abs(float(actual)))
                ratio = abs(float(estimated) - float(actual)) / denom
                labels = {**identity, "dimension": dimension}
                bump("usage_estimate_error_ratio_sum", labels, ratio)
                bump("usage_estimate_error_ratio_count", labels, 1.0)
        band = event.get("headroom_band") or event.get("band")
        if band or kind in {"usage_headroom", "headroom"}:
            bump(
                "usage_headroom_band",
                {
                    **identity,
                    "dimension": dimension,
                    "band": _usage_label_token(
                        band or "unknown",
                        allowed=_USAGE_BOUNDED_VALUES["band"],
                        default="other",
                    ),
                },
                1.0,
            )
        if kind in {"usage_ledger_health", "ledger_health"} or "ledger_health" in event:
            health = event.get("ledger_health")
            if health in {False, "unhealthy", "error"}:
                ledger_healthy = False
            bump(
                "usage_ledger_health",
                {
                    "state": "healthy"
                    if health not in {False, "unhealthy", "error"}
                    else "unhealthy"
                },
                1.0,
            )

    if "usage_ledger_health" not in {key[0] for key in series}:
        bump(
            "usage_ledger_health",
            {"state": "healthy" if ledger_healthy else "unhealthy"},
            1.0,
        )

    # Cap provider/deployment/stage cardinality for the projection summary.
    samples = [
        {
            "name": name,
            "labels": dict(labels),
            "value": value,
        }
        for (name, labels), value in sorted(
            series.items(), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    return {
        "schema": SUPERVISOR_USAGE_METRICS_SCHEMA,
        "schema_version": SUPERVISOR_USAGE_METRICS_SCHEMA_VERSION,
        "requirement_id": SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID,
        "observed_at": _now_iso(now),
        "series_count": len(samples),
        "samples": samples,
        "provider_count": min(len(providers), MAX_USAGE_PROVIDER_LABELS),
        "deployment_count": min(len(deployments), MAX_USAGE_DEPLOYMENT_LABELS),
        "stage_count": min(len(stages), MAX_USAGE_STAGE_LABELS),
        "forbidden_label_keys": sorted(_USAGE_FORBIDDEN_LABEL_KEYS),
        "completion_authoritative": False,
        "operational_evidence_only": True,
        "metric_names": sorted(_USAGE_METRIC_LABELS),
    }


# ---------------------------------------------------------------------------
# Benchmark causal-span joins (PDR-071)
# ---------------------------------------------------------------------------
#
# Capacity admission and lifecycle reduction remain authoritative above.
# These helpers only project existing snapshot dimensions onto benchmark
# causal spans; they never invent measured zeros for missing sensors.

BENCHMARK_SPAN_CLOCK_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.benchmark-span-clock-metrics@1"
)
BENCHMARK_SPAN_CLOCK_SCHEMA_VERSION = 1


def bind_metric_row_to_span_identity(
    row: Mapping[str, Any],
    *,
    span_id: str,
    run_id: str = "",
    case_id: str = "",
    arm_id: str = "",
    attempt: int = 0,
    process_id: str = "",
) -> dict[str, Any]:
    """Attach causal-span ancestry fields to one scheduler metric row.

    The original counters are preserved; span fields are additive so existing
    gauge consumers continue to read the same keys.
    """

    if not isinstance(row, Mapping):
        raise TypeError("metric row must be a mapping")
    bound = dict(row)
    bound["span_id"] = str(span_id or "")
    if run_id:
        bound["run_id"] = str(run_id)
    if case_id:
        bound["case_id"] = str(case_id)
    if arm_id:
        bound["arm_id"] = str(arm_id)
    if attempt:
        bound["attempt"] = int(attempt)
    if process_id:
        bound["process_id"] = str(process_id)
    return bound


def project_snapshot_metrics_for_span(
    snapshot: SchedulerSnapshot | Mapping[str, Any],
    *,
    span_id: str,
    task_id: str = "",
    task_cid: str = "",
    run_id: str = "",
    case_id: str = "",
    arm_id: str = "",
    attempt: int = 0,
    process_id: str = "",
) -> dict[str, Any]:
    """Project queue/merge/implementation waits for one causal span.

    Returns a content-stable dictionary of clock dimensions joined by span
    identity.  Missing per-task rows produce explicit nulls rather than
    fabricated zeros so benchmark telemetry can emit ``unavailable`` samples.
    """

    payload = dict(snapshot) if not isinstance(snapshot, Mapping) else snapshot
    if not isinstance(payload, Mapping):
        raise TypeError("snapshot must be a mapping")
    metrics = list(payload.get("metrics") or [])
    target_task = str(task_cid or task_id or "")
    matched: list[Mapping[str, Any]] = []
    for row in metrics:
        if not isinstance(row, Mapping):
            continue
        row_task = str(row.get("task_cid") or row.get("task_id") or "")
        if target_task and row_task and target_task not in (
            row_task,
            f"task:{row_task}",
        ):
            if not (
                row_task.endswith(target_task) or target_task.endswith(row_task)
            ):
                continue
        matched.append(row)

    def _sum(name: str) -> float | None:
        if not matched:
            return None
        total = 0.0
        saw = False
        for row in matched:
            if name not in row:
                continue
            total += _number(row.get(name))
            saw = True
        return total if saw else None

    queue_wait = _sum("queue_wait_seconds")
    implementation = _sum("implementation_duration_seconds")
    validation = _sum("validation_duration_seconds")
    merge_wait = _sum("merge_wait_seconds")
    makespan: float | None
    if any(value is not None for value in (queue_wait, implementation, validation, merge_wait)):
        makespan = (
            (queue_wait or 0.0)
            + (implementation or 0.0)
            + (validation or 0.0)
            + (merge_wait or 0.0)
        )
    else:
        makespan = None

    critical_path: float | None = None
    if matched:
        path_values = []
        for row in matched:
            if (
                "implementation_duration_seconds" not in row
                and "validation_duration_seconds" not in row
            ):
                continue
            path_values.append(
                _number(row.get("implementation_duration_seconds"))
                + _number(row.get("validation_duration_seconds"))
            )
        if path_values:
            critical_path = max(path_values)

    phase_counts = payload.get("phase_counts") or {}
    ready_width = None
    observed_width = None
    if isinstance(phase_counts, Mapping):
        ready_width = int(phase_counts.get("ready") or 0)
        observed_width = int(phase_counts.get("active") or 0)

    rows = [
        bind_metric_row_to_span_identity(
            dict(row),
            span_id=span_id,
            run_id=run_id,
            case_id=case_id,
            arm_id=arm_id,
            attempt=attempt,
            process_id=process_id,
        )
        for row in matched
    ]
    return {
        "schema": BENCHMARK_SPAN_CLOCK_SCHEMA,
        "schema_version": BENCHMARK_SPAN_CLOCK_SCHEMA_VERSION,
        "span_id": str(span_id),
        "run_id": str(run_id or ""),
        "case_id": str(case_id or ""),
        "arm_id": str(arm_id or ""),
        "task_id": str(task_id or target_task or ""),
        "attempt": int(attempt or 0),
        "process_id": str(process_id or ""),
        "matched_row_count": len(matched),
        "queue_wait_seconds": queue_wait,
        "implementation_duration_seconds": implementation,
        "validation_duration_seconds": validation,
        "merge_wait_seconds": merge_wait,
        "end_to_end_makespan_seconds": makespan,
        "critical_path_seconds": critical_path,
        "ready_width": ready_width,
        "observed_width": observed_width,
        "admitted_width": payload.get("admitted_width"),
        "rows": rows,
        # Nulls above must be projected as unavailable by benchmark_telemetry;
        # this join never rewrites them into measured zeros.
        "missing_dimensions_are_null": True,
        "capacity_admission_unchanged": True,
    }


def join_scheduler_snapshot_to_benchmark_span(
    snapshot: SchedulerSnapshot | Mapping[str, Any],
    span: Any,
) -> dict[str, Any]:
    """Join a scheduler snapshot to a ``BenchmarkCausalSpan``-like object."""

    span_id = str(getattr(span, "span_id", "") or "")
    if not span_id and isinstance(span, Mapping):
        span_id = str(span.get("span_id") or "")
    if not span_id:
        raise ValueError("span must expose span_id")

    def _field(name: str, default: Any = "") -> Any:
        if hasattr(span, name):
            return getattr(span, name)
        if isinstance(span, Mapping):
            return span.get(name, default)
        return default

    return project_snapshot_metrics_for_span(
        snapshot,
        span_id=span_id,
        task_id=str(_field("task_id", "") or ""),
        run_id=str(_field("run_id", "") or ""),
        case_id=str(_field("case_id", "") or ""),
        arm_id=str(_field("arm_id", "") or ""),
        attempt=int(_field("attempt", 0) or 0),
        process_id=str(_field("process_id", "") or ""),
    )


# Proof observability uses the same operator-facing module as a discovery
# surface while keeping its stricter public-projection policy isolated.
from ..proof.proof_metrics import (  # noqa: E402  (intentional late compatibility import)
    PROOF_BENCHMARK_SCHEMA,
    PROOF_METRIC_DIMENSIONS,
    PROOF_METRICS_SCHEMA,
    ProofBenchmarkReport,
    ProofBenchmarkThresholds,
    ProofMetricsSnapshot,
    build_proof_benchmark_report,
    build_proof_metrics,
    build_proof_metrics_snapshot,
    normalize_proof_metric_identity,
)
from ..proof.formal_verification_policy import (  # noqa: E402
    PROOF_ROLLOUT_STATUS_SCHEMA,
    ProofRolloutStatus,
    build_proof_rollout_status,
)


__all__ = [
    "GOAL_COMPLETION_DIAGNOSTICS_SCHEMA",
    "GOAL_COMPLETION_DIAGNOSTICS_SCHEMA_VERSION",
    "LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS",
    "MAX_PROOF_ROLLOUT_QUERY_ROWS",
    "PROOF_ROLLOUT_QUERY_SCHEMA",
    "PROOF_ROLLOUT_QUERY_SCHEMA_VERSION",
    "REFILL_SCAN_FAILED_REASONS",
    "REFILL_SCAN_SKIPPED_REASONS",
    "REFILL_SCAN_SUCCESS_REASONS",
    "REFILL_SCAN_TERMINAL_REASONS",
    "RESOURCE_ADMISSION_EVENT_TYPES",
    "RESOURCE_ADMISSION_METRICS_SCHEMA",
    "RESOURCE_ADMISSION_METRICS_SCHEMA_VERSION",
    "RESOURCE_ADMISSION_STAGES",
    "SCHEDULER_PHASES",
    "SCHEDULER_SNAPSHOT_SCHEMA",
    "SCHEDULER_SNAPSHOT_SCHEMA_VERSION",
    "SchedulerSnapshot",
    "PROOF_METRIC_DIMENSIONS",
    "PROOF_METRICS_SCHEMA",
    "PROOF_BENCHMARK_SCHEMA",
    "PROOF_ROLLOUT_STATUS_SCHEMA",
    "ProofBenchmarkReport",
    "ProofBenchmarkThresholds",
    "ProofMetricsSnapshot",
    "ProofRolloutStatus",
    "build_proof_benchmark_report",
    "build_proof_metrics",
    "build_proof_metrics_snapshot",
    "build_proof_rollout_status",
    "build_scheduler_snapshot",
    "build_scheduler_snapshot_from_paths",
    "derive_scheduler_snapshot",
    "goal_completion_diagnostics",
    "normalize_metric_identity",
    "normalize_proof_metric_identity",
    "publish_scheduler_snapshot",
    "publish_proof_rollout_status",
    "proof_rollout_diagnostics",
    "project_goal_completion_diagnostics",
    "project_resource_admission_metrics",
    "query_proof_rollout_diagnostics",
    "query_proof_rollout_status",
    "read_proof_rollout_status",
    "read_scheduler_snapshot",
    "ready_task_cids",
    "scheduler_metrics_snapshot",
    "scheduler_snapshot",
    "scheduler_state_events",
    "write_scheduler_snapshot",
    "write_proof_rollout_query",
    "write_proof_rollout_status",
    "SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID",
    "SUPERVISOR_USAGE_METRICS_SCHEMA",
    "SUPERVISOR_USAGE_METRICS_SCHEMA_VERSION",
    "USAGE_GOVERNANCE_EVENT_TYPES",
    "MAX_USAGE_METRIC_SERIES",
    "forbidden_usage_metric_label_keys",
    "project_usage_governance_metrics",
    "BENCHMARK_SPAN_CLOCK_SCHEMA",
    "BENCHMARK_SPAN_CLOCK_SCHEMA_VERSION",
    "bind_metric_row_to_span_identity",
    "join_scheduler_snapshot_to_benchmark_span",
    "project_snapshot_metrics_for_span",
]
