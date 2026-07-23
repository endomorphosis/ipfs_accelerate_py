"""Bounded runtime temporal monitoring for supervisor event traces.

This module is intentionally a runtime *counterexample detector*, not a proof
engine.  A clean finite prefix is reported as ``no_violation_observed`` and is
never promoted to evidence that an unbounded temporal property is true.

The monitor accepts the JSONL events already emitted by the implementation
daemon and also the canonical event names used by lease, proof, validation,
and merge services.  Timestamped events are reordered inside a finite window;
events outside that window and events without timestamps are represented
explicitly as inconclusive observations.
"""

from __future__ import annotations

import fcntl
import hashlib
import heapq
import json
import math
import os
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .event_log import event_log_sources, read_jsonl_events


RUNTIME_TEMPORAL_MONITOR_VERSION: Final = 1
RUNTIME_TEMPORAL_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-temporal-policy@1"
)
RUNTIME_TEMPORAL_COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-temporal-counterexample@1"
)
RUNTIME_TEMPORAL_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-temporal-report@1"
)
RUNTIME_TEMPORAL_REOPEN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-temporal-reopen-request@1"
)

_UNKNOWN_TASK = "<unknown-task>"
_UNKNOWN_LANE = "<unknown-lane>"
_UNKNOWN_TREE = "<unknown-tree>"
_UNKNOWN_EPOCH = "<unknown-epoch>"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def _text(value: Any, default: str = "") -> str:
    result = str(value or "").strip()
    return result or default


def _boolean(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _first_text(payload: Mapping[str, Any], names: Sequence[str], default: str) -> str:
    for name in names:
        value = _text(payload.get(name))
        if value:
            return value
    return default


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: Any) -> tuple[float | None, str]:
    """Return a UTC epoch timestamp and its stable textual representation."""

    if value is None or value == "":
        return None, ""
    if isinstance(value, bool):
        return None, ""
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            return None, ""
        return result, datetime.fromtimestamp(result, timezone.utc).isoformat()
    text = str(value).strip()
    if not text:
        return None, ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            result = float(text)
        except ValueError:
            return None, text
        if not math.isfinite(result):
            return None, text
        return result, datetime.fromtimestamp(result, timezone.utc).isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.timestamp(), parsed.isoformat()


def _bounded_public(value: Any, *, max_string_bytes: int = 2048, depth: int = 0) -> Any:
    """Produce a deterministic, JSON-safe and size-conscious event projection."""

    if depth >= 6:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if len(encoded) <= max_string_bytes:
            return value
        suffix = "...<truncated>"
        kept = encoded[: max(0, max_string_bytes - len(suffix))].decode(
            "utf-8", errors="ignore"
        )
        return kept + suffix
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_public(item, max_string_bytes=max_string_bytes, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:128]
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _bounded_public(item, max_string_bytes=max_string_bytes, depth=depth + 1)
            for item in list(value)[:128]
        ]
    return _bounded_public(str(value), max_string_bytes=max_string_bytes, depth=depth + 1)


class TemporalPropertyKind(str, Enum):
    EVENT_ORDERING = "event_ordering"
    LEASE_EXPIRATION = "lease_expiration"
    NO_ACTION_AFTER_STOP = "no_action_after_revocation_or_cancellation"
    PROOF_BEFORE_MERGE = "proof_before_merge"
    BOUNDED_RETRY = "bounded_retry"
    EVENTUAL_TERMINAL = "eventual_terminal_status"
    RESOURCE_RELEASE = "resource_release_deadline"


class MonitorVerdict(str, Enum):
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"
    NO_VIOLATION_OBSERVED = "no_violation_observed"


class NoticeSeverity(str, Enum):
    INFO = "info"
    INCONCLUSIVE = "inconclusive"


class NoticeCode(str, Enum):
    DUPLICATE_EVENT = "duplicate_event"
    MISSING_TIMESTAMP = "missing_timestamp"
    INVALID_TIMESTAMP = "invalid_timestamp"
    OUT_OF_ORDER_WINDOW_EXCEEDED = "out_of_order_window_exceeded"
    UNKNOWN_TASK = "unknown_task"
    POLICY_IDENTITY_MISMATCH = "policy_identity_mismatch"
    PARTITION_EVICTED = "partition_evicted"
    EVENT_HISTORY_TRUNCATED = "event_history_truncated"
    LEASE_EXPIRY_UNKNOWN = "lease_expiry_unknown"
    OPEN_OBLIGATION = "open_obligation"
    OBSERVATION_RETENTION_TRUNCATED = "observation_retention_truncated"
    REOPEN_FAILED = "reopen_failed"


@dataclass(frozen=True)
class TemporalProperty:
    """One versioned monitor property and its reviewed finite MTL profile."""

    kind: TemporalPropertyKind
    property_id: str
    version: int
    formula: str
    bound_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("property_id", "formula"):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} is required")
        _positive_int(self.version, name="property version")
        if self.bound_seconds is not None:
            _finite_nonnegative(self.bound_seconds, name="bound_seconds")

    @property
    def identity(self) -> str:
        return _identity("temporal-property", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "property_id": self.property_id,
            "version": self.version,
            "formula": self.formula,
        }
        if self.bound_seconds is not None:
            result["bound_seconds"] = self.bound_seconds
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalProperty":
        return cls(
            kind=TemporalPropertyKind(str(payload.get("kind") or "")),
            property_id=_text(payload.get("property_id")),
            version=int(payload.get("version") or 0),
            formula=_text(payload.get("formula")),
            bound_seconds=(
                None
                if payload.get("bound_seconds") is None
                else float(payload["bound_seconds"])
            ),
        )


def _default_properties(
    terminal_deadline_seconds: float,
    resource_release_deadline_seconds: float,
) -> tuple[TemporalProperty, ...]:
    return (
        TemporalProperty(
            TemporalPropertyKind.EVENT_ORDERING,
            "supervisor.event-ordering",
            1,
            "finish -> previously(start); terminal -> previously(finish)",
        ),
        TemporalProperty(
            TemporalPropertyKind.LEASE_EXPIRATION,
            "supervisor.lease-expiration",
            1,
            "action -> active_lease and event_time < lease_expiry",
        ),
        TemporalProperty(
            TemporalPropertyKind.NO_ACTION_AFTER_STOP,
            "supervisor.no-action-after-stop",
            1,
            "G((revoked or cancelled) -> G(not action))",
        ),
        TemporalProperty(
            TemporalPropertyKind.PROOF_BEFORE_MERGE,
            "supervisor.proof-before-merge",
            1,
            "merge -> previously(accepted_proof)",
        ),
        TemporalProperty(
            TemporalPropertyKind.BOUNDED_RETRY,
            "supervisor.bounded-retry",
            1,
            "retry_count <= max_retries",
        ),
        TemporalProperty(
            TemporalPropertyKind.EVENTUAL_TERMINAL,
            "supervisor.eventual-terminal",
            1,
            "start -> F[0,terminal_deadline](terminal)",
            terminal_deadline_seconds,
        ),
        TemporalProperty(
            TemporalPropertyKind.RESOURCE_RELEASE,
            "supervisor.resource-release",
            1,
            "terminal_with_resource -> F[0,release_deadline](released)",
            resource_release_deadline_seconds,
        ),
    )


@dataclass(frozen=True)
class TemporalMonitorPolicy:
    """Versioned policy interpreted by :class:`RuntimeTemporalMonitor`."""

    policy_name: str = "supervisor-runtime-mtl"
    version: int = 1
    max_retries: int = 3
    terminal_deadline_seconds: float = 3600.0
    resource_release_deadline_seconds: float = 60.0
    require_lease_for_actions: bool = True
    require_proof_before_merge: bool = True
    properties: tuple[TemporalProperty, ...] = ()

    def __post_init__(self) -> None:
        if not _text(self.policy_name):
            raise ValueError("policy_name is required")
        _positive_int(self.version, name="policy version")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        _finite_nonnegative(
            self.terminal_deadline_seconds, name="terminal_deadline_seconds"
        )
        _finite_nonnegative(
            self.resource_release_deadline_seconds,
            name="resource_release_deadline_seconds",
        )
        properties = self.properties or _default_properties(
            self.terminal_deadline_seconds,
            self.resource_release_deadline_seconds,
        )
        kinds = [item.kind for item in properties]
        if len(kinds) != len(set(kinds)):
            raise ValueError("temporal property kinds must be unique")
        object.__setattr__(self, "properties", tuple(properties))

    @property
    def policy_id(self) -> str:
        return _identity("runtime-temporal-policy", self.to_dict())

    def property(self, kind: TemporalPropertyKind) -> TemporalProperty:
        for item in self.properties:
            if item.kind is kind:
                return item
        raise KeyError(kind.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TEMPORAL_POLICY_SCHEMA,
            "policy_name": self.policy_name,
            "version": self.version,
            "max_retries": self.max_retries,
            "terminal_deadline_seconds": self.terminal_deadline_seconds,
            "resource_release_deadline_seconds": self.resource_release_deadline_seconds,
            "require_lease_for_actions": self.require_lease_for_actions,
            "require_proof_before_merge": self.require_proof_before_merge,
            "properties": [item.to_dict() for item in self.properties],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalMonitorPolicy":
        raw_properties = payload.get("properties")
        properties: tuple[TemporalProperty, ...] = ()
        if isinstance(raw_properties, Sequence) and not isinstance(
            raw_properties, (str, bytes, bytearray)
        ):
            properties = tuple(
                item
                if isinstance(item, TemporalProperty)
                else TemporalProperty.from_dict(item)
                for item in raw_properties
                if isinstance(item, (TemporalProperty, Mapping))
            )
        return cls(
            policy_name=_text(payload.get("policy_name"), "supervisor-runtime-mtl"),
            version=int(payload.get("version") or 1),
            max_retries=int(payload.get("max_retries", 3)),
            terminal_deadline_seconds=float(
                payload.get("terminal_deadline_seconds", 3600.0)
            ),
            resource_release_deadline_seconds=float(
                payload.get("resource_release_deadline_seconds", 60.0)
            ),
            require_lease_for_actions=_boolean(
                payload.get("require_lease_for_actions"), default=True
            ),
            require_proof_before_merge=_boolean(
                payload.get("require_proof_before_merge"), default=True
            ),
            properties=properties,
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "TemporalMonitorPolicy":
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("temporal monitor policy JSON must contain an object")
        return cls.from_dict(value)


RuntimeTemporalPolicy = TemporalMonitorPolicy


@dataclass(frozen=True, order=True)
class TracePartition:
    task_id: str
    lane_id: str
    tree_id: str
    policy_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "lane_id": self.lane_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TracePartition":
        return cls(
            task_id=_text(payload.get("task_id"), _UNKNOWN_TASK),
            lane_id=_text(payload.get("lane_id"), _UNKNOWN_LANE),
            tree_id=_text(payload.get("tree_id"), _UNKNOWN_TREE),
            policy_id=_text(payload.get("policy_id")),
        )


TemporalPartitionKey = TracePartition


@dataclass(frozen=True)
class TemporalMonitorConfig:
    out_of_order_window_seconds: float = 5.0
    max_partitions: int = 2048
    max_events_per_partition: int = 128
    max_pending_events_per_partition: int = 256
    max_resources_per_partition: int = 128
    max_duplicate_identities: int = 65_536
    max_counterexample_events: int = 32
    max_counterexamples_in_memory: int = 4096
    max_notices_in_memory: int = 4096
    max_reopen_requests_in_memory: int = 4096
    counterexample_path: Path | str | None = None
    reopen_path: Path | str | None = None

    def __post_init__(self) -> None:
        _finite_nonnegative(
            self.out_of_order_window_seconds,
            name="out_of_order_window_seconds",
        )
        for name in (
            "max_partitions",
            "max_events_per_partition",
            "max_pending_events_per_partition",
            "max_resources_per_partition",
            "max_duplicate_identities",
            "max_counterexample_events",
            "max_counterexamples_in_memory",
            "max_notices_in_memory",
            "max_reopen_requests_in_memory",
        ):
            _positive_int(getattr(self, name), name=name)
        for name in ("counterexample_path", "reopen_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


RuntimeTemporalMonitorConfig = TemporalMonitorConfig


@dataclass(frozen=True)
class NormalizedSupervisorEvent:
    event_id: str
    event_type: str
    partition: TracePartition
    epoch_id: str
    timestamp: float | None
    timestamp_text: str
    sequence: int | None
    arrival_index: int
    payload: Mapping[str, Any]
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "event_id": self.event_id,
            "type": self.event_type,
            "partition": self.partition.to_dict(),
            "epoch_id": self.epoch_id,
            "arrival_index": self.arrival_index,
            "payload": _bounded_public(self.payload),
        }
        if self.timestamp_text:
            result["timestamp"] = self.timestamp_text
        if self.sequence is not None:
            result["sequence"] = self.sequence
        if self.source:
            result["source"] = self.source
        return result


@dataclass(frozen=True)
class MonitorNotice:
    code: NoticeCode
    severity: NoticeSeverity
    message: str
    partition: TracePartition
    event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "partition": self.partition.to_dict(),
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class TemporalCounterexample:
    counterexample_id: str
    property_id: str
    property_version: int
    property_kind: TemporalPropertyKind
    policy_id: str
    partition: TracePartition
    reason: str
    observed_at: str
    trigger_event_id: str
    events: tuple[Mapping[str, Any], ...]
    reopened_work: bool = False

    @property
    def is_proof(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TEMPORAL_COUNTEREXAMPLE_SCHEMA,
            "monitor_version": RUNTIME_TEMPORAL_MONITOR_VERSION,
            "counterexample_id": self.counterexample_id,
            "property_id": self.property_id,
            "property_version": self.property_version,
            "property_kind": self.property_kind.value,
            "policy_id": self.policy_id,
            "partition": self.partition.to_dict(),
            "reason": self.reason,
            "observed_at": self.observed_at,
            "trigger_event_id": self.trigger_event_id,
            "events": [_bounded_public(item) for item in self.events],
            "reopened_work": self.reopened_work,
            "is_proof": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalCounterexample":
        raw_events = payload.get("events")
        events = (
            tuple(dict(item) for item in raw_events if isinstance(item, Mapping))
            if isinstance(raw_events, Sequence)
            and not isinstance(raw_events, (str, bytes, bytearray))
            else ()
        )
        partition = payload.get("partition")
        if not isinstance(partition, Mapping):
            raise ValueError("counterexample partition is required")
        return cls(
            counterexample_id=_text(payload.get("counterexample_id")),
            property_id=_text(payload.get("property_id")),
            property_version=int(payload.get("property_version") or 0),
            property_kind=TemporalPropertyKind(
                str(payload.get("property_kind") or "")
            ),
            policy_id=_text(payload.get("policy_id")),
            partition=TracePartition.from_dict(partition),
            reason=_text(payload.get("reason")),
            observed_at=_text(payload.get("observed_at")),
            trigger_event_id=_text(payload.get("trigger_event_id")),
            events=events,
            reopened_work=bool(payload.get("reopened_work", False)),
        )


RuntimeTemporalCounterexample = TemporalCounterexample
TemporalViolation = TemporalCounterexample


@dataclass(frozen=True)
class WorkReopenRequest:
    request_id: str
    task_id: str
    lane_id: str
    tree_id: str
    policy_id: str
    counterexample_id: str
    reason: str
    requested_at: str
    status: str = "reopened"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TEMPORAL_REOPEN_SCHEMA,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "lane_id": self.lane_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "counterexample_id": self.counterexample_id,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class TemporalMonitorReport:
    policy_id: str
    verdict: MonitorVerdict
    counterexamples: tuple[TemporalCounterexample, ...]
    notices: tuple[MonitorNotice, ...]
    reopen_requests: tuple[WorkReopenRequest, ...]
    events_received: int
    events_evaluated: int
    duplicate_events: int
    partitions_observed: int
    active_partitions: int
    counterexamples_durable: bool
    finalized: bool
    counterexamples_observed: int = 0
    notices_observed: int = 0
    reopen_requests_observed: int = 0
    retained_observations_truncated: bool = False

    @property
    def violations(self) -> tuple[TemporalCounterexample, ...]:
        return self.counterexamples

    @property
    def proved(self) -> bool:
        return False

    @property
    def no_violation_observed(self) -> bool:
        return self.verdict is MonitorVerdict.NO_VIOLATION_OBSERVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_TEMPORAL_REPORT_SCHEMA,
            "monitor_version": RUNTIME_TEMPORAL_MONITOR_VERSION,
            "policy_id": self.policy_id,
            "verdict": self.verdict.value,
            "counterexamples": [item.to_dict() for item in self.counterexamples],
            "notices": [item.to_dict() for item in self.notices],
            "reopen_requests": [item.to_dict() for item in self.reopen_requests],
            "events_received": self.events_received,
            "events_evaluated": self.events_evaluated,
            "duplicate_events": self.duplicate_events,
            "partitions_observed": self.partitions_observed,
            "active_partitions": self.active_partitions,
            "counterexamples_durable": self.counterexamples_durable,
            "finalized": self.finalized,
            "counterexamples_observed": self.counterexamples_observed,
            "notices_observed": self.notices_observed,
            "reopen_requests_observed": self.reopen_requests_observed,
            "retained_observations_truncated": self.retained_observations_truncated,
            "proved": False,
        }


RuntimeTemporalMonitorResult = TemporalMonitorReport
TemporalMonitorResult = TemporalMonitorReport


def _append_durable_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0, os.SEEK_END)
            stream.write(_canonical_json(_bounded_public(payload)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _existing_identities(
    path: Path | None,
    field_name: str,
    *,
    max_identities: int,
) -> OrderedDict[str, None]:
    if path is None or not path.exists() or path.is_dir():
        return OrderedDict()
    identities: OrderedDict[str, None] = OrderedDict()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        return identities
    with stream:
        for raw_line in stream:
            try:
                payload = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            value = _text(payload.get(field_name))
            if value:
                identities[value] = None
                identities.move_to_end(value)
                while len(identities) > max_identities:
                    identities.popitem(last=False)
    return identities


def _durable_has_identity(
    path: Path | None, field_name: str, identity: str
) -> bool:
    """Check an old durable identity without retaining an unbounded index."""

    if path is None or not path.exists() or path.is_dir():
        return False
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        return False
    with stream:
        for raw_line in stream:
            if identity not in raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, Mapping) and _text(payload.get(field_name)) == identity:
                return True
    return False


def load_temporal_counterexamples(path: Path | str) -> tuple[dict[str, Any], ...]:
    """Load durable counterexamples, deduplicated by content identity."""

    records: dict[str, dict[str, Any]] = {}
    for item in read_jsonl_events(Path(path)):
        identity = _text(item.get("counterexample_id"))
        if identity:
            records[identity] = item
    return tuple(records[key] for key in sorted(records))


load_runtime_temporal_counterexamples = load_temporal_counterexamples


@dataclass
class _PartitionState:
    history: deque[NormalizedSupervisorEvent]
    pending: list[tuple[float, int, int, NormalizedSupervisorEvent]] = field(
        default_factory=list
    )
    max_seen_timestamp: float | None = None
    last_timestamp: float | None = None
    sequences: OrderedDict[str, int] = field(default_factory=OrderedDict)
    started: bool = False
    implementation_finished: bool = False
    terminal: bool = False
    cancelled_or_revoked: bool = False
    accepted_proof: bool = False
    lease_active: bool = False
    lease_seen: bool = False
    lease_expiry: float | None = None
    retries: int = 0
    retry_reported: bool = False
    resources: dict[str, float | None] = field(default_factory=dict)
    resource_release_deadlines: dict[str, float] = field(default_factory=dict)
    terminal_deadline: float | None = None
    terminal_deadline_reported: bool = False
    release_deadlines_reported: set[str] = field(default_factory=set)
    history_truncation_reported: bool = False

    def has_live_obligations(self) -> bool:
        return bool(
            (self.started and not self.terminal)
            or self.lease_active
            or self.resources
            or self.resource_release_deadlines
        )


_START_TYPES = {
    "action_started",
    "implementation_started",
    "task_started",
    "work_started",
}
_IMPLEMENTATION_FINISH_TYPES = {
    "action_finished",
    "implementation_finished",
    "work_finished",
}
_TERMINAL_TYPES = {
    "task_completed",
    "task_failed",
    "task_cancelled",
    "task_canceled",
    "task_revoked",
    "terminal_status",
}
_STOP_TYPES = {
    "cancelled",
    "canceled",
    "revoked",
    "lease_revoked",
    "task_cancelled",
    "task_canceled",
    "task_revoked",
    "action_cancelled",
    "action_canceled",
    "cancellation_requested",
    "revocation_requested",
    "work_cancelled",
    "work_canceled",
    "lease_cancelled",
    "lease_canceled",
}
_PROOF_TYPES = {
    "proof_accepted",
    "proof_verified",
    "kernel_verified",
    "model_check_succeeded",
    "formal_proof_verified",
}
_MERGE_TYPES = {
    "merge_started",
    "merge_finished",
    "merge_reconciled",
    "task_merged",
}
_LEASE_ACQUIRE_TYPES = {"lease_acquired", "lease_granted", "lease_renewed"}
_LEASE_END_TYPES = {
    "lease_expired",
    "lease_expiration",
    "lease_released",
    "lease_revoked",
    "lease_cancelled",
    "lease_canceled",
}
_RESOURCE_ACQUIRE_TYPES = {"resource_acquired", "resource_allocated"}
_RESOURCE_RELEASE_TYPES = {
    "resource_released",
    "resource_deallocated",
    "worktree_pool_lease_released",
    "merged_worktree_cleanup",
}
_RETRY_TYPES = {"retry", "retry_started", "retry_scheduled", "implementation_retried"}
_NON_ACTION_TYPES = (
    _STOP_TYPES
    | _LEASE_ACQUIRE_TYPES
    | _LEASE_END_TYPES
    | _PROOF_TYPES
    | _RESOURCE_RELEASE_TYPES
    | {
        "daemon_start",
        "daemon_started",
        "supervisor_restart",
        "restart_epoch",
        "heartbeat",
        "lane_heartbeat",
        "policy_changed",
    }
)


def _status_is_terminal(event: NormalizedSupervisorEvent) -> bool:
    status = _text(
        event.payload.get("status")
        or event.payload.get("state")
        or event.payload.get("phase")
    ).lower()
    return event.event_type in _TERMINAL_TYPES or status in {
        "completed",
        "complete",
        "failed",
        "cancelled",
        "canceled",
        "revoked",
        "merged",
        "terminal",
    }


def _proof_accepted(event: NormalizedSupervisorEvent) -> bool:
    if event.event_type in _PROOF_TYPES:
        status = _text(
            event.payload.get("status") or event.payload.get("verdict") or "accepted"
        ).lower()
        return status in {"accepted", "verified", "proved", "succeeded", "success", "passed"}
    return bool(
        event.payload.get("proof_verified")
        or event.payload.get("proof_accepted")
        or event.payload.get("kernel_verified")
    )


def _is_merge(event: NormalizedSupervisorEvent) -> bool:
    if event.event_type in _MERGE_TYPES:
        if event.event_type == "merge_reconciled":
            return bool(event.payload.get("resolved", True))
        merge = event.payload.get("merge_result")
        if isinstance(merge, Mapping) and "merged" in merge:
            return bool(merge.get("merged"))
        return True
    merge = event.payload.get("merge_result")
    return isinstance(merge, Mapping) and bool(merge.get("merged"))


def _is_action(event: NormalizedSupervisorEvent) -> bool:
    explicit = event.payload.get("is_action")
    if isinstance(explicit, bool):
        return explicit
    if event.event_type in _NON_ACTION_TYPES:
        return False
    return (
        event.event_type in _START_TYPES
        or event.event_type in _IMPLEMENTATION_FINISH_TYPES
        or event.event_type in _MERGE_TYPES
        or event.event_type in _RETRY_TYPES
        or event.event_type.startswith(("action_", "validation_", "merge_", "implementation_"))
    )


class RuntimeTemporalMonitor:
    """Streaming, bounded and partition-aware supervisor trace monitor."""

    def __init__(
        self,
        policy: TemporalMonitorPolicy | None = None,
        config: TemporalMonitorConfig | None = None,
        *,
        counterexample_path: Path | str | None = None,
        reopen_path: Path | str | None = None,
        reopen_callback: Callable[[WorkReopenRequest], Any] | None = None,
    ) -> None:
        self.policy = policy or TemporalMonitorPolicy()
        base_config = config or TemporalMonitorConfig()
        if counterexample_path is not None or reopen_path is not None:
            base_config = TemporalMonitorConfig(
                **{
                    **base_config.__dict__,
                    "counterexample_path": (
                        counterexample_path
                        if counterexample_path is not None
                        else base_config.counterexample_path
                    ),
                    "reopen_path": (
                        reopen_path if reopen_path is not None else base_config.reopen_path
                    ),
                }
            )
        if (
            base_config.counterexample_path is not None
            and base_config.reopen_path is None
        ):
            counterexample_file = Path(base_config.counterexample_path)
            base_config = TemporalMonitorConfig(
                **{
                    **base_config.__dict__,
                    "reopen_path": counterexample_file.with_name(
                        f"{counterexample_file.stem}.reopen-requests.jsonl"
                    ),
                }
            )
        self.config = base_config
        self.reopen_callback = reopen_callback
        self._states: OrderedDict[TracePartition, _PartitionState] = OrderedDict()
        self._partition_admissions = 0
        self._seen_events: OrderedDict[str, None] = OrderedDict()
        self._counterexamples: list[TemporalCounterexample] = []
        self._notices: list[MonitorNotice] = []
        self._reopen_requests: list[WorkReopenRequest] = []
        self._counterexamples_observed = 0
        self._notices_observed = 0
        self._reopen_requests_observed = 0
        self._retained_observations_truncated = False
        self._inconclusive_observed = False
        self._persisted_counterexamples = _existing_identities(
            self.config.counterexample_path,
            "counterexample_id",
            max_identities=self.config.max_duplicate_identities,
        )
        self._persisted_reopens = _existing_identities(
            self.config.reopen_path,
            "request_id",
            max_identities=self.config.max_duplicate_identities,
        )
        self._arrival_index = 0
        self._events_received = 0
        self._events_evaluated = 0
        self._duplicate_events = 0
        self._finalized = False

    def _partition(self, payload: Mapping[str, Any]) -> TracePartition:
        task_id = _first_text(
            payload,
            ("canonical_task_key", "task_id", "task_cid", "work_id"),
            _UNKNOWN_TASK,
        )
        lane_id = _first_text(
            payload,
            ("lane_id", "lane", "bundle", "bundle_id", "track"),
            _UNKNOWN_LANE,
        )
        tree_id = _first_text(
            payload,
            (
                "repository_tree_id",
                "tree_id",
                "tree_cid",
                "repository_tree_cid",
                "repo_id",
            ),
            _UNKNOWN_TREE,
        )
        policy_id = _first_text(
            payload,
            ("runtime_temporal_policy_id", "temporal_policy_id", "policy_id"),
            self.policy.policy_id,
        )
        return TracePartition(task_id, lane_id, tree_id, policy_id)

    def _normalize(
        self, payload: Mapping[str, Any], source: Path | str | None
    ) -> NormalizedSupervisorEvent:
        self._arrival_index += 1
        event_type = _text(
            payload.get("type") or payload.get("event_type") or payload.get("kind"),
            "unknown",
        ).lower()
        partition = self._partition(payload)
        epoch_id = _first_text(
            payload,
            ("restart_epoch", "epoch_id", "epoch", "run_id", "daemon_instance_id"),
            _UNKNOWN_EPOCH,
        )
        raw_timestamp = payload.get("timestamp")
        if raw_timestamp in (None, ""):
            raw_timestamp = payload.get("occurred_at", payload.get("event_time"))
        timestamp, timestamp_text = _timestamp(raw_timestamp)
        raw_sequence = payload.get("sequence", payload.get("sequence_number"))
        try:
            sequence = (
                None
                if raw_sequence in (None, "") or isinstance(raw_sequence, bool)
                else int(raw_sequence)
            )
        except (TypeError, ValueError):
            sequence = None
        explicit_id = _first_text(
            payload,
            ("event_id", "id", "receipt_id", "event_cid"),
            "",
        )
        if explicit_id:
            event_id = f"{epoch_id}:{explicit_id}"
        else:
            event_id = _identity(
                "supervisor-event",
                {
                    "epoch_id": epoch_id,
                    "event": _bounded_public(payload),
                },
            )
        return NormalizedSupervisorEvent(
            event_id=event_id,
            event_type=event_type,
            partition=partition,
            epoch_id=epoch_id,
            timestamp=timestamp,
            timestamp_text=timestamp_text,
            sequence=sequence,
            arrival_index=self._arrival_index,
            payload=dict(payload),
            source=str(source or ""),
        )

    def _state(self, partition: TracePartition) -> _PartitionState:
        state = self._states.get(partition)
        if state is not None:
            self._states.move_to_end(partition)
            return state
        if len(self._states) >= self.config.max_partitions:
            evicted_partition, evicted = self._states.popitem(last=False)
            self._drain(evicted_partition, evicted, flush_all=True)
            if evicted.has_live_obligations():
                self._notice(
                    NoticeCode.PARTITION_EVICTED,
                    NoticeSeverity.INCONCLUSIVE,
                    "bounded partition capacity evicted state with live temporal obligations",
                    evicted_partition,
                )
        state = _PartitionState(history=deque(maxlen=self.config.max_events_per_partition))
        self._states[partition] = state
        self._partition_admissions += 1
        return state

    def _notice(
        self,
        code: NoticeCode,
        severity: NoticeSeverity,
        message: str,
        partition: TracePartition,
        event_id: str = "",
    ) -> None:
        self._notices_observed += 1
        if severity is NoticeSeverity.INCONCLUSIVE:
            self._inconclusive_observed = True
        notice = MonitorNotice(code, severity, message, partition, event_id)
        if len(self._notices) >= self.config.max_notices_in_memory:
            self._notices.pop(0)
            self._retained_observations_truncated = True
        self._notices.append(notice)

    def ingest(
        self,
        payload: Mapping[str, Any],
        *,
        source: Path | str | None = None,
    ) -> tuple[TemporalCounterexample, ...]:
        """Ingest one event and return counterexamples newly observed by this call."""

        if not isinstance(payload, Mapping):
            raise TypeError("supervisor event must be a mapping")
        if self._finalized:
            raise RuntimeError("cannot ingest after finalize")
        before = len(self._counterexamples)
        self._events_received += 1
        event = self._normalize(payload, source)
        if event.event_id in self._seen_events:
            self._duplicate_events += 1
            self._seen_events.move_to_end(event.event_id)
            self._notice(
                NoticeCode.DUPLICATE_EVENT,
                NoticeSeverity.INFO,
                "duplicate event was ignored",
                event.partition,
                event.event_id,
            )
            return ()
        self._seen_events[event.event_id] = None
        while len(self._seen_events) > self.config.max_duplicate_identities:
            self._seen_events.popitem(last=False)

        if event.partition.task_id == _UNKNOWN_TASK:
            self._notice(
                NoticeCode.UNKNOWN_TASK,
                NoticeSeverity.INCONCLUSIVE,
                "event cannot be associated with affected work",
                event.partition,
                event.event_id,
            )
        if event.partition.policy_id != self.policy.policy_id:
            self._notice(
                NoticeCode.POLICY_IDENTITY_MISMATCH,
                NoticeSeverity.INCONCLUSIVE,
                "event names a different temporal policy identity and was not evaluated",
                event.partition,
                event.event_id,
            )
            return ()

        state = self._state(event.partition)
        if event.timestamp is None:
            code = (
                NoticeCode.MISSING_TIMESTAMP
                if not event.timestamp_text
                else NoticeCode.INVALID_TIMESTAMP
            )
            self._notice(
                code,
                NoticeSeverity.INCONCLUSIVE,
                "event has no usable timestamp; metric deadlines cannot be evaluated",
                event.partition,
                event.event_id,
            )
            self._evaluate(state, event)
        else:
            prior_watermark = (
                None
                if state.max_seen_timestamp is None
                else state.max_seen_timestamp
                - self.config.out_of_order_window_seconds
            )
            if prior_watermark is not None and event.timestamp < prior_watermark:
                self._notice(
                    NoticeCode.OUT_OF_ORDER_WINDOW_EXCEEDED,
                    NoticeSeverity.INCONCLUSIVE,
                    (
                        f"event timestamp is older than committed watermark "
                        f"{datetime.fromtimestamp(prior_watermark, timezone.utc).isoformat()}"
                    ),
                    event.partition,
                    event.event_id,
                )
                self._append_history(state, event)
            else:
                state.max_seen_timestamp = max(
                    event.timestamp,
                    state.max_seen_timestamp
                    if state.max_seen_timestamp is not None
                    else event.timestamp,
                )
                heapq.heappush(
                    state.pending,
                    (
                        event.timestamp,
                        (
                            event.sequence
                            if event.sequence is not None
                            else event.arrival_index
                        ),
                        event.arrival_index,
                        event,
                    ),
                )
                if len(state.pending) > self.config.max_pending_events_per_partition:
                    _timestamp_value, _order, _arrival, forced = heapq.heappop(
                        state.pending
                    )
                    self._notice(
                        NoticeCode.OUT_OF_ORDER_WINDOW_EXCEEDED,
                        NoticeSeverity.INCONCLUSIVE,
                        "pending reorder buffer bound forced early event commitment",
                        event.partition,
                        forced.event_id,
                    )
                    self._evaluate(state, forced)
                self._drain(event.partition, state)
        return tuple(self._counterexamples[before:])

    observe = ingest
    process_event = ingest

    def ingest_many(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        source: Path | str | None = None,
    ) -> tuple[TemporalCounterexample, ...]:
        before = len(self._counterexamples)
        for event in events:
            self.ingest(event, source=source)
        return tuple(self._counterexamples[before:])

    process_events = ingest_many

    def ingest_logs(
        self,
        paths: Iterable[Path | str] | Path | str,
        *,
        include_rotated: bool = True,
        repair_active: bool = False,
    ) -> tuple[TemporalCounterexample, ...]:
        """Read active and rotated JSONL logs without assuming file order is time order."""

        raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        before = len(self._counterexamples)
        for path in event_log_sources(raw_paths, include_rotated=include_rotated):
            repair = repair_active and ".rotated-" not in path.name
            for event in read_jsonl_events(path, repair=repair):
                self.ingest(event, source=path)
        return tuple(self._counterexamples[before:])

    process_logs = ingest_logs

    def _drain(
        self,
        partition: TracePartition,
        state: _PartitionState,
        *,
        flush_all: bool = False,
    ) -> None:
        del partition
        watermark = (
            float("inf")
            if flush_all
            else (
                -float("inf")
                if state.max_seen_timestamp is None
                else state.max_seen_timestamp
                - self.config.out_of_order_window_seconds
            )
        )
        while state.pending and state.pending[0][0] <= watermark:
            _timestamp_value, _order, _arrival, event = heapq.heappop(state.pending)
            self._evaluate(state, event)

    def flush(self) -> TemporalMonitorReport:
        for partition, state in list(self._states.items()):
            self._drain(partition, state, flush_all=True)
        return self.report()

    def _append_history(
        self, state: _PartitionState, event: NormalizedSupervisorEvent
    ) -> None:
        full = len(state.history) == state.history.maxlen
        state.history.append(event)
        if full and not state.history_truncation_reported:
            state.history_truncation_reported = True
            self._notice(
                NoticeCode.EVENT_HISTORY_TRUNCATED,
                NoticeSeverity.INFO,
                "counterexample context is bounded; oldest partition event was discarded",
                event.partition,
                event.event_id,
            )

    def _evaluate(self, state: _PartitionState, event: NormalizedSupervisorEvent) -> None:
        self._events_evaluated += 1
        if event.timestamp is not None:
            self._check_deadlines(state, event.timestamp, event)
            state.last_timestamp = event.timestamp

        if event.sequence is not None:
            previous = state.sequences.get(event.epoch_id)
            if previous is not None and event.sequence <= previous:
                self._violate(
                    TemporalPropertyKind.EVENT_ORDERING,
                    event,
                    state,
                    (
                        f"sequence {event.sequence} did not advance beyond {previous} "
                        f"inside restart epoch {event.epoch_id}"
                    ),
                )
            state.sequences[event.epoch_id] = max(
                event.sequence, previous if previous is not None else event.sequence
            )
            state.sequences.move_to_end(event.epoch_id)
            while len(state.sequences) > self.config.max_events_per_partition:
                state.sequences.popitem(last=False)

        is_action = _is_action(event)
        if state.cancelled_or_revoked and is_action:
            self._violate(
                TemporalPropertyKind.NO_ACTION_AFTER_STOP,
                event,
                state,
                f"action {event.event_type} occurred after revocation or cancellation",
            )

        if event.event_type in _LEASE_ACQUIRE_TYPES:
            state.lease_seen = True
            state.lease_active = True
            expiry_raw = (
                event.payload.get("expires_at")
                or event.payload.get("lease_expires_at")
                or event.payload.get("expiration")
                or event.payload.get("deadline")
            )
            expiry, _expiry_text = _timestamp(expiry_raw)
            state.lease_expiry = expiry
            if expiry is None:
                self._notice(
                    NoticeCode.LEASE_EXPIRY_UNKNOWN,
                    NoticeSeverity.INCONCLUSIVE,
                    "lease event has no usable expiration timestamp",
                    event.partition,
                    event.event_id,
                )
        elif event.event_type in _LEASE_END_TYPES:
            state.lease_active = False
            if event.event_type in {
                "lease_revoked",
                "lease_cancelled",
                "lease_canceled",
            }:
                state.cancelled_or_revoked = (
                    state.cancelled_or_revoked
                    or event.event_type
                    in {"lease_revoked", "lease_cancelled", "lease_canceled"}
                )

        if is_action:
            if self.policy.require_lease_for_actions and not state.lease_active:
                self._violate(
                    TemporalPropertyKind.LEASE_EXPIRATION,
                    event,
                    state,
                    f"action {event.event_type} occurred without an active lease",
                )
            elif (
                state.lease_expiry is not None
                and event.timestamp is not None
                and event.timestamp >= state.lease_expiry
            ):
                self._violate(
                    TemporalPropertyKind.LEASE_EXPIRATION,
                    event,
                    state,
                    (
                        f"action {event.event_type} occurred at or after lease expiry "
                        f"{datetime.fromtimestamp(state.lease_expiry, timezone.utc).isoformat()}"
                    ),
                )

        if event.event_type in _START_TYPES:
            state.started = True
            state.terminal = False
            state.terminal_deadline_reported = False
            if event.timestamp is not None:
                state.terminal_deadline = (
                    event.timestamp + self.policy.terminal_deadline_seconds
                )
        if event.event_type in _IMPLEMENTATION_FINISH_TYPES:
            if not state.started:
                self._violate(
                    TemporalPropertyKind.EVENT_ORDERING,
                    event,
                    state,
                    f"{event.event_type} occurred before a start event",
                )
            state.implementation_finished = True

        if _proof_accepted(event):
            state.accepted_proof = True

        if _is_merge(event):
            if not state.implementation_finished:
                self._violate(
                    TemporalPropertyKind.EVENT_ORDERING,
                    event,
                    state,
                    "merge occurred before implementation finished",
                )
            if self.policy.require_proof_before_merge and not state.accepted_proof:
                self._violate(
                    TemporalPropertyKind.PROOF_BEFORE_MERGE,
                    event,
                    state,
                    "merge occurred before an accepted proof event",
                )

        attempt = event.payload.get("attempt")
        if event.event_type in _RETRY_TYPES:
            state.retries += 1
        elif event.event_type in _START_TYPES and attempt not in (None, ""):
            try:
                state.retries = max(state.retries, max(0, int(attempt) - 1))
            except (TypeError, ValueError):
                pass
        if state.retries > self.policy.max_retries and not state.retry_reported:
            state.retry_reported = True
            self._violate(
                TemporalPropertyKind.BOUNDED_RETRY,
                event,
                state,
                (
                    f"observed {state.retries} retries, exceeding configured "
                    f"maximum {self.policy.max_retries}"
                ),
            )

        if event.event_type in _RESOURCE_ACQUIRE_TYPES:
            resource_id = _first_text(
                event.payload, ("resource_id", "resource", "resource_cid"), "<resource>"
            )
            if (
                resource_id not in state.resources
                and len(state.resources) >= self.config.max_resources_per_partition
            ):
                evicted_resource = next(iter(state.resources))
                state.resources.pop(evicted_resource, None)
                state.resource_release_deadlines.pop(evicted_resource, None)
                state.release_deadlines_reported.discard(evicted_resource)
                self._notice(
                    NoticeCode.PARTITION_EVICTED,
                    NoticeSeverity.INCONCLUSIVE,
                    (
                        "bounded resource capacity evicted live resource "
                        f"{evicted_resource}"
                    ),
                    event.partition,
                    event.event_id,
                )
            state.resources[resource_id] = event.timestamp
        elif event.event_type in _RESOURCE_RELEASE_TYPES:
            resource_id = _first_text(
                event.payload, ("resource_id", "resource", "resource_cid"), ""
            )
            if resource_id:
                state.resources.pop(resource_id, None)
                state.resource_release_deadlines.pop(resource_id, None)
                state.release_deadlines_reported.discard(resource_id)
            elif event.event_type in {
                "worktree_pool_lease_released",
                "merged_worktree_cleanup",
            }:
                state.resources.clear()
                state.resource_release_deadlines.clear()
                state.release_deadlines_reported.clear()

        is_terminal = _status_is_terminal(event)
        if is_terminal:
            successful_terminal = (
                event.event_type in {"task_completed", "task_merged"}
                or _text(event.payload.get("status") or event.payload.get("state")).lower()
                in {"completed", "complete", "merged"}
            )
            if successful_terminal and not state.started:
                self._violate(
                    TemporalPropertyKind.EVENT_ORDERING,
                    event,
                    state,
                    "successful terminal status occurred before work started",
                )
            elif successful_terminal and not state.implementation_finished:
                self._violate(
                    TemporalPropertyKind.EVENT_ORDERING,
                    event,
                    state,
                    "successful terminal status occurred before implementation finished",
                )
            state.terminal = True
            state.terminal_deadline = None
            if event.timestamp is not None:
                for resource_id in state.resources:
                    state.resource_release_deadlines[resource_id] = (
                        event.timestamp
                        + self.policy.resource_release_deadline_seconds
                    )

        status_value = _text(
            event.payload.get("status") or event.payload.get("state")
        ).lower()
        if event.event_type in _STOP_TYPES or status_value in {
            "cancelled",
            "canceled",
            "revoked",
        }:
            state.cancelled_or_revoked = True
            if event.timestamp is not None:
                for resource_id in state.resources:
                    state.resource_release_deadlines[resource_id] = (
                        event.timestamp
                        + self.policy.resource_release_deadline_seconds
                    )

        self._append_history(state, event)

    def _check_deadlines(
        self,
        state: _PartitionState,
        now_timestamp: float,
        trigger: NormalizedSupervisorEvent,
    ) -> None:
        if (
            state.terminal_deadline is not None
            and not state.terminal
            and not state.terminal_deadline_reported
            and now_timestamp > state.terminal_deadline
        ):
            state.terminal_deadline_reported = True
            self._violate(
                TemporalPropertyKind.EVENTUAL_TERMINAL,
                trigger,
                state,
                (
                    "task did not reach a terminal status by "
                    f"{datetime.fromtimestamp(state.terminal_deadline, timezone.utc).isoformat()}"
                ),
            )
        for resource_id, deadline in sorted(state.resource_release_deadlines.items()):
            if (
                resource_id in state.resources
                and resource_id not in state.release_deadlines_reported
                and now_timestamp > deadline
            ):
                state.release_deadlines_reported.add(resource_id)
                self._violate(
                    TemporalPropertyKind.RESOURCE_RELEASE,
                    trigger,
                    state,
                    (
                        f"resource {resource_id} was not released by "
                        f"{datetime.fromtimestamp(deadline, timezone.utc).isoformat()}"
                    ),
                )

    def advance_time(self, now: datetime | str | float | int) -> TemporalMonitorReport:
        """Advance the finite trace watermark and check metric deadlines."""

        timestamp, text = _timestamp(now)
        if timestamp is None:
            raise ValueError(f"invalid monitor time: {text or now!r}")
        self.flush()
        for partition, state in self._states.items():
            trigger = NormalizedSupervisorEvent(
                event_id=_identity(
                    "monitor-watermark", {"partition": partition.to_dict(), "time": timestamp}
                ),
                event_type="monitor_watermark",
                partition=partition,
                epoch_id="<monitor>",
                timestamp=timestamp,
                timestamp_text=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                sequence=None,
                arrival_index=self._arrival_index,
                payload={"type": "monitor_watermark", "timestamp": timestamp},
            )
            self._check_deadlines(state, timestamp, trigger)
        return self.report()

    def finalize(
        self, now: datetime | str | float | int | None = None
    ) -> TemporalMonitorReport:
        """Close the finite prefix, optionally checking deadlines at ``now``."""

        if self._finalized:
            return self.report()
        self.flush()
        if now is not None:
            self.advance_time(now)
        for partition, state in self._states.items():
            if (
                state.started
                and not state.terminal
                and not state.terminal_deadline_reported
            ):
                self._notice(
                    NoticeCode.OPEN_OBLIGATION,
                    NoticeSeverity.INCONCLUSIVE,
                    "finite trace ended before the eventual-terminal obligation was settled",
                    partition,
                )
            for resource_id in sorted(state.resource_release_deadlines):
                if (
                    resource_id in state.resources
                    and resource_id not in state.release_deadlines_reported
                ):
                    self._notice(
                        NoticeCode.OPEN_OBLIGATION,
                        NoticeSeverity.INCONCLUSIVE,
                        (
                            "finite trace ended before the release obligation for "
                            f"resource {resource_id} was settled"
                        ),
                        partition,
                    )
        self._finalized = True
        return self.report()

    close = finalize

    def _violate(
        self,
        kind: TemporalPropertyKind,
        event: NormalizedSupervisorEvent,
        state: _PartitionState,
        reason: str,
    ) -> None:
        try:
            prop = self.policy.property(kind)
        except KeyError:
            return
        context_events = [item.to_dict() for item in state.history]
        if not context_events or context_events[-1].get("event_id") != event.event_id:
            context_events.append(event.to_dict())
        context_events = context_events[-self.config.max_counterexample_events :]
        identity_payload = {
            "property_identity": prop.identity,
            "policy_id": self.policy.policy_id,
            "partition": event.partition.to_dict(),
            "reason": reason,
            "trigger_event_id": event.event_id,
        }
        counterexample_id = _identity("runtime-counterexample", identity_payload)
        if any(
            item.counterexample_id == counterexample_id
            for item in self._counterexamples
        ):
            return
        counterexample = TemporalCounterexample(
            counterexample_id=counterexample_id,
            property_id=prop.property_id,
            property_version=prop.version,
            property_kind=kind,
            policy_id=self.policy.policy_id,
            partition=event.partition,
            reason=reason,
            observed_at=_utc_now().isoformat(),
            trigger_event_id=event.event_id,
            events=tuple(context_events),
        )
        self._counterexamples_observed += 1
        if (
            len(self._counterexamples)
            >= self.config.max_counterexamples_in_memory
        ):
            self._counterexamples.pop(0)
            self._retained_observations_truncated = True
        self._counterexamples.append(counterexample)

        novel_durable = (
            counterexample_id not in self._persisted_counterexamples
            and not _durable_has_identity(
                self.config.counterexample_path,
                "counterexample_id",
                counterexample_id,
            )
        )
        if novel_durable and self.config.counterexample_path is not None:
            _append_durable_jsonl(
                self.config.counterexample_path, counterexample.to_dict()
            )
            self._persisted_counterexamples[counterexample_id] = None
            while (
                len(self._persisted_counterexamples)
                > self.config.max_duplicate_identities
            ):
                self._persisted_counterexamples.popitem(last=False)
        if novel_durable:
            self._reopen(counterexample)

    def _reopen(self, counterexample: TemporalCounterexample) -> None:
        if counterexample.partition.task_id == _UNKNOWN_TASK:
            return
        payload = {
            "task_id": counterexample.partition.task_id,
            "lane_id": counterexample.partition.lane_id,
            "tree_id": counterexample.partition.tree_id,
            "policy_id": counterexample.policy_id,
            "counterexample_id": counterexample.counterexample_id,
            "reason": counterexample.reason,
        }
        request = WorkReopenRequest(
            request_id=_identity("runtime-reopen", payload),
            requested_at=_utc_now().isoformat(),
            **payload,
        )
        if request.request_id in self._persisted_reopens or _durable_has_identity(
            self.config.reopen_path, "request_id", request.request_id
        ):
            return
        if self.config.reopen_path is not None:
            _append_durable_jsonl(self.config.reopen_path, request.to_dict())
            self._persisted_reopens[request.request_id] = None
            while len(self._persisted_reopens) > self.config.max_duplicate_identities:
                self._persisted_reopens.popitem(last=False)
        self._reopen_requests_observed += 1
        if (
            len(self._reopen_requests)
            >= self.config.max_reopen_requests_in_memory
        ):
            self._reopen_requests.pop(0)
            self._retained_observations_truncated = True
        self._reopen_requests.append(request)
        if self.reopen_callback is not None:
            try:
                self.reopen_callback(request)
            except Exception as exc:  # a failed integration must not stop monitoring
                self._notice(
                    NoticeCode.REOPEN_FAILED,
                    NoticeSeverity.INCONCLUSIVE,
                    f"reopen callback failed: {type(exc).__name__}: {exc}",
                    counterexample.partition,
                    counterexample.trigger_event_id,
                )

    def report(self) -> TemporalMonitorReport:
        if self._counterexamples_observed:
            verdict = MonitorVerdict.VIOLATED
        elif self._inconclusive_observed:
            verdict = MonitorVerdict.INCONCLUSIVE
        else:
            verdict = MonitorVerdict.NO_VIOLATION_OBSERVED
        return TemporalMonitorReport(
            policy_id=self.policy.policy_id,
            verdict=verdict,
            counterexamples=tuple(self._counterexamples),
            notices=tuple(self._notices),
            reopen_requests=tuple(self._reopen_requests),
            events_received=self._events_received,
            events_evaluated=self._events_evaluated,
            duplicate_events=self._duplicate_events,
            partitions_observed=self._partition_admissions,
            active_partitions=len(self._states),
            counterexamples_durable=self.config.counterexample_path is not None,
            finalized=self._finalized,
            counterexamples_observed=self._counterexamples_observed,
            notices_observed=self._notices_observed,
            reopen_requests_observed=self._reopen_requests_observed,
            retained_observations_truncated=self._retained_observations_truncated,
        )


SupervisorRuntimeTemporalMonitor = RuntimeTemporalMonitor


def monitor_event_trace(
    events: Iterable[Mapping[str, Any]],
    *,
    policy: TemporalMonitorPolicy | None = None,
    config: TemporalMonitorConfig | None = None,
    counterexample_path: Path | str | None = None,
    reopen_path: Path | str | None = None,
    reopen_callback: Callable[[WorkReopenRequest], Any] | None = None,
    now: datetime | str | float | int | None = None,
) -> TemporalMonitorReport:
    """Monitor a finite event trace and return a deliberately non-proof report."""

    monitor = RuntimeTemporalMonitor(
        policy,
        config,
        counterexample_path=counterexample_path,
        reopen_path=reopen_path,
        reopen_callback=reopen_callback,
    )
    monitor.ingest_many(events)
    return monitor.finalize(now)


monitor_supervisor_events = monitor_event_trace


def monitor_event_logs(
    paths: Iterable[Path | str] | Path | str,
    *,
    policy: TemporalMonitorPolicy | None = None,
    config: TemporalMonitorConfig | None = None,
    counterexample_path: Path | str | None = None,
    reopen_path: Path | str | None = None,
    reopen_callback: Callable[[WorkReopenRequest], Any] | None = None,
    include_rotated: bool = True,
    repair_active: bool = False,
    now: datetime | str | float | int | None = None,
) -> TemporalMonitorReport:
    monitor = RuntimeTemporalMonitor(
        policy,
        config,
        counterexample_path=counterexample_path,
        reopen_path=reopen_path,
        reopen_callback=reopen_callback,
    )
    monitor.ingest_logs(
        paths, include_rotated=include_rotated, repair_active=repair_active
    )
    return monitor.finalize(now)


monitor_supervisor_event_logs = monitor_event_logs


__all__ = [
    "MonitorNotice",
    "MonitorVerdict",
    "NormalizedSupervisorEvent",
    "NoticeCode",
    "NoticeSeverity",
    "RUNTIME_TEMPORAL_COUNTEREXAMPLE_SCHEMA",
    "RUNTIME_TEMPORAL_MONITOR_VERSION",
    "RUNTIME_TEMPORAL_POLICY_SCHEMA",
    "RUNTIME_TEMPORAL_REOPEN_SCHEMA",
    "RUNTIME_TEMPORAL_REPORT_SCHEMA",
    "RuntimeTemporalCounterexample",
    "RuntimeTemporalMonitor",
    "RuntimeTemporalMonitorConfig",
    "RuntimeTemporalMonitorResult",
    "RuntimeTemporalPolicy",
    "SupervisorRuntimeTemporalMonitor",
    "TemporalCounterexample",
    "TemporalMonitorConfig",
    "TemporalMonitorPolicy",
    "TemporalMonitorReport",
    "TemporalMonitorResult",
    "TemporalPartitionKey",
    "TemporalProperty",
    "TemporalPropertyKind",
    "TemporalViolation",
    "TracePartition",
    "WorkReopenRequest",
    "load_runtime_temporal_counterexamples",
    "load_temporal_counterexamples",
    "monitor_event_logs",
    "monitor_event_trace",
    "monitor_supervisor_event_logs",
    "monitor_supervisor_events",
]
