"""DuckDB-backed authoritative event, audit, log, metric, and cursor store.

DQP-013 / DOEP-032 / DOEP-033 / DatabaseEventLog@1
==================================================

:class:`DatabaseEventLog` is the durable authority for domain events,
structured logs, metrics, explicit application audits, stream heads,
retention, integrity checkpoints, consumer cursors, idempotent
consumption, and materialized-state replay/recovery. Physical delivery
is at-least-once; logical transitions are exactly-once per
``(consumer_id, event_id)``. JSONL is an export adapter only: deleting
or tampering with an export has no authority effect.

Idempotent consumption and materialized-state replay are bindings of
this store, not a second event subsystem. A worker or model assertion
cannot mark an event consumed, skip suffix replay, or certify recovery.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Iterator

from ..control.control_contracts import (
    CursorReplayError,
    EventCursor,
    EventCursorError,
    EventPage,
)
from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_EVENT_LOG_INTERFACE: Final[str] = "DatabaseEventLog@1"
EVENT_CURSOR_INTERFACE: Final[str] = "EventCursor@1"
CONSUMER_CHECKPOINT_INTERFACE: Final[str] = "ConsumerCheckpoint@1"
IDEMPOTENT_EVENT_CONSUMPTION_BINDING: Final[str] = "IdempotentEventConsumption@1"
IDEMPOTENT_EVENT_CONSUMPTION_INTERFACE: Final[str] = (
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING
)
IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES: Final[tuple[str, ...]] = (
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    CONSUMER_CHECKPOINT_INTERFACE,
)
MATERIALIZED_STATE_REPLAY_BINDING: Final[str] = "MaterializedStateReplay@1"
MATERIALIZED_STATE_REPLAY_INTERFACE: Final[str] = (
    MATERIALIZED_STATE_REPLAY_BINDING
)
MATERIALIZED_STATE_REPLAY_CONSUMES: Final[tuple[str, ...]] = (
    DATABASE_EVENT_LOG_INTERFACE,
    EVENT_CURSOR_INTERFACE,
    CONSUMER_CHECKPOINT_INTERFACE,
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
)

DATABASE_EVENT_LOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-event-log@1"
)
CONSUMER_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/consumer-checkpoint@1"
)
IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/idempotent-event-consumption@1"
)
EVENT_CONSUMPTION_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/event-consumption-record@1"
)
INTEGRITY_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/integrity-checkpoint@1"
)
STREAM_HEAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/stream-head@1"
)
AUDIT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/audit-record@1"
)
JSONL_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/event-jsonl-export-receipt@1"
)
MATERIALIZED_STATE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/materialized-state-snapshot@1"
)
MATERIALIZED_STATE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/materialized-state-recovery@1"
)
MATERIALIZED_STATE_REPLAY_SCHEMA: Final[str] = (
    MATERIALIZED_STATE_RECOVERY_SCHEMA
)

DEFAULT_STREAM_ID: Final[str] = "stream:default"
DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-event-log"
DEFAULT_PAGE_LIMIT: Final[int] = 256
MAX_PAGE_LIMIT: Final[int] = 4_096
MAX_BODY_BYTES: Final[int] = 262_144
MAX_RECURSION_DEPTH: Final[int] = 8
MAX_AUDIT_NESTING: Final[int] = 2

_RESERVED_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "stream_id",
        "sequence",
        "global_sequence",
        "previous_event_id",
        "snapshot_id",
        "position",
    }
)

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    global_sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    previous_event_id VARCHAR NOT NULL DEFAULT '',
    snapshot_id VARCHAR NOT NULL DEFAULT '',
    redacted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE UNIQUE INDEX IF NOT EXISTS domain_events_stream_seq_uidx
    ON domain_events(stream_id, sequence);
CREATE UNIQUE INDEX IF NOT EXISTS domain_events_global_seq_uidx
    ON domain_events(global_sequence);
CREATE INDEX IF NOT EXISTS domain_events_task_idx
    ON domain_events(task_cid, sequence);

CREATE TABLE IF NOT EXISTS structured_logs (
    log_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL DEFAULT '',
    span_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS structured_logs_component_idx
    ON structured_logs(component, recorded_at);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id VARCHAR PRIMARY KEY,
    metric_name VARCHAR NOT NULL UNIQUE,
    unit VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_samples (
    sample_id VARCHAR PRIMARY KEY,
    metric_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    value_milli BIGINT NOT NULL,
    labels_json VARCHAR NOT NULL DEFAULT '{}',
    stratum VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS metric_samples_metric_idx
    ON metric_samples(metric_id, observed_at);

CREATE TABLE IF NOT EXISTS audit_records (
    audit_id VARCHAR PRIMARY KEY,
    event_id VARCHAR NOT NULL,
    actor_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    nesting_depth BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS audit_records_subject_idx
    ON audit_records(subject_kind, subject_id, recorded_at);

CREATE TABLE IF NOT EXISTS consumer_checkpoints (
    consumer_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL,
    cursor_token VARCHAR NOT NULL,
    checkpoint_digest VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS consumed_events (
    consumer_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    applied_at VARCHAR NOT NULL,
    consumption_digest VARCHAR NOT NULL,
    PRIMARY KEY (consumer_id, event_id)
);
CREATE INDEX IF NOT EXISTS consumed_events_consumer_seq_idx
    ON consumed_events(consumer_id, stream_id, sequence);

CREATE TABLE IF NOT EXISTS materialized_state_snapshots (
    snapshot_cid VARCHAR PRIMARY KEY,
    projection_id VARCHAR NOT NULL,
    stream_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL DEFAULT '',
    cursor_token VARCHAR NOT NULL,
    state_json VARCHAR NOT NULL,
    state_digest VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    snapshot_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS materialized_state_snapshots_projection_idx
    ON materialized_state_snapshots(projection_id, stream_id, position);

CREATE TABLE IF NOT EXISTS integrity_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    earliest_sequence BIGINT NOT NULL,
    latest_sequence BIGINT NOT NULL,
    event_count BIGINT NOT NULL,
    chain_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS stream_heads (
    stream_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    latest_sequence BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL DEFAULT '',
    global_sequence BIGINT NOT NULL DEFAULT 0,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseEventLogError(RuntimeError):
    """Base error for database event-log failures."""


class DatabaseEventLogConflictError(DatabaseEventLogError):
    """Duplicate identity, sequence conflict, or stale cursor."""


class DatabaseEventLogIntegrityError(DatabaseEventLogError):
    """Chain, digest, or retention integrity failure."""


class DatabaseEventLogBoundsError(DatabaseEventLogError, ValueError):
    """Payload, page, or recursion bound exceeded."""


class DatabaseEventLogNotOpenError(DatabaseEventLogError):
    """Operation requires an open event log."""


class DuckDBUnavailableError(DatabaseEventLogError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class LogSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(str, Enum):
    APPEND = "append"
    RETAIN = "retain"
    EXPORT = "export"
    CHECKPOINT = "checkpoint"
    REDACT = "redact"
    POLL = "poll"
    CONSUME = "consume"
    SNAPSHOT = "snapshot"
    RECOVER = "recover"


class ConsumptionStatus(str, Enum):
    """Durable consumption-row status. Pending rows are retried, not skipped."""

    PENDING = "pending"
    APPLIED = "applied"


class ConsumptionOutcome(str, Enum):
    """Logical result of one consume attempt against one event identity."""

    APPLIED = "applied"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class MaterializedSnapshotStatus(str, Enum):
    """Durable snapshot-row status. Pending rows are ignored on recovery."""

    PENDING = "pending"
    COMMITTED = "committed"


class MaterializedRecoveryOutcome(str, Enum):
    """Logical result of one materialized-state recovery."""

    RECOVERED = "recovered"
    IDEMPOTENT_REPLAY = "idempotent_replay"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseEventLogError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseEventLogError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseEventLogBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseEventLogBoundsError(f"{name} must be a positive integer")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        # Floats and other non-canonical values are stringified deterministically.
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_body(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    depth: int = 0,
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise DatabaseEventLogBoundsError(
            f"event body exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    for key in _RESERVED_BODY_KEYS:
        raw.pop(key, None)
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseEventLogError("event body must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DatabaseEventLogBoundsError(
            f"event body exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return cleaned


def _event_identity(value: Mapping[str, Any]) -> str:
    body = {
        key: item
        for key, item in value.items()
        if key != "event_id"
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    # DuckDBRow and sqlite3.Row support key iteration.
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split a SQL script into single statements (no procedure bodies)."""

    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        # Drop pure comment-only lines.
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerCheckpoint:
    """Durable consumer position bound to one stream and snapshot."""

    consumer_id: str
    cursor: EventCursor
    updated_at: str = ""
    schema: str = CONSUMER_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        object.__setattr__(
            self,
            "updated_at",
            _text(self.updated_at or _utc_iso(), "updated_at"),
        )
        if self.schema != CONSUMER_CHECKPOINT_SCHEMA:
            raise DatabaseEventLogError("unsupported consumer checkpoint schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": CONSUMER_CHECKPOINT_INTERFACE,
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "updated_at": self.updated_at,
        }

    @property
    def checkpoint_digest(self) -> str:
        body = {
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "updated_at": self.updated_at,
        }
        return _sha256_hex(_canonical_json(body).encode("utf-8"))


def _consumption_digest(
    *,
    consumer_id: str,
    event_id: str,
    stream_id: str,
    sequence: int,
    status: str,
) -> str:
    body = {
        "consumer_id": consumer_id,
        "event_id": event_id,
        "stream_id": stream_id,
        "sequence": sequence,
        "status": status,
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


@dataclass(frozen=True)
class EventConsumptionRecord:
    """One consumer's durable decision for one event identity."""

    consumer_id: str
    event_id: str
    stream_id: str
    sequence: int
    outcome: ConsumptionOutcome
    status: ConsumptionStatus = ConsumptionStatus.APPLIED
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    applied_at: str = ""
    schema: str = EVENT_CONSUMPTION_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        object.__setattr__(
            self, "sequence", _nonneg_int(self.sequence, "sequence")
        )
        if not isinstance(self.outcome, ConsumptionOutcome):
            object.__setattr__(
                self, "outcome", ConsumptionOutcome(str(self.outcome))
            )
        if not isinstance(self.status, ConsumptionStatus):
            object.__setattr__(
                self, "status", ConsumptionStatus(str(self.status))
            )
        object.__setattr__(
            self,
            "snapshot_id",
            _text(self.snapshot_id or DEFAULT_SNAPSHOT_ID, "snapshot_id"),
        )
        object.__setattr__(
            self,
            "applied_at",
            _text(self.applied_at or _utc_iso(), "applied_at"),
        )
        if self.schema != EVENT_CONSUMPTION_RECORD_SCHEMA:
            raise DatabaseEventLogError("unsupported event consumption schema")

    @property
    def consumption_digest(self) -> str:
        return _consumption_digest(
            consumer_id=self.consumer_id,
            event_id=self.event_id,
            stream_id=self.stream_id,
            sequence=self.sequence,
            status=self.status.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
            "consumer_id": self.consumer_id,
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "outcome": self.outcome.value,
            "status": self.status.value,
            "snapshot_id": self.snapshot_id,
            "applied_at": self.applied_at,
            "consumption_digest": self.consumption_digest,
            "worker_assertion_is_authority": False,
        }


@dataclass(frozen=True)
class EventConsumptionPage:
    """Bounded consume page: applied identities, durable cursor, checkpoint."""

    consumer_id: str
    records: tuple[EventConsumptionRecord, ...]
    next_cursor: EventCursor
    checkpoint: ConsumerCheckpoint
    has_more: bool = False
    schema: str = IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        if not isinstance(self.records, tuple):
            object.__setattr__(self, "records", tuple(self.records))
        if not isinstance(self.next_cursor, EventCursor):
            raise TypeError("next_cursor must be an EventCursor")
        if not isinstance(self.checkpoint, ConsumerCheckpoint):
            raise TypeError("checkpoint must be a ConsumerCheckpoint")
        if not isinstance(self.has_more, bool):
            raise DatabaseEventLogError("has_more must be a boolean")
        if self.schema != IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA:
            raise DatabaseEventLogError(
                "unsupported idempotent event consumption schema"
            )

    @property
    def applied_event_ids(self) -> tuple[str, ...]:
        return tuple(
            record.event_id
            for record in self.records
            if record.outcome is ConsumptionOutcome.APPLIED
        )

    @property
    def replayed_event_ids(self) -> tuple[str, ...]:
        return tuple(
            record.event_id
            for record in self.records
            if record.outcome is ConsumptionOutcome.IDEMPOTENT_REPLAY
        )

    @property
    def cursor(self) -> EventCursor:
        return self.next_cursor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": IDEMPOTENT_EVENT_CONSUMPTION_INTERFACE,
            "binding": IDEMPOTENT_EVENT_CONSUMPTION_BINDING,
            "consumes": list(IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES),
            "carrier": DATABASE_EVENT_LOG_INTERFACE,
            "consumer_id": self.consumer_id,
            "records": [record.to_dict() for record in self.records],
            "applied_event_ids": list(self.applied_event_ids),
            "replayed_event_ids": list(self.replayed_event_ids),
            "next_cursor": self.next_cursor.to_record(),
            "checkpoint": self.checkpoint.to_dict(),
            "has_more": self.has_more,
            "worker_assertion_is_authority": False,
        }


def _projection_consumer_id(projection_id: str) -> str:
    return f"consumer:materialized:{projection_id}"


def _bounded_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}
    if not isinstance(state, Mapping) or isinstance(state, (str, bytes)):
        raise DatabaseEventLogError("materialized state must project to an object")
    raw = dict(state)
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DatabaseEventLogBoundsError(
            f"materialized state exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return raw


def _state_digest(state: Mapping[str, Any]) -> str:
    return _sha256_hex(_canonical_json(dict(state)).encode("utf-8"))


def _snapshot_cid(
    *,
    projection_id: str,
    cursor: EventCursor,
    state: Mapping[str, Any],
) -> str:
    body = {
        "projection_id": projection_id,
        "cursor": cursor.to_record(),
        "state_digest": _state_digest(state),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def _snapshot_digest(
    *,
    snapshot_cid: str,
    projection_id: str,
    cursor: EventCursor,
    state: Mapping[str, Any],
    status: str,
) -> str:
    body = {
        "snapshot_cid": snapshot_cid,
        "projection_id": projection_id,
        "cursor": cursor.to_record(),
        "state_digest": _state_digest(state),
        "status": status,
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


@dataclass(frozen=True)
class MaterializedStateSnapshot:
    """Durable projection snapshot bound to one stream cursor.

    Pending rows are crash-window markers and never recovery authority.
    Events remain the source of truth; a snapshot is a restart hint.
    """

    projection_id: str
    cursor: EventCursor
    state: Mapping[str, Any]
    status: MaterializedSnapshotStatus = MaterializedSnapshotStatus.COMMITTED
    recorded_at: str = ""
    snapshot_cid: str = ""
    schema: str = MATERIALIZED_STATE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projection_id", _text(self.projection_id, "projection_id")
        )
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        cleaned = _bounded_state(self.state)
        object.__setattr__(self, "state", MappingProxyType(cleaned))
        if not isinstance(self.status, MaterializedSnapshotStatus):
            object.__setattr__(
                self, "status", MaterializedSnapshotStatus(str(self.status))
            )
        object.__setattr__(
            self,
            "recorded_at",
            _text(self.recorded_at or _utc_iso(), "recorded_at"),
        )
        computed_cid = _snapshot_cid(
            projection_id=self.projection_id,
            cursor=self.cursor,
            state=cleaned,
        )
        selected_cid = _text(
            self.snapshot_cid or computed_cid, "snapshot_cid"
        )
        if self.snapshot_cid and selected_cid != computed_cid:
            raise DatabaseEventLogIntegrityError(
                "materialized snapshot_cid does not match content identity"
            )
        object.__setattr__(self, "snapshot_cid", computed_cid)
        if self.schema != MATERIALIZED_STATE_SNAPSHOT_SCHEMA:
            raise DatabaseEventLogError(
                "unsupported materialized state snapshot schema"
            )

    @property
    def state_digest(self) -> str:
        return _state_digest(self.state)

    @property
    def snapshot_digest(self) -> str:
        return _snapshot_digest(
            snapshot_cid=self.snapshot_cid,
            projection_id=self.projection_id,
            cursor=self.cursor,
            state=self.state,
            status=self.status.value,
        )

    @property
    def consumer_id(self) -> str:
        return _projection_consumer_id(self.projection_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": MATERIALIZED_STATE_REPLAY_INTERFACE,
            "binding": MATERIALIZED_STATE_REPLAY_BINDING,
            "consumes": list(MATERIALIZED_STATE_REPLAY_CONSUMES),
            "carrier": DATABASE_EVENT_LOG_INTERFACE,
            "snapshot_cid": self.snapshot_cid,
            "projection_id": self.projection_id,
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "state": dict(self.state),
            "status": self.status.value,
            "recorded_at": self.recorded_at,
            "state_digest": self.state_digest,
            "snapshot_digest": self.snapshot_digest,
            "authoritative": False,
            "worker_assertion_is_authority": False,
        }


@dataclass(frozen=True)
class MaterializedStateRecovery:
    """Reconstructed projection: committed snapshot plus suffix replay."""

    projection_id: str
    state: Mapping[str, Any]
    cursor: EventCursor
    snapshot: MaterializedStateSnapshot | None
    applied_event_ids: tuple[str, ...]
    outcome: MaterializedRecoveryOutcome
    recovered_from_snapshot: bool = False
    checkpoint: ConsumerCheckpoint | None = None
    has_more: bool = False
    schema: str = MATERIALIZED_STATE_RECOVERY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projection_id", _text(self.projection_id, "projection_id")
        )
        cleaned = _bounded_state(self.state)
        object.__setattr__(self, "state", MappingProxyType(cleaned))
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        if self.snapshot is not None and not isinstance(
            self.snapshot, MaterializedStateSnapshot
        ):
            raise TypeError("snapshot must be a MaterializedStateSnapshot or None")
        if not isinstance(self.applied_event_ids, tuple):
            object.__setattr__(
                self, "applied_event_ids", tuple(self.applied_event_ids)
            )
        if not isinstance(self.outcome, MaterializedRecoveryOutcome):
            object.__setattr__(
                self, "outcome", MaterializedRecoveryOutcome(str(self.outcome))
            )
        if not isinstance(self.recovered_from_snapshot, bool):
            raise DatabaseEventLogError(
                "recovered_from_snapshot must be a boolean"
            )
        if self.checkpoint is not None and not isinstance(
            self.checkpoint, ConsumerCheckpoint
        ):
            raise TypeError("checkpoint must be a ConsumerCheckpoint or None")
        if not isinstance(self.has_more, bool):
            raise DatabaseEventLogError("has_more must be a boolean")
        if self.schema != MATERIALIZED_STATE_RECOVERY_SCHEMA:
            raise DatabaseEventLogError(
                "unsupported materialized state recovery schema"
            )

    @property
    def state_digest(self) -> str:
        return _state_digest(self.state)

    @property
    def consumer_id(self) -> str:
        return _projection_consumer_id(self.projection_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": MATERIALIZED_STATE_REPLAY_INTERFACE,
            "binding": MATERIALIZED_STATE_REPLAY_BINDING,
            "consumes": list(MATERIALIZED_STATE_REPLAY_CONSUMES),
            "carrier": DATABASE_EVENT_LOG_INTERFACE,
            "projection_id": self.projection_id,
            "consumer_id": self.consumer_id,
            "state": dict(self.state),
            "cursor": self.cursor.to_record(),
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "checkpoint": None
            if self.checkpoint is None
            else self.checkpoint.to_dict(),
            "applied_event_ids": list(self.applied_event_ids),
            "outcome": self.outcome.value,
            "recovered_from_snapshot": self.recovered_from_snapshot,
            "has_more": self.has_more,
            "state_digest": self.state_digest,
            "worker_assertion_is_authority": False,
        }


@dataclass(frozen=True)
class StreamHead:
    """Authoritative head of one event stream."""

    stream_id: str
    snapshot_id: str
    latest_sequence: int
    last_event_id: str = ""
    global_sequence: int = 0
    updated_at: str = ""
    schema: str = STREAM_HEAD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "latest_sequence",
            _nonneg_int(self.latest_sequence, "latest_sequence"),
        )
        object.__setattr__(
            self,
            "global_sequence",
            _nonneg_int(self.global_sequence, "global_sequence"),
        )
        object.__setattr__(
            self,
            "last_event_id",
            _text(self.last_event_id, "last_event_id", required=False),
        )
        object.__setattr__(
            self,
            "updated_at",
            _text(self.updated_at or _utc_iso(), "updated_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stream_id": self.stream_id,
            "snapshot_id": self.snapshot_id,
            "latest_sequence": self.latest_sequence,
            "last_event_id": self.last_event_id,
            "global_sequence": self.global_sequence,
            "updated_at": self.updated_at,
        }

    def as_cursor(self) -> EventCursor:
        if self.latest_sequence == 0:
            return EventCursor.initial(
                self.stream_id, snapshot_id=self.snapshot_id
            )
        return EventCursor(
            stream_id=self.stream_id,
            snapshot_id=self.snapshot_id,
            position=self.latest_sequence,
            last_event_id=self.last_event_id,
        )


@dataclass(frozen=True)
class IntegrityCheckpoint:
    """Content-addressed digest over a retained stream window."""

    checkpoint_id: str
    stream_id: str
    earliest_sequence: int
    latest_sequence: int
    event_count: int
    chain_digest: str
    recorded_at: str = ""
    schema: str = INTEGRITY_CHECKPOINT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checkpoint_id": self.checkpoint_id,
            "stream_id": self.stream_id,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "event_count": self.event_count,
            "chain_digest": self.chain_digest,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class DomainEvent:
    """Immutable append-only domain event projection."""

    event_id: str
    stream_id: str
    sequence: int
    global_sequence: int
    event_type: str
    recorded_at: str
    body: Mapping[str, Any]
    previous_event_id: str = ""
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    task_cid: str = ""
    attempt_id: str = ""
    session_id: str = ""
    redacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "global_sequence": self.global_sequence,
            "type": self.event_type,
            "event_type": self.event_type,
            "recorded_at": self.recorded_at,
            "timestamp": self.recorded_at,
            "previous_event_id": self.previous_event_id,
            "snapshot_id": self.snapshot_id,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "redacted": self.redacted,
            "body": dict(self.body),
            **dict(self.body),
        }


@dataclass(frozen=True)
class JsonlExportReceipt:
    """Non-authoritative receipt for a JSONL export render."""

    export_path: str
    stream_id: str
    event_count: int
    earliest_sequence: int
    latest_sequence: int
    content_digest: str
    recorded_at: str
    authority: str = "export_only"
    schema: str = JSONL_EXPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "export_path": self.export_path,
            "stream_id": self.stream_id,
            "event_count": self.event_count,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "content_digest": self.content_digest,
            "recorded_at": self.recorded_at,
            "authority": self.authority,
            "authoritative": False,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseEventLog:
    """Append-only DuckDB event authority with cursor polling and export."""

    INTERFACE: Final[str] = DATABASE_EVENT_LOG_INTERFACE
    IDEMPOTENT_EVENT_CONSUMPTION_BINDING: Final[str] = (
        IDEMPOTENT_EVENT_CONSUMPTION_BINDING
    )
    MATERIALIZED_STATE_REPLAY_BINDING: Final[str] = (
        MATERIALIZED_STATE_REPLAY_BINDING
    )

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseEventLog; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._audit_depth = 0
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseEventLog":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            # Best-effort column upgrades when control-plane schema pre-exists.
            self._ensure_optional_columns(connection)
            for key, value in (
                ("interface", DATABASE_EVENT_LOG_INTERFACE),
                ("schema", DATABASE_EVENT_LOG_SCHEMA),
                ("snapshot_id", self._snapshot_id),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO event_log_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseEventLog":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseEventLogNotOpenError("DatabaseEventLog is not open")
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        """Persist autocommit-style writes when no explicit txn is open."""

        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    @staticmethod
    def _ensure_optional_columns(connection: Any) -> None:
        """Add extension columns when domain_events already exists without them."""

        existing: set[str] = set()
        try:
            rows = connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE lower(table_name) = 'domain_events'
                """
            ).fetchall()
            for row in rows:
                mapping = _row_mapping(row)
                name = mapping.get("column_name") or mapping.get("COLUMN_NAME")
                if name:
                    existing.add(str(name).casefold())
                elif mapping:
                    existing.add(str(next(iter(mapping.values()))).casefold())
        except Exception:
            try:
                rows = connection.execute("DESCRIBE domain_events").fetchall()
            except Exception:
                return
            for row in rows:
                mapping = _row_mapping(row)
                values = list(mapping.values())
                if values:
                    existing.add(str(values[0]).casefold())
        alterations = []
        if "previous_event_id" not in existing:
            alterations.append(
                "ALTER TABLE domain_events ADD COLUMN previous_event_id "
                "VARCHAR DEFAULT ''"
            )
        if "snapshot_id" not in existing:
            alterations.append(
                "ALTER TABLE domain_events ADD COLUMN snapshot_id "
                "VARCHAR DEFAULT ''"
            )
        if "redacted" not in existing:
            alterations.append(
                "ALTER TABLE domain_events ADD COLUMN redacted "
                "BOOLEAN DEFAULT FALSE"
            )
        for statement in alterations:
            try:
                connection.execute(statement)
            except Exception:
                pass

    # -- append --------------------------------------------------------------

    def append_event(
        self,
        event_type: str,
        body: Mapping[str, Any] | None = None,
        *,
        stream_id: str = DEFAULT_STREAM_ID,
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
        event_id: str | None = None,
        recorded_at: str | None = None,
        redact: bool | None = None,
    ) -> DomainEvent:
        """Append one typed event. Sequence and event_id are immutable after commit."""

        selected_type = _text(event_type, "event_type")
        selected_stream = _text(stream_id, "stream_id")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact)
        stamp = _text(recorded_at or _utc_iso(), "recorded_at")

        with self._lock:
            connection = self._require()
            head = self._load_head(connection, selected_stream)
            sequence = int(head.latest_sequence) + 1
            global_sequence = self._next_global_sequence(connection)
            previous_event_id = head.last_event_id
            identity_material = {
                "stream_id": selected_stream,
                "snapshot_id": self._snapshot_id,
                "sequence": sequence,
                "global_sequence": global_sequence,
                "event_type": selected_type,
                "task_cid": _text(task_cid, "task_cid", required=False),
                "attempt_id": _text(attempt_id, "attempt_id", required=False),
                "session_id": _text(session_id, "session_id", required=False),
                "recorded_at": stamp,
                "previous_event_id": previous_event_id,
                "body": payload,
                "redacted": do_redact,
            }
            computed_id = _event_identity(identity_material)
            selected_id = _text(event_id or computed_id, "event_id")
            existing = self._get_event_row(connection, selected_id)
            if existing is not None:
                # Exact identity replay coalesces. Reusing an event_id for a
                # different payload fails closed.
                prior = self._row_to_event(existing)
                same_payload = (
                    prior.event_type == selected_type
                    and dict(prior.body) == payload
                    and prior.stream_id == selected_stream
                )
                if same_payload:
                    return prior
                raise DatabaseEventLogConflictError(
                    "event_id already exists with a different payload"
                )
            if event_id is not None and selected_id != computed_id:
                # Client-supplied IDs must equal content identity for new rows.
                raise DatabaseEventLogConflictError(
                    "supplied event_id does not match content identity"
                )

            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(
                    """
                    INSERT INTO domain_events (
                        event_id, stream_id, sequence, global_sequence,
                        event_type, task_cid, attempt_id, session_id,
                        recorded_at, body_json, previous_event_id,
                        snapshot_id, redacted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_id,
                        selected_stream,
                        sequence,
                        global_sequence,
                        selected_type,
                        _text(task_cid, "task_cid", required=False),
                        _text(attempt_id, "attempt_id", required=False),
                        _text(session_id, "session_id", required=False),
                        stamp,
                        _canonical_json(payload),
                        previous_event_id,
                        self._snapshot_id,
                        do_redact,
                    ],
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO stream_heads (
                        stream_id, snapshot_id, latest_sequence, last_event_id,
                        global_sequence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_stream,
                        self._snapshot_id,
                        sequence,
                        selected_id,
                        global_sequence,
                        stamp,
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise

            return DomainEvent(
                event_id=selected_id,
                stream_id=selected_stream,
                sequence=sequence,
                global_sequence=global_sequence,
                event_type=selected_type,
                recorded_at=stamp,
                body=MappingProxyType(payload),
                previous_event_id=previous_event_id,
                snapshot_id=self._snapshot_id,
                task_cid=_text(task_cid, "task_cid", required=False),
                attempt_id=_text(attempt_id, "attempt_id", required=False),
                session_id=_text(session_id, "session_id", required=False),
                redacted=do_redact,
            )

    def append_log(
        self,
        message: str,
        *,
        severity: LogSeverity | str = LogSeverity.INFO,
        component: str = "agent_supervisor",
        body: Mapping[str, Any] | None = None,
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        span_id: str = "",
    ) -> dict[str, Any]:
        """Append one structured log row (not a domain-event authority record)."""

        text = _text(message, "message")
        if isinstance(severity, LogSeverity):
            level = severity.value
        else:
            level = _text(severity, "severity").casefold()
            try:
                level = LogSeverity(level).value
            except ValueError as exc:
                raise DatabaseEventLogError(
                    f"unknown log severity {severity!r}"
                ) from exc
        payload = _bounded_body(body, redact=self._auto_redact)
        stamp = _utc_iso()
        log_id = _sha256_hex(
            _canonical_json(
                {
                    "severity": level,
                    "component": component,
                    "message": text,
                    "recorded_at": stamp,
                    "body": payload,
                }
            ).encode("utf-8")
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT OR IGNORE INTO structured_logs (
                    log_id, severity, component, trace_id, span_id,
                    task_cid, attempt_id, session_id, recorded_at,
                    message, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    log_id,
                    level,
                    _text(component, "component"),
                    _text(trace_id, "trace_id", required=False),
                    _text(span_id, "span_id", required=False),
                    _text(task_cid, "task_cid", required=False),
                    _text(attempt_id, "attempt_id", required=False),
                    _text(session_id, "session_id", required=False),
                    stamp,
                    text,
                    _canonical_json(payload),
                ],
            )
            self._commit_if_idle(connection)
        return {
            "log_id": log_id,
            "severity": level,
            "component": component,
            "message": text,
            "recorded_at": stamp,
            "body": payload,
        }

    def append_metric_sample(
        self,
        metric_name: str,
        value_milli: int,
        *,
        unit: str = "count",
        description: str = "",
        labels: Mapping[str, Any] | None = None,
        stratum: str = "",
    ) -> dict[str, Any]:
        """Register a metric (if needed) and append one sample."""

        name = _text(metric_name, "metric_name")
        if isinstance(value_milli, bool) or not isinstance(value_milli, int):
            raise DatabaseEventLogBoundsError("value_milli must be an integer")
        stamp = _utc_iso()
        metric_id = _sha256_hex(name.encode("utf-8"))
        labels_payload = _bounded_body(labels, redact=self._auto_redact)
        sample_id = _sha256_hex(
            _canonical_json(
                {
                    "metric_id": metric_id,
                    "observed_at": stamp,
                    "value_milli": value_milli,
                    "labels": labels_payload,
                    "stratum": stratum,
                }
            ).encode("utf-8")
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                INSERT OR IGNORE INTO metrics (
                    metric_id, metric_name, unit, description, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    metric_id,
                    name,
                    _text(unit, "unit"),
                    _text(description, "description", required=False),
                    stamp,
                ],
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO metric_samples (
                    sample_id, metric_id, observed_at, value_milli,
                    labels_json, stratum
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    sample_id,
                    metric_id,
                    stamp,
                    value_milli,
                    _canonical_json(labels_payload),
                    _text(stratum, "stratum", required=False),
                ],
            )
            self._commit_if_idle(connection)
        return {
            "sample_id": sample_id,
            "metric_id": metric_id,
            "metric_name": name,
            "value_milli": value_milli,
            "observed_at": stamp,
            "labels": labels_payload,
        }

    def append_audit(
        self,
        action: AuditAction | str,
        *,
        actor_id: str,
        subject_kind: str,
        subject_id: str,
        body: Mapping[str, Any] | None = None,
        emit_event: bool = True,
    ) -> dict[str, Any]:
        """Record an explicit application audit (never inferred from Quack logs).

        Recursive audit emission is depth-bounded so audit-of-audit cannot
        recurse unbounded.
        """

        if self._audit_depth >= MAX_AUDIT_NESTING:
            raise DatabaseEventLogBoundsError(
                f"audit nesting exceeds bound {MAX_AUDIT_NESTING}"
            )
        if isinstance(action, AuditAction):
            action_text = action.value
        else:
            action_text = _text(action, "action")
        actor = _text(actor_id, "actor_id")
        kind = _text(subject_kind, "subject_kind")
        subject = _text(subject_id, "subject_id")
        payload = _bounded_body(body, redact=self._auto_redact)
        stamp = _utc_iso()
        event_id = ""
        self._audit_depth += 1
        try:
            if emit_event:
                event = self.append_event(
                    f"audit.{action_text}",
                    {
                        "actor_id": actor,
                        "subject_kind": kind,
                        "subject_id": subject,
                        "action": action_text,
                        "details": payload,
                    },
                    stream_id="stream:audit",
                )
                event_id = event.event_id
            audit_id = _sha256_hex(
                _canonical_json(
                    {
                        "event_id": event_id,
                        "actor_id": actor,
                        "action": action_text,
                        "subject_kind": kind,
                        "subject_id": subject,
                        "recorded_at": stamp,
                        "body": payload,
                        "nesting_depth": self._audit_depth,
                    }
                ).encode("utf-8")
            )
            with self._lock:
                connection = self._require()
                connection.execute(
                    """
                    INSERT OR IGNORE INTO audit_records (
                        audit_id, event_id, actor_id, action, subject_kind,
                        subject_id, recorded_at, body_json, nesting_depth
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        audit_id,
                        event_id,
                        actor,
                        action_text,
                        kind,
                        subject,
                        stamp,
                        _canonical_json(payload),
                        self._audit_depth,
                    ],
                )
                self._commit_if_idle(connection)
            return {
                "schema": AUDIT_RECORD_SCHEMA,
                "audit_id": audit_id,
                "event_id": event_id,
                "actor_id": actor,
                "action": action_text,
                "subject_kind": kind,
                "subject_id": subject,
                "recorded_at": stamp,
                "body": payload,
                "nesting_depth": self._audit_depth,
            }
        finally:
            self._audit_depth -= 1

    # -- read / poll ---------------------------------------------------------

    def initial_cursor(self, stream_id: str = DEFAULT_STREAM_ID) -> EventCursor:
        return EventCursor.initial(
            _text(stream_id, "stream_id"), snapshot_id=self._snapshot_id
        )

    def stream_head(self, stream_id: str = DEFAULT_STREAM_ID) -> StreamHead:
        with self._lock:
            return self._load_head(self._require(), _text(stream_id, "stream_id"))

    def latest_cursor(self, stream_id: str = DEFAULT_STREAM_ID) -> EventCursor:
        return self.stream_head(stream_id).as_cursor()

    def get_event(self, event_id: str) -> DomainEvent | None:
        with self._lock:
            row = self._get_event_row(self._require(), _text(event_id, "event_id"))
            return None if row is None else self._row_to_event(row)

    def poll(
        self,
        cursor: EventCursor | Mapping[str, Any] | str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        stream_id: str | None = None,
    ) -> EventPage:
        """Replay at most ``limit`` events strictly after ``cursor`` (coalesced)."""

        page_limit = _positive_int(limit, "limit")
        if page_limit > MAX_PAGE_LIMIT:
            raise DatabaseEventLogBoundsError(
                f"limit exceeds the {MAX_PAGE_LIMIT} bound"
            )
        selected = self._coerce_cursor(cursor)
        selected_stream = _text(
            stream_id or selected.stream_id, "stream_id"
        )
        with self._lock:
            connection = self._require()
            if selected_stream != DEFAULT_STREAM_ID:
                known_stream = connection.execute(
                    """
                    SELECT 1 FROM stream_heads WHERE stream_id = ?
                    UNION ALL
                    SELECT 1 FROM domain_events WHERE stream_id = ?
                    LIMIT 1
                    """,
                    [selected_stream, selected_stream],
                ).fetchone()
                if known_stream is None:
                    raise CursorReplayError(
                        "event cursor stream is not registered in this event log"
                    )
            head = self._load_head(connection, selected_stream)
            earliest = self._earliest_sequence(connection, selected_stream)
            try:
                selected.assert_replayable(
                    stream_id=selected_stream,
                    earliest_position=earliest,
                    latest_position=head.latest_sequence,
                    snapshot_id=self._snapshot_id,
                )
            except CursorReplayError:
                raise
            if selected.position and selected.last_event_id:
                anchor = self._event_at_sequence(
                    connection, selected_stream, selected.position
                )
                if anchor is None:
                    raise CursorReplayError(
                        "event cursor predates the retained replay window"
                    )
                if str(anchor.get("event_id") or "") != selected.last_event_id:
                    raise CursorReplayError(
                        "event cursor last_event_id does not match the stream"
                    )

            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence,
                       event_type, task_cid, attempt_id, session_id,
                       recorded_at, body_json, previous_event_id,
                       snapshot_id, redacted
                FROM domain_events
                WHERE stream_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                [selected_stream, selected.position, page_limit + 1],
            ).fetchall()

            # Coalesce exact identity duplicates (should be unique by index).
            seen: dict[int, str] = {}
            events: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                sequence = int(mapping["sequence"])
                event_id = str(mapping["event_id"])
                known = seen.get(sequence)
                if known == event_id:
                    continue
                if known is not None and known != event_id:
                    raise DatabaseEventLogIntegrityError(
                        f"conflicting identities at sequence {sequence}"
                    )
                seen[sequence] = event_id
                events.append(self._row_to_event(mapping).to_dict())

            has_more = len(events) > page_limit
            page_events = events[:page_limit]
            next_cursor = selected
            if page_events:
                last = page_events[-1]
                next_cursor = selected.advance(
                    position=int(last["sequence"]),
                    event_id=str(last["event_id"]),
                    snapshot_id=self._snapshot_id,
                )
            return EventPage(
                events=tuple(page_events),
                next_cursor=next_cursor,
                has_more=has_more,
            )

    def replay(
        self,
        cursor: EventCursor | Mapping[str, Any] | str | None = None,
        *,
        stream_id: str = DEFAULT_STREAM_ID,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Iterator[dict[str, Any]]:
        """Yield all events after ``cursor`` via bounded polling pages."""

        current = (
            self.initial_cursor(stream_id)
            if cursor is None
            else self._coerce_cursor(cursor)
        )
        while True:
            page = self.poll(current, limit=limit, stream_id=stream_id)
            for event in page.events:
                yield dict(event)
            current = page.next_cursor
            if not page.has_more or not page.events:
                break

    # -- consumer checkpoints / idempotent consumption -----------------------

    def consumer_cursor(
        self,
        consumer_id: str,
        stream_id: str = DEFAULT_STREAM_ID,
    ) -> EventCursor:
        """Return the durable consumer cursor, or the stream's initial cursor."""

        checkpoint = self.load_consumer_checkpoint(consumer_id)
        selected_stream = _text(stream_id, "stream_id")
        if checkpoint is None:
            return self.initial_cursor(selected_stream)
        if checkpoint.cursor.stream_id != selected_stream:
            raise DatabaseEventLogConflictError(
                "consumer checkpoint is bound to a different stream"
            )
        return checkpoint.cursor

    def save_consumer_checkpoint(
        self,
        consumer_id: str,
        cursor: EventCursor | Mapping[str, Any] | str,
    ) -> ConsumerCheckpoint:
        selected = self._coerce_cursor(cursor)
        consumer = _text(consumer_id, "consumer_id")
        with self._lock:
            connection = self._require()
            return self._save_consumer_checkpoint_unlocked(
                connection, consumer, selected
            )

    def load_consumer_checkpoint(
        self, consumer_id: str
    ) -> ConsumerCheckpoint | None:
        with self._lock:
            return self._load_consumer_checkpoint_unlocked(
                self._require(), _text(consumer_id, "consumer_id")
            )

    def event_consumed(self, consumer_id: str, event_id: str) -> bool:
        """Return True when this consumer has applied ``event_id``."""

        with self._lock:
            row = self._get_consumed_row(
                self._require(),
                _text(consumer_id, "consumer_id"),
                _text(event_id, "event_id"),
            )
        return row is not None and str(row.get("status") or "") == (
            ConsumptionStatus.APPLIED.value
        )

    def consumed_event_ids(self, consumer_id: str) -> tuple[str, ...]:
        """Return applied event identities for ``consumer_id`` in stream order."""

        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT event_id FROM consumed_events
                WHERE consumer_id = ? AND status = ?
                ORDER BY sequence ASC
                """,
                [
                    _text(consumer_id, "consumer_id"),
                    ConsumptionStatus.APPLIED.value,
                ],
            ).fetchall()
        return tuple(str(_row_mapping(row)["event_id"]) for row in rows)

    def consume(
        self,
        consumer_id: str,
        *,
        stream_id: str = DEFAULT_STREAM_ID,
        limit: int = DEFAULT_PAGE_LIMIT,
        handler: Callable[[Mapping[str, Any]], Any] | None = None,
        cursor: EventCursor | Mapping[str, Any] | str | None = None,
        worker_assertion: bool = False,
    ) -> EventConsumptionPage:
        """Apply the next page of events with exactly-once logical transitions.

        Physical delivery remains at-least-once: a crash after the handler
        runs but before the applied row commits may re-invoke the handler.
        After an applied row exists, the same ``event_id`` is an idempotent
        replay and the handler is not called again. ``worker_assertion`` is
        not consumption authority and cannot skip the durable table.
        """

        del worker_assertion  # Never authoritative; durable rows decide.
        consumer = _text(consumer_id, "consumer_id")
        selected_stream = _text(stream_id, "stream_id")
        page_limit = _positive_int(limit, "limit")
        if page_limit > MAX_PAGE_LIMIT:
            raise DatabaseEventLogBoundsError(
                f"limit exceeds the {MAX_PAGE_LIMIT} bound"
            )

        with self._lock:
            connection = self._require()
            stored = self._load_consumer_checkpoint_unlocked(
                connection, consumer
            )
            if stored is None:
                current = (
                    self._coerce_cursor(cursor)
                    if cursor is not None
                    else EventCursor.initial(
                        selected_stream, snapshot_id=self._snapshot_id
                    )
                )
                if current.stream_id != selected_stream:
                    raise DatabaseEventLogConflictError(
                        "supplied cursor stream does not match consume stream"
                    )
            else:
                if stored.cursor.stream_id != selected_stream:
                    raise DatabaseEventLogConflictError(
                        "consumer checkpoint is bound to a different stream"
                    )
                current = stored.cursor
                if cursor is not None:
                    supplied = self._coerce_cursor(cursor)
                    if not self._cursors_equivalent(supplied, current):
                        raise DatabaseEventLogConflictError(
                            "supplied cursor does not match the durable "
                            "consumer checkpoint"
                        )

        page = self.poll(current, limit=page_limit, stream_id=selected_stream)
        records: list[EventConsumptionRecord] = []
        last_cursor = current

        for event in page.events:
            event_id = _text(event.get("event_id"), "event_id")
            sequence = _nonneg_int(int(event.get("sequence") or 0), "sequence")
            event_stream = _text(
                event.get("stream_id") or selected_stream, "stream_id"
            )
            with self._lock:
                connection = self._require()
                existing = self._get_consumed_row(
                    connection, consumer, event_id
                )
                already_applied = (
                    existing is not None
                    and str(existing.get("status") or "")
                    == ConsumptionStatus.APPLIED.value
                )
                if already_applied:
                    stamp = str(existing.get("applied_at") or _utc_iso())
                    record = EventConsumptionRecord(
                        consumer_id=consumer,
                        event_id=event_id,
                        stream_id=event_stream,
                        sequence=sequence,
                        outcome=ConsumptionOutcome.IDEMPOTENT_REPLAY,
                        status=ConsumptionStatus.APPLIED,
                        snapshot_id=self._snapshot_id,
                        applied_at=stamp,
                    )
                    last_cursor = current.advance(
                        position=sequence,
                        event_id=event_id,
                        snapshot_id=self._snapshot_id,
                    )
                    records.append(record)
                    continue
                stamp = _utc_iso()
                self._upsert_consumed_unlocked(
                    connection,
                    consumer_id=consumer,
                    event_id=event_id,
                    stream_id=event_stream,
                    sequence=sequence,
                    status=ConsumptionStatus.PENDING,
                    applied_at=stamp,
                )
                self._commit_if_idle(connection)

            if handler is not None:
                try:
                    handler(event)
                except Exception:
                    # Leave the pending row; checkpoint stays at last_cursor.
                    if records:
                        self.save_consumer_checkpoint(consumer, last_cursor)
                    raise

            with self._lock:
                connection = self._require()
                stamp = _utc_iso()
                self._upsert_consumed_unlocked(
                    connection,
                    consumer_id=consumer,
                    event_id=event_id,
                    stream_id=event_stream,
                    sequence=sequence,
                    status=ConsumptionStatus.APPLIED,
                    applied_at=stamp,
                )
                last_cursor = current.advance(
                    position=sequence,
                    event_id=event_id,
                    snapshot_id=self._snapshot_id,
                )
                self._save_consumer_checkpoint_unlocked(
                    connection, consumer, last_cursor
                )
            records.append(
                EventConsumptionRecord(
                    consumer_id=consumer,
                    event_id=event_id,
                    stream_id=event_stream,
                    sequence=sequence,
                    outcome=ConsumptionOutcome.APPLIED,
                    status=ConsumptionStatus.APPLIED,
                    snapshot_id=self._snapshot_id,
                    applied_at=stamp,
                )
            )

        if page.events:
            checkpoint_cursor = last_cursor
        else:
            checkpoint_cursor = page.next_cursor
        checkpoint = self.save_consumer_checkpoint(consumer, checkpoint_cursor)
        return EventConsumptionPage(
            consumer_id=consumer,
            records=tuple(records),
            next_cursor=checkpoint.cursor,
            checkpoint=checkpoint,
            has_more=page.has_more,
        )

    # -- materialized-state replay / recovery --------------------------------

    def load_materialized_snapshot(
        self,
        projection_id: str,
        stream_id: str = DEFAULT_STREAM_ID,
    ) -> MaterializedStateSnapshot | None:
        """Return the latest committed snapshot, ignoring pending crash rows."""

        with self._lock:
            return self._load_latest_committed_snapshot_unlocked(
                self._require(),
                _text(projection_id, "projection_id"),
                _text(stream_id, "stream_id"),
            )

    def save_materialized_snapshot(
        self,
        projection_id: str,
        state: Mapping[str, Any] | None,
        cursor: EventCursor | Mapping[str, Any] | str,
        *,
        worker_assertion: bool = False,
    ) -> MaterializedStateSnapshot:
        """Persist a committed projection snapshot via a pending crash window.

        ``worker_assertion`` cannot skip digest, cursor, or rewind checks.
        """

        del worker_assertion
        selected = self._coerce_cursor(cursor)
        consumer_projection = _text(projection_id, "projection_id")
        payload = _bounded_state(state)
        with self._lock:
            connection = self._require()
            return self._commit_materialized_snapshot_unlocked(
                connection,
                projection_id=consumer_projection,
                state=payload,
                cursor=selected,
            )

    def recover_materialized_state(
        self,
        projection_id: str,
        *,
        reducer: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        initial_state: Mapping[str, Any] | None = None,
        stream_id: str = DEFAULT_STREAM_ID,
        limit: int = DEFAULT_PAGE_LIMIT,
        persist: bool = True,
        until_head: bool = True,
        worker_assertion: bool = False,
    ) -> MaterializedStateRecovery:
        """Rebuild projection state from the last committed snapshot plus suffix.

        Pending snapshot rows are ignored. Events after the snapshot cursor
        are folded through ``reducer``. A worker assertion cannot skip that
        suffix or certify recovery. Snapshots are restart hints; the event
        log remains authority. ``until_head`` replays every remaining page;
        ``materialize`` uses a single bounded page.
        """

        del worker_assertion
        selected_projection = _text(projection_id, "projection_id")
        selected_stream = _text(stream_id, "stream_id")
        page_limit = _positive_int(limit, "limit")
        if page_limit > MAX_PAGE_LIMIT:
            raise DatabaseEventLogBoundsError(
                f"limit exceeds the {MAX_PAGE_LIMIT} bound"
            )

        with self._lock:
            snapshot = self._load_latest_committed_snapshot_unlocked(
                self._require(), selected_projection
            )
        if snapshot is not None:
            if snapshot.cursor.stream_id != selected_stream:
                raise DatabaseEventLogConflictError(
                    "materialized snapshot is bound to a different stream"
                )
            # Fail closed when the snapshot cursor is outside the retained
            # window; poll() also enforces this, but recovery must not
            # silently start from origin after a retained snapshot.
            self.poll(snapshot.cursor, limit=1, stream_id=selected_stream)
            state = dict(snapshot.state)
            current = snapshot.cursor
            recovered_from_snapshot = True
        else:
            state = _bounded_state(initial_state)
            current = self.initial_cursor(selected_stream)
            recovered_from_snapshot = False

        applied: list[str] = []
        has_more = False
        while True:
            page = self.poll(current, limit=page_limit, stream_id=selected_stream)
            has_more = page.has_more
            if not page.events:
                current = page.next_cursor
                break
            if reducer is None:
                raise DatabaseEventLogError(
                    "reducer is required to replay events after a "
                    "materialized snapshot"
                )
            for event in page.events:
                folded = reducer(state, event)
                if not isinstance(folded, Mapping):
                    raise DatabaseEventLogError(
                        "materialized reducer must return an object"
                    )
                state = _bounded_state(folded)
                applied.append(_text(event.get("event_id"), "event_id"))
            current = page.next_cursor
            if not page.has_more or not until_head:
                break

        if applied:
            outcome = MaterializedRecoveryOutcome.RECOVERED
        else:
            outcome = MaterializedRecoveryOutcome.IDEMPOTENT_REPLAY

        committed: MaterializedStateSnapshot | None = snapshot
        checkpoint: ConsumerCheckpoint | None = None
        if persist:
            committed = self.save_materialized_snapshot(
                selected_projection, state, current
            )
            checkpoint = self.save_consumer_checkpoint(
                _projection_consumer_id(selected_projection),
                current,
            )
        else:
            checkpoint = self.load_consumer_checkpoint(
                _projection_consumer_id(selected_projection)
            )

        return MaterializedStateRecovery(
            projection_id=selected_projection,
            state=state,
            cursor=current,
            snapshot=committed,
            applied_event_ids=tuple(applied),
            outcome=outcome,
            recovered_from_snapshot=recovered_from_snapshot,
            checkpoint=checkpoint,
            has_more=has_more,
        )

    def materialize(
        self,
        projection_id: str,
        *,
        reducer: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        initial_state: Mapping[str, Any] | None = None,
        stream_id: str = DEFAULT_STREAM_ID,
        limit: int = DEFAULT_PAGE_LIMIT,
        worker_assertion: bool = False,
    ) -> MaterializedStateRecovery:
        """Fold one bounded page into durable materialized state and persist it."""

        return self.recover_materialized_state(
            projection_id,
            reducer=reducer,
            initial_state=initial_state,
            stream_id=stream_id,
            limit=limit,
            persist=True,
            until_head=False,
            worker_assertion=worker_assertion,
        )

    # -- integrity / retention -----------------------------------------------

    def write_integrity_checkpoint(
        self, stream_id: str = DEFAULT_STREAM_ID
    ) -> IntegrityCheckpoint:
        selected_stream = _text(stream_id, "stream_id")
        with self._lock:
            connection = self._require()
            computed = self._compute_chain_digest(selected_stream)
            earliest = int(computed["earliest_sequence"])
            latest = int(computed["latest_sequence"])
            count = int(computed["event_count"])
            digest = str(computed["chain_digest"])
            stamp = _utc_iso()
            checkpoint_id = _sha256_hex(
                _canonical_json(
                    {
                        "stream_id": selected_stream,
                        "earliest_sequence": earliest,
                        "latest_sequence": latest,
                        "event_count": count,
                        "chain_digest": digest,
                        "recorded_at": stamp,
                    }
                ).encode("utf-8")
            )
            body = {
                "stream_id": selected_stream,
                "chain_digest": digest,
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO integrity_checkpoints (
                    checkpoint_id, stream_id, earliest_sequence,
                    latest_sequence, event_count, chain_digest,
                    recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    checkpoint_id,
                    selected_stream,
                    earliest,
                    latest,
                    count,
                    digest,
                    stamp,
                    _canonical_json(body),
                ],
            )
            self._commit_if_idle(connection)
            return IntegrityCheckpoint(
                checkpoint_id=checkpoint_id,
                stream_id=selected_stream,
                earliest_sequence=earliest,
                latest_sequence=latest,
                event_count=count,
                chain_digest=digest,
                recorded_at=stamp,
            )

    def verify_integrity_checkpoint(
        self, checkpoint: IntegrityCheckpoint | Mapping[str, Any]
    ) -> bool:
        if isinstance(checkpoint, IntegrityCheckpoint):
            selected = checkpoint
        else:
            selected = IntegrityCheckpoint(
                checkpoint_id=str(checkpoint["checkpoint_id"]),
                stream_id=str(checkpoint["stream_id"]),
                earliest_sequence=int(checkpoint["earliest_sequence"]),
                latest_sequence=int(checkpoint["latest_sequence"]),
                event_count=int(checkpoint["event_count"]),
                chain_digest=str(checkpoint["chain_digest"]),
                recorded_at=str(checkpoint.get("recorded_at") or ""),
            )
        with self._lock:
            recomputed = self._compute_chain_digest(selected.stream_id)
        if recomputed["chain_digest"] != selected.chain_digest:
            raise DatabaseEventLogIntegrityError(
                "integrity checkpoint chain digest mismatch"
            )
        if int(recomputed["event_count"]) != selected.event_count:
            raise DatabaseEventLogIntegrityError(
                "integrity checkpoint event count mismatch"
            )
        return True

    def _compute_chain_digest(self, stream_id: str) -> dict[str, Any]:
        """Recompute the retained hash chain (caller must hold ``_lock``)."""

        connection = self._require()
        rows = connection.execute(
            """
            SELECT event_id, sequence, previous_event_id
            FROM domain_events
            WHERE stream_id = ?
            ORDER BY sequence ASC
            """,
            [stream_id],
        ).fetchall()
        if not rows:
            return {
                "earliest_sequence": 0,
                "latest_sequence": 0,
                "event_count": 0,
                "chain_digest": _sha256_hex(b"empty"),
            }
        chain: list[str] = []
        previous = ""
        for row in rows:
            mapping = _row_mapping(row)
            event_id = str(mapping["event_id"])
            prior = str(mapping.get("previous_event_id") or "")
            if previous and prior != previous:
                raise DatabaseEventLogIntegrityError(
                    "event chain broken while verifying integrity"
                )
            chain.append(f"{mapping['sequence']}:{event_id}")
            previous = event_id
        return {
            "earliest_sequence": int(_row_mapping(rows[0])["sequence"]),
            "latest_sequence": int(_row_mapping(rows[-1])["sequence"]),
            "event_count": len(rows),
            "chain_digest": _sha256_hex("\n".join(chain).encode("utf-8")),
        }

    def apply_retention(
        self,
        *,
        stream_id: str = DEFAULT_STREAM_ID,
        retain_recent: int | None = None,
        before_sequence: int | None = None,
    ) -> dict[str, Any]:
        """Drop events older than the retained window. Heads remain monotonic."""

        selected_stream = _text(stream_id, "stream_id")
        if retain_recent is None and before_sequence is None:
            raise DatabaseEventLogError(
                "retain_recent or before_sequence is required"
            )
        with self._lock:
            connection = self._require()
            head = self._load_head(connection, selected_stream)
            if before_sequence is not None:
                cutoff = _positive_int(before_sequence, "before_sequence")
            else:
                keep = _positive_int(retain_recent, "retain_recent")
                cutoff = max(1, head.latest_sequence - keep + 1)
            # Never delete the head event when retain_recent is at least 1.
            if head.latest_sequence and cutoff > head.latest_sequence:
                cutoff = head.latest_sequence
            deleted = connection.execute(
                """
                DELETE FROM domain_events
                WHERE stream_id = ? AND sequence < ?
                """,
                [selected_stream, cutoff],
            )
            # DuckDB may not populate rowcount; compute via remaining.
            remaining = connection.execute(
                """
                SELECT COUNT(*) AS c FROM domain_events WHERE stream_id = ?
                """,
                [selected_stream],
            ).fetchone()
            remaining_count = int(_row_mapping(remaining).get("c") or 0)
            earliest = self._earliest_sequence(connection, selected_stream)
            self._commit_if_idle(connection)
            result = {
                "stream_id": selected_stream,
                "cutoff_sequence": cutoff,
                "earliest_sequence": earliest,
                "latest_sequence": head.latest_sequence,
                "remaining_count": remaining_count,
                "deleted_hint": getattr(deleted, "rowcount", -1),
            }
            return result

    # -- export (non-authoritative) ------------------------------------------

    def export_jsonl(
        self,
        path: Path | str,
        *,
        stream_id: str = DEFAULT_STREAM_ID,
    ) -> JsonlExportReceipt:
        """Render events to JSONL. The file is never read back as authority."""

        export_path = Path(path)
        selected_stream = _text(stream_id, "stream_id")
        lines: list[str] = []
        earliest = 0
        latest = 0
        for event in self.replay(stream_id=selected_stream, limit=DEFAULT_PAGE_LIMIT):
            sequence = int(event.get("sequence") or 0)
            if not earliest:
                earliest = sequence
            latest = sequence
            lines.append(_canonical_json(event))
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_bytes(payload)
        digest = _sha256_hex(payload)
        receipt = JsonlExportReceipt(
            export_path=str(export_path),
            stream_id=selected_stream,
            event_count=len(lines),
            earliest_sequence=earliest,
            latest_sequence=latest,
            content_digest=digest,
            recorded_at=_utc_iso(),
        )
        # Explicit audit of export; export itself remains non-authoritative.
        try:
            self.append_audit(
                AuditAction.EXPORT,
                actor_id="database_event_log",
                subject_kind="jsonl_export",
                subject_id=str(export_path),
                body=receipt.to_dict(),
                emit_event=True,
            )
        except DatabaseEventLogBoundsError:
            # Nested audit bound: export still succeeds.
            pass
        return receipt

    def authority_unaffected_by_export_deletion(
        self, export_path: Path | str
    ) -> bool:
        """Deleting an export must not change stream heads or event counts."""

        path = Path(export_path)
        before = {
            stream: self.stream_head(stream).to_dict()
            for stream in self.list_streams()
        }
        if path.exists():
            path.unlink()
        after = {
            stream: self.stream_head(stream).to_dict()
            for stream in self.list_streams()
        }
        return before == after

    def list_streams(self) -> tuple[str, ...]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT stream_id FROM stream_heads ORDER BY stream_id ASC"
            ).fetchall()
        return tuple(str(_row_mapping(row)["stream_id"]) for row in rows)

    def list_audits(
        self, *, subject_id: str | None = None, limit: int = 100
    ) -> tuple[dict[str, Any], ...]:
        page_limit = _positive_int(limit, "limit")
        with self._lock:
            connection = self._require()
            if subject_id:
                rows = connection.execute(
                    """
                    SELECT audit_id, event_id, actor_id, action, subject_kind,
                           subject_id, recorded_at, body_json, nesting_depth
                    FROM audit_records
                    WHERE subject_id = ?
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [_text(subject_id, "subject_id"), page_limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT audit_id, event_id, actor_id, action, subject_kind,
                           subject_id, recorded_at, body_json, nesting_depth
                    FROM audit_records
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [page_limit],
                ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            mapping = _row_mapping(row)
            body = json.loads(str(mapping.get("body_json") or "{}"))
            results.append(
                {
                    "audit_id": mapping["audit_id"],
                    "event_id": mapping["event_id"],
                    "actor_id": mapping["actor_id"],
                    "action": mapping["action"],
                    "subject_kind": mapping["subject_kind"],
                    "subject_id": mapping["subject_id"],
                    "recorded_at": mapping["recorded_at"],
                    "body": body,
                    "nesting_depth": int(mapping.get("nesting_depth") or 0),
                }
            )
        return tuple(results)

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _cursors_equivalent(left: EventCursor, right: EventCursor) -> bool:
        return (
            left.stream_id == right.stream_id
            and left.position == right.position
            and left.last_event_id == right.last_event_id
            and left.snapshot_id == right.snapshot_id
        )

    def _load_consumer_checkpoint_unlocked(
        self, connection: Any, consumer_id: str
    ) -> ConsumerCheckpoint | None:
        rows = connection.execute(
            """
            SELECT consumer_id, stream_id, snapshot_id, position,
                   last_event_id, updated_at, cursor_token,
                   checkpoint_digest
            FROM consumer_checkpoints WHERE consumer_id = ? LIMIT 1
            """,
            [consumer_id],
        ).fetchall()
        if not rows:
            return None
        row = _row_mapping(rows[0])
        cursor = EventCursor(
            stream_id=str(row["stream_id"]),
            snapshot_id=str(row["snapshot_id"]),
            position=int(row["position"]),
            last_event_id=str(row["last_event_id"] or ""),
        )
        checkpoint = ConsumerCheckpoint(
            consumer_id=str(row["consumer_id"]),
            cursor=cursor,
            updated_at=str(row["updated_at"]),
        )
        if checkpoint.checkpoint_digest != str(row["checkpoint_digest"]):
            raise DatabaseEventLogIntegrityError(
                "consumer checkpoint digest mismatch"
            )
        return checkpoint

    def _save_consumer_checkpoint_unlocked(
        self,
        connection: Any,
        consumer_id: str,
        selected: EventCursor,
    ) -> ConsumerCheckpoint:
        head = self._load_head(connection, selected.stream_id)
        selected.assert_replayable(
            stream_id=selected.stream_id,
            earliest_position=self._earliest_sequence(
                connection, selected.stream_id
            ),
            latest_position=head.latest_sequence,
            snapshot_id=self._snapshot_id,
        )
        existing = self._load_consumer_checkpoint_unlocked(
            connection, consumer_id
        )
        if existing is not None:
            if existing.cursor.stream_id != selected.stream_id:
                raise DatabaseEventLogConflictError(
                    "consumer checkpoint is bound to a different stream"
                )
            if selected.position < existing.cursor.position:
                raise DatabaseEventLogConflictError(
                    "consumer checkpoint cannot rewind"
                )
        checkpoint = ConsumerCheckpoint(
            consumer_id=consumer_id,
            cursor=selected,
            updated_at=_utc_iso(),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO consumer_checkpoints (
                consumer_id, stream_id, snapshot_id, position,
                last_event_id, updated_at, cursor_token, checkpoint_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                checkpoint.consumer_id,
                selected.stream_id,
                selected.snapshot_id,
                selected.position,
                selected.last_event_id,
                checkpoint.updated_at,
                selected.to_token(),
                checkpoint.checkpoint_digest,
            ],
        )
        self._commit_if_idle(connection)
        return checkpoint

    def _get_consumed_row(
        self, connection: Any, consumer_id: str, event_id: str
    ) -> dict[str, Any] | None:
        rows = connection.execute(
            """
            SELECT consumer_id, event_id, stream_id, sequence, snapshot_id,
                   status, applied_at, consumption_digest
            FROM consumed_events
            WHERE consumer_id = ? AND event_id = ?
            LIMIT 1
            """,
            [consumer_id, event_id],
        ).fetchall()
        if not rows:
            return None
        return _row_mapping(rows[0])

    def _load_latest_committed_snapshot_unlocked(
        self,
        connection: Any,
        projection_id: str,
        stream_id: str | None = None,
    ) -> MaterializedStateSnapshot | None:
        if stream_id is None:
            rows = connection.execute(
                """
                SELECT snapshot_cid, projection_id, stream_id, snapshot_id,
                       position, last_event_id, cursor_token, state_json,
                       state_digest, status, recorded_at, snapshot_digest
                FROM materialized_state_snapshots
                WHERE projection_id = ? AND status = ?
                ORDER BY position DESC
                LIMIT 1
                """,
                [
                    projection_id,
                    MaterializedSnapshotStatus.COMMITTED.value,
                ],
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT snapshot_cid, projection_id, stream_id, snapshot_id,
                       position, last_event_id, cursor_token, state_json,
                       state_digest, status, recorded_at, snapshot_digest
                FROM materialized_state_snapshots
                WHERE projection_id = ? AND stream_id = ? AND status = ?
                ORDER BY position DESC
                LIMIT 1
                """,
                [
                    projection_id,
                    stream_id,
                    MaterializedSnapshotStatus.COMMITTED.value,
                ],
            ).fetchall()
        if not rows:
            return None
        return self._row_to_snapshot(_row_mapping(rows[0]))

    def _row_to_snapshot(self, row: Mapping[str, Any]) -> MaterializedStateSnapshot:
        body_raw = row.get("state_json") or "{}"
        if isinstance(body_raw, Mapping):
            state = dict(body_raw)
        else:
            try:
                state = json.loads(str(body_raw))
            except json.JSONDecodeError as exc:
                raise DatabaseEventLogIntegrityError(
                    "materialized snapshot state is not valid JSON"
                ) from exc
        if not isinstance(state, dict):
            raise DatabaseEventLogIntegrityError(
                "materialized snapshot state must be an object"
            )
        cursor = EventCursor(
            stream_id=str(row["stream_id"]),
            snapshot_id=str(row["snapshot_id"] or self._snapshot_id),
            position=int(row["position"] or 0),
            last_event_id=str(row["last_event_id"] or ""),
        )
        snapshot = MaterializedStateSnapshot(
            projection_id=str(row["projection_id"]),
            cursor=cursor,
            state=state,
            status=MaterializedSnapshotStatus(str(row["status"])),
            recorded_at=str(row.get("recorded_at") or ""),
            snapshot_cid=str(row.get("snapshot_cid") or ""),
        )
        stored_state_digest = str(row.get("state_digest") or "")
        stored_snapshot_digest = str(row.get("snapshot_digest") or "")
        if stored_state_digest and stored_state_digest != snapshot.state_digest:
            raise DatabaseEventLogIntegrityError(
                "materialized snapshot state digest mismatch"
            )
        if (
            stored_snapshot_digest
            and stored_snapshot_digest != snapshot.snapshot_digest
        ):
            raise DatabaseEventLogIntegrityError(
                "materialized snapshot digest mismatch"
            )
        stored_cid = str(row.get("snapshot_cid") or "")
        if stored_cid and stored_cid != snapshot.snapshot_cid:
            raise DatabaseEventLogIntegrityError(
                "materialized snapshot_cid mismatch"
            )
        return snapshot

    def _upsert_snapshot_unlocked(
        self,
        connection: Any,
        snapshot: MaterializedStateSnapshot,
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO materialized_state_snapshots (
                snapshot_cid, projection_id, stream_id, snapshot_id,
                position, last_event_id, cursor_token, state_json,
                state_digest, status, recorded_at, snapshot_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot.snapshot_cid,
                snapshot.projection_id,
                snapshot.cursor.stream_id,
                snapshot.cursor.snapshot_id,
                snapshot.cursor.position,
                snapshot.cursor.last_event_id,
                snapshot.cursor.to_token(),
                _canonical_json(dict(snapshot.state)),
                snapshot.state_digest,
                snapshot.status.value,
                snapshot.recorded_at,
                snapshot.snapshot_digest,
            ],
        )

    def _commit_materialized_snapshot_unlocked(
        self,
        connection: Any,
        *,
        projection_id: str,
        state: Mapping[str, Any],
        cursor: EventCursor,
    ) -> MaterializedStateSnapshot:
        head = self._load_head(connection, cursor.stream_id)
        cursor.assert_replayable(
            stream_id=cursor.stream_id,
            earliest_position=self._earliest_sequence(
                connection, cursor.stream_id
            ),
            latest_position=head.latest_sequence,
            snapshot_id=self._snapshot_id,
        )
        existing = self._load_latest_committed_snapshot_unlocked(
            connection, projection_id, cursor.stream_id
        )
        payload = _bounded_state(state)
        if existing is not None:
            if existing.cursor.stream_id != cursor.stream_id:
                raise DatabaseEventLogConflictError(
                    "materialized snapshot is bound to a different stream"
                )
            if cursor.position < existing.cursor.position:
                raise DatabaseEventLogConflictError(
                    "materialized snapshot cannot rewind"
                )
            if cursor.position == existing.cursor.position:
                same_state = dict(existing.state) == payload
                same_cursor = self._cursors_equivalent(existing.cursor, cursor)
                if same_state and same_cursor:
                    return existing
                raise DatabaseEventLogConflictError(
                    "materialized snapshot already exists with a different state"
                )
        pending = MaterializedStateSnapshot(
            projection_id=projection_id,
            cursor=cursor,
            state=payload,
            status=MaterializedSnapshotStatus.PENDING,
        )
        self._upsert_snapshot_unlocked(connection, pending)
        self._commit_if_idle(connection)
        committed = MaterializedStateSnapshot(
            projection_id=projection_id,
            cursor=cursor,
            state=payload,
            status=MaterializedSnapshotStatus.COMMITTED,
            recorded_at=pending.recorded_at,
            snapshot_cid=pending.snapshot_cid,
        )
        self._upsert_snapshot_unlocked(connection, committed)
        self._commit_if_idle(connection)
        return committed

    def _upsert_consumed_unlocked(
        self,
        connection: Any,
        *,
        consumer_id: str,
        event_id: str,
        stream_id: str,
        sequence: int,
        status: ConsumptionStatus,
        applied_at: str,
    ) -> None:
        digest = _consumption_digest(
            consumer_id=consumer_id,
            event_id=event_id,
            stream_id=stream_id,
            sequence=sequence,
            status=status.value,
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO consumed_events (
                consumer_id, event_id, stream_id, sequence, snapshot_id,
                status, applied_at, consumption_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                consumer_id,
                event_id,
                stream_id,
                sequence,
                self._snapshot_id,
                status.value,
                applied_at,
                digest,
            ],
        )

    def _coerce_cursor(
        self, cursor: EventCursor | Mapping[str, Any] | str
    ) -> EventCursor:
        if isinstance(cursor, EventCursor):
            selected = cursor
        elif isinstance(cursor, str):
            selected = EventCursor.from_token(cursor)
        elif isinstance(cursor, Mapping):
            selected = EventCursor.from_dict(cursor)
        else:
            raise EventCursorError(
                "cursor must be an EventCursor, record, or token"
            )
        if selected.snapshot_id and selected.snapshot_id != self._snapshot_id:
            raise CursorReplayError(
                "event cursor snapshot does not match the event stream"
            )
        if not selected.snapshot_id:
            selected = EventCursor(
                stream_id=selected.stream_id,
                position=selected.position,
                last_event_id=selected.last_event_id,
                snapshot_id=self._snapshot_id,
            )
        return selected

    def _load_head(self, connection: Any, stream_id: str) -> StreamHead:
        rows = connection.execute(
            """
            SELECT stream_id, snapshot_id, latest_sequence, last_event_id,
                   global_sequence, updated_at
            FROM stream_heads WHERE stream_id = ? LIMIT 1
            """,
            [stream_id],
        ).fetchall()
        if rows:
            mapping = _row_mapping(rows[0])
            return StreamHead(
                stream_id=str(mapping["stream_id"]),
                snapshot_id=str(mapping["snapshot_id"] or self._snapshot_id),
                latest_sequence=int(mapping["latest_sequence"] or 0),
                last_event_id=str(mapping["last_event_id"] or ""),
                global_sequence=int(mapping["global_sequence"] or 0),
                updated_at=str(mapping["updated_at"] or ""),
            )
        # Derive from events if head row is missing (imported control plane).
        rows = connection.execute(
            """
            SELECT event_id, sequence, global_sequence, recorded_at, snapshot_id
            FROM domain_events
            WHERE stream_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            [stream_id],
        ).fetchall()
        if not rows:
            return StreamHead(
                stream_id=stream_id,
                snapshot_id=self._snapshot_id,
                latest_sequence=0,
                last_event_id="",
                global_sequence=0,
            )
        mapping = _row_mapping(rows[0])
        return StreamHead(
            stream_id=stream_id,
            snapshot_id=str(mapping.get("snapshot_id") or self._snapshot_id),
            latest_sequence=int(mapping["sequence"] or 0),
            last_event_id=str(mapping["event_id"] or ""),
            global_sequence=int(mapping.get("global_sequence") or 0),
            updated_at=str(mapping.get("recorded_at") or ""),
        )

    def _next_global_sequence(self, connection: Any) -> int:
        rows = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) AS watermark FROM domain_events"
        ).fetchall()
        if not rows:
            return 1
        return int(_row_mapping(rows[0]).get("watermark") or 0) + 1

    def _earliest_sequence(self, connection: Any, stream_id: str) -> int:
        rows = connection.execute(
            """
            SELECT COALESCE(MIN(sequence), 0) AS earliest
            FROM domain_events WHERE stream_id = ?
            """,
            [stream_id],
        ).fetchall()
        if not rows:
            return 0
        return int(_row_mapping(rows[0]).get("earliest") or 0)

    def _event_at_sequence(
        self, connection: Any, stream_id: str, sequence: int
    ) -> dict[str, Any] | None:
        rows = connection.execute(
            """
            SELECT event_id, stream_id, sequence, global_sequence,
                   event_type, task_cid, attempt_id, session_id,
                   recorded_at, body_json, previous_event_id,
                   snapshot_id, redacted
            FROM domain_events
            WHERE stream_id = ? AND sequence = ?
            LIMIT 1
            """,
            [stream_id, sequence],
        ).fetchall()
        if not rows:
            return None
        return _row_mapping(rows[0])

    def _get_event_row(
        self, connection: Any, event_id: str
    ) -> dict[str, Any] | None:
        rows = connection.execute(
            """
            SELECT event_id, stream_id, sequence, global_sequence,
                   event_type, task_cid, attempt_id, session_id,
                   recorded_at, body_json, previous_event_id,
                   snapshot_id, redacted
            FROM domain_events WHERE event_id = ? LIMIT 1
            """,
            [event_id],
        ).fetchall()
        if not rows:
            return None
        return _row_mapping(rows[0])

    def _row_to_event(self, row: Mapping[str, Any]) -> DomainEvent:
        body_raw = row.get("body_json") or "{}"
        if isinstance(body_raw, Mapping):
            body = dict(body_raw)
        else:
            try:
                body = json.loads(str(body_raw))
            except json.JSONDecodeError:
                body = {}
        if not isinstance(body, dict):
            body = {}
        return DomainEvent(
            event_id=str(row["event_id"]),
            stream_id=str(row["stream_id"]),
            sequence=int(row["sequence"]),
            global_sequence=int(row["global_sequence"]),
            event_type=str(row["event_type"]),
            recorded_at=str(row["recorded_at"]),
            body=MappingProxyType(body),
            previous_event_id=str(row.get("previous_event_id") or ""),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            task_cid=str(row.get("task_cid") or ""),
            attempt_id=str(row.get("attempt_id") or ""),
            session_id=str(row.get("session_id") or ""),
            redacted=bool(row.get("redacted")),
        )


def open_database_event_log(
    database_path: Path | str,
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    auto_redact: bool = True,
) -> DatabaseEventLog:
    """Open and return an initialized :class:`DatabaseEventLog`."""

    return DatabaseEventLog(
        database_path,
        snapshot_id=snapshot_id,
        auto_redact=auto_redact,
    ).open()


__all__ = (
    "AUDIT_RECORD_SCHEMA",
    "AuditAction",
    "CONSUMER_CHECKPOINT_INTERFACE",
    "CONSUMER_CHECKPOINT_SCHEMA",
    "ConsumptionOutcome",
    "ConsumptionStatus",
    "ConsumerCheckpoint",
    "DATABASE_EVENT_LOG_INTERFACE",
    "DATABASE_EVENT_LOG_SCHEMA",
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_SNAPSHOT_ID",
    "DEFAULT_STREAM_ID",
    "DatabaseEventLog",
    "DatabaseEventLogBoundsError",
    "DatabaseEventLogConflictError",
    "DatabaseEventLogError",
    "DatabaseEventLogIntegrityError",
    "DatabaseEventLogNotOpenError",
    "DomainEvent",
    "DuckDBUnavailableError",
    "EVENT_CONSUMPTION_RECORD_SCHEMA",
    "EVENT_CURSOR_INTERFACE",
    "EventConsumptionPage",
    "EventConsumptionRecord",
    "IDEMPOTENT_EVENT_CONSUMPTION_BINDING",
    "IDEMPOTENT_EVENT_CONSUMPTION_CONSUMES",
    "IDEMPOTENT_EVENT_CONSUMPTION_INTERFACE",
    "IDEMPOTENT_EVENT_CONSUMPTION_SCHEMA",
    "INTEGRITY_CHECKPOINT_SCHEMA",
    "IntegrityCheckpoint",
    "JSONL_EXPORT_RECEIPT_SCHEMA",
    "JsonlExportReceipt",
    "LogSeverity",
    "MATERIALIZED_STATE_RECOVERY_SCHEMA",
    "MATERIALIZED_STATE_REPLAY_BINDING",
    "MATERIALIZED_STATE_REPLAY_CONSUMES",
    "MATERIALIZED_STATE_REPLAY_INTERFACE",
    "MATERIALIZED_STATE_REPLAY_SCHEMA",
    "MATERIALIZED_STATE_SNAPSHOT_SCHEMA",
    "MaterializedRecoveryOutcome",
    "MaterializedSnapshotStatus",
    "MaterializedStateRecovery",
    "MaterializedStateSnapshot",
    "REDACTION_MARKER",
    "STREAM_HEAD_SCHEMA",
    "StreamHead",
    "duckdb_available",
    "open_database_event_log",
)
