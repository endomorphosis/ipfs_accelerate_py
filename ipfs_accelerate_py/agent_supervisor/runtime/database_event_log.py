"""DuckDB-backed authoritative event, audit, log, metric, and cursor store.

DQP-013 / DatabaseEventLog@1
============================

:class:`DatabaseEventLog` is the durable authority for domain events,
structured logs, metrics, explicit application audits, stream heads,
retention, integrity checkpoints, and consumer cursors. JSONL is an export
adapter only: deleting or tampering with an export has no authority effect.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
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

DATABASE_EVENT_LOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-event-log@1"
)
CONSUMER_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/consumer-checkpoint@1"
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

    # -- consumer checkpoints ------------------------------------------------

    def save_consumer_checkpoint(
        self,
        consumer_id: str,
        cursor: EventCursor | Mapping[str, Any] | str,
    ) -> ConsumerCheckpoint:
        selected = self._coerce_cursor(cursor)
        checkpoint = ConsumerCheckpoint(
            consumer_id=_text(consumer_id, "consumer_id"),
            cursor=selected,
            updated_at=_utc_iso(),
        )
        with self._lock:
            connection = self._require()
            head = self._load_head(connection, selected.stream_id)
            selected.assert_replayable(
                stream_id=selected.stream_id,
                earliest_position=self._earliest_sequence(
                    connection, selected.stream_id
                ),
                latest_position=head.latest_sequence,
                snapshot_id=self._snapshot_id,
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

    def load_consumer_checkpoint(
        self, consumer_id: str
    ) -> ConsumerCheckpoint | None:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT consumer_id, stream_id, snapshot_id, position,
                       last_event_id, updated_at, cursor_token,
                       checkpoint_digest
                FROM consumer_checkpoints WHERE consumer_id = ? LIMIT 1
                """,
                [_text(consumer_id, "consumer_id")],
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
    "EVENT_CURSOR_INTERFACE",
    "INTEGRITY_CHECKPOINT_SCHEMA",
    "IntegrityCheckpoint",
    "JSONL_EXPORT_RECEIPT_SCHEMA",
    "JsonlExportReceipt",
    "LogSeverity",
    "REDACTION_MARKER",
    "STREAM_HEAD_SCHEMA",
    "StreamHead",
    "duckdb_available",
    "open_database_event_log",
)
