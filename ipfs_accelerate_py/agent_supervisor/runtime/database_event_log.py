"""DuckDB-authoritative event, audit, log, metric, and cursor store (DQP-013).

Interfaces: ``DatabaseEventLog@1``, ``EventCursor@1``, ``ConsumerCheckpoint@1``

JSONL event logs remain a deterministic export surface only. Authority for
typed events, structured logs, metrics, traces, application audit records,
stream heads, retention watermarks, integrity checkpoints, and consumer
cursors lives in the control-plane DuckDB tables (plus a small set of
event-store bookkeeping tables owned by this module).

Application audit is an explicit insert path. Quack transport diagnostics
(``quack_query_telemetry``) are never treated as the application audit ledger.
Recursive logging that would re-enter the event store is depth-bounded.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..control.control_contracts import (
    CursorReplayError,
    EventCursor,
    EventCursorError,
    EventPage,
    replay_event_page,
)
from ..self_improvement.supervisor_v2_contracts import (
    MAX_PROJECTION_BYTES,
    MAX_RECEIPT_BYTES,
)
from ..task_sources.control_plane_contracts import (
    ControlPlaneBoundsError,
    ControlPlaneContractError,
    StateAuthorityClass,
    canonical_json_bytes,
    content_identity,
)
from ..task_sources.control_plane_schema import install_control_plane_schema
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

DATABASE_EVENT_LOG_INTERFACE: Final = "DatabaseEventLog@1"
EVENT_CURSOR_INTERFACE: Final = "EventCursor@1"
CONSUMER_CHECKPOINT_INTERFACE: Final = "ConsumerCheckpoint@1"

DATABASE_EVENT_LOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/database-event-log@1"
)
CONSUMER_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/consumer-checkpoint@1"
)
STREAM_HEAD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/event-stream-head@1"
)
INTEGRITY_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/event-integrity-checkpoint@1"
)
EVENT_EXPORT_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/event-jsonl-export-receipt@1"
)
APPLICATION_AUDIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/application-audit@1"
)

DATABASE_EVENT_LOG_VERSION: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 256
MAX_PAGE_LIMIT: Final[int] = 4_096
MAX_RECURSIVE_LOG_DEPTH: Final[int] = 2
DEFAULT_STREAM_ID: Final = "domain_events:default"
APPLICATION_AUDIT_COMPONENT: Final = "application_audit"
APPLICATION_AUDIT_EVENT_TYPE: Final = "application.audit"
QUACK_DIAGNOSTIC_COMPONENT: Final = "quack_query_telemetry"

_RESERVED_EVENT_FIELDS: Final = frozenset(
    {
        "stream_id",
        "snapshot_id",
        "sequence",
        "position",
        "event_id",
        "previous_event_id",
        "global_sequence",
        "recorded_at",
        "type",
        "event_type",
    }
)

_BOOKKEEPING_SQL: Final = """
CREATE TABLE IF NOT EXISTS event_stream_heads (
    stream_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    latest_sequence BIGINT NOT NULL,
    earliest_sequence BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL,
    last_global_sequence BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_consumer_checkpoints (
    consumer_id VARCHAR NOT NULL,
    stream_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL,
    checkpoint_id VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (consumer_id, stream_id)
);

CREATE TABLE IF NOT EXISTS event_integrity_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    earliest_sequence BIGINT NOT NULL,
    latest_sequence BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL,
    population_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS application_audit_records (
    audit_id VARCHAR PRIMARY KEY,
    action VARCHAR NOT NULL,
    actor_id VARCHAR NOT NULL DEFAULT '',
    subject_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log_exports (
    export_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    destination VARCHAR NOT NULL,
    artifact_digest VARCHAR NOT NULL,
    event_count BIGINT NOT NULL,
    earliest_sequence BIGINT NOT NULL,
    latest_sequence BIGINT NOT NULL,
    authority_class VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS event_consumer_checkpoints_stream_idx
    ON event_consumer_checkpoints(stream_id, position);
CREATE INDEX IF NOT EXISTS application_audit_records_subject_idx
    ON application_audit_records(subject_id, recorded_at);
CREATE INDEX IF NOT EXISTS event_integrity_checkpoints_stream_idx
    ON event_integrity_checkpoints(stream_id, latest_sequence);
"""

_tls = threading.local()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseEventLogError(RuntimeError):
    """Base failure for the database-backed event store."""


class DatabaseEventLogNotOpenError(DatabaseEventLogError):
    """Store was used before open or after close."""


class DatabaseEventLogImmutabilityError(DatabaseEventLogError):
    """An immutable event identity or sequence was rewritten."""


class DatabaseEventLogBoundsError(DatabaseEventLogError):
    """A payload or recursion bound was exceeded."""


class DatabaseEventLogAuthorityError(DatabaseEventLogError):
    """An export or diagnostic surface was treated as authority."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RecordKind(str, Enum):
    """Closed population of durable event-store record kinds."""

    EVENT = "event"
    LOG = "log"
    METRIC = "metric"
    TRACE = "trace"
    AUDIT = "audit"
    CHECKPOINT = "checkpoint"
    EXPORT = "export"


class LogSeverity(str, Enum):
    """Closed severity ladder for structured logs and traces."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_mapping(value: Mapping[str, Any], *, maximum: int) -> dict[str, Any]:
    try:
        encoded = canonical_json_bytes(dict(value))
    except (ControlPlaneBoundsError, ControlPlaneContractError, TypeError, ValueError) as exc:
        raise DatabaseEventLogBoundsError(
            "event payload must contain canonical JSON values"
        ) from exc
    if len(encoded) > maximum:
        raise DatabaseEventLogBoundsError(
            f"payload exceeds the {maximum}-byte persistence bound"
        )
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabaseEventLogBoundsError(
            "event payload is not canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise DatabaseEventLogBoundsError("event payload must be an object")
    return decoded


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise DatabaseEventLogError(f"{field_name} is required")
    if "\x00" in text:
        raise DatabaseEventLogError(f"{field_name} contains NUL")
    if len(text.encode("utf-8")) > 8_192:
        raise DatabaseEventLogBoundsError(f"{field_name} exceeds text bound")
    return text


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseEventLogError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseEventLogError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _event_identity(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("event_id", None)
    return "sha256:" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _snapshot_for_stream(stream_id: str) -> str:
    digest = hashlib.sha256(stream_id.encode("utf-8")).hexdigest()
    return f"event-stream-snapshot:sha256:{digest}"


def _logging_depth() -> int:
    return int(getattr(_tls, "depth", 0) or 0)


def _enter_logging() -> int:
    depth = _logging_depth() + 1
    _tls.depth = depth
    return depth


def _exit_logging() -> None:
    depth = _logging_depth() - 1
    _tls.depth = max(0, depth)


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split SQL text on semicolons outside single-quoted strings."""

    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    index = 0
    length = len(sql_text)
    while index < length:
        ch = sql_text[index]
        if in_single:
            buf.append(ch)
            if ch == "'":
                if index + 1 < length and sql_text[index + 1] == "'":
                    buf.append("'")
                    index += 2
                    continue
                in_single = False
            index += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            index += 1
            continue
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
            index += 1
            continue
        buf.append(ch)
        index += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerCheckpoint:
    """Durable consumer resume point bound to one stream and snapshot.

    Interface: ``ConsumerCheckpoint@1``.
    """

    SCHEMA: ClassVar[str] = CONSUMER_CHECKPOINT_SCHEMA
    INTERFACE: ClassVar[str] = CONSUMER_CHECKPOINT_INTERFACE

    consumer_id: str
    cursor: EventCursor
    checkpoint_id: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        if not isinstance(self.cursor, EventCursor):
            raise TypeError("cursor must be an EventCursor")
        stamp = str(self.updated_at or "").strip() or utc_now()
        object.__setattr__(self, "updated_at", stamp)
        payload = {
            "schema": self.SCHEMA,
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "updated_at": self.updated_at,
        }
        expected = content_identity(payload)
        claimed = str(self.checkpoint_id or "").strip()
        if claimed and claimed != expected:
            raise DatabaseEventLogImmutabilityError(
                "consumer checkpoint identity mismatch"
            )
        object.__setattr__(self, "checkpoint_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "checkpoint_id": self.checkpoint_id,
            "updated_at": self.updated_at,
        }

    to_record = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsumerCheckpoint":
        if not isinstance(payload, Mapping):
            raise DatabaseEventLogError(
                "consumer checkpoint payload must be an object"
            )
        cursor_value = payload.get("cursor")
        if isinstance(cursor_value, EventCursor):
            cursor = cursor_value
        elif isinstance(cursor_value, Mapping):
            cursor = EventCursor.from_dict(cursor_value)
        else:
            raise DatabaseEventLogError(
                "consumer checkpoint cursor must be an EventCursor record"
            )
        return cls(
            consumer_id=str(payload.get("consumer_id") or ""),
            cursor=cursor,
            checkpoint_id=str(payload.get("checkpoint_id") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class StreamHead:
    """Authoritative head of one event stream."""

    SCHEMA: ClassVar[str] = STREAM_HEAD_SCHEMA

    stream_id: str
    snapshot_id: str
    earliest_sequence: int
    latest_sequence: int
    last_event_id: str
    last_global_sequence: int
    updated_at: str
    revision: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "snapshot_id": self.snapshot_id,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "last_event_id": self.last_event_id,
            "last_global_sequence": self.last_global_sequence,
            "updated_at": self.updated_at,
            "revision": self.revision,
        }

    def to_cursor(self) -> EventCursor:
        if self.latest_sequence <= 0:
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
    """Content-addressed integrity witness for a retained stream window."""

    SCHEMA: ClassVar[str] = INTEGRITY_CHECKPOINT_SCHEMA

    checkpoint_id: str
    stream_id: str
    earliest_sequence: int
    latest_sequence: int
    last_event_id: str
    population_digest: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "stream_id": self.stream_id,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "last_event_id": self.last_event_id,
            "population_digest": self.population_digest,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class EventJsonlExportReceipt:
    """Non-authority receipt for a JSONL export of database events."""

    SCHEMA: ClassVar[str] = EVENT_EXPORT_RECEIPT_SCHEMA

    export_id: str
    stream_id: str
    destination: str
    artifact_digest: str
    event_count: int
    earliest_sequence: int
    latest_sequence: int
    authority_class: StateAuthorityClass
    recorded_at: str

    def __post_init__(self) -> None:
        if self.authority_class is not StateAuthorityClass.EXPORT:
            raise DatabaseEventLogAuthorityError(
                "JSONL export receipts must use export authority"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "export_id": self.export_id,
            "stream_id": self.stream_id,
            "destination": self.destination,
            "artifact_digest": self.artifact_digest,
            "event_count": self.event_count,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "authority_class": self.authority_class.value,
            "recorded_at": self.recorded_at,
            "authoritative": False,
            "is_export_only": True,
        }


@dataclass(frozen=True)
class ApplicationAuditRecord:
    """Explicit application audit entry (never inferred from Quack diagnostics)."""

    SCHEMA: ClassVar[str] = APPLICATION_AUDIT_SCHEMA

    audit_id: str
    action: str
    actor_id: str
    subject_id: str
    task_cid: str
    session_id: str
    recorded_at: str
    event_id: str
    body: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "audit_id": self.audit_id,
            "action": self.action,
            "actor_id": self.actor_id,
            "subject_id": self.subject_id,
            "task_cid": self.task_cid,
            "session_id": self.session_id,
            "recorded_at": self.recorded_at,
            "event_id": self.event_id,
            "body": dict(self.body),
            "source": "application_explicit",
        }


# ---------------------------------------------------------------------------
# DatabaseEventLog
# ---------------------------------------------------------------------------


class DatabaseEventLog:
    """Append-only DuckDB event store with cursor polling and export-only JSONL.

    Interface: ``DatabaseEventLog@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_EVENT_LOG_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_EVENT_LOG_SCHEMA
    VERSION: ClassVar[int] = DATABASE_EVENT_LOG_VERSION

    def __init__(
        self,
        database_path: Path | str,
        *,
        store_id: str = "control.duckdb",
        default_stream_id: str = DEFAULT_STREAM_ID,
        install_schema: bool = True,
        owner_id: str = "database-event-log",
        max_recursive_log_depth: int = MAX_RECURSIVE_LOG_DEPTH,
    ) -> None:
        self.database_path = Path(database_path)
        self.store_id = _text(store_id, "store_id")
        self.default_stream_id = _text(default_stream_id, "default_stream_id")
        self.owner_id = _text(owner_id, "owner_id")
        self.max_recursive_log_depth = _nonnegative_int(
            max_recursive_log_depth, "max_recursive_log_depth"
        )
        self._install_schema = bool(install_schema)
        self._closed = False
        self._opened = False
        self._lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    def open(self) -> "DatabaseEventLog":
        with self._lock:
            if self._closed:
                raise DatabaseEventLogNotOpenError(
                    "cannot reopen a closed DatabaseEventLog"
                )
            if self._opened:
                return self
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            if self._install_schema:
                install_control_plane_schema(
                    self.database_path,
                    application_version="0.0.45",
                    tool_version="1.5.2",
                    owner_id=self.owner_id,
                )
            with open_duckdb_connection(self.database_path) as connection:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                # Ensure the default stream head exists for empty stores.
                self._ensure_stream_head_locked(
                    connection, self.default_stream_id
                )
            self._opened = True
            return self

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._opened = False

    def __enter__(self) -> "DatabaseEventLog":
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self.is_open:
            raise DatabaseEventLogNotOpenError(
                "DatabaseEventLog is not open"
            )

    # -- stream heads --------------------------------------------------------

    def _ensure_stream_head_locked(
        self, connection: Any, stream_id: str
    ) -> StreamHead:
        stream = _text(stream_id, "stream_id")
        row = connection.execute(
            """
            SELECT stream_id, snapshot_id, latest_sequence, earliest_sequence,
                   last_event_id, last_global_sequence, updated_at, revision
            FROM event_stream_heads WHERE stream_id = ? LIMIT 1
            """,
            [stream],
        ).fetchone()
        if row is not None:
            return StreamHead(
                stream_id=str(row[0]),
                snapshot_id=str(row[1]),
                latest_sequence=int(row[2]),
                earliest_sequence=int(row[3]),
                last_event_id=str(row[4] or ""),
                last_global_sequence=int(row[5]),
                updated_at=str(row[6]),
                revision=int(row[7]),
            )
        snapshot = _snapshot_for_stream(stream)
        stamp = utc_now()
        connection.execute(
            """
            INSERT INTO event_stream_heads (
                stream_id, snapshot_id, latest_sequence, earliest_sequence,
                last_event_id, last_global_sequence, updated_at, revision
            ) VALUES (?, ?, 0, 0, '', 0, ?, 0)
            """,
            [stream, snapshot, stamp],
        )
        return StreamHead(
            stream_id=stream,
            snapshot_id=snapshot,
            latest_sequence=0,
            earliest_sequence=0,
            last_event_id="",
            last_global_sequence=0,
            updated_at=stamp,
            revision=0,
        )

    def stream_head(self, stream_id: str | None = None) -> StreamHead:
        self._require_open()
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            return self._ensure_stream_head_locked(connection, stream)

    def initial_cursor(self, stream_id: str | None = None) -> EventCursor:
        head = self.stream_head(stream_id)
        return EventCursor.initial(head.stream_id, snapshot_id=head.snapshot_id)

    def latest_cursor(self, stream_id: str | None = None) -> EventCursor:
        return self.stream_head(stream_id).to_cursor()

    # -- append event --------------------------------------------------------

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
        recorded_at: str | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Append one immutable typed domain event and advance the stream head."""

        self._require_open()
        kind = _text(event_type, "event_type")
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        body_in = dict(payload or {})
        for field_name in _RESERVED_EVENT_FIELDS:
            body_in.pop(field_name, None)
        default_limit = (
            MAX_RECEIPT_BYTES
            if "receipt" in kind.casefold()
            else MAX_PROJECTION_BYTES
        )
        if max_bytes is not None:
            limit = min(_positive_int(max_bytes, "max_bytes"), default_limit)
        else:
            limit = default_limit
        body = _canonical_mapping(body_in, maximum=limit)
        stamp = str(recorded_at or "").strip() or utc_now()
        task = _text(task_cid, "task_cid", required=False)
        attempt = _text(attempt_id, "attempt_id", required=False)
        session = _text(session_id, "session_id", required=False)

        with self._lock, open_duckdb_connection(self.database_path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                head = self._ensure_stream_head_locked(connection, stream)
                sequence = int(head.latest_sequence) + 1
                previous_event_id = str(head.last_event_id or "")
                global_row = connection.execute(
                    "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
                ).fetchone()
                global_sequence = int(global_row[0] if global_row else 0) + 1
                envelope: dict[str, Any] = {
                    "type": kind,
                    "timestamp": stamp,
                    **body,
                    "stream_id": stream,
                    "snapshot_id": head.snapshot_id,
                    "sequence": sequence,
                    "previous_event_id": previous_event_id,
                    "global_sequence": global_sequence,
                    "task_cid": task,
                    "attempt_id": attempt,
                    "session_id": session,
                    "recorded_at": stamp,
                }
                event_id = _event_identity(envelope)
                envelope["event_id"] = event_id

                # Reject rewrites of an existing event_id with a different body.
                existing = connection.execute(
                    """
                    SELECT event_id, stream_id, sequence, body_json
                    FROM domain_events WHERE event_id = ? LIMIT 1
                    """,
                    [event_id],
                ).fetchone()
                if existing is not None:
                    # Identity already durable; treat as exact idempotent replay.
                    stored_body = json.loads(str(existing[3] or "{}"))
                    if (
                        str(existing[1]) != stream
                        or int(existing[2]) != sequence
                    ):
                        # Same content identity at a different position is
                        # impossible for content-addressed envelopes that include
                        # sequence; fail closed if the store is inconsistent.
                        raise DatabaseEventLogImmutabilityError(
                            f"event_id {event_id} already bound immutably"
                        )
                    connection.execute("ROLLBACK")
                    return dict(stored_body) if isinstance(stored_body, dict) else envelope

                # Reject sequence collision with a different identity.
                conflict = connection.execute(
                    """
                    SELECT event_id FROM domain_events
                    WHERE stream_id = ? AND sequence = ? LIMIT 1
                    """,
                    [stream, sequence],
                ).fetchone()
                if conflict is not None:
                    raise DatabaseEventLogImmutabilityError(
                        f"stream sequence {sequence} is immutable and occupied"
                    )

                connection.execute(
                    """
                    INSERT INTO domain_events (
                        event_id, stream_id, sequence, global_sequence,
                        event_type, task_cid, attempt_id, session_id,
                        recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        event_id,
                        stream,
                        sequence,
                        global_sequence,
                        kind,
                        task,
                        attempt,
                        session,
                        stamp,
                        json.dumps(
                            envelope, sort_keys=True, separators=(",", ":")
                        ),
                    ],
                )
                earliest = (
                    sequence
                    if int(head.earliest_sequence) == 0
                    else int(head.earliest_sequence)
                )
                connection.execute(
                    """
                    UPDATE event_stream_heads
                    SET latest_sequence = ?,
                        earliest_sequence = ?,
                        last_event_id = ?,
                        last_global_sequence = ?,
                        updated_at = ?,
                        revision = revision + 1
                    WHERE stream_id = ?
                    """,
                    [
                        sequence,
                        earliest,
                        event_id,
                        global_sequence,
                        stamp,
                        stream,
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        return envelope

    # -- polling / replay ----------------------------------------------------

    def _load_stream_events(
        self,
        connection: Any,
        *,
        stream_id: str,
        after_sequence: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT event_id, stream_id, sequence, global_sequence, event_type,
                   task_cid, attempt_id, session_id, recorded_at, body_json
            FROM domain_events
            WHERE stream_id = ? AND sequence > ?
            ORDER BY sequence ASC
            LIMIT ?
            """,
            [stream_id, after_sequence, limit],
        ).fetchall()
        events: list[dict[str, Any]] = []
        seen: dict[int, str] = {}
        for row in rows:
            event_id = str(row[0])
            sequence = int(row[2])
            known = seen.get(sequence)
            if known is not None:
                if known != event_id:
                    raise CursorReplayError(
                        f"event sequence {sequence} has conflicting identities"
                    )
                # Exact duplicate coalesced — no second effect.
                continue
            seen[sequence] = event_id
            body_raw = str(row[9] or "{}")
            try:
                body = json.loads(body_raw)
            except json.JSONDecodeError as exc:
                raise CursorReplayError(
                    f"event {event_id} has malformed body_json"
                ) from exc
            if not isinstance(body, dict):
                raise CursorReplayError(
                    f"event {event_id} body must be an object"
                )
            event = dict(body)
            event.setdefault("event_id", event_id)
            event.setdefault("stream_id", str(row[1]))
            event.setdefault("sequence", sequence)
            event.setdefault("global_sequence", int(row[3]))
            event.setdefault("type", str(row[4]))
            event.setdefault("task_cid", str(row[5] or ""))
            event.setdefault("attempt_id", str(row[6] or ""))
            event.setdefault("session_id", str(row[7] or ""))
            event.setdefault("recorded_at", str(row[8] or ""))
            # Verify immutable identity still matches stored envelope.
            expected = _event_identity(event)
            if event_id != expected:
                raise DatabaseEventLogImmutabilityError(
                    f"event {sequence} identity was rewritten"
                )
            event["event_id"] = event_id
            events.append(event)
        return events

    def poll(
        self,
        cursor: EventCursor | Mapping[str, Any] | str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        stream_id: str | None = None,
    ) -> EventPage:
        """Replay at most ``limit`` events strictly after ``cursor``."""

        self._require_open()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if limit > MAX_PAGE_LIMIT:
            raise DatabaseEventLogBoundsError(
                f"limit exceeds the {MAX_PAGE_LIMIT} page bound"
            )
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
        stream = _text(stream_id or selected.stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            head = self._ensure_stream_head_locked(connection, stream)
            selected.assert_replayable(
                stream_id=head.stream_id,
                earliest_position=int(head.earliest_sequence or 0),
                latest_position=int(head.latest_sequence or 0),
                snapshot_id=head.snapshot_id,
            )
            if selected.position > 0:
                if (
                    head.earliest_sequence
                    and selected.position < head.earliest_sequence
                ):
                    raise CursorReplayError(
                        "event cursor predates the retained replay window"
                    )
                anchor = connection.execute(
                    """
                    SELECT event_id FROM domain_events
                    WHERE stream_id = ? AND sequence = ? LIMIT 1
                    """,
                    [stream, selected.position],
                ).fetchone()
                if anchor is None:
                    raise CursorReplayError(
                        "event cursor anchor is not present in the stream"
                    )
                if str(anchor[0]) != selected.last_event_id:
                    raise CursorReplayError(
                        "event cursor anchor does not match the replay population"
                    )
            # An initial cursor after retention means "start of the retained
            # window", not "sequence zero must still exist".
            after_sequence = selected.position
            replay_cursor = selected
            if (
                selected.position == 0
                and int(head.earliest_sequence or 0) > 1
            ):
                after_sequence = int(head.earliest_sequence) - 1
                replay_cursor = EventCursor.initial(
                    stream, snapshot_id=head.snapshot_id
                )
            # Fetch one extra to detect has_more accurately after coalescing.
            raw = self._load_stream_events(
                connection,
                stream_id=stream,
                after_sequence=after_sequence,
                limit=limit + 1,
            )
            if (
                selected.position == 0
                and int(head.earliest_sequence or 0) > 1
                and raw
            ):
                # Build a page relative to the retained prefix without requiring
                # deleted prefix sequences to still be present.
                page_events = tuple(raw[:limit])
                if page_events:
                    last = page_events[-1]
                    next_cursor = EventCursor(
                        stream_id=stream,
                        snapshot_id=head.snapshot_id,
                        position=int(last["sequence"]),
                        last_event_id=str(last["event_id"]),
                    )
                else:
                    next_cursor = replay_cursor
                has_more = len(raw) > limit or (
                    next_cursor.position < int(head.latest_sequence or 0)
                )
                return EventPage(
                    events=page_events,
                    next_cursor=next_cursor,
                    has_more=has_more,
                )
            page = replay_event_page(
                raw[:limit],
                selected,
                limit=limit,
                stream_id=stream,
                snapshot_id=head.snapshot_id,
            )
            has_more = len(raw) > limit or (
                page.next_cursor.position < int(head.latest_sequence or 0)
            )
            if has_more == page.has_more:
                return page
            return EventPage(
                events=page.events,
                next_cursor=page.next_cursor,
                has_more=has_more,
            )

    read_page = poll
    read_event_page = poll

    # -- consumer checkpoints ------------------------------------------------

    def save_checkpoint(
        self,
        consumer_id: str,
        cursor: EventCursor | Mapping[str, Any] | str,
        *,
        stream_id: str | None = None,
    ) -> ConsumerCheckpoint:
        """Persist a consumer resume cursor; identical payloads are no-ops."""

        self._require_open()
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
        stream = _text(stream_id or selected.stream_id, "stream_id")
        if selected.stream_id != stream:
            raise CursorReplayError(
                "consumer checkpoint stream does not match cursor"
            )
        head = self.stream_head(stream)
        selected.assert_replayable(
            stream_id=head.stream_id,
            earliest_position=int(head.earliest_sequence or 0),
            latest_position=int(head.latest_sequence or 0),
            snapshot_id=head.snapshot_id,
        )
        checkpoint = ConsumerCheckpoint(
            consumer_id=consumer_id,
            cursor=selected,
        )
        body = json.dumps(
            checkpoint.to_dict(), sort_keys=True, separators=(",", ":")
        )
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            existing = connection.execute(
                """
                SELECT checkpoint_id FROM event_consumer_checkpoints
                WHERE consumer_id = ? AND stream_id = ? LIMIT 1
                """,
                [checkpoint.consumer_id, selected.stream_id],
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO event_consumer_checkpoints (
                        consumer_id, stream_id, snapshot_id, position,
                        last_event_id, checkpoint_id, updated_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        checkpoint.consumer_id,
                        selected.stream_id,
                        selected.snapshot_id,
                        selected.position,
                        selected.last_event_id,
                        checkpoint.checkpoint_id,
                        checkpoint.updated_at,
                        body,
                    ],
                )
            else:
                connection.execute(
                    """
                    UPDATE event_consumer_checkpoints
                    SET snapshot_id = ?,
                        position = ?,
                        last_event_id = ?,
                        checkpoint_id = ?,
                        updated_at = ?,
                        body_json = ?
                    WHERE consumer_id = ? AND stream_id = ?
                    """,
                    [
                        selected.snapshot_id,
                        selected.position,
                        selected.last_event_id,
                        checkpoint.checkpoint_id,
                        checkpoint.updated_at,
                        body,
                        checkpoint.consumer_id,
                        selected.stream_id,
                    ],
                )
        return checkpoint

    def load_checkpoint(
        self,
        consumer_id: str,
        *,
        stream_id: str | None = None,
    ) -> ConsumerCheckpoint | None:
        self._require_open()
        consumer = _text(consumer_id, "consumer_id")
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT body_json FROM event_consumer_checkpoints
                WHERE consumer_id = ? AND stream_id = ? LIMIT 1
                """,
                [consumer, stream],
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        return ConsumerCheckpoint.from_dict(payload)

    # -- structured logs / traces / metrics / audit --------------------------

    def append_log(
        self,
        message: str,
        *,
        severity: LogSeverity | str = LogSeverity.INFO,
        component: str = "runtime",
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        body: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Append a structured log. Recursive re-entry is depth-bounded."""

        self._require_open()
        depth = _enter_logging()
        try:
            if depth > self.max_recursive_log_depth:
                # Bound recursive logging: drop nested log effects silently.
                return None
            if isinstance(severity, LogSeverity):
                level = severity
            else:
                level = LogSeverity(str(severity).strip().lower())
            stamp = str(recorded_at or "").strip() or utc_now()
            payload = _canonical_mapping(
                dict(body or {}), maximum=MAX_PROJECTION_BYTES
            )
            record = {
                "message": _text(message, "message"),
                "severity": level.value,
                "component": _text(component, "component"),
                "task_cid": _text(task_cid, "task_cid", required=False),
                "attempt_id": _text(attempt_id, "attempt_id", required=False),
                "session_id": _text(session_id, "session_id", required=False),
                "trace_id": _text(trace_id, "trace_id", required=False),
                "span_id": _text(span_id, "span_id", required=False),
                "recorded_at": stamp,
                "body": payload,
            }
            log_id = content_identity(
                {
                    "schema": "structured-log@1",
                    **record,
                }
            )
            with self._lock, open_duckdb_connection(
                self.database_path
            ) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM structured_logs WHERE log_id = ? LIMIT 1",
                    [log_id],
                ).fetchone()
                if exists is None:
                    connection.execute(
                        """
                        INSERT INTO structured_logs (
                            log_id, severity, component, trace_id, span_id,
                            task_cid, attempt_id, session_id, recorded_at,
                            message, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            log_id,
                            level.value,
                            record["component"],
                            record["trace_id"],
                            record["span_id"],
                            record["task_cid"],
                            record["attempt_id"],
                            record["session_id"],
                            stamp,
                            record["message"],
                            json.dumps(
                                payload, sort_keys=True, separators=(",", ":")
                            ),
                        ],
                    )
            record["log_id"] = log_id
            return record
        finally:
            _exit_logging()

    def append_trace(
        self,
        message: str,
        *,
        component: str = "runtime",
        trace_id: str = "",
        span_id: str = "",
        task_cid: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return self.append_log(
            message,
            severity=LogSeverity.TRACE,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            task_cid=task_cid,
            body=body,
        )

    def append_audit(
        self,
        action: str,
        *,
        actor_id: str = "",
        subject_id: str = "",
        task_cid: str = "",
        session_id: str = "",
        body: Mapping[str, Any] | None = None,
        emit_event: bool = True,
        recorded_at: str | None = None,
    ) -> ApplicationAuditRecord:
        """Insert an explicit application audit record.

        Quack diagnostics are never consulted. Audit is always an explicit
        application write. Optional domain-event projection uses a dedicated
        event type so consumers can distinguish audit from diagnostics.
        """

        self._require_open()
        depth = _enter_logging()
        try:
            if depth > self.max_recursive_log_depth:
                raise DatabaseEventLogBoundsError(
                    "recursive application audit logging exceeded its bound"
                )
            stamp = str(recorded_at or "").strip() or utc_now()
            payload = _canonical_mapping(
                dict(body or {}), maximum=MAX_RECEIPT_BYTES
            )
            action_text = _text(action, "action")
            actor = _text(actor_id, "actor_id", required=False)
            subject = _text(subject_id, "subject_id", required=False)
            task = _text(task_cid, "task_cid", required=False)
            session = _text(session_id, "session_id", required=False)
            event_id = ""
            if emit_event:
                event = self.append_event(
                    APPLICATION_AUDIT_EVENT_TYPE,
                    {
                        "action": action_text,
                        "actor_id": actor,
                        "subject_id": subject,
                        "audit_body": payload,
                    },
                    task_cid=task,
                    session_id=session,
                    recorded_at=stamp,
                )
                event_id = str(event["event_id"])
            audit_payload = {
                "schema": APPLICATION_AUDIT_SCHEMA,
                "action": action_text,
                "actor_id": actor,
                "subject_id": subject,
                "task_cid": task,
                "session_id": session,
                "recorded_at": stamp,
                "event_id": event_id,
                "body": payload,
                "source": "application_explicit",
            }
            audit_id = content_identity(audit_payload)
            with self._lock, open_duckdb_connection(
                self.database_path
            ) as connection:
                audit_exists = connection.execute(
                    "SELECT 1 FROM application_audit_records "
                    "WHERE audit_id = ? LIMIT 1",
                    [audit_id],
                ).fetchone()
                if audit_exists is None:
                    connection.execute(
                        """
                        INSERT INTO application_audit_records (
                            audit_id, action, actor_id, subject_id, task_cid,
                            session_id, recorded_at, event_id, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            audit_id,
                            action_text,
                            actor,
                            subject,
                            task,
                            session,
                            stamp,
                            event_id,
                            json.dumps(
                                payload, sort_keys=True, separators=(",", ":")
                            ),
                        ],
                    )
                # Mirror into structured_logs under the application_audit
                # component so operators can query one log surface without
                # elevating quack_query_telemetry to audit authority.
                log_id = content_identity(
                    {
                        "schema": "structured-log@1",
                        "audit_id": audit_id,
                        "component": APPLICATION_AUDIT_COMPONENT,
                    }
                )
                log_exists = connection.execute(
                    "SELECT 1 FROM structured_logs WHERE log_id = ? LIMIT 1",
                    [log_id],
                ).fetchone()
                if log_exists is None:
                    connection.execute(
                        """
                        INSERT INTO structured_logs (
                            log_id, severity, component, trace_id, span_id,
                            task_cid, attempt_id, session_id, recorded_at,
                            message, body_json
                        ) VALUES (?, ?, ?, '', '', ?, '', ?, ?, ?, ?)
                        """,
                        [
                            log_id,
                            LogSeverity.INFO.value,
                            APPLICATION_AUDIT_COMPONENT,
                            task,
                            session,
                            stamp,
                            f"audit:{action_text}",
                            json.dumps(
                                {
                                    "audit_id": audit_id,
                                    "action": action_text,
                                    "source": "application_explicit",
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ],
                    )
            return ApplicationAuditRecord(
                audit_id=audit_id,
                action=action_text,
                actor_id=actor,
                subject_id=subject,
                task_cid=task,
                session_id=session,
                recorded_at=stamp,
                event_id=event_id,
                body=MappingProxyType(payload),
            )
        finally:
            _exit_logging()

    def list_audit(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        subject_id: str = "",
    ) -> tuple[ApplicationAuditRecord, ...]:
        self._require_open()
        selected_limit = _positive_int(limit, "limit")
        subject = _text(subject_id, "subject_id", required=False)
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            if subject:
                rows = connection.execute(
                    """
                    SELECT audit_id, action, actor_id, subject_id, task_cid,
                           session_id, recorded_at, event_id, body_json
                    FROM application_audit_records
                    WHERE subject_id = ?
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [subject, selected_limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT audit_id, action, actor_id, subject_id, task_cid,
                           session_id, recorded_at, event_id, body_json
                    FROM application_audit_records
                    ORDER BY recorded_at ASC
                    LIMIT ?
                    """,
                    [selected_limit],
                ).fetchall()
        records: list[ApplicationAuditRecord] = []
        for row in rows:
            body = json.loads(str(row[8] or "{}"))
            if not isinstance(body, dict):
                body = {}
            records.append(
                ApplicationAuditRecord(
                    audit_id=str(row[0]),
                    action=str(row[1]),
                    actor_id=str(row[2] or ""),
                    subject_id=str(row[3] or ""),
                    task_cid=str(row[4] or ""),
                    session_id=str(row[5] or ""),
                    recorded_at=str(row[6]),
                    event_id=str(row[7] or ""),
                    body=MappingProxyType(body),
                )
            )
        return tuple(records)

    def quack_diagnostics_are_not_audit(self) -> bool:
        """Policy witness: Quack telemetry is diagnostic, never application audit."""

        return True

    def record_metric(
        self,
        metric_name: str,
        value_milli: int,
        *,
        unit: str = "count",
        description: str = "",
        labels: Mapping[str, Any] | None = None,
        observed_at: str | None = None,
        stratum: str = "",
    ) -> dict[str, Any]:
        self._require_open()
        name = _text(metric_name, "metric_name")
        if isinstance(value_milli, bool) or not isinstance(value_milli, int):
            raise DatabaseEventLogError("value_milli must be an integer")
        stamp = str(observed_at or "").strip() or utc_now()
        unit_text = _text(unit, "unit")
        description_text = _text(description, "description", required=False)
        labels_obj = _canonical_mapping(
            dict(labels or {}), maximum=MAX_RECEIPT_BYTES
        )
        metric_id = content_identity(
            {
                "schema": "metric@1",
                "metric_name": name,
                "unit": unit_text,
            }
        )
        sample_id = content_identity(
            {
                "schema": "metric-sample@1",
                "metric_id": metric_id,
                "observed_at": stamp,
                "value_milli": value_milli,
                "labels": labels_obj,
                "stratum": str(stratum or ""),
            }
        )
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            existing = connection.execute(
                "SELECT metric_id FROM metrics WHERE metric_name = ? LIMIT 1",
                [name],
            ).fetchone()
            if existing is not None:
                resolved_metric_id = str(existing[0])
            else:
                connection.execute(
                    """
                    INSERT INTO metrics (
                        metric_id, metric_name, unit, description, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [metric_id, name, unit_text, description_text, stamp],
                )
                resolved_metric_id = metric_id
            sample_exists = connection.execute(
                "SELECT 1 FROM metric_samples WHERE sample_id = ? LIMIT 1",
                [sample_id],
            ).fetchone()
            if sample_exists is None:
                connection.execute(
                    """
                    INSERT INTO metric_samples (
                        sample_id, metric_id, observed_at, value_milli,
                        labels_json, stratum
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        sample_id,
                        resolved_metric_id,
                        stamp,
                        value_milli,
                        json.dumps(
                            labels_obj, sort_keys=True, separators=(",", ":")
                        ),
                        str(stratum or ""),
                    ],
                )
        return {
            "metric_id": resolved_metric_id,
            "sample_id": sample_id,
            "metric_name": name,
            "value_milli": value_milli,
            "observed_at": stamp,
            "unit": unit_text,
            "labels": labels_obj,
            "stratum": str(stratum or ""),
        }

    # -- integrity / retention -----------------------------------------------

    def write_integrity_checkpoint(
        self, stream_id: str | None = None
    ) -> IntegrityCheckpoint:
        self._require_open()
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            head = self._ensure_stream_head_locked(connection, stream)
            rows = connection.execute(
                """
                SELECT event_id, sequence FROM domain_events
                WHERE stream_id = ?
                ORDER BY sequence ASC
                """,
                [stream],
            ).fetchall()
            population = [
                {"event_id": str(row[0]), "sequence": int(row[1])}
                for row in rows
            ]
            digest = _sha256_hex(canonical_json_bytes(population))
            stamp = utc_now()
            payload = {
                "schema": INTEGRITY_CHECKPOINT_SCHEMA,
                "stream_id": stream,
                "earliest_sequence": head.earliest_sequence,
                "latest_sequence": head.latest_sequence,
                "last_event_id": head.last_event_id,
                "population_digest": digest,
                "recorded_at": stamp,
            }
            checkpoint_id = content_identity(payload)
            exists = connection.execute(
                "SELECT 1 FROM event_integrity_checkpoints "
                "WHERE checkpoint_id = ? LIMIT 1",
                [checkpoint_id],
            ).fetchone()
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO event_integrity_checkpoints (
                        checkpoint_id, stream_id, earliest_sequence,
                        latest_sequence, last_event_id, population_digest,
                        recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        checkpoint_id,
                        stream,
                        head.earliest_sequence,
                        head.latest_sequence,
                        head.last_event_id,
                        digest,
                        stamp,
                        json.dumps(
                            payload, sort_keys=True, separators=(",", ":")
                        ),
                    ],
                )
        return IntegrityCheckpoint(
            checkpoint_id=checkpoint_id,
            stream_id=stream,
            earliest_sequence=head.earliest_sequence,
            latest_sequence=head.latest_sequence,
            last_event_id=head.last_event_id,
            population_digest=digest,
            recorded_at=stamp,
        )

    def apply_retention(
        self,
        stream_id: str | None = None,
        *,
        retain_recent: int,
    ) -> dict[str, Any]:
        """Drop prefix events while preserving stream head and consumer safety.

        Retention never rewrites retained event_ids or sequences. Consumers
        whose checkpoints fall before the new earliest sequence must fail
        closed on the next poll (cursor expiry), not silently skip.
        """

        self._require_open()
        keep = _positive_int(retain_recent, "retain_recent")
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                head = self._ensure_stream_head_locked(connection, stream)
                if head.latest_sequence <= keep:
                    connection.execute("COMMIT")
                    return {
                        "stream_id": stream,
                        "deleted": 0,
                        "earliest_sequence": head.earliest_sequence,
                        "latest_sequence": head.latest_sequence,
                    }
                new_earliest = head.latest_sequence - keep + 1
                deleted_row = connection.execute(
                    """
                    SELECT COUNT(*) FROM domain_events
                    WHERE stream_id = ? AND sequence < ?
                    """,
                    [stream, new_earliest],
                ).fetchone()
                deleted = int(deleted_row[0] if deleted_row else 0)
                connection.execute(
                    """
                    DELETE FROM domain_events
                    WHERE stream_id = ? AND sequence < ?
                    """,
                    [stream, new_earliest],
                )
                stamp = utc_now()
                connection.execute(
                    """
                    UPDATE event_stream_heads
                    SET earliest_sequence = ?, updated_at = ?,
                        revision = revision + 1
                    WHERE stream_id = ?
                    """,
                    [new_earliest, stamp, stream],
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        return {
            "stream_id": stream,
            "deleted": deleted,
            "earliest_sequence": new_earliest,
            "latest_sequence": head.latest_sequence,
        }

    # -- JSONL export (non-authority) ----------------------------------------

    def export_jsonl(
        self,
        destination: Path | str,
        *,
        stream_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> EventJsonlExportReceipt:
        """Render events to JSONL. The file is export-only, never authority."""

        self._require_open()
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        after = _nonnegative_int(after_sequence, "after_sequence")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_limit = (
            MAX_PAGE_LIMIT * 64
            if limit is None
            else _positive_int(limit, "limit")
        )
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            events = self._load_stream_events(
                connection,
                stream_id=stream,
                after_sequence=after,
                limit=selected_limit,
            )
        lines = [
            json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
            for event in events
        ]
        payload = "".join(lines).encode("utf-8")
        digest = _sha256_hex(payload)
        # Atomic write so partial exports are never observed as complete.
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream_handle:
                stream_handle.write(payload)
                stream_handle.flush()
                os.fsync(stream_handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        stamp = utc_now()
        earliest = int(events[0]["sequence"]) if events else 0
        latest = int(events[-1]["sequence"]) if events else 0
        export_payload = {
            "schema": EVENT_EXPORT_RECEIPT_SCHEMA,
            "stream_id": stream,
            "destination": str(path),
            "artifact_digest": digest,
            "event_count": len(events),
            "earliest_sequence": earliest,
            "latest_sequence": latest,
            "authority_class": StateAuthorityClass.EXPORT.value,
            "recorded_at": stamp,
        }
        export_id = content_identity(export_payload)
        receipt = EventJsonlExportReceipt(
            export_id=export_id,
            stream_id=stream,
            destination=str(path),
            artifact_digest=digest,
            event_count=len(events),
            earliest_sequence=earliest,
            latest_sequence=latest,
            authority_class=StateAuthorityClass.EXPORT,
            recorded_at=stamp,
        )
        with self._lock, open_duckdb_connection(self.database_path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM event_log_exports WHERE export_id = ? LIMIT 1",
                [export_id],
            ).fetchone()
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO event_log_exports (
                        export_id, stream_id, destination, artifact_digest,
                        event_count, earliest_sequence, latest_sequence,
                        authority_class, recorded_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        export_id,
                        stream,
                        str(path),
                        digest,
                        len(events),
                        earliest,
                        latest,
                        StateAuthorityClass.EXPORT.value,
                        stamp,
                        json.dumps(
                            receipt.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ],
                )
        return receipt

    def delete_export_has_no_authority_effect(
        self,
        destination: Path | str,
        *,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete an exported JSONL file and prove database authority is intact."""

        self._require_open()
        path = Path(destination)
        stream = _text(stream_id or self.default_stream_id, "stream_id")
        before = self.stream_head(stream)
        watermark_before = before.latest_sequence
        events_before = self.poll(
            self.initial_cursor(stream), limit=MAX_PAGE_LIMIT, stream_id=stream
        )
        if path.exists():
            path.unlink()
        after = self.stream_head(stream)
        events_after = self.poll(
            self.initial_cursor(stream), limit=MAX_PAGE_LIMIT, stream_id=stream
        )
        if after.latest_sequence != watermark_before:
            raise DatabaseEventLogAuthorityError(
                "export deletion must not change stream authority"
            )
        if [e.get("event_id") for e in events_before.events] != [
            e.get("event_id") for e in events_after.events
        ]:
            raise DatabaseEventLogAuthorityError(
                "export deletion must not change pollable events"
            )
        return {
            "destination": str(path),
            "exists": path.exists(),
            "stream_id": stream,
            "latest_sequence": after.latest_sequence,
            "event_count": len(events_after.events),
            "authority_class": StateAuthorityClass.AUTHORITATIVE.value,
            "export_authority_class": StateAuthorityClass.EXPORT.value,
            "authoritative_effect": False,
        }

    # -- introspection -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "version": self.VERSION,
            "database_path": str(self.database_path),
            "store_id": self.store_id,
            "default_stream_id": self.default_stream_id,
            "is_open": self.is_open,
            "max_recursive_log_depth": self.max_recursive_log_depth,
            "authority_class": StateAuthorityClass.AUTHORITATIVE.value,
            "jsonl_authority_class": StateAuthorityClass.EXPORT.value,
        }


def open_database_event_log(
    database_path: Path | str,
    **kwargs: Any,
) -> DatabaseEventLog:
    """Open and return a ready DatabaseEventLog."""

    return DatabaseEventLog(database_path, **kwargs).open()


__all__ = [
    "APPLICATION_AUDIT_COMPONENT",
    "APPLICATION_AUDIT_EVENT_TYPE",
    "APPLICATION_AUDIT_SCHEMA",
    "ApplicationAuditRecord",
    "CONSUMER_CHECKPOINT_INTERFACE",
    "CONSUMER_CHECKPOINT_SCHEMA",
    "ConsumerCheckpoint",
    "DATABASE_EVENT_LOG_INTERFACE",
    "DATABASE_EVENT_LOG_SCHEMA",
    "DATABASE_EVENT_LOG_VERSION",
    "DEFAULT_PAGE_LIMIT",
    "DEFAULT_STREAM_ID",
    "DatabaseEventLog",
    "DatabaseEventLogAuthorityError",
    "DatabaseEventLogBoundsError",
    "DatabaseEventLogError",
    "DatabaseEventLogImmutabilityError",
    "DatabaseEventLogNotOpenError",
    "EVENT_CURSOR_INTERFACE",
    "EVENT_EXPORT_RECEIPT_SCHEMA",
    "EventJsonlExportReceipt",
    "INTEGRITY_CHECKPOINT_SCHEMA",
    "IntegrityCheckpoint",
    "LogSeverity",
    "MAX_RECURSIVE_LOG_DEPTH",
    "QUACK_DIAGNOSTIC_COMPONENT",
    "RecordKind",
    "STREAM_HEAD_SCHEMA",
    "StreamHead",
    "open_database_event_log",
    "utc_now",
]
