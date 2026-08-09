"""DuckDB-backed domain event, audit, log, metric, and cursor authority.

DQP-013 / DatabaseEventLog@1
============================

Replaces JSONL as the authoritative event, audit, structured-log, metric,
trace, stream-head, retention, integrity-checkpoint, and consumer-cursor store.
JSONL remains an export-only rendering surface: deleting an exported file has
no authority effect on the database.

Interfaces:

* ``DatabaseEventLog@1`` — append-only typed streams with monotonic sequences
* ``EventCursor@1`` — re-exported content-addressed replay cursor
* ``ConsumerCheckpoint@1`` — durable per-consumer stream positions

Application audit is explicit (``append_audit``) rather than inferred from
Quack diagnostics. Recursive structured-log emission is depth-bounded so a
logging failure cannot re-enter unbounded logging. Cold import of this module
performs no filesystem, database, network, provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..control.control_contracts import (
    ABSOLUTE_MAX_CONTROL_ITEMS,
    CursorReplayError,
    EventCursor,
    EventCursorError,
    EventPage,
    replay_event_page,
)
from ..task_sources.control_plane_migrations import duckdb_available
from ..task_sources.control_plane_schema import install_control_plane_schema
from ..task_sources.duckdb_state import open_duckdb_connection

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
STREAM_HEAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/event-stream-head@1"
)
INTEGRITY_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/event-integrity-checkpoint@1"
)
AUDIT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/application-audit@1"
)
STRUCTURED_LOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/structured-log@1"
)
METRIC_SAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/metric-sample@1"
)

DATABASE_EVENT_LOG_VERSION: Final[int] = 1
DEFAULT_PAGE_LIMIT: Final[int] = 256
MAX_PAGE_LIMIT: Final[int] = min(1_024, ABSOLUTE_MAX_CONTROL_ITEMS)
MAX_EVENT_BODY_BYTES: Final[int] = 262_144
MAX_LOG_MESSAGE_BYTES: Final[int] = 16_384
MAX_RECURSION_DEPTH: Final[int] = 2
DEFAULT_RETENTION_MIN_SEQUENCE: Final[int] = 1

AUDIT_EVENT_TYPE: Final[str] = "application.audit"
_REDACTED: Final[str] = "[REDACTED]"
_SENSITIVE_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "private_key",
    "credential",
)

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS event_stream_heads (
    stream_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    earliest_sequence BIGINT NOT NULL,
    latest_sequence BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL,
    last_global_sequence BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS event_consumer_checkpoints (
    consumer_id VARCHAR NOT NULL,
    stream_id VARCHAR NOT NULL,
    position BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    cursor_json VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (consumer_id, stream_id)
);

CREATE TABLE IF NOT EXISTS event_integrity_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    global_sequence BIGINT NOT NULL,
    event_count BIGINT NOT NULL,
    digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
"""

_LOG_DEPTH = threading.local()
_LOG_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseEventLogError(RuntimeError):
    """Base failure for database event-log operations."""


class DatabaseEventLogNotOpenError(DatabaseEventLogError):
    """Operation requires an open event log."""


class DatabaseEventLogIntegrityError(DatabaseEventLogError, CursorReplayError):
    """Stream integrity, identity, or cursor anchor failure."""


class DatabaseEventLogDuplicateError(DatabaseEventLogError):
    """An event identity collides with a different payload."""


class DatabaseEventLogPayloadError(DatabaseEventLogError, ValueError):
    """Payload bounds or canonicalization failure."""


class DatabaseEventLogUnavailableError(DatabaseEventLogError):
    """DuckDB dependency is not available."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json_bytes(value: Any, *, maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise DatabaseEventLogPayloadError(
            "payload must contain canonical JSON values"
        ) from exc
    if len(encoded) > maximum:
        raise DatabaseEventLogPayloadError(
            f"payload exceeds the {maximum}-byte bound"
        )
    return encoded


def _canonical_json(value: Any, *, maximum: int = MAX_EVENT_BODY_BYTES) -> str:
    return _canonical_json_bytes(value, maximum=maximum).decode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json_bytes(dict(value), maximum=MAX_EVENT_BODY_BYTES)
    ).hexdigest()
    return f"sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise DatabaseEventLogError(f"{name} is required")
    if "\x00" in text:
        raise DatabaseEventLogError(f"{name} contains NUL")
    if len(text.encode("utf-8")) > 4_096:
        raise DatabaseEventLogError(f"{name} exceeds the identity bound")
    return text


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatabaseEventLogError(f"{name} must be a non-negative integer")
    if value < 0:
        raise DatabaseEventLogError(f"{name} must be a non-negative integer")
    return value


def _positive_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit > MAX_PAGE_LIMIT:
        raise ValueError(f"limit exceeds the {MAX_PAGE_LIMIT} page bound")
    return limit


def _is_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS)


def redact_value(value: Any, *, depth: int = 0) -> Any:
    """Return a redacted deep copy of nested mappings/sequences."""

    if depth > 16:
        return _REDACTED
    if isinstance(value, Mapping):
        return {
            str(key): (
                _REDACTED
                if _is_sensitive_key(str(key))
                else redact_value(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, depth=depth + 1) for item in value]
    return value


def _log_depth() -> int:
    return int(getattr(_LOG_DEPTH, "depth", 0) or 0)


def _enter_log() -> int:
    depth = _log_depth() + 1
    _LOG_DEPTH.depth = depth
    return depth


def _exit_log() -> None:
    depth = max(0, _log_depth() - 1)
    _LOG_DEPTH.depth = depth


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return {key: row[key] for key in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        pass
    if hasattr(row, "_columns") and hasattr(row, "_values"):
        return {
            str(column): value
            for column, value in zip(row._columns, row._values)
        }
    raise DatabaseEventLogError("unsupported database row shape")


def _snapshot_for_path(path: Path) -> str:
    try:
        identity = str(path.resolve())
    except OSError:
        identity = str(path.absolute())
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"event-db-snapshot:sha256:{digest}"


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split simple semicolon-delimited DDL for DuckDB execute()."""

    statements: list[str] = []
    for chunk in sql_text.split(";"):
        statement = chunk.strip()
        if statement:
            statements.append(statement)
    return statements


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class LogSeverity(str, Enum):
    """Closed severity population for structured application logs."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class StreamHead:
    """Current retained window and tip of one event stream."""

    SCHEMA: Final[str] = STREAM_HEAD_SCHEMA

    stream_id: str
    snapshot_id: str
    earliest_sequence: int
    latest_sequence: int
    last_event_id: str = ""
    last_global_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _text(self.stream_id, "stream_id"))
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self,
            "earliest_sequence",
            _nonnegative(self.earliest_sequence, "earliest_sequence"),
        )
        object.__setattr__(
            self,
            "latest_sequence",
            _nonnegative(self.latest_sequence, "latest_sequence"),
        )
        object.__setattr__(
            self,
            "last_event_id",
            _text(self.last_event_id, "last_event_id", required=False),
        )
        object.__setattr__(
            self,
            "last_global_sequence",
            _nonnegative(self.last_global_sequence, "last_global_sequence"),
        )
        if self.earliest_sequence and self.latest_sequence:
            if self.earliest_sequence > self.latest_sequence + 1:
                raise DatabaseEventLogIntegrityError(
                    "stream head earliest sequence exceeds latest"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "snapshot_id": self.snapshot_id,
            "earliest_sequence": self.earliest_sequence,
            "latest_sequence": self.latest_sequence,
            "last_event_id": self.last_event_id,
            "last_global_sequence": self.last_global_sequence,
        }


@dataclass(frozen=True)
class ConsumerCheckpoint:
    """Durable consumer cursor bound to one stream and snapshot.

    Interface: ``ConsumerCheckpoint@1``
    """

    INTERFACE: Final[str] = CONSUMER_CHECKPOINT_INTERFACE
    SCHEMA: Final[str] = CONSUMER_CHECKPOINT_SCHEMA

    consumer_id: str
    cursor: EventCursor
    recorded_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id")
        )
        if not isinstance(self.cursor, EventCursor):
            if isinstance(self.cursor, Mapping):
                object.__setattr__(
                    self, "cursor", EventCursor.from_dict(self.cursor)
                )
            else:
                raise TypeError("cursor must be an EventCursor")
        recorded = str(self.recorded_at or "").strip() or _utc_iso()
        object.__setattr__(self, "recorded_at", recorded)

    @property
    def stream_id(self) -> str:
        return self.cursor.stream_id

    @property
    def position(self) -> int:
        return self.cursor.position

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "consumer_id": self.consumer_id,
            "cursor": self.cursor.to_record(),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConsumerCheckpoint":
        if not isinstance(value, Mapping):
            raise DatabaseEventLogError("consumer checkpoint must be an object")
        cursor_value = value.get("cursor")
        if isinstance(cursor_value, EventCursor):
            cursor = cursor_value
        elif isinstance(cursor_value, Mapping):
            cursor = EventCursor.from_dict(cursor_value)
        else:
            raise DatabaseEventLogError(
                "consumer checkpoint cursor is required"
            )
        return cls(
            consumer_id=str(value.get("consumer_id") or ""),
            cursor=cursor,
            recorded_at=str(value.get("recorded_at") or ""),
        )


@dataclass(frozen=True)
class IntegrityCheckpoint:
    """Digest of a stream prefix used for integrity / replay proofs."""

    SCHEMA: Final[str] = INTEGRITY_CHECKPOINT_SCHEMA

    checkpoint_id: str
    stream_id: str
    global_sequence: int
    event_count: int
    digest: str
    recorded_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_id", _text(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self, "stream_id", _text(self.stream_id, "stream_id", required=False)
        )
        object.__setattr__(
            self,
            "global_sequence",
            _nonnegative(self.global_sequence, "global_sequence"),
        )
        object.__setattr__(
            self, "event_count", _nonnegative(self.event_count, "event_count")
        )
        object.__setattr__(self, "digest", _text(self.digest, "digest"))
        object.__setattr__(
            self, "recorded_at", str(self.recorded_at or "").strip() or _utc_iso()
        )
        if not isinstance(self.body, Mapping):
            raise DatabaseEventLogError("integrity checkpoint body must be object")
        object.__setattr__(self, "body", MappingProxyType(dict(self.body)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "stream_id": self.stream_id,
            "global_sequence": self.global_sequence,
            "event_count": self.event_count,
            "digest": self.digest,
            "recorded_at": self.recorded_at,
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class DomainEventRecord:
    """One durable domain-event row projected as a logical event object."""

    event_id: str
    stream_id: str
    sequence: int
    global_sequence: int
    event_type: str
    task_cid: str
    attempt_id: str
    session_id: str
    recorded_at: str
    body: Mapping[str, Any]
    snapshot_id: str = ""
    previous_event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "position": self.sequence,
            "global_sequence": self.global_sequence,
            "type": self.event_type,
            "event_type": self.event_type,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "recorded_at": self.recorded_at,
            "timestamp": self.recorded_at,
            "snapshot_id": self.snapshot_id,
            "previous_event_id": self.previous_event_id,
            "body": dict(self.body),
        }
        return payload


# ---------------------------------------------------------------------------
# Database event log
# ---------------------------------------------------------------------------


class DatabaseEventLog:
    """Authoritative DuckDB event, audit, log, metric, and cursor store.

    Interface: ``DatabaseEventLog@1``
    """

    INTERFACE: Final[str] = DATABASE_EVENT_LOG_INTERFACE
    SCHEMA: Final[str] = DATABASE_EVENT_LOG_SCHEMA
    VERSION: Final[int] = DATABASE_EVENT_LOG_VERSION

    def __init__(
        self,
        database_path: Path | str,
        *,
        install_schema: bool = True,
        snapshot_id: str | None = None,
        owner_id: str = "database-event-log",
    ) -> None:
        if not duckdb_available():
            raise DatabaseEventLogUnavailableError(
                "DuckDB is required for DatabaseEventLog; install the optional "
                "duckdb dependency"
            )
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_id = _text(owner_id, "owner_id")
        self._closed = False
        self._lock = threading.RLock()
        if install_schema:
            install_control_plane_schema(
                self.path,
                application_version="0.0.45",
                tool_version="database-event-log",
                owner_id=self._owner_id,
            )
        self._snapshot_id = (
            _text(snapshot_id, "snapshot_id")
            if snapshot_id
            else _snapshot_for_path(self.path)
        )
        with open_duckdb_connection(self.path) as connection:
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            connection.execute(
                """
                INSERT OR REPLACE INTO event_log_meta(key, value)
                VALUES ('snapshot_id', ?)
                """,
                [self._snapshot_id],
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO event_log_meta(key, value)
                VALUES ('interface', ?)
                """,
                [self.INTERFACE],
            )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "DatabaseEventLog":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise DatabaseEventLogNotOpenError("database event log is closed")

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    # -- stream heads --------------------------------------------------------

    def stream_head(self, stream_id: str) -> StreamHead:
        self._require_open()
        stream = _text(stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.path) as connection:
            return self._load_stream_head(connection, stream)

    def _load_stream_head(self, connection: Any, stream_id: str) -> StreamHead:
        cursor = connection.execute(
            """
            SELECT stream_id, snapshot_id, earliest_sequence, latest_sequence,
                   last_event_id, last_global_sequence
            FROM event_stream_heads WHERE stream_id = ?
            """,
            [stream_id],
        )
        row = cursor.fetchone()
        if row is None:
            return StreamHead(
                stream_id=stream_id,
                snapshot_id=self._snapshot_id,
                earliest_sequence=0,
                latest_sequence=0,
                last_event_id="",
                last_global_sequence=0,
            )
        data = _row_mapping(row)
        return StreamHead(
            stream_id=str(data["stream_id"]),
            snapshot_id=str(data["snapshot_id"]),
            earliest_sequence=int(data["earliest_sequence"]),
            latest_sequence=int(data["latest_sequence"]),
            last_event_id=str(data["last_event_id"] or ""),
            last_global_sequence=int(data["last_global_sequence"] or 0),
        )

    def _upsert_stream_head(
        self,
        connection: Any,
        head: StreamHead,
        *,
        recorded_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO event_stream_heads (
                stream_id, snapshot_id, earliest_sequence, latest_sequence,
                last_event_id, last_global_sequence, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                head.stream_id,
                head.snapshot_id,
                head.earliest_sequence,
                head.latest_sequence,
                head.last_event_id,
                head.last_global_sequence,
                recorded_at,
            ],
        )

    def _next_global_sequence(self, connection: Any) -> int:
        cursor = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) AS watermark FROM domain_events"
        )
        row = cursor.fetchone()
        if row is None:
            return 1
        data = _row_mapping(row)
        return int(data.get("watermark") or 0) + 1

    # -- append --------------------------------------------------------------

    def append_event(
        self,
        stream_id: str,
        event_type: str,
        body: Mapping[str, Any] | None = None,
        *,
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
        event_id: str | None = None,
        recorded_at: str | None = None,
        redact: bool = True,
    ) -> dict[str, Any]:
        """Append one typed domain event and return the durable projection."""

        return self.append_events(
            [
                {
                    "stream_id": stream_id,
                    "event_type": event_type,
                    "body": dict(body or {}),
                    "task_cid": task_cid,
                    "attempt_id": attempt_id,
                    "session_id": session_id,
                    "event_id": event_id,
                    "recorded_at": recorded_at,
                }
            ],
            redact=redact,
        )[0]

    def append_events(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        redact: bool = True,
    ) -> list[dict[str, Any]]:
        """Append one or more events atomically; all succeed or none do."""

        self._require_open()
        if not events:
            return []
        prepared: list[dict[str, Any]] = []
        for raw in events:
            if not isinstance(raw, Mapping):
                raise DatabaseEventLogError("event entries must be objects")
            stream_id = _text(raw.get("stream_id"), "stream_id")
            event_type = _text(raw.get("event_type", raw.get("type")), "event_type")
            body_raw = raw.get("body") or {}
            if not isinstance(body_raw, Mapping):
                raise DatabaseEventLogError("event body must be an object")
            body = redact_value(body_raw) if redact else dict(body_raw)
            prepared.append(
                {
                    "stream_id": stream_id,
                    "event_type": event_type,
                    "body": body,
                    "task_cid": _text(
                        raw.get("task_cid"), "task_cid", required=False
                    ),
                    "attempt_id": _text(
                        raw.get("attempt_id"), "attempt_id", required=False
                    ),
                    "session_id": _text(
                        raw.get("session_id"), "session_id", required=False
                    ),
                    "event_id": (
                        _text(raw.get("event_id"), "event_id")
                        if raw.get("event_id")
                        else None
                    ),
                    "recorded_at": str(raw.get("recorded_at") or "").strip()
                    or _utc_iso(),
                }
            )

        with self._lock, open_duckdb_connection(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                results = [
                    self._append_one(connection, item) for item in prepared
                ]
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return results

    def append_with_projection(
        self,
        stream_id: str,
        event_type: str,
        body: Mapping[str, Any],
        *,
        projection_key: str,
        projection_value: Mapping[str, Any],
        task_cid: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Atomically append an event and store a named projection value.

        The projection is persisted in ``event_log_meta`` under a namespaced
        key so event and projection either both land or neither does.
        """

        self._require_open()
        stream = _text(stream_id, "stream_id")
        key = _text(projection_key, "projection_key")
        if not isinstance(projection_value, Mapping):
            raise DatabaseEventLogError("projection_value must be an object")
        namespaced = f"projection:{stream}:{key}"
        with self._lock, open_duckdb_connection(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                event = self._append_one(
                    connection,
                    {
                        "stream_id": stream,
                        "event_type": _text(event_type, "event_type"),
                        "body": redact_value(body),
                        "task_cid": _text(task_cid, "task_cid", required=False),
                        "attempt_id": "",
                        "session_id": _text(
                            session_id, "session_id", required=False
                        ),
                        "event_id": None,
                        "recorded_at": _utc_iso(),
                    },
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO event_log_meta(key, value)
                    VALUES (?, ?)
                    """,
                    [
                        namespaced,
                        _canonical_json(
                            {
                                "event_id": event["event_id"],
                                "sequence": event["sequence"],
                                "value": dict(projection_value),
                            }
                        ),
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return event

    def get_projection(
        self, stream_id: str, projection_key: str
    ) -> Mapping[str, Any] | None:
        self._require_open()
        namespaced = (
            f"projection:{_text(stream_id, 'stream_id')}:"
            f"{_text(projection_key, 'projection_key')}"
        )
        with self._lock, open_duckdb_connection(self.path) as connection:
            cursor = connection.execute(
                "SELECT value FROM event_log_meta WHERE key = ?",
                [namespaced],
            )
            row = cursor.fetchone()
            if row is None:
                return None
            raw = _row_mapping(row)["value"]
            return json.loads(str(raw))

    def _append_one(
        self, connection: Any, item: Mapping[str, Any]
    ) -> dict[str, Any]:
        stream_id = str(item["stream_id"])
        head = self._load_stream_head(connection, stream_id)
        if head.snapshot_id != self._snapshot_id:
            raise DatabaseEventLogIntegrityError(
                "stream head snapshot does not match this event log"
            )
        body = dict(item["body"])
        sequence = head.latest_sequence + 1
        global_sequence = self._next_global_sequence(connection)
        previous_event_id = head.last_event_id
        recorded_at = str(item["recorded_at"])
        identity_payload = {
            "stream_id": stream_id,
            "snapshot_id": self._snapshot_id,
            "sequence": sequence,
            "event_type": item["event_type"],
            "task_cid": item["task_cid"],
            "attempt_id": item["attempt_id"],
            "session_id": item["session_id"],
            "previous_event_id": previous_event_id,
            "body": body,
        }
        computed_event_id = _content_id(identity_payload)
        supplied = item.get("event_id")
        event_id = str(supplied) if supplied else computed_event_id
        existing = connection.execute(
            "SELECT event_id, body_json, sequence, stream_id, event_type "
            "FROM domain_events WHERE event_id = ?",
            [event_id],
        ).fetchone()
        if existing is not None:
            existing_row = _row_mapping(existing)
            existing_body = json.loads(str(existing_row["body_json"]))
            nested = existing_body.get("body")
            same_payload = (
                str(existing_row["stream_id"]) == stream_id
                and str(existing_row["event_type"]) == item["event_type"]
                and (
                    nested == body
                    if isinstance(nested, Mapping)
                    else existing_body == body
                )
            )
            if same_payload:
                # Exact duplicate identity already durable: coalesce.
                return self._project_existing(connection, event_id)
            raise DatabaseEventLogDuplicateError(
                f"event_id {event_id!r} collides with a different payload"
            )

        body_json = _canonical_json(
            {
                "schema": DATABASE_EVENT_LOG_SCHEMA,
                "event_type": item["event_type"],
                "snapshot_id": self._snapshot_id,
                "previous_event_id": previous_event_id,
                "body": body,
            }
        )
        try:
            connection.execute(
                """
                INSERT INTO domain_events (
                    event_id, stream_id, sequence, global_sequence, event_type,
                    task_cid, attempt_id, session_id, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    event_id,
                    stream_id,
                    sequence,
                    global_sequence,
                    item["event_type"],
                    item["task_cid"],
                    item["attempt_id"],
                    item["session_id"],
                    recorded_at,
                    body_json,
                ],
            )
        except Exception as exc:
            message = str(exc).casefold()
            if "unique" in message or "constraint" in message or "duplicate" in message:
                # Concurrent exact race: re-read and coalesce or fail closed.
                existing = connection.execute(
                    "SELECT event_id, body_json FROM domain_events WHERE event_id = ?",
                    [event_id],
                ).fetchone()
                if existing is not None:
                    return self._project_existing(connection, event_id)
                raise DatabaseEventLogDuplicateError(
                    "event identity or stream sequence collision"
                ) from exc
            raise

        new_head = StreamHead(
            stream_id=stream_id,
            snapshot_id=self._snapshot_id,
            earliest_sequence=(
                sequence
                if head.latest_sequence == 0 or head.earliest_sequence == 0
                else head.earliest_sequence
            ),
            latest_sequence=sequence,
            last_event_id=event_id,
            last_global_sequence=global_sequence,
        )
        self._upsert_stream_head(connection, new_head, recorded_at=recorded_at)
        record = DomainEventRecord(
            event_id=event_id,
            stream_id=stream_id,
            sequence=sequence,
            global_sequence=global_sequence,
            event_type=str(item["event_type"]),
            task_cid=str(item["task_cid"]),
            attempt_id=str(item["attempt_id"]),
            session_id=str(item["session_id"]),
            recorded_at=recorded_at,
            body=MappingProxyType(body),
            snapshot_id=self._snapshot_id,
            previous_event_id=previous_event_id,
        )
        return record.to_dict()

    def _project_existing(self, connection: Any, event_id: str) -> dict[str, Any]:
        cursor = connection.execute(
            """
            SELECT event_id, stream_id, sequence, global_sequence, event_type,
                   task_cid, attempt_id, session_id, recorded_at, body_json
            FROM domain_events WHERE event_id = ?
            """,
            [event_id],
        )
        row = cursor.fetchone()
        if row is None:
            raise DatabaseEventLogIntegrityError(
                f"coalesced event {event_id!r} is missing"
            )
        return self._row_to_event(_row_mapping(row)).to_dict()

    def _row_to_event(self, data: Mapping[str, Any]) -> DomainEventRecord:
        body_raw = json.loads(str(data["body_json"]))
        if not isinstance(body_raw, Mapping):
            raise DatabaseEventLogIntegrityError("event body_json is malformed")
        nested = body_raw.get("body")
        body = nested if isinstance(nested, Mapping) else body_raw
        return DomainEventRecord(
            event_id=str(data["event_id"]),
            stream_id=str(data["stream_id"]),
            sequence=int(data["sequence"]),
            global_sequence=int(data["global_sequence"]),
            event_type=str(data["event_type"]),
            task_cid=str(data.get("task_cid") or ""),
            attempt_id=str(data.get("attempt_id") or ""),
            session_id=str(data.get("session_id") or ""),
            recorded_at=str(data["recorded_at"]),
            body=MappingProxyType(dict(body)),
            snapshot_id=str(
                body_raw.get("snapshot_id") or self._snapshot_id
            ),
            previous_event_id=str(body_raw.get("previous_event_id") or ""),
        )

    # -- cursors and polling -------------------------------------------------

    def initial_cursor(self, stream_id: str) -> EventCursor:
        self._require_open()
        stream = _text(stream_id, "stream_id")
        return EventCursor.initial(stream, snapshot_id=self._snapshot_id)

    def latest_cursor(self, stream_id: str) -> EventCursor:
        head = self.stream_head(stream_id)
        if head.latest_sequence == 0:
            return EventCursor.initial(head.stream_id, snapshot_id=head.snapshot_id)
        return EventCursor(
            stream_id=head.stream_id,
            position=head.latest_sequence,
            last_event_id=head.last_event_id,
            snapshot_id=head.snapshot_id,
        )

    def poll(
        self,
        cursor: EventCursor | Mapping[str, Any] | str,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> EventPage:
        """Bounded poll strictly after ``cursor`` with gap/duplicate protection."""

        self._require_open()
        selected_limit = _positive_limit(limit)
        selected = self._coerce_cursor(cursor)
        with self._lock, open_duckdb_connection(self.path) as connection:
            head = self._load_stream_head(connection, selected.stream_id)
            earliest = (
                head.earliest_sequence
                if head.latest_sequence and head.earliest_sequence
                else 0
            )
            selected.assert_replayable(
                stream_id=head.stream_id,
                earliest_position=earliest,
                latest_position=head.latest_sequence,
                snapshot_id=head.snapshot_id,
            )
            # Fetch one extra so we can detect has_more and verify anchors.
            fetch_limit = selected_limit + 2
            after = selected.position
            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence, event_type,
                       task_cid, attempt_id, session_id, recorded_at, body_json
                FROM domain_events
                WHERE stream_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                [selected.stream_id, after, fetch_limit],
            ).fetchall()
            events = [
                self._row_to_event(_row_mapping(row)).to_dict() for row in rows
            ]
            # Include the cursor anchor event when present so replay_event_page
            # can verify last_event_id. A cursor exactly one behind the retained
            # floor is still valid after retention even though its anchor row
            # has been pruned.
            if selected.position > 0:
                anchor = connection.execute(
                    """
                    SELECT event_id, stream_id, sequence, global_sequence, event_type,
                           task_cid, attempt_id, session_id, recorded_at, body_json
                    FROM domain_events
                    WHERE stream_id = ? AND sequence = ?
                    LIMIT 1
                    """,
                    [selected.stream_id, selected.position],
                ).fetchone()
                if anchor is not None:
                    anchor_event = self._row_to_event(_row_mapping(anchor)).to_dict()
                    events = [anchor_event, *events]
                elif selected.position < max(0, head.earliest_sequence - 1):
                    raise CursorReplayError(
                        "event cursor predates the retained replay window"
                    )
                elif selected.position == max(0, head.earliest_sequence - 1):
                    # Retention boundary: next event must be earliest_sequence.
                    pass
                else:
                    raise CursorReplayError(
                        "event cursor anchor is missing from the stream"
                    )
        # Coalesce exact duplicate sequences (should not occur under unique index).
        coalesced: dict[int, dict[str, Any]] = {}
        for event in events:
            sequence = int(event["sequence"])
            known = coalesced.get(sequence)
            if known is not None:
                if known["event_id"] != event["event_id"]:
                    raise DatabaseEventLogIntegrityError(
                        f"sequence {sequence} has conflicting identities"
                    )
                continue
            coalesced[sequence] = event
        ordered = [coalesced[key] for key in sorted(coalesced)]
        return replay_event_page(
            ordered,
            selected,
            limit=selected_limit,
            stream_id=selected.stream_id,
            snapshot_id=self._snapshot_id,
        )

    read_page = poll

    def _coerce_cursor(
        self, cursor: EventCursor | Mapping[str, Any] | str
    ) -> EventCursor:
        if isinstance(cursor, EventCursor):
            return cursor
        if isinstance(cursor, str):
            return EventCursor.from_token(cursor)
        if isinstance(cursor, Mapping):
            return EventCursor.from_dict(cursor)
        raise EventCursorError(
            "cursor must be an EventCursor, canonical cursor record, or token"
        )

    # -- consumer checkpoints ------------------------------------------------

    def save_consumer_checkpoint(
        self,
        consumer_id: str,
        cursor: EventCursor | Mapping[str, Any] | str,
    ) -> ConsumerCheckpoint:
        """Persist a consumer cursor after verifying it is still replayable."""

        self._require_open()
        selected = self._coerce_cursor(cursor)
        consumer = _text(consumer_id, "consumer_id")
        head = self.stream_head(selected.stream_id)
        earliest = (
            head.earliest_sequence
            if head.latest_sequence and head.earliest_sequence
            else 0
        )
        selected.assert_replayable(
            stream_id=head.stream_id,
            earliest_position=earliest,
            latest_position=head.latest_sequence,
            snapshot_id=head.snapshot_id,
        )
        if selected.position > 0:
            # Anchor event must still exist under retention, except exactly at
            # the retained floor boundary (earliest - 1) where the prior row
            # may have been pruned intentionally.
            with self._lock, open_duckdb_connection(self.path) as connection:
                row = connection.execute(
                    """
                    SELECT event_id FROM domain_events
                    WHERE stream_id = ? AND sequence = ?
                    """,
                    [selected.stream_id, selected.position],
                ).fetchone()
                if row is None:
                    boundary = max(0, head.earliest_sequence - 1)
                    if selected.position < boundary:
                        raise CursorReplayError(
                            "consumer checkpoint predates the retained window"
                        )
                    if selected.position > boundary:
                        raise CursorReplayError(
                            "consumer checkpoint anchor is not retained"
                        )
                elif str(_row_mapping(row)["event_id"]) != selected.last_event_id:
                    raise CursorReplayError(
                        "consumer checkpoint anchor does not match stream"
                    )
        recorded_at = _utc_iso()
        checkpoint = ConsumerCheckpoint(
            consumer_id=consumer, cursor=selected, recorded_at=recorded_at
        )
        with self._lock, open_duckdb_connection(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO event_consumer_checkpoints (
                    consumer_id, stream_id, position, last_event_id,
                    snapshot_id, cursor_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    checkpoint.consumer_id,
                    selected.stream_id,
                    selected.position,
                    selected.last_event_id,
                    selected.snapshot_id,
                    _canonical_json(selected.to_record()),
                    recorded_at,
                ],
            )
        return checkpoint

    def load_consumer_checkpoint(
        self, consumer_id: str, stream_id: str
    ) -> ConsumerCheckpoint | None:
        self._require_open()
        consumer = _text(consumer_id, "consumer_id")
        stream = _text(stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.path) as connection:
            row = connection.execute(
                """
                SELECT consumer_id, cursor_json, recorded_at
                FROM event_consumer_checkpoints
                WHERE consumer_id = ? AND stream_id = ?
                """,
                [consumer, stream],
            ).fetchone()
            if row is None:
                return None
            data = _row_mapping(row)
            cursor = EventCursor.from_dict(json.loads(str(data["cursor_json"])))
            return ConsumerCheckpoint(
                consumer_id=str(data["consumer_id"]),
                cursor=cursor,
                recorded_at=str(data["recorded_at"]),
            )

    # -- structured logs / metrics / audit -----------------------------------

    def append_structured_log(
        self,
        *,
        severity: LogSeverity | str,
        component: str,
        message: str,
        body: Mapping[str, Any] | None = None,
        trace_id: str = "",
        span_id: str = "",
        task_cid: str = "",
        attempt_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any] | None:
        """Append one structured log row with bounded recursive re-entry."""

        self._require_open()
        depth = _enter_log()
        try:
            if depth > MAX_RECURSION_DEPTH:
                # Bound recursive logging: drop nested emissions silently.
                return None
            try:
                native_severity = (
                    severity
                    if isinstance(severity, LogSeverity)
                    else LogSeverity(str(severity))
                )
            except ValueError as exc:
                raise DatabaseEventLogError(
                    f"unknown log severity {severity!r}"
                ) from exc
            message_text = str(message or "")
            if len(message_text.encode("utf-8")) > MAX_LOG_MESSAGE_BYTES:
                raise DatabaseEventLogPayloadError(
                    f"log message exceeds the {MAX_LOG_MESSAGE_BYTES}-byte bound"
                )
            recorded_at = _utc_iso()
            log_id = f"log:{uuid.uuid4()}"
            body_value = redact_value(body or {})
            body_json = _canonical_json(
                {
                    "schema": STRUCTURED_LOG_SCHEMA,
                    "body": body_value,
                },
                maximum=MAX_EVENT_BODY_BYTES,
            )
            with self._lock, open_duckdb_connection(self.path) as connection:
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
                        native_severity.value,
                        _text(component, "component"),
                        _text(trace_id, "trace_id", required=False),
                        _text(span_id, "span_id", required=False),
                        _text(task_cid, "task_cid", required=False),
                        _text(attempt_id, "attempt_id", required=False),
                        _text(session_id, "session_id", required=False),
                        recorded_at,
                        message_text,
                        body_json,
                    ],
                )
            return {
                "schema": STRUCTURED_LOG_SCHEMA,
                "log_id": log_id,
                "severity": native_severity.value,
                "component": _text(component, "component"),
                "message": message_text,
                "recorded_at": recorded_at,
                "body": body_value,
            }
        finally:
            _exit_log()

    def register_metric(
        self,
        metric_name: str,
        *,
        unit: str = "count",
        description: str = "",
    ) -> str:
        self._require_open()
        name = _text(metric_name, "metric_name")
        metric_id = f"metric:{_content_id({'name': name})[7:23]}"
        with self._lock, open_duckdb_connection(self.path) as connection:
            existing = connection.execute(
                "SELECT metric_id FROM metrics WHERE metric_name = ?",
                [name],
            ).fetchone()
            if existing is not None:
                return str(_row_mapping(existing)["metric_id"])
            connection.execute(
                """
                INSERT INTO metrics (
                    metric_id, metric_name, unit, description, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    metric_id,
                    name,
                    _text(unit, "unit"),
                    str(description or ""),
                    _utc_iso(),
                ],
            )
        return metric_id

    def append_metric_sample(
        self,
        metric_name: str,
        value_milli: int,
        *,
        labels: Mapping[str, Any] | None = None,
        stratum: str = "",
    ) -> dict[str, Any]:
        self._require_open()
        if isinstance(value_milli, bool) or not isinstance(value_milli, int):
            raise DatabaseEventLogError("value_milli must be an integer")
        metric_id = self.register_metric(metric_name)
        sample_id = f"sample:{uuid.uuid4()}"
        observed_at = _utc_iso()
        labels_value = redact_value(labels or {})
        with self._lock, open_duckdb_connection(self.path) as connection:
            connection.execute(
                """
                INSERT INTO metric_samples (
                    sample_id, metric_id, observed_at, value_milli,
                    labels_json, stratum
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    sample_id,
                    metric_id,
                    observed_at,
                    value_milli,
                    _canonical_json(labels_value, maximum=16_384),
                    _text(stratum, "stratum", required=False),
                ],
            )
        return {
            "schema": METRIC_SAMPLE_SCHEMA,
            "sample_id": sample_id,
            "metric_id": metric_id,
            "metric_name": _text(metric_name, "metric_name"),
            "value_milli": value_milli,
            "observed_at": observed_at,
            "labels": labels_value,
            "stratum": stratum,
        }

    def append_audit(
        self,
        action: str,
        *,
        actor: str,
        resource: str,
        outcome: str = "success",
        details: Mapping[str, Any] | None = None,
        stream_id: str = "audit",
        session_id: str = "",
        task_cid: str = "",
    ) -> dict[str, Any]:
        """Record an explicit application audit event (not Quack diagnostics)."""

        body = {
            "schema": AUDIT_RECORD_SCHEMA,
            "action": _text(action, "action"),
            "actor": _text(actor, "actor"),
            "resource": _text(resource, "resource"),
            "outcome": _text(outcome, "outcome"),
            "details": redact_value(details or {}),
        }
        return self.append_event(
            stream_id,
            AUDIT_EVENT_TYPE,
            body,
            task_cid=task_cid,
            session_id=session_id,
            redact=True,
        )

    # -- integrity / retention / export --------------------------------------

    def _stream_digest(
        self, connection: Any, stream_id: str = ""
    ) -> tuple[str, int, int, list[dict[str, Any]]]:
        if stream_id:
            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence
                FROM domain_events
                WHERE stream_id = ?
                ORDER BY sequence ASC
                """,
                [stream_id],
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence
                FROM domain_events
                ORDER BY global_sequence ASC
                """
            ).fetchall()
        population = [_row_mapping(row) for row in rows]
        digest_payload = [
            {
                "event_id": str(item["event_id"]),
                "stream_id": str(item["stream_id"]),
                "sequence": int(item["sequence"]),
                "global_sequence": int(item["global_sequence"]),
            }
            for item in population
        ]
        digest = _content_id({"events": digest_payload})
        global_sequence = (
            int(population[-1]["global_sequence"]) if population else 0
        )
        return digest, global_sequence, len(population), population

    def create_integrity_checkpoint(
        self, stream_id: str = ""
    ) -> IntegrityCheckpoint:
        self._require_open()
        stream = _text(stream_id, "stream_id", required=False)
        with self._lock, open_duckdb_connection(self.path) as connection:
            digest, global_sequence, event_count, _population = self._stream_digest(
                connection, stream
            )
            checkpoint_id = f"integrity:{digest[7:39]}"
            recorded_at = _utc_iso()
            body = {
                "event_count": event_count,
                "stream_id": stream,
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO event_integrity_checkpoints (
                    checkpoint_id, stream_id, global_sequence, event_count,
                    digest, recorded_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    checkpoint_id,
                    stream,
                    global_sequence,
                    event_count,
                    digest,
                    recorded_at,
                    _canonical_json(body),
                ],
            )
        return IntegrityCheckpoint(
            checkpoint_id=checkpoint_id,
            stream_id=stream,
            global_sequence=global_sequence,
            event_count=event_count,
            digest=digest,
            recorded_at=recorded_at,
            body=body,
        )

    def verify_integrity_checkpoint(
        self, checkpoint: IntegrityCheckpoint | Mapping[str, Any] | str
    ) -> bool:
        self._require_open()
        if isinstance(checkpoint, IntegrityCheckpoint):
            selected = checkpoint
        elif isinstance(checkpoint, Mapping):
            selected = IntegrityCheckpoint(
                checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
                stream_id=str(checkpoint.get("stream_id") or ""),
                global_sequence=int(checkpoint.get("global_sequence") or 0),
                event_count=int(checkpoint.get("event_count") or 0),
                digest=str(checkpoint.get("digest") or ""),
                recorded_at=str(checkpoint.get("recorded_at") or ""),
                body=checkpoint.get("body") or {},
            )
        else:
            checkpoint_id = _text(checkpoint, "checkpoint_id")
            with self._lock, open_duckdb_connection(self.path) as connection:
                row = connection.execute(
                    """
                    SELECT checkpoint_id, stream_id, global_sequence, event_count,
                           digest, recorded_at, body_json
                    FROM event_integrity_checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    [checkpoint_id],
                ).fetchone()
                if row is None:
                    raise DatabaseEventLogIntegrityError(
                        f"integrity checkpoint {checkpoint_id!r} is missing"
                    )
                data = _row_mapping(row)
                selected = IntegrityCheckpoint(
                    checkpoint_id=str(data["checkpoint_id"]),
                    stream_id=str(data["stream_id"] or ""),
                    global_sequence=int(data["global_sequence"]),
                    event_count=int(data["event_count"]),
                    digest=str(data["digest"]),
                    recorded_at=str(data["recorded_at"]),
                    body=json.loads(str(data["body_json"])),
                )
        with self._lock, open_duckdb_connection(self.path) as connection:
            digest, _global_sequence, event_count, _population = self._stream_digest(
                connection, selected.stream_id
            )
        if digest != selected.digest:
            raise DatabaseEventLogIntegrityError(
                "integrity checkpoint digest mismatch"
            )
        if event_count != selected.event_count:
            raise DatabaseEventLogIntegrityError(
                "integrity checkpoint event count mismatch"
            )
        return True

    def apply_retention(
        self,
        stream_id: str,
        *,
        retain_after_sequence: int,
    ) -> StreamHead:
        """Drop events at or before the retention floor; update stream head.

        Consumer cursors at or before the floor become expired on next poll.
        """

        self._require_open()
        stream = _text(stream_id, "stream_id")
        floor = _nonnegative(retain_after_sequence, "retain_after_sequence")
        with self._lock, open_duckdb_connection(self.path) as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                head = self._load_stream_head(connection, stream)
                if head.latest_sequence == 0:
                    connection.execute("COMMIT")
                    return head
                if floor < head.earliest_sequence:
                    connection.execute("COMMIT")
                    return head
                if floor >= head.latest_sequence:
                    # Always retain the tip event.
                    floor = max(head.earliest_sequence - 1, head.latest_sequence - 1)
                connection.execute(
                    """
                    DELETE FROM domain_events
                    WHERE stream_id = ? AND sequence <= ?
                    """,
                    [stream, floor],
                )
                new_earliest = floor + 1
                if new_earliest > head.latest_sequence:
                    new_earliest = head.latest_sequence
                new_head = StreamHead(
                    stream_id=stream,
                    snapshot_id=head.snapshot_id,
                    earliest_sequence=new_earliest,
                    latest_sequence=head.latest_sequence,
                    last_event_id=head.last_event_id,
                    last_global_sequence=head.last_global_sequence,
                )
                self._upsert_stream_head(
                    connection, new_head, recorded_at=_utc_iso()
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return new_head

    def export_jsonl(
        self,
        destination: Path | str,
        *,
        stream_id: str | None = None,
        after_sequence: int = 0,
    ) -> dict[str, Any]:
        """Render a read-only JSONL export. Deleting the file has no authority effect."""

        self._require_open()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        after = _nonnegative(after_sequence, "after_sequence")
        with self._lock, open_duckdb_connection(self.path) as connection:
            if stream_id:
                stream = _text(stream_id, "stream_id")
                rows = connection.execute(
                    """
                    SELECT event_id, stream_id, sequence, global_sequence, event_type,
                           task_cid, attempt_id, session_id, recorded_at, body_json
                    FROM domain_events
                    WHERE stream_id = ? AND sequence > ?
                    ORDER BY sequence ASC
                    """,
                    [stream, after],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT event_id, stream_id, sequence, global_sequence, event_type,
                           task_cid, attempt_id, session_id, recorded_at, body_json
                    FROM domain_events
                    WHERE global_sequence > ?
                    ORDER BY global_sequence ASC
                    """,
                    [after],
                ).fetchall()
            events = [
                self._row_to_event(_row_mapping(row)).to_dict() for row in rows
            ]
        lines = [
            json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for event in events
        ]
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        path.write_bytes(payload)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        return {
            "path": str(path),
            "event_count": len(events),
            "sha256": digest,
            "authoritative": False,
            "snapshot_id": self._snapshot_id,
        }

    def list_event_ids(self, stream_id: str) -> tuple[str, ...]:
        self._require_open()
        stream = _text(stream_id, "stream_id")
        with self._lock, open_duckdb_connection(self.path) as connection:
            rows = connection.execute(
                """
                SELECT event_id FROM domain_events
                WHERE stream_id = ?
                ORDER BY sequence ASC
                """,
                [stream],
            ).fetchall()
        return tuple(str(_row_mapping(row)["event_id"]) for row in rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "version": self.VERSION,
            "snapshot_id": self._snapshot_id,
            "path": str(self.path),
        }


def open_database_event_log(
    database_path: Path | str,
    *,
    install_schema: bool = True,
    snapshot_id: str | None = None,
) -> DatabaseEventLog:
    """Open a DatabaseEventLog, installing the control-plane schema by default."""

    return DatabaseEventLog(
        database_path,
        install_schema=install_schema,
        snapshot_id=snapshot_id,
    )


__all__ = (
    "AUDIT_EVENT_TYPE",
    "AUDIT_RECORD_SCHEMA",
    "CONSUMER_CHECKPOINT_INTERFACE",
    "CONSUMER_CHECKPOINT_SCHEMA",
    "ConsumerCheckpoint",
    "DATABASE_EVENT_LOG_INTERFACE",
    "DATABASE_EVENT_LOG_SCHEMA",
    "DatabaseEventLog",
    "DatabaseEventLogDuplicateError",
    "DatabaseEventLogError",
    "DatabaseEventLogIntegrityError",
    "DatabaseEventLogNotOpenError",
    "DatabaseEventLogPayloadError",
    "DatabaseEventLogUnavailableError",
    "DomainEventRecord",
    "EVENT_CURSOR_INTERFACE",
    "EventCursor",
    "EventPage",
    "INTEGRITY_CHECKPOINT_SCHEMA",
    "IntegrityCheckpoint",
    "LogSeverity",
    "STREAM_HEAD_SCHEMA",
    "StreamHead",
    "open_database_event_log",
    "redact_value",
)
