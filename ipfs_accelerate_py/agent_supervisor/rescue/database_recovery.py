"""Transactional recovery, reconciliation, quarantine, and rescue decisions.

DQP-019 / RecoveryAction@1
==========================

:class:`DatabaseRecovery` is the durable authority for recovery actions that
share task, attempt, worktree, fence, and event coordinates with the merge and
validation pipeline. Actions are idempotent by key, queryable after commit,
and never treat a JSON receipt file as settlement authority.

Authority rules (fail-closed)
-----------------------------
* Every recovery action has a closed action kind and terminal status.
* Identical idempotency keys replay the durable result without side effects.
* Retry budgets are enforced; exhaustion produces quarantine, not success.
* Reconciliation replay is bounded and records exact subject coordinates.
* Rescue decisions are queryable events, not process-local memory.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_RECOVERY_INTERFACE: Final[str] = "DatabaseRecovery@1"
RECOVERY_ACTION_INTERFACE: Final[str] = "RecoveryAction@1"
RECONCILIATION_RECEIPT_INTERFACE: Final[str] = "ReconciliationReceipt@1"

DATABASE_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-recovery@1"
)
RECOVERY_ACTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/recovery-action@1"
)
RECONCILIATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/reconciliation-receipt@1"
)
RECOVERY_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/recovery-event@1"
)

DEFAULT_MAX_RETRIES: Final[int] = 3
MAX_PAYLOAD_BYTES: Final[int] = 262_144

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS recovery_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS recovery_subjects (
    subject_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    entry_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    retry_count BIGINT NOT NULL DEFAULT 0,
    max_retries BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    registered_at_ms BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS recovery_subjects_task_idx
    ON recovery_subjects(task_cid, status);
CREATE INDEX IF NOT EXISTS recovery_subjects_entry_idx
    ON recovery_subjects(entry_id, status);

CREATE TABLE IF NOT EXISTS recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    worktree_id VARCHAR NOT NULL DEFAULT '',
    entry_id VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    action_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    applied_at_ms BIGINT NOT NULL DEFAULT 0,
    result_digest VARCHAR NOT NULL DEFAULT '',
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS recovery_actions_idempotency_uidx
    ON recovery_actions(idempotency_key);
CREATE INDEX IF NOT EXISTS recovery_actions_subject_idx
    ON recovery_actions(subject_id, decided_at_ms);
CREATE INDEX IF NOT EXISTS recovery_actions_task_idx
    ON recovery_actions(task_cid, decided_at_ms);
CREATE INDEX IF NOT EXISTS recovery_actions_kind_idx
    ON recovery_actions(action_kind, status);

CREATE TABLE IF NOT EXISTS reconciliation_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    action_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    entry_id VARCHAR NOT NULL DEFAULT '',
    replayed_event_count BIGINT NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS reconciliation_receipts_subject_idx
    ON reconciliation_receipts(subject_id, recorded_at_ms);

CREATE TABLE IF NOT EXISTS recovery_events (
    event_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    action_id VARCHAR NOT NULL DEFAULT '',
    event_type VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS recovery_events_subject_idx
    ON recovery_events(subject_id, observed_at_ms);
"""

ClockMs = Callable[[], int]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRecoveryError(RuntimeError):
    """Base fail-closed error for database recovery."""

    code = "DQP_RECOVERY_ERROR"


class DatabaseRecoveryNotOpenError(DatabaseRecoveryError):
    """Operation requires an open recovery store."""

    code = "DQP_RECOVERY_NOT_OPEN"


class DatabaseRecoveryConflictError(DatabaseRecoveryError):
    """Identity or status conflict for a recovery action."""

    code = "DQP_RECOVERY_CONFLICT"


class DatabaseRecoveryBoundsError(DatabaseRecoveryError, ValueError):
    """Payload or retry budget bound exceeded."""

    code = "DQP_RECOVERY_BOUNDS"


class DatabaseRecoveryExhaustedError(DatabaseRecoveryError):
    """Retry budget exhausted; subject is quarantined."""

    code = "DQP_RECOVERY_EXHAUSTED"


class DuckDBUnavailableError(DatabaseRecoveryError):
    """Optional DuckDB dependency is not installed."""

    code = "DQP_DUCKDB_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class SubjectKind(str, Enum):
    TASK = "task"
    ATTEMPT = "attempt"
    WORKTREE = "worktree"
    MERGE_ENTRY = "merge_entry"
    VALIDATION_RUN = "validation_run"
    SESSION = "session"


class SubjectStatus(str, Enum):
    OPEN = "open"
    RECOVERING = "recovering"
    RECONCILED = "reconciled"
    QUARANTINED = "quarantined"
    RESCUED = "rescued"
    CLOSED = "closed"


class ActionKind(str, Enum):
    REPLAY = "replay"
    RECONCILE = "reconcile"
    RETRY = "retry"
    QUARANTINE = "quarantine"
    RESCUE = "rescue"
    FENCE_STALE_CLAIM = "fence_stale_claim"
    INTERRUPT_VALIDATION = "interrupt_validation"
    INTERRUPT_MERGE = "interrupt_merge"


class ActionStatus(str, Enum):
    DECIDED = "decided"
    APPLIED = "applied"
    REPLAYED = "replayed"
    REJECTED = "rejected"
    EXHAUSTED = "exhausted"


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


def _default_clock_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _utc_iso_from_ms(epoch_ms: int) -> str:
    return (
        datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRecoveryError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRecoveryError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseRecoveryBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseRecoveryBoundsError(f"{name} must be a positive integer")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    name: str,
    max_bytes: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    raw = dict(body or {})
    encoded = _canonical_json(raw).encode("utf-8")
    if len(encoded) > max_bytes:
        raise DatabaseRecoveryBoundsError(
            f"{name} exceeds the {max_bytes}-byte bound"
        )
    return raw


def _row_mapping(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        keys = []
    if keys:
        return {str(key): row[key] for key in keys}
    try:
        return {str(index): row[index] for index in range(len(row))}  # type: ignore[arg-type]
    except Exception:
        return {}


def _row_get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
        upper = name.upper()
        if upper in mapping and mapping[upper] is not None:
            return mapping[upper]
        lower = name.lower()
        if lower in mapping and mapping[lower] is not None:
            return mapping[lower]
    wanted = {name.lower() for name in names}
    for key, value in mapping.items():
        if str(key).lower() in wanted and value is not None:
            return value
    return default


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _default_idempotency_key(
    *,
    subject_id: str,
    action_kind: str,
    reason: str,
    body: Mapping[str, Any],
) -> str:
    return _sha256_hex(
        _canonical_json(
            {
                "subject_id": subject_id,
                "action_kind": action_kind,
                "reason": reason,
                "body": dict(body),
            }
        ).encode("utf-8")
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoverySubject:
    """A durable recovery subject with retry budget."""

    subject_id: str
    subject_kind: SubjectKind
    status: SubjectStatus
    max_retries: int
    registered_at_ms: int
    updated_at_ms: int
    task_cid: str = ""
    attempt_id: str = ""
    worktree_id: str = ""
    entry_id: str = ""
    fencing_token: int = 0
    fence_epoch: int = 0
    retry_count: int = 0
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        kind = self.subject_kind
        if not isinstance(kind, SubjectKind):
            kind = SubjectKind(str(kind).strip().lower())
            object.__setattr__(self, "subject_kind", kind)
        status = self.status
        if not isinstance(status, SubjectStatus):
            status = SubjectStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "max_retries", _positive_int(int(self.max_retries), "max_retries")
        )
        object.__setattr__(
            self,
            "registered_at_ms",
            _nonneg_int(int(self.registered_at_ms), "registered_at_ms"),
        )
        object.__setattr__(
            self, "updated_at_ms", _nonneg_int(int(self.updated_at_ms), "updated_at_ms")
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self, "entry_id", _text(self.entry_id, "entry_id", required=False)
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "retry_count", _nonneg_int(int(self.retry_count), "retry_count")
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    @property
    def retries_remaining(self) -> int:
        return max(0, self.max_retries - self.retry_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind.value,
            "status": self.status.value,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "worktree_id": self.worktree_id,
            "entry_id": self.entry_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "retry_count": int(self.retry_count),
            "max_retries": int(self.max_retries),
            "retries_remaining": int(self.retries_remaining),
            "registered_at_ms": int(self.registered_at_ms),
            "updated_at_ms": int(self.updated_at_ms),
            "body": dict(self.body),
        }


@dataclass(frozen=True)
class RecoveryAction:
    """One idempotent, queryable recovery decision and application."""

    INTERFACE: ClassVar[str] = RECOVERY_ACTION_INTERFACE
    SCHEMA: ClassVar[str] = RECOVERY_ACTION_SCHEMA

    action_id: str
    subject_id: str
    subject_kind: SubjectKind
    subject_ref: str
    action_kind: ActionKind
    status: ActionStatus
    idempotency_key: str
    decided_at_ms: int
    task_cid: str = ""
    attempt_id: str = ""
    worktree_id: str = ""
    entry_id: str = ""
    fencing_token: int = 0
    fence_epoch: int = 0
    applied_at_ms: int = 0
    result_digest: str = ""
    reason: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        kind = self.subject_kind
        if not isinstance(kind, SubjectKind):
            kind = SubjectKind(str(kind).strip().lower())
            object.__setattr__(self, "subject_kind", kind)
        object.__setattr__(
            self, "subject_ref", _text(self.subject_ref, "subject_ref")
        )
        action_kind = self.action_kind
        if not isinstance(action_kind, ActionKind):
            action_kind = ActionKind(str(action_kind).strip().lower())
            object.__setattr__(self, "action_kind", action_kind)
        status = self.status
        if not isinstance(status, ActionStatus):
            status = ActionStatus(str(status).strip().lower())
            object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "idempotency_key", _text(self.idempotency_key, "idempotency_key")
        )
        object.__setattr__(
            self, "decided_at_ms", _nonneg_int(int(self.decided_at_ms), "decided_at_ms")
        )
        object.__setattr__(
            self, "task_cid", _text(self.task_cid, "task_cid", required=False)
        )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", required=False)
        )
        object.__setattr__(
            self, "entry_id", _text(self.entry_id, "entry_id", required=False)
        )
        object.__setattr__(
            self,
            "fencing_token",
            _nonneg_int(int(self.fencing_token), "fencing_token"),
        )
        object.__setattr__(
            self, "fence_epoch", _nonneg_int(int(self.fence_epoch), "fence_epoch")
        )
        object.__setattr__(
            self, "applied_at_ms", _nonneg_int(int(self.applied_at_ms), "applied_at_ms")
        )
        object.__setattr__(
            self,
            "result_digest",
            _text(self.result_digest, "result_digest", required=False),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "body",
            MappingProxyType(_bounded_mapping(dict(self.body or {}), name="body")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "action_id": self.action_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind.value,
            "subject_ref": self.subject_ref,
            "task_cid": self.task_cid,
            "attempt_id": self.attempt_id,
            "worktree_id": self.worktree_id,
            "entry_id": self.entry_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "action_kind": self.action_kind.value,
            "status": self.status.value,
            "idempotency_key": self.idempotency_key,
            "decided_at_ms": int(self.decided_at_ms),
            "applied_at_ms": int(self.applied_at_ms),
            "decided_at": _utc_iso_from_ms(self.decided_at_ms),
            "applied_at": (
                _utc_iso_from_ms(self.applied_at_ms) if self.applied_at_ms else ""
            ),
            "result_digest": self.result_digest,
            "reason": self.reason,
            "body": dict(self.body),
            "authority": "database",
            "json_receipt_authority": "none",
        }


@dataclass(frozen=True)
class ReconciliationReceipt:
    """Bounded reconciliation replay receipt."""

    INTERFACE: ClassVar[str] = RECONCILIATION_RECEIPT_INTERFACE
    SCHEMA: ClassVar[str] = RECONCILIATION_RECEIPT_SCHEMA

    receipt_id: str
    subject_id: str
    action_id: str
    status: str
    recorded_at_ms: int
    task_cid: str = ""
    entry_id: str = ""
    replayed_event_count: int = 0
    body: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "receipt_id": self.receipt_id,
            "subject_id": self.subject_id,
            "action_id": self.action_id,
            "task_cid": self.task_cid,
            "entry_id": self.entry_id,
            "replayed_event_count": int(self.replayed_event_count),
            "status": self.status,
            "recorded_at_ms": int(self.recorded_at_ms),
            "recorded_at": _utc_iso_from_ms(self.recorded_at_ms),
            "body": dict(self.body),
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseRecovery:
    """DuckDB-backed recovery, reconciliation, and rescue authority.

    Interface: ``DatabaseRecovery@1`` with projected record
    ``RecoveryAction@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_RECOVERY_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_RECOVERY_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        clock_ms: ClockMs | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRecovery; install the optional "
                "duckdb dependency"
            )
        self._path = Path(database_path)
        self._clock_ms = clock_ms or _default_clock_ms
        self._max_retries = _positive_int(int(max_retries), "max_retries")
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def open(self) -> "DatabaseRecovery":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            try:
                for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                    connection.execute(statement)
                for key, value in (
                    ("interface", DATABASE_RECOVERY_INTERFACE),
                    ("schema", DATABASE_RECOVERY_SCHEMA),
                ):
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO recovery_metadata(key, value)
                        VALUES (?, ?)
                        """,
                        [key, value],
                    )
                self._connection = connection
                self._closed = False
                self._commit_if_idle(connection)
                return self
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass
                raise

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

    def __enter__(self) -> "DatabaseRecovery":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def authority_policy(self) -> dict[str, str]:
        return {
            "semantic_authority": "database",
            "recovery_authority": "database",
            "json_receipt_authority": "none",
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
        }

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseRecoveryNotOpenError("DatabaseRecovery is not open")
        return self._connection

    def _begin(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        try:
            connection.execute("BEGIN TRANSACTION")
        except Exception:
            pass

    def _rollback_if_open(self, connection: Any) -> None:
        try:
            rollback = getattr(connection, "rollback", None)
            if callable(rollback) and getattr(connection, "in_transaction", False):
                rollback()
                return
            raw = getattr(connection, "_connection", None)
            raw_rollback = getattr(raw, "rollback", None) if raw is not None else None
            if callable(raw_rollback):
                raw_rollback()
        except Exception:
            pass

    def _commit_if_idle(self, connection: Any) -> None:
        try:
            if getattr(connection, "in_transaction", False):
                commit = getattr(connection, "commit", None)
                if callable(commit):
                    commit()
                    return
            raw = getattr(connection, "_connection", None)
            raw_commit = getattr(raw, "commit", None) if raw is not None else None
            if callable(raw_commit):
                raw_commit()
                return
            commit = getattr(connection, "commit", None)
            if callable(commit):
                commit()
        except Exception:
            pass

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    def _record_event(
        self,
        connection: Any,
        *,
        subject_id: str,
        event_type: str,
        action_id: str = "",
        body: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> None:
        now = self._now_ms() if now_ms is None else int(now_ms)
        connection.execute(
            """
            INSERT INTO recovery_events(
                event_id, subject_id, action_id, event_type, observed_at_ms, body_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                _new_id("recovery-event"),
                subject_id,
                action_id,
                event_type,
                now,
                _canonical_json(_bounded_mapping(body, name="event body")),
            ],
        )

    # -- subjects ------------------------------------------------------------

    def register_subject(
        self,
        *,
        subject_kind: SubjectKind | str,
        subject_ref: str,
        task_cid: str = "",
        attempt_id: str = "",
        worktree_id: str = "",
        entry_id: str = "",
        fencing_token: int = 0,
        fence_epoch: int = 0,
        max_retries: int | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> RecoverySubject:
        """Register or return a recovery subject with a retry budget."""

        kind = (
            subject_kind
            if isinstance(subject_kind, SubjectKind)
            else SubjectKind(str(subject_kind).strip().lower())
        )
        ref = _text(subject_ref, "subject_ref")
        subject_id = f"{kind.value}:{ref}"
        retries = (
            self._max_retries
            if max_retries is None
            else _positive_int(int(max_retries), "max_retries")
        )
        payload = _bounded_mapping(body, name="subject body")
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT * FROM recovery_subjects WHERE subject_id = ?",
                    [subject_id],
                ).fetchone()
                if existing is not None:
                    self._commit_if_idle(connection)
                    return self._subject_from_row(existing)
                connection.execute(
                    """
                    INSERT INTO recovery_subjects(
                        subject_id, subject_kind, task_cid, attempt_id, worktree_id,
                        entry_id, fencing_token, fence_epoch, retry_count, max_retries,
                        status, registered_at_ms, updated_at_ms, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                    """,
                    [
                        subject_id,
                        kind.value,
                        _text(task_cid, "task_cid", required=False),
                        _text(attempt_id, "attempt_id", required=False),
                        _text(worktree_id, "worktree_id", required=False),
                        _text(entry_id, "entry_id", required=False),
                        _nonneg_int(int(fencing_token), "fencing_token"),
                        _nonneg_int(int(fence_epoch), "fence_epoch"),
                        retries,
                        SubjectStatus.OPEN.value,
                        now,
                        now,
                        _canonical_json(payload),
                    ],
                )
                self._record_event(
                    connection,
                    subject_id=subject_id,
                    event_type="subject_registered",
                    body={"subject_kind": kind.value, "subject_ref": ref},
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM recovery_subjects WHERE subject_id = ?",
                    [subject_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._subject_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def get_subject(self, subject_id: str) -> RecoverySubject | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM recovery_subjects WHERE subject_id = ?",
                [_text(subject_id, "subject_id")],
            ).fetchone()
        return None if row is None else self._subject_from_row(row)

    # -- decide / apply ------------------------------------------------------

    def decide_action(
        self,
        *,
        subject_id: str,
        action_kind: ActionKind | str,
        reason: str = "",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Decide a recovery action; identical keys are replay-safe."""

        sid = _text(subject_id, "subject_id")
        kind = (
            action_kind
            if isinstance(action_kind, ActionKind)
            else ActionKind(str(action_kind).strip().lower())
        )
        reason_text = _text(reason, "reason", required=False)
        payload = _bounded_mapping(body, name="action body")
        if "json_receipt_path" in payload or "queue_file" in payload:
            raise DatabaseRecoveryError(
                "JSON receipt or queue file alone cannot decide recovery"
            )
        key = _text(idempotency_key, "idempotency_key", required=False)
        if not key:
            key = _default_idempotency_key(
                subject_id=sid,
                action_kind=kind.value,
                reason=reason_text,
                body=payload,
            )
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT * FROM recovery_actions WHERE idempotency_key = ?",
                    [key],
                ).fetchone()
                if existing is not None:
                    action = self._action_from_row(existing)
                    # Mark as replayed if already applied, keep durable identity.
                    if action.status in {
                        ActionStatus.APPLIED,
                        ActionStatus.REPLAYED,
                        ActionStatus.EXHAUSTED,
                    }:
                        if action.status is ActionStatus.APPLIED:
                            connection.execute(
                                """
                                UPDATE recovery_actions
                                SET status = 'replayed'
                                WHERE action_id = ? AND status = 'applied'
                                """,
                                [action.action_id],
                            )
                            row = connection.execute(
                                "SELECT * FROM recovery_actions WHERE action_id = ?",
                                [action.action_id],
                            ).fetchone()
                            self._record_event(
                                connection,
                                subject_id=action.subject_id,
                                action_id=action.action_id,
                                event_type="action_replayed",
                                body={"idempotency_key": key},
                                now_ms=now,
                            )
                            self._commit_if_idle(connection)
                            return self._action_from_row(row)
                        self._commit_if_idle(connection)
                        return action
                    self._commit_if_idle(connection)
                    return action

                subject_row = connection.execute(
                    "SELECT * FROM recovery_subjects WHERE subject_id = ?",
                    [sid],
                ).fetchone()
                if subject_row is None:
                    raise DatabaseRecoveryConflictError(f"unknown subject {sid}")
                subject = self._subject_from_row(subject_row)
                if subject.status is SubjectStatus.QUARANTINED and kind is not ActionKind.RESCUE:
                    raise DatabaseRecoveryExhaustedError(
                        f"subject {sid} is quarantined; only rescue is admitted"
                    )

                action_id = _new_id("recovery-action")
                connection.execute(
                    """
                    INSERT INTO recovery_actions(
                        action_id, subject_id, subject_kind, subject_ref, task_cid,
                        attempt_id, worktree_id, entry_id, fencing_token, fence_epoch,
                        action_kind, status, idempotency_key, decided_at_ms, reason,
                        body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        action_id,
                        subject.subject_id,
                        subject.subject_kind.value,
                        subject.subject_id.split(":", 1)[-1],
                        subject.task_cid,
                        subject.attempt_id,
                        subject.worktree_id,
                        subject.entry_id,
                        subject.fencing_token,
                        subject.fence_epoch,
                        kind.value,
                        ActionStatus.DECIDED.value,
                        key,
                        now,
                        reason_text,
                        _canonical_json(payload),
                    ],
                )
                connection.execute(
                    """
                    UPDATE recovery_subjects
                    SET status = 'recovering', updated_at_ms = ?
                    WHERE subject_id = ?
                    """,
                    [now, subject.subject_id],
                )
                self._record_event(
                    connection,
                    subject_id=subject.subject_id,
                    action_id=action_id,
                    event_type="action_decided",
                    body={
                        "action_kind": kind.value,
                        "idempotency_key": key,
                        "reason": reason_text,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM recovery_actions WHERE action_id = ?",
                    [action_id],
                ).fetchone()
                self._commit_if_idle(connection)
                return self._action_from_row(row)
            except Exception:
                self._rollback_if_open(connection)
                raise

    def apply_action(
        self,
        action: RecoveryAction,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Apply a decided action; identical keys remain replay-safe."""

        payload = _bounded_mapping(result, name="result body")
        if "json_receipt_path" in payload or "queue_file" in payload:
            raise DatabaseRecoveryError(
                "JSON receipt or queue file alone cannot apply recovery"
            )
        now = self._now_ms()
        result_digest = _sha256_hex(_canonical_json(payload).encode("utf-8"))

        with self._lock:
            connection = self._require()
            self._begin(connection)
            try:
                existing = connection.execute(
                    "SELECT * FROM recovery_actions WHERE action_id = ?",
                    [action.action_id],
                ).fetchone()
                if existing is None:
                    raise DatabaseRecoveryConflictError(
                        f"unknown recovery action {action.action_id}"
                    )
                current = self._action_from_row(existing)
                if current.status in {
                    ActionStatus.APPLIED,
                    ActionStatus.REPLAYED,
                    ActionStatus.EXHAUSTED,
                }:
                    if (
                        current.result_digest
                        and current.result_digest != result_digest
                        and payload
                    ):
                        raise DatabaseRecoveryConflictError(
                            "idempotent recovery action result digest mismatch"
                        )
                    if current.status is ActionStatus.APPLIED:
                        connection.execute(
                            """
                            UPDATE recovery_actions
                            SET status = 'replayed'
                            WHERE action_id = ? AND status = 'applied'
                            """,
                            [current.action_id],
                        )
                        row = connection.execute(
                            "SELECT * FROM recovery_actions WHERE action_id = ?",
                            [current.action_id],
                        ).fetchone()
                        self._record_event(
                            connection,
                            subject_id=current.subject_id,
                            action_id=current.action_id,
                            event_type="action_replayed",
                            body={"result_digest": current.result_digest},
                            now_ms=now,
                        )
                        self._commit_if_idle(connection)
                        return self._action_from_row(row)
                    self._commit_if_idle(connection)
                    return current
                if current.status is not ActionStatus.DECIDED:
                    raise DatabaseRecoveryConflictError(
                        f"action {current.action_id} status is {current.status.value}"
                    )

                subject_row = connection.execute(
                    "SELECT * FROM recovery_subjects WHERE subject_id = ?",
                    [current.subject_id],
                ).fetchone()
                if subject_row is None:
                    raise DatabaseRecoveryConflictError(
                        f"unknown subject {current.subject_id}"
                    )
                subject = self._subject_from_row(subject_row)

                subject_status = subject.status
                retry_count = subject.retry_count
                action_status = ActionStatus.APPLIED

                if current.action_kind is ActionKind.RETRY:
                    if subject.retry_count >= subject.max_retries:
                        action_status = ActionStatus.EXHAUSTED
                        subject_status = SubjectStatus.QUARANTINED
                    else:
                        retry_count = subject.retry_count + 1
                        if retry_count >= subject.max_retries:
                            action_status = ActionStatus.EXHAUSTED
                            subject_status = SubjectStatus.QUARANTINED
                        else:
                            subject_status = SubjectStatus.OPEN
                elif current.action_kind is ActionKind.QUARANTINE:
                    subject_status = SubjectStatus.QUARANTINED
                elif current.action_kind is ActionKind.RESCUE:
                    subject_status = SubjectStatus.RESCUED
                elif current.action_kind in {
                    ActionKind.REPLAY,
                    ActionKind.RECONCILE,
                    ActionKind.FENCE_STALE_CLAIM,
                    ActionKind.INTERRUPT_VALIDATION,
                    ActionKind.INTERRUPT_MERGE,
                }:
                    subject_status = SubjectStatus.RECONCILED
                else:
                    subject_status = SubjectStatus.CLOSED

                connection.execute(
                    """
                    UPDATE recovery_actions
                    SET status = ?,
                        applied_at_ms = ?,
                        result_digest = ?,
                        body_json = ?
                    WHERE action_id = ? AND status = 'decided'
                    """,
                    [
                        action_status.value,
                        now,
                        result_digest,
                        _canonical_json({**dict(current.body), **payload}),
                        current.action_id,
                    ],
                )
                connection.execute(
                    """
                    UPDATE recovery_subjects
                    SET status = ?,
                        retry_count = ?,
                        updated_at_ms = ?
                    WHERE subject_id = ?
                    """,
                    [
                        subject_status.value,
                        retry_count,
                        now,
                        subject.subject_id,
                    ],
                )

                if current.action_kind is ActionKind.RECONCILE:
                    receipt_id = _new_id("reconciliation")
                    replayed = int(payload.get("replayed_event_count") or 0)
                    connection.execute(
                        """
                        INSERT INTO reconciliation_receipts(
                            receipt_id, subject_id, action_id, task_cid, entry_id,
                            replayed_event_count, status, recorded_at_ms, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            receipt_id,
                            subject.subject_id,
                            current.action_id,
                            subject.task_cid,
                            subject.entry_id,
                            max(0, replayed),
                            "reconciled",
                            now,
                            _canonical_json(payload),
                        ],
                    )

                self._record_event(
                    connection,
                    subject_id=subject.subject_id,
                    action_id=current.action_id,
                    event_type="action_applied",
                    body={
                        "action_kind": current.action_kind.value,
                        "status": action_status.value,
                        "result_digest": result_digest,
                        "subject_status": subject_status.value,
                    },
                    now_ms=now,
                )
                row = connection.execute(
                    "SELECT * FROM recovery_actions WHERE action_id = ?",
                    [current.action_id],
                ).fetchone()
                self._commit_if_idle(connection)
                applied = self._action_from_row(row)
                if applied.status is ActionStatus.EXHAUSTED:
                    # Still return the durable exhausted action; callers inspect status.
                    return applied
                return applied
            except Exception:
                self._rollback_if_open(connection)
                raise

    def decide_and_apply(
        self,
        *,
        subject_id: str,
        action_kind: ActionKind | str,
        reason: str = "",
        idempotency_key: str = "",
        body: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> RecoveryAction:
        """Convenience path for decide then apply under one caller transaction."""

        action = self.decide_action(
            subject_id=subject_id,
            action_kind=action_kind,
            reason=reason,
            idempotency_key=idempotency_key,
            body=body,
        )
        if action.status in {
            ActionStatus.APPLIED,
            ActionStatus.REPLAYED,
            ActionStatus.EXHAUSTED,
        }:
            return action
        return self.apply_action(action, result=result)

    # -- queries -------------------------------------------------------------

    def get_action(self, action_id: str) -> RecoveryAction | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM recovery_actions WHERE action_id = ?",
                [_text(action_id, "action_id")],
            ).fetchone()
        return None if row is None else self._action_from_row(row)

    def get_action_by_idempotency_key(
        self, idempotency_key: str
    ) -> RecoveryAction | None:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM recovery_actions WHERE idempotency_key = ?",
                [_text(idempotency_key, "idempotency_key")],
            ).fetchone()
        return None if row is None else self._action_from_row(row)

    def list_actions(
        self,
        *,
        subject_id: str = "",
        task_cid: str = "",
        action_kind: str | ActionKind | None = None,
        status: str | ActionStatus | None = None,
    ) -> tuple[RecoveryAction, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(_text(subject_id, "subject_id"))
        if task_cid:
            clauses.append("task_cid = ?")
            params.append(_text(task_cid, "task_cid"))
        if action_kind is not None:
            kind_value = (
                action_kind.value
                if isinstance(action_kind, ActionKind)
                else str(action_kind)
            )
            clauses.append("action_kind = ?")
            params.append(kind_value)
        if status is not None:
            status_value = (
                status.value if isinstance(status, ActionStatus) else str(status)
            )
            clauses.append("status = ?")
            params.append(status_value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM recovery_actions
                {where}
                ORDER BY decided_at_ms ASC, action_id ASC
                """,
                params,
            ).fetchall()
        return tuple(self._action_from_row(row) for row in rows)

    def list_reconciliation_receipts(
        self, *, subject_id: str = ""
    ) -> tuple[ReconciliationReceipt, ...]:
        with self._lock:
            connection = self._require()
            if subject_id:
                rows = connection.execute(
                    """
                    SELECT * FROM reconciliation_receipts
                    WHERE subject_id = ?
                    ORDER BY recorded_at_ms ASC
                    """,
                    [_text(subject_id, "subject_id")],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM reconciliation_receipts
                    ORDER BY recorded_at_ms ASC
                    """
                ).fetchall()
        return tuple(self._reconciliation_from_row(row) for row in rows)

    def events(self, *, subject_id: str = "") -> tuple[dict[str, Any], ...]:
        with self._lock:
            connection = self._require()
            if subject_id:
                rows = connection.execute(
                    """
                    SELECT * FROM recovery_events
                    WHERE subject_id = ?
                    ORDER BY observed_at_ms ASC, event_id ASC
                    """,
                    [_text(subject_id, "subject_id")],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM recovery_events
                    ORDER BY observed_at_ms ASC, event_id ASC
                    """
                ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            mapping = _row_mapping(row)
            body_raw = _row_get(mapping, "body_json", default="{}")
            try:
                body = json.loads(body_raw or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                body = {}
            events.append(
                {
                    "schema": RECOVERY_EVENT_SCHEMA,
                    "event_id": str(_row_get(mapping, "event_id", default="")),
                    "subject_id": str(_row_get(mapping, "subject_id", default="")),
                    "action_id": str(_row_get(mapping, "action_id", default="") or ""),
                    "event_type": str(_row_get(mapping, "event_type", default="")),
                    "observed_at_ms": int(
                        _row_get(mapping, "observed_at_ms", default=0) or 0
                    ),
                    "body": body if isinstance(body, Mapping) else {},
                }
            )
        return tuple(events)

    # -- row mappers ---------------------------------------------------------

    def _subject_from_row(self, row: Any) -> RecoverySubject:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return RecoverySubject(
            subject_id=str(_row_get(mapping, "subject_id", default="")),
            subject_kind=SubjectKind(
                str(_row_get(mapping, "subject_kind", default="task"))
            ),
            task_cid=str(_row_get(mapping, "task_cid", default="") or ""),
            attempt_id=str(_row_get(mapping, "attempt_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            entry_id=str(_row_get(mapping, "entry_id", default="") or ""),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            retry_count=int(_row_get(mapping, "retry_count", default=0) or 0),
            max_retries=int(_row_get(mapping, "max_retries", default=1) or 1),
            status=SubjectStatus(
                str(_row_get(mapping, "status", default="open"))
            ),
            registered_at_ms=int(
                _row_get(mapping, "registered_at_ms", default=0) or 0
            ),
            updated_at_ms=int(_row_get(mapping, "updated_at_ms", default=0) or 0),
            body=body if isinstance(body, Mapping) else {},
        )

    def _action_from_row(self, row: Any) -> RecoveryAction:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return RecoveryAction(
            action_id=str(_row_get(mapping, "action_id", default="")),
            subject_id=str(_row_get(mapping, "subject_id", default="")),
            subject_kind=SubjectKind(
                str(_row_get(mapping, "subject_kind", default="task"))
            ),
            subject_ref=str(_row_get(mapping, "subject_ref", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="") or ""),
            attempt_id=str(_row_get(mapping, "attempt_id", default="") or ""),
            worktree_id=str(_row_get(mapping, "worktree_id", default="") or ""),
            entry_id=str(_row_get(mapping, "entry_id", default="") or ""),
            fencing_token=int(_row_get(mapping, "fencing_token", default=0) or 0),
            fence_epoch=int(_row_get(mapping, "fence_epoch", default=0) or 0),
            action_kind=ActionKind(
                str(_row_get(mapping, "action_kind", default="retry"))
            ),
            status=ActionStatus(
                str(_row_get(mapping, "status", default="decided"))
            ),
            idempotency_key=str(
                _row_get(mapping, "idempotency_key", default="")
            ),
            decided_at_ms=int(_row_get(mapping, "decided_at_ms", default=0) or 0),
            applied_at_ms=int(_row_get(mapping, "applied_at_ms", default=0) or 0),
            result_digest=str(
                _row_get(mapping, "result_digest", default="") or ""
            ),
            reason=str(_row_get(mapping, "reason", default="") or ""),
            body=body if isinstance(body, Mapping) else {},
        )

    def _reconciliation_from_row(self, row: Any) -> ReconciliationReceipt:
        mapping = _row_mapping(row)
        body_raw = _row_get(mapping, "body_json", default="{}")
        try:
            body = json.loads(body_raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            body = {}
        return ReconciliationReceipt(
            receipt_id=str(_row_get(mapping, "receipt_id", default="")),
            subject_id=str(_row_get(mapping, "subject_id", default="")),
            action_id=str(_row_get(mapping, "action_id", default="")),
            task_cid=str(_row_get(mapping, "task_cid", default="") or ""),
            entry_id=str(_row_get(mapping, "entry_id", default="") or ""),
            replayed_event_count=int(
                _row_get(mapping, "replayed_event_count", default=0) or 0
            ),
            status=str(_row_get(mapping, "status", default="")),
            recorded_at_ms=int(
                _row_get(mapping, "recorded_at_ms", default=0) or 0
            ),
            body=body if isinstance(body, Mapping) else {},
        )


def open_database_recovery(
    database_path: Path | str,
    *,
    clock_ms: ClockMs | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> DatabaseRecovery:
    """Open a :class:`DatabaseRecovery` on ``database_path``."""

    return DatabaseRecovery(
        database_path,
        clock_ms=clock_ms,
        max_retries=max_retries,
    ).open()


__all__ = [
    "DATABASE_RECOVERY_INTERFACE",
    "RECOVERY_ACTION_INTERFACE",
    "RECONCILIATION_RECEIPT_INTERFACE",
    "DATABASE_RECOVERY_SCHEMA",
    "RECOVERY_ACTION_SCHEMA",
    "RECONCILIATION_RECEIPT_SCHEMA",
    "DEFAULT_MAX_RETRIES",
    "SubjectKind",
    "SubjectStatus",
    "ActionKind",
    "ActionStatus",
    "RecoverySubject",
    "RecoveryAction",
    "ReconciliationReceipt",
    "DatabaseRecovery",
    "DatabaseRecoveryError",
    "DatabaseRecoveryNotOpenError",
    "DatabaseRecoveryConflictError",
    "DatabaseRecoveryBoundsError",
    "DatabaseRecoveryExhaustedError",
    "DuckDBUnavailableError",
    "duckdb_available",
    "open_database_recovery",
]
