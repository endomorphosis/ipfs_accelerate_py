"""Transactional intent repository for objectives, goals, plans, tasks, queues.

DQP-012 / IntentRepository@1 / PlanRevisionRepository@1
=======================================================

Migrates objectives, goals, plans, tasks, queue backoff, attempts, blocks, and
completion evidence into the control-plane DuckDB schema. Every mutation advances
normalized projections and appends a domain event in the **same** transaction —
no cross-file saga is required.

Invariants
----------
* Canonical identities (``objective_id``, ``goal_cid``, ``plan_cid``,
  ``task_cid``) are stable; display aliases never serve as durable keys.
* Task/goal completion requires **current** required evidence (validation
  results or evidence nodes bound to the live acceptance criteria). Exported
  status strings are never completion authority.
* Rebuilding projections from admitted domain events matches the live rows.
* CAS heads protect concurrent writers on objectives, plans, and tasks.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    ControlPlaneBoundsError,
    ControlPlaneContractError,
    ControlPlaneIdentityError,
    canonical_json_bytes,
    content_identity,
)
from .control_plane_migrations import duckdb_available
from .control_plane_schema import install_control_plane_schema
from .duckdb_state import exclusive_file_lock, open_duckdb_connection


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

INTENT_REPOSITORY_INTERFACE: Final[str] = "IntentRepository@1"
PLAN_REVISION_REPOSITORY_INTERFACE: Final[str] = "PlanRevisionRepository@1"

INTENT_REPOSITORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-repository@1"
)
PLAN_REVISION_REPOSITORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-repository@1"
)
INTENT_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-event@1"
)
INTENT_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-snapshot@1"
)
INTENT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-receipt@1"
)
QUEUE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-queue-entry@1"
)
COMPLETION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
)
PLAN_HEAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-plan-head@1"
)

INTENT_STREAM_ID: Final[str] = "stream:intent"
DEFAULT_OWNER_ID: Final[str] = "intent-repository:local"
DEFAULT_SESSION_ID: Final[str] = "session:intent"

MAX_ID_BYTES: Final[int] = 512
MAX_BODY_BYTES: Final[int] = 262_144
MAX_PAGE_LIMIT: Final[int] = 1_000
DEFAULT_PAGE_LIMIT: Final[int] = 100
MAX_EVENTS: Final[int] = 1_000_000
MAX_ACCEPTANCE: Final[int] = 256
MAX_VALIDATIONS: Final[int] = 256
MAX_OUTPUTS: Final[int] = 256
MAX_DEPENDENCIES: Final[int] = 1_024
MAX_EVIDENCE: Final[int] = 4_096
DEFAULT_EVIDENCE_FRESHNESS_SECONDS: Final[int] = 3_600

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")

_READY_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "proposed",
        "admitted",
        "pending",
        "ready",
        "todo",
        "queued",
        "retrying",
    }
)
_COMPLETED_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "skipped", "complete", "done"}
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_COMPLETED_STATUSES,
        "cancelled",
        "failed",
        "quarantined",
        "rejected",
    }
)
_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_READY_STATUSES,
        *_TERMINAL_STATUSES,
        "claimed",
        "in_progress",
        "running",
        "blocked",
    }
)
_GOAL_OPEN_STATUSES: Final[frozenset[str]] = frozenset(
    {"open", "active", "reopened", "provisionally_complete", "analysis_inconclusive"}
)
_GOAL_CLOSED_STATUSES: Final[frozenset[str]] = frozenset(
    {"verified_complete", "completed", "complete", "done", "blocked"}
)

# Intent-owned projection tables fully rebuilt from admitted intent events.
_PROJECTION_TABLES: Final[tuple[str, ...]] = (
    "objectives",
    "objective_revisions",
    "goals",
    "goal_edges",
    "plans",
    "plan_revisions",
    "planning_decisions",
    "plan_candidates",
    "tasks",
    "task_revisions",
    "task_dependencies",
    "task_outputs",
    "task_acceptance",
    "task_validations",
    "task_assignments",
    "task_blocks",
    "task_attempts",
    "completion_receipts",
    "evidence_nodes",
)

# Shared tables: only intent-owned rows are cleared (not the whole table).
_SHARED_QUEUE_LEASE_SCHEMA: Final[str] = QUEUE_ENTRY_SCHEMA


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntentRepositoryError(RuntimeError):
    """Base fail-closed error for intent repository operations."""


class IntentRepositoryConflictError(IntentRepositoryError):
    """CAS head, fence, or expected-revision conflict."""


class IntentRepositoryIntegrityError(IntentRepositoryError):
    """Schema, identity, or projection integrity failure."""


class IntentRepositoryBoundsError(IntentRepositoryError, ValueError):
    """A count, byte, or page bound was exceeded."""


class IntentRepositoryNotOpenError(IntentRepositoryError):
    """Operation requires an open repository session."""


class IntentCompletionError(IntentRepositoryError):
    """Completion refused because required current evidence is missing."""


class IntentEvidenceError(IntentRepositoryError):
    """Evidence material is stale, foreign, or incomplete."""


class DuckDBUnavailableError(IntentRepositoryError):
    """DuckDB is required but missing from the environment."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class IntentEventType(str, Enum):
    """Closed set of admitted intent domain-event types."""

    OBJECTIVE_UPSERTED = "intent.objective_upserted"
    OBJECTIVE_REVISED = "intent.objective_revised"
    GOAL_UPSERTED = "intent.goal_upserted"
    GOAL_EDGE_LINKED = "intent.goal_edge_linked"
    GOAL_REOPENED = "intent.goal_reopened"
    PLAN_UPSERTED = "intent.plan_upserted"
    PLAN_REVISION_APPENDED = "intent.plan_revision_appended"
    PLAN_SUPERSEDED = "intent.plan_superseded"
    PLAN_CONTINUED = "intent.plan_continued"
    PLAN_HEAD_SET = "intent.plan_head_set"
    TASK_UPSERTED = "intent.task_upserted"
    TASK_DEPENDENCIES_SET = "intent.task_dependencies_set"
    TASK_OUTPUTS_SET = "intent.task_outputs_set"
    TASK_ACCEPTANCE_SET = "intent.task_acceptance_set"
    TASK_VALIDATIONS_SET = "intent.task_validations_set"
    TASK_STATUS_CHANGED = "intent.task_status_changed"
    TASK_BLOCKED = "intent.task_blocked"
    TASK_UNBLOCKED = "intent.task_unblocked"
    ATTEMPT_RECORDED = "intent.attempt_recorded"
    QUEUE_BACKOFF = "intent.queue_backoff"
    QUEUE_RETRY = "intent.queue_retry"
    EVIDENCE_RECORDED = "intent.evidence_recorded"
    VALIDATION_RECORDED = "intent.validation_recorded"
    COMPLETION_RECORDED = "intent.completion_recorded"
    RECOVERY_APPLIED = "intent.recovery_applied"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_duckdb() -> Any:
    if not duckdb_available():
        raise DuckDBUnavailableError(
            "DuckDB is required for IntentRepository; install the optional "
            "duckdb dependency"
        )
    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise DuckDBUnavailableError(
            "DuckDB is required for IntentRepository"
        ) from exc
    return duckdb


def _utc_iso(moment: datetime | None = None) -> str:
    value = moment or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _identifier(value: Any, *, noun: str) -> str:
    if not isinstance(value, str):
        raise ControlPlaneIdentityError(f"{noun} must be a string")
    text = value.strip()
    if not text:
        raise ControlPlaneIdentityError(f"{noun} must not be empty")
    if len(text.encode("utf-8")) > MAX_ID_BYTES:
        raise ControlPlaneBoundsError(f"{noun} exceeds its byte bound")
    if "\x00" in text or not _SAFE_ID.match(text):
        raise ControlPlaneIdentityError(f"{noun} is not a safe identifier")
    return text


def _optional_identifier(value: Any, *, noun: str) -> str:
    if value is None or value == "":
        return ""
    return _identifier(value, noun=noun)


def _status(value: Any, *, allowed: frozenset[str], noun: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text not in allowed:
        raise IntentRepositoryError(
            f"{noun} status {value!r} is not in the closed set"
        )
    return text


def _jsonable(value: Any) -> Any:
    """Coerce nested values into canonical-JSON-safe Python structures."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise IntentRepositoryError("float values are not allowed in intent JSON")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: str(item))
    raise IntentRepositoryError(
        f"unsupported intent JSON value type: {type(value).__name__}"
    )


def _canonical(value: Any, *, noun: str = "payload") -> str:
    try:
        payload = canonical_json_bytes(_jsonable(value))
    except (ControlPlaneContractError, ControlPlaneBoundsError) as exc:
        raise IntentRepositoryError(f"{noun} is not canonical JSON") from exc
    if len(payload) > MAX_BODY_BYTES:
        raise IntentRepositoryBoundsError(f"{noun} exceeds body byte bound")
    return payload.decode("utf-8")


def _decode_json(value: Any, *, noun: str = "json") -> Any:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    text = str(value)
    if not text:
        return {}
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntentRepositoryIntegrityError(
            f"{noun} is not valid JSON"
        ) from exc


def _mapping(value: Any, *, noun: str = "mapping") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise IntentRepositoryError(f"{noun} must be a mapping")
    return {str(key): member for key, member in value.items()}


def _bounded_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_PAGE_LIMIT
    ):
        raise IntentRepositoryBoundsError(
            f"limit must be in [1, {MAX_PAGE_LIMIT}]"
        )
    return limit


def _nonneg_int(value: Any, *, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntentRepositoryBoundsError(f"{noun} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, noun: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IntentRepositoryBoundsError(f"{noun} must be a positive integer")
    return value


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentReceipt:
    """Durable receipt for one intent mutation."""

    SCHEMA: ClassVar[str] = INTENT_RECEIPT_SCHEMA

    event_id: str
    event_type: str
    global_sequence: int
    recorded_at: str
    subject_id: str
    revision: int
    changed: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "global_sequence": int(self.global_sequence),
            "recorded_at": self.recorded_at,
            "subject_id": self.subject_id,
            "revision": int(self.revision),
            "changed": bool(self.changed),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class IntentSnapshot:
    """Generation-bound snapshot of intent projections."""

    SCHEMA: ClassVar[str] = INTENT_SNAPSHOT_SCHEMA

    objective_count: int
    goal_count: int
    plan_count: int
    task_count: int
    dependency_count: int
    event_watermark: int
    projection_cid: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "objective_count": int(self.objective_count),
            "goal_count": int(self.goal_count),
            "plan_count": int(self.plan_count),
            "task_count": int(self.task_count),
            "dependency_count": int(self.dependency_count),
            "event_watermark": int(self.event_watermark),
            "projection_cid": self.projection_cid,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class QueueEntry:
    """Queue backoff / selection state for one task."""

    SCHEMA: ClassVar[str] = QUEUE_ENTRY_SCHEMA

    task_cid: str
    attempt: int = 0
    retry_not_before_ms: int = 0
    selection_penalty: int = 0
    consecutive_failures: int = 0
    state: str = "ready"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_cid": self.task_cid,
            "attempt": int(self.attempt),
            "retry_not_before_ms": int(self.retry_not_before_ms),
            "selection_penalty": int(self.selection_penalty),
            "consecutive_failures": int(self.consecutive_failures),
            "state": self.state,
            "reason": self.reason,
        }

    def is_cooled_down(self, *, now_ms: int | None = None) -> bool:
        clock = _now_ms() if now_ms is None else int(now_ms)
        return int(self.retry_not_before_ms) > clock


@dataclass(frozen=True)
class PlanHead:
    """Active plan head for a goal."""

    SCHEMA: ClassVar[str] = PLAN_HEAD_SCHEMA

    plan_cid: str
    goal_cid: str
    revision: int
    status: str
    superseded_by: str = ""
    continuation_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_cid": self.plan_cid,
            "goal_cid": self.goal_cid,
            "revision": int(self.revision),
            "status": self.status,
            "superseded_by": self.superseded_by,
            "continuation_of": self.continuation_of,
        }


# ---------------------------------------------------------------------------
# IntentRepository
# ---------------------------------------------------------------------------


class IntentRepository:
    """Transactional authority for intent-domain control-plane state.

    Interface: ``IntentRepository@1``.
    """

    INTERFACE: ClassVar[str] = INTENT_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = INTENT_REPOSITORY_SCHEMA

    def __init__(
        self,
        database_path: str | Path,
        *,
        owner_id: str = DEFAULT_OWNER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        install_schema: bool = True,
        evidence_freshness_seconds: int = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
        lock_timeout_seconds: float = 30.0,
        clock_ms: Any | None = None,
    ) -> None:
        _require_duckdb()
        self.database_path = Path(database_path).absolute()
        self.owner_id = _identifier(owner_id, noun="owner_id")
        self.session_id = _identifier(session_id, noun="session_id")
        if (
            isinstance(evidence_freshness_seconds, bool)
            or not isinstance(evidence_freshness_seconds, int)
            or evidence_freshness_seconds < 0
        ):
            raise IntentRepositoryBoundsError(
                "evidence_freshness_seconds must be a non-negative integer"
            )
        self.evidence_freshness_seconds = int(evidence_freshness_seconds)
        if lock_timeout_seconds <= 0:
            raise IntentRepositoryBoundsError(
                "lock_timeout_seconds must be positive"
            )
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._clock_ms = clock_ms or _now_ms
        self._lock_path = self.database_path.with_name(
            f".{self.database_path.name}.intent.lock"
        )
        self._open = False
        self._closed = False
        if install_schema:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.database_path.exists():
                install_control_plane_schema(
                    self.database_path,
                    application_version="0.0.45",
                    tool_version="1.5.2",
                    owner_id=self.owner_id,
                )
            else:
                # Ensure schema is present for pre-created empty files.
                try:
                    connection = open_duckdb_connection(self.database_path)
                    try:
                        tables = {
                            str(row[0])
                            for row in connection.execute("SHOW TABLES").fetchall()
                        }
                    finally:
                        connection.close()
                    if "tasks" not in tables:
                        install_control_plane_schema(
                            self.database_path,
                            application_version="0.0.45",
                            tool_version="1.5.2",
                            owner_id=self.owner_id,
                        )
                except Exception:
                    install_control_plane_schema(
                        self.database_path,
                        application_version="0.0.45",
                        tool_version="1.5.2",
                        owner_id=self.owner_id,
                    )
        self._open = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open and not self._closed

    def close(self) -> None:
        self._closed = True
        self._open = False

    def __enter__(self) -> IntentRepository:
        self._require_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed or not self._open:
            raise IntentRepositoryNotOpenError("intent repository is not open")

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[Any]:
        self._require_open()
        # Match DuckDBTaskSource / StateTransaction durability: begin with SQL,
        # commit/rollback with SQL, and always close the adapter explicitly.
        # Avoid relying on DuckDBConnection.__exit__ transaction bookkeeping,
        # which can mark a SQL-started transaction inactive before COMMIT runs.
        if write:
            with exclusive_file_lock(
                self._lock_path, timeout_seconds=self.lock_timeout_seconds
            ):
                connection = open_duckdb_connection(self.database_path)
                try:
                    connection.execute("BEGIN TRANSACTION")
                    try:
                        yield connection
                        connection.execute("COMMIT")
                    except BaseException:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        raise
                finally:
                    connection.close()
        else:
            connection = open_duckdb_connection(self.database_path)
            try:
                yield connection
            finally:
                connection.close()

    # -- event plumbing ------------------------------------------------------

    def _next_global_sequence(self, connection: Any) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
        ).fetchone()
        return int(row[0] if row else 0) + 1

    def _next_stream_sequence(self, connection: Any) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM domain_events "
            "WHERE stream_id = ?",
            [INTENT_STREAM_ID],
        ).fetchone()
        return int(row[0] if row else 0) + 1

    def _append_event(
        self,
        connection: Any,
        *,
        event_type: IntentEventType | str,
        subject_id: str,
        body: Mapping[str, Any],
        task_cid: str = "",
        attempt_id: str = "",
    ) -> IntentReceipt:
        global_sequence = self._next_global_sequence(connection)
        if global_sequence > MAX_EVENTS:
            raise IntentRepositoryBoundsError("domain event population exceeded")
        stream_sequence = self._next_stream_sequence(connection)
        recorded_at = _utc_iso()
        event_type_value = (
            event_type.value
            if isinstance(event_type, IntentEventType)
            else str(event_type)
        )
        body_payload = {
            "schema": INTENT_EVENT_SCHEMA,
            "event_type": event_type_value,
            "subject_id": subject_id,
            "body": _jsonable(dict(body)),
            "recorded_at": recorded_at,
            "owner_id": self.owner_id,
        }
        event_id = content_identity(
            {
                "stream_id": INTENT_STREAM_ID,
                "sequence": stream_sequence,
                "global_sequence": global_sequence,
                "event_type": event_type_value,
                "body": body_payload,
            }
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                INTENT_STREAM_ID,
                stream_sequence,
                global_sequence,
                event_type_value,
                task_cid or "",
                attempt_id or "",
                self.session_id,
                recorded_at,
                _canonical(body_payload, noun="event body"),
            ],
        )
        return IntentReceipt(
            event_id=event_id,
            event_type=event_type_value,
            global_sequence=global_sequence,
            recorded_at=recorded_at,
            subject_id=subject_id,
            revision=int(body.get("revision") or 0),
            changed=True,
            details=MappingProxyType(dict(body)),
        )

    def event_watermark(self) -> int:
        with self._connection(write=False) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()
            return int(row[0] if row else 0)

    def list_events(
        self,
        *,
        after_global_sequence: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> tuple[Mapping[str, Any], ...]:
        selected = _bounded_limit(limit)
        after = _nonneg_int(after_global_sequence, noun="after_global_sequence")
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT event_id, stream_id, sequence, global_sequence, event_type,
                       task_cid, attempt_id, session_id, recorded_at, body_json
                FROM domain_events
                WHERE global_sequence > ?
                ORDER BY global_sequence ASC
                LIMIT ?
                """,
                [after, selected],
            ).fetchall()
        return tuple(
            MappingProxyType(
                {
                    "event_id": str(row[0]),
                    "stream_id": str(row[1]),
                    "sequence": int(row[2]),
                    "global_sequence": int(row[3]),
                    "event_type": str(row[4]),
                    "task_cid": str(row[5] or ""),
                    "attempt_id": str(row[6] or ""),
                    "session_id": str(row[7] or ""),
                    "recorded_at": str(row[8]),
                    "body": _decode_json(row[9], noun="event body"),
                }
            )
            for row in rows
        )

    # -- objectives ----------------------------------------------------------

    def upsert_objective(
        self,
        *,
        objective_id: str,
        objective_alias: str,
        title: str,
        status: str = "open",
        priority: str = "P2",
        parent_objective_id: str = "",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        oid = _identifier(objective_id, noun="objective_id")
        alias = _identifier(objective_alias, noun="objective_alias")
        title_text = str(title or "").strip() or alias
        status_text = str(status or "open").strip().lower()
        priority_text = str(priority or "P2").strip() or "P2"
        parent = _optional_identifier(
            parent_objective_id, noun="parent_objective_id"
        )
        body_map = _mapping(body, noun="objective body")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            existing = connection.execute(
                "SELECT revision, status, body_json FROM objectives "
                "WHERE objective_id = ?",
                [oid],
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "objective CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO objectives (
                        objective_id, objective_alias, parent_objective_id,
                        title, status, priority, created_at, updated_at,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        oid,
                        alias,
                        parent,
                        title_text,
                        status_text,
                        priority_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="objective body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if (
                    expected_revision is not None
                    and expected_revision != current_revision
                ):
                    raise IntentRepositoryConflictError(
                        "objective revision CAS is stale"
                    )
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE objectives SET
                        objective_alias = ?, parent_objective_id = ?,
                        title = ?, status = ?, priority = ?, updated_at = ?,
                        revision = ?, body_json = ?
                    WHERE objective_id = ? AND revision = ?
                    """,
                    [
                        alias,
                        parent,
                        title_text,
                        status_text,
                        priority_text,
                        now,
                        revision,
                        _canonical(body_map, noun="objective body"),
                        oid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO objective_revisions (
                    objective_id, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="objective revision body"),
                    now,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.OBJECTIVE_UPSERTED,
                subject_id=oid,
                body={
                    "objective_id": oid,
                    "objective_alias": alias,
                    "parent_objective_id": parent,
                    "title": title_text,
                    "status": status_text,
                    "priority": priority_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def get_objective(self, objective_id: str) -> Mapping[str, Any] | None:
        oid = _identifier(objective_id, noun="objective_id")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT objective_id, objective_alias, parent_objective_id,
                       title, status, priority, created_at, updated_at,
                       revision, body_json
                FROM objectives WHERE objective_id = ? OR objective_alias = ?
                LIMIT 1
                """,
                [oid, oid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2] or ""),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "created_at": str(row[6]),
                "updated_at": str(row[7]),
                "revision": int(row[8]),
                "body": _decode_json(row[9], noun="objective body"),
            }
        )

    # -- goals ---------------------------------------------------------------

    def upsert_goal(
        self,
        *,
        goal_cid: str,
        goal_alias: str,
        title: str,
        objective_id: str = "",
        parent_goal_cid: str = "",
        ordinal: int = 0,
        status: str = "open",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> IntentReceipt:
        gcid = _identifier(goal_cid, noun="goal_cid")
        alias = _identifier(goal_alias, noun="goal_alias")
        title_text = str(title or "").strip() or alias
        oid = _optional_identifier(objective_id, noun="objective_id")
        parent = _optional_identifier(parent_goal_cid, noun="parent_goal_cid")
        ord_value = _nonneg_int(ordinal, noun="ordinal")
        status_text = str(status or "open").strip().lower()
        body_map = _mapping(body, noun="goal body")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            existing = connection.execute(
                "SELECT revision FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "goal CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO goals (
                        goal_cid, goal_alias, objective_id, parent_goal_cid,
                        ordinal, title, status, created_at, updated_at,
                        revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        gcid,
                        alias,
                        oid,
                        parent,
                        ord_value,
                        title_text,
                        status_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="goal body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if (
                    expected_revision is not None
                    and expected_revision != current_revision
                ):
                    raise IntentRepositoryConflictError("goal revision CAS is stale")
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE goals SET
                        goal_alias = ?, objective_id = ?, parent_goal_cid = ?,
                        ordinal = ?, title = ?, status = ?, updated_at = ?,
                        revision = ?, body_json = ?
                    WHERE goal_cid = ? AND revision = ?
                    """,
                    [
                        alias,
                        oid,
                        parent,
                        ord_value,
                        title_text,
                        status_text,
                        now,
                        revision,
                        _canonical(body_map, noun="goal body"),
                        gcid,
                        current_revision,
                    ],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_UPSERTED,
                subject_id=gcid,
                body={
                    "goal_cid": gcid,
                    "goal_alias": alias,
                    "objective_id": oid,
                    "parent_goal_cid": parent,
                    "ordinal": ord_value,
                    "title": title_text,
                    "status": status_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def get_goal(self, goal_cid: str) -> Mapping[str, Any] | None:
        gcid = _identifier(goal_cid, noun="goal_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT goal_cid, goal_alias, objective_id, parent_goal_cid,
                       ordinal, title, status, created_at, updated_at,
                       revision, body_json
                FROM goals WHERE goal_cid = ? OR goal_alias = ?
                LIMIT 1
                """,
                [gcid, gcid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2] or ""),
                "parent_goal_cid": str(row[3] or ""),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "created_at": str(row[7]),
                "updated_at": str(row[8]),
                "revision": int(row[9]),
                "body": _decode_json(row[10], noun="goal body"),
            }
        )

    def link_goal_edge(
        self,
        *,
        parent_goal_cid: str,
        child_goal_cid: str,
        edge_kind: str = "depends_on",
    ) -> IntentReceipt:
        parent = _identifier(parent_goal_cid, noun="parent_goal_cid")
        child = _identifier(child_goal_cid, noun="child_goal_cid")
        kind = _identifier(edge_kind, noun="edge_kind")
        if parent == child:
            raise IntentRepositoryError("goal edge cannot be reflexive")
        with self._connection(write=True) as connection:
            for gcid in (parent, child):
                row = connection.execute(
                    "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
                ).fetchone()
                if row is None:
                    raise IntentRepositoryIntegrityError(
                        f"goal {gcid!r} does not exist for edge"
                    )
            connection.execute(
                """
                DELETE FROM goal_edges
                WHERE parent_goal_cid = ? AND child_goal_cid = ? AND edge_kind = ?
                """,
                [parent, child, kind],
            )
            connection.execute(
                """
                INSERT INTO goal_edges (
                    parent_goal_cid, child_goal_cid, edge_kind
                ) VALUES (?, ?, ?)
                """,
                [parent, child, kind],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_EDGE_LINKED,
                subject_id=parent,
                body={
                    "parent_goal_cid": parent,
                    "child_goal_cid": child,
                    "edge_kind": kind,
                    "revision": 0,
                },
            )

    def reopen_goal(
        self,
        *,
        goal_cid: str,
        expected_revision: int,
        reason: str = "reopened",
    ) -> IntentReceipt:
        gcid = _identifier(goal_cid, noun="goal_cid")
        expected = _nonneg_int(expected_revision, noun="expected_revision")
        reason_text = str(reason or "reopened").strip() or "reopened"
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, status, body_json FROM goals WHERE goal_cid = ?",
                [gcid],
            ).fetchone()
            if row is None:
                raise KeyError(gcid)
            current_revision = int(row[0])
            if current_revision != expected:
                raise IntentRepositoryConflictError("goal revision CAS is stale")
            previous_status = str(row[1])
            body_map = _decode_json(row[2], noun="goal body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            body_map["reopen_reason"] = reason_text
            body_map["previous_status"] = previous_status
            revision = current_revision + 1
            connection.execute(
                """
                UPDATE goals SET status = ?, updated_at = ?, revision = ?,
                    body_json = ?
                WHERE goal_cid = ? AND revision = ?
                """,
                [
                    "reopened",
                    now,
                    revision,
                    _canonical(body_map, noun="goal body"),
                    gcid,
                    current_revision,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.GOAL_REOPENED,
                subject_id=gcid,
                body={
                    "goal_cid": gcid,
                    "previous_status": previous_status,
                    "status": "reopened",
                    "reason": reason_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    # -- plans (also exposed via PlanRevisionRepository) ---------------------

    def upsert_plan(
        self,
        *,
        plan_cid: str,
        goal_cid: str,
        plan_alias: str,
        status: str = "active",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        set_head: bool = True,
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        gcid = _identifier(goal_cid, noun="goal_cid")
        alias = _identifier(plan_alias, noun="plan_alias")
        status_text = str(status or "active").strip().lower()
        body_map = _mapping(body, noun="plan body")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            goal_row = connection.execute(
                "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if goal_row is None:
                raise IntentRepositoryIntegrityError(
                    f"goal {gcid!r} does not exist for plan"
                )
            existing = connection.execute(
                "SELECT revision FROM plans WHERE plan_cid = ?", [pcid]
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "plan CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO plans (
                        plan_cid, goal_cid, plan_alias, status, created_at,
                        updated_at, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        pcid,
                        gcid,
                        alias,
                        status_text,
                        now,
                        now,
                        revision,
                        _canonical(body_map, noun="plan body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if (
                    expected_revision is not None
                    and expected_revision != current_revision
                ):
                    raise IntentRepositoryConflictError("plan revision CAS is stale")
                revision = current_revision + 1
                connection.execute(
                    """
                    UPDATE plans SET goal_cid = ?, plan_alias = ?, status = ?,
                        updated_at = ?, revision = ?, body_json = ?
                    WHERE plan_cid = ? AND revision = ?
                    """,
                    [
                        gcid,
                        alias,
                        status_text,
                        now,
                        revision,
                        _canonical(body_map, noun="plan body"),
                        pcid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    pcid,
                    revision,
                    _canonical(body_map, noun="plan revision body"),
                    now,
                ],
            )
            if set_head and status_text == "active":
                # Demote other active heads for the same goal.
                connection.execute(
                    """
                    UPDATE plans SET status = 'superseded', updated_at = ?
                    WHERE goal_cid = ? AND plan_cid <> ? AND status = 'active'
                    """,
                    [now, gcid, pcid],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_UPSERTED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "goal_cid": gcid,
                    "plan_alias": alias,
                    "status": status_text,
                    "revision": revision,
                    "body": body_map,
                    "set_head": bool(set_head),
                    "recorded_at": now,
                },
            )

    def get_plan(self, plan_cid: str) -> Mapping[str, Any] | None:
        pcid = _identifier(plan_cid, noun="plan_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT plan_cid, goal_cid, plan_alias, status, created_at,
                       updated_at, revision, body_json
                FROM plans WHERE plan_cid = ? OR plan_alias = ?
                LIMIT 1
                """,
                [pcid, pcid],
            ).fetchone()
        if row is None:
            return None
        return MappingProxyType(
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "created_at": str(row[4]),
                "updated_at": str(row[5]),
                "revision": int(row[6]),
                "body": _decode_json(row[7], noun="plan body"),
            }
        )

    def get_plan_head(self, goal_cid: str) -> PlanHead | None:
        gcid = _identifier(goal_cid, noun="goal_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT plan_cid, goal_cid, revision, status, body_json
                FROM plans
                WHERE goal_cid = ? AND status = 'active'
                ORDER BY revision DESC, plan_cid ASC
                LIMIT 1
                """,
                [gcid],
            ).fetchone()
        if row is None:
            return None
        body = _decode_json(row[4], noun="plan body")
        body_map = body if isinstance(body, dict) else {}
        return PlanHead(
            plan_cid=str(row[0]),
            goal_cid=str(row[1]),
            revision=int(row[2]),
            status=str(row[3]),
            superseded_by=str(body_map.get("superseded_by") or ""),
            continuation_of=str(body_map.get("continuation_of") or ""),
        )

    def append_plan_revision(
        self,
        *,
        plan_cid: str,
        body: Mapping[str, Any] | None = None,
        expected_revision: int,
        delta: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        body_map = _mapping(body, noun="plan body")
        delta_map = _mapping(delta, noun="plan delta") if delta is not None else {}
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, body_json, goal_cid, plan_alias, status "
                "FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            current_revision = int(row[0])
            if current_revision != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            previous_body = _decode_json(row[1], noun="plan body")
            if not isinstance(previous_body, dict):
                previous_body = {}
            merged = dict(previous_body)
            merged.update(body_map)
            if delta_map:
                merged["last_delta"] = delta_map
            revision = current_revision + 1
            connection.execute(
                """
                UPDATE plans SET revision = ?, updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    revision,
                    now,
                    _canonical(merged, noun="plan body"),
                    pcid,
                    current_revision,
                ],
            )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [pcid, revision, _canonical(merged, noun="plan revision"), now],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_REVISION_APPENDED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "goal_cid": str(row[2]),
                    "plan_alias": str(row[3]),
                    "status": str(row[4]),
                    "revision": revision,
                    "body": merged,
                    "delta": delta_map,
                    "recorded_at": now,
                },
            )

    def supersede_plan(
        self,
        *,
        plan_cid: str,
        successor_plan_cid: str,
        expected_revision: int,
        reason: str = "superseded",
    ) -> IntentReceipt:
        pcid = _identifier(plan_cid, noun="plan_cid")
        successor = _identifier(successor_plan_cid, noun="successor_plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        reason_text = str(reason or "superseded").strip() or "superseded"
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, body_json, goal_cid FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            if int(row[0]) != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            succ = connection.execute(
                "SELECT 1 FROM plans WHERE plan_cid = ?", [successor]
            ).fetchone()
            if succ is None:
                raise IntentRepositoryIntegrityError(
                    f"successor plan {successor!r} does not exist"
                )
            body_map = _decode_json(row[1], noun="plan body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            body_map["superseded_by"] = successor
            body_map["supersede_reason"] = reason_text
            revision = expected + 1
            connection.execute(
                """
                UPDATE plans SET status = 'superseded', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    revision,
                    now,
                    _canonical(body_map, noun="plan body"),
                    pcid,
                    expected,
                ],
            )
            connection.execute(
                """
                UPDATE plans SET status = 'active', updated_at = ?
                WHERE plan_cid = ?
                """,
                [now, successor],
            )
            connection.execute(
                """
                INSERT INTO planning_decisions (
                    decision_id, plan_cid, goal_cid, decision_kind,
                    decided_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    content_identity(
                        {
                            "kind": "supersession",
                            "plan_cid": pcid,
                            "successor": successor,
                            "revision": revision,
                        }
                    ),
                    pcid,
                    str(row[2]),
                    "supersession",
                    now,
                    _canonical(
                        {
                            "predecessor": pcid,
                            "successor": successor,
                            "reason": reason_text,
                        },
                        noun="supersession decision",
                    ),
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_SUPERSEDED,
                subject_id=pcid,
                body={
                    "plan_cid": pcid,
                    "successor_plan_cid": successor,
                    "goal_cid": str(row[2]),
                    "reason": reason_text,
                    "revision": revision,
                    "body": body_map,
                    "recorded_at": now,
                },
            )

    def continue_plan(
        self,
        *,
        plan_cid: str,
        continuation_plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        """Create/activate a continuation plan bound to the predecessor head."""

        pcid = _identifier(plan_cid, noun="plan_cid")
        cont = _identifier(continuation_plan_cid, noun="continuation_plan_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        body_map = _mapping(body, noun="continuation body")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision, goal_cid, plan_alias, body_json "
                "FROM plans WHERE plan_cid = ?",
                [pcid],
            ).fetchone()
            if row is None:
                raise KeyError(pcid)
            if int(row[0]) != expected:
                raise IntentRepositoryConflictError("plan revision CAS is stale")
            gcid = str(row[1])
            cont_body = dict(body_map)
            cont_body["continuation_of"] = pcid
            cont_exists = connection.execute(
                "SELECT revision FROM plans WHERE plan_cid = ?", [cont]
            ).fetchone()
            if cont_exists is None:
                connection.execute(
                    """
                    INSERT INTO plans (
                        plan_cid, goal_cid, plan_alias, status, created_at,
                        updated_at, revision, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cont,
                        gcid,
                        f"{row[2]}-cont",
                        "active",
                        now,
                        now,
                        1,
                        _canonical(cont_body, noun="continuation plan body"),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO plan_revisions (
                        plan_cid, revision, body_json, recorded_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        cont,
                        1,
                        _canonical(cont_body, noun="continuation revision"),
                        now,
                    ],
                )
                cont_revision = 1
            else:
                cont_revision = int(cont_exists[0]) + 1
                connection.execute(
                    """
                    UPDATE plans SET status = 'active', revision = ?,
                        updated_at = ?, body_json = ?
                    WHERE plan_cid = ?
                    """,
                    [
                        cont_revision,
                        now,
                        _canonical(cont_body, noun="continuation plan body"),
                        cont,
                    ],
                )
            pred_body = _decode_json(row[3], noun="plan body")
            if not isinstance(pred_body, dict):
                pred_body = {}
            pred_body = dict(pred_body)
            pred_body["continued_by"] = cont
            connection.execute(
                """
                UPDATE plans SET status = 'continued', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ? AND revision = ?
                """,
                [
                    expected + 1,
                    now,
                    _canonical(pred_body, noun="plan body"),
                    pcid,
                    expected,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.PLAN_CONTINUED,
                subject_id=cont,
                body={
                    "plan_cid": cont,
                    "continuation_of": pcid,
                    "goal_cid": gcid,
                    "revision": cont_revision,
                    "body": cont_body,
                    "recorded_at": now,
                },
            )

    # -- tasks ---------------------------------------------------------------

    def upsert_task(
        self,
        *,
        task_cid: str,
        task_alias: str,
        goal_cid: str,
        ordinal: int = 0,
        status: str = "ready",
        priority: str = "P2",
        plan_cid: str = "",
        objective_id: str = "",
        body: Mapping[str, Any] | None = None,
        identity: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        dependencies: Sequence[str] | None = None,
        outputs: Sequence[Mapping[str, Any]] | None = None,
        acceptance: Sequence[Mapping[str, Any] | str] | None = None,
        validations: Sequence[Mapping[str, Any] | str | Sequence[str]] | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        alias = _identifier(task_alias, noun="task_alias")
        gcid = _identifier(goal_cid, noun="goal_cid")
        ord_value = _nonneg_int(ordinal, noun="ordinal")
        status_text = _status(status, allowed=_TASK_STATUSES, noun="task")
        priority_text = str(priority or "P2").strip() or "P2"
        pcid = _optional_identifier(plan_cid, noun="plan_cid")
        oid = _optional_identifier(objective_id, noun="objective_id")
        body_map = _mapping(body, noun="task body")
        identity_map = _mapping(identity, noun="task identity")
        # Identity material is canonical and must not include mutable aliases
        # as keys; always bind the durable task_cid.
        identity_map = {
            **identity_map,
            "task_cid": tcid,
            "task_alias": alias,
        }
        now = _utc_iso()

        with self._connection(write=True) as connection:
            goal_row = connection.execute(
                "SELECT 1 FROM goals WHERE goal_cid = ?", [gcid]
            ).fetchone()
            if goal_row is None:
                raise IntentRepositoryIntegrityError(
                    f"goal {gcid!r} does not exist for task"
                )
            existing = connection.execute(
                "SELECT revision, status FROM tasks WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if existing is None:
                if expected_revision is not None and expected_revision != 0:
                    raise IntentRepositoryConflictError(
                        "task CAS expected revision does not match create"
                    )
                revision = 1
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_cid, objective_id,
                        ordinal, status, revision, priority, created_at,
                        updated_at, identity_json, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        alias,
                        gcid,
                        pcid,
                        oid,
                        ord_value,
                        status_text,
                        revision,
                        priority_text,
                        now,
                        now,
                        _canonical(identity_map, noun="task identity"),
                        _canonical(body_map, noun="task body"),
                    ],
                )
            else:
                current_revision = int(existing[0])
                if (
                    expected_revision is not None
                    and expected_revision != current_revision
                ):
                    raise IntentRepositoryConflictError("task revision CAS is stale")
                revision = current_revision + 1
                # Canonical task_cid is immutable; alias/goal/plan may update.
                connection.execute(
                    """
                    UPDATE tasks SET
                        task_alias = ?, goal_cid = ?, plan_cid = ?,
                        objective_id = ?, ordinal = ?, status = ?,
                        revision = ?, priority = ?, updated_at = ?,
                        identity_json = ?, body_json = ?
                    WHERE task_cid = ? AND revision = ?
                    """,
                    [
                        alias,
                        gcid,
                        pcid,
                        oid,
                        ord_value,
                        status_text,
                        revision,
                        priority_text,
                        now,
                        _canonical(identity_map, noun="task identity"),
                        _canonical(body_map, noun="task body"),
                        tcid,
                        current_revision,
                    ],
                )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="task revision body"),
                    now,
                ],
            )
            resolved_dependencies: list[str] = []
            if dependencies is not None:
                self._set_dependencies_on(connection, tcid, dependencies)
                resolved_dependencies = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT dependency_task_cid FROM task_dependencies "
                        "WHERE task_cid = ? ORDER BY dependency_task_cid",
                        [tcid],
                    ).fetchall()
                ]
            if outputs is not None:
                self._set_outputs_on(connection, tcid, outputs)
            if acceptance is not None:
                self._set_acceptance_on(connection, tcid, acceptance)
            if validations is not None:
                self._set_validations_on(connection, tcid, validations)
            event_body: dict[str, Any] = {
                "task_cid": tcid,
                "task_alias": alias,
                "goal_cid": gcid,
                "plan_cid": pcid,
                "objective_id": oid,
                "ordinal": ord_value,
                "status": status_text,
                "priority": priority_text,
                "revision": revision,
                "identity": identity_map,
                "body": body_map,
                "recorded_at": now,
            }
            # Only emit relation fields that were explicitly provided so rebuild
            # does not wipe prior edges when an upsert omits them.
            if dependencies is not None:
                event_body["dependencies"] = resolved_dependencies
            if outputs is not None:
                event_body["outputs"] = [
                    dict(item) if isinstance(item, Mapping) else item
                    for item in outputs
                ]
            if acceptance is not None:
                event_body["acceptance"] = [
                    dict(item) if isinstance(item, Mapping) else item
                    for item in acceptance
                ]
            if validations is not None:
                event_body["validations"] = [
                    list(item)
                    if isinstance(item, Sequence)
                    and not isinstance(item, (str, Mapping, bytes, bytearray))
                    else (dict(item) if isinstance(item, Mapping) else item)
                    for item in validations
                ]
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_UPSERTED,
                subject_id=tcid,
                task_cid=tcid,
                body=event_body,
            )

    def _set_dependencies_on(
        self, connection: Any, task_cid: str, dependencies: Sequence[str]
    ) -> None:
        if len(dependencies) > MAX_DEPENDENCIES:
            raise IntentRepositoryBoundsError("dependency count exceeds bound")
        connection.execute(
            "DELETE FROM task_dependencies WHERE task_cid = ?", [task_cid]
        )
        seen: set[str] = set()
        for raw in dependencies:
            dep = _identifier(raw, noun="dependency_task_cid")
            # Prefer durable CID when the dependency was referenced by alias.
            resolved = connection.execute(
                "SELECT task_cid FROM tasks "
                "WHERE task_cid = ? OR task_alias = ? "
                "ORDER BY task_cid LIMIT 1",
                [dep, dep],
            ).fetchone()
            if resolved is not None:
                dep = str(resolved[0])
            if dep == task_cid:
                raise IntentRepositoryError("task cannot depend on itself")
            if dep in seen:
                continue
            seen.add(dep)
            connection.execute(
                """
                INSERT INTO task_dependencies (
                    task_cid, dependency_task_cid, kind
                ) VALUES (?, ?, ?)
                """,
                [task_cid, dep, "depends_on"],
            )

    def _set_outputs_on(
        self, connection: Any, task_cid: str, outputs: Sequence[Mapping[str, Any]]
    ) -> None:
        if len(outputs) > MAX_OUTPUTS:
            raise IntentRepositoryBoundsError("output count exceeds bound")
        connection.execute(
            "DELETE FROM task_outputs WHERE task_cid = ?", [task_cid]
        )
        for ordinal, item in enumerate(outputs):
            mapping = _mapping(item, noun="task output")
            path = _identifier(
                mapping.get("path") or mapping.get("effect_id") or f"output:{ordinal}",
                noun="output path",
            )
            connection.execute(
                """
                INSERT INTO task_outputs (
                    task_cid, ordinal, path, effect_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    path,
                    _canonical(mapping, noun="output effect"),
                ],
            )

    def _set_acceptance_on(
        self,
        connection: Any,
        task_cid: str,
        acceptance: Sequence[Mapping[str, Any] | str],
    ) -> None:
        if len(acceptance) > MAX_ACCEPTANCE:
            raise IntentRepositoryBoundsError("acceptance count exceeds bound")
        connection.execute(
            "DELETE FROM task_acceptance WHERE task_cid = ?", [task_cid]
        )
        for ordinal, item in enumerate(acceptance):
            if isinstance(item, str):
                criterion = item.strip()
                policy: dict[str, Any] = {"criterion": criterion}
            else:
                mapping = _mapping(item, noun="acceptance")
                criterion = str(
                    mapping.get("criterion")
                    or mapping.get("statement")
                    or mapping.get("criterion_key")
                    or f"criterion:{ordinal}"
                ).strip()
                policy = dict(mapping)
            if not criterion:
                raise IntentRepositoryError("acceptance criterion must not be empty")
            connection.execute(
                """
                INSERT INTO task_acceptance (
                    task_cid, ordinal, criterion, evidence_policy_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    criterion,
                    _canonical(policy, noun="acceptance policy"),
                ],
            )

    def _set_validations_on(
        self,
        connection: Any,
        task_cid: str,
        validations: Sequence[Mapping[str, Any] | str | Sequence[str]],
    ) -> None:
        if len(validations) > MAX_VALIDATIONS:
            raise IntentRepositoryBoundsError("validation count exceeds bound")
        connection.execute(
            "DELETE FROM task_validations WHERE task_cid = ?", [task_cid]
        )
        for ordinal, item in enumerate(validations):
            if isinstance(item, str):
                argv = [item]
                policy: dict[str, Any] = {}
            elif isinstance(item, Mapping):
                mapping = _mapping(item, noun="validation")
                raw_argv = mapping.get("argv") or mapping.get("validation_commands")
                if isinstance(raw_argv, str):
                    argv = [raw_argv]
                elif isinstance(raw_argv, Sequence):
                    argv = [str(part) for part in raw_argv]
                else:
                    argv = [str(mapping.get("command") or f"validation:{ordinal}")]
                policy = {
                    key: value
                    for key, value in mapping.items()
                    if key not in {"argv", "validation_commands", "command"}
                }
            elif isinstance(item, Sequence):
                argv = [str(part) for part in item]
                policy = {}
            else:
                raise IntentRepositoryError("validation entry has unsupported type")
            connection.execute(
                """
                INSERT INTO task_validations (
                    task_cid, ordinal, argv_json, policy_json
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    task_cid,
                    ordinal,
                    _canonical(list(argv), noun="validation argv"),
                    _canonical(policy, noun="validation policy"),
                ],
            )

    def set_task_dependencies(
        self, task_cid: str, dependencies: Sequence[str]
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT revision FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if row is None:
                raise KeyError(tcid)
            self._set_dependencies_on(connection, tcid, dependencies)
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_DEPENDENCIES_SET,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "dependencies": [
                        _identifier(item, noun="dependency_task_cid")
                        for item in dependencies
                    ],
                    "revision": int(row[0]),
                },
            )

    def get_task(self, task_cid_or_alias: str) -> Mapping[str, Any] | None:
        key = _identifier(task_cid_or_alias, noun="task_cid")
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                       ordinal, status, revision, priority, created_at,
                       updated_at, identity_json, body_json
                FROM tasks
                WHERE task_cid = ? OR task_alias = ?
                ORDER BY task_cid
                LIMIT 2
                """,
                [key, key],
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise IntentRepositoryIntegrityError(
                    "task CID/alias lookup is ambiguous"
                )
            row = rows[0]
            tcid = str(row[0])
            deps = [
                str(item[0])
                for item in connection.execute(
                    "SELECT dependency_task_cid FROM task_dependencies "
                    "WHERE task_cid = ? ORDER BY dependency_task_cid",
                    [tcid],
                ).fetchall()
            ]
            outputs = [
                {
                    "ordinal": int(item[0]),
                    "path": str(item[1]),
                    "effect": _decode_json(item[2], noun="output effect"),
                }
                for item in connection.execute(
                    "SELECT ordinal, path, effect_json FROM task_outputs "
                    "WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
            acceptance = [
                {
                    "ordinal": int(item[0]),
                    "criterion": str(item[1]),
                    "evidence_policy": _decode_json(
                        item[2], noun="acceptance policy"
                    ),
                }
                for item in connection.execute(
                    "SELECT ordinal, criterion, evidence_policy_json "
                    "FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
            validations = [
                {
                    "ordinal": int(item[0]),
                    "argv": _decode_json(item[1], noun="validation argv"),
                    "policy": _decode_json(item[2], noun="validation policy"),
                }
                for item in connection.execute(
                    "SELECT ordinal, argv_json, policy_json "
                    "FROM task_validations WHERE task_cid = ? ORDER BY ordinal",
                    [tcid],
                ).fetchall()
            ]
        return MappingProxyType(
            {
                "task_cid": tcid,
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3] or ""),
                "objective_id": str(row[4] or ""),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8] or ""),
                "created_at": str(row[9] or ""),
                "updated_at": str(row[10] or ""),
                "identity": _decode_json(row[11], noun="task identity"),
                "body": _decode_json(row[12], noun="task body"),
                "dependencies": tuple(deps),
                "outputs": tuple(outputs),
                "acceptance": tuple(acceptance),
                "validations": tuple(validations),
            }
        )

    def list_tasks(
        self,
        *,
        status: str | Iterable[str] | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[Mapping[str, Any], ...]:
        selected = _bounded_limit(limit)
        off = _nonneg_int(offset, noun="offset")
        statuses: tuple[str, ...]
        if status is None:
            statuses = ()
        elif isinstance(status, str):
            statuses = (_status(status, allowed=_TASK_STATUSES, noun="task"),)
        else:
            statuses = tuple(
                sorted(
                    {
                        _status(item, allowed=_TASK_STATUSES, noun="task")
                        for item in status
                    }
                )
            )
        with self._connection(write=False) as connection:
            if statuses:
                placeholders = ", ".join("?" for _ in statuses)
                rows = connection.execute(
                    f"""
                    SELECT task_cid FROM tasks
                    WHERE status IN ({placeholders})
                    ORDER BY ordinal, task_cid
                    LIMIT ? OFFSET ?
                    """,
                    [*statuses, selected, off],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT task_cid FROM tasks
                    ORDER BY ordinal, task_cid
                    LIMIT ? OFFSET ?
                    """,
                    [selected, off],
                ).fetchall()
        results: list[Mapping[str, Any]] = []
        for row in rows:
            task = self.get_task(str(row[0]))
            if task is not None:
                results.append(task)
        return tuple(results)

    # -- evidence / completion -----------------------------------------------

    def record_evidence(
        self,
        *,
        task_cid: str,
        evidence_kind: str,
        digest: str,
        body: Mapping[str, Any] | None = None,
        evidence_id: str = "",
        parent_evidence_id: str = "",
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        kind = _identifier(evidence_kind, noun="evidence_kind")
        digest_text = _identifier(digest, noun="digest")
        body_map = _mapping(body, noun="evidence body")
        eid = (
            _identifier(evidence_id, noun="evidence_id")
            if evidence_id
            else content_identity(
                {
                    "task_cid": tcid,
                    "evidence_kind": kind,
                    "digest": digest_text,
                    "body": body_map,
                }
            )
        )
        parent = _optional_identifier(parent_evidence_id, noun="parent_evidence_id")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            count = connection.execute(
                "SELECT COUNT(*) FROM evidence_nodes WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if count and int(count[0]) >= MAX_EVIDENCE:
                raise IntentRepositoryBoundsError("evidence population exceeds bound")
            connection.execute(
                "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                [eid],
            )
            connection.execute(
                """
                INSERT INTO evidence_nodes (
                    evidence_id, parent_evidence_id, task_cid, evidence_kind,
                    digest, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    eid,
                    parent,
                    tcid,
                    kind,
                    digest_text,
                    now,
                    _canonical(body_map, noun="evidence body"),
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.EVIDENCE_RECORDED,
                subject_id=eid,
                task_cid=tcid,
                body={
                    "evidence_id": eid,
                    "parent_evidence_id": parent,
                    "task_cid": tcid,
                    "evidence_kind": kind,
                    "digest": digest_text,
                    "body": body_map,
                    "created_at": now,
                    "revision": 0,
                },
            )

    def record_validation_result(
        self,
        *,
        task_cid: str,
        outcome: str,
        evidence_digest: str,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        outcome_text = str(outcome or "").strip().lower()
        if outcome_text not in {"passed", "failed", "error", "skipped"}:
            raise IntentRepositoryError(
                f"validation outcome {outcome!r} is not in the closed set"
            )
        digest = _identifier(evidence_digest, noun="evidence_digest")
        body_map = _mapping(body, noun="validation body")
        argv_list = [str(item) for item in (argv or ())]
        now = _utc_iso()
        run_id = content_identity(
            {
                "task_cid": tcid,
                "attempt_id": attempt_id,
                "argv": argv_list,
                "recorded_at": now,
            }
        )
        result_id = content_identity(
            {
                "run_id": run_id,
                "outcome": outcome_text,
                "evidence_digest": digest,
            }
        )
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            connection.execute(
                """
                INSERT INTO validation_runs (
                    run_id, task_cid, attempt_id, started_at, finished_at,
                    status, command_digest, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    tcid,
                    attempt_id or "",
                    now,
                    now,
                    outcome_text,
                    content_identity({"argv": argv_list}),
                    _canonical(
                        {"argv": argv_list, **body_map},
                        noun="validation run body",
                    ),
                ],
            )
            connection.execute(
                """
                INSERT INTO validation_results (
                    result_id, run_id, task_cid, ordinal, outcome,
                    evidence_digest, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result_id,
                    run_id,
                    tcid,
                    0,
                    outcome_text,
                    digest,
                    _canonical(body_map, noun="validation result body"),
                ],
            )
            # Mirror a current evidence node so completion can join on digests.
            if outcome_text == "passed":
                evidence_id = content_identity(
                    {
                        "task_cid": tcid,
                        "evidence_kind": "validation",
                        "digest": digest,
                        "run_id": run_id,
                    }
                )
                connection.execute(
                    "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                    [evidence_id],
                )
                connection.execute(
                    """
                    INSERT INTO evidence_nodes (
                        evidence_id, parent_evidence_id, task_cid, evidence_kind,
                        digest, created_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evidence_id,
                        "",
                        tcid,
                        "validation",
                        digest,
                        now,
                        _canonical(
                            {
                                "run_id": run_id,
                                "result_id": result_id,
                                "argv": argv_list,
                                "outcome": outcome_text,
                            },
                            noun="validation evidence body",
                        ),
                    ],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.VALIDATION_RECORDED,
                subject_id=result_id,
                task_cid=tcid,
                attempt_id=attempt_id or "",
                body={
                    "result_id": result_id,
                    "run_id": run_id,
                    "task_cid": tcid,
                    "outcome": outcome_text,
                    "evidence_digest": digest,
                    "argv": argv_list,
                    "body": body_map,
                    "recorded_at": now,
                    "revision": 0,
                },
            )

    def current_evidence_for_task(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        tcid = _identifier(task_cid, noun="task_cid")
        clock = int(now_ms if now_ms is not None else self._clock_ms())
        freshness_ms = self.evidence_freshness_seconds * 1000
        with self._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT evidence_id, parent_evidence_id, task_cid, evidence_kind,
                       digest, created_at, body_json
                FROM evidence_nodes
                WHERE task_cid = ?
                ORDER BY created_at DESC, evidence_id ASC
                """,
                [tcid],
            ).fetchall()
        current: list[Mapping[str, Any]] = []
        for row in rows:
            created_at = str(row[5] or "")
            created_ms = _parse_iso_ms(created_at)
            if freshness_ms > 0 and created_ms > 0:
                if clock - created_ms > freshness_ms:
                    continue
            current.append(
                MappingProxyType(
                    {
                        "evidence_id": str(row[0]),
                        "parent_evidence_id": str(row[1] or ""),
                        "task_cid": str(row[2] or ""),
                        "evidence_kind": str(row[3]),
                        "digest": str(row[4]),
                        "created_at": created_at,
                        "body": _decode_json(row[6], noun="evidence body"),
                    }
                )
            )
        return tuple(current)

    def required_evidence_satisfied(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return whether every acceptance criterion has current evidence."""

        task = self.get_task(task_cid)
        if task is None:
            raise KeyError(task_cid)
        acceptance = list(task.get("acceptance") or ())
        if not acceptance:
            # No declared acceptance: completion still requires at least one
            # current validation evidence node (fail-closed for empty claims).
            current = self.current_evidence_for_task(task_cid, now_ms=now_ms)
            validation = [
                item
                for item in current
                if str(item.get("evidence_kind") or "")
                in {"validation", "test", "acceptance"}
            ]
            if validation:
                return True, ()
            return False, ("required:current_validation_evidence",)

        current = self.current_evidence_for_task(task_cid, now_ms=now_ms)
        digests = {
            str(item.get("digest") or "")
            for item in current
            if item.get("digest")
        }
        kinds = {
            str(item.get("evidence_kind") or "")
            for item in current
        }
        missing: list[str] = []
        for item in acceptance:
            if not isinstance(item, Mapping):
                continue
            policy = item.get("evidence_policy") or {}
            if not isinstance(policy, Mapping):
                policy = {}
            criterion = str(item.get("criterion") or "")
            required_digest = str(
                policy.get("required_digest")
                or policy.get("evidence_digest")
                or policy.get("digest")
                or ""
            ).strip()
            required_kind = str(
                policy.get("evidence_kind")
                or policy.get("kind")
                or ""
            ).strip()
            if required_digest:
                if required_digest not in digests:
                    missing.append(f"digest:{required_digest}")
                continue
            if required_kind:
                if required_kind not in kinds:
                    missing.append(f"kind:{required_kind}")
                continue
            # Default: any current evidence satisfies the criterion when no
            # explicit digest/kind is declared, but evidence must exist.
            if not current:
                missing.append(f"criterion:{criterion or item.get('ordinal')}")
        return (not missing), tuple(missing)

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_revision: int,
        new_status: str,
        receipt: Mapping[str, Any] | None = None,
        evidence_digests: Sequence[str] | None = None,
        allow_completion_without_evidence: bool = False,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        expected = _positive_int(expected_revision, noun="expected_revision")
        status_text = _status(new_status, allowed=_TASK_STATUSES, noun="task")
        receipt_map = _mapping(receipt, noun="status receipt")
        now = _utc_iso()

        with self._connection(write=True) as connection:
            row = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, status, revision, body_json
                FROM tasks WHERE task_cid = ? OR task_alias = ?
                ORDER BY task_cid LIMIT 2
                """,
                [tcid, tcid],
            ).fetchall()
            if not row:
                raise KeyError(tcid)
            if len(row) > 1:
                raise IntentRepositoryIntegrityError(
                    "task CID/alias lookup is ambiguous"
                )
            task_row = row[0]
            resolved_cid = str(task_row[0])
            previous_status = str(task_row[3])
            current_revision = int(task_row[4])
            if current_revision != expected:
                raise IntentRepositoryConflictError("task revision CAS is stale")
            if previous_status == status_text:
                return IntentReceipt(
                    event_id="",
                    event_type=IntentEventType.TASK_STATUS_CHANGED.value,
                    global_sequence=self._next_global_sequence(connection) - 1,
                    recorded_at=now,
                    subject_id=resolved_cid,
                    revision=current_revision,
                    changed=False,
                    details=MappingProxyType(
                        {
                            "task_cid": resolved_cid,
                            "status": status_text,
                            "previous_status": previous_status,
                        }
                    ),
                )

            completing = status_text in _COMPLETED_STATUSES
            if completing and not allow_completion_without_evidence:
                # Gate completion on current required evidence inside the same
                # transaction that mutates status.
                missing = self._missing_evidence_on(
                    connection,
                    resolved_cid,
                    evidence_digests=evidence_digests,
                )
                if missing:
                    raise IntentCompletionError(
                        "completion refused without current required evidence: "
                        + ", ".join(missing)
                    )

            revision = current_revision + 1
            body_map = _decode_json(task_row[5], noun="task body")
            if not isinstance(body_map, dict):
                body_map = {}
            body_map = dict(body_map)
            if receipt_map:
                body_map["completion_receipt"] = receipt_map
            connection.execute(
                """
                UPDATE tasks SET status = ?, revision = ?, updated_at = ?,
                    body_json = ?
                WHERE task_cid = ? AND revision = ?
                """,
                [
                    status_text,
                    revision,
                    now,
                    _canonical(body_map, noun="task body"),
                    resolved_cid,
                    current_revision,
                ],
            )
            connection.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    resolved_cid,
                    revision,
                    status_text,
                    _canonical(body_map, noun="task revision body"),
                    now,
                ],
            )
            event_body: dict[str, Any] = {
                "task_cid": resolved_cid,
                "task_alias": str(task_row[1]),
                "goal_cid": str(task_row[2]),
                "previous_status": previous_status,
                "status": status_text,
                "revision": revision,
                "receipt": receipt_map,
                "recorded_at": now,
            }
            if completing:
                evidence_digest = content_identity(
                    {
                        "task_cid": resolved_cid,
                        "revision": revision,
                        "receipt": receipt_map,
                        "evidence_digests": list(evidence_digests or ()),
                    }
                )
                receipt_cid = content_identity(
                    {
                        "namespace": "completion-receipt",
                        "task_cid": resolved_cid,
                        "revision": revision,
                        "evidence_digest": evidence_digest,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO completion_receipts (
                        receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                        fencing_token, completed_at, validation_run_id,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt_cid,
                        resolved_cid,
                        str(task_row[2]),
                        "",
                        "",
                        0,
                        now,
                        "",
                        evidence_digest,
                        _canonical(
                            {
                                "schema": COMPLETION_EVIDENCE_SCHEMA,
                                "receipt": receipt_map,
                                "evidence_digests": list(evidence_digests or ()),
                                "revision": revision,
                            },
                            noun="completion receipt",
                        ),
                    ],
                )
                event_body["completion_receipt_cid"] = receipt_cid
                event_body["evidence_digest"] = evidence_digest
                return self._append_event(
                    connection,
                    event_type=IntentEventType.COMPLETION_RECORDED,
                    subject_id=resolved_cid,
                    task_cid=resolved_cid,
                    body=event_body,
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_STATUS_CHANGED,
                subject_id=resolved_cid,
                task_cid=resolved_cid,
                body=event_body,
            )

    def _missing_evidence_on(
        self,
        connection: Any,
        task_cid: str,
        *,
        evidence_digests: Sequence[str] | None = None,
        now_ms: int | None = None,
    ) -> tuple[str, ...]:
        clock = int(now_ms if now_ms is not None else self._clock_ms())
        freshness_ms = self.evidence_freshness_seconds * 1000
        acceptance_rows = connection.execute(
            """
            SELECT ordinal, criterion, evidence_policy_json
            FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal
            """,
            [task_cid],
        ).fetchall()
        evidence_rows = connection.execute(
            """
            SELECT evidence_kind, digest, created_at
            FROM evidence_nodes WHERE task_cid = ?
            """,
            [task_cid],
        ).fetchall()
        current_digests: set[str] = set()
        current_kinds: set[str] = set()
        # DuckDBRow iterates column names (Mapping protocol); always index values.
        for row in evidence_rows:
            kind = str(row[0])
            digest = str(row[1])
            created_at = str(row[2] or "")
            created_ms = _parse_iso_ms(created_at)
            if freshness_ms > 0 and created_ms > 0 and clock - created_ms > freshness_ms:
                continue
            current_digests.add(digest)
            current_kinds.add(kind)
        # Caller-supplied digests are advisory cross-checks only; completion
        # authority comes from current stored evidence nodes, never invented
        # digests that are not already recorded against the task.
        if evidence_digests:
            provided = {
                _identifier(item, noun="evidence_digest") for item in evidence_digests
            }
            if not provided.issubset(current_digests):
                return tuple(
                    f"digest:{digest}"
                    for digest in sorted(provided - current_digests)
                )
        missing: list[str] = []
        if not acceptance_rows:
            if not current_digests:
                missing.append("required:current_validation_evidence")
            return tuple(missing)
        for row in acceptance_rows:
            ordinal = row[0]
            criterion = row[1]
            policy_json = row[2]
            policy = _decode_json(policy_json, noun="acceptance policy")
            if not isinstance(policy, dict):
                policy = {}
            required_digest = str(
                policy.get("required_digest")
                or policy.get("evidence_digest")
                or policy.get("digest")
                or ""
            ).strip()
            required_kind = str(
                policy.get("evidence_kind") or policy.get("kind") or ""
            ).strip()
            if required_digest:
                if required_digest not in current_digests:
                    missing.append(f"digest:{required_digest}")
                continue
            if required_kind:
                if required_kind not in current_kinds:
                    missing.append(f"kind:{required_kind}")
                continue
            if not current_digests:
                missing.append(f"criterion:{criterion or ordinal}")
        return tuple(missing)

    # -- queue / attempts / blocks -------------------------------------------

    def record_queue_backoff(
        self,
        *,
        task_cid: str,
        delay_ms: int,
        reason: str = "backoff",
        selection_penalty: int = 0,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        delay = _nonneg_int(delay_ms, noun="delay_ms")
        reason_text = str(reason or "backoff").strip() or "backoff"
        penalty = _nonneg_int(selection_penalty, noun="selection_penalty")
        now_ms = int(self._clock_ms())
        retry_not_before = now_ms + delay
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            lease = connection.execute(
                "SELECT attempt, fencing_token FROM leases WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if lease is None:
                attempt = 1
                connection.execute(
                    """
                    INSERT INTO leases (
                        task_cid, claim_cid, resolution_cid, claimant_did,
                        logical_epoch, fencing_token, expires_at_ms, attempt,
                        state, started_at_ms, release_reason, retry_not_before_ms,
                        owner_session_id, fence_epoch, revision, extension_schema,
                        extension_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        f"claim:queue:{tcid}",
                        f"resolution:queue:{tcid}",
                        self.owner_id,
                        1,
                        1,
                        0,
                        attempt,
                        "released",
                        now_ms,
                        reason_text,
                        retry_not_before,
                        self.session_id,
                        1,
                        1,
                        QUEUE_ENTRY_SCHEMA,
                        _canonical(
                            {
                                "selection_penalty": penalty,
                                "consecutive_failures": 1,
                                "reason": reason_text,
                            },
                            noun="queue extension",
                        ),
                    ],
                )
            else:
                attempt = int(lease[0]) + 1
                connection.execute(
                    """
                    UPDATE leases SET
                        attempt = ?, retry_not_before_ms = ?,
                        release_reason = ?, state = 'released',
                        extension_schema = ?, extension_json = ?,
                        revision = revision + 1
                    WHERE task_cid = ?
                    """,
                    [
                        attempt,
                        retry_not_before,
                        reason_text,
                        QUEUE_ENTRY_SCHEMA,
                        _canonical(
                            {
                                "selection_penalty": penalty,
                                "consecutive_failures": attempt,
                                "reason": reason_text,
                            },
                            noun="queue extension",
                        ),
                        tcid,
                    ],
                )
            return self._append_event(
                connection,
                event_type=IntentEventType.QUEUE_BACKOFF,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "attempt": attempt,
                    "retry_not_before_ms": retry_not_before,
                    "delay_ms": delay,
                    "selection_penalty": penalty,
                    "reason": reason_text,
                    "revision": attempt,
                },
            )

    def record_queue_retry(self, *, task_cid: str) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=True) as connection:
            lease = connection.execute(
                "SELECT attempt FROM leases WHERE task_cid = ?", [tcid]
            ).fetchone()
            if lease is None:
                raise KeyError(tcid)
            connection.execute(
                """
                UPDATE leases SET
                    retry_not_before_ms = 0,
                    release_reason = '',
                    extension_json = ?,
                    revision = revision + 1
                WHERE task_cid = ?
                """,
                [
                    _canonical(
                        {
                            "selection_penalty": 0,
                            "consecutive_failures": 0,
                            "reason": "",
                        },
                        noun="queue extension",
                    ),
                    tcid,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.QUEUE_RETRY,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "attempt": int(lease[0]),
                    "retry_not_before_ms": 0,
                    "revision": int(lease[0]),
                },
            )

    def get_queue_entry(self, task_cid: str) -> QueueEntry | None:
        tcid = _identifier(task_cid, noun="task_cid")
        with self._connection(write=False) as connection:
            row = connection.execute(
                """
                SELECT task_cid, attempt, retry_not_before_ms, state,
                       release_reason, extension_json
                FROM leases WHERE task_cid = ?
                """,
                [tcid],
            ).fetchone()
        if row is None:
            return None
        extension = _decode_json(row[5], noun="queue extension")
        if not isinstance(extension, dict):
            extension = {}
        return QueueEntry(
            task_cid=str(row[0]),
            attempt=int(row[1] or 0),
            retry_not_before_ms=int(row[2] or 0),
            selection_penalty=int(extension.get("selection_penalty") or 0),
            consecutive_failures=int(extension.get("consecutive_failures") or 0),
            state=str(row[3] or "released"),
            reason=str(row[4] or extension.get("reason") or ""),
        )

    def record_attempt(
        self,
        *,
        task_cid: str,
        status: str = "started",
        owner_session_id: str = "",
        fencing_token: int = 1,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        status_text = str(status or "started").strip().lower()
        owner = _optional_identifier(owner_session_id, noun="owner_session_id") or self.session_id
        fence = _positive_int(fencing_token, noun="fencing_token")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) FROM task_attempts "
                "WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            attempt_number = int(row[0] if row else 0) + 1
            attempt_id = content_identity(
                {
                    "task_cid": tcid,
                    "attempt_number": attempt_number,
                    "owner_session_id": owner,
                }
            )
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_cid, attempt_number, owner_session_id,
                    fencing_token, fence_epoch, started_at, finished_at,
                    status, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt_id,
                    tcid,
                    attempt_number,
                    owner,
                    fence,
                    1,
                    now,
                    "",
                    status_text,
                    1,
                ],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.ATTEMPT_RECORDED,
                subject_id=attempt_id,
                task_cid=tcid,
                attempt_id=attempt_id,
                body={
                    "attempt_id": attempt_id,
                    "task_cid": tcid,
                    "attempt_number": attempt_number,
                    "owner_session_id": owner,
                    "fencing_token": fence,
                    "status": status_text,
                    "revision": 1,
                    "started_at": now,
                },
            )

    def block_task(
        self,
        *,
        task_cid: str,
        blocker_kind: str,
        blocker_id: str,
        reason: str,
    ) -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        kind = _identifier(blocker_kind, noun="blocker_kind")
        bid = _identifier(blocker_id, noun="blocker_id")
        reason_text = str(reason or "").strip() or "blocked"
        now = _utc_iso()
        block_id = content_identity(
            {
                "task_cid": tcid,
                "blocker_kind": kind,
                "blocker_id": bid,
                "reason": reason_text,
                "created_at": now,
            }
        )
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT revision FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            connection.execute(
                """
                INSERT INTO task_blocks (
                    block_id, task_cid, blocker_kind, blocker_id, reason,
                    created_at, cleared_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [block_id, tcid, kind, bid, reason_text, now, "", "active"],
            )
            current_revision = int(task_row[0])
            connection.execute(
                """
                UPDATE tasks SET status = 'blocked', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [current_revision + 1, now, tcid],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_BLOCKED,
                subject_id=block_id,
                task_cid=tcid,
                body={
                    "block_id": block_id,
                    "task_cid": tcid,
                    "blocker_kind": kind,
                    "blocker_id": bid,
                    "reason": reason_text,
                    "revision": current_revision + 1,
                    "created_at": now,
                },
            )

    def unblock_task(self, *, task_cid: str, block_id: str = "") -> IntentReceipt:
        tcid = _identifier(task_cid, noun="task_cid")
        now = _utc_iso()
        with self._connection(write=True) as connection:
            task_row = connection.execute(
                "SELECT revision, status FROM tasks WHERE task_cid = ?",
                [tcid],
            ).fetchone()
            if task_row is None:
                raise KeyError(tcid)
            if block_id:
                bid = _identifier(block_id, noun="block_id")
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE block_id = ? AND task_cid = ?
                    """,
                    [now, bid, tcid],
                )
            else:
                connection.execute(
                    """
                    UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                    WHERE task_cid = ? AND state = 'active'
                    """,
                    [now, tcid],
                )
            revision = int(task_row[0]) + 1
            connection.execute(
                """
                UPDATE tasks SET status = 'ready', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [revision, now, tcid],
            )
            return self._append_event(
                connection,
                event_type=IntentEventType.TASK_UNBLOCKED,
                subject_id=tcid,
                task_cid=tcid,
                body={
                    "task_cid": tcid,
                    "block_id": block_id or "",
                    "revision": revision,
                    "cleared_at": now,
                },
            )

    # -- readiness / selection -----------------------------------------------

    def select_ready_tasks(
        self,
        *,
        limit: int = DEFAULT_PAGE_LIMIT,
        now_ms: int | None = None,
        include_completion_candidates: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return dependency-ready tasks that are not cooling down or blocked.

        Completed tasks are never selected. Tasks that *would* complete still
        require current evidence when ``include_completion_candidates`` is used
        by higher layers; this selector itself only returns non-terminal ready
        work.
        """

        selected = _bounded_limit(limit)
        clock = int(now_ms if now_ms is not None else self._clock_ms())
        _ = include_completion_candidates  # reserved for future selection modes
        with self._connection(write=False) as connection:
            task_rows = connection.execute(
                """
                SELECT task_cid, task_alias, goal_cid, ordinal, status, revision
                FROM tasks
                ORDER BY ordinal, task_cid
                """
            ).fetchall()
            dep_rows = connection.execute(
                "SELECT task_cid, dependency_task_cid FROM task_dependencies"
            ).fetchall()
            lease_rows = connection.execute(
                "SELECT task_cid, retry_not_before_ms FROM leases"
            ).fetchall()
            active_blocks = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT task_cid FROM task_blocks WHERE state = 'active'"
                ).fetchall()
            }
            completed = {
                str(row[0])
                for row in connection.execute(
                    "SELECT task_cid FROM tasks WHERE status IN ("
                    + ", ".join("?" for _ in _COMPLETED_STATUSES)
                    + ")",
                    list(_COMPLETED_STATUSES),
                ).fetchall()
            }
        dependencies: dict[str, set[str]] = {}
        # DuckDBRow is a Mapping: iterate rows and index columns, never unpack.
        for row in dep_rows:
            dependencies.setdefault(str(row[0]), set()).add(str(row[1]))
        cooldown = {
            str(row[0]): int(row[1] or 0)
            for row in lease_rows
        }
        ready: list[Mapping[str, Any]] = []
        for row in task_rows:
            tcid = str(row[0])
            alias = str(row[1])
            goal_cid = str(row[2])
            ordinal = int(row[3])
            status = str(row[4])
            revision = int(row[5])
            if status not in _READY_STATUSES:
                continue
            if tcid in active_blocks:
                continue
            if cooldown.get(tcid, 0) > clock:
                continue
            deps = dependencies.get(tcid, set())
            if not deps.issubset(completed):
                continue
            ready.append(
                MappingProxyType(
                    {
                        "task_cid": tcid,
                        "task_alias": alias,
                        "goal_cid": goal_cid,
                        "ordinal": ordinal,
                        "status": status,
                        "revision": revision,
                        "dependencies": tuple(sorted(deps)),
                    }
                )
            )
            if len(ready) >= selected:
                break
        return tuple(ready)

    # -- recovery / rebuild --------------------------------------------------

    def recover(self) -> IntentReceipt:
        """Recover intent projections from admitted events if they diverge.

        Recovery is a pure database operation: rebuild projections from the
        event stream and emit a recovery receipt. No external files are read.
        """

        before = self.snapshot()
        rebuilt = self.rebuild_projections_from_events()
        with self._connection(write=True) as connection:
            return self._append_event(
                connection,
                event_type=IntentEventType.RECOVERY_APPLIED,
                subject_id="intent:recovery",
                body={
                    "before_projection_cid": before.projection_cid,
                    "after_projection_cid": rebuilt.projection_cid,
                    "event_watermark": rebuilt.event_watermark,
                    "revision": rebuilt.event_watermark,
                    "recorded_at": _utc_iso(),
                },
            )

    def rebuild_projections_from_events(self) -> IntentSnapshot:
        """Clear intent projections and re-apply admitted intent events.

        Returns the rebuilt snapshot. Domain events themselves are retained.
        """

        with self._connection(write=True) as connection:
            events = connection.execute(
                """
                SELECT event_id, event_type, task_cid, body_json, global_sequence
                FROM domain_events
                WHERE stream_id = ?
                ORDER BY global_sequence ASC
                """,
                [INTENT_STREAM_ID],
            ).fetchall()
            replayed_validation_run_ids: set[str] = set()
            replayed_validation_result_ids: set[str] = set()
            # Preserve non-intent domain events; only rebuild intent projections.
            for table in _PROJECTION_TABLES:
                try:
                    connection.execute(f"DELETE FROM {table}")
                except Exception:
                    # Some tables may be empty or not present in partial installs.
                    pass
            # Leases are shared with the lease coordinator; only clear queue
            # entries owned by this repository's extension schema.
            try:
                connection.execute(
                    "DELETE FROM leases WHERE extension_schema = ?",
                    [_SHARED_QUEUE_LEASE_SCHEMA],
                )
            except Exception:
                pass
            for event_row in events:
                # DuckDBRow iterates keys; index into values explicitly.
                event_type = str(event_row[1])
                body_json = event_row[3]
                body_wrapper = _decode_json(body_json, noun="event body")
                if not isinstance(body_wrapper, dict):
                    continue
                payload = body_wrapper.get("body")
                if not isinstance(payload, dict):
                    payload = body_wrapper
                if event_type == IntentEventType.VALIDATION_RECORDED.value:
                    run_id = str(payload.get("run_id") or "")
                    result_id = str(payload.get("result_id") or "")
                    if run_id:
                        replayed_validation_run_ids.add(run_id)
                    if result_id:
                        replayed_validation_result_ids.add(result_id)
                self._apply_event_payload(
                    connection,
                    event_type=event_type,
                    payload=payload,
                )
            # DuckDB's immediate unique-index checks can reject a delete and
            # reinsert of the same ``(run_id, ordinal)`` in one transaction.
            # Validation projections are therefore updated in place during
            # replay, then rows absent from the admitted event stream are
            # removed before this transaction commits.
            for row in connection.execute(
                "SELECT result_id FROM validation_results"
            ).fetchall():
                result_id = str(row[0])
                if result_id not in replayed_validation_result_ids:
                    connection.execute(
                        "DELETE FROM validation_results WHERE result_id = ?",
                        [result_id],
                    )
            for row in connection.execute(
                "SELECT run_id FROM validation_runs"
            ).fetchall():
                run_id = str(row[0])
                if run_id not in replayed_validation_run_ids:
                    connection.execute(
                        "DELETE FROM validation_runs WHERE run_id = ?",
                        [run_id],
                    )
        return self.snapshot()

    def _apply_event_payload(
        self,
        connection: Any,
        *,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Project one admitted event into current-state tables (idempotent)."""

        now = str(payload.get("recorded_at") or _utc_iso())
        if event_type == IntentEventType.OBJECTIVE_UPSERTED.value:
            oid = str(payload["objective_id"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute("DELETE FROM objectives WHERE objective_id = ?", [oid])
            connection.execute(
                """
                INSERT INTO objectives (
                    objective_id, objective_alias, parent_objective_id, title,
                    status, priority, created_at, updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    str(payload.get("objective_alias") or oid),
                    str(payload.get("parent_objective_id") or ""),
                    str(payload.get("title") or oid),
                    str(payload.get("status") or "open"),
                    str(payload.get("priority") or "P2"),
                    now,
                    now,
                    revision,
                    _canonical(body, noun="objective body"),
                ],
            )
            connection.execute(
                "DELETE FROM objective_revisions WHERE objective_id = ? AND revision = ?",
                [oid, revision],
            )
            connection.execute(
                """
                INSERT INTO objective_revisions (
                    objective_id, revision, status, body_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    oid,
                    revision,
                    str(payload.get("status") or "open"),
                    _canonical(body, noun="objective revision"),
                    now,
                ],
            )
            return

        if event_type == IntentEventType.GOAL_UPSERTED.value:
            gcid = str(payload["goal_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute("DELETE FROM goals WHERE goal_cid = ?", [gcid])
            connection.execute(
                """
                INSERT INTO goals (
                    goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                    title, status, created_at, updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    gcid,
                    str(payload.get("goal_alias") or gcid),
                    str(payload.get("objective_id") or ""),
                    str(payload.get("parent_goal_cid") or ""),
                    int(payload.get("ordinal") or 0),
                    str(payload.get("title") or gcid),
                    str(payload.get("status") or "open"),
                    now,
                    now,
                    revision,
                    _canonical(body, noun="goal body"),
                ],
            )
            return

        if event_type == IntentEventType.GOAL_EDGE_LINKED.value:
            parent = str(payload["parent_goal_cid"])
            child = str(payload["child_goal_cid"])
            kind = str(payload.get("edge_kind") or "depends_on")
            connection.execute(
                """
                DELETE FROM goal_edges
                WHERE parent_goal_cid = ? AND child_goal_cid = ? AND edge_kind = ?
                """,
                [parent, child, kind],
            )
            connection.execute(
                """
                INSERT INTO goal_edges (
                    parent_goal_cid, child_goal_cid, edge_kind
                ) VALUES (?, ?, ?)
                """,
                [parent, child, kind],
            )
            return

        if event_type == IntentEventType.GOAL_REOPENED.value:
            gcid = str(payload["goal_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute(
                """
                UPDATE goals SET status = 'reopened', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE goal_cid = ?
                """,
                [revision, now, _canonical(body, noun="goal body"), gcid],
            )
            return

        if event_type in {
            IntentEventType.PLAN_UPSERTED.value,
            IntentEventType.PLAN_REVISION_APPENDED.value,
            IntentEventType.PLAN_CONTINUED.value,
        }:
            pcid = str(payload["plan_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            status = str(payload.get("status") or "active")
            goal_cid = str(payload.get("goal_cid") or "")
            if event_type == IntentEventType.PLAN_CONTINUED.value:
                status = "active"
            connection.execute("DELETE FROM plans WHERE plan_cid = ?", [pcid])
            connection.execute(
                """
                INSERT INTO plans (
                    plan_cid, goal_cid, plan_alias, status, created_at,
                    updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    pcid,
                    goal_cid,
                    str(payload.get("plan_alias") or pcid),
                    status,
                    now,
                    now,
                    revision,
                    _canonical(body, noun="plan body"),
                ],
            )
            connection.execute(
                "DELETE FROM plan_revisions WHERE plan_cid = ? AND revision = ?",
                [pcid, revision],
            )
            connection.execute(
                """
                INSERT INTO plan_revisions (
                    plan_cid, revision, body_json, recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                [pcid, revision, _canonical(body, noun="plan revision"), now],
            )
            # Mirror live upsert_plan head demotion so rebuild status matches.
            if (
                event_type == IntentEventType.PLAN_UPSERTED.value
                and bool(payload.get("set_head"))
                and status == "active"
                and goal_cid
            ):
                connection.execute(
                    """
                    UPDATE plans SET status = 'superseded', updated_at = ?
                    WHERE goal_cid = ? AND plan_cid <> ? AND status = 'active'
                    """,
                    [now, goal_cid, pcid],
                )
            if event_type == IntentEventType.PLAN_CONTINUED.value:
                predecessor = str(payload.get("continuation_of") or "")
                if predecessor:
                    pred_row = connection.execute(
                        "SELECT revision, body_json FROM plans WHERE plan_cid = ?",
                        [predecessor],
                    ).fetchone()
                    if pred_row is not None:
                        pred_body = _decode_json(pred_row[1], noun="plan body")
                        if not isinstance(pred_body, dict):
                            pred_body = {}
                        else:
                            pred_body = dict(pred_body)
                        pred_body["continued_by"] = pcid
                        connection.execute(
                            """
                            UPDATE plans SET status = 'continued',
                                revision = ?, updated_at = ?, body_json = ?
                            WHERE plan_cid = ?
                            """,
                            [
                                int(pred_row[0]) + 1,
                                now,
                                _canonical(pred_body, noun="plan body"),
                                predecessor,
                            ],
                        )
            return

        if event_type == IntentEventType.PLAN_SUPERSEDED.value:
            pcid = str(payload["plan_cid"])
            successor = str(payload.get("successor_plan_cid") or "")
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            connection.execute(
                """
                UPDATE plans SET status = 'superseded', revision = ?,
                    updated_at = ?, body_json = ?
                WHERE plan_cid = ?
                """,
                [revision, now, _canonical(body, noun="plan body"), pcid],
            )
            if successor:
                connection.execute(
                    """
                    UPDATE plans SET status = 'active', updated_at = ?
                    WHERE plan_cid = ?
                    """,
                    [now, successor],
                )
            return

        if event_type == IntentEventType.TASK_UPSERTED.value:
            tcid = str(payload["task_cid"])
            revision = int(payload.get("revision") or 1)
            body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
            identity = (
                payload.get("identity")
                if isinstance(payload.get("identity"), dict)
                else {"task_cid": tcid}
            )
            connection.execute("DELETE FROM tasks WHERE task_cid = ?", [tcid])
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tcid,
                    str(payload.get("task_alias") or tcid),
                    str(payload.get("goal_cid") or ""),
                    str(payload.get("plan_cid") or ""),
                    str(payload.get("objective_id") or ""),
                    int(payload.get("ordinal") or 0),
                    str(payload.get("status") or "ready"),
                    revision,
                    str(payload.get("priority") or "P2"),
                    now,
                    now,
                    _canonical(identity, noun="task identity"),
                    _canonical(body, noun="task body"),
                ],
            )
            if "dependencies" in payload:
                deps = payload.get("dependencies") or []
                if isinstance(deps, Sequence) and not isinstance(deps, (str, bytes)):
                    self._set_dependencies_on(
                        connection, tcid, [str(item) for item in deps]
                    )
            if "outputs" in payload:
                outputs = payload.get("outputs") or []
                if isinstance(outputs, Sequence) and not isinstance(
                    outputs, (str, bytes)
                ):
                    self._set_outputs_on(
                        connection,
                        tcid,
                        [item for item in outputs if isinstance(item, Mapping)],
                    )
            if "acceptance" in payload:
                acceptance = payload.get("acceptance") or []
                if isinstance(acceptance, Sequence) and not isinstance(
                    acceptance, (str, bytes)
                ):
                    self._set_acceptance_on(connection, tcid, list(acceptance))
            if "validations" in payload:
                validations = payload.get("validations") or []
                if isinstance(validations, Sequence) and not isinstance(
                    validations, (str, bytes)
                ):
                    self._set_validations_on(connection, tcid, list(validations))
            return

        if event_type == IntentEventType.TASK_DEPENDENCIES_SET.value:
            tcid = str(payload["task_cid"])
            deps = payload.get("dependencies") or []
            if isinstance(deps, Sequence):
                self._set_dependencies_on(
                    connection, tcid, [str(item) for item in deps]
                )
            return

        if event_type in {
            IntentEventType.TASK_STATUS_CHANGED.value,
            IntentEventType.COMPLETION_RECORDED.value,
        }:
            tcid = str(payload["task_cid"])
            revision = int(payload.get("revision") or 1)
            status = str(payload.get("status") or "ready")
            receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else {}
            existing_body_row = connection.execute(
                "SELECT body_json FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if existing_body_row is not None:
                body = _decode_json(existing_body_row[0], noun="task body")
                if not isinstance(body, dict):
                    body = {}
                else:
                    body = dict(body)
            else:
                body = {}
            if receipt:
                body["completion_receipt"] = receipt
            connection.execute(
                """
                UPDATE tasks SET status = ?, revision = ?, updated_at = ?,
                    body_json = ?
                WHERE task_cid = ?
                """,
                [
                    status,
                    revision,
                    now,
                    _canonical(body, noun="task body"),
                    tcid,
                ],
            )
            # Ensure row exists when replaying status after a partial wipe.
            exists = connection.execute(
                "SELECT 1 FROM tasks WHERE task_cid = ?", [tcid]
            ).fetchone()
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_cid, objective_id,
                        ordinal, status, revision, priority, created_at,
                        updated_at, identity_json, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        str(payload.get("task_alias") or tcid),
                        str(payload.get("goal_cid") or ""),
                        "",
                        "",
                        0,
                        status,
                        revision,
                        "P2",
                        now,
                        now,
                        _canonical({"task_cid": tcid}, noun="identity"),
                        _canonical(body, noun="task body"),
                    ],
                )
            if event_type == IntentEventType.COMPLETION_RECORDED.value:
                receipt_cid = str(
                    payload.get("completion_receipt_cid")
                    or content_identity(
                        {
                            "task_cid": tcid,
                            "revision": revision,
                            "status": status,
                        }
                    )
                )
                evidence_digest = str(
                    payload.get("evidence_digest")
                    or content_identity({"task_cid": tcid, "revision": revision})
                )
                connection.execute(
                    "DELETE FROM completion_receipts WHERE receipt_cid = ?",
                    [receipt_cid],
                )
                connection.execute(
                    """
                    INSERT INTO completion_receipts (
                        receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                        fencing_token, completed_at, validation_run_id,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt_cid,
                        tcid,
                        str(payload.get("goal_cid") or ""),
                        "",
                        "",
                        0,
                        now,
                        "",
                        evidence_digest,
                        _canonical(
                            {
                                "schema": COMPLETION_EVIDENCE_SCHEMA,
                                "receipt": receipt,
                                "revision": revision,
                            },
                            noun="completion receipt",
                        ),
                    ],
                )
            return

        if event_type == IntentEventType.EVIDENCE_RECORDED.value:
            evidence_id = str(payload["evidence_id"])
            connection.execute(
                "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                [evidence_id],
            )
            connection.execute(
                """
                INSERT INTO evidence_nodes (
                    evidence_id, parent_evidence_id, task_cid, evidence_kind,
                    digest, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    evidence_id,
                    str(payload.get("parent_evidence_id") or ""),
                    str(payload.get("task_cid") or ""),
                    str(payload.get("evidence_kind") or "evidence"),
                    str(payload.get("digest") or ""),
                    now,
                    _canonical(
                        payload.get("body")
                        if isinstance(payload.get("body"), dict)
                        else {},
                        noun="evidence body",
                    ),
                ],
            )
            return

        if event_type == IntentEventType.VALIDATION_RECORDED.value:
            run_id = str(payload.get("run_id") or "")
            result_id = str(payload.get("result_id") or "")
            tcid = str(payload.get("task_cid") or "")
            if run_id:
                connection.execute(
                    """
                    INSERT INTO validation_runs (
                        run_id, task_cid, attempt_id, started_at, finished_at,
                        status, command_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id) DO UPDATE SET
                        task_cid = EXCLUDED.task_cid,
                        attempt_id = EXCLUDED.attempt_id,
                        started_at = EXCLUDED.started_at,
                        finished_at = EXCLUDED.finished_at,
                        status = EXCLUDED.status,
                        command_digest = EXCLUDED.command_digest,
                        body_json = EXCLUDED.body_json
                    """,
                    [
                        run_id,
                        tcid,
                        "",
                        now,
                        now,
                        str(payload.get("outcome") or "passed"),
                        content_identity(
                            {"argv": list(payload.get("argv") or ())}
                        ),
                        _canonical(
                            {
                                "argv": list(payload.get("argv") or ()),
                                **(
                                    payload.get("body")
                                    if isinstance(payload.get("body"), dict)
                                    else {}
                                ),
                            },
                            noun="validation run",
                        ),
                    ],
                )
            if result_id:
                connection.execute(
                    """
                    INSERT INTO validation_results (
                        result_id, run_id, task_cid, ordinal, outcome,
                        evidence_digest, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, ordinal) DO UPDATE SET
                        result_id = EXCLUDED.result_id,
                        task_cid = EXCLUDED.task_cid,
                        outcome = EXCLUDED.outcome,
                        evidence_digest = EXCLUDED.evidence_digest,
                        body_json = EXCLUDED.body_json
                    """,
                    [
                        result_id,
                        run_id,
                        tcid,
                        0,
                        str(payload.get("outcome") or "passed"),
                        str(payload.get("evidence_digest") or ""),
                        _canonical(
                            payload.get("body")
                            if isinstance(payload.get("body"), dict)
                            else {},
                            noun="validation result",
                        ),
                    ],
                )
            if str(payload.get("outcome") or "") == "passed" and tcid:
                evidence_id = content_identity(
                    {
                        "task_cid": tcid,
                        "evidence_kind": "validation",
                        "digest": str(payload.get("evidence_digest") or ""),
                        "run_id": run_id,
                    }
                )
                connection.execute(
                    "DELETE FROM evidence_nodes WHERE evidence_id = ?",
                    [evidence_id],
                )
                connection.execute(
                    """
                    INSERT INTO evidence_nodes (
                        evidence_id, parent_evidence_id, task_cid, evidence_kind,
                        digest, created_at, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        evidence_id,
                        "",
                        tcid,
                        "validation",
                        str(payload.get("evidence_digest") or ""),
                        now,
                        _canonical(
                            {"run_id": run_id, "result_id": result_id},
                            noun="validation evidence",
                        ),
                    ],
                )
            return

        if event_type == IntentEventType.QUEUE_BACKOFF.value:
            tcid = str(payload["task_cid"])
            attempt = int(payload.get("attempt") or 1)
            retry = int(payload.get("retry_not_before_ms") or 0)
            reason = str(payload.get("reason") or "backoff")
            penalty = int(payload.get("selection_penalty") or 0)
            exists = connection.execute(
                "SELECT 1 FROM leases WHERE task_cid = ?", [tcid]
            ).fetchone()
            extension = _canonical(
                {
                    "selection_penalty": penalty,
                    "consecutive_failures": attempt,
                    "reason": reason,
                },
                noun="queue extension",
            )
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO leases (
                        task_cid, claim_cid, resolution_cid, claimant_did,
                        logical_epoch, fencing_token, expires_at_ms, attempt,
                        state, started_at_ms, release_reason, retry_not_before_ms,
                        owner_session_id, fence_epoch, revision, extension_schema,
                        extension_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        tcid,
                        f"claim:queue:{tcid}",
                        f"resolution:queue:{tcid}",
                        self.owner_id,
                        1,
                        1,
                        0,
                        attempt,
                        "released",
                        0,
                        reason,
                        retry,
                        self.session_id,
                        1,
                        1,
                        QUEUE_ENTRY_SCHEMA,
                        extension,
                    ],
                )
            else:
                connection.execute(
                    """
                    UPDATE leases SET attempt = ?, retry_not_before_ms = ?,
                        release_reason = ?, extension_schema = ?,
                        extension_json = ?, revision = revision + 1
                    WHERE task_cid = ?
                    """,
                    [attempt, retry, reason, QUEUE_ENTRY_SCHEMA, extension, tcid],
                )
            return

        if event_type == IntentEventType.QUEUE_RETRY.value:
            tcid = str(payload["task_cid"])
            connection.execute(
                """
                UPDATE leases SET retry_not_before_ms = 0, release_reason = '',
                    extension_json = ?, revision = revision + 1
                WHERE task_cid = ?
                """,
                [
                    _canonical(
                        {
                            "selection_penalty": 0,
                            "consecutive_failures": 0,
                            "reason": "",
                        },
                        noun="queue extension",
                    ),
                    tcid,
                ],
            )
            return

        if event_type == IntentEventType.ATTEMPT_RECORDED.value:
            attempt_id = str(payload["attempt_id"])
            connection.execute(
                "DELETE FROM task_attempts WHERE attempt_id = ?", [attempt_id]
            )
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_cid, attempt_number, owner_session_id,
                    fencing_token, fence_epoch, started_at, finished_at,
                    status, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    attempt_id,
                    str(payload.get("task_cid") or ""),
                    int(payload.get("attempt_number") or 1),
                    str(payload.get("owner_session_id") or self.session_id),
                    int(payload.get("fencing_token") or 1),
                    1,
                    now,
                    "",
                    str(payload.get("status") or "started"),
                    1,
                ],
            )
            return

        if event_type == IntentEventType.TASK_BLOCKED.value:
            block_id = str(payload["block_id"])
            tcid = str(payload["task_cid"])
            connection.execute(
                "DELETE FROM task_blocks WHERE block_id = ?", [block_id]
            )
            connection.execute(
                """
                INSERT INTO task_blocks (
                    block_id, task_cid, blocker_kind, blocker_id, reason,
                    created_at, cleared_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    block_id,
                    tcid,
                    str(payload.get("blocker_kind") or "manual"),
                    str(payload.get("blocker_id") or "unknown"),
                    str(payload.get("reason") or "blocked"),
                    now,
                    "",
                    "active",
                ],
            )
            connection.execute(
                """
                UPDATE tasks SET status = 'blocked', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [int(payload.get("revision") or 1), now, tcid],
            )
            return

        if event_type == IntentEventType.TASK_UNBLOCKED.value:
            tcid = str(payload["task_cid"])
            connection.execute(
                """
                UPDATE task_blocks SET state = 'cleared', cleared_at = ?
                WHERE task_cid = ? AND state = 'active'
                """,
                [now, tcid],
            )
            connection.execute(
                """
                UPDATE tasks SET status = 'ready', revision = ?, updated_at = ?
                WHERE task_cid = ?
                """,
                [int(payload.get("revision") or 1), now, tcid],
            )
            return

        # Recovery and unknown types are intentionally no-ops for projection.

    def snapshot(self) -> IntentSnapshot:
        with self._connection(write=False) as connection:
            objective_count = int(
                connection.execute("SELECT COUNT(*) FROM objectives").fetchone()[0]
            )
            goal_count = int(
                connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
            )
            plan_count = int(
                connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
            )
            task_count = int(
                connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            )
            dependency_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM task_dependencies"
                ).fetchone()[0]
            )
            watermark = int(
                connection.execute(
                    "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
                ).fetchone()[0]
            )
            task_rows = connection.execute(
                """
                SELECT task_cid, status, revision FROM tasks
                ORDER BY task_cid
                """
            ).fetchall()
            plan_rows = connection.execute(
                """
                SELECT plan_cid, status, revision FROM plans
                ORDER BY plan_cid
                """
            ).fetchall()
            goal_rows = connection.execute(
                """
                SELECT goal_cid, status, revision FROM goals
                ORDER BY goal_cid
                """
            ).fetchall()
        material = {
            "objectives": objective_count,
            "goals": [
                {"goal_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in goal_rows
            ],
            "plans": [
                {"plan_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in plan_rows
            ],
            "tasks": [
                {"task_cid": str(r[0]), "status": str(r[1]), "revision": int(r[2])}
                for r in task_rows
            ],
            "dependency_count": dependency_count,
            "event_watermark": watermark,
        }
        return IntentSnapshot(
            objective_count=objective_count,
            goal_count=goal_count,
            plan_count=plan_count,
            task_count=task_count,
            dependency_count=dependency_count,
            event_watermark=watermark,
            projection_cid=content_identity(material),
            recorded_at=_utc_iso(),
        )

    def plan_revisions(self) -> PlanRevisionRepository:
        """Return the plan-revision repository view over this intent store."""

        return PlanRevisionRepository(self)


# ---------------------------------------------------------------------------
# PlanRevisionRepository
# ---------------------------------------------------------------------------


class PlanRevisionRepository:
    """Plan revision heads, deltas, supersession, and continuation.

    Interface: ``PlanRevisionRepository@1``.

    Thin, typed facade over :class:`IntentRepository` so plan-revision callers
    do not need the full intent surface. All mutations remain single-transaction
    database operations with domain events.
    """

    INTERFACE: ClassVar[str] = PLAN_REVISION_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = PLAN_REVISION_REPOSITORY_SCHEMA

    def __init__(self, intent: IntentRepository) -> None:
        if not isinstance(intent, IntentRepository):
            raise TypeError("PlanRevisionRepository requires an IntentRepository")
        self._intent = intent

    @property
    def intent(self) -> IntentRepository:
        return self._intent

    def upsert(
        self,
        *,
        plan_cid: str,
        goal_cid: str,
        plan_alias: str,
        status: str = "active",
        body: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
        set_head: bool = True,
    ) -> IntentReceipt:
        return self._intent.upsert_plan(
            plan_cid=plan_cid,
            goal_cid=goal_cid,
            plan_alias=plan_alias,
            status=status,
            body=body,
            expected_revision=expected_revision,
            set_head=set_head,
        )

    def append_revision(
        self,
        *,
        plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
        delta: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.append_plan_revision(
            plan_cid=plan_cid,
            expected_revision=expected_revision,
            body=body,
            delta=delta,
        )

    def supersede(
        self,
        *,
        plan_cid: str,
        successor_plan_cid: str,
        expected_revision: int,
        reason: str = "superseded",
    ) -> IntentReceipt:
        return self._intent.supersede_plan(
            plan_cid=plan_cid,
            successor_plan_cid=successor_plan_cid,
            expected_revision=expected_revision,
            reason=reason,
        )

    def continue_from(
        self,
        *,
        plan_cid: str,
        continuation_plan_cid: str,
        expected_revision: int,
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        return self._intent.continue_plan(
            plan_cid=plan_cid,
            continuation_plan_cid=continuation_plan_cid,
            expected_revision=expected_revision,
            body=body,
        )

    def get(self, plan_cid: str) -> Mapping[str, Any] | None:
        return self._intent.get_plan(plan_cid)

    def head(self, goal_cid: str) -> PlanHead | None:
        return self._intent.get_plan_head(goal_cid)

    def list_revisions(self, plan_cid: str) -> tuple[Mapping[str, Any], ...]:
        pcid = _identifier(plan_cid, noun="plan_cid")
        with self._intent._connection(write=False) as connection:
            rows = connection.execute(
                """
                SELECT plan_cid, revision, body_json, recorded_at
                FROM plan_revisions
                WHERE plan_cid = ?
                ORDER BY revision ASC
                """,
                [pcid],
            ).fetchall()
        return tuple(
            MappingProxyType(
                {
                    "plan_cid": str(row[0]),
                    "revision": int(row[1]),
                    "body": _decode_json(row[2], noun="plan revision body"),
                    "recorded_at": str(row[3]),
                }
            )
            for row in rows
        )


# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------


def _parse_iso_ms(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return int(moment.timestamp() * 1000)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def open_intent_repository(
    database_path: str | Path,
    *,
    owner_id: str = DEFAULT_OWNER_ID,
    session_id: str = DEFAULT_SESSION_ID,
    install_schema: bool = True,
    evidence_freshness_seconds: int = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> IntentRepository:
    """Open an intent repository against ``control.duckdb`` (or test path)."""

    return IntentRepository(
        database_path,
        owner_id=owner_id,
        session_id=session_id,
        install_schema=install_schema,
        evidence_freshness_seconds=evidence_freshness_seconds,
    )


__all__ = (
    "INTENT_REPOSITORY_INTERFACE",
    "PLAN_REVISION_REPOSITORY_INTERFACE",
    "INTENT_REPOSITORY_SCHEMA",
    "PLAN_REVISION_REPOSITORY_SCHEMA",
    "IntentEventType",
    "IntentRepository",
    "IntentRepositoryError",
    "IntentRepositoryConflictError",
    "IntentRepositoryIntegrityError",
    "IntentRepositoryBoundsError",
    "IntentRepositoryNotOpenError",
    "IntentCompletionError",
    "IntentEvidenceError",
    "DuckDBUnavailableError",
    "IntentReceipt",
    "IntentSnapshot",
    "QueueEntry",
    "PlanHead",
    "PlanRevisionRepository",
    "open_intent_repository",
    "duckdb_available",
)
