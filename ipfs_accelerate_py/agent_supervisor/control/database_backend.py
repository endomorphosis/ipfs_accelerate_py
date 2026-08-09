"""Database-backed supervisor control backend (DQP-029).

Interfaces: ``DatabaseSupervisorBackend@1``

:class:`DatabaseSupervisorBackend` is a direct Python adapter for
:class:`~.control_plane.SupervisorControlService`.  It reads and mutates the
control-plane DuckDB schema for status, health, goals, tasks, lanes, daemons,
events, logs, metrics, worktrees, mutations, AST summaries, receipts, and
lifecycle transitions.  Adapters never shell out; raw SQL is rejected at the
parameter boundary.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Union

from .control_contracts import (
    MUTATION_OPERATIONS,
    READ_OPERATIONS,
    Operation,
    OperationRequest,
)
from .control_plane import (
    BackendConflictError,
    BackendResponse,
    InvalidLifecycleTransitionError,
    OperationUnavailableError,
    SupervisorLifecycleState,
    lifecycle_transition_is_legal,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SUPERVISOR_BACKEND_INTERFACE: Final[str] = "DatabaseSupervisorBackend@1"
DATABASE_SUPERVISOR_BACKEND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-supervisor-backend@1"
)
DATABASE_PROGRAM_TARGET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-program-target@1"
)
DATABASE_LIFECYCLE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-lifecycle-event@1"
)

DATABASE_SUPERVISOR_BACKEND_VERSION: Final[int] = 1
DEFAULT_TARGET_ID: Final[str] = "supervisor"
DEFAULT_OWNER_ID: Final[str] = "database-supervisor-backend:local"
DEFAULT_STREAM_ID: Final[str] = "stream:control"
DEFAULT_PAGE_LIMIT: Final[int] = 50
MAX_PAGE_LIMIT: Final[int] = 256
MAX_OFFSET: Final[int] = 1_000_000
MAX_BODY_BYTES: Final[int] = 262_144

# Lifecycle action -> requested state (mirrors control_plane vocabulary).
_ACTION_REQUESTED_STATE: Final[Mapping[Operation, SupervisorLifecycleState]] = (
    MappingProxyType(
        {
            Operation.START: SupervisorLifecycleState.STARTING,
            Operation.PAUSE: SupervisorLifecycleState.PAUSED,
            Operation.RESUME: SupervisorLifecycleState.HEALTHY,
            Operation.DRAIN: SupervisorLifecycleState.DRAINING,
            Operation.STOP: SupervisorLifecycleState.STOPPING,
            Operation.RETRY: SupervisorLifecycleState.STARTING,
            Operation.CANCEL: SupervisorLifecycleState.STOPPING,
            Operation.QUARANTINE: SupervisorLifecycleState.BLOCKED,
        }
    )
)

_LIFECYCLE_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    _ACTION_REQUESTED_STATE
)

_READ_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.CAPABILITIES,
        Operation.STATUS,
        Operation.HEALTH,
        Operation.METRICS,
        Operation.GOALS,
        Operation.TASKS,
        Operation.BUNDLES,
        Operation.LANES,
        Operation.EVENTS,
        Operation.RECEIPTS,
        Operation.CACHE_INSPECT,
        Operation.ARTIFACT_QUERY,
    }
)

_BUILTIN_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    _READ_OPERATIONS | _LIFECYCLE_OPERATIONS
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_RESOURCE_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseSupervisorBackendError(RuntimeError):
    """Base error for database supervisor backend failures."""


class DatabaseSupervisorBackendUnavailableError(DatabaseSupervisorBackendError):
    """DuckDB or the control-plane schema is unavailable."""


class DatabaseSupervisorBackendBoundsError(
    DatabaseSupervisorBackendError, ValueError
):
    """A request exceeded a hard bound."""


class DatabaseSupervisorBackendConflictError(
    DatabaseSupervisorBackendError, BackendConflictError
):
    """The current database state rejects the request."""


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


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json_loads(raw: Any) -> Any:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return {"value": raw}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _bounded_window(request: OperationRequest) -> tuple[int, int]:
    limit_raw = request.parameters.get("limit", DEFAULT_PAGE_LIMIT)
    offset_raw = request.parameters.get("offset", 0)
    if isinstance(limit_raw, bool) or not isinstance(limit_raw, int):
        raise DatabaseSupervisorBackendBoundsError("limit must be an integer")
    if isinstance(offset_raw, bool) or not isinstance(offset_raw, int):
        raise DatabaseSupervisorBackendBoundsError("offset must be an integer")
    limit = min(max(limit_raw, 1), min(request.bounds.max_items, MAX_PAGE_LIMIT))
    if offset_raw < 0 or offset_raw > MAX_OFFSET:
        raise DatabaseSupervisorBackendBoundsError("offset is out of bounds")
    return limit, offset_raw


def _target_id(request: OperationRequest) -> str:
    value = str(
        request.parameters.get("target_id")
        or request.parameters.get("program_id")
        or DEFAULT_TARGET_ID
    ).strip()
    if not value or not _SAFE_ID.fullmatch(value):
        raise ValueError("target_id must be a compact identifier")
    return value


def _resource_kind(value: Any, *, default: str) -> str:
    text = str(value or default).strip().lower()
    if not _RESOURCE_KIND_RE.fullmatch(text):
        raise ValueError("resource kind must be a lowercase identifier")
    return text


def _reject_raw_sql(request: OperationRequest) -> None:
    sql = str(request.parameters.get("sql") or "").strip()
    if sql:
        raise ValueError(
            "raw SQL is disabled at the database supervisor control boundary"
        )


def _redact(value: Any) -> Any:
    from ..task_sources.control_plane_contracts import redact_mapping

    return redact_mapping(value)


def _page(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    selected = list(items[offset : offset + limit])
    return {
        "items": selected,
        "count": len(selected),
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(selected) < len(items),
    }


# ---------------------------------------------------------------------------
# Program target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseProgramTarget:
    """Explicit database program target for control operations."""

    SCHEMA: Final[str] = DATABASE_PROGRAM_TARGET_SCHEMA

    database_path: str
    program_id: str = DEFAULT_TARGET_ID
    repository_id: str = "repository:local"
    store_id: str = "store:control"

    def __post_init__(self) -> None:
        path = str(self.database_path).strip()
        if not path:
            raise ValueError("database_path must not be empty")
        object.__setattr__(self, "database_path", path)
        program = str(self.program_id).strip() or DEFAULT_TARGET_ID
        if not _SAFE_ID.fullmatch(program):
            raise ValueError("program_id must be a compact identifier")
        object.__setattr__(self, "program_id", program)

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.SCHEMA,
            "database_path": self.database_path,
            "program_id": self.program_id,
            "repository_id": self.repository_id,
            "store_id": self.store_id,
        }


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class DatabaseSupervisorBackend:
    """DuckDB-backed backend for the canonical supervisor control service.

    Registration and construction are inert.  The first :meth:`execute` call
    installs the control-plane schema when needed and opens a short-lived
    connection under the shared DuckDB file lock.
    """

    INTERFACE: Final[str] = DATABASE_SUPERVISOR_BACKEND_INTERFACE
    SCHEMA: Final[str] = DATABASE_SUPERVISOR_BACKEND_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        owner_id: str = DEFAULT_OWNER_ID,
        install_schema: bool = True,
        clock_ms: Callable[[], int] = _now_ms,
        ensure_schema: Callable[[Path], None] | None = None,
    ) -> None:
        path = Path(database_path)
        if not str(path):
            raise ValueError("database_path must not be empty")
        self._path = path
        self._owner_id = str(owner_id).strip() or DEFAULT_OWNER_ID
        self._install_schema = bool(install_schema)
        self._clock_ms = clock_ms
        self._ensure_schema = ensure_schema
        self._lock = threading.RLock()
        self._schema_ready = False
        self._event_sequence = 0
        # Discovery/capability probes remain side-effect free.
        self.optional_providers_loaded = False
        self.processes_started = False

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        return tuple(sorted(_BUILTIN_OPERATIONS, key=lambda item: item.value))

    def target(self, program_id: str = DEFAULT_TARGET_ID) -> DatabaseProgramTarget:
        return DatabaseProgramTarget(
            database_path=str(self._path),
            program_id=program_id,
        )

    def discovery_manifest(self) -> Mapping[str, Any]:
        """Return inert discovery metadata (no database I/O)."""

        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "version": DATABASE_SUPERVISOR_BACKEND_VERSION,
            "database_path": str(self._path),
            "operations": [item.value for item in self.registered_operations],
            "read_operations": sorted(item.value for item in _READ_OPERATIONS),
            "lifecycle_operations": sorted(
                item.value for item in _LIFECYCLE_OPERATIONS
            ),
            "side_effects": False,
            "shell_out": False,
            "raw_sql": False,
        }

    # -- connection / schema ------------------------------------------------

    def _require_duckdb(self) -> None:
        if not duckdb_available():
            raise DatabaseSupervisorBackendUnavailableError(
                "DuckDB is required for DatabaseSupervisorBackend"
            )

    def _prepare_schema(self) -> None:
        if self._schema_ready:
            return
        self._require_duckdb()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._ensure_schema is not None:
            self._ensure_schema(self._path)
        elif self._install_schema:
            from ..task_sources.control_plane_schema import (
                install_control_plane_schema,
            )

            install_control_plane_schema(
                self._path,
                application_version="0.0.45",
                tool_version="1.5.2",
                owner_id=self._owner_id,
            )
        self._schema_ready = True

    def _connect(self) -> Any:
        self._prepare_schema()
        from ..task_sources.duckdb_state import open_duckdb_connection

        return open_duckdb_connection(self._path)

    # -- generic query helpers ----------------------------------------------

    def _fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            cursor = connection.execute(sql, list(params))
            rows = cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, Mapping):
                result.append(dict(row))
            else:
                # Plain tuple fallback when a raw duckdb cursor is used.
                result.append({"value": row})
        return result

    def _fetch_one(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> dict[str, Any] | None:
        rows = self._fetch_all(sql, params)
        return rows[0] if rows else None

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._connect() as connection:
            connection.execute(sql, list(params))

    def _windowed(
        self,
        request: OperationRequest,
        sql: str,
        params: Sequence[Any] = (),
        *,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        limit, offset = _bounded_window(request)
        rows = self._fetch_all(sql, params)
        if transform is not None:
            items = [dict(transform(row)) for row in rows]
        else:
            items = [dict(row) for row in rows]
        page = _page(items, limit=limit, offset=offset)
        page["items"] = [_redact(item) for item in page["items"]]
        return page

    # -- entity readers -----------------------------------------------------

    def _ensure_supervisor_row(self, target_id: str) -> dict[str, Any]:
        existing = self._fetch_one(
            "SELECT * FROM supervisor_instances WHERE supervisor_id = ? LIMIT 1",
            (target_id,),
        )
        if existing is not None:
            return existing
        now = _utc_now()
        self._execute(
            """
            INSERT INTO supervisor_instances (
                supervisor_id, repository_id, process_birth_id,
                started_at, stopped_at, status, revision,
                extension_schema, extension_json
            ) VALUES (?, ?, ?, ?, NULL, ?, 0, '', ?)
            """,
            (
                target_id,
                "repository:local",
                f"birth:{target_id}",
                now,
                SupervisorLifecycleState.STOPPED.value,
                _json_dumps({"fence_epoch": 0, "generation": 0}),
            ),
        )
        loaded = self._fetch_one(
            "SELECT * FROM supervisor_instances WHERE supervisor_id = ? LIMIT 1",
            (target_id,),
        )
        if loaded is None:
            raise DatabaseSupervisorBackendError(
                "failed to materialize supervisor instance row"
            )
        return loaded

    def _status_payload(self, request: OperationRequest) -> dict[str, Any]:
        target_id = _target_id(request)
        row = self._ensure_supervisor_row(target_id)
        extension = _json_loads(row.get("extension_json"))
        if not isinstance(extension, Mapping):
            extension = {}
        state = str(row.get("status") or SupervisorLifecycleState.STOPPED.value)
        payload = {
            "target_id": target_id,
            "program_id": target_id,
            "state": state,
            "status": state,
            "repository_id": row.get("repository_id") or "",
            "process_birth_id": row.get("process_birth_id") or "",
            "started_at": row.get("started_at") or "",
            "stopped_at": row.get("stopped_at"),
            "revision": int(row.get("revision") or 0),
            "generation": int(extension.get("generation") or 0),
            "fence_epoch": int(extension.get("fence_epoch") or 0),
            "phase": str(extension.get("phase") or state),
            "database_path": str(self._path),
            "authority": "database",
            "schema": DATABASE_SUPERVISOR_BACKEND_SCHEMA,
        }
        return _redact(payload)

    def _health_payload(self, request: OperationRequest) -> dict[str, Any]:
        status = self._status_payload(request)
        state = SupervisorLifecycleState(str(status["state"]))
        samples = self._fetch_all(
            """
            SELECT sample_id, subject_kind, subject_id, observed_at, status, body_json
            FROM health_samples
            WHERE subject_id = ?
            ORDER BY observed_at DESC
            LIMIT 16
            """,
            (status["target_id"],),
        )
        sample_items = []
        for row in samples:
            body = _json_loads(row.get("body_json"))
            sample_items.append(
                _redact(
                    {
                        "sample_id": row.get("sample_id"),
                        "subject_kind": row.get("subject_kind"),
                        "subject_id": row.get("subject_id"),
                        "observed_at": row.get("observed_at"),
                        "status": row.get("status"),
                        "body": body if isinstance(body, Mapping) else {},
                    }
                )
            )
        healthy = state in {
            SupervisorLifecycleState.HEALTHY,
            SupervisorLifecycleState.DEGRADED,
            SupervisorLifecycleState.STARTING,
            SupervisorLifecycleState.DRAINING,
        }
        return {
            **status,
            "healthy": healthy,
            "accepts_new_work": state.accepts_new_work,
            "terminal": state.terminal,
            "samples": sample_items,
        }

    def _goals(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                   title, status, created_at, updated_at, revision, body_json
            FROM goals
            ORDER BY ordinal ASC, goal_cid ASC
            """,
            transform=lambda row: {
                "goal_cid": row.get("goal_cid"),
                "goal_alias": row.get("goal_alias"),
                "objective_id": row.get("objective_id"),
                "parent_goal_cid": row.get("parent_goal_cid"),
                "ordinal": row.get("ordinal"),
                "title": row.get("title"),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "revision": row.get("revision"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _tasks(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                   ordinal, status, revision, priority, created_at, updated_at,
                   identity_json, body_json
            FROM tasks
            ORDER BY ordinal ASC, task_cid ASC
            """,
            transform=lambda row: {
                "task_cid": row.get("task_cid"),
                "task_alias": row.get("task_alias"),
                "goal_cid": row.get("goal_cid"),
                "plan_cid": row.get("plan_cid"),
                "objective_id": row.get("objective_id"),
                "ordinal": row.get("ordinal"),
                "status": row.get("status"),
                "revision": row.get("revision"),
                "priority": row.get("priority"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "identity": _json_loads(row.get("identity_json")),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _lanes(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT daemon_id, supervisor_id, process_birth_id, role,
                   started_at, stopped_at, status, revision
            FROM daemon_instances
            ORDER BY daemon_id ASC
            """,
        )

    def _daemons(self, request: OperationRequest) -> dict[str, Any]:
        return self._lanes(request)

    def _events(self, request: OperationRequest) -> dict[str, Any]:
        kind = _resource_kind(request.parameters.get("kind"), default="events")
        if kind in {"logs", "log", "structured_logs"}:
            return self._logs(request)
        after = int(request.parameters.get("after_sequence") or 0)
        if after < 0:
            raise DatabaseSupervisorBackendBoundsError(
                "after_sequence must be non-negative"
            )
        return self._windowed(
            request,
            """
            SELECT event_id, stream_id, sequence, global_sequence, event_type,
                   task_cid, attempt_id, session_id, recorded_at, body_json
            FROM domain_events
            WHERE global_sequence > ?
            ORDER BY global_sequence ASC
            """,
            (after,),
            transform=lambda row: {
                "event_id": row.get("event_id"),
                "stream_id": row.get("stream_id"),
                "sequence": row.get("sequence"),
                "global_sequence": row.get("global_sequence"),
                "event_type": row.get("event_type"),
                "task_cid": row.get("task_cid"),
                "attempt_id": row.get("attempt_id"),
                "session_id": row.get("session_id"),
                "recorded_at": row.get("recorded_at"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _logs(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT log_id, severity, component, trace_id, span_id, task_cid,
                   attempt_id, session_id, recorded_at, message, body_json
            FROM structured_logs
            ORDER BY recorded_at ASC, log_id ASC
            """,
            transform=lambda row: {
                "log_id": row.get("log_id"),
                "severity": row.get("severity"),
                "component": row.get("component"),
                "trace_id": row.get("trace_id"),
                "span_id": row.get("span_id"),
                "task_cid": row.get("task_cid"),
                "attempt_id": row.get("attempt_id"),
                "session_id": row.get("session_id"),
                "recorded_at": row.get("recorded_at"),
                "message": row.get("message"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _metrics(self, request: OperationRequest) -> dict[str, Any]:
        metrics = self._fetch_all(
            """
            SELECT metric_id, metric_name, unit, description, created_at
            FROM metrics
            ORDER BY metric_name ASC
            """
        )
        samples = self._fetch_all(
            """
            SELECT sample_id, metric_id, observed_at, value_milli, labels_json, stratum
            FROM metric_samples
            ORDER BY observed_at DESC
            LIMIT 256
            """
        )
        limit, offset = _bounded_window(request)
        metric_items = [
            {
                "metric_id": row.get("metric_id"),
                "metric_name": row.get("metric_name"),
                "unit": row.get("unit"),
                "description": row.get("description"),
                "created_at": row.get("created_at"),
            }
            for row in metrics
        ]
        sample_items = [
            _redact(
                {
                    "sample_id": row.get("sample_id"),
                    "metric_id": row.get("metric_id"),
                    "observed_at": row.get("observed_at"),
                    "value_milli": row.get("value_milli"),
                    "labels": _json_loads(row.get("labels_json")),
                    "stratum": row.get("stratum"),
                }
            )
            for row in samples
        ]
        page = _page(metric_items, limit=limit, offset=offset)
        page["samples"] = sample_items[:limit]
        return page

    def _receipts(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
                   fencing_token, completed_at, validation_run_id,
                   evidence_digest, body_json
            FROM completion_receipts
            ORDER BY completed_at ASC, receipt_cid ASC
            """,
            transform=lambda row: {
                "receipt_id": row.get("receipt_cid"),
                "receipt_cid": row.get("receipt_cid"),
                "task_cid": row.get("task_cid"),
                "goal_cid": row.get("goal_cid"),
                "attempt_id": row.get("attempt_id"),
                "claim_cid": row.get("claim_cid"),
                "fencing_token": row.get("fencing_token"),
                "completed_at": row.get("completed_at"),
                "validation_run_id": row.get("validation_run_id"),
                "evidence_digest": row.get("evidence_digest"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _worktrees(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT worktree_id, repository_id, path, head_commit_id, branch_name,
                   owner_session_id, status, created_at, updated_at, revision,
                   fence_epoch
            FROM worktrees
            ORDER BY worktree_id ASC
            """,
        )

    def _mutations(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT mutation_id, task_cid, attempt_id, before_snapshot_id,
                   after_snapshot_id, status, created_at, body_json
            FROM mutations
            ORDER BY created_at ASC, mutation_id ASC
            """,
            transform=lambda row: {
                "mutation_id": row.get("mutation_id"),
                "task_cid": row.get("task_cid"),
                "attempt_id": row.get("attempt_id"),
                "before_snapshot_id": row.get("before_snapshot_id"),
                "after_snapshot_id": row.get("after_snapshot_id"),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _ast_summary(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT node_id, snapshot_id, file_id, parent_node_id, node_kind,
                   node_path, fingerprint, start_byte, end_byte
            FROM ast_nodes
            ORDER BY file_id ASC, start_byte ASC, node_id ASC
            """,
        )

    def _bundles(self, request: OperationRequest) -> dict[str, Any]:
        # Bundle index is projected from goals grouped by objective when no
        # dedicated bundle table is populated.
        return self._windowed(
            request,
            """
            SELECT objective_id AS bundle_id,
                   COUNT(*) AS goal_count,
                   MIN(status) AS sample_status
            FROM goals
            GROUP BY objective_id
            ORDER BY objective_id ASC
            """,
        )

    def _cache_inspect(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            """
            SELECT cache_key, task_cid, manifest_cid, decision_digest,
                   created_at, expires_at, hit_count, body_json
            FROM decision_cache_entries
            ORDER BY created_at ASC, cache_key ASC
            """,
            transform=lambda row: {
                "cache_key": row.get("cache_key"),
                "task_cid": row.get("task_cid"),
                "manifest_cid": row.get("manifest_cid"),
                "decision_digest": row.get("decision_digest"),
                "created_at": row.get("created_at"),
                "expires_at": row.get("expires_at"),
                "hit_count": row.get("hit_count"),
                "body": _json_loads(row.get("body_json")),
            },
        )

    def _artifact_query(self, request: OperationRequest) -> dict[str, Any]:
        _reject_raw_sql(request)
        resource = _resource_kind(
            request.parameters.get("resource") or request.parameters.get("table"),
            default="worktrees",
        )
        dispatch = {
            "worktrees": self._worktrees,
            "mutations": self._mutations,
            "ast": self._ast_summary,
            "ast_nodes": self._ast_summary,
            "daemons": self._daemons,
            "daemon_instances": self._daemons,
            "logs": self._logs,
            "structured_logs": self._logs,
            "receipts": self._receipts,
            "completion_receipts": self._receipts,
            "goals": self._goals,
            "tasks": self._tasks,
            "events": self._events,
            "metrics": self._metrics,
            "bundles": self._bundles,
            "lanes": self._lanes,
            "cache": self._cache_inspect,
            "decision_cache_entries": self._cache_inspect,
        }
        handler = dispatch.get(resource)
        if handler is None:
            raise OperationUnavailableError(
                f"artifact resource {resource!r} is not supported"
            )
        result = handler(request)
        return {"resource": resource, **result}

    def _capabilities(self, request: OperationRequest) -> dict[str, Any]:
        del request  # capabilities are static for this backend
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "operations": [item.value for item in self.registered_operations],
            "read_authority": sorted(item.value for item in READ_OPERATIONS),
            "mutation_authority": sorted(
                item.value for item in MUTATION_OPERATIONS if item in _BUILTIN_OPERATIONS
            ),
            "database_path": str(self._path),
            "shell_out": False,
            "raw_sql": False,
            "side_effect_free_discovery": True,
        }

    # -- lifecycle mutations ------------------------------------------------

    def _append_domain_event(
        self,
        *,
        event_type: str,
        body: Mapping[str, Any],
        task_cid: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            self._event_sequence += 1
            sequence = self._event_sequence
        # Reconcile with durable max sequence.
        head = self._fetch_one(
            "SELECT COALESCE(MAX(global_sequence), 0) AS head FROM domain_events"
        )
        head_value = int((head or {}).get("head") or 0)
        if sequence <= head_value:
            sequence = head_value + 1
            with self._lock:
                self._event_sequence = sequence
        event_id = f"event:{uuid.uuid4()}"
        recorded_at = _utc_now()
        redacted_body = _redact(dict(body))
        self._execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?)
            """,
            (
                event_id,
                DEFAULT_STREAM_ID,
                sequence,
                sequence,
                event_type,
                task_cid,
                session_id,
                recorded_at,
                _json_dumps(redacted_body),
            ),
        )
        return {
            "event_id": event_id,
            "stream_id": DEFAULT_STREAM_ID,
            "sequence": sequence,
            "global_sequence": sequence,
            "event_type": event_type,
            "recorded_at": recorded_at,
            "body": redacted_body,
            "schema": DATABASE_LIFECYCLE_EVENT_SCHEMA,
        }

    def _lifecycle(
        self, request: OperationRequest
    ) -> BackendResponse:
        operation = request.operation
        if operation not in _ACTION_REQUESTED_STATE:
            raise OperationUnavailableError(
                f"operation {operation.value} is not a lifecycle action"
            )
        target_id = _target_id(request)
        requested = _ACTION_REQUESTED_STATE[operation]
        requested_value = str(request.parameters.get("requested_state") or "").strip()
        if requested_value and requested_value not in {
            requested.value,
            operation.value,
        }:
            raise InvalidLifecycleTransitionError(
                f"{operation.value} requests {requested.value}, not {requested_value}"
            )
        reason = str(request.parameters.get("reason") or operation.value).strip()
        row = self._ensure_supervisor_row(target_id)
        extension = _json_loads(row.get("extension_json"))
        if not isinstance(extension, Mapping):
            extension = {}
        previous_state = SupervisorLifecycleState(
            str(row.get("status") or SupervisorLifecycleState.STOPPED.value)
        )
        current_fence = int(extension.get("fence_epoch") or 0)
        if (
            request.fencing_epoch is not None
            and request.fencing_epoch < current_fence
        ):
            raise DatabaseSupervisorBackendConflictError(
                f"stale fencing epoch {request.fencing_epoch}; "
                f"current epoch is {current_fence}"
            )

        # Exact request-id replay is coalesced without a second transition.
        priors = self._fetch_all(
            """
            SELECT event_id, body_json FROM domain_events
            WHERE event_type = 'lifecycle.transition'
            ORDER BY global_sequence DESC
            LIMIT 64
            """
        )
        for prior in priors:
            body = _json_loads(prior.get("body_json"))
            if (
                isinstance(body, Mapping)
                and body.get("request_id") == request.request_id
                and body.get("accepted") is True
            ):
                return BackendResponse(
                    data={
                        "status": self._status_payload(request),
                        "event": body,
                        "previous_state": body.get("previous_state"),
                        "state": body.get("state"),
                        "accepted": True,
                        "idempotent": True,
                    },
                    changed=False,
                    applied_effect_ids=(),
                )

        if previous_state is requested or (
            operation is Operation.STOP
            and previous_state is SupervisorLifecycleState.STOPPED
        ) or (
            operation is Operation.PAUSE
            and previous_state is SupervisorLifecycleState.PAUSED
        ) or (
            operation is Operation.RESUME
            and previous_state is SupervisorLifecycleState.HEALTHY
        ) or (
            operation is Operation.DRAIN
            and previous_state
            in {
                SupervisorLifecycleState.DRAINING,
                SupervisorLifecycleState.STOPPED,
            }
        ) or (
            operation in {Operation.STOP, Operation.CANCEL}
            and previous_state
            in {
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.STOPPED,
            }
        ) or (
            operation is Operation.RETRY
            and previous_state is SupervisorLifecycleState.STARTING
        ) or (
            operation is Operation.QUARANTINE
            and previous_state is SupervisorLifecycleState.BLOCKED
        ):
            event = self._append_domain_event(
                event_type="lifecycle.transition",
                body={
                    "request_id": request.request_id,
                    "target_id": target_id,
                    "action": operation.value,
                    "previous_state": previous_state.value,
                    "state": previous_state.value,
                    "accepted": True,
                    "changed": False,
                    "reason": reason,
                    "fence_epoch": request.fencing_epoch,
                },
                session_id=str(request.parameters.get("session_id") or ""),
            )
            return BackendResponse(
                data={
                    "status": self._status_payload(request),
                    "event": event,
                    "previous_state": previous_state.value,
                    "state": previous_state.value,
                    "accepted": True,
                    "idempotent": True,
                },
                changed=False,
                applied_effect_ids=(),
            )

        if not lifecycle_transition_is_legal(previous_state, requested):
            raise InvalidLifecycleTransitionError(
                f"invalid transition {previous_state.value} -> {requested.value}: "
                f"{reason}"
            )

        generation = int(extension.get("generation") or 0)
        if operation in {Operation.START, Operation.RETRY}:
            generation += 1
        fence_epoch = (
            int(request.fencing_epoch)
            if request.fencing_epoch is not None
            else current_fence
        )
        now = _utc_now()
        stopped_at = (
            now
            if requested
            in {
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.STOPPED,
                SupervisorLifecycleState.FAILED,
            }
            else None
        )
        new_extension = {
            "generation": generation,
            "fence_epoch": fence_epoch,
            "phase": str(request.parameters.get("phase") or requested.value),
            "reason": reason,
            "updated_at_ms": self._clock_ms(),
        }
        revision = int(row.get("revision") or 0) + 1
        started_at = row.get("started_at") or now
        if operation in {Operation.START, Operation.RETRY}:
            started_at = now
        self._execute(
            """
            UPDATE supervisor_instances
            SET status = ?,
                stopped_at = ?,
                revision = ?,
                extension_json = ?,
                started_at = ?
            WHERE supervisor_id = ?
            """,
            (
                requested.value,
                stopped_at,
                revision,
                _json_dumps(new_extension),
                started_at,
                target_id,
            ),
        )
        # Promote STARTING -> HEALTHY when the operator supplies an explicit
        # ready flag so status/health/stop are usable without a second process.
        final_state = requested
        if (
            requested is SupervisorLifecycleState.STARTING
            and bool(request.parameters.get("ready", False))
        ):
            final_state = SupervisorLifecycleState.HEALTHY
            self._execute(
                """
                UPDATE supervisor_instances
                SET status = ?, revision = revision + 1
                WHERE supervisor_id = ?
                """,
                (final_state.value, target_id),
            )

        event = self._append_domain_event(
            event_type="lifecycle.transition",
            body={
                "request_id": request.request_id,
                "target_id": target_id,
                "action": operation.value,
                "previous_state": previous_state.value,
                "state": final_state.value,
                "accepted": True,
                "changed": True,
                "reason": reason,
                "fence_epoch": fence_epoch,
                "generation": generation,
            },
            session_id=str(request.parameters.get("session_id") or ""),
        )
        # Record a health sample for observability.
        self._execute(
            """
            INSERT INTO health_samples (
                sample_id, subject_kind, subject_id, observed_at, status, body_json
            ) VALUES (?, 'supervisor', ?, ?, ?, ?)
            """,
            (
                f"health:{uuid.uuid4()}",
                target_id,
                now,
                final_state.value,
                _json_dumps({"action": operation.value, "reason": reason}),
            ),
        )
        status = self._status_payload(request)
        return BackendResponse(
            data={
                "status": status,
                "event": event,
                "previous_state": previous_state.value,
                "state": final_state.value,
                "accepted": True,
                "idempotent": False,
            },
            changed=True,
            applied_effect_ids=tuple(
                item.effect_id for item in request.expected_effects
            ),
        )

    # -- public execute -----------------------------------------------------

    def execute(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any]]:
        """Dispatch one typed control operation against the database."""

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        _reject_raw_sql(request)
        operation = request.operation
        if operation is Operation.CAPABILITIES:
            return BackendResponse(data=self._capabilities(request))
        if operation is Operation.STATUS:
            return BackendResponse(data=self._status_payload(request))
        if operation is Operation.HEALTH:
            return BackendResponse(data=self._health_payload(request))
        if operation is Operation.METRICS:
            return BackendResponse(data=self._metrics(request))
        if operation is Operation.GOALS:
            return BackendResponse(data=self._goals(request))
        if operation is Operation.TASKS:
            return BackendResponse(data=self._tasks(request))
        if operation is Operation.BUNDLES:
            return BackendResponse(data=self._bundles(request))
        if operation is Operation.LANES:
            return BackendResponse(data=self._lanes(request))
        if operation is Operation.EVENTS:
            return BackendResponse(data=self._events(request))
        if operation is Operation.RECEIPTS:
            return BackendResponse(data=self._receipts(request))
        if operation is Operation.CACHE_INSPECT:
            return BackendResponse(data=self._cache_inspect(request))
        if operation is Operation.ARTIFACT_QUERY:
            return BackendResponse(data=self._artifact_query(request))
        if operation in _LIFECYCLE_OPERATIONS:
            return self._lifecycle(request)
        raise OperationUnavailableError(
            f"operation {operation.value} has no database adapter"
        )

    # -- seed helpers used by operations / tests ----------------------------

    def append_log(
        self,
        *,
        message: str,
        severity: str = "info",
        component: str = "control",
        body: Mapping[str, Any] | None = None,
        task_cid: str = "",
    ) -> dict[str, Any]:
        """Append one structured log row (operator/tooling helper)."""

        log_id = f"log:{uuid.uuid4()}"
        recorded_at = _utc_now()
        redacted = _redact(dict(body or {}))
        self._execute(
            """
            INSERT INTO structured_logs (
                log_id, severity, component, trace_id, span_id, task_cid,
                attempt_id, session_id, recorded_at, message, body_json
            ) VALUES (?, ?, ?, '', '', ?, '', '', ?, ?, ?)
            """,
            (
                log_id,
                severity,
                component,
                task_cid,
                recorded_at,
                message,
                _json_dumps(redacted),
            ),
        )
        return {
            "log_id": log_id,
            "severity": severity,
            "component": component,
            "recorded_at": recorded_at,
            "message": message,
            "body": redacted,
        }

    def record_backup(
        self,
        *,
        destination_uri: str,
        artifact_digest: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a backup snapshot receipt in the control-plane schema."""

        backup_id = f"backup:{uuid.uuid4()}"
        created_at = _utc_now()
        payload = _redact(dict(body or {}))
        self._execute(
            """
            INSERT INTO backup_snapshots (
                backup_id, store_id, database_uuid, schema_revision, generation,
                artifact_digest, created_at, destination_uri, status, body_json
            ) VALUES (?, ?, ?, 1, 1, ?, ?, ?, 'completed', ?)
            """,
            (
                backup_id,
                "store:control",
                "00000000-0000-4000-8000-000000000001",
                artifact_digest,
                created_at,
                destination_uri,
                _json_dumps(payload),
            ),
        )
        return {
            "backup_id": backup_id,
            "destination_uri": destination_uri,
            "artifact_digest": artifact_digest,
            "created_at": created_at,
            "status": "completed",
            "body": payload,
            "authority": "export",
        }


def open_database_supervisor_backend(
    database_path: Path | str,
    **kwargs: Any,
) -> DatabaseSupervisorBackend:
    """Open a :class:`DatabaseSupervisorBackend` on ``database_path``."""

    return DatabaseSupervisorBackend(database_path, **kwargs)


__all__ = (
    "DATABASE_LIFECYCLE_EVENT_SCHEMA",
    "DATABASE_PROGRAM_TARGET_SCHEMA",
    "DATABASE_SUPERVISOR_BACKEND_INTERFACE",
    "DATABASE_SUPERVISOR_BACKEND_SCHEMA",
    "DATABASE_SUPERVISOR_BACKEND_VERSION",
    "DatabaseProgramTarget",
    "DatabaseSupervisorBackend",
    "DatabaseSupervisorBackendBoundsError",
    "DatabaseSupervisorBackendConflictError",
    "DatabaseSupervisorBackendError",
    "DatabaseSupervisorBackendUnavailableError",
    "duckdb_available",
    "open_database_supervisor_backend",
)
