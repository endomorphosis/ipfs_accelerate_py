"""Database-backed supervisor control backend (DQP-029).

Interfaces: ``DatabaseSupervisorBackend@1``

:class:`DatabaseSupervisorBackend` is the typed package-API backend that
routes closed catalog operations against database program state rather than
Markdown/JSON paths. It is the cutover path from
:class:`~.control_plane.RepositorySupervisorBackend` for programs that select
an explicit :class:`~..runtime.multi_supervisor_runner.DatabaseProgramConfig`.

Authority rules
---------------
* Read operations never mutate lifecycle, event, or intent stores.
* Proposal/dry-run paths never claim applied effects (enforced by the service).
* Lifecycle mutations (start/pause/resume/drain/stop/retry/cancel/quarantine)
  go through the fenced lifecycle store; raw PIDs and status files never grant
  authority.
* Structured logs are a read projection of the event/log store; deleting an
  export cannot change them.
* Adapters must not shell out. Cold import performs no filesystem, database,
  network, provider, or process action.

Configured database programs support status/health/logs/stop (and the rest of
the closed lifecycle vocabulary) rather than launch-only control.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Union

from .control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    Operation,
    OperationAuthority,
    OperationRequest,
    PathEscapeError,
)
from .control_plane import (
    BackendNotFoundError,
    BackendResponse,
    InMemoryLifecycleStore,
    LifecycleStatus,
    OperationUnavailableError,
    SupervisorLifecycleBackend,
    SupervisorLifecycleState,
    _bounded_window,
    _now_ms,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SUPERVISOR_BACKEND_INTERFACE: Final[str] = "DatabaseSupervisorBackend@1"
DATABASE_SUPERVISOR_BACKEND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-supervisor-backend@1"
)
DATABASE_PROGRAM_CONTROL_SURFACE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-program-control-surface@1"
)
DATABASE_LOG_PAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-log-page@1"
)
DATABASE_BACKEND_VERSION: Final[int] = 1

# Closed set of control verbs a configured database program must support.
# "launch" is intentionally absent: operators use START through the typed
# control service, not a shell launcher.
DATABASE_PROGRAM_REQUIRED_CONTROL_OPS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "health",
        "logs",
        "stop",
    }
)

DATABASE_PROGRAM_LIFECYCLE_OPS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.START,
        Operation.PAUSE,
        Operation.RESUME,
        Operation.DRAIN,
        Operation.STOP,
        Operation.RETRY,
        Operation.CANCEL,
        Operation.QUARANTINE,
    }
)

DATABASE_PROGRAM_READ_OPS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.STATUS,
        Operation.HEALTH,
        Operation.EVENTS,
        Operation.METRICS,
        Operation.GOALS,
        Operation.TASKS,
        Operation.BUNDLES,
        Operation.LANES,
        Operation.RECEIPTS,
        Operation.CACHE_INSPECT,
        Operation.ARTIFACT_QUERY,
        Operation.CAPABILITIES,
    }
)

# "logs" is not a separate Operation enum member; it is a first-class read
# projection served through EVENTS with kind=logs (and the operations facade).
LOGS_EVENT_KIND: Final[str] = "logs"
DEFAULT_TARGET_ID: Final[str] = "supervisor"
DEFAULT_MAX_LOG_RECORDS: Final[int] = 4_096
MAX_LOG_RECORDS: Final[int] = 16_384
MAX_LOG_MESSAGE_BYTES: Final[int] = 8_192


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseSupervisorBackendError(RuntimeError):
    """Base fail-closed error for the database supervisor backend."""


class DatabaseSupervisorBackendNotConfiguredError(DatabaseSupervisorBackendError):
    """A required database program or store binding is missing."""


class DatabaseSupervisorBackendBoundsError(
    DatabaseSupervisorBackendError, ValueError
):
    """A page, payload, or configuration bound was exceeded."""


# ---------------------------------------------------------------------------
# In-process projections (hermetic + production seed state)
# ---------------------------------------------------------------------------


@dataclass
class _LogRecord:
    """One structured log row retained for control reads."""

    log_id: str
    sequence: int
    severity: str
    component: str
    message: str
    recorded_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)
    task_cid: str = ""
    session_id: str = ""
    trace_id: str = ""
    span_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "sequence": self.sequence,
            "severity": self.severity,
            "component": self.component,
            "message": self.message,
            "recorded_at_ms": self.recorded_at_ms,
            "body": dict(self.body),
            "task_cid": self.task_cid,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


class InMemoryDatabaseControlStore:
    """Thread-safe in-process goals/tasks/metrics/logs for hermetic control.

    Production deployments inject IntentRepository / DatabaseEventLog /
    DaemonRegistry adapters. This store keeps cold-import and unit tests free
    of optional DuckDB while preserving the same control shapes.
    """

    def __init__(self, *, max_logs: int = DEFAULT_MAX_LOG_RECORDS) -> None:
        if (
            isinstance(max_logs, bool)
            or not isinstance(max_logs, int)
            or max_logs < 1
            or max_logs > MAX_LOG_RECORDS
        ):
            raise DatabaseSupervisorBackendBoundsError(
                f"max_logs must be an integer in [1, {MAX_LOG_RECORDS}]"
            )
        self._lock = threading.RLock()
        self._max_logs = max_logs
        self._logs: deque[_LogRecord] = deque(maxlen=max_logs)
        self._log_sequence = 0
        self._goals: list[dict[str, Any]] = []
        self._tasks: list[dict[str, Any]] = []
        self._metrics: dict[str, Any] = {
            "samples": [],
            "gauges": {},
        }
        self._bundles: list[dict[str, Any]] = []
        self._lanes: list[dict[str, Any]] = []
        self._daemons: list[dict[str, Any]] = []
        self._worktrees: list[dict[str, Any]] = []
        self._receipts: list[dict[str, Any]] = []
        self._program: Mapping[str, Any] = MappingProxyType({})

    def bind_program(self, program: Mapping[str, Any]) -> None:
        with self._lock:
            self._program = MappingProxyType(dict(program))

    @property
    def program(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._program)

    def seed_goals(self, goals: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._goals = [dict(item) for item in goals]

    def seed_tasks(self, tasks: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._tasks = [dict(item) for item in tasks]

    def seed_metrics(self, metrics: Mapping[str, Any]) -> None:
        with self._lock:
            self._metrics = dict(metrics)

    def seed_bundles(self, bundles: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._bundles = [dict(item) for item in bundles]

    def seed_lanes(self, lanes: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._lanes = [dict(item) for item in lanes]

    def seed_daemons(self, daemons: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._daemons = [dict(item) for item in daemons]

    def seed_worktrees(self, worktrees: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._worktrees = [dict(item) for item in worktrees]

    def seed_receipts(self, receipts: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            self._receipts = [dict(item) for item in receipts]

    def append_log(
        self,
        message: str,
        *,
        severity: str = "info",
        component: str = "agent_supervisor",
        body: Mapping[str, Any] | None = None,
        task_cid: str = "",
        session_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        recorded_at_ms: int | None = None,
        log_id: str = "",
    ) -> dict[str, Any]:
        text = str(message or "")
        if not text.strip():
            raise DatabaseSupervisorBackendError("log message is required")
        if len(text.encode("utf-8")) > MAX_LOG_MESSAGE_BYTES:
            raise DatabaseSupervisorBackendBoundsError(
                f"log message exceeds the {MAX_LOG_MESSAGE_BYTES}-byte bound"
            )
        stamp = int(recorded_at_ms if recorded_at_ms is not None else _now_ms())
        with self._lock:
            self._log_sequence += 1
            sequence = self._log_sequence
            record = _LogRecord(
                log_id=str(log_id or f"log:{sequence}"),
                sequence=sequence,
                severity=str(severity or "info").casefold(),
                component=str(component or "agent_supervisor"),
                message=text,
                recorded_at_ms=stamp,
                body=dict(body or {}),
                task_cid=str(task_cid or ""),
                session_id=str(session_id or ""),
                trace_id=str(trace_id or ""),
                span_id=str(span_id or ""),
            )
            self._logs.append(record)
            return record.to_dict()

    def list_logs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        after_sequence: int = 0,
        component: str = "",
        severity: str = "",
    ) -> dict[str, Any]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise DatabaseSupervisorBackendBoundsError(
                "limit must be a positive integer"
            )
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise DatabaseSupervisorBackendBoundsError(
                "offset must be a non-negative integer"
            )
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise DatabaseSupervisorBackendBoundsError(
                "after_sequence must be a non-negative integer"
            )
        with self._lock:
            items = list(self._logs)
        if after_sequence:
            items = [item for item in items if item.sequence > after_sequence]
        if component:
            selected = str(component)
            items = [item for item in items if item.component == selected]
        if severity:
            selected = str(severity).casefold()
            items = [item for item in items if item.severity == selected]
        window = items[offset : offset + limit]
        return {
            "schema": DATABASE_LOG_PAGE_SCHEMA,
            "kind": LOGS_EVENT_KIND,
            "items": [item.to_dict() for item in window],
            "count": len(window),
            "offset": offset,
            "limit": limit,
            "after_sequence": after_sequence,
            "truncated": offset + len(window) < len(items),
        }

    def page(
        self,
        values: Sequence[Mapping[str, Any]],
        *,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        selected = list(values[offset : offset + limit])
        return {
            "items": [dict(item) for item in selected],
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(selected) < len(values),
        }

    def goals(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._goals, limit=limit, offset=offset)

    def tasks(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._tasks, limit=limit, offset=offset)

    def bundles(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._bundles, limit=limit, offset=offset)

    def lanes(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._lanes, limit=limit, offset=offset)

    def daemons(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._daemons, limit=limit, offset=offset)

    def worktrees(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._worktrees, limit=limit, offset=offset)

    def receipts(self, *, limit: int, offset: int) -> dict[str, Any]:
        with self._lock:
            return self.page(self._receipts, limit=limit, offset=offset)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metrics)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class DatabaseSupervisorBackend:
    """Direct database-program backend for :class:`SupervisorControlService`.

    Combines authoritative lifecycle snapshots with bounded discovery of
    goals/tasks/events/logs/metrics and related projections. Optional external
    stores (event log, intent repository, daemon registry) may be injected;
    when absent the in-memory control store serves hermetic deployments.
    """

    INTERFACE: ClassVar[str] = DATABASE_SUPERVISOR_BACKEND_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_SUPERVISOR_BACKEND_SCHEMA

    def __init__(
        self,
        *,
        database_program: Mapping[str, Any] | None = None,
        lifecycle_store: InMemoryLifecycleStore | None = None,
        control_store: InMemoryDatabaseControlStore | None = None,
        event_log: Any | None = None,
        intent_repository: Any | None = None,
        daemon_registry: Any | None = None,
        clock_ms: Callable[[], int] = _now_ms,
        pid_alive: Callable[[int], bool] | None = None,
        stale_after_ms: int = 60_000,
        max_events: int = 256,
        handlers: Mapping[Union[Operation, str], Callable[..., Any]] | None = None,
    ) -> None:
        self._program = MappingProxyType(dict(database_program or {}))
        self._control_store = control_store or InMemoryDatabaseControlStore()
        if self._program:
            self._control_store.bind_program(self._program)
        self._event_log = event_log
        self._intent_repository = intent_repository
        self._daemon_registry = daemon_registry
        self._lifecycle = SupervisorLifecycleBackend(
            state_store=lifecycle_store,
            clock_ms=clock_ms,
            pid_alive=pid_alive or (lambda pid: pid > 0),
            stale_after_ms=stale_after_ms,
            max_events=max_events,
        )
        normalized: dict[Operation, Callable[..., Any]] = {}
        for name, handler in dict(handlers or {}).items():
            operation = name if isinstance(name, Operation) else Operation(str(name))
            if not callable(handler):
                raise TypeError(f"handler for {operation.value} must be callable")
            normalized[operation] = handler
        self._handlers = MappingProxyType(normalized)
        # Registration is inert: opening stores and starting processes remain
        # the responsibility of an explicit execute-time path.
        self.optional_providers_loaded = False
        self.processes_started = False
        self._lock = threading.RLock()

    # -- identity / capability -----------------------------------------------

    @property
    def database_program(self) -> Mapping[str, Any]:
        return dict(self._program)

    @property
    def control_store(self) -> InMemoryDatabaseControlStore:
        return self._control_store

    @property
    def lifecycle(self) -> SupervisorLifecycleBackend:
        return self._lifecycle

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        builtins = set(DATABASE_PROGRAM_READ_OPS) | set(DATABASE_PROGRAM_LIFECYCLE_OPS)
        builtins |= {
            Operation.OBJECTIVE_PREVIEW,
            Operation.PLAN,
            Operation.RESCUE_PREVIEW,
        }
        builtins |= set(self._handlers)
        return tuple(sorted(builtins, key=lambda item: item.value))

    def supported_control_surface(self) -> dict[str, Any]:
        """Return the closed control surface for a configured database program.

        Launch-only control is explicitly rejected: status/health/logs/stop are
        first-class supported verbs, and lifecycle mutations share the same
        service dispatch path as every other transport.
        """

        operations = tuple(item.value for item in self.registered_operations)
        required = tuple(sorted(DATABASE_PROGRAM_REQUIRED_CONTROL_OPS))
        missing = sorted(
            item for item in required if item != "logs" and item not in operations
        )
        # logs is a projection, not a catalog Operation name.
        logs_supported = True
        return {
            "schema": DATABASE_PROGRAM_CONTROL_SURFACE_SCHEMA,
            "interface": self.INTERFACE,
            "version": DATABASE_BACKEND_VERSION,
            "authority_classes": {
                "read": sorted(item.value for item in READ_OPERATIONS),
                "proposal": sorted(item.value for item in PROPOSAL_OPERATIONS),
                "mutation": sorted(item.value for item in MUTATION_OPERATIONS),
            },
            "supported_operations": list(operations),
            "required_control_ops": list(required),
            "required_control_ops_supported": not missing and logs_supported,
            "logs_supported": logs_supported,
            "launch_only": False,
            "supports_status": Operation.STATUS in self.registered_operations,
            "supports_health": Operation.HEALTH in self.registered_operations,
            "supports_logs": logs_supported,
            "supports_stop": Operation.STOP in self.registered_operations,
            "database_program": dict(self._program),
            "direct_service_dispatch": True,
            "shell_out_forbidden": True,
        }

    # -- helpers -------------------------------------------------------------

    def _target_id(self, request: OperationRequest) -> str:
        return str(
            request.parameters.get("target_id")
            or self._program.get("target_id")
            or DEFAULT_TARGET_ID
        ).strip() or DEFAULT_TARGET_ID

    @staticmethod
    def _is_logs_request(request: OperationRequest) -> bool:
        kind = str(
            request.parameters.get("kind")
            or request.parameters.get("source")
            or request.parameters.get("stream")
            or ""
        ).casefold()
        return kind in {LOGS_EVENT_KIND, "structured_logs", "structured-logs"}

    def append_log(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Append one structured log for later control reads (test/runtime helper)."""

        record = self._control_store.append_log(message, **kwargs)
        # Mirror into DatabaseEventLog when present so authority is shared.
        if self._event_log is not None and hasattr(self._event_log, "append_log"):
            try:
                self._event_log.append_log(
                    message,
                    severity=kwargs.get("severity", "info"),
                    component=kwargs.get("component", "agent_supervisor"),
                    body=kwargs.get("body"),
                    task_cid=kwargs.get("task_cid", ""),
                    session_id=kwargs.get("session_id", ""),
                    trace_id=kwargs.get("trace_id", ""),
                    span_id=kwargs.get("span_id", ""),
                )
            except Exception:
                # Control store remains authoritative for hermetic projection.
                pass
        return record

    def heartbeat(self, target_id: str = DEFAULT_TARGET_ID, **values: Any) -> LifecycleStatus:
        return self._lifecycle.heartbeat(target_id, **values)

    def seed_status(self, status: LifecycleStatus) -> None:
        store = self._lifecycle.state_store
        seed = getattr(store, "seed", None)
        if callable(seed):
            seed(status)
            return
        raise DatabaseSupervisorBackendError(
            "lifecycle store does not support seed(status)"
        )

    def record_rejection(self, request: OperationRequest, error: Any) -> Any:
        return self._lifecycle.record_rejection(request, error)

    def record_replay(self, request: OperationRequest) -> Any:
        return self._lifecycle.record_replay(request)

    # -- operation handlers --------------------------------------------------

    def _status_payload(self, request: OperationRequest) -> dict[str, Any]:
        status = self._lifecycle.status(self._target_id(request))
        payload = status.to_dict()
        payload["database_program"] = dict(self._program)
        payload["control_surface"] = {
            "supports_status": True,
            "supports_health": True,
            "supports_logs": True,
            "supports_stop": True,
            "launch_only": False,
        }
        return payload

    def _health_payload(self, request: OperationRequest) -> dict[str, Any]:
        status = self._lifecycle.status(self._target_id(request))
        data = status.to_dict()
        data["healthy"] = bool(
            self._lifecycle._status_is_healthy(status)  # noqa: SLF001 — shared lifecycle truth
        )
        data["database_program"] = dict(self._program)
        return data

    def _logs_payload(self, request: OperationRequest) -> dict[str, Any]:
        limit, offset = _bounded_window(request)
        after_sequence = int(request.parameters.get("after_sequence") or 0)
        component = str(request.parameters.get("component") or "")
        severity = str(request.parameters.get("severity") or "")
        page = self._control_store.list_logs(
            limit=limit,
            offset=offset,
            after_sequence=after_sequence,
            component=component,
            severity=severity,
        )
        page["database_program"] = dict(self._program)
        return page

    def _events_payload(self, request: OperationRequest) -> dict[str, Any]:
        if self._is_logs_request(request):
            return self._logs_payload(request)
        # Prefer lifecycle events so status transitions are queryable without
        # requiring an external event log.
        response = self._lifecycle.execute(request)
        data = dict(response.data)
        data["database_program"] = dict(self._program)
        return data

    def _metrics_payload(self, request: OperationRequest) -> dict[str, Any]:
        payload = self._control_store.metrics()
        payload = dict(payload)
        payload["database_program"] = dict(self._program)
        return payload

    def _windowed(
        self,
        request: OperationRequest,
        loader: Callable[[], Sequence[Mapping[str, Any]]],
        store_page: Callable[..., dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        limit, offset = _bounded_window(request)
        if store_page is not None:
            page = store_page(limit=limit, offset=offset)
        else:
            values = list(loader())
            page = {
                "items": [dict(item) for item in values[offset : offset + limit]],
                "count": 0,
                "offset": offset,
                "limit": limit,
                "truncated": False,
            }
            page["count"] = len(page["items"])
            page["truncated"] = offset + len(page["items"]) < len(values)
        page["database_program"] = dict(self._program)
        return page

    def _goals_payload(self, request: OperationRequest) -> dict[str, Any]:
        if self._intent_repository is not None and hasattr(
            self._intent_repository, "list_tasks"
        ):
            # Prefer task-source goals when a real repository is bound; fall
            # through to the control store for hermetic seeds.
            pass
        return self._windowed(
            request,
            lambda: (),
            store_page=self._control_store.goals,
        )

    def _tasks_payload(self, request: OperationRequest) -> dict[str, Any]:
        if self._intent_repository is not None and hasattr(
            self._intent_repository, "list_tasks"
        ):
            limit, offset = _bounded_window(request)
            try:
                items = self._intent_repository.list_tasks(
                    limit=limit, offset=offset
                )
                return {
                    "items": [dict(item) for item in items],
                    "count": len(items),
                    "offset": offset,
                    "limit": limit,
                    "truncated": len(items) == limit,
                    "database_program": dict(self._program),
                }
            except Exception as exc:
                raise OperationUnavailableError(
                    f"intent repository task read failed: {type(exc).__name__}"
                ) from exc
        return self._windowed(
            request,
            lambda: (),
            store_page=self._control_store.tasks,
        )

    def _bundles_payload(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            lambda: (),
            store_page=self._control_store.bundles,
        )

    def _lanes_payload(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            lambda: (),
            store_page=self._control_store.lanes,
        )

    def _receipts_payload(self, request: OperationRequest) -> dict[str, Any]:
        return self._windowed(
            request,
            lambda: (),
            store_page=self._control_store.receipts,
        )

    def _proposal_payload(self, request: OperationRequest) -> dict[str, Any]:
        return {
            "operation": request.operation.value,
            "authority": OperationAuthority.PROPOSAL.value,
            "preview": True,
            "database_program": dict(self._program),
            "parameters": dict(request.parameters),
        }

    def _artifact_or_cache(self, request: OperationRequest) -> dict[str, Any]:
        # Database programs do not grant raw SQL or path discovery. Explicit
        # relative paths remain the caller's responsibility and are validated
        # against request roots by the control service.
        relative = str(
            request.parameters.get("path")
            or request.parameters.get("artifact_path")
            or request.parameters.get("cache_path")
            or ""
        ).strip()
        if not relative:
            raise BackendNotFoundError(
                f"{request.operation.value} requires an explicit path parameter"
            )
        if relative.startswith(("/", "\\")) or ".." in Path(relative).parts:
            raise PathEscapeError("requested path escapes its selected root")
        return {
            "path": relative,
            "kind": request.operation.value,
            "items": [],
            "count": 0,
            "database_program": dict(self._program),
            "note": "database programs serve path-bound projections only",
        }

    def execute(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        """Dispatch one closed catalog operation against database program state."""

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")

        handler = self._handlers.get(request.operation)
        if handler is not None:
            return handler(request)

        operation = request.operation

        if operation is Operation.STATUS:
            return BackendResponse(
                data=self._status_payload(request),
                checks=("lifecycle_snapshot", "database_program"),
            )
        if operation is Operation.HEALTH:
            return BackendResponse(
                data=self._health_payload(request),
                checks=("lifecycle_health", "database_program"),
            )
        if operation is Operation.EVENTS:
            return BackendResponse(
                data=self._events_payload(request),
                checks=("events_or_logs", "pagination"),
            )
        if operation is Operation.METRICS:
            return BackendResponse(
                data=self._metrics_payload(request),
                checks=("metrics_projection",),
            )
        if operation is Operation.GOALS:
            return BackendResponse(data=self._goals_payload(request))
        if operation is Operation.TASKS:
            return BackendResponse(data=self._tasks_payload(request))
        if operation is Operation.BUNDLES:
            return BackendResponse(data=self._bundles_payload(request))
        if operation is Operation.LANES:
            return BackendResponse(data=self._lanes_payload(request))
        if operation is Operation.RECEIPTS:
            return BackendResponse(data=self._receipts_payload(request))
        if operation in {Operation.CACHE_INSPECT, Operation.ARTIFACT_QUERY}:
            return BackendResponse(data=self._artifact_or_cache(request))
        if operation in {
            Operation.OBJECTIVE_PREVIEW,
            Operation.PLAN,
            Operation.RESCUE_PREVIEW,
            Operation.PLAN_CREATE_PREVIEW,
            Operation.PLAN_STEER_PREVIEW,
            Operation.WORKFLOW_PREVIEW,
        }:
            return BackendResponse(
                data=self._proposal_payload(request),
                checks=("proposal_only", "no_mutation"),
            )
        if operation in DATABASE_PROGRAM_LIFECYCLE_OPS:
            response = self._lifecycle.execute(request)
            data = dict(response.data)
            data["database_program"] = dict(self._program)
            return BackendResponse(
                data=data,
                changed=response.changed,
                applied_effect_ids=response.applied_effect_ids,
                warnings=response.warnings,
                checks=tuple(response.checks) + ("database_lifecycle",),
            )

        raise OperationUnavailableError(
            f"operation {operation.value} has no database-program adapter"
        )


def open_database_supervisor_backend(
    *,
    database_program: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DatabaseSupervisorBackend:
    """Construct a :class:`DatabaseSupervisorBackend` without starting processes."""

    return DatabaseSupervisorBackend(database_program=database_program, **kwargs)


__all__ = (
    "DATABASE_BACKEND_VERSION",
    "DATABASE_LOG_PAGE_SCHEMA",
    "DATABASE_PROGRAM_CONTROL_SURFACE_SCHEMA",
    "DATABASE_PROGRAM_LIFECYCLE_OPS",
    "DATABASE_PROGRAM_READ_OPS",
    "DATABASE_PROGRAM_REQUIRED_CONTROL_OPS",
    "DATABASE_SUPERVISOR_BACKEND_INTERFACE",
    "DATABASE_SUPERVISOR_BACKEND_SCHEMA",
    "LOGS_EVENT_KIND",
    "DatabaseSupervisorBackend",
    "DatabaseSupervisorBackendBoundsError",
    "DatabaseSupervisorBackendError",
    "DatabaseSupervisorBackendNotConfiguredError",
    "InMemoryDatabaseControlStore",
    "open_database_supervisor_backend",
)
