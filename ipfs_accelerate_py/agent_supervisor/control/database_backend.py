"""Database-backed supervisor control backend (DQP-029).

Interface: ``DatabaseSupervisorBackend@1``

Adapts :class:`DatabaseControlOperations` to the transport-neutral
:class:`~.control_plane.SupervisorControlService` backend protocol so Python,
CLI, and MCP surfaces share canonical ``OperationRequest`` /
``OperationResult`` identity and direct service dispatch.

Adapters never shell out. Raw credentials are rejected; only opaque secret
handles are admitted. Discovery remains side-effect free. Lifecycle mutations
retain authorization, idempotency, lease, fence, and expected-effect checks at
the service boundary.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Union

from .control_contracts import (
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    Operation,
    OperationError,
    OperationRequest,
)
from .control_plane import (
    BackendResponse,
    OperationUnavailableError,
)
from .database_operations import (
    DATABASE_CONTROL_OPERATIONS_INTERFACE,
    DATABASE_PROGRAM_TARGET_INTERFACE,
    DatabaseControlOperations,
    DatabaseProgramTarget,
    ProgramAuthorityMode,
    open_database_control_operations,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SUPERVISOR_BACKEND_INTERFACE: Final[str] = "DatabaseSupervisorBackend@1"
DATABASE_SUPERVISOR_BACKEND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-supervisor-backend@1"
)
DATABASE_SUPERVISOR_BACKEND_VERSION: Final[int] = 1

# Operations the database backend implements end-to-end.
_DATABASE_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {
        *READ_OPERATIONS,
        *PROPOSAL_OPERATIONS,
        Operation.START,
        Operation.PAUSE,
        Operation.RESUME,
        Operation.DRAIN,
        Operation.STOP,
        Operation.RETRY,
        Operation.CANCEL,
        Operation.QUARANTINE,
        Operation.RESTART,
    }
)


class DatabaseSupervisorBackend:
    """Direct Python backend for configured database supervisor programs.

    Interface: ``DatabaseSupervisorBackend@1``.

    Paths and process effects are never inferred from ambient state. A program
    must be registered explicitly. Discovery and capability handshake do not
    start processes or load optional providers.
    """

    INTERFACE: ClassVar[str] = DATABASE_SUPERVISOR_BACKEND_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_SUPERVISOR_BACKEND_SCHEMA
    VERSION: ClassVar[int] = DATABASE_SUPERVISOR_BACKEND_VERSION

    def __init__(
        self,
        operations: DatabaseControlOperations | None = None,
        *,
        programs: Sequence[DatabaseProgramTarget | Mapping[str, Any]] = (),
        seed: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
        clock_ms: Any | None = None,
        stale_after_ms: int = 60_000,
    ) -> None:
        self._operations = operations or open_database_control_operations(
            clock_ms=clock_ms,
            stale_after_ms=stale_after_ms,
        )
        # Registration is inert: no process start, no provider load.
        self.optional_providers_loaded = False
        self.processes_started = False
        for program in programs:
            program_id = (
                program.program_id
                if isinstance(program, DatabaseProgramTarget)
                else str(program.get("program_id") or "")
            )
            program_seed = None
            if seed and program_id in seed:
                program_seed = seed[program_id]
            self._operations.register_program(program, seed=program_seed)

    @property
    def operations(self) -> DatabaseControlOperations:
        return self._operations

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        return tuple(sorted(_DATABASE_OPERATIONS, key=lambda item: item.value))

    def register_program(
        self,
        target: DatabaseProgramTarget | Mapping[str, Any],
        *,
        seed: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> DatabaseProgramTarget:
        """Register a database program target without starting processes."""

        return self._operations.register_program(target, seed=seed)

    def discover(self) -> Mapping[str, Any]:
        """Side-effect-free discovery of programs and supported domains."""

        return self._operations.discover()

    def record_rejection(
        self, request: OperationRequest, error: OperationError
    ) -> None:
        """Record a denied mutation without applying effects."""

        try:
            program_id = self._operations.resolve_program_id(request.parameters)
        except Exception:
            return
        self._operations.append_log(
            program_id,
            severity="warning",
            component="control",
            message=f"rejected:{error.code.value}",
            body={
                "operation": request.operation.value,
                "error_code": error.code.value,
                "error_message": error.message,
            },
        )

    def record_replay(self, request: OperationRequest) -> None:
        """Record an exact idempotent replay observation."""

        try:
            program_id = self._operations.resolve_program_id(request.parameters)
        except Exception:
            return
        self._operations.append_log(
            program_id,
            severity="info",
            component="control",
            message="idempotent_replay",
            body={
                "operation": request.operation.value,
                "request_id": request.request_id,
            },
        )

    def execute(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any]]:
        """Execute one operation through the typed database control layer."""

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        if request.operation not in _DATABASE_OPERATIONS:
            raise OperationUnavailableError(
                f"operation {request.operation.value} has no database backend adapter"
            )
        response = self._operations.execute_request(request)
        # Mirror process-start flag for capability reports.
        if self._operations.processes_started:
            self.processes_started = True
        return response


def build_database_supervisor_backend(
    *,
    program_id: str = "program:default",
    store_id: str = "control.duckdb",
    authority_mode: str | ProgramAuthorityMode = ProgramAuthorityMode.EMBEDDED,
    endpoint_secret_handle: str = "",
    store_generation: str = "1",
    schema_revision: str = "1",
    repository_id: str = "",
    export_profile: str = "default",
    seed: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    clock_ms: Any | None = None,
    stale_after_ms: int = 60_000,
) -> DatabaseSupervisorBackend:
    """Construct a backend with one registered database program.

    Construction and registration are inert. Callers must issue an authorized
    ``start`` mutation to mark the program running.
    """

    target = DatabaseProgramTarget(
        program_id=program_id,
        store_id=store_id,
        authority_mode=authority_mode,
        endpoint_secret_handle=endpoint_secret_handle,
        store_generation=store_generation,
        schema_revision=schema_revision,
        repository_id=repository_id,
        export_profile=export_profile,
    )
    return DatabaseSupervisorBackend(
        programs=(target,),
        seed={program_id: seed} if seed else None,
        clock_ms=clock_ms,
        stale_after_ms=stale_after_ms,
    )


def database_backend_from_state_root(
    state_root: str | Path,
    *,
    program_id: str = "program:default",
    store_id: str = "control.duckdb",
    authority_mode: str | ProgramAuthorityMode = ProgramAuthorityMode.EMBEDDED,
    endpoint_secret_handle: str = "",
    **kwargs: Any,
) -> DatabaseSupervisorBackend:
    """Bind a backend identity to a state root without opening the database file.

    The state root is recorded only as repository/program identity metadata.
    Opening DuckDB files remains the responsibility of lower-layer repositories
    when a deployment wires them; this backend keeps hermetic in-process
    authority for control status/health/logs/lifecycle.
    """

    root = Path(state_root)
    return build_database_supervisor_backend(
        program_id=program_id,
        store_id=store_id,
        authority_mode=authority_mode,
        endpoint_secret_handle=endpoint_secret_handle,
        repository_id=f"state:{root.as_posix()}",
        **kwargs,
    )


__all__ = (
    "DATABASE_CONTROL_OPERATIONS_INTERFACE",
    "DATABASE_PROGRAM_TARGET_INTERFACE",
    "DATABASE_SUPERVISOR_BACKEND_INTERFACE",
    "DATABASE_SUPERVISOR_BACKEND_SCHEMA",
    "DATABASE_SUPERVISOR_BACKEND_VERSION",
    "DatabaseProgramTarget",
    "DatabaseSupervisorBackend",
    "ProgramAuthorityMode",
    "build_database_supervisor_backend",
    "database_backend_from_state_root",
)
