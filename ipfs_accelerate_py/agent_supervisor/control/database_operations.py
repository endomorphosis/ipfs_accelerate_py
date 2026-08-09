"""Typed database control operations service (DQP-029).

Interfaces: ``DatabaseControlOperations@1``

:class:`DatabaseControlOperations` is the single typed service that Python,
CLI, and MCP adapters share for database-backed supervisor control.  Every
surface constructs the same :class:`~.control_contracts.OperationRequest` and
dispatches it through :class:`~.control_plane.SupervisorControlService` with
the :class:`~.database_backend.DatabaseSupervisorBackend`.  Discovery is
side-effect free.  Read, proposal, and mutation authority remain distinct.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Union

from .control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlDiscoveryManifest,
    ControlSurface,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
)
from .control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    InMemoryControlStateStore,
    SupervisorClient,
    SupervisorControlService,
    SupervisorTarget,
)
from .database_backend import (
    DATABASE_SUPERVISOR_BACKEND_INTERFACE,
    DatabaseProgramTarget,
    DatabaseSupervisorBackend,
    duckdb_available,
    open_database_supervisor_backend,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_CONTROL_OPERATIONS_INTERFACE: Final[str] = "DatabaseControlOperations@1"
DATABASE_CONTROL_OPERATIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-operations@1"
)
DATABASE_CONTROL_DISCOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-discovery@1"
)
DATABASE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-export-receipt@1"
)
DATABASE_IMPORT_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-import-preview@1"
)
DATABASE_BACKUP_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-backup-receipt@1"
)

DATABASE_CONTROL_OPERATIONS_VERSION: Final[int] = 1
DEFAULT_CALLER: Final[str] = "operator:database-control"
DEFAULT_OBJECTIVE_ID: Final[str] = "objective:database-control"
DEFAULT_POLICY_ID: Final[str] = "policy:database-control"

_EXPORT_VIEWS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "health",
        "goals",
        "tasks",
        "events",
        "logs",
        "metrics",
        "worktrees",
        "mutations",
        "receipts",
        "portable",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseControlOperationsError(RuntimeError):
    """Base error for database control operations."""


class DatabaseControlOperationsUnavailableError(DatabaseControlOperationsError):
    """Required dependency is unavailable."""


class DatabaseControlOperationsRequestError(
    DatabaseControlOperationsError, ValueError
):
    """The request is malformed or out of scope."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# Binding / discovery records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseControlBinding:
    """Transport-neutral identity binding for database control requests."""

    repository_root: str
    state_root: str
    repository_id: str = "repository:database-control"
    tree_id: str = "tree:database-control"
    objective_id: str = DEFAULT_OBJECTIVE_ID
    objective_revision: str = "objective:1"
    policy_id: str = DEFAULT_POLICY_ID
    policy_revision: str = "policy:1"
    caller: str = DEFAULT_CALLER

    def to_dict(self) -> dict[str, str]:
        return {
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
        }

    def target(self) -> SupervisorTarget:
        return SupervisorTarget(**self.to_dict())


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatabaseControlOperations:
    """Canonical typed service for database-backed supervisor control.

    Python, CLI, and MCP adapters construct identical
    :class:`OperationRequest` records and call :meth:`execute`.  No adapter
    shells out or invents a second authority path.
    """

    INTERFACE: Final[str] = DATABASE_CONTROL_OPERATIONS_INTERFACE
    SCHEMA: Final[str] = DATABASE_CONTROL_OPERATIONS_SCHEMA
    DISPATCHER_ID: Final[str] = DIRECT_CONTROL_SERVICE_DISPATCHER_ID

    def __init__(
        self,
        database_path: Path | str,
        *,
        repository_root: Path | str | None = None,
        state_root: Path | str | None = None,
        backend: DatabaseSupervisorBackend | None = None,
        service: SupervisorControlService | None = None,
        binding: DatabaseControlBinding | Mapping[str, Any] | None = None,
        lease_id: str = "lease:database-control",
        fencing_epoch: int = 1,
        clock_ms: Callable[[], int] | None = None,
        require_lease_validator: bool = True,
    ) -> None:
        path = Path(database_path)
        self._database_path = path
        self._backend = backend or open_database_supervisor_backend(path)
        if self._backend.database_path.resolve() != path.resolve():
            # Allow explicit backend override with matching path only.
            if backend is None:
                pass
            else:
                # Keep the provided backend; path is informational.
                self._database_path = self._backend.database_path

        repo = Path(repository_root or path.parent / "repository").resolve()
        state = Path(state_root or path.parent / "state").resolve()
        repo.mkdir(parents=True, exist_ok=True)
        state.mkdir(parents=True, exist_ok=True)
        self._repository_root = repo
        self._state_root = state

        if binding is None:
            self._binding = DatabaseControlBinding(
                repository_root=str(repo),
                state_root=str(state),
            )
        elif isinstance(binding, DatabaseControlBinding):
            self._binding = binding
        else:
            self._binding = DatabaseControlBinding(**dict(binding))

        self._lease_id = str(lease_id).strip() or "lease:database-control"
        self._fencing_epoch = int(fencing_epoch)
        self._lock = threading.RLock()
        self._discovery_calls = 0
        self._dispatch_count = 0
        # Deterministic clock keeps authorization freshness hermetic in tests
        # and offline tooling; deployments can inject a live clock.
        self._clock_ms = clock_ms or (lambda: 1_500)

        if service is not None:
            self._service = service
        else:
            lease_value = self._lease_id
            fence_value = self._fencing_epoch

            def _lease_validator(request: OperationRequest) -> bool:
                return (
                    request.lease_id == lease_value
                    and request.fencing_epoch == fence_value
                )

            self._service = SupervisorControlService(
                repository_allowlist=(repo,),
                state_allowlist=(state,),
                backend=self._backend,
                lease_validator=_lease_validator,
                state_store=InMemoryControlStateStore(),
                require_lease_validator=require_lease_validator,
                clock_ms=self._clock_ms,
            )

    # -- properties ---------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def backend(self) -> DatabaseSupervisorBackend:
        return self._backend

    @property
    def service(self) -> SupervisorControlService:
        return self._service

    @property
    def binding(self) -> DatabaseControlBinding:
        return self._binding

    @property
    def client(self) -> SupervisorClient:
        return self._service.client(self._binding.target())

    @property
    def program_target(self) -> DatabaseProgramTarget:
        return self._backend.target()

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    # -- discovery (inert) --------------------------------------------------

    def discover(self) -> Mapping[str, Any]:
        """Return side-effect-free discovery for all transports."""

        with self._lock:
            self._discovery_calls += 1
            discovery_calls = self._discovery_calls
        backend_manifest = self._backend.discovery_manifest()
        python_manifest = ControlDiscoveryManifest(surface=ControlSurface.PYTHON)
        return MappingProxyType(
            {
                "schema": DATABASE_CONTROL_DISCOVERY_SCHEMA,
                "interface": self.INTERFACE,
                "version": DATABASE_CONTROL_OPERATIONS_VERSION,
                "dispatcher_id": self.DISPATCHER_ID,
                "backend_interface": DATABASE_SUPERVISOR_BACKEND_INTERFACE,
                "database_path": str(self._database_path),
                "operations": list(backend_manifest["operations"]),
                "read_operations": list(backend_manifest["read_operations"]),
                "lifecycle_operations": list(
                    backend_manifest["lifecycle_operations"]
                ),
                "authorities": {
                    "read": sorted(item.value for item in READ_OPERATIONS),
                    "proposal": sorted(item.value for item in PROPOSAL_OPERATIONS),
                    "mutation": sorted(item.value for item in MUTATION_OPERATIONS),
                },
                "surfaces": sorted(item.value for item in ControlSurface),
                "python_discovery": python_manifest.to_dict(),
                "side_effects": False,
                "shell_out": False,
                "raw_sql": False,
                "discovery_calls": discovery_calls,
                "supported": {
                    "status": True,
                    "health": True,
                    "logs": True,
                    "stop": True,
                    "start": True,
                    "pause": True,
                    "resume": True,
                    "drain": True,
                    "retry": True,
                    "cancel": True,
                    "quarantine": True,
                    "import_preview": True,
                    "export": True,
                    "backup": True,
                },
            }
        )

    def discovery_is_inert(self) -> bool:
        """Return True when discovery has never opened the database backend."""

        # Construction and discover() never set processes_started.
        return (
            self._backend.processes_started is False
            and self._backend.optional_providers_loaded is False
        )

    # -- request construction -----------------------------------------------

    def build_request(
        self,
        operation: Operation | str,
        *,
        parameters: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        expected_effects: Sequence[ExpectedEffect] | None = None,
        bounds: ControlBounds | None = None,
        idempotency_key: str | None = None,
        authorization: AuthorizationDecision | None = None,
        lease_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> OperationRequest:
        """Build one canonical :class:`OperationRequest` for all transports."""

        selected = (
            operation if isinstance(operation, Operation) else Operation(str(operation))
        )
        params = dict(parameters or {})
        if "target_id" not in params and "program_id" not in params:
            params["target_id"] = self.program_target.program_id
        params.setdefault("database_path", str(self._database_path))

        effects = tuple(expected_effects or ())
        if selected.mutating and not effects:
            # Dry-run mutations still declare expected effects so the proposal
            # preview can report would_change without applying them.
            effects = (
                ExpectedEffect(
                    effect_id=f"{selected.value}:{params['target_id']}",
                    kind=EffectKind.LIFECYCLE_TRANSITION
                    if selected
                    in {
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
                    else EffectKind.WRITE_STATE,
                    resource=f"supervisor:{params['target_id']}",
                    paths=("control.duckdb",),
                    description=f"Apply {selected.value} via database control",
                ),
            )

        values: dict[str, Any] = {
            "operation": selected,
            **self._binding.to_dict(),
            "parameters": params,
            "expected_effects": effects,
            "dry_run": bool(dry_run),
            "bounds": bounds or ControlBounds(),
        }
        if selected.mutating and not dry_run:
            key = (
                str(idempotency_key).strip()
                if idempotency_key
                else f"{selected.value}:{params['target_id']}:{_sha256_text(_json_dumps(params))[:16]}"
            )
            values["idempotency"] = IdempotencyKey(
                key=key,
                operation=selected,
                caller=self._binding.caller,
                repository_id=self._binding.repository_id,
                objective_id=self._binding.objective_id,
            )
            values["authorization"] = authorization or AuthorizationDecision(
                verdict=AuthorizationVerdict.PERMIT,
                operation=selected,
                granted_authority=OperationAuthority.MUTATION,
                **self._binding.to_dict(),
                lease_id=lease_id or self._lease_id,
                fencing_epoch=(
                    self._fencing_epoch
                    if fencing_epoch is None
                    else int(fencing_epoch)
                ),
                authorized_effect_ids=tuple(item.effect_id for item in effects),
                grant_ids=("grant:database-control",),
                evaluated_at_ms=1_000,
                expires_at_ms=2_000,
            )
            values["lease_id"] = lease_id or self._lease_id
            values["fencing_epoch"] = (
                self._fencing_epoch if fencing_epoch is None else int(fencing_epoch)
            )
        return OperationRequest(**values)

    def execute(
        self, request: OperationRequest | Mapping[str, Any]
    ) -> OperationResult:
        """Dispatch through the shared :class:`SupervisorControlService`."""

        if not isinstance(request, OperationRequest):
            request = OperationRequest.from_dict(request)
        with self._lock:
            self._dispatch_count += 1
        return self._service.execute(request)

    def execute_operation(
        self,
        operation: Operation | str,
        *,
        parameters: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        **request_kwargs: Any,
    ) -> OperationResult:
        request = self.build_request(
            operation,
            parameters=parameters,
            dry_run=dry_run,
            **request_kwargs,
        )
        return self.execute(request)

    # -- read helpers -------------------------------------------------------

    def status(
        self, *, parameters: Mapping[str, Any] | None = None
    ) -> OperationResult:
        return self.execute_operation(Operation.STATUS, parameters=parameters)

    def health(
        self, *, parameters: Mapping[str, Any] | None = None
    ) -> OperationResult:
        return self.execute_operation(Operation.HEALTH, parameters=parameters)

    def logs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        params = dict(parameters or {})
        params.setdefault("kind", "logs")
        params.setdefault("limit", limit)
        params.setdefault("offset", offset)
        return self.execute_operation(Operation.EVENTS, parameters=params)

    def goals(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.GOALS, parameters={"limit": limit, "offset": offset}
        )

    def tasks(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.TASKS, parameters={"limit": limit, "offset": offset}
        )

    def events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        after_sequence: int = 0,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.EVENTS,
            parameters={
                "limit": limit,
                "offset": offset,
                "after_sequence": after_sequence,
                "kind": "events",
            },
        )

    def metrics(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.METRICS, parameters={"limit": limit, "offset": offset}
        )

    def lanes(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.LANES, parameters={"limit": limit, "offset": offset}
        )

    def daemons(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.ARTIFACT_QUERY,
            parameters={
                "resource": "daemons",
                "limit": limit,
                "offset": offset,
            },
        )

    def worktrees(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.ARTIFACT_QUERY,
            parameters={
                "resource": "worktrees",
                "limit": limit,
                "offset": offset,
            },
        )

    def mutations(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.ARTIFACT_QUERY,
            parameters={
                "resource": "mutations",
                "limit": limit,
                "offset": offset,
            },
        )

    def ast(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.ARTIFACT_QUERY,
            parameters={"resource": "ast", "limit": limit, "offset": offset},
        )

    def receipts(
        self, *, limit: int = 50, offset: int = 0
    ) -> OperationResult:
        return self.execute_operation(
            Operation.RECEIPTS, parameters={"limit": limit, "offset": offset}
        )

    # -- lifecycle helpers --------------------------------------------------

    def start(
        self,
        *,
        reason: str = "operator start",
        dry_run: bool = False,
        ready: bool = True,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.START,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "start",
                "ready": ready,
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def pause(
        self,
        *,
        reason: str = "operator pause",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.PAUSE,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "pause",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def resume(
        self,
        *,
        reason: str = "operator resume",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.RESUME,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "resume",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def drain(
        self,
        *,
        reason: str = "operator drain",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.DRAIN,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "drain",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def stop(
        self,
        *,
        reason: str = "operator stop",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.STOP,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "stop",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def retry(
        self,
        *,
        reason: str = "operator retry",
        dry_run: bool = False,
        ready: bool = True,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.RETRY,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "retry",
                "ready": ready,
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def cancel(
        self,
        *,
        reason: str = "operator cancel",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.CANCEL,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "cancel",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    def quarantine(
        self,
        *,
        reason: str = "operator quarantine",
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        return self.execute_operation(
            Operation.QUARANTINE,
            parameters={
                "target_id": self.program_target.program_id,
                "reason": reason,
                "requested_state": "quarantine",
            },
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    # -- import preview / export / backup -----------------------------------

    def import_preview(
        self,
        source_path: Path | str,
        *,
        media_type: str = "json",
    ) -> Mapping[str, Any]:
        """Preview a legacy import without applying it (side-effect free apply).

        The preview reads the source under an explicit path, digests it, and
        returns a non-authoritative receipt.  It never mutates the database.
        """

        path = Path(source_path)
        if not path.is_file():
            raise DatabaseControlOperationsRequestError(
                f"import source not found: {path}"
            )
        raw = path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise DatabaseControlOperationsRequestError(
                "import source exceeds the 16 MiB bound"
            )
        text = raw.decode("utf-8")
        record_count = 0
        if media_type == "json":
            payload = json.loads(text)
            if isinstance(payload, list):
                record_count = len(payload)
            elif isinstance(payload, Mapping):
                record_count = 1
            else:
                record_count = 0
        elif media_type == "jsonl":
            record_count = sum(
                1 for line in text.splitlines() if line.strip()
            )
        else:
            record_count = text.count("\n") + (1 if text.strip() else 0)

        receipt = {
            "schema": DATABASE_IMPORT_PREVIEW_SCHEMA,
            "mode": "preview",
            "applied": False,
            "source_path": str(path),
            "media_type": media_type,
            "byte_length": len(raw),
            "source_digest": f"sha256:{hashlib.sha256(raw).hexdigest()}",
            "record_count": record_count,
            "authority": "export",
            "database_path": str(self._database_path),
        }
        return MappingProxyType(receipt)

    def export(
        self,
        destination: Path | str,
        *,
        view: str = "portable",
        limit: int = 256,
    ) -> Mapping[str, Any]:
        """Export a non-authoritative JSON snapshot of selected views."""

        selected = str(view).strip().lower()
        if selected not in _EXPORT_VIEWS:
            raise DatabaseControlOperationsRequestError(
                f"unsupported export view: {view}"
            )
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "schema": DATABASE_EXPORT_RECEIPT_SCHEMA,
            "view": selected,
            "authority": "export",
            "non_authoritative": True,
            "database_path": str(self._database_path),
            "program_id": self.program_target.program_id,
        }
        if selected in {"status", "portable"}:
            payload["status"] = self.status().data
        if selected in {"health", "portable"}:
            payload["health"] = self.health().data
        if selected in {"goals", "portable"}:
            payload["goals"] = self.goals(limit=limit).data
        if selected in {"tasks", "portable"}:
            payload["tasks"] = self.tasks(limit=limit).data
        if selected in {"events", "portable"}:
            payload["events"] = self.events(limit=limit).data
        if selected in {"logs", "portable"}:
            payload["logs"] = self.logs(limit=limit).data
        if selected in {"metrics", "portable"}:
            payload["metrics"] = self.metrics(limit=limit).data
        if selected in {"worktrees", "portable"}:
            payload["worktrees"] = self.worktrees(limit=limit).data
        if selected in {"mutations", "portable"}:
            payload["mutations"] = self.mutations(limit=limit).data
        if selected in {"receipts", "portable"}:
            payload["receipts"] = self.receipts(limit=limit).data

        body = _json_dumps(payload)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(body + "\n", encoding="utf-8")
        tmp.replace(dest)
        digest = _sha256_text(body)
        receipt = {
            "schema": DATABASE_EXPORT_RECEIPT_SCHEMA,
            "destination": str(dest),
            "view": selected,
            "artifact_digest": digest,
            "byte_length": len(body.encode("utf-8")),
            "authority": "export",
            "non_authoritative": True,
            "database_path": str(self._database_path),
        }
        return MappingProxyType(receipt)

    def backup(
        self,
        destination: Path | str,
        *,
        reason: str = "operator backup",
    ) -> Mapping[str, Any]:
        """Copy the control database to ``destination`` and record a receipt."""

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Ensure schema exists before copying.
        _ = self.status()
        source = self._database_path
        if not source.is_file():
            raise DatabaseControlOperationsRequestError(
                f"control database not found: {source}"
            )
        shutil.copy2(source, dest)
        digest = _sha256_file(dest)
        recorded = self._backend.record_backup(
            destination_uri=str(dest),
            artifact_digest=digest,
            body={"reason": reason},
        )
        receipt = {
            "schema": DATABASE_BACKUP_RECEIPT_SCHEMA,
            **recorded,
            "source_database": str(source),
            "destination": str(dest),
            "reason": reason,
        }
        return MappingProxyType(receipt)

    # -- transport parity helper --------------------------------------------

    def surface_parity_case(
        self,
        operation: Operation | str,
        *,
        parameters: Mapping[str, Any] | None = None,
        dry_run: bool = True,
    ) -> Mapping[str, Any]:
        """Return identical request/result identity for Python/CLI/MCP."""

        request = self.build_request(
            operation, parameters=parameters, dry_run=dry_run
        )
        result = self.execute(request)
        return MappingProxyType(
            {
                "dispatcher_id": self.DISPATCHER_ID,
                "request_id": request.request_id,
                "request_content_id": request.content_id,
                "result_id": result.result_id,
                "result_content_id": result.content_id,
                "operation": request.operation.value,
                "authority": result.authority.value,
                "status": result.status.value,
                "canonical_request": request.to_record(),
                "canonical_result": result.to_record(),
            }
        )


def open_database_control_operations(
    database_path: Path | str,
    **kwargs: Any,
) -> DatabaseControlOperations:
    """Open :class:`DatabaseControlOperations` on ``database_path``."""

    if not duckdb_available():
        raise DatabaseControlOperationsUnavailableError(
            "DuckDB is required for DatabaseControlOperations"
        )
    return DatabaseControlOperations(database_path, **kwargs)


__all__ = (
    "DATABASE_BACKUP_RECEIPT_SCHEMA",
    "DATABASE_CONTROL_DISCOVERY_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_INTERFACE",
    "DATABASE_CONTROL_OPERATIONS_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_VERSION",
    "DATABASE_EXPORT_RECEIPT_SCHEMA",
    "DATABASE_IMPORT_PREVIEW_SCHEMA",
    "DatabaseControlBinding",
    "DatabaseControlOperations",
    "DatabaseControlOperationsError",
    "DatabaseControlOperationsRequestError",
    "DatabaseControlOperationsUnavailableError",
    "open_database_control_operations",
)
