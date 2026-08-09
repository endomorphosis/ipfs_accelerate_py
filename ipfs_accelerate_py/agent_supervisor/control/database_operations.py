"""Database-backed control operations facade (DQP-029).

Interfaces: ``DatabaseControlOperations@1``

:class:`DatabaseControlOperations` is the transport-neutral typed service that
Python, CLI, and MCP adapters share for configured database programs. Every
public method builds a canonical :class:`OperationRequest`, dispatches it
through :class:`SupervisorControlService.execute` (never a shell), and returns
the canonical :class:`OperationResult`.

Configured database programs expose status/health/logs/stop (and the rest of
the closed lifecycle vocabulary) rather than launch-only control. Read,
proposal, and mutation authority remain distinct: reads never mutate, dry-run
mutations never apply effects, and real mutations require authorization,
lease/fence, expected effects, and idempotency.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Union

from .control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlSurface,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    decode_operation_request,
)
from .control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    InMemoryControlStateStore,
    InMemoryLifecycleStore,
    LifecycleStatus,
    SupervisorControlService,
    normalize_control_request,
)
from .database_backend import (
    DATABASE_PROGRAM_REQUIRED_CONTROL_OPS,
    DATABASE_SUPERVISOR_BACKEND_INTERFACE,
    LOGS_EVENT_KIND,
    DatabaseSupervisorBackend,
    InMemoryDatabaseControlStore,
    open_database_supervisor_backend,
)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_CONTROL_OPERATIONS_INTERFACE: Final[str] = "DatabaseControlOperations@1"
DATABASE_CONTROL_OPERATIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-operations@1"
)
DATABASE_CONTROL_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-binding@1"
)
DATABASE_TRANSPORT_PARITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-control-transport-parity@1"
)
DATABASE_CONTROL_OPERATIONS_VERSION: Final[int] = 1

DEFAULT_CALLER: Final[str] = "operator:database-control"
DEFAULT_TARGET_ID: Final[str] = "supervisor"
DEFAULT_LEASE_ID: Final[str] = "lease:database-control"
DEFAULT_FENCING_EPOCH: Final[int] = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseControlOperationsError(RuntimeError):
    """Base fail-closed error for database control operations."""


class DatabaseControlAuthorityError(DatabaseControlOperationsError):
    """Read/proposal/mutation authority separation was violated."""


class DatabaseControlConfigurationError(DatabaseControlOperationsError):
    """Binding, program, or root configuration is incomplete or unsafe."""


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseControlBinding:
    """Identity binding shared by every request on one operations instance."""

    SCHEMA: ClassVar[str] = DATABASE_CONTROL_BINDING_SCHEMA

    repository_root: str
    state_root: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str = DEFAULT_CALLER
    target_id: str = DEFAULT_TARGET_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_root",
            str(Path(self.repository_root).expanduser().resolve(strict=False)),
        )
        object.__setattr__(
            self,
            "state_root",
            str(Path(self.state_root).expanduser().resolve(strict=False)),
        )
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "target_id",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise DatabaseControlConfigurationError(f"{name} is required")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "target_id": self.target_id,
        }

    def as_request_kwargs(self) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Operations facade
# ---------------------------------------------------------------------------


class DatabaseControlOperations:
    """Typed database-program control API with direct service dispatch.

    Python, CLI, and MCP callers share this facade so request/result content
    identities remain identical across transports. Adapters decode or build an
    :class:`OperationRequest` and call :meth:`dispatch` — they never shell out
    and never infer mutation permission from a filesystem path.
    """

    INTERFACE: ClassVar[str] = DATABASE_CONTROL_OPERATIONS_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_CONTROL_OPERATIONS_SCHEMA
    DISPATCHER_ID: ClassVar[str] = DIRECT_CONTROL_SERVICE_DISPATCHER_ID

    def __init__(
        self,
        binding: DatabaseControlBinding,
        *,
        database_program: Mapping[str, Any] | None = None,
        backend: DatabaseSupervisorBackend | None = None,
        service: SupervisorControlService | None = None,
        lifecycle_store: InMemoryLifecycleStore | None = None,
        control_store: InMemoryDatabaseControlStore | None = None,
        state_store: InMemoryControlStateStore | None = None,
        lease_validator: Callable[[OperationRequest], Any] | None = None,
        authorization_validator: Callable[[OperationRequest], Any] | None = None,
        clock_ms: Callable[[], int] | None = None,
        require_lease_validator: bool = True,
    ) -> None:
        if not isinstance(binding, DatabaseControlBinding):
            raise TypeError("binding must be a DatabaseControlBinding")
        self._binding = binding
        self._program = MappingProxyType(dict(database_program or {}))
        if backend is not None and not isinstance(backend, DatabaseSupervisorBackend):
            raise TypeError("backend must be a DatabaseSupervisorBackend")
        self._backend = backend or open_database_supervisor_backend(
            database_program=self._program,
            lifecycle_store=lifecycle_store,
            control_store=control_store,
            clock_ms=clock_ms or (lambda: 1_500),
        )
        if service is not None:
            self._service = service
        else:
            kwargs: dict[str, Any] = {
                "repository_allowlist": (binding.repository_root,),
                "state_allowlist": (binding.state_root,),
                "backend": self._backend,
                "state_store": state_store or InMemoryControlStateStore(),
                "require_lease_validator": require_lease_validator,
            }
            if lease_validator is not None:
                kwargs["lease_validator"] = lease_validator
            elif require_lease_validator:
                kwargs["lease_validator"] = _default_lease_validator
            if authorization_validator is not None:
                kwargs["authorization_validator"] = authorization_validator
            if clock_ms is not None:
                kwargs["clock_ms"] = clock_ms
            self._service = SupervisorControlService(**kwargs)
        # Discovery / construction is inert.
        self._discovered = False

    # -- properties ----------------------------------------------------------

    @property
    def binding(self) -> DatabaseControlBinding:
        return self._binding

    @property
    def backend(self) -> DatabaseSupervisorBackend:
        return self._backend

    @property
    def service(self) -> SupervisorControlService:
        return self._service

    @property
    def database_program(self) -> Mapping[str, Any]:
        return dict(self._program)

    @property
    def dispatcher_id(self) -> str:
        return self.DISPATCHER_ID

    # -- discovery / capability ----------------------------------------------

    def discover(self) -> dict[str, Any]:
        """Return the side-effect-free control surface for this program.

        Discovery never starts processes, opens providers, or mutates stores.
        """

        surface = self._backend.supported_control_surface()
        report = self._service.capability_report()
        self._discovered = True
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "version": DATABASE_CONTROL_OPERATIONS_VERSION,
            "dispatcher_id": self.DISPATCHER_ID,
            "backend_interface": DATABASE_SUPERVISOR_BACKEND_INTERFACE,
            "binding": self._binding.to_dict(),
            "database_program": dict(self._program),
            "control_surface": surface,
            "supported_operations": list(surface["supported_operations"]),
            "required_control_ops": sorted(DATABASE_PROGRAM_REQUIRED_CONTROL_OPS),
            "required_control_ops_supported": bool(
                surface.get("required_control_ops_supported")
            ),
            "launch_only": False,
            "supports_status": True,
            "supports_health": True,
            "supports_logs": True,
            "supports_stop": True,
            "authority_classes": {
                "read": OperationAuthority.READ.value,
                "proposal": OperationAuthority.PROPOSAL.value,
                "mutation": OperationAuthority.MUTATION.value,
            },
            "capability_report_id": report.content_id,
            "optional_providers_loaded": report.optional_providers_loaded,
            "processes_started": report.processes_started,
            "processes_started_by_discovery": False,
            "direct_service_dispatch": True,
        }

    def supported_operations(self) -> tuple[str, ...]:
        return tuple(self._backend.supported_control_surface()["supported_operations"])

    def assert_not_launch_only(self) -> None:
        """Fail closed when status/health/logs/stop are missing."""

        surface = self._backend.supported_control_surface()
        if surface.get("launch_only"):
            raise DatabaseControlConfigurationError(
                "database program control is launch-only; status/health/logs/stop "
                "are required"
            )
        for name, flag in (
            ("status", "supports_status"),
            ("health", "supports_health"),
            ("logs", "supports_logs"),
            ("stop", "supports_stop"),
        ):
            if not surface.get(flag):
                raise DatabaseControlConfigurationError(
                    f"database program control missing required op: {name}"
                )

    # -- request construction ------------------------------------------------

    def build_request(
        self,
        operation: Union[Operation, str],
        *,
        parameters: Mapping[str, Any] | None = None,
        dry_run: bool = False,
        expected_effects: Sequence[ExpectedEffect | Mapping[str, Any]] = (),
        idempotency_key: str = "",
        authorization: AuthorizationDecision | Mapping[str, Any] | None = None,
        lease_id: str = "",
        fencing_epoch: int | None = None,
        bounds: ControlBounds | None = None,
    ) -> OperationRequest:
        """Build one canonical OperationRequest for this binding."""

        selected = (
            operation if isinstance(operation, Operation) else Operation(str(operation))
        )
        params = dict(parameters or {})
        params.setdefault("target_id", self._binding.target_id)
        effects = _coerce_effects(expected_effects)
        values: dict[str, Any] = {
            "operation": selected,
            **self._binding.as_request_kwargs(),
            "parameters": params,
            "bounds": bounds or ControlBounds(),
            "dry_run": bool(dry_run),
            "expected_effects": effects,
        }
        if selected in MUTATION_OPERATIONS and not dry_run:
            if not effects:
                raise DatabaseControlAuthorityError(
                    "mutation requests must declare expected effects"
                )
            key = str(idempotency_key or "").strip()
            if not key:
                raise DatabaseControlAuthorityError(
                    "mutation requests require an idempotency key"
                )
            values["idempotency"] = IdempotencyKey(
                key=key,
                operation=selected,
                caller=self._binding.caller,
                repository_id=self._binding.repository_id,
                objective_id=self._binding.objective_id,
            )
            if authorization is None:
                raise DatabaseControlAuthorityError(
                    "mutation requests require an authorization decision"
                )
            if isinstance(authorization, AuthorizationDecision):
                values["authorization"] = authorization
            else:
                values["authorization"] = AuthorizationDecision.from_dict(
                    authorization
                )
            values["lease_id"] = str(lease_id or DEFAULT_LEASE_ID)
            values["fencing_epoch"] = (
                DEFAULT_FENCING_EPOCH
                if fencing_epoch is None
                else int(fencing_epoch)
            )
        elif authorization is not None:
            if isinstance(authorization, AuthorizationDecision):
                values["authorization"] = authorization
            else:
                values["authorization"] = AuthorizationDecision.from_dict(
                    authorization
                )
        return OperationRequest(**values)

    def build_mutation_request(
        self,
        operation: Union[Operation, str],
        *,
        parameters: Mapping[str, Any] | None = None,
        idempotency_key: str,
        dry_run: bool = False,
        effect_resource: str = "",
        effect_paths: Sequence[str] = ("lifecycle/status.json",),
        lease_id: str = DEFAULT_LEASE_ID,
        fencing_epoch: int = DEFAULT_FENCING_EPOCH,
        evaluated_at_ms: int = 1_000,
        expires_at_ms: int = 60_000,
    ) -> OperationRequest:
        """Build a fully bound lifecycle mutation request for this program."""

        selected = (
            operation if isinstance(operation, Operation) else Operation(str(operation))
        )
        if selected not in MUTATION_OPERATIONS:
            raise DatabaseControlAuthorityError(
                f"{selected.value} is not a mutation operation"
            )
        resource = effect_resource or f"supervisor:{self._binding.target_id}"
        effect = ExpectedEffect(
            effect_id=f"{selected.value}:{self._binding.target_id}",
            kind=EffectKind.LIFECYCLE_TRANSITION,
            resource=resource,
            paths=tuple(effect_paths),
            description=f"Database program {selected.value}",
        )
        params = dict(parameters or {})
        params.setdefault("target_id", self._binding.target_id)
        params.setdefault("reason", "database-control-operations")
        params.setdefault("requested_state", selected.value)
        authorization = AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=selected,
            granted_authority=OperationAuthority.MUTATION,
            repository_root=self._binding.repository_root,
            state_root=self._binding.state_root,
            repository_id=self._binding.repository_id,
            tree_id=self._binding.tree_id,
            objective_id=self._binding.objective_id,
            objective_revision=self._binding.objective_revision,
            policy_id=self._binding.policy_id,
            policy_revision=self._binding.policy_revision,
            caller=self._binding.caller,
            lease_id=lease_id,
            fencing_epoch=fencing_epoch,
            authorized_effect_ids=(effect.effect_id,),
            grant_ids=("grant:database-control",),
            evaluated_at_ms=evaluated_at_ms,
            expires_at_ms=expires_at_ms,
        )
        return self.build_request(
            selected,
            parameters=params,
            dry_run=dry_run,
            expected_effects=(effect,),
            idempotency_key=idempotency_key,
            authorization=authorization,
            lease_id=lease_id,
            fencing_epoch=fencing_epoch,
        )

    # -- direct service dispatch ---------------------------------------------

    def dispatch(
        self, request: Union[OperationRequest, Mapping[str, Any]]
    ) -> OperationResult:
        """Dispatch through the canonical SupervisorControlService only."""

        normalized = normalize_control_request(request)
        self._assert_authority_separation(normalized)
        return self._service.execute(normalized)

    # Alias used by CLI/MCP adapter vocabulary.
    execute = dispatch
    handle = dispatch

    def dispatch_surface(
        self,
        surface: Union[ControlSurface, str],
        request: Union[OperationRequest, Mapping[str, Any]],
    ) -> OperationResult:
        """Dispatch as a named transport while preserving request identity.

        CLI and MCP adapters are thin: they decode the same request bytes and
        call the same service. This method models that contract without
        shelling out.
        """

        selected = (
            surface
            if isinstance(surface, ControlSurface)
            else ControlSurface(str(surface).casefold())
        )
        if selected is ControlSurface.PYTHON:
            payload: OperationRequest | Mapping[str, Any] = request
        elif selected is ControlSurface.CLI:
            # CLI decodes request JSON then dispatches.
            if isinstance(request, OperationRequest):
                payload = decode_operation_request(request.to_dict())
            else:
                payload = decode_operation_request(dict(request))
        elif selected is ControlSurface.MCP:
            # MCP tools decode the same canonical record over JSON.
            if isinstance(request, OperationRequest):
                encoded = json.loads(request.to_json())
            else:
                encoded = json.loads(
                    json.dumps(dict(request), sort_keys=True, separators=(",", ":"))
                )
            payload = decode_operation_request(encoded)
        else:
            raise DatabaseControlOperationsError(
                f"unsupported control surface: {selected!r}"
            )
        return self.dispatch(payload)

    def transport_parity(
        self, request: Union[OperationRequest, Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Execute one request on Python/CLI/MCP and compare identities."""

        normalized = normalize_control_request(request)
        # Prove every surface decodes to the same request identity first.
        cli_request = decode_operation_request(normalized.to_dict())
        mcp_request = decode_operation_request(
            json.loads(normalized.to_json())
        )
        request_ids_match = (
            normalized.request_id
            == cli_request.request_id
            == mcp_request.request_id
        )
        python_result = self.dispatch_surface(ControlSurface.PYTHON, normalized)
        cli_result = self.dispatch_surface(ControlSurface.CLI, normalized)
        mcp_result = self.dispatch_surface(ControlSurface.MCP, normalized)
        equal = (
            request_ids_match
            and python_result.content_id == cli_result.content_id
            and cli_result.content_id == mcp_result.content_id
            and python_result.to_record() == cli_result.to_record()
            and cli_result.to_record() == mcp_result.to_record()
        )
        return {
            "schema": DATABASE_TRANSPORT_PARITY_SCHEMA,
            "request_id": normalized.request_id,
            "dispatcher_id": self.DISPATCHER_ID,
            "python_result_id": python_result.content_id,
            "cli_result_id": cli_result.content_id,
            "mcp_result_id": mcp_result.content_id,
            "identical": equal,
            "request_ids_match": request_ids_match,
            "status": python_result.status.value,
            "surfaces": [
                ControlSurface.PYTHON.value,
                ControlSurface.CLI.value,
                ControlSurface.MCP.value,
            ],
        }

    # -- high-level reads ----------------------------------------------------

    def status(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.STATUS, parameters=parameters)
        )

    def health(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.HEALTH, parameters=parameters)
        )

    def logs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        after_sequence: int = 0,
        component: str = "",
        severity: str = "",
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        """Read structured logs for the configured database program.

        Logs are a first-class control verb for database programs. They are
        served as a read projection through the EVENTS catalog operation with
        ``kind=logs`` so transports share one request/result contract.
        """

        params = dict(parameters or {})
        params["kind"] = LOGS_EVENT_KIND
        params["limit"] = int(limit)
        params["offset"] = int(offset)
        if after_sequence:
            params["after_sequence"] = int(after_sequence)
        if component:
            params["component"] = component
        if severity:
            params["severity"] = severity
        return self.dispatch(
            self.build_request(Operation.EVENTS, parameters=params)
        )

    def events(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.EVENTS, parameters=parameters)
        )

    def metrics(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.METRICS, parameters=parameters)
        )

    def goals(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.GOALS, parameters=parameters)
        )

    def tasks(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> OperationResult:
        return self.dispatch(
            self.build_request(Operation.TASKS, parameters=parameters)
        )

    def capabilities(self) -> OperationResult:
        return self.dispatch(self.build_request(Operation.CAPABILITIES))

    # -- high-level lifecycle mutations --------------------------------------

    def start(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.START,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def pause(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.PAUSE,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def resume(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.RESUME,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def drain(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.DRAIN,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def stop(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        """Stop the configured database program through fenced lifecycle control."""

        return self.dispatch(
            self.build_mutation_request(
                Operation.STOP,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def retry(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.RETRY,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def cancel(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.CANCEL,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    def quarantine(
        self,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        parameters: Mapping[str, Any] | None = None,
        **mutation_kwargs: Any,
    ) -> OperationResult:
        return self.dispatch(
            self.build_mutation_request(
                Operation.QUARANTINE,
                parameters=parameters,
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                **mutation_kwargs,
            )
        )

    # -- helpers -------------------------------------------------------------

    def seed_status(self, status: LifecycleStatus) -> None:
        self._backend.seed_status(status)

    def append_log(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._backend.append_log(message, **kwargs)

    def heartbeat(self, **values: Any) -> LifecycleStatus:
        return self._backend.heartbeat(self._binding.target_id, **values)

    @staticmethod
    def _assert_authority_separation(request: OperationRequest) -> None:
        authority = request.operation.authority
        if request.operation in READ_OPERATIONS:
            if authority is not OperationAuthority.READ:
                raise DatabaseControlAuthorityError(
                    "read operations must carry read authority"
                )
            if request.dry_run is False and request.expected_effects:
                # Reads may omit effects; non-empty effects on a read must
                # still be observe-only (enforced by OperationRequest).
                for effect in request.expected_effects:
                    if effect.authority is not OperationAuthority.READ:
                        raise DatabaseControlAuthorityError(
                            "read requests cannot declare mutation effects"
                        )
        elif request.operation in PROPOSAL_OPERATIONS:
            if authority is not OperationAuthority.PROPOSAL:
                raise DatabaseControlAuthorityError(
                    "proposal operations must carry proposal authority"
                )
        elif request.operation in MUTATION_OPERATIONS:
            if request.dry_run:
                if request.effective_authority is not OperationAuthority.PROPOSAL:
                    raise DatabaseControlAuthorityError(
                        "dry-run mutations must demote to proposal authority"
                    )
            elif authority is not OperationAuthority.MUTATION:
                raise DatabaseControlAuthorityError(
                    "real mutations must carry mutation authority"
                )
        else:
            raise DatabaseControlAuthorityError(
                f"unknown operation authority class for {request.operation.value}"
            )


def _default_lease_validator(request: OperationRequest) -> bool:
    if request.operation not in MUTATION_OPERATIONS or request.dry_run:
        return True
    return bool(request.lease_id) and request.fencing_epoch is not None


def _coerce_effects(
    values: Sequence[ExpectedEffect | Mapping[str, Any]],
) -> tuple[ExpectedEffect, ...]:
    effects: list[ExpectedEffect] = []
    for item in values:
        if isinstance(item, ExpectedEffect):
            effects.append(item)
        elif isinstance(item, Mapping):
            effects.append(ExpectedEffect.from_dict(item))
        else:
            raise TypeError("expected_effects items must be ExpectedEffect or mapping")
    return tuple(effects)


def open_database_control_operations(
    binding: DatabaseControlBinding,
    *,
    database_program: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> DatabaseControlOperations:
    """Construct operations for one configured database program (no launch)."""

    ops = DatabaseControlOperations(
        binding,
        database_program=database_program,
        **kwargs,
    )
    ops.assert_not_launch_only()
    return ops


__all__ = (
    "DATABASE_CONTROL_BINDING_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_INTERFACE",
    "DATABASE_CONTROL_OPERATIONS_SCHEMA",
    "DATABASE_CONTROL_OPERATIONS_VERSION",
    "DATABASE_TRANSPORT_PARITY_SCHEMA",
    "DatabaseControlAuthorityError",
    "DatabaseControlBinding",
    "DatabaseControlConfigurationError",
    "DatabaseControlOperations",
    "DatabaseControlOperationsError",
    "open_database_control_operations",
)
