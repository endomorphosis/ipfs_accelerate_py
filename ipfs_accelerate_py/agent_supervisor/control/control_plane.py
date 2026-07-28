"""Shared, transport-neutral control service for the agent supervisor.

The CLI and MCP surfaces intentionally do not implement supervisor policy.
They construct :class:`~.control_contracts.OperationRequest` records and pass
them to :class:`SupervisorControlService`.  This module is the single place
that applies target allowlists, bounds, authorization freshness, lease
fencing, idempotency, dry-run semantics, stable errors, and audit receipts.

Backends are ordinary Python callables.  No operation is converted to a shell
command.  The included :class:`RepositorySupervisorBackend` supplies bounded
read adapters for the package's existing JSON, objective, task-board, event,
cache, and artifact APIs.  Runtime deployments register the mutating package
APIs they operate; an unregistered mutation fails closed as ``unavailable``.
"""

from __future__ import annotations

import fcntl
import hashlib
import heapq
import json
import os
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, Union

from .control_contracts import (
    CONTROL_CATALOG_VERSION,
    CONTROL_CONTRACT_VERSION,
    DEFAULT_CONTROL_CATALOG,
    DOWNSTREAM_EFFECT_PREVIEW_OPERATIONS,
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    AuthorityViolationError,
    AuthorizationBindingError,
    CapabilityDegradation,
    CapabilityReport,
    ControlBounds,
    ControlBoundsError,
    ControlContractError,
    ControlDiscoveryManifest,
    ControlDiscoveryRuntimeState,
    ControlMutationRuntimeState,
    ControlOperationDescriptor,
    ControlSurface,
    DryRunPreview,
    EffectClaim,
    ErrorCode,
    EventCursor,
    EventCursorError,
    ExpectedEffect,
    LifecycleAction,
    LifecycleCommand,
    Operation,
    OperationCatalog,
    OperationAuthority,
    OperationCapability,
    OperationError,
    OperationRequest,
    OperationResult,
    OperationStatus,
    PaginationKind,
    PathEscapeError,
    UnsupportedCapabilityError,
    canonical_control_json_bytes,
    decode_operation_request,
)


CONTROL_SERVICE_VERSION: Final[str] = "1.0.0"
CONTROL_CONFORMANCE_V2_REQUIREMENT_ID: Final[str] = (
    "107787885166558411314422313513714746721"
)
CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-operation-conformance-case@2"
)
CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-catalog-conformance-evidence@2"
)
CONTROL_AUDIT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-audit-receipt@1"
)
CONTROL_MUTATION_TRANSACTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-mutation-transaction@1"
)
CONTROL_BACKEND_RESPONSE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-backend-response@1"
)
LIFECYCLE_STATUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-status@1"
)
LIFECYCLE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-event@1"
)
CONTROL_MUTATION_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-mutation-event@1"
)
CONTROL_SURFACE_PUBLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-surface-publication@1"
)
CONTROL_BEHAVIOR_NORMALIZATION_VERSION: Final[str] = (
    "agent-supervisor-control-normalization@1"
)
DIRECT_CONTROL_SERVICE_DISPATCHER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.control.control_plane:"
    "SupervisorControlService.execute"
)
DEFAULT_QUERY_LIMIT: Final[int] = 50
DEFAULT_MAX_QUERY_ITEMS: Final[int] = 256
DEFAULT_MAX_OFFSET: Final[int] = 1_000_000
DEFAULT_MAX_CONTROL_EVENTS: Final[int] = 256
CONTROL_REDACTION_MARKER: Final[str] = "[REDACTED]"
CONTROL_SENSITIVE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "set_cookie",
        "ssh_key",
        "token",
    }
)
_CONTROL_SENSITIVE_FIELD_SUFFIXES: Final[tuple[str, ...]] = (
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_CONTROL_SENSITIVE_ASSIGNMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?ix)
    (
        \b(?:access[_-]?token|api[_-]?key|authorization|client[_-]?secret|
        cookie|credentials?|password|passwd|private[_-]?key|refresh[_-]?token|
        secret|session[_-]?token|set[_-]?cookie|ssh[_-]?key|token)\b
        \s*[:=]\s*
    )
    (?:
        "(?:\\.|[^"\\])*"
        |
        '(?:\\.|[^'\\])*'
        |
        [^\s,;]+
    )
    """
)
_CONTROL_BEARER_CREDENTIAL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"
)
CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py",
    "ipfs_accelerate_py.agent_supervisor.ipfs_datasets_",
    "ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider",
    "ipfs_accelerate_py.agent_supervisor.proof.leanstral_goal_development",
    "ipfs_accelerate_py.agent_supervisor.proof.leanstral_goal_lifecycle",
    "ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm",
)


class SupervisorControlError(RuntimeError):
    """Base exception raised by service configuration and client misuse."""


class ControlCatalogConformanceError(SupervisorControlError):
    """A published surface differs from the closed canonical catalog."""


class TargetNotAllowedError(SupervisorControlError):
    """Raised internally when a request targets a root outside an allowlist."""


class OperationUnavailableError(SupervisorControlError):
    """A configured backend does not implement the requested operation."""


class LeaseValidationError(SupervisorControlError):
    """A mutation does not carry a current authoritative lease and fence."""


class StaleLeaseError(LeaseValidationError):
    """The supplied lease or fencing epoch is no longer current."""


class StaleTreeError(SupervisorControlError):
    """The request's repository/tree identity is no longer current."""


class IdempotencyConflictError(SupervisorControlError):
    """An idempotency key was already used for a different request."""


class TransactionConflictError(SupervisorControlError):
    """A mutation transaction compare-and-swap precondition did not hold."""


class PartialMutationError(SupervisorControlError):
    """A backend reports that a multi-step mutation only partially completed."""

    def __init__(
        self,
        message: str,
        *,
        applied_effect_ids: Iterable[str] = (),
        recovery: Union["MutationRecoveryAction", str] = "repair",
    ) -> None:
        super().__init__(message)
        self.applied_effect_ids = tuple(
            sorted({str(item).strip() for item in applied_effect_ids})
        )
        if any(not item for item in self.applied_effect_ids):
            raise ValueError("applied_effect_ids must not contain empty values")
        self.recovery = MutationRecoveryAction(
            str(getattr(recovery, "value", recovery))
        )


class BackendNotFoundError(SupervisorControlError):
    """The requested backend object does not exist."""


class BackendConflictError(SupervisorControlError):
    """The backend rejected an otherwise valid request due to current state."""


class BackendCancelledError(SupervisorControlError):
    """Backend execution was cancelled."""


class BackendTimeoutError(SupervisorControlError):
    """Backend execution exceeded its bound."""


class InvalidLifecycleTransitionError(BackendConflictError):
    """A lifecycle command is not legal from the authoritative state."""


class SupervisorLifecycleState(str, Enum):
    """Closed supervisor lifecycle vocabulary shared by every control surface."""

    STOPPED = "stopped"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    PAUSED = "paused"
    DRAINING = "draining"
    BLOCKED = "blocked"
    STOPPING = "stopping"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            SupervisorLifecycleState.STOPPED,
            SupervisorLifecycleState.FAILED,
        }

    @property
    def accepts_new_work(self) -> bool:
        return self in {
            SupervisorLifecycleState.HEALTHY,
            SupervisorLifecycleState.DEGRADED,
        }


LEGAL_LIFECYCLE_TRANSITIONS: Final[
    Mapping[SupervisorLifecycleState, frozenset[SupervisorLifecycleState]]
] = MappingProxyType(
    {
        SupervisorLifecycleState.STOPPED: frozenset(
            {SupervisorLifecycleState.STARTING}
        ),
        SupervisorLifecycleState.STARTING: frozenset(
            {
                SupervisorLifecycleState.HEALTHY,
                SupervisorLifecycleState.DEGRADED,
                SupervisorLifecycleState.BLOCKED,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.HEALTHY: frozenset(
            {
                SupervisorLifecycleState.DEGRADED,
                SupervisorLifecycleState.PAUSED,
                SupervisorLifecycleState.DRAINING,
                SupervisorLifecycleState.BLOCKED,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.DEGRADED: frozenset(
            {
                SupervisorLifecycleState.HEALTHY,
                SupervisorLifecycleState.PAUSED,
                SupervisorLifecycleState.DRAINING,
                SupervisorLifecycleState.BLOCKED,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.PAUSED: frozenset(
            {
                SupervisorLifecycleState.HEALTHY,
                SupervisorLifecycleState.DRAINING,
                SupervisorLifecycleState.BLOCKED,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.DRAINING: frozenset(
            {
                SupervisorLifecycleState.STOPPED,
                SupervisorLifecycleState.BLOCKED,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.BLOCKED: frozenset(
            {
                SupervisorLifecycleState.STARTING,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.STOPPING: frozenset(
            {
                SupervisorLifecycleState.STOPPED,
                SupervisorLifecycleState.FAILED,
            }
        ),
        SupervisorLifecycleState.FAILED: frozenset(
            {
                SupervisorLifecycleState.STARTING,
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.STOPPED,
            }
        ),
    }
)


def lifecycle_transition_is_legal(
    previous: Union[SupervisorLifecycleState, str],
    requested: Union[SupervisorLifecycleState, str],
) -> bool:
    """Return whether a transition is legal, including idempotent self-edges."""

    source = (
        previous
        if isinstance(previous, SupervisorLifecycleState)
        else SupervisorLifecycleState(str(previous))
    )
    target = (
        requested
        if isinstance(requested, SupervisorLifecycleState)
        else SupervisorLifecycleState(str(requested))
    )
    return source is target or target in LEGAL_LIFECYCLE_TRANSITIONS[source]


def _lifecycle_text_tuple(value: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError("lifecycle collection must be an array")
    result = tuple(
        sorted({str(item).strip() for item in value if str(item).strip()})
    )
    if len(result) > DEFAULT_MAX_CONTROL_EVENTS:
        raise ControlBoundsError("lifecycle collection exceeds 256 items")
    if any(len(item.encode("utf-8")) > 2048 for item in result):
        raise ControlBoundsError("lifecycle collection item exceeds 2048 bytes")
    return result


def _bounded_lifecycle_text(value: Any, name: str) -> str:
    result = str(value).strip()
    if len(result.encode("utf-8")) > 2048:
        raise ControlBoundsError(f"{name} exceeds 2048 bytes")
    return result


def _lifecycle_record_int(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: int = 0,
    nullable: bool = False,
) -> Union[int, None]:
    value = payload.get(name)
    if value in (None, "") and nullable:
        return None
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class LifecycleStatus:
    """One canonical status/health snapshot for a supervisor target."""

    target_id: str
    state: SupervisorLifecycleState = SupervisorLifecycleState.STOPPED
    phase: str = "stopped"
    heartbeat_at_ms: int = 0
    pid: Union[int, None] = None
    active_leases: tuple[str, ...] = ()
    refill_state: str = "idle"
    backpressure: bool = False
    backpressure_reasons: tuple[str, ...] = ()
    terminal_reason: str = ""
    transition_id: str = ""
    generation: int = 0
    fencing_epoch: Union[int, None] = None
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        target_id = _bounded_lifecycle_text(self.target_id, "target_id")
        if not target_id:
            raise ValueError("lifecycle target_id is required")
        object.__setattr__(self, "target_id", target_id)
        state = (
            self.state
            if isinstance(self.state, SupervisorLifecycleState)
            else SupervisorLifecycleState(str(self.state))
        )
        object.__setattr__(self, "state", state)
        phase = _bounded_lifecycle_text(self.phase, "phase") or state.value
        object.__setattr__(self, "phase", phase)
        for name in ("heartbeat_at_ms", "generation", "updated_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.pid is not None and (
            isinstance(self.pid, bool)
            or not isinstance(self.pid, int)
            or self.pid <= 0
        ):
            raise ValueError("pid must be a positive integer or null")
        if self.fencing_epoch is not None and (
            isinstance(self.fencing_epoch, bool)
            or not isinstance(self.fencing_epoch, int)
            or self.fencing_epoch < 0
        ):
            raise ValueError("fencing_epoch must be a non-negative integer or null")
        if not isinstance(self.backpressure, bool):
            raise ValueError("backpressure must be boolean")
        object.__setattr__(
            self, "active_leases", _lifecycle_text_tuple(self.active_leases)
        )
        object.__setattr__(
            self,
            "backpressure_reasons",
            _lifecycle_text_tuple(self.backpressure_reasons),
        )
        object.__setattr__(
            self,
            "refill_state",
            _bounded_lifecycle_text(self.refill_state, "refill_state") or "idle",
        )
        object.__setattr__(
            self,
            "terminal_reason",
            _bounded_lifecycle_text(self.terminal_reason, "terminal_reason"),
        )
        object.__setattr__(
            self,
            "transition_id",
            _bounded_lifecycle_text(self.transition_id, "transition_id"),
        )

    @property
    def heartbeat_at(self) -> str:
        return _utc_timestamp(self.heartbeat_at_ms) if self.heartbeat_at_ms else ""

    @property
    def updated_at(self) -> str:
        return _utc_timestamp(self.updated_at_ms) if self.updated_at_ms else ""

    @property
    def healthy(self) -> bool:
        return self.state is SupervisorLifecycleState.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_STATUS_SCHEMA,
            "target_id": self.target_id,
            "state": self.state.value,
            "phase": self.phase,
            "heartbeat_at_ms": self.heartbeat_at_ms,
            "heartbeat_at": self.heartbeat_at,
            "pid": self.pid,
            "active_leases": list(self.active_leases),
            "active_lease_count": len(self.active_leases),
            "refill_state": self.refill_state,
            "backpressure": self.backpressure,
            "backpressure_reasons": list(self.backpressure_reasons),
            "terminal_reason": self.terminal_reason,
            "transition_id": self.transition_id,
            "generation": self.generation,
            "fencing_epoch": self.fencing_epoch,
            "updated_at_ms": self.updated_at_ms,
            "updated_at": self.updated_at,
        }

    to_record = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleStatus":
        schema = str(payload.get("schema") or "")
        if schema and schema != LIFECYCLE_STATUS_SCHEMA:
            raise ValueError("unsupported lifecycle status schema")
        active_leases = payload.get("active_leases") or ()
        backpressure_reasons = payload.get("backpressure_reasons") or ()
        if not isinstance(active_leases, (list, tuple)):
            raise ValueError("active_leases must be an array")
        if not isinstance(backpressure_reasons, (list, tuple)):
            raise ValueError("backpressure_reasons must be an array")
        return cls(
            target_id=str(payload.get("target_id") or ""),
            state=SupervisorLifecycleState(str(payload.get("state") or "")),
            phase=str(payload.get("phase") or ""),
            heartbeat_at_ms=_lifecycle_record_int(
                payload, "heartbeat_at_ms"
            )
            or 0,
            pid=_lifecycle_record_int(payload, "pid", nullable=True),
            active_leases=tuple(active_leases),
            refill_state=str(payload.get("refill_state") or "idle"),
            backpressure=payload.get("backpressure", False),
            backpressure_reasons=tuple(backpressure_reasons),
            terminal_reason=str(payload.get("terminal_reason") or ""),
            transition_id=str(payload.get("transition_id") or ""),
            generation=_lifecycle_record_int(payload, "generation") or 0,
            fencing_epoch=_lifecycle_record_int(
                payload, "fencing_epoch", nullable=True
            ),
            updated_at_ms=_lifecycle_record_int(payload, "updated_at_ms")
            or 0,
        )


@dataclass(frozen=True)
class LifecycleEvent:
    """Bounded, replayable record of one lifecycle state decision."""

    sequence: int
    target_id: str
    action: str
    accepted: bool
    previous_state: SupervisorLifecycleState
    state: SupervisorLifecycleState
    reason: str
    request_id: str
    occurred_at_ms: int
    changed: bool = False
    replayed: bool = False
    recovered: bool = False
    fencing_epoch: Union[int, None] = None
    event_id: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("lifecycle event sequence must be positive")
        if (
            isinstance(self.occurred_at_ms, bool)
            or not isinstance(self.occurred_at_ms, int)
            or self.occurred_at_ms < 0
        ):
            raise ValueError("occurred_at_ms must be a non-negative integer")
        if self.fencing_epoch is not None and (
            isinstance(self.fencing_epoch, bool)
            or not isinstance(self.fencing_epoch, int)
            or self.fencing_epoch < 0
        ):
            raise ValueError("fencing_epoch must be non-negative or null")
        for name in ("accepted", "changed", "replayed", "recovered"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for value, name in (
            (self.target_id, "target_id"),
            (self.action, "action"),
            (self.reason, "reason"),
            (self.request_id, "request_id"),
        ):
            bounded = _bounded_lifecycle_text(value, name)
            if name in {"target_id", "action"} and not bounded:
                raise ValueError(f"{name} is required")
        previous = (
            self.previous_state
            if isinstance(self.previous_state, SupervisorLifecycleState)
            else SupervisorLifecycleState(str(self.previous_state))
        )
        current = (
            self.state
            if isinstance(self.state, SupervisorLifecycleState)
            else SupervisorLifecycleState(str(self.state))
        )
        object.__setattr__(self, "previous_state", previous)
        object.__setattr__(self, "state", current)
        payload = self._payload()
        expected = _content_id(payload)
        if self.event_id and self.event_id != expected:
            raise ValueError("lifecycle event identity does not match")
        object.__setattr__(self, "event_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_EVENT_SCHEMA,
            "sequence": self.sequence,
            "target_id": self.target_id,
            "action": self.action,
            "accepted": self.accepted,
            "changed": self.changed,
            "replayed": self.replayed,
            "recovered": self.recovered,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "reason": self.reason,
            "request_id": self.request_id,
            "fencing_epoch": self.fencing_epoch,
            "occurred_at_ms": self.occurred_at_ms,
            "occurred_at": _utc_timestamp(self.occurred_at_ms),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "event_id": self.event_id}

    to_record = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleEvent":
        schema = str(payload.get("schema") or "")
        if schema and schema != LIFECYCLE_EVENT_SCHEMA:
            raise ValueError("unsupported lifecycle event schema")
        for name in ("accepted", "changed", "replayed", "recovered"):
            value = payload.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        return cls(
            sequence=_lifecycle_record_int(payload, "sequence") or 0,
            target_id=str(payload.get("target_id") or ""),
            action=str(payload.get("action") or ""),
            accepted=payload.get("accepted", False),
            changed=payload.get("changed", False),
            replayed=payload.get("replayed", False),
            recovered=payload.get("recovered", False),
            previous_state=SupervisorLifecycleState(
                str(payload.get("previous_state") or "")
            ),
            state=SupervisorLifecycleState(str(payload.get("state") or "")),
            reason=str(payload.get("reason") or ""),
            request_id=str(payload.get("request_id") or ""),
            fencing_epoch=_lifecycle_record_int(
                payload, "fencing_epoch", nullable=True
            ),
            occurred_at_ms=_lifecycle_record_int(
                payload, "occurred_at_ms"
            )
            or 0,
            event_id=str(payload.get("event_id") or ""),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _utc_timestamp(now_ms: int) -> str:
    return (
        datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _current_child_process_ids() -> tuple[int, ...]:
    """Read the current OS child inventory without importing a process API."""

    process_id = os.getpid()
    children_path = Path(
        f"/proc/{process_id}/task/{process_id}/children"
    )
    try:
        raw = children_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ()
    result: set[int] = set()
    for item in raw.split():
        try:
            child_id = int(item)
        except ValueError:
            continue
        if child_id > 0:
            result.add(child_id)
    return tuple(sorted(result))


def capture_control_discovery_runtime_state(
    *,
    service_resolution_count: int = 0,
    optional_provider_load_count: int = 0,
    process_start_count: int = 0,
    optional_provider_prefixes: Sequence[
        str
    ] = CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES,
) -> ControlDiscoveryRuntimeState:
    """Capture a read-only discovery state for an independently instrumented run.

    Callers which intercept service resolution, optional-provider loading, or
    process starts pass their cumulative counters here.  Module and child
    inventories supplement those counters without importing a provider or a
    process-management package merely to inspect discovery.
    """

    prefixes = tuple(
        sorted(
            {
                str(item).strip()
                for item in optional_provider_prefixes
                if str(item).strip()
            }
        )
    )
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix)
                for prefix in prefixes
            )
        )
    )
    return ControlDiscoveryRuntimeState(
        optional_provider_modules=loaded,
        child_process_ids=_current_child_process_ids(),
        service_resolution_count=service_resolution_count,
        optional_provider_load_count=optional_provider_load_count,
        process_start_count=process_start_count,
    )


def _canonical_json_value(value: Any) -> Any:
    """Return a strict, bounded-contract-compatible JSON projection."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise ValueError("backend data contains a non-finite float")
        # Control contracts deliberately reject floats.  Stable decimal text
        # retains the observation without weakening their canonical format.
        return format(value, ".17g")
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical_json_value(item) for item in value]
    if is_dataclass(value):
        return _canonical_json_value(asdict(value))
    for method_name in ("to_record", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _canonical_json_value(method())
    raise ValueError(
        f"backend data contains unsupported value type {type(value).__name__}"
    )


def _normalized_control_field_name(value: Any) -> str:
    """Normalize a structured result field for conservative secret matching."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_sensitive_control_field(value: Any) -> bool:
    normalized = _normalized_control_field_name(value)
    return normalized in CONTROL_SENSITIVE_FIELD_NAMES or normalized.endswith(
        _CONTROL_SENSITIVE_FIELD_SUFFIXES
    )


def redact_control_text(value: str) -> str:
    """Redact common credential assignments in untrusted backend text.

    Free-form values cannot be classified perfectly, so the control boundary
    redacts only explicit credential assignments. Structured result fields
    receive the stronger key-based treatment in :func:`redact_control_data`.
    """

    if not isinstance(value, str):
        raise TypeError("control text must be a string")
    value = _CONTROL_BEARER_CREDENTIAL_RE.sub(
        f"Bearer {CONTROL_REDACTION_MARKER}",
        value,
    )
    return _CONTROL_SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{CONTROL_REDACTION_MARKER}",
        value,
    )


def redact_control_data(value: Any) -> Any:
    """Return a recursively redacted JSON-compatible control result.

    This function runs after backend values have been projected to canonical
    JSON types. It preserves collection shape and non-sensitive values so the
    normal request bounds and canonical result identity remain authoritative.
    """

    if isinstance(value, Mapping):
        return {
            str(key): (
                CONTROL_REDACTION_MARKER
                if _is_sensitive_control_field(key)
                else redact_control_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_control_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_control_data(item) for item in value)
    if isinstance(value, str):
        return redact_control_text(value)
    return value


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_control_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _publication_string_map(
    value: Mapping[Union[Operation, str], str],
    *,
    name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ControlCatalogConformanceError(f"{name} must be a mapping")
    normalized: dict[str, str] = {}
    for raw_operation, raw_value in value.items():
        try:
            operation = (
                raw_operation
                if isinstance(raw_operation, Operation)
                else Operation(str(raw_operation))
            )
        except ValueError as exc:
            raise ControlCatalogConformanceError(
                f"{name} contains unknown operation {raw_operation!r}"
            ) from exc
        text = str(raw_value).strip()
        if not text:
            raise ControlCatalogConformanceError(
                f"{name}[{operation.value!r}] must not be empty"
            )
        if operation.value in normalized:
            raise ControlCatalogConformanceError(
                f"{name} contains duplicate operation {operation.value!r}"
            )
        normalized[operation.value] = text
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class ControlSurfacePublication:
    """Provider-free declaration of one complete transport surface."""

    surface: ControlSurface
    catalog_id: str
    operations: tuple[Union[Operation, str], ...]
    request_schema_ids: Mapping[Union[Operation, str], str]
    result_schema_ids: Mapping[Union[Operation, str], str]
    behavior_ids: Mapping[Union[Operation, str], str]
    dispatcher_ids: Mapping[Union[Operation, str], str]
    dispatch_mode: str = "direct_service"
    catalog_version: int = CONTROL_CATALOG_VERSION
    provider_free: bool = True
    process_free: bool = True

    def __post_init__(self) -> None:
        try:
            surface = (
                self.surface
                if isinstance(self.surface, ControlSurface)
                else ControlSurface(str(self.surface))
            )
        except ValueError as exc:
            raise ControlCatalogConformanceError(
                f"unknown control surface {self.surface!r}"
            ) from exc
        object.__setattr__(self, "surface", surface)
        if (
            isinstance(self.catalog_version, bool)
            or not isinstance(self.catalog_version, int)
            or self.catalog_version < 1
        ):
            raise ControlCatalogConformanceError(
                "catalog_version must be a positive integer"
            )
        catalog_id = str(self.catalog_id).strip()
        if not catalog_id:
            raise ControlCatalogConformanceError("catalog_id must not be empty")
        object.__setattr__(self, "catalog_id", catalog_id)
        try:
            operations = tuple(
                sorted(
                    (
                        item
                        if isinstance(item, Operation)
                        else Operation(str(item))
                        for item in self.operations
                    ),
                    key=lambda item: item.value,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ControlCatalogConformanceError(
                "operations contains an unknown operation"
            ) from exc
        if len(operations) != len(set(operations)):
            raise ControlCatalogConformanceError(
                "operations contains duplicate operations"
            )
        object.__setattr__(self, "operations", operations)
        for name in (
            "request_schema_ids",
            "result_schema_ids",
            "behavior_ids",
            "dispatcher_ids",
        ):
            object.__setattr__(
                self,
                name,
                _publication_string_map(getattr(self, name), name=name),
            )
        mode = str(self.dispatch_mode).strip()
        if not mode:
            raise ControlCatalogConformanceError(
                "dispatch_mode must not be empty"
            )
        object.__setattr__(self, "dispatch_mode", mode)
        for name in ("provider_free", "process_free"):
            if not isinstance(getattr(self, name), bool):
                raise ControlCatalogConformanceError(
                    f"{name} must be a boolean"
                )

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.operations)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CONTROL_SURFACE_PUBLICATION_SCHEMA,
            "schema_version": CONTROL_CONTRACT_VERSION,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "catalog_version": self.catalog_version,
            "catalog_id": self.catalog_id,
            "surface": self.surface.value,
            "operations": self.operation_names,
            "request_schema_ids": dict(self.request_schema_ids),
            "result_schema_ids": dict(self.result_schema_ids),
            "behavior_ids": dict(self.behavior_ids),
            "dispatcher_ids": dict(self.dispatcher_ids),
            "dispatch_mode": self.dispatch_mode,
            "provider_free": self.provider_free,
            "process_free": self.process_free,
        }
        payload["content_id"] = _content_id(payload)
        return payload

    to_record = to_dict

    @property
    def content_id(self) -> str:
        return self.to_dict()["content_id"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlSurfacePublication":
        if not isinstance(payload, Mapping):
            raise ControlCatalogConformanceError(
                "control surface publication must be an object"
            )
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "catalog_version",
            "catalog_id",
            "surface",
            "operations",
            "request_schema_ids",
            "result_schema_ids",
            "behavior_ids",
            "dispatcher_ids",
            "dispatch_mode",
            "provider_free",
            "process_free",
            "content_id",
        }
        extra = set(payload).difference(allowed)
        if extra:
            raise ControlCatalogConformanceError(
                "control surface publication contains unknown fields: "
                + ", ".join(sorted(extra))
            )
        required = allowed.difference({"content_id"})
        missing = required.difference(payload)
        if missing:
            raise ControlCatalogConformanceError(
                "control surface publication is missing fields: "
                + ", ".join(sorted(missing))
            )
        if payload["schema"] != CONTROL_SURFACE_PUBLICATION_SCHEMA:
            raise ControlCatalogConformanceError(
                "control surface publication schema is unsupported"
            )
        for version_field in ("schema_version", "contract_version"):
            version = payload[version_field]
            if (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version != CONTROL_CONTRACT_VERSION
            ):
                raise ControlCatalogConformanceError(
                    f"control surface publication {version_field} is unsupported"
                )
        publication = cls(
            surface=payload["surface"],
            catalog_id=payload["catalog_id"],
            operations=tuple(payload["operations"]),
            request_schema_ids=payload["request_schema_ids"],
            result_schema_ids=payload["result_schema_ids"],
            behavior_ids=payload["behavior_ids"],
            dispatcher_ids=payload["dispatcher_ids"],
            dispatch_mode=payload["dispatch_mode"],
            catalog_version=payload["catalog_version"],
            provider_free=payload["provider_free"],
            process_free=payload["process_free"],
        )
        claimed_id = payload.get("content_id")
        if claimed_id not in (None, publication.content_id):
            raise ControlCatalogConformanceError(
                "control surface publication content_id does not match"
            )
        return publication


def control_operation_behavior_id(
    descriptor: ControlOperationDescriptor,
) -> str:
    """Return the transport-independent behavior fingerprint for an operation."""

    if not isinstance(descriptor, ControlOperationDescriptor):
        raise TypeError("descriptor must be a ControlOperationDescriptor")
    return _content_id(
        {
            "normalization_version": CONTROL_BEHAVIOR_NORMALIZATION_VERSION,
            "operation_descriptor_id": descriptor.content_id,
            "request_schema_id": descriptor.request_schema_id,
            "result_schema_id": descriptor.result_schema_id,
            "error_codes": tuple(item.value for item in ErrorCode),
            "statuses": tuple(item.value for item in OperationStatus),
            "timeout_code": ErrorCode.TIMED_OUT.value,
            "cancellation_code": ErrorCode.CANCELLED.value,
            "degradation": descriptor.degradation.value,
            "pagination": descriptor.pagination.content_id,
            "target": descriptor.target_descriptor.content_id,
        }
    )


def _validate_canonical_catalog(catalog: OperationCatalog) -> OperationCatalog:
    if not isinstance(catalog, OperationCatalog):
        raise ControlCatalogConformanceError(
            "catalog must be an OperationCatalog"
        )
    canonical = DEFAULT_CONTROL_CATALOG
    if catalog.catalog_version != CONTROL_CATALOG_VERSION:
        raise ControlCatalogConformanceError(
            "catalog version differs from the canonical service version"
        )
    if catalog.operation_names != canonical.operation_names:
        missing = sorted(
            set(canonical.operation_names).difference(catalog.operation_names)
        )
        extra = sorted(
            set(catalog.operation_names).difference(canonical.operation_names)
        )
        raise ControlCatalogConformanceError(
            "catalog operation population differs from the canonical "
            f"population; missing={missing}, extra={extra}"
        )
    for operation in canonical.operations:
        actual = catalog.operation(operation)
        expected = canonical.operation(operation)
        if actual.request_schema_id != expected.request_schema_id:
            raise ControlCatalogConformanceError(
                f"request schema drift for operation {operation.value}"
            )
        if actual.result_schema_id != expected.result_schema_id:
            raise ControlCatalogConformanceError(
                f"result schema drift for operation {operation.value}"
            )
        if actual.content_id != expected.content_id:
            raise ControlCatalogConformanceError(
                f"behavior drift for operation {operation.value}"
            )
    if catalog.content_id != canonical.content_id:
        raise ControlCatalogConformanceError(
            "catalog identity differs from the canonical catalog"
        )
    return catalog


def validate_control_surface_publication(
    publication: Union[ControlSurfacePublication, Mapping[str, Any]],
    catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
) -> ControlSurfacePublication:
    """Fail closed on missing, extra, schema-drifted, or behavioral entries."""

    canonical_catalog = _validate_canonical_catalog(catalog)
    if not isinstance(publication, ControlSurfacePublication):
        publication = ControlSurfacePublication.from_dict(publication)
    expected_operations = canonical_catalog.operations
    expected_names = frozenset(canonical_catalog.operation_names)
    if publication.catalog_version != canonical_catalog.catalog_version:
        raise ControlCatalogConformanceError(
            "publication catalog_version does not match the catalog"
        )
    if publication.catalog_id != canonical_catalog.content_id:
        raise ControlCatalogConformanceError(
            "publication catalog_id does not match the catalog"
        )
    if publication.operations != expected_operations:
        missing = sorted(expected_names.difference(publication.operation_names))
        extra = sorted(
            set(publication.operation_names).difference(expected_names)
        )
        raise ControlCatalogConformanceError(
            "publication operation population is not closed; "
            f"missing={missing}, extra={extra}"
        )
    populations = {
        "request_schema_ids": (
            publication.request_schema_ids,
            {
                item.operation.value: item.request_schema_id
                for item in canonical_catalog
            },
        ),
        "result_schema_ids": (
            publication.result_schema_ids,
            {
                item.operation.value: item.result_schema_id
                for item in canonical_catalog
            },
        ),
        "behavior_ids": (
            publication.behavior_ids,
            {
                item.operation.value: control_operation_behavior_id(item)
                for item in canonical_catalog
            },
        ),
    }
    for name, (actual, expected) in populations.items():
        if set(actual) != expected_names:
            missing = sorted(expected_names.difference(actual))
            extra = sorted(set(actual).difference(expected_names))
            raise ControlCatalogConformanceError(
                f"{name} population differs; missing={missing}, extra={extra}"
            )
        drifted = sorted(
            operation
            for operation, identity in expected.items()
            if actual[operation] != identity
        )
        if drifted:
            raise ControlCatalogConformanceError(
                f"{name} drift for operations {drifted}"
            )
    if set(publication.dispatcher_ids) != expected_names:
        missing = sorted(expected_names.difference(publication.dispatcher_ids))
        extra = sorted(
            set(publication.dispatcher_ids).difference(expected_names)
        )
        raise ControlCatalogConformanceError(
            "dispatcher population differs; "
            f"missing={missing}, extra={extra}"
        )
    if publication.dispatch_mode != "direct_service":
        raise ControlCatalogConformanceError(
            "control adapters must use direct_service dispatch"
        )
    if any(
        dispatcher != DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        for dispatcher in publication.dispatcher_ids.values()
    ):
        raise ControlCatalogConformanceError(
            "every operation must dispatch directly to "
            "SupervisorControlService.execute"
        )
    if not publication.provider_free or not publication.process_free:
        raise ControlCatalogConformanceError(
            "catalog publication must be provider-free and process-free"
        )
    return publication


def _normalized_absolute(value: Union[str, Path], *, label: str) -> Path:
    raw = os.fspath(value)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(os.path.normpath(raw))
    if path == Path(path.anchor):
        raise ValueError(f"{label} must not be the filesystem root")
    return path.resolve(strict=False)


def _normalize_allowlist(
    values: Iterable[Union[str, Path]], *, label: str
) -> tuple[Path, ...]:
    paths = {_normalized_absolute(item, label=label) for item in values}
    if not paths:
        raise ValueError(f"{label} must not be empty")
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_parameter(
    request: OperationRequest,
    *names: str,
    required: bool = True,
) -> str:
    for name in names:
        value = request.parameters.get(name)
        if value not in (None, ""):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a repository-relative string")
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise PathEscapeError(f"{name} must be repository-relative")
            return candidate.as_posix().removeprefix("./")
    if required:
        raise ValueError(f"one of {', '.join(names)} is required")
    return ""


def _bounded_window(request: OperationRequest) -> tuple[int, int]:
    raw_limit = request.parameters.get(
        "limit", min(DEFAULT_QUERY_LIMIT, request.bounds.max_items)
    )
    raw_offset = request.parameters.get("offset", 0)
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or raw_limit < 1
    ):
        raise ControlBoundsError("limit must be a positive integer")
    if raw_limit > request.bounds.max_items:
        raise ControlBoundsError("limit exceeds the request item bound")
    if (
        isinstance(raw_offset, bool)
        or not isinstance(raw_offset, int)
        or raw_offset < 0
    ):
        raise ControlBoundsError("offset must be a non-negative integer")
    if raw_offset > DEFAULT_MAX_OFFSET:
        raise ControlBoundsError("offset exceeds the absolute query bound")
    return raw_limit, raw_offset


def normalize_control_request(
    request: Union[OperationRequest, Mapping[str, Any]],
) -> OperationRequest:
    """Decode a request through the one canonical transport-neutral boundary."""

    if isinstance(request, OperationRequest):
        # Round-trip typed values too.  This catches a subclass or corrupted
        # in-memory record before it reaches a transport-independent backend.
        return OperationRequest.from_dict(request.to_record())
    return decode_operation_request(request)


def normalize_control_result(
    result: Union[OperationResult, Mapping[str, Any]],
    request: Union[OperationRequest, Mapping[str, Any]],
) -> OperationResult:
    """Decode and bind a result exactly as CLI and MCP adapters must."""

    canonical_request = normalize_control_request(request)
    canonical_result = (
        OperationResult.from_dict(result.to_record())
        if isinstance(result, OperationResult)
        else OperationResult.from_dict(result)
    )
    canonical_result.validate_against(canonical_request)
    return canonical_result


def _selector_text(value: Any, selector: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlContractError(
            f"target selector {selector} must be a non-empty string"
        )
    return value.strip()


def normalize_control_target(
    request: OperationRequest,
    descriptor: ControlOperationDescriptor,
    *,
    service_id: str = "ipfs-accelerate-agent-supervisor",
) -> Mapping[str, str]:
    """Resolve the catalog target descriptor to one canonical selector map."""

    if not isinstance(request, OperationRequest):
        raise TypeError("request must be an OperationRequest")
    if not isinstance(descriptor, ControlOperationDescriptor):
        raise TypeError("descriptor must be a ControlOperationDescriptor")
    if descriptor.operation is not request.operation:
        raise ControlCatalogConformanceError(
            "target descriptor operation does not match the request"
        )
    parameters = request.parameters
    declared_target = parameters.get("target", {})
    if declared_target and not isinstance(declared_target, Mapping):
        raise ControlContractError("target must be an object")
    declared_target = dict(declared_target)
    required = frozenset(
        descriptor.target_descriptor.required_selectors
    )
    extra = set(declared_target).difference(required)
    if extra:
        raise ControlContractError(
            "target contains selectors not declared for the operation: "
            + ", ".join(sorted(str(item) for item in extra))
        )
    target_id = parameters.get("target_id", "")
    defaults: Mapping[str, Any] = {
        "service_id": parameters.get("service_id") or target_id or service_id,
        "repository_id": request.repository_id,
        "tree_id": request.tree_id,
        "objective_id": request.objective_id,
        "task_id": parameters.get("task_id") or target_id,
        "bundle_id": parameters.get("bundle_id") or target_id,
        "lane_id": parameters.get("lane_id") or target_id,
        "stream_id": (
            parameters.get("stream_id")
            or target_id
            or f"{request.repository_id}:events"
        ),
        "receipt_id": parameters.get("receipt_id") or target_id,
        "cache_namespace": (
            parameters.get("cache_namespace") or target_id or "default"
        ),
        "artifact_id": parameters.get("artifact_id") or target_id,
        "validation_id": parameters.get("validation_id") or target_id,
        "preview_ref": parameters.get("preview_ref") or target_id,
        "incident_cid": parameters.get("incident_cid") or target_id,
        "rescue_plan_cid": parameters.get("rescue_plan_cid") or target_id,
    }
    canonical: dict[str, str] = {}
    for selector in descriptor.target_descriptor.required_selectors:
        top_level = parameters.get(selector)
        nested = declared_target.get(selector)
        if top_level not in (None, "") and nested not in (None, ""):
            if top_level != nested:
                raise ControlContractError(
                    f"target selector {selector} is inconsistent"
                )
        declared = top_level if top_level not in (None, "") else nested
        authoritative = defaults.get(selector)
        if selector in {
            "repository_id",
            "tree_id",
            "objective_id",
        }:
            if declared not in (None, "") and declared != authoritative:
                raise ControlContractError(
                    f"target selector {selector} does not match the request"
                )
            selected = authoritative
        else:
            selected = (
                declared if declared not in (None, "") else authoritative
            )
        canonical[selector] = _selector_text(selected, selector)
    return MappingProxyType(canonical)


def normalize_control_pagination(
    request: OperationRequest,
    descriptor: ControlOperationDescriptor,
    *,
    target: Union[Mapping[str, str], None] = None,
) -> Mapping[str, Any]:
    """Validate and normalize offset, query cursor, or event cursor input."""

    if descriptor.operation is not request.operation:
        raise ControlCatalogConformanceError(
            "pagination descriptor operation does not match the request"
        )
    explicit_limit = request.parameters.get("limit")
    page_limit = descriptor.pagination.validate_limit(explicit_limit)
    if page_limit > request.bounds.max_items:
        raise ControlBoundsError("page limit exceeds the request item bound")
    raw_offset = request.parameters.get("offset", 0)
    if (
        isinstance(raw_offset, bool)
        or not isinstance(raw_offset, int)
        or raw_offset < 0
        or raw_offset > DEFAULT_MAX_OFFSET
    ):
        raise ControlBoundsError("offset is outside the canonical query bound")
    if descriptor.pagination.kind is PaginationKind.NONE:
        if raw_offset:
            raise ControlBoundsError(
                "non-paginated operations cannot specify an offset"
            )
        return MappingProxyType(
            {"kind": PaginationKind.NONE.value, "limit": 1, "offset": 0}
        )
    raw_cursor = request.parameters.get(
        "event_cursor", request.parameters.get("cursor", "")
    )
    if descriptor.pagination.kind is PaginationKind.EVENT_CURSOR:
        selected_target = target or normalize_control_target(request, descriptor)
        stream_id = selected_target["stream_id"]
        if raw_cursor:
            if not isinstance(raw_cursor, str):
                raise EventCursorError("event cursor must be an opaque token")
            cursor = EventCursor.from_token(raw_cursor)
            cursor.assert_replayable(
                stream_id=stream_id,
                earliest_position=0,
                latest_position=max(cursor.position, raw_offset),
                snapshot_id=str(request.parameters.get("snapshot_id") or ""),
            )
        else:
            cursor = EventCursor.initial(
                stream_id,
                snapshot_id=str(request.parameters.get("snapshot_id") or ""),
            )
        return MappingProxyType(
            {
                "kind": PaginationKind.EVENT_CURSOR.value,
                "limit": page_limit,
                "offset": raw_offset,
                "cursor": cursor.to_token(),
                "stream_id": stream_id,
            }
        )
    if raw_cursor and not isinstance(raw_cursor, str):
        raise ControlContractError("query cursor must be an opaque string")
    return MappingProxyType(
        {
            "kind": PaginationKind.CURSOR.value,
            "limit": page_limit,
            "offset": raw_offset,
            "cursor": str(raw_cursor),
        }
    )


def validate_control_catalog_publication(
    catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
) -> OperationCatalog:
    """Validate the exact immutable catalog eligible for publication."""

    return _validate_canonical_catalog(catalog)


def validate_control_surface_manifest(
    manifest: Union[ControlDiscoveryManifest, Mapping[str, Any]],
    *,
    catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
) -> ControlDiscoveryManifest:
    """Reject a discovery manifest with missing, extra, or drifted schemas."""

    selected_catalog = _validate_canonical_catalog(catalog)
    selected_manifest = (
        manifest
        if isinstance(manifest, ControlDiscoveryManifest)
        else ControlDiscoveryManifest.from_dict(manifest)
    )
    if selected_manifest.operations != selected_catalog.operations:
        raise ControlCatalogConformanceError(
            "control surface manifest operation population differs from catalog"
        )
    expected_requests = {
        item.operation.value: item.request_schema_id
        for item in selected_catalog
    }
    expected_results = {
        item.operation.value: item.result_schema_id
        for item in selected_catalog
    }
    if dict(selected_manifest.request_schema_ids) != expected_requests:
        raise ControlCatalogConformanceError(
            "control surface manifest request schema population drift"
        )
    if dict(selected_manifest.result_schema_ids) != expected_results:
        raise ControlCatalogConformanceError(
            "control surface manifest result schema population drift"
        )
    return selected_manifest


def validate_operation_request_against_catalog(
    request: Union[OperationRequest, Mapping[str, Any]],
    *,
    catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
    service_id: str = "ipfs-accelerate-agent-supervisor",
) -> OperationRequest:
    """Normalize one request and enforce its exact catalog declaration."""

    selected_catalog = _validate_canonical_catalog(catalog)
    decoded = normalize_control_request(request)
    descriptor = selected_catalog.operation(decoded.operation)
    descriptor.validate_bounds(
        decoded.bounds,
        page_limit=decoded.parameters.get("limit"),
    )
    target = normalize_control_target(
        decoded,
        descriptor,
        service_id=service_id,
    )
    normalize_control_pagination(decoded, descriptor, target=target)
    if decoded.dry_run and not descriptor.supports_dry_run:
        raise ControlContractError(
            f"operation {decoded.operation.value} does not support dry-run"
        )
    return decoded


def _conformance_cli_exit_status(result: OperationResult) -> int:
    if result.succeeded:
        return 0
    if result.status is OperationStatus.CONFLICT:
        return 3
    if result.status is OperationStatus.NOT_FOUND:
        return 4
    if result.status is OperationStatus.DENIED:
        return 2
    if result.status is OperationStatus.TIMED_OUT:
        return 124
    if result.status is OperationStatus.CANCELLED:
        return 130
    return 1


@dataclass(frozen=True)
class ControlOperationConformanceCase:
    """One independently invoked Python/CLI/MCP catalog fixture."""

    scenario: str
    request: Union[OperationRequest, Mapping[str, Any]]
    python_result: Union[OperationResult, Mapping[str, Any]]
    cli_result: Union[OperationResult, Mapping[str, Any]]
    mcp_result: Union[OperationResult, Mapping[str, Any]]
    cli_exit_status: int

    def __post_init__(self) -> None:
        scenario = str(self.scenario).strip()
        if not scenario or len(scenario.encode("utf-8")) > 256:
            raise ControlContractError(
                "conformance scenario must be bounded text"
            )
        object.__setattr__(self, "scenario", scenario)
        request = validate_operation_request_against_catalog(self.request)
        object.__setattr__(self, "request", request)
        results: list[OperationResult] = []
        for name in ("python_result", "cli_result", "mcp_result"):
            value = getattr(self, name)
            result = (
                value
                if isinstance(value, OperationResult)
                else OperationResult.from_dict(value)
                if isinstance(value, Mapping)
                else None
            )
            if result is None:
                raise ControlContractError(
                    f"{name} must be an OperationResult"
                )
            result = normalize_control_result(result, request)
            object.__setattr__(self, name, result)
            results.append(result)
        canonical = tuple(
            canonical_control_json_bytes(item.to_record())
            for item in results
        )
        if canonical[1:] != canonical[:-1]:
            raise ControlContractError(
                "Python, CLI, and MCP conformance results are behaviorally "
                "inconsistent"
            )
        if (
            isinstance(self.cli_exit_status, bool)
            or not isinstance(self.cli_exit_status, int)
            or self.cli_exit_status
            != _conformance_cli_exit_status(results[1])
        ):
            raise ControlContractError(
                "CLI exit status does not match the canonical operation result"
            )

    @property
    def operation(self) -> Operation:
        assert isinstance(self.request, OperationRequest)
        return self.request.operation

    @property
    def status(self) -> OperationStatus:
        assert isinstance(self.python_result, OperationResult)
        return self.python_result.status

    @property
    def error_code(self) -> str:
        assert isinstance(self.python_result, OperationResult)
        return (
            self.python_result.error.code.value
            if self.python_result.error is not None
            else ""
        )

    @property
    def effect_ids(self) -> tuple[str, ...]:
        assert isinstance(self.python_result, OperationResult)
        return tuple(item.effect_id for item in self.python_result.effects)

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.request, OperationRequest)
        assert isinstance(self.python_result, OperationResult)
        assert isinstance(self.cli_result, OperationResult)
        assert isinstance(self.mcp_result, OperationResult)
        return {
            "schema": CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA,
            "schema_version": 2,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "scenario": self.scenario,
            "operation": self.operation.value,
            "status": self.status.value,
            "error_code": self.error_code,
            "effect_ids": self.effect_ids,
            "cli_exit_status": self.cli_exit_status,
            "request": self.request.to_record(),
            "python_result": self.python_result.to_record(),
            "cli_result": self.cli_result.to_record(),
            "mcp_result": self.mcp_result.to_record(),
        }

    @property
    def content_id(self) -> str:
        return _content_id(self._payload())

    def to_record(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    to_dict = to_record

    def canonical_bytes(self) -> bytes:
        return canonical_control_json_bytes(self.to_record())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlOperationConformanceCase":
        if not isinstance(payload, Mapping):
            raise ControlContractError("conformance case must be an object")
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "scenario",
            "operation",
            "status",
            "error_code",
            "effect_ids",
            "cli_exit_status",
            "request",
            "python_result",
            "cli_result",
            "mcp_result",
            "content_id",
        }
        extra = set(payload).difference(allowed)
        if extra:
            raise ControlContractError(
                "conformance case contains unknown fields: "
                + ", ".join(sorted(extra))
            )
        if (
            payload.get("schema")
            != CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA
            or payload.get("schema_version") != 2
            or payload.get("contract_version") != CONTROL_CONTRACT_VERSION
        ):
            raise ControlContractError("conformance case schema is invalid")
        result = cls(
            scenario=payload.get("scenario", ""),
            request=payload.get("request") or {},
            python_result=payload.get("python_result") or {},
            cli_result=payload.get("cli_result") or {},
            mcp_result=payload.get("mcp_result") or {},
            cli_exit_status=payload.get("cli_exit_status", -1),
        )
        if payload.get("operation") != result.operation.value:
            raise ControlContractError("conformance case operation drift")
        if payload.get("status") != result.status.value:
            raise ControlContractError("conformance case status drift")
        if payload.get("error_code") != result.error_code:
            raise ControlContractError("conformance case error drift")
        if tuple(payload.get("effect_ids", ())) != result.effect_ids:
            raise ControlContractError("conformance case effect drift")
        if payload.get("content_id") not in (None, result.content_id):
            raise ControlContractError("conformance case identity mismatch")
        return result


@dataclass(frozen=True)
class ControlCatalogConformanceEvidence:
    """Publication receipt for the exact v2 catalog and transport matrix."""

    catalog: Union[OperationCatalog, Mapping[str, Any]]
    manifests: tuple[
        Union[ControlDiscoveryManifest, Mapping[str, Any]], ...
    ]
    cases: tuple[
        Union[ControlOperationConformanceCase, Mapping[str, Any]], ...
    ]
    requirement_id: str = CONTROL_CONFORMANCE_V2_REQUIREMENT_ID

    def __post_init__(self) -> None:
        catalog = self.catalog
        if not isinstance(catalog, OperationCatalog):
            if not isinstance(catalog, Mapping):
                raise ControlContractError("conformance catalog is malformed")
            catalog = OperationCatalog.from_dict(catalog)
        catalog = validate_control_catalog_publication(catalog)
        object.__setattr__(self, "catalog", catalog)
        if self.requirement_id != CONTROL_CONFORMANCE_V2_REQUIREMENT_ID:
            raise ControlContractError(
                "control conformance requirement identity mismatch"
            )

        manifests: list[ControlDiscoveryManifest] = []
        for value in self.manifests:
            manifest = (
                value
                if isinstance(value, ControlDiscoveryManifest)
                else ControlDiscoveryManifest.from_dict(value)
                if isinstance(value, Mapping)
                else None
            )
            if manifest is None:
                raise ControlContractError(
                    "conformance manifest is malformed"
                )
            manifests.append(
                validate_control_surface_manifest(manifest, catalog=catalog)
            )
        manifests.sort(key=lambda item: item.surface.value)
        expected_surfaces = tuple(
            sorted(ControlSurface, key=lambda item: item.value)
        )
        if (
            tuple(item.surface for item in manifests) != expected_surfaces
            or len({item.surface for item in manifests})
            != len(expected_surfaces)
        ):
            raise ControlContractError(
                "catalog publication requires exactly Python, CLI, and MCP "
                "manifests"
            )
        if len({item.schema_population_id for item in manifests}) != 1:
            raise ControlContractError(
                "control surface schema populations are inconsistent"
            )
        object.__setattr__(self, "manifests", tuple(manifests))

        cases: list[ControlOperationConformanceCase] = []
        for value in self.cases:
            case = (
                value
                if isinstance(value, ControlOperationConformanceCase)
                else ControlOperationConformanceCase.from_dict(value)
                if isinstance(value, Mapping)
                else None
            )
            if case is None:
                raise ControlContractError(
                    "conformance case is malformed"
                )
            cases.append(case)
        if not cases:
            raise ControlContractError(
                "catalog publication requires conformance cases"
            )
        scenario_keys = {(item.operation, item.scenario) for item in cases}
        if len(scenario_keys) != len(cases):
            raise ControlContractError("conformance cases must be unique")
        operation_population = [item.operation for item in cases]
        if len(operation_population) != len(set(operation_population)):
            raise ControlContractError(
                "catalog publication requires exactly one conformance case "
                "per operation"
            )
        actual_operations = {item.operation for item in cases}
        expected_operations = set(catalog.operations)
        if actual_operations != expected_operations:
            missing = sorted(
                item.value
                for item in expected_operations - actual_operations
            )
            extra = sorted(
                item.value
                for item in actual_operations - expected_operations
            )
            raise ControlContractError(
                "catalog publication conformance population drift; "
                f"missing={missing}, extra={extra}"
            )
        object.__setattr__(
            self,
            "cases",
            tuple(
                sorted(
                    cases,
                    key=lambda item: (
                        item.operation.value,
                        item.scenario,
                    ),
                )
            ),
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (CONTROL_CONFORMANCE_V2_REQUIREMENT_ID,)

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def catalog_id(self) -> str:
        assert isinstance(self.catalog, OperationCatalog)
        return self.catalog.catalog_id

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.catalog, OperationCatalog)
        return {
            "schema": CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA,
            "schema_version": 2,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "catalog_id": self.catalog_id,
            "catalog": self.catalog.to_record(),
            "manifests": tuple(
                item.to_record() for item in self.manifests
            ),
            "cases": tuple(item.to_record() for item in self.cases),
        }

    @property
    def content_id(self) -> str:
        return _content_id(self._payload())

    def to_record(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    to_dict = to_record

    def canonical_bytes(self) -> bytes:
        return canonical_control_json_bytes(self.to_record())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlCatalogConformanceEvidence":
        if not isinstance(payload, Mapping):
            raise ControlContractError(
                "conformance evidence must be an object"
            )
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "requirement_id",
            "catalog_id",
            "catalog",
            "manifests",
            "cases",
            "content_id",
        }
        extra = set(payload).difference(allowed)
        if extra:
            raise ControlContractError(
                "conformance evidence contains unknown fields: "
                + ", ".join(sorted(extra))
            )
        if (
            payload.get("schema")
            != CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA
            or payload.get("schema_version") != 2
            or payload.get("contract_version") != CONTROL_CONTRACT_VERSION
        ):
            raise ControlContractError(
                "conformance evidence schema is invalid"
            )
        result = cls(
            catalog=payload.get("catalog") or {},
            manifests=tuple(payload.get("manifests", ())),
            cases=tuple(payload.get("cases", ())),
            requirement_id=payload.get("requirement_id", ""),
        )
        if payload.get("catalog_id") != result.catalog_id:
            raise ControlContractError(
                "conformance evidence catalog identity mismatch"
            )
        if payload.get("content_id") not in (None, result.content_id):
            raise ControlContractError(
                "conformance evidence identity mismatch"
            )
        return result


def validate_catalog_publication(
    catalog: OperationCatalog,
    manifests: Sequence[
        Union[ControlDiscoveryManifest, Mapping[str, Any]]
    ],
    cases: Sequence[
        Union[ControlOperationConformanceCase, Mapping[str, Any]]
    ],
) -> ControlCatalogConformanceEvidence:
    """Publish only after the complete three-surface matrix passes."""

    return ControlCatalogConformanceEvidence(
        catalog=catalog,
        manifests=tuple(manifests),
        cases=tuple(cases),
    )


publish_control_catalog = validate_catalog_publication


class MutationTransactionPhase(str, Enum):
    """Durable phases for one idempotency-bound mutation transaction."""

    PREPARED = "prepared"
    DISPATCHING = "dispatching"
    COMMITTED = "committed"
    COMPENSATION_REQUIRED = "compensation_required"
    REPAIR_REQUIRED = "repair_required"
    COMPENSATED = "compensated"
    REPAIRED = "repaired"

    @property
    def terminal(self) -> bool:
        return self in {
            MutationTransactionPhase.COMMITTED,
            MutationTransactionPhase.COMPENSATED,
            MutationTransactionPhase.REPAIRED,
        }

    @property
    def requires_recovery(self) -> bool:
        return self in {
            MutationTransactionPhase.COMPENSATION_REQUIRED,
            MutationTransactionPhase.REPAIR_REQUIRED,
        }


class MutationRecoveryAction(str, Enum):
    """Closed recovery vocabulary for an interrupted multi-step mutation."""

    NONE = "none"
    COMPENSATE = "compensate"
    REPAIR = "repair"


_LEGAL_MUTATION_TRANSACTION_TRANSITIONS: Final[
    Mapping[MutationTransactionPhase, frozenset[MutationTransactionPhase]]
] = MappingProxyType(
    {
        MutationTransactionPhase.PREPARED: frozenset(
            {MutationTransactionPhase.DISPATCHING}
        ),
        MutationTransactionPhase.DISPATCHING: frozenset(
            {
                MutationTransactionPhase.COMMITTED,
                MutationTransactionPhase.COMPENSATION_REQUIRED,
                MutationTransactionPhase.REPAIR_REQUIRED,
            }
        ),
        MutationTransactionPhase.COMPENSATION_REQUIRED: frozenset(
            {MutationTransactionPhase.COMPENSATED}
        ),
        MutationTransactionPhase.REPAIR_REQUIRED: frozenset(
            {MutationTransactionPhase.REPAIRED}
        ),
        MutationTransactionPhase.COMMITTED: frozenset(),
        MutationTransactionPhase.COMPENSATED: frozenset(),
        MutationTransactionPhase.REPAIRED: frozenset(),
    }
)


@dataclass(frozen=True)
class MutationTransactionState:
    """Compare-and-swap state for a real mutation.

    The transaction identity is derived only from the complete request and its
    caller-scoped idempotency key.  ``revision`` changes on every durable
    transition and is the compare-and-swap token exposed to recovery tooling.
    A stored result makes terminal and partial-failure replay exact.
    """

    request_id: str
    operation: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    idempotency_key: str
    lease_id: str
    fencing_epoch: int
    effect_ids: tuple[str, ...]
    phase: MutationTransactionPhase = MutationTransactionPhase.PREPARED
    revision: int = 0
    applied_effect_ids: tuple[str, ...] = ()
    recovery_action: MutationRecoveryAction = MutationRecoveryAction.NONE
    failure_code: str = ""
    result: Union[OperationResult, None] = None
    updated_at_ms: int = 0
    transaction_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "operation",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "idempotency_key",
            "lease_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value or "\x00" in value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        if isinstance(self.fencing_epoch, bool) or not isinstance(
            self.fencing_epoch, int
        ) or self.fencing_epoch < 0:
            raise ValueError("fencing_epoch must be a non-negative integer")
        for name in ("revision", "updated_at_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "phase",
            MutationTransactionPhase(
                str(getattr(self.phase, "value", self.phase))
            ),
        )
        object.__setattr__(
            self,
            "recovery_action",
            MutationRecoveryAction(
                str(getattr(self.recovery_action, "value", self.recovery_action))
            ),
        )
        effect_ids = tuple(sorted({str(item).strip() for item in self.effect_ids}))
        applied = tuple(
            sorted({str(item).strip() for item in self.applied_effect_ids})
        )
        if not effect_ids or any(not item for item in effect_ids):
            raise ValueError("effect_ids must contain non-empty values")
        if any(not item for item in applied) or not set(applied).issubset(effect_ids):
            raise ValueError("applied_effect_ids must be a subset of effect_ids")
        object.__setattr__(self, "effect_ids", effect_ids)
        object.__setattr__(self, "applied_effect_ids", applied)
        object.__setattr__(
            self,
            "failure_code",
            str(self.failure_code).strip(),
        )
        if self.phase is MutationTransactionPhase.COMPENSATION_REQUIRED:
            expected_recovery = MutationRecoveryAction.COMPENSATE
        elif self.phase is MutationTransactionPhase.REPAIR_REQUIRED:
            expected_recovery = MutationRecoveryAction.REPAIR
        else:
            expected_recovery = MutationRecoveryAction.NONE
        if self.recovery_action is not expected_recovery:
            raise ValueError("recovery_action does not match transaction phase")
        if self.phase.requires_recovery and not self.failure_code:
            raise ValueError("recovery-required transactions need a failure_code")
        if self.result is not None:
            if not isinstance(self.result, OperationResult):
                raise TypeError("transaction result must be an OperationResult")
            if self.result.request_id != self.request_id:
                raise ValueError("transaction result does not match request_id")
        identity = _content_id(self._identity_payload())
        if self.transaction_id and self.transaction_id != identity:
            raise ValueError("transaction_id does not match its immutable binding")
        object.__setattr__(self, "transaction_id", identity)

    @classmethod
    def prepare(
        cls, request: OperationRequest, *, now_ms: int
    ) -> "MutationTransactionState":
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            raise ValueError("only real mutations have transaction state")
        return cls(
            request_id=request.request_id,
            operation=request.operation.value,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            objective_revision=request.objective_revision,
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
            caller=request.caller,
            idempotency_key=request.idempotency_key,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch
            if request.fencing_epoch is not None
            else -1,
            effect_ids=tuple(item.effect_id for item in request.expected_effects),
            updated_at_ms=now_ms,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_MUTATION_TRANSACTION_SCHEMA,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "idempotency_key": self.idempotency_key,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "effect_ids": self.effect_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "transaction_id": self.transaction_id,
            "phase": self.phase.value,
            "revision": self.revision,
            "applied_effect_ids": list(self.applied_effect_ids),
            "recovery_action": self.recovery_action.value,
            "failure_code": self.failure_code,
            "result": self.result.to_record() if self.result is not None else None,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutationTransactionState":
        if payload.get("schema") != CONTROL_MUTATION_TRANSACTION_SCHEMA:
            raise ValueError("unknown mutation transaction schema")
        if payload.get("contract_version") != CONTROL_CONTRACT_VERSION:
            raise ValueError("unsupported mutation transaction contract version")
        allowed = {
            "schema",
            "contract_version",
            "request_id",
            "operation",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "idempotency_key",
            "lease_id",
            "fencing_epoch",
            "effect_ids",
            "transaction_id",
            "phase",
            "revision",
            "applied_effect_ids",
            "recovery_action",
            "failure_code",
            "result",
            "updated_at_ms",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "mutation transaction contains unknown fields: "
                + ", ".join(unknown)
            )
        result_payload = payload.get("result")
        if result_payload is not None and not isinstance(result_payload, Mapping):
            raise ValueError("mutation transaction result must be an object or null")
        return cls(
            request_id=payload.get("request_id", ""),
            operation=payload.get("operation", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            caller=payload.get("caller", ""),
            idempotency_key=payload.get("idempotency_key", ""),
            lease_id=payload.get("lease_id", ""),
            fencing_epoch=payload.get("fencing_epoch", -1),
            effect_ids=tuple(payload.get("effect_ids") or ()),
            phase=payload.get("phase", ""),
            revision=payload.get("revision", -1),
            applied_effect_ids=tuple(payload.get("applied_effect_ids") or ()),
            recovery_action=payload.get("recovery_action", ""),
            failure_code=payload.get("failure_code", ""),
            result=(
                OperationResult.from_dict(result_payload)
                if isinstance(result_payload, Mapping)
                else None
            ),
            updated_at_ms=payload.get("updated_at_ms", -1),
            transaction_id=payload.get("transaction_id", ""),
        )


@dataclass(frozen=True)
class BackendResponse:
    """Normalized return from a direct Python operation adapter.

    ``changed`` is required for mutating operations.  A mapping returned
    directly by a handler is accepted for compatibility and means that all
    declared effects were applied when the operation is a real mutation.
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    changed: bool = False
    applied_effect_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.data, Mapping):
            raise TypeError("backend response data must be a mapping")
        if not isinstance(self.changed, bool):
            raise TypeError("backend response changed must be boolean")
        effect_ids = tuple(
            sorted({str(item).strip() for item in self.applied_effect_ids})
        )
        if any(not item for item in effect_ids):
            raise ValueError("applied effect IDs must not be empty")
        object.__setattr__(self, "applied_effect_ids", effect_ids)
        object.__setattr__(
            self,
            "warnings",
            tuple(sorted({str(item).strip() for item in self.warnings if str(item).strip()})),
        )
        object.__setattr__(
            self,
            "checks",
            tuple(sorted({str(item).strip() for item in self.checks if str(item).strip()})),
        )


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
_LIFECYCLE_UNSET: Final[object] = object()


class InMemoryLifecycleStore:
    """Thread-safe authoritative lifecycle snapshots and bounded event replay."""

    def __init__(
        self,
        initial_status: Union[LifecycleStatus, None] = None,
        *,
        max_events: int = DEFAULT_MAX_CONTROL_EVENTS,
    ) -> None:
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
            or max_events < 1
            or max_events > 4096
        ):
            raise ValueError("max_events must be an integer in [1, 4096]")
        self._lock = threading.RLock()
        self._max_events = max_events
        self._statuses: dict[str, LifecycleStatus] = {}
        self._events: deque[LifecycleEvent] = deque(maxlen=max_events)
        self._sequence = 0
        if initial_status is not None:
            self._statuses[initial_status.target_id] = initial_status

    @property
    def max_events(self) -> int:
        return self._max_events

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            yield

    def _default(self, target_id: str, now_ms: int) -> LifecycleStatus:
        return LifecycleStatus(
            target_id=target_id,
            heartbeat_at_ms=now_ms,
            updated_at_ms=now_ms,
        )

    def _append_event_locked(
        self,
        *,
        target_id: str,
        action: str,
        accepted: bool,
        previous_state: SupervisorLifecycleState,
        state: SupervisorLifecycleState,
        reason: str,
        request_id: str,
        occurred_at_ms: int,
        changed: bool,
        replayed: bool = False,
        recovered: bool = False,
        fencing_epoch: Union[int, None] = None,
    ) -> LifecycleEvent:
        self._sequence += 1
        event = LifecycleEvent(
            sequence=self._sequence,
            target_id=target_id,
            action=action,
            accepted=accepted,
            previous_state=previous_state,
            state=state,
            reason=str(reason).strip(),
            request_id=str(request_id).strip(),
            occurred_at_ms=occurred_at_ms,
            changed=changed,
            replayed=replayed,
            recovered=recovered,
            fencing_epoch=fencing_epoch,
        )
        self._events.append(event)
        return event

    def seed(self, status: LifecycleStatus) -> None:
        """Install a snapshot for migration/recovery tests and process handoff."""

        if not isinstance(status, LifecycleStatus):
            raise TypeError("status must be a LifecycleStatus")
        with self._lock:
            self._statuses[status.target_id] = status

    def _recover_locked(
        self,
        status: LifecycleStatus,
        *,
        now_ms: int,
        pid_alive: Callable[[int], bool],
        stale_after_ms: int,
    ) -> LifecycleStatus:
        if status.pid is None:
            return status
        heartbeat_stale = (
            status.heartbeat_at_ms <= 0
            or now_ms - status.heartbeat_at_ms >= stale_after_ms
        )
        try:
            alive = bool(pid_alive(status.pid))
        except Exception:
            alive = False
        if alive:
            if (
                heartbeat_stale
                and status.state is SupervisorLifecycleState.HEALTHY
            ):
                degraded = replace(
                    status,
                    state=SupervisorLifecycleState.DEGRADED,
                    phase="heartbeat_stale",
                    transition_id=_content_id(
                        {
                            "target_id": status.target_id,
                            "from": status.state.value,
                            "to": SupervisorLifecycleState.DEGRADED.value,
                            "reason": "heartbeat_stale",
                            "at": now_ms,
                        }
                    ),
                    updated_at_ms=now_ms,
                )
                self._statuses[status.target_id] = degraded
                self._append_event_locked(
                    target_id=status.target_id,
                    action="recover",
                    accepted=True,
                    previous_state=status.state,
                    state=degraded.state,
                    reason="heartbeat_stale",
                    request_id="",
                    occurred_at_ms=now_ms,
                    changed=True,
                    recovered=True,
                    fencing_epoch=status.fencing_epoch,
                )
                return degraded
            return status
        if status.state is SupervisorLifecycleState.STARTING:
            recovered_state = SupervisorLifecycleState.STOPPED
            terminal_reason = "interrupted_start_stale_pid"
        elif status.state is SupervisorLifecycleState.STOPPING:
            recovered_state = SupervisorLifecycleState.STOPPED
            terminal_reason = "stop_completed_after_stale_pid"
        elif status.state is SupervisorLifecycleState.DRAINING:
            recovered_state = SupervisorLifecycleState.FAILED
            terminal_reason = "drain_interrupted_stale_pid"
        elif status.state in {
            SupervisorLifecycleState.HEALTHY,
            SupervisorLifecycleState.DEGRADED,
            SupervisorLifecycleState.PAUSED,
            SupervisorLifecycleState.BLOCKED,
        }:
            recovered_state = SupervisorLifecycleState.FAILED
            terminal_reason = "runtime_stale_pid"
        else:
            return status
        recovered = replace(
            status,
            state=recovered_state,
            phase="recovered",
            pid=None,
            active_leases=(),
            terminal_reason=terminal_reason,
            transition_id=_content_id(
                {
                    "target_id": status.target_id,
                    "from": status.state.value,
                    "to": recovered_state.value,
                    "reason": terminal_reason,
                    "at": now_ms,
                }
            ),
            updated_at_ms=now_ms,
        )
        self._statuses[status.target_id] = recovered
        self._append_event_locked(
            target_id=status.target_id,
            action="recover",
            accepted=True,
            previous_state=status.state,
            state=recovered_state,
            reason=terminal_reason,
            request_id="",
            occurred_at_ms=now_ms,
            changed=True,
            recovered=True,
            fencing_epoch=status.fencing_epoch,
        )
        return recovered

    def snapshot(
        self,
        target_id: str,
        *,
        now_ms: int,
        pid_alive: Callable[[int], bool],
        stale_after_ms: int,
        recover: bool = True,
    ) -> LifecycleStatus:
        target = str(target_id).strip()
        if not target:
            raise ValueError("target_id is required")
        with self._lock:
            status = self._statuses.get(target) or self._default(target, now_ms)
            self._statuses.setdefault(target, status)
            if recover:
                status = self._recover_locked(
                    status,
                    now_ms=now_ms,
                    pid_alive=pid_alive,
                    stale_after_ms=stale_after_ms,
                )
            return status

    @staticmethod
    def _idempotent_action(
        operation: Operation, state: SupervisorLifecycleState
    ) -> bool:
        return (
            (operation is Operation.START and state in {
                SupervisorLifecycleState.STARTING,
                SupervisorLifecycleState.HEALTHY,
                SupervisorLifecycleState.DEGRADED,
            })
            or (operation is Operation.PAUSE and state is SupervisorLifecycleState.PAUSED)
            or (
                operation is Operation.RESUME
                and state is SupervisorLifecycleState.HEALTHY
            )
            or (operation is Operation.DRAIN and state in {
                SupervisorLifecycleState.DRAINING,
                SupervisorLifecycleState.STOPPED,
            })
            or (operation in {Operation.STOP, Operation.CANCEL} and state in {
                SupervisorLifecycleState.STOPPING,
                SupervisorLifecycleState.STOPPED,
            })
            or (operation is Operation.RETRY and state is SupervisorLifecycleState.STARTING)
            or (
                operation is Operation.QUARANTINE
                and state is SupervisorLifecycleState.BLOCKED
            )
        )

    def transition(
        self,
        request: OperationRequest,
        *,
        now_ms: int,
        pid_alive: Callable[[int], bool],
        stale_after_ms: int,
    ) -> tuple[LifecycleStatus, LifecycleStatus, LifecycleEvent]:
        operation = request.operation
        if operation not in _ACTION_REQUESTED_STATE:
            raise ValueError("request is not a lifecycle operation")
        target_id = str(request.parameters.get("target_id") or "supervisor").strip()
        reason = str(request.parameters.get("reason") or operation.value).strip()
        requested = _ACTION_REQUESTED_STATE[operation]
        requested_value = str(
            request.parameters.get("requested_state") or ""
        ).strip()
        if requested_value and requested_value not in {
            requested.value,
            operation.value,
        }:
            raise InvalidLifecycleTransitionError(
                f"{operation.value} requests {requested.value}, not "
                f"{requested_value}"
            )
        with self._lock:
            original = self._statuses.get(target_id) or self._default(
                target_id, now_ms
            )
            self._statuses.setdefault(target_id, original)
            for prior_event in reversed(self._events):
                if (
                    prior_event.request_id == request.request_id
                    and prior_event.action == operation.value
                    and prior_event.accepted
                ):
                    return original, original, prior_event
            previous = self._recover_locked(
                original,
                now_ms=now_ms,
                pid_alive=pid_alive,
                stale_after_ms=stale_after_ms,
            )
            recovered = previous is not original
            if (
                previous.fencing_epoch is not None
                and request.fencing_epoch is not None
                and request.fencing_epoch < previous.fencing_epoch
            ):
                event = self._append_event_locked(
                    target_id=target_id,
                    action=operation.value,
                    accepted=False,
                    previous_state=previous.state,
                    state=previous.state,
                    reason=(
                        f"stale fencing epoch {request.fencing_epoch}; "
                        f"current epoch is {previous.fencing_epoch}"
                    ),
                    request_id=request.request_id,
                    occurred_at_ms=now_ms,
                    changed=False,
                    fencing_epoch=request.fencing_epoch,
                )
                raise StaleLeaseError(event.reason)
            if self._idempotent_action(operation, previous.state):
                status = previous
                if (
                    request.fencing_epoch is not None
                    and (
                        previous.fencing_epoch is None
                        or request.fencing_epoch > previous.fencing_epoch
                    )
                ):
                    status = replace(
                        previous,
                        fencing_epoch=request.fencing_epoch,
                        updated_at_ms=now_ms,
                    )
                    self._statuses[target_id] = status
                event = self._append_event_locked(
                    target_id=target_id,
                    action=operation.value,
                    accepted=True,
                    previous_state=previous.state,
                    state=status.state,
                    reason=reason,
                    request_id=request.request_id,
                    occurred_at_ms=now_ms,
                    changed=False,
                    replayed=True,
                    recovered=recovered,
                    fencing_epoch=request.fencing_epoch,
                )
                return previous, status, event
            if not lifecycle_transition_is_legal(previous.state, requested):
                event = self._append_event_locked(
                    target_id=target_id,
                    action=operation.value,
                    accepted=False,
                    previous_state=previous.state,
                    state=previous.state,
                    reason=(
                        f"invalid transition {previous.state.value}"
                        f" -> {requested.value}: {reason}"
                    ),
                    request_id=request.request_id,
                    occurred_at_ms=now_ms,
                    changed=False,
                    fencing_epoch=request.fencing_epoch,
                )
                raise InvalidLifecycleTransitionError(event.reason)
            generation = previous.generation
            if operation in {Operation.START, Operation.RETRY}:
                generation += 1
            phase = str(request.parameters.get("phase") or requested.value).strip()
            pid_value = request.parameters.get("pid")
            pid = previous.pid
            if pid_value not in (None, ""):
                pid = int(pid_value)
            if requested is SupervisorLifecycleState.STARTING:
                # A recovered start must never retain the stale predecessor.
                pid = None if pid_value in (None, "") else pid
            status = replace(
                previous,
                state=requested,
                phase=phase,
                pid=pid,
                refill_state=(
                    "paused"
                    if operation is Operation.PAUSE
                    else "draining"
                    if operation is Operation.DRAIN
                    else "idle"
                    if operation in {
                        Operation.START,
                        Operation.RESUME,
                        Operation.RETRY,
                    }
                    else previous.refill_state
                ),
                backpressure=operation
                in {
                    Operation.PAUSE,
                    Operation.DRAIN,
                    Operation.STOP,
                    Operation.CANCEL,
                    Operation.QUARANTINE,
                },
                backpressure_reasons=(
                    (operation.value,)
                    if operation
                    in {
                        Operation.PAUSE,
                        Operation.DRAIN,
                        Operation.STOP,
                        Operation.CANCEL,
                        Operation.QUARANTINE,
                    }
                    else ()
                ),
                terminal_reason=(
                    reason
                    if requested
                    in {
                        SupervisorLifecycleState.STOPPING,
                        SupervisorLifecycleState.FAILED,
                    }
                    else ""
                ),
                transition_id=_content_id(
                    {
                        "request_id": request.request_id,
                        "target_id": target_id,
                        "from": previous.state.value,
                        "to": requested.value,
                        "generation": generation,
                    }
                ),
                generation=generation,
                fencing_epoch=request.fencing_epoch,
                heartbeat_at_ms=now_ms,
                updated_at_ms=now_ms,
            )
            self._statuses[target_id] = status
            event = self._append_event_locked(
                target_id=target_id,
                action=operation.value,
                accepted=True,
                previous_state=previous.state,
                state=status.state,
                reason=reason,
                request_id=request.request_id,
                occurred_at_ms=now_ms,
                changed=True,
                recovered=recovered,
                fencing_epoch=request.fencing_epoch,
            )
            return previous, status, event

    def heartbeat(
        self,
        target_id: str,
        *,
        now_ms: int,
        state: Union[SupervisorLifecycleState, str, None] = None,
        phase: Union[str, None] = None,
        pid: Any = _LIFECYCLE_UNSET,
        active_leases: Union[Iterable[str], None] = None,
        refill_state: Union[str, None] = None,
        backpressure: Union[bool, None] = None,
        backpressure_reasons: Union[Iterable[str], None] = None,
        terminal_reason: Union[str, None] = None,
    ) -> LifecycleStatus:
        """Record liveness and operational dimensions in the same schema."""

        target = str(target_id).strip()
        with self._lock:
            previous = self._statuses.get(target) or self._default(target, now_ms)
            requested = previous.state if state is None else (
                state
                if isinstance(state, SupervisorLifecycleState)
                else SupervisorLifecycleState(str(state))
            )
            if requested is not previous.state and not lifecycle_transition_is_legal(
                previous.state, requested
            ):
                raise InvalidLifecycleTransitionError(
                    f"invalid heartbeat transition {previous.state.value}"
                    f" -> {requested.value}"
                )
            leases = (
                previous.active_leases
                if active_leases is None
                else _lifecycle_text_tuple(active_leases)
            )
            effective_pid = previous.pid if pid is _LIFECYCLE_UNSET else pid
            if (
                previous.state is SupervisorLifecycleState.DRAINING
                and not leases
                and requested is SupervisorLifecycleState.DRAINING
            ):
                requested = (
                    SupervisorLifecycleState.STOPPED
                    if effective_pid is None
                    else SupervisorLifecycleState.STOPPING
                )
                terminal_reason = terminal_reason or previous.terminal_reason or (
                    "drained"
                    if requested is SupervisorLifecycleState.STOPPED
                    else "drain_complete_stopping"
                )
            if (
                previous.state is SupervisorLifecycleState.STOPPING
                and requested is SupervisorLifecycleState.STOPPING
                and not leases
                and pid is None
            ):
                requested = SupervisorLifecycleState.STOPPED
                terminal_reason = (
                    terminal_reason or previous.terminal_reason or "stopped"
                )
            status = replace(
                previous,
                state=requested,
                phase=str(phase or requested.value).strip(),
                heartbeat_at_ms=now_ms,
                pid=(
                    None
                    if requested
                    in {
                        SupervisorLifecycleState.STOPPED,
                        SupervisorLifecycleState.FAILED,
                    }
                    else effective_pid
                ),
                active_leases=leases,
                refill_state=(
                    previous.refill_state
                    if refill_state is None
                    else str(refill_state)
                ),
                backpressure=(
                    previous.backpressure
                    if backpressure is None
                    else backpressure
                ),
                backpressure_reasons=(
                    previous.backpressure_reasons
                    if backpressure_reasons is None
                    else tuple(backpressure_reasons)
                ),
                terminal_reason=(
                    str(terminal_reason or "")
                    if requested.terminal
                    or requested is SupervisorLifecycleState.STOPPING
                    else ""
                ),
                transition_id=(
                    _content_id(
                        {
                            "target_id": target,
                            "from": previous.state.value,
                            "to": requested.value,
                            "reason": str(terminal_reason or "heartbeat"),
                            "at": now_ms,
                        }
                    )
                    if requested is not previous.state
                    else previous.transition_id
                ),
                updated_at_ms=now_ms,
            )
            self._statuses[target] = status
            if requested is not previous.state:
                self._append_event_locked(
                    target_id=target,
                    action="heartbeat",
                    accepted=True,
                    previous_state=previous.state,
                    state=requested,
                    reason=status.terminal_reason or "runtime heartbeat",
                    request_id="",
                    occurred_at_ms=now_ms,
                    changed=True,
                    fencing_epoch=status.fencing_epoch,
                )
            return status

    def record_decision(
        self,
        request: OperationRequest,
        *,
        now_ms: int,
        accepted: bool,
        reason: str,
        replayed: bool = False,
    ) -> LifecycleEvent:
        """Record a service-level denial or exact replay without changing state."""

        target_id = str(request.parameters.get("target_id") or "supervisor").strip()
        with self._lock:
            current = self._statuses.get(target_id) or self._default(
                target_id, now_ms
            )
            self._statuses.setdefault(target_id, current)
            # Invalid transition is already recorded by ``transition``.
            if (
                not accepted
                and self._events
                and self._events[-1].request_id == request.request_id
                and self._events[-1].action == request.operation.value
                and not self._events[-1].accepted
            ):
                return self._events[-1]
            return self._append_event_locked(
                target_id=target_id,
                action=request.operation.value,
                accepted=accepted,
                previous_state=current.state,
                state=current.state,
                reason=reason,
                request_id=request.request_id,
                occurred_at_ms=now_ms,
                changed=False,
                replayed=replayed,
                fencing_epoch=request.fencing_epoch,
            )

    def events(
        self,
        *,
        target_id: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
        offset: int = 0,
        after_sequence: int = 0,
    ) -> tuple[LifecycleEvent, ...]:
        if limit < 1 or limit > self._max_events:
            raise ControlBoundsError("lifecycle event limit exceeds store bound")
        if offset < 0 or after_sequence < 0:
            raise ControlBoundsError("event offset/cursor must be non-negative")
        with self._lock:
            items = [
                event
                for event in self._events
                if event.sequence > after_sequence
                and (not target_id or event.target_id == target_id)
            ]
            return tuple(items[offset : offset + limit])


class JsonLifecycleStore(InMemoryLifecycleStore):
    """Crash-safe lifecycle state with a bounded JSONL event replay window."""

    def __init__(
        self,
        state_root: Union[str, Path],
        *,
        filename: str = "supervisor-lifecycle.json",
        events_filename: str = "supervisor-lifecycle-events.jsonl",
        max_events: int = DEFAULT_MAX_CONTROL_EVENTS,
    ) -> None:
        self._state_root = _normalized_absolute(state_root, label="state_root")
        for value, label in (
            (filename, "lifecycle filename"),
            (events_filename, "lifecycle events filename"),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or path == Path("."):
                raise ValueError(f"{label} must be a contained relative path")
        self._state_path = self._state_root / filename
        self._events_path = self._state_root / events_filename
        self._lock_path = self._state_root / ".supervisor-lifecycle.lock"
        super().__init__(max_events=max_events)
        self._state_root.mkdir(parents=True, exist_ok=True)
        with self._file_guard():
            self._load_locked()

    @contextmanager
    def _file_guard(self) -> Iterator[None]:
        self._state_root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _load_locked(self) -> None:
        statuses: dict[str, LifecycleStatus] = {}
        state_corrupt = False
        if self._state_path.exists():
            try:
                payload = json.loads(self._state_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("lifecycle state root must be an object")
                raw_statuses = payload.get("statuses")
                if not isinstance(raw_statuses, Mapping):
                    raise ValueError("lifecycle statuses must be an object")
                if isinstance(raw_statuses, Mapping):
                    for target_id, item in raw_statuses.items():
                        if isinstance(item, Mapping):
                            status = LifecycleStatus.from_dict(item)
                            if status.target_id == target_id:
                                statuses[target_id] = status
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # An interrupted state write is recoverable from the last
                # valid event prefix; never trust a partially decoded state.
                statuses = {}
                state_corrupt = True
        events: deque[LifecycleEvent] = deque(maxlen=self._max_events)
        sequence = 0
        if self._events_path.exists():
            try:
                with self._events_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                            if not isinstance(item, Mapping):
                                continue
                            event = LifecycleEvent.from_dict(item)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            continue
                        events.append(event)
                        sequence = max(sequence, event.sequence)
            except OSError:
                events.clear()
        if state_corrupt:
            targets = sorted({event.target_id for event in events if event.target_id})
            for target_id in targets or ["supervisor"]:
                latest = next(
                    (
                        event
                        for event in reversed(events)
                        if event.target_id == target_id
                    ),
                    None,
                )
                occurred_at_ms = (
                    latest.occurred_at_ms if latest is not None else _now_ms()
                )
                statuses[target_id] = LifecycleStatus(
                    target_id=target_id,
                    state=SupervisorLifecycleState.FAILED,
                    phase="state_recovery",
                    heartbeat_at_ms=occurred_at_ms,
                    terminal_reason="lifecycle_state_corrupt",
                    generation=0,
                    fencing_epoch=(
                        latest.fencing_epoch if latest is not None else None
                    ),
                    updated_at_ms=occurred_at_ms,
                )
        self._statuses = statuses
        self._events = events
        self._sequence = sequence

    def _persist_locked(self) -> None:
        payload = {
            "schema": LIFECYCLE_STATUS_SCHEMA,
            "statuses": {
                target: status.to_dict()
                for target, status in sorted(self._statuses.items())
            },
        }
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        encoded_state = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(encoded_state)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._state_path)
        events_temporary = self._events_path.with_suffix(
            self._events_path.suffix + ".tmp"
        )
        with events_temporary.open("w", encoding="utf-8") as stream:
            for event in self._events:
                stream.write(
                    json.dumps(
                        event.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(events_temporary, self._events_path)

    def seed(self, status: LifecycleStatus) -> None:
        with self._file_guard(), self._lock:
            self._load_locked()
            super().seed(status)
            self._persist_locked()

    def snapshot(self, *args: Any, **kwargs: Any) -> LifecycleStatus:
        with self._file_guard(), self._lock:
            self._load_locked()
            result = super().snapshot(*args, **kwargs)
            self._persist_locked()
            return result

    def transition(
        self, request: OperationRequest, **kwargs: Any
    ) -> tuple[LifecycleStatus, LifecycleStatus, LifecycleEvent]:
        with self._file_guard(), self._lock:
            self._load_locked()
            try:
                result = super().transition(request, **kwargs)
            finally:
                self._persist_locked()
            return result

    def heartbeat(self, *args: Any, **kwargs: Any) -> LifecycleStatus:
        with self._file_guard(), self._lock:
            self._load_locked()
            result = super().heartbeat(*args, **kwargs)
            self._persist_locked()
            return result

    def events(self, **kwargs: Any) -> tuple[LifecycleEvent, ...]:
        with self._file_guard(), self._lock:
            self._load_locked()
            return super().events(**kwargs)

    def record_decision(
        self, request: OperationRequest, **kwargs: Any
    ) -> LifecycleEvent:
        with self._file_guard(), self._lock:
            self._load_locked()
            result = super().record_decision(request, **kwargs)
            self._persist_locked()
            return result


def _default_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class SupervisorLifecycleBackend:
    """Authoritative lifecycle/status backend for :class:`SupervisorControlService`."""

    def __init__(
        self,
        state_store: Union[InMemoryLifecycleStore, None] = None,
        *,
        clock_ms: Callable[[], int] = _now_ms,
        pid_alive: Callable[[int], bool] = _default_pid_alive,
        stale_after_ms: int = 60_000,
        max_events: int = DEFAULT_MAX_CONTROL_EVENTS,
    ) -> None:
        if (
            isinstance(stale_after_ms, bool)
            or not isinstance(stale_after_ms, int)
            or stale_after_ms < 1
        ):
            raise ValueError("stale_after_ms must be a positive integer")
        self.state_store = state_store or InMemoryLifecycleStore(
            max_events=max_events
        )
        self._clock_ms = clock_ms
        self._pid_alive = pid_alive
        self._stale_after_ms = stale_after_ms
        self.optional_providers_loaded = False
        self.processes_started = False

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        return tuple(
            sorted(
                {
                    Operation.STATUS,
                    Operation.HEALTH,
                    Operation.EVENTS,
                    *tuple(_ACTION_REQUESTED_STATE),
                },
                key=lambda item: item.value,
            )
        )

    @staticmethod
    def _target_id(request: OperationRequest) -> str:
        return str(request.parameters.get("target_id") or "supervisor").strip()

    def status(self, target_id: str = "supervisor") -> LifecycleStatus:
        return self.state_store.snapshot(
            target_id,
            now_ms=self._clock_ms(),
            pid_alive=self._pid_alive,
            stale_after_ms=self._stale_after_ms,
            recover=True,
        )

    def heartbeat(self, target_id: str = "supervisor", **values: Any) -> LifecycleStatus:
        return self.state_store.heartbeat(
            target_id, now_ms=self._clock_ms(), **values
        )

    def _status_is_healthy(self, status: LifecycleStatus) -> bool:
        if status.state is not SupervisorLifecycleState.HEALTHY:
            return False
        now_ms = self._clock_ms()
        if (
            status.heartbeat_at_ms <= 0
            or status.heartbeat_at_ms > now_ms
            or now_ms - status.heartbeat_at_ms >= self._stale_after_ms
        ):
            return False
        if status.pid is None:
            return False
        try:
            return bool(self._pid_alive(status.pid))
        except Exception:
            return False

    def record_rejection(
        self, request: OperationRequest, error: OperationError
    ) -> LifecycleEvent:
        return self.state_store.record_decision(
            request,
            now_ms=self._clock_ms(),
            accepted=False,
            reason=f"{error.code.value}: {error.message}",
        )

    def record_replay(self, request: OperationRequest) -> LifecycleEvent:
        return self.state_store.record_decision(
            request,
            now_ms=self._clock_ms(),
            accepted=True,
            reason="exact idempotent replay",
            replayed=True,
        )

    def execute(self, request: OperationRequest) -> BackendResponse:
        if request.operation in {Operation.STATUS, Operation.HEALTH}:
            status = self.status(self._target_id(request))
            data = status.to_dict()
            if request.operation is Operation.HEALTH:
                data["healthy"] = self._status_is_healthy(status)
            return BackendResponse(data=data)
        if request.operation is Operation.EVENTS:
            limit, offset = _bounded_window(request)
            after_sequence = int(request.parameters.get("after_sequence") or 0)
            events = self.state_store.events(
                target_id=self._target_id(request)
                if request.parameters.get("target_id")
                else "",
                limit=min(limit, self.state_store.max_events),
                offset=offset,
                after_sequence=after_sequence,
            )
            return BackendResponse(
                data={
                    "items": [item.to_dict() for item in events],
                    "count": len(events),
                    "limit": limit,
                    "offset": offset,
                    "after_sequence": after_sequence,
                    "truncated": len(events) == limit,
                }
            )
        if request.operation not in _ACTION_REQUESTED_STATE:
            raise OperationUnavailableError(
                f"operation {request.operation.value} is not implemented"
            )
        previous, status, event = self.state_store.transition(
            request,
            now_ms=self._clock_ms(),
            pid_alive=self._pid_alive,
            stale_after_ms=self._stale_after_ms,
        )
        return BackendResponse(
            data={
                "status": status.to_dict(),
                "event": event.to_dict(),
                "previous_state": event.previous_state.value,
                "state": status.state.value,
                "accepted": True,
                "idempotent": not event.changed,
            },
            changed=event.changed,
            applied_effect_ids=(
                tuple(item.effect_id for item in request.expected_effects)
                if event.changed
                else ()
            ),
        )


OperationHandler = Callable[[OperationRequest], Union[BackendResponse, Mapping[str, Any], Any]]


class WorkflowPreviewHandler(Protocol):
    """Side-effect-free adapter for one bound workflow proposal."""

    def __call__(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        ...


class WorkflowMaterializeHandler(Protocol):
    """Authorized adapter invoked inside the shared mutation transaction."""

    def __call__(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        ...


class RestartHandler(Protocol):
    """Fenced lifecycle-restart adapter; implementations are registered lazily."""

    def __call__(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        ...


class RescuePreviewHandler(Protocol):
    """Side-effect-free adapter for current incident-bound rescue proposals."""

    def __call__(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        ...


class RescueHandler(Protocol):
    """Authorized incident-bound rescue adapter for one bounded plan."""

    def __call__(
        self, request: OperationRequest
    ) -> Union[BackendResponse, Mapping[str, Any], Any]:
        ...


class LeaseFenceValidator(Protocol):
    """Checks authoritative current lease state before mutation dispatch."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


class AuthorizationValidator(Protocol):
    """Optional live policy check in addition to the bound contract decision."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


class TargetIdentityValidator(Protocol):
    """Checks repository and tree identities against authoritative state."""

    def validate(self, request: OperationRequest) -> Union[bool, None]:
        ...


@dataclass(frozen=True)
class ControlAuditReceipt:
    """Content-addressed audit record for one service decision."""

    request_id: str
    operation: str
    authority: str
    status: str
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    caller: str
    authorization_decision_id: str
    grant_ids: tuple[str, ...]
    dry_run: bool
    idempotency_key: str
    lease_id: str
    fencing_epoch: Union[int, None]
    effect_ids: tuple[str, ...]
    applied_effect_ids: tuple[str, ...]
    error_code: str
    occurred_at_ms: int
    occurred_at: str
    receipt_id: str = ""

    def __post_init__(self) -> None:
        payload = self._payload()
        expected = _content_id(payload)
        if self.receipt_id and self.receipt_id != expected:
            raise ValueError("audit receipt identity does not match its payload")
        object.__setattr__(self, "receipt_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_AUDIT_RECEIPT_SCHEMA,
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "authority": self.authority,
            "status": self.status,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "caller": self.caller,
            "authorization_decision_id": self.authorization_decision_id,
            "grant_ids": self.grant_ids,
            "dry_run": self.dry_run,
            "idempotency_key": self.idempotency_key,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "effect_ids": self.effect_ids,
            "applied_effect_ids": self.applied_effect_ids,
            "error_code": self.error_code,
            "occurred_at_ms": self.occurred_at_ms,
            "occurred_at": self.occurred_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}


class ControlStateStore(Protocol):
    """Persistence boundary for transactions, replay results, and audit."""

    def transaction(self, request: OperationRequest) -> Any:
        ...

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        ...

    def begin_mutation(
        self, request: OperationRequest, *, now_ms: int
    ) -> MutationTransactionState:
        ...

    def get_mutation(
        self, request: OperationRequest
    ) -> Union[MutationTransactionState, None]:
        ...

    def compare_and_swap_mutation(
        self,
        request: OperationRequest,
        *,
        expected_revision: int,
        phase: MutationTransactionPhase,
        now_ms: int,
        applied_effect_ids: Iterable[str] = (),
        failure_code: str = "",
        result: Union[OperationResult, None] = None,
    ) -> MutationTransactionState:
        ...

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        ...

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        ...

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        ...


class InMemoryControlStateStore:
    """Thread-safe state store suitable for embedding and deterministic tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._idempotency: dict[str, tuple[str, OperationResult]] = {}
        self._mutations: dict[str, MutationTransactionState] = {}
        self._receipts: list[dict[str, Any]] = []

    @contextmanager
    def transaction(self, request: OperationRequest) -> Iterator[None]:
        """Serialize a mutation decision through dispatch and persistence."""

        del request
        with self._lock:
            yield

    @staticmethod
    def _key(request: OperationRequest) -> str:
        return "\x1f".join(
            (
                request.state_root,
                request.repository_id,
                request.objective_id,
                request.caller,
                request.operation.value,
                request.idempotency_key,
            )
        )

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        with self._lock:
            return self._idempotency.get(self._key(request))

    def begin_mutation(
        self, request: OperationRequest, *, now_ms: int
    ) -> MutationTransactionState:
        key = self._key(request)
        prepared = MutationTransactionState.prepare(request, now_ms=now_ms)
        with self._lock:
            existing = self._mutations.get(key)
            if existing is not None:
                if existing.request_id != request.request_id:
                    raise IdempotencyConflictError(
                        "idempotency key is already bound to changed effects "
                        "or another request binding"
                    )
                return existing
            self._mutations[key] = prepared
            return prepared

    def get_mutation(
        self, request: OperationRequest
    ) -> Union[MutationTransactionState, None]:
        with self._lock:
            return self._mutations.get(self._key(request))

    def compare_and_swap_mutation(
        self,
        request: OperationRequest,
        *,
        expected_revision: int,
        phase: MutationTransactionPhase,
        now_ms: int,
        applied_effect_ids: Iterable[str] = (),
        failure_code: str = "",
        result: Union[OperationResult, None] = None,
    ) -> MutationTransactionState:
        key = self._key(request)
        phase = MutationTransactionPhase(str(getattr(phase, "value", phase)))
        with self._lock:
            current = self._mutations.get(key)
            if current is None:
                raise TransactionConflictError("mutation transaction is absent")
            if current.request_id != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is bound to another request"
                )
            if current.revision != expected_revision:
                raise TransactionConflictError(
                    f"stale transaction revision {expected_revision}; "
                    f"current revision is {current.revision}"
                )
            if phase not in _LEGAL_MUTATION_TRANSACTION_TRANSITIONS[current.phase]:
                raise TransactionConflictError(
                    f"illegal mutation transaction transition "
                    f"{current.phase.value}->{phase.value}"
                )
            recovery_action = MutationRecoveryAction.NONE
            if phase is MutationTransactionPhase.COMPENSATION_REQUIRED:
                recovery_action = MutationRecoveryAction.COMPENSATE
            elif phase is MutationTransactionPhase.REPAIR_REQUIRED:
                recovery_action = MutationRecoveryAction.REPAIR
            updated = replace(
                current,
                phase=phase,
                revision=current.revision + 1,
                applied_effect_ids=tuple(applied_effect_ids),
                recovery_action=recovery_action,
                failure_code=failure_code,
                result=result,
                updated_at_ms=now_ms,
            )
            self._mutations[key] = updated
            return updated

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        key = self._key(request)
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is not None and existing[0] != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to another request"
                )
            self._idempotency[key] = (request.request_id, result)

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        del request
        with self._lock:
            self._receipts.append(receipt.to_dict())

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        del request
        with self._lock:
            newest = list(reversed(self._receipts))
            return newest[offset : offset + limit]


class JsonlControlStateStore(InMemoryControlStateStore):
    """Durable JSONL audit and exact-result idempotency store.

    Records retain canonical :class:`OperationResult` payloads, so a restarted
    service can return the exact prior result without invoking a mutating
    backend again.  Multi-writer deployments may replace this with a database
    implementation of :class:`ControlStateStore` for transactional reservation
    across processes.
    """

    def __init__(
        self,
        filename: str = "control-audit.jsonl",
        idempotency_filename: str = "control-idempotency.jsonl",
        transaction_filename: str = "control-transactions.jsonl",
    ) -> None:
        super().__init__()
        for value, label in (
            (filename, "control audit filename"),
            (idempotency_filename, "control idempotency filename"),
            (transaction_filename, "control transaction filename"),
        ):
            if (
                not str(value).strip()
                or Path(value) == Path(".")
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(f"{label} must be relative")
        self._filename = filename
        self._idempotency_filename = idempotency_filename
        self._transaction_filename = transaction_filename

    @staticmethod
    def _idempotency_record_key(request: OperationRequest) -> dict[str, str]:
        return {
            "repository_id": request.repository_id,
            "objective_id": request.objective_id,
            "caller": request.caller,
            "operation": request.operation.value,
            "idempotency_key": request.idempotency_key,
        }

    @contextmanager
    def transaction(self, request: OperationRequest) -> Iterator[None]:
        lock_path = Path(request.state_root) / ".control-transaction.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                with self._lock:
                    yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def get_idempotent(
        self, request: OperationRequest
    ) -> Union[tuple[str, OperationResult], None]:
        cached = super().get_idempotent(request)
        if cached is not None or not request.idempotency_key:
            return cached
        path = Path(request.state_root) / self._idempotency_filename
        if not path.exists():
            return None
        expected = self._idempotency_record_key(request)
        matching: Union[Mapping[str, Any], None] = None
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(record, Mapping) and not any(
                            record.get(name) != value
                            for name, value in expected.items()
                        ):
                            matching = record
            except OSError as exc:
                raise IdempotencyConflictError(
                    "idempotency state is unreadable"
                ) from exc
        if matching is not None:
            raw_result = matching.get("result")
            try:
                if not isinstance(raw_result, Mapping):
                    raise ValueError("stored result is absent")
                result = OperationResult.from_dict(raw_result)
                request_id = str(matching.get("request_id") or "")
            except Exception as exc:
                raise IdempotencyConflictError(
                    "idempotency state contains an invalid matching result"
                ) from exc
            # Populate the fast path only after canonical decoding succeeds.
            with self._lock:
                self._idempotency[self._key(request)] = (request_id, result)
            return request_id, result
        return None

    def _append_mutation(
        self,
        request: OperationRequest,
        state: MutationTransactionState,
    ) -> None:
        path = Path(request.state_root) / self._transaction_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            state.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def get_mutation(
        self, request: OperationRequest
    ) -> Union[MutationTransactionState, None]:
        cached = super().get_mutation(request)
        if cached is not None or not request.idempotency_key:
            return cached
        path = Path(request.state_root) / self._transaction_filename
        if not path.exists():
            return None
        expected = self._idempotency_record_key(request)
        latest: Union[MutationTransactionState, None] = None
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(raw, Mapping) or any(
                        raw.get(name) != value for name, value in expected.items()
                    ):
                        continue
                    try:
                        candidate = MutationTransactionState.from_dict(raw)
                    except (TypeError, ValueError, ControlContractError) as exc:
                        raise TransactionConflictError(
                            "matching mutation transaction state is invalid"
                        ) from exc
                    if latest is None or candidate.revision > latest.revision:
                        latest = candidate
                    elif candidate.revision == latest.revision and candidate != latest:
                        raise TransactionConflictError(
                            "mutation transaction contains divergent revisions"
                        )
        except OSError as exc:
            raise TransactionConflictError(
                "mutation transaction state is unreadable"
            ) from exc
        if latest is not None:
            with self._lock:
                self._mutations[self._key(request)] = latest
        return latest

    def begin_mutation(
        self, request: OperationRequest, *, now_ms: int
    ) -> MutationTransactionState:
        existing = self.get_mutation(request)
        if existing is not None:
            if existing.request_id != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to changed effects "
                    "or another request binding"
                )
            return existing
        state = super().begin_mutation(request, now_ms=now_ms)
        self._append_mutation(request, state)
        return state

    def compare_and_swap_mutation(
        self,
        request: OperationRequest,
        *,
        expected_revision: int,
        phase: MutationTransactionPhase,
        now_ms: int,
        applied_effect_ids: Iterable[str] = (),
        failure_code: str = "",
        result: Union[OperationResult, None] = None,
    ) -> MutationTransactionState:
        # Loading before the in-memory CAS makes restart recovery use the same
        # revision semantics as a continuously running process.
        self.get_mutation(request)
        state = super().compare_and_swap_mutation(
            request,
            expected_revision=expected_revision,
            phase=phase,
            now_ms=now_ms,
            applied_effect_ids=applied_effect_ids,
            failure_code=failure_code,
            result=result,
        )
        self._append_mutation(request, state)
        return state

    def put_idempotent(
        self, request: OperationRequest, result: OperationResult
    ) -> None:
        existing = self.get_idempotent(request)
        if existing is not None:
            if existing[0] != request.request_id:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to another request"
                )
            return
        path = Path(request.state_root) / self._idempotency_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "ipfs_accelerate_py/agent-supervisor/control-idempotency@1",
            **self._idempotency_record_key(request),
            "request_id": request.request_id,
            "result": result.to_record(),
        }
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        super().put_idempotent(request, result)

    def append_receipt(
        self, request: OperationRequest, receipt: ControlAuditReceipt
    ) -> None:
        super().append_receipt(request, receipt)
        path = Path(request.state_root) / self._filename
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            receipt.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def query_receipts(
        self, request: OperationRequest, *, limit: int, offset: int
    ) -> Sequence[Mapping[str, Any]]:
        path = Path(request.state_root) / self._filename
        if not path.exists():
            return super().query_receipts(
                request, limit=limit, offset=offset
            )
        from collections import deque

        records: Any = deque(maxlen=offset + limit)
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, Mapping):
                            records.append(item)
            except OSError:
                return ()
        newest = list(reversed(records))
        return newest[offset : offset + limit]


class RepositorySupervisorBackend:
    """Direct package-API backend for bounded supervisor inspection.

    Paths are always explicit repository- or state-relative parameters.  This
    avoids unsafe discovery defaults and lets the service validate every path
    against the selected allowlisted root before any package API is called.
    Mutations and domain-specific proposal operations are supplied as Python
    handlers by the embedding runtime.
    """

    def __init__(
        self,
        handlers: Union[Mapping[Union[Operation, str], OperationHandler], None] = None,
    ) -> None:
        normalized: dict[Operation, OperationHandler] = {}
        for name, handler in dict(handlers or {}).items():
            operation = name if isinstance(name, Operation) else Operation(str(name))
            if not callable(handler):
                raise TypeError(f"handler for {operation.value} must be callable")
            normalized[operation] = handler
        self._handlers = MappingProxyType(normalized)
        # Registration is inert: providers and processes remain the
        # responsibility of a handler at explicit execution time.
        self.optional_providers_loaded = False
        self.processes_started = False

    @property
    def registered_operations(self) -> tuple[Operation, ...]:
        builtins = set(READ_OPERATIONS)
        return tuple(sorted(builtins | set(self._handlers), key=lambda item: item.value))

    def _resolve(
        self,
        request: OperationRequest,
        relative: str,
        *,
        state: bool,
    ) -> Path:
        root = Path(request.state_root if state else request.repository_root).resolve()
        resolved = (root / relative).resolve(strict=False)
        if not _under(resolved, root):
            raise PathEscapeError("requested path escapes its selected root")
        return resolved

    @staticmethod
    def _read_text(path: Path, *, maximum_bytes: int) -> str:
        if not path.exists():
            raise BackendNotFoundError(f"control data not found: {path.name}")
        if not path.is_file():
            raise ValueError(f"control data path is not a file: {path.name}")
        if path.stat().st_size > maximum_bytes:
            raise ControlBoundsError(
                f"control data exceeds the {maximum_bytes}-byte request bound"
            )
        return path.read_text(encoding="utf-8")

    @classmethod
    def _read_json(cls, path: Path, *, maximum_bytes: int) -> Any:
        return json.loads(
            cls._read_text(path, maximum_bytes=maximum_bytes)
        )

    @staticmethod
    def _window(items: Sequence[Any], request: OperationRequest) -> dict[str, Any]:
        limit, offset = _bounded_window(request)
        selected = list(items[offset : offset + limit])
        return {
            "items": selected,
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(selected) < len(items),
        }

    def _json_document(
        self,
        request: OperationRequest,
        names: Sequence[str],
        *,
        state: bool = True,
    ) -> Mapping[str, Any]:
        relative = _relative_parameter(request, *names)
        value = self._read_json(
            self._resolve(request, relative, state=state),
            maximum_bytes=request.bounds.max_serialized_bytes,
        )
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return self._window(value, request)
        return {"value": value}

    def _goals(self, request: OperationRequest) -> Mapping[str, Any]:
        from ..objectives.objective_graph import parse_goal_heap

        relative = _relative_parameter(request, "objective_path", "path")
        path = self._resolve(request, relative, state=False)
        if not path.exists():
            raise BackendNotFoundError(f"objective heap not found: {relative}")
        text = self._read_text(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        goals = [
            {
                "goal_id": item.goal_id,
                "title": item.title,
                "status": item.status,
                "fields": dict(item.fields),
            }
            for item in parse_goal_heap(text)
        ]
        return self._window(goals, request)

    def _tasks(self, request: OperationRequest) -> Mapping[str, Any]:
        from ..task_sources.todo_vector_index import parse_todo_blocks

        relative = _relative_parameter(request, "todo_path", "path")
        path = self._resolve(request, relative, state=False)
        if not path.exists():
            raise BackendNotFoundError(f"task board not found: {relative}")
        prefix = request.parameters.get("task_header_prefix", "")
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("task_header_prefix must be a non-empty string")
        text = self._read_text(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        tasks = [
            {
                "task_id": task_id,
                "title": title,
                "source_line": source_line,
                "fields": fields,
                "status": str(fields.get("status") or "todo").strip().lower(),
            }
            for task_id, title, source_line, fields in parse_todo_blocks(
                text,
                task_header_prefix=prefix,
            )
        ]
        return self._window(tasks, request)

    def _events(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(request, "events_path", "path")
        path = self._resolve(request, relative, state=True)
        return self._jsonl_window(
            path,
            request,
            newest_first=bool(request.parameters.get("newest_first", False)),
        )

    @staticmethod
    def _jsonl_window(
        path: Path,
        request: OperationRequest,
        *,
        newest_first: bool = False,
    ) -> Mapping[str, Any]:
        limit, offset = _bounded_window(request)
        if not path.exists():
            return {
                "items": [],
                "count": 0,
                "offset": offset,
                "limit": limit,
                "truncated": False,
            }
        if not path.is_file():
            raise ValueError(f"JSONL path is not a file: {path.name}")
        needed = offset + limit + 1
        if newest_first:
            from collections import deque

            retained: Any = deque(maxlen=needed)
        else:
            retained = []
        valid_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                if len(raw_line.encode("utf-8")) > request.bounds.max_text_bytes:
                    raise ControlBoundsError(
                        "JSONL record exceeds the request text bound"
                    )
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, Mapping):
                    continue
                if newest_first:
                    retained.append(dict(value))
                elif valid_count >= offset:
                    retained.append(dict(value))
                    if len(retained) >= limit + 1:
                        break
                valid_count += 1
        if newest_first:
            values = list(reversed(retained))
            selected = values[offset : offset + limit]
            truncated = len(values) > offset + limit
        else:
            selected = retained[:limit]
            truncated = len(retained) > limit
        return {
            "items": selected,
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
        }

    def _receipts(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(
            request, "receipts_path", "path", required=False
        )
        if not relative:
            raise OperationUnavailableError(
                "receipt reads are served by the control state store"
            )
        path = self._resolve(request, relative, state=True)
        if path.is_dir():
            limit, offset = _bounded_window(request)
            candidates = heapq.nsmallest(
                offset + limit + 1,
                (
                    item
                    for item in path.rglob("*.json")
                    if item.is_file() and _under(item.resolve(), path.resolve())
                ),
                key=lambda item: item.as_posix(),
            )
            selected = candidates[offset : offset + limit]
            records = []
            for item in selected:
                try:
                    value = self._read_json(
                        item,
                        maximum_bytes=request.bounds.max_serialized_bytes,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                records.append(
                    {
                        "path": item.relative_to(path).as_posix(),
                        "receipt": value,
                    }
                )
            return {
                "items": records,
                "count": len(records),
                "offset": offset,
                "limit": limit,
                "truncated": offset + len(selected) < len(candidates),
            }
        if path.suffix == ".jsonl":
            return self._jsonl_window(path, request)
        value = self._read_json(
            path, maximum_bytes=request.bounds.max_serialized_bytes
        )
        values = value if isinstance(value, list) else [value]
        return self._window(values, request)

    def _cache(self, request: OperationRequest) -> Mapping[str, Any]:
        relative = _relative_parameter(request, "cache_path", "path")
        path = self._resolve(request, relative, state=True)
        if not path.exists():
            raise BackendNotFoundError(f"cache path not found: {relative}")
        limit, offset = _bounded_window(request)
        if path.is_file():
            value = self._read_json(
                path, maximum_bytes=request.bounds.max_serialized_bytes
            )
            return {"path": relative, "kind": "file", "value": value}
        entries = heapq.nsmallest(
            offset + limit + 1,
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
        selected = entries[offset : offset + limit]
        return {
            "path": relative,
            "kind": "directory",
            "entries": [
                {
                    "path": item.relative_to(path).as_posix(),
                    "size_bytes": item.stat().st_size,
                }
                for item in selected
            ],
            "count": len(selected),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(selected) < len(entries),
        }

    def _artifact(self, request: OperationRequest) -> Mapping[str, Any]:
        from ..runtime.artifact_store import query_artifact

        relative = _relative_parameter(request, "artifact_path", "path")
        root_name = str(request.parameters.get("root") or "state")
        if root_name not in {"repository", "state"}:
            raise ValueError("artifact root must be 'repository' or 'state'")
        state = root_name == "state"
        path = self._resolve(request, relative, state=state)
        limit, _offset = _bounded_window(request)
        columns = request.parameters.get("columns", ("*",))
        if isinstance(columns, str):
            columns = (columns,)
        if not isinstance(columns, Sequence):
            raise ValueError("columns must be a sequence")
        if len(columns) > request.bounds.max_items:
            raise ControlBoundsError("columns exceed the request item bound")
        sql = str(request.parameters.get("sql") or "").strip()
        if sql:
            raise ValueError(
                "raw SQL is disabled at the supervisor control boundary"
            )
        where = str(request.parameters.get("where") or "").strip()
        folded_where = where.lower()
        if (
            ";" in where
            or "--" in where
            or "/*" in where
            or re.search(
                r"\b(select|from|join|attach|copy|pragma|install|load|read_\w+)\b",
                folded_where,
            )
        ):
            raise ValueError(
                "where must be a simple expression without subqueries or I/O"
            )
        return query_artifact(
            path,
            table=str(request.parameters.get("table") or "") or None,
            columns=tuple(str(item) for item in columns),
            where=where,
            sql="",
            limit=limit,
            kind=str(request.parameters.get("kind") or "") or None,
        )

    def execute(self, request: OperationRequest) -> Union[BackendResponse, Mapping[str, Any], Any]:
        handler = self._handlers.get(request.operation)
        if handler is not None:
            return handler(request)
        if request.operation in {
            Operation.STATUS,
            Operation.HEALTH,
            Operation.METRICS,
            Operation.BUNDLES,
            Operation.LANES,
        }:
            names = {
                Operation.STATUS: ("status_path", "path"),
                Operation.HEALTH: ("health_path", "path"),
                Operation.METRICS: ("metrics_path", "path"),
                Operation.BUNDLES: ("bundle_index_path", "path"),
                Operation.LANES: ("lane_manifest_path", "path"),
            }[request.operation]
            return self._json_document(request, names)
        if request.operation is Operation.GOALS:
            return self._goals(request)
        if request.operation is Operation.TASKS:
            return self._tasks(request)
        if request.operation is Operation.EVENTS:
            return self._events(request)
        if request.operation is Operation.RECEIPTS:
            return self._receipts(request)
        if request.operation is Operation.CACHE_INSPECT:
            return self._cache(request)
        if request.operation is Operation.ARTIFACT_QUERY:
            return self._artifact(request)
        raise OperationUnavailableError(
            f"operation {request.operation.value} has no direct Python adapter"
        )


@dataclass(frozen=True)
class SupervisorTarget:
    """Binding used by :class:`SupervisorClient` to construct read requests."""

    repository_root: str
    state_root: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str

    def request(
        self,
        operation: Union[Operation, str],
        *,
        parameters: Union[Mapping[str, Any], None] = None,
        bounds: Union[ControlBounds, None] = None,
        dry_run: bool = False,
        expected_effects: Sequence[ExpectedEffect] = (),
    ) -> OperationRequest:
        selected = operation if isinstance(operation, Operation) else Operation(operation)
        if selected in MUTATION_OPERATIONS and not dry_run:
            raise AuthorizationBindingError(
                "SupervisorTarget constructs only read, proposal, or dry-run requests"
            )
        return OperationRequest(
            operation=selected,
            repository_root=self.repository_root,
            state_root=self.state_root,
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            objective_id=self.objective_id,
            objective_revision=self.objective_revision,
            policy_id=self.policy_id,
            policy_revision=self.policy_revision,
            caller=self.caller,
            parameters=dict(parameters or {}),
            bounds=bounds or ControlBounds(),
            dry_run=dry_run,
            expected_effects=tuple(expected_effects),
        )


class SupervisorControlService:
    """Typed policy and dispatch boundary shared by Python, CLI, and MCP."""

    def __init__(
        self,
        *,
        repository_allowlist: Union[Iterable[Union[str, Path]], None] = None,
        state_allowlist: Union[Iterable[Union[str, Path]], None] = None,
        allowed_repository_roots: Union[Iterable[Union[str, Path]], None] = None,
        allowed_state_roots: Union[Iterable[Union[str, Path]], None] = None,
        backend: Union[RepositorySupervisorBackend, Any, None] = None,
        handlers: Union[Mapping[Union[Operation, str], OperationHandler], None] = None,
        lease_validator: Union[LeaseFenceValidator, Callable[[OperationRequest], Any], None] = None,
        authorization_validator: Union[
            AuthorizationValidator, Callable[[OperationRequest], Any], None
        ] = None,
        identity_validator: Union[
            TargetIdentityValidator, Callable[[OperationRequest], Any], None
        ] = None,
        state_store: Union[ControlStateStore, None] = None,
        catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
        service_id: str = "ipfs-accelerate-agent-supervisor",
        service_version: str = CONTROL_SERVICE_VERSION,
        max_query_items: int = DEFAULT_MAX_QUERY_ITEMS,
        require_lease_validator: bool = True,
        decision_runtime: Any = None,
        decision_runtime_cancellation: Any = None,
        clock_ms: Callable[[], int] = _now_ms,
    ) -> None:
        repositories = (
            repository_allowlist
            if repository_allowlist is not None
            else allowed_repository_roots
        )
        states = (
            state_allowlist if state_allowlist is not None else allowed_state_roots
        )
        if repositories is None or states is None:
            raise ValueError(
                "explicit repository and state root allowlists are required"
            )
        self._repository_roots = _normalize_allowlist(
            repositories, label="repository allowlist"
        )
        self._state_roots = _normalize_allowlist(states, label="state allowlist")
        if backend is not None and handlers:
            raise ValueError("supply backend or handlers, not both")
        self._backend = backend or RepositorySupervisorBackend(handlers)
        self._lease_validator = lease_validator
        self._authorization_validator = authorization_validator
        self._identity_validator = identity_validator
        self._state_store = state_store or JsonlControlStateStore()
        self._catalog = _validate_canonical_catalog(catalog)
        self._service_id = str(service_id).strip()
        self._service_version = str(service_version).strip()
        if not self._service_id or not self._service_version:
            raise ValueError("service_id and service_version must not be empty")
        if (
            isinstance(max_query_items, bool)
            or not isinstance(max_query_items, int)
            or max_query_items < 1
        ):
            raise ValueError("max_query_items must be a positive integer")
        self._max_query_items = min(max_query_items, DEFAULT_MAX_QUERY_ITEMS)
        self._require_lease_validator = bool(require_lease_validator)
        self._decision_runtime = decision_runtime
        self._decision_runtime_cancellation = decision_runtime_cancellation
        self._pending_runtime_decision: Any = None
        self._clock_ms = clock_ms
        self._lock = threading.RLock()
        self._mutation_dispatch_count = 0
        self._mutation_audit_receipt_count = 0
        self._last_mutation_dispatch_request_id = ""
        self._last_mutation_audit_receipt_id = ""
        registered = getattr(self._backend, "registered_operations", None)
        if registered is None:
            self._registered_operations = frozenset(Operation)
        else:
            try:
                self._registered_operations = frozenset(
                    item
                    if isinstance(item, Operation)
                    else Operation(str(item))
                    for item in registered
                )
            except (TypeError, ValueError) as exc:
                raise ControlCatalogConformanceError(
                    "backend registered_operations contains an unknown operation"
                ) from exc
        self._registered_operations = frozenset(
            {
                *self._registered_operations,
                Operation.CAPABILITIES,
                Operation.RECEIPTS,
            }
        )
        self._capability_report = self._build_capability_report()

    @property
    def repository_allowlist(self) -> tuple[str, ...]:
        return tuple(item.as_posix() for item in self._repository_roots)

    @property
    def state_allowlist(self) -> tuple[str, ...]:
        return tuple(item.as_posix() for item in self._state_roots)

    @property
    def catalog(self) -> OperationCatalog:
        return self._catalog

    def operation_catalog(self) -> OperationCatalog:
        """Return the already validated immutable catalog without discovery."""

        return self._catalog

    def _build_capability_report(self) -> CapabilityReport:
        bounds = ControlBounds(
            max_items=self._max_query_items,
            max_paths=min(128, self._max_query_items),
            max_effects=min(64, self._max_query_items),
        )
        capabilities = tuple(
            OperationCapability(
                operation=operation,
                authority=operation.authority,
                bounds=bounds,
                supports_dry_run=operation in MUTATION_OPERATIONS,
                requires_idempotency=operation in MUTATION_OPERATIONS,
                requires_authorization=operation in MUTATION_OPERATIONS,
            )
            for operation in sorted(
                self._registered_operations, key=lambda item: item.value
            )
        )
        return CapabilityReport(
            service_id=self._service_id,
            service_version=self._service_version,
            capabilities=capabilities,
            optional_providers_loaded=bool(
                getattr(self._backend, "optional_providers_loaded", False)
            ),
            processes_started=bool(
                getattr(self._backend, "processes_started", False)
            ),
        )

    def capability_report(self) -> CapabilityReport:
        """Return the side-effect-free capability handshake."""

        return self._capability_report

    def mutation_runtime_state(self) -> ControlMutationRuntimeState:
        """Return the observed dispatch/audit population for mutation proof.

        The snapshot is maintained by the service itself, inside the same
        lock as authorization, idempotency lookup, dispatch, and audit
        persistence.  It therefore distinguishes a backend invocation from
        an exact replay without trusting a backend-supplied counter.
        """

        with self._lock:
            return ControlMutationRuntimeState(
                dispatch_count=self._mutation_dispatch_count,
                audit_receipt_count=self._mutation_audit_receipt_count,
                last_dispatch_request_id=(
                    self._last_mutation_dispatch_request_id
                ),
                last_audit_receipt_id=(
                    self._last_mutation_audit_receipt_id
                ),
            )

    def mutation_transaction(
        self, request: OperationRequest
    ) -> Union[MutationTransactionState, None]:
        """Return the durable CAS state for one exactly bound mutation."""

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            raise ValueError("transaction status is available for real mutations")
        self._check_target(request)
        transaction = getattr(self._state_store, "transaction", None)

        @contextmanager
        def no_transaction() -> Iterator[None]:
            yield

        guard = transaction(request) if callable(transaction) else no_transaction()
        with self._lock, guard:
            return self._state_store.get_mutation(request)

    def recover_mutation(
        self,
        request: OperationRequest,
        *,
        expected_revision: int,
        action: Union[MutationRecoveryAction, str],
    ) -> MutationTransactionState:
        """Run a typed compensation or repair under the original exact permit.

        Recovery is itself a mutation.  The current target, authorization,
        lease, and fence are therefore revalidated immediately before the
        backend hook is called.  ``expected_revision`` prevents two operators
        from compensating or repairing the same partial result.
        """

        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        selected = MutationRecoveryAction(
            str(getattr(action, "value", action))
        )
        if selected is MutationRecoveryAction.NONE:
            raise ValueError("recovery action must be compensate or repair")
        transaction = getattr(self._state_store, "transaction", None)

        @contextmanager
        def no_transaction() -> Iterator[None]:
            yield

        guard = transaction(request) if callable(transaction) else no_transaction()
        with self._lock, guard:
            self._check_target(request)
            self._check_bounds(request)
            self._check_authorization(request)
            self._check_lease(request)
            self._prepare_decision_runtime(request)
            current = self._state_store.get_mutation(request)
            if current is None:
                raise TransactionConflictError("mutation transaction is absent")
            if current.revision != expected_revision:
                raise TransactionConflictError(
                    f"stale transaction revision {expected_revision}; "
                    f"current revision is {current.revision}"
                )
            if current.recovery_action is not selected:
                raise TransactionConflictError(
                    f"transaction requires {current.recovery_action.value}, "
                    f"not {selected.value}"
                )
            hook = getattr(self._backend, selected.value, None)
            if not callable(hook):
                raise OperationUnavailableError(
                    f"backend does not implement {selected.value} recovery"
                )
            if self._decision_runtime is None:
                outcome = hook(request, current)
            else:
                decision = self._pending_runtime_decision

                def recover_dispatch() -> dict[str, Any]:
                    value = hook(request, current)
                    runtime_request = getattr(
                        decision, "decision_request", None
                    )
                    return {
                        "outcome": value,
                        "observed_effects": tuple(
                            getattr(runtime_request, "expected_effects", ())
                        ),
                    }

                executed = self._decision_runtime.authorize_mutation(
                    decision, recover_dispatch
                )
                wrapped = getattr(executed, "value", executed)
                outcome = (
                    wrapped.get("outcome")
                    if isinstance(wrapped, Mapping)
                    else wrapped
                )
            if outcome is False:
                raise BackendConflictError(
                    f"backend rejected {selected.value} recovery"
                )
            terminal = (
                MutationTransactionPhase.COMPENSATED
                if selected is MutationRecoveryAction.COMPENSATE
                else MutationTransactionPhase.REPAIRED
            )
            return self._state_store.compare_and_swap_mutation(
                request,
                expected_revision=current.revision,
                phase=terminal,
                now_ms=self._clock_ms(),
                applied_effect_ids=(
                    ()
                    if selected is MutationRecoveryAction.COMPENSATE
                    else current.applied_effect_ids
                ),
                result=current.result,
            )

    recover_mutation_transaction = recover_mutation

    def discovery_manifest(self) -> ControlDiscoveryManifest:
        """Return deterministic Python discovery metadata without dispatch."""

        expected = tuple(sorted(Operation, key=lambda item: item.value))
        if self._capability_report.supported_operations != expected:
            raise OperationUnavailableError(
                "Python discovery requires the complete control vocabulary"
            )
        return ControlDiscoveryManifest(surface=ControlSurface.PYTHON)

    def surface_publication(self) -> ControlSurfacePublication:
        """Return the validated, process-free Python surface publication."""

        return control_service_publication(self, catalog=self._catalog)

    def client(
        self,
        target: Union[SupervisorTarget, Mapping[str, Any], None] = None,
        **binding: Any,
    ) -> "SupervisorClient":
        return SupervisorClient(self, target=target, **binding)

    def _check_target(self, request: OperationRequest) -> None:
        repository = _normalized_absolute(
            request.repository_root, label="repository_root"
        )
        state = _normalized_absolute(request.state_root, label="state_root")
        if repository not in self._repository_roots:
            raise TargetNotAllowedError("repository_root is not allowlisted")
        if state not in self._state_roots:
            raise TargetNotAllowedError("state_root is not allowlisted")
        if self._identity_validator is not None:
            self._invoke_validator(
                self._identity_validator,
                request,
                denial=StaleTreeError,
            )

    def _check_bounds(self, request: OperationRequest) -> None:
        descriptor = self._catalog.operation(request.operation)
        explicit_limit = request.parameters.get("limit")
        descriptor.validate_bounds(
            request.bounds,
            page_limit=explicit_limit,
        )
        target = normalize_control_target(
            request,
            descriptor,
            service_id=self._service_id,
        )
        normalize_control_pagination(
            request,
            descriptor,
            target=target,
        )
        limit, _offset = _bounded_window(request)
        if (
            "limit" in request.parameters
            and limit > self._max_query_items
        ):
            raise ControlBoundsError("limit exceeds the service query bound")

    def _degraded_backend_response(
        self, request: OperationRequest
    ) -> Union[BackendResponse, None]:
        if request.operation in self._registered_operations:
            return None
        descriptor = self._catalog.operation(request.operation)
        if descriptor.degradation is CapabilityDegradation.PROPOSAL_ONLY:
            return BackendResponse(
                data={
                    "operation": request.operation.value,
                    "degraded": True,
                    "degradation": descriptor.degradation.value,
                    "backend_capability": descriptor.backend_capability,
                    "supported": False,
                },
                changed=False,
                warnings=(
                    f"{descriptor.backend_capability} is unavailable; "
                    "returning a proposal-only result",
                ),
                checks=("catalog_capability", "proposal_only"),
            )
        # LOCAL_READ_ONLY is advertised only when a local read implementation
        # really exists. A backend that omits the operation cannot claim that
        # degradation merely because the catalog permits it.
        raise UnsupportedCapabilityError(
            f"operation {request.operation.value} requires backend capability "
            f"{descriptor.backend_capability!r}"
        )

    @staticmethod
    def _invoke_validator(
        validator: Any,
        request: OperationRequest,
        *,
        denial: type[Exception],
    ) -> None:
        method = getattr(validator, "validate", None)
        if not callable(method):
            method = getattr(validator, "authorize", None)
        try:
            result = method(request) if callable(method) else validator(request)
        except denial:
            raise
        except Exception as exc:
            raise denial(str(exc) or "validator denied the bound request") from exc
        if result is False:
            raise denial("validator denied the bound request")

    def _check_authorization(self, request: OperationRequest) -> None:
        decision = request.authorization
        if decision is not None:
            now = self._clock_ms()
            if decision.evaluated_at_ms > now:
                raise AuthorizationBindingError(
                    "authorization decision is not yet valid"
                )
            if decision.expires_at_ms is not None and now >= decision.expires_at_ms:
                raise AuthorizationBindingError(
                    "authorization decision has expired"
                )
        if self._authorization_validator is not None:
            self._invoke_validator(
                self._authorization_validator,
                request,
                denial=AuthorizationBindingError,
            )

    def _check_lease(self, request: OperationRequest) -> None:
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            return
        if self._lease_validator is None:
            if self._require_lease_validator:
                raise LeaseValidationError(
                    "a live lease/fencing validator is required for mutation"
                )
            return
        try:
            self._invoke_validator(
                self._lease_validator,
                request,
                denial=StaleLeaseError,
            )
        except StaleLeaseError:
            raise
        except LeaseValidationError:
            raise
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "stale" in name or "expired" in name or "fenc" in name:
                raise StaleLeaseError(str(exc) or "lease is stale") from exc
            raise LeaseValidationError(str(exc) or "lease validation failed") from exc

    def _check_idempotency(
        self, request: OperationRequest
    ) -> Union[OperationResult, None]:
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            return None
        existing = self._state_store.get_idempotent(request)
        if existing is None:
            transaction = self._state_store.get_mutation(request)
            if transaction is None or transaction.result is None:
                return None
            existing = (transaction.request_id, transaction.result)
        request_id, result = existing
        if request_id != request.request_id:
            raise IdempotencyConflictError(
                "idempotency key is already bound to another request"
            )
        result.validate_against(request)
        return result

    @staticmethod
    def _normalize_backend_response(value: Any, request: OperationRequest) -> BackendResponse:
        if isinstance(value, BackendResponse):
            response = value
        elif isinstance(value, Mapping):
            response = BackendResponse(
                data=value,
                changed=request.operation in MUTATION_OPERATIONS
                and not request.dry_run,
                applied_effect_ids=tuple(
                    item.effect_id for item in request.expected_effects
                )
                if request.operation in MUTATION_OPERATIONS
                and not request.dry_run
                else (),
            )
        else:
            response = BackendResponse(data={"result": _canonical_json_value(value)})
        declared = {item.effect_id for item in request.expected_effects}
        if not set(response.applied_effect_ids).issubset(declared):
            raise ControlContractError(
                "backend claimed an effect not declared by the request"
            )
        if response.applied_effect_ids and not response.changed:
            raise ControlContractError(
                "backend cannot apply effects while reporting no change"
            )
        if response.applied_effect_ids and (
            request.operation not in MUTATION_OPERATIONS or request.dry_run
        ):
            raise AuthorityViolationError(
                "proposal and dry-run handlers cannot claim applied effects"
            )
        canonical_data = _canonical_json_value(response.data)
        return BackendResponse(
            data=redact_control_data(canonical_data),
            changed=response.changed,
            applied_effect_ids=response.applied_effect_ids,
            warnings=tuple(redact_control_text(item) for item in response.warnings),
            checks=tuple(redact_control_text(item) for item in response.checks),
        )

    def _dispatch(self, request: OperationRequest) -> BackendResponse:
        if request.operation is Operation.CAPABILITIES:
            report = self.capability_report()
            return BackendResponse(
                data={
                    "service_id": report.service_id,
                    "service_version": report.service_version,
                    "catalog_id": self._catalog.content_id,
                    "operations": tuple(
                        item.value for item in report.supported_operations
                    ),
                    "capability_report_id": report.content_id,
                    "optional_providers_loaded": (
                        report.optional_providers_loaded
                    ),
                    "processes_started": report.processes_started,
                },
                checks=("catalog_population", "provider_free", "process_free"),
            )
        if request.operation is Operation.RECEIPTS and not any(
            request.parameters.get(name)
            for name in ("receipts_path", "path")
        ):
            limit, offset = _bounded_window(request)
            items = self._state_store.query_receipts(
                request, limit=limit, offset=offset
            )
            return BackendResponse(
                data={
                    "items": list(items),
                    "count": len(items),
                    "limit": limit,
                    "offset": offset,
                    "truncated": len(items) == limit,
                }
            )
        degraded = self._degraded_backend_response(request)
        if degraded is not None:
            return degraded
        execute = getattr(self._backend, "execute", None)
        if not callable(execute):
            raise OperationUnavailableError(
                "control backend does not provide execute(request)"
            )
        if request.operation in MUTATION_OPERATIONS and not request.dry_run:
            self._mutation_dispatch_count += 1
            self._last_mutation_dispatch_request_id = request.request_id
        started_ns = time.monotonic_ns()
        value = execute(request)
        elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000
        if elapsed_ms > request.bounds.timeout_ms:
            raise BackendTimeoutError(
                f"backend execution exceeded {request.bounds.timeout_ms}ms"
            )
        return self._normalize_backend_response(value, request)

    @staticmethod
    def _status_for_error(code: ErrorCode) -> OperationStatus:
        if code in {ErrorCode.UNAUTHORIZED, ErrorCode.FORBIDDEN}:
            return OperationStatus.DENIED
        if code is ErrorCode.NOT_FOUND:
            return OperationStatus.NOT_FOUND
        if code in {
            ErrorCode.CONFLICT,
            ErrorCode.STALE_TREE,
            ErrorCode.STALE_LEASE,
            ErrorCode.IDEMPOTENCY_CONFLICT,
            ErrorCode.INVALID_LIFECYCLE_TRANSITION,
        }:
            return OperationStatus.CONFLICT
        if code in {
            ErrorCode.UNAVAILABLE,
            ErrorCode.UNSUPPORTED_CAPABILITY,
        }:
            return OperationStatus.UNAVAILABLE
        if code is ErrorCode.TIMED_OUT:
            return OperationStatus.TIMED_OUT
        if code is ErrorCode.CANCELLED:
            return OperationStatus.CANCELLED
        return OperationStatus.FAILED

    @staticmethod
    def _stable_error(exc: BaseException) -> OperationError:
        message = str(exc).strip() or type(exc).__name__
        field = ""
        retryable = False
        if isinstance(exc, TargetNotAllowedError):
            code = ErrorCode.FORBIDDEN
            field = "repository_root" if "repository" in message else "state_root"
        elif isinstance(exc, AuthorizationBindingError):
            code = ErrorCode.UNAUTHORIZED
        elif isinstance(exc, StaleLeaseError):
            code = ErrorCode.STALE_LEASE
            retryable = True
        elif isinstance(exc, StaleTreeError):
            code = ErrorCode.STALE_TREE
            retryable = True
        elif isinstance(exc, LeaseValidationError):
            code = ErrorCode.STALE_LEASE
            retryable = True
        elif isinstance(exc, IdempotencyConflictError):
            code = ErrorCode.IDEMPOTENCY_CONFLICT
        elif isinstance(exc, (TransactionConflictError, PartialMutationError)):
            code = ErrorCode.CONFLICT
        elif type(exc).__name__ in {
            "DecisionRuntimeDenied",
            "DecisionRuntimeBypassError",
            "DecisionRuntimeEffectMismatch",
        }:
            code = ErrorCode.UNAUTHORIZED
        elif type(exc).__name__ == "DecisionRuntimeCancelled":
            code = ErrorCode.CANCELLED
        elif isinstance(exc, BackendNotFoundError) or isinstance(
            exc, FileNotFoundError
        ):
            code = ErrorCode.NOT_FOUND
        elif isinstance(exc, InvalidLifecycleTransitionError):
            code = ErrorCode.INVALID_LIFECYCLE_TRANSITION
        elif isinstance(exc, BackendConflictError):
            code = ErrorCode.CONFLICT
        elif isinstance(exc, BackendCancelledError):
            code = ErrorCode.CANCELLED
        elif isinstance(exc, (BackendTimeoutError, TimeoutError)):
            code = ErrorCode.TIMED_OUT
            retryable = True
        elif isinstance(exc, UnsupportedCapabilityError):
            code = ErrorCode.UNSUPPORTED_CAPABILITY
        elif isinstance(exc, EventCursorError):
            code = ErrorCode.INVALID_CURSOR
        elif isinstance(exc, OperationUnavailableError):
            code = ErrorCode.UNAVAILABLE
        elif isinstance(exc, PermissionError):
            code = ErrorCode.FORBIDDEN
        elif isinstance(exc, PathEscapeError):
            code = ErrorCode.PATH_ESCAPE
            field = "path"
        elif isinstance(exc, ControlBoundsError):
            code = ErrorCode.BOUNDS_EXCEEDED
        elif isinstance(
            exc,
            (ControlContractError, ValueError, TypeError, json.JSONDecodeError),
        ):
            code = ErrorCode.INVALID_REQUEST
        elif type(exc).__name__ in {"CancelledError", "CancellationError"}:
            code = ErrorCode.CANCELLED
        else:
            code = ErrorCode.INTERNAL_ERROR
            message = "control operation failed"
        message = redact_control_text(message)
        return OperationError(
            code=code,
            message=message[:2048],
            retryable=retryable,
            field=field,
            details={"exception_type": type(exc).__name__},
        )

    def _receipt(
        self,
        request: OperationRequest,
        *,
        status: OperationStatus,
        applied_effect_ids: Iterable[str] = (),
        error: Union[OperationError, None] = None,
    ) -> ControlAuditReceipt:
        now = self._clock_ms()
        authorization = request.authorization
        return ControlAuditReceipt(
            request_id=request.request_id,
            operation=request.operation.value,
            authority=request.effective_authority.value,
            status=status.value,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            authorization_decision_id=(
                authorization.decision_id if authorization is not None else ""
            ),
            grant_ids=(
                authorization.grant_ids if authorization is not None else ()
            ),
            dry_run=request.dry_run,
            idempotency_key=request.idempotency_key,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch,
            effect_ids=tuple(item.effect_id for item in request.expected_effects),
            applied_effect_ids=tuple(sorted(set(applied_effect_ids))),
            error_code=error.code.value if error else "",
            occurred_at_ms=now,
            occurred_at=_utc_timestamp(now),
        )

    @staticmethod
    def _claims(
        request: OperationRequest,
        applied_effect_ids: Iterable[str],
        receipt_id: str,
    ) -> tuple[EffectClaim, ...]:
        applied = set(applied_effect_ids)
        return tuple(
            EffectClaim(
                effect_id=effect.effect_id,
                kind=effect.kind,
                resource=effect.resource,
                paths=effect.paths,
                applied=effect.effect_id in applied,
                receipt_id=receipt_id if effect.effect_id in applied else "",
            )
            for effect in request.expected_effects
        )

    @staticmethod
    def _preview(
        request: OperationRequest,
        response: Union[BackendResponse, None] = None,
    ) -> DryRunPreview:
        return DryRunPreview(
            request_id=request.request_id,
            operation=request.operation,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            expected_effects=request.expected_effects,
            checks=(response.checks if response else ("authorization", "bounds", "allowlists")),
            warnings=response.warnings if response else (),
            would_change=bool(request.expected_effects)
            if response is None
            else response.changed,
        )

    def _success_result(
        self,
        request: OperationRequest,
        response: BackendResponse,
        *,
        transaction_state: Union[MutationTransactionState, None] = None,
    ) -> OperationResult:
        applied = (
            response.applied_effect_ids
            if request.operation in MUTATION_OPERATIONS and not request.dry_run
            else ()
        )
        receipt = self._receipt(
            request,
            status=OperationStatus.SUCCEEDED,
            applied_effect_ids=applied,
        )
        preview = None
        authority = request.effective_authority
        if request.dry_run and request.operation in MUTATION_OPERATIONS:
            preview = self._preview(request, response)
            authority = OperationAuthority.PROPOSAL
        elif request.operation in PROPOSAL_OPERATIONS:
            preview = self._preview(request, response)
        result = OperationResult(
            request_id=request.request_id,
            operation=request.operation,
            authority=authority,
            status=OperationStatus.SUCCEEDED,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            bounds=request.bounds,
            data=response.data,
            # Mutation-shaped expected effects belong in the proposal preview
            # during dry-run.  Even an unapplied mutation EffectClaim would
            # exceed the result's proposal-only authority.
            effects=(
                ()
                if (
                    request.dry_run
                    and request.operation in MUTATION_OPERATIONS
                )
                or request.operation in DOWNSTREAM_EFFECT_PREVIEW_OPERATIONS
                else self._claims(request, applied, receipt.receipt_id)
            ),
            preview=preview,
            idempotency_key=request.idempotency_key,
            audit_receipt_id=receipt.receipt_id,
        )
        result.validate_against(request)
        self._state_store.append_receipt(request, receipt)
        if request.operation in MUTATION_OPERATIONS and not request.dry_run:
            if transaction_state is None:
                raise TransactionConflictError(
                    "real mutation completed without transaction state"
                )
            transaction_state = self._state_store.compare_and_swap_mutation(
                request,
                expected_revision=transaction_state.revision,
                phase=MutationTransactionPhase.COMMITTED,
                now_ms=self._clock_ms(),
                applied_effect_ids=applied,
                result=result,
            )
            self._state_store.put_idempotent(request, result)
            self._mutation_audit_receipt_count += 1
            self._last_mutation_audit_receipt_id = receipt.receipt_id
        return result

    def _error_result(
        self,
        request: OperationRequest,
        exc: BaseException,
        *,
        transaction_state: Union[MutationTransactionState, None] = None,
    ) -> OperationResult:
        error = self._stable_error(exc)
        status = self._status_for_error(error.code)
        applied_effect_ids: tuple[str, ...] = ()
        recovery_action = MutationRecoveryAction.REPAIR
        if isinstance(exc, PartialMutationError):
            applied_effect_ids = exc.applied_effect_ids
            recovery_action = exc.recovery
        declared = {item.effect_id for item in request.expected_effects}
        if not set(applied_effect_ids).issubset(declared):
            error = self._stable_error(
                ControlContractError(
                    "partial mutation reported an undeclared applied effect"
                )
            )
            status = self._status_for_error(error.code)
            applied_effect_ids = ()
            recovery_action = MutationRecoveryAction.REPAIR
        receipt = self._receipt(
            request,
            status=status,
            applied_effect_ids=applied_effect_ids,
            error=error,
        )
        recovery_phase = (
            MutationTransactionPhase.COMPENSATION_REQUIRED
            if recovery_action is MutationRecoveryAction.COMPENSATE
            else MutationTransactionPhase.REPAIR_REQUIRED
        )
        transaction_data: dict[str, Any] = {}
        if transaction_state is not None:
            transaction_data = {
                "transaction": {
                    **transaction_state.to_dict(),
                    "phase": recovery_phase.value,
                    "revision": transaction_state.revision + 1,
                    "applied_effect_ids": list(applied_effect_ids),
                    "recovery_action": recovery_action.value,
                    "failure_code": error.code.value,
                }
            }
        result = OperationResult(
            request_id=request.request_id,
            operation=request.operation,
            authority=request.effective_authority,
            status=status,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            caller=request.caller,
            bounds=request.bounds,
            data=transaction_data,
            effects=(
                self._claims(
                    request,
                    applied_effect_ids,
                    receipt.receipt_id,
                )
                if applied_effect_ids
                else ()
            ),
            error=error,
            idempotency_key=request.idempotency_key,
            audit_receipt_id=receipt.receipt_id,
        )
        result.validate_against(request)
        try:
            self._state_store.append_receipt(request, receipt)
            if transaction_state is not None:
                transaction_state = self._state_store.compare_and_swap_mutation(
                    request,
                    expected_revision=transaction_state.revision,
                    phase=recovery_phase,
                    now_ms=self._clock_ms(),
                    applied_effect_ids=applied_effect_ids,
                    failure_code=error.code.value,
                    result=result,
                )
                self._state_store.put_idempotent(request, result)
                self._mutation_audit_receipt_count += 1
                self._last_mutation_audit_receipt_id = receipt.receipt_id
        except Exception:
            # Stable operation errors must never be replaced by an audit sink
            # failure.  Successful operations deliberately fail if durable
            # auditing fails, because reporting an unaudited mutation as
            # success would violate the control boundary.
            pass
        return result

    def _preflight_dispatch_boundary(
        self, request: OperationRequest
    ) -> Union[OperationResult, None]:
        """Apply every runtime guard before a mutating adapter can dispatch.

        Structural authorization, idempotency scope, declared effects, and
        path containment have already been checked by ``OperationRequest``.
        This boundary adds deployment allowlists, current identity and
        authorization checks, replay/conflict detection, and the live
        lease/fencing decision.  CLI and MCP decode before resolving their
        service, then converge here with direct Python calls.
        """

        self._check_target(request)
        self._check_bounds(request)
        self._check_authorization(request)
        replay = self._check_idempotency(request)
        if replay is not None:
            return replay
        self._check_lease(request)
        self._prepare_decision_runtime(request)
        return None

    @staticmethod
    def _control_runtime_boundary(request: OperationRequest) -> str:
        if request.operation is Operation.VALIDATION_REPLAY:
            return "validation_execution"
        if request.operation in {
            Operation.OBJECTIVE_REFINE,
            Operation.OBJECTIVE_RECONCILE,
            Operation.BACKLOG_REFILL,
        }:
            return "task_board_mutation"
        return "tool_invocation"

    def _runtime_cancelled(self) -> bool:
        value = self._decision_runtime_cancellation
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if callable(value):
            return bool(value())
        checker = getattr(value, "is_set", None)
        if callable(checker):
            return bool(checker())
        raise TypeError(
            "decision_runtime_cancellation must be a boolean, predicate, "
            "event, or None"
        )

    def _prepare_decision_runtime(self, request: OperationRequest) -> None:
        """Resolve one shared Python/CLI/MCP decision before dispatch."""

        self._pending_runtime_decision = None
        if request.operation not in MUTATION_OPERATIONS or request.dry_run:
            return
        if self._runtime_cancelled():
            raise BackendCancelledError(
                "control operation cancelled before runtime decision"
            )
        runtime = self._decision_runtime
        runtime_config = request.parameters.get("decision_runtime")
        if runtime_config is not None:
            from ..context.decision_runtime import (
                DecisionRuntime,
                DecisionRuntimeConfig,
            )

            decoded = DecisionRuntimeConfig.from_dict(runtime_config)
            if runtime is None:
                runtime = DecisionRuntime(
                    decoded,
                    cancellation=self._decision_runtime_cancellation,
                )
                self._decision_runtime = runtime
            elif (
                getattr(runtime, "config", None) is not None
                and getattr(runtime.config, "config_id", None)
                != decoded.config_id
            ):
                raise ControlContractError(
                    "control request decision_runtime config differs from "
                    "the active service runtime"
                )
        if runtime is None:
            return
        route = getattr(runtime, "route", None)
        if not callable(route):
            raise TypeError("decision_runtime must expose route()")
        self._pending_runtime_decision = route(
            self._control_runtime_boundary(request),
            {
                "transport_request_id": request.request_id,
                "operation": request.operation.value,
                "repository_id": request.repository_id,
                "tree_id": request.tree_id,
                "objective_id": request.objective_id,
                "policy_id": request.policy_id,
                "policy_revision": request.policy_revision,
                "caller": request.caller,
                "lease_id": request.lease_id,
                "fencing_epoch": request.fencing_epoch,
                "idempotency_key": request.idempotency_key,
                "expected_effect_ids": tuple(
                    item.effect_id for item in request.expected_effects
                ),
            },
        )

    def _dispatch_with_decision_runtime(
        self, request: OperationRequest
    ) -> BackendResponse:
        """Check the current permit immediately adjacent to backend dispatch."""

        runtime = self._decision_runtime
        decision = self._pending_runtime_decision
        if runtime is None:
            return self._dispatch(request)
        if self._runtime_cancelled():
            raise BackendCancelledError(
                "control operation cancelled before backend dispatch"
            )
        authorize = getattr(runtime, "authorize_mutation", None)
        if not callable(authorize):
            raise TypeError(
                "decision_runtime must expose authorize_mutation()"
            )

        def dispatch() -> dict[str, Any]:
            response = self._dispatch(request)
            runtime_request = getattr(decision, "decision_request", None)
            expected = tuple(
                effect
                for effect in getattr(runtime_request, "expected_effects", ())
                if effect.effect_id in set(response.applied_effect_ids)
            )
            return {
                "response": response,
                "observed_effects": expected,
            }

        result = authorize(decision, dispatch)
        wrapped = getattr(result, "value", result)
        if not isinstance(wrapped, Mapping) or not isinstance(
            wrapped.get("response"), BackendResponse
        ):
            raise ControlContractError(
                "decision runtime returned an invalid backend response"
            )
        return wrapped["response"]

    def execute(
        self, request: Union[OperationRequest, Mapping[str, Any]]
    ) -> OperationResult:
        """Validate, dispatch, audit, and return one typed operation result."""

        if not isinstance(request, OperationRequest):
            request = decode_operation_request(request)
        transaction = getattr(self._state_store, "transaction", None)

        @contextmanager
        def no_transaction() -> Iterator[None]:
            yield

        guard = transaction(request) if callable(transaction) else no_transaction()
        with self._lock, guard:
            backend_accepted = False
            mutation_state: Union[MutationTransactionState, None] = None
            try:
                replay = self._preflight_dispatch_boundary(request)
                if replay is not None:
                    hook = getattr(self._backend, "record_replay", None)
                    if callable(hook) and request.operation in MUTATION_OPERATIONS:
                        hook(request)
                    return replay
                if request.operation in MUTATION_OPERATIONS and not request.dry_run:
                    mutation_state = self._state_store.begin_mutation(
                        request, now_ms=self._clock_ms()
                    )
                    if mutation_state.phase is MutationTransactionPhase.COMMITTED:
                        if mutation_state.result is None:
                            raise TransactionConflictError(
                                "committed mutation has no durable result"
                            )
                        mutation_state.result.validate_against(request)
                        self._state_store.put_idempotent(
                            request, mutation_state.result
                        )
                        return mutation_state.result
                    if mutation_state.phase is not MutationTransactionPhase.PREPARED:
                        if mutation_state.result is not None:
                            mutation_state.result.validate_against(request)
                            return mutation_state.result
                        raise TransactionConflictError(
                            "mutation transaction requires "
                            f"{mutation_state.recovery_action.value} at revision "
                            f"{mutation_state.revision}"
                        )
                    mutation_state = self._state_store.compare_and_swap_mutation(
                        request,
                        expected_revision=mutation_state.revision,
                        phase=MutationTransactionPhase.DISPATCHING,
                        now_ms=self._clock_ms(),
                    )
                if request.dry_run and request.operation in MUTATION_OPERATIONS:
                    # A dry run never invokes a mutating adapter.
                    response = BackendResponse(
                        data={"dry_run": True, "would_change": bool(request.expected_effects)},
                        changed=bool(request.expected_effects),
                        checks=("authorization", "bounds", "allowlists", "expected_effects"),
                    )
                else:
                    response = self._dispatch_with_decision_runtime(request)
                    backend_accepted = True
                return self._success_result(
                    request,
                    response,
                    transaction_state=mutation_state,
                )
            except BaseException as exc:
                if not isinstance(exc, Exception) and type(exc).__name__ not in {
                    "CancelledError",
                    "CancellationError",
                }:
                    raise
                result = self._error_result(
                    request,
                    exc,
                    transaction_state=(
                        mutation_state
                        if mutation_state is not None
                        and mutation_state.phase
                        is MutationTransactionPhase.DISPATCHING
                        else None
                    ),
                )
                hook = getattr(self._backend, "record_rejection", None)
                if (
                    callable(hook)
                    and request.operation in MUTATION_OPERATIONS
                    and not backend_accepted
                ):
                    try:
                        if result.error is not None:
                            hook(request, result.error)
                    except Exception:
                        pass
                return result

    handle = execute
    dispatch = execute

    def _operation(
        self, operation: Operation, request: OperationRequest
    ) -> OperationResult:
        if not isinstance(request, OperationRequest):
            raise TypeError("request must be an OperationRequest")
        if request.operation is not operation:
            raise ValueError(
                f"request operation must be {operation.value}, got {request.operation.value}"
            )
        return self.execute(request)

    def capabilities(
        self, request: Union[OperationRequest, None] = None
    ) -> Union[CapabilityReport, OperationResult]:
        return (
            self.capability_report()
            if request is None
            else self._operation(Operation.CAPABILITIES, request)
        )

    def status(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.STATUS, request)

    def health(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.HEALTH, request)

    def metrics(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.METRICS, request)

    def goals(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.GOALS, request)

    def tasks(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.TASKS, request)

    def bundles(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.BUNDLES, request)

    def lanes(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.LANES, request)

    def events(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.EVENTS, request)

    def receipts(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RECEIPTS, request)

    def cache_inspect(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.CACHE_INSPECT, request)

    cache = cache_inspect

    def artifact_query(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.ARTIFACT_QUERY, request)

    def objective_preview(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_PREVIEW, request)

    preview = objective_preview

    def objective_refine(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_REFINE, request)

    refine = objective_refine

    def objective_reconcile(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.OBJECTIVE_RECONCILE, request)

    reconcile = objective_reconcile

    def backlog_refill(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.BACKLOG_REFILL, request)

    refill = backlog_refill

    def plan(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.PLAN, request)

    def workflow_preview(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.WORKFLOW_PREVIEW, request)

    def workflow_materialize(
        self, request: OperationRequest
    ) -> OperationResult:
        return self._operation(Operation.WORKFLOW_MATERIALIZE, request)

    def start(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.START, request)

    def pause(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.PAUSE, request)

    def resume(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RESUME, request)

    def drain(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.DRAIN, request)

    def stop(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.STOP, request)

    def restart(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RESTART, request)

    def retry(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RETRY, request)

    def cancel(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.CANCEL, request)

    def quarantine(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.QUARANTINE, request)

    def validation_replay(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.VALIDATION_REPLAY, request)

    replay_validation = validation_replay

    def rescue_preview(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RESCUE_PREVIEW, request)

    def rescue(self, request: OperationRequest) -> OperationResult:
        return self._operation(Operation.RESCUE, request)

    def lifecycle(
        self,
        request: OperationRequest,
        command: Union[LifecycleCommand, None] = None,
    ) -> OperationResult:
        if command is not None:
            if request.operation is not command.operation:
                raise ValueError("lifecycle command does not match request operation")
            if request.dry_run != command.dry_run:
                raise ValueError("lifecycle command dry_run does not match request")
            target = str(request.parameters.get("target_id") or "")
            if target != command.target_id:
                raise ValueError("lifecycle command target does not match request")
            reason = str(request.parameters.get("reason") or "")
            if reason != command.reason:
                raise ValueError("lifecycle command reason does not match request")
            requested_state = str(
                request.parameters.get("requested_state") or ""
            )
            if requested_state != command.requested_state:
                raise ValueError(
                    "lifecycle command requested_state does not match request"
                )
        if request.operation not in {
            action.operation for action in LifecycleAction
        }:
            raise ValueError("request is not a lifecycle operation")
        return self.execute(request)


def control_service_publication(
    service: Union[SupervisorControlService, None] = None,
    *,
    catalog: OperationCatalog = DEFAULT_CONTROL_CATALOG,
) -> ControlSurfacePublication:
    """Publish the exhaustive Python service surface without dispatching it."""

    selected_catalog = (
        service.operation_catalog() if service is not None else catalog
    )
    _validate_canonical_catalog(selected_catalog)
    service_type = type(service) if service is not None else SupervisorControlService
    if not callable(getattr(service_type, "execute", None)):
        raise ControlCatalogConformanceError(
            "SupervisorControlService.execute is not callable"
        )
    missing_methods = tuple(
        operation.value
        for operation in selected_catalog.operations
        if not callable(getattr(service_type, operation.value, None))
    )
    if missing_methods:
        raise ControlCatalogConformanceError(
            "Python service is missing catalog operation methods: "
            + ", ".join(missing_methods)
        )
    publication = ControlSurfacePublication(
        surface=ControlSurface.PYTHON,
        catalog_id=selected_catalog.content_id,
        catalog_version=selected_catalog.catalog_version,
        operations=selected_catalog.operations,
        request_schema_ids={
            item.operation: item.request_schema_id
            for item in selected_catalog
        },
        result_schema_ids={
            item.operation: item.result_schema_id
            for item in selected_catalog
        },
        behavior_ids={
            item.operation: control_operation_behavior_id(item)
            for item in selected_catalog
        },
        dispatcher_ids={
            item.operation: DIRECT_CONTROL_SERVICE_DISPATCHER_ID
            for item in selected_catalog
        },
    )
    return validate_control_surface_publication(
        publication,
        selected_catalog,
    )


class SupervisorClient:
    """Read-oriented facade over :class:`SupervisorControlService`.

    A target binding lets the facade construct read and proposal requests.
    Mutations are exposed only for callers which already hold a fully formed
    :class:`OperationRequest`; the client never manufactures authorization,
    idempotency, leases, or fencing epochs.
    """

    def __init__(
        self,
        service: SupervisorControlService,
        target: Union[SupervisorTarget, Mapping[str, Any], None] = None,
        *,
        bounds: Union[ControlBounds, None] = None,
        **binding: Any,
    ) -> None:
        if not isinstance(service, SupervisorControlService):
            raise TypeError("service must be a SupervisorControlService")
        if target is not None and binding:
            raise ValueError("supply target or binding fields, not both")
        if isinstance(target, Mapping):
            target = SupervisorTarget(**dict(target))
        elif target is None and binding:
            target = SupervisorTarget(**binding)
        elif target is not None and not isinstance(target, SupervisorTarget):
            raise TypeError("target must be a SupervisorTarget or mapping")
        self._service = service
        self._target = target
        self._bounds = bounds or ControlBounds()

    @property
    def service(self) -> SupervisorControlService:
        return self._service

    @property
    def target(self) -> Union[SupervisorTarget, None]:
        return self._target

    def capabilities(self) -> CapabilityReport:
        return self._service.capability_report()

    def execute(self, request: OperationRequest) -> OperationResult:
        return self._service.execute(request)

    def _read(
        self,
        operation: Operation,
        parameters: Union[Mapping[str, Any], None] = None,
        **values: Any,
    ) -> OperationResult:
        if self._target is None:
            raise ValueError("a SupervisorTarget is required to construct requests")
        merged = dict(parameters or {})
        overlap = set(merged).intersection(values)
        if overlap:
            raise ValueError(
                "duplicate request parameters: " + ", ".join(sorted(overlap))
            )
        merged.update(values)
        request = self._target.request(
            operation, parameters=merged, bounds=self._bounds
        )
        return self._service.execute(request)

    def _authorized(
        self, operation: Operation, request: OperationRequest
    ) -> OperationResult:
        return self._service._operation(operation, request)

    def status(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.STATUS, parameters, **values)

    def health(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.HEALTH, parameters, **values)

    def metrics(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.METRICS, parameters, **values)

    def goals(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.GOALS, parameters, **values)

    def tasks(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.TASKS, parameters, **values)

    def bundles(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.BUNDLES, parameters, **values)

    def lanes(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.LANES, parameters, **values)

    def events(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.EVENTS, parameters, **values)

    def receipts(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.RECEIPTS, parameters, **values)

    def cache_inspect(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.CACHE_INSPECT, parameters, **values)

    cache = cache_inspect

    def artifact_query(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.ARTIFACT_QUERY, parameters, **values)

    def objective_preview(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.OBJECTIVE_PREVIEW, parameters, **values)

    preview = objective_preview

    def plan(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.PLAN, parameters, **values)

    def workflow_preview(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.WORKFLOW_PREVIEW, parameters, **values)

    def rescue_preview(
        self, parameters: Union[Mapping[str, Any], None] = None, **values: Any
    ) -> OperationResult:
        return self._read(Operation.RESCUE_PREVIEW, parameters, **values)

    def objective_refine(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.OBJECTIVE_REFINE, request)

    refine = objective_refine

    def objective_reconcile(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.OBJECTIVE_RECONCILE, request)

    reconcile = objective_reconcile

    def backlog_refill(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.BACKLOG_REFILL, request)

    refill = backlog_refill

    def workflow_materialize(
        self, request: OperationRequest
    ) -> OperationResult:
        return self._authorized(Operation.WORKFLOW_MATERIALIZE, request)

    def start(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.START, request)

    def pause(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.PAUSE, request)

    def resume(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RESUME, request)

    def drain(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.DRAIN, request)

    def stop(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.STOP, request)

    def restart(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RESTART, request)

    def retry(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RETRY, request)

    def cancel(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.CANCEL, request)

    def quarantine(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.QUARANTINE, request)

    def validation_replay(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.VALIDATION_REPLAY, request)

    replay_validation = validation_replay

    def rescue(self, request: OperationRequest) -> OperationResult:
        return self._authorized(Operation.RESCUE, request)

    def lifecycle(
        self,
        request: OperationRequest,
        command: Union[LifecycleCommand, None] = None,
    ) -> OperationResult:
        return self._service.lifecycle(request, command)


ReadOnlySupervisorClient = SupervisorClient
SupervisorReadClient = SupervisorClient
PythonSupervisorBackend = RepositorySupervisorBackend
ControlService = SupervisorControlService


__all__ = [
    "CONTROL_AUDIT_RECEIPT_SCHEMA",
    "CONTROL_BACKEND_RESPONSE_SCHEMA",
    "CONTROL_BEHAVIOR_NORMALIZATION_VERSION",
    "CONTROL_CATALOG_CONFORMANCE_EVIDENCE_SCHEMA",
    "CONTROL_CONFORMANCE_V2_REQUIREMENT_ID",
    "CONTROL_MUTATION_EVENT_SCHEMA",
    "CONTROL_MUTATION_TRANSACTION_SCHEMA",
    "CONTROL_OPERATION_CONFORMANCE_CASE_SCHEMA",
    "CONTROL_OPTIONAL_PROVIDER_MODULE_PREFIXES",
    "CONTROL_REDACTION_MARKER",
    "CONTROL_SENSITIVE_FIELD_NAMES",
    "CONTROL_SERVICE_VERSION",
    "CONTROL_SURFACE_PUBLICATION_SCHEMA",
    "DEFAULT_MAX_CONTROL_EVENTS",
    "DIRECT_CONTROL_SERVICE_DISPATCHER_ID",
    "LEGAL_LIFECYCLE_TRANSITIONS",
    "LIFECYCLE_EVENT_SCHEMA",
    "LIFECYCLE_STATUS_SCHEMA",
    "BackendCancelledError",
    "BackendConflictError",
    "BackendNotFoundError",
    "BackendResponse",
    "BackendTimeoutError",
    "AuthorizationValidator",
    "ControlCatalogConformanceError",
    "ControlCatalogConformanceEvidence",
    "ControlAuditReceipt",
    "ControlService",
    "ControlStateStore",
    "ControlSurfacePublication",
    "ControlOperationConformanceCase",
    "IdempotencyConflictError",
    "InMemoryControlStateStore",
    "InMemoryLifecycleStore",
    "InvalidLifecycleTransitionError",
    "JsonLifecycleStore",
    "JsonlControlStateStore",
    "LifecycleEvent",
    "LifecycleStatus",
    "LeaseFenceValidator",
    "LeaseValidationError",
    "MutationRecoveryAction",
    "MutationTransactionPhase",
    "MutationTransactionState",
    "OperationHandler",
    "OperationUnavailableError",
    "PartialMutationError",
    "PythonSupervisorBackend",
    "ReadOnlySupervisorClient",
    "RepositorySupervisorBackend",
    "RescueHandler",
    "RescuePreviewHandler",
    "RestartHandler",
    "StaleLeaseError",
    "StaleTreeError",
    "SupervisorClient",
    "SupervisorControlError",
    "SupervisorControlService",
    "SupervisorLifecycleBackend",
    "SupervisorLifecycleState",
    "SupervisorReadClient",
    "SupervisorTarget",
    "TargetNotAllowedError",
    "TargetIdentityValidator",
    "TransactionConflictError",
    "WorkflowMaterializeHandler",
    "WorkflowPreviewHandler",
    "capture_control_discovery_runtime_state",
    "control_operation_behavior_id",
    "control_service_publication",
    "lifecycle_transition_is_legal",
    "normalize_control_pagination",
    "normalize_control_request",
    "normalize_control_result",
    "normalize_control_target",
    "publish_control_catalog",
    "redact_control_data",
    "redact_control_text",
    "validate_catalog_publication",
    "validate_control_catalog_publication",
    "validate_control_surface_manifest",
    "validate_control_surface_publication",
    "validate_operation_request_against_catalog",
]
