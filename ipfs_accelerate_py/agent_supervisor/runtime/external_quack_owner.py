"""Fenced EAAEF facade over the sole live :class:`QuackStateServer` owner.

EAAEF-093 used to model an owner with process-local dictionaries and a Python
list standing in for Quack.  That model was unsafe as qualification evidence:
it opened no DuckDB database, bound no Quack endpoint, and advanced an integer
instead of acquiring the durable state-owner generation.

The facade in this module owns no resources.  A READY ``QuackStateServer``
creates it from the server's exact live identity.  It never opens a database,
creates a dispatcher, accepts a task-source callback, or exposes SQL.  It also
refuses to synthesize a generic daemon gateway while the canonical 39-operation
owner handler, host artifacts, and Plan R2 admission remain unqualified.

DOEP-044 extends this same facade with owner-loss and owner-restart recovery.
A lost exclusive owner cannot complete a task.  A restarted owner is admitted
only as a later generation and fence of the same store.  Stale leases and
worker assertions never bypass that gate.  This is not a second owner, backup
service, or completion authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from ..task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    TaskStateSnapshot,
    content_identity,
)
from ..task_sources.quack_daemon_gateway import (
    QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS,
    REQUIRED_QUACK_DAEMON_OPERATIONS,
    QuackDaemonGatewayError,
    quack_daemon_owner_operation_dispositions,
)

if TYPE_CHECKING:
    from .quack_state_server import QuackStateServer, StateServerIdentity


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_QUACK_OWNER_INTERFACE: Final[str] = "ExternalQuackOwner@1"
EXTERNAL_QUACK_OWNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-owner@1"
)
OWNER_LEASE_INTERFACE: Final[str] = "ExternalQuackOwnerLease@1"
OWNER_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-owner-lease@1"
)
# Import-compatible identities for the retired process-local artifacts.  No
# issuer or transport implementation remains behind these names.
ENVELOPE_INTERFACE: Final[str] = "ExternalQuackEnvelope@1"
ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-envelope@1"
)
APPLY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-apply-receipt@1"
)
TRANSPORT_INTERFACE: Final[str] = "BoundedQuackTransport@1"
TRANSPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/bounded-quack-transport@1"
)
EXTERNAL_QUACK_OWNER_QUALIFICATION_STATUS: Final[str] = (
    "real_owner_bound_gateway_operations_unqualified_fail_closed"
)
EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER: Final[str] = (
    "canonical_39_operation_owner_handler_unqualified"
)
OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING: Final[str] = (
    "OwnerLossAndOwnerRestartRecovery@1"
)
OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE: Final[str] = (
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
)
OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/owner-loss-and-owner-restart-recovery@1"
)
OWNER_LOSS_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/owner-loss-observation@1"
)
OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES: Final[tuple[str, ...]] = (
    EXTERNAL_QUACK_OWNER_INTERFACE,
    OWNER_LEASE_INTERFACE,
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
)
_STOPPED_OWNER_LIFECYCLES: Final[frozenset[str]] = frozenset(
    {"stopped", "stopping", "failed"}
)

# Compatibility identities retained for later EAAEF integration tests.  They
# no longer describe a process-local qualification implementation.
INITIAL_EPOCH: Final[int] = 1
INITIAL_FENCE: Final[int] = 1
LIVE_QUACK_PORT: Final[int] = 19495
ALLOWED_OPERATIONS: Final[frozenset[str]] = frozenset()
REMOTE_CAPABILITIES: Final[frozenset[str]] = frozenset()

_FACADE_CONSTRUCTION_TOKEN: Final[object] = object()
_SQL_OPERATION_MARKERS: Final[frozenset[str]] = frozenset(
    {"sql", "execute_sql", "remote_update_sql", "update", "query"}
)


class ExternalQuackOwnerError(ValueError):
    """The external owner facade was unavailable or cross-bound."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ExternalQuackOwnerNotReady(ExternalQuackOwnerError):
    """The backing server is not the exact READY exclusive owner."""


class StaleOwnerError(ExternalQuackOwnerError):
    """A lease belongs to an earlier server generation or fence."""


class DuplicateOwnerError(ExternalQuackOwnerError):
    """Compatibility error name for a refused second owner."""


class RemoteSqlRefusedError(ExternalQuackOwnerError):
    """SQL is outside the closed command-gateway vocabulary."""


class RetiredInMemoryOwnerError(ExternalQuackOwnerError):
    """A caller attempted to use the retired process-local owner model."""


class UnsignedEnvelopeError(RetiredInMemoryOwnerError):
    """Compatibility name for the retired content-hash envelope model."""


class OwnerLossKind(str, Enum):
    """Closed vocabulary for exclusive-owner loss."""

    NOT_READY = "owner_not_ready"
    LOST_HOLD = "lost_hold"
    PROCESS_STOPPED = "process_stopped"
    STALE_GENERATION = "stale_generation"
    STALE_FENCE = "stale_fence"


class OwnerRecoveryOutcome(str, Enum):
    """Closed vocabulary for owner-loss and owner-restart recovery."""

    OWNER_LOST = "owner_lost"
    SUCCESSOR_ADMITTED = "successor_admitted"
    STALE_REJECTED = "stale_rejected"
    INVALID_FAILOVER = "invalid_failover"


class TransportAuthError(RetiredInMemoryOwnerError):
    """Compatibility name for the retired list-backed transport model."""


def _retired_model() -> RetiredInMemoryOwnerError:
    return RetiredInMemoryOwnerError(
        "the process-local EAAEF-093 owner model is retired; bind the exact "
        "READY QuackStateServer owner instead",
        reason_code="in_memory_owner_retired",
    )


def issue_envelope(**_kwargs: Any) -> dict[str, Any]:
    """Refuse the retired content-hash envelope issuer.

    Signed operational commands are issued by the independently admitted
    command-authorizer path, not by an owner helper.
    """

    raise _retired_model()


def verify_envelope(_envelope: object) -> dict[str, Any]:
    """Refuse the retired content-hash envelope verifier."""

    raise _retired_model()


class BoundedQuackTransport:
    """Compatibility tombstone for the retired list-backed transport."""

    INTERFACE: Final[str] = TRANSPORT_INTERFACE
    SCHEMA: Final[str] = TRANSPORT_SCHEMA

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise _retired_model()


class TransportSession:
    """Compatibility tombstone for retired process-local sessions."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise _retired_model()


@dataclass(frozen=True, slots=True)
class OwnerLease:
    """Public, token-free binding to one live owner generation."""

    board_namespace: str
    server_id: str
    store_id: str
    database_uuid: str
    generation: int
    fence_epoch: int
    secret_handle: str
    listen_uri: str
    shard_id: str

    @property
    def owner_id(self) -> str:
        return self.server_id

    @property
    def epoch(self) -> int:
        return self.generation

    @property
    def fence(self) -> int:
        return self.fence_epoch

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": OWNER_LEASE_SCHEMA,
                "interface": OWNER_LEASE_INTERFACE,
                "board_namespace": self.board_namespace,
                "server_id": self.server_id,
                "store_id": self.store_id,
                "database_uuid": self.database_uuid,
                "generation": self.generation,
                "fence_epoch": self.fence_epoch,
                "secret_handle": self.secret_handle,
                "listen_uri": self.listen_uri,
                "shard_id": self.shard_id,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))


def _same_owner_store(previous: OwnerLease, current: OwnerLease) -> bool:
    return (
        current.board_namespace == previous.board_namespace
        and current.shard_id == previous.shard_id
        and current.store_id == previous.store_id
        and current.database_uuid == previous.database_uuid
    )


def owner_lease_is_current(lease: OwnerLease, current: OwnerLease) -> bool:
    """Return whether ``lease`` is the exact live owner generation."""

    return (
        isinstance(lease, OwnerLease)
        and isinstance(current, OwnerLease)
        and lease == current
    )


def assert_owner_restart_successor(
    previous: OwnerLease,
    current: OwnerLease,
    *,
    worker_assertion: bool = False,
) -> OwnerLease:
    """Admit only a later generation and fence of the same owner store.

    ``worker_assertion`` is diagnostic only and never authorizes an invalid
    failover or a stale owner lease.
    """

    del worker_assertion
    if (
        not isinstance(previous, OwnerLease)
        or not isinstance(current, OwnerLease)
        or not _same_owner_store(previous, current)
        or current.generation <= previous.generation
        or current.fence_epoch <= previous.fence_epoch
        or current.server_id == previous.server_id
    ):
        raise StaleOwnerError(
            "replacement is not a later generation of the same owner store",
            reason_code="invalid_failover",
        )
    return current


@dataclass(frozen=True, slots=True)
class OwnerLossObservation:
    """Operational record that the exclusive owner was lost or went stale."""

    kind: OwnerLossKind
    outcome: OwnerRecoveryOutcome = OwnerRecoveryOutcome.OWNER_LOST
    previous: OwnerLease | None = None
    current: OwnerLease | None = None
    reason_code: str = "owner_not_ready"
    worker_assertion: bool = False
    authorizes_completion: bool = False
    worker_assertion_is_authority: bool = False
    schema: str = OWNER_LOSS_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OwnerLossKind):
            object.__setattr__(self, "kind", OwnerLossKind(str(self.kind)))
        if not isinstance(self.outcome, OwnerRecoveryOutcome):
            object.__setattr__(
                self, "outcome", OwnerRecoveryOutcome(str(self.outcome))
            )
        if self.previous is not None and not isinstance(self.previous, OwnerLease):
            raise ExternalQuackOwnerError(
                "owner-loss previous lease is not the exact typed binding",
                reason_code="stale_owner",
            )
        if self.current is not None and not isinstance(self.current, OwnerLease):
            raise ExternalQuackOwnerError(
                "owner-loss current lease is not the exact typed binding",
                reason_code="stale_owner",
            )
        object.__setattr__(self, "reason_code", str(self.reason_code or "").strip())
        object.__setattr__(self, "worker_assertion", bool(self.worker_assertion))
        object.__setattr__(self, "authorizes_completion", False)
        object.__setattr__(self, "worker_assertion_is_authority", False)
        if self.schema != OWNER_LOSS_OBSERVATION_SCHEMA:
            raise ExternalQuackOwnerError(
                "unsupported owner-loss observation schema",
                reason_code="malformed_binding",
            )

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "interface": OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE,
                "binding": OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING,
                "consumes": list(OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES),
                "carrier": EXTERNAL_QUACK_OWNER_INTERFACE,
                "kind": self.kind.value,
                "outcome": self.outcome.value,
                "previous_lease_cid": None
                if self.previous is None
                else self.previous.content_id,
                "current_lease_cid": None
                if self.current is None
                else self.current.content_id,
                "reason_code": self.reason_code,
                "worker_assertion": self.worker_assertion,
                "authorizes_completion": False,
                "worker_assertion_is_authority": False,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))


@dataclass(frozen=True, slots=True)
class OwnerRestartRecovery:
    """Operational record that a later owner generation succeeded a lost owner."""

    previous: OwnerLease
    current: OwnerLease
    outcome: OwnerRecoveryOutcome = OwnerRecoveryOutcome.SUCCESSOR_ADMITTED
    worker_assertion: bool = False
    authorizes_completion: bool = False
    worker_assertion_is_authority: bool = False
    schema: str = OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA

    def __post_init__(self) -> None:
        admitted = assert_owner_restart_successor(self.previous, self.current)
        object.__setattr__(self, "current", admitted)
        if not isinstance(self.outcome, OwnerRecoveryOutcome):
            object.__setattr__(
                self, "outcome", OwnerRecoveryOutcome(str(self.outcome))
            )
        if self.outcome is not OwnerRecoveryOutcome.SUCCESSOR_ADMITTED:
            raise StaleOwnerError(
                "owner-restart recovery is only recorded for an admitted successor",
                reason_code="invalid_failover",
            )
        object.__setattr__(self, "worker_assertion", bool(self.worker_assertion))
        object.__setattr__(self, "authorizes_completion", False)
        object.__setattr__(self, "worker_assertion_is_authority", False)
        if self.schema != OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA:
            raise ExternalQuackOwnerError(
                "unsupported owner-restart recovery schema",
                reason_code="malformed_binding",
            )

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "interface": OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE,
                "binding": OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING,
                "consumes": list(OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES),
                "carrier": EXTERNAL_QUACK_OWNER_INTERFACE,
                "outcome": self.outcome.value,
                "previous_lease_cid": self.previous.content_id,
                "current_lease_cid": self.current.content_id,
                "previous_generation": self.previous.generation,
                "current_generation": self.current.generation,
                "previous_fence_epoch": self.previous.fence_epoch,
                "current_fence_epoch": self.current.fence_epoch,
                "previous_server_id": self.previous.server_id,
                "current_server_id": self.current.server_id,
                "worker_assertion": self.worker_assertion,
                "authorizes_completion": False,
                "worker_assertion_is_authority": False,
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))


def detect_owner_loss(
    *,
    lease: OwnerLease | None = None,
    current: OwnerLease | None = None,
    ready: bool = True,
    held: bool = True,
    lifecycle: str = "ready",
    worker_assertion: bool = False,
) -> OwnerLossObservation | None:
    """Return an observation when the exclusive owner is lost or stale.

    ``worker_assertion`` never conceals owner loss or a stale generation.
    """

    lifecycle_name = str(lifecycle or "").strip().casefold()
    if lifecycle_name in _STOPPED_OWNER_LIFECYCLES:
        kind = OwnerLossKind.PROCESS_STOPPED
        reason = "process_stopped"
    elif not ready:
        kind = OwnerLossKind.NOT_READY
        reason = "owner_not_ready"
    elif not held:
        kind = OwnerLossKind.LOST_HOLD
        reason = "lost_hold"
    elif (
        isinstance(lease, OwnerLease)
        and isinstance(current, OwnerLease)
        and lease != current
    ):
        if lease.generation != current.generation:
            kind = OwnerLossKind.STALE_GENERATION
            reason = "stale_owner"
        else:
            kind = OwnerLossKind.STALE_FENCE
            reason = "stale_owner"
    else:
        return None
    return OwnerLossObservation(
        kind=kind,
        outcome=OwnerRecoveryOutcome.OWNER_LOST,
        previous=lease if isinstance(lease, OwnerLease) else None,
        current=current if isinstance(current, OwnerLease) else None,
        reason_code=reason,
        worker_assertion=bool(worker_assertion),
    )


def recover_from_owner_restart(
    previous: OwnerLease,
    current: OwnerLease,
    *,
    worker_assertion: bool = False,
) -> OwnerRestartRecovery:
    """Record that a later owner generation succeeded the lost owner."""

    admitted = assert_owner_restart_successor(
        previous, current, worker_assertion=worker_assertion
    )
    return OwnerRestartRecovery(
        previous=previous,
        current=admitted,
        outcome=OwnerRecoveryOutcome.SUCCESSOR_ADMITTED,
        worker_assertion=bool(worker_assertion),
    )


def assert_stale_owner_cannot_complete(
    snapshot: TaskStateSnapshot,
    *,
    lease: OwnerLease,
    current: OwnerLease | None,
    ready: bool = True,
    held: bool = True,
    lifecycle: str = "ready",
    worker_assertion: bool = False,
    live_snapshot: TaskStateSnapshot | None = None,
) -> OwnerLease:
    """Reject completion unless the live owner generation is current.

    Consumes ``CanonicalTaskStateMachine@1``.  This does not terminalize a
    task: a worker or model assertion cannot complete against a lost owner
    or a stale owner generation.
    """

    if not isinstance(snapshot, TaskStateSnapshot):
        raise ExternalQuackOwnerError(
            "owner-loss completion requires a canonical TaskStateSnapshot",
            reason_code="malformed_binding",
        )
    if snapshot.INTERFACE != CANONICAL_TASK_STATE_MACHINE_INTERFACE:
        raise ExternalQuackOwnerError(
            "owner-loss completion requires CanonicalTaskStateMachine@1",
            reason_code="malformed_binding",
        )
    loss = detect_owner_loss(
        lease=lease,
        current=current,
        ready=ready,
        held=held,
        lifecycle=lifecycle,
        worker_assertion=worker_assertion,
    )
    if loss is not None and loss.kind in {
        OwnerLossKind.NOT_READY,
        OwnerLossKind.LOST_HOLD,
        OwnerLossKind.PROCESS_STOPPED,
    }:
        raise ExternalQuackOwnerNotReady(
            "lost exclusive owner cannot complete a task",
            reason_code=loss.reason_code,
        )
    if not isinstance(lease, OwnerLease) or not isinstance(current, OwnerLease):
        raise StaleOwnerError(
            "owner lease is not the exact typed binding",
            reason_code="stale_owner",
        )
    if lease != current:
        raise StaleOwnerError(
            "stale owner generation or fence cannot complete a task",
            reason_code="stale_owner",
        )
    live = snapshot if live_snapshot is None else live_snapshot
    if not isinstance(live, TaskStateSnapshot) or not snapshot.may_complete_against(
        live
    ):
        raise ExternalQuackOwnerError(
            "worker or model assertion cannot complete a task without a "
            "current owner generation",
            reason_code="worker_assertion_insufficient",
        )
    return current


class ExternalQuackOwner:
    """Resource-free facade bound to one exact READY ``QuackStateServer``."""

    INTERFACE: Final[str] = EXTERNAL_QUACK_OWNER_INTERFACE
    SCHEMA: Final[str] = EXTERNAL_QUACK_OWNER_SCHEMA
    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING: Final[str] = (
        OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
    )
    CONSUMES_TASK_STATE_MACHINE: Final[str] = CANONICAL_TASK_STATE_MACHINE_INTERFACE

    __slots__ = (
        "_board_namespace",
        "_identity",
        "_owner_server",
        "_shard_id",
    )

    def __init__(
        self,
        owner_server: QuackStateServer | object,
        *,
        shard_id: str = "",
        board_namespace: str = "",
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _FACADE_CONSTRUCTION_TOKEN:
            raise TypeError(
                "ExternalQuackOwner is issued only by a READY QuackStateServer"
            )
        identity = getattr(owner_server, "identity", None)
        if identity is None:
            raise ExternalQuackOwnerNotReady(
                "external owner facade requires a READY state owner",
                reason_code="owner_not_ready",
            )
        self._owner_server = owner_server
        self._identity = identity
        self._board_namespace = str(board_namespace or "").strip()
        self._shard_id = str(shard_id or "").strip()
        if not self._board_namespace or not self._shard_id:
            raise ExternalQuackOwnerError(
                "board_namespace and shard_id are required",
                reason_code="malformed_binding",
            )
        self._require_current_identity()

    def _require_current_identity(self) -> StateServerIdentity:
        from .quack_state_server import ServerLifecycle

        server = self._owner_server
        identity = getattr(server, "identity", None)
        owner = getattr(server, "_owner", None)
        connection = getattr(server, "_connection", None)
        if (
            getattr(server, "lifecycle", None) is not ServerLifecycle.READY
            or identity is None
            or identity is not self._identity
            or connection is None
            or owner is None
            or not getattr(owner, "held", False)
            or getattr(owner, "fence_token", "") == ""
        ):
            raise ExternalQuackOwnerNotReady(
                "backing QuackStateServer is not the exact READY exclusive owner",
                reason_code="owner_not_ready",
            )
        return identity

    @property
    def owner_id(self) -> str:
        return self._identity.server_id

    @property
    def board_namespace(self) -> str:
        return self._board_namespace

    @property
    def shard_id(self) -> str:
        return self._shard_id

    @property
    def epoch(self) -> int:
        return int(self._identity.generation)

    @property
    def fence(self) -> int:
        return int(self._identity.fence_epoch)

    @property
    def listen_uri(self) -> str:
        return self._identity.listen_uri

    @property
    def bound_port(self) -> int:
        return int(self.listen_uri.rsplit(":", 1)[1])

    @property
    def operational_table_exposed(self) -> bool:
        return False

    @property
    def production_admitted(self) -> bool:
        return False

    def lease(self) -> OwnerLease:
        identity = self._require_current_identity()
        return OwnerLease(
            board_namespace=self._board_namespace,
            server_id=identity.server_id,
            store_id=identity.store_id,
            database_uuid=identity.database_uuid,
            generation=int(identity.generation),
            fence_epoch=int(identity.fence_epoch),
            secret_handle=identity.secret_handle,
            listen_uri=identity.listen_uri,
            shard_id=self._shard_id,
        )

    def assert_current(
        self, lease: OwnerLease, *, worker_assertion: bool = False
    ) -> OwnerLease:
        if not isinstance(lease, OwnerLease):
            raise StaleOwnerError(
                "owner lease is not the exact typed binding",
                reason_code="stale_owner",
            )
        current = self.lease()
        if current != lease:
            raise StaleOwnerError(
                "stale owner generation or fence rejected",
                reason_code="stale_owner",
            )
        del worker_assertion
        return current

    def assert_successor(
        self, previous: OwnerLease, *, worker_assertion: bool = False
    ) -> OwnerLease:
        current = self.lease()
        return assert_owner_restart_successor(
            previous, current, worker_assertion=worker_assertion
        )

    def observe_owner_loss(
        self,
        lease: OwnerLease | None = None,
        *,
        worker_assertion: bool = False,
    ) -> OwnerLossObservation:
        """Classify exclusive-owner loss against this facade's live identity."""

        server = self._owner_server
        lifecycle = str(getattr(getattr(server, "lifecycle", None), "value", "") or "")
        owner = getattr(server, "_owner", None)
        held = bool(getattr(owner, "held", False)) if owner is not None else False
        try:
            current = self.lease()
            ready = True
            held = True
        except ExternalQuackOwnerNotReady:
            current = None
            ready = False
        observation = detect_owner_loss(
            lease=lease,
            current=current,
            ready=ready,
            held=held,
            lifecycle=lifecycle,
            worker_assertion=worker_assertion,
        )
        if observation is None:
            raise ExternalQuackOwnerError(
                "exclusive owner is current; no owner-loss to recover",
                reason_code="owner_not_lost",
            )
        return observation

    def recover_owner_restart(
        self, previous: OwnerLease, *, worker_assertion: bool = False
    ) -> OwnerRestartRecovery:
        """Admit this facade as the later generation of a lost owner."""

        current = self.assert_successor(previous, worker_assertion=worker_assertion)
        return recover_from_owner_restart(
            previous, current, worker_assertion=worker_assertion
        )

    def assert_task_may_complete(
        self,
        snapshot: TaskStateSnapshot,
        *,
        lease: OwnerLease | None = None,
        current: TaskStateSnapshot | None = None,
        worker_assertion: bool = False,
    ) -> OwnerLease:
        """Reject completion when the exclusive owner is lost or stale.

        This method does not terminalize the task.
        """

        try:
            live_lease = self.lease()
            ready = True
            held = True
            lifecycle = "ready"
        except ExternalQuackOwnerNotReady:
            live_lease = None
            ready = False
            held = False
            server = self._owner_server
            lifecycle = str(
                getattr(getattr(server, "lifecycle", None), "value", "") or "stopped"
            )
        bound = lease if lease is not None else live_lease
        if bound is None:
            raise ExternalQuackOwnerNotReady(
                "lost exclusive owner cannot complete a task",
                reason_code="owner_not_ready",
            )
        return assert_stale_owner_cannot_complete(
            snapshot,
            lease=bound,
            current=live_lease,
            ready=ready,
            held=held,
            lifecycle=lifecycle,
            worker_assertion=worker_assertion,
            live_snapshot=current,
        )

    def require_operation(self, operation: str) -> None:
        """Reject SQL and every still-unqualified generic daemon operation."""

        name = str(operation or "").strip()
        lowered = name.casefold()
        if (
            name not in REQUIRED_QUACK_DAEMON_OPERATIONS
            and any(marker in lowered for marker in _SQL_OPERATION_MARKERS)
        ):
            raise RemoteSqlRefusedError(
                "remote UPDATE and arbitrary SQL are outside the owner gateway",
                reason_code="remote_sql_refused",
            )
        if name not in REQUIRED_QUACK_DAEMON_OPERATIONS:
            raise QuackDaemonGatewayError(
                "operation is outside the closed 39-operation daemon vocabulary"
            )
        disposition = quack_daemon_owner_operation_dispositions()[name]
        reason = str(disposition.get("reason_code") or "")
        raise QuackDaemonGatewayError(
            f"{EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER}: operation={name};"
            f"reason_code={reason or 'owner_dispatcher_unavailable'}"
        )

    def daemon_gateway(self) -> None:
        """Fail closed until one signed owner-dispatch capability is admitted.

        ``QuackStateServer`` currently owns ``TypedStateOwnerGateway``.  It
        cannot truthfully synthesize the separate signed host capability and
        complete 39-operation dispatcher required by
        ``QuackDaemonCommandGateway@1``.  Returning a structural lookalike here
        would recreate the fake authority that this facade retires.
        """

        self._require_current_identity()
        raise QuackDaemonGatewayError(
            f"{EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER}: the real owner cannot "
            "self-issue the missing signed dispatcher/host admission"
        )

    def evidence(self) -> Mapping[str, Any]:
        lease = self.lease()
        return MappingProxyType(
            {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "qualification_status": EXTERNAL_QUACK_OWNER_QUALIFICATION_STATUS,
                "backing_owner_interface": "QuackStateServer@1",
                "lease_cid": lease.content_id,
                "board_namespace": lease.board_namespace,
                "shard_id": lease.shard_id,
                "server_id": lease.server_id,
                "store_id": lease.store_id,
                "owner_generation": lease.generation,
                "fence_epoch": lease.fence_epoch,
                "listen_uri": lease.listen_uri,
                "opens_database": False,
                "creates_dispatcher": False,
                "local_sidecar_writes": False,
                "direct_task_source": False,
                "arbitrary_sql_enabled": False,
                "production_admitted": False,
                "canonical_owner_handler_qualification_status": (
                    QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS
                ),
                "production_blockers": [
                    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER
                ],
                "owner_loss_and_owner_restart_recovery_binding": (
                    OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING
                ),
            }
        )


def _bind_external_quack_owner(
    *,
    owner_server: QuackStateServer,
    board_namespace: str,
    shard_id: str,
) -> ExternalQuackOwner:
    """Construct the facade only from ``QuackStateServer`` owner code."""

    return ExternalQuackOwner(
        owner_server,
        board_namespace=board_namespace,
        shard_id=shard_id,
        _construction_token=_FACADE_CONSTRUCTION_TOKEN,
    )


__all__ = (
    "ALLOWED_OPERATIONS",
    "APPLY_RECEIPT_SCHEMA",
    "BoundedQuackTransport",
    "CONTRACT_VERSION",
    "DuplicateOwnerError",
    "ENVELOPE_INTERFACE",
    "ENVELOPE_SCHEMA",
    "EXTERNAL_QUACK_OWNER_INTERFACE",
    "EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER",
    "EXTERNAL_QUACK_OWNER_QUALIFICATION_STATUS",
    "EXTERNAL_QUACK_OWNER_SCHEMA",
    "ExternalQuackOwner",
    "ExternalQuackOwnerError",
    "ExternalQuackOwnerNotReady",
    "INITIAL_EPOCH",
    "INITIAL_FENCE",
    "LIVE_QUACK_PORT",
    "OWNER_LEASE_INTERFACE",
    "OWNER_LEASE_SCHEMA",
    "OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_BINDING",
    "OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_CONSUMES",
    "OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_INTERFACE",
    "OWNER_LOSS_AND_OWNER_RESTART_RECOVERY_SCHEMA",
    "OWNER_LOSS_OBSERVATION_SCHEMA",
    "OwnerLease",
    "OwnerLossKind",
    "OwnerLossObservation",
    "OwnerRecoveryOutcome",
    "OwnerRestartRecovery",
    "REMOTE_CAPABILITIES",
    "RemoteSqlRefusedError",
    "RetiredInMemoryOwnerError",
    "StaleOwnerError",
    "TRANSPORT_INTERFACE",
    "TRANSPORT_SCHEMA",
    "TransportAuthError",
    "TransportSession",
    "UnsignedEnvelopeError",
    "assert_owner_restart_successor",
    "assert_stale_owner_cannot_complete",
    "detect_owner_loss",
    "issue_envelope",
    "owner_lease_is_current",
    "recover_from_owner_restart",
    "verify_envelope",
)
