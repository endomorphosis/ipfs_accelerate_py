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
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from ..task_sources.control_plane_contracts import content_identity
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


class ExternalQuackOwner:
    """Resource-free facade bound to one exact READY ``QuackStateServer``."""

    INTERFACE: Final[str] = EXTERNAL_QUACK_OWNER_INTERFACE
    SCHEMA: Final[str] = EXTERNAL_QUACK_OWNER_SCHEMA

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

    def assert_current(self, lease: OwnerLease) -> OwnerLease:
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
        return current

    def assert_successor(self, previous: OwnerLease) -> OwnerLease:
        current = self.lease()
        if (
            not isinstance(previous, OwnerLease)
            or current.board_namespace != previous.board_namespace
            or current.shard_id != previous.shard_id
            or current.store_id != previous.store_id
            or current.database_uuid != previous.database_uuid
            or current.generation <= previous.generation
            or current.fence_epoch <= previous.fence_epoch
            or current.server_id == previous.server_id
        ):
            raise StaleOwnerError(
                "replacement is not a later generation of the same owner store",
                reason_code="invalid_failover",
            )
        return current

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
    "OwnerLease",
    "REMOTE_CAPABILITIES",
    "RemoteSqlRefusedError",
    "RetiredInMemoryOwnerError",
    "StaleOwnerError",
    "TRANSPORT_INTERFACE",
    "TRANSPORT_SCHEMA",
    "TransportAuthError",
    "TransportSession",
    "UnsignedEnvelopeError",
    "issue_envelope",
    "verify_envelope",
)
