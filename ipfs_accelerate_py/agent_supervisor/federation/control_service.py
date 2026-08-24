"""Closed, typed control boundary for an admitted federation.

Federation creation remains the responsibility of :mod:`.trigger`, because it
has a distinct authentication and budget-admission contract.  This module is
the corresponding boundary for every *post-admission* control command.  It
does not open DuckDB, interpret SQL, select a filesystem location, or fall
back from Quack to an embedded store.  A concrete state owner must implement
the atomic command/outbox/audit transaction behind this narrow interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import ClassVar, Final, Mapping, Protocol

from .contracts import (
    ClosedContract,
    FederationAuthorityError,
    FederationCommand,
    FederationCommandResult,
    FederationContractError,
    FederationOperation,
    _identifier,
    _integer,
    _timestamp,
)


FEDERATION_CONTROL_SERVICE_INTERFACE: Final[str] = "FederationControlService@1"
FEDERATION_CONTROL_SERVICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/control-service@1"
)
FEDERATION_CONTROL_AUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/control-authorization@1"
)
FEDERATION_CONTROL_AUDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/control-audit@1"
)
# ``CREATE`` is intentionally absent.  Accepting it here would bypass the
# authenticated trigger's delegation, root resolution, and budget reservation
# transaction.  The remaining closed enum values are all post-admission
# mutations and are handled by the owner as one typed command transaction.
POST_ADMISSION_OPERATIONS: Final[frozenset[FederationOperation]] = frozenset(
    set(FederationOperation) - {FederationOperation.CREATE}
)


class FederationControlServiceError(FederationAuthorityError):
    """Base error for rejected federation control requests."""


class FederationControlCapabilityError(FederationControlServiceError):
    """The exclusive typed state-owner capability is absent or unqualified."""


class FederationControlAuthorizationError(FederationControlServiceError):
    """A server-side authorization did not bind the submitted command."""


class FederationControlStaleError(FederationControlServiceError):
    """The request was made against stale generation, lease, or fence state."""


class FederationControlResultError(FederationControlServiceError):
    """A state owner returned a result outside the command's closed contract."""


@dataclass(frozen=True)
class FederationControlCapability:
    """Positive evidence for the only permitted live control path.

    This is deliberately an all-positive contract: omitted evidence is not a
    compatibility mode.  DuckLake may be present as history projection, but
    it can never be the source of this capability or command authority.
    """

    state_owner_available: bool
    interface: str
    exclusive_state_owner: bool
    quack_transport: bool
    typed_operations: bool
    direct_duckdb_access: bool
    ducklake_authoritative: bool

    def __post_init__(self) -> None:
        if self.state_owner_available is not True:
            raise FederationControlCapabilityError("typed state owner is unavailable")
        if self.interface != "TypedStateOwnerFederationControl@1":
            raise FederationControlCapabilityError("state-owner interface is unqualified")
        if self.exclusive_state_owner is not True:
            raise FederationControlCapabilityError("state owner is not exclusive")
        if self.quack_transport is not True:
            raise FederationControlCapabilityError("control requires Quack transport")
        if self.typed_operations is not True:
            raise FederationControlCapabilityError("state owner lacks typed operations")
        if self.direct_duckdb_access is not False:
            raise FederationControlCapabilityError("direct DuckDB access is prohibited")
        if self.ducklake_authoritative is not False:
            raise FederationControlCapabilityError("DuckLake cannot control a federation")


def qualified_federation_control_capability() -> FederationControlCapability:
    """Return hermetic capability evidence for a qualified typed state owner."""

    return FederationControlCapability(
        state_owner_available=True,
        interface="TypedStateOwnerFederationControl@1",
        exclusive_state_owner=True,
        quack_transport=True,
        typed_operations=True,
        direct_duckdb_access=False,
        ducklake_authoritative=False,
    )


@dataclass(frozen=True)
class FederationControlAuthorization(ClosedContract):
    """Server-issued authorization bound to one command CID and current fence."""

    SCHEMA: ClassVar[str] = FEDERATION_CONTROL_AUTHORIZATION_SCHEMA

    authorization_id: str
    command_cid: str
    caller_did: str
    tenant_id: str
    operation: FederationOperation
    target_id: str
    policy_ref: str
    policy_revision: int
    control_plane_generation: int
    fencing_epoch: int
    lease_id: str
    expires_at: str
    decided_at: str

    FIELD_DECODERS: ClassVar[Mapping[str, object]] = MappingProxyType(
        {"operation": FederationOperation}
    )

    def __post_init__(self) -> None:
        for field in (
            "authorization_id",
            "command_cid",
            "caller_did",
            "tenant_id",
            "target_id",
            "policy_ref",
            "lease_id",
        ):
            _identifier(getattr(self, field), field)
        if not isinstance(self.operation, FederationOperation):
            raise FederationContractError("control authorization operation is not closed")
        _integer(self.policy_revision, "policy_revision", minimum=1)
        _integer(self.control_plane_generation, "control_plane_generation", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        expires = _timestamp(self.expires_at, "expires_at")
        decided = _timestamp(self.decided_at, "decided_at")
        if _as_epoch(decided) > _as_epoch(expires):
            raise FederationControlAuthorizationError(
                "control authorization was decided after it expired"
            )


@dataclass(frozen=True)
class FederationControlAuditReceipt(ClosedContract):
    """Compact client-visible evidence of an owner-side command transaction."""

    SCHEMA: ClassVar[str] = FEDERATION_CONTROL_AUDIT_SCHEMA

    audit_id: str
    command_cid: str
    authorization_id: str
    result_ref: str
    outcome: str
    control_plane_generation: int
    fencing_epoch: int
    recorded_at: str

    def __post_init__(self) -> None:
        for field in ("audit_id", "command_cid", "authorization_id", "result_ref"):
            _identifier(getattr(self, field), field)
        if self.outcome not in {"applied", "dry_run", "failed", "rejected"}:
            raise FederationControlResultError("audit outcome is outside the closed vocabulary")
        _integer(self.control_plane_generation, "control_plane_generation", minimum=1)
        _integer(self.fencing_epoch, "fencing_epoch", minimum=1)
        _timestamp(self.recorded_at, "recorded_at")


@dataclass(frozen=True)
class FederationControlResponse:
    """The typed command result together with its mandatory audit evidence."""

    result: FederationCommandResult
    audit: FederationControlAuditReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.result, FederationCommandResult):
            raise FederationControlResultError("response result must be FederationCommandResult")
        if not isinstance(self.audit, FederationControlAuditReceipt):
            raise FederationControlResultError("response audit must be FederationControlAuditReceipt")


class FederationCommandAuthorizer(Protocol):
    """Server-side policy/delegation/lease authority; never a caller payload."""

    def authorize(self, command: FederationCommand) -> FederationControlAuthorization: ...


class FederationControlStateOwner(Protocol):
    """Atomic owner operation for state, event, outbox, idempotency, and audit."""

    def execute_federation_command(
        self,
        command: FederationCommand,
        authorization: FederationControlAuthorization,
    ) -> FederationControlResponse: ...


def _as_epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class FederationControlService:
    """Authenticate the live control path before delegating one typed command.

    The service intentionally has no fallback backend.  It performs no direct
    persistence: the state owner owns CAS/idempotency, generation, leases,
    fences, events, outbox rows and audit storage atomically.
    """

    INTERFACE: Final[str] = FEDERATION_CONTROL_SERVICE_INTERFACE

    def __init__(
        self,
        *,
        authorizer: FederationCommandAuthorizer,
        state_owner: FederationControlStateOwner,
        capability: FederationControlCapability,
        now: Callable[[], float] = lambda: datetime.now(timezone.utc).timestamp(),
    ) -> None:
        if not callable(getattr(authorizer, "authorize", None)):
            raise FederationControlCapabilityError("control authorizer is unavailable")
        if not callable(getattr(state_owner, "execute_federation_command", None)):
            raise FederationControlCapabilityError(
                "typed federation state-owner operation is unavailable"
            )
        if not isinstance(capability, FederationControlCapability):
            raise FederationControlCapabilityError("control capability must be typed")
        # Re-run the positive capability checks in case a malformed object was
        # constructed through an unsafe deserializer.
        capability.__post_init__()
        self._authorizer = authorizer
        self._state_owner = state_owner
        self._capability = capability
        self._now = now

    @property
    def capability(self) -> FederationControlCapability:
        return self._capability

    def execute(self, command: FederationCommand) -> FederationControlResponse:
        """Execute one authorized post-admission command through the owner."""

        # Capability evidence is a live fail-closed gate, not merely a
        # construction-time assertion.  Rechecking it prevents a service
        # whose owner/transport evidence has been revoked (or unsafely
        # mutated by an embedding process) from dispatching another command.
        self._capability.__post_init__()
        if not isinstance(command, FederationCommand):
            raise FederationControlServiceError("control service accepts FederationCommand only")
        self._validate_command(command)
        authorization = self._authorizer.authorize(command)
        if not isinstance(authorization, FederationControlAuthorization):
            raise FederationControlAuthorizationError(
                "authorizer returned no typed federation control authorization"
            )
        self._validate_authorization(command, authorization)
        response = self._state_owner.execute_federation_command(command, authorization)
        self._validate_response(command, authorization, response)
        return response

    # ``dispatch`` is the transport-neutral name used by direct adapters.
    dispatch = execute

    def _validate_command(self, command: FederationCommand) -> None:
        if command.operation not in POST_ADMISSION_OPERATIONS:
            raise FederationControlAuthorizationError(
                "federation.create is accepted only by FederationControlGateway"
            )
        if not command.target_id.startswith("federation:"):
            raise FederationControlServiceError("command target must be a federation identity")
        if command.expected_revision < 1:
            raise FederationControlStaleError("post-admission command requires a revision")
        if command.expected_generation != command.binding.control_plane_generation:
            raise FederationControlStaleError(
                "command generation differs from its authority binding"
            )
        if not command.dry_run and not command.expected_effects:
            raise FederationControlServiceError(
                "live control command must declare bounded expected effects"
            )
        if self._now() > _as_epoch(command.binding.expires_at):
            raise FederationControlAuthorizationError("federation authority has expired")

    def _validate_authorization(
        self,
        command: FederationCommand,
        authorization: FederationControlAuthorization,
    ) -> None:
        if authorization.command_cid != command.cid:
            raise FederationControlAuthorizationError("authorization binds another command")
        if authorization.operation is not command.operation:
            raise FederationControlAuthorizationError("authorization operation differs")
        if authorization.target_id != command.target_id:
            raise FederationControlAuthorizationError("authorization target differs")
        binding = command.binding
        if (
            authorization.tenant_id != binding.tenant_id
            or authorization.policy_ref != binding.policy_ref
            or authorization.policy_revision != binding.policy_revision
        ):
            raise FederationControlAuthorizationError("authorization policy scope differs")
        if authorization.control_plane_generation != command.expected_generation:
            raise FederationControlStaleError("authorization generation differs")
        if authorization.fencing_epoch != command.expected_fencing_epoch:
            raise FederationControlStaleError("authorization fence differs")
        if _as_epoch(authorization.expires_at) > _as_epoch(binding.expires_at):
            raise FederationControlAuthorizationError(
                "authorization outlives the federation authority"
            )
        if self._now() > _as_epoch(authorization.expires_at):
            raise FederationControlAuthorizationError("control authorization has expired")

    @staticmethod
    def _validate_response(
        command: FederationCommand,
        authorization: FederationControlAuthorization,
        response: FederationControlResponse,
    ) -> None:
        if not isinstance(response, FederationControlResponse):
            raise FederationControlResultError("state owner returned no typed control response")
        result = response.result
        audit = response.audit
        if result.binding != command.binding:
            raise FederationControlResultError("result binding differs from command")
        if result.revision < command.expected_revision:
            raise FederationControlResultError("result revision predates command")
        if command.dry_run:
            if result.outcome != "dry_run" or audit.outcome != "dry_run":
                raise FederationControlResultError("dry-run command reported a live outcome")
        elif (
            result.outcome not in {"applied", "failed", "rejected"}
            or audit.outcome != result.outcome
        ):
            # A concurrent authoritative CAS/lease check can reject a command
            # after its pre-dispatch authorization.  That is a valid
            # fail-closed outcome, provided the owner returns matching typed
            # evidence instead of claiming application.
            raise FederationControlResultError("live command has an invalid owner outcome")
        if command.cid not in result.evidence_refs:
            raise FederationControlResultError("result lacks command evidence")
        if audit.command_cid != command.cid:
            raise FederationControlResultError("audit binds another command")
        if audit.authorization_id != authorization.authorization_id:
            raise FederationControlResultError("audit binds another authorization")
        if audit.result_ref != result.cid:
            raise FederationControlResultError("audit result reference differs")
        if audit.control_plane_generation != command.expected_generation:
            raise FederationControlResultError("audit generation differs from command")
        if audit.fencing_epoch != command.expected_fencing_epoch:
            raise FederationControlResultError("audit fence differs")


__all__ = [
    "FEDERATION_CONTROL_AUDIT_SCHEMA",
    "FEDERATION_CONTROL_AUTHORIZATION_SCHEMA",
    "FEDERATION_CONTROL_SERVICE_INTERFACE",
    "FEDERATION_CONTROL_SERVICE_SCHEMA",
    "POST_ADMISSION_OPERATIONS",
    "FederationCommandAuthorizer",
    "FederationControlAuditReceipt",
    "FederationControlAuthorization",
    "FederationControlAuthorizationError",
    "FederationControlCapability",
    "FederationControlCapabilityError",
    "FederationControlResponse",
    "FederationControlResultError",
    "FederationControlService",
    "FederationControlServiceError",
    "FederationControlStaleError",
    "FederationControlStateOwner",
    "qualified_federation_control_capability",
]
