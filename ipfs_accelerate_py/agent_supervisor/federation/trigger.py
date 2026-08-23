"""Authenticated external-agent federation trigger gateway.

The request names immutable repository identities, never a database path or
SQL.  Authentication, delegation, repository/tree resolution, policy, and
budget admission are injected server-side authorities.  A model cannot supply
any of their outcomes.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP017, UP042

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from .contracts import (
    PROGRAM_ID,
    BudgetReservation,
    ClosedContract,
    FederationAuthorityError,
    FederationAuthorizationDecision,
    FederationAuthorizationReason,
    FederationAuthorizationVerdict,
    FederationContractError,
    FederationIdentity,
    FederationOperation,
    FederationPolicy,
    FederationReceipt,
    FederationRequest,
    _identifier,
    _operation,
    _strings,
    _text,
    _timestamp,
)

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/causal-federation"


class AuthenticationAlgorithm(str, Enum):
    HMAC_SHA256 = "HMAC_SHA256"


class FederationTriggerError(FederationAuthorityError):
    """Base fail-closed trigger rejection."""


class AuthenticationRejected(FederationTriggerError):
    """Caller authentication failed."""


class DelegationRejected(FederationTriggerError):
    """Delegation chain validation failed."""


class PolicyRejected(FederationTriggerError):
    """Server-side policy denied the request."""


class RepositoryResolutionRejected(FederationTriggerError):
    """Repository or tree roots did not resolve exactly."""


class BudgetRejected(FederationTriggerError):
    """Hierarchical resource/token reservation was denied."""


class BudgetReconciliationRequired(BudgetRejected):
    """A failed admission left a reservation requiring authoritative recovery."""


@dataclass(frozen=True)
class AuthenticationEvidence(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/authentication-evidence@1"

    evidence_id: str
    caller_did: str
    algorithm: AuthenticationAlgorithm
    key_handle: str
    request_cid: str
    audience: str
    nonce: str
    issued_at: str
    expires_at: str
    signature: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"algorithm": AuthenticationAlgorithm}
    )

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "caller_did",
            "key_handle",
            "request_cid",
            "audience",
            "nonce",
        ):
            _identifier(getattr(self, name), name)
        if not self.key_handle.startswith("handle:"):
            raise FederationContractError("key_handle must be opaque")
        if not isinstance(self.algorithm, AuthenticationAlgorithm):
            raise FederationContractError("authentication algorithm is not closed")
        _timestamp(self.issued_at, "issued_at")
        _timestamp(self.expires_at, "expires_at")
        _text(self.signature, "signature", maximum=512)


@dataclass(frozen=True)
class DelegationGrant(ClosedContract):
    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/delegation-grant@1"

    grant_id: str
    issuer_did: str
    subject_did: str
    parent_grant_ref: str
    audience: str
    operations: tuple[str, ...]
    repository_ids: tuple[str, ...]
    policy_ref: str
    issued_at: str
    expires_at: str
    key_handle: str
    signature: str

    FIELD_DECODERS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"operations": tuple, "repository_ids": tuple}
    )

    def __post_init__(self) -> None:
        for name in (
            "grant_id",
            "issuer_did",
            "subject_did",
            "audience",
            "policy_ref",
            "key_handle",
        ):
            _identifier(getattr(self, name), name)
        _identifier(self.parent_grant_ref, "parent_grant_ref", required=False)
        operations = _strings(
            self.operations,
            "operations",
            maximum=64,
            required=True,
            identities=False,
        )
        for operation in operations:
            _operation(operation, "operations")
        _strings(self.repository_ids, "repository_ids", maximum=256, required=True)
        _timestamp(self.issued_at, "issued_at")
        _timestamp(self.expires_at, "expires_at")
        if not self.key_handle.startswith("handle:"):
            raise FederationContractError("delegation key_handle must be opaque")
        _text(self.signature, "signature", maximum=512)

    def unsigned_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["signature"] = ""
        return payload


@dataclass(frozen=True)
class ResolvedRepository:
    requested_ref: str
    repository_id: str
    tree_id: str
    semantic_state_root: str


class AuthenticationAuthority(Protocol):
    def verify(self, request: FederationRequest, evidence: AuthenticationEvidence) -> None: ...


class DelegationAuthority(Protocol):
    def verify_chain(
        self,
        request: FederationRequest,
        grant_refs: Sequence[str],
    ) -> tuple[DelegationGrant, ...]: ...


class PolicyAuthority(Protocol):
    def get_policy(self, policy_ref: str) -> FederationPolicy: ...


class RepositoryAuthority(Protocol):
    def resolve(self, repository_refs: Sequence[str]) -> tuple[ResolvedRepository, ...]: ...


class BudgetAuthority(Protocol):
    """Server-side, idempotent, expiring reservation authority.

    ``reserve`` must bind the request idempotency key and expiry.  Repeating
    the same admitted request returns the same reservation.  Unconsumed
    reservations expire no later than the request and ``release`` is itself
    idempotent so a failed state-owner transaction cannot leak capacity.
    """

    def reserve(
        self,
        request: FederationRequest,
        policy: FederationPolicy,
    ) -> BudgetReservation: ...

    def release(
        self,
        reservation: BudgetReservation,
        *,
        idempotency_key: str,
        reason: str,
    ) -> None: ...


class FederationAdmissionStore(Protocol):
    """Authoritative state owner for atomic federation admission.

    A successful call must bind the reservation, federation row, initial
    domain event, outbox row, idempotency result, and generation advance in
    one transaction.  A failed call must leave none of those mutations.
    """

    def create_federation(
        self,
        *,
        request: FederationRequest,
        policy: FederationPolicy,
        repositories: Sequence[ResolvedRepository],
        budget_reservation: BudgetReservation,
        authentication_evidence_ref: str,
        authorization_decision: FederationAuthorizationDecision,
    ) -> tuple[FederationIdentity, FederationReceipt]: ...

    def lookup_federation_creation(
        self,
        *,
        idempotency_key: str,
        tenant_id: str,
        federation_id: str,
    ) -> tuple[FederationIdentity, FederationReceipt] | None: ...


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def validated_delegation_chain_identity(
    grant_refs: Sequence[str],
    grants: Sequence[DelegationGrant],
) -> str:
    """Bind the ordered validated grants without persisting their signatures."""

    return content_identity(
        {
            "schema": f"{_SCHEMA_PREFIX}/validated-delegation-chain@1",
            "grant_refs": list(grant_refs),
            "grant_cids": [grant.cid for grant in grants],
        }
    )


def resolved_authorization_scope_identity(
    repositories: Sequence[ResolvedRepository],
    effect_scope: Sequence[str],
) -> str:
    """Bind server-resolved repository/tree roots and the admitted effects."""

    return content_identity(
        {
            "schema": f"{_SCHEMA_PREFIX}/resolved-authorization-scope@1",
            "repositories": [
                {
                    "repository_id": item.repository_id,
                    "tree_id": item.tree_id,
                    "semantic_state_root": item.semantic_state_root,
                }
                for item in repositories
            ],
            "effect_scope": list(effect_scope),
        }
    )


class HmacAuthenticationAuthority:
    """Loopback/server-side HMAC authenticator using opaque key handles."""

    def __init__(
        self,
        key_resolver: Callable[[str, str], bytes],
        *,
        now: Callable[[], float] = lambda: datetime.now(timezone.utc).timestamp(),
    ) -> None:
        self._key_resolver = key_resolver
        self._now = now

    @staticmethod
    def request_message(request: FederationRequest) -> bytes:
        return canonical_json_bytes(request.to_dict())

    @staticmethod
    def sign_request(request: FederationRequest, key: bytes) -> str:
        digest = hmac.new(key, HmacAuthenticationAuthority.request_message(request), hashlib.sha256)
        return base64.urlsafe_b64encode(digest.digest()).decode("ascii").rstrip("=")

    def verify(self, request: FederationRequest, evidence: AuthenticationEvidence) -> None:
        if evidence.caller_did != request.caller_did:
            raise AuthenticationRejected("authentication caller differs")
        if evidence.request_cid != request.cid:
            raise AuthenticationRejected("authentication evidence binds another request")
        if evidence.audience != request.audience or evidence.nonce != request.nonce:
            raise AuthenticationRejected("authentication audience or nonce differs")
        now = self._now()
        if _epoch(evidence.issued_at) > now:
            raise AuthenticationRejected("authentication evidence is not yet valid")
        if now > _epoch(evidence.expires_at) or now > _epoch(request.expiry):
            raise AuthenticationRejected("authentication or request expired")
        if _epoch(request.expiry) > _epoch(evidence.expires_at):
            raise AuthenticationRejected(
                "federation lifetime exceeds authentication authority"
            )
        key = self._key_resolver(evidence.caller_did, evidence.key_handle)
        if not isinstance(key, bytes) or len(key) < 16:
            raise AuthenticationRejected("authentication key handle is unavailable")
        expected = self.sign_request(request, key)
        if not hmac.compare_digest(expected, evidence.signature):
            raise AuthenticationRejected("authentication signature mismatch")


class HmacDelegationAuthority:
    """Finite delegation-chain verifier with server-side grant/key resolvers."""

    def __init__(
        self,
        grant_resolver: Callable[[str], DelegationGrant],
        key_resolver: Callable[[str, str], bytes],
        *,
        now: Callable[[], float] = lambda: datetime.now(timezone.utc).timestamp(),
    ) -> None:
        self._grant_resolver = grant_resolver
        self._key_resolver = key_resolver
        self._now = now

    @staticmethod
    def sign_grant(grant: DelegationGrant, key: bytes) -> str:
        digest = hmac.new(key, canonical_json_bytes(grant.unsigned_dict()), hashlib.sha256)
        return base64.urlsafe_b64encode(digest.digest()).decode("ascii").rstrip("=")

    def verify_chain(
        self,
        request: FederationRequest,
        grant_refs: Sequence[str],
    ) -> tuple[DelegationGrant, ...]:
        if len(grant_refs) > 16:
            raise DelegationRejected("delegation depth exceeds policy bound")
        grants = tuple(self._grant_resolver(ref) for ref in grant_refs)
        if not grants:
            return ()
        seen: set[str] = set()
        previous: DelegationGrant | None = None
        now = self._now()
        for grant in grants:
            if grant.grant_id in seen:
                raise DelegationRejected("delegation chain contains a cycle")
            seen.add(grant.grant_id)
            if _epoch(grant.issued_at) > now:
                raise DelegationRejected("delegation grant is not yet valid")
            if now > _epoch(grant.expires_at):
                raise DelegationRejected("delegation grant expired")
            if _epoch(request.expiry) > _epoch(grant.expires_at):
                raise DelegationRejected(
                    "federation lifetime exceeds delegation authority"
                )
            if grant.audience != request.audience:
                raise DelegationRejected("delegation audience differs")
            if "federation.create" not in grant.operations:
                raise DelegationRejected("delegation does not permit federation.create")
            if not set(request.repository_roots).issubset(grant.repository_ids):
                raise DelegationRejected("delegation repository scope is insufficient")
            if grant.policy_ref != request.policy_ref:
                raise DelegationRejected("delegation policy scope differs")
            if previous is not None:
                if grant.parent_grant_ref != previous.grant_id:
                    raise DelegationRejected("delegation parent link differs")
                if grant.issuer_did != previous.subject_did:
                    raise DelegationRejected("delegation subject/issuer continuity differs")
                if _epoch(grant.expires_at) > _epoch(previous.expires_at):
                    raise DelegationRejected(
                        "child delegation outlives its parent authority"
                    )
            elif grant.parent_grant_ref:
                raise DelegationRejected("first supplied grant has an omitted parent")
            key = self._key_resolver(grant.issuer_did, grant.key_handle)
            if not isinstance(key, bytes) or len(key) < 16:
                raise DelegationRejected("delegation key handle is unavailable")
            expected = self.sign_grant(grant, key)
            if not hmac.compare_digest(expected, grant.signature):
                raise DelegationRejected("delegation signature mismatch")
            previous = grant
        if grants[-1].subject_did != request.caller_did:
            raise DelegationRejected("delegation terminal subject is not the caller")
        return grants


class FederationControlGateway:
    """Canonical typed external trigger path."""

    def __init__(
        self,
        *,
        audience: str,
        authenticator: AuthenticationAuthority,
        delegations: DelegationAuthority,
        policies: PolicyAuthority,
        repositories: RepositoryAuthority,
        budgets: BudgetAuthority,
        store: FederationAdmissionStore,
        now: Callable[[], float] = lambda: datetime.now(timezone.utc).timestamp(),
    ) -> None:
        self._audience = _identifier(audience, "gateway audience")
        self._authenticator = authenticator
        self._delegations = delegations
        self._policies = policies
        self._repositories = repositories
        self._budgets = budgets
        self._store = store
        self._now = now

    def create(
        self,
        request: FederationRequest,
        evidence: AuthenticationEvidence,
    ) -> tuple[FederationIdentity, FederationReceipt]:
        if not isinstance(request, FederationRequest):
            raise FederationTriggerError("gateway accepts only FederationRequest")
        if not isinstance(evidence, AuthenticationEvidence):
            raise FederationTriggerError("gateway requires AuthenticationEvidence")
        if request.program_id != PROGRAM_ID:
            raise PolicyRejected("program identity is not admitted by this gateway")
        if request.audience != self._audience:
            raise AuthenticationRejected("request audience differs from gateway")
        if self._now() > _epoch(request.expiry):
            raise AuthenticationRejected("federation request expired")

        self._authenticator.verify(request, evidence)
        grants = self._delegations.verify_chain(
            request, request.delegation_chain
        )
        # Authorities may be injected through non-HMAC implementations.  The
        # gateway therefore independently enforces the effective lifetime of
        # every typed authentication/delegation record it can observe.
        if _epoch(request.expiry) > _epoch(evidence.expires_at):
            raise AuthenticationRejected(
                "federation lifetime exceeds authentication authority"
            )
        if any(
            _epoch(request.expiry) > _epoch(grant.expires_at)
            for grant in grants
        ):
            raise DelegationRejected(
                "federation lifetime exceeds delegation authority"
            )
        policy = self._policies.get_policy(request.policy_ref)
        if (
            policy.record_id != request.policy_ref
            or policy.revision != request.binding.policy_revision
        ):
            raise PolicyRejected("policy identity or revision differs")
        if request.caller_did not in policy.allowed_callers:
            raise PolicyRejected("caller is not permitted by policy")
        if request.audience not in policy.allowed_audiences:
            raise PolicyRejected("audience is not permitted by policy")
        if "federation.create" not in policy.allowed_operations:
            raise PolicyRejected("policy does not permit federation.create")
        if not set(request.effect_scope).issubset(policy.allowed_effects):
            raise PolicyRejected("requested effect scope exceeds policy")
        if request.maximum_supervisors > policy.maximum_supervisors:
            raise PolicyRejected("requested supervisor count exceeds policy")
        if request.maximum_subagents > policy.maximum_subagents:
            raise PolicyRejected("requested subagent count exceeds policy")

        resolved = self._repositories.resolve(request.repository_roots)
        if len(resolved) != len(request.repository_roots):
            raise RepositoryResolutionRejected("not every repository root resolved")
        if tuple(item.repository_id for item in resolved) != request.binding.repository_ids:
            raise RepositoryResolutionRejected("server repository identities differ")
        if tuple(item.tree_id for item in resolved) != request.binding.repository_tree_ids:
            raise RepositoryResolutionRejected("server tree identities differ")
        if (
            tuple(item.semantic_state_root for item in resolved)
            != request.binding.semantic_state_roots
        ):
            raise RepositoryResolutionRejected("server semantic roots differ")

        decision_time = self._now()
        if decision_time > _epoch(request.expiry):
            raise AuthenticationRejected("federation request expired before admission")
        decided_at = datetime.fromtimestamp(
            decision_time,
            timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        authorization_decision = FederationAuthorizationDecision(
            request_cid=request.cid,
            caller_did=request.caller_did,
            delegation_chain_cid=validated_delegation_chain_identity(
                request.delegation_chain,
                grants,
            ),
            audience=request.audience,
            operation=FederationOperation.CREATE,
            resolved_scope_cid=resolved_authorization_scope_identity(
                resolved,
                request.effect_scope,
            ),
            policy_id=policy.record_id,
            policy_revision=policy.revision,
            verdict=FederationAuthorizationVerdict.ADMITTED,
            reason=(
                FederationAuthorizationReason.AUTHENTICATED_DELEGATED_POLICY_ADMITTED
            ),
            authentication_evidence_cid=evidence.cid,
            expires_at=request.expiry,
            decided_at=decided_at,
        )

        reservation = self._budgets.reserve(request, policy)
        if not isinstance(reservation, BudgetReservation):
            raise BudgetRejected("budget authority returned no typed reservation")
        if (
            reservation.binding != request.binding
            or reservation.owner_id != f"federation:{request.cid}"
            or reservation.request_cid != request.cid
            or reservation.idempotency_key != request.idempotency_key
            or reservation.policy_ref != policy.record_id
            or reservation.policy_revision != policy.revision
            or reservation.resource_budget_ref != request.resource_budget.record_id
            or reservation.token_budget_ref != request.token_budget.record_id
            or reservation.status != "reserved"
        ):
            raise BudgetRejected("budget reservation scope differs from admitted request")
        if self._now() > _epoch(reservation.expires_at):
            raise BudgetRejected("budget reservation expired before admission")
        try:
            return self._store.create_federation(
                request=request,
                policy=policy,
                repositories=resolved,
                budget_reservation=reservation,
                authentication_evidence_ref=evidence.evidence_id,
                authorization_decision=authorization_decision,
            )
        except Exception as admission_error:
            # A transport failure can occur after commit but before the caller
            # receives the result.  Never compensate that ambiguous outcome by
            # releasing capacity: first resolve the exact idempotency record
            # through the authoritative state owner.
            try:
                recovered = self._store.lookup_federation_creation(
                    idempotency_key=request.idempotency_key,
                    tenant_id=request.binding.tenant_id,
                    federation_id=f"federation:{request.cid}",
                )
            except Exception:
                raise BudgetReconciliationRequired(
                    "federation admission outcome is unknown; its expiring "
                    "budget reservation remains held for authoritative reconciliation"
                ) from admission_error
            if recovered is not None:
                return recovered
            try:
                self._budgets.release(
                    reservation,
                    idempotency_key=request.idempotency_key,
                    reason="federation_admission_failed",
                )
            except Exception:
                raise BudgetReconciliationRequired(
                    "federation admission failed and its budget reservation "
                    "requires authoritative reconciliation"
                ) from admission_error
            raise


__all__ = [
    "AuthenticationAlgorithm",
    "AuthenticationAuthority",
    "AuthenticationEvidence",
    "AuthenticationRejected",
    "BudgetAuthority",
    "BudgetReconciliationRequired",
    "BudgetRejected",
    "DelegationAuthority",
    "DelegationGrant",
    "DelegationRejected",
    "FederationAdmissionStore",
    "FederationControlGateway",
    "FederationTriggerError",
    "HmacAuthenticationAuthority",
    "HmacDelegationAuthority",
    "PolicyAuthority",
    "PolicyRejected",
    "RepositoryAuthority",
    "RepositoryResolutionRejected",
    "ResolvedRepository",
    "resolved_authorization_scope_identity",
    "validated_delegation_chain_identity",
]
