"""Authoritative, typed federation-admission budget reservations.

The calculator in this module cannot mutate a database.  It derives a closed
``BudgetReservation`` from an admitted request and delegates the compare-and-
swap to a state-owner store.  The store is responsible for persisting the
reservation and enforcing capacity in one transaction; tests or callers may
not replace that transaction with an opaque string.
"""

# Python 3.8 compatibility requires ``datetime.timezone.utc``.
# ruff: noqa: UP017

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Protocol

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    BudgetDimension,
    BudgetDimensionName,
    BudgetReservation,
    FederationAuthorityError,
    FederationContractError,
    FederationPolicy,
    FederationRequest,
)


class BudgetReservationStore(Protocol):
    """Exclusive state-owner operations used by the budget authority."""

    def lookup_federation_budget_reservation(
        self,
        idempotency_key: str,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> BudgetReservation | None: ...

    def reserve_federation_budget(
        self,
        reservation: BudgetReservation,
        *,
        capacity: Mapping[BudgetDimensionName, int],
    ) -> BudgetReservation: ...

    def release_federation_budget(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        idempotency_key: str,
        reason: str,
    ) -> None: ...


class AuthoritativeBudgetAuthority:
    """Production adapter over a durable state-owner reservation store.

    Capacity is operator-supplied server state.  It is never read from the
    triggering request, and missing dimensions grant no capacity.
    """

    def __init__(
        self,
        store: BudgetReservationStore,
        *,
        capacity: Mapping[BudgetDimensionName, int],
        authority_id: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not authority_id or not authority_id.startswith("authority:"):
            raise FederationContractError(
                "budget authority_id must be a server authority identity"
            )
        normalized: dict[BudgetDimensionName, int] = {}
        for raw_name, raw_ceiling in capacity.items():
            name = (
                raw_name
                if isinstance(raw_name, BudgetDimensionName)
                else BudgetDimensionName(raw_name)
            )
            if isinstance(raw_ceiling, bool) or not isinstance(raw_ceiling, int):
                raise FederationContractError("budget capacity must be integral")
            if raw_ceiling < 0:
                raise FederationContractError("budget capacity cannot be negative")
            normalized[name] = raw_ceiling
        if not normalized:
            raise FederationContractError("budget capacity must not be empty")
        self._store = store
        self._capacity = normalized
        self._authority_id = authority_id
        self._now = now

    @staticmethod
    def _requested_dimensions(request: FederationRequest) -> tuple[BudgetDimension, ...]:
        requested: dict[BudgetDimensionName, BudgetDimension] = {}
        for budget in (request.resource_budget, request.token_budget):
            for dimension in budget.dimensions:
                if dimension.name in requested:
                    raise FederationAuthorityError(
                        "resource and token reservations overlap the same dimension"
                    )
                requested[dimension.name] = BudgetDimension(
                    name=dimension.name,
                    ceiling=dimension.ceiling,
                    reserved=dimension.ceiling,
                    consumed=0,
                )
        if not requested:
            raise FederationAuthorityError("federation request has no budget dimensions")
        return tuple(requested[name] for name in sorted(requested, key=lambda item: item.value))

    def reserve(
        self,
        request: FederationRequest,
        policy: FederationPolicy,
    ) -> BudgetReservation:
        if policy.binding != request.binding:
            raise FederationAuthorityError("budget policy binding differs from request")
        dimensions = self._requested_dimensions(request)
        missing = [item.name.value for item in dimensions if item.name not in self._capacity]
        if missing:
            raise FederationAuthorityError(
                f"budget capacity telemetry is missing dimensions: {sorted(missing)}"
            )
        over = [
            item.name.value
            for item in dimensions
            if item.reserved > self._capacity[item.name]
        ]
        if over:
            raise FederationAuthorityError(
                f"requested budget exceeds server capacity: {sorted(over)}"
            )
        federation_id = f"federation:{request.cid}"
        persisted = self._store.lookup_federation_budget_reservation(
            request.idempotency_key,
            tenant_id=request.binding.tenant_id,
            federation_id=federation_id,
        )
        if persisted is not None:
            expected_dimensions = tuple(dimensions)
            state_revision = (persisted.status, persisted.revision)
            if (
                persisted.binding != request.binding
                or persisted.owner_id != federation_id
                or persisted.parent_budget_id != request.binding.budget_ref
                or persisted.dimensions != expected_dimensions
                or state_revision
                not in {("reserved", 1), ("consumed", 2)}
                or persisted.request_cid != request.cid
                or persisted.idempotency_key != request.idempotency_key
                or persisted.policy_ref != policy.record_id
                or persisted.policy_revision != policy.revision
                or persisted.resource_budget_ref
                != request.resource_budget.record_id
                or persisted.token_budget_ref != request.token_budget.record_id
                or persisted.expires_at != request.expiry
                or not persisted.authorization_evidence_ref.startswith(
                    "budget-admission:"
                )
            ):
                raise FederationAuthorityError(
                    "persisted budget reservation differs from the retry authority"
                )
            # Federation creation consumes the admission reservation in the
            # same transaction.  An exact retry still needs the immutable
            # pre-consumption object so the create command can reconcile its
            # own idempotency record without minting a new time-bound identity.
            if state_revision == ("consumed", 2):
                return BudgetReservation.from_dict(
                    {
                        **persisted.to_dict(),
                        "revision": 1,
                        "status": "reserved",
                    }
                )
            return persisted
        issued = self._now().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        evidence_body = {
            "authority_id": self._authority_id,
            "tenant_id": request.binding.tenant_id,
            "federation_id": federation_id,
            "request_cid": request.cid,
            "idempotency_key": request.idempotency_key,
            "policy_ref": policy.record_id,
            "policy_revision": policy.revision,
            "resource_budget_ref": request.resource_budget.record_id,
            "token_budget_ref": request.token_budget.record_id,
            "dimensions": [item.to_dict() for item in dimensions],
            "issued_at": issued,
            "expires_at": request.expiry,
        }
        evidence_ref = f"budget-admission:{content_identity(evidence_body)}"
        reservation = BudgetReservation(
            record_id=f"budget-reservation:{content_identity(evidence_body)}",
            revision=1,
            binding=request.binding,
            parent_budget_id=request.binding.budget_ref,
            owner_id=federation_id,
            dimensions=dimensions,
            status="reserved",
            request_cid=request.cid,
            idempotency_key=request.idempotency_key,
            policy_ref=policy.record_id,
            policy_revision=policy.revision,
            resource_budget_ref=request.resource_budget.record_id,
            token_budget_ref=request.token_budget.record_id,
            issued_at=issued,
            expires_at=request.expiry,
            authorization_evidence_ref=evidence_ref,
        )
        admitted = self._store.reserve_federation_budget(
            reservation,
            capacity=self._capacity,
        )
        if admitted != reservation:
            raise FederationAuthorityError(
                "state owner returned a different budget reservation"
            )
        return admitted

    def release(
        self,
        reservation: BudgetReservation,
        *,
        idempotency_key: str,
        reason: str,
    ) -> None:
        if not isinstance(reservation, BudgetReservation):
            raise FederationContractError("release requires BudgetReservation")
        if reservation.idempotency_key != idempotency_key:
            raise FederationAuthorityError("budget release idempotency scope differs")
        self._store.release_federation_budget(
            reservation.record_id,
            tenant_id=reservation.binding.tenant_id,
            federation_id=reservation.owner_id,
            idempotency_key=idempotency_key,
            reason=reason,
        )


__all__ = [
    "AuthoritativeBudgetAuthority",
    "BudgetReservationStore",
]
