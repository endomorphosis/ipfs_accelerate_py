# The package still supports Python 3.8, where ``datetime.UTC`` is absent.
# ruff: noqa: UP017

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.budgets import (
    AuthoritativeBudgetAuthority,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    BudgetDimensionName,
    BudgetReservation,
    FederationAuthorityError,
)
from test.api.causal_federation.test_trigger import sample_policy, sample_request


class DurableBudgetStoreFake:
    """Models the state-owner CAS boundary, including restart durability."""

    def __init__(self, rows: dict[str, BudgetReservation] | None = None) -> None:
        self.rows = rows if rows is not None else {}
        self.released: set[str] = set()

    def reserve_federation_budget(
        self,
        reservation: BudgetReservation,
        *,
        capacity: dict[BudgetDimensionName, int],
    ) -> BudgetReservation:
        existing = self.rows.get(reservation.idempotency_key)
        if existing is not None:
            if existing != reservation:
                raise FederationAuthorityError("idempotency request differs")
            return existing
        active = {
            name: sum(
                item.reserved
                for record in self.rows.values()
                if record.record_id not in self.released
                for item in record.dimensions
                if item.name is name
            )
            for name in capacity
        }
        for dimension in reservation.dimensions:
            if active[dimension.name] + dimension.reserved > capacity[dimension.name]:
                raise FederationAuthorityError("capacity exhausted")
        self.rows[reservation.idempotency_key] = reservation
        return reservation

    def lookup_federation_budget_reservation(
        self,
        idempotency_key: str,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> BudgetReservation | None:
        record = self.rows.get(idempotency_key)
        if record is None:
            return None
        if (
            record.binding.tenant_id != tenant_id
            or record.owner_id != federation_id
        ):
            raise FederationAuthorityError("lookup scope differs")
        return record

    def release_federation_budget(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        idempotency_key: str,
        reason: str,
    ) -> None:
        del reason
        record = self.rows[idempotency_key]
        if (
            record.record_id != reservation_id
            or record.binding.tenant_id != tenant_id
            or record.owner_id != federation_id
        ):
            raise FederationAuthorityError("release scope differs")
        self.released.add(record.record_id)


def _authority(store: DurableBudgetStoreFake) -> AuthoritativeBudgetAuthority:
    return AuthoritativeBudgetAuthority(
        store,
        capacity={
            BudgetDimensionName.CPU_MILLIS: 200,
            BudgetDimensionName.INPUT_TOKENS: 200,
        },
        authority_id="authority:budget:test",
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def test_authoritative_budget_reservation_is_typed_idempotent_and_restart_safe() -> None:
    request = sample_request()
    policy = sample_policy(request.binding)
    rows: dict[str, BudgetReservation] = {}

    first = _authority(DurableBudgetStoreFake(rows)).reserve(request, policy)
    second = _authority(DurableBudgetStoreFake(rows)).reserve(request, policy)

    assert first == second
    assert first.request_cid == request.cid
    assert first.owner_id == f"federation:{request.cid}"
    assert first.authorization_evidence_ref.startswith("budget-admission:")
    assert len(rows) == 1


def test_authoritative_budget_retry_reuses_persisted_time_bound_identity() -> None:
    request = sample_request()
    policy = sample_policy(request.binding)
    store = DurableBudgetStoreFake()
    current = datetime(2030, 1, 1, tzinfo=timezone.utc)
    calls = 0

    def advancing_clock() -> datetime:
        nonlocal current, calls
        observed = current
        current += timedelta(seconds=30)
        calls += 1
        return observed

    authority = AuthoritativeBudgetAuthority(
        store,
        capacity={
            BudgetDimensionName.CPU_MILLIS: 200,
            BudgetDimensionName.INPUT_TOKENS: 200,
        },
        authority_id="authority:budget:test",
        now=advancing_clock,
    )
    first = authority.reserve(request, policy)
    second = authority.reserve(request, policy)

    assert second == first
    assert second.issued_at == "2030-01-01T00:00:00Z"
    assert calls == 1


def test_authoritative_budget_missing_capacity_and_cross_scope_release_fail_closed() -> None:
    request = sample_request()
    policy = sample_policy(request.binding)
    store = DurableBudgetStoreFake()
    authority = _authority(store)
    reservation = authority.reserve(request, policy)

    with pytest.raises(FederationAuthorityError):
        authority.release(
            reservation,
            idempotency_key="idempotency:foreign",
            reason="test",
        )

    missing = AuthoritativeBudgetAuthority(
        store,
        capacity={BudgetDimensionName.CPU_MILLIS: 200},
        authority_id="authority:budget:test",
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(FederationAuthorityError, match="missing dimensions"):
        missing.reserve(request, policy)
