# The package still supports Python 3.8, where ``datetime.UTC`` is absent.
# ruff: noqa: UP017

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.budgets import (
    AuthoritativeBudgetAuthority,
    BudgetAuthorityError,
    BudgetCasError,
    BudgetKind,
    HierarchicalBudgetLedger,
    refuse_ducklake_budget_authority,
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
        if record.binding.tenant_id != tenant_id or record.owner_id != federation_id:
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


def _ledger() -> HierarchicalBudgetLedger:
    ledger = HierarchicalBudgetLedger()
    ledger.open_root(
        account_id="budget:federation",
        owner_id="federation:test",
        dimensions={
            BudgetDimensionName.CPU_MILLIS: 1_000,
            BudgetDimensionName.INPUT_TOKENS: 500,
            BudgetDimensionName.VALIDATION_MILLIS: 200,
        },
    )
    return ledger


def test_children_cannot_exceed_parent_remainder() -> None:
    ledger = _ledger()
    first = ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:supervisor-a",
        child_owner_id="supervisor:a",
        dimension=BudgetDimensionName.CPU_MILLIS,
        amount=600,
        expected_parent_revision=1,
    )
    assert first.resulting_revision == 2
    assert ledger.account("budget:supervisor-a").kind is BudgetKind.SUPERVISOR
    with pytest.raises(BudgetAuthorityError, match="exceeds parent remainder"):
        ledger.allocate_child(
            parent_account_id="budget:federation",
            child_account_id="budget:supervisor-b",
            child_owner_id="supervisor:b",
            dimension=BudgetDimensionName.CPU_MILLIS,
            amount=500,
            expected_parent_revision=2,
        )
    ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:supervisor-b",
        child_owner_id="supervisor:b",
        dimension=BudgetDimensionName.CPU_MILLIS,
        amount=400,
        expected_parent_revision=2,
    )
    assert ledger.conserved("budget:federation", BudgetDimensionName.CPU_MILLIS) is True


def test_consume_return_and_sibling_transfer_conserve_parent_reservation() -> None:
    ledger = _ledger()
    ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:supervisor-a",
        child_owner_id="supervisor:a",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=300,
        expected_parent_revision=1,
    )
    ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:supervisor-b",
        child_owner_id="supervisor:b",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=100,
        expected_parent_revision=2,
    )
    consume = ledger.consume(
        account_id="budget:supervisor-a",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=50,
        expected_revision=1,
    )
    assert consume.resulting_revision == 2
    ledger.return_unused(
        child_account_id="budget:supervisor-a",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=100,
        expected_child_revision=2,
        expected_parent_revision=3,
    )
    assert ledger.conserved("budget:federation", BudgetDimensionName.INPUT_TOKENS) is True
    ledger.transfer(
        source_account_id="budget:supervisor-a",
        target_account_id="budget:supervisor-b",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=50,
        expected_source_revision=3,
        expected_target_revision=1,
    )
    assert (
        ledger.account("budget:supervisor-a").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling
        == 150
    )
    assert (
        ledger.account("budget:supervisor-b").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling
        == 150
    )
    assert ledger.conserved("budget:federation", BudgetDimensionName.INPUT_TOKENS) is True


def test_stale_revision_fails_closed() -> None:
    ledger = _ledger()
    with pytest.raises(BudgetCasError, match="epoch does not match"):
        ledger.allocate_child(
            parent_account_id="budget:federation",
            child_account_id="budget:supervisor-a",
            child_owner_id="supervisor:a",
            dimension=BudgetDimensionName.CPU_MILLIS,
            amount=10,
            expected_parent_revision=99,
        )


def test_validation_reserve_cannot_fund_speculative_reasoning() -> None:
    ledger = _ledger()
    with pytest.raises(BudgetAuthorityError, match="validation reserves cannot fund"):
        ledger.reallocate(
            account_id="budget:federation",
            source_dimension=BudgetDimensionName.VALIDATION_MILLIS,
            target_dimension=BudgetDimensionName.INPUT_TOKENS,
            amount=20,
            expected_revision=1,
        )
    with pytest.raises(BudgetAuthorityError, match="validation reserves cannot fund"):
        ledger.consume(
            account_id="budget:federation",
            dimension=BudgetDimensionName.VALIDATION_MILLIS,
            amount=10,
            expected_revision=1,
            speculative=True,
        )


def test_ducklake_cannot_admit_budget_authority() -> None:
    with pytest.raises(BudgetAuthorityError, match="DuckLake cannot admit"):
        refuse_ducklake_budget_authority({"authoritative": True})
    ledger = _ledger()
    with pytest.raises(BudgetAuthorityError, match="DuckLake cannot admit"):
        ledger.allocate_child(
            parent_account_id="budget:federation",
            child_account_id="budget:supervisor-a",
            child_owner_id="supervisor:a",
            dimension=BudgetDimensionName.CPU_MILLIS,
            amount=10,
            expected_parent_revision=1,
            ducklake_receipt={"schedules": True},
        )
