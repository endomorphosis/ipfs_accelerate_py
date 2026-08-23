"""Authoritative, typed federation-admission budget reservations.

The calculator in this module cannot mutate a database.  It derives a closed
``BudgetReservation`` from an admitted request and delegates the compare-and-
swap to a state-owner store.  The store is responsible for persisting the
reservation and enforcing capacity in one transaction; tests or callers may
not replace that transaction with an opaque string.
"""

# Python 3.8 compatibility requires ``datetime.timezone.utc`` and ``str, Enum``.
# ruff: noqa: UP017, UP042

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Protocol

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    BudgetDimension,
    BudgetDimensionName,
    BudgetReservation,
    FederationAuthorityError,
    FederationContractError,
    FederationPolicy,
    FederationRequest,
    _identifier,
    _integer,
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
            raise FederationContractError("budget authority_id must be a server authority identity")
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
        over = [item.name.value for item in dimensions if item.reserved > self._capacity[item.name]]
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
                or state_revision not in {("reserved", 1), ("consumed", 2)}
                or persisted.request_cid != request.cid
                or persisted.idempotency_key != request.idempotency_key
                or persisted.policy_ref != policy.record_id
                or persisted.policy_revision != policy.revision
                or persisted.resource_budget_ref != request.resource_budget.record_id
                or persisted.token_budget_ref != request.token_budget.record_id
                or persisted.expires_at != request.expiry
                or not persisted.authorization_evidence_ref.startswith("budget-admission:")
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
            raise FederationAuthorityError("state owner returned a different budget reservation")
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


# ---------------------------------------------------------------------------
# Hierarchical Federation -> Supervisor -> Subagent -> Operation ledger
# ---------------------------------------------------------------------------


class BudgetKind(str, Enum):
    FEDERATION = "federation"
    SUPERVISOR = "supervisor"
    SUBAGENT = "subagent"
    OPERATION = "operation"


class BudgetLedgerError(FederationContractError):
    """Base typed hierarchical-budget failure."""


class BudgetAuthorityError(FederationAuthorityError, BudgetLedgerError):
    """An attempt to overspend, convert validation reserve, or ignore CAS."""


class BudgetCasError(BudgetAuthorityError):
    """Expected revision did not match the current account epoch."""


CHILD_KIND = {
    BudgetKind.FEDERATION: BudgetKind.SUPERVISOR,
    BudgetKind.SUPERVISOR: BudgetKind.SUBAGENT,
    BudgetKind.SUBAGENT: BudgetKind.OPERATION,
}
SPECULATIVE_DIMENSIONS = frozenset(
    {
        BudgetDimensionName.MODEL_CALLS,
        BudgetDimensionName.INPUT_TOKENS,
        BudgetDimensionName.OUTPUT_TOKENS,
        BudgetDimensionName.PROVIDER_SPEND_MICROS,
    }
)
VALIDATION_DIMENSIONS = frozenset(
    {
        BudgetDimensionName.PROOF_MILLIS,
        BudgetDimensionName.VALIDATION_MILLIS,
    }
)


def refuse_ducklake_budget_authority(receipt: Mapping[str, object] | None) -> None:
    if not receipt:
        return
    if receipt.get("authoritative") is True or receipt.get("schedules") is True:
        raise BudgetAuthorityError("DuckLake cannot admit hierarchical budget authority")


@dataclass(frozen=True)
class BudgetSlice:
    """One dimension's ceiling, descendant reservation, and local consumption."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/budget-slice@1"

    ceiling: int
    reserved: int = 0
    consumed: int = 0

    def __post_init__(self) -> None:
        _integer(self.ceiling, "ceiling")
        _integer(self.reserved, "reserved")
        _integer(self.consumed, "consumed")
        if self.reserved + self.consumed > self.ceiling:
            raise BudgetAuthorityError("budget slice exceeds its ceiling")

    @property
    def available(self) -> int:
        return self.ceiling - self.reserved - self.consumed


@dataclass(frozen=True)
class BudgetAccount:
    """One node in the Federation -> Supervisor -> Subagent -> Operation tree."""

    SCHEMA: ClassVar[str] = "ipfs_accelerate_py/agent-supervisor/causal-federation/budget-account@1"

    account_id: str
    kind: BudgetKind
    owner_id: str
    parent_account_id: str
    dimensions: Mapping[BudgetDimensionName, BudgetSlice]
    revision: int

    def __post_init__(self) -> None:
        _identifier(self.account_id, "account_id")
        if not isinstance(self.kind, BudgetKind):
            raise FederationContractError("budget kind is not closed")
        _identifier(self.owner_id, "owner_id")
        _identifier(self.parent_account_id, "parent_account_id", required=False)
        if self.kind is BudgetKind.FEDERATION and self.parent_account_id:
            raise FederationContractError("federation budget cannot name a parent")
        if self.kind is not BudgetKind.FEDERATION and not self.parent_account_id:
            raise FederationContractError("child budget requires a parent account")
        if not isinstance(self.dimensions, Mapping) or not self.dimensions:
            raise FederationContractError("budget account requires dimensions")
        frozen: dict[BudgetDimensionName, BudgetSlice] = {}
        for name, slice_ in self.dimensions.items():
            dimension = name if isinstance(name, BudgetDimensionName) else BudgetDimensionName(name)
            if not isinstance(slice_, BudgetSlice):
                raise FederationContractError("budget dimensions must be BudgetSlice records")
            frozen[dimension] = slice_
        object.__setattr__(self, "dimensions", MappingProxyType(frozen))
        _integer(self.revision, "revision", minimum=1)

    def slice_for(self, dimension: BudgetDimensionName) -> BudgetSlice:
        try:
            return self.dimensions[dimension]
        except KeyError as exc:
            raise BudgetAuthorityError(f"budget account lacks dimension {dimension.value}") from exc


@dataclass(frozen=True)
class BudgetLedgerReceipt:
    """CAS receipt for one hierarchical budget mutation."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/budget-ledger-receipt@1"
    )

    operation: str
    account_id: str
    dimension: BudgetDimensionName
    amount: int
    expected_revision: int
    resulting_revision: int
    parent_account_id: str = ""
    counterpart_account_id: str = ""

    def __post_init__(self) -> None:
        _identifier(self.operation, "operation")
        _identifier(self.account_id, "account_id")
        if not isinstance(self.dimension, BudgetDimensionName):
            raise FederationContractError("budget dimension is not closed")
        _integer(self.amount, "amount", minimum=1)
        _integer(self.expected_revision, "expected_revision")
        _integer(self.resulting_revision, "resulting_revision", minimum=1)
        _identifier(self.parent_account_id, "parent_account_id", required=False)
        _identifier(self.counterpart_account_id, "counterpart_account_id", required=False)


class HierarchicalBudgetLedger:
    """In-memory CAS ledger. Children cannot exceed parents."""

    def __init__(self) -> None:
        self._accounts: dict[str, BudgetAccount] = {}
        self._children: dict[str, set[str]] = {}

    def account(self, account_id: str) -> BudgetAccount:
        try:
            return self._accounts[_identifier(account_id, "account_id")]
        except KeyError as exc:
            raise BudgetLedgerError("budget account is absent") from exc

    def open_root(
        self,
        *,
        account_id: str,
        owner_id: str,
        dimensions: Mapping[BudgetDimensionName, int],
    ) -> BudgetAccount:
        slices = {
            name: BudgetSlice(ceiling=_integer(ceiling, "ceiling", minimum=1))
            for name, ceiling in dimensions.items()
        }
        account = BudgetAccount(
            account_id=account_id,
            kind=BudgetKind.FEDERATION,
            owner_id=owner_id,
            parent_account_id="",
            dimensions=slices,
            revision=1,
        )
        if account.account_id in self._accounts:
            raise BudgetLedgerError("federation budget identity is already bound")
        self._accounts[account.account_id] = account
        self._children[account.account_id] = set()
        return account

    def allocate_child(
        self,
        *,
        parent_account_id: str,
        child_account_id: str,
        child_owner_id: str,
        dimension: BudgetDimensionName,
        amount: int,
        expected_parent_revision: int,
        ducklake_receipt: Mapping[str, object] | None = None,
    ) -> BudgetLedgerReceipt:
        refuse_ducklake_budget_authority(ducklake_receipt)
        parent = self.account(parent_account_id)
        self._assert_revision(parent, expected_parent_revision)
        child_kind = CHILD_KIND.get(parent.kind)
        if child_kind is None:
            raise BudgetAuthorityError("operation budgets cannot allocate children")
        amount = _integer(amount, "amount", minimum=1)
        available = parent.slice_for(dimension).available
        if amount > available:
            raise BudgetAuthorityError("child allocation exceeds parent remainder")
        child = self._accounts.get(child_account_id)
        if child is None:
            child = BudgetAccount(
                account_id=child_account_id,
                kind=child_kind,
                owner_id=child_owner_id,
                parent_account_id=parent.account_id,
                dimensions={dimension: BudgetSlice(ceiling=amount)},
                revision=1,
            )
            self._accounts[child.account_id] = child
            self._children[child.account_id] = set()
            self._children[parent.account_id].add(child.account_id)
        else:
            if child.parent_account_id != parent.account_id:
                raise BudgetAuthorityError("child budget parent differs")
            if child.kind is not child_kind:
                raise BudgetAuthorityError("child budget kind is not the next hierarchy level")
            current = child.dimensions.get(dimension, BudgetSlice(ceiling=0))
            child = replace(
                child,
                dimensions={
                    **dict(child.dimensions),
                    dimension: BudgetSlice(
                        ceiling=current.ceiling + amount,
                        reserved=current.reserved,
                        consumed=current.consumed,
                    ),
                },
                revision=child.revision + 1,
            )
            self._accounts[child.account_id] = child
        parent_slice = parent.slice_for(dimension)
        parent = replace(
            parent,
            dimensions={
                **dict(parent.dimensions),
                dimension: BudgetSlice(
                    ceiling=parent_slice.ceiling,
                    reserved=parent_slice.reserved + amount,
                    consumed=parent_slice.consumed,
                ),
            },
            revision=parent.revision + 1,
        )
        self._accounts[parent.account_id] = parent
        return BudgetLedgerReceipt(
            operation="allocate",
            account_id=child.account_id,
            dimension=dimension,
            amount=amount,
            expected_revision=expected_parent_revision,
            resulting_revision=parent.revision,
            parent_account_id=parent.account_id,
        )

    def consume(
        self,
        *,
        account_id: str,
        dimension: BudgetDimensionName,
        amount: int,
        expected_revision: int,
        speculative: bool = False,
    ) -> BudgetLedgerReceipt:
        account = self.account(account_id)
        self._assert_revision(account, expected_revision)
        amount = _integer(amount, "amount", minimum=1)
        if speculative and dimension in VALIDATION_DIMENSIONS:
            raise BudgetAuthorityError("validation reserves cannot fund speculative reasoning")
        slice_ = account.slice_for(dimension)
        if amount > slice_.available:
            raise BudgetAuthorityError("consumption exceeds remaining budget")
        account = replace(
            account,
            dimensions={
                **dict(account.dimensions),
                dimension: BudgetSlice(
                    ceiling=slice_.ceiling,
                    reserved=slice_.reserved,
                    consumed=slice_.consumed + amount,
                ),
            },
            revision=account.revision + 1,
        )
        self._accounts[account.account_id] = account
        return BudgetLedgerReceipt(
            operation="consume",
            account_id=account.account_id,
            dimension=dimension,
            amount=amount,
            expected_revision=expected_revision,
            resulting_revision=account.revision,
            parent_account_id=account.parent_account_id,
        )

    def return_unused(
        self,
        *,
        child_account_id: str,
        dimension: BudgetDimensionName,
        amount: int,
        expected_child_revision: int,
        expected_parent_revision: int,
    ) -> BudgetLedgerReceipt:
        child = self.account(child_account_id)
        parent = self.account(child.parent_account_id)
        self._assert_revision(child, expected_child_revision)
        self._assert_revision(parent, expected_parent_revision)
        amount = _integer(amount, "amount", minimum=1)
        child_slice = child.slice_for(dimension)
        if amount > child_slice.available:
            raise BudgetAuthorityError("return exceeds unused child remainder")
        parent_slice = parent.slice_for(dimension)
        if amount > parent_slice.reserved:
            raise BudgetAuthorityError("return exceeds parent reservation")
        child = replace(
            child,
            dimensions={
                **dict(child.dimensions),
                dimension: BudgetSlice(
                    ceiling=child_slice.ceiling - amount,
                    reserved=child_slice.reserved,
                    consumed=child_slice.consumed,
                ),
            },
            revision=child.revision + 1,
        )
        parent = replace(
            parent,
            dimensions={
                **dict(parent.dimensions),
                dimension: BudgetSlice(
                    ceiling=parent_slice.ceiling,
                    reserved=parent_slice.reserved - amount,
                    consumed=parent_slice.consumed,
                ),
            },
            revision=parent.revision + 1,
        )
        self._accounts[child.account_id] = child
        self._accounts[parent.account_id] = parent
        return BudgetLedgerReceipt(
            operation="return",
            account_id=child.account_id,
            dimension=dimension,
            amount=amount,
            expected_revision=expected_child_revision,
            resulting_revision=child.revision,
            parent_account_id=parent.account_id,
        )

    def transfer(
        self,
        *,
        source_account_id: str,
        target_account_id: str,
        dimension: BudgetDimensionName,
        amount: int,
        expected_source_revision: int,
        expected_target_revision: int,
    ) -> BudgetLedgerReceipt:
        source = self.account(source_account_id)
        target = self.account(target_account_id)
        if source.parent_account_id != target.parent_account_id:
            raise BudgetAuthorityError("budget transfer requires sibling accounts")
        if source.account_id == target.account_id:
            raise BudgetAuthorityError("budget transfer cannot target the source")
        self._assert_revision(source, expected_source_revision)
        self._assert_revision(target, expected_target_revision)
        amount = _integer(amount, "amount", minimum=1)
        source_slice = source.slice_for(dimension)
        if amount > source_slice.available:
            raise BudgetAuthorityError("transfer exceeds unused source remainder")
        target_slice = target.dimensions.get(dimension, BudgetSlice(ceiling=0))
        source = replace(
            source,
            dimensions={
                **dict(source.dimensions),
                dimension: BudgetSlice(
                    ceiling=source_slice.ceiling - amount,
                    reserved=source_slice.reserved,
                    consumed=source_slice.consumed,
                ),
            },
            revision=source.revision + 1,
        )
        target = replace(
            target,
            dimensions={
                **dict(target.dimensions),
                dimension: BudgetSlice(
                    ceiling=target_slice.ceiling + amount,
                    reserved=target_slice.reserved,
                    consumed=target_slice.consumed,
                ),
            },
            revision=target.revision + 1,
        )
        self._accounts[source.account_id] = source
        self._accounts[target.account_id] = target
        return BudgetLedgerReceipt(
            operation="transfer",
            account_id=source.account_id,
            dimension=dimension,
            amount=amount,
            expected_revision=expected_source_revision,
            resulting_revision=source.revision,
            parent_account_id=source.parent_account_id,
            counterpart_account_id=target.account_id,
        )

    def reallocate(
        self,
        *,
        account_id: str,
        source_dimension: BudgetDimensionName,
        target_dimension: BudgetDimensionName,
        amount: int,
        expected_revision: int,
    ) -> BudgetLedgerReceipt:
        if source_dimension in VALIDATION_DIMENSIONS and target_dimension in SPECULATIVE_DIMENSIONS:
            raise BudgetAuthorityError("validation reserves cannot fund speculative reasoning")
        account = self.account(account_id)
        self._assert_revision(account, expected_revision)
        amount = _integer(amount, "amount", minimum=1)
        source = account.slice_for(source_dimension)
        if amount > source.available:
            raise BudgetAuthorityError("reallocation exceeds unused remainder")
        target = account.dimensions.get(target_dimension, BudgetSlice(ceiling=0))
        account = replace(
            account,
            dimensions={
                **dict(account.dimensions),
                source_dimension: BudgetSlice(
                    ceiling=source.ceiling - amount,
                    reserved=source.reserved,
                    consumed=source.consumed,
                ),
                target_dimension: BudgetSlice(
                    ceiling=target.ceiling + amount,
                    reserved=target.reserved,
                    consumed=target.consumed,
                ),
            },
            revision=account.revision + 1,
        )
        self._accounts[account.account_id] = account
        return BudgetLedgerReceipt(
            operation="reallocate",
            account_id=account.account_id,
            dimension=target_dimension,
            amount=amount,
            expected_revision=expected_revision,
            resulting_revision=account.revision,
            parent_account_id=account.parent_account_id,
        )

    def conserved(self, parent_account_id: str, dimension: BudgetDimensionName) -> bool:
        parent = self.account(parent_account_id)
        children = [self.account(item) for item in self._children.get(parent.account_id, ())]
        child_ceilings = sum(
            child.dimensions[dimension].ceiling
            for child in children
            if dimension in child.dimensions
        )
        return parent.slice_for(dimension).reserved == child_ceilings

    @staticmethod
    def _assert_revision(account: BudgetAccount, expected: int) -> None:
        if expected != account.revision:
            raise BudgetCasError("budget epoch does not match the expected value")


__all__ = [
    "AuthoritativeBudgetAuthority",
    "BudgetAccount",
    "BudgetAuthorityError",
    "BudgetCasError",
    "BudgetKind",
    "BudgetLedgerError",
    "BudgetLedgerReceipt",
    "BudgetReservationStore",
    "BudgetSlice",
    "CHILD_KIND",
    "HierarchicalBudgetLedger",
    "SPECULATIVE_DIMENSIONS",
    "VALIDATION_DIMENSIONS",
    "refuse_ducklake_budget_authority",
]
