"""Objective-level reservation and reconciliation for cognitive work.

The coordinator in this module composes the existing provider-usage and token
attribution authorities by recording their immutable receipt IDs.  It does
not estimate provider-native token usage, settle provider cost, schedule
resources, or turn accounting into completion evidence.

Every action must first obtain a reservation bound to one question, action,
purpose, and idempotency key.  Reconciliation preserves actual usage even
when a provider overruns its reservation; such an overrun closes the ledger
fail-closed.  The canonical :class:`BudgetLedger` snapshot is sufficient for
deterministic restart and replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import Any

from .contracts import (
    BudgetDimension,
    BudgetExhaustion,
    BudgetExhaustionReason,
    BudgetLedger,
    BudgetPurpose,
    BudgetReservation,
    BudgetReservationStatus,
    CognitiveBudget,
    TerminalStatus,
)

_MAX_VALUE = (1 << 63) - 1
_MAX_ID_BYTES = 512


class CognitiveBudgetProtocolError(ValueError):
    """The caller violated reservation or idempotency protocol."""


_DIMENSIONS = tuple(BudgetDimension)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_VALUE:
        raise CognitiveBudgetProtocolError(f"{name} must be a bounded non-negative integer")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise CognitiveBudgetProtocolError(f"{name} must be a string")
    result = value.strip()
    if (
        not result
        or len(result.encode("utf-8")) > _MAX_ID_BYTES
        or any(char.isspace() for char in result)
        or "\x00" in result
    ):
        raise CognitiveBudgetProtocolError(f"{name} must be a compact bounded identifier")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise CognitiveBudgetProtocolError(f"{name} has an unsupported value") from exc


def _ids(values: Any, name: str) -> tuple[str, ...]:
    raw: tuple[Any, ...] | list[Any]
    if values is None:
        raw = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, (list, tuple)):
        raw = values
    else:
        raise CognitiveBudgetProtocolError(f"{name} must be a sequence of identifiers")
    if len(raw) > 4_096:
        raise CognitiveBudgetProtocolError(f"{name} contains too many identifiers")
    return tuple(sorted({_identifier(item, name) for item in raw}))


@dataclass(frozen=True)
class CognitiveCost:
    """Integer-unit requested or observed cognitive cost vector."""

    total_model_calls: int = 0
    strong_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider_spend_micros: int = 0
    proof_time_ms: int = 0
    validation_time_ms: int = 0
    human_questions: int = 0
    repair_rounds: int = 0
    plan_branches: int = 0
    context_expansions: int = 0
    wall_time_ms: int = 0

    def __post_init__(self) -> None:
        for dimension in _DIMENSIONS:
            object.__setattr__(
                self,
                dimension.value,
                _integer(getattr(self, dimension.value), dimension.value),
            )
        if self.strong_model_calls > self.total_model_calls:
            raise CognitiveBudgetProtocolError("strong-model calls cannot exceed total model calls")

    def __getitem__(self, dimension: BudgetDimension) -> int:
        return getattr(self, _enum(dimension, BudgetDimension, "dimension").value)

    @property
    def is_zero(self) -> bool:
        return not any(self[dimension] for dimension in _DIMENSIONS)

    def to_dict(self) -> dict[str, int]:
        return {dimension.value: self[dimension] for dimension in _DIMENSIONS}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CognitiveCost:
        if not isinstance(value, Mapping):
            raise CognitiveBudgetProtocolError("cost must be a mapping")
        allowed = {dimension.value for dimension in _DIMENSIONS}
        if set(value).difference(allowed):
            raise CognitiveBudgetProtocolError("cost contains unsupported dimensions")
        return cls(**{name: value.get(name, 0) for name in allowed})


def _cost(value: CognitiveCost | Mapping[str, Any]) -> CognitiveCost:
    if isinstance(value, CognitiveCost):
        return value
    return CognitiveCost.from_mapping(value)


ReservationOutcome = BudgetReservation | BudgetExhaustion


def _maximum_field(dimension: BudgetDimension) -> str:
    return "max_" + dimension.value


def _reserved_form(record: BudgetReservation) -> BudgetReservation:
    return replace(
        record,
        status=BudgetReservationStatus.RESERVED,
        actual_total_model_calls=0,
        actual_strong_model_calls=0,
        actual_input_tokens=0,
        actual_output_tokens=0,
        actual_provider_spend_micros=0,
        actual_proof_time_ms=0,
        actual_validation_time_ms=0,
        actual_human_questions=0,
        actual_repair_rounds=0,
        actual_plan_branches=0,
        actual_context_expansions=0,
        actual_wall_time_ms=0,
        provider_usage_receipt_ids=(),
        token_measurement_ids=(),
    )


class ObjectiveCognitiveBudgetLedger:
    """Thread-safe objective/epoch budget state with canonical snapshots."""

    def __init__(self, budget: CognitiveBudget, *, epoch: int = 0) -> None:
        if not isinstance(budget, CognitiveBudget):
            if isinstance(budget, Mapping):
                budget = CognitiveBudget.from_dict(budget)
            else:
                raise CognitiveBudgetProtocolError("budget must be a CognitiveBudget")
        self._budget = budget
        self._epoch = _integer(epoch, "epoch")
        self._records: dict[str, BudgetReservation] = {}
        self._exhaustions: dict[str, BudgetExhaustion] = {}
        self._aliases: dict[str, str] = {}
        self._terminal_status = TerminalStatus.PENDING
        self._lock = RLock()

    @property
    def budget(self) -> CognitiveBudget:
        return self._budget

    @property
    def epoch(self) -> int:
        return self._epoch

    @classmethod
    def from_snapshot(cls, snapshot: BudgetLedger) -> ObjectiveCognitiveBudgetLedger:
        if not isinstance(snapshot, BudgetLedger):
            if isinstance(snapshot, Mapping):
                snapshot = BudgetLedger.from_dict(snapshot)
            else:
                raise CognitiveBudgetProtocolError("snapshot must be a BudgetLedger")
        result = cls(snapshot.budget, epoch=snapshot.epoch)
        result._terminal_status = snapshot.status
        for record in snapshot.reservations:
            result._install(record)
        for exhaustion in snapshot.exhaustions:
            result._install_exhaustion(exhaustion)
        if result.snapshot().content_id != snapshot.content_id:
            raise CognitiveBudgetProtocolError(
                "restart snapshot is not a canonical projection of attributed reservations"
            )
        return result

    def _install(self, record: BudgetReservation) -> None:
        if record.idempotency_key in self._records or record.idempotency_key in self._exhaustions:
            raise CognitiveBudgetProtocolError("duplicate reservation idempotency key")
        self._records[record.idempotency_key] = record
        self._aliases[record.reservation_id] = record.idempotency_key
        self._aliases[_reserved_form(record).reservation_id] = record.idempotency_key

    def _install_exhaustion(self, exhaustion: BudgetExhaustion) -> None:
        if (
            exhaustion.idempotency_key in self._records
            or exhaustion.idempotency_key in self._exhaustions
        ):
            raise CognitiveBudgetProtocolError("duplicate budget outcome idempotency key")
        self._exhaustions[exhaustion.idempotency_key] = exhaustion

    def _replace(self, previous: BudgetReservation, current: BudgetReservation) -> None:
        self._records[current.idempotency_key] = current
        self._aliases[previous.reservation_id] = current.idempotency_key
        self._aliases[current.reservation_id] = current.idempotency_key
        self._aliases[_reserved_form(current).reservation_id] = current.idempotency_key

    def _record_for(self, reservation_id: str) -> BudgetReservation:
        key = self._aliases.get(_identifier(reservation_id, "reservation_id"))
        if key is None:
            raise CognitiveBudgetProtocolError(
                "usage cannot be reconciled without a current reservation"
            )
        return self._records[key]

    def _totals(self, *, include_reserved: bool) -> CognitiveCost:
        values = {dimension.value: 0 for dimension in _DIMENSIONS}
        for record in self._records.values():
            if record.status is BudgetReservationStatus.RECONCILED:
                prefix = "actual_"
            elif include_reserved and record.status is BudgetReservationStatus.RESERVED:
                prefix = "max_"
            else:
                continue
            for dimension in _DIMENSIONS:
                values[dimension.value] += getattr(record, prefix + dimension.value)
        return CognitiveCost(**values)

    def _limit(self, dimension: BudgetDimension, purpose: BudgetPurpose) -> tuple[int, int]:
        limit = getattr(self._budget, _maximum_field(dimension))
        protected = 0
        if (
            dimension is BudgetDimension.VALIDATION_TIME_MS
            and purpose is not BudgetPurpose.VALIDATION
        ):
            protected = self._budget.validation_reserve_ms
        elif dimension is BudgetDimension.PROOF_TIME_MS and purpose is not BudgetPurpose.PROOF:
            protected = self._budget.proof_reserve_ms
        return limit - protected, protected

    def available(self, purpose: BudgetPurpose) -> CognitiveCost:
        normalized_purpose = _enum(purpose, BudgetPurpose, "purpose")
        with self._lock:
            totals = self._totals(include_reserved=True)
            values: dict[str, int] = {}
            for dimension in _DIMENSIONS:
                limit, _ = self._limit(dimension, normalized_purpose)
                values[dimension.value] = max(0, limit - totals[dimension])
            values[BudgetDimension.STRONG_MODEL_CALLS.value] = min(
                values[BudgetDimension.STRONG_MODEL_CALLS.value],
                values[BudgetDimension.TOTAL_MODEL_CALLS.value],
            )
            return CognitiveCost(**values)

    def reserve(
        self,
        *,
        idempotency_key: str,
        question_id: str,
        action_id: str,
        purpose: BudgetPurpose,
        requested: CognitiveCost | Mapping[str, Any],
        expires_at_ms: int = 0,
    ) -> ReservationOutcome:
        requested_cost = _cost(requested)
        normalized_purpose = _enum(purpose, BudgetPurpose, "purpose")
        candidate = BudgetReservation(
            budget_id=self._budget.budget_id,
            idempotency_key=idempotency_key,
            question_id=question_id,
            action_id=action_id,
            purpose=normalized_purpose,
            status=BudgetReservationStatus.RESERVED,
            expires_at_ms=expires_at_ms,
            max_total_model_calls=requested_cost.total_model_calls,
            max_strong_model_calls=requested_cost.strong_model_calls,
            max_input_tokens=requested_cost.input_tokens,
            max_output_tokens=requested_cost.output_tokens,
            max_provider_spend_micros=requested_cost.provider_spend_micros,
            max_proof_time_ms=requested_cost.proof_time_ms,
            max_validation_time_ms=requested_cost.validation_time_ms,
            max_human_questions=requested_cost.human_questions,
            max_repair_rounds=requested_cost.repair_rounds,
            max_plan_branches=requested_cost.plan_branches,
            max_context_expansions=requested_cost.context_expansions,
            max_wall_time_ms=requested_cost.wall_time_ms,
        )
        with self._lock:
            previous = self._records.get(candidate.idempotency_key)
            if previous is not None:
                if _reserved_form(previous).content_id != candidate.content_id:
                    raise CognitiveBudgetProtocolError(
                        "idempotency key was replayed with a different reservation request"
                    )
                return previous
            prior_exhaustion = self._exhaustions.get(candidate.idempotency_key)
            if prior_exhaustion is not None:
                if prior_exhaustion.request_fingerprint != candidate.content_id:
                    raise CognitiveBudgetProtocolError(
                        "idempotency key was replayed with a different reservation request"
                    )
                return prior_exhaustion
            snapshot = self.snapshot()
            if self._terminal_status is not TerminalStatus.PENDING:
                dimension = next(
                    (item for item in _DIMENSIONS if requested_cost[item]),
                    BudgetDimension.WALL_TIME_MS,
                )
                exhaustion = BudgetExhaustion(
                    budget_id=self._budget.budget_id,
                    ledger_id=snapshot.ledger_id,
                    idempotency_key=candidate.idempotency_key,
                    request_fingerprint=candidate.content_id,
                    question_id=candidate.question_id,
                    action_id=candidate.action_id,
                    purpose=normalized_purpose,
                    dimension=dimension,
                    reason=BudgetExhaustionReason.LEDGER_TERMINAL,
                    requested=requested_cost[dimension],
                    available=0,
                )
                self._install_exhaustion(exhaustion)
                return exhaustion
            totals = self._totals(include_reserved=True)
            for dimension in _DIMENSIONS:
                limit, protected = self._limit(dimension, normalized_purpose)
                available = max(0, limit - totals[dimension])
                amount = requested_cost[dimension]
                if amount <= available:
                    continue
                raw_available = max(
                    0,
                    getattr(self._budget, _maximum_field(dimension)) - totals[dimension],
                )
                if protected and amount <= raw_available:
                    reason = (
                        BudgetExhaustionReason.VALIDATION_RESERVE_PROTECTED
                        if dimension is BudgetDimension.VALIDATION_TIME_MS
                        else BudgetExhaustionReason.PROOF_RESERVE_PROTECTED
                    )
                else:
                    reason = BudgetExhaustionReason.CAPACITY_EXHAUSTED
                exhaustion = BudgetExhaustion(
                    budget_id=self._budget.budget_id,
                    ledger_id=snapshot.ledger_id,
                    idempotency_key=candidate.idempotency_key,
                    request_fingerprint=candidate.content_id,
                    question_id=candidate.question_id,
                    action_id=candidate.action_id,
                    purpose=normalized_purpose,
                    dimension=dimension,
                    reason=reason,
                    requested=amount,
                    available=available,
                    protected_reserve=protected,
                )
                self._install_exhaustion(exhaustion)
                return exhaustion
            self._install(candidate)
            return candidate

    def reconcile(
        self,
        reservation_id: str,
        actual: CognitiveCost | Mapping[str, Any],
        *,
        provider_usage_receipt_ids: tuple[str, ...] = (),
        token_measurement_ids: tuple[str, ...] = (),
    ) -> BudgetReservation:
        actual_cost = _cost(actual)
        provider_receipts = _ids(provider_usage_receipt_ids, "provider_usage_receipt_ids")
        token_measurements = _ids(token_measurement_ids, "token_measurement_ids")
        with self._lock:
            record = self._record_for(reservation_id)
            candidate = replace(
                record,
                status=BudgetReservationStatus.RECONCILED,
                provider_usage_receipt_ids=provider_receipts,
                token_measurement_ids=token_measurements,
                actual_total_model_calls=actual_cost.total_model_calls,
                actual_strong_model_calls=actual_cost.strong_model_calls,
                actual_input_tokens=actual_cost.input_tokens,
                actual_output_tokens=actual_cost.output_tokens,
                actual_provider_spend_micros=actual_cost.provider_spend_micros,
                actual_proof_time_ms=actual_cost.proof_time_ms,
                actual_validation_time_ms=actual_cost.validation_time_ms,
                actual_human_questions=actual_cost.human_questions,
                actual_repair_rounds=actual_cost.repair_rounds,
                actual_plan_branches=actual_cost.plan_branches,
                actual_context_expansions=actual_cost.context_expansions,
                actual_wall_time_ms=actual_cost.wall_time_ms,
            )
            if record.status is BudgetReservationStatus.RECONCILED:
                if record.content_id != candidate.content_id:
                    raise CognitiveBudgetProtocolError(
                        "reconciliation replay conflicts with recorded actual usage"
                    )
                return record
            if record.status in {
                BudgetReservationStatus.RELEASED,
                BudgetReservationStatus.CANCELLED,
            }:
                raise CognitiveBudgetProtocolError(
                    "released or cancelled capacity cannot be reconciled"
                )
            self._replace(record, candidate)
            totals = self._totals(include_reserved=True)
            reservation_overrun = any(
                actual_cost[dimension] > getattr(record, _maximum_field(dimension))
                for dimension in _DIMENSIONS
            )
            global_overrun = any(
                totals[dimension] > getattr(self._budget, _maximum_field(dimension))
                for dimension in _DIMENSIONS
            )
            # The reservation is the action's admitted maximum, not a hint.
            # Preserve actual use for attribution, but close the objective
            # ledger even when unrelated global capacity happens to remain.
            if reservation_overrun or global_overrun:
                self._terminal_status = TerminalStatus.EXHAUSTED
            return candidate

    def release(self, reservation_id: str, *, cancelled: bool = False) -> BudgetReservation:
        with self._lock:
            record = self._record_for(reservation_id)
            requested_status = (
                BudgetReservationStatus.CANCELLED if cancelled else BudgetReservationStatus.RELEASED
            )
            if record.status is requested_status:
                return record
            if record.status is not BudgetReservationStatus.RESERVED:
                raise CognitiveBudgetProtocolError(
                    "only unreconciled reservations may be released or cancelled"
                )
            current = replace(record, status=requested_status)
            self._replace(record, current)
            return current

    def reservation(self, reservation_id: str) -> BudgetReservation:
        with self._lock:
            return self._record_for(reservation_id)

    def snapshot(self) -> BudgetLedger:
        with self._lock:
            records = tuple(self._records[key] for key in sorted(self._records))
            exhaustions = tuple(self._exhaustions[key] for key in sorted(self._exhaustions))
            committed = self._totals(include_reserved=False)
            provider_receipts = tuple(
                sorted(
                    {
                        receipt_id
                        for record in records
                        if record.status is BudgetReservationStatus.RECONCILED
                        for receipt_id in record.provider_usage_receipt_ids
                    }
                )
            )
            token_measurements = tuple(
                sorted(
                    {
                        measurement_id
                        for record in records
                        if record.status is BudgetReservationStatus.RECONCILED
                        for measurement_id in record.token_measurement_ids
                    }
                )
            )
            return BudgetLedger(
                budget=self._budget,
                epoch=self._epoch,
                reservations=records,
                exhaustions=exhaustions,
                provider_usage_receipt_ids=provider_receipts,
                token_measurement_ids=token_measurements,
                status=self._terminal_status,
                committed_total_model_calls=committed.total_model_calls,
                committed_strong_model_calls=committed.strong_model_calls,
                committed_input_tokens=committed.input_tokens,
                committed_output_tokens=committed.output_tokens,
                committed_provider_spend_micros=committed.provider_spend_micros,
                committed_proof_time_ms=committed.proof_time_ms,
                committed_validation_time_ms=committed.validation_time_ms,
                committed_human_questions=committed.human_questions,
                committed_repair_rounds=committed.repair_rounds,
                committed_plan_branches=committed.plan_branches,
                committed_context_expansions=committed.context_expansions,
                committed_wall_time_ms=committed.wall_time_ms,
            )


__all__ = [
    "BudgetDimension",
    "BudgetExhaustion",
    "BudgetExhaustionReason",
    "CognitiveBudgetProtocolError",
    "CognitiveCost",
    "ObjectiveCognitiveBudgetLedger",
    "ReservationOutcome",
]
