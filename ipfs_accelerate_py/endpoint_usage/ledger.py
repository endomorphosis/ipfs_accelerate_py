"""Materialized usage ledger: windows, headroom, snapshots, and compaction.

The ledger is append-only at the semantic event level. Materialized counters
and windows may be compacted, but corrections never silently rewrite source
events — they append and reference the superseded event.

:class:`UsageLedger` is a pure document transformer: it does not perform I/O.
:class:`UsageCoordinator` owns the compare-and-set transaction boundary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .identity import content_cid, stable_id
from .schema import (
    AvailabilityState,
    DimensionHeadroom,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    Provenance,
    Quantity,
    QuantityKind,
    ReservationState,
    UsageDimension,
    UsageErrorCode,
    UsageEvent,
    UsageEventKind,
    UsageLimit,
    UsageReservation,
    UsageSnapshot,
    UsageVector,
    UsageVectorEntry,
    WindowKind,
)
from .store import (
    ATOMIC_USAGE_LEDGER_REQUIREMENT_ID,
    datetime_to_ms,
    empty_ledger_document,
    parse_rfc3339,
    rfc3339_to_ms,
)


# Re-export for AST/evidence discovery.
__requirement_id__ = ATOMIC_USAGE_LEDGER_REQUIREMENT_ID


NEAR_LIMIT_RATIO_MICROS = 100_000  # 10% remaining → near_limit
DEFAULT_SNAPSHOT_FRESH_MS = 5_000


class LedgerError(ValueError):
    """Semantic ledger failure (invalid transition, capacity, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = UsageErrorCode.CAPACITY_UNAVAILABLE.value,
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason_codes = tuple(reason_codes)


class CapacityDenied(LedgerError):
    """Reservation would exceed one or more hard limits/budgets."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
        limit_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            code=UsageErrorCode.LIMIT_EXHAUSTED.value,
            reason_codes=reason_codes or ("limit_exhausted",),
        )
        self.limit_id = limit_id


class StaleSnapshot(LedgerError):
    """Caller provided an outdated usage_revision."""

    def __init__(self, message: str = "stale usage snapshot revision") -> None:
        super().__init__(
            message,
            code=UsageErrorCode.STALE_SNAPSHOT.value,
            reason_codes=("stale_snapshot",),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def dimension_key(dimension: UsageDimension | str, currency: Optional[str] = None) -> str:
    dim = dimension.value if isinstance(dimension, UsageDimension) else str(dimension)
    if currency:
        return "%s:%s" % (dim, currency)
    return dim


def parse_dimension_key(key: str) -> Tuple[str, Optional[str]]:
    if ":" in key:
        dim, currency = key.split(":", 1)
        return dim, currency
    return key, None


def vector_to_amounts(vector: UsageVector | Mapping[str, Any]) -> Dict[str, int]:
    if isinstance(vector, UsageVector):
        entries = vector.entries
    else:
        entries = UsageVector.from_dict(vector).entries
    amounts: Dict[str, int] = {}
    for entry in entries:
        if entry.amount.kind is not QuantityKind.FINITE or entry.amount.value is None:
            raise LedgerError(
                "usage amounts must be finite",
                code=UsageErrorCode.INVALID_UNIT_WINDOW.value,
                reason_codes=("invalid_amount",),
            )
        key = dimension_key(entry.dimension, entry.currency)
        amounts[key] = amounts.get(key, 0) + int(entry.amount.value)
    return amounts


def amounts_to_vector(amounts: Mapping[str, int]) -> UsageVector:
    entries = []
    for key, value in sorted(amounts.items()):
        if value < 0:
            raise LedgerError(
                "negative usage amount",
                code=UsageErrorCode.NEGATIVE_VALUE.value,
                reason_codes=("negative_value",),
            )
        dim_name, currency = parse_dimension_key(key)
        dimension = UsageDimension(dim_name)
        entries.append(
            UsageVectorEntry(
                dimension=dimension,
                amount=Quantity.finite(int(value)),
                currency=currency,
            )
        )
    return UsageVector(entries=tuple(entries))


def merge_amounts(
    left: Mapping[str, int], right: Mapping[str, int], *, scale_right: int = 1
) -> Dict[str, int]:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, 0) + int(value) * scale_right
    return out


def _limit_from_dict(data: Mapping[str, Any]) -> UsageLimit:
    if isinstance(data, UsageLimit):
        return data
    return UsageLimit.from_dict(data)


def _active_reservation_states() -> frozenset:
    return frozenset(
        (
            ReservationState.PENDING.value,
            ReservationState.HELD.value,
        )
    )


# ---------------------------------------------------------------------------
# Window accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSample:
    """One charged contribution inside a sliding or fixed window."""

    occurred_ms: int
    amount: int
    dimension_key: str
    source_id: str  # reservation_id or event_id
    kind: str  # reserved | committed | settled


def _window_length_ms(window: LimitWindow) -> Optional[int]:
    return window.length_ms


def _window_reset_ms(window: LimitWindow) -> Optional[int]:
    if window.reset_at is None:
        return None
    return rfc3339_to_ms(window.reset_at)


def _window_anchor_ms(window: LimitWindow) -> Optional[int]:
    if window.anchor_at is None:
        return None
    return rfc3339_to_ms(window.anchor_at)


def fixed_window_start_ms(window: LimitWindow, now_ms: int) -> Optional[int]:
    """Return the inclusive start of the active fixed window, if defined."""

    length = _window_length_ms(window)
    if length is None or length <= 0:
        return None
    anchor = _window_anchor_ms(window)
    reset = _window_reset_ms(window)
    if reset is not None:
        # Window is [reset - length, reset) cycling if past reset.
        if now_ms < reset:
            return reset - length
        # Deterministic cycle forward from reset.
        elapsed = now_ms - reset
        cycles = elapsed // length
        return reset + cycles * length
    if anchor is not None:
        if now_ms < anchor:
            return anchor
        elapsed = now_ms - anchor
        cycles = elapsed // length
        return anchor + cycles * length
    # Default: align to epoch multiples of length.
    return (now_ms // length) * length


def sample_in_window(
    sample_ms: int,
    window: LimitWindow,
    now_ms: int,
) -> bool:
    kind = window.kind
    if kind is WindowKind.LIFETIME:
        return True
    if kind is WindowKind.CONCURRENT:
        return True
    if kind is WindowKind.SLIDING:
        length = _window_length_ms(window)
        if length is None:
            return True
        return sample_ms >= now_ms - length and sample_ms <= now_ms
    if kind is WindowKind.FIXED:
        length = _window_length_ms(window)
        start = fixed_window_start_ms(window, now_ms)
        if length is None or start is None:
            return True
        return start <= sample_ms < start + length
    if kind is WindowKind.BILLING:
        reset = _window_reset_ms(window)
        anchor = _window_anchor_ms(window)
        if reset is not None and now_ms >= reset:
            # Past reset: only samples at/after reset count (new period).
            return sample_ms >= reset
        if anchor is not None and sample_ms < anchor:
            return False
        if reset is not None:
            return sample_ms < reset
        return True
    if kind is WindowKind.TOKEN_BUCKET:
        # Token bucket does not use historical samples the same way; treated separately.
        return True
    return True


def effective_ceiling(limit: UsageLimit, *, partition_scale: Optional[Tuple[int, int]] = None) -> Optional[int]:
    """Return finite effective ceiling after safety reserve, or None if unknown/unlimited."""

    ceiling = limit.ceiling
    if ceiling.kind is QuantityKind.UNKNOWN:
        return None
    if ceiling.kind is QuantityKind.UNLIMITED:
        return None  # signal unlimited as a separate path
    assert ceiling.value is not None
    value = int(ceiling.value)
    reserve = limit.window.safety_reserve or 0
    value = max(0, value - int(reserve))
    if partition_scale is not None:
        num, den = partition_scale
        if den <= 0:
            raise LedgerError("invalid partition denominator")
        value = (value * num) // den
    return value


def is_unlimited_ceiling(limit: UsageLimit) -> bool:
    return limit.ceiling.kind is QuantityKind.UNLIMITED


def is_unknown_ceiling(limit: UsageLimit) -> bool:
    return limit.ceiling.kind is QuantityKind.UNKNOWN


# ---------------------------------------------------------------------------
# UsageLedger — pure document operations
# ---------------------------------------------------------------------------


class UsageLedger:
    """Pure functions over a ledger document for materialization and policy.

    One compare-and-set transaction (owned by the coordinator) loads a
    document, applies mutations through this class, and commits.
    """

    requirement_id = ATOMIC_USAGE_LEDGER_REQUIREMENT_ID

    def __init__(self, document: Optional[Mapping[str, Any]] = None) -> None:
        if document is None:
            self._doc = empty_ledger_document()
        else:
            self._doc = copy.deepcopy(dict(document))

    @property
    def document(self) -> Dict[str, Any]:
        return self._doc

    def clone(self) -> "UsageLedger":
        return UsageLedger(self._doc)

    # -- limits / budgets ---------------------------------------------------

    def set_limits(
        self,
        scope_id: str,
        limits: Sequence[UsageLimit | Mapping[str, Any]],
    ) -> None:
        out = []
        for item in limits:
            if isinstance(item, UsageLimit):
                limit = item
            else:
                limit = UsageLimit.from_dict(item)
            if limit.scope_id != scope_id:
                raise LedgerError(
                    "limit scope_id mismatch",
                    code=UsageErrorCode.INVALID_SCOPE.value,
                    reason_codes=("invalid_scope",),
                )
            out.append(limit.to_dict())
        self._doc.setdefault("limits", {})[scope_id] = out

    def get_limits(self, scope_id: str) -> Tuple[UsageLimit, ...]:
        raw = (self._doc.get("limits") or {}).get(scope_id) or []
        return tuple(_limit_from_dict(item) for item in raw)

    def set_caller_budget(
        self,
        scope_id: str,
        budget: UsageVector | Mapping[str, Any],
    ) -> None:
        amounts = vector_to_amounts(budget if isinstance(budget, UsageVector) else UsageVector.from_dict(budget))
        self._doc.setdefault("caller_budgets", {})[scope_id] = amounts

    def get_caller_budget(self, scope_id: str) -> Dict[str, int]:
        raw = (self._doc.get("caller_budgets") or {}).get(scope_id) or {}
        return {str(k): int(v) for k, v in raw.items()}

    def disable_scope(self, scope_id: str, *, reason: str = "disabled") -> None:
        self._doc.setdefault("disabled_scopes", {})[scope_id] = reason

    def enable_scope(self, scope_id: str) -> None:
        self._doc.setdefault("disabled_scopes", {}).pop(scope_id, None)

    def set_cooldown(self, scope_id: str, until: str) -> None:
        # Validate timestamp.
        parse_rfc3339(until)
        self._doc.setdefault("cooldown_until", {})[scope_id] = until

    def clear_cooldown(self, scope_id: str) -> None:
        self._doc.setdefault("cooldown_until", {}).pop(scope_id, None)

    # -- event append -------------------------------------------------------

    def next_sequence(self) -> int:
        return int(self._doc.get("next_sequence") or 1)

    def append_event(self, event: UsageEvent | Mapping[str, Any]) -> UsageEvent:
        if isinstance(event, Mapping):
            data = dict(event)
        else:
            data = event.to_dict()
        if data.get("sequence") is None:
            data["sequence"] = self.next_sequence()
        # event_id is content-addressed over sequence; always recompute.
        data["event_id"] = None
        event_obj = UsageEvent.from_dict(data)
        seq = event_obj.sequence
        assert seq is not None
        if seq != self.next_sequence():
            # Allow only the next sequence for append-only integrity.
            if seq < self.next_sequence():
                raise LedgerError(
                    "event sequence %s already used" % seq,
                    code=UsageErrorCode.RESERVATION_CONFLICT.value,
                    reason_codes=("sequence_conflict",),
                )
            raise LedgerError(
                "event sequence %s is not next (%s)" % (seq, self.next_sequence()),
                code=UsageErrorCode.RESERVATION_CONFLICT.value,
                reason_codes=("sequence_gap",),
            )
        events = list(self._doc.get("events") or [])
        events.append(event_obj.to_dict())
        self._doc["events"] = events
        self._doc["next_sequence"] = seq + 1
        if event_obj.kind is UsageEventKind.CORRECTION:
            if not event_obj.supersedes_event_id:
                raise LedgerError(
                    "correction requires supersedes_event_id",
                    reason_codes=("correction_missing_ref",),
                )
            self._doc.setdefault("corrections", {})[event_obj.supersedes_event_id] = (
                event_obj.event_id
            )
        return event_obj

    # -- reservation records ------------------------------------------------

    def put_reservation_record(self, record: Mapping[str, Any]) -> None:
        reservation_id = record.get("reservation_id")
        if not reservation_id:
            raise LedgerError("reservation record requires reservation_id")
        self._doc.setdefault("reservations", {})[str(reservation_id)] = copy.deepcopy(
            dict(record)
        )

    def get_reservation_record(self, reservation_id: str) -> Optional[Dict[str, Any]]:
        raw = (self._doc.get("reservations") or {}).get(reservation_id)
        return copy.deepcopy(raw) if raw is not None else None

    def iter_reservation_records(self, scope_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for record in (self._doc.get("reservations") or {}).values():
            if scope_id is not None and record.get("scope_id") != scope_id:
                continue
            out.append(copy.deepcopy(record))
        return out

    def put_idempotency(self, key: str, decision: Mapping[str, Any]) -> None:
        self._doc.setdefault("idempotency", {})[key] = copy.deepcopy(dict(decision))

    def get_idempotency(self, key: str) -> Optional[Dict[str, Any]]:
        raw = (self._doc.get("idempotency") or {}).get(key)
        return copy.deepcopy(raw) if raw is not None else None

    def stream_settled(self, reservation_id: str) -> Dict[str, int]:
        raw = (self._doc.get("stream_settled") or {}).get(reservation_id) or {}
        return {str(k): int(v) for k, v in raw.items()}

    def set_stream_settled(self, reservation_id: str, amounts: Mapping[str, int]) -> None:
        self._doc.setdefault("stream_settled", {})[reservation_id] = {
            str(k): int(v) for k, v in amounts.items()
        }

    def batch_charge_state(self, batch_id: str) -> Dict[str, Any]:
        raw = (self._doc.get("batch_charges") or {}).get(batch_id)
        if raw is None:
            return {"overhead_charged": False, "members": {}}
        return copy.deepcopy(raw)

    def set_batch_charge_state(self, batch_id: str, state: Mapping[str, Any]) -> None:
        self._doc.setdefault("batch_charges", {})[batch_id] = copy.deepcopy(dict(state))

    # -- occupancy ----------------------------------------------------------

    def _samples_for_scope(self, scope_id: str) -> List[WindowSample]:
        """Build window samples from active reservations and committed charges.

        Active holds contribute residual reserved capacity plus already
        settled/committed amounts exactly once (no double count). Terminal
        records contribute only their charged amounts.
        """

        samples: List[WindowSample] = []
        for record in self.iter_reservation_records(scope_id):
            state = record.get("state")
            created_ms = (
                rfc3339_to_ms(record["created_at"]) if record.get("created_at") else 0
            )
            reserved = {
                str(k): int(v) for k, v in (record.get("reserved_amounts") or {}).items()
            }
            committed = {
                str(k): int(v) for k, v in (record.get("committed_amounts") or {}).items()
            }
            charged = {
                str(k): int(v) for k, v in (record.get("charged_amounts") or {}).items()
            }
            rid = str(record.get("reservation_id"))
            if state in _active_reservation_states():
                settled = self.stream_settled(rid)
                keys = set(reserved) | set(committed) | set(settled)
                for key in keys:
                    already = max(committed.get(key, 0), settled.get(key, 0))
                    hold = max(0, reserved.get(key, 0) - already)
                    if hold > 0:
                        samples.append(
                            WindowSample(
                                occurred_ms=created_ms,
                                amount=hold,
                                dimension_key=key,
                                source_id=rid,
                                kind="reserved",
                            )
                        )
                    if already > 0:
                        samples.append(
                            WindowSample(
                                occurred_ms=created_ms,
                                amount=already,
                                dimension_key=key,
                                source_id=rid,
                                kind="settled",
                            )
                        )
            elif state in (
                ReservationState.COMMITTED.value,
                ReservationState.RELEASED.value,
                ReservationState.EXPIRED.value,
            ):
                # Skip reset records that zeroed charges.
                if record.get("reset_at") and not charged:
                    continue
                for key, amount in charged.items():
                    if amount > 0:
                        samples.append(
                            WindowSample(
                                occurred_ms=created_ms,
                                amount=amount,
                                dimension_key=key,
                                source_id=rid,
                                kind="committed",
                            )
                        )
        return samples

    def occupancy(
        self,
        scope_id: str,
        limit: UsageLimit,
        now: datetime,
    ) -> int:
        """Return units currently counting against *limit* at *now*."""

        now_ms = datetime_to_ms(now)
        dim_key = dimension_key(limit.dimension, limit.currency)
        window = limit.window
        samples = self._samples_for_scope(scope_id)

        if window.kind is WindowKind.CONCURRENT:
            # Concurrent occupancy = sum of held reserved for this dimension.
            total = 0
            for sample in samples:
                if sample.dimension_key != dim_key:
                    continue
                if sample.kind != "reserved":
                    continue
                record = self.get_reservation_record(sample.source_id)
                if record is None:
                    continue
                if record.get("state") not in _active_reservation_states():
                    continue
                expires = record.get("expires_at")
                if expires is not None and rfc3339_to_ms(expires) <= now_ms:
                    continue
                total += sample.amount
            return total

        if window.kind is WindowKind.TOKEN_BUCKET:
            return self._token_bucket_used(scope_id, limit, now_ms, dim_key, samples)

        total = 0
        for sample in samples:
            if sample.dimension_key != dim_key:
                continue
            if not sample_in_window(sample.occurred_ms, window, now_ms):
                continue
            total += sample.amount
        return total

    def _token_bucket_used(
        self,
        scope_id: str,
        limit: UsageLimit,
        now_ms: int,
        dim_key: str,
        samples: Sequence[WindowSample],
    ) -> int:
        """Return used tokens relative to burst after refill (conservative).

        We model tokens = min(burst, refilled) - consumed_in_window. Occupancy
        is returned as (burst - available) so ceiling comparison works with
        burst as the ceiling when configured.
        """
        burst = limit.window.burst
        refill = limit.window.refill_per_second or 0
        if burst is None:
            burst = effective_ceiling(limit) or 0
        # Sum consumption in the recent length window (or since anchor).
        window = limit.window
        consumed = 0
        for sample in samples:
            if sample.dimension_key != dim_key:
                continue
            if sample.kind == "reserved":
                # In-flight also consumes tokens.
                pass
            if not sample_in_window(sample.occurred_ms, window, now_ms):
                # For token bucket without length, count lifetime.
                if window.length_ms is None and window.reset_at is None:
                    consumed += sample.amount
                continue
            consumed += sample.amount
        # Conservative: do not invent refill that would free capacity mid-CAS
        # beyond what elapsed time from earliest sample allows.
        if not samples:
            return 0
        earliest = min(s.occurred_ms for s in samples if s.dimension_key == dim_key) if any(
            s.dimension_key == dim_key for s in samples
        ) else now_ms
        elapsed_s = max(0, (now_ms - earliest) // 1000)
        refilled = elapsed_s * int(refill)
        # available = min(burst, burst - consumed + refilled) clamped
        # used = burst - available
        net = max(0, consumed - refilled)
        return min(int(burst), net)

    def headroom_for_limit(
        self,
        scope_id: str,
        limit: UsageLimit,
        now: datetime,
        *,
        partition_scale: Optional[Tuple[int, int]] = None,
    ) -> DimensionHeadroom:
        used = self.occupancy(scope_id, limit, now)
        if is_unknown_ceiling(limit):
            return DimensionHeadroom(
                dimension=limit.dimension,
                available=Quantity.unknown(),
                ceiling=Quantity.unknown(),
                reserved=Quantity.finite(used),
                currency=limit.currency,
                state=AvailabilityState.UNKNOWN,
            )
        if is_unlimited_ceiling(limit):
            return DimensionHeadroom(
                dimension=limit.dimension,
                available=Quantity.unlimited(),
                ceiling=Quantity.unlimited(),
                reserved=Quantity.finite(used),
                currency=limit.currency,
                state=AvailabilityState.AVAILABLE,
            )
        ceiling_value = effective_ceiling(limit, partition_scale=partition_scale)
        assert ceiling_value is not None
        available = max(0, ceiling_value - used)
        state = AvailabilityState.AVAILABLE
        if available == 0:
            state = AvailabilityState.EXHAUSTED
        elif ceiling_value > 0 and (available * 1_000_000) // ceiling_value <= NEAR_LIMIT_RATIO_MICROS:
            state = AvailabilityState.NEAR_LIMIT
        cooldown = (self._doc.get("cooldown_until") or {}).get(scope_id)
        next_eligible = None
        if cooldown:
            try:
                if datetime_to_ms(now) < rfc3339_to_ms(cooldown):
                    state = AvailabilityState.COOLING_DOWN
                    next_eligible = cooldown
            except ValueError:
                pass
        if (self._doc.get("disabled_scopes") or {}).get(scope_id):
            state = AvailabilityState.DISABLED
        # next eligible from window reset
        if next_eligible is None and available == 0:
            reset = limit.window.reset_at
            if reset is not None:
                next_eligible = reset
            elif limit.window.kind is WindowKind.FIXED and limit.window.length_ms:
                start = fixed_window_start_ms(limit.window, datetime_to_ms(now))
                if start is not None:
                    end_ms = start + int(limit.window.length_ms)
                    next_eligible = _to_rfc3339(
                        datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
                    )
        return DimensionHeadroom(
            dimension=limit.dimension,
            available=Quantity.finite(available),
            ceiling=Quantity.finite(ceiling_value),
            reserved=Quantity.finite(used),
            currency=limit.currency,
            state=state,
            next_eligible_at=next_eligible,
        )

    def check_capacity(
        self,
        scope_id: str,
        requested: Mapping[str, int],
        now: datetime,
        *,
        caller_budget: Optional[Mapping[str, int]] = None,
        partition_scale: Optional[Tuple[int, int]] = None,
        ignore_reservation_id: Optional[str] = None,
    ) -> None:
        """Fail closed if *requested* cannot be granted at *now*."""

        if (self._doc.get("disabled_scopes") or {}).get(scope_id):
            raise CapacityDenied(
                "scope is disabled",
                reason_codes=("scope_disabled",),
            )
        cooldown = (self._doc.get("cooldown_until") or {}).get(scope_id)
        if cooldown is not None:
            try:
                if datetime_to_ms(now) < rfc3339_to_ms(cooldown):
                    raise CapacityDenied(
                        "scope is cooling down until %s" % cooldown,
                        reason_codes=("cooling_down",),
                    )
            except ValueError as exc:
                raise LedgerError(
                    "invalid cooldown timestamp",
                    reason_codes=("invalid_cooldown",),
                ) from exc

        # Caller budget (request-local or configured).
        budget = dict(self.get_caller_budget(scope_id))
        if caller_budget:
            for key, value in caller_budget.items():
                budget[key] = min(budget[key], int(value)) if key in budget else int(value)
        for key, need in requested.items():
            if key in budget and need > budget[key]:
                raise CapacityDenied(
                    "caller budget exceeded for %s" % key,
                    reason_codes=("caller_budget_exceeded", key.replace(":", ".")),
                )

        limits = self.get_limits(scope_id)
        if not limits:
            # No configured limits: fail closed for hard admission unless budget only.
            # Policy: allow only when caller_budget covers all dimensions.
            if not budget:
                raise CapacityDenied(
                    "no configured limits or caller budget for scope",
                    reason_codes=("no_limits_configured",),
                )
            for key, need in requested.items():
                if key not in budget:
                    raise CapacityDenied(
                        "dimension %s not covered by caller budget" % key,
                        reason_codes=("budget_dimension_missing",),
                    )
            return

        # Check every limit that overlaps a requested dimension.
        for limit in limits:
            if limit.enforcement is LimitEnforcement.DIAGNOSTIC:
                continue
            dim_key = dimension_key(limit.dimension, limit.currency)
            need = int(requested.get(dim_key, 0))
            if need <= 0 and limit.window.kind is not WindowKind.CONCURRENT:
                # Still check concurrent if requesting concurrent dim.
                if dim_key not in requested:
                    continue
            if is_unknown_ceiling(limit) and limit.enforcement is LimitEnforcement.HARD:
                raise CapacityDenied(
                    "hard limit has unknown ceiling for %s" % dim_key,
                    reason_codes=("unknown_ceiling",),
                    limit_id=limit.limit_id,
                )
            if is_unlimited_ceiling(limit):
                continue
            used = self.occupancy(scope_id, limit, now)
            # Optionally subtract the ignored reservation's hold (for settle updates).
            if ignore_reservation_id:
                record = self.get_reservation_record(ignore_reservation_id)
                if record and record.get("state") in _active_reservation_states():
                    held = int((record.get("reserved_amounts") or {}).get(dim_key, 0))
                    settled = self.stream_settled(ignore_reservation_id).get(dim_key, 0)
                    used = max(0, used - max(0, held - settled))
            ceiling_value = effective_ceiling(limit, partition_scale=partition_scale)
            assert ceiling_value is not None
            if used + need > ceiling_value:
                if limit.enforcement is LimitEnforcement.SOFT:
                    continue
                raise CapacityDenied(
                    "limit %s exhausted: used=%s need=%s ceiling=%s"
                    % (limit.limit_id, used, need, ceiling_value),
                    reason_codes=("limit_exhausted", limit.dimension.value),
                    limit_id=limit.limit_id,
                )

    # -- snapshots ----------------------------------------------------------

    def build_snapshot(
        self,
        scope_id: str,
        now: datetime,
        *,
        partition_scale: Optional[Tuple[int, int]] = None,
        fresh_ms: int = DEFAULT_SNAPSHOT_FRESH_MS,
    ) -> UsageSnapshot:
        limits = self.get_limits(scope_id)
        headroom = tuple(
            self.headroom_for_limit(
                scope_id, limit, now, partition_scale=partition_scale
            )
            for limit in limits
        )
        active = []
        now_ms = datetime_to_ms(now)
        for record in self.iter_reservation_records(scope_id):
            state = record.get("state")
            if state not in _active_reservation_states():
                continue
            expires = record.get("expires_at")
            if expires is not None and rfc3339_to_ms(expires) <= now_ms:
                continue
            reservation = UsageReservation.from_dict(record["reservation"])
            active.append(reservation)
        observed = _to_rfc3339(now)
        fresh_until = _to_rfc3339(
            datetime.fromtimestamp((now_ms + fresh_ms) / 1000.0, tz=timezone.utc)
        )
        # Aggregate availability.
        if (self._doc.get("disabled_scopes") or {}).get(scope_id):
            state = AvailabilityState.DISABLED
        elif not limits:
            state = AvailabilityState.UNKNOWN
        else:
            states = [item.state for item in headroom]
            if any(s is AvailabilityState.DISABLED for s in states):
                state = AvailabilityState.DISABLED
            elif any(s is AvailabilityState.COOLING_DOWN for s in states):
                state = AvailabilityState.COOLING_DOWN
            elif any(s is AvailabilityState.EXHAUSTED for s in states):
                state = AvailabilityState.EXHAUSTED
            elif any(s is AvailabilityState.NEAR_LIMIT for s in states):
                state = AvailabilityState.NEAR_LIMIT
            elif any(s is AvailabilityState.UNKNOWN for s in states):
                state = AvailabilityState.UNKNOWN
            else:
                state = AvailabilityState.AVAILABLE
        next_eligible_candidates = [
            item.next_eligible_at for item in headroom if item.next_eligible_at
        ]
        next_eligible = min(next_eligible_candidates) if next_eligible_candidates else None
        reason_codes: List[str] = []
        if state is AvailabilityState.EXHAUSTED:
            reason_codes.append("limit_exhausted")
        if state is AvailabilityState.COOLING_DOWN:
            reason_codes.append("cooling_down")
        return UsageSnapshot(
            scope_id=scope_id,
            observed_at=observed,
            fresh_until=fresh_until,
            state=state,
            limits=limits,
            headroom=headroom,
            reservations=tuple(active),
            next_eligible_at=next_eligible,
            reason_codes=tuple(reason_codes),
        )

    # -- compaction / reset -------------------------------------------------

    def compact(self, *, retain_events: int = 0) -> Dict[str, Any]:
        """Compact the event log while preserving replay from the checkpoint.

        Events after ``compacted_through`` remain. A checkpoint captures
        reservation/limit/index materialization so replay remains exact.
        """

        events = list(self._doc.get("events") or [])
        if not events:
            return {"compacted": 0, "compacted_through": self._doc.get("compacted_through", 0)}
        if retain_events < 0:
            raise LedgerError("retain_events must be non-negative")
        if retain_events >= len(events):
            return {"compacted": 0, "compacted_through": self._doc.get("compacted_through", 0)}
        drop_count = len(events) - retain_events
        dropped = events[:drop_count]
        kept = events[drop_count:]
        last_seq = int(dropped[-1]["sequence"])
        checkpoint = {
            "compacted_through": last_seq,
            "limits": copy.deepcopy(self._doc.get("limits") or {}),
            "caller_budgets": copy.deepcopy(self._doc.get("caller_budgets") or {}),
            "reservations": copy.deepcopy(self._doc.get("reservations") or {}),
            "idempotency": copy.deepcopy(self._doc.get("idempotency") or {}),
            "stream_settled": copy.deepcopy(self._doc.get("stream_settled") or {}),
            "batch_charges": copy.deepcopy(self._doc.get("batch_charges") or {}),
            "corrections": copy.deepcopy(self._doc.get("corrections") or {}),
            "cooldown_until": copy.deepcopy(self._doc.get("cooldown_until") or {}),
            "disabled_scopes": copy.deepcopy(self._doc.get("disabled_scopes") or {}),
            "events_digest": content_cid({"events": dropped}),
        }
        self._doc["checkpoint"] = checkpoint
        self._doc["compacted_through"] = last_seq
        self._doc["events"] = kept
        return {
            "compacted": drop_count,
            "compacted_through": last_seq,
            "events_digest": checkpoint["events_digest"],
        }

    def reset_scope(
        self,
        scope_id: str,
        *,
        now: datetime,
        reason: str = "admin_reset",
        expected_revision: Optional[str] = None,
    ) -> UsageEvent:
        """Deterministic admin reset of counters/reservations for a scope.

        Does not erase historical events; appends a correction-style release of
        active holds and zeroes charged amounts going forward via a reset event.
        """

        # Release active reservations.
        for record in self.iter_reservation_records(scope_id):
            if record.get("state") in _active_reservation_states():
                record = copy.deepcopy(record)
                record["state"] = ReservationState.RELEASED.value
                record["charged_amounts"] = {}
                record["committed_amounts"] = {}
                reservation = dict(record["reservation"])
                reservation["state"] = ReservationState.RELEASED.value
                record["reservation"] = reservation
                self.put_reservation_record(record)
                self._doc.setdefault("stream_settled", {}).pop(
                    record["reservation_id"], None
                )
        # Zero charged history for terminal records of this scope (forward-looking
        # occupancy): mark charged_amounts empty so windows start fresh.
        for record in self.iter_reservation_records(scope_id):
            record = copy.deepcopy(record)
            record["charged_amounts"] = {}
            record["committed_amounts"] = {}
            record["reset_at"] = _to_rfc3339(now)
            record["reset_reason"] = reason
            self.put_reservation_record(record)
        self.clear_cooldown(scope_id)
        event = self.append_event(
            UsageEvent(
                kind=UsageEventKind.RELEASE,
                scope_id=scope_id,
                occurred_at=_to_rfc3339(now),
                units=UsageVector(),
                reason_codes=("admin_reset", reason.replace(" ", "_")[:64]),
                provenance=Provenance(
                    source=LimitSource.RECONCILED,
                    observed_at=_to_rfc3339(now),
                    reason_codes=("admin_reset",),
                ),
            )
        )
        return event

    def replay_events_from_checkpoint(self) -> List[Dict[str, Any]]:
        """Return the ordered event stream visible after compaction."""

        return list(self._doc.get("events") or [])


def apply_partition_to_limits(
    limits: Sequence[UsageLimit],
    numerator: int,
    denominator: int,
) -> Tuple[UsageLimit, ...]:
    """Return limits with ceilings floored to the partition share."""

    out = []
    for limit in limits:
        if limit.ceiling.kind is not QuantityKind.FINITE or limit.ceiling.value is None:
            out.append(limit)
            continue
        scaled = (int(limit.ceiling.value) * numerator) // denominator
        if (
            limit.remaining.kind is QuantityKind.FINITE
            and limit.remaining.value is not None
        ):
            remaining = Quantity.finite(min(scaled, int(limit.remaining.value)))
        else:
            remaining = limit.remaining
        out.append(
            UsageLimit(
                scope_id=limit.scope_id,
                dimension=limit.dimension,
                ceiling=Quantity.finite(scaled),
                window=limit.window,
                remaining=remaining,
                used=limit.used,
                enforcement=limit.enforcement,
                confidence_micros=limit.confidence_micros,
                confidence=limit.confidence,
                provenance=limit.provenance,
                currency=limit.currency,
            )
        )
    return tuple(out)


__all__ = [
    "ATOMIC_USAGE_LEDGER_REQUIREMENT_ID",
    "CapacityDenied",
    "DEFAULT_SNAPSHOT_FRESH_MS",
    "LedgerError",
    "NEAR_LIMIT_RATIO_MICROS",
    "StaleSnapshot",
    "UsageLedger",
    "WindowSample",
    "amounts_to_vector",
    "apply_partition_to_limits",
    "dimension_key",
    "effective_ceiling",
    "fixed_window_start_ms",
    "merge_amounts",
    "parse_dimension_key",
    "sample_in_window",
    "vector_to_amounts",
]
