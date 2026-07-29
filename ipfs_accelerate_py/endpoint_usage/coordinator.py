"""Atomic reservation coordinator over a :class:`UsageLedgerStore`.

One compare-and-set transaction checks caller budget, configured/provider
limits, all overlapping windows, active reservations, and lease/fence before
granting capacity. Reservation identities bind request/attempt/idempotency/
scope/estimate/revision/TTL/owner. Replay returns the same decision; a new
attempt receives a distinct reservation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .identity import stable_id
from .ledger import (
    CapacityDenied,
    LedgerError,
    StaleSnapshot,
    UsageLedger,
    amounts_to_vector,
    apply_partition_to_limits,
    dimension_key,
    merge_amounts,
    vector_to_amounts,
)
from .schema import (
    LimitSource,
    Provenance,
    ReservationState,
    UsageErrorCode,
    UsageEstimate,
    UsageEvent,
    UsageEventKind,
    UsageLimit,
    UsageReservation,
    UsageSnapshot,
    UsageVector,
)
from .store import (
    ATOMIC_USAGE_LEDGER_REQUIREMENT_ID,
    AdmissionAuthorityError,
    CapacityPartition,
    Clock,
    CompareAndSetConflict,
    UsageLedgerStore,
    datetime_to_ms,
    idempotency_index_key,
    migrate_document,
    read_only_recovery_view,
    rfc3339_to_ms,
)


DEFAULT_RESERVATION_TTL_MS = 30_000
DEFAULT_CAS_RETRIES = 8
PROVIDER_CHARGEABLE_DEFAULT = frozenset(
    {
        "requests",
        "batch_items",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "embedding_inputs",
        "embedding_tokens",
        "vectors",
        "images",
        "pixels",
        "media_bytes",
        "audio_seconds",
        "characters",
        "cost_micros",
    }
)


def _to_rfc3339(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class ReserveDecision:
    """Outcome of an atomic reserve attempt (granted or denied)."""

    granted: bool
    reservation: Optional[UsageReservation]
    reservation_id: Optional[str]
    usage_revision: str
    snapshot: UsageSnapshot
    reason_codes: Tuple[str, ...] = ()
    error_code: Optional[str] = None
    replayed: bool = False
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "granted": self.granted,
            "reservation": self.reservation.to_dict() if self.reservation else None,
            "reservation_id": self.reservation_id,
            "usage_revision": self.usage_revision,
            "snapshot": self.snapshot.to_dict(),
            "reason_codes": list(self.reason_codes),
            "error_code": self.error_code,
            "replayed": self.replayed,
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class SettlementResult:
    reservation_id: str
    state: ReservationState
    charged: UsageVector
    event_id: str
    usage_revision: str
    replayed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "state": self.state.value,
            "charged": self.charged.to_dict(),
            "event_id": self.event_id,
            "usage_revision": self.usage_revision,
            "replayed": self.replayed,
        }


class UsageCoordinator:
    """Reservation and settlement coordinator backed by a fenced store."""

    requirement_id = ATOMIC_USAGE_LEDGER_REQUIREMENT_ID

    def __init__(
        self,
        store: UsageLedgerStore,
        *,
        writer_id: Optional[str] = None,
        fence: Optional[int] = None,
        partition: Optional[CapacityPartition] = None,
        cas_retries: int = DEFAULT_CAS_RETRIES,
        provider_chargeable_dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        if not getattr(store, "authorizes_admission", False):
            raise AdmissionAuthorityError(
                "store does not authorize admission; "
                "eventual IPFS replication alone cannot authorize admission"
            )
        self._store = store
        self._writer_id = writer_id
        self._fence = fence
        self._partition = partition
        self._cas_retries = max(1, int(cas_retries))
        self._provider_chargeable = frozenset(
            provider_chargeable_dimensions
            if provider_chargeable_dimensions is not None
            else PROVIDER_CHARGEABLE_DEFAULT
        )

    @property
    def store(self) -> UsageLedgerStore:
        return self._store

    @property
    def clock(self) -> Clock:
        return self._store.clock

    @property
    def partition(self) -> Optional[CapacityPartition]:
        return self._partition

    def _partition_scale(self) -> Optional[Tuple[int, int]]:
        if self._partition is None:
            return None
        return (self._partition.numerator, self._partition.denominator)

    def _cas(self, mutate) -> Any:
        """Run *mutate(ledger) -> (result, ledger)* inside a CAS loop."""

        last_conflict: Optional[Exception] = None
        for _ in range(self._cas_retries):
            document = self._store.read()
            if self._fence is not None and int(document.get("fence") or 0) > int(
                self._fence
            ):
                from .store import StaleFenceError

                raise StaleFenceError(
                    "coordinator fence %s is stale (store fence %s)"
                    % (self._fence, document.get("fence"))
                )
            expected = int(document["revision"])
            ledger = UsageLedger(document)
            try:
                result, ledger = mutate(ledger)
            except (CapacityDenied, LedgerError, StaleSnapshot):
                raise
            new_doc = ledger.document
            if self._partition is not None:
                new_doc["partition"] = self._partition.to_dict()
            try:
                committed = self._store.compare_and_set(
                    expected,
                    new_doc,
                    writer_id=self._writer_id,
                    fence=self._fence,
                )
            except CompareAndSetConflict as exc:
                last_conflict = exc
                continue
            # Attach committed revision into result if it is a known type.
            return result, committed
        raise CompareAndSetConflict(
            "exhausted CAS retries (%s): %s"
            % (self._cas_retries, last_conflict)
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_limits(
        self,
        scope_id: str,
        limits: Sequence[UsageLimit | Mapping[str, Any]],
        *,
        apply_partition: bool = True,
    ) -> UsageSnapshot:
        parsed = []
        for item in limits:
            if isinstance(item, UsageLimit):
                parsed.append(item)
            else:
                parsed.append(UsageLimit.from_dict(item))
        if apply_partition and self._partition is not None:
            parsed = list(
                apply_partition_to_limits(
                    parsed,
                    self._partition.numerator,
                    self._partition.denominator,
                )
            )

        def mutate(ledger: UsageLedger):
            ledger.set_limits(scope_id, parsed)
            now = self.clock.now()
            snap = ledger.build_snapshot(
                scope_id, now, partition_scale=None  # already scaled into limits
            )
            return snap, ledger

        snap, _ = self._cas(mutate)
        return snap

    def configure_caller_budget(
        self, scope_id: str, budget: UsageVector | Mapping[str, Any]
    ) -> None:
        vector = budget if isinstance(budget, UsageVector) else UsageVector.from_dict(budget)

        def mutate(ledger: UsageLedger):
            ledger.set_caller_budget(scope_id, vector)
            return None, ledger

        self._cas(mutate)

    def snapshot(self, scope_id: str) -> UsageSnapshot:
        document = self._store.read()
        ledger = UsageLedger(document)
        return ledger.build_snapshot(
            scope_id,
            self.clock.now(),
            partition_scale=None,
        )

    # ------------------------------------------------------------------
    # Reserve
    # ------------------------------------------------------------------

    def reserve(
        self,
        scope_id: str,
        requested: UsageVector | Mapping[str, Any],
        *,
        request_id: str,
        attempt_id: str = "1",
        idempotency_key: str,
        owner_id: str,
        lease_id: Optional[str] = None,
        estimate: Optional[UsageEstimate | Mapping[str, Any]] = None,
        expected_usage_revision: Optional[str] = None,
        ttl_ms: int = DEFAULT_RESERVATION_TTL_MS,
        caller_budget: Optional[UsageVector | Mapping[str, Any]] = None,
        fence: Optional[int] = None,
    ) -> ReserveDecision:
        """Atomically reserve multi-dimension capacity for one attempt.

        Reservation IDs bind request/attempt/idempotency/scope/estimate/
        revision/TTL/owner through the content-addressed
        :class:`UsageReservation` fields (request_id encodes attempt; TTL is
        bound via expires_at material and idempotency framing).
        """

        if ttl_ms <= 0:
            raise LedgerError(
                "ttl_ms must be positive",
                code=UsageErrorCode.INVALID_UNIT_WINDOW.value,
                reason_codes=("invalid_ttl",),
            )
        vector = (
            requested
            if isinstance(requested, UsageVector)
            else UsageVector.from_dict(requested)
        )
        amounts = vector_to_amounts(vector)
        estimate_obj: Optional[UsageEstimate] = None
        if estimate is not None:
            estimate_obj = (
                estimate
                if isinstance(estimate, UsageEstimate)
                else UsageEstimate.from_dict(estimate)
            )
        budget_amounts = None
        if caller_budget is not None:
            budget_amounts = vector_to_amounts(
                caller_budget
                if isinstance(caller_budget, UsageVector)
                else UsageVector.from_dict(caller_budget)
            )
        effective_fence = fence if fence is not None else self._fence
        bound_request_id = "%s#attempt=%s" % (request_id, attempt_id)
        # Bind estimate/revision/TTL into the reservation identity via
        # idempotency framing while preserving client idempotency_key for replay.
        idem_key = idempotency_index_key(
            scope_id=scope_id,
            request_id=request_id,
            attempt_id=attempt_id,
            idempotency_key=idempotency_key,
        )

        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            snap = ledger.build_snapshot(scope_id, now)
            if expected_usage_revision is not None and snap.usage_revision != expected_usage_revision:
                raise StaleSnapshot(
                    "expected usage_revision %s but snapshot is %s"
                    % (expected_usage_revision, snap.usage_revision)
                )

            prior = ledger.get_idempotency(idem_key)
            if prior is not None:
                return self._replay_decision(ledger, prior, scope_id, now), ledger

            # Build reservation with identity-binding fields.
            expires = now + timedelta(milliseconds=ttl_ms)
            estimate_id = estimate_obj.estimate_id if estimate_obj else None
            # Frame idempotency_key for reservation_id binding of estimate/rev/TTL.
            framed_idem = stable_id(
                "uresbind",
                {
                    "idempotency_key": idempotency_key,
                    "attempt_id": attempt_id,
                    "estimate_id": estimate_id,
                    "usage_revision": snap.usage_revision,
                    "ttl_ms": ttl_ms,
                    "owner_id": owner_id,
                },
            )
            reservation = UsageReservation(
                scope_id=scope_id,
                reserved=vector,
                state=ReservationState.HELD,
                request_id=bound_request_id,
                idempotency_key=framed_idem,
                owner_id=owner_id,
                lease_id=lease_id,
                fence=effective_fence,
                created_at=_to_rfc3339(now),
                expires_at=_to_rfc3339(expires),
                estimate_id=estimate_id,
            )
            try:
                ledger.check_capacity(
                    scope_id,
                    amounts,
                    now,
                    caller_budget=budget_amounts,
                    partition_scale=None,
                )
            except CapacityDenied as exc:
                denied = UsageReservation(
                    scope_id=scope_id,
                    reserved=vector,
                    state=ReservationState.REJECTED,
                    request_id=bound_request_id,
                    idempotency_key=framed_idem,
                    owner_id=owner_id,
                    lease_id=lease_id,
                    fence=effective_fence,
                    created_at=_to_rfc3339(now),
                    expires_at=_to_rfc3339(expires),
                    estimate_id=estimate_id,
                    reason_codes=exc.reason_codes,
                )
                event = ledger.append_event(
                    UsageEvent(
                        kind=UsageEventKind.RESERVATION,
                        scope_id=scope_id,
                        occurred_at=_to_rfc3339(now),
                        request_id=bound_request_id,
                        reservation_id=denied.reservation_id,
                        estimate_id=estimate_id,
                        units=vector,
                        reason_codes=exc.reason_codes,
                        provenance=Provenance(
                            source=LimitSource.LOCAL_OBSERVATION,
                            observed_at=_to_rfc3339(now),
                            reason_codes=exc.reason_codes,
                        ),
                    )
                )
                decision = {
                    "granted": False,
                    "reservation": denied.to_dict(),
                    "reservation_id": denied.reservation_id,
                    "reason_codes": list(exc.reason_codes),
                    "error_code": exc.code,
                    "event_id": event.event_id,
                    "usage_revision_at_decision": snap.usage_revision,
                }
                ledger.put_idempotency(idem_key, decision)
                new_snap = ledger.build_snapshot(scope_id, now)
                result = ReserveDecision(
                    granted=False,
                    reservation=denied,
                    reservation_id=denied.reservation_id,
                    usage_revision=new_snap.usage_revision or "",
                    snapshot=new_snap,
                    reason_codes=tuple(exc.reason_codes),
                    error_code=exc.code,
                    replayed=False,
                    event_id=event.event_id,
                )
                return result, ledger

            record = {
                "reservation_id": reservation.reservation_id,
                "scope_id": scope_id,
                "state": ReservationState.HELD.value,
                "reservation": reservation.to_dict(),
                "reserved_amounts": amounts,
                "committed_amounts": {},
                "charged_amounts": {},
                "request_id": request_id,
                "attempt_id": attempt_id,
                "idempotency_key": idempotency_key,
                "owner_id": owner_id,
                "lease_id": lease_id,
                "fence": effective_fence,
                "estimate_id": estimate_id,
                "usage_revision": snap.usage_revision,
                "ttl_ms": ttl_ms,
                "created_at": reservation.created_at,
                "expires_at": reservation.expires_at,
                "dispatched": False,
                "dispatched_at": None,
            }
            ledger.put_reservation_record(record)
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.RESERVATION,
                    scope_id=scope_id,
                    occurred_at=_to_rfc3339(now),
                    request_id=bound_request_id,
                    reservation_id=reservation.reservation_id,
                    estimate_id=estimate_id,
                    units=vector,
                    reason_codes=("reserved",),
                    provenance=Provenance(
                        source=LimitSource.LOCAL_OBSERVATION,
                        observed_at=_to_rfc3339(now),
                        reason_codes=("reserved",),
                    ),
                )
            )
            decision = {
                "granted": True,
                "reservation": reservation.to_dict(),
                "reservation_id": reservation.reservation_id,
                "reason_codes": ["reserved"],
                "error_code": None,
                "event_id": event.event_id,
                "usage_revision_at_decision": snap.usage_revision,
            }
            ledger.put_idempotency(idem_key, decision)
            new_snap = ledger.build_snapshot(scope_id, now)
            result = ReserveDecision(
                granted=True,
                reservation=reservation,
                reservation_id=reservation.reservation_id,
                usage_revision=new_snap.usage_revision or "",
                snapshot=new_snap,
                reason_codes=("reserved",),
                error_code=None,
                replayed=False,
                event_id=event.event_id,
            )
            return result, ledger

        result, _ = self._cas(mutate)
        return result

    def _replay_decision(
        self,
        ledger: UsageLedger,
        prior: Mapping[str, Any],
        scope_id: str,
        now: datetime,
    ) -> ReserveDecision:
        reservation = None
        if prior.get("reservation"):
            reservation = UsageReservation.from_dict(prior["reservation"])
        snap = ledger.build_snapshot(scope_id, now)
        return ReserveDecision(
            granted=bool(prior.get("granted")),
            reservation=reservation,
            reservation_id=prior.get("reservation_id"),
            usage_revision=snap.usage_revision or "",
            snapshot=snap,
            reason_codes=tuple(prior.get("reason_codes") or ()),
            error_code=prior.get("error_code"),
            replayed=True,
            event_id=prior.get("event_id"),
        )

    # ------------------------------------------------------------------
    # Dispatch tracking
    # ------------------------------------------------------------------

    def mark_dispatched(self, reservation_id: str) -> UsageReservation:
        """Mark reservation as dispatched to the provider (affects cancel policy)."""

        def mutate(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    code=UsageErrorCode.RESERVATION_CONFLICT.value,
                    reason_codes=("unknown_reservation",),
                )
            self._assert_owner_live(record, ledger)
            if record.get("state") not in (
                ReservationState.HELD.value,
                ReservationState.PENDING.value,
            ):
                raise LedgerError(
                    "reservation %s is not held" % reservation_id,
                    code=UsageErrorCode.RESERVATION_CONFLICT.value,
                    reason_codes=("invalid_state",),
                )
            now = self.clock.now()
            record["dispatched"] = True
            record["dispatched_at"] = _to_rfc3339(now)
            ledger.put_reservation_record(record)
            reservation = UsageReservation.from_dict(record["reservation"])
            return reservation, ledger

        reservation, _ = self._cas(mutate)
        return reservation

    def _assert_owner_live(self, record: Mapping[str, Any], ledger: UsageLedger) -> None:
        """Expired/crashed owners cannot mutate; reclamation owns release."""

        expires = record.get("expires_at")
        if expires is not None:
            now_ms = datetime_to_ms(self.clock.now())
            if rfc3339_to_ms(expires) <= now_ms:
                raise LedgerError(
                    "reservation expired; reclaim instead of owner mutation",
                    code=UsageErrorCode.RESERVATION_CONFLICT.value,
                    reason_codes=("reservation_expired",),
                )
        if self._fence is not None and record.get("fence") is not None:
            if int(record["fence"]) > int(self._fence):
                from .store import StaleFenceError

                raise StaleFenceError(
                    "reservation fence %s exceeds coordinator fence %s"
                    % (record["fence"], self._fence)
                )

    # ------------------------------------------------------------------
    # Cancel / release / settle / commit
    # ------------------------------------------------------------------

    def cancel(
        self,
        reservation_id: str,
        *,
        reason: str = "cancelled",
    ) -> SettlementResult:
        """Cancel a reservation.

        Before dispatch: release all reserved usage.
        After dispatch: conservatively settle provider-chargeable dimensions
        at the reserved amounts (provider may still charge).
        """

        def mutate(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    code=UsageErrorCode.RESERVATION_CONFLICT.value,
                    reason_codes=("unknown_reservation",),
                )
            # Idempotent terminal.
            if record.get("state") in (
                ReservationState.RELEASED.value,
                ReservationState.COMMITTED.value,
                ReservationState.EXPIRED.value,
            ):
                charged = amounts_to_vector(record.get("charged_amounts") or {})
                snap = ledger.build_snapshot(record["scope_id"], self.clock.now())
                return (
                    SettlementResult(
                        reservation_id=reservation_id,
                        state=ReservationState(record["state"]),
                        charged=charged,
                        event_id=record.get("terminal_event_id") or "",
                        usage_revision=snap.usage_revision or "",
                        replayed=True,
                    ),
                    ledger,
                )
            self._assert_owner_live(record, ledger)
            now = self.clock.now()
            dispatched = bool(record.get("dispatched"))
            reserved = {str(k): int(v) for k, v in (record.get("reserved_amounts") or {}).items()}
            if not dispatched:
                charged: Dict[str, int] = {}
                kind = UsageEventKind.RELEASE
                state = ReservationState.RELEASED
                reason_codes = ("cancelled_before_dispatch", reason)
            else:
                charged = {
                    key: amount
                    for key, amount in reserved.items()
                    if key.split(":", 1)[0] in self._provider_chargeable
                }
                # Concurrent dimensions always release on cancel.
                charged = {
                    key: amount
                    for key, amount in charged.items()
                    if not key.startswith("concurrent_")
                }
                kind = UsageEventKind.COMMIT
                state = ReservationState.COMMITTED
                reason_codes = ("cancelled_after_dispatch", reason)

            record = copy.deepcopy(record)
            record["state"] = state.value
            record["charged_amounts"] = charged
            record["committed_amounts"] = dict(charged)
            reservation_dict = dict(record["reservation"])
            reservation_dict["state"] = state.value
            record["reservation"] = reservation_dict
            event = ledger.append_event(
                UsageEvent(
                    kind=kind,
                    scope_id=record["scope_id"],
                    occurred_at=_to_rfc3339(now),
                    request_id=record.get("reservation", {}).get("request_id"),
                    reservation_id=reservation_id,
                    estimate_id=record.get("estimate_id"),
                    units=amounts_to_vector(charged) if charged else UsageVector(),
                    reason_codes=tuple(
                        r.replace(" ", "_")[:64] for r in reason_codes
                    ),
                    provenance=Provenance(
                        source=LimitSource.LOCAL_OBSERVATION,
                        observed_at=_to_rfc3339(now),
                    ),
                )
            )
            record["terminal_event_id"] = event.event_id
            ledger.put_reservation_record(record)
            ledger.set_stream_settled(reservation_id, charged)
            snap = ledger.build_snapshot(record["scope_id"], now)
            return (
                SettlementResult(
                    reservation_id=reservation_id,
                    state=state,
                    charged=amounts_to_vector(charged) if charged else UsageVector(),
                    event_id=event.event_id or "",
                    usage_revision=snap.usage_revision or "",
                ),
                ledger,
            )

        result, _ = self._cas(mutate)
        return result

    def release(self, reservation_id: str, *, reason: str = "released") -> SettlementResult:
        """Release all remaining held capacity (unused after commit path)."""

        def mutate(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    reason_codes=("unknown_reservation",),
                )
            if record.get("state") == ReservationState.RELEASED.value:
                charged = amounts_to_vector(record.get("charged_amounts") or {})
                snap = ledger.build_snapshot(record["scope_id"], self.clock.now())
                return (
                    SettlementResult(
                        reservation_id=reservation_id,
                        state=ReservationState.RELEASED,
                        charged=charged,
                        event_id=record.get("terminal_event_id") or "",
                        usage_revision=snap.usage_revision or "",
                        replayed=True,
                    ),
                    ledger,
                )
            self._assert_owner_live(record, ledger)
            now = self.clock.now()
            # Keep already committed/charged; release only residual hold.
            charged = {
                str(k): int(v) for k, v in (record.get("charged_amounts") or {}).items()
            }
            record = copy.deepcopy(record)
            record["state"] = ReservationState.RELEASED.value
            reservation_dict = dict(record["reservation"])
            reservation_dict["state"] = ReservationState.RELEASED.value
            record["reservation"] = reservation_dict
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.RELEASE,
                    scope_id=record["scope_id"],
                    occurred_at=_to_rfc3339(now),
                    reservation_id=reservation_id,
                    units=UsageVector(),
                    reason_codes=(reason.replace(" ", "_")[:64],),
                    provenance=Provenance(
                        source=LimitSource.LOCAL_OBSERVATION,
                        observed_at=_to_rfc3339(now),
                    ),
                )
            )
            record["terminal_event_id"] = event.event_id
            ledger.put_reservation_record(record)
            snap = ledger.build_snapshot(record["scope_id"], now)
            return (
                SettlementResult(
                    reservation_id=reservation_id,
                    state=ReservationState.RELEASED,
                    charged=amounts_to_vector(charged) if charged else UsageVector(),
                    event_id=event.event_id or "",
                    usage_revision=snap.usage_revision or "",
                ),
                ledger,
            )

        result, _ = self._cas(mutate)
        return result

    def settle_stream(
        self,
        reservation_id: str,
        cumulative: UsageVector | Mapping[str, Any],
    ) -> SettlementResult:
        """Monotonic stream settlement: cumulative amounts must not decrease."""

        vector = (
            cumulative
            if isinstance(cumulative, UsageVector)
            else UsageVector.from_dict(cumulative)
        )
        new_amounts = vector_to_amounts(vector)

        def mutate(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    reason_codes=("unknown_reservation",),
                )
            self._assert_owner_live(record, ledger)
            if record.get("state") not in (
                ReservationState.HELD.value,
                ReservationState.PENDING.value,
            ):
                raise LedgerError(
                    "stream settle requires held reservation",
                    reason_codes=("invalid_state",),
                )
            prior = ledger.stream_settled(reservation_id)
            for key, value in new_amounts.items():
                if value < prior.get(key, 0):
                    raise LedgerError(
                        "stream settlement must be monotonic for %s" % key,
                        code=UsageErrorCode.NEGATIVE_VALUE.value,
                        reason_codes=("non_monotonic_stream",),
                    )
            # Cap at reserved.
            reserved = {
                str(k): int(v) for k, v in (record.get("reserved_amounts") or {}).items()
            }
            capped = {}
            for key, value in new_amounts.items():
                cap = reserved.get(key, value)
                capped[key] = min(value, cap)
            # Merge keys only increasing.
            merged = dict(prior)
            merged.update(capped)
            for key, value in prior.items():
                merged[key] = max(value, merged.get(key, 0))
            ledger.set_stream_settled(reservation_id, merged)
            record = copy.deepcopy(record)
            record["committed_amounts"] = dict(merged)
            ledger.put_reservation_record(record)
            now = self.clock.now()
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.STREAM_SETTLEMENT,
                    scope_id=record["scope_id"],
                    occurred_at=_to_rfc3339(now),
                    reservation_id=reservation_id,
                    units=amounts_to_vector(merged),
                    reason_codes=("stream_settlement",),
                    provenance=Provenance(
                        source=LimitSource.LOCAL_OBSERVATION,
                        observed_at=_to_rfc3339(now),
                    ),
                )
            )
            snap = ledger.build_snapshot(record["scope_id"], now)
            return (
                SettlementResult(
                    reservation_id=reservation_id,
                    state=ReservationState.HELD,
                    charged=amounts_to_vector(merged),
                    event_id=event.event_id or "",
                    usage_revision=snap.usage_revision or "",
                ),
                ledger,
            )

        result, _ = self._cas(mutate)
        return result

    def commit(
        self,
        reservation_id: str,
        actual: Optional[UsageVector | Mapping[str, Any]] = None,
        *,
        observation_id: Optional[str] = None,
        release_unused: bool = True,
    ) -> SettlementResult:
        """Commit provider-reported actual usage and optionally release unused."""

        def mutate(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    reason_codes=("unknown_reservation",),
                )
            if record.get("state") == ReservationState.COMMITTED.value:
                charged = amounts_to_vector(record.get("charged_amounts") or {})
                snap = ledger.build_snapshot(record["scope_id"], self.clock.now())
                return (
                    SettlementResult(
                        reservation_id=reservation_id,
                        state=ReservationState.COMMITTED,
                        charged=charged,
                        event_id=record.get("terminal_event_id") or "",
                        usage_revision=snap.usage_revision or "",
                        replayed=True,
                    ),
                    ledger,
                )
            self._assert_owner_live(record, ledger)
            now = self.clock.now()
            reserved = {
                str(k): int(v) for k, v in (record.get("reserved_amounts") or {}).items()
            }
            settled = ledger.stream_settled(reservation_id)
            if actual is None:
                charged = dict(settled) if settled else dict(reserved)
            else:
                vector = (
                    actual
                    if isinstance(actual, UsageVector)
                    else UsageVector.from_dict(actual)
                )
                charged = vector_to_amounts(vector)
            # Conservative: never charge less than already stream-settled.
            for key, value in settled.items():
                charged[key] = max(int(charged.get(key, 0)), value)
            if not release_unused:
                # Keep residual reserved as charged.
                for key, value in reserved.items():
                    charged[key] = max(int(charged.get(key, 0)), value)

            record = copy.deepcopy(record)
            record["state"] = ReservationState.COMMITTED.value
            record["charged_amounts"] = charged
            record["committed_amounts"] = dict(charged)
            reservation_dict = dict(record["reservation"])
            reservation_dict["state"] = ReservationState.COMMITTED.value
            record["reservation"] = reservation_dict
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.COMMIT,
                    scope_id=record["scope_id"],
                    occurred_at=_to_rfc3339(now),
                    reservation_id=reservation_id,
                    observation_id=observation_id,
                    units=amounts_to_vector(charged) if charged else UsageVector(),
                    reason_codes=("committed",),
                    provenance=Provenance(
                        source=LimitSource.LOCAL_OBSERVATION,
                        observed_at=_to_rfc3339(now),
                    ),
                )
            )
            record["terminal_event_id"] = event.event_id
            ledger.put_reservation_record(record)
            ledger.set_stream_settled(reservation_id, charged)
            snap = ledger.build_snapshot(record["scope_id"], now)
            return (
                SettlementResult(
                    reservation_id=reservation_id,
                    state=ReservationState.COMMITTED,
                    charged=amounts_to_vector(charged) if charged else UsageVector(),
                    event_id=event.event_id or "",
                    usage_revision=snap.usage_revision or "",
                ),
                ledger,
            )

        result, _ = self._cas(mutate)
        return result

    def timeout(
        self,
        reservation_id: str,
        *,
        after_dispatch: Optional[bool] = None,
    ) -> SettlementResult:
        """Timeout handling mirrors cancel policy with explicit reason."""

        def mutate_mark(ledger: UsageLedger):
            record = ledger.get_reservation_record(reservation_id)
            if record is None:
                raise LedgerError(
                    "unknown reservation %s" % reservation_id,
                    reason_codes=("unknown_reservation",),
                )
            if after_dispatch is True and not record.get("dispatched"):
                record = copy.deepcopy(record)
                record["dispatched"] = True
                record["dispatched_at"] = _to_rfc3339(self.clock.now())
                ledger.put_reservation_record(record)
            return None, ledger

        if after_dispatch is True:
            self._cas(mutate_mark)
        return self.cancel(reservation_id, reason="timeout")

    # ------------------------------------------------------------------
    # Batch settlement
    # ------------------------------------------------------------------

    def settle_batch(
        self,
        *,
        batch_id: str,
        scope_id: str,
        shared_overhead: UsageVector | Mapping[str, Any],
        members: Mapping[str, UsageVector | Mapping[str, Any]],
        request_id: str,
        owner_id: str,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        """Charge shared overhead once and each member exactly once."""

        overhead_vec = (
            shared_overhead
            if isinstance(shared_overhead, UsageVector)
            else UsageVector.from_dict(shared_overhead)
        )
        member_vecs = {
            mid: (vec if isinstance(vec, UsageVector) else UsageVector.from_dict(vec))
            for mid, vec in members.items()
        }

        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            state = ledger.batch_charge_state(batch_id)
            members_charged = dict(state.get("members") or {})
            events = []
            charged_total: Dict[str, int] = {}

            if not state.get("overhead_charged"):
                amounts = vector_to_amounts(overhead_vec)
                # Record as a committed synthetic charge via reservation-less event.
                event = ledger.append_event(
                    UsageEvent(
                        kind=UsageEventKind.COMMIT,
                        scope_id=scope_id,
                        occurred_at=_to_rfc3339(now),
                        request_id="%s#batch-overhead" % request_id,
                        units=overhead_vec,
                        reason_codes=("batch_shared_overhead", batch_id[:48]),
                        provenance=Provenance(
                            source=LimitSource.LOCAL_OBSERVATION,
                            observed_at=_to_rfc3339(now),
                        ),
                    )
                )
                events.append(event.event_id)
                # Materialize as a committed reservation record for occupancy.
                rid = stable_id(
                    "ubatch", batch_id, "overhead", scope_id, idempotency_key
                )
                reservation = UsageReservation(
                    scope_id=scope_id,
                    reserved=overhead_vec,
                    state=ReservationState.COMMITTED,
                    request_id="%s#batch-overhead" % request_id,
                    idempotency_key=stable_id("ubatchidem", batch_id, "overhead"),
                    owner_id=owner_id,
                    created_at=_to_rfc3339(now),
                    expires_at=_to_rfc3339(now),
                    reason_codes=("batch_shared_overhead",),
                )
                # Override identity: use constructed reservation fields.
                # UsageReservation recomputes id from fields.
                rid = reservation.reservation_id
                ledger.put_reservation_record(
                    {
                        "reservation_id": rid,
                        "scope_id": scope_id,
                        "state": ReservationState.COMMITTED.value,
                        "reservation": reservation.to_dict(),
                        "reserved_amounts": amounts,
                        "committed_amounts": amounts,
                        "charged_amounts": amounts,
                        "request_id": request_id,
                        "attempt_id": "batch-overhead",
                        "idempotency_key": idempotency_key,
                        "owner_id": owner_id,
                        "created_at": reservation.created_at,
                        "expires_at": reservation.expires_at,
                        "dispatched": True,
                        "batch_id": batch_id,
                        "batch_role": "overhead",
                        "terminal_event_id": event.event_id,
                    }
                )
                charged_total = merge_amounts(charged_total, amounts)
                state["overhead_charged"] = True
            else:
                # Already charged overhead — do not double charge.
                pass

            for member_id, vec in sorted(member_vecs.items()):
                if members_charged.get(member_id):
                    continue
                amounts = vector_to_amounts(vec)
                event = ledger.append_event(
                    UsageEvent(
                        kind=UsageEventKind.COMMIT,
                        scope_id=scope_id,
                        occurred_at=_to_rfc3339(now),
                        request_id="%s#member=%s" % (request_id, member_id),
                        units=vec,
                        reason_codes=("batch_member",),
                        provenance=Provenance(
                            source=LimitSource.LOCAL_OBSERVATION,
                            observed_at=_to_rfc3339(now),
                        ),
                    )
                )
                events.append(event.event_id)
                reservation = UsageReservation(
                    scope_id=scope_id,
                    reserved=vec,
                    state=ReservationState.COMMITTED,
                    request_id="%s#member=%s" % (request_id, member_id),
                    idempotency_key=stable_id(
                        "ubatchidem", batch_id, "member", member_id
                    ),
                    owner_id=owner_id,
                    created_at=_to_rfc3339(now),
                    expires_at=_to_rfc3339(now),
                    reason_codes=("batch_member",),
                )
                rid = reservation.reservation_id
                ledger.put_reservation_record(
                    {
                        "reservation_id": rid,
                        "scope_id": scope_id,
                        "state": ReservationState.COMMITTED.value,
                        "reservation": reservation.to_dict(),
                        "reserved_amounts": amounts,
                        "committed_amounts": amounts,
                        "charged_amounts": amounts,
                        "request_id": request_id,
                        "attempt_id": "member:%s" % member_id,
                        "idempotency_key": idempotency_key,
                        "owner_id": owner_id,
                        "created_at": reservation.created_at,
                        "expires_at": reservation.expires_at,
                        "dispatched": True,
                        "batch_id": batch_id,
                        "batch_role": "member",
                        "batch_member_id": member_id,
                        "terminal_event_id": event.event_id,
                    }
                )
                members_charged[member_id] = True
                charged_total = merge_amounts(charged_total, amounts)

            state["members"] = members_charged
            ledger.set_batch_charge_state(batch_id, state)
            snap = ledger.build_snapshot(scope_id, now)
            return (
                {
                    "batch_id": batch_id,
                    "overhead_charged": True,
                    "members_charged": sorted(
                        mid for mid, ok in members_charged.items() if ok
                    ),
                    "charged": amounts_to_vector(charged_total).to_dict()
                    if charged_total
                    else UsageVector().to_dict(),
                    "event_ids": events,
                    "usage_revision": snap.usage_revision,
                },
                ledger,
            )

        result, _ = self._cas(mutate)
        return result

    # ------------------------------------------------------------------
    # Corrections / refunds / reset / reclaim
    # ------------------------------------------------------------------

    def correct(
        self,
        scope_id: str,
        *,
        supersedes_event_id: str,
        units: UsageVector | Mapping[str, Any],
        reason: str = "correction",
        reservation_id: Optional[str] = None,
    ) -> UsageEvent:
        """Append a correction that references a prior event (never rewrites)."""

        vector = units if isinstance(units, UsageVector) else UsageVector.from_dict(units)

        def mutate(ledger: UsageLedger):
            events = ledger.document.get("events") or []
            found = any(e.get("event_id") == supersedes_event_id for e in events)
            # Also allow corrections referencing pre-compaction events via corrections map.
            corrections = ledger.document.get("corrections") or {}
            if not found and supersedes_event_id not in corrections:
                # Still allow if checkpoint may have dropped it — record reference.
                pass
            now = self.clock.now()
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.CORRECTION,
                    scope_id=scope_id,
                    occurred_at=_to_rfc3339(now),
                    reservation_id=reservation_id,
                    supersedes_event_id=supersedes_event_id,
                    units=vector,
                    reason_codes=(reason.replace(" ", "_")[:64],),
                    provenance=Provenance(
                        source=LimitSource.RECONCILED,
                        observed_at=_to_rfc3339(now),
                        reason_codes=("correction",),
                    ),
                )
            )
            if reservation_id:
                record = ledger.get_reservation_record(reservation_id)
                if record is not None:
                    record = copy.deepcopy(record)
                    record["charged_amounts"] = vector_to_amounts(vector)
                    record["committed_amounts"] = vector_to_amounts(vector)
                    ledger.put_reservation_record(record)
            return event, ledger

        event, _ = self._cas(mutate)
        return event

    def refund(
        self,
        scope_id: str,
        units: UsageVector | Mapping[str, Any],
        *,
        reservation_id: Optional[str] = None,
        reason: str = "provider_refund",
    ) -> UsageEvent:
        vector = units if isinstance(units, UsageVector) else UsageVector.from_dict(units)

        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            # Refund reduces charged amounts on the reservation when present.
            if reservation_id:
                record = ledger.get_reservation_record(reservation_id)
                if record is not None:
                    record = copy.deepcopy(record)
                    charged = {
                        str(k): int(v)
                        for k, v in (record.get("charged_amounts") or {}).items()
                    }
                    refund_amounts = vector_to_amounts(vector)
                    for key, value in refund_amounts.items():
                        charged[key] = max(0, charged.get(key, 0) - value)
                    record["charged_amounts"] = charged
                    ledger.put_reservation_record(record)
            event = ledger.append_event(
                UsageEvent(
                    kind=UsageEventKind.REFUND,
                    scope_id=scope_id,
                    occurred_at=_to_rfc3339(now),
                    reservation_id=reservation_id,
                    units=vector,
                    reason_codes=(reason.replace(" ", "_")[:64],),
                    provenance=Provenance(
                        source=LimitSource.RECONCILED,
                        observed_at=_to_rfc3339(now),
                        reason_codes=("refund",),
                    ),
                )
            )
            return event, ledger

        event, _ = self._cas(mutate)
        return event

    def reclaim_expired(self, scope_id: Optional[str] = None) -> Dict[str, Any]:
        """Reclaim capacity from expired/crashed owners without double-release."""

        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            now_ms = datetime_to_ms(now)
            reclaimed = []
            for record in ledger.iter_reservation_records(scope_id):
                if record.get("state") not in (
                    ReservationState.HELD.value,
                    ReservationState.PENDING.value,
                ):
                    continue
                expires = record.get("expires_at")
                if expires is None or rfc3339_to_ms(expires) > now_ms:
                    continue
                # Expired: if never dispatched, release fully; if dispatched,
                # conservatively charge provider-chargeable dimensions.
                reserved = {
                    str(k): int(v)
                    for k, v in (record.get("reserved_amounts") or {}).items()
                }
                dispatched = bool(record.get("dispatched"))
                if not dispatched:
                    charged: Dict[str, int] = {}
                    state = ReservationState.EXPIRED
                    kind = UsageEventKind.EXPIRY_RECOVERY
                else:
                    charged = {
                        key: amount
                        for key, amount in reserved.items()
                        if key.split(":", 1)[0] in self._provider_chargeable
                        and not key.startswith("concurrent_")
                    }
                    state = ReservationState.EXPIRED
                    kind = UsageEventKind.EXPIRY_RECOVERY
                record = copy.deepcopy(record)
                # Guard against double-release races: only transition from held.
                current = ledger.get_reservation_record(record["reservation_id"])
                if current is None or current.get("state") not in (
                    ReservationState.HELD.value,
                    ReservationState.PENDING.value,
                ):
                    continue
                record["state"] = state.value
                record["charged_amounts"] = charged
                record["committed_amounts"] = dict(charged)
                reservation_dict = dict(record["reservation"])
                reservation_dict["state"] = state.value
                record["reservation"] = reservation_dict
                event = ledger.append_event(
                    UsageEvent(
                        kind=kind,
                        scope_id=record["scope_id"],
                        occurred_at=_to_rfc3339(now),
                        reservation_id=record["reservation_id"],
                        units=amounts_to_vector(charged) if charged else UsageVector(),
                        reason_codes=("expiry_reclamation",),
                        provenance=Provenance(
                            source=LimitSource.LOCAL_OBSERVATION,
                            observed_at=_to_rfc3339(now),
                        ),
                    )
                )
                record["terminal_event_id"] = event.event_id
                ledger.put_reservation_record(record)
                reclaimed.append(record["reservation_id"])
            return {"reclaimed": reclaimed, "count": len(reclaimed)}, ledger

        result, _ = self._cas(mutate)
        return result

    def reset(
        self,
        scope_id: str,
        *,
        reason: str = "admin_reset",
        expected_usage_revision: Optional[str] = None,
    ) -> UsageEvent:
        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            if expected_usage_revision is not None:
                snap = ledger.build_snapshot(scope_id, now)
                if snap.usage_revision != expected_usage_revision:
                    raise StaleSnapshot()
            event = ledger.reset_scope(scope_id, now=now, reason=reason)
            return event, ledger

        event, _ = self._cas(mutate)
        return event

    def compact(self, *, retain_events: int = 0) -> Dict[str, Any]:
        def mutate(ledger: UsageLedger):
            receipt = ledger.compact(retain_events=retain_events)
            return receipt, ledger

        receipt, _ = self._cas(mutate)
        return receipt

    def checkpoint(self) -> Dict[str, Any]:
        return self._store.checkpoint()

    def migrate(self, *, target_schema_version: str = "1.0") -> Dict[str, Any]:
        document = self._store.read()
        migrated = migrate_document(
            document, target_schema_version=target_schema_version
        )

        def mutate(ledger: UsageLedger):
            # Replace document fields from migration result.
            ledger._doc.clear()
            ledger._doc.update(copy.deepcopy(migrated))
            return {"migrated": True, "schema_version": target_schema_version}, ledger

        result, _ = self._cas(mutate)
        return result

    def recovery_view(self) -> Dict[str, Any]:
        return read_only_recovery_view(self._store.read())

    def append_observation(
        self,
        scope_id: str,
        *,
        kind: UsageEventKind,
        units: Optional[UsageVector | Mapping[str, Any]] = None,
        reservation_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        reason_codes: Sequence[str] = (),
        cooldown_until: Optional[str] = None,
        limits_update: Optional[Sequence[UsageLimit | Mapping[str, Any]]] = None,
    ) -> UsageEvent:
        """Append success/failure observation; may tighten cooldowns/limits."""

        if kind not in (
            UsageEventKind.OBSERVATION_SUCCESS,
            UsageEventKind.OBSERVATION_FAILURE,
        ):
            raise LedgerError(
                "append_observation requires observation kind",
                reason_codes=("invalid_kind",),
            )
        vector = UsageVector()
        if units is not None:
            vector = (
                units if isinstance(units, UsageVector) else UsageVector.from_dict(units)
            )

        def mutate(ledger: UsageLedger):
            now = self.clock.now()
            if limits_update:
                parsed = [
                    item if isinstance(item, UsageLimit) else UsageLimit.from_dict(item)
                    for item in limits_update
                ]
                # Merge with existing by limit_id.
                existing = {lim.limit_id: lim for lim in ledger.get_limits(scope_id)}
                for lim in parsed:
                    existing[lim.limit_id] = lim
                ledger.set_limits(scope_id, list(existing.values()))
            if cooldown_until:
                ledger.set_cooldown(scope_id, cooldown_until)
            event = ledger.append_event(
                UsageEvent(
                    kind=kind,
                    scope_id=scope_id,
                    occurred_at=_to_rfc3339(now),
                    request_id=request_id,
                    reservation_id=reservation_id,
                    observation_id=observation_id,
                    units=vector,
                    reason_codes=tuple(reason_codes),
                    provenance=Provenance(
                        source=LimitSource.RESPONSE_BODY
                        if kind is UsageEventKind.OBSERVATION_SUCCESS
                        else LimitSource.ERROR,
                        observed_at=_to_rfc3339(now),
                    ),
                )
            )
            return event, ledger

        event, _ = self._cas(mutate)
        return event


__all__ = [
    "ATOMIC_USAGE_LEDGER_REQUIREMENT_ID",
    "DEFAULT_CAS_RETRIES",
    "DEFAULT_RESERVATION_TTL_MS",
    "PROVIDER_CHARGEABLE_DEFAULT",
    "ReserveDecision",
    "SettlementResult",
    "UsageCoordinator",
]
