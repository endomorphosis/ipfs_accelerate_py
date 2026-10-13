"""Atomic endpoint usage admission for the existing physical batch scheduler.

The endpoint coordinator owns quota. Host leases still come from the supplied
resource admission adapter. This module neither authorizes from a snapshot nor
creates a second scheduler. Without a trusted usage extractor, settlement uses
declared input/output ceilings and is explicitly an estimate, not a provider bill.

Reservation expiry is crash recovery, not provider cancellation. Configure a TTL
covering the provider's bounded physical timeout and settlement margin; logical
subscriber deadlines alone do not terminate a callback or renew its reservation.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

from ipfs_accelerate_py.endpoint_usage.coordinator import SettlementResult, UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import is_pseudonym
from ipfs_accelerate_py.endpoint_usage.schema import (
    QuantityKind,
    ReservationState,
    UsageDimension,
    UsageReservation,
    UsageVector,
)

from .provider_batch_scheduler import (
    ProviderAdmission,
    ProviderBatchAdmissionGrant,
    ProviderBatchCapacity,
    ProviderBatchKey,
    ProviderBatchRequest,
)


PHYSICAL_BATCH_USAGE_REQUIREMENT_ID = "requirement:physical-batch-atomic-usage.v1"
UsageExtractor = Callable[[Sequence[ProviderBatchRequest], Any], UsageVector]


class PhysicalBatchUsageError(RuntimeError):
    """Usage lifecycle could not safely authorize or settle a physical call."""


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _estimate(
    requests: Sequence[ProviderBatchRequest], overhead: int, *, concurrent: bool
) -> UsageVector:
    inputs = sum(item.context_limit for item in requests) + overhead
    outputs = sum(item.token_budget for item in requests)
    values = dict(
        requests=1,
        batch_items=len(requests),
        input_tokens=inputs,
        output_tokens=outputs,
        total_tokens=inputs + outputs,
    )
    if concurrent:
        values["concurrent_requests"] = 1
    return UsageVector.of(**values)


def _measured(value: UsageVector, member_count: int) -> UsageVector:
    if not isinstance(value, UsageVector):
        raise PhysicalBatchUsageError("usage extractor must return a UsageVector")
    allowed = {
        UsageDimension.INPUT_TOKENS,
        UsageDimension.OUTPUT_TOKENS,
        UsageDimension.TOTAL_TOKENS,
    }
    if {item.dimension for item in value.entries} != allowed:
        raise PhysicalBatchUsageError("usage extractor must supply input, output, and total tokens")
    if any(item.amount.kind is not QuantityKind.FINITE for item in value.entries):
        raise PhysicalBatchUsageError("measured token usage must be finite")
    amounts = {item.dimension.value: int(item.amount) for item in value.entries}
    if amounts["total_tokens"] != amounts["input_tokens"] + amounts["output_tokens"]:
        raise PhysicalBatchUsageError("measured total tokens must equal input plus output")
    # Concurrency is a hold, never a consumed quantity. Physical cardinality is
    # known locally and cannot be changed by a provider's usage observation.
    return UsageVector.of(requests=1, batch_items=member_count, **amounts)


class _PhysicalReservation:
    def __init__(
        self,
        owner: "PhysicalBatchUsageAdmission",
        host: ProviderBatchAdmissionGrant,
        requests: Sequence[ProviderBatchRequest],
        reservation_id: str,
        receipt: dict[str, Any],
    ) -> None:
        self.owner = owner
        self.host = host
        self.requests = tuple(requests)
        self.reservation_id = reservation_id
        self.receipt = receipt
        self.live: tuple[ProviderBatchRequest, ...] = ()
        self.dispatched = False
        self.finished = False
        self.settlement_attempted = False
        self.closed = False
        self.lock = threading.RLock()

    def _validate_settlement(self, result: SettlementResult, *, closed: bool = False) -> None:
        allowed = {ReservationState.COMMITTED}
        if closed:
            allowed.update((ReservationState.RELEASED, ReservationState.EXPIRED))
        if (
            not isinstance(result, SettlementResult)
            or result.reservation_id != self.reservation_id
            or result.state not in allowed
            or not isinstance(result.event_id, str)
            or not result.event_id
            or any(item.amount.kind is not QuantityKind.FINITE for item in result.charged.entries)
        ):
            raise PhysicalBatchUsageError("unconfirmed physical usage settlement")

    def before_dispatch(self, requests: Sequence[ProviderBatchRequest]) -> None:
        with self.lock:
            if self.closed or self.dispatched:
                raise PhysicalBatchUsageError("physical reservation cannot dispatch twice")
            selected = tuple(requests)
            original = {item.request_id: item.execution_fingerprint for item in self.requests}
            if (
                not selected
                or len({item.request_id for item in selected}) != len(selected)
                or any(original.get(item.request_id) != item.execution_fingerprint for item in selected)
            ):
                raise PhysicalBatchUsageError("dispatch members differ from reserved work")
            if self.host.before_dispatch is not None:
                self.host.before_dispatch(selected)
            # Mark failure is terminal to this dispatch: never proceed after an
            # expired reservation, a stale fence, or an ambiguous store write.
            try:
                acknowledgement = self.owner.coordinator.mark_dispatched(self.reservation_id)
                if (
                    not isinstance(acknowledgement, UsageReservation)
                    or acknowledgement.reservation_id != self.reservation_id
                    or acknowledgement.state not in (ReservationState.HELD, ReservationState.PENDING)
                ):
                    raise PhysicalBatchUsageError("invalid usage dispatch acknowledgement")
            except Exception as exc:
                self.receipt["state"] = "dispatch_mark_failed"
                raise PhysicalBatchUsageError("usage dispatch mark failed") from exc
            self.live = selected
            self.dispatched = True
            self.receipt.update(
                state="dispatched",
                dispatched_member_count=len(selected),
                dispatched_request_digests=[_digest(item.request_id) for item in selected],
            )

    def _settle(self, raw: Any, error: BaseException | None) -> None:
        self.settlement_attempted = True
        try:
            if error is not None:
                result = self.owner.coordinator.commit(
                    self.reservation_id,
                    actual=_estimate(self.live, self.owner.shared_overhead_tokens, concurrent=False),
                )
                basis = "conservative_live_ceiling"
            else:
                if self.owner.usage_extractor is None:
                    units = _estimate(self.live, self.owner.shared_overhead_tokens, concurrent=False)
                    basis = "estimated_ceiling"
                else:
                    units = _measured(self.owner.usage_extractor(self.live, raw), len(self.live))
                    basis = "provider_reported"
                result = self.owner.coordinator.commit(self.reservation_id, actual=units)
            self._validate_settlement(result)
            self.finished = True
            self.receipt.update(
                state=result.state.value,
                settlement_basis=basis,
                settled=result.charged.to_dict(),
                event_id=result.event_id,
                usage_revision=result.usage_revision,
            )
        except Exception as exc:
            # A failed store response may follow a successful write. Neither
            # release nor an invented receipt is evidence of settlement.
            self.receipt.update(state="settlement_pending", settlement_basis="unconfirmed")
            raise PhysicalBatchUsageError("physical usage settlement requires reconciliation") from exc

    def after_dispatch(
        self,
        requests: Sequence[ProviderBatchRequest],
        raw: Any,
        error: BaseException | None,
    ) -> None:
        with self.lock:
            if not self.dispatched or self.settlement_attempted:
                raise PhysicalBatchUsageError("physical usage settlement called outside dispatch")
            if tuple(item.execution_fingerprint for item in requests) != tuple(
                item.execution_fingerprint for item in self.live
            ):
                raise PhysicalBatchUsageError("settlement members differ from dispatched work")
            try:
                self._settle(raw, error)
            finally:
                # Preserve an existing host lifecycle hook even when endpoint
                # settlement fails. Its release hook runs separately in close.
                if self.host.after_dispatch is not None:
                    self.host.after_dispatch(requests, raw, error)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if not self.finished and not self.settlement_attempted:
                    # The coordinator knows whether an ambiguous mark reached
                    # durable storage; cancel applies its corresponding policy.
                    result = self.owner.coordinator.cancel(
                        self.reservation_id, reason="physical_batch_closed"
                    )
                    self._validate_settlement(result, closed=True)
                    self.finished = True
                    self.receipt.update(
                        state=result.state.value,
                        settlement_basis=(
                            "conservative_reserved_ceiling"
                            if result.state is ReservationState.COMMITTED else "undispatched"
                        ),
                        settled=result.charged.to_dict(),
                        event_id=result.event_id,
                        usage_revision=result.usage_revision,
                    )
            except Exception as exc:
                self.receipt.update(state="settlement_pending", settlement_basis="unconfirmed")
                raise PhysicalBatchUsageError("physical usage cleanup requires reconciliation") from exc
            finally:
                self.host.release_once()


class PhysicalBatchUsageAdmission:
    """Enforce one real endpoint reservation per physical batch invocation.

    ``scope_supplier`` is trusted deployment configuration, not request input.
    Every request must name that exact scope, which also participates in batch
    compatibility. ``context_limit`` bounds input and ``token_budget`` bounds
    output for each unique physical execution. The explicit aggregate ceiling
    prevents accidental unbounded requests; configured coordinator limits remain
    authoritative across schedulers sharing its store.

    ``usage_extractor`` is a trusted adapter returning complete finite input,
    output, and total token counts for the entire physical response, including
    shared overhead. Raw responses are never retained in usage receipts.
    """

    requirement_id = PHYSICAL_BATCH_USAGE_REQUIREMENT_ID

    def __init__(
        self,
        coordinator: UsageCoordinator,
        *,
        scope_supplier: Callable[[str], str],
        max_batch_tokens: int,
        reservation_ttl_ms: int,
        settlement_margin_ms: int = 5_000,
        host_admission: ProviderAdmission | None = None,
        shared_overhead_tokens: int = 0,
        usage_extractor: UsageExtractor | None = None,
        owner_id: str = "supervisor-physical-batch",
        retry_initial_ms: int = 100,
        retry_max_ms: int = 1_000,
        backoff_scope_limit: int = 128,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(coordinator, UsageCoordinator):
            raise TypeError("coordinator must be a real UsageCoordinator")
        if not callable(scope_supplier):
            raise TypeError("scope_supplier must be trusted callable configuration")
        if usage_extractor is not None and not callable(usage_extractor):
            raise TypeError("usage_extractor must be callable")
        if not isinstance(owner_id, str) or not owner_id.strip() or len(owner_id.encode("utf-8")) > 128:
            raise ValueError("owner_id must be bounded nonempty text")
        self.coordinator = coordinator
        self.scope_supplier = scope_supplier
        self.max_batch_tokens = _integer(max_batch_tokens, "max_batch_tokens")
        self.reservation_ttl_ms = _integer(reservation_ttl_ms, "reservation_ttl_ms")
        self.settlement_margin_ms = _integer(settlement_margin_ms, "settlement_margin_ms")
        self.shared_overhead_tokens = _integer(shared_overhead_tokens, "shared_overhead_tokens", minimum=0)
        self.host_admission = host_admission
        self.usage_extractor = usage_extractor
        self.owner_id = owner_id
        self.retry_initial_ms = _integer(retry_initial_ms, "retry_initial_ms")
        self.retry_max_ms = _integer(retry_max_ms, "retry_max_ms")
        if self.retry_initial_ms > self.retry_max_ms or self.retry_max_ms > 1_000:
            raise ValueError("retry intervals must satisfy initial <= max <= 1000 ms")
        self.backoff_scope_limit = _integer(backoff_scope_limit, "backoff_scope_limit")
        self._clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._backoff: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._backoff_lock = threading.Lock()

    def _ready(self, scope: str) -> bool:
        with self._backoff_lock:
            prior = self._backoff.get(scope)
            if prior is None:
                return True
            self._backoff.move_to_end(scope)
            return self._clock_ms() >= prior[0]

    def _defer(self, scope: str) -> None:
        with self._backoff_lock:
            prior = self._backoff.get(scope)
            delay = self.retry_initial_ms if prior is None else min(self.retry_max_ms, prior[1] * 2)
            self._backoff[scope] = (self._clock_ms() + delay, delay)
            self._backoff.move_to_end(scope)
            while len(self._backoff) > self.backoff_scope_limit:
                self._backoff.popitem(last=False)

    def _clear_backoff(self, scope: str) -> None:
        with self._backoff_lock:
            self._backoff.pop(scope, None)

    def __call__(
        self,
        key: ProviderBatchKey,
        requests: Sequence[ProviderBatchRequest],
        capacity: ProviderBatchCapacity,
    ) -> ProviderBatchAdmissionGrant:
        selected = tuple(requests)
        try:
            scope = self.scope_supplier(key.provider_id)
        except Exception:
            return ProviderBatchAdmissionGrant(False, reason="usage_scope_unavailable")
        if (
            not isinstance(scope, str)
            or not scope.strip()
            or not is_pseudonym(scope)
            or not selected
            or any(item.usage_scope_id != scope or item.batch_key != key for item in selected)
            or len({item.execution_fingerprint for item in selected}) != len(selected)
            or len({item.request_id for item in selected}) != len(selected)
        ):
            return ProviderBatchAdmissionGrant(False, reason="usage_scope_or_members_invalid", retryable=False)
        if any(item.context_limit <= 0 or item.token_budget <= 0 for item in selected):
            return ProviderBatchAdmissionGrant(False, reason="usage_token_bounds_required", retryable=False)
        if any(item.timeout_ms <= 0 for item in selected):
            return ProviderBatchAdmissionGrant(False, reason="usage_dispatch_timeout_required", retryable=False)
        if self.reservation_ttl_ms <= max(item.timeout_ms for item in selected) + self.settlement_margin_ms:
            return ProviderBatchAdmissionGrant(False, reason="usage_reservation_ttl_insufficient", retryable=False)
        requested = _estimate(selected, self.shared_overhead_tokens, concurrent=True)
        if int(requested.get(UsageDimension.TOTAL_TOKENS).amount) > self.max_batch_tokens:
            return ProviderBatchAdmissionGrant(False, reason="usage_batch_budget_exceeded", retryable=False)
        if not self._ready(scope):
            return ProviderBatchAdmissionGrant(False, reason="usage_reservation_backoff")
        host = ProviderBatchAdmissionGrant(True)
        if self.host_admission is not None:
            host = ProviderBatchAdmissionGrant.from_value(self.host_admission(key, selected, capacity))
            if not host.admitted:
                return host
        # Each failed reservation attempt has a fresh identity: the coordinator
        # deliberately remembers denied idempotency decisions as well as grants.
        physical_id = f"physical-batch-usage:{uuid.uuid4().hex}"
        try:
            # Snapshot identities include observation time. Comparing a prior
            # snapshot to one rebuilt at reserve time would reject every real
            # advancing clock. Reserve evaluates current limits inside its CAS;
            # no externally observed snapshot authorizes this admission.
            decision = self.coordinator.reserve(
                scope,
                requested,
                request_id=physical_id,
                idempotency_key=physical_id,
                owner_id=self.owner_id,
                ttl_ms=self.reservation_ttl_ms,
            )
        except Exception:
            self._defer(scope)
            host.release_once()
            return ProviderBatchAdmissionGrant(False, reason="usage_reservation_unavailable")
        if not decision.granted:
            self._defer(scope)
            host.release_once()
            return ProviderBatchAdmissionGrant(
                False,
                reason="usage_reservation_denied:" + ",".join(decision.reason_codes),
            )
        if not decision.reservation_id:
            host.release_once()
            raise PhysicalBatchUsageError("coordinator granted usage without reservation identity")
        self._clear_backoff(scope)
        receipt = {
            "requirement_id": self.requirement_id,
            "reservation_id": decision.reservation_id,
            "scope_digest": _digest(scope),
            "physical_call_id": physical_id,
            "state": "held",
            "settlement_basis": "unsettled",
            "requested": requested.to_dict(),
            "reserved_member_count": len(selected),
            "reserved_request_digests": [_digest(item.request_id) for item in selected],
            "shared_overhead_tokens": self.shared_overhead_tokens,
            "usage_revision": decision.usage_revision,
        }
        reservation = _PhysicalReservation(self, host, selected, decision.reservation_id, receipt)
        return ProviderBatchAdmissionGrant(
            True,
            release=reservation.close,
            before_dispatch=reservation.before_dispatch,
            after_dispatch=reservation.after_dispatch,
            usage_receipt=receipt,
            allow_fallback=False,
        )


__all__ = [
    "PHYSICAL_BATCH_USAGE_REQUIREMENT_ID",
    "PhysicalBatchUsageAdmission",
    "PhysicalBatchUsageError",
]
