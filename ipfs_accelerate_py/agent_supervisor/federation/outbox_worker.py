"""Event-driven owner-local projection from the transactional outbox.

The worker never polls DuckDB on a timer.  It drains a bounded restart backlog
once, then waits on a state-owner wake source keyed by the committed event
watermark.  A wake only causes queries when the watermark advances.  Routing
is at-least-once; the durable router and the final outbox disposition are both
idempotent, so a crash between them replays without duplicating a delivery.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    FederationBoundsError,
    FederationContractError,
    _finite,
    _identifier,
    _integer,
)
from .durable_event_router import (
    DurableEventRouter,
    DurableRouteCommit,
    DurableRouteResult,
    DurableRoutingBackpressure,
)
from .event_router import RouteResult
from .events import DomainEvent

MAX_OUTBOX_SCOPES_PER_DRAIN = 256
MAX_OUTBOX_EVENTS_PER_SCOPE = 1_024
MAX_SUBSCRIPTIONS_PER_SCOPE = 4_096


@dataclass(frozen=True)
class OutboxScope:
    """One exact tenant/federation routing authority slice."""

    tenant_id: str
    federation_id: str

    def __post_init__(self) -> None:
        _identifier(self.tenant_id, "tenant_id")
        _identifier(self.federation_id, "federation_id")


@dataclass(frozen=True)
class OutboxWake:
    """Result of one server-owned wait registration."""

    after_sequence: int
    committed_sequence: int
    after_notification_generation: int = 0
    notification_generation: int = 0
    timed_out: bool = False
    cancelled: bool = False
    server_shutdown: bool = False

    def __post_init__(self) -> None:
        after = _integer(self.after_sequence, "after_sequence", minimum=0)
        committed = _integer(
            self.committed_sequence,
            "committed_sequence",
            minimum=0,
        )
        after_generation = _integer(
            self.after_notification_generation,
            "after_notification_generation",
            minimum=0,
        )
        generation = _integer(
            self.notification_generation,
            "notification_generation",
            minimum=0,
        )
        if generation < after_generation:
            raise FederationContractError("outbox notification generation moved backwards")
        if committed < after and generation <= after_generation:
            raise FederationContractError("outbox wake watermark moved backwards")
        for name in ("timed_out", "cancelled", "server_shutdown"):
            if type(getattr(self, name)) is not bool:  # noqa: E721
                raise FederationContractError(f"{name} must be boolean")


class StateOwnerOutboxWake:
    """One process-owned condition for the authoritative outbox pump.

    The same lock protects the committed-watermark check and wait
    registration.  A commit between those steps therefore cannot be lost,
    and an idle deadline causes no database query or periodic wakeup.
    """

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        if not callable(monotonic):
            raise FederationContractError("outbox monotonic clock must be callable")
        self._monotonic = monotonic
        self._condition = threading.Condition(threading.RLock())
        self._committed_sequence = 0
        self._notification_generation = 0
        self._cancelled = False
        self._shutdown = False
        self._wakeup_count = 0

    @property
    def committed_sequence(self) -> int:
        with self._condition:
            return self._committed_sequence

    @property
    def wakeup_count(self) -> int:
        with self._condition:
            return self._wakeup_count

    @property
    def notification_generation(self) -> int:
        """Return the monotonic capacity-change generation."""

        with self._condition:
            return self._notification_generation

    def notify_committed(self, global_sequence: int) -> bool:
        sequence = _integer(
            global_sequence,
            "global_sequence",
            minimum=1,
        )
        with self._condition:
            advanced = sequence > self._committed_sequence
            self._committed_sequence = max(self._committed_sequence, sequence)
            # Capacity-releasing commands legitimately name an older source
            # event.  The independent generation makes that commit observable
            # without lying about or rewinding the event watermark.
            self._notification_generation += 1
            self._condition.notify_all()
            return advanced

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def clear_cancellation(self) -> None:
        with self._condition:
            self._cancelled = False

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def wait_for_outbox(
        self,
        *,
        after_sequence: int,
        after_notification_generation: int = 0,
        deadline_monotonic: float,
    ) -> OutboxWake:
        after = _integer(after_sequence, "after_sequence", minimum=0)
        after_generation = _integer(
            after_notification_generation,
            "after_notification_generation",
            minimum=0,
        )
        deadline = _finite(
            deadline_monotonic,
            "deadline_monotonic",
            minimum=0.0,
        )
        with self._condition:
            while True:
                if self._shutdown:
                    return OutboxWake(
                        after,
                        self._committed_sequence,
                        after_generation,
                        self._notification_generation,
                        server_shutdown=True,
                    )
                if self._cancelled:
                    return OutboxWake(
                        after,
                        self._committed_sequence,
                        after_generation,
                        self._notification_generation,
                        cancelled=True,
                    )
                if (
                    self._committed_sequence > after
                    or self._notification_generation > after_generation
                ):
                    return OutboxWake(
                        after,
                        self._committed_sequence,
                        after_generation,
                        self._notification_generation,
                    )
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return OutboxWake(
                        after,
                        self._committed_sequence,
                        after_generation,
                        self._notification_generation,
                        timed_out=True,
                    )
                awakened = self._condition.wait_for(
                    lambda: (
                        self._committed_sequence > after
                        or self._notification_generation > after_generation
                        or self._cancelled
                        or self._shutdown
                    ),
                    timeout=remaining,
                )
                if not awakened:
                    return OutboxWake(
                        after,
                        self._committed_sequence,
                        after_generation,
                        self._notification_generation,
                        timed_out=True,
                    )
                self._wakeup_count += 1


@dataclass(frozen=True)
class OutboxDisposition:
    """Atomic state-owner acknowledgement of routed outbox events."""

    disposition_id: str
    event_ids: tuple[str, ...]
    routed_global_sequence: int
    store_generation: int

    def __post_init__(self) -> None:
        _identifier(self.disposition_id, "disposition_id")
        if not self.event_ids:
            raise FederationContractError("outbox disposition cannot be empty")
        if len(self.event_ids) > MAX_OUTBOX_EVENTS_PER_SCOPE:
            raise FederationBoundsError("outbox disposition exceeds event bound")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise FederationContractError("outbox disposition repeats an event")
        for event_id in self.event_ids:
            _identifier(event_id, "event_id")
        _integer(
            self.routed_global_sequence,
            "routed_global_sequence",
            minimum=1,
        )
        _integer(self.store_generation, "store_generation", minimum=1)


@dataclass(frozen=True)
class OutboxDrainReceipt:
    """Bounded content-addressed observation for one drain pass."""

    receipt_id: str
    scope_count: int
    event_count: int
    delivery_count: int
    routed_global_sequence: int
    dispositions: tuple[OutboxDisposition, ...]
    retryable_scopes: tuple[OutboxScope, ...] = ()
    observed_global_sequence: int = 0

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "receipt_id")
        _integer(
            self.scope_count,
            "scope_count",
            minimum=0,
            maximum=MAX_OUTBOX_SCOPES_PER_DRAIN,
        )
        _integer(self.event_count, "event_count", minimum=0)
        _integer(self.delivery_count, "delivery_count", minimum=0)
        _integer(
            self.routed_global_sequence,
            "routed_global_sequence",
            minimum=0,
        )
        if len(self.dispositions) != self.scope_count:
            raise FederationContractError(
                "outbox drain dispositions do not match scope count"
            )
        if len(self.retryable_scopes) > MAX_OUTBOX_SCOPES_PER_DRAIN:
            raise FederationBoundsError("retryable outbox scopes exceed their bound")
        if len(set(self.retryable_scopes)) != len(self.retryable_scopes):
            raise FederationContractError("retryable outbox scopes are duplicated")
        _integer(
            self.observed_global_sequence,
            "observed_global_sequence",
            minimum=0,
        )


class OutboxRoutingRepository(Protocol):
    """Closed owner capability consumed by the worker."""

    def pending_outbox_scopes(self, *, maximum: int) -> tuple[OutboxScope, ...]: ...

    def pending_outbox_events(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[DomainEvent, ...]: ...

    def active_subscription_ids(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[str, ...]: ...

    def mark_outbox_routed(
        self,
        scope: OutboxScope,
        events: Sequence[DomainEvent],
        *,
        route_batch_id: str,
        delivery_count: int,
        subscription_count: int,
        idempotency_key: str,
    ) -> OutboxDisposition: ...


class OutboxWakeSource(Protocol):
    """Process-safe state-owner condition; it performs no database query."""

    def wait_for_outbox(
        self,
        *,
        after_sequence: int,
        after_notification_generation: int = 0,
        deadline_monotonic: float,
    ) -> OutboxWake: ...


class EventDrivenOutboxWorker:
    """Drain committed outbox rows and block until the owner signals change."""

    def __init__(
        self,
        repository: OutboxRoutingRepository,
        router_factory: Callable[[OutboxScope], DurableEventRouter],
        wake_source: OutboxWakeSource,
        *,
        maximum_scopes: int = MAX_OUTBOX_SCOPES_PER_DRAIN,
        maximum_events_per_scope: int = MAX_OUTBOX_EVENTS_PER_SCOPE,
        maximum_subscriptions_per_scope: int = MAX_SUBSCRIPTIONS_PER_SCOPE,
    ) -> None:
        if repository is None or wake_source is None or not callable(router_factory):
            raise FederationContractError(
                "outbox worker requires repository, router factory, and wake source"
            )
        self._repository = repository
        self._router_factory = router_factory
        self._wake_source = wake_source
        self._maximum_scopes = _integer(
            maximum_scopes,
            "maximum_scopes",
            minimum=1,
            maximum=MAX_OUTBOX_SCOPES_PER_DRAIN,
        )
        self._maximum_events = _integer(
            maximum_events_per_scope,
            "maximum_events_per_scope",
            minimum=1,
            maximum=MAX_OUTBOX_EVENTS_PER_SCOPE,
        )
        self._maximum_subscriptions = _integer(
            maximum_subscriptions_per_scope,
            "maximum_subscriptions_per_scope",
            minimum=1,
            maximum=MAX_SUBSCRIPTIONS_PER_SCOPE,
        )
        self._watermark = 0
        self._wait_sequence = 0
        self._notification_generation = 0

    @property
    def watermark(self) -> int:
        return self._watermark

    def drain_once(self) -> OutboxDrainReceipt:
        scopes = self._repository.pending_outbox_scopes(
            maximum=self._maximum_scopes
        )
        if not isinstance(scopes, tuple):
            raise FederationContractError("pending outbox scopes must be a tuple")
        if len(scopes) > self._maximum_scopes:
            raise FederationBoundsError(
                "state owner exceeded the pending outbox scope bound"
            )
        if any(not isinstance(scope, OutboxScope) for scope in scopes):
            raise FederationContractError("pending outbox scope is not typed")
        if len(set(scopes)) != len(scopes):
            raise FederationContractError("pending outbox scopes are duplicated")
        # Neither owner query ordering nor process restart timing may become
        # part of a content-addressed routing/disposition identity.
        scopes = tuple(
            sorted(scopes, key=lambda item: (item.tenant_id, item.federation_id))
        )
        dispositions: list[OutboxDisposition] = []
        retryable_scopes: list[OutboxScope] = []
        event_count = 0
        delivery_count = 0
        pass_watermark = self._watermark
        observed_watermark = self._wait_sequence
        for scope in scopes:
            events = self._repository.pending_outbox_events(
                scope,
                maximum=self._maximum_events,
            )
            if not isinstance(events, tuple):
                raise FederationContractError("pending outbox events must be a tuple")
            if not events:
                raise FederationContractError(
                    "pending outbox scope returned no pending events"
                )
            if len(events) > self._maximum_events:
                raise FederationBoundsError(
                    "state owner exceeded pending event bound"
                )
            if any(
                not isinstance(event, DomainEvent)
                for event in events
            ):
                raise FederationContractError("pending outbox event is not typed")
            if any(
                event.tenant_id != scope.tenant_id
                or event.federation_id != scope.federation_id
                for event in events
            ):
                raise FederationContractError(
                    "pending outbox event crossed its authority scope"
                )
            if len({event.event_id for event in events}) != len(events):
                raise FederationContractError("pending outbox events are duplicated")
            if len({event.global_sequence for event in events}) != len(events):
                raise FederationContractError(
                    "pending outbox events repeat a global sequence"
                )
            events = tuple(
                sorted(events, key=lambda item: (item.global_sequence, item.event_id))
            )
            observed_watermark = max(
                observed_watermark,
                max(item.global_sequence for item in events),
            )
            subscriptions = self._repository.active_subscription_ids(
                scope,
                maximum=self._maximum_subscriptions,
            )
            if not isinstance(subscriptions, tuple):
                raise FederationContractError(
                    "active subscription identities must be a tuple"
                )
            if len(subscriptions) > self._maximum_subscriptions:
                raise FederationBoundsError(
                    "state owner exceeded the active subscription bound"
                )
            subscriptions = tuple(
                _identifier(subscription_id, "subscription_id")
                for subscription_id in subscriptions
            )
            if len(set(subscriptions)) != len(subscriptions):
                raise FederationContractError(
                    "active subscription identities are duplicated"
                )
            subscriptions = tuple(sorted(subscriptions))
            router = self._router_factory(scope)
            for subscription_id in subscriptions:
                router.restore_subscription(
                    tenant_id=scope.tenant_id,
                    federation_id=scope.federation_id,
                    subscription_id=subscription_id,
                )
            try:
                routed: DurableRouteResult = router.route(events)
            except DurableRoutingBackpressure:
                retryable_scopes.append(scope)
                event_count += len(events)
                continue
            if (
                not isinstance(routed, DurableRouteResult)
                or not isinstance(routed.routing, RouteResult)
                or not isinstance(routed.commit, DurableRouteCommit)
            ):
                raise FederationContractError(
                    "durable router returned an invalid route result"
                )
            if routed.routing.input_events != len(events):
                raise FederationContractError(
                    "durable router input accounting differs from outbox events"
                )
            _integer(
                routed.routing.enqueued_deliveries,
                "enqueued_deliveries",
                minimum=0,
                maximum=len(events) * len(subscriptions),
            )
            admitted_delivery_count = (
                routed.routing.enqueued_deliveries
                + routed.routing.duplicate_deliveries_suppressed
            )
            if routed.routing.backpressured_subscriptions:
                retryable_scopes.append(scope)
                event_count += len(events)
                delivery_count += routed.routing.enqueued_deliveries
                continue
            event_ids = tuple(event.event_id for event in events)
            disposition_key = content_identity(
                {
                    "scope": {
                        "tenant_id": scope.tenant_id,
                        "federation_id": scope.federation_id,
                    },
                    "event_ids": list(event_ids),
                    "route_batch_id": routed.commit.batch_id,
                }
            )
            try:
                disposition = self._repository.mark_outbox_routed(
                    scope,
                    events,
                    route_batch_id=routed.commit.batch_id,
                    delivery_count=admitted_delivery_count,
                    subscription_count=len(subscriptions),
                    idempotency_key=f"outbox-route:{disposition_key}",
                )
            except DurableRoutingBackpressure:
                retryable_scopes.append(scope)
                event_count += len(events)
                delivery_count += routed.routing.enqueued_deliveries
                continue
            if (
                not isinstance(disposition, OutboxDisposition)
                or disposition.event_ids != event_ids
                or disposition.routed_global_sequence
                != max(event.global_sequence for event in events)
            ):
                raise FederationContractError(
                    "outbox disposition differs from routed events"
                )
            dispositions.append(disposition)
            event_count += len(events)
            delivery_count += admitted_delivery_count
            pass_watermark = max(
                pass_watermark,
                disposition.routed_global_sequence,
            )
        self._watermark = pass_watermark
        self._wait_sequence = observed_watermark
        receipt_body = {
            "scope_count": len(dispositions),
            "event_count": event_count,
            "delivery_count": delivery_count,
            "routed_global_sequence": pass_watermark,
            "disposition_ids": [item.disposition_id for item in dispositions],
            "retryable_scopes": [
                {
                    "tenant_id": item.tenant_id,
                    "federation_id": item.federation_id,
                }
                for item in retryable_scopes
            ],
            "observed_global_sequence": observed_watermark,
        }
        return OutboxDrainReceipt(
            receipt_id=f"outbox-drain:{content_identity(receipt_body)}",
            scope_count=len(dispositions),
            event_count=event_count,
            delivery_count=delivery_count,
            routed_global_sequence=pass_watermark,
            dispositions=tuple(dispositions),
            retryable_scopes=tuple(retryable_scopes),
            observed_global_sequence=observed_watermark,
        )

    def wait_and_drain(self, *, deadline_monotonic: float) -> OutboxDrainReceipt | None:
        """Block once and query only after an advancing commit notification."""

        deadline = _finite(
            deadline_monotonic,
            "deadline_monotonic",
            minimum=0.0,
        )
        after_sequence = self._wait_sequence
        after_generation = self._notification_generation
        wake = self._wake_source.wait_for_outbox(
            after_sequence=after_sequence,
            after_notification_generation=after_generation,
            deadline_monotonic=deadline,
        )
        if not isinstance(wake, OutboxWake):
            raise FederationContractError("outbox wake source returned an invalid result")
        if wake.after_sequence != after_sequence:
            raise FederationContractError(
                "outbox wake does not match its wait registration"
            )
        if wake.after_notification_generation != after_generation:
            raise FederationContractError(
                "outbox wake generation does not match its wait registration"
            )
        if wake.cancelled or wake.server_shutdown:
            return None
        if wake.timed_out or (
            wake.committed_sequence <= self._wait_sequence
            and wake.notification_generation <= self._notification_generation
        ):
            return None
        self._notification_generation = wake.notification_generation
        receipt = self.drain_once()
        # A wake may cover a transaction whose event was already routed by a
        # prior crash recovery.  Preserve the notification watermark after an
        # empty observation.  A non-empty bounded pass may have left more rows
        # behind, so it must retain the disposition watermark and let the same
        # owner watermark trigger the next blocking registration immediately.
        if receipt.event_count == 0:
            self._watermark = max(self._watermark, wake.committed_sequence)
            self._wait_sequence = max(self._wait_sequence, wake.committed_sequence)
        return receipt


__all__ = [
    "EventDrivenOutboxWorker",
    "MAX_OUTBOX_EVENTS_PER_SCOPE",
    "MAX_OUTBOX_SCOPES_PER_DRAIN",
    "MAX_SUBSCRIPTIONS_PER_SCOPE",
    "OutboxDisposition",
    "OutboxDrainReceipt",
    "OutboxRoutingRepository",
    "OutboxScope",
    "OutboxWake",
    "OutboxWakeSource",
    "StateOwnerOutboxWake",
]
