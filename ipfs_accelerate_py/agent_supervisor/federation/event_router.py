"""Bounded event routing, coalescing, retry, and dead-letter policy.

This module is a deterministic state-owner component.  It does not delete or
rewrite the authoritative event log: coalescing changes only which compact
wakeup notification is delivered.  Safety-significant transitions always
remain one delivery per event.  A durable adapter is expected to persist the
returned delivery attempts, dead letters, acknowledgements, and cursor changes
in the same canonical control plane.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP017, UP042

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType

from ..task_sources.control_plane_contracts import content_identity
from .contracts import FederationBoundsError, FederationContractError, _identifier, _integer
from .events import (
    MAX_CHANGED_FACTS,
    MAX_EVENT_BATCH,
    DeadLetter,
    DeliveryAttempt,
    DeliveryState,
    DomainEvent,
    EventClass,
    EventSubscription,
    SubscriptionState,
)
from .subscriptions import CausalSelectorEvaluator, event_matches_subscription


class EventRouterError(RuntimeError):
    """Base error for a bounded router operation."""


class RouterBackpressure(EventRouterError):
    """A consumer's durable pending ceiling prevents another delivery."""


class DeliveryOwnershipError(EventRouterError):
    """A consumer attempted to finish a delivery it does not own."""


class CoalescingMode(str, Enum):
    NONE = "none"
    UNION_CHANGED_FACTS = "union_changed_facts"
    LATEST_GENERATION = "latest_generation"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class CoalescingDecision:
    """A wakeup projection over immutable input events.

    ``representative_event`` is an existing authoritative event.  The union of
    facts is advisory routing metadata and cannot be mistaken for a newly
    committed event or causal edge.
    """

    representative_event: DomainEvent
    input_event_ids: tuple[str, ...]
    changed_fact_refs: tuple[str, ...]
    mode: CoalescingMode

    def __post_init__(self) -> None:
        if not isinstance(self.representative_event, DomainEvent):
            raise FederationContractError("representative_event must be a DomainEvent")
        if not self.input_event_ids:
            raise FederationContractError("coalescing decision requires input events")
        for event_id in self.input_event_ids:
            _identifier(event_id, "input_event_id")
        if len(set(self.input_event_ids)) != len(self.input_event_ids):
            raise FederationContractError("coalescing decision contains duplicate events")
        if len(self.changed_fact_refs) > MAX_CHANGED_FACTS:
            raise FederationBoundsError("coalesced changed-fact bound exceeded")
        if not isinstance(self.mode, CoalescingMode):
            raise FederationContractError("coalescing mode is not closed")
        if self.representative_event.coalescing_forbidden and len(self.input_event_ids) != 1:
            raise FederationContractError("safety-significant events cannot be coalesced")

    @property
    def decision_id(self) -> str:
        return f"coalescing:{content_identity(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/causal-federation/coalescing-decision@1",
            "representative_event_id": self.representative_event.event_id,
            "input_event_ids": list(self.input_event_ids),
            "changed_fact_refs": list(self.changed_fact_refs),
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class QueuedDelivery:
    delivery_id: str
    subscription_id: str
    subscription_revision: int
    consumer_id: str
    decision: CoalescingDecision
    attempt_number: int = 0

    def __post_init__(self) -> None:
        for name in ("delivery_id", "subscription_id", "consumer_id"):
            _identifier(getattr(self, name), name)
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.attempt_number, "attempt_number", maximum=1_000)


@dataclass(frozen=True)
class RouteResult:
    input_events: int
    matched_subscriptions: int
    enqueued_deliveries: int
    coalesced_events: int
    duplicate_deliveries_suppressed: int
    expired_events: int
    backpressured_subscriptions: tuple[str, ...]


@dataclass(frozen=True)
class FailureResult:
    attempt: DeliveryAttempt
    dead_letter: DeadLetter | None
    retry_scheduled: bool
    subscription_quarantined: bool


def _event_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _routing_group(event: DomainEvent) -> tuple[str, ...]:
    """Return the bounded identity of a coalescible wakeup family."""

    if event.event_type is EventClass.SYMBOL_CHANGED:
        return (
            "symbol-refresh",
            event.tenant_id,
            event.federation_id,
            event.repository_id,
            event.tree_id,
            event.supervisor_id,
        )
    if event.event_type is EventClass.PROVIDER_CAPACITY_CHANGED:
        return (
            "provider-capacity",
            event.tenant_id,
            event.federation_id,
            event.resource_class,
        )
    # A producer-selected key is a closed, compact identity and may explicitly
    # nominate repeated current-disposition notifications for suppression.
    return (
        "deduplication",
        event.tenant_id,
        event.federation_id,
        event.event_type.value,
        event.deduplication_key,
    )


def plan_coalescing(events: Sequence[DomainEvent]) -> tuple[CoalescingDecision, ...]:
    """Plan bounded wakeup coalescing without mutating event history."""

    if len(events) > MAX_EVENT_BATCH:
        raise FederationBoundsError("router input batch exceeds the event bound")
    ordered = tuple(sorted(events, key=lambda item: item.global_sequence))
    if any(not isinstance(item, DomainEvent) for item in ordered):
        raise FederationContractError("router input contains a non-event")

    grouped: dict[tuple[str, ...], list[DomainEvent]] = defaultdict(list)
    singles: list[CoalescingDecision] = []
    for event in ordered:
        if event.coalescing_forbidden:
            singles.append(
                CoalescingDecision(
                    representative_event=event,
                    input_event_ids=(event.event_id,),
                    changed_fact_refs=event.changed_fact_refs,
                    mode=CoalescingMode.NONE,
                )
            )
        else:
            grouped[_routing_group(event)].append(event)

    planned = list(singles)
    for family in grouped.values():
        representative = family[-1]
        if representative.event_type is EventClass.PROVIDER_CAPACITY_CHANGED:
            # Capacity updates supersede prior generations.  Retaining facts
            # from an older generation would present stale capacity as part of
            # the current disposition.
            facts = representative.changed_fact_refs
        else:
            facts = tuple(
                dict.fromkeys(fact for event in family for fact in event.changed_fact_refs)
            )
        if len(facts) > MAX_CHANGED_FACTS:
            # Widen safely by splitting; never truncate affected facts.
            for event in family:
                planned.append(
                    CoalescingDecision(
                        representative_event=event,
                        input_event_ids=(event.event_id,),
                        changed_fact_refs=event.changed_fact_refs,
                        mode=CoalescingMode.NONE,
                    )
                )
            continue
        if len(family) == 1:
            mode = CoalescingMode.NONE
        elif representative.event_type is EventClass.SYMBOL_CHANGED:
            mode = CoalescingMode.UNION_CHANGED_FACTS
        elif representative.event_type is EventClass.PROVIDER_CAPACITY_CHANGED:
            mode = CoalescingMode.LATEST_GENERATION
        else:
            mode = CoalescingMode.SUPERSEDED
        planned.append(
            CoalescingDecision(
                representative_event=representative,
                input_event_ids=tuple(item.event_id for item in family),
                changed_fact_refs=facts,
                mode=mode,
            )
        )
    return tuple(sorted(planned, key=lambda item: item.representative_event.global_sequence))


class BoundedEventRouter:
    """In-owner router with bounded queues and explicit durable outputs.

    This implementation is useful for the state-owner service and hermetic
    qualification.  It intentionally exposes no background polling loop.
    Callers invoke :meth:`route` after an outbox commit notification.
    """

    def __init__(
        self,
        *,
        maximum_subscriptions: int = 4_096,
        maximum_fanout_per_event: int = 256,
        circuit_breaker_failures: int = 16,
    ) -> None:
        _integer(maximum_subscriptions, "maximum_subscriptions", minimum=1, maximum=65_536)
        _integer(maximum_fanout_per_event, "maximum_fanout_per_event", minimum=1, maximum=4_096)
        _integer(circuit_breaker_failures, "circuit_breaker_failures", minimum=1, maximum=1_000)
        self._maximum_subscriptions = maximum_subscriptions
        self._maximum_fanout = maximum_fanout_per_event
        self._circuit_breaker_failures = circuit_breaker_failures
        self._subscriptions: dict[str, EventSubscription] = {}
        self._queues: dict[str, deque[QueuedDelivery]] = defaultdict(deque)
        self._inflight: dict[str, QueuedDelivery] = {}
        self._known_deliveries: set[str] = set()
        self._durably_known_deliveries: set[str] = set()
        # The planner is reconstructed for every owner-local outbox pass.  A
        # durable adapter therefore seeds identities already committed and the
        # current nonterminal queue population before planning.  Keeping this
        # state separate from ``_queues`` avoids manufacturing in-memory
        # deliveries while still enforcing the persisted pending ceiling.
        self._durable_pending_counts: dict[str, int] = defaultdict(int)
        self._newly_queued_counts: dict[str, int] = defaultdict(int)
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._quarantined: set[str] = set()
        self._dead_letters: list[DeadLetter] = []

    @property
    def subscriptions(self) -> Mapping[str, EventSubscription]:
        return MappingProxyType(dict(self._subscriptions))

    @property
    def dead_letters(self) -> tuple[DeadLetter, ...]:
        return tuple(self._dead_letters)

    def register(self, subscription: EventSubscription) -> None:
        if not isinstance(subscription, EventSubscription):
            raise FederationContractError("subscription must be EventSubscription")
        existing = self._subscriptions.get(subscription.subscription_id)
        if existing is None and len(self._subscriptions) >= self._maximum_subscriptions:
            raise FederationBoundsError("router subscription ceiling reached")
        if existing is not None and subscription.revision <= existing.revision:
            if subscription == existing:
                return
            raise FederationContractError("subscription revision did not advance")
        if existing is not None:
            # A subscription revision is an authority boundary.  Work queued or
            # exposed under the prior selectors, state, or consumer must never
            # leak into the replacement revision.
            self._queues.pop(subscription.subscription_id, None)
            for delivery_id, delivery in tuple(self._inflight.items()):
                if delivery.subscription_id == subscription.subscription_id:
                    self._inflight.pop(delivery_id, None)
            self._consecutive_failures[subscription.subscription_id] = 0
        self._subscriptions[subscription.subscription_id] = subscription
        if subscription.state is not SubscriptionState.QUARANTINED:
            self._quarantined.discard(subscription.subscription_id)

    def seed_durable_state(
        self,
        *,
        known_delivery_ids: Sequence[str],
        pending_by_subscription: Mapping[str, int],
    ) -> None:
        """Seed bounded persisted state into a freshly reconstructed planner."""

        if len(known_delivery_ids) > 65_536:
            raise FederationBoundsError("durable delivery seed exceeds its bound")
        known = tuple(_identifier(item, "delivery_id") for item in known_delivery_ids)
        if len(set(known)) != len(known):
            raise FederationContractError("durable delivery seed contains duplicates")
        pending: dict[str, int] = {}
        for subscription_id, value in pending_by_subscription.items():
            identity = _identifier(subscription_id, "subscription_id")
            subscription = self._subscriptions.get(identity)
            if subscription is None:
                raise FederationContractError(
                    "durable pending seed names an unregistered subscription"
                )
            pending[identity] = _integer(
                value,
                "pending_delivery_count",
                minimum=0,
                maximum=65_536,
            )
        self._known_deliveries.update(known)
        self._durably_known_deliveries.update(known)
        self._durable_pending_counts = defaultdict(int, pending)

    def route(
        self,
        events: Sequence[DomainEvent],
        *,
        causal_evaluator: CausalSelectorEvaluator | None = None,
        now: str | None = None,
    ) -> RouteResult:
        # Validate and order once, but coalesce *after* applying each
        # subscription's selectors.  A global coalescing decision can contain
        # symbol A and symbol B while a consumer is authorized for only A; in
        # that case using B as the representative either leaks B or causes the
        # durable repository to reject the entire wakeup.  Per-subscription
        # planning preserves complete coverage without widening authority.
        ordered = tuple(sorted(events, key=lambda item: item.global_sequence))
        if len(ordered) > MAX_EVENT_BATCH:
            raise FederationBoundsError("router input batch exceeds the event bound")
        if any(not isinstance(item, DomainEvent) for item in ordered):
            raise FederationContractError("router input contains a non-event")
        now_value = _event_time(now) if now else datetime.now(timezone.utc)
        expired_ids = {
            event.event_id
            for event in ordered
            if event.expires_at and _event_time(event.expires_at) <= now_value
        }
        enqueued = 0
        duplicates = 0
        matched: set[str] = set()
        backpressured: set[str] = set()
        per_event_fanout: dict[str, int] = defaultdict(int)
        coalesced = 0

        for subscription in self._subscriptions.values():
            if subscription.state is not SubscriptionState.ACTIVE:
                continue
            if subscription.subscription_id in self._quarantined:
                continue
            if _event_time(subscription.expires_at) <= now_value:
                continue
            eligible = tuple(
                event
                for event in ordered
                if event.event_id not in expired_ids
                and event_matches_subscription(
                    event,
                    subscription,
                    causal_selector_evaluator=causal_evaluator,
                )
            )
            decisions = plan_coalescing(eligible)
            for decision in decisions:
                live_input_ids = decision.input_event_ids
                if any(
                    per_event_fanout[event_id] >= self._maximum_fanout
                    for event_id in live_input_ids
                ):
                    backpressured.add(subscription.subscription_id)
                    continue
                delivery_id = f"delivery:{content_identity({'subscription_id': subscription.subscription_id, 'subscription_revision': subscription.revision, 'input_event_ids': list(live_input_ids)})}"
                if delivery_id in self._known_deliveries:
                    duplicates += 1
                    if delivery_id in self._durably_known_deliveries:
                        # A fresh durable planner must carry replayed records
                        # into the content-addressed batch so a crash replay
                        # binds the same complete delivery set.  A delivery
                        # known only to this in-memory router stays queued once.
                        self._queues[subscription.subscription_id].append(
                            QueuedDelivery(
                                delivery_id=delivery_id,
                                subscription_id=subscription.subscription_id,
                                subscription_revision=subscription.revision,
                                consumer_id=subscription.consumer_id,
                                decision=decision,
                            )
                        )
                        matched.add(subscription.subscription_id)
                    continue
                queue = self._queues[subscription.subscription_id]
                if (
                    self._durable_pending_counts[subscription.subscription_id]
                    + self._newly_queued_counts[subscription.subscription_id]
                    >= subscription.maximum_pending
                ):
                    backpressured.add(subscription.subscription_id)
                    continue
                queue.append(
                    QueuedDelivery(
                        delivery_id=delivery_id,
                        subscription_id=subscription.subscription_id,
                        subscription_revision=subscription.revision,
                        consumer_id=subscription.consumer_id,
                        decision=decision,
                    )
                )
                self._known_deliveries.add(delivery_id)
                self._newly_queued_counts[subscription.subscription_id] += 1
                for event_id in live_input_ids:
                    per_event_fanout[event_id] += 1
                matched.add(subscription.subscription_id)
                enqueued += 1
                coalesced += max(0, len(live_input_ids) - 1)

        return RouteResult(
            input_events=len(ordered),
            matched_subscriptions=len(matched),
            enqueued_deliveries=enqueued,
            coalesced_events=coalesced,
            duplicate_deliveries_suppressed=duplicates,
            expired_events=len(expired_ids),
            backpressured_subscriptions=tuple(sorted(backpressured)),
        )

    def take(self, subscription_id: str, *, maximum: int) -> tuple[QueuedDelivery, ...]:
        subscription = self._subscriptions.get(subscription_id)
        if subscription is None:
            raise EventRouterError("unknown subscription")
        if (
            subscription.state is not SubscriptionState.ACTIVE
            or subscription_id in self._quarantined
            or _event_time(subscription.expires_at) <= datetime.now(timezone.utc)
        ):
            return ()
        limit = _integer(maximum, "maximum", minimum=1, maximum=subscription.maximum_batch)
        queue = self._queues[subscription_id]
        selected: list[QueuedDelivery] = []
        while queue and len(selected) < limit:
            delivery = queue.popleft()
            if (
                delivery.subscription_revision != subscription.revision
                or delivery.consumer_id != subscription.consumer_id
            ):
                continue
            self._inflight[delivery.delivery_id] = delivery
            selected.append(delivery)
        return tuple(selected)

    def acknowledge(self, delivery_id: str, *, consumer_id: str) -> None:
        delivery = self._owned_inflight(delivery_id, consumer_id)
        self._inflight.pop(delivery.delivery_id, None)
        self._consecutive_failures[delivery.subscription_id] = 0

    def fail(
        self,
        delivery_id: str,
        *,
        consumer_id: str,
        error_code: str,
        evidence_ref: str,
        recorded_at: str,
        expires_at: str = "",
    ) -> FailureResult:
        delivery = self._owned_inflight(delivery_id, consumer_id)
        _identifier(error_code, "error_code")
        _identifier(evidence_ref, "evidence_ref")
        self._inflight.pop(delivery.delivery_id, None)
        attempt_number = delivery.attempt_number + 1
        subscription = self._subscriptions[delivery.subscription_id]
        attempt_id = f"delivery-attempt:{content_identity({'delivery_id': delivery_id, 'attempt': attempt_number, 'error_code': error_code})}"
        failures = self._consecutive_failures[delivery.subscription_id] + 1
        self._consecutive_failures[delivery.subscription_id] = failures
        quarantined = failures >= self._circuit_breaker_failures
        if quarantined:
            self._quarantined.add(delivery.subscription_id)

        if attempt_number > subscription.retry_budget:
            dead_letter = DeadLetter(
                dead_letter_id=f"dead-letter:{content_identity({'delivery_id': delivery_id, 'attempt': attempt_number})}",
                event_id=delivery.decision.representative_event.event_id,
                subscription_id=delivery.subscription_id,
                consumer_id=delivery.consumer_id,
                retry_count=attempt_number,
                error_code=error_code,
                evidence_ref=evidence_ref,
                quarantined=quarantined,
                created_at=recorded_at,
                expires_at=expires_at,
            )
            self._dead_letters.append(dead_letter)
            return FailureResult(
                attempt=DeliveryAttempt(
                    attempt_id=attempt_id,
                    event_id=delivery.decision.representative_event.event_id,
                    subscription_id=delivery.subscription_id,
                    consumer_id=delivery.consumer_id,
                    attempt_number=attempt_number,
                    state=DeliveryState.DEAD_LETTERED,
                    error_code=error_code,
                    recorded_at=recorded_at,
                ),
                dead_letter=dead_letter,
                retry_scheduled=False,
                subscription_quarantined=quarantined,
            )

        retry = QueuedDelivery(
            delivery_id=delivery.delivery_id,
            subscription_id=delivery.subscription_id,
            subscription_revision=delivery.subscription_revision,
            consumer_id=delivery.consumer_id,
            decision=delivery.decision,
            attempt_number=attempt_number,
        )
        self._queues[delivery.subscription_id].append(retry)
        return FailureResult(
            attempt=DeliveryAttempt(
                attempt_id=attempt_id,
                event_id=delivery.decision.representative_event.event_id,
                subscription_id=delivery.subscription_id,
                consumer_id=delivery.consumer_id,
                attempt_number=attempt_number,
                state=DeliveryState.RETRY,
                error_code=error_code,
                recorded_at=recorded_at,
            ),
            dead_letter=None,
            retry_scheduled=True,
            subscription_quarantined=quarantined,
        )

    def retry_dead_letter(self, dead_letter_id: str) -> None:
        match = next(
            (item for item in self._dead_letters if item.dead_letter_id == dead_letter_id),
            None,
        )
        if match is None:
            raise EventRouterError("unknown dead letter")
        # A retry command must first reactivate the subscription through an
        # authorized revision.  This method only clears the circuit-breaker
        # observation; the durable outbox remains the source of replay.
        self._quarantined.discard(match.subscription_id)
        self._consecutive_failures[match.subscription_id] = 0

    def _owned_inflight(self, delivery_id: str, consumer_id: str) -> QueuedDelivery:
        _identifier(delivery_id, "delivery_id")
        _identifier(consumer_id, "consumer_id")
        delivery = self._inflight.get(delivery_id)
        if delivery is None or delivery.consumer_id != consumer_id:
            raise DeliveryOwnershipError("delivery is not owned by this consumer")
        return delivery


__all__ = [
    "BoundedEventRouter",
    "CoalescingDecision",
    "CoalescingMode",
    "DeliveryOwnershipError",
    "EventRouterError",
    "FailureResult",
    "QueuedDelivery",
    "RouteResult",
    "RouterBackpressure",
    "plan_coalescing",
]
