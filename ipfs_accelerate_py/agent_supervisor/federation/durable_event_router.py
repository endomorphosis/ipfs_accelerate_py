"""Durable state-owner adapter for bounded federation event routing.

``BoundedEventRouter`` is intentionally an in-memory routing planner.  This
module supplies the narrow authority boundary that makes its outputs durable:
queued deliveries and their complete coalescing coverage are committed before
they can be exposed to a consumer.  Delivery attempts, retries, circuit-breaker
quarantine, and dead letters likewise pass through one typed repository.

The repository is an already-authorized state-owner capability.  This module
does not open DuckDB, accept SQL, start a polling loop, or fall back to an
embedded authority.  Network delivery remains at-least-once; authoritative
effects still require idempotency, CAS, leases, and fencing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    FederationBoundsError,
    FederationContractError,
    _identifier,
    _integer,
    _timestamp,
)
from .event_router import (
    BoundedEventRouter,
    CoalescingMode,
    FailureResult,
    QueuedDelivery,
    RouteResult,
)
from .events import (
    MAX_EVENT_BATCH,
    DeadLetter,
    DeliveryAttempt,
    DeliveryState,
    DomainEvent,
    EventSubscription,
    SubscriptionState,
)
from .subscriptions import CausalSelectorEvaluator

MAX_DURABLE_DELIVERIES_PER_COMMIT = 65_536


class DurableEventRouterError(RuntimeError):
    """A durable routing invariant or state-owner operation failed."""


class DurableRoutingBackpressure(DurableEventRouterError):
    """Authoritative queue admission changed and the route must be retried."""


@dataclass(frozen=True)
class DurableSubscriptionRoutingState:
    """Current nonterminal queue population for one canonical subscription."""

    subscription_id: str
    subscription_revision: int
    pending_deliveries: int
    maximum_pending: int

    def __post_init__(self) -> None:
        _identifier(self.subscription_id, "subscription_id")
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        _integer(self.pending_deliveries, "pending_deliveries", minimum=0)
        _integer(self.maximum_pending, "maximum_pending", minimum=1)


@dataclass(frozen=True)
class DurableRoutingState:
    """Bounded persisted inputs used to reconstruct a routing planner."""

    known_delivery_ids: tuple[str, ...]
    subscriptions: tuple[DurableSubscriptionRoutingState, ...]
    maximum_fanout_per_event: int
    store_generation: int

    def __post_init__(self) -> None:
        if len(self.known_delivery_ids) > MAX_DURABLE_DELIVERIES_PER_COMMIT:
            raise FederationBoundsError("durable routing state exceeds its delivery bound")
        for delivery_id in self.known_delivery_ids:
            _identifier(delivery_id, "delivery_id")
        if len(set(self.known_delivery_ids)) != len(self.known_delivery_ids):
            raise FederationContractError("durable routing state repeats a delivery")
        if any(
            not isinstance(item, DurableSubscriptionRoutingState)
            for item in self.subscriptions
        ):
            raise FederationContractError("durable routing state has an invalid subscription")
        identities = tuple(item.subscription_id for item in self.subscriptions)
        if len(set(identities)) != len(identities):
            raise FederationContractError("durable routing state repeats a subscription")
        _integer(
            self.maximum_fanout_per_event,
            "maximum_fanout_per_event",
            minimum=1,
            maximum=4_096,
        )
        _integer(self.store_generation, "store_generation", minimum=1)


@dataclass(frozen=True)
class CoalescingCoverageRecord:
    """Auditable coverage for one consumer wakeup projection."""

    coverage_id: str
    decision_id: str
    subscription_id: str
    subscription_revision: int
    representative_event_id: str
    input_event_ids: tuple[str, ...]
    changed_fact_refs: tuple[str, ...]
    mode: CoalescingMode

    def __post_init__(self) -> None:
        for name in (
            "coverage_id",
            "decision_id",
            "subscription_id",
            "representative_event_id",
        ):
            _identifier(getattr(self, name), name)
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        if not self.input_event_ids:
            raise FederationContractError("coalescing coverage cannot be empty")
        if len(self.input_event_ids) > MAX_EVENT_BATCH:
            raise FederationBoundsError("coalescing coverage exceeds the event bound")
        if len(set(self.input_event_ids)) != len(self.input_event_ids):
            raise FederationContractError("coalescing coverage contains duplicate events")
        for event_id in self.input_event_ids:
            _identifier(event_id, "input_event_id")
        for fact_ref in self.changed_fact_refs:
            _identifier(fact_ref, "changed_fact_ref")
        if not isinstance(self.mode, CoalescingMode):
            raise FederationContractError("coalescing mode is not closed")
        if self.mode is CoalescingMode.NONE and len(self.input_event_ids) != 1:
            raise FederationContractError("uncoalesced coverage must bind exactly one event")
        if self.representative_event_id not in self.input_event_ids:
            raise FederationContractError("representative event is outside coverage")

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage_id": self.coverage_id,
            "decision_id": self.decision_id,
            "subscription_id": self.subscription_id,
            "subscription_revision": self.subscription_revision,
            "representative_event_id": self.representative_event_id,
            "input_event_ids": list(self.input_event_ids),
            "changed_fact_refs": list(self.changed_fact_refs),
            "mode": self.mode.value,
        }


@dataclass(frozen=True)
class DurableQueuedDelivery:
    """A pending delivery and its complete immutable event coverage."""

    delivery: QueuedDelivery
    coverage: CoalescingCoverageRecord

    def __post_init__(self) -> None:
        if not isinstance(self.delivery, QueuedDelivery):
            raise FederationContractError("delivery must be a QueuedDelivery")
        if not isinstance(self.coverage, CoalescingCoverageRecord):
            raise FederationContractError("coverage must be a CoalescingCoverageRecord")
        decision = self.delivery.decision
        if self.coverage.subscription_id != self.delivery.subscription_id:
            raise FederationContractError("coverage subscription does not match delivery")
        if self.coverage.subscription_revision != self.delivery.subscription_revision:
            raise FederationContractError("coverage revision does not match delivery")
        if self.coverage.decision_id != decision.decision_id:
            raise FederationContractError("coverage decision does not match delivery")
        if self.coverage.representative_event_id != decision.representative_event.event_id:
            raise FederationContractError("coverage representative does not match delivery")
        if self.coverage.input_event_ids != decision.input_event_ids:
            raise FederationContractError("coverage events do not match delivery")
        if self.coverage.changed_fact_refs != decision.changed_fact_refs:
            raise FederationContractError("coverage facts do not match delivery")
        if self.coverage.mode is not decision.mode:
            raise FederationContractError("coverage mode does not match delivery")
        if decision.representative_event.coalescing_forbidden:
            if self.coverage.mode is not CoalescingMode.NONE:
                raise FederationContractError("safety-significant delivery was coalesced")
            if len(self.coverage.input_event_ids) != 1:
                raise FederationContractError("safety-significant coverage is not singular")

    def to_dict(self) -> dict[str, object]:
        return {
            "delivery_id": self.delivery.delivery_id,
            "subscription_id": self.delivery.subscription_id,
            "subscription_revision": self.delivery.subscription_revision,
            "consumer_id": self.delivery.consumer_id,
            "attempt_number": self.delivery.attempt_number,
            "coverage": self.coverage.to_dict(),
        }


@dataclass(frozen=True)
class DurableRouteBatch:
    """One bounded, content-addressed state-owner routing commit."""

    batch_id: str
    deliveries: tuple[DurableQueuedDelivery, ...]
    maximum_fanout_per_event: int = 256

    def __post_init__(self) -> None:
        _identifier(self.batch_id, "batch_id")
        if len(self.deliveries) > MAX_DURABLE_DELIVERIES_PER_COMMIT:
            raise FederationBoundsError("durable routing commit exceeds its delivery bound")
        if any(not isinstance(item, DurableQueuedDelivery) for item in self.deliveries):
            raise FederationContractError("routing batch contains an invalid delivery")
        delivery_ids = tuple(item.delivery.delivery_id for item in self.deliveries)
        if len(set(delivery_ids)) != len(delivery_ids):
            raise FederationContractError("routing batch contains duplicate deliveries")
        _integer(
            self.maximum_fanout_per_event,
            "maximum_fanout_per_event",
            minimum=1,
            maximum=4_096,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "deliveries": [item.to_dict() for item in self.deliveries],
            "maximum_fanout_per_event": self.maximum_fanout_per_event,
        }


@dataclass(frozen=True)
class DurableRouteCommit:
    """Idempotent disposition returned by the canonical state owner."""

    batch_id: str
    inserted_delivery_ids: tuple[str, ...]
    existing_delivery_ids: tuple[str, ...]
    store_generation: int

    def __post_init__(self) -> None:
        _identifier(self.batch_id, "batch_id")
        _integer(self.store_generation, "store_generation", minimum=1)
        all_ids = self.inserted_delivery_ids + self.existing_delivery_ids
        for delivery_id in all_ids:
            _identifier(delivery_id, "delivery_id")
        if len(set(all_ids)) != len(all_ids):
            raise FederationContractError("route commit has overlapping dispositions")


@dataclass(frozen=True)
class DurableRouteResult:
    routing: RouteResult
    commit: DurableRouteCommit


@dataclass(frozen=True)
class ExposedDelivery:
    """A delivery that may be returned only after durable attempt creation."""

    queued: DurableQueuedDelivery
    attempt: DeliveryAttempt

    def __post_init__(self) -> None:
        if not isinstance(self.queued, DurableQueuedDelivery):
            raise FederationContractError("queued must be a durable delivery")
        if not isinstance(self.attempt, DeliveryAttempt):
            raise FederationContractError("attempt must be a DeliveryAttempt")
        delivery = self.queued.delivery
        event = delivery.decision.representative_event
        if self.attempt.event_id != event.event_id:
            raise FederationContractError("attempt event does not match delivery")
        if self.attempt.subscription_id != delivery.subscription_id:
            raise FederationContractError("attempt subscription does not match delivery")
        if self.attempt.consumer_id != delivery.consumer_id:
            raise FederationContractError("attempt consumer does not match delivery")
        if self.attempt.attempt_number != delivery.attempt_number + 1:
            raise FederationContractError("attempt number does not follow delivery state")
        if self.attempt.state is not DeliveryState.DELIVERED:
            raise FederationContractError("an exposed attempt must be delivered")


@dataclass(frozen=True)
class DurableDeliveryFailure:
    """Evidence-bearing failure command for one durably exposed attempt."""

    exposed: ExposedDelivery
    error_code: str
    evidence_ref: str
    recorded_at: str
    expires_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.exposed, ExposedDelivery):
            raise FederationContractError("failure must bind an exposed delivery")
        _identifier(self.error_code, "error_code")
        _identifier(self.evidence_ref, "evidence_ref")
        _timestamp(self.recorded_at, "recorded_at")
        if self.expires_at:
            _timestamp(self.expires_at, "expires_at")


@dataclass(frozen=True)
class DurableFailureCommit:
    """Atomic retry/dead-letter/quarantine disposition from the state owner."""

    failure_id: str
    result: FailureResult
    store_generation: int

    def __post_init__(self) -> None:
        _identifier(self.failure_id, "failure_id")
        if not isinstance(self.result, FailureResult):
            raise FederationContractError("failure result has the wrong type")
        _integer(self.store_generation, "store_generation", minimum=1)


@dataclass(frozen=True)
class DeadLetterRetryCommit:
    """Durable disposition for an authorized dead-letter requeue."""

    dead_letter_id: str
    delivery_id: str
    subscription_id: str
    subscription_revision: int
    requeued: bool
    unquarantined: bool
    store_generation: int

    def __post_init__(self) -> None:
        for name in ("dead_letter_id", "delivery_id", "subscription_id"):
            _identifier(getattr(self, name), name)
        _integer(self.subscription_revision, "subscription_revision", minimum=1)
        if type(self.requeued) is not bool or type(self.unquarantined) is not bool:
            raise FederationContractError("dead-letter retry flags must be boolean")
        _integer(self.store_generation, "store_generation", minimum=1)


class DurableEventRouterRepository(Protocol):
    """Typed capability implemented by the exclusive canonical state owner."""

    def store_generation(self) -> int:
        """Return the current authoritative store generation."""

    def load_subscription(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
    ) -> EventSubscription:
        """Resolve the exact persisted subscription and its closed bounds."""

    def persist_routed_batch(
        self,
        batch: DurableRouteBatch,
        *,
        idempotency_key: str,
    ) -> DurableRouteCommit:
        """Atomically persist delivery queues and coalescing coverage."""

    def load_routing_state(
        self,
        events: Sequence[DomainEvent],
        subscriptions: Sequence[EventSubscription],
        *,
        maximum_known_deliveries: int,
    ) -> DurableRoutingState:
        """Load bounded pending counts, admitted fanout, and replay identities."""

    def load_deliverable_deliveries(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
        expected_fencing_epoch: int,
    ) -> tuple[DurableQueuedDelivery, ...]:
        """Return bounded pending or replayable deliveries for one owner."""

    def record_delivery_attempt(
        self,
        attempt: DeliveryAttempt,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> DeliveryAttempt:
        """Persist a delivered attempt before it is exposed to its consumer."""

    def record_delivery_failure(
        self,
        failure: DurableDeliveryFailure,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_revision: int,
        retry_budget: int,
        circuit_breaker_failures: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> DurableFailureCommit:
        """Atomically persist failure, retry/dead-letter, and quarantine state."""

    def is_subscription_quarantined(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> bool:
        """Read the durable circuit-breaker state for routing admission."""

    def list_dead_letters(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
    ) -> tuple[DeadLetter, ...]:
        """Read a bounded dead-letter projection from the state owner."""

    def retry_dead_letter(
        self,
        dead_letter_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
        subscription_revision: int,
        expected_fencing_epoch: int,
        recorded_at: str,
        idempotency_key: str,
    ) -> DeadLetterRetryCommit:
        """Resolve, requeue, and unquarantine one owned dead letter atomically."""


def _coverage(delivery: QueuedDelivery) -> CoalescingCoverageRecord:
    decision = delivery.decision
    body = {
        "decision_id": decision.decision_id,
        "subscription_id": delivery.subscription_id,
        "subscription_revision": delivery.subscription_revision,
        "input_event_ids": list(decision.input_event_ids),
    }
    return CoalescingCoverageRecord(
        coverage_id=f"coalescing-coverage:{content_identity(body)}",
        decision_id=decision.decision_id,
        subscription_id=delivery.subscription_id,
        subscription_revision=delivery.subscription_revision,
        representative_event_id=decision.representative_event.event_id,
        input_event_ids=decision.input_event_ids,
        changed_fact_refs=decision.changed_fact_refs,
        mode=decision.mode,
    )


class DurableEventRouter:
    """Persist-first adapter over a fresh bounded planner per outbox batch."""

    def __init__(
        self,
        repository: DurableEventRouterRepository,
        *,
        maximum_subscriptions: int = 4_096,
        maximum_fanout_per_event: int = 256,
        circuit_breaker_failures: int = 16,
        maximum_deliveries_per_commit: int = MAX_DURABLE_DELIVERIES_PER_COMMIT,
    ) -> None:
        if repository is None:
            raise FederationContractError("a canonical state-owner repository is required")
        _integer(maximum_subscriptions, "maximum_subscriptions", minimum=1, maximum=65_536)
        _integer(
            maximum_fanout_per_event,
            "maximum_fanout_per_event",
            minimum=1,
            maximum=4_096,
        )
        _integer(
            circuit_breaker_failures,
            "circuit_breaker_failures",
            minimum=1,
            maximum=1_000,
        )
        _integer(
            maximum_deliveries_per_commit,
            "maximum_deliveries_per_commit",
            minimum=1,
            maximum=MAX_DURABLE_DELIVERIES_PER_COMMIT,
        )
        self._repository = repository
        self._maximum_subscriptions = maximum_subscriptions
        self._maximum_fanout = maximum_fanout_per_event
        self._circuit_breaker_failures = circuit_breaker_failures
        self._maximum_deliveries = maximum_deliveries_per_commit
        self._registration_guard = BoundedEventRouter(
            maximum_subscriptions=maximum_subscriptions,
            maximum_fanout_per_event=maximum_fanout_per_event,
            circuit_breaker_failures=circuit_breaker_failures,
        )

    @property
    def subscriptions(self) -> Mapping[str, EventSubscription]:
        return MappingProxyType(dict(self._registration_guard.subscriptions))

    def register(self, subscription: EventSubscription) -> None:
        """Register an already-authorized canonical subscription revision."""

        if not isinstance(subscription, EventSubscription):
            raise FederationContractError("subscription must be EventSubscription")
        canonical = self._repository.load_subscription(
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            subscription_id=subscription.subscription_id,
        )
        if canonical != subscription:
            raise DurableEventRouterError(
                "registered subscription differs from canonical persisted bounds"
            )
        self._registration_guard.register(canonical)

    def restore_subscription(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
    ) -> EventSubscription:
        """Load and register the canonical persisted revision after restart."""

        canonical = self._repository.load_subscription(
            tenant_id=_identifier(tenant_id, "tenant_id"),
            federation_id=_identifier(federation_id, "federation_id"),
            subscription_id=_identifier(subscription_id, "subscription_id"),
        )
        self._registration_guard.register(canonical)
        return canonical

    def route(
        self,
        events: Sequence[DomainEvent],
        *,
        causal_evaluator: CausalSelectorEvaluator | None = None,
        now: str | None = None,
    ) -> DurableRouteResult:
        """Plan and atomically persist a batch before returning its disposition."""

        values = tuple(events)
        if len(values) > MAX_EVENT_BATCH:
            raise FederationBoundsError("router input batch exceeds the event bound")
        if any(not isinstance(item, DomainEvent) for item in values):
            raise FederationContractError("router input contains a non-event")
        scopes = {(item.tenant_id, item.federation_id) for item in values}
        if len(scopes) > 1:
            raise DurableEventRouterError("durable route crosses authority scope")
        eligible: list[EventSubscription] = []
        for subscription in sorted(
            self._registration_guard.subscriptions.values(),
            key=lambda item: item.subscription_id,
        ):
            if scopes and (
                subscription.tenant_id,
                subscription.federation_id,
            ) not in scopes:
                continue
            if self._repository.is_subscription_quarantined(
                subscription.subscription_id,
                subscription.revision,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
            ):
                continue
            eligible.append(subscription)

        routing_state = self._repository.load_routing_state(
            values,
            eligible,
            maximum_known_deliveries=self._maximum_deliveries,
        )
        effective_fanout = min(
            self._maximum_fanout,
            routing_state.maximum_fanout_per_event,
        )
        planner = BoundedEventRouter(
            maximum_subscriptions=self._maximum_subscriptions,
            maximum_fanout_per_event=effective_fanout,
            circuit_breaker_failures=self._circuit_breaker_failures,
        )
        for subscription in eligible:
            planner.register(subscription)
        state_by_subscription = {
            item.subscription_id: item for item in routing_state.subscriptions
        }
        if set(state_by_subscription) != {
            item.subscription_id for item in eligible
        }:
            raise DurableEventRouterError(
                "state owner omitted canonical subscription routing state"
            )
        for subscription in eligible:
            state = state_by_subscription[subscription.subscription_id]
            if (
                state.subscription_revision != subscription.revision
                or state.maximum_pending != subscription.maximum_pending
            ):
                raise DurableEventRouterError(
                    "state owner returned stale subscription routing bounds"
                )
        planner.seed_durable_state(
            known_delivery_ids=routing_state.known_delivery_ids,
            pending_by_subscription={
                item.subscription_id: item.pending_deliveries
                for item in routing_state.subscriptions
            },
        )

        routing = planner.route(values, causal_evaluator=causal_evaluator, now=now)
        if routing.enqueued_deliveries > self._maximum_deliveries:
            raise FederationBoundsError("planned durable deliveries exceed the commit bound")

        deliveries: list[DurableQueuedDelivery] = []
        for subscription in eligible:
            while True:
                selected = planner.take(
                    subscription.subscription_id,
                    maximum=subscription.maximum_batch,
                )
                if not selected:
                    break
                deliveries.extend(
                    DurableQueuedDelivery(delivery=item, coverage=_coverage(item))
                    for item in selected
                )
        if len(deliveries) != (
            routing.enqueued_deliveries
            + routing.duplicate_deliveries_suppressed
        ):
            raise DurableEventRouterError("planner delivery accounting did not close")

        batch_body = [item.to_dict() for item in deliveries]
        batch = DurableRouteBatch(
            batch_id=f"durable-route:{content_identity({'deliveries': batch_body, 'maximum_fanout_per_event': effective_fanout})}",
            deliveries=tuple(deliveries),
            maximum_fanout_per_event=effective_fanout,
        )
        idempotency_key = f"route-idempotency:{content_identity(batch.to_dict())}"
        commit = self._repository.persist_routed_batch(
            batch,
            idempotency_key=idempotency_key,
        )
        self._validate_route_commit(batch, commit)
        return DurableRouteResult(routing=routing, commit=commit)

    def take(
        self,
        subscription_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
        expected_fencing_epoch: int,
        recorded_at: str,
    ) -> tuple[ExposedDelivery, ...]:
        """Durably record each delivery attempt before returning it."""

        subscription = self._subscription(subscription_id)
        self._assert_scope(
            subscription,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        if subscription.state is not SubscriptionState.ACTIVE:
            raise DurableEventRouterError("subscription is not active for delivery")
        limit = _integer(
            maximum,
            "maximum",
            minimum=1,
            maximum=subscription.maximum_batch,
        )
        fence = _integer(expected_fencing_epoch, "expected_fencing_epoch", minimum=1)
        _timestamp(recorded_at, "recorded_at")
        queued = self._repository.load_deliverable_deliveries(
            subscription.subscription_id,
            subscription.revision,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            maximum=limit,
            expected_fencing_epoch=fence,
        )
        if len(queued) > limit:
            raise DurableEventRouterError("state owner exceeded the requested delivery bound")
        if len({item.delivery.delivery_id for item in queued}) != len(queued):
            raise DurableEventRouterError("state owner returned duplicate deliveries")

        exposed: list[ExposedDelivery] = []
        for item in queued:
            self._validate_loaded_delivery(item, subscription)
            number = item.delivery.attempt_number + 1
            attempt_body = {
                "delivery_id": item.delivery.delivery_id,
                "attempt_number": number,
            }
            attempt = DeliveryAttempt(
                attempt_id=f"delivery-attempt:{content_identity(attempt_body)}",
                event_id=item.delivery.decision.representative_event.event_id,
                subscription_id=subscription.subscription_id,
                consumer_id=subscription.consumer_id,
                attempt_number=number,
                state=DeliveryState.DELIVERED,
                error_code="",
                recorded_at=recorded_at,
            )
            key = f"delivery-attempt-idempotency:{content_identity(attempt.to_dict())}"
            persisted = self._repository.record_delivery_attempt(
                attempt,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                subscription_revision=subscription.revision,
                expected_fencing_epoch=fence,
                idempotency_key=key,
            )
            if persisted != attempt:
                raise DurableEventRouterError("state owner returned a mismatched attempt")
            exposed.append(ExposedDelivery(queued=item, attempt=persisted))
        return tuple(exposed)

    def fail(
        self,
        exposed: ExposedDelivery,
        *,
        tenant_id: str,
        federation_id: str,
        consumer_id: str,
        error_code: str,
        evidence_ref: str,
        recorded_at: str,
        expected_fencing_epoch: int,
        expires_at: str = "",
    ) -> FailureResult:
        """Persist retry/dead-letter/quarantine disposition before returning it."""

        if not isinstance(exposed, ExposedDelivery):
            raise FederationContractError("exposed must be an ExposedDelivery")
        subscription = self._subscription(exposed.attempt.subscription_id)
        self._assert_scope(
            subscription,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        if subscription.state is not SubscriptionState.ACTIVE:
            raise DurableEventRouterError("subscription is not active for failure handling")
        consumer_id = _identifier(consumer_id, "consumer_id")
        if consumer_id != subscription.consumer_id or consumer_id != exposed.attempt.consumer_id:
            raise DurableEventRouterError("delivery is not owned by this consumer")
        fence = _integer(expected_fencing_epoch, "expected_fencing_epoch", minimum=1)
        failure = DurableDeliveryFailure(
            exposed=exposed,
            error_code=error_code,
            evidence_ref=evidence_ref,
            recorded_at=recorded_at,
            expires_at=expires_at,
        )
        failure_body = {
            "attempt_id": exposed.attempt.attempt_id,
            "error_code": failure.error_code,
            "evidence_ref": failure.evidence_ref,
        }
        failure_id = f"delivery-failure:{content_identity(failure_body)}"
        commit = self._repository.record_delivery_failure(
            failure,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            subscription_revision=subscription.revision,
            retry_budget=subscription.retry_budget,
            circuit_breaker_failures=self._circuit_breaker_failures,
            expected_fencing_epoch=fence,
            idempotency_key=f"failure-idempotency:{content_identity(failure_body)}",
        )
        if commit.failure_id != failure_id:
            raise DurableEventRouterError("state owner returned a mismatched failure")
        result = commit.result
        if result.attempt.attempt_id != exposed.attempt.attempt_id:
            raise DurableEventRouterError("failure disposition changed attempt identity")
        if result.attempt.state not in {DeliveryState.RETRY, DeliveryState.DEAD_LETTERED}:
            raise DurableEventRouterError("failure disposition has an illegal state")
        return result

    def dead_letters(
        self,
        subscription_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
    ) -> tuple[DeadLetter, ...]:
        subscription = self._subscription(subscription_id)
        self._assert_scope(
            subscription,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        limit = _integer(maximum, "maximum", minimum=1, maximum=MAX_EVENT_BATCH)
        values = self._repository.list_dead_letters(
            subscription.subscription_id,
            subscription.revision,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            maximum=limit,
        )
        if len(values) > limit or any(not isinstance(item, DeadLetter) for item in values):
            raise DurableEventRouterError("state owner returned invalid dead letters")
        return values

    def retry_dead_letter(
        self,
        dead_letter_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
        expected_fencing_epoch: int,
        recorded_at: str,
        idempotency_key: str,
    ) -> DeadLetterRetryCommit:
        """Atomically requeue an owned dead letter and reset its breaker."""

        subscription = self._subscription(subscription_id)
        self._assert_scope(
            subscription,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        commit = self._repository.retry_dead_letter(
            _identifier(dead_letter_id, "dead_letter_id"),
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            expected_fencing_epoch=_integer(
                expected_fencing_epoch,
                "expected_fencing_epoch",
                minimum=1,
            ),
            recorded_at=_timestamp(recorded_at, "recorded_at"),
            idempotency_key=_identifier(idempotency_key, "idempotency_key"),
        )
        # Retry is also the authorized reactivation transition.  Refresh all
        # persisted revisions so this process cannot retain a quarantined or
        # otherwise stale caller-supplied view after the atomic commit.
        restored: list[EventSubscription] = []
        for current in self._registration_guard.subscriptions.values():
            restored.append(
                self._repository.load_subscription(
                    tenant_id=current.tenant_id,
                    federation_id=current.federation_id,
                    subscription_id=current.subscription_id,
                )
            )
        guard = BoundedEventRouter(
            maximum_subscriptions=self._maximum_subscriptions,
            maximum_fanout_per_event=self._maximum_fanout,
            circuit_breaker_failures=self._circuit_breaker_failures,
        )
        for current in restored:
            guard.register(current)
        self._registration_guard = guard
        return commit

    def _subscription(self, subscription_id: str) -> EventSubscription:
        subscription_id = _identifier(subscription_id, "subscription_id")
        subscription = self._registration_guard.subscriptions.get(subscription_id)
        if subscription is None:
            raise DurableEventRouterError("unknown subscription")
        return subscription

    @staticmethod
    def _assert_scope(
        subscription: EventSubscription,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> None:
        tenant = _identifier(tenant_id, "tenant_id")
        federation = _identifier(federation_id, "federation_id")
        if (
            subscription.tenant_id != tenant
            or subscription.federation_id != federation
        ):
            raise DurableEventRouterError("subscription authority scope differs")

    @staticmethod
    def _validate_loaded_delivery(
        item: DurableQueuedDelivery,
        subscription: EventSubscription,
    ) -> None:
        if not isinstance(item, DurableQueuedDelivery):
            raise DurableEventRouterError("state owner returned an invalid delivery")
        delivery = item.delivery
        if delivery.subscription_id != subscription.subscription_id:
            raise DurableEventRouterError("state owner returned a cross-subscription delivery")
        if delivery.subscription_revision != subscription.revision:
            raise DurableEventRouterError("state owner returned a stale subscription revision")
        if delivery.consumer_id != subscription.consumer_id:
            raise DurableEventRouterError("state owner returned a cross-consumer delivery")

    @staticmethod
    def _validate_route_commit(
        batch: DurableRouteBatch,
        commit: DurableRouteCommit,
    ) -> None:
        if not isinstance(commit, DurableRouteCommit):
            raise DurableEventRouterError("state owner returned an invalid route commit")
        if commit.batch_id != batch.batch_id:
            raise DurableEventRouterError("state owner returned a mismatched route batch")
        expected = {item.delivery.delivery_id for item in batch.deliveries}
        disposed = set(commit.inserted_delivery_ids) | set(commit.existing_delivery_ids)
        if disposed != expected:
            raise DurableEventRouterError("state owner did not dispose every planned delivery")


__all__ = [
    "CoalescingCoverageRecord",
    "DeadLetterRetryCommit",
    "DurableDeliveryFailure",
    "DurableEventRouter",
    "DurableEventRouterError",
    "DurableEventRouterRepository",
    "DurableFailureCommit",
    "DurableQueuedDelivery",
    "DurableRoutingBackpressure",
    "DurableRoutingState",
    "DurableRouteBatch",
    "DurableRouteCommit",
    "DurableRouteResult",
    "DurableSubscriptionRoutingState",
    "ExposedDelivery",
    "MAX_DURABLE_DELIVERIES_PER_COMMIT",
]
