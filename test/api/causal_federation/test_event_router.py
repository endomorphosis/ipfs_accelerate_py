"""Focused CASF event-router coalescing, delivery, and failure tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.event_router import (
    BoundedEventRouter,
    CoalescingMode,
    DeliveryOwnershipError,
    plan_coalescing,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DeliveryState,
    EventClass,
    EventEffectClass,
    EventSelector,
    EventSubscription,
    SelectorKind,
    SubscriptionState,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)

NOW = "2026-08-21T12:00:00Z"
LATER = "2099-01-01T00:00:00Z"


def event(
    sequence: int,
    *,
    event_type: EventClass = EventClass.TASK_READY,
    changed_fact_refs: tuple[str, ...] = ("fact:task",),
    effect_class: EventEffectClass = EventEffectClass.AUTHORITATIVE_STATE,
    resource_class: str = "cpu",
    symbol_id: str = "symbol:target",
    deduplication_key: str | None = None,
    expires_at: str = "",
):
    draft = EventDraft(
        event_type=event_type,
        stream_id="stream:federation-1",
        causal_parent_ids=(),
        correlation_id="correlation:router-test",
        causation_id=f"causation:{sequence}",
        tenant_id="tenant:one",
        federation_id="federation:one",
        supervisor_id="supervisor:one",
        task_id=f"task:{sequence}",
        repository_id="repository:one",
        tree_id="tree:one",
        symbol_id=symbol_id,
        resource_class=resource_class,
        payload_ref=f"artifact:event-{sequence}",
        changed_fact_refs=changed_fact_refs,
        effect_class=effect_class,
        expires_at=expires_at,
        deduplication_key=deduplication_key or f"dedup:{sequence}",
    )
    materialized, _ = materialize_event(
        draft,
        stream_sequence=sequence,
        global_sequence=sequence,
        recorded_at=NOW,
    )
    return materialized


def subscription(
    suffix: str = "one",
    *,
    revision: int = 1,
    event_classes: tuple[EventClass, ...] = (EventClass.TASK_READY,),
    maximum_batch: int = 4,
    maximum_pending: int = 8,
    retry_budget: int = 1,
    expires_at: str = LATER,
    state: SubscriptionState = SubscriptionState.ACTIVE,
    selectors: tuple[EventSelector, ...] = (),
) -> EventSubscription:
    return EventSubscription(
        subscription_id=f"subscription:{suffix}",
        tenant_id="tenant:one",
        federation_id="federation:one",
        consumer_id=f"consumer:{suffix}",
        revision=revision,
        event_classes=event_classes,
        selectors=selectors,
        maximum_batch=maximum_batch,
        maximum_pending=maximum_pending,
        retry_budget=retry_budget,
        expires_at=expires_at,
        state=state,
    )


def test_symbol_changes_coalesce_to_one_union_without_losing_fact_order() -> None:
    events = (
        event(
            1,
            event_type=EventClass.SYMBOL_CHANGED,
            symbol_id="symbol:a",
            changed_fact_refs=("symbol:a", "symbol:shared"),
        ),
        event(
            2,
            event_type=EventClass.SYMBOL_CHANGED,
            symbol_id="symbol:b",
            changed_fact_refs=("symbol:shared", "symbol:b"),
        ),
        event(
            3,
            event_type=EventClass.SYMBOL_CHANGED,
            symbol_id="symbol:c",
            changed_fact_refs=("symbol:c",),
        ),
    )

    decisions = plan_coalescing(events)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.mode is CoalescingMode.UNION_CHANGED_FACTS
    assert decision.representative_event == events[-1]
    assert decision.input_event_ids == tuple(item.event_id for item in events)
    assert decision.changed_fact_refs == (
        "symbol:a",
        "symbol:shared",
        "symbol:b",
        "symbol:c",
    )


def test_mixed_symbol_coalescing_wakes_selector_for_an_earlier_symbol() -> None:
    router = BoundedEventRouter()
    symbol_a = subscription(
        "symbol-a",
        event_classes=(EventClass.SYMBOL_CHANGED,),
        selectors=(EventSelector(SelectorKind.SYMBOL, "symbol:a"),),
    )
    router.register(symbol_a)
    changed = (
        event(
            1,
            event_type=EventClass.SYMBOL_CHANGED,
            symbol_id="symbol:a",
            changed_fact_refs=("symbol:a",),
        ),
        event(
            2,
            event_type=EventClass.SYMBOL_CHANGED,
            symbol_id="symbol:b",
            changed_fact_refs=("symbol:b",),
        ),
    )

    routed = router.route(changed, now=NOW)
    delivered = router.take(symbol_a.subscription_id, maximum=1)

    assert routed.enqueued_deliveries == 1
    assert len(delivered) == 1
    assert delivered[0].decision.input_event_ids == (changed[0].event_id,)
    assert delivered[0].decision.representative_event == changed[0]
    assert delivered[0].decision.changed_fact_refs == ("symbol:a",)
    assert delivered[0].decision.mode is CoalescingMode.NONE


def test_provider_capacity_coalescing_delivers_latest_generation_only() -> None:
    older = event(
        1,
        event_type=EventClass.PROVIDER_CAPACITY_CHANGED,
        resource_class="provider:gpu",
        changed_fact_refs=("capacity:generation:1",),
    )
    latest = event(
        2,
        event_type=EventClass.PROVIDER_CAPACITY_CHANGED,
        resource_class="provider:gpu",
        changed_fact_refs=("capacity:generation:2",),
    )

    (decision,) = plan_coalescing((older, latest))

    assert decision.mode is CoalescingMode.LATEST_GENERATION
    assert decision.representative_event is latest
    assert decision.input_event_ids == (older.event_id, latest.event_id)
    assert decision.changed_fact_refs == latest.changed_fact_refs


@pytest.mark.parametrize(
    ("event_type", "effect_class"),
    (
        (EventClass.LEASE_EXPIRING, EventEffectClass.LEASE_OR_FENCE),
        (EventClass.PROOF_COMPLETED, EventEffectClass.PROOF_LINEAGE),
        (EventClass.TASK_COMPLETED, EventEffectClass.PAYMENT),
    ),
)
def test_safety_events_are_never_coalesced(
    event_type: EventClass,
    effect_class: EventEffectClass,
) -> None:
    events = (
        event(
            1,
            event_type=event_type,
            effect_class=effect_class,
            deduplication_key="dedup:safety",
        ),
        event(
            2,
            event_type=event_type,
            effect_class=effect_class,
            deduplication_key="dedup:safety",
        ),
    )

    decisions = plan_coalescing(events)

    assert len(decisions) == 2
    assert all(item.mode is CoalescingMode.NONE for item in decisions)
    assert all(len(item.input_event_ids) == 1 for item in decisions)


def test_duplicate_delivery_is_suppressed_across_replay() -> None:
    router = BoundedEventRouter()
    registered = subscription()
    router.register(registered)
    task_event = event(1)

    first = router.route((task_event,), now=NOW)
    replay = router.route((task_event,), now=NOW)

    assert first.enqueued_deliveries == 1
    assert first.duplicate_deliveries_suppressed == 0
    assert replay.enqueued_deliveries == 0
    assert replay.duplicate_deliveries_suppressed == 1
    assert len(router.take(registered.subscription_id, maximum=1)) == 1


def test_fanout_and_pending_ceilings_apply_backpressure_without_overfill() -> None:
    fanout_router = BoundedEventRouter(maximum_fanout_per_event=2)
    subscriptions = tuple(subscription(str(index)) for index in range(3))
    for item in subscriptions:
        fanout_router.register(item)

    fanout = fanout_router.route((event(1),), now=NOW)

    assert fanout.enqueued_deliveries == 2
    assert fanout.matched_subscriptions == 2
    assert fanout.backpressured_subscriptions == (subscriptions[-1].subscription_id,)

    pending_router = BoundedEventRouter()
    bounded = subscription("bounded", maximum_batch=1, maximum_pending=1)
    pending_router.register(bounded)
    assert pending_router.route((event(10),), now=NOW).enqueued_deliveries == 1

    blocked = pending_router.route((event(11),), now=NOW)

    assert blocked.enqueued_deliveries == 0
    assert blocked.backpressured_subscriptions == (bounded.subscription_id,)
    assert len(pending_router.take(bounded.subscription_id, maximum=1)) == 1
    assert pending_router.take(bounded.subscription_id, maximum=1) == ()


def test_expired_events_and_subscriptions_never_enter_delivery_queues() -> None:
    router = BoundedEventRouter()
    active = subscription("active")
    expired_subscription = subscription(
        "expired",
        expires_at="2026-08-21T11:59:59Z",
    )
    router.register(active)
    router.register(expired_subscription)
    expired_event = event(1, expires_at=NOW)

    result = router.route((expired_event,), now=NOW)

    assert result.expired_events == 1
    assert result.enqueued_deliveries == 0
    assert result.matched_subscriptions == 0
    assert router.take(active.subscription_id, maximum=1) == ()
    assert router.take(expired_subscription.subscription_id, maximum=1) == ()


def test_retry_budget_exhaustion_creates_one_evidenced_dead_letter() -> None:
    router = BoundedEventRouter(circuit_breaker_failures=10)
    registered = subscription("retry", maximum_batch=1, retry_budget=1)
    router.register(registered)
    router.route((event(1),), now=NOW)
    first_delivery = router.take(registered.subscription_id, maximum=1)[0]

    first_failure = router.fail(
        first_delivery.delivery_id,
        consumer_id=registered.consumer_id,
        error_code="provider_unavailable",
        evidence_ref="evidence:first-failure",
        recorded_at=NOW,
    )
    assert first_failure.retry_scheduled is True
    assert first_failure.dead_letter is None
    assert first_failure.attempt.state is DeliveryState.RETRY
    assert first_failure.attempt.attempt_number == 1

    retry = router.take(registered.subscription_id, maximum=1)[0]
    exhausted = router.fail(
        retry.delivery_id,
        consumer_id=registered.consumer_id,
        error_code="provider_unavailable",
        evidence_ref="evidence:retry-exhausted",
        recorded_at=NOW,
        expires_at=LATER,
    )

    assert exhausted.retry_scheduled is False
    assert exhausted.attempt.state is DeliveryState.DEAD_LETTERED
    assert exhausted.attempt.attempt_number == 2
    assert exhausted.dead_letter is not None
    assert exhausted.dead_letter.retry_count == 2
    assert exhausted.dead_letter.evidence_ref == "evidence:retry-exhausted"
    assert router.dead_letters == (exhausted.dead_letter,)
    assert router.take(registered.subscription_id, maximum=1) == ()


def test_circuit_breaker_retry_from_old_revision_is_not_released_after_reactivation() -> None:
    router = BoundedEventRouter(circuit_breaker_failures=1)
    registered = subscription("circuit", maximum_batch=1, retry_budget=3)
    router.register(registered)
    router.route((event(1),), now=NOW)
    delivery = router.take(registered.subscription_id, maximum=1)[0]

    failed = router.fail(
        delivery.delivery_id,
        consumer_id=registered.consumer_id,
        error_code="consumer_crash",
        evidence_ref="evidence:consumer-crash",
        recorded_at=NOW,
    )

    assert failed.retry_scheduled is True
    assert failed.subscription_quarantined is True
    assert router.take(registered.subscription_id, maximum=1) == ()
    assert router.route((event(2),), now=NOW).enqueued_deliveries == 0

    router.register(replace(registered, revision=2))
    assert router.take(registered.subscription_id, maximum=1) == ()

    assert router.route((event(2),), now=NOW).enqueued_deliveries == 1
    (fresh,) = router.take(registered.subscription_id, maximum=1)
    assert fresh.subscription_revision == 2
    assert fresh.attempt_number == 0


def test_selector_revision_discards_queued_work_from_the_old_selector() -> None:
    router = BoundedEventRouter()
    original = subscription(
        "selector-revision",
        event_classes=(EventClass.SYMBOL_CHANGED,),
        selectors=(EventSelector(SelectorKind.SYMBOL, "symbol:a"),),
    )
    router.register(original)
    changed_a = event(1, event_type=EventClass.SYMBOL_CHANGED, symbol_id="symbol:a")
    changed_b = event(2, event_type=EventClass.SYMBOL_CHANGED, symbol_id="symbol:b")
    assert router.route((changed_a,), now=NOW).enqueued_deliveries == 1

    revised = replace(
        original,
        revision=2,
        selectors=(EventSelector(SelectorKind.SYMBOL, "symbol:b"),),
    )
    router.register(revised)

    assert router.take(original.subscription_id, maximum=1) == ()
    assert router.route((changed_a,), now=NOW).enqueued_deliveries == 0
    assert router.route((changed_b,), now=NOW).enqueued_deliveries == 1
    (delivery,) = router.take(original.subscription_id, maximum=1)
    assert delivery.subscription_revision == revised.revision
    assert delivery.decision.representative_event == changed_b


def test_non_active_revision_discards_queue_and_cannot_serve_or_route() -> None:
    router = BoundedEventRouter()
    active = subscription("paused-revision")
    router.register(active)
    assert router.route((event(1),), now=NOW).enqueued_deliveries == 1

    router.register(
        replace(active, revision=2, state=SubscriptionState.PAUSED)
    )

    assert router.take(active.subscription_id, maximum=1) == ()
    assert router.route((event(2),), now=NOW).enqueued_deliveries == 0


def test_revision_change_revokes_an_already_exposed_delivery() -> None:
    router = BoundedEventRouter()
    original = subscription("inflight-revision", maximum_batch=1)
    router.register(original)
    router.route((event(1),), now=NOW)
    delivery = router.take(original.subscription_id, maximum=1)[0]

    router.register(replace(original, revision=2))

    with pytest.raises(DeliveryOwnershipError):
        router.acknowledge(delivery.delivery_id, consumer_id=original.consumer_id)


def test_only_the_delivery_owner_can_acknowledge_or_fail() -> None:
    router = BoundedEventRouter()
    registered = subscription("owner", maximum_batch=1)
    router.register(registered)
    router.route((event(1),), now=NOW)
    delivery = router.take(registered.subscription_id, maximum=1)[0]

    with pytest.raises(DeliveryOwnershipError):
        router.acknowledge(delivery.delivery_id, consumer_id="consumer:intruder")
    with pytest.raises(DeliveryOwnershipError):
        router.fail(
            delivery.delivery_id,
            consumer_id="consumer:intruder",
            error_code="forged_failure",
            evidence_ref="evidence:forged",
            recorded_at=NOW,
        )

    router.acknowledge(delivery.delivery_id, consumer_id=registered.consumer_id)
    with pytest.raises(DeliveryOwnershipError):
        router.acknowledge(delivery.delivery_id, consumer_id=registered.consumer_id)
