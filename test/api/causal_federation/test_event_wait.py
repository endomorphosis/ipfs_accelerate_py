from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import events
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationAuthorityError,
    FederationBoundsError,
    FederationContractError,
    UnknownNormativeFieldError,
)
from ipfs_accelerate_py.agent_supervisor.federation.event_wait import (
    StaleSubscriptionError,
    StateOwnerEventWait,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.federation.subscriptions import (
    CausalSelectorUnavailable,
    assert_subscription_scope,
    event_matches_subscription,
)

NOW = "2030-01-01T00:00:00Z"
EXPIRY = "2099-01-01T00:00:00Z"


def deadline(seconds: float) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=seconds)  # noqa: UP017
    return value.isoformat().replace("+00:00", "Z")


def sample_draft(
    *,
    event_type: events.EventClass = events.EventClass.TASK_READY,
    effect_class: events.EventEffectClass = events.EventEffectClass.AUTHORITATIVE_STATE,
) -> EventDraft:
    return EventDraft(
        event_type=event_type,
        stream_id="stream:test",
        causal_parent_ids=(),
        correlation_id="correlation:test",
        causation_id="causation:test",
        tenant_id="tenant:test",
        federation_id="federation:test",
        supervisor_id="supervisor:test",
        task_id="task:test",
        repository_id="repo:test",
        tree_id="tree:test",
        goal_id="goal:test",
        subgoal_id="subgoal:test",
        symbol_id="symbol:test",
        contract_id="contract:test",
        proof_obligation_id="proof:test",
        resource_class="resource.cpu",
        payload_ref="payload:test",
        changed_fact_refs=("fact:test",),
        effect_class=effect_class,
        expires_at=EXPIRY,
        deduplication_key="dedupe:test",
    )


def sample_event(
    sequence: int = 1,
    *,
    event_type: events.EventClass = events.EventClass.TASK_READY,
    effect_class: events.EventEffectClass = events.EventEffectClass.AUTHORITATIVE_STATE,
) -> events.DomainEvent:
    event, _ = materialize_event(
        sample_draft(event_type=event_type, effect_class=effect_class),
        stream_sequence=sequence,
        global_sequence=sequence,
        recorded_at=NOW,
    )
    return event


def sample_subscription(
    *,
    selectors: tuple[events.EventSelector, ...] = (),
    revision: int = 1,
    state: events.SubscriptionState = events.SubscriptionState.ACTIVE,
) -> events.EventSubscription:
    return events.EventSubscription(
        subscription_id="subscription:test",
        tenant_id="tenant:test",
        federation_id="federation:test",
        consumer_id="consumer:test",
        revision=revision,
        event_classes=(events.EventClass.TASK_READY,),
        selectors=selectors,
        maximum_batch=32,
        maximum_pending=128,
        retry_budget=3,
        expires_at=EXPIRY,
        state=state,
    )


def sample_wait_request(
    *,
    wait_seconds: float = 1.0,
    subscription_revision: int = 1,
) -> events.EventWaitRequest:
    return events.EventWaitRequest(
        consumer_id="consumer:test",
        after_cursor=0,
        subscription_id="subscription:test",
        subscription_revision=subscription_revision,
        deadline=deadline(wait_seconds),
        maximum_events=32,
    )


class InMemoryEventSource:
    def __init__(self, *, subscription_revision: int = 1) -> None:
        self._lock = threading.Lock()
        self._events: list[events.DomainEvent] = []
        self._subscription_revision = subscription_revision
        self.queries = 0
        self.query_observed = threading.Event()

    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[events.DomainEvent, ...]:
        del consumer_id, subscription_id
        if subscription_revision != self._subscription_revision:
            raise StaleSubscriptionError("subscription revision differs")
        with self._lock:
            self.queries += 1
            available = tuple(
                event for event in self._events if event.global_sequence > after_cursor
            )[:maximum_events]
        self.query_observed.set()
        return available

    def store_generation(self) -> int:
        return 1

    def append(self, event: events.DomainEvent) -> None:
        with self._lock:
            self._events.append(event)


def event_contract_samples() -> tuple[Any, ...]:
    event = sample_event()
    _, outbox = materialize_event(
        sample_draft(),
        stream_sequence=1,
        global_sequence=1,
        recorded_at=NOW,
    )
    selector = events.EventSelector(events.SelectorKind.TASK, "task:test")
    subscription = sample_subscription(selectors=(selector,))
    request = events.EventWaitRequest(
        consumer_id="consumer:test",
        after_cursor=0,
        subscription_id="subscription:test",
        subscription_revision=1,
        deadline=EXPIRY,
        maximum_events=32,
    )
    return (
        event,
        selector,
        subscription,
        events.ConsumerCursor(
            consumer_id="consumer:test",
            subscription_id="subscription:test",
            subscription_revision=1,
            global_sequence=1,
            store_generation=1,
            revision=1,
            updated_at=NOW,
        ),
        request,
        events.EventBatch(
            consumer_id="consumer:test",
            subscription_id="subscription:test",
            subscription_revision=1,
            after_cursor=0,
            next_cursor=1,
            store_generation=1,
            events=(event,),
            timed_out=False,
            cancelled=False,
            server_shutdown=False,
        ),
        events.DeliveryAttempt(
            attempt_id="delivery:test",
            event_id=event.event_id,
            subscription_id="subscription:test",
            consumer_id="consumer:test",
            attempt_number=1,
            state=events.DeliveryState.DELIVERED,
            error_code="",
            recorded_at=NOW,
        ),
        events.DeadLetter(
            dead_letter_id="dead-letter:test",
            event_id=event.event_id,
            subscription_id="subscription:test",
            consumer_id="consumer:test",
            retry_count=3,
            error_code="delivery.failed",
            evidence_ref="evidence:test",
            quarantined=True,
            created_at=NOW,
            expires_at=EXPIRY,
        ),
        events.EventAcknowledgement(
            acknowledgement_id="ack:test",
            event_id=event.event_id,
            consumer_id="consumer:test",
            subscription_id="subscription:test",
            subscription_revision=1,
            global_sequence=1,
            processed_effect_ref="effect:test",
            recorded_at=NOW,
        ),
        outbox,
    )


@pytest.mark.parametrize(
    "record",
    event_contract_samples(),
    ids=lambda record: type(record).__name__,
)
def test_event_and_outbox_contract_round_trips(record: Any) -> None:
    contract_type = type(record)
    decoded = contract_type.from_dict(record.to_dict())

    assert decoded == record
    assert decoded.cid == record.cid


@pytest.mark.parametrize(
    "record",
    event_contract_samples(),
    ids=lambda record: type(record).__name__,
)
def test_event_and_outbox_contracts_reject_unknown_fields(record: Any) -> None:
    payload = record.to_dict()
    payload["sql"] = "SELECT * FROM control"

    with pytest.raises(UnknownNormativeFieldError):
        type(record).from_dict(payload)


def test_materialized_event_and_outbox_have_deterministic_bound_identities() -> None:
    first_event, first_outbox = materialize_event(
        sample_draft(),
        stream_sequence=1,
        global_sequence=7,
        recorded_at=NOW,
    )
    second_event, second_outbox = materialize_event(
        sample_draft(),
        stream_sequence=1,
        global_sequence=7,
        recorded_at=NOW,
    )

    assert first_event == second_event
    assert first_outbox == second_outbox
    assert first_event.event_id.startswith("event:")
    assert first_outbox.outbox_id.startswith("outbox:")
    assert first_outbox.event_id == first_event.event_id
    assert first_outbox.event_cid == first_event.event_cid
    assert first_outbox.global_sequence == first_event.global_sequence


@pytest.mark.parametrize(
    "effect_class",
    [
        events.EventEffectClass.LEASE_OR_FENCE,
        events.EventEffectClass.EXTERNAL_IRREVERSIBLE,
        events.EventEffectClass.SECURITY_OR_LEGAL,
        events.EventEffectClass.PAYMENT,
        events.EventEffectClass.PROOF_LINEAGE,
    ],
)
def test_safety_events_cannot_be_coalesced(
    effect_class: events.EventEffectClass,
) -> None:
    assert sample_event(effect_class=effect_class).coalescing_forbidden


def test_wait_has_no_lost_wakeup_between_query_and_registration() -> None:
    source = InMemoryEventSource()
    wait = StateOwnerEventWait(source)
    request = sample_wait_request(wait_seconds=2)
    result: list[events.EventBatch] = []
    worker = threading.Thread(target=lambda: result.append(wait.wait_for_events(request)))

    worker.start()
    assert source.query_observed.wait(timeout=1)
    source.append(sample_event())
    wait.notify_committed(1)
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result[0].events == (sample_event(),)
    assert result[0].next_cursor == 1
    assert source.queries == 2
    assert wait.query_count == 2
    assert wait.wakeup_count == 1


def test_idle_deadline_performs_one_query_and_no_periodic_wakeups() -> None:
    source = InMemoryEventSource()
    wait = StateOwnerEventWait(source)

    batch = wait.wait_for_events(sample_wait_request(wait_seconds=0.05))

    assert batch.timed_out
    assert not batch.events
    assert source.queries == 1
    assert wait.query_count == 1
    assert wait.wakeup_count == 0


def test_expired_deadline_returns_immediately_after_one_bounded_query() -> None:
    source = InMemoryEventSource()
    wait = StateOwnerEventWait(source)

    batch = wait.wait_for_events(sample_wait_request(wait_seconds=-1))

    assert batch.timed_out
    assert source.queries == 1
    assert wait.query_count == 1


def test_cancellation_wakes_only_the_blocked_consumer() -> None:
    source = InMemoryEventSource()
    wait = StateOwnerEventWait(source)
    result: list[events.EventBatch] = []
    worker = threading.Thread(
        target=lambda: result.append(wait.wait_for_events(sample_wait_request(wait_seconds=2)))
    )

    worker.start()
    assert source.query_observed.wait(timeout=1)
    wait.cancel("consumer:test")
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert result[0].cancelled
    assert not result[0].events
    assert wait.query_count == 1


def test_duplicate_or_out_of_order_notifications_do_not_create_wake_generations() -> None:
    source = InMemoryEventSource()
    wait = StateOwnerEventWait(source)

    wait.notify_committed(4)
    wait.notify_committed(4)
    wait.notify_committed(3)

    assert wait.notification_generation == 1


def test_stale_subscription_revision_fails_closed() -> None:
    source = InMemoryEventSource(subscription_revision=2)
    wait = StateOwnerEventWait(source)

    with pytest.raises(StaleSubscriptionError):
        wait.wait_for_events(sample_wait_request(wait_seconds=0, subscription_revision=1))


def test_subscription_selectors_are_bounded_and_sql_free() -> None:
    selectors = tuple(
        events.EventSelector(events.SelectorKind.TASK, f"task:{index}") for index in range(1_025)
    )

    with pytest.raises(FederationBoundsError):
        sample_subscription(selectors=selectors)

    selector_payload = events.EventSelector(
        events.SelectorKind.TASK,
        "task:test",
    ).to_dict()
    selector_payload["kind"] = "sql"
    with pytest.raises(ValueError):
        events.EventSelector.from_dict(selector_payload)


def test_subscription_scope_and_closed_selector_matching() -> None:
    subscription = sample_subscription(
        selectors=(
            events.EventSelector(events.SelectorKind.REPOSITORY, "repo:other"),
            events.EventSelector(events.SelectorKind.REPOSITORY, "repo:test"),
            events.EventSelector(events.SelectorKind.TASK, "task:test"),
        )
    )
    event = sample_event()

    assert event_matches_subscription(event, subscription)
    assert not event_matches_subscription(
        replace(event, tenant_id="tenant:other"),
        subscription,
    )
    assert_subscription_scope(
        subscription,
        tenant_id="tenant:test",
        federation_id="federation:test",
        consumer_id="consumer:test",
    )
    with pytest.raises(FederationAuthorityError):
        assert_subscription_scope(
            subscription,
            tenant_id="tenant:other",
            federation_id="federation:test",
            consumer_id="consumer:test",
        )


def test_causal_selector_requires_an_admitted_evaluator() -> None:
    selector = events.EventSelector(
        events.SelectorKind.CAUSAL_DESCENDANT,
        "node:ancestor",
    )
    subscription = sample_subscription(selectors=(selector,))

    with pytest.raises(CausalSelectorUnavailable):
        event_matches_subscription(sample_event(), subscription)

    assert event_matches_subscription(
        sample_event(),
        subscription,
        causal_selector_evaluator=lambda observed, event: (
            observed == selector and event.task_id == "task:test"
        ),
    )


def test_event_batch_rejects_conflicting_terminal_flags() -> None:
    with pytest.raises(FederationContractError):
        events.EventBatch(
            consumer_id="consumer:test",
            subscription_id="subscription:test",
            subscription_revision=1,
            after_cursor=0,
            next_cursor=0,
            store_generation=1,
            events=(),
            timed_out=True,
            cancelled=True,
            server_shutdown=False,
        )


def test_owner_local_wait_capability_does_not_claim_remote_quack_qualification() -> None:
    capability = StateOwnerEventWait(InMemoryEventSource()).capability()

    assert capability["server_owned"] is True
    assert capability["idle_repeated_database_scans"] is False
    assert capability["remote_quack_transport_qualified"] is False
