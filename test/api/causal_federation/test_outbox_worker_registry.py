"""Real DuckDB integration qualification for the outbox routing worker."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.federation.durable_event_router import (
    DurableEventRouter,
    DurableQueuedDelivery,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DeliveryState,
    DomainEvent,
    EventAcknowledgement,
    EventClass,
    EventSelector,
    EventSubscription,
    SelectorKind,
    SubscriptionState,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox_worker import (
    EventDrivenOutboxWorker,
    OutboxScope,
    StateOwnerOutboxWake,
)
from ipfs_accelerate_py.agent_supervisor.federation.registry import (
    FederationStateRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from test.api.causal_federation.test_contracts import EXPIRY, NOW
from test.api.causal_federation.test_registry import (
    _create,
    _insert_event_fixtures,
    _open_repository,
    _reopen_repository,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for outbox-worker repository integration tests",
)


@dataclass(frozen=True)
class RoutingHarness:
    database: Path
    client: QuackStateClient
    repository: FederationStateRepository
    scope: OutboxScope
    subscription: EventSubscription
    events: tuple[DomainEvent, ...]


def _provision(tmp_path: Path) -> RoutingHarness:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path
    )
    identity, _ = _create(repository, request=request, policy=policy)
    subscription = EventSubscription(
        subscription_id="subscription:outbox-worker-real",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
        consumer_id="consumer:outbox-worker-real",
        revision=1,
        event_classes=tuple(EventClass),
        selectors=(),
        maximum_batch=64,
        maximum_pending=256,
        retry_budget=3,
        expires_at=EXPIRY,
        state=SubscriptionState.ACTIVE,
    )
    repository.register_subscription(
        subscription,
        idempotency_key="subscription-register:outbox-worker-real",
    )
    scope = OutboxScope(binding.tenant_id, identity.record_id)
    scopes = repository.pending_outbox_scopes(maximum=8)
    events = repository.pending_outbox_events(scope, maximum=32)
    assert scopes == (scope,)
    assert events
    assert repository.active_subscription_ids(scope, maximum=8) == (
        subscription.subscription_id,
    )
    return RoutingHarness(
        database=database,
        client=client,
        repository=repository,
        scope=scope,
        subscription=subscription,
        events=events,
    )


def _worker(repository: FederationStateRepository) -> EventDrivenOutboxWorker:
    return EventDrivenOutboxWorker(
        repository,
        lambda _scope: DurableEventRouter(repository),
        StateOwnerOutboxWake(),
        maximum_scopes=8,
        maximum_events_per_scope=32,
        maximum_subscriptions_per_scope=8,
    )


def _worker_with_wake(
    repository: FederationStateRepository,
    wake: StateOwnerOutboxWake,
) -> EventDrivenOutboxWorker:
    return EventDrivenOutboxWorker(
        repository,
        lambda _scope: DurableEventRouter(repository),
        wake,
        maximum_scopes=8,
        maximum_events_per_scope=32,
        maximum_subscriptions_per_scope=8,
    )


def _ack_next(
    repository: FederationStateRepository,
    client: QuackStateClient,
    subscription: EventSubscription,
    *,
    ordinal: int,
) -> DomainEvent:
    router = DurableEventRouter(repository)
    router.register(subscription)
    exposed = router.take(
        subscription.subscription_id,
        tenant_id=subscription.tenant_id,
        federation_id=subscription.federation_id,
        maximum=1,
        expected_fencing_epoch=client.load_generation().fence_epoch,
        recorded_at=NOW,
    )[0]
    event = exposed.queued.delivery.decision.representative_event
    cursor = repository.get_cursor(
        tenant_id=subscription.tenant_id,
        federation_id=subscription.federation_id,
        consumer_id=subscription.consumer_id,
        subscription_id=subscription.subscription_id,
    )
    acknowledgement = EventAcknowledgement(
        acknowledgement_id=f"acknowledgement:capacity:{ordinal}",
        event_id=event.event_id,
        consumer_id=subscription.consumer_id,
        subscription_id=subscription.subscription_id,
        subscription_revision=subscription.revision,
        global_sequence=event.global_sequence,
        processed_effect_ref=f"effect:capacity:{ordinal}",
        recorded_at=NOW,
    )
    repository.acknowledge_event(
        acknowledgement,
        tenant_id=subscription.tenant_id,
        federation_id=subscription.federation_id,
        delivery_attempt_id=exposed.attempt.attempt_id,
        expected_cursor_revision=cursor.revision,
        expected_fencing_epoch=client.load_generation().fence_epoch,
        idempotency_key=f"acknowledge:capacity:{ordinal}",
    )
    assert exposed.attempt.state is DeliveryState.DELIVERED
    return event


def _visible_queue(harness: RoutingHarness) -> tuple[DurableQueuedDelivery, ...]:
    return harness.repository.load_deliverable_deliveries(
        harness.subscription.subscription_id,
        harness.subscription.revision,
        tenant_id=harness.scope.tenant_id,
        federation_id=harness.scope.federation_id,
        maximum=harness.subscription.maximum_pending,
        expected_fencing_epoch=harness.client.load_generation().fence_epoch,
    )


def _database_projection(database: Path) -> dict[str, list[tuple[object, ...]]]:
    with open_duckdb_connection(database) as connection:
        def rows(sql: str) -> list[tuple[object, ...]]:
            values = connection.execute(sql).fetchall()
            return [
                tuple(row[index] for index in range(len(row)))
                for row in values
            ]

        return {
            "dispositions": rows(
                """
                SELECT disposition_id, route_batch_id, first_global_sequence,
                       last_global_sequence, event_count, delivery_count,
                       subscription_count, status, revision, content_ref,
                       body_json
                FROM outbox_routing_dispositions
                ORDER BY first_global_sequence, disposition_id
                """
            ),
            "members": rows(
                """
                SELECT disposition_id, event_id, global_sequence, ordinal
                FROM outbox_routing_disposition_events
                ORDER BY disposition_id, ordinal
                """
            ),
            "outbox": rows(
                """
                SELECT event_id, global_sequence, status, revision
                FROM transactional_outbox
                ORDER BY global_sequence, event_id
                """
            ),
            "queue": rows(
                """
                SELECT delivery_id, subscription_id, status, attempt_number
                FROM event_delivery_queue
                ORDER BY delivery_id
                """
            ),
            "coverage": rows(
                """
                SELECT coverage_id, subscription_id, input_event_count,
                       coalescing_mode
                FROM event_coalescing_coverage
                ORDER BY coverage_id
                """
            ),
            "inputs": rows(
                """
                SELECT coverage_id, event_id, ordinal
                FROM event_coalescing_inputs
                ORDER BY coverage_id, ordinal
                """
            ),
        }


def test_real_worker_routes_marks_and_exposes_normalized_durable_state(
    tmp_path: Path,
) -> None:
    harness = _provision(tmp_path)
    source_ids = tuple(item.event_id for item in harness.events)
    source_sequences = tuple(item.global_sequence for item in harness.events)
    routed_queue: tuple[DurableQueuedDelivery, ...] = ()
    first_receipt = None
    try:
        assert harness.repository.pending_outbox_scopes(maximum=1) == (
            harness.scope,
        )
        assert harness.repository.pending_outbox_events(
            harness.scope,
            maximum=1,
        ) == harness.events[:1]
        assert harness.repository.active_subscription_ids(
            harness.scope,
            maximum=1,
        ) == (harness.subscription.subscription_id,)

        outbox_worker = _worker(harness.repository)
        first_receipt = outbox_worker.drain_once()
        generation_after_first = harness.client.load_generation()
        routed_queue = _visible_queue(harness)

        assert first_receipt.scope_count == 1
        assert first_receipt.event_count == len(harness.events)
        assert first_receipt.delivery_count == len(routed_queue) > 0
        assert first_receipt.routed_global_sequence == max(source_sequences)
        assert first_receipt.dispositions[0].event_ids == source_ids
        assert harness.repository.pending_outbox_scopes(maximum=8) == ()
        assert {
            event_id
            for queued in routed_queue
            for event_id in queued.coverage.input_event_ids
        } == set(source_ids)

        second_receipt = outbox_worker.drain_once()
        assert second_receipt.scope_count == 0
        assert second_receipt.event_count == 0
        assert second_receipt.delivery_count == 0
        assert second_receipt.dispositions == ()
        assert second_receipt.routed_global_sequence == (
            first_receipt.routed_global_sequence
        )
        assert harness.client.load_generation() == generation_after_first
        assert tuple(item.delivery.delivery_id for item in _visible_queue(harness)) == (
            tuple(item.delivery.delivery_id for item in routed_queue)
        )
    finally:
        harness.client.close()

    assert first_receipt is not None
    projection = _database_projection(harness.database)
    assert len(projection["dispositions"]) == 1
    disposition = projection["dispositions"][0]
    assert disposition[0] == first_receipt.dispositions[0].disposition_id
    assert str(disposition[1]).startswith("durable-route:")
    assert disposition[2:9] == (
        min(source_sequences),
        max(source_sequences),
        len(source_ids),
        first_receipt.delivery_count,
        1,
        "committed",
        1,
    )
    body = json.loads(str(disposition[10]))
    assert body["event_ids"] == list(source_ids)
    assert body["global_sequences"] == list(source_sequences)
    assert body["delivery_count"] == first_receipt.delivery_count
    assert body["subscription_count"] == 1
    assert projection["members"] == [
        (
            first_receipt.dispositions[0].disposition_id,
            event.event_id,
            event.global_sequence,
            ordinal,
        )
        for ordinal, event in enumerate(harness.events, start=1)
    ]
    assert projection["outbox"] == [
        (event.event_id, event.global_sequence, "routed", 2)
        for event in harness.events
    ]
    assert len(projection["queue"]) == first_receipt.delivery_count
    assert all(
        row[1:] == (harness.subscription.subscription_id, "pending", 0)
        for row in projection["queue"]
    )
    assert len(projection["coverage"]) == first_receipt.delivery_count
    assert {str(row[1]) for row in projection["coverage"]} == {
        harness.subscription.subscription_id
    }
    assert {str(row[1]) for row in projection["inputs"]} == set(source_ids)


def test_child_event_reads_require_exact_durable_subscription_routing(
    tmp_path: Path,
) -> None:
    harness = _provision(tmp_path)
    routed_queue: tuple[DurableQueuedDelivery, ...] = ()
    unrelated_class = next(
        event_class
        for event_class in EventClass
        if all(event.event_type is not event_class for event in harness.events)
    )
    unrelated_subscription = EventSubscription(
        subscription_id="subscription:same-federation-unrelated",
        tenant_id=harness.scope.tenant_id,
        federation_id=harness.scope.federation_id,
        consumer_id="consumer:same-federation-unrelated",
        revision=1,
        event_classes=(unrelated_class,),
        selectors=(),
        maximum_batch=8,
        maximum_pending=8,
        retry_budget=3,
        expires_at=EXPIRY,
        state=SubscriptionState.ACTIVE,
    )
    harness.repository.register_subscription(
        unrelated_subscription,
        idempotency_key="subscription-register:same-federation-unrelated",
    )
    source = harness.events[0]
    owner_scope = {
        "event_id": source.event_id,
        "tenant_id": harness.scope.tenant_id,
        "federation_id": harness.scope.federation_id,
    }
    matching_child_scope = {
        **owner_scope,
        "subscription_id": harness.subscription.subscription_id,
        "subscription_revision": harness.subscription.revision,
        "consumer_id": harness.subscription.consumer_id,
    }
    try:
        # The exclusive owner must be able to validate an event and outbox before
        # routing creates a durable queue row.  A child cannot use mere
        # same-federation subscription existence to obtain either record.
        assert len(
            harness.client.execute("casf_select_event_for_routing", owner_scope)
        ) == 1
        assert len(
            harness.client.execute("casf_select_outbox_for_routing", owner_scope)
        ) == 1
        assert not harness.client.execute(
            "casf_select_event_for_ack", matching_child_scope
        )
        assert not harness.client.execute(
            "casf_select_outbox_for_delivery", matching_child_scope
        )

        receipt = _worker(harness.repository).drain_once()
        routed_queue = _visible_queue(harness)
        assert receipt.delivery_count == len(routed_queue) > 0
        representative = routed_queue[0].delivery.decision.representative_event
        owner_scope = {
            "event_id": representative.event_id,
            "tenant_id": harness.scope.tenant_id,
            "federation_id": harness.scope.federation_id,
        }
        matching_child_scope = {
            **owner_scope,
            "subscription_id": harness.subscription.subscription_id,
            "subscription_revision": harness.subscription.revision,
            "consumer_id": harness.subscription.consumer_id,
        }
        unrelated_child_scope = {
            **owner_scope,
            "subscription_id": unrelated_subscription.subscription_id,
            "subscription_revision": unrelated_subscription.revision,
            "consumer_id": unrelated_subscription.consumer_id,
        }

        assert len(
            harness.client.execute("casf_select_event_for_ack", matching_child_scope)
        ) == 1
        assert len(
            harness.client.execute(
                "casf_select_outbox_for_delivery", matching_child_scope
            )
        ) == 1
        assert not harness.client.execute(
            "casf_select_event_for_ack", unrelated_child_scope
        )
        assert not harness.client.execute(
            "casf_select_outbox_for_delivery", unrelated_child_scope
        )
        assert not harness.repository.load_deliverable_deliveries(
            unrelated_subscription.subscription_id,
            unrelated_subscription.revision,
            tenant_id=harness.scope.tenant_id,
            federation_id=harness.scope.federation_id,
            maximum=8,
            expected_fencing_epoch=harness.client.load_generation().fence_epoch,
        )
    finally:
        harness.client.close()


def test_disposition_failure_rolls_back_members_and_marks_then_replays(
    tmp_path: Path,
) -> None:
    harness = _provision(tmp_path)
    source_ids = tuple(item.event_id for item in harness.events)
    failure_phase = "after_outbox_disposition_before_mark"

    def fail(phase: str) -> None:
        if phase == failure_phase:
            raise RuntimeError(f"injected failure at {phase}")

    harness.repository._test_failure_hook = fail
    failed_worker = _worker(harness.repository)
    queue_before_replay: tuple[DurableQueuedDelivery, ...] = ()
    try:
        with pytest.raises(RuntimeError, match=failure_phase):
            failed_worker.drain_once()
        assert failed_worker.watermark == 0
        assert harness.repository.pending_outbox_events(
            harness.scope,
            maximum=32,
        ) == harness.events
        queue_before_replay = _visible_queue(harness)
        assert queue_before_replay
    finally:
        harness.client.close()

    failed_projection = _database_projection(harness.database)
    assert failed_projection["dispositions"] == []
    assert failed_projection["members"] == []
    assert all(row[2:] == ("pending", 1) for row in failed_projection["outbox"])
    assert len(failed_projection["queue"]) == len(queue_before_replay)
    assert len(failed_projection["coverage"]) == len(queue_before_replay)
    assert {str(row[1]) for row in failed_projection["inputs"]} == set(source_ids)

    client, repository = _reopen_repository(harness.database)
    recovered_queue: tuple[DurableQueuedDelivery, ...] = ()
    recovered_receipt = None
    try:
        assert repository.pending_outbox_scopes(maximum=8) == (harness.scope,)
        recovered_receipt = _worker(repository).drain_once()
        recovered_queue = repository.load_deliverable_deliveries(
            harness.subscription.subscription_id,
            harness.subscription.revision,
            tenant_id=harness.scope.tenant_id,
            federation_id=harness.scope.federation_id,
            maximum=harness.subscription.maximum_pending,
            expected_fencing_epoch=client.load_generation().fence_epoch,
        )
        assert repository.pending_outbox_scopes(maximum=8) == ()
    finally:
        client.close()

    assert recovered_receipt is not None
    assert recovered_receipt.event_count == len(harness.events)
    assert recovered_receipt.delivery_count == len(queue_before_replay)
    assert tuple(item.delivery.delivery_id for item in recovered_queue) == tuple(
        item.delivery.delivery_id for item in queue_before_replay
    )
    recovered_projection = _database_projection(harness.database)
    assert len(recovered_projection["dispositions"]) == 1
    assert len(recovered_projection["members"]) == len(harness.events)
    assert all(row[2:] == ("routed", 2) for row in recovered_projection["outbox"])
    assert recovered_projection["queue"] == failed_projection["queue"]
    assert recovered_projection["coverage"] == failed_projection["coverage"]
    assert recovered_projection["inputs"] == failed_projection["inputs"]


def test_durable_pending_capacity_retries_after_restart_and_ack_wake(
    tmp_path: Path,
) -> None:
    wake = StateOwnerOutboxWake()
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        outbox_notifier=wake.notify_committed,
    )
    identity, _ = _create(repository, request=request, policy=policy)
    subscription = EventSubscription(
        subscription_id="subscription:capacity-one",
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
        consumer_id="consumer:capacity-one",
        revision=1,
        event_classes=tuple(EventClass),
        selectors=(),
        maximum_batch=8,
        maximum_pending=8,
        retry_budget=3,
        expires_at=EXPIRY,
        state=SubscriptionState.ACTIVE,
    )
    repository.register_subscription(
        subscription,
        maximum_fanout=8,
        idempotency_key="subscription-register:capacity-one",
    )
    scope = OutboxScope(binding.tenant_id, identity.record_id)
    client.close()
    _insert_event_fixtures(
        database,
        binding=binding,
        federation_id=identity.record_id,
        repository_ids=(binding.repository_ids[0],) * 8,
    )
    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            UPDATE transactional_outbox
            SET next_attempt_at = '2026-01-01T00:00:00Z'
            WHERE status = 'pending'
            """
        )
    client, repository = _reopen_repository(
        database,
        outbox_notifier=wake.notify_committed,
    )
    source_events = repository.pending_outbox_events(scope, maximum=32)
    first = _worker_with_wake(repository, wake).drain_once()
    assert first.scope_count == 0
    assert first.retryable_scopes == (scope,)
    assert first.delivery_count == subscription.maximum_pending
    assert repository.pending_outbox_events(scope, maximum=32) == source_events
    client.close()

    restarted_wake = StateOwnerOutboxWake()
    client, repository = _reopen_repository(
        database,
        outbox_notifier=restarted_wake.notify_committed,
    )
    try:
        restarted = _worker_with_wake(repository, restarted_wake)
        still_full = restarted.drain_once()
        assert still_full.retryable_scopes == (scope,)
        assert still_full.delivery_count == 0

        prior_sequence = 0
        completed = None
        releases_required = len(source_events) - subscription.maximum_pending
        for ordinal in range(1, releases_required + 1):
            acknowledged = _ack_next(
                repository,
                client,
                subscription,
                ordinal=ordinal,
            )
            assert acknowledged.global_sequence > prior_sequence
            prior_sequence = acknowledged.global_sequence
            completed = restarted.wait_and_drain(
                deadline_monotonic=time.monotonic() + 1.0
            )
            assert completed is not None
            if ordinal < releases_required:
                assert completed.retryable_scopes == (scope,)
                assert completed.delivery_count == 1
        assert completed is not None
        assert completed.scope_count == 1
        assert completed.retryable_scopes == ()
        assert completed.delivery_count == len(source_events)
        assert repository.pending_outbox_scopes(maximum=8) == ()
    finally:
        client.close()

    projection = _database_projection(database)
    delivery_ids = tuple(str(row[0]) for row in projection["queue"])
    assert len(delivery_ids) == len(source_events)
    assert len(set(delivery_ids)) == len(delivery_ids)
    released = len(source_events) - subscription.maximum_pending
    assert [row[2] for row in projection["queue"]].count("acknowledged") == released
    assert [row[2] for row in projection["queue"]].count("pending") == (
        subscription.maximum_pending
    )
    assert len(projection["dispositions"]) == 1


def test_fanout_pages_across_restart_without_loss_or_duplicate_delivery(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path
    )
    identity, _ = _create(repository, request=request, policy=policy)
    scope = OutboxScope(binding.tenant_id, identity.record_id)
    subscriptions = tuple(
        EventSubscription(
            subscription_id=f"subscription:fanout:{ordinal}",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id=f"consumer:fanout:{ordinal}",
            revision=1,
            event_classes=tuple(EventClass),
            selectors=(),
            maximum_batch=32,
            maximum_pending=32,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        for ordinal in range(3)
    )
    for ordinal, subscription in enumerate(subscriptions):
        repository.register_subscription(
            subscription,
            maximum_fanout=2,
            idempotency_key=f"subscription-register:fanout:{ordinal}",
        )
    source_events = repository.pending_outbox_events(scope, maximum=32)
    first = _worker(repository).drain_once()
    assert first.scope_count == 0
    assert first.retryable_scopes == (scope,)
    assert first.delivery_count == len(source_events) * 2
    assert repository.pending_outbox_events(scope, maximum=32) == source_events
    client.close()

    client, repository = _reopen_repository(database)
    try:
        completed = _worker(repository).drain_once()
        assert completed.scope_count == 1
        assert completed.retryable_scopes == ()
        assert completed.delivery_count == len(source_events) * 3
        assert repository.pending_outbox_scopes(maximum=8) == ()
    finally:
        client.close()

    projection = _database_projection(database)
    delivery_ids = tuple(str(row[0]) for row in projection["queue"])
    assert len(delivery_ids) == len(source_events) * 3
    assert len(set(delivery_ids)) == len(delivery_ids)
    assert len(projection["inputs"]) == len(source_events) * 3
    assert len(projection["dispositions"]) == 1


def test_unavailable_causal_selector_is_rejected_without_poisoning_router(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path
    )
    del database
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        causal = EventSubscription(
            subscription_id="subscription:causal-unavailable",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:causal-unavailable",
            revision=1,
            event_classes=tuple(EventClass),
            selectors=(
                EventSelector(
                    kind=SelectorKind.CAUSAL_ANCESTOR,
                    value="causal-node:not-admitted",
                ),
            ),
            maximum_batch=8,
            maximum_pending=8,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        with pytest.raises(FederationAuthorityError, match="unavailable"):
            repository.register_subscription(
                causal,
                idempotency_key="subscription-register:causal-unavailable",
            )
        ordinary = EventSubscription(
            subscription_id="subscription:ordinary-survivor",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:ordinary-survivor",
            revision=1,
            event_classes=tuple(EventClass),
            selectors=(),
            maximum_batch=8,
            maximum_pending=16,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        repository.register_subscription(
            ordinary,
            idempotency_key="subscription-register:ordinary-survivor",
        )
        receipt = _worker(repository).drain_once()
        assert receipt.scope_count == 1
        assert receipt.retryable_scopes == ()
        assert repository.active_subscription_ids(
            OutboxScope(binding.tenant_id, identity.record_id),
            maximum=8,
        ) == (ordinary.subscription_id,)
    finally:
        client.close()
