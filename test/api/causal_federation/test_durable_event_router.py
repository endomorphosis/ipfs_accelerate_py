"""Hermetic durability qualification for the CASF event-router adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.durable_event_router import (
    DeadLetterRetryCommit,
    DurableDeliveryFailure,
    DurableEventRouter,
    DurableEventRouterError,
    DurableFailureCommit,
    DurableQueuedDelivery,
    DurableRouteBatch,
    DurableRouteCommit,
    DurableRoutingState,
    DurableSubscriptionRoutingState,
)
from ipfs_accelerate_py.agent_supervisor.federation.event_router import (
    CoalescingMode,
    FailureResult,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DeadLetter,
    DeliveryAttempt,
    DeliveryState,
    EventClass,
    EventEffectClass,
    EventSubscription,
    SubscriptionState,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from test.api.causal_federation.test_contracts import EXPIRY
from test.api.causal_federation.test_registry import (
    _create,
    _open_repository,
    _reopen_repository,
)

NOW = "2026-08-21T12:00:00Z"
LATER = "2099-01-01T00:00:00Z"


def event(
    sequence: int,
    *,
    event_type: EventClass = EventClass.TASK_READY,
    effect_class: EventEffectClass = EventEffectClass.AUTHORITATIVE_STATE,
    deduplication_key: str | None = None,
):
    draft = EventDraft(
        event_type=event_type,
        stream_id="stream:federation-1",
        causal_parent_ids=(),
        correlation_id="correlation:durable-router-test",
        causation_id=f"causation:{sequence}",
        tenant_id="tenant:one",
        federation_id="federation:one",
        supervisor_id="supervisor:one",
        task_id=f"task:{sequence}",
        repository_id="repository:one",
        tree_id="tree:one",
        symbol_id="symbol:target",
        resource_class="cpu",
        payload_ref=f"artifact:event-{sequence}",
        changed_fact_refs=(f"fact:event-{sequence}",),
        effect_class=effect_class,
        expires_at="",
        deduplication_key=deduplication_key or f"dedup:{sequence}",
    )
    materialized, _ = materialize_event(
        draft,
        stream_sequence=sequence,
        global_sequence=sequence,
        recorded_at=NOW,
    )
    return materialized


def subscription(*, retry_budget: int = 1) -> EventSubscription:
    return EventSubscription(
        subscription_id="subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        consumer_id="consumer:durable",
        revision=1,
        event_classes=tuple(EventClass),
        selectors=(),
        maximum_batch=8,
        maximum_pending=32,
        retry_budget=retry_budget,
        expires_at=LATER,
        state=SubscriptionState.ACTIVE,
    )


class InMemoryStateOwnerRepository:
    """One-authority fake; all mutations are typed and idempotent."""

    def __init__(self) -> None:
        self.deliveries: dict[str, DurableQueuedDelivery] = {}
        self.subscriptions: dict[tuple[str, str, str], EventSubscription] = {}
        self.delivery_states: dict[str, str] = {}
        self.route_commits: dict[str, DurableRouteCommit] = {}
        self.attempts: dict[str, DeliveryAttempt] = {}
        self.attempt_delivery: dict[str, str] = {}
        self.failure_commits: dict[str, DurableFailureCommit] = {}
        self.failure_counts: dict[tuple[str, int], int] = {}
        self.quarantined: set[tuple[str, int]] = set()
        self.dead_letter_values: dict[str, DeadLetter] = {}
        self.coverage: dict[str, object] = {}
        self.log: list[str] = []
        self.generation = 1
        self.reject_route_commit = False

    def load_subscription(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
    ) -> EventSubscription:
        return self.subscriptions[(tenant_id, federation_id, subscription_id)]

    def store_generation(self) -> int:
        return self.generation

    def load_routing_state(
        self,
        events,
        subscriptions,
        *,
        maximum_known_deliveries: int,
    ) -> DurableRoutingState:
        event_ids = {item.event_id for item in events}
        known = tuple(
            delivery_id
            for delivery_id, record in self.deliveries.items()
            if set(record.coverage.input_event_ids).intersection(event_ids)
        )
        assert len(known) <= maximum_known_deliveries
        states = tuple(
            DurableSubscriptionRoutingState(
                subscription_id=item.subscription_id,
                subscription_revision=item.revision,
                pending_deliveries=sum(
                    1
                    for delivery_id, record in self.deliveries.items()
                    if record.delivery.subscription_id == item.subscription_id
                    and self.delivery_states[delivery_id]
                    in {"pending", "retry", "delivered"}
                ),
                maximum_pending=item.maximum_pending,
            )
            for item in subscriptions
        )
        return DurableRoutingState(
            known_delivery_ids=known,
            subscriptions=states,
            maximum_fanout_per_event=256,
            store_generation=self.generation,
        )

    def persist_routed_batch(
        self,
        batch: DurableRouteBatch,
        *,
        idempotency_key: str,
    ) -> DurableRouteCommit:
        self.log.append(f"persist-route:{batch.batch_id}")
        if self.reject_route_commit:
            raise RuntimeError("synthetic state-owner transaction failure")
        prior = self.route_commits.get(idempotency_key)
        if prior is not None:
            return prior
        inserted: list[str] = []
        existing: list[str] = []
        for record in batch.deliveries:
            delivery_id = record.delivery.delivery_id
            if delivery_id in self.deliveries:
                existing.append(delivery_id)
                continue
            self.deliveries[delivery_id] = record
            self.delivery_states[delivery_id] = "pending"
            self.coverage[record.coverage.coverage_id] = record.coverage
            inserted.append(delivery_id)
        self.generation += 1
        result = DurableRouteCommit(
            batch_id=batch.batch_id,
            inserted_delivery_ids=tuple(inserted),
            existing_delivery_ids=tuple(existing),
            store_generation=self.generation,
        )
        self.route_commits[idempotency_key] = result
        return result

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
        self.log.append(f"load-deliverable:{subscription_id}")
        assert (tenant_id, federation_id) == ("tenant:one", "federation:one")
        assert expected_fencing_epoch == 1
        if (subscription_id, subscription_revision) in self.quarantined:
            return ()
        values = (
            record
            for delivery_id, record in self.deliveries.items()
            if record.delivery.subscription_id == subscription_id
            and record.delivery.subscription_revision == subscription_revision
            and self.delivery_states[delivery_id] in {"pending", "delivered"}
        )
        return tuple(values)[:maximum]

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
        self.log.append(f"record-attempt:{attempt.attempt_id}")
        assert (tenant_id, federation_id) == ("tenant:one", "federation:one")
        assert expected_fencing_epoch == 1
        prior = self.attempts.get(idempotency_key)
        if prior is not None:
            return prior
        match = next(
            (
                (delivery_id, record)
                for delivery_id, record in self.deliveries.items()
                if record.delivery.subscription_id == attempt.subscription_id
                and record.delivery.subscription_revision == subscription_revision
                and record.delivery.consumer_id == attempt.consumer_id
                and record.delivery.decision.representative_event.event_id == attempt.event_id
                and record.delivery.attempt_number + 1 == attempt.attempt_number
                and self.delivery_states[delivery_id] in {"pending", "delivered"}
            ),
            None,
        )
        if match is None:
            raise RuntimeError("attempt has no current queued delivery")
        delivery_id, _ = match
        self.delivery_states[delivery_id] = "delivered"
        self.attempt_delivery[attempt.attempt_id] = delivery_id
        self.attempts[idempotency_key] = attempt
        return attempt

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
        self.log.append(f"record-failure:{failure.exposed.attempt.attempt_id}")
        assert (tenant_id, federation_id) == ("tenant:one", "federation:one")
        assert expected_fencing_epoch == 1
        prior = self.failure_commits.get(idempotency_key)
        if prior is not None:
            return prior
        exposed_attempt = failure.exposed.attempt
        delivery_id = self.attempt_delivery[exposed_attempt.attempt_id]
        record = self.deliveries[delivery_id]
        key = (record.delivery.subscription_id, subscription_revision)
        failures = self.failure_counts.get(key, 0) + 1
        self.failure_counts[key] = failures
        quarantined = failures >= circuit_breaker_failures
        if quarantined:
            self.quarantined.add(key)
            scope = (tenant_id, federation_id, record.delivery.subscription_id)
            self.subscriptions[scope] = replace(
                self.subscriptions[scope],
                state=SubscriptionState.QUARANTINED,
            )

        exhausted = (
            exposed_attempt.attempt_number > retry_budget
            or quarantined
        )
        state = DeliveryState.DEAD_LETTERED if exhausted else DeliveryState.RETRY
        attempt = replace(
            exposed_attempt,
            state=state,
            error_code=failure.error_code,
            recorded_at=failure.recorded_at,
        )
        if exhausted:
            dead_letter = DeadLetter(
                dead_letter_id=(
                    "dead-letter:"
                    + content_identity(
                        {
                            "delivery_id": delivery_id,
                            "attempt": exposed_attempt.attempt_number,
                        }
                    )
                ),
                event_id=exposed_attempt.event_id,
                subscription_id=exposed_attempt.subscription_id,
                consumer_id=exposed_attempt.consumer_id,
                retry_count=exposed_attempt.attempt_number,
                error_code=failure.error_code,
                evidence_ref=failure.evidence_ref,
                quarantined=quarantined,
                created_at=failure.recorded_at,
                expires_at=failure.expires_at,
            )
            self.dead_letter_values[dead_letter.dead_letter_id] = dead_letter
            self.delivery_states[delivery_id] = "dead-lettered"
        else:
            dead_letter = None
            self.deliveries[delivery_id] = DurableQueuedDelivery(
                delivery=replace(
                    record.delivery,
                    attempt_number=exposed_attempt.attempt_number,
                ),
                coverage=record.coverage,
            )
            self.delivery_states[delivery_id] = "pending"

        result = FailureResult(
            attempt=attempt,
            dead_letter=dead_letter,
            retry_scheduled=not exhausted,
            subscription_quarantined=quarantined,
        )
        body = {
            "attempt_id": exposed_attempt.attempt_id,
            "error_code": failure.error_code,
            "evidence_ref": failure.evidence_ref,
        }
        self.generation += 1
        commit = DurableFailureCommit(
            failure_id=f"delivery-failure:{content_identity(body)}",
            result=result,
            store_generation=self.generation,
        )
        self.failure_commits[idempotency_key] = commit
        return commit

    def is_subscription_quarantined(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> bool:
        self.log.append(f"read-quarantine:{subscription_id}")
        assert (tenant_id, federation_id) == ("tenant:one", "federation:one")
        return (subscription_id, subscription_revision) in self.quarantined

    def list_dead_letters(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
    ) -> tuple[DeadLetter, ...]:
        assert (tenant_id, federation_id) == ("tenant:one", "federation:one")
        del subscription_revision
        return tuple(
            item
            for item in self.dead_letter_values.values()
            if item.subscription_id == subscription_id
        )[:maximum]

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
        del recorded_at, idempotency_key
        assert expected_fencing_epoch == 1
        dead_letter = self.dead_letter_values.pop(dead_letter_id)
        assert dead_letter.subscription_id == subscription_id
        matching = next(
            (item for item in self.deliveries.values() if item.delivery.decision.representative_event.event_id == dead_letter.event_id),
            None,
        )
        assert matching is not None
        delivery_id = matching.delivery.delivery_id
        self.deliveries[delivery_id] = DurableQueuedDelivery(
            delivery=replace(
                matching.delivery,
                attempt_number=dead_letter.retry_count,
            ),
            coverage=matching.coverage,
        )
        self.delivery_states[delivery_id] = "pending"
        key = (subscription_id, subscription_revision)
        self.failure_counts[key] = 0
        self.quarantined.discard(key)
        scope = (tenant_id, federation_id, subscription_id)
        self.subscriptions[scope] = replace(
            self.subscriptions[scope],
            state=SubscriptionState.ACTIVE,
        )
        self.generation += 1
        return DeadLetterRetryCommit(
            dead_letter_id=dead_letter_id,
            delivery_id=delivery_id,
            subscription_id=subscription_id,
            subscription_revision=subscription_revision,
            requeued=True,
            unquarantined=True,
            store_generation=self.generation,
        )


def adapter(repository: InMemoryStateOwnerRepository, **kwargs: int) -> DurableEventRouter:
    record = subscription()
    repository.subscriptions.setdefault(
        (record.tenant_id, record.federation_id, record.subscription_id),
        record,
    )
    router = DurableEventRouter(repository, **kwargs)
    router.restore_subscription(
        tenant_id=record.tenant_id,
        federation_id=record.federation_id,
        subscription_id=record.subscription_id,
    )
    return router


def test_route_is_persisted_before_any_delivery_is_exposed() -> None:
    repository = InMemoryStateOwnerRepository()
    router = adapter(repository)

    routed = router.route((event(1),), now=NOW)

    assert routed.routing.enqueued_deliveries == 1
    assert len(repository.deliveries) == 1
    assert len(repository.coverage) == 1
    exposed = router.take(
        "subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        maximum=1,
        expected_fencing_epoch=1,
        recorded_at=NOW,
    )
    assert len(exposed) == 1
    persist_index = next(
        index for index, value in enumerate(repository.log) if value.startswith("persist-route:")
    )
    attempt_index = next(
        index for index, value in enumerate(repository.log) if value.startswith("record-attempt:")
    )
    assert persist_index < attempt_index
    assert exposed[0].attempt in repository.attempts.values()


def test_failed_route_commit_exposes_nothing_and_replay_can_recover() -> None:
    repository = InMemoryStateOwnerRepository()
    repository.reject_route_commit = True
    router = adapter(repository)

    with pytest.raises(RuntimeError, match="state-owner transaction failure"):
        router.route((event(1),), now=NOW)

    assert repository.deliveries == {}
    assert (
        router.take(
            "subscription:durable",
            tenant_id="tenant:one",
            federation_id="federation:one",
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )
        == ()
    )
    repository.reject_route_commit = False
    assert router.route((event(1),), now=NOW).routing.enqueued_deliveries == 1
    assert len(repository.deliveries) == 1


def test_replay_is_idempotent_across_router_process_reconstruction() -> None:
    repository = InMemoryStateOwnerRepository()
    first = adapter(repository)
    first_result = first.route((event(1),), now=NOW)
    replay_result = first.route((event(1),), now=NOW)
    reconstructed = adapter(repository)
    restart_replay = reconstructed.route((event(1),), now=NOW)

    assert len(repository.deliveries) == 1
    assert len(repository.coverage) == 1
    assert replay_result.commit == first_result.commit
    assert restart_replay.commit == first_result.commit


def test_retry_dead_letter_and_quarantine_survive_adapter_restart() -> None:
    repository = InMemoryStateOwnerRepository()
    first = adapter(repository, circuit_breaker_failures=2)
    first.route((event(1),), now=NOW)
    initial = first.take(
        "subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        maximum=1,
        expected_fencing_epoch=1,
        recorded_at=NOW,
    )[0]
    retry = first.fail(
        initial,
        tenant_id="tenant:one",
        federation_id="federation:one",
        consumer_id="consumer:durable",
        error_code="consumer_crash",
        evidence_ref="evidence:first-crash",
        recorded_at=NOW,
        expected_fencing_epoch=1,
    )
    assert retry.retry_scheduled is True
    assert retry.attempt.state is DeliveryState.RETRY

    reconstructed = adapter(repository, circuit_breaker_failures=2)
    second = reconstructed.take(
        "subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        maximum=1,
        expected_fencing_epoch=1,
        recorded_at=NOW,
    )[0]
    assert second.attempt.attempt_number == 2
    exhausted = reconstructed.fail(
        second,
        tenant_id="tenant:one",
        federation_id="federation:one",
        consumer_id="consumer:durable",
        error_code="consumer_crash",
        evidence_ref="evidence:second-crash",
        recorded_at=NOW,
        expected_fencing_epoch=1,
        expires_at=LATER,
    )
    assert exhausted.retry_scheduled is False
    assert exhausted.subscription_quarantined is True
    assert exhausted.dead_letter is not None

    after_restart = adapter(repository, circuit_breaker_failures=2)
    assert after_restart.dead_letters(
        "subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        maximum=4,
    ) == (exhausted.dead_letter,)
    blocked = after_restart.route((event(2),), now=NOW)
    assert blocked.routing.enqueued_deliveries == 0
    assert len(repository.deliveries) == 1
    retry_commit = after_restart.retry_dead_letter(
        exhausted.dead_letter.dead_letter_id,
        tenant_id="tenant:one",
        federation_id="federation:one",
        subscription_id="subscription:durable",
        expected_fencing_epoch=1,
        recorded_at=NOW,
        idempotency_key="dead-letter-retry:durable",
    )
    assert retry_commit.requeued is True
    assert retry_commit.unquarantined is True
    assert after_restart.dead_letters(
        "subscription:durable",
        tenant_id="tenant:one",
        federation_id="federation:one",
        maximum=4,
    ) == ()
    assert after_restart.route((event(2),), now=NOW).routing.enqueued_deliveries == 1


def test_registration_rejects_caller_bounds_that_differ_from_persisted_state() -> None:
    repository = InMemoryStateOwnerRepository()
    canonical = subscription()
    repository.subscriptions[
        (canonical.tenant_id, canonical.federation_id, canonical.subscription_id)
    ] = canonical
    router = DurableEventRouter(repository)

    with pytest.raises(DurableEventRouterError, match="canonical persisted bounds"):
        router.register(replace(canonical, maximum_pending=canonical.maximum_pending + 1))


def test_safety_events_have_singular_persisted_coverage() -> None:
    repository = InMemoryStateOwnerRepository()
    router = adapter(repository)
    events = (
        event(
            1,
            event_type=EventClass.LEASE_EXPIRING,
            effect_class=EventEffectClass.LEASE_OR_FENCE,
            deduplication_key="dedup:safety",
        ),
        event(
            2,
            event_type=EventClass.LEASE_EXPIRING,
            effect_class=EventEffectClass.LEASE_OR_FENCE,
            deduplication_key="dedup:safety",
        ),
    )

    result = router.route(events, now=NOW)

    assert result.routing.enqueued_deliveries == 2
    assert len(repository.coverage) == 2
    assert all(item.mode is CoalescingMode.NONE for item in repository.coverage.values())
    assert all(len(item.input_event_ids) == 1 for item in repository.coverage.values())


def test_take_fails_closed_for_nonactive_registered_subscription() -> None:
    repository = InMemoryStateOwnerRepository()
    paused = replace(subscription(), state=SubscriptionState.PAUSED)
    repository.subscriptions[(paused.tenant_id, paused.federation_id, paused.subscription_id)] = paused
    router = DurableEventRouter(repository)
    router.register(paused)

    with pytest.raises(DurableEventRouterError, match="not active"):
        router.take(
            "subscription:durable",
            tenant_id="tenant:one",
            federation_id="federation:one",
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_real_repository_routing_retry_dead_letter_survives_restart(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    subscription_record = EventSubscription(
        subscription_id="subscription:real-durable",
        tenant_id=binding.tenant_id,
        federation_id="federation:pending",
        consumer_id="consumer:real-durable",
        revision=1,
        event_classes=(EventClass.GOAL_CHANGED,),
        selectors=(),
        maximum_batch=4,
        maximum_pending=8,
        retry_budget=1,
        expires_at=EXPIRY,
        state=SubscriptionState.ACTIVE,
    )
    routed_event = None
    first_commit = None
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription_record = replace(
            subscription_record,
            federation_id=identity.record_id,
        )
        repository.register_subscription(
            subscription_record,
            idempotency_key="subscription-register:real-durable",
        )
        routed_event = repository._events_for_loaded_subscription(
            subscription_record,
            after_cursor=0,
            maximum_events=1,
        )[0]
        router = DurableEventRouter(repository, circuit_breaker_failures=2)
        router.register(subscription_record)
        first_commit = router.route((routed_event,), now=NOW).commit
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        restarted = DurableEventRouter(repository, circuit_breaker_failures=2)
        restarted.register(subscription_record)
        assert restarted.route((routed_event,), now=NOW).commit == first_commit
        first = restarted.take(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )[0]
        retried = restarted.fail(
            first,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            consumer_id=subscription_record.consumer_id,
            error_code="consumer_crash",
            evidence_ref="evidence:real-first-crash",
            recorded_at=NOW,
            expected_fencing_epoch=1,
        )
        assert retried.retry_scheduled is True
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        restarted = DurableEventRouter(repository, circuit_breaker_failures=2)
        restarted.register(subscription_record)
        second = restarted.take(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )[0]
        assert second.attempt.attempt_number == 2
        exhausted = restarted.fail(
            second,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            consumer_id=subscription_record.consumer_id,
            error_code="consumer_crash",
            evidence_ref="evidence:real-second-crash",
            recorded_at=NOW,
            expected_fencing_epoch=1,
            expires_at=EXPIRY,
        )
        assert exhausted.dead_letter is not None
        assert exhausted.subscription_quarantined is True
        assert restarted.dead_letters(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=4,
        ) == (exhausted.dead_letter,)
        retry_commit = restarted.retry_dead_letter(
            exhausted.dead_letter.dead_letter_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            subscription_id=subscription_record.subscription_id,
            expected_fencing_epoch=1,
            recorded_at=NOW,
            idempotency_key="dead-letter-retry:real-durable",
        )
        assert retry_commit.requeued is True
        assert retry_commit.unquarantined is True
        assert restarted.dead_letters(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=4,
        ) == ()
        replayed = restarted.take(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )[0]
        assert replayed.attempt.attempt_number == 3
    finally:
        client.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB is required")
def test_breaker_trip_dead_letters_before_large_retry_budget_after_restart(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    subscription_record = EventSubscription(
        subscription_id="subscription:breaker-before-budget",
        tenant_id=binding.tenant_id,
        federation_id="federation:pending",
        consumer_id="consumer:breaker-before-budget",
        revision=1,
        event_classes=(EventClass.GOAL_CHANGED,),
        selectors=(),
        maximum_batch=4,
        maximum_pending=8,
        retry_budget=8,
        expires_at=EXPIRY,
        state=SubscriptionState.ACTIVE,
    )
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription_record = replace(
            subscription_record,
            federation_id=identity.record_id,
        )
        repository.register_subscription(
            subscription_record,
            idempotency_key="subscription-register:breaker-before-budget",
        )
        routed_event = repository._events_for_loaded_subscription(
            subscription_record,
            after_cursor=0,
            maximum_events=1,
        )[0]
        router = DurableEventRouter(repository, circuit_breaker_failures=2)
        router.register(subscription_record)
        router.route((routed_event,), now=NOW)
        first = router.take(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )[0]
        retry = router.fail(
            first,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            consumer_id=subscription_record.consumer_id,
            error_code="consumer_crash",
            evidence_ref="evidence:breaker-first",
            recorded_at=NOW,
            expected_fencing_epoch=1,
        )
        assert retry.retry_scheduled is True
        assert retry.dead_letter is None
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        restarted = DurableEventRouter(repository, circuit_breaker_failures=2)
        restarted.register(subscription_record)
        second = restarted.take(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=1,
            expected_fencing_epoch=1,
            recorded_at=NOW,
        )[0]
        terminal = restarted.fail(
            second,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            consumer_id=subscription_record.consumer_id,
            error_code="consumer_crash",
            evidence_ref="evidence:breaker-second",
            recorded_at=NOW,
            expected_fencing_epoch=1,
            expires_at=EXPIRY,
        )
        assert terminal.retry_scheduled is False
        assert terminal.subscription_quarantined is True
        assert terminal.dead_letter is not None
        assert terminal.attempt.state is DeliveryState.DEAD_LETTERED
        assert restarted.dead_letters(
            subscription_record.subscription_id,
            tenant_id=subscription_record.tenant_id,
            federation_id=subscription_record.federation_id,
            maximum=4,
        ) == (terminal.dead_letter,)
    finally:
        client.close()
