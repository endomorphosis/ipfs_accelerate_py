"""Hermetic qualification for the event-driven transactional-outbox worker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationBoundsError,
    FederationContractError,
)
from ipfs_accelerate_py.agent_supervisor.federation.durable_event_router import (
    DurableRouteCommit,
    DurableRouteResult,
)
from ipfs_accelerate_py.agent_supervisor.federation.event_router import RouteResult
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DomainEvent,
    EventClass,
    EventEffectClass,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.federation.outbox_worker import (
    MAX_OUTBOX_EVENTS_PER_SCOPE,
    MAX_OUTBOX_SCOPES_PER_DRAIN,
    MAX_SUBSCRIPTIONS_PER_SCOPE,
    EventDrivenOutboxWorker,
    OutboxDisposition,
    OutboxScope,
    OutboxWake,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)

NOW = "2026-08-21T12:00:00Z"


def event(
    global_sequence: int,
    *,
    tenant_id: str = "tenant:one",
    federation_id: str = "federation:one",
    identity: str | None = None,
) -> DomainEvent:
    seed = identity or str(global_sequence)
    draft = EventDraft(
        event_type=EventClass.TASK_READY,
        stream_id="stream:outbox-worker-test",
        causal_parent_ids=(),
        correlation_id="correlation:outbox-worker-test",
        causation_id=f"causation:{seed}",
        tenant_id=tenant_id,
        federation_id=federation_id,
        supervisor_id="supervisor:one",
        task_id=f"task:{seed}",
        repository_id="repository:one",
        tree_id="tree:one",
        symbol_id=f"symbol:{seed}",
        resource_class="cpu",
        payload_ref=f"artifact:event-{seed}",
        changed_fact_refs=(f"fact:event-{seed}",),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        expires_at="",
        deduplication_key=f"dedup:{seed}",
    )
    materialized, _ = materialize_event(
        draft,
        stream_sequence=global_sequence,
        global_sequence=global_sequence,
        recorded_at=NOW,
    )
    return materialized


LogEntry = tuple[Any, ...]


class RecordingRepository:
    """Closed in-memory owner fake with optional idempotent removal."""

    def __init__(
        self,
        scope_events: dict[OutboxScope, Sequence[DomainEvent]],
        subscriptions: dict[OutboxScope, Sequence[str]] | None = None,
        *,
        remove_on_mark: bool = False,
    ) -> None:
        self.scope_order = tuple(scope_events)
        self.pending = {
            scope: list(events) for scope, events in scope_events.items()
        }
        self.subscriptions = {
            scope: tuple(values)
            for scope, values in (subscriptions or {}).items()
        }
        self.remove_on_mark = remove_on_mark
        self.scope_response: tuple[OutboxScope, ...] | None = None
        self.event_responses: dict[OutboxScope, tuple[DomainEvent, ...]] = {}
        self.subscription_responses: dict[OutboxScope, tuple[str, ...]] = {}
        self.log: list[LogEntry] = []
        self.mark_calls: list[LogEntry] = []
        self.mark_failures = 0
        self._generation = 1
        self._dispositions: dict[str, OutboxDisposition] = {}

    def pending_outbox_scopes(self, *, maximum: int) -> tuple[OutboxScope, ...]:
        self.log.append(("scopes", maximum))
        if self.scope_response is not None:
            return self.scope_response
        return tuple(scope for scope in self.scope_order if self.pending[scope])[
            :maximum
        ]

    def pending_outbox_events(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[DomainEvent, ...]:
        self.log.append(("events", scope, maximum))
        if scope in self.event_responses:
            return self.event_responses[scope]
        return tuple(self.pending[scope][:maximum])

    def active_subscription_ids(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[str, ...]:
        self.log.append(("subscriptions", scope, maximum))
        if scope in self.subscription_responses:
            return self.subscription_responses[scope]
        return tuple(self.subscriptions.get(scope, ()))[:maximum]

    def mark_outbox_routed(
        self,
        scope: OutboxScope,
        events: Sequence[DomainEvent],
        *,
        route_batch_id: str,
        delivery_count: int,
        subscription_count: int,
        idempotency_key: str,
    ) -> OutboxDisposition:
        event_ids = tuple(item.event_id for item in events)
        call = (
            "mark",
            scope,
            event_ids,
            route_batch_id,
            idempotency_key,
            delivery_count,
            subscription_count,
        )
        self.log.append(call)
        self.mark_calls.append(call)
        if self.mark_failures:
            self.mark_failures -= 1
            raise RuntimeError("synthetic failure before outbox disposition commit")
        disposition = self._dispositions.get(idempotency_key)
        if disposition is None:
            self._generation += 1
            disposition = OutboxDisposition(
                disposition_id=(
                    f"outbox-disposition:{content_identity({'key': idempotency_key})}"
                ),
                event_ids=event_ids,
                routed_global_sequence=max(item.global_sequence for item in events),
                store_generation=self._generation,
            )
            self._dispositions[idempotency_key] = disposition
        if self.remove_on_mark:
            marked = set(event_ids)
            self.pending[scope] = [
                item for item in self.pending[scope] if item.event_id not in marked
            ]
        return disposition


@dataclass(frozen=True)
class RouteObservation:
    batch_id: str
    event_ids: tuple[str, ...]
    subscription_ids: tuple[str, ...]
    inserted_delivery_ids: tuple[str, ...]
    existing_delivery_ids: tuple[str, ...]


class RecordingRouter:
    def __init__(self, factory: RouterFactory, scope: OutboxScope) -> None:
        self._factory = factory
        self._scope = scope
        self._subscriptions: list[str] = []

    def restore_subscription(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
    ) -> None:
        assert (tenant_id, federation_id) == (
            self._scope.tenant_id,
            self._scope.federation_id,
        )
        self._subscriptions.append(subscription_id)
        self._factory.log.append(("restore", self._scope, subscription_id))

    def route(self, events: Sequence[DomainEvent]) -> DurableRouteResult:
        event_ids = tuple(item.event_id for item in events)
        subscription_ids = tuple(self._subscriptions)
        body = {
            "scope": {
                "tenant_id": self._scope.tenant_id,
                "federation_id": self._scope.federation_id,
            },
            "event_ids": list(event_ids),
            "subscription_ids": list(subscription_ids),
        }
        batch_id = f"durable-route:{content_identity(body)}"
        delivery_ids = tuple(
            f"delivery:{content_identity({'event': event_id, 'subscription': subscription_id})}"
            for subscription_id in subscription_ids
            for event_id in event_ids
        )
        inserted = tuple(
            delivery_id
            for delivery_id in delivery_ids
            if delivery_id not in self._factory.persisted_delivery_ids
        )
        existing = tuple(
            delivery_id
            for delivery_id in delivery_ids
            if delivery_id in self._factory.persisted_delivery_ids
        )
        self._factory.persisted_delivery_ids.update(inserted)
        observation = RouteObservation(
            batch_id=batch_id,
            event_ids=event_ids,
            subscription_ids=subscription_ids,
            inserted_delivery_ids=inserted,
            existing_delivery_ids=existing,
        )
        self._factory.routes.append(observation)
        self._factory.log.append(("route", self._scope, event_ids, batch_id))
        if self._factory.fail_routes:
            self._factory.fail_routes -= 1
            raise RuntimeError("synthetic durable route failure")
        return DurableRouteResult(
            routing=RouteResult(
                input_events=len(event_ids),
                matched_subscriptions=len(subscription_ids) if event_ids else 0,
                enqueued_deliveries=len(delivery_ids),
                coalesced_events=0,
                duplicate_deliveries_suppressed=0,
                expired_events=0,
                backpressured_subscriptions=(),
            ),
            commit=DurableRouteCommit(
                batch_id=batch_id,
                inserted_delivery_ids=inserted,
                existing_delivery_ids=existing,
                store_generation=1,
            ),
        )


class RouterFactory:
    def __init__(self, log: list[LogEntry]) -> None:
        self.log = log
        self.routers: list[RecordingRouter] = []
        self.routes: list[RouteObservation] = []
        self.persisted_delivery_ids: set[str] = set()
        self.fail_routes = 0

    def __call__(self, scope: OutboxScope) -> RecordingRouter:
        self.log.append(("factory", scope))
        router = RecordingRouter(self, scope)
        self.routers.append(router)
        return router


class RecordingWakeSource:
    def __init__(
        self,
        responses: Sequence[OutboxWake | Callable[[int], OutboxWake]],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[int, float]] = []

    def wait_for_outbox(
        self,
        *,
        after_sequence: int,
        after_notification_generation: int = 0,
        deadline_monotonic: float,
    ) -> OutboxWake:
        self.calls.append((after_sequence, deadline_monotonic))
        response = self.responses.pop(0)
        value = response(after_sequence) if callable(response) else response
        if (
            value.after_notification_generation
            != after_notification_generation
        ):
            value = OutboxWake(
                after_sequence=value.after_sequence,
                committed_sequence=value.committed_sequence,
                after_notification_generation=after_notification_generation,
                notification_generation=max(
                    after_notification_generation,
                    value.notification_generation,
                ),
                timed_out=value.timed_out,
                cancelled=value.cancelled,
                server_shutdown=value.server_shutdown,
            )
        return value


def worker(
    repository: RecordingRepository,
    wake_source: RecordingWakeSource | None = None,
    **bounds: int,
) -> tuple[EventDrivenOutboxWorker, RouterFactory, RecordingWakeSource]:
    wake = wake_source or RecordingWakeSource(
        [lambda after: OutboxWake(after, after, timed_out=True)]
    )
    routers = RouterFactory(repository.log)
    return (
        EventDrivenOutboxWorker(repository, routers, wake, **bounds),
        routers,
        wake,
    )


def test_initial_drain_is_bounded_and_orders_restore_route_then_mark() -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    repository = RecordingRepository(
        {scope: (event(2), event(1))},
        {scope: ("subscription:z", "subscription:a")},
        remove_on_mark=True,
    )
    outbox_worker, routers, wake = worker(
        repository,
        maximum_scopes=3,
        maximum_events_per_scope=2,
        maximum_subscriptions_per_scope=2,
    )

    receipt = outbox_worker.drain_once()

    assert receipt.scope_count == 1
    assert receipt.event_count == 2
    assert receipt.delivery_count == 4
    assert receipt.routed_global_sequence == 2
    assert outbox_worker.watermark == 2
    assert wake.calls == []
    assert routers.routes[0].event_ids == (event(1).event_id, event(2).event_id)
    assert routers.routes[0].subscription_ids == (
        "subscription:a",
        "subscription:z",
    )
    operations = [entry[0] for entry in repository.log]
    assert operations == [
        "scopes",
        "events",
        "subscriptions",
        "factory",
        "restore",
        "restore",
        "route",
        "mark",
    ]
    assert repository.log[0] == ("scopes", 3)
    assert repository.log[1] == ("events", scope, 2)
    assert repository.log[2] == ("subscriptions", scope, 2)
    assert repository.pending[scope] == []


def test_failed_mark_replays_stable_route_and_disposition_identities() -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    first = event(1)
    second = event(2)
    repository = RecordingRepository(
        {scope: (second, first)},
        {scope: ("subscription:z", "subscription:a")},
    )
    repository.mark_failures = 1
    outbox_worker, routers, _ = worker(repository)

    with pytest.raises(RuntimeError, match="before outbox disposition"):
        outbox_worker.drain_once()
    assert outbox_worker.watermark == 0

    # A state-owner restart is not allowed to make repository row order part
    # of either idempotency identity.
    repository.pending[scope].reverse()
    repository.subscriptions[scope] = tuple(
        reversed(repository.subscriptions[scope])
    )
    first_receipt = outbox_worker.drain_once()
    second_receipt = outbox_worker.drain_once()

    assert len(routers.routes) == 3
    assert {item.batch_id for item in routers.routes} == {
        routers.routes[0].batch_id
    }
    assert routers.routes[0].inserted_delivery_ids
    assert routers.routes[1].inserted_delivery_ids == ()
    assert routers.routes[1].existing_delivery_ids == (
        routers.routes[0].inserted_delivery_ids
    )
    assert {call[4] for call in repository.mark_calls} == {
        repository.mark_calls[0][4]
    }
    assert first_receipt == second_receipt
    assert first_receipt.dispositions[0] is second_receipt.dispositions[0]


@pytest.mark.parametrize(
    "wake",
    [
        OutboxWake(0, 0, timed_out=True),
        OutboxWake(0, 0),
        OutboxWake(0, 1, cancelled=True),
        OutboxWake(0, 1, server_shutdown=True),
    ],
    ids=("timeout", "no-advance", "cancelled", "server-shutdown"),
)
def test_idle_or_terminal_wake_never_queries_the_repository(wake: OutboxWake) -> None:
    repository = RecordingRepository({})
    wake_source = RecordingWakeSource([wake])
    outbox_worker, routers, _ = worker(repository, wake_source)

    assert outbox_worker.wait_and_drain(deadline_monotonic=12.5) is None

    assert wake_source.calls == [(0, 12.5)]
    assert repository.log == []
    assert routers.routers == []
    assert outbox_worker.watermark == 0


def test_advancing_wake_does_not_skip_rows_left_by_a_bounded_drain() -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    repository = RecordingRepository(
        {scope: tuple(event(sequence) for sequence in range(1, 6))},
        remove_on_mark=True,
    )
    wake_source = RecordingWakeSource(
        [
            lambda after: OutboxWake(after, 5),
            lambda after: OutboxWake(after, 5),
        ]
    )
    outbox_worker, _, _ = worker(
        repository,
        wake_source,
        maximum_scopes=1,
        maximum_events_per_scope=2,
    )

    initial = outbox_worker.drain_once()
    first_wake = outbox_worker.wait_and_drain(deadline_monotonic=20.0)
    second_wake = outbox_worker.wait_and_drain(deadline_monotonic=21.0)

    assert initial.event_count == 2
    assert first_wake is not None and first_wake.event_count == 2
    assert first_wake.routed_global_sequence == 4
    assert second_wake is not None and second_wake.event_count == 1
    assert second_wake.routed_global_sequence == 5
    assert wake_source.calls == [(2, 20.0), (4, 21.0)]
    assert outbox_worker.watermark == 5
    assert repository.pending[scope] == []
    assert [entry[2] for entry in repository.log if entry[0] == "events"] == [
        2,
        2,
        2,
    ]


def test_empty_advancing_wake_adopts_the_owner_watermark() -> None:
    repository = RecordingRepository({})
    wake_source = RecordingWakeSource(
        [
            lambda after: OutboxWake(after, 7),
            lambda after: OutboxWake(after, after, timed_out=True),
        ]
    )
    outbox_worker, _, _ = worker(repository, wake_source)

    receipt = outbox_worker.wait_and_drain(deadline_monotonic=30.0)
    assert receipt is not None
    assert receipt.scope_count == 0
    assert receipt.routed_global_sequence == 0
    assert outbox_worker.watermark == 7

    assert outbox_worker.wait_and_drain(deadline_monotonic=31.0) is None
    assert wake_source.calls == [(0, 30.0), (7, 31.0)]
    assert repository.log == [("scopes", MAX_OUTBOX_SCOPES_PER_DRAIN)]


def test_mismatched_wake_registration_fails_closed_without_a_query() -> None:
    repository = RecordingRepository({})
    wake_source = RecordingWakeSource([OutboxWake(1, 2)])
    outbox_worker, _, _ = worker(repository, wake_source)

    with pytest.raises(FederationContractError, match="registration"):
        outbox_worker.wait_and_drain(deadline_monotonic=40.0)

    assert repository.log == []
    assert outbox_worker.watermark == 0


@pytest.mark.parametrize(
    ("duplicate", "over_bound", "error"),
    [
        (True, False, FederationContractError),
        (False, True, FederationBoundsError),
    ],
    ids=("duplicate", "over-bound"),
)
def test_invalid_pending_scopes_fail_before_any_routing(
    duplicate: bool,
    over_bound: bool,
    error: type[Exception],
) -> None:
    first_scope = OutboxScope("tenant:one", "federation:one")
    second_scope = OutboxScope("tenant:two", "federation:two")
    repository = RecordingRepository(
        {
            first_scope: (event(1),),
            second_scope: (
                event(
                    2,
                    tenant_id="tenant:two",
                    federation_id="federation:two",
                ),
            ),
        }
    )
    if duplicate:
        repository.scope_response = (first_scope, first_scope)
    elif over_bound:
        repository.scope_response = (first_scope, second_scope)
    outbox_worker, routers, _ = worker(repository, maximum_scopes=1)

    with pytest.raises(error):
        outbox_worker.drain_once()

    assert repository.log == [("scopes", 1)]
    assert routers.routers == []


@pytest.mark.parametrize(
    ("values", "maximum", "error"),
    [
        (lambda: (event(1), event(1)), 2, FederationContractError),
        (lambda: (event(1), event(2)), 1, FederationBoundsError),
        (
            lambda: (
                event(
                    1,
                    tenant_id="tenant:other",
                    federation_id="federation:one",
                ),
            ),
            1,
            FederationContractError,
        ),
        (
            lambda: (event(1, identity="first"), event(1, identity="second")),
            2,
            FederationContractError,
        ),
    ],
    ids=("duplicate-id", "over-bound", "cross-scope", "duplicate-sequence"),
)
def test_invalid_pending_events_fail_before_restore_or_route(
    values: Callable[[], tuple[DomainEvent, ...]],
    maximum: int,
    error: type[Exception],
) -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    repository = RecordingRepository(
        {scope: values()},
        {scope: ("subscription:one",)},
    )
    repository.event_responses[scope] = values()
    outbox_worker, routers, _ = worker(
        repository,
        maximum_events_per_scope=maximum,
    )

    with pytest.raises(error):
        outbox_worker.drain_once()

    assert [entry[0] for entry in repository.log] == ["scopes", "events"]
    assert routers.routers == []
    assert repository.mark_calls == []


@pytest.mark.parametrize(
    ("subscriptions", "maximum", "error"),
    [
        (
            ("subscription:one", "subscription:one"),
            2,
            FederationContractError,
        ),
        (
            ("subscription:one", "subscription:two"),
            1,
            FederationBoundsError,
        ),
    ],
    ids=("duplicate", "over-bound"),
)
def test_invalid_subscriptions_fail_before_factory_restore_or_route(
    subscriptions: tuple[str, ...],
    maximum: int,
    error: type[Exception],
) -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    repository = RecordingRepository(
        {scope: (event(1),)},
        {scope: subscriptions},
    )
    repository.subscription_responses[scope] = subscriptions
    outbox_worker, routers, _ = worker(
        repository,
        maximum_subscriptions_per_scope=maximum,
    )

    with pytest.raises(error):
        outbox_worker.drain_once()

    assert [entry[0] for entry in repository.log] == [
        "scopes",
        "events",
        "subscriptions",
    ]
    assert routers.routers == []
    assert repository.mark_calls == []


@pytest.mark.parametrize(
    "bounds",
    [
        {"maximum_scopes": 0},
        {"maximum_scopes": MAX_OUTBOX_SCOPES_PER_DRAIN + 1},
        {"maximum_events_per_scope": MAX_OUTBOX_EVENTS_PER_SCOPE + 1},
        {
            "maximum_subscriptions_per_scope": (
                MAX_SUBSCRIPTIONS_PER_SCOPE + 1
            )
        },
    ],
)
def test_worker_rejects_invalid_configured_bounds(bounds: dict[str, int]) -> None:
    repository = RecordingRepository({})
    wake_source = RecordingWakeSource([])
    routers = RouterFactory(repository.log)

    with pytest.raises(FederationBoundsError):
        EventDrivenOutboxWorker(repository, routers, wake_source, **bounds)


def test_route_failure_never_marks_or_advances_the_watermark() -> None:
    scope = OutboxScope("tenant:one", "federation:one")
    repository = RecordingRepository({scope: (event(1),)})
    outbox_worker, routers, _ = worker(repository)
    routers.fail_routes = 1

    with pytest.raises(RuntimeError, match="durable route failure"):
        outbox_worker.drain_once()

    assert repository.mark_calls == []
    assert outbox_worker.watermark == 0
