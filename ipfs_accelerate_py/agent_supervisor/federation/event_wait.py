"""State-owner event wait and explicit adaptive-long-poll compatibility.

``StateOwnerEventWait`` uses one process-owned condition for all consumers.
It performs one bounded query before blocking and then queries again only after
the state owner publishes a newer committed event generation.  A generation
snapshot covers the query/register boundary without holding the condition
across a state-owner read, preventing both lost wakeups and lock inversion with
the post-commit notifier.

The adaptive client is deliberately labelled unqualified.  It exists only for
remote Quack builds that cannot expose the typed long wait; callers may not use
it to pass the event-driven promotion gate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from .events import DomainEvent, EventBatch, EventWaitRequest


class EventWaitError(RuntimeError):
    """Base event wait failure."""


class StaleSubscriptionError(EventWaitError):
    """The wait request does not name the current subscription revision."""


class EventSource(Protocol):
    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]: ...

    def store_generation(self) -> int: ...


def _deadline_monotonic(
    deadline: str,
    *,
    now_wall: Callable[[], float],
    monotonic: Callable[[], float],
) -> float:
    parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise EventWaitError("deadline must include a timezone")
    remaining = parsed.timestamp() - now_wall()
    return monotonic() + max(0.0, remaining)


class StateOwnerEventWait:
    """One server-owned condition shared by all federation consumers."""

    def __init__(
        self,
        source: EventSource,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._condition = threading.Condition(threading.RLock())
        self._notification_generation = 0
        self._event_watermark = 0
        self._cancelled_consumers: set[str] = set()
        self._shutdown = False
        self._query_count = 0
        self._wakeup_count = 0

    @property
    def query_count(self) -> int:
        with self._condition:
            return self._query_count

    @property
    def wakeup_count(self) -> int:
        with self._condition:
            return self._wakeup_count

    @property
    def notification_generation(self) -> int:
        with self._condition:
            return self._notification_generation

    def notify_committed(self, global_sequence: int) -> None:
        """Publish an already committed event watermark to blocked waiters."""

        sequence = int(global_sequence)
        if sequence < 1:
            raise EventWaitError("committed event sequence must be positive")
        with self._condition:
            if sequence < self._event_watermark:
                # Out-of-order duplicate notification is harmless.  It must not
                # manufacture another wake generation.
                return
            if sequence == self._event_watermark:
                return
            self._event_watermark = sequence
            self._notification_generation += 1
            self._condition.notify_all()

    def cancel(self, consumer_id: str) -> None:
        with self._condition:
            self._cancelled_consumers.add(str(consumer_id))
            self._condition.notify_all()

    def clear_cancel(self, consumer_id: str) -> None:
        with self._condition:
            self._cancelled_consumers.discard(str(consumer_id))

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def wait_for_events(self, request: EventWaitRequest) -> EventBatch:
        if not isinstance(request, EventWaitRequest):
            raise EventWaitError("request must be EventWaitRequest")
        deadline = _deadline_monotonic(
            request.deadline,
            now_wall=self._wall_time,
            monotonic=self._monotonic,
        )
        while True:
            with self._condition:
                if self._shutdown:
                    empty_reason = "shutdown"
                elif request.consumer_id in self._cancelled_consumers:
                    empty_reason = "cancelled"
                else:
                    empty_reason = ""
                observed_generation = self._notification_generation
                if not empty_reason:
                    self._query_count += 1

            if empty_reason:
                return self._empty_batch(
                    request,
                    cancelled=empty_reason == "cancelled",
                    server_shutdown=empty_reason == "shutdown",
                )

            # Never hold the notification condition across an authoritative
            # state read.  A commit observer may need this same condition while
            # the state-owner transaction/client lock is still held.  The
            # generation comparison below closes the resulting race window.
            events = self._source.events_for_subscription(
                consumer_id=request.consumer_id,
                subscription_id=request.subscription_id,
                subscription_revision=request.subscription_revision,
                after_cursor=request.after_cursor,
                maximum_events=request.maximum_events,
            )
            if events:
                return EventBatch(
                    consumer_id=request.consumer_id,
                    subscription_id=request.subscription_id,
                    subscription_revision=request.subscription_revision,
                    after_cursor=request.after_cursor,
                    next_cursor=events[-1].global_sequence,
                    store_generation=self._source.store_generation(),
                    events=events,
                    timed_out=False,
                    cancelled=False,
                    server_shutdown=False,
                )

            with self._condition:
                if self._shutdown:
                    empty_reason = "shutdown"
                if request.consumer_id in self._cancelled_consumers:
                    empty_reason = "cancelled"
                if empty_reason:
                    pass
                elif self._notification_generation != observed_generation:
                    self._wakeup_count += 1
                    continue

                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    empty_reason = "timed_out"
                else:
                    changed = self._condition.wait_for(
                        lambda observed_generation=observed_generation: (
                            self._notification_generation != observed_generation
                            or request.consumer_id in self._cancelled_consumers
                            or self._shutdown
                        ),
                        timeout=remaining,
                    )
                    if not changed:
                        empty_reason = "timed_out"
                    else:
                        self._wakeup_count += 1
                        continue

            return self._empty_batch(
                request,
                timed_out=empty_reason == "timed_out",
                cancelled=empty_reason == "cancelled",
                server_shutdown=empty_reason == "shutdown",
            )

    def _empty_batch(
        self,
        request: EventWaitRequest,
        *,
        timed_out: bool = False,
        cancelled: bool = False,
        server_shutdown: bool = False,
    ) -> EventBatch:
        return EventBatch(
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
            subscription_revision=request.subscription_revision,
            after_cursor=request.after_cursor,
            next_cursor=request.after_cursor,
            store_generation=self._source.store_generation(),
            events=(),
            timed_out=timed_out,
            cancelled=cancelled,
            server_shutdown=server_shutdown,
        )

    def capability(self) -> dict[str, object]:
        return {
            "interface": "StateOwnerEventWait@1",
            "server_owned": True,
            "blocking_condition": True,
            "lost_wakeup_check_register_guard": True,
            "idle_repeated_database_scans": False,
            "remote_quack_transport_qualified": False,
            "qualification": "owner_local_hermetic_only",
        }


class AdaptiveLongPollEventWaitClient:
    """Bounded fallback for a remote transport without typed server wait."""

    def __init__(
        self,
        fetch: Callable[[EventWaitRequest], EventBatch],
        *,
        minimum_interval_seconds: float = 0.25,
        maximum_interval_seconds: float = 5.0,
        backoff_multiplier: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        if minimum_interval_seconds <= 0 or maximum_interval_seconds < minimum_interval_seconds:
            raise EventWaitError("adaptive polling bounds are invalid")
        if backoff_multiplier <= 1:
            raise EventWaitError("adaptive polling multiplier must exceed one")
        self._fetch = fetch
        self._minimum = float(minimum_interval_seconds)
        self._maximum = float(maximum_interval_seconds)
        self._multiplier = float(backoff_multiplier)
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_time = wall_time

    def wait_for_events(self, request: EventWaitRequest) -> EventBatch:
        deadline = _deadline_monotonic(
            request.deadline,
            now_wall=self._wall_time,
            monotonic=self._monotonic,
        )
        interval = self._minimum
        while True:
            batch = self._fetch(request)
            if batch.events or batch.cancelled or batch.server_shutdown:
                return batch
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return EventBatch(
                    consumer_id=request.consumer_id,
                    subscription_id=request.subscription_id,
                    subscription_revision=request.subscription_revision,
                    after_cursor=request.after_cursor,
                    next_cursor=request.after_cursor,
                    store_generation=batch.store_generation,
                    events=(),
                    timed_out=True,
                    cancelled=False,
                    server_shutdown=False,
                )
            self._sleep(min(interval, remaining))
            interval = min(self._maximum, interval * self._multiplier)

    @staticmethod
    def capability() -> dict[str, object]:
        return {
            "interface": "AdaptiveLongPollEventWait@1",
            "bounded": True,
            "backs_off_when_idle": True,
            "event_driven_qualification": False,
            "reason": "remote Quack push-style wait unavailable",
        }


__all__ = [
    "AdaptiveLongPollEventWaitClient",
    "EventSource",
    "EventWaitError",
    "StaleSubscriptionError",
    "StateOwnerEventWait",
]
