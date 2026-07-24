"""In-process single-flight coordination for the analysis cache.

``AnalysisCache`` owns persistence, integrity checking, expiry, and exact-key
invalidation.  This module adds only execution coordination: concurrent work
for one :class:`~analysis_cache.AnalysisCacheKey` is represented by one shared
future while unrelated keys remain fully parallel.

The cache is always consulted with ``require_completion_evidence=True``.
Consequently a stale, partial, failed, timed-out, inconclusive, or corrupt
entry can be observed in the returned diagnostics but can never bypass the
producer as an authoritative cache hit.  Followers receive a shared execution
result only after the leader has published it, and the result's completion
authority remains derived from a fresh exact-key cache lookup.

Coordination is deliberately in-process.  The future used as the rendezvous
point is safe for both threads and asyncio event loops, so synchronous and
asynchronous callers in the same process also collapse onto one producer.
Cross-process coordination belongs in a durable lease implementation.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, TypeVar, Union

from .analysis_cache import (
    AnalysisCache,
    AnalysisCacheEntry,
    AnalysisCacheKey,
    AnalysisCacheLookupResult,
    AnalysisCacheLookupStatus,
    AnalysisCacheStoreResult,
    AnalysisOutcome,
    AnalysisReceipt,
)


T = TypeVar("T")
ProducerValue = Union[
    Mapping[str, Any],
    AnalysisReceipt,
    AnalysisCacheEntry,
    AnalysisCacheLookupResult,
    AnalysisCacheStoreResult,
]
SyncProducer = Callable[[], ProducerValue]
AsyncProducer = Callable[[], Union[ProducerValue, Awaitable[ProducerValue]]]


class CacheCoordinationError(RuntimeError):
    """Base class for analysis cache coordination failures."""


class CacheCoordinationTimeout(CacheCoordinationError, TimeoutError):
    """A follower did not observe the leader's result within its wait bound."""


class CacheProducerResultError(CacheCoordinationError, TypeError):
    """A producer returned a value that cannot be persisted or validated."""


class CacheCoordinationStatus(str, Enum):
    """How a coordinated analysis result was obtained."""

    CACHE_HIT = "cache_hit"
    PRODUCED = "produced"
    SHARED = "shared"

    # Readable compatibility spellings.
    HIT = "cache_hit"
    LEADER = "produced"
    MISS_PRODUCED = "produced"
    FOLLOWER = "shared"
    COALESCED = "shared"


CoordinatorStatus = CacheCoordinationStatus
CacheCoordinatorStatus = CacheCoordinationStatus


@dataclass(frozen=True)
class CachePublication:
    """Per-production persistence decision used inside one shared flight.

    This lets a producer coalesce expensive work while applying a result-aware
    cache policy: conclusive receipts can be durable, negative receipts can
    have a bounded TTL, and explicitly non-cacheable receipts can still be
    delivered to all in-process followers.
    """

    value: Any
    store: bool = True
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store, bool):
            raise ValueError("store must be a boolean")
        if self.ttl_seconds is not None and (
            isinstance(self.ttl_seconds, bool)
            or not isinstance(self.ttl_seconds, int)
            or self.ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")


@dataclass(frozen=True)
class CacheCoordinationResult:
    """Typed outcome of one coordinated cache operation.

    ``producer_value`` is retained for local callers that need a typed value
    even when it was not completion-eligible.  It is intentionally omitted
    from :meth:`to_dict`: analysis bodies and other large producer values must
    not accidentally cross an audit or scheduler boundary.
    """

    status: CacheCoordinationStatus
    key: AnalysisCacheKey
    lookup: AnalysisCacheLookupResult
    store_result: AnalysisCacheStoreResult | None = None
    producer_value: Any = None
    leader: bool = False
    waited: bool = False

    @property
    def cache_hit(self) -> bool:
        return self.status is CacheCoordinationStatus.CACHE_HIT

    @property
    def produced(self) -> bool:
        return self.status is CacheCoordinationStatus.PRODUCED

    @property
    def shared(self) -> bool:
        return self.status is CacheCoordinationStatus.SHARED

    @property
    def coalesced(self) -> bool:
        return self.shared

    @property
    def follower(self) -> bool:
        return self.shared

    @property
    def entry(self) -> AnalysisCacheEntry | None:
        if self.lookup.entry is not None:
            return self.lookup.entry
        if self.store_result is not None:
            return self.store_result.entry
        return None

    @property
    def receipt(self) -> Mapping[str, Any] | None:
        """Return the compact receipt without implying completion authority."""

        entry = self.entry
        return entry.receipt if entry is not None else None

    @property
    def outcome(self) -> AnalysisOutcome | None:
        entry = self.entry
        return entry.status if entry is not None else self.lookup.outcome

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.lookup.reason_codes:
            return self.lookup.reason_codes
        if self.store_result is not None:
            return self.store_result.reason_codes
        return ()

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""

    @property
    def is_completion_evidence(self) -> bool:
        """Whether a fresh exact-key lookup grants completion authority."""

        return bool(
            self.lookup.status is AnalysisCacheLookupStatus.HIT
            and self.lookup.key == self.key
            and self.lookup.is_completion_evidence
        )

    @property
    def completion_evidence(self) -> bool:
        return self.is_completion_evidence

    @property
    def authoritative(self) -> bool:
        return self.is_completion_evidence

    @property
    def value(self) -> Any:
        """Return the producer value, falling back to the compact receipt."""

        return (
            self.producer_value
            if self.producer_value is not None
            else self.receipt
        )

    def require_completion_evidence(self) -> Mapping[str, Any]:
        if not self.is_completion_evidence or self.receipt is None:
            raise CacheCoordinationError(
                "coordinated analysis result is not completion evidence"
            )
        return self.receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "key_id": self.key.key_id,
            "cache_lookup_status": self.lookup.status.value,
            "cache_hit": self.cache_hit,
            "produced": self.produced,
            "shared": self.shared,
            "leader": self.leader,
            "waited": self.waited,
            "is_completion_evidence": self.is_completion_evidence,
            "reason_codes": list(self.reason_codes),
            "outcome": self.outcome.value if self.outcome is not None else "",
            "stored": bool(self.store_result and self.store_result.stored),
        }


CoordinatorResult = CacheCoordinationResult
CacheCoordinatorResult = CacheCoordinationResult


@dataclass(frozen=True)
class CacheCoordinatorMetrics:
    """Atomic snapshot of coordinator activity."""

    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    leaders: int = 0
    followers: int = 0
    produced: int = 0
    shared: int = 0
    completion_results: int = 0
    non_authoritative_results: int = 0
    producer_failures: int = 0
    wait_timeouts: int = 0
    active_flights: int = 0

    @property
    def collapsed_count(self) -> int:
        return self.followers

    @property
    def single_flight_savings(self) -> int:
        return self.followers

    @property
    def hit_ratio(self) -> float:
        return self.cache_hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "leaders": self.leaders,
            "followers": self.followers,
            "produced": self.produced,
            "shared": self.shared,
            "completion_results": self.completion_results,
            "non_authoritative_results": self.non_authoritative_results,
            "producer_failures": self.producer_failures,
            "wait_timeouts": self.wait_timeouts,
            "active_flights": self.active_flights,
            "collapsed_count": self.collapsed_count,
            "single_flight_savings": self.single_flight_savings,
            "hit_ratio": self.hit_ratio,
        }


CoordinatorMetrics = CacheCoordinatorMetrics


@dataclass
class _Flight:
    future: Future[CacheCoordinationResult]


class AnalysisCacheCoordinator:
    """Collapse identical in-process analysis misses to one producer."""

    def __init__(
        self,
        cache: AnalysisCache,
        *,
        wait_timeout_seconds: float | None = 30.0,
    ) -> None:
        if not isinstance(cache, AnalysisCache):
            raise ValueError("cache must be an AnalysisCache")
        if wait_timeout_seconds is not None and (
            isinstance(wait_timeout_seconds, bool)
            or not isinstance(wait_timeout_seconds, (int, float))
            or wait_timeout_seconds <= 0
        ):
            raise ValueError(
                "wait_timeout_seconds must be positive or None"
            )
        self.cache = cache
        self.wait_timeout_seconds = (
            None
            if wait_timeout_seconds is None
            else float(wait_timeout_seconds)
        )
        self._lock = threading.RLock()
        self._flights: dict[str, _Flight] = {}
        self._metric_values: dict[str, int] = {
            name: 0
            for name in CacheCoordinatorMetrics.__dataclass_fields__
            if name != "active_flights"
        }

    def _coerce_key(
        self, key: AnalysisCacheKey | Mapping[str, Any]
    ) -> AnalysisCacheKey:
        return (
            key
            if isinstance(key, AnalysisCacheKey)
            else AnalysisCacheKey.from_dict(key)
        )

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._metric_values[name] += amount

    def metrics(self) -> CacheCoordinatorMetrics:
        with self._lock:
            return CacheCoordinatorMetrics(
                **self._metric_values,
                active_flights=len(self._flights),
            )

    stats = metrics
    metrics_snapshot = metrics

    def reset_metrics(self) -> CacheCoordinatorMetrics:
        """Reset counters and return the snapshot that was replaced."""

        with self._lock:
            previous = CacheCoordinatorMetrics(
                **self._metric_values,
                active_flights=len(self._flights),
            )
            for name in self._metric_values:
                self._metric_values[name] = 0
            return previous

    def _completion_lookup(
        self, key: AnalysisCacheKey
    ) -> AnalysisCacheLookupResult:
        return self.cache.lookup(key, require_completion_evidence=True)

    @staticmethod
    def _is_exact_completion_hit(
        lookup: AnalysisCacheLookupResult, key: AnalysisCacheKey
    ) -> bool:
        return bool(
            lookup.status is AnalysisCacheLookupStatus.HIT
            and lookup.key == key
            and lookup.is_completion_evidence
        )

    def _cache_hit_result(
        self,
        key: AnalysisCacheKey,
        lookup: AnalysisCacheLookupResult,
    ) -> CacheCoordinationResult:
        return CacheCoordinationResult(
            status=CacheCoordinationStatus.CACHE_HIT,
            key=key,
            lookup=lookup,
        )

    def _begin(
        self, key: AnalysisCacheKey
    ) -> tuple[
        CacheCoordinationResult | None,
        _Flight | None,
        bool,
    ]:
        """Return ``(cached, flight, leader)`` without running user code."""

        self._increment("requests")
        lookup = self._completion_lookup(key)
        if self._is_exact_completion_hit(lookup, key):
            self._increment("cache_hits")
            self._increment("completion_results")
            return self._cache_hit_result(key, lookup), None, False

        with self._lock:
            # Close the lookup-to-registration race.  A preceding leader may
            # have populated the exact cache immediately before this lock.
            lookup = self._completion_lookup(key)
            if self._is_exact_completion_hit(lookup, key):
                self._metric_values["cache_hits"] += 1
                self._metric_values["completion_results"] += 1
                return self._cache_hit_result(key, lookup), None, False
            self._metric_values["cache_misses"] += 1
            existing = self._flights.get(key.key_id)
            if existing is not None:
                self._metric_values["followers"] += 1
                return None, existing, False
            flight = _Flight(Future())
            self._flights[key.key_id] = flight
            self._metric_values["leaders"] += 1
            return None, flight, True

    def _finish_flight(
        self, key: AnalysisCacheKey, flight: _Flight
    ) -> None:
        with self._lock:
            if self._flights.get(key.key_id) is flight:
                del self._flights[key.key_id]

    @staticmethod
    def _validate_result_key(
        expected: AnalysisCacheKey, actual: AnalysisCacheKey
    ) -> None:
        if actual != expected:
            raise CacheProducerResultError(
                "producer result is bound to a different analysis cache key"
            )

    def _publish_producer_value(
        self,
        key: AnalysisCacheKey,
        value: ProducerValue | CachePublication,
        *,
        ttl_seconds: int | None,
    ) -> CacheCoordinationResult:
        store_result: AnalysisCacheStoreResult | None = None
        should_store = True
        if isinstance(value, CachePublication):
            should_store = value.store
            ttl_seconds = value.ttl_seconds
            value = value.value

        if not should_store:
            if not isinstance(
                value,
                (
                    Mapping,
                    AnalysisReceipt,
                    AnalysisCacheEntry,
                    AnalysisCacheLookupResult,
                    AnalysisCacheStoreResult,
                ),
            ):
                raise CacheProducerResultError(
                    "non-stored publication has an unsupported value"
                )
            lookup = self._completion_lookup(key)
            result = CacheCoordinationResult(
                status=CacheCoordinationStatus.PRODUCED,
                key=key,
                lookup=lookup,
                producer_value=value,
                leader=True,
            )
            self._increment("produced")
            self._increment("non_authoritative_results")
            return result

        if isinstance(value, AnalysisCacheLookupResult):
            self._validate_result_key(key, value.key)
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisCacheStoreResult):
            self._validate_result_key(key, value.key)
            store_result = value
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisCacheEntry):
            self._validate_result_key(key, value.key)
            store_result = self.cache.put(
                key,
                value.receipt,
                status=value.status,
                ttl_seconds=ttl_seconds,
            )
            lookup = self._completion_lookup(key)
        elif isinstance(value, AnalysisReceipt):
            store_result = self.cache.put(
                key, value, ttl_seconds=ttl_seconds
            )
            lookup = self._completion_lookup(key)
        elif isinstance(value, Mapping):
            store_result = self.cache.put(
                key, value, ttl_seconds=ttl_seconds
            )
            lookup = self._completion_lookup(key)
        else:
            raise CacheProducerResultError(
                "producer must return a compact receipt, cache entry, "
                "lookup result, or store result"
            )

        result = CacheCoordinationResult(
            status=CacheCoordinationStatus.PRODUCED,
            key=key,
            lookup=lookup,
            store_result=store_result,
            producer_value=value,
            leader=True,
        )
        self._increment("produced")
        self._increment(
            "completion_results"
            if result.is_completion_evidence
            else "non_authoritative_results"
        )
        return result

    def _shared_result(
        self, result: CacheCoordinationResult
    ) -> CacheCoordinationResult:
        # Authority remains whatever the leader established with an exact
        # post-store cache lookup.  The status only describes coordination.
        shared = replace(
            result,
            status=CacheCoordinationStatus.SHARED,
            leader=False,
            waited=True,
        )
        self._increment("shared")
        self._increment(
            "completion_results"
            if shared.is_completion_evidence
            else "non_authoritative_results"
        )
        return shared

    def _timeout(
        self, key: AnalysisCacheKey, timeout: float | None
    ) -> CacheCoordinationTimeout:
        self._increment("wait_timeouts")
        detail = "without a deadline" if timeout is None else f"after {timeout:g}s"
        return CacheCoordinationTimeout(
            f"timed out waiting for analysis single flight {key.key_id} {detail}"
        )

    def get_or_compute(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        producer: SyncProducer,
        *,
        ttl_seconds: int | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> CacheCoordinationResult:
        """Return an exact completion hit or run ``producer`` once.

        ``producer`` is called with no arguments.  Use
        :meth:`async_get_or_compute` for coroutine producers.
        """

        if not callable(producer):
            raise ValueError("producer must be callable")
        cache_key = self._coerce_key(key)
        cached, flight, leader = self._begin(cache_key)
        if cached is not None:
            return cached
        assert flight is not None

        if leader:
            try:
                value = producer()
                if inspect.isawaitable(value):
                    close = getattr(value, "close", None)
                    if callable(close):
                        close()
                    raise CacheProducerResultError(
                        "synchronous producer returned an awaitable; use "
                        "async_get_or_compute"
                    )
                result = self._publish_producer_value(
                    cache_key, value, ttl_seconds=ttl_seconds
                )
                flight.future.set_result(result)
                return result
            except BaseException as exc:
                self._increment("producer_failures")
                flight.future.set_exception(exc)
                raise
            finally:
                self._finish_flight(cache_key, flight)

        timeout = (
            self.wait_timeout_seconds
            if wait_timeout_seconds is None
            else self._validate_timeout(wait_timeout_seconds)
        )
        try:
            result = flight.future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise self._timeout(cache_key, timeout) from exc
        return self._shared_result(result)

    def _validate_timeout(self, value: float | None) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError("wait_timeout_seconds must be positive or None")
        return float(value)

    async def async_get_or_compute(
        self,
        key: AnalysisCacheKey | Mapping[str, Any],
        producer: AsyncProducer,
        *,
        ttl_seconds: int | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> CacheCoordinationResult:
        """Async counterpart that shares flights with synchronous callers."""

        if not callable(producer):
            raise ValueError("producer must be callable")
        cache_key = self._coerce_key(key)
        cached, flight, leader = self._begin(cache_key)
        if cached is not None:
            return cached
        assert flight is not None

        if leader:
            try:
                value = producer()
                if inspect.isawaitable(value):
                    value = await value
                result = self._publish_producer_value(
                    cache_key, value, ttl_seconds=ttl_seconds
                )
                flight.future.set_result(result)
                return result
            except BaseException as exc:
                self._increment("producer_failures")
                flight.future.set_exception(exc)
                raise
            finally:
                self._finish_flight(cache_key, flight)

        timeout = (
            self.wait_timeout_seconds
            if wait_timeout_seconds is None
            else self._validate_timeout(wait_timeout_seconds)
        )
        wrapped = asyncio.wrap_future(flight.future)
        try:
            if timeout is None:
                result = await asyncio.shield(wrapped)
            else:
                result = await asyncio.wait_for(
                    asyncio.shield(wrapped), timeout=timeout
                )
        except asyncio.TimeoutError as exc:
            raise self._timeout(cache_key, timeout) from exc
        return self._shared_result(result)

    # Conventional synchronous spellings.
    coordinate = get_or_compute
    run = get_or_compute
    execute = get_or_compute
    single_flight = get_or_compute
    execute_single_flight = get_or_compute
    run_single_flight = get_or_compute

    # Conventional asynchronous spellings.
    aget_or_compute = async_get_or_compute
    acoordinate = async_get_or_compute
    arun = async_get_or_compute
    aexecute = async_get_or_compute
    async_single_flight = async_get_or_compute
    execute_async_single_flight = async_get_or_compute


CacheCoordinator = AnalysisCacheCoordinator
SingleFlightCacheCoordinator = AnalysisCacheCoordinator


__all__ = [
    "AnalysisCacheCoordinator",
    "CacheCoordinationError",
    "CacheCoordinationResult",
    "CacheCoordinationStatus",
    "CacheCoordinationTimeout",
    "CacheCoordinator",
    "CacheCoordinatorMetrics",
    "CacheCoordinatorResult",
    "CacheCoordinatorStatus",
    "CacheProducerResultError",
    "CoordinatorMetrics",
    "CoordinatorResult",
    "CoordinatorStatus",
    "SingleFlightCacheCoordinator",
]
