"""Source-scoped snapshot caching with independent capability and health TTLs.

This cache stores catalog metadata only.  It must never be used for prompts,
media, invocation output, or provider responses.  Refresh work is single-flight
per source, while invalidation remains projection-specific.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .events import (
    CacheView,
    CatalogEventBus,
    CatalogInvalidationEvent,
    CatalogMetrics,
)
from .identity import content_cid
from .schema import CatalogSnapshot, OperationalState


DEFAULT_CAPABILITIES_TTL = 300.0
DEFAULT_HEALTH_TTL = 15.0
MAX_CACHE_TTL = 86_400.0
MAX_CACHE_SOURCES = 1_024
MAX_HEALTH_SAMPLES = 10_000


def _ttl(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) <= MAX_CACHE_TTL
    ):
        raise ValueError("%s must be greater than zero and at most %d" % (field, MAX_CACHE_TTL))
    return float(value)


def _source(value: Any) -> str:
    # Event construction owns canonical source validation and keeps the two
    # modules on exactly the same source-name contract.
    return CatalogInvalidationEvent(
        "registration", source=value
    ).source  # type: ignore[return-value]


def _view(value: Any) -> CacheView:
    if isinstance(value, CacheView):
        return value
    try:
        return CacheView(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown catalog cache view") from exc


def _wall_timestamp(clock: Callable[[], Any]) -> str:
    value = clock()
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        selected = datetime.fromtimestamp(float(value), timezone.utc)
    elif isinstance(value, str):
        # Receipt/schema timestamp validators are intentionally not imported.
        # Normalizing here catches malformed injected clocks immediately.
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            selected = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("wall clock returned an invalid timestamp") from exc
    else:
        raise ValueError("wall clock must return a timestamp")
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("wall clock timestamp must be timezone-aware")
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class CachePolicy:
    """Independent freshness policy for descriptive and observed data."""

    capabilities_ttl: float = DEFAULT_CAPABILITIES_TTL
    health_ttl: float = DEFAULT_HEALTH_TTL
    allow_stale_on_error: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities_ttl",
            _ttl(self.capabilities_ttl, "capabilities_ttl"),
        )
        object.__setattr__(self, "health_ttl", _ttl(self.health_ttl, "health_ttl"))
        if not isinstance(self.allow_stale_on_error, bool):
            raise ValueError("allow_stale_on_error must be boolean")

    def ttl_for(self, view: Any) -> float:
        selected = _view(view)
        return (
            self.capabilities_ttl
            if selected is CacheView.CAPABILITIES
            else self.health_ttl
        )

    @property
    def static_ttl(self) -> float:
        return self.capabilities_ttl

    @property
    def dynamic_ttl(self) -> float:
        return self.health_ttl


CatalogCachePolicy = CachePolicy


@dataclass(frozen=True)
class HealthSample:
    """One endpoint-free observation suitable for the dynamic health cache."""

    source: str
    record_id: str
    state: OperationalState
    observed_at: str
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source(self.source))
        if (
            not isinstance(self.record_id, str)
            or not self.record_id
            or len(self.record_id.encode("utf-8")) > 256
        ):
            raise ValueError("health record_id is invalid")
        if not isinstance(self.state, OperationalState):
            object.__setattr__(
                self, "state", OperationalState.from_dict(self.state)
            )
        # Reuse the timestamp normalization path through a constant clock.
        object.__setattr__(
            self, "observed_at", _wall_timestamp(lambda: self.observed_at)
        )
        if self.revision is not None and (
            not isinstance(self.revision, str)
            or not self.revision
            or len(self.revision.encode("utf-8")) > 512
        ):
            raise ValueError("health revision is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "record_id": self.record_id,
            "state": self.state.to_dict(),
            "observed_at": self.observed_at,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HealthSample":
        fields = {"source", "record_id", "state", "observed_at", "revision"}
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("HealthSample has missing or unknown fields")
        return cls(**dict(data))


@dataclass(frozen=True)
class HealthSnapshot:
    """Deterministic collection of health samples."""

    samples: Tuple[HealthSample, ...] = ()
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.samples, (str, bytes, Mapping)):
            raise ValueError("health samples must be a bounded array")
        parsed = tuple(
            item if isinstance(item, HealthSample) else HealthSample.from_dict(item)
            for item in self.samples
        )
        if len(parsed) > MAX_HEALTH_SAMPLES:
            raise ValueError("health sample count exceeds the cache bound")
        keys = [(item.source, item.record_id) for item in parsed]
        if len(keys) != len(set(keys)):
            raise ValueError("health snapshot contains duplicate records")
        parsed = tuple(
            sorted(parsed, key=lambda item: (item.source, item.record_id))
        )
        object.__setattr__(self, "samples", parsed)
        expected = content_cid({"health_samples": [item.to_dict() for item in parsed]})
        if self.revision is not None and self.revision != expected:
            raise ValueError("health revision does not match canonical content")
        object.__setattr__(self, "revision", expected)

    @property
    def cid(self) -> str:
        return self.revision  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": [item.to_dict() for item in self.samples],
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HealthSnapshot":
        if not isinstance(data, Mapping) or set(data) != {"samples", "revision"}:
            raise ValueError("HealthSnapshot has missing or unknown fields")
        return cls(samples=tuple(data["samples"]), revision=data["revision"])


def _value_cid(value: Any) -> str:
    if isinstance(value, CatalogSnapshot):
        return value.revision  # type: ignore[return-value]
    for field in ("cid", "revision"):
        selected = getattr(value, field, None)
        if isinstance(selected, str) and selected:
            return selected
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    return content_cid(value)


def _record_count(value: Any) -> int:
    if isinstance(value, CatalogSnapshot):
        return sum(
            len(getattr(value, field))
            for field in ("providers", "models", "deployments", "bindings")
        )
    if isinstance(value, HealthSnapshot):
        return len(value.samples)
    snapshot = getattr(value, "snapshot", None)
    if isinstance(snapshot, CatalogSnapshot):
        return _record_count(snapshot)
    return 1


@dataclass(frozen=True)
class CacheEntry:
    """One immutable cache record."""

    source: str
    view: CacheView
    value: Any
    cid: str
    stored_at: float
    expires_at: float
    stale: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _source(self.source))
        object.__setattr__(self, "view", _view(self.view))
        if (
            not isinstance(self.cid, str)
            or not self.cid
            or len(self.cid.encode("utf-8")) > 512
        ):
            raise ValueError("cache CID is invalid")
        for field in ("stored_at", "expires_at"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("cache timestamps must be numeric")
        if self.expires_at <= self.stored_at:
            raise ValueError("cache expiry must follow storage time")
        if not isinstance(self.stale, bool):
            raise ValueError("cache stale marker must be boolean")

    @property
    def revision(self) -> str:
        return self.cid

    def is_fresh(self, now: float) -> bool:
        return not self.stale and now < self.expires_at


CachedSnapshot = CacheEntry
CacheRecord = CacheEntry


@dataclass
class _Flight:
    view: CacheView
    condition: threading.Condition
    done: bool = False
    error: Optional[BaseException] = None
    cancelled: bool = False


class CatalogSnapshotCache:
    """Bounded, source-scoped cache with synchronous and asynchronous flights."""

    def __init__(
        self,
        policy: Optional[CachePolicy] = None,
        *,
        capabilities_ttl: Optional[float] = None,
        health_ttl: Optional[float] = None,
        max_sources: int = MAX_CACHE_SOURCES,
        clock: Optional[Callable[[], float]] = None,
        wall_clock: Optional[Callable[[], Any]] = None,
        metrics: Optional[CatalogMetrics] = None,
        events: Optional[CatalogEventBus] = None,
    ) -> None:
        if policy is not None and (
            capabilities_ttl is not None or health_ttl is not None
        ):
            raise ValueError("pass policy or TTL keyword arguments, not both")
        if policy is None:
            policy = CachePolicy(
                capabilities_ttl=(
                    DEFAULT_CAPABILITIES_TTL
                    if capabilities_ttl is None
                    else capabilities_ttl
                ),
                health_ttl=(
                    DEFAULT_HEALTH_TTL if health_ttl is None else health_ttl
                ),
            )
        if not isinstance(policy, CachePolicy):
            raise TypeError("policy must be a CachePolicy")
        if (
            isinstance(max_sources, bool)
            or not isinstance(max_sources, int)
            or not 1 <= max_sources <= MAX_CACHE_SOURCES
        ):
            raise ValueError("max_sources is invalid")
        self.policy = policy
        self.max_sources = max_sources
        self._clock = time.monotonic if clock is None else clock
        self._wall_clock = (
            lambda: datetime.now(timezone.utc)
            if wall_clock is None
            else wall_clock
        )
        if not callable(self._clock) or not callable(self._wall_clock):
            raise TypeError("cache clocks must be callable")
        self.metrics = CatalogMetrics(clock=self._clock) if metrics is None else metrics
        if not isinstance(self.metrics, CatalogMetrics):
            raise TypeError("metrics must be CatalogMetrics")
        self._lock = threading.RLock()
        self._entries = {}  # type: Dict[Tuple[str, CacheView], CacheEntry]
        self._flights = {}  # type: Dict[str, _Flight]
        self._async_flights = {}  # type: Dict[Tuple[int, str], asyncio.Task]
        self._invalidated = set()  # type: set[Tuple[str, CacheView]]
        self._unsubscribe = events.subscribe(self.invalidate) if events else None

    @property
    def capabilities_ttl(self) -> float:
        return self.policy.capabilities_ttl

    @property
    def health_ttl(self) -> float:
        return self.policy.health_ttl

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("cache clock must return a numeric value")
        return float(value)

    def _key(self, source: Any, view: Any) -> Tuple[str, CacheView]:
        return _source(source), _view(view)

    def _miss_reason(
        self, key: Tuple[str, CacheView], entry: Optional[CacheEntry], now: float
    ) -> str:
        if key in self._invalidated:
            return "invalidated"
        if entry is None:
            return "absent"
        if not entry.is_fresh(now):
            return "expired"
        return "other"

    def peek(
        self, source: str, view: Any, *, include_stale: bool = True
    ) -> Optional[CacheEntry]:
        key = self._key(source, view)
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_fresh(now):
            return entry
        self.metrics.set_stale_records(
            key[0], key[1], _record_count(entry.value)
        )
        return replace(entry, stale=True) if include_stale else None

    entry = peek

    def _fresh(self, key: Tuple[str, CacheView], now: float) -> Optional[CacheEntry]:
        entry = self._entries.get(key)
        return entry if entry is not None and entry.is_fresh(now) else None

    def _install(
        self,
        key: Tuple[str, CacheView],
        value: Any,
        *,
        now: Optional[float] = None,
    ) -> CacheEntry:
        selected_now = self._now() if now is None else now
        cid = _value_cid(value)
        with self._lock:
            sources = {item[0] for item in self._entries}
            if key[0] not in sources and len(sources) >= self.max_sources:
                raise ValueError("catalog cache source capacity exceeded")
            previous = self._entries.get(key)
            # Reuse the prior immutable object as well as its CID when canonical
            # content did not change.  This makes identity reuse observable to
            # callers without retaining duplicate snapshots.
            selected_value = (
                previous.value
                if previous is not None and previous.cid == cid
                else value
            )
            entry = CacheEntry(
                source=key[0],
                view=key[1],
                value=selected_value,
                cid=cid,
                stored_at=selected_now,
                expires_at=selected_now + self.policy.ttl_for(key[1]),
            )
            self._entries[key] = entry
            self._invalidated.discard(key)
        self.metrics.set_stale_records(key[0], key[1], 0)
        return entry

    def put(self, source: str, view: Any, value: Any) -> CacheEntry:
        key = self._key(source, view)
        with self._lock:
            sources = {item[0] for item in self._entries}
            if key[0] not in sources and len(sources) >= self.max_sources:
                raise ValueError("catalog cache source capacity exceeded")
        return self._install(key, value)

    def get_or_refresh(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        *,
        force: bool = False,
        allow_stale_on_error: Optional[bool] = None,
    ) -> CacheEntry:
        """Return a fresh entry, collapsing concurrent loads for one source."""

        if not callable(loader):
            raise TypeError("cache loader must be callable")
        if not isinstance(force, bool):
            raise ValueError("force must be boolean")
        stale_allowed = (
            self.policy.allow_stale_on_error
            if allow_stale_on_error is None
            else allow_stale_on_error
        )
        if not isinstance(stale_allowed, bool):
            raise ValueError("allow_stale_on_error must be boolean")
        key = self._key(source, view)
        joined = False
        while True:
            now = self._now()
            with self._lock:
                fresh = None if force and not joined else self._fresh(key, now)
                if fresh is not None:
                    self.metrics.record_cache_hit(key[0], key[1])
                    return fresh
                existing = self._entries.get(key)
                flight = self._flights.get(key[0])
                if flight is not None:
                    joined = True
                    while not flight.done:
                        flight.condition.wait()
                    if flight.cancelled:
                        continue
                    if flight.view == key[1] and flight.error is not None:
                        if stale_allowed and existing is not None:
                            self.metrics.set_stale_records(
                                key[0],
                                key[1],
                                _record_count(existing.value),
                            )
                            return replace(existing, stale=True)
                        raise flight.error
                    continue
                self.metrics.record_cache_miss(
                    key[0], key[1], self._miss_reason(key, existing, now)
                )
                condition = threading.Condition(self._lock)
                flight = _Flight(key[1], condition)
                self._flights[key[0]] = flight
                break

        started = self._now()
        try:
            value = loader()
            if inspect.isawaitable(value):
                if inspect.iscoroutine(value):
                    value.close()
                raise TypeError("synchronous cache loader returned an awaitable")
            entry = self._install(key, value)
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            with self._lock:
                flight.error = exc
                flight.cancelled = cancelled
                flight.done = True
                self._flights.pop(key[0], None)
                flight.condition.notify_all()
                existing = self._entries.get(key)
            self.metrics.record_source_latency(
                key[0], max(0.0, self._now() - started)
            )
            if (
                not cancelled
                and isinstance(exc, Exception)
                and stale_allowed
                and existing is not None
            ):
                self.metrics.set_stale_records(
                    key[0], key[1], _record_count(existing.value)
                )
                return replace(existing, stale=True)
            raise
        with self._lock:
            flight.done = True
            self._flights.pop(key[0], None)
            flight.condition.notify_all()
        self.metrics.record_source_latency(
            key[0], max(0.0, self._now() - started)
        )
        return entry

    def refresh(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        **kwargs: Any
    ) -> CacheEntry:
        """Force a source refresh while still joining an existing flight."""

        kwargs.setdefault("force", True)
        return self.get_or_refresh(source, view, loader, **kwargs)

    get_or_load = get_or_refresh

    def get(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        **kwargs: Any
    ) -> Any:
        return self.get_or_refresh(source, view, loader, **kwargs).value

    def get_capabilities(
        self, source: str, loader: Callable[[], Any], **kwargs: Any
    ) -> Any:
        return self.get(source, CacheView.CAPABILITIES, loader, **kwargs)

    def get_health(
        self, source: str, loader: Callable[[], Any], **kwargs: Any
    ) -> Any:
        return self.get(source, CacheView.HEALTH, loader, **kwargs)

    async def _async_load(
        self,
        flight_key: Tuple[int, str],
        key: Tuple[str, CacheView],
        loader: Callable[[], Any],
    ) -> CacheEntry:
        started = self._now()
        try:
            value = loader()
            if inspect.isawaitable(value):
                value = await value
            return self._install(key, value)
        finally:
            self.metrics.record_source_latency(
                key[0], max(0.0, self._now() - started)
            )
            with self._lock:
                current = self._async_flights.get(flight_key)
                if current is asyncio.current_task():
                    self._async_flights.pop(flight_key, None)

    async def get_or_refresh_async(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        *,
        force: bool = False,
    ) -> CacheEntry:
        """Async single-flight whose shared loader survives waiter cancellation."""

        if not callable(loader):
            raise TypeError("cache loader must be callable")
        key = self._key(source, view)
        loop = asyncio.get_running_loop()
        joined = False
        while True:
            now = self._now()
            with self._lock:
                fresh = None if force and not joined else self._fresh(key, now)
                if fresh is not None:
                    self.metrics.record_cache_hit(key[0], key[1])
                    return fresh
                existing = self._entries.get(key)
                flight_key = (id(loop), key[0])
                task = self._async_flights.get(flight_key)
                if task is None:
                    self.metrics.record_cache_miss(
                        key[0], key[1], self._miss_reason(key, existing, now)
                    )
                    task = loop.create_task(
                        self._async_load(flight_key, key, loader)
                    )
                    self._async_flights[flight_key] = task
                    task_view = key[1]
                else:
                    joined = True
                    task_view = getattr(task, "_catalog_cache_view", None)
                # This private marker never contains user data and lets a
                # different projection wait for the source flight then retry.
                if not hasattr(task, "_catalog_cache_view"):
                    setattr(task, "_catalog_cache_view", key[1])
                    task_view = key[1]
            result = await asyncio.shield(task)
            if task_view == key[1]:
                return result

    async def get_async(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        **kwargs: Any
    ) -> Any:
        return (
            await self.get_or_refresh_async(source, view, loader, **kwargs)
        ).value

    async def refresh_async(
        self,
        source: str,
        view: Any,
        loader: Callable[[], Any],
        **kwargs: Any
    ) -> CacheEntry:
        kwargs.setdefault("force", True)
        return await self.get_or_refresh_async(source, view, loader, **kwargs)

    async def get_capabilities_async(
        self, source: str, loader: Callable[[], Any], **kwargs: Any
    ) -> Any:
        return await self.get_async(
            source, CacheView.CAPABILITIES, loader, **kwargs
        )

    async def get_health_async(
        self, source: str, loader: Callable[[], Any], **kwargs: Any
    ) -> Any:
        return await self.get_async(source, CacheView.HEALTH, loader, **kwargs)

    def invalidate(
        self,
        event_or_source: Any = None,
        views: Optional[Iterable[Any]] = None,
        *,
        source: Optional[str] = None,
    ) -> Tuple[Tuple[str, CacheView], ...]:
        """Invalidate only the source/view pairs selected by an event."""

        if source is not None:
            if event_or_source is not None:
                raise ValueError("source was provided more than once")
            event_or_source = source
        if isinstance(event_or_source, CatalogInvalidationEvent):
            if views is not None:
                raise ValueError("views cannot accompany an invalidation event")
            selected_source = event_or_source.source
            selected_views = event_or_source.affected_views
        else:
            selected_source = (
                None if event_or_source is None else _source(event_or_source)
            )
            if views is None:
                selected_views = tuple(CacheView)
            elif isinstance(views, (str, bytes, CacheView)):
                selected_views = (_view(views),)
            else:
                selected_views = tuple(_view(item) for item in views)
            if not selected_views:
                raise ValueError("invalidation must affect at least one view")
        with self._lock:
            keys = tuple(
                sorted(
                    (
                        key
                        for key in self._entries
                        if (selected_source is None or key[0] == selected_source)
                        and key[1] in selected_views
                    ),
                    key=lambda item: (item[0], item[1].value),
                )
            )
            for key in keys:
                del self._entries[key]
                self._invalidated.add(key)
        for source_name, selected_view in keys:
            self.metrics.set_stale_records(source_name, selected_view, 0)
        return keys

    invalidate_event = invalidate
    handle_event = invalidate

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._invalidated.clear()
        return count

    def entries(self) -> Tuple[CacheEntry, ...]:
        with self._lock:
            return tuple(
                self._entries[key]
                for key in sorted(
                    self._entries, key=lambda item: (item[0], item[1].value)
                )
            )

    def stale_entries(self) -> Tuple[CacheEntry, ...]:
        now = self._now()
        return tuple(
            replace(item, stale=True)
            for item in self.entries()
            if not item.is_fresh(now)
        )

    @property
    def timestamp(self) -> str:
        """Current injected wall time, exposed for deterministic diagnostics."""

        return _wall_timestamp(self._wall_clock)


SnapshotCache = CatalogSnapshotCache
CatalogCache = CatalogSnapshotCache
SourceSnapshotCache = CatalogSnapshotCache


__all__ = [
    "CacheEntry",
    "CachePolicy",
    "CacheRecord",
    "CachedSnapshot",
    "CatalogCache",
    "CatalogCachePolicy",
    "CatalogSnapshotCache",
    "DEFAULT_CAPABILITIES_TTL",
    "DEFAULT_HEALTH_TTL",
    "HealthSample",
    "HealthSnapshot",
    "MAX_CACHE_SOURCES",
    "MAX_CACHE_TTL",
    "MAX_HEALTH_SAMPLES",
    "SnapshotCache",
    "SourceSnapshotCache",
]
