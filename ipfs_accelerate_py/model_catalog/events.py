"""Typed catalog invalidation events and bounded in-process observability.

The catalog is intentionally usable without an observability dependency.  This
module provides the small common contract used by cache, registry, deployment,
and federation adapters: events describe *which projection* changed, while
metrics keep only low-cardinality operational labels.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple


MAX_EVENT_RECORDS = 1_000
MAX_EVENT_SUBSCRIBERS = 256
MAX_METRIC_SERIES = 2_048
MAX_METRIC_SOURCES = 64

_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9._/-]{0,126}[a-z0-9])?$")
_LABEL_VALUE = re.compile(r"^[a-z0-9](?:[a-z0-9._/-]{0,62}[a-z0-9])?$")


class CatalogView(str, Enum):
    """Independently cached catalog projections."""

    CAPABILITIES = "capabilities"
    HEALTH = "health"


# CacheView is the more natural spelling for cache consumers.
CacheView = CatalogView


class CatalogEventType(str, Enum):
    """Events which can make one or more catalog projections stale."""

    REGISTRATION = "registration"
    DEPLOYMENT_LIFECYCLE = "deployment_lifecycle"
    CREDENTIAL_STATE = "credential_state"
    EXPLICIT_REFRESH = "explicit_refresh"
    PEER_REVISION = "peer_revision"


# EventKind is retained as a concise spelling for producers.
EventKind = CatalogEventType


_DEFAULT_VIEWS = {
    CatalogEventType.REGISTRATION: (CatalogView.CAPABILITIES,),
    CatalogEventType.DEPLOYMENT_LIFECYCLE: (CatalogView.HEALTH,),
    CatalogEventType.CREDENTIAL_STATE: (CatalogView.HEALTH,),
    CatalogEventType.EXPLICIT_REFRESH: (
        CatalogView.CAPABILITIES,
        CatalogView.HEALTH,
    ),
    CatalogEventType.PEER_REVISION: (
        CatalogView.CAPABILITIES,
        CatalogView.HEALTH,
    ),
}


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("%s must be canonical text" % field)
    value = value.casefold()
    if (
        len(value.encode("utf-8")) > 128
        or not _NAME.fullmatch(value)
        or "//" in value
        or ".." in value
    ):
        raise ValueError("%s must be canonical text" % field)
    return value


def _event_type(value: Any) -> CatalogEventType:
    if isinstance(value, CatalogEventType):
        return value
    try:
        return CatalogEventType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown catalog event type") from exc


def _views(
    values: Optional[Iterable[Any]], kind: CatalogEventType
) -> Tuple[CatalogView, ...]:
    if values is None:
        return _DEFAULT_VIEWS[kind]
    if isinstance(values, (str, bytes, Mapping)):
        values = (values,)
    result = []
    for value in values:
        try:
            selected = value if isinstance(value, CatalogView) else CatalogView(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown catalog view") from exc
        if selected not in result:
            result.append(selected)
    if not result:
        raise ValueError("an invalidation event must affect at least one view")
    return tuple(sorted(result, key=lambda item: item.value))


@dataclass(frozen=True)
class CatalogInvalidationEvent:
    """A bounded declaration that selected source projections are stale.

    ``source=None`` deliberately means every source.  Record identities are
    optional diagnostic scope; cache implementations never need to inspect
    record bodies or endpoint data to process an event.
    """

    kind: CatalogEventType
    source: Optional[str] = None
    record_ids: Tuple[str, ...] = ()
    views: Optional[Tuple[CatalogView, ...]] = None
    previous_revision: Optional[str] = None
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        kind = _event_type(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.source is not None:
            object.__setattr__(self, "source", _name(self.source, "source"))
        if isinstance(self.record_ids, (str, bytes, Mapping)):
            raise ValueError("record_ids must be a bounded array")
        record_ids = tuple(sorted(set(self.record_ids)))
        if len(record_ids) > MAX_EVENT_RECORDS or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > 256
            for item in record_ids
        ):
            raise ValueError("record_ids are invalid or excessive")
        object.__setattr__(self, "record_ids", record_ids)
        object.__setattr__(self, "views", _views(self.views, kind))
        for field in ("previous_revision", "revision"):
            value = getattr(self, field)
            if value is not None and (
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > 512
            ):
                raise ValueError("%s is invalid" % field)

    @property
    def event_type(self) -> CatalogEventType:
        return self.kind

    @property
    def affected_views(self) -> Tuple[CatalogView, ...]:
        return self.views  # type: ignore[return-value]

    def affects(self, source: str, view: Any) -> bool:
        canonical = _name(source, "source")
        selected = view if isinstance(view, CatalogView) else CatalogView(view)
        return (self.source is None or self.source == canonical) and (
            selected in self.affected_views
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source": self.source,
            "record_ids": list(self.record_ids),
            "views": [item.value for item in self.affected_views],
            "previous_revision": self.previous_revision,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CatalogInvalidationEvent":
        fields = {
            "kind",
            "source",
            "record_ids",
            "views",
            "previous_revision",
            "revision",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("CatalogInvalidationEvent has missing or unknown fields")
        return cls(
            kind=data["kind"],
            source=data["source"],
            record_ids=tuple(data["record_ids"]),
            views=tuple(data["views"]),
            previous_revision=data["previous_revision"],
            revision=data["revision"],
        )


InvalidationEvent = CatalogInvalidationEvent
CatalogEvent = CatalogInvalidationEvent


def registration_event(
    source: str, record_ids: Iterable[str] = ()
) -> CatalogInvalidationEvent:
    return CatalogInvalidationEvent(
        CatalogEventType.REGISTRATION,
        source=source,
        record_ids=tuple(record_ids),
    )


def deployment_lifecycle_event(
    source: str, record_ids: Iterable[str] = ()
) -> CatalogInvalidationEvent:
    return CatalogInvalidationEvent(
        CatalogEventType.DEPLOYMENT_LIFECYCLE,
        source=source,
        record_ids=tuple(record_ids),
    )


def credential_state_event(
    source: str, record_ids: Iterable[str] = ()
) -> CatalogInvalidationEvent:
    return CatalogInvalidationEvent(
        CatalogEventType.CREDENTIAL_STATE,
        source=source,
        record_ids=tuple(record_ids),
    )


def explicit_refresh_event(
    source: Optional[str] = None,
    *,
    views: Optional[Iterable[Any]] = None,
) -> CatalogInvalidationEvent:
    return CatalogInvalidationEvent(
        CatalogEventType.EXPLICIT_REFRESH,
        source=source,
        views=views,  # type: ignore[arg-type]
    )


def peer_revision_event(
    source: str,
    revision: str,
    *,
    previous_revision: Optional[str] = None,
) -> CatalogInvalidationEvent:
    return CatalogInvalidationEvent(
        CatalogEventType.PEER_REVISION,
        source=source,
        revision=revision,
        previous_revision=previous_revision,
    )


class CatalogEventBus:
    """Synchronous, bounded event fan-out.

    Subscribers run outside the bus lock.  A failing subscriber cannot prevent
    the remaining subscribers from observing the invalidation.
    """

    def __init__(self, max_subscribers: int = MAX_EVENT_SUBSCRIBERS) -> None:
        if (
            isinstance(max_subscribers, bool)
            or not isinstance(max_subscribers, int)
            or not 1 <= max_subscribers <= MAX_EVENT_SUBSCRIBERS
        ):
            raise ValueError(
                "max_subscribers must be between 1 and %d"
                % MAX_EVENT_SUBSCRIBERS
            )
        self.max_subscribers = max_subscribers
        self._lock = threading.RLock()
        self._subscribers = []  # type: list[Callable[[CatalogInvalidationEvent], Any]]

    def subscribe(
        self, callback: Callable[[CatalogInvalidationEvent], Any]
    ) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("event subscriber must be callable")
        with self._lock:
            if callback not in self._subscribers:
                if len(self._subscribers) >= self.max_subscribers:
                    raise ValueError("catalog event subscriber capacity exceeded")
                self._subscribers.append(callback)

        def unsubscribe() -> None:
            self.unsubscribe(callback)

        return unsubscribe

    def unsubscribe(
        self, callback: Callable[[CatalogInvalidationEvent], Any]
    ) -> bool:
        with self._lock:
            if callback not in self._subscribers:
                return False
            self._subscribers.remove(callback)
            return True

    def publish(self, event: CatalogInvalidationEvent) -> Tuple[BaseException, ...]:
        if not isinstance(event, CatalogInvalidationEvent):
            raise TypeError("event must be a CatalogInvalidationEvent")
        with self._lock:
            subscribers = tuple(self._subscribers)
        failures = []
        for callback in subscribers:
            try:
                callback(event)
            except BaseException as exc:
                failures.append(exc)
        return tuple(failures)

    emit = publish


EventBus = CatalogEventBus


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: Tuple[Tuple[str, str], ...]
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "labels": dict(self.labels),
            "value": self.value,
        }


_METRIC_LABELS = {
    "catalog_source_latency_seconds_count": ("source",),
    "catalog_source_latency_seconds_sum": ("source",),
    "catalog_cache_hits_total": ("source", "view"),
    "catalog_cache_misses_total": ("source", "view", "reason"),
    "catalog_stale_records": ("source", "view"),
    "catalog_conflicts_total": ("kind",),
    "catalog_no_match_total": ("reason",),
    "catalog_resolutions_total": ("outcome",),
    "catalog_health_transitions_total": ("from_state", "to_state"),
}

_BOUNDED_VALUES = {
    "view": frozenset(("capabilities", "health")),
    "reason": frozenset(
        (
            "absent",
            "expired",
            "invalidated",
            "constraints",
            "policy",
            "health",
            "ambiguous",
            "unavailable",
            "other",
        )
    ),
    "kind": frozenset(("precedence", "ambiguous", "alias", "other")),
    "outcome": frozenset(("selected", "no_match", "error")),
    "from_state": frozenset(("healthy", "unhealthy", "unknown")),
    "to_state": frozenset(("healthy", "unhealthy", "unknown")),
}


class CatalogMetrics:
    """Dependency-free metrics with an allowlisted, bounded label space."""

    def __init__(
        self,
        *,
        max_series: int = MAX_METRIC_SERIES,
        max_sources: int = MAX_METRIC_SOURCES,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if (
            isinstance(max_series, bool)
            or not isinstance(max_series, int)
            or not 1 <= max_series <= MAX_METRIC_SERIES
        ):
            raise ValueError("max_series is invalid")
        if (
            isinstance(max_sources, bool)
            or not isinstance(max_sources, int)
            or not 1 <= max_sources <= MAX_METRIC_SOURCES
        ):
            raise ValueError("max_sources is invalid")
        if clock is None:
            import time

            clock = time.monotonic
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.max_series = max_series
        self.max_sources = max_sources
        self._clock = clock
        self._lock = threading.RLock()
        self._values = {}  # type: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]
        self._sources = set()  # type: set[str]

    def _source_label(self, value: Any) -> str:
        try:
            canonical = _name(value, "source")
        except (TypeError, ValueError):
            return "other"
        with self._lock:
            if canonical in self._sources:
                return canonical
            if len(self._sources) >= self.max_sources:
                return "other"
            self._sources.add(canonical)
        return canonical

    def _labels(
        self, metric: str, labels: Mapping[str, Any]
    ) -> Tuple[Tuple[str, str], ...]:
        expected = _METRIC_LABELS.get(metric)
        if expected is None:
            raise ValueError("unknown catalog metric: %s" % metric)
        if set(labels) != set(expected):
            raise ValueError("metric labels must exactly match the metric contract")
        result = []
        for name in expected:
            value = labels[name]
            if name == "source":
                selected = self._source_label(value)
            else:
                selected = str(value).strip().casefold()
                allowed = _BOUNDED_VALUES[name]
                if selected not in allowed:
                    selected = "other" if "other" in allowed else "unknown"
            if (
                len(selected.encode("utf-8")) > 64
                or not _LABEL_VALUE.fullmatch(selected)
            ):
                selected = "other"
            result.append((name, selected))
        return tuple(result)

    def _update(
        self,
        metric: str,
        amount: float,
        labels: Mapping[str, Any],
        *,
        gauge: bool = False,
    ) -> None:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("metric value must be numeric")
        if amount < 0:
            raise ValueError("metric value must be non-negative")
        selected = self._labels(metric, labels)
        key = (metric, selected)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_series:
                return
            self._values[key] = float(amount) if gauge else (
                self._values.get(key, 0.0) + float(amount)
            )

    def record_source_latency(self, source: str, seconds: float) -> None:
        if seconds < 0:
            seconds = 0.0
        labels = {"source": source}
        self._update("catalog_source_latency_seconds_count", 1, labels)
        self._update("catalog_source_latency_seconds_sum", seconds, labels)

    observe_source_latency = record_source_latency

    @contextmanager
    def time_source(self, source: str) -> Iterator[None]:
        started = self._clock()
        try:
            yield
        finally:
            self.record_source_latency(source, max(0.0, self._clock() - started))

    def record_cache_hit(self, source: str, view: Any) -> None:
        selected = view.value if isinstance(view, CatalogView) else view
        self._update(
            "catalog_cache_hits_total",
            1,
            {"source": source, "view": selected},
        )

    cache_hit = record_cache_hit

    def record_cache_miss(self, source: str, view: Any, reason: str) -> None:
        selected = view.value if isinstance(view, CatalogView) else view
        self._update(
            "catalog_cache_misses_total",
            1,
            {"source": source, "view": selected, "reason": reason},
        )

    cache_miss = record_cache_miss

    def set_stale_records(self, source: str, view: Any, count: int) -> None:
        selected = view.value if isinstance(view, CatalogView) else view
        self._update(
            "catalog_stale_records",
            count,
            {"source": source, "view": selected},
            gauge=True,
        )

    stale_records = set_stale_records

    def record_conflict(self, kind: str = "other") -> None:
        self._update("catalog_conflicts_total", 1, {"kind": kind})

    conflict = record_conflict

    @staticmethod
    def classify_no_match(reasons: Iterable[str]) -> str:
        text = " ".join(str(item).casefold() for item in reasons)
        if "policy" in text:
            return "policy"
        if "health" in text or "healthy" in text:
            return "health"
        if "ambiguous" in text:
            return "ambiguous"
        if "unavailable" in text or "missing" in text:
            return "unavailable"
        if "constraint" in text or "mismatch" in text or "unsupported" in text:
            return "constraints"
        return "other"

    def record_no_match(self, reason: str = "other") -> None:
        self._update("catalog_no_match_total", 1, {"reason": reason})

    no_match = record_no_match

    def record_resolution(
        self,
        result: Any = None,
        *,
        outcome: Optional[str] = None,
        no_match_reason: Optional[str] = None,
    ) -> None:
        if outcome is None:
            if result is None:
                outcome = "error"
            else:
                outcome = "selected" if bool(getattr(result, "found", False)) else "no_match"
        self._update("catalog_resolutions_total", 1, {"outcome": outcome})
        if outcome == "no_match":
            reason = no_match_reason
            if reason is None:
                reason = self.classify_no_match(getattr(result, "reasons", ()))
            self.record_no_match(reason)

    resolution = record_resolution

    @staticmethod
    def _health_label(value: Any) -> str:
        if value is True or value == "healthy":
            return "healthy"
        if value is False or value == "unhealthy":
            return "unhealthy"
        return "unknown"

    def record_health_transition(self, previous: Any, current: Any) -> None:
        before = self._health_label(previous)
        after = self._health_label(current)
        if before == after:
            return
        self._update(
            "catalog_health_transitions_total",
            1,
            {"from_state": before, "to_state": after},
        )

    health_transition = record_health_transition

    def snapshot(self) -> Tuple[MetricSample, ...]:
        with self._lock:
            values = tuple(self._values.items())
        return tuple(
            MetricSample(name=name, labels=labels, value=value)
            for (name, labels), value in sorted(values)
        )

    def to_dict(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(item.to_dict() for item in self.snapshot())

    def value(self, metric: str, **labels: Any) -> float:
        selected = self._labels(metric, labels)
        with self._lock:
            return self._values.get((metric, selected), 0.0)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._sources.clear()


Metrics = CatalogMetrics
ObservabilityMetrics = CatalogMetrics


__all__ = [
    "CacheView",
    "CatalogEvent",
    "CatalogEventBus",
    "CatalogEventType",
    "CatalogInvalidationEvent",
    "CatalogMetrics",
    "CatalogView",
    "EventBus",
    "EventKind",
    "InvalidationEvent",
    "MAX_EVENT_RECORDS",
    "MAX_EVENT_SUBSCRIBERS",
    "MAX_METRIC_SERIES",
    "MAX_METRIC_SOURCES",
    "MetricSample",
    "Metrics",
    "ObservabilityMetrics",
    "credential_state_event",
    "deployment_lifecycle_event",
    "explicit_refresh_event",
    "peer_revision_event",
    "registration_event",
]
