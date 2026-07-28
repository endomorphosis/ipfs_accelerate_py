"""Event-derived low-cardinality usage observability.

Metrics are derived from ledger events and control-plane outcomes so dashboards
cannot become a second source of truth. Labels are bounded to provider,
deployment, state, and reason vocabularies — never request id, credential,
tenant, alias, model string, or endpoint URL cardinality.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .controls import HEADROOM_BANDS, headroom_band
from .schema import (
    AvailabilityState,
    Quantity,
    UsageDimension,
    UsageEventKind,
    UsageSnapshot,
)

USAGE_OBSERVABILITY_REQUIREMENT_ID = (
    "requirement:endpoint-usage-observability.v1"
)
USAGE_METRICS_SCHEMA_VERSION = "ai.endpoint_usage.metrics.v1"

MAX_METRIC_SERIES = 4_096
MAX_PROVIDER_LABELS = 64
MAX_DEPLOYMENT_LABELS = 128
MAX_LABEL_BYTES = 64

_LABEL_VALUE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")

# Forbidden high-cardinality label keys (defense in depth).
_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "request",
        "request_id",
        "credential",
        "credential_id",
        "credential_pseudonym",
        "tenant",
        "tenant_id",
        "alias",
        "model",
        "model_id",
        "model_string",
        "endpoint",
        "endpoint_uri",
        "endpoint_url",
        "url",
        "account",
        "account_pseudonym",
        "user",
        "session",
    }
)

_PROVIDER_SAFE = re.compile(r"^provider:[a-z0-9._:-]{1,80}$|^[a-z0-9._:-]{1,64}$")
_DEPLOYMENT_SAFE = re.compile(
    r"^deployment:[a-z0-9._:-]{1,80}$|^[a-z0-9._:-]{1,64}$"
)

# Metric name -> required label keys (exact match).
_METRIC_LABELS: Dict[str, Tuple[str, ...]] = {
    "usage_reservations_total": ("provider", "deployment", "outcome"),
    "usage_reservation_denials_total": ("provider", "deployment", "reason"),
    "usage_reservation_expiry_total": ("provider", "deployment", "reason"),
    "usage_estimate_error_ratio_sum": ("provider", "deployment", "dimension"),
    "usage_estimate_error_ratio_count": ("provider", "deployment", "dimension"),
    "usage_headroom_band": ("provider", "deployment", "dimension", "band"),
    "usage_resets_total": ("provider", "deployment", "reason"),
    "usage_stale_scopes": ("provider", "state"),
    "usage_unknown_scopes": ("provider", "state"),
    "usage_waits_total": ("provider", "deployment", "reason"),
    "usage_reroutes_total": ("provider", "deployment", "reason"),
    "usage_fallbacks_total": ("provider", "deployment", "reason"),
    "usage_reconciliation_total": ("provider", "deployment", "kind"),
    "usage_store_health": ("state",),
    "usage_control_mutations_total": ("operation", "outcome"),
    "usage_events_total": ("kind", "reason"),
}

_BOUNDED_VALUES: Dict[str, frozenset] = {
    "outcome": frozenset(
        ("accepted", "denied", "expired", "success", "error", "other")
    ),
    "reason": frozenset(
        (
            "ok",
            "limit_exhausted",
            "capacity_unavailable",
            "policy_denied",
            "stale_snapshot",
            "reservation_conflict",
            "cooling_down",
            "timeout",
            "cancelled",
            "fallback",
            "reroute",
            "wait",
            "import",
            "correction",
            "reset",
            "expiry_reclamation",
            "store_unhealthy",
            "unknown",
            "other",
        )
    ),
    "dimension": frozenset(item.value for item in UsageDimension) | frozenset(("other",)),
    "band": frozenset(HEADROOM_BANDS) | frozenset(("other",)),
    "state": frozenset(item.value for item in AvailabilityState)
    | frozenset(("healthy", "unhealthy", "other")),
    "kind": frozenset(
        tuple(item.value for item in UsageEventKind)
        + ("import", "correction", "override", "reset", "other")
    ),
    "operation": frozenset(
        (
            "import",
            "correct",
            "override",
            "reset",
            "status",
            "health",
            "limits",
            "headroom",
            "reservations",
            "receipts",
            "route_preview",
            "adapter_capabilities",
            "other",
        )
    ),
}


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


class UsageObservability:
    """Dependency-free metrics with an allowlisted, bounded label space."""

    requirement_id = USAGE_OBSERVABILITY_REQUIREMENT_ID

    def __init__(
        self,
        *,
        max_series: int = MAX_METRIC_SERIES,
        max_providers: int = MAX_PROVIDER_LABELS,
        max_deployments: int = MAX_DEPLOYMENT_LABELS,
    ) -> None:
        if (
            isinstance(max_series, bool)
            or not isinstance(max_series, int)
            or not 1 <= max_series <= MAX_METRIC_SERIES
        ):
            raise ValueError("max_series is invalid")
        if (
            isinstance(max_providers, bool)
            or not isinstance(max_providers, int)
            or not 1 <= max_providers <= MAX_PROVIDER_LABELS
        ):
            raise ValueError("max_providers is invalid")
        if (
            isinstance(max_deployments, bool)
            or not isinstance(max_deployments, int)
            or not 1 <= max_deployments <= MAX_DEPLOYMENT_LABELS
        ):
            raise ValueError("max_deployments is invalid")
        self.max_series = max_series
        self.max_providers = max_providers
        self.max_deployments = max_deployments
        self._lock = threading.RLock()
        self._values: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._providers: set[str] = set()
        self._deployments: set[str] = set()

    # -- label hygiene -----------------------------------------------------

    def _bounded_token(
        self,
        value: Any,
        *,
        allowed: Optional[frozenset] = None,
        default: str = "other",
    ) -> str:
        if value is None:
            return default
        text = str(value).strip().casefold()
        if not text:
            return default
        if len(text.encode("utf-8")) > MAX_LABEL_BYTES:
            return default
        if not _LABEL_VALUE.fullmatch(text):
            # Soft-sanitize: keep alnum/._:- only.
            cleaned = re.sub(r"[^a-z0-9._:-]+", "-", text).strip("-")[:MAX_LABEL_BYTES]
            text = cleaned if cleaned and _LABEL_VALUE.fullmatch(cleaned) else default
        if allowed is not None and text not in allowed:
            return default if default in (allowed or frozenset()) else "other"
        return text

    def _provider_label(self, value: Any) -> str:
        if value is None:
            return "other"
        text = str(value).strip().casefold()
        if not text or not _PROVIDER_SAFE.fullmatch(text):
            return "other"
        if len(text.encode("utf-8")) > MAX_LABEL_BYTES:
            return "other"
        with self._lock:
            if text in self._providers:
                return text
            if len(self._providers) >= self.max_providers:
                return "other"
            self._providers.add(text)
        return text

    def _deployment_label(self, value: Any) -> str:
        if value is None:
            return "other"
        text = str(value).strip().casefold()
        if not text or not _DEPLOYMENT_SAFE.fullmatch(text):
            return "other"
        if len(text.encode("utf-8")) > MAX_LABEL_BYTES:
            return "other"
        with self._lock:
            if text in self._deployments:
                return text
            if len(self._deployments) >= self.max_deployments:
                return "other"
            self._deployments.add(text)
        return text

    def _labels(
        self, metric: str, labels: Mapping[str, Any]
    ) -> Tuple[Tuple[str, str], ...]:
        expected = _METRIC_LABELS.get(metric)
        if expected is None:
            raise ValueError("unknown usage metric: %s" % metric)
        # Reject forbidden high-cardinality keys even if extra.
        for key in labels:
            if str(key).casefold() in _FORBIDDEN_LABEL_KEYS:
                raise ValueError("forbidden metric label: %s" % key)
        if set(labels) != set(expected):
            raise ValueError("metric labels must exactly match the metric contract")
        result: List[Tuple[str, str]] = []
        for name in expected:
            raw = labels[name]
            if name == "provider":
                selected = self._provider_label(raw)
            elif name == "deployment":
                selected = self._deployment_label(raw)
            else:
                allowed = _BOUNDED_VALUES.get(name)
                selected = self._bounded_token(raw, allowed=allowed, default="other")
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
            if gauge:
                self._values[key] = float(amount)
            else:
                self._values[key] = self._values.get(key, 0.0) + float(amount)

    # -- event-derived recording ------------------------------------------

    def ingest_event(
        self,
        event: Mapping[str, Any] | Any,
        *,
        provider: str = "other",
        deployment: str = "other",
    ) -> None:
        """Derive counters from one ledger event."""

        if hasattr(event, "to_dict"):
            payload = event.to_dict()
        elif isinstance(event, Mapping):
            payload = dict(event)
        else:
            return
        kind = str(payload.get("kind") or "other")
        reasons = payload.get("reason_codes") or ()
        reason = str(reasons[0]) if reasons else "ok"
        self._update(
            "usage_events_total",
            1,
            {"kind": kind, "reason": reason},
        )
        if kind == UsageEventKind.RESERVATION.value:
            self.record_reservation(
                provider=provider,
                deployment=deployment,
                outcome="accepted",
            )
        elif kind == UsageEventKind.EXPIRY_RECOVERY.value:
            self.record_expiry(
                provider=provider,
                deployment=deployment,
                reason="expiry_reclamation",
            )
        elif kind == UsageEventKind.CORRECTION.value:
            self.record_reconciliation(
                "correction", provider=provider, deployment=deployment
            )
        elif kind in {
            UsageEventKind.OBSERVATION_SUCCESS.value,
            UsageEventKind.OBSERVATION_FAILURE.value,
        }:
            self.record_reconciliation(
                "import", provider=provider, deployment=deployment
            )
            # Estimate error when both estimate and observation units present
            # is computed by record_estimate_error from the control/router path.
        # Reset-like reason codes
        if any("reset" in str(item).casefold() for item in reasons):
            self.record_reset(provider=provider, deployment=deployment, reason="reset")

    def record_reservation(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        outcome: str = "accepted",
    ) -> None:
        self._update(
            "usage_reservations_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "outcome": outcome,
            },
        )

    def record_denial(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "limit_exhausted",
    ) -> None:
        self._update(
            "usage_reservation_denials_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )
        self.record_reservation(
            provider=provider, deployment=deployment, outcome="denied"
        )

    def record_expiry(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "expiry_reclamation",
    ) -> None:
        self._update(
            "usage_reservation_expiry_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )
        self.record_reservation(
            provider=provider, deployment=deployment, outcome="expired"
        )

    def record_estimate_error(
        self,
        estimated: int,
        actual: int,
        *,
        provider: str = "other",
        deployment: str = "other",
        dimension: str = "total_tokens",
    ) -> None:
        """Record |estimate-actual|/max(actual,1) into sum/count for averages."""

        denom = max(1, abs(int(actual)))
        ratio = abs(int(estimated) - int(actual)) / float(denom)
        labels = {
            "provider": provider,
            "deployment": deployment,
            "dimension": dimension,
        }
        self._update("usage_estimate_error_ratio_sum", ratio, labels)
        self._update("usage_estimate_error_ratio_count", 1, labels)

    def record_headroom_snapshot(
        self,
        snapshot: UsageSnapshot | Mapping[str, Any],
        *,
        provider: str = "other",
        deployment: str = "other",
    ) -> None:
        if isinstance(snapshot, UsageSnapshot):
            headroom = snapshot.headroom
            state = snapshot.state
        else:
            headroom = snapshot.get("headroom") or ()
            state_raw = snapshot.get("state")
            try:
                state = AvailabilityState(str(state_raw))
            except Exception:
                state = AvailabilityState.UNKNOWN
        if state is AvailabilityState.STALE:
            self._update(
                "usage_stale_scopes",
                1,
                {"provider": provider, "state": "stale"},
                gauge=True,
            )
        if state is AvailabilityState.UNKNOWN:
            self._update(
                "usage_unknown_scopes",
                1,
                {"provider": provider, "state": "unknown"},
                gauge=True,
            )
        for item in headroom:
            if hasattr(item, "to_dict"):
                dim = item.dimension.value if hasattr(item.dimension, "value") else str(item.dimension)
                band = headroom_band(item.available, item.ceiling, state=item.state)
            elif isinstance(item, Mapping):
                dim = str(item.get("dimension") or "other")
                avail = (
                    Quantity.from_dict(item["available"])
                    if isinstance(item.get("available"), Mapping)
                    else None
                )
                ceil = (
                    Quantity.from_dict(item["ceiling"])
                    if isinstance(item.get("ceiling"), Mapping)
                    else None
                )
                try:
                    item_state = AvailabilityState(str(item.get("state")))
                except Exception:
                    item_state = None
                band = headroom_band(avail, ceil, state=item_state)
            else:
                continue
            self._update(
                "usage_headroom_band",
                1,
                {
                    "provider": provider,
                    "deployment": deployment,
                    "dimension": dim,
                    "band": band,
                },
                gauge=True,
            )

    def record_reset(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "reset",
    ) -> None:
        self._update(
            "usage_resets_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )

    def record_wait(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "wait",
    ) -> None:
        self._update(
            "usage_waits_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )

    def record_reroute(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "reroute",
    ) -> None:
        self._update(
            "usage_reroutes_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )

    def record_fallback(
        self,
        *,
        provider: str = "other",
        deployment: str = "other",
        reason: str = "fallback",
    ) -> None:
        self._update(
            "usage_fallbacks_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "reason": reason,
            },
        )

    def record_reconciliation(
        self,
        kind: str,
        *,
        provider: str = "other",
        deployment: str = "other",
    ) -> None:
        self._update(
            "usage_reconciliation_total",
            1,
            {
                "provider": provider,
                "deployment": deployment,
                "kind": kind,
            },
        )

    def record_store_health(self, *, healthy: bool) -> None:
        self._update(
            "usage_store_health",
            1.0 if healthy else 0.0,
            {"state": "healthy" if healthy else "unhealthy"},
            gauge=True,
        )

    def record_control_mutation(self, operation: str, success: bool) -> None:
        self._update(
            "usage_control_mutations_total",
            1,
            {
                "operation": operation,
                "outcome": "success" if success else "error",
            },
        )

    def ingest_document(
        self,
        document: Mapping[str, Any],
        *,
        provider_by_scope: Optional[Mapping[str, str]] = None,
        deployment_by_scope: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Recompute event-derived counters from a ledger document snapshot."""

        providers = dict(provider_by_scope or {})
        deployments = dict(deployment_by_scope or {})
        events = document.get("events") or []
        if not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, Mapping):
                continue
            scope_id = str(event.get("scope_id") or "")
            self.ingest_event(
                event,
                provider=providers.get(scope_id, "other"),
                deployment=deployments.get(scope_id, "other"),
            )
        # Store health from basic document shape.
        healthy = (
            isinstance(document.get("revision"), int)
            and isinstance(document.get("events"), list)
        )
        self.record_store_health(healthy=healthy)

    # -- export ------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            samples = [
                MetricSample(name=name, labels=labels, value=value).to_dict()
                for (name, labels), value in sorted(
                    self._values.items(),
                    key=lambda item: (item[0][0], item[0][1]),
                )
            ]
            return {
                "schema_version": USAGE_METRICS_SCHEMA_VERSION,
                "requirement_id": self.requirement_id,
                "series_count": len(samples),
                "max_series": self.max_series,
                "samples": samples,
            }

    def samples(self) -> Tuple[MetricSample, ...]:
        with self._lock:
            return tuple(
                MetricSample(name=name, labels=labels, value=value)
                for (name, labels), value in sorted(
                    self._values.items(),
                    key=lambda item: (item[0][0], item[0][1]),
                )
            )

    def metric_names(self) -> Tuple[str, ...]:
        return tuple(sorted(_METRIC_LABELS))

    def label_contract(self) -> Dict[str, Tuple[str, ...]]:
        return {name: labels for name, labels in _METRIC_LABELS.items()}


def forbidden_metric_label_keys() -> frozenset:
    return _FORBIDDEN_LABEL_KEYS


__all__ = [
    "USAGE_OBSERVABILITY_REQUIREMENT_ID",
    "USAGE_METRICS_SCHEMA_VERSION",
    "MAX_METRIC_SERIES",
    "MetricSample",
    "UsageObservability",
    "forbidden_metric_label_keys",
]
