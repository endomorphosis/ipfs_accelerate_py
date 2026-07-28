"""Endpoint-usage projection into fair resource admission (ASI-167).

This module is the declared ASI-167 surface for resource scheduling. ASREF
lands the historical :mod:`resource_scheduler` implementation under
``runtime.resource_scheduler``; this file projects exact endpoint usage
snapshots into compatible :class:`ProviderCapacity` records and applies the
conservative intersection of supervisor ancestor budgets, multi-window
endpoint headroom, concurrency/context, health/circuit/retry-after, deadline,
host constraints, and distributed lease policy.

Importing this module (or calling :func:`install_endpoint_usage_admission`)
installs the usage-aware APIs onto the runtime module so both historical and
package imports see the same symbols.  Default/off mode preserves existing
ordering, capacity, and admission semantics.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

from ipfs_accelerate_py.agent_supervisor.runtime import resource_scheduler as _runtime
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import *  # noqa: F403
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    UNKNOWN_LIMIT,
    AdmissionDecision,
    HostResourceSnapshot,
    LaneResourceRequirements,
    ProviderCapacity,
    ResourcePolicy,
    ResourceScheduler,
    normalize_provider_capacities,
    normalize_provider_capacity,
)

# ---------------------------------------------------------------------------
# Requirement / schema identities
# ---------------------------------------------------------------------------

ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID: str = (
    "requirement:endpoint-usage-fair-resource-admission.v1"
)
ENDPOINT_USAGE_ADMISSION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/endpoint-usage-admission@1"
)
USAGE_CAPACITY_UNAVAILABLE: str = "usage_capacity_unavailable"
LEGACY_UNBOUNDED_SENTINEL: int = UNKNOWN_LIMIT  # -1; never project unknown → unlimited


class UsageAdmissionMode(str, Enum):
    """Rollout mode for endpoint-usage projection into resource admission."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    ENFORCE = "enforce"


class UnknownStalePolicy(str, Enum):
    """How unknown or stale capacity fields are treated.

    Unknown/stale must never become unlimited via a legacy ``-1`` projection.
    """

    FAIL_CLOSED = "fail_closed"
    CONSERVATIVE_ZERO = "conservative_zero"
    IGNORE_IN_OFF = "ignore_in_off"


class UsageAdmissionAction(str, Enum):
    """Scheduler choice when usage capacity is considered."""

    ADMIT = "admit"
    ROUTE = "route"
    WAIT = "wait"
    FALLBACK = "fallback"
    DENY = "deny"


class EndpointSnapshotProtocol(Protocol):
    """Minimal injected view of an endpoint usage snapshot."""

    scope_id: str
    usage_revision: Any
    observed_at: Any
    fresh_until: Any
    state: Any
    headroom: Any
    reservations: Any
    next_eligible_at: Any
    reason_codes: Any


class UsageCoordinatorProtocol(Protocol):
    """Optional injected coordinator for single-flight refresh / reserve."""

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...

    def reserve(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _as_mode(value: Any) -> UsageAdmissionMode:
    if isinstance(value, UsageAdmissionMode):
        return value
    text = str(value or UsageAdmissionMode.OFF.value).strip().lower()
    try:
        return UsageAdmissionMode(text)
    except ValueError as exc:
        raise ValueError(f"unknown usage admission mode: {value!r}") from exc


def _as_unknown_policy(value: Any) -> UnknownStalePolicy:
    if isinstance(value, UnknownStalePolicy):
        return value
    text = str(value or UnknownStalePolicy.FAIL_CLOSED.value).strip().lower()
    try:
        return UnknownStalePolicy(text)
    except ValueError as exc:
        raise ValueError(f"unknown stale-field policy: {value!r}") from exc


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _quantity_units(quantity: Any, *, unknown_policy: UnknownStalePolicy) -> int | None:
    """Project a Quantity-like value to an int limit.

    Returns:
      * non-negative int for finite amounts
      * ``None`` for unknown (caller must apply mode policy; never treat as unlimited)
      * ``LEGACY_UNBOUNDED_SENTINEL`` only for *explicit* unlimited
    """

    if quantity is None:
        return None
    kind = _enum_value(getattr(quantity, "kind", None) or (quantity.get("kind") if isinstance(quantity, Mapping) else None))
    if kind in {"", "unknown"}:
        return None
    if kind == "unlimited":
        return LEGACY_UNBOUNDED_SENTINEL
    if kind == "finite":
        raw = getattr(quantity, "value", None)
        if raw is None and isinstance(quantity, Mapping):
            raw = quantity.get("value")
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value
    # Mapping without kind: treat numeric as finite, -1 as unknown (not unlimited).
    if isinstance(quantity, Mapping):
        if "value" in quantity and quantity.get("kind") is None:
            try:
                value = int(quantity["value"])
            except (TypeError, ValueError):
                return None
            if value < 0:
                # Legacy -1 in unknown projection must not become unlimited.
                return None
            return value
    if isinstance(quantity, bool):
        return None
    if isinstance(quantity, int):
        if quantity < 0:
            return None
        return quantity
    return None


def _resolve_unknown(
    units: int | None,
    *,
    mode: UsageAdmissionMode,
    policy: UnknownStalePolicy,
    field_name: str,
) -> tuple[int, list[str]]:
    """Map unknown capacity to a concrete admission limit under mode policy."""

    if units is not None:
        return units, []
    reasons: list[str] = [f"unknown_{field_name}"]
    if mode is UsageAdmissionMode.OFF:
        # Off mode ignores usage projection entirely upstream; if called, keep legacy.
        return LEGACY_UNBOUNDED_SENTINEL, []
    if policy is UnknownStalePolicy.IGNORE_IN_OFF and mode is UsageAdmissionMode.OFF:
        return LEGACY_UNBOUNDED_SENTINEL, []
    if mode in {UsageAdmissionMode.OBSERVE, UsageAdmissionMode.SHADOW} and policy is UnknownStalePolicy.IGNORE_IN_OFF:
        return LEGACY_UNBOUNDED_SENTINEL, reasons
    # fail_closed / conservative_zero: never unlimited
    if policy is UnknownStalePolicy.FAIL_CLOSED and mode is UsageAdmissionMode.ENFORCE:
        return 0, reasons + ["fail_closed_unknown"]
    return 0, reasons + ["conservative_zero_unknown"]


def _headroom_entries(snapshot: Any) -> tuple[Any, ...]:
    if snapshot is None:
        return ()
    raw = getattr(snapshot, "headroom", None)
    if raw is None and isinstance(snapshot, Mapping):
        raw = snapshot.get("headroom")
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        return ()
    try:
        return tuple(raw)
    except TypeError:
        return ()


def _dimension_name(entry: Any) -> str:
    dim = getattr(entry, "dimension", None)
    if dim is None and isinstance(entry, Mapping):
        dim = entry.get("dimension")
    return _enum_value(dim)


def _entry_available(entry: Any) -> Any:
    if hasattr(entry, "available"):
        return entry.available
    if isinstance(entry, Mapping):
        return entry.get("available")
    return None


def _entry_reserved(entry: Any) -> Any:
    if hasattr(entry, "reserved"):
        return entry.reserved
    if isinstance(entry, Mapping):
        return entry.get("reserved")
    return None


def _entry_state(entry: Any) -> str:
    state = getattr(entry, "state", None)
    if state is None and isinstance(entry, Mapping):
        state = entry.get("state")
    return _enum_value(state)


def _entry_next_eligible(entry: Any) -> str:
    value = getattr(entry, "next_eligible_at", None)
    if value is None and isinstance(entry, Mapping):
        value = entry.get("next_eligible_at")
    return str(value or "").strip()


def _snapshot_field(snapshot: Any, name: str, default: Any = None) -> Any:
    if snapshot is None:
        return default
    if hasattr(snapshot, name):
        return getattr(snapshot, name)
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, default)
    return default


def _parse_time_ms(value: Any, *, now_ms: int) -> int | None:
    """Parse RFC3339 / epoch-ms / relative ms into absolute epoch ms when possible."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        # Heuristic: small values are relative ms offsets.
        if number < 10_000_000_000:
            return now_ms + max(0, number)
        return number
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_time_ms(int(text), now_ms=now_ms)
    # RFC3339-ish: rely on time.strptime for common Zulu form without extra deps.
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            cleaned = text.replace("Z", "+0000") if "%z" in fmt and text.endswith("Z") else text
            if fmt.endswith("Z"):
                cleaned = text
            parsed = time.strptime(cleaned if not fmt.endswith("%z") else text.replace("Z", "+0000"), fmt.replace("Z", ""))
            return int(time.mktime(parsed) * 1000)
        except (ValueError, OverflowError, OSError):
            continue
    return None


def _min_finite(*values: int) -> int:
    finite = [v for v in values if v >= 0]
    if not finite:
        return LEGACY_UNBOUNDED_SENTINEL
    return min(finite)


def _intersect_capacity(left: int, right: int) -> int:
    """Conservative intersection of two capacity fields (-1 = unbounded)."""

    if left < 0 and right < 0:
        return LEGACY_UNBOUNDED_SENTINEL
    if left < 0:
        return max(0, right)
    if right < 0:
        return max(0, left)
    return max(0, min(left, right))


# ---------------------------------------------------------------------------
# Projection: endpoint snapshot → ProviderCapacity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointCapacityProjection:
    """Compatibility projection of an endpoint snapshot into ProviderCapacity."""

    provider_id: str
    capacity: ProviderCapacity
    scope_id: str = ""
    usage_revision: str = ""
    freshness_state: str = "unknown"
    next_eligible_at: str = ""
    next_eligible_at_ms: int = 0
    reason_codes: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    stale: bool = False
    mode: str = UsageAdmissionMode.OFF.value
    projection_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ENDPOINT_USAGE_ADMISSION_SCHEMA,
            "requirement_id": ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID,
            "provider_id": self.provider_id,
            "capacity": self.capacity.to_dict(),
            "scope_id": self.scope_id,
            "usage_revision": self.usage_revision,
            "freshness_state": self.freshness_state,
            "next_eligible_at": self.next_eligible_at,
            "next_eligible_at_ms": self.next_eligible_at_ms,
            "reason_codes": list(self.reason_codes),
            "unknown_fields": list(self.unknown_fields),
            "stale": self.stale,
            "mode": self.mode,
            "projection_id": self.projection_id,
        }


def project_provider_capacity_from_usage_snapshot(
    snapshot: Any,
    *,
    provider_id: str,
    base: ProviderCapacity | Mapping[str, Any] | None = None,
    mode: UsageAdmissionMode | str = UsageAdmissionMode.ENFORCE,
    unknown_policy: UnknownStalePolicy | str = UnknownStalePolicy.FAIL_CLOSED,
    now_ms: int | None = None,
    observed_at_ms: int | None = None,
    retry_after_ms: int | None = None,
    circuit_open: bool = False,
) -> EndpointCapacityProjection:
    """Project an endpoint usage snapshot into a compatibility ProviderCapacity.

    Effective limits are the conservative intersection of base telemetry and
    endpoint multi-window headroom.  Unknown/stale fields follow
    ``unknown_policy`` and **cannot** become unlimited through a legacy ``-1``
    projection under assist/enforce.
    """

    mode_e = _as_mode(mode)
    policy_e = _as_unknown_policy(unknown_policy)
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    identity = str(provider_id or "").strip().lower()
    if not identity:
        raise ValueError("provider_id must be non-empty")

    if base is None:
        base_cap = ProviderCapacity(provider_id=identity)
    else:
        base_cap = normalize_provider_capacity(base, provider_id=identity)

    if mode_e is UsageAdmissionMode.OFF or snapshot is None:
        return EndpointCapacityProjection(
            provider_id=identity,
            capacity=base_cap,
            mode=mode_e.value,
            projection_id=_content_id(
                {"provider_id": identity, "mode": mode_e.value, "off": True}
            ),
        )

    scope_id = str(_snapshot_field(snapshot, "scope_id", "") or "")
    usage_revision = str(_snapshot_field(snapshot, "usage_revision", "") or "")
    state = _enum_value(_snapshot_field(snapshot, "state", "unknown"))
    reason_codes: list[str] = []
    snap_reasons = _snapshot_field(snapshot, "reason_codes", ()) or ()
    try:
        reason_codes.extend(str(item) for item in snap_reasons if str(item))
    except TypeError:
        pass

    stale = state == "stale"
    fresh_until = _snapshot_field(snapshot, "fresh_until")
    fresh_until_ms = _parse_time_ms(fresh_until, now_ms=now)
    if fresh_until_ms is not None and fresh_until_ms < now:
        stale = True
        reason_codes.append("snapshot_expired")

    unknown_fields: list[str] = []

    # Aggregate headroom by dimension.
    request_units: int | None = None
    token_units: int | None = None
    concurrent_units: int | None = None
    next_eligible = str(_snapshot_field(snapshot, "next_eligible_at", "") or "")
    next_eligible_ms = _parse_time_ms(next_eligible, now_ms=now) or 0

    for entry in _headroom_entries(snapshot):
        dim = _dimension_name(entry)
        available = _quantity_units(_entry_available(entry), unknown_policy=policy_e)
        reserved = _quantity_units(_entry_reserved(entry), unknown_policy=policy_e)
        entry_state = _entry_state(entry)
        if entry_state in {"exhausted", "cooling_down", "disabled", "unroutable"}:
            reason_codes.append(f"headroom_{dim}_{entry_state}")
        if available is None:
            unknown_fields.append(dim)
        # Subtract active reservations conservatively when both finite.
        effective: int | None
        if available is None:
            effective = None
        elif available < 0:
            effective = LEGACY_UNBOUNDED_SENTINEL
        else:
            reserved_units = 0 if reserved is None or reserved < 0 else reserved
            effective = max(0, available - reserved_units)

        if dim in {"requests", "batch_items"}:
            if request_units is None:
                request_units = effective
            elif effective is not None:
                request_units = _intersect_capacity(request_units, effective)
        elif dim in {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "embedding_tokens",
        }:
            if token_units is None:
                token_units = effective
            elif effective is not None:
                token_units = _intersect_capacity(token_units, effective)
        elif dim in {"concurrent_requests", "concurrent_streams"}:
            if concurrent_units is None:
                concurrent_units = effective
            elif effective is not None:
                concurrent_units = _intersect_capacity(concurrent_units, effective)
        entry_next = _entry_next_eligible(entry)
        entry_next_ms = _parse_time_ms(entry_next, now_ms=now)
        if entry_next_ms is not None and (next_eligible_ms == 0 or entry_next_ms < next_eligible_ms):
            next_eligible_ms = entry_next_ms
            next_eligible = entry_next

    # Apply unknown policy (never unlimited under enforce/assist).
    quota, quota_reasons = _resolve_unknown(
        request_units, mode=mode_e, policy=policy_e, field_name="quota_remaining"
    )
    tokens, token_reasons = _resolve_unknown(
        token_units, mode=mode_e, policy=policy_e, field_name="token_budget_remaining"
    )
    concurrent, concurrent_reasons = _resolve_unknown(
        concurrent_units, mode=mode_e, policy=policy_e, field_name="max_concurrency"
    )
    reason_codes.extend(quota_reasons)
    reason_codes.extend(token_reasons)
    reason_codes.extend(concurrent_reasons)

    if stale:
        reason_codes.append("stale_snapshot")
        if mode_e is UsageAdmissionMode.ENFORCE and policy_e is UnknownStalePolicy.FAIL_CLOSED:
            quota = 0
            tokens = 0
            concurrent = 0
            reason_codes.append("fail_closed_stale")

    if state in {"exhausted", "disabled", "unroutable"}:
        quota = 0
        concurrent = 0
        reason_codes.append(f"snapshot_{state}")
    if state == "cooling_down" and next_eligible_ms > now:
        reason_codes.append("cooling_down")

    healthy = bool(base_cap.healthy)
    if circuit_open or state in {"disabled", "unroutable", "exhausted"}:
        healthy = False
        reason_codes.append("circuit_or_unavailable")

    effective_retry = max(
        0,
        int(retry_after_ms if retry_after_ms is not None else base_cap.retry_after_ms),
    )
    if next_eligible_ms > now:
        wait_ms = next_eligible_ms - now
        effective_retry = max(effective_retry, wait_ms)

    # Intersect with base telemetry (base -1 means "not reported" for quota/tokens).
    projected_quota = _intersect_capacity(base_cap.quota_remaining, quota)
    projected_tokens = _intersect_capacity(base_cap.token_budget_remaining, tokens)
    # Concurrency: base always has a concrete max_concurrency (>=0).
    if concurrent < 0:
        projected_max = base_cap.max_concurrency
    else:
        projected_max = min(base_cap.max_concurrency, concurrent) if base_cap.max_concurrency else concurrent
        projected_max = max(0, int(projected_max))

    # Under assist/enforce, never leave projected finite fields as "unknown upgraded to -1".
    if mode_e in {UsageAdmissionMode.ASSIST, UsageAdmissionMode.ENFORCE}:
        if request_units is None and projected_quota < 0:
            projected_quota = 0
            reason_codes.append("blocked_legacy_unlimited_quota")
        if token_units is None and projected_tokens < 0:
            projected_tokens = 0
            reason_codes.append("blocked_legacy_unlimited_tokens")

    observed = int(observed_at_ms if observed_at_ms is not None else base_cap.observed_at_ms or now)
    capacity = ProviderCapacity(
        provider_id=identity,
        healthy=healthy,
        quota_remaining=projected_quota,
        latency_ms=base_cap.latency_ms,
        context_window_tokens=base_cap.context_window_tokens,
        token_budget_remaining=projected_tokens,
        max_concurrency=max(0, projected_max),
        active_requests=base_cap.active_requests,
        capabilities=base_cap.capabilities,
        observed_at_ms=observed,
        retry_after_ms=effective_retry,
    )
    projection_id = _content_id(
        {
            "provider_id": identity,
            "scope_id": scope_id,
            "usage_revision": usage_revision,
            "quota": projected_quota,
            "tokens": projected_tokens,
            "max_concurrency": capacity.max_concurrency,
            "mode": mode_e.value,
        }
    )
    return EndpointCapacityProjection(
        provider_id=identity,
        capacity=capacity,
        scope_id=scope_id,
        usage_revision=usage_revision,
        freshness_state=state or ("stale" if stale else "unknown"),
        next_eligible_at=next_eligible,
        next_eligible_at_ms=int(next_eligible_ms or 0),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        unknown_fields=tuple(dict.fromkeys(unknown_fields)),
        stale=stale,
        mode=mode_e.value,
        projection_id=projection_id,
    )


# ---------------------------------------------------------------------------
# Hierarchical budget intersection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalBudgetLimit:
    """One typed budget limit at a supervisor scope level."""

    dimension: str
    remaining: int
    window: str = "lifetime"
    currency: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", str(self.dimension or "").strip().lower())
        if not self.dimension:
            raise ValueError("budget dimension must be non-empty")
        if isinstance(self.remaining, bool) or not isinstance(self.remaining, int):
            raise ValueError("budget remaining must be an int")
        if self.remaining < -1:
            raise ValueError("budget remaining must be -1 or non-negative")
        object.__setattr__(self, "window", str(self.window or "lifetime").strip().lower())
        object.__setattr__(self, "currency", str(self.currency or "").strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "remaining": self.remaining,
            "window": self.window,
            "currency": self.currency,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HierarchicalBudgetLimit":
        return cls(
            dimension=str(value.get("dimension") or value.get("name") or ""),
            remaining=int(value.get("remaining", value.get("limit", 0))),
            window=str(value.get("window") or "lifetime"),
            currency=str(value.get("currency") or ""),
        )


@dataclass(frozen=True)
class HierarchicalBudgetView:
    """Ancestor budget chain: child may only lower parent remaining."""

    limits: tuple[HierarchicalBudgetLimit, ...] = ()
    scope_ids: tuple[str, ...] = ()

    def effective_remaining(self, dimension: str, *, currency: str = "") -> int:
        dim = str(dimension or "").strip().lower()
        cur = str(currency or "").strip().lower()
        matched = [
            item.remaining
            for item in self.limits
            if item.dimension == dim and item.currency == cur
        ]
        if not matched:
            return LEGACY_UNBOUNDED_SENTINEL
        finite = [v for v in matched if v >= 0]
        if not finite:
            return LEGACY_UNBOUNDED_SENTINEL
        return min(finite)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": [item.to_dict() for item in self.limits],
            "scope_ids": list(self.scope_ids),
        }

    @classmethod
    def from_value(cls, value: Any) -> "HierarchicalBudgetView":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            limits_raw = value.get("limits") or value.get("budget_limits") or ()
            scopes = value.get("scope_ids") or value.get("scopes") or ()
            limits = tuple(
                item
                if isinstance(item, HierarchicalBudgetLimit)
                else HierarchicalBudgetLimit.from_mapping(item)
                for item in limits_raw
            )
            return cls(
                limits=limits,
                scope_ids=tuple(str(s) for s in scopes if str(s).strip()),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            limits = tuple(
                item
                if isinstance(item, HierarchicalBudgetLimit)
                else HierarchicalBudgetLimit.from_mapping(item)
                for item in value
            )
            return cls(limits=limits)
        raise TypeError("hierarchical budget must be a mapping or sequence")


def intersect_with_ancestor_budgets(
    capacity: ProviderCapacity,
    budget: HierarchicalBudgetView | Mapping[str, Any] | None,
) -> ProviderCapacity:
    """Lower capacity by nested supervisor ancestor budgets (never raise)."""

    view = HierarchicalBudgetView.from_value(budget)
    if not view.limits:
        return capacity
    quota = view.effective_remaining("requests")
    tokens = view.effective_remaining("total_tokens")
    if tokens < 0:
        tokens = view.effective_remaining("input_tokens")
    concurrent = view.effective_remaining("concurrent_requests")
    return replace(
        capacity,
        quota_remaining=_intersect_capacity(capacity.quota_remaining, quota),
        token_budget_remaining=_intersect_capacity(capacity.token_budget_remaining, tokens),
        max_concurrency=(
            capacity.max_concurrency
            if concurrent < 0
            else max(0, min(capacity.max_concurrency, concurrent))
        ),
    )


# ---------------------------------------------------------------------------
# Weighted fair queue + per-scope reserves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FairQueueScope:
    """One fairness scope (tenant / goal / task / lane) with weight and reserve."""

    scope_id: str
    kind: str = "lane"  # tenant|goal|task|lane
    weight: int = 1
    reserved_slots: int = 0
    reserved_quota: int = 0
    reserved_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", str(self.scope_id or "").strip())
        if not self.scope_id:
            raise ValueError("fair queue scope_id must be non-empty")
        object.__setattr__(self, "kind", str(self.kind or "lane").strip().lower())
        for name in ("weight", "reserved_slots", "reserved_quota", "reserved_tokens"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name == "weight" and value == 0:
                raise ValueError("weight must be positive")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightedFairQueue:
    """Deficit-weighted fair queue with per-scope reserves against starvation."""

    scopes: dict[str, FairQueueScope] = field(default_factory=dict)
    deficits: dict[str, int] = field(default_factory=dict)
    served: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def register(self, scope: FairQueueScope | Mapping[str, Any]) -> FairQueueScope:
        item = scope if isinstance(scope, FairQueueScope) else FairQueueScope(**dict(scope))
        with self._lock:
            self.scopes[item.scope_id] = item
            self.deficits.setdefault(item.scope_id, 0)
            self.served.setdefault(item.scope_id, 0)
        return item

    def reserved_total(self, attribute: str) -> int:
        with self._lock:
            return sum(int(getattr(scope, attribute)) for scope in self.scopes.values())

    def available_for_scope(
        self,
        scope_id: str,
        *,
        total_slots: int,
        active_by_scope: Mapping[str, int] | None = None,
    ) -> int:
        """Slots a scope may still take without consuming others' reserves.

        Other scopes' ``reserved_slots`` are withheld from the shared pool so a
        heavy tenant/goal/lane cannot exhaust a shared account window.
        """

        active = dict(active_by_scope or {})
        total = max(0, int(total_slots))
        with self._lock:
            scope = self.scopes.get(scope_id)
            reserved_others = sum(
                s.reserved_slots
                for sid, s in self.scopes.items()
                if sid != scope_id
            )
            used_self = int(active.get(scope_id, 0))
            used_others = sum(v for key, v in active.items() if key != scope_id)
            free_global = max(0, total - used_self - used_others)
            # Pool available to this scope after protecting foreign reserves.
            shared_room = max(0, total - reserved_others - used_self)
            if scope is None:
                return min(free_global, shared_room)
            # Own reserve is always usable up to reserved_slots, even under pressure.
            own_reserve_room = max(0, scope.reserved_slots - used_self)
            return max(own_reserve_room, min(free_global, shared_room))

    def select_next(
        self,
        waiting_scope_ids: Sequence[str],
        *,
        total_slots: int = 1,
        active_by_scope: Mapping[str, int] | None = None,
    ) -> str | None:
        """Pick the next scope using deficit weights; respects reserves."""

        if not waiting_scope_ids:
            return None
        active = dict(active_by_scope or {})
        with self._lock:
            candidates: list[str] = []
            for scope_id in waiting_scope_ids:
                sid = str(scope_id)
                if self.available_for_scope(sid, total_slots=total_slots, active_by_scope=active) <= 0:
                    continue
                candidates.append(sid)
                if sid not in self.scopes:
                    self.scopes[sid] = FairQueueScope(scope_id=sid)
                    self.deficits.setdefault(sid, 0)
                    self.served.setdefault(sid, 0)
                self.deficits[sid] = self.deficits.get(sid, 0) + self.scopes[sid].weight
            if not candidates:
                # Fall back to pure weight order so work is not stuck when all
                # appear over reserve (still deterministic).
                candidates = [str(s) for s in waiting_scope_ids]
                for sid in candidates:
                    if sid not in self.scopes:
                        self.scopes[sid] = FairQueueScope(scope_id=sid)
                        self.deficits.setdefault(sid, 0)
                    self.deficits[sid] = self.deficits.get(sid, 0) + self.scopes[sid].weight
            best = max(candidates, key=lambda sid: (self.deficits.get(sid, 0), -self.served.get(sid, 0), sid))
            self.deficits[best] = max(0, self.deficits.get(best, 0) - 1)
            self.served[best] = self.served.get(best, 0) + 1
            return best

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scopes": {k: v.to_dict() for k, v in sorted(self.scopes.items())},
                "deficits": {k: int(v) for k, v in sorted(self.deficits.items())},
                "served": {k: int(v) for k, v in sorted(self.served.items())},
            }


# ---------------------------------------------------------------------------
# Single-flight refresh + reset event wakeup
# ---------------------------------------------------------------------------


class SingleFlightRefresh:
    """Prevent thundering-herd snapshot refresh for one key."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, BaseException] = {}

    def do(self, key: str, factory: Callable[[], Any]) -> Any:
        token = str(key or "").strip() or "_"
        leader = False
        with self._lock:
            event = self._inflight.get(token)
            if event is None:
                event = threading.Event()
                self._inflight[token] = event
                leader = True
        if leader:
            try:
                result = factory()
                with self._lock:
                    self._results[token] = result
                return result
            except BaseException as exc:  # noqa: BLE001 - re-raise after fanout
                with self._lock:
                    self._errors[token] = exc
                raise
            finally:
                with self._lock:
                    self._inflight.pop(token, None)
                    event.set()
                    # Keep results briefly for late joiners.
        else:
            event.wait(timeout=30.0)
            with self._lock:
                if token in self._errors:
                    raise self._errors[token]
                return self._results.get(token)


@dataclass
class ResetEventCursor:
    """Bounded jittered wakeup for next-eligible reset/capacity events."""

    cursor_ms: int = 0
    max_wakeups: int = 64
    jitter_ms: int = 50
    _pending: deque[tuple[int, str]] = field(default_factory=deque, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def note_next_eligible(self, at_ms: int, key: str = "") -> None:
        when = max(0, int(at_ms))
        with self._lock:
            self._pending.append((when, str(key or "")))
            while len(self._pending) > self.max_wakeups:
                self._pending.popleft()

    def due(self, now_ms: int) -> tuple[str, ...]:
        now = int(now_ms)
        due_keys: list[str] = []
        with self._lock:
            remaining: deque[tuple[int, str]] = deque()
            for when, key in self._pending:
                # Apply deterministic-ish jitter bound (0..jitter_ms).
                jitter = self._rng.randint(0, max(0, self.jitter_ms)) if self.jitter_ms else 0
                if when + jitter <= now:
                    due_keys.append(key)
                    self.cursor_ms = max(self.cursor_ms, when)
                else:
                    remaining.append((when, key))
            self._pending = remaining
        return tuple(due_keys)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cursor_ms": self.cursor_ms,
                "max_wakeups": self.max_wakeups,
                "jitter_ms": self.jitter_ms,
                "pending": [[when, key] for when, key in self._pending],
            }


# ---------------------------------------------------------------------------
# Usage-aware admission decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageAwareAdmissionDecision:
    """Authoritative usage-aware admission result (operational, not completion)."""

    admitted: bool
    action: str
    lane_id: str = ""
    provider_id: str = ""
    reasons: tuple[str, ...] = ()
    resource_decision: AdmissionDecision | None = None
    projection: EndpointCapacityProjection | None = None
    next_eligible_at_ms: int = 0
    wait_ms: int = 0
    route_provider_id: str = ""
    fallback_authorized: bool = False
    mode: str = UsageAdmissionMode.OFF.value
    decision_id: str = ""
    backpressure: bool = False

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ENDPOINT_USAGE_ADMISSION_SCHEMA,
            "requirement_id": ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID,
            "admitted": self.admitted,
            "action": self.action,
            "lane_id": self.lane_id,
            "provider_id": self.provider_id,
            "reasons": list(self.reasons),
            "resource_decision": (
                self.resource_decision.to_dict() if self.resource_decision is not None else None
            ),
            "projection": self.projection.to_dict() if self.projection is not None else None,
            "next_eligible_at_ms": self.next_eligible_at_ms,
            "wait_ms": self.wait_ms,
            "route_provider_id": self.route_provider_id,
            "fallback_authorized": self.fallback_authorized,
            "mode": self.mode,
            "decision_id": self.decision_id,
            "backpressure": self.backpressure,
            "usage_capacity_unavailable": (
                not self.admitted and USAGE_CAPACITY_UNAVAILABLE in self.reasons
            ),
        }


def evaluate_usage_aware_admission(
    *,
    resource_decision: AdmissionDecision | None = None,
    projection: EndpointCapacityProjection | None = None,
    mode: UsageAdmissionMode | str = UsageAdmissionMode.ENFORCE,
    deadline_ms: int = 0,
    now_ms: int | None = None,
    alternate_providers: Sequence[str] = (),
    fallback_authorized: bool = False,
    distributed_lease_ok: bool = True,
    required_tokens: int = 0,
    required_quota_units: int = 1,
) -> UsageAwareAdmissionDecision:
    """Choose admit / route / wait / fallback / deny without weakening authority.

    Effective admission is the conservative intersection of the host/resource
    decision and endpoint projection.  Off/observe modes never change the
    resource decision's admit bit.
    """

    mode_e = _as_mode(mode)
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    lane_id = resource_decision.lane_id if resource_decision is not None else ""
    provider_id = (
        (projection.provider_id if projection is not None else "")
        or (resource_decision.provider_id if resource_decision is not None else "")
    )
    reasons: list[str] = []
    if resource_decision is not None:
        reasons.extend(resource_decision.reasons)

    if mode_e is UsageAdmissionMode.OFF:
        admitted = bool(resource_decision.admitted) if resource_decision is not None else True
        action = UsageAdmissionAction.ADMIT.value if admitted else UsageAdmissionAction.DENY.value
        return UsageAwareAdmissionDecision(
            admitted=admitted,
            action=action,
            lane_id=lane_id,
            provider_id=provider_id,
            reasons=tuple(reasons),
            resource_decision=resource_decision,
            projection=projection,
            mode=mode_e.value,
            decision_id=_content_id({"mode": "off", "lane_id": lane_id, "admitted": admitted}),
        )

    if not distributed_lease_ok:
        reasons.append("distributed_lease_unavailable")
        if mode_e is UsageAdmissionMode.ENFORCE:
            return UsageAwareAdmissionDecision(
                admitted=False,
                action=UsageAdmissionAction.DENY.value,
                lane_id=lane_id,
                provider_id=provider_id,
                reasons=tuple(dict.fromkeys(reasons + [USAGE_CAPACITY_UNAVAILABLE])),
                resource_decision=resource_decision,
                projection=projection,
                mode=mode_e.value,
                backpressure=True,
                decision_id=_content_id({"deny": "lease", "lane_id": lane_id}),
            )

    resource_ok = True if resource_decision is None else bool(resource_decision.admitted)
    usage_ok = True
    next_eligible = 0
    wait_ms = 0

    if projection is not None:
        reasons.extend(projection.reason_codes)
        next_eligible = int(projection.next_eligible_at_ms or 0)
        cap = projection.capacity
        if not cap.healthy:
            usage_ok = False
            reasons.append("provider_unhealthy")
        if cap.retry_after_ms > 0:
            usage_ok = False
            reasons.append("provider_backoff")
            wait_ms = max(wait_ms, cap.retry_after_ms)
        if cap.available_concurrency <= 0:
            usage_ok = False
            reasons.append("provider_concurrency")
        if cap.quota_remaining >= 0 and cap.quota_remaining < max(0, int(required_quota_units)):
            usage_ok = False
            reasons.append("provider_quota")
        if cap.token_budget_remaining >= 0 and cap.token_budget_remaining < max(0, int(required_tokens)):
            usage_ok = False
            reasons.append("provider_token_budget")
        if projection.stale and mode_e is UsageAdmissionMode.ENFORCE:
            usage_ok = False
            reasons.append("stale_snapshot")
        if next_eligible > now:
            wait_ms = max(wait_ms, next_eligible - now)

    # Observe/shadow: record usage reasons but do not override resource admit.
    if mode_e in {UsageAdmissionMode.OBSERVE, UsageAdmissionMode.SHADOW}:
        admitted = resource_ok
        action = UsageAdmissionAction.ADMIT.value if admitted else UsageAdmissionAction.DENY.value
        return UsageAwareAdmissionDecision(
            admitted=admitted,
            action=action,
            lane_id=lane_id,
            provider_id=provider_id,
            reasons=tuple(dict.fromkeys(reasons)),
            resource_decision=resource_decision,
            projection=projection,
            next_eligible_at_ms=next_eligible,
            wait_ms=wait_ms,
            mode=mode_e.value,
            decision_id=_content_id(
                {"mode": mode_e.value, "lane_id": lane_id, "admitted": admitted}
            ),
        )

    # Assist/enforce: conservative intersection.
    if resource_ok and usage_ok:
        return UsageAwareAdmissionDecision(
            admitted=True,
            action=UsageAdmissionAction.ADMIT.value,
            lane_id=lane_id,
            provider_id=provider_id,
            reasons=tuple(dict.fromkeys(reasons)),
            resource_decision=resource_decision,
            projection=projection,
            next_eligible_at_ms=next_eligible,
            wait_ms=0,
            mode=mode_e.value,
            decision_id=_content_id(
                {"mode": mode_e.value, "lane_id": lane_id, "admitted": True}
            ),
        )

    # Prefer alternate route when configured.
    alternates = [str(p).strip().lower() for p in alternate_providers if str(p).strip()]
    alternates = [p for p in alternates if p and p != provider_id]
    if alternates and mode_e in {UsageAdmissionMode.ASSIST, UsageAdmissionMode.ENFORCE}:
        return UsageAwareAdmissionDecision(
            admitted=False,
            action=UsageAdmissionAction.ROUTE.value,
            lane_id=lane_id,
            provider_id=provider_id,
            route_provider_id=alternates[0],
            reasons=tuple(dict.fromkeys(reasons + ["reroute_eligible"])),
            resource_decision=resource_decision,
            projection=projection,
            next_eligible_at_ms=next_eligible,
            wait_ms=wait_ms,
            mode=mode_e.value,
            decision_id=_content_id(
                {"mode": mode_e.value, "lane_id": lane_id, "route": alternates[0]}
            ),
        )

    # Bounded wait when next-eligible fits the deadline.
    if wait_ms > 0:
        deadline_ok = deadline_ms <= 0 or (now + wait_ms) <= deadline_ms
        if deadline_ok:
            return UsageAwareAdmissionDecision(
                admitted=False,
                action=UsageAdmissionAction.WAIT.value,
                lane_id=lane_id,
                provider_id=provider_id,
                reasons=tuple(dict.fromkeys(reasons + ["wait_next_eligible"])),
                resource_decision=resource_decision,
                projection=projection,
                next_eligible_at_ms=next_eligible,
                wait_ms=wait_ms,
                mode=mode_e.value,
                backpressure=True,
                decision_id=_content_id(
                    {"mode": mode_e.value, "lane_id": lane_id, "wait_ms": wait_ms}
                ),
            )

    if fallback_authorized:
        return UsageAwareAdmissionDecision(
            admitted=False,
            action=UsageAdmissionAction.FALLBACK.value,
            lane_id=lane_id,
            provider_id=provider_id,
            reasons=tuple(dict.fromkeys(reasons + ["authorized_fallback"])),
            resource_decision=resource_decision,
            projection=projection,
            next_eligible_at_ms=next_eligible,
            wait_ms=wait_ms,
            fallback_authorized=True,
            mode=mode_e.value,
            decision_id=_content_id(
                {"mode": mode_e.value, "lane_id": lane_id, "fallback": True}
            ),
        )

    return UsageAwareAdmissionDecision(
        admitted=False,
        action=UsageAdmissionAction.DENY.value,
        lane_id=lane_id,
        provider_id=provider_id,
        reasons=tuple(dict.fromkeys(reasons + [USAGE_CAPACITY_UNAVAILABLE])),
        resource_decision=resource_decision,
        projection=projection,
        next_eligible_at_ms=next_eligible,
        wait_ms=wait_ms,
        mode=mode_e.value,
        backpressure=True,
        decision_id=_content_id(
            {"mode": mode_e.value, "lane_id": lane_id, "deny": USAGE_CAPACITY_UNAVAILABLE}
        ),
    )


# ---------------------------------------------------------------------------
# Usage-aware ResourceScheduler wrapper
# ---------------------------------------------------------------------------


class UsageAwareResourceScheduler(ResourceScheduler):
    """ResourceScheduler with optional endpoint-usage projection and fairness."""

    def __init__(
        self,
        policy: ResourcePolicy | Mapping[str, Any] | None = None,
        *,
        host_sampler: Callable[..., HostResourceSnapshot] | None = None,
        usage_mode: UsageAdmissionMode | str = UsageAdmissionMode.OFF,
        unknown_policy: UnknownStalePolicy | str = UnknownStalePolicy.FAIL_CLOSED,
        usage_snapshot_supplier: Callable[[str], Any] | None = None,
        usage_coordinator: UsageCoordinatorProtocol | None = None,
        fair_queue: WeightedFairQueue | None = None,
        reset_cursor: ResetEventCursor | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"policy": policy}
        if host_sampler is not None:
            kwargs["host_sampler"] = host_sampler
        super().__init__(**kwargs)
        self.usage_mode = _as_mode(usage_mode)
        self.unknown_policy = _as_unknown_policy(unknown_policy)
        self.usage_snapshot_supplier = usage_snapshot_supplier
        self.usage_coordinator = usage_coordinator
        self.fair_queue = fair_queue or WeightedFairQueue()
        self.reset_cursor = reset_cursor or ResetEventCursor()
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._single_flight = SingleFlightRefresh()
        self._usage_lock = threading.RLock()

    def refresh_snapshot(self, provider_id: str) -> Any:
        """Single-flight refresh of an endpoint snapshot for ``provider_id``."""

        key = str(provider_id or "").strip().lower()

        def _load() -> Any:
            if self.usage_snapshot_supplier is not None:
                return self.usage_snapshot_supplier(key)
            coordinator = self.usage_coordinator
            if coordinator is not None and hasattr(coordinator, "snapshot"):
                return coordinator.snapshot(key)
            return None

        return self._single_flight.do(f"snapshot:{key}", _load)

    def project_provider(
        self,
        provider_id: str,
        *,
        base: ProviderCapacity | Mapping[str, Any] | None = None,
        snapshot: Any = None,
    ) -> EndpointCapacityProjection:
        snap = snapshot
        if snap is None and self.usage_mode is not UsageAdmissionMode.OFF:
            snap = self.refresh_snapshot(provider_id)
        return project_provider_capacity_from_usage_snapshot(
            snap,
            provider_id=provider_id,
            base=base,
            mode=self.usage_mode,
            unknown_policy=self.unknown_policy,
            now_ms=self._clock_ms(),
        )

    def evaluate_with_usage(
        self,
        requirement: LaneResourceRequirements | Mapping[str, Any],
        *,
        host: HostResourceSnapshot | Mapping[str, Any],
        providers: Mapping[str, Any] | Iterable[ProviderCapacity | Mapping[str, Any]] | None = None,
        usage_snapshots: Mapping[str, Any] | None = None,
        ancestor_budget: HierarchicalBudgetView | Mapping[str, Any] | None = None,
        deadline_ms: int = 0,
        alternate_providers: Sequence[str] = (),
        fallback_authorized: bool = False,
        distributed_lease_ok: bool = True,
        **evaluate_kwargs: Any,
    ) -> UsageAwareAdmissionDecision:
        """Evaluate host/provider capacity then intersect endpoint usage."""

        req = (
            requirement
            if isinstance(requirement, LaneResourceRequirements)
            else LaneResourceRequirements.from_mapping(requirement)
        )
        # Project providers when usage mode is active.
        projected_list: list[ProviderCapacity] = []
        projections: dict[str, EndpointCapacityProjection] = {}
        normalized = normalize_provider_capacities(providers)
        snapshots = dict(usage_snapshots or {})
        for provider in normalized:
            snap = snapshots.get(provider.provider_id)
            if snap is None and self.usage_mode is not UsageAdmissionMode.OFF:
                if self.usage_snapshot_supplier is not None:
                    snap = self.refresh_snapshot(provider.provider_id)
            projection = project_provider_capacity_from_usage_snapshot(
                snap,
                provider_id=provider.provider_id,
                base=provider,
                mode=self.usage_mode,
                unknown_policy=self.unknown_policy,
                now_ms=self._clock_ms(),
            )
            cap = intersect_with_ancestor_budgets(projection.capacity, ancestor_budget)
            if cap is not projection.capacity:
                projection = replace(projection, capacity=cap)
            projections[provider.provider_id] = projection
            projected_list.append(projection.capacity)
            if projection.next_eligible_at_ms:
                self.reset_cursor.note_next_eligible(
                    projection.next_eligible_at_ms, provider.provider_id
                )

        providers_for_eval: Any = projected_list if projected_list else providers
        resource_decision = self.evaluate(
            req,
            host=host,
            providers=providers_for_eval,
            **evaluate_kwargs,
        )
        projection = None
        if resource_decision.provider_id:
            projection = projections.get(resource_decision.provider_id)
        elif projected_list:
            # Use first projected when resource layer did not select.
            projection = projections.get(projected_list[0].provider_id)

        decision = evaluate_usage_aware_admission(
            resource_decision=resource_decision,
            projection=projection,
            mode=self.usage_mode,
            deadline_ms=deadline_ms,
            now_ms=self._clock_ms(),
            alternate_providers=alternate_providers,
            fallback_authorized=fallback_authorized,
            distributed_lease_ok=distributed_lease_ok,
            required_tokens=req.token_budget,
            required_quota_units=req.quota_units,
        )
        return decision

    def wake_due_resets(self) -> tuple[str, ...]:
        """Return provider keys whose next-eligible time has passed (jittered)."""

        return self.reset_cursor.due(self._clock_ms())


# ---------------------------------------------------------------------------
# Install onto runtime module
# ---------------------------------------------------------------------------

_INSTALL_LOCK = threading.RLock()
_INSTALLED = False

_EXPORTS: dict[str, Any] = {
    "ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID": ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID,
    "ENDPOINT_USAGE_ADMISSION_SCHEMA": ENDPOINT_USAGE_ADMISSION_SCHEMA,
    "USAGE_CAPACITY_UNAVAILABLE": USAGE_CAPACITY_UNAVAILABLE,
    "UsageAdmissionMode": UsageAdmissionMode,
    "UnknownStalePolicy": UnknownStalePolicy,
    "UsageAdmissionAction": UsageAdmissionAction,
    "EndpointCapacityProjection": EndpointCapacityProjection,
    "project_provider_capacity_from_usage_snapshot": project_provider_capacity_from_usage_snapshot,
    "HierarchicalBudgetLimit": HierarchicalBudgetLimit,
    "HierarchicalBudgetView": HierarchicalBudgetView,
    "intersect_with_ancestor_budgets": intersect_with_ancestor_budgets,
    "FairQueueScope": FairQueueScope,
    "WeightedFairQueue": WeightedFairQueue,
    "SingleFlightRefresh": SingleFlightRefresh,
    "ResetEventCursor": ResetEventCursor,
    "UsageAwareAdmissionDecision": UsageAwareAdmissionDecision,
    "evaluate_usage_aware_admission": evaluate_usage_aware_admission,
    "UsageAwareResourceScheduler": UsageAwareResourceScheduler,
}


def install_endpoint_usage_admission(module: Any | None = None) -> None:
    """Install ASI-167 symbols onto the runtime resource_scheduler module."""

    global _INSTALLED
    target = module if module is not None else _runtime
    with _INSTALL_LOCK:
        for name, value in _EXPORTS.items():
            setattr(target, name, value)
        # Extend __all__ when present.
        existing = list(getattr(target, "__all__", ()) or ())
        for name in _EXPORTS:
            if name not in existing:
                existing.append(name)
        try:
            target.__all__ = existing  # type: ignore[attr-defined]
        except Exception:
            pass
        _INSTALLED = True


install_endpoint_usage_admission()

# Ensure historical import path name also resolves to symbols when this file
# is loaded under a non-alias module name.
sys.modules.setdefault(
    "ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler", _runtime
)

__all__ = sorted(
    set(getattr(_runtime, "__all__", ()))
    | set(_EXPORTS)
    | {"install_endpoint_usage_admission"}
)
