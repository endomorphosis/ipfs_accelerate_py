"""Usage-aware candidate resolution over one catalog and one usage revision.

This module is pure planning: hard-filter then soft-rank eligible bindings
against immutable :class:`UsageSnapshot` material.  It never reserves capacity,
refreshes a catalog source, probes a provider, instantiates credentials, or
exposes raw endpoints.

Hard gates run before any soft ranking input is considered.  Unlike usage
dimensions remain a headroom vector; saturation percentages are never summed
into a fictional universal token score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from .identity import stable_id
from .schema import (
    MAX_CANDIDATES,
    MAX_LIMITS,
    AvailabilityState,
    DimensionHeadroom,
    Quantity,
    QuantityKind,
    ResolutionCandidate,
    RoutingMode,
    RoutingPolicy,
    UsageAwareResolution,
    UsageDimension,
    UsageErrorCode,
    UsageLimit,
    UsageSnapshot,
    UsageVector,
    UsageVectorEntry,
)

USAGE_AWARE_RESOLUTION_REQUIREMENT_ID = "requirement:usage-aware-resolution.v1"
USAGE_REVISION_OFF = "usage_off"
USAGE_REVISION_UNAVAILABLE = "usage_unavailable"

DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 256
MAX_EXPLANATION_INPUTS = 32

# Terminal hard-reject availability states under enforce/assist modes.
_HARD_UNAVAILABLE = frozenset(
    {
        AvailabilityState.EXHAUSTED,
        AvailabilityState.DISABLED,
        AvailabilityState.UNROUTABLE,
    }
)

_HARD_DENY_REASON = {
    AvailabilityState.EXHAUSTED: "limit_exhausted",
    AvailabilityState.DISABLED: "endpoint_disabled",
    AvailabilityState.UNROUTABLE: "endpoint_unroutable",
    AvailabilityState.COOLING_DOWN: "cooling_down",
    AvailabilityState.STALE: "stale_snapshot",
    AvailabilityState.UNKNOWN: "unknown_state",
}


class ResolutionError(ValueError):
    """Invalid usage-aware resolution request or planning input."""


class UsageServiceUnavailable(LookupError):
    """Raised when a usage facade is invoked without an injected service."""

    def __init__(self, message: str = "usage service is not configured") -> None:
        super().__init__(message)
        self.code = UsageErrorCode.CAPACITY_UNAVAILABLE.value


class RevisionMismatch(RuntimeError):
    """Catalog or usage revision changed between planning stages."""

    def __init__(
        self,
        message: str,
        *,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        kind: str = "usage",
    ) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.kind = kind
        self.code = UsageErrorCode.STALE_SNAPSHOT.value


class UnknownLimitPolicy(str, Enum):
    """How to treat unknown ceilings or missing usage state."""

    DENY = "deny"
    ALLOW_CONFIGURED = "allow_configured"
    OBSERVE = "observe"


class StaleSnapshotPolicy(str, Enum):
    """How to treat a snapshot past its ``fresh_until`` horizon."""

    DENY = "deny"
    ALLOW = "allow"
    OBSERVE = "observe"


class UsageSnapshotReader(Protocol):
    """Minimal read-only surface required by ModelManager usage facades."""

    def snapshot(self, scope_id: str) -> UsageSnapshot:
        """Return one immutable usage snapshot for *scope_id*."""


@dataclass(frozen=True)
class UsageLimitPage:
    """Bounded, paginated page of usage limits for one scope."""

    scope_id: str
    usage_revision: str
    items: Tuple[UsageLimit, ...]
    next_cursor: Optional[str] = None
    total: int = 0
    schema_version: str = "ai.endpoint_usage.limit_page.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "usage_revision": self.usage_revision,
            "items": [item.to_dict() for item in self.items],
            "next_cursor": self.next_cursor,
            "total": self.total,
        }


@dataclass(frozen=True)
class UsageRoutingRequest:
    """Planning inputs that overlay dynamic usage on a static catalog result.

    Catalog operation/modality/capability, authorization, pins, context,
    device/locality, and data-governance gates are applied by the static
    catalog resolver before this request is evaluated.  This type carries the
    remaining usage-plane gates and soft-ranking preferences.
    """

    required: UsageVector = field(default_factory=UsageVector)
    unknown_limit_policy: UnknownLimitPolicy = UnknownLimitPolicy.DENY
    stale_snapshot_policy: StaleSnapshotPolicy = StaleSnapshotPolicy.DENY
    preferred_binding_id: Optional[str] = None
    preferred_provider_id: Optional[str] = None
    preferred_scope_id: Optional[str] = None
    affinity_binding_id: Optional[str] = None
    media_bytes: Optional[int] = None
    images: Optional[int] = None
    audio_seconds: Optional[int] = None
    max_cost_micros: Optional[int] = None
    cost_currency: Optional[str] = None
    deadline_at: Optional[str] = None
    now: Optional[str] = None
    # Soft ranking signals supplied by the caller (never invent probes).
    health_by_binding: Mapping[str, bool] = field(default_factory=dict)
    circuit_open_by_binding: Mapping[str, bool] = field(default_factory=dict)
    latency_ms_by_binding: Mapping[str, int] = field(default_factory=dict)
    queue_delay_ms_by_binding: Mapping[str, int] = field(default_factory=dict)
    quality_preference_by_binding: Mapping[str, int] = field(default_factory=dict)
    locality_by_binding: Mapping[str, str] = field(default_factory=dict)
    # When True, missing snapshot for a binding is a hard reject under enforce.
    require_snapshot: bool = True
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = self.required
        if not isinstance(required, UsageVector):
            required = UsageVector.from_dict(required)
        object.__setattr__(self, "required", required)
        object.__setattr__(
            self,
            "unknown_limit_policy",
            _enum_like(self.unknown_limit_policy, UnknownLimitPolicy, "unknown_limit_policy"),
        )
        object.__setattr__(
            self,
            "stale_snapshot_policy",
            _enum_like(self.stale_snapshot_policy, StaleSnapshotPolicy, "stale_snapshot_policy"),
        )
        for name in (
            "preferred_binding_id",
            "preferred_provider_id",
            "preferred_scope_id",
            "affinity_binding_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ResolutionError("%s must be non-empty text when set" % name)
        for name in ("media_bytes", "images", "audio_seconds", "max_cost_micros"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ResolutionError("%s must be a non-negative integer" % name)
        if self.cost_currency is not None:
            if not isinstance(self.cost_currency, str) or len(self.cost_currency) != 3:
                raise ResolutionError("cost_currency must be a 3-letter ISO code")
            object.__setattr__(self, "cost_currency", self.cost_currency.upper())
        for name in ("deadline_at", "now"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ResolutionError("%s must be an RFC 3339 string" % name)
        if not isinstance(self.require_snapshot, bool):
            raise ResolutionError("require_snapshot must be a boolean")
        object.__setattr__(self, "health_by_binding", dict(self.health_by_binding or {}))
        object.__setattr__(
            self, "circuit_open_by_binding", dict(self.circuit_open_by_binding or {})
        )
        object.__setattr__(
            self, "latency_ms_by_binding", dict(self.latency_ms_by_binding or {})
        )
        object.__setattr__(
            self,
            "queue_delay_ms_by_binding",
            dict(self.queue_delay_ms_by_binding or {}),
        )
        object.__setattr__(
            self,
            "quality_preference_by_binding",
            dict(self.quality_preference_by_binding or {}),
        )
        object.__setattr__(
            self, "locality_by_binding", dict(self.locality_by_binding or {})
        )
        codes = tuple(self.reason_codes or ())
        if len(codes) > 32:
            raise ResolutionError("reason_codes exceeds maximum count")
        object.__setattr__(self, "reason_codes", codes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required.to_dict(),
            "unknown_limit_policy": self.unknown_limit_policy.value,
            "stale_snapshot_policy": self.stale_snapshot_policy.value,
            "preferred_binding_id": self.preferred_binding_id,
            "preferred_provider_id": self.preferred_provider_id,
            "preferred_scope_id": self.preferred_scope_id,
            "affinity_binding_id": self.affinity_binding_id,
            "media_bytes": self.media_bytes,
            "images": self.images,
            "audio_seconds": self.audio_seconds,
            "max_cost_micros": self.max_cost_micros,
            "cost_currency": self.cost_currency,
            "deadline_at": self.deadline_at,
            "now": self.now,
            "require_snapshot": self.require_snapshot,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "UsageRoutingRequest":
        if not isinstance(data, Mapping):
            raise ResolutionError("UsageRoutingRequest must be an object")
        return cls(**dict(data))


@dataclass(frozen=True)
class StaticCandidate:
    """Minimal static eligibility surface consumed by usage planning.

    Constructed from a catalog :class:`ResolutionCandidate` without retaining
    raw endpoint URIs or credential material.
    """

    binding_id: str
    provider_id: str
    model_id: Optional[str] = None
    deployment_id: Optional[str] = None
    scope_id: Optional[str] = None
    catalog_score: int = 0
    locality: Optional[str] = None
    authorized: Optional[bool] = None
    healthy: Optional[bool] = None
    routable: Optional[bool] = None
    configured: Optional[bool] = None
    labels: Mapping[str, str] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not self.binding_id:
            raise ResolutionError("binding_id is required")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ResolutionError("provider_id is required")
        object.__setattr__(self, "labels", dict(self.labels or {}))
        object.__setattr__(self, "reasons", tuple(self.reasons or ()))


def _enum_like(value: Any, enum_type: type, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ResolutionError("unknown %s: %r" % (field_name, value)) from exc
    raise ResolutionError("%s must be a string or enum" % field_name)


def _parse_rfc3339(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(
            raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        )
    except ValueError as exc:
        raise ResolutionError("timestamp is not RFC 3339: %r" % value) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResolutionError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _now(request: UsageRoutingRequest) -> datetime:
    if request.now is not None:
        parsed = _parse_rfc3339(request.now)
        if parsed is None:
            raise ResolutionError("now must be a valid RFC 3339 timestamp")
        return parsed
    # Planning may run without a wall clock; treat missing now as epoch for
    # pure/deterministic tests.  Callers that care about freshness should set
    # ``now`` or inject snapshots that already encode state.
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def composite_usage_revision(
    snapshots: Sequence[UsageSnapshot],
    *,
    mode: RoutingMode = RoutingMode.ENFORCE,
) -> str:
    """Bind every consulted snapshot into one stable usage revision identity.

    Concurrent snapshot changes surface as a different composite revision
    rather than silently mixing counters from different materializations.
    """

    if mode is RoutingMode.OFF:
        return USAGE_REVISION_OFF
    if not snapshots:
        return USAGE_REVISION_UNAVAILABLE
    material = [
        {
            "scope_id": snap.scope_id,
            "usage_revision": snap.usage_revision,
            "state": snap.state.value if isinstance(snap.state, AvailabilityState) else str(snap.state),
            "observed_at": snap.observed_at,
        }
        for snap in sorted(snapshots, key=lambda item: item.scope_id or "")
    ]
    return stable_id("urev", material)


def headroom_index(
    headroom: Sequence[DimensionHeadroom],
) -> Dict[Tuple[str, Optional[str]], DimensionHeadroom]:
    """Index headroom entries by (dimension, currency)."""

    index: Dict[Tuple[str, Optional[str]], DimensionHeadroom] = {}
    for item in headroom:
        key = (item.dimension.value, item.currency)
        index[key] = item
    return index


def saturation_micros(available: Quantity, ceiling: Quantity) -> Optional[int]:
    """Return used/ceiling in micros for finite quantities; else None.

    Never mixes dimensions — callers compare saturations per dimension.
    """

    if (
        available.kind is not QuantityKind.FINITE
        or ceiling.kind is not QuantityKind.FINITE
        or available.value is None
        or ceiling.value is None
        or ceiling.value <= 0
    ):
        return None
    used = max(0, ceiling.value - available.value)
    return min(1_000_000, (used * 1_000_000) // ceiling.value)


def tightest_dimensions(
    headroom: Sequence[DimensionHeadroom],
    required: UsageVector,
) -> Tuple[str, ...]:
    """Return required dimensions ordered by highest saturation first.

    Dimensions with unknown saturation sort after known ones, then by name.
    """

    index = headroom_index(headroom)
    scored: List[Tuple[int, int, str]] = []
    for entry in required.entries:
        item = index.get((entry.dimension.value, entry.currency))
        if item is None:
            scored.append((2, 0, entry.dimension.value))
            continue
        sat = saturation_micros(item.available, item.ceiling)
        if sat is None:
            scored.append((1, 0, entry.dimension.value))
        else:
            scored.append((0, -sat, entry.dimension.value))
    scored.sort()
    # Dedupe while preserving order.
    seen = set()
    ordered: List[str] = []
    for _, __, name in scored:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def is_snapshot_stale(snapshot: UsageSnapshot, now: datetime) -> bool:
    """Return whether *snapshot* is past its freshness horizon at *now*."""

    if snapshot.state is AvailabilityState.STALE:
        return True
    fresh_until = _parse_rfc3339(snapshot.fresh_until)
    if fresh_until is None:
        return False
    return now > fresh_until


def _finite_need(entry: UsageVectorEntry) -> Optional[int]:
    if entry.amount.kind is QuantityKind.FINITE:
        return int(entry.amount.value or 0)
    if entry.amount.kind is QuantityKind.UNKNOWN:
        return None
    # Unlimited need is nonsensical for a request vector.
    return None


def _covers_need(available: Quantity, need: int) -> bool:
    if available.kind is QuantityKind.UNLIMITED:
        return True
    if available.kind is QuantityKind.UNKNOWN:
        return False
    return (available.value or 0) >= need


def hard_filter_candidate(
    candidate: StaticCandidate,
    snapshot: Optional[UsageSnapshot],
    request: UsageRoutingRequest,
    policy: RoutingPolicy,
    *,
    now: Optional[datetime] = None,
) -> Tuple[bool, Tuple[str, ...], Tuple[DimensionHeadroom, ...]]:
    """Apply hard usage/policy gates. Soft score is never consulted here.

    Returns ``(accepted, rejection_reasons, headroom)``.
    """

    clock = now if now is not None else _now(request)
    reasons: List[str] = []
    headroom: Tuple[DimensionHeadroom, ...] = ()

    # Authorization/state already applied by catalog; re-check explicit fails.
    if candidate.authorized is False:
        reasons.append("authorization_denied")
    if candidate.routable is False:
        reasons.append("endpoint_unroutable")
    if candidate.configured is False:
        reasons.append("endpoint_unconfigured")

    # Data governance / policy labels: deny when explicit deny labels present.
    labels = candidate.labels or {}
    for key, value in labels.items():
        lowered = "%s=%s" % (key.casefold(), str(value).casefold())
        if lowered in (
            "data.governance=deny",
            "policy.data=deny",
            "data_governance=deny",
            "policy.export=forbidden",
        ):
            reasons.append("data_governance_denied")
            break

    # Explicit pins already resolved by catalog; affinity is soft only.

    # Cost ceiling from routing policy and/or usage request.
    cost_ceiling = policy.cost_ceiling_micros
    if request.max_cost_micros is not None:
        cost_ceiling = (
            request.max_cost_micros
            if cost_ceiling is None
            else min(cost_ceiling, request.max_cost_micros)
        )
    cost_currency = policy.cost_currency or request.cost_currency

    # Media hard gates from the request (when specified).
    media_needs: List[UsageVectorEntry] = list(request.required.entries)
    if request.media_bytes is not None:
        media_needs.append(
            UsageVectorEntry(
                dimension=UsageDimension.MEDIA_BYTES,
                amount=Quantity.finite(request.media_bytes),
            )
        )
    if request.images is not None:
        media_needs.append(
            UsageVectorEntry(
                dimension=UsageDimension.IMAGES,
                amount=Quantity.finite(request.images),
            )
        )
    if request.audio_seconds is not None:
        media_needs.append(
            UsageVectorEntry(
                dimension=UsageDimension.AUDIO_SECONDS,
                amount=Quantity.finite(request.audio_seconds),
            )
        )

    if snapshot is None:
        if request.require_snapshot and policy.mode in (
            RoutingMode.ENFORCE,
            RoutingMode.ASSIST,
        ):
            reasons.append("missing_usage_snapshot")
        elif request.unknown_limit_policy is UnknownLimitPolicy.DENY and policy.mode in (
            RoutingMode.ENFORCE,
            RoutingMode.ASSIST,
            RoutingMode.SHADOW,
        ):
            reasons.append("unknown_state")
        # Observe/off may keep the candidate.
        return (not reasons, tuple(sorted(set(reasons))), headroom)

    headroom = tuple(snapshot.headroom)
    state = snapshot.state
    # OBSERVE never changes selection: record nothing as a hard reject for
    # usage-plane state.  SHADOW/ASSIST/ENFORCE apply hard gates.
    enforcing = policy.mode in (
        RoutingMode.ENFORCE,
        RoutingMode.ASSIST,
        RoutingMode.SHADOW,
    )

    if state in _HARD_UNAVAILABLE and enforcing:
        reasons.append(_HARD_DENY_REASON.get(state, "endpoint_unavailable"))

    if state is AvailabilityState.COOLING_DOWN and enforcing:
        reasons.append("cooling_down")

    stale = is_snapshot_stale(snapshot, clock)
    if (stale or state is AvailabilityState.STALE) and enforcing:
        if request.stale_snapshot_policy is StaleSnapshotPolicy.DENY:
            reasons.append("stale_snapshot")
        # ALLOW / OBSERVE policies keep the candidate.

    if state is AvailabilityState.UNKNOWN and enforcing:
        if request.unknown_limit_policy is UnknownLimitPolicy.DENY:
            reasons.append("unknown_state")
        elif request.unknown_limit_policy is UnknownLimitPolicy.ALLOW_CONFIGURED:
            # Require at least one configured finite hard limit.
            hard_limits = [
                lim
                for lim in snapshot.limits
                if lim.enforcement.value == "hard"
                and lim.ceiling.kind is QuantityKind.FINITE
            ]
            if not hard_limits and policy.mode in (
                RoutingMode.ENFORCE,
                RoutingMode.ASSIST,
            ):
                reasons.append("unknown_state")

    # Deadline: next eligible must not be after the caller deadline when wait
    # is disallowed; when wait is allowed it must fit max_wait_ms.
    deadline = _parse_rfc3339(request.deadline_at)
    next_eligible = _parse_rfc3339(snapshot.next_eligible_at)
    if enforcing and deadline is not None and next_eligible is not None and next_eligible > deadline:
        reasons.append("deadline_exceeded")
    if (
        enforcing
        and next_eligible is not None
        and policy.allow_wait
        and policy.max_wait_ms is not None
        and next_eligible > clock
    ):
        wait_ms = int((next_eligible - clock).total_seconds() * 1000)
        if wait_ms > int(policy.max_wait_ms):
            reasons.append("wait_exceeds_max")
    if (
        enforcing
        and next_eligible is not None
        and next_eligible > clock
        and not policy.allow_wait
    ):
        if state in (
            AvailabilityState.EXHAUSTED,
            AvailabilityState.COOLING_DOWN,
        ):
            reasons.append("wait_not_allowed")

    # Circuit open is a hard reject when reported under enforcing modes.
    if enforcing and request.circuit_open_by_binding.get(candidate.binding_id):
        reasons.append("circuit_open")

    # Required usage vector and media needs against headroom.
    index = headroom_index(headroom)
    for entry in media_needs:
        if not enforcing:
            break
        need = _finite_need(entry)
        if need is None:
            if entry.amount.kind is QuantityKind.UNKNOWN:
                if request.unknown_limit_policy is UnknownLimitPolicy.DENY:
                    reasons.append("unknown_required_%s" % entry.dimension.value)
            continue
        item = index.get((entry.dimension.value, entry.currency))
        if item is None:
            # No published headroom for a required dimension.
            if request.unknown_limit_policy is UnknownLimitPolicy.DENY:
                reasons.append("missing_headroom_%s" % entry.dimension.value)
            elif request.unknown_limit_policy is UnknownLimitPolicy.ALLOW_CONFIGURED:
                matching = [
                    lim
                    for lim in snapshot.limits
                    if lim.dimension is entry.dimension
                    and lim.ceiling.kind is QuantityKind.FINITE
                ]
                if not matching and policy.mode in (
                    RoutingMode.ENFORCE,
                    RoutingMode.ASSIST,
                ):
                    reasons.append("missing_headroom_%s" % entry.dimension.value)
            continue
        if item.state in _HARD_UNAVAILABLE:
            reasons.append("limit_exhausted")
            reasons.append(entry.dimension.value)
            continue
        if item.available.kind is QuantityKind.UNKNOWN:
            if request.unknown_limit_policy is UnknownLimitPolicy.DENY:
                reasons.append("unknown_headroom_%s" % entry.dimension.value)
            continue
        if not _covers_need(item.available, need):
            reasons.append("insufficient_headroom_%s" % entry.dimension.value)

    # Cost ceiling gate: projected cost against policy envelope.
    if enforcing and cost_ceiling is not None:
        cost_need = None
        for entry in request.required.entries:
            if entry.dimension is UsageDimension.COST_MICROS:
                cost_need = _finite_need(entry)
                break
        if cost_need is not None and cost_need > cost_ceiling:
            reasons.append("cost_ceiling_exceeded")
        cost_item = index.get((UsageDimension.COST_MICROS.value, cost_currency))
        if cost_item is not None and cost_need is not None:
            if not _covers_need(cost_item.available, cost_need):
                reasons.append("insufficient_headroom_cost_micros")

    # Prefer-local hard filter is not applied — locality is soft unless the
    # static catalog already filtered labels.

    unique = tuple(sorted(set(reasons)))
    return (not unique, unique, headroom)


def build_ranking_inputs(
    candidate: StaticCandidate,
    snapshot: Optional[UsageSnapshot],
    request: UsageRoutingRequest,
    policy: RoutingPolicy,
    headroom: Sequence[DimensionHeadroom],
) -> Tuple[Tuple[str, Union[int, float, str, bool, None]], ...]:
    """Expose bounded soft-ranking inputs without collapsing unlike units."""

    inputs: Dict[str, Union[int, float, str, bool, None]] = {}
    inputs["catalog_score"] = int(candidate.catalog_score)
    inputs["binding_priority"] = int(candidate.catalog_score)

    if request.affinity_binding_id and candidate.binding_id == request.affinity_binding_id:
        inputs["affinity"] = True
    else:
        inputs["affinity"] = False
    if request.preferred_binding_id and candidate.binding_id == request.preferred_binding_id:
        inputs["preferred_binding"] = True
    else:
        inputs["preferred_binding"] = False
    if request.preferred_provider_id and candidate.provider_id == request.preferred_provider_id:
        inputs["preferred_provider"] = True
    else:
        inputs["preferred_provider"] = False
    if request.preferred_scope_id and candidate.scope_id == request.preferred_scope_id:
        inputs["preferred_scope"] = True
    else:
        inputs["preferred_scope"] = False

    locality = request.locality_by_binding.get(candidate.binding_id) or candidate.locality
    if locality is not None:
        inputs["locality"] = str(locality)[:64]
        inputs["prefer_local_match"] = bool(
            policy.prefer_local and str(locality).casefold() == "local"
        )
    else:
        inputs["prefer_local_match"] = False

    health = request.health_by_binding.get(candidate.binding_id)
    if health is None:
        health = candidate.healthy
    if health is not None:
        inputs["health"] = bool(health)
    circuit_open = bool(request.circuit_open_by_binding.get(candidate.binding_id, False))
    inputs["circuit_open"] = circuit_open
    if candidate.binding_id in request.latency_ms_by_binding:
        inputs["latency_ms"] = int(request.latency_ms_by_binding[candidate.binding_id])
    if candidate.binding_id in request.queue_delay_ms_by_binding:
        inputs["queue_delay_ms"] = int(
            request.queue_delay_ms_by_binding[candidate.binding_id]
        )
    if candidate.binding_id in request.quality_preference_by_binding:
        inputs["quality_preference"] = int(
            request.quality_preference_by_binding[candidate.binding_id]
        )

    tightest = tightest_dimensions(headroom, request.required)
    if tightest:
        inputs["tightest_dimension"] = tightest[0]
        # Expose ordered tightest dimensions as a bounded scalar list via count
        # and primary/secondary names (avoid non-scalar values).
        inputs["tightest_dimension_count"] = len(tightest)
        if len(tightest) > 1:
            inputs["second_tightest_dimension"] = tightest[1]

    index = headroom_index(headroom)
    # Per-dimension saturation micros — never summed across dimensions.
    for entry in request.required.entries:
        item = index.get((entry.dimension.value, entry.currency))
        key_prefix = "sat_%s" % entry.dimension.value
        if item is None:
            inputs[key_prefix] = None
            continue
        sat = saturation_micros(item.available, item.ceiling)
        inputs[key_prefix] = sat
        if item.available.kind is QuantityKind.FINITE:
            inputs["headroom_%s" % entry.dimension.value] = int(item.available.value or 0)
        elif item.available.kind is QuantityKind.UNLIMITED:
            inputs["headroom_%s" % entry.dimension.value] = -1
        if item.next_eligible_at is not None:
            inputs["reset_%s" % entry.dimension.value] = item.next_eligible_at

    if snapshot is not None:
        inputs["availability_state"] = snapshot.state.value
        if snapshot.next_eligible_at is not None:
            inputs["next_eligible_at"] = snapshot.next_eligible_at
        if snapshot.fresh_until is not None:
            inputs["fresh_until"] = snapshot.fresh_until
        # Reset horizon: earliest next_eligible across headroom.
        horizons = [h.next_eligible_at for h in headroom if h.next_eligible_at]
        if snapshot.next_eligible_at:
            horizons.append(snapshot.next_eligible_at)
        if horizons:
            inputs["reset_horizon"] = min(horizons)

    if policy.cost_ceiling_micros is not None:
        inputs["cost_ceiling_micros"] = int(policy.cost_ceiling_micros)
    if policy.prefer_local:
        inputs["policy_prefer_local"] = True

    # Bound the map size deterministically.
    items = sorted(inputs.items(), key=lambda pair: pair[0])[:MAX_EXPLANATION_INPUTS]
    return tuple(items)


def ranking_sort_key(
    ranking_inputs: Mapping[str, Any] | Sequence[Tuple[str, Any]],
) -> Tuple[Any, ...]:
    """Deterministic multi-key sort that never sums unlike dimensions.

    Lower tuple sorts first (better).  Order:
    1. circuit closed / healthy / preferred / affinity
    2. tightest required-dimension saturation (lower better)
    3. secondary dimension saturation
    4. lower latency / queue delay
    5. prefer-local match
    6. higher quality preference
    7. higher catalog score
    8. earlier reset horizon (string compare on RFC3339 is chronological)
    """

    if isinstance(ranking_inputs, Mapping):
        data = dict(ranking_inputs)
    else:
        data = {name: value for name, value in ranking_inputs}

    def _flag(name: str, *, invert: bool = False) -> int:
        value = bool(data.get(name))
        if invert:
            value = not value
        # True is better -> 0
        return 0 if value else 1

    def _sat(name: str) -> int:
        value = data.get(name)
        if value is None:
            return 1_000_001  # unknown sorts after known
        return int(value)

    def _int(name: str, default: int = 0) -> int:
        value = data.get(name)
        if value is None:
            return default
        return int(value)

    tightest = data.get("tightest_dimension")
    primary_sat_key = "sat_%s" % tightest if tightest else None
    second = data.get("second_tightest_dimension")
    secondary_sat_key = "sat_%s" % second if second else None

    return (
        _flag("circuit_open", invert=True),  # closed first
        _flag("health"),
        _flag("preferred_binding"),
        _flag("affinity"),
        _flag("preferred_provider"),
        _flag("preferred_scope"),
        _sat(primary_sat_key) if primary_sat_key else 1_000_001,
        _sat(secondary_sat_key) if secondary_sat_key else 1_000_001,
        _int("latency_ms", default=1_000_000_000),
        _int("queue_delay_ms", default=1_000_000_000),
        _flag("prefer_local_match"),
        -_int("quality_preference", default=0),
        -_int("catalog_score", default=0),
        str(data.get("reset_horizon") or "9999-12-31T23:59:59Z"),
    )


def static_candidate_from_catalog(item: Any, *, scope_id: Optional[str] = None) -> StaticCandidate:
    """Project a catalog resolution candidate into a secret-free static view."""

    binding = getattr(item, "binding", None)
    provider = getattr(item, "provider", None)
    model = getattr(item, "model", None)
    deployment = getattr(item, "deployment", None)
    binding_id = getattr(item, "binding_id", None) or getattr(binding, "binding_id", None)
    provider_id = getattr(item, "provider_id", None) or getattr(provider, "provider_id", None)
    if not binding_id or not provider_id:
        raise ResolutionError("catalog candidate missing binding_id or provider_id")
    model_id = getattr(item, "model_id", None)
    if model_id is None and model is not None:
        model_id = getattr(model, "model_id", None)
    deployment_id = getattr(item, "deployment_id", None)
    if deployment_id is None and deployment is not None:
        deployment_id = getattr(deployment, "deployment_id", None)
    score = int(getattr(item, "score", 0) or 0)
    reasons = tuple(getattr(item, "reasons", ()) or ())

    labels: Dict[str, str] = {}
    for source in (provider, model, deployment, binding):
        if source is None:
            continue
        raw = getattr(source, "labels", ()) or ()
        if isinstance(raw, Mapping):
            labels.update({str(k): str(v) for k, v in raw.items()})
        else:
            for pair in raw:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    labels[str(pair[0])] = str(pair[1])

    locality = labels.get("locality") or labels.get("device.locality")
    state = None
    for source in (binding, deployment, provider, model):
        if source is None:
            continue
        state = getattr(source, "state", None)
        if state is not None:
            break

    def _state_flag(name: str) -> Optional[bool]:
        if state is None:
            return None
        return getattr(state, name, None)

    return StaticCandidate(
        binding_id=binding_id,
        provider_id=provider_id,
        model_id=model_id,
        deployment_id=deployment_id,
        scope_id=scope_id,
        catalog_score=score,
        locality=locality,
        authorized=_state_flag("authorized"),
        healthy=_state_flag("healthy"),
        routable=_state_flag("routable"),
        configured=_state_flag("configured"),
        labels=labels,
        reasons=reasons,
    )


def resolve_usage_aware(
    *,
    catalog_revision: str,
    candidates: Sequence[Union[StaticCandidate, Any]],
    snapshots_by_scope: Mapping[str, UsageSnapshot],
    policy: Optional[RoutingPolicy] = None,
    request: Optional[UsageRoutingRequest] = None,
    scope_by_binding: Optional[Mapping[str, str]] = None,
    limit: int = MAX_CANDIDATES,
) -> UsageAwareResolution:
    """Hard-filter then soft-rank candidates against one usage materialization.

    Parameters
    ----------
    catalog_revision:
        Immutable catalog snapshot revision the static candidates came from.
    candidates:
        Static candidates (or catalog resolution candidates) already filtered
        by operation/modality/capability, authorization, pins, context, device,
        locality, and data-governance labels.
    snapshots_by_scope:
        Mapping of ``scope_id -> UsageSnapshot`` drawn from one planning pass.
        Concurrent mutations are detected later via composite usage_revision
        mismatch rather than re-reading mid-plan.
    policy:
        Explicit routing policy; defaults to ``RoutingMode.OFF``.
    request:
        Usage-plane gates and soft preferences.
    scope_by_binding:
        Optional ``binding_id -> scope_id`` map used when a candidate lacks
        ``scope_id``.
    limit:
        Maximum accepted candidates returned (bounded).
    """

    if not isinstance(catalog_revision, str) or not catalog_revision:
        raise ResolutionError("catalog_revision is required")
    policy = policy if policy is not None else RoutingPolicy(mode=RoutingMode.OFF)
    if not isinstance(policy, RoutingPolicy):
        policy = RoutingPolicy.from_dict(policy)
    request = request if request is not None else UsageRoutingRequest()
    if not isinstance(request, UsageRoutingRequest):
        request = UsageRoutingRequest.from_dict(request)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_CANDIDATES
    ):
        raise ResolutionError("limit must be between 1 and %d" % MAX_CANDIDATES)

    scope_map = dict(scope_by_binding or {})
    static_items: List[StaticCandidate] = []
    for item in candidates:
        if isinstance(item, StaticCandidate):
            cand = item
        else:
            binding_id = getattr(item, "binding_id", None)
            if binding_id is None and getattr(item, "binding", None) is not None:
                binding_id = item.binding.binding_id
            scope_id = scope_map.get(binding_id) if binding_id else None
            cand = static_candidate_from_catalog(item, scope_id=scope_id)
        if cand.scope_id is None and cand.binding_id in scope_map:
            cand = StaticCandidate(
                binding_id=cand.binding_id,
                provider_id=cand.provider_id,
                model_id=cand.model_id,
                deployment_id=cand.deployment_id,
                scope_id=scope_map[cand.binding_id],
                catalog_score=cand.catalog_score,
                locality=cand.locality,
                authorized=cand.authorized,
                healthy=cand.healthy,
                routable=cand.routable,
                configured=cand.configured,
                labels=cand.labels,
                reasons=cand.reasons,
            )
        static_items.append(cand)

    # OFF mode: preserve static eligibility order without usage hard gates.
    if policy.mode is RoutingMode.OFF:
        accepted: List[ResolutionCandidate] = []
        for rank, cand in enumerate(static_items[:limit]):
            scope_id = cand.scope_id or stable_id("scope", "off", cand.binding_id)
            accepted.append(
                ResolutionCandidate(
                    binding_id=cand.binding_id,
                    scope_id=scope_id,
                    rank=rank,
                    state=AvailabilityState.UNKNOWN,
                    headroom=(),
                    rejection_reasons=(),
                    ranking_inputs=(
                        ("catalog_score", cand.catalog_score),
                        ("usage_routing", "off"),
                    ),
                )
            )
        selected = accepted[0].binding_id if accepted else None
        return UsageAwareResolution(
            catalog_revision=catalog_revision,
            usage_revision=USAGE_REVISION_OFF,
            policy_id=policy.policy_id,
            candidates=tuple(accepted),
            rejected=(),
            selected_binding_id=selected,
            reason_codes=tuple(sorted(set(request.reason_codes) | {"usage_routing_off"})),
        )

    clock = _now(request)
    accepted_rows: List[Tuple[Tuple[Any, ...], ResolutionCandidate, Optional[str]]] = []
    rejected_rows: List[ResolutionCandidate] = []
    consulted: List[UsageSnapshot] = []
    seen_scope: set = set()

    for cand in static_items:
        scope_id = cand.scope_id
        snapshot = snapshots_by_scope.get(scope_id) if scope_id else None
        if snapshot is not None and scope_id not in seen_scope:
            consulted.append(snapshot)
            seen_scope.add(scope_id)

        ok, reject_reasons, headroom = hard_filter_candidate(
            cand, snapshot, request, policy, now=clock
        )
        state = (
            snapshot.state
            if snapshot is not None
            else AvailabilityState.UNKNOWN
        )
        ranking = build_ranking_inputs(cand, snapshot, request, policy, headroom)
        effective_scope = scope_id or stable_id("scope", "unknown", cand.binding_id)

        if not ok:
            rejected_rows.append(
                ResolutionCandidate(
                    binding_id=cand.binding_id,
                    scope_id=effective_scope,
                    rank=0,
                    state=state,
                    headroom=headroom,
                    rejection_reasons=reject_reasons,
                    ranking_inputs=ranking,
                )
            )
            continue

        sort_key = ranking_sort_key(ranking)
        next_eligible = snapshot.next_eligible_at if snapshot is not None else None
        accepted_rows.append(
            (
                sort_key + (cand.binding_id,),
                ResolutionCandidate(
                    binding_id=cand.binding_id,
                    scope_id=effective_scope,
                    rank=0,
                    state=state,
                    headroom=headroom,
                    rejection_reasons=(),
                    ranking_inputs=ranking,
                ),
                next_eligible,
            )
        )

    accepted_rows.sort(key=lambda row: row[0])
    ranked: List[ResolutionCandidate] = []
    next_eligible_times: List[str] = []
    for rank, (_key, candidate, next_eligible) in enumerate(accepted_rows[:limit]):
        ranked.append(
            ResolutionCandidate(
                binding_id=candidate.binding_id,
                scope_id=candidate.scope_id,
                rank=rank,
                state=candidate.state,
                headroom=candidate.headroom,
                rejection_reasons=candidate.rejection_reasons,
                ranking_inputs=candidate.ranking_inputs,
            )
        )
        if next_eligible:
            next_eligible_times.append(next_eligible)

    # Include rejected next-eligible for no-capacity planning.
    for item in rejected_rows:
        for hr in item.headroom:
            if hr.next_eligible_at:
                next_eligible_times.append(hr.next_eligible_at)

    usage_revision = composite_usage_revision(consulted, mode=policy.mode)
    selected = ranked[0].binding_id if ranked else None
    reason_codes: List[str] = list(request.reason_codes)
    if not ranked:
        reason_codes.append("no_eligible_candidates")
    if rejected_rows and ranked:
        reason_codes.append("partial_rejection")
    if not consulted and policy.mode is not RoutingMode.OFF:
        reason_codes.append("no_usage_snapshots")

    return UsageAwareResolution(
        catalog_revision=catalog_revision,
        usage_revision=usage_revision,
        policy_id=policy.policy_id,
        candidates=tuple(ranked),
        rejected=tuple(
            sorted(rejected_rows, key=lambda item: item.binding_id)[:MAX_CANDIDATES]
        ),
        selected_binding_id=selected,
        next_eligible_at=min(next_eligible_times) if next_eligible_times else None,
        reason_codes=tuple(sorted(set(reason_codes))),
    )


def list_limits_page(
    snapshot: UsageSnapshot,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    cursor: Optional[str] = None,
    dimension: Optional[Union[str, UsageDimension]] = None,
) -> UsageLimitPage:
    """Return a bounded, deterministic page of limits from one snapshot."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_PAGE_LIMIT
    ):
        raise ResolutionError("limit must be between 1 and %d" % MAX_PAGE_LIMIT)
    items = list(snapshot.limits)
    if dimension is not None:
        dim = (
            dimension
            if isinstance(dimension, UsageDimension)
            else UsageDimension(str(dimension))
        )
        items = [item for item in items if item.dimension is dim]
    # Stable order by limit_id (schema already sorts, reaffirm).
    items.sort(key=lambda item: item.limit_id or "")
    start = 0
    if cursor is not None:
        if not isinstance(cursor, str) or not cursor:
            raise ResolutionError("cursor must be non-empty text")
        # Cursor is the last limit_id from the previous page.
        for idx, item in enumerate(items):
            if item.limit_id == cursor:
                start = idx + 1
                break
        else:
            raise ResolutionError("cursor does not match any limit in this snapshot")
    page = items[start : start + limit]
    next_cursor = None
    if start + limit < len(items) and page:
        next_cursor = page[-1].limit_id
    # Bound page size to MAX_LIMITS as well.
    page = page[:MAX_LIMITS]
    return UsageLimitPage(
        scope_id=snapshot.scope_id,
        usage_revision=snapshot.usage_revision or USAGE_REVISION_UNAVAILABLE,
        items=tuple(page),
        next_cursor=next_cursor,
        total=len(items),
    )


def filter_headroom(
    snapshot: UsageSnapshot,
    *,
    dimension: Optional[Union[str, UsageDimension]] = None,
) -> Tuple[DimensionHeadroom, ...]:
    """Return headroom entries, optionally filtered by dimension."""

    items = list(snapshot.headroom)
    if dimension is not None:
        dim = (
            dimension
            if isinstance(dimension, UsageDimension)
            else UsageDimension(str(dimension))
        )
        items = [item for item in items if item.dimension is dim]
    items.sort(key=lambda item: (item.dimension.value, item.currency or ""))
    return tuple(items)


def read_snapshot(
    service: UsageSnapshotReader,
    scope_id: str,
    *,
    expected_usage_revision: Optional[str] = None,
) -> UsageSnapshot:
    """Read one snapshot and optionally enforce an expected revision pin."""

    if not isinstance(scope_id, str) or not scope_id:
        raise ResolutionError("scope_id is required")
    snapshot = service.snapshot(scope_id)
    if not isinstance(snapshot, UsageSnapshot):
        # Allow duck-typed services that return mappings.
        if isinstance(snapshot, Mapping):
            snapshot = UsageSnapshot.from_dict(snapshot)
        else:
            raise ResolutionError("usage service returned an invalid snapshot")
    if (
        expected_usage_revision is not None
        and snapshot.usage_revision != expected_usage_revision
    ):
        raise RevisionMismatch(
            "usage revision mismatch for scope %s" % scope_id,
            expected=expected_usage_revision,
            actual=snapshot.usage_revision,
            kind="usage",
        )
    return snapshot


def collect_snapshots(
    service: UsageSnapshotReader,
    scope_ids: Iterable[str],
    *,
    expected_revisions: Optional[Mapping[str, str]] = None,
) -> Dict[str, UsageSnapshot]:
    """Collect snapshots for many scopes in one planning pass.

    Each scope is read once.  A revision pin mismatch raises
    :class:`RevisionMismatch` rather than mixing revisions.
    """

    expected = dict(expected_revisions or {})
    out: Dict[str, UsageSnapshot] = {}
    for scope_id in scope_ids:
        if not scope_id or scope_id in out:
            continue
        out[scope_id] = read_snapshot(
            service,
            scope_id,
            expected_usage_revision=expected.get(scope_id),
        )
    return out


__all__ = [
    "USAGE_AWARE_RESOLUTION_REQUIREMENT_ID",
    "USAGE_REVISION_OFF",
    "USAGE_REVISION_UNAVAILABLE",
    "DEFAULT_PAGE_LIMIT",
    "MAX_PAGE_LIMIT",
    "ResolutionError",
    "UsageServiceUnavailable",
    "RevisionMismatch",
    "UnknownLimitPolicy",
    "StaleSnapshotPolicy",
    "UsageSnapshotReader",
    "UsageLimitPage",
    "UsageRoutingRequest",
    "StaticCandidate",
    "composite_usage_revision",
    "headroom_index",
    "saturation_micros",
    "tightest_dimensions",
    "is_snapshot_stale",
    "hard_filter_candidate",
    "build_ranking_inputs",
    "ranking_sort_key",
    "static_candidate_from_catalog",
    "resolve_usage_aware",
    "list_limits_page",
    "filter_headroom",
    "read_snapshot",
    "collect_snapshots",
]
