"""Shared route admission, ranking, fallback, and attempt protocol.

Hard capability, authorization, exact-pin, safety/data, locality/device, cost,
media, context, and deadline constraints cannot be offset by score.  Soft
ranking runs only after hard filtering and uses affinity/stickiness, projected
tightest-dimension saturation, reset horizon, health/circuit, latency/queue,
cost, locality, and optional policy quality.

The router-facing protocol closes the selection race with an atomic
reservation and tries the next eligible candidate only after a typed denial.
Fallback classes are explicit; exact pins default to ``none``; each
retry/fallback receives a new linked attempt and reservation.  Unsafe
semantic, client, and side-effecting errors never fallback.  Wait versus
reroute honors one shared deadline and maximum attempt bound.  Admission
jitter, single-flight refresh, and circuit breakers limit herds.

This module never invokes a provider.  Callers supply an invoke callback
(or settle/observe after their own dispatch).  No network, credential store,
or model load occurs on import.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

from .coordinator import (
    DEFAULT_RESERVATION_TTL_MS,
    ReserveDecision,
    SettlementResult,
    UsageCoordinator,
)
from .identity import stable_id
from .ledger import CapacityDenied, LedgerError, StaleSnapshot
from .receipts import (
    AttemptLink,
    FinalStatus,
    ReceiptChain,
    RouteReceiptDraft,
    build_receipt_chain,
    build_usage_routing_receipt,
    candidates_digest,
    hard_rejection_digest,
    ranking_inputs_digest,
)
from .resolution import (
    USAGE_REVISION_OFF,
    StaticCandidate,
    UsageRoutingRequest,
    hard_filter_candidate,
    resolve_usage_aware,
)
from .schema import (
    AvailabilityState,
    EstimateMethod,
    FallbackClass,
    ProviderUsageObservation,
    Quantity,
    QuantityKind,
    ResolutionCandidate,
    RoutingMode,
    RoutingPolicy,
    UsageAwareResolution,
    UsageDimension,
    UsageErrorCode,
    UsageEstimate,
    UsageEventKind,
    UsageReservation,
    UsageRoutingReceipt,
    UsageSnapshot,
    UsageVector,
    UsageVectorEntry,
)
from .store import CompareAndSetConflict

ROUTE_ADMISSION_REQUIREMENT_ID = "requirement:usage-route-admission.v1"

DEFAULT_MAX_ATTEMPTS = 1
DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_COOLDOWN_MS = 30_000
DEFAULT_JITTER_MAX_MS = 250
MAX_ATTEMPT_BOUND = 32


class RoutingError(ValueError):
    """Invalid route admission input or protocol state."""


class DenialKind(str, Enum):
    """Typed reservation/selection denials — only these advance the candidate cursor."""

    HARD_FILTER = "hard_filter"
    CAPACITY = "capacity"
    RESERVATION_CONFLICT = "reservation_conflict"
    STALE_SNAPSHOT = "stale_snapshot"
    CIRCUIT_OPEN = "circuit_open"
    POLICY = "policy"
    PIN = "pin"
    FALLBACK_BOUNDARY = "fallback_boundary"
    DEADLINE = "deadline"
    MAX_ATTEMPTS = "max_attempts"
    NO_CANDIDATES = "no_candidates"
    CAS_EXHAUSTED = "cas_exhausted"
    UNKNOWN = "unknown"


class ErrorSafetyClass(str, Enum):
    """Classify invoke failures for retry/fallback safety.

    ``semantic``, ``client``, and ``side_effect`` never trigger fallback.
    """

    SUCCESS = "success"
    CAPACITY = "capacity"  # 429/503/quota — safe to wait or reroute
    TRANSIENT = "transient"  # network blip — safe retry same binding
    SEMANTIC = "semantic"  # bad request semantics — never fallback
    CLIENT = "client"  # 4xx client error — never fallback
    SIDE_EFFECT = "side_effect"  # may have mutated state — never fallback
    UNKNOWN = "unknown"


class WaitOrReroute(str, Enum):
    """Decision between waiting for capacity and rerouting."""

    REROUTE = "reroute"
    WAIT = "wait"
    FAIL = "fail"


@dataclass(frozen=True)
class RoutePin:
    """Exact identity pins.  Default fallback boundary is ``none``.

    When any pin field is set, the effective fallback class is forced to
    :attr:`FallbackClass.NONE` unless *allow_fallback_with_pin* is True and
    the policy explicitly names a compatible class.
    """

    provider_id: Optional[str] = None
    model_id: Optional[str] = None
    deployment_id: Optional[str] = None
    binding_id: Optional[str] = None
    allow_fallback_with_pin: bool = False

    def __post_init__(self) -> None:
        for name in ("provider_id", "model_id", "deployment_id", "binding_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RoutingError("%s must be non-empty text when set" % name)
        if not isinstance(self.allow_fallback_with_pin, bool):
            raise RoutingError("allow_fallback_with_pin must be a boolean")

    @property
    def is_exact(self) -> bool:
        return any(
            (
                self.provider_id,
                self.model_id,
                self.deployment_id,
                self.binding_id,
            )
        )

    def effective_fallback(self, policy: RoutingPolicy) -> FallbackClass:
        """Exact pins default to ``none``; otherwise use the policy class."""

        if self.is_exact and not self.allow_fallback_with_pin:
            return FallbackClass.NONE
        return policy.fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "binding_id": self.binding_id,
            "allow_fallback_with_pin": self.allow_fallback_with_pin,
        }


@dataclass(frozen=True)
class RouteCandidateMeta:
    """Identity surface used for fallback-class boundary checks.

    Constructed from planning candidates without retaining raw endpoints.
    """

    binding_id: str
    provider_id: str
    model_id: Optional[str] = None
    deployment_id: Optional[str] = None
    scope_id: Optional[str] = None
    equivalent_model_group: Optional[str] = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.binding_id or not self.provider_id:
            raise RoutingError("binding_id and provider_id are required")
        object.__setattr__(self, "labels", dict(self.labels or {}))


def meta_from_static(candidate: StaticCandidate) -> RouteCandidateMeta:
    """Project a static planning candidate into fallback-class metadata."""

    group = None
    labels = dict(candidate.labels or {})
    group = labels.get("equivalent_model") or labels.get("model.equivalent_group")
    return RouteCandidateMeta(
        binding_id=candidate.binding_id,
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        deployment_id=candidate.deployment_id,
        scope_id=candidate.scope_id,
        equivalent_model_group=group,
        labels=labels,
    )


def fallback_class_allows(
    origin: RouteCandidateMeta,
    candidate: RouteCandidateMeta,
    fallback: FallbackClass,
) -> bool:
    """Return whether *candidate* is inside the *fallback* boundary of *origin*.

    Classes are distinguishable:
    - ``none``: only the exact binding
    - ``same_deployment``: same deployment_id
    - ``same_provider``: same provider_id
    - ``same_model``: same model_id (and typically provider)
    - ``equivalent_model``: shared equivalent_model_group label
    - ``cross_provider``: any remaining candidate
    """

    if fallback is FallbackClass.NONE:
        return candidate.binding_id == origin.binding_id
    if fallback is FallbackClass.SAME_DEPLOYMENT:
        if not origin.deployment_id or not candidate.deployment_id:
            return candidate.binding_id == origin.binding_id
        return candidate.deployment_id == origin.deployment_id
    if fallback is FallbackClass.SAME_PROVIDER:
        return candidate.provider_id == origin.provider_id
    if fallback is FallbackClass.SAME_MODEL:
        if not origin.model_id or not candidate.model_id:
            return candidate.binding_id == origin.binding_id
        return (
            candidate.model_id == origin.model_id
            and candidate.provider_id == origin.provider_id
        )
    if fallback is FallbackClass.EQUIVALENT_MODEL:
        if origin.equivalent_model_group and candidate.equivalent_model_group:
            return origin.equivalent_model_group == candidate.equivalent_model_group
        # Without an equivalence group, degrade to same_model.
        return fallback_class_allows(origin, candidate, FallbackClass.SAME_MODEL)
    if fallback is FallbackClass.CROSS_PROVIDER:
        return True
    return False


def apply_pin_filter(
    candidates: Sequence[StaticCandidate],
    pin: RoutePin,
) -> Tuple[List[StaticCandidate], List[Tuple[StaticCandidate, Tuple[str, ...]]]]:
    """Hard-filter candidates against exact pins.  Score cannot offset a pin."""

    accepted: List[StaticCandidate] = []
    rejected: List[Tuple[StaticCandidate, Tuple[str, ...]]] = []
    for cand in candidates:
        reasons: List[str] = []
        if pin.binding_id and cand.binding_id != pin.binding_id:
            reasons.append("pin_binding_mismatch")
        if pin.provider_id and cand.provider_id != pin.provider_id:
            reasons.append("pin_provider_mismatch")
        if pin.model_id and cand.model_id != pin.model_id:
            reasons.append("pin_model_mismatch")
        if pin.deployment_id and cand.deployment_id != pin.deployment_id:
            reasons.append("pin_deployment_mismatch")
        if reasons:
            rejected.append((cand, tuple(sorted(set(reasons)))))
        else:
            accepted.append(cand)
    return accepted, rejected


def is_fallback_safe(error_class: ErrorSafetyClass) -> bool:
    """Unsafe semantic/client/side-effecting errors never fallback."""

    return error_class in (
        ErrorSafetyClass.CAPACITY,
        ErrorSafetyClass.TRANSIENT,
    )


def classify_invoke_error(
    *,
    http_status: Optional[int] = None,
    reason_codes: Sequence[str] = (),
    side_effecting: bool = False,
    semantic: bool = False,
    client_error: bool = False,
) -> ErrorSafetyClass:
    """Map status/reason signals onto a fallback-safety class."""

    if side_effecting:
        return ErrorSafetyClass.SIDE_EFFECT
    if semantic:
        return ErrorSafetyClass.SEMANTIC
    if client_error:
        return ErrorSafetyClass.CLIENT
    codes = {str(c).casefold() for c in reason_codes}
    if codes & {
        "semantic_error",
        "invalid_request",
        "context_overflow",
        "context_length",
        "tool_side_effect",
        "side_effect",
        "not_retryable",
    }:
        if "tool_side_effect" in codes or "side_effect" in codes:
            return ErrorSafetyClass.SIDE_EFFECT
        return ErrorSafetyClass.SEMANTIC
    if http_status is not None:
        if http_status in (429, 503):
            return ErrorSafetyClass.CAPACITY
        if 400 <= http_status < 500 and http_status != 408:
            return ErrorSafetyClass.CLIENT
        if http_status in (408, 500, 502, 504):
            return ErrorSafetyClass.TRANSIENT
    if codes & {
        "limit_exhausted",
        "rate_limited",
        "quota_exceeded",
        "capacity_unavailable",
        "cooling_down",
    }:
        return ErrorSafetyClass.CAPACITY
    if codes & {"timeout", "connection_error", "unavailable"}:
        return ErrorSafetyClass.TRANSIENT
    return ErrorSafetyClass.UNKNOWN


def admission_jitter_ms(
    request_id: str,
    *,
    max_ms: int = DEFAULT_JITTER_MAX_MS,
    salt: str = "",
) -> int:
    """Deterministic admission jitter in ``[0, max_ms]`` to desynchronize herds."""

    if max_ms <= 0:
        return 0
    material = ("%s|%s" % (request_id, salt)).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    value = int.from_bytes(digest[:4], "big")
    return value % (int(max_ms) + 1)


def decide_wait_or_reroute(
    *,
    policy: RoutingPolicy,
    now: datetime,
    next_eligible_at: Optional[datetime],
    deadline_at: Optional[datetime],
    has_reroute_candidate: bool,
    attempts_used: int,
) -> WaitOrReroute:
    """Honor one deadline and max-attempt bound for wait versus reroute."""

    if attempts_used >= policy.max_attempts:
        return WaitOrReroute.FAIL
    if deadline_at is not None and now >= deadline_at:
        return WaitOrReroute.FAIL

    # Prefer reroute when another eligible candidate exists.
    if has_reroute_candidate:
        return WaitOrReroute.REROUTE

    if not policy.allow_wait or next_eligible_at is None:
        return WaitOrReroute.FAIL

    if next_eligible_at <= now:
        return WaitOrReroute.REROUTE  # capacity should be free; replan

    wait_ms = int((next_eligible_at - now).total_seconds() * 1000)
    if policy.max_wait_ms is not None and wait_ms > int(policy.max_wait_ms):
        return WaitOrReroute.FAIL
    if deadline_at is not None and next_eligible_at > deadline_at:
        return WaitOrReroute.FAIL
    if policy.deadline_ms is not None:
        # Policy wall budget measured from *now* when no absolute deadline given.
        if wait_ms > int(policy.deadline_ms):
            return WaitOrReroute.FAIL
    return WaitOrReroute.WAIT


# ---------------------------------------------------------------------------
# Circuit breaker + single-flight
# ---------------------------------------------------------------------------


@dataclass
class CircuitState:
    """Per-binding circuit breaker state."""

    failures: int = 0
    opened_at_ms: Optional[int] = None
    state: str = "closed"  # closed | open | half_open

    def is_open(self, now_ms: int, cooldown_ms: int) -> bool:
        if self.state != "open":
            return False
        if self.opened_at_ms is None:
            return True
        if now_ms - self.opened_at_ms >= cooldown_ms:
            self.state = "half_open"
            return False
        return True


class CircuitBreakerRegistry:
    """In-process circuit breakers that prevent retry herds on a hot binding."""

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
        cooldown_ms: int = DEFAULT_CIRCUIT_COOLDOWN_MS,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self._threshold = max(1, int(failure_threshold))
        self._cooldown_ms = max(1, int(cooldown_ms))
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._states: Dict[str, CircuitState] = {}
        self._lock = threading.Lock()

    def is_open(self, binding_id: str) -> bool:
        with self._lock:
            state = self._states.get(binding_id)
            if state is None:
                return False
            return state.is_open(self._clock_ms(), self._cooldown_ms)

    def record_success(self, binding_id: str) -> None:
        with self._lock:
            self._states[binding_id] = CircuitState()

    def record_failure(self, binding_id: str) -> None:
        with self._lock:
            state = self._states.get(binding_id) or CircuitState()
            state.failures += 1
            if state.failures >= self._threshold:
                state.state = "open"
                state.opened_at_ms = self._clock_ms()
            self._states[binding_id] = state

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                key: {
                    "failures": value.failures,
                    "state": value.state,
                    "opened_at_ms": value.opened_at_ms,
                }
                for key, value in self._states.items()
            }


class SingleFlight:
    """Collapse concurrent identical admission keys into one in-flight work item."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}

    def do(self, key: str, fn: Callable[[], Any]) -> Any:
        """Run *fn* once per *key*; concurrent callers wait for the winner."""

        with self._lock:
            if key in self._results:
                return self._results[key]
            event = self._inflight.get(key)
            is_leader = event is None
            if is_leader:
                event = threading.Event()
                self._inflight[key] = event
        if not is_leader:
            event.wait(timeout=60.0)
            with self._lock:
                if key in self._results:
                    return self._results[key]
                # Leader failed without publishing; run ourselves.
                self._inflight.pop(key, None)
            return self.do(key, fn)
        try:
            result = fn()
            with self._lock:
                self._results[key] = result
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()
                # Bound cache: drop after publish so later admissions re-run.
                # Keep only a short sticky window by not retaining forever.
                # Callers that need sticky replay use coordinator idempotency.
                self._results.pop(key, None)


# ---------------------------------------------------------------------------
# Attempt / admission results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedDenial:
    """A typed denial that may advance the candidate cursor."""

    kind: DenialKind
    reason_codes: Tuple[str, ...] = ()
    binding_id: Optional[str] = None
    scope_id: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason_codes": list(self.reason_codes),
            "binding_id": self.binding_id,
            "scope_id": self.scope_id,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class InvokeOutcome:
    """Result of the router-owned invoke step (callback-supplied)."""

    success: bool
    observation: Optional[ProviderUsageObservation] = None
    settled: Optional[UsageVector] = None
    error_class: ErrorSafetyClass = ErrorSafetyClass.SUCCESS
    reason_codes: Tuple[str, ...] = ()
    side_effecting: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise RoutingError("success must be a boolean")
        if not isinstance(self.error_class, ErrorSafetyClass):
            object.__setattr__(
                self, "error_class", ErrorSafetyClass(str(self.error_class))
            )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes or ()))


@dataclass(frozen=True)
class RouteAttemptResult:
    """One attempt's outcome: reservation race closed, optionally settled."""

    attempt_id: str
    parent_attempt_id: Optional[str]
    attempt_index: int
    binding_id: Optional[str]
    scope_id: Optional[str]
    reservation_id: Optional[str]
    reservation: Optional[UsageReservation]
    decision: Optional[ReserveDecision]
    denial: Optional[TypedDenial]
    fallback_class: FallbackClass
    resolution: Optional[UsageAwareResolution]
    estimate: Optional[UsageEstimate]
    observation: Optional[ProviderUsageObservation]
    settlement: Optional[SettlementResult]
    error_class: ErrorSafetyClass
    final_status: str
    reason_codes: Tuple[str, ...]
    receipt: Optional[UsageRoutingReceipt]
    next_eligible_at: Optional[str] = None
    usage_revision: Optional[str] = None
    catalog_revision: Optional[str] = None
    jitter_ms: int = 0

    @property
    def granted(self) -> bool:
        return self.reservation is not None and self.denial is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "parent_attempt_id": self.parent_attempt_id,
            "attempt_index": self.attempt_index,
            "binding_id": self.binding_id,
            "scope_id": self.scope_id,
            "reservation_id": self.reservation_id,
            "denial": self.denial.to_dict() if self.denial else None,
            "fallback_class": self.fallback_class.value,
            "error_class": self.error_class.value,
            "final_status": self.final_status,
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt.receipt_id if self.receipt else None,
            "next_eligible_at": self.next_eligible_at,
            "usage_revision": self.usage_revision,
            "catalog_revision": self.catalog_revision,
            "jitter_ms": self.jitter_ms,
            "granted": self.granted,
        }


@dataclass(frozen=True)
class RouteAdmissionResult:
    """Terminal outcome of the multi-attempt admission protocol."""

    success: bool
    attempts: Tuple[RouteAttemptResult, ...]
    selected: Optional[RouteAttemptResult]
    chain: ReceiptChain
    receipt: Optional[UsageRoutingReceipt]
    final_status: str
    reason_codes: Tuple[str, ...]
    wait_or_reroute: Optional[WaitOrReroute] = None
    next_eligible_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "attempts": [item.to_dict() for item in self.attempts],
            "selected_attempt_id": self.selected.attempt_id if self.selected else None,
            "chain": self.chain.to_dict(),
            "receipt_id": self.receipt.receipt_id if self.receipt else None,
            "final_status": self.final_status,
            "reason_codes": list(self.reason_codes),
            "wait_or_reroute": self.wait_or_reroute.value
            if self.wait_or_reroute
            else None,
            "next_eligible_at": self.next_eligible_at,
        }


class SnapshotProvider(Protocol):
    """Minimal snapshot surface for admission planning."""

    def snapshot(self, scope_id: str) -> UsageSnapshot:
        ...


# ---------------------------------------------------------------------------
# Planner (hard filter + rank; no reservation)
# ---------------------------------------------------------------------------


def plan_route(
    *,
    catalog_revision: str,
    candidates: Sequence[StaticCandidate],
    snapshots_by_scope: Mapping[str, UsageSnapshot],
    policy: Optional[RoutingPolicy] = None,
    request: Optional[UsageRoutingRequest] = None,
    pin: Optional[RoutePin] = None,
    meta_by_binding: Optional[Mapping[str, RouteCandidateMeta]] = None,
    origin_binding_id: Optional[str] = None,
    limit: int = 128,
) -> UsageAwareResolution:
    """Hard-filter then soft-rank.  Hard gates cannot be offset by score.

    Applies exact pins and the effective fallback boundary before ranking.
    """

    policy = policy if policy is not None else RoutingPolicy(mode=RoutingMode.ENFORCE)
    request = request if request is not None else UsageRoutingRequest()
    pin = pin if pin is not None else RoutePin()

    static_list = list(candidates)
    pinned, pin_rejected = apply_pin_filter(static_list, pin)
    static_list = pinned

    # Fallback boundary relative to origin (first preferred / prior selection).
    effective_fallback = pin.effective_fallback(policy)
    if origin_binding_id and meta_by_binding and origin_binding_id in meta_by_binding:
        origin = meta_by_binding[origin_binding_id]
        filtered: List[StaticCandidate] = []
        for cand in static_list:
            meta = meta_by_binding.get(cand.binding_id) or meta_from_static(cand)
            if fallback_class_allows(origin, meta, effective_fallback):
                filtered.append(cand)
        static_list = filtered

    resolution = resolve_usage_aware(
        catalog_revision=catalog_revision,
        candidates=static_list,
        snapshots_by_scope=snapshots_by_scope,
        policy=policy,
        request=request,
        limit=limit,
    )

    # Fold pin rejections into reason codes when nothing remains.
    if pin_rejected and not resolution.candidates:
        extra = list(resolution.reason_codes) + ["pin_rejected"]
        return UsageAwareResolution(
            catalog_revision=resolution.catalog_revision,
            usage_revision=resolution.usage_revision,
            policy_id=resolution.policy_id,
            candidates=resolution.candidates,
            rejected=resolution.rejected,
            selected_binding_id=resolution.selected_binding_id,
            next_eligible_at=resolution.next_eligible_at,
            reason_codes=tuple(sorted(set(extra))),
        )
    return resolution


# ---------------------------------------------------------------------------
# Admission protocol
# ---------------------------------------------------------------------------


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
        raise RoutingError("timestamp is not RFC 3339: %r" % value) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RoutingError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _to_rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _attempt_id(request_id: str, index: int) -> str:
    return stable_id("attempt", request_id, str(index))


def _estimate_for_scope(
    scope_id: str,
    operation: str,
    requested: UsageVector,
    *,
    method: EstimateMethod = EstimateMethod.CONSERVATIVE,
    estimated_at: Optional[str] = None,
) -> UsageEstimate:
    return UsageEstimate(
        scope_id=scope_id,
        operation=operation,
        requested=requested,
        method=method,
        estimated_at=estimated_at,
    )


class UsageRouteAdmission:
    """Router-facing attempt protocol over a :class:`UsageCoordinator`.

    Lifecycle per attempt:
      plan → atomic reserve → (typed denial → next candidate) → invoke callback
      → observe → settle → receipt.  Retries/fallbacks mint a new attempt id
      and a new reservation linked to the prior attempt.
    """

    requirement_id = ROUTE_ADMISSION_REQUIREMENT_ID

    def __init__(
        self,
        coordinator: UsageCoordinator,
        *,
        owner_id: str = "route-admission",
        circuits: Optional[CircuitBreakerRegistry] = None,
        single_flight: Optional[SingleFlight] = None,
        jitter_max_ms: int = DEFAULT_JITTER_MAX_MS,
        reservation_ttl_ms: int = DEFAULT_RESERVATION_TTL_MS,
        apply_jitter_sleep: bool = False,
    ) -> None:
        if coordinator is None:
            raise RoutingError("coordinator is required")
        self._coord = coordinator
        self._owner_id = owner_id
        self._circuits = circuits or CircuitBreakerRegistry(
            clock_ms=lambda: int(
                self._coord.clock.now().timestamp() * 1000
            )
        )
        self._single_flight = single_flight or SingleFlight()
        self._jitter_max_ms = max(0, int(jitter_max_ms))
        self._ttl_ms = max(1, int(reservation_ttl_ms))
        self._apply_jitter_sleep = bool(apply_jitter_sleep)

    @property
    def coordinator(self) -> UsageCoordinator:
        return self._coord

    @property
    def circuits(self) -> CircuitBreakerRegistry:
        return self._circuits

    def admit(
        self,
        *,
        catalog_revision: str,
        candidates: Sequence[StaticCandidate],
        request_id: str,
        idempotency_key: str,
        operation: str,
        requested: UsageVector,
        policy: Optional[RoutingPolicy] = None,
        request: Optional[UsageRoutingRequest] = None,
        pin: Optional[RoutePin] = None,
        meta_by_binding: Optional[Mapping[str, RouteCandidateMeta]] = None,
        snapshots_by_scope: Optional[Mapping[str, UsageSnapshot]] = None,
        snapshot_provider: Optional[SnapshotProvider] = None,
        invoke: Optional[Callable[[RouteAttemptResult], InvokeOutcome]] = None,
        caller_id: Optional[str] = None,
        origin_binding_id: Optional[str] = None,
        parent_attempt_id: Optional[str] = None,
        start_attempt_index: int = 0,
        settle_on_success: bool = True,
        release_on_failure: bool = True,
    ) -> RouteAdmissionResult:
        """Run the full multi-attempt admission protocol.

        Parameters
        ----------
        invoke:
            Optional callback invoked only after a reservation is granted.
            When omitted, the first successful reservation is returned as the
            selected attempt (pre-dispatch); the caller must settle later.
        """

        if not catalog_revision or not isinstance(catalog_revision, str):
            raise RoutingError("catalog_revision is required")
        if not request_id or not isinstance(request_id, str):
            raise RoutingError("request_id is required")
        if not idempotency_key or not isinstance(idempotency_key, str):
            raise RoutingError("idempotency_key is required")
        if not operation or not isinstance(operation, str):
            raise RoutingError("operation is required")
        if not isinstance(requested, UsageVector) or not requested.entries:
            raise RoutingError("requested usage vector must be non-empty")

        policy = policy if policy is not None else RoutingPolicy(
            mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE, max_attempts=1
        )
        if not isinstance(policy, RoutingPolicy):
            policy = RoutingPolicy.from_dict(policy)
        request = request if request is not None else UsageRoutingRequest()
        pin = pin if pin is not None else RoutePin()
        effective_fallback = pin.effective_fallback(policy)

        # Build meta map from static candidates when not provided.
        meta_map: Dict[str, RouteCandidateMeta] = dict(meta_by_binding or {})
        static_candidates = list(candidates)
        for cand in static_candidates:
            if cand.binding_id not in meta_map:
                meta_map[cand.binding_id] = meta_from_static(cand)

        # Collect snapshots for planning.
        snap_map: Dict[str, UsageSnapshot] = dict(snapshots_by_scope or {})
        if snapshot_provider is not None:
            for cand in static_candidates:
                sid = cand.scope_id
                if sid and sid not in snap_map:
                    try:
                        snap_map[sid] = snapshot_provider.snapshot(sid)
                    except Exception:
                        # Missing snapshots are handled by hard_filter_candidate.
                        pass
        # Prefer coordinator snapshots when scopes known and not supplied.
        for cand in static_candidates:
            sid = cand.scope_id
            if sid and sid not in snap_map:
                try:
                    snap_map[sid] = self._coord.snapshot(sid)
                except Exception:
                    pass

        deadline_at = _parse_rfc3339(request.deadline_at)
        if deadline_at is None and policy.deadline_ms is not None:
            deadline_at = self._coord.clock.now() + timedelta(
                milliseconds=int(policy.deadline_ms)
            )
        now = self._coord.clock.now()
        if request.now is not None:
            now = _parse_rfc3339(request.now) or now

        max_attempts = min(int(policy.max_attempts), MAX_ATTEMPT_BOUND)
        attempts: List[RouteAttemptResult] = []
        parent = parent_attempt_id
        origin = origin_binding_id
        excluded_bindings: set = set()
        terminal_receipt: Optional[UsageRoutingReceipt] = None
        wait_decision: Optional[WaitOrReroute] = None
        next_eligible: Optional[str] = None

        for index in range(start_attempt_index, start_attempt_index + max_attempts):
            now = self._coord.clock.now()
            if request.now is not None and index == start_attempt_index:
                now = _parse_rfc3339(request.now) or now
            if deadline_at is not None and now >= deadline_at:
                wait_decision = WaitOrReroute.FAIL
                break

            jitter = admission_jitter_ms(
                request_id, max_ms=self._jitter_max_ms, salt=str(index)
            )
            if self._apply_jitter_sleep and jitter > 0:
                time.sleep(jitter / 1000.0)

            # Exclude circuit-open and previously denied bindings for this request.
            open_circuits = {
                bid: True
                for bid in meta_map
                if self._circuits.is_open(bid) or bid in excluded_bindings
            }
            plan_request = UsageRoutingRequest(
                required=request.required if request.required.entries else requested,
                unknown_limit_policy=request.unknown_limit_policy,
                stale_snapshot_policy=request.stale_snapshot_policy,
                preferred_binding_id=request.preferred_binding_id,
                preferred_provider_id=request.preferred_provider_id,
                preferred_scope_id=request.preferred_scope_id,
                affinity_binding_id=request.affinity_binding_id or origin,
                media_bytes=request.media_bytes,
                images=request.images,
                audio_seconds=request.audio_seconds,
                max_cost_micros=request.max_cost_micros,
                cost_currency=request.cost_currency,
                deadline_at=request.deadline_at
                or (_to_rfc3339(deadline_at) if deadline_at else None),
                now=_to_rfc3339(now),
                health_by_binding=request.health_by_binding,
                circuit_open_by_binding={
                    **dict(request.circuit_open_by_binding or {}),
                    **open_circuits,
                },
                latency_ms_by_binding=request.latency_ms_by_binding,
                queue_delay_ms_by_binding=request.queue_delay_ms_by_binding,
                quality_preference_by_binding=request.quality_preference_by_binding,
                locality_by_binding=request.locality_by_binding,
                require_snapshot=request.require_snapshot,
                reason_codes=request.reason_codes,
            )

            # Single-flight identical plan keys to avoid refresh herds.
            plan_key = stable_id(
                "plan",
                catalog_revision,
                request_id,
                str(index),
                sorted(snap_map.keys()),
            )

            def _plan() -> UsageAwareResolution:
                # Refresh snapshots under single-flight.
                refreshed = dict(snap_map)
                for sid in list(refreshed.keys()):
                    try:
                        refreshed[sid] = self._coord.snapshot(sid)
                    except Exception:
                        pass
                snap_map.update(refreshed)
                return plan_route(
                    catalog_revision=catalog_revision,
                    candidates=static_candidates,
                    snapshots_by_scope=snap_map,
                    policy=policy,
                    request=plan_request,
                    pin=pin,
                    meta_by_binding=meta_map,
                    origin_binding_id=origin,
                )

            resolution = self._single_flight.do(plan_key, _plan)
            next_eligible = resolution.next_eligible_at or next_eligible

            if not resolution.candidates:
                attempt_id = _attempt_id(request_id, index)
                denial = TypedDenial(
                    kind=DenialKind.NO_CANDIDATES,
                    reason_codes=tuple(resolution.reason_codes)
                    or ("no_eligible_candidates",),
                )
                # Wait vs fail when no candidates.
                next_dt = _parse_rfc3339(resolution.next_eligible_at)
                wait_decision = decide_wait_or_reroute(
                    policy=policy,
                    now=now,
                    next_eligible_at=next_dt,
                    deadline_at=deadline_at,
                    has_reroute_candidate=False,
                    attempts_used=index - start_attempt_index + 1,
                )
                receipt = self._build_receipt(
                    catalog_revision=catalog_revision,
                    usage_revision=resolution.usage_revision,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    idempotency_key="%s#%s" % (idempotency_key, attempt_id),
                    operation=operation,
                    policy=policy,
                    resolution=resolution,
                    estimate=None,
                    reservation=None,
                    observation=None,
                    settled=UsageVector(),
                    estimated=requested,
                    fallback_class=effective_fallback,
                    final_status=FinalStatus.CAPACITY_UNAVAILABLE.value
                    if wait_decision is not WaitOrReroute.WAIT
                    else FinalStatus.REJECTED.value,
                    reason_codes=denial.reason_codes,
                    caller_id=caller_id,
                    parent_attempt_id=parent,
                    chain_links=attempts,
                    created_at=_to_rfc3339(now),
                )
                attempt = RouteAttemptResult(
                    attempt_id=attempt_id,
                    parent_attempt_id=parent,
                    attempt_index=index,
                    binding_id=None,
                    scope_id=None,
                    reservation_id=None,
                    reservation=None,
                    decision=None,
                    denial=denial,
                    fallback_class=effective_fallback,
                    resolution=resolution,
                    estimate=None,
                    observation=None,
                    settlement=None,
                    error_class=ErrorSafetyClass.CAPACITY,
                    final_status=receipt.final_status,
                    reason_codes=denial.reason_codes,
                    receipt=receipt,
                    next_eligible_at=resolution.next_eligible_at,
                    usage_revision=resolution.usage_revision,
                    catalog_revision=catalog_revision,
                    jitter_ms=jitter,
                )
                attempts.append(attempt)
                terminal_receipt = receipt
                if wait_decision is WaitOrReroute.WAIT:
                    break
                break

            # Walk ranked candidates; advance only on typed denial.
            attempt_result: Optional[RouteAttemptResult] = None
            for cand in resolution.candidates:
                if cand.binding_id in excluded_bindings:
                    continue
                if self._circuits.is_open(cand.binding_id):
                    excluded_bindings.add(cand.binding_id)
                    continue

                # Fallback boundary relative to origin once one was selected.
                if origin and cand.binding_id in meta_map and origin in meta_map:
                    if not fallback_class_allows(
                        meta_map[origin], meta_map[cand.binding_id], effective_fallback
                    ):
                        continue

                scope_id = cand.scope_id
                attempt_id = _attempt_id(request_id, index)
                # Distinct idempotency per attempt+binding so retries never reuse.
                attempt_idem = stable_id(
                    "idem",
                    idempotency_key,
                    attempt_id,
                    cand.binding_id,
                )
                estimate = _estimate_for_scope(
                    scope_id,
                    operation,
                    requested,
                    estimated_at=_to_rfc3339(now),
                )

                decision = self._reserve_atomic(
                    scope_id=scope_id,
                    requested=requested,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    idempotency_key=attempt_idem,
                    estimate=estimate,
                    expected_usage_revision=None,
                )

                if not decision.granted:
                    denial = self._denial_from_decision(decision, cand)
                    excluded_bindings.add(cand.binding_id)
                    # Only capacity/stale/conflict are typed denials that advance.
                    if denial.kind in (
                        DenialKind.CAPACITY,
                        DenialKind.STALE_SNAPSHOT,
                        DenialKind.RESERVATION_CONFLICT,
                        DenialKind.CAS_EXHAUSTED,
                    ):
                        # Record partial denial and try next candidate.
                        attempt_result = RouteAttemptResult(
                            attempt_id=attempt_id,
                            parent_attempt_id=parent,
                            attempt_index=index,
                            binding_id=cand.binding_id,
                            scope_id=scope_id,
                            reservation_id=decision.reservation_id,
                            reservation=decision.reservation,
                            decision=decision,
                            denial=denial,
                            fallback_class=effective_fallback,
                            resolution=resolution,
                            estimate=estimate,
                            observation=None,
                            settlement=None,
                            error_class=ErrorSafetyClass.CAPACITY,
                            final_status=FinalStatus.REJECTED.value,
                            reason_codes=denial.reason_codes,
                            receipt=None,
                            next_eligible_at=decision.snapshot.next_eligible_at
                            if decision.snapshot
                            else None,
                            usage_revision=decision.usage_revision,
                            catalog_revision=catalog_revision,
                            jitter_ms=jitter,
                        )
                        # Continue to next candidate within same attempt index
                        # only after typed denial — do not invent a new attempt.
                        continue
                    # Policy/other: stop this candidate walk.
                    continue

                # Granted: close the race.  Optionally invoke.
                pre_invoke = RouteAttemptResult(
                    attempt_id=attempt_id,
                    parent_attempt_id=parent,
                    attempt_index=index,
                    binding_id=cand.binding_id,
                    scope_id=scope_id,
                    reservation_id=decision.reservation_id,
                    reservation=decision.reservation,
                    decision=decision,
                    denial=None,
                    fallback_class=effective_fallback,
                    resolution=resolution,
                    estimate=estimate,
                    observation=None,
                    settlement=None,
                    error_class=ErrorSafetyClass.SUCCESS,
                    final_status=FinalStatus.UNKNOWN.value,
                    reason_codes=("reserved",),
                    receipt=None,
                    next_eligible_at=None,
                    usage_revision=decision.usage_revision,
                    catalog_revision=catalog_revision,
                    jitter_ms=jitter,
                )

                if invoke is None:
                    # Return granted reservation for caller-owned dispatch.
                    receipt = self._build_receipt(
                        catalog_revision=catalog_revision,
                        usage_revision=decision.usage_revision,
                        request_id=request_id,
                        attempt_id=attempt_id,
                        idempotency_key=attempt_idem,
                        operation=operation,
                        policy=policy,
                        resolution=resolution,
                        estimate=estimate,
                        reservation=decision.reservation,
                        observation=None,
                        settled=UsageVector(),
                        estimated=requested,
                        fallback_class=effective_fallback,
                        final_status=FinalStatus.UNKNOWN.value,
                        reason_codes=("reserved",),
                        caller_id=caller_id,
                        parent_attempt_id=parent,
                        chain_links=attempts,
                        created_at=_to_rfc3339(self._coord.clock.now()),
                        selected_binding_id=cand.binding_id,
                        scope_id=scope_id,
                    )
                    granted = RouteAttemptResult(
                        attempt_id=attempt_id,
                        parent_attempt_id=parent,
                        attempt_index=index,
                        binding_id=cand.binding_id,
                        scope_id=scope_id,
                        reservation_id=decision.reservation_id,
                        reservation=decision.reservation,
                        decision=decision,
                        denial=None,
                        fallback_class=effective_fallback,
                        resolution=resolution,
                        estimate=estimate,
                        observation=None,
                        settlement=None,
                        error_class=ErrorSafetyClass.SUCCESS,
                        final_status=FinalStatus.UNKNOWN.value,
                        reason_codes=("reserved",),
                        receipt=receipt,
                        usage_revision=decision.usage_revision,
                        catalog_revision=catalog_revision,
                        jitter_ms=jitter,
                    )
                    attempts.append(granted)
                    chain = self._chain_from_attempts(attempts)
                    return RouteAdmissionResult(
                        success=True,
                        attempts=tuple(attempts),
                        selected=granted,
                        chain=chain,
                        receipt=receipt,
                        final_status=FinalStatus.UNKNOWN.value,
                        reason_codes=("reserved",),
                        next_eligible_at=next_eligible,
                    )

                # Mark dispatched then invoke.
                try:
                    self._coord.mark_dispatched(decision.reservation_id)
                except (LedgerError, CompareAndSetConflict):
                    pass

                outcome = invoke(pre_invoke)
                if not isinstance(outcome, InvokeOutcome):
                    raise RoutingError("invoke callback must return InvokeOutcome")

                settlement: Optional[SettlementResult] = None
                final_status = FinalStatus.FAILED.value
                error_class = outcome.error_class
                reason_codes = list(outcome.reason_codes) or list(pre_invoke.reason_codes)

                if outcome.success:
                    self._circuits.record_success(cand.binding_id)
                    if settle_on_success:
                        try:
                            settlement = self._coord.commit(
                                decision.reservation_id,
                                outcome.settled
                                or (
                                    outcome.observation.usage
                                    if outcome.observation
                                    else None
                                ),
                                observation_id=(
                                    outcome.observation.observation_id
                                    if outcome.observation
                                    else None
                                ),
                            )
                            final_status = FinalStatus.COMMITTED.value
                            reason_codes.append("committed")
                        except (LedgerError, CompareAndSetConflict) as exc:
                            final_status = FinalStatus.FAILED.value
                            reason_codes.append("settle_failed")
                            error_class = ErrorSafetyClass.UNKNOWN
                    else:
                        final_status = FinalStatus.UNKNOWN.value
                        reason_codes.append("reserved_awaiting_settle")
                    if outcome.observation is not None:
                        try:
                            self._coord.append_observation(
                                scope_id,
                                kind=UsageEventKind.OBSERVATION_SUCCESS,
                                units=outcome.observation.usage,
                                reservation_id=decision.reservation_id,
                                observation_id=outcome.observation.observation_id,
                                request_id=request_id,
                                reason_codes=outcome.observation.reason_codes,
                                limits_update=outcome.observation.limits or None,
                            )
                        except (LedgerError, CompareAndSetConflict):
                            pass
                else:
                    # Failure path: classify safety for fallback.
                    if outcome.observation is not None:
                        error_class = classify_invoke_error(
                            http_status=outcome.observation.http_status,
                            reason_codes=outcome.observation.reason_codes
                            or outcome.reason_codes,
                            side_effecting=outcome.side_effecting,
                        )
                    elif outcome.side_effecting:
                        error_class = ErrorSafetyClass.SIDE_EFFECT
                    self._circuits.record_failure(cand.binding_id)
                    if outcome.observation is not None:
                        cooldown = None
                        if outcome.observation.retry_after_ms:
                            cd = self._coord.clock.now() + timedelta(
                                milliseconds=int(outcome.observation.retry_after_ms)
                            )
                            cooldown = _to_rfc3339(cd)
                        try:
                            self._coord.append_observation(
                                scope_id,
                                kind=UsageEventKind.OBSERVATION_FAILURE,
                                units=outcome.observation.usage,
                                reservation_id=decision.reservation_id,
                                observation_id=outcome.observation.observation_id,
                                request_id=request_id,
                                reason_codes=outcome.observation.reason_codes
                                or outcome.reason_codes,
                                cooldown_until=cooldown,
                                limits_update=outcome.observation.limits or None,
                            )
                        except (LedgerError, CompareAndSetConflict):
                            pass
                    if release_on_failure:
                        try:
                            settlement = self._coord.cancel(
                                decision.reservation_id, reason="invoke_failed"
                            )
                            if settlement.state.value == "released":
                                final_status = FinalStatus.RELEASED.value
                            else:
                                final_status = FinalStatus.COMMITTED.value
                        except (LedgerError, CompareAndSetConflict):
                            final_status = FinalStatus.FAILED.value
                    else:
                        final_status = FinalStatus.FAILED.value
                    reason_codes.append("invoke_failed")
                    reason_codes.append(error_class.value)

                receipt = self._build_receipt(
                    catalog_revision=catalog_revision,
                    usage_revision=decision.usage_revision,
                    request_id=request_id,
                    attempt_id=attempt_id,
                    idempotency_key=attempt_idem,
                    operation=operation,
                    policy=policy,
                    resolution=resolution,
                    estimate=estimate,
                    reservation=decision.reservation,
                    observation=outcome.observation,
                    settled=(
                        settlement.charged
                        if settlement is not None
                        else (outcome.settled or UsageVector())
                    ),
                    estimated=requested,
                    fallback_class=effective_fallback,
                    final_status=final_status,
                    reason_codes=tuple(reason_codes),
                    caller_id=caller_id,
                    parent_attempt_id=parent,
                    chain_links=attempts,
                    created_at=_to_rfc3339(self._coord.clock.now()),
                    selected_binding_id=cand.binding_id,
                    scope_id=scope_id,
                )
                attempt_result = RouteAttemptResult(
                    attempt_id=attempt_id,
                    parent_attempt_id=parent,
                    attempt_index=index,
                    binding_id=cand.binding_id,
                    scope_id=scope_id,
                    reservation_id=decision.reservation_id,
                    reservation=decision.reservation,
                    decision=decision,
                    denial=None if outcome.success else TypedDenial(
                        kind=DenialKind.CAPACITY
                        if error_class is ErrorSafetyClass.CAPACITY
                        else DenialKind.UNKNOWN,
                        reason_codes=tuple(reason_codes),
                        binding_id=cand.binding_id,
                        scope_id=scope_id,
                    )
                    if not outcome.success
                    else None,
                    fallback_class=effective_fallback,
                    resolution=resolution,
                    estimate=estimate,
                    observation=outcome.observation,
                    settlement=settlement,
                    error_class=error_class,
                    final_status=final_status,
                    reason_codes=tuple(sorted(set(reason_codes))),
                    receipt=receipt,
                    usage_revision=decision.usage_revision,
                    catalog_revision=catalog_revision,
                    jitter_ms=jitter,
                )
                attempts.append(attempt_result)
                terminal_receipt = receipt

                if outcome.success:
                    chain = self._chain_from_attempts(attempts)
                    return RouteAdmissionResult(
                        success=True,
                        attempts=tuple(attempts),
                        selected=attempt_result,
                        chain=chain,
                        receipt=receipt,
                        final_status=final_status,
                        reason_codes=attempt_result.reason_codes,
                        next_eligible_at=next_eligible,
                    )

                # Failure: only safe classes may fallback / retry.
                excluded_bindings.add(cand.binding_id)
                if not is_fallback_safe(error_class):
                    # Never fallback on semantic/client/side-effect.
                    chain = self._chain_from_attempts(attempts)
                    return RouteAdmissionResult(
                        success=False,
                        attempts=tuple(attempts),
                        selected=attempt_result,
                        chain=chain,
                        receipt=receipt,
                        final_status=final_status,
                        reason_codes=tuple(
                            sorted(
                                set(attempt_result.reason_codes)
                                | {"no_fallback_unsafe_error"}
                            )
                        ),
                        next_eligible_at=next_eligible,
                    )

                # Safe to try next candidate within attempt? Policy uses new
                # attempt for each retry/fallback with linked parent.
                origin = origin or cand.binding_id
                parent = attempt_id
                break  # advance to next attempt index for fallback
            else:
                # Exhausted candidates without a grant.
                if attempt_result is not None and attempt_result.denial is not None:
                    # Last denial attempt recorded.
                    if attempt_result not in attempts:
                        # Build receipt for last denial.
                        receipt = self._build_receipt(
                            catalog_revision=catalog_revision,
                            usage_revision=attempt_result.usage_revision
                            or USAGE_REVISION_OFF,
                            request_id=request_id,
                            attempt_id=attempt_result.attempt_id,
                            idempotency_key="%s#%s"
                            % (idempotency_key, attempt_result.attempt_id),
                            operation=operation,
                            policy=policy,
                            resolution=resolution,
                            estimate=attempt_result.estimate,
                            reservation=attempt_result.reservation,
                            observation=None,
                            settled=UsageVector(),
                            estimated=requested,
                            fallback_class=effective_fallback,
                            final_status=FinalStatus.REJECTED.value,
                            reason_codes=attempt_result.reason_codes,
                            caller_id=caller_id,
                            parent_attempt_id=parent,
                            chain_links=attempts,
                            created_at=_to_rfc3339(self._coord.clock.now()),
                            selected_binding_id=attempt_result.binding_id,
                            scope_id=attempt_result.scope_id,
                        )
                        attempt_result = RouteAttemptResult(
                            attempt_id=attempt_result.attempt_id,
                            parent_attempt_id=attempt_result.parent_attempt_id,
                            attempt_index=attempt_result.attempt_index,
                            binding_id=attempt_result.binding_id,
                            scope_id=attempt_result.scope_id,
                            reservation_id=attempt_result.reservation_id,
                            reservation=attempt_result.reservation,
                            decision=attempt_result.decision,
                            denial=attempt_result.denial,
                            fallback_class=attempt_result.fallback_class,
                            resolution=attempt_result.resolution,
                            estimate=attempt_result.estimate,
                            observation=None,
                            settlement=None,
                            error_class=attempt_result.error_class,
                            final_status=FinalStatus.REJECTED.value,
                            reason_codes=attempt_result.reason_codes,
                            receipt=receipt,
                            next_eligible_at=attempt_result.next_eligible_at,
                            usage_revision=attempt_result.usage_revision,
                            catalog_revision=catalog_revision,
                            jitter_ms=jitter,
                        )
                        attempts.append(attempt_result)
                        terminal_receipt = receipt
                        parent = attempt_result.attempt_id
                else:
                    # No candidate could be reserved.
                    attempt_id = _attempt_id(request_id, index)
                    denial = TypedDenial(
                        kind=DenialKind.CAPACITY,
                        reason_codes=("all_candidates_denied",),
                    )
                    receipt = self._build_receipt(
                        catalog_revision=catalog_revision,
                        usage_revision=resolution.usage_revision,
                        request_id=request_id,
                        attempt_id=attempt_id,
                        idempotency_key="%s#%s" % (idempotency_key, attempt_id),
                        operation=operation,
                        policy=policy,
                        resolution=resolution,
                        estimate=None,
                        reservation=None,
                        observation=None,
                        settled=UsageVector(),
                        estimated=requested,
                        fallback_class=effective_fallback,
                        final_status=FinalStatus.CAPACITY_UNAVAILABLE.value,
                        reason_codes=denial.reason_codes,
                        caller_id=caller_id,
                        parent_attempt_id=parent,
                        chain_links=attempts,
                        created_at=_to_rfc3339(now),
                    )
                    attempt_result = RouteAttemptResult(
                        attempt_id=attempt_id,
                        parent_attempt_id=parent,
                        attempt_index=index,
                        binding_id=None,
                        scope_id=None,
                        reservation_id=None,
                        reservation=None,
                        decision=None,
                        denial=denial,
                        fallback_class=effective_fallback,
                        resolution=resolution,
                        estimate=None,
                        observation=None,
                        settlement=None,
                        error_class=ErrorSafetyClass.CAPACITY,
                        final_status=receipt.final_status,
                        reason_codes=denial.reason_codes,
                        receipt=receipt,
                        next_eligible_at=resolution.next_eligible_at,
                        usage_revision=resolution.usage_revision,
                        catalog_revision=catalog_revision,
                        jitter_ms=jitter,
                    )
                    attempts.append(attempt_result)
                    terminal_receipt = receipt
                    parent = attempt_id

                # Decide wait vs fail for next loop iteration.
                next_dt = _parse_rfc3339(next_eligible)
                remaining_candidates = [
                    c
                    for c in resolution.candidates
                    if c.binding_id not in excluded_bindings
                ]
                wait_decision = decide_wait_or_reroute(
                    policy=policy,
                    now=self._coord.clock.now(),
                    next_eligible_at=next_dt,
                    deadline_at=deadline_at,
                    has_reroute_candidate=bool(remaining_candidates)
                    and effective_fallback is not FallbackClass.NONE,
                    attempts_used=index - start_attempt_index + 1,
                )
                if wait_decision is WaitOrReroute.FAIL:
                    break
                if wait_decision is WaitOrReroute.WAIT:
                    break
                # REROUTE continues the loop with a new attempt.
                continue

            # If we broke from candidate loop after a safe failure, continue
            # outer attempt loop only when fallback permits and attempts remain.
            if attempt_result and not attempt_result.granted:
                if effective_fallback is FallbackClass.NONE:
                    break
                if not is_fallback_safe(attempt_result.error_class):
                    break
                # Linked next attempt.
                continue

        chain = self._chain_from_attempts(attempts)
        selected = attempts[-1] if attempts else None
        final_status = (
            selected.final_status
            if selected is not None
            else FinalStatus.CAPACITY_UNAVAILABLE.value
        )
        reasons: List[str] = []
        for item in attempts:
            reasons.extend(item.reason_codes)
        if not attempts:
            reasons.append("no_attempts")
            final_status = FinalStatus.CAPACITY_UNAVAILABLE.value
        if wait_decision is WaitOrReroute.WAIT:
            reasons.append("wait_for_capacity")
        if wait_decision is WaitOrReroute.FAIL:
            reasons.append("deadline_or_attempt_bound")

        return RouteAdmissionResult(
            success=False,
            attempts=tuple(attempts),
            selected=selected,
            chain=chain,
            receipt=terminal_receipt or (selected.receipt if selected else None),
            final_status=final_status,
            reason_codes=tuple(sorted(set(reasons))),
            wait_or_reroute=wait_decision,
            next_eligible_at=next_eligible,
        )

    def _reserve_atomic(
        self,
        *,
        scope_id: str,
        requested: UsageVector,
        request_id: str,
        attempt_id: str,
        idempotency_key: str,
        estimate: UsageEstimate,
        expected_usage_revision: Optional[str],
    ) -> ReserveDecision:
        """Atomic reserve; maps coordinator errors to typed denials."""

        try:
            return self._coord.reserve(
                scope_id,
                requested,
                request_id=request_id,
                attempt_id=attempt_id,
                idempotency_key=idempotency_key,
                owner_id=self._owner_id,
                estimate=estimate,
                expected_usage_revision=expected_usage_revision,
                ttl_ms=self._ttl_ms,
            )
        except StaleSnapshot as exc:
            snap = self._coord.snapshot(scope_id)
            return ReserveDecision(
                granted=False,
                reservation=None,
                reservation_id=None,
                usage_revision=snap.usage_revision or "",
                snapshot=snap,
                reason_codes=("stale_snapshot",),
                error_code=UsageErrorCode.STALE_SNAPSHOT.value,
            )
        except CapacityDenied as exc:
            snap = self._coord.snapshot(scope_id)
            return ReserveDecision(
                granted=False,
                reservation=None,
                reservation_id=None,
                usage_revision=snap.usage_revision or "",
                snapshot=snap,
                reason_codes=tuple(exc.reason_codes)
                if hasattr(exc, "reason_codes")
                else ("limit_exhausted",),
                error_code=getattr(exc, "code", UsageErrorCode.LIMIT_EXHAUSTED.value),
            )
        except CompareAndSetConflict:
            snap = self._coord.snapshot(scope_id)
            return ReserveDecision(
                granted=False,
                reservation=None,
                reservation_id=None,
                usage_revision=snap.usage_revision or "",
                snapshot=snap,
                reason_codes=("cas_exhausted",),
                error_code=UsageErrorCode.RESERVATION_CONFLICT.value,
            )
        except LedgerError as exc:
            snap = self._coord.snapshot(scope_id)
            return ReserveDecision(
                granted=False,
                reservation=None,
                reservation_id=None,
                usage_revision=snap.usage_revision or "",
                snapshot=snap,
                reason_codes=tuple(getattr(exc, "reason_codes", ()) or ("ledger_error",)),
                error_code=getattr(exc, "code", UsageErrorCode.RESERVATION_CONFLICT.value),
            )

    def _denial_from_decision(
        self, decision: ReserveDecision, cand: ResolutionCandidate
    ) -> TypedDenial:
        codes = tuple(decision.reason_codes or ())
        error = decision.error_code
        if error == UsageErrorCode.STALE_SNAPSHOT.value or "stale_snapshot" in codes:
            kind = DenialKind.STALE_SNAPSHOT
        elif error == UsageErrorCode.RESERVATION_CONFLICT.value or "cas_exhausted" in codes:
            kind = (
                DenialKind.CAS_EXHAUSTED
                if "cas_exhausted" in codes
                else DenialKind.RESERVATION_CONFLICT
            )
        elif error == UsageErrorCode.LIMIT_EXHAUSTED.value or any(
            "exhaust" in c or "insufficient" in c or "limit" in c for c in codes
        ):
            kind = DenialKind.CAPACITY
        else:
            kind = DenialKind.CAPACITY if not decision.granted else DenialKind.UNKNOWN
        return TypedDenial(
            kind=kind,
            reason_codes=codes or ("denied",),
            binding_id=cand.binding_id,
            scope_id=cand.scope_id,
            error_code=error,
        )

    def _chain_from_attempts(
        self, attempts: Sequence[RouteAttemptResult]
    ) -> ReceiptChain:
        links: List[AttemptLink] = []
        for item in attempts:
            links.append(
                AttemptLink(
                    attempt_id=item.attempt_id,
                    parent_attempt_id=item.parent_attempt_id,
                    reservation_id=item.reservation_id,
                    binding_id=item.binding_id,
                    scope_id=item.scope_id,
                    fallback_class=item.fallback_class,
                    denial_kind=item.denial.kind.value if item.denial else None,
                    reason_codes=item.reason_codes,
                    final_status=item.final_status,
                    created_at=None,
                )
            )
        if not links:
            return ReceiptChain(links=())
        return build_receipt_chain(links)

    def _build_receipt(
        self,
        *,
        catalog_revision: str,
        usage_revision: str,
        request_id: str,
        attempt_id: str,
        idempotency_key: str,
        operation: str,
        policy: RoutingPolicy,
        resolution: Optional[UsageAwareResolution],
        estimate: Optional[UsageEstimate],
        reservation: Optional[UsageReservation],
        observation: Optional[ProviderUsageObservation],
        settled: UsageVector,
        estimated: UsageVector,
        fallback_class: FallbackClass,
        final_status: str,
        reason_codes: Sequence[str],
        caller_id: Optional[str],
        parent_attempt_id: Optional[str],
        chain_links: Sequence[RouteAttemptResult],
        created_at: Optional[str],
        selected_binding_id: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> UsageRoutingReceipt:
        hard_dig = (
            hard_rejection_digest(resolution.rejected) if resolution is not None else None
        )
        rank_dig = (
            ranking_inputs_digest(resolution.candidates)
            if resolution is not None
            else None
        )
        cand_dig = (
            candidates_digest(
                resolution.candidates if resolution else None,
                resolution.rejected if resolution else None,
            )
            if resolution is not None
            else None
        )
        # Include current attempt as the last chain link when building.
        provisional = list(chain_links)
        chain = self._chain_from_attempts(provisional)
        # Extend chain with the current attempt identity if not already present.
        if not any(link.attempt_id == attempt_id for link in chain.links):
            extended = list(chain.links) + [
                AttemptLink(
                    attempt_id=attempt_id,
                    parent_attempt_id=parent_attempt_id,
                    reservation_id=reservation.reservation_id if reservation else None,
                    binding_id=selected_binding_id,
                    scope_id=scope_id,
                    fallback_class=fallback_class,
                    reason_codes=tuple(reason_codes),
                    final_status=final_status,
                    created_at=created_at,
                )
            ]
            chain = build_receipt_chain(extended)

        draft = RouteReceiptDraft(
            catalog_revision=catalog_revision,
            usage_revision=usage_revision or USAGE_REVISION_OFF,
            request_id=request_id,
            attempt_id=attempt_id,
            idempotency_key=idempotency_key,
            operation=operation,
            policy_id=policy.policy_id,
            resolution_id=resolution.resolution_id if resolution else None,
            selected_binding_id=selected_binding_id,
            scope_id=scope_id,
            reservation_id=reservation.reservation_id if reservation else None,
            estimate_id=estimate.estimate_id if estimate else None,
            observation_id=observation.observation_id if observation else None,
            caller_id=caller_id,
            estimated=estimated,
            settled=settled,
            fallback_class=fallback_class,
            final_status=final_status,
            next_eligible_at=resolution.next_eligible_at if resolution else None,
            reason_codes=tuple(reason_codes),
            created_at=created_at,
            hard_rejection_digest=hard_dig,
            ranking_inputs_digest=rank_dig,
            candidates_digest=cand_dig,
            chain=chain,
            candidate_count=len(resolution.candidates) if resolution else 0,
            rejected_count=len(resolution.rejected) if resolution else 0,
        )
        return build_usage_routing_receipt(
            draft,
            resolution=resolution,
            estimate=estimate,
            reservation=reservation,
            observation=observation,
        )


def score_cannot_bypass_hard_gate(
    candidate: StaticCandidate,
    snapshot: Optional[UsageSnapshot],
    request: UsageRoutingRequest,
    policy: RoutingPolicy,
    *,
    now: Optional[datetime] = None,
) -> bool:
    """Return True when hard gates reject *candidate* regardless of score.

    Used by tests and callers to prove score cannot offset hard limits,
    capability/authorization proxies, cost, media, or deadline constraints.
    """

    ok, _reasons, _headroom = hard_filter_candidate(
        candidate, snapshot, request, policy, now=now
    )
    return not ok


__all__ = [
    "ROUTE_ADMISSION_REQUIREMENT_ID",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_CIRCUIT_FAILURE_THRESHOLD",
    "DEFAULT_CIRCUIT_COOLDOWN_MS",
    "DEFAULT_JITTER_MAX_MS",
    "MAX_ATTEMPT_BOUND",
    "RoutingError",
    "DenialKind",
    "ErrorSafetyClass",
    "WaitOrReroute",
    "RoutePin",
    "RouteCandidateMeta",
    "meta_from_static",
    "fallback_class_allows",
    "apply_pin_filter",
    "is_fallback_safe",
    "classify_invoke_error",
    "admission_jitter_ms",
    "decide_wait_or_reroute",
    "CircuitState",
    "CircuitBreakerRegistry",
    "SingleFlight",
    "TypedDenial",
    "InvokeOutcome",
    "RouteAttemptResult",
    "RouteAdmissionResult",
    "SnapshotProvider",
    "plan_route",
    "UsageRouteAdmission",
    "score_cannot_bypass_hard_gate",
]
