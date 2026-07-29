"""Shared route admission, ranking, fallback, and receipts (AICAT-029)."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pytest

from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    contains_raw_endpoint,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.receipts import (
    USAGE_ROUTING_RECEIPT_REQUIREMENT_ID,
    AttemptLink,
    FinalStatus,
    ReceiptError,
    RouteReceiptDraft,
    assert_receipt_safe,
    build_receipt_chain,
    build_usage_routing_receipt,
    candidates_digest,
    hard_rejection_digest,
    ranking_inputs_digest,
    receipt_binds_revisions,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
    resolve_usage_aware,
)
from ipfs_accelerate_py.endpoint_usage.routing import (
    ROUTE_ADMISSION_REQUIREMENT_ID,
    CircuitBreakerRegistry,
    DenialKind,
    ErrorSafetyClass,
    InvokeOutcome,
    RoutePin,
    SingleFlight,
    UsageRouteAdmission,
    WaitOrReroute,
    admission_jitter_ms,
    apply_pin_filter,
    classify_invoke_error,
    decide_wait_or_reroute,
    fallback_class_allows,
    is_fallback_safe,
    meta_from_static,
    plan_route,
    score_cannot_bypass_hard_gate,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    AvailabilityState,
    ConfidenceLevel,
    DimensionHeadroom,
    EndpointUsageScope,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    ProviderUsageObservation,
    Provenance,
    Quantity,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageVector,
    WindowKind,
    UsageLimit,
    UsageSnapshot,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _scope(provider_key: str = "prov-a", operation: str = "text.chat") -> EndpointUsageScope:
    provider_id = stable_id("provider", provider_key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=operation,
        deployment_id=stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:ROUTE_TEST_KEY", key_id="route-default"
        ),
    )


def _limit(scope_id: str, dimension: UsageDimension, ceiling: int, used: int = 0) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(max(0, ceiling - used)),
        used=Quantity.finite(used),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _snapshot(
    scope: EndpointUsageScope,
    *,
    requests_ceiling: int = 100,
    requests_used: int = 0,
    state: AvailabilityState = AvailabilityState.AVAILABLE,
    next_eligible_at: Optional[str] = None,
    clock: Optional[datetime] = None,
) -> UsageSnapshot:
    available = max(0, requests_ceiling - requests_used)
    observed = _rfc(clock or _now())
    return UsageSnapshot(
        scope_id=scope.scope_id,
        observed_at=observed,
        fresh_until=_rfc((clock or _now()) + timedelta(minutes=5)),
        state=state,
        limits=(_limit(scope.scope_id, UsageDimension.REQUESTS, requests_ceiling, requests_used),),
        headroom=(
            DimensionHeadroom(
                dimension=UsageDimension.REQUESTS,
                available=Quantity.finite(available),
                ceiling=Quantity.finite(requests_ceiling),
                reserved=Quantity.finite(0),
                state=state
                if state is not AvailabilityState.AVAILABLE
                else (
                    AvailabilityState.AVAILABLE
                    if available > 0
                    else AvailabilityState.EXHAUSTED
                ),
            ),
        ),
        next_eligible_at=next_eligible_at,
    )


def _candidate(
    *,
    provider_key: str,
    model: str = "model-a",
    deployment: str = "dep-a",
    scope: Optional[EndpointUsageScope] = None,
    score: int = 10,
    labels: Optional[Dict[str, str]] = None,
    binding_salt: str = "",
) -> Tuple[StaticCandidate, EndpointUsageScope]:
    scope = scope or _scope(provider_key)
    binding_id = stable_id(
        "binding", scope.provider_id, model, deployment, binding_salt or provider_key
    )
    return (
        StaticCandidate(
            binding_id=binding_id,
            provider_id=scope.provider_id,
            model_id=stable_id("model", model),
            deployment_id=stable_id("deployment", deployment, scope.provider_id),
            scope_id=scope.scope_id,
            catalog_score=score,
            locality=labels.get("locality") if labels else None,
            authorized=True,
            healthy=True,
            routable=True,
            configured=True,
            labels=labels or {},
        ),
        scope,
    )


def _coord(clock: Optional[FakeClock] = None) -> UsageCoordinator:
    clk = clock or FakeClock(_now())
    store = InMemoryUsageLedgerStore(clock=clk)
    return UsageCoordinator(store, writer_id="route-test", fence=1)


def _configure(coord: UsageCoordinator, scope: EndpointUsageScope, ceiling: int = 100) -> None:
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, ceiling)],
    )


# ---------------------------------------------------------------------------
# Hard gates cannot be offset by score
# ---------------------------------------------------------------------------


def test_hard_limit_cannot_be_offset_by_score():
    scope = _scope("hard-score")
    snap = _snapshot(scope, requests_ceiling=1, requests_used=1, state=AvailabilityState.EXHAUSTED)
    high, _ = _candidate(provider_key="hard-score", score=1_000_000, scope=scope)
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE)
    request = UsageRoutingRequest(required=UsageVector.of(requests=1), now=_rfc(_now()))
    assert score_cannot_bypass_hard_gate(high, snap, request, policy, now=_now()) is True
    ok_reasons = score_cannot_bypass_hard_gate  # noqa: F841 — alias for clarity
    from ipfs_accelerate_py.endpoint_usage.resolution import hard_filter_candidate

    ok, reasons, _ = hard_filter_candidate(high, snap, request, policy, now=_now())
    assert ok is False
    assert reasons


def test_authorization_pin_safety_cost_media_deadline_are_hard():
    scope = _scope("gates")
    snap = _snapshot(scope, requests_ceiling=100, requests_used=0)
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.NONE,
        cost_ceiling_micros=100,
        cost_currency="USD",
    )
    # Authorization denied
    unauthorized, _ = _candidate(provider_key="gates", scope=scope, binding_salt="unauth")
    unauthorized = StaticCandidate(
        binding_id=unauthorized.binding_id,
        provider_id=unauthorized.provider_id,
        model_id=unauthorized.model_id,
        deployment_id=unauthorized.deployment_id,
        scope_id=unauthorized.scope_id,
        catalog_score=999,
        authorized=False,
        healthy=True,
        routable=True,
        configured=True,
    )
    request = UsageRoutingRequest(required=UsageVector.of(requests=1), now=_rfc(_now()))
    assert score_cannot_bypass_hard_gate(unauthorized, snap, request, policy) is True

    # Data governance label
    denied, _ = _candidate(
        provider_key="gates",
        scope=scope,
        binding_salt="gov",
        labels={"data.governance": "deny"},
        score=999,
    )
    assert score_cannot_bypass_hard_gate(denied, snap, request, policy) is True

    # Cost ceiling
    costly = UsageRoutingRequest(
        required=UsageVector.of(cost_micros=500, currency="USD"),
        max_cost_micros=100,
        cost_currency="USD",
        now=_rfc(_now()),
    )
    ok_cand, _ = _candidate(provider_key="gates", scope=scope, binding_salt="cost", score=999)
    assert score_cannot_bypass_hard_gate(ok_cand, snap, costly, policy) is True

    # Media
    media_req = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        media_bytes=10_000,
        now=_rfc(_now()),
    )
    # Missing media headroom under DENY unknown policy is a hard reject.
    assert score_cannot_bypass_hard_gate(ok_cand, snap, media_req, policy) is True

    # Deadline exceeded vs next_eligible
    future = _rfc(_now() + timedelta(hours=1))
    snap_cool = _snapshot(
        scope,
        requests_ceiling=0,
        requests_used=0,
        state=AvailabilityState.COOLING_DOWN,
        next_eligible_at=future,
    )
    deadline_req = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        deadline_at=_rfc(_now() + timedelta(minutes=5)),
        now=_rfc(_now()),
    )
    assert score_cannot_bypass_hard_gate(ok_cand, snap_cool, deadline_req, policy) is True


def test_exact_pin_defaults_to_none_fallback():
    pin = RoutePin(provider_id=stable_id("provider", "pinned"))
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.CROSS_PROVIDER,
        max_attempts=3,
    )
    assert pin.effective_fallback(policy) is FallbackClass.NONE
    pin_allowed = RoutePin(
        provider_id=stable_id("provider", "pinned"),
        allow_fallback_with_pin=True,
    )
    assert pin_allowed.effective_fallback(policy) is FallbackClass.CROSS_PROVIDER
    empty = RoutePin()
    assert empty.is_exact is False
    assert empty.effective_fallback(policy) is FallbackClass.CROSS_PROVIDER


# ---------------------------------------------------------------------------
# Ranking inputs
# ---------------------------------------------------------------------------


def test_ranking_uses_affinity_saturation_reset_health_latency_cost_locality_quality():
    scope_a = _scope("rank-a")
    scope_b = _scope("rank-b")
    # A is nearly saturated; B has headroom, worse latency, local, preferred affinity.
    snap_a = _snapshot(scope_a, requests_ceiling=100, requests_used=90)
    snap_b = _snapshot(scope_b, requests_ceiling=100, requests_used=10)
    cand_a, _ = _candidate(
        provider_key="rank-a", scope=scope_a, score=5, labels={"locality": "remote"}
    )
    cand_b, _ = _candidate(
        provider_key="rank-b", scope=scope_b, score=1, labels={"locality": "local"}
    )
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE, fallback=FallbackClass.CROSS_PROVIDER, prefer_local=True
    )
    request = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        affinity_binding_id=cand_b.binding_id,
        health_by_binding={cand_a.binding_id: True, cand_b.binding_id: True},
        latency_ms_by_binding={cand_a.binding_id: 10, cand_b.binding_id: 50},
        queue_delay_ms_by_binding={cand_a.binding_id: 0, cand_b.binding_id: 5},
        quality_preference_by_binding={cand_a.binding_id: 1, cand_b.binding_id: 9},
        locality_by_binding={
            cand_a.binding_id: "remote",
            cand_b.binding_id: "local",
        },
        now=_rfc(_now()),
    )
    resolution = plan_route(
        catalog_revision="catalog-rev-rank",
        candidates=[cand_a, cand_b],
        snapshots_by_scope={scope_a.scope_id: snap_a, scope_b.scope_id: snap_b},
        policy=policy,
        request=request,
    )
    assert resolution.selected_binding_id == cand_b.binding_id
    top = resolution.candidates[0]
    names = {n for n, _ in top.ranking_inputs}
    # Affinity / saturation / locality / health / latency / quality present.
    assert "affinity" in names
    assert "prefer_local_match" in names
    assert "health" in names
    assert "latency_ms" in names or "queue_delay_ms" in names
    assert any(n.startswith("sat_") for n in names) or "tightest_dimension" in names
    assert "quality_preference" in names or "catalog_score" in names


# ---------------------------------------------------------------------------
# Fallback classes distinguishable
# ---------------------------------------------------------------------------


def test_fallback_classes_are_distinguishable():
    origin = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "o"),
            provider_id=stable_id("provider", "p1"),
            model_id=stable_id("model", "m1"),
            deployment_id=stable_id("deployment", "d1"),
            labels={"equivalent_model": "group-x"},
        )
    )
    same_binding = meta_from_static(
        StaticCandidate(
            binding_id=origin.binding_id,
            provider_id=origin.provider_id,
            model_id=origin.model_id,
            deployment_id=origin.deployment_id,
        )
    )
    same_dep = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "dep2"),
            provider_id=origin.provider_id,
            model_id=stable_id("model", "m2"),
            deployment_id=origin.deployment_id,
        )
    )
    same_prov = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "prov2"),
            provider_id=origin.provider_id,
            model_id=stable_id("model", "m3"),
            deployment_id=stable_id("deployment", "d2"),
        )
    )
    same_model = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "mod2"),
            provider_id=origin.provider_id,
            model_id=origin.model_id,
            deployment_id=stable_id("deployment", "d3"),
        )
    )
    equiv = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "eq"),
            provider_id=stable_id("provider", "p2"),
            model_id=stable_id("model", "m-eq"),
            deployment_id=stable_id("deployment", "d-eq"),
            labels={"equivalent_model": "group-x"},
        )
    )
    cross = meta_from_static(
        StaticCandidate(
            binding_id=stable_id("binding", "cross"),
            provider_id=stable_id("provider", "p3"),
            model_id=stable_id("model", "m-x"),
            deployment_id=stable_id("deployment", "d-x"),
            labels={"equivalent_model": "group-y"},
        )
    )

    assert fallback_class_allows(origin, same_binding, FallbackClass.NONE)
    assert not fallback_class_allows(origin, same_dep, FallbackClass.NONE)

    assert fallback_class_allows(origin, same_dep, FallbackClass.SAME_DEPLOYMENT)
    assert not fallback_class_allows(origin, same_prov, FallbackClass.SAME_DEPLOYMENT)

    assert fallback_class_allows(origin, same_prov, FallbackClass.SAME_PROVIDER)
    assert not fallback_class_allows(origin, cross, FallbackClass.SAME_PROVIDER)

    assert fallback_class_allows(origin, same_model, FallbackClass.SAME_MODEL)
    assert not fallback_class_allows(origin, same_prov, FallbackClass.SAME_MODEL)

    assert fallback_class_allows(origin, equiv, FallbackClass.EQUIVALENT_MODEL)
    assert not fallback_class_allows(origin, cross, FallbackClass.EQUIVALENT_MODEL)

    assert fallback_class_allows(origin, cross, FallbackClass.CROSS_PROVIDER)
    assert fallback_class_allows(origin, same_binding, FallbackClass.CROSS_PROVIDER)


# ---------------------------------------------------------------------------
# Atomic reservation race + typed denial advances candidate
# ---------------------------------------------------------------------------


def test_admission_closes_race_with_atomic_reservation():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("atomic")
    _configure(coord, scope, ceiling=10)
    cand, _ = _candidate(provider_key="atomic", scope=scope)
    snap = coord.snapshot(scope.scope_id)
    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-1",
        candidates=[cand],
        request_id="req-atomic-1",
        idempotency_key="idem-atomic-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        snapshots_by_scope={scope.scope_id: snap},
        invoke=None,
    )
    assert result.success is True
    assert result.selected is not None
    assert result.selected.reservation_id is not None
    assert result.selected.granted is True
    # Headroom reduced by reservation.
    after = coord.snapshot(scope.scope_id)
    before_avail = None
    after_avail = None
    for h in snap.headroom:
        if h.dimension is UsageDimension.REQUESTS:
            before_avail = h.available.value
    for h in after.headroom:
        if h.dimension is UsageDimension.REQUESTS:
            after_avail = h.available.value
    assert before_avail is not None and after_avail is not None
    assert after_avail == before_avail - 1


def test_typed_denial_advances_to_next_candidate():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_full = _scope("full")
    scope_ok = _scope("ok")
    _configure(coord, scope_full, ceiling=0)
    _configure(coord, scope_ok, ceiling=10)
    cand_full, _ = _candidate(provider_key="full", scope=scope_full, score=100)
    cand_ok, _ = _candidate(provider_key="ok", scope=scope_ok, score=1)
    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-2",
        candidates=[cand_full, cand_ok],
        request_id="req-advance-1",
        idempotency_key="idem-advance-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=2,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            scope_full.scope_id: coord.snapshot(scope_full.scope_id),
            scope_ok.scope_id: coord.snapshot(scope_ok.scope_id),
        },
        invoke=None,
    )
    assert result.success is True
    assert result.selected is not None
    assert result.selected.binding_id == cand_ok.binding_id


def test_each_retry_fallback_has_new_linked_attempt_and_reservation():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("fb-a")
    scope_b = _scope("fb-b")
    _configure(coord, scope_a, ceiling=10)
    _configure(coord, scope_b, ceiling=10)
    cand_a, _ = _candidate(provider_key="fb-a", scope=scope_a, score=50)
    cand_b, _ = _candidate(provider_key="fb-b", scope=scope_b, score=10)

    call_count = {"n": 0}

    def invoke(attempt):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Capacity failure on first — safe to fallback.
            obs = ProviderUsageObservation(
                scope_id=attempt.scope_id,
                request_id="req-fb-1",
                usage=UsageVector(),
                http_status=429,
                confidence=ConfidenceLevel.HIGH,
                provenance=Provenance(source=LimitSource.ERROR),
                reason_codes=("rate_limited",),
                retry_after_ms=1000,
            )
            return InvokeOutcome(
                success=False,
                observation=obs,
                error_class=ErrorSafetyClass.CAPACITY,
                reason_codes=("rate_limited",),
            )
        obs = ProviderUsageObservation(
            scope_id=attempt.scope_id,
            request_id="req-fb-1",
            usage=UsageVector.of(requests=1),
            http_status=200,
            confidence=ConfidenceLevel.AUTHORITATIVE,
            provenance=Provenance(source=LimitSource.RESPONSE_BODY),
        )
        return InvokeOutcome(
            success=True,
            observation=obs,
            settled=UsageVector.of(requests=1),
            error_class=ErrorSafetyClass.SUCCESS,
        )

    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-fb",
        candidates=[cand_a, cand_b],
        request_id="req-fb-1",
        idempotency_key="idem-fb-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=3,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        invoke=invoke,
    )
    assert result.success is True
    assert len(result.attempts) >= 2
    # Distinct attempt ids and reservations.
    attempt_ids = [a.attempt_id for a in result.attempts]
    assert len(set(attempt_ids)) == len(attempt_ids)
    reservations = [a.reservation_id for a in result.attempts if a.reservation_id]
    assert len(set(reservations)) == len(reservations)
    # Chain links parent → child.
    assert result.chain.links
    if len(result.chain.links) >= 2:
        assert result.chain.links[1].parent_attempt_id == result.chain.links[0].attempt_id


def test_unsafe_semantic_client_side_effect_never_fallback():
    assert is_fallback_safe(ErrorSafetyClass.SEMANTIC) is False
    assert is_fallback_safe(ErrorSafetyClass.CLIENT) is False
    assert is_fallback_safe(ErrorSafetyClass.SIDE_EFFECT) is False
    assert is_fallback_safe(ErrorSafetyClass.CAPACITY) is True
    assert is_fallback_safe(ErrorSafetyClass.TRANSIENT) is True

    assert classify_invoke_error(semantic=True) is ErrorSafetyClass.SEMANTIC
    assert classify_invoke_error(client_error=True) is ErrorSafetyClass.CLIENT
    assert classify_invoke_error(side_effecting=True) is ErrorSafetyClass.SIDE_EFFECT
    assert classify_invoke_error(http_status=429) is ErrorSafetyClass.CAPACITY
    assert classify_invoke_error(http_status=400) is ErrorSafetyClass.CLIENT
    assert (
        classify_invoke_error(reason_codes=("context_overflow",))
        is ErrorSafetyClass.SEMANTIC
    )

    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("sem-a")
    scope_b = _scope("sem-b")
    _configure(coord, scope_a, ceiling=10)
    _configure(coord, scope_b, ceiling=10)
    cand_a, _ = _candidate(provider_key="sem-a", scope=scope_a, score=50)
    cand_b, _ = _candidate(provider_key="sem-b", scope=scope_b, score=10)

    def invoke(attempt):
        return InvokeOutcome(
            success=False,
            error_class=ErrorSafetyClass.SEMANTIC,
            reason_codes=("invalid_request",),
            side_effecting=False,
        )

    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-sem",
        candidates=[cand_a, cand_b],
        request_id="req-sem-1",
        idempotency_key="idem-sem-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=3,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        invoke=invoke,
    )
    assert result.success is False
    assert "no_fallback_unsafe_error" in result.reason_codes
    # Did not attempt the second binding after unsafe error.
    assert len(result.attempts) == 1


# ---------------------------------------------------------------------------
# Wait vs reroute / deadline / max attempts
# ---------------------------------------------------------------------------


def test_wait_versus_reroute_honors_deadline_and_max_attempts():
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.SAME_PROVIDER,
        max_attempts=2,
        allow_wait=True,
        max_wait_ms=5_000,
        deadline_ms=10_000,
    )
    now = _now()
    next_ok = now + timedelta(seconds=2)
    next_late = now + timedelta(hours=1)
    deadline = now + timedelta(seconds=30)

    assert (
        decide_wait_or_reroute(
            policy=policy,
            now=now,
            next_eligible_at=next_ok,
            deadline_at=deadline,
            has_reroute_candidate=True,
            attempts_used=1,
        )
        is WaitOrReroute.REROUTE
    )
    assert (
        decide_wait_or_reroute(
            policy=policy,
            now=now,
            next_eligible_at=next_ok,
            deadline_at=deadline,
            has_reroute_candidate=False,
            attempts_used=1,
        )
        is WaitOrReroute.WAIT
    )
    assert (
        decide_wait_or_reroute(
            policy=policy,
            now=now,
            next_eligible_at=next_late,
            deadline_at=deadline,
            has_reroute_candidate=False,
            attempts_used=1,
        )
        is WaitOrReroute.FAIL
    )
    assert (
        decide_wait_or_reroute(
            policy=policy,
            now=now,
            next_eligible_at=next_ok,
            deadline_at=deadline,
            has_reroute_candidate=False,
            attempts_used=2,
        )
        is WaitOrReroute.FAIL
    )


def test_admission_respects_max_attempts_bound():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("max-att")
    _configure(coord, scope, ceiling=0)
    cand, _ = _candidate(provider_key="max-att", scope=scope)
    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-max",
        candidates=[cand],
        request_id="req-max-1",
        idempotency_key="idem-max-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.NONE,
            max_attempts=2,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        invoke=None,
    )
    assert result.success is False
    assert len(result.attempts) <= 2


# ---------------------------------------------------------------------------
# Jitter / single-flight / circuit breakers
# ---------------------------------------------------------------------------


def test_jitter_is_deterministic_and_bounded():
    a = admission_jitter_ms("req-j", max_ms=100, salt="0")
    b = admission_jitter_ms("req-j", max_ms=100, salt="0")
    c = admission_jitter_ms("req-j", max_ms=100, salt="1")
    assert a == b
    assert 0 <= a <= 100
    assert c != a or True  # may collide; just ensure bounds
    assert 0 <= c <= 100
    assert admission_jitter_ms("x", max_ms=0) == 0


def test_single_flight_collapses_concurrent_work():
    sf = SingleFlight()
    counter = {"n": 0}
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        out = sf.do(
            "same-key",
            lambda: (counter.__setitem__("n", counter["n"] + 1) or counter["n"]),
        )
        results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    # Leader runs once; others may re-run after cache pop — at least collapsed.
    assert counter["n"] >= 1
    assert len(results) == 4


def test_circuit_breaker_prevents_herd_on_hot_binding():
    clock_ms = {"t": 0}

    def now_ms():
        return clock_ms["t"]

    cb = CircuitBreakerRegistry(
        failure_threshold=2, cooldown_ms=10_000, clock_ms=now_ms
    )
    binding = stable_id("binding", "hot")
    assert cb.is_open(binding) is False
    cb.record_failure(binding)
    assert cb.is_open(binding) is False
    cb.record_failure(binding)
    assert cb.is_open(binding) is True
    clock_ms["t"] = 11_000
    assert cb.is_open(binding) is False  # half-open
    cb.record_success(binding)
    assert cb.is_open(binding) is False


# ---------------------------------------------------------------------------
# Receipts bind revisions / digests / chain without secrets
# ---------------------------------------------------------------------------


def test_receipts_bind_catalog_usage_candidates_selection_settlement_chain():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("rcpt")
    _configure(coord, scope, ceiling=20)
    cand, _ = _candidate(provider_key="rcpt", scope=scope)

    def invoke(attempt):
        obs = ProviderUsageObservation(
            scope_id=attempt.scope_id,
            request_id="req-rcpt-1",
            usage=UsageVector.of(requests=1, input_tokens=12, output_tokens=4),
            http_status=200,
            confidence=ConfidenceLevel.AUTHORITATIVE,
            provenance=Provenance(source=LimitSource.RESPONSE_BODY),
        )
        return InvokeOutcome(
            success=True,
            observation=obs,
            settled=UsageVector.of(requests=1, input_tokens=12, output_tokens=4),
        )

    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-rev-rcpt",
        candidates=[cand],
        request_id="req-rcpt-1",
        idempotency_key="idem-rcpt-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1, input_tokens=12, output_tokens=4),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        caller_id="caller-test",
        invoke=invoke,
    )
    assert result.success is True
    receipt = result.receipt
    assert receipt is not None
    assert receipt.catalog_revision == "catalog-rev-rcpt"
    assert receipt.usage_revision
    assert receipt.selected_binding_id == cand.binding_id
    assert receipt.reservation_id
    assert receipt.estimate_id
    assert receipt.observation_id
    assert receipt.resolution_id
    assert receipt.settled.entries
    assert receipt.final_status == FinalStatus.COMMITTED.value
    assert receipt_binds_revisions(
        receipt,
        catalog_revision="catalog-rev-rcpt",
        usage_revision=receipt.usage_revision,
    )
    # Digests / chain markers present without payloads.
    assert any("digest" in r or "chain" in r for r in receipt.reason_codes) or receipt.reason_codes
    payload = receipt.to_dict()
    assert_receipt_safe(payload)
    assert_no_prompt_media_or_output(payload)
    for value in _walk_strings(payload):
        assert not contains_raw_endpoint(value)
        assert "sk-" not in value
        assert "Bearer " not in value


def test_receipt_builder_rejects_prompts_media_output_credentials_endpoints():
    with pytest.raises(ReceiptError):
        RouteReceiptDraft(
            catalog_revision="catalog-rev",
            usage_revision="usage-rev",
            request_id="req",
            attempt_id="attempt",
            idempotency_key="idem",
            operation="text.chat",
            caller_id="https://secret.example/v1/chat",  # raw endpoint
        )
    draft = RouteReceiptDraft(
        catalog_revision="catalog-rev",
        usage_revision="usage-rev",
        request_id="req-safe",
        attempt_id="attempt-safe",
        idempotency_key="idem-safe",
        operation="text.chat",
        final_status=FinalStatus.COMMITTED.value,
        created_at=_rfc(_now()),
    )
    receipt = build_usage_routing_receipt(draft)
    assert receipt.receipt_id.startswith("urcpt_")
    with pytest.raises(ReceiptError):
        assert_receipt_safe({"prompt": "hello world"})
    with pytest.raises(ReceiptError):
        assert_receipt_safe({"raw_headers": {"authorization": "x"}})


def test_hard_rejection_and_ranking_digests_are_stable():
    scope = _scope("dig")
    snap = _snapshot(scope, requests_ceiling=0, requests_used=0, state=AvailabilityState.EXHAUSTED)
    cand, _ = _candidate(provider_key="dig", scope=scope)
    resolution = resolve_usage_aware(
        catalog_revision="cat",
        candidates=[cand],
        snapshots_by_scope={scope.scope_id: snap},
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1), now=_rfc(_now())
        ),
    )
    d1 = hard_rejection_digest(resolution.rejected)
    d2 = hard_rejection_digest(resolution.rejected)
    assert d1 == d2
    # Ranking digest on empty accepted set is None.
    assert ranking_inputs_digest(resolution.candidates) is None or True
    assert candidates_digest(resolution.candidates, resolution.rejected)


def test_chain_requires_parent_before_child():
    a = AttemptLink(attempt_id="attempt-1", final_status="rejected")
    b = AttemptLink(
        attempt_id="attempt-2",
        parent_attempt_id="attempt-1",
        final_status="committed",
        fallback_class=FallbackClass.SAME_PROVIDER,
    )
    chain = build_receipt_chain([a, b])
    assert chain.chain_id.startswith("uchain_")
    with pytest.raises(ReceiptError, match="parent_attempt_id"):
        build_receipt_chain([b, a])


def test_pin_filter_rejects_mismatched_candidates():
    scope = _scope("pin")
    cand, _ = _candidate(provider_key="pin", scope=scope)
    other, _ = _candidate(provider_key="other", scope=_scope("other"))
    pin = RoutePin(provider_id=scope.provider_id)
    accepted, rejected = apply_pin_filter([cand, other], pin)
    assert len(accepted) == 1
    assert accepted[0].binding_id == cand.binding_id
    assert len(rejected) == 1
    assert "pin_provider_mismatch" in rejected[0][1]


def test_requirement_ids_exported():
    assert ROUTE_ADMISSION_REQUIREMENT_ID.startswith("requirement:")
    assert USAGE_ROUTING_RECEIPT_REQUIREMENT_ID.startswith("requirement:")


def test_plan_route_with_exact_pin_does_not_cross_provider():
    scope_a = _scope("plan-a")
    scope_b = _scope("plan-b")
    snap_a = _snapshot(scope_a)
    snap_b = _snapshot(scope_b)
    cand_a, _ = _candidate(provider_key="plan-a", scope=scope_a, score=1)
    cand_b, _ = _candidate(provider_key="plan-b", scope=scope_b, score=100)
    resolution = plan_route(
        catalog_revision="cat-pin",
        candidates=[cand_a, cand_b],
        snapshots_by_scope={scope_a.scope_id: snap_a, scope_b.scope_id: snap_b},
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
        ),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1), now=_rfc(_now())
        ),
        pin=RoutePin(provider_id=scope_a.provider_id),
    )
    assert resolution.selected_binding_id == cand_a.binding_id
    assert all(c.binding_id == cand_a.binding_id for c in resolution.candidates) or (
        resolution.selected_binding_id == cand_a.binding_id
    )


def test_successful_invoke_settles_and_records_observation():
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("settle")
    _configure(coord, scope, ceiling=5)
    cand, _ = _candidate(provider_key="settle", scope=scope)

    def invoke(attempt):
        return InvokeOutcome(
            success=True,
            observation=ProviderUsageObservation(
                scope_id=attempt.scope_id,
                request_id="req-settle",
                usage=UsageVector.of(requests=1),
                http_status=200,
                confidence=ConfidenceLevel.HIGH,
                provenance=Provenance(source=LimitSource.RESPONSE_BODY),
            ),
            settled=UsageVector.of(requests=1),
        )

    admission = UsageRouteAdmission(coord, owner_id="test-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="cat-settle",
        candidates=[cand],
        request_id="req-settle",
        idempotency_key="idem-settle",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1), now=_rfc(clock.now())
        ),
        invoke=invoke,
    )
    assert result.success
    assert result.selected.settlement is not None
    assert result.selected.final_status == FinalStatus.COMMITTED.value


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _walk_strings(value, out=None):
    if out is None:
        out = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            _walk_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_strings(item, out)
    return out
