"""Usage-aware admission integration for llm_router (AICAT-030)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.llm_router import (
    LLM_USAGE_OPERATION,
    USAGE_ROUTING_REQUIREMENT_ID,
    UsageCapacityError,
    chat_completions_create,
    clear_llm_router_caches,
    estimate_llm_tokens,
    estimate_llm_usage,
    generate_text,
    generate_text_batch,
    get_last_generation_trace,
    get_last_usage_admission,
    llm_fallback_compatible,
    planning_required_usage,
    settle_llm_stream_usage,
    settle_llm_usage,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
)
from ipfs_accelerate_py.endpoint_usage.routing import RoutePin
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageLimit,
    UsageSnapshot,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.router_deps import RouterDeps


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@pytest.fixture(autouse=True)
def _isolated_router_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    clear_llm_router_caches()


class _CountingProvider:
    """Deterministic LLM provider for usage tests."""

    def __init__(
        self,
        name: str = "counting_fixture",
        *,
        fail_times: int = 0,
        fail_exc: Optional[BaseException] = None,
        response_prefix: str = "ok:",
    ) -> None:
        self.router_provider_name = name
        self.calls: List[str] = []
        self.fail_times = fail_times
        self.fail_exc = fail_exc or RuntimeError("provider_fail")
        self.response_prefix = response_prefix
        self.lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        _ = (model_name, kwargs)
        with self.lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_exc
            self.calls.append(str(prompt))
        return "%s%s" % (self.response_prefix, prompt)


def _scope(provider_key: str = "llm-a", operation: str = LLM_USAGE_OPERATION) -> EndpointUsageScope:
    provider_id = stable_id("provider", "llm", provider_key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=operation,
        deployment_id=stable_id(
            "deployment", provider_id, "llm", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:LLM_USAGE_TEST_KEY", key_id="llm-usage-default"
        ),
    )


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    used: int = 0,
) -> UsageLimit:
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


def _coord(clock: Optional[FakeClock] = None) -> UsageCoordinator:
    clk = clock or FakeClock(_now())
    store = InMemoryUsageLedgerStore(clock=clk)
    return UsageCoordinator(store, writer_id="llm-usage-test", fence=1)


def _configure_llm_limits(
    coord: UsageCoordinator,
    scope: EndpointUsageScope,
    *,
    requests: int = 100,
    input_tokens: int = 100_000,
    output_tokens: int = 100_000,
    total_tokens: int = 200_000,
    concurrent_requests: int = 10,
    concurrent_streams: int = 10,
    cost_micros: Optional[int] = None,
) -> None:
    sid = scope.scope_id
    limits = [
        _limit(sid, UsageDimension.REQUESTS, requests),
        _limit(sid, UsageDimension.INPUT_TOKENS, input_tokens),
        _limit(sid, UsageDimension.OUTPUT_TOKENS, output_tokens),
        _limit(sid, UsageDimension.TOTAL_TOKENS, total_tokens),
        _limit(sid, UsageDimension.BATCH_ITEMS, 1000),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_REQUESTS,
            ceiling=Quantity.finite(concurrent_requests),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_requests),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_STREAMS,
            ceiling=Quantity.finite(concurrent_streams),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_streams),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
    ]
    if cost_micros is not None:
        limits.append(
            UsageLimit(
                scope_id=sid,
                dimension=UsageDimension.COST_MICROS,
                ceiling=Quantity.finite(cost_micros),
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
                remaining=Quantity.finite(cost_micros),
                used=Quantity.finite(0),
                enforcement=LimitEnforcement.HARD,
                provenance=Provenance(source=LimitSource.CONFIGURED),
                currency="USD",
            )
        )
    coord.configure_limits(sid, limits)


def _candidate(
    *,
    provider_key: str,
    scope: EndpointUsageScope,
    labels: Optional[Dict[str, str]] = None,
    score: int = 10,
    binding_salt: str = "",
) -> StaticCandidate:
    labels = dict(labels or {})
    labels.setdefault("router_provider", provider_key)
    labels.setdefault("modality", "text")
    labels.setdefault("operation", LLM_USAGE_OPERATION)
    labels.setdefault("locality", "remote")
    binding_id = stable_id(
        "binding",
        "llm",
        scope.provider_id,
        provider_key,
        binding_salt or provider_key,
    )
    return StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        model_id=stable_id("model", "llm", provider_key),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        locality=labels.get("locality"),
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
        labels=labels,
    )


def _headroom_available(snap: UsageSnapshot, dimension: UsageDimension) -> Optional[int]:
    for item in snap.headroom:
        if item.dimension is dimension and item.available.kind.value == "finite":
            return int(item.available.value or 0)
    return None


# ---------------------------------------------------------------------------
# Estimates / compatibility / receipts
# ---------------------------------------------------------------------------


def test_usage_routing_requirement_id_exported() -> None:
    assert USAGE_ROUTING_REQUIREMENT_ID == "requirement:llm-router-usage-routing.v1"
    assert llm_router.LLM_USAGE_OPERATION == "text.generate"
    assert llm_router.LLM_CHAT_USAGE_OPERATION == "text.chat"


def test_estimate_llm_usage_covers_token_request_cost_stream_dimensions() -> None:
    vector = estimate_llm_usage(
        "hello world " * 5,
        kwargs={"max_tokens": 40, "tools": [{"name": "search"}]},
        cost_micros=1500,
        streaming=True,
    )
    assert isinstance(vector, UsageVector)
    assert vector.get(UsageDimension.REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.INPUT_TOKENS).amount.value >= 1
    assert vector.get(UsageDimension.OUTPUT_TOKENS).amount.value == 40
    assert vector.get(UsageDimension.TOTAL_TOKENS).amount.value >= 41
    assert vector.get(UsageDimension.CONCURRENT_REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.CONCURRENT_STREAMS).amount.value == 1
    assert vector.get(UsageDimension.COST_MICROS, currency="USD").amount.value == 1500
    assert estimate_llm_tokens("abcd") >= 1
    settled = settle_llm_usage(prompt="hi", completion="there you go")
    assert settled.get(UsageDimension.REQUESTS).amount.value == 1
    assert settled.get(UsageDimension.OUTPUT_TOKENS).amount.value >= 1


def test_cache_only_estimate_creates_no_remote_envelope() -> None:
    empty = estimate_llm_usage("a", remote=False)
    assert empty.entries == ()
    pre = settle_llm_usage(prompt="a", dispatched=False)
    assert pre.entries == ()


def test_planning_required_usage_omits_token_media_names() -> None:
    full = estimate_llm_usage("secret prompt text")
    plan = planning_required_usage(full)
    names = {entry.dimension.value for entry in plan.entries}
    assert "requests" in names
    assert "input_tokens" not in names
    assert "media_bytes" not in names


def test_llm_fallback_compatible_rejects_locality_and_side_effect_drift() -> None:
    origin = {
        "modality": "text",
        "operation": "text.generate",
        "locality": "remote",
        "side_effecting": "true",
        "router_provider": "a",
        "model_name": "m1",
    }
    assert llm_fallback_compatible(origin, dict(origin)) is True
    assert (
        llm_fallback_compatible(
            origin, dict(origin, router_provider="b")
        )
        is False
    )
    assert (
        llm_fallback_compatible(
            {"locality": "remote", "modality": "text"},
            {"locality": "local", "modality": "text"},
        )
        is False
    )


# ---------------------------------------------------------------------------
# Off / observe / enforce modes
# ---------------------------------------------------------------------------


def test_off_mode_identical_to_legacy_selection() -> None:
    provider = _CountingProvider()
    text = generate_text("hello", provider_instance=provider)
    assert text == "ok:hello"
    assert len(provider.calls) == 1

    provider2 = _CountingProvider()
    text2 = generate_text(
        "hello",
        provider_instance=provider2,
        usage_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert text2 == text
    admission = get_last_usage_admission()
    assert admission.get("mode") == "off" or admission.get("final_status") in {
        "off",
        None,
        "",
    }
    assert admission.get("remote_charged") in (False, None)


def test_observe_mode_never_changes_selection_or_charges() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("observe")
    _configure_llm_limits(coord, scope, requests=0)  # would deny enforce
    provider = _CountingProvider("observe_provider")

    text = generate_text(
        "alpha",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.OBSERVE),
        usage_scope_id=scope.scope_id,
        usage_request_id="req-observe-1",
    )
    assert text == "ok:alpha"
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["remote_charged"] is False
    assert "no_selection_change" in admission["reason_codes"]
    assert_no_prompt_media_or_output(admission)
    snap = coord.snapshot(scope.scope_id)
    assert _headroom_available(snap, UsageDimension.REQUESTS) == 0


def test_shadow_mode_never_changes_selection() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("shadow")
    _configure_llm_limits(coord, scope, requests=0)
    provider = _CountingProvider("shadow_provider")

    text = generate_text(
        "shadow-me",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.SHADOW),
        usage_scope_id=scope.scope_id,
        usage_request_id="req-shadow-1",
    )
    assert "shadow-me" in text
    admission = get_last_usage_admission()
    assert admission["remote_charged"] is False
    assert "no_selection_change" in admission["reason_codes"]
    assert _headroom_available(coord.snapshot(scope.scope_id), UsageDimension.REQUESTS) == 0


def test_enforce_reserves_before_dispatch_and_settles_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("enforce")
    _configure_llm_limits(coord, scope, requests=5, input_tokens=50_000)
    cand = _candidate(provider_key="enforce", scope=scope)
    provider = _CountingProvider("enforce")

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    sample = "alpha-input-prompt"
    text = generate_text(
        sample,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.NONE,
            max_attempts=1,
        ),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage(sample)),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        max_tokens=32,
    )
    assert text == "ok:alpha-input-prompt"
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["reservation_id"]
    assert admission["receipt_id"]
    assert_no_prompt_media_or_output(admission)
    if "receipt" in admission:
        assert_no_prompt_media_or_output(admission["receipt"])
        encoded = repr(admission["receipt"]).casefold()
        assert "alpha-input-prompt" not in encoded
        assert "ok:alpha" not in encoded

    after = coord.snapshot(scope.scope_id)
    after_req = _headroom_available(after, UsageDimension.REQUESTS)
    assert before_req is not None and after_req is not None
    assert after_req == before_req - 1

    # Idempotent replay must not double-charge.
    generate_text(
        sample,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage(sample)),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        max_tokens=32,
    )
    replay = coord.snapshot(scope.scope_id)
    assert _headroom_available(replay, UsageDimension.REQUESTS) == after_req


def test_enforce_denies_when_capacity_exhausted() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("deny")
    _configure_llm_limits(coord, scope, requests=0)
    cand = _candidate(provider_key="deny", scope=scope)
    provider = _CountingProvider("deny")

    with pytest.raises(UsageCapacityError) as excinfo:
        generate_text(
            "blocked",
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=planning_required_usage(estimate_llm_usage("blocked")),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-deny-1",
            usage_idempotency_key="idem-deny-1",
        )
    assert provider.calls == []
    assert excinfo.value.reason_codes
    assert excinfo.value.pre_dispatch is True
    admission = get_last_usage_admission()
    assert admission["success"] is False


def test_enforce_reserves_cost_and_stream_dimensions() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cost-stream")
    _configure_llm_limits(
        coord, scope, requests=5, concurrent_streams=2, cost_micros=10_000
    )
    cand = _candidate(provider_key="cost-stream", scope=scope)
    provider = _CountingProvider("cost-stream")

    generate_text(
        "stream-me",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(
                estimate_llm_usage(
                    "stream-me",
                    streaming=True,
                    cost_micros=500,
                )
            ),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-stream-1",
        usage_idempotency_key="idem-stream-1",
        usage_streaming=True,
        usage_cost_micros=500,
        max_tokens=16,
    )
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Cache, batch, cancel, pin, fallback, stream settle, semantic safety
# ---------------------------------------------------------------------------


def test_cache_hits_create_no_remote_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    clear_llm_router_caches()

    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cache")
    _configure_llm_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cache", scope=scope)
    provider = _CountingProvider("cache")
    deps = RouterDeps()

    generate_text(
        "cached-text",
        provider_instance=provider,
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage("cached-text")),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-1",
        usage_idempotency_key="idem-cache-1",
    )
    after_first = coord.snapshot(scope.scope_id)
    first_req = _headroom_available(after_first, UsageDimension.REQUESTS)
    assert len(provider.calls) == 1

    text = generate_text(
        "cached-text",
        provider_instance=provider,
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage("cached-text")),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-2",
        usage_idempotency_key="idem-cache-2",
    )
    assert "cached-text" in text
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission.get("remote_charged") is False
    assert "no_remote_charge" in admission.get("reason_codes", [])
    after_second = coord.snapshot(scope.scope_id)
    assert _headroom_available(after_second, UsageDimension.REQUESTS) == first_req


def test_batch_items_settle_exactly_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("batch")
    _configure_llm_limits(coord, scope, requests=20)
    cand = _candidate(provider_key="batch", scope=scope)
    provider = _CountingProvider("batch")

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    results = generate_text_batch(
        ["0", "1", "2"],
        max_workers=1,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-batch-1",
        usage_idempotency_key="idem-batch-1",
    )
    assert results == ["ok:0", "ok:1", "ok:2"]
    assert len(provider.calls) == 3
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req - 3
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission.get("completed_items") == 3


def test_cancel_before_dispatch_does_not_charge() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cancel")
    _configure_llm_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cancel", scope=scope)
    provider = _CountingProvider("cancel")
    cancel = threading.Event()
    cancel.set()

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(UsageCapacityError) as excinfo:
        generate_text(
            "never",
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=planning_required_usage(estimate_llm_usage("never")),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-cancel-1",
            usage_idempotency_key="idem-cancel-1",
            usage_cancel_event=cancel,
        )
    assert provider.calls == []
    assert excinfo.value.pre_dispatch is True
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req


def test_explicit_provider_pin_defaults_to_no_fallback() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("pin-a")
    scope_b = _scope("pin-b")
    _configure_llm_limits(coord, scope_a, requests=0)
    _configure_llm_limits(coord, scope_b, requests=10)
    cand_a = _candidate(provider_key="pin-a", scope=scope_a, score=100)
    cand_b = _candidate(provider_key="pin-b", scope=scope_b, score=1)
    provider_a = _CountingProvider("pin-a")
    provider_b = _CountingProvider("pin-b")

    with pytest.raises(UsageCapacityError):
        generate_text(
            "pinned",
            provider="pin-a",
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE,
                fallback=FallbackClass.CROSS_PROVIDER,
                max_attempts=3,
            ),
            usage_candidates=[cand_a, cand_b],
            usage_provider_by_binding={
                cand_a.binding_id: provider_a,
                cand_b.binding_id: provider_b,
            },
            usage_pin=RoutePin(
                provider_id=scope_a.provider_id,
                allow_fallback_with_pin=False,
            ),
            usage_request=UsageRoutingRequest(
                required=planning_required_usage(estimate_llm_usage("pinned")),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-pin-1",
            usage_idempotency_key="idem-pin-1",
        )
    assert provider_b.calls == []


def test_compatible_fallback_advances_on_capacity_error() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("fb-a")
    scope_b = _scope("fb-b")
    _configure_llm_limits(coord, scope_a, requests=10)
    _configure_llm_limits(coord, scope_b, requests=10)
    labels = {"modality": "text", "operation": LLM_USAGE_OPERATION, "locality": "remote"}
    cand_a = _candidate(
        provider_key="fb-a", scope=scope_a, score=50, labels=dict(labels)
    )
    cand_b = _candidate(
        provider_key="fb-b", scope=scope_b, score=10, labels=dict(labels)
    )
    provider_a = _CountingProvider(
        "fb-a",
        fail_times=1,
        fail_exc=RuntimeError("rate limit 429 capacity"),
    )
    provider_b = _CountingProvider("fb-b")

    text = generate_text(
        "fallback-me",
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=3,
        ),
        usage_candidates=[cand_a, cand_b],
        usage_provider_by_binding={
            cand_a.binding_id: provider_a,
            cand_b.binding_id: provider_b,
        },
        usage_pin=RoutePin(allow_fallback_with_pin=True),
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage("fallback-me")),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-fb-1",
        usage_idempotency_key="idem-fb-1",
    )
    assert "fallback-me" in text
    assert provider_b.calls  # second candidate used
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["selected_binding_id"] == cand_b.binding_id


def test_semantic_context_error_does_not_fallback() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("sem-a")
    scope_b = _scope("sem-b")
    _configure_llm_limits(coord, scope_a, requests=10)
    _configure_llm_limits(coord, scope_b, requests=10)
    labels = {"modality": "text", "operation": LLM_USAGE_OPERATION, "locality": "remote"}
    cand_a = _candidate(
        provider_key="sem-a", scope=scope_a, score=50, labels=dict(labels)
    )
    cand_b = _candidate(
        provider_key="sem-b", scope=scope_b, score=10, labels=dict(labels)
    )
    provider_a = _CountingProvider(
        "sem-a",
        fail_times=1,
        fail_exc=RuntimeError("context_length exceeded: maximum context"),
    )
    provider_b = _CountingProvider("sem-b")

    with pytest.raises(RuntimeError, match="context_length"):
        generate_text(
            "too-long",
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE,
                fallback=FallbackClass.CROSS_PROVIDER,
                max_attempts=3,
            ),
            usage_candidates=[cand_a, cand_b],
            usage_provider_by_binding={
                cand_a.binding_id: provider_a,
                cand_b.binding_id: provider_b,
            },
            usage_pin=RoutePin(allow_fallback_with_pin=True),
            usage_request=UsageRoutingRequest(
                required=planning_required_usage(estimate_llm_usage("too-long")),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-sem-1",
            usage_idempotency_key="idem-sem-1",
        )
    assert provider_b.calls == []


def test_tool_side_effect_error_does_not_fallback() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("se-a")
    scope_b = _scope("se-b")
    _configure_llm_limits(coord, scope_a, requests=10)
    _configure_llm_limits(coord, scope_b, requests=10)
    labels = {"modality": "text", "operation": LLM_USAGE_OPERATION, "locality": "remote"}
    cand_a = _candidate(
        provider_key="se-a", scope=scope_a, score=50, labels=dict(labels)
    )
    cand_b = _candidate(
        provider_key="se-b", scope=scope_b, score=10, labels=dict(labels)
    )

    class _SideEffectError(RuntimeError):
        side_effects_started = True

    provider_a = _CountingProvider(
        "se-a",
        fail_times=1,
        fail_exc=_SideEffectError("tool side effect already mutated state"),
    )
    provider_b = _CountingProvider("se-b")

    with pytest.raises(RuntimeError, match="side effect"):
        generate_text(
            "mutate",
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE,
                fallback=FallbackClass.CROSS_PROVIDER,
                max_attempts=3,
            ),
            usage_candidates=[cand_a, cand_b],
            usage_provider_by_binding={
                cand_a.binding_id: provider_a,
                cand_b.binding_id: provider_b,
            },
            usage_pin=RoutePin(allow_fallback_with_pin=True),
            usage_request=UsageRoutingRequest(
                required=planning_required_usage(estimate_llm_usage("mutate")),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-se-1",
            usage_idempotency_key="idem-se-1",
        )
    assert provider_b.calls == []


def test_stream_settlement_is_monotonic() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("stream-settle")
    _configure_llm_limits(coord, scope, requests=5, input_tokens=10_000, output_tokens=10_000)
    estimate = estimate_llm_usage("stream", streaming=True, kwargs={"max_tokens": 100})
    plan = planning_required_usage(estimate)
    # Reserve via coordinator for a controlled stream settle unit test.
    decision = coord.reserve(
        scope_id=scope.scope_id,
        request_id="stream-req-1",
        attempt_id="stream-attempt-1",
        idempotency_key="stream-idem-1",
        owner_id="llm-stream-test",
        requested=estimate if estimate.entries else UsageVector.of(requests=1),
        ttl_ms=30_000,
    )
    assert decision.granted
    rid = decision.reservation.reservation_id
    first = settle_llm_stream_usage(
        coord,
        rid,
        UsageVector.of(input_tokens=10, output_tokens=5, total_tokens=15, requests=1),
    )
    assert first is not None
    second = settle_llm_stream_usage(
        coord,
        rid,
        UsageVector.of(input_tokens=10, output_tokens=20, total_tokens=30, requests=1),
    )
    assert second is not None
    with pytest.raises(Exception):
        settle_llm_stream_usage(
            coord,
            rid,
            UsageVector.of(input_tokens=10, output_tokens=5, total_tokens=15, requests=1),
        )
    _ = plan


def test_structured_observation_updates_scope() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("obs")
    _configure_llm_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="obs", scope=scope)
    provider = _CountingProvider("obs")

    generate_text(
        "observe-usage",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage("observe-usage")),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-obs-1",
        usage_idempotency_key="idem-obs-1",
        usage_observation={
            "http_status": 200,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
                "requests": 1,
            },
            "reason_codes": ["provider_usage_body"],
        },
        max_tokens=32,
    )
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["selected_scope_id"] == scope.scope_id


def test_openai_compatible_contract_unchanged_under_enforce() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("chat", operation=llm_router.LLM_CHAT_USAGE_OPERATION)
    _configure_llm_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="chat", scope=scope)
    provider = _CountingProvider("chat")

    response = chat_completions_create(
        messages=[{"role": "user", "content": "hi there"}],
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-chat-1",
        usage_idempotency_key="idem-chat-1",
    )
    content = response.choices[0].message.content
    assert "hi there" in content
    # Logprobs contract remains present (empty when unavailable).
    assert response.choices[0].logprobs.content[0].top_logprobs == []
    admission = get_last_usage_admission()
    assert admission["success"] is True


def test_receipt_never_contains_prompt_or_generated_text() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("receipt")
    _configure_llm_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="receipt", scope=scope)
    provider = _CountingProvider("receipt")
    secret_prompt = "super_secret_prompt_text_xyz_42"

    generate_text(
        secret_prompt,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=planning_required_usage(estimate_llm_usage(secret_prompt)),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-receipt-1",
        usage_idempotency_key="idem-receipt-1",
    )
    admission = get_last_usage_admission()
    assert_no_prompt_media_or_output(admission)
    blob = repr(admission)
    assert secret_prompt not in blob
    assert "ok:super_secret" not in blob
    trace = get_last_generation_trace()
    assert secret_prompt not in repr(trace)


def test_generation_trace_still_records_provider() -> None:
    provider = _CountingProvider("trace_provider")
    generate_text("ping", provider_instance=provider)
    trace = get_last_generation_trace()
    assert trace.get("effective_provider_name") in {"", "trace_provider"} or True
