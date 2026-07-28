"""Usage-aware admission integration for embeddings_router (AICAT-031)."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pytest

import ipfs_accelerate_py.embeddings_router as embeddings_router
from ipfs_accelerate_py.embeddings_router import (
    USAGE_ROUTING_REQUIREMENT_ID,
    EmbeddingsRouterError,
    UsageCapacityError,
    clear_embeddings_router_caches,
    embed_texts,
    embed_texts_batched,
    embedding_fallback_compatible,
    estimate_embedding_tokens,
    estimate_embedding_usage,
    get_last_embedding_trace,
    get_last_usage_admission,
    settle_embedding_usage,
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
    AvailabilityState,
    DimensionHeadroom,
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
    clear_embeddings_router_caches()


class _CountingProvider:
    """Deterministic embeddings provider for usage tests."""

    def __init__(
        self,
        name: str = "counting_fixture",
        *,
        dimension: int = 2,
        fail_times: int = 0,
        fail_exc: Optional[BaseException] = None,
    ) -> None:
        self.router_provider_name = name
        self.dimension = dimension
        self.calls: List[List[str]] = []
        self.fail_times = fail_times
        self.fail_exc = fail_exc or RuntimeError("provider_fail")
        self.lock = threading.Lock()

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> List[List[float]]:
        _ = (model_name, device, kwargs)
        items = list(texts)
        with self.lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_exc
            self.calls.append(items)
        return [
            [float(index), float(index) + 0.5][: self.dimension]
            + [0.0] * max(0, self.dimension - 2)
            for index, _ in enumerate(items)
        ]


def _scope(provider_key: str = "emb-a") -> EndpointUsageScope:
    provider_id = stable_id("provider", "embeddings", provider_key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=embeddings_router.EMBEDDING_USAGE_OPERATION,
        deployment_id=stable_id(
            "deployment", provider_id, "embed", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:EMB_USAGE_TEST_KEY", key_id="emb-usage-default"
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
    return UsageCoordinator(store, writer_id="emb-usage-test", fence=1)


def _configure_embedding_limits(
    coord: UsageCoordinator,
    scope: EndpointUsageScope,
    *,
    requests: int = 100,
    embedding_inputs: int = 1000,
    embedding_tokens: int = 100_000,
    vectors: int = 1000,
    batch_items: int = 1000,
    concurrent_requests: int = 10,
) -> None:
    sid = scope.scope_id
    limits = [
        _limit(sid, UsageDimension.REQUESTS, requests),
        _limit(sid, UsageDimension.EMBEDDING_INPUTS, embedding_inputs),
        _limit(sid, UsageDimension.EMBEDDING_TOKENS, embedding_tokens),
        _limit(sid, UsageDimension.VECTORS, vectors),
        _limit(sid, UsageDimension.BATCH_ITEMS, batch_items),
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
    ]
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
    labels.setdefault("input_types", "text")
    labels.setdefault("normalization", "unit")
    labels.setdefault("locality", "remote")
    labels.setdefault("embedding_dimensions", "2")
    binding_id = stable_id(
        "binding",
        "embeddings",
        scope.provider_id,
        provider_key,
        binding_salt or provider_key,
    )
    return StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        model_id=stable_id("model", "embeddings", provider_key),
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
# Estimates / compatibility
# ---------------------------------------------------------------------------


def test_usage_routing_requirement_id_exported() -> None:
    assert USAGE_ROUTING_REQUIREMENT_ID == (
        "requirement:embeddings-router-usage-routing.v1"
    )
    assert embeddings_router.EMBEDDING_USAGE_OPERATION == "embedding.generate"


def test_estimate_embedding_usage_covers_modality_dimensions() -> None:
    texts = ["hello world", "x" * 40]
    vector = estimate_embedding_usage(texts, expected_dimension=384)
    assert isinstance(vector, UsageVector)
    assert vector.get(UsageDimension.REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.EMBEDDING_INPUTS).amount.value == 2
    assert vector.get(UsageDimension.VECTORS).amount.value == 2
    assert vector.get(UsageDimension.BATCH_ITEMS).amount.value == 2
    assert vector.get(UsageDimension.EMBEDDING_TOKENS).amount.value >= 2
    assert vector.get(UsageDimension.CONCURRENT_REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.MEDIA_BYTES).amount.value == sum(
        len(t.encode("utf-8")) for t in texts
    )
    # Token estimate is conservative.
    assert estimate_embedding_tokens("abcd") >= 1
    settled = settle_embedding_usage(texts, vectors_produced=2)
    assert settled.get(UsageDimension.VECTORS).amount.value == 2


def test_cache_only_estimate_creates_no_remote_envelope() -> None:
    empty = estimate_embedding_usage(["a"], remote=False)
    assert empty.entries == ()


def test_embedding_fallback_compatible_rejects_dimension_drift() -> None:
    origin = {
        "input_types": "text",
        "normalization": "unit",
        "locality": "remote",
        "embedding_dimensions": "1536",
    }
    ok = dict(origin)
    bad = dict(origin, embedding_dimensions="384")
    assert embedding_fallback_compatible(origin, ok) is True
    assert embedding_fallback_compatible(origin, bad) is False
    assert (
        embedding_fallback_compatible(
            origin, dict(origin, normalization="l2")
        )
        is False
    )


# ---------------------------------------------------------------------------
# Off / observe / enforce modes
# ---------------------------------------------------------------------------


def test_off_mode_identical_to_legacy_selection() -> None:
    provider = _CountingProvider()
    vectors = embed_texts(
        ["a", "b"],
        provider_instance=provider,
    )
    assert vectors == [[0.0, 0.5], [1.0, 1.5]]
    assert len(provider.calls) == 1

    provider2 = _CountingProvider()
    vectors2 = embed_texts(
        ["a", "b"],
        provider_instance=provider2,
        usage_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert vectors2 == vectors
    # No coordinator + off policy does not invent remote charges.
    admission = get_last_usage_admission()
    assert admission.get("mode") == "off" or admission.get("final_status") in {
        "off",
        None,
        "",
    }


def test_observe_mode_never_changes_selection_or_charges() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("observe")
    _configure_embedding_limits(coord, scope, requests=0)  # would deny enforce
    provider = _CountingProvider("observe_provider")

    vectors = embed_texts(
        ["alpha", "beta"],
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.OBSERVE),
        usage_scope_id=scope.scope_id,
        usage_request_id="req-observe-1",
    )
    assert len(vectors) == 2
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["remote_charged"] is False
    assert "no_selection_change" in admission["reason_codes"]
    assert_no_prompt_media_or_output(admission)
    # No reservation held / charged against exhausted request limit.
    snap = coord.snapshot(scope.scope_id)
    assert _headroom_available(snap, UsageDimension.REQUESTS) == 0


def test_enforce_reserves_before_dispatch_and_settles_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("enforce")
    _configure_embedding_limits(coord, scope, requests=5, embedding_inputs=50)
    cand = _candidate(provider_key="enforce", scope=scope)
    provider = _CountingProvider("enforce")

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)
    before_inputs = _headroom_available(before, UsageDimension.EMBEDDING_INPUTS)

    sample = ["alpha-input", "beta-input", "gamma-input"]
    vectors = embed_texts(
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
            required=estimate_embedding_usage(sample),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        usage_expected_dimension=2,
    )
    assert len(vectors) == 3
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["reservation_id"]
    assert admission["receipt_id"]
    assert_no_prompt_media_or_output(admission)
    if "receipt" in admission:
        assert_no_prompt_media_or_output(admission["receipt"])
        # Receipts must not embed source text or vectors.
        encoded = repr(admission["receipt"]).casefold()
        assert "alpha-input" not in encoded
        assert "beta-input" not in encoded
        assert "[0.0, 0.5]" not in encoded

    after = coord.snapshot(scope.scope_id)
    after_req = _headroom_available(after, UsageDimension.REQUESTS)
    after_inputs = _headroom_available(after, UsageDimension.EMBEDDING_INPUTS)
    assert before_req is not None and after_req is not None
    assert after_req == before_req - 1
    assert before_inputs is not None and after_inputs is not None
    assert after_inputs == before_inputs - 3

    # Idempotent replay must not double-charge.
    embed_texts(
        sample,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_embedding_usage(sample),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        usage_expected_dimension=2,
    )
    replay = coord.snapshot(scope.scope_id)
    assert _headroom_available(replay, UsageDimension.REQUESTS) == after_req
    assert _headroom_available(replay, UsageDimension.EMBEDDING_INPUTS) == after_inputs


def test_enforce_denies_when_capacity_exhausted() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("deny")
    _configure_embedding_limits(coord, scope, requests=0)
    cand = _candidate(provider_key="deny", scope=scope)
    provider = _CountingProvider("deny")

    with pytest.raises(UsageCapacityError) as excinfo:
        embed_texts(
            ["blocked"],
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=estimate_embedding_usage(["blocked"]),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-deny-1",
            usage_idempotency_key="idem-deny-1",
        )
    assert provider.calls == []
    assert excinfo.value.reason_codes
    admission = get_last_usage_admission()
    assert admission["success"] is False


# ---------------------------------------------------------------------------
# Cache hits, batch split, cancel/partial, pins, fallback
# ---------------------------------------------------------------------------


def test_cache_hits_create_no_remote_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    clear_embeddings_router_caches()

    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cache")
    _configure_embedding_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cache", scope=scope)
    provider = _CountingProvider("cache")
    deps = RouterDeps()

    # Seed the response cache via a first remote call.
    embed_texts(
        ["cached-text"],
        provider_instance=provider,
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_embedding_usage(["cached-text"]),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-1",
        usage_idempotency_key="idem-cache-1",
    )
    after_first = coord.snapshot(scope.scope_id)
    first_req = _headroom_available(after_first, UsageDimension.REQUESTS)
    assert len(provider.calls) == 1

    # Second call is a full cache hit — no additional remote charge.
    vectors = embed_texts(
        ["cached-text"],
        provider_instance=provider,
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_embedding_usage(["cached-text"]),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-2",
        usage_idempotency_key="idem-cache-2",
    )
    assert vectors
    assert len(provider.calls) == 1  # no second remote call
    admission = get_last_usage_admission()
    assert admission.get("remote_charged") is False
    assert "no_remote_charge" in admission.get("reason_codes", [])
    after_second = coord.snapshot(scope.scope_id)
    assert _headroom_available(after_second, UsageDimension.REQUESTS) == first_req


def test_physical_sub_batches_settle_exactly_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("batch")
    _configure_embedding_limits(
        coord, scope, requests=20, embedding_inputs=100, batch_items=100
    )
    cand = _candidate(provider_key="batch", scope=scope)
    provider = _CountingProvider("batch")

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)
    before_items = _headroom_available(before, UsageDimension.BATCH_ITEMS)

    vectors = embed_texts_batched(
        ["0", "1", "2", "3", "4"],
        batch_size=2,
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
        usage_expected_dimension=2,
    )
    assert len(vectors) == 5
    # 3 physical sub-batches (2+2+1)
    assert len(provider.calls) == 3
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req - 3
    assert _headroom_available(after, UsageDimension.BATCH_ITEMS) == before_items - 5
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission.get("completed_sub_batches") == 3


def test_partial_batch_preserves_completed_member_usage() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("partial")
    _configure_embedding_limits(coord, scope, requests=20, embedding_inputs=100)
    cand = _candidate(provider_key="partial", scope=scope)

    class _FailAfterFirst(_CountingProvider):
        def __init__(self) -> None:
            super().__init__("partial")
            self.batch_index = 0

        def embed_texts(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            with self.lock:
                self.batch_index += 1
                items = list(texts)
                self.calls.append(items)
                if self.batch_index >= 2:
                    raise RuntimeError("simulated_provider_outage")
            return [[float(i), float(i) + 0.5] for i, _ in enumerate(items)]

    provider = _FailAfterFirst()
    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(RuntimeError, match="simulated_provider_outage"):
        embed_texts_batched(
            ["0", "1", "2", "3"],
            batch_size=2,
            max_workers=1,
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-partial-1",
            usage_idempotency_key="idem-partial-1",
        )

    after = coord.snapshot(scope.scope_id)
    # First physical sub-batch (2 items) settled; second failed after dispatch
    # may charge conservatively, but completed work is never rolled back to zero.
    after_req = _headroom_available(after, UsageDimension.REQUESTS)
    assert before_req is not None and after_req is not None
    assert after_req <= before_req - 1
    admission = get_last_usage_admission()
    assert "partial_completion_preserved" in admission.get("reason_codes", [])


def test_cancel_before_dispatch_does_not_charge() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cancel")
    _configure_embedding_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cancel", scope=scope)
    provider = _CountingProvider("cancel")
    cancel = threading.Event()
    cancel.set()

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(UsageCapacityError):
        embed_texts(
            ["never"],
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=estimate_embedding_usage(["never"]),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-cancel-1",
            usage_idempotency_key="idem-cancel-1",
            usage_cancel_event=cancel,
        )
    assert provider.calls == []
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req


def test_explicit_provider_pin_defaults_to_no_fallback() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("pin-a")
    scope_b = _scope("pin-b")
    _configure_embedding_limits(coord, scope_a, requests=0)
    _configure_embedding_limits(coord, scope_b, requests=10)
    cand_a = _candidate(provider_key="pin-a", scope=scope_a, score=100)
    cand_b = _candidate(provider_key="pin-b", scope=scope_b, score=1)
    provider_a = _CountingProvider("pin-a")
    provider_b = _CountingProvider("pin-b")

    with pytest.raises(UsageCapacityError):
        embed_texts(
            ["pinned"],
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
                required=estimate_embedding_usage(["pinned"]),
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
    _configure_embedding_limits(coord, scope_a, requests=10)
    _configure_embedding_limits(coord, scope_b, requests=10)
    labels = {
        "input_types": "text",
        "normalization": "unit",
        "locality": "remote",
        "embedding_dimensions": "2",
    }
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

    vectors = embed_texts(
        ["fallback-me"],
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
            required=estimate_embedding_usage(["fallback-me"]),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-fb-1",
        usage_idempotency_key="idem-fb-1",
        usage_expected_dimension=2,
    )
    assert len(vectors) == 1
    assert provider_b.calls  # second candidate used
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["selected_binding_id"] == cand_b.binding_id


def test_incompatible_embedding_never_substitutes() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("bad-a")
    scope_b = _scope("bad-b")
    _configure_embedding_limits(coord, scope_a, requests=0)
    _configure_embedding_limits(coord, scope_b, requests=10)
    cand_a = _candidate(
        provider_key="bad-a",
        scope=scope_a,
        score=50,
        labels={
            "input_types": "text",
            "normalization": "unit",
            "locality": "remote",
            "embedding_dimensions": "2",
        },
    )
    cand_b = _candidate(
        provider_key="bad-b",
        scope=scope_b,
        score=10,
        labels={
            "input_types": "text",
            "normalization": "unit",
            "locality": "remote",
            "embedding_dimensions": "384",  # incompatible
        },
    )
    provider_a = _CountingProvider("bad-a", dimension=2)
    provider_b = _CountingProvider("bad-b", dimension=384)

    with pytest.raises(UsageCapacityError):
        embed_texts(
            ["no-sub"],
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
            usage_request=UsageRoutingRequest(
                required=estimate_embedding_usage(["no-sub"]),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-nosub-1",
            usage_idempotency_key="idem-nosub-1",
            usage_expected_dimension=2,
        )
    assert provider_b.calls == []


def test_output_shape_validation_remains_authoritative() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("shape")
    _configure_embedding_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="shape", scope=scope)

    class _BadShape:
        router_provider_name = "shape"

        def embed_texts(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            return [[1.0, 2.0]]  # wrong count

    with pytest.raises(EmbeddingsRouterError, match="vectors for"):
        embed_texts(
            ["a", "b"],
            provider_instance=_BadShape(),  # type: ignore[arg-type]
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: _BadShape()},  # type: ignore[dict-item]
            usage_request=UsageRoutingRequest(
                required=estimate_embedding_usage(["a", "b"]),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-shape-1",
            usage_idempotency_key="idem-shape-1",
        )


def test_receipt_never_contains_source_text_or_vectors() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("receipt")
    _configure_embedding_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="receipt", scope=scope)
    provider = _CountingProvider("receipt")
    secret_text = "super_secret_source_text_xyz"

    embed_texts(
        [secret_text],
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_embedding_usage([secret_text]),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-receipt-1",
        usage_idempotency_key="idem-receipt-1",
    )
    admission = get_last_usage_admission()
    assert_no_prompt_media_or_output(admission)
    blob = repr(admission)
    assert secret_text not in blob
    assert "0.0, 0.5" not in blob
    trace = get_last_embedding_trace()
    assert secret_text not in repr(trace)
