"""Usage-aware admission integration for multimodal_router (AICAT-032)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import pytest

import ipfs_accelerate_py.multimodal_router as multimodal_router
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
    EndpointUsageScope,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.multimodal_router import (
    MULTIMODAL_USAGE_OPERATION,
    USAGE_ROUTING_REQUIREMENT_ID,
    MultimodalRouterError,
    UsageCapacityError,
    clear_multimodal_router_caches,
    estimate_multimodal_usage,
    estimate_text_tokens,
    generate_multimodal,
    get_last_multimodal_trace,
    get_last_usage_admission,
    inspect_media_reference,
    multimodal_fallback_compatible,
    settle_multimodal_usage,
    validate_multimodal_media_input,
)
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
    monkeypatch.setattr(multimodal_router, "_PROVIDER_REGISTRY", {})
    clear_multimodal_router_caches()


class _CountingProvider:
    """Deterministic multimodal provider for usage tests."""

    def __init__(
        self,
        name: str = "counting_fixture",
        *,
        fail_times: int = 0,
        fail_exc: Optional[BaseException] = None,
        reply: str = "ok",
    ) -> None:
        self.router_provider_name = name
        self.calls: List[Dict[str, object]] = []
        self.fail_times = fail_times
        self.fail_exc = fail_exc or RuntimeError("provider_fail")
        self.reply = reply
        self.lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, bytes]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        with self.lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_exc
            self.calls.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "model_name": model_name,
                    "device": device,
                    "kwargs": dict(kwargs),
                }
            )
        return f"{self.reply}:{model_name or 'none'}:{prompt}"


def _scope(provider_key: str = "mm-a") -> EndpointUsageScope:
    provider_id = stable_id("provider", "multimodal", provider_key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=MULTIMODAL_USAGE_OPERATION,
        deployment_id=stable_id(
            "deployment", provider_id, "vision", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:MM_USAGE_TEST_KEY", key_id="mm-usage-default"
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
    return UsageCoordinator(store, writer_id="mm-usage-test", fence=1)


def _configure_multimodal_limits(
    coord: UsageCoordinator,
    scope: EndpointUsageScope,
    *,
    requests: int = 100,
    images: int = 100,
    pixels: int = 100_000_000,
    media_bytes: int = 50_000_000,
    input_tokens: int = 100_000,
    output_tokens: int = 100_000,
    concurrent_requests: int = 10,
) -> None:
    sid = scope.scope_id
    limits = [
        _limit(sid, UsageDimension.REQUESTS, requests),
        _limit(sid, UsageDimension.IMAGES, images),
        _limit(sid, UsageDimension.PIXELS, pixels),
        _limit(sid, UsageDimension.MEDIA_BYTES, media_bytes),
        _limit(sid, UsageDimension.INPUT_TOKENS, input_tokens),
        _limit(sid, UsageDimension.OUTPUT_TOKENS, output_tokens),
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
    labels.setdefault("operation", MULTIMODAL_USAGE_OPERATION)
    labels.setdefault("locality", "remote")
    labels.setdefault("output_media_types", "text/plain")
    labels.setdefault("input_media_types", "image/*,text/plain")
    labels.setdefault("image_input_modes", "inline,uri")
    labels.setdefault("max_images", "1")
    labels.setdefault("mime_family", "image/*")
    labels.setdefault("requires_remote_upload", "0")
    binding_id = stable_id(
        "binding",
        "multimodal",
        scope.provider_id,
        provider_key,
        binding_salt or provider_key,
    )
    return StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        model_id=stable_id("model", "multimodal", provider_key),
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
# Estimates / compatibility / media policy
# ---------------------------------------------------------------------------


def test_usage_routing_requirement_id_exported() -> None:
    assert USAGE_ROUTING_REQUIREMENT_ID == (
        "requirement:multimodal-router-usage-routing.v1"
    )
    assert MULTIMODAL_USAGE_OPERATION == "multimodal.generate"


def test_estimate_multimodal_usage_covers_modality_dimensions() -> None:
    prompt = "describe this scene carefully"
    image = b"\x89PNG" + b"x" * 200
    vector = estimate_multimodal_usage(
        prompt,
        image=image,
        max_output_tokens=64,
        width=32,
        height=32,
    )
    assert isinstance(vector, UsageVector)
    assert vector.get(UsageDimension.REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.IMAGES).amount.value == 1
    assert vector.get(UsageDimension.PIXELS).amount.value == 32 * 32
    assert vector.get(UsageDimension.MEDIA_BYTES).amount.value == len(image)
    assert vector.get(UsageDimension.INPUT_TOKENS).amount.value >= estimate_text_tokens(
        prompt
    )
    assert vector.get(UsageDimension.OUTPUT_TOKENS).amount.value == 64
    assert vector.get(UsageDimension.CONCURRENT_REQUESTS).amount.value == 1
    settled = settle_multimodal_usage(
        prompt, image=image, output_text="a short caption", width=32, height=32
    )
    assert settled.get(UsageDimension.REQUESTS).amount.value == 1
    assert settled.get(UsageDimension.IMAGES).amount.value == 1


def test_cache_only_estimate_creates_no_remote_envelope() -> None:
    empty = estimate_multimodal_usage("hello", image=b"x", remote=False)
    assert empty.entries == ()


def test_media_remains_referenced_not_embedded_in_facts() -> None:
    secret = b"SUPER_SECRET_IMAGE_BYTES_XYZ"
    facts = inspect_media_reference(secret, width=10, height=10)
    assert facts.image_count == 1
    assert facts.media_bytes == len(secret)
    assert facts.local_only is True
    blob = repr(facts)
    assert "SUPER_SECRET_IMAGE_BYTES_XYZ" not in blob


def test_multimodal_fallback_compatible_rejects_contract_drift() -> None:
    origin = {
        "operation": MULTIMODAL_USAGE_OPERATION,
        "locality": "remote",
        "output_media_types": "text/plain",
        "mime_family": "image/*",
        "image_count": "1",
        "max_images": "1",
        "image_input_mode": "inline",
        "forbid_remote_upload": "1",
        "input_mime": "image/png",
    }
    ok = dict(origin, image_input_modes="inline,uri", requires_remote_upload="0")
    assert multimodal_fallback_compatible(origin, ok) is True
    assert (
        multimodal_fallback_compatible(
            origin, dict(ok, locality="local")
        )
        is False
    )
    assert (
        multimodal_fallback_compatible(
            origin, dict(ok, requires_remote_upload="1")
        )
        is False
    )
    assert (
        multimodal_fallback_compatible(
            origin,
            dict(ok, image_input_modes="uri", requires_remote_upload="0"),
        )
        is False
    )


def test_adversarial_ssrf_and_mime_fail_before_reservation() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("ssrf")
    _configure_multimodal_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="ssrf", scope=scope)
    provider = _CountingProvider("ssrf")

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(MultimodalRouterError, match="SSRF|scheme|blocked"):
        generate_multimodal(
            "peek",
            image="http://169.254.169.254/latest/meta-data/",
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
            usage_request_id="req-ssrf-1",
            usage_idempotency_key="idem-ssrf-1",
        )
    assert provider.calls == []
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req

    with pytest.raises(MultimodalRouterError, match="scheme"):
        validate_multimodal_media_input("file:///etc/passwd")

    with pytest.raises(MultimodalRouterError, match="MIME"):
        validate_multimodal_media_input(
            "data:application/x-msdownload;base64,AAA="
        )

    with pytest.raises(MultimodalRouterError, match="max_media_bytes"):
        validate_multimodal_media_input(b"x" * 100, max_media_bytes=10)


# ---------------------------------------------------------------------------
# Off / observe / enforce modes
# ---------------------------------------------------------------------------


def test_off_mode_identical_to_legacy_selection() -> None:
    provider = _CountingProvider()
    text = generate_multimodal(
        "caption",
        image=b"img",
        provider_instance=provider,
        model_name="v1",
    )
    assert text == "ok:v1:caption"
    assert len(provider.calls) == 1

    provider2 = _CountingProvider()
    text2 = generate_multimodal(
        "caption",
        image=b"img",
        provider_instance=provider2,
        model_name="v1",
        usage_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert text2 == text
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
    _configure_multimodal_limits(coord, scope, requests=0)
    provider = _CountingProvider("observe_provider")

    text = generate_multimodal(
        "alpha",
        image=b"pixels",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.OBSERVE),
        usage_scope_id=scope.scope_id,
        usage_request_id="req-observe-1",
    )
    assert "alpha" in text
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["remote_charged"] is False
    assert "no_selection_change" in admission["reason_codes"]
    assert_no_prompt_media_or_output(admission)
    snap = coord.snapshot(scope.scope_id)
    assert _headroom_available(snap, UsageDimension.REQUESTS) == 0


def test_enforce_reserves_before_dispatch_and_settles_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("enforce")
    _configure_multimodal_limits(coord, scope, requests=5, images=50)
    cand = _candidate(provider_key="enforce", scope=scope)
    provider = _CountingProvider("enforce")
    image = b"\x89PNG" + b"payload-bytes"

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)
    before_images = _headroom_available(before, UsageDimension.IMAGES)

    text = generate_multimodal(
        "describe-me",
        image=image,
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
            required=estimate_multimodal_usage("describe-me", image=image),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        usage_width=64,
        usage_height=64,
        max_tokens=32,
    )
    assert "describe-me" in text
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["reservation_id"]
    assert admission["receipt_id"]
    assert_no_prompt_media_or_output(admission)
    if "receipt" in admission:
        assert_no_prompt_media_or_output(admission["receipt"])
        encoded = repr(admission["receipt"]).casefold()
        assert "describe-me" not in encoded
        assert "payload-bytes" not in encoded

    after = coord.snapshot(scope.scope_id)
    after_req = _headroom_available(after, UsageDimension.REQUESTS)
    after_images = _headroom_available(after, UsageDimension.IMAGES)
    assert before_req is not None and after_req is not None
    assert after_req == before_req - 1
    assert before_images is not None and after_images is not None
    assert after_images == before_images - 1

    # Idempotent replay must not double-charge.
    generate_multimodal(
        "describe-me",
        image=image,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_multimodal_usage("describe-me", image=image),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-enforce-1",
        usage_idempotency_key="idem-enforce-1",
        usage_width=64,
        usage_height=64,
        max_tokens=32,
    )
    replay = coord.snapshot(scope.scope_id)
    assert _headroom_available(replay, UsageDimension.REQUESTS) == after_req
    assert _headroom_available(replay, UsageDimension.IMAGES) == after_images


def test_enforce_denies_when_capacity_exhausted() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("deny")
    _configure_multimodal_limits(coord, scope, requests=0)
    cand = _candidate(provider_key="deny", scope=scope)
    provider = _CountingProvider("deny")

    with pytest.raises(UsageCapacityError) as excinfo:
        generate_multimodal(
            "blocked",
            image=b"x",
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=estimate_multimodal_usage("blocked", image=b"x"),
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
# Cache hits, cancel, pins, fallback, receipt safety
# ---------------------------------------------------------------------------


def test_cache_hits_create_no_remote_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    clear_multimodal_router_caches()

    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cache")
    _configure_multimodal_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cache", scope=scope)
    provider = _CountingProvider("cache")
    deps = RouterDeps()
    image = b"cached-image"

    generate_multimodal(
        "cached-prompt",
        image=image,
        provider_instance=provider,
        model_name="fixture/model",
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_multimodal_usage("cached-prompt", image=image),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-1",
        usage_idempotency_key="idem-cache-1",
    )
    after_first = coord.snapshot(scope.scope_id)
    first_req = _headroom_available(after_first, UsageDimension.REQUESTS)
    assert len(provider.calls) == 1

    text = generate_multimodal(
        "cached-prompt",
        image=image,
        provider_instance=provider,
        model_name="fixture/model",
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_multimodal_usage("cached-prompt", image=image),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-2",
        usage_idempotency_key="idem-cache-2",
    )
    assert text
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission.get("remote_charged") is False
    assert "no_remote_charge" in admission.get("reason_codes", [])
    after_second = coord.snapshot(scope.scope_id)
    assert _headroom_available(after_second, UsageDimension.REQUESTS) == first_req


def test_cancel_before_dispatch_does_not_charge() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cancel")
    _configure_multimodal_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="cancel", scope=scope)
    provider = _CountingProvider("cancel")
    cancel = threading.Event()
    cancel.set()

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(UsageCapacityError):
        generate_multimodal(
            "never",
            image=b"x",
            provider_instance=provider,
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},
            usage_request=UsageRoutingRequest(
                required=estimate_multimodal_usage("never", image=b"x"),
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
    _configure_multimodal_limits(coord, scope_a, requests=0)
    _configure_multimodal_limits(coord, scope_b, requests=10)
    cand_a = _candidate(provider_key="pin-a", scope=scope_a, score=100)
    cand_b = _candidate(provider_key="pin-b", scope=scope_b, score=1)
    provider_a = _CountingProvider("pin-a")
    provider_b = _CountingProvider("pin-b")

    with pytest.raises(UsageCapacityError):
        generate_multimodal(
            "pinned",
            image=b"x",
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
                required=estimate_multimodal_usage("pinned", image=b"x"),
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
    _configure_multimodal_limits(coord, scope_a, requests=10)
    _configure_multimodal_limits(coord, scope_b, requests=10)
    labels = {
        "operation": MULTIMODAL_USAGE_OPERATION,
        "locality": "remote",
        "output_media_types": "text/plain",
        "mime_family": "image/*",
        "input_media_types": "image/*,text/plain",
        "image_input_modes": "inline,uri",
        "max_images": "1",
        "requires_remote_upload": "0",
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

    text = generate_multimodal(
        "fallback-me",
        image=b"x",
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
            required=estimate_multimodal_usage("fallback-me", image=b"x"),
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


def test_forbidden_remote_upload_route_never_selected() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("up-a")
    scope_b = _scope("up-b")
    _configure_multimodal_limits(coord, scope_a, requests=0)
    _configure_multimodal_limits(coord, scope_b, requests=10)
    cand_a = _candidate(
        provider_key="up-a",
        scope=scope_a,
        score=50,
        labels={
            "operation": MULTIMODAL_USAGE_OPERATION,
            "locality": "remote",
            "output_media_types": "text/plain",
            "mime_family": "image/*",
            "image_input_modes": "inline,uri",
            "requires_remote_upload": "0",
        },
    )
    cand_b = _candidate(
        provider_key="up-b",
        scope=scope_b,
        score=10,
        labels={
            "operation": MULTIMODAL_USAGE_OPERATION,
            "locality": "remote",
            "output_media_types": "text/plain",
            "mime_family": "image/*",
            "image_input_modes": "uri",
            "requires_remote_upload": "1",
        },
    )
    provider_a = _CountingProvider("up-a")
    provider_b = _CountingProvider("up-b")

    with pytest.raises(UsageCapacityError):
        generate_multimodal(
            "no-upload",
            image=b"local-only-bytes",
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
                required=estimate_multimodal_usage(
                    "no-upload", image=b"local-only-bytes"
                ),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-upload-1",
            usage_idempotency_key="idem-upload-1",
        )
    assert provider_b.calls == []


def test_provider_observation_updates_only_exact_scope() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("obs-scope")
    other = _scope("other-scope")
    _configure_multimodal_limits(coord, scope, requests=5)
    _configure_multimodal_limits(coord, other, requests=5)
    cand = _candidate(provider_key="obs-scope", scope=scope)
    provider = _CountingProvider("obs-scope")

    before_other = _headroom_available(
        coord.snapshot(other.scope_id), UsageDimension.REQUESTS
    )
    generate_multimodal(
        "scoped",
        image=b"x",
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_multimodal_usage("scoped", image=b"x"),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-scope-1",
        usage_idempotency_key="idem-scope-1",
        usage_observation={
            "http_status": 200,
            "usage": UsageVector.of(requests=1, images=1),
            "reason_codes": ("provider_usage",),
        },
    )
    after_other = _headroom_available(
        coord.snapshot(other.scope_id), UsageDimension.REQUESTS
    )
    assert after_other == before_other
    admission = get_last_usage_admission()
    assert admission["selected_scope_id"] == scope.scope_id


def test_receipt_never_contains_prompt_or_media() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("receipt")
    _configure_multimodal_limits(coord, scope, requests=5)
    cand = _candidate(provider_key="receipt", scope=scope)
    provider = _CountingProvider("receipt")
    secret_prompt = "super_secret_prompt_xyz"
    secret_image = b"super_secret_image_bytes_xyz"

    generate_multimodal(
        secret_prompt,
        image=secret_image,
        provider_instance=provider,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},
        usage_request=UsageRoutingRequest(
            required=estimate_multimodal_usage(secret_prompt, image=secret_image),
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
    assert "super_secret_image_bytes_xyz" not in blob
    trace = get_last_multimodal_trace()
    assert secret_prompt not in repr(trace)
    assert "super_secret_image_bytes_xyz" not in repr(trace)
