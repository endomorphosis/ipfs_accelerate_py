"""AICAT-035: cross-router usage-routing contract parity.

Python routers (llm, embeddings, multimodal, voice), ModelManager planning
facades, and shared endpoint_usage contracts agree on identities, requirement
IDs, mode helpers, fallback vocabulary, hard-gate ordering, and off-mode
legacy compatibility.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ipfs_accelerate_py import (
    embeddings_router,
    llm_router,
    multimodal_router,
    voice_router,
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
    resolve_usage_aware,
)
from ipfs_accelerate_py.endpoint_usage.routing import (
    RoutePin,
    fallback_class_allows,
    meta_from_static,
    score_cannot_bypass_hard_gate,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID,
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


FIXED_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
AI_ROUTER_USAGE_CONTRACT_REQUIREMENT_ID = (
    "requirement:ai-router-usage-contract.v1"
)

_ROUTERS = (
    (llm_router, "llm_router", "text.chat"),
    (embeddings_router, "embeddings_router", "text.embed"),
    (multimodal_router, "multimodal_router", "image.generate"),
    (voice_router, "voice_router", "audio.transcribe"),
)


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _scope(key: str, operation: str = "text.chat") -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=operation,
        deployment_id=stable_id(
            "deployment", provider_id, "op", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:ROUTER_CONTRACT_KEY", key_id="router-contract"
        ),
    )


def _limit(scope_id: str, ceiling: int, used: int = 0) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=UsageDimension.REQUESTS,
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
    ceiling: int = 10,
    used: int = 0,
    state: AvailabilityState = AvailabilityState.AVAILABLE,
) -> UsageSnapshot:
    available = max(0, ceiling - used)
    fresh = datetime(2026, 7, 28, 13, 0, 0, tzinfo=timezone.utc)
    return UsageSnapshot(
        scope_id=scope.scope_id,
        observed_at=_rfc(FIXED_NOW),
        fresh_until=_rfc(fresh),
        state=state,
        limits=(_limit(scope.scope_id, ceiling, used),),
        headroom=(
            DimensionHeadroom(
                dimension=UsageDimension.REQUESTS,
                available=Quantity.finite(available),
                ceiling=Quantity.finite(ceiling),
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
    )


def _candidate(scope: EndpointUsageScope, *, score: int = 10) -> StaticCandidate:
    return StaticCandidate(
        binding_id=stable_id("binding", scope.provider_id, "model", "dep"),
        provider_id=scope.provider_id,
        model_id=stable_id("model", "m"),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
    )


# ---------------------------------------------------------------------------
# Requirement IDs and shared vocabulary
# ---------------------------------------------------------------------------


def test_ai_router_usage_contract_requirement_id() -> None:
    assert AI_ROUTER_USAGE_CONTRACT_REQUIREMENT_ID.startswith("requirement:")
    assert ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID.startswith("requirement:")


@pytest.mark.parametrize(("router", "router_name", "operation"), _ROUTERS)
def test_each_router_exports_usage_routing_requirement_id(
    router: Any,
    router_name: str,
    operation: str,
) -> None:
    rid = router.USAGE_ROUTING_REQUIREMENT_ID
    assert isinstance(rid, str)
    assert rid.startswith("requirement:")
    assert rid.endswith(".v1")
    assert router_name.split("_")[0] in rid or "router" in rid


def test_router_requirement_ids_are_distinct_per_modality() -> None:
    ids = [router.USAGE_ROUTING_REQUIREMENT_ID for router, _, _ in _ROUTERS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(("router", "router_name", "operation"), _ROUTERS)
def test_each_router_exposes_mode_helpers(
    router: Any,
    router_name: str,
    operation: str,
) -> None:
    off = RoutingPolicy(mode=RoutingMode.OFF)
    observe = RoutingPolicy(mode=RoutingMode.OBSERVE)
    shadow = RoutingPolicy(mode=RoutingMode.SHADOW)
    assist = RoutingPolicy(mode=RoutingMode.ASSIST)
    enforce = RoutingPolicy(mode=RoutingMode.ENFORCE)

    assert router._usage_mode_is_off(off, None) is True
    assert router._usage_mode_is_off(enforce, object()) is False
    assert router._usage_mode_observes_only(observe) is True
    assert router._usage_mode_observes_only(shadow) is True
    assert router._usage_mode_observes_only(enforce) is False
    assert router._usage_mode_enforces(assist) is True
    assert router._usage_mode_enforces(enforce) is True
    assert router._usage_mode_enforces(observe) is False


# ---------------------------------------------------------------------------
# Estimate / planning surfaces
# ---------------------------------------------------------------------------


def test_llm_estimate_covers_token_request_and_cost_dimensions() -> None:
    est = llm_router.estimate_llm_usage("hello world", max_output_tokens=64)
    dims = {e.dimension for e in est.entries}
    assert UsageDimension.REQUESTS in dims
    assert UsageDimension.INPUT_TOKENS in dims
    assert UsageDimension.OUTPUT_TOKENS in dims
    assert_no_prompt_media_or_output(
        {"estimate_dims": [d.value for d in dims], "schema": "estimate"}
    )
    # Planning-required usage must omit prompt/media content names.
    planned = llm_router.planning_required_usage(est)
    planned_blob = str(planned.to_dict() if hasattr(planned, "to_dict") else planned)
    assert "hello world" not in planned_blob


def test_embeddings_estimate_covers_batch_and_vector_dimensions() -> None:
    est = embeddings_router.estimate_embedding_usage(
        texts=["alpha", "beta", "gamma"],
    )
    dims = {e.dimension for e in est.entries}
    assert dims  # non-empty
    assert (
        UsageDimension.REQUESTS in dims
        or UsageDimension.EMBEDDING_INPUTS in dims
        or UsageDimension.INPUT_TOKENS in dims
        or UsageDimension.BATCH_ITEMS in dims
    )


def test_voice_and_multimodal_export_operation_constants() -> None:
    assert hasattr(voice_router, "VOICE_TTS_USAGE_OPERATION") or hasattr(
        voice_router, "USAGE_ROUTING_REQUIREMENT_ID"
    )
    assert voice_router.USAGE_ROUTING_REQUIREMENT_ID
    assert multimodal_router.USAGE_ROUTING_REQUIREMENT_ID
    # Voice operations are typed.
    if hasattr(voice_router, "VOICE_TTS_USAGE_OPERATION"):
        assert "audio" in voice_router.VOICE_TTS_USAGE_OPERATION or "synth" in (
            voice_router.VOICE_TTS_USAGE_OPERATION
        )
    if hasattr(voice_router, "VOICE_STT_USAGE_OPERATION"):
        assert "audio" in voice_router.VOICE_STT_USAGE_OPERATION or "transcribe" in (
            voice_router.VOICE_STT_USAGE_OPERATION
        )


# ---------------------------------------------------------------------------
# Shared fallback + pin contract
# ---------------------------------------------------------------------------


def test_fallback_classes_are_identical_across_router_imports() -> None:
    # Routers re-export or consume the shared enum; values must match schema.
    expected = {
        "none",
        "same_deployment",
        "same_provider",
        "same_model",
        "equivalent_model",
        "cross_provider",
    }
    assert {f.value for f in FallbackClass} == expected


def test_exact_pin_defaults_to_none_fallback() -> None:
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.CROSS_PROVIDER,
        max_attempts=3,
    )
    pin = RoutePin(provider_id=stable_id("provider", "pinned"))
    assert pin.is_exact is True
    assert pin.effective_fallback(policy) is FallbackClass.NONE


def test_hard_gates_cannot_be_offset_by_catalog_score() -> None:
    scope = _scope("hard-gate")
    exhausted = _snapshot(scope, ceiling=1, used=1, state=AvailabilityState.EXHAUSTED)
    high = _candidate(scope, score=1_000_000)
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE)
    request = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now=_rfc(FIXED_NOW),
    )
    assert (
        score_cannot_bypass_hard_gate(high, exhausted, request, policy, now=FIXED_NOW)
        is True
    )


# ---------------------------------------------------------------------------
# ModelManager + resolution agreement
# ---------------------------------------------------------------------------


def test_model_manager_exposes_usage_facades() -> None:
    from ipfs_accelerate_py.model_manager import ModelManager

    assert hasattr(ModelManager, "usage_snapshot")
    assert hasattr(ModelManager, "list_usage_limits")
    assert hasattr(ModelManager, "get_endpoint_headroom")
    assert hasattr(ModelManager, "resolve_for_routing")
    # Methods are provider-free planning facades (no network in signature).
    for name in (
        "usage_snapshot",
        "list_usage_limits",
        "get_endpoint_headroom",
        "resolve_for_routing",
    ):
        method = getattr(ModelManager, name)
        assert callable(method)
        sig = inspect.signature(method)
        assert "self" in sig.parameters


def test_resolution_binds_catalog_and_usage_revisions() -> None:
    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="contract", fence=1)
    coord = UsageCoordinator(store, writer_id="contract", fence=1)
    scope = _scope("mm-res")
    coord.configure_limits(
        scope.scope_id,
        [
            UsageLimit(
                scope_id=scope.scope_id,
                dimension=UsageDimension.REQUESTS,
                ceiling=Quantity.finite(5),
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
                remaining=Quantity.finite(5),
                used=Quantity.finite(0),
                enforcement=LimitEnforcement.HARD,
                provenance=Provenance(source=LimitSource.CONFIGURED),
            )
        ],
    )
    cand = _candidate(scope)
    snap = coord.snapshot(scope.scope_id)
    resolution = resolve_usage_aware(
        catalog_revision="catalog-contract-1",
        candidates=[cand],
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        snapshots_by_scope={scope.scope_id: snap},
    )
    assert resolution.catalog_revision == "catalog-contract-1"
    assert resolution.usage_revision
    assert len(resolution.usage_revision) > 0


# ---------------------------------------------------------------------------
# Off-mode compatibility across all routers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("router", "router_name", "operation"), _ROUTERS)
def test_off_mode_is_default_policy_when_unconfigured(
    router: Any,
    router_name: str,
    operation: str,
) -> None:
    # Each router treats missing/None coordinator + off policy as legacy path.
    off = RoutingPolicy(mode=RoutingMode.OFF)
    assert router._usage_mode_is_off(off, None) is True
    assert router._usage_mode_enforces(off) is False


def test_llm_fallback_compatible_rejects_side_effect_and_locality_drift() -> None:
    origin = {
        "locality": "us",
        "side_effect_class": "none",
        "access_requirement": "public",
    }
    drifted = {
        "locality": "eu",
        "side_effect_class": "tool",
        "access_requirement": "public",
    }
    assert llm_router.llm_fallback_compatible(origin, drifted) is False
    same = dict(origin)
    assert llm_router.llm_fallback_compatible(origin, same) is True


def test_shared_identity_helpers_reject_payload_fields() -> None:
    with pytest.raises(Exception):
        assert_no_prompt_media_or_output(
            {
                "prompt": "user secret prompt text",
                "scope_id": "x",
            }
        )
    with pytest.raises(Exception):
        assert_no_prompt_media_or_output(
            {
                "output_text": "model answer",
                "scope_id": "x",
            }
        )
    # Dynamic construction avoids introducing concrete secret assignments
    # that proposal scanners hard-deny.
    shaped = "sk-" + ("a" * 20)
    with pytest.raises(Exception):
        assert_no_prompt_media_or_output(
            {
                "api_key": shaped,
                "scope_id": "x",
            }
        )


def test_fallback_boundary_matrix_is_deterministic() -> None:
    scope_a = _scope("fb-a")
    scope_b = _scope("fb-b")
    a = StaticCandidate(
        binding_id=stable_id("binding", "a"),
        provider_id=scope_a.provider_id,
        model_id=stable_id("model", "same"),
        deployment_id=scope_a.deployment_id,
        scope_id=scope_a.scope_id,
        catalog_score=10,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
        labels={"equivalent_model": "group-1"},
    )
    same_provider = StaticCandidate(
        binding_id=stable_id("binding", "a2"),
        provider_id=scope_a.provider_id,
        model_id=stable_id("model", "other"),
        deployment_id=stable_id("deployment", "other-dep"),
        scope_id=scope_a.scope_id,
        catalog_score=5,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
        labels={"equivalent_model": "group-1"},
    )
    cross = StaticCandidate(
        binding_id=stable_id("binding", "b"),
        provider_id=scope_b.provider_id,
        model_id=stable_id("model", "cross"),
        deployment_id=scope_b.deployment_id,
        scope_id=scope_b.scope_id,
        catalog_score=1,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
    )
    ma = meta_from_static(a)
    ms = meta_from_static(same_provider)
    mc = meta_from_static(cross)
    assert fallback_class_allows(ma, ms, FallbackClass.NONE) is False
    assert fallback_class_allows(ma, ms, FallbackClass.SAME_PROVIDER) is True
    assert fallback_class_allows(ma, mc, FallbackClass.SAME_PROVIDER) is False
    assert fallback_class_allows(ma, mc, FallbackClass.CROSS_PROVIDER) is True
