from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ipfs_accelerate_py.model_catalog.cache import (
    CachePolicy,
    CatalogSnapshotCache,
    HealthSample,
    HealthSnapshot,
)
from ipfs_accelerate_py.model_catalog.events import (
    CacheView,
    CatalogEventBus,
    CatalogInvalidationEvent,
    CatalogMetrics,
    credential_state_event,
    deployment_lifecycle_event,
    explicit_refresh_event,
    peer_revision_event,
    registration_event,
)
from ipfs_accelerate_py.model_catalog.receipts import (
    ReceiptValidationError,
    SelectionReceipt,
    SourceProvenance,
    create_selection_receipt,
)
from ipfs_accelerate_py.model_catalog.resolver import (
    CatalogResolver,
    ResolutionRequest,
)
from ipfs_accelerate_py.model_catalog.schema import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.value

    def advance(self, seconds):
        with self._lock:
            self.value += seconds


def _snapshot(name="alpha", priority=2, healthy=True):
    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_CHAT, Operation.STREAM),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
        max_context_tokens=16_384,
    )
    provenance = (
        Provenance(
            source="routers.text",
            source_record_id="internal-record-not-needed",
            observed_at="2030-01-01T00:00:00Z",
            issuer="local-router",
        ),
    )
    provider = ProviderDescriptor(
        name=name,
        capabilities=(capability,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(known=True, configured=True),
        provenance=provenance,
        labels={"locality": "local", "policy.tier": "standard"},
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="%s-chat" % name,
        capabilities=(capability,),
        lifecycle=LifecycleState.READY,
        provenance=provenance,
        labels={"device": "cpu"},
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=model.model_id,
        name="%s-production" % name,
        endpoint_uri="https://%s.example.test/v1" % name,
        capabilities=(capability,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(
            authorized=True,
            reachable=True,
            healthy=healthy,
        ),
        provenance=provenance,
    )
    binding = RouterBinding(
        router="llm_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_CHAT, Operation.STREAM),
        priority=priority,
        state=OperationalState(routable=True),
        provenance=provenance,
    )
    return CatalogSnapshot(
        providers=(provider,),
        models=(model,),
        deployments=(deployment,),
        bindings=(binding,),
    )


def _health(source="routers.text", healthy=True):
    return HealthSnapshot(
        (
            HealthSample(
                source=source,
                record_id="deployment_abc",
                state=OperationalState(healthy=healthy, reachable=True),
                observed_at="2030-01-01T00:00:00Z",
            ),
        )
    )


def test_capability_and_health_entries_have_independent_ttls():
    clock = FakeClock()
    cache = CatalogSnapshotCache(
        CachePolicy(capabilities_ttl=30, health_ttl=5),
        clock=clock,
    )
    calls = {"capabilities": 0, "health": 0}

    def capabilities():
        calls["capabilities"] += 1
        return _snapshot()

    def health():
        calls["health"] += 1
        return _health()

    cache.get_capabilities("routers.text", capabilities)
    cache.get_health("routers.text", health)
    clock.advance(6)
    cache.get_capabilities("routers.text", capabilities)
    cache.get_health("routers.text", health)

    assert calls == {"capabilities": 1, "health": 2}
    assert cache.peek("routers.text", CacheView.CAPABILITIES).stale is False


def test_unchanged_refresh_reuses_snapshot_value_and_content_cid():
    clock = FakeClock()
    cache = CatalogSnapshotCache(clock=clock)
    first_snapshot = _snapshot()
    first = cache.get_or_refresh(
        "routers.text", CacheView.CAPABILITIES, lambda: first_snapshot
    )
    clock.advance(1)
    equivalent = CatalogSnapshot.from_dict(first_snapshot.to_dict())
    second = cache.get_or_refresh(
        "routers.text",
        CacheView.CAPABILITIES,
        lambda: equivalent,
        force=True,
    )

    assert first.cid == second.cid == first_snapshot.revision
    assert second.value is first_snapshot
    assert second.stored_at > first.stored_at


def test_typed_events_invalidate_only_affected_source_views():
    bus = CatalogEventBus()
    cache = CatalogSnapshotCache(events=bus)
    for source in ("routers.text", "peers.remote"):
        cache.put(source, CacheView.CAPABILITIES, _snapshot(source.replace(".", "-")))
        cache.put(source, CacheView.HEALTH, _health(source))

    bus.publish(credential_state_event("routers.text"))
    assert cache.peek("routers.text", CacheView.CAPABILITIES) is not None
    assert cache.peek("routers.text", CacheView.HEALTH) is None
    assert cache.peek("peers.remote", CacheView.HEALTH) is not None

    cache.put("routers.text", CacheView.HEALTH, _health())
    bus.publish(registration_event("routers.text"))
    assert cache.peek("routers.text", CacheView.CAPABILITIES) is None
    assert cache.peek("routers.text", CacheView.HEALTH) is not None

    bus.publish(deployment_lifecycle_event("peers.remote"))
    assert cache.peek("peers.remote", CacheView.CAPABILITIES) is not None
    assert cache.peek("peers.remote", CacheView.HEALTH) is None

    cache.put("peers.remote", CacheView.HEALTH, _health("peers.remote"))
    bus.publish(peer_revision_event("peers.remote", _snapshot().revision))
    assert cache.peek("peers.remote", CacheView.CAPABILITIES) is None
    assert cache.peek("peers.remote", CacheView.HEALTH) is None


def test_explicit_refresh_can_target_one_view_or_all_sources():
    cache = CatalogSnapshotCache()
    for source in ("one", "two"):
        cache.put(source, CacheView.CAPABILITIES, _snapshot(source))
        cache.put(source, CacheView.HEALTH, _health(source))

    cache.invalidate(
        explicit_refresh_event(views="capabilities")
    )

    assert not cache.peek("one", CacheView.CAPABILITIES)
    assert not cache.peek("two", CacheView.CAPABILITIES)
    assert cache.peek("one", CacheView.HEALTH)
    assert cache.peek("two", CacheView.HEALTH)


def test_concurrent_refresh_is_single_flight_per_source():
    cache = CatalogSnapshotCache()
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    call_lock = threading.Lock()

    def loader():
        nonlocal calls
        with call_lock:
            calls += 1
        entered.set()
        assert release.wait(5)
        return _snapshot()

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(
                cache.get_or_refresh,
                "routers.text",
                CacheView.CAPABILITIES,
                loader,
            )
            for _ in range(12)
        ]
        assert entered.wait(5)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert calls == 1
    assert len({item.cid for item in results}) == 1
    assert len({id(item.value) for item in results}) == 1


def test_cancelled_async_waiter_does_not_cancel_or_poison_shared_refresh():
    async def exercise():
        cache = CatalogSnapshotCache()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def loader():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return _snapshot()

        cancelled_waiter = asyncio.create_task(
            cache.get_or_refresh_async(
                "routers.text", CacheView.CAPABILITIES, loader
            )
        )
        await entered.wait()
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        successful_waiter = asyncio.create_task(
            cache.get_or_refresh_async(
                "routers.text", CacheView.CAPABILITIES, loader
            )
        )
        release.set()
        result = await successful_waiter
        assert result.cid == _snapshot().revision
        assert calls == 1

    asyncio.run(exercise())


def test_failed_or_cancelled_refresh_clears_sync_flight_state():
    cache = CatalogSnapshotCache()
    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return _snapshot()

    with pytest.raises(RuntimeError, match="transient"):
        cache.get_or_refresh("routers.text", CacheView.CAPABILITIES, loader)
    result = cache.get_or_refresh(
        "routers.text", CacheView.CAPABILITIES, loader
    )

    assert calls == 2
    assert result.value == _snapshot()


def test_selection_receipt_is_complete_deterministic_and_endpoint_free():
    first = _snapshot("alpha", priority=4)
    second = _snapshot("beta", priority=1)
    snapshot = CatalogSnapshot(
        providers=first.providers + second.providers,
        models=first.models + second.models,
        deployments=first.deployments + second.deployments,
        bindings=first.bindings + second.bindings,
    )
    result = CatalogResolver().resolve(
        snapshot,
        ResolutionRequest(
            operation=Operation.TEXT_CHAT,
            modality=Modality.TEXT,
            policy={"tier": "standard"},
            health=True,
            limit=10,
        ),
    )
    receipt = create_selection_receipt(
        result,
        started_at="2030-01-01T00:00:00Z",
        decided_at="2030-01-01T00:00:00.250000Z",
    )
    payload = receipt.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert len(receipt.candidates) == 2
    assert receipt.selected_binding == result.candidates[0].binding_id
    assert receipt.catalog_revision == snapshot.revision
    assert receipt.policy_filters
    assert all(item.ranking_inputs for item in receipt.candidates)
    assert [item.boundary for item in receipt.fallback_boundaries] == [
        "primary",
        "provider",
    ]
    assert receipt.source_provenance[0].source == "routers.text"
    assert receipt.started_at == "2030-01-01T00:00:00.000000Z"
    assert receipt.decided_at == "2030-01-01T00:00:00.250000Z"
    assert "example.test" not in encoded
    assert "endpoint" not in encoded.casefold()
    assert "internal-record-not-needed" not in encoded
    assert SelectionReceipt.from_dict(payload) == receipt


def test_receipt_contract_rejects_credentials_and_raw_endpoints():
    with pytest.raises(ReceiptValidationError):
        SourceProvenance(source="https://catalog.example.test")
    with pytest.raises(ReceiptValidationError):
        SourceProvenance(source="sk-abcdefghijklmnopqrstuvwxyz")


def test_metrics_cover_required_signals_with_bounded_labels():
    clock = FakeClock()
    metrics = CatalogMetrics(max_sources=2, clock=clock)
    metrics.record_source_latency("source.one", 0.25)
    metrics.record_cache_hit("source.one", CacheView.CAPABILITIES)
    metrics.record_cache_miss("source.one", CacheView.HEALTH, "expired")
    metrics.set_stale_records("source.one", CacheView.HEALTH, 3)
    metrics.record_conflict("precedence")
    metrics.record_resolution(outcome="no_match", no_match_reason="policy")
    metrics.record_health_transition(None, True)
    metrics.record_health_transition(True, False)
    metrics.record_cache_hit("source.two", CacheView.HEALTH)
    metrics.record_cache_hit("unbounded.third", CacheView.HEALTH)

    assert metrics.value(
        "catalog_source_latency_seconds_sum", source="source.one"
    ) == 0.25
    assert metrics.value(
        "catalog_cache_hits_total",
        source="source.one",
        view="capabilities",
    ) == 1
    assert metrics.value(
        "catalog_no_match_total", reason="policy"
    ) == 1
    assert metrics.value(
        "catalog_health_transitions_total",
        from_state="healthy",
        to_state="unhealthy",
    ) == 1
    samples = metrics.snapshot()
    assert any(
        dict(item.labels).get("source") == "other" for item in samples
    )
    assert all(
        set(dict(item.labels)) <= {
            "source",
            "view",
            "reason",
            "kind",
            "outcome",
            "from_state",
            "to_state",
        }
        for item in samples
    )


def test_events_are_bounded_and_round_trip():
    event = CatalogInvalidationEvent(
        kind="peer_revision",
        source="peers.example",
        record_ids=("model_b", "model_a"),
        revision=_snapshot().revision,
    )
    assert event.record_ids == ("model_a", "model_b")
    assert CatalogInvalidationEvent.from_dict(event.to_dict()) == event
    with pytest.raises(ValueError, match="record_ids"):
        CatalogInvalidationEvent(
            kind="registration",
            source="source",
            record_ids=tuple("r%d" % index for index in range(1_001)),
        )
