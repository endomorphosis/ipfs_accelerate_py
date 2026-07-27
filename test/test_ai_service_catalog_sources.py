from __future__ import annotations

import threading
import time

import pytest

from ipfs_accelerate_py.model_catalog.catalog import (
    AIServiceCatalog,
    CatalogSourceError,
    RefreshPolicy,
    RefreshPolicyError,
)
from ipfs_accelerate_py.model_catalog.resolver import ResolutionRequest
from ipfs_accelerate_py.model_catalog.schema import (
    CapabilityDescriptor,
    CatalogSnapshot,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.sources.routers import RouterCatalogSource
from ipfs_accelerate_py.model_catalog.sources.persistent import (
    PersistentCatalogSource,
)
from ipfs_accelerate_py.model_catalog.sources.static import (
    CatalogSourceResult,
    SourceDiagnostic,
    SourceMetadata,
    StaticCatalogSource,
)


def _capability(
    operation: Operation = Operation.TEXT_GENERATE,
    modality: Modality = Modality.TEXT,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        operations=(operation,),
        input_modalities=(modality,),
        output_modalities=(modality,),
    )


def _records(
    provider_name: str,
    model_name: str,
    *,
    source: str,
    description: str = "",
    configured: bool | None = None,
    operation: Operation = Operation.TEXT_GENERATE,
    modality: Modality = Modality.TEXT,
    binding: bool = True,
) -> CatalogSnapshot:
    provenance = (Provenance(source=source),)
    capability = _capability(operation, modality)
    provider = ProviderDescriptor(
        name=provider_name,
        description=description,
        capabilities=(capability,),
        state=OperationalState(configured=configured),
        provenance=provenance,
        labels={"locality": "local"},
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name=model_name,
        capabilities=(capability,),
        state=OperationalState(configured=configured),
        provenance=provenance,
        labels={"locality": "local"},
    )
    bindings = (
        RouterBinding(
            router="test_router",
            provider_id=provider.provider_id,
            model_id=model.model_id,
            operations=(operation,),
            state=OperationalState(configured=configured),
            provenance=provenance,
        ),
    ) if binding else ()
    return CatalogSnapshot(
        providers=(provider,),
        models=(model,),
        bindings=bindings,
    )


class MemorySource:
    side_effecting = False

    def __init__(
        self,
        source: str,
        precedence: int,
        snapshot: CatalogSnapshot,
        *,
        side_effecting: bool = False,
    ) -> None:
        self.source = source
        self.precedence = precedence
        self.current = snapshot
        self.side_effecting = side_effecting
        self.load_calls = 0
        self.refresh_calls = 0
        self.fail = False
        self.diagnostics = ()

    def _result(self) -> CatalogSourceResult:
        return CatalogSourceResult(
            snapshot=self.current,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=self.current.revision,
            ),
            diagnostics=self.diagnostics,
        )

    def load(self) -> CatalogSourceResult:
        self.load_calls += 1
        return self._result()

    def refresh(self) -> CatalogSourceResult:
        self.refresh_calls += 1
        if self.fail:
            raise RuntimeError("credential=must-not-appear")
        return self._result()


class FakeRouter:
    __name__ = "fake_router"

    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self.current = snapshot
        self.snapshot_calls = 0
        self.list_calls = 0
        self.resolve_calls = 0

    def get_catalog_snapshot(self) -> CatalogSnapshot:
        self.snapshot_calls += 1
        return self.current

    def list_providers(self):
        self.list_calls += 1
        raise AssertionError("snapshot-capable router must not use list fallback")

    def list_models(self):
        self.list_calls += 1
        raise AssertionError("snapshot-capable router must not use list fallback")

    def resolve_model(self):
        self.resolve_calls += 1
        raise AssertionError("catalog discovery must not call router resolution")


def test_router_adapter_reads_only_discovery_snapshot_and_enforces_bound():
    snapshot = _records("router-provider", "router-model", source="router.static")
    router = FakeRouter(snapshot)

    result = RouterCatalogSource(
        router, source="routers.fake", max_records=3
    ).load()

    assert result.snapshot == snapshot
    assert result.precedence == 30
    assert router.snapshot_calls == 1
    assert router.list_calls == 0
    assert router.resolve_calls == 0
    with pytest.raises(ValueError, match="maximum record count"):
        RouterCatalogSource(
            router, source="routers.too-small", max_records=2
        ).load()


def test_sources_merge_by_identity_with_explicit_precedence_and_provenance():
    static = MemorySource(
        "capabilities.static",
        10,
        _records(
            "shared",
            "model",
            source="capabilities.static",
            description="static description",
            operation=Operation.TEXT_CHAT,
            binding=False,
        ),
    )
    persistent = MemorySource(
        "metadata.persistent",
        20,
        _records(
            "shared",
            "model",
            source="metadata.persistent",
            configured=True,
            operation=Operation.TEXT_GENERATE,
            binding=False,
        ),
    )
    router = MemorySource(
        "routers.text",
        30,
        _records(
            "shared",
            "model",
            source="routers.text",
            description="router description",
            operation=Operation.TEXT_GENERATE,
        ),
    )

    catalog = AIServiceCatalog(
        {
            static.source: static,
            persistent.source: persistent,
            router.source: router,
        }
    )
    provider = catalog.list_providers().items[0]

    assert provider.description == "router description"
    assert provider.state.configured is True
    assert {item.source for item in provider.provenance} == {
        "capabilities.static",
        "metadata.persistent",
        "routers.text",
    }
    assert {
        operation
        for capability in provider.capabilities
        for operation in capability.operations
    } == {Operation.TEXT_CHAT, Operation.TEXT_GENERATE}
    claims = catalog.claims(provider.provider_id, record_type="providers")
    assert [(item.source, item.precedence) for item in claims] == [
        ("routers.text", 30),
        ("metadata.persistent", 20),
        ("capabilities.static", 10),
    ]
    conflicts = [
        item
        for item in catalog.diagnostics()
        if item.code == "precedence_conflict" and item.field == "description"
    ]
    assert conflicts
    assert conflicts[0].winner_source == "routers.text"


def test_catalog_accepts_persistent_and_static_source_adapters():
    static = StaticCatalogSource(
        [
            {
                "provider": "adapter",
                "model": "shared",
                "description": "static capability record",
                "operations": ["text.chat"],
            }
        ],
        source="adapter.static",
        observed_at="2026-01-01T00:00:00Z",
    )
    persistent = PersistentCatalogSource(
        [
            {
                "provider": "adapter",
                "model": "shared",
                "description": "persistent metadata record",
                "operations": ["text.generate"],
            }
        ],
        source="adapter.persistent",
        observed_at="2026-01-02T00:00:00Z",
    )

    catalog = AIServiceCatalog([static, persistent])
    model = catalog.list_models().items[0]

    assert model.description == "persistent metadata record"
    assert {item.source for item in model.provenance} == {
        "adapter.static",
        "adapter.persistent",
    }
    assert {
        operation
        for capability in model.capabilities
        for operation in capability.operations
    } == {Operation.TEXT_CHAT, Operation.TEXT_GENERATE}


def test_listing_get_resolution_and_health_do_not_reload_sources():
    source = MemorySource(
        "routers.safe",
        30,
        _records("safe", "model", source="routers.safe", configured=True),
    )
    catalog = AIServiceCatalog({source.source: source})
    initial_calls = (source.load_calls, source.refresh_calls)
    request = ResolutionRequest(
        operation=Operation.TEXT_GENERATE,
        provider="safe",
    )

    assert catalog.list_models().total == 1
    assert catalog.get("safe", record_type="providers") is not None
    assert catalog.resolve(request).found
    assert catalog.snapshot() is catalog.snapshot()
    assert catalog.health().record_counts == (
        ("providers", 1),
        ("models", 1),
        ("deployments", 0),
        ("bindings", 1),
    )
    assert (source.load_calls, source.refresh_calls) == initial_calls


def test_registration_precedence_override_is_authoritative():
    source = MemorySource(
        "metadata.override",
        10,
        _records("override", "model", source="metadata.override", binding=False),
    )
    catalog = AIServiceCatalog()

    state = catalog.register_source(
        source.source,
        source,
        precedence=77,
    )

    assert state.precedence == 77
    assert {
        claim.precedence for claim in catalog.claims(source=source.source)
    } == {77}


def test_explicit_refresh_is_named_and_policy_gates_before_any_source_runs():
    safe = MemorySource(
        "metadata.safe",
        20,
        _records("safe", "v1", source="metadata.safe", binding=False),
    )
    active = MemorySource(
        "deployments.live",
        40,
        _records("live", "v1", source="deployments.live"),
        side_effecting=True,
    )
    catalog = AIServiceCatalog({safe.source: safe, active.source: active})

    with pytest.raises(CatalogSourceError, match="one or more"):
        catalog.refresh(())
    with pytest.raises(RefreshPolicyError, match="deployments.live"):
        catalog.refresh((safe.source, active.source))
    assert safe.refresh_calls == 0
    assert active.refresh_calls == 0

    result = catalog.refresh(
        (active.source,),
        policy=RefreshPolicy(
            allow_side_effects=True,
            allowed_sources=(active.source,),
        ),
    )
    assert result.failed == ()
    assert result.refreshed == (active.source,)
    assert active.refresh_calls == 1
    assert safe.refresh_calls == 0


def test_partial_failure_retains_old_claims_and_publishes_healthy_source():
    failing = MemorySource(
        "source.failing",
        20,
        _records("retained", "v1", source="source.failing", binding=False),
    )
    healthy = MemorySource(
        "source.healthy",
        20,
        _records("updated", "v1", source="source.healthy", binding=False),
    )
    catalog = AIServiceCatalog({failing.source: failing, healthy.source: healthy})
    old_snapshot = catalog.snapshot()

    failing.fail = True
    failing.current = _records(
        "erased-if-published", "v2", source="source.failing", binding=False
    )
    healthy.current = _records(
        "updated", "v2", source="source.healthy", binding=False
    )
    result = catalog.refresh((failing.source, healthy.source))

    assert result.failed == (failing.source,)
    assert result.refreshed == (healthy.source,)
    assert catalog.get("retained", record_type="providers") is not None
    assert catalog.get("erased-if-published", record_type="providers") is None
    assert catalog.get("v2", record_type="models") is not None
    assert "must-not-appear" not in repr(result.to_dict())
    assert old_snapshot != catalog.snapshot()
    assert old_snapshot.models[1 if len(old_snapshot.models) > 1 else 0].name == "v1"


def test_source_diagnostics_are_bounded_and_healthy_rows_remain_queryable():
    source = MemorySource(
        "source.partial",
        10,
        _records("partial", "valid", source="source.partial", binding=False),
    )
    source.diagnostics = (
        SourceDiagnostic(
            index=1,
            code="malformed_row",
            message="one row was rejected",
            source_record_id="bad-row",
        ),
    )
    catalog = AIServiceCatalog({source.source: source})

    assert catalog.list_models().total == 1
    diagnostic = next(
        item for item in catalog.diagnostics() if item.code == "malformed_row"
    )
    assert diagnostic.source == source.source
    assert diagnostic.record_id == "bad-row"


def test_source_and_aggregate_output_bounds_fail_without_erasing_old_generation():
    first = MemorySource(
        "bound.first",
        10,
        _records("first", "model", source="bound.first", binding=False),
    )
    second = MemorySource(
        "bound.second",
        10,
        _records("second", "model", source="bound.second", binding=False),
    )
    catalog = AIServiceCatalog(
        {first.source: first},
        max_source_records=2,
        max_output_records=3,
    )
    old_revision = catalog.revision

    state = catalog.register_source(second.source, second)

    assert not state.healthy
    assert catalog.revision == old_revision
    assert catalog.get("first", record_type="providers") is not None
    assert catalog.get("second", record_type="providers") is None
    assert any(
        item.code == "source_refresh_failed" and item.source == second.source
        for item in catalog.diagnostics()
    )

    oversized = MemorySource(
        "bound.oversized",
        10,
        _records("large", "model", source="bound.oversized"),
    )
    source_bound_catalog = AIServiceCatalog(
        max_source_records=2,
        max_output_records=10,
    )
    state = source_bound_catalog.register_source(oversized.source, oversized)
    assert not state.loaded
    assert not state.healthy
    assert len(source_bound_catalog) == 0


def test_deterministic_filter_order_and_snapshot_bound_pagination():
    sources = {
        name: MemorySource(
            "source.%s" % name,
            10,
            _records(
                name,
                "model-%s" % name,
                source="source.%s" % name,
                binding=False,
            ),
        )
        for name in ("charlie", "alpha", "bravo")
    }
    catalog = AIServiceCatalog(
        {source.source: source for source in sources.values()}
    )
    isolated = catalog.snapshot()

    first = catalog.list_providers(limit=2, snapshot=isolated)
    second = catalog.list_providers(
        limit=2, cursor=first.next_cursor, snapshot=isolated
    )

    # Stable IDs, rather than insertion order, are the canonical order.
    assert first.items + second.items == isolated.providers
    assert catalog.list_models(provider="alpha").total == 1
    isolated_model_names = tuple(item.name for item in isolated.models)

    sources["alpha"].current = _records(
        "alpha", "replacement", source="source.alpha", binding=False
    )
    catalog.refresh(("source.alpha",))
    assert tuple(item.name for item in isolated.models) == isolated_model_names
    assert catalog.list_models(snapshot=isolated).snapshot_revision == isolated.revision
    assert catalog.list_models(provider="alpha").items[0].name == "replacement"


class BlockingSource(MemorySource):
    def __init__(self, *args, entered: threading.Event, release: threading.Event, **kwargs):
        super().__init__(*args, **kwargs)
        self.entered = entered
        self.release = release

    def refresh(self) -> CatalogSourceResult:
        self.refresh_calls += 1
        self.entered.set()
        assert self.release.wait(timeout=5)
        return self._result()


def test_concurrent_readers_observe_only_complete_refresh_generations():
    entered = threading.Event()
    release = threading.Event()
    first = MemorySource(
        "atomic.first",
        10,
        _records("first-old", "model", source="atomic.first", binding=False),
    )
    second = BlockingSource(
        "atomic.second",
        10,
        _records("second-old", "model", source="atomic.second", binding=False),
        entered=entered,
        release=release,
    )
    catalog = AIServiceCatalog({first.source: first, second.source: second})
    first.current = _records(
        "first-new", "model", source="atomic.first", binding=False
    )
    second.current = _records(
        "second-new", "model", source="atomic.second", binding=False
    )
    observed = set()
    stopped = threading.Event()

    def read_catalog() -> None:
        while not stopped.is_set():
            observed.add(tuple(item.name for item in catalog.snapshot().providers))
            time.sleep(0.0005)

    reader = threading.Thread(target=read_catalog)
    worker = threading.Thread(
        target=lambda: catalog.refresh((first.source, second.source))
    )
    reader.start()
    worker.start()
    assert entered.wait(timeout=5)
    time.sleep(0.03)
    release.set()
    worker.join(timeout=5)
    time.sleep(0.03)
    stopped.set()
    reader.join(timeout=5)

    old = tuple(
        item.name
        for item in CatalogSnapshot(
            providers=(
                _records("first-old", "model", source="atomic.first", binding=False).providers[0],
                _records("second-old", "model", source="atomic.second", binding=False).providers[0],
            )
        ).providers
    )
    new = tuple(item.name for item in catalog.snapshot().providers)
    assert observed <= {old, new}
    assert new in observed
