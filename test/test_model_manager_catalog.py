"""Contract tests for the canonical catalog facade on ModelManager."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.model_catalog import (
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
from ipfs_accelerate_py.model_catalog.catalog import (
    AIServiceCatalog,
    RefreshPolicy,
    RefreshPolicyError,
)
from ipfs_accelerate_py.model_catalog.sources.static import (
    CatalogSourceResult,
    SourceMetadata,
    StaticCatalogSource,
)
from ipfs_accelerate_py.model_manager import (
    CatalogLookupResult,
    DataType,
    IOSpec,
    ModelManager,
    ModelMetadata,
    ModelType,
)


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    """Create managers without optional storage, graph, or audit integrations."""

    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    created = []

    def factory(*, catalog=None, project_legacy_models=None, name=None):
        path = tmp_path / (name or "manager-%d.json" % len(created))
        manager = ModelManager(
            storage_path=str(path),
            use_database=False,
            enable_ipfs=False,
            catalog=catalog,
            project_legacy_models=project_legacy_models,
        )
        created.append(manager)
        return manager

    yield factory

    for manager in created:
        manager.close()


def _metadata(model_id: str) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_name=model_id.rsplit("/", 1)[-1],
        model_type=ModelType.LANGUAGE_MODEL,
        architecture="CatalogForCausalLM",
        inputs=[IOSpec("prompt", DataType.TEXT)],
        outputs=[IOSpec("text", DataType.TEXT)],
        tags=["catalog"],
    )


def _snapshot(
    provider_name: str,
    model_name: str,
    *,
    aliases=(),
    source: str = "router.fixture",
) -> CatalogSnapshot:
    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_CHAT,),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
    )
    provenance = (Provenance(source=source),)
    provider = ProviderDescriptor(
        name=provider_name,
        aliases=aliases,
        capabilities=(capability,),
        state=OperationalState(configured=True),
        provenance=provenance,
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name=model_name,
        capabilities=(capability,),
        provenance=provenance,
    )
    binding = RouterBinding(
        router="fixture_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        operations=(Operation.TEXT_CHAT,),
        state=OperationalState(routable=True),
        provenance=provenance,
    )
    return CatalogSnapshot(
        providers=(provider,),
        models=(model,),
        bindings=(binding,),
    )


class MemorySource:
    def __init__(
        self,
        source: str,
        snapshot: CatalogSnapshot,
        *,
        side_effecting: bool = False,
    ) -> None:
        self.source = source
        self.precedence = 30
        self.side_effecting = side_effecting
        self.current = snapshot
        self.load_calls = 0
        self.refresh_calls = 0
        self.invoke_calls = 0
        self.fail_refresh = False

    def _result(self) -> CatalogSourceResult:
        return CatalogSourceResult(
            snapshot=self.current,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=self.current.revision,
            ),
        )

    def load(self) -> CatalogSourceResult:
        self.load_calls += 1
        return self._result()

    def refresh(self) -> CatalogSourceResult:
        self.refresh_calls += 1
        if self.fail_refresh:
            raise RuntimeError("Bearer secret-must-not-leak")
        return self._result()

    def invoke(self, *args, **kwargs):
        self.invoke_calls += 1
        raise AssertionError("ModelManager must not own provider invocation")


def test_empty_manager_is_a_stable_versioned_catalog(manager_factory):
    manager = manager_factory()

    snapshot = manager.snapshot()
    assert snapshot.schema_version == "1.0"
    assert snapshot.revision == manager.catalog_revision
    assert manager.list_services().total == 0
    assert manager.list_models(canonical=True).total == 0
    assert manager.list_models() == []
    assert manager.snapshot() is manager.catalog.snapshot()


def test_legacy_registry_projects_provenance_and_crud_stays_compatible(
    manager_factory,
):
    manager = manager_factory()
    original_revision = manager.catalog_revision

    assert manager.add_model(_metadata("acme/one")) is True
    legacy = manager.list_models()
    canonical = manager.list_models(canonical=True)

    assert isinstance(legacy, list)
    assert isinstance(legacy[0], ModelMetadata)
    assert canonical.total == 1
    assert canonical.snapshot_revision != original_revision
    assert canonical.items[0].name == "one"
    assert canonical.items[0].provenance[0].source == "model-manager.persistent"
    assert manager.search_models("one") == legacy
    assert manager.get_model("acme/one") is legacy[0]

    assert manager.remove_model("acme/one") is True
    assert manager.list_models() == []
    assert manager.list_models(canonical=True).total == 0


def test_canonical_pagination_is_bounded_stable_and_snapshot_scoped(
    manager_factory,
):
    source = StaticCatalogSource(
        [
            {"provider": name, "model": "chat", "operations": ["text.chat"]}
            for name in ("charlie", "alpha", "bravo")
        ],
        source="static.fixture",
        precedence=10,
    )
    catalog = AIServiceCatalog({source.source: source})
    manager = manager_factory(catalog=catalog)
    isolated = manager.snapshot()

    first = manager.list_services(limit=2, snapshot=isolated)
    second = manager.list_services(
        limit=2,
        cursor=first.next_cursor,
        snapshot=isolated,
    )

    assert len(first.items) <= 2
    assert first.items + second.items == isolated.providers
    assert first.snapshot_revision == second.snapshot_revision == isolated.revision
    assert manager.list_models(limit=1).record_type == "models"
    assert manager.list_catalog_models(limit=1).total == 3


def test_lookup_returns_typed_no_match_and_ambiguity_diagnostics(
    manager_factory,
):
    left = _snapshot("left", "chat", aliases=("shared",), source="source.left")
    right = _snapshot("right", "chat", aliases=("shared",), source="source.right")
    left_source = MemorySource("source.left", left)
    right_source = MemorySource("source.right", right)
    manager = manager_factory(
        catalog=AIServiceCatalog(
            {left_source.source: left_source, right_source.source: right_source}
        )
    )

    found = manager.get_service("left")
    missing = manager.get_model_descriptor("absent")
    ambiguous = manager.get_service("shared")

    assert isinstance(found, CatalogLookupResult)
    assert found.found and found.record.name == "left"
    assert found.snapshot_revision == manager.catalog_revision
    assert not missing.found
    assert missing.diagnostics[0].code == "no_match"
    assert ambiguous.ambiguous
    assert ambiguous.diagnostics[0].code == "ambiguous_identifier"
    assert ambiguous.record is None


def test_read_facade_has_no_refresh_probe_or_invocation_side_effects(
    manager_factory,
):
    source = MemorySource("router.readonly", _snapshot("readonly", "chat"))
    manager = manager_factory(
        catalog=AIServiceCatalog({source.source: source})
    )
    assert source.load_calls == 1

    isolated = manager.snapshot()
    manager.list_services(snapshot=isolated)
    manager.list_models(canonical=True, snapshot=isolated)
    manager.get_service("readonly", snapshot=isolated)
    manager.get_model_descriptor("chat", snapshot=isolated)
    result = manager.resolve(operation="text.chat", snapshot=isolated)
    manager.health(snapshot=isolated)

    assert result.found
    assert source.load_calls == 1
    assert source.refresh_calls == 0
    assert source.invoke_calls == 0
    assert not hasattr(manager, "invoke")


def test_refresh_is_explicit_named_and_policy_gated(manager_factory):
    source = MemorySource(
        "deployments.active",
        _snapshot("active", "chat", source="deployments.active"),
        side_effecting=True,
    )
    manager = manager_factory(
        catalog=AIServiceCatalog({source.source: source})
    )

    with pytest.raises(RefreshPolicyError, match="deployments.active"):
        manager.refresh((source.source,))
    assert source.refresh_calls == 0

    result = manager.refresh_catalog(
        (source.source,),
        policy=RefreshPolicy(
            allow_side_effects=True,
            allowed_sources=(source.source,),
        ),
    )
    assert result.failed == ()
    assert source.refresh_calls == 1


def test_source_failure_retains_last_good_records_and_is_typed_in_health(
    manager_factory,
):
    source = MemorySource(
        "router.failing",
        _snapshot("retained", "v1", source="router.failing"),
    )
    manager = manager_factory(
        catalog=AIServiceCatalog({source.source: source})
    )
    old_revision = manager.catalog_revision
    source.current = _snapshot("replacement", "v2", source="router.failing")
    source.fail_refresh = True

    result = manager.refresh((source.source,))

    assert result.failed == (source.source,)
    assert manager.catalog_revision == old_revision
    assert manager.get_service("retained").found
    assert not manager.get_service("replacement").found
    assert manager.health().partial
    assert "secret-must-not-leak" not in repr(result.to_dict())


def test_concurrent_facade_reads_observe_one_immutable_revision(manager_factory):
    source = MemorySource(
        "router.concurrent",
        _snapshot("concurrent", "chat", source="router.concurrent"),
    )
    manager = manager_factory(
        catalog=AIServiceCatalog({source.source: source})
    )
    expected_revision = manager.catalog_revision
    barrier = threading.Barrier(8)

    def read_many():
        barrier.wait(timeout=5)
        return {
            (
                manager.snapshot().revision,
                manager.list_services().snapshot_revision,
                manager.list_models(canonical=True).snapshot_revision,
            )
            for _ in range(100)
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: read_many(), range(8)))

    assert all(
        result == {(expected_revision, expected_revision, expected_revision)}
        for result in results
    )
    assert source.refresh_calls == 0


def test_old_enveloped_registry_migration_is_idempotent_and_non_writing(
    monkeypatch, tmp_path
):
    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    path = tmp_path / "old-registry.json"
    document = {
        "schema_version": "model-manager.registry.v0",
        "source_revision": "legacy-7",
        "models": {
            "legacy/example": {
                "model_name": "Example",
                "model_type": "language_model",
                "architecture": "LegacyArchitecture",
                "inputs": [{"name": "tokens", "data_type": "tokens"}],
                "outputs": [{"name": "logits", "data_type": "logits"}],
                "unknown_old_field": "ignored",
            }
        },
    }
    encoded = json.dumps(document, sort_keys=True)
    path.write_text(encoded, encoding="utf-8")

    first = ModelManager(storage_path=str(path), use_database=False)
    second = ModelManager(storage_path=str(path), use_database=False)

    assert first.get_model("legacy/example").architecture == "LegacyArchitecture"
    assert second.catalog_revision == first.catalog_revision
    assert path.read_text(encoding="utf-8") == encoded
    # Avoid turning this read-only migration test into an explicit save.
    first._save_data = lambda: None
    second._save_data = lambda: None
    first.close()
    second.close()


def test_atomic_persistence_failure_keeps_previous_registry(monkeypatch, tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("previous", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(model_manager_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement"):
        ModelManager._atomic_write_text(str(path), "new")

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(".registry.json.*.tmp")) == []
