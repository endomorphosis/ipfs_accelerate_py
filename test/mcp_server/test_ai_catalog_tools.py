"""Contract tests for native MCP access to the canonical AI service catalog."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog
from ipfs_accelerate_py.model_catalog.sources.static import (
    CatalogSourceResult,
    SourceMetadata,
)
from ipfs_accelerate_py.model_manager import ModelManager


def _run(awaitable: Any) -> Dict[str, Any]:
    return asyncio.run(awaitable)


def _records(
    provider_name: str,
    model_name: str = "chat",
    *,
    aliases: Tuple[str, ...] = (),
    source: str,
    endpoint: str = "http://127.0.0.1:8080/v1",
) -> Tuple[Any, ...]:
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
        state=OperationalState(configured=True, healthy=True),
        provenance=provenance,
        labels=(("region", "test"),),
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name=model_name,
        capabilities=(capability,),
        state=OperationalState(configured=True),
        provenance=provenance,
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=model.model_id,
        name="private",
        endpoint_uri=endpoint,
        capabilities=(capability,),
        state=OperationalState(
            configured=True,
            authorized=True,
            reachable=True,
            healthy=True,
        ),
        provenance=provenance,
    )
    binding = RouterBinding(
        router="fixture_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_CHAT,),
        state=OperationalState(routable=True),
        provenance=provenance,
    )
    return provider, model, deployment, binding


def _snapshot(*groups: Iterable[Any]) -> CatalogSnapshot:
    records = tuple(item for group in groups for item in group)
    return CatalogSnapshot(
        providers=tuple(
            item for item in records if isinstance(item, ProviderDescriptor)
        ),
        models=tuple(item for item in records if isinstance(item, ModelDescriptor)),
        deployments=tuple(
            item for item in records if isinstance(item, DeploymentDescriptor)
        ),
        bindings=tuple(item for item in records if isinstance(item, RouterBinding)),
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
            raise RuntimeError("Bearer source-private-secret")
        return self._result()

    def invoke(self, *args: Any, **kwargs: Any) -> None:
        self.invoke_calls += 1
        raise AssertionError("catalog reads must not invoke a provider")


@pytest.fixture
def install_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    created = []

    def install(*sources: MemorySource) -> ModelManager:
        catalog = AIServiceCatalog({source.source: source for source in sources})
        manager = ModelManager(
            storage_path=str(tmp_path / ("manager-%d.json" % len(created))),
            use_database=False,
            enable_ipfs=False,
            catalog=catalog,
            project_legacy_models=False,
        )
        created.append(manager)
        monkeypatch.setattr(
            model_manager_module,
            "get_default_model_manager",
            lambda: manager,
        )
        return manager

    yield install

    for manager in created:
        manager.close()


class ToolRecorder:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[definition["name"]] = definition


def test_cold_registration_adds_catalog_tools_without_resolving_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions = 0

    def forbidden_manager() -> None:
        nonlocal resolutions
        resolutions += 1
        raise AssertionError("registration resolved ModelManager")

    monkeypatch.setattr(
        model_manager_module,
        "get_default_model_manager",
        forbidden_manager,
    )
    registry = ToolRecorder()

    native_model_tools.register_native_model_tools(registry)

    expected = {
        "model_catalog_list_services",
        "model_catalog_list_models",
        "model_catalog_get",
        "model_catalog_resolve",
        "model_catalog_health",
        "model_catalog_refresh",
    }
    legacy = {
        "model_search",
        "model_recommend",
        "model_get_details",
        "model_get_stats",
        "model_list_served",
        "model_get_served",
        "model_list_hf_inference",
        "model_get_hf_metadata",
        "model_build_hf_ipld_document",
        "model_get_hf_ipld_cid",
        "model_publish_hf_ipld_to_ipfs",
        "model_load_hf_ipld_from_ipfs",
    }
    assert expected | legacy <= set(registry.tools)
    assert resolutions == 0
    refresh_schema = registry.tools["model_catalog_refresh"]["input_schema"]
    assert set(refresh_schema["required"]) == {"sources", "authority"}
    assert refresh_schema["properties"]["authority"]["const"] is True
    assert (
        registry.tools["model_catalog_list_models"]["input_schema"]["properties"][
            "limit"
        ]["maximum"]
        == 1000
    )


def test_list_tools_delegate_with_versioned_canonical_identity_parity(
    install_manager,
) -> None:
    source = MemorySource(
        "fixture.catalog",
        _snapshot(_records("acme", source="fixture.catalog")),
    )
    manager = install_manager(source)
    python_services = manager.list_services(labels={"region": "test"})
    python_models = manager.list_catalog_models(provider="acme")

    services = _run(
        native_model_tools.model_catalog_list_services(
            labels={"region": "test"},
            limit=1,
        )
    )
    models = _run(
        native_model_tools.model_catalog_list_models(
            provider="acme",
            limit=1,
        )
    )

    assert services["status"] == models["status"] == "success"
    assert services["schema_version"] == manager.snapshot().schema_version
    assert services["catalog_revision"] == manager.catalog_revision
    assert services["tool_schema_version"] == "ai.catalog.mcp.v1"
    assert (
        services["items"][0]["provider_id"]
        == python_services.items[0].provider_id
    )
    assert services["services"] == services["items"]
    assert models["items"][0]["model_id"] == python_models.items[0].model_id
    assert models["models"] == models["items"]
    assert source.load_calls == 1
    assert source.refresh_calls == source.invoke_calls == 0


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (
            native_model_tools.model_catalog_list_services,
            {"state": {"not_a_state": True}},
        ),
        (
            native_model_tools.model_catalog_list_models,
            {"labels": {"region": 7}},
        ),
        (
            native_model_tools.model_catalog_list_services,
            {"modality": "telepathy"},
        ),
        (
            native_model_tools.model_catalog_list_models,
            {"limit": 1001},
        ),
    ],
)
def test_malformed_filters_return_typed_bounded_errors(
    install_manager,
    tool,
    kwargs,
) -> None:
    source = MemorySource(
        "fixture.filters",
        _snapshot(_records("filters", source="fixture.filters")),
    )
    manager = install_manager(source)

    result = _run(tool(**kwargs))

    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_filter"
    assert result["catalog_revision"] == manager.catalog_revision
    assert json.dumps(result).find("telepathy") == -1
    assert source.refresh_calls == source.invoke_calls == 0


def test_pagination_is_bounded_and_cursor_revision_mismatch_is_typed(
    install_manager,
) -> None:
    initial_groups = tuple(
        _records("provider-%04d" % index, source="fixture.large")
        for index in range(3)
    )
    source = MemorySource("fixture.large", _snapshot(*initial_groups))
    manager = install_manager(source)

    first = _run(
        native_model_tools.model_catalog_list_models(limit=2)
    )

    assert first["status"] == "success"
    assert first["count"] == len(first["items"]) == 2
    assert first["total"] == 3
    assert first["next_cursor"]
    old_cursor = first["next_cursor"]
    source.current = _snapshot(
        _records("replacement", source="fixture.large")
    )
    manager.refresh((source.source,))

    stale = _run(
        native_model_tools.model_catalog_list_models(
            limit=2,
            cursor=old_cursor,
        )
    )

    assert stale["status"] == "error"
    assert stale["error"]["code"] == "cursor_revision_mismatch"
    assert stale["catalog_revision"] == manager.catalog_revision


def test_get_returns_typed_no_match_and_ambiguity_errors(
    install_manager,
) -> None:
    left = MemorySource(
        "fixture.left",
        _snapshot(
            _records(
                "left",
                aliases=("shared",),
                source="fixture.left",
            )
        ),
    )
    right = MemorySource(
        "fixture.right",
        _snapshot(
            _records(
                "right",
                aliases=("shared",),
                source="fixture.right",
            )
        ),
    )
    install_manager(left, right)

    missing = _run(native_model_tools.model_catalog_get("absent"))
    ambiguous = _run(
        native_model_tools.model_catalog_get(
            "shared",
            record_type="providers",
        )
    )

    assert missing["error"]["code"] == "no_match"
    assert missing["diagnostics"][0]["code"] == "no_match"
    assert ambiguous["error"]["code"] == "ambiguous_identifier"
    assert ambiguous["diagnostics"][0]["ambiguous"] is True
    assert ambiguous["record"] is None
    assert left.refresh_calls == right.refresh_calls == 0


def test_get_and_resolve_redact_private_endpoint_but_preserve_ids(
    install_manager,
) -> None:
    records = _records("private", source="fixture.private")
    source = MemorySource("fixture.private", _snapshot(records))
    manager = install_manager(source)
    deployment = records[2]

    fetched = _run(
        native_model_tools.model_catalog_get(
            deployment.deployment_id,
            record_type="deployments",
        )
    )
    resolved = _run(
        native_model_tools.model_catalog_resolve(
            operation="text.chat",
            provider="private",
        )
    )

    assert fetched["status"] == resolved["status"] == "success"
    assert fetched["record"]["deployment_id"] == deployment.deployment_id
    assert fetched["record"]["endpoint_uri"] == "[REDACTED]"
    candidate = resolved["resolution"]["candidates"][0]
    assert candidate["provider"]["provider_id"] == records[0].provider_id
    assert candidate["model"]["model_id"] == records[1].model_id
    assert candidate["deployment"]["deployment_id"] == deployment.deployment_id
    assert candidate["deployment"]["endpoint_uri"] == "[REDACTED]"
    serialized = json.dumps((fetched, resolved))
    assert "127.0.0.1" not in serialized
    assert manager.get(
        deployment.deployment_id,
        record_type="deployments",
    ).record.deployment_id == fetched["record"]["deployment_id"]
    assert source.refresh_calls == source.invoke_calls == 0


def test_resolve_no_match_and_health_are_typed_side_effect_free(
    install_manager,
) -> None:
    source = MemorySource(
        "fixture.readonly",
        _snapshot(_records("readonly", source="fixture.readonly")),
    )
    manager = install_manager(source)

    missing = _run(
        native_model_tools.model_catalog_resolve(
            operation="audio.transcribe",
        )
    )
    health = _run(native_model_tools.model_catalog_health())

    assert missing["error"]["code"] == "no_match"
    assert missing["resolution"]["snapshot_revision"] == manager.catalog_revision
    assert health["status"] == "success"
    assert health["health"]["snapshot_revision"] == manager.catalog_revision
    assert health["catalog_revision"] == manager.catalog_revision
    assert source.load_calls == 1
    assert source.refresh_calls == source.invoke_calls == 0


def test_refresh_requires_explicit_authority_and_named_sources(
    install_manager,
) -> None:
    source = MemorySource(
        "fixture.privileged",
        _snapshot(_records("refresh", source="fixture.privileged")),
        side_effecting=True,
    )
    manager = install_manager(source)

    denied = _run(
        native_model_tools.model_catalog_refresh([source.source])
    )
    unnamed = _run(
        native_model_tools.model_catalog_refresh([], authority=True)
    )

    assert denied["error"]["code"] == "refresh_denied"
    assert unnamed["error"]["code"] == "invalid_sources"
    assert source.refresh_calls == 0

    refreshed = _run(
        native_model_tools.model_catalog_refresh(
            [source.source],
            authority=True,
        )
    )

    assert refreshed["status"] == "success"
    assert refreshed["refreshed"] == [source.source]
    assert refreshed["catalog_revision"] == manager.catalog_revision
    assert source.refresh_calls == 1


def test_refresh_source_failure_is_typed_and_does_not_leak_secrets(
    install_manager,
) -> None:
    source = MemorySource(
        "fixture.failure",
        _snapshot(_records("retained", source="fixture.failure")),
        side_effecting=True,
    )
    manager = install_manager(source)
    original_revision = manager.catalog_revision
    source.fail_refresh = True

    failed = _run(
        native_model_tools.model_catalog_refresh(
            [source.source],
            authority=True,
        )
    )

    assert failed["status"] == "error"
    assert failed["error"]["code"] == "source_refresh_failed"
    assert failed["failed"] == [source.source]
    assert failed["catalog_revision"] == original_revision
    assert manager.catalog_revision == original_revision
    assert "source-private-secret" not in json.dumps(failed)
    assert failed["source_states"][0]["last_error"] == "source raised RuntimeError"
