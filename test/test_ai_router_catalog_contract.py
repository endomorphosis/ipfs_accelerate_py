"""Common discovery-contract tests for every canonical AI router."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py import (
    embeddings_router,
    llm_router,
    multimodal_router,
    voice_router,
)
from ipfs_accelerate_py.model_catalog import (
    CatalogSnapshot,
    ModelDescriptor,
    ProviderDescriptor,
    RouterBinding,
)


_ROUTERS = (
    (llm_router, "llm_router"),
    (embeddings_router, "embeddings_router"),
    (multimodal_router, "multimodal_router"),
    (voice_router, "voice_router"),
)


@pytest.mark.parametrize(("router", "router_name"), _ROUTERS)
def test_router_exposes_one_typed_side_effect_free_discovery_contract(
    router: Any,
    router_name: str,
) -> None:
    providers = tuple(router.list_providers())
    models = tuple(router.list_models())
    snapshot = router.get_catalog_snapshot()

    assert providers
    assert isinstance(snapshot, CatalogSnapshot)
    assert snapshot == router.catalog_snapshot()
    assert all(isinstance(provider, ProviderDescriptor) for provider in providers)
    assert all(isinstance(model, ModelDescriptor) for model in models)
    assert all(isinstance(binding, RouterBinding) for binding in snapshot.bindings)

    provider_ids = {provider.provider_id for provider in providers}
    model_ids = {model.model_id for model in models}
    assert len(provider_ids) == len(providers)
    assert len(model_ids) == len(models)
    assert provider_ids == {provider.provider_id for provider in snapshot.providers}
    assert model_ids == {model.model_id for model in snapshot.models}

    for provider in providers:
        resolved = router.get_provider_descriptor(provider.name)
        assert resolved.provider_id == provider.provider_id

    assert len(snapshot.bindings) == len(models)
    assert all(binding.router == router_name for binding in snapshot.bindings)
    assert all(binding.provider_id in provider_ids for binding in snapshot.bindings)
    assert {binding.model_id for binding in snapshot.bindings} == model_ids


def test_router_snapshots_share_schema_and_canonical_binding_identity() -> None:
    snapshots = tuple(router.get_catalog_snapshot() for router, _name in _ROUTERS)
    schema_versions = {snapshot.schema_version for snapshot in snapshots}
    binding_ids = [
        binding.binding_id
        for snapshot in snapshots
        for binding in snapshot.bindings
    ]

    assert len(schema_versions) == 1
    assert all(snapshot.revision for snapshot in snapshots)
    assert len(binding_ids) == len(set(binding_ids))
    for snapshot in snapshots:
        assert CatalogSnapshot.from_dict(snapshot.to_dict()) == snapshot
