from __future__ import annotations

import json
from typing import Iterable, Optional

import pytest

from ipfs_accelerate_py import embeddings_router
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    ProviderDescriptor,
)


_DISCOVERY_ENV = (
    "IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER",
    "IPFS_DATASETS_PY_EMBEDDINGS_PROVIDER",
    "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
    "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "IPFS_ACCELERATE_PY_HF_API_TOKEN",
    "IPFS_DATASETS_PY_HF_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
    "XAI_API_KEY",
    "ipfs_accelerate_py_XAI_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER",
    "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
    "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
    "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
    "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings_router, "_PROVIDER_REGISTRY", {})
    for name in _DISCOVERY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    embeddings_router.clear_embeddings_router_caches()


def _capability(record: ProviderDescriptor | ModelDescriptor) -> CapabilityDescriptor:
    assert len(record.capabilities) == 1
    return record.capabilities[0]


def test_builtin_descriptors_publish_embedding_specific_metadata() -> None:
    providers = embeddings_router.list_providers()

    assert [provider.name for provider in providers] == sorted(
        {
            "accelerate",
            "adapter",
            "backend_manager",
            "gemini_cli",
            "hf_inference_api",
            "huggingface",
            "meta_ai",
            "openrouter",
            "xai",
        }
    )
    assert all(isinstance(provider, ProviderDescriptor) for provider in providers)

    openrouter = embeddings_router.get_provider_descriptor("openrouter")
    capability = _capability(openrouter)
    assert capability.operations == (Operation.BATCH, Operation.EMBEDDING_GENERATE)
    assert capability.input_modalities == (Modality.TEXT,)
    assert capability.output_modalities == (Modality.EMBEDDING,)
    assert capability.embedding_dimensions == 1536
    assert capability.max_context_tokens == 8191
    assert capability.max_batch_size is None
    assert openrouter.state.authorized is False
    assert openrouter.state.reachable is None
    assert dict(openrouter.labels) == {
        "access_requirement": "required",
        "batching": "supported",
        "device": "provider-managed",
        "input_types": "text",
        "locality": "remote",
        "normalization": "model-dependent",
    }

    huggingface = embeddings_router.get_provider_descriptor("hf")
    assert huggingface.name == "huggingface"
    assert huggingface.state.authorized is True
    assert huggingface.state.configured is None
    assert dict(huggingface.labels)["locality"] == "local"
    assert dict(huggingface.labels)["device"] == "cpu,cuda"
    assert embeddings_router.get_provider_descriptor("hf_api").state.authorized is None


def test_model_facts_and_unknown_overrides_are_not_invented() -> None:
    known = embeddings_router.resolve_model(
        "text-embedding-3-small",
        provider="openrouter",
    )
    assert _capability(known).embedding_dimensions == 1536
    assert _capability(known).max_context_tokens == 8191
    assert dict(known.labels)["normalization"] == "unit"

    unknown = embeddings_router.resolve_model(
        "vendor/new-embedding-model",
        provider="openrouter",
    )
    assert unknown.name == "vendor/new-embedding-model"
    assert dict(unknown.labels)["invocation_model"] == "vendor/new-embedding-model"
    assert _capability(unknown).embedding_dimensions is None
    assert _capability(unknown).max_context_tokens is None
    assert dict(unknown.labels)["normalization"] == "unknown"


def test_environment_model_override_changes_discovery_without_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL",
        "example/Unpublished-Embed",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured-for-test")

    provider = embeddings_router.get_provider_descriptor("openrouter")
    model = embeddings_router.list_models("openrouter")[0]

    assert provider.lifecycle is LifecycleState.CONFIGURED
    assert provider.state.configured is True
    assert provider.state.authorized is True
    assert provider.state.reachable is None
    assert _capability(provider).embedding_dimensions is None
    assert model.name == "example/unpublished-embed"
    assert dict(model.labels)["invocation_model"] == "example/Unpublished-Embed"
    assert _capability(model).embedding_dimensions is None
    assert "configured-for-test" not in json.dumps(provider.to_dict())


def test_discovery_does_not_construct_factories_import_runtimes_or_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def factory() -> object:
        events.append("factory")
        raise AssertionError("discovery constructed a provider")

    embeddings_router.register_embeddings_provider("fixture", factory)

    def blocked(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        events.append("side-effect")
        raise AssertionError("discovery attempted a runtime side effect")

    monkeypatch.setattr(embeddings_router.importlib, "import_module", blocked)
    monkeypatch.setattr(embeddings_router.urllib.request, "urlopen", blocked)
    for name in (
        "_get_openrouter_provider",
        "_get_hf_inference_api_provider",
        "_get_xai_embeddings_provider",
        "_get_meta_ai_embeddings_provider",
        "_get_gemini_cli_provider",
        "_get_huggingface_provider",
        "_get_local_adapter_provider",
        "_get_accelerate_provider",
        "_get_backend_manager_provider",
    ):
        monkeypatch.setattr(embeddings_router, name, blocked)

    assert embeddings_router.get_provider_descriptor("fixture").name == "fixture"
    assert "fixture" in {provider.name for provider in embeddings_router.list_providers()}
    assert isinstance(embeddings_router.get_catalog_snapshot(), CatalogSnapshot)
    assert events == []


def test_dynamic_registration_is_deterministic_and_keeps_unknowns_unknown() -> None:
    def factory() -> object:
        raise AssertionError("factory should not be called")

    before = embeddings_router.get_catalog_snapshot()
    embeddings_router.register_embeddings_provider("zeta_fixture", factory)
    embeddings_router.register_embeddings_provider("alpha_fixture", factory)

    names = [provider.name for provider in embeddings_router.list_providers()]
    assert names == sorted(names)
    alpha = embeddings_router.get_provider_descriptor("alpha_fixture")
    assert alpha.state.known is True
    assert alpha.state.configured is True
    assert alpha.state.authorized is None
    assert alpha.state.routable is None
    assert dict(alpha.labels)["device"] == "unknown"
    assert dict(alpha.labels)["normalization"] == "unknown"
    assert embeddings_router.list_models("alpha_fixture") == []

    first = embeddings_router.get_catalog_snapshot()
    second = embeddings_router.get_catalog_snapshot()
    assert first.revision != before.revision
    assert first.revision == second.revision
    assert first.to_dict() == second.to_dict()


def test_registered_metadata_and_resolution_match_generation_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class _FixtureProvider:
        router_provider_name = "catalog_fixture"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> list[list[float]]:
            _ = (model_name, device, kwargs)
            items = list(texts)
            calls.append(items)
            return [[float(index), float(index + 1)] for index, _ in enumerate(items)]

    provider = ProviderDescriptor(
        name="catalog_fixture",
        aliases=("catalog_alias",),
        capabilities=(
            CapabilityDescriptor(
                operations=(Operation.EMBEDDING_GENERATE, Operation.BATCH),
                input_modalities=(Modality.TEXT,),
                output_modalities=(Modality.EMBEDDING,),
                max_batch_size=8,
                embedding_dimensions=2,
            ),
        ),
        lifecycle=LifecycleState.READY,
        labels={
            "batching": "supported",
            "device": "cpu",
            "input_types": "text",
            "locality": "local",
            "normalization": "none",
        },
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="fixture/embed-v1",
        capabilities=provider.capabilities,
        lifecycle=LifecycleState.READY,
        labels={"invocation_model": "fixture/embed-v1"},
    )
    embeddings_router.register_embeddings_provider(
        "catalog_fixture",
        _FixtureProvider,
        descriptor=provider,
        models=(model,),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER", "catalog_alias")

    resolved = embeddings_router.resolve_model("fixture/embed-v1")
    vectors = embeddings_router.embed_texts(
        ["one", "two"],
        model_name="fixture/embed-v1",
    )

    assert resolved.provider_id == provider.provider_id
    assert resolved.model_id == model.model_id
    assert embeddings_router.get_last_embedding_trace()["provider_used"] == ("catalog_fixture")
    assert vectors == [[0.0, 1.0], [1.0, 2.0]]
    assert calls == [["one", "two"]]


def test_catalog_projection_contains_typed_model_bindings() -> None:
    snapshot = embeddings_router.catalog_snapshot()

    assert isinstance(snapshot, CatalogSnapshot)
    assert {provider.provider_id for provider in snapshot.providers} == {
        provider.provider_id for provider in embeddings_router.list_providers()
    }
    assert {model.model_id for model in snapshot.models} == {
        model.model_id for model in embeddings_router.list_models()
    }
    assert len(snapshot.bindings) == len(snapshot.models)
    assert all(binding.router == "embeddings_router" for binding in snapshot.bindings)
    assert all(
        binding.operations == (Operation.BATCH, Operation.EMBEDDING_GENERATE)
        for binding in snapshot.bindings
    )
    assert {binding.model_id for binding in snapshot.bindings} == {
        model.model_id for model in snapshot.models
    }
