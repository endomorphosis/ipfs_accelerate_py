from __future__ import annotations

import json
from typing import Optional

import pytest

from ipfs_accelerate_py import multimodal_router
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
)


_DISCOVERY_ENV = (
    "IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER",
    "IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER",
    "IPFS_ACCELERATE_PY_MULTIMODAL_MODEL",
    "IPFS_ACCELERATE_PY_MULTIMODAL_DEVICE",
    "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "IPFS_ACCELERATE_PY_OPENROUTER_MULTIMODAL_MODEL",
    "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "IPFS_ACCELERATE_PY_OPENAI_MULTIMODAL_MODEL",
    "XAI_API_KEY",
    "ipfs_accelerate_py_XAI_API_KEY",
    "ipfs_accelerate_py_XAI_MULTIMODAL_MODEL",
    "ipfs_accelerate_py_MULTIMODAL_MODEL",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL",
)


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multimodal_router, "_PROVIDER_REGISTRY", {})
    for name in _DISCOVERY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    multimodal_router.clear_multimodal_router_caches()


def _capability(
    record: ProviderDescriptor | ModelDescriptor,
) -> CapabilityDescriptor:
    assert len(record.capabilities) == 1
    return record.capabilities[0]


def test_builtin_descriptors_publish_directional_multimodal_metadata() -> None:
    providers = multimodal_router.list_providers()

    assert [provider.name for provider in providers] == sorted(
        {
            "backend_manager",
            "huggingface",
            "meta_ai",
            "openai",
            "openrouter",
            "xai",
        }
    )
    assert all(isinstance(provider, ProviderDescriptor) for provider in providers)
    for provider in providers:
        capability = _capability(provider)
        labels = dict(provider.labels)
        assert capability.operations == (
            Operation.TEXT_GENERATE,
            Operation.VISION_GENERATE,
        )
        assert capability.input_modalities == (
            Modality.IMAGE,
            Modality.TEXT,
        )
        assert capability.output_modalities == (Modality.TEXT,)
        assert capability.media_types == ("image/*", "text/plain")
        assert capability.max_batch_size is None
        assert capability.max_input_bytes is None
        assert capability.max_output_bytes is None
        assert labels["image_input_modes"] == "inline,uri"
        assert labels["inline_input_types"] == "bytes,data-uri,file-path"
        assert labels["uri_schemes"] == "http,https"
        assert labels["input_media_types"] == "image/*,text/plain"
        assert labels["output_media_types"] == "text/plain"
        assert labels["max_images"] == "1"
        assert labels["streaming"] == "unsupported"
        assert labels["batching"] == "unsupported"
        assert provider.state.known is True
        assert provider.state.reachable is None
        assert provider.state.healthy is None

    assert multimodal_router.get_provider_descriptor("gpt4o").name == "openai"
    assert multimodal_router.get_provider_descriptor("grok").name == "xai"
    assert multimodal_router.get_provider_descriptor("spark").name == "meta_ai"
    assert multimodal_router.get_provider_descriptor("hf").name == "huggingface"
    assert (
        multimodal_router.get_provider_descriptor("accelerate").name
        == "backend_manager"
    )


def test_listing_filters_media_transport_counts_and_runtime_constraints() -> None:
    all_names = {
        provider.name for provider in multimodal_router.list_providers()
    }

    assert {
        provider.name
        for provider in multimodal_router.list_providers(
            operation=Operation.VISION_GENERATE,
            input_modality=Modality.IMAGE,
            output_modality=Modality.TEXT,
            media_type="image/png",
            image_input_mode="inline",
            image_count=1,
            streaming=False,
            batching=False,
        )
    } == all_names
    assert multimodal_router.list_providers(image_count=2) == []
    assert multimodal_router.list_providers(streaming=True) == []
    assert multimodal_router.list_providers(batching=True) == []
    assert multimodal_router.list_providers(media_type="audio/wav") == []
    assert [
        provider.name
        for provider in multimodal_router.list_providers(locality="local")
    ] == ["huggingface"]
    assert {
        provider.name
        for provider in multimodal_router.list_providers(authorized=False)
    } == {"meta_ai", "openai", "openrouter", "xai"}


def test_known_model_facts_and_unknown_overrides_are_not_invented() -> None:
    known = multimodal_router.resolve_model(
        "openai/gpt-4o",
        provider="openrouter",
        media_type="image/jpeg",
        image_input_mode="uri",
        image_count=1,
    )
    assert known.name == "openai/gpt-4o"
    assert _capability(known).max_context_tokens == 128_000
    assert dict(known.labels)["invocation_model"] == "openai/gpt-4o"

    unknown = multimodal_router.resolve_model(
        "vendor/Future-Vision",
        provider="openrouter",
    )
    assert unknown.name == "vendor/future-vision"
    assert unknown.architecture is None
    assert _capability(unknown).max_context_tokens is None
    assert _capability(unknown).max_input_bytes is None
    assert dict(unknown.labels)["invocation_model"] == "vendor/Future-Vision"

    with pytest.raises(ValueError, match="incompatible"):
        multimodal_router.resolve_model(
            provider="openrouter",
            image_count=2,
        )
    with pytest.raises(ValueError, match="does not support operation"):
        multimodal_router.resolve_model(
            provider="openrouter",
            operation=Operation.STREAM,
        )


def test_environment_metadata_changes_without_leaking_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured-for-test")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_OPENROUTER_MULTIMODAL_MODEL",
        "vendor/Unpublished-Vision",
    )

    provider = multimodal_router.get_provider_descriptor("openrouter")
    model = multimodal_router.list_models("openrouter")[0]

    assert provider.lifecycle is LifecycleState.CONFIGURED
    assert provider.state.configured is True
    assert provider.state.authorized is True
    assert provider.state.routable is True
    assert provider.state.reachable is None
    assert _capability(provider).max_context_tokens is None
    assert model.name == "vendor/unpublished-vision"
    assert dict(model.labels)["invocation_model"] == "vendor/Unpublished-Vision"
    assert _capability(model).max_context_tokens is None
    assert "configured-for-test" not in json.dumps(provider.to_dict())


def test_discovery_never_constructs_clients_fetches_media_or_loads_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def factory() -> object:
        events.append("factory")
        raise AssertionError("discovery constructed a provider")

    multimodal_router.register_multimodal_provider("fixture", factory)

    def blocked(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        events.append("side-effect")
        raise AssertionError("discovery attempted a runtime side effect")

    monkeypatch.setattr(multimodal_router.urllib.request, "urlopen", blocked)
    monkeypatch.setattr(multimodal_router, "_encode_image_for_api", blocked)
    monkeypatch.setattr(multimodal_router.llm_router, "get_llm_provider", blocked)
    for name in (
        "_builtin_provider_by_name",
        "_get_openrouter_provider",
        "_get_openai_provider",
        "_get_xai_multimodal_provider",
        "_get_meta_ai_multimodal_provider",
        "_get_huggingface_provider",
        "_get_backend_manager_provider",
    ):
        monkeypatch.setattr(multimodal_router, name, blocked)

    assert multimodal_router.get_provider_descriptor("fixture").name == "fixture"
    assert "fixture" in {
        provider.name for provider in multimodal_router.list_providers()
    }
    assert multimodal_router.list_models("fixture") == []
    assert isinstance(multimodal_router.get_catalog_snapshot(), CatalogSnapshot)
    assert events == []


def test_dynamic_alias_resolution_matches_existing_generation_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Optional[str], object, Optional[str]]] = []

    class FixtureProvider:
        def generate(
            self,
            prompt: str,
            *,
            image: object = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = kwargs
            calls.append((prompt, model_name, image, device))
            return f"{model_name}:{prompt}"

    provider = ProviderDescriptor(
        name="catalog_fixture",
        aliases=("catalog_alias",),
        capabilities=(
            CapabilityDescriptor(
                operations=(Operation.VISION_GENERATE,),
                input_modalities=(Modality.IMAGE, Modality.TEXT),
                output_modalities=(Modality.TEXT,),
                media_types=("image/*", "text/plain"),
                max_input_bytes=4096,
            ),
        ),
        lifecycle=LifecycleState.READY,
        state=OperationalState(
            known=True,
            configured=True,
            authorized=True,
            reachable=True,
            healthy=True,
            routable=True,
        ),
        labels={
            "access_requirement": "none",
            "batching": "unsupported",
            "device": "cpu",
            "image_input_modes": "inline,uri",
            "inline_input_types": "bytes,data-uri,file-path",
            "input_media_types": "image/*,text/plain",
            "locality": "local",
            "max_images": "1",
            "output_media_types": "text/plain",
            "streaming": "unsupported",
            "uri_schemes": "http,https",
        },
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="fixture/vision-v1",
        capabilities=provider.capabilities,
        lifecycle=LifecycleState.READY,
        state=provider.state,
        labels={"invocation_model": "fixture/vision-v1"},
    )
    multimodal_router.register_multimodal_provider(
        "catalog_fixture",
        FixtureProvider,
        descriptor=provider,
        models=(model,),
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER",
        "catalog_alias",
    )

    resolved = multimodal_router.resolve_model(
        "fixture/vision-v1",
        media_type="image/png",
        image_input_mode="inline",
        image_count=1,
        size_bytes=1024,
        device="cpu",
        authorized=True,
        ready=True,
    )
    generated = multimodal_router.generate_multimodal(
        "describe",
        image=b"image",
        model_name="fixture/vision-v1",
        device="cpu",
    )

    assert resolved.provider_id == provider.provider_id
    assert resolved.model_id == model.model_id
    assert generated == "fixture/vision-v1:describe"
    assert calls == [
        ("describe", "fixture/vision-v1", b"image", "cpu"),
    ]


def test_dynamic_model_filters_apply_provider_and_model_constraints() -> None:
    provider = ProviderDescriptor(
        name="bounded_fixture",
        aliases=("bounded_alias",),
        capabilities=(
            CapabilityDescriptor(
                operations=(Operation.VISION_GENERATE,),
                input_modalities=(Modality.IMAGE, Modality.TEXT),
                output_modalities=(Modality.TEXT,),
                media_types=("image/*", "text/plain"),
                max_input_bytes=8192,
            ),
        ),
        lifecycle=LifecycleState.READY,
        state=OperationalState(
            known=True,
            configured=True,
            authorized=True,
            reachable=True,
            healthy=True,
            routable=True,
        ),
        labels={
            "device": "cuda",
            "locality": "local",
        },
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="fixture/bounded-vision",
        capabilities=(
            CapabilityDescriptor(
                operations=(Operation.VISION_GENERATE,),
                input_modalities=(Modality.IMAGE, Modality.TEXT),
                output_modalities=(Modality.TEXT,),
                media_types=("image/*", "text/plain"),
                max_input_bytes=4096,
            ),
        ),
        lifecycle=LifecycleState.READY,
        state=provider.state,
        labels={"invocation_model": "fixture/bounded-vision"},
    )
    multimodal_router.register_multimodal_provider(
        "bounded_fixture",
        lambda: object(),
        descriptor=provider,
        models=(model,),
    )

    assert [
        record.name
        for record in multimodal_router.list_models(
            "bounded_alias",
            locality="local",
            device="cuda",
            size_bytes=4096,
            ready=True,
        )
    ] == ["fixture/bounded-vision"]
    assert (
        multimodal_router.list_models(
            "bounded_fixture",
            locality="remote",
        )
        == []
    )
    assert (
        multimodal_router.list_models(
            "bounded_fixture",
            device="cpu",
        )
        == []
    )
    assert (
        multimodal_router.list_models(
            "bounded_fixture",
            size_bytes=4097,
        )
        == []
    )
    with pytest.raises(ValueError, match="model .* incompatible"):
        multimodal_router.resolve_model(
            "fixture/bounded-vision",
            provider="bounded_fixture",
            size_bytes=4097,
        )


def test_builtin_alias_precedence_is_deterministic_on_dynamic_collision() -> None:
    collision = ProviderDescriptor(
        name="alias_collision_fixture",
        aliases=("gpt4o",),
    )
    multimodal_router.register_multimodal_provider(
        collision.name,
        lambda: object(),
        descriptor=collision,
    )

    assert multimodal_router.get_provider_descriptor("gpt4o").name == "openai"
    assert (
        multimodal_router.get_provider_descriptor(
            "alias_collision_fixture"
        ).aliases
        == ("gpt4o",)
    )


@pytest.mark.parametrize(
    ("constraint", "value"),
    (
        ("image_count", 1.5),
        ("image_count", True),
        ("size_bytes", "1024"),
        ("size_bytes", False),
    ),
)
def test_numeric_constraints_reject_non_integer_values(
    constraint: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        multimodal_router.list_providers(**{constraint: value})


def test_dynamic_registration_and_snapshot_order_are_deterministic() -> None:
    def factory() -> object:
        raise AssertionError("factory should not be called")

    before = multimodal_router.get_catalog_snapshot()
    multimodal_router.register_multimodal_provider("zeta_fixture", factory)
    multimodal_router.register_multimodal_provider("alpha_fixture", factory)

    names = [
        provider.name for provider in multimodal_router.list_providers()
    ]
    assert names == sorted(names)
    alpha = multimodal_router.get_provider_descriptor("alpha_fixture")
    assert alpha.state.known is True
    assert alpha.state.configured is True
    assert alpha.state.authorized is None
    assert alpha.state.reachable is None
    assert alpha.state.healthy is None
    assert alpha.state.routable is None
    assert dict(alpha.labels)["device"] == "unknown"
    assert dict(alpha.labels)["streaming"] == "unknown"
    assert dict(alpha.labels)["batching"] == "unknown"
    assert multimodal_router.list_models("alpha_fixture") == []

    first = multimodal_router.get_catalog_snapshot()
    second = multimodal_router.catalog_snapshot()
    assert first.revision != before.revision
    assert first.revision == second.revision
    assert first.to_dict() == second.to_dict()


def test_catalog_projection_contains_typed_model_bindings() -> None:
    snapshot = multimodal_router.catalog_snapshot()

    assert isinstance(snapshot, CatalogSnapshot)
    assert {provider.provider_id for provider in snapshot.providers} == {
        provider.provider_id for provider in multimodal_router.list_providers()
    }
    assert {model.model_id for model in snapshot.models} == {
        model.model_id for model in multimodal_router.list_models()
    }
    assert len(snapshot.bindings) == len(snapshot.models)
    assert all(
        binding.router == "multimodal_router"
        for binding in snapshot.bindings
    )
    assert all(
        binding.operations
        == (Operation.TEXT_GENERATE, Operation.VISION_GENERATE)
        for binding in snapshot.bindings
    )
    assert {binding.model_id for binding in snapshot.bindings} == {
        model.model_id for model in snapshot.models
    }
