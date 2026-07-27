from __future__ import annotations

import json
from typing import Optional

import pytest

from ipfs_accelerate_py import llm_router
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
    "ipfs_accelerate_py_LLM_PROVIDER",
    "IPFS_ACCELERATE_PY_LLM_PROVIDER",
    "IPFS_DATASETS_PY_LLM_PROVIDER",
    "ipfs_accelerate_py_ENABLE_IPFS_ACCELERATE",
    "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
    "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
    "ipfs_accelerate_py_LLM_MODEL",
    "IPFS_ACCELERATE_PY_LLM_MODEL",
    "IPFS_DATASETS_PY_LLM_MODEL",
    "ipfs_accelerate_py_OPENROUTER_API_KEY",
    "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
    "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "OPENAI_TOKEN",
    "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
    "ipfs_accelerate_py_OPENAI_API_KEY",
    "IPFS_ACCELERATE_PY_HF_API_TOKEN",
    "ipfs_accelerate_py_HF_API_TOKEN",
    "IPFS_DATASETS_PY_HF_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HF_TOKEN",
    "XAI_API_KEY",
    "ipfs_accelerate_py_XAI_API_KEY",
    "IPFS_ACCELERATE_PY_XAI_API_KEY",
    "IPFS_DATASETS_PY_XAI_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_router, "_PROVIDER_REGISTRY", {})
    for name in _DISCOVERY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    llm_router.clear_llm_router_caches()


def _capability(record: ProviderDescriptor | ModelDescriptor) -> CapabilityDescriptor:
    assert len(record.capabilities) == 1
    return record.capabilities[0]


def test_all_builtin_providers_publish_typed_llm_metadata() -> None:
    providers = llm_router.list_providers()

    assert [provider.name for provider in providers] == sorted(
        {
            "accelerate",
            "claude_code",
            "claude_py",
            "codex_cli",
            "copilot_cli",
            "copilot_sdk",
            "gemini_cli",
            "gemini_py",
            "grok_cli",
            "hf_inference_api",
            "llama_cpp",
            "llama_cpp_native",
            "local_hf",
            "meta_ai",
            "mistral_vibe",
            "mock",
            "openai",
            "openrouter",
            "p2p_task_queue",
            "xai",
        }
    )
    assert all(isinstance(provider, ProviderDescriptor) for provider in providers)
    for provider in providers:
        capability = _capability(provider)
        labels = dict(provider.labels)
        assert Operation.TEXT_GENERATE in capability.operations
        assert Operation.BATCH in capability.operations
        assert capability.input_modalities == (Modality.TEXT,)
        assert capability.output_modalities == (Modality.TEXT,)
        assert {
            "access_requirement",
            "batching",
            "device",
            "locality",
            "model_hint",
            "streaming",
            "tools",
        } <= labels.keys()
        assert provider.state.known is True
        assert provider.state.reachable is None or provider.name == "mock"
        assert provider.state.healthy is None or provider.name == "mock"

    assert llm_router.get_provider_descriptor("codex").name == "codex_cli"
    assert llm_router.get_provider_descriptor("hf_api").name == "hf_inference_api"
    assert llm_router.get_provider_descriptor("llama.cpp").name == "llama_cpp"
    assert llm_router.get_provider_descriptor("meta").name == "meta_ai"
    assert llm_router.get_provider_descriptor("hf").name == "local_hf"


def test_known_context_limits_and_unknown_model_facts() -> None:
    openrouter = llm_router.get_provider_descriptor("openrouter")
    assert _capability(openrouter).max_context_tokens == 128_000
    assert openrouter.state.configured is False
    assert openrouter.state.authorized is False
    assert openrouter.state.reachable is None

    known = llm_router.resolve_model(
        "openai/gpt-4o-mini",
        provider="openrouter",
    )
    assert known.name == "openai/gpt-4o-mini"
    assert _capability(known).max_context_tokens == 128_000
    assert dict(known.labels)["invocation_model"] == "openai/gpt-4o-mini"

    unknown = llm_router.resolve_model(
        "vendor/Future-Model",
        provider="openrouter",
    )
    assert unknown.name == "vendor/future-model"
    assert unknown.architecture is None
    assert _capability(unknown).max_context_tokens is None
    assert dict(unknown.labels)["invocation_model"] == "vendor/Future-Model"


def test_environment_metadata_is_projected_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured-for-test")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_OPENROUTER_MODEL",
        "vendor/Unpublished-Chat",
    )

    provider = llm_router.get_provider_descriptor("openrouter")
    model = llm_router.list_models("openrouter")[0]

    assert provider.lifecycle is LifecycleState.CONFIGURED
    assert provider.state.configured is True
    assert provider.state.authorized is True
    assert provider.state.reachable is None
    assert _capability(provider).max_context_tokens is None
    assert model.name == "vendor/unpublished-chat"
    assert _capability(model).max_context_tokens is None
    assert "configured-for-test" not in json.dumps(provider.to_dict())


def test_discovery_never_constructs_clients_installs_probes_or_loads_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def factory() -> object:
        events.append("factory")
        raise AssertionError("discovery constructed a provider")

    llm_router.register_llm_provider("fixture", factory)

    def blocked(*args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        events.append("side-effect")
        raise AssertionError("discovery attempted a runtime side effect")

    monkeypatch.setattr(llm_router.importlib, "import_module", blocked)
    monkeypatch.setattr(llm_router.urllib.request, "urlopen", blocked)
    monkeypatch.setattr(llm_router.subprocess, "run", blocked)
    for name in (
        "_builtin_provider_by_name",
        "_get_accelerate_provider",
        "_get_local_hf_provider",
        "_get_openrouter_provider",
        "_get_openai_provider",
        "_get_hf_inference_api_provider",
        "_get_llama_cpp_provider",
        "_get_llama_cpp_native_provider",
        "_get_mistral_vibe_provider",
        "_get_xai_provider",
        "_get_meta_ai_provider",
    ):
        monkeypatch.setattr(llm_router, name, blocked)

    assert llm_router.get_provider_descriptor("fixture").name == "fixture"
    assert "fixture" in {provider.name for provider in llm_router.list_providers()}
    assert isinstance(llm_router.get_catalog_snapshot(), CatalogSnapshot)
    assert events == []


def test_dynamic_metadata_alias_and_resolution_match_generate_text() -> None:
    calls: list[tuple[str, Optional[str]]] = []

    class FixtureProvider:
        def generate(
            self,
            prompt: str,
            *,
            model_name: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = kwargs
            calls.append((prompt, model_name))
            return f"{model_name}:{prompt}"

    provider = ProviderDescriptor(
        name="catalog_fixture",
        aliases=("catalog_alias",),
        capabilities=(
            CapabilityDescriptor(
                operations=(
                    Operation.TEXT_GENERATE,
                    Operation.TEXT_CHAT,
                    Operation.BATCH,
                    Operation.TOOL_CALL,
                ),
                input_modalities=(Modality.TEXT,),
                output_modalities=(Modality.TEXT,),
                max_context_tokens=4096,
            ),
        ),
        lifecycle=LifecycleState.READY,
        labels={
            "access_requirement": "none",
            "batching": "supported",
            "device": "cpu",
            "locality": "local",
            "streaming": "unknown",
            "tools": "supported",
        },
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="fixture/chat-v1",
        capabilities=provider.capabilities,
        lifecycle=LifecycleState.READY,
        labels={"invocation_model": "fixture/chat-v1"},
    )
    llm_router.register_llm_provider(
        "catalog_fixture",
        FixtureProvider,
        descriptor=provider,
        models=(model,),
    )

    resolved = llm_router.resolve_model(
        "fixture/chat-v1",
        provider="catalog_alias",
    )
    generated = llm_router.generate_text(
        "hello",
        provider="catalog_alias",
        model_name="fixture/chat-v1",
        allow_local_fallback=False,
    )

    assert resolved.provider_id == provider.provider_id
    assert resolved.model_id == model.model_id
    assert generated == "fixture/chat-v1:hello"
    assert calls == [("hello", "fixture/chat-v1")]


def test_dynamic_registration_order_unknowns_and_snapshot_are_deterministic() -> None:
    def factory() -> object:
        raise AssertionError("factory should not be called")

    before = llm_router.get_catalog_snapshot()
    llm_router.register_llm_provider("zeta_fixture", factory)
    llm_router.register_llm_provider("alpha_fixture", factory)

    names = [provider.name for provider in llm_router.list_providers()]
    assert names == sorted(names)
    alpha = llm_router.get_provider_descriptor("alpha_fixture")
    assert alpha.state.known is True
    assert alpha.state.configured is True
    assert alpha.state.authorized is None
    assert alpha.state.reachable is None
    assert alpha.state.healthy is None
    assert alpha.state.routable is None
    assert dict(alpha.labels)["device"] == "unknown"
    assert dict(alpha.labels)["streaming"] == "unknown"
    assert dict(alpha.labels)["tools"] == "unknown"
    assert llm_router.list_models("alpha_fixture") == []

    first = llm_router.get_catalog_snapshot()
    second = llm_router.catalog_snapshot()
    assert first.revision != before.revision
    assert first.revision == second.revision
    assert first.to_dict() == second.to_dict()


def test_catalog_snapshot_contains_typed_model_bindings() -> None:
    snapshot = llm_router.get_catalog_snapshot()

    assert isinstance(snapshot, CatalogSnapshot)
    assert {provider.provider_id for provider in snapshot.providers} == {
        provider.provider_id for provider in llm_router.list_providers()
    }
    assert {model.model_id for model in snapshot.models} == {
        model.model_id for model in llm_router.list_models()
    }
    assert len(snapshot.bindings) == len(snapshot.models)
    assert all(binding.router == "llm_router" for binding in snapshot.bindings)
    assert all(Operation.TEXT_GENERATE in binding.operations for binding in snapshot.bindings)
    assert {binding.model_id for binding in snapshot.bindings} == {
        model.model_id for model in snapshot.models
    }
