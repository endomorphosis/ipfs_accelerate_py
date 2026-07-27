"""Parity and convergence tests for the legacy API model registries."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.api_backends.api_models_registry import (
    APIModelsRegistry,
    api_models,
)
from ipfs_accelerate_py.api_integrations.model_registry import (
    API_MODEL_SEED_ROWS,
    APIModel,
    APIModelRegistry,
    APIProviderType,
    LEGACY_REGISTRY_DEPRECATION,
    RUNTIME_SOURCE_NAME,
)


RICH_FIXTURES = tuple(
    model
    for group in (
        APIModelRegistry.OPENAI_MODELS,
        APIModelRegistry.ANTHROPIC_MODELS,
        APIModelRegistry.GOOGLE_MODELS,
        APIModelRegistry.GROQ_MODELS,
        APIModelRegistry.COHERE_MODELS,
    )
    for model in group
)

EXPECTED_BACKEND_COUNTS = {
    "claude": 11,
    "cohere": 2,
    "gemini": 7,
    "groq": 14,
    "hf_tei": 14,
    "hf_tgi": 17,
    "ollama": 23,
    "openai_api": 52,
    "ovms": 19,
}

GENERATED_PROJECTION_SHA256 = (
    "c521d83265b34bf684e5d0839b92c7468a4f4569d36e8484759bae42df6ea64c"
)


def _legacy_dict(model):
    return dataclasses.asdict(model)


def test_rich_legacy_fields_are_exact_catalog_projections():
    registry = APIModelRegistry()

    assert len(RICH_FIXTURES) == 15
    for expected in RICH_FIXTURES:
        actual = registry.get_model(
            "%s/%s" % (expected.provider.value, expected.model_id)
        )
        assert isinstance(actual, APIModel)
        assert _legacy_dict(actual) == _legacy_dict(expected)


def test_canonical_seed_has_unique_catalog_identities_and_provider_metadata():
    registry = APIModelRegistry()
    snapshot = registry.catalog.snapshot()

    assert len(API_MODEL_SEED_ROWS) > len(snapshot.models)
    assert len(snapshot.models) == 159
    assert len({model.model_id for model in snapshot.models}) == len(
        snapshot.models
    )
    assert len({provider.provider_id for provider in snapshot.providers}) == len(
        snapshot.providers
    )

    providers = {provider.name: provider for provider in snapshot.providers}
    assert set(providers) == {
        "anthropic",
        "cohere",
        "google",
        "groq",
        "huggingface",
        "ollama",
        "openai",
        "ovms",
    }
    assert "gemini" in providers["google"].aliases
    assert "claude" in providers["anthropic"].aliases
    assert "openvino" in providers["ovms"].aliases
    assert all(provider.provenance for provider in providers.values())
    assert all(model.provenance for model in snapshot.models)


def test_generated_projection_fixture_detects_any_legacy_field_drift():
    registry = APIModelRegistry()
    payload = {
        "models": registry.export_models(),
        "backend_models": registry.get_backend_model_lists(),
        "providers": [
            provider.to_dict()
            for provider in registry.catalog.snapshot().providers
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == GENERATED_PROJECTION_SHA256


def test_provider_and_model_aliases_resolve_to_canonical_ids():
    registry = APIModelRegistry()

    assert registry.resolve_provider_id("gemini") == registry.resolve_provider_id(
        "google"
    )
    assert registry.resolve_provider_id("claude") == registry.resolve_provider_id(
        APIProviderType.ANTHROPIC
    )
    assert registry.resolve_provider_id("openvino") == registry.resolve_provider_id(
        "ovms"
    )
    assert registry.resolve_provider_id("hf-tgi") == registry.resolve_provider_id(
        "huggingface"
    )
    assert registry.resolve_provider_id("openai-api") == (
        registry.resolve_provider_id("openai")
    )

    dated = registry.resolve_model_id("claude-3-opus-20240229")
    assert dated is not None
    assert registry.resolve_model_id("anthropic/claude-3-opus") == dated
    assert registry.resolve_model_id("claude/claude-3-opus") == dated
    assert registry.resolve_model_id("openai_api/gpt-4") == (
        registry.resolve_model_id("openai/gpt-4")
    )
    assert registry.resolve_model_id("ovms/bert-base-uncased") == (
        registry.resolve_model_id("openvino/bert-base-uncased")
    )
    assert registry.resolve_model_name("claude-3-opus") == (
        "claude-3-opus-20240229"
    )
    assert registry.get_model(dated).model_id == "claude-3-opus-20240229"


def test_legacy_list_search_recommend_validate_get_and_export_shapes():
    registry = APIModelRegistry()

    assert isinstance(registry.get_all_models(), list)
    assert all(isinstance(model, APIModel) for model in registry.list_models())
    assert all(
        model.provider == APIProviderType.OPENAI
        for model in registry.list_models(provider="openai_api")
    )
    assert registry.search_models("GPT-4")[0].model_id == "gpt-4-turbo"
    assert registry.validate_model(
        "openai/gpt-4", provider="openai", pipeline_type="text-generation"
    )
    assert not registry.validate_model(
        "openai/gpt-4", provider="anthropic"
    )
    assert isinstance(registry.recommend_models("text-generation"), list)
    assert isinstance(registry.recommend_model("text-generation"), APIModel)

    first = registry.export_models()
    second = registry.export_models()
    assert first == second
    assert isinstance(first, list)
    assert isinstance(first[0], dict)
    assert set(first[0]) == {field.name for field in dataclasses.fields(APIModel)}
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert isinstance(registry.export_catalog(), dict)


def test_runtime_additions_publish_through_runtime_catalog_source():
    registry = APIModelRegistry()
    backend_registry = api_models(registry=registry)
    before_revision = registry.catalog_revision
    provider_before = registry.catalog.get("openai", record_type="providers")
    custom = APIModel(
        model_id="acme-chat-v1",
        model_name="Acme Chat V1",
        provider=APIProviderType.OPENAI,
        pipeline_types=["text-generation", "conversational"],
        context_length=4096,
        supports_streaming=True,
        cost_per_1k_tokens={"input": 0.001, "output": 0.002},
        function_calling=True,
    )

    assert registry.add_model(custom) is None
    assert registry.catalog_revision != before_revision
    assert registry.get_model("openai/acme-chat-v1") == custom
    stable_id = registry.resolve_model_id("openai/acme-chat-v1")
    claims = registry.catalog.claims(
        stable_id, record_type="models", source=RUNTIME_SOURCE_NAME
    )
    assert len(claims) == 1
    assert backend_registry.get_backend_for_model("openai/acme-chat-v1") == (
        "openai_api"
    )
    assert "openai/acme-chat-v1" in backend_registry.get_models("openai")
    provider_after = registry.catalog.get("openai", record_type="providers")
    assert provider_after.aliases == provider_before.aliases
    assert provider_after.display_name == provider_before.display_name

    # A new registry sees the canonical seed, not a mutated module-level list.
    assert APIModelRegistry().get_model("openai/acme-chat-v1") is None


def test_invalid_runtime_addition_does_not_change_catalog_or_seed():
    registry = APIModelRegistry()
    before_revision = registry.catalog_revision
    before_seed = tuple(dict(row) for row in API_MODEL_SEED_ROWS)

    invalid = APIModel(
        model_id="invalid context",
        model_name="Invalid Context",
        provider=APIProviderType.OPENAI,
        pipeline_types=["text-generation"],
        context_length=0,
    )
    with pytest.raises(ValueError, match="invalid API model"):
        registry.add_model(invalid)

    assert registry.catalog_revision == before_revision
    assert tuple(dict(row) for row in API_MODEL_SEED_ROWS) == before_seed
    assert registry.get_model("openai/invalid context") is None


def test_backend_inventory_is_a_deterministic_projection():
    registry = APIModelRegistry()
    legacy = APIModelsRegistry(registry=registry)

    assert isinstance(legacy, api_models)
    assert {
        backend: len(models)
        for backend, models in legacy.model_lists.items()
    } == EXPECTED_BACKEND_COUNTS
    assert legacy.model_lists == legacy.export_models()
    assert legacy.get_backend_for_model("openai/gpt-4") == "openai_api"
    assert legacy.get_backend_for_model("openai_api/gpt-4") == "openai_api"
    assert legacy.get_backend_for_model("anthropic/claude-3-opus") == "claude"
    assert legacy.get_backend_for_model(
        "huggingface/all-mpnet-base-v2"
    ) == "hf_tei"
    assert legacy.get_backend_for_model("meta-llama/example") == "meta_ai"
    assert legacy.get_backend_for_model("meta-spark/example") == "meta_ai"
    assert legacy.get_backend_for_model("unknown/example") is None
    assert legacy.is_compatible_model("google", "google/gemini-pro")
    assert isinstance(legacy.get_models_for_backend("openai"), list)
    assert isinstance(legacy.search_models("gpt", "openai"), list)
    assert isinstance(legacy.get_model("openai/gpt-4"), APIModel)


def test_deprecation_is_documented_reversible_and_has_no_removal_version():
    assert LEGACY_REGISTRY_DEPRECATION == {
        "deprecated": True,
        "replacement": "ModelManager model catalog",
        "removal_scheduled": False,
        "reversible": True,
    }
    assert APIModelRegistry.deprecation is LEGACY_REGISTRY_DEPRECATION
    assert api_models.deprecation is LEGACY_REGISTRY_DEPRECATION


@pytest.mark.parametrize(
    "module_name",
    (
        "ipfs_accelerate_py.api_integrations.model_registry",
        "ipfs_accelerate_py.api_backends.api_models_registry",
    ),
)
def test_cold_import_does_not_attempt_network_discovery(module_name):
    script = """
import importlib
import sys

def audit(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
        raise RuntimeError("network discovery attempted: " + event)

sys.addaudithook(audit)
importlib.import_module(%r)
""" % module_name
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
