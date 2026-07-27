"""Regression tests for the public API model-registry compatibility surface."""

from ipfs_accelerate_py.api_backends.api_models_registry import (
    APIModelsRegistry,
    api_models,
)
from ipfs_accelerate_py.api_integrations.model_registry import (
    APIModel,
    APIModelRegistry,
    APIProviderType,
    get_all_pipeline_types,
    get_api_models_for_pipeline,
)


def test_registry_retains_provider_and_pipeline_lookup_shapes():
    registry = APIModelRegistry()

    openai_models = registry.get_models_by_provider(APIProviderType.OPENAI)
    text_models = registry.get_models_by_pipeline_type("text-generation")

    assert isinstance(openai_models, list)
    assert openai_models
    assert all(isinstance(model, APIModel) for model in openai_models)
    assert all(model.provider is APIProviderType.OPENAI for model in openai_models)
    assert isinstance(text_models, list)
    assert text_models
    assert all("text-generation" in model.pipeline_types for model in text_models)


def test_registry_get_and_all_models_preserve_legacy_values():
    registry = APIModelRegistry()

    model = registry.get_model("gpt-4")

    assert isinstance(model, APIModel)
    assert model.model_id == "gpt-4"
    assert model.provider is APIProviderType.OPENAI
    assert model in registry.get_all_models()
    assert APIProviderType.OPENAI in registry.get_all_providers()
    assert "text-generation" in registry.get_supported_pipeline_types()


def test_custom_model_registration_is_visible_through_all_legacy_lookups():
    registry = APIModelRegistry()
    custom = APIModel(
        model_id="compatibility-regression-model",
        model_name="Compatibility Regression Model",
        provider=APIProviderType.COHERE,
        pipeline_types=["text-generation", "conversational"],
        context_length=2048,
        supports_streaming=True,
    )

    registry.register_custom_model(custom)

    assert registry.get_model(custom.model_id) == custom
    assert custom in registry.get_models_by_provider(APIProviderType.COHERE)
    assert custom in registry.get_models_by_pipeline_type("conversational")


def test_backend_registry_keeps_historical_alias_and_return_shapes():
    registry = APIModelRegistry()
    backend_registry = APIModelsRegistry(registry=registry)

    assert isinstance(backend_registry, api_models)
    assert isinstance(backend_registry.model_lists, dict)
    assert isinstance(backend_registry.get_models("openai"), list)
    assert backend_registry.get_backend_for_model("openai/gpt-4") == "openai_api"
    assert backend_registry.get_backend_for_model(
        "anthropic/claude-3-opus"
    ) == "claude"
    assert backend_registry.get_backend_for_model("unknown/model") is None


def test_module_helpers_keep_list_and_set_return_types():
    models = get_api_models_for_pipeline("text-generation")
    pipeline_types = get_all_pipeline_types()

    assert isinstance(models, list)
    assert models
    assert all(isinstance(model, APIModel) for model in models)
    assert isinstance(pipeline_types, set)
    assert "text-generation" in pipeline_types
