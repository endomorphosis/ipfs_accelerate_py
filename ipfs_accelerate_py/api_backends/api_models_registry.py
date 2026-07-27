"""Backend-name compatibility projection over the canonical API model catalog.

The old implementation scanned ``model_list/*.json`` and initialized a
distributed storage wrapper at import time.  Static knowledge now comes only
from ``api_integrations.model_registry`` and this module performs no file,
storage, credential, or network discovery.

This class remains available indefinitely as a reversible compatibility
adapter.  New callers should resolve models through ModelManager's catalog.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..api_integrations.model_registry import (
    APIModel,
    APIModelRegistry,
    APIProviderType,
    LEGACY_REGISTRY_DEPRECATION,
    get_global_api_model_registry,
)


_BACKEND_ALIASES: Mapping[str, str] = {
    "anthropic": "claude",
    "claude": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "groq": "groq",
    "hf-tei": "hf_tei",
    "hf-tgi": "hf_tgi",
    "hf_tei": "hf_tei",
    "hf_tgi": "hf_tgi",
    "huggingface": "hf_tgi",
    "meta-ai": "meta_ai",
    "meta_ai": "meta_ai",
    "meta-llama": "meta_ai",
    "meta-spark": "meta_ai",
    "meta_spark": "meta_ai",
    "ollama": "ollama",
    "openai": "openai_api",
    "openai-api": "openai_api",
    "openai_api": "openai_api",
    "openvino": "ovms",
    "ovms": "ovms",
}

_PREFIX_BACKENDS: Mapping[str, str] = {
    "anthropic": "claude",
    "claude": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "groq": "groq",
    "hf-tei": "hf_tei",
    "hf-tgi": "hf_tgi",
    "hf_tei": "hf_tei",
    "hf_tgi": "hf_tgi",
    "huggingface": "hf_tgi",
    "meta-ai": "meta_ai",
    "meta_ai": "meta_ai",
    "meta-llama": "meta_ai",
    "meta-spark": "meta_ai",
    "meta_spark": "meta_ai",
    "ollama": "ollama",
    "openai": "openai_api",
    "openai-api": "openai_api",
    "openai_api": "openai_api",
    "openvino": "ovms",
    "ovms": "ovms",
}


class api_models:
    """Preserve backend routing APIs as deterministic catalog projections."""

    deprecation = LEGACY_REGISTRY_DEPRECATION

    def __init__(
        self,
        resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        registry: Optional[APIModelRegistry] = None,
    ) -> None:
        self.resources = resources if resources is not None else {}
        self.metadata = metadata if metadata is not None else {}
        injected = registry or self.resources.get("api_model_registry")
        if injected is not None and not isinstance(injected, APIModelRegistry):
            raise TypeError("api_model_registry resource must be APIModelRegistry")
        self.registry = injected or get_global_api_model_registry()

    @property
    def model_lists(self) -> Dict[str, List[str]]:
        """Return fresh lists matching the historical dictionary value shape."""

        return self.registry.get_backend_model_lists()

    @property
    def catalog_revision(self) -> str:
        return self.registry.catalog_revision

    def _backend_name(self, value: str) -> str:
        if not isinstance(value, str):
            return ""
        return _BACKEND_ALIASES.get(value.strip().casefold(), value.strip().casefold())

    def get_backend_for_model(self, model_name: str) -> Optional[str]:
        """Determine the backend for a canonical name or model alias."""

        if not isinstance(model_name, str) or not model_name.strip():
            return None
        selected = model_name.strip()
        selected_folded = selected.casefold()

        # Prefer the catalog projection's explicit backend membership.
        for backend, models in self.model_lists.items():
            if selected_folded in {item.casefold() for item in models}:
                return backend

        # A canonical/legacy alias may not be the spelling stored for a
        # backend, so resolve it before falling back to its provider prefix.
        model = self.registry.get_model(selected)
        if model is not None:
            if model.provider == APIProviderType.HUGGINGFACE:
                if "feature-extraction" in model.pipeline_types:
                    return "hf_tei"
                return "hf_tgi"
            return {
                APIProviderType.ANTHROPIC: "claude",
                APIProviderType.GOOGLE: "gemini",
                APIProviderType.OPENAI: "openai_api",
                APIProviderType.OVMS: "ovms",
            }.get(model.provider, model.provider.value)

        prefix = selected_folded.split("/", 1)[0] if "/" in selected_folded else ""
        backend = _PREFIX_BACKENDS.get(prefix)
        if backend == "hf_tgi" and any(
            term in selected_folded
            for term in ("embedding", "encoder", "sentence")
        ):
            return "hf_tei"
        return backend

    def get_models_for_backend(self, backend_name: str) -> List[str]:
        """Return a new list for the requested backend or backend alias."""

        return list(self.model_lists.get(self._backend_name(backend_name), ()))

    def get_models(self, api_name: str) -> List[str]:
        """Historical alias for :meth:`get_models_for_backend`."""

        return self.get_models_for_backend(api_name)

    def list_models(self, api_name: Optional[str] = None) -> List[str]:
        """List one backend or all qualified model names deterministically."""

        if api_name is not None:
            return self.get_models_for_backend(api_name)
        result: List[str] = []
        seen = set()
        for models in self.model_lists.values():
            for model in models:
                folded = model.casefold()
                if folded not in seen:
                    result.append(model)
                    seen.add(folded)
        return result

    def search_models(
        self, query: str, api_name: Optional[str] = None
    ) -> List[str]:
        """Search qualified legacy model names."""

        if not isinstance(query, str):
            return []
        needle = query.strip().casefold()
        models = self.list_models(api_name)
        if not needle:
            return models
        return [model for model in models if needle in model.casefold()]

    def is_compatible_model(self, api_name: str, model_name: str) -> bool:
        """Check explicit membership, including catalog aliases."""

        backend = self._backend_name(api_name)
        resolved = self.get_backend_for_model(model_name)
        return resolved == backend

    def validate_model(self, api_name: str, model_name: str) -> bool:
        """Compatibility spelling for :meth:`is_compatible_model`."""

        return self.is_compatible_model(api_name, model_name)

    def get_model(self, model_name: str) -> Optional[APIModel]:
        """Return the established APIModel shape for any recognized alias."""

        return self.registry.get_model(model_name)

    def add_model(self, model: APIModel) -> None:
        """Persist an addition through the registry's runtime catalog source."""

        self.registry.add_model(model)

    def recommend_models(
        self, pipeline_type: str, **kwargs: Any
    ) -> List[APIModel]:
        """Delegate deterministic recommendation to the catalog projection."""

        return self.registry.recommend_models(pipeline_type, **kwargs)

    def recommend_model(
        self, pipeline_type: str, **kwargs: Any
    ) -> Optional[APIModel]:
        """Return the first deterministic recommendation."""

        return self.registry.recommend_model(pipeline_type, **kwargs)

    def export_models(self) -> Dict[str, List[str]]:
        """Export the historical backend-to-list mapping."""

        return self.model_lists

    export = export_models


APIModelsRegistry = api_models

__all__ = [
    "APIModelsRegistry",
    "api_models",
]
