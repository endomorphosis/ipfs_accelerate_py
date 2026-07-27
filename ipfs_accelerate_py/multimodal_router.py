"""Multimodal router for ipfs_accelerate_py.

This module provides a stable, reusable entrypoint for multimodal inference
(image+text → text, image captioning, visual question answering, etc.) that
integrates with existing ipfs_accelerate_py infrastructure.

Design goals:
- Avoid import-time side effects (no heavy imports at module import).
- Allow optional hooks/providers (backend manager, custom remote endpoints).
- Provide a reliable local fallback via HuggingFace transformers.
- Reuse existing patterns from llm_router and embeddings_router.

Environment variables:
- `IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER`: force provider name
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: enable backend manager provider
- `IPFS_ACCELERATE_PY_MULTIMODAL_MODEL`: HF model name for local adapter
    (default: llava-hf/llava-1.5-7b-hf)
- `IPFS_ACCELERATE_PY_MULTIMODAL_DEVICE`: device for local adapter (cpu/cuda)

Additional optional providers (opt-in by selecting provider):
- `openrouter`: OpenRouter multimodal chat completions
    - `OPENROUTER_API_KEY` or `IPFS_ACCELERATE_PY_OPENROUTER_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENROUTER_MULTIMODAL_MODEL`
    - `IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL`
- `openai`: OpenAI vision API (GPT-4V / GPT-4o)
    - `OPENAI_API_KEY` or `IPFS_ACCELERATE_PY_OPENAI_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENAI_MULTIMODAL_MODEL` (default: gpt-4o)
    - `IPFS_ACCELERATE_PY_OPENAI_BASE_URL`
- `xai`: xAI Grok vision (grok-2-vision-1212) via OpenAI-compatible endpoint
    - `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY`
    - `ipfs_accelerate_py_XAI_MULTIMODAL_MODEL` (default: grok-2-vision-1212)
    - `ipfs_accelerate_py_XAI_BASE_URL` (default: https://api.x.ai/v1)
- `meta_ai`: Meta Llama vision (Llama-3.2-90B-Vision-Instruct) via OpenAI-compatible endpoint
    - `META_AI_API_KEY` or `ipfs_accelerate_py_META_AI_API_KEY`
    - `ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL` (default: meta-llama/Llama-3.2-90B-Vision-Instruct)
    - `ipfs_accelerate_py_META_AI_BASE_URL` (default: https://api.llamameta.net/v1)
- `huggingface`: HuggingFace transformers (LLaVA, InstructBLIP, etc.)
- `backend_manager`: Use InferenceBackendManager for distributed inference
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import logging
import mimetypes
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from . import llm_router
from .model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from .router_deps import RouterDeps, get_default_router_deps

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_DETAIL = "auto"
LLMRouterError = llm_router.LLMRouterError
get_llm_provider = llm_router.get_llm_provider
clear_llm_router_caches = llm_router.clear_llm_router_caches
chat_completions_create = llm_router.chat_completions_create


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_enabled() -> bool:
    value = (
        os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE")
        or "1"
    )
    return value.strip() != "0"


def _response_cache_enabled() -> bool:
    value = (
        os.environ.get("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_RESPONSE_CACHE")
    )
    if value is None:
        return True
    return str(value).strip() != "0"


def _stable_kwargs_digest(kwargs: Dict[str, object]) -> str:
    if not kwargs:
        return ""
    try:
        payload = json.dumps(kwargs, sort_keys=True, default=repr, ensure_ascii=False)
    except Exception:
        payload = repr(sorted(kwargs.items(), key=lambda x: str(x[0])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _text_digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _image_digest(image: Optional[Union[str, bytes]]) -> str:
    if image is None:
        return ""
    if isinstance(image, bytes):
        return hashlib.sha256(image).hexdigest()[:16]
    return hashlib.sha256(str(image).encode("utf-8")).hexdigest()[:16]


def _response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    prompt: str,
    image: Optional[Union[str, bytes]] = None,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = (model_name or "").strip()
    return (
        f"multimodal_response::{provider_key}::{model_key}"
        f"::{_text_digest(prompt)}::{_image_digest(image)}::{_stable_kwargs_digest(kwargs)}"
    )


@runtime_checkable
class MultimodalProvider(Protocol):
    """Provider interface for multimodal inference."""

    def generate(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, bytes]] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str: ...


ProviderFactory = Callable[[], MultimodalProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory
    descriptor: Optional[ProviderDescriptor] = None
    models: Tuple[ModelDescriptor, ...] = ()


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}
_PROVIDER_REGISTRY_LOCK = threading.RLock()


def register_multimodal_provider(
    name: str,
    factory: ProviderFactory,
    *,
    descriptor: ProviderDescriptor | Mapping[str, object] | None = None,
    models: Sequence[ModelDescriptor | Mapping[str, object]] = (),
) -> None:
    """Register a provider and optional side-effect-free catalog metadata.

    Discovery retains ``factory`` without calling it.  When catalog metadata
    is omitted, router-level input and output facts remain known while
    provider-specific deployment facts remain explicitly unknown.
    """

    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    normalized = name.strip().lower()
    provider_descriptor = _registered_provider_descriptor(normalized, descriptor)
    model_descriptors = _registered_model_descriptors(
        provider_descriptor,
        models,
    )
    with _PROVIDER_REGISTRY_LOCK:
        _PROVIDER_REGISTRY[normalized] = ProviderInfo(
            name=normalized,
            factory=factory,
            descriptor=provider_descriptor,
            models=model_descriptors,
        )


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


_MULTIMODAL_CATALOG_PROVENANCE = (
    Provenance(source="multimodal_router.static"),
)
_MULTIMODAL_REGISTRY_PROVENANCE = (
    Provenance(source="multimodal_router.registry"),
)
_MULTIMODAL_MEDIA_TYPES = ("image/*", "text/plain")
_MULTIMODAL_OPERATIONS = (
    Operation.TEXT_GENERATE,
    Operation.VISION_GENERATE,
)


def _multimodal_capability(
    *,
    max_context_tokens: Optional[int] = None,
    operations: Sequence[Operation] = _MULTIMODAL_OPERATIONS,
    media_types: Sequence[str] = _MULTIMODAL_MEDIA_TYPES,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        operations=tuple(operations),
        input_modalities=(Modality.TEXT, Modality.IMAGE),
        output_modalities=(Modality.TEXT,),
        media_types=tuple(media_types),
        max_context_tokens=max_context_tokens,
        # The router exposes one request at a time and does not enforce byte
        # limits.  None deliberately means unknown, not unlimited.
        max_batch_size=None,
        max_input_bytes=None,
        max_output_bytes=None,
    )


def _default_multimodal_labels(
    *,
    authorization: str,
    device: str,
    locality: str,
    streaming: str = "unsupported",
    batching: str = "unsupported",
) -> Dict[str, str]:
    """Return router-owned media contract labels.

    ``CapabilityDescriptor`` carries typed modality and byte-limit fields.
    These labels preserve distinctions not represented by schema v1: image
    count, URI versus inline transport, and directional media families.
    """

    return {
        "access_requirement": authorization,
        "batching": batching,
        "device": device,
        "image_input_modes": "inline,uri",
        "inline_input_types": "bytes,data-uri,file-path",
        "input_media_types": "image/*,text/plain",
        "locality": locality,
        "max_images": "1",
        "output_media_types": "text/plain",
        "streaming": streaming,
        "uri_schemes": "http,https",
    }


def _registered_provider_descriptor(
    name: str,
    descriptor: ProviderDescriptor | Mapping[str, object] | None,
) -> ProviderDescriptor:
    if descriptor is None:
        return ProviderDescriptor(
            name=name,
            description="Dynamically registered multimodal provider.",
            capabilities=(_multimodal_capability(),),
            lifecycle=LifecycleState.DECLARED,
            state=OperationalState(
                known=True,
                configured=True,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
            provenance=_MULTIMODAL_REGISTRY_PROVENANCE,
            labels=_default_multimodal_labels(
                authorization="unknown",
                device="unknown",
                locality="unknown",
                streaming="unknown",
                batching="unknown",
            ),
        )
    if isinstance(descriptor, ProviderDescriptor):
        resolved = descriptor
    elif isinstance(descriptor, Mapping):
        values = dict(descriptor)
        values.setdefault("name", name)
        resolved = ProviderDescriptor(**values)
    else:
        raise TypeError(
            "descriptor must be a ProviderDescriptor, mapping, or None"
        )
    if resolved.name != name:
        raise ValueError(
            "Provider descriptor name must match the registered name"
        )
    return resolved


def _registered_model_descriptors(
    provider: ProviderDescriptor,
    models: Sequence[ModelDescriptor | Mapping[str, object]],
) -> Tuple[ModelDescriptor, ...]:
    if isinstance(models, (str, bytes, Mapping)):
        raise TypeError("models must be a sequence of model descriptors")
    output: List[ModelDescriptor] = []
    for model in models:
        if isinstance(model, ModelDescriptor):
            resolved = model
        elif isinstance(model, Mapping):
            values = dict(model)
            values.setdefault("provider_id", provider.provider_id)
            resolved = ModelDescriptor(**values)
        else:
            raise TypeError(
                "models must contain ModelDescriptor records or mappings"
            )
        if resolved.provider_id != provider.provider_id:
            raise ValueError(
                "Model descriptor provider_id does not match provider"
            )
        output.append(resolved)
    identities = [model.model_id for model in output]
    if len(identities) != len(set(identities)):
        raise ValueError("models contain duplicate identities")
    return tuple(
        sorted(output, key=lambda model: (model.name, model.model_id or ""))
    )


@dataclass(frozen=True)
class _MultimodalProviderSpec:
    name: str
    aliases: Tuple[str, ...]
    description: str
    locality: str
    device: str
    authorization: str
    model_env: Tuple[str, ...] = ()
    default_model: Optional[str] = None


_BUILTIN_PROVIDER_SPECS: Tuple[_MultimodalProviderSpec, ...] = (
    _MultimodalProviderSpec(
        name="openrouter",
        aliases=(),
        description="OpenRouter OpenAI-compatible multimodal API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "IPFS_ACCELERATE_PY_OPENROUTER_MULTIMODAL_MODEL",
            "IPFS_ACCELERATE_PY_MULTIMODAL_MODEL",
        ),
        default_model="openai/gpt-4o",
    ),
    _MultimodalProviderSpec(
        name="openai",
        aliases=("gpt-4o", "gpt-4v", "gpt4o", "gpt4v"),
        description="OpenAI multimodal chat completions API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "IPFS_ACCELERATE_PY_OPENAI_MULTIMODAL_MODEL",
            "IPFS_ACCELERATE_PY_MULTIMODAL_MODEL",
        ),
        default_model="gpt-4o",
    ),
    _MultimodalProviderSpec(
        name="xai",
        aliases=("grok", "xai_grok"),
        description="xAI OpenAI-compatible multimodal API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_XAI_MULTIMODAL_MODEL",
            "ipfs_accelerate_py_MULTIMODAL_MODEL",
        ),
        default_model="grok-2-vision-1212",
    ),
    _MultimodalProviderSpec(
        name="meta_ai",
        aliases=(
            "meta",
            "meta-ai",
            "meta_llama",
            "meta_spark",
            "spark",
        ),
        description="Meta AI OpenAI-compatible multimodal API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL",
            "ipfs_accelerate_py_MULTIMODAL_MODEL",
        ),
        default_model="meta-llama/Llama-3.2-90B-Vision-Instruct",
    ),
    _MultimodalProviderSpec(
        name="huggingface",
        aliases=("hf", "local_hf"),
        description="Local Hugging Face transformers multimodal pipeline.",
        locality="local",
        device="cpu,cuda",
        authorization="none",
        model_env=("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL",),
        default_model="llava-hf/llava-1.5-7b-hf",
    ),
    _MultimodalProviderSpec(
        name="backend_manager",
        aliases=("accelerate",),
        description="Distributed inference backend manager multimodal provider.",
        locality="distributed",
        device="runtime-selected",
        authorization="unknown",
        model_env=("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL",),
    ),
)
_BUILTIN_PROVIDER_SPEC_BY_NAME = {
    spec.name: spec for spec in _BUILTIN_PROVIDER_SPECS
}
_BUILTIN_PROVIDER_ALIAS_TO_NAME = {
    alias: spec.name
    for spec in _BUILTIN_PROVIDER_SPECS
    for alias in spec.aliases
}


def _catalog_model_name(value: object) -> str:
    """Normalize an invocation identifier into the catalog name grammar."""

    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"[^a-z0-9._/-]+", "-", normalized)
    normalized = re.sub(r"/{2,}", "/", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    normalized = normalized.strip("._/-")
    if not normalized:
        normalized = "default"
    return normalized[:128].rstrip("._/-") or "default"


def _model_facts(model_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Return stable built-in facts only; unknown overrides stay unknown."""

    normalized = str(model_name or "").strip().casefold()
    if normalized in {"gpt-4o", "openai/gpt-4o"}:
        return 128_000, "multimodal-transformer"
    if normalized == "llava-hf/llava-1.5-7b-hf":
        return None, "llava"
    if normalized == "meta-llama/llama-3.2-90b-vision-instruct":
        return None, "llama-vision"
    return None, None


def _effective_spec_model(spec: _MultimodalProviderSpec) -> Optional[str]:
    return _coalesce_env(*spec.model_env) or spec.default_model


def _env_has_value(*names: str) -> bool:
    return bool(_coalesce_env(*names))


def _remote_provider_authorized(name: str) -> Optional[bool]:
    if name == "openrouter":
        return _env_has_value(
            "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        )
    if name == "openai":
        return _env_has_value(
            "IPFS_ACCELERATE_PY_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
    if name == "xai":
        return _env_has_value(
            "XAI_API_KEY",
            "ipfs_accelerate_py_XAI_API_KEY",
        )
    if name == "meta_ai":
        return _env_has_value(
            "META_AI_API_KEY",
            "ipfs_accelerate_py_META_AI_API_KEY",
        )
    return None


def _builtin_provider_state(
    spec: _MultimodalProviderSpec,
) -> Tuple[LifecycleState, OperationalState]:
    authorized = _remote_provider_authorized(spec.name)
    if authorized is not None:
        return (
            (
                LifecycleState.CONFIGURED
                if authorized
                else LifecycleState.DECLARED
            ),
            OperationalState(
                known=True,
                configured=authorized,
                authorized=authorized,
                reachable=None,
                healthy=None,
                routable=authorized,
            ),
        )
    if spec.authorization == "none":
        return (
            LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=None,
                authorized=True,
                reachable=None,
                healthy=None,
                routable=None,
            ),
        )
    if spec.name == "backend_manager":
        enabled = _truthy(
            os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")
        )
        return (
            LifecycleState.CONFIGURED
            if enabled
            else LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=enabled,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
        )
    return (
        LifecycleState.DECLARED,
        OperationalState(
            known=True,
            configured=None,
            authorized=None,
            reachable=None,
            healthy=None,
            routable=None,
        ),
    )


def _builtin_provider_descriptor(
    spec: _MultimodalProviderSpec,
) -> ProviderDescriptor:
    model_name = _effective_spec_model(spec)
    context_tokens, _ = _model_facts(model_name or "")
    lifecycle, state = _builtin_provider_state(spec)
    return ProviderDescriptor(
        name=spec.name,
        aliases=spec.aliases,
        description=spec.description,
        capabilities=(
            _multimodal_capability(max_context_tokens=context_tokens),
        ),
        lifecycle=lifecycle,
        state=state,
        provenance=_MULTIMODAL_CATALOG_PROVENANCE,
        labels=_default_multimodal_labels(
            authorization=spec.authorization,
            device=spec.device,
            locality=spec.locality,
        ),
    )


def _provider_descriptors_by_name() -> Dict[str, ProviderDescriptor]:
    descriptors = {
        spec.name: _builtin_provider_descriptor(spec)
        for spec in _BUILTIN_PROVIDER_SPECS
    }
    with _PROVIDER_REGISTRY_LOCK:
        registered = tuple(_PROVIDER_REGISTRY.values())
    for info in registered:
        # Dynamic registration has the same precedence as invocation.
        descriptors[info.name] = (
            info.descriptor
            or _registered_provider_descriptor(info.name, None)
        )
    return descriptors


def _canonical_provider_name(name: str) -> str:
    requested = str(name or "").strip().lower()
    if not requested:
        raise ValueError("Multimodal provider name must be non-empty")
    descriptors = _provider_descriptors_by_name()
    if requested in descriptors:
        return requested
    # Preserve the invocation surface's established built-in alias precedence.
    # A dynamic descriptor may publish the same alias, but generation has
    # historically routed that spelling through ``_builtin_provider_by_name``.
    builtin_name = _BUILTIN_PROVIDER_ALIAS_TO_NAME.get(requested)
    if builtin_name is not None and builtin_name in descriptors:
        return builtin_name
    matches = sorted(
        descriptor.name
        for descriptor in descriptors.values()
        if requested in descriptor.aliases
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous multimodal provider alias {name!r}: "
            f"{', '.join(matches)}"
        )
    raise ValueError(f"Unknown multimodal provider: {name}")


def get_provider_descriptor(name: str) -> ProviderDescriptor:
    """Return one provider descriptor by canonical name or invocation alias."""

    canonical = _canonical_provider_name(name)
    return _provider_descriptors_by_name()[canonical]


def _descriptor_operations(
    descriptor: ProviderDescriptor | ModelDescriptor,
) -> frozenset[Operation]:
    return frozenset(
        operation
        for capability in descriptor.capabilities
        for operation in capability.operations
    )


def _canonical_operation(
    operation: Optional[str | Operation],
) -> Optional[Operation]:
    if operation is None:
        return None
    if isinstance(operation, Operation):
        return operation
    try:
        return Operation(str(operation).strip().casefold())
    except ValueError as exc:
        raise ValueError(
            f"Unknown multimodal operation: {operation!r}"
        ) from exc


def _canonical_modality(
    modality: Optional[str | Modality],
    *,
    field_name: str,
) -> Optional[Modality]:
    if modality is None:
        return None
    if isinstance(modality, Modality):
        return modality
    try:
        return Modality(str(modality).strip().casefold())
    except ValueError as exc:
        raise ValueError(
            f"Unknown {field_name}: {modality!r}"
        ) from exc


def _media_type_matches(
    requested: str,
    declared: Sequence[str],
) -> bool:
    normalized = str(requested or "").strip().casefold()
    if "/" not in normalized:
        return False
    requested_family = normalized.split("/", 1)[0] + "/*"
    declared_set = {item.casefold() for item in declared}
    if normalized in declared_set or requested_family in declared_set:
        return True
    if normalized.endswith("/*"):
        prefix = normalized[:-1]
        return any(item.startswith(prefix) for item in declared_set)
    return False


def _matches_catalog_constraints(
    descriptor: ProviderDescriptor | ModelDescriptor,
    *,
    operation: Optional[Operation],
    input_modality: Optional[Modality],
    output_modality: Optional[Modality],
    media_type: Optional[str],
    image_input_mode: Optional[str],
    image_count: Optional[int],
    size_bytes: Optional[int],
    streaming: Optional[bool],
    batching: Optional[bool],
    locality: Optional[str],
    device: Optional[str],
    authorized: Optional[bool],
    ready: Optional[bool],
) -> bool:
    operations = _descriptor_operations(descriptor)
    if operation is not None and operation not in operations:
        return False
    if streaming is True and Operation.STREAM not in operations:
        return False
    if streaming is False and Operation.STREAM in operations:
        return False
    if batching is True and Operation.BATCH not in operations:
        return False
    if batching is False and Operation.BATCH in operations:
        return False

    capabilities = (
        descriptor.capabilities
        if operation is None
        else tuple(
            capability
            for capability in descriptor.capabilities
            if operation in capability.operations
        )
    )
    if input_modality is not None and not any(
        input_modality in capability.input_modalities
        for capability in capabilities
    ):
        return False
    if output_modality is not None and not any(
        output_modality in capability.output_modalities
        for capability in capabilities
    ):
        return False
    if media_type is not None:
        declared_media = tuple(
            item
            for capability in capabilities
            for item in capability.media_types
        )
        if declared_media and not _media_type_matches(
            media_type,
            declared_media,
        ):
            return False

    labels = dict(descriptor.labels)
    if image_input_mode is not None:
        requested_mode = str(image_input_mode).strip().casefold()
        if requested_mode not in {"inline", "uri"}:
            raise ValueError(
                "image_input_mode must be 'inline' or 'uri'"
            )
        known_modes = {
            item.strip().casefold()
            for item in labels.get("image_input_modes", "").split(",")
            if item.strip()
        }
        if (
            known_modes
            and "unknown" not in known_modes
            and requested_mode not in known_modes
        ):
            return False
    if image_count is not None:
        if (
            isinstance(image_count, bool)
            or not isinstance(image_count, int)
            or image_count < 0
        ):
            raise ValueError("image_count must be a non-negative integer")
        maximum = labels.get("max_images")
        if maximum and maximum.isdigit() and image_count > int(maximum):
            return False
    if size_bytes is not None:
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError("size_bytes must be a non-negative integer")
        known_limits = [
            capability.max_input_bytes
            for capability in capabilities
            if capability.max_input_bytes is not None
        ]
        if known_limits and size_bytes > max(known_limits):
            return False
    if locality is not None:
        actual = labels.get("locality")
        requested = str(locality).strip().casefold()
        if actual not in (None, "unknown") and actual != requested:
            return False
    if device is not None:
        actual_devices = {
            item.strip().casefold()
            for item in labels.get("device", "").split(",")
            if item.strip()
        }
        requested = str(device).strip().casefold()
        open_devices = {
            "unknown",
            "provider-managed",
            "runtime-selected",
        }
        if (
            actual_devices
            and not actual_devices.intersection(open_devices)
            and requested not in actual_devices
        ):
            return False
    if authorized is not None and descriptor.state.authorized is not authorized:
        return False
    if ready is not None and descriptor.state.routable is not ready:
        return False
    return True


def list_providers(
    *,
    operation: Optional[str | Operation] = None,
    input_modality: Optional[str | Modality] = None,
    output_modality: Optional[str | Modality] = None,
    media_type: Optional[str] = None,
    image_input_mode: Optional[str] = None,
    image_count: Optional[int] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
) -> List[ProviderDescriptor]:
    """List compatible descriptors without resolving a runtime provider."""

    selected_operation = _canonical_operation(operation)
    selected_input = _canonical_modality(
        input_modality,
        field_name="input modality",
    )
    selected_output = _canonical_modality(
        output_modality,
        field_name="output modality",
    )
    descriptors = [
        descriptor
        for _, descriptor in sorted(_provider_descriptors_by_name().items())
    ]
    return [
        descriptor
        for descriptor in descriptors
        if _matches_catalog_constraints(
            descriptor,
            operation=selected_operation,
            input_modality=selected_input,
            output_modality=selected_output,
            media_type=media_type,
            image_input_mode=image_input_mode,
            image_count=image_count,
            size_bytes=size_bytes,
            streaming=streaming,
            batching=batching,
            locality=locality,
            device=device,
            authorized=authorized,
            ready=ready,
        )
    ]


def _model_descriptor(
    provider: ProviderDescriptor,
    model_name: str,
) -> ModelDescriptor:
    context_tokens, architecture = _model_facts(model_name)
    with _PROVIDER_REGISTRY_LOCK:
        dynamically_registered = provider.name in _PROVIDER_REGISTRY
    if dynamically_registered:
        capabilities = provider.capabilities or (_multimodal_capability(),)
        provenance = _MULTIMODAL_REGISTRY_PROVENANCE
    else:
        capabilities = (
            _multimodal_capability(max_context_tokens=context_tokens),
        )
        provenance = _MULTIMODAL_CATALOG_PROVENANCE
    return ModelDescriptor(
        provider_id=provider.provider_id,
        name=_catalog_model_name(model_name),
        architecture=architecture,
        capabilities=capabilities,
        lifecycle=provider.lifecycle,
        state=provider.state,
        provenance=provenance,
        labels={
            **dict(provider.labels),
            "invocation_model": model_name,
        },
    )


def _models_for_provider(
    provider_name: str,
) -> Tuple[ModelDescriptor, ...]:
    descriptors = _provider_descriptors_by_name()
    provider = descriptors[provider_name]
    with _PROVIDER_REGISTRY_LOCK:
        registered = _PROVIDER_REGISTRY.get(provider_name)
    if registered is not None:
        return registered.models
    spec = _BUILTIN_PROVIDER_SPEC_BY_NAME[provider_name]
    model_name = _effective_spec_model(spec)
    if not model_name:
        return ()
    return (_model_descriptor(provider, model_name),)


def list_models(
    provider: Optional[str] = None,
    *,
    operation: Optional[str | Operation] = None,
    input_modality: Optional[str | Modality] = None,
    output_modality: Optional[str | Modality] = None,
    media_type: Optional[str] = None,
    image_input_mode: Optional[str] = None,
    image_count: Optional[int] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
) -> List[ModelDescriptor]:
    """List statically known and dynamically registered model hints."""

    selected_operation = _canonical_operation(operation)
    selected_input = _canonical_modality(
        input_modality,
        field_name="input modality",
    )
    selected_output = _canonical_modality(
        output_modality,
        field_name="output modality",
    )
    provider_descriptors = _provider_descriptors_by_name()
    provider_names = (
        (_canonical_provider_name(provider),)
        if provider is not None
        else tuple(sorted(provider_descriptors))
    )
    models = [
        model
        for provider_name in provider_names
        if _matches_catalog_constraints(
            provider_descriptors[provider_name],
            operation=selected_operation,
            input_modality=selected_input,
            output_modality=selected_output,
            media_type=media_type,
            image_input_mode=image_input_mode,
            image_count=image_count,
            size_bytes=size_bytes,
            streaming=streaming,
            batching=batching,
            locality=locality,
            device=device,
            authorized=authorized,
            ready=ready,
        )
        for model in _models_for_provider(provider_name)
    ]
    return sorted(
        (
            model
            for model in models
            if _matches_catalog_constraints(
                model,
                operation=selected_operation,
                input_modality=selected_input,
                output_modality=selected_output,
                media_type=media_type,
                image_input_mode=image_input_mode,
                image_count=image_count,
                size_bytes=size_bytes,
                streaming=streaming,
                batching=batching,
                locality=locality,
                device=device,
                authorized=authorized,
                ready=ready,
            )
        ),
        key=lambda model: (
            model.provider_id,
            model.name,
            model.model_id or "",
        ),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _select_discovery_provider(
    provider: Optional[str],
    *,
    deps: Optional[RouterDeps],
) -> str:
    """Mirror runtime selection using only already-known process state."""

    if provider:
        return _canonical_provider_name(provider)

    preferred = os.getenv(
        "IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER",
        "",
    ).strip()
    if preferred:
        try:
            return _canonical_provider_name(preferred)
        except ValueError:
            # Runtime historically ignores an unknown environment override and
            # continues through automatic providers.
            pass

    resolved_deps = deps or get_default_router_deps()
    if (
        _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"))
        and getattr(resolved_deps, "backend_manager", None) is not None
    ):
        return "backend_manager"
    for name in ("openrouter", "xai", "meta_ai", "openai"):
        if _remote_provider_authorized(name):
            return name
    if _module_available("transformers"):
        return "huggingface"
    raise RuntimeError(
        "No multimodal provider is statically resolvable for the requested "
        "constraints"
    )


def resolve_model(
    model_name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    operation: Optional[str | Operation] = Operation.VISION_GENERATE,
    input_modality: Optional[str | Modality] = None,
    output_modality: Optional[str | Modality] = None,
    media_type: Optional[str] = None,
    image_input_mode: Optional[str] = None,
    image_count: Optional[int] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
    deps: Optional[RouterDeps] = None,
) -> ModelDescriptor:
    """Resolve an invocation-compatible provider/model without side effects."""

    if model is not None:
        if model_name is not None and str(model_name) != str(model):
            raise ValueError("model and model_name specify different values")
        model_name = str(model)
    selected_operation = _canonical_operation(operation)
    if selected_operation not in {
        Operation.TEXT_GENERATE,
        Operation.VISION_GENERATE,
    }:
        value = (
            selected_operation.value
            if selected_operation is not None
            else None
        )
        raise ValueError(
            f"Multimodal router does not support operation {value!r}"
        )
    selected_input = _canonical_modality(
        input_modality,
        field_name="input modality",
    )
    selected_output = _canonical_modality(
        output_modality,
        field_name="output modality",
    )
    provider_name = _select_discovery_provider(provider, deps=deps)
    provider_descriptor = get_provider_descriptor(provider_name)
    if not _matches_catalog_constraints(
        provider_descriptor,
        operation=selected_operation,
        input_modality=selected_input,
        output_modality=selected_output,
        media_type=media_type,
        image_input_mode=image_input_mode,
        image_count=image_count,
        size_bytes=size_bytes,
        streaming=streaming,
        batching=batching,
        locality=locality,
        device=device,
        authorized=authorized,
        ready=ready,
    ):
        raise ValueError(
            f"Multimodal provider {provider_name!r} is incompatible with "
            "the requested constraints"
        )

    known_models = _models_for_provider(provider_name)
    requested_model = str(model_name or "").strip()
    if not requested_model:
        if not known_models:
            raise ValueError(
                f"Multimodal provider {provider_name!r} has no known default "
                "model; specify model_name explicitly"
            )
        resolved_model = known_models[0]
    else:
        requested_key = requested_model.casefold()
        resolved_model = next(
            (
                descriptor
                for descriptor in known_models
                if requested_key
                in {
                    descriptor.name.casefold(),
                    str(
                        dict(descriptor.labels).get(
                            "invocation_model",
                            descriptor.name,
                        )
                    ).casefold(),
                    *(alias.casefold() for alias in descriptor.aliases),
                }
            ),
            None,
        )
        if resolved_model is None:
            resolved_model = _model_descriptor(
                provider_descriptor,
                requested_model,
            )

    if not _matches_catalog_constraints(
        resolved_model,
        operation=selected_operation,
        input_modality=selected_input,
        output_modality=selected_output,
        media_type=media_type,
        image_input_mode=image_input_mode,
        image_count=image_count,
        size_bytes=size_bytes,
        streaming=streaming,
        batching=batching,
        locality=locality,
        device=device,
        authorized=authorized,
        ready=ready,
    ):
        raise ValueError(
            f"Multimodal model {resolved_model.name!r} is incompatible with "
            "the requested constraints"
        )
    return resolved_model


def get_catalog_snapshot() -> CatalogSnapshot:
    """Project current router metadata into a deterministic catalog snapshot."""

    providers = tuple(list_providers())
    models = tuple(list_models())
    bindings = tuple(
        RouterBinding(
            router="multimodal_router",
            provider_id=model.provider_id,
            model_id=model.model_id,
            operations=tuple(
                sorted(
                    {
                        operation
                        for capability in model.capabilities
                        for operation in capability.operations
                    },
                    key=lambda operation: operation.value,
                )
            ),
            priority=index,
            state=model.state,
            provenance=_MULTIMODAL_CATALOG_PROVENANCE,
            labels={
                "invocation_model": dict(model.labels).get(
                    "invocation_model",
                    model.name,
                )
            },
        )
        for index, model in enumerate(models)
    )
    return CatalogSnapshot(
        providers=providers,
        models=models,
        bindings=bindings,
    )


def catalog_snapshot() -> CatalogSnapshot:
    """Compatibility alias for catalog source adapters."""

    return get_catalog_snapshot()


def _encode_image_for_api(image: Union[str, bytes]) -> tuple[str, str]:
    """Return (url_or_base64_data_uri, media_type) for an image.

    If *image* is a URL string, return it as-is.
    If *image* is bytes or a local file path, base64-encode it.
    """
    if isinstance(image, str):
        stripped = image.strip()
        if stripped.startswith(("http://", "https://", "data:")):
            return stripped, "url"
        # Local file path
        try:
            with open(stripped, "rb") as fh:
                raw = fh.read()
        except Exception:
            return stripped, "url"
        ext = os.path.splitext(stripped)[-1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}", "base64"

    # bytes
    b64 = base64.b64encode(image).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", "base64"


def _get_openrouter_provider() -> Optional[MultimodalProvider]:
    """Get OpenRouter multimodal provider."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE")

    class _OpenRouterMultimodalProvider:
        def generate(
            self,
            prompt: str,
            *,
            image: Optional[Union[str, bytes]] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_MULTIMODAL_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL")
                or "openai/gpt-4o"
            )

            content: list = []
            if image is not None:
                img_src, kind = _encode_image_for_api(image)
                if kind == "url":
                    content.append({"type": "image_url", "image_url": {"url": img_src}})
                else:
                    content.append({"type": "image_url", "image_url": {"url": img_src}})
            content.append({"type": "text", "text": str(prompt)})

            messages = [{"role": "user", "content": content}]
            payload: Dict[str, object] = {
                "model": model,
                "messages": messages,
            }
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]

            headers = {
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if referer:
                headers["HTTP-Referer"] = referer
            if app_title:
                headers["X-Title"] = app_title

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers,
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("OpenRouter returned invalid JSON") from exc

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("OpenRouter multimodal response missing choices")
            return str(choices[0].get("message", {}).get("content", "") or "")

    return _OpenRouterMultimodalProvider()


def _get_openai_provider() -> Optional[MultimodalProvider]:
    """Get OpenAI vision provider."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    class _OpenAIMultimodalProvider:
        def generate(
            self,
            prompt: str,
            *,
            image: Optional[Union[str, bytes]] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENAI_MULTIMODAL_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL")
                or "gpt-4o"
            )

            content: list = []
            if image is not None:
                img_src, _ = _encode_image_for_api(image)
                content.append({"type": "image_url", "image_url": {"url": img_src}})
            content.append({"type": "text", "text": str(prompt)})

            messages = [{"role": "user", "content": content}]
            payload: Dict[str, object] = {
                "model": model,
                "messages": messages,
            }
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("OpenAI returned invalid JSON") from exc

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("OpenAI multimodal response missing choices")
            return str(choices[0].get("message", {}).get("content", "") or "")

    return _OpenAIMultimodalProvider()


def _get_xai_multimodal_provider() -> Optional[MultimodalProvider]:
    """Get xAI Grok vision provider (grok-2-vision-1212) via OpenAI-compatible endpoint."""
    api_key = (
        os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("ipfs_accelerate_py_XAI_API_KEY", "").strip()
    )
    if not api_key:
        return None

    base_url = os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")

    class _XAIMultimodalProvider:
        def generate(
            self,
            prompt: str,
            *,
            image: Optional[Union[str, bytes]] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_XAI_MULTIMODAL_MODEL")
                or os.getenv("ipfs_accelerate_py_MULTIMODAL_MODEL")
                or "grok-2-vision-1212"
            )

            content: list = []
            if image is not None:
                img_src, _ = _encode_image_for_api(image)
                content.append({"type": "image_url", "image_url": {"url": img_src}})
            content.append({"type": "text", "text": str(prompt)})

            messages = [{"role": "user", "content": content}]
            payload: Dict[str, object] = {
                "model": model,
                "messages": messages,
            }
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"xAI HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"xAI request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("xAI returned invalid JSON") from exc

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("xAI multimodal response missing choices")
            return str(choices[0].get("message", {}).get("content", "") or "")

    return _XAIMultimodalProvider()


def _get_meta_ai_multimodal_provider() -> Optional[MultimodalProvider]:
    """Get Meta AI vision provider (Llama-3.2-90B-Vision-Instruct) via OpenAI-compatible endpoint."""
    api_key = (
        os.environ.get("META_AI_API_KEY", "").strip()
        or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY", "").strip()
    )
    if not api_key:
        return None

    base_url = os.getenv("ipfs_accelerate_py_META_AI_BASE_URL", "https://api.llamameta.net/v1").rstrip("/")

    class _MetaAIMultimodalProvider:
        def generate(
            self,
            prompt: str,
            *,
            image: Optional[Union[str, bytes]] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL")
                or os.getenv("ipfs_accelerate_py_MULTIMODAL_MODEL")
                or "meta-llama/Llama-3.2-90B-Vision-Instruct"
            )

            content: list = []
            if image is not None:
                img_src, _ = _encode_image_for_api(image)
                content.append({"type": "image_url", "image_url": {"url": img_src}})
            content.append({"type": "text", "text": str(prompt)})

            messages = [{"role": "user", "content": content}]
            payload: Dict[str, object] = {
                "model": model,
                "messages": messages,
            }
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"Meta AI HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"Meta AI request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("Meta AI returned invalid JSON") from exc

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("Meta AI multimodal response missing choices")
            return str(choices[0].get("message", {}).get("content", "") or "")

    return _MetaAIMultimodalProvider()


def _get_huggingface_provider() -> Optional[MultimodalProvider]:
    """Get HuggingFace multimodal provider using transformers."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None

    class _HuggingFaceMultimodalProvider:
        def __init__(self):
            self._pipelines: Dict[str, object] = {}

        def _load_image(self, image: Union[str, bytes]):
            from PIL import Image as PILImage
            import io

            if isinstance(image, bytes):
                return PILImage.open(io.BytesIO(image)).convert("RGB")
            stripped = str(image).strip()
            if stripped.startswith(("http://", "https://")):
                import urllib.request as _ur
                with _ur.urlopen(stripped, timeout=30) as resp:
                    data = resp.read()
                return PILImage.open(io.BytesIO(data)).convert("RGB")
            if stripped.startswith("data:"):
                # data URI
                header, b64data = stripped.split(",", 1)
                raw = base64.b64decode(b64data)
                return PILImage.open(io.BytesIO(raw)).convert("RGB")
            return PILImage.open(stripped).convert("RGB")

        def generate(
            self,
            prompt: str,
            *,
            image: Optional[Union[str, bytes]] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            from transformers import pipeline as hf_pipeline

            model = model_name or os.getenv(
                "IPFS_ACCELERATE_PY_MULTIMODAL_MODEL", "llava-hf/llava-1.5-7b-hf"
            )
            device_str = device or os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_DEVICE", "cpu")

            cache_key = f"{model}::{device_str}"
            if cache_key not in self._pipelines:
                try:
                    import torch

                    task = "image-to-text" if image is not None else "text-generation"
                    pipe = hf_pipeline(
                        task,
                        model=model,
                        device=0 if (device_str == "cuda" and torch.cuda.is_available()) else -1,
                    )
                    self._pipelines[cache_key] = pipe
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load HuggingFace multimodal model '{model}': {exc}"
                    ) from exc

            pipe = self._pipelines[cache_key]
            max_new_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", 256)))

            if image is not None:
                pil_image = self._load_image(image)
                result = pipe(pil_image, prompt=str(prompt), max_new_tokens=max_new_tokens)
                if isinstance(result, list) and result:
                    first = result[0]
                    if isinstance(first, dict):
                        return str(first.get("generated_text", first.get("text", "")) or "")
                    return str(first)
                return str(result)

            # Text-only fallback
            result = pipe(str(prompt), max_new_tokens=max_new_tokens)
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    return str(first.get("generated_text", first.get("text", "")) or "")
                return str(first)
            return str(result)

    return _HuggingFaceMultimodalProvider()


def _get_backend_manager_provider(deps: RouterDeps) -> Optional[MultimodalProvider]:
    """Get provider that uses InferenceBackendManager for distributed/multiplexed inference."""
    if not _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")):
        return None

    try:
        manager = deps.get_backend_manager(
            purpose="multimodal_router",
            enable_health_checks=True,
            load_balancing_strategy=os.getenv(
                "IPFS_ACCELERATE_PY_MULTIMODAL_LOAD_BALANCING", "round_robin"
            ),
        )
        if manager is None:
            return None

        class _BackendManagerMultimodalProvider:
            def generate(
                self,
                prompt: str,
                *,
                image: Optional[Union[str, bytes]] = None,
                model_name: Optional[str] = None,
                device: Optional[str] = None,
                **kwargs: object,
            ) -> str:
                backend = manager.select_backend_for_task(
                    task="multimodal",
                    model=model_name or os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL", ""),
                    protocol="any",
                )

                if backend is None:
                    raise RuntimeError("No available backend for multimodal task")

                payload: Dict[str, object] = {"prompt": str(prompt), "device": device, **kwargs}
                if image is not None:
                    if isinstance(image, bytes):
                        payload["image_b64"] = base64.b64encode(image).decode("ascii")
                    else:
                        payload["image"] = str(image)

                result = manager.execute_inference(
                    backend_id=backend["id"],
                    task="multimodal",
                    payload=payload,
                )

                text = result.get("text") or result.get("generated_text", "")
                return str(text)

        return _BackendManagerMultimodalProvider()
    except Exception as exc:
        logger.debug(f"Backend manager provider unavailable: {exc}")
        return None


def _provider_cache_key() -> tuple:
    return (
        os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "").strip(),
        os.getenv("XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_MULTIMODAL_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "").strip(),
        os.getenv("META_AI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_DEVICE", "").strip(),
    )


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[MultimodalProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key == "openrouter":
        return _get_openrouter_provider()
    if key in {"openai", "gpt4v", "gpt-4v", "gpt4o", "gpt-4o"}:
        return _get_openai_provider()
    if key in {"xai", "grok", "xai_grok"}:
        return _get_xai_multimodal_provider()
    if key in {"meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"}:
        return _get_meta_ai_multimodal_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_huggingface_provider()
    if key in {"backend_manager", "accelerate"}:
        return _get_backend_manager_provider(deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> MultimodalProvider:
    if preferred:
        preferred_key = str(preferred).strip().lower()
        try:
            preferred_key = _canonical_provider_name(preferred_key)
        except ValueError:
            pass
        with _PROVIDER_REGISTRY_LOCK:
            info = _PROVIDER_REGISTRY.get(preferred_key)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_key, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown multimodal provider: {preferred}")

    preferred_env = os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER", "").strip()
    if preferred_env:
        preferred_key = preferred_env.lower()
        try:
            preferred_key = _canonical_provider_name(preferred_key)
        except ValueError:
            pass
        with _PROVIDER_REGISTRY_LOCK:
            info = _PROVIDER_REGISTRY.get(preferred_key)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_key, deps=deps)
        if builtin is not None:
            return builtin

    backend_manager_provider = _get_backend_manager_provider(deps)
    if backend_manager_provider is not None:
        return backend_manager_provider

    for name in ["openrouter", "xai", "meta_ai", "openai"]:
        candidate = _builtin_provider_by_name(name, deps=deps)
        if candidate is not None:
            return candidate

    hf_provider = _get_huggingface_provider()
    if hf_provider is not None:
        return hf_provider

    raise RuntimeError(
        "No multimodal provider available. "
        "Install `transformers` and `Pillow` for local inference, or configure an API key."
    )


@lru_cache(maxsize=32)
def _resolve_provider_cached(preferred: Optional[str], cache_key: tuple) -> MultimodalProvider:
    _ = cache_key
    return _resolve_provider_uncached(preferred, deps=get_default_router_deps())


def get_multimodal_provider(
    provider: Optional[str] = None,
    *,
    deps: Optional[RouterDeps] = None,
    use_cache: Optional[bool] = None,
) -> MultimodalProvider:
    """Resolve a multimodal provider with optional dependency injection."""
    resolved_deps = deps or get_default_router_deps()
    cache_ok = _cache_enabled() if use_cache is None else bool(use_cache)

    if not cache_ok:
        return _resolve_provider_uncached(provider, deps=resolved_deps)

    if deps is not None:
        cache_key = _provider_cache_key()
        deps_key = f"multimodal_provider::{(provider or '').strip().lower()}::{hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]}"
        cached = resolved_deps.get_cached(deps_key)
        if cached is not None:
            return cached
        return resolved_deps.set_cached(deps_key, _resolve_provider_uncached(provider, deps=resolved_deps))

    return _resolve_provider_cached(provider, _provider_cache_key())


def generate_multimodal(
    prompt: str,
    *,
    image: Optional[Union[str, bytes]] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[MultimodalProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> str:
    """Generate text from a prompt and optional image.

    Args:
        prompt: Text prompt or question
        image: Optional image — URL string, local file path, bytes, or data URI
        model_name: Optional model name to use
        device: Optional device (cpu/cuda)
        provider: Optional provider name
        provider_instance: Optional pre-created provider instance
        deps: Optional RouterDeps for dependency injection
        **kwargs: Additional arguments passed to the provider (max_tokens,
            temperature, etc.)

    Returns:
        Generated text string
    """
    resolved_deps = deps or get_default_router_deps()

    if _response_cache_enabled():
        cache_key = _response_cache_key(
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            image=image,
            kwargs=dict(kwargs),
        )
        try:
            getter = getattr(resolved_deps, "get_cached_or_remote", None)
            cached = getter(cache_key) if callable(getter) else resolved_deps.get_cached(cache_key)
            if isinstance(cached, str):
                return cached
        except Exception:
            pass

    backend = provider_instance or get_multimodal_provider(provider, deps=resolved_deps)
    try:
        result = backend.generate(
            prompt,
            image=image,
            model_name=model_name,
            device=device,
            **kwargs,
        )
        text = str(result)
        if _response_cache_enabled():
            try:
                cache_key = _response_cache_key(
                    provider=provider,
                    model_name=model_name,
                    prompt=prompt,
                    image=image,
                    kwargs=dict(kwargs),
                )
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(cache_key, text)
                else:
                    resolved_deps.set_cached(cache_key, text)
            except Exception:
                pass
        return text
    except Exception as primary_error:
        logger.debug(f"Primary multimodal provider failed: {primary_error}")
        if provider is None:
            hf_provider = _get_huggingface_provider()
            if hf_provider is not None and backend is not hf_provider:
                return hf_provider.generate(
                    prompt,
                    image=image,
                    model_name=model_name,
                    device=device,
                    **kwargs,
                )
        raise


def clear_multimodal_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""
    _resolve_provider_cached.cache_clear()


def _guess_mime_type(path: str | Path, mime_type: str | None = None) -> str:
    if mime_type and str(mime_type).strip():
        return str(mime_type).strip()
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def encode_image_as_data_url(
    image_path: str | Path,
    *,
    mime_type: str | None = None,
) -> str:
    """Encode a local image using the OpenAI-compatible data URL shape."""

    path = Path(image_path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_guess_mime_type(path, mime_type=mime_type)};base64,{encoded}"


def build_text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": str(text or "")}


def build_image_part(
    *,
    image_path: str | Path | None = None,
    image_url: str | None = None,
    mime_type: str | None = None,
    detail: str | None = DEFAULT_IMAGE_DETAIL,
) -> dict[str, Any]:
    if image_url:
        url = str(image_url).strip()
    elif image_path is not None:
        url = encode_image_as_data_url(image_path, mime_type=mime_type)
    else:
        raise ValueError("build_image_part requires image_path or image_url")

    payload: dict[str, Any] = {"url": url}
    if detail and str(detail).strip():
        payload["detail"] = str(detail).strip()
    return {"type": "image_url", "image_url": payload}


def build_multimodal_messages(
    *,
    prompt: str,
    image_paths: Sequence[str | Path] | None = None,
    image_urls: Sequence[str] | None = None,
    system_prompt: str | None = None,
    additional_text_blocks: Sequence[str] | None = None,
    image_detail: str | None = DEFAULT_IMAGE_DETAIL,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal chat messages."""

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    content: list[dict[str, Any]] = [build_text_part(prompt)]
    for block in additional_text_blocks or ():
        if str(block or "").strip():
            content.append(build_text_part(str(block)))
    for path in image_paths or ():
        content.append(build_image_part(image_path=path, detail=image_detail))
    for url in image_urls or ():
        content.append(build_image_part(image_url=url, detail=image_detail))

    messages.append({"role": "user", "content": content})
    return messages


def _flatten_content_part(part: Any) -> str:
    if isinstance(part, dict):
        part_type = str(part.get("type") or "").strip().lower()
        if part_type == "text":
            return str(part.get("text") or "").strip()
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "").strip()
            else:
                url = str(image_url or "").strip()
            if not url:
                return ""
            return "[image attachment included]" if url.startswith("data:") else f"[image: {url}]"
    return str(part or "").strip()


def _flatten_messages_to_prompt(messages: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        content = message.get("content")
        if isinstance(content, list):
            rendered = "\n".join(
                filter(None, (_flatten_content_part(part) for part in content))
            )
        else:
            rendered = str(content or "").strip()
        lines.append(f"{role}: {rendered}")
    return "\n".join(lines).strip()


def generate_multimodal_text(
    prompt: str,
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[Any] = None,
    deps: Optional[RouterDeps] = None,
    image_paths: Sequence[str | Path] | None = None,
    image_urls: Sequence[str] | None = None,
    system_prompt: str | None = None,
    additional_text_blocks: Sequence[str] | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
    image_detail: str | None = DEFAULT_IMAGE_DETAIL,
    **kwargs: object,
) -> str:
    """Generate from one or more images through the canonical LLM router."""

    normalized_image_paths = [
        str(Path(path).expanduser()) for path in image_paths or ()
    ]
    normalized_image_urls = [str(url) for url in image_urls or ()]
    resolved_messages = (
        list(messages)
        if messages is not None
        else build_multimodal_messages(
            prompt=prompt,
            image_paths=normalized_image_paths,
            image_urls=normalized_image_urls,
            system_prompt=system_prompt,
            additional_text_blocks=additional_text_blocks,
            image_detail=image_detail,
        )
    )
    backend = provider_instance or llm_router.get_llm_provider(provider, deps=deps)

    if isinstance(backend, llm_router.OpenAIChatCompletionsProvider):
        response = llm_router.chat_completions_create(
            messages=resolved_messages,  # type: ignore[arg-type]
            model=model_name,
            provider=provider,
            provider_instance=backend,
            deps=deps,
            **kwargs,
        )
        return response.choices[0].message.content

    if isinstance(backend, llm_router.NativeMultimodalProvider):
        return backend.generate_multimodal(
            prompt,
            model_name=model_name,
            image_paths=normalized_image_paths,
            image_urls=normalized_image_urls,
            system_prompt=system_prompt,
            additional_text_blocks=[
                str(block) for block in additional_text_blocks or ()
            ],
            messages=resolved_messages,
            **kwargs,
        )

    return llm_router.generate_text(
        _flatten_messages_to_prompt(resolved_messages),
        model_name=model_name,
        provider=provider,
        provider_instance=backend,
        deps=deps,
        **kwargs,
    )


class MultimodalRouter:
    """Object-oriented compatibility facade over ``generate_multimodal_text``."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        deps: RouterDeps | None = None,
        **config: object,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.deps = deps
        self.config = dict(config)

    def generate(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        image_paths: Sequence[str | Path] | None = None,
        image_urls: Sequence[str] | None = None,
        system_prompt: str | None = None,
        additional_text_blocks: Sequence[str] | None = None,
        messages: Sequence[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> str:
        effective_config = dict(self.config)
        effective_config.update(kwargs)
        return generate_multimodal_text(
            prompt,
            model_name=model_name or self.model_name,
            provider=self.provider,
            deps=self.deps,
            image_paths=image_paths,
            image_urls=image_urls,
            system_prompt=system_prompt,
            additional_text_blocks=additional_text_blocks,
            messages=messages,
            **effective_config,
        )


generate_text = generate_multimodal_text
