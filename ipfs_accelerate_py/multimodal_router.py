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
- `meta_ai`: Meta Muse Spark multimodal API via OpenAI-compatible endpoint
    - encrypted credential `meta_ai_api_key`, `MODEL_API_KEY`,
      `META_AI_API_KEY`, or `ipfs_accelerate_py_META_AI_API_KEY`
    - `ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL` (default: muse-spark-1.1)
    - `ipfs_accelerate_py_META_AI_BASE_URL` (default: https://api.meta.ai/v1)
- `huggingface`: HuggingFace transformers (LLaVA, InstructBLIP, etc.)
- `backend_manager`: Use InferenceBackendManager for distributed inference
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import ipaddress
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.parse
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
from .common.meta_model_api import (
    META_MODEL_API_BASE_URL,
    META_MODEL_API_DEFAULT_MODEL,
    meta_model_api_key_fingerprint,
    normalize_meta_model_name,
    resolve_meta_model_api_key,
)
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


class MultimodalRouterError(RuntimeError):
    """Raised when a provider or media input violates the multimodal contract."""


class UsageCapacityError(MultimodalRouterError):
    """Raised when usage-aware admission denies capacity before or during dispatch."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
        next_eligible_at: Optional[str] = None,
        admission: Optional[object] = None,
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(reason_codes or ())
        self.next_eligible_at = next_eligible_at
        self.admission = admission


# Evidence identity for AICAT-G130 / AICAT-032 multimodal usage integration.
USAGE_ROUTING_REQUIREMENT_ID = "requirement:multimodal-router-usage-routing.v1"
MULTIMODAL_USAGE_OPERATION = "multimodal.generate"

# Default conservative ceilings applied only when callers/env request enforcement
# of size/MIME policy (never invent unlimited remote quotas).
_DEFAULT_MAX_MEDIA_BYTES = 20 * 1024 * 1024
_SSRF_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "0.0.0.0",
    }
)
_SSRF_BLOCKED_SCHEMES = frozenset(
    {
        "file",
        "ftp",
        "gopher",
        "dict",
        "sftp",
        "tftp",
        "jar",
        "ldap",
        "ldaps",
    }
)
_ALLOWED_IMAGE_MIME_PREFIXES = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/*",
)

_LAST_MULTIMODAL_TRACE = threading.local()
_LAST_USAGE_ADMISSION = threading.local()


def _set_last_multimodal_trace(**values: object) -> None:
    _LAST_MULTIMODAL_TRACE.payload = dict(values)


def get_last_multimodal_trace() -> Dict[str, object]:
    """Return a copy of the most recent multimodal-call trace for this thread."""

    payload = getattr(_LAST_MULTIMODAL_TRACE, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _set_last_usage_admission(payload: Optional[Mapping[str, object]]) -> None:
    _LAST_USAGE_ADMISSION.payload = dict(payload) if payload is not None else None


def get_last_usage_admission() -> Dict[str, object]:
    """Return a copy of the most recent usage-admission result for this thread.

    Operational evidence only: never prompts, media bytes, or generated text.
    """

    payload = getattr(_LAST_USAGE_ADMISSION, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


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
        description="Meta Muse Spark OpenAI-compatible multimodal API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        model_env=(
            "ipfs_accelerate_py_META_AI_MULTIMODAL_MODEL",
            "ipfs_accelerate_py_MULTIMODAL_MODEL",
        ),
        default_model=META_MODEL_API_DEFAULT_MODEL,
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
    if normalized == META_MODEL_API_DEFAULT_MODEL.casefold():
        return None, "muse-spark"
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
            "MODEL_API_KEY",
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
    """Get the Muse Spark multimodal provider."""

    api_key = resolve_meta_model_api_key()
    if not api_key:
        return None

    base_url = os.getenv(
        "ipfs_accelerate_py_META_AI_BASE_URL",
        META_MODEL_API_BASE_URL,
    ).rstrip("/")

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
                or META_MODEL_API_DEFAULT_MODEL
            )
            model = normalize_meta_model_name(model)

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
            max_completion_tokens = kwargs.get(
                "max_completion_tokens",
                kwargs.get("max_tokens", kwargs.get("max_new_tokens")),
            )
            if max_completion_tokens is not None:
                payload["max_completion_tokens"] = int(max_completion_tokens)
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
        meta_model_api_key_fingerprint(),
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


# ---------------------------------------------------------------------------
# Usage-aware admission (optional; off mode is the default legacy path)
# ---------------------------------------------------------------------------


def _provider_name(
    backend: Optional[object],
    *,
    requested: Optional[str] = None,
) -> str:
    if backend is not None:
        name = getattr(backend, "router_provider_name", None)
        if name:
            return str(name)
    return str(requested or "").strip()


def estimate_text_tokens(text: str) -> int:
    """Conservative text token estimate for multimodal admission."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return 1
    char_estimate = (len(text) + 3) // 4
    word_estimate = max(1, len(text.split()))
    byte_estimate = (len(text.encode("utf-8")) + 2) // 3
    return max(1, char_estimate, word_estimate, byte_estimate)


@dataclass(frozen=True)
class MediaReferenceFacts:
    """Bounded, non-payload facts about a media reference.

    The image bytes/URI themselves are never stored here and must never enter
    the ledger or routing receipt.
    """

    image_count: int
    media_bytes: int
    pixels: Optional[int]
    mime_type: Optional[str]
    input_mode: str  # none | inline | uri
    scheme: Optional[str]
    host_kind: Optional[str]
    local_only: bool


def _parse_data_uri_mime(value: str) -> Optional[str]:
    if not value.startswith("data:"):
        return None
    header = value[5:].split(",", 1)[0]
    mime = header.split(";", 1)[0].strip().casefold()
    return mime or None


def _host_is_blocked(host: str) -> bool:
    cleaned = (host or "").strip().casefold().rstrip(".")
    if not cleaned:
        return True
    if cleaned in _SSRF_BLOCKED_HOSTS or cleaned.endswith(".localhost"):
        return True
    if cleaned.endswith(".internal") or cleaned.endswith(".local"):
        return True
    # Strip IPv6 brackets.
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    try:
        ip = ipaddress.ip_address(cleaned)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def inspect_media_reference(
    image: Optional[Union[str, bytes]],
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    declared_media_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
) -> MediaReferenceFacts:
    """Derive reference-only media facts without retaining payload content."""

    if image is None:
        return MediaReferenceFacts(
            image_count=0,
            media_bytes=0,
            pixels=None,
            mime_type=mime_type,
            input_mode="none",
            scheme=None,
            host_kind=None,
            local_only=True,
        )

    pixels: Optional[int] = None
    if width is not None and height is not None:
        try:
            w = int(width)
            h = int(height)
            if w > 0 and h > 0:
                pixels = w * h
        except (TypeError, ValueError):
            pixels = None

    if isinstance(image, bytes):
        media_bytes = (
            int(declared_media_bytes)
            if declared_media_bytes is not None
            else len(image)
        )
        return MediaReferenceFacts(
            image_count=1,
            media_bytes=max(0, media_bytes),
            pixels=pixels,
            mime_type=(mime_type or "image/jpeg").casefold(),
            input_mode="inline",
            scheme=None,
            host_kind="inline-bytes",
            local_only=True,
        )

    if not isinstance(image, str):
        raise TypeError("image must be str, bytes, or None")

    stripped = image.strip()
    if stripped.startswith("data:"):
        mime = mime_type or _parse_data_uri_mime(stripped) or "image/jpeg"
        # Estimate payload size from base64 tail without decoding full content
        # into the ledger path.
        comma = stripped.find(",")
        payload = stripped[comma + 1 :] if comma >= 0 else ""
        # Rough decoded-byte estimate for base64.
        media_bytes = (
            int(declared_media_bytes)
            if declared_media_bytes is not None
            else max(0, (len(payload) * 3) // 4)
        )
        return MediaReferenceFacts(
            image_count=1,
            media_bytes=media_bytes,
            pixels=pixels,
            mime_type=str(mime).casefold(),
            input_mode="inline",
            scheme="data",
            host_kind="data-uri",
            local_only=True,
        )

    if stripped.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(stripped)
        scheme = (parsed.scheme or "").casefold()
        host = parsed.hostname or ""
        media_bytes = int(declared_media_bytes) if declared_media_bytes is not None else 0
        return MediaReferenceFacts(
            image_count=1,
            media_bytes=max(0, media_bytes),
            pixels=pixels,
            mime_type=(mime_type or "image/*").casefold(),
            input_mode="uri",
            scheme=scheme,
            host_kind="public" if host and not _host_is_blocked(host) else "blocked-or-private",
            local_only=False,
        )

    # Local path or opaque reference: treat as inline/local-only.
    media_bytes = 0
    if declared_media_bytes is not None:
        media_bytes = int(declared_media_bytes)
    else:
        try:
            media_bytes = max(0, int(os.path.getsize(stripped)))
        except OSError:
            media_bytes = max(0, len(stripped.encode("utf-8")))
    guessed = mimetypes.guess_type(stripped)[0]
    return MediaReferenceFacts(
        image_count=1,
        media_bytes=media_bytes,
        pixels=pixels,
        mime_type=(mime_type or guessed or "image/*").casefold(),
        input_mode="inline",
        scheme="file-path",
        host_kind="local-path",
        local_only=True,
    )


def validate_multimodal_media_input(
    image: Optional[Union[str, bytes]],
    *,
    max_media_bytes: Optional[int] = None,
    allowed_mime_prefixes: Sequence[str] = _ALLOWED_IMAGE_MIME_PREFIXES,
    allow_remote_uri: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    declared_media_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
) -> MediaReferenceFacts:
    """Fail closed on adversarial size/MIME/SSRF-shaped media before reserve.

    Media payloads are inspected only for bounds; the reference itself is never
    copied into usage state.
    """

    if image is None:
        return inspect_media_reference(None)

    if isinstance(image, str):
        stripped = image.strip()
        lower = stripped.casefold()
        # SSRF-shaped schemes fail before any reservation or provider call.
        for scheme in _SSRF_BLOCKED_SCHEMES:
            if lower.startswith(f"{scheme}:"):
                raise MultimodalRouterError(
                    f"media URI scheme {scheme!r} is not permitted"
                )
        if lower.startswith(("http://", "https://")):
            if not allow_remote_uri:
                raise MultimodalRouterError(
                    "remote media URIs are not permitted by current policy"
                )
            parsed = urllib.parse.urlparse(stripped)
            host = parsed.hostname or ""
            if not host or _host_is_blocked(host):
                raise MultimodalRouterError(
                    "media URI host is blocked by SSRF policy"
                )
            # Reject userinfo (credential-shaped) and non-http(s) after normalize.
            if parsed.username or parsed.password:
                raise MultimodalRouterError(
                    "media URI must not embed credentials"
                )
        elif lower.startswith("data:"):
            mime = (mime_type or _parse_data_uri_mime(stripped) or "").casefold()
            if mime and not any(
                mime == prefix.casefold() or mime.startswith(prefix.rstrip("*").casefold())
                for prefix in allowed_mime_prefixes
            ):
                # image/* is allowed via prefix match on "image/"
                if not mime.startswith("image/"):
                    raise MultimodalRouterError(
                        f"media MIME type {mime!r} is not permitted"
                    )

    facts = inspect_media_reference(
        image,
        width=width,
        height=height,
        declared_media_bytes=declared_media_bytes,
        mime_type=mime_type,
    )
    if facts.mime_type and facts.mime_type not in {"image/*", "application/octet-stream"}:
        mime = facts.mime_type
        if not any(
            mime == prefix.casefold()
            or mime.startswith(prefix.rstrip("*").casefold())
            for prefix in allowed_mime_prefixes
        ):
            if not mime.startswith("image/"):
                raise MultimodalRouterError(
                    f"media MIME type {mime!r} is not permitted"
                )

    ceiling = max_media_bytes
    if ceiling is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_MULTIMODAL_MAX_MEDIA_BYTES",
            "IPFS_DATASETS_PY_MULTIMODAL_MAX_MEDIA_BYTES",
        )
        if raw:
            try:
                ceiling = int(raw)
            except (TypeError, ValueError):
                ceiling = None
    if ceiling is not None and facts.media_bytes > int(ceiling):
        raise MultimodalRouterError(
            f"media exceeds max_media_bytes bound ({facts.media_bytes} > {ceiling})"
        )
    if facts.image_count > 1:
        raise MultimodalRouterError("image_count exceeds multimodal router bound")
    return facts


def estimate_multimodal_usage(
    prompt: str,
    *,
    image: Optional[Union[str, bytes]] = None,
    max_output_tokens: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    declared_media_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
    include_concurrency: bool = True,
    remote: bool = True,
    media_facts: Optional[MediaReferenceFacts] = None,
) -> "object":
    """Build a conservative multi-dimension usage vector for multimodal work.

    Dimensions covered when applicable: requests, images, pixels, media_bytes,
    input_tokens, output_tokens, concurrent_requests, and cost_micros.
    Media content is referenced only via scalar facts — never embedded.
    """

    from .endpoint_usage.schema import UsageVector

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not remote:
        return UsageVector()

    facts = media_facts or inspect_media_reference(
        image,
        width=width,
        height=height,
        declared_media_bytes=declared_media_bytes,
        mime_type=mime_type,
    )
    input_tokens = estimate_text_tokens(prompt)
    # Vision models typically bill additional image tokens; keep a floor when
    # an image is present so headroom cannot be under-reserved.
    if facts.image_count > 0:
        image_token_floor = 85
        if facts.pixels is not None:
            image_token_floor = max(image_token_floor, (int(facts.pixels) + 767) // 768)
        input_tokens += image_token_floor

    output_tokens = 1
    if max_output_tokens is not None:
        try:
            output_tokens = max(1, int(max_output_tokens))
        except (TypeError, ValueError):
            output_tokens = 1

    amounts: Dict[str, int] = {
        "requests": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if facts.image_count > 0:
        amounts["images"] = facts.image_count
    if facts.pixels is not None and facts.pixels > 0:
        amounts["pixels"] = int(facts.pixels)
    if facts.media_bytes > 0:
        amounts["media_bytes"] = int(facts.media_bytes)
    if include_concurrency:
        amounts["concurrent_requests"] = 1
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    return UsageVector.of(**amounts)


def settle_multimodal_usage(
    prompt: str,
    *,
    image: Optional[Union[str, bytes]] = None,
    output_text: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    declared_media_bytes: Optional[int] = None,
    mime_type: Optional[str] = None,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
    media_facts: Optional[MediaReferenceFacts] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> "object":
    """Actual remote usage for a completed multimodal invocation."""

    from .endpoint_usage.schema import UsageVector

    facts = media_facts or inspect_media_reference(
        image,
        width=width,
        height=height,
        declared_media_bytes=declared_media_bytes,
        mime_type=mime_type,
    )
    estimated = estimate_multimodal_usage(
        prompt,
        image=image,
        max_output_tokens=max_output_tokens,
        media_facts=facts,
        remote=True,
    )
    amounts: Dict[str, int] = {}
    for entry in getattr(estimated, "entries", ()) or ():
        name = str(getattr(entry.dimension, "value", entry.dimension) or "")
        value = getattr(entry.amount, "value", None)
        if value is not None:
            amounts[name] = int(value)
    if input_tokens is not None:
        amounts["input_tokens"] = max(0, int(input_tokens))
    if output_tokens is not None:
        amounts["output_tokens"] = max(0, int(output_tokens))
    elif output_text is not None:
        amounts["output_tokens"] = estimate_text_tokens(str(output_text))
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    if not amounts:
        return UsageVector()
    return UsageVector.of(**amounts)


# Ranking-input names that embed these substrings are rejected by receipt
# digests. Still reserve the full vector; only the planning ``required``
# surface is filtered for receipt safety.
_RECEIPT_UNSAFE_DIMENSION_MARKERS = (
    "token",
    "media",
    "prompt",
    "message",
    "payload",
    "endpoint",
    "credential",
    "secret",
    "password",
    "authorization",
)


def planning_required_usage(requested: "object") -> "object":
    """Return a receipt-safe planning vector derived from a full estimate."""

    from .endpoint_usage.schema import UsageVector

    if not isinstance(requested, UsageVector):
        return UsageVector()
    safe: List[object] = []
    for entry in requested.entries:
        name = str(getattr(entry.dimension, "value", entry.dimension) or "")
        lowered = name.casefold()
        if any(marker in lowered for marker in _RECEIPT_UNSAFE_DIMENSION_MARKERS):
            continue
        safe.append(entry)
    return UsageVector(entries=tuple(safe))  # type: ignore[arg-type]


def _normalize_usage_policy(policy: object) -> "object":
    from .endpoint_usage.schema import RoutingMode, RoutingPolicy

    if policy is None:
        return RoutingPolicy(mode=RoutingMode.OFF)
    if isinstance(policy, RoutingPolicy):
        return policy
    if isinstance(policy, Mapping):
        return RoutingPolicy.from_dict(policy)
    raise TypeError("usage_policy must be a RoutingPolicy, mapping, or None")


def _usage_mode_is_off(policy: object, coordinator: object) -> bool:
    from .endpoint_usage.schema import RoutingMode

    if coordinator is None:
        return True
    mode = getattr(policy, "mode", RoutingMode.OFF)
    return mode is RoutingMode.OFF or str(mode) == RoutingMode.OFF.value


def _usage_mode_observes_only(policy: object) -> bool:
    from .endpoint_usage.schema import RoutingMode

    mode = getattr(policy, "mode", RoutingMode.OFF)
    return mode in (RoutingMode.OBSERVE, RoutingMode.SHADOW) or str(mode) in {
        RoutingMode.OBSERVE.value,
        RoutingMode.SHADOW.value,
    }


def _usage_mode_enforces(policy: object) -> bool:
    from .endpoint_usage.schema import RoutingMode

    mode = getattr(policy, "mode", RoutingMode.OFF)
    return mode in (RoutingMode.ENFORCE, RoutingMode.ASSIST) or str(mode) in {
        RoutingMode.ENFORCE.value,
        RoutingMode.ASSIST.value,
    }


def _multimodal_compatibility_labels(
    *,
    provider_name: str,
    model_name: Optional[str],
    device: Optional[str],
    media_facts: MediaReferenceFacts,
    kwargs: Mapping[str, object],
    operation: str = MULTIMODAL_USAGE_OPERATION,
) -> Dict[str, str]:
    labels: Dict[str, str] = {
        "router_provider": str(provider_name or ""),
        "operation": str(operation or MULTIMODAL_USAGE_OPERATION),
        "output_media_types": "text/plain",
        "max_images": "1",
    }
    try:
        descriptor = get_provider_descriptor(provider_name) if provider_name else None
    except Exception:
        descriptor = None
    if descriptor is not None:
        for key in (
            "locality",
            "device",
            "access_requirement",
            "input_media_types",
            "output_media_types",
            "image_input_modes",
            "max_images",
            "uri_schemes",
            "data.governance",
            "data_governance",
        ):
            value = dict(descriptor.labels or {}).get(key)
            if value is not None:
                labels[key] = str(value)
    if media_facts.mime_type:
        labels["mime_family"] = (
            "image/*"
            if media_facts.mime_type.startswith("image/")
            else media_facts.mime_type
        )
        labels["input_mime"] = media_facts.mime_type
    labels["image_count"] = str(media_facts.image_count)
    labels["image_input_mode"] = media_facts.input_mode
    if media_facts.pixels is not None:
        labels["pixels"] = str(media_facts.pixels)
    if model_name:
        labels["model_name"] = str(model_name)
    if device:
        labels["device"] = str(device)
    for key in (
        "locality",
        "data.governance",
        "data_governance",
        "access_requirement",
        "output_media_types",
        "operation",
    ):
        if key in kwargs and kwargs[key] is not None:
            labels[key] = str(kwargs[key])
    # Local-only media cannot fall back onto a route that requires remote upload.
    if media_facts.local_only and media_facts.image_count > 0:
        labels["forbid_remote_upload"] = "1"
        labels["requires_remote_upload"] = "0"
    return labels


def multimodal_fallback_compatible(
    origin_labels: Mapping[str, str],
    candidate_labels: Mapping[str, str],
) -> bool:
    """Return True when a fallback preserves multimodal contracts.

    Fallback must preserve operation, model compatibility, MIME, item count,
    dimensions, safety/data governance, locality/device, authorization, and
    output contract. It cannot use a route that requires a forbidden remote
    upload.
    """

    origin = {str(k): str(v) for k, v in origin_labels.items()}
    candidate = {str(k): str(v) for k, v in candidate_labels.items()}

    for key in (
        "operation",
        "locality",
        "device",
        "output_media_types",
        "mime_family",
    ):
        if key in origin and origin[key] not in {"", "unknown", "provider-managed"}:
            if candidate.get(key, origin[key]) != origin[key]:
                return False

    if "image_count" in origin:
        try:
            origin_count = int(origin["image_count"])
            cand_max = candidate.get("max_images")
            if cand_max and cand_max.isdigit() and origin_count > int(cand_max):
                return False
        except (TypeError, ValueError):
            return False

    if "pixels" in origin and "pixels" in candidate:
        if candidate["pixels"] != origin["pixels"]:
            # Allow candidate without pixel pin; reject explicit mismatch.
            return False

    origin_mime = origin.get("input_mime") or origin.get("mime_family")
    cand_mimes = (
        candidate.get("input_media_types")
        or candidate.get("mime_family")
        or candidate.get("input_mime")
        or ""
    )
    if origin_mime and cand_mimes:
        allowed = [part.strip() for part in cand_mimes.split(",") if part.strip()]
        if allowed and not any(
            origin_mime == item
            or item.endswith("/*")
            and origin_mime.startswith(item[:-1])
            or item == "image/*"
            and origin_mime.startswith("image/")
            for item in allowed
        ):
            return False

    origin_access = origin.get("access_requirement")
    cand_access = candidate.get("access_requirement")
    if origin_access == "required" and cand_access not in (
        None,
        "required",
        "optional",
    ):
        return False

    origin_gov = origin.get("data.governance") or origin.get("data_governance")
    cand_gov = candidate.get("data.governance") or candidate.get("data_governance")
    if origin_gov and cand_gov and cand_gov != origin_gov:
        return False
    if cand_gov and str(cand_gov).casefold() in {"deny", "forbidden", "blocked"}:
        return False

    # Forbidden remote upload: local/inline media must not move to a candidate
    # that requires uploading bytes to a remote host first.
    if origin.get("forbid_remote_upload") in {"1", "true", "yes"}:
        if candidate.get("requires_remote_upload") in {"1", "true", "yes"}:
            return False
        modes = {
            part.strip()
            for part in str(candidate.get("image_input_modes") or "").split(",")
            if part.strip()
        }
        if modes and "inline" not in modes and "uri" in modes:
            # URI-only remote endpoints cannot accept local-only media without upload.
            if origin.get("image_input_mode") == "inline":
                return False

    origin_model = origin.get("model_name")
    cand_model = candidate.get("model_name")
    if origin_model and cand_model and origin_model != cand_model:
        # Model pin is soft-compatible only when equivalence labels match.
        if origin.get("equivalent_model") and candidate.get("equivalent_model"):
            if origin.get("equivalent_model") != candidate.get("equivalent_model"):
                return False
        elif origin.get("model_compatible_with") != cand_model:
            return False
    return True


def _build_multimodal_static_candidate(
    *,
    provider_name: str,
    model_name: Optional[str],
    device: Optional[str],
    scope_id: str,
    media_facts: MediaReferenceFacts,
    kwargs: Mapping[str, object],
    score: int = 10,
    authorized: bool = True,
) -> "object":
    from .endpoint_usage.identity import stable_id
    from .endpoint_usage.resolution import StaticCandidate

    labels = _multimodal_compatibility_labels(
        provider_name=provider_name,
        model_name=model_name,
        device=device,
        media_facts=media_facts,
        kwargs=kwargs,
    )
    provider_id = stable_id("provider", "multimodal", provider_name)
    model_id = stable_id(
        "model", "multimodal", provider_name, model_name or "default"
    )
    deployment_id = stable_id(
        "deployment", "multimodal", provider_name, device or "default"
    )
    binding_id = stable_id(
        "binding", "multimodal", provider_name, model_name or "default", scope_id
    )
    return StaticCandidate(
        binding_id=binding_id,
        provider_id=provider_id,
        model_id=model_id,
        deployment_id=deployment_id,
        scope_id=scope_id,
        catalog_score=score,
        locality=labels.get("locality"),
        authorized=authorized,
        healthy=True,
        routable=True,
        configured=True,
        labels=labels,
    )


def _filter_compatible_candidates(
    candidates: Sequence[object],
    *,
    origin_labels: Mapping[str, str],
) -> List[object]:
    kept: List[object] = []
    for cand in candidates:
        labels = dict(getattr(cand, "labels", None) or {})
        if multimodal_fallback_compatible(origin_labels, labels):
            kept.append(cand)
    return kept


def _admission_result_to_trace(result: object) -> Dict[str, object]:
    """Reduce an admission result to a redacted operational trace."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output

    selected = getattr(result, "selected", None)
    receipt = getattr(result, "receipt", None)
    payload: Dict[str, object] = {
        "success": bool(getattr(result, "success", False)),
        "final_status": str(getattr(result, "final_status", "") or ""),
        "reason_codes": list(getattr(result, "reason_codes", ()) or ()),
        "next_eligible_at": getattr(result, "next_eligible_at", None),
        "attempt_count": len(getattr(result, "attempts", ()) or ()),
        "selected_binding_id": getattr(selected, "binding_id", None)
        if selected
        else None,
        "selected_scope_id": getattr(selected, "scope_id", None)
        if selected
        else None,
        "reservation_id": getattr(selected, "reservation_id", None)
        if selected
        else None,
        "receipt_id": getattr(receipt, "receipt_id", None) if receipt else None,
        "usage_revision": getattr(selected, "usage_revision", None)
        if selected
        else None,
        "catalog_revision": getattr(selected, "catalog_revision", None)
        if selected
        else None,
        "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
    }
    if receipt is not None:
        try:
            receipt_dict = receipt.to_dict()
            assert_no_prompt_media_or_output(receipt_dict)
            payload["receipt"] = receipt_dict
        except Exception:
            payload["receipt"] = {"receipt_id": payload.get("receipt_id")}
    assert_no_prompt_media_or_output(payload)
    return payload


def _parse_provider_observation(
    *,
    scope_id: str,
    request_id: str,
    observation: object,
    settled: object,
) -> Optional[object]:
    """Parse optional provider observation; never retain media or prompts."""

    if observation is None:
        return None
    from .endpoint_usage.schema import (
        ConfidenceLevel,
        LimitSource,
        Provenance,
        ProviderUsageObservation,
        UsageVector,
    )

    if isinstance(observation, ProviderUsageObservation):
        return observation
    if not isinstance(observation, Mapping):
        return None
    try:
        from .endpoint_usage.adapters import parse_provider_observation

        if any(
            key in observation
            for key in ("headers", "body", "family", "http_status", "usage")
        ):
            payload = dict(observation)
            payload.setdefault("scope_id", scope_id)
            payload.setdefault("request_id", request_id)
            return parse_provider_observation(payload)
    except Exception:
        logger.debug("multimodal usage observation adapter failed", exc_info=True)

    usage = observation.get("usage")
    if usage is None:
        usage = settled if isinstance(settled, UsageVector) else UsageVector()
    elif not isinstance(usage, UsageVector):
        try:
            usage = UsageVector.from_dict(usage)
        except Exception:
            usage = settled if isinstance(settled, UsageVector) else UsageVector()
    try:
        return ProviderUsageObservation(
            scope_id=scope_id,
            request_id=request_id,
            usage=usage,
            http_status=observation.get("http_status"),
            confidence=ConfidenceLevel.HIGH
            if observation.get("http_status") == 200
            else ConfidenceLevel.MEDIUM,
            provenance=Provenance(source=LimitSource.RESPONSE_BODY),
            reason_codes=tuple(observation.get("reason_codes") or ()),
            retry_after_ms=observation.get("retry_after_ms"),
            limits=tuple(observation.get("limits") or ()),
        )
    except Exception:
        logger.debug("multimodal usage observation construct failed", exc_info=True)
        return None


def _resolve_usage_pin(
    *,
    pin: object,
    provider: Optional[str],
    allow_fallback_with_pin: bool,
) -> object:
    from .endpoint_usage.identity import stable_id
    from .endpoint_usage.routing import RoutePin

    if pin is not None:
        if isinstance(pin, RoutePin):
            return pin
        if isinstance(pin, Mapping):
            return RoutePin(
                provider_id=pin.get("provider_id"),
                model_id=pin.get("model_id"),
                deployment_id=pin.get("deployment_id"),
                binding_id=pin.get("binding_id"),
                allow_fallback_with_pin=bool(
                    pin.get("allow_fallback_with_pin", allow_fallback_with_pin)
                ),
            )
        raise TypeError("usage_pin must be a RoutePin, mapping, or None")
    if provider:
        return RoutePin(
            provider_id=stable_id("provider", "multimodal", provider),
            allow_fallback_with_pin=allow_fallback_with_pin,
        )
    return RoutePin()


def _record_usage_observe_shadow(
    *,
    prompt: str,
    media_facts: MediaReferenceFacts,
    remote_charged: bool,
    usage_coordinator: object,
    usage_policy: object,
    usage_scope_id: Optional[str],
    usage_request_id: Optional[str],
    usage_cost_micros: Optional[int],
    usage_cost_currency: Optional[str],
    success: bool,
    provider_used: str,
    max_output_tokens: Optional[int],
) -> None:
    """Observe/shadow diagnostics: estimate only; never change selection or charge."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output, stable_id
    from .endpoint_usage.schema import RoutingMode, UsageEventKind

    policy = usage_policy
    mode = getattr(policy, "mode", RoutingMode.OBSERVE)
    estimate = estimate_multimodal_usage(
        prompt,
        media_facts=media_facts,
        max_output_tokens=max_output_tokens,
        cost_micros=usage_cost_micros,
        cost_currency=usage_cost_currency,
        remote=remote_charged,
    )
    payload: Dict[str, object] = {
        "success": success,
        "final_status": "observed" if success else "observe_error",
        "reason_codes": [
            "usage_observe"
            if mode is RoutingMode.OBSERVE or str(mode) == "observe"
            else "usage_shadow",
            "no_selection_change",
        ],
        "attempt_count": 0,
        "estimate_entries": len(getattr(estimate, "entries", ()) or ()),
        "provider_used": str(provider_used or ""),
        "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
        "remote_charged": False,
        "mode": str(getattr(mode, "value", mode)),
        "image_count": media_facts.image_count,
    }
    if not remote_charged:
        payload["reason_codes"] = list(payload["reason_codes"]) + [
            "cache_hit",
            "no_remote_charge",
        ]
    if usage_scope_id and usage_coordinator is not None:
        try:
            snap = usage_coordinator.snapshot(usage_scope_id)  # type: ignore[attr-defined]
            payload["usage_revision"] = getattr(snap, "usage_revision", None)
            payload["scope_id"] = usage_scope_id
        except Exception:
            payload["reason_codes"] = list(payload["reason_codes"]) + [
                "snapshot_unavailable"
            ]
        if (
            success
            and remote_charged
            and str(getattr(mode, "value", mode)) == "shadow"
        ):
            try:
                from .endpoint_usage.schema import UsageVector as _UsageVector

                usage_coordinator.append_observation(  # type: ignore[attr-defined]
                    usage_scope_id,
                    kind=UsageEventKind.OBSERVATION_SUCCESS,
                    units=_UsageVector(),
                    request_id=usage_request_id
                    or stable_id("mreq", "shadow", usage_scope_id),
                    reason_codes=("shadow_observe",),
                )
            except Exception:
                logger.debug("shadow observation append failed", exc_info=True)
    _ = estimate
    assert_no_prompt_media_or_output(payload)
    _set_last_usage_admission(payload)


def _generate_multimodal_with_usage_admission(
    prompt: str,
    *,
    image: Optional[Union[str, bytes]],
    model_name: Optional[str],
    device: Optional[str],
    provider: Optional[str],
    provider_instance: Optional[MultimodalProvider],
    deps: "RouterDeps",
    kwargs: Dict[str, object],
    usage_coordinator: object,
    usage_policy: object,
    usage_candidates: Optional[Sequence[object]],
    usage_pin: object,
    usage_request: object,
    usage_request_id: Optional[str],
    usage_idempotency_key: Optional[str],
    usage_catalog_revision: Optional[str],
    usage_provider_by_binding: Optional[Mapping[str, MultimodalProvider]],
    usage_observation: object,
    usage_cost_micros: Optional[int],
    usage_cost_currency: Optional[str],
    usage_cancel_event: Optional[threading.Event],
    usage_max_media_bytes: Optional[int],
    usage_width: Optional[int],
    usage_height: Optional[int],
    usage_declared_media_bytes: Optional[int],
    usage_mime_type: Optional[str],
    started: float,
) -> str:
    """Reserve, dispatch, settle one multimodal unit under admission."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output, stable_id
    from .endpoint_usage.resolution import StaticCandidate, UsageRoutingRequest
    from .endpoint_usage.routing import (
        ErrorSafetyClass,
        InvokeOutcome,
        UsageRouteAdmission,
        classify_invoke_error,
        meta_from_static,
    )
    from .endpoint_usage.schema import (
        FallbackClass,
        RoutingPolicy,
        UsageVector,
    )

    policy = usage_policy
    if not isinstance(policy, RoutingPolicy):
        policy = _normalize_usage_policy(policy)

    # Adversarial size/MIME/SSRF-shaped inputs fail before reservation.
    max_output_tokens = kwargs.get("max_tokens")
    if max_output_tokens is None:
        max_output_tokens = kwargs.get("max_output_tokens")
    try:
        media_facts = validate_multimodal_media_input(
            image,
            max_media_bytes=usage_max_media_bytes,
            width=usage_width,
            height=usage_height,
            declared_media_bytes=usage_declared_media_bytes,
            mime_type=usage_mime_type,
        )
    except MultimodalRouterError as exc:
        _set_last_usage_admission(
            {
                "success": False,
                "final_status": "media_rejected",
                "reason_codes": ["media_policy_rejected", type(exc).__name__],
                "attempt_count": 0,
                "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                "remote_charged": False,
            }
        )
        raise

    resolved_for_cache = provider_instance
    cache_provider_name = (
        _provider_name(resolved_for_cache, requested=provider)
        if resolved_for_cache is not None
        else str(provider or "")
    )
    if resolved_for_cache is None and provider:
        try:
            resolved_for_cache = get_multimodal_provider(provider, deps=deps)
            cache_provider_name = _provider_name(
                resolved_for_cache, requested=provider
            )
        except Exception:
            resolved_for_cache = None

    cache_enabled = _response_cache_enabled()
    if cache_enabled and cache_provider_name:
        cache_key = _response_cache_key(
            provider=cache_provider_name,
            model_name=model_name,
            prompt=prompt,
            image=image,
            kwargs=dict(kwargs),
        )
        try:
            getter = getattr(deps, "get_cached_or_remote", None)
            cached = (
                getter(cache_key)
                if callable(getter)
                else deps.get_cached(cache_key)
            )
            if isinstance(cached, str):
                _set_last_usage_admission(
                    {
                        "success": True,
                        "final_status": "cache_hit",
                        "reason_codes": ["cache_hit", "no_remote_charge"],
                        "attempt_count": 0,
                        "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                        "remote_charged": False,
                    }
                )
                assert_no_prompt_media_or_output(get_last_usage_admission())
                _set_last_multimodal_trace(
                    status="ok",
                    provider_requested=str(provider or ""),
                    provider_used=cache_provider_name,
                    model_name=str(model_name or ""),
                    device=str(device or ""),
                    cache_hit=True,
                    fallback_used=False,
                    usage_mode=str(getattr(policy.mode, "value", policy.mode)),
                    remote_charged=False,
                    image_count=media_facts.image_count,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                )
                return cached
        except Exception:
            pass

    if usage_cancel_event is not None and usage_cancel_event.is_set():
        raise UsageCapacityError(
            "multimodal usage admission cancelled before dispatch",
            reason_codes=("cancelled_before_dispatch",),
        )

    requested = estimate_multimodal_usage(
        prompt,
        media_facts=media_facts,
        max_output_tokens=int(max_output_tokens)
        if isinstance(max_output_tokens, int)
        or (
            isinstance(max_output_tokens, str) and str(max_output_tokens).isdigit()
        )
        else None,
        cost_micros=usage_cost_micros,
        cost_currency=usage_cost_currency,
        remote=True,
    )
    request_id = usage_request_id or stable_id(
        "mreq", "multimodal", str(time.time_ns()), str(media_facts.image_count)
    )
    idempotency_key = usage_idempotency_key or stable_id(
        "midem", request_id, str(media_facts.image_count)
    )
    catalog_revision = usage_catalog_revision or stable_id(
        "cat", "multimodal_router", USAGE_ROUTING_REQUIREMENT_ID
    )

    pin = _resolve_usage_pin(
        pin=usage_pin,
        provider=provider,
        allow_fallback_with_pin=False,
    )

    candidates: List[object]
    if usage_candidates is not None:
        candidates = list(usage_candidates)
    else:
        backend = provider_instance or get_multimodal_provider(provider, deps=deps)
        provider_used = _provider_name(backend, requested=provider)
        scope_id = stable_id(
            "scope", "multimodal", provider_used, model_name or "default"
        )
        ureq_probe = usage_request
        if isinstance(ureq_probe, Mapping):
            preferred_scope = ureq_probe.get("preferred_scope_id")
        else:
            preferred_scope = getattr(ureq_probe, "preferred_scope_id", None)
        if preferred_scope:
            scope_id = str(preferred_scope)
        candidates = [
            _build_multimodal_static_candidate(
                provider_name=provider_used,
                model_name=model_name,
                device=device,
                scope_id=scope_id,
                media_facts=media_facts,
                kwargs=kwargs,
            )
        ]
        usage_provider_by_binding = {
            candidates[0].binding_id: backend,  # type: ignore[attr-defined]
            **dict(usage_provider_by_binding or {}),
        }

    if not candidates:
        raise UsageCapacityError(
            "no multimodal usage candidates",
            reason_codes=("no_candidates",),
        )

    first = candidates[0]
    origin_labels = dict(getattr(first, "labels", None) or {})
    origin_labels.update(
        _multimodal_compatibility_labels(
            provider_name=str(
                origin_labels.get("router_provider") or provider or ""
            ),
            model_name=model_name,
            device=device,
            media_facts=media_facts,
            kwargs=kwargs,
        )
    )
    candidates = _filter_compatible_candidates(
        candidates, origin_labels=origin_labels
    ) or list(candidates[:1])

    meta_by_binding = {
        cand.binding_id: meta_from_static(cand)  # type: ignore[attr-defined]
        for cand in candidates
        if isinstance(cand, StaticCandidate)
    }

    planning_required = planning_required_usage(requested)
    ureq = usage_request
    if ureq is None:
        ureq = UsageRoutingRequest(
            required=planning_required,
            require_snapshot=True,
        )
    elif isinstance(ureq, Mapping):
        ureq = UsageRoutingRequest.from_dict(ureq)
    elif not isinstance(ureq, UsageRoutingRequest):
        raise TypeError("usage_request must be UsageRoutingRequest, mapping, or None")
    source_required = ureq.required if ureq.required.entries else requested
    safe_required = planning_required_usage(source_required)
    if not safe_required.entries:
        safe_required = planning_required
    ureq = UsageRoutingRequest(
        required=safe_required,
        unknown_limit_policy=ureq.unknown_limit_policy,
        stale_snapshot_policy=ureq.stale_snapshot_policy,
        preferred_binding_id=ureq.preferred_binding_id,
        preferred_provider_id=ureq.preferred_provider_id,
        preferred_scope_id=ureq.preferred_scope_id,
        affinity_binding_id=ureq.affinity_binding_id,
        media_bytes=ureq.media_bytes,
        images=ureq.images,
        audio_seconds=ureq.audio_seconds,
        max_cost_micros=ureq.max_cost_micros,
        cost_currency=ureq.cost_currency,
        deadline_at=ureq.deadline_at,
        now=ureq.now,
        health_by_binding=ureq.health_by_binding,
        circuit_open_by_binding=ureq.circuit_open_by_binding,
        latency_ms_by_binding=ureq.latency_ms_by_binding,
        queue_delay_ms_by_binding=ureq.queue_delay_ms_by_binding,
        quality_preference_by_binding=ureq.quality_preference_by_binding,
        locality_by_binding=ureq.locality_by_binding,
        require_snapshot=ureq.require_snapshot,
        reason_codes=ureq.reason_codes,
    )

    provider_map: Dict[str, MultimodalProvider] = dict(
        usage_provider_by_binding or {}
    )
    result_holder: Dict[str, object] = {}
    fallback_used = False
    used_model_name = model_name
    provider_used_name = cache_provider_name
    invoke_error_holder: Dict[str, BaseException] = {}

    def invoke(attempt: object) -> InvokeOutcome:
        nonlocal fallback_used, used_model_name, provider_used_name
        if usage_cancel_event is not None and usage_cancel_event.is_set():
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.CLIENT,
                reason_codes=("cancelled_before_dispatch",),
                side_effecting=False,
            )
        binding_id = getattr(attempt, "binding_id", None)
        scope_id = getattr(attempt, "scope_id", None) or ""
        active_backend: Optional[MultimodalProvider] = None
        labels: Dict[str, str] = {}
        if binding_id and binding_id in provider_map:
            active_backend = provider_map[binding_id]
            for cand in candidates:
                if getattr(cand, "binding_id", None) == binding_id:
                    labels = dict(getattr(cand, "labels", None) or {})
                    break
        else:
            for cand in candidates:
                if getattr(cand, "binding_id", None) == binding_id:
                    labels = dict(getattr(cand, "labels", None) or {})
                    break
            if labels and not multimodal_fallback_compatible(origin_labels, labels):
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.SEMANTIC,
                    reason_codes=("incompatible_multimodal_candidate",),
                    side_effecting=False,
                )
            # Reject routes that would require a forbidden remote upload.
            if origin_labels.get("forbid_remote_upload") in {"1", "true", "yes"}:
                if labels.get("requires_remote_upload") in {"1", "true", "yes"}:
                    return InvokeOutcome(
                        success=False,
                        error_class=ErrorSafetyClass.SEMANTIC,
                        reason_codes=("forbidden_remote_upload",),
                        side_effecting=False,
                    )
            router_name = labels.get("router_provider") or provider
            try:
                active_backend = provider_instance or get_multimodal_provider(
                    router_name, deps=deps
                )
            except Exception as exc:
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.TRANSIENT,
                    reason_codes=("provider_resolve_failed", type(exc).__name__),
                    side_effecting=False,
                )
        assert active_backend is not None
        try:
            text = str(
                active_backend.generate(
                    prompt,
                    image=image,
                    model_name=model_name,
                    device=device,
                    **kwargs,
                )
            )
        except Exception as exc:
            invoke_error_holder["error"] = exc
            error_class = classify_invoke_error(
                reason_codes=(type(exc).__name__,),
            )
            message = str(exc).casefold()
            if any(
                token in message
                for token in ("rate limit", "429", "quota", "capacity", "503")
            ):
                error_class = ErrorSafetyClass.CAPACITY
            return InvokeOutcome(
                success=False,
                error_class=error_class,
                reason_codes=("provider_error", type(exc).__name__),
                side_effecting=False,
            )

        settled = settle_multimodal_usage(
            prompt,
            media_facts=media_facts,
            output_text=text,
            max_output_tokens=int(max_output_tokens)
            if isinstance(max_output_tokens, int)
            else None,
            cost_micros=usage_cost_micros,
            cost_currency=usage_cost_currency,
        )
        obs = _parse_provider_observation(
            scope_id=str(scope_id),
            request_id=request_id,
            observation=usage_observation,
            settled=settled,
        )
        provider_used_name = _provider_name(active_backend, requested=provider)
        used_model_name = model_name
        result_holder["text"] = text
        result_holder["provider_used"] = provider_used_name
        result_holder["settled"] = settled
        return InvokeOutcome(
            success=True,
            observation=obs,
            settled=settled,
            error_class=ErrorSafetyClass.SUCCESS,
            reason_codes=("generated",),
        )

    admission = UsageRouteAdmission(
        usage_coordinator,  # type: ignore[arg-type]
        owner_id="multimodal_router",
        jitter_max_ms=0,
    )
    effective_policy = policy
    if getattr(pin, "is_exact", False) and not getattr(
        pin, "allow_fallback_with_pin", False
    ):
        if policy.fallback is not FallbackClass.NONE:
            effective_policy = RoutingPolicy(
                mode=policy.mode,
                fallback=FallbackClass.NONE,
                max_attempts=1,
                deadline_ms=policy.deadline_ms,
                allow_wait=policy.allow_wait,
                max_wait_ms=policy.max_wait_ms,
                prefer_local=policy.prefer_local,
                cost_ceiling_micros=policy.cost_ceiling_micros,
                cost_currency=policy.cost_currency,
            )

    result = admission.admit(
        catalog_revision=catalog_revision,
        candidates=candidates,  # type: ignore[arg-type]
        request_id=request_id,
        idempotency_key=idempotency_key,
        operation=MULTIMODAL_USAGE_OPERATION,
        requested=requested if isinstance(requested, UsageVector) else UsageVector(),
        policy=effective_policy,
        request=ureq,
        pin=pin,  # type: ignore[arg-type]
        meta_by_binding=meta_by_binding,
        invoke=invoke,
        caller_id="multimodal_router",
    )
    _set_last_usage_admission(_admission_result_to_trace(result))

    if not result.success or "text" not in result_holder:
        original = invoke_error_holder.get("error")
        capacity_like = any(
            code
            in {
                "no_eligible_candidates",
                "all_candidates_denied",
                "capacity",
                "deadline_or_attempt_bound",
                "pin_rejected",
            }
            or "capacity" in code
            or "headroom" in code
            for code in (result.reason_codes or ())
        )
        if original is not None and not capacity_like:
            raise original
        raise UsageCapacityError(
            "multimodal usage admission failed: %s"
            % (",".join(result.reason_codes) or result.final_status),
            reason_codes=result.reason_codes,
            next_eligible_at=result.next_eligible_at,
            admission=result,
        )

    text = str(result_holder["text"])
    provider_used_name = str(result_holder.get("provider_used") or provider_used_name)
    if cache_enabled:
        try:
            cache_key = _response_cache_key(
                provider=provider_used_name or cache_provider_name or provider,
                model_name=used_model_name,
                prompt=prompt,
                image=image,
                kwargs=dict(kwargs),
            )
            setter = getattr(deps, "set_cached_and_remote", None)
            if callable(setter):
                setter(cache_key, text)
            else:
                deps.set_cached(cache_key, text)
        except Exception:
            pass

    _set_last_multimodal_trace(
        status="ok",
        provider_requested=str(provider or ""),
        provider_used=provider_used_name,
        model_name=str(used_model_name or ""),
        device=str(device or ""),
        cache_hit=False,
        fallback_used=fallback_used
        or bool(
            result.selected
            and result.attempts
            and len(result.attempts) > 1
        ),
        usage_mode=str(getattr(policy.mode, "value", policy.mode)),
        remote_charged=True,
        reservation_id=getattr(result.selected, "reservation_id", None)
        if result.selected
        else None,
        receipt_id=getattr(result.receipt, "receipt_id", None)
        if result.receipt
        else None,
        image_count=media_facts.image_count,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return text


def generate_multimodal(
    prompt: str,
    *,
    image: Optional[Union[str, bytes]] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[MultimodalProvider] = None,
    deps: Optional[RouterDeps] = None,
    usage_coordinator: Optional[object] = None,
    usage_policy: Optional[object] = None,
    usage_candidates: Optional[Sequence[object]] = None,
    usage_pin: Optional[object] = None,
    usage_request: Optional[object] = None,
    usage_request_id: Optional[str] = None,
    usage_idempotency_key: Optional[str] = None,
    usage_catalog_revision: Optional[str] = None,
    usage_provider_by_binding: Optional[Mapping[str, MultimodalProvider]] = None,
    usage_observation: Optional[object] = None,
    usage_cost_micros: Optional[int] = None,
    usage_cost_currency: Optional[str] = None,
    usage_cancel_event: Optional[threading.Event] = None,
    usage_scope_id: Optional[str] = None,
    usage_max_media_bytes: Optional[int] = None,
    usage_width: Optional[int] = None,
    usage_height: Optional[int] = None,
    usage_declared_media_bytes: Optional[int] = None,
    usage_mime_type: Optional[str] = None,
    **kwargs: object,
) -> str:
    """Generate text from a prompt and optional image.

    Optional usage-aware admission (AICAT-032) is inactive unless a
    ``usage_coordinator`` is supplied with a non-``off`` ``usage_policy``.
    Off mode and a missing coordinator preserve legacy selection and errors
    exactly. Enforce/assist reserve before remote dispatch; observe/shadow
    never change the selected provider; cache hits create no remote charge.
    Media remains referenced and never enters the ledger or receipt.

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
    started = time.perf_counter()
    resolved_deps = deps or get_default_router_deps()
    policy = _normalize_usage_policy(usage_policy)
    if usage_scope_id is None and usage_request is not None:
        if isinstance(usage_request, Mapping):
            usage_scope_id = usage_request.get("preferred_scope_id")  # type: ignore[assignment]
        else:
            usage_scope_id = getattr(usage_request, "preferred_scope_id", None)

    if usage_coordinator is not None and _usage_mode_enforces(policy):
        return _generate_multimodal_with_usage_admission(
            prompt,
            image=image,
            model_name=model_name,
            device=device,
            provider=provider,
            provider_instance=provider_instance,
            deps=resolved_deps,
            kwargs=dict(kwargs),
            usage_coordinator=usage_coordinator,
            usage_policy=policy,
            usage_candidates=usage_candidates,
            usage_pin=usage_pin,
            usage_request=usage_request,
            usage_request_id=usage_request_id,
            usage_idempotency_key=usage_idempotency_key,
            usage_catalog_revision=usage_catalog_revision,
            usage_provider_by_binding=usage_provider_by_binding,
            usage_observation=usage_observation,
            usage_cost_micros=usage_cost_micros,
            usage_cost_currency=usage_cost_currency,
            usage_cancel_event=usage_cancel_event,
            usage_max_media_bytes=usage_max_media_bytes,
            usage_width=usage_width,
            usage_height=usage_height,
            usage_declared_media_bytes=usage_declared_media_bytes,
            usage_mime_type=usage_mime_type,
            started=started,
        )

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
            cached = (
                getter(cache_key)
                if callable(getter)
                else resolved_deps.get_cached(cache_key)
            )
            if isinstance(cached, str):
                if usage_coordinator is not None and _usage_mode_observes_only(policy):
                    facts = inspect_media_reference(
                        image,
                        width=usage_width,
                        height=usage_height,
                        declared_media_bytes=usage_declared_media_bytes,
                        mime_type=usage_mime_type,
                    )
                    _record_usage_observe_shadow(
                        prompt=prompt,
                        media_facts=facts,
                        remote_charged=False,
                        usage_coordinator=usage_coordinator,
                        usage_policy=policy,
                        usage_scope_id=usage_scope_id,
                        usage_request_id=usage_request_id,
                        usage_cost_micros=usage_cost_micros,
                        usage_cost_currency=usage_cost_currency,
                        success=True,
                        provider_used=str(provider or ""),
                        max_output_tokens=None,
                    )
                elif usage_coordinator is None or _usage_mode_is_off(
                    policy, usage_coordinator
                ):
                    _set_last_usage_admission(
                        {
                            "success": True,
                            "final_status": "off",
                            "reason_codes": ["usage_routing_off", "cache_hit"],
                            "attempt_count": 0,
                            "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                            "remote_charged": False,
                            "mode": "off",
                        }
                    )
                _set_last_multimodal_trace(
                    status="ok",
                    provider_requested=str(provider or ""),
                    provider_used=str(provider or ""),
                    model_name=str(model_name or ""),
                    device=str(device or ""),
                    cache_hit=True,
                    fallback_used=False,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                )
                return cached
        except Exception:
            pass

    backend = provider_instance or get_multimodal_provider(
        provider, deps=resolved_deps
    )
    provider_used = _provider_name(backend, requested=provider)
    fallback_used = False
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
                    provider=provider_used or provider,
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
        _set_last_multimodal_trace(
            status="ok",
            provider_requested=str(provider or ""),
            provider_used=provider_used,
            model_name=str(model_name or ""),
            device=str(device or ""),
            cache_hit=False,
            fallback_used=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        if usage_coordinator is not None and _usage_mode_observes_only(policy):
            facts = inspect_media_reference(
                image,
                width=usage_width,
                height=usage_height,
                declared_media_bytes=usage_declared_media_bytes,
                mime_type=usage_mime_type,
            )
            _record_usage_observe_shadow(
                prompt=prompt,
                media_facts=facts,
                remote_charged=True,
                usage_coordinator=usage_coordinator,
                usage_policy=policy,
                usage_scope_id=usage_scope_id,
                usage_request_id=usage_request_id,
                usage_cost_micros=usage_cost_micros,
                usage_cost_currency=usage_cost_currency,
                success=True,
                provider_used=provider_used,
                max_output_tokens=kwargs.get("max_tokens")  # type: ignore[arg-type]
                if isinstance(kwargs.get("max_tokens"), int)
                else None,
            )
        elif usage_coordinator is None or _usage_mode_is_off(
            policy, usage_coordinator
        ):
            _set_last_usage_admission(
                {
                    "success": True,
                    "final_status": "off",
                    "reason_codes": ["usage_routing_off"],
                    "attempt_count": 0,
                    "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                    "remote_charged": None,
                    "mode": "off",
                }
            )
        return text
    except Exception as primary_error:
        logger.debug(f"Primary multimodal provider failed: {primary_error}")
        if provider is None:
            hf_provider = _get_huggingface_provider()
            if hf_provider is not None and backend is not hf_provider:
                text = str(
                    hf_provider.generate(
                        prompt,
                        image=image,
                        model_name=model_name,
                        device=device,
                        **kwargs,
                    )
                )
                fallback_used = True
                provider_used = "huggingface"
                _set_last_multimodal_trace(
                    status="ok",
                    provider_requested=str(provider or ""),
                    provider_used=provider_used,
                    model_name=str(model_name or ""),
                    device=str(device or ""),
                    cache_hit=False,
                    fallback_used=fallback_used,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                )
                if usage_coordinator is not None and _usage_mode_observes_only(
                    policy
                ):
                    facts = inspect_media_reference(image)
                    _record_usage_observe_shadow(
                        prompt=prompt,
                        media_facts=facts,
                        remote_charged=True,
                        usage_coordinator=usage_coordinator,
                        usage_policy=policy,
                        usage_scope_id=usage_scope_id,
                        usage_request_id=usage_request_id,
                        usage_cost_micros=usage_cost_micros,
                        usage_cost_currency=usage_cost_currency,
                        success=True,
                        provider_used=provider_used,
                        max_output_tokens=None,
                    )
                elif usage_coordinator is None or _usage_mode_is_off(
                    policy, usage_coordinator
                ):
                    _set_last_usage_admission(
                        {
                            "success": True,
                            "final_status": "off",
                            "reason_codes": ["usage_routing_off"],
                            "attempt_count": 0,
                            "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                            "remote_charged": None,
                            "mode": "off",
                        }
                    )
                return text
        _set_last_multimodal_trace(
            status="error",
            provider_requested=str(provider or ""),
            provider_used=provider_used,
            model_name=str(model_name or ""),
            device=str(device or ""),
            cache_hit=False,
            fallback_used=False,
            error_type=type(primary_error).__name__,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
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
