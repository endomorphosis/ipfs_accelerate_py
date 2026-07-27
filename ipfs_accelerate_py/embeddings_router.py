"""Embeddings router for ipfs_accelerate_py.

This module provides a stable, reusable entrypoint for generating embeddings
that integrates with existing ipfs_accelerate_py infrastructure.

Design goals:
- Avoid import-time side effects (no heavy imports at module import).
- Allow optional hooks/providers (backend manager, custom remote endpoints).
- Provide a reliable local fallback (Gemini CLI -> HF transformers).
- Reuse existing CLI/SDK wrappers (no duplication).
- Support endpoint multiplexing via InferenceBackendManager.

Environment variables:
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: enable backend manager provider
- `IPFS_ACCELERATE_PY_EMBEDDINGS_BACKEND`: force backend for local adapter
- `IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL`: HF model name for local adapter
- `IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE`: device for local adapter (cpu/cuda)

Additional optional providers (opt-in by selecting provider):
- `openrouter`: OpenRouter embeddings endpoint
    - `OPENROUTER_API_KEY` or `IPFS_ACCELERATE_PY_OPENROUTER_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL`
    - `IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL` (default: https://openrouter.ai/api/v1)
- `xai`: xAI Grok embeddings via OpenAI-compatible endpoint
    - `XAI_API_KEY` or `ipfs_accelerate_py_XAI_API_KEY`
    - `ipfs_accelerate_py_XAI_EMBEDDINGS_MODEL` (default: v1)
    - `ipfs_accelerate_py_XAI_BASE_URL` (default: https://api.x.ai/v1)
- `meta_ai`: Meta AI / Llama embeddings via OpenAI-compatible endpoint
    - `META_AI_API_KEY` or `ipfs_accelerate_py_META_AI_API_KEY`
    - `ipfs_accelerate_py_META_AI_EMBEDDINGS_MODEL` (default: meta-llama/Llama-3.3-70B-Instruct)
    - `ipfs_accelerate_py_META_AI_BASE_URL` (default: https://api.llamameta.net/v1)
- `gemini_cli`: Gemini CLI embeddings via existing integration
- `huggingface`: HuggingFace embeddings via existing integration
- `backend_manager`: Use InferenceBackendManager for distributed/multiplexed inference
"""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
import hashlib
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
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


def get_accelerate_manager(
    *,
    deps: Optional[RouterDeps] = None,
    purpose: str = "embeddings",
    enable_distributed: bool = True,
    resources: Optional[dict[str, object]] = None,
    ipfs_gateway: Optional[str] = None,
) -> object | None:
    """Return a router dependency's cached accelerator manager when supported."""

    resolved = deps or get_default_router_deps()
    factory = getattr(resolved, "get_accelerate_manager", None)
    if not callable(factory):
        return None
    try:
        return factory(
            purpose=purpose,
            enable_distributed=enable_distributed,
            resources=resources,
            ipfs_gateway=ipfs_gateway,
        )
    except Exception:
        return None


def get_accelerate_status() -> dict[str, object]:
    """Return lightweight availability status without initializing a backend."""

    env_value = os.environ.get("IPFS_ACCELERATE_ENABLED", "1").lower()
    env_disabled = env_value in {"0", "false", "no", "disabled"}
    if env_disabled:
        return {
            "available": False,
            "enabled": False,
            "env_disabled": True,
            "env_var": env_value,
        }

    try:
        import importlib.util

        available = importlib.util.find_spec("ipfs_accelerate_py") is not None
    except Exception:
        available = False
    return {
        "available": available,
        "enabled": True,
        "env_disabled": False,
        "env_var": env_value,
    }


class EmbeddingsRouterError(RuntimeError):
    """Raised when a provider violates the embeddings router contract."""


_LAST_EMBEDDING_TRACE = threading.local()
_EMBEDDING_PROGRESS_LOCK = threading.Lock()
_LAST_EMBEDDING_PROGRESS: Dict[str, object] = {
    "stage": "",
    "total_items": 0,
    "completed_items": 0,
    "total_batches": 0,
    "completed_batches": 0,
}


def _set_last_embedding_trace(**values: object) -> None:
    _LAST_EMBEDDING_TRACE.payload = dict(values)


def get_last_embedding_trace() -> Dict[str, object]:
    """Return a copy of the most recent embedding-call trace for this thread."""

    payload = getattr(_LAST_EMBEDDING_TRACE, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _update_embedding_progress(**values: object) -> Dict[str, object]:
    with _EMBEDDING_PROGRESS_LOCK:
        _LAST_EMBEDDING_PROGRESS.update(values)
        return dict(_LAST_EMBEDDING_PROGRESS)


def _reset_embedding_progress(**values: object) -> Dict[str, object]:
    with _EMBEDDING_PROGRESS_LOCK:
        _LAST_EMBEDDING_PROGRESS.clear()
        _LAST_EMBEDDING_PROGRESS.update(values)
        return dict(_LAST_EMBEDDING_PROGRESS)


def get_embedding_progress() -> Dict[str, object]:
    """Return a thread-safe snapshot of the current bounded batch."""

    with _EMBEDDING_PROGRESS_LOCK:
        return dict(_LAST_EMBEDDING_PROGRESS)


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
        return True  # Default to enabled
    return str(value).strip() != "0"


def _response_cache_key_strategy() -> str:
    return (
        os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE_KEY")
        or "sha256"
    ).strip().lower() or "sha256"


def _response_cache_cid_base() -> str:
    return (
        os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE")
        or os.environ.get("IPFS_DATASETS_PY_ROUTER_CACHE_CID_BASE")
        or "base32"
    ).strip() or "base32"


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


def _effective_model_key(*, provider_key: str, model_name: Optional[str], kwargs: Dict[str, object]) -> str:
    """Best-effort model identifier for caching.

    Embeddings callers sometimes pass model via kwargs (e.g. ``model=...``), and
    the local adapter uses env defaults. Cache keys must include the effective
    model to avoid cross-model collisions.
    """

    direct = (model_name or "").strip()
    if direct:
        return direct

    for key in ("model", "model_name", "model_id"):
        try:
            value = kwargs.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    pk = (provider_key or "auto").strip().lower()
    if pk == "openrouter":
        return (
            os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_DATASETS_PY_OPENROUTER_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_MODEL")
            or ""
        ).strip()
    if _is_hf_inference_provider_name(pk):
        return (
            os.getenv("IPFS_ACCELERATE_PY_HF_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_DATASETS_PY_HF_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL")
            or os.getenv("IPFS_DATASETS_PY_HF_INFERENCE_MODEL")
            or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL")
            or os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_MODEL")
            or ""
        ).strip()

    # Local adapter / default.
    return _coalesce_env(
        "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
        "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
    )


def _response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    device: Optional[str],
    text: str,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = _effective_model_key(provider_key=provider_key, model_name=model_name, kwargs=kwargs)
    device_key = (device or "").strip().lower()

    strategy = _response_cache_key_strategy()
    if strategy == "cid":
        try:
            from .utils.cid_utils import cid_for_obj

            payload = {
                "type": "embeddings_response",
                "provider": provider_key,
                "model": model_key,
                "device": device_key,
                "text": text or "",
                "kwargs": kwargs or {},
            }
            cid = cid_for_obj(payload, base=_response_cache_cid_base())
            return f"embeddings_response_cid::{cid}"
        except Exception:
            pass  # Fall back to sha256

    kw_digest = _stable_kwargs_digest(kwargs)
    return f"embeddings_response::{provider_key}::{model_key}::{device_key}::{_text_digest(text)}::{kw_digest}"


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Provider interface for embedding generation."""

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> List[List[float]]: ...


ProviderFactory = Callable[[], EmbeddingsProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory
    descriptor: Optional[ProviderDescriptor] = None
    models: Tuple[ModelDescriptor, ...] = ()


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}
_PROVIDER_REGISTRY_LOCK = threading.RLock()


def _registered_provider_descriptor(
    name: str,
    descriptor: ProviderDescriptor | Mapping[str, object] | None,
) -> ProviderDescriptor:
    if descriptor is None:
        return ProviderDescriptor(
            name=name,
            description="Dynamically registered embeddings provider.",
            capabilities=(_embedding_capability(),),
            lifecycle=LifecycleState.DECLARED,
            state=OperationalState(
                known=True,
                configured=True,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
            provenance=(Provenance(source="embeddings_router.registry"),),
            labels={
                "access_requirement": "unknown",
                "batching": "supported",
                "device": "unknown",
                "input_types": "text",
                "locality": "unknown",
                "normalization": "unknown",
            },
        )
    if isinstance(descriptor, ProviderDescriptor):
        resolved = descriptor
    elif isinstance(descriptor, Mapping):
        values = dict(descriptor)
        values.setdefault("name", name)
        resolved = ProviderDescriptor(**values)
    else:
        raise TypeError("descriptor must be a ProviderDescriptor, mapping, or None")
    if resolved.name != name:
        raise ValueError("Provider descriptor name must match the registered name")
    return resolved


def _registered_model_descriptors(
    provider: ProviderDescriptor,
    models: Sequence[ModelDescriptor | Mapping[str, object]],
) -> Tuple[ModelDescriptor, ...]:
    if isinstance(models, (str, bytes, Mapping)):
        raise TypeError("models must be a sequence of model descriptors")
    output: list[ModelDescriptor] = []
    for model in models:
        if isinstance(model, ModelDescriptor):
            resolved = model
        elif isinstance(model, Mapping):
            values = dict(model)
            values.setdefault("provider_id", provider.provider_id)
            resolved = ModelDescriptor(**values)
        else:
            raise TypeError("models must contain ModelDescriptor records or mappings")
        if resolved.provider_id != provider.provider_id:
            raise ValueError("Model descriptor provider_id does not match provider")
        output.append(resolved)
    identities = [model.model_id for model in output]
    if len(identities) != len(set(identities)):
        raise ValueError("models contain duplicate identities")
    return tuple(sorted(output, key=lambda model: (model.name, model.model_id or "")))


def register_embeddings_provider(
    name: str,
    factory: ProviderFactory,
    *,
    descriptor: ProviderDescriptor | Mapping[str, object] | None = None,
    models: Sequence[ModelDescriptor | Mapping[str, object]] = (),
) -> None:
    """Register a custom provider and optional side-effect-free catalog metadata.

    ``factory`` is retained without being called by discovery.  When metadata
    is omitted the provider is still discoverable, while provider-specific
    facts such as device, authorization, normalization, and model names remain
    explicitly unknown.
    """

    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    normalized = name.strip().lower()
    provider_descriptor = _registered_provider_descriptor(normalized, descriptor)
    model_descriptors = _registered_model_descriptors(provider_descriptor, models)
    with _PROVIDER_REGISTRY_LOCK:
        _PROVIDER_REGISTRY[normalized] = ProviderInfo(
            name=normalized,
            factory=factory,
            descriptor=provider_descriptor,
            models=model_descriptors,
        )


def _provider_name(
    provider: EmbeddingsProvider,
    *,
    requested: Optional[str] = None,
) -> str:
    explicit = str(requested or "").strip().lower()
    if explicit:
        return explicit
    tagged = str(getattr(provider, "router_provider_name", "") or "").strip().lower()
    if tagged:
        return tagged
    return provider.__class__.__name__.strip("_").lower() or "custom"


def _normalize_embedding_vectors(
    value: object,
    *,
    expected_count: int,
) -> List[List[float]]:
    """Validate provider output and normalize it to finite float vectors."""

    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise EmbeddingsRouterError("embeddings provider must return a sequence")
    if len(value) != expected_count:
        raise EmbeddingsRouterError(
            "embeddings provider returned "
            f"{len(value)} vectors for {expected_count} inputs"
        )
    if expected_count == 0:
        return []

    vectors: List[List[float]] = []
    dimension: Optional[int] = None
    for row in value:
        if hasattr(row, "tolist") and callable(getattr(row, "tolist")):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or not row:
            raise EmbeddingsRouterError(
                "each embedding must be a non-empty numeric sequence"
            )
        try:
            vector = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            raise EmbeddingsRouterError(
                "embedding values must be numeric"
            ) from exc
        if not all(math.isfinite(item) for item in vector):
            raise EmbeddingsRouterError(
                "embedding values must all be finite"
            )
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise EmbeddingsRouterError(
                "embeddings provider returned inconsistent dimensions"
            )
        vectors.append(vector)
    return vectors


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


@dataclass(frozen=True)
class _EmbeddingProviderSpec:
    name: str
    aliases: Tuple[str, ...]
    description: str
    locality: str
    device: str
    authorization: str
    normalization: str
    model_env: Tuple[str, ...] = ()
    default_model: Optional[str] = None


_BUILTIN_PROVIDER_SPECS: Tuple[_EmbeddingProviderSpec, ...] = (
    _EmbeddingProviderSpec(
        name="openrouter",
        aliases=(),
        description="OpenRouter OpenAI-compatible embeddings API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        normalization="model-dependent",
        model_env=(
            "IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_OPENROUTER_EMBEDDINGS_MODEL",
            "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
        ),
        default_model="text-embedding-3-small",
    ),
    _EmbeddingProviderSpec(
        name="hf_inference_api",
        aliases=("hf_api", "hf_inference", "huggingface_inference"),
        description="Hugging Face hosted inference embeddings API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        normalization="optional",
        model_env=(
            "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_HF_EMBEDDINGS_MODEL",
            "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
            "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
            "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
        ),
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    _EmbeddingProviderSpec(
        name="xai",
        aliases=("grok", "xai_grok"),
        description="xAI OpenAI-compatible embeddings API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        normalization="unknown",
        model_env=(
            "ipfs_accelerate_py_XAI_EMBEDDINGS_MODEL",
            "ipfs_accelerate_py_EMBEDDINGS_MODEL",
        ),
        default_model="v1",
    ),
    _EmbeddingProviderSpec(
        name="meta_ai",
        aliases=("meta", "meta-ai", "meta_llama", "meta_spark", "spark"),
        description="Meta AI OpenAI-compatible embeddings API.",
        locality="remote",
        device="provider-managed",
        authorization="required",
        normalization="unknown",
        model_env=(
            "ipfs_accelerate_py_META_AI_EMBEDDINGS_MODEL",
            "ipfs_accelerate_py_EMBEDDINGS_MODEL",
        ),
        default_model="meta-llama/Llama-3.3-70B-Instruct",
    ),
    _EmbeddingProviderSpec(
        name="gemini_cli",
        aliases=("gemini",),
        description="Gemini CLI embeddings integration.",
        locality="remote",
        device="provider-managed",
        authorization="unknown",
        normalization="unknown",
        model_env=("IPFS_ACCELERATE_PY_GEMINI_EMBEDDINGS_MODEL",),
        default_model="embedding-001",
    ),
    _EmbeddingProviderSpec(
        name="huggingface",
        aliases=("hf", "local_hf"),
        description="Local sentence-transformers or transformers embeddings.",
        locality="local",
        device="cpu,cuda",
        authorization="none",
        normalization="optional",
        model_env=(
            "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
        ),
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    _EmbeddingProviderSpec(
        name="adapter",
        aliases=("local", "local_adapter"),
        description="Dependency-injectable local transformers embeddings adapter.",
        locality="local",
        device="cpu,cuda",
        authorization="none",
        normalization="none",
        model_env=(
            "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
        ),
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    ),
    _EmbeddingProviderSpec(
        name="accelerate",
        aliases=(),
        description="Distributed ipfs_accelerate_py embeddings provider.",
        locality="distributed",
        device="runtime-selected",
        authorization="unknown",
        normalization="unknown",
        model_env=(
            "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
            "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
        ),
    ),
    _EmbeddingProviderSpec(
        name="backend_manager",
        aliases=(),
        description="Multiplexed inference backend manager embeddings provider.",
        locality="distributed",
        device="runtime-selected",
        authorization="unknown",
        normalization="unknown",
        model_env=("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",),
    ),
)
_BUILTIN_PROVIDER_SPEC_BY_NAME = {
    spec.name: spec for spec in _BUILTIN_PROVIDER_SPECS
}


def _embedding_capability(
    *,
    embedding_dimensions: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        operations=(Operation.EMBEDDING_GENERATE, Operation.BATCH),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.EMBEDDING,),
        max_context_tokens=max_context_tokens,
        max_batch_size=max_batch_size,
        embedding_dimensions=embedding_dimensions,
    )


def _model_facts(model_name: str) -> Tuple[Optional[int], Optional[int], str]:
    """Return only stable, built-in model facts; all other facts stay unknown."""

    normalized = str(model_name or "").strip().casefold()
    if normalized == "text-embedding-3-small":
        return 1536, 8191, "unit"
    if normalized == "sentence-transformers/all-minilm-l6-v2":
        return 384, 256, "optional"
    if normalized == "embedding-001":
        return 768, 2048, "unknown"
    return None, None, "unknown"


def _catalog_model_name(value: object) -> str:
    """Normalize an invocation model override into the shared name grammar."""

    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"[^a-z0-9._/-]+", "-", normalized)
    normalized = re.sub(r"/{2,}", "/", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    normalized = normalized.strip("._/-")
    if not normalized:
        normalized = "default"
    return normalized[:128].rstrip("._/-") or "default"


def _model_architecture(model_name: str) -> Optional[str]:
    normalized = str(model_name or "").casefold()
    if normalized.startswith("sentence-transformers/") or normalized.startswith(
        "meta-llama/"
    ):
        return "transformer"
    return None


def _effective_spec_model(spec: _EmbeddingProviderSpec) -> Optional[str]:
    return _coalesce_env(*spec.model_env) or spec.default_model


def _env_has_value(*names: str) -> bool:
    return bool(_coalesce_env(*names))


def _remote_provider_authorized(name: str) -> Optional[bool]:
    if name == "openrouter":
        return _env_has_value(
            "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
            "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY",
        )
    if name == "hf_inference_api":
        # Do not call _resolve_hf_api_token here: its fallback imports
        # huggingface_hub and reads its credential store, which discovery
        # deliberately avoids. With no environment token authorization is
        # therefore unknown rather than false.
        if _env_has_value(
            "IPFS_ACCELERATE_PY_HF_API_TOKEN",
            "IPFS_DATASETS_PY_HF_API_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGINGFACE_API_TOKEN",
            "HF_TOKEN",
        ):
            return True
        return None
    if name == "xai":
        return _env_has_value("XAI_API_KEY", "ipfs_accelerate_py_XAI_API_KEY")
    if name == "meta_ai":
        return _env_has_value(
            "META_AI_API_KEY", "ipfs_accelerate_py_META_AI_API_KEY"
        )
    return None


def _builtin_provider_state(
    spec: _EmbeddingProviderSpec,
) -> Tuple[LifecycleState, OperationalState]:
    authorized = _remote_provider_authorized(spec.name)
    if authorized is not None:
        return (
            LifecycleState.CONFIGURED if authorized else LifecycleState.DECLARED,
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
        enabled = _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"))
        return (
            LifecycleState.CONFIGURED if enabled else LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=enabled,
                authorized=None,
                reachable=None,
                healthy=None,
                routable=None,
            ),
        )
    if spec.name == "accelerate":
        enabled = _coalesce_env(
            "IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE",
            "IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE",
        )
        configured = None if not enabled else _truthy(enabled)
        return (
            LifecycleState.CONFIGURED
            if configured is True
            else LifecycleState.DECLARED,
            OperationalState(
                known=True,
                configured=configured,
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
    spec: _EmbeddingProviderSpec,
) -> ProviderDescriptor:
    model_name = _effective_spec_model(spec)
    dimensions, context_tokens, _ = _model_facts(model_name or "")
    lifecycle, state = _builtin_provider_state(spec)
    return ProviderDescriptor(
        name=spec.name,
        aliases=spec.aliases,
        description=spec.description,
        capabilities=(
            _embedding_capability(
                embedding_dimensions=dimensions,
                max_context_tokens=context_tokens,
            ),
        ),
        lifecycle=lifecycle,
        state=state,
        provenance=(Provenance(source="embeddings_router.static"),),
        labels={
            "access_requirement": spec.authorization,
            "batching": "supported",
            "device": spec.device,
            "input_types": "text",
            "locality": spec.locality,
            "normalization": spec.normalization,
        },
    )


def _provider_descriptors_by_name() -> Dict[str, ProviderDescriptor]:
    descriptors = {
        spec.name: _builtin_provider_descriptor(spec)
        for spec in _BUILTIN_PROVIDER_SPECS
    }
    with _PROVIDER_REGISTRY_LOCK:
        registered = tuple(_PROVIDER_REGISTRY.values())
    for info in registered:
        # Registration has precedence over a built-in with the same public
        # name, matching provider invocation.
        descriptors[info.name] = info.descriptor or _registered_provider_descriptor(
            info.name,
            None,
        )
    return descriptors


def list_providers() -> List[ProviderDescriptor]:
    """List provider descriptors without resolving or constructing providers."""

    return [
        descriptor
        for _, descriptor in sorted(_provider_descriptors_by_name().items())
    ]


def _canonical_provider_name(name: str) -> str:
    requested = str(name or "").strip().lower()
    if not requested:
        raise ValueError("Embeddings provider name must be non-empty")
    descriptors = _provider_descriptors_by_name()
    if requested in descriptors:
        return requested
    matches = sorted(
        descriptor.name
        for descriptor in descriptors.values()
        if requested in descriptor.aliases
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous embeddings provider alias {name!r}: {', '.join(matches)}"
        )
    raise ValueError(f"Unknown embeddings provider: {name}")


def get_provider_descriptor(name: str) -> ProviderDescriptor:
    """Return the descriptor for a provider canonical name or alias."""

    canonical = _canonical_provider_name(name)
    return _provider_descriptors_by_name()[canonical]


def _model_descriptor(
    provider: ProviderDescriptor,
    model_name: str,
    *,
    normalization: Optional[str] = None,
) -> ModelDescriptor:
    dimensions, context_tokens, known_normalization = _model_facts(model_name)
    normalized = normalization or known_normalization
    return ModelDescriptor(
        provider_id=provider.provider_id,
        name=_catalog_model_name(model_name),
        architecture=_model_architecture(model_name),
        capabilities=(
            _embedding_capability(
                embedding_dimensions=dimensions,
                max_context_tokens=context_tokens,
            ),
        ),
        lifecycle=provider.lifecycle,
        state=provider.state,
        provenance=(Provenance(source="embeddings_router.static"),),
        labels={
            "access_requirement": dict(provider.labels).get(
                "access_requirement", "unknown"
            ),
            "batching": "supported",
            "device": dict(provider.labels).get("device", "unknown"),
            "input_types": "text",
            "locality": dict(provider.labels).get("locality", "unknown"),
            "normalization": normalized,
            "invocation_model": model_name,
        },
    )


def _models_for_provider(provider_name: str) -> Tuple[ModelDescriptor, ...]:
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
    normalization = "none" if spec.normalization == "none" else None
    return (
        _model_descriptor(
            provider,
            model_name,
            normalization=normalization,
        ),
    )


def list_models(provider: Optional[str] = None) -> List[ModelDescriptor]:
    """List statically known or registered model descriptors."""

    if provider is not None:
        provider_names = (_canonical_provider_name(provider),)
    else:
        provider_names = tuple(sorted(_provider_descriptors_by_name()))
    models = [
        model
        for provider_name in provider_names
        for model in _models_for_provider(provider_name)
    ]
    return sorted(
        models,
        key=lambda model: (model.provider_id, model.name, model.model_id or ""),
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
    if provider:
        return _canonical_provider_name(provider)

    preferred = _coalesce_env(
        "IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER",
        "IPFS_DATASETS_PY_EMBEDDINGS_PROVIDER",
    )
    if preferred:
        try:
            return _canonical_provider_name(preferred)
        except ValueError:
            # Invocation currently falls through when its environment override
            # is unknown, so discovery does the same.
            pass

    resolved_deps = deps or get_default_router_deps()
    managers = getattr(resolved_deps, "accelerate_managers", {})
    if isinstance(managers, Mapping) and managers.get("embeddings_router") is not None:
        return "accelerate"
    if (
        _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"))
        and getattr(resolved_deps, "backend_manager", None) is not None
    ):
        return "backend_manager"

    for name in ("openrouter", "hf_inference_api", "xai", "meta_ai"):
        if _remote_provider_authorized(name):
            return name
    if _module_available(
        "ipfs_accelerate_py.cli_integrations.gemini_cli_integration"
    ):
        return "gemini_cli"
    if _module_available("transformers"):
        return "huggingface"
    raise RuntimeError(
        "No embeddings provider is statically resolvable for the requested constraints"
    )


def resolve_model(
    model_name: Optional[str] = None,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    deps: Optional[RouterDeps] = None,
    **constraints: object,
) -> ModelDescriptor:
    """Resolve an embedding model using the router's explicit selection rules.

    Resolution is metadata-only.  It never calls a provider factory.  Unknown
    model overrides remain valid because embedding generation forwards them to
    the selected provider; their dimension and token limits remain ``None``.
    """

    _ = device  # Providers receive this hint but current selection ignores it.
    if model is not None:
        if model_name is not None and str(model_name) != str(model):
            raise ValueError("model and model_name specify different values")
        model_name = str(model)
    operation = constraints.pop("operation", Operation.EMBEDDING_GENERATE)
    if constraints:
        unknown = ", ".join(sorted(str(key) for key in constraints))
        raise TypeError(f"Unknown embedding resolution constraints: {unknown}")
    operation_value = (
        operation.value if isinstance(operation, Operation) else str(operation)
    )
    if operation_value not in {
        Operation.EMBEDDING_GENERATE.value,
        Operation.BATCH.value,
    }:
        raise ValueError(
            f"Embeddings router does not support operation {operation_value!r}"
        )

    provider_name = _select_discovery_provider(provider, deps=deps)
    provider_descriptor = get_provider_descriptor(provider_name)
    known_models = _models_for_provider(provider_name)
    requested_model = str(model_name or "").strip()
    if not requested_model:
        if not known_models:
            raise ValueError(
                f"Embeddings provider {provider_name!r} has no known default model; "
                "specify model_name explicitly"
            )
        return known_models[0]

    requested_key = requested_model.casefold()
    for descriptor in known_models:
        labels = dict(descriptor.labels)
        router_name = labels.get(
            "invocation_model",
            labels.get("router_model_name", descriptor.name),
        )
        if requested_key in {
            descriptor.name.casefold(),
            str(router_name).casefold(),
            *(alias.casefold() for alias in descriptor.aliases),
        }:
            return descriptor
    return _model_descriptor(provider_descriptor, requested_model)


def get_catalog_snapshot() -> CatalogSnapshot:
    """Project router discovery records into a deterministic catalog snapshot."""

    providers = tuple(list_providers())
    models = tuple(list_models())
    provider_by_id = {provider.provider_id: provider for provider in providers}
    bindings = tuple(
        RouterBinding(
            router="embeddings_router",
            provider_id=model.provider_id,
            model_id=model.model_id,
            operations=(Operation.EMBEDDING_GENERATE, Operation.BATCH),
            priority=index,
            state=provider_by_id[model.provider_id].state,
            provenance=(Provenance(source="embeddings_router.static"),),
            labels={
                "invocation_model": dict(model.labels).get(
                    "invocation_model",
                    dict(model.labels).get("router_model_name", model.name),
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


def _resolve_hf_api_token() -> str:
    token = _coalesce_env(
        "IPFS_ACCELERATE_PY_HF_API_TOKEN",
        "IPFS_DATASETS_PY_HF_API_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
    )
    if token:
        return token
    try:
        hub = importlib.import_module("huggingface_hub")
        getter = getattr(hub, "get_token", None)
        return str(getter() if callable(getter) else "").strip()
    except Exception:
        return ""


def _hf_token_fingerprint() -> str:
    token = _resolve_hf_api_token()
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] if token else ""


def _resolve_hf_bill_to(*, kwargs: Optional[dict[str, object]] = None) -> str:
    if kwargs:
        for key in ("hf_bill_to", "bill_to", "organization", "org"):
            value = kwargs.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return _coalesce_env(
        "OPENROUTER_HF_BILL_TO",
        "IPFS_ACCELERATE_PY_HF_BILL_TO",
        "IPFS_DATASETS_PY_HF_BILL_TO",
        "HUGGINGFACE_BILL_TO",
        "HF_BILL_TO",
        "HF_ORGANIZATION",
        "HUGGINGFACE_ORG",
    )


def _is_hf_inference_provider_name(name: Optional[str]) -> bool:
    return str(name or "").strip().lower() in {
        "hf_api",
        "hf_inference",
        "hf_inference_api",
        "huggingface_inference",
    }


def _is_hf_embedding_compatibility_error(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    if "http 402" in message:
        return False
    return any(
        token in message
        for token in (
            "http 404",
            "not found",
            "missing 1 required positional argument: 'sentences'",
            "pipeline",
            "task",
            "unsupported",
            "does not support",
        )
    )


def _hf_embeddings_default_fallback_models() -> list[str]:
    return [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "thenlper/gte-small",
    ]


def _hf_dynamic_model_discovery_enabled(*, kwargs: dict[str, object]) -> bool:
    raw = kwargs.get("hf_dynamic_model_discovery")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_DYNAMIC_MODEL_DISCOVERY",
            "IPFS_DATASETS_PY_HF_DYNAMIC_MODEL_DISCOVERY",
        ) or "1"
    return _truthy(str(raw))


def _hf_embeddings_discovery_limit(*, kwargs: dict[str, object]) -> int:
    raw = kwargs.get("hf_embeddings_discovery_limit")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_DISCOVERY_LIMIT",
            "IPFS_DATASETS_PY_HF_EMBEDDINGS_DISCOVERY_LIMIT",
        ) or "20"
    try:
        return max(1, int(raw))
    except Exception:
        return 20


def _hf_embeddings_discovery_tags(*, kwargs: dict[str, object]) -> list[str]:
    raw = kwargs.get("hf_embeddings_discovery_tags")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_DISCOVERY_TAGS",
            "IPFS_DATASETS_PY_HF_EMBEDDINGS_DISCOVERY_TAGS",
        ) or "feature-extraction,sentence-similarity"
    return [item.strip() for item in str(raw).split(",") if item.strip()]


@lru_cache(maxsize=32)
def _discover_hf_models_for_pipeline(
    *,
    pipeline_tag: str,
    limit: int,
) -> tuple[str, ...]:
    try:
        hub = importlib.import_module("huggingface_hub")
        api_cls = getattr(hub, "HfApi", None)
        if api_cls is None:
            return ()
        models = api_cls().list_models(
            inference_provider="hf-inference",
            pipeline_tag=pipeline_tag,
            limit=max(1, int(limit)),
            token=_resolve_hf_api_token() or None,
        )
        output: list[str] = []
        for item in models:
            model_id = str(getattr(item, "id", "") or "").strip()
            if model_id and model_id not in output:
                output.append(model_id)
        return tuple(output)
    except Exception:
        return ()


def _hf_embeddings_fallback_models(*, kwargs: dict[str, object]) -> list[str]:
    raw = kwargs.get("hf_model_fallbacks")
    if raw is None:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_FALLBACK_MODELS",
            "IPFS_DATASETS_PY_HF_EMBEDDINGS_FALLBACK_MODELS",
        )
    if str(raw or "").strip():
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    models: list[str] = []
    if _hf_dynamic_model_discovery_enabled(kwargs=kwargs):
        for tag in _hf_embeddings_discovery_tags(kwargs=kwargs):
            for model_id in _discover_hf_models_for_pipeline(
                pipeline_tag=tag,
                limit=_hf_embeddings_discovery_limit(kwargs=kwargs),
            ):
                if model_id not in models:
                    models.append(model_id)
    for model_id in _hf_embeddings_default_fallback_models():
        if model_id not in models:
            models.append(model_id)
    return models


def _get_openrouter_provider() -> Optional[EmbeddingsProvider]:
    """Get OpenRouter embeddings provider."""
    credential = _coalesce_env(
        "IPFS_ACCELERATE_PY_OPENROUTER_API_KEY",
        "IPFS_DATASETS_PY_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
    )
    if not credential:
        return None

    base_url = (
        _coalesce_env(
            "IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL",
            "IPFS_DATASETS_PY_OPENROUTER_BASE_URL",
        )
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    app_title = os.getenv("OPENROUTER_APP_TITLE")

    class _OpenRouterEmbeddingsProvider:
        router_provider_name = "openrouter"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            _ = device
            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL")
                or os.getenv("IPFS_DATASETS_PY_OPENROUTER_EMBEDDINGS_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL")
                or os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_MODEL")
                or "text-embedding-3-small"
            )
            inputs = list(texts)
            payload = {"model": model, "input": inputs}

            headers = {
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if referer:
                headers["HTTP-Referer"] = referer
            if app_title:
                headers["X-Title"] = app_title
            bill_to = _resolve_hf_bill_to(kwargs=dict(kwargs))
            if bill_to:
                headers["X-HF-Bill-To"] = bill_to

            req = urllib.request.Request(
                f"{base_url}/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers
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

            items = data.get("data")
            if not isinstance(items, list):
                raise RuntimeError("OpenRouter embeddings response missing data")
            embeddings: List[List[float]] = []
            for item in items:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise RuntimeError("OpenRouter embeddings item missing embedding")
                vec = item["embedding"]
                if not isinstance(vec, list):
                    raise RuntimeError("OpenRouter embedding must be a list")
                embeddings.append([float(x) for x in vec])
            if len(embeddings) != len(inputs):
                # Best-effort: still return what we got if non-empty.
                if embeddings:
                    return embeddings
                raise RuntimeError("OpenRouter returned no embeddings")
            return embeddings

    return _OpenRouterEmbeddingsProvider()


def _normalize_hf_embedding_payload(data: object) -> List[List[float]]:
    if isinstance(data, dict):
        if isinstance(data.get("error"), str):
            raise RuntimeError(f"HF Inference API error: {data.get('error')}")
        if isinstance(data.get("embeddings"), list):
            data = data["embeddings"]
    if not isinstance(data, list) or not data:
        raise RuntimeError("HF Inference API embeddings response missing vectors")
    if isinstance(data[0], (int, float)):
        return [[float(value) for value in data]]
    vectors: List[List[float]] = []
    for item in data:
        if isinstance(item, list):
            vectors.append([float(value) for value in item])
        elif isinstance(item, dict) and isinstance(item.get("embedding"), list):
            vectors.append([float(value) for value in item["embedding"]])
        else:
            raise RuntimeError(
                "HF Inference API returned malformed embedding vector"
            )
    return vectors


def _get_hf_inference_api_provider() -> Optional[EmbeddingsProvider]:
    api_token = _resolve_hf_api_token()
    if not api_token:
        return None
    base_url = (
        _coalesce_env(
            "IPFS_ACCELERATE_PY_HF_INFERENCE_BASE_URL",
            "IPFS_DATASETS_PY_HF_INFERENCE_BASE_URL",
        )
        or "https://router.huggingface.co/hf-inference/models"
    ).rstrip("/")

    class _HFInferenceAPIEmbeddingsProvider:
        router_provider_name = "hf_inference_api"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            _ = device
            model = model_name or _coalesce_env(
                "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_MODEL",
                "IPFS_DATASETS_PY_HF_EMBEDDINGS_MODEL",
                "IPFS_ACCELERATE_PY_HF_INFERENCE_MODEL",
                "IPFS_DATASETS_PY_HF_INFERENCE_MODEL",
                "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
                "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
            ) or "sentence-transformers/all-MiniLM-L6-v2"
            inputs = list(texts)
            if not inputs:
                return []
            timeout = float(kwargs.get("timeout", 120))
            wait_for_model_raw = kwargs.get("wait_for_model", True)
            use_cache_raw = kwargs.get("use_cache", True)
            wait_for_model = (
                _truthy(wait_for_model_raw)
                if isinstance(wait_for_model_raw, str)
                else bool(wait_for_model_raw)
            )
            use_cache = (
                _truthy(use_cache_raw)
                if isinstance(use_cache_raw, str)
                else bool(use_cache_raw)
            )
            payload: dict[str, object] = {
                "inputs": inputs,
                "options": {
                    "wait_for_model": wait_for_model,
                    "use_cache": use_cache,
                },
            }
            for key in ("truncate", "truncation", "normalize"):
                value = kwargs.get(key)
                if value is not None:
                    payload[key] = value
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            bill_to = _resolve_hf_bill_to(kwargs=dict(kwargs))
            if bill_to:
                headers["X-HF-Bill-To"] = bill_to
            request = urllib.request.Request(
                f"{base_url}/{model}",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = (
                    exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                )
                raise RuntimeError(
                    f"HF Inference API HTTP {exc.code}: {detail or exc.reason}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"HF Inference API request failed: {exc}"
                ) from exc
            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError(
                    "HF Inference API returned invalid JSON"
                ) from exc
            vectors = _normalize_hf_embedding_payload(data)
            if not vectors:
                raise RuntimeError("HF Inference API returned no embeddings")
            return vectors

    return _HFInferenceAPIEmbeddingsProvider()


def _get_xai_embeddings_provider() -> Optional[EmbeddingsProvider]:
    """Get xAI Grok embeddings provider via OpenAI-compatible endpoint."""
    api_key = (
        os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("ipfs_accelerate_py_XAI_API_KEY", "").strip()
    )
    if not api_key:
        return None

    base_url = os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")

    class _XAIEmbeddingsProvider:
        router_provider_name = "xai"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            _ = device
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_XAI_EMBEDDINGS_MODEL")
                or os.getenv("ipfs_accelerate_py_EMBEDDINGS_MODEL")
                or "v1"
            )
            inputs = list(texts)
            payload = {"model": model, "input": inputs}

            req = urllib.request.Request(
                f"{base_url}/embeddings",
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
                raise RuntimeError(f"xAI embeddings request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("xAI returned invalid JSON") from exc

            items = data.get("data")
            if not isinstance(items, list):
                raise RuntimeError("xAI embeddings response missing data")
            embeddings: List[List[float]] = []
            for item in items:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise RuntimeError("xAI embeddings item missing embedding")
                vec = item["embedding"]
                if not isinstance(vec, list):
                    raise RuntimeError("xAI embedding must be a list")
                embeddings.append([float(x) for x in vec])
            if not embeddings:
                raise RuntimeError("xAI returned no embeddings")
            return embeddings

    return _XAIEmbeddingsProvider()


def _get_meta_ai_embeddings_provider() -> Optional[EmbeddingsProvider]:
    """Get Meta AI (Llama) embeddings provider via OpenAI-compatible endpoint."""
    api_key = (
        os.environ.get("META_AI_API_KEY", "").strip()
        or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY", "").strip()
    )
    if not api_key:
        return None

    base_url = os.getenv("ipfs_accelerate_py_META_AI_BASE_URL", "https://api.llamameta.net/v1").rstrip("/")

    class _MetaAIEmbeddingsProvider:
        router_provider_name = "meta_ai"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            _ = device
            model = (
                model_name
                or os.getenv("ipfs_accelerate_py_META_AI_EMBEDDINGS_MODEL")
                or os.getenv("ipfs_accelerate_py_EMBEDDINGS_MODEL")
                or "meta-llama/Llama-3.3-70B-Instruct"
            )
            inputs = list(texts)
            payload = {"model": model, "input": inputs}

            req = urllib.request.Request(
                f"{base_url}/embeddings",
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
                raise RuntimeError(f"Meta AI embeddings request failed: {exc}") from exc

            try:
                data = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("Meta AI returned invalid JSON") from exc

            items = data.get("data")
            if not isinstance(items, list):
                raise RuntimeError("Meta AI embeddings response missing data")
            embeddings: List[List[float]] = []
            for item in items:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise RuntimeError("Meta AI embeddings item missing embedding")
                vec = item["embedding"]
                if not isinstance(vec, list):
                    raise RuntimeError("Meta AI embedding must be a list")
                embeddings.append([float(x) for x in vec])
            if not embeddings:
                raise RuntimeError("Meta AI returned no embeddings")
            return embeddings

    return _MetaAIEmbeddingsProvider()


def _get_gemini_cli_provider() -> Optional[EmbeddingsProvider]:
    """Get Gemini CLI embeddings provider using existing integration."""
    try:
        from ipfs_accelerate_py.cli_integrations.gemini_cli_integration import GeminiCLIIntegration
    except Exception:
        return None

    class _GeminiCLIEmbeddingsProvider:
        router_provider_name = "gemini_cli"

        def __init__(self):
            self._client = None

        def _get_client(self):
            if self._client is None:
                try:
                    self._client = GeminiCLIIntegration()
                except Exception as e:
                    logger.debug(f"Failed to initialize Gemini integration: {e}")
                    return None
            return self._client

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            client = self._get_client()
            if client is None:
                raise RuntimeError("Gemini integration not available")
            
            # Check if the client has an embed_texts method
            if not hasattr(client, 'embed_texts'):
                # Fallback: Use generate_embeddings if available
                if hasattr(client, 'generate_embeddings'):
                    inputs = list(texts)
                    result = client.generate_embeddings(
                        texts=inputs,
                        model=model_name or os.getenv("IPFS_ACCELERATE_PY_GEMINI_EMBEDDINGS_MODEL", "embedding-001")
                    )
                    if result.get("success") and "embeddings" in result:
                        return result["embeddings"]
                raise RuntimeError("Gemini integration does not support embeddings")
            
            inputs = list(texts)
            return client.embed_texts(inputs, model_name=model_name, device=device, **kwargs)

    return _GeminiCLIEmbeddingsProvider()


def _get_huggingface_provider() -> Optional[EmbeddingsProvider]:
    """Get HuggingFace embeddings provider using transformers."""
    try:
        import transformers
    except ImportError:
        return None

    class _HuggingFaceEmbeddingsProvider:
        router_provider_name = "huggingface"

        def __init__(self):
            self._models = {}
            self._models_lock = threading.Lock()

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            model = model_name or _coalesce_env(
                "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
                "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
            ) or "sentence-transformers/all-MiniLM-L6-v2"
            device_str = device or _coalesce_env(
                "IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE",
                "IPFS_DATASETS_PY_EMBEDDINGS_DEVICE",
            ) or "cpu"
            
            # Get or create model
            cache_key = f"{model}::{device_str}"
            with self._models_lock:
                if cache_key not in self._models:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._models[cache_key] = SentenceTransformer(
                            model,
                            device=device_str,
                        )
                    except ImportError:
                        # Fall back to transformers directly
                        from transformers import AutoTokenizer, AutoModel
                        import torch

                        tokenizer = AutoTokenizer.from_pretrained(model)
                        model_obj = AutoModel.from_pretrained(model)
                        if device_str == "cuda" and torch.cuda.is_available():
                            model_obj = model_obj.to("cuda")
                        self._models[cache_key] = (
                            tokenizer,
                            model_obj,
                            device_str,
                        )

                model_obj = self._models[cache_key]
            inputs = list(texts)
            
            # Use SentenceTransformer if available
            if hasattr(model_obj, 'encode'):
                encode_options = {
                    key: kwargs[key]
                    for key in (
                        "batch_size",
                        "show_progress_bar",
                        "normalize_embeddings",
                        "precision",
                    )
                    if key in kwargs
                }
                embeddings = model_obj.encode(
                    inputs,
                    convert_to_numpy=True,
                    **encode_options,
                )
                return [emb.tolist() for emb in embeddings]
            
            # Otherwise use transformers directly
            tokenizer, transformer_model, dev = model_obj
            import torch
            
            embeddings = []
            for text in inputs:
                encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                if dev == "cuda" and torch.cuda.is_available():
                    encoded = {k: v.to("cuda") for k, v in encoded.items()}
                
                with torch.no_grad():
                    output = transformer_model(**encoded)
                    # Use [CLS] token embedding or mean pooling
                    embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                    embeddings.append(embedding.tolist())
            
            return embeddings

    return _HuggingFaceEmbeddingsProvider()


def _get_local_adapter_provider(
    *,
    deps: Optional[RouterDeps] = None,
) -> Optional[EmbeddingsProvider]:
    """Build the dependency-injectable local transformers adapter."""

    resolved_deps = deps or get_default_router_deps()

    def _resolve_module(name: str) -> object | None:
        cache_key = f"pip::{name}"
        getter = getattr(resolved_deps, "get_cached", None)
        cached = getter(cache_key) if callable(getter) else None
        if cached is not None:
            return cached
        try:
            module = importlib.import_module(name)
        except Exception:
            return None
        setter = getattr(resolved_deps, "set_cached", None)
        if callable(setter):
            setter(cache_key, module)
        return module

    torch_module = _resolve_module("torch")
    transformers_module = _resolve_module("transformers")
    if torch_module is None or transformers_module is None:
        return None

    auto_tokenizer = getattr(transformers_module, "AutoTokenizer", None)
    auto_model = getattr(transformers_module, "AutoModel", None)
    if auto_tokenizer is None or auto_model is None:
        return None

    class _LocalAdapterProvider:
        router_provider_name = "adapter"

        def __init__(self) -> None:
            self._runtimes: dict[tuple[str, str], tuple[object, object]] = {}

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            inputs = list(texts)
            if not inputs:
                return []
            model = model_name or _coalesce_env(
                "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
                "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
            ) or "sentence-transformers/all-MiniLM-L6-v2"
            device_name = device or _coalesce_env(
                "IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE",
                "IPFS_DATASETS_PY_EMBEDDINGS_DEVICE",
            )
            if not device_name:
                try:
                    device_name = (
                        "cuda"
                        if bool(torch_module.cuda.is_available())
                        else "cpu"
                    )
                except Exception:
                    device_name = "cpu"

            runtime_key = (model, device_name)
            runtime = self._runtimes.get(runtime_key)
            if runtime is None:
                tokenizer = auto_tokenizer.from_pretrained(model)
                model_object = auto_model.from_pretrained(model)
                move_model = getattr(model_object, "to", None)
                if callable(move_model):
                    model_object = move_model(device_name)
                evaluate = getattr(model_object, "eval", None)
                if callable(evaluate):
                    evaluate()
                runtime = (tokenizer, model_object)
                self._runtimes[runtime_key] = runtime

            tokenizer, model_object = runtime
            tokenizer_kwargs: dict[str, object] = {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            max_length = kwargs.get("max_length")
            if max_length is not None:
                tokenizer_kwargs["max_length"] = int(max_length)
            encoded = tokenizer(inputs, **tokenizer_kwargs)
            if isinstance(encoded, dict):
                encoded = {
                    key: (
                        value.to(device_name)
                        if callable(getattr(value, "to", None))
                        else value
                    )
                    for key, value in encoded.items()
                }
            no_grad = getattr(torch_module, "no_grad", None)
            if not callable(no_grad):
                raise RuntimeError("torch.no_grad is unavailable")
            with no_grad():
                output = model_object(**encoded)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise RuntimeError(
                        "transformers model did not return last_hidden_state"
                    )
                pooled = hidden.mean(dim=1)
            detach = getattr(pooled, "detach", None)
            if callable(detach):
                pooled = detach()
            cpu = getattr(pooled, "cpu", None)
            if callable(cpu):
                pooled = cpu()
            tolist = getattr(pooled, "tolist", None)
            values = tolist() if callable(tolist) else pooled
            if (
                len(inputs) == 1
                and isinstance(values, list)
                and values
                and isinstance(values[0], (int, float))
            ):
                values = [values]
            return _normalize_embedding_vectors(
                values,
                expected_count=len(inputs),
            )

    return _LocalAdapterProvider()


def _get_accelerate_provider(deps: RouterDeps) -> Optional[EmbeddingsProvider]:
    enable_value = (
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_IPFS_ACCELERATE")
        or os.getenv("IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE")
    )
    if enable_value and not _truthy(enable_value):
        return None
    manager_factory = getattr(deps, "get_accelerate_manager", None)
    if not callable(manager_factory):
        return None
    try:
        manager = manager_factory(
            purpose="embeddings_router",
            enable_distributed=True,
            resources={"purpose": "embeddings_router"},
        )
    except Exception:
        return None
    if manager is None:
        return None

    class _AccelerateEmbeddingsProvider:
        router_provider_name = "accelerate"

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            payload = {"texts": list(texts), "device": device, **kwargs}
            result = manager.run_inference(
                model_name
                or _coalesce_env(
                    "IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL",
                    "IPFS_DATASETS_PY_EMBEDDINGS_MODEL",
                ),
                payload,
                task_type="embedding",
            )
            embedded = result.get("embeddings")
            if isinstance(embedded, list):
                return [[float(value) for value in row] for row in embedded]
            raise RuntimeError(
                "ipfs_accelerate_py provider did not return embeddings"
            )

    return _AccelerateEmbeddingsProvider()


def _get_backend_manager_provider(deps: RouterDeps) -> Optional[EmbeddingsProvider]:
    """Get provider that uses InferenceBackendManager for distributed/multiplexed inference."""
    if not _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")):
        return None

    try:
        manager = deps.get_backend_manager(
            purpose="embeddings_router",
            enable_health_checks=True,
            load_balancing_strategy=os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_LOAD_BALANCING", "round_robin"),
        )
        if manager is None:
            return None

        class _BackendManagerEmbeddingsProvider:
            router_provider_name = "backend_manager"

            def embed_texts(
                self,
                texts: Iterable[str],
                *,
                model_name: Optional[str] = None,
                device: Optional[str] = None,
                **kwargs: object,
            ) -> List[List[float]]:
                # Select backend for embedding task
                backend = manager.select_backend_for_task(
                    task="text-embedding",
                    model=model_name or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL", ""),
                    protocol="any"
                )
                
                if backend is None:
                    raise RuntimeError("No available backend for text-embedding")
                
                # Execute inference via backend
                inputs = list(texts)
                payload = {
                    "texts": inputs,
                    "device": device,
                    **kwargs
                }
                
                result = manager.execute_inference(
                    backend_id=backend["id"],
                    task="text-embedding",
                    payload=payload
                )
                
                # Extract embeddings from result
                embeddings = result.get("embeddings")
                if isinstance(embeddings, list):
                    return [[float(x) for x in row] for row in embeddings]
                raise RuntimeError("Backend manager provider did not return embeddings")

        return _BackendManagerEmbeddingsProvider()
    except Exception as e:
        logger.debug(f"Backend manager provider unavailable: {e}")
        return None


def _provider_cache_key() -> tuple:
    return (
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER", "").strip(),
        os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_PROVIDER", "").strip(),
        os.getenv("IPFS_DATASETS_PY_ENABLE_IPFS_ACCELERATE", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_OPENROUTER_BASE_URL", "").strip(),
        _hf_token_fingerprint(),
        os.getenv("IPFS_ACCELERATE_PY_HF_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_HF_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_HF_INFERENCE_BASE_URL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_HF_INFERENCE_BASE_URL", "").strip(),
        os.getenv("XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_XAI_BASE_URL", "").strip(),
        os.getenv("META_AI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_API_KEY", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("ipfs_accelerate_py_META_AI_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_BACKEND", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE", "").strip(),
        os.getenv("IPFS_DATASETS_PY_EMBEDDINGS_DEVICE", "").strip(),
    )


def _deps_provider_cache_key(preferred: Optional[str], cache_key: tuple) -> str:
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    return f"embeddings_provider::{(preferred or '').strip().lower()}::{digest}"


@lru_cache(maxsize=32)
def _resolve_provider_cached(preferred: Optional[str], cache_key: tuple) -> EmbeddingsProvider:
    _ = cache_key
    return _resolve_provider_uncached(preferred, deps=get_default_router_deps())


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[EmbeddingsProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key == "openrouter":
        return _get_openrouter_provider()
    if _is_hf_inference_provider_name(key):
        return _get_hf_inference_api_provider()
    if key in {"xai", "grok", "xai_grok"}:
        return _get_xai_embeddings_provider()
    if key in {"meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"}:
        return _get_meta_ai_embeddings_provider()
    if key in {"gemini", "gemini_cli"}:
        return _get_gemini_cli_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_huggingface_provider()
    if key in {"adapter", "local", "local_adapter"}:
        return _get_local_adapter_provider(deps=deps)
    if key == "accelerate":
        return _get_accelerate_provider(deps) or _get_backend_manager_provider(deps)
    if key == "backend_manager":
        return _get_backend_manager_provider(deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> EmbeddingsProvider:
    if preferred:
        preferred_key = preferred.strip().lower()
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
        raise ValueError(f"Unknown embeddings provider: {preferred}")

    # 1) Registered providers can opt-in via env ordering if desired.
    preferred_env = _coalesce_env(
        "IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER",
        "IPFS_DATASETS_PY_EMBEDDINGS_PROVIDER",
    ).lower()
    if preferred_env:
        try:
            preferred_env = _canonical_provider_name(preferred_env)
        except ValueError:
            pass
        with _PROVIDER_REGISTRY_LOCK:
            info = _PROVIDER_REGISTRY.get(preferred_env)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_env, deps=deps)
        if builtin is not None:
            return builtin

    # 2) Optional injected accelerator or backend manager provider.
    accelerate_provider = _get_accelerate_provider(deps)
    if accelerate_provider is not None:
        return accelerate_provider
    backend_manager_provider = _get_backend_manager_provider(deps)
    if backend_manager_provider is not None:
        return backend_manager_provider

    # Try optional providers if available.
    for name in [
        "openrouter",
        "hf_inference_api",
        "xai",
        "meta_ai",
        "gemini_cli",
    ]:
        candidate = _builtin_provider_by_name(name, deps=deps)
        if candidate is not None:
            return candidate

    # 3) Local HuggingFace fallback.
    hf_provider = _get_huggingface_provider()
    if hf_provider is not None:
        return hf_provider

    raise RuntimeError(
        "No embeddings provider available. Install `sentence-transformers` or `transformers` or register a custom provider."
    )


def get_embeddings_provider(
    provider: Optional[str] = None,
    *,
    deps: Optional[RouterDeps] = None,
    use_cache: Optional[bool] = None,
) -> EmbeddingsProvider:
    """Resolve an embeddings provider with optional dependency injection."""

    resolved_deps = deps or get_default_router_deps()
    cache_ok = _cache_enabled() if use_cache is None else bool(use_cache)

    if not cache_ok:
        return _resolve_provider_uncached(provider, deps=resolved_deps)

    if deps is not None:
        cache_key = _provider_cache_key()
        deps_key = _deps_provider_cache_key(provider, cache_key)
        cached = resolved_deps.get_cached(deps_key)
        if cached is not None:
            return cached
        return resolved_deps.set_cached(deps_key, _resolve_provider_uncached(provider, deps=resolved_deps))

    return _resolve_provider_cached(provider, _provider_cache_key())


def embed_texts(
    texts: Iterable[str],
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[EmbeddingsProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> List[List[float]]:
    """Generate validated embeddings while preserving input order."""

    started = time.perf_counter()
    resolved_deps = deps or get_default_router_deps()
    inputs = list(texts)
    if any(not isinstance(text, str) for text in inputs):
        raise TypeError("texts must contain only strings")
    if not inputs:
        _set_last_embedding_trace(
            status="ok",
            provider_requested=str(provider or ""),
            provider_used="",
            model_name=str(model_name or ""),
            device=str(device or ""),
            input_count=0,
            output_count=0,
            dimension=0,
            cache_hits=0,
            cache_misses=0,
            fallback_used=False,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return []

    try:
        backend = provider_instance or get_embeddings_provider(
            provider,
            deps=resolved_deps,
        )
    except Exception as exc:
        _set_last_embedding_trace(
            status="error",
            provider_requested=str(provider or ""),
            provider_used="",
            model_name=str(model_name or ""),
            device=str(device or ""),
            input_count=len(inputs),
            output_count=0,
            dimension=0,
            cache_hits=0,
            cache_misses=len(inputs),
            fallback_used=False,
            error_type=type(exc).__name__,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        raise

    provider_used = _provider_name(backend, requested=provider)
    cache_enabled = _response_cache_enabled()
    cached_vectors: List[Optional[List[float]]] = [None] * len(inputs)
    missing_indices: List[int] = []

    if cache_enabled:
        for index, text in enumerate(inputs):
            try:
                cache_key = _response_cache_key(
                    provider=provider_used,
                    model_name=model_name,
                    device=device,
                    text=text,
                    kwargs=dict(kwargs),
                )
                getter = getattr(resolved_deps, "get_cached_or_remote", None)
                cached = (
                    getter(cache_key)
                    if callable(getter)
                    else resolved_deps.get_cached(cache_key)
                )
                cached_vectors[index] = _normalize_embedding_vectors(
                    [cached],
                    expected_count=1,
                )[0]
            except Exception:
                missing_indices.append(index)
    else:
        missing_indices = list(range(len(inputs)))

    cache_hits = len(inputs) - len(missing_indices)
    fallback_used = False
    used_model_name = model_name

    def _cache_vectors(
        source_texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        cache_provider: str,
        cache_model_name: Optional[str],
    ) -> None:
        if not cache_enabled:
            return
        for text, vector in zip(source_texts, vectors):
            try:
                cache_key = _response_cache_key(
                    provider=cache_provider,
                    model_name=cache_model_name,
                    device=device,
                    text=text,
                    kwargs=dict(kwargs),
                )
                value = [float(item) for item in vector]
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(cache_key, value)
                else:
                    resolved_deps.set_cached(cache_key, value)
            except Exception:
                continue

    def _generate(
        active_backend: EmbeddingsProvider,
        source_texts: Sequence[str],
    ) -> List[List[float]]:
        nonlocal fallback_used, used_model_name

        def _generate_for_model(active_model_name: Optional[str]) -> List[List[float]]:
            raw = active_backend.embed_texts(
                source_texts,
                model_name=active_model_name,
                device=device,
                **kwargs,
            )
            return _normalize_embedding_vectors(
                raw,
                expected_count=len(source_texts),
            )

        try:
            used_model_name = model_name
            return _generate_for_model(model_name)
        except Exception as initial_error:
            if not (
                _is_hf_inference_provider_name(provider_used)
                and _is_hf_embedding_compatibility_error(initial_error)
            ):
                raise

            attempted = {
                value
                for value in (
                    str(model_name or "").strip(),
                    _coalesce_env(
                        "IPFS_ACCELERATE_PY_HF_EMBEDDINGS_MODEL",
                        "IPFS_DATASETS_PY_HF_EMBEDDINGS_MODEL",
                    ),
                )
                if value
            }
            for fallback_model in _hf_embeddings_fallback_models(
                kwargs=dict(kwargs)
            ):
                if fallback_model in attempted:
                    continue
                attempted.add(fallback_model)
                try:
                    vectors = _generate_for_model(fallback_model)
                except Exception:
                    continue
                used_model_name = fallback_model
                fallback_used = True
                return vectors
            raise initial_error

    try:
        if missing_indices:
            missing_texts = [inputs[index] for index in missing_indices]
            generated = _generate(backend, missing_texts)
            for index, vector in zip(missing_indices, generated):
                cached_vectors[index] = vector
            _cache_vectors(
                missing_texts,
                generated,
                cache_provider=provider_used,
                cache_model_name=used_model_name,
            )

        try:
            result = _normalize_embedding_vectors(
                cached_vectors,
                expected_count=len(inputs),
            )
        except EmbeddingsRouterError:
            # A stale or externally supplied response-cache entry may have a
            # different dimension. Recompute the complete homogeneous batch.
            if cache_hits <= 0:
                raise
            result = _generate(backend, inputs)
            cache_hits = 0
            missing_indices = list(range(len(inputs)))
            _cache_vectors(
                inputs,
                result,
                cache_provider=provider_used,
                cache_model_name=used_model_name,
            )
    except Exception as primary_error:
        logger.debug("Primary embeddings provider failed: %s", primary_error)
        if (
            provider is None
            and provider_instance is None
            and provider_used != "huggingface"
        ):
            try:
                fallback = get_embeddings_provider(
                    "huggingface",
                    deps=resolved_deps,
                )
            except Exception:
                fallback = None
            if fallback is not None:
                try:
                    result = _generate(fallback, inputs)
                    provider_used = "huggingface"
                    fallback_used = True
                    cache_hits = 0
                    missing_indices = list(range(len(inputs)))
                    _cache_vectors(
                        inputs,
                        result,
                        cache_provider=provider_used,
                        cache_model_name=used_model_name,
                    )
                except Exception:
                    _set_last_embedding_trace(
                        status="error",
                        provider_requested="",
                        provider_used=provider_used,
                        model_name=str(model_name or ""),
                        device=str(device or ""),
                        input_count=len(inputs),
                        output_count=0,
                        dimension=0,
                        cache_hits=cache_hits,
                        cache_misses=len(missing_indices),
                        fallback_used=True,
                        error_type=type(primary_error).__name__,
                        elapsed_ms=round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                    )
                    raise primary_error
            else:
                _set_last_embedding_trace(
                    status="error",
                    provider_requested="",
                    provider_used=provider_used,
                    model_name=str(model_name or ""),
                    device=str(device or ""),
                    input_count=len(inputs),
                    output_count=0,
                    dimension=0,
                    cache_hits=cache_hits,
                    cache_misses=len(missing_indices),
                    fallback_used=False,
                    error_type=type(primary_error).__name__,
                    elapsed_ms=round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                )
                raise
        else:
            _set_last_embedding_trace(
                status="error",
                provider_requested=str(provider or ""),
                provider_used=provider_used,
                model_name=str(model_name or ""),
                device=str(device or ""),
                input_count=len(inputs),
                output_count=0,
                dimension=0,
                cache_hits=cache_hits,
                cache_misses=len(missing_indices),
                fallback_used=False,
                error_type=type(primary_error).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise

    dimension = len(result[0]) if result else 0
    _set_last_embedding_trace(
        status="ok",
        provider_requested=str(provider or ""),
        provider_used=provider_used,
        model_name=str(model_name or ""),
        device=str(device or ""),
        input_count=len(inputs),
        output_count=len(result),
        dimension=dimension,
        cache_hits=cache_hits,
        cache_misses=len(missing_indices),
        fallback_used=fallback_used,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return result


def _embedding_batch_worker_count(
    *,
    size: int,
    max_workers: Optional[int],
    device: Optional[str],
) -> int:
    if size <= 1:
        return 1
    if max_workers is not None:
        try:
            requested = max(1, min(int(max_workers), size))
        except (TypeError, ValueError):
            requested = 1
    else:
        raw = _coalesce_env(
            "IPFS_ACCELERATE_EMBEDDINGS_ROUTER_BATCH_WORKERS",
            "IPFS_ACCELERATE_PY_EMBEDDINGS_ROUTER_BATCH_WORKERS",
        )
        try:
            requested = max(1, min(int(raw), size)) if raw else 1
        except (TypeError, ValueError):
            requested = 1
    if str(device or "").strip().lower().startswith("cuda"):
        return 1
    return requested


def embed_texts_batched(
    texts: Iterable[str],
    *,
    batch_size: int = 128,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[EmbeddingsProvider] = None,
    deps: Optional[RouterDeps] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], object]] = None,
    **kwargs: object,
) -> List[List[float]]:
    """Embed a bounded, ordered batch while reusing one provider instance."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    items = list(texts)
    if any(not isinstance(text, str) for text in items):
        raise TypeError("texts must contain only strings")

    total_batches = (
        (len(items) + batch_size - 1) // batch_size if items else 0
    )

    def _report(**values: object) -> None:
        snapshot = _update_embedding_progress(**values)
        if progress_callback is not None:
            try:
                progress_callback(snapshot)
            except Exception:
                logger.debug("Embedding progress callback failed", exc_info=True)

    start_snapshot = _reset_embedding_progress(
        stage="start",
        total_items=len(items),
        completed_items=0,
        total_batches=total_batches,
        completed_batches=0,
        dimension=0,
    )
    if progress_callback is not None:
        try:
            progress_callback(start_snapshot)
        except Exception:
            logger.debug("Embedding progress callback failed", exc_info=True)
    if not items:
        _report(stage="done")
        _set_last_embedding_trace(
            status="ok",
            provider_requested=str(provider or ""),
            provider_used="",
            model_name=str(model_name or ""),
            device=str(device or ""),
            input_count=0,
            output_count=0,
            dimension=0,
            cache_hits=0,
            cache_misses=0,
            fallback_used=False,
            batch_count=0,
            elapsed_ms=0.0,
        )
        return []

    started = time.perf_counter()
    resolved_deps = deps or get_default_router_deps()
    backend = provider_instance or get_embeddings_provider(
        provider,
        deps=resolved_deps,
    )
    ranges = list(range(0, len(items), batch_size))
    workers = _embedding_batch_worker_count(
        size=len(ranges),
        max_workers=max_workers,
        device=device,
    )

    def _embed_batch(start: int) -> tuple[List[List[float]], Dict[str, object]]:
        batch = items[start : start + batch_size]
        vectors = embed_texts(
            batch,
            model_name=model_name,
            device=device,
            provider=provider,
            provider_instance=backend,
            deps=resolved_deps,
            **kwargs,
        )
        return vectors, get_last_embedding_trace()

    batch_results: Dict[int, List[List[float]]] = {}
    traces: List[Dict[str, object]] = []
    completed_items = 0
    try:
        if workers <= 1:
            for completed_batches, start in enumerate(ranges, start=1):
                vectors, trace = _embed_batch(start)
                batch_results[start] = vectors
                traces.append(trace)
                completed_items += len(vectors)
                _report(
                    stage="running",
                    completed_items=completed_items,
                    completed_batches=completed_batches,
                    dimension=len(vectors[0]) if vectors else 0,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                futures = {
                    executor.submit(_embed_batch, start): start
                    for start in ranges
                }
                completed_batches = 0
                for future in concurrent.futures.as_completed(futures):
                    start = futures[future]
                    vectors, trace = future.result()
                    batch_results[start] = vectors
                    traces.append(trace)
                    completed_batches += 1
                    completed_items += len(vectors)
                    _report(
                        stage="running",
                        completed_items=completed_items,
                        completed_batches=completed_batches,
                        dimension=len(vectors[0]) if vectors else 0,
                    )
    except Exception as exc:
        _report(stage="error", error_type=type(exc).__name__)
        _set_last_embedding_trace(
            status="error",
            provider_requested=str(provider or ""),
            provider_used=_provider_name(backend, requested=provider),
            model_name=str(model_name or ""),
            device=str(device or ""),
            input_count=len(items),
            output_count=completed_items,
            dimension=0,
            cache_hits=sum(int(trace.get("cache_hits", 0)) for trace in traces),
            cache_misses=sum(int(trace.get("cache_misses", 0)) for trace in traces),
            fallback_used=any(bool(trace.get("fallback_used")) for trace in traces),
            batch_count=len(traces),
            error_type=type(exc).__name__,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        raise

    ordered = [
        vector
        for start in ranges
        for vector in batch_results[start]
    ]
    result = _normalize_embedding_vectors(ordered, expected_count=len(items))
    dimension = len(result[0]) if result else 0
    provider_used = next(
        (
            str(trace.get("provider_used") or "")
            for trace in traces
            if trace.get("provider_used")
        ),
        _provider_name(backend, requested=provider),
    )
    _report(
        stage="done",
        completed_items=len(result),
        completed_batches=total_batches,
        dimension=dimension,
    )
    _set_last_embedding_trace(
        status="ok",
        provider_requested=str(provider or ""),
        provider_used=provider_used,
        model_name=str(model_name or ""),
        device=str(device or ""),
        input_count=len(items),
        output_count=len(result),
        dimension=dimension,
        cache_hits=sum(int(trace.get("cache_hits", 0)) for trace in traces),
        cache_misses=sum(int(trace.get("cache_misses", 0)) for trace in traces),
        fallback_used=any(bool(trace.get("fallback_used")) for trace in traces),
        batch_count=total_batches,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return result


def clear_embeddings_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""
    _resolve_provider_cached.cache_clear()
    _discover_hf_models_for_pipeline.cache_clear()
    _set_last_embedding_trace()
    _reset_embedding_progress(
        stage="",
        total_items=0,
        completed_items=0,
        total_batches=0,
        completed_batches=0,
        dimension=0,
    )


def embed_text(
    text: str,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: object,
) -> List[float]:
    """Generate an embedding for a single text.
    
    Args:
        text: Text to embed
        model_name: Optional model name to use
        device: Optional device (cpu/cuda)
        provider: Optional provider name
        **kwargs: Additional arguments passed to the provider
        
    Returns:
        Embedding vector
    """

    return embed_texts([text], model_name=model_name, device=device, provider=provider, **kwargs)[0]
