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
import json
import os
import hashlib
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from .router_deps import RouterDeps, get_default_router_deps

logger = logging.getLogger(__name__)


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
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE", "1").strip() != "0"


def _response_cache_enabled() -> bool:
    value = os.environ.get("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE")
    if value is None:
        return True  # Default to enabled
    return str(value).strip() != "0"


def _response_cache_key_strategy() -> str:
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY", "sha256").strip().lower() or "sha256"


def _response_cache_cid_base() -> str:
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE", "base32").strip() or "base32"


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
            or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL")
            or ""
        ).strip()

    # Local adapter / default.
    return (os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL", "") or "").strip()


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
            from .ipfs_multiformats import cid_for_obj
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


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_embeddings_provider(name: str, factory: ProviderFactory) -> None:
    """Register a custom embeddings provider."""

    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    normalized = name.strip().lower()
    _PROVIDER_REGISTRY[normalized] = ProviderInfo(name=normalized, factory=factory)


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


def _get_openrouter_provider() -> Optional[EmbeddingsProvider]:
    """Get OpenRouter embeddings provider."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
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
                or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL")
                or "text-embedding-3-small"
            )
            inputs = list(texts)
            payload = {"model": model, "input": inputs}

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if referer:
                headers["HTTP-Referer"] = referer
            if app_title:
                headers["X-Title"] = app_title

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

        def embed_texts(
            self,
            texts: Iterable[str],
            *,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> List[List[float]]:
            model = model_name or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            device_str = device or os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE", "cpu")
            
            # Get or create model
            cache_key = f"{model}::{device_str}"
            if cache_key not in self._models:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._models[cache_key] = SentenceTransformer(model, device=device_str)
                except ImportError:
                    # Fall back to transformers directly
                    from transformers import AutoTokenizer, AutoModel
                    import torch
                    
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    model_obj = AutoModel.from_pretrained(model)
                    if device_str == "cuda" and torch.cuda.is_available():
                        model_obj = model_obj.to("cuda")
                    self._models[cache_key] = (tokenizer, model_obj, device_str)
            
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
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_API_KEY", "").strip(),
        os.getenv("OPENROUTER_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_EMBEDDINGS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENROUTER_BASE_URL", "").strip(),
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
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_DEVICE", "").strip(),
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
    if key in {"xai", "grok", "xai_grok"}:
        return _get_xai_embeddings_provider()
    if key in {"meta_ai", "meta-ai", "meta_llama", "meta", "meta_spark", "spark"}:
        return _get_meta_ai_embeddings_provider()
    if key in {"gemini", "gemini_cli"}:
        return _get_gemini_cli_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_huggingface_provider()
    if key in {"backend_manager", "accelerate"}:
        return _get_backend_manager_provider(deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> EmbeddingsProvider:
    if preferred:
        preferred_key = preferred.strip().lower()
        info = _PROVIDER_REGISTRY.get(preferred_key)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_key, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown embeddings provider: {preferred}")

    # 1) Registered providers can opt-in via env ordering if desired.
    preferred_env = (
        os.getenv("IPFS_ACCELERATE_PY_EMBEDDINGS_PROVIDER", "")
        .strip()
        .lower()
    )
    if preferred_env:
        info = _PROVIDER_REGISTRY.get(preferred_env)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_env, deps=deps)
        if builtin is not None:
            return builtin

    # 2) Optional backend manager provider.
    backend_manager_provider = _get_backend_manager_provider(deps)
    if backend_manager_provider is not None:
        return backend_manager_provider

    # Try optional providers if available.
    for name in ["openrouter", "xai", "meta_ai", "gemini_cli"]:
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

    def _cache_vectors(
        source_texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        cache_provider: str,
    ) -> None:
        if not cache_enabled:
            return
        for text, vector in zip(source_texts, vectors):
            try:
                cache_key = _response_cache_key(
                    provider=cache_provider,
                    model_name=model_name,
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
        raw = active_backend.embed_texts(
            source_texts,
            model_name=model_name,
            device=device,
            **kwargs,
        )
        return _normalize_embedding_vectors(
            raw,
            expected_count=len(source_texts),
        )

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
            _cache_vectors(inputs, result, cache_provider=provider_used)
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
                    _cache_vectors(inputs, result, cache_provider=provider_used)
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
