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
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

from .router_deps import RouterDeps, get_default_router_deps

logger = logging.getLogger(__name__)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_enabled() -> bool:
    return os.environ.get("IPFS_ACCELERATE_PY_ROUTER_CACHE", "1").strip() != "0"


def _response_cache_enabled() -> bool:
    value = os.environ.get("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE")
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


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_multimodal_provider(name: str, factory: ProviderFactory) -> None:
    """Register a custom multimodal provider."""
    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


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
        info = _PROVIDER_REGISTRY.get(preferred)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown multimodal provider: {preferred}")

    preferred_env = os.getenv("IPFS_ACCELERATE_PY_MULTIMODAL_PROVIDER", "").strip()
    if preferred_env:
        info = _PROVIDER_REGISTRY.get(preferred_env)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred_env, deps=deps)
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
