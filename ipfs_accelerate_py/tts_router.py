"""Text-to-speech router for ipfs_accelerate_py.

This module provides a stable, reusable entrypoint for text-to-speech synthesis
that integrates with existing ipfs_accelerate_py infrastructure.

Design goals:
- Avoid import-time side effects (no heavy imports at module import).
- Allow optional hooks/providers (backend manager, custom remote endpoints).
- Provide a reliable local fallback via HuggingFace transformers.
- Return audio as raw bytes (wav/mp3) or write to a file path.
- Reuse existing patterns from llm_router and embeddings_router.

Environment variables:
- `IPFS_ACCELERATE_PY_TTS_PROVIDER`: force provider name
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: enable backend manager provider
- `IPFS_ACCELERATE_PY_TTS_MODEL`: HF model name for local adapter
    (default: suno/bark-small)
- `IPFS_ACCELERATE_PY_TTS_DEVICE`: device for local adapter (cpu/cuda)
- `IPFS_ACCELERATE_PY_TTS_OUTPUT_FORMAT`: audio output format hint (wav/mp3)

Additional optional providers (opt-in by selecting provider):
- `openai`: OpenAI TTS API
    - `OPENAI_API_KEY` or `IPFS_ACCELERATE_PY_OPENAI_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENAI_TTS_MODEL` (default: tts-1)
    - `IPFS_ACCELERATE_PY_OPENAI_TTS_VOICE` (default: alloy)
    - `IPFS_ACCELERATE_PY_OPENAI_BASE_URL`
- `elevenlabs`: ElevenLabs TTS API
    - `ELEVENLABS_API_KEY` or `IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY`
    - `IPFS_ACCELERATE_PY_ELEVENLABS_VOICE_ID` (default: Rachel)
    - `IPFS_ACCELERATE_PY_ELEVENLABS_MODEL_ID` (default: eleven_monolingual_v1)
- `huggingface`: HuggingFace transformers (Bark, SpeechT5, etc.)
- `backend_manager`: Use InferenceBackendManager for distributed inference
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Protocol, Union, runtime_checkable

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


def _response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    text: str,
    voice: Optional[str] = None,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = (model_name or "").strip()
    voice_key = (voice or "").strip()
    return (
        f"tts_response::{provider_key}::{model_key}::{voice_key}"
        f"::{_text_digest(text)}::{_stable_kwargs_digest(kwargs)}"
    )


@runtime_checkable
class TTSProvider(Protocol):
    """Provider interface for text-to-speech synthesis."""

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs: object,
    ) -> bytes: ...


ProviderFactory = Callable[[], TTSProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_tts_provider(name: str, factory: ProviderFactory) -> None:
    """Register a custom TTS provider."""
    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _get_openai_provider() -> Optional[TTSProvider]:
    """Get OpenAI TTS provider."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    class _OpenAITTSProvider:
        def synthesize(
            self,
            text: str,
            *,
            voice: Optional[str] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            output_format: Optional[str] = None,
            **kwargs: object,
        ) -> bytes:
            _ = device
            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENAI_TTS_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL")
                or "tts-1"
            )
            selected_voice = (
                voice
                or os.getenv("IPFS_ACCELERATE_PY_OPENAI_TTS_VOICE")
                or "alloy"
            )
            fmt = (
                output_format
                or os.getenv("IPFS_ACCELERATE_PY_TTS_OUTPUT_FORMAT")
                or "mp3"
            )

            payload: Dict[str, object] = {
                "model": model,
                "input": str(text),
                "voice": selected_voice,
                "response_format": fmt,
            }
            if "speed" in kwargs:
                payload["speed"] = kwargs["speed"]

            req = urllib.request.Request(
                f"{base_url}/audio/speech",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"OpenAI TTS HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"OpenAI TTS request failed: {exc}") from exc

    return _OpenAITTSProvider()


def _get_elevenlabs_provider() -> Optional[TTSProvider]:
    """Get ElevenLabs TTS provider."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY")
    if not api_key:
        return None

    class _ElevenLabsTTSProvider:
        def synthesize(
            self,
            text: str,
            *,
            voice: Optional[str] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            output_format: Optional[str] = None,
            **kwargs: object,
        ) -> bytes:
            _ = device
            _ = output_format
            voice_id = (
                voice
                or os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_VOICE_ID")
                or "Rachel"
            )
            model_id = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_MODEL_ID")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL")
                or "eleven_monolingual_v1"
            )

            payload: Dict[str, object] = {
                "text": str(text),
                "model_id": model_id,
                "voice_settings": {
                    "stability": float(kwargs.get("stability", 0.5)),
                    "similarity_boost": float(kwargs.get("similarity_boost", 0.75)),
                },
            }

            req = urllib.request.Request(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"ElevenLabs HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc

    return _ElevenLabsTTSProvider()


def _get_huggingface_provider() -> Optional[TTSProvider]:
    """Get HuggingFace TTS provider using transformers (Bark, SpeechT5, etc.)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None

    class _HuggingFaceTTSProvider:
        def __init__(self):
            self._models: Dict[str, object] = {}

        def synthesize(
            self,
            text: str,
            *,
            voice: Optional[str] = None,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            output_format: Optional[str] = None,
            **kwargs: object,
        ) -> bytes:
            import io
            import numpy as np
            import scipy.io.wavfile as wav_io

            model = model_name or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL", "suno/bark-small")
            device_str = device or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cpu")

            cache_key = f"{model}::{device_str}"
            if cache_key not in self._models:
                try:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    pipe = hf_pipeline(
                        "text-to-speech",
                        model=model,
                        device=0 if (device_str == "cuda" and torch.cuda.is_available()) else -1,
                    )
                    self._models[cache_key] = pipe
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load HuggingFace TTS model '{model}': {exc}"
                    ) from exc

            pipe = self._models[cache_key]
            forward_kwargs: Dict[str, object] = {}
            if voice:
                forward_kwargs["speaker_embeddings"] = voice
            if "speaker" in kwargs:
                forward_kwargs["speaker_embeddings"] = kwargs["speaker"]

            result = pipe(str(text), forward_params=forward_kwargs if forward_kwargs else None)

            # result is a dict with "audio" (numpy array) and "sampling_rate"
            audio_array = result.get("audio")
            sampling_rate = result.get("sampling_rate", 22050)

            if audio_array is None:
                raise RuntimeError("HuggingFace TTS pipeline returned no audio")

            # Convert to WAV bytes
            buf = io.BytesIO()
            if hasattr(audio_array, "squeeze"):
                audio_array = audio_array.squeeze()
            audio_int16 = (np.array(audio_array) * 32767).astype(np.int16)
            wav_io.write(buf, int(sampling_rate), audio_int16)
            return buf.getvalue()

    return _HuggingFaceTTSProvider()


def _get_backend_manager_provider(deps: RouterDeps) -> Optional[TTSProvider]:
    """Get provider that uses InferenceBackendManager for distributed/multiplexed inference."""
    if not _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")):
        return None

    try:
        manager = deps.get_backend_manager(
            purpose="tts_router",
            enable_health_checks=True,
            load_balancing_strategy=os.getenv(
                "IPFS_ACCELERATE_PY_TTS_LOAD_BALANCING", "round_robin"
            ),
        )
        if manager is None:
            return None

        class _BackendManagerTTSProvider:
            def synthesize(
                self,
                text: str,
                *,
                voice: Optional[str] = None,
                model_name: Optional[str] = None,
                device: Optional[str] = None,
                output_format: Optional[str] = None,
                **kwargs: object,
            ) -> bytes:
                import base64

                backend = manager.select_backend_for_task(
                    task="text-to-speech",
                    model=model_name or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL", ""),
                    protocol="any",
                )

                if backend is None:
                    raise RuntimeError("No available backend for text-to-speech task")

                payload: Dict[str, object] = {
                    "text": str(text),
                    "device": device,
                    **kwargs,
                }
                if voice:
                    payload["voice"] = voice
                if output_format:
                    payload["output_format"] = output_format

                result = manager.execute_inference(
                    backend_id=backend["id"],
                    task="text-to-speech",
                    payload=payload,
                )

                audio = result.get("audio")
                if isinstance(audio, bytes):
                    return audio
                if isinstance(audio, str):
                    return base64.b64decode(audio)
                raise RuntimeError("Backend manager TTS provider did not return audio bytes")

        return _BackendManagerTTSProvider()
    except Exception as exc:
        logger.debug(f"Backend manager provider unavailable: {exc}")
        return None


def _provider_cache_key() -> tuple:
    return (
        os.getenv("IPFS_ACCELERATE_PY_TTS_PROVIDER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "").strip(),
        os.getenv("ELEVENLABS_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "").strip(),
    )


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[TTSProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in {"openai", "openai_tts"}:
        return _get_openai_provider()
    if key in {"elevenlabs", "eleven_labs", "eleven"}:
        return _get_elevenlabs_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_huggingface_provider()
    if key in {"backend_manager", "accelerate"}:
        return _get_backend_manager_provider(deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> TTSProvider:
    if preferred:
        info = _PROVIDER_REGISTRY.get(preferred)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown TTS provider: {preferred}")

    preferred_env = os.getenv("IPFS_ACCELERATE_PY_TTS_PROVIDER", "").strip()
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

    for name in ["openai", "elevenlabs"]:
        candidate = _builtin_provider_by_name(name, deps=deps)
        if candidate is not None:
            return candidate

    hf_provider = _get_huggingface_provider()
    if hf_provider is not None:
        return hf_provider

    raise RuntimeError(
        "No TTS provider available. "
        "Install `transformers`, `scipy`, and `numpy` for local inference, "
        "or configure an API key (OPENAI_API_KEY / ELEVENLABS_API_KEY)."
    )


@lru_cache(maxsize=32)
def _resolve_provider_cached(preferred: Optional[str], cache_key: tuple) -> TTSProvider:
    _ = cache_key
    return _resolve_provider_uncached(preferred, deps=get_default_router_deps())


def get_tts_provider(
    provider: Optional[str] = None,
    *,
    deps: Optional[RouterDeps] = None,
    use_cache: Optional[bool] = None,
) -> TTSProvider:
    """Resolve a TTS provider with optional dependency injection."""
    resolved_deps = deps or get_default_router_deps()
    cache_ok = _cache_enabled() if use_cache is None else bool(use_cache)

    if not cache_ok:
        return _resolve_provider_uncached(provider, deps=resolved_deps)

    if deps is not None:
        cache_key = _provider_cache_key()
        deps_key = f"tts_provider::{(provider or '').strip().lower()}::{hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]}"
        cached = resolved_deps.get_cached(deps_key)
        if cached is not None:
            return cached
        return resolved_deps.set_cached(deps_key, _resolve_provider_uncached(provider, deps=resolved_deps))

    return _resolve_provider_cached(provider, _provider_cache_key())


def text_to_speech(
    text: str,
    *,
    voice: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    output_format: Optional[str] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[TTSProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> Union[bytes, str]:
    """Synthesize speech from text.

    Args:
        text: Text to synthesize
        voice: Optional voice name/ID (provider-specific)
        model_name: Optional model name to use
        device: Optional device (cpu/cuda)
        output_format: Optional audio format hint (wav/mp3)
        output_path: Optional file path to write audio bytes to.
            When provided, the audio is written to the file and the path is
            returned as a string. Otherwise raw bytes are returned.
        provider: Optional provider name
        provider_instance: Optional pre-created provider instance
        deps: Optional RouterDeps for dependency injection
        **kwargs: Additional arguments passed to the provider

    Returns:
        Raw audio bytes, or *output_path* string when output_path is given.
    """
    resolved_deps = deps or get_default_router_deps()

    if _response_cache_enabled():
        cache_key = _response_cache_key(
            provider=provider,
            model_name=model_name,
            text=text,
            voice=voice,
            kwargs=dict(kwargs),
        )
        try:
            getter = getattr(resolved_deps, "get_cached_or_remote", None)
            cached = getter(cache_key) if callable(getter) else resolved_deps.get_cached(cache_key)
            if isinstance(cached, bytes) and cached:
                if output_path:
                    with open(output_path, "wb") as fh:
                        fh.write(cached)
                    return output_path
                return cached
        except Exception:
            pass

    backend = provider_instance or get_tts_provider(provider, deps=resolved_deps)
    try:
        audio_bytes = backend.synthesize(
            text,
            voice=voice,
            model_name=model_name,
            device=device,
            output_format=output_format,
            **kwargs,
        )
        if not isinstance(audio_bytes, bytes):
            raise RuntimeError(f"TTS provider returned {type(audio_bytes).__name__}, expected bytes")

        if _response_cache_enabled():
            try:
                cache_key = _response_cache_key(
                    provider=provider,
                    model_name=model_name,
                    text=text,
                    voice=voice,
                    kwargs=dict(kwargs),
                )
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(cache_key, audio_bytes)
                else:
                    resolved_deps.set_cached(cache_key, audio_bytes)
            except Exception:
                pass

        if output_path:
            with open(output_path, "wb") as fh:
                fh.write(audio_bytes)
            return output_path
        return audio_bytes

    except Exception as primary_error:
        logger.debug(f"Primary TTS provider failed: {primary_error}")
        if provider is None:
            hf_provider = _get_huggingface_provider()
            if hf_provider is not None and backend is not hf_provider:
                audio_bytes = hf_provider.synthesize(
                    text,
                    voice=voice,
                    model_name=model_name,
                    device=device,
                    output_format=output_format,
                    **kwargs,
                )
                if output_path:
                    with open(output_path, "wb") as fh:
                        fh.write(audio_bytes)
                    return output_path
                return audio_bytes
        raise


def clear_tts_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""
    _resolve_provider_cached.cache_clear()
