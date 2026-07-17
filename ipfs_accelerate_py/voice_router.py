"""Voice router for ipfs_accelerate_py.

This module provides a stable, reusable entrypoint for voice processing —
both text-to-speech (TTS) synthesis and speech-to-text (STT) transcription —
that integrates with existing ipfs_accelerate_py infrastructure.

Design goals:
- Avoid import-time side effects (no heavy imports at module import).
- Allow optional hooks/providers (backend manager, custom remote endpoints).
- Provide a reliable local fallback via HuggingFace transformers.
- TTS: Return audio as raw bytes (wav/mp3) or write to a file path.
- STT: Return transcription as a plain string.
- Reuse existing patterns from llm_router, multimodal_router, and tts_router.

Environment variables:
- `IPFS_ACCELERATE_PY_VOICE_PROVIDER`: force provider name
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: enable backend manager provider
- `IPFS_ACCELERATE_PY_TTS_MODEL`: HF model name for TTS (default: suno/bark-small)
- `IPFS_ACCELERATE_PY_STT_MODEL`: HF model name for STT (default: openai/whisper-base)
- `IPFS_ACCELERATE_PY_VOICE_DEVICE`: device for local adapters (cpu/cuda)
- `IPFS_ACCELERATE_PY_TTS_OUTPUT_FORMAT`: audio output format hint (wav/mp3)

Additional optional providers (opt-in by selecting provider):
- `openai`: OpenAI TTS + Whisper ASR
    - `OPENAI_API_KEY` or `IPFS_ACCELERATE_PY_OPENAI_API_KEY`
    - `IPFS_ACCELERATE_PY_OPENAI_TTS_MODEL` (default: tts-1)
    - `IPFS_ACCELERATE_PY_OPENAI_TTS_VOICE` (default: alloy)
    - `IPFS_ACCELERATE_PY_OPENAI_STT_MODEL` (default: whisper-1)
    - `IPFS_ACCELERATE_PY_OPENAI_BASE_URL`
- `elevenlabs`: ElevenLabs TTS (no STT)
    - `ELEVENLABS_API_KEY` or `IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY`
    - `IPFS_ACCELERATE_PY_ELEVENLABS_VOICE_ID` (default: Rachel)
    - `IPFS_ACCELERATE_PY_ELEVENLABS_MODEL_ID` (default: eleven_monolingual_v1)
- `assemblyai`: AssemblyAI STT (no TTS)
    - `ASSEMBLYAI_API_KEY` or `IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY`
- `huggingface`: HuggingFace transformers (Bark TTS + Whisper STT)
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


def _audio_digest(audio: Union[str, bytes]) -> str:
    if isinstance(audio, bytes):
        return hashlib.sha256(audio).hexdigest()[:16]
    return hashlib.sha256(str(audio or "").encode("utf-8")).hexdigest()[:16]


def _tts_response_cache_key(
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
        f"voice_tts::{provider_key}::{model_key}::{voice_key}"
        f"::{_text_digest(text)}::{_stable_kwargs_digest(kwargs)}"
    )


def _stt_response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    audio: Union[str, bytes],
    language: Optional[str] = None,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = (model_name or "").strip()
    lang_key = (language or "").strip()
    return (
        f"voice_stt::{provider_key}::{model_key}::{lang_key}"
        f"::{_audio_digest(audio)}::{_stable_kwargs_digest(kwargs)}"
    )


@runtime_checkable
class VoiceProvider(Protocol):
    """Provider interface for voice processing (TTS and/or STT).

    Providers may implement either or both methods.  Calling a method the
    provider does not support raises ``NotImplementedError``.
    """

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

    def transcribe(
        self,
        audio: Union[str, bytes],
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str: ...


ProviderFactory = Callable[[], VoiceProvider]


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_voice_provider(name: str, factory: ProviderFactory) -> None:
    """Register a custom voice provider."""
    if not name or not name.strip():
        raise ValueError("Provider name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


# ---------------------------------------------------------------------------
# Built-in provider implementations
# ---------------------------------------------------------------------------

def _get_openai_provider() -> Optional[VoiceProvider]:
    """Get OpenAI voice provider (TTS via /audio/speech + STT via /audio/transcriptions)."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("IPFS_ACCELERATE_PY_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    class _OpenAIVoiceProvider:
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

        def transcribe(
            self,
            audio: Union[str, bytes],
            *,
            model_name: Optional[str] = None,
            language: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            import io
            import mimetypes

            model = (
                model_name
                or os.getenv("IPFS_ACCELERATE_PY_OPENAI_STT_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_STT_MODEL")
                or "whisper-1"
            )

            # Resolve audio to bytes + filename
            if isinstance(audio, str):
                audio_path = audio.strip()
                with open(audio_path, "rb") as fh:
                    audio_bytes = fh.read()
                filename = os.path.basename(audio_path) or "audio.wav"
            else:
                audio_bytes = audio
                filename = "audio.wav"

            # Build multipart/form-data manually
            boundary = "----VoiceRouterBoundary" + hashlib.sha256(audio_bytes[:64]).hexdigest()[:12]
            parts: list[bytes] = []

            def _field(name: str, value: str) -> bytes:
                return (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode("utf-8")

            parts.append(_field("model", model))
            if language:
                parts.append(_field("language", language))
            if "prompt" in kwargs:
                parts.append(_field("prompt", str(kwargs["prompt"])))

            mime_type = mimetypes.guess_type(filename)[0] or "audio/wav"
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                    f"Content-Type: {mime_type}\r\n\r\n"
                ).encode("utf-8")
                + audio_bytes
                + b"\r\n"
            )
            parts.append(f"--{boundary}--\r\n".encode("utf-8"))
            body = b"".join(parts)

            req = urllib.request.Request(
                f"{base_url}/audio/transcriptions",
                data=body,
                method="POST",
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )

            try:
                with urllib.request.urlopen(req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                raise RuntimeError(f"OpenAI STT HTTP {exc.code}: {detail or exc.reason}") from exc
            except Exception as exc:
                raise RuntimeError(f"OpenAI STT request failed: {exc}") from exc

            try:
                data = json.loads(raw)
                return str(data.get("text", "") or "")
            except Exception:
                return raw

    return _OpenAIVoiceProvider()


def _get_elevenlabs_provider() -> Optional[VoiceProvider]:
    """Get ElevenLabs voice provider (TTS only)."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY")
    if not api_key:
        return None

    class _ElevenLabsVoiceProvider:
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

        def transcribe(
            self,
            audio: Union[str, bytes],
            *,
            model_name: Optional[str] = None,
            language: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            raise NotImplementedError("ElevenLabs provider does not support STT transcription")

    return _ElevenLabsVoiceProvider()


def _get_assemblyai_provider() -> Optional[VoiceProvider]:
    """Get AssemblyAI voice provider (STT only)."""
    api_key = _coalesce_env("IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY")
    if not api_key:
        return None

    class _AssemblyAIVoiceProvider:
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
            raise NotImplementedError("AssemblyAI provider does not support TTS synthesis")

        def transcribe(
            self,
            audio: Union[str, bytes],
            *,
            model_name: Optional[str] = None,
            language: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            _ = device
            _ = model_name
            base_url = "https://api.assemblyai.com/v2"

            # Upload audio
            if isinstance(audio, str):
                audio_path = audio.strip()
                # If it looks like a URL, pass it directly; otherwise upload the file
                if audio_path.startswith(("http://", "https://")):
                    audio_url = audio_path
                else:
                    with open(audio_path, "rb") as fh:
                        audio_bytes = fh.read()
                    upload_req = urllib.request.Request(
                        f"{base_url}/upload",
                        data=audio_bytes,
                        method="POST",
                        headers={
                            "authorization": api_key,
                            "content-type": "application/octet-stream",
                        },
                    )
                    try:
                        with urllib.request.urlopen(upload_req, timeout=float(kwargs.get("timeout", 120))) as resp:
                            upload_data = json.loads(resp.read().decode("utf-8"))
                        audio_url = upload_data["upload_url"]
                    except Exception as exc:
                        raise RuntimeError(f"AssemblyAI upload failed: {exc}") from exc
            else:
                upload_req = urllib.request.Request(
                    f"{base_url}/upload",
                    data=audio,
                    method="POST",
                    headers={
                        "authorization": api_key,
                        "content-type": "application/octet-stream",
                    },
                )
                try:
                    with urllib.request.urlopen(upload_req, timeout=float(kwargs.get("timeout", 120))) as resp:
                        upload_data = json.loads(resp.read().decode("utf-8"))
                    audio_url = upload_data["upload_url"]
                except Exception as exc:
                    raise RuntimeError(f"AssemblyAI upload failed: {exc}") from exc

            # Submit transcription job
            transcript_payload: Dict[str, object] = {"audio_url": audio_url}
            if language:
                transcript_payload["language_code"] = language

            transcript_req = urllib.request.Request(
                f"{base_url}/transcript",
                data=json.dumps(transcript_payload).encode("utf-8"),
                method="POST",
                headers={
                    "authorization": api_key,
                    "content-type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(transcript_req, timeout=float(kwargs.get("timeout", 120))) as resp:
                    transcript_data = json.loads(resp.read().decode("utf-8"))
                transcript_id = transcript_data["id"]
            except Exception as exc:
                raise RuntimeError(f"AssemblyAI transcript submission failed: {exc}") from exc

            # Poll for result
            import time

            poll_timeout = float(kwargs.get("poll_timeout", 300))
            poll_interval = float(kwargs.get("poll_interval", 3))
            deadline = time.monotonic() + poll_timeout

            while time.monotonic() < deadline:
                poll_req = urllib.request.Request(
                    f"{base_url}/transcript/{transcript_id}",
                    method="GET",
                    headers={"authorization": api_key},
                )
                try:
                    with urllib.request.urlopen(poll_req, timeout=30) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                except Exception as exc:
                    raise RuntimeError(f"AssemblyAI poll failed: {exc}") from exc

                status = result.get("status")
                if status == "completed":
                    return str(result.get("text", "") or "")
                if status == "error":
                    raise RuntimeError(f"AssemblyAI transcription error: {result.get('error')}")
                time.sleep(poll_interval)

            raise RuntimeError(f"AssemblyAI transcription timed out after {poll_timeout}s")

    return _AssemblyAIVoiceProvider()


def _get_huggingface_provider() -> Optional[VoiceProvider]:
    """Get HuggingFace voice provider (Bark/SpeechT5 TTS + Whisper STT)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return None

    class _HuggingFaceVoiceProvider:
        def __init__(self) -> None:
            self._tts_models: Dict[str, object] = {}
            self._stt_models: Dict[str, object] = {}

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
            device_str = device or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE") or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cpu")

            cache_key = f"{model}::{device_str}"
            if cache_key not in self._tts_models:
                try:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    pipe = hf_pipeline(
                        "text-to-speech",
                        model=model,
                        device=0 if (device_str == "cuda" and torch.cuda.is_available()) else -1,
                    )
                    self._tts_models[cache_key] = pipe
                except Exception as exc:
                    raise RuntimeError(f"Failed to load HuggingFace TTS model '{model}': {exc}") from exc

            pipe = self._tts_models[cache_key]
            forward_kwargs: Dict[str, object] = {}
            if voice:
                forward_kwargs["speaker_embeddings"] = voice
            if "speaker" in kwargs:
                forward_kwargs["speaker_embeddings"] = kwargs["speaker"]

            result = pipe(str(text), forward_params=forward_kwargs if forward_kwargs else None)

            audio_array = result.get("audio")
            sampling_rate = result.get("sampling_rate", 22050)

            if audio_array is None:
                raise RuntimeError("HuggingFace TTS pipeline returned no audio")

            buf = io.BytesIO()
            if hasattr(audio_array, "squeeze"):
                audio_array = audio_array.squeeze()
            audio_int16 = (np.array(audio_array) * 32767).astype(np.int16)
            wav_io.write(buf, int(sampling_rate), audio_int16)
            return buf.getvalue()

        def transcribe(
            self,
            audio: Union[str, bytes],
            *,
            model_name: Optional[str] = None,
            language: Optional[str] = None,
            device: Optional[str] = None,
            **kwargs: object,
        ) -> str:
            import io

            model = model_name or os.getenv("IPFS_ACCELERATE_PY_STT_MODEL", "openai/whisper-base")
            device_str = device or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE") or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cpu")

            cache_key = f"{model}::{device_str}"
            if cache_key not in self._stt_models:
                try:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model=model,
                        device=0 if (device_str == "cuda" and torch.cuda.is_available()) else -1,
                    )
                    self._stt_models[cache_key] = pipe
                except Exception as exc:
                    raise RuntimeError(f"Failed to load HuggingFace STT model '{model}': {exc}") from exc

            pipe = self._stt_models[cache_key]

            # Resolve audio to a form the pipeline accepts
            if isinstance(audio, bytes):
                import numpy as np
                import scipy.io.wavfile as wav_io

                buf = io.BytesIO(audio)
                try:
                    sample_rate, data = wav_io.read(buf)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    audio_input: object = {"array": data.astype(np.float32) / 32768.0, "sampling_rate": sample_rate}
                except Exception:
                    # Fall back to raw bytes path — some pipelines accept it
                    audio_input = audio
            else:
                audio_input = audio

            generate_kwargs: Dict[str, object] = {}
            if language:
                generate_kwargs["language"] = language

            result = pipe(audio_input, generate_kwargs=generate_kwargs if generate_kwargs else None)

            if isinstance(result, dict):
                return str(result.get("text", "") or "")
            return str(result or "")

    return _HuggingFaceVoiceProvider()


def _get_backend_manager_provider(deps: RouterDeps) -> Optional[VoiceProvider]:
    """Get provider backed by InferenceBackendManager for distributed inference."""
    if not _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")):
        return None

    try:
        manager = deps.get_backend_manager(
            purpose="voice_router",
            enable_health_checks=True,
            load_balancing_strategy=os.getenv(
                "IPFS_ACCELERATE_PY_VOICE_LOAD_BALANCING", "round_robin"
            ),
        )
        if manager is None:
            return None

        class _BackendManagerVoiceProvider:
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

                payload: Dict[str, object] = {"text": str(text), "device": device, **kwargs}
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

            def transcribe(
                self,
                audio: Union[str, bytes],
                *,
                model_name: Optional[str] = None,
                language: Optional[str] = None,
                device: Optional[str] = None,
                **kwargs: object,
            ) -> str:
                import base64

                backend = manager.select_backend_for_task(
                    task="automatic-speech-recognition",
                    model=model_name or os.getenv("IPFS_ACCELERATE_PY_STT_MODEL", ""),
                    protocol="any",
                )
                if backend is None:
                    raise RuntimeError("No available backend for speech-to-text task")

                if isinstance(audio, bytes):
                    audio_payload: object = base64.b64encode(audio).decode("ascii")
                else:
                    audio_payload = audio

                payload: Dict[str, object] = {"audio": audio_payload, "device": device, **kwargs}
                if language:
                    payload["language"] = language

                result = manager.execute_inference(
                    backend_id=backend["id"],
                    task="automatic-speech-recognition",
                    payload=payload,
                )

                text = result.get("text")
                if text is not None:
                    return str(text)
                raise RuntimeError("Backend manager STT provider did not return text")

        return _BackendManagerVoiceProvider()
    except Exception as exc:
        logger.debug(f"Backend manager provider unavailable: {exc}")
        return None


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _provider_cache_key() -> tuple:
    return (
        os.getenv("IPFS_ACCELERATE_PY_VOICE_PROVIDER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "").strip(),
        os.getenv("OPENAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "").strip(),
        os.getenv("ELEVENLABS_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "").strip(),
        os.getenv("ASSEMBLYAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_STT_MODEL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE", "").strip(),
    )


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[VoiceProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in {"openai", "openai_voice"}:
        return _get_openai_provider()
    if key in {"elevenlabs", "eleven_labs", "eleven"}:
        return _get_elevenlabs_provider()
    if key in {"assemblyai", "assembly_ai"}:
        return _get_assemblyai_provider()
    if key in {"hf", "huggingface", "local_hf"}:
        return _get_huggingface_provider()
    if key in {"backend_manager", "accelerate"}:
        return _get_backend_manager_provider(deps)
    return None


def _resolve_provider_uncached(preferred: Optional[str], *, deps: RouterDeps) -> VoiceProvider:
    if preferred:
        info = _PROVIDER_REGISTRY.get(preferred)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(preferred, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown voice provider: {preferred}")

    preferred_env = os.getenv("IPFS_ACCELERATE_PY_VOICE_PROVIDER", "").strip()
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
        "No voice provider available. "
        "Install `transformers`, `scipy`, and `numpy` for local inference, "
        "or configure an API key (OPENAI_API_KEY / ELEVENLABS_API_KEY / ASSEMBLYAI_API_KEY)."
    )


@lru_cache(maxsize=32)
def _resolve_provider_cached(preferred: Optional[str], cache_key: tuple) -> VoiceProvider:
    _ = cache_key
    return _resolve_provider_uncached(preferred, deps=get_default_router_deps())


def get_voice_provider(
    provider: Optional[str] = None,
    *,
    deps: Optional[RouterDeps] = None,
    use_cache: Optional[bool] = None,
) -> VoiceProvider:
    """Resolve a voice provider with optional dependency injection."""
    resolved_deps = deps or get_default_router_deps()
    cache_ok = _cache_enabled() if use_cache is None else bool(use_cache)

    if not cache_ok:
        return _resolve_provider_uncached(provider, deps=resolved_deps)

    if deps is not None:
        cache_key = _provider_cache_key()
        deps_key = (
            f"voice_provider::{(provider or '').strip().lower()}"
            f"::{hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]}"
        )
        cached = resolved_deps.get_cached(deps_key)
        if cached is not None:
            return cached
        return resolved_deps.set_cached(
            deps_key, _resolve_provider_uncached(provider, deps=resolved_deps)
        )

    return _resolve_provider_cached(provider, _provider_cache_key())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_to_speech(
    text: str,
    *,
    voice: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    output_format: Optional[str] = None,
    output_path: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[VoiceProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> Union[bytes, str]:
    """Synthesize speech from text.

    Args:
        text: Text to synthesize.
        voice: Optional voice name/ID (provider-specific).
        model_name: Optional TTS model name.
        device: Optional device hint (cpu/cuda).
        output_format: Optional audio format hint (wav/mp3).
        output_path: Optional file path to write audio bytes to.
            When provided, the audio is written to the file and the path is
            returned as a string.  Otherwise raw bytes are returned.
        provider: Optional provider name.
        provider_instance: Optional pre-created provider instance.
        deps: Optional RouterDeps for dependency injection.
        **kwargs: Additional arguments forwarded to the provider.

    Returns:
        Raw audio bytes, or *output_path* string when output_path is given.
    """
    resolved_deps = deps or get_default_router_deps()

    if _response_cache_enabled():
        cache_key = _tts_response_cache_key(
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

    backend = provider_instance or get_voice_provider(provider, deps=resolved_deps)
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
            raise RuntimeError(f"Voice provider synthesize() returned {type(audio_bytes).__name__}, expected bytes")

        if _response_cache_enabled():
            try:
                ck = _tts_response_cache_key(
                    provider=provider,
                    model_name=model_name,
                    text=text,
                    voice=voice,
                    kwargs=dict(kwargs),
                )
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(ck, audio_bytes)
                else:
                    resolved_deps.set_cached(ck, audio_bytes)
            except Exception:
                pass

        if output_path:
            with open(output_path, "wb") as fh:
                fh.write(audio_bytes)
            return output_path
        return audio_bytes

    except Exception as primary_error:
        logger.debug(f"Primary voice TTS provider failed: {primary_error}")
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


def speech_to_text(
    audio: Union[str, bytes],
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[VoiceProvider] = None,
    deps: Optional[RouterDeps] = None,
    **kwargs: object,
) -> str:
    """Transcribe speech audio to text.

    Args:
        audio: Audio data as raw bytes (WAV/MP3/etc.) or a local file path string.
        model_name: Optional STT model name.
        language: Optional language hint (BCP-47, e.g. "en").
        device: Optional device hint (cpu/cuda).
        provider: Optional provider name.
        provider_instance: Optional pre-created provider instance.
        deps: Optional RouterDeps for dependency injection.
        **kwargs: Additional arguments forwarded to the provider.

    Returns:
        Transcription as a plain string.
    """
    resolved_deps = deps or get_default_router_deps()

    if _response_cache_enabled():
        cache_key = _stt_response_cache_key(
            provider=provider,
            model_name=model_name,
            audio=audio,
            language=language,
            kwargs=dict(kwargs),
        )
        try:
            getter = getattr(resolved_deps, "get_cached_or_remote", None)
            cached = getter(cache_key) if callable(getter) else resolved_deps.get_cached(cache_key)
            if isinstance(cached, str) and cached:
                return cached
        except Exception:
            pass

    # For STT, prefer providers that actually support transcription
    # when no provider is explicitly selected.
    if provider is None and provider_instance is None:
        for name in ["openai", "assemblyai", "huggingface", "backend_manager"]:
            try:
                candidate = _builtin_provider_by_name(name, deps=resolved_deps)
                if candidate is None:
                    continue
                # Quick capability check
                if isinstance(candidate, VoiceProvider):
                    backend: VoiceProvider = candidate
                    break
            except Exception:
                continue
        else:
            backend = get_voice_provider(provider, deps=resolved_deps)
    else:
        backend = provider_instance or get_voice_provider(provider, deps=resolved_deps)

    try:
        transcription = backend.transcribe(
            audio,
            model_name=model_name,
            language=language,
            device=device,
            **kwargs,
        )
        if not isinstance(transcription, str):
            raise RuntimeError(f"Voice provider transcribe() returned {type(transcription).__name__}, expected str")

        if _response_cache_enabled():
            try:
                ck = _stt_response_cache_key(
                    provider=provider,
                    model_name=model_name,
                    audio=audio,
                    language=language,
                    kwargs=dict(kwargs),
                )
                setter = getattr(resolved_deps, "set_cached_and_remote", None)
                if callable(setter):
                    setter(ck, transcription)
                else:
                    resolved_deps.set_cached(ck, transcription)
            except Exception:
                pass

        return transcription

    except Exception as primary_error:
        logger.debug(f"Primary voice STT provider failed: {primary_error}")
        if provider is None:
            hf_provider = _get_huggingface_provider()
            if hf_provider is not None and backend is not hf_provider:
                return hf_provider.transcribe(
                    audio,
                    model_name=model_name,
                    language=language,
                    device=device,
                    **kwargs,
                )
        raise


def clear_voice_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""
    _resolve_provider_cached.cache_clear()
