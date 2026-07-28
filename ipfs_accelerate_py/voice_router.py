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
- Reuse existing patterns from llm_router, multimodal_router, and embeddings_router.

Environment variables:
- `IPFS_ACCELERATE_PY_VOICE_PROVIDER`: force provider name
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: enable backend manager provider
- `IPFS_ACCELERATE_PY_TTS_MODEL`: HF model name for TTS (default: suno/bark-small)
- `IPFS_ACCELERATE_PY_STT_MODEL`: HF model name for STT (default: openai/whisper-base)
- `IPFS_ACCELERATE_PY_TTS_DEVICE`: device for local TTS (falls back to VOICE_DEVICE)
- `IPFS_ACCELERATE_PY_STT_DEVICE`: device for local STT (falls back to VOICE_DEVICE)
- `IPFS_ACCELERATE_PY_VOICE_DEVICE`: shared device fallback for local adapters (cpu/cuda)
- `IPFS_ACCELERATE_PY_TTS_OUTPUT_FORMAT`: audio output format hint (wav/mp3)
- `IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS`: ordered Abby IndexTTS HTTP URLs
- `IPFS_ACCELERATE_PY_ABBY_WHISPER_BASE_URL`: Abby Whisper HTTP model base URL
- `IPFS_ACCELERATE_PY_ABBY_*_TOKEN`: optional remote-provider credentials
- `IPFS_ACCELERATE_PY_ABBY_*_TIMEOUT_SECONDS`: bounded provider timeouts

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
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import MappingProxyType
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
    Union,
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
    redact_secrets,
)
from .router_deps import RouterDeps, get_default_router_deps
from .voice_audio_resolver import (
    PrecomputedAudioResolution,
    PrecomputedVoiceAudioResolver,
    SynthesisIdentity,
    spoken_text_sha256 as precomputed_spoken_text_sha256,
)
from .voice_templates import (
    buildVoiceGraphRagPromptParts,
    normalize_spoken_text,
    template_fields,
)

logger = logging.getLogger(__name__)

VOICE_TURN_CONTRACT_VERSION = "1.0"
TELEPHONE_TURN_CONTRACT_VERSION = "1.0"
VOICE_STAGE_STATUSES = frozenset({"succeeded", "failed", "skipped"})
VOICE_TURN_STATUSES = frozenset({"completed", "degraded", "text_only", "failed"})

# Evidence identity for AICAT-G130 / AICAT-033 voice usage integration.
USAGE_ROUTING_REQUIREMENT_ID = "requirement:voice-router-usage-routing.v1"
VOICE_TTS_USAGE_OPERATION = "audio.synthesize"
VOICE_STT_USAGE_OPERATION = "audio.transcribe"

_LAST_USAGE_ADMISSION = threading.local()
_LAST_VOICE_USAGE_TRACE = threading.local()

# Ranking-input names that embed these substrings are rejected by receipt
# digests. Full reservation envelopes still include tokens/media_bytes.
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
    "transcript",
    "synthesis",
    "voice_sample",
)


class VoiceRouterError(RuntimeError):
    """Raised when a provider violates the voice router contract."""


class UsageCapacityError(VoiceRouterError):
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


def _set_last_usage_admission(payload: Optional[Mapping[str, object]]) -> None:
    _LAST_USAGE_ADMISSION.payload = dict(payload) if payload is not None else None


def get_last_usage_admission() -> Dict[str, object]:
    """Return a copy of the most recent usage-admission result for this thread.

    Operational evidence only: never transcript, synthesis text, audio, or credentials.
    """

    payload = getattr(_LAST_USAGE_ADMISSION, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _set_last_voice_usage_trace(**values: object) -> None:
    _LAST_VOICE_USAGE_TRACE.payload = dict(values)


def get_last_voice_usage_trace() -> Dict[str, object]:
    """Return a copy of the most recent voice usage trace for this thread."""

    payload = getattr(_LAST_VOICE_USAGE_TRACE, "payload", None)
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


def _audio_digest(audio: Union[str, bytes]) -> str:
    if isinstance(audio, bytes):
        return hashlib.sha256(audio).hexdigest()[:16]
    if isinstance(audio, str) and os.path.isfile(audio):
        digest = hashlib.sha256()
        try:
            with open(audio, "rb") as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()[:16]
        except OSError:
            pass
    return hashlib.sha256(str(audio or "").encode("utf-8")).hexdigest()[:16]


def _provider_instance_cache_identity(
    provider_instance: Optional[object],
    provider_name: Optional[str] = None,
) -> Optional[str]:
    """Return a cache namespace that cannot cross-contaminate instances."""
    if provider_instance is None:
        normalized_name = str(provider_name or "").strip().lower()
        environment_key_factory = globals().get("_provider_cache_key")
        environment_key = (
            environment_key_factory() if callable(environment_key_factory) else ()
        )
        digest = hashlib.sha256(repr(environment_key).encode("utf-8")).hexdigest()[:16]
        if normalized_name:
            revisions = globals().get("_PROVIDER_REGISTRY_REVISIONS", {})
            revision = revisions.get(normalized_name, 0)
            return f"{normalized_name}::revision-{revision}::{digest}"
        return f"auto::{digest}"
    explicit = getattr(provider_instance, "cache_identity", None)
    if callable(explicit):
        explicit = explicit()
    provider_type = provider_instance.__class__
    type_name = f"{provider_type.__module__}.{provider_type.__qualname__}"
    if explicit is not None and str(explicit).strip():
        explicit_digest = hashlib.sha256(
            str(explicit).strip().encode("utf-8")
        ).hexdigest()[:16]
        return f"instance::{type_name}::{explicit_digest}"
    # An injected instance with no declared stable identity is intentionally
    # process-local. Reusing a remote cache entry from another instance could
    # return speech from the wrong model, tenant, or voice configuration.
    return f"instance::{type_name}::{id(provider_instance)}"


def _tts_response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    text: str,
    voice: Optional[str] = None,
    device: Optional[str] = None,
    output_format: Optional[str] = None,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = (model_name or "").strip()
    voice_key = (voice or "").strip()
    device_key = (device or "").strip().lower()
    format_key = (output_format or "").strip().lower().lstrip(".")
    return (
        f"voice_tts::{provider_key}::{model_key}::{voice_key}"
        f"::{device_key}::{format_key}"
        f"::{_text_digest(text)}::{_stable_kwargs_digest(kwargs)}"
    )


def _stt_response_cache_key(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    audio: Union[str, bytes],
    language: Optional[str] = None,
    device: Optional[str] = None,
    kwargs: Dict[str, object],
) -> str:
    provider_key = (provider or "auto").strip().lower()
    model_key = (model_name or "").strip()
    lang_key = (language or "").strip()
    device_key = (device or "").strip().lower()
    return (
        f"voice_stt::{provider_key}::{model_key}::{lang_key}::{device_key}"
        f"::{_audio_digest(audio)}::{_stable_kwargs_digest(kwargs)}"
    )


@runtime_checkable
class VoiceProvider(Protocol):
    """Provider interface for voice processing (TTS and/or STT).

    Provider objects expose both methods for structural/runtime protocol
    checks. An unsupported method raises ``NotImplementedError`` and its
    operation is declared false in :class:`VoiceProviderCapabilities`.
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
class VoiceProviderCapabilities:
    """Machine-readable operations supported by a voice provider."""

    transcription: bool = True
    synthesis: bool = True
    streaming: bool = False
    audio_formats: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("transcription", "synthesis", "streaming"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        raw_formats = (
            (self.audio_formats,)
            if isinstance(self.audio_formats, str)
            else (self.audio_formats or ())
        )
        formats = tuple(
            dict.fromkeys(
                str(value).strip().lower().lstrip(".")
                for value in raw_formats
                if str(value).strip().lstrip(".")
            )
        )
        object.__setattr__(self, "audio_formats", formats)

    @property
    def can_transcribe(self) -> bool:
        return self.transcription

    @property
    def can_synthesize(self) -> bool:
        return self.synthesis

    def supports(self, operation: str) -> bool:
        """Return whether *operation* is supported by this provider.

        The accepted operation names match both the provider method names and
        the pipeline stage names so callers do not need provider-specific
        translation logic.
        """
        normalized = str(operation or "").strip().lower()
        if normalized in {"transcribe", "transcription", "stt", "speech_to_text"}:
            return self.transcription
        if normalized in {"synthesize", "synthesis", "tts", "text_to_speech"}:
            return self.synthesis
        if normalized == "streaming":
            return self.streaming
        return False

    def to_dict(self) -> Dict[str, object]:
        return {
            "transcription": self.transcription,
            "synthesis": self.synthesis,
            "streaming": self.streaming,
            "audio_formats": list(self.audio_formats),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VoiceProviderCapabilities":
        if not isinstance(value, Mapping):
            raise TypeError("VoiceProviderCapabilities.from_dict requires a mapping")

        def _boolean(name: str, default: bool) -> bool:
            raw_value = value.get(name, default)
            if not isinstance(raw_value, bool):
                raise TypeError(f"{name} must be a boolean")
            return raw_value

        raw_formats = value.get("audio_formats", ())
        if isinstance(raw_formats, str):
            raw_formats = (raw_formats,)
        if not isinstance(raw_formats, Sequence):
            raise TypeError("audio_formats must be a string or sequence")
        return cls(
            transcription=_boolean("transcription", True),
            synthesis=_boolean("synthesis", True),
            streaming=_boolean("streaming", False),
            audio_formats=tuple(str(item) for item in raw_formats),
        )


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    factory: ProviderFactory
    capabilities: VoiceProviderCapabilities = field(
        default_factory=VoiceProviderCapabilities
    )

    def __post_init__(self) -> None:
        name = str(self.name or "").strip().lower()
        if not name:
            raise ValueError("ProviderInfo.name must be non-empty")
        if not callable(self.factory):
            raise TypeError("ProviderInfo.factory must be callable")
        if not isinstance(self.capabilities, VoiceProviderCapabilities):
            raise TypeError(
                "ProviderInfo.capabilities must be VoiceProviderCapabilities"
            )
        object.__setattr__(self, "name", name)

    def to_dict(self) -> Dict[str, object]:
        """Serialize provider metadata without attempting to serialize code."""
        return {
            "name": self.name,
            "capabilities": self.capabilities.to_dict(),
        }


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}
_PROVIDER_REGISTRY_REVISIONS: Dict[str, int] = {}


def register_voice_provider(
    name: str,
    factory: ProviderFactory,
    *,
    capabilities: Optional[VoiceProviderCapabilities] = None,
) -> None:
    """Register a custom voice provider and its optional capabilities."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("Provider name must be non-empty")
    if not callable(factory):
        raise TypeError("Provider factory must be callable")
    if capabilities is not None and not isinstance(
        capabilities, VoiceProviderCapabilities
    ):
        raise TypeError("capabilities must be VoiceProviderCapabilities or None")
    _PROVIDER_REGISTRY[normalized_name] = ProviderInfo(
        name=normalized_name,
        factory=factory,
        capabilities=capabilities or VoiceProviderCapabilities(),
    )
    _PROVIDER_REGISTRY_REVISIONS[normalized_name] = (
        _PROVIDER_REGISTRY_REVISIONS.get(normalized_name, 0) + 1
    )
    # A re-registration is expected to take effect immediately. The global
    # resolver exists by the time public registration can be called.
    resolver = globals().get("_resolve_provider_cached")
    if resolver is not None:
        resolver.cache_clear()


_BUILTIN_PROVIDER_CAPABILITIES: Mapping[str, VoiceProviderCapabilities] = {
    "abby_indextts": VoiceProviderCapabilities(
        transcription=False,
        audio_formats=("wav", "mp3", "flac", "ogg"),
    ),
    "abby_whisper": VoiceProviderCapabilities(
        synthesis=False,
        audio_formats=("wav", "mp3", "flac", "ogg", "webm", "m4a"),
    ),
    "openai": VoiceProviderCapabilities(),
    "elevenlabs": VoiceProviderCapabilities(transcription=False),
    "assemblyai": VoiceProviderCapabilities(synthesis=False),
    "huggingface": VoiceProviderCapabilities(),
    "backend_manager": VoiceProviderCapabilities(),
}

_BUILTIN_PROVIDER_ALIASES: Mapping[str, str] = {
    "abby_index_tts": "abby_indextts",
    "indextts": "abby_indextts",
    "abby_hf_whisper": "abby_whisper",
    "hf_whisper": "abby_whisper",
    "openai_voice": "openai",
    "eleven_labs": "elevenlabs",
    "eleven": "elevenlabs",
    "assembly_ai": "assemblyai",
    "hf": "huggingface",
    "local_hf": "huggingface",
    "accelerate": "backend_manager",
}


_AUDIO_MIME_TYPES: Mapping[str, str] = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "pcm": "audio/l16",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "webm": "audio/webm",
}

_VOICE_CATALOG_PROVENANCE = (Provenance(source="voice_router"),)


@dataclass(frozen=True)
class _VoiceCatalogMetadata:
    display_name: str
    description: str
    locality: str
    device: str
    access_type: str
    transcription_media_types: Tuple[str, ...] = ()
    synthesis_media_types: Tuple[str, ...] = ()
    languages: str = "provider-defined"
    voices: str = "provider-defined"
    default_voice: Optional[str] = None
    default_transcription_model: Optional[str] = None
    default_synthesis_model: Optional[str] = None
    max_input_bytes: Optional[int] = None
    max_output_bytes: Optional[int] = None
    max_duration_seconds: Optional[int] = None
    sample_rates_hz: Tuple[int, ...] = ()


_BUILTIN_VOICE_CATALOG: Mapping[str, _VoiceCatalogMetadata] = {
    "abby_indextts": _VoiceCatalogMetadata(
        display_name="Abby IndexTTS",
        description="Remote Abby IndexTTS speech synthesis.",
        locality="remote",
        device="remote",
        access_type="optional-token",
        synthesis_media_types=(
            "audio/flac",
            "audio/mpeg",
            "audio/ogg",
            "audio/wav",
        ),
        voices="provider-defined",
        default_synthesis_model="Publicus/IndexTTS-2-Demo",
    ),
    "abby_whisper": _VoiceCatalogMetadata(
        display_name="Abby Whisper",
        description="Remote Hugging Face Whisper speech transcription.",
        locality="remote",
        device="remote",
        access_type="optional-token",
        transcription_media_types=(
            "audio/flac",
            "audio/mp4",
            "audio/mpeg",
            "audio/ogg",
            "audio/wav",
            "audio/webm",
        ),
        languages="multilingual",
        default_transcription_model="openai/whisper-large-v3-turbo",
    ),
    "openai": _VoiceCatalogMetadata(
        display_name="OpenAI Voice",
        description="OpenAI speech synthesis and audio transcription APIs.",
        locality="remote",
        device="remote",
        access_type="api-key",
        transcription_media_types=(
            "audio/mp4",
            "audio/mpeg",
            "audio/wav",
            "audio/webm",
        ),
        synthesis_media_types=(
            "audio/aac",
            "audio/flac",
            "audio/l16",
            "audio/mpeg",
            "audio/ogg",
            "audio/wav",
        ),
        languages="multilingual",
        voices="provider-defined",
        default_voice="alloy",
        default_transcription_model="whisper-1",
        default_synthesis_model="tts-1",
    ),
    "elevenlabs": _VoiceCatalogMetadata(
        display_name="ElevenLabs",
        description="ElevenLabs speech synthesis API.",
        locality="remote",
        device="remote",
        access_type="api-key",
        synthesis_media_types=("audio/mpeg",),
        languages="model-defined",
        voices="provider-defined",
        default_voice="Rachel",
        default_synthesis_model="eleven_monolingual_v1",
    ),
    "assemblyai": _VoiceCatalogMetadata(
        display_name="AssemblyAI",
        description="AssemblyAI speech transcription API.",
        locality="remote",
        device="remote",
        access_type="api-key",
        transcription_media_types=("audio/*",),
        languages="provider-defined",
        default_transcription_model="default",
    ),
    "huggingface": _VoiceCatalogMetadata(
        display_name="Hugging Face Voice",
        description="Local transformers pipelines for speech synthesis and transcription.",
        locality="local",
        device="cpu,cuda",
        access_type="none",
        transcription_media_types=(
            "audio/flac",
            "audio/mp4",
            "audio/mpeg",
            "audio/ogg",
            "audio/wav",
            "audio/webm",
        ),
        synthesis_media_types=("audio/wav",),
        languages="model-defined",
        voices="model-defined",
        default_transcription_model="openai/whisper-base",
        default_synthesis_model="suno/bark-small",
    ),
    "backend_manager": _VoiceCatalogMetadata(
        display_name="Inference Backend Manager",
        description="Distributed voice inference selected by InferenceBackendManager.",
        locality="distributed",
        device="provider-defined",
        access_type="backend-policy",
        transcription_media_types=("audio/*",),
        synthesis_media_types=("audio/*",),
        languages="provider-defined",
        voices="provider-defined",
        default_transcription_model="default-stt",
        default_synthesis_model="default-tts",
    ),
}


def _catalog_name(value: object, *, fallback: str = "default") -> str:
    """Return a bounded canonical catalog name for router-owned hints."""
    raw_value = str(value or "").strip()
    if redact_secrets(raw_value) != raw_value:
        return fallback
    normalized = raw_value.casefold()
    normalized = re.sub(r"[^a-z0-9._/-]+", "-", normalized)
    normalized = re.sub(r"/{2,}", "/", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    normalized = normalized.strip("._/-")
    if not normalized:
        normalized = fallback
    return normalized[:128].rstrip("._/-") or fallback


def _canonical_operation(operation: Optional[Union[str, Operation]]) -> Optional[Operation]:
    if operation is None:
        return None
    if isinstance(operation, Operation):
        if operation in {Operation.AUDIO_TRANSCRIBE, Operation.AUDIO_SYNTHESIZE}:
            return operation
        raise ValueError(f"Unsupported voice operation: {operation.value}")
    normalized = str(operation or "").strip().casefold().replace("-", "_")
    if normalized in {
        "audio.transcribe",
        "transcribe",
        "transcription",
        "speech_to_text",
        "stt",
    }:
        return Operation.AUDIO_TRANSCRIBE
    if normalized in {
        "audio.synthesize",
        "synthesize",
        "synthesis",
        "text_to_speech",
        "tts",
    }:
        return Operation.AUDIO_SYNTHESIZE
    raise ValueError(f"Unsupported voice operation: {operation}")


def _mime_types(formats: Iterable[str]) -> Tuple[str, ...]:
    result = set()
    for value in formats:
        normalized = str(value or "").strip().casefold()
        if "/" in normalized:
            result.add(normalized)
        elif normalized.lstrip(".") in _AUDIO_MIME_TYPES:
            result.add(_AUDIO_MIME_TYPES[normalized.lstrip(".")])
    return tuple(sorted(result))


def _provider_aliases(name: str) -> Tuple[str, ...]:
    # A dynamically registered canonical name wins over a built-in alias in
    # the invocation resolver, so discovery must expose the same precedence.
    return tuple(
        sorted(
            alias
            for alias, canonical in _BUILTIN_PROVIDER_ALIASES.items()
            if canonical == name and alias not in _PROVIDER_REGISTRY
        )
    )


def _provider_catalog_metadata(name: str) -> _VoiceCatalogMetadata:
    metadata = _BUILTIN_VOICE_CATALOG.get(name)
    if metadata is not None:
        return metadata
    capabilities = _PROVIDER_REGISTRY[name].capabilities
    media_types = _mime_types(capabilities.audio_formats)
    return _VoiceCatalogMetadata(
        display_name=name.replace("_", " ").replace("-", " ").title(),
        description="Dynamically registered voice provider.",
        locality="unknown",
        device="provider-defined",
        access_type="provider-defined",
        transcription_media_types=media_types,
        synthesis_media_types=media_types,
        default_transcription_model="default",
        default_synthesis_model="default",
    )


def _provider_configuration(name: str) -> Tuple[Optional[bool], Optional[bool]]:
    """Return static configured/authorized facts without constructing clients."""
    if name in _PROVIDER_REGISTRY:
        return True, None
    if name == "openai":
        configured = bool(
            _coalesce_env("IPFS_ACCELERATE_PY_OPENAI_API_KEY", "OPENAI_API_KEY")
        )
        return configured, configured
    if name == "elevenlabs":
        configured = bool(
            _coalesce_env(
                "IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"
            )
        )
        return configured, configured
    if name == "assemblyai":
        configured = bool(
            _coalesce_env(
                "IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY"
            )
        )
        return configured, configured
    if name == "abby_indextts":
        configured = bool(
            _coalesce_env(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS",
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URL",
                "WALLET_INDEXTTS_SPACE_URL",
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_FALLBACK_URL",
                "WALLET_INDEXTTS_FALLBACK_SPACE_URL",
            )
        )
        authorized = (
            True
            if _coalesce_env(
                "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TOKEN", "HF_TOKEN"
            )
            else None
        )
        return configured, authorized
    if name == "abby_whisper":
        # The adapter has a public default base URL even when no override is set.
        authorized = (
            True
            if _coalesce_env(
                "IPFS_ACCELERATE_PY_ABBY_WHISPER_TOKEN",
                "WALLET_HF_WHISPER_TOKEN",
                "HF_TOKEN",
            )
            else None
        )
        return True, authorized
    if name == "huggingface":
        # Package and model availability intentionally remain unknown: checking
        # either here would violate the side-effect-free discovery contract.
        return None, True
    if name == "backend_manager":
        return _truthy(os.getenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER")), None
    return None, None


def _provider_state(name: str) -> OperationalState:
    configured, authorized = _provider_configuration(name)
    if name in _PROVIDER_REGISTRY:
        routable: Optional[bool] = True
    elif name in {"openai", "elevenlabs", "assemblyai", "abby_indextts"}:
        routable = configured
    elif name == "abby_whisper":
        routable = True
    else:
        routable = None
    return OperationalState(
        known=True,
        configured=configured,
        authorized=authorized,
        reachable=None,
        healthy=None,
        routable=routable,
    )


def _provider_capability_descriptors(
    name: str,
) -> Tuple[CapabilityDescriptor, ...]:
    capabilities = get_voice_provider_capabilities(name)
    metadata = _provider_catalog_metadata(name)
    records = []
    if capabilities.transcription:
        operations = [Operation.AUDIO_TRANSCRIBE]
        if capabilities.streaming:
            operations.append(Operation.STREAM)
        records.append(
            CapabilityDescriptor(
                operations=tuple(operations),
                input_modalities=(Modality.AUDIO,),
                output_modalities=(Modality.TEXT,),
                media_types=metadata.transcription_media_types
                or _mime_types(capabilities.audio_formats),
                max_input_bytes=metadata.max_input_bytes,
            )
        )
    if capabilities.synthesis:
        operations = [Operation.AUDIO_SYNTHESIZE]
        if capabilities.streaming:
            operations.append(Operation.STREAM)
        records.append(
            CapabilityDescriptor(
                operations=tuple(operations),
                input_modalities=(Modality.TEXT,),
                output_modalities=(Modality.AUDIO,),
                media_types=metadata.synthesis_media_types
                or _mime_types(capabilities.audio_formats),
                max_output_bytes=metadata.max_output_bytes,
            )
        )
    return tuple(records)


def _provider_descriptor(name: str) -> ProviderDescriptor:
    metadata = _provider_catalog_metadata(name)
    capabilities = get_voice_provider_capabilities(name)
    state = _provider_state(name)
    if name in _PROVIDER_REGISTRY:
        readiness = "registered-unverified"
    elif state.configured is True:
        readiness = "configured-unverified"
    elif state.configured is False:
        readiness = "not-configured"
    else:
        readiness = "unknown"
    labels = {
        "router": "voice_router",
        "locality": metadata.locality,
        "device": metadata.device,
        "access_type": metadata.access_type,
        "readiness": readiness,
        "streaming": str(capabilities.streaming).lower(),
        "batching": "false",
        "audio.languages": metadata.languages,
        "audio.voices": metadata.voices,
    }
    if metadata.default_voice:
        labels["audio.default_voice"] = metadata.default_voice
    if metadata.sample_rates_hz:
        labels["audio.sample_rates_hz"] = ",".join(
            str(value) for value in metadata.sample_rates_hz
        )
    if metadata.max_duration_seconds:
        labels["audio.max_duration_seconds"] = str(
            metadata.max_duration_seconds
        )
    lifecycle = (
        LifecycleState.CONFIGURED
        if state.configured is True
        else LifecycleState.DECLARED
    )
    return ProviderDescriptor(
        name=name,
        display_name=metadata.display_name,
        aliases=_provider_aliases(name),
        description=metadata.description,
        capabilities=_provider_capability_descriptors(name),
        lifecycle=lifecycle,
        state=state,
        provenance=_VOICE_CATALOG_PROVENANCE,
        labels=labels,
    )


def _catalog_provider_names() -> Tuple[str, ...]:
    # Registry entries replace same-named built-ins, exactly as invocation does.
    return tuple(
        sorted(set(_BUILTIN_PROVIDER_CAPABILITIES) | set(_PROVIDER_REGISTRY))
    )


def _model_names_for_provider(
    name: str, operation: Optional[Operation] = None
) -> Tuple[Tuple[str, Operation], ...]:
    metadata = _provider_catalog_metadata(name)
    capabilities = get_voice_provider_capabilities(name)
    records = []
    if capabilities.transcription and operation in (None, Operation.AUDIO_TRANSCRIBE):
        default = metadata.default_transcription_model or "default"
        if name == "abby_whisper":
            default = (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_MODEL")
                or os.getenv("WALLET_HF_WHISPER_MODEL_NAME")
                or default
            )
        elif name == "openai":
            default = (
                os.getenv("IPFS_ACCELERATE_PY_OPENAI_STT_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_STT_MODEL")
                or default
            )
        elif name == "huggingface":
            default = os.getenv("IPFS_ACCELERATE_PY_STT_MODEL") or default
        elif name == "backend_manager":
            default = os.getenv("IPFS_ACCELERATE_PY_STT_MODEL") or default
        records.append((_catalog_name(default, fallback="default-stt"), Operation.AUDIO_TRANSCRIBE))
    if capabilities.synthesis and operation in (None, Operation.AUDIO_SYNTHESIZE):
        default = metadata.default_synthesis_model or "default"
        if name == "abby_indextts":
            default = (
                os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_MODEL")
                or os.getenv("WALLET_INDEXTTS_MODEL_NAME")
                or default
            )
        elif name == "openai":
            default = (
                os.getenv("IPFS_ACCELERATE_PY_OPENAI_TTS_MODEL")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL")
                or default
            )
        elif name == "elevenlabs":
            default = (
                os.getenv("IPFS_ACCELERATE_PY_ELEVENLABS_MODEL_ID")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL")
                or default
            )
        elif name in {"huggingface", "backend_manager"}:
            default = os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL") or default
        records.append((_catalog_name(default, fallback="default-tts"), Operation.AUDIO_SYNTHESIZE))
    return tuple(records)


def _model_descriptors_for_provider(
    provider_descriptor: ProviderDescriptor,
    operation: Optional[Operation] = None,
) -> Tuple[ModelDescriptor, ...]:
    metadata = _provider_catalog_metadata(provider_descriptor.name)
    grouped: Dict[str, list[Operation]] = {}
    for model_name, model_operation in _model_names_for_provider(
        provider_descriptor.name, operation
    ):
        grouped.setdefault(model_name, []).append(model_operation)
    records = []
    for model_name, model_operations in grouped.items():
        capability_records = tuple(
            capability
            for capability in provider_descriptor.capabilities
            if any(
                model_operation in capability.operations
                for model_operation in model_operations
            )
        )
        labels = {
            "router": "voice_router",
            "locality": metadata.locality,
            "device": metadata.device,
            "streaming": str(
                Operation.STREAM
                in {
                    item
                    for capability in capability_records
                    for item in capability.operations
                }
            ).lower(),
            "batching": "false",
            "audio.languages": metadata.languages,
            "audio.voices": metadata.voices,
            "audio.operations": ",".join(
                sorted(item.value for item in set(model_operations))
            ),
        }
        if metadata.default_voice:
            labels["audio.default_voice"] = metadata.default_voice
        if metadata.sample_rates_hz:
            labels["audio.sample_rates_hz"] = ",".join(
                str(value) for value in metadata.sample_rates_hz
            )
        if metadata.max_duration_seconds:
            labels["audio.max_duration_seconds"] = str(
                metadata.max_duration_seconds
            )
        records.append(
            ModelDescriptor(
                provider_id=provider_descriptor.provider_id,
                name=model_name,
                display_name=model_name,
                description=f"Voice router model hint for {provider_descriptor.name}.",
                capabilities=capability_records,
                lifecycle=provider_descriptor.lifecycle,
                state=provider_descriptor.state,
                provenance=_VOICE_CATALOG_PROVENANCE,
                labels=labels,
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def _descriptor_operations(
    descriptor: Union[ProviderDescriptor, ModelDescriptor]
) -> frozenset[Operation]:
    return frozenset(
        operation
        for capability in descriptor.capabilities
        for operation in capability.operations
    )


def _label(descriptor: Union[ProviderDescriptor, ModelDescriptor], name: str) -> Optional[str]:
    return dict(descriptor.labels).get(name)


def _matches_catalog_constraints(
    descriptor: Union[ProviderDescriptor, ModelDescriptor],
    *,
    operation: Optional[Operation],
    language: Optional[str],
    voice: Optional[str],
    media_type: Optional[str],
    sample_rate_hz: Optional[int],
    duration_seconds: Optional[float],
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
    if batching is True and Operation.BATCH not in operations:
        return False
    if locality is not None:
        actual = _label(descriptor, "locality")
        if actual not in (None, "unknown") and actual != str(locality).casefold():
            return False
    if device is not None:
        actual_devices = {
            item.strip().casefold()
            for item in (_label(descriptor, "device") or "").split(",")
            if item.strip()
        }
        requested_device = str(device).strip().casefold()
        if actual_devices and "provider-defined" not in actual_devices and requested_device not in actual_devices:
            return False
    if language is not None:
        languages = (_label(descriptor, "audio.languages") or "").casefold()
        requested_language = str(language).strip().casefold()
        known_open = {"multilingual", "provider-defined", "model-defined", ""}
        if languages not in known_open and requested_language not in {
            item.strip() for item in languages.split(",")
        }:
            return False
    if voice is not None:
        voices = (_label(descriptor, "audio.voices") or "").casefold()
        requested_voice = str(voice).strip().casefold()
        known_open = {"provider-defined", "model-defined", ""}
        if voices not in known_open and requested_voice not in {
            item.strip() for item in voices.split(",")
        }:
            return False
    if media_type is not None:
        requested_media = _mime_types((media_type,))
        if not requested_media:
            return False
        known_media = {
            item
            for capability in descriptor.capabilities
            for item in capability.media_types
        }
        if known_media and "audio/*" not in known_media and requested_media[0] not in known_media:
            return False
    if sample_rate_hz is not None:
        known_rates = {
            int(item)
            for item in (_label(descriptor, "audio.sample_rates_hz") or "").split(",")
            if item.strip().isdigit()
        }
        if known_rates and int(sample_rate_hz) not in known_rates:
            return False
    if duration_seconds is not None:
        maximum = _label(descriptor, "audio.max_duration_seconds")
        if maximum is not None and float(duration_seconds) > float(maximum):
            return False
    if size_bytes is not None:
        relevant_capabilities = (
            descriptor.capabilities
            if operation is None
            else tuple(
                capability
                for capability in descriptor.capabilities
                if operation in capability.operations
            )
        )
        known_limits = [
            limit
            for capability in relevant_capabilities
            for limit in (capability.max_input_bytes, capability.max_output_bytes)
            if limit is not None
        ]
        if known_limits and int(size_bytes) > max(known_limits):
            return False
    if authorized is not None and descriptor.state.authorized is not authorized:
        return False
    if ready is not None and descriptor.state.routable is not ready:
        return False
    return True


def list_providers(
    *,
    operation: Optional[Union[str, Operation]] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    media_type: Optional[str] = None,
    sample_rate_hz: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
) -> Tuple[ProviderDescriptor, ...]:
    """List canonical voice providers without resolving or constructing one."""
    selected_operation = _canonical_operation(operation)
    records = tuple(
        _provider_descriptor(name) for name in _catalog_provider_names()
    )
    return tuple(
        record
        for record in records
        if _matches_catalog_constraints(
            record,
            operation=selected_operation,
            language=language,
            voice=voice,
            media_type=media_type,
            sample_rate_hz=sample_rate_hz,
            duration_seconds=duration_seconds,
            size_bytes=size_bytes,
            streaming=streaming,
            batching=batching,
            locality=locality,
            device=device,
            authorized=authorized,
            ready=ready,
        )
    )


def get_provider_descriptor(name: str) -> ProviderDescriptor:
    """Return one provider descriptor, honoring invocation alias precedence."""
    normalized = str(name or "").strip().casefold()
    if not normalized:
        raise ValueError("Provider name must be non-empty")
    if normalized in _PROVIDER_REGISTRY:
        canonical = normalized
    else:
        canonical = _BUILTIN_PROVIDER_ALIASES.get(normalized, normalized)
    if canonical not in _PROVIDER_REGISTRY and canonical not in _BUILTIN_PROVIDER_CAPABILITIES:
        raise ValueError(f"Unknown voice provider: {name}")
    return _provider_descriptor(canonical)


def list_models(
    provider: Optional[str] = None,
    *,
    operation: Optional[Union[str, Operation]] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    media_type: Optional[str] = None,
    sample_rate_hz: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
) -> Tuple[ModelDescriptor, ...]:
    """List configured model hints projected from provider defaults."""
    selected_operation = _canonical_operation(operation)
    providers = (
        (get_provider_descriptor(provider),)
        if provider is not None
        else list_providers()
    )
    records = tuple(
        model
        for provider_record in providers
        for model in _model_descriptors_for_provider(
            provider_record, selected_operation
        )
    )
    return tuple(
        sorted(
            (
                record
                for record in records
                if _matches_catalog_constraints(
                    record,
                    operation=selected_operation,
                    language=language,
                    voice=voice,
                    media_type=media_type,
                    sample_rate_hz=sample_rate_hz,
                    duration_seconds=duration_seconds,
                    size_bytes=size_bytes,
                    streaming=streaming,
                    batching=batching,
                    locality=locality,
                    device=device,
                    authorized=authorized,
                    ready=ready,
                )
            ),
            key=lambda record: (record.provider_id, record.name),
        )
    )


def resolve_model(
    model: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    operation: Optional[Union[str, Operation]] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    media_type: Optional[str] = None,
    sample_rate_hz: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    size_bytes: Optional[int] = None,
    streaming: Optional[bool] = None,
    batching: Optional[bool] = None,
    locality: Optional[str] = None,
    device: Optional[str] = None,
    authorized: Optional[bool] = None,
    ready: Optional[bool] = None,
) -> ModelDescriptor:
    """Resolve an invocation-compatible model/provider pair.

    Explicit provider names use the same dynamic-registration and alias
    precedence as :func:`get_voice_provider`. An explicit model override is
    accepted for a compatible explicit provider because the invocation API
    forwards such overrides to the provider.
    """
    if model is not None and model_name is not None and model != model_name:
        raise ValueError("model and model_name must agree when both are provided")
    requested_model = model if model is not None else model_name
    selected_operation = _canonical_operation(operation)
    candidates = list_models(
        provider,
        operation=selected_operation,
        language=language,
        voice=voice,
        media_type=media_type,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
        streaming=streaming,
        batching=batching,
        locality=locality,
        device=device,
        authorized=authorized,
        ready=ready,
    )
    if requested_model is not None:
        canonical_model = _catalog_name(requested_model)
        matches = tuple(
            candidate
            for candidate in candidates
            if canonical_model == candidate.name
            or canonical_model in candidate.aliases
        )
        if matches:
            candidates = matches
        elif provider is not None:
            provider_record = get_provider_descriptor(provider)
            if not _matches_catalog_constraints(
                provider_record,
                operation=selected_operation,
                language=language,
                voice=voice,
                media_type=media_type,
                sample_rate_hz=sample_rate_hz,
                duration_seconds=duration_seconds,
                size_bytes=size_bytes,
                streaming=streaming,
                batching=batching,
                locality=locality,
                device=device,
                authorized=authorized,
                ready=ready,
            ):
                candidates = ()
            else:
                capability_records = tuple(
                    capability
                    for capability in provider_record.capabilities
                    if selected_operation is None
                    or selected_operation in capability.operations
                )
                return ModelDescriptor(
                    provider_id=provider_record.provider_id,
                    name=canonical_model,
                    display_name=canonical_model,
                    description=(
                        f"Explicit voice model override for {provider_record.name}."
                    ),
                    capabilities=capability_records,
                    lifecycle=provider_record.lifecycle,
                    state=provider_record.state,
                    provenance=_VOICE_CATALOG_PROVENANCE,
                    labels={
                        **dict(provider_record.labels),
                        "explicit_override": "true",
                    },
                )
        else:
            candidates = matches
    if not candidates:
        detail = f" model {requested_model!r}" if requested_model is not None else ""
        raise ValueError(f"No compatible voice{detail} provider was found")

    if provider is None:
        statically_viable = tuple(
            candidate
            for candidate in candidates
            if candidate.state.routable is not False
        )
        if statically_viable:
            candidates = statically_viable

    preferred_name = os.getenv("IPFS_ACCELERATE_PY_VOICE_PROVIDER", "").strip()
    if preferred_name:
        try:
            preferred_id = get_provider_descriptor(preferred_name).provider_id
        except ValueError:
            preferred_id = None
        if preferred_id is not None:
            preferred = tuple(
                candidate
                for candidate in candidates
                if candidate.provider_id == preferred_id
            )
            if preferred:
                return preferred[0]

    operation_order = {
        Operation.AUDIO_TRANSCRIBE: (
            "openai",
            "assemblyai",
            "huggingface",
            "backend_manager",
            "abby_whisper",
        ),
        Operation.AUDIO_SYNTHESIZE: (
            "backend_manager",
            "openai",
            "elevenlabs",
            "huggingface",
            "abby_indextts",
        ),
    }
    provider_by_id = {
        descriptor.provider_id: descriptor.name for descriptor in list_providers()
    }
    order = operation_order.get(selected_operation, ())
    rank = {name: index for index, name in enumerate(order)}
    return min(
        candidates,
        key=lambda candidate: (
            rank.get(provider_by_id.get(candidate.provider_id, ""), len(rank)),
            provider_by_id.get(candidate.provider_id, ""),
            candidate.name,
        ),
    )


def get_catalog_snapshot() -> CatalogSnapshot:
    """Project the current voice router registry into one immutable snapshot."""
    providers = list_providers()
    models = tuple(
        model
        for provider in providers
        for model in _model_descriptors_for_provider(provider)
    )
    bindings = tuple(
        RouterBinding(
            router="voice_router",
            provider_id=model.provider_id,
            model_id=model.model_id,
            operations=tuple(
                operation
                for operation in _descriptor_operations(model)
                if operation not in {Operation.BATCH, Operation.STREAM}
            )
            + tuple(
                operation
                for operation in (Operation.BATCH, Operation.STREAM)
                if operation in _descriptor_operations(model)
            ),
            state=model.state,
            provenance=_VOICE_CATALOG_PROVENANCE,
            labels={"source": "router"},
        )
        for model in models
    )
    return CatalogSnapshot(
        providers=providers,
        models=models,
        bindings=bindings,
    )


# Common source-adapter spelling retained alongside the explicit getter.
catalog_snapshot = get_catalog_snapshot


def get_voice_provider_capabilities(name: str) -> VoiceProviderCapabilities:
    """Return declared capabilities without constructing a provider.

    This makes capability discovery safe for optional and remote providers:
    no model import, credential lookup, or network request occurs.
    """
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("Provider name must be non-empty")
    info = _PROVIDER_REGISTRY.get(normalized_name)
    if info is not None:
        return info.capabilities
    builtin_name = _BUILTIN_PROVIDER_ALIASES.get(normalized_name, normalized_name)
    capabilities = _BUILTIN_PROVIDER_CAPABILITIES.get(builtin_name)
    if capabilities is None:
        raise ValueError(f"Unknown voice provider: {name}")
    return capabilities


def _coalesce_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


# ---------------------------------------------------------------------------
# Grounded Abby voice-turn contracts
# ---------------------------------------------------------------------------

DEFAULT_GROUNDED_FALLBACK = (
    "I couldn't verify enough current information to answer safely. "
    "Please contact 211 for help."
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: object) -> object:
    """Return a deterministic JSON-safe representation without raw bytes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {
            "sha256": _sha256_bytes(value),
            "size_bytes": len(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return repr(value)


@dataclass(frozen=True)
class GroundingEvidence:
    """A current evidence record used to bind a response-template slot."""

    source_id: str
    cid: Optional[str] = None
    uri: Optional[str] = None
    text: Optional[str] = None
    facts: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = str(self.source_id or "").strip()
        if not source_id:
            raise ValueError("GroundingEvidence.source_id must be non-empty")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "facts", dict(self.facts or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "cid": self.cid,
            "uri": self.uri,
            "text": self.text,
            "facts": _json_safe(self.facts),
            "metadata": _json_safe(self.metadata),
        }


# Descriptive alias used by consumers that model provenance as sources.
VoiceGroundingSource = GroundingEvidence


@dataclass(frozen=True)
class GroundedSlot:
    """A rendered slot value and the evidence records that support it."""

    name: str
    value: object
    source_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("GroundedSlot.name must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "source_ids",
            tuple(
                dict.fromkeys(
                    str(source_id).strip()
                    for source_id in (self.source_ids or ())
                    if str(source_id).strip()
                )
            ),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "value": _json_safe(self.value),
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class VoiceResponsePlan:
    """A response frame returned by GraphRAG, never an uncited final answer."""

    template_id: str
    template: str
    slots: Tuple[GroundedSlot, ...] = ()
    evidence: Tuple[GroundingEvidence, ...] = ()
    confidence: float = 1.0
    intent: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        template_id = str(self.template_id or "").strip()
        template = str(self.template or "").strip()
        if not template_id:
            raise ValueError("VoiceResponsePlan.template_id must be non-empty")
        if not template:
            raise ValueError("VoiceResponsePlan.template must be non-empty")
        object.__setattr__(self, "template_id", template_id)
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "slots", tuple(self.slots or ()))
        object.__setattr__(self, "evidence", tuple(self.evidence or ()))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> Dict[str, object]:
        return {
            "template_id": self.template_id,
            "template": self.template,
            "slots": [slot.to_dict() for slot in self.slots],
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "intent": self.intent,
            "metadata": _json_safe(self.metadata),
        }

    @property
    def sources(self) -> Tuple[GroundingEvidence, ...]:
        """Alias for GraphRAG stores that call evidence records sources."""
        return self.evidence


@runtime_checkable
class VoiceTemplateProvider(Protocol):
    """Retrieves grounded response plans for a caller transcript."""

    def retrieve(
        self,
        transcript: str,
        *,
        context: Optional[Mapping[str, object]] = None,
        language: Optional[str] = None,
    ) -> Optional[VoiceResponsePlan]: ...


@dataclass(frozen=True)
class VoiceStageTrace:
    """Serializable receipt for one attempt at a pipeline stage."""

    stage: str
    status: str
    duration_ms: float
    provider: Optional[str] = None
    error: Optional[str] = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stage = str(self.stage or "").strip()
        status = str(self.status or "").strip().lower()
        if not stage:
            raise ValueError("VoiceStageTrace.stage must be non-empty")
        if status not in VOICE_STAGE_STATUSES:
            raise ValueError(
                "VoiceStageTrace.status must be one of "
                + ", ".join(sorted(VOICE_STAGE_STATUSES))
            )
        duration_ms = float(self.duration_ms)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError(
                "VoiceStageTrace.duration_ms must be finite and non-negative"
            )
        provider = str(self.provider).strip() if self.provider is not None else None
        error = str(self.error).strip() if self.error is not None else None
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "provider", provider or None)
        object.__setattr__(self, "error", error or None)
        object.__setattr__(
            self, "details", MappingProxyType(dict(self.details or {}))
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "provider": self.provider,
            "error": self.error,
            "details": _json_safe(self.details),
        }


@dataclass(frozen=True)
class VoiceTurnRequest:
    """Input contract for one Abby turn.

    ``audio`` starts the pipeline at STT. A supplied ``transcript`` supports
    trusted upstream STT and deterministic replays. At least one must be
    non-empty; when both are supplied the transcript wins and the audio hash is
    still retained in provenance.
    """

    audio: Optional[Union[str, bytes]] = None
    transcript: Optional[str] = None
    request_id: Optional[str] = None
    context: Mapping[str, object] = field(default_factory=dict)
    grounding: Mapping[str, object] = field(default_factory=dict)
    language: Optional[str] = None
    locale: Optional[str] = None
    voice: Optional[str] = None
    stt_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    stt_providers: Tuple[str, ...] = ()
    tts_providers: Tuple[str, ...] = ()
    stt_model: Optional[str] = None
    tts_model: Optional[str] = None
    device: Optional[str] = None
    output_format: Optional[str] = None
    minimum_template_confidence: float = 0.0
    max_template_results: int = 5
    fallback_text: str = DEFAULT_GROUNDED_FALLBACK
    stt_options: Mapping[str, object] = field(default_factory=dict)
    tts_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transcript = str(self.transcript).strip() if self.transcript is not None else ""
        valid_audio = (
            isinstance(self.audio, bytes)
            and bool(self.audio)
            or isinstance(self.audio, str)
            and bool(self.audio.strip())
        )
        if not transcript and not valid_audio:
            raise ValueError("VoiceTurnRequest requires non-empty audio or transcript")
        if self.audio is not None and not isinstance(self.audio, (str, bytes)):
            raise TypeError("VoiceTurnRequest.audio must be bytes, a path/URL string, or None")
        minimum_confidence = float(self.minimum_template_confidence)
        if (
            not math.isfinite(minimum_confidence)
            or not 0.0 <= minimum_confidence <= 1.0
        ):
            raise ValueError("minimum_template_confidence must be between 0 and 1")
        if int(self.max_template_results) < 1:
            raise ValueError("max_template_results must be at least 1")
        fallback_text = str(self.fallback_text or "").strip()
        if not fallback_text:
            raise ValueError("fallback_text must be non-empty")
        request_id = (
            str(self.request_id).strip() if self.request_id is not None else ""
        )
        object.__setattr__(self, "transcript", transcript or None)
        object.__setattr__(self, "request_id", request_id or None)
        object.__setattr__(
            self, "context", MappingProxyType(dict(self.context or {}))
        )
        object.__setattr__(
            self, "grounding", MappingProxyType(dict(self.grounding or {}))
        )
        for field_name in (
            "language",
            "locale",
            "voice",
            "stt_model",
            "tts_model",
            "device",
            "output_format",
        ):
            raw_value = getattr(self, field_name)
            normalized = (
                str(raw_value).strip() if raw_value is not None else ""
            )
            object.__setattr__(self, field_name, normalized or None)
        for field_name in ("stt_provider", "tts_provider"):
            raw_value = getattr(self, field_name)
            normalized = (
                str(raw_value).strip().lower() if raw_value is not None else ""
            )
            object.__setattr__(self, field_name, normalized or None)
        object.__setattr__(
            self,
            "stt_providers",
            tuple(
                dict.fromkeys(
                    str(name).strip().lower()
                    for name in self.stt_providers
                    if str(name).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "tts_providers",
            tuple(
                dict.fromkeys(
                    str(name).strip().lower()
                    for name in self.tts_providers
                    if str(name).strip()
                )
            ),
        )
        object.__setattr__(
            self, "minimum_template_confidence", minimum_confidence
        )
        object.__setattr__(self, "max_template_results", int(self.max_template_results))
        object.__setattr__(self, "fallback_text", fallback_text)
        object.__setattr__(
            self, "stt_options", MappingProxyType(dict(self.stt_options or {}))
        )
        object.__setattr__(
            self, "tts_options", MappingProxyType(dict(self.tts_options or {}))
        )

    @property
    def effective_language(self) -> Optional[str]:
        return self.language or self.locale

    @property
    def input_audio_sha256(self) -> Optional[str]:
        if isinstance(self.audio, bytes):
            return _sha256_bytes(self.audio)
        if isinstance(self.audio, str):
            if os.path.isfile(self.audio):
                digest = hashlib.sha256()
                try:
                    with open(self.audio, "rb") as input_file:
                        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                            digest.update(chunk)
                    return digest.hexdigest()
                except OSError:
                    # The provider reports unreadable inputs. Serialization
                    # remains total and never exposes the path itself.
                    pass
            return _sha256_text(self.audio)
        return None

    def to_dict(self, *, include_audio: bool = False) -> Dict[str, object]:
        """Return a JSON-safe request.

        Raw caller audio and local paths are excluded by default. Passing
        ``include_audio=True`` is an explicit wire-transport choice; byte audio
        is then base64 encoded and string inputs are emitted as ``audio``.
        """
        payload: Dict[str, object] = {
            "contract_version": VOICE_TURN_CONTRACT_VERSION,
            "request_id": self.request_id,
            "transcript": self.transcript,
            "input_audio_sha256": self.input_audio_sha256,
            "input_audio_size_bytes": len(self.audio)
            if isinstance(self.audio, bytes)
            else None,
            "context": _json_safe(self.context),
            "grounding": _json_safe(self.grounding),
            "language": self.language,
            "locale": self.locale,
            "voice": self.voice,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "stt_providers": list(self.stt_providers),
            "tts_providers": list(self.tts_providers),
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "device": self.device,
            "output_format": self.output_format,
            "minimum_template_confidence": self.minimum_template_confidence,
            "max_template_results": self.max_template_results,
            "fallback_text": self.fallback_text,
            "stt_options": _json_safe(self.stt_options),
            "tts_options": _json_safe(self.tts_options),
        }
        if include_audio and isinstance(self.audio, bytes):
            import base64

            payload["audio_base64"] = base64.b64encode(self.audio).decode("ascii")
        elif include_audio and isinstance(self.audio, str):
            payload["audio"] = self.audio
        return payload


@dataclass(frozen=True)
class TelephoneTurnState:
    """Privacy-safe state supplied by a telephone webhook or SIP adapter.

    The state deliberately carries no caller audio or transcript. ``call_id``
    is retained in memory so an adapter can correlate turns, but serialized
    receipts expose only its SHA-256 digest. A caller advances the immutable
    state after each completed adapter invocation.
    """

    call_id: str
    turn_index: int = 0
    max_turns: int = 12
    barge_in: bool = False
    previous_response_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        call_id = str(self.call_id or "").strip()
        if not call_id:
            raise ValueError("TelephoneTurnState.call_id must be non-empty")
        if isinstance(self.turn_index, bool) or int(self.turn_index) < 0:
            raise ValueError("TelephoneTurnState.turn_index must be non-negative")
        if isinstance(self.max_turns, bool) or int(self.max_turns) < 1:
            raise ValueError("TelephoneTurnState.max_turns must be at least 1")
        if not isinstance(self.barge_in, bool):
            raise TypeError("TelephoneTurnState.barge_in must be a boolean")
        previous = (
            str(self.previous_response_sha256).strip().lower()
            if self.previous_response_sha256 is not None
            else ""
        )
        if previous and not re.fullmatch(r"[0-9a-f]{64}", previous):
            raise ValueError(
                "TelephoneTurnState.previous_response_sha256 must be a SHA-256 hex digest"
            )
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "turn_index", int(self.turn_index))
        object.__setattr__(self, "max_turns", int(self.max_turns))
        object.__setattr__(self, "previous_response_sha256", previous or None)

    @property
    def call_id_sha256(self) -> str:
        return _sha256_text(self.call_id)

    def to_context(self) -> Dict[str, object]:
        """Return context safe for GraphRAG collaborators and receipts."""

        return {
            "surface": "telephone",
            "telephone_contract_version": TELEPHONE_TURN_CONTRACT_VERSION,
            "call_id_sha256": self.call_id_sha256,
            "turn_index": self.turn_index,
            "max_turns": self.max_turns,
            "barge_in": self.barge_in,
            "previous_response_sha256": self.previous_response_sha256,
        }

    def to_dict(self) -> Dict[str, object]:
        return dict(self.to_context())

    def advance(
        self,
        result: "VoiceTurnResult",
        *,
        barge_in: bool = False,
    ) -> "TelephoneTurnState":
        """Return the next immutable state without retaining response text."""

        if not isinstance(result, VoiceTurnResult):
            raise TypeError("result must be a VoiceTurnResult")
        return TelephoneTurnState(
            call_id=self.call_id,
            turn_index=self.turn_index + 1,
            max_turns=self.max_turns,
            barge_in=barge_in,
            previous_response_sha256=result.provenance.response_text_sha256,
        )


@dataclass(frozen=True)
class VoiceTurnProvenance:
    """Machine provenance retained separately from citation-free speech."""

    stt_provider: Optional[str] = None
    template_provider: Optional[str] = None
    template_id: Optional[str] = None
    tts_provider: Optional[str] = None
    evidence: Tuple[GroundingEvidence, ...] = ()
    grounded_slots: Tuple[GroundedSlot, ...] = ()
    input_audio_sha256: Optional[str] = None
    transcript_sha256: Optional[str] = None
    response_text_sha256: Optional[str] = None
    output_audio_sha256: Optional[str] = None
    pipeline: str = "abby-grounded-voice-v1"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "stt_provider",
            "template_provider",
            "template_id",
            "tts_provider",
            "input_audio_sha256",
            "transcript_sha256",
            "response_text_sha256",
            "output_audio_sha256",
        ):
            raw_value = getattr(self, field_name)
            normalized = (
                str(raw_value).strip() if raw_value is not None else ""
            )
            object.__setattr__(self, field_name, normalized or None)
        pipeline = str(self.pipeline or "").strip()
        if not pipeline:
            raise ValueError("VoiceTurnProvenance.pipeline must be non-empty")
        if any(
            not isinstance(item, GroundingEvidence) for item in (self.evidence or ())
        ):
            raise TypeError("VoiceTurnProvenance.evidence entries must be GroundingEvidence")
        if any(
            not isinstance(item, GroundedSlot) for item in (self.grounded_slots or ())
        ):
            raise TypeError("VoiceTurnProvenance.grounded_slots entries must be GroundedSlot")
        object.__setattr__(self, "pipeline", pipeline)
        object.__setattr__(self, "evidence", tuple(self.evidence or ()))
        object.__setattr__(self, "grounded_slots", tuple(self.grounded_slots or ()))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "contract_version": VOICE_TURN_CONTRACT_VERSION,
            "pipeline": self.pipeline,
            "stt_provider": self.stt_provider,
            "template_provider": self.template_provider,
            "template_id": self.template_id,
            "tts_provider": self.tts_provider,
            "evidence": [item.to_dict() for item in self.evidence],
            "grounded_slots": [slot.to_dict() for slot in self.grounded_slots],
            "input_audio_sha256": self.input_audio_sha256,
            "transcript_sha256": self.transcript_sha256,
            "response_text_sha256": self.response_text_sha256,
            "output_audio_sha256": self.output_audio_sha256,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class VoiceTurnResult:
    """Complete, JSON-serializable receipt for a unified voice turn."""

    request_id: str
    status: str
    transcript: str
    response_text: str
    audio: Optional[bytes]
    audio_format: Optional[str]
    provenance: VoiceTurnProvenance
    traces: Tuple[VoiceStageTrace, ...] = ()
    fallback_reasons: Tuple[str, ...] = ()
    cache_key: Optional[str] = None

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        status = str(self.status or "").strip().lower()
        if not request_id:
            raise ValueError("VoiceTurnResult.request_id must be non-empty")
        if status not in VOICE_TURN_STATUSES:
            raise ValueError(
                "VoiceTurnResult.status must be one of "
                + ", ".join(sorted(VOICE_TURN_STATUSES))
            )
        if not isinstance(self.transcript, str):
            raise TypeError("VoiceTurnResult.transcript must be a string")
        if not isinstance(self.response_text, str) or not self.response_text.strip():
            raise ValueError("VoiceTurnResult.response_text must be non-empty")
        if self.audio is not None and (
            not isinstance(self.audio, bytes) or not self.audio
        ):
            raise TypeError("VoiceTurnResult.audio must be non-empty bytes or None")
        if not isinstance(self.provenance, VoiceTurnProvenance):
            raise TypeError("VoiceTurnResult.provenance must be VoiceTurnProvenance")
        traces = tuple(self.traces or ())
        if any(not isinstance(trace, VoiceStageTrace) for trace in traces):
            raise TypeError("VoiceTurnResult.traces entries must be VoiceStageTrace")
        reasons = tuple(
            dict.fromkeys(
                str(reason).strip()
                for reason in (self.fallback_reasons or ())
                if str(reason).strip()
            )
        )
        audio_format = (
            str(self.audio_format).strip().lower().lstrip(".")
            if self.audio_format is not None
            else None
        )
        cache_key = (
            str(self.cache_key).strip() if self.cache_key is not None else None
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "response_text", self.response_text.strip())
        object.__setattr__(self, "audio_format", audio_format or None)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(self, "cache_key", cache_key or None)

    @property
    def spoken_text(self) -> str:
        return self.response_text

    @property
    def fallbacks(self) -> Tuple[str, ...]:
        """Compatibility alias for early objective drafts."""
        return self.fallback_reasons

    @property
    def fallback_reason(self) -> Optional[str]:
        """Primary degradation reason for clients that display one reason."""
        return self.fallback_reasons[0] if self.fallback_reasons else None

    @property
    def degraded(self) -> bool:
        return self.status != "completed"

    @property
    def template_id(self) -> Optional[str]:
        return self.provenance.template_id

    @property
    def intent(self) -> Optional[str]:
        value = self.provenance.metadata.get("intent")
        return str(value) if value is not None else None

    @property
    def sources(self) -> Tuple[GroundingEvidence, ...]:
        return self.provenance.evidence

    @property
    def total_duration_ms(self) -> float:
        return round(sum(trace.duration_ms for trace in self.traces), 3)

    @property
    def provider_selection(self) -> Dict[str, Optional[str]]:
        return {
            "transcription": self.provenance.stt_provider,
            "retrieval": self.provenance.template_provider,
            "synthesis": self.provenance.tts_provider,
        }

    def validated_cache_miss_event(
        self,
        *,
        validation_receipt_id: str,
        response_id: str = "",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Optional["VoiceCacheMissEvent"]:
        """Emit the deterministic event for a validated live-TTS cache miss.

        Cache hits and text-only turns return ``None``. The validation receipt
        is mandatory so callers cannot accidentally make unvalidated audio
        eligible for a response-DAG append.
        """

        receipt_id = str(validation_receipt_id or "").strip()
        if not receipt_id:
            raise ValueError("validation_receipt_id must be non-empty")
        # Imported here because the dependency-light event module is imported
        # at the end of this module to avoid a router/result import cycle.
        from .voice_cache_miss import build_voice_cache_miss_event

        return build_voice_cache_miss_event(
            self,
            response_id=response_id,
            validation_receipt_id=receipt_id,
            validation_passed=True,
            metadata=dict(metadata or {}),
        )

    def to_dict(self, *, include_audio: bool = False) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "contract_version": VOICE_TURN_CONTRACT_VERSION,
            "request_id": self.request_id,
            "status": self.status,
            "degraded": self.degraded,
            "transcript": self.transcript,
            "response_text": self.response_text,
            "spoken_text": self.spoken_text,
            "audio_format": self.audio_format,
            "audio_size_bytes": len(self.audio) if self.audio is not None else 0,
            "provenance": self.provenance.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "fallback_reasons": list(self.fallback_reasons),
            "fallback_reason": self.fallback_reason,
            "provider_selection": self.provider_selection,
            "total_duration_ms": self.total_duration_ms,
            "cache_key": self.cache_key,
        }
        if include_audio and self.audio is not None:
            import base64

            payload["audio_base64"] = base64.b64encode(self.audio).decode("ascii")
        return payload


def _coerce_evidence(value: object, *, default_id: Optional[str] = None) -> GroundingEvidence:
    if isinstance(value, GroundingEvidence):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("GraphRAG evidence entries must be mappings")
    source_id = (
        value.get("source_id")
        or value.get("id")
        or default_id
        or value.get("cid")
        or value.get("uri")
    )
    metadata = (
        dict(value.get("metadata"))
        if isinstance(value.get("metadata"), Mapping)
        else {}
    )
    for key, item in value.items():
        if key not in {
            "source_id",
            "id",
            "cid",
            "uri",
            "text",
            "excerpt",
            "facts",
            "metadata",
        }:
            metadata.setdefault(str(key), item)
    return GroundingEvidence(
        source_id=str(source_id or ""),
        cid=str(value["cid"]) if value.get("cid") is not None else None,
        uri=str(value["uri"]) if value.get("uri") is not None else None,
        text=str(value.get("text") or value.get("excerpt") or "") or None,
        facts=value.get("facts") if isinstance(value.get("facts"), Mapping) else {},
        metadata=metadata,
    )


def _normalize_evidence(raw: object) -> Tuple[GroundingEvidence, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("source_id", "id", "cid", "uri", "facts")):
            return (_coerce_evidence(raw),)
        return tuple(
            _coerce_evidence(value, default_id=str(key))
            for key, value in raw.items()
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(_coerce_evidence(value) for value in raw)
    raise ValueError("GraphRAG evidence must be a mapping or sequence")


def _source_ids_for_fact(
    name: str,
    value: object,
    evidence: Sequence[GroundingEvidence],
) -> Tuple[str, ...]:
    exact = tuple(
        item.source_id
        for item in evidence
        if name in item.facts and item.facts[name] == value
    )
    if exact:
        return exact
    # Some stores stringify scalar Arrow/Parquet values during retrieval.
    return tuple(
        item.source_id
        for item in evidence
        if name in item.facts and str(item.facts[name]) == str(value)
    )


def _coerce_response_plan(value: object) -> VoiceResponsePlan:
    if isinstance(value, VoiceResponsePlan):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("GraphRAG response plan must be a mapping")

    evidence_raw = value.get("evidence", value.get("sources"))
    if evidence_raw is None and isinstance(value.get("provenance"), Mapping):
        evidence_raw = value["provenance"].get("evidence") or value["provenance"].get("sources")
    evidence = _normalize_evidence(evidence_raw)
    raw_slots = value.get("slots") or ()
    slot_sources = value.get("slot_sources")
    slot_sources = slot_sources if isinstance(slot_sources, Mapping) else {}
    slots = []
    if isinstance(raw_slots, Mapping):
        raw_slot_items = raw_slots.items()
    elif isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)):
        raw_slot_items = enumerate(raw_slots)
    else:
        raise ValueError("GraphRAG response-plan slots must be a mapping or sequence")

    for key, raw_slot in raw_slot_items:
        if isinstance(raw_slot, GroundedSlot):
            slots.append(raw_slot)
            continue
        if isinstance(raw_slot, Mapping):
            name = str(raw_slot.get("name") or key)
            slot_value = raw_slot.get("value")
            raw_source_ids = (
                raw_slot.get("source_ids")
                or raw_slot.get("evidence_ids")
                or raw_slot.get("citations")
                or slot_sources.get(name)
                or ()
            )
        else:
            name = str(key)
            slot_value = raw_slot
            raw_source_ids = slot_sources.get(name) or ()
        if isinstance(raw_source_ids, str):
            source_ids = (raw_source_ids,)
        else:
            source_ids = tuple(str(item) for item in raw_source_ids)
        if not source_ids:
            source_ids = _source_ids_for_fact(name, slot_value, evidence)
        slots.append(GroundedSlot(name=name, value=slot_value, source_ids=source_ids))

    return VoiceResponsePlan(
        template_id=str(value.get("template_id") or value.get("id") or ""),
        template=str(
            value.get("template")
            or value.get("template_text")
            or value.get("response_frame")
            or ""
        ),
        slots=tuple(slots),
        evidence=evidence,
        confidence=float(value.get("confidence", value.get("score", 1.0))),
        intent=str(value["intent"]) if value.get("intent") is not None else None,
        metadata=value.get("metadata")
        if isinstance(value.get("metadata"), Mapping)
        else {},
    )


def _call_with_supported_keywords(
    function: Callable[..., object],
    first_arg: str,
    **kwargs: object,
) -> object:
    """Call an injected adapter while tolerating older narrow signatures."""
    try:
        import inspect

        signature = inspect.signature(function)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        selected = (
            kwargs
            if accepts_kwargs
            else {name: value for name, value in kwargs.items() if name in signature.parameters}
        )
    except (TypeError, ValueError):
        selected = kwargs
    return function(first_arg, **selected)


class GraphRAGVoiceTemplateProvider:
    """Lazy adapter over an ``ipfs_datasets_py`` GraphRAG retriever.

    The backend is injected deliberately: importing this module never imports
    GraphRAG, IPLD, vector-store, or model dependencies. Supported backend
    methods are ``retrieve_voice_template``, ``retrieve_template``, ``retrieve``
    and ``search``; a callable backend is also accepted.
    """

    provider_name = "graphrag"

    def __init__(self, backend: object, *, minimum_confidence: float = 0.0) -> None:
        if backend is None:
            raise ValueError("GraphRAG backend must be provided")
        self.backend = backend
        self.minimum_confidence = float(minimum_confidence)
        self.last_prompt_parts: Optional[Mapping[str, object]] = None

    def _backend_callable(self) -> Callable[..., object]:
        for method_name in (
            "retrieve_voice_template",
            "retrieve_template",
            "retrieve",
            "search",
        ):
            method = getattr(self.backend, method_name, None)
            if callable(method):
                return method
        if callable(self.backend):
            return self.backend
        raise TypeError("GraphRAG backend has no supported retrieval method")

    def retrieve(
        self,
        transcript: str,
        *,
        context: Optional[Mapping[str, object]] = None,
        language: Optional[str] = None,
        grounding: Optional[Union[Mapping[str, object], Sequence[object]]] = None,
        max_results: int = 5,
    ) -> Optional[VoiceResponsePlan]:
        # Build this envelope at the router boundary so every injected backend
        # sees the same canonical query contract.  It is retained for audit
        # and debugging, but is never treated as a generated answer or slot
        # source.  Backends that explicitly opt into ``prompt_parts`` receive
        # it below; older backends keep their narrow, compatible signatures.
        prompt_parts = buildVoiceGraphRagPromptParts(
            transcript,
            context=context,
            language=language,
            grounding=grounding,
            max_results=max_results,
        )
        self.last_prompt_parts = prompt_parts
        backend = self._backend_callable()
        backend_kwargs: Dict[str, object] = {
            "context": dict(prompt_parts["context"]),
            "language": prompt_parts["language"],
            "grounding": (
                dict(grounding)
                if isinstance(grounding, Mapping)
                else list(grounding or ())
            ),
            "max_results": prompt_parts["max_results"],
        }
        try:
            import inspect

            signature = inspect.signature(backend)
            if "prompt_parts" in signature.parameters:
                backend_kwargs["prompt_parts"] = dict(prompt_parts)
        except (TypeError, ValueError):
            pass
        raw = _call_with_supported_keywords(
            backend,
            prompt_parts["query"],
            **backend_kwargs,
        )
        if raw is None:
            return None
        candidates: Sequence[object]
        if isinstance(raw, Mapping):
            nested = (
                raw.get("candidates")
                or raw.get("results")
                or raw.get("items")
                or raw.get("templates")
            )
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                candidates = nested
            else:
                candidates = (raw,)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            candidates = raw
        else:
            candidates = (raw,)

        plans = [_coerce_response_plan(candidate) for candidate in candidates]
        plans = [plan for plan in plans if plan.confidence >= self.minimum_confidence]
        if not plans:
            return None
        return max(plans, key=lambda plan: plan.confidence)


# ---------------------------------------------------------------------------
# Built-in provider implementations
# ---------------------------------------------------------------------------

def _get_openai_provider() -> Optional[VoiceProvider]:
    """Get OpenAI voice provider (TTS via /audio/speech + STT via /audio/transcriptions)."""
    credential_value = _coalesce_env(
        "IPFS_ACCELERATE_PY_OPENAI_API_KEY", "OPENAI_API_KEY"
    )
    if not credential_value:
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
                    "Authorization": "Bearer " + credential_value,
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
                    "Authorization": "Bearer " + credential_value,
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
    credential_value = _coalesce_env(
        "IPFS_ACCELERATE_PY_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"
    )
    if not credential_value:
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
                    "xi-api-key": credential_value,
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
    credential_value = _coalesce_env(
        "IPFS_ACCELERATE_PY_ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY"
    )
    if not credential_value:
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
                            "authorization": credential_value,
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
                        "authorization": credential_value,
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
                    "authorization": credential_value,
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
                    headers={"authorization": credential_value},
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

            model = model_name or os.getenv(
                "IPFS_ACCELERATE_PY_TTS_MODEL", "suno/bark-small"
            )
            device_str = (
                device
                or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE")
                or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE")
                or "cpu"
            )

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

            model = model_name or os.getenv(
                "IPFS_ACCELERATE_PY_STT_MODEL", "openai/whisper-base"
            )
            device_str = (
                device
                or os.getenv("IPFS_ACCELERATE_PY_STT_DEVICE")
                or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE")
                or "cpu"
            )

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
            english_only_whisper = (
                "whisper" in model.casefold()
                and model.casefold().rsplit("/", 1)[-1].endswith(".en")
            )
            if language and not english_only_whisper:
                selected_language = str(language).strip()
                if "whisper" in model.casefold():
                    # Whisper accepts ISO-639-1 codes/names, not regional
                    # dataset locales such as en-US.
                    selected_language = selected_language.split("-", 1)[0].casefold()
                generate_kwargs["language"] = selected_language

            pipeline_kwargs: dict[str, object] = {}
            for name in ("chunk_length_s", "stride_length_s", "return_timestamps"):
                if name in kwargs:
                    pipeline_kwargs[name] = kwargs[name]
            if "whisper" in model.casefold():
                # Transformers automatically enters Whisper long-form mode for
                # inputs over 30 seconds, which requires timestamp tokens.
                pipeline_kwargs.setdefault("return_timestamps", True)
            if generate_kwargs:
                pipeline_kwargs["generate_kwargs"] = generate_kwargs

            result = pipe(
                audio_input,
                **pipeline_kwargs,
            )

            if isinstance(result, dict):
                return str(result.get("text", "") or "")
            return str(result or "")

    return _HuggingFaceVoiceProvider()


def _await_from_sync(value: object) -> object:
    """Resolve an awaitable without changing the synchronous voice API.

    ``InferenceBackendManager.execute_task`` is async, while the legacy voice
    entrypoints are intentionally synchronous.  A nested ``asyncio.run`` is
    invalid when callers already have an event loop, so that case is isolated
    in a short-lived worker thread with its own loop.
    """
    import asyncio
    import inspect

    if not inspect.isawaitable(value):
        return value

    async def _wait() -> object:
        return await value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_wait())

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="voice-backend-manager",
    ) as executor:
        return executor.submit(asyncio.run, _wait()).result()


_BACKEND_RESULT_MISSING = object()


def _backend_manager_result_value(
    result: object,
    *keys: str,
) -> object:
    """Read a backend value from canonical or recorder-wrapped results."""
    if not isinstance(result, Mapping):
        return result

    for key in keys:
        if key in result:
            return result[key]

    nested = result.get("result", _BACKEND_RESULT_MISSING)
    if isinstance(nested, Mapping):
        for key in keys:
            if key in nested:
                return nested[key]
        return _BACKEND_RESULT_MISSING
    return nested


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
            provider_name = "backend_manager"

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

                model = model_name or os.getenv("IPFS_ACCELERATE_PY_TTS_MODEL", "")
                payload: Dict[str, object] = {
                    "text": str(text),
                    "device": device,
                    **kwargs,
                }
                if voice:
                    payload["voice"] = voice
                if output_format:
                    payload["output_format"] = output_format

                result = _await_from_sync(manager.execute_task(
                    task="text-to-speech",
                    model=model,
                    inputs=[str(text)],
                    parameters=payload,
                ))

                audio = _backend_manager_result_value(
                    result,
                    "audio",
                    "audio_bytes",
                    "audio_b64",
                )
                if isinstance(audio, bytes):
                    return audio
                if isinstance(audio, (bytearray, memoryview)):
                    return bytes(audio)
                if isinstance(audio, str):
                    try:
                        return base64.b64decode(audio, validate=True)
                    except Exception as exc:
                        raise RuntimeError(
                            "Backend manager TTS provider returned invalid base64 audio"
                        ) from exc
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

                if isinstance(audio, bytes):
                    audio_payload: object = base64.b64encode(audio).decode("ascii")
                else:
                    audio_payload = audio

                model = model_name or os.getenv("IPFS_ACCELERATE_PY_STT_MODEL", "")
                payload: Dict[str, object] = {
                    "audio": audio_payload,
                    "device": device,
                    **kwargs,
                }
                if language:
                    payload["language"] = language

                result = _await_from_sync(manager.execute_task(
                    task="automatic-speech-recognition",
                    model=model,
                    inputs=[audio_payload],
                    parameters=payload,
                ))

                text = _backend_manager_result_value(
                    result,
                    "text",
                    "transcript",
                    "transcription",
                )
                if text is not _BACKEND_RESULT_MISSING and text is not None:
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
        os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_STT_DEVICE", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_FALLBACK_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TOKEN", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_MODEL", "").strip(),
        os.getenv("WALLET_INDEXTTS_SPACE_URL", "").strip(),
        os.getenv("WALLET_INDEXTTS_FALLBACK_SPACE_URL", "").strip(),
        os.getenv("WALLET_INDEXTTS_MODEL_NAME", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_URLS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_BASE_URL", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_TOKEN", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_MODEL", "").strip(),
        os.getenv("WALLET_HF_WHISPER_BASE_URL", "").strip(),
        os.getenv("WALLET_HF_WHISPER_TOKEN", "").strip(),
        os.getenv("WALLET_HF_WHISPER_MODEL_NAME", "").strip(),
        os.getenv("HF_TOKEN", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_HF_BILL_TO", "").strip(),
        os.getenv("IPFS_DATASETS_PY_HF_BILL_TO", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TIMEOUT_SECONDS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_TIMEOUT_SECONDS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_MAX_RETRIES", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_WHISPER_MAX_RETRIES", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_MAX_RETRIES", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_RETRY_BACKOFF_SECONDS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_RETRY_BACKOFF_MULTIPLIER", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_RETRY_MAX_BACKOFF_SECONDS", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_CIRCUIT_FAILURE_THRESHOLD", "").strip(),
        os.getenv("IPFS_ACCELERATE_PY_ABBY_CIRCUIT_RECOVERY_SECONDS", "").strip(),
    )


def _builtin_provider_by_name(name: str, deps: RouterDeps) -> Optional[VoiceProvider]:
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in {"abby_indextts", "abby_index_tts", "indextts"}:
        from .voice_providers.abby import IndexTTSHTTPProvider

        return IndexTTSHTTPProvider.from_environment()
    if key in {"abby_whisper", "abby_hf_whisper", "hf_whisper"}:
        from .voice_providers.abby import HuggingFaceWhisperHTTPProvider

        return HuggingFaceWhisperHTTPProvider.from_environment()
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
        normalized_preferred = str(preferred).strip().lower()
        info = _PROVIDER_REGISTRY.get(normalized_preferred)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(normalized_preferred, deps=deps)
        if builtin is not None:
            return builtin
        raise ValueError(f"Unknown voice provider: {preferred}")

    preferred_env = os.getenv("IPFS_ACCELERATE_PY_VOICE_PROVIDER", "").strip()
    if preferred_env:
        normalized_env = preferred_env.lower()
        info = _PROVIDER_REGISTRY.get(normalized_env)
        if info is not None:
            return info.factory()
        builtin = _builtin_provider_by_name(normalized_env, deps=deps)
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
        normalized_provider = (provider or "").strip().lower()
        registry_revision = _PROVIDER_REGISTRY_REVISIONS.get(
            normalized_provider, 0
        )
        deps_key = (
            f"voice_provider::{normalized_provider}::revision-{registry_revision}"
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
# Unified grounded voice-turn orchestration
# ---------------------------------------------------------------------------

def _provider_display_name(provider: object, fallback: Optional[str] = None) -> str:
    for attribute in ("name", "provider_name"):
        value = getattr(provider, attribute, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    if fallback:
        return fallback
    name = provider.__class__.__name__.strip("_")
    return name or "unknown"


def _template_provider_name(provider: Optional[object]) -> Optional[str]:
    if provider is None:
        return None
    return _provider_display_name(provider, "template_provider")


def _collaborator_cache_identity(
    collaborator: Optional[object],
    fallback: Optional[str] = None,
) -> Optional[str]:
    if collaborator is None:
        return fallback
    explicit = getattr(collaborator, "cache_identity", None)
    if callable(explicit):
        explicit = explicit()
    collaborator_type = collaborator.__class__
    type_name = f"{collaborator_type.__module__}.{collaborator_type.__qualname__}"
    if explicit is not None and str(explicit).strip():
        return f"{type_name}::{str(explicit).strip()}"
    return f"{type_name}::{id(collaborator)}"


def _safe_stage_error(
    error: Exception, *, sensitive_values: Sequence[object] = ()
) -> str:
    """Normalize adapter errors without embedding caller audio or tracebacks."""
    message = " ".join(str(error).replace("\x00", "").split())
    # Credentials occasionally appear as URL query values in remote errors.
    message = re.sub(
        r"(?i)(api[_-]?key|token|authorization|secret)=?[\w.+/=-]+",
        r"\1=[redacted]",
        message,
    )
    message = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+|bearer\s+)[^\s,;]+",
        r"\1[redacted]",
        message,
    )
    message = re.sub(
        r"(?i)([?&](?:api[_-]?key|token|access_token|secret)=)[^&#\s]+",
        r"\1[redacted]",
        message,
    )
    for sensitive in sensitive_values:
        if isinstance(sensitive, bytes):
            sample = (
                sensitive
                if len(sensitive) <= 8192
                else sensitive[:4096] + sensitive[-4096:]
            )
            decoded = sample.decode("utf-8", errors="ignore")
            fragments = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{7,}", decoded)
            for fragment in tuple(fragments):
                pieces = fragment.split("-")
                fragments.extend(
                    "-".join(pieces[index:])
                    for index in range(1, len(pieces))
                    if len("-".join(pieces[index:])) >= 8
                )
        else:
            decoded = str(sensitive or "").strip()
            fragments = [decoded] if len(decoded) >= 8 else []
        for fragment in sorted(set(fragments), key=len, reverse=True):
            message = message.replace(fragment, "[redacted-input]")
    if len(message) > 240:
        message = message[:237] + "..."
    prefix = error.__class__.__name__
    return f"{prefix}: {message}" if message else prefix


def _duration_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


def _voice_turn_cache_key(
    request: VoiceTurnRequest,
    template_provider: Optional[object],
    *,
    stt_provider: Optional[object] = None,
    tts_provider: Optional[object] = None,
    fallback_template_provider: Optional[object] = None,
) -> str:
    payload = {
        "pipeline": "abby-grounded-voice-v1",
        "audio_sha256": request.input_audio_sha256,
        "transcript_sha256": _sha256_text(request.transcript)
        if request.transcript
        else None,
        "context": _json_safe(request.context),
        "grounding": _json_safe(request.grounding),
        "language": request.effective_language,
        "voice": request.voice,
        "stt_provider": request.stt_provider,
        "tts_provider": request.tts_provider,
        "stt_providers": request.stt_providers,
        "tts_providers": request.tts_providers,
        "stt_model": request.stt_model,
        "tts_model": request.tts_model,
        "device": request.device,
        "output_format": request.output_format,
        "minimum_template_confidence": request.minimum_template_confidence,
        "max_template_results": request.max_template_results,
        "fallback_text_sha256": _sha256_text(request.fallback_text),
        "stt_options": _json_safe(request.stt_options),
        "tts_options": _json_safe(request.tts_options),
        "stt_provider_instance": _collaborator_cache_identity(
            stt_provider, request.stt_provider
        ),
        "tts_provider_instance": _collaborator_cache_identity(
            tts_provider, request.tts_provider
        ),
        "template_provider": _collaborator_cache_identity(
            template_provider, _template_provider_name(template_provider)
        ),
        "fallback_template_provider": _collaborator_cache_identity(
            fallback_template_provider,
            _template_provider_name(fallback_template_provider),
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return f"abby_voice_turn::{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def voice_turn_cache_key(
    request: VoiceTurnRequest,
    *,
    template_provider: Optional[VoiceTemplateProvider] = None,
    fallback_template_provider: Optional[VoiceTemplateProvider] = None,
    stt_provider: Optional[VoiceProvider] = None,
    tts_provider: Optional[VoiceProvider] = None,
) -> str:
    """Return the privacy-safe identity used for a voice-turn receipt."""
    if not isinstance(request, VoiceTurnRequest):
        raise TypeError("request must be a VoiceTurnRequest")
    return _voice_turn_cache_key(
        request,
        template_provider,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        fallback_template_provider=fallback_template_provider,
    )


def _registry_supports(name: str, operation: str) -> bool:
    normalized_name = str(name or "").strip().lower()
    info = _PROVIDER_REGISTRY.get(normalized_name)
    if info is not None:
        return info.capabilities.supports(operation)
    builtin_name = _BUILTIN_PROVIDER_ALIASES.get(normalized_name, normalized_name)
    capabilities = _BUILTIN_PROVIDER_CAPABILITIES.get(builtin_name)
    return capabilities.supports(operation) if capabilities is not None else True


def _provider_candidates(
    primary: Optional[VoiceProvider],
    *,
    preferred: Optional[str],
    fallbacks: Sequence[str],
    operation: str,
    deps: RouterDeps,
) -> Tuple[Tuple[str, Optional[VoiceProvider], Optional[Exception]], ...]:
    """Resolve an ordered, de-duplicated provider chain without invoking it."""
    candidates: list[Tuple[str, Optional[VoiceProvider], Optional[Exception]]] = []
    identities = set()
    if primary is not None:
        label = _provider_display_name(primary, "injected")
        candidates.append((label, primary, None))
        identities.add(id(primary))

    names = tuple(name for name in ((preferred,) + tuple(fallbacks)) if name)
    for name in names:
        if not _registry_supports(name, operation):
            continue
        try:
            provider = get_voice_provider(name, deps=deps)
            if id(provider) in identities:
                continue
            identities.add(id(provider))
            candidates.append((name, provider, None))
        except Exception as error:
            candidates.append((name, None, error))

    # An explicit provider chain is authoritative even when every named
    # provider was filtered out by capabilities. Falling through to automatic
    # selection would invoke an unrequested provider and defeat policy.
    explicit_chain = primary is not None or bool(preferred) or bool(fallbacks)
    if not candidates and not explicit_chain:
        try:
            provider = get_voice_provider(None, deps=deps)
            candidates.append((_provider_display_name(provider, "auto"), provider, None))
        except Exception as error:
            candidates.append(("auto", None, error))
    return tuple(candidates)


def _provider_receipt_details(provider: object) -> Dict[str, object]:
    """Return a provider's privacy-safe last-call receipt, when available."""
    receipt = getattr(provider, "last_receipt", None)
    to_dict = getattr(receipt, "to_dict", None)
    if not callable(to_dict):
        return {}
    try:
        value = to_dict()
    except Exception:
        return {}
    return {"provider_receipt": _json_safe(value)}


def _close_awaitable_result(value: object) -> None:
    """Close an unexpected coroutine returned through the synchronous API."""
    if not hasattr(value, "__await__"):
        return
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _grounding_override_slots(
    plan: VoiceResponsePlan,
    grounding: Mapping[str, object],
) -> Tuple[GroundedSlot, ...]:
    if not grounding:
        return plan.slots
    slots = {slot.name: slot for slot in plan.slots}
    for name, raw in grounding.items():
        slot_name = str(name)
        if isinstance(raw, GroundedSlot):
            slots[slot_name] = GroundedSlot(
                name=slot_name,
                value=raw.value,
                source_ids=raw.source_ids,
            )
            continue
        if isinstance(raw, Mapping) and "value" in raw:
            value = raw.get("value")
            source_ids_raw = raw.get("source_ids") or raw.get("evidence_ids") or ()
            source_ids = (
                (source_ids_raw,)
                if isinstance(source_ids_raw, str)
                else tuple(str(item) for item in source_ids_raw)
            )
        else:
            value = raw
            source_ids = ()
        if not source_ids:
            source_ids = _source_ids_for_fact(slot_name, value, plan.evidence)
        slots[slot_name] = GroundedSlot(slot_name, value, source_ids)
    return tuple(slots.values())


def _template_fields(template: str) -> Tuple[str, ...]:
    try:
        return template_fields(template)
    except ValueError as error:
        raise ValueError(f"invalid_template_slot: {error}") from error


_TELEPHONE_DIGIT_WORDS: Mapping[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_TELEPHONE_UNSAFE_SPOKEN_MARKER_RE = re.compile(
    r"\b(?:negative|open\s+parenthes(?:is|es)|close\s+parenthes(?:is|es)|"
    r"left\s+parenthes(?:is|es)|right\s+parenthes(?:is|es)|hyphen|dash)\b",
    re.IGNORECASE,
)


def _telephone_digits_to_words(value: str) -> str:
    return " ".join(
        _TELEPHONE_DIGIT_WORDS[character]
        for character in value
        if character in _TELEPHONE_DIGIT_WORDS
    )


def _normalize_telephone_slot_value(name: str, value: object) -> str:
    """Normalize phone/address facts before they reach a telephone TTS call."""

    spoken = " ".join(str(value or "").split())
    slot_name = str(name or "").strip().casefold()
    if slot_name in {"phone", "phone_number", "telephone", "telephone_number"}:
        digits = re.sub(r"\D", "", spoken)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) == 10:
            spoken = (
                f"{_telephone_digits_to_words(digits[:3])}, "
                f"{_telephone_digits_to_words(digits[3:6])}, "
                f"{_telephone_digits_to_words(digits[6:])}"
            )
        elif digits:
            spoken = _telephone_digits_to_words(digits)
    elif slot_name in {"address", "street_address", "location_address"}:
        # A numeric address range such as ``11-32`` is an address identifier,
        # not subtraction. Joining and spelling the digits prevents IndexTTS
        # and downstream telephone codecs from producing "negative".
        spoken = re.sub(
            r"(?<!\w)\d+(?:\s*[-–—]\s*\d+)+(?!\w)",
            lambda match: _telephone_digits_to_words(
                re.sub(r"\D", "", match.group(0))
            ),
            spoken,
        )
        spoken = re.sub(
            r"^\d{1,6}\b",
            lambda match: _telephone_digits_to_words(match.group(0)),
            spoken,
        )
        spoken = re.sub(
            r"\b(?P<zip>\d{5})(?:\s*[-–—]\s*(?P<plus4>\d{4}))?\b",
            lambda match: (
                f"ZIP code {_telephone_digits_to_words(match.group('zip'))}"
                + (
                    f" {_telephone_digits_to_words(match.group('plus4'))}"
                    if match.group("plus4")
                    else ""
                )
            ),
            spoken,
        )

    # Parenthetical punctuation is visual structure. Preserve its content
    # without asking a speech provider to pronounce the delimiters.
    spoken = re.sub(
        r"\((?P<content>[^()]*)\)",
        lambda match: f", {match.group('content').strip()}, "
        if match.group("content").strip()
        else " ",
        spoken,
    )
    return " ".join(spoken.split())


def _assert_telephone_spoken_safety(text: str) -> None:
    """Fail closed before synthesis when telephone speech retains a trap."""

    if "(" in text or ")" in text:
        raise ValueError("telephone_spoken_parenthesis_marker")
    if _TELEPHONE_UNSAFE_SPOKEN_MARKER_RE.search(text):
        raise ValueError("telephone_spoken_negative_or_punctuation_marker")
    if re.search(r"\d\s*[-–—]\s*\d", text):
        raise ValueError("telephone_spoken_numeric_hyphenation")


def _render_grounded_plan(
    plan: VoiceResponsePlan,
    *,
    grounding: Mapping[str, object],
    minimum_confidence: float,
    telephone_safe: bool = False,
) -> Tuple[str, Tuple[GroundedSlot, ...]]:
    if plan.confidence < minimum_confidence:
        raise ValueError(
            "template_below_confidence: "
            f"{plan.confidence:.3f} < {minimum_confidence:.3f}"
        )

    fields = _template_fields(plan.template)
    slots = _grounding_override_slots(plan, grounding)
    slots_by_name: Dict[str, GroundedSlot] = {}
    duplicate_names = set()
    for slot in slots:
        if slot.name in slots_by_name:
            duplicate_names.add(slot.name)
        slots_by_name[slot.name] = slot
    if duplicate_names:
        raise ValueError(
            "duplicate_template_slots: " + ", ".join(sorted(duplicate_names))
        )

    missing = [name for name in fields if name not in slots_by_name]
    if missing:
        raise ValueError("missing_template_slots: " + ", ".join(missing))

    evidence_by_id = {item.source_id: item for item in plan.evidence}
    rendered_values: Dict[str, str] = {}
    for name in fields:
        slot = slots_by_name[name]
        if slot.value is None or not str(slot.value).strip():
            raise ValueError(f"ungrounded_slot: {name} has an empty value")
        if not slot.source_ids:
            raise ValueError(f"ungrounded_slot: {name} has no evidence source")
        unknown = [source_id for source_id in slot.source_ids if source_id not in evidence_by_id]
        if unknown:
            raise ValueError(
                f"ungrounded_slot: {name} cites unknown sources {', '.join(unknown)}"
            )
        fact_sources = [
            evidence_by_id[source_id]
            for source_id in slot.source_ids
            if name in evidence_by_id[source_id].facts
        ]
        fact_matches = [
            source
            for source in fact_sources
            if source.facts[name] == slot.value
            or str(source.facts[name]) == str(slot.value)
        ]
        # A structured fact, when present, must match exactly. Evidence stores
        # that only expose a cited document/excerpt remain usable; the router
        # cannot invent a conflicting value because the slot still has to cite
        # that current record.
        if fact_sources and not fact_matches:
            raise ValueError(
                f"ungrounded_slot: {name} does not match a cited current fact"
            )
        rendered_values[name] = (
            _normalize_telephone_slot_value(name, slot.value)
            if telephone_safe
            else str(slot.value).strip()
        )

    if fields and not plan.evidence:
        raise ValueError("missing_grounding_evidence")
    try:
        rendered = plan.template.format_map(rendered_values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid_template: {error}") from error
    rendered = _normalize_spoken_text(rendered)
    if telephone_safe:
        _assert_telephone_spoken_safety(rendered)
    return rendered, tuple(slots_by_name[name] for name in fields)


def _normalize_spoken_text(text: str) -> str:
    """Remove visual citations while retaining their machine provenance."""
    return normalize_spoken_text(text)


def _audio_format(audio: Optional[bytes], requested: Optional[str]) -> Optional[str]:
    if audio is None:
        return None
    if requested:
        return requested.lower().lstrip(".")
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "wav"
    if audio.startswith(b"ID3") or audio[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    if audio.startswith(b"OggS"):
        return "ogg"
    return "bin"


def _synthesis_identity_from_request(request: VoiceTurnRequest) -> SynthesisIdentity:
    """Derive the full synthesis identity used by the exact audio resolver."""

    options = dict(request.tts_options or {})
    provider = (
        request.tts_provider
        or str(options.get("provider") or "").strip().lower()
        or "precomputed"
    )
    model = (
        request.tts_model
        or str(options.get("model") or options.get("model_name") or "").strip()
        or "default"
    )
    voice = (
        request.voice
        or str(options.get("voice") or "").strip()
        or "default"
    )
    locale = (
        request.locale
        or request.language
        or str(options.get("locale") or options.get("language") or "").strip()
        or "en-US"
    )
    codec = (
        request.output_format
        or str(options.get("codec") or options.get("output_format") or "").strip()
        or "wav"
    )
    provider_version = str(
        options.get("provider_version") or options.get("version") or "unspecified"
    ).strip() or "unspecified"
    sample_rate_hz = int(options.get("sample_rate_hz") or options.get("sample_rate") or 24_000)
    channels = int(options.get("channels") or 1)
    reference = options.get("reference_audio_sha256")
    if reference is None and isinstance(options.get("reference_audio"), Mapping):
        reference = options["reference_audio"].get("sha256")  # type: ignore[index]
    generation_settings = options.get("generation_settings")
    if not isinstance(generation_settings, Mapping):
        generation_settings = {}
    return SynthesisIdentity(
        provider=provider,
        model=model,
        voice=voice,
        provider_version=provider_version,
        locale=locale,
        codec=codec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        reference_audio_sha256=str(reference) if reference is not None else None,
        generation_settings=dict(generation_settings),
    )


def process_voice_turn(
    request: VoiceTurnRequest,
    *,
    stt_provider: Optional[VoiceProvider] = None,
    template_provider: Optional[VoiceTemplateProvider] = None,
    fallback_template_provider: Optional[VoiceTemplateProvider] = None,
    tts_provider: Optional[VoiceProvider] = None,
    stt_provider_instance: Optional[VoiceProvider] = None,
    tts_provider_instance: Optional[VoiceProvider] = None,
    audio_resolver: Optional[PrecomputedVoiceAudioResolver] = None,
    deps: Optional[RouterDeps] = None,
) -> VoiceTurnResult:
    """Run STT → grounded response-plan retrieval → rendering → precomputed/TTS.

    Runtime resolution prefers an injected :class:`PrecomputedVoiceAudioResolver`
    when the rendered spoken text and full synthesis identity match exactly.
    Resolver misses fall through to live TTS or text-only output and never
    serve a near or stale match. Runtime failures are returned as structured
    degraded receipts. Invalid request contracts still raise immediately,
    keeping programmer errors separate from provider availability.

    Runtime caller audio and transcripts are neither cached into the public
    release nor written into ordinary receipts.
    """
    if not isinstance(request, VoiceTurnRequest):
        raise TypeError("request must be a VoiceTurnRequest")
    resolved_deps = deps or get_default_router_deps()
    primary_stt = stt_provider if stt_provider is not None else stt_provider_instance
    primary_tts = tts_provider if tts_provider is not None else tts_provider_instance
    traces: list[VoiceStageTrace] = []
    fallback_reasons: list[str] = []
    cache_key = _voice_turn_cache_key(
        request,
        template_provider,
        stt_provider=primary_stt,
        tts_provider=primary_tts,
        fallback_template_provider=fallback_template_provider,
    )
    request_id = request.request_id or cache_key.rsplit("::", 1)[-1][:24]

    transcript = request.transcript or ""
    used_stt_provider: Optional[str] = None
    if transcript:
        used_stt_provider = "supplied_transcript"
        traces.append(
            VoiceStageTrace(
                "transcription",
                "skipped",
                0.0,
                provider=used_stt_provider,
                details={"reason": "transcript_supplied"},
            )
        )
    else:
        transcription_failures = 0
        stt_candidates = _provider_candidates(
            primary_stt,
            preferred=request.stt_provider,
            fallbacks=request.stt_providers,
            operation="transcription",
            deps=resolved_deps,
        )
        for attempt, (
            provider_name,
            provider_object,
            resolution_error,
        ) in enumerate(stt_candidates, start=1):
            started_at = time.perf_counter()
            attempt_details = {
                "attempt": attempt,
                "retry": attempt > 1,
                "will_retry": attempt < len(stt_candidates),
            }
            if resolution_error is not None or provider_object is None:
                transcription_failures += 1
                traces.append(
                    VoiceStageTrace(
                        "transcription",
                        "failed",
                        _duration_ms(started_at),
                        provider=provider_name,
                        error=_safe_stage_error(
                            resolution_error
                            or RuntimeError("provider could not be resolved")
                        ),
                        details=attempt_details,
                    )
                )
                continue
            try:
                raw_transcript = provider_object.transcribe(
                    request.audio,  # type: ignore[arg-type]
                    model_name=request.stt_model,
                    language=request.effective_language,
                    device=request.device,
                    **dict(request.stt_options),
                )
                if not isinstance(raw_transcript, str) or not raw_transcript.strip():
                    _close_awaitable_result(raw_transcript)
                    raise TypeError("transcribe returned no non-empty text")
                transcript = raw_transcript.strip()
                used_stt_provider = provider_name
                traces.append(
                    VoiceStageTrace(
                        "transcription",
                        "succeeded",
                        _duration_ms(started_at),
                        provider=provider_name,
                        details={
                            **attempt_details,
                            **_provider_receipt_details(provider_object),
                        },
                    )
                )
                provider_receipt = getattr(provider_object, "last_receipt", None)
                if transcription_failures or bool(
                    getattr(provider_receipt, "degraded", False)
                ):
                    fallback_reasons.append("stt_provider_fallback")
                break
            except Exception as error:
                transcription_failures += 1
                traces.append(
                    VoiceStageTrace(
                        "transcription",
                        "failed",
                        _duration_ms(started_at),
                        provider=provider_name,
                        error=_safe_stage_error(
                            error, sensitive_values=(request.audio,)
                        ),
                        details={
                            **attempt_details,
                            **_provider_receipt_details(provider_object),
                        },
                    )
                )
        if not transcript:
            fallback_reasons.append("stt_failed")

    plan: Optional[VoiceResponsePlan] = None
    grounded_slots: Tuple[GroundedSlot, ...] = ()
    template_name = _template_provider_name(template_provider)
    active_template_name = template_name
    if transcript and template_provider is not None:
        started_at = time.perf_counter()
        try:
            raw_plan = _call_with_supported_keywords(
                template_provider.retrieve,
                transcript,
                context=dict(request.context),
                language=request.effective_language,
                grounding=dict(request.grounding),
                max_results=request.max_template_results,
            )
            if raw_plan is None:
                raise LookupError("no grounded response template matched")
            plan = _coerce_response_plan(raw_plan)
            if plan.confidence < request.minimum_template_confidence:
                raise LookupError(
                    "template confidence "
                    f"{plan.confidence:.3f} is below "
                    f"{request.minimum_template_confidence:.3f}"
                )
            traces.append(
                VoiceStageTrace(
                    "retrieval",
                    "succeeded",
                    _duration_ms(started_at),
                    provider=template_name,
                    details={
                        "template_id": plan.template_id,
                        "confidence": plan.confidence,
                        "evidence_count": len(plan.evidence),
                    },
                )
            )
        except Exception as error:
            plan = None
            fallback_reasons.append("template_retrieval_failed")
            traces.append(
                VoiceStageTrace(
                    "retrieval",
                    "failed",
                    _duration_ms(started_at),
                    provider=template_name,
                    error=_safe_stage_error(error),
                )
            )
    else:
        traces.append(
            VoiceStageTrace(
                "retrieval",
                "skipped",
                0.0,
                provider=template_name,
                details={
                    "reason": "transcription_unavailable"
                    if not transcript
                    else "template_provider_unavailable"
                },
            )
        )
        if transcript and template_provider is None:
            fallback_reasons.append("template_provider_unavailable")

    if transcript and plan is None and fallback_template_provider is not None:
        fallback_template_name = _template_provider_name(fallback_template_provider)
        started_at = time.perf_counter()
        try:
            raw_plan = _call_with_supported_keywords(
                fallback_template_provider.retrieve,
                transcript,
                context=dict(request.context),
                language=request.effective_language,
                grounding=dict(request.grounding),
                max_results=request.max_template_results,
            )
            if raw_plan is None:
                raise LookupError("no fallback response template matched")
            plan = _coerce_response_plan(raw_plan)
            if plan.confidence < request.minimum_template_confidence:
                raise LookupError(
                    "fallback template confidence "
                    f"{plan.confidence:.3f} is below "
                    f"{request.minimum_template_confidence:.3f}"
                )
            active_template_name = fallback_template_name
            fallback_reasons.append("fallback_template_provider_used")
            traces.append(
                VoiceStageTrace(
                    "fallback_retrieval",
                    "succeeded",
                    _duration_ms(started_at),
                    provider=fallback_template_name,
                    details={
                        "template_id": plan.template_id,
                        "confidence": plan.confidence,
                        "evidence_count": len(plan.evidence),
                        "slotted_template": True,
                    },
                )
            )
        except Exception as error:
            traces.append(
                VoiceStageTrace(
                    "fallback_retrieval",
                    "failed",
                    _duration_ms(started_at),
                    provider=fallback_template_name,
                    error=_safe_stage_error(error),
                )
            )

    response_text = request.fallback_text
    if plan is not None:
        started_at = time.perf_counter()
        try:
            response_text, grounded_slots = _render_grounded_plan(
                plan,
                grounding=request.grounding,
                minimum_confidence=request.minimum_template_confidence,
                telephone_safe=str(request.context.get("surface") or "")
                .strip()
                .casefold()
                in {"telephone", "telephony", "sip", "twilio"},
            )
            traces.append(
                VoiceStageTrace(
                    "rendering",
                    "succeeded",
                    _duration_ms(started_at),
                    provider=active_template_name,
                    details={"grounded_slot_count": len(grounded_slots)},
                )
            )
        except Exception as error:
            fallback_reasons.append("grounding_validation_failed")
            traces.append(
                VoiceStageTrace(
                    "rendering",
                    "failed",
                    _duration_ms(started_at),
                    provider=active_template_name,
                    error=_safe_stage_error(error),
                )
            )
    else:
        traces.append(
            VoiceStageTrace(
                "rendering",
                "skipped",
                0.0,
                provider=active_template_name,
                details={"reason": "grounded_template_unavailable"},
            )
        )

    output_audio: Optional[bytes] = None
    used_tts_provider: Optional[str] = None
    synthesis_failures = 0
    precomputed_resolution: Optional[PrecomputedAudioResolution] = None

    # Runtime resolution: exact precomputed audio before live TTS. Failure
    # falls through to live TTS or text-only and never serves a near/stale match.
    if audio_resolver is not None:
        started_at = time.perf_counter()
        try:
            synthesis_identity = _synthesis_identity_from_request(request)
            precomputed_resolution = audio_resolver.resolve(
                response_text,
                synthesis_identity,
                template_id=plan.template_id if plan is not None else None,
            )
            if precomputed_resolution.hit:
                output_audio = precomputed_resolution.audio
                used_tts_provider = "precomputed"
                traces.append(
                    VoiceStageTrace(
                        "synthesis",
                        "succeeded",
                        _duration_ms(started_at),
                        provider="precomputed",
                        details={
                            "audio_size_bytes": len(output_audio or b""),
                            "precomputed": True,
                            "runtime_resolution": True,
                            "resolver_reason": precomputed_resolution.reason,
                            "spoken_text_sha256": precomputed_spoken_text_sha256(
                                response_text
                            ),
                            "synthesis_identity": synthesis_identity.to_dict(),
                            **dict(precomputed_resolution.details),
                        },
                    )
                )
            else:
                # Deterministic resolver miss reason is retained on the stage
                # trace. Live TTS may still complete without degrading GraphRAG
                # provenance; if live TTS is also unavailable the final status
                # becomes text_only and the miss remains auditable.
                traces.append(
                    VoiceStageTrace(
                        "synthesis",
                        "skipped",
                        _duration_ms(started_at),
                        provider="precomputed",
                        details={
                            "precomputed": False,
                            "runtime_resolution": True,
                            "resolver_reason": precomputed_resolution.reason,
                            "spoken_text_sha256": precomputed_spoken_text_sha256(
                                response_text
                            ),
                            "synthesis_identity": synthesis_identity.to_dict(),
                            "live_tts_fallback": True,
                            **dict(precomputed_resolution.details),
                        },
                    )
                )
        except Exception as error:
            traces.append(
                VoiceStageTrace(
                    "synthesis",
                    "failed",
                    _duration_ms(started_at),
                    provider="precomputed",
                    error=_safe_stage_error(
                        error, sensitive_values=(response_text,)
                    ),
                    details={
                        "runtime_resolution": True,
                        "precomputed": False,
                        "live_tts_fallback": True,
                        "resolver_reason": "precomputed_audio_resolver_failed",
                    },
                )
            )

    if output_audio is None:
        tts_candidates = _provider_candidates(
            primary_tts,
            preferred=request.tts_provider,
            fallbacks=request.tts_providers,
            operation="synthesis",
            deps=resolved_deps,
        )
        for attempt, (
            provider_name,
            provider_object,
            resolution_error,
        ) in enumerate(tts_candidates, start=1):
            started_at = time.perf_counter()
            attempt_details = {
                "attempt": attempt,
                "retry": attempt > 1,
                "will_retry": attempt < len(tts_candidates),
            }
            if resolution_error is not None or provider_object is None:
                synthesis_failures += 1
                traces.append(
                    VoiceStageTrace(
                        "synthesis",
                        "failed",
                        _duration_ms(started_at),
                        provider=provider_name,
                        error=_safe_stage_error(
                            resolution_error
                            or RuntimeError("provider could not be resolved")
                        ),
                        details=attempt_details,
                    )
                )
                continue
            try:
                raw_audio = provider_object.synthesize(
                    response_text,
                    voice=request.voice,
                    model_name=request.tts_model,
                    device=request.device,
                    output_format=request.output_format,
                    **dict(request.tts_options),
                )
                if not isinstance(raw_audio, bytes) or not raw_audio:
                    _close_awaitable_result(raw_audio)
                    raise TypeError("synthesize returned no non-empty audio bytes")
                output_audio = raw_audio
                used_tts_provider = provider_name
                traces.append(
                    VoiceStageTrace(
                        "synthesis",
                        "succeeded",
                        _duration_ms(started_at),
                        provider=provider_name,
                        details={
                            **attempt_details,
                            "audio_size_bytes": len(raw_audio),
                            "precomputed": False,
                            **_provider_receipt_details(provider_object),
                        },
                    )
                )
                provider_receipt = getattr(provider_object, "last_receipt", None)
                if synthesis_failures or bool(
                    getattr(provider_receipt, "degraded", False)
                ):
                    fallback_reasons.append("tts_provider_fallback")
                break
            except Exception as error:
                synthesis_failures += 1
                traces.append(
                    VoiceStageTrace(
                        "synthesis",
                        "failed",
                        _duration_ms(started_at),
                        provider=provider_name,
                        error=_safe_stage_error(
                            error, sensitive_values=(response_text,)
                        ),
                        details={
                            **attempt_details,
                            **_provider_receipt_details(provider_object),
                        },
                    )
                )
    if output_audio is None:
        fallback_reasons.append("tts_failed")

    # Preserve first occurrence and pipeline order for deterministic receipts.
    fallback_tuple = tuple(dict.fromkeys(fallback_reasons))
    if not transcript:
        status = "failed"
    elif output_audio is None:
        status = "text_only"
    elif fallback_tuple:
        status = "degraded"
    else:
        status = "completed"

    provenance = VoiceTurnProvenance(
        stt_provider=used_stt_provider,
        template_provider=active_template_name,
        template_id=plan.template_id if plan is not None else None,
        tts_provider=used_tts_provider,
        evidence=plan.evidence if plan is not None else (),
        grounded_slots=grounded_slots,
        input_audio_sha256=request.input_audio_sha256,
        transcript_sha256=_sha256_text(transcript) if transcript else None,
        response_text_sha256=_sha256_text(response_text),
        output_audio_sha256=_sha256_bytes(output_audio)
        if output_audio is not None
        else None,
        metadata={
            "intent": plan.intent if plan is not None else None,
            "template_confidence": plan.confidence if plan is not None else None,
            "fallback_reasons": fallback_tuple,
            "precomputed_audio": (
                precomputed_resolution.to_dict()
                if precomputed_resolution is not None
                else None
            ),
        },
    )
    return VoiceTurnResult(
        request_id=request_id,
        status=status,
        transcript=transcript,
        response_text=response_text,
        audio=output_audio,
        audio_format=_audio_format(output_audio, request.output_format),
        provenance=provenance,
        traces=tuple(traces),
        fallback_reasons=fallback_tuple,
        cache_key=cache_key,
    )


def process_telephone_turn(
    request: VoiceTurnRequest,
    state: TelephoneTurnState,
    *,
    stt_provider: Optional[VoiceProvider] = None,
    template_provider: Optional[VoiceTemplateProvider] = None,
    fallback_template_provider: Optional[VoiceTemplateProvider] = None,
    tts_provider: Optional[VoiceProvider] = None,
    stt_provider_instance: Optional[VoiceProvider] = None,
    tts_provider_instance: Optional[VoiceProvider] = None,
    audio_resolver: Optional[PrecomputedVoiceAudioResolver] = None,
    deps: Optional[RouterDeps] = None,
) -> VoiceTurnResult:
    """Run one telephone turn through :func:`process_voice_turn`.

    This is a thin webhook/SIP boundary: it adds deterministic turn and
    barge-in context, never persists caller media, and emits explicit ingress,
    retry (on provider stage details), egress, and escalation evidence.
    Provider exhaustion degrades to a structured text-only human handoff.
    """

    if not isinstance(request, VoiceTurnRequest):
        raise TypeError("request must be a VoiceTurnRequest")
    if not isinstance(state, TelephoneTurnState):
        raise TypeError("state must be a TelephoneTurnState")
    existing_surface = str(request.context.get("surface") or "").strip().casefold()
    if existing_surface and existing_surface not in {
        "telephone",
        "telephony",
        "sip",
        "twilio",
    }:
        raise ValueError("telephone request context has a conflicting surface")
    _assert_telephone_spoken_safety(request.fallback_text)

    telephone_context = dict(request.context)
    telephone_context.update(state.to_context())
    telephone_request = replace(request, context=telephone_context)
    ingress_trace = VoiceStageTrace(
        "telephone_ingress",
        "succeeded",
        0.0,
        provider="shared_voice_router",
        details={
            **state.to_context(),
            "caller_audio_persisted": False,
            "caller_transcript_persisted": False,
        },
    )

    if state.turn_index >= state.max_turns:
        escalation_reason = "maximum_turns_reached"
        provenance = VoiceTurnProvenance(
            stt_provider="not_dispatched",
            template_provider=None,
            template_id=None,
            tts_provider=None,
            input_audio_sha256=request.input_audio_sha256,
            transcript_sha256=(
                _sha256_text(request.transcript) if request.transcript else None
            ),
            response_text_sha256=_sha256_text(request.fallback_text),
            metadata={
                "telephone": {
                    **state.to_context(),
                    "escalation_required": True,
                    "escalation_reason": escalation_reason,
                },
                "fallback_reasons": (
                    "telephone_max_turns_reached",
                    "telephone_human_escalation",
                ),
            },
        )
        return VoiceTurnResult(
            request_id=request.request_id
            or f"telephone-{state.call_id_sha256[:16]}-{state.turn_index}",
            status="text_only",
            transcript=request.transcript or "",
            response_text=request.fallback_text,
            audio=None,
            audio_format=None,
            provenance=provenance,
            traces=(
                ingress_trace,
                VoiceStageTrace(
                    "telephone_escalation",
                    "succeeded",
                    0.0,
                    provider="human_handoff",
                    details={
                        "reason": escalation_reason,
                        "turn_index": state.turn_index,
                        "max_turns": state.max_turns,
                    },
                ),
                VoiceStageTrace(
                    "telephone_egress",
                    "succeeded",
                    0.0,
                    provider="telephone_adapter",
                    details={
                        "delivery": "text_only_handoff",
                        "twiml_media_compatible": False,
                    },
                ),
            ),
            fallback_reasons=(
                "telephone_max_turns_reached",
                "telephone_human_escalation",
            ),
        )

    result = process_voice_turn(
        telephone_request,
        stt_provider=stt_provider,
        template_provider=template_provider,
        fallback_template_provider=fallback_template_provider,
        tts_provider=tts_provider,
        stt_provider_instance=stt_provider_instance,
        tts_provider_instance=tts_provider_instance,
        audio_resolver=audio_resolver,
        deps=deps,
    )

    intent = str(result.intent or "").strip().casefold()
    explicit_handoff = bool(
        request.context.get("human_escalation_requested")
        or request.context.get("request_human")
    )
    escalation_reason: Optional[str] = None
    if explicit_handoff or intent in {
        "live_agent",
        "human_handoff",
        "human_escalation",
    }:
        escalation_reason = "human_requested"
    elif result.status in {"text_only", "failed"}:
        escalation_reason = "provider_exhausted"

    escalation_required = escalation_reason is not None
    fallback_reasons = list(result.fallback_reasons)
    status = result.status
    if escalation_required:
        fallback_reasons.append("telephone_human_escalation")
        if status == "completed":
            status = "degraded"

    telephone_metadata = {
        **state.to_context(),
        "next_turn_index": state.turn_index + 1,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
    }
    provenance_metadata = dict(result.provenance.metadata)
    provenance_metadata["telephone"] = telephone_metadata
    provenance_metadata["fallback_reasons"] = tuple(
        dict.fromkeys(fallback_reasons)
    )
    provenance = replace(result.provenance, metadata=provenance_metadata)
    escalation_trace = VoiceStageTrace(
        "telephone_escalation",
        "succeeded" if escalation_required else "skipped",
        0.0,
        provider="human_handoff" if escalation_required else "telephone_adapter",
        details={
            "reason": escalation_reason or "not_required",
            "turn_index": state.turn_index,
            "barge_in": state.barge_in,
        },
    )
    egress_trace = VoiceStageTrace(
        "telephone_egress",
        "succeeded",
        0.0,
        provider="telephone_adapter",
        details={
            "delivery": (
                "audio_then_handoff"
                if result.audio is not None and escalation_required
                else "audio"
                if result.audio is not None
                else "text_only_handoff"
            ),
            "audio_format": result.audio_format,
            "twiml_media_compatible": result.audio is not None
            and result.audio_format in {"mp3", "wav", "ogg"},
        },
    )
    return replace(
        result,
        status=status,
        provenance=provenance,
        traces=(ingress_trace, *result.traces, escalation_trace, egress_trace),
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Usage-aware admission (optional; off mode is the default legacy path)
# ---------------------------------------------------------------------------


def estimate_synthesis_tokens(text: str) -> int:
    """Conservative synthesis token estimate for TTS admission."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return 1
    char_estimate = (len(text) + 3) // 4
    word_estimate = max(1, len(text.split()))
    byte_estimate = (len(text.encode("utf-8")) + 2) // 3
    return max(1, char_estimate, word_estimate, byte_estimate)


def _audio_media_bytes(audio: Union[str, bytes]) -> int:
    if isinstance(audio, bytes):
        return len(audio)
    if isinstance(audio, str) and os.path.isfile(audio):
        try:
            return int(os.path.getsize(audio))
        except OSError:
            return max(1, len(audio.encode("utf-8")))
    if isinstance(audio, str):
        return max(1, len(audio.encode("utf-8")))
    raise TypeError("audio must be bytes or a filesystem path string")


def _parse_wav_duration_seconds(payload: bytes) -> Optional[float]:
    """Best-effort WAV duration from a RIFF header (no full decode)."""

    if len(payload) < 44 or payload[0:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return None
    # Scan for fmt and data chunks.
    offset = 12
    sample_rate = 0
    channels = 0
    bits_per_sample = 0
    data_size = 0
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        chunk_size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
        data_start = offset + 8
        data_end = data_start + chunk_size
        if chunk_id == b"fmt " and chunk_size >= 16 and data_end <= len(payload):
            channels = int.from_bytes(payload[data_start + 2 : data_start + 4], "little")
            sample_rate = int.from_bytes(payload[data_start + 4 : data_start + 8], "little")
            bits_per_sample = int.from_bytes(
                payload[data_start + 14 : data_start + 16], "little"
            )
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        # Chunks are word-aligned.
        offset = data_end + (chunk_size % 2)
        if chunk_size <= 0:
            break
    if sample_rate > 0 and channels > 0 and bits_per_sample > 0 and data_size > 0:
        bytes_per_second = sample_rate * channels * max(1, bits_per_sample // 8)
        if bytes_per_second > 0:
            return float(data_size) / float(bytes_per_second)
    return None


def estimate_audio_seconds(
    audio: Union[str, bytes],
    *,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    declared_seconds: Optional[Union[int, float]] = None,
) -> int:
    """Conservative audio duration estimate (ceil seconds, minimum 1 when non-empty)."""

    if declared_seconds is not None:
        try:
            value = float(declared_seconds)
        except (TypeError, ValueError) as exc:
            raise TypeError("declared_seconds must be numeric") from exc
        if value <= 0:
            return 1
        return max(1, int(math.ceil(value)))

    media_bytes = _audio_media_bytes(audio)
    if media_bytes <= 0:
        return 1

    payload: Optional[bytes] = None
    if isinstance(audio, bytes):
        payload = audio
    elif isinstance(audio, str) and os.path.isfile(audio):
        try:
            with open(audio, "rb") as handle:
                payload = handle.read(min(media_bytes, 1024 * 1024))
        except OSError:
            payload = None

    if payload:
        wav_seconds = _parse_wav_duration_seconds(payload)
        if wav_seconds is not None and wav_seconds > 0:
            return max(1, int(math.ceil(wav_seconds)))

    rate = int(sample_rate) if sample_rate else 16_000
    ch = int(channels) if channels else 1
    if rate > 0 and ch > 0:
        # Assume 16-bit PCM when rate/channels are known; over-estimates
        # compressed audio slightly, which is fail-closed for admission.
        bytes_per_second = max(1, rate * ch * 2)
        return max(1, int(math.ceil(media_bytes / float(bytes_per_second))))

    # Compressed-audio fallback (~32 kbps) — over-estimate duration.
    return max(1, int(math.ceil(media_bytes / 4000.0)))


def estimate_synthesis_usage(
    text: str,
    *,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
    include_concurrency: bool = True,
    streaming: bool = False,
    remote: bool = True,
) -> "object":
    """Build a conservative multi-dimension usage vector for TTS work.

    Dimensions: requests, characters, input_tokens (synthesis tokens),
    media_bytes (UTF-8 text), concurrent_requests, concurrent_streams, cost.
    """

    from .endpoint_usage.schema import UsageVector

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not remote:
        return UsageVector()
    if not text:
        return UsageVector.of(requests=1)

    characters = len(text)
    tokens = estimate_synthesis_tokens(text)
    media_bytes = len(text.encode("utf-8"))
    amounts: Dict[str, int] = {
        "requests": 1,
        "characters": characters,
        "input_tokens": tokens,
        "media_bytes": media_bytes,
    }
    if include_concurrency:
        amounts["concurrent_requests"] = 1
        if streaming:
            amounts["concurrent_streams"] = 1
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    return UsageVector.of(**amounts)


def estimate_transcription_usage(
    audio: Union[str, bytes],
    *,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    declared_seconds: Optional[Union[int, float]] = None,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
    include_concurrency: bool = True,
    streaming: bool = False,
    remote: bool = True,
) -> "object":
    """Build a conservative multi-dimension usage vector for STT work.

    Dimensions: requests, audio_seconds, media_bytes, concurrent_requests,
    concurrent_streams, and cost as applicable.
    """

    from .endpoint_usage.schema import UsageVector

    if not remote:
        return UsageVector()
    media_bytes = _audio_media_bytes(audio)
    audio_seconds = estimate_audio_seconds(
        audio,
        sample_rate=sample_rate,
        channels=channels,
        declared_seconds=declared_seconds,
    )
    amounts: Dict[str, int] = {
        "requests": 1,
        "audio_seconds": audio_seconds,
        "media_bytes": max(0, media_bytes),
    }
    if include_concurrency:
        amounts["concurrent_requests"] = 1
        if streaming:
            amounts["concurrent_streams"] = 1
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    return UsageVector.of(**amounts)


def settle_synthesis_usage(
    text: str,
    *,
    audio_bytes: Optional[bytes] = None,
    characters: Optional[int] = None,
    tokens: Optional[int] = None,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
) -> "object":
    """Actual remote usage for a completed synthesis call."""

    from .endpoint_usage.schema import UsageVector

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    char_count = int(characters) if characters is not None else len(text)
    token_count = (
        int(tokens) if tokens is not None else estimate_synthesis_tokens(text)
    )
    amounts: Dict[str, int] = {
        "requests": 1,
        "characters": max(0, char_count),
        "input_tokens": max(0, token_count),
    }
    if audio_bytes is not None:
        amounts["media_bytes"] = len(audio_bytes)
    else:
        amounts["media_bytes"] = len(text.encode("utf-8"))
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    return UsageVector.of(**amounts)


def settle_transcription_usage(
    audio: Union[str, bytes],
    *,
    audio_seconds: Optional[int] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    declared_seconds: Optional[Union[int, float]] = None,
    cost_micros: Optional[int] = None,
    cost_currency: Optional[str] = None,
) -> "object":
    """Actual remote usage for a completed transcription call."""

    from .endpoint_usage.schema import UsageVector

    seconds = (
        int(audio_seconds)
        if audio_seconds is not None
        else estimate_audio_seconds(
            audio,
            sample_rate=sample_rate,
            channels=channels,
            declared_seconds=declared_seconds,
        )
    )
    amounts: Dict[str, int] = {
        "requests": 1,
        "audio_seconds": max(0, seconds),
        "media_bytes": _audio_media_bytes(audio),
    }
    if cost_micros is not None:
        amounts["cost_micros"] = int(cost_micros)
        return UsageVector.of(currency=cost_currency or "USD", **amounts)
    return UsageVector.of(**amounts)


def planning_required_usage(requested: "object") -> "object":
    """Return a receipt-safe planning vector derived from a full estimate.

    Token and media dimensions remain in the atomic reservation envelope but
    are omitted from ranking input names so route receipts stay redaction-safe.
    """

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


def apply_voice_stream_settlements(
    coordinator: object,
    reservation_id: str,
    partials: Sequence[object],
) -> List[object]:
    """Apply monotonic cumulative stream settlements for a held reservation.

    Each partial must be a :class:`UsageVector` or mapping accepted by
    ``UsageCoordinator.settle_stream``. Amounts must not decrease.
    """

    if not reservation_id:
        raise ValueError("reservation_id is required")
    settle = getattr(coordinator, "settle_stream", None)
    if not callable(settle):
        raise TypeError("coordinator must provide settle_stream")
    results: List[object] = []
    for partial in partials:
        results.append(settle(reservation_id, partial))
    return results


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


def _voice_compatibility_labels(
    *,
    provider_name: str,
    operation: str,
    model_name: Optional[str],
    device: Optional[str],
    voice: Optional[str] = None,
    language: Optional[str] = None,
    output_format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    kwargs: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    labels: Dict[str, str] = {
        "router_provider": str(provider_name or ""),
        "operation": str(operation or ""),
    }
    kwargs = dict(kwargs or {})
    try:
        descriptor = get_provider_descriptor(provider_name) if provider_name else None
    except Exception:
        descriptor = None
    if descriptor is not None:
        for key in (
            "locality",
            "device",
            "access_requirement",
            "languages",
            "voices",
            "data.governance",
            "data_retention",
        ):
            value = dict(descriptor.labels or {}).get(key)
            if value is not None:
                labels[key] = str(value)
        meta = _BUILTIN_VOICE_CATALOG.get(str(provider_name or "").strip().lower())
        if meta is not None:
            labels.setdefault("locality", meta.locality)
            labels.setdefault("device", meta.device)
            if meta.languages:
                labels.setdefault("languages", meta.languages)
            if meta.voices:
                labels.setdefault("voices", meta.voices)
            if meta.sample_rates_hz:
                labels.setdefault(
                    "sample_rates_hz",
                    ",".join(str(rate) for rate in meta.sample_rates_hz),
                )
    if model_name:
        labels["model_name"] = str(model_name)
    if device:
        labels["device"] = str(device)
    if voice:
        labels["voice"] = str(voice)
    if language:
        labels["language"] = str(language)
    if output_format:
        codec = str(output_format).strip().lower().lstrip(".")
        labels["codec"] = codec
        labels["output_format"] = codec
    if sample_rate is not None:
        labels["sample_rate"] = str(int(sample_rate))
    if channels is not None:
        labels["channels"] = str(int(channels))
    for key in (
        "locality",
        "device",
        "codec",
        "output_format",
        "sample_rate",
        "channels",
        "language",
        "voice",
        "data_retention",
        "data.governance",
        "access_requirement",
    ):
        if key in kwargs and kwargs[key] is not None:
            labels[key] = str(kwargs[key])
    return labels


def voice_fallback_compatible(
    origin_labels: Mapping[str, str],
    candidate_labels: Mapping[str, str],
) -> bool:
    """Return True when a fallback candidate preserves voice contracts.

    Fallback must preserve operation, language, voice, model compatibility,
    codec, sample rate/channels, locality/device, data-retention, authorization,
    and output contract when declared on the origin.
    """

    origin = {str(k): str(v) for k, v in origin_labels.items()}
    candidate = {str(k): str(v) for k, v in candidate_labels.items()}

    for key in (
        "operation",
        "language",
        "voice",
        "codec",
        "output_format",
        "sample_rate",
        "channels",
        "locality",
        "device",
        "model_name",
    ):
        if key in origin and origin[key] not in {"", "unknown", "provider-defined", "provider-managed"}:
            if candidate.get(key, origin[key]) != origin[key]:
                return False

    origin_access = origin.get("access_requirement")
    cand_access = candidate.get("access_requirement")
    if origin_access == "required" and cand_access not in (None, "required", "optional"):
        return False

    origin_gov = origin.get("data.governance") or origin.get("data_governance")
    cand_gov = candidate.get("data.governance") or candidate.get("data_governance")
    if origin_gov and cand_gov and cand_gov != origin_gov:
        return False
    if cand_gov and str(cand_gov).casefold() in {"deny", "forbidden", "blocked"}:
        return False

    origin_retention = origin.get("data_retention")
    cand_retention = candidate.get("data_retention")
    if origin_retention and cand_retention and cand_retention != origin_retention:
        return False

    # Sample-rate set compatibility when origin declares a concrete set.
    origin_rates = origin.get("sample_rates_hz")
    cand_rates = candidate.get("sample_rates_hz")
    if origin_rates and cand_rates:
        origin_set = {part.strip() for part in origin_rates.split(",") if part.strip()}
        cand_set = {part.strip() for part in cand_rates.split(",") if part.strip()}
        if origin_set and cand_set and origin_set.isdisjoint(cand_set):
            return False

    return True


def _build_voice_static_candidate(
    *,
    provider_name: str,
    operation: str,
    model_name: Optional[str],
    device: Optional[str],
    scope_id: str,
    voice: Optional[str] = None,
    language: Optional[str] = None,
    output_format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    kwargs: Optional[Mapping[str, object]] = None,
    score: int = 10,
    authorized: bool = True,
) -> "object":
    from .endpoint_usage.identity import stable_id
    from .endpoint_usage.resolution import StaticCandidate

    labels = _voice_compatibility_labels(
        provider_name=provider_name,
        operation=operation,
        model_name=model_name,
        device=device,
        voice=voice,
        language=language,
        output_format=output_format,
        sample_rate=sample_rate,
        channels=channels,
        kwargs=kwargs,
    )
    provider_id = stable_id("provider", "voice", provider_name)
    model_id = stable_id("model", "voice", provider_name, model_name or "default")
    deployment_id = stable_id(
        "deployment", "voice", provider_name, device or "default"
    )
    binding_id = stable_id(
        "binding", "voice", provider_name, operation, model_name or "default", scope_id
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


def _filter_compatible_voice_candidates(
    candidates: Sequence[object],
    *,
    origin_labels: Mapping[str, str],
) -> List[object]:
    kept: List[object] = []
    for cand in candidates:
        labels = dict(getattr(cand, "labels", None) or {})
        if voice_fallback_compatible(origin_labels, labels):
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
        "selected_binding_id": getattr(selected, "binding_id", None) if selected else None,
        "selected_scope_id": getattr(selected, "scope_id", None) if selected else None,
        "reservation_id": getattr(selected, "reservation_id", None) if selected else None,
        "receipt_id": getattr(receipt, "receipt_id", None) if receipt else None,
        "usage_revision": getattr(selected, "usage_revision", None) if selected else None,
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
    """Parse optional provider observation; never retain audio/text/credentials."""

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
        # Guard: observation must target the exact reserved scope.
        obs_scope = getattr(observation, "scope_id", None)
        if obs_scope and str(obs_scope) != str(scope_id):
            logger.debug(
                "voice usage observation scope mismatch; ignoring (exact-scope only)"
            )
            return None
        return observation
    if not isinstance(observation, Mapping):
        return None
    # Provider metadata updates apply only to the exact reserved scope.
    obs_scope = observation.get("scope_id")
    if obs_scope is not None and str(obs_scope) != str(scope_id):
        logger.debug(
            "voice usage observation scope mismatch; ignoring (exact-scope only)"
        )
        return None
    try:
        from .endpoint_usage.adapters import parse_provider_observation

        if any(
            key in observation
            for key in ("headers", "body", "family", "http_status", "usage")
        ):
            payload = dict(observation)
            payload["scope_id"] = scope_id
            payload.setdefault("request_id", request_id)
            # Never feed raw transcript/synthesis/audio into adapters.
            for forbidden in (
                "transcript",
                "text",
                "audio",
                "audio_bytes",
                "voice_sample",
                "synthesis_text",
            ):
                payload.pop(forbidden, None)
            return parse_provider_observation(payload)
    except Exception:
        logger.debug("voice usage observation adapter failed", exc_info=True)

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
        logger.debug("voice usage observation construct failed", exc_info=True)
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
    # Explicit provider selection is an exact pin by default (no fallback).
    if provider:
        return RoutePin(
            provider_id=stable_id("provider", "voice", provider),
            allow_fallback_with_pin=allow_fallback_with_pin,
        )
    return RoutePin()


def _bind_usage_routing_request(
    *,
    usage_request: object,
    requested: object,
) -> object:
    from .endpoint_usage.resolution import UsageRoutingRequest
    from .endpoint_usage.schema import UsageVector

    planning_required = planning_required_usage(requested)
    ureq = usage_request
    if ureq is None:
        return UsageRoutingRequest(
            required=planning_required,
            require_snapshot=True,
        )
    if isinstance(ureq, Mapping):
        ureq = UsageRoutingRequest.from_dict(ureq)
    elif not isinstance(ureq, UsageRoutingRequest):
        raise TypeError("usage_request must be UsageRoutingRequest, mapping, or None")
    source_required = ureq.required if ureq.required.entries else requested
    safe_required = planning_required_usage(source_required)
    if not safe_required.entries:
        safe_required = planning_required
    if not isinstance(safe_required, UsageVector):
        safe_required = planning_required
    return UsageRoutingRequest(
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


def _effective_policy_for_pin(policy: object, pin: object) -> object:
    from .endpoint_usage.schema import FallbackClass, RoutingPolicy

    if not isinstance(policy, RoutingPolicy):
        policy = _normalize_usage_policy(policy)
    if getattr(pin, "is_exact", False) and not getattr(
        pin, "allow_fallback_with_pin", False
    ):
        if policy.fallback is not FallbackClass.NONE:
            return RoutingPolicy(
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
    return policy


def _tts_cache_lookup(
    *,
    deps: RouterDeps,
    provider_identity: Optional[str],
    model_name: Optional[str],
    text: str,
    voice: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    kwargs: Mapping[str, object],
    output_path: Optional[str],
) -> Optional[Union[bytes, str]]:
    if not _response_cache_enabled():
        return None
    cache_key = _tts_response_cache_key(
        provider=provider_identity,
        model_name=model_name,
        text=text,
        voice=voice,
        device=device,
        output_format=output_format,
        kwargs=dict(kwargs),
    )
    try:
        getter = getattr(deps, "get_cached_or_remote", None)
        cached = getter(cache_key) if callable(getter) else deps.get_cached(cache_key)
        if isinstance(cached, bytes) and cached:
            if output_path:
                with open(output_path, "wb") as fh:
                    fh.write(cached)
                return output_path
            return cached
    except Exception:
        return None
    return None


def _tts_cache_store(
    *,
    deps: RouterDeps,
    provider_identity: Optional[str],
    model_name: Optional[str],
    text: str,
    voice: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    kwargs: Mapping[str, object],
    audio_bytes: bytes,
) -> None:
    if not _response_cache_enabled():
        return
    try:
        ck = _tts_response_cache_key(
            provider=provider_identity,
            model_name=model_name,
            text=text,
            voice=voice,
            device=device,
            output_format=output_format,
            kwargs=dict(kwargs),
        )
        setter = getattr(deps, "set_cached_and_remote", None)
        if callable(setter):
            setter(ck, audio_bytes)
        else:
            deps.set_cached(ck, audio_bytes)
    except Exception:
        pass


def _stt_cache_lookup(
    *,
    deps: RouterDeps,
    provider_identity: Optional[str],
    model_name: Optional[str],
    audio: Union[str, bytes],
    language: Optional[str],
    device: Optional[str],
    kwargs: Mapping[str, object],
) -> Optional[str]:
    if not _response_cache_enabled():
        return None
    cache_key = _stt_response_cache_key(
        provider=provider_identity,
        model_name=model_name,
        audio=audio,
        language=language,
        device=device,
        kwargs=dict(kwargs),
    )
    try:
        getter = getattr(deps, "get_cached_or_remote", None)
        cached = getter(cache_key) if callable(getter) else deps.get_cached(cache_key)
        if isinstance(cached, str) and cached:
            return cached
    except Exception:
        return None
    return None


def _stt_cache_store(
    *,
    deps: RouterDeps,
    provider_identity: Optional[str],
    model_name: Optional[str],
    audio: Union[str, bytes],
    language: Optional[str],
    device: Optional[str],
    kwargs: Mapping[str, object],
    transcription: str,
) -> None:
    if not _response_cache_enabled():
        return
    try:
        ck = _stt_response_cache_key(
            provider=provider_identity,
            model_name=model_name,
            audio=audio,
            language=language,
            device=device,
            kwargs=dict(kwargs),
        )
        setter = getattr(deps, "set_cached_and_remote", None)
        if callable(setter):
            setter(ck, transcription)
        else:
            deps.set_cached(ck, transcription)
    except Exception:
        pass


def _write_output_path(audio_bytes: bytes, output_path: Optional[str]) -> Union[bytes, str]:
    if output_path:
        with open(output_path, "wb") as fh:
            fh.write(audio_bytes)
        return output_path
    return audio_bytes


def _legacy_text_to_speech(
    text: str,
    *,
    voice: Optional[str],
    model_name: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    output_path: Optional[str],
    provider: Optional[str],
    provider_instance: Optional[VoiceProvider],
    deps: RouterDeps,
    kwargs: Dict[str, object],
) -> Union[bytes, str]:
    cached = _tts_cache_lookup(
        deps=deps,
        provider_identity=_provider_instance_cache_identity(provider_instance, provider),
        model_name=model_name,
        text=text,
        voice=voice,
        device=device,
        output_format=output_format,
        kwargs=kwargs,
        output_path=output_path,
    )
    if cached is not None:
        return cached

    backend = provider_instance or get_voice_provider(provider, deps=deps)
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
            _close_awaitable_result(audio_bytes)
            raise RuntimeError(
                f"Voice provider synthesize() returned {type(audio_bytes).__name__}, expected bytes"
            )
        _tts_cache_store(
            deps=deps,
            provider_identity=_provider_instance_cache_identity(
                provider_instance, provider
            ),
            model_name=model_name,
            text=text,
            voice=voice,
            device=device,
            output_format=output_format,
            kwargs=kwargs,
            audio_bytes=audio_bytes,
        )
        return _write_output_path(audio_bytes, output_path)
    except Exception as primary_error:
        logger.debug("Primary voice TTS provider failed: %s", primary_error)
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
                return _write_output_path(audio_bytes, output_path)
        raise


def _legacy_speech_to_text(
    audio: Union[str, bytes],
    *,
    model_name: Optional[str],
    language: Optional[str],
    device: Optional[str],
    provider: Optional[str],
    provider_instance: Optional[VoiceProvider],
    deps: RouterDeps,
    kwargs: Dict[str, object],
) -> str:
    cached = _stt_cache_lookup(
        deps=deps,
        provider_identity=_provider_instance_cache_identity(provider_instance, provider),
        model_name=model_name,
        audio=audio,
        language=language,
        device=device,
        kwargs=kwargs,
    )
    if cached is not None:
        return cached

    if provider is None and provider_instance is None:
        for name in ["openai", "assemblyai", "huggingface", "backend_manager"]:
            try:
                candidate = _builtin_provider_by_name(name, deps=deps)
                if candidate is None:
                    continue
                if isinstance(candidate, VoiceProvider):
                    backend: VoiceProvider = candidate
                    break
            except Exception:
                continue
        else:
            backend = get_voice_provider(provider, deps=deps)
    else:
        backend = provider_instance or get_voice_provider(provider, deps=deps)

    try:
        transcription = backend.transcribe(
            audio,
            model_name=model_name,
            language=language,
            device=device,
            **kwargs,
        )
        if not isinstance(transcription, str):
            _close_awaitable_result(transcription)
            raise RuntimeError(
                f"Voice provider transcribe() returned {type(transcription).__name__}, expected str"
            )
        _stt_cache_store(
            deps=deps,
            provider_identity=_provider_instance_cache_identity(
                provider_instance, provider
            ),
            model_name=model_name,
            audio=audio,
            language=language,
            device=device,
            kwargs=kwargs,
            transcription=transcription,
        )
        return transcription
    except Exception as primary_error:
        logger.debug("Primary voice STT provider failed: %s", primary_error)
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


def _record_voice_usage_observe_shadow(
    *,
    operation: str,
    estimate: object,
    usage_coordinator: object,
    usage_policy: object,
    usage_scope_id: Optional[str],
    usage_request_id: Optional[str],
    success: bool,
    provider_used: str,
    remote_charged: bool,
) -> None:
    """Observe/shadow diagnostics: estimate only; never change selection or charge."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output, stable_id
    from .endpoint_usage.schema import RoutingMode, UsageEventKind

    policy = usage_policy
    mode = getattr(policy, "mode", RoutingMode.OBSERVE)
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
        "operation": operation,
        "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
        "remote_charged": False,
        "mode": str(getattr(mode, "value", mode)),
    }
    if not remote_charged:
        payload["reason_codes"] = list(payload["reason_codes"]) + [
            "cache_hit" if success else "no_remote",
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
                    or stable_id("vreq", "shadow", usage_scope_id),
                    reason_codes=("shadow_observe",),
                )
            except Exception:
                logger.debug("shadow observation append failed", exc_info=True)
    assert_no_prompt_media_or_output(payload)
    _set_last_usage_admission(payload)


def _text_to_speech_with_usage_admission(
    text: str,
    *,
    voice: Optional[str],
    model_name: Optional[str],
    device: Optional[str],
    output_format: Optional[str],
    output_path: Optional[str],
    provider: Optional[str],
    provider_instance: Optional[VoiceProvider],
    deps: RouterDeps,
    kwargs: Dict[str, object],
    usage_coordinator: object,
    usage_policy: object,
    usage_candidates: Optional[Sequence[object]],
    usage_pin: object,
    usage_request: object,
    usage_request_id: Optional[str],
    usage_idempotency_key: Optional[str],
    usage_catalog_revision: Optional[str],
    usage_provider_by_binding: Optional[Mapping[str, VoiceProvider]],
    usage_observation: object,
    usage_cost_micros: Optional[int],
    usage_cost_currency: Optional[str],
    usage_cancel_event: Optional[threading.Event],
    usage_timeout_seconds: Optional[float],
    usage_stream_partials: Optional[Sequence[object]],
    usage_streaming: bool,
    sample_rate: Optional[int],
    channels: Optional[int],
    started: float,
) -> Union[bytes, str]:
    """Reserve, dispatch, settle one TTS unit under usage admission."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output, stable_id
    from .endpoint_usage.resolution import StaticCandidate
    from .endpoint_usage.routing import (
        ErrorSafetyClass,
        InvokeOutcome,
        UsageRouteAdmission,
        classify_invoke_error,
        meta_from_static,
    )
    from .endpoint_usage.schema import FallbackClass, RoutingPolicy, UsageVector

    policy = usage_policy
    if not isinstance(policy, RoutingPolicy):
        policy = _normalize_usage_policy(policy)

    provider_identity = _provider_instance_cache_identity(provider_instance, provider)
    cached = _tts_cache_lookup(
        deps=deps,
        provider_identity=provider_identity,
        model_name=model_name,
        text=text,
        voice=voice,
        device=device,
        output_format=output_format,
        kwargs=kwargs,
        output_path=output_path,
    )
    if cached is not None:
        _set_last_usage_admission(
            {
                "success": True,
                "final_status": "cache_hit",
                "reason_codes": ["cache_hit", "no_remote_charge"],
                "attempt_count": 0,
                "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                "remote_charged": False,
                "operation": VOICE_TTS_USAGE_OPERATION,
            }
        )
        assert_no_prompt_media_or_output(get_last_usage_admission())
        _set_last_voice_usage_trace(
            status="ok",
            operation=VOICE_TTS_USAGE_OPERATION,
            provider_requested=str(provider or ""),
            provider_used=str(provider or ""),
            model_name=str(model_name or ""),
            remote_charged=False,
            usage_mode=str(getattr(policy.mode, "value", policy.mode)),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return cached

    if usage_cancel_event is not None and usage_cancel_event.is_set():
        raise UsageCapacityError(
            "voice TTS usage admission cancelled before dispatch",
            reason_codes=("cancelled_before_dispatch",),
        )
    if usage_timeout_seconds is not None and usage_timeout_seconds <= 0:
        raise UsageCapacityError(
            "voice TTS usage admission timed out before dispatch",
            reason_codes=("timeout_before_dispatch",),
        )

    requested = estimate_synthesis_usage(
        text,
        cost_micros=usage_cost_micros,
        cost_currency=usage_cost_currency,
        streaming=usage_streaming,
        remote=True,
    )
    request_id = usage_request_id or stable_id(
        "vreq", "tts", str(time.time_ns()), str(len(text))
    )
    idempotency_key = usage_idempotency_key or stable_id(
        "videm", request_id, "tts"
    )
    catalog_revision = usage_catalog_revision or stable_id(
        "cat", "voice_router", USAGE_ROUTING_REQUIREMENT_ID
    )
    pin = _resolve_usage_pin(
        pin=usage_pin, provider=provider, allow_fallback_with_pin=False
    )

    if usage_candidates is not None:
        candidates: List[object] = list(usage_candidates)
    else:
        backend = provider_instance or get_voice_provider(provider, deps=deps)
        provider_used = _provider_display_name(backend, provider)
        scope_id = stable_id(
            "scope", "voice", "tts", provider_used, model_name or "default"
        )
        ureq_probe = usage_request
        if isinstance(ureq_probe, Mapping):
            preferred_scope = ureq_probe.get("preferred_scope_id")
        else:
            preferred_scope = getattr(ureq_probe, "preferred_scope_id", None)
        if preferred_scope:
            scope_id = str(preferred_scope)
        candidates = [
            _build_voice_static_candidate(
                provider_name=provider_used,
                operation=VOICE_TTS_USAGE_OPERATION,
                model_name=model_name,
                device=device,
                scope_id=scope_id,
                voice=voice,
                output_format=output_format,
                sample_rate=sample_rate,
                channels=channels,
                kwargs=kwargs,
            )
        ]
        usage_provider_by_binding = {
            candidates[0].binding_id: backend,  # type: ignore[attr-defined]
            **dict(usage_provider_by_binding or {}),
        }

    if not candidates:
        raise UsageCapacityError(
            "no voice TTS usage candidates",
            reason_codes=("no_candidates",),
        )

    origin_labels = dict(getattr(candidates[0], "labels", None) or {})
    origin_labels.setdefault("operation", VOICE_TTS_USAGE_OPERATION)
    if voice:
        origin_labels.setdefault("voice", str(voice))
    if output_format:
        origin_labels.setdefault(
            "codec", str(output_format).strip().lower().lstrip(".")
        )
    candidates = _filter_compatible_voice_candidates(
        candidates, origin_labels=origin_labels
    ) or list(candidates[:1])

    meta_by_binding = {
        cand.binding_id: meta_from_static(cand)  # type: ignore[attr-defined]
        for cand in candidates
        if isinstance(cand, StaticCandidate)
    }
    ureq = _bind_usage_routing_request(
        usage_request=usage_request, requested=requested
    )
    provider_map: Dict[str, VoiceProvider] = dict(usage_provider_by_binding or {})
    result_holder: Dict[str, object] = {}
    invoke_error_holder: Dict[str, BaseException] = {}
    dispatched_holder: Dict[str, bool] = {"dispatched": False}
    deadline = (
        time.perf_counter() + float(usage_timeout_seconds)
        if usage_timeout_seconds is not None
        else None
    )

    def invoke(attempt: object) -> InvokeOutcome:
        if usage_cancel_event is not None and usage_cancel_event.is_set():
            code = (
                "cancelled_after_dispatch"
                if dispatched_holder["dispatched"]
                else "cancelled_before_dispatch"
            )
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.CLIENT,
                reason_codes=(code,),
                side_effecting=bool(dispatched_holder["dispatched"]),
            )
        if deadline is not None and time.perf_counter() > deadline:
            code = (
                "timeout_after_dispatch"
                if dispatched_holder["dispatched"]
                else "timeout_before_dispatch"
            )
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.CLIENT,
                reason_codes=(code,),
                side_effecting=bool(dispatched_holder["dispatched"]),
            )

        binding_id = getattr(attempt, "binding_id", None)
        scope_id = getattr(attempt, "scope_id", None) or ""
        reservation_id = getattr(attempt, "reservation_id", None) or ""
        active_backend: Optional[VoiceProvider] = None
        if binding_id and binding_id in provider_map:
            active_backend = provider_map[binding_id]
        else:
            labels: Dict[str, str] = {}
            for cand in candidates:
                if getattr(cand, "binding_id", None) == binding_id:
                    labels = dict(getattr(cand, "labels", None) or {})
                    break
            if labels and not voice_fallback_compatible(origin_labels, labels):
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.SEMANTIC,
                    reason_codes=("incompatible_voice_candidate",),
                    side_effecting=False,
                )
            router_name = labels.get("router_provider") or provider
            try:
                active_backend = provider_instance or get_voice_provider(
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

        # Stream partials (if any) settle monotonically before final commit.
        if usage_stream_partials and reservation_id:
            try:
                apply_voice_stream_settlements(
                    usage_coordinator,
                    str(reservation_id),
                    usage_stream_partials,
                )
                result_holder["stream_partials"] = len(usage_stream_partials)
            except Exception as exc:
                invoke_error_holder["error"] = exc
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.CLIENT,
                    reason_codes=("non_monotonic_stream", type(exc).__name__),
                    side_effecting=False,
                )

        dispatched_holder["dispatched"] = True
        try:
            audio_bytes = active_backend.synthesize(
                text,
                voice=voice,
                model_name=model_name,
                device=device,
                output_format=output_format,
                **kwargs,
            )
        except Exception as exc:
            invoke_error_holder["error"] = exc
            error_class = classify_invoke_error(reason_codes=(type(exc).__name__,))
            message = str(exc).casefold()
            if any(
                token in message
                for token in ("rate limit", "429", "quota", "capacity", "503")
            ):
                error_class = ErrorSafetyClass.CAPACITY
            # Capacity/transient failures must remain fallback-safe; do not
            # mark side_effecting or the admission protocol will refuse reroute.
            return InvokeOutcome(
                success=False,
                error_class=error_class,
                reason_codes=("provider_error", type(exc).__name__),
                side_effecting=False,
            )

        if not isinstance(audio_bytes, bytes):
            _close_awaitable_result(audio_bytes)
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.SEMANTIC,
                reason_codes=("output_validation_failed", "non_bytes"),
                side_effecting=False,
            )

        if deadline is not None and time.perf_counter() > deadline:
            # Post-dispatch timeout: remote work may have completed; settle actual.
            settled_timeout = settle_synthesis_usage(
                text,
                audio_bytes=audio_bytes,
                cost_micros=usage_cost_micros,
                cost_currency=usage_cost_currency,
            )
            result_holder["audio_bytes"] = audio_bytes
            result_holder["provider_used"] = _provider_display_name(
                active_backend, provider
            )
            result_holder["settled"] = settled_timeout
            result_holder["timeout_after_dispatch"] = True
            obs = _parse_provider_observation(
                scope_id=str(scope_id),
                request_id=request_id,
                observation=usage_observation,
                settled=settled_timeout,
            )
            return InvokeOutcome(
                success=True,
                observation=obs,
                settled=settled_timeout,
                error_class=ErrorSafetyClass.SUCCESS,
                reason_codes=("synthesized", "timeout_after_dispatch"),
            )

        settled = settle_synthesis_usage(
            text,
            audio_bytes=audio_bytes,
            cost_micros=usage_cost_micros,
            cost_currency=usage_cost_currency,
        )
        obs = _parse_provider_observation(
            scope_id=str(scope_id),
            request_id=request_id,
            observation=usage_observation,
            settled=settled,
        )
        result_holder["audio_bytes"] = audio_bytes
        result_holder["provider_used"] = _provider_display_name(
            active_backend, provider
        )
        result_holder["settled"] = settled
        return InvokeOutcome(
            success=True,
            observation=obs,
            settled=settled,
            error_class=ErrorSafetyClass.SUCCESS,
            reason_codes=("synthesized",),
        )

    admission = UsageRouteAdmission(
        usage_coordinator,  # type: ignore[arg-type]
        owner_id="voice_router",
        jitter_max_ms=0,
    )
    effective_policy = _effective_policy_for_pin(policy, pin)
    result = admission.admit(
        catalog_revision=catalog_revision,
        candidates=candidates,  # type: ignore[arg-type]
        request_id=request_id,
        idempotency_key=idempotency_key,
        operation=VOICE_TTS_USAGE_OPERATION,
        requested=requested if isinstance(requested, UsageVector) else UsageVector(),
        policy=effective_policy,  # type: ignore[arg-type]
        request=ureq,  # type: ignore[arg-type]
        pin=pin,  # type: ignore[arg-type]
        meta_by_binding=meta_by_binding,
        invoke=invoke,
        caller_id="voice_router",
    )
    admission_trace = _admission_result_to_trace(result)
    if result_holder.get("stream_partials"):
        admission_trace["stream_partials"] = result_holder["stream_partials"]
    if result_holder.get("timeout_after_dispatch"):
        admission_trace["reason_codes"] = list(admission_trace.get("reason_codes") or []) + [
            "timeout_after_dispatch"
        ]
    _set_last_usage_admission(admission_trace)

    if not result.success or "audio_bytes" not in result_holder:
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
            "voice TTS usage admission failed: %s"
            % (",".join(result.reason_codes) or result.final_status),
            reason_codes=result.reason_codes,
            next_eligible_at=result.next_eligible_at,
            admission=result,
        )

    audio_bytes = result_holder["audio_bytes"]  # type: ignore[assignment]
    assert isinstance(audio_bytes, bytes)
    provider_used_name = str(result_holder.get("provider_used") or provider or "")
    _tts_cache_store(
        deps=deps,
        provider_identity=_provider_instance_cache_identity(
            provider_instance, provider_used_name or provider
        ),
        model_name=model_name,
        text=text,
        voice=voice,
        device=device,
        output_format=output_format,
        kwargs=kwargs,
        audio_bytes=audio_bytes,
    )
    _set_last_voice_usage_trace(
        status="ok",
        operation=VOICE_TTS_USAGE_OPERATION,
        provider_requested=str(provider or ""),
        provider_used=provider_used_name,
        model_name=str(model_name or ""),
        remote_charged=True,
        usage_mode=str(getattr(policy.mode, "value", policy.mode)),
        reservation_id=getattr(result.selected, "reservation_id", None)
        if result.selected
        else None,
        receipt_id=getattr(result.receipt, "receipt_id", None)
        if result.receipt
        else None,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return _write_output_path(audio_bytes, output_path)


def _speech_to_text_with_usage_admission(
    audio: Union[str, bytes],
    *,
    model_name: Optional[str],
    language: Optional[str],
    device: Optional[str],
    provider: Optional[str],
    provider_instance: Optional[VoiceProvider],
    deps: RouterDeps,
    kwargs: Dict[str, object],
    usage_coordinator: object,
    usage_policy: object,
    usage_candidates: Optional[Sequence[object]],
    usage_pin: object,
    usage_request: object,
    usage_request_id: Optional[str],
    usage_idempotency_key: Optional[str],
    usage_catalog_revision: Optional[str],
    usage_provider_by_binding: Optional[Mapping[str, VoiceProvider]],
    usage_observation: object,
    usage_cost_micros: Optional[int],
    usage_cost_currency: Optional[str],
    usage_cancel_event: Optional[threading.Event],
    usage_timeout_seconds: Optional[float],
    usage_stream_partials: Optional[Sequence[object]],
    usage_streaming: bool,
    sample_rate: Optional[int],
    channels: Optional[int],
    declared_seconds: Optional[Union[int, float]],
    started: float,
) -> str:
    """Reserve, dispatch, settle one STT unit under usage admission."""

    from .endpoint_usage.identity import assert_no_prompt_media_or_output, stable_id
    from .endpoint_usage.resolution import StaticCandidate
    from .endpoint_usage.routing import (
        ErrorSafetyClass,
        InvokeOutcome,
        UsageRouteAdmission,
        classify_invoke_error,
        meta_from_static,
    )
    from .endpoint_usage.schema import FallbackClass, RoutingPolicy, UsageVector

    policy = usage_policy
    if not isinstance(policy, RoutingPolicy):
        policy = _normalize_usage_policy(policy)

    provider_identity = _provider_instance_cache_identity(provider_instance, provider)
    cached = _stt_cache_lookup(
        deps=deps,
        provider_identity=provider_identity,
        model_name=model_name,
        audio=audio,
        language=language,
        device=device,
        kwargs=kwargs,
    )
    if cached is not None:
        _set_last_usage_admission(
            {
                "success": True,
                "final_status": "cache_hit",
                "reason_codes": ["cache_hit", "no_remote_charge"],
                "attempt_count": 0,
                "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                "remote_charged": False,
                "operation": VOICE_STT_USAGE_OPERATION,
            }
        )
        assert_no_prompt_media_or_output(get_last_usage_admission())
        _set_last_voice_usage_trace(
            status="ok",
            operation=VOICE_STT_USAGE_OPERATION,
            provider_requested=str(provider or ""),
            provider_used=str(provider or ""),
            model_name=str(model_name or ""),
            remote_charged=False,
            usage_mode=str(getattr(policy.mode, "value", policy.mode)),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return cached

    if usage_cancel_event is not None and usage_cancel_event.is_set():
        raise UsageCapacityError(
            "voice STT usage admission cancelled before dispatch",
            reason_codes=("cancelled_before_dispatch",),
        )
    if usage_timeout_seconds is not None and usage_timeout_seconds <= 0:
        raise UsageCapacityError(
            "voice STT usage admission timed out before dispatch",
            reason_codes=("timeout_before_dispatch",),
        )

    requested = estimate_transcription_usage(
        audio,
        sample_rate=sample_rate,
        channels=channels,
        declared_seconds=declared_seconds,
        cost_micros=usage_cost_micros,
        cost_currency=usage_cost_currency,
        streaming=usage_streaming,
        remote=True,
    )
    request_id = usage_request_id or stable_id(
        "vreq", "stt", str(time.time_ns()), str(_audio_media_bytes(audio))
    )
    idempotency_key = usage_idempotency_key or stable_id(
        "videm", request_id, "stt"
    )
    catalog_revision = usage_catalog_revision or stable_id(
        "cat", "voice_router", USAGE_ROUTING_REQUIREMENT_ID
    )
    pin = _resolve_usage_pin(
        pin=usage_pin, provider=provider, allow_fallback_with_pin=False
    )

    if usage_candidates is not None:
        candidates = list(usage_candidates)
    else:
        if provider is None and provider_instance is None:
            backend = None
            for name in ["openai", "assemblyai", "huggingface", "backend_manager"]:
                try:
                    candidate = _builtin_provider_by_name(name, deps=deps)
                    if candidate is not None and isinstance(candidate, VoiceProvider):
                        backend = candidate
                        break
                except Exception:
                    continue
            if backend is None:
                backend = get_voice_provider(provider, deps=deps)
        else:
            backend = provider_instance or get_voice_provider(provider, deps=deps)
        provider_used = _provider_display_name(backend, provider)
        scope_id = stable_id(
            "scope", "voice", "stt", provider_used, model_name or "default"
        )
        ureq_probe = usage_request
        if isinstance(ureq_probe, Mapping):
            preferred_scope = ureq_probe.get("preferred_scope_id")
        else:
            preferred_scope = getattr(ureq_probe, "preferred_scope_id", None)
        if preferred_scope:
            scope_id = str(preferred_scope)
        candidates = [
            _build_voice_static_candidate(
                provider_name=provider_used,
                operation=VOICE_STT_USAGE_OPERATION,
                model_name=model_name,
                device=device,
                scope_id=scope_id,
                language=language,
                sample_rate=sample_rate,
                channels=channels,
                kwargs=kwargs,
            )
        ]
        usage_provider_by_binding = {
            candidates[0].binding_id: backend,  # type: ignore[attr-defined]
            **dict(usage_provider_by_binding or {}),
        }

    if not candidates:
        raise UsageCapacityError(
            "no voice STT usage candidates",
            reason_codes=("no_candidates",),
        )

    origin_labels = dict(getattr(candidates[0], "labels", None) or {})
    origin_labels.setdefault("operation", VOICE_STT_USAGE_OPERATION)
    if language:
        origin_labels.setdefault("language", str(language))
    candidates = _filter_compatible_voice_candidates(
        candidates, origin_labels=origin_labels
    ) or list(candidates[:1])

    meta_by_binding = {
        cand.binding_id: meta_from_static(cand)  # type: ignore[attr-defined]
        for cand in candidates
        if isinstance(cand, StaticCandidate)
    }
    ureq = _bind_usage_routing_request(
        usage_request=usage_request, requested=requested
    )
    provider_map: Dict[str, VoiceProvider] = dict(usage_provider_by_binding or {})
    result_holder: Dict[str, object] = {}
    invoke_error_holder: Dict[str, BaseException] = {}
    dispatched_holder: Dict[str, bool] = {"dispatched": False}
    deadline = (
        time.perf_counter() + float(usage_timeout_seconds)
        if usage_timeout_seconds is not None
        else None
    )

    def invoke(attempt: object) -> InvokeOutcome:
        if usage_cancel_event is not None and usage_cancel_event.is_set():
            code = (
                "cancelled_after_dispatch"
                if dispatched_holder["dispatched"]
                else "cancelled_before_dispatch"
            )
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.CLIENT,
                reason_codes=(code,),
                side_effecting=bool(dispatched_holder["dispatched"]),
            )
        if deadline is not None and time.perf_counter() > deadline:
            code = (
                "timeout_after_dispatch"
                if dispatched_holder["dispatched"]
                else "timeout_before_dispatch"
            )
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.CLIENT,
                reason_codes=(code,),
                side_effecting=bool(dispatched_holder["dispatched"]),
            )

        binding_id = getattr(attempt, "binding_id", None)
        scope_id = getattr(attempt, "scope_id", None) or ""
        reservation_id = getattr(attempt, "reservation_id", None) or ""
        active_backend: Optional[VoiceProvider] = None
        if binding_id and binding_id in provider_map:
            active_backend = provider_map[binding_id]
        else:
            labels = {}
            for cand in candidates:
                if getattr(cand, "binding_id", None) == binding_id:
                    labels = dict(getattr(cand, "labels", None) or {})
                    break
            if labels and not voice_fallback_compatible(origin_labels, labels):
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.SEMANTIC,
                    reason_codes=("incompatible_voice_candidate",),
                    side_effecting=False,
                )
            router_name = labels.get("router_provider") or provider
            try:
                active_backend = provider_instance or get_voice_provider(
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

        if usage_stream_partials and reservation_id:
            try:
                apply_voice_stream_settlements(
                    usage_coordinator,
                    str(reservation_id),
                    usage_stream_partials,
                )
                result_holder["stream_partials"] = len(usage_stream_partials)
            except Exception as exc:
                invoke_error_holder["error"] = exc
                return InvokeOutcome(
                    success=False,
                    error_class=ErrorSafetyClass.CLIENT,
                    reason_codes=("non_monotonic_stream", type(exc).__name__),
                    side_effecting=False,
                )

        dispatched_holder["dispatched"] = True
        try:
            transcription = active_backend.transcribe(
                audio,
                model_name=model_name,
                language=language,
                device=device,
                **kwargs,
            )
        except Exception as exc:
            invoke_error_holder["error"] = exc
            error_class = classify_invoke_error(reason_codes=(type(exc).__name__,))
            message = str(exc).casefold()
            if any(
                token in message
                for token in ("rate limit", "429", "quota", "capacity", "503")
            ):
                error_class = ErrorSafetyClass.CAPACITY
            # Capacity/transient failures must remain fallback-safe; do not
            # mark side_effecting or the admission protocol will refuse reroute.
            return InvokeOutcome(
                success=False,
                error_class=error_class,
                reason_codes=("provider_error", type(exc).__name__),
                side_effecting=False,
            )

        if not isinstance(transcription, str):
            _close_awaitable_result(transcription)
            return InvokeOutcome(
                success=False,
                error_class=ErrorSafetyClass.SEMANTIC,
                reason_codes=("output_validation_failed", "non_str"),
                side_effecting=False,
            )

        settled = settle_transcription_usage(
            audio,
            sample_rate=sample_rate,
            channels=channels,
            declared_seconds=declared_seconds,
            cost_micros=usage_cost_micros,
            cost_currency=usage_cost_currency,
        )
        obs = _parse_provider_observation(
            scope_id=str(scope_id),
            request_id=request_id,
            observation=usage_observation,
            settled=settled,
        )
        # Never place transcript content into admission holders that may
        # leak into receipts — only length metadata.
        result_holder["transcription"] = transcription
        result_holder["transcript_chars"] = len(transcription)
        result_holder["provider_used"] = _provider_display_name(
            active_backend, provider
        )
        result_holder["settled"] = settled
        reason_codes = ["transcribed"]
        if deadline is not None and time.perf_counter() > deadline:
            reason_codes.append("timeout_after_dispatch")
            result_holder["timeout_after_dispatch"] = True
        return InvokeOutcome(
            success=True,
            observation=obs,
            settled=settled,
            error_class=ErrorSafetyClass.SUCCESS,
            reason_codes=tuple(reason_codes),
        )

    admission = UsageRouteAdmission(
        usage_coordinator,  # type: ignore[arg-type]
        owner_id="voice_router",
        jitter_max_ms=0,
    )
    effective_policy = _effective_policy_for_pin(policy, pin)
    result = admission.admit(
        catalog_revision=catalog_revision,
        candidates=candidates,  # type: ignore[arg-type]
        request_id=request_id,
        idempotency_key=idempotency_key,
        operation=VOICE_STT_USAGE_OPERATION,
        requested=requested if isinstance(requested, UsageVector) else UsageVector(),
        policy=effective_policy,  # type: ignore[arg-type]
        request=ureq,  # type: ignore[arg-type]
        pin=pin,  # type: ignore[arg-type]
        meta_by_binding=meta_by_binding,
        invoke=invoke,
        caller_id="voice_router",
    )
    admission_trace = _admission_result_to_trace(result)
    if result_holder.get("stream_partials"):
        admission_trace["stream_partials"] = result_holder["stream_partials"]
    if result_holder.get("timeout_after_dispatch"):
        admission_trace["reason_codes"] = list(
            admission_trace.get("reason_codes") or []
        ) + ["timeout_after_dispatch"]
    # Explicitly ensure no transcript leaks into admission payload.
    admission_trace.pop("transcription", None)
    admission_trace.pop("transcript", None)
    _set_last_usage_admission(admission_trace)

    if not result.success or "transcription" not in result_holder:
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
            "voice STT usage admission failed: %s"
            % (",".join(result.reason_codes) or result.final_status),
            reason_codes=result.reason_codes,
            next_eligible_at=result.next_eligible_at,
            admission=result,
        )

    transcription = result_holder["transcription"]  # type: ignore[assignment]
    assert isinstance(transcription, str)
    provider_used_name = str(result_holder.get("provider_used") or provider or "")
    _stt_cache_store(
        deps=deps,
        provider_identity=_provider_instance_cache_identity(
            provider_instance, provider_used_name or provider
        ),
        model_name=model_name,
        audio=audio,
        language=language,
        device=device,
        kwargs=kwargs,
        transcription=transcription,
    )
    _set_last_voice_usage_trace(
        status="ok",
        operation=VOICE_STT_USAGE_OPERATION,
        provider_requested=str(provider or ""),
        provider_used=provider_used_name,
        model_name=str(model_name or ""),
        remote_charged=True,
        usage_mode=str(getattr(policy.mode, "value", policy.mode)),
        reservation_id=getattr(result.selected, "reservation_id", None)
        if result.selected
        else None,
        receipt_id=getattr(result.receipt, "receipt_id", None)
        if result.receipt
        else None,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return transcription


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
    usage_coordinator: Optional[object] = None,
    usage_policy: Optional[object] = None,
    usage_candidates: Optional[Sequence[object]] = None,
    usage_pin: Optional[object] = None,
    usage_request: Optional[object] = None,
    usage_request_id: Optional[str] = None,
    usage_idempotency_key: Optional[str] = None,
    usage_catalog_revision: Optional[str] = None,
    usage_provider_by_binding: Optional[Mapping[str, VoiceProvider]] = None,
    usage_observation: Optional[object] = None,
    usage_cost_micros: Optional[int] = None,
    usage_cost_currency: Optional[str] = None,
    usage_cancel_event: Optional[threading.Event] = None,
    usage_timeout_seconds: Optional[float] = None,
    usage_stream_partials: Optional[Sequence[object]] = None,
    usage_streaming: bool = False,
    usage_scope_id: Optional[str] = None,
    usage_sample_rate: Optional[int] = None,
    usage_channels: Optional[int] = None,
    **kwargs: object,
) -> Union[bytes, str]:
    """Synthesize speech from text.

    Optional usage-aware admission (AICAT-033) is inactive unless a
    ``usage_coordinator`` is supplied with a non-``off`` ``usage_policy``.
    Off mode and a missing coordinator preserve legacy selection exactly.
    Enforce/assist reserve before remote dispatch; observe/shadow never change
    the selected provider; cache hits create no remote charge.

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
    started = time.perf_counter()
    resolved_deps = deps or get_default_router_deps()
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    policy = _normalize_usage_policy(usage_policy)
    if usage_scope_id is None and usage_request is not None:
        if isinstance(usage_request, Mapping):
            usage_scope_id = usage_request.get("preferred_scope_id")  # type: ignore[assignment]
        else:
            usage_scope_id = getattr(usage_request, "preferred_scope_id", None)

    if usage_coordinator is not None and _usage_mode_enforces(policy):
        return _text_to_speech_with_usage_admission(
            text,
            voice=voice,
            model_name=model_name,
            device=device,
            output_format=output_format,
            output_path=output_path,
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
            usage_timeout_seconds=usage_timeout_seconds,
            usage_stream_partials=usage_stream_partials,
            usage_streaming=bool(usage_streaming),
            sample_rate=usage_sample_rate,
            channels=usage_channels,
            started=started,
        )

    result = _legacy_text_to_speech(
        text,
        voice=voice,
        model_name=model_name,
        device=device,
        output_format=output_format,
        output_path=output_path,
        provider=provider,
        provider_instance=provider_instance,
        deps=resolved_deps,
        kwargs=dict(kwargs),
    )
    provider_used = _provider_display_name(provider_instance, provider) if provider_instance else str(provider or "")
    if usage_coordinator is not None and _usage_mode_observes_only(policy):
        remote = not (
            isinstance(result, bytes)
            and False  # cache path already returned above only in enforce
        )
        # Observe after legacy path: if cache was hit, no remote charge.
        # We cannot always know cache hit without re-checking; treat as remote
        # unless result came from a zero-call path. Conservative: estimate remote.
        estimate = estimate_synthesis_usage(
            text,
            cost_micros=usage_cost_micros,
            cost_currency=usage_cost_currency,
            streaming=usage_streaming,
            remote=True,
        )
        _record_voice_usage_observe_shadow(
            operation=VOICE_TTS_USAGE_OPERATION,
            estimate=estimate,
            usage_coordinator=usage_coordinator,
            usage_policy=policy,
            usage_scope_id=usage_scope_id,
            usage_request_id=usage_request_id,
            success=True,
            provider_used=provider_used,
            remote_charged=True,
        )
    elif usage_coordinator is None or _usage_mode_is_off(policy, usage_coordinator):
        _set_last_usage_admission(
            {
                "success": True,
                "final_status": "off",
                "reason_codes": ["usage_routing_off"],
                "attempt_count": 0,
                "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                "remote_charged": None,
                "mode": "off",
                "operation": VOICE_TTS_USAGE_OPERATION,
            }
        )
    return result


def speech_to_text(
    audio: Union[str, bytes],
    *,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    provider_instance: Optional[VoiceProvider] = None,
    deps: Optional[RouterDeps] = None,
    usage_coordinator: Optional[object] = None,
    usage_policy: Optional[object] = None,
    usage_candidates: Optional[Sequence[object]] = None,
    usage_pin: Optional[object] = None,
    usage_request: Optional[object] = None,
    usage_request_id: Optional[str] = None,
    usage_idempotency_key: Optional[str] = None,
    usage_catalog_revision: Optional[str] = None,
    usage_provider_by_binding: Optional[Mapping[str, VoiceProvider]] = None,
    usage_observation: Optional[object] = None,
    usage_cost_micros: Optional[int] = None,
    usage_cost_currency: Optional[str] = None,
    usage_cancel_event: Optional[threading.Event] = None,
    usage_timeout_seconds: Optional[float] = None,
    usage_stream_partials: Optional[Sequence[object]] = None,
    usage_streaming: bool = False,
    usage_scope_id: Optional[str] = None,
    usage_sample_rate: Optional[int] = None,
    usage_channels: Optional[int] = None,
    usage_audio_seconds: Optional[Union[int, float]] = None,
    **kwargs: object,
) -> str:
    """Transcribe speech audio to text.

    Optional usage-aware admission (AICAT-033) is inactive unless a
    ``usage_coordinator`` is supplied with a non-``off`` ``usage_policy``.

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
    started = time.perf_counter()
    resolved_deps = deps or get_default_router_deps()

    policy = _normalize_usage_policy(usage_policy)
    if usage_scope_id is None and usage_request is not None:
        if isinstance(usage_request, Mapping):
            usage_scope_id = usage_request.get("preferred_scope_id")  # type: ignore[assignment]
        else:
            usage_scope_id = getattr(usage_request, "preferred_scope_id", None)

    if usage_coordinator is not None and _usage_mode_enforces(policy):
        return _speech_to_text_with_usage_admission(
            audio,
            model_name=model_name,
            language=language,
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
            usage_timeout_seconds=usage_timeout_seconds,
            usage_stream_partials=usage_stream_partials,
            usage_streaming=bool(usage_streaming),
            sample_rate=usage_sample_rate,
            channels=usage_channels,
            declared_seconds=usage_audio_seconds,
            started=started,
        )

    result = _legacy_speech_to_text(
        audio,
        model_name=model_name,
        language=language,
        device=device,
        provider=provider,
        provider_instance=provider_instance,
        deps=resolved_deps,
        kwargs=dict(kwargs),
    )
    provider_used = (
        _provider_display_name(provider_instance, provider)
        if provider_instance
        else str(provider or "")
    )
    if usage_coordinator is not None and _usage_mode_observes_only(policy):
        estimate = estimate_transcription_usage(
            audio,
            sample_rate=usage_sample_rate,
            channels=usage_channels,
            declared_seconds=usage_audio_seconds,
            cost_micros=usage_cost_micros,
            cost_currency=usage_cost_currency,
            streaming=usage_streaming,
            remote=True,
        )
        _record_voice_usage_observe_shadow(
            operation=VOICE_STT_USAGE_OPERATION,
            estimate=estimate,
            usage_coordinator=usage_coordinator,
            usage_policy=policy,
            usage_scope_id=usage_scope_id,
            usage_request_id=usage_request_id,
            success=True,
            provider_used=provider_used,
            remote_charged=True,
        )
    elif usage_coordinator is None or _usage_mode_is_off(policy, usage_coordinator):
        _set_last_usage_admission(
            {
                "success": True,
                "final_status": "off",
                "reason_codes": ["usage_routing_off"],
                "attempt_count": 0,
                "requirement_id": USAGE_ROUTING_REQUIREMENT_ID,
                "remote_charged": None,
                "mode": "off",
                "operation": VOICE_STT_USAGE_OPERATION,
            }
        )
    return result


def clear_voice_router_caches() -> None:
    """Clear internal provider caches (useful for tests)."""
    _resolve_provider_cached.cache_clear()


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (formerly in tts_router)
# ---------------------------------------------------------------------------
#: Alias for :class:`VoiceProvider` – kept for code that imported TTSProvider
#: from the old ``tts_router`` module.
TTSProvider = VoiceProvider

#: Alias for :func:`get_voice_provider`.
get_tts_provider = get_voice_provider

#: Alias for :func:`register_voice_provider`.
register_tts_provider = register_voice_provider

#: Alias for :func:`clear_voice_router_caches`.
clear_tts_router_caches = clear_voice_router_caches

# Package-owned cache-miss event contracts are dependency-light and imported
# late to avoid a circular dependency on the runtime result classes above.
from .voice_cache_miss import (  # noqa: E402
    VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION,
    VoiceCacheMissEvent,
    VoiceCacheMissEventError,
    build_voice_cache_miss_event,
)

__all__ = [
    # Core voice (TTS + STT)
    "VOICE_TURN_CONTRACT_VERSION",
    "TELEPHONE_TURN_CONTRACT_VERSION",
    "VOICE_STAGE_STATUSES",
    "VOICE_TURN_STATUSES",
    "USAGE_ROUTING_REQUIREMENT_ID",
    "VOICE_TTS_USAGE_OPERATION",
    "VOICE_STT_USAGE_OPERATION",
    "VoiceRouterError",
    "UsageCapacityError",
    "VoiceProvider",
    "VoiceProviderCapabilities",
    "ProviderInfo",
    "ProviderFactory",
    "RouterDeps",
    "get_default_router_deps",
    "register_voice_provider",
    "get_voice_provider_capabilities",
    "get_voice_provider",
    "list_providers",
    "get_provider_descriptor",
    "list_models",
    "resolve_model",
    "get_catalog_snapshot",
    "catalog_snapshot",
    "text_to_speech",
    "speech_to_text",
    "clear_voice_router_caches",
    # Usage-aware admission (AICAT-033)
    "estimate_synthesis_tokens",
    "estimate_audio_seconds",
    "estimate_synthesis_usage",
    "estimate_transcription_usage",
    "settle_synthesis_usage",
    "settle_transcription_usage",
    "planning_required_usage",
    "apply_voice_stream_settlements",
    "voice_fallback_compatible",
    "get_last_usage_admission",
    "get_last_voice_usage_trace",
    # Unified grounded Abby voice turn
    "DEFAULT_GROUNDED_FALLBACK",
    "GroundingEvidence",
    "VoiceGroundingSource",
    "GroundedSlot",
    "VoiceResponsePlan",
    "VoiceTemplateProvider",
    "GraphRAGVoiceTemplateProvider",
    "buildVoiceGraphRagPromptParts",
    "VoiceStageTrace",
    "VoiceTurnRequest",
    "TelephoneTurnState",
    "VoiceTurnProvenance",
    "VoiceTurnResult",
    "voice_turn_cache_key",
    "process_voice_turn",
    "process_telephone_turn",
    "VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION",
    "VoiceCacheMissEvent",
    "VoiceCacheMissEventError",
    "build_voice_cache_miss_event",
    # Exact precomputed audio runtime resolution (G019)
    "PrecomputedAudioResolution",
    "PrecomputedVoiceAudioResolver",
    "SynthesisIdentity",
    # Backward-compat TTS aliases (formerly in tts_router)
    "TTSProvider",
    "get_tts_provider",
    "register_tts_provider",
    "clear_tts_router_caches",
]
