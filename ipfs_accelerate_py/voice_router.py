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
import math
import os
import re
import string
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from .router_deps import RouterDeps, get_default_router_deps

logger = logging.getLogger(__name__)

VOICE_TURN_CONTRACT_VERSION = "1.0"
VOICE_STAGE_STATUSES = frozenset({"succeeded", "failed", "skipped"})
VOICE_TURN_STATUSES = frozenset({"completed", "degraded", "text_only", "failed"})


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
    "openai": VoiceProviderCapabilities(),
    "elevenlabs": VoiceProviderCapabilities(transcription=False),
    "assemblyai": VoiceProviderCapabilities(synthesis=False),
    "huggingface": VoiceProviderCapabilities(),
    "backend_manager": VoiceProviderCapabilities(),
}

_BUILTIN_PROVIDER_ALIASES: Mapping[str, str] = {
    "openai_voice": "openai",
    "eleven_labs": "elevenlabs",
    "eleven": "elevenlabs",
    "assembly_ai": "assemblyai",
    "hf": "huggingface",
    "local_hf": "huggingface",
    "accelerate": "backend_manager",
}


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
        grounding: Optional[Mapping[str, object]] = None,
        max_results: int = 5,
    ) -> Optional[VoiceResponsePlan]:
        raw = _call_with_supported_keywords(
            self._backend_callable(),
            transcript,
            context=dict(context or {}),
            language=language,
            grounding=dict(grounding or {}),
            max_results=max_results,
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

            model = model_name or os.getenv(
                "IPFS_ACCELERATE_PY_TTS_MODEL", "suno/bark-small"
            )
            device_str = (
                device
                or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cpu")
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
                or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE")
                or os.getenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cpu")
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


def _safe_stage_error(error: Exception) -> str:
    """Normalize adapter errors without embedding caller audio or tracebacks."""
    message = " ".join(str(error).replace("\x00", "").split())
    # Credentials occasionally appear as URL query values in remote errors.
    message = re.sub(
        r"(?i)(api[_-]?key|token|authorization|secret)=?[\w.+/=-]+",
        r"\1=[redacted]",
        message,
    )
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
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return f"abby_voice_turn::{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def voice_turn_cache_key(
    request: VoiceTurnRequest,
    *,
    template_provider: Optional[VoiceTemplateProvider] = None,
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

    if not candidates:
        try:
            provider = get_voice_provider(None, deps=deps)
            candidates.append((_provider_display_name(provider, "auto"), provider, None))
        except Exception as error:
            candidates.append(("auto", None, error))
    return tuple(candidates)


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
    fields = []
    try:
        parsed = string.Formatter().parse(template)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name:
                raise ValueError("invalid_template_slot: empty field")
            if any(marker in field_name for marker in (".", "[", "]")):
                raise ValueError(
                    f"invalid_template_slot: unsafe field expression {field_name!r}"
                )
            if format_spec or conversion:
                raise ValueError(
                    f"invalid_template_slot: formatting is not allowed for {field_name!r}"
                )
            fields.append(field_name)
    except (KeyError, IndexError, ValueError) as error:
        if str(error).startswith("invalid_template_slot"):
            raise
        raise ValueError(f"invalid_template: {error}") from error
    return tuple(dict.fromkeys(fields))


def _render_grounded_plan(
    plan: VoiceResponsePlan,
    *,
    grounding: Mapping[str, object],
    minimum_confidence: float,
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
        rendered_values[name] = str(slot.value).strip()

    if fields and not plan.evidence:
        raise ValueError("missing_grounding_evidence")
    try:
        rendered = plan.template.format_map(rendered_values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid_template: {error}") from error
    return _normalize_spoken_text(rendered), tuple(
        slots_by_name[name] for name in fields
    )


def _normalize_spoken_text(text: str) -> str:
    """Remove visual citations while retaining their machine provenance."""
    spoken = str(text or "")
    spoken = re.sub(
        r"(?i)\[(?:source|citation|evidence)(?:\s+\d+)?\]"
        r"\((?:https?|ipfs)://[^)]+\)",
        "",
        spoken,
    )
    spoken = re.sub(r"\[([^\]]+)\]\((?:https?|ipfs)://[^)]+\)", r"\1", spoken)
    spoken = re.sub(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", spoken)
    spoken = re.sub(
        r"(?is)\s+(?:sources?|evidence|citations?)\s*:\s*"
        r"(?:(?:https?|ipfs)://\S+|bafy[a-z0-9]+).*$",
        "",
        spoken,
    )
    spoken = re.sub(r"(?i)(?:https?|ipfs)://\S+", "", spoken)
    spoken = re.sub(r"\bbafy[a-z0-9]{20,}\b", "", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"[ \t]+", " ", spoken)
    spoken = re.sub(r"\s+([,.;:!?])", r"\1", spoken)
    spoken = re.sub(r"([.!?])(?:\s*\1)+", r"\1", spoken)
    spoken = re.sub(r"\s*\n+\s*", " ", spoken).strip(" \t\r\n-")
    if not spoken:
        raise ValueError("empty_spoken_response_after_citation_stripping")
    return spoken


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


def process_voice_turn(
    request: VoiceTurnRequest,
    *,
    stt_provider: Optional[VoiceProvider] = None,
    template_provider: Optional[VoiceTemplateProvider] = None,
    tts_provider: Optional[VoiceProvider] = None,
    stt_provider_instance: Optional[VoiceProvider] = None,
    tts_provider_instance: Optional[VoiceProvider] = None,
    deps: Optional[RouterDeps] = None,
) -> VoiceTurnResult:
    """Run STT → grounded response-plan retrieval → rendering → TTS.

    Runtime failures are returned as structured degraded receipts. Invalid
    request contracts still raise immediately, keeping programmer errors
    separate from provider availability.
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
        for provider_name, provider_object, resolution_error in _provider_candidates(
            primary_stt,
            preferred=request.stt_provider,
            fallbacks=request.stt_providers,
            operation="transcription",
            deps=resolved_deps,
        ):
            started_at = time.perf_counter()
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
                    raise TypeError("transcribe returned no non-empty text")
                transcript = raw_transcript.strip()
                used_stt_provider = provider_name
                traces.append(
                    VoiceStageTrace(
                        "transcription",
                        "succeeded",
                        _duration_ms(started_at),
                        provider=provider_name,
                    )
                )
                if transcription_failures:
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
                        error=_safe_stage_error(error),
                    )
                )
        if not transcript:
            fallback_reasons.append("stt_failed")

    plan: Optional[VoiceResponsePlan] = None
    grounded_slots: Tuple[GroundedSlot, ...] = ()
    template_name = _template_provider_name(template_provider)
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

    response_text = request.fallback_text
    if plan is not None:
        started_at = time.perf_counter()
        try:
            response_text, grounded_slots = _render_grounded_plan(
                plan,
                grounding=request.grounding,
                minimum_confidence=request.minimum_template_confidence,
            )
            traces.append(
                VoiceStageTrace(
                    "rendering",
                    "succeeded",
                    _duration_ms(started_at),
                    provider=template_name,
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
                    provider=template_name,
                    error=_safe_stage_error(error),
                )
            )
    else:
        traces.append(
            VoiceStageTrace(
                "rendering",
                "skipped",
                0.0,
                provider=template_name,
                details={"reason": "grounded_template_unavailable"},
            )
        )

    output_audio: Optional[bytes] = None
    used_tts_provider: Optional[str] = None
    synthesis_failures = 0
    for provider_name, provider_object, resolution_error in _provider_candidates(
        primary_tts,
        preferred=request.tts_provider,
        fallbacks=request.tts_providers,
        operation="synthesis",
        deps=resolved_deps,
    ):
        started_at = time.perf_counter()
        if resolution_error is not None or provider_object is None:
            synthesis_failures += 1
            traces.append(
                VoiceStageTrace(
                    "synthesis",
                    "failed",
                    _duration_ms(started_at),
                    provider=provider_name,
                    error=_safe_stage_error(
                        resolution_error or RuntimeError("provider could not be resolved")
                    ),
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
                raise TypeError("synthesize returned no non-empty audio bytes")
            output_audio = raw_audio
            used_tts_provider = provider_name
            traces.append(
                VoiceStageTrace(
                    "synthesis",
                    "succeeded",
                    _duration_ms(started_at),
                    provider=provider_name,
                    details={"audio_size_bytes": len(raw_audio)},
                )
            )
            if synthesis_failures:
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
                    error=_safe_stage_error(error),
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
        template_provider=template_name,
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
            provider=_provider_instance_cache_identity(provider_instance, provider),
            model_name=model_name,
            text=text,
            voice=voice,
            device=device,
            output_format=output_format,
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
                    provider=_provider_instance_cache_identity(provider_instance, provider),
                    model_name=model_name,
                    text=text,
                    voice=voice,
                    device=device,
                    output_format=output_format,
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
            provider=_provider_instance_cache_identity(provider_instance, provider),
            model_name=model_name,
            audio=audio,
            language=language,
            device=device,
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
                    provider=_provider_instance_cache_identity(provider_instance, provider),
                    model_name=model_name,
                    audio=audio,
                    language=language,
                    device=device,
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

__all__ = [
    # Core voice (TTS + STT)
    "VOICE_TURN_CONTRACT_VERSION",
    "VOICE_STAGE_STATUSES",
    "VOICE_TURN_STATUSES",
    "VoiceProvider",
    "VoiceProviderCapabilities",
    "ProviderInfo",
    "ProviderFactory",
    "RouterDeps",
    "get_default_router_deps",
    "register_voice_provider",
    "get_voice_provider_capabilities",
    "get_voice_provider",
    "text_to_speech",
    "speech_to_text",
    "clear_voice_router_caches",
    # Unified grounded Abby voice turn
    "DEFAULT_GROUNDED_FALLBACK",
    "GroundingEvidence",
    "VoiceGroundingSource",
    "GroundedSlot",
    "VoiceResponsePlan",
    "VoiceTemplateProvider",
    "GraphRAGVoiceTemplateProvider",
    "VoiceStageTrace",
    "VoiceTurnRequest",
    "VoiceTurnProvenance",
    "VoiceTurnResult",
    "voice_turn_cache_key",
    "process_voice_turn",
    # Backward-compat TTS aliases (formerly in tts_router)
    "TTSProvider",
    "get_tts_provider",
    "register_tts_provider",
    "clear_tts_router_caches",
]
