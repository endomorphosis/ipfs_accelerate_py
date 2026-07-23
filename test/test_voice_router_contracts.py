"""Focused contract tests for the public Abby voice-router boundary.

These tests are deliberately offline. They pin serialization, cache identity,
capability-aware routing, and the legacy TTS/STT entry points without importing
model or remote-provider dependencies.
"""

from __future__ import annotations

import base64
import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from ipfs_accelerate_py.router_deps import RouterDeps
from ipfs_accelerate_py.voice_router import (
    ProviderInfo,
    VOICE_STAGE_STATUSES,
    VOICE_TURN_CONTRACT_VERSION,
    VOICE_TURN_STATUSES,
    VoiceProvider,
    VoiceProviderCapabilities,
    VoiceStageTrace,
    VoiceTurnProvenance,
    VoiceTurnRequest,
    VoiceTurnResult,
    clear_voice_router_caches,
    get_voice_provider,
    get_voice_provider_capabilities,
    process_voice_turn,
    register_voice_provider,
    speech_to_text,
    text_to_speech,
    voice_turn_cache_key,
)


class FixedProvider:
    def __init__(
        self,
        name: str,
        *,
        audio: bytes = b"RIFF\x00\x00\x00\x00WAVEcontract-audio",
        transcript: str = "contract transcript",
    ) -> None:
        self.provider_name = name
        self.audio = audio
        self.transcript = transcript
        self.synthesis_calls = []
        self.transcription_calls = []

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
        self.synthesis_calls.append(
            (text, voice, model_name, device, output_format, dict(kwargs))
        )
        return self.audio

    def transcribe(
        self,
        audio: object,
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        self.transcription_calls.append(
            (audio, model_name, language, device, dict(kwargs))
        )
        return self.transcript


def test_public_contract_symbols_are_exported_from_package() -> None:
    import ipfs_accelerate_py as package

    assert package.VOICE_TURN_CONTRACT_VERSION == VOICE_TURN_CONTRACT_VERSION
    assert package.VOICE_STAGE_STATUSES == VOICE_STAGE_STATUSES
    assert package.VOICE_TURN_STATUSES == VOICE_TURN_STATUSES
    assert package.VoiceProviderCapabilities is VoiceProviderCapabilities
    assert package.VoiceTurnRequest is VoiceTurnRequest
    assert package.VoiceTurnResult is VoiceTurnResult
    assert package.voice_turn_cache_key is voice_turn_cache_key
    assert isinstance(FixedProvider("protocol"), VoiceProvider)


def test_provider_capabilities_are_normalized_serializable_and_queryable() -> None:
    capabilities = VoiceProviderCapabilities(
        transcription=False,
        synthesis=True,
        streaming=True,
        audio_formats=("WAV", ".mp3", "wav", ""),
    )

    assert capabilities.audio_formats == ("wav", "mp3")
    assert capabilities.supports("tts")
    assert capabilities.supports("synthesis")
    assert capabilities.supports("streaming")
    assert not capabilities.supports("stt")
    assert not capabilities.supports("unknown")
    assert json.loads(json.dumps(capabilities.to_dict())) == {
        "transcription": False,
        "synthesis": True,
        "streaming": True,
        "audio_formats": ["wav", "mp3"],
    }
    assert VoiceProviderCapabilities.from_dict(capabilities.to_dict()) == capabilities
    with pytest.raises(TypeError, match="boolean"):
        VoiceProviderCapabilities.from_dict({"transcription": "false"})
    with pytest.raises(TypeError, match="boolean"):
        VoiceProviderCapabilities(transcription=1)  # type: ignore[arg-type]

    assert not get_voice_provider_capabilities("ELEVEN_LABS").can_transcribe
    assert not get_voice_provider_capabilities("assemblyai").can_synthesize


def test_provider_info_and_registry_are_canonical_and_introspectable() -> None:
    clear_voice_router_caches()
    provider = FixedProvider("canonical")
    capabilities = VoiceProviderCapabilities(
        transcription=False, audio_formats=("wav",)
    )
    register_voice_provider(
        "  Contract-Mixed-Case  ",
        lambda: provider,
        capabilities=capabilities,
    )

    assert get_voice_provider("CONTRACT-MIXED-CASE", use_cache=False) is provider
    assert get_voice_provider_capabilities(" contract-mixed-case ") == capabilities
    info = ProviderInfo(" EXAMPLE ", lambda: provider, capabilities)
    assert info.name == "example"
    assert info.to_dict() == {
        "name": "example",
        "capabilities": capabilities.to_dict(),
    }
    with pytest.raises(ValueError):
        register_voice_provider(" ", lambda: provider)
    with pytest.raises(TypeError):
        register_voice_provider("invalid-factory", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        register_voice_provider(
            "invalid-capabilities",
            lambda: provider,
            capabilities={},  # type: ignore[arg-type]
        )


def test_reregistration_invalidates_global_provider_cache() -> None:
    clear_voice_router_caches()
    deps = RouterDeps()
    first = FixedProvider("first")
    second = FixedProvider("second")
    register_voice_provider("contract-reregister", lambda: first)
    assert get_voice_provider("contract-reregister") is first
    assert get_voice_provider("contract-reregister", deps=deps) is first

    register_voice_provider("CONTRACT-REREGISTER", lambda: second)

    assert get_voice_provider("contract-reregister") is second
    assert get_voice_provider("contract-reregister", deps=deps) is second


def test_request_contract_normalizes_validates_and_is_immutable() -> None:
    context: Dict[str, object] = {"channel": "phone"}
    request = VoiceTurnRequest(
        audio=b"caller-audio",
        transcript="  caller transcript  ",
        request_id="  turn-1  ",
        context=context,
        language=" en ",
        locale=" en-US ",
        stt_providers=(" primary ", "primary", "", "fallback"),
        tts_providers=("speech", "speech"),
        minimum_template_confidence=0.25,
        max_template_results=3,
        stt_options={"temperature": 0},
        tts_options={"speed": 1.0},
    )
    context["channel"] = "mutated"

    assert request.transcript == "caller transcript"
    assert request.request_id == "turn-1"
    assert request.effective_language == "en"
    assert request.context == {"channel": "phone"}
    assert request.stt_providers == ("primary", "fallback")
    assert request.tts_providers == ("speech",)
    with pytest.raises(TypeError):
        request.context["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.transcript = "changed"  # type: ignore[misc]

    for kwargs in (
        {},
        {"audio": b""},
        {"audio": object()},
        {"transcript": " "},
        {"transcript": "ok", "minimum_template_confidence": float("nan")},
        {"transcript": "ok", "minimum_template_confidence": -0.01},
        {"transcript": "ok", "minimum_template_confidence": 1.01},
        {"transcript": "ok", "max_template_results": 0},
        {"transcript": "ok", "fallback_text": " "},
    ):
        with pytest.raises((TypeError, ValueError)):
            VoiceTurnRequest(**kwargs)


def test_request_serialization_is_json_safe_and_private_by_default(
    tmp_path: Path,
) -> None:
    raw_audio = b"\x00private-caller-audio\xff"
    request = VoiceTurnRequest(
        audio=raw_audio,
        request_id="turn-json",
        context={"raw": b"context-bytes"},
        fallback_text="Safe fallback",
        stt_options={"nested": [1, 2]},
        tts_options={"format": "wav"},
    )
    payload = request.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["contract_version"] == VOICE_TURN_CONTRACT_VERSION
    assert payload["fallback_text"] == "Safe fallback"
    assert payload["stt_options"] == {"nested": [1, 2]}
    assert payload["tts_options"] == {"format": "wav"}
    assert "audio_base64" not in payload
    assert raw_audio.hex() not in encoded
    assert payload["input_audio_sha256"]
    assert request.to_dict(include_audio=True)["audio_base64"] == base64.b64encode(
        raw_audio
    ).decode("ascii")

    audio_path = tmp_path / "private-caller.wav"
    audio_path.write_bytes(raw_audio)
    path_request = VoiceTurnRequest(audio=str(audio_path))
    path_payload = path_request.to_dict()
    assert str(audio_path) not in json.dumps(path_payload)
    assert path_request.input_audio_sha256 == request.input_audio_sha256
    assert path_request.to_dict(include_audio=True)["audio"] == str(audio_path)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("audio", b"different-audio"),
        ("context", {"channel": "chat"}),
        ("grounding", {"city": "Oakland"}),
        ("language", "es"),
        ("locale", "en-US"),
        ("voice", "calm"),
        ("stt_provider", "remote-stt"),
        ("tts_provider", "remote-tts"),
        ("stt_providers", ("fallback-stt",)),
        ("tts_providers", ("fallback-tts",)),
        ("stt_model", "whisper-small"),
        ("tts_model", "speech-model"),
        ("device", "cuda"),
        ("output_format", "mp3"),
        ("minimum_template_confidence", 0.5),
        ("max_template_results", 2),
        ("fallback_text", "A different safe fallback."),
        ("stt_options", {"temperature": 0.1}),
        ("tts_options", {"speed": 1.1}),
    ],
)
def test_turn_cache_identity_covers_output_affecting_request_fields(
    field_name: str,
    field_value: object,
) -> None:
    base: Dict[str, Any] = {"audio": b"base-audio", "request_id": "request-a"}
    changed = dict(base)
    changed[field_name] = field_value

    assert voice_turn_cache_key(VoiceTurnRequest(**base)) != voice_turn_cache_key(
        VoiceTurnRequest(**changed)
    )


def test_turn_cache_identity_ignores_request_id_and_never_embeds_content() -> None:
    first = VoiceTurnRequest(
        transcript="sensitive transcript", request_id="request-one"
    )
    second = VoiceTurnRequest(
        transcript="sensitive transcript", request_id="request-two"
    )

    first_key = voice_turn_cache_key(first)
    assert first_key == voice_turn_cache_key(second)
    assert "sensitive transcript" not in first_key
    assert "request-one" not in first_key


def test_trace_contract_validates_and_serializes_details() -> None:
    details = {"payload": b"private", "nested": {"ok": True}}
    trace = VoiceStageTrace(
        stage=" synthesis ",
        status="SUCCEEDED",
        duration_ms=1.23456,
        provider=" fixed ",
        details=details,
    )
    details["payload"] = b"changed"
    payload = trace.to_dict()

    assert trace.stage == "synthesis"
    assert trace.status == "succeeded"
    assert trace.provider == "fixed"
    assert payload["duration_ms"] == 1.235
    assert payload["details"]["payload"]["size_bytes"] == len(b"private")
    json.dumps(payload)
    with pytest.raises(TypeError):
        trace.details["new"] = True  # type: ignore[index]
    for kwargs in (
        {"stage": "", "status": "succeeded", "duration_ms": 0},
        {"stage": "tts", "status": "unknown", "duration_ms": 0},
        {"stage": "tts", "status": "failed", "duration_ms": -1},
        {"stage": "tts", "status": "failed", "duration_ms": float("nan")},
        {"stage": "tts", "status": "failed", "duration_ms": float("inf")},
    ):
        with pytest.raises(ValueError):
            VoiceStageTrace(**kwargs)


def test_result_contract_exposes_stable_receipt_without_audio_by_default() -> None:
    trace = VoiceStageTrace("synthesis", "succeeded", 2.3456, provider="fixed")
    provenance = VoiceTurnProvenance(
        stt_provider="supplied_transcript",
        tts_provider="fixed",
        transcript_sha256="transcript-hash",
        response_text_sha256="response-hash",
        output_audio_sha256="audio-hash",
        metadata={"intent": "resource_search"},
    )
    result = VoiceTurnResult(
        request_id=" turn-result ",
        status="DEGRADED",
        transcript="hello",
        response_text="  safe response  ",
        audio=b"RIFF\x00\x00\x00\x00WAVEaudio",
        audio_format=".WAV",
        provenance=provenance,
        traces=[trace],  # type: ignore[arg-type]
        fallback_reasons=["provider_fallback", "provider_fallback"],  # type: ignore[arg-type]
        cache_key=" cache-key ",
    )
    payload = result.to_dict()

    assert result.request_id == "turn-result"
    assert result.status == "degraded"
    assert result.response_text == "safe response"
    assert result.audio_format == "wav"
    assert result.fallback_reason == "provider_fallback"
    assert result.fallbacks == ("provider_fallback",)
    assert result.spoken_text == result.response_text
    assert result.intent == "resource_search"
    assert result.total_duration_ms == 2.346
    assert result.provider_selection == {
        "transcription": "supplied_transcript",
        "retrieval": None,
        "synthesis": "fixed",
    }
    assert payload["contract_version"] == VOICE_TURN_CONTRACT_VERSION
    assert payload["audio_size_bytes"] == len(result.audio or b"")
    assert "audio_base64" not in payload
    assert result.to_dict(include_audio=True)["audio_base64"] == base64.b64encode(
        result.audio or b""
    ).decode("ascii")
    json.dumps(payload)

    with pytest.raises(ValueError):
        VoiceTurnResult(
            request_id="turn",
            status="unknown",
            transcript="",
            response_text="fallback",
            audio=None,
            audio_format=None,
            provenance=VoiceTurnProvenance(),
        )
    with pytest.raises(TypeError):
        VoiceTurnResult(
            request_id="turn",
            status="text_only",
            transcript="hello",
            response_text="fallback",
            audio="not-bytes",  # type: ignore[arg-type]
            audio_format=None,
            provenance=VoiceTurnProvenance(),
        )


def test_capability_aware_turn_routing_skips_unsupported_factory() -> None:
    clear_voice_router_caches()
    unsupported_factory_calls = []
    tts_provider = FixedProvider("contract-tts")
    register_voice_provider(
        "contract-stt-only",
        lambda: unsupported_factory_calls.append(True) or FixedProvider("bad"),
        capabilities=VoiceProviderCapabilities(synthesis=False),
    )
    register_voice_provider(
        "contract-tts-only",
        lambda: tts_provider,
        capabilities=VoiceProviderCapabilities(transcription=False),
    )

    result = process_voice_turn(
        VoiceTurnRequest(
            transcript="caller asks for help",
            tts_providers=("contract-stt-only", "contract-tts-only"),
        ),
        deps=RouterDeps(),
    )

    assert not unsupported_factory_calls
    assert result.audio == tts_provider.audio
    assert result.provenance.tts_provider == "contract-tts-only"
    assert [trace.provider for trace in result.traces if trace.stage == "synthesis"] == [
        "contract-tts-only"
    ]


def test_legacy_entrypoint_signatures_and_provider_injection_remain_compatible(
    tmp_path: Path,
) -> None:
    tts_parameters = inspect.signature(text_to_speech).parameters
    stt_parameters = inspect.signature(speech_to_text).parameters
    assert list(tts_parameters) == [
        "text",
        "voice",
        "model_name",
        "device",
        "output_format",
        "output_path",
        "provider",
        "provider_instance",
        "deps",
        "kwargs",
    ]
    assert list(stt_parameters) == [
        "audio",
        "model_name",
        "language",
        "device",
        "provider",
        "provider_instance",
        "deps",
        "kwargs",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in tts_parameters.items()
        if name not in {"text", "kwargs"}
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in stt_parameters.items()
        if name not in {"audio", "kwargs"}
    )

    provider = FixedProvider("legacy")
    deps = RouterDeps()
    output_path = tmp_path / "reply.wav"
    returned = text_to_speech(
        "hello",
        voice="calm",
        model_name="tts-model",
        device="cpu",
        output_format="wav",
        output_path=str(output_path),
        provider_instance=provider,
        deps=deps,
        speed=1.0,
    )
    transcript = speech_to_text(
        b"audio",
        model_name="stt-model",
        language="en",
        device="cpu",
        provider_instance=provider,
        deps=deps,
        prompt="211",
    )

    assert returned == str(output_path)
    assert output_path.read_bytes() == provider.audio
    assert transcript == provider.transcript
    assert provider.synthesis_calls == [
        ("hello", "calm", "tts-model", "cpu", "wav", {"speed": 1.0})
    ]
    assert provider.transcription_calls == [
        (b"audio", "stt-model", "en", "cpu", {"prompt": "211"})
    ]


def test_legacy_cache_identity_separates_provider_instances_and_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    deps = RouterDeps()
    first = FixedProvider("same-name", audio=b"first", transcript="first")
    second = FixedProvider("same-name", audio=b"second", transcript="second")

    assert text_to_speech("same", provider_instance=first, deps=deps) == b"first"
    assert text_to_speech("same", provider_instance=second, deps=deps) == b"second"
    assert (
        text_to_speech(
            "same", provider_instance=first, output_format="mp3", deps=deps
        )
        == b"first"
    )
    assert speech_to_text(b"same", provider_instance=first, deps=deps) == "first"
    assert speech_to_text(b"same", provider_instance=second, deps=deps) == "second"

    path = tmp_path / "mutable.wav"
    path.write_bytes(b"first-content")
    assert speech_to_text(str(path), provider_instance=first, deps=deps) == "first"
    calls_before = len(first.transcription_calls)
    path.write_bytes(b"second-content")
    assert speech_to_text(str(path), provider_instance=first, deps=deps) == "first"
    assert len(first.transcription_calls) == calls_before + 1
