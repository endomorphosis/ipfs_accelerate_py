"""Offline validation for Abby remote providers and router fallback behavior."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from ipfs_accelerate_py.router_deps import RouterDeps
from ipfs_accelerate_py.voice_providers.abby import (
    AbbyProviderError,
    AbbyResiliencePolicy,
    HTTPRequest,
    HTTPResponse,
    HuggingFaceWhisperHTTPProvider,
    IndexTTSHTTPProvider,
)
from ipfs_accelerate_py.voice_router import (
    VoiceProviderCapabilities,
    VoiceTurnRequest,
    get_voice_provider,
    get_voice_provider_capabilities,
    process_voice_turn,
    register_voice_provider,
)

WAV_AUDIO = b"RIFF\x14\x00\x00\x00WAVEfmt abby-audio"


class RecordingTransport:
    def __init__(self, outcomes: List[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: List[Tuple[HTTPRequest, float]] = []

    def __call__(self, request: HTTPRequest, timeout: float) -> HTTPResponse:
        self.calls.append((request, timeout))
        if not self.outcomes:
            raise AssertionError("unexpected transport call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, HTTPResponse)
        return outcome


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FixedProvider:
    def __init__(
        self,
        name: str,
        *,
        audio: bytes = WAV_AUDIO,
        transcript: str = "I need food assistance",
        synthesis_error: Optional[Exception] = None,
        transcription_error: Optional[Exception] = None,
    ) -> None:
        self.provider_name = name
        self.audio = audio
        self.transcript = transcript
        self.synthesis_error = synthesis_error
        self.transcription_error = transcription_error
        self.calls: List[str] = []

    def synthesize(self, text: str, **kwargs: object) -> bytes:
        self.calls.append("synthesis")
        if self.synthesis_error is not None:
            raise self.synthesis_error
        return self.audio

    def transcribe(self, audio: object, **kwargs: object) -> str:
        self.calls.append("transcription")
        if self.transcription_error is not None:
            raise self.transcription_error
        return self.transcript


def json_response(value: object, status: int = 200) -> HTTPResponse:
    return HTTPResponse(
        status,
        json.dumps(value).encode("utf-8"),
        {"Content-Type": "application/json"},
    )


def test_abby_provider_import_and_builtin_capabilities_are_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URL", raising=False)
    monkeypatch.delenv("WALLET_INDEXTTS_SPACE_URL", raising=False)
    monkeypatch.delenv("WALLET_INDEXTTS_FALLBACK_SPACE_URL", raising=False)

    tts_capabilities = get_voice_provider_capabilities("indextts")
    stt_capabilities = get_voice_provider_capabilities("abby_hf_whisper")
    assert tts_capabilities.can_synthesize
    assert not tts_capabilities.can_transcribe
    assert stt_capabilities.can_transcribe
    assert not stt_capabilities.can_synthesize

    provider = get_voice_provider("abby_indextts", use_cache=False)
    assert isinstance(provider, IndexTTSHTTPProvider)
    assert provider.endpoints == ()
    with pytest.raises(AbbyProviderError, match="no configured endpoint"):
        provider.synthesize("hello")
    assert provider.last_receipt is not None
    assert provider.last_receipt.error_code == "provider_not_configured"


def test_indextts_adapter_normalizes_wire_request_and_base64_response() -> None:
    encoded = base64.b64encode(WAV_AUDIO).decode("ascii")
    transport = RecordingTransport([json_response({"data": {"audioBase64": encoded}})])
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test/generate"],
        token="tts-secret",
        bill_to="publicus",
        default_model="Publicus/IndexTTS",
        policy=AbbyResiliencePolicy(timeout_seconds=7, max_retries=0),
        transport=transport,
    )

    result = provider.synthesize(
        "Call 2-1-1.",
        voice="Abby",
        model_name="IndexTeam/IndexTTS-2",
        output_format=".wav",
        temperature=0.2,
        model="cannot override",
    )

    assert result == WAV_AUDIO
    request, timeout = transport.calls[0]
    assert request.method == "POST"
    assert request.url == "https://tts.example.test/generate"
    assert timeout == 7
    assert request.headers["Authorization"] == "Bearer tts-secret"
    assert request.headers["X-HF-Bill-To"] == "publicus"
    payload = json.loads(request.body or b"{}")
    assert payload == {
        "text": "Call 2-1-1.",
        "model": "IndexTeam/IndexTTS-2",
        "output_format": "wav",
        "voice": "Abby",
        "temperature": 0.2,
    }
    assert provider.last_receipt is not None
    receipt = provider.last_receipt.to_dict()
    assert receipt["status"] == "completed"
    assert "tts-secret" not in json.dumps(receipt)


def test_indextts_accepts_direct_audio_and_same_origin_download() -> None:
    direct = RecordingTransport(
        [HTTPResponse(200, WAV_AUDIO, {"Content-Type": "audio/wav"})]
    )
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test/generate"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=direct,
    )
    assert provider.synthesize("hello") == WAV_AUDIO

    download = RecordingTransport(
        [
            json_response({"result": {"audio_url": "/files/answer.wav"}}),
            HTTPResponse(200, WAV_AUDIO, {"Content-Type": "audio/wav"}),
        ]
    )
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test/generate"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=download,
    )
    assert provider.synthesize("hello") == WAV_AUDIO
    assert [call[0].method for call in download.calls] == ["POST", "GET"]
    assert download.calls[1][0].url == "https://tts.example.test/files/answer.wav"

    unsafe = RecordingTransport(
        [json_response({"audio_url": "http://metadata.internal/secret"})]
    )
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test/generate"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=unsafe,
    )
    with pytest.raises(AbbyProviderError, match="failed across"):
        provider.synthesize("hello")
    assert len(unsafe.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        json_response({}),
        json_response({"audioBase64": "not-base64"}),
        HTTPResponse(200, b"", {"Content-Type": "audio/wav"}),
    ],
)
def test_indextts_rejects_malformed_or_empty_audio(
    response: HTTPResponse,
) -> None:
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test/generate"],
        policy=AbbyResiliencePolicy(max_retries=2),
        transport=RecordingTransport([response]),
    )
    with pytest.raises(AbbyProviderError) as caught:
        provider.synthesize("hello")
    assert caught.value.code == "invalid_remote_response"
    assert caught.value.retryable is False
    assert provider.last_receipt is not None
    assert len(provider.last_receipt.attempts) == 1


def test_whisper_adapter_normalizes_bytes_model_headers_and_nested_text() -> None:
    transport = RecordingTransport(
        [json_response({"results": [{"chunks": [{"text": "  food help  "}]}]})]
    )
    provider = HuggingFaceWhisperHTTPProvider(
        ["https://router.example.test/models"],
        token="whisper-secret",
        bill_to="publicus",
        policy=AbbyResiliencePolicy(timeout_seconds=9, max_retries=0),
        transport=transport,
    )

    transcript = provider.transcribe(
        WAV_AUDIO,
        model_name="openai/whisper large",
        language="en-US",
        content_type="audio/x-wav",
    )

    assert transcript == "food help"
    request, timeout = transport.calls[0]
    assert request.url == (
        "https://router.example.test/models/openai/whisper%20large"
    )
    assert request.body == WAV_AUDIO
    assert request.headers["Content-Type"] == "audio/x-wav"
    assert request.headers["Authorization"] == "Bearer whisper-secret"
    assert request.headers["X-HF-Bill-To"] == "publicus"
    assert request.headers["X-Wallet-STT-Language"] == "en-US"
    assert timeout == 9


def test_whisper_reads_local_audio_and_rejects_invalid_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "caller.webm"
    audio_path.write_bytes(b"OggS-private-audio")
    transport = RecordingTransport([json_response({"text": " transcript "})])
    provider = HuggingFaceWhisperHTTPProvider(
        ["https://router.example.test/models"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=transport,
    )
    assert provider.transcribe(str(audio_path)) == "transcript"
    assert transport.calls[0][0].body == b"OggS-private-audio"
    assert transport.calls[0][0].headers["Content-Type"] in {
        "video/webm",
        "audio/webm",
    }

    for audio in (b"", str(tmp_path / "missing.wav")):
        with pytest.raises(ValueError):
            provider.transcribe(audio)
    assert len(transport.calls) == 1


def test_retry_backoff_endpoint_order_and_degraded_receipt() -> None:
    sleeps: List[float] = []
    transport = RecordingTransport(
        [
            TimeoutError(
                "Authorization: Bearer super-secret timed out for safe prompt"
            ),
            HTTPResponse(503, b"busy"),
            json_response({"audioBase64": base64.b64encode(WAV_AUDIO).decode()}),
        ]
    )
    provider = IndexTTSHTTPProvider(
        ["https://primary.example.test", "https://fallback.example.test"],
        policy=AbbyResiliencePolicy(
            timeout_seconds=3,
            max_retries=1,
            backoff_seconds=0.25,
            backoff_multiplier=2,
            max_backoff_seconds=1,
            circuit_failure_threshold=3,
        ),
        transport=transport,
        sleeper=sleeps.append,
    )

    assert provider.synthesize("safe prompt") == WAV_AUDIO
    assert [call[0].url for call in transport.calls] == [
        "https://primary.example.test",
        "https://primary.example.test",
        "https://fallback.example.test",
    ]
    assert [call[1] for call in transport.calls] == [3, 3, 3]
    assert sleeps == [0.25]
    assert provider.last_receipt is not None
    receipt_json = json.dumps(provider.last_receipt.to_dict())
    assert provider.last_receipt.status == "degraded"
    assert provider.last_receipt.selected_endpoint == "https://fallback.example.test"
    assert "super-secret" not in receipt_json
    assert "safe prompt" not in receipt_json


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_transient_http_statuses_are_retried(status: int) -> None:
    transport = RecordingTransport(
        [HTTPResponse(status, b"failed"), json_response({"text": "ok"})]
    )
    provider = HuggingFaceWhisperHTTPProvider(
        ["https://whisper.example.test"],
        policy=AbbyResiliencePolicy(
            max_retries=1, backoff_seconds=0, circuit_failure_threshold=2
        ),
        transport=transport,
    )
    assert provider.transcribe(WAV_AUDIO) == "ok"
    assert len(transport.calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_terminal_http_statuses_are_not_retried_or_counted_for_circuit(
    status: int,
) -> None:
    transport = RecordingTransport([HTTPResponse(status, b"no")])
    provider = HuggingFaceWhisperHTTPProvider(
        ["https://whisper.example.test"],
        policy=AbbyResiliencePolicy(
            max_retries=3, circuit_failure_threshold=1
        ),
        transport=transport,
    )
    with pytest.raises(AbbyProviderError) as caught:
        provider.transcribe(WAV_AUDIO)
    assert not caught.value.retryable
    assert len(transport.calls) == 1
    assert provider.circuit_state() == "closed"


def test_circuit_opens_fast_fails_and_recovers_with_half_open_probe() -> None:
    clock = FakeClock()
    transport = RecordingTransport(
        [
            TimeoutError("first"),
            TimeoutError("second"),
            json_response({"text": "recovered"}),
        ]
    )
    provider = HuggingFaceWhisperHTTPProvider(
        ["https://whisper.example.test"],
        policy=AbbyResiliencePolicy(
            max_retries=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=10,
        ),
        transport=transport,
        clock=clock,
    )

    for _ in range(2):
        with pytest.raises(AbbyProviderError):
            provider.transcribe(WAV_AUDIO)
    assert provider.circuit_state() == "open"
    with pytest.raises(AbbyProviderError) as open_error:
        provider.transcribe(WAV_AUDIO)
    assert open_error.value.code == "circuit_open"
    assert len(transport.calls) == 2

    clock.advance(10)
    assert provider.transcribe(WAV_AUDIO) == "recovered"
    assert provider.circuit_state() == "closed"
    assert len(transport.calls) == 3


def test_circuit_half_open_failure_reopens() -> None:
    clock = FakeClock()
    transport = RecordingTransport(
        [TimeoutError("open"), TimeoutError("probe")]
    )
    provider = IndexTTSHTTPProvider(
        ["https://tts.example.test"],
        policy=AbbyResiliencePolicy(
            max_retries=0,
            circuit_failure_threshold=1,
            circuit_recovery_seconds=5,
        ),
        transport=transport,
        clock=clock,
    )
    with pytest.raises(AbbyProviderError):
        provider.synthesize("hello")
    clock.advance(5)
    with pytest.raises(AbbyProviderError):
        provider.synthesize("hello")
    assert provider.circuit_state() == "open"
    with pytest.raises(AbbyProviderError) as caught:
        provider.synthesize("hello")
    assert caught.value.code == "circuit_open"
    assert len(transport.calls) == 2


def test_provider_cache_identity_isolated_by_endpoint_and_token() -> None:
    policy = AbbyResiliencePolicy(max_retries=0)
    first = IndexTTSHTTPProvider(
        ["https://one.example.test"], token="one", policy=policy
    )
    second = IndexTTSHTTPProvider(
        ["https://two.example.test"], token="one", policy=policy
    )
    third = IndexTTSHTTPProvider(
        ["https://one.example.test"], token="two", policy=policy
    )
    assert len({first.cache_identity, second.cache_identity, third.cache_identity}) == 3
    assert "one" not in first.cache_identity


def test_router_uses_remote_then_local_tts_and_records_degraded_receipt() -> None:
    remote_transport = RecordingTransport([HTTPResponse(503, b"busy")])
    remote = IndexTTSHTTPProvider(
        ["https://tts.example.test"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=remote_transport,
    )
    local = FixedProvider("abby-local-tts")
    register_voice_provider(
        "abby-test-local-tts",
        lambda: local,
        capabilities=VoiceProviderCapabilities(transcription=False),
    )

    result = process_voice_turn(
        VoiceTurnRequest(
            transcript="I need food",
            tts_providers=("abby-test-local-tts",),
            fallback_text="Please contact 211.",
        ),
        tts_provider=remote,
        deps=RouterDeps(),
    )

    assert result.status == "degraded"
    assert result.audio == WAV_AUDIO
    assert result.provenance.tts_provider == "abby-test-local-tts"
    assert "tts_provider_fallback" in result.fallback_reasons
    synthesis = [trace for trace in result.traces if trace.stage == "synthesis"]
    assert [(trace.provider, trace.status) for trace in synthesis] == [
        ("abby_indextts", "failed"),
        ("abby-test-local-tts", "succeeded"),
    ]
    failed_receipt = synthesis[0].details["provider_receipt"]
    assert failed_receipt["status"] == "degraded"
    assert failed_receipt["attempts"][0]["http_status"] == 503


def test_router_uses_remote_then_local_stt_in_exact_order() -> None:
    remote = HuggingFaceWhisperHTTPProvider(
        ["https://whisper.example.test"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=RecordingTransport([TimeoutError("unavailable")]),
    )
    local = FixedProvider("abby-local-stt", transcript="successful transcript")
    tts = FixedProvider("reply-tts")
    register_voice_provider(
        "abby-test-local-stt",
        lambda: local,
        capabilities=VoiceProviderCapabilities(synthesis=False),
    )

    result = process_voice_turn(
        VoiceTurnRequest(
            audio=WAV_AUDIO,
            stt_providers=("abby-test-local-stt",),
            fallback_text="Please contact 211.",
        ),
        stt_provider=remote,
        tts_provider=tts,
        deps=RouterDeps(),
    )

    assert result.transcript == "successful transcript"
    assert result.provenance.stt_provider == "abby-test-local-stt"
    assert "stt_provider_fallback" in result.fallback_reasons
    transcription = [
        trace for trace in result.traces if trace.stage == "transcription"
    ]
    assert [(trace.provider, trace.status) for trace in transcription] == [
        ("abby_whisper", "failed"),
        ("abby-test-local-stt", "succeeded"),
    ]


def test_capability_only_explicit_chain_does_not_fall_through_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: List[str] = []
    register_voice_provider(
        "abby-test-stt-only",
        lambda: constructed.append("constructed") or FixedProvider("stt-only"),
        capabilities=VoiceProviderCapabilities(synthesis=False),
    )

    result = process_voice_turn(
        VoiceTurnRequest(
            transcript="hello",
            tts_provider="abby-test-stt-only",
            fallback_text="safe reply",
        ),
        deps=RouterDeps(),
    )

    assert constructed == []
    assert result.status == "text_only"
    assert result.audio is None
    assert result.provenance.tts_provider is None
    assert "tts_failed" in result.fallback_reasons


def test_all_stt_failures_return_private_structured_failed_receipt() -> None:
    raw_audio = b"RIFF\x00\x00\x00\x00WAVE-private-caller-audio"
    provider = FixedProvider(
        "bad-stt",
        transcription_error=RuntimeError(
            "Authorization: Bearer caller-secret failed for private-caller-audio"
        ),
    )
    tts = FixedProvider("fallback-tts")

    result = process_voice_turn(
        VoiceTurnRequest(audio=raw_audio, fallback_text="Please contact 211."),
        stt_provider=provider,
        tts_provider=tts,
        deps=RouterDeps(),
    )

    assert result.status == "failed"
    assert result.transcript == ""
    assert result.audio == WAV_AUDIO
    assert "stt_failed" in result.fallback_reasons
    encoded = json.dumps(result.to_dict())
    assert "caller-secret" not in encoded
    assert "private-caller-audio" not in encoded
    assert "Bearer [redacted]" in encoded


def test_synchronous_router_closes_unexpected_coroutine_and_falls_back() -> None:
    class AsyncProvider(FixedProvider):
        async def synthesize(self, text: str, **kwargs: object) -> bytes:
            return WAV_AUDIO

    fallback = FixedProvider("sync-fallback")
    register_voice_provider(
        "abby-test-sync-fallback",
        lambda: fallback,
        capabilities=VoiceProviderCapabilities(transcription=False),
    )
    result = process_voice_turn(
        VoiceTurnRequest(
            transcript="hello",
            tts_providers=("abby-test-sync-fallback",),
            fallback_text="safe",
        ),
        tts_provider=AsyncProvider("async"),
        deps=RouterDeps(),
    )
    assert result.audio == WAV_AUDIO
    assert result.provenance.tts_provider == "abby-test-sync-fallback"
    failed = [
        trace
        for trace in result.traces
        if trace.stage == "synthesis" and trace.status == "failed"
    ]
    assert "non-empty audio bytes" in str(failed[0].error)


def test_async_transport_is_rejected_and_coroutine_closed() -> None:
    async def async_transport(
        request: HTTPRequest, timeout: float
    ) -> HTTPResponse:
        return json_response({"text": "not accepted"})

    provider = HuggingFaceWhisperHTTPProvider(
        ["https://whisper.example.test"],
        policy=AbbyResiliencePolicy(max_retries=0),
        transport=async_transport,  # type: ignore[arg-type]
    )
    with pytest.raises(AbbyProviderError) as caught:
        provider.transcribe(WAV_AUDIO)
    assert caught.value.code == "invalid_remote_response"

    # The synchronous adapter remains deterministic even if called by code
    # that itself is running in an event loop; it never nests asyncio.run().
    async def invoke() -> str:
        try:
            provider.transcribe(WAV_AUDIO)
        except AbbyProviderError as error:
            return error.code
        return "unexpected"

    assert asyncio.run(invoke()) == "invalid_remote_response"
