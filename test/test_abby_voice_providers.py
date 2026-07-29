"""Offline validation for Abby remote providers and router fallback behavior."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import pytest

from ipfs_accelerate_py import voice_providers
from ipfs_accelerate_py.voice_providers import abby as abby_module
from ipfs_accelerate_py.router_deps import RouterDeps
from ipfs_accelerate_py.voice_providers.abby import (
    AbbyProviderError,
    AbbyResiliencePolicy,
    HTTPRequest,
    HTTPResponse,
    HuggingFaceWhisperHTTPProvider,
    IndexTTSHTTPProvider,
    PUBLICUS_INDEXTTS_MODEL,
    PUBLICUS_INDEXTTS_SPACE_URL,
    PublicusIndexTTSProvider,
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


class FakePublicusSpaceClient:
    def __init__(
        self,
        endpoint: str,
        timeout: float,
        headers_factory,
        *,
        wait_outcome: Optional[object] = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.headers = dict(headers_factory())
        self.wait_outcome = wait_outcome
        self.uploads: List[Tuple[str, bytes, str]] = []
        self.resolved_api_names: List[str] = []
        self.queue_calls: List[Tuple[int, Tuple[object, ...]]] = []
        self.fetches: List[object] = []
        self.closed = False

    def get_config(self) -> Mapping[str, object]:
        return {"dependencies": "fake"}

    def resolve_fn_index(
        self, api_name: str, config: Mapping[str, object]
    ) -> int:
        assert config == {"dependencies": "fake"}
        self.resolved_api_names.append(api_name)
        return 7 if api_name == "/gen_batch" else 6

    def lookup_dependency_input_count(
        self, fn_index: int, config: Mapping[str, object]
    ) -> int:
        assert fn_index in {6, 7}
        assert config == {"dependencies": "fake"}
        return 25

    def upload_file(self, name: str, data: bytes, mime_type: str) -> object:
        self.uploads.append((name, data, mime_type))
        return [{"path": "/tmp/gradio/reference.wav"}]

    def queue_join(self, fn_index: int, data: Sequence[object]) -> str:
        self.queue_calls.append((fn_index, tuple(data)))
        return "session"

    def wait_for_queue_result(self, session_hash: str, **kwargs: object) -> object:
        assert session_hash == "session"
        assert kwargs["timeout_seconds"] == self.timeout
        if isinstance(self.wait_outcome, BaseException):
            raise self.wait_outcome
        if self.wait_outcome is not None:
            return self.wait_outcome
        fn_index = self.queue_calls[-1][0]
        if fn_index == 7:
            return {
                "data": [
                    {"path": "/tmp/gradio/preview.wav"},
                    [
                        {"path": "/tmp/gradio/one.wav"},
                        {"path": "/tmp/gradio/two.wav"},
                    ],
                    {"path": "/tmp/gradio/audio.zip"},
                ]
            }
        return {"data": [{"path": "/tmp/gradio/single.wav"}]}

    def fetch_file(self, reference: object) -> Tuple[bytes, str]:
        self.fetches.append(reference)
        path = (
            str(reference.get("path", ""))
            if isinstance(reference, Mapping)
            else str(reference)
        )
        marker = path.rsplit("/", 1)[-1].encode("ascii")
        return WAV_AUDIO + marker, "audio/wav"

    def close(self) -> None:
        self.closed = True


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
    for variable in (
        "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URLS",
        "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_URL",
        "WALLET_INDEXTTS_SPACE_URL",
        "WALLET_INDEXTTS_FALLBACK_SPACE_URL",
        "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "IPFS_DATASETS_PY_HF_API_TOKEN",
        "IPFS_ACCELERATE_PY_ABBY_INDEXTTS_TIMEOUT_SECONDS",
        "IPFS_ACCELERATE_PY_ABBY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        abby_module,
        "_cached_huggingface_token",
        lambda: "cached-hub-token",
    )

    tts_capabilities = get_voice_provider_capabilities("indextts")
    stt_capabilities = get_voice_provider_capabilities("abby_hf_whisper")
    assert tts_capabilities.can_synthesize
    assert not tts_capabilities.can_transcribe
    assert stt_capabilities.can_transcribe
    assert not stt_capabilities.can_synthesize

    provider = get_voice_provider("abby_indextts", use_cache=False)
    assert isinstance(provider, IndexTTSHTTPProvider)
    assert isinstance(provider, PublicusIndexTTSProvider)
    assert voice_providers.PublicusIndexTTSProvider is IndexTTSHTTPProvider
    assert provider.endpoints == (PUBLICUS_INDEXTTS_SPACE_URL,)
    assert provider.default_model == PUBLICUS_INDEXTTS_MODEL
    assert provider.backend == "publicus_gradio"
    assert provider.authenticated
    assert provider._authorization_headers()["X-HF-Bill-To"] == "Publicus"
    assert provider.policy.timeout_seconds == 900
    assert dict(provider.gradio_contract)["input_count"] == 25


def test_publicus_group_billing_can_be_explicitly_disabled() -> None:
    provider = IndexTTSHTTPProvider(
        token="private-token",
        bill_to="",
    )

    assert provider._authorization_headers() == {
        "Authorization": "Bearer private-token"
    }


def test_explicit_hf_token_avoids_cached_token_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "explicit-token")

    def forbidden() -> str:
        raise AssertionError("cached Hub token should not be consulted")

    monkeypatch.setattr(abby_module, "_cached_huggingface_token", forbidden)
    provider = IndexTTSHTTPProvider.from_environment()
    assert provider.authenticated


def test_publicus_single_contract_is_authenticated_and_receipt_is_private() -> None:
    clients: List[FakePublicusSpaceClient] = []

    def factory(endpoint, timeout, headers_factory):
        client = FakePublicusSpaceClient(endpoint, timeout, headers_factory)
        clients.append(client)
        return client

    provider = IndexTTSHTTPProvider(
        token="private-hf-token",
        bill_to="publicus",
        reference_audio=b"RIFF-reference-voice",
        voice_description="calm, clear voice",
        policy=AbbyResiliencePolicy(timeout_seconds=13, max_retries=0),
        space_client_factory=factory,
    )

    audio = provider.synthesize("Call 211 for help.")

    assert audio == WAV_AUDIO + b"single.wav"
    assert len(clients) == 1
    client = clients[0]
    assert client.endpoint == PUBLICUS_INDEXTTS_SPACE_URL
    assert client.headers == {
        "Authorization": "Bearer private-hf-token",
        "X-HF-Bill-To": "publicus",
    }
    assert client.resolved_api_names == ["/gen_single"]
    fn_index, data = client.queue_calls[0]
    assert fn_index == 6
    assert len(data) == 25
    assert data[0] == "Same as the voice reference"
    assert data[2] == "Call 211 for help."
    assert data[13] == "calm, clear voice"
    assert data[16] == 0
    assert client.closed
    assert provider.last_receipt is not None
    encoded_receipt = json.dumps(provider.last_receipt.to_dict())
    assert "private-hf-token" not in encoded_receipt
    assert "Call 211 for help." not in encoded_receipt
    assert "reference-voice" not in encoded_receipt


def test_publicus_batch_uses_fn7_and_reuses_upload_across_client_sessions() -> None:
    clients: List[FakePublicusSpaceClient] = []

    def factory(endpoint, timeout, headers_factory):
        client = FakePublicusSpaceClient(endpoint, timeout, headers_factory)
        clients.append(client)
        return client

    provider = IndexTTSHTTPProvider(
        token="batch-token",
        reference_audio=b"RIFF-reference-voice",
        policy=AbbyResiliencePolicy(timeout_seconds=20, max_retries=0),
        space_client_factory=factory,
    )

    outputs = provider.synthesize_batch(["first response", "second response"])
    single = provider.synthesize("third response")

    assert outputs == (
        WAV_AUDIO + b"one.wav",
        WAV_AUDIO + b"two.wav",
    )
    assert single == WAV_AUDIO + b"single.wav"
    assert len(clients) == 2
    batch_client, single_client = clients
    assert len(batch_client.uploads) == 1
    assert single_client.uploads == []
    batch_fn_index, batch_data = batch_client.queue_calls[0]
    assert batch_fn_index == 7
    assert batch_client.resolved_api_names == ["/gen_batch"]
    assert len(batch_data) == 25
    assert json.loads(str(batch_data[2])) == [
        "first response",
        "second response",
    ]
    assert batch_data[16] == 2
    assert batch_client.fetches == [
        {"path": "/tmp/gradio/one.wav"},
        {"path": "/tmp/gradio/two.wav"},
    ]
    assert single_client.resolved_api_names == ["/gen_single"]
    assert all(client.closed for client in clients)


@pytest.mark.parametrize(
    "queue_error",
    [
        TimeoutError("ZeroGPU queue timed out"),
        RuntimeError(
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'/tmp/gradio/expired/reference.wav'"
        ),
    ],
    ids=("timeout", "stale-gradio-upload"),
)
def test_publicus_retry_reuploads_reference_after_queue_failure(
    queue_error: Exception,
) -> None:
    clients: List[FakePublicusSpaceClient] = []
    outcomes = [queue_error, None]

    def factory(endpoint, timeout, headers_factory):
        client = FakePublicusSpaceClient(
            endpoint,
            timeout,
            headers_factory,
            wait_outcome=outcomes[len(clients)],
        )
        clients.append(client)
        return client

    provider = IndexTTSHTTPProvider(
        token="retry-token",
        reference_audio=b"RIFF-reference-voice",
        policy=AbbyResiliencePolicy(
            timeout_seconds=30,
            max_retries=1,
            backoff_seconds=0,
        ),
        space_client_factory=factory,
    )

    assert provider.synthesize("retry me") == WAV_AUDIO + b"single.wav"
    assert len(clients) == 2
    assert [len(client.uploads) for client in clients] == [1, 1]
    assert all(client.closed for client in clients)
    assert provider.last_receipt is not None
    assert provider.last_receipt.status == "degraded"
    assert "retry-token" not in json.dumps(provider.last_receipt.to_dict())


def test_publicus_failure_falls_back_to_compatible_generic_http_endpoint() -> None:
    clients: List[FakePublicusSpaceClient] = []

    def factory(endpoint, timeout, headers_factory):
        client = FakePublicusSpaceClient(
            endpoint,
            timeout,
            headers_factory,
            wait_outcome=RuntimeError("Space unavailable"),
        )
        clients.append(client)
        return client

    generic_transport = RecordingTransport(
        [json_response({"audioBase64": base64.b64encode(WAV_AUDIO).decode()})]
    )
    provider = IndexTTSHTTPProvider(
        [
            PUBLICUS_INDEXTTS_SPACE_URL,
            "https://tts.example.test/generate",
        ],
        token="fallback-token",
        reference_audio=b"RIFF-reference-voice",
        policy=AbbyResiliencePolicy(timeout_seconds=10, max_retries=0),
        space_client_factory=factory,
        transport=generic_transport,
    )

    assert provider.synthesize("fallback response") == WAV_AUDIO
    assert len(clients) == 1
    assert generic_transport.calls[0][0].url == (
        "https://tts.example.test/generate"
    )
    assert provider.last_receipt is not None
    assert provider.last_receipt.status == "degraded"
    assert provider.last_receipt.selected_endpoint == (
        "https://tts.example.test/generate"
    )
    assert "fallback-token" not in json.dumps(provider.last_receipt.to_dict())


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
