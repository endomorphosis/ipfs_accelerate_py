"""Usage-aware admission integration for voice_router (AICAT-033)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import pytest

from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
)
from ipfs_accelerate_py.endpoint_usage.routing import RoutePin
from ipfs_accelerate_py.endpoint_usage.schema import (
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageLimit,
    UsageSnapshot,
    UsageVector,
    WindowKind,
    EndpointUsageScope,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.router_deps import RouterDeps
from ipfs_accelerate_py.voice_router import (
    USAGE_ROUTING_REQUIREMENT_ID,
    VOICE_STT_USAGE_OPERATION,
    VOICE_TTS_USAGE_OPERATION,
    UsageCapacityError,
    apply_voice_stream_settlements,
    clear_voice_router_caches,
    estimate_audio_seconds,
    estimate_synthesis_tokens,
    estimate_synthesis_usage,
    estimate_transcription_usage,
    get_last_usage_admission,
    get_last_voice_usage_trace,
    settle_synthesis_usage,
    settle_transcription_usage,
    speech_to_text,
    text_to_speech,
    voice_fallback_compatible,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@pytest.fixture(autouse=True)
def _isolated_router_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "0")
    clear_voice_router_caches()


class _CountingTTS:
    """Deterministic TTS provider for usage tests."""

    def __init__(
        self,
        name: str = "counting_tts",
        *,
        fail_times: int = 0,
        fail_exc: Optional[BaseException] = None,
        audio: bytes = b"WAVAUDIO",
    ) -> None:
        self.router_provider_name = name
        self.calls: List[str] = []
        self.fail_times = fail_times
        self.fail_exc = fail_exc or RuntimeError("provider_fail")
        self.audio = audio
        self.lock = threading.Lock()

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
        _ = (voice, model_name, device, output_format, kwargs)
        with self.lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_exc
            self.calls.append(text)
        return self.audio + str(len(self.calls)).encode("ascii")

    def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("TTS-only fixture")


class _CountingSTT:
    """Deterministic STT provider for usage tests."""

    def __init__(
        self,
        name: str = "counting_stt",
        *,
        fail_times: int = 0,
        fail_exc: Optional[BaseException] = None,
        transcript: str = "hello world",
    ) -> None:
        self.router_provider_name = name
        self.calls: List[int] = []
        self.fail_times = fail_times
        self.fail_exc = fail_exc or RuntimeError("provider_fail")
        self.transcript = transcript
        self.lock = threading.Lock()

    def synthesize(self, text, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError("STT-only fixture")

    def transcribe(
        self,
        audio: Union[str, bytes],
        *,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: object,
    ) -> str:
        _ = (model_name, language, device, kwargs)
        size = len(audio) if isinstance(audio, (bytes, bytearray)) else len(str(audio))
        with self.lock:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail_exc
            self.calls.append(size)
        return self.transcript


def _scope(provider_key: str, operation: str) -> EndpointUsageScope:
    provider_id = stable_id("provider", "voice", provider_key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=operation,
        deployment_id=stable_id(
            "deployment", provider_id, operation, "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:VOICE_USAGE_TEST_KEY", key_id="voice-usage-default"
        ),
    )


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    used: int = 0,
    window: Optional[LimitWindow] = None,
) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        window=window or LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(max(0, ceiling - used)),
        used=Quantity.finite(used),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _coord(clock: Optional[FakeClock] = None) -> UsageCoordinator:
    clk = clock or FakeClock(_now())
    store = InMemoryUsageLedgerStore(clock=clk)
    return UsageCoordinator(store, writer_id="voice-usage-test", fence=1)


def _configure_tts_limits(
    coord: UsageCoordinator,
    scope: EndpointUsageScope,
    *,
    requests: int = 100,
    characters: int = 100_000,
    input_tokens: int = 100_000,
    concurrent_requests: int = 10,
    concurrent_streams: int = 10,
) -> None:
    sid = scope.scope_id
    limits = [
        _limit(sid, UsageDimension.REQUESTS, requests),
        _limit(sid, UsageDimension.CHARACTERS, characters),
        _limit(sid, UsageDimension.INPUT_TOKENS, input_tokens),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_REQUESTS,
            ceiling=Quantity.finite(concurrent_requests),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_requests),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_STREAMS,
            ceiling=Quantity.finite(concurrent_streams),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_streams),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
    ]
    coord.configure_limits(sid, limits)


def _configure_stt_limits(
    coord: UsageCoordinator,
    scope: EndpointUsageScope,
    *,
    requests: int = 100,
    audio_seconds: int = 10_000,
    concurrent_requests: int = 10,
    concurrent_streams: int = 10,
) -> None:
    sid = scope.scope_id
    limits = [
        _limit(sid, UsageDimension.REQUESTS, requests),
        _limit(sid, UsageDimension.AUDIO_SECONDS, audio_seconds),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_REQUESTS,
            ceiling=Quantity.finite(concurrent_requests),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_requests),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
        UsageLimit(
            scope_id=sid,
            dimension=UsageDimension.CONCURRENT_STREAMS,
            ceiling=Quantity.finite(concurrent_streams),
            window=LimitWindow(kind=WindowKind.CONCURRENT),
            remaining=Quantity.finite(concurrent_streams),
            used=Quantity.finite(0),
            enforcement=LimitEnforcement.HARD,
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
    ]
    coord.configure_limits(sid, limits)


def _candidate(
    *,
    provider_key: str,
    scope: EndpointUsageScope,
    operation: str,
    labels: Optional[Dict[str, str]] = None,
    score: int = 10,
    binding_salt: str = "",
) -> StaticCandidate:
    labels = dict(labels or {})
    labels.setdefault("router_provider", provider_key)
    labels.setdefault("operation", operation)
    labels.setdefault("locality", "remote")
    labels.setdefault("device", "remote")
    if operation == VOICE_TTS_USAGE_OPERATION:
        labels.setdefault("voice", "alloy")
        labels.setdefault("codec", "wav")
        labels.setdefault("output_format", "wav")
    else:
        labels.setdefault("language", "en")
    binding_id = stable_id(
        "binding",
        "voice",
        scope.provider_id,
        provider_key,
        operation,
        binding_salt or provider_key,
    )
    return StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        model_id=stable_id("model", "voice", provider_key),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        locality=labels.get("locality"),
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
        labels=labels,
    )


def _headroom_available(snap: UsageSnapshot, dimension: UsageDimension) -> Optional[int]:
    for item in snap.headroom:
        if item.dimension is dimension and item.available.kind.value == "finite":
            return int(item.available.value or 0)
    return None


# ---------------------------------------------------------------------------
# Estimates / compatibility
# ---------------------------------------------------------------------------


def test_usage_routing_requirement_id_exported() -> None:
    assert USAGE_ROUTING_REQUIREMENT_ID == (
        "requirement:voice-router-usage-routing.v1"
    )
    assert VOICE_TTS_USAGE_OPERATION == "audio.synthesize"
    assert VOICE_STT_USAGE_OPERATION == "audio.transcribe"


def test_estimate_synthesis_usage_covers_modality_dimensions() -> None:
    text = "hello spoken world"
    vector = estimate_synthesis_usage(text, streaming=True)
    assert isinstance(vector, UsageVector)
    assert vector.get(UsageDimension.REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.CHARACTERS).amount.value == len(text)
    assert vector.get(UsageDimension.INPUT_TOKENS).amount.value >= 1
    assert vector.get(UsageDimension.MEDIA_BYTES).amount.value == len(
        text.encode("utf-8")
    )
    assert vector.get(UsageDimension.CONCURRENT_REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.CONCURRENT_STREAMS).amount.value == 1
    assert estimate_synthesis_tokens("abcd") >= 1
    settled = settle_synthesis_usage(text, audio_bytes=b"audio-out")
    assert settled.get(UsageDimension.CHARACTERS).amount.value == len(text)
    assert settled.get(UsageDimension.MEDIA_BYTES).amount.value == len(b"audio-out")


def test_estimate_transcription_usage_covers_modality_dimensions() -> None:
    audio = b"x" * 16_000
    vector = estimate_transcription_usage(
        audio, declared_seconds=2.2, streaming=True
    )
    assert vector.get(UsageDimension.REQUESTS).amount.value == 1
    assert vector.get(UsageDimension.AUDIO_SECONDS).amount.value == 3
    assert vector.get(UsageDimension.MEDIA_BYTES).amount.value == 16_000
    assert vector.get(UsageDimension.CONCURRENT_STREAMS).amount.value == 1
    assert estimate_audio_seconds(audio, declared_seconds=1.1) == 2
    settled = settle_transcription_usage(audio, declared_seconds=2.2)
    assert settled.get(UsageDimension.AUDIO_SECONDS).amount.value == 3


def test_cache_only_estimate_creates_no_remote_envelope() -> None:
    empty = estimate_synthesis_usage("a", remote=False)
    assert empty.entries == ()
    empty_stt = estimate_transcription_usage(b"x", remote=False)
    assert empty_stt.entries == ()


def test_voice_fallback_compatible_preserves_contracts() -> None:
    origin = {
        "operation": VOICE_TTS_USAGE_OPERATION,
        "voice": "alloy",
        "codec": "wav",
        "sample_rate": "24000",
        "channels": "1",
        "locality": "remote",
        "device": "remote",
        "language": "en",
        "data_retention": "none",
    }
    ok = dict(origin)
    assert voice_fallback_compatible(origin, ok) is True
    assert voice_fallback_compatible(origin, dict(origin, voice="echo")) is False
    assert voice_fallback_compatible(origin, dict(origin, codec="mp3")) is False
    assert voice_fallback_compatible(origin, dict(origin, sample_rate="16000")) is False
    assert voice_fallback_compatible(origin, dict(origin, locality="local")) is False
    assert (
        voice_fallback_compatible(
            origin, dict(origin, operation=VOICE_STT_USAGE_OPERATION)
        )
        is False
    )
    assert (
        voice_fallback_compatible(origin, dict(origin, data_retention="store-30d"))
        is False
    )


# ---------------------------------------------------------------------------
# Off / observe / enforce modes
# ---------------------------------------------------------------------------


def test_off_mode_identical_to_legacy_selection() -> None:
    provider = _CountingTTS()
    audio = text_to_speech("hello", provider_instance=provider)  # type: ignore[arg-type]
    assert isinstance(audio, bytes)
    assert len(provider.calls) == 1

    provider2 = _CountingTTS()
    audio2 = text_to_speech(
        "hello",
        provider_instance=provider2,  # type: ignore[arg-type]
        usage_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert audio2 == audio
    admission = get_last_usage_admission()
    assert admission.get("mode") == "off" or admission.get("final_status") in {
        "off",
        None,
        "",
    }


def test_observe_mode_never_changes_selection_or_charges() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("observe", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=0)  # would deny enforce
    provider = _CountingTTS("observe_provider")

    audio = text_to_speech(
        "spoken text",
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.OBSERVE),
        usage_scope_id=scope.scope_id,
        usage_request_id="req-observe-1",
    )
    assert isinstance(audio, bytes)
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["remote_charged"] is False
    assert "no_selection_change" in admission["reason_codes"]
    assert_no_prompt_media_or_output(admission)
    snap = coord.snapshot(scope.scope_id)
    assert _headroom_available(snap, UsageDimension.REQUESTS) == 0


def test_enforce_tts_reserves_before_dispatch_and_settles_once() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("enforce-tts", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5, characters=10_000)
    cand = _candidate(
        provider_key="enforce-tts",
        scope=scope,
        operation=VOICE_TTS_USAGE_OPERATION,
    )
    provider = _CountingTTS("enforce-tts")
    sample = "alpha spoken input"

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)
    before_chars = _headroom_available(before, UsageDimension.CHARACTERS)

    audio = text_to_speech(
        sample,
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE, max_attempts=1
        ),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage(sample),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-tts-1",
        usage_idempotency_key="idem-tts-1",
    )
    assert isinstance(audio, bytes)
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["reservation_id"]
    assert admission["receipt_id"]
    assert_no_prompt_media_or_output(admission)
    if "receipt" in admission:
        assert_no_prompt_media_or_output(admission["receipt"])
        encoded = repr(admission["receipt"]).casefold()
        assert sample.casefold() not in encoded
        assert "wavaudio" not in encoded

    after = coord.snapshot(scope.scope_id)
    after_req = _headroom_available(after, UsageDimension.REQUESTS)
    after_chars = _headroom_available(after, UsageDimension.CHARACTERS)
    assert before_req is not None and after_req is not None
    assert after_req == before_req - 1
    assert before_chars is not None and after_chars is not None
    assert after_chars == before_chars - len(sample)

    # Idempotent replay must not double-charge.
    text_to_speech(
        sample,
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage(sample),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-tts-1",
        usage_idempotency_key="idem-tts-1",
    )
    replay = coord.snapshot(scope.scope_id)
    assert _headroom_available(replay, UsageDimension.REQUESTS) == after_req
    assert _headroom_available(replay, UsageDimension.CHARACTERS) == after_chars


def test_enforce_stt_reserves_audio_seconds() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("enforce-stt", VOICE_STT_USAGE_OPERATION)
    _configure_stt_limits(coord, scope, requests=5, audio_seconds=100)
    cand = _candidate(
        provider_key="enforce-stt",
        scope=scope,
        operation=VOICE_STT_USAGE_OPERATION,
    )
    provider = _CountingSTT("enforce-stt", transcript="transcribed phrase")
    audio = b"\x00" * 8000

    before = coord.snapshot(scope.scope_id)
    before_sec = _headroom_available(before, UsageDimension.AUDIO_SECONDS)

    text = speech_to_text(
        audio,
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE, max_attempts=1
        ),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_transcription_usage(audio, declared_seconds=4),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-stt-1",
        usage_idempotency_key="idem-stt-1",
        usage_audio_seconds=4,
    )
    assert text == "transcribed phrase"
    assert len(provider.calls) == 1
    after = coord.snapshot(scope.scope_id)
    after_sec = _headroom_available(after, UsageDimension.AUDIO_SECONDS)
    assert before_sec is not None and after_sec is not None
    assert after_sec == before_sec - 4
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert_no_prompt_media_or_output(admission)
    # Receipts must not contain transcript content.
    blob = repr(admission)
    assert "transcribed phrase" not in blob


def test_enforce_denies_when_capacity_exhausted() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("deny", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=0)
    cand = _candidate(
        provider_key="deny", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    provider = _CountingTTS("deny")

    with pytest.raises(UsageCapacityError) as excinfo:
        text_to_speech(
            "blocked",
            provider_instance=provider,  # type: ignore[arg-type]
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
            usage_request=UsageRoutingRequest(
                required=estimate_synthesis_usage("blocked"),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-deny-1",
            usage_idempotency_key="idem-deny-1",
        )
    assert provider.calls == []
    assert excinfo.value.reason_codes
    admission = get_last_usage_admission()
    assert admission["success"] is False


# ---------------------------------------------------------------------------
# Cache hits, cancel/timeout, stream, pins, fallback, receipts
# ---------------------------------------------------------------------------


def test_cache_hits_create_no_remote_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    clear_voice_router_caches()

    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cache", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5)
    cand = _candidate(
        provider_key="cache", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    provider = _CountingTTS("cache")
    deps = RouterDeps()

    text_to_speech(
        "cached-speech",
        provider_instance=provider,  # type: ignore[arg-type]
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage("cached-speech"),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-1",
        usage_idempotency_key="idem-cache-1",
    )
    after_first = coord.snapshot(scope.scope_id)
    first_req = _headroom_available(after_first, UsageDimension.REQUESTS)
    assert len(provider.calls) == 1

    audio = text_to_speech(
        "cached-speech",
        provider_instance=provider,  # type: ignore[arg-type]
        deps=deps,
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage("cached-speech"),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-cache-2",
        usage_idempotency_key="idem-cache-2",
    )
    assert isinstance(audio, bytes)
    assert len(provider.calls) == 1
    admission = get_last_usage_admission()
    assert admission.get("remote_charged") is False
    assert "no_remote_charge" in admission.get("reason_codes", [])
    after_second = coord.snapshot(scope.scope_id)
    assert _headroom_available(after_second, UsageDimension.REQUESTS) == first_req


def test_cancel_before_dispatch_does_not_charge() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("cancel", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5)
    cand = _candidate(
        provider_key="cancel", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    provider = _CountingTTS("cancel")
    cancel = threading.Event()
    cancel.set()

    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(UsageCapacityError) as excinfo:
        text_to_speech(
            "never",
            provider_instance=provider,  # type: ignore[arg-type]
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
            usage_request=UsageRoutingRequest(
                required=estimate_synthesis_usage("never"),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-cancel-1",
            usage_idempotency_key="idem-cancel-1",
            usage_cancel_event=cancel,
        )
    assert provider.calls == []
    assert "cancelled_before_dispatch" in excinfo.value.reason_codes or any(
        "cancel" in c for c in excinfo.value.reason_codes
    )
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req


def test_timeout_before_dispatch_does_not_charge() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("timeout", VOICE_STT_USAGE_OPERATION)
    _configure_stt_limits(coord, scope, requests=5)
    cand = _candidate(
        provider_key="timeout", scope=scope, operation=VOICE_STT_USAGE_OPERATION
    )
    provider = _CountingSTT("timeout")
    before = coord.snapshot(scope.scope_id)
    before_req = _headroom_available(before, UsageDimension.REQUESTS)

    with pytest.raises(UsageCapacityError) as excinfo:
        speech_to_text(
            b"audio",
            provider_instance=provider,  # type: ignore[arg-type]
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
            usage_request=UsageRoutingRequest(
                required=estimate_transcription_usage(b"audio", declared_seconds=1),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-timeout-1",
            usage_idempotency_key="idem-timeout-1",
            usage_timeout_seconds=0,
            usage_audio_seconds=1,
        )
    assert provider.calls == []
    assert any("timeout" in c for c in excinfo.value.reason_codes)
    after = coord.snapshot(scope.scope_id)
    assert _headroom_available(after, UsageDimension.REQUESTS) == before_req


def test_streaming_settles_monotonic_partial_usage() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("stream", VOICE_STT_USAGE_OPERATION)
    _configure_stt_limits(coord, scope, requests=5, audio_seconds=100)
    cand = _candidate(
        provider_key="stream", scope=scope, operation=VOICE_STT_USAGE_OPERATION
    )
    provider = _CountingSTT("stream", transcript="partial stream ok")
    audio = b"\x01" * 4000

    partials = [
        UsageVector.of(requests=1, audio_seconds=1),
        UsageVector.of(requests=1, audio_seconds=2),
        UsageVector.of(requests=1, audio_seconds=3),
    ]
    text = speech_to_text(
        audio,
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
        ),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_transcription_usage(audio, declared_seconds=3),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-stream-1",
        usage_idempotency_key="idem-stream-1",
        usage_stream_partials=partials,
        usage_streaming=True,
        usage_audio_seconds=3,
    )
    assert text == "partial stream ok"
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission.get("stream_partials") == 3
    after = coord.snapshot(scope.scope_id)
    # Final commit settles full reserved audio_seconds (3).
    assert _headroom_available(after, UsageDimension.AUDIO_SECONDS) == 97


def test_non_monotonic_stream_partials_fail() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("nonmono", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5, characters=10_000)
    cand = _candidate(
        provider_key="nonmono", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    provider = _CountingTTS("nonmono")

    partials = [
        UsageVector.of(requests=1, characters=10),
        UsageVector.of(requests=1, characters=5),  # decreases — invalid
    ]
    with pytest.raises((UsageCapacityError, Exception)):
        text_to_speech(
            "stream me",
            provider_instance=provider,  # type: ignore[arg-type]
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE
            ),
            usage_candidates=[cand],
            usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
            usage_request=UsageRoutingRequest(
                required=estimate_synthesis_usage("stream me"),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-nonmono-1",
            usage_idempotency_key="idem-nonmono-1",
            usage_stream_partials=partials,
            usage_streaming=True,
        )


def test_explicit_provider_pin_defaults_to_no_fallback() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("pin-a", VOICE_TTS_USAGE_OPERATION)
    scope_b = _scope("pin-b", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope_a, requests=0)
    _configure_tts_limits(coord, scope_b, requests=10)
    labels = {
        "operation": VOICE_TTS_USAGE_OPERATION,
        "voice": "alloy",
        "codec": "wav",
        "locality": "remote",
        "device": "remote",
    }
    cand_a = _candidate(
        provider_key="pin-a",
        scope=scope_a,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=100,
        labels=dict(labels),
    )
    cand_b = _candidate(
        provider_key="pin-b",
        scope=scope_b,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=1,
        labels=dict(labels),
    )
    provider_a = _CountingTTS("pin-a")
    provider_b = _CountingTTS("pin-b")

    with pytest.raises(UsageCapacityError):
        text_to_speech(
            "pinned",
            provider="pin-a",
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE,
                fallback=FallbackClass.CROSS_PROVIDER,
                max_attempts=3,
            ),
            usage_candidates=[cand_a, cand_b],
            usage_provider_by_binding={
                cand_a.binding_id: provider_a,  # type: ignore[dict-item]
                cand_b.binding_id: provider_b,  # type: ignore[dict-item]
            },
            usage_pin=RoutePin(
                provider_id=scope_a.provider_id,
                allow_fallback_with_pin=False,
            ),
            usage_request=UsageRoutingRequest(
                required=estimate_synthesis_usage("pinned"),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-pin-1",
            usage_idempotency_key="idem-pin-1",
        )
    assert provider_b.calls == []


def test_compatible_fallback_advances_on_capacity_error() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("fb-a", VOICE_TTS_USAGE_OPERATION)
    scope_b = _scope("fb-b", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope_a, requests=10)
    _configure_tts_limits(coord, scope_b, requests=10)
    labels = {
        "operation": VOICE_TTS_USAGE_OPERATION,
        "voice": "alloy",
        "codec": "wav",
        "locality": "remote",
        "device": "remote",
        "sample_rate": "24000",
        "channels": "1",
    }
    cand_a = _candidate(
        provider_key="fb-a",
        scope=scope_a,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=50,
        labels=dict(labels),
    )
    cand_b = _candidate(
        provider_key="fb-b",
        scope=scope_b,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=10,
        labels=dict(labels),
    )
    provider_a = _CountingTTS(
        "fb-a",
        fail_times=1,
        fail_exc=RuntimeError("rate limit 429 capacity"),
    )
    provider_b = _CountingTTS("fb-b")

    audio = text_to_speech(
        "fallback-me",
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=3,
        ),
        usage_candidates=[cand_a, cand_b],
        usage_provider_by_binding={
            cand_a.binding_id: provider_a,  # type: ignore[dict-item]
            cand_b.binding_id: provider_b,  # type: ignore[dict-item]
        },
        usage_pin=RoutePin(allow_fallback_with_pin=True),
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage("fallback-me"),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-fb-1",
        usage_idempotency_key="idem-fb-1",
    )
    assert isinstance(audio, bytes)
    assert provider_b.calls  # second candidate used
    admission = get_last_usage_admission()
    assert admission["success"] is True
    assert admission["selected_binding_id"] == cand_b.binding_id


def test_incompatible_voice_never_substitutes() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope_a = _scope("bad-a", VOICE_TTS_USAGE_OPERATION)
    scope_b = _scope("bad-b", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope_a, requests=0)
    _configure_tts_limits(coord, scope_b, requests=10)
    cand_a = _candidate(
        provider_key="bad-a",
        scope=scope_a,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=50,
        labels={
            "operation": VOICE_TTS_USAGE_OPERATION,
            "voice": "alloy",
            "codec": "wav",
            "locality": "remote",
            "device": "remote",
        },
    )
    cand_b = _candidate(
        provider_key="bad-b",
        scope=scope_b,
        operation=VOICE_TTS_USAGE_OPERATION,
        score=10,
        labels={
            "operation": VOICE_TTS_USAGE_OPERATION,
            "voice": "echo",  # incompatible voice
            "codec": "wav",
            "locality": "remote",
            "device": "remote",
        },
    )
    provider_a = _CountingTTS("bad-a")
    provider_b = _CountingTTS("bad-b")

    with pytest.raises(UsageCapacityError):
        text_to_speech(
            "no-sub",
            usage_coordinator=coord,
            usage_policy=RoutingPolicy(
                mode=RoutingMode.ENFORCE,
                fallback=FallbackClass.CROSS_PROVIDER,
                max_attempts=3,
            ),
            usage_candidates=[cand_a, cand_b],
            usage_provider_by_binding={
                cand_a.binding_id: provider_a,  # type: ignore[dict-item]
                cand_b.binding_id: provider_b,  # type: ignore[dict-item]
            },
            usage_request=UsageRoutingRequest(
                required=estimate_synthesis_usage("no-sub"),
                now=_rfc(clock.now()),
                require_snapshot=True,
            ),
            usage_request_id="req-nosub-1",
            usage_idempotency_key="idem-nosub-1",
        )
    assert provider_b.calls == []


def test_provider_metadata_updates_only_exact_scope() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("meta", VOICE_TTS_USAGE_OPERATION)
    other = _scope("meta-other", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5)
    _configure_tts_limits(coord, other, requests=5)
    cand = _candidate(
        provider_key="meta", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    provider = _CountingTTS("meta")

    # Observation claims a different scope — must be ignored (exact-scope only).
    foreign_observation = {
        "scope_id": other.scope_id,
        "http_status": 200,
        "usage": UsageVector.of(requests=1, characters=3).to_dict(),
        "reason_codes": ["foreign_scope"],
    }
    text_to_speech(
        "abc",
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage("abc"),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-meta-1",
        usage_idempotency_key="idem-meta-1",
        usage_observation=foreign_observation,
    )
    # Other scope must remain at full headroom (no cross-scope metadata write).
    other_snap = coord.snapshot(other.scope_id)
    assert _headroom_available(other_snap, UsageDimension.REQUESTS) == 5


def test_receipt_never_contains_transcript_synthesis_or_audio() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("receipt", VOICE_TTS_USAGE_OPERATION)
    _configure_tts_limits(coord, scope, requests=5)
    cand = _candidate(
        provider_key="receipt", scope=scope, operation=VOICE_TTS_USAGE_OPERATION
    )
    secret_text = "super_secret_synthesis_text_xyz"
    provider = _CountingTTS("receipt", audio=b"SECRET_AUDIO_BYTES_PAYLOAD")

    text_to_speech(
        secret_text,
        provider_instance=provider,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[cand],
        usage_provider_by_binding={cand.binding_id: provider},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_synthesis_usage(secret_text),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-receipt-1",
        usage_idempotency_key="idem-receipt-1",
    )
    admission = get_last_usage_admission()
    assert_no_prompt_media_or_output(admission)
    blob = repr(admission)
    assert secret_text not in blob
    assert "SECRET_AUDIO_BYTES_PAYLOAD" not in blob
    assert "sk-" not in blob
    assert "https://" not in blob
    trace = get_last_voice_usage_trace()
    assert secret_text not in repr(trace)

    # STT path
    stt_scope = _scope("receipt-stt", VOICE_STT_USAGE_OPERATION)
    _configure_stt_limits(coord, stt_scope, requests=5)
    stt_cand = _candidate(
        provider_key="receipt-stt",
        scope=stt_scope,
        operation=VOICE_STT_USAGE_OPERATION,
    )
    secret_transcript = "super_secret_transcript_abc"
    stt = _CountingSTT("receipt-stt", transcript=secret_transcript)
    speech_to_text(
        b"\x00\x01\x02secret-audio-marker",
        provider_instance=stt,  # type: ignore[arg-type]
        usage_coordinator=coord,
        usage_policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        usage_candidates=[stt_cand],
        usage_provider_by_binding={stt_cand.binding_id: stt},  # type: ignore[dict-item]
        usage_request=UsageRoutingRequest(
            required=estimate_transcription_usage(
                b"\x00\x01\x02secret-audio-marker", declared_seconds=1
            ),
            now=_rfc(clock.now()),
            require_snapshot=True,
        ),
        usage_request_id="req-receipt-stt-1",
        usage_idempotency_key="idem-receipt-stt-1",
        usage_audio_seconds=1,
    )
    stt_admission = get_last_usage_admission()
    assert_no_prompt_media_or_output(stt_admission)
    stt_blob = repr(stt_admission)
    assert secret_transcript not in stt_blob
    assert "secret-audio-marker" not in stt_blob


def test_apply_voice_stream_settlements_helper() -> None:
    clock = FakeClock(_now())
    coord = _coord(clock)
    scope = _scope("helper", VOICE_STT_USAGE_OPERATION)
    _configure_stt_limits(coord, scope, requests=5, audio_seconds=50)
    decision = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1, audio_seconds=10),
        request_id="helper-req",
        idempotency_key="helper-idem",
        owner_id="voice_router",
        ttl_ms=60_000,
    )
    assert decision.granted
    rid = decision.reservation_id
    assert rid
    results = apply_voice_stream_settlements(
        coord,
        rid,
        [
            UsageVector.of(requests=1, audio_seconds=2),
            UsageVector.of(requests=1, audio_seconds=5),
        ],
    )
    assert len(results) == 2
    coord.commit(rid, UsageVector.of(requests=1, audio_seconds=10))
    snap = coord.snapshot(scope.scope_id)
    assert _headroom_available(snap, UsageDimension.AUDIO_SECONDS) == 40
