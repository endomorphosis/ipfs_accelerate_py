from __future__ import annotations

from ipfs_accelerate_py.voice_router import (
    VoiceStageTrace,
    VoiceTurnProvenance,
    VoiceTurnResult,
    build_voice_cache_miss_event,
)


def _miss_result(request_id: str = "turn-1") -> VoiceTurnResult:
    audio = b"RIFF-cache-miss-WAVE"
    response = "Call five zero three, five five five, zero one zero zero."
    import hashlib

    return VoiceTurnResult(
        request_id=request_id,
        status="degraded",
        transcript="I need help.",
        response_text=response,
        audio=audio,
        audio_format="wav",
        provenance=VoiceTurnProvenance(
            template_provider="fixture-graphrag",
            template_id="template-phone",
            tts_provider="abby_indextts",
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            output_audio_sha256=hashlib.sha256(audio).hexdigest(),
            metadata={"intent": "resource_phone"},
        ),
        traces=(
            VoiceStageTrace(
                "retrieval",
                "succeeded",
                1.0,
                provider="fixture-graphrag",
                details={
                    "template_id": "template-phone",
                    "confidence": 0.99,
                    "evidence_count": 1,
                },
            ),
            VoiceStageTrace(
                "synthesis",
                "skipped",
                0.1,
                provider="precomputed",
                details={
                    "precomputed": False,
                    "resolver_reason": "exact_text_not_found",
                    "runtime_resolution": True,
                    "live_tts_fallback": True,
                    "spoken_text_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "synthesis_identity": {
                        "provider": "abby_indextts",
                        "model": "IndexTTS-2",
                        "voice": "abby",
                    },
                },
            ),
            VoiceStageTrace(
                "synthesis",
                "succeeded",
                20.0,
                provider="abby_indextts",
                details={"audio_size_bytes": len(audio)},
            ),
        ),
        fallback_reasons=("tts_provider_fallback",),
    )


def test_cache_miss_event_is_idempotent_redacted_and_validation_gated() -> None:
    first = build_voice_cache_miss_event(
        _miss_result("turn-1"),
        response_id="response-phone",
        validation_receipt_id="asr-validation-1",
        validation_passed=True,
    )
    second = build_voice_cache_miss_event(
        _miss_result("turn-2"),
        response_id="response-phone",
        validation_receipt_id="asr-validation-1",
        validation_passed=True,
    )

    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert first.request_id != second.request_id
    assert first.ready_for_dag_append is True
    payload = first.to_dict()
    assert "audio" not in payload
    assert b"RIFF" not in repr(payload).encode()
    assert payload["resolver_miss_reason"] == "exact_text_not_found"


def test_hit_or_text_only_turn_does_not_emit_cache_miss_event() -> None:
    result = _miss_result()
    hit = VoiceTurnResult(
        request_id=result.request_id,
        status="completed",
        transcript=result.transcript,
        response_text=result.response_text,
        audio=result.audio,
        audio_format=result.audio_format,
        provenance=result.provenance,
        traces=(
            VoiceStageTrace(
                "synthesis",
                "succeeded",
                1.0,
                provider="precomputed",
                details={"precomputed": True, "resolver_reason": "exact_match"},
            ),
        ),
    )

    assert build_voice_cache_miss_event(hit) is None
