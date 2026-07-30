from __future__ import annotations

import json
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256

import pytest

from ipfs_accelerate_py.voice_response_dag_sink import (
    LocalResponseDAGQueue,
    LocalResponseDAGQueueError,
)
from ipfs_accelerate_py.voice_router import (
    GroundedSlot,
    GroundingEvidence,
    VoiceStageTrace,
    VoiceTurnProvenance,
    VoiceTurnResult,
)

RESPONSE = "Call five zero three, five five five, zero one zero zero."
TEMPLATE = "Call {phone}."
AUDIO = b"RIFF-validated-local-response-dag-WAVE"


def _miss_result(*, surface: str = "website") -> VoiceTurnResult:
    return VoiceTurnResult(
        request_id="private-request-id",
        status="degraded",
        transcript="private caller transcript",
        response_text=RESPONSE,
        audio=AUDIO,
        audio_format="wav",
        provenance=VoiceTurnProvenance(
            template_provider="fixture-graphrag",
            template_id="template-phone",
            tts_provider="abby_indextts",
            evidence=(
                GroundingEvidence(
                    source_id="service-phone",
                    cid="bafy-service-phone",
                ),
            ),
            grounded_slots=(
                GroundedSlot(
                    "phone",
                    "five zero three, five five five, zero one zero zero",
                    ("service-phone",),
                ),
            ),
            response_text_sha256=sha256(RESPONSE.encode()).hexdigest(),
            output_audio_sha256=sha256(AUDIO).hexdigest(),
            metadata={
                "intent": "resource_phone",
                "response_template": TEMPLATE,
                "surface": surface,
            },
        ),
        traces=(
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
                    "spoken_text_sha256": sha256(RESPONSE.encode()).hexdigest(),
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
                details={"audio_size_bytes": len(AUDIO)},
            ),
        ),
        fallback_reasons=("tts_provider_fallback",),
    )


def _audio_descriptor() -> dict[str, object]:
    return {
        "uri": (
            "hf://datasets/Publicus/211-abby-tts/"
            "response_dag/audio/response-phone.wav"
        )
    }


def test_website_miss_is_durable_idempotent_and_private(tmp_path) -> None:
    queue = LocalResponseDAGQueue(tmp_path / "response-dag-queue")
    result = _miss_result()

    first = result.enqueue_validated_cache_miss_candidate(
        sink=queue,
        validation_receipt_id="whisper-pass-001",
        audio_descriptor=_audio_descriptor(),
        response_id="response-phone",
    )
    second = result.enqueue_validated_cache_miss_candidate(
        sink=queue,
        validation_receipt_id="whisper-pass-001",
        audio_descriptor=_audio_descriptor(),
        response_id="response-phone",
    )

    assert first is not None and second is not None
    assert first.receipt.status == "appended"
    assert second.receipt.status == "duplicate"
    assert first.candidate.candidate_id == second.candidate.candidate_id
    assert first.receipt.remote_writes is False
    assert len(queue) == 1

    reopened = LocalResponseDAGQueue(queue.root)
    loaded = reopened.load(first.candidate.candidate_id)
    assert loaded.to_dict() == first.candidate.to_dict()
    assert len(loaded.template_rows) == 1
    assert len(loaded.vocabulary_rows) == 1
    assert loaded.metadata["surface"] == "website"
    assert loaded.vocabulary_rows[0]["source_cids"] == ["bafy-service-phone"]

    queue_file = queue.root / first.receipt.relative_path
    serialized = queue_file.read_text(encoding="utf-8")
    assert "private caller transcript" not in serialized
    assert "private-request-id" not in serialized
    assert AUDIO.hex() not in serialized
    assert json.loads(serialized)["remote_writes"] is False
    assert stat.S_IMODE(queue_file.stat().st_mode) == 0o600


def test_telephone_miss_infers_surface_without_call_or_session_state(tmp_path) -> None:
    result = _miss_result(surface="telephone")
    queued = result.enqueue_validated_cache_miss_candidate(
        sink=LocalResponseDAGQueue(tmp_path / "queue"),
        validation_receipt_id="whisper-pass-telephone-001",
        audio_descriptor=_audio_descriptor(),
    )

    assert queued is not None
    assert queued.candidate.metadata == {"surface": "telephone"}
    serialized = json.dumps(queued.candidate.to_dict(), sort_keys=True)
    assert "call_id" not in serialized
    assert "session_id" not in serialized
    assert result.transcript not in serialized


def test_cache_hit_does_not_create_a_queue_record(tmp_path) -> None:
    result = _miss_result()
    hit = replace(
        result,
        status="completed",
        traces=(
            VoiceStageTrace(
                "synthesis",
                "succeeded",
                1.0,
                provider="precomputed",
                details={"precomputed": True, "resolver_reason": "exact_match"},
            ),
        ),
        fallback_reasons=(),
    )
    queue = LocalResponseDAGQueue(tmp_path / "queue")

    assert (
        hit.enqueue_validated_cache_miss_candidate(
            sink=queue,
            validation_receipt_id="whisper-pass-hit",
            audio_descriptor=_audio_descriptor(),
        )
        is None
    )
    assert len(queue) == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"caller_transcript": "private"},
        {"nested": {"authorization": "Bearer private"}},
        {"debug": "hf_abcdefghijklmnopqrstuvwxyz123456"},
    ],
)
def test_private_or_credential_metadata_fails_before_append(
    tmp_path,
    metadata,
) -> None:
    queue = LocalResponseDAGQueue(tmp_path / "queue")

    with pytest.raises(LocalResponseDAGQueueError):
        _miss_result().enqueue_validated_cache_miss_candidate(
            sink=queue,
            validation_receipt_id="whisper-pass-private-test",
            audio_descriptor=_audio_descriptor(),
            metadata=metadata,
        )

    assert len(queue) == 0


def test_conflicting_existing_candidate_is_never_overwritten(tmp_path) -> None:
    queue = LocalResponseDAGQueue(tmp_path / "queue")
    result = _miss_result()
    queued = result.enqueue_validated_cache_miss_candidate(
        sink=queue,
        validation_receipt_id="whisper-pass-conflict",
        audio_descriptor=_audio_descriptor(),
    )
    assert queued is not None
    target = queue.root / queued.receipt.relative_path
    target.write_bytes(b'{"tampered":true}\n')
    target.chmod(0o600)

    with pytest.raises(LocalResponseDAGQueueError, match="conflicting bytes"):
        result.enqueue_validated_cache_miss_candidate(
            sink=queue,
            validation_receipt_id="whisper-pass-conflict",
            audio_descriptor=_audio_descriptor(),
        )

    assert target.read_bytes() == b'{"tampered":true}\n'


def test_concurrent_duplicate_appends_publish_one_immutable_record(tmp_path) -> None:
    queue = LocalResponseDAGQueue(tmp_path / "queue")
    result = _miss_result()

    def append_once(_index: int):
        queued = result.enqueue_validated_cache_miss_candidate(
            sink=queue,
            validation_receipt_id="whisper-pass-concurrent",
            audio_descriptor=_audio_descriptor(),
        )
        assert queued is not None
        return queued.receipt

    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = tuple(executor.map(append_once, range(24)))

    assert sum(receipt.appended for receipt in receipts) == 1
    assert {receipt.payload_sha256 for receipt in receipts} == {
        receipts[0].payload_sha256
    }
    assert len(queue) == 1
    assert not tuple((queue.root / ".staging").iterdir())


def test_queue_rejects_non_private_root_and_remote_path(tmp_path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(LocalResponseDAGQueueError, match="private"):
        LocalResponseDAGQueue(shared)
    with pytest.raises(LocalResponseDAGQueueError, match="local filesystem"):
        LocalResponseDAGQueue("hf://datasets/Publicus/211-abby-tts")
