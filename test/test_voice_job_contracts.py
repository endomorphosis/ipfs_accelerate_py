from __future__ import annotations

import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.voice_jobs.contracts import (
    ArtifactDescriptor,
    VoiceASRJob,
    VoiceAudioValidationJob,
    VoiceJobContractError,
    VoiceJobError,
    VoiceJobLineage,
    VoiceJobResult,
    VoiceTTSJob,
    canonical_task_type,
    voice_job_from_payload,
)

_AUDIO_SHA = "a" * 64


def _lineage(**changes):
    values = {
        "workset_id": "abby-voice-workset:sha256:" + "1" * 64,
        "manifest_id": "abby-voice-work-manifest:sha256:" + "2" * 64,
        "source_manifest_id": "abby-voice-source:sha256:" + "3" * 64,
        "work_item_id": "abby-voice-work:sha256:" + "4" * 64,
        "subject_id": "response:sha256:" + "5" * 64,
        "subject_schema_version": "abby_voice_response_v2",
        "policy_id": "policy:sha256:" + "6" * 64,
    }
    values.update(changes)
    return VoiceJobLineage(**values)


def _artifact(**changes):
    values = {
        "uri": "ipfs://bafybeigdyrzt/audio.wav",
        "cid": "bafybeigdyrzt",
        "sha256": _AUDIO_SHA,
        "size_bytes": 4096,
        "media_type": "audio/wav",
    }
    values.update(changes)
    return ArtifactDescriptor(**values)


def _tts(**changes):
    values = {
        "spoken_text": "The canonical Abby response.",
        "locale": "en-US",
        "provider": "index-tts",
        "model_name": "IndexTeam/IndexTTS-2",
        "voice": "abby-v2",
        "provider_version": "index-tts-2.0",
        "lineage": _lineage(),
        "codec": "wav",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "generation_settings": {"temperature_milli": 750, "seed": 17},
    }
    values.update(changes)
    return VoiceTTSJob(**values)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    (
        ("tts", "voice.tts"),
        ("voice.tts", "voice.tts"),
        ("stt", "voice.asr"),
        ("speech-to-text", "voice.asr"),
        ("automatic-speech-recognition", "voice.asr"),
        ("voice.asr", "voice.asr"),
        ("audio_validation", "voice.audio-validate"),
    ),
)
def test_voice_task_types_normalize_to_canonical_names(alias, canonical):
    assert canonical_task_type(alias) == canonical


def test_tts_identity_is_full_hash_deterministic_and_covers_output_inputs():
    first = _tts()
    replay = _tts(generation_settings={"seed": 17, "temperature_milli": 750})

    assert first.task_id == replay.task_id
    assert len(first.task_id) == 64
    int(first.task_id, 16)
    assert first.to_payload() == replay.to_payload()

    variants = (
        _tts(spoken_text="The canonical Abby response!"),
        _tts(locale="en-GB"),
        _tts(provider="other-tts"),
        _tts(model_name="IndexTeam/IndexTTS-3"),
        _tts(voice="abby-v3"),
        _tts(provider_version="index-tts-2.1"),
        _tts(codec="flac"),
        _tts(sample_rate_hz=48_000),
        _tts(channels=2),
        _tts(generation_settings={"temperature_milli": 751, "seed": 17}),
        _tts(reference_audio=_artifact()),
    )
    assert all(variant.task_id != first.task_id for variant in variants)
    assert first.to_dict()["spoken_text_sha256"] == hashlib.sha256(
        first.spoken_text.encode("utf-8")
    ).hexdigest()


def test_asr_and_validation_support_immutable_artifacts_or_upstream_tasks():
    tts = _tts()
    dependent_lineage = _lineage(depends_on_task_ids=(tts.task_id,))

    asr = VoiceASRJob(
        provider="whisper",
        model_name="openai/whisper-large-v3",
        provider_version="transformers-4",
        source_task_id=tts.task_id,
        lineage=dependent_lineage,
        decoding_settings={"beam_size": 5},
    )
    validation = VoiceAudioValidationJob(
        model_name="abby-audio-quality-v1",
        source_audio=_artifact(),
        lineage=_lineage(),
        validation_policy={"max_clipping_ppm": 10, "minimum_duration_ms": 100},
    )

    assert voice_job_from_payload(asr.to_payload()) == asr
    assert voice_job_from_payload(validation.to_payload()) == validation
    assert asr.to_payload()["source_audio"] is None
    assert validation.to_payload()["source_audio"]["sha256"] == _AUDIO_SHA

    with pytest.raises(VoiceJobContractError, match="exactly one"):
        VoiceASRJob(
            provider="whisper",
            model_name="whisper",
            provider_version="1",
            lineage=dependent_lineage,
            source_audio=_artifact(),
            source_task_id=tts.task_id,
        )
    with pytest.raises(VoiceJobContractError, match="depends_on_task_ids"):
        VoiceAudioValidationJob(
            model_name="quality-v1",
            lineage=_lineage(),
            source_task_id=tts.task_id,
        )


def test_lineage_propagation_survives_request_transport_and_result_receipt():
    """Evidence: lineage propagation retains every G013 identity end to end."""

    upstream = _tts()
    lineage = _lineage(
        manifest_id="abby-voice-asr-manifest:sha256:" + "7" * 64,
        work_item_id="abby-voice-asr-work:sha256:" + "8" * 64,
        depends_on_task_ids=(upstream.task_id,),
        publication_id="abby-voice-release:sha256:" + "9" * 64,
    )
    request = VoiceASRJob(
        provider="whisper",
        model_name="openai/whisper-large-v3",
        provider_version="2026-07",
        source_task_id=upstream.task_id,
        lineage=lineage,
        retention_policy="publication",
    )

    payload = json.loads(json.dumps(request.to_payload(), sort_keys=True))
    restored_request = voice_job_from_payload(payload)
    result = VoiceJobResult.from_job(
        restored_request,
        artifacts=(_artifact(uri="ipfs://bafybeigdyrzt/transcript.json", media_type="application/json"),),
        quality_metrics={"word_error_rate_ppm": 12_500},
        provider_receipt={
            "provider": "whisper",
            "model": "openai/whisper-large-v3",
            "latency_ms": 125,
        },
    )
    restored_result = VoiceJobResult.from_payload(
        json.loads(json.dumps(result.to_payload(), sort_keys=True))
    )

    assert payload["lineage"] == payload["_lineage"]
    assert payload["lineage"]["task_id"] == request.task_id
    assert restored_request.lineage == request.lineage.with_task_id(request.task_id)
    assert restored_result.lineage == restored_request.lineage
    assert restored_result.lineage.identity_dict() == lineage.identity_dict()
    assert restored_result.task_id == request.task_id


def test_payload_rejects_tampered_task_or_lineage_identity():
    payload = _tts().to_payload()
    bad_task = copy.deepcopy(payload)
    bad_task["task_id"] = "f" * 64
    with pytest.raises(VoiceJobContractError, match="task_id"):
        voice_job_from_payload(bad_task)

    bad_lineage = copy.deepcopy(payload)
    bad_lineage["_lineage"]["policy_id"] = "other-policy"
    with pytest.raises(VoiceJobContractError, match="disagree"):
        voice_job_from_payload(bad_lineage)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("audio_base64", "UklGRg=="),
        ("api_key", "private"),
        ("local_path", "/tmp/audio.wav"),
        ("spoken_text_sha256", "f" * 64),
    ),
)
def test_payload_rejects_unknown_or_redundant_tampered_fields(field, value):
    payload = _tts().to_payload()
    payload[field] = value
    with pytest.raises(VoiceJobContractError, match="canonical contract"):
        voice_job_from_payload(payload)


def test_payload_requires_schema_and_both_matching_lineage_objects():
    payload = _tts().to_payload()
    payload.pop("schema_version")
    with pytest.raises(VoiceJobContractError, match="schema_version"):
        voice_job_from_payload(payload)

    payload = _tts().to_payload()
    payload["_lineage"] = "tampered"
    with pytest.raises(VoiceJobContractError, match="lineage and _lineage"):
        voice_job_from_payload(payload)


def test_runtime_stt_rejects_publication_lineage_and_retention():
    with pytest.raises(VoiceJobContractError, match="publication lineage"):
        VoiceASRJob(
            provider="browser",
            model_name="speech-recognition",
            provider_version="1",
            source_audio=_artifact(),
            purpose="runtime_stt",
            lineage=_lineage(publication_id="release:sha256:" + "f" * 64),
        )
    with pytest.raises(VoiceJobContractError, match="non-retained"):
        VoiceASRJob(
            provider="browser",
            model_name="speech-recognition",
            provider_version="1",
            source_audio=_artifact(),
            purpose="runtime_stt",
            retention_policy="result",
            lineage=_lineage(),
        )


@pytest.mark.parametrize(
    "uri",
    (
        "/tmp/audio.wav",
        "file:///tmp/audio.wav",
        "data:audio/wav;base64,UklGRg==",
        "https://user:password@example.test/audio.wav",
        "https://example.test/audio.wav?access_token=secret",
        "https://example.test/../private/audio.wav",
    ),
)
def test_artifact_descriptors_reject_raw_local_or_credentialed_audio(uri):
    with pytest.raises(VoiceJobContractError, match="uri"):
        _artifact(uri=uri)


def test_contract_payloads_reject_audio_bytes_secrets_and_non_integer_metrics():
    with pytest.raises(VoiceJobContractError, match="inline audio"):
        _tts(generation_settings={"audio_base64": "UklGRg=="})
    with pytest.raises(VoiceJobContractError, match="credentials"):
        _tts(generation_settings={"api_key": "private"})
    assert _tts(
        generation_settings={"max_tokens": 128, "suppress_tokens": [1, 2]}
    ).generation_settings["suppress_tokens"] == (1, 2)

    job = _tts()
    with pytest.raises(VoiceJobContractError, match="integers"):
        VoiceJobResult.from_job(job, quality_metrics={"snr_millidb": 12.5})
    with pytest.raises(VoiceJobContractError, match="credentials"):
        VoiceJobResult.from_job(job, provider_receipt={"access_token": "private"})


@pytest.mark.parametrize(
    "unsafe_value",
    (
        "data:audio/wav;base64,UklGRg==",
        "file:///tmp/audio.wav",
        "/tmp/audio.wav",
        "UklGRklGRg==",
    ),
)
def test_settings_reject_inline_audio_and_local_paths_under_innocuous_keys(
    unsafe_value,
):
    with pytest.raises(VoiceJobContractError, match="inline audio or local paths"):
        _tts(generation_settings={"input": unsafe_value})


@pytest.mark.parametrize(
    "unsafe_value",
    (
        "Bearer private-token",
        "https://user:password@example.invalid/receipt",
        "https://example.invalid/receipt?api_key=private",
    ),
)
def test_provider_receipts_reject_credentials_hidden_in_values(unsafe_value):
    with pytest.raises(VoiceJobContractError, match="credentials"):
        VoiceJobResult.from_job(
            _tts(),
            provider_receipt={"backend": unsafe_value},
        )


def test_failed_results_require_typed_retryability_and_round_trip():
    job = _tts()
    result = VoiceJobResult.from_job(
        job,
        status="failed",
        error=VoiceJobError(code="provider_timeout", retryable=True),
    )
    restored = VoiceJobResult.from_payload(result.to_payload())

    assert restored == result
    assert restored.error is not None
    assert restored.error.retryable is True
    assert restored.error.terminal is False

    with pytest.raises(VoiceJobContractError, match="must be empty"):
        VoiceJobError(
            code="provider_error",
            retryable=False,
            message="Bearer private-token; private transcript text",
        )
    with pytest.raises(VoiceJobContractError, match="machine identifier"):
        VoiceJobError(
            code="private transcript text",
            retryable=False,
        )


@pytest.mark.parametrize(
    "uri",
    (
        "https:///tmp/audio.wav",
        "sqlite:///tmp/audio.wav",
        "javascript:alert(1)",
    ),
)
def test_artifact_descriptors_require_supported_external_uri_authority(uri):
    with pytest.raises(VoiceJobContractError, match="external artifact"):
        _artifact(uri=uri)


def test_contract_rejects_mp4_audio_base64_and_unstructured_provider_receipts():
    mp4_audio = "AAAAHGZ0eXBtcDQyAAAAAG1wNDJtcDQxaXNvbQ=="
    with pytest.raises(VoiceJobContractError, match="inline audio"):
        _tts(generation_settings={"input": mp4_audio})
    with pytest.raises(VoiceJobContractError, match="unsupported provider_receipt"):
        VoiceJobResult.from_job(
            _tts(),
            provider_receipt={"details": "private transcript text"},
        )
