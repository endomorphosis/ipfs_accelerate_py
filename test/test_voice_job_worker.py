"""Offline acceptance tests for durable Abby voice-job execution.

These tests intentionally inject every provider, fetcher, and backend.  A test
failure must never be hidden by a model download or a network request.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import json
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
from ipfs_accelerate_py.p2p_tasks.task_types import (
    VOICE_TASK_TYPES,
    canonical_task_type,
    normalize_task_types,
)
from ipfs_accelerate_py.p2p_tasks.worker import (
    _compute_supported_task_types,
    _mesh_safe_task_types,
    run_worker,
)
from ipfs_accelerate_py.voice_jobs.contracts import (
    ArtifactDescriptor,
    VoiceASRJob,
    VoiceAudioValidationJob,
    VoiceJobLineage,
    VoiceJobResult,
    VoiceTTSJob,
)
from ipfs_accelerate_py.voice_jobs.executor import (
    ArtifactPolicy,
    ArtifactResolver,
    VoiceJobExecutionError,
    execute_task,
    execute_voice_asr_job,
    execute_voice_audio_validation_job,
    execute_voice_tts_job,
)


def _wav_bytes(
    *,
    frames: int = 800,
    sample_rate: int = 8_000,
    channels: int = 1,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames * channels)
    return output.getvalue()


def _pcm16_wav_bytes(
    samples: tuple[int, ...],
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
) -> bytes:
    output = io.BytesIO()
    pcm = b"".join(
        int(sample).to_bytes(2, byteorder="little", signed=True)
        for sample in samples
    )
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _write_audio_descriptor(root: Path, name: str, data: bytes) -> dict[str, object]:
    path = root / name
    path.write_bytes(data)
    return {
        "uri": path.resolve().as_uri(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "media_type": "audio/wav",
    }


def _lineage(
    *,
    depends_on_task_ids: tuple[str, ...] = (),
    publication_id: str = "",
) -> VoiceJobLineage:
    return VoiceJobLineage(
        workset_id="abby-voice-workset:sha256:" + "1" * 64,
        manifest_id="abby-voice-work-manifest:sha256:" + "2" * 64,
        source_manifest_id="abby-voice-source:sha256:" + "3" * 64,
        work_item_id="abby-voice-work:sha256:" + "4" * 64,
        subject_id="response:sha256:" + "5" * 64,
        subject_schema_version="abby_voice_response_v2",
        policy_id="policy:sha256:" + "6" * 64,
        depends_on_task_ids=depends_on_task_ids,
        publication_id=publication_id,
    )


def _external_audio_descriptor(
    data: bytes,
    *,
    name: str = "audio.wav",
    media_type: str = "audio/wav",
) -> ArtifactDescriptor:
    digest = hashlib.sha256(data).hexdigest()
    cid = "bafybeigdyrztexternalfixture"
    return ArtifactDescriptor(
        uri=f"ipfs://{cid}/{name}",
        cid=cid,
        sha256=digest,
        size_bytes=len(data),
        media_type=media_type,
    )


def _resolver(
    tmp_path: Path,
    *,
    input_root: Path | None = None,
    max_input_bytes: int = 1_000_000,
    max_decoded_bytes: int = 1_000_000,
    max_duration_ms: int = 60_000,
    allowed_schemes: frozenset[str] = frozenset({"artifact", "file", "ipfs"}),
    fetcher: Any = None,
    source_task_resolver: Any = None,
) -> ArtifactResolver:
    roots = (input_root,) if input_root is not None else ()
    return ArtifactResolver(
        ArtifactPolicy(
            output_root=tmp_path / "artifacts",
            allowed_file_roots=roots,
            allowed_schemes=allowed_schemes,
            max_input_bytes=max_input_bytes,
            max_decoded_bytes=max_decoded_bytes,
            max_duration_ms=max_duration_ms,
        ),
        fetcher=fetcher,
        source_task_resolver=source_task_resolver,
    )


def test_tts_execution_persists_rehashed_audio_without_queue_bytes(
    tmp_path: Path,
) -> None:
    audio = _wav_bytes()
    calls: list[dict[str, object]] = []

    def synthesize(text: str, **kwargs: object) -> bytes:
        calls.append({"text": text, **kwargs})
        return audio

    resolver = _resolver(tmp_path)
    job = VoiceTTSJob(
        spoken_text="A safe offline response.",
        locale="en-US",
        provider="fixture-tts",
        model_name="fixture-model",
        voice="abby",
        provider_version="fixture-1",
        lineage=_lineage(),
        codec="wav",
        sample_rate_hz=8_000,
        channels=1,
        generation_settings={"temperature": 0},
    )
    result = execute_voice_tts_job(
        job,
        resolver=resolver,
        text_to_speech_fn=synthesize,
        clock=iter((1.0, 1.025)).__next__,
    )

    assert calls == [
        {
            "text": "A safe offline response.",
            "voice": "abby",
            "model_name": "fixture-model",
            "device": None,
            "output_format": "wav",
            "provider": "fixture-tts",
            "temperature": 0,
        }
    ]
    assert result["status"] == "completed"
    assert result["task_type"] == "voice.tts"
    assert set(result) == {
        "artifacts",
        "error",
        "lineage",
        "provider_receipt",
        "quality_metrics",
        "schema_version",
        "status",
        "task_id",
        "task_type",
    }
    assert result["provider_receipt"] == {
        "provider": "fixture-tts",
        "model": "fixture-model",
        "provider_version": "fixture-1",
        "latency_ms": 25,
        "attempt_count": 1,
    }
    assert result["lineage"] == job.lineage.to_dict()
    assert result["error"] is None
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert artifact["cid"]
    assert artifact["uri"] == f"ipfs://{artifact['cid']}"
    assert artifact["sha256"] == hashlib.sha256(audio).hexdigest()
    assert artifact["size_bytes"] == len(audio)
    assert resolver.resolve(artifact) == audio
    assert result["quality_metrics"]["duration_ms"] == 100
    assert VoiceJobResult.from_payload(result).to_payload() == result
    serialized = json.dumps(result, sort_keys=True)
    assert "A safe offline response." not in serialized
    assert base64.b64encode(audio).decode("ascii") not in serialized
    assert str(tmp_path) not in serialized


def test_asr_execution_verifies_source_task_and_keeps_transcript_private(
    tmp_path: Path,
) -> None:
    audio = _wav_bytes(frames=400)
    descriptor = _external_audio_descriptor(audio, name="caller.wav")
    source_task_id = "a" * 64
    source_calls: list[str] = []
    provider_calls: list[tuple[bytes, dict[str, object]]] = []

    def source_task(task_id: str) -> dict[str, object]:
        source_calls.append(task_id)
        return {"artifacts": [descriptor.to_dict()]}

    def transcribe(data: bytes, **kwargs: object) -> str:
        provider_calls.append((data, kwargs))
        return "private offline transcript"

    resolver = _resolver(
        tmp_path,
        fetcher=lambda uri, limit: audio,
        source_task_resolver=source_task,
    )
    job = VoiceASRJob(
        provider="fixture-asr",
        model_name="fixture-whisper",
        provider_version="fixture-1",
        lineage=_lineage(depends_on_task_ids=(source_task_id,)),
        source_task_id=source_task_id,
        purpose="dataset_asr_validation",
        locale="en",
        decoding_settings={"beam_size": 1},
        retention_policy="result",
    )
    result = execute_voice_asr_job(
        job,
        resolver=resolver,
        speech_to_text_fn=transcribe,
        clock=iter((3.0, 3.009)).__next__,
    )

    assert source_calls == [source_task_id]
    assert provider_calls == [
        (
            audio,
            {
                "model_name": "fixture-whisper",
                "language": "en",
                "device": None,
                "provider": "fixture-asr",
                "beam_size": 1,
            },
        )
    ]
    assert result["task_type"] == "voice.asr"
    assert result["provider_receipt"]["latency_ms"] == 9
    assert result["quality_metrics"]["transcript_bytes"] == len(
        b"private offline transcript"
    )
    assert len(result["artifacts"]) == 1
    transcript_artifact = result["artifacts"][0]
    assert transcript_artifact["sha256"] == hashlib.sha256(
        b"private offline transcript"
    ).hexdigest()
    assert transcript_artifact["media_type"] == "text/plain;charset=utf-8"
    assert resolver.resolve(transcript_artifact) == b"private offline transcript"
    assert VoiceJobResult.from_payload(result).to_payload() == result
    serialized = json.dumps(result, sort_keys=True)
    assert "private offline transcript" not in serialized
    assert str(tmp_path) not in serialized

    non_retained_job = VoiceASRJob(
        provider="fixture-asr",
        model_name="fixture-whisper",
        provider_version="fixture-1",
        lineage=_lineage(),
        source_audio=descriptor,
        purpose="dataset_asr_validation",
        locale="en",
        retention_policy="none",
    )
    non_retained_result = execute_voice_asr_job(
        non_retained_job,
        resolver=resolver,
        speech_to_text_fn=lambda data, **kwargs: "non-retained transcript",
    )
    assert non_retained_result["artifacts"] == []
    assert non_retained_result["provider_receipt"]["response_id_sha256"] == (
        hashlib.sha256(b"non-retained transcript").hexdigest()
    )
    assert "non-retained transcript" not in json.dumps(non_retained_result)

    runtime_job = VoiceASRJob(
        provider="fixture-asr",
        model_name="fixture-whisper",
        provider_version="fixture-1",
        lineage=_lineage(),
        source_audio=descriptor,
        purpose="runtime_stt",
        locale="en",
    )
    runtime_result = execute_task(
        {
            "task_id": runtime_job.task_id,
            "task_type": runtime_job.task_type,
            "model_name": runtime_job.model_name,
            "payload": runtime_job.to_payload(),
        },
        resolver=resolver,
        speech_to_text_fn=lambda data, **kwargs: "ephemeral transcript",
    )
    assert runtime_result["task_type"] == "voice.asr"
    assert runtime_result["artifacts"] == []
    assert "transcript_sha256" not in runtime_result
    assert runtime_result["provider_receipt"]["response_id_sha256"] == hashlib.sha256(
        b"ephemeral transcript"
    ).hexdigest()
    assert VoiceJobResult.from_payload(runtime_result).to_payload() == runtime_result
    assert "ephemeral transcript" not in json.dumps(runtime_result)


def test_executor_rejects_noncanonical_legacy_job_mapping(
    tmp_path: Path,
) -> None:
    provider_called = False

    def synthesize(text: str, **kwargs: object) -> bytes:
        nonlocal provider_called
        provider_called = True
        return _wav_bytes()

    with pytest.raises(
        VoiceJobExecutionError,
        match="^voice_job_contract_invalid$",
    ):
        execute_voice_tts_job(
            {
                "task_id": "legacy-task-id",
                "task_type": "tts",
                "spoken_text": "This mapping lacks canonical lineage.",
            },
            resolver=_resolver(tmp_path),
            text_to_speech_fn=synthesize,
        )
    assert provider_called is False


def test_audio_validation_decodes_wav_and_enforces_job_duration_policy(
    tmp_path: Path,
) -> None:
    audio = _wav_bytes(frames=2_000, sample_rate=8_000, channels=2)
    descriptor = _external_audio_descriptor(audio, name="validation.wav")
    resolver = _resolver(tmp_path, fetcher=lambda uri, limit: audio)

    job = VoiceAudioValidationJob(
        model_name="fixture-quality",
        lineage=_lineage(),
        source_audio=descriptor,
        validation_policy={
            "minimum_duration_ms": 200,
            "maximum_duration_ms": 300,
        },
    )
    result = execute_voice_audio_validation_job(
        job,
        resolver=resolver,
    )
    assert result["task_type"] == "voice.audio-validate"
    assert result["artifacts"] == [descriptor.to_dict()]
    metrics = result["quality_metrics"]
    assert metrics["channels"] == 2
    assert metrics["sample_rate_hz"] == 8_000
    assert metrics["frames"] == 2_000
    assert metrics["duration_ms"] == 250
    assert metrics["decoded_bytes"] == 8_000
    # WAV paths must emit acoustic ratios (silent fixture => 100% silence).
    assert metrics["silence_ratio_bp"] == 10_000
    assert metrics["clipping_ratio_bp"] == 0
    assert metrics["trailing_silence_ms"] == 250
    assert VoiceJobResult.from_payload(result).to_payload() == result

    with pytest.raises(
        VoiceJobExecutionError, match="^audio_duration_below_policy$"
    ):
        execute_voice_audio_validation_job(
            VoiceAudioValidationJob(
                model_name="fixture-quality",
                lineage=_lineage(),
                source_audio=descriptor,
                validation_policy={"minimum_duration_ms": 251},
            ),
            resolver=resolver,
        )


@pytest.mark.parametrize(
    ("media_type", "name", "expected_format"),
    [
        ("audio/mpeg", "legacy.mp3", "mp3"),
        ("audio/ogg", "legacy.ogg", "ogg"),
        ("audio/flac", "legacy.flac", "flac"),
    ],
)
def test_audio_validation_decodes_non_wav_and_emits_acoustic_metrics(
    tmp_path: Path,
    media_type: str,
    name: str,
    expected_format: str,
) -> None:
    encoded = f"bounded-{expected_format}-fixture".encode("ascii")
    decoded = _pcm16_wav_bytes((0, 32_767, 1_000, -1_000))
    descriptor = _external_audio_descriptor(
        encoded,
        name=name,
        media_type=media_type,
    )
    resolver = _resolver(tmp_path, fetcher=lambda uri, limit: encoded)
    calls: list[tuple[bytes, str, ArtifactPolicy]] = []

    def decode(
        data: bytes,
        input_format: str,
        policy: ArtifactPolicy,
    ) -> bytes:
        calls.append((data, input_format, policy))
        return decoded

    job = VoiceAudioValidationJob(
        model_name="fixture-quality",
        lineage=_lineage(),
        source_audio=descriptor,
    )
    result = execute_voice_audio_validation_job(
        job,
        resolver=resolver,
        audio_decoder_fn=decode,
    )

    assert calls == [(encoded, expected_format, resolver.policy)]
    assert result["quality_metrics"] == {
        "encoded_bytes": len(encoded),
        "channels": 1,
        "sample_rate_hz": 8_000,
        "frames": 4,
        "duration_ms": 1,
        "decoded_bytes": 8,
        "silence_ratio_bp": 2_500,
        "clipping_ratio_bp": 2_500,
        "trailing_silence_ms": 0,
    }
    assert VoiceJobResult.from_payload(result).to_payload() == result


@pytest.mark.parametrize(
    ("samples", "channels", "expected_trailing_silence_ms"),
    [
        ((1_000, 0, 0), 1, 2),
        ((0, 0, 1_000), 1, 0),
        ((0, 0, 0, 1_000, 0, 0, 0, 0), 2, 2),
    ],
    ids=("nonzero-suffix", "non-silent-ending", "multichannel-frame"),
)
def test_audio_validation_emits_contiguous_trailing_silence(
    tmp_path: Path,
    samples: tuple[int, ...],
    channels: int,
    expected_trailing_silence_ms: int,
) -> None:
    audio = _pcm16_wav_bytes(
        samples,
        sample_rate=1_000,
        channels=channels,
    )
    descriptor = _external_audio_descriptor(audio, name="trailing-silence.wav")
    resolver = _resolver(tmp_path, fetcher=lambda uri, limit: audio)

    result = execute_voice_audio_validation_job(
        VoiceAudioValidationJob(
            model_name="fixture-quality",
            lineage=_lineage(),
            source_audio=descriptor,
        ),
        resolver=resolver,
    )

    assert (
        result["quality_metrics"]["trailing_silence_ms"]
        == expected_trailing_silence_ms
    )


def test_non_wav_ffmpeg_decoder_is_shell_free_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    encoded = b"fake-mp3"
    decoded = _pcm16_wav_bytes((0, 1_000, -1_000))
    descriptor = _external_audio_descriptor(
        encoded,
        name="legacy.mp3",
        media_type="audio/mpeg",
    )
    resolver = _resolver(
        tmp_path,
        fetcher=lambda uri, limit: encoded,
        max_decoded_bytes=1_000,
        max_duration_ms=2_000,
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(decoded)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "ipfs_accelerate_py.voice_jobs.executor.subprocess.run",
        run,
    )
    result = execute_voice_audio_validation_job(
        VoiceAudioValidationJob(
            model_name="fixture-quality",
            lineage=_lineage(),
            source_audio=descriptor,
        ),
        resolver=resolver,
    )

    assert result["quality_metrics"]["decoded_bytes"] == 6
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "ffmpeg"
    assert "-nostdin" in command
    assert command[command.index("-f") + 1] == "mp3"
    assert command[command.index("-fs") + 1] == str(1_000 + 64 * 1024)
    assert command[command.index("-t") + 1] == "3.000"
    assert "shell" not in kwargs
    assert kwargs["input"] == encoded
    assert kwargs["timeout"] == resolver.policy.decoder_timeout_seconds
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing", "audio_decoder_unavailable"),
        ("timeout", "audio_decode_timeout"),
        ("failed", "audio_decode_failed"),
    ],
)
def test_non_wav_decoder_failures_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    encoded = b"untrusted-mp3"
    descriptor = _external_audio_descriptor(
        encoded,
        name="untrusted.mp3",
        media_type="audio/mpeg",
    )
    resolver = _resolver(tmp_path, fetcher=lambda uri, limit: encoded)
    if case == "missing":
        failure: Any = FileNotFoundError("ffmpeg")
    elif case == "timeout":
        failure = subprocess.TimeoutExpired(cmd=("ffmpeg",), timeout=1)
    else:
        failure = None

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if failure is not None:
            raise failure
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(
        "ipfs_accelerate_py.voice_jobs.executor.subprocess.run",
        run,
    )
    with pytest.raises(VoiceJobExecutionError, match=f"^{expected_code}$"):
        execute_voice_audio_validation_job(
            VoiceAudioValidationJob(
                model_name="fixture-quality",
                lineage=_lineage(),
                source_audio=descriptor,
            ),
            resolver=resolver,
        )


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("decoded", "audio_decoded_too_large"),
        ("duration", "audio_duration_exceeded"),
        ("malformed", "audio_decode_failed"),
    ],
)
def test_non_wav_decoded_audio_enforces_post_decode_ceilings(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    encoded = b"compressed-fixture"
    descriptor = _external_audio_descriptor(
        encoded,
        name="untrusted.ogg",
        media_type="audio/ogg",
    )
    if case == "decoded":
        decoded = _pcm16_wav_bytes(tuple(range(11)))
        resolver = _resolver(
            tmp_path,
            fetcher=lambda uri, limit: encoded,
            max_decoded_bytes=20,
        )
    elif case == "duration":
        decoded = _pcm16_wav_bytes(tuple(0 for _ in range(9)), sample_rate=8_000)
        resolver = _resolver(
            tmp_path,
            fetcher=lambda uri, limit: encoded,
            max_duration_ms=1,
        )
    else:
        decoded = b"not-pcm-wav"
        resolver = _resolver(tmp_path, fetcher=lambda uri, limit: encoded)

    with pytest.raises(VoiceJobExecutionError, match=f"^{expected_code}$"):
        execute_voice_audio_validation_job(
            VoiceAudioValidationJob(
                model_name="fixture-quality",
                lineage=_lineage(),
                source_audio=descriptor,
            ),
            resolver=resolver,
            audio_decoder_fn=lambda data, input_format, policy: decoded,
        )


def test_non_wav_validation_rejects_unallowlisted_audio_before_decoder(
    tmp_path: Path,
) -> None:
    encoded = b"untrusted-aac"
    descriptor = _external_audio_descriptor(
        encoded,
        name="untrusted.aac",
        media_type="audio/aac",
    )
    resolver = _resolver(tmp_path, fetcher=lambda uri, limit: encoded)
    decoder_called = False

    def decode(data: bytes, input_format: str, policy: ArtifactPolicy) -> bytes:
        nonlocal decoder_called
        decoder_called = True
        return _wav_bytes()

    with pytest.raises(
        VoiceJobExecutionError,
        match="^audio_decoder_unsupported_media$",
    ):
        execute_voice_audio_validation_job(
            VoiceAudioValidationJob(
                model_name="fixture-quality",
                lineage=_lineage(),
                source_audio=descriptor,
            ),
            resolver=resolver,
            audio_decoder_fn=decode,
        )
    assert decoder_called is False


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("scheme", "artifact_scheme_not_allowed"),
        ("traversal", "artifact_path_traversal"),
        ("root", "file_root_not_allowed"),
        ("size", "artifact_size_mismatch"),
        ("checksum", "artifact_checksum_mismatch"),
        ("encoded_size", "artifact_too_large"),
        ("decompressed_size", "artifact_decompressed_too_large"),
    ],
)
def test_artifact_resolver_rejects_untrusted_or_oversized_inputs(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    input_root = tmp_path / "allowed"
    input_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    wav = _wav_bytes(frames=64)
    descriptor = _write_audio_descriptor(input_root, "safe.wav", wav)
    resolver = _resolver(tmp_path, input_root=input_root)

    if case == "scheme":
        descriptor = {
            **descriptor,
            "uri": "data:audio/wav;base64,UklGRg==",
        }
    elif case == "traversal":
        descriptor = {
            **descriptor,
            "uri": "artifact://voice/%2e%2e/secret.wav",
        }
    elif case == "root":
        descriptor = _write_audio_descriptor(outside_root, "escape.wav", wav)
    elif case == "size":
        descriptor = {**descriptor, "size_bytes": len(wav) + 1}
    elif case == "checksum":
        descriptor = {**descriptor, "sha256": "0" * 64}
    elif case == "encoded_size":
        resolver = _resolver(
            tmp_path,
            input_root=input_root,
            max_input_bytes=len(wav) - 1,
        )
    elif case == "decompressed_size":
        compressed = gzip.compress(wav)
        descriptor = _write_audio_descriptor(
            input_root,
            "compressed.wav.gz",
            compressed,
        )
        descriptor["content_encoding"] = "gzip"
        resolver = _resolver(
            tmp_path,
            input_root=input_root,
            max_input_bytes=len(compressed) + 1,
            max_decoded_bytes=len(wav) - 1,
        )

    with pytest.raises(VoiceJobExecutionError, match=f"^{expected_code}$"):
        resolver.resolve(descriptor)


def test_artifact_resolver_rejects_ssrf_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fetch_calls: list[str] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.voice_jobs.executor.socket.getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))],
    )
    resolver = _resolver(
        tmp_path,
        allowed_schemes=frozenset({"https"}),
        fetcher=lambda uri, limit: fetch_calls.append(uri) or b"never",
    )
    descriptor = {
        "uri": "https://metadata.example.invalid/audio.wav",
        "sha256": hashlib.sha256(b"never").hexdigest(),
        "size_bytes": 5,
        "media_type": "audio/wav",
    }

    with pytest.raises(VoiceJobExecutionError, match="^artifact_ssrf_rejected$"):
        resolver.resolve(descriptor)
    assert fetch_calls == []


@pytest.mark.parametrize(
    ("case", "max_duration_ms", "expected_code"),
    [
        ("invalid", 60_000, "audio_decode_failed"),
        ("duration", 1_000, "audio_duration_exceeded"),
    ],
)
def test_audio_validation_rejects_decode_and_duration_bombs(
    tmp_path: Path,
    case: str,
    max_duration_ms: int,
    expected_code: str,
) -> None:
    data = b"not-a-wave" if case == "invalid" else _wav_bytes(frames=8_001)
    descriptor = _external_audio_descriptor(data, name="untrusted.wav")
    resolver = _resolver(
        tmp_path,
        fetcher=lambda uri, limit: data,
        max_duration_ms=max_duration_ms,
    )
    with pytest.raises(VoiceJobExecutionError, match=f"^{expected_code}$"):
        execute_voice_audio_validation_job(
            VoiceAudioValidationJob(
                model_name="fixture-quality",
                lineage=_lineage(),
                source_audio=descriptor,
            ),
            resolver=resolver,
        )


def test_voice_task_aliases_are_canonical_and_mesh_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.p2p_tasks.service import _supported_task_types_env

    assert canonical_task_type("text-to-speech") == "voice.tts"
    assert canonical_task_type("STT") == "voice.asr"
    assert canonical_task_type("automatic-speech-recognition") == "voice.asr"
    assert canonical_task_type("audio_validate") == "voice.audio-validate"

    advertised = _compute_supported_task_types(
        supported_task_types=["tts", "speech-to-text", "audio_validate"],
        accelerate_instance=None,
    )
    mesh_advertised = _mesh_safe_task_types(advertised)
    for task_type in VOICE_TASK_TYPES:
        assert task_type in advertised
        assert task_type in mesh_advertised

    # Older queue spellings remain claimable, but only one canonical operation
    # is used by dispatch and receipts.
    assert normalize_task_types(
        ["tts", "voice.tts", "stt", "voice.asr"],
        expand_aliases=False,
    ) == ["voice.tts", "voice.asr"]

    monkeypatch.setenv(
        "IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES",
        "tts,speech-to-text,audio_validate",
    )
    worker_supported = _compute_supported_task_types(
        supported_task_types=None,
        accelerate_instance=None,
    )
    assert _supported_task_types_env() == worker_supported


@pytest.mark.parametrize(
    ("submitted_type", "canonical_type"),
    [
        ("text-to-speech", "voice.tts"),
        ("speech-to-text", "voice.asr"),
        ("audio_validate", "voice.audio-validate"),
    ],
)
def test_worker_dispatches_voice_handlers_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    submitted_type: str,
    canonical_type: str,
) -> None:
    import ipfs_accelerate_py.voice_jobs.executor as executor

    calls: list[dict[str, Any]] = []

    def fake_execute(task: dict[str, Any], **_: object) -> dict[str, Any]:
        calls.append(task)
        return {
            "status": "succeeded",
            "task_type": canonical_task_type(task["task_type"]),
            "provider_receipt": {"provider": "offline-fixture"},
        }

    monkeypatch.setattr(executor, "execute_task", fake_execute)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ROUTE_VIA_BACKEND_MANAGER", "0")

    queue_path = str(tmp_path / "voice-worker.duckdb")
    queue = TaskQueue(queue_path)
    task_id = queue.submit(
        task_type=submitted_type,
        model_name="offline-model",
        payload={"fixture": canonical_type},
    )

    assert (
        run_worker(
            queue_path=queue_path,
            worker_id="offline-voice-worker",
            poll_interval_s=0.01,
            once=True,
            supported_task_types=[canonical_type],
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0]["task_type"] == submitted_type

    queued = queue.get(task_id)
    assert queued is not None
    assert queued["status"] == "completed"
    assert queued["result"]["task_type"] == canonical_type


def test_backend_manager_voice_provider_uses_current_async_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipfs_accelerate_py.voice_router as voice_router
    from ipfs_accelerate_py.router_deps import RouterDeps

    calls: list[dict[str, Any]] = []

    class AsyncBackendManager:
        def select_backend_for_task(self, **_: object) -> object:
            raise AssertionError("voice adapter must not pre-select or index BackendInfo")

        def execute_inference(self, **_: object) -> object:
            raise AssertionError("retired execute_inference API must not be called")

        async def execute_task(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            if kwargs["task"] == "text-to-speech":
                return {
                    "backend_id": "backend-object",
                    "result": {"audio": base64.b64encode(b"offline-audio").decode("ascii")},
                }
            return {
                "backend_id": "backend-object",
                "result": {"transcript": "offline transcript"},
            }

    monkeypatch.setenv("IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER", "1")
    provider = voice_router._get_backend_manager_provider(
        RouterDeps(backend_manager=AsyncBackendManager())
    )
    assert provider is not None

    # Invoke from inside an active event loop too: the public provider boundary
    # remains synchronous even though InferenceBackendManager.execute_task is
    # async.
    async def invoke() -> tuple[bytes, str]:
        audio = provider.synthesize(
            "hello",
            model_name="tts-model",
            voice="abby",
            device="cuda:1",
            output_format="wav",
        )
        transcript = provider.transcribe(
            b"caller-audio",
            model_name="asr-model",
            language="en",
            device="cpu",
        )
        return audio, transcript

    assert asyncio.run(invoke()) == (b"offline-audio", "offline transcript")
    assert [call["task"] for call in calls] == [
        "text-to-speech",
        "automatic-speech-recognition",
    ]
    assert calls[0]["model"] == "tts-model"
    assert calls[0]["inputs"] == ["hello"]
    assert calls[0]["parameters"]["device"] == "cuda:1"
    assert calls[1]["model"] == "asr-model"
    assert calls[1]["inputs"] == [
        base64.b64encode(b"caller-audio").decode("ascii")
    ]
    assert calls[1]["parameters"]["device"] == "cpu"
    assert all("protocol" not in call for call in calls)


def test_huggingface_tts_and_stt_use_independent_device_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    import ipfs_accelerate_py.voice_router as voice_router

    constructed: list[tuple[str, int]] = []
    stt_calls: list[dict[str, object]] = []

    class FakePipeline:
        def __init__(self, task: str) -> None:
            self.task = task

        def __call__(self, value: object, **kwargs: object) -> dict[str, object]:
            if self.task == "text-to-speech":
                return {
                    "audio": np.zeros(32, dtype=np.float32),
                    "sampling_rate": 8_000,
                }
            stt_calls.append(kwargs)
            return {"text": "device-isolated transcript"}

    def fake_pipeline(task: str, *, model: str, device: int) -> FakePipeline:
        assert model
        constructed.append((task, device))
        return FakePipeline(task)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_VOICE_DEVICE", "cuda")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TTS_DEVICE", "cuda")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_STT_DEVICE", "cpu")

    provider = voice_router._get_huggingface_provider()
    assert provider is not None
    assert provider.synthesize("hello").startswith(b"RIFF")
    assert provider.transcribe(
        _wav_bytes(),
        model_name="openai/whisper-base",
        language="en-US",
    ) == "device-isolated transcript"
    assert provider.transcribe(
        _wav_bytes(),
        model_name="openai/whisper-base.en",
        language="en-US",
    ) == "device-isolated transcript"
    assert constructed == [
        ("text-to-speech", 0),
        ("automatic-speech-recognition", -1),
        ("automatic-speech-recognition", -1),
    ]
    assert stt_calls == [
        {
            "generate_kwargs": {"language": "en"},
            "return_timestamps": True,
        },
        {
            "return_timestamps": True,
        },
    ]
