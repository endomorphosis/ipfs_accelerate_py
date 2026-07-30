"""Offline tests for the deployment-selected local response-DAG runtime."""

from __future__ import annotations

import stat
from hashlib import sha256
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.voice_response_dag_runtime import (
    RESPONSE_DAG_AUDIO_ROOT_ENV,
    RESPONSE_DAG_QUEUE_ROOT_ENV,
    RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV,
    LocalWhisperResponseDAGPostprocessor,
    load_local_voice_response_dag_runtime_from_environment,
)
from ipfs_accelerate_py.voice_response_dag_sink import (
    LocalResponseDAGQueueError,
)

AUDIO = b"RIFF-independent-whisper-validation-WAVE"
RESPONSE = "Call 503-555-0100."


def _result(
    *,
    text: str = RESPONSE,
    audio: bytes = AUDIO,
    audio_format: str = "wav",
) -> SimpleNamespace:
    return SimpleNamespace(
        response_text=text,
        audio=audio,
        audio_format=audio_format,
    )


def test_runtime_is_disabled_without_an_explicit_queue_root() -> None:
    assert (
        load_local_voice_response_dag_runtime_from_environment(environ={})
        is None
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        (RESPONSE_DAG_QUEUE_ROOT_ENV, "relative/queue"),
        (RESPONSE_DAG_QUEUE_ROOT_ENV, "https://example.test/queue"),
        (RESPONSE_DAG_AUDIO_ROOT_ENV, "relative/audio"),
        (RESPONSE_DAG_AUDIO_ROOT_ENV, "file:///tmp/audio"),
    ),
)
def test_runtime_rejects_non_absolute_local_roots(
    tmp_path,
    name: str,
    value: str,
) -> None:
    environ = {RESPONSE_DAG_QUEUE_ROOT_ENV: str(tmp_path / "queue")}
    environ[name] = value
    with pytest.raises(LocalResponseDAGQueueError):
        load_local_voice_response_dag_runtime_from_environment(
            environ=environ
        )


def test_runtime_rejects_invalid_wer_configuration(tmp_path) -> None:
    with pytest.raises(LocalResponseDAGQueueError):
        load_local_voice_response_dag_runtime_from_environment(
            environ={
                RESPONSE_DAG_QUEUE_ROOT_ENV: str(tmp_path / "queue"),
                RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV: "not-an-integer",
            }
        )


def test_independent_validation_persists_content_addressed_private_audio(
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    def transcribe(audio: bytes, **kwargs: object) -> str:
        calls.append({"audio": audio, **kwargs})
        return "Call five oh three five five five zero one zero zero"

    runtime = load_local_voice_response_dag_runtime_from_environment(
        environ={
            RESPONSE_DAG_QUEUE_ROOT_ENV: str(tmp_path / "queue"),
        },
        transcriber=transcribe,
    )

    assert runtime is not None
    assert runtime.remote_writes is False
    artifacts = runtime.postprocessor.validate_and_store_local(_result())
    assert artifacts is not None
    assert artifacts.remote_writes is False
    assert artifacts.validation_receipt.passed is True
    assert artifacts.validation_receipt.rendered_text_sha256 == sha256(
        RESPONSE.encode("utf-8")
    ).hexdigest()
    assert artifacts.validation_receipt.output_audio_sha256 == sha256(
        AUDIO
    ).hexdigest()
    assert artifacts.audio_descriptor["content_sha256"] == sha256(
        AUDIO
    ).hexdigest()
    assert artifacts.audio_descriptor["media_type"] == "audio/wav"
    audio_path = (
        runtime.postprocessor.audio_root
        / f"{sha256(AUDIO).hexdigest()}.wav"
    )
    assert audio_path.read_bytes() == AUDIO
    assert stat.S_IMODE(audio_path.stat().st_mode) == 0o600
    assert calls == [
        {
            "audio": AUDIO,
            "content_type": "audio/wav",
            "device": None,
            "language": "en",
            "model_name": "openai/whisper-base",
        }
    ]

    duplicate = runtime.postprocessor.validate_and_store_local(_result())
    assert duplicate is not None
    assert duplicate.to_dict() == artifacts.to_dict()


@pytest.mark.parametrize(
    "transcript",
    (
        "Call negative five zero three five five five zero one zero zero",
        "Call five zero three five five five zero one zero one",
        "Call five zero three hyphen five five five zero one zero zero",
    ),
)
def test_spoken_negative_hyphen_or_wrong_digit_is_not_queued(
    tmp_path,
    transcript: str,
) -> None:
    postprocessor = LocalWhisperResponseDAGPostprocessor(
        tmp_path / "audio",
        transcriber=lambda _audio, **_kwargs: transcript,
    )

    assert postprocessor.validate_and_store_local(_result()) is None
    assert postprocessor.last_error_code == "asr_round_trip_mismatch"
    assert list(postprocessor.audio_root.iterdir()) == []


def test_validator_failure_does_not_persist_audio(tmp_path) -> None:
    def fail(_audio: bytes, **_kwargs: object) -> str:
        raise RuntimeError("offline")

    postprocessor = LocalWhisperResponseDAGPostprocessor(
        tmp_path / "audio",
        transcriber=fail,
    )

    assert postprocessor.validate_and_store_local(_result()) is None
    assert postprocessor.last_error_code == "RuntimeError"
    assert list(postprocessor.audio_root.iterdir()) == []


def test_unsafe_audio_format_is_rejected_before_persistence(tmp_path) -> None:
    postprocessor = LocalWhisperResponseDAGPostprocessor(
        tmp_path / "audio",
        transcriber=lambda _audio, **_kwargs: RESPONSE,
    )

    with pytest.raises(
        LocalResponseDAGQueueError,
        match="simple codec name",
    ):
        postprocessor.validate_and_store_local(
            _result(audio_format="../../queue")
        )
    assert list(postprocessor.audio_root.iterdir()) == []
