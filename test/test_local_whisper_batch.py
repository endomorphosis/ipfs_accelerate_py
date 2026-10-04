"""Unit tests for LocalWhisperBatchSession (no model download)."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.voice_jobs.local_whisper_batch import (
    LocalWhisperBatchSession,
    batch_transcribe_files,
)


def test_batch_session_uses_injected_function(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"ID3fake")

    def fake_stt(path, **kwargs):
        assert Path(path).name == "clip.mp3"
        assert kwargs.get("provider") == "huggingface"
        return "Portland"

    session = LocalWhisperBatchSession(
        model_name="openai/whisper-tiny",
        device="cpu",
        speech_to_text_fn=fake_stt,
    )
    text = session.transcribe_path(audio)
    assert text == "Portland"

    results = list(
        session.transcribe_paths(
            [audio],
            expected_by_path={str(audio): "Portland"},
            max_wer_bp=1000,
        )
    )
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].matched is True
    assert results[0].wer_bp == 0


def test_batch_transcribe_files_wrapper(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    def fake_stt(path, **kwargs):
        return "hello"

    session = LocalWhisperBatchSession(speech_to_text_fn=fake_stt)
    out = list(session.transcribe_paths([audio]))
    assert out[0].transcript == "hello"
