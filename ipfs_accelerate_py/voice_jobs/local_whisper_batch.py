"""Local batch Whisper transcription via the voice_router HuggingFace provider.

The public :func:`speech_to_text` entrypoint already falls back to a cached
transformers ASR pipeline.  This module keeps a single provider instance warm
across thousands of short BM25/vocabulary clips so rescue validation is not
dominated by model reload cost.

MP3/OGG inputs are written to temporary files so the pipeline can demux them
reliably (raw-byte WAV decoding only handles PCM WAV).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .executor import ArtifactPolicy, VoiceJobExecutionError


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    env = (
        os.getenv("IPFS_ACCELERATE_PY_STT_DEVICE")
        or os.getenv("IPFS_ACCELERATE_PY_VOICE_DEVICE")
        or ""
    ).strip()
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True, slots=True)
class BatchTranscriptResult:
    """One batch transcription observation."""

    path: str
    transcript: str
    ok: bool
    error: str = ""
    expected_text: str | None = None
    wer_bp: int | None = None
    matched: bool | None = None


class LocalWhisperBatchSession:
    """Warm local Whisper session for sequential/batch file transcription."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        language: str = "en",
        provider_name: str = "huggingface",
        speech_to_text_fn: Callable[..., str] | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "IPFS_ACCELERATE_PY_STT_MODEL", "openai/whisper-base"
        )
        self.device = _resolve_device(device)
        self.language = language
        self.provider_name = provider_name
        self._speech_to_text_fn = speech_to_text_fn
        self._provider: Any | None = None

    def _ensure_provider(self) -> Any:
        if self._speech_to_text_fn is not None:
            return None
        if self._provider is not None:
            return self._provider
        # Import lazily so unit tests can inject a pure function.
        from ipfs_accelerate_py.voice_router import (
            _get_huggingface_provider,
            get_voice_provider,
        )

        if self.provider_name in {"huggingface", "hf", "local_hf"}:
            provider = _get_huggingface_provider()
            if provider is None:
                raise VoiceJobExecutionError("audio_decoder_unavailable")
            self._provider = provider
            return provider
        self._provider = get_voice_provider(self.provider_name)
        return self._provider

    def transcribe_path(self, path: str | Path) -> str:
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise VoiceJobExecutionError("audio_decode_failed")
        # Prefer path form so transformers/ffmpeg can demux non-WAV containers.
        if self._speech_to_text_fn is not None:
            return str(
                self._speech_to_text_fn(
                    str(path),
                    model_name=self.model_name,
                    language=self.language,
                    device=self.device,
                    provider=self.provider_name,
                )
            )
        provider = self._ensure_provider()
        text = provider.transcribe(
            str(path),
            model_name=self.model_name,
            language=self.language,
            device=self.device,
        )
        if not isinstance(text, str):
            raise VoiceJobExecutionError("voice_provider_invalid_transcript")
        return text

    def transcribe_bytes(self, data: bytes, *, suffix: str = ".mp3") -> str:
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise VoiceJobExecutionError("audio_decode_failed")
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(bytes(data))
            temp_name = handle.name
        try:
            return self.transcribe_path(temp_name)
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass

    def transcribe_paths(
        self,
        paths: Iterable[str | Path],
        *,
        expected_by_path: dict[str, str] | None = None,
        max_wer_bp: int = 2_500,
    ) -> Iterator[BatchTranscriptResult]:
        """Yield transcripts for each path, optionally scoring against expected text."""

        from ipfs_datasets_py.voice.audio_quality import word_error_rate_bp
        from ipfs_datasets_py.voice.normalize import (
            normalize_indextts_spoken_text,
            normalized_text_identity,
        )

        expected_by_path = expected_by_path or {}
        import re

        from ipfs_datasets_py.voice.audio_quality import character_error_rate_bp

        def _alnum(text: str) -> str:
            return re.sub(r"[^a-z0-9]", "", text.casefold())

        def _vocab_match(expected: str, transcript: str) -> tuple[bool, int]:
            """Score short BM25 terms with CER/substring forgiveness.

            Whisper often expands spellings (``hilight``→``highlight``,
            ``birthdates``→``birth dates``) on single-token vocabulary clips.
            """

            hyp = normalized_text_identity(
                normalize_indextts_spoken_text(transcript)
            )
            ref = normalized_text_identity(
                normalize_indextts_spoken_text(expected)
            )
            if ref and hyp == ref:
                return True, 0
            wer = word_error_rate_bp(expected, transcript)
            if wer <= max_wer_bp:
                return True, wer
            ref_a, hyp_a = _alnum(ref), _alnum(hyp)
            if ref_a and hyp_a and (ref_a == hyp_a or ref_a in hyp_a or hyp_a in ref_a):
                return True, min(wer, 1_500)
            # Character error on alnum forms is more stable for short tokens.
            if ref_a and hyp_a:
                cer = character_error_rate_bp(ref_a, hyp_a)
                if cer <= max(max_wer_bp, 3_500):
                    return True, cer
            return False, wer

        # Warm the model once on the first file.
        for raw_path in paths:
            path = str(Path(raw_path))
            expected = expected_by_path.get(path)
            try:
                transcript = self.transcribe_path(path)
                ok = bool(transcript and transcript.strip())
                wer: int | None = None
                matched: bool | None = None
                if expected is not None and ok:
                    matched, wer = _vocab_match(expected, transcript)
                yield BatchTranscriptResult(
                    path=path,
                    transcript=transcript if ok else "",
                    ok=ok,
                    expected_text=expected,
                    wer_bp=wer,
                    matched=matched,
                )
            except Exception as exc:
                yield BatchTranscriptResult(
                    path=path,
                    transcript="",
                    ok=False,
                    error=str(exc)[:500],
                    expected_text=expected,
                    matched=False,
                )


def batch_transcribe_files(
    paths: Sequence[str | Path],
    *,
    model_name: str | None = None,
    device: str | None = None,
    language: str = "en",
    expected_by_path: dict[str, str] | None = None,
    max_wer_bp: int = 2_500,
) -> list[BatchTranscriptResult]:
    """Convenience wrapper that materializes all batch results."""

    session = LocalWhisperBatchSession(
        model_name=model_name,
        device=device,
        language=language,
        provider_name="huggingface",
    )
    return list(
        session.transcribe_paths(
            paths,
            expected_by_path=expected_by_path,
            max_wer_bp=max_wer_bp,
        )
    )


__all__ = [
    "BatchTranscriptResult",
    "LocalWhisperBatchSession",
    "batch_transcribe_files",
]
