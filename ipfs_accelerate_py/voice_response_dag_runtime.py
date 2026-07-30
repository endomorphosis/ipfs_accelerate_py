"""Environment-owned local runtime for validated voice response-DAG misses.

The request path may validate newly synthesized audio and durably stage an
append candidate, but it must never publish to Hugging Face, IPFS, or another
remote sink.  This module therefore owns only:

* a private, append-only :class:`LocalResponseDAGQueue`;
* independent ASR round-trip validation; and
* content-addressed local audio persistence used by the queued descriptor.

Publication remains a separate, explicitly authorized worker operation.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from .voice_response_dag_sink import (
    IndependentVoiceValidationReceipt,
    LocalResponseDAGQueue,
    LocalResponseDAGQueueError,
    LocalValidatedVoiceCacheMissArtifacts,
    _ensure_private_directory,
    _fsync_directory,
)

RESPONSE_DAG_QUEUE_ROOT_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_QUEUE_ROOT"
)
RESPONSE_DAG_AUDIO_ROOT_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_AUDIO_ROOT"
)
RESPONSE_DAG_VALIDATOR_PROVIDER_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_VALIDATOR_PROVIDER"
)
RESPONSE_DAG_VALIDATOR_MODEL_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_VALIDATOR_MODEL"
)
RESPONSE_DAG_VALIDATOR_LANGUAGE_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_VALIDATOR_LANGUAGE"
)
RESPONSE_DAG_VALIDATOR_DEVICE_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_VALIDATOR_DEVICE"
)
RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV: Final = (
    "IPFS_ACCELERATE_PY_ABBY_RESPONSE_DAG_VALIDATOR_MAX_WER_BP"
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_CANONICAL_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_AUDIO_MEDIA_TYPES = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "wave": "audio/wav",
    "webm": "audio/webm",
}
_AUDIO_SUFFIXES = {
    "aac": ".aac",
    "flac": ".flac",
    "m4a": ".m4a",
    "mp3": ".mp3",
    "mpeg": ".mp3",
    "ogg": ".ogg",
    "opus": ".opus",
    "wav": ".wav",
    "wave": ".wav",
    "webm": ".webm",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _comparison_tokens(value: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    result: list[str] = []
    for token in _TOKEN_RE.findall(normalized):
        if token.isdigit():
            result.extend(_CANONICAL_DIGIT_WORDS[digit] for digit in token)
        elif token in _DIGIT_WORDS:
            result.append(_CANONICAL_DIGIT_WORDS[_DIGIT_WORDS[token]])
        else:
            result.append(token)
    return tuple(result)


def _spoken_digit_sequence(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_DIGIT_WORDS[token] for token in tokens if token in _DIGIT_WORDS)


def _word_error_rate_bp(
    expected_tokens: tuple[str, ...],
    actual_tokens: tuple[str, ...],
) -> int:
    if not expected_tokens:
        return 0 if not actual_tokens else 10_000
    previous = list(range(len(actual_tokens) + 1))
    for expected_index, expected in enumerate(expected_tokens, start=1):
        current = [expected_index]
        for actual_index, actual in enumerate(actual_tokens, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (expected != actual),
                )
            )
        previous = current
    return min(
        10_000,
        (previous[-1] * 10_000 + len(expected_tokens) // 2)
        // len(expected_tokens),
    )


def _call_with_supported_keywords(
    function: Callable[..., object],
    audio: bytes,
    **kwargs: object,
) -> object:
    try:
        signature = inspect.signature(function)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        selected = (
            kwargs
            if accepts_kwargs
            else {
                name: value
                for name, value in kwargs.items()
                if name in signature.parameters
            }
        )
    except (TypeError, ValueError):
        selected = kwargs
    return function(audio, **selected)


def _require_absolute_local_path(value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise LocalResponseDAGQueueError(f"{label} must be configured")
    if "://" in raw:
        raise LocalResponseDAGQueueError(
            f"{label} must be a local filesystem path"
        )
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise LocalResponseDAGQueueError(f"{label} must be an absolute path")
    return requested


class LocalWhisperResponseDAGPostprocessor:
    """Independently validate and persist one live-TTS cache-miss artifact."""

    remote_writes = False

    def __init__(
        self,
        audio_root: str | Path,
        *,
        provider_name: str = "abby_whisper",
        model_name: str = "openai/whisper-base",
        language: str = "en",
        device: str | None = None,
        max_wer_bp: int = 0,
        transcriber: Callable[..., str] | None = None,
    ) -> None:
        requested_root = Path(audio_root).expanduser()
        if not requested_root.is_absolute():
            raise LocalResponseDAGQueueError(
                "response-DAG audio root must be an absolute path"
            )
        self.audio_root = requested_root.resolve()
        _ensure_private_directory(self.audio_root)
        self.provider_name = str(provider_name or "").strip().casefold()
        self.model_name = str(model_name or "").strip()
        self.language = str(language or "").strip() or "en"
        self.device = str(device or "").strip() or None
        if not self.provider_name:
            raise LocalResponseDAGQueueError(
                "response-DAG validator provider must be non-empty"
            )
        if not self.model_name:
            raise LocalResponseDAGQueueError(
                "response-DAG validator model must be non-empty"
            )
        if (
            isinstance(max_wer_bp, bool)
            or not isinstance(max_wer_bp, int)
            or not 0 <= max_wer_bp <= 10_000
        ):
            raise LocalResponseDAGQueueError(
                "response-DAG validator max WER must be 0-10000 basis points"
            )
        self.max_wer_bp = max_wer_bp
        self._transcriber = transcriber
        self._provider: object | None = None
        self._validation_lock = threading.Lock()
        self.last_error_code: str | None = None
        self.last_wer_bp: int | None = None

    @property
    def validator_identity(self) -> str:
        return f"{self.provider_name}:{self.model_name}"

    def _transcribe(self, audio: bytes, *, media_type: str) -> str:
        options = {
            "content_type": media_type,
            "device": self.device,
            "language": self.language,
            "model_name": self.model_name,
        }
        if self._transcriber is not None:
            result = _call_with_supported_keywords(
                self._transcriber,
                audio,
                **options,
            )
        else:
            if self._provider is None:
                from .voice_router import get_voice_provider

                self._provider = get_voice_provider(self.provider_name)
            transcribe = getattr(self._provider, "transcribe", None)
            if not callable(transcribe):
                raise LocalResponseDAGQueueError(
                    "response-DAG validator provider cannot transcribe audio"
                )
            result = _call_with_supported_keywords(
                transcribe,
                audio,
                **options,
            )
        transcript = str(result or "").strip()
        if not transcript:
            raise LocalResponseDAGQueueError(
                "response-DAG validator returned an empty transcript"
            )
        return transcript

    def _persist_audio(self, audio: bytes, *, suffix: str) -> Path:
        digest = sha256(audio).hexdigest()
        target = self.audio_root / f"{digest}{suffix}"
        if target.is_symlink():
            raise LocalResponseDAGQueueError(
                "response-DAG audio target must not be a symlink"
            )
        if target.exists():
            if not target.is_file() or target.read_bytes() != audio:
                raise LocalResponseDAGQueueError(
                    "response-DAG audio target contains conflicting bytes"
                )
            if target.stat().st_mode & 0o077:
                raise LocalResponseDAGQueueError(
                    "response-DAG audio files must be private (0600)"
                )
            return target

        temporary = self.audio_root / (
            f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.partial"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(audio)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target, follow_symlinks=False)
                _fsync_directory(self.audio_root)
            except FileExistsError:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.read_bytes() != audio
                ):
                    raise LocalResponseDAGQueueError(
                        "response-DAG audio target contains conflicting bytes"
                    )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
                _fsync_directory(self.audio_root)
            except FileNotFoundError:
                pass
        return target

    def validate_and_store_local(
        self,
        result: object,
    ) -> LocalValidatedVoiceCacheMissArtifacts | None:
        response_text = str(getattr(result, "response_text", "") or "").strip()
        audio = getattr(result, "audio", None)
        if not response_text or not isinstance(audio, bytes) or not audio:
            return None
        audio_format = str(
            getattr(result, "audio_format", "") or "wav"
        ).strip().casefold().lstrip(".")
        if not re.fullmatch(r"[a-z0-9]{1,16}", audio_format):
            raise LocalResponseDAGQueueError(
                "response-DAG audio format must be a simple codec name"
            )
        media_type = _AUDIO_MEDIA_TYPES.get(
            audio_format,
            f"audio/{audio_format}",
        )
        suffix = _AUDIO_SUFFIXES.get(audio_format, f".{audio_format}")

        with self._validation_lock:
            try:
                transcript = self._transcribe(audio, media_type=media_type)
            except Exception as exc:
                self.last_error_code = exc.__class__.__name__
                self.last_wer_bp = None
                return None

        expected_tokens = _comparison_tokens(response_text)
        actual_tokens = _comparison_tokens(transcript)
        wer_bp = _word_error_rate_bp(expected_tokens, actual_tokens)
        self.last_wer_bp = wer_bp
        expected_digits = _spoken_digit_sequence(expected_tokens)
        actual_digits = _spoken_digit_sequence(actual_tokens)
        if (
            not expected_tokens
            or wer_bp > self.max_wer_bp
            or expected_digits != actual_digits
        ):
            self.last_error_code = "asr_round_trip_mismatch"
            return None

        persisted = self._persist_audio(audio, suffix=suffix)
        rendered_digest = sha256(response_text.encode("utf-8")).hexdigest()
        audio_digest = sha256(audio).hexdigest()
        observation = {
            "audio_sha256": audio_digest,
            "max_wer_bp": self.max_wer_bp,
            "rendered_text_sha256": rendered_digest,
            "transcript_sha256": sha256(
                " ".join(actual_tokens).encode("utf-8")
            ).hexdigest(),
            "validator_identity": self.validator_identity,
            "wer_bp": wer_bp,
        }
        validation_digest = sha256(_canonical_bytes(observation)).hexdigest()
        receipt = IndependentVoiceValidationReceipt(
            validation_receipt_id=(
                f"abby-voice-runtime-validation:sha256:{validation_digest}"
            ),
            rendered_text_sha256=rendered_digest,
            output_audio_sha256=audio_digest,
            validator_identity=self.validator_identity,
            validation_method="asr_round_trip",
        )
        self.last_error_code = None
        return LocalValidatedVoiceCacheMissArtifacts(
            validation_receipt=receipt,
            audio_descriptor={
                "byte_length": len(audio),
                "content_sha256": audio_digest,
                "media_type": media_type,
                "uri": persisted.as_uri(),
            },
            response_id=f"voice-response:sha256:{rendered_digest}",
        )


@dataclass(frozen=True, slots=True)
class LocalVoiceResponseDAGRuntime:
    """Package-owned sink and independent validator selected by deployment."""

    sink: LocalResponseDAGQueue
    postprocessor: LocalWhisperResponseDAGPostprocessor
    remote_writes: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sink, LocalResponseDAGQueue):
            raise TypeError("sink must be a LocalResponseDAGQueue")
        if not isinstance(
            self.postprocessor,
            LocalWhisperResponseDAGPostprocessor,
        ):
            raise TypeError(
                "postprocessor must be a LocalWhisperResponseDAGPostprocessor"
            )
        if self.remote_writes is not False:
            raise LocalResponseDAGQueueError(
                "local response-DAG runtime cannot perform remote writes"
            )


def load_local_voice_response_dag_runtime_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    transcriber: Callable[..., str] | None = None,
) -> LocalVoiceResponseDAGRuntime | None:
    """Build the local-only runtime when its absolute queue root is configured."""

    values = os.environ if environ is None else environ
    queue_raw = str(values.get(RESPONSE_DAG_QUEUE_ROOT_ENV, "") or "").strip()
    if not queue_raw:
        return None
    queue_root = _require_absolute_local_path(
        queue_raw,
        label=RESPONSE_DAG_QUEUE_ROOT_ENV,
    )
    audio_raw = str(values.get(RESPONSE_DAG_AUDIO_ROOT_ENV, "") or "").strip()
    audio_root = (
        _require_absolute_local_path(
            audio_raw,
            label=RESPONSE_DAG_AUDIO_ROOT_ENV,
        )
        if audio_raw
        else queue_root / "validated-audio"
    )
    raw_max_wer = str(
        values.get(RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV, "0") or "0"
    ).strip()
    try:
        max_wer_bp = int(raw_max_wer)
    except ValueError as exc:
        raise LocalResponseDAGQueueError(
            f"{RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV} must be an integer"
        ) from exc
    # Initialize the queue first so its root is created with the required
    # private mode before the default nested audio directory is added.
    sink = LocalResponseDAGQueue(queue_root)
    postprocessor = LocalWhisperResponseDAGPostprocessor(
        audio_root,
        provider_name=str(
            values.get(
                RESPONSE_DAG_VALIDATOR_PROVIDER_ENV,
                "abby_whisper",
            )
            or "abby_whisper"
        ),
        model_name=str(
            values.get(
                RESPONSE_DAG_VALIDATOR_MODEL_ENV,
                "openai/whisper-base",
            )
            or "openai/whisper-base"
        ),
        language=str(
            values.get(RESPONSE_DAG_VALIDATOR_LANGUAGE_ENV, "en") or "en"
        ),
        device=str(
            values.get(RESPONSE_DAG_VALIDATOR_DEVICE_ENV, "") or ""
        )
        or None,
        max_wer_bp=max_wer_bp,
        transcriber=transcriber,
    )
    return LocalVoiceResponseDAGRuntime(
        sink=sink,
        postprocessor=postprocessor,
    )


__all__ = [
    "RESPONSE_DAG_AUDIO_ROOT_ENV",
    "RESPONSE_DAG_QUEUE_ROOT_ENV",
    "RESPONSE_DAG_VALIDATOR_DEVICE_ENV",
    "RESPONSE_DAG_VALIDATOR_LANGUAGE_ENV",
    "RESPONSE_DAG_VALIDATOR_MAX_WER_BP_ENV",
    "RESPONSE_DAG_VALIDATOR_MODEL_ENV",
    "RESPONSE_DAG_VALIDATOR_PROVIDER_ENV",
    "LocalVoiceResponseDAGRuntime",
    "LocalWhisperResponseDAGPostprocessor",
    "load_local_voice_response_dag_runtime_from_environment",
]
