"""Bounded, offline-testable execution for durable voice jobs.

Queue rows contain descriptors, never audio bytes.  This module resolves and
verifies those descriptors before provider use, stores generated artifacts
outside the queue database, and returns privacy-safe descriptors and receipts.
Provider retries and circuit breaking remain owned by :mod:`voice_router`.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import inspect
import io
import ipaddress
import math
import os
import socket
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from ..p2p_tasks.task_types import VOICE_TASK_TYPES, canonical_task_type
from .contracts import (
    ArtifactDescriptor,
    VoiceJob,
    VoiceJobContractError,
    VoiceJobResult,
    voice_job_from_payload,
)


class VoiceJobExecutionError(ValueError):
    """A safe, machine-readable voice execution failure."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(str(code))
        self.code = str(code)
        self.retryable = bool(retryable)


ArtifactFetcher = Callable[[str, int], bytes]
SourceTaskResolver = Callable[[str], Mapping[str, Any] | None]
AudioDecoder = Callable[[bytes, str, "ArtifactPolicy"], bytes]

_BASIS_POINT_SCALE = 10_000
_FFMPEG_DURATION_OVERREAD_MS = 1_000
_FFMPEG_WAV_OVERHEAD_BYTES = 64 * 1024
_NON_WAV_INPUT_FORMATS = {
    "audio/flac": "flac",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
}


@dataclass(frozen=True)
class ArtifactPolicy:
    """Limits and allowlists applied before audio reaches a provider."""

    output_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("IPFS_ACCELERATE_PY_VOICE_ARTIFACT_ROOT")
            or Path(tempfile.gettempdir()) / "ipfs-accelerate-voice-artifacts"
        )
    )
    allowed_file_roots: tuple[Path, ...] = ()
    allowed_schemes: frozenset[str] = frozenset({"artifact", "file", "ipfs"})
    max_input_bytes: int = 32 * 1024 * 1024
    max_decoded_bytes: int = 64 * 1024 * 1024
    max_duration_ms: int = 30 * 60 * 1000
    decoder_timeout_seconds: float = 30.0
    silence_peak_threshold_bp: int = 100
    clipping_peak_threshold_bp: int = 9_900

    def __post_init__(self) -> None:
        output_root = Path(self.output_root).expanduser().resolve()
        roots = tuple(Path(root).expanduser().resolve() for root in self.allowed_file_roots)
        schemes = frozenset(str(item).strip().lower() for item in self.allowed_schemes if str(item).strip())
        if not schemes:
            raise ValueError("allowed_schemes must not be empty")
        for name in ("max_input_bytes", "max_decoded_bytes", "max_duration_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.decoder_timeout_seconds, bool)
            or not isinstance(self.decoder_timeout_seconds, int | float)
            or not math.isfinite(self.decoder_timeout_seconds)
            or self.decoder_timeout_seconds <= 0
        ):
            raise ValueError("decoder_timeout_seconds must be positive")
        for name in ("silence_peak_threshold_bp", "clipping_peak_threshold_bp"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > _BASIS_POINT_SCALE
            ):
                raise ValueError(f"{name} must be an integer in 0..{_BASIS_POINT_SCALE}")
        if self.silence_peak_threshold_bp > self.clipping_peak_threshold_bp:
            raise ValueError(
                "silence_peak_threshold_bp must not exceed clipping_peak_threshold_bp"
            )
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "allowed_file_roots", roots)
        object.__setattr__(self, "allowed_schemes", schemes)


class ArtifactResolver:
    """Resolve allowlisted artifact URIs with checksum and size verification."""

    def __init__(
        self,
        policy: ArtifactPolicy | None = None,
        *,
        fetcher: ArtifactFetcher | None = None,
        source_task_resolver: SourceTaskResolver | None = None,
    ) -> None:
        self.policy = policy or ArtifactPolicy()
        self.fetcher = fetcher
        self.source_task_resolver = source_task_resolver

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _artifact_path(self, parsed: Any) -> Path:
        if parsed.netloc != "voice":
            raise VoiceJobExecutionError("artifact_namespace_not_allowed")
        relative = Path(unquote(parsed.path).lstrip("/"))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise VoiceJobExecutionError("artifact_path_traversal")
        target = (self.policy.output_root / relative).resolve()
        if not self._inside(target, self.policy.output_root):
            raise VoiceJobExecutionError("artifact_path_traversal")
        return target

    def _file_path(self, parsed: Any) -> Path:
        if parsed.netloc not in {"", "localhost"}:
            raise VoiceJobExecutionError("file_host_not_allowed")
        path = Path(unquote(parsed.path)).expanduser().resolve()
        if not self.policy.allowed_file_roots:
            raise VoiceJobExecutionError("file_root_not_allowed")
        if not any(self._inside(path, root) for root in self.policy.allowed_file_roots):
            raise VoiceJobExecutionError("file_root_not_allowed")
        return path

    @staticmethod
    def _reject_private_network(parsed: Any) -> None:
        host = str(parsed.hostname or "").strip().lower()
        if not host:
            raise VoiceJobExecutionError("artifact_host_missing")
        if (
            host in {"localhost", "localhost.localdomain"}
            or host.endswith((".local", ".internal", ".localhost"))
        ):
            raise VoiceJobExecutionError("artifact_ssrf_rejected")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            try:
                addresses = {
                    item[4][0] for item in socket.getaddrinfo(host, None)
                }
            except socket.gaierror as exc:
                raise VoiceJobExecutionError(
                    "artifact_host_unresolvable", retryable=True
                ) from exc
            for address in addresses:
                try:
                    resolved = ipaddress.ip_address(address)
                except ValueError:
                    continue
                if not resolved.is_global:
                    raise VoiceJobExecutionError("artifact_ssrf_rejected")
        else:
            if not ip.is_global:
                raise VoiceJobExecutionError("artifact_ssrf_rejected")

    def _read_bounded(self, path: Path) -> bytes:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise VoiceJobExecutionError("artifact_unavailable", retryable=True) from exc
        if size > self.policy.max_input_bytes:
            raise VoiceJobExecutionError("artifact_too_large")
        try:
            with path.open("rb") as handle:
                data = handle.read(self.policy.max_input_bytes + 1)
        except OSError as exc:
            raise VoiceJobExecutionError("artifact_unavailable", retryable=True) from exc
        if len(data) > self.policy.max_input_bytes:
            raise VoiceJobExecutionError("artifact_too_large")
        return data

    @staticmethod
    def _raw_sha256_cid(digest: str) -> str:
        """Return the canonical CIDv1/raw identifier for a SHA-256 digest."""

        try:
            digest_bytes = bytes.fromhex(digest)
        except ValueError as exc:
            raise VoiceJobExecutionError("artifact_sha256_required") from exc
        if len(digest_bytes) != 32:
            raise VoiceJobExecutionError("artifact_sha256_required")
        cid_bytes = b"\x01\x55\x12\x20" + digest_bytes
        encoded = base64.b32encode(cid_bytes).decode("ascii").lower().rstrip("=")
        return f"b{encoded}"

    def _cached_ipfs_path(self, parsed: Any) -> Path | None:
        """Map an executor-produced IPFS descriptor to its verified local cache."""

        if parsed.path not in {"", "/"}:
            return None
        encoded = parsed.netloc
        if not encoded.startswith("b"):
            return None
        payload = encoded[1:].upper()
        payload += "=" * (-len(payload) % 8)
        try:
            cid_bytes = base64.b32decode(payload, casefold=True)
        except (ValueError, binascii.Error):
            return None
        if len(cid_bytes) != 36 or cid_bytes[:4] != b"\x01\x55\x12\x20":
            return None
        digest = cid_bytes[4:].hex()
        directory = (self.policy.output_root / digest[:2]).resolve()
        if not self._inside(directory, self.policy.output_root):
            return None
        for candidate in sorted(directory.glob(f"{digest}.*")):
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if self._inside(resolved, self.policy.output_root) and resolved.is_file():
                return resolved
        return None

    def _fetch(self, uri: str, parsed: Any) -> bytes:
        scheme = parsed.scheme.lower()
        if scheme == "artifact":
            return self._read_bounded(self._artifact_path(parsed))
        if scheme == "file":
            return self._read_bounded(self._file_path(parsed))
        if scheme == "ipfs":
            cached_path = self._cached_ipfs_path(parsed)
            if cached_path is not None and cached_path.is_file():
                return self._read_bounded(cached_path)
        if scheme in {"http", "https"}:
            self._reject_private_network(parsed)
        if self.fetcher is None:
            raise VoiceJobExecutionError("artifact_fetcher_unavailable")
        try:
            data = self.fetcher(uri, self.policy.max_input_bytes)
        except VoiceJobExecutionError:
            raise
        except Exception as exc:
            raise VoiceJobExecutionError("artifact_fetch_failed", retryable=True) from exc
        if not isinstance(data, bytes):
            raise VoiceJobExecutionError("artifact_fetch_invalid")
        if len(data) > self.policy.max_input_bytes:
            raise VoiceJobExecutionError("artifact_too_large")
        return data

    def _bounded_decompress(self, data: bytes, descriptor: Mapping[str, Any]) -> bytes:
        encoding = str(descriptor.get("content_encoding") or "").strip().lower()
        if encoding not in {"gzip", "x-gzip"} and not data.startswith(b"\x1f\x8b"):
            return data
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
                decoded = stream.read(self.policy.max_decoded_bytes + 1)
        except (OSError, EOFError) as exc:
            raise VoiceJobExecutionError("artifact_decompression_failed") from exc
        if len(decoded) > self.policy.max_decoded_bytes:
            raise VoiceJobExecutionError("artifact_decompressed_too_large")
        return decoded

    def resolve(self, descriptor: Mapping[str, Any]) -> bytes:
        if not isinstance(descriptor, Mapping):
            raise VoiceJobExecutionError("artifact_descriptor_required")
        uri = str(descriptor.get("uri") or "").strip()
        if not uri:
            raise VoiceJobExecutionError("artifact_uri_required")
        parsed = urlsplit(uri)
        scheme = parsed.scheme.lower()
        if scheme not in self.policy.allowed_schemes:
            raise VoiceJobExecutionError("artifact_scheme_not_allowed")
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise VoiceJobExecutionError("artifact_uri_ambiguous")
        for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
            normalized_key = key.strip().lower().replace("-", "_")
            if any(
                marker in normalized_key
                for marker in (
                    "api_key",
                    "auth",
                    "credential",
                    "password",
                    "secret",
                    "signature",
                    "token",
                )
            ):
                raise VoiceJobExecutionError("artifact_uri_credentials_rejected")
        if "\\" in unquote(parsed.path) or ".." in Path(unquote(parsed.path)).parts:
            raise VoiceJobExecutionError("artifact_path_traversal")

        raw = self._fetch(uri, parsed)
        declared_size = descriptor.get("size_bytes")
        if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
            raise VoiceJobExecutionError("artifact_size_required")
        if len(raw) != declared_size:
            raise VoiceJobExecutionError("artifact_size_mismatch")
        declared_sha = str(descriptor.get("sha256") or "").strip().lower()
        if len(declared_sha) != 64 or any(ch not in "0123456789abcdef" for ch in declared_sha):
            raise VoiceJobExecutionError("artifact_sha256_required")
        if hashlib.sha256(raw).hexdigest() != declared_sha:
            raise VoiceJobExecutionError("artifact_checksum_mismatch")

        decoded = self._bounded_decompress(raw, descriptor)
        if len(decoded) > self.policy.max_decoded_bytes:
            raise VoiceJobExecutionError("artifact_decoded_too_large")
        return decoded

    def resolve_source(self, job: Mapping[str, Any]) -> tuple[bytes, Mapping[str, Any]]:
        descriptor = job.get("source_audio")
        if isinstance(descriptor, Mapping):
            return self.resolve(descriptor), descriptor
        source_task_id = str(job.get("source_task_id") or "").strip()
        if not source_task_id or self.source_task_resolver is None:
            raise VoiceJobExecutionError("source_audio_required")
        try:
            resolved = self.source_task_resolver(source_task_id)
        except VoiceJobExecutionError:
            raise
        except Exception as exc:
            raise VoiceJobExecutionError(
                "source_task_artifact_unavailable", retryable=True
            ) from exc
        if not isinstance(resolved, Mapping):
            raise VoiceJobExecutionError("source_task_artifact_unavailable", retryable=True)
        candidate = resolved.get("artifact")
        if candidate is None:
            artifacts = resolved.get("artifacts")
            if isinstance(artifacts, list | tuple) and artifacts:
                candidate = artifacts[0]
        if not isinstance(candidate, Mapping):
            raise VoiceJobExecutionError("source_task_artifact_unavailable", retryable=True)
        return self.resolve(candidate), candidate

    def persist(self, data: bytes, *, suffix: str, media_type: str) -> dict[str, Any]:
        if not isinstance(data, bytes) or not data:
            raise VoiceJobExecutionError("artifact_output_empty")
        if len(data) > self.policy.max_decoded_bytes:
            raise VoiceJobExecutionError("artifact_output_too_large")
        digest = hashlib.sha256(data).hexdigest()
        clean_suffix = "".join(ch for ch in str(suffix).lower() if ch.isalnum()) or "bin"
        relative = Path(digest[:2]) / f"{digest}.{clean_suffix}"
        target = (self.policy.output_root / relative).resolve()
        if not self._inside(target, self.policy.output_root):
            raise VoiceJobExecutionError("artifact_output_path_invalid")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                fd, temporary = tempfile.mkstemp(
                    prefix=f".{digest}.", dir=str(target.parent)
                )
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
        except OSError as exc:
            raise VoiceJobExecutionError(
                "artifact_persistence_failed", retryable=True
            ) from exc
        stored = self._read_bounded(target)
        if stored != data or hashlib.sha256(stored).hexdigest() != digest:
            raise VoiceJobExecutionError("artifact_persistence_mismatch")
        cid = self._raw_sha256_cid(digest)
        return {
            "uri": f"ipfs://{cid}",
            "cid": cid,
            "sha256": digest,
            "size_bytes": len(stored),
            "media_type": str(media_type),
        }


def _job_mapping(job: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(job, Mapping):
        return dict(job)
    for method_name in ("to_payload", "to_dict"):
        method = getattr(job, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, Mapping):
                return dict(value)
    raise VoiceJobExecutionError("voice_job_mapping_required")


def _canonical_job(
    job: Mapping[str, Any] | Any,
    *,
    expected_task_type: str,
) -> tuple[VoiceJob, dict[str, Any]]:
    payload = _job_mapping(job)
    try:
        canonical_job = voice_job_from_payload(payload)
    except VoiceJobContractError as exc:
        raise VoiceJobExecutionError("voice_job_contract_invalid") from exc
    if canonical_job.task_type != expected_task_type:
        raise VoiceJobExecutionError(
            f"{expected_task_type.replace('.', '_').replace('-', '_')}_job_required"
        )
    return canonical_job, canonical_job.to_payload()


def _provider_receipt(job: Mapping[str, Any], *, latency_ms: int) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "provider": str(job.get("provider") or "default"),
        "latency_ms": max(0, int(latency_ms)),
        "attempt_count": 1,
    }
    model = str(job.get("model_name") or "").strip()
    if model:
        receipt["model"] = model
    version = str(job.get("provider_version") or "").strip()
    if version:
        receipt["provider_version"] = version
    return receipt


def _result(
    job: VoiceJob,
    *,
    artifact: Mapping[str, Any] | None,
    receipt: Mapping[str, Any],
    quality_metrics: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    artifacts = (
        (ArtifactDescriptor.from_dict(artifact),)
        if artifact is not None
        else ()
    )
    return VoiceJobResult.from_job(
        job,
        status="completed",
        artifacts=artifacts,
        provider_receipt=receipt,
        quality_metrics=quality_metrics or {},
    ).to_payload()


def _inspect_wav(data: bytes, policy: ArtifactPolicy) -> dict[str, int]:
    """Inspect WAV and always emit silence/clipping acoustic ratios.

    PCM16 mono/stereo WAV is inspected in-process. Other WAV encodings are
    rejected rather than silently omitting acoustic gates.
    """

    return _inspect_decoded_pcm_wav(data, policy)


def _decode_audio_with_ffmpeg(
    data: bytes,
    input_format: str,
    policy: ArtifactPolicy,
) -> bytes:
    """Decode one allowlisted compressed container to bounded PCM WAV.

    The input demuxer is fixed by the trusted MIME allowlist, the subprocess
    never invokes a shell, and both ffmpeg's output limit and a post-run size
    check constrain the temporary artifact.  The one-second duration
    overrun lets the caller distinguish an overlong input from a valid input
    exactly at the configured ceiling.
    """

    if input_format not in frozenset(_NON_WAV_INPUT_FORMATS.values()):
        raise VoiceJobExecutionError("audio_decoder_unsupported_media")
    if len(data) > policy.max_input_bytes:
        raise VoiceJobExecutionError("artifact_too_large")
    output_limit = policy.max_decoded_bytes + _FFMPEG_WAV_OVERHEAD_BYTES
    decode_limit_seconds = (
        policy.max_duration_ms + _FFMPEG_DURATION_OVERREAD_MS
    ) / 1000
    with tempfile.TemporaryDirectory(prefix="ipfs-voice-decode-") as directory:
        output_path = Path(directory) / "decoded.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format,
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c:a",
            "pcm_s16le",
            "-t",
            f"{decode_limit_seconds:.3f}",
            "-fs",
            str(output_limit),
            "-f",
            "wav",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                input=data,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=policy.decoder_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise VoiceJobExecutionError("audio_decoder_unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise VoiceJobExecutionError("audio_decode_timeout") from exc
        try:
            output_size = output_path.stat().st_size
        except OSError:
            output_size = 0
        if output_size >= output_limit:
            raise VoiceJobExecutionError("audio_decoded_too_large")
        if completed.returncode != 0 or output_size <= 0:
            raise VoiceJobExecutionError("audio_decode_failed")
        try:
            with output_path.open("rb") as handle:
                decoded = handle.read(output_limit + 1)
        except OSError as exc:
            raise VoiceJobExecutionError("audio_decode_failed") from exc
    if len(decoded) > output_limit:
        raise VoiceJobExecutionError("audio_decoded_too_large")
    return decoded


def _pcm16_acoustic_ratios(
    pcm: bytes,
    *,
    policy: ArtifactPolicy,
) -> tuple[int, int]:
    if len(pcm) % 2:
        raise VoiceJobExecutionError("audio_decode_failed")
    total = len(pcm) // 2
    if total == 0:
        return 0, 0
    max_amplitude = (1 << 15) - 1
    silence_limit = (
        max_amplitude * policy.silence_peak_threshold_bp
    ) // _BASIS_POINT_SCALE
    clipping_limit = (
        max_amplitude * policy.clipping_peak_threshold_bp
    ) // _BASIS_POINT_SCALE
    silent = 0
    clipped = 0
    for offset in range(0, len(pcm), 2):
        magnitude = abs(int.from_bytes(pcm[offset : offset + 2], "little", signed=True))
        if magnitude <= silence_limit:
            silent += 1
        if magnitude >= clipping_limit:
            clipped += 1
    return (
        int(round((silent / total) * _BASIS_POINT_SCALE)),
        int(round((clipped / total) * _BASIS_POINT_SCALE)),
    )


def _pcm16_trailing_silence_ms(
    pcm: bytes,
    *,
    channels: int,
    sample_rate: int,
    policy: ArtifactPolicy,
) -> int:
    """Return the contiguous silent PCM16 suffix duration in milliseconds.

    Silence is evaluated per interleaved frame: every channel in a frame must
    be at or below the configured peak threshold.  The ceiling conversion is
    consistent with the WAV duration metric and preserves sub-millisecond
    suffixes as a measurable millisecond.
    """

    frame_bytes = channels * 2
    if channels <= 0 or sample_rate <= 0 or len(pcm) % frame_bytes:
        raise VoiceJobExecutionError("audio_decode_failed")
    max_amplitude = (1 << 15) - 1
    silence_limit = (
        max_amplitude * policy.silence_peak_threshold_bp
    ) // _BASIS_POINT_SCALE
    trailing_frames = 0
    for frame_offset in range(len(pcm) - frame_bytes, -1, -frame_bytes):
        frame_is_silent = True
        for channel_offset in range(frame_offset, frame_offset + frame_bytes, 2):
            magnitude = abs(
                int.from_bytes(
                    pcm[channel_offset : channel_offset + 2],
                    "little",
                    signed=True,
                )
            )
            if magnitude > silence_limit:
                frame_is_silent = False
                break
        if not frame_is_silent:
            break
        trailing_frames += 1
    return (trailing_frames * 1000 + sample_rate - 1) // sample_rate


def _inspect_decoded_pcm_wav(
    data: bytes,
    policy: ArtifactPolicy,
) -> dict[str, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = int(audio.getnchannels())
            sample_rate = int(audio.getframerate())
            frames = int(audio.getnframes())
            sample_width = int(audio.getsampwidth())
            compression = str(audio.getcomptype())
            if channels <= 0 or sample_rate <= 0 or frames < 0:
                raise VoiceJobExecutionError("audio_metadata_invalid")
            if sample_width != 2 or compression != "NONE":
                raise VoiceJobExecutionError("audio_decode_failed")
            decoded_bytes = frames * channels * sample_width
            if decoded_bytes > policy.max_decoded_bytes:
                raise VoiceJobExecutionError("audio_decoded_too_large")
            pcm = audio.readframes(frames)
    except VoiceJobExecutionError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise VoiceJobExecutionError("audio_decode_failed") from exc
    if len(pcm) != decoded_bytes:
        raise VoiceJobExecutionError("audio_decode_failed")
    duration_ms = (frames * 1000 + sample_rate - 1) // sample_rate
    if duration_ms > policy.max_duration_ms:
        raise VoiceJobExecutionError("audio_duration_exceeded")
    silence_ratio_bp, clipping_ratio_bp = _pcm16_acoustic_ratios(
        pcm,
        policy=policy,
    )
    trailing_silence_ms = _pcm16_trailing_silence_ms(
        pcm,
        channels=channels,
        sample_rate=sample_rate,
        policy=policy,
    )
    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "duration_ms": duration_ms,
        "decoded_bytes": decoded_bytes,
        "silence_ratio_bp": silence_ratio_bp,
        "clipping_ratio_bp": clipping_ratio_bp,
        "trailing_silence_ms": trailing_silence_ms,
    }


def _audio_metrics(
    data: bytes,
    descriptor: Mapping[str, Any],
    policy: ArtifactPolicy,
    *,
    audio_decoder_fn: AudioDecoder | None = None,
) -> dict[str, int]:
    if len(data) > policy.max_input_bytes:
        raise VoiceJobExecutionError("artifact_too_large")
    media_type = str(descriptor.get("media_type") or "").lower().split(";", 1)[0].strip()
    uri = str(descriptor.get("uri") or "").lower()
    # Prefer declared media type. Only fall back to URI suffix when media is
    # absent so a mislabeled non-WAV URI ending in ".wav" cannot skip decode.
    is_wav_media = media_type in {"audio/wav", "audio/x-wav", "audio/wave"}
    if is_wav_media or (not media_type and uri.endswith(".wav")):
        return _inspect_wav(data, policy)
    if not media_type.startswith("audio/"):
        raise VoiceJobExecutionError("artifact_media_type_not_audio")
    input_format = _NON_WAV_INPUT_FORMATS.get(media_type)
    if input_format is None:
        raise VoiceJobExecutionError("audio_decoder_unsupported_media")
    decoder = audio_decoder_fn or _decode_audio_with_ffmpeg
    try:
        decoded = decoder(data, input_format, policy)
    except VoiceJobExecutionError:
        raise
    except Exception as exc:
        raise VoiceJobExecutionError("audio_decode_failed") from exc
    if not isinstance(decoded, bytes):
        raise VoiceJobExecutionError("audio_decode_failed")
    metrics = _inspect_decoded_pcm_wav(decoded, policy)
    return {"encoded_bytes": len(data), **metrics}


def execute_voice_tts_job(
    job: Mapping[str, Any] | Any,
    *,
    resolver: ArtifactResolver | None = None,
    text_to_speech_fn: Callable[..., Any] | None = None,
    audio_decoder_fn: AudioDecoder | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    canonical_job, payload = _canonical_job(
        job,
        expected_task_type="voice.tts",
    )
    spoken_text = payload.get("spoken_text")
    if not isinstance(spoken_text, str) or not spoken_text.strip():
        raise VoiceJobExecutionError("spoken_text_required")
    active_resolver = resolver or ArtifactResolver()
    if text_to_speech_fn is None:
        from ..voice_router import text_to_speech as text_to_speech_fn

    options = payload.get("generation_settings")
    kwargs = dict(options) if isinstance(options, Mapping) else {}
    started = clock()
    try:
        audio = text_to_speech_fn(
            spoken_text,
            voice=payload.get("voice") or None,
            model_name=payload.get("model_name") or None,
            device=payload.get("device") or None,
            output_format=payload.get("codec")
            or payload.get("output_format")
            or None,
            provider=payload.get("provider") or None,
            **kwargs,
        )
    except VoiceJobExecutionError:
        raise
    except Exception as exc:
        raise VoiceJobExecutionError("tts_provider_failed", retryable=True) from exc
    if inspect.isawaitable(audio):
        if inspect.iscoroutine(audio):
            audio.close()
        raise VoiceJobExecutionError("voice_provider_async_result")
    if isinstance(audio, str):
        # The router may return a path only when explicitly asked to write one;
        # durable handlers never request that unsafe form.
        raise VoiceJobExecutionError("voice_provider_returned_path")
    if not isinstance(audio, bytes):
        raise VoiceJobExecutionError("voice_provider_invalid_audio")
    codec = str(payload.get("codec") or payload.get("output_format") or "wav").lower()
    media_type = "audio/mpeg" if codec in {"mp3", "mpeg"} else f"audio/{codec}"
    metrics = _audio_metrics(
        audio,
        {"media_type": media_type, "uri": f"output.{codec}"},
        active_resolver.policy,
        audio_decoder_fn=audio_decoder_fn,
    )
    artifact = active_resolver.persist(audio, suffix=codec, media_type=media_type)
    latency_ms = round(max(0.0, clock() - started) * 1000)
    return _result(
        canonical_job,
        artifact=artifact,
        receipt=_provider_receipt(payload, latency_ms=latency_ms),
        quality_metrics={key: int(value) for key, value in metrics.items()},
    )


def execute_voice_asr_job(
    job: Mapping[str, Any] | Any,
    *,
    resolver: ArtifactResolver | None = None,
    speech_to_text_fn: Callable[..., Any] | None = None,
    audio_decoder_fn: AudioDecoder | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    canonical_job, payload = _canonical_job(
        job,
        expected_task_type="voice.asr",
    )
    active_resolver = resolver or ArtifactResolver()
    audio, descriptor = active_resolver.resolve_source(payload)
    metrics = _audio_metrics(
        audio,
        descriptor,
        active_resolver.policy,
        audio_decoder_fn=audio_decoder_fn,
    )
    if speech_to_text_fn is None:
        from ..voice_router import speech_to_text as speech_to_text_fn

    options = payload.get("decoding_settings")
    kwargs = dict(options) if isinstance(options, Mapping) else {}
    started = clock()
    try:
        transcript = speech_to_text_fn(
            audio,
            model_name=payload.get("model_name") or None,
            language=payload.get("locale") or payload.get("language") or None,
            device=payload.get("device") or None,
            provider=payload.get("provider") or None,
            **kwargs,
        )
    except VoiceJobExecutionError:
        raise
    except Exception as exc:
        raise VoiceJobExecutionError("asr_provider_failed", retryable=True) from exc
    if inspect.isawaitable(transcript):
        if inspect.iscoroutine(transcript):
            transcript.close()
        raise VoiceJobExecutionError("voice_provider_async_result")
    if not isinstance(transcript, str):
        raise VoiceJobExecutionError("voice_provider_invalid_transcript")
    transcript_bytes = transcript.encode("utf-8")
    artifact = None
    retention_policy = str(payload.get("retention_policy") or "none")
    if retention_policy in {"result", "publication"}:
        artifact = active_resolver.persist(
            transcript_bytes,
            suffix="txt",
            media_type="text/plain;charset=utf-8",
        )
    latency_ms = round(max(0.0, clock() - started) * 1000)
    receipt = _provider_receipt(payload, latency_ms=latency_ms)
    if artifact is None:
        # Non-retained ASR intentionally emits no transcript artifact. Preserve
        # only its privacy-safe digest in the contract-approved response hash.
        receipt["response_id_sha256"] = hashlib.sha256(transcript_bytes).hexdigest()
    result = _result(
        canonical_job,
        artifact=artifact,
        receipt=receipt,
        quality_metrics={**{key: int(value) for key, value in metrics.items()}, "transcript_bytes": len(transcript_bytes)},
    )
    return result


def execute_voice_audio_validation_job(
    job: Mapping[str, Any] | Any,
    *,
    resolver: ArtifactResolver | None = None,
    audio_decoder_fn: AudioDecoder | None = None,
) -> dict[str, Any]:
    canonical_job, payload = _canonical_job(
        job,
        expected_task_type="voice.audio-validate",
    )
    active_resolver = resolver or ArtifactResolver()
    audio, descriptor = active_resolver.resolve_source(payload)
    metrics = _audio_metrics(
        audio,
        descriptor,
        active_resolver.policy,
        audio_decoder_fn=audio_decoder_fn,
    )
    policy = payload.get("validation_policy")
    if isinstance(policy, Mapping):
        minimum = policy.get("minimum_duration_ms")
        maximum = policy.get("maximum_duration_ms")
        duration = metrics.get("duration_ms")
        if isinstance(minimum, int) and duration is not None and duration < minimum:
            raise VoiceJobExecutionError("audio_duration_below_policy")
        if isinstance(maximum, int) and duration is not None and duration > maximum:
            raise VoiceJobExecutionError("audio_duration_above_policy")
    return _result(
        canonical_job,
        artifact=dict(descriptor),
        receipt=_provider_receipt(payload, latency_ms=0),
        quality_metrics={key: int(value) for key, value in metrics.items()},
    )


def _queue_job(task: Mapping[str, Any]) -> dict[str, Any]:
    nested = task.get("payload")
    job = dict(nested) if isinstance(nested, Mapping) else dict(task)
    for key in ("task_type", "task_id", "model_name"):
        if not job.get(key) and task.get(key) is not None:
            job[key] = task.get(key)
    return job


def execute_task(
    task: Mapping[str, Any],
    *,
    resolver: ArtifactResolver | None = None,
    text_to_speech_fn: Callable[..., Any] | None = None,
    speech_to_text_fn: Callable[..., Any] | None = None,
    audio_decoder_fn: AudioDecoder | None = None,
) -> dict[str, Any]:
    """Execute one canonical voice job from a queue row or bare payload."""

    if not isinstance(task, Mapping):
        raise VoiceJobExecutionError("voice_task_mapping_required")
    job = _queue_job(task)
    task_type = canonical_task_type(job.get("task_type"))
    if task_type not in VOICE_TASK_TYPES:
        raise VoiceJobExecutionError("unsupported_voice_task_type")
    job["task_type"] = task_type
    if task_type == "voice.tts":
        return execute_voice_tts_job(
            job,
            resolver=resolver,
            text_to_speech_fn=text_to_speech_fn,
            audio_decoder_fn=audio_decoder_fn,
        )
    if task_type == "voice.asr":
        return execute_voice_asr_job(
            job,
            resolver=resolver,
            speech_to_text_fn=speech_to_text_fn,
            audio_decoder_fn=audio_decoder_fn,
        )
    return execute_voice_audio_validation_job(
        job,
        resolver=resolver,
        audio_decoder_fn=audio_decoder_fn,
    )


__all__ = [
    "ArtifactPolicy",
    "ArtifactResolver",
    "VoiceJobExecutionError",
    "execute_task",
    "execute_voice_asr_job",
    "execute_voice_audio_validation_job",
    "execute_voice_tts_job",
]
