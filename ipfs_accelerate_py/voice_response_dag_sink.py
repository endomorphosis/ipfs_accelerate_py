"""Durable local staging for validated voice response-DAG cache misses.

The runtime owns the queue while :mod:`ipfs_datasets_py.voice.response_dag`
owns the candidate contract.  This module deliberately has no remote client,
publisher, or credential surface: it can only append immutable local records.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

LOCAL_RESPONSE_DAG_QUEUE_SCHEMA_VERSION = "abby_voice_response_dag_local_queue_v1"
INDEPENDENT_VOICE_VALIDATION_RECEIPT_SCHEMA_VERSION = (
    "abby_voice_independent_validation_receipt_v1"
)

_PRIVATE_QUEUE_KEYS = frozenset(
    {
        "audio_base64",
        "caller_audio",
        "caller_transcript",
        "call_id",
        "input",
        "input_audio",
        "input_audio_path",
        "input_text",
        "prompt",
        "raw_audio",
        "request_id",
        "session_id",
        "transcript",
        "user_prompt",
    }
)
_SECRET_KEY_MARKERS = (
    "access_key",
    "api_key",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "private_key",
    "secret",
    "signature",
    "token",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
)
_SAFE_RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SURFACE_ALIASES = {
    "web": "website",
    "website": "website",
    "sip": "telephone",
    "telephone": "telephone",
    "telephony": "telephone",
    "twilio": "telephone",
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


class LocalResponseDAGQueueError(RuntimeError):
    """A response-DAG candidate could not be staged without data loss."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalResponseDAGQueueError(
            "response-DAG queue records must be deterministic JSON"
        ) from exc


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_thaw_json(item) for item in value]
    return value


def _assert_privacy_safe(value: Any, *, path: str = "candidate") -> None:
    if isinstance(value, bytes | bytearray | memoryview):
        raise LocalResponseDAGQueueError(f"{path} must not contain raw bytes")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            digest_only = normalized.endswith("_sha256")
            if normalized in _PRIVATE_QUEUE_KEYS and not digest_only:
                raise LocalResponseDAGQueueError(
                    f"{path}.{key} must not contain private turn input"
                )
            if any(marker in normalized for marker in _SECRET_KEY_MARKERS):
                raise LocalResponseDAGQueueError(
                    f"{path}.{key} must not contain credentials"
                )
            _assert_privacy_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            _assert_privacy_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise LocalResponseDAGQueueError(
                f"{path} appears to contain credential material"
            )
        return
    if value is None or isinstance(value, int | float | bool):
        return
    raise LocalResponseDAGQueueError(
        f"{path} must contain only deterministic JSON values"
    )


def _response_dag_contracts() -> tuple[Any, Any]:
    try:
        from ipfs_datasets_py.voice.response_dag import (
            ResponseDAGAppendCandidate,
            append_response_dag_candidate,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "ipfs_datasets_py.voice.response_dag":
            raise LocalResponseDAGQueueError(
                "ipfs_datasets_py.voice.response_dag could not load"
            ) from exc
        from ._voice_response_dag_compat import (
            ResponseDAGAppendCandidate,
            append_response_dag_candidate,
        )
    except ImportError as exc:
        raise LocalResponseDAGQueueError(
            "ipfs_datasets_py.voice.response_dag could not load"
        ) from exc
    return ResponseDAGAppendCandidate, append_response_dag_candidate


def _validated_candidate(value: Any) -> Any:
    candidate_type, _ = _response_dag_contracts()
    if isinstance(value, candidate_type):
        candidate = value
    else:
        to_dict = getattr(value, "to_dict", None)
        raw = to_dict() if callable(to_dict) else value
        if not isinstance(raw, Mapping):
            raise LocalResponseDAGQueueError(
                "candidate must be a ResponseDAGAppendCandidate or mapping"
            )
        try:
            candidate = candidate_type(
                cache_miss_event_id=raw.get("cache_miss_event_id", ""),
                validation_receipt_id=raw.get("validation_receipt_id", ""),
                nodes=tuple(raw.get("nodes") or ()),
                edges=tuple(raw.get("edges") or ()),
                rendered_text_sha256=raw.get("rendered_text_sha256", ""),
                output_audio_sha256=raw.get("output_audio_sha256", ""),
                schema_version=raw.get("schema_version", ""),
                candidate_id=raw.get("candidate_id", ""),
                metadata=raw.get("metadata") or {},
            )
        except Exception as exc:
            raise LocalResponseDAGQueueError(
                f"invalid response-DAG candidate: {exc}"
            ) from exc
    payload = _thaw_json(candidate.to_dict())
    if payload.get("append_only") is not True:
        raise LocalResponseDAGQueueError(
            "response-DAG candidate must be append-only"
        )
    validation_id = str(payload.get("validation_receipt_id") or "")
    if not _SAFE_RECEIPT_ID_RE.fullmatch(validation_id):
        raise LocalResponseDAGQueueError(
            "validation_receipt_id must be an opaque, non-private identifier"
        )
    _assert_privacy_safe(payload)
    return candidate


@dataclass(frozen=True, slots=True)
class IndependentVoiceValidationReceipt:
    """Content-bound proof that a separate validator accepted live TTS.

    The compact receipt intentionally excludes the validator transcript and
    raw audio.  It binds the independent decision to the rendered text and
    exact output bytes that will be represented by the response-DAG candidate.
    """

    validation_receipt_id: str
    rendered_text_sha256: str
    output_audio_sha256: str
    validator_identity: str
    validation_method: str = "asr_round_trip"
    passed: bool = True
    schema_version: str = INDEPENDENT_VOICE_VALIDATION_RECEIPT_SCHEMA_VERSION
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        receipt_id = str(self.validation_receipt_id or "").strip()
        if not _SAFE_RECEIPT_ID_RE.fullmatch(receipt_id):
            raise LocalResponseDAGQueueError(
                "validation_receipt_id must be an opaque, non-private identifier"
            )
        rendered_digest = str(self.rendered_text_sha256 or "").strip().casefold()
        audio_digest = str(self.output_audio_sha256 or "").strip().casefold()
        if not _SHA256_RE.fullmatch(rendered_digest):
            raise LocalResponseDAGQueueError(
                "validation receipt rendered_text_sha256 must be a full "
                "lowercase SHA-256"
            )
        if not _SHA256_RE.fullmatch(audio_digest):
            raise LocalResponseDAGQueueError(
                "validation receipt output_audio_sha256 must be a full "
                "lowercase SHA-256"
            )
        validator = str(self.validator_identity or "").strip()
        method = str(self.validation_method or "").strip().casefold()
        if not validator:
            raise LocalResponseDAGQueueError(
                "independent validation requires validator_identity"
            )
        if not _SAFE_RECEIPT_ID_RE.fullmatch(validator):
            raise LocalResponseDAGQueueError(
                "validator_identity must be an opaque, non-private identifier"
            )
        if not method or not re.fullmatch(r"[a-z][a-z0-9._+-]{0,63}", method):
            raise LocalResponseDAGQueueError(
                "validation_method must be a stable machine identifier"
            )
        if self.passed is not True:
            raise LocalResponseDAGQueueError(
                "response-DAG staging requires an explicitly passed "
                "independent validation receipt"
            )
        if self.schema_version != INDEPENDENT_VOICE_VALIDATION_RECEIPT_SCHEMA_VERSION:
            raise LocalResponseDAGQueueError(
                f"unsupported independent validation receipt schema: "
                f"{self.schema_version}"
            )
        object.__setattr__(self, "validation_receipt_id", receipt_id)
        object.__setattr__(self, "rendered_text_sha256", rendered_digest)
        object.__setattr__(self, "output_audio_sha256", audio_digest)
        object.__setattr__(self, "validator_identity", validator)
        object.__setattr__(self, "validation_method", method)
        computed = sha256(_canonical_bytes(self.identity_dict())).hexdigest()
        supplied = str(self.receipt_sha256 or "").strip().casefold()
        if supplied and supplied != computed:
            raise LocalResponseDAGQueueError(
                "receipt_sha256 does not match independent validation content"
            )
        object.__setattr__(self, "receipt_sha256", computed)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "output_audio_sha256": self.output_audio_sha256,
            "passed": self.passed,
            "rendered_text_sha256": self.rendered_text_sha256,
            "schema_version": self.schema_version,
            "validation_method": self.validation_method,
            "validation_receipt_id": self.validation_receipt_id,
            "validator_identity": self.validator_identity,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_value(cls, value: Any) -> IndependentVoiceValidationReceipt:
        if isinstance(value, cls):
            return value
        to_dict = getattr(value, "to_dict", None)
        raw = to_dict() if callable(to_dict) else value
        if not isinstance(raw, Mapping):
            raise LocalResponseDAGQueueError(
                "validation_receipt must be an "
                "IndependentVoiceValidationReceipt or mapping"
            )
        _assert_privacy_safe(raw, path="validation_receipt")
        try:
            return cls(
                validation_receipt_id=raw.get("validation_receipt_id", ""),
                rendered_text_sha256=raw.get("rendered_text_sha256", ""),
                output_audio_sha256=raw.get("output_audio_sha256", ""),
                validator_identity=raw.get("validator_identity", ""),
                validation_method=raw.get(
                    "validation_method", "asr_round_trip"
                ),
                passed=raw.get("passed", False),
                schema_version=raw.get(
                    "schema_version",
                    INDEPENDENT_VOICE_VALIDATION_RECEIPT_SCHEMA_VERSION,
                ),
                receipt_sha256=raw.get("receipt_sha256", ""),
            )
        except LocalResponseDAGQueueError:
            raise
        except Exception as exc:
            raise LocalResponseDAGQueueError(
                f"invalid independent validation receipt: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class LocalValidatedVoiceCacheMissArtifacts:
    """Post-synthesis, locally persisted inputs for durable DAG staging."""

    validation_receipt: IndependentVoiceValidationReceipt | Mapping[str, Any]
    audio_descriptor: Mapping[str, Any]
    response_id: str = ""
    remote_writes: bool = False

    def __post_init__(self) -> None:
        receipt = IndependentVoiceValidationReceipt.from_value(
            self.validation_receipt
        )
        if not isinstance(self.audio_descriptor, Mapping):
            raise LocalResponseDAGQueueError(
                "local cache-miss audio_descriptor must be a mapping"
            )
        descriptor = _thaw_json(self.audio_descriptor)
        _assert_privacy_safe(descriptor, path="audio_descriptor")
        response_id = str(self.response_id or "").strip()
        if response_id and not _SAFE_RECEIPT_ID_RE.fullmatch(response_id):
            raise LocalResponseDAGQueueError(
                "response_id must be an opaque, non-private identifier"
            )
        if self.remote_writes is not False:
            raise LocalResponseDAGQueueError(
                "post-synthesis cache-miss artifacts must be local-only"
            )
        object.__setattr__(self, "validation_receipt", receipt)
        object.__setattr__(self, "audio_descriptor", descriptor)
        object.__setattr__(self, "response_id", response_id)

    @classmethod
    def from_value(cls, value: Any) -> LocalValidatedVoiceCacheMissArtifacts:
        if isinstance(value, cls):
            return value
        to_dict = getattr(value, "to_dict", None)
        raw = to_dict() if callable(to_dict) else value
        if not isinstance(raw, Mapping):
            raise LocalResponseDAGQueueError(
                "post-synthesis validator must return "
                "LocalValidatedVoiceCacheMissArtifacts, a mapping, or None"
            )
        _assert_privacy_safe(raw, path="post_synthesis_validation")
        return cls(
            validation_receipt=raw.get("validation_receipt") or {},
            audio_descriptor=raw.get("audio_descriptor") or {},
            response_id=raw.get("response_id", ""),
            remote_writes=raw.get("remote_writes", False),
        )

    def to_dict(self) -> dict[str, Any]:
        receipt = IndependentVoiceValidationReceipt.from_value(
            self.validation_receipt
        )
        return {
            "audio_descriptor": _thaw_json(self.audio_descriptor),
            "remote_writes": self.remote_writes,
            "response_id": self.response_id,
            "validation_receipt": receipt.to_dict(),
        }


def _candidate_envelope(candidate: Any) -> tuple[dict[str, Any], bytes]:
    payload = _thaw_json(candidate.to_dict())
    candidate_bytes = _canonical_bytes(payload)
    envelope = {
        "candidate": payload,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": sha256(candidate_bytes).hexdigest(),
        "publication_status": "local_pending",
        "remote_writes": False,
        "schema_version": LOCAL_RESPONSE_DAG_QUEUE_SCHEMA_VERSION,
    }
    body = _canonical_bytes(envelope) + b"\n"
    return envelope, body


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise LocalResponseDAGQueueError(
            f"response-DAG queue directory must not be a symlink: {path}"
        )
    created = False
    if not path.exists():
        try:
            path.mkdir(parents=True, mode=0o700)
            created = True
        except FileExistsError:
            # Another website/telephone worker may initialize the same queue.
            pass
    if path.is_symlink() or not path.is_dir():
        raise LocalResponseDAGQueueError(
            f"response-DAG queue path is not a private directory: {path}"
        )
    if created:
        os.chmod(path, 0o700)
        _fsync_directory(path.parent)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise LocalResponseDAGQueueError(
            f"response-DAG queue directory must be private (0700): {path}"
        )


@dataclass(frozen=True, slots=True)
class LocalResponseDAGQueueReceipt:
    """Privacy-safe receipt for one local append attempt."""

    candidate_id: str
    candidate_sha256: str
    payload_sha256: str
    relative_path: str
    status: str
    publication_status: str = "local_pending"
    remote_writes: bool = False
    schema_version: str = LOCAL_RESPONSE_DAG_QUEUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"appended", "duplicate"}:
            raise LocalResponseDAGQueueError(
                "queue receipt status must be appended or duplicate"
            )
        if self.remote_writes is not False:
            raise LocalResponseDAGQueueError(
                "the local response-DAG queue cannot perform remote writes"
            )

    @property
    def appended(self) -> bool:
        return self.status == "appended"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "payload_sha256": self.payload_sha256,
            "publication_status": self.publication_status,
            "relative_path": self.relative_path,
            "remote_writes": self.remote_writes,
            "schema_version": self.schema_version,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class QueuedVoiceCacheMissCandidate:
    """Event, data-contract candidate, and durable local queue receipt."""

    event: Any
    candidate: Any
    receipt: LocalResponseDAGQueueReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_miss_event_id": self.event.event_id,
            "candidate_id": self.candidate.candidate_id,
            "queue": self.receipt.to_dict(),
        }


class LocalResponseDAGQueue:
    """Content-addressed, append-only queue with crash-safe local publication."""

    def __init__(self, root: str | Path) -> None:
        if isinstance(root, str) and "://" in root:
            raise LocalResponseDAGQueueError(
                "response-DAG queue root must be a local filesystem path"
            )
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise LocalResponseDAGQueueError(
                "response-DAG queue root must not be a symlink"
            )
        self.root = requested.resolve()
        _ensure_private_directory(self.root)
        self._candidate_root = self.root / "candidates"
        self._staging_root = self.root / ".staging"
        _ensure_private_directory(self._candidate_root)
        _ensure_private_directory(self._staging_root)

    @staticmethod
    def _filename(candidate_id: str) -> str:
        return sha256(candidate_id.encode("utf-8")).hexdigest() + ".json"

    def _target(self, candidate_id: str) -> Path:
        return self._candidate_root / self._filename(candidate_id)

    def append(self, candidate: Any) -> LocalResponseDAGQueueReceipt:
        """Durably append one validated candidate or return a duplicate receipt."""

        candidate = _validated_candidate(candidate)
        envelope, body = _candidate_envelope(candidate)
        target = self._target(candidate.candidate_id)
        relative_path = target.relative_to(self.root).as_posix()
        temporary = self._staging_root / (
            f"{self._filename(candidate.candidate_id)}."
            f"{os.getpid()}.{uuid.uuid4().hex}.partial"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target, follow_symlinks=False)
                _fsync_directory(self._candidate_root)
                status = "appended"
            except FileExistsError:
                if target.is_symlink() or not target.is_file():
                    raise LocalResponseDAGQueueError(
                        "append-only candidate target is not a regular file"
                    )
                existing = target.read_bytes()
                if existing != body:
                    raise LocalResponseDAGQueueError(
                        "append-only candidate target contains conflicting bytes"
                    )
                status = "duplicate"
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
                _fsync_directory(self._staging_root)
            except FileNotFoundError:
                pass
        return LocalResponseDAGQueueReceipt(
            candidate_id=candidate.candidate_id,
            candidate_sha256=envelope["candidate_sha256"],
            payload_sha256=sha256(body).hexdigest(),
            relative_path=relative_path,
            status=status,
        )

    def load(self, candidate_id: str) -> Any:
        """Load and fully revalidate one immutable queued candidate."""

        normalized = str(candidate_id or "").strip()
        if not normalized:
            raise LocalResponseDAGQueueError("candidate_id must not be empty")
        target = self._target(normalized)
        if target.is_symlink() or not target.is_file():
            raise LocalResponseDAGQueueError(
                f"queued response-DAG candidate not found: {normalized}"
            )
        mode = stat.S_IMODE(target.stat().st_mode)
        if mode & 0o077:
            raise LocalResponseDAGQueueError(
                "queued response-DAG candidate file must be private (0600)"
            )
        try:
            envelope = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalResponseDAGQueueError(
                "queued response-DAG candidate is not valid JSON"
            ) from exc
        if (
            not isinstance(envelope, Mapping)
            or envelope.get("schema_version")
            != LOCAL_RESPONSE_DAG_QUEUE_SCHEMA_VERSION
            or envelope.get("publication_status") != "local_pending"
            or envelope.get("remote_writes") is not False
        ):
            raise LocalResponseDAGQueueError(
                "queued response-DAG envelope failed the local-only contract"
            )
        raw_candidate = envelope.get("candidate")
        if not isinstance(raw_candidate, Mapping):
            raise LocalResponseDAGQueueError(
                "queued response-DAG envelope lacks a candidate"
            )
        candidate = _validated_candidate(raw_candidate)
        candidate_sha = sha256(_canonical_bytes(candidate.to_dict())).hexdigest()
        if (
            envelope.get("candidate_id") != candidate.candidate_id
            or envelope.get("candidate_sha256") != candidate_sha
            or candidate.candidate_id != normalized
            or target.name != self._filename(candidate.candidate_id)
        ):
            raise LocalResponseDAGQueueError(
                "queued response-DAG candidate failed integrity validation"
            )
        return candidate

    def candidate_ids(self) -> tuple[str, ...]:
        """Return all fully validated pending candidate IDs."""

        result = []
        for path in sorted(self._candidate_root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise LocalResponseDAGQueueError(
                    "response-DAG queue contains a non-regular candidate"
                )
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LocalResponseDAGQueueError(
                    f"response-DAG queue contains invalid JSON: {path.name}"
                ) from exc
            candidate_id = str(
                envelope.get("candidate_id") if isinstance(envelope, Mapping) else ""
            )
            self.load(candidate_id)
            result.append(candidate_id)
        return tuple(sorted(result))

    def __len__(self) -> int:
        return len(self.candidate_ids())


def _derived_slot_bindings(result: Any) -> dict[str, Any]:
    provenance = getattr(result, "provenance", None)
    evidence = tuple(getattr(provenance, "evidence", ()) or ())
    evidence_cids = {
        str(getattr(item, "source_id", "") or ""): str(
            getattr(item, "cid", "") or getattr(item, "source_id", "") or ""
        )
        for item in evidence
    }
    bindings: dict[str, Any] = {}
    for slot in tuple(getattr(provenance, "grounded_slots", ()) or ()):
        source_cids = [
            evidence_cids.get(str(source_id), str(source_id))
            for source_id in tuple(getattr(slot, "source_ids", ()) or ())
            if str(source_id)
        ]
        bindings[str(getattr(slot, "name", "") or "")] = {
            "source_cids": source_cids,
            "value": getattr(slot, "value", None),
        }
    return bindings


def _surface_metadata(result: Any, surface: str) -> dict[str, str]:
    normalized = str(surface or "").strip().casefold()
    provenance = getattr(result, "provenance", None)
    provenance_metadata = getattr(provenance, "metadata", {})
    if not normalized and isinstance(provenance_metadata, Mapping):
        normalized = str(provenance_metadata.get("surface") or "").casefold()
        if not normalized and isinstance(provenance_metadata.get("telephone"), Mapping):
            normalized = "telephone"
    if not normalized:
        return {}
    try:
        return {"surface": _SURFACE_ALIASES[normalized]}
    except KeyError as exc:
        raise LocalResponseDAGQueueError(
            f"unsupported voice surface for response-DAG queue: {normalized}"
        ) from exc


def _stable_audio_descriptor(
    result: Any,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a caller-persisted descriptor against the returned audio."""

    descriptor = dict(value)
    _assert_privacy_safe(descriptor, path="audio_descriptor")
    audio = getattr(result, "audio", None)
    if not isinstance(audio, bytes) or not audio:
        raise LocalResponseDAGQueueError(
            "validated cache miss must contain non-empty output audio"
        )
    actual_audio_sha = sha256(audio).hexdigest()
    supplied_audio_sha = str(
        descriptor.get("content_sha256") or ""
    ).strip().casefold()
    if not _SHA256_RE.fullmatch(supplied_audio_sha):
        raise LocalResponseDAGQueueError(
            "stable audio_descriptor.content_sha256 must be an explicit "
            "full lowercase SHA-256"
        )
    if supplied_audio_sha != actual_audio_sha:
        raise LocalResponseDAGQueueError(
            "audio_descriptor.content_sha256 does not match returned audio"
        )
    supplied_byte_length = descriptor.get("byte_length")
    if (
        isinstance(supplied_byte_length, bool)
        or not isinstance(supplied_byte_length, int)
        or supplied_byte_length <= 0
    ):
        raise LocalResponseDAGQueueError(
            "stable audio_descriptor.byte_length must be an explicit "
            "positive integer"
        )
    if supplied_byte_length != len(audio):
        raise LocalResponseDAGQueueError(
            "audio_descriptor.byte_length does not match returned audio"
        )
    supplied_media_type = str(
        descriptor.get("media_type")
        or descriptor.get("mime_type")
        or ""
    ).strip().casefold()
    if not supplied_media_type.startswith("audio/"):
        raise LocalResponseDAGQueueError(
            "stable audio_descriptor.media_type must be explicit audio/*"
        )
    audio_format = str(getattr(result, "audio_format", "") or "").casefold()
    expected_media_type = (
        _AUDIO_MEDIA_TYPES.get(audio_format, f"audio/{audio_format}")
        if audio_format
        else ""
    )
    if expected_media_type and supplied_media_type != expected_media_type:
        raise LocalResponseDAGQueueError(
            "audio_descriptor.media_type does not match returned audio format"
        )
    if not str(descriptor.get("uri") or "").strip() and not str(
        descriptor.get("ipfs_cid") or ""
    ).strip():
        raise LocalResponseDAGQueueError(
            "stable audio descriptor requires an external uri or ipfs_cid"
        )
    uri = str(descriptor.get("uri") or "").strip()
    if uri:
        parsed_uri = urlsplit(uri)
        scheme = parsed_uri.scheme.casefold()
        if not scheme:
            raise LocalResponseDAGQueueError(
                "audio_descriptor.uri must be an absolute URI"
            )
        if parsed_uri.username is not None or parsed_uri.password is not None:
            raise LocalResponseDAGQueueError(
                "audio_descriptor.uri must not contain credentials"
            )
        for encoded_fields in (parsed_uri.query, parsed_uri.fragment):
            for query_key, _query_value in parse_qsl(
                encoded_fields,
                keep_blank_values=True,
            ):
                normalized_key = query_key.casefold().replace("-", "_")
                if any(
                    marker in normalized_key
                    for marker in _SECRET_KEY_MARKERS
                ):
                    raise LocalResponseDAGQueueError(
                        "audio_descriptor.uri must not contain credentials"
                    )
        if scheme == "file":
            if parsed_uri.netloc not in {"", "localhost"}:
                raise LocalResponseDAGQueueError(
                    "file audio_descriptor.uri must identify a local file"
                )
            local_path = Path(unquote(parsed_uri.path))
            if local_path.is_symlink() or not local_path.is_file():
                raise LocalResponseDAGQueueError(
                    "file audio_descriptor.uri must identify an existing "
                    "regular file"
                )
            persisted_digest = sha256()
            persisted_size = 0
            try:
                with local_path.open("rb") as persisted_audio:
                    for chunk in iter(
                        lambda: persisted_audio.read(1024 * 1024),
                        b"",
                    ):
                        persisted_digest.update(chunk)
                        persisted_size += len(chunk)
            except OSError as exc:
                raise LocalResponseDAGQueueError(
                    "file audio_descriptor.uri could not be read"
                ) from exc
            if (
                persisted_digest.hexdigest() != actual_audio_sha
                or persisted_size != len(audio)
            ):
                raise LocalResponseDAGQueueError(
                    "file audio_descriptor.uri does not contain returned audio"
                )
        elif scheme == "hf":
            if not re.search(r"@[0-9a-f]{40}(?:/|$)", uri):
                raise LocalResponseDAGQueueError(
                    "hf audio_descriptor.uri must use an immutable commit SHA"
                )
        elif (
            scheme in {"http", "https"}
            and (parsed_uri.hostname or "").casefold()
            in {"hf.co", "huggingface.co", "www.huggingface.co"}
            and not re.search(
                r"/resolve/[0-9a-f]{40}(?:/|$)",
                parsed_uri.path,
            )
        ):
            raise LocalResponseDAGQueueError(
                "Hugging Face audio_descriptor.uri must use an immutable "
                "commit SHA"
            )
    descriptor["content_sha256"] = supplied_audio_sha
    descriptor["byte_length"] = supplied_byte_length
    descriptor["media_type"] = supplied_media_type
    descriptor.pop("mime_type", None)
    return descriptor


def enqueue_validated_cache_miss_candidate(
    result: Any,
    *,
    sink: LocalResponseDAGQueue,
    validation_receipt: IndependentVoiceValidationReceipt | Mapping[str, Any] | None,
    audio_descriptor: Mapping[str, Any],
    response_id: str = "",
    template_text: str = "",
    slot_bindings: Mapping[str, Any] | None = None,
    surface: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> QueuedVoiceCacheMissCandidate | None:
    """Convert one validated live-TTS miss into one durable local DAG append.

    Cache hits return ``None``. The caller must supply a content-bound,
    independent audio-validation receipt and a stable external audio
    descriptor. This function never uploads audio or contacts the descriptor
    URI.
    """

    if not isinstance(sink, LocalResponseDAGQueue):
        raise TypeError("sink must be a LocalResponseDAGQueue")
    if not isinstance(audio_descriptor, Mapping):
        raise TypeError("audio_descriptor must be a mapping")
    if validation_receipt is None:
        raise LocalResponseDAGQueueError(
            "an explicit independent validation_receipt is required"
        )
    receipt = IndependentVoiceValidationReceipt.from_value(validation_receipt)
    event_builder = getattr(result, "validated_cache_miss_event", None)
    if not callable(event_builder):
        raise TypeError("result must be a VoiceTurnResult-compatible object")
    event = event_builder(
        validation_receipt_id=receipt.validation_receipt_id,
        response_id=response_id,
    )
    if event is None:
        return None

    audio = getattr(result, "audio", None)
    if not isinstance(audio, bytes) or not audio:
        raise LocalResponseDAGQueueError(
            "validated cache miss must contain non-empty output audio"
        )
    actual_audio_sha = sha256(audio).hexdigest()
    actual_rendered_sha = sha256(
        str(getattr(result, "response_text", "") or "").encode("utf-8")
    ).hexdigest()
    if receipt.output_audio_sha256 != actual_audio_sha:
        raise LocalResponseDAGQueueError(
            "independent validation receipt does not match returned audio"
        )
    if receipt.rendered_text_sha256 != actual_rendered_sha:
        raise LocalResponseDAGQueueError(
            "independent validation receipt does not match rendered text"
        )
    live_tts_provider = str(
        getattr(getattr(result, "provenance", None), "tts_provider", "") or ""
    ).strip()
    if (
        live_tts_provider
        and receipt.validator_identity.casefold()
        == live_tts_provider.casefold()
    ):
        raise LocalResponseDAGQueueError(
            "independent validator_identity must differ from live TTS provider"
        )
    descriptor = _stable_audio_descriptor(result, audio_descriptor)

    provenance = getattr(result, "provenance", None)
    provenance_metadata = getattr(provenance, "metadata", {})
    resolved_template = str(template_text or "").strip()
    if not resolved_template and isinstance(provenance_metadata, Mapping):
        resolved_template = str(
            provenance_metadata.get("response_template") or ""
        ).strip()
    resolved_bindings = (
        dict(slot_bindings)
        if slot_bindings is not None
        else _derived_slot_bindings(result)
    )
    candidate_metadata = dict(metadata or {})
    candidate_metadata.update(_surface_metadata(result, surface))
    candidate_metadata["validation_receipt"] = receipt.to_dict()
    candidate_metadata["validation_receipt_sha256"] = receipt.receipt_sha256
    candidate_metadata["validation_method"] = receipt.validation_method
    _assert_privacy_safe(candidate_metadata, path="metadata")

    _, append_candidate = _response_dag_contracts()
    candidate = append_candidate(
        event,
        response_text=str(getattr(result, "response_text", "") or ""),
        audio_descriptor=descriptor,
        template_text=resolved_template,
        slot_bindings=resolved_bindings,
        metadata=candidate_metadata,
    )
    receipt = sink.append(candidate)
    return QueuedVoiceCacheMissCandidate(
        event=event,
        candidate=candidate,
        receipt=receipt,
    )


__all__ = [
    "INDEPENDENT_VOICE_VALIDATION_RECEIPT_SCHEMA_VERSION",
    "LOCAL_RESPONSE_DAG_QUEUE_SCHEMA_VERSION",
    "IndependentVoiceValidationReceipt",
    "LocalValidatedVoiceCacheMissArtifacts",
    "LocalResponseDAGQueue",
    "LocalResponseDAGQueueError",
    "LocalResponseDAGQueueReceipt",
    "QueuedVoiceCacheMissCandidate",
    "enqueue_validated_cache_miss_candidate",
]
