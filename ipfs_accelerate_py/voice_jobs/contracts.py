"""Dependency-light, content-addressed contracts for distributed voice jobs.

The queue payloads defined here contain artifact *descriptors*, never artifact
bytes.  A request identity covers every output-affecting input and is used
directly as the queue task id, making retries safe to de-duplicate.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar
from urllib.parse import parse_qsl, unquote, urlsplit

VOICE_JOB_SCHEMA_VERSION = "abby_voice_job_v1"
VOICE_JOB_RESULT_SCHEMA_VERSION = "abby_voice_job_result_v1"

VOICE_TTS_TASK_TYPE = "voice.tts"
VOICE_ASR_TASK_TYPE = "voice.asr"
VOICE_AUDIO_VALIDATION_TASK_TYPE = "voice.audio-validate"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_TYPE_ALIASES = {
    "voice.tts": VOICE_TTS_TASK_TYPE,
    "tts": VOICE_TTS_TASK_TYPE,
    "text-to-speech": VOICE_TTS_TASK_TYPE,
    "voice.asr": VOICE_ASR_TASK_TYPE,
    "asr": VOICE_ASR_TASK_TYPE,
    "stt": VOICE_ASR_TASK_TYPE,
    "speech-to-text": VOICE_ASR_TASK_TYPE,
    "automatic-speech-recognition": VOICE_ASR_TASK_TYPE,
    "voice.audio-validate": VOICE_AUDIO_VALIDATION_TASK_TYPE,
    "audio-validate": VOICE_AUDIO_VALIDATION_TASK_TYPE,
    "audio-validation": VOICE_AUDIO_VALIDATION_TASK_TYPE,
}
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "signature",
    }
)
_PROVIDER_RECEIPT_STRING_KEYS = frozenset(
    {
        "backend",
        "model",
        "provider",
        "provider_version",
        "region",
    }
)
_PROVIDER_RECEIPT_INTEGER_KEYS = frozenset(
    {
        "attempt_count",
        "latency_ms",
        "status_code",
    }
)
_PROVIDER_RECEIPT_HASH_KEYS = frozenset(
    {
        "provider_request_id_sha256",
        "request_id_sha256",
        "response_id_sha256",
        "worker_id_sha256",
    }
)
_PROVIDER_RECEIPT_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_EXTERNAL_URI_SCHEMES = frozenset(
    {
        "gs",
        "hf",
        "http",
        "https",
        "ipfs",
        "s3",
    }
)
_AUDIO_MAGIC_PREFIXES = (
    b"ID3",
    b"OggS",
    b"RIFF",
    b"fLaC",
    b"\x1aE\xdf\xa3",
)
_INLINE_AUDIO_KEYS = (
    "audio_base64",
    "audio_bytes",
    "base64_audio",
    "local_path",
    "waveform",
)


def _is_secret_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized in _SECRET_KEYS


def _looks_like_base64_audio(value: str) -> bool:
    compact = "".join(value.split())
    if len(compact) < 12 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", compact):
        return False
    padded = compact + "=" * (-len(compact) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return False
    return (
        decoded.startswith(_AUDIO_MAGIC_PREFIXES)
        or len(decoded) >= 8 and decoded[4:8] == b"ftyp"
        or len(decoded) >= 2
        and decoded[0] == 0xFF
        and decoded[1] & 0xF0 == 0xF0
    )


def _freeze_provider_receipt(value: Mapping[str, Any]) -> MappingProxyType:
    receipt: dict[str, Any] = {}
    allowed = (
        _PROVIDER_RECEIPT_STRING_KEYS
        | _PROVIDER_RECEIPT_INTEGER_KEYS
        | _PROVIDER_RECEIPT_HASH_KEYS
    )
    for key, item in value.items():
        if _is_secret_key(key):
            raise VoiceJobContractError("provider_receipt must not contain credentials")
        if key not in allowed:
            raise VoiceJobContractError(f"unsupported provider_receipt field: {key!r}")
        if key in _PROVIDER_RECEIPT_HASH_KEYS:
            receipt[key] = _require_sha256(f"provider_receipt.{key}", item)
        elif key in _PROVIDER_RECEIPT_INTEGER_KEYS:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise VoiceJobContractError(
                    f"provider_receipt.{key} must be a non-negative integer"
                )
            receipt[key] = item
        elif (
            not isinstance(item, str)
            or not _PROVIDER_RECEIPT_IDENTIFIER_RE.fullmatch(
                _freeze_json(item, path=f"provider_receipt.{key}")
            )
        ):
            raise VoiceJobContractError(
                f"provider_receipt.{key} must be a canonical identifier"
            )
        else:
            receipt[key] = item
    return MappingProxyType(receipt)


class VoiceJobContractError(ValueError):
    """Raised when a voice job would violate the transport contract."""


def canonical_task_type(value: str) -> str:
    """Return the canonical voice task type, including historical ASR aliases."""

    normalized = str(value or "").strip().casefold().replace("_", "-")
    try:
        return _TASK_TYPE_ALIASES[normalized]
    except KeyError as exc:
        raise VoiceJobContractError(f"unsupported voice task type: {value!r}") from exc


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise VoiceJobContractError(f"{name} must be a stable non-empty string")
    return value


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise VoiceJobContractError(f"{name} must be a full lowercase SHA-256")
    return value


def _freeze_json(value: Any, *, path: str = "value") -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.casefold()
        if (
            lowered.startswith(("data:", "file:", "~/"))
            or stripped.startswith(("/", "\\\\"))
            or re.match(r"^[a-zA-Z]:[\\/]", stripped)
            or _looks_like_base64_audio(stripped)
        ):
            raise VoiceJobContractError(f"{path} must not contain inline audio or local paths")
        if lowered.startswith(("bearer ", "basic ")):
            raise VoiceJobContractError(f"{path} must not contain credentials")
        if "://" in stripped:
            try:
                parsed = urlsplit(stripped)
            except ValueError as exc:
                raise VoiceJobContractError(f"{path} contains an invalid URI") from exc
            if parsed.username is not None or parsed.password is not None:
                raise VoiceJobContractError(f"{path} must not contain credentials")
            for key, _item in parse_qsl(parsed.query, keep_blank_values=True):
                if _is_secret_key(key):
                    raise VoiceJobContractError(f"{path} must not contain credentials")
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise VoiceJobContractError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, bytes):
        raise VoiceJobContractError(f"{path} must not contain bytes")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise VoiceJobContractError(f"{path} keys must be non-empty strings")
            lowered = key.casefold()
            if _is_secret_key(key):
                raise VoiceJobContractError(f"{path} must not contain credentials")
            if any(part in lowered for part in _INLINE_AUDIO_KEYS):
                raise VoiceJobContractError(f"{path} must not contain inline audio or local paths")
            if (
                any(part in lowered for part in ("private_text", "response_text", "spoken_text", "transcript"))
                and not lowered.endswith(("_hash", "_sha256"))
            ):
                raise VoiceJobContractError(f"{path} must not contain private transcript text")
            result[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise VoiceJobContractError(f"{path} contains a non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VoiceJobContractError("voice contract is not canonical JSON") from exc


def _content_task_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _validate_schema(payload: Mapping[str, Any], expected: str) -> None:
    if payload.get("schema_version") != expected:
        raise VoiceJobContractError(f"unsupported schema_version: {payload.get('schema_version')!r}")


def _validate_canonical_request_payload(payload: Mapping[str, Any], job: Any) -> None:
    """Reject non-canonical, incomplete, or extended transport payloads."""

    supplied = dict(payload)
    supplied["task_type"] = canonical_task_type(str(supplied.get("task_type") or ""))
    if supplied != job.to_payload():
        raise VoiceJobContractError(
            "voice request payload must exactly match its canonical contract"
        )


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Immutable external artifact identity suitable for a DuckDB task row."""

    uri: str
    sha256: str
    size_bytes: int
    media_type: str = "audio/wav"
    cid: str = ""

    def __post_init__(self) -> None:
        _require_sha256("sha256", self.sha256)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise VoiceJobContractError("size_bytes must be a non-negative integer")
        if (
            not isinstance(self.media_type, str)
            or "/" not in self.media_type
            or any(character.isspace() for character in self.media_type)
        ):
            raise VoiceJobContractError("media_type must be a valid MIME type")
        if not self.uri and not self.cid:
            raise VoiceJobContractError("artifact requires uri or cid")
        if self.uri:
            self._validate_uri(self.uri)
        if self.cid:
            if (
                self.cid.strip() != self.cid
                or any(character.isspace() for character in self.cid)
                or any(character in self.cid for character in "/\\:")
            ):
                raise VoiceJobContractError("cid must be a bare content identifier")

    @staticmethod
    def _validate_uri(uri: str) -> None:
        if not isinstance(uri, str) or uri.strip() != uri or any(character.isspace() for character in uri):
            raise VoiceJobContractError("uri must be an external artifact URI")
        try:
            parsed = urlsplit(uri)
        except ValueError as exc:
            raise VoiceJobContractError("uri must be a valid external artifact URI") from exc
        scheme = parsed.scheme.casefold()
        if scheme not in _EXTERNAL_URI_SCHEMES or not parsed.netloc:
            raise VoiceJobContractError("uri must identify an external artifact")
        if parsed.username is not None or parsed.password is not None:
            raise VoiceJobContractError("uri must not contain credentials")
        if "\\" in unquote(parsed.path) or ".." in unquote(parsed.path).split("/") or parsed.fragment:
            raise VoiceJobContractError("uri must not contain a local or ambiguous path")
        for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_secret_key(key):
                raise VoiceJobContractError("uri must not contain credentials")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactDescriptor:
        return cls(
            uri=str(payload.get("uri") or ""),
            cid=str(payload.get("cid") or payload.get("ipfs_cid") or ""),
            sha256=str(payload.get("sha256") or payload.get("content_sha256") or ""),
            size_bytes=payload.get("size_bytes", payload.get("byte_length")),  # type: ignore[arg-type]
            media_type=str(payload.get("media_type") or payload.get("mime_type") or "audio/wav"),
        )

    from_mapping = from_dict


@dataclass(frozen=True, slots=True)
class VoiceJobLineage:
    """Dataset identities that must survive request, queue, and result hops."""

    workset_id: str
    manifest_id: str
    source_manifest_id: str
    work_item_id: str
    subject_id: str
    subject_schema_version: str
    policy_id: str
    depends_on_task_ids: tuple[str, ...] = ()
    publication_id: str = ""
    task_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "workset_id",
            "manifest_id",
            "source_manifest_id",
            "work_item_id",
            "subject_id",
            "subject_schema_version",
            "policy_id",
        ):
            _require_text(name, getattr(self, name))
        dependencies = tuple(sorted(set(self.depends_on_task_ids)))
        for task_id in dependencies:
            _require_sha256("depends_on_task_ids item", task_id)
        if self.task_id:
            _require_sha256("task_id", self.task_id)
        if self.publication_id:
            _require_text("publication_id", self.publication_id)
        object.__setattr__(self, "depends_on_task_ids", dependencies)

    def identity_dict(self) -> dict[str, Any]:
        """Return lineage inputs; transport task_id is deliberately excluded."""

        return {
            "depends_on_task_ids": list(self.depends_on_task_ids),
            "manifest_id": self.manifest_id,
            "policy_id": self.policy_id,
            "publication_id": self.publication_id,
            "source_manifest_id": self.source_manifest_id,
            "subject_id": self.subject_id,
            "subject_schema_version": self.subject_schema_version,
            "work_item_id": self.work_item_id,
            "workset_id": self.workset_id,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["task_id"] = self.task_id
        return payload

    def with_task_id(self, task_id: str) -> VoiceJobLineage:
        _require_sha256("task_id", task_id)
        if self.task_id and self.task_id != task_id:
            raise VoiceJobContractError("lineage task_id does not match deterministic request")
        return self if self.task_id == task_id else replace(self, task_id=task_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VoiceJobLineage:
        dependencies = payload.get("depends_on_task_ids", ())
        if isinstance(dependencies, str):
            dependencies = (dependencies,)
        return cls(
            workset_id=str(payload.get("workset_id") or ""),
            manifest_id=str(payload.get("manifest_id") or ""),
            source_manifest_id=str(payload.get("source_manifest_id") or ""),
            work_item_id=str(payload.get("work_item_id") or payload.get("work_id") or ""),
            subject_id=str(payload.get("subject_id") or ""),
            subject_schema_version=str(payload.get("subject_schema_version") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            depends_on_task_ids=tuple(str(item) for item in dependencies),  # type: ignore[arg-type]
            publication_id=str(payload.get("publication_id") or ""),
            task_id=str(payload.get("task_id") or ""),
        )

    from_mapping = from_dict


class _VoiceJob:
    task_type: ClassVar[str]
    schema_version: str
    lineage: VoiceJobLineage
    task_id: str
    model_name: str

    def identity_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        # ``_lineage`` is the established p2p worker transport key.  Keeping
        # the typed ``lineage`` field too makes the shared JSON contract clear.
        payload["_lineage"] = dict(payload["lineage"])
        return payload

    def _computed_task_id(self) -> str:
        return _content_task_id(self.identity_dict())

    def _finalize_identity(self) -> None:
        computed = self._computed_task_id()
        if self.task_id and self.task_id != computed:
            raise VoiceJobContractError("task_id does not match deterministic request content")
        object.__setattr__(self, "task_id", computed)
        object.__setattr__(self, "lineage", self.lineage.with_task_id(computed))


@dataclass(frozen=True, slots=True)
class VoiceTTSJob(_VoiceJob):
    spoken_text: str
    locale: str
    provider: str
    model_name: str
    voice: str
    provider_version: str
    lineage: VoiceJobLineage
    codec: str = "wav"
    sample_rate_hz: int = 24_000
    channels: int = 1
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    reference_audio: ArtifactDescriptor | None = None
    schema_version: str = VOICE_JOB_SCHEMA_VERSION
    task_id: str = ""

    task_type: ClassVar[str] = VOICE_TTS_TASK_TYPE

    def __post_init__(self) -> None:
        for name in ("locale", "provider", "model_name", "voice", "provider_version", "codec"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.spoken_text, str) or not self.spoken_text.strip():
            raise VoiceJobContractError("spoken_text must not be empty")
        normalized = unicodedata.normalize("NFC", self.spoken_text.replace("\r\n", "\n").replace("\r", "\n"))
        object.__setattr__(self, "spoken_text", normalized)
        if isinstance(self.sample_rate_hz, bool) or not isinstance(self.sample_rate_hz, int) or self.sample_rate_hz <= 0:
            raise VoiceJobContractError("sample_rate_hz must be a positive integer")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels <= 0:
            raise VoiceJobContractError("channels must be a positive integer")
        object.__setattr__(
            self, "generation_settings", _freeze_json(self.generation_settings, path="generation_settings")
        )
        if self.reference_audio and not self.reference_audio.media_type.startswith("audio/"):
            raise VoiceJobContractError("reference_audio media_type must be audio/*")
        _validate_schema({"schema_version": self.schema_version}, VOICE_JOB_SCHEMA_VERSION)
        self._finalize_identity()

    @property
    def spoken_text_sha256(self) -> str:
        return hashlib.sha256(self.spoken_text.encode("utf-8")).hexdigest()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "codec": self.codec,
            "generation_settings": _thaw_json(self.generation_settings),
            "lineage": self.lineage.identity_dict(),
            "locale": self.locale,
            "model_name": self.model_name,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "reference_audio": self.reference_audio.to_dict() if self.reference_audio else None,
            "sample_rate_hz": self.sample_rate_hz,
            "schema_version": self.schema_version,
            "spoken_text": self.spoken_text,
            "spoken_text_sha256": self.spoken_text_sha256,
            "task_type": self.task_type,
            "voice": self.voice,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["lineage"] = self.lineage.to_dict()
        payload["task_id"] = self.task_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VoiceTTSJob:
        _validate_schema(payload, VOICE_JOB_SCHEMA_VERSION)
        if canonical_task_type(str(payload.get("task_type") or cls.task_type)) != cls.task_type:
            raise VoiceJobContractError("payload is not a TTS job")
        reference = payload.get("reference_audio")
        job = cls(
            spoken_text=str(payload.get("spoken_text") or ""),
            locale=str(payload.get("locale") or ""),
            provider=str(payload.get("provider") or ""),
            model_name=str(payload.get("model_name") or ""),
            voice=str(payload.get("voice") or ""),
            provider_version=str(payload.get("provider_version") or ""),
            lineage=VoiceJobLineage.from_dict(_lineage_payload(payload)),
            codec=str(payload.get("codec") or "wav"),
            sample_rate_hz=payload.get("sample_rate_hz", 24_000),  # type: ignore[arg-type]
            channels=payload.get("channels", 1),  # type: ignore[arg-type]
            generation_settings=_mapping(payload.get("generation_settings"), "generation_settings"),
            reference_audio=ArtifactDescriptor.from_dict(reference) if isinstance(reference, Mapping) else None,
            task_id=str(payload.get("task_id") or ""),
        )
        _validate_canonical_request_payload(payload, job)
        return job

    from_payload = from_dict


def _source_identity(source_audio: ArtifactDescriptor | None, source_task_id: str) -> dict[str, Any]:
    return {
        "source_audio": source_audio.to_dict() if source_audio else None,
        "source_task_id": source_task_id,
    }


def _validate_source(
    source_audio: ArtifactDescriptor | None,
    source_task_id: str,
    lineage: VoiceJobLineage,
) -> None:
    if (source_audio is None) == (not source_task_id):
        raise VoiceJobContractError("exactly one of source_audio or source_task_id is required")
    if source_audio is not None and not source_audio.media_type.startswith("audio/"):
        raise VoiceJobContractError("source_audio media_type must be audio/*")
    if source_task_id:
        _require_sha256("source_task_id", source_task_id)
        if source_task_id not in lineage.depends_on_task_ids:
            raise VoiceJobContractError("source_task_id must appear in lineage.depends_on_task_ids")


@dataclass(frozen=True, slots=True)
class VoiceASRJob(_VoiceJob):
    provider: str
    model_name: str
    provider_version: str
    lineage: VoiceJobLineage
    source_audio: ArtifactDescriptor | None = None
    source_task_id: str = ""
    purpose: str = "dataset_asr_validation"
    locale: str = ""
    decoding_settings: Mapping[str, Any] = field(default_factory=dict)
    retention_policy: str = "none"
    schema_version: str = VOICE_JOB_SCHEMA_VERSION
    task_id: str = ""

    task_type: ClassVar[str] = VOICE_ASR_TASK_TYPE

    def __post_init__(self) -> None:
        for name in ("provider", "model_name", "provider_version"):
            _require_text(name, getattr(self, name))
        if self.purpose not in {"runtime_stt", "dataset_asr_validation"}:
            raise VoiceJobContractError("purpose must be runtime_stt or dataset_asr_validation")
        if self.retention_policy not in {"none", "result", "publication"}:
            raise VoiceJobContractError("unsupported retention_policy")
        if self.purpose == "runtime_stt":
            if self.lineage.publication_id:
                raise VoiceJobContractError("runtime_stt rejects publication lineage")
            if self.retention_policy != "none":
                raise VoiceJobContractError("runtime_stt results are non-retained by default")
        _validate_source(self.source_audio, self.source_task_id, self.lineage)
        object.__setattr__(
            self, "decoding_settings", _freeze_json(self.decoding_settings, path="decoding_settings")
        )
        _validate_schema({"schema_version": self.schema_version}, VOICE_JOB_SCHEMA_VERSION)
        self._finalize_identity()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "decoding_settings": _thaw_json(self.decoding_settings),
            "lineage": self.lineage.identity_dict(),
            "locale": self.locale,
            "model_name": self.model_name,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "purpose": self.purpose,
            "retention_policy": self.retention_policy,
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            **_source_identity(self.source_audio, self.source_task_id),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["lineage"] = self.lineage.to_dict()
        payload["task_id"] = self.task_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VoiceASRJob:
        _validate_schema(payload, VOICE_JOB_SCHEMA_VERSION)
        if canonical_task_type(str(payload.get("task_type") or cls.task_type)) != cls.task_type:
            raise VoiceJobContractError("payload is not an ASR job")
        source = payload.get("source_audio")
        job = cls(
            provider=str(payload.get("provider") or ""),
            model_name=str(payload.get("model_name") or ""),
            provider_version=str(payload.get("provider_version") or ""),
            lineage=VoiceJobLineage.from_dict(_lineage_payload(payload)),
            source_audio=ArtifactDescriptor.from_dict(source) if isinstance(source, Mapping) else None,
            source_task_id=str(payload.get("source_task_id") or ""),
            purpose=str(payload.get("purpose") or "dataset_asr_validation"),
            locale=str(payload.get("locale") or ""),
            decoding_settings=_mapping(payload.get("decoding_settings"), "decoding_settings"),
            retention_policy=str(payload.get("retention_policy") or "none"),
            task_id=str(payload.get("task_id") or ""),
        )
        _validate_canonical_request_payload(payload, job)
        return job

    from_payload = from_dict


@dataclass(frozen=True, slots=True)
class VoiceAudioValidationJob(_VoiceJob):
    model_name: str
    lineage: VoiceJobLineage
    source_audio: ArtifactDescriptor | None = None
    source_task_id: str = ""
    validation_policy: Mapping[str, Any] = field(default_factory=dict)
    provider: str = "local"
    policy_version: str = "1"
    schema_version: str = VOICE_JOB_SCHEMA_VERSION
    task_id: str = ""

    task_type: ClassVar[str] = VOICE_AUDIO_VALIDATION_TASK_TYPE

    def __post_init__(self) -> None:
        for name in ("model_name", "provider", "policy_version"):
            _require_text(name, getattr(self, name))
        _validate_source(self.source_audio, self.source_task_id, self.lineage)
        object.__setattr__(
            self, "validation_policy", _freeze_json(self.validation_policy, path="validation_policy")
        )
        _validate_schema({"schema_version": self.schema_version}, VOICE_JOB_SCHEMA_VERSION)
        self._finalize_identity()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "lineage": self.lineage.identity_dict(),
            "model_name": self.model_name,
            "policy_version": self.policy_version,
            "provider": self.provider,
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "validation_policy": _thaw_json(self.validation_policy),
            **_source_identity(self.source_audio, self.source_task_id),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload["lineage"] = self.lineage.to_dict()
        payload["task_id"] = self.task_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VoiceAudioValidationJob:
        _validate_schema(payload, VOICE_JOB_SCHEMA_VERSION)
        if canonical_task_type(str(payload.get("task_type") or cls.task_type)) != cls.task_type:
            raise VoiceJobContractError("payload is not an audio-validation job")
        source = payload.get("source_audio")
        job = cls(
            model_name=str(payload.get("model_name") or ""),
            lineage=VoiceJobLineage.from_dict(_lineage_payload(payload)),
            source_audio=ArtifactDescriptor.from_dict(source) if isinstance(source, Mapping) else None,
            source_task_id=str(payload.get("source_task_id") or ""),
            validation_policy=_mapping(payload.get("validation_policy"), "validation_policy"),
            provider=str(payload.get("provider") or "local"),
            policy_version=str(payload.get("policy_version") or "1"),
            task_id=str(payload.get("task_id") or ""),
        )
        _validate_canonical_request_payload(payload, job)
        return job

    from_payload = from_dict


VoiceJob = VoiceTTSJob | VoiceASRJob | VoiceAudioValidationJob


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise VoiceJobContractError(f"{name} must be an object")
    return value


def _lineage_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    lineage = payload.get("lineage")
    transport_lineage = payload.get("_lineage")
    if not isinstance(lineage, Mapping) or not isinstance(transport_lineage, Mapping):
        raise VoiceJobContractError("payload requires matching lineage and _lineage objects")
    if dict(lineage) != dict(transport_lineage):
        raise VoiceJobContractError("lineage and _lineage disagree")
    return lineage


def voice_job_from_payload(payload: Mapping[str, Any]) -> VoiceJob:
    if not isinstance(payload, Mapping):
        raise VoiceJobContractError("voice request payload must be an object")
    task_type = canonical_task_type(str(payload.get("task_type") or ""))
    if task_type == VOICE_TTS_TASK_TYPE:
        return VoiceTTSJob.from_dict(payload)
    if task_type == VOICE_ASR_TASK_TYPE:
        return VoiceASRJob.from_dict(payload)
    return VoiceAudioValidationJob.from_dict(payload)


@dataclass(frozen=True, slots=True)
class VoiceJobError:
    code: str
    retryable: bool
    message: str = ""

    def __post_init__(self) -> None:
        _require_text("error code", self.code)
        if not _ERROR_CODE_RE.fullmatch(self.code):
            raise VoiceJobContractError("error code must be a canonical machine identifier")
        if not isinstance(self.retryable, bool):
            raise VoiceJobContractError("retryable must be boolean")
        if not isinstance(self.message, str):
            raise VoiceJobContractError("error message must be text")
        if self.message:
            raise VoiceJobContractError(
                "error message must be empty; use the machine-readable code"
            )

    @property
    def terminal(self) -> bool:
        return not self.retryable

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VoiceJobError:
        return cls(
            code=str(payload.get("code") or ""),
            retryable=payload.get("retryable"),  # type: ignore[arg-type]
            message=str(payload.get("message") or ""),
        )


@dataclass(frozen=True, slots=True)
class VoiceJobResult:
    """Typed terminal or retryable receipt retaining the request lineage."""

    task_id: str
    task_type: str
    status: str
    lineage: VoiceJobLineage
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    quality_metrics: Mapping[str, int] = field(default_factory=dict)
    provider_receipt: Mapping[str, Any] = field(default_factory=dict)
    error: VoiceJobError | None = None
    schema_version: str = VOICE_JOB_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256("task_id", self.task_id)
        task_type = canonical_task_type(self.task_type)
        if self.status not in {"completed", "failed", "cancelled"}:
            raise VoiceJobContractError("unsupported result status")
        if self.lineage.task_id != self.task_id:
            raise VoiceJobContractError("result lineage task_id must match task_id")
        artifacts = tuple(self.artifacts)
        metrics: dict[str, int] = {}
        for key, value in self.quality_metrics.items():
            _require_text("quality metric name", key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise VoiceJobContractError("quality metrics must be integers")
            metrics[key] = value
        receipt = _freeze_provider_receipt(self.provider_receipt)
        if self.status == "completed" and self.error is not None:
            raise VoiceJobContractError("completed result must not contain an error")
        if self.status == "failed" and self.error is None:
            raise VoiceJobContractError("failed result requires a typed error")
        _validate_schema({"schema_version": self.schema_version}, VOICE_JOB_RESULT_SCHEMA_VERSION)
        object.__setattr__(self, "task_type", task_type)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "quality_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "provider_receipt", receipt)

    @classmethod
    def from_job(
        cls,
        job: VoiceJob,
        *,
        status: str = "completed",
        artifacts: tuple[ArtifactDescriptor, ...] = (),
        quality_metrics: Mapping[str, int] | None = None,
        provider_receipt: Mapping[str, Any] | None = None,
        error: VoiceJobError | None = None,
    ) -> VoiceJobResult:
        return cls(
            task_id=job.task_id,
            task_type=job.task_type,
            status=status,
            lineage=job.lineage,
            artifacts=artifacts,
            quality_metrics=quality_metrics or {},
            provider_receipt=provider_receipt or {},
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "error": self.error.to_dict() if self.error else None,
            "lineage": self.lineage.to_dict(),
            "provider_receipt": _thaw_json(self.provider_receipt),
            "quality_metrics": dict(self.quality_metrics),
            "schema_version": self.schema_version,
            "status": self.status,
            "task_id": self.task_id,
            "task_type": self.task_type,
        }

    to_payload = to_dict

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VoiceJobResult:
        if not isinstance(payload, Mapping):
            raise VoiceJobContractError("voice result payload must be an object")
        _validate_schema(payload, VOICE_JOB_RESULT_SCHEMA_VERSION)
        artifacts = payload.get("artifacts", ())
        if not isinstance(artifacts, list | tuple):
            raise VoiceJobContractError("artifacts must be an array")
        error = payload.get("error")
        parsed_artifacts: list[ArtifactDescriptor] = []
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise VoiceJobContractError("each artifact must be an object")
            parsed_artifacts.append(ArtifactDescriptor.from_dict(item))
        lineage = payload.get("lineage")
        if not isinstance(lineage, Mapping):
            raise VoiceJobContractError("voice result payload requires lineage")
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            task_type=str(payload.get("task_type") or ""),
            status=str(payload.get("status") or ""),
            lineage=VoiceJobLineage.from_dict(lineage),
            artifacts=tuple(parsed_artifacts),
            quality_metrics=_mapping(payload.get("quality_metrics"), "quality_metrics"),  # type: ignore[arg-type]
            provider_receipt=_mapping(payload.get("provider_receipt"), "provider_receipt"),
            error=VoiceJobError.from_dict(error) if isinstance(error, Mapping) else None,
        )
        if dict(payload) != result.to_payload():
            raise VoiceJobContractError(
                "voice result payload must exactly match its canonical contract"
            )
        return result

    from_dict = from_payload


__all__ = [
    "ArtifactDescriptor",
    "VOICE_ASR_TASK_TYPE",
    "VOICE_AUDIO_VALIDATION_TASK_TYPE",
    "VOICE_JOB_RESULT_SCHEMA_VERSION",
    "VOICE_JOB_SCHEMA_VERSION",
    "VOICE_TTS_TASK_TYPE",
    "VoiceASRJob",
    "VoiceAudioValidationJob",
    "VoiceJob",
    "VoiceJobContractError",
    "VoiceJobError",
    "VoiceJobLineage",
    "VoiceJobResult",
    "VoiceTTSJob",
    "canonical_task_type",
    "voice_job_from_payload",
]
