"""Redacted, deterministic cache-miss receipts for the Abby voice runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any

VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION = "abby_voice_cache_miss_event_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TRACE_DETAIL_KEYS = frozenset(
    {
        "confidence",
        "evidence_count",
        "grounded_slot_count",
        "live_tts_fallback",
        "precomputed",
        "resolver_reason",
        "runtime_resolution",
        "slotted_template",
        "spoken_text_sha256",
        "synthesis_identity",
        "template_id",
    }
)
_SECRET_KEY_MARKERS = (
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "token",
)
_PRIVATE_TEXT_KEYS = frozenset(
    {
        "input",
        "input_audio",
        "output_audio",
        "prompt",
        "response_text",
        "spoken_text",
        "transcript",
    }
)
_SYNTHESIS_IDENTITY_FIELDS = frozenset(
    {
        "channels",
        "codec",
        "generation_settings_sha256",
        "identity_sha256",
        "locale",
        "model",
        "provider",
        "provider_version",
        "reference_audio_sha256",
        "sample_rate_hz",
        "voice",
    }
)


class VoiceCacheMissEventError(ValueError):
    """A turn could not be represented as a safe cache-miss event."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise VoiceCacheMissEventError(f"{field_name} must not be empty")
    return result


def _digest(value: Any, *, field_name: str, required: bool = True) -> str:
    result = _text(value, field_name=field_name, required=required).casefold()
    if result and not _SHA256_RE.fullmatch(result):
        raise VoiceCacheMissEventError(
            f"{field_name} must be a full lowercase SHA-256"
        )
    return result


def _redacted_json(value: Any, *, path: str = "value") -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise VoiceCacheMissEventError(f"{path} must not contain raw bytes")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if any(marker in name.casefold() for marker in _SECRET_KEY_MARKERS):
                raise VoiceCacheMissEventError(
                    f"{path}.{name} must not contain credentials"
                )
            if name.casefold() in _PRIVATE_TEXT_KEYS:
                raise VoiceCacheMissEventError(
                    f"{path}.{name} must not contain private turn content"
                )
            result[name] = _redacted_json(item, path=f"{path}.{name}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            _redacted_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _redacted_json(to_dict(), path=path)
    raise VoiceCacheMissEventError(
        f"{path} must contain only deterministic JSON values"
    )


def _safe_synthesis_identity(value: Any) -> Mapping[str, Any]:
    """Retain routing identity while hashing free-form generation settings."""

    if not isinstance(value, Mapping) or not value:
        raise VoiceCacheMissEventError(
            "synthesis_identity must be a non-empty mapping"
        )
    safe = _redacted_json(value, path="synthesis_identity")
    if not isinstance(safe, Mapping):
        raise VoiceCacheMissEventError("synthesis_identity must be a mapping")
    already_safe = (
        "identity_sha256" in safe
        and set(safe).issubset(_SYNTHESIS_IDENTITY_FIELDS)
    )
    selected = {
        str(key): item
        for key, item in safe.items()
        if str(key) in _SYNTHESIS_IDENTITY_FIELDS and item not in (None, "")
    }
    settings = safe.get("generation_settings")
    if settings not in (None, {}, []):
        selected["generation_settings_sha256"] = sha256(
            _canonical_bytes(settings)
        ).hexdigest()
    # Bind omitted/extension fields without placing their possibly free-form
    # values in the event payload.
    identity_digest = safe.get("identity_sha256") if already_safe else None
    selected["identity_sha256"] = _digest(
        identity_digest or sha256(_canonical_bytes(safe)).hexdigest(),
        field_name="synthesis_identity.identity_sha256",
    )
    if "generation_settings_sha256" in selected:
        selected["generation_settings_sha256"] = _digest(
            selected["generation_settings_sha256"],
            field_name="synthesis_identity.generation_settings_sha256",
        )
    if not any(key in selected for key in ("provider", "model", "voice")):
        raise VoiceCacheMissEventError(
            "synthesis_identity requires provider, model, or voice"
        )
    return selected


def _trace_mapping(trace: Any) -> Mapping[str, Any]:
    if isinstance(trace, Mapping):
        return trace
    to_dict = getattr(trace, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    raise VoiceCacheMissEventError("voice stage trace must be a mapping")


def _safe_trace_receipt(trace: Mapping[str, Any]) -> dict[str, Any]:
    details = trace.get("details")
    details = details if isinstance(details, Mapping) else {}
    selected: dict[str, Any] = {}
    for key, value in details.items():
        name = str(key)
        if name not in _ALLOWED_TRACE_DETAIL_KEYS:
            continue
        selected[name] = (
            dict(_safe_synthesis_identity(value))
            if name == "synthesis_identity"
            else _redacted_json(value, path=f"trace.details.{name}")
        )
    return {
        "details": selected,
        "provider": _text(trace.get("provider"), field_name="trace.provider", required=False)
        or None,
        "stage": _text(trace.get("stage"), field_name="trace.stage"),
        "status": _text(trace.get("status"), field_name="trace.status"),
    }


@dataclass(frozen=True, slots=True)
class VoiceCacheMissEvent:
    """A redacted event emitted after exact-audio miss and live synthesis."""

    rendered_text_sha256: str
    synthesis_identity: Mapping[str, Any]
    resolver_miss_reason: str
    output_audio_sha256: str
    live_tts_provider: str
    request_id: str = ""
    template_id: str = ""
    response_id: str = ""
    intent: str = ""
    audio_format: str = ""
    validation_receipt_id: str = ""
    validation_passed: bool = False
    stage_receipts: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION
    event_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rendered_digest = _digest(
            self.rendered_text_sha256, field_name="rendered_text_sha256"
        )
        audio_digest = _digest(
            self.output_audio_sha256, field_name="output_audio_sha256"
        )
        identity = _safe_synthesis_identity(self.synthesis_identity)
        miss_reason = _text(
            self.resolver_miss_reason, field_name="resolver_miss_reason"
        )
        provider = _text(self.live_tts_provider, field_name="live_tts_provider")
        if self.schema_version != VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION:
            raise VoiceCacheMissEventError(
                f"unsupported cache-miss event schema: {self.schema_version}"
            )
        if not isinstance(self.validation_passed, bool):
            raise VoiceCacheMissEventError("validation_passed must be a boolean")
        validation_id = _text(
            self.validation_receipt_id,
            field_name="validation_receipt_id",
            required=False,
        )
        if self.validation_passed and not validation_id:
            raise VoiceCacheMissEventError(
                "validated cache miss requires validation_receipt_id"
            )
        receipts = tuple(
            _redacted_json(receipt, path=f"stage_receipts[{index}]")
            for index, receipt in enumerate(self.stage_receipts)
        )
        metadata = _redacted_json(self.metadata, path="metadata")
        if not isinstance(metadata, Mapping):
            raise VoiceCacheMissEventError("metadata must be a mapping")

        object.__setattr__(self, "rendered_text_sha256", rendered_digest)
        object.__setattr__(self, "output_audio_sha256", audio_digest)
        object.__setattr__(self, "synthesis_identity", MappingProxyType(dict(identity)))
        object.__setattr__(self, "resolver_miss_reason", miss_reason)
        object.__setattr__(self, "live_tts_provider", provider)
        for name in (
            "request_id",
            "template_id",
            "response_id",
            "intent",
            "audio_format",
            "validation_receipt_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "stage_receipts",
            tuple(MappingProxyType(dict(receipt)) for receipt in receipts),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata)))

        computed = (
            "abby-voice-cache-miss:sha256:"
            + sha256(_canonical_bytes(self.identity_dict())).hexdigest()
        )
        if self.event_id and self.event_id != computed:
            raise VoiceCacheMissEventError(
                "event_id does not match deterministic cache-miss identity"
            )
        object.__setattr__(self, "event_id", computed)

    @property
    def ready_for_dag_append(self) -> bool:
        return self.validation_passed and bool(self.validation_receipt_id)

    def identity_dict(self) -> dict[str, Any]:
        """Fields that make repeat validation attempts one semantic event."""

        return {
            "audio_format": self.audio_format,
            "intent": self.intent,
            "live_tts_provider": self.live_tts_provider,
            "output_audio_sha256": self.output_audio_sha256,
            "rendered_text_sha256": self.rendered_text_sha256,
            "resolver_miss_reason": self.resolver_miss_reason,
            "response_id": self.response_id,
            "schema_version": self.schema_version,
            "synthesis_identity": dict(self.synthesis_identity),
            "template_id": self.template_id,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_dict()
        payload.update(
            {
                "event_id": self.event_id,
                "metadata": dict(self.metadata),
                "ready_for_dag_append": self.ready_for_dag_append,
                "request_id": self.request_id,
                "stage_receipts": [
                    dict(receipt) for receipt in self.stage_receipts
                ],
                "validation_passed": self.validation_passed,
                "validation_receipt_id": self.validation_receipt_id,
            }
        )
        return payload


def build_voice_cache_miss_event(
    result: Any,
    *,
    response_id: str = "",
    validation_receipt_id: str = "",
    validation_passed: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> VoiceCacheMissEvent | None:
    """Return an event only for exact resolver miss followed by live TTS.

    ``result`` is intentionally duck-typed so this dependency-light module
    does not import the large voice router or create a circular import.
    """

    traces = tuple(getattr(result, "traces", ()) or ())
    miss_trace: Mapping[str, Any] | None = None
    live_trace: Mapping[str, Any] | None = None
    safe_receipts: list[dict[str, Any]] = []
    for trace_index, raw_trace in enumerate(traces):
        trace = _trace_mapping(raw_trace)
        safe_receipts.append(_safe_trace_receipt(trace))
        details = trace.get("details")
        details = details if isinstance(details, Mapping) else {}
        if (
            str(trace.get("stage") or "") == "synthesis"
            and str(trace.get("provider") or "") == "precomputed"
            and details.get("precomputed") is False
            and str(details.get("resolver_reason") or "").strip()
            and (
                str(trace.get("status") or "") == "skipped"
                or details.get("resolver_reason")
                == "precomputed_audio_validation_failed"
            )
        ):
            miss_trace = trace
            miss_trace_index = trace_index
        elif (
            str(trace.get("stage") or "") == "synthesis"
            and str(trace.get("status") or "") == "succeeded"
            and str(trace.get("provider") or "") != "precomputed"
            and miss_trace is not None
            and trace_index > miss_trace_index
        ):
            live_trace = trace

    if miss_trace is None or live_trace is None:
        return None
    provenance = getattr(result, "provenance", None)
    if provenance is None:
        raise VoiceCacheMissEventError("voice result provenance is required")
    miss_details = miss_trace.get("details")
    miss_details = miss_details if isinstance(miss_details, Mapping) else {}
    synthesis_identity = miss_details.get("synthesis_identity")
    if not isinstance(synthesis_identity, Mapping) or not synthesis_identity:
        raise VoiceCacheMissEventError(
            "precomputed miss trace lacks synthesis_identity"
        )
    response_text = _text(
        getattr(result, "response_text", ""), field_name="response_text"
    )
    rendered_digest = sha256(response_text.encode("utf-8")).hexdigest()
    provenance_digest = getattr(provenance, "response_text_sha256", None)
    if provenance_digest and str(provenance_digest) != rendered_digest:
        raise VoiceCacheMissEventError(
            "voice result response text does not match provenance digest"
        )
    output_digest = str(getattr(provenance, "output_audio_sha256", "") or "")
    audio = getattr(result, "audio", None)
    if isinstance(audio, bytes):
        actual_output_digest = sha256(audio).hexdigest()
        if output_digest and output_digest != actual_output_digest:
            raise VoiceCacheMissEventError(
                "voice result audio does not match provenance digest"
            )
        output_digest = actual_output_digest
    metadata_value = getattr(provenance, "metadata", {}) or {}
    intent = (
        str(metadata_value.get("intent") or "")
        if isinstance(metadata_value, Mapping)
        else ""
    )
    return VoiceCacheMissEvent(
        rendered_text_sha256=rendered_digest,
        synthesis_identity=synthesis_identity,
        resolver_miss_reason=str(miss_details.get("resolver_reason") or ""),
        output_audio_sha256=output_digest,
        live_tts_provider=str(live_trace.get("provider") or ""),
        request_id=str(getattr(result, "request_id", "") or ""),
        template_id=str(getattr(provenance, "template_id", "") or ""),
        response_id=response_id,
        intent=intent,
        audio_format=str(getattr(result, "audio_format", "") or ""),
        validation_receipt_id=validation_receipt_id,
        validation_passed=validation_passed,
        stage_receipts=tuple(safe_receipts),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "VOICE_CACHE_MISS_EVENT_SCHEMA_VERSION",
    "VoiceCacheMissEvent",
    "VoiceCacheMissEventError",
    "build_voice_cache_miss_event",
]
