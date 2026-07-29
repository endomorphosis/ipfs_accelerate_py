"""Bounded asynchronous transport for non-authoritative software analysis.

The transport deliberately separates *declaration* from *activation*.
Capabilities are registered as immutable data and can be discovered without
importing or calling a provider.  A lazy provider factory (including the
optional :mod:`ipfs_datasets_py` factory) is invoked only after a request has
passed bounds checks, admission, and schema negotiation.

Providers are read-only evidence producers.  Their verdicts are always
diagnostic/proposal-tier and can never acquire completion, proof, validation,
merge, or mutation authority through this transport.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


ANALYSIS_TRANSPORT_VERSION: Final[int] = 1
ANALYSIS_TRANSPORT_PROTOCOL_VERSION: Final[int] = 1
ANALYSIS_TRANSPORT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-transport-request@1"
)
ANALYSIS_TRANSPORT_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-transport-result@1"
)
ANALYSIS_TRANSPORT_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-transport-capability@1"
)
ANALYSIS_TRANSPORT_PROGRESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-transport-progress@1"
)

DEFAULT_MAX_QUESTION_BYTES: Final[int] = 8 * 1024
DEFAULT_MAX_REQUEST_BYTES: Final[int] = 64 * 1024
DEFAULT_MAX_RESULT_BYTES: Final[int] = 128 * 1024
DEFAULT_MAX_REFERENCE_BYTES: Final[int] = 8 * 1024
DEFAULT_MAX_REFERENCES: Final[int] = 64
DEFAULT_MAX_BATCH_SIZE: Final[int] = 16
DEFAULT_MAX_PROGRESS_EVENTS: Final[int] = 64
DEFAULT_MAX_CONCURRENCY: Final[int] = 4
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 64
DEFAULT_TIMEOUT_MS: Final[int] = 30_000

_MAX_TEXT_BYTES: Final[int] = 16 * 1024
_MAX_DEPTH: Final[int] = 12
_MAX_COST_COUNTERS: Final[int] = 32
_MAX_COST_VALUE: Final[int] = 2**63 - 1

_FORBIDDEN_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "body",
        "completion",
        "content",
        "decoded_model_output",
        "embedding",
        "file_contents",
        "graph",
        "model_output",
        "patch",
        "prompt",
        "raw",
        "raw_output",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
    }
)
_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "artifact_content_id",
        "artifact_id",
        "byte_count",
        "chunk_id",
        "cid",
        "dataset_id",
        "digest",
        "evidence_id",
        "kind",
        "media_type",
        "model_id",
        "path",
        "producer_id",
        "provider_id",
        "record_id",
        "reference_id",
        "revision",
        "score_millionths",
        "sha256",
        "summary",
        "symbol",
        "tree_id",
        "uri",
    }
)


class AnalysisTransportError(ValueError):
    """A caller-authored transport contract is invalid."""


class AnalysisCapabilityDriftError(AnalysisTransportError):
    """A provider response no longer matches the negotiated capability."""


class AnalysisProviderKind(str, Enum):
    LOCAL = "local"
    IPFS_DATASETS = "ipfs_datasets_py"
    OPTIONAL = "optional"


class AnalysisProviderHealth(str, Enum):
    LAZY = "lazy"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"

    @property
    def usable(self) -> bool:
        return self in {AnalysisProviderHealth.LAZY, AnalysisProviderHealth.HEALTHY}


class AnalysisTransportStatus(str, Enum):
    COMPLETED = "completed"
    FALLBACK = "fallback"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BACKPRESSURE = "backpressure"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    MALFORMED_OUTPUT = "malformed_output"
    MALFORMED = "malformed_output"
    CAPABILITY_DRIFT = "capability_drift"
    CAPABILITY_CHANGED = "capability_drift"
    PROVIDER_LOST = "provider_lost"
    FAILED = "failed"

    @property
    def successful(self) -> bool:
        return self in {
            AnalysisTransportStatus.COMPLETED,
            AnalysisTransportStatus.FALLBACK,
        }

    @property
    def terminal(self) -> bool:
        return True


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise AnalysisTransportError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AnalysisTransportError(f"{name} must not be empty")
    if "\x00" in result:
        raise AnalysisTransportError(f"{name} must not contain NUL bytes")
    if len(result.encode("utf-8")) > max_bytes:
        raise AnalysisTransportError(
            f"{name} exceeds the maximum of {max_bytes} UTF-8 bytes"
        )
    return result


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisTransportError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise AnalysisTransportError(f"{name} must be <= {maximum}")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise AnalysisTransportError(
            f"{name} must be one of: "
            + ", ".join(item.value for item in enum_type)
        ) from exc


def _canonical(value: Any, *, name: str = "value", depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise AnalysisTransportError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisTransportError(f"{name} must contain finite numbers")
        return format(value, ".17g")
    if isinstance(value, Enum):
        return _canonical(value.value, name=name, depth=depth + 1)
    if isinstance(value, datetime):
        candidate = value
        if candidate.tzinfo is None:
            raise AnalysisTransportError(f"{name} datetime must be timezone-aware")
        return candidate.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AnalysisTransportError(f"{name} keys must be strings")
        return {
            key: _canonical(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical(item, name=name, depth=depth + 1) for item in value
        ]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict(), name=name, depth=depth + 1)
    raise AnalysisTransportError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def _json_bytes(value: Any, *, name: str = "value") -> bytes:
    return json.dumps(
        _canonical(value, name=name),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(namespace: str, value: Any) -> str:
    return (
        f"{namespace}:sha256:"
        + hashlib.sha256(_json_bytes(value, name=namespace)).hexdigest()
    )


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(_canonical(value)))


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    for name in ("cancelled", "is_cancelled", "is_set"):
        value = getattr(token, name, None)
        if callable(value):
            try:
                return bool(value())
            except TypeError:
                continue
        if value is not None:
            return bool(value)
    return False


def _deadline_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise AnalysisTransportError("deadline must be timezone-aware")
        return value.timestamp()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisTransportError(
            "deadline must be a timezone-aware datetime or Unix timestamp"
        )
    if not math.isfinite(float(value)):
        raise AnalysisTransportError("deadline must be finite")
    return float(value)


def _normalize_references(
    values: Any,
    *,
    name: str,
    max_references: int,
    max_reference_bytes: int,
    truncate: bool,
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise AnalysisTransportError(f"{name} must be a sequence")
    output: list[Mapping[str, Any]] = []
    was_truncated = False
    for index, value in enumerate(source):
        if len(output) >= max_references:
            if truncate:
                was_truncated = True
                break
            raise AnalysisTransportError(
                f"{name} exceeds maximum count {max_references}"
            )
        if not isinstance(value, Mapping):
            raise AnalysisTransportError(f"{name}[{index}] must be an object")
        unknown = set(value) - _REFERENCE_FIELDS
        forbidden = set(value) & _FORBIDDEN_REFERENCE_FIELDS
        if forbidden:
            raise AnalysisTransportError(
                f"{name}[{index}] embeds forbidden payload fields: "
                + ", ".join(sorted(forbidden))
            )
        if unknown:
            raise AnalysisTransportError(
                f"{name}[{index}] contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        normalized = _canonical(value, name=f"{name}[{index}]")
        if not any(
            normalized.get(key)
            for key in (
                "reference_id",
                "artifact_id",
                "artifact_content_id",
                "evidence_id",
                "record_id",
                "cid",
                "digest",
                "sha256",
                "uri",
                "path",
            )
        ):
            raise AnalysisTransportError(
                f"{name}[{index}] must contain a content or location identity"
            )
        encoded = _json_bytes(normalized, name=f"{name}[{index}]")
        if len(encoded) > max_reference_bytes:
            if truncate:
                was_truncated = True
                continue
            raise AnalysisTransportError(
                f"{name}[{index}] exceeds {max_reference_bytes} bytes"
            )
        output.append(MappingProxyType(dict(normalized)))
    return tuple(output), was_truncated


def _normalize_cost(value: Any) -> Mapping[str, int]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AnalysisTransportError("cost must be an object of integer counters")
    if len(value) > _MAX_COST_COUNTERS:
        raise AnalysisTransportError(
            f"cost exceeds maximum counter count {_MAX_COST_COUNTERS}"
        )
    result: dict[str, int] = {}
    for key, item in sorted(value.items()):
        normalized_key = _text(key, "cost key", max_bytes=64)
        result[normalized_key] = _integer(
            item, f"cost.{normalized_key}", maximum=_MAX_COST_VALUE
        )
    return MappingProxyType(result)


def _forbidden_payload_paths(value: Any, *, path: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in _FORBIDDEN_REFERENCE_FIELDS:
                found.append(item_path)
            found.extend(_forbidden_payload_paths(item, path=item_path))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            found.extend(_forbidden_payload_paths(item, path=f"{path}[{index}]"))
    return tuple(found)


@dataclass(frozen=True)
class AnalysisTransportBounds:
    """Hard request, result, queue, and execution bounds."""

    max_question_bytes: int = DEFAULT_MAX_QUESTION_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_BYTES
    max_artifact_references: int = DEFAULT_MAX_REFERENCES
    max_evidence_references: int = DEFAULT_MAX_REFERENCES
    max_provenance_references: int = DEFAULT_MAX_REFERENCES
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    max_progress_events: int = DEFAULT_MAX_PROGRESS_EVENTS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        maxima = {
            "max_question_bytes": 1024 * 1024,
            "max_request_bytes": 4 * 1024 * 1024,
            "max_result_bytes": 16 * 1024 * 1024,
            "max_reference_bytes": 256 * 1024,
            "max_artifact_references": 4096,
            "max_evidence_references": 4096,
            "max_provenance_references": 4096,
            "max_batch_size": 256,
            "max_progress_events": 4096,
            "max_concurrency": 1024,
            "max_queue_size": 65_536,
            "timeout_ms": 10 * 60 * 1000,
        }
        for name, maximum in maxima.items():
            if name == "max_result_bytes":
                minimum = 4096
            else:
                minimum = (
                    0 if name in {"max_queue_size", "max_progress_events"} else 1
                )
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name), name, minimum=minimum, maximum=maximum
                ),
            )
        if self.max_question_bytes > self.max_request_bytes:
            raise AnalysisTransportError(
                "max_question_bytes cannot exceed max_request_bytes"
            )
        if self.max_reference_bytes > max(
            self.max_request_bytes, self.max_result_bytes
        ):
            raise AnalysisTransportError(
                "max_reference_bytes exceeds both request and result bounds"
            )

    @classmethod
    def from_value(
        cls, value: "AnalysisTransportBounds | Mapping[str, Any] | None"
    ) -> "AnalysisTransportBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("bounds must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisTransportError(
                "unknown bounds: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class AnalysisTransportPolicy:
    """Negotiation and fallback policy for one transport."""

    bounds: AnalysisTransportBounds = field(default_factory=AnalysisTransportBounds)
    protocol_versions: tuple[int, ...] = (ANALYSIS_TRANSPORT_PROTOCOL_VERSION,)
    request_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_REQUEST_SCHEMA,)
    result_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_RESULT_SCHEMA,)
    fallback_provider_id: str = ""
    fallback_statuses: tuple[AnalysisTransportStatus, ...] = (
        AnalysisTransportStatus.BACKPRESSURE,
        AnalysisTransportStatus.UNSUPPORTED,
        AnalysisTransportStatus.UNAVAILABLE,
        AnalysisTransportStatus.MALFORMED_OUTPUT,
        AnalysisTransportStatus.CAPABILITY_DRIFT,
        AnalysisTransportStatus.PROVIDER_LOST,
        AnalysisTransportStatus.FAILED,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bounds", AnalysisTransportBounds.from_value(self.bounds)
        )
        protocols = tuple(
            sorted(
                {
                    _integer(item, "protocol_versions", minimum=1, maximum=65_535)
                    for item in self.protocol_versions
                },
                reverse=True,
            )
        )
        if not protocols:
            raise AnalysisTransportError("protocol_versions must not be empty")
        object.__setattr__(self, "protocol_versions", protocols)
        for name in ("request_schemas", "result_schemas"):
            raw = getattr(self, name)
            if isinstance(raw, str) or not isinstance(raw, Sequence):
                raise AnalysisTransportError(f"{name} must be a sequence")
            values = tuple(
                sorted(
                    {
                        _text(item, name, max_bytes=256)
                        for item in raw
                    }
                )
            )
            if not values:
                raise AnalysisTransportError(f"{name} must not be empty")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "fallback_provider_id",
            _text(
                self.fallback_provider_id,
                "fallback_provider_id",
                required=False,
                max_bytes=256,
            ),
        )
        statuses = tuple(
            sorted(
                {
                    _enum(item, AnalysisTransportStatus, "fallback_statuses")
                    for item in self.fallback_statuses
                },
                key=lambda item: item.value,
            )
        )
        if any(
            item
            in {
                AnalysisTransportStatus.COMPLETED,
                AnalysisTransportStatus.FALLBACK,
                AnalysisTransportStatus.CANCELLED,
                AnalysisTransportStatus.TIMED_OUT,
            }
            for item in statuses
        ):
            raise AnalysisTransportError(
                "successful, cancelled, and timed-out statuses cannot trigger fallback"
            )
        object.__setattr__(self, "fallback_statuses", statuses)

    @classmethod
    def from_value(
        cls, value: "AnalysisTransportPolicy | Mapping[str, Any] | None"
    ) -> "AnalysisTransportPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("policy must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisTransportError(
                "unknown policy fields: " + ", ".join(sorted(unknown))
            )
        fields = dict(value)
        fields["bounds"] = AnalysisTransportBounds.from_value(fields.get("bounds"))
        return cls(**fields)


@dataclass(frozen=True)
class AnalysisCapability:
    """Side-effect-free declaration of one provider's maximum capabilities."""

    provider_id: str
    capability_revision: str
    operations: tuple[str, ...]
    provider_kind: AnalysisProviderKind = AnalysisProviderKind.LOCAL
    provider_version: str = "unknown"
    protocol_versions: tuple[int, ...] = (ANALYSIS_TRANSPORT_PROTOCOL_VERSION,)
    request_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_REQUEST_SCHEMA,)
    result_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_RESULT_SCHEMA,)
    health: AnalysisProviderHealth = AnalysisProviderHealth.LAZY
    max_batch_size: int = 1
    max_concurrency: int = 1
    supports_cancellation: bool = True
    supports_progress: bool = False
    supports_batching: bool = False

    def __post_init__(self) -> None:
        for name in ("provider_id", "capability_revision", "provider_version"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, max_bytes=256),
            )
        object.__setattr__(
            self,
            "provider_kind",
            _enum(self.provider_kind, AnalysisProviderKind, "provider_kind"),
        )
        if isinstance(self.operations, str) or not isinstance(
            self.operations, Sequence
        ):
            raise AnalysisTransportError("operations must be a sequence")
        operations = tuple(
            sorted({_text(item, "operations", max_bytes=256) for item in self.operations})
        )
        if not operations:
            raise AnalysisTransportError("operations must not be empty")
        object.__setattr__(self, "operations", operations)
        protocols = tuple(
            sorted(
                {
                    _integer(item, "protocol_versions", minimum=1, maximum=65_535)
                    for item in self.protocol_versions
                },
                reverse=True,
            )
        )
        if not protocols:
            raise AnalysisTransportError("protocol_versions must not be empty")
        object.__setattr__(self, "protocol_versions", protocols)
        for name in ("request_schemas", "result_schemas"):
            raw = getattr(self, name)
            if isinstance(raw, str) or not isinstance(raw, Sequence):
                raise AnalysisTransportError(f"{name} must be a sequence")
            values = tuple(
                sorted({_text(item, name, max_bytes=256) for item in raw})
            )
            if not values:
                raise AnalysisTransportError(f"{name} must not be empty")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self, "health", _enum(self.health, AnalysisProviderHealth, "health")
        )
        object.__setattr__(
            self,
            "max_batch_size",
            _integer(self.max_batch_size, "max_batch_size", minimum=1, maximum=256),
        )
        object.__setattr__(
            self,
            "max_concurrency",
            _integer(
                self.max_concurrency, "max_concurrency", minimum=1, maximum=1024
            ),
        )
        for name in (
            "supports_cancellation",
            "supports_progress",
            "supports_batching",
        ):
            if not isinstance(getattr(self, name), bool):
                raise AnalysisTransportError(f"{name} must be a boolean")
        if self.supports_batching is False and self.max_batch_size != 1:
            raise AnalysisTransportError(
                "max_batch_size must be 1 when batching is unsupported"
            )

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def capability_id(self) -> str:
        return _content_id("analysis-capability", self._payload())

    def supports(self, operation: str) -> bool:
        return operation in self.operations and self.health.usable

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_TRANSPORT_CAPABILITY_SCHEMA,
            "transport_version": ANALYSIS_TRANSPORT_VERSION,
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind.value,
            "provider_version": self.provider_version,
            "capability_revision": self.capability_revision,
            "operations": list(self.operations),
            "protocol_versions": list(self.protocol_versions),
            "request_schemas": list(self.request_schemas),
            "result_schemas": list(self.result_schemas),
            "health": self.health.value,
            "max_batch_size": self.max_batch_size,
            "max_concurrency": self.max_concurrency,
            "supports_cancellation": self.supports_cancellation,
            "supports_progress": self.supports_progress,
            "supports_batching": self.supports_batching,
            "non_authoritative": True,
            "completion_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, **self._payload()}

    @classmethod
    def from_value(
        cls,
        value: "AnalysisCapability | Mapping[str, Any]",
        *,
        provider_id: str = "",
    ) -> "AnalysisCapability":
        if isinstance(value, cls):
            if provider_id and value.provider_id != provider_id:
                raise AnalysisTransportError("capability provider_id does not match")
            return value
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("capability must be an object")
        if (
            "schema" in value
            and value.get("schema") != ANALYSIS_TRANSPORT_CAPABILITY_SCHEMA
        ):
            raise AnalysisTransportError("unsupported capability schema")
        if (
            "transport_version" in value
            and value.get("transport_version") != ANALYSIS_TRANSPORT_VERSION
        ):
            raise AnalysisTransportError("unsupported capability transport version")
        aliases = dict(value)
        aliases.pop("schema", None)
        aliases.pop("transport_version", None)
        aliases.pop("capability_id", None)
        aliases.pop("non_authoritative", None)
        aliases.pop("completion_authority", None)
        if "operation" in aliases and "operations" not in aliases:
            aliases["operations"] = (aliases.pop("operation"),)
        if "protocol_version" in aliases and "protocol_versions" not in aliases:
            aliases["protocol_versions"] = (aliases.pop("protocol_version"),)
        if "request_schema" in aliases and "request_schemas" not in aliases:
            aliases["request_schemas"] = (aliases.pop("request_schema"),)
        if "result_schema" in aliases and "result_schemas" not in aliases:
            aliases["result_schemas"] = (aliases.pop("result_schema"),)
        if "response_schema" in aliases and "result_schemas" not in aliases:
            aliases["result_schemas"] = (aliases.pop("response_schema"),)
        aliases.setdefault("provider_id", provider_id)
        unknown = set(aliases) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisTransportError(
                "capability contains unknown fields: " + ", ".join(sorted(unknown))
            )
        result = cls(**aliases)
        claimed = value.get("capability_id")
        if claimed is not None and claimed != result.capability_id:
            raise AnalysisTransportError("capability identity does not match")
        if value.get("non_authoritative", True) is not True:
            raise AnalysisTransportError("capability cannot claim authority")
        if value.get("completion_authority", False) is not False:
            raise AnalysisTransportError("capability cannot claim completion authority")
        return result


@dataclass(frozen=True)
class NegotiatedAnalysisCapability:
    provider_id: str
    capability_id: str
    capability_revision: str
    operation: str
    protocol_version: int
    request_schema: str
    result_schema: str
    max_batch_size: int
    supports_cancellation: bool
    supports_progress: bool
    supports_batching: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                getattr(self, name).value
                if isinstance(getattr(self, name), Enum)
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class AnalysisRequest:
    """Compact question and content-addressed inputs for one analysis."""

    operation: str
    question: str
    artifact_references: tuple[Mapping[str, Any], ...] = ()
    request_id: str = ""
    preferred_provider_id: str = ""
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    deadline: datetime | float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _text(self.operation, "operation", max_bytes=256)
        )
        object.__setattr__(
            self, "question", _text(self.question, "question", max_bytes=1024 * 1024)
        )
        references, _ = _normalize_references(
            self.artifact_references,
            name="artifact_references",
            max_references=4096,
            max_reference_bytes=256 * 1024,
            truncate=False,
        )
        object.__setattr__(self, "artifact_references", references)
        object.__setattr__(
            self,
            "preferred_provider_id",
            _text(
                self.preferred_provider_id,
                "preferred_provider_id",
                required=False,
                max_bytes=256,
            ),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _integer(self.timeout_ms, "timeout_ms", minimum=1, maximum=10 * 60 * 1000),
        )
        _deadline_timestamp(self.deadline)
        if not isinstance(self.metadata, Mapping):
            raise AnalysisTransportError("metadata must be an object")
        metadata = _canonical(self.metadata, name="metadata")
        forbidden_metadata = _forbidden_payload_paths(metadata)
        if forbidden_metadata:
            raise AnalysisTransportError(
                "metadata embeds forbidden payload fields: "
                + ", ".join(forbidden_metadata)
            )
        if len(_json_bytes(metadata, name="metadata")) > 8 * 1024:
            raise AnalysisTransportError("metadata exceeds 8192 bytes")
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata)))
        request_id = _text(
            self.request_id, "request_id", required=False, max_bytes=256
        )
        if not request_id:
            request_id = _content_id("analysis-request", self._identity_payload())
        object.__setattr__(self, "request_id", request_id)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
            "transport_version": ANALYSIS_TRANSPORT_VERSION,
            "operation": self.operation,
            "question": self.question,
            "artifact_references": [
                dict(item) for item in self.artifact_references
            ],
            "preferred_provider_id": self.preferred_provider_id,
            "timeout_ms": self.timeout_ms,
            "deadline": (
                _canonical(self.deadline, name="deadline")
                if self.deadline is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }

    def to_dict(
        self, negotiated: NegotiatedAnalysisCapability | None = None
    ) -> dict[str, Any]:
        payload = {"request_id": self.request_id, **self._identity_payload()}
        if negotiated is not None:
            payload.update(
                {
                    "schema": negotiated.request_schema,
                    "protocol_version": negotiated.protocol_version,
                    "capability_id": negotiated.capability_id,
                    "capability_revision": negotiated.capability_revision,
                }
            )
        return payload

    @classmethod
    def from_value(
        cls, value: "AnalysisRequest | Mapping[str, Any]"
    ) -> "AnalysisRequest":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("request must be an AnalysisRequest or object")
        fields = dict(value)
        for name in (
            "schema",
            "transport_version",
            "protocol_version",
            "capability_id",
            "capability_revision",
        ):
            fields.pop(name, None)
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise AnalysisTransportError(
                "request contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(**fields)

    def validate_bounds(self, bounds: AnalysisTransportBounds) -> None:
        if len(self.question.encode("utf-8")) > bounds.max_question_bytes:
            raise AnalysisTransportError(
                f"question exceeds {bounds.max_question_bytes} bytes"
            )
        _normalize_references(
            self.artifact_references,
            name="artifact_references",
            max_references=bounds.max_artifact_references,
            max_reference_bytes=bounds.max_reference_bytes,
            truncate=False,
        )
        if len(_json_bytes(self.to_dict(), name="request")) > bounds.max_request_bytes:
            raise AnalysisTransportError(
                f"request exceeds {bounds.max_request_bytes} bytes"
            )


@dataclass(frozen=True)
class AnalysisProgress:
    request_id: str
    sequence: int
    message: str
    completed_units: int = 0
    total_units: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id", max_bytes=256)
        )
        object.__setattr__(
            self, "sequence", _integer(self.sequence, "sequence", maximum=2**31 - 1)
        )
        object.__setattr__(
            self, "message", _text(self.message, "message", max_bytes=1024)
        )
        object.__setattr__(
            self,
            "completed_units",
            _integer(self.completed_units, "completed_units", maximum=_MAX_COST_VALUE),
        )
        object.__setattr__(
            self,
            "total_units",
            _integer(self.total_units, "total_units", maximum=_MAX_COST_VALUE),
        )
        if self.total_units and self.completed_units > self.total_units:
            raise AnalysisTransportError(
                "progress completed_units cannot exceed total_units"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_TRANSPORT_PROGRESS_SCHEMA,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "message": self.message,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
        }

    @classmethod
    def from_value(
        cls,
        value: "AnalysisProgress | Mapping[str, Any] | str",
        *,
        request_id: str,
        sequence: int,
    ) -> "AnalysisProgress":
        if isinstance(value, cls):
            if value.request_id != request_id:
                raise AnalysisTransportError("progress request_id does not match")
            return value
        if isinstance(value, str):
            return cls(request_id=request_id, sequence=sequence, message=value)
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("progress must be a string or object")
        allowed = {
            "schema",
            "request_id",
            "sequence",
            "message",
            "completed_units",
            "total_units",
        }
        if set(value) - allowed:
            raise AnalysisTransportError("progress contains unknown fields")
        return cls(
            request_id=value.get("request_id", request_id),
            sequence=value.get("sequence", sequence),
            message=value.get("message", ""),
            completed_units=value.get("completed_units", 0),
            total_units=value.get("total_units", 0),
        )


@dataclass(frozen=True)
class AnalysisResult:
    """One bounded terminal result.  Authority fields are derived and fixed."""

    request_id: str
    operation: str
    status: AnalysisTransportStatus
    reason_code: str
    provider_id: str = ""
    provider_kind: AnalysisProviderKind = AnalysisProviderKind.LOCAL
    capability_id: str = ""
    capability_revision: str = ""
    protocol_version: int = ANALYSIS_TRANSPORT_PROTOCOL_VERSION
    request_schema: str = ANALYSIS_TRANSPORT_REQUEST_SCHEMA
    result_schema: str = ANALYSIS_TRANSPORT_RESULT_SCHEMA
    evidence_references: tuple[Mapping[str, Any], ...] = ()
    provenance_references: tuple[Mapping[str, Any], ...] = ()
    cost: Mapping[str, int] = field(default_factory=dict)
    verdict: str = "inconclusive"
    truncated: bool = False
    progress: tuple[AnalysisProgress, ...] = ()
    progress_truncated: bool = False
    fallback_from_provider_id: str = ""
    fallback_attempted: bool = False
    fallback_reason_code: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "operation",
            "reason_code",
            "verdict",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, max_bytes=1024),
            )
        for name in (
            "provider_id",
            "capability_id",
            "capability_revision",
            "request_schema",
            "result_schema",
            "fallback_from_provider_id",
            "fallback_reason_code",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=False,
                    max_bytes=256,
                ),
            )
        object.__setattr__(
            self, "status", _enum(self.status, AnalysisTransportStatus, "status")
        )
        object.__setattr__(
            self,
            "provider_kind",
            _enum(self.provider_kind, AnalysisProviderKind, "provider_kind"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            _integer(
                self.protocol_version,
                "protocol_version",
                minimum=1,
                maximum=65_535,
            ),
        )
        evidence, evidence_truncated = _normalize_references(
            self.evidence_references,
            name="evidence_references",
            max_references=4096,
            max_reference_bytes=256 * 1024,
            truncate=True,
        )
        provenance, provenance_truncated = _normalize_references(
            self.provenance_references,
            name="provenance_references",
            max_references=4096,
            max_reference_bytes=256 * 1024,
            truncate=True,
        )
        object.__setattr__(self, "evidence_references", evidence)
        object.__setattr__(self, "provenance_references", provenance)
        object.__setattr__(self, "cost", _normalize_cost(self.cost))
        if not isinstance(self.truncated, bool):
            raise AnalysisTransportError("truncated must be a boolean")
        object.__setattr__(
            self,
            "truncated",
            self.truncated or evidence_truncated or provenance_truncated,
        )
        progress: list[AnalysisProgress] = []
        for index, value in enumerate(self.progress):
            progress.append(
                AnalysisProgress.from_value(
                    value, request_id=self.request_id, sequence=index
                )
            )
        object.__setattr__(self, "progress", tuple(progress))
        for name in ("progress_truncated", "fallback_attempted"):
            if not isinstance(getattr(self, name), bool):
                raise AnalysisTransportError(f"{name} must be a boolean")
        if self.status is AnalysisTransportStatus.FALLBACK:
            if not self.fallback_from_provider_id or not self.fallback_attempted:
                raise AnalysisTransportError(
                    "fallback results must identify the failed provider"
                )
        if self.status is not AnalysisTransportStatus.FALLBACK and (
            self.fallback_from_provider_id
        ):
            if not self.fallback_attempted:
                raise AnalysisTransportError(
                    "fallback provider identity requires fallback_attempted"
                )

    @property
    def successful(self) -> bool:
        return self.status.successful

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def authority(self) -> str:
        return "diagnostic"

    @property
    def provenance(self) -> tuple[Mapping[str, Any], ...]:
        return self.provenance_references

    @property
    def fallback_used(self) -> bool:
        return self.status is AnalysisTransportStatus.FALLBACK

    @property
    def result_id(self) -> str:
        return _content_id("analysis-transport-result", self._payload())

    @property
    def content_id(self) -> str:
        return self.result_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_TRANSPORT_RESULT_SCHEMA,
            "transport_version": ANALYSIS_TRANSPORT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind.value,
            "capability_id": self.capability_id,
            "capability_revision": self.capability_revision,
            "protocol_version": self.protocol_version,
            "request_schema": self.request_schema,
            "result_schema": self.result_schema,
            "evidence_references": [
                dict(item) for item in self.evidence_references
            ],
            "provenance_references": [
                dict(item) for item in self.provenance_references
            ],
            "cost": dict(self.cost),
            "verdict": self.verdict,
            "truncated": self.truncated,
            "progress": [item.to_dict() for item in self.progress],
            "progress_truncated": self.progress_truncated,
            "fallback_from_provider_id": self.fallback_from_provider_id,
            "fallback_attempted": self.fallback_attempted,
            "fallback_reason_code": self.fallback_reason_code,
            "non_authoritative": True,
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"result_id": self.result_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisResult":
        if not isinstance(value, Mapping):
            raise AnalysisTransportError("result must be an object")
        allowed = {
            "result_id",
            "schema",
            "transport_version",
            "request_id",
            "operation",
            "status",
            "reason_code",
            "provider_id",
            "provider_kind",
            "capability_id",
            "capability_revision",
            "protocol_version",
            "request_schema",
            "result_schema",
            "evidence_references",
            "provenance_references",
            "cost",
            "verdict",
            "truncated",
            "progress",
            "progress_truncated",
            "fallback_from_provider_id",
            "fallback_attempted",
            "fallback_reason_code",
            "non_authoritative",
            "completion_authority",
            "safe_for_completion_reasoning",
        }
        if set(value) - allowed:
            raise AnalysisTransportError("result contains unknown fields")
        if value.get("schema") != ANALYSIS_TRANSPORT_RESULT_SCHEMA:
            raise AnalysisTransportError("unsupported result schema")
        if value.get("transport_version") != ANALYSIS_TRANSPORT_VERSION:
            raise AnalysisTransportError("unsupported transport version")
        fixed = {
            "non_authoritative": True,
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
        }
        for name, expected in fixed.items():
            if value.get(name) is not expected:
                raise AnalysisTransportError(f"result {name} claim does not match")
        result = cls(
            **{
                name: value.get(name)
                for name in cls.__dataclass_fields__
                if name in value
            }
        )
        if value.get("result_id") != result.result_id:
            raise AnalysisTransportError("result identity does not match")
        return result

    def __bool__(self) -> bool:
        raise TypeError("AnalysisResult has no truth value; inspect status explicitly")


@dataclass(frozen=True)
class AnalysisTransportHealth:
    providers: tuple[AnalysisCapability, ...]
    active_requests: int
    queued_requests: int
    max_concurrency: int
    max_queue_size: int
    accepted_requests: int
    rejected_requests: int
    completed_requests: int
    failed_requests: int

    @property
    def healthy(self) -> bool:
        return (
            self.active_requests < self.max_concurrency
            and self.queued_requests <= self.max_queue_size
            and any(item.health.usable for item in self.providers)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "max_concurrency": self.max_concurrency,
            "max_queue_size": self.max_queue_size,
            "accepted_requests": self.accepted_requests,
            "rejected_requests": self.rejected_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "providers": [item.to_dict() for item in self.providers],
        }


class AnalysisCancellationToken:
    """Small thread/async-compatible cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        while not self._event.is_set():
            await asyncio.sleep(0.01)


@dataclass
class _ProviderRegistration:
    capability: AnalysisCapability
    factory: Callable[[], Any] | None = None
    instance: Any = None
    activated: bool = False
    runtime_health: AnalysisProviderHealth | None = None
    failures: int = 0


@dataclass
class _Metrics:
    accepted_requests: int = 0
    rejected_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0


class _ProgressCollector:
    def __init__(
        self,
        request_id: str,
        maximum: int,
        callback: Callable[[AnalysisProgress], Any] | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.request_id = request_id
        self.maximum = maximum
        self.callback = callback
        self.loop = loop
        self.events: list[AnalysisProgress] = []
        self.truncated = False
        self._lock = threading.Lock()

    def __call__(self, value: Any) -> None:
        with self._lock:
            if len(self.events) >= self.maximum:
                self.truncated = True
                return
            try:
                event = AnalysisProgress.from_value(
                    value,
                    request_id=self.request_id,
                    sequence=len(self.events),
                )
            except AnalysisTransportError:
                self.truncated = True
                return
            self.events.append(event)
        if self.callback is not None:
            self.loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: AnalysisProgress) -> None:
        if self.callback is None:
            return
        try:
            result = self.callback(event)
            if inspect.isawaitable(result):
                asyncio.create_task(result)
        except Exception:
            # Progress observers are informational and cannot fail dispatch.
            return


class AnalysisTransport:
    """Capability-negotiated async dispatcher with deterministic local fallback."""

    def __init__(
        self,
        *,
        policy: AnalysisTransportPolicy | Mapping[str, Any] | None = None,
        local_provider: Any = None,
        local_capability: AnalysisCapability | Mapping[str, Any] | None = None,
        optional_provider_factory: Callable[[], Any] | None = None,
        optional_capability: AnalysisCapability | Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = AnalysisTransportPolicy.from_value(policy)
        self._providers: dict[str, _ProviderRegistration] = {}
        self._provider_order: list[str] = []
        self._semaphore = asyncio.Semaphore(self.policy.bounds.max_concurrency)
        self._state_lock = asyncio.Lock()
        self._active = 0
        self._queued = 0
        self._metrics = _Metrics()
        if local_provider is not None or local_capability is not None:
            if local_provider is None or local_capability is None:
                raise AnalysisTransportError(
                    "local_provider and local_capability must be supplied together"
                )
            self.register_provider(local_capability, provider=local_provider)
        if optional_provider_factory is not None or optional_capability is not None:
            if optional_provider_factory is None or optional_capability is None:
                raise AnalysisTransportError(
                    "optional_provider_factory and optional_capability "
                    "must be supplied together"
                )
            self.register_lazy_provider(
                optional_capability, factory=optional_provider_factory
            )

    def register_provider(
        self,
        capability: AnalysisCapability | Mapping[str, Any],
        *,
        provider: Any,
        replace_existing: bool = False,
    ) -> None:
        declared = AnalysisCapability.from_value(capability)
        if provider is None:
            raise AnalysisTransportError("provider must not be None")
        self._register(
            _ProviderRegistration(
                capability=declared,
                instance=provider,
                activated=True,
                runtime_health=declared.health,
            ),
            replace_existing=replace_existing,
        )

    register_local_provider = register_provider

    def register_lazy_provider(
        self,
        capability: AnalysisCapability | Mapping[str, Any],
        *,
        factory: Callable[[], Any],
        replace_existing: bool = False,
    ) -> None:
        declared = AnalysisCapability.from_value(capability)
        if not callable(factory):
            raise AnalysisTransportError("provider factory must be callable")
        self._register(
            _ProviderRegistration(capability=declared, factory=factory),
            replace_existing=replace_existing,
        )

    def register_optional_module(
        self,
        capability: AnalysisCapability | Mapping[str, Any],
        *,
        module_name: str = "ipfs_datasets_py",
        attribute: str = "analysis_provider",
        replace_existing: bool = False,
    ) -> None:
        """Register an optional module without importing or probing it."""

        declared = AnalysisCapability.from_value(capability)
        normalized_module = _text(module_name, "module_name", max_bytes=512)
        normalized_attribute = _text(attribute, "attribute", max_bytes=256)

        def load() -> Any:
            module = importlib.import_module(normalized_module)
            provider = getattr(module, normalized_attribute)
            return provider() if inspect.isclass(provider) else provider

        self.register_lazy_provider(
            declared, factory=load, replace_existing=replace_existing
        )

    register_ipfs_datasets_provider = register_optional_module

    def _register(
        self,
        registration: _ProviderRegistration,
        *,
        replace_existing: bool,
    ) -> None:
        provider_id = registration.capability.provider_id
        if provider_id in self._providers and not replace_existing:
            raise AnalysisTransportError(f"provider already registered: {provider_id}")
        if provider_id not in self._providers:
            self._provider_order.append(provider_id)
        self._providers[provider_id] = registration
        if (
            not self.policy.fallback_provider_id
            and registration.capability.provider_kind is AnalysisProviderKind.LOCAL
        ):
            self.policy = replace(self.policy, fallback_provider_id=provider_id)

    def discover_capabilities(
        self, operation: str | None = None
    ) -> tuple[AnalysisCapability, ...]:
        """Return declarations only; never activate, import, or probe a provider."""

        normalized = (
            _text(operation, "operation", max_bytes=256)
            if operation is not None
            else None
        )
        return tuple(
            registration.capability
            for provider_id in self._provider_order
            for registration in (self._providers[provider_id],)
            if normalized is None or normalized in registration.capability.operations
        )

    discover = discover_capabilities

    async def discover_capabilities_async(
        self, operation: str | None = None
    ) -> tuple[AnalysisCapability, ...]:
        """Async facade that retains declaration-only discovery semantics."""

        return self.discover_capabilities(operation)

    discover_async = discover_capabilities_async

    def negotiate(
        self,
        provider_id: str,
        operation: str,
    ) -> NegotiatedAnalysisCapability | None:
        normalized_id = _text(provider_id, "provider_id", max_bytes=256)
        normalized_operation = _text(operation, "operation", max_bytes=256)
        registration = self._providers.get(normalized_id)
        if registration is None:
            return None
        capability = registration.capability
        if not capability.supports(normalized_operation):
            return None
        protocols = [
            item
            for item in self.policy.protocol_versions
            if item in capability.protocol_versions
        ]
        request_schemas = [
            item
            for item in self.policy.request_schemas
            if item in capability.request_schemas
        ]
        result_schemas = [
            item
            for item in self.policy.result_schemas
            if item in capability.result_schemas
        ]
        if not protocols or not request_schemas or not result_schemas:
            return None
        return NegotiatedAnalysisCapability(
            provider_id=capability.provider_id,
            capability_id=capability.capability_id,
            capability_revision=capability.capability_revision,
            operation=normalized_operation,
            protocol_version=max(protocols),
            request_schema=request_schemas[0],
            result_schema=result_schemas[0],
            max_batch_size=min(
                capability.max_batch_size, self.policy.bounds.max_batch_size
            ),
            supports_cancellation=capability.supports_cancellation,
            supports_progress=capability.supports_progress,
            supports_batching=capability.supports_batching,
        )

    def health_snapshot(self) -> AnalysisTransportHealth:
        capabilities = []
        for provider_id in self._provider_order:
            registration = self._providers[provider_id]
            health = registration.runtime_health
            capabilities.append(
                replace(registration.capability, health=health)
                if health is not None and health is not registration.capability.health
                else registration.capability
            )
        return AnalysisTransportHealth(
            providers=tuple(capabilities),
            active_requests=self._active,
            queued_requests=self._queued,
            max_concurrency=self.policy.bounds.max_concurrency,
            max_queue_size=self.policy.bounds.max_queue_size,
            accepted_requests=self._metrics.accepted_requests,
            rejected_requests=self._metrics.rejected_requests,
            completed_requests=self._metrics.completed_requests,
            failed_requests=self._metrics.failed_requests,
        )

    health = health_snapshot

    async def dispatch(
        self,
        request: AnalysisRequest | Mapping[str, Any],
        *,
        provider_id: str | None = None,
        timeout_ms: int | None = None,
        deadline: datetime | float | None = None,
        cancellation_token: Any = None,
        progress_callback: Callable[[AnalysisProgress], Any] | None = None,
    ) -> AnalysisResult:
        normalized = AnalysisRequest.from_value(request)
        normalized.validate_bounds(self.policy.bounds)
        absolute_deadline = self._effective_deadline(
            normalized, timeout_ms=timeout_ms, deadline=deadline
        )
        if _cancelled(cancellation_token):
            return self._terminal(
                normalized, AnalysisTransportStatus.CANCELLED, "cancelled_before_admission"
            )
        admitted = await self._admit(absolute_deadline, cancellation_token)
        if admitted is not None:
            self._metrics.rejected_requests += 1
            return self._terminal(normalized, admitted[0], admitted[1])
        self._metrics.accepted_requests += 1
        try:
            result = await self._dispatch_admitted(
                normalized,
                provider_id=provider_id,
                absolute_deadline=absolute_deadline,
                cancellation_token=cancellation_token,
                progress_callback=progress_callback,
            )
            if result.successful:
                self._metrics.completed_requests += 1
            else:
                self._metrics.failed_requests += 1
            return result
        finally:
            await self._release()

    async def dispatch_batch(
        self,
        requests: Sequence[AnalysisRequest | Mapping[str, Any]],
        *,
        provider_id: str | None = None,
        timeout_ms: int | None = None,
        deadline: datetime | float | None = None,
        cancellation_token: Any = None,
        progress_callback: Callable[[AnalysisProgress], Any] | None = None,
    ) -> tuple[AnalysisResult, ...]:
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise AnalysisTransportError("requests must be a sequence")
        if not requests:
            raise AnalysisTransportError("requests must not be empty")
        if len(requests) > self.policy.bounds.max_batch_size:
            raise AnalysisTransportError(
                f"batch exceeds {self.policy.bounds.max_batch_size} requests"
            )
        normalized = tuple(AnalysisRequest.from_value(item) for item in requests)
        for request in normalized:
            request.validate_bounds(self.policy.bounds)

        # Native batching is used only when every member resolves to one
        # compatible provider and capability. Otherwise the public batch API
        # still preserves bounded asynchronous member dispatch.
        selected = (
            provider_id
            or normalized[0].preferred_provider_id
            or self._select_provider(normalized[0], None)
        )
        if selected and all(
            (not item.preferred_provider_id or item.preferred_provider_id == selected)
            for item in normalized
        ):
            negotiations = tuple(
                self.negotiate(selected, item.operation) for item in normalized
            )
            first = negotiations[0]
            compatible = (
                first is not None
                and first.supports_batching
                and len(normalized) <= first.max_batch_size
                and all(
                    item is not None
                    and item.provider_id == first.provider_id
                    and item.protocol_version == first.protocol_version
                    and item.request_schema == first.request_schema
                    and item.result_schema == first.result_schema
                    and item.capability_revision == first.capability_revision
                    for item in negotiations
                )
            )
            if compatible:
                return await self._dispatch_native_batch(
                    normalized,
                    provider_id=selected,
                    negotiations=tuple(
                        item for item in negotiations if item is not None
                    ),
                    timeout_ms=timeout_ms,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                    progress_callback=progress_callback,
                )

        return tuple(
            await asyncio.gather(
                *(
                    self.dispatch(
                        item,
                        provider_id=provider_id,
                        timeout_ms=timeout_ms,
                        deadline=deadline,
                        cancellation_token=cancellation_token,
                        progress_callback=progress_callback,
                    )
                    for item in normalized
                )
            )
        )

    dispatch_many = dispatch_batch
    analyze = dispatch
    analyze_async = dispatch
    analyze_batch = dispatch_batch
    analyze_batch_async = dispatch_batch

    async def _dispatch_native_batch(
        self,
        requests: tuple[AnalysisRequest, ...],
        *,
        provider_id: str,
        negotiations: tuple[NegotiatedAnalysisCapability, ...],
        timeout_ms: int | None,
        deadline: datetime | float | None,
        cancellation_token: Any,
        progress_callback: Callable[[AnalysisProgress], Any] | None,
    ) -> tuple[AnalysisResult, ...]:
        deadlines = [
            self._effective_deadline(item, timeout_ms=timeout_ms, deadline=deadline)
            for item in requests
        ]
        absolute_deadline = min(deadlines)
        admitted = await self._admit(absolute_deadline, cancellation_token)
        if admitted is not None:
            self._metrics.rejected_requests += len(requests)
            return tuple(self._terminal(item, admitted[0], admitted[1]) for item in requests)
        self._metrics.accepted_requests += len(requests)
        try:
            registration = self._providers[provider_id]
            activation = await self._activate_and_validate(
                registration, negotiations[0]
            )
            if isinstance(activation, tuple):
                status, reason = activation
                primary = tuple(
                    self._terminal(
                        item,
                        status,
                        reason,
                        provider=registration.capability,
                        negotiated=negotiations[index],
                    )
                    for index, item in enumerate(requests)
                )
                return await self._fallback_batch(
                    requests, primary, absolute_deadline, cancellation_token
                )
            provider = activation
            method = self._provider_method(provider, batch=True)
            if method is None:
                primary = tuple(
                    self._terminal(
                        item,
                        AnalysisTransportStatus.UNSUPPORTED,
                        "batch_dispatch_unavailable",
                        provider=registration.capability,
                        negotiated=negotiations[index],
                    )
                    for index, item in enumerate(requests)
                )
                return await self._fallback_batch(
                    requests, primary, absolute_deadline, cancellation_token
                )
            collectors = {
                item.request_id: _ProgressCollector(
                    item.request_id,
                    self.policy.bounds.max_progress_events,
                    progress_callback,
                    asyncio.get_running_loop(),
                )
                for item in requests
            }

            def batch_progress(value: Any) -> None:
                if isinstance(value, Mapping):
                    request_id = value.get("request_id")
                    if request_id in collectors:
                        collectors[str(request_id)](value)

            started = time.monotonic()
            outcome, raw = await self._invoke_controlled(
                method,
                tuple(requests),
                absolute_deadline=absolute_deadline,
                cancellation_token=cancellation_token,
                progress=batch_progress,
                negotiated=negotiations[0],
            )
            elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
            if outcome is not None:
                status, reason = outcome
                primary = tuple(
                    self._terminal(
                        item,
                        status,
                        reason,
                        provider=registration.capability,
                        negotiated=negotiations[index],
                        cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                        progress=tuple(collectors[item.request_id].events),
                        progress_truncated=collectors[item.request_id].truncated,
                    )
                    for index, item in enumerate(requests)
                )
                self._record_provider_failure(registration, status)
                return await self._fallback_batch(
                    requests, primary, absolute_deadline, cancellation_token
                )
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                status_reason = (
                    AnalysisTransportStatus.MALFORMED_OUTPUT,
                    "batch_result_not_sequence",
                )
                primary = tuple(
                    self._terminal(
                        item,
                        *status_reason,
                        provider=registration.capability,
                        negotiated=negotiations[index],
                    )
                    for index, item in enumerate(requests)
                )
                self._record_provider_failure(
                    registration, AnalysisTransportStatus.MALFORMED_OUTPUT
                )
                return await self._fallback_batch(
                    requests, primary, absolute_deadline, cancellation_token
                )
            if len(raw) != len(requests):
                primary = tuple(
                    self._terminal(
                        item,
                        AnalysisTransportStatus.MALFORMED_OUTPUT,
                        "batch_result_count_mismatch",
                        provider=registration.capability,
                        negotiated=negotiations[index],
                    )
                    for index, item in enumerate(requests)
                )
                return await self._fallback_batch(
                    requests, primary, absolute_deadline, cancellation_token
                )
            results = []
            for index, (request, value) in enumerate(zip(requests, raw)):
                collector = collectors[request.request_id]
                try:
                    result = self._normalize_provider_result(
                        request,
                        value,
                        registration.capability,
                        negotiations[index],
                        elapsed_ms=elapsed_ms,
                        progress=tuple(collector.events),
                        progress_truncated=collector.truncated,
                    )
                except AnalysisCapabilityDriftError:
                    result = self._terminal(
                        request,
                        AnalysisTransportStatus.CAPABILITY_DRIFT,
                        "capability_drift",
                        provider=registration.capability,
                        negotiated=negotiations[index],
                        cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                        progress=tuple(collector.events),
                        progress_truncated=collector.truncated,
                    )
                except AnalysisTransportError:
                    result = self._terminal(
                        request,
                        AnalysisTransportStatus.MALFORMED_OUTPUT,
                        "malformed_provider_output",
                        provider=registration.capability,
                        negotiated=negotiations[index],
                        cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                        progress=tuple(collector.events),
                        progress_truncated=collector.truncated,
                    )
                results.append(result)
            registration.runtime_health = AnalysisProviderHealth.HEALTHY
            completed_items: list[AnalysisResult] = []
            for request, result in zip(requests, results):
                if not result.successful:
                    result = await self._maybe_fallback(
                        request,
                        result,
                        absolute_deadline=absolute_deadline,
                        cancellation_token=cancellation_token,
                        progress_callback=None,
                    )
                completed_items.append(result)
            completed = tuple(completed_items)
            if all(item.successful for item in results):
                registration.failures = 0
            else:
                first_failure = next(
                    item.status for item in results if not item.successful
                )
                self._record_provider_failure(
                    registration, first_failure
                )
            failed = sum(not item.successful for item in completed)
            self._metrics.completed_requests += len(completed) - failed
            self._metrics.failed_requests += failed
            return completed
        finally:
            await self._release()

    async def _fallback_batch(
        self,
        requests: tuple[AnalysisRequest, ...],
        primary: tuple[AnalysisResult, ...],
        absolute_deadline: float,
        cancellation_token: Any,
    ) -> tuple[AnalysisResult, ...]:
        results = []
        for request, failure in zip(requests, primary):
            results.append(
                await self._maybe_fallback(
                    request,
                    failure,
                    absolute_deadline=absolute_deadline,
                    cancellation_token=cancellation_token,
                    progress_callback=None,
                )
            )
        failed = sum(not item.successful for item in results)
        self._metrics.completed_requests += len(results) - failed
        self._metrics.failed_requests += failed
        return tuple(results)

    async def _dispatch_admitted(
        self,
        request: AnalysisRequest,
        *,
        provider_id: str | None,
        absolute_deadline: float,
        cancellation_token: Any,
        progress_callback: Callable[[AnalysisProgress], Any] | None,
    ) -> AnalysisResult:
        selected = self._select_provider(request, provider_id)
        if selected is None:
            primary = self._terminal(
                request, AnalysisTransportStatus.UNAVAILABLE, "no_provider_available"
            )
        else:
            primary = await self._attempt(
                request,
                selected,
                absolute_deadline=absolute_deadline,
                cancellation_token=cancellation_token,
                progress_callback=progress_callback,
            )
        return await self._maybe_fallback(
            request,
            primary,
            absolute_deadline=absolute_deadline,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )

    def _select_provider(
        self, request: AnalysisRequest, provider_id: str | None
    ) -> str | None:
        explicit = provider_id or request.preferred_provider_id
        if explicit:
            normalized = _text(explicit, "provider_id", max_bytes=256)
            return normalized if normalized in self._providers else None
        fallback_id = self.policy.fallback_provider_id
        for candidate in self._provider_order:
            if candidate == fallback_id:
                continue
            if self.negotiate(candidate, request.operation) is not None:
                return candidate
        if fallback_id and fallback_id in self._providers:
            return fallback_id
        for candidate in self._provider_order:
            if self.negotiate(candidate, request.operation) is not None:
                return candidate
        return None

    async def _attempt(
        self,
        request: AnalysisRequest,
        provider_id: str,
        *,
        absolute_deadline: float,
        cancellation_token: Any,
        progress_callback: Callable[[AnalysisProgress], Any] | None,
    ) -> AnalysisResult:
        registration = self._providers.get(provider_id)
        if registration is None:
            return self._terminal(
                request, AnalysisTransportStatus.UNAVAILABLE, "provider_not_registered"
            )
        negotiated = self.negotiate(provider_id, request.operation)
        if negotiated is None:
            return self._terminal(
                request,
                AnalysisTransportStatus.UNSUPPORTED,
                "capability_negotiation_failed",
                provider=registration.capability,
            )
        if _cancelled(cancellation_token):
            return self._terminal(
                request,
                AnalysisTransportStatus.CANCELLED,
                "cancelled_before_activation",
                provider=registration.capability,
                negotiated=negotiated,
            )
        if time.time() >= absolute_deadline:
            return self._terminal(
                request,
                AnalysisTransportStatus.TIMED_OUT,
                "deadline_expired_before_activation",
                provider=registration.capability,
                negotiated=negotiated,
            )
        activation = await self._activate_and_validate(registration, negotiated)
        if isinstance(activation, tuple):
            status, reason = activation
            self._record_provider_failure(registration, status)
            return self._terminal(
                request,
                status,
                reason,
                provider=registration.capability,
                negotiated=negotiated,
            )
        provider = activation
        method = self._provider_method(provider, batch=False)
        if method is None:
            return self._terminal(
                request,
                AnalysisTransportStatus.UNSUPPORTED,
                "dispatch_method_unavailable",
                provider=registration.capability,
                negotiated=negotiated,
            )
        collector = _ProgressCollector(
            request.request_id,
            self.policy.bounds.max_progress_events,
            progress_callback,
            asyncio.get_running_loop(),
        )
        started = time.monotonic()
        outcome, raw = await self._invoke_controlled(
            method,
            request,
            absolute_deadline=absolute_deadline,
            cancellation_token=cancellation_token,
            progress=collector if negotiated.supports_progress else None,
            negotiated=negotiated,
        )
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        if outcome is not None:
            status, reason = outcome
            self._record_provider_failure(registration, status)
            return self._terminal(
                request,
                status,
                reason,
                provider=registration.capability,
                negotiated=negotiated,
                cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                progress=tuple(collector.events),
                progress_truncated=collector.truncated,
            )
        try:
            result = self._normalize_provider_result(
                request,
                raw,
                registration.capability,
                negotiated,
                elapsed_ms=elapsed_ms,
                progress=tuple(collector.events),
                progress_truncated=collector.truncated,
            )
        except AnalysisCapabilityDriftError:
            self._record_provider_failure(
                registration, AnalysisTransportStatus.CAPABILITY_DRIFT
            )
            return self._terminal(
                request,
                AnalysisTransportStatus.CAPABILITY_DRIFT,
                "capability_drift",
                provider=registration.capability,
                negotiated=negotiated,
                cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                progress=tuple(collector.events),
                progress_truncated=collector.truncated,
            )
        except AnalysisTransportError:
            self._record_provider_failure(
                registration, AnalysisTransportStatus.MALFORMED_OUTPUT
            )
            return self._terminal(
                request,
                AnalysisTransportStatus.MALFORMED_OUTPUT,
                "malformed_provider_output",
                provider=registration.capability,
                negotiated=negotiated,
                cost={"wall_time_ms": elapsed_ms, "provider_calls": 1},
                progress=tuple(collector.events),
                progress_truncated=collector.truncated,
            )
        registration.runtime_health = AnalysisProviderHealth.HEALTHY
        registration.failures = 0
        return result

    async def _maybe_fallback(
        self,
        request: AnalysisRequest,
        primary: AnalysisResult,
        *,
        absolute_deadline: float,
        cancellation_token: Any,
        progress_callback: Callable[[AnalysisProgress], Any] | None,
    ) -> AnalysisResult:
        fallback_id = self.policy.fallback_provider_id
        if (
            primary.successful
            or primary.status not in self.policy.fallback_statuses
            or not fallback_id
            or primary.provider_id == fallback_id
            or fallback_id not in self._providers
            or _cancelled(cancellation_token)
            or time.time() >= absolute_deadline
        ):
            return primary
        fallback = await self._attempt(
            request,
            fallback_id,
            absolute_deadline=absolute_deadline,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        if fallback.status is AnalysisTransportStatus.COMPLETED:
            combined_cost = dict(fallback.cost)
            for key, value in primary.cost.items():
                combined_cost[key] = min(
                    _MAX_COST_VALUE, combined_cost.get(key, 0) + value
                )
            return replace(
                fallback,
                status=AnalysisTransportStatus.FALLBACK,
                reason_code=primary.reason_code,
                cost=combined_cost,
                fallback_from_provider_id=primary.provider_id or "unresolved",
                fallback_attempted=True,
                fallback_reason_code=primary.status.value,
            )
        return replace(
            primary,
            fallback_from_provider_id=fallback.provider_id or fallback_id,
            fallback_attempted=True,
            fallback_reason_code=f"{fallback.status.value}:{fallback.reason_code}",
        )

    async def _activate_and_validate(
        self,
        registration: _ProviderRegistration,
        negotiated: NegotiatedAnalysisCapability,
    ) -> Any | tuple[AnalysisTransportStatus, str]:
        if registration.runtime_health in {
            AnalysisProviderHealth.UNAVAILABLE,
            AnalysisProviderHealth.INCOMPATIBLE,
        }:
            return (
                AnalysisTransportStatus.UNAVAILABLE,
                "provider_health_unavailable",
            )
        if not registration.activated:
            try:
                assert registration.factory is not None
                instance = registration.factory()
                if inspect.isawaitable(instance):
                    instance = await instance
                if instance is None:
                    raise LookupError("provider factory returned None")
                registration.instance = instance
                registration.activated = True
            except (ImportError, ModuleNotFoundError, AttributeError, LookupError):
                registration.runtime_health = AnalysisProviderHealth.UNAVAILABLE
                return (
                    AnalysisTransportStatus.UNAVAILABLE,
                    "provider_activation_unavailable",
                )
            except Exception:
                registration.runtime_health = AnalysisProviderHealth.DEGRADED
                return AnalysisTransportStatus.FAILED, "provider_activation_failed"
        provider = registration.instance
        try:
            runtime_capability = await self._runtime_capability(
                provider, registration.capability
            )
        except Exception:
            registration.runtime_health = AnalysisProviderHealth.DEGRADED
            return AnalysisTransportStatus.MALFORMED_OUTPUT, "malformed_capability"
        if runtime_capability.health in {
            AnalysisProviderHealth.UNAVAILABLE,
            AnalysisProviderHealth.INCOMPATIBLE,
        }:
            registration.runtime_health = runtime_capability.health
            return AnalysisTransportStatus.UNAVAILABLE, "provider_became_unavailable"
        if not self._capability_matches(
            registration.capability, runtime_capability, negotiated
        ):
            registration.runtime_health = AnalysisProviderHealth.INCOMPATIBLE
            return AnalysisTransportStatus.CAPABILITY_DRIFT, "capability_drift"
        registration.runtime_health = AnalysisProviderHealth.HEALTHY
        return provider

    async def _runtime_capability(
        self, provider: Any, declared: AnalysisCapability
    ) -> AnalysisCapability:
        raw: Any = None
        found = False
        for name in (
            "runtime_capability",
            "get_capability",
            "capability",
            "capabilities",
        ):
            if not hasattr(provider, name):
                continue
            raw = getattr(provider, name)
            found = True
            if callable(raw):
                raw = raw()
            if inspect.isawaitable(raw):
                raw = await raw
            break
        if not found or raw is None:
            return declared
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
            matches = [
                item
                for item in raw
                if (
                    isinstance(item, AnalysisCapability)
                    and item.provider_id == declared.provider_id
                )
                or (
                    isinstance(item, Mapping)
                    and item.get("provider_id", declared.provider_id)
                    == declared.provider_id
                )
            ]
            if len(matches) != 1:
                raise AnalysisTransportError(
                    "runtime capabilities must contain exactly one provider declaration"
                )
            raw = matches[0]
        return AnalysisCapability.from_value(raw, provider_id=declared.provider_id)

    @staticmethod
    def _capability_matches(
        declared: AnalysisCapability,
        runtime: AnalysisCapability,
        negotiated: NegotiatedAnalysisCapability,
    ) -> bool:
        return (
            runtime.provider_id == declared.provider_id
            and runtime.capability_revision == declared.capability_revision
            and runtime.provider_kind is declared.provider_kind
            and runtime.provider_version == declared.provider_version
            and runtime.operations == declared.operations
            and runtime.protocol_versions == declared.protocol_versions
            and runtime.request_schemas == declared.request_schemas
            and runtime.result_schemas == declared.result_schemas
            and runtime.max_batch_size == declared.max_batch_size
            and runtime.max_concurrency == declared.max_concurrency
            and runtime.supports_cancellation == declared.supports_cancellation
            and runtime.supports_progress == declared.supports_progress
            and runtime.supports_batching == declared.supports_batching
            and negotiated.operation in runtime.operations
            and negotiated.protocol_version in runtime.protocol_versions
            and negotiated.request_schema in runtime.request_schemas
            and negotiated.result_schema in runtime.result_schemas
        )

    @staticmethod
    def _provider_method(provider: Any, *, batch: bool) -> Callable[..., Any] | None:
        names = (
            ("analyze_batch", "dispatch_batch", "batch")
            if batch
            else ("analyze", "dispatch", "__call__")
        )
        for name in names:
            method = getattr(provider, name, None)
            if callable(method):
                return method
        return provider if not batch and callable(provider) else None

    async def _invoke_controlled(
        self,
        method: Callable[..., Any],
        payload: Any,
        *,
        absolute_deadline: float,
        cancellation_token: Any,
        progress: Callable[[Any], None] | None,
        negotiated: NegotiatedAnalysisCapability,
    ) -> tuple[
        tuple[AnalysisTransportStatus, str] | None,
        Any,
    ]:
        remaining = absolute_deadline - time.time()
        if remaining <= 0:
            return (
                AnalysisTransportStatus.TIMED_OUT,
                "deadline_expired_before_dispatch",
            ), None
        if _cancelled(cancellation_token):
            return (
                AnalysisTransportStatus.CANCELLED,
                "cancelled_before_dispatch",
            ), None

        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(method)
            supports_kwargs = any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in signature.parameters.values()
            )
            for name, value in (
                (
                    "cancellation_token",
                    cancellation_token if negotiated.supports_cancellation else None,
                ),
                ("progress", progress),
                ("progress_callback", progress),
                ("negotiated_capability", negotiated),
            ):
                if value is not None and (supports_kwargs or name in signature.parameters):
                    kwargs[name] = value
        except (TypeError, ValueError):
            kwargs = {}

        async def invoke() -> Any:
            if inspect.iscoroutinefunction(method):
                return await method(payload, **kwargs)
            value = await asyncio.to_thread(method, payload, **kwargs)
            if inspect.isawaitable(value):
                return await value
            return value

        task = asyncio.create_task(invoke())
        cancellation_waiter = asyncio.create_task(
            self._wait_for_cancellation(cancellation_token)
        )
        try:
            done, _ = await asyncio.wait(
                {task, cancellation_waiter},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                cancellation_waiter.cancel()
                try:
                    return None, task.result()
                except asyncio.CancelledError:
                    return (
                        AnalysisTransportStatus.CANCELLED,
                        "provider_cancelled",
                    ), None
                except (ConnectionError, BrokenPipeError, EOFError):
                    return (
                        AnalysisTransportStatus.PROVIDER_LOST,
                        "provider_connection_lost",
                    ), None
                except Exception:
                    return (
                        AnalysisTransportStatus.FAILED,
                        "provider_execution_failed",
                    ), None
            if cancellation_waiter in done and _cancelled(cancellation_token):
                task.cancel()
                return (
                    AnalysisTransportStatus.CANCELLED,
                    "cancelled_during_dispatch",
                ), None
            task.cancel()
            return (
                AnalysisTransportStatus.TIMED_OUT,
                "deadline_exceeded",
            ), None
        except asyncio.CancelledError:
            task.cancel()
            cancellation_waiter.cancel()
            raise
        finally:
            if not cancellation_waiter.done():
                cancellation_waiter.cancel()

    @staticmethod
    async def _wait_for_cancellation(token: Any) -> None:
        if token is None:
            await asyncio.Future()
        waiter = getattr(token, "wait", None)
        if callable(waiter):
            value = waiter()
            if inspect.isawaitable(value):
                await value
                return
        while not _cancelled(token):
            await asyncio.sleep(0.01)

    def _normalize_provider_result(
        self,
        request: AnalysisRequest,
        raw: Any,
        capability: AnalysisCapability,
        negotiated: NegotiatedAnalysisCapability,
        *,
        elapsed_ms: int,
        progress: tuple[AnalysisProgress, ...],
        progress_truncated: bool,
    ) -> AnalysisResult:
        if isinstance(raw, AnalysisResult):
            if (
                raw.request_id != request.request_id
                or raw.operation != request.operation
                or raw.provider_id != capability.provider_id
                or raw.capability_revision != negotiated.capability_revision
                or raw.capability_id != negotiated.capability_id
                or raw.protocol_version != negotiated.protocol_version
                or raw.request_schema != negotiated.request_schema
                or raw.result_schema != negotiated.result_schema
                or raw.status is not AnalysisTransportStatus.COMPLETED
            ):
                raise AnalysisTransportError(
                    "typed provider result does not match active request"
                )
            if not raw.non_authoritative or raw.completion_authority:
                raise AnalysisTransportError("provider result claims authority")
            return self._fit_result(raw)
        if not isinstance(raw, Mapping):
            raise AnalysisTransportError("provider output must be an object")
        if len(_json_bytes(raw, name="provider output")) > self.policy.bounds.max_result_bytes:
            raise AnalysisTransportError("provider output exceeds result byte bound")
        allowed = {
            "schema",
            "protocol_version",
            "request_id",
            "operation",
            "capability_id",
            "capability_revision",
            "evidence_references",
            "provenance",
            "provenance_references",
            "cost",
            "verdict",
            "truncated",
            "non_authoritative",
            "completion_authority",
            "safe_for_completion_reasoning",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise AnalysisTransportError(
                "provider output contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        expected = {
            "schema": negotiated.result_schema,
            "protocol_version": negotiated.protocol_version,
            "request_id": request.request_id,
            "operation": request.operation,
            "capability_id": negotiated.capability_id,
            "capability_revision": negotiated.capability_revision,
        }
        for name, expected_value in expected.items():
            if raw.get(name) != expected_value:
                error_type = (
                    AnalysisCapabilityDriftError
                    if name
                    in {
                        "schema",
                        "protocol_version",
                        "capability_id",
                        "capability_revision",
                    }
                    else AnalysisTransportError
                )
                raise error_type(f"provider output {name} does not match")
        if raw.get("non_authoritative", True) is not True:
            raise AnalysisTransportError("provider output claims authority")
        if raw.get("completion_authority", False) is not False:
            raise AnalysisTransportError("provider output claims completion authority")
        if raw.get("safe_for_completion_reasoning", False) is not False:
            raise AnalysisTransportError("provider output claims completion safety")
        if not isinstance(raw.get("truncated", False), bool):
            raise AnalysisTransportError("provider output truncated must be a boolean")
        evidence, evidence_truncated = _normalize_references(
            raw.get("evidence_references", ()),
            name="evidence_references",
            max_references=self.policy.bounds.max_evidence_references,
            max_reference_bytes=self.policy.bounds.max_reference_bytes,
            truncate=True,
        )
        provenance, provenance_truncated = _normalize_references(
            raw.get(
                "provenance_references",
                raw.get("provenance", ()),
            ),
            name="provenance_references",
            max_references=self.policy.bounds.max_provenance_references,
            max_reference_bytes=self.policy.bounds.max_reference_bytes,
            truncate=True,
        )
        cost = dict(_normalize_cost(raw.get("cost", {})))
        cost["wall_time_ms"] = min(
            _MAX_COST_VALUE, cost.get("wall_time_ms", 0) + elapsed_ms
        )
        cost["provider_calls"] = min(
            _MAX_COST_VALUE, cost.get("provider_calls", 0) + 1
        )
        return self._fit_result(AnalysisResult(
            request_id=request.request_id,
            operation=request.operation,
            status=AnalysisTransportStatus.COMPLETED,
            reason_code="completed",
            provider_id=capability.provider_id,
            provider_kind=capability.provider_kind,
            capability_id=negotiated.capability_id,
            capability_revision=negotiated.capability_revision,
            protocol_version=negotiated.protocol_version,
            request_schema=negotiated.request_schema,
            result_schema=negotiated.result_schema,
            evidence_references=evidence,
            provenance_references=provenance,
            cost=cost,
            verdict=_text(
                raw.get("verdict", "inconclusive"),
                "verdict",
                max_bytes=1024,
            ),
            truncated=bool(raw.get("truncated", False))
            or evidence_truncated
            or provenance_truncated,
            progress=progress,
            progress_truncated=progress_truncated,
        ))

    def _terminal(
        self,
        request: AnalysisRequest,
        status: AnalysisTransportStatus,
        reason_code: str,
        *,
        provider: AnalysisCapability | None = None,
        negotiated: NegotiatedAnalysisCapability | None = None,
        cost: Mapping[str, int] | None = None,
        progress: tuple[AnalysisProgress, ...] = (),
        progress_truncated: bool = False,
    ) -> AnalysisResult:
        return self._fit_result(AnalysisResult(
            request_id=request.request_id,
            operation=request.operation,
            status=status,
            reason_code=reason_code,
            provider_id=provider.provider_id if provider else "",
            provider_kind=(
                provider.provider_kind if provider else AnalysisProviderKind.LOCAL
            ),
            capability_id=negotiated.capability_id if negotiated else "",
            capability_revision=(
                negotiated.capability_revision if negotiated else ""
            ),
            protocol_version=(
                negotiated.protocol_version
                if negotiated
                else ANALYSIS_TRANSPORT_PROTOCOL_VERSION
            ),
            request_schema=(
                negotiated.request_schema
                if negotiated
                else ANALYSIS_TRANSPORT_REQUEST_SCHEMA
            ),
            result_schema=(
                negotiated.result_schema
                if negotiated
                else ANALYSIS_TRANSPORT_RESULT_SCHEMA
            ),
            cost=cost or {},
            verdict="inconclusive",
            progress=progress,
            progress_truncated=progress_truncated,
        ))

    def _fit_result(self, result: AnalysisResult) -> AnalysisResult:
        """Project optional collections until the configured result bound fits."""

        maximum = self.policy.bounds.max_result_bytes
        current = result
        while len(_json_bytes(current.to_dict(), name="result")) > maximum:
            if current.progress:
                current = replace(
                    current,
                    progress=current.progress[:-1],
                    progress_truncated=True,
                    truncated=True,
                )
                continue
            if current.provenance_references:
                current = replace(
                    current,
                    provenance_references=current.provenance_references[:-1],
                    truncated=True,
                )
                continue
            if current.evidence_references:
                current = replace(
                    current,
                    evidence_references=current.evidence_references[:-1],
                    truncated=True,
                )
                continue
            # Bounds below the fixed terminal envelope cannot be honored.
            # Reject such policy at construction; this is a defensive guard
            # for future schema growth.
            raise AnalysisTransportError(
                "max_result_bytes is smaller than the fixed result envelope"
            )
        return current

    def _effective_deadline(
        self,
        request: AnalysisRequest,
        *,
        timeout_ms: int | None,
        deadline: datetime | float | None,
    ) -> float:
        effective_timeout = (
            _integer(
                timeout_ms,
                "timeout_ms",
                minimum=1,
                maximum=self.policy.bounds.timeout_ms,
            )
            if timeout_ms is not None
            else min(request.timeout_ms, self.policy.bounds.timeout_ms)
        )
        candidates = [time.time() + effective_timeout / 1000.0]
        request_deadline = _deadline_timestamp(request.deadline)
        override_deadline = _deadline_timestamp(deadline)
        if request_deadline is not None:
            candidates.append(request_deadline)
        if override_deadline is not None:
            candidates.append(override_deadline)
        return min(candidates)

    async def _admit(
        self, absolute_deadline: float, cancellation_token: Any
    ) -> tuple[AnalysisTransportStatus, str] | None:
        async with self._state_lock:
            if self._active >= self.policy.bounds.max_concurrency:
                if self._queued >= self.policy.bounds.max_queue_size:
                    return (
                        AnalysisTransportStatus.BACKPRESSURE,
                        "transport_queue_full",
                    )
                self._queued += 1
                queued = True
            else:
                queued = False
        if queued:
            remaining = absolute_deadline - time.time()
            if remaining <= 0:
                async with self._state_lock:
                    self._queued -= 1
                return AnalysisTransportStatus.TIMED_OUT, "deadline_expired_in_queue"
            acquire = asyncio.create_task(self._semaphore.acquire())
            cancel = asyncio.create_task(self._wait_for_cancellation(cancellation_token))
            try:
                done, _ = await asyncio.wait(
                    {acquire, cancel},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if acquire not in done:
                    acquire.cancel()
                    if cancel in done and _cancelled(cancellation_token):
                        return (
                            AnalysisTransportStatus.CANCELLED,
                            "cancelled_in_queue",
                        )
                    return (
                        AnalysisTransportStatus.TIMED_OUT,
                        "deadline_expired_in_queue",
                    )
            except asyncio.CancelledError:
                acquire.cancel()
                raise
            finally:
                if not cancel.done():
                    cancel.cancel()
                async with self._state_lock:
                    self._queued -= 1
            async with self._state_lock:
                self._active += 1
            return None
        await self._semaphore.acquire()
        async with self._state_lock:
            self._active += 1
        return None

    async def _release(self) -> None:
        async with self._state_lock:
            self._active -= 1
        self._semaphore.release()

    @staticmethod
    def _record_provider_failure(
        registration: _ProviderRegistration, status: AnalysisTransportStatus
    ) -> None:
        registration.failures += 1
        if status in {
            AnalysisTransportStatus.PROVIDER_LOST,
            AnalysisTransportStatus.UNAVAILABLE,
        }:
            registration.runtime_health = AnalysisProviderHealth.UNAVAILABLE
        elif status in {
            AnalysisTransportStatus.CAPABILITY_DRIFT,
            AnalysisTransportStatus.UNSUPPORTED,
        }:
            registration.runtime_health = AnalysisProviderHealth.INCOMPATIBLE
        else:
            registration.runtime_health = AnalysisProviderHealth.DEGRADED


# Compact compatibility spellings for callers that use "transport" prefixes.
TransportBounds = AnalysisTransportBounds
TransportPolicy = AnalysisTransportPolicy
TransportStatus = AnalysisTransportStatus
TransportRequest = AnalysisRequest
TransportResult = AnalysisResult
ProviderCapability = AnalysisCapability
NegotiatedCapability = NegotiatedAnalysisCapability
CancellationToken = AnalysisCancellationToken


__all__ = [
    "ANALYSIS_TRANSPORT_CAPABILITY_SCHEMA",
    "ANALYSIS_TRANSPORT_PROGRESS_SCHEMA",
    "ANALYSIS_TRANSPORT_PROTOCOL_VERSION",
    "ANALYSIS_TRANSPORT_REQUEST_SCHEMA",
    "ANALYSIS_TRANSPORT_RESULT_SCHEMA",
    "ANALYSIS_TRANSPORT_VERSION",
    "AnalysisCancellationToken",
    "AnalysisCapability",
    "AnalysisCapabilityDriftError",
    "AnalysisProgress",
    "AnalysisProviderHealth",
    "AnalysisProviderKind",
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisTransport",
    "AnalysisTransportBounds",
    "AnalysisTransportError",
    "AnalysisTransportHealth",
    "AnalysisTransportPolicy",
    "AnalysisTransportStatus",
    "CancellationToken",
    "NegotiatedAnalysisCapability",
    "NegotiatedCapability",
    "ProviderCapability",
    "TransportBounds",
    "TransportPolicy",
    "TransportRequest",
    "TransportResult",
    "TransportStatus",
]
