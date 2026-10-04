"""Version 1 contracts for the side-effect-free AI service catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar
from urllib.parse import urlsplit, urlunsplit

from .identity import (
    canonical_json,
    content_cid,
    deployment_identity,
    is_secret_key,
    is_secret_value,
    model_identity,
    provider_identity,
    router_binding_identity,
)

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset((SCHEMA_VERSION,))

MAX_NAME_LENGTH = 128
MAX_DISPLAY_NAME_LENGTH = 256
MAX_DESCRIPTION_LENGTH = 4096
MAX_ALIASES = 64
MAX_CAPABILITIES = 32
MAX_PROVENANCE = 64
MAX_LABELS = 64
MAX_SNAPSHOT_RECORDS = 10_000
MAX_SIZE_BYTES = 1 << 40
MAX_TOKEN_COUNT = 100_000_000
MAX_BATCH_SIZE = 1_000_000

_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9._/-]{0,126}[a-z0-9])?$")
_ROUTER = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MIME = re.compile(r"^[a-z0-9!#$&^_.+-]+/(?:[a-z0-9!#$&^_.+-]+|\*)$")
_LABEL_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")
_T = TypeVar("_T")


class SchemaValidationError(ValueError):
    """A catalog record violates the v1 contract."""


class Operation(str, Enum):
    TEXT_GENERATE = "text.generate"
    TEXT_CHAT = "text.chat"
    EMBEDDING_GENERATE = "embedding.generate"
    VISION_GENERATE = "vision.generate"
    AUDIO_TRANSCRIBE = "audio.transcribe"
    AUDIO_SYNTHESIZE = "audio.synthesize"
    BATCH = "batch"
    STREAM = "stream"
    TOOL_CALL = "tool.call"


# Explicit name used by some integrations.
OperationName = Operation


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"
    JSON = "json"


class LifecycleState(str, Enum):
    UNKNOWN = "unknown"
    DECLARED = "declared"
    CONFIGURED = "configured"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _version(value: Any) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_SCHEMA_VERSIONS:
        _fail("unsupported schema_version: %r" % (value,))
    return value


def _text(value: Any, field_name: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail("%s must be a string" % field_name)
    value = value.strip()
    if not value and not empty:
        _fail("%s must not be empty" % field_name)
    if len(value.encode("utf-8")) > maximum:
        _fail("%s exceeds %d UTF-8 bytes" % (field_name, maximum))
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail("%s contains control characters" % field_name)
    if is_secret_value(value):
        _fail("%s contains credential-shaped data" % field_name)
    return value


def _name(value: Any, field_name: str = "name") -> str:
    if isinstance(value, str) and value != value.strip():
        _fail("%s must not contain surrounding whitespace" % field_name)
    value = _text(value, field_name, MAX_NAME_LENGTH).casefold()
    if not _NAME.fullmatch(value) or "//" in value or ".." in value:
        _fail("%s is not a canonical catalog name" % field_name)
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, 128)
    if not re.fullmatch(r"[a-z]+_[0-9a-f]{64}", value):
        _fail("%s is not a stable catalog identifier" % field_name)
    return value


def _enum(value: Any, enum_type: Type[_T], field_name: str) -> _T:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        _fail("%s must be a string" % field_name)
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError:
        _fail("unknown %s: %r" % (field_name, value))
    raise AssertionError("unreachable")


def _tuple_enum(
    values: Any, enum_type: Type[_T], field_name: str, *, required: bool = False
) -> Tuple[_T, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("%s must be an array" % field_name)
    parsed = tuple(_enum(item, enum_type, field_name) for item in values)
    result = tuple(sorted(set(parsed), key=lambda item: str(getattr(item, "value", item))))
    if required and not result:
        _fail("%s must not be empty" % field_name)
    return result


def _aliases(values: Any, canonical_name: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("aliases must be an array")
    if len(values) > MAX_ALIASES:
        _fail("aliases exceeds maximum count")
    aliases = tuple(sorted({_name(item, "alias") for item in values}))
    if canonical_name in aliases:
        _fail("aliases must not repeat the canonical name")
    return aliases


def _optional_positive(value: Any, field_name: str, maximum: int) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("%s must be an integer or null" % field_name)
    if value <= 0 or value > maximum:
        _fail("%s must be between 1 and %d" % (field_name, maximum))
    return value


def _timestamp(value: Any, field_name: str, *, optional: bool = True) -> Optional[str]:
    if value is None and optional:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            _fail("%s must not be empty" % field_name)
        if not _RFC3339.fullmatch(raw):
            _fail("%s must be an RFC 3339 timestamp" % field_name)
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
        except ValueError:
            _fail("%s must be an RFC 3339 timestamp" % field_name)
    else:
        _fail("%s must be an RFC 3339 string" % field_name)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("%s must include a timezone" % field_name)
    parsed = parsed.astimezone(timezone.utc)
    # A fixed precision makes normalized timestamps lexicographically ordered,
    # which is useful for deterministic expiry and update comparisons.
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _uri(value: Any, field_name: str, *, endpoint: bool = False) -> Optional[str]:
    if value is None:
        return None
    value = _text(value, field_name, 2048)
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        _fail("%s is malformed" % field_name)
    allowed = ("http", "https", "unix") if endpoint else ("http", "https")
    if parts.scheme.casefold() not in allowed:
        _fail("%s has an unsupported URI scheme" % field_name)
    if parts.username is not None or parts.password is not None:
        _fail("%s must not contain user information" % field_name)
    if parts.query or parts.fragment:
        _fail("%s must not contain query credentials or fragments" % field_name)
    if parts.scheme != "unix" and not parts.hostname:
        _fail("%s must include a host" % field_name)
    scheme = parts.scheme.casefold()
    if scheme == "unix":
        if not parts.path.startswith("/"):
            _fail("%s unix URI must use an absolute path" % field_name)
        return urlunsplit((scheme, "", parts.path, "", ""))
    host = parts.hostname.casefold()  # type: ignore[union-attr]
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = "%s:%d" % (host, port)
    path = parts.path or "/"
    return urlunsplit((scheme, host, path, "", ""))


def _strict_mapping(
    data: Any,
    allowed: Iterable[str],
    required: Iterable[str],
    name: str,
) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        _fail("%s must be an object" % name)
    keys = set(data)
    non_strings = [key for key in keys if not isinstance(key, str)]
    if non_strings:
        _fail("%s keys must be strings" % name)
    unknown = keys - set(allowed)
    if unknown:
        _fail("%s contains unknown fields: %s" % (name, ", ".join(sorted(unknown))))
    missing = set(required) - keys
    if missing:
        _fail("%s is missing required fields: %s" % (name, ", ".join(sorted(missing))))
    return dict(data)


def _labels(value: Any) -> Tuple[Tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        _fail("labels must be an object")
    result = []
    for pair in values:
        if not isinstance(pair, Sequence) or len(pair) != 2:
            _fail("each label must be a key/value pair")
        key = _text(pair[0], "label key", 64).casefold()
        if not _LABEL_KEY.fullmatch(key):
            _fail("invalid label key: %s" % key)
        if is_secret_key(key):
            _fail("credential-bearing label keys are forbidden")
        result.append((key, _text(pair[1], "label value", 256, empty=True)))
    if len(result) > MAX_LABELS or len({key for key, _ in result}) != len(result):
        _fail("labels are duplicated or exceed the maximum count")
    return tuple(sorted(result))


def _record_tuple(values: Any, cls: Type[_T], field_name: str) -> Tuple[_T, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("%s must be an array" % field_name)
    if len(values) > MAX_SNAPSHOT_RECORDS:
        _fail("%s exceeds maximum record count" % field_name)
    parsed = tuple(
        item if isinstance(item, cls) else cls.from_dict(item)  # type: ignore[attr-defined]
        for item in values
    )
    id_field = {
        "ProviderDescriptor": "provider_id",
        "ModelDescriptor": "model_id",
        "DeploymentDescriptor": "deployment_id",
        "RouterBinding": "binding_id",
    }[cls.__name__]
    ids = [getattr(item, id_field) for item in parsed]
    if len(ids) != len(set(ids)):
        _fail("%s contains duplicate identities" % field_name)
    return tuple(sorted(parsed, key=lambda item: getattr(item, id_field)))


@dataclass(frozen=True)
class OperationalState:
    """Independent observed facts; ``None`` means the fact is unknown.

    No field is derived from another.  For example, a known remote provider may
    be neither configured nor reachable, and a reachable endpoint may have
    unknown authorization and health.
    """

    known: Optional[bool] = None
    configured: Optional[bool] = None
    authorized: Optional[bool] = None
    reachable: Optional[bool] = None
    healthy: Optional[bool] = None
    routable: Optional[bool] = None

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if value is not None and not isinstance(value, bool):
                _fail("%s must be a boolean or null" % name)

    def to_dict(self) -> Dict[str, Optional[bool]]:
        return {
            "known": self.known,
            "configured": self.configured,
            "authorized": self.authorized,
            "reachable": self.reachable,
            "healthy": self.healthy,
            "routable": self.routable,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "OperationalState":
        fields = ("known", "configured", "authorized", "reachable", "healthy", "routable")
        values = _strict_mapping(data, fields, (), "OperationalState")
        return cls(**{name: values.get(name) for name in fields})


# Compatibility-oriented descriptive aliases, not alternate state models.
AvailabilityState = OperationalState
ServiceState = OperationalState


@dataclass(frozen=True)
class CapabilityDescriptor:
    schema_version: str = SCHEMA_VERSION
    operations: Tuple[Operation, ...] = ()
    input_modalities: Tuple[Modality, ...] = ()
    output_modalities: Tuple[Modality, ...] = ()
    media_types: Tuple[str, ...] = ()
    max_context_tokens: Optional[int] = None
    max_batch_size: Optional[int] = None
    max_input_bytes: Optional[int] = None
    max_output_bytes: Optional[int] = None
    embedding_dimensions: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        operations = _tuple_enum(self.operations, Operation, "operations", required=True)
        if not set(operations) - {Operation.BATCH, Operation.STREAM}:
            _fail("batch and stream must qualify at least one invokable operation")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "input_modalities",
            _tuple_enum(self.input_modalities, Modality, "input_modalities"),
        )
        object.__setattr__(
            self,
            "output_modalities",
            _tuple_enum(self.output_modalities, Modality, "output_modalities"),
        )
        if isinstance(self.media_types, (str, bytes, Mapping)) or not isinstance(
            self.media_types, (Sequence, set, frozenset)
        ):
            _fail("media_types must be an array")
        media_types = tuple(
            sorted({_text(item, "media type", 128).casefold() for item in self.media_types})
        )
        if len(media_types) > 64 or any(not _MIME.fullmatch(item) for item in media_types):
            _fail("media_types contains an invalid or excessive MIME type")
        object.__setattr__(self, "media_types", media_types)
        for name, maximum in (
            ("max_context_tokens", MAX_TOKEN_COUNT),
            ("max_batch_size", MAX_BATCH_SIZE),
            ("max_input_bytes", MAX_SIZE_BYTES),
            ("max_output_bytes", MAX_SIZE_BYTES),
            ("embedding_dimensions", 10_000_000),
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive(getattr(self, name), name, maximum),
            )
        if self.embedding_dimensions is not None and Operation.EMBEDDING_GENERATE not in operations:
            _fail("embedding_dimensions requires embedding.generate")
        if self.max_batch_size is not None and Operation.BATCH not in operations:
            _fail("max_batch_size requires the batch operation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operations": [item.value for item in self.operations],
            "input_modalities": [item.value for item in self.input_modalities],
            "output_modalities": [item.value for item in self.output_modalities],
            "media_types": list(self.media_types),
            "max_context_tokens": self.max_context_tokens,
            "max_batch_size": self.max_batch_size,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "embedding_dimensions": self.embedding_dimensions,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CapabilityDescriptor":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(
            data,
            fields,
            ("schema_version", "operations"),
            "CapabilityDescriptor",
        )
        return cls(**values)


@dataclass(frozen=True)
class Provenance:
    schema_version: str = SCHEMA_VERSION
    source: str = ""
    source_record_id: Optional[str] = None
    observed_at: Optional[str] = None
    expires_at: Optional[str] = None
    issuer: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(self, "source", _name(self.source, "source"))
        for name in ("source_record_id", "issuer"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name, 512))
        observed = _timestamp(self.observed_at, "observed_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if observed and expires and expires <= observed:
            _fail("expires_at must be later than observed_at")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_record_id": self.source_record_id,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "issuer": self.issuer,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Provenance":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**_strict_mapping(data, fields, ("schema_version", "source"), "Provenance"))


ProvenanceDescriptor = Provenance


def _capabilities(values: Any) -> Tuple[CapabilityDescriptor, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("capabilities must be an array")
    if len(values) > MAX_CAPABILITIES:
        _fail("capabilities exceeds maximum count")
    parsed = tuple(
        item if isinstance(item, CapabilityDescriptor) else CapabilityDescriptor.from_dict(item)
        for item in values
    )
    keys = [canonical_json(item) for item in parsed]
    if len(keys) != len(set(keys)):
        _fail("capabilities contains duplicates")
    return tuple(item for _, item in sorted(zip(keys, parsed), key=lambda pair: pair[0]))


def _provenance(values: Any) -> Tuple[Provenance, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(
        values, (Sequence, set, frozenset)
    ):
        _fail("provenance must be an array")
    if len(values) > MAX_PROVENANCE:
        _fail("provenance exceeds maximum count")
    parsed = tuple(
        item if isinstance(item, Provenance) else Provenance.from_dict(item) for item in values
    )
    keys = [canonical_json(item) for item in parsed]
    if len(keys) != len(set(keys)):
        _fail("provenance contains duplicates")
    return tuple(item for _, item in sorted(zip(keys, parsed), key=lambda pair: pair[0]))


def _state(value: Any) -> OperationalState:
    return value if isinstance(value, OperationalState) else OperationalState.from_dict(value)


@dataclass(frozen=True)
class ProviderDescriptor:
    name: str
    provider_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    display_name: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    description: str = ""
    website_uri: Optional[str] = None
    documentation_uri: Optional[str] = None
    capabilities: Tuple[CapabilityDescriptor, ...] = ()
    lifecycle: LifecycleState = LifecycleState.UNKNOWN
    state: OperationalState = field(default_factory=OperationalState)
    provenance: Tuple[Provenance, ...] = ()
    labels: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        name = _name(self.name)
        object.__setattr__(self, "name", name)
        expected = provider_identity(name)
        if (
            self.provider_id is not None
            and _identifier(self.provider_id, "provider_id") != expected
        ):
            _fail("provider_id does not match canonical identity fields")
        object.__setattr__(self, "provider_id", expected)
        if self.display_name is not None:
            object.__setattr__(
                self,
                "display_name",
                _text(
                    self.display_name,
                    "display_name",
                    MAX_DISPLAY_NAME_LENGTH,
                ),
            )
        object.__setattr__(self, "aliases", _aliases(self.aliases, name))
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                "description",
                MAX_DESCRIPTION_LENGTH,
                empty=True,
            ),
        )
        object.__setattr__(self, "website_uri", _uri(self.website_uri, "website_uri"))
        object.__setattr__(
            self,
            "documentation_uri",
            _uri(self.documentation_uri, "documentation_uri"),
        )
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))
        object.__setattr__(
            self,
            "lifecycle",
            _enum(self.lifecycle, LifecycleState, "lifecycle"),
        )
        object.__setattr__(self, "state", _state(self.state))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "labels", _labels(self.labels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "name": self.name,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "description": self.description,
            "website_uri": self.website_uri,
            "documentation_uri": self.documentation_uri,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "lifecycle": self.lifecycle.value,
            "state": self.state.to_dict(),
            "provenance": [item.to_dict() for item in self.provenance],
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProviderDescriptor":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(data, fields, ("schema_version", "name"), "ProviderDescriptor")
        return cls(**values)

    @property
    def cid(self) -> str:
        return content_cid(self)


@dataclass(frozen=True)
class ModelDescriptor:
    provider_id: str
    name: str
    model_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    display_name: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    description: str = ""
    architecture: Optional[str] = None
    capabilities: Tuple[CapabilityDescriptor, ...] = ()
    lifecycle: LifecycleState = LifecycleState.UNKNOWN
    state: OperationalState = field(default_factory=OperationalState)
    provenance: Tuple[Provenance, ...] = ()
    labels: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        provider_id = _identifier(self.provider_id, "provider_id")
        name = _name(self.name)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "name", name)
        expected = model_identity(provider_id, name)
        if self.model_id is not None and _identifier(self.model_id, "model_id") != expected:
            _fail("model_id does not match canonical identity fields")
        object.__setattr__(self, "model_id", expected)
        if self.display_name is not None:
            object.__setattr__(
                self,
                "display_name",
                _text(
                    self.display_name,
                    "display_name",
                    MAX_DISPLAY_NAME_LENGTH,
                ),
            )
        object.__setattr__(self, "aliases", _aliases(self.aliases, name))
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                "description",
                MAX_DESCRIPTION_LENGTH,
                empty=True,
            ),
        )
        if self.architecture is not None:
            object.__setattr__(
                self,
                "architecture",
                _text(self.architecture, "architecture", 256),
            )
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))
        object.__setattr__(self, "lifecycle", _enum(self.lifecycle, LifecycleState, "lifecycle"))
        object.__setattr__(self, "state", _state(self.state))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "labels", _labels(self.labels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "name": self.name,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "description": self.description,
            "architecture": self.architecture,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "lifecycle": self.lifecycle.value,
            "state": self.state.to_dict(),
            "provenance": [item.to_dict() for item in self.provenance],
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ModelDescriptor":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(
            data,
            fields,
            ("schema_version", "provider_id", "name"),
            "ModelDescriptor",
        )
        return cls(**values)

    @property
    def cid(self) -> str:
        return content_cid(self)


@dataclass(frozen=True)
class DeploymentDescriptor:
    provider_id: str
    name: str
    endpoint_uri: str
    model_id: Optional[str] = None
    deployment_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    capabilities: Tuple[CapabilityDescriptor, ...] = ()
    lifecycle: LifecycleState = LifecycleState.UNKNOWN
    state: OperationalState = field(default_factory=OperationalState)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    provenance: Tuple[Provenance, ...] = ()
    labels: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        provider_id = _identifier(self.provider_id, "provider_id")
        model_id = "" if self.model_id is None else _identifier(self.model_id, "model_id")
        name = _name(self.name)
        endpoint = _uri(self.endpoint_uri, "endpoint_uri", endpoint=True)
        if endpoint is None:
            _fail("endpoint_uri must not be null")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id or None)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "endpoint_uri", endpoint)
        expected = deployment_identity(provider_id, model_id, name, endpoint)
        if (
            self.deployment_id is not None
            and _identifier(self.deployment_id, "deployment_id") != expected
        ):
            _fail("deployment_id does not match canonical identity fields")
        object.__setattr__(self, "deployment_id", expected)
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))
        object.__setattr__(
            self,
            "lifecycle",
            _enum(self.lifecycle, LifecycleState, "lifecycle"),
        )
        object.__setattr__(self, "state", _state(self.state))
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if created and updated and updated < created:
            _fail("updated_at must not precede created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "labels", _labels(self.labels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "name": self.name,
            "endpoint_uri": self.endpoint_uri,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "lifecycle": self.lifecycle.value,
            "state": self.state.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": [item.to_dict() for item in self.provenance],
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DeploymentDescriptor":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(
            data,
            fields,
            ("schema_version", "provider_id", "name", "endpoint_uri"),
            "DeploymentDescriptor",
        )
        return cls(**values)

    @property
    def cid(self) -> str:
        return content_cid(self)


@dataclass(frozen=True)
class RouterBinding:
    router: str
    provider_id: str
    operations: Tuple[Operation, ...]
    model_id: Optional[str] = None
    deployment_id: Optional[str] = None
    binding_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    priority: int = 0
    state: OperationalState = field(default_factory=OperationalState)
    provenance: Tuple[Provenance, ...] = ()
    labels: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        router = _text(self.router, "router", 64).casefold()
        if not _ROUTER.fullmatch(router):
            _fail("router is not a canonical name")
        provider_id = _identifier(self.provider_id, "provider_id")
        model_id = "" if self.model_id is None else _identifier(self.model_id, "model_id")
        deployment_id = (
            "" if self.deployment_id is None else _identifier(self.deployment_id, "deployment_id")
        )
        if not model_id and not deployment_id:
            _fail("a router binding requires model_id or deployment_id")
        operations = _tuple_enum(self.operations, Operation, "operations", required=True)
        if not set(operations) - {Operation.BATCH, Operation.STREAM}:
            _fail("a router binding requires an invokable operation")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or not -1_000_000 <= self.priority <= 1_000_000
        ):
            _fail("priority must be an integer between -1000000 and 1000000")
        object.__setattr__(self, "router", router)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "model_id", model_id or None)
        object.__setattr__(self, "deployment_id", deployment_id or None)
        object.__setattr__(self, "operations", operations)
        expected = router_binding_identity(router, provider_id, model_id, deployment_id)
        if self.binding_id is not None and _identifier(self.binding_id, "binding_id") != expected:
            _fail("binding_id does not match canonical identity fields")
        object.__setattr__(self, "binding_id", expected)
        object.__setattr__(self, "state", _state(self.state))
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        object.__setattr__(self, "labels", _labels(self.labels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "binding_id": self.binding_id,
            "router": self.router,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "operations": [item.value for item in self.operations],
            "priority": self.priority,
            "state": self.state.to_dict(),
            "provenance": [item.to_dict() for item in self.provenance],
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "RouterBinding":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(
            data,
            fields,
            ("schema_version", "router", "provider_id", "operations"),
            "RouterBinding",
        )
        return cls(**values)

    @property
    def cid(self) -> str:
        return content_cid(self)


@dataclass(frozen=True)
class CatalogSnapshot:
    providers: Tuple[ProviderDescriptor, ...] = ()
    models: Tuple[ModelDescriptor, ...] = ()
    deployments: Tuple[DeploymentDescriptor, ...] = ()
    bindings: Tuple[RouterBinding, ...] = ()
    schema_version: str = SCHEMA_VERSION
    created_at: Optional[str] = None
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _version(self.schema_version))
        object.__setattr__(
            self,
            "providers",
            _record_tuple(self.providers, ProviderDescriptor, "providers"),
        )
        object.__setattr__(
            self,
            "models",
            _record_tuple(self.models, ModelDescriptor, "models"),
        )
        object.__setattr__(
            self,
            "deployments",
            _record_tuple(
                self.deployments,
                DeploymentDescriptor,
                "deployments",
            ),
        )
        object.__setattr__(
            self,
            "bindings",
            _record_tuple(self.bindings, RouterBinding, "bindings"),
        )
        object.__setattr__(
            self,
            "created_at",
            _timestamp(self.created_at, "created_at"),
        )
        expected = content_cid(self.content_dict())
        if self.revision is not None and self.revision != expected:
            _fail("revision does not match canonical snapshot content")
        object.__setattr__(self, "revision", expected)

    def content_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "providers": [item.to_dict() for item in self.providers],
            "models": [item.to_dict() for item in self.models],
            "deployments": [item.to_dict() for item in self.deployments],
            "bindings": [item.to_dict() for item in self.bindings],
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self.content_dict()
        result.update({"created_at": self.created_at, "revision": self.revision})
        return result

    @classmethod
    def from_dict(cls, data: Any) -> "CatalogSnapshot":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = _strict_mapping(data, fields, ("schema_version",), "CatalogSnapshot")
        return cls(**values)

    @property
    def cid(self) -> str:
        return self.revision  # type: ignore[return-value]


__all__ = [
    "AvailabilityState",
    "CapabilityDescriptor",
    "CatalogSnapshot",
    "DeploymentDescriptor",
    "LifecycleState",
    "MAX_ALIASES",
    "MAX_BATCH_SIZE",
    "MAX_CAPABILITIES",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_SIZE_BYTES",
    "MAX_SNAPSHOT_RECORDS",
    "MAX_TOKEN_COUNT",
    "Modality",
    "ModelDescriptor",
    "Operation",
    "OperationName",
    "OperationalState",
    "ProviderDescriptor",
    "Provenance",
    "ProvenanceDescriptor",
    "RouterBinding",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaValidationError",
    "ServiceState",
]
