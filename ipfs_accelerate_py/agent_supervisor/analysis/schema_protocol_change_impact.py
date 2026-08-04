"""Schema, constructor, serialization, and protocol change-impact analysis.

RPR-030: given a typed :class:`ProgramContractDelta` and schema/protocol
consumer observations, detect field/variant, constructor/factory/builder,
serialization, and public-protocol impacts and emit one explicit compatibility
disposition plus canonical
:class:`~ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts.ConsumerMigrationObligation`
per consumer.

Rules enforced here:

* Added, removed, renamed, and retyped fields and variants are first-class
  change kinds with per-consumer compatibility directions
  (backward / forward / full / incompatible / unknown).
* Constructor, factory, and builder consumers are inventoried independently of
  serializers, persistence, cache keys, equality/hash, version negotiation,
  migrations, and generated clients.
* JSON, protobuf, IDL, database, message, RPC, HTTP, and CLI schema surfaces
  are closed protocol kinds.
* Required defaults and migrations never claim authority from the changed
  implementation alone; they need independent reviewed evidence.
* Generated and read-only roots produce regeneration or external obligations
  rather than direct in-tree writes.
* Missing or dynamic codecs remain open frontiers and cannot discharge others.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .change_propagation_contracts import (
    CHANGE_PROPAGATION_VERSION,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    MAX_CONSUMER_COUNT,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    obligation_set_identity,
)

# ---------------------------------------------------------------------------
# Schemas / bounds
# ---------------------------------------------------------------------------

SCHEMA_PROTOCOL_CHANGE_IMPACT_VERSION: Final[str] = "schema-protocol-change-impact@1"
SCHEMA_PROTOCOL_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/schema-protocol-impact@1"
)
SCHEMA_FIELD_CHANGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/schema-field-change@1"
)
CONSTRUCTOR_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/constructor-impact@1"
)
SERIALIZATION_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/serialization-impact@1"
)
PROTOCOL_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-impact@1"
)
SCHEMA_CONSUMER_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/schema-consumer-observation@1"
)
SCHEMA_CONSUMER_IMPACT_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/schema-consumer-impact-entry@1"
)

PRODUCER_ID: Final[str] = "schema-protocol-change-impact@1"

MAX_FIELD_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_ENTRIES: Final[int] = MAX_CONSUMER_COUNT
MAX_FIELD_CHANGES: Final[int] = 512
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_REASON_CODES: Final[int] = 128
MAX_CLAUSE_IDS: Final[int] = 256

_GENERATED_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/generated/",
    "/_generated/",
    "/gen/",
    ".generated.",
    "_pb2.py",
    "_pb2_grpc.py",
    ".g.dart",
    ".pb.go",
    ".pb.cc",
    "_generated.",
    "/build/",
    "/dist/",
)
_READ_ONLY_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/vendor/",
    "/third_party/",
    "/external/",
    "/.git/",
    "/node_modules/",
    "/site-packages/",
)

_SCHEMA_DELTA_KINDS: Final[frozenset[DeltaKind]] = frozenset(
    {
        DeltaKind.SCHEMA_CHANGE,
        DeltaKind.SERIALIZATION_CHANGE,
        DeltaKind.PROTOCOL_CHANGE,
        DeltaKind.FIELD_INTRO,
        DeltaKind.FIELD_REMOVE,
        DeltaKind.CONSTRUCTOR_INTRO,
        DeltaKind.CONSTRUCTOR_REMOVE,
        DeltaKind.FACTORY_INTRO,
        DeltaKind.FACTORY_REMOVE,
        DeltaKind.DATA_STRUCTURE_INTRO,
        DeltaKind.DATA_STRUCTURE_REMOVE,
        DeltaKind.NULLABILITY_CHANGE,
        DeltaKind.PARAMETER_DEFAULT,
    }
)

_FIELD_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:field|member|property|column|key|variant|case)\s*[=:]\s*"
    r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?",
    re.IGNORECASE,
)
_RENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:rename[sd]?|renamed)\s+"
    r"(?:(?:field|member|property|column|key|variant|case)\s+)?"
    r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?\s*(?:->|to|as)\s*"
    r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?",
    re.IGNORECASE,
)
_RETYPE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:retype[sd]?|type\s+change[sd]?)\s+"
    r"(?:(?:field|member|property|column|key)\s+)?"
    r"[`'\"]?([A-Za-z_][\w.]*)[`'\"]?\s*(?::|from)?\s*"
    r"[`'\"]?([A-Za-z_][\w.\[\]| ]*)[`'\"]?\s*(?:->|to)\s*"
    r"[`'\"]?([A-Za-z_][\w.\[\]| ]*)[`'\"]?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchemaProtocolChangeImpactError(ValueError):
    """Malformed schema/protocol impact input or invariant failure."""


class SchemaProtocolChangeImpactBoundsError(SchemaProtocolChangeImpactError):
    """A record exceeded its hard compactness bound."""


class SchemaProtocolChangeImpactAuthorityError(SchemaProtocolChangeImpactError):
    """Roots, authority evidence, or write-mode policy was violated."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class FieldChangeKind(str, Enum):
    """Closed field / variant mutation kinds."""

    ADDED = "added"
    REMOVED = "removed"
    RENAMED = "renamed"
    RETYPED = "retyped"
    VARIANT_ADDED = "variant_added"
    VARIANT_REMOVED = "variant_removed"
    VARIANT_RENAMED = "variant_renamed"

    @classmethod
    def coerce(cls, value: Any) -> "FieldChangeKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, FieldChangeKind] = {
            "added": cls.ADDED,
            "add": cls.ADDED,
            "intro": cls.ADDED,
            "field_add": cls.ADDED,
            "field_intro": cls.ADDED,
            "removed": cls.REMOVED,
            "remove": cls.REMOVED,
            "delete": cls.REMOVED,
            "field_remove": cls.REMOVED,
            "renamed": cls.RENAMED,
            "rename": cls.RENAMED,
            "field_rename": cls.RENAMED,
            "retyped": cls.RETYPED,
            "retype": cls.RETYPED,
            "type_change": cls.RETYPED,
            "field_retype": cls.RETYPED,
            "variant_added": cls.VARIANT_ADDED,
            "variant_add": cls.VARIANT_ADDED,
            "case_added": cls.VARIANT_ADDED,
            "variant_removed": cls.VARIANT_REMOVED,
            "variant_remove": cls.VARIANT_REMOVED,
            "case_removed": cls.VARIANT_REMOVED,
            "variant_renamed": cls.VARIANT_RENAMED,
            "variant_rename": cls.VARIANT_RENAMED,
            "case_renamed": cls.VARIANT_RENAMED,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported field change kind: {value!r}"
            ) from exc


class SchemaSurfaceKind(str, Enum):
    """Closed public schema / protocol surface families."""

    JSON = "json"
    PROTOBUF = "protobuf"
    IDL = "idl"
    DATABASE = "database"
    MESSAGE = "message"
    RPC = "rpc"
    HTTP = "http"
    CLI = "cli"

    @classmethod
    def coerce(cls, value: Any) -> "SchemaSurfaceKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, SchemaSurfaceKind] = {
            "json": cls.JSON,
            "json_schema": cls.JSON,
            "openapi": cls.HTTP,
            "protobuf": cls.PROTOBUF,
            "proto": cls.PROTOBUF,
            "pb": cls.PROTOBUF,
            "grpc": cls.RPC,
            "idl": cls.IDL,
            "avro": cls.IDL,
            "thrift": cls.IDL,
            "capnp": cls.IDL,
            "database": cls.DATABASE,
            "db": cls.DATABASE,
            "sql": cls.DATABASE,
            "table": cls.DATABASE,
            "message": cls.MESSAGE,
            "queue": cls.MESSAGE,
            "event": cls.MESSAGE,
            "rpc": cls.RPC,
            "http": cls.HTTP,
            "rest": cls.HTTP,
            "cli": cls.CLI,
            "command_line": cls.CLI,
            "argparse": cls.CLI,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported schema surface kind: {value!r}"
            ) from exc


class ConstructionKind(str, Enum):
    """Closed construction / factory surface kinds."""

    CONSTRUCTOR = "constructor"
    FACTORY = "factory"
    BUILDER = "builder"

    @classmethod
    def coerce(cls, value: Any) -> "ConstructionKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, ConstructionKind] = {
            "constructor": cls.CONSTRUCTOR,
            "ctor": cls.CONSTRUCTOR,
            "__init__": cls.CONSTRUCTOR,
            "factory": cls.FACTORY,
            "create": cls.FACTORY,
            "builder": cls.BUILDER,
            "build": cls.BUILDER,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported construction kind: {value!r}"
            ) from exc


class SerializationFacet(str, Enum):
    """Closed serialization / persistence / identity facets."""

    SERIALIZER = "serializer"
    DESERIALIZER = "deserializer"
    PERSISTENCE = "persistence"
    CACHE_KEY = "cache_key"
    EQUALITY = "equality"
    HASH = "hash"
    VERSION_NEGOTIATION = "version_negotiation"
    MIGRATION = "migration"
    GENERATED_CLIENT = "generated_client"

    @classmethod
    def coerce(cls, value: Any) -> "SerializationFacet":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, SerializationFacet] = {
            "serializer": cls.SERIALIZER,
            "serialize": cls.SERIALIZER,
            "encode": cls.SERIALIZER,
            "to_json": cls.SERIALIZER,
            "deserializer": cls.DESERIALIZER,
            "deserialize": cls.DESERIALIZER,
            "decode": cls.DESERIALIZER,
            "from_json": cls.DESERIALIZER,
            "persistence": cls.PERSISTENCE,
            "storage": cls.PERSISTENCE,
            "store": cls.PERSISTENCE,
            "cache_key": cls.CACHE_KEY,
            "cache": cls.CACHE_KEY,
            "equality": cls.EQUALITY,
            "eq": cls.EQUALITY,
            "__eq__": cls.EQUALITY,
            "hash": cls.HASH,
            "__hash__": cls.HASH,
            "version_negotiation": cls.VERSION_NEGOTIATION,
            "versioning": cls.VERSION_NEGOTIATION,
            "compat_version": cls.VERSION_NEGOTIATION,
            "migration": cls.MIGRATION,
            "migrate": cls.MIGRATION,
            "generated_client": cls.GENERATED_CLIENT,
            "generated": cls.GENERATED_CLIENT,
            "stub": cls.GENERATED_CLIENT,
            "binding": cls.GENERATED_CLIENT,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported serialization facet: {value!r}"
            ) from exc


class CompatibilityDirection(str, Enum):
    """Per-consumer schema evolution compatibility direction."""

    BACKWARD = "backward"
    """New consumers accept old payloads (old writers remain valid)."""

    FORWARD = "forward"
    """Old consumers accept new payloads (new writers remain valid for old readers)."""

    FULL = "full"
    """Both backward and forward for this consumer role."""

    INCOMPATIBLE = "incompatible"
    """Neither direction holds for this consumer role."""

    UNKNOWN = "unknown"
    """Insufficient evidence to classify compatibility."""

    @classmethod
    def coerce(cls, value: Any) -> "CompatibilityDirection":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, CompatibilityDirection] = {
            "backward": cls.BACKWARD,
            "backwards": cls.BACKWARD,
            "backward_compatible": cls.BACKWARD,
            "forward": cls.FORWARD,
            "forwards": cls.FORWARD,
            "forward_compatible": cls.FORWARD,
            "full": cls.FULL,
            "both": cls.FULL,
            "full_compatible": cls.FULL,
            "incompatible": cls.INCOMPATIBLE,
            "breaking": cls.INCOMPATIBLE,
            "none": cls.INCOMPATIBLE,
            "unknown": cls.UNKNOWN,
            "undetermined": cls.UNKNOWN,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported compatibility direction: {value!r}"
            ) from exc


class AuthorityKind(str, Enum):
    """Independent authority required before a claim may discharge."""

    NONE = "none"
    REVIEWED_IDL = "reviewed_idl"
    SCHEMA_DEFAULT = "schema_default"
    MIGRATION_MANIFEST = "migration_manifest"
    COMPATIBILITY_POLICY = "compatibility_policy"
    GENERATED_MANIFEST = "generated_manifest"
    NORMATIVE_SPEC = "normative_spec"

    @classmethod
    def coerce(cls, value: Any) -> "AuthorityKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, AuthorityKind] = {
            "none": cls.NONE,
            "": cls.NONE,
            "reviewed_idl": cls.REVIEWED_IDL,
            "idl": cls.REVIEWED_IDL,
            "schema_default": cls.SCHEMA_DEFAULT,
            "default": cls.SCHEMA_DEFAULT,
            "migration_manifest": cls.MIGRATION_MANIFEST,
            "migration": cls.MIGRATION_MANIFEST,
            "compatibility_policy": cls.COMPATIBILITY_POLICY,
            "compat_policy": cls.COMPATIBILITY_POLICY,
            "generated_manifest": cls.GENERATED_MANIFEST,
            "normative_spec": cls.NORMATIVE_SPEC,
            "spec": cls.NORMATIVE_SPEC,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported authority kind: {value!r}"
            ) from exc


class WriteMode(str, Enum):
    """How an obligation may be discharged against a path root."""

    DIRECT = "direct"
    REGENERATION = "regeneration"
    EXTERNAL_OBLIGATION = "external_obligation"
    FRONTIER = "frontier"
    NONE = "none"

    @classmethod
    def coerce(cls, value: Any) -> "WriteMode":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, WriteMode] = {
            "direct": cls.DIRECT,
            "write": cls.DIRECT,
            "regeneration": cls.REGENERATION,
            "regenerate": cls.REGENERATION,
            "generated": cls.REGENERATION,
            "external_obligation": cls.EXTERNAL_OBLIGATION,
            "external": cls.EXTERNAL_OBLIGATION,
            "upstream": cls.EXTERNAL_OBLIGATION,
            "frontier": cls.FRONTIER,
            "none": cls.NONE,
            "n/a": cls.NONE,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported write mode: {value!r}"
            ) from exc


class SchemaConsumerRole(str, Enum):
    """Closed catalogue of schema/protocol consumer roles."""

    FIELD_READER = "field_reader"
    FIELD_WRITER = "field_writer"
    CONSTRUCTOR = "constructor"
    FACTORY = "factory"
    BUILDER = "builder"
    SERIALIZER = "serializer"
    DESERIALIZER = "deserializer"
    PERSISTENCE = "persistence"
    CACHE_KEY = "cache_key"
    EQUALITY_HASH = "equality_hash"
    VERSION_NEGOTIATION = "version_negotiation"
    MIGRATION = "migration"
    GENERATED_CLIENT = "generated_client"
    SCHEMA_SURFACE = "schema_surface"
    PROTOCOL_SURFACE = "protocol_surface"
    DYNAMIC_CODEC = "dynamic_codec"
    MISSING_CODEC = "missing_codec"

    @classmethod
    def coerce(cls, value: Any) -> "SchemaConsumerRole":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases: Mapping[str, SchemaConsumerRole] = {
            "field_reader": cls.FIELD_READER,
            "reader": cls.FIELD_READER,
            "field_writer": cls.FIELD_WRITER,
            "writer": cls.FIELD_WRITER,
            "constructor": cls.CONSTRUCTOR,
            "ctor": cls.CONSTRUCTOR,
            "factory": cls.FACTORY,
            "builder": cls.BUILDER,
            "serializer": cls.SERIALIZER,
            "deserializer": cls.DESERIALIZER,
            "persistence": cls.PERSISTENCE,
            "cache_key": cls.CACHE_KEY,
            "cache": cls.CACHE_KEY,
            "equality_hash": cls.EQUALITY_HASH,
            "equality": cls.EQUALITY_HASH,
            "hash": cls.EQUALITY_HASH,
            "version_negotiation": cls.VERSION_NEGOTIATION,
            "versioning": cls.VERSION_NEGOTIATION,
            "migration": cls.MIGRATION,
            "generated_client": cls.GENERATED_CLIENT,
            "generated": cls.GENERATED_CLIENT,
            "schema_surface": cls.SCHEMA_SURFACE,
            "schema": cls.SCHEMA_SURFACE,
            "protocol_surface": cls.PROTOCOL_SURFACE,
            "protocol": cls.PROTOCOL_SURFACE,
            "dynamic_codec": cls.DYNAMIC_CODEC,
            "dynamic": cls.DYNAMIC_CODEC,
            "missing_codec": cls.MISSING_CODEC,
            "missing": cls.MISSING_CODEC,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise SchemaProtocolChangeImpactError(
                f"unsupported schema consumer role: {value!r}"
            ) from exc


_REQUIRED_ROLES: Final[frozenset[str]] = frozenset(
    {
        SchemaConsumerRole.FIELD_READER.value,
        SchemaConsumerRole.FIELD_WRITER.value,
        SchemaConsumerRole.CONSTRUCTOR.value,
        SchemaConsumerRole.FACTORY.value,
        SchemaConsumerRole.BUILDER.value,
        SchemaConsumerRole.SERIALIZER.value,
        SchemaConsumerRole.DESERIALIZER.value,
        SchemaConsumerRole.PERSISTENCE.value,
        SchemaConsumerRole.CACHE_KEY.value,
        SchemaConsumerRole.EQUALITY_HASH.value,
        SchemaConsumerRole.VERSION_NEGOTIATION.value,
        SchemaConsumerRole.MIGRATION.value,
        SchemaConsumerRole.GENERATED_CLIENT.value,
        SchemaConsumerRole.SCHEMA_SURFACE.value,
        SchemaConsumerRole.PROTOCOL_SURFACE.value,
        SchemaConsumerRole.DYNAMIC_CODEC.value,
        SchemaConsumerRole.MISSING_CODEC.value,
    }
)

_REQUIRED_SURFACES: Final[frozenset[str]] = frozenset(
    item.value for item in SchemaSurfaceKind
)

_READER_ROLES: Final[frozenset[SchemaConsumerRole]] = frozenset(
    {
        SchemaConsumerRole.FIELD_READER,
        SchemaConsumerRole.DESERIALIZER,
        SchemaConsumerRole.VERSION_NEGOTIATION,
    }
)

_WRITER_ROLES: Final[frozenset[SchemaConsumerRole]] = frozenset(
    {
        SchemaConsumerRole.FIELD_WRITER,
        SchemaConsumerRole.CONSTRUCTOR,
        SchemaConsumerRole.FACTORY,
        SchemaConsumerRole.BUILDER,
        SchemaConsumerRole.SERIALIZER,
        SchemaConsumerRole.PERSISTENCE,
    }
)

_IDENTITY_ROLES: Final[frozenset[SchemaConsumerRole]] = frozenset(
    {
        SchemaConsumerRole.CACHE_KEY,
        SchemaConsumerRole.EQUALITY_HASH,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_FIELD_BYTES,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise SchemaProtocolChangeImpactError(f"{name} must be a string")
    if text != text.strip() or "\x00" in text:
        raise SchemaProtocolChangeImpactError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not text:
        raise SchemaProtocolChangeImpactError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise SchemaProtocolChangeImpactBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    result = _text(value, name, required=True)
    if any(character.isspace() for character in result):
        raise SchemaProtocolChangeImpactError(f"{name} must be a compact identifier")
    return result


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    coerce = getattr(kind, "coerce", None)
    if callable(coerce):
        return coerce(value)
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise SchemaProtocolChangeImpactError(f"invalid {name}: {value!r}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaProtocolChangeImpactError(f"{name} must be a boolean")
    return value


def _string_tuple(
    value: Any,
    name: str,
    *,
    limit: int = MAX_REFERENCE_COUNT,
    required: bool = False,
    sort: bool = True,
) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raise SchemaProtocolChangeImpactError(f"{name} must be a sequence of strings")
    elif isinstance(value, Sequence):
        items = value
    else:
        raise SchemaProtocolChangeImpactError(f"{name} must be a sequence of strings")
    if len(items) > limit:
        raise SchemaProtocolChangeImpactBoundsError(f"{name} exceeds its item bound")
    normalized = [
        _text(item, name, required=False)
        for item in items
        if item is not None and str(item).strip()
    ]
    result = tuple(sorted(set(normalized))) if sort else tuple(normalized)
    if required and not result:
        raise SchemaProtocolChangeImpactError(f"{name} is required")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    return str(value)


def _roots(value: Any) -> PropagationAuthorityRoots:
    if isinstance(value, PropagationAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            PropagationAuthorityRoots.from_dict(value)
            if "schema" in value
            else PropagationAuthorityRoots(**dict(value))
        )
    raise SchemaProtocolChangeImpactError("roots must be PropagationAuthorityRoots")


def _node_ref(value: Any) -> GraphNodeRef:
    if isinstance(value, GraphNodeRef):
        return value
    if isinstance(value, Mapping):
        return (
            GraphNodeRef.from_dict(value)
            if "schema" in value
            else GraphNodeRef(**dict(value))
        )
    raise SchemaProtocolChangeImpactError("node must be a GraphNodeRef")


def _path_is_generated(path: str) -> bool:
    lowered = f"/{path}".replace("\\", "/").casefold()
    return any(marker in lowered for marker in _GENERATED_PATH_MARKERS)


def _path_is_read_only(path: str) -> bool:
    lowered = f"/{path}".replace("\\", "/").casefold()
    return any(marker in lowered for marker in _READ_ONLY_PATH_MARKERS)


def _repo_path(value: Any, name: str = "path") -> str:
    path = _text(value, name, limit=MAX_PATH_BYTES)
    if path.startswith("/") or ".." in path.split("/"):
        raise SchemaProtocolChangeImpactError(
            f"{name} must be a repository-relative path without parent escapes"
        )
    return path


# ---------------------------------------------------------------------------
# Field / impact records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class SchemaFieldChange:
    """One added/removed/renamed/retyped field or variant mutation."""

    kind: FieldChangeKind
    field_name: str
    previous_name: str = ""
    previous_type_ref: str = ""
    type_ref: str = ""
    required: bool = False
    has_default: bool = False
    default_ref: str = ""
    default_authority: AuthorityKind = AuthorityKind.NONE
    variant: bool = False
    clause_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    reason: str = ""
    schema: str = SCHEMA_FIELD_CHANGE_SCHEMA
    change_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", FieldChangeKind.coerce(self.kind))
        object.__setattr__(
            self, "field_name", _identifier(self.field_name, "field_name")
        )
        object.__setattr__(
            self,
            "previous_name",
            _text(self.previous_name, "previous_name", required=False),
        )
        object.__setattr__(
            self,
            "previous_type_ref",
            _text(self.previous_type_ref, "previous_type_ref", required=False),
        )
        object.__setattr__(
            self, "type_ref", _text(self.type_ref, "type_ref", required=False)
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "has_default", _bool(self.has_default, "has_default"))
        object.__setattr__(
            self, "default_ref", _text(self.default_ref, "default_ref", required=False)
        )
        object.__setattr__(
            self,
            "default_authority",
            AuthorityKind.coerce(self.default_authority),
        )
        object.__setattr__(self, "variant", _bool(self.variant, "variant"))
        object.__setattr__(
            self, "clause_ids", _string_tuple(self.clause_ids, "clause_ids")
        )
        object.__setattr__(
            self, "evidence_refs", _string_tuple(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SCHEMA_FIELD_CHANGE_SCHEMA, "schema"),
        )
        if self.schema != SCHEMA_FIELD_CHANGE_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported field change schema: {self.schema}"
            )
        if self.kind in {
            FieldChangeKind.RENAMED,
            FieldChangeKind.VARIANT_RENAMED,
        } and not self.previous_name:
            raise SchemaProtocolChangeImpactError(
                "renamed field changes require previous_name"
            )
        if self.kind is FieldChangeKind.RETYPED and not (
            self.previous_type_ref or self.type_ref
        ):
            raise SchemaProtocolChangeImpactError(
                "retyped field changes require previous_type_ref and/or type_ref"
            )
        if self.has_default and not self.default_ref and not self.default_authority:
            # Defaults may be named only by authority; empty both is invalid.
            pass
        if self.has_default and self.default_authority is AuthorityKind.NONE:
            # Presence of a default without independent authority is allowed
            # as an observation, but analyzers must not treat it as discharge.
            pass
        claimed = str(self.change_id or "").strip()
        object.__setattr__(self, "change_id", "")
        actual = _identity("schema-field-change", self._identity_payload())
        if claimed and claimed != actual:
            raise SchemaProtocolChangeImpactError(
                "schema field change identity does not match payload"
            )
        object.__setattr__(self, "change_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "field_name": self.field_name,
            "previous_name": self.previous_name,
            "previous_type_ref": self.previous_type_ref,
            "type_ref": self.type_ref,
            "required": self.required,
            "has_default": self.has_default,
            "default_ref": self.default_ref,
            "default_authority": self.default_authority.value,
            "variant": self.variant,
            "clause_ids": list(self.clause_ids),
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }

    @property
    def default_has_independent_authority(self) -> bool:
        return self.has_default and self.default_authority is not AuthorityKind.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "change_id": self.change_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaFieldChange":
        if not isinstance(payload, Mapping):
            raise SchemaProtocolChangeImpactError("field change payload must be a mapping")
        return cls(
            kind=payload.get("kind") or FieldChangeKind.ADDED,
            field_name=str(payload.get("field_name") or ""),
            previous_name=str(payload.get("previous_name") or ""),
            previous_type_ref=str(payload.get("previous_type_ref") or ""),
            type_ref=str(payload.get("type_ref") or ""),
            required=bool(payload.get("required", False)),
            has_default=bool(payload.get("has_default", False)),
            default_ref=str(payload.get("default_ref") or ""),
            default_authority=payload.get("default_authority") or AuthorityKind.NONE,
            variant=bool(payload.get("variant", False)),
            clause_ids=tuple(payload.get("clause_ids") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            reason=str(payload.get("reason") or ""),
            change_id=str(payload.get("change_id") or ""),
        )


@dataclass(frozen=True)
class ConstructorImpact:
    """Impact of a schema change on constructors, factories, or builders."""

    kind: ConstructionKind
    subject_symbol_id: str
    path: str
    affected_field_names: tuple[str, ...] = ()
    added_required_fields: tuple[str, ...] = ()
    removed_fields: tuple[str, ...] = ()
    compatibility: CompatibilityDirection = CompatibilityDirection.UNKNOWN
    needs_independent_default_authority: bool = False
    clause_ids: tuple[str, ...] = ()
    reason: str = ""
    schema: str = CONSTRUCTOR_IMPACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ConstructionKind.coerce(self.kind))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self,
            "affected_field_names",
            _string_tuple(self.affected_field_names, "affected_field_names"),
        )
        object.__setattr__(
            self,
            "added_required_fields",
            _string_tuple(self.added_required_fields, "added_required_fields"),
        )
        object.__setattr__(
            self,
            "removed_fields",
            _string_tuple(self.removed_fields, "removed_fields"),
        )
        object.__setattr__(
            self,
            "compatibility",
            CompatibilityDirection.coerce(self.compatibility),
        )
        object.__setattr__(
            self,
            "needs_independent_default_authority",
            _bool(
                self.needs_independent_default_authority,
                "needs_independent_default_authority",
            ),
        )
        object.__setattr__(
            self, "clause_ids", _string_tuple(self.clause_ids, "clause_ids")
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or CONSTRUCTOR_IMPACT_SCHEMA, "schema"),
        )
        if self.schema != CONSTRUCTOR_IMPACT_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported constructor impact schema: {self.schema}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "subject_symbol_id": self.subject_symbol_id,
            "path": self.path,
            "affected_field_names": list(self.affected_field_names),
            "added_required_fields": list(self.added_required_fields),
            "removed_fields": list(self.removed_fields),
            "compatibility": self.compatibility.value,
            "needs_independent_default_authority": (
                self.needs_independent_default_authority
            ),
            "clause_ids": list(self.clause_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstructorImpact":
        return cls(
            kind=payload.get("kind") or ConstructionKind.CONSTRUCTOR,
            subject_symbol_id=str(payload.get("subject_symbol_id") or ""),
            path=str(payload.get("path") or ""),
            affected_field_names=tuple(payload.get("affected_field_names") or ()),
            added_required_fields=tuple(payload.get("added_required_fields") or ()),
            removed_fields=tuple(payload.get("removed_fields") or ()),
            compatibility=payload.get("compatibility")
            or CompatibilityDirection.UNKNOWN,
            needs_independent_default_authority=bool(
                payload.get("needs_independent_default_authority", False)
            ),
            clause_ids=tuple(payload.get("clause_ids") or ()),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class SerializationImpact:
    """Impact on serializers, persistence, identity, versioning, or migrations."""

    facet: SerializationFacet
    subject_symbol_id: str
    path: str
    affected_field_names: tuple[str, ...] = ()
    compatibility: CompatibilityDirection = CompatibilityDirection.UNKNOWN
    codec_status: str = "present"
    needs_migration_authority: bool = False
    write_mode: WriteMode = WriteMode.DIRECT
    clause_ids: tuple[str, ...] = ()
    reason: str = ""
    schema: str = SERIALIZATION_IMPACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "facet", SerializationFacet.coerce(self.facet))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self,
            "affected_field_names",
            _string_tuple(self.affected_field_names, "affected_field_names"),
        )
        object.__setattr__(
            self,
            "compatibility",
            CompatibilityDirection.coerce(self.compatibility),
        )
        status = _text(self.codec_status, "codec_status", required=True)
        if status not in {"present", "missing", "dynamic"}:
            raise SchemaProtocolChangeImpactError(
                "codec_status must be present, missing, or dynamic"
            )
        object.__setattr__(self, "codec_status", status)
        object.__setattr__(
            self,
            "needs_migration_authority",
            _bool(self.needs_migration_authority, "needs_migration_authority"),
        )
        object.__setattr__(self, "write_mode", WriteMode.coerce(self.write_mode))
        object.__setattr__(
            self, "clause_ids", _string_tuple(self.clause_ids, "clause_ids")
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SERIALIZATION_IMPACT_SCHEMA, "schema"),
        )
        if self.schema != SERIALIZATION_IMPACT_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported serialization impact schema: {self.schema}"
            )
        if self.codec_status in {"missing", "dynamic"} and self.write_mode is WriteMode.DIRECT:
            raise SchemaProtocolChangeImpactAuthorityError(
                "missing/dynamic codecs cannot claim direct write mode"
            )
        if (
            self.facet is SerializationFacet.GENERATED_CLIENT
            and self.write_mode is WriteMode.DIRECT
        ):
            raise SchemaProtocolChangeImpactAuthorityError(
                "generated clients cannot claim direct write mode"
            )

    @property
    def is_frontier(self) -> bool:
        return self.codec_status in {"missing", "dynamic"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "facet": self.facet.value,
            "subject_symbol_id": self.subject_symbol_id,
            "path": self.path,
            "affected_field_names": list(self.affected_field_names),
            "compatibility": self.compatibility.value,
            "codec_status": self.codec_status,
            "needs_migration_authority": self.needs_migration_authority,
            "write_mode": self.write_mode.value,
            "clause_ids": list(self.clause_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SerializationImpact":
        return cls(
            facet=payload.get("facet") or SerializationFacet.SERIALIZER,
            subject_symbol_id=str(payload.get("subject_symbol_id") or ""),
            path=str(payload.get("path") or ""),
            affected_field_names=tuple(payload.get("affected_field_names") or ()),
            compatibility=payload.get("compatibility")
            or CompatibilityDirection.UNKNOWN,
            codec_status=str(payload.get("codec_status") or "present"),
            needs_migration_authority=bool(
                payload.get("needs_migration_authority", False)
            ),
            write_mode=payload.get("write_mode") or WriteMode.DIRECT,
            clause_ids=tuple(payload.get("clause_ids") or ()),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class ProtocolImpact:
    """Impact on a public JSON/protobuf/IDL/database/message/RPC/HTTP/CLI surface."""

    surface: SchemaSurfaceKind
    subject_symbol_id: str
    path: str
    affected_field_names: tuple[str, ...] = ()
    compatibility: CompatibilityDirection = CompatibilityDirection.UNKNOWN
    version_negotiation_required: bool = False
    write_mode: WriteMode = WriteMode.DIRECT
    clause_ids: tuple[str, ...] = ()
    reason: str = ""
    schema: str = PROTOCOL_IMPACT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", SchemaSurfaceKind.coerce(self.surface))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self,
            "affected_field_names",
            _string_tuple(self.affected_field_names, "affected_field_names"),
        )
        object.__setattr__(
            self,
            "compatibility",
            CompatibilityDirection.coerce(self.compatibility),
        )
        object.__setattr__(
            self,
            "version_negotiation_required",
            _bool(self.version_negotiation_required, "version_negotiation_required"),
        )
        object.__setattr__(self, "write_mode", WriteMode.coerce(self.write_mode))
        object.__setattr__(
            self, "clause_ids", _string_tuple(self.clause_ids, "clause_ids")
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", required=False)
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or PROTOCOL_IMPACT_SCHEMA, "schema"),
        )
        if self.schema != PROTOCOL_IMPACT_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported protocol impact schema: {self.schema}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "surface": self.surface.value,
            "subject_symbol_id": self.subject_symbol_id,
            "path": self.path,
            "affected_field_names": list(self.affected_field_names),
            "compatibility": self.compatibility.value,
            "version_negotiation_required": self.version_negotiation_required,
            "write_mode": self.write_mode.value,
            "clause_ids": list(self.clause_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolImpact":
        return cls(
            surface=payload.get("surface") or SchemaSurfaceKind.JSON,
            subject_symbol_id=str(payload.get("subject_symbol_id") or ""),
            path=str(payload.get("path") or ""),
            affected_field_names=tuple(payload.get("affected_field_names") or ()),
            compatibility=payload.get("compatibility")
            or CompatibilityDirection.UNKNOWN,
            version_negotiation_required=bool(
                payload.get("version_negotiation_required", False)
            ),
            write_mode=payload.get("write_mode") or WriteMode.DIRECT,
            clause_ids=tuple(payload.get("clause_ids") or ()),
            reason=str(payload.get("reason") or ""),
        )


# ---------------------------------------------------------------------------
# Consumer observation / entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaConsumerObservation:
    """Bounded observation of one schema/protocol consumer of a changed type."""

    consumer_id: str
    role: SchemaConsumerRole
    path: str
    symbol_id: str
    subject_symbol_id: str
    surface: SchemaSurfaceKind | None = None
    construction_kind: ConstructionKind | None = None
    serialization_facet: SerializationFacet | None = None
    codec_status: str = "present"
    generated: bool = False
    read_only: bool = False
    supplies_field_names: tuple[str, ...] = ()
    ignores_unknown_fields: bool = False
    accepts_missing_optional: bool = True
    authority_refs: tuple[str, ...] = ()
    authority_kinds: tuple[AuthorityKind, ...] = ()
    node: GraphNodeRef | None = None
    evidence_refs: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_CONSUMER_OBSERVATION_SCHEMA
    observation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", _identifier(self.consumer_id, "consumer_id")
        )
        object.__setattr__(self, "role", SchemaConsumerRole.coerce(self.role))
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(self, "symbol_id", _identifier(self.symbol_id, "symbol_id"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        if self.surface is not None:
            object.__setattr__(
                self, "surface", SchemaSurfaceKind.coerce(self.surface)
            )
        if self.construction_kind is not None:
            object.__setattr__(
                self,
                "construction_kind",
                ConstructionKind.coerce(self.construction_kind),
            )
        if self.serialization_facet is not None:
            object.__setattr__(
                self,
                "serialization_facet",
                SerializationFacet.coerce(self.serialization_facet),
            )
        status = _text(self.codec_status or "present", "codec_status")
        if status not in {"present", "missing", "dynamic"}:
            raise SchemaProtocolChangeImpactError(
                "codec_status must be present, missing, or dynamic"
            )
        object.__setattr__(self, "codec_status", status)
        generated = _bool(self.generated, "generated") or _path_is_generated(self.path)
        read_only = _bool(self.read_only, "read_only") or _path_is_read_only(self.path)
        object.__setattr__(self, "generated", generated)
        object.__setattr__(self, "read_only", read_only)
        object.__setattr__(
            self,
            "supplies_field_names",
            _string_tuple(self.supplies_field_names, "supplies_field_names"),
        )
        object.__setattr__(
            self,
            "ignores_unknown_fields",
            _bool(self.ignores_unknown_fields, "ignores_unknown_fields"),
        )
        object.__setattr__(
            self,
            "accepts_missing_optional",
            _bool(self.accepts_missing_optional, "accepts_missing_optional"),
        )
        object.__setattr__(
            self, "authority_refs", _string_tuple(self.authority_refs, "authority_refs")
        )
        kinds: list[AuthorityKind] = []
        raw_kinds = self.authority_kinds or ()
        if isinstance(raw_kinds, (str, bytes, bytearray)) or not isinstance(
            raw_kinds, Sequence
        ):
            raise SchemaProtocolChangeImpactError(
                "authority_kinds must be a sequence"
            )
        for item in raw_kinds:
            kind = AuthorityKind.coerce(item)
            if kind is not AuthorityKind.NONE and kind not in kinds:
                kinds.append(kind)
        object.__setattr__(
            self,
            "authority_kinds",
            tuple(sorted(kinds, key=lambda item: item.value)),
        )
        if self.node is not None:
            object.__setattr__(self, "node", _node_ref(self.node))
        object.__setattr__(
            self, "evidence_refs", _string_tuple(self.evidence_refs, "evidence_refs")
        )
        attrs = self.attributes or {}
        if not isinstance(attrs, Mapping):
            raise SchemaProtocolChangeImpactError("attributes must be a mapping")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType({str(key): _plain(value) for key, value in attrs.items()}),
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SCHEMA_CONSUMER_OBSERVATION_SCHEMA, "schema"),
        )
        if self.schema != SCHEMA_CONSUMER_OBSERVATION_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported consumer observation schema: {self.schema}"
            )
        # Role / facet consistency soft-checks.
        if self.role is SchemaConsumerRole.CONSTRUCTOR and self.construction_kind is None:
            object.__setattr__(self, "construction_kind", ConstructionKind.CONSTRUCTOR)
        if self.role is SchemaConsumerRole.FACTORY and self.construction_kind is None:
            object.__setattr__(self, "construction_kind", ConstructionKind.FACTORY)
        if self.role is SchemaConsumerRole.BUILDER and self.construction_kind is None:
            object.__setattr__(self, "construction_kind", ConstructionKind.BUILDER)
        claimed = str(self.observation_id or "").strip()
        object.__setattr__(self, "observation_id", "")
        actual = _identity("schema-consumer-observation", self._identity_payload())
        if claimed and claimed != actual:
            raise SchemaProtocolChangeImpactError(
                "schema consumer observation identity does not match payload"
            )
        object.__setattr__(self, "observation_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "consumer_id": self.consumer_id,
            "role": self.role.value,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "subject_symbol_id": self.subject_symbol_id,
            "surface": self.surface.value if self.surface else "",
            "construction_kind": (
                self.construction_kind.value if self.construction_kind else ""
            ),
            "serialization_facet": (
                self.serialization_facet.value if self.serialization_facet else ""
            ),
            "codec_status": self.codec_status,
            "generated": self.generated,
            "read_only": self.read_only,
            "supplies_field_names": list(self.supplies_field_names),
            "ignores_unknown_fields": self.ignores_unknown_fields,
            "accepts_missing_optional": self.accepts_missing_optional,
            "authority_refs": list(self.authority_refs),
            "authority_kinds": [item.value for item in self.authority_kinds],
            "node": self.node.to_dict() if self.node is not None else None,
            "evidence_refs": list(self.evidence_refs),
            "attributes": dict(self.attributes),
        }

    def has_authority(self, kind: AuthorityKind) -> bool:
        return kind in self.authority_kinds

    def supplies_field(self, name: str) -> bool:
        return name in self.supplies_field_names

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "observation_id": self.observation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaConsumerObservation":
        if not isinstance(payload, Mapping):
            raise SchemaProtocolChangeImpactError(
                "consumer observation payload must be a mapping"
            )
        surface = payload.get("surface") or None
        if surface == "":
            surface = None
        construction = payload.get("construction_kind") or None
        if construction == "":
            construction = None
        facet = payload.get("serialization_facet") or None
        if facet == "":
            facet = None
        node = payload.get("node")
        return cls(
            consumer_id=str(payload.get("consumer_id") or ""),
            role=payload.get("role") or SchemaConsumerRole.FIELD_READER,
            path=str(payload.get("path") or ""),
            symbol_id=str(payload.get("symbol_id") or ""),
            subject_symbol_id=str(payload.get("subject_symbol_id") or ""),
            surface=surface,
            construction_kind=construction,
            serialization_facet=facet,
            codec_status=str(payload.get("codec_status") or "present"),
            generated=bool(payload.get("generated", False)),
            read_only=bool(payload.get("read_only", False)),
            supplies_field_names=tuple(payload.get("supplies_field_names") or ()),
            ignores_unknown_fields=bool(payload.get("ignores_unknown_fields", False)),
            accepts_missing_optional=bool(
                payload.get("accepts_missing_optional", True)
            ),
            authority_refs=tuple(payload.get("authority_refs") or ()),
            authority_kinds=tuple(payload.get("authority_kinds") or ()),
            node=node,
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            attributes=dict(payload.get("attributes") or {}),
            observation_id=str(payload.get("observation_id") or ""),
        )


@dataclass(frozen=True)
class SchemaConsumerImpactEntry:
    """One per-consumer compatibility and migration obligation row."""

    observation: SchemaConsumerObservation
    disposition: ConsumerDisposition
    compatibility: CompatibilityDirection
    write_mode: WriteMode
    affected_field_names: tuple[str, ...] = ()
    clause_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    required_authority: tuple[AuthorityKind, ...] = ()
    obligation: ConsumerMigrationObligation | None = None
    constructor_impact: ConstructorImpact | None = None
    serialization_impact: SerializationImpact | None = None
    protocol_impact: ProtocolImpact | None = None
    schema: str = SCHEMA_CONSUMER_IMPACT_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.observation, SchemaConsumerObservation):
            raise SchemaProtocolChangeImpactError(
                "observation must be SchemaConsumerObservation"
            )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ConsumerDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "compatibility",
            CompatibilityDirection.coerce(self.compatibility),
        )
        object.__setattr__(self, "write_mode", WriteMode.coerce(self.write_mode))
        object.__setattr__(
            self,
            "affected_field_names",
            _string_tuple(self.affected_field_names, "affected_field_names"),
        )
        object.__setattr__(
            self, "clause_ids", _string_tuple(self.clause_ids, "clause_ids")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(self.reason_codes, "reason_codes", limit=MAX_REASON_CODES),
        )
        authorities: list[AuthorityKind] = []
        for item in self.required_authority or ():
            kind = AuthorityKind.coerce(item)
            if kind is not AuthorityKind.NONE and kind not in authorities:
                authorities.append(kind)
        object.__setattr__(
            self,
            "required_authority",
            tuple(sorted(authorities, key=lambda item: item.value)),
        )
        if self.obligation is not None and not isinstance(
            self.obligation, ConsumerMigrationObligation
        ):
            raise SchemaProtocolChangeImpactError(
                "obligation must be ConsumerMigrationObligation"
            )
        if self.constructor_impact is not None and not isinstance(
            self.constructor_impact, ConstructorImpact
        ):
            raise SchemaProtocolChangeImpactError(
                "constructor_impact must be ConstructorImpact"
            )
        if self.serialization_impact is not None and not isinstance(
            self.serialization_impact, SerializationImpact
        ):
            raise SchemaProtocolChangeImpactError(
                "serialization_impact must be SerializationImpact"
            )
        if self.protocol_impact is not None and not isinstance(
            self.protocol_impact, ProtocolImpact
        ):
            raise SchemaProtocolChangeImpactError(
                "protocol_impact must be ProtocolImpact"
            )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SCHEMA_CONSUMER_IMPACT_ENTRY_SCHEMA, "schema"),
        )
        if self.schema != SCHEMA_CONSUMER_IMPACT_ENTRY_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported impact entry schema: {self.schema}"
            )
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        if self.disposition is ConsumerDisposition.COMPATIBLE:
            if self.compatibility is CompatibilityDirection.INCOMPATIBLE:
                raise SchemaProtocolChangeImpactAuthorityError(
                    "compatible disposition cannot claim incompatible direction"
                )
            if self.obligation is not None and (
                self.obligation.missing_input_ids
                or self.obligation.behavior_contract_ids
            ):
                raise SchemaProtocolChangeImpactAuthorityError(
                    "compatible obligations cannot require missing inputs"
                )
            if self.required_authority:
                raise SchemaProtocolChangeImpactAuthorityError(
                    "compatible disposition cannot still require authority"
                )
        if self.disposition is ConsumerDisposition.FRONTIER:
            if self.write_mode not in {WriteMode.FRONTIER, WriteMode.NONE}:
                raise SchemaProtocolChangeImpactAuthorityError(
                    "frontier disposition requires frontier/none write mode"
                )
            if self.obligation is not None and self.obligation.proof_refs:
                raise SchemaProtocolChangeImpactAuthorityError(
                    "frontier obligations cannot carry proof authority"
                )
        if self.write_mode is WriteMode.DIRECT and (
            self.observation.generated or self.observation.read_only
        ):
            raise SchemaProtocolChangeImpactAuthorityError(
                "generated/read-only consumers cannot use direct write mode"
            )
        if self.disposition is ConsumerDisposition.MIGRATE and self.obligation is None:
            raise SchemaProtocolChangeImpactError(
                "migrate disposition requires an obligation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "observation": self.observation.to_dict(),
            "disposition": self.disposition.value,
            "compatibility": self.compatibility.value,
            "write_mode": self.write_mode.value,
            "affected_field_names": list(self.affected_field_names),
            "clause_ids": list(self.clause_ids),
            "reason_codes": list(self.reason_codes),
            "required_authority": [item.value for item in self.required_authority],
            "obligation": self.obligation.to_dict() if self.obligation else None,
            "constructor_impact": (
                self.constructor_impact.to_dict() if self.constructor_impact else None
            ),
            "serialization_impact": (
                self.serialization_impact.to_dict()
                if self.serialization_impact
                else None
            ),
            "protocol_impact": (
                self.protocol_impact.to_dict() if self.protocol_impact else None
            ),
        }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaProtocolImpact:
    """Deterministic schema/protocol impact report for one contract delta.

    Also known in objective AST queries as ``SchemaProtocolImpact``.
    """

    roots: PropagationAuthorityRoots
    delta_id: str
    subject_symbol_id: str
    field_changes: tuple[SchemaFieldChange, ...]
    entries: tuple[SchemaConsumerImpactEntry, ...]
    constructor_impacts: tuple[ConstructorImpact, ...] = ()
    serialization_impacts: tuple[SerializationImpact, ...] = ()
    protocol_impacts: tuple[ProtocolImpact, ...] = ()
    frontier_consumer_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    schema: str = SCHEMA_PROTOCOL_IMPACT_SCHEMA
    contract_version: int = CHANGE_PROPAGATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "delta_id", _identifier(self.delta_id, "delta_id"))
        object.__setattr__(
            self,
            "subject_symbol_id",
            _identifier(self.subject_symbol_id, "subject_symbol_id"),
        )
        if not isinstance(self.field_changes, Sequence) or isinstance(
            self.field_changes, (str, bytes, bytearray)
        ):
            raise SchemaProtocolChangeImpactError("field_changes must be a sequence")
        if len(self.field_changes) > MAX_FIELD_CHANGES:
            raise SchemaProtocolChangeImpactBoundsError(
                "field_changes exceeds its item bound"
            )
        if not all(isinstance(item, SchemaFieldChange) for item in self.field_changes):
            raise SchemaProtocolChangeImpactError(
                "field_changes must contain SchemaFieldChange values"
            )
        field_changes = tuple(
            sorted(
                self.field_changes,
                key=lambda item: (item.kind.value, item.field_name, item.change_id),
            )
        )
        object.__setattr__(self, "field_changes", field_changes)

        if not isinstance(self.entries, Sequence) or isinstance(
            self.entries, (str, bytes, bytearray)
        ):
            raise SchemaProtocolChangeImpactError("entries must be a sequence")
        if len(self.entries) > MAX_ENTRIES:
            raise SchemaProtocolChangeImpactBoundsError(
                "entries exceeds its item bound"
            )
        if not all(isinstance(item, SchemaConsumerImpactEntry) for item in self.entries):
            raise SchemaProtocolChangeImpactError(
                "entries must contain SchemaConsumerImpactEntry values"
            )
        consumer_ids = [item.observation.consumer_id for item in self.entries]
        if len(consumer_ids) != len(set(consumer_ids)):
            raise SchemaProtocolChangeImpactError(
                "impact entries must have unique consumer_ids"
            )
        entries = tuple(
            sorted(
                self.entries,
                key=lambda item: (
                    item.observation.role.value,
                    item.observation.path,
                    item.observation.consumer_id,
                ),
            )
        )
        object.__setattr__(self, "entries", entries)

        object.__setattr__(
            self,
            "constructor_impacts",
            tuple(
                sorted(
                    self.constructor_impacts,
                    key=lambda item: (item.kind.value, item.path, item.subject_symbol_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "serialization_impacts",
            tuple(
                sorted(
                    self.serialization_impacts,
                    key=lambda item: (
                        item.facet.value,
                        item.path,
                        item.subject_symbol_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "protocol_impacts",
            tuple(
                sorted(
                    self.protocol_impacts,
                    key=lambda item: (
                        item.surface.value,
                        item.path,
                        item.subject_symbol_id,
                    ),
                )
            ),
        )
        frontiers = _string_tuple(
            self.frontier_consumer_ids or (),
            "frontier_consumer_ids",
            limit=MAX_ENTRIES,
        )
        derived_frontiers = tuple(
            item.observation.consumer_id
            for item in entries
            if item.disposition is ConsumerDisposition.FRONTIER
        )
        object.__setattr__(
            self,
            "frontier_consumer_ids",
            tuple(sorted(set(frontiers) | set(derived_frontiers))),
        )
        object.__setattr__(
            self, "evidence_refs", _string_tuple(self.evidence_refs, "evidence_refs")
        )
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or SCHEMA_PROTOCOL_IMPACT_SCHEMA, "schema"),
        )
        if self.schema != SCHEMA_PROTOCOL_IMPACT_SCHEMA:
            raise SchemaProtocolChangeImpactError(
                f"unsupported schema protocol impact schema: {self.schema}"
            )
        if not isinstance(self.contract_version, int) or self.contract_version < 1:
            raise SchemaProtocolChangeImpactError(
                "contract_version must be a positive integer"
            )
        # One compatible consumer cannot clear others.
        if self.compatible_entries and self.migrate_entries:
            if not self.one_compatible_cannot_discharge_others():
                raise SchemaProtocolChangeImpactAuthorityError(
                    "compatible consumers cannot discharge independent migrate obligations"
                )

    @property
    def migrate_entries(self) -> tuple[SchemaConsumerImpactEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.disposition
            in {
                ConsumerDisposition.MIGRATE,
                ConsumerDisposition.ADAPTER,
                ConsumerDisposition.UPSTREAM,
            }
        )

    @property
    def compatible_entries(self) -> tuple[SchemaConsumerImpactEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.disposition is ConsumerDisposition.COMPATIBLE
        )

    @property
    def frontier_entries(self) -> tuple[SchemaConsumerImpactEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.disposition is ConsumerDisposition.FRONTIER
        )

    @property
    def obligations(self) -> tuple[ConsumerMigrationObligation, ...]:
        return tuple(
            item.obligation for item in self.entries if item.obligation is not None
        )

    def obligation_set_id(self) -> str:
        obligations = self.obligations
        if not obligations:
            raise SchemaProtocolChangeImpactError(
                "obligation set identity requires at least one obligation"
            )
        return obligation_set_identity(obligations)

    def one_compatible_cannot_discharge_others(self) -> bool:
        """Structural invariant: compatible rows never clear migrate rows."""
        if not self.compatible_entries:
            return True
        return bool(self.migrate_entries) or not any(
            item.disposition
            in {
                ConsumerDisposition.MIGRATE,
                ConsumerDisposition.ADAPTER,
                ConsumerDisposition.UPSTREAM,
            }
            for item in self.entries
        )

    def compatibility_by_consumer(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                item.observation.consumer_id: item.compatibility.value
                for item in self.entries
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contract_version": self.contract_version,
            "producer_id": self.producer_id,
            "roots": self.roots.to_dict(),
            "delta_id": self.delta_id,
            "subject_symbol_id": self.subject_symbol_id,
            "field_changes": [item.to_dict() for item in self.field_changes],
            "entries": [item.to_dict() for item in self.entries],
            "constructor_impacts": [item.to_dict() for item in self.constructor_impacts],
            "serialization_impacts": [
                item.to_dict() for item in self.serialization_impacts
            ],
            "protocol_impacts": [item.to_dict() for item in self.protocol_impacts],
            "frontier_consumer_ids": list(self.frontier_consumer_ids),
            "evidence_refs": list(self.evidence_refs),
        }


# Alias used by objective AST queries.
SchemaProtocolChangeImpact = SchemaProtocolImpact


# ---------------------------------------------------------------------------
# Field-change extraction and compatibility rules
# ---------------------------------------------------------------------------


def _clause_field_changes(clause: ContractClauseDelta) -> list[SchemaFieldChange]:
    """Best-effort extraction of field mutations from a clause reason/kind."""
    reason = clause.reason or ""
    changes: list[SchemaFieldChange] = []
    rename = _RENAME_RE.search(reason)
    if rename:
        changes.append(
            SchemaFieldChange(
                kind=(
                    FieldChangeKind.VARIANT_RENAMED
                    if "variant" in reason.casefold() or "case" in reason.casefold()
                    else FieldChangeKind.RENAMED
                ),
                field_name=rename.group(2),
                previous_name=rename.group(1),
                variant="variant" in reason.casefold() or "case" in reason.casefold(),
                clause_ids=(clause.clause_id,),
                reason=reason,
            )
        )
        return changes

    retype = _RETYPE_RE.search(reason)
    if retype:
        changes.append(
            SchemaFieldChange(
                kind=FieldChangeKind.RETYPED,
                field_name=retype.group(1),
                previous_type_ref=retype.group(2).strip(),
                type_ref=retype.group(3).strip(),
                clause_ids=(clause.clause_id,),
                reason=reason,
            )
        )
        return changes

    names = _FIELD_NAME_RE.findall(reason)
    lowered = reason.casefold()
    is_variant = "variant" in lowered or "case" in lowered or "enum" in lowered
    required = "required" in lowered and "optional" not in lowered
    has_default = "default" in lowered
    default_authority = AuthorityKind.NONE
    if has_default and any(
        token in lowered
        for token in ("reviewed", "idl", "manifest", "schema_default", "normative")
    ):
        if "migration" in lowered:
            default_authority = AuthorityKind.MIGRATION_MANIFEST
        elif "idl" in lowered or "reviewed" in lowered:
            default_authority = AuthorityKind.REVIEWED_IDL
        elif "schema_default" in lowered or "default provider" in lowered:
            default_authority = AuthorityKind.SCHEMA_DEFAULT
        else:
            default_authority = AuthorityKind.COMPATIBILITY_POLICY

    if clause.kind is DeltaKind.FIELD_INTRO or any(
        token in lowered for token in ("field add", "added field", "introduc")
    ):
        kind = FieldChangeKind.VARIANT_ADDED if is_variant else FieldChangeKind.ADDED
        for name in names or ("field",):
            changes.append(
                SchemaFieldChange(
                    kind=kind,
                    field_name=name,
                    required=required,
                    has_default=has_default,
                    default_ref="default:present" if has_default else "",
                    default_authority=default_authority,
                    variant=is_variant,
                    clause_ids=(clause.clause_id,),
                    reason=reason,
                )
            )
        return changes

    if clause.kind is DeltaKind.FIELD_REMOVE or any(
        token in lowered for token in ("field remove", "removed field", "delete field")
    ):
        kind = (
            FieldChangeKind.VARIANT_REMOVED if is_variant else FieldChangeKind.REMOVED
        )
        for name in names or ("field",):
            changes.append(
                SchemaFieldChange(
                    kind=kind,
                    field_name=name,
                    required=required,
                    variant=is_variant,
                    clause_ids=(clause.clause_id,),
                    reason=reason,
                )
            )
        return changes

    if clause.kind in {
        DeltaKind.SCHEMA_CHANGE,
        DeltaKind.SERIALIZATION_CHANGE,
        DeltaKind.PROTOCOL_CHANGE,
        DeltaKind.DATA_STRUCTURE_INTRO,
        DeltaKind.DATA_STRUCTURE_REMOVE,
        DeltaKind.NULLABILITY_CHANGE,
    }:
        # Generic schema clause: treat named fields as added when disposition
        # is breaking and reason mentions add, else unknown structural.
        if names:
            if any(token in lowered for token in ("remove", "delete", "drop")):
                kind = (
                    FieldChangeKind.VARIANT_REMOVED
                    if is_variant
                    else FieldChangeKind.REMOVED
                )
            elif any(token in lowered for token in ("rename",)):
                kind = (
                    FieldChangeKind.VARIANT_RENAMED
                    if is_variant
                    else FieldChangeKind.RENAMED
                )
            elif any(token in lowered for token in ("retype", "type change")):
                kind = FieldChangeKind.RETYPED
            else:
                kind = (
                    FieldChangeKind.VARIANT_ADDED if is_variant else FieldChangeKind.ADDED
                )
            for name in names:
                prev = ""
                if kind in {FieldChangeKind.RENAMED, FieldChangeKind.VARIANT_RENAMED}:
                    # Without explicit previous, keep name as both markers fail closed.
                    prev = name
                changes.append(
                    SchemaFieldChange(
                        kind=kind,
                        field_name=name,
                        previous_name=prev,
                        required=required,
                        has_default=has_default,
                        default_ref="default:present" if has_default else "",
                        default_authority=default_authority,
                        variant=is_variant,
                        clause_ids=(clause.clause_id,),
                        reason=reason,
                    )
                )
    return changes


def extract_field_changes(
    delta: ProgramContractDelta,
    explicit: Sequence[SchemaFieldChange | Mapping[str, Any]] = (),
) -> tuple[SchemaFieldChange, ...]:
    """Merge explicit field changes with those extracted from delta clauses."""
    collected: list[SchemaFieldChange] = []
    seen: set[str] = set()

    def _add(change: SchemaFieldChange) -> None:
        key = f"{change.kind.value}:{change.field_name}:{change.previous_name}:{change.type_ref}"
        if key not in seen:
            seen.add(key)
            collected.append(change)

    for raw in explicit or ():
        change = (
            raw
            if isinstance(raw, SchemaFieldChange)
            else SchemaFieldChange.from_dict(raw)
        )
        _add(change)

    for clause in delta.clauses:
        if clause.kind not in _SCHEMA_DELTA_KINDS and clause.kind not in {
            DeltaKind.FIELD_INTRO,
            DeltaKind.FIELD_REMOVE,
        }:
            # Still attempt extraction for SCHEMA/SERIALIZATION/PROTOCOL etc.
            if clause.kind not in {
                DeltaKind.SCHEMA_CHANGE,
                DeltaKind.SERIALIZATION_CHANGE,
                DeltaKind.PROTOCOL_CHANGE,
            }:
                continue
        for change in _clause_field_changes(clause):
            _add(change)

    return tuple(
        sorted(
            collected,
            key=lambda item: (item.kind.value, item.field_name, item.change_id),
        )
    )


def classify_field_compatibility(
    change: SchemaFieldChange,
    role: SchemaConsumerRole,
    *,
    ignores_unknown_fields: bool = False,
    accepts_missing_optional: bool = True,
    supplies_field: bool = False,
    has_default_authority: bool = False,
) -> tuple[CompatibilityDirection, tuple[str, ...], tuple[AuthorityKind, ...]]:
    """Classify one field change for one consumer role."""
    reasons: list[str] = []
    authority: list[AuthorityKind] = []
    is_reader = role in _READER_ROLES or role is SchemaConsumerRole.SCHEMA_SURFACE
    is_writer = role in _WRITER_ROLES
    is_identity = role in _IDENTITY_ROLES
    is_migration = role is SchemaConsumerRole.MIGRATION
    is_generated = role is SchemaConsumerRole.GENERATED_CLIENT
    is_protocol = role is SchemaConsumerRole.PROTOCOL_SURFACE

    if role in {
        SchemaConsumerRole.DYNAMIC_CODEC,
        SchemaConsumerRole.MISSING_CODEC,
    }:
        return (
            CompatibilityDirection.UNKNOWN,
            ("codec_frontier", f"role:{role.value}"),
            (),
        )

    kind = change.kind

    if kind is FieldChangeKind.ADDED:
        if change.required and not change.has_default:
            if supplies_field:
                reasons.append(f"supplies_required_field:{change.field_name}")
                return CompatibilityDirection.FULL, tuple(reasons), ()
            if is_writer or is_generated or is_protocol or is_migration:
                reasons.append(f"required_field_added:{change.field_name}")
                return (
                    CompatibilityDirection.INCOMPATIBLE,
                    tuple(reasons),
                    (),
                )
            if is_reader:
                # New required field: old payloads lack it → not backward for new readers.
                reasons.append(f"reader_requires_new_field:{change.field_name}")
                return (
                    CompatibilityDirection.INCOMPATIBLE,
                    tuple(reasons),
                    (),
                )
            if is_identity:
                reasons.append(f"identity_includes_new_required:{change.field_name}")
                return (
                    CompatibilityDirection.INCOMPATIBLE,
                    tuple(reasons),
                    (),
                )
            reasons.append(f"required_field_added:{change.field_name}")
            return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()

        # Optional or defaulted add.
        if change.has_default and not (
            change.default_has_independent_authority or has_default_authority
        ):
            authority.append(AuthorityKind.SCHEMA_DEFAULT)
            reasons.append("required_default_needs_independent_authority")
            # Without authority, cannot claim full compatibility.
            if is_writer or is_generated:
                return (
                    CompatibilityDirection.UNKNOWN,
                    tuple(reasons),
                    tuple(authority),
                )

        if is_reader:
            if accepts_missing_optional or change.has_default:
                reasons.append(f"optional_field_added_reader_ok:{change.field_name}")
                # New optional fields: new readers accept old payloads (backward).
                # Old readers need ignore-unknown for forward; this is the new reader.
                return CompatibilityDirection.BACKWARD, tuple(reasons), tuple(authority)
            reasons.append(f"reader_rejects_missing_optional:{change.field_name}")
            return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), tuple(authority)

        if is_writer:
            if supplies_field or change.has_default:
                reasons.append(f"writer_can_emit_optional:{change.field_name}")
                # Old writers omit the field → forward for readers that ignore unknowns
                # is a reader concern; writers of new schema are fine.
                return CompatibilityDirection.FULL, tuple(reasons), tuple(authority)
            reasons.append(f"writer_must_emit_new_field:{change.field_name}")
            return CompatibilityDirection.FORWARD, tuple(reasons), tuple(authority)

        if is_identity:
            reasons.append(f"identity_may_ignore_optional:{change.field_name}")
            return CompatibilityDirection.FORWARD, tuple(reasons), tuple(authority)

        if is_migration:
            authority.append(AuthorityKind.MIGRATION_MANIFEST)
            reasons.append("migration_required_for_field_add")
            return CompatibilityDirection.UNKNOWN, tuple(reasons), tuple(authority)

        if ignores_unknown_fields or is_protocol or is_generated:
            reasons.append(f"optional_field_added:{change.field_name}")
            return CompatibilityDirection.FORWARD, tuple(reasons), tuple(authority)

        reasons.append(f"optional_field_added:{change.field_name}")
        return CompatibilityDirection.FORWARD, tuple(reasons), tuple(authority)

    if kind is FieldChangeKind.REMOVED:
        if supplies_field:
            # Consumer still references the removed field.
            reasons.append(f"consumer_still_uses_removed_field:{change.field_name}")
            return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()
        if is_migration:
            authority.append(AuthorityKind.MIGRATION_MANIFEST)
            reasons.append("migration_required_for_field_remove")
            return (
                CompatibilityDirection.INCOMPATIBLE,
                tuple(reasons),
                tuple(authority),
            )
        if is_reader:
            reasons.append(f"reader_field_removed:{change.field_name}")
            return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()
        if is_writer:
            # Writers no longer emit it; old readers that required it break (not their concern
            # fully) — writer itself is backward relative to new schema.
            reasons.append(f"writer_field_removed:{change.field_name}")
            return CompatibilityDirection.BACKWARD, tuple(reasons), ()
        if is_identity:
            reasons.append(f"identity_field_removed:{change.field_name}")
            return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()
        reasons.append(f"field_removed:{change.field_name}")
        return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()

    if kind in {FieldChangeKind.RENAMED, FieldChangeKind.RETYPED}:
        if supplies_field and kind is FieldChangeKind.RENAMED:
            # Still using old name → incompatible until migrated.
            reasons.append(f"uses_old_name:{change.previous_name or change.field_name}")
        reasons.append(f"field_{kind.value}:{change.field_name}")
        if is_migration:
            authority.append(AuthorityKind.MIGRATION_MANIFEST)
            return (
                CompatibilityDirection.INCOMPATIBLE,
                tuple(reasons),
                tuple(authority),
            )
        return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()

    if kind is FieldChangeKind.VARIANT_ADDED:
        if is_reader and not ignores_unknown_fields:
            reasons.append(f"reader_unknown_variant:{change.field_name}")
            return CompatibilityDirection.BACKWARD, tuple(reasons), ()
        if is_reader:
            reasons.append(f"reader_ignores_unknown_variant:{change.field_name}")
            return CompatibilityDirection.FULL, tuple(reasons), ()
        if is_writer:
            reasons.append(f"writer_variant_added_optional:{change.field_name}")
            return CompatibilityDirection.FULL, tuple(reasons), ()
        reasons.append(f"variant_added:{change.field_name}")
        return CompatibilityDirection.FORWARD, tuple(reasons), ()

    if kind is FieldChangeKind.VARIANT_REMOVED:
        reasons.append(f"variant_removed:{change.field_name}")
        if is_migration:
            authority.append(AuthorityKind.MIGRATION_MANIFEST)
            return (
                CompatibilityDirection.INCOMPATIBLE,
                tuple(reasons),
                tuple(authority),
            )
        return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()

    if kind is FieldChangeKind.VARIANT_RENAMED:
        reasons.append(f"variant_renamed:{change.field_name}")
        if is_migration:
            authority.append(AuthorityKind.MIGRATION_MANIFEST)
            return (
                CompatibilityDirection.INCOMPATIBLE,
                tuple(reasons),
                tuple(authority),
            )
        return CompatibilityDirection.INCOMPATIBLE, tuple(reasons), ()

    return CompatibilityDirection.UNKNOWN, ("unclassified_field_change",), ()


def _merge_compatibility(
    current: CompatibilityDirection, new: CompatibilityDirection
) -> CompatibilityDirection:
    rank = {
        CompatibilityDirection.FULL: 0,
        CompatibilityDirection.BACKWARD: 1,
        CompatibilityDirection.FORWARD: 1,
        CompatibilityDirection.UNKNOWN: 2,
        CompatibilityDirection.INCOMPATIBLE: 3,
    }
    if current is new:
        return current
    if (
        {current, new}
        == {CompatibilityDirection.BACKWARD, CompatibilityDirection.FORWARD}
    ):
        return CompatibilityDirection.FULL
    if rank[new] > rank[current]:
        return new
    if rank[current] > rank[new]:
        return current
    # Equal rank but different (backward vs forward already handled).
    return CompatibilityDirection.UNKNOWN


def classify_consumer_compatibility(
    changes: Sequence[SchemaFieldChange],
    observation: SchemaConsumerObservation,
) -> tuple[CompatibilityDirection, tuple[str, ...], tuple[AuthorityKind, ...]]:
    """Aggregate field-change compatibility for one consumer observation."""
    if observation.role in {
        SchemaConsumerRole.DYNAMIC_CODEC,
        SchemaConsumerRole.MISSING_CODEC,
    } or observation.codec_status in {"missing", "dynamic"}:
        status = (
            "missing_codec"
            if observation.codec_status == "missing"
            or observation.role is SchemaConsumerRole.MISSING_CODEC
            else "dynamic_codec"
        )
        return (
            CompatibilityDirection.UNKNOWN,
            (status, "frontier_codec"),
            (),
        )

    if not changes:
        return CompatibilityDirection.FULL, ("no_field_changes",), ()

    direction = CompatibilityDirection.FULL
    reasons: list[str] = []
    authority: list[AuthorityKind] = []
    for change in changes:
        has_auth = (
            change.default_has_independent_authority
            or observation.has_authority(AuthorityKind.SCHEMA_DEFAULT)
            or observation.has_authority(AuthorityKind.REVIEWED_IDL)
            or observation.has_authority(AuthorityKind.COMPATIBILITY_POLICY)
        )
        item_dir, item_reasons, item_auth = classify_field_compatibility(
            change,
            observation.role,
            ignores_unknown_fields=observation.ignores_unknown_fields,
            accepts_missing_optional=observation.accepts_missing_optional,
            supplies_field=observation.supplies_field(change.field_name)
            or (
                bool(change.previous_name)
                and observation.supplies_field(change.previous_name)
            ),
            has_default_authority=has_auth,
        )
        direction = _merge_compatibility(direction, item_dir)
        reasons.extend(item_reasons)
        for kind in item_auth:
            if kind not in authority:
                authority.append(kind)

    # Migration role always needs migration authority for incompatible shapes.
    if observation.role is SchemaConsumerRole.MIGRATION:
        if AuthorityKind.MIGRATION_MANIFEST not in authority:
            if direction is not CompatibilityDirection.FULL:
                authority.append(AuthorityKind.MIGRATION_MANIFEST)
                reasons.append("migration_needs_independent_authority")

    # Strip authority that the observation already carries.
    remaining = tuple(
        kind for kind in authority if not observation.has_authority(kind)
    )
    # Independent authority is required to *discharge* a claim; it never upgrades
    # an incompatible shape to full compatibility by itself.  When authority is
    # still missing and the shape looked fully compatible, hold at UNKNOWN.
    if remaining and direction is CompatibilityDirection.FULL:
        direction = CompatibilityDirection.UNKNOWN
        reasons.append("pending_independent_authority")

    return direction, tuple(sorted(set(reasons))), remaining


def _disposition_for(
    observation: SchemaConsumerObservation,
    compatibility: CompatibilityDirection,
    required_authority: Sequence[AuthorityKind],
    *,
    clause_breaking: bool,
) -> tuple[ConsumerDisposition, WriteMode, tuple[str, ...]]:
    reasons: list[str] = []

    if observation.role in {
        SchemaConsumerRole.DYNAMIC_CODEC,
        SchemaConsumerRole.MISSING_CODEC,
    } or observation.codec_status in {"missing", "dynamic"}:
        return (
            ConsumerDisposition.FRONTIER,
            WriteMode.FRONTIER,
            ("codec_frontier", f"codec_status:{observation.codec_status}"),
        )

    write_mode = WriteMode.DIRECT
    if observation.generated:
        write_mode = WriteMode.REGENERATION
        reasons.append("generated_root_requires_regeneration")
    elif observation.read_only:
        write_mode = WriteMode.EXTERNAL_OBLIGATION
        reasons.append("read_only_root_requires_external_obligation")

    if required_authority:
        reasons.append("independent_authority_required")
        for kind in required_authority:
            reasons.append(f"authority:{kind.value}")

    if compatibility is CompatibilityDirection.FULL and not required_authority:
        if not clause_breaking:
            return ConsumerDisposition.COMPATIBLE, WriteMode.NONE, tuple(reasons) or (
                "fully_compatible",
            )
        # Breaking clauses elsewhere still allow this consumer to be compatible
        # when it already satisfies the shape.
        return (
            ConsumerDisposition.COMPATIBLE,
            WriteMode.NONE,
            tuple(reasons) or ("consumer_satisfies_breaking_clauses",),
        )

    if compatibility in {
        CompatibilityDirection.BACKWARD,
        CompatibilityDirection.FORWARD,
    } and not required_authority:
        # Partial compatibility still needs migration for the other direction
        # when the clause set is breaking overall.
        if clause_breaking:
            if observation.generated:
                return (
                    ConsumerDisposition.MIGRATE,
                    WriteMode.REGENERATION,
                    tuple(reasons) + ("partial_compatibility_requires_migration",),
                )
            if observation.read_only:
                return (
                    ConsumerDisposition.UPSTREAM,
                    WriteMode.EXTERNAL_OBLIGATION,
                    tuple(reasons) + ("partial_compatibility_external",),
                )
            return (
                ConsumerDisposition.ADAPTER,
                write_mode,
                tuple(reasons) + ("partial_compatibility_adapter",),
            )
        return (
            ConsumerDisposition.COMPATIBLE,
            WriteMode.NONE,
            tuple(reasons) + (f"direction:{compatibility.value}",),
        )

    if compatibility is CompatibilityDirection.UNKNOWN:
        if required_authority:
            if observation.generated:
                return (
                    ConsumerDisposition.MIGRATE,
                    WriteMode.REGENERATION,
                    tuple(reasons) + ("unknown_pending_authority",),
                )
            if observation.read_only:
                return (
                    ConsumerDisposition.UPSTREAM,
                    WriteMode.EXTERNAL_OBLIGATION,
                    tuple(reasons) + ("unknown_pending_authority_external",),
                )
            return (
                ConsumerDisposition.MIGRATE,
                write_mode if write_mode is not WriteMode.DIRECT else WriteMode.DIRECT,
                tuple(reasons) + ("unknown_pending_authority",),
            )
        return (
            ConsumerDisposition.FRONTIER,
            WriteMode.FRONTIER,
            tuple(reasons) + ("unknown_compatibility",),
        )

    # INCOMPATIBLE
    if observation.generated:
        return (
            ConsumerDisposition.MIGRATE,
            WriteMode.REGENERATION,
            tuple(reasons) + ("incompatible_generated",),
        )
    if observation.read_only:
        return (
            ConsumerDisposition.UPSTREAM,
            WriteMode.EXTERNAL_OBLIGATION,
            tuple(reasons) + ("incompatible_read_only",),
        )
    if observation.role is SchemaConsumerRole.MIGRATION:
        return (
            ConsumerDisposition.MIGRATE,
            WriteMode.DIRECT,
            tuple(reasons) + ("migration_obligation",),
        )
    return (
        ConsumerDisposition.MIGRATE,
        write_mode,
        tuple(reasons) + ("incompatible_consumer",),
    )


def _default_node(
    observation: SchemaConsumerObservation,
) -> GraphNodeRef:
    if observation.node is not None:
        return observation.node
    kind = "schema"
    if observation.role in {
        SchemaConsumerRole.CONSTRUCTOR,
        SchemaConsumerRole.FACTORY,
        SchemaConsumerRole.BUILDER,
    }:
        kind = observation.role.value
    elif observation.role in {
        SchemaConsumerRole.SERIALIZER,
        SchemaConsumerRole.DESERIALIZER,
    }:
        kind = observation.role.value
    elif observation.generated:
        kind = "generated"
    return GraphNodeRef(
        node_id=f"node:{observation.consumer_id}",
        kind=kind,
        path=observation.path,
        symbol_id=observation.symbol_id,
        artifact_id=f"blob:{observation.consumer_id}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:schema-protocol-impact",
    )


def _build_obligation(
    roots: PropagationAuthorityRoots,
    delta: ProgramContractDelta,
    observation: SchemaConsumerObservation,
    disposition: ConsumerDisposition,
    clause_ids: Sequence[str],
    *,
    missing_fields: Sequence[str] = (),
    write_mode: WriteMode = WriteMode.DIRECT,
) -> ConsumerMigrationObligation | None:
    if disposition in {
        ConsumerDisposition.COMPATIBLE,
        ConsumerDisposition.EXCLUDED,
        ConsumerDisposition.ABSTAIN,
        ConsumerDisposition.REVIEW_ONLY,
    }:
        return None

    delta_id = _identity(
        "program-contract-delta",
        {
            "change_set_id": delta.change_set_id,
            "subject_symbol_id": delta.subject_symbol_id,
            "before": delta.before_contract_ref,
            "after": delta.after_contract_ref,
            "clause_ids": [item.clause_id for item in delta.clauses],
        },
    )
    # Prefer stable delta content_id when available.
    delta_ref = getattr(delta, "content_id", None) or delta_id

    missing_ids = tuple(
        f"missing-field:{name}" for name in sorted(set(missing_fields)) if name
    )
    behavior_ids: tuple[str, ...] = ()
    if write_mode is WriteMode.REGENERATION:
        behavior_ids = (f"behavior:regenerate:{observation.consumer_id}",)
    elif write_mode is WriteMode.EXTERNAL_OBLIGATION:
        behavior_ids = (f"behavior:external:{observation.consumer_id}",)
    elif observation.role is SchemaConsumerRole.MIGRATION:
        behavior_ids = (f"behavior:migration:{observation.consumer_id}",)

    if disposition is ConsumerDisposition.FRONTIER:
        # Frontier obligations are allowed without proof, with FRONTIER disposition.
        pass

    obligation_id = _identity(
        "consumer-migration-obligation",
        {
            "consumer_id": observation.consumer_id,
            "delta_id": delta_ref,
            "disposition": disposition.value,
            "clause_ids": list(clause_ids),
            "path": observation.path,
            "role": observation.role.value,
            "write_mode": write_mode.value,
        },
    )
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=obligation_id,
        consumer_id=observation.consumer_id,
        delta_id=str(delta_ref),
        disposition=disposition,
        clause_ids=tuple(clause_ids) or (delta.clauses[0].clause_id,),
        node=_default_node(observation),
        proof_refs=(),
        missing_input_ids=missing_ids,
        behavior_contract_ids=behavior_ids,
        invalidation_refs=(),
    )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@dataclass
class SchemaProtocolChangeAnalyzer:
    """Analyze schema/constructor/serialization/protocol impacts for one delta."""

    roots: PropagationAuthorityRoots | None = None

    def bind(
        self, *, roots: PropagationAuthorityRoots | None = None
    ) -> "SchemaProtocolChangeAnalyzer":
        if roots is not None:
            self.roots = _roots(roots)
        return self

    def analyze(
        self,
        delta: ProgramContractDelta,
        consumers: Sequence[SchemaConsumerObservation | Mapping[str, Any]] = (),
        *,
        field_changes: Sequence[SchemaFieldChange | Mapping[str, Any]] = (),
        roots: PropagationAuthorityRoots | None = None,
        evidence_refs: Sequence[str] = (),
    ) -> SchemaProtocolImpact:
        bound_roots = _roots(roots or self.roots or delta.roots)
        if bound_roots.to_dict() != delta.roots.to_dict():
            # Require exact root binding when both provided.
            if roots is not None or self.roots is not None:
                raise SchemaProtocolChangeImpactAuthorityError(
                    "analyzer roots must match program contract delta roots"
                )

        if not delta.clauses:
            raise SchemaProtocolChangeImpactError(
                "delta must include at least one clause"
            )

        changes = extract_field_changes(delta, field_changes)
        observations = [
            item
            if isinstance(item, SchemaConsumerObservation)
            else SchemaConsumerObservation.from_dict(item)
            for item in consumers
        ]
        if len(observations) > MAX_ENTRIES:
            raise SchemaProtocolChangeImpactBoundsError(
                "consumers exceeds its item bound"
            )

        # Deduplicate exact consumer_ids (last wins after deterministic sort later).
        by_id: dict[str, SchemaConsumerObservation] = {}
        for observation in observations:
            by_id[observation.consumer_id] = observation
        observations = list(by_id.values())

        schema_clauses = tuple(
            clause
            for clause in delta.clauses
            if clause.kind in _SCHEMA_DELTA_KINDS
            or clause.kind
            in {
                DeltaKind.SCHEMA_CHANGE,
                DeltaKind.SERIALIZATION_CHANGE,
                DeltaKind.PROTOCOL_CHANGE,
                DeltaKind.FIELD_INTRO,
                DeltaKind.FIELD_REMOVE,
                DeltaKind.CONSTRUCTOR_INTRO,
                DeltaKind.CONSTRUCTOR_REMOVE,
                DeltaKind.FACTORY_INTRO,
                DeltaKind.FACTORY_REMOVE,
            }
        )
        if not schema_clauses:
            schema_clauses = tuple(delta.clauses)
        clause_ids = tuple(item.clause_id for item in schema_clauses)
        clause_breaking = any(
            item.disposition is DeltaDisposition.BREAKING for item in schema_clauses
        )

        entries: list[SchemaConsumerImpactEntry] = []
        constructor_impacts: list[ConstructorImpact] = []
        serialization_impacts: list[SerializationImpact] = []
        protocol_impacts: list[ProtocolImpact] = []
        frontier_ids: list[str] = []

        field_names = tuple(sorted({item.field_name for item in changes}))
        added_required = tuple(
            sorted(
                {
                    item.field_name
                    for item in changes
                    if item.kind is FieldChangeKind.ADDED
                    and item.required
                    and not item.has_default
                }
            )
        )
        removed = tuple(
            sorted(
                {
                    item.field_name
                    for item in changes
                    if item.kind
                    in {FieldChangeKind.REMOVED, FieldChangeKind.VARIANT_REMOVED}
                }
            )
        )

        for observation in observations:
            compatibility, reason_codes, required_authority = (
                classify_consumer_compatibility(changes, observation)
            )
            # Unsupported/unknown clauses force frontier for non-codec roles.
            if any(
                item.disposition is DeltaDisposition.UNSUPPORTED
                for item in schema_clauses
            ) and observation.role not in {
                SchemaConsumerRole.DYNAMIC_CODEC,
                SchemaConsumerRole.MISSING_CODEC,
            }:
                compatibility = CompatibilityDirection.UNKNOWN
                reason_codes = tuple(
                    sorted(set(reason_codes) | {"unsupported_clause"})
                )
            if any(
                item.disposition is DeltaDisposition.UNKNOWN for item in schema_clauses
            ) and observation.codec_status == "present":
                if compatibility is CompatibilityDirection.FULL:
                    compatibility = CompatibilityDirection.UNKNOWN
                    reason_codes = tuple(
                        sorted(set(reason_codes) | {"unknown_clause"})
                    )

            disposition, write_mode, disp_reasons = _disposition_for(
                observation,
                compatibility,
                required_authority,
                clause_breaking=clause_breaking,
            )
            all_reasons = tuple(sorted(set(reason_codes) | set(disp_reasons)))

            missing_fields: list[str] = []
            if disposition in {
                ConsumerDisposition.MIGRATE,
                ConsumerDisposition.ADAPTER,
                ConsumerDisposition.UPSTREAM,
            }:
                for change in changes:
                    if change.kind is FieldChangeKind.ADDED and change.required:
                        if not observation.supplies_field(change.field_name):
                            if observation.role in _WRITER_ROLES | {
                                SchemaConsumerRole.GENERATED_CLIENT,
                                SchemaConsumerRole.CONSTRUCTOR,
                                SchemaConsumerRole.FACTORY,
                                SchemaConsumerRole.BUILDER,
                            }:
                                missing_fields.append(change.field_name)
                    if change.kind is FieldChangeKind.RENAMED:
                        if observation.supplies_field(change.previous_name):
                            missing_fields.append(change.field_name)

            obligation = _build_obligation(
                bound_roots,
                delta,
                observation,
                disposition,
                clause_ids,
                missing_fields=missing_fields,
                write_mode=write_mode,
            )

            ctor_impact: ConstructorImpact | None = None
            ser_impact: SerializationImpact | None = None
            proto_impact: ProtocolImpact | None = None

            if observation.role in {
                SchemaConsumerRole.CONSTRUCTOR,
                SchemaConsumerRole.FACTORY,
                SchemaConsumerRole.BUILDER,
            } or observation.construction_kind is not None:
                kind = observation.construction_kind or ConstructionKind.coerce(
                    observation.role.value
                )
                ctor_impact = ConstructorImpact(
                    kind=kind,
                    subject_symbol_id=observation.symbol_id,
                    path=observation.path,
                    affected_field_names=field_names,
                    added_required_fields=added_required,
                    removed_fields=removed,
                    compatibility=compatibility,
                    needs_independent_default_authority=any(
                        item is AuthorityKind.SCHEMA_DEFAULT
                        for item in required_authority
                    )
                    or any(
                        change.has_default and not change.default_has_independent_authority
                        for change in changes
                    ),
                    clause_ids=clause_ids,
                    reason=";".join(all_reasons[:8]),
                )
                constructor_impacts.append(ctor_impact)

            if observation.role in {
                SchemaConsumerRole.SERIALIZER,
                SchemaConsumerRole.DESERIALIZER,
                SchemaConsumerRole.PERSISTENCE,
                SchemaConsumerRole.CACHE_KEY,
                SchemaConsumerRole.EQUALITY_HASH,
                SchemaConsumerRole.VERSION_NEGOTIATION,
                SchemaConsumerRole.MIGRATION,
                SchemaConsumerRole.GENERATED_CLIENT,
                SchemaConsumerRole.DYNAMIC_CODEC,
                SchemaConsumerRole.MISSING_CODEC,
            } or observation.serialization_facet is not None:
                facet = observation.serialization_facet
                if facet is None:
                    facet_map = {
                        SchemaConsumerRole.SERIALIZER: SerializationFacet.SERIALIZER,
                        SchemaConsumerRole.DESERIALIZER: SerializationFacet.DESERIALIZER,
                        SchemaConsumerRole.PERSISTENCE: SerializationFacet.PERSISTENCE,
                        SchemaConsumerRole.CACHE_KEY: SerializationFacet.CACHE_KEY,
                        SchemaConsumerRole.EQUALITY_HASH: SerializationFacet.HASH,
                        SchemaConsumerRole.VERSION_NEGOTIATION: (
                            SerializationFacet.VERSION_NEGOTIATION
                        ),
                        SchemaConsumerRole.MIGRATION: SerializationFacet.MIGRATION,
                        SchemaConsumerRole.GENERATED_CLIENT: (
                            SerializationFacet.GENERATED_CLIENT
                        ),
                        SchemaConsumerRole.DYNAMIC_CODEC: SerializationFacet.SERIALIZER,
                        SchemaConsumerRole.MISSING_CODEC: SerializationFacet.SERIALIZER,
                    }
                    facet = facet_map.get(
                        observation.role, SerializationFacet.SERIALIZER
                    )
                codec_status = observation.codec_status
                ser_write = write_mode
                if codec_status in {"missing", "dynamic"}:
                    ser_write = WriteMode.FRONTIER
                elif observation.generated or facet is SerializationFacet.GENERATED_CLIENT:
                    ser_write = WriteMode.REGENERATION
                ser_impact = SerializationImpact(
                    facet=facet,
                    subject_symbol_id=observation.symbol_id,
                    path=observation.path,
                    affected_field_names=field_names,
                    compatibility=compatibility,
                    codec_status=codec_status,
                    needs_migration_authority=(
                        observation.role is SchemaConsumerRole.MIGRATION
                        or AuthorityKind.MIGRATION_MANIFEST in required_authority
                        or any(
                            change.kind
                            in {
                                FieldChangeKind.REMOVED,
                                FieldChangeKind.RENAMED,
                                FieldChangeKind.RETYPED,
                                FieldChangeKind.VARIANT_REMOVED,
                                FieldChangeKind.VARIANT_RENAMED,
                            }
                            for change in changes
                        )
                        and observation.role
                        in {
                            SchemaConsumerRole.PERSISTENCE,
                            SchemaConsumerRole.MIGRATION,
                        }
                    ),
                    write_mode=ser_write,
                    clause_ids=clause_ids,
                    reason=";".join(all_reasons[:8]),
                )
                serialization_impacts.append(ser_impact)

            if (
                observation.role
                in {
                    SchemaConsumerRole.SCHEMA_SURFACE,
                    SchemaConsumerRole.PROTOCOL_SURFACE,
                }
                or observation.surface is not None
            ):
                surface = observation.surface or SchemaSurfaceKind.JSON
                proto_write = write_mode
                if observation.generated:
                    proto_write = WriteMode.REGENERATION
                elif observation.read_only:
                    proto_write = WriteMode.EXTERNAL_OBLIGATION
                proto_impact = ProtocolImpact(
                    surface=surface,
                    subject_symbol_id=observation.symbol_id,
                    path=observation.path,
                    affected_field_names=field_names,
                    compatibility=compatibility,
                    version_negotiation_required=(
                        observation.role is SchemaConsumerRole.VERSION_NEGOTIATION
                        or any(
                            change.kind
                            in {
                                FieldChangeKind.RENAMED,
                                FieldChangeKind.RETYPED,
                                FieldChangeKind.REMOVED,
                                FieldChangeKind.VARIANT_REMOVED,
                            }
                            for change in changes
                        )
                        or compatibility
                        in {
                            CompatibilityDirection.INCOMPATIBLE,
                            CompatibilityDirection.FORWARD,
                            CompatibilityDirection.BACKWARD,
                        }
                    ),
                    write_mode=proto_write
                    if proto_write is not WriteMode.NONE
                    else WriteMode.DIRECT,
                    clause_ids=clause_ids,
                    reason=";".join(all_reasons[:8]),
                )
                protocol_impacts.append(proto_impact)

            if disposition is ConsumerDisposition.FRONTIER:
                frontier_ids.append(observation.consumer_id)

            entries.append(
                SchemaConsumerImpactEntry(
                    observation=observation,
                    disposition=disposition,
                    compatibility=compatibility,
                    write_mode=write_mode,
                    affected_field_names=field_names,
                    clause_ids=clause_ids,
                    reason_codes=all_reasons,
                    required_authority=required_authority,
                    obligation=obligation,
                    constructor_impact=ctor_impact,
                    serialization_impact=ser_impact,
                    protocol_impact=proto_impact,
                )
            )

        delta_id = getattr(delta, "content_id", None) or _identity(
            "program-contract-delta",
            {
                "change_set_id": delta.change_set_id,
                "subject_symbol_id": delta.subject_symbol_id,
                "clause_ids": [item.clause_id for item in delta.clauses],
            },
        )

        return SchemaProtocolImpact(
            roots=bound_roots,
            delta_id=str(delta_id),
            subject_symbol_id=delta.subject_symbol_id,
            field_changes=changes,
            entries=tuple(entries),
            constructor_impacts=tuple(constructor_impacts),
            serialization_impacts=tuple(serialization_impacts),
            protocol_impacts=tuple(protocol_impacts),
            frontier_consumer_ids=tuple(frontier_ids),
            evidence_refs=_string_tuple(evidence_refs, "evidence_refs"),
        )


def build_schema_protocol_impact(
    delta: ProgramContractDelta,
    consumers: Sequence[SchemaConsumerObservation | Mapping[str, Any]] = (),
    *,
    field_changes: Sequence[SchemaFieldChange | Mapping[str, Any]] = (),
    roots: PropagationAuthorityRoots | None = None,
    evidence_refs: Sequence[str] = (),
) -> SchemaProtocolImpact:
    """Functional façade over :class:`SchemaProtocolChangeAnalyzer`."""
    return SchemaProtocolChangeAnalyzer(roots=roots or delta.roots).analyze(
        delta,
        consumers,
        field_changes=field_changes,
        evidence_refs=evidence_refs,
    )


def required_consumer_roles() -> frozenset[str]:
    """Closed catalogue of consumer roles the analyzer must be able to name."""
    return _REQUIRED_ROLES


def required_schema_surfaces() -> frozenset[str]:
    """Closed catalogue of schema/protocol surfaces."""
    return _REQUIRED_SURFACES


def all_field_change_kinds() -> tuple[FieldChangeKind, ...]:
    return tuple(FieldChangeKind)


def all_serialization_facets() -> tuple[SerializationFacet, ...]:
    return tuple(SerializationFacet)


def all_compatibility_directions() -> tuple[CompatibilityDirection, ...]:
    return tuple(CompatibilityDirection)


__all__ = [
    "AuthorityKind",
    "CompatibilityDirection",
    "CONSTRUCTOR_IMPACT_SCHEMA",
    "ConstructionKind",
    "ConstructorImpact",
    "FieldChangeKind",
    "PROTOCOL_IMPACT_SCHEMA",
    "PRODUCER_ID",
    "ProtocolImpact",
    "SCHEMA_CONSUMER_IMPACT_ENTRY_SCHEMA",
    "SCHEMA_CONSUMER_OBSERVATION_SCHEMA",
    "SCHEMA_FIELD_CHANGE_SCHEMA",
    "SCHEMA_PROTOCOL_CHANGE_IMPACT_VERSION",
    "SCHEMA_PROTOCOL_IMPACT_SCHEMA",
    "SERIALIZATION_IMPACT_SCHEMA",
    "SchemaConsumerImpactEntry",
    "SchemaConsumerObservation",
    "SchemaConsumerRole",
    "SchemaFieldChange",
    "SchemaProtocolChangeAnalyzer",
    "SchemaProtocolChangeImpact",
    "SchemaProtocolChangeImpactAuthorityError",
    "SchemaProtocolChangeImpactBoundsError",
    "SchemaProtocolChangeImpactError",
    "SchemaProtocolImpact",
    "SchemaSurfaceKind",
    "SerializationFacet",
    "SerializationImpact",
    "WriteMode",
    "all_compatibility_directions",
    "all_field_change_kinds",
    "all_serialization_facets",
    "build_schema_protocol_impact",
    "classify_consumer_compatibility",
    "classify_field_compatibility",
    "extract_field_changes",
    "required_consumer_roles",
    "required_schema_surfaces",
]
