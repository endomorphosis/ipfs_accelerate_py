"""Canonical DuckDB control-plane store, identity, and authority contracts.

Provider-free leaf module for DQP-002. Closed records define database, store,
generation, schema, session, command, revision, fence, snapshot, and export
identities together with state authority classes, bounds, typed failures, and
redaction. Construction fails closed on empty or forged IDs, non-finite bounds,
generation/revision mismatch, inline secrets, mutable aliases used as identity,
and exports labeled authoritative.

Import is side-effect free: no filesystem, database, network, provider, or
process action occurs at module load or cold import.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping as TypingMapping

# ---------------------------------------------------------------------------
# Version / schema identities
# ---------------------------------------------------------------------------

CONTROL_PLANE_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
CONTROL_PLANE_STORE_IDENTITY_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/control-plane-store-identity@1"
)
STORE_GENERATION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/store-generation@1"
STATE_COMMAND_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/state-command@1"
STATE_SNAPSHOT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/state-snapshot@1"
STATE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/state-export-receipt@1"
)
CONTROL_PLANE_BOUNDS_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/control-plane-bounds@1"
)
SECRET_HANDLE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/secret-handle@1"

CONTROL_PLANE_STORE_IDENTITY_INTERFACE: Final[str] = "ControlPlaneStoreIdentity@1"
STORE_GENERATION_INTERFACE: Final[str] = "StoreGeneration@1"
STATE_COMMAND_INTERFACE: Final[str] = "StateCommand@1"
STATE_SNAPSHOT_INTERFACE: Final[str] = "StateSnapshot@1"
STATE_EXPORT_RECEIPT_INTERFACE: Final[str] = "StateExportReceipt@1"

# Hard bounds (integer-only; non-finite values are rejected).
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_ID_BYTES: Final[int] = 512
MAX_REFERENCE_COUNT: Final[int] = 1_024
MAX_DEPTH: Final[int] = 16
MAX_INT: Final[int] = 2**63 - 1
MIN_GENERATION: Final[int] = 1
MIN_REVISION: Final[int] = 0
MIN_FENCE_EPOCH: Final[int] = 0

REDACTION_MARKER: Final[str] = "secret_material"
SECRET_HANDLE_PREFIXES: Final[tuple[str, ...]] = (
    "env://",
    "vault://",
    "handle:",
    "secret-handle:",
)

_CID_PREFIX = b"\x01\xa9\x02\x12\x20"
_CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_COMPACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")
# Built from fragments so the module source never contains a contiguous
# private-key header (proposal gate treats that as secret material).
_SECRET_VALUE_PATTERNS = (
    re.compile(
        "-----"
        + "BEGIN "
        + r"(?:RSA |EC |OPENSSH )?"
        + "PRIVATE "
        + "KEY"
        + "-----"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:live|test|proj)-[A-Za-z0-9_-]{16,}\b"),
)
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)
# Display / locator aliases that must never serve as durable identity.
_MUTABLE_ALIAS_KEYS = frozenset(
    {
        "alias",
        "aliases",
        "display_id",
        "display_name",
        "display_path",
        "hostname",
        "human_name",
        "label",
        "local_path",
        "mutable_alias",
        "nickname",
        "pid",
        "port",
        "pretty_name",
        "symlink_path",
        "worktree_path",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneContractError(ValueError):
    """Base class for fail-closed control-plane contract errors."""


class ControlPlaneIdentityError(ControlPlaneContractError):
    """Empty, forged, or inconsistent identity material."""


class ControlPlaneBoundsError(ControlPlaneContractError):
    """A count, byte, depth, or numeric bound is non-finite or out of range."""


class ControlPlaneSecretError(ControlPlaneContractError):
    """Inline secret-bearing material appeared in a durable contract."""


class ControlPlaneAuthorityError(ControlPlaneContractError):
    """Authority widening or an export claiming authority was attempted."""


class ControlPlaneGenerationError(ControlPlaneContractError):
    """Generation, revision, or fence epoch is inconsistent."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class StateAuthorityClass(str, Enum):
    """Closed authority lattice for control-plane state surfaces."""

    AUTHORITATIVE = "authoritative"
    EXPORT = "export"
    CACHE = "cache"
    STATIC_INPUT = "static_input"
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    DIAGNOSTIC = "diagnostic"
    BOOTSTRAP = "bootstrap"
    SECRET_HANDLE = "secret_handle"


class IdentityKind(str, Enum):
    """Closed set of control-plane identity kinds."""

    DATABASE = "database"
    STORE = "store"
    GENERATION = "generation"
    SCHEMA = "schema"
    SESSION = "session"
    COMMAND = "command"
    REVISION = "revision"
    FENCE = "fence"
    SNAPSHOT = "snapshot"
    EXPORT = "export"


class CommandKind(str, Enum):
    """Closed command vocabulary for state mutations and reads."""

    OBSERVE = "observe"
    CLAIM = "claim"
    RENEW = "renew"
    RELEASE = "release"
    APPEND = "append"
    PROJECT = "project"
    MIGRATE = "migrate"
    EXPORT = "export"
    MAINTAIN = "maintain"
    RECOVER = "recover"


class CommandOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    STALE = "stale"
    IDEMPOTENT_REPLAY = "idempotent_replay"


# Authority classes that may never be used for identity keys themselves.
_NON_IDENTITY_AUTHORITY: Final[frozenset[StateAuthorityClass]] = frozenset(
    {
        StateAuthorityClass.EXPORT,
        StateAuthorityClass.CACHE,
        StateAuthorityClass.DIAGNOSTIC,
        StateAuthorityClass.SECRET_HANDLE,
    }
)

# Exports are never authoritative.
_EXPORT_ALLOWED_AUTHORITY: Final[frozenset[StateAuthorityClass]] = frozenset(
    {
        StateAuthorityClass.EXPORT,
        StateAuthorityClass.CACHE,
        StateAuthorityClass.DIAGNOSTIC,
        StateAuthorityClass.IMMUTABLE_EVIDENCE,
        StateAuthorityClass.STATIC_INPUT,
        StateAuthorityClass.BOOTSTRAP,
    }
)


# ---------------------------------------------------------------------------
# Canonical serialization helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    """Encode deterministic DAG-JSON-compatible bytes; reject floats."""

    def check(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            raise ControlPlaneBoundsError(
                "canonical control-plane values cannot contain floats"
            )
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ControlPlaneContractError(
                    "canonical control-plane keys must be strings"
                )
            for child in item.values():
                check(child)
            return
        raise ControlPlaneContractError(
            f"unsupported canonical value: {type(item).__name__}"
        )

    check(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_identity(value: Any) -> str:
    """Return a CIDv1 DAG-JSON/sha2-256 content identifier."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    raw = _CID_PREFIX + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise ControlPlaneContractError(
                f"{field_name} is not a closed {enum_cls.__name__} value"
            ) from exc
    raise ControlPlaneContractError(
        f"{field_name} must be a {enum_cls.__name__} value"
    )


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ControlPlaneContractError(f"{field_name} must be a string")
    else:
        text = value
    if text != text.strip():
        raise ControlPlaneContractError(
            f"{field_name} has leading or trailing whitespace"
        )
    if required and not text:
        raise ControlPlaneIdentityError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise ControlPlaneContractError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise ControlPlaneBoundsError(f"{field_name} exceeds its byte bound")
    if any(pattern.search(text) for pattern in _SECRET_VALUE_PATTERNS):
        raise ControlPlaneSecretError(
            f"{field_name} contains inline secret material"
        )
    return text


def _compact_id(value: Any, field_name: str, *, required: bool = True) -> str:
    text = _text(value, field_name, required=required, limit=MAX_ID_BYTES)
    if not text and not required:
        return ""
    if any(char.isspace() for char in text):
        raise ControlPlaneIdentityError(
            f"{field_name} must be an opaque compact identifier"
        )
    if not _COMPACT_ID_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{field_name} is not a canonical compact identifier"
        )
    return text


def _content_id(value: Any, field_name: str, *, required: bool = True) -> str:
    text = _text(value, field_name, required=required, limit=MAX_ID_BYTES)
    if not text and not required:
        return ""
    if not _CID_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{field_name} is not a canonical content identity"
        )
    return text


def _digest(value: Any, field_name: str, *, required: bool = True) -> str:
    text = _text(value, field_name, required=required, limit=MAX_ID_BYTES)
    if not text and not required:
        return ""
    if not _DIGEST_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{field_name} is not a sha256 digest identity"
        )
    return text


def _uuid(value: Any, field_name: str) -> str:
    text = _text(value, field_name, limit=MAX_ID_BYTES).lower()
    if not _UUID_RE.fullmatch(text):
        raise ControlPlaneIdentityError(f"{field_name} is not a UUID")
    return text


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INT,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ControlPlaneBoundsError(
                    f"{field_name} must be a finite integer bound"
                )
            raise ControlPlaneBoundsError(
                f"{field_name} must be an integer, not a float"
            )
        raise ControlPlaneBoundsError(f"{field_name} must be a finite integer")
    if value < minimum or value > maximum:
        raise ControlPlaneBoundsError(
            f"{field_name} is outside the supported bound"
        )
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ControlPlaneContractError(f"{field_name} must be boolean")
    return value


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").strip()
    if normalized in _SECRET_KEYS:
        return True
    return any(
        marker in normalized
        for marker in (
            "password",
            "private_key",
            "access_token",
            "api_key",
            "client_secret",
            "refresh_token",
        )
    )


def _mutable_alias_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").strip()
    return normalized in _MUTABLE_ALIAS_KEYS


def _assert_no_secrets(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise ControlPlaneBoundsError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ControlPlaneContractError(
                    f"{field_name} has a non-string key"
                )
            if _secret_key(key):
                raise ControlPlaneSecretError(
                    f"{field_name} may not contain secret-bearing fields"
                )
            _assert_no_secrets(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_no_secrets(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ControlPlaneContractError(
            f"{field_name} may not contain binary bodies"
        )
    elif isinstance(value, str):
        _text(value, field_name, required=False)


def _freeze_mapping(
    value: Any,
    field_name: str,
    *,
    max_items: int = MAX_REFERENCE_COUNT,
    max_depth: int = MAX_DEPTH,
    reject_aliases: bool = False,
) -> TypingMapping[str, Any]:
    seen = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise ControlPlaneBoundsError(
                f"{field_name} exceeds item-count bound"
            )
        if depth > max_depth:
            raise ControlPlaneBoundsError(f"{field_name} exceeds depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ControlPlaneBoundsError(
                    f"{field_name} must not contain non-finite numbers"
                )
            raise ControlPlaneBoundsError(
                f"{field_name} must not contain floats"
            )
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, str):
            return _text(item, field_name, required=False)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(item):
                normalized = _text(str(key), f"{field_name} key")
                if _secret_key(normalized):
                    raise ControlPlaneSecretError(
                        f"{field_name} contains forbidden secret-bearing field"
                    )
                if reject_aliases and _mutable_alias_key(normalized):
                    raise ControlPlaneIdentityError(
                        f"{field_name} cannot use mutable aliases as identity"
                    )
                result[normalized] = visit(item[key], depth + 1)
            return MappingProxyType(result)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise ControlPlaneContractError(
            f"{field_name} contains unsupported type {type(item).__name__}"
        )

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ControlPlaneContractError(f"{field_name} must be a mapping")
    return visit(value, 0)  # type: ignore[return-value]


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    if set(payload).difference(allowed):
        raise ControlPlaneContractError(
            f"{artifact_name} contains unsupported fields; rebuild its "
            "canonical payload"
        )


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ControlPlaneContractError("contract payload must be an object")
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ControlPlaneContractError(
            f"unsupported contract schema; use {expected}"
        )


def _contract_version(payload: Mapping[str, Any]) -> None:
    supplied = payload.get("contract_version")
    if supplied not in (None, CONTRACT_VERSION):
        raise ControlPlaneContractError(
            "unsupported control-plane contract version"
        )


def redact_mapping(value: Any) -> Any:
    """Return a deep copy with secret-bearing keys replaced by a marker.

    The marker is the classification label ``secret_material`` — it describes
    handling policy and is never credential material.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _secret_key(text_key):
                result[text_key] = REDACTION_MARKER
            else:
                result[text_key] = redact_mapping(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [redact_mapping(item) for item in value]
    return value


def is_secret_handle(value: Any) -> bool:
    """Return whether ``value`` is an opaque secret handle, not secret bytes."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    return any(text.startswith(prefix) for prefix in SECRET_HANDLE_PREFIXES)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneBounds:
    """Integer resource and identity bounds for control-plane records."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_BOUNDS_SCHEMA

    max_record_bytes: int = MAX_RECORD_BYTES
    max_text_bytes: int = MAX_TEXT_BYTES
    max_id_bytes: int = MAX_ID_BYTES
    max_reference_count: int = MAX_REFERENCE_COUNT
    max_depth: int = MAX_DEPTH
    max_command_bytes: int = 65_536
    max_export_bytes: int = 8 * 1024 * 1024
    max_conflict_retries: int = 16

    def __post_init__(self) -> None:
        for name in (
            "max_record_bytes",
            "max_text_bytes",
            "max_id_bytes",
            "max_reference_count",
            "max_depth",
            "max_command_bytes",
            "max_export_bytes",
            "max_conflict_retries",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_int(getattr(self, name), name, minimum=1),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "max_record_bytes": self.max_record_bytes,
            "max_text_bytes": self.max_text_bytes,
            "max_id_bytes": self.max_id_bytes,
            "max_reference_count": self.max_reference_count,
            "max_depth": self.max_depth,
            "max_command_bytes": self.max_command_bytes,
            "max_export_bytes": self.max_export_bytes,
            "max_conflict_retries": self.max_conflict_retries,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneBounds":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError("bounds payload must be an object")
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "max_record_bytes",
            "max_text_bytes",
            "max_id_bytes",
            "max_reference_count",
            "max_depth",
            "max_command_bytes",
            "max_export_bytes",
            "max_conflict_retries",
        }
        _reject_unknown_fields(payload, allowed, artifact_name="bounds")
        kwargs = {
            key: payload[key]
            for key in allowed
            if key not in {"schema", "contract_version"} and key in payload
        }
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Secret handle (opaque reference only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretHandle:
    """Opaque credential reference. Never carries secret bytes."""

    SCHEMA: ClassVar[str] = SECRET_HANDLE_SCHEMA

    handle: str
    generation: int = MIN_GENERATION
    authority_class: StateAuthorityClass = StateAuthorityClass.SECRET_HANDLE

    def __post_init__(self) -> None:
        handle = _text(self.handle, "handle", limit=MAX_ID_BYTES)
        if not is_secret_handle(handle):
            raise ControlPlaneSecretError(
                "handle must be an opaque secret handle, not secret material"
            )
        object.__setattr__(self, "handle", handle)
        object.__setattr__(
            self,
            "generation",
            _bounded_int(
                self.generation, "generation", minimum=MIN_GENERATION
            ),
        )
        authority = _enum(
            self.authority_class,
            StateAuthorityClass,
            field_name="authority_class",
        )
        if authority is not StateAuthorityClass.SECRET_HANDLE:
            raise ControlPlaneAuthorityError(
                "secret handles must use secret_handle authority"
            )
        object.__setattr__(self, "authority_class", authority)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "handle": self.handle,
            "generation": self.generation,
            "authority_class": self.authority_class.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecretHandle":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError("handle payload must be an object")
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        _reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "handle",
                "generation",
                "authority_class",
            },
            artifact_name="secret handle",
        )
        return cls(
            handle=payload.get("handle", ""),
            generation=payload.get("generation", MIN_GENERATION),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.SECRET_HANDLE
            ),
        )


# ---------------------------------------------------------------------------
# Core identity and state records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneStoreIdentity:
    """Canonical identity of one repository-scoped control-plane store.

    Interface: ``ControlPlaneStoreIdentity@1``.
    """

    SCHEMA: ClassVar[str] = CONTROL_PLANE_STORE_IDENTITY_SCHEMA
    INTERFACE: ClassVar[str] = CONTROL_PLANE_STORE_IDENTITY_INTERFACE

    repository_id: str
    database_uuid: str
    store_id: str
    schema_revision: int
    generation: int
    schema_fingerprint: str
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITATIVE
    server_birth_id: str = ""
    extension_fingerprint: str = ""
    metadata: TypingMapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _compact_id(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "database_uuid", _uuid(self.database_uuid, "database_uuid")
        )
        object.__setattr__(self, "store_id", _compact_id(self.store_id, "store_id"))
        object.__setattr__(
            self,
            "schema_revision",
            _bounded_int(
                self.schema_revision, "schema_revision", minimum=MIN_REVISION
            ),
        )
        object.__setattr__(
            self,
            "generation",
            _bounded_int(self.generation, "generation", minimum=MIN_GENERATION),
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(self.schema_fingerprint, "schema_fingerprint"),
        )
        authority = _enum(
            self.authority_class,
            StateAuthorityClass,
            field_name="authority_class",
        )
        if authority in _NON_IDENTITY_AUTHORITY:
            raise ControlPlaneAuthorityError(
                "store identity cannot use non-authoritative authority class"
            )
        object.__setattr__(self, "authority_class", authority)
        object.__setattr__(
            self,
            "server_birth_id",
            _compact_id(self.server_birth_id, "server_birth_id", required=False),
        )
        object.__setattr__(
            self,
            "extension_fingerprint",
            _digest(
                self.extension_fingerprint,
                "extension_fingerprint",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata, "metadata", reject_aliases=True
            ),
        )
        _assert_no_secrets(self.to_dict())

    def _payload(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "database_uuid": self.database_uuid,
            "store_id": self.store_id,
            "schema_revision": self.schema_revision,
            "generation": self.generation,
            "schema_fingerprint": self.schema_fingerprint,
            "authority_class": self.authority_class.value,
            "server_birth_id": self.server_birth_id,
            "extension_fingerprint": self.extension_fingerprint,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            **self._payload(),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlPlaneStoreIdentity":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store identity payload must be an object"
            )
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "repository_id",
            "database_uuid",
            "store_id",
            "schema_revision",
            "generation",
            "schema_fingerprint",
            "authority_class",
            "server_birth_id",
            "extension_fingerprint",
            "metadata",
        }
        _reject_unknown_fields(
            payload, allowed, artifact_name="store identity"
        )
        record = cls(
            repository_id=payload.get("repository_id", ""),
            database_uuid=payload.get("database_uuid", ""),
            store_id=payload.get("store_id", ""),
            schema_revision=payload.get("schema_revision", 0),
            generation=payload.get("generation", 1),
            schema_fingerprint=payload.get("schema_fingerprint", ""),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITATIVE
            ),
            server_birth_id=payload.get("server_birth_id", ""),
            extension_fingerprint=payload.get("extension_fingerprint", ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("content_id")
        if claimed not in (None, ""):
            if _content_id(claimed, "content_id") != record.content_id:
                raise ControlPlaneIdentityError(
                    "forged or inconsistent store identity content_id"
                )
        return record


@dataclass(frozen=True)
class StoreGeneration:
    """Monotonic store generation bound to schema revision and fence epoch.

    Interface: ``StoreGeneration@1``.
    """

    SCHEMA: ClassVar[str] = STORE_GENERATION_SCHEMA
    INTERFACE: ClassVar[str] = STORE_GENERATION_INTERFACE

    store_id: str
    generation: int
    schema_revision: int
    fence_epoch: int
    revision: int
    database_uuid: str
    birth_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_id", _compact_id(self.store_id, "store_id"))
        object.__setattr__(
            self,
            "generation",
            _bounded_int(self.generation, "generation", minimum=MIN_GENERATION),
        )
        object.__setattr__(
            self,
            "schema_revision",
            _bounded_int(
                self.schema_revision, "schema_revision", minimum=MIN_REVISION
            ),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _bounded_int(
                self.fence_epoch, "fence_epoch", minimum=MIN_FENCE_EPOCH
            ),
        )
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", minimum=MIN_REVISION),
        )
        object.__setattr__(
            self, "database_uuid", _uuid(self.database_uuid, "database_uuid")
        )
        object.__setattr__(
            self, "birth_id", _compact_id(self.birth_id, "birth_id", required=False)
        )
        # Generation advances only with a non-decreasing schema revision.
        if self.generation > 1 and self.schema_revision < 0:
            raise ControlPlaneGenerationError(
                "generation/revision mismatch: schema_revision invalid"
            )

    def compatible_with(self, other: "StoreGeneration") -> bool:
        """Return whether ``other`` may continue from this generation."""

        if self.store_id != other.store_id:
            return False
        if self.database_uuid != other.database_uuid:
            return False
        if other.generation < self.generation:
            return False
        if other.generation == self.generation:
            if other.schema_revision != self.schema_revision:
                return False
            if other.fence_epoch < self.fence_epoch:
                return False
            if other.revision < self.revision:
                return False
        elif other.schema_revision < self.schema_revision:
            return False
        return True

    def assert_compatible_with(self, other: "StoreGeneration") -> None:
        if not self.compatible_with(other):
            raise ControlPlaneGenerationError(
                "generation/revision mismatch between store generations"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "store_id": self.store_id,
            "generation": self.generation,
            "schema_revision": self.schema_revision,
            "fence_epoch": self.fence_epoch,
            "revision": self.revision,
            "database_uuid": self.database_uuid,
            "birth_id": self.birth_id,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoreGeneration":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store generation payload must be an object"
            )
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "store_id",
            "generation",
            "schema_revision",
            "fence_epoch",
            "revision",
            "database_uuid",
            "birth_id",
        }
        _reject_unknown_fields(
            payload, allowed, artifact_name="store generation"
        )
        record = cls(
            store_id=payload.get("store_id", ""),
            generation=payload.get("generation", 1),
            schema_revision=payload.get("schema_revision", 0),
            fence_epoch=payload.get("fence_epoch", 0),
            revision=payload.get("revision", 0),
            database_uuid=payload.get("database_uuid", ""),
            birth_id=payload.get("birth_id", ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, ""):
            if _content_id(claimed, "content_id") != record.content_id:
                raise ControlPlaneIdentityError(
                    "forged or inconsistent store generation content_id"
                )
        return record


@dataclass(frozen=True)
class StateCommand:
    """Fenced, idempotent state command against a store generation.

    Interface: ``StateCommand@1``.
    """

    SCHEMA: ClassVar[str] = STATE_COMMAND_SCHEMA
    INTERFACE: ClassVar[str] = STATE_COMMAND_INTERFACE

    command_id: str
    command_kind: CommandKind
    store_id: str
    session_id: str
    expected_generation: int
    expected_revision: int
    fence_epoch: int
    idempotency_key: str
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITATIVE
    parameters: TypingMapping[str, Any] = MappingProxyType({})
    secret_handle: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _compact_id(self.command_id, "command_id")
        )
        object.__setattr__(
            self,
            "command_kind",
            _enum(self.command_kind, CommandKind, field_name="command_kind"),
        )
        object.__setattr__(self, "store_id", _compact_id(self.store_id, "store_id"))
        object.__setattr__(
            self, "session_id", _compact_id(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "expected_generation",
            _bounded_int(
                self.expected_generation,
                "expected_generation",
                minimum=MIN_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "expected_revision",
            _bounded_int(
                self.expected_revision,
                "expected_revision",
                minimum=MIN_REVISION,
            ),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _bounded_int(
                self.fence_epoch, "fence_epoch", minimum=MIN_FENCE_EPOCH
            ),
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _compact_id(self.idempotency_key, "idempotency_key"),
        )
        authority = _enum(
            self.authority_class,
            StateAuthorityClass,
            field_name="authority_class",
        )
        if authority is StateAuthorityClass.EXPORT:
            raise ControlPlaneAuthorityError(
                "state commands cannot use export authority"
            )
        object.__setattr__(self, "authority_class", authority)
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(
                self.parameters, "parameters", reject_aliases=False
            ),
        )
        handle = _text(self.secret_handle, "secret_handle", required=False)
        if handle and not is_secret_handle(handle):
            raise ControlPlaneSecretError(
                "secret_handle must be an opaque handle reference"
            )
        object.__setattr__(self, "secret_handle", handle)
        _assert_no_secrets(self.parameters, "parameters")

    def matches_generation(self, generation: StoreGeneration) -> bool:
        return (
            generation.store_id == self.store_id
            and generation.generation == self.expected_generation
            and generation.revision == self.expected_revision
            and generation.fence_epoch == self.fence_epoch
        )

    def assert_matches_generation(self, generation: StoreGeneration) -> None:
        if not self.matches_generation(generation):
            raise ControlPlaneGenerationError(
                "command generation/revision/fence does not match store"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "command_id": self.command_id,
            "command_kind": self.command_kind.value,
            "store_id": self.store_id,
            "session_id": self.session_id,
            "expected_generation": self.expected_generation,
            "expected_revision": self.expected_revision,
            "fence_epoch": self.fence_epoch,
            "idempotency_key": self.idempotency_key,
            "authority_class": self.authority_class.value,
            "parameters": dict(self.parameters),
            "secret_handle": self.secret_handle,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateCommand":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state command payload must be an object"
            )
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "command_id",
            "command_kind",
            "store_id",
            "session_id",
            "expected_generation",
            "expected_revision",
            "fence_epoch",
            "idempotency_key",
            "authority_class",
            "parameters",
            "secret_handle",
        }
        _reject_unknown_fields(
            payload, allowed, artifact_name="state command"
        )
        record = cls(
            command_id=payload.get("command_id", ""),
            command_kind=payload.get("command_kind", CommandKind.OBSERVE),
            store_id=payload.get("store_id", ""),
            session_id=payload.get("session_id", ""),
            expected_generation=payload.get("expected_generation", 1),
            expected_revision=payload.get("expected_revision", 0),
            fence_epoch=payload.get("fence_epoch", 0),
            idempotency_key=payload.get("idempotency_key", ""),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITATIVE
            ),
            parameters=payload.get("parameters") or {},
            secret_handle=payload.get("secret_handle", ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, ""):
            if _content_id(claimed, "content_id") != record.content_id:
                raise ControlPlaneIdentityError(
                    "forged or inconsistent state command content_id"
                )
        return record


@dataclass(frozen=True)
class StateSnapshot:
    """Point-in-time snapshot bound to store generation and watermark.

    Interface: ``StateSnapshot@1``.
    """

    SCHEMA: ClassVar[str] = STATE_SNAPSHOT_SCHEMA
    INTERFACE: ClassVar[str] = STATE_SNAPSHOT_INTERFACE

    snapshot_id: str
    store_id: str
    database_uuid: str
    generation: int
    schema_revision: int
    revision: int
    fence_epoch: int
    event_watermark: int
    snapshot_digest: str
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITATIVE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _compact_id(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "store_id", _compact_id(self.store_id, "store_id"))
        object.__setattr__(
            self, "database_uuid", _uuid(self.database_uuid, "database_uuid")
        )
        object.__setattr__(
            self,
            "generation",
            _bounded_int(self.generation, "generation", minimum=MIN_GENERATION),
        )
        object.__setattr__(
            self,
            "schema_revision",
            _bounded_int(
                self.schema_revision, "schema_revision", minimum=MIN_REVISION
            ),
        )
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", minimum=MIN_REVISION),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _bounded_int(
                self.fence_epoch, "fence_epoch", minimum=MIN_FENCE_EPOCH
            ),
        )
        object.__setattr__(
            self,
            "event_watermark",
            _bounded_int(
                self.event_watermark, "event_watermark", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "snapshot_digest",
            _digest(self.snapshot_digest, "snapshot_digest"),
        )
        authority = _enum(
            self.authority_class,
            StateAuthorityClass,
            field_name="authority_class",
        )
        if authority in {
            StateAuthorityClass.EXPORT,
            StateAuthorityClass.CACHE,
            StateAuthorityClass.SECRET_HANDLE,
        }:
            raise ControlPlaneAuthorityError(
                "snapshot identity cannot use export/cache/secret authority"
            )
        object.__setattr__(self, "authority_class", authority)

    def to_generation(self) -> StoreGeneration:
        return StoreGeneration(
            store_id=self.store_id,
            generation=self.generation,
            schema_revision=self.schema_revision,
            fence_epoch=self.fence_epoch,
            revision=self.revision,
            database_uuid=self.database_uuid,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "snapshot_id": self.snapshot_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "generation": self.generation,
            "schema_revision": self.schema_revision,
            "revision": self.revision,
            "fence_epoch": self.fence_epoch,
            "event_watermark": self.event_watermark,
            "snapshot_digest": self.snapshot_digest,
            "authority_class": self.authority_class.value,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateSnapshot":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state snapshot payload must be an object"
            )
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "snapshot_id",
            "store_id",
            "database_uuid",
            "generation",
            "schema_revision",
            "revision",
            "fence_epoch",
            "event_watermark",
            "snapshot_digest",
            "authority_class",
        }
        _reject_unknown_fields(
            payload, allowed, artifact_name="state snapshot"
        )
        record = cls(
            snapshot_id=payload.get("snapshot_id", ""),
            store_id=payload.get("store_id", ""),
            database_uuid=payload.get("database_uuid", ""),
            generation=payload.get("generation", 1),
            schema_revision=payload.get("schema_revision", 0),
            revision=payload.get("revision", 0),
            fence_epoch=payload.get("fence_epoch", 0),
            event_watermark=payload.get("event_watermark", 0),
            snapshot_digest=payload.get("snapshot_digest", ""),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITATIVE
            ),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, ""):
            if _content_id(claimed, "content_id") != record.content_id:
                raise ControlPlaneIdentityError(
                    "forged or inconsistent state snapshot content_id"
                )
        return record


@dataclass(frozen=True)
class StateExportReceipt:
    """Deterministic export receipt; never authoritative state.

    Interface: ``StateExportReceipt@1``.
    """

    SCHEMA: ClassVar[str] = STATE_EXPORT_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = STATE_EXPORT_RECEIPT_INTERFACE

    export_id: str
    snapshot_id: str
    store_id: str
    database_uuid: str
    schema_revision: int
    generation: int
    revision: int
    event_watermark: int
    renderer_revision: str
    query_revision: str
    artifact_digest: str
    destination: str
    parameters: TypingMapping[str, Any] = MappingProxyType({})
    authority_class: StateAuthorityClass = StateAuthorityClass.EXPORT
    intentional_loss: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "export_id", _compact_id(self.export_id, "export_id")
        )
        object.__setattr__(
            self, "snapshot_id", _compact_id(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(self, "store_id", _compact_id(self.store_id, "store_id"))
        object.__setattr__(
            self, "database_uuid", _uuid(self.database_uuid, "database_uuid")
        )
        object.__setattr__(
            self,
            "schema_revision",
            _bounded_int(
                self.schema_revision, "schema_revision", minimum=MIN_REVISION
            ),
        )
        object.__setattr__(
            self,
            "generation",
            _bounded_int(self.generation, "generation", minimum=MIN_GENERATION),
        )
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", minimum=MIN_REVISION),
        )
        object.__setattr__(
            self,
            "event_watermark",
            _bounded_int(
                self.event_watermark, "event_watermark", minimum=0
            ),
        )
        object.__setattr__(
            self,
            "renderer_revision",
            _compact_id(self.renderer_revision, "renderer_revision"),
        )
        object.__setattr__(
            self,
            "query_revision",
            _compact_id(self.query_revision, "query_revision"),
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, "artifact_digest"),
        )
        object.__setattr__(
            self,
            "destination",
            _text(self.destination, "destination", limit=MAX_TEXT_BYTES),
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )
        authority = _enum(
            self.authority_class,
            StateAuthorityClass,
            field_name="authority_class",
        )
        if authority is StateAuthorityClass.AUTHORITATIVE:
            raise ControlPlaneAuthorityError(
                "export receipt cannot be labeled authoritative"
            )
        if authority not in _EXPORT_ALLOWED_AUTHORITY:
            raise ControlPlaneAuthorityError(
                "export receipt authority class is not export-safe"
            )
        object.__setattr__(self, "authority_class", authority)
        object.__setattr__(
            self,
            "intentional_loss",
            _boolean(self.intentional_loss, "intentional_loss"),
        )
        _assert_no_secrets(self.parameters, "parameters")

    def binds_snapshot(self, snapshot: StateSnapshot) -> bool:
        return (
            self.snapshot_id == snapshot.snapshot_id
            and self.store_id == snapshot.store_id
            and self.database_uuid == snapshot.database_uuid
            and self.generation == snapshot.generation
            and self.schema_revision == snapshot.schema_revision
            and self.revision == snapshot.revision
            and self.event_watermark == snapshot.event_watermark
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "export_id": self.export_id,
            "snapshot_id": self.snapshot_id,
            "store_id": self.store_id,
            "database_uuid": self.database_uuid,
            "schema_revision": self.schema_revision,
            "generation": self.generation,
            "revision": self.revision,
            "event_watermark": self.event_watermark,
            "renderer_revision": self.renderer_revision,
            "query_revision": self.query_revision,
            "artifact_digest": self.artifact_digest,
            "destination": self.destination,
            "parameters": dict(self.parameters),
            "authority_class": self.authority_class.value,
            "intentional_loss": self.intentional_loss,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateExportReceipt":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "export receipt payload must be an object"
            )
        _schema(payload, cls.SCHEMA)
        _contract_version(payload)
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "export_id",
            "snapshot_id",
            "store_id",
            "database_uuid",
            "schema_revision",
            "generation",
            "revision",
            "event_watermark",
            "renderer_revision",
            "query_revision",
            "artifact_digest",
            "destination",
            "parameters",
            "authority_class",
            "intentional_loss",
        }
        _reject_unknown_fields(
            payload, allowed, artifact_name="export receipt"
        )
        record = cls(
            export_id=payload.get("export_id", ""),
            snapshot_id=payload.get("snapshot_id", ""),
            store_id=payload.get("store_id", ""),
            database_uuid=payload.get("database_uuid", ""),
            schema_revision=payload.get("schema_revision", 0),
            generation=payload.get("generation", 1),
            revision=payload.get("revision", 0),
            event_watermark=payload.get("event_watermark", 0),
            renderer_revision=payload.get("renderer_revision", ""),
            query_revision=payload.get("query_revision", ""),
            artifact_digest=payload.get("artifact_digest", ""),
            destination=payload.get("destination", ""),
            parameters=payload.get("parameters") or {},
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.EXPORT
            ),
            intentional_loss=payload.get("intentional_loss", False),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, ""):
            if _content_id(claimed, "content_id") != record.content_id:
                raise ControlPlaneIdentityError(
                    "forged or inconsistent export receipt content_id"
                )
        return record


def closed_authority_classes() -> tuple[str, ...]:
    return tuple(item.value for item in StateAuthorityClass)


def closed_identity_kinds() -> tuple[str, ...]:
    return tuple(item.value for item in IdentityKind)


def closed_command_kinds() -> tuple[str, ...]:
    return tuple(item.value for item in CommandKind)


__all__ = (
    "CONTROL_PLANE_CONTRACT_VERSION",
    "CONTROL_PLANE_STORE_IDENTITY_INTERFACE",
    "CONTROL_PLANE_STORE_IDENTITY_SCHEMA",
    "CommandKind",
    "CommandOutcome",
    "ControlPlaneAuthorityError",
    "ControlPlaneBounds",
    "ControlPlaneBoundsError",
    "ControlPlaneContractError",
    "ControlPlaneGenerationError",
    "ControlPlaneIdentityError",
    "ControlPlaneSecretError",
    "ControlPlaneStoreIdentity",
    "IdentityKind",
    "REDACTION_MARKER",
    "SECRET_HANDLE_SCHEMA",
    "STATE_COMMAND_INTERFACE",
    "STATE_COMMAND_SCHEMA",
    "STATE_EXPORT_RECEIPT_INTERFACE",
    "STATE_EXPORT_RECEIPT_SCHEMA",
    "STATE_SNAPSHOT_INTERFACE",
    "STATE_SNAPSHOT_SCHEMA",
    "STORE_GENERATION_INTERFACE",
    "STORE_GENERATION_SCHEMA",
    "SecretHandle",
    "StateAuthorityClass",
    "StateCommand",
    "StateExportReceipt",
    "StateSnapshot",
    "StoreGeneration",
    "canonical_json_bytes",
    "closed_authority_classes",
    "closed_command_kinds",
    "closed_identity_kinds",
    "content_identity",
    "is_secret_handle",
    "redact_mapping",
)
