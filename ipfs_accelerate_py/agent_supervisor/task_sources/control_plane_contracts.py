"""Canonical store, schema, identity, and authority contracts for the control plane.

This module is deliberately provider-free and side-effect free.  Importing it
must not open a filesystem path, database, network socket, provider client, or
subprocess.  Records are closed, frozen, content-addressed, and fail closed on
empty, forged, or inconsistent identities; non-finite bounds; generation or
revision mismatch; secret material; mutable aliases used as identity; and any
export labeled authoritative.

Interfaces (v1):

* ``ControlPlaneStoreIdentity``
* ``StoreGeneration``
* ``StateCommand``
* ``StateSnapshot``
* ``StateExportReceipt``
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTROL_PLANE_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION

CONTROL_PLANE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-bounds@1"
)
CONTROL_PLANE_STORE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-store-identity@1"
)
STORE_GENERATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/store-generation@1"
)
SCHEMA_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/schema-identity@1"
)
SESSION_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/session-identity@1"
)
FENCE_TOKEN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fence-token@1"
)
STATE_REVISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-revision@1"
)
STATE_COMMAND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-command@1"
)
STATE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-snapshot@1"
)
STATE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-export-receipt@1"
)

ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_ITEMS: Final[int] = 4_096
ABSOLUTE_MAX_DEPTH: Final[int] = 32
ABSOLUTE_MAX_SERIALIZED_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_GENERATION: Final[int] = 2**63 - 1
ABSOLUTE_MAX_REVISION: Final[int] = 2**63 - 1
ABSOLUTE_MAX_FENCING_EPOCH: Final[int] = 2**63 - 1
ABSOLUTE_MAX_WATERMARK: Final[int] = 2**63 - 1
ABSOLUTE_MAX_TIMESTAMP_MS: Final[int] = 2**63 - 1
MAX_REDACTION_MARK: Final[str] = "[REDACTED]"

_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")

# Mutable aliases are useful for display and routing, but never identity.
_MUTABLE_ALIAS_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "alias",
        "aliases",
        "board_namespace",
        "display_id",
        "display_name",
        "display_path",
        "hostname",
        "human_label",
        "label",
        "mutable_alias",
        "name",
        "nickname",
        "path",
        "pid",
        "short_name",
        "symlink",
        "title",
    }
)

_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "quack_password",
        "quack_token",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)

_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneContractError(ContractValidationError):
    """Base error for malformed control-plane contracts."""


class ControlPlaneBoundsError(ControlPlaneContractError):
    """A count, byte, depth, generation, or integer bound is non-finite."""


class ControlPlaneIdentityError(ControlPlaneContractError):
    """An identity is empty, forged, inconsistent, or alias-based."""


class ControlPlaneSecretError(ControlPlaneContractError):
    """A durable contract contains secret-bearing material."""


class ControlPlaneGenerationError(ControlPlaneContractError):
    """Store generation and revision expectations do not match."""


class ControlPlaneAuthorityError(ControlPlaneContractError):
    """An authority class or export authority claim is illegal."""


class ControlPlaneCompatibilityError(ControlPlaneContractError):
    """Schema or contract version is unsupported."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class StateAuthorityClass(str, Enum):
    """Classification of a state sink relative to control-plane authority.

    Only ``AUTHORITY`` may drive orchestration decisions.  Exports, caches,
    OS bootstrap handles, and diagnostics are never authoritative.
    """

    AUTHORITY = "authority"
    STATIC_INPUT = "static_input"
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    CACHE = "cache"
    EXPORT = "export"
    OS_BOOTSTRAP = "os_bootstrap"
    EMERGENCY_DIAGNOSTIC = "emergency_diagnostic"

    @property
    def may_authorize_decisions(self) -> bool:
        return self is StateAuthorityClass.AUTHORITY

    @property
    def is_export_or_projection(self) -> bool:
        return self in {
            StateAuthorityClass.EXPORT,
            StateAuthorityClass.CACHE,
            StateAuthorityClass.OS_BOOTSTRAP,
            StateAuthorityClass.EMERGENCY_DIAGNOSTIC,
        }


class StateCommandKind(str, Enum):
    """Closed command vocabulary for control-plane state operations."""

    READ = "read"
    MUTATE = "mutate"
    CLAIM = "claim"
    RENEW = "renew"
    RELEASE = "release"
    MIGRATE = "migrate"
    SNAPSHOT = "snapshot"
    EXPORT = "export"
    IMPORT = "import"
    MAINTENANCE = "maintenance"

    @property
    def requires_fence(self) -> bool:
        return self in {
            StateCommandKind.MUTATE,
            StateCommandKind.CLAIM,
            StateCommandKind.RENEW,
            StateCommandKind.RELEASE,
            StateCommandKind.MIGRATE,
            StateCommandKind.IMPORT,
            StateCommandKind.MAINTENANCE,
        }

    @property
    def requires_idempotency(self) -> bool:
        return self.requires_fence


class ExportProfile(str, Enum):
    """Supported deterministic export render profiles."""

    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    PORTABLE_BUNDLE = "portable_bundle"


class ExportFidelity(str, Enum):
    """Whether an export is lossless for re-import."""

    LOSSLESS = "lossless"
    INTENTIONALLY_LOSSY = "intentionally_lossy"


class RedactionPolicy(str, Enum):
    """How secret-bearing fields are handled in public projections."""

    REJECT = "reject"
    REDACT = "redact"
    OMIT = "omit"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({str(item.value) for item in enum_type}))
        raise ControlPlaneContractError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ControlPlaneContractError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ControlPlaneIdentityError(f"{name} must not be empty")
    if "\x00" in text:
        raise ControlPlaneContractError(f"{name} must not contain NUL")
    if len(text.encode("utf-8")) > max_bytes:
        raise ControlPlaneBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return text


def _finite_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Accept only true finite integers; reject bool, float, NaN, and inf."""

    if isinstance(value, bool) or not isinstance(value, int):
        # Explicitly reject floats (including inf/nan) and numeric strings.
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ControlPlaneBoundsError(
                    f"{name} must be a finite integer bound"
                )
            raise ControlPlaneBoundsError(
                f"{name} must be a finite integer, not a float"
            )
        raise ControlPlaneBoundsError(f"{name} must be a finite integer")
    if value < minimum:
        raise ControlPlaneBoundsError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ControlPlaneBoundsError(f"{name} exceeds its absolute limit")
    return value


def _optional_text(value: Any, name: str, *, max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES) -> str:
    return _text(value, name, required=False, max_bytes=max_bytes)


def _digest(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required)
    if not text:
        return ""
    if not _DIGEST_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{name} must be a sha256:<64-hex> digest"
        )
    return text


def _uuid(value: Any, name: str) -> str:
    text = _text(value, name, required=True).casefold()
    if not _UUID_RE.fullmatch(text):
        raise ControlPlaneIdentityError(f"{name} must be a UUID")
    return text


def _cid_or_digest(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required)
    if not text:
        return ""
    if _DIGEST_RE.fullmatch(text) or _CID_RE.fullmatch(text):
        return text
    raise ControlPlaneIdentityError(
        f"{name} must be a content CID or sha256 digest"
    )


def _normalize_secret_key(key: str) -> str:
    return key.strip().casefold().replace("-", "_").replace(" ", "_")


def _is_secret_key(key: str) -> bool:
    normalized = _normalize_secret_key(key)
    if normalized in _SECRET_KEYS:
        return True
    return any(
        normalized == marker
        or normalized.endswith("_" + marker)
        or marker in normalized.split("_")
        for marker in _SECRET_KEYS
    )


def _looks_like_secret_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)


def _reject_secrets(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_secret_key(key):
                raise ControlPlaneSecretError(
                    f"{path}.{key} contains secret-bearing material"
                )
            _reject_secrets(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secrets(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _looks_like_secret_value(value):
        raise ControlPlaneSecretError(
            f"{path} contains secret-bearing material"
        )


def redact_secrets(value: Any) -> Any:
    """Return a deep copy with secret keys/values replaced by a redaction mark.

    This helper is pure and never writes to disk or contacts a provider.
    """

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_secret_key(key):
                redacted[key] = MAX_REDACTION_MARK
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, (list, tuple)):
        items = [redact_secrets(item) for item in value]
        return type(value)(items) if not isinstance(value, list) else items
    if isinstance(value, str) and _looks_like_secret_value(value):
        return MAX_REDACTION_MARK
    return value


def _reject_mutable_alias_identity(payload: Mapping[str, Any], *, context: str) -> None:
    """Reject payloads that attempt to use mutable aliases as sole identity."""

    for key in payload:
        normalized = _normalize_secret_key(str(key))
        if normalized in _MUTABLE_ALIAS_IDENTITY_KEYS:
            raise ControlPlaneIdentityError(
                f"{context} cannot use mutable alias {key!r} as identity"
            )


def _freeze_mapping(
    value: Any,
    *,
    name: str,
    max_depth: int = ABSOLUTE_MAX_DEPTH,
    max_items: int = ABSOLUTE_MAX_ITEMS,
    max_text_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
    allow_secrets: bool = False,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ControlPlaneContractError(f"{name} must be a mapping")
    if not allow_secrets:
        _reject_secrets(value, path=name)

    seen = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise ControlPlaneBoundsError(f"{name} exceeds its item-count limit")
        if depth > max_depth:
            raise ControlPlaneBoundsError(f"{name} exceeds its nesting-depth limit")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise ControlPlaneBoundsError(
                f"{name} must not contain non-finite or floating bounds"
            )
        if isinstance(item, str):
            return _text(item, name, required=False, max_bytes=max_text_bytes)
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ControlPlaneContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            for key in sorted(item):
                normalized_key = _text(key, f"{name} key", max_bytes=max_text_bytes)
                frozen[normalized_key] = visit(item[key], depth + 1)
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1) for member in item)
        raise ControlPlaneContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ControlPlaneContractError("control-plane contract payload must be an object")
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ControlPlaneCompatibilityError(
            f"unsupported control-plane schema {supplied!r}; expected {expected}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, CONTROL_PLANE_CONTRACT_VERSION):
        raise ControlPlaneCompatibilityError(
            "unsupported control-plane contract version"
        )


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str], noun: str) -> None:
    unknown = set(payload) - set(allowed) - {"schema", "contract_version", "schema_version", "content_id", "cid"}
    if unknown:
        raise ControlPlaneContractError(
            f"{noun} contains unsupported fields: {', '.join(sorted(unknown))}"
        )


def _check_claimed_identity(
    payload: Mapping[str, Any],
    computed: str,
    *,
    field_names: Sequence[str] = ("content_id", "cid", "identity"),
) -> None:
    for field_name in field_names:
        claimed = payload.get(field_name)
        if claimed in (None, ""):
            continue
        claimed_text = _text(claimed, field_name, required=True)
        if claimed_text != computed:
            raise ControlPlaneIdentityError(
                f"forged or inconsistent {field_name}; rebuild from canonical payload"
            )


def _identity_material(*parts: str) -> str:
    payload = {"parts": list(parts), "v": CONTROL_PLANE_CONTRACT_VERSION}
    return content_identity(payload)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneBounds(CanonicalContract):
    """Hard integer bounds for control-plane payloads and operations."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_BOUNDS_SCHEMA

    max_serialized_bytes: int = ABSOLUTE_MAX_SERIALIZED_BYTES
    max_items: int = ABSOLUTE_MAX_ITEMS
    max_depth: int = ABSOLUTE_MAX_DEPTH
    max_text_bytes: int = ABSOLUTE_MAX_TEXT_BYTES
    max_generation: int = ABSOLUTE_MAX_GENERATION
    max_revision: int = ABSOLUTE_MAX_REVISION
    max_fencing_epoch: int = ABSOLUTE_MAX_FENCING_EPOCH

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_serialized_bytes",
            _finite_int(
                self.max_serialized_bytes,
                "max_serialized_bytes",
                minimum=1,
                maximum=ABSOLUTE_MAX_SERIALIZED_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_items",
            _finite_int(
                self.max_items, "max_items", minimum=1, maximum=ABSOLUTE_MAX_ITEMS
            ),
        )
        object.__setattr__(
            self,
            "max_depth",
            _finite_int(
                self.max_depth, "max_depth", minimum=1, maximum=ABSOLUTE_MAX_DEPTH
            ),
        )
        object.__setattr__(
            self,
            "max_text_bytes",
            _finite_int(
                self.max_text_bytes,
                "max_text_bytes",
                minimum=1,
                maximum=ABSOLUTE_MAX_TEXT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_generation",
            _finite_int(
                self.max_generation,
                "max_generation",
                minimum=1,
                maximum=ABSOLUTE_MAX_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "max_revision",
            _finite_int(
                self.max_revision,
                "max_revision",
                minimum=1,
                maximum=ABSOLUTE_MAX_REVISION,
            ),
        )
        object.__setattr__(
            self,
            "max_fencing_epoch",
            _finite_int(
                self.max_fencing_epoch,
                "max_fencing_epoch",
                minimum=1,
                maximum=ABSOLUTE_MAX_FENCING_EPOCH,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "max_depth": self.max_depth,
            "max_fencing_epoch": self.max_fencing_epoch,
            "max_generation": self.max_generation,
            "max_items": self.max_items,
            "max_revision": self.max_revision,
            "max_serialized_bytes": self.max_serialized_bytes,
            "max_text_bytes": self.max_text_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneBounds":
        _schema(payload, CONTROL_PLANE_BOUNDS_SCHEMA)
        _reject_unknown(
            payload,
            {
                "max_serialized_bytes",
                "max_items",
                "max_depth",
                "max_text_bytes",
                "max_generation",
                "max_revision",
                "max_fencing_epoch",
            },
            "control-plane bounds",
        )
        instance = cls(
            max_serialized_bytes=payload.get(
                "max_serialized_bytes", ABSOLUTE_MAX_SERIALIZED_BYTES
            ),
            max_items=payload.get("max_items", ABSOLUTE_MAX_ITEMS),
            max_depth=payload.get("max_depth", ABSOLUTE_MAX_DEPTH),
            max_text_bytes=payload.get("max_text_bytes", ABSOLUTE_MAX_TEXT_BYTES),
            max_generation=payload.get("max_generation", ABSOLUTE_MAX_GENERATION),
            max_revision=payload.get("max_revision", ABSOLUTE_MAX_REVISION),
            max_fencing_epoch=payload.get(
                "max_fencing_epoch", ABSOLUTE_MAX_FENCING_EPOCH
            ),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class SchemaIdentity(CanonicalContract):
    """Checksum-bound schema identity for the control plane catalog."""

    SCHEMA: ClassVar[str] = SCHEMA_IDENTITY_SCHEMA

    schema_revision: int
    migration_catalog_digest: str
    schema_fingerprint: str = ""
    minimum_application_version: str = "1"
    maximum_application_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_revision",
            _finite_int(
                self.schema_revision,
                "schema_revision",
                minimum=1,
                maximum=ABSOLUTE_MAX_REVISION,
            ),
        )
        object.__setattr__(
            self,
            "migration_catalog_digest",
            _digest(self.migration_catalog_digest, "migration_catalog_digest"),
        )
        fingerprint = _optional_text(self.schema_fingerprint, "schema_fingerprint")
        if not fingerprint:
            fingerprint = _digest(
                "sha256:"
                + hashlib.sha256(
                    f"{self.schema_revision}:{self.migration_catalog_digest}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "schema_fingerprint",
            )
        else:
            fingerprint = _digest(fingerprint, "schema_fingerprint")
        object.__setattr__(self, "schema_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "minimum_application_version",
            _text(self.minimum_application_version, "minimum_application_version"),
        )
        object.__setattr__(
            self,
            "maximum_application_version",
            _optional_text(
                self.maximum_application_version, "maximum_application_version"
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "maximum_application_version": self.maximum_application_version,
            "migration_catalog_digest": self.migration_catalog_digest,
            "minimum_application_version": self.minimum_application_version,
            "schema_fingerprint": self.schema_fingerprint,
            "schema_revision": self.schema_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaIdentity":
        _schema(payload, SCHEMA_IDENTITY_SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema_revision",
                "migration_catalog_digest",
                "schema_fingerprint",
                "minimum_application_version",
                "maximum_application_version",
            },
            "schema identity",
        )
        instance = cls(
            schema_revision=payload["schema_revision"],
            migration_catalog_digest=payload["migration_catalog_digest"],
            schema_fingerprint=payload.get("schema_fingerprint", ""),
            minimum_application_version=payload.get(
                "minimum_application_version", "1"
            ),
            maximum_application_version=payload.get(
                "maximum_application_version", ""
            ),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class ControlPlaneStoreIdentity(CanonicalContract):
    """Stable identity of one repository-scoped control-plane store.

    Identity is bound to repository content identity, a store UUID, and schema
    revision.  Display aliases, paths, hostnames, and PIDs are never identity.
    """

    SCHEMA: ClassVar[str] = CONTROL_PLANE_STORE_IDENTITY_SCHEMA

    repository_id: str
    database_uuid: str
    schema_revision: int
    store_namespace: str = "default"
    schema_fingerprint: str = ""
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _cid_or_digest(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "database_uuid", _uuid(self.database_uuid, "database_uuid")
        )
        object.__setattr__(
            self,
            "schema_revision",
            _finite_int(
                self.schema_revision,
                "schema_revision",
                minimum=1,
                maximum=ABSOLUTE_MAX_REVISION,
            ),
        )
        namespace = _text(self.store_namespace, "store_namespace")
        # Namespace is a stable logical partition, not a mutable display alias.
        if _normalize_secret_key(namespace) in _MUTABLE_ALIAS_IDENTITY_KEYS:
            raise ControlPlaneIdentityError(
                "store_namespace cannot be a mutable alias identity key"
            )
        object.__setattr__(self, "store_namespace", namespace)
        fingerprint = _optional_text(self.schema_fingerprint, "schema_fingerprint")
        if fingerprint:
            fingerprint = _digest(fingerprint, "schema_fingerprint")
        object.__setattr__(self, "schema_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "authority_class",
            _enum(self.authority_class, StateAuthorityClass, "authority_class"),
        )
        if self.authority_class is not StateAuthorityClass.AUTHORITY:
            raise ControlPlaneAuthorityError(
                "control-plane store identity authority_class must be authority"
            )

    @property
    def store_id(self) -> str:
        return _identity_material(
            self.repository_id,
            self.database_uuid,
            str(self.schema_revision),
            self.store_namespace,
            self.schema_fingerprint,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "authority_class": self.authority_class.value,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "database_uuid": self.database_uuid,
            "repository_id": self.repository_id,
            "schema_fingerprint": self.schema_fingerprint,
            "schema_revision": self.schema_revision,
            "store_id": self.store_id,
            "store_namespace": self.store_namespace,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneStoreIdentity":
        _schema(payload, CONTROL_PLANE_STORE_IDENTITY_SCHEMA)
        _reject_mutable_alias_identity(payload, context="store identity")
        _reject_unknown(
            payload,
            {
                "repository_id",
                "database_uuid",
                "schema_revision",
                "store_namespace",
                "schema_fingerprint",
                "authority_class",
                "store_id",
            },
            "store identity",
        )
        instance = cls(
            repository_id=payload["repository_id"],
            database_uuid=payload["database_uuid"],
            schema_revision=payload["schema_revision"],
            store_namespace=payload.get("store_namespace", "default"),
            schema_fingerprint=payload.get("schema_fingerprint", ""),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITY
            ),
        )
        claimed_store_id = payload.get("store_id")
        if claimed_store_id not in (None, "") and claimed_store_id != instance.store_id:
            raise ControlPlaneIdentityError(
                "forged or inconsistent store_id; rebuild from canonical payload"
            )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class StoreGeneration(CanonicalContract):
    """Monotonic generation of a control-plane store after restore/rotate."""

    SCHEMA: ClassVar[str] = STORE_GENERATION_SCHEMA

    store: ControlPlaneStoreIdentity
    generation: int
    schema_revision: int
    opened_at_ms: int = 0
    parent_generation: int = 0
    generation_reason: str = "bootstrap"

    def __post_init__(self) -> None:
        if not isinstance(self.store, ControlPlaneStoreIdentity):
            if isinstance(self.store, Mapping):
                object.__setattr__(
                    self, "store", ControlPlaneStoreIdentity.from_dict(self.store)
                )
            else:
                raise ControlPlaneContractError(
                    "store must be a ControlPlaneStoreIdentity"
                )
        object.__setattr__(
            self,
            "generation",
            _finite_int(
                self.generation,
                "generation",
                minimum=1,
                maximum=ABSOLUTE_MAX_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "schema_revision",
            _finite_int(
                self.schema_revision,
                "schema_revision",
                minimum=1,
                maximum=ABSOLUTE_MAX_REVISION,
            ),
        )
        if self.schema_revision != self.store.schema_revision:
            raise ControlPlaneGenerationError(
                "generation schema_revision must match store schema_revision"
            )
        object.__setattr__(
            self,
            "opened_at_ms",
            _finite_int(
                self.opened_at_ms,
                "opened_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )
        object.__setattr__(
            self,
            "parent_generation",
            _finite_int(
                self.parent_generation,
                "parent_generation",
                minimum=0,
                maximum=ABSOLUTE_MAX_GENERATION,
            ),
        )
        if self.parent_generation >= self.generation and self.parent_generation != 0:
            raise ControlPlaneGenerationError(
                "parent_generation must be less than generation"
            )
        object.__setattr__(
            self,
            "generation_reason",
            _text(self.generation_reason, "generation_reason"),
        )

    @property
    def generation_id(self) -> str:
        return _identity_material(
            self.store.store_id,
            str(self.generation),
            str(self.schema_revision),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation,
            "generation_id": self.generation_id,
            "generation_reason": self.generation_reason,
            "opened_at_ms": self.opened_at_ms,
            "parent_generation": self.parent_generation,
            "schema_revision": self.schema_revision,
            "store": self.store.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoreGeneration":
        _schema(payload, STORE_GENERATION_SCHEMA)
        _reject_mutable_alias_identity(payload, context="store generation")
        _reject_unknown(
            payload,
            {
                "store",
                "generation",
                "schema_revision",
                "opened_at_ms",
                "parent_generation",
                "generation_reason",
                "generation_id",
            },
            "store generation",
        )
        instance = cls(
            store=payload["store"],
            generation=payload["generation"],
            schema_revision=payload["schema_revision"],
            opened_at_ms=payload.get("opened_at_ms", 0),
            parent_generation=payload.get("parent_generation", 0),
            generation_reason=payload.get("generation_reason", "bootstrap"),
        )
        claimed = payload.get("generation_id")
        if claimed not in (None, "") and claimed != instance.generation_id:
            raise ControlPlaneIdentityError(
                "forged or inconsistent generation_id; rebuild from canonical payload"
            )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class SessionIdentity(CanonicalContract):
    """Process-birth-bound client or server session identity."""

    SCHEMA: ClassVar[str] = SESSION_IDENTITY_SCHEMA

    session_id: str
    process_birth_id: str
    store_generation: StoreGeneration
    fencing_epoch: int
    opened_at_ms: int = 0
    principal_id: str = "principal:anonymous"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", _cid_or_digest(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id, "process_birth_id"),
        )
        # process_birth_id must not be a bare numeric PID alias.
        if self.process_birth_id.isdigit():
            raise ControlPlaneIdentityError(
                "process_birth_id cannot be a mutable PID alias"
            )
        if not isinstance(self.store_generation, StoreGeneration):
            if isinstance(self.store_generation, Mapping):
                object.__setattr__(
                    self,
                    "store_generation",
                    StoreGeneration.from_dict(self.store_generation),
                )
            else:
                raise ControlPlaneContractError(
                    "store_generation must be a StoreGeneration"
                )
        object.__setattr__(
            self,
            "fencing_epoch",
            _finite_int(
                self.fencing_epoch,
                "fencing_epoch",
                minimum=1,
                maximum=ABSOLUTE_MAX_FENCING_EPOCH,
            ),
        )
        object.__setattr__(
            self,
            "opened_at_ms",
            _finite_int(
                self.opened_at_ms,
                "opened_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )
        object.__setattr__(
            self, "principal_id", _text(self.principal_id, "principal_id")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "fencing_epoch": self.fencing_epoch,
            "opened_at_ms": self.opened_at_ms,
            "principal_id": self.principal_id,
            "process_birth_id": self.process_birth_id,
            "session_id": self.session_id,
            "store_generation": self.store_generation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionIdentity":
        _schema(payload, SESSION_IDENTITY_SCHEMA)
        _reject_mutable_alias_identity(payload, context="session identity")
        _reject_unknown(
            payload,
            {
                "session_id",
                "process_birth_id",
                "store_generation",
                "fencing_epoch",
                "opened_at_ms",
                "principal_id",
            },
            "session identity",
        )
        instance = cls(
            session_id=payload["session_id"],
            process_birth_id=payload["process_birth_id"],
            store_generation=payload["store_generation"],
            fencing_epoch=payload["fencing_epoch"],
            opened_at_ms=payload.get("opened_at_ms", 0),
            principal_id=payload.get("principal_id", "principal:anonymous"),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class FenceToken(CanonicalContract):
    """Fencing token that prevents stale sessions from writing."""

    SCHEMA: ClassVar[str] = FENCE_TOKEN_SCHEMA

    lease_id: str
    session_id: str
    fencing_epoch: int
    scope: str = "store"
    expires_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(
            self, "session_id", _cid_or_digest(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _finite_int(
                self.fencing_epoch,
                "fencing_epoch",
                minimum=1,
                maximum=ABSOLUTE_MAX_FENCING_EPOCH,
            ),
        )
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self,
            "expires_at_ms",
            _finite_int(
                self.expires_at_ms,
                "expires_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "expires_at_ms": self.expires_at_ms,
            "fencing_epoch": self.fencing_epoch,
            "lease_id": self.lease_id,
            "scope": self.scope,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FenceToken":
        _schema(payload, FENCE_TOKEN_SCHEMA)
        _reject_unknown(
            payload,
            {
                "lease_id",
                "session_id",
                "fencing_epoch",
                "scope",
                "expires_at_ms",
            },
            "fence token",
        )
        instance = cls(
            lease_id=payload["lease_id"],
            session_id=payload["session_id"],
            fencing_epoch=payload["fencing_epoch"],
            scope=payload.get("scope", "store"),
            expires_at_ms=payload.get("expires_at_ms", 0),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class StateRevision(CanonicalContract):
    """Compare-and-swap revision bound to a store generation."""

    SCHEMA: ClassVar[str] = STATE_REVISION_SCHEMA

    generation: int
    revision: int
    stream: str = "default"
    watermark: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation",
            _finite_int(
                self.generation,
                "generation",
                minimum=1,
                maximum=ABSOLUTE_MAX_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "revision",
            _finite_int(
                self.revision,
                "revision",
                minimum=0,
                maximum=ABSOLUTE_MAX_REVISION,
            ),
        )
        object.__setattr__(self, "stream", _text(self.stream, "stream"))
        object.__setattr__(
            self,
            "watermark",
            _finite_int(
                self.watermark,
                "watermark",
                minimum=0,
                maximum=ABSOLUTE_MAX_WATERMARK,
            ),
        )

    def matches_generation(self, generation: StoreGeneration | int) -> bool:
        expected = (
            generation.generation
            if isinstance(generation, StoreGeneration)
            else int(generation)
        )
        return self.generation == expected

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation,
            "revision": self.revision,
            "stream": self.stream,
            "watermark": self.watermark,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateRevision":
        _schema(payload, STATE_REVISION_SCHEMA)
        _reject_unknown(
            payload,
            {"generation", "revision", "stream", "watermark"},
            "state revision",
        )
        instance = cls(
            generation=payload["generation"],
            revision=payload["revision"],
            stream=payload.get("stream", "default"),
            watermark=payload.get("watermark", 0),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class StateCommand(CanonicalContract):
    """Typed, fenced, idempotent command against a store generation."""

    SCHEMA: ClassVar[str] = STATE_COMMAND_SCHEMA

    kind: StateCommandKind
    store_generation: StoreGeneration
    expected_revision: StateRevision
    command_id: str
    idempotency_key: str = ""
    fence: FenceToken | None = None
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    issued_at_ms: int = 0
    bounds: ControlPlaneBounds = field(default_factory=ControlPlaneBounds)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, StateCommandKind, "kind")
        )
        if not isinstance(self.store_generation, StoreGeneration):
            if isinstance(self.store_generation, Mapping):
                object.__setattr__(
                    self,
                    "store_generation",
                    StoreGeneration.from_dict(self.store_generation),
                )
            else:
                raise ControlPlaneContractError(
                    "store_generation must be a StoreGeneration"
                )
        if not isinstance(self.expected_revision, StateRevision):
            if isinstance(self.expected_revision, Mapping):
                object.__setattr__(
                    self,
                    "expected_revision",
                    StateRevision.from_dict(self.expected_revision),
                )
            else:
                raise ControlPlaneContractError(
                    "expected_revision must be a StateRevision"
                )
        if not self.expected_revision.matches_generation(self.store_generation):
            raise ControlPlaneGenerationError(
                "expected_revision generation does not match store generation"
            )
        object.__setattr__(
            self, "command_id", _cid_or_digest(self.command_id, "command_id")
        )
        if not isinstance(self.bounds, ControlPlaneBounds):
            if isinstance(self.bounds, Mapping):
                object.__setattr__(
                    self, "bounds", ControlPlaneBounds.from_dict(self.bounds)
                )
            else:
                raise ControlPlaneContractError("bounds must be ControlPlaneBounds")
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(
                self.parameters,
                name="parameters",
                max_depth=self.bounds.max_depth,
                max_items=self.bounds.max_items,
                max_text_bytes=self.bounds.max_text_bytes,
            ),
        )
        if self.fence is not None and not isinstance(self.fence, FenceToken):
            if isinstance(self.fence, Mapping):
                object.__setattr__(self, "fence", FenceToken.from_dict(self.fence))
            else:
                raise ControlPlaneContractError("fence must be a FenceToken")
        if self.kind.requires_fence and self.fence is None:
            raise ControlPlaneContractError(
                f"{self.kind.value} commands require a fence token"
            )
        idempotency = _optional_text(self.idempotency_key, "idempotency_key")
        if self.kind.requires_idempotency and not idempotency:
            raise ControlPlaneContractError(
                f"{self.kind.value} commands require an idempotency key"
            )
        object.__setattr__(self, "idempotency_key", idempotency)
        object.__setattr__(
            self,
            "issued_at_ms",
            _finite_int(
                self.issued_at_ms,
                "issued_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )
        encoded = self.canonical_bytes()
        if len(encoded) > self.bounds.max_serialized_bytes:
            raise ControlPlaneBoundsError(
                "state command exceeds max_serialized_bytes"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "command_id": self.command_id,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "expected_revision": self.expected_revision.to_dict(),
            "fence": None if self.fence is None else self.fence.to_dict(),
            "idempotency_key": self.idempotency_key,
            "issued_at_ms": self.issued_at_ms,
            "kind": self.kind.value,
            "parameters": dict(self.parameters),
            "store_generation": self.store_generation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateCommand":
        _schema(payload, STATE_COMMAND_SCHEMA)
        _reject_mutable_alias_identity(payload, context="state command")
        _reject_unknown(
            payload,
            {
                "kind",
                "store_generation",
                "expected_revision",
                "command_id",
                "idempotency_key",
                "fence",
                "parameters",
                "issued_at_ms",
                "bounds",
            },
            "state command",
        )
        instance = cls(
            kind=payload["kind"],
            store_generation=payload["store_generation"],
            expected_revision=payload["expected_revision"],
            command_id=payload["command_id"],
            idempotency_key=payload.get("idempotency_key", ""),
            fence=payload.get("fence"),
            parameters=payload.get("parameters", {}),
            issued_at_ms=payload.get("issued_at_ms", 0),
            bounds=payload.get("bounds", ControlPlaneBounds()),
        )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class StateSnapshot(CanonicalContract):
    """Point-in-time snapshot bound to store generation and revision watermark."""

    SCHEMA: ClassVar[str] = STATE_SNAPSHOT_SCHEMA

    store_generation: StoreGeneration
    revision: StateRevision
    snapshot_digest: str
    captured_at_ms: int = 0
    row_count: int = 0
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITY

    def __post_init__(self) -> None:
        if not isinstance(self.store_generation, StoreGeneration):
            if isinstance(self.store_generation, Mapping):
                object.__setattr__(
                    self,
                    "store_generation",
                    StoreGeneration.from_dict(self.store_generation),
                )
            else:
                raise ControlPlaneContractError(
                    "store_generation must be a StoreGeneration"
                )
        if not isinstance(self.revision, StateRevision):
            if isinstance(self.revision, Mapping):
                object.__setattr__(
                    self, "revision", StateRevision.from_dict(self.revision)
                )
            else:
                raise ControlPlaneContractError("revision must be a StateRevision")
        if not self.revision.matches_generation(self.store_generation):
            raise ControlPlaneGenerationError(
                "snapshot revision generation does not match store generation"
            )
        object.__setattr__(
            self, "snapshot_digest", _digest(self.snapshot_digest, "snapshot_digest")
        )
        object.__setattr__(
            self,
            "captured_at_ms",
            _finite_int(
                self.captured_at_ms,
                "captured_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )
        object.__setattr__(
            self,
            "row_count",
            _finite_int(self.row_count, "row_count", minimum=0),
        )
        object.__setattr__(
            self,
            "authority_class",
            _enum(self.authority_class, StateAuthorityClass, "authority_class"),
        )
        if self.authority_class is not StateAuthorityClass.AUTHORITY:
            raise ControlPlaneAuthorityError(
                "state snapshot authority_class must be authority"
            )

    @property
    def snapshot_id(self) -> str:
        return _identity_material(
            self.store_generation.generation_id,
            str(self.revision.revision),
            str(self.revision.watermark),
            self.snapshot_digest,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "authority_class": self.authority_class.value,
            "captured_at_ms": self.captured_at_ms,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "revision": self.revision.to_dict(),
            "row_count": self.row_count,
            "snapshot_digest": self.snapshot_digest,
            "snapshot_id": self.snapshot_id,
            "store_generation": self.store_generation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateSnapshot":
        _schema(payload, STATE_SNAPSHOT_SCHEMA)
        _reject_mutable_alias_identity(payload, context="state snapshot")
        _reject_unknown(
            payload,
            {
                "store_generation",
                "revision",
                "snapshot_digest",
                "captured_at_ms",
                "row_count",
                "authority_class",
                "snapshot_id",
            },
            "state snapshot",
        )
        instance = cls(
            store_generation=payload["store_generation"],
            revision=payload["revision"],
            snapshot_digest=payload["snapshot_digest"],
            captured_at_ms=payload.get("captured_at_ms", 0),
            row_count=payload.get("row_count", 0),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITY
            ),
        )
        claimed = payload.get("snapshot_id")
        if claimed not in (None, "") and claimed != instance.snapshot_id:
            raise ControlPlaneIdentityError(
                "forged or inconsistent snapshot_id; rebuild from canonical payload"
            )
        _check_claimed_identity(payload, instance.content_id)
        return instance


@dataclass(frozen=True)
class StateExportReceipt(CanonicalContract):
    """Receipt for a deterministic non-authoritative export of a snapshot.

    Exports are projections.  Labeling an export authoritative is always
    rejected so that human/tool artifacts cannot re-enter as decision authority.
    """

    SCHEMA: ClassVar[str] = STATE_EXPORT_RECEIPT_SCHEMA

    snapshot: StateSnapshot
    profile: ExportProfile
    fidelity: ExportFidelity
    artifact_digest: str
    renderer_revision: str
    view_revision: str
    destination: str = ""
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    exported_at_ms: int = 0
    intentionally_omitted_fields: tuple[str, ...] = ()
    authority_class: StateAuthorityClass = StateAuthorityClass.EXPORT
    redaction_policy: RedactionPolicy = RedactionPolicy.REJECT

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, StateSnapshot):
            if isinstance(self.snapshot, Mapping):
                object.__setattr__(
                    self, "snapshot", StateSnapshot.from_dict(self.snapshot)
                )
            else:
                raise ControlPlaneContractError("snapshot must be a StateSnapshot")
        object.__setattr__(
            self, "profile", _enum(self.profile, ExportProfile, "profile")
        )
        object.__setattr__(
            self, "fidelity", _enum(self.fidelity, ExportFidelity, "fidelity")
        )
        object.__setattr__(
            self, "artifact_digest", _digest(self.artifact_digest, "artifact_digest")
        )
        object.__setattr__(
            self,
            "renderer_revision",
            _text(self.renderer_revision, "renderer_revision"),
        )
        object.__setattr__(
            self, "view_revision", _text(self.view_revision, "view_revision")
        )
        object.__setattr__(
            self, "destination", _optional_text(self.destination, "destination")
        )
        # Destination paths are not identity; reject secret material only.
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, name="parameters"),
        )
        object.__setattr__(
            self,
            "exported_at_ms",
            _finite_int(
                self.exported_at_ms,
                "exported_at_ms",
                minimum=0,
                maximum=ABSOLUTE_MAX_TIMESTAMP_MS,
            ),
        )
        omissions = self.intentionally_omitted_fields or ()
        if isinstance(omissions, str):
            raise ControlPlaneContractError(
                "intentionally_omitted_fields must be a sequence of strings"
            )
        cleaned: list[str] = []
        for item in omissions:
            text = _text(item, "intentionally_omitted_fields item")
            if text not in cleaned:
                cleaned.append(text)
        object.__setattr__(self, "intentionally_omitted_fields", tuple(cleaned))
        object.__setattr__(
            self,
            "authority_class",
            _enum(self.authority_class, StateAuthorityClass, "authority_class"),
        )
        if self.authority_class is StateAuthorityClass.AUTHORITY:
            raise ControlPlaneAuthorityError(
                "export labeled authoritative is rejected; exports are projections"
            )
        if self.authority_class is not StateAuthorityClass.EXPORT:
            raise ControlPlaneAuthorityError(
                "state export receipt authority_class must be export"
            )
        object.__setattr__(
            self,
            "redaction_policy",
            _enum(self.redaction_policy, RedactionPolicy, "redaction_policy"),
        )
        if (
            self.fidelity is ExportFidelity.INTENTIONALLY_LOSSY
            and not self.intentionally_omitted_fields
            and self.profile is ExportProfile.MARKDOWN
        ):
            raise ControlPlaneContractError(
                "intentionally lossy Markdown exports must declare omitted fields"
            )

    @property
    def export_id(self) -> str:
        return _identity_material(
            self.snapshot.snapshot_id,
            self.profile.value,
            self.fidelity.value,
            self.artifact_digest,
            self.renderer_revision,
            self.view_revision,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
            "authority_class": self.authority_class.value,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "destination": self.destination,
            "export_id": self.export_id,
            "exported_at_ms": self.exported_at_ms,
            "fidelity": self.fidelity.value,
            "intentionally_omitted_fields": list(self.intentionally_omitted_fields),
            "parameters": dict(self.parameters),
            "profile": self.profile.value,
            "redaction_policy": self.redaction_policy.value,
            "renderer_revision": self.renderer_revision,
            "snapshot": self.snapshot.to_dict(),
            "view_revision": self.view_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateExportReceipt":
        _schema(payload, STATE_EXPORT_RECEIPT_SCHEMA)
        # Exports may carry a destination path string, but never mutable alias
        # identity keys at the top level as identity substitutes.
        alias_keys = set(payload) & _MUTABLE_ALIAS_IDENTITY_KEYS
        if alias_keys:
            raise ControlPlaneIdentityError(
                "export receipt cannot use mutable alias "
                f"{sorted(alias_keys)[0]!r} as identity"
            )
        _reject_unknown(
            payload,
            {
                "snapshot",
                "profile",
                "fidelity",
                "artifact_digest",
                "renderer_revision",
                "view_revision",
                "destination",
                "parameters",
                "exported_at_ms",
                "intentionally_omitted_fields",
                "authority_class",
                "redaction_policy",
                "export_id",
            },
            "state export receipt",
        )
        # Explicit authority claim of "authoritative" is rejected even before
        # enum coercion when supplied as a free-form string that would map.
        authority = payload.get("authority_class", StateAuthorityClass.EXPORT)
        if str(getattr(authority, "value", authority)).casefold() in {
            "authority",
            "authoritative",
        }:
            raise ControlPlaneAuthorityError(
                "export labeled authoritative is rejected; exports are projections"
            )
        instance = cls(
            snapshot=payload["snapshot"],
            profile=payload["profile"],
            fidelity=payload["fidelity"],
            artifact_digest=payload["artifact_digest"],
            renderer_revision=payload["renderer_revision"],
            view_revision=payload["view_revision"],
            destination=payload.get("destination", ""),
            parameters=payload.get("parameters", {}),
            exported_at_ms=payload.get("exported_at_ms", 0),
            intentionally_omitted_fields=tuple(
                payload.get("intentionally_omitted_fields", ())
            ),
            authority_class=authority,
            redaction_policy=payload.get("redaction_policy", RedactionPolicy.REJECT),
        )
        claimed = payload.get("export_id")
        if claimed not in (None, "") and claimed != instance.export_id:
            raise ControlPlaneIdentityError(
                "forged or inconsistent export_id; rebuild from canonical payload"
            )
        _check_claimed_identity(payload, instance.content_id)
        return instance


def classify_state_authority(label: Any) -> StateAuthorityClass:
    """Parse a closed authority-class label or fail closed."""

    return _enum(label, StateAuthorityClass, "authority_class")


def assert_generation_revision_consistent(
    generation: StoreGeneration | Mapping[str, Any],
    revision: StateRevision | Mapping[str, Any],
) -> None:
    """Fail closed when a revision is bound to the wrong store generation."""

    gen = (
        generation
        if isinstance(generation, StoreGeneration)
        else StoreGeneration.from_dict(generation)
    )
    rev = (
        revision
        if isinstance(revision, StateRevision)
        else StateRevision.from_dict(revision)
    )
    if not rev.matches_generation(gen):
        raise ControlPlaneGenerationError(
            "generation/revision mismatch: revision generation "
            f"{rev.generation} != store generation {gen.generation}"
        )


def canonical_control_plane_json_bytes(value: Any) -> bytes:
    """Encode a control-plane contract or mapping as canonical JSON bytes."""

    if isinstance(value, CanonicalContract):
        return value.canonical_bytes()
    return canonical_json_bytes(value)


__all__ = (
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_FENCING_EPOCH",
    "ABSOLUTE_MAX_GENERATION",
    "ABSOLUTE_MAX_ITEMS",
    "ABSOLUTE_MAX_REVISION",
    "ABSOLUTE_MAX_SERIALIZED_BYTES",
    "ABSOLUTE_MAX_TEXT_BYTES",
    "ABSOLUTE_MAX_TIMESTAMP_MS",
    "ABSOLUTE_MAX_WATERMARK",
    "CONTRACT_VERSION",
    "CONTROL_PLANE_BOUNDS_SCHEMA",
    "CONTROL_PLANE_CONTRACT_VERSION",
    "CONTROL_PLANE_STORE_IDENTITY_SCHEMA",
    "ControlPlaneAuthorityError",
    "ControlPlaneBounds",
    "ControlPlaneBoundsError",
    "ControlPlaneCompatibilityError",
    "ControlPlaneContractError",
    "ControlPlaneGenerationError",
    "ControlPlaneIdentityError",
    "ControlPlaneSecretError",
    "ControlPlaneStoreIdentity",
    "ExportFidelity",
    "ExportProfile",
    "FENCE_TOKEN_SCHEMA",
    "FenceToken",
    "MAX_REDACTION_MARK",
    "RedactionPolicy",
    "SCHEMA_IDENTITY_SCHEMA",
    "SCHEMA_VERSION",
    "SESSION_IDENTITY_SCHEMA",
    "STATE_COMMAND_SCHEMA",
    "STATE_EXPORT_RECEIPT_SCHEMA",
    "STATE_REVISION_SCHEMA",
    "STATE_SNAPSHOT_SCHEMA",
    "STORE_GENERATION_SCHEMA",
    "SchemaIdentity",
    "SessionIdentity",
    "StateAuthorityClass",
    "StateCommand",
    "StateCommandKind",
    "StateExportReceipt",
    "StateRevision",
    "StateSnapshot",
    "StoreGeneration",
    "assert_generation_revision_consistent",
    "canonical_control_plane_json_bytes",
    "classify_state_authority",
    "redact_secrets",
)
