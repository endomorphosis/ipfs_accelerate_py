"""Canonical store, schema, identity, and authority contracts for the control plane.

This module is the provider-free serialization boundary for DuckDB/Quack
control-plane identities.  Closed, immutable records define:

* database / store / generation / schema / session identities;
* command, revision, fence, snapshot, and export receipts;
* state authority classes, integer bounds, typed failures, and redaction.

Identities are derived from canonical DAG-JSON and never accepted from a
caller as an unverified claim.  Display aliases, PIDs, hostnames, and file
paths may be carried as non-authoritative annotations but cannot serve as
identity material.  Exports are always non-authoritative projections.

Importing this module performs no filesystem, database, network, provider, or
process action.  It depends only on the standard library and the in-package
:mod:`task_identity` helpers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .task_identity import canonical_content_cid, canonical_json_bytes


# ---------------------------------------------------------------------------
# Schema and interface versions
# ---------------------------------------------------------------------------

CONTROL_PLANE_CONTRACT_VERSION: Final[int] = 1
CONTROL_PLANE_STORE_IDENTITY_INTERFACE: Final[str] = "ControlPlaneStoreIdentity@1"
STORE_GENERATION_INTERFACE: Final[str] = "StoreGeneration@1"
STATE_COMMAND_INTERFACE: Final[str] = "StateCommand@1"
STATE_SNAPSHOT_INTERFACE: Final[str] = "StateSnapshot@1"
STATE_EXPORT_RECEIPT_INTERFACE: Final[str] = "StateExportReceipt@1"

CONTROL_PLANE_STORE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-store-identity@1"
)
STORE_GENERATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-store-generation@1"
)
SCHEMA_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-schema-identity@1"
)
SESSION_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-session-identity@1"
)
STATE_COMMAND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-state-command@1"
)
STATE_REVISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-state-revision@1"
)
FENCE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-fence-identity@1"
)
STATE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-state-snapshot@1"
)
STATE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-state-export-receipt@1"
)
CONTROL_PLANE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-bounds@1"
)
SECRET_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-secret-handle@1"
)
CONTROL_PLANE_FAILURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-failure@1"
)

# Absolute integer bounds (fail closed beyond these ceilings).
ABSOLUTE_MAX_ITEMS: Final[int] = 10_000
ABSOLUTE_MAX_SERIALIZED_BYTES: Final[int] = 16 * 1024 * 1024
ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 65_536
ABSOLUTE_MAX_PATHS: Final[int] = 4_096
ABSOLUTE_MAX_DEPTH: Final[int] = 64
ABSOLUTE_MAX_TIMEOUT_MS: Final[int] = 24 * 60 * 60 * 1000
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_ANNOTATION_BYTES: Final[int] = 1_024
MAX_METADATA_ENTRIES: Final[int] = 32

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{50,}$")
_HANDLE_RE: Final[re.Pattern[str]] = re.compile(
    r"^secret-handle:[a-z0-9][a-z0-9._:-]{1,126}$"
)
_STABLE_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$"
)

# Keys that must never appear in public contract payloads (secret material).
_SECRET_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_value",
        "session_token",
        "token",
        "quack_token",
        "auth_token",
        "bearer",
        "passwd",
        "private_witness",
    }
)

# Annotations that may be present for display but never as identity material.
_MUTABLE_ALIAS_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "alias",
        "display_alias",
        "display_id",
        "display_name",
        "display_task_id",
        "hostname",
        "host",
        "local_path",
        "path",
        "pid",
        "process_id",
        "status_path",
        "worktree_path",
    }
)

_SECRET_VALUE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|pk|api)[_-]?[a-z0-9]{20,}\b|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b|"
    r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|"
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b)"
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class StateAuthorityClass(str, Enum):
    """Closed classification of supervisor state sinks and projections.

    Only ``AUTHORITATIVE`` grants orchestration write authority.  Exports,
    caches, OS bootstrap handles, and diagnostics never do.
    """

    AUTHORITATIVE = "authoritative"
    STATIC_INPUT = "static_input"
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    CACHE = "cache"
    EXPORT = "export"
    OS_BOOTSTRAP = "os_bootstrap"
    EMERGENCY_DIAGNOSTIC = "emergency_diagnostic"

    @property
    def grants_write_authority(self) -> bool:
        return self is StateAuthorityClass.AUTHORITATIVE

    @property
    def is_projection(self) -> bool:
        return self in {
            StateAuthorityClass.EXPORT,
            StateAuthorityClass.CACHE,
            StateAuthorityClass.OS_BOOTSTRAP,
            StateAuthorityClass.EMERGENCY_DIAGNOSTIC,
        }


class StateCommandKind(str, Enum):
    """Closed set of state-mutation command kinds."""

    CLAIM = "claim"
    RENEW = "renew"
    RELEASE = "release"
    ADVANCE = "advance"
    COMPLETE = "complete"
    CANCEL = "cancel"
    APPEND_EVENT = "append_event"
    UPSERT_PROJECTION = "upsert_projection"
    IMPORT = "import"
    EXPORT = "export"
    MAINTAIN = "maintain"
    ROTATE_GENERATION = "rotate_generation"


class ControlPlaneFailureCode(str, Enum):
    """Machine-readable fail-closed reasons for contract rejection."""

    EMPTY_IDENTITY = "empty_identity"
    FORGED_IDENTITY = "forged_identity"
    INCONSISTENT_IDENTITY = "inconsistent_identity"
    GENERATION_MISMATCH = "generation_mismatch"
    REVISION_MISMATCH = "revision_mismatch"
    NON_FINITE_BOUNDS = "non_finite_bounds"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    SECRET_MATERIAL = "secret_material"
    MUTABLE_ALIAS_AS_IDENTITY = "mutable_alias_as_identity"
    EXPORT_AUTHORITY_CLAIM = "export_authority_claim"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_ENUM = "invalid_enum"
    INVALID_TYPE = "invalid_type"
    MALFORMED_DIGEST = "malformed_digest"
    MALFORMED_UUID = "malformed_uuid"
    MALFORMED_CID = "malformed_cid"
    RECORD_TOO_LARGE = "record_too_large"


class ExportProfile(str, Enum):
    """Closed export profile vocabulary."""

    HUMAN_TASKBOARD = "human_taskboard"
    HUMAN_OBJECTIVES = "human_objectives"
    STATUS_JSON = "status_json"
    EVENT_JSONL = "event_jsonl"
    AUDIT_JSONL = "audit_jsonl"
    ANALYSIS_PARQUET = "analysis_parquet"
    PORTABLE_BUNDLE = "portable_bundle"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ControlPlaneContractError(ValueError):
    """Base fail-closed error for control-plane contracts."""

    def __init__(
        self,
        message: str,
        *,
        code: ControlPlaneFailureCode = ControlPlaneFailureCode.INVALID_TYPE,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = str(message)


class EmptyIdentityError(ControlPlaneContractError):
    """An identity field was empty or whitespace-only."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.EMPTY_IDENTITY)


class ForgedIdentityError(ControlPlaneContractError):
    """A claimed content_id or digest did not match the canonical payload."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.FORGED_IDENTITY)


class InconsistentIdentityError(ControlPlaneContractError):
    """Cross-field identity bindings disagreed."""

    def __init__(
        self,
        message: str,
        *,
        code: ControlPlaneFailureCode = ControlPlaneFailureCode.INCONSISTENT_IDENTITY,
    ) -> None:
        super().__init__(message, code=code)


class GenerationMismatchError(InconsistentIdentityError):
    """Command or snapshot generation did not match the store generation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.GENERATION_MISMATCH)


class RevisionMismatchError(InconsistentIdentityError):
    """Expected revision did not match the bound revision identity."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.REVISION_MISMATCH)


class NonFiniteBoundsError(ControlPlaneContractError):
    """A bound was non-finite, non-integer, or otherwise unsafe."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.NON_FINITE_BOUNDS)


class SecretMaterialError(ControlPlaneContractError):
    """Secret material appeared in a public contract field."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=ControlPlaneFailureCode.SECRET_MATERIAL)


class MutableAliasIdentityError(ControlPlaneContractError):
    """A mutable alias was supplied as identity material."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message, code=ControlPlaneFailureCode.MUTABLE_ALIAS_AS_IDENTITY
        )


class ExportAuthorityError(ControlPlaneContractError):
    """An export claimed authoritative state authority."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message, code=ControlPlaneFailureCode.EXPORT_AUTHORITY_CLAIM
        )


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted({str(item.value) for item in enum_type}))
        raise ControlPlaneContractError(
            f"{field_name} must be one of: {allowed}",
            code=ControlPlaneFailureCode.INVALID_ENUM,
        ) from exc


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ControlPlaneContractError(
            f"{field_name} must be a string",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    text = value.strip()
    if not text:
        raise EmptyIdentityError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise ControlPlaneContractError(
            f"{field_name} must not contain NUL",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    _reject_secret_text(text, field_name)
    return text


def _optional_text(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ControlPlaneContractError(
            f"{field_name} must be a string",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    text = value.strip()
    if "\x00" in text:
        raise ControlPlaneContractError(
            f"{field_name} must not contain NUL",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    if text:
        _reject_secret_text(text, field_name)
    return text


def _positive_int(value: Any, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise NonFiniteBoundsError(
                    f"{field_name} must be a finite integer bound"
                )
            raise NonFiniteBoundsError(
                f"{field_name} must be an integer; floats are rejected"
            )
        raise NonFiniteBoundsError(f"{field_name} must be an integer")
    if value < minimum:
        raise NonFiniteBoundsError(
            f"{field_name} must be >= {minimum}"
        )
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    return _positive_int(value, field_name, minimum=0)


def _uuid(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).casefold()
    if not _UUID_RE.match(text):
        raise ControlPlaneContractError(
            f"{field_name} must be a UUID",
            code=ControlPlaneFailureCode.MALFORMED_UUID,
        )
    return text


def _sha256_digest(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).casefold()
    if not _SHA256_RE.match(text):
        raise ControlPlaneContractError(
            f"{field_name} must be sha256:<64-hex>",
            code=ControlPlaneFailureCode.MALFORMED_DIGEST,
        )
    return text


def _cid(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).casefold()
    if not _CID_RE.match(text):
        raise ControlPlaneContractError(
            f"{field_name} must be a CIDv1 base32 identity",
            code=ControlPlaneFailureCode.MALFORMED_CID,
        )
    return text


def _stable_id(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if not _STABLE_ID_RE.match(text):
        raise ControlPlaneContractError(
            f"{field_name} has an invalid identity shape",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    if text.casefold() in _MUTABLE_ALIAS_FIELDS:
        raise MutableAliasIdentityError(
            f"{field_name} cannot be a mutable alias name"
        )
    # Reject pure numeric PIDs and bare host-looking tokens used as identity.
    if text.isdigit():
        raise MutableAliasIdentityError(
            f"{field_name} cannot be a bare process id"
        )
    return text


def _reject_secret_text(text: str, field_name: str) -> None:
    if _SECRET_VALUE_RE.search(text):
        raise SecretMaterialError(
            f"{field_name} must not contain secret material"
        )


def _reject_secret_keys(payload: Mapping[str, Any], *, context: str) -> None:
    for key in payload:
        normalized = str(key).strip().casefold().replace("-", "_")
        if normalized in _SECRET_FIELD_MARKERS:
            raise SecretMaterialError(
                f"{context} must not include secret field {key!r}"
            )
        # Compound secret field names (client_secret, db_password, …).  Exact
        # digest fields such as fence_token are intentionally not matched.
        for marker in (
            "password",
            "passwd",
            "secret",
            "private_key",
            "api_key",
            "access_token",
            "refresh_token",
            "session_token",
            "auth_token",
            "quack_token",
            "credential",
        ):
            if (
                normalized == marker
                or normalized.endswith("_" + marker)
                or normalized.startswith(marker + "_")
            ):
                if "handle" in normalized:
                    continue
                raise SecretMaterialError(
                    f"{context} must not include secret field {key!r}"
                )


def _reject_mutable_alias_identity(
    identity_value: str,
    annotations: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    """Reject identity equal to a mutable annotation value."""

    for key, raw in annotations.items():
        key_norm = str(key).strip().casefold().replace("-", "_")
        if key_norm not in _MUTABLE_ALIAS_FIELDS:
            continue
        if isinstance(raw, (int, str)) and str(raw).strip() == identity_value:
            raise MutableAliasIdentityError(
                f"{field_name} cannot equal mutable alias {key_norm!r}"
            )


def _canonical_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ControlPlaneContractError(
            f"{field_name} must be a mapping",
            code=ControlPlaneFailureCode.INVALID_TYPE,
        )
    if len(value) > MAX_METADATA_ENTRIES:
        raise ControlPlaneContractError(
            f"{field_name} exceeds metadata entry bound",
            code=ControlPlaneFailureCode.BOUNDS_EXCEEDED,
        )
    _reject_secret_keys(value, context=field_name)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ControlPlaneContractError(
                f"{field_name} keys must be non-empty strings",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        key_norm = key.strip()
        if key_norm.casefold().replace("-", "_") in _MUTABLE_ALIAS_FIELDS:
            # Allowed only as annotation values, not nested identity objects.
            if isinstance(item, Mapping):
                raise MutableAliasIdentityError(
                    f"{field_name}.{key_norm} cannot carry nested identity"
                )
        result[key_norm] = _canonical_json_value(item, field_name=f"{field_name}.{key_norm}")
    # Size-bound the annotation blob.
    encoded = canonical_json_bytes(result)
    if len(encoded) > MAX_ANNOTATION_BYTES * MAX_METADATA_ENTRIES:
        raise ControlPlaneContractError(
            f"{field_name} exceeds annotation size bound",
            code=ControlPlaneFailureCode.BOUNDS_EXCEEDED,
        )
    return result


def _canonical_json_value(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            _reject_secret_text(value, field_name)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteBoundsError(
                f"{field_name} must not be NaN or infinity"
            )
        raise NonFiniteBoundsError(
            f"{field_name} must not be a float; use integer units"
        )
    if isinstance(value, Enum):
        return _canonical_json_value(value.value, field_name=field_name)
    if isinstance(value, Mapping):
        return _canonical_mapping(value, field_name=field_name)
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, field_name=f"{field_name}[]")
            for item in value
        ]
    raise ControlPlaneContractError(
        f"{field_name} has unsupported type {type(value).__name__}",
        code=ControlPlaneFailureCode.INVALID_TYPE,
    )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ControlPlaneContractError(
            f"{context} has unknown fields: {', '.join(unknown)}",
            code=ControlPlaneFailureCode.UNKNOWN_FIELD,
        )


def _require_schema(payload: Mapping[str, Any], expected: str) -> None:
    schema = payload.get("schema", expected)
    if schema != expected:
        raise ControlPlaneContractError(
            f"schema must be {expected}",
            code=ControlPlaneFailureCode.INVALID_SCHEMA,
        )


def _claimed_content_id(payload: Mapping[str, Any], expected: str, context: str) -> None:
    claimed = payload.get("content_id", payload.get("cid", payload.get("identity")))
    if claimed in (None, ""):
        return
    if not isinstance(claimed, str) or claimed != expected:
        raise ForgedIdentityError(
            f"{context} content_id is forged or inconsistent"
        )


def _bounded_record(payload: Mapping[str, Any], *, context: str) -> None:
    size = len(canonical_json_bytes(payload))
    if size > MAX_RECORD_BYTES:
        raise ControlPlaneContractError(
            f"{context} exceeds max record bytes ({size} > {MAX_RECORD_BYTES})",
            code=ControlPlaneFailureCode.RECORD_TOO_LARGE,
        )


def content_identity(value: Any) -> str:
    """Return the package-local CIDv1 identity for a canonical payload."""

    return canonical_content_cid(value)


def redact_public_text(value: Any) -> str:
    """Return a secret-free public projection of ``value``."""

    text = str(value or "")
    redacted = _SECRET_VALUE_RE.sub("[redacted]", text)
    for marker in sorted(_SECRET_FIELD_MARKERS, key=len, reverse=True):
        redacted = re.sub(
            re.escape(marker),
            "[redacted-field]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted[:MAX_ANNOTATION_BYTES]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CanonicalRecord:
    """Shared mixin for immutable content-addressed control-plane records."""

    SCHEMA: ClassVar[str] = ""

    def _payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": self.SCHEMA, **self._payload()}
        _bounded_record(body, context=type(self).__name__)
        return body

    def to_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_json(self) -> str:
        return self.to_json_bytes().decode("utf-8")

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def cid(self) -> str:
        return self.content_id

    @property
    def identity(self) -> str:
        return self.content_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}


@dataclass(frozen=True)
class ControlPlaneBounds(_CanonicalRecord):
    """Integer resource bounds carried by commands and sessions."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_BOUNDS_SCHEMA

    max_items: int = 256
    max_serialized_bytes: int = 262_144
    max_text_bytes: int = 8_192
    max_paths: int = 128
    max_depth: int = 8
    timeout_ms: int = 30_000
    max_retries: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_items", _positive_int(self.max_items, "max_items")
        )
        object.__setattr__(
            self,
            "max_serialized_bytes",
            _positive_int(self.max_serialized_bytes, "max_serialized_bytes"),
        )
        object.__setattr__(
            self,
            "max_text_bytes",
            _positive_int(self.max_text_bytes, "max_text_bytes"),
        )
        object.__setattr__(
            self, "max_paths", _positive_int(self.max_paths, "max_paths")
        )
        object.__setattr__(
            self, "max_depth", _positive_int(self.max_depth, "max_depth")
        )
        object.__setattr__(
            self, "timeout_ms", _positive_int(self.timeout_ms, "timeout_ms")
        )
        object.__setattr__(
            self,
            "max_retries",
            _non_negative_int(self.max_retries, "max_retries"),
        )
        if self.max_items > ABSOLUTE_MAX_ITEMS:
            raise NonFiniteBoundsError("max_items exceeds absolute limit")
        if self.max_serialized_bytes > ABSOLUTE_MAX_SERIALIZED_BYTES:
            raise NonFiniteBoundsError(
                "max_serialized_bytes exceeds absolute limit"
            )
        if self.max_text_bytes > ABSOLUTE_MAX_TEXT_BYTES:
            raise NonFiniteBoundsError("max_text_bytes exceeds absolute limit")
        if self.max_paths > ABSOLUTE_MAX_PATHS:
            raise NonFiniteBoundsError("max_paths exceeds absolute limit")
        if self.max_depth > ABSOLUTE_MAX_DEPTH:
            raise NonFiniteBoundsError("max_depth exceeds absolute limit")
        if self.timeout_ms > ABSOLUTE_MAX_TIMEOUT_MS:
            raise NonFiniteBoundsError("timeout_ms exceeds absolute limit")
        if self.max_paths > self.max_items:
            raise NonFiniteBoundsError("max_paths cannot exceed max_items")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "max_depth": self.max_depth,
            "max_items": self.max_items,
            "max_paths": self.max_paths,
            "max_retries": self.max_retries,
            "max_serialized_bytes": self.max_serialized_bytes,
            "max_text_bytes": self.max_text_bytes,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneBounds":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError("bounds payload must be a mapping")
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="bounds")
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "max_items",
                "max_serialized_bytes",
                "max_text_bytes",
                "max_paths",
                "max_depth",
                "timeout_ms",
                "max_retries",
                "content_id",
                "cid",
                "identity",
            },
            context="bounds",
        )
        defaults = cls()
        result = cls(
            max_items=payload.get("max_items", defaults.max_items),
            max_serialized_bytes=payload.get(
                "max_serialized_bytes", defaults.max_serialized_bytes
            ),
            max_text_bytes=payload.get("max_text_bytes", defaults.max_text_bytes),
            max_paths=payload.get("max_paths", defaults.max_paths),
            max_depth=payload.get("max_depth", defaults.max_depth),
            timeout_ms=payload.get("timeout_ms", defaults.timeout_ms),
            max_retries=payload.get("max_retries", defaults.max_retries),
        )
        _claimed_content_id(payload, result.content_id, "bounds")
        return result


@dataclass(frozen=True)
class SecretHandle(_CanonicalRecord):
    """Opaque reference to a protected secret; never carries secret bytes."""

    SCHEMA: ClassVar[str] = SECRET_HANDLE_SCHEMA

    handle_id: str
    purpose: str = "quack_auth"
    generation: int = 1

    def __post_init__(self) -> None:
        handle = _required_text(self.handle_id, "handle_id")
        if not _HANDLE_RE.match(handle):
            raise ControlPlaneContractError(
                "handle_id must be secret-handle:<stable-id>",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        object.__setattr__(self, "handle_id", handle)
        object.__setattr__(
            self, "purpose", _required_text(self.purpose, "purpose")
        )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation,
            "handle_id": self.handle_id,
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SecretHandle":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "secret handle payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="secret handle")
        # Explicit secret-value fields are always rejected even if unknown-field
        # checks would catch them under different names.
        for banned in ("value", "secret", "token", "password", "credential"):
            if banned in payload:
                raise SecretMaterialError(
                    "secret handle must not embed secret material"
                )
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "handle_id",
                "purpose",
                "generation",
                "content_id",
                "cid",
                "identity",
            },
            context="secret handle",
        )
        result = cls(
            handle_id=payload.get("handle_id", ""),
            purpose=payload.get("purpose", "quack_auth"),
            generation=payload.get("generation", 1),
        )
        _claimed_content_id(payload, result.content_id, "secret handle")
        return result


@dataclass(frozen=True)
class SchemaIdentity(_CanonicalRecord):
    """Checksum-bound schema revision identity for the control plane."""

    SCHEMA: ClassVar[str] = SCHEMA_IDENTITY_SCHEMA

    schema_revision: int
    schema_fingerprint: str
    catalog_fingerprint: str
    migration_head: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_revision",
            _positive_int(self.schema_revision, "schema_revision"),
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            _sha256_digest(self.schema_fingerprint, "schema_fingerprint"),
        )
        object.__setattr__(
            self,
            "catalog_fingerprint",
            _sha256_digest(self.catalog_fingerprint, "catalog_fingerprint"),
        )
        object.__setattr__(
            self,
            "migration_head",
            _non_negative_int(self.migration_head, "migration_head"),
        )
        if self.migration_head > self.schema_revision:
            raise InconsistentIdentityError(
                "migration_head cannot exceed schema_revision"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "catalog_fingerprint": self.catalog_fingerprint,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "migration_head": self.migration_head,
            "schema_fingerprint": self.schema_fingerprint,
            "schema_revision": self.schema_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaIdentity":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "schema identity payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="schema identity")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "schema_revision",
                "schema_fingerprint",
                "catalog_fingerprint",
                "migration_head",
                "content_id",
                "cid",
                "identity",
            },
            context="schema identity",
        )
        result = cls(
            schema_revision=payload.get("schema_revision", 0),
            schema_fingerprint=payload.get("schema_fingerprint", ""),
            catalog_fingerprint=payload.get("catalog_fingerprint", ""),
            migration_head=payload.get("migration_head", 0),
        )
        _claimed_content_id(payload, result.content_id, "schema identity")
        return result


@dataclass(frozen=True)
class StoreGeneration(_CanonicalRecord):
    """Monotonic store generation used for fencing after restore/takeover."""

    SCHEMA: ClassVar[str] = STORE_GENERATION_SCHEMA

    generation: int
    store_uuid: str
    epoch_id: str
    created_at_ms: int
    reason: str = "bootstrap"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(self, "store_uuid", _uuid(self.store_uuid, "store_uuid"))
        object.__setattr__(self, "epoch_id", _stable_id(self.epoch_id, "epoch_id"))
        object.__setattr__(
            self,
            "created_at_ms",
            _non_negative_int(self.created_at_ms, "created_at_ms"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "created_at_ms": self.created_at_ms,
            "epoch_id": self.epoch_id,
            "generation": self.generation,
            "reason": self.reason,
            "store_uuid": self.store_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoreGeneration":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store generation payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="store generation")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "generation",
                "store_uuid",
                "epoch_id",
                "created_at_ms",
                "reason",
                "content_id",
                "cid",
                "identity",
            },
            context="store generation",
        )
        result = cls(
            generation=payload.get("generation", 0),
            store_uuid=payload.get("store_uuid", ""),
            epoch_id=payload.get("epoch_id", ""),
            created_at_ms=payload.get("created_at_ms", 0),
            reason=payload.get("reason", "bootstrap"),
        )
        _claimed_content_id(payload, result.content_id, "store generation")
        return result


@dataclass(frozen=True)
class ControlPlaneStoreIdentity(_CanonicalRecord):
    """Canonical identity of one repository-scoped control.duckdb authority."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_STORE_IDENTITY_SCHEMA

    store_uuid: str
    repository_id: str
    generation: StoreGeneration
    schema_identity: SchemaIdentity
    authority: StateAuthorityClass = StateAuthorityClass.AUTHORITATIVE
    annotations: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_uuid", _uuid(self.store_uuid, "store_uuid"))
        object.__setattr__(
            self, "repository_id", _stable_id(self.repository_id, "repository_id")
        )
        if not isinstance(self.generation, StoreGeneration):
            raise ControlPlaneContractError(
                "generation must be a StoreGeneration",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        if self.generation.store_uuid != self.store_uuid:
            raise InconsistentIdentityError(
                "store_uuid must match generation.store_uuid"
            )
        if not isinstance(self.schema_identity, SchemaIdentity):
            raise ControlPlaneContractError(
                "schema_identity must be a SchemaIdentity",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, StateAuthorityClass, field_name="authority"),
        )
        if self.authority is not StateAuthorityClass.AUTHORITATIVE:
            raise InconsistentIdentityError(
                "control-plane store identity must be authoritative"
            )
        annotations = _canonical_mapping(
            dict(self.annotations), field_name="annotations"
        )
        object.__setattr__(self, "annotations", MappingProxyType(annotations))
        _reject_mutable_alias_identity(
            self.store_uuid, annotations, field_name="store_uuid"
        )
        _reject_mutable_alias_identity(
            self.repository_id, annotations, field_name="repository_id"
        )
        # Mutable aliases must never be treated as the store identity key.
        for key in annotations:
            if str(key).casefold().replace("-", "_") in {
                "store_id",
                "store_key",
                "identity",
                "id",
            }:
                raise MutableAliasIdentityError(
                    "annotations cannot redefine store identity"
                )

    @property
    def generation_number(self) -> int:
        return self.generation.generation

    @property
    def schema_revision(self) -> int:
        return self.schema_identity.schema_revision

    def _payload(self) -> dict[str, Any]:
        return {
            "annotations": dict(self.annotations),
            "authority": self.authority.value,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation.to_dict(),
            "repository_id": self.repository_id,
            "schema_identity": self.schema_identity.to_dict(),
            "store_uuid": self.store_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneStoreIdentity":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store identity payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="store identity")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "store_uuid",
                "repository_id",
                "generation",
                "schema_identity",
                "authority",
                "annotations",
                "content_id",
                "cid",
                "identity",
            },
            context="store identity",
        )
        generation_raw = payload.get("generation")
        if not isinstance(generation_raw, Mapping):
            raise ControlPlaneContractError(
                "generation must be an object",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        schema_raw = payload.get("schema_identity")
        if not isinstance(schema_raw, Mapping):
            raise ControlPlaneContractError(
                "schema_identity must be an object",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        result = cls(
            store_uuid=payload.get("store_uuid", ""),
            repository_id=payload.get("repository_id", ""),
            generation=StoreGeneration.from_dict(generation_raw),
            schema_identity=SchemaIdentity.from_dict(schema_raw),
            authority=payload.get(
                "authority", StateAuthorityClass.AUTHORITATIVE
            ),
            annotations=payload.get("annotations", {}),
        )
        _claimed_content_id(payload, result.content_id, "store identity")
        return result


@dataclass(frozen=True)
class SessionIdentity(_CanonicalRecord):
    """Client or server session bound to store generation and process birth."""

    SCHEMA: ClassVar[str] = SESSION_IDENTITY_SCHEMA

    session_id: str
    store_uuid: str
    generation: int
    process_birth_id: str
    role: str = "client"
    annotations: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", _stable_id(self.session_id, "session_id")
        )
        object.__setattr__(self, "store_uuid", _uuid(self.store_uuid, "store_uuid"))
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(
            self,
            "process_birth_id",
            _stable_id(self.process_birth_id, "process_birth_id"),
        )
        object.__setattr__(self, "role", _required_text(self.role, "role"))
        annotations = _canonical_mapping(
            dict(self.annotations), field_name="annotations"
        )
        object.__setattr__(self, "annotations", MappingProxyType(annotations))
        # PID alone is not process birth identity.
        if self.process_birth_id.isdigit():
            raise MutableAliasIdentityError(
                "process_birth_id cannot be a bare PID"
            )
        _reject_mutable_alias_identity(
            self.session_id, annotations, field_name="session_id"
        )
        _reject_mutable_alias_identity(
            self.process_birth_id, annotations, field_name="process_birth_id"
        )
        if "pid" in {k.casefold() for k in annotations} and str(
            annotations.get("pid", annotations.get("PID", ""))
        ) == self.process_birth_id:
            raise MutableAliasIdentityError(
                "process_birth_id cannot equal mutable pid annotation"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "annotations": dict(self.annotations),
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation,
            "process_birth_id": self.process_birth_id,
            "role": self.role,
            "session_id": self.session_id,
            "store_uuid": self.store_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionIdentity":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "session identity payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="session identity")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "session_id",
                "store_uuid",
                "generation",
                "process_birth_id",
                "role",
                "annotations",
                "content_id",
                "cid",
                "identity",
            },
            context="session identity",
        )
        result = cls(
            session_id=payload.get("session_id", ""),
            store_uuid=payload.get("store_uuid", ""),
            generation=payload.get("generation", 0),
            process_birth_id=payload.get("process_birth_id", ""),
            role=payload.get("role", "client"),
            annotations=payload.get("annotations", {}),
        )
        _claimed_content_id(payload, result.content_id, "session identity")
        return result


@dataclass(frozen=True)
class StateRevision(_CanonicalRecord):
    """Compare-and-swap revision for one mutable state row or stream."""

    SCHEMA: ClassVar[str] = STATE_REVISION_SCHEMA

    revision: int
    store_uuid: str
    generation: int
    stream_id: str
    revision_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision", _positive_int(self.revision, "revision")
        )
        object.__setattr__(self, "store_uuid", _uuid(self.store_uuid, "store_uuid"))
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(
            self, "stream_id", _stable_id(self.stream_id, "stream_id")
        )
        object.__setattr__(
            self,
            "revision_digest",
            _sha256_digest(self.revision_digest, "revision_digest"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "generation": self.generation,
            "revision": self.revision,
            "revision_digest": self.revision_digest,
            "store_uuid": self.store_uuid,
            "stream_id": self.stream_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateRevision":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state revision payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="state revision")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "revision",
                "store_uuid",
                "generation",
                "stream_id",
                "revision_digest",
                "content_id",
                "cid",
                "identity",
            },
            context="state revision",
        )
        result = cls(
            revision=payload.get("revision", 0),
            store_uuid=payload.get("store_uuid", ""),
            generation=payload.get("generation", 0),
            stream_id=payload.get("stream_id", ""),
            revision_digest=payload.get("revision_digest", ""),
        )
        _claimed_content_id(payload, result.content_id, "state revision")
        return result


@dataclass(frozen=True)
class FenceIdentity(_CanonicalRecord):
    """Lease fence binding owner session, scope, epoch, and expected revision."""

    SCHEMA: ClassVar[str] = FENCE_IDENTITY_SCHEMA

    fence_epoch: int
    fence_token: str
    session_id: str
    store_uuid: str
    generation: int
    scope_id: str
    expected_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fence_epoch", _positive_int(self.fence_epoch, "fence_epoch")
        )
        object.__setattr__(
            self, "fence_token", _sha256_digest(self.fence_token, "fence_token")
        )
        object.__setattr__(
            self, "session_id", _stable_id(self.session_id, "session_id")
        )
        object.__setattr__(self, "store_uuid", _uuid(self.store_uuid, "store_uuid"))
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(self, "scope_id", _stable_id(self.scope_id, "scope_id"))
        object.__setattr__(
            self,
            "expected_revision",
            _positive_int(self.expected_revision, "expected_revision"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "expected_revision": self.expected_revision,
            "fence_epoch": self.fence_epoch,
            "fence_token": self.fence_token,
            "generation": self.generation,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "store_uuid": self.store_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FenceIdentity":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "fence identity payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="fence identity")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "fence_epoch",
                "fence_token",
                "session_id",
                "store_uuid",
                "generation",
                "scope_id",
                "expected_revision",
                "content_id",
                "cid",
                "identity",
            },
            context="fence identity",
        )
        result = cls(
            fence_epoch=payload.get("fence_epoch", 0),
            fence_token=payload.get("fence_token", ""),
            session_id=payload.get("session_id", ""),
            store_uuid=payload.get("store_uuid", ""),
            generation=payload.get("generation", 0),
            scope_id=payload.get("scope_id", ""),
            expected_revision=payload.get("expected_revision", 0),
        )
        _claimed_content_id(payload, result.content_id, "fence identity")
        return result


@dataclass(frozen=True)
class StateCommand(_CanonicalRecord):
    """Fenced, idempotent state transition command against one store generation."""

    SCHEMA: ClassVar[str] = STATE_COMMAND_SCHEMA

    command_id: str
    kind: StateCommandKind
    store: ControlPlaneStoreIdentity
    session: SessionIdentity
    fence: FenceIdentity
    expected_revision: StateRevision
    idempotency_key: str
    bounds: ControlPlaneBounds
    parameters: Mapping[str, Any] = MappingProxyType({})
    issued_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _stable_id(self.command_id, "command_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, StateCommandKind, field_name="kind")
        )
        if not isinstance(self.store, ControlPlaneStoreIdentity):
            raise ControlPlaneContractError(
                "store must be ControlPlaneStoreIdentity",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        if not isinstance(self.session, SessionIdentity):
            raise ControlPlaneContractError(
                "session must be SessionIdentity",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        if not isinstance(self.fence, FenceIdentity):
            raise ControlPlaneContractError(
                "fence must be FenceIdentity",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        if not isinstance(self.expected_revision, StateRevision):
            raise ControlPlaneContractError(
                "expected_revision must be StateRevision",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        if not isinstance(self.bounds, ControlPlaneBounds):
            raise ControlPlaneContractError(
                "bounds must be ControlPlaneBounds",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        object.__setattr__(
            self,
            "idempotency_key",
            _stable_id(self.idempotency_key, "idempotency_key"),
        )
        object.__setattr__(
            self,
            "issued_at_ms",
            _non_negative_int(self.issued_at_ms, "issued_at_ms"),
        )
        parameters = _canonical_mapping(
            dict(self.parameters), field_name="parameters"
        )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

        # Generation consistency across store, session, fence, and revision.
        store_generation = self.store.generation_number
        if self.session.store_uuid != self.store.store_uuid:
            raise InconsistentIdentityError(
                "session.store_uuid must match store.store_uuid"
            )
        if self.session.generation != store_generation:
            raise GenerationMismatchError(
                "session.generation must match store generation"
            )
        if self.fence.store_uuid != self.store.store_uuid:
            raise InconsistentIdentityError(
                "fence.store_uuid must match store.store_uuid"
            )
        if self.fence.generation != store_generation:
            raise GenerationMismatchError(
                "fence.generation must match store generation"
            )
        if self.fence.session_id != self.session.session_id:
            raise InconsistentIdentityError(
                "fence.session_id must match session.session_id"
            )
        if self.expected_revision.store_uuid != self.store.store_uuid:
            raise InconsistentIdentityError(
                "expected_revision.store_uuid must match store.store_uuid"
            )
        if self.expected_revision.generation != store_generation:
            raise GenerationMismatchError(
                "expected_revision.generation must match store generation"
            )
        if self.fence.expected_revision != self.expected_revision.revision:
            raise RevisionMismatchError(
                "fence.expected_revision must match expected_revision.revision"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "command_id": self.command_id,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "expected_revision": self.expected_revision.to_dict(),
            "fence": self.fence.to_dict(),
            "idempotency_key": self.idempotency_key,
            "issued_at_ms": self.issued_at_ms,
            "kind": self.kind.value,
            "parameters": dict(self.parameters),
            "session": self.session.to_dict(),
            "store": self.store.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateCommand":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state command payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="state command")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "command_id",
                "kind",
                "store",
                "session",
                "fence",
                "expected_revision",
                "idempotency_key",
                "bounds",
                "parameters",
                "issued_at_ms",
                "content_id",
                "cid",
                "identity",
            },
            context="state command",
        )
        for field_name in (
            "store",
            "session",
            "fence",
            "expected_revision",
            "bounds",
        ):
            if not isinstance(payload.get(field_name), Mapping):
                raise ControlPlaneContractError(
                    f"{field_name} must be an object",
                    code=ControlPlaneFailureCode.INVALID_TYPE,
                )
        result = cls(
            command_id=payload.get("command_id", ""),
            kind=payload.get("kind", ""),
            store=ControlPlaneStoreIdentity.from_dict(payload["store"]),
            session=SessionIdentity.from_dict(payload["session"]),
            fence=FenceIdentity.from_dict(payload["fence"]),
            expected_revision=StateRevision.from_dict(
                payload["expected_revision"]
            ),
            idempotency_key=payload.get("idempotency_key", ""),
            bounds=ControlPlaneBounds.from_dict(payload["bounds"]),
            parameters=payload.get("parameters", {}),
            issued_at_ms=payload.get("issued_at_ms", 0),
        )
        _claimed_content_id(payload, result.content_id, "state command")
        return result


@dataclass(frozen=True)
class StateSnapshot(_CanonicalRecord):
    """Point-in-time snapshot bound to store generation and transaction watermark."""

    SCHEMA: ClassVar[str] = STATE_SNAPSHOT_SCHEMA

    snapshot_id: str
    store: ControlPlaneStoreIdentity
    transaction_watermark: int
    snapshot_digest: str
    captured_at_ms: int
    authority: StateAuthorityClass = StateAuthorityClass.AUTHORITATIVE
    annotations: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _stable_id(self.snapshot_id, "snapshot_id")
        )
        if not isinstance(self.store, ControlPlaneStoreIdentity):
            raise ControlPlaneContractError(
                "store must be ControlPlaneStoreIdentity",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        object.__setattr__(
            self,
            "transaction_watermark",
            _non_negative_int(
                self.transaction_watermark, "transaction_watermark"
            ),
        )
        object.__setattr__(
            self,
            "snapshot_digest",
            _sha256_digest(self.snapshot_digest, "snapshot_digest"),
        )
        object.__setattr__(
            self,
            "captured_at_ms",
            _non_negative_int(self.captured_at_ms, "captured_at_ms"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, StateAuthorityClass, field_name="authority"),
        )
        if self.authority is not StateAuthorityClass.AUTHORITATIVE:
            raise InconsistentIdentityError(
                "state snapshot authority must be authoritative"
            )
        annotations = _canonical_mapping(
            dict(self.annotations), field_name="annotations"
        )
        object.__setattr__(self, "annotations", MappingProxyType(annotations))
        _reject_mutable_alias_identity(
            self.snapshot_id, annotations, field_name="snapshot_id"
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "annotations": dict(self.annotations),
            "authority": self.authority.value,
            "captured_at_ms": self.captured_at_ms,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "snapshot_digest": self.snapshot_digest,
            "snapshot_id": self.snapshot_id,
            "store": self.store.to_dict(),
            "transaction_watermark": self.transaction_watermark,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateSnapshot":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state snapshot payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="state snapshot")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "snapshot_id",
                "store",
                "transaction_watermark",
                "snapshot_digest",
                "captured_at_ms",
                "authority",
                "annotations",
                "content_id",
                "cid",
                "identity",
            },
            context="state snapshot",
        )
        store_raw = payload.get("store")
        if not isinstance(store_raw, Mapping):
            raise ControlPlaneContractError(
                "store must be an object",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        result = cls(
            snapshot_id=payload.get("snapshot_id", ""),
            store=ControlPlaneStoreIdentity.from_dict(store_raw),
            transaction_watermark=payload.get("transaction_watermark", 0),
            snapshot_digest=payload.get("snapshot_digest", ""),
            captured_at_ms=payload.get("captured_at_ms", 0),
            authority=payload.get(
                "authority", StateAuthorityClass.AUTHORITATIVE
            ),
            annotations=payload.get("annotations", {}),
        )
        _claimed_content_id(payload, result.content_id, "state snapshot")
        return result


@dataclass(frozen=True)
class StateExportReceipt(_CanonicalRecord):
    """Deterministic export receipt; never grants authoritative write authority."""

    SCHEMA: ClassVar[str] = STATE_EXPORT_RECEIPT_SCHEMA

    export_id: str
    snapshot: StateSnapshot
    profile: ExportProfile
    view_revision: str
    renderer_revision: str
    artifact_digest: str
    destination: str
    parameters: Mapping[str, Any] = MappingProxyType({})
    authority: StateAuthorityClass = StateAuthorityClass.EXPORT
    exported_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "export_id", _stable_id(self.export_id, "export_id")
        )
        if not isinstance(self.snapshot, StateSnapshot):
            raise ControlPlaneContractError(
                "snapshot must be StateSnapshot",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        object.__setattr__(
            self,
            "profile",
            _enum(self.profile, ExportProfile, field_name="profile"),
        )
        object.__setattr__(
            self,
            "view_revision",
            _stable_id(self.view_revision, "view_revision"),
        )
        object.__setattr__(
            self,
            "renderer_revision",
            _stable_id(self.renderer_revision, "renderer_revision"),
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _sha256_digest(self.artifact_digest, "artifact_digest"),
        )
        object.__setattr__(
            self, "destination", _required_text(self.destination, "destination")
        )
        object.__setattr__(
            self,
            "exported_at_ms",
            _non_negative_int(self.exported_at_ms, "exported_at_ms"),
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, StateAuthorityClass, field_name="authority"),
        )
        if self.authority is StateAuthorityClass.AUTHORITATIVE:
            raise ExportAuthorityError(
                "export receipt cannot be labeled authoritative"
            )
        if self.authority is not StateAuthorityClass.EXPORT:
            raise InconsistentIdentityError(
                "export receipt authority must be export"
            )
        parameters = _canonical_mapping(
            dict(self.parameters), field_name="parameters"
        )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
            "authority": self.authority.value,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "destination": self.destination,
            "export_id": self.export_id,
            "exported_at_ms": self.exported_at_ms,
            "parameters": dict(self.parameters),
            "profile": self.profile.value,
            "renderer_revision": self.renderer_revision,
            "snapshot": self.snapshot.to_dict(),
            "view_revision": self.view_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateExportReceipt":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "export receipt payload must be a mapping"
            )
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="export receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "export_id",
                "snapshot",
                "profile",
                "view_revision",
                "renderer_revision",
                "artifact_digest",
                "destination",
                "parameters",
                "authority",
                "exported_at_ms",
                "content_id",
                "cid",
                "identity",
            },
            context="export receipt",
        )
        snapshot_raw = payload.get("snapshot")
        if not isinstance(snapshot_raw, Mapping):
            raise ControlPlaneContractError(
                "snapshot must be an object",
                code=ControlPlaneFailureCode.INVALID_TYPE,
            )
        result = cls(
            export_id=payload.get("export_id", ""),
            snapshot=StateSnapshot.from_dict(snapshot_raw),
            profile=payload.get("profile", ""),
            view_revision=payload.get("view_revision", ""),
            renderer_revision=payload.get("renderer_revision", ""),
            artifact_digest=payload.get("artifact_digest", ""),
            destination=payload.get("destination", ""),
            parameters=payload.get("parameters", {}),
            authority=payload.get("authority", StateAuthorityClass.EXPORT),
            exported_at_ms=payload.get("exported_at_ms", 0),
        )
        _claimed_content_id(payload, result.content_id, "export receipt")
        return result


@dataclass(frozen=True)
class ControlPlaneFailure(_CanonicalRecord):
    """Typed, secret-free failure projection for contract rejections."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_FAILURE_SCHEMA

    code: ControlPlaneFailureCode
    message: str
    subject: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "code", _enum(self.code, ControlPlaneFailureCode, field_name="code")
        )
        message = redact_public_text(_required_text(self.message, "message"))
        if not message:
            raise EmptyIdentityError("message must not be empty after redaction")
        object.__setattr__(self, "message", message[:MAX_ANNOTATION_BYTES])
        object.__setattr__(
            self, "subject", _optional_text(self.subject, "subject")[:256]
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "message": self.message,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlPlaneFailure":
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError("failure payload must be a mapping")
        _require_schema(payload, cls.SCHEMA)
        _reject_secret_keys(payload, context="failure")
        _reject_unknown(
            payload,
            {
                "schema",
                "contract_version",
                "code",
                "message",
                "subject",
                "content_id",
                "cid",
                "identity",
            },
            context="failure",
        )
        result = cls(
            code=payload.get("code", ""),
            message=payload.get("message", ""),
            subject=payload.get("subject", ""),
        )
        _claimed_content_id(payload, result.content_id, "failure")
        return result

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ControlPlaneFailure":
        if isinstance(exc, ControlPlaneContractError):
            return cls(code=exc.code, message=str(exc.message), subject="")
        return cls(
            code=ControlPlaneFailureCode.INVALID_TYPE,
            message=redact_public_text(str(exc) or type(exc).__name__),
        )


def validate_generation_revision_alignment(
    *,
    store_generation: int,
    command_generation: int,
    expected_revision: int,
    observed_revision: int,
) -> None:
    """Fail closed when generation or revision CAS pretends to match."""

    _positive_int(store_generation, "store_generation")
    _positive_int(command_generation, "command_generation")
    _positive_int(expected_revision, "expected_revision")
    _non_negative_int(observed_revision, "observed_revision")
    if command_generation != store_generation:
        raise GenerationMismatchError(
            "command generation does not match store generation"
        )
    if expected_revision != observed_revision:
        raise RevisionMismatchError(
            "expected revision does not match observed revision"
        )


__all__ = (
    "ABSOLUTE_MAX_DEPTH",
    "ABSOLUTE_MAX_ITEMS",
    "ABSOLUTE_MAX_PATHS",
    "ABSOLUTE_MAX_SERIALIZED_BYTES",
    "ABSOLUTE_MAX_TEXT_BYTES",
    "ABSOLUTE_MAX_TIMEOUT_MS",
    "CONTROL_PLANE_BOUNDS_SCHEMA",
    "CONTROL_PLANE_CONTRACT_VERSION",
    "CONTROL_PLANE_FAILURE_SCHEMA",
    "CONTROL_PLANE_STORE_IDENTITY_INTERFACE",
    "CONTROL_PLANE_STORE_IDENTITY_SCHEMA",
    "ControlPlaneBounds",
    "ControlPlaneContractError",
    "ControlPlaneFailure",
    "ControlPlaneFailureCode",
    "ControlPlaneStoreIdentity",
    "EmptyIdentityError",
    "ExportAuthorityError",
    "ExportProfile",
    "FENCE_IDENTITY_SCHEMA",
    "FenceIdentity",
    "ForgedIdentityError",
    "GenerationMismatchError",
    "InconsistentIdentityError",
    "MutableAliasIdentityError",
    "NonFiniteBoundsError",
    "RevisionMismatchError",
    "SCHEMA_IDENTITY_SCHEMA",
    "SECRET_HANDLE_SCHEMA",
    "SESSION_IDENTITY_SCHEMA",
    "STATE_COMMAND_INTERFACE",
    "STATE_COMMAND_SCHEMA",
    "STATE_EXPORT_RECEIPT_INTERFACE",
    "STATE_EXPORT_RECEIPT_SCHEMA",
    "STATE_REVISION_SCHEMA",
    "STATE_SNAPSHOT_INTERFACE",
    "STATE_SNAPSHOT_SCHEMA",
    "STORE_GENERATION_INTERFACE",
    "STORE_GENERATION_SCHEMA",
    "SchemaIdentity",
    "SecretHandle",
    "SecretMaterialError",
    "SessionIdentity",
    "StateAuthorityClass",
    "StateCommand",
    "StateCommandKind",
    "StateExportReceipt",
    "StateRevision",
    "StateSnapshot",
    "StoreGeneration",
    "content_identity",
    "redact_public_text",
    "validate_generation_revision_alignment",
)
