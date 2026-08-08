"""Provider-free control-plane store, schema, identity, and authority contracts.

Closed records define database/store/generation/schema/session/command/revision/
fence/snapshot/export identities, state authority classes, bounds, typed
failures, and redaction helpers.

This module is a pure serialization and validation boundary. Importing it
never opens a filesystem path, database, network socket, provider client, or
child process. Secrets may not appear in durable records; display aliases and
other mutable labels are never identity material; exports cannot claim
authority.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .task_identity import canonical_content_cid, canonical_json_bytes


# ---------------------------------------------------------------------------
# Interface / schema versions
# ---------------------------------------------------------------------------

CONTROL_PLANE_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = CONTROL_PLANE_CONTRACT_VERSION

CONTROL_PLANE_STORE_IDENTITY_INTERFACE: Final[str] = "ControlPlaneStoreIdentity@1"
STORE_GENERATION_INTERFACE: Final[str] = "StoreGeneration@1"
STATE_COMMAND_INTERFACE: Final[str] = "StateCommand@1"
STATE_SNAPSHOT_INTERFACE: Final[str] = "StateSnapshot@1"
STATE_EXPORT_RECEIPT_INTERFACE: Final[str] = "StateExportReceipt@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
CONTROL_PLANE_BOUNDS_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-bounds@1"
SCHEMA_IDENTITY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-schema-identity@1"
STORE_GENERATION_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-store-generation@1"
STORE_IDENTITY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-store-identity@1"
SESSION_IDENTITY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-session-identity@1"
REVISION_TOKEN_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-revision-token@1"
FENCE_TOKEN_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-fence-token@1"
IDEMPOTENCY_BINDING_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/control-plane-idempotency-binding@1"
)
STATE_COMMAND_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-state-command@1"
STATE_SNAPSHOT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/control-plane-state-snapshot@1"
STATE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/control-plane-state-export-receipt@1"
)

# Hard bounds (integer units only; non-finite values are rejected).
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 4_096
MAX_DEPTH: Final[int] = 32
MAX_INT: Final[int] = 2**63 - 1
MAX_PARAMETERS: Final[int] = 256
MAX_SCOPE_PARTS: Final[int] = 64

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_CID_RE: Final = re.compile(r"^b[a-z2-7]{50,}$")
_UUID_RE: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_NAMESPACED_ID_RE: Final = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.-]*(?::[A-Za-z0-9_./:@+-]+)+$"
)
_COMPACT_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+/-]*$")
_ISO_UTC_RE: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$"
)

# Identity fields must never be filled with these mutable labels.
_MUTABLE_ALIAS_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "alias",
        "display",
        "display_alias",
        "display_id",
        "display_name",
        "hostname",
        "host_name",
        "human_name",
        "label",
        "nickname",
        "pid",
        "process_id",
        "pretty_name",
        "short_name",
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
        "prompt",
        "prompt_body",
        "prompt_text",
        "raw_log",
        "refresh_token",
        "secret",
        "secret_handle",
        "session_token",
        "source_body",
        "source_text",
        "token",
    }
)

_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)

_REDACTED_PLACEHOLDER: Final[str] = "[REDACTED]"


# ---------------------------------------------------------------------------
# Typed failures
# ---------------------------------------------------------------------------


class ControlPlaneContractError(ValueError):
    """Base error for malformed or unsafe control-plane contracts."""


class ControlPlaneIdentityError(ControlPlaneContractError):
    """Empty, forged, inconsistent, or non-canonical identity material."""


class ControlPlaneBoundsError(ControlPlaneContractError):
    """A count, byte, depth, or integer limit is non-finite or out of range."""


class ControlPlaneGenerationError(ControlPlaneContractError):
    """Store generation and revision tokens do not form a consistent pair."""


class ControlPlaneSecretError(ControlPlaneContractError):
    """A durable contract contains inline secret-bearing material."""


class ControlPlaneAuthorityError(ControlPlaneContractError):
    """An export or projection claimed unearned authority."""


class ControlPlaneAliasError(ControlPlaneContractError):
    """A mutable display alias was offered as durable identity."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class StateAuthorityClass(str, Enum):
    """Closed authority classification for supervisor state material."""

    AUTHORITY = "authority"
    STATIC_INPUT = "static_input"
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    CACHE = "cache"
    EXPORT = "export"
    OS_BOOTSTRAP = "os_bootstrap"
    EMERGENCY_DIAGNOSTIC = "emergency_diagnostic"

    @property
    def is_authoritative(self) -> bool:
        return self is StateAuthorityClass.AUTHORITY


class CommandKind(str, Enum):
    """Closed set of state-command classes."""

    READ = "read"
    MUTATION = "mutation"
    MAINTENANCE = "maintenance"
    EXPORT = "export"


class ExportProfile(str, Enum):
    """Supported deterministic export render profiles."""

    MARKDOWN_TASKBOARD = "markdown_taskboard"
    JSON_STATUS = "json_status"
    JSONL_AUDIT = "jsonl_audit"
    CSV_ANALYSIS = "csv_analysis"
    PARQUET_ANALYSIS = "parquet_analysis"
    PORTABLE_BUNDLE = "portable_bundle"


# ---------------------------------------------------------------------------
# Primitive validators
# ---------------------------------------------------------------------------


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ControlPlaneContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _reject_non_finite_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControlPlaneBoundsError(
                f"{field_name} must be a finite bound (got non-finite float)"
            )
        raise ControlPlaneBoundsError(
            f"{field_name} must not contain floating-point values"
        )


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise ControlPlaneIdentityError(f"{field_name} must not be empty")
        return ""
    if not isinstance(value, str):
        raise ControlPlaneContractError(f"{field_name} must be a string")
    if value != value.strip():
        raise ControlPlaneContractError(
            f"{field_name} has leading or trailing whitespace"
        )
    if required and not value:
        raise ControlPlaneIdentityError(f"{field_name} must not be empty")
    if "\x00" in value:
        raise ControlPlaneContractError(f"{field_name} must not contain NUL")
    if len(value.encode("utf-8")) > limit:
        raise ControlPlaneBoundsError(f"{field_name} exceeds its byte bound")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise ControlPlaneSecretError(
            f"{field_name} contains inline secret material"
        )
    return value


def _boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ControlPlaneContractError(f"{field_name} must be boolean")
    return value


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INT,
) -> int:
    _reject_non_finite_number(value, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlPlaneBoundsError(f"{field_name} must be a finite integer")
    if value < minimum or value > maximum:
        raise ControlPlaneBoundsError(
            f"{field_name} is outside the supported bound "
            f"[{minimum}, {maximum}]"
        )
    return value


def _positive_int(value: Any, field_name: str) -> int:
    return _bounded_int(value, field_name, minimum=1)


def _non_negative_int(value: Any, field_name: str) -> int:
    return _bounded_int(value, field_name, minimum=0)


def _utc_timestamp(value: Any, field_name: str) -> str:
    text = _text(value, field_name)
    if not _ISO_UTC_RE.fullmatch(text):
        raise ControlPlaneContractError(
            f"{field_name} must be an ISO-8601 UTC timestamp ending in Z"
        )
    return text


def _digest(value: Any, field_name: str, *, required: bool = True) -> str:
    text = _text(value, field_name, required=required)
    if not text and not required:
        return ""
    if not _DIGEST_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{field_name} must be a lowercase sha256:<hex64> digest"
        )
    return text


def _is_mutable_alias_token(value: str) -> bool:
    lowered = value.casefold()
    if lowered.startswith("pid:") or lowered.startswith("process:"):
        return True
    if lowered in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if "/" in value or "\\" in value:
        # Paths are mutable location aliases, never durable identity.
        return True
    if " " in value:
        return True
    if value.isdigit():
        # Bare numeric PIDs / display counters.
        return True
    return False


def _opaque_identity(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    allow_uuid: bool = True,
) -> str:
    """Accept CIDv1, sha256 digest, UUID, or namespaced compact identity."""

    text = _text(value, field_name, required=required)
    if not text and not required:
        return ""
    if _is_mutable_alias_token(text):
        raise ControlPlaneAliasError(
            f"{field_name} must not use a mutable alias as identity"
        )
    field_key = field_name.rsplit(".", 1)[-1].casefold().replace("-", "_")
    if field_key in _MUTABLE_ALIAS_MARKERS or any(
        marker in field_key for marker in ("display", "alias", "hostname", "pid")
    ):
        raise ControlPlaneAliasError(
            f"{field_name} is a mutable alias field and cannot be identity"
        )
    if _DIGEST_RE.fullmatch(text) or _CID_RE.fullmatch(text):
        return text
    if allow_uuid and _UUID_RE.fullmatch(text):
        return text
    if _NAMESPACED_ID_RE.fullmatch(text) or _COMPACT_TOKEN_RE.fullmatch(text):
        if any(char.isspace() for char in text):
            raise ControlPlaneIdentityError(
                f"{field_name} must be an opaque compact identifier"
            )
        return text
    raise ControlPlaneIdentityError(
        f"{field_name} must be a CIDv1, sha256 digest, UUID, or namespaced id"
    )


def _secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in _SECRET_KEYS:
        return True
    return any(
        marker in normalized
        for marker in (
            "password",
            "private_key",
            "access_token",
            "api_key",
            "secret",
            "credential",
        )
    )


def _assert_no_secrets(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        _reject_non_finite_number(value, field_name)
        raise ControlPlaneBoundsError(
            f"{field_name} must not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ControlPlaneContractError(
                    f"{field_name} has a non-string key"
                )
            normalized = key.lower().replace("-", "_").strip()
            if _secret_key(normalized):
                raise ControlPlaneSecretError(
                    f"{field_name} contains forbidden secret-bearing field "
                    f"{key!r}"
                )
            _assert_no_secrets(item, f"{field_name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_no_secrets(item, f"{field_name}[{index}]")
    elif isinstance(value, (bytes, bytearray)):
        raise ControlPlaneContractError(
            f"{field_name} may not contain binary bodies"
        )
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise ControlPlaneSecretError(
                f"{field_name} contains inline secret material"
            )


def redact_mapping(value: Any) -> Any:
    """Return a deep copy with secret-bearing keys replaced by a placeholder.

    Non-mapping leaves are preserved. This helper is pure and never writes.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _secret_key(key_text):
                result[key_text] = _REDACTED_PLACEHOLDER
            else:
                result[key_text] = redact_mapping(item)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [redact_mapping(item) for item in value]
    return value


def _freeze_mapping(
    value: Any,
    field_name: str,
    *,
    max_items: int = MAX_PARAMETERS,
    max_depth: int = MAX_DEPTH,
) -> Mapping[str, Any]:
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
            return _bounded_int(item, field_name)
        if isinstance(item, float):
            _reject_non_finite_number(item, field_name)
            raise ControlPlaneBoundsError(
                f"{field_name} must not contain floats"
            )
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, str):
            return _text(item, field_name, required=False)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(item, key=lambda k: str(k)):
                key_text = _text(str(key), f"{field_name} key")
                if _secret_key(key_text):
                    raise ControlPlaneSecretError(
                        f"{field_name} contains forbidden secret-bearing field"
                    )
                result[key_text] = visit(item[key], depth + 1)
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
    frozen = visit(value, 0)
    assert isinstance(frozen, Mapping)
    return frozen


def _scope_parts(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str):
        raw = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray)
    ):
        raw = values
    else:
        raise ControlPlaneContractError(
            f"{field_name} must be a sequence of scope tokens"
        )
    if len(raw) > MAX_SCOPE_PARTS:
        raise ControlPlaneBoundsError(f"{field_name} exceeds its item bound")
    parts: list[str] = []
    for value in raw:
        token = _text(value, field_name)
        if any(char.isspace() for char in token):
            raise ControlPlaneContractError(
                f"{field_name} tokens must be compact"
            )
        if token not in parts:
            parts.append(token)
    if not parts:
        raise ControlPlaneIdentityError(f"{field_name} must not be empty")
    return tuple(parts)


def content_identity(value: Any) -> str:
    """Return the canonical CIDv1 for a control-plane contract payload."""

    return canonical_content_cid(value)


def _verify_claimed_identity(
    claimed: str | None,
    payload: Mapping[str, Any],
    field_name: str,
) -> str:
    expected = content_identity(payload)
    if claimed is None or claimed == "":
        return expected
    claimed_text = _opaque_identity(claimed, field_name)
    if claimed_text != expected:
        raise ControlPlaneIdentityError(
            f"{field_name} is forged or inconsistent with canonical identity"
        )
    return expected


def _bounded_record(payload: Mapping[str, Any], name: str) -> None:
    _assert_no_secrets(payload, name)
    encoded = canonical_json_bytes(payload)
    if len(encoded) > MAX_RECORD_BYTES:
        raise ControlPlaneBoundsError(f"{name} exceeds the record byte bound")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneBounds:
    """Integer resource bounds for control-plane records and commands."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_BOUNDS_SCHEMA

    max_record_bytes: int = MAX_RECORD_BYTES
    max_text_bytes: int = MAX_TEXT_BYTES
    max_reference_count: int = MAX_REFERENCE_COUNT
    max_depth: int = MAX_DEPTH
    max_parameters: int = MAX_PARAMETERS
    max_command_effects: int = 64
    max_export_artifacts: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_record_bytes",
            _positive_int(self.max_record_bytes, "max_record_bytes"),
        )
        object.__setattr__(
            self,
            "max_text_bytes",
            _positive_int(self.max_text_bytes, "max_text_bytes"),
        )
        object.__setattr__(
            self,
            "max_reference_count",
            _positive_int(self.max_reference_count, "max_reference_count"),
        )
        object.__setattr__(
            self, "max_depth", _positive_int(self.max_depth, "max_depth")
        )
        object.__setattr__(
            self,
            "max_parameters",
            _positive_int(self.max_parameters, "max_parameters"),
        )
        object.__setattr__(
            self,
            "max_command_effects",
            _positive_int(self.max_command_effects, "max_command_effects"),
        )
        object.__setattr__(
            self,
            "max_export_artifacts",
            _positive_int(self.max_export_artifacts, "max_export_artifacts"),
        )
        _bounded_record(self.to_dict(), "ControlPlaneBounds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "max_record_bytes": self.max_record_bytes,
            "max_text_bytes": self.max_text_bytes,
            "max_reference_count": self.max_reference_count,
            "max_depth": self.max_depth,
            "max_parameters": self.max_parameters,
            "max_command_effects": self.max_command_effects,
            "max_export_artifacts": self.max_export_artifacts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlPlaneBounds:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError("bounds payload must be a mapping")
        return cls(
            max_record_bytes=payload.get("max_record_bytes", MAX_RECORD_BYTES),
            max_text_bytes=payload.get("max_text_bytes", MAX_TEXT_BYTES),
            max_reference_count=payload.get(
                "max_reference_count", MAX_REFERENCE_COUNT
            ),
            max_depth=payload.get("max_depth", MAX_DEPTH),
            max_parameters=payload.get("max_parameters", MAX_PARAMETERS),
            max_command_effects=payload.get("max_command_effects", 64),
            max_export_artifacts=payload.get("max_export_artifacts", 256),
        )


@dataclass(frozen=True)
class SchemaIdentity:
    """Checksum-bound schema identity for the control-plane store."""

    SCHEMA: ClassVar[str] = SCHEMA_IDENTITY_SCHEMA

    schema_revision: int
    schema_fingerprint: str
    catalog_fingerprint: str
    migration_head: int = 0
    application_version: str = "0.0.0"
    tool_version: str = "0.0.0"
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_revision",
            _non_negative_int(self.schema_revision, "schema_revision"),
        )
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(self.schema_fingerprint, "schema_fingerprint"),
        )
        object.__setattr__(
            self,
            "catalog_fingerprint",
            _digest(self.catalog_fingerprint, "catalog_fingerprint"),
        )
        object.__setattr__(
            self,
            "migration_head",
            _non_negative_int(self.migration_head, "migration_head"),
        )
        object.__setattr__(
            self,
            "application_version",
            _text(self.application_version, "application_version"),
        )
        object.__setattr__(
            self, "tool_version", _text(self.tool_version, "tool_version")
        )
        if self.migration_head > self.schema_revision:
            raise ControlPlaneGenerationError(
                "migration_head cannot exceed schema_revision"
            )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "SchemaIdentity")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "schema_revision": self.schema_revision,
            "schema_fingerprint": self.schema_fingerprint,
            "catalog_fingerprint": self.catalog_fingerprint,
            "migration_head": self.migration_head,
            "application_version": self.application_version,
            "tool_version": self.tool_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SchemaIdentity:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "schema identity payload must be a mapping"
            )
        return cls(
            schema_revision=payload.get("schema_revision", 0),
            schema_fingerprint=payload["schema_fingerprint"],
            catalog_fingerprint=payload["catalog_fingerprint"],
            migration_head=payload.get(
                "migration_head", payload.get("schema_revision", 0)
            ),
            application_version=payload.get("application_version", "0.0.0"),
            tool_version=payload.get("tool_version", "0.0.0"),
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class StoreGeneration:
    """Monotonic store generation: credential rotation and startup epoch."""

    SCHEMA: ClassVar[str] = STORE_GENERATION_SCHEMA
    INTERFACE: ClassVar[str] = STORE_GENERATION_INTERFACE

    generation: int
    credential_generation: int
    startup_epoch: int
    fencing_epoch: int = 1
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(
            self,
            "credential_generation",
            _positive_int(self.credential_generation, "credential_generation"),
        )
        object.__setattr__(
            self,
            "startup_epoch",
            _positive_int(self.startup_epoch, "startup_epoch"),
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive_int(self.fencing_epoch, "fencing_epoch"),
        )
        # Credential rotation cannot outrun the parent store generation.
        if self.credential_generation > self.generation:
            raise ControlPlaneGenerationError(
                "credential_generation cannot exceed store generation"
            )
        if self.startup_epoch > self.generation:
            raise ControlPlaneGenerationError(
                "startup_epoch cannot exceed store generation"
            )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "StoreGeneration")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "generation": self.generation,
            "credential_generation": self.credential_generation,
            "startup_epoch": self.startup_epoch,
            "fencing_epoch": self.fencing_epoch,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    def matches_revision(self, revision: int) -> bool:
        """Return whether ``revision`` is admissible under this generation."""

        try:
            value = _non_negative_int(revision, "revision")
        except ControlPlaneContractError:
            return False
        # Revisions are generation-local counters; zero is the empty store.
        return value >= 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StoreGeneration:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store generation payload must be a mapping"
            )
        return cls(
            generation=payload["generation"],
            credential_generation=payload.get(
                "credential_generation", payload["generation"]
            ),
            startup_epoch=payload.get("startup_epoch", payload["generation"]),
            fencing_epoch=payload.get("fencing_epoch", 1),
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class ControlPlaneStoreIdentity:
    """Canonical identity of one control-plane database/store instance.

    Display aliases, hostnames, PIDs, and filesystem paths are intentionally
    absent from identity material. Location is bound only through digests.
    """

    SCHEMA: ClassVar[str] = STORE_IDENTITY_SCHEMA
    INTERFACE: ClassVar[str] = CONTROL_PLANE_STORE_IDENTITY_INTERFACE

    repository_id: str
    database_uuid: str
    schema: SchemaIdentity
    generation: StoreGeneration
    store_path_digest: str = ""
    extension_fingerprint: str = ""
    listen_uri_digest: str = ""
    process_birth_id: str = ""
    authority_class: StateAuthorityClass = StateAuthorityClass.AUTHORITY
    display_alias: str = ""
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_id",
            _opaque_identity(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "database_uuid",
            _opaque_identity(self.database_uuid, "database_uuid"),
        )
        if not isinstance(self.schema, SchemaIdentity):
            if isinstance(self.schema, Mapping):
                object.__setattr__(
                    self, "schema", SchemaIdentity.from_dict(self.schema)
                )
            else:
                raise ControlPlaneContractError(
                    "schema must be a SchemaIdentity"
                )
        if not isinstance(self.generation, StoreGeneration):
            if isinstance(self.generation, Mapping):
                object.__setattr__(
                    self,
                    "generation",
                    StoreGeneration.from_dict(self.generation),
                )
            else:
                raise ControlPlaneContractError(
                    "generation must be a StoreGeneration"
                )
        object.__setattr__(
            self,
            "store_path_digest",
            _digest(self.store_path_digest, "store_path_digest", required=False),
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
            "listen_uri_digest",
            _digest(self.listen_uri_digest, "listen_uri_digest", required=False),
        )
        object.__setattr__(
            self,
            "process_birth_id",
            _opaque_identity(
                self.process_birth_id, "process_birth_id", required=False
            ),
        )
        authority = _enum(
            self.authority_class, StateAuthorityClass, "authority_class"
        )
        object.__setattr__(self, "authority_class", authority)
        if authority is not StateAuthorityClass.AUTHORITY:
            raise ControlPlaneAuthorityError(
                "ControlPlaneStoreIdentity authority_class must be authority"
            )
        # Display alias is optional provenance only; never part of identity.
        alias = _text(self.display_alias, "display_alias", required=False)
        object.__setattr__(self, "display_alias", alias)

        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "ControlPlaneStoreIdentity")

    @property
    def store_id(self) -> str:
        """Derived stable store key (content identity of core binding)."""

        return self.content_id

    @property
    def schema_revision(self) -> int:
        return self.schema.schema_revision

    @property
    def store_generation(self) -> int:
        return self.generation.generation

    def _identity_payload(self) -> dict[str, Any]:
        # display_alias is intentionally excluded from identity bytes.
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "repository_id": self.repository_id,
            "database_uuid": self.database_uuid,
            "schema_identity": self.schema._identity_payload(),
            "generation": self.generation._identity_payload(),
            "store_path_digest": self.store_path_digest,
            "extension_fingerprint": self.extension_fingerprint,
            "listen_uri_digest": self.listen_uri_digest,
            "process_birth_id": self.process_birth_id,
            "authority_class": self.authority_class.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "display_alias": self.display_alias,
            "content_id": self.content_id,
            "store_id": self.store_id,
        }

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlPlaneStoreIdentity:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "store identity payload must be a mapping"
            )
        schema_payload = payload.get("schema_identity", payload.get("schema"))
        generation_payload = payload.get("generation")
        return cls(
            repository_id=payload["repository_id"],
            database_uuid=payload["database_uuid"],
            schema=schema_payload,
            generation=generation_payload,
            store_path_digest=str(payload.get("store_path_digest") or ""),
            extension_fingerprint=str(
                payload.get("extension_fingerprint") or ""
            ),
            listen_uri_digest=str(payload.get("listen_uri_digest") or ""),
            process_birth_id=str(payload.get("process_birth_id") or ""),
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.AUTHORITY
            ),
            display_alias=str(payload.get("display_alias") or ""),
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class SessionIdentity:
    """Owner session bound to a store generation and fencing epoch."""

    SCHEMA: ClassVar[str] = SESSION_IDENTITY_SCHEMA

    session_id: str
    owner_principal: str
    generation: int
    fencing_epoch: int
    opened_at: str
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", _opaque_identity(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "owner_principal",
            _opaque_identity(self.owner_principal, "owner_principal"),
        )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive_int(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self, "opened_at", _utc_timestamp(self.opened_at, "opened_at")
        )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "SessionIdentity")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "session_id": self.session_id,
            "owner_principal": self.owner_principal,
            "generation": self.generation,
            "fencing_epoch": self.fencing_epoch,
            "opened_at": self.opened_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SessionIdentity:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "session identity payload must be a mapping"
            )
        return cls(
            session_id=payload["session_id"],
            owner_principal=payload["owner_principal"],
            generation=payload["generation"],
            fencing_epoch=payload["fencing_epoch"],
            opened_at=payload["opened_at"],
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class RevisionToken:
    """Compare-and-swap revision bound to a store generation."""

    SCHEMA: ClassVar[str] = REVISION_TOKEN_SCHEMA

    revision: int
    generation: int
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revision", _non_negative_int(self.revision, "revision")
        )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "RevisionToken")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "revision": self.revision,
            "generation": self.generation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    def assert_matches_generation(self, generation: StoreGeneration | int) -> None:
        expected = (
            generation.generation
            if isinstance(generation, StoreGeneration)
            else _positive_int(generation, "generation")
        )
        if self.generation != expected:
            raise ControlPlaneGenerationError(
                "revision generation does not match store generation"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RevisionToken:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "revision token payload must be a mapping"
            )
        return cls(
            revision=payload["revision"],
            generation=payload["generation"],
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class FenceToken:
    """Lease fencing token: epoch + owner session + scope."""

    SCHEMA: ClassVar[str] = FENCE_TOKEN_SCHEMA

    fencing_epoch: int
    session_id: str
    scope: tuple[str, ...]
    generation: int
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fencing_epoch",
            _positive_int(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self, "session_id", _opaque_identity(self.session_id, "session_id")
        )
        object.__setattr__(self, "scope", _scope_parts(self.scope, "scope"))
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "FenceToken")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "fencing_epoch": self.fencing_epoch,
            "session_id": self.session_id,
            "scope": list(self.scope),
            "generation": self.generation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    def assert_matches_generation(self, generation: StoreGeneration | int) -> None:
        expected = (
            generation.generation
            if isinstance(generation, StoreGeneration)
            else _positive_int(generation, "generation")
        )
        if self.generation != expected:
            raise ControlPlaneGenerationError(
                "fence generation does not match store generation"
            )
        if isinstance(generation, StoreGeneration):
            if self.fencing_epoch > generation.fencing_epoch + 1:
                # Fence may lead by at most one during rotation.
                raise ControlPlaneGenerationError(
                    "fence fencing_epoch is ahead of store fencing_epoch"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FenceToken:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "fence token payload must be a mapping"
            )
        return cls(
            fencing_epoch=payload["fencing_epoch"],
            session_id=payload["session_id"],
            scope=payload.get("scope") or (),
            generation=payload["generation"],
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class IdempotencyBinding:
    """Scoped idempotency key for a state command."""

    SCHEMA: ClassVar[str] = IDEMPOTENCY_BINDING_SCHEMA

    key: str
    operation: str
    caller: str
    generation: int
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, "key"))
        if any(char.isspace() for char in self.key):
            raise ControlPlaneContractError(
                "idempotency key must be a compact token"
            )
        object.__setattr__(
            self, "operation", _opaque_identity(self.operation, "operation")
        )
        object.__setattr__(
            self, "caller", _opaque_identity(self.caller, "caller")
        )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "IdempotencyBinding")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "key": self.key,
            "operation": self.operation,
            "caller": self.caller,
            "generation": self.generation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IdempotencyBinding:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "idempotency binding payload must be a mapping"
            )
        return cls(
            key=payload["key"],
            operation=payload["operation"],
            caller=payload["caller"],
            generation=payload["generation"],
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class StateCommand:
    """Fenced, generation-bound state command with optional CAS revision."""

    SCHEMA: ClassVar[str] = STATE_COMMAND_SCHEMA
    INTERFACE: ClassVar[str] = STATE_COMMAND_INTERFACE

    command_id: str
    kind: CommandKind
    store_id: str
    repository_id: str
    database_uuid: str
    generation: int
    expected_revision: int
    issued_at: str
    fence: FenceToken | None = None
    idempotency: IdempotencyBinding | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _opaque_identity(self.command_id, "command_id")
        )
        kind = _enum(self.kind, CommandKind, "kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "store_id", _opaque_identity(self.store_id, "store_id")
        )
        object.__setattr__(
            self,
            "repository_id",
            _opaque_identity(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "database_uuid",
            _opaque_identity(self.database_uuid, "database_uuid"),
        )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation")
        )
        object.__setattr__(
            self,
            "expected_revision",
            _non_negative_int(self.expected_revision, "expected_revision"),
        )
        object.__setattr__(
            self, "issued_at", _utc_timestamp(self.issued_at, "issued_at")
        )

        fence = self.fence
        if fence is not None and not isinstance(fence, FenceToken):
            if isinstance(fence, Mapping):
                fence = FenceToken.from_dict(fence)
            else:
                raise ControlPlaneContractError("fence must be a FenceToken")
        if kind in {CommandKind.MUTATION, CommandKind.MAINTENANCE}:
            if fence is None:
                raise ControlPlaneContractError(
                    f"{kind.value} commands require a fence token"
                )
            fence.assert_matches_generation(self.generation)
        elif fence is not None:
            fence.assert_matches_generation(self.generation)
        object.__setattr__(self, "fence", fence)

        idempotency = self.idempotency
        if idempotency is not None and not isinstance(
            idempotency, IdempotencyBinding
        ):
            if isinstance(idempotency, Mapping):
                idempotency = IdempotencyBinding.from_dict(idempotency)
            else:
                raise ControlPlaneContractError(
                    "idempotency must be an IdempotencyBinding"
                )
        if kind is CommandKind.MUTATION:
            if idempotency is None:
                raise ControlPlaneContractError(
                    "mutation commands require an idempotency binding"
                )
            if idempotency.generation != self.generation:
                raise ControlPlaneGenerationError(
                    "idempotency generation does not match command generation"
                )
        elif idempotency is not None and idempotency.generation != self.generation:
            raise ControlPlaneGenerationError(
                "idempotency generation does not match command generation"
            )
        object.__setattr__(self, "idempotency", idempotency)

        parameters = _freeze_mapping(self.parameters, "parameters")
        object.__setattr__(self, "parameters", parameters)

        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "StateCommand")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "command_id": self.command_id,
            "kind": self.kind.value,
            "store_id": self.store_id,
            "repository_id": self.repository_id,
            "database_uuid": self.database_uuid,
            "generation": self.generation,
            "expected_revision": self.expected_revision,
            "issued_at": self.issued_at,
            "fence": None if self.fence is None else self.fence._identity_payload(),
            "idempotency": (
                None
                if self.idempotency is None
                else self.idempotency._identity_payload()
            ),
            "parameters": dict(self.parameters),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    def assert_matches_store(self, store: ControlPlaneStoreIdentity) -> None:
        """Fail closed when the command targets a different store generation."""

        if self.store_id != store.store_id:
            raise ControlPlaneIdentityError(
                "command store_id does not match store identity"
            )
        if self.repository_id != store.repository_id:
            raise ControlPlaneIdentityError(
                "command repository_id does not match store identity"
            )
        if self.database_uuid != store.database_uuid:
            raise ControlPlaneIdentityError(
                "command database_uuid does not match store identity"
            )
        if self.generation != store.store_generation:
            raise ControlPlaneGenerationError(
                "command generation does not match store generation"
            )
        if self.fence is not None:
            self.fence.assert_matches_generation(store.generation)

    def assert_revision_consistent(self, token: RevisionToken) -> None:
        if token.generation != self.generation:
            raise ControlPlaneGenerationError(
                "revision generation does not match command generation"
            )
        if token.revision != self.expected_revision:
            raise ControlPlaneGenerationError(
                "revision does not match command expected_revision"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StateCommand:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state command payload must be a mapping"
            )
        return cls(
            command_id=payload["command_id"],
            kind=payload["kind"],
            store_id=payload["store_id"],
            repository_id=payload["repository_id"],
            database_uuid=payload["database_uuid"],
            generation=payload["generation"],
            expected_revision=payload["expected_revision"],
            issued_at=payload["issued_at"],
            fence=payload.get("fence"),
            idempotency=payload.get("idempotency"),
            parameters=payload.get("parameters") or {},
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class StateSnapshot:
    """Point-in-time snapshot bound to store identity and transaction watermark."""

    SCHEMA: ClassVar[str] = STATE_SNAPSHOT_SCHEMA
    INTERFACE: ClassVar[str] = STATE_SNAPSHOT_INTERFACE

    snapshot_id: str
    store: ControlPlaneStoreIdentity
    revision: int
    transaction_watermark: str
    captured_at: str
    schema_fingerprint: str = ""
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _opaque_identity(self.snapshot_id, "snapshot_id"),
        )
        if not isinstance(self.store, ControlPlaneStoreIdentity):
            if isinstance(self.store, Mapping):
                object.__setattr__(
                    self,
                    "store",
                    ControlPlaneStoreIdentity.from_dict(self.store),
                )
            else:
                raise ControlPlaneContractError(
                    "store must be a ControlPlaneStoreIdentity"
                )
        object.__setattr__(
            self, "revision", _non_negative_int(self.revision, "revision")
        )
        object.__setattr__(
            self,
            "transaction_watermark",
            _opaque_identity(
                self.transaction_watermark, "transaction_watermark"
            ),
        )
        object.__setattr__(
            self, "captured_at", _utc_timestamp(self.captured_at, "captured_at")
        )
        fingerprint = self.schema_fingerprint or self.store.schema.schema_fingerprint
        object.__setattr__(
            self,
            "schema_fingerprint",
            _digest(fingerprint, "schema_fingerprint"),
        )
        if self.schema_fingerprint != self.store.schema.schema_fingerprint:
            raise ControlPlaneIdentityError(
                "snapshot schema_fingerprint does not match store schema"
            )
        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "StateSnapshot")

    @property
    def generation(self) -> int:
        return self.store.store_generation

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "snapshot_id": self.snapshot_id,
            "store": self.store._identity_payload(),
            "revision": self.revision,
            "transaction_watermark": self.transaction_watermark,
            "captured_at": self.captured_at,
            "schema_fingerprint": self.schema_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    def revision_token(self) -> RevisionToken:
        return RevisionToken(
            revision=self.revision, generation=self.generation
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StateSnapshot:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "state snapshot payload must be a mapping"
            )
        return cls(
            snapshot_id=payload["snapshot_id"],
            store=payload["store"],
            revision=payload["revision"],
            transaction_watermark=payload["transaction_watermark"],
            captured_at=payload["captured_at"],
            schema_fingerprint=str(payload.get("schema_fingerprint") or ""),
            content_id=str(payload.get("content_id") or ""),
        )


@dataclass(frozen=True)
class StateExportReceipt:
    """Deterministic export receipt. Exports are never authoritative state."""

    SCHEMA: ClassVar[str] = STATE_EXPORT_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = STATE_EXPORT_RECEIPT_INTERFACE

    export_id: str
    snapshot: StateSnapshot
    profile: ExportProfile
    renderer_revision: str
    query_revision: str
    parameters_digest: str
    artifact_digest: str
    destination_digest: str
    exported_at: str
    authority_class: StateAuthorityClass = StateAuthorityClass.EXPORT
    is_authoritative: bool = False
    omitted_fields: tuple[str, ...] = ()
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "export_id", _opaque_identity(self.export_id, "export_id")
        )
        if not isinstance(self.snapshot, StateSnapshot):
            if isinstance(self.snapshot, Mapping):
                object.__setattr__(
                    self, "snapshot", StateSnapshot.from_dict(self.snapshot)
                )
            else:
                raise ControlPlaneContractError(
                    "snapshot must be a StateSnapshot"
                )
        profile = _enum(self.profile, ExportProfile, "profile")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(
            self,
            "renderer_revision",
            _opaque_identity(self.renderer_revision, "renderer_revision"),
        )
        object.__setattr__(
            self,
            "query_revision",
            _opaque_identity(self.query_revision, "query_revision"),
        )
        object.__setattr__(
            self,
            "parameters_digest",
            _digest(self.parameters_digest, "parameters_digest"),
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, "artifact_digest"),
        )
        object.__setattr__(
            self,
            "destination_digest",
            _digest(self.destination_digest, "destination_digest"),
        )
        object.__setattr__(
            self, "exported_at", _utc_timestamp(self.exported_at, "exported_at")
        )
        authority = _enum(
            self.authority_class, StateAuthorityClass, "authority_class"
        )
        object.__setattr__(self, "authority_class", authority)
        is_authoritative = _boolean(self.is_authoritative, "is_authoritative")
        object.__setattr__(self, "is_authoritative", is_authoritative)

        # Exports can never be labeled authoritative.
        if is_authoritative:
            raise ControlPlaneAuthorityError(
                "export labeled authoritative is forbidden"
            )
        if authority is StateAuthorityClass.AUTHORITY:
            raise ControlPlaneAuthorityError(
                "export authority_class cannot be authority"
            )
        if authority is not StateAuthorityClass.EXPORT:
            raise ControlPlaneAuthorityError(
                "export authority_class must be export"
            )

        omitted = self.omitted_fields
        if omitted is None:
            omitted_tuple: tuple[str, ...] = ()
        elif isinstance(omitted, str):
            raise ControlPlaneContractError(
                "omitted_fields must be a sequence of field names"
            )
        elif not isinstance(omitted, Sequence):
            raise ControlPlaneContractError(
                "omitted_fields must be a sequence of field names"
            )
        else:
            names: list[str] = []
            for name in omitted:
                text = _text(name, "omitted_fields")
                if text not in names:
                    names.append(text)
            omitted_tuple = tuple(names)
        object.__setattr__(self, "omitted_fields", omitted_tuple)

        payload = self._identity_payload()
        content_id = _verify_claimed_identity(
            self.content_id or None, payload, "content_id"
        )
        object.__setattr__(self, "content_id", content_id)
        _bounded_record(self.to_dict(), "StateExportReceipt")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTROL_PLANE_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "export_id": self.export_id,
            "snapshot": self.snapshot._identity_payload(),
            "profile": self.profile.value,
            "renderer_revision": self.renderer_revision,
            "query_revision": self.query_revision,
            "parameters_digest": self.parameters_digest,
            "artifact_digest": self.artifact_digest,
            "destination_digest": self.destination_digest,
            "exported_at": self.exported_at,
            "authority_class": self.authority_class.value,
            "is_authoritative": self.is_authoritative,
            "omitted_fields": list(self.omitted_fields),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StateExportReceipt:
        if not isinstance(payload, Mapping):
            raise ControlPlaneContractError(
                "export receipt payload must be a mapping"
            )
        return cls(
            export_id=payload["export_id"],
            snapshot=payload["snapshot"],
            profile=payload["profile"],
            renderer_revision=payload["renderer_revision"],
            query_revision=payload["query_revision"],
            parameters_digest=payload["parameters_digest"],
            artifact_digest=payload["artifact_digest"],
            destination_digest=payload["destination_digest"],
            exported_at=payload["exported_at"],
            authority_class=payload.get(
                "authority_class", StateAuthorityClass.EXPORT
            ),
            is_authoritative=payload.get("is_authoritative", False),
            omitted_fields=payload.get("omitted_fields") or (),
            content_id=str(payload.get("content_id") or ""),
        )


def assert_generation_revision_match(
    generation: StoreGeneration | int,
    revision: RevisionToken | int,
) -> None:
    """Fail closed when a revision token is not bound to ``generation``."""

    generation_value = (
        generation.generation
        if isinstance(generation, StoreGeneration)
        else _positive_int(generation, "generation")
    )
    if isinstance(revision, RevisionToken):
        revision.assert_matches_generation(generation_value)
        return
    # Bare revision integers are generation-local; only reject non-finite.
    _non_negative_int(revision, "revision")


def closed_authority_classes() -> tuple[str, ...]:
    return tuple(item.value for item in StateAuthorityClass)


def closed_command_kinds() -> tuple[str, ...]:
    return tuple(item.value for item in CommandKind)


def closed_export_profiles() -> tuple[str, ...]:
    return tuple(item.value for item in ExportProfile)


__all__ = (
    "CONTROL_PLANE_CONTRACT_VERSION",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "CONTROL_PLANE_STORE_IDENTITY_INTERFACE",
    "STORE_GENERATION_INTERFACE",
    "STATE_COMMAND_INTERFACE",
    "STATE_SNAPSHOT_INTERFACE",
    "STATE_EXPORT_RECEIPT_INTERFACE",
    "CONTROL_PLANE_BOUNDS_SCHEMA",
    "SCHEMA_IDENTITY_SCHEMA",
    "STORE_GENERATION_SCHEMA",
    "STORE_IDENTITY_SCHEMA",
    "SESSION_IDENTITY_SCHEMA",
    "REVISION_TOKEN_SCHEMA",
    "FENCE_TOKEN_SCHEMA",
    "IDEMPOTENCY_BINDING_SCHEMA",
    "STATE_COMMAND_SCHEMA",
    "STATE_SNAPSHOT_SCHEMA",
    "STATE_EXPORT_RECEIPT_SCHEMA",
    "MAX_RECORD_BYTES",
    "MAX_TEXT_BYTES",
    "MAX_REFERENCE_COUNT",
    "MAX_DEPTH",
    "MAX_INT",
    "ControlPlaneContractError",
    "ControlPlaneIdentityError",
    "ControlPlaneBoundsError",
    "ControlPlaneGenerationError",
    "ControlPlaneSecretError",
    "ControlPlaneAuthorityError",
    "ControlPlaneAliasError",
    "StateAuthorityClass",
    "CommandKind",
    "ExportProfile",
    "ControlPlaneBounds",
    "SchemaIdentity",
    "StoreGeneration",
    "ControlPlaneStoreIdentity",
    "SessionIdentity",
    "RevisionToken",
    "FenceToken",
    "IdempotencyBinding",
    "StateCommand",
    "StateSnapshot",
    "StateExportReceipt",
    "content_identity",
    "redact_mapping",
    "assert_generation_revision_match",
    "closed_authority_classes",
    "closed_command_kinds",
    "closed_export_profiles",
)
