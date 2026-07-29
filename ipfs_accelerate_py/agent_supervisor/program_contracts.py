"""Versioned expected/observed program contract intermediate representation.

This module is the deterministic IR for program contracts used by the VFS
symbolic-assurance pipeline.  Expectations and observations are distinct
record kinds so implementation behavior can never silently become its own
validation oracle.

Source precedence for *expectations* is closed and ordered:

1. reviewed MCP++/MCP IDL, JSON Schema, typed interfaces, and protocol specs;
2. public signatures, type annotations, and stable exports;
3. executable contract and conformance tests;
4. normative documentation;
5. compatibility manifests and generated SDKs.

Implementation observations may only populate observed contracts.  Conflicting
expectations are reported rather than silently resolved.  Large source, AST,
schema, and witness bodies live in content-addressed artifacts; these records
carry compact facts and references only.

Identities are derived from canonical DAG-JSON and are never caller-supplied.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, TypeVar

from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


PROGRAM_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = PROGRAM_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = PROGRAM_CONTRACT_VERSION

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_CLAUSE_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_PARAMETERS: Final[int] = 128
MAX_ERRORS: Final[int] = 128
MAX_EFFECTS: Final[int] = 128
MAX_CAPABILITIES: Final[int] = 128
MAX_ASSUMPTIONS: Final[int] = 64
MAX_UNSUPPORTED: Final[int] = 64
MAX_REFINEMENTS: Final[int] = 64
MAX_CONFLICTS: Final[int] = 64
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_BUNDLE_BYTES: Final[int] = 1_048_576
MAX_NESTING_DEPTH: Final[int] = 16
MAX_RESOURCE_QUANTITY: Final[int] = 1_000_000_000_000
MILLION: Final[int] = 1_000_000

SYMBOL_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/symbol-identity@1"
)
INTERFACE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/interface-identity@1"
)
SOURCE_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/source-reference@1"
)
TYPE_SHAPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/type-shape@1"
)
PARAMETER_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/parameter-spec@1"
)
RETURN_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/return-spec@1"
)
ERROR_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/error-spec@1"
)
SIDE_EFFECT_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/side-effect-spec@1"
)
CAPABILITY_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/capability-spec@1"
)
AUTHORIZATION_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/authorization-spec@1"
)
IDEMPOTENCE_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/idempotence-spec@1"
)
ORDERING_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/ordering-spec@1"
)
ATOMICITY_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/atomicity-spec@1"
)
CONSISTENCY_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/consistency-spec@1"
)
RESOURCE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/resource-bounds@1"
)
FALLBACK_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/fallback-spec@1"
)
SYNC_ASYNC_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/sync-async-spec@1"
)
APPLICABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/applicability@1"
)
ASSUMPTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/assumption@1"
)
UNSUPPORTED_SEMANTICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/unsupported-semantics@1"
)
CONTRACT_REFINEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/refinement@1"
)
CONTRACT_CONFLICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/conflict@1"
)
EXPECTED_PROGRAM_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/expected@1"
)
OBSERVED_PROGRAM_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/observed@1"
)
PROGRAM_CONTRACT_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract/bundle@1"
)


class ProgramContractError(ContractValidationError):
    """Base error for malformed or unsafe program-contract IR records."""


class ContractBoundsError(ProgramContractError):
    """A compact IR record exceeded an explicit item, text, or byte bound."""


class ForgedIdentityError(ProgramContractError):
    """A caller-supplied identity or derived projection was forged."""


class ForgedSourceError(ProgramContractError):
    """A source role, precedence, or observation was presented as expectation."""


class ContractConflictError(ProgramContractError):
    """Conflicting expectations were silently resolved or suppressed."""


class UnsupportedVersionError(ProgramContractError):
    """A payload declared an unsupported contract or schema version."""


class CircularExpectationError(ProgramContractError):
    """Implementation observation was used as the expectation oracle."""


class SubtypingError(ProgramContractError):
    """A claimed refinement or subtyping relation is invalid."""


class ProgramContractRole(str, Enum):
    """Whether a record states an expectation or an observation."""

    EXPECTED = "expected"
    OBSERVED = "observed"


class ContractSourceKind(str, Enum):
    """Closed vocabulary for contract provenance.

    Lower :meth:`rank` values are higher authority for expectations.  The
    implementation-observation kind is never a valid expectation source.
    """

    REVIEWED_INTERFACE = "reviewed_interface"
    PUBLIC_SIGNATURE = "public_signature"
    CONTRACT_TEST = "contract_test"
    NORMATIVE_DOCUMENTATION = "normative_documentation"
    COMPATIBILITY_MANIFEST = "compatibility_manifest"
    IMPLEMENTATION_OBSERVATION = "implementation_observation"

    @property
    def rank(self) -> int:
        return {
            ContractSourceKind.REVIEWED_INTERFACE: 0,
            ContractSourceKind.PUBLIC_SIGNATURE: 1,
            ContractSourceKind.CONTRACT_TEST: 2,
            ContractSourceKind.NORMATIVE_DOCUMENTATION: 3,
            ContractSourceKind.COMPATIBILITY_MANIFEST: 4,
            ContractSourceKind.IMPLEMENTATION_OBSERVATION: 100,
        }[self]

    @property
    def may_define_expectation(self) -> bool:
        return self is not ContractSourceKind.IMPLEMENTATION_OBSERVATION

    @property
    def is_observation_only(self) -> bool:
        return self is ContractSourceKind.IMPLEMENTATION_OBSERVATION


# Ordered precedence for expectations (excluding observation-only sources).
SOURCE_PRECEDENCE: Final[tuple[ContractSourceKind, ...]] = (
    ContractSourceKind.REVIEWED_INTERFACE,
    ContractSourceKind.PUBLIC_SIGNATURE,
    ContractSourceKind.CONTRACT_TEST,
    ContractSourceKind.NORMATIVE_DOCUMENTATION,
    ContractSourceKind.COMPATIBILITY_MANIFEST,
)


class SemanticAspect(str, Enum):
    """Dimensions covered by the program-contract IR."""

    IDENTITY = "identity"
    SOURCE_PRECEDENCE = "source_precedence"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    SYNC_ASYNC = "sync_async"
    ERRORS = "errors"
    SIDE_EFFECTS = "side_effects"
    CAPABILITIES = "capabilities"
    AUTHORIZATION = "authorization"
    IDEMPOTENCE = "idempotence"
    ORDERING = "ordering"
    ATOMICITY = "atomicity"
    CONSISTENCY = "consistency"
    RESOURCE_BOUNDS = "resource_bounds"
    FALLBACK_DEGRADATION = "fallback_degradation"


class SupportStatus(str, Enum):
    """Whether a semantic aspect is representable in this contract."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ASSUMED = "assumed"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ConfidenceClass(str, Enum):
    """Bounded confidence vocabulary (not a continuous probability)."""

    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class SyncMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    DUAL = "dual"
    UNKNOWN = "unknown"


class ParameterKind(str, Enum):
    POSITIONAL = "positional"
    KEYWORD = "keyword"
    VARIADIC = "variadic"
    BODY = "body"
    HEADER = "header"
    QUERY = "query"
    PATH = "path"
    CONTEXT = "context"
    RETURN = "return"
    OTHER = "other"


class Optionality(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class EffectKind(str, Enum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    STATE_MUTATION = "state_mutation"
    LOGGING = "logging"
    METRICS = "metrics"
    CACHE = "cache"
    CRYPTO = "crypto"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class EffectPolarity(str, Enum):
    FORBIDDEN = "forbidden"
    ALLOWED = "allowed"
    REQUIRED = "required"
    OBSERVED = "observed"
    UNKNOWN = "unknown"


class CapabilityMode(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    NEGOTIATED = "negotiated"
    FORBIDDEN = "forbidden"
    OBSERVED = "observed"
    UNKNOWN = "unknown"


class AuthorizationMode(str, Enum):
    NONE = "none"
    PRINCIPAL = "principal"
    CAPABILITY = "capability"
    POLICY = "policy"
    PATH_SCOPE = "path_scope"
    TOKEN = "token"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class IdempotenceMode(str, Enum):
    PURE = "pure"
    IDEMPOTENT = "idempotent"
    CONDITIONALLY_IDEMPOTENT = "conditionally_idempotent"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


class OrderingMode(str, Enum):
    UNORDERED = "unordered"
    TOTAL = "total"
    PARTIAL = "partial"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    UNKNOWN = "unknown"


class AtomicityMode(str, Enum):
    ATOMIC = "atomic"
    NON_ATOMIC = "non_atomic"
    BEST_EFFORT = "best_effort"
    TRANSACTIONAL = "transactional"
    UNKNOWN = "unknown"


class ConsistencyMode(str, Enum):
    STRONG = "strong"
    EVENTUAL = "eventual"
    CAUSAL = "causal"
    SESSION = "session"
    READ_YOUR_WRITES = "read_your_writes"
    UNKNOWN = "unknown"


class DegradationMode(str, Enum):
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"
    FALLBACK = "fallback"
    DEGRADED_SUCCESS = "degraded_success"
    RETRY = "retry"
    CIRCUIT_BREAK = "circuit_break"
    UNKNOWN = "unknown"


class ConflictKind(str, Enum):
    SOURCE_DISAGREEMENT = "source_disagreement"
    TYPE_MISMATCH = "type_mismatch"
    EFFECT_MISMATCH = "effect_mismatch"
    BOUND_MISMATCH = "bound_mismatch"
    ROLE_VIOLATION = "role_violation"
    PRECEDENCE_COLLISION = "precedence_collision"
    SELF_EXPECTATION = "self_expectation"
    OTHER = "other"


class RefinementRelation(str, Enum):
    EQUIVALENT = "equivalent"
    STRICT_SUBTYPE = "strict_subtype"
    STRICT_SUPERTYPE = "strict_supertype"
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TypeConstructor(str, Enum):
    ANY = "any"
    NEVER = "never"
    NULL = "null"
    BOOL = "bool"
    INT = "int"
    BYTES = "bytes"
    STRING = "string"
    ENUM = "enum"
    ARRAY = "array"
    OBJECT = "object"
    UNION = "union"
    INTERSECTION = "intersection"
    REFERENCE = "reference"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ProgramContractError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise ProgramContractError(f"{field_name} must not be empty")
    if "\x00" in result:
        raise ProgramContractError(f"{field_name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise ContractBoundsError(
            f"{field_name} exceeds {maximum} UTF-8 bytes"
        )
    return result


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgramContractError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramContractError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ContractBoundsError(
            f"{field_name} must be at least {minimum}{suffix}"
        )
    return value


def _optional_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = MAX_RESOURCE_QUANTITY,
) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name=field_name, minimum=minimum, maximum=maximum)


def _enum(value: Any, enum_type: type[E], *, field_name: str) -> E:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ProgramContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_TEXT_BYTES,
) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ProgramContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[str] = []
    for index, value in enumerate(values):
        item = _text(
            value,
            field_name=f"{field_name}[{index}]",
            maximum=item_bytes,
        )
        if item in result:
            raise ProgramContractError(
                f"{field_name} must not contain duplicates"
            )
        result.append(item)
    if required and not result:
        raise ProgramContractError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _sha256(value: Any, *, field_name: str, required: bool = False) -> str:
    result = _text(value, field_name=field_name, required=required).lower()
    if not result:
        return ""
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ProgramContractError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{digest}"


def _check_header(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ProgramContractError("contract payload must be an object")
    if payload.get("schema") not in (None, "", expected_schema):
        raise ProgramContractError(
            f"unsupported schema; expected {expected_schema}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, PROGRAM_CONTRACT_VERSION):
        raise UnsupportedVersionError(
            "unsupported program-contract version"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    if set(payload).difference(allowed):
        raise ProgramContractError(
            f"{artifact_name} contains unsupported fields"
        )


def _check_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ForgedIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _bounded(
    value: CanonicalContract,
    *,
    maximum: int = MAX_RECORD_BYTES,
    artifact_name: str,
) -> None:
    if len(value.canonical_bytes()) > maximum:
        raise ContractBoundsError(
            f"{artifact_name} exceeds {maximum} canonical bytes"
        )


def _record(
    value: Any,
    record_type: type[T],
    *,
    field_name: str,
    optional: bool = False,
) -> T | None:
    if value is None and optional:
        return None
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        return record_type.from_dict(value)  # type: ignore[attr-defined]
    raise ProgramContractError(
        f"{field_name} must be a {record_type.__name__} record"
    )


def _records(
    values: Any,
    record_type: type[T],
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
    preserve_order: bool = False,
) -> tuple[T, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise ProgramContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise ContractBoundsError(f"{field_name} exceeds {maximum} items")
    normalized = tuple(
        _record(item, record_type, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )
    if not normalized:
        return ()
    if hasattr(normalized[0], "content_id"):
        identities = tuple(item.content_id for item in normalized)  # type: ignore[attr-defined]
        if len(identities) != len(set(identities)):
            raise ProgramContractError(
                f"{field_name} contains duplicate identities"
            )
        if not preserve_order:
            return tuple(sorted(normalized, key=lambda item: item.content_id))  # type: ignore[attr-defined]
    return normalized


def _header_fields() -> set[str]:
    return {
        "schema",
        "schema_version",
        "contract_version",
        "content_id",
    }


class _ProgramContract(CanonicalContract):
    """Shared helpers for program-contract IR records."""

    @property
    def schema_version(self) -> int:
        return PROGRAM_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Identity and source records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolIdentity(_ProgramContract):
    """Stable symbol/interface-facing identity within a repository tree."""

    SCHEMA: ClassVar[str] = SYMBOL_IDENTITY_SCHEMA

    repository_id: str
    tree_id: str
    module_path: str
    symbol_name: str
    qualified_name: str = ""
    language: str = ""
    span_start: int | None = None
    span_end: int | None = None
    blob_cid: str = ""

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "module_path", "symbol_name"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, field_name="qualified_name", required=False)
            or f"{self.module_path}:{self.symbol_name}",
        )
        object.__setattr__(
            self,
            "language",
            _text(self.language, field_name="language", required=False),
        )
        object.__setattr__(
            self,
            "blob_cid",
            _text(self.blob_cid, field_name="blob_cid", required=False),
        )
        if self.span_start is not None:
            object.__setattr__(
                self,
                "span_start",
                _integer(self.span_start, field_name="span_start"),
            )
        if self.span_end is not None:
            object.__setattr__(
                self,
                "span_end",
                _integer(self.span_end, field_name="span_end"),
            )
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ProgramContractError("span_end must be >= span_start")
        _bounded(self, artifact_name="symbol identity")

    @property
    def symbol_id(self) -> str:
        return self.content_id

    def binds_same_subject(self, other: "SymbolIdentity") -> bool:
        return (
            self.repository_id == other.repository_id
            and self.tree_id == other.tree_id
            and self.qualified_name == other.qualified_name
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "qualified_name": self.qualified_name,
            "language": self.language,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "blob_cid": self.blob_cid,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "symbol_id": self.symbol_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SymbolIdentity":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "module_path",
            "symbol_name",
            "qualified_name",
            "language",
            "span_start",
            "span_end",
            "blob_cid",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"symbol_id"},
            artifact_name="symbol identity",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            module_path=payload.get("module_path", ""),
            symbol_name=payload.get("symbol_name", ""),
            qualified_name=payload.get("qualified_name", ""),
            language=payload.get("language", ""),
            span_start=payload.get("span_start"),
            span_end=payload.get("span_end"),
            blob_cid=payload.get("blob_cid", ""),
        )
        _check_identity(
            payload,
            result.symbol_id,
            names=("symbol_id", "content_id"),
            artifact_name="symbol identity",
        )
        return result


@dataclass(frozen=True)
class InterfaceIdentity(_ProgramContract):
    """Stable interface surface identity (API, MCP tool, protocol method)."""

    SCHEMA: ClassVar[str] = INTERFACE_IDENTITY_SCHEMA

    interface_name: str
    surface: str
    version: str = ""
    method: str = ""
    protocol: str = ""
    media_type: str = ""
    path_or_uri: str = ""

    def __post_init__(self) -> None:
        for name in ("interface_name", "surface"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        for name in ("version", "method", "protocol", "media_type", "path_or_uri"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        _bounded(self, artifact_name="interface identity")

    @property
    def interface_id(self) -> str:
        return self.content_id

    def binds_same_surface(self, other: "InterfaceIdentity") -> bool:
        return (
            self.interface_name == other.interface_name
            and self.surface == other.surface
            and self.method == other.method
            and self.protocol == other.protocol
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "interface_name": self.interface_name,
            "surface": self.surface,
            "version": self.version,
            "method": self.method,
            "protocol": self.protocol,
            "media_type": self.media_type,
            "path_or_uri": self.path_or_uri,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "interface_id": self.interface_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterfaceIdentity":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "interface_name",
            "surface",
            "version",
            "method",
            "protocol",
            "media_type",
            "path_or_uri",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"interface_id"},
            artifact_name="interface identity",
        )
        result = cls(
            interface_name=payload.get("interface_name", ""),
            surface=payload.get("surface", ""),
            version=payload.get("version", ""),
            method=payload.get("method", ""),
            protocol=payload.get("protocol", ""),
            media_type=payload.get("media_type", ""),
            path_or_uri=payload.get("path_or_uri", ""),
        )
        _check_identity(
            payload,
            result.interface_id,
            names=("interface_id", "content_id"),
            artifact_name="interface identity",
        )
        return result


@dataclass(frozen=True)
class SourceReference(_ProgramContract):
    """Provenance reference for a contract clause.

    Expectation sources must use a kind that may define expectations.
    Observation sources may use implementation_observation.
    """

    SCHEMA: ClassVar[str] = SOURCE_REFERENCE_SCHEMA

    source_kind: ContractSourceKind
    role: ProgramContractRole
    artifact_id: str
    locator: str = ""
    extractor_rule: str = ""
    confidence: ConfidenceClass = ConfidenceClass.HIGH
    sha256: str = ""
    span_start: int | None = None
    span_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_kind",
            _enum(self.source_kind, ContractSourceKind, field_name="source_kind"),
        )
        object.__setattr__(
            self,
            "role",
            _enum(self.role, ProgramContractRole, field_name="role"),
        )
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, field_name="artifact_id")
        )
        for name in ("locator", "extractor_rule"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "confidence",
            _enum(self.confidence, ConfidenceClass, field_name="confidence"),
        )
        object.__setattr__(
            self, "sha256", _sha256(self.sha256, field_name="sha256")
        )
        if self.span_start is not None:
            object.__setattr__(
                self,
                "span_start",
                _integer(self.span_start, field_name="span_start"),
            )
        if self.span_end is not None:
            object.__setattr__(
                self,
                "span_end",
                _integer(self.span_end, field_name="span_end"),
            )
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ProgramContractError("span_end must be >= span_start")
        if (
            self.role is ProgramContractRole.EXPECTED
            and not self.source_kind.may_define_expectation
        ):
            raise ForgedSourceError(
                "implementation observations cannot define expectations"
            )
        if (
            self.role is ProgramContractRole.OBSERVED
            and self.source_kind is not ContractSourceKind.IMPLEMENTATION_OBSERVATION
            and self.confidence is ConfidenceClass.CERTAIN
        ):
            # Observed sources may cite non-observation artifacts for context,
            # but may not claim CERTAIN expectation authority through them.
            pass
        _bounded(self, artifact_name="source reference")

    @property
    def source_id(self) -> str:
        return self.content_id

    @property
    def precedence_rank(self) -> int:
        return self.source_kind.rank

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "source_kind": self.source_kind,
            "role": self.role,
            "artifact_id": self.artifact_id,
            "locator": self.locator,
            "extractor_rule": self.extractor_rule,
            "confidence": self.confidence,
            "sha256": self.sha256,
            "span_start": self.span_start,
            "span_end": self.span_end,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "source_id": self.source_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceReference":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "source_kind",
            "role",
            "artifact_id",
            "locator",
            "extractor_rule",
            "confidence",
            "sha256",
            "span_start",
            "span_end",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"source_id"},
            artifact_name="source reference",
        )
        result = cls(
            source_kind=payload.get("source_kind", ""),
            role=payload.get("role", ""),
            artifact_id=payload.get("artifact_id", ""),
            locator=payload.get("locator", ""),
            extractor_rule=payload.get("extractor_rule", ""),
            confidence=payload.get("confidence", ConfidenceClass.HIGH),
            sha256=payload.get("sha256", ""),
            span_start=payload.get("span_start"),
            span_end=payload.get("span_end"),
        )
        _check_identity(
            payload,
            result.source_id,
            names=("source_id", "content_id"),
            artifact_name="source reference",
        )
        return result


# ---------------------------------------------------------------------------
# Type and IO semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeShape(_ProgramContract):
    """Bounded type shape used by inputs, outputs, and errors."""

    SCHEMA: ClassVar[str] = TYPE_SHAPE_SCHEMA

    constructor: TypeConstructor
    name: str = ""
    nullable: bool = False
    item: "TypeShape | None" = None
    fields: tuple[tuple[str, "TypeShape"], ...] = ()
    alternatives: tuple["TypeShape", ...] = ()
    enum_values: tuple[str, ...] = ()
    reference: str = ""
    constraints: tuple[str, ...] = ()
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "constructor",
            _enum(self.constructor, TypeConstructor, field_name="constructor"),
        )
        object.__setattr__(
            self, "name", _text(self.name, field_name="name", required=False)
        )
        object.__setattr__(
            self, "nullable", _boolean(self.nullable, field_name="nullable")
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        object.__setattr__(
            self,
            "reference",
            _text(self.reference, field_name="reference", required=False),
        )
        object.__setattr__(
            self,
            "constraints",
            _strings(
                self.constraints,
                field_name="constraints",
                preserve_order=True,
                maximum=32,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "enum_values",
            _strings(
                self.enum_values,
                field_name="enum_values",
                preserve_order=True,
                maximum=256,
            ),
        )
        if self.item is not None:
            object.__setattr__(
                self,
                "item",
                _record(self.item, TypeShape, field_name="item"),
            )
        normalized_fields: list[tuple[str, TypeShape]] = []
        raw_fields = self.fields or ()
        if isinstance(raw_fields, Mapping):
            raise ProgramContractError("fields must be a sequence of pairs")
        if not isinstance(raw_fields, Sequence) or isinstance(
            raw_fields, (str, bytes, bytearray)
        ):
            raise ProgramContractError("fields must be a sequence of pairs")
        if len(raw_fields) > MAX_PARAMETERS:
            raise ContractBoundsError(
                f"fields exceeds {MAX_PARAMETERS} items"
            )
        seen_names: set[str] = set()
        for index, entry in enumerate(raw_fields):
            if (
                not isinstance(entry, Sequence)
                or isinstance(entry, (str, bytes, bytearray))
                or len(entry) != 2
            ):
                raise ProgramContractError(
                    f"fields[{index}] must be a (name, TypeShape) pair"
                )
            field_name = _text(entry[0], field_name=f"fields[{index}].name")
            if field_name in seen_names:
                raise ProgramContractError(
                    "fields must not contain duplicate names"
                )
            seen_names.add(field_name)
            field_type = _record(
                entry[1], TypeShape, field_name=f"fields[{index}].type"
            )
            assert field_type is not None
            normalized_fields.append((field_name, field_type))
        object.__setattr__(
            self,
            "fields",
            tuple(sorted(normalized_fields, key=lambda item: item[0])),
        )
        object.__setattr__(
            self,
            "alternatives",
            _records(
                self.alternatives,
                TypeShape,
                field_name="alternatives",
                maximum=32,
            ),
        )
        if self.constructor is TypeConstructor.ARRAY and self.item is None:
            if self.support is SupportStatus.SUPPORTED:
                raise ProgramContractError(
                    "array TypeShape requires an item type"
                )
        if self.constructor is TypeConstructor.UNSUPPORTED:
            object.__setattr__(self, "support", SupportStatus.UNSUPPORTED)
        _bounded(self, artifact_name="type shape")

    @property
    def type_id(self) -> str:
        return self.content_id

    def is_subtype_of(self, other: "TypeShape") -> bool:
        """Return whether this shape is a structural subtype of ``other``."""

        if other.constructor is TypeConstructor.ANY:
            return True
        if self.constructor is TypeConstructor.NEVER:
            return True
        if self.constructor is TypeConstructor.UNSUPPORTED:
            return False
        if other.constructor is TypeConstructor.UNSUPPORTED:
            return False
        if other.constructor is TypeConstructor.UNION:
            return any(self.is_subtype_of(alt) for alt in other.alternatives)
        if self.constructor is TypeConstructor.UNION:
            return all(
                alt.is_subtype_of(other) for alt in self.alternatives
            ) and bool(self.alternatives)
        if self.constructor is TypeConstructor.INTERSECTION:
            return any(
                alt.is_subtype_of(other) for alt in self.alternatives
            ) and bool(self.alternatives)
        if other.constructor is TypeConstructor.INTERSECTION:
            return all(
                self.is_subtype_of(alt) for alt in other.alternatives
            ) and bool(other.alternatives)
        if self.constructor != other.constructor:
            return False
        if self.nullable and not other.nullable:
            return False
        if self.constructor is TypeConstructor.ENUM:
            if not self.enum_values:
                return False
            return set(self.enum_values).issubset(set(other.enum_values)) or (
                not other.enum_values
            )
        if self.constructor is TypeConstructor.ARRAY:
            if self.item is None or other.item is None:
                return self.item is None and other.item is None
            return self.item.is_subtype_of(other.item)
        if self.constructor is TypeConstructor.OBJECT:
            other_fields = dict(other.fields)
            self_fields = dict(self.fields)
            # Width subtyping: extra fields on self are fine; required other
            # fields must be present and covariant.
            for name, other_type in other_fields.items():
                if name not in self_fields:
                    return False
                if not self_fields[name].is_subtype_of(other_type):
                    return False
            return True
        if self.constructor is TypeConstructor.REFERENCE:
            return self.reference == other.reference and bool(self.reference)
        if self.name and other.name and self.name != other.name:
            return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "constructor": self.constructor,
            "name": self.name,
            "nullable": self.nullable,
            "item": None if self.item is None else self.item.to_dict(),
            "fields": [
                {"name": name, "type": shape.to_dict()}
                for name, shape in self.fields
            ],
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "enum_values": list(self.enum_values),
            "reference": self.reference,
            "constraints": list(self.constraints),
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "type_id": self.type_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypeShape":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "constructor",
            "name",
            "nullable",
            "item",
            "fields",
            "alternatives",
            "enum_values",
            "reference",
            "constraints",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"type_id"},
            artifact_name="type shape",
        )
        raw_fields = payload.get("fields") or ()
        pairs: list[tuple[Any, Any]] = []
        for entry in raw_fields:
            if isinstance(entry, Mapping):
                pairs.append((entry.get("name", ""), entry.get("type")))
            elif isinstance(entry, Sequence) and len(entry) == 2:
                pairs.append((entry[0], entry[1]))
            else:
                raise ProgramContractError(
                    "fields entries must be objects or pairs"
                )
        result = cls(
            constructor=payload.get("constructor", ""),
            name=payload.get("name", ""),
            nullable=bool(payload.get("nullable", False)),
            item=payload.get("item"),
            fields=tuple(pairs),
            alternatives=tuple(payload.get("alternatives") or ()),
            enum_values=tuple(payload.get("enum_values") or ()),
            reference=payload.get("reference", ""),
            constraints=tuple(payload.get("constraints") or ()),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.type_id,
            names=("type_id", "content_id"),
            artifact_name="type shape",
        )
        return result


@dataclass(frozen=True)
class ParameterSpec(_ProgramContract):
    """One input or output parameter."""

    SCHEMA: ClassVar[str] = PARAMETER_SPEC_SCHEMA

    name: str
    type_shape: TypeShape
    kind: ParameterKind = ParameterKind.POSITIONAL
    optionality: Optionality = Optionality.REQUIRED
    default_summary: str = ""
    description: str = ""
    position: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, field_name="name"))
        object.__setattr__(
            self,
            "type_shape",
            _record(self.type_shape, TypeShape, field_name="type_shape"),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ParameterKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "optionality",
            _enum(self.optionality, Optionality, field_name="optionality"),
        )
        object.__setattr__(
            self,
            "default_summary",
            _text(
                self.default_summary,
                field_name="default_summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        if self.position is not None:
            object.__setattr__(
                self,
                "position",
                _integer(self.position, field_name="position"),
            )
        _bounded(self, artifact_name="parameter spec")

    @property
    def parameter_id(self) -> str:
        return self.content_id

    def is_input_compatible_with(self, required: "ParameterSpec") -> bool:
        """Contravariant input check: provided type must accept required values."""

        if self.name != required.name and self.position != required.position:
            return False
        # Contravariance: expected (required) input type must be subtype of
        # observed/declared acceptor, i.e. required.is_subtype_of(self).
        return required.type_shape.is_subtype_of(self.type_shape)

    def is_output_compatible_with(self, required: "ParameterSpec") -> bool:
        """Covariant output check: actual type must subtype expected type."""

        return self.type_shape.is_subtype_of(required.type_shape)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "name": self.name,
            "type_shape": self.type_shape.to_dict(),
            "kind": self.kind,
            "optionality": self.optionality,
            "default_summary": self.default_summary,
            "description": self.description,
            "position": self.position,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "parameter_id": self.parameter_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "name",
            "type_shape",
            "kind",
            "optionality",
            "default_summary",
            "description",
            "position",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"parameter_id"},
            artifact_name="parameter spec",
        )
        result = cls(
            name=payload.get("name", ""),
            type_shape=payload.get("type_shape"),
            kind=payload.get("kind", ParameterKind.POSITIONAL),
            optionality=payload.get("optionality", Optionality.REQUIRED),
            default_summary=payload.get("default_summary", ""),
            description=payload.get("description", ""),
            position=payload.get("position"),
        )
        _check_identity(
            payload,
            result.parameter_id,
            names=("parameter_id", "content_id"),
            artifact_name="parameter spec",
        )
        return result


@dataclass(frozen=True)
class ReturnSpec(_ProgramContract):
    """Return/output specification."""

    SCHEMA: ClassVar[str] = RETURN_SPEC_SCHEMA

    type_shape: TypeShape
    optionality: Optionality = Optionality.REQUIRED
    description: str = ""
    multi_value: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type_shape",
            _record(self.type_shape, TypeShape, field_name="type_shape"),
        )
        object.__setattr__(
            self,
            "optionality",
            _enum(self.optionality, Optionality, field_name="optionality"),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "multi_value",
            _boolean(self.multi_value, field_name="multi_value"),
        )
        _bounded(self, artifact_name="return spec")

    @property
    def return_id(self) -> str:
        return self.content_id

    def is_subtype_of(self, other: "ReturnSpec") -> bool:
        return self.type_shape.is_subtype_of(other.type_shape)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "type_shape": self.type_shape.to_dict(),
            "optionality": self.optionality,
            "description": self.description,
            "multi_value": self.multi_value,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "return_id": self.return_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReturnSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "type_shape",
            "optionality",
            "description",
            "multi_value",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"return_id"},
            artifact_name="return spec",
        )
        result = cls(
            type_shape=payload.get("type_shape"),
            optionality=payload.get("optionality", Optionality.REQUIRED),
            description=payload.get("description", ""),
            multi_value=bool(payload.get("multi_value", False)),
        )
        _check_identity(
            payload,
            result.return_id,
            names=("return_id", "content_id"),
            artifact_name="return spec",
        )
        return result


# ---------------------------------------------------------------------------
# Semantic aspect specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorSpec(_ProgramContract):
    """Declared or observed error class."""

    SCHEMA: ClassVar[str] = ERROR_SPEC_SCHEMA

    error_name: str
    error_type: TypeShape | None = None
    code: str = ""
    retriable: bool = False
    conditions: tuple[str, ...] = ()
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "error_name", _text(self.error_name, field_name="error_name")
        )
        if self.error_type is not None:
            object.__setattr__(
                self,
                "error_type",
                _record(self.error_type, TypeShape, field_name="error_type"),
            )
        object.__setattr__(
            self, "code", _text(self.code, field_name="code", required=False)
        )
        object.__setattr__(
            self, "retriable", _boolean(self.retriable, field_name="retriable")
        )
        object.__setattr__(
            self,
            "conditions",
            _strings(
                self.conditions,
                field_name="conditions",
                preserve_order=True,
                maximum=32,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="error spec")

    @property
    def error_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "error_name": self.error_name,
            "error_type": (
                None if self.error_type is None else self.error_type.to_dict()
            ),
            "code": self.code,
            "retriable": self.retriable,
            "conditions": list(self.conditions),
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "error_id": self.error_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ErrorSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "error_name",
            "error_type",
            "code",
            "retriable",
            "conditions",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"error_id"},
            artifact_name="error spec",
        )
        result = cls(
            error_name=payload.get("error_name", ""),
            error_type=payload.get("error_type"),
            code=payload.get("code", ""),
            retriable=bool(payload.get("retriable", False)),
            conditions=tuple(payload.get("conditions") or ()),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.error_id,
            names=("error_id", "content_id"),
            artifact_name="error spec",
        )
        return result


@dataclass(frozen=True)
class SideEffectSpec(_ProgramContract):
    """Declared or observed side effect."""

    SCHEMA: ClassVar[str] = SIDE_EFFECT_SPEC_SCHEMA

    effect_kind: EffectKind
    polarity: EffectPolarity
    target: str = ""
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effect_kind",
            _enum(self.effect_kind, EffectKind, field_name="effect_kind"),
        )
        object.__setattr__(
            self,
            "polarity",
            _enum(self.polarity, EffectPolarity, field_name="polarity"),
        )
        object.__setattr__(
            self,
            "target",
            _text(self.target, field_name="target", required=False),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="side-effect spec")

    @property
    def effect_id(self) -> str:
        return self.content_id

    def is_allowed_by(self, allowance: "SideEffectSpec") -> bool:
        """Whether this observed effect is permitted by an expected allowance."""

        if allowance.polarity is EffectPolarity.FORBIDDEN:
            return self.effect_kind != allowance.effect_kind
        if allowance.effect_kind is EffectKind.NONE:
            return self.effect_kind is EffectKind.NONE
        if self.effect_kind != allowance.effect_kind:
            return False
        if allowance.target and self.target and self.target != allowance.target:
            return False
        return allowance.polarity in {
            EffectPolarity.ALLOWED,
            EffectPolarity.REQUIRED,
            EffectPolarity.OBSERVED,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "effect_kind": self.effect_kind,
            "polarity": self.polarity,
            "target": self.target,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "effect_id": self.effect_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SideEffectSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "effect_kind",
            "polarity",
            "target",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"effect_id"},
            artifact_name="side-effect spec",
        )
        result = cls(
            effect_kind=payload.get("effect_kind", ""),
            polarity=payload.get("polarity", ""),
            target=payload.get("target", ""),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.effect_id,
            names=("effect_id", "content_id"),
            artifact_name="side-effect spec",
        )
        return result


@dataclass(frozen=True)
class CapabilitySpec(_ProgramContract):
    """Required, optional, negotiated, or observed capability."""

    SCHEMA: ClassVar[str] = CAPABILITY_SPEC_SCHEMA

    capability_name: str
    mode: CapabilityMode
    version: str = ""
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_name",
            _text(self.capability_name, field_name="capability_name"),
        )
        object.__setattr__(
            self, "mode", _enum(self.mode, CapabilityMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, field_name="version", required=False),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="capability spec")

    @property
    def capability_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "capability_name": self.capability_name,
            "mode": self.mode,
            "version": self.version,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "capability_id": self.capability_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilitySpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "capability_name",
            "mode",
            "version",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"capability_id"},
            artifact_name="capability spec",
        )
        result = cls(
            capability_name=payload.get("capability_name", ""),
            mode=payload.get("mode", ""),
            version=payload.get("version", ""),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.capability_id,
            names=("capability_id", "content_id"),
            artifact_name="capability spec",
        )
        return result


@dataclass(frozen=True)
class AuthorizationSpec(_ProgramContract):
    """Authorization requirements or observations."""

    SCHEMA: ClassVar[str] = AUTHORIZATION_SPEC_SCHEMA

    mode: AuthorizationMode
    principals: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum(self.mode, AuthorizationMode, field_name="mode"),
        )
        object.__setattr__(
            self,
            "principals",
            _strings(self.principals, field_name="principals", maximum=64),
        )
        object.__setattr__(
            self,
            "scopes",
            _strings(self.scopes, field_name="scopes", maximum=64),
        )
        object.__setattr__(
            self,
            "policies",
            _strings(self.policies, field_name="policies", maximum=64),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="authorization spec")

    @property
    def authorization_id(self) -> str:
        return self.content_id

    def is_refinement_of(self, other: "AuthorizationSpec") -> bool:
        """Tighter authorization (more principals/scopes constrained) refines."""

        if other.mode is AuthorizationMode.NONE:
            return True
        if self.mode is AuthorizationMode.UNKNOWN:
            return False
        if self.mode != other.mode and other.mode is not AuthorizationMode.UNKNOWN:
            return False
        if other.principals and not set(other.principals).issuperset(
            set(self.principals)
        ):
            # Self must not introduce principals outside the expected set when
            # expected constrains the set.
            if self.principals and not set(self.principals).issubset(
                set(other.principals)
            ):
                return False
        if other.scopes and self.scopes:
            return set(self.scopes).issubset(set(other.scopes))
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "principals": list(self.principals),
            "scopes": list(self.scopes),
            "policies": list(self.policies),
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "authorization_id": self.authorization_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "mode",
            "principals",
            "scopes",
            "policies",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"authorization_id"},
            artifact_name="authorization spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            principals=tuple(payload.get("principals") or ()),
            scopes=tuple(payload.get("scopes") or ()),
            policies=tuple(payload.get("policies") or ()),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.authorization_id,
            names=("authorization_id", "content_id"),
            artifact_name="authorization spec",
        )
        return result


@dataclass(frozen=True)
class IdempotenceSpec(_ProgramContract):
    SCHEMA: ClassVar[str] = IDEMPOTENCE_SPEC_SCHEMA

    mode: IdempotenceMode
    key_parameters: tuple[str, ...] = ()
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode",
            _enum(self.mode, IdempotenceMode, field_name="mode"),
        )
        object.__setattr__(
            self,
            "key_parameters",
            _strings(
                self.key_parameters, field_name="key_parameters", maximum=32
            ),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="idempotence spec")

    @property
    def idempotence_id(self) -> str:
        return self.content_id

    def is_refinement_of(self, other: "IdempotenceSpec") -> bool:
        """Stronger idempotence refines weaker promises."""

        order = {
            IdempotenceMode.PURE: 0,
            IdempotenceMode.IDEMPOTENT: 1,
            IdempotenceMode.CONDITIONALLY_IDEMPOTENT: 2,
            IdempotenceMode.NON_IDEMPOTENT: 3,
            IdempotenceMode.UNKNOWN: 4,
        }
        return order[self.mode] <= order[other.mode]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "key_parameters": list(self.key_parameters),
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "idempotence_id": self.idempotence_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdempotenceSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {"mode", "key_parameters", "description", "support"}
        _reject_unknown(
            payload,
            fields | _header_fields() | {"idempotence_id"},
            artifact_name="idempotence spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            key_parameters=tuple(payload.get("key_parameters") or ()),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.idempotence_id,
            names=("idempotence_id", "content_id"),
            artifact_name="idempotence spec",
        )
        return result


@dataclass(frozen=True)
class OrderingSpec(_ProgramContract):
    SCHEMA: ClassVar[str] = ORDERING_SPEC_SCHEMA

    mode: OrderingMode
    related_symbols: tuple[str, ...] = ()
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, OrderingMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "related_symbols",
            _strings(
                self.related_symbols, field_name="related_symbols", maximum=64
            ),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="ordering spec")

    @property
    def ordering_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "related_symbols": list(self.related_symbols),
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "ordering_id": self.ordering_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrderingSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {"mode", "related_symbols", "description", "support"}
        _reject_unknown(
            payload,
            fields | _header_fields() | {"ordering_id"},
            artifact_name="ordering spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            related_symbols=tuple(payload.get("related_symbols") or ()),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.ordering_id,
            names=("ordering_id", "content_id"),
            artifact_name="ordering spec",
        )
        return result


@dataclass(frozen=True)
class AtomicitySpec(_ProgramContract):
    SCHEMA: ClassVar[str] = ATOMICITY_SPEC_SCHEMA

    mode: AtomicityMode
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, AtomicityMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="atomicity spec")

    @property
    def atomicity_id(self) -> str:
        return self.content_id

    def is_refinement_of(self, other: "AtomicitySpec") -> bool:
        order = {
            AtomicityMode.TRANSACTIONAL: 0,
            AtomicityMode.ATOMIC: 1,
            AtomicityMode.BEST_EFFORT: 2,
            AtomicityMode.NON_ATOMIC: 3,
            AtomicityMode.UNKNOWN: 4,
        }
        return order[self.mode] <= order[other.mode]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "atomicity_id": self.atomicity_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomicitySpec":
        _check_header(payload, cls.SCHEMA)
        fields = {"mode", "description", "support"}
        _reject_unknown(
            payload,
            fields | _header_fields() | {"atomicity_id"},
            artifact_name="atomicity spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.atomicity_id,
            names=("atomicity_id", "content_id"),
            artifact_name="atomicity spec",
        )
        return result


@dataclass(frozen=True)
class ConsistencySpec(_ProgramContract):
    SCHEMA: ClassVar[str] = CONSISTENCY_SPEC_SCHEMA

    mode: ConsistencyMode
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, ConsistencyMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="consistency spec")

    @property
    def consistency_id(self) -> str:
        return self.content_id

    def is_refinement_of(self, other: "ConsistencySpec") -> bool:
        order = {
            ConsistencyMode.STRONG: 0,
            ConsistencyMode.CAUSAL: 1,
            ConsistencyMode.SESSION: 2,
            ConsistencyMode.READ_YOUR_WRITES: 3,
            ConsistencyMode.EVENTUAL: 4,
            ConsistencyMode.UNKNOWN: 5,
        }
        return order[self.mode] <= order[other.mode]

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "consistency_id": self.consistency_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsistencySpec":
        _check_header(payload, cls.SCHEMA)
        fields = {"mode", "description", "support"}
        _reject_unknown(
            payload,
            fields | _header_fields() | {"consistency_id"},
            artifact_name="consistency spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.consistency_id,
            names=("consistency_id", "content_id"),
            artifact_name="consistency spec",
        )
        return result


@dataclass(frozen=True)
class ResourceBounds(_ProgramContract):
    """Integer-unit resource bounds (no floats)."""

    SCHEMA: ClassVar[str] = RESOURCE_BOUNDS_SCHEMA

    max_wall_time_ms: int | None = None
    max_cpu_time_ms: int | None = None
    max_memory_bytes: int | None = None
    max_payload_bytes: int | None = None
    max_output_bytes: int | None = None
    max_calls: int | None = None
    max_concurrency: int | None = None
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        for name in (
            "max_wall_time_ms",
            "max_cpu_time_ms",
            "max_memory_bytes",
            "max_payload_bytes",
            "max_output_bytes",
            "max_calls",
            "max_concurrency",
        ):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="resource bounds")

    @property
    def bounds_id(self) -> str:
        return self.content_id

    def is_refinement_of(self, other: "ResourceBounds") -> bool:
        """Tighter (smaller) bounds refine looser bounds."""

        for name in (
            "max_wall_time_ms",
            "max_cpu_time_ms",
            "max_memory_bytes",
            "max_payload_bytes",
            "max_output_bytes",
            "max_calls",
            "max_concurrency",
        ):
            self_val = getattr(self, name)
            other_val = getattr(other, name)
            if other_val is None:
                continue
            if self_val is None:
                return False
            if self_val > other_val:
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_cpu_time_ms": self.max_cpu_time_ms,
            "max_memory_bytes": self.max_memory_bytes,
            "max_payload_bytes": self.max_payload_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_calls": self.max_calls,
            "max_concurrency": self.max_concurrency,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "bounds_id": self.bounds_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceBounds":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "max_wall_time_ms",
            "max_cpu_time_ms",
            "max_memory_bytes",
            "max_payload_bytes",
            "max_output_bytes",
            "max_calls",
            "max_concurrency",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"bounds_id"},
            artifact_name="resource bounds",
        )
        result = cls(
            max_wall_time_ms=payload.get("max_wall_time_ms"),
            max_cpu_time_ms=payload.get("max_cpu_time_ms"),
            max_memory_bytes=payload.get("max_memory_bytes"),
            max_payload_bytes=payload.get("max_payload_bytes"),
            max_output_bytes=payload.get("max_output_bytes"),
            max_calls=payload.get("max_calls"),
            max_concurrency=payload.get("max_concurrency"),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.bounds_id,
            names=("bounds_id", "content_id"),
            artifact_name="resource bounds",
        )
        return result


@dataclass(frozen=True)
class FallbackSpec(_ProgramContract):
    """Fallback and degradation behavior."""

    SCHEMA: ClassVar[str] = FALLBACK_SPEC_SCHEMA

    mode: DegradationMode
    fallback_symbol: str = ""
    fallback_interface: str = ""
    conditions: tuple[str, ...] = ()
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, DegradationMode, field_name="mode")
        )
        object.__setattr__(
            self,
            "fallback_symbol",
            _text(
                self.fallback_symbol,
                field_name="fallback_symbol",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "fallback_interface",
            _text(
                self.fallback_interface,
                field_name="fallback_interface",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "conditions",
            _strings(
                self.conditions,
                field_name="conditions",
                preserve_order=True,
                maximum=32,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        _bounded(self, artifact_name="fallback spec")

    @property
    def fallback_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "fallback_symbol": self.fallback_symbol,
            "fallback_interface": self.fallback_interface,
            "conditions": list(self.conditions),
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "fallback_id": self.fallback_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FallbackSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "mode",
            "fallback_symbol",
            "fallback_interface",
            "conditions",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"fallback_id"},
            artifact_name="fallback spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            fallback_symbol=payload.get("fallback_symbol", ""),
            fallback_interface=payload.get("fallback_interface", ""),
            conditions=tuple(payload.get("conditions") or ()),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.fallback_id,
            names=("fallback_id", "content_id"),
            artifact_name="fallback spec",
        )
        return result


@dataclass(frozen=True)
class SyncAsyncSpec(_ProgramContract):
    SCHEMA: ClassVar[str] = SYNC_ASYNC_SPEC_SCHEMA

    mode: SyncMode
    awaitable: bool = False
    callback_style: bool = False
    description: str = ""
    support: SupportStatus = SupportStatus.SUPPORTED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _enum(self.mode, SyncMode, field_name="mode")
        )
        object.__setattr__(
            self, "awaitable", _boolean(self.awaitable, field_name="awaitable")
        )
        object.__setattr__(
            self,
            "callback_style",
            _boolean(self.callback_style, field_name="callback_style"),
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "support",
            _enum(self.support, SupportStatus, field_name="support"),
        )
        if self.mode is SyncMode.ASYNC and not (
            self.awaitable or self.callback_style
        ):
            # Async may still be supported without explicit style markers.
            pass
        _bounded(self, artifact_name="sync/async spec")

    @property
    def sync_async_id(self) -> str:
        return self.content_id

    def is_compatible_with(self, other: "SyncAsyncSpec") -> bool:
        if other.mode is SyncMode.UNKNOWN or self.mode is SyncMode.UNKNOWN:
            return True
        if other.mode is SyncMode.DUAL:
            return self.mode in {SyncMode.SYNC, SyncMode.ASYNC, SyncMode.DUAL}
        return self.mode is other.mode or self.mode is SyncMode.DUAL

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "awaitable": self.awaitable,
            "callback_style": self.callback_style,
            "description": self.description,
            "support": self.support,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "sync_async_id": self.sync_async_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyncAsyncSpec":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "mode",
            "awaitable",
            "callback_style",
            "description",
            "support",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"sync_async_id"},
            artifact_name="sync/async spec",
        )
        result = cls(
            mode=payload.get("mode", ""),
            awaitable=bool(payload.get("awaitable", False)),
            callback_style=bool(payload.get("callback_style", False)),
            description=payload.get("description", ""),
            support=payload.get("support", SupportStatus.SUPPORTED),
        )
        _check_identity(
            payload,
            result.sync_async_id,
            names=("sync_async_id", "content_id"),
            artifact_name="sync/async spec",
        )
        return result


# ---------------------------------------------------------------------------
# Meta: applicability, assumptions, unsupported, refinement, conflict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Applicability(_ProgramContract):
    """When a contract clause applies."""

    SCHEMA: ClassVar[str] = APPLICABILITY_SCHEMA

    conditions: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    environments: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    always: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "conditions",
            _strings(
                self.conditions,
                field_name="conditions",
                preserve_order=True,
                maximum=32,
                item_bytes=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "surfaces",
            _strings(self.surfaces, field_name="surfaces", maximum=32),
        )
        object.__setattr__(
            self,
            "environments",
            _strings(self.environments, field_name="environments", maximum=32),
        )
        object.__setattr__(
            self,
            "versions",
            _strings(self.versions, field_name="versions", maximum=32),
        )
        object.__setattr__(
            self, "always", _boolean(self.always, field_name="always")
        )
        object.__setattr__(
            self,
            "description",
            _text(
                self.description,
                field_name="description",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        if self.always and (
            self.conditions or self.surfaces or self.environments or self.versions
        ):
            # Conditional applicability must not claim always=True.
            object.__setattr__(self, "always", False)
        _bounded(self, artifact_name="applicability")

    @property
    def applicability_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "conditions": list(self.conditions),
            "surfaces": list(self.surfaces),
            "environments": list(self.environments),
            "versions": list(self.versions),
            "always": self.always,
            "description": self.description,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "applicability_id": self.applicability_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Applicability":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "conditions",
            "surfaces",
            "environments",
            "versions",
            "always",
            "description",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"applicability_id"},
            artifact_name="applicability",
        )
        result = cls(
            conditions=tuple(payload.get("conditions") or ()),
            surfaces=tuple(payload.get("surfaces") or ()),
            environments=tuple(payload.get("environments") or ()),
            versions=tuple(payload.get("versions") or ()),
            always=bool(payload.get("always", True)),
            description=payload.get("description", ""),
        )
        _check_identity(
            payload,
            result.applicability_id,
            names=("applicability_id", "content_id"),
            artifact_name="applicability",
        )
        return result


@dataclass(frozen=True)
class Assumption(_ProgramContract):
    """Explicit assumption required for the contract to hold."""

    SCHEMA: ClassVar[str] = ASSUMPTION_SCHEMA

    statement: str
    aspect: SemanticAspect
    confidence: ConfidenceClass = ConfidenceClass.MEDIUM
    source_artifact_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement",
            _text(
                self.statement,
                field_name="statement",
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "aspect",
            _enum(self.aspect, SemanticAspect, field_name="aspect"),
        )
        object.__setattr__(
            self,
            "confidence",
            _enum(self.confidence, ConfidenceClass, field_name="confidence"),
        )
        object.__setattr__(
            self,
            "source_artifact_id",
            _text(
                self.source_artifact_id,
                field_name="source_artifact_id",
                required=False,
            ),
        )
        _bounded(self, artifact_name="assumption")

    @property
    def assumption_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "statement": self.statement,
            "aspect": self.aspect,
            "confidence": self.confidence,
            "source_artifact_id": self.source_artifact_id,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "assumption_id": self.assumption_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Assumption":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "statement",
            "aspect",
            "confidence",
            "source_artifact_id",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"assumption_id"},
            artifact_name="assumption",
        )
        result = cls(
            statement=payload.get("statement", ""),
            aspect=payload.get("aspect", ""),
            confidence=payload.get("confidence", ConfidenceClass.MEDIUM),
            source_artifact_id=payload.get("source_artifact_id", ""),
        )
        _check_identity(
            payload,
            result.assumption_id,
            names=("assumption_id", "content_id"),
            artifact_name="assumption",
        )
        return result


@dataclass(frozen=True)
class UnsupportedSemantics(_ProgramContract):
    """Aspect that cannot be represented or checked yet."""

    SCHEMA: ClassVar[str] = UNSUPPORTED_SEMANTICS_SCHEMA

    aspect: SemanticAspect
    reason: str
    residual: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aspect",
            _enum(self.aspect, SemanticAspect, field_name="aspect"),
        )
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, field_name="reason", maximum=MAX_CLAUSE_BYTES),
        )
        object.__setattr__(
            self,
            "residual",
            _text(
                self.residual,
                field_name="residual",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        _bounded(self, artifact_name="unsupported semantics")

    @property
    def unsupported_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "aspect": self.aspect,
            "reason": self.reason,
            "residual": self.residual,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "unsupported_id": self.unsupported_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnsupportedSemantics":
        _check_header(payload, cls.SCHEMA)
        fields = {"aspect", "reason", "residual"}
        _reject_unknown(
            payload,
            fields | _header_fields() | {"unsupported_id"},
            artifact_name="unsupported semantics",
        )
        result = cls(
            aspect=payload.get("aspect", ""),
            reason=payload.get("reason", ""),
            residual=payload.get("residual", ""),
        )
        _check_identity(
            payload,
            result.unsupported_id,
            names=("unsupported_id", "content_id"),
            artifact_name="unsupported semantics",
        )
        return result


@dataclass(frozen=True)
class ContractRefinement(_ProgramContract):
    """Declared refinement relation between two expected contracts."""

    SCHEMA: ClassVar[str] = CONTRACT_REFINEMENT_SCHEMA

    base_contract_id: str
    refined_contract_id: str
    relation: RefinementRelation
    aspects: tuple[SemanticAspect, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_contract_id",
            _text(self.base_contract_id, field_name="base_contract_id"),
        )
        object.__setattr__(
            self,
            "refined_contract_id",
            _text(self.refined_contract_id, field_name="refined_contract_id"),
        )
        if self.base_contract_id == self.refined_contract_id:
            raise ProgramContractError(
                "refinement must relate two distinct contracts"
            )
        object.__setattr__(
            self,
            "relation",
            _enum(self.relation, RefinementRelation, field_name="relation"),
        )
        aspects = self.aspects or ()
        if isinstance(aspects, str) or not isinstance(aspects, Sequence):
            raise ProgramContractError("aspects must be a sequence")
        if len(aspects) > MAX_COLLECTION_ITEMS:
            raise ContractBoundsError(
                f"aspects exceeds {MAX_COLLECTION_ITEMS} items"
            )
        normalized = tuple(
            _enum(item, SemanticAspect, field_name=f"aspects[{index}]")
            for index, item in enumerate(aspects)
        )
        # Stable order by aspect value.
        object.__setattr__(
            self,
            "aspects",
            tuple(sorted(set(normalized), key=lambda item: item.value)),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        _bounded(self, artifact_name="contract refinement")

    @property
    def refinement_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "base_contract_id": self.base_contract_id,
            "refined_contract_id": self.refined_contract_id,
            "relation": self.relation,
            "aspects": [item.value for item in self.aspects],
            "summary": self.summary,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "refinement_id": self.refinement_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractRefinement":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "base_contract_id",
            "refined_contract_id",
            "relation",
            "aspects",
            "summary",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"refinement_id"},
            artifact_name="contract refinement",
        )
        result = cls(
            base_contract_id=payload.get("base_contract_id", ""),
            refined_contract_id=payload.get("refined_contract_id", ""),
            relation=payload.get("relation", ""),
            aspects=tuple(payload.get("aspects") or ()),
            summary=payload.get("summary", ""),
        )
        _check_identity(
            payload,
            result.refinement_id,
            names=("refinement_id", "content_id"),
            artifact_name="contract refinement",
        )
        return result


@dataclass(frozen=True)
class ContractConflict(_ProgramContract):
    """Explicit conflict between sources or clauses (never silently resolved)."""

    SCHEMA: ClassVar[str] = CONTRACT_CONFLICT_SCHEMA

    kind: ConflictKind
    aspect: SemanticAspect
    left_source_id: str
    right_source_id: str
    summary: str
    left_summary: str = ""
    right_summary: str = ""
    resolved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ConflictKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "aspect",
            _enum(self.aspect, SemanticAspect, field_name="aspect"),
        )
        object.__setattr__(
            self,
            "left_source_id",
            _text(self.left_source_id, field_name="left_source_id"),
        )
        object.__setattr__(
            self,
            "right_source_id",
            _text(self.right_source_id, field_name="right_source_id"),
        )
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, field_name="summary", maximum=MAX_CLAUSE_BYTES),
        )
        object.__setattr__(
            self,
            "left_summary",
            _text(
                self.left_summary,
                field_name="left_summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "right_summary",
            _text(
                self.right_summary,
                field_name="right_summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self, "resolved", _boolean(self.resolved, field_name="resolved")
        )
        if self.resolved:
            raise ContractConflictError(
                "conflicts must not be marked resolved inside the IR; "
                "report them for explicit adjudication"
            )
        _bounded(self, artifact_name="contract conflict")

    @property
    def conflict_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "kind": self.kind,
            "aspect": self.aspect,
            "left_source_id": self.left_source_id,
            "right_source_id": self.right_source_id,
            "summary": self.summary,
            "left_summary": self.left_summary,
            "right_summary": self.right_summary,
            "resolved": self.resolved,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "conflict_id": self.conflict_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractConflict":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "kind",
            "aspect",
            "left_source_id",
            "right_source_id",
            "summary",
            "left_summary",
            "right_summary",
            "resolved",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"conflict_id"},
            artifact_name="contract conflict",
        )
        result = cls(
            kind=payload.get("kind", ""),
            aspect=payload.get("aspect", ""),
            left_source_id=payload.get("left_source_id", ""),
            right_source_id=payload.get("right_source_id", ""),
            summary=payload.get("summary", ""),
            left_summary=payload.get("left_summary", ""),
            right_summary=payload.get("right_summary", ""),
            resolved=bool(payload.get("resolved", False)),
        )
        _check_identity(
            payload,
            result.conflict_id,
            names=("conflict_id", "content_id"),
            artifact_name="contract conflict",
        )
        return result


# ---------------------------------------------------------------------------
# Top-level expected / observed contracts and bundle
# ---------------------------------------------------------------------------


def _normalize_parameters(
    values: Any, *, field_name: str
) -> tuple[ParameterSpec, ...]:
    return _records(
        values,
        ParameterSpec,
        field_name=field_name,
        maximum=MAX_PARAMETERS,
        preserve_order=True,
    )


def _normalize_errors(values: Any) -> tuple[ErrorSpec, ...]:
    return _records(values, ErrorSpec, field_name="errors", maximum=MAX_ERRORS)


def _normalize_effects(values: Any) -> tuple[SideEffectSpec, ...]:
    return _records(
        values, SideEffectSpec, field_name="side_effects", maximum=MAX_EFFECTS
    )


def _normalize_capabilities(values: Any) -> tuple[CapabilitySpec, ...]:
    return _records(
        values,
        CapabilitySpec,
        field_name="capabilities",
        maximum=MAX_CAPABILITIES,
    )


def _normalize_sources(
    values: Any,
    *,
    expected_role: ProgramContractRole,
    field_name: str = "sources",
) -> tuple[SourceReference, ...]:
    sources = _records(
        values, SourceReference, field_name=field_name, maximum=64
    )
    for source in sources:
        if source.role is not expected_role:
            raise ForgedSourceError(
                f"{field_name} role must be {expected_role.value}"
            )
        if (
            expected_role is ProgramContractRole.EXPECTED
            and not source.source_kind.may_define_expectation
        ):
            raise CircularExpectationError(
                "implementation observations cannot define expectations"
            )
    return sources


def _primary_source_kind(
    sources: Sequence[SourceReference],
) -> ContractSourceKind:
    if not sources:
        raise ProgramContractError("at least one source is required")
    return min(sources, key=lambda item: item.precedence_rank).source_kind


@dataclass(frozen=True)
class ExpectedProgramContract(_ProgramContract):
    """Typed expectation under explicit source precedence.

    Never constructed from implementation observations alone.
    """

    SCHEMA: ClassVar[str] = EXPECTED_PROGRAM_CONTRACT_SCHEMA

    symbol: SymbolIdentity
    interface: InterfaceIdentity
    policy_revision: str
    sources: tuple[SourceReference, ...]
    inputs: tuple[ParameterSpec, ...] = ()
    returns: ReturnSpec | None = None
    errors: tuple[ErrorSpec, ...] = ()
    sync_async: SyncAsyncSpec | None = None
    side_effects: tuple[SideEffectSpec, ...] = ()
    capabilities: tuple[CapabilitySpec, ...] = ()
    authorization: AuthorizationSpec | None = None
    idempotence: IdempotenceSpec | None = None
    ordering: OrderingSpec | None = None
    atomicity: AtomicitySpec | None = None
    consistency: ConsistencySpec | None = None
    resource_bounds: ResourceBounds | None = None
    fallback: FallbackSpec | None = None
    applicability: Applicability | None = None
    assumptions: tuple[Assumption, ...] = ()
    unsupported: tuple[UnsupportedSemantics, ...] = ()
    conflicts: tuple[ContractConflict, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _record(self.symbol, SymbolIdentity, field_name="symbol"),
        )
        object.__setattr__(
            self,
            "interface",
            _record(self.interface, InterfaceIdentity, field_name="interface"),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, field_name="policy_revision"),
        )
        object.__setattr__(
            self,
            "sources",
            _normalize_sources(
                self.sources, expected_role=ProgramContractRole.EXPECTED
            ),
        )
        if not self.sources:
            raise ProgramContractError(
                "expected contracts require at least one non-observation source"
            )
        object.__setattr__(
            self, "inputs", _normalize_parameters(self.inputs, field_name="inputs")
        )
        if self.returns is not None:
            object.__setattr__(
                self,
                "returns",
                _record(self.returns, ReturnSpec, field_name="returns"),
            )
        object.__setattr__(self, "errors", _normalize_errors(self.errors))
        if self.sync_async is not None:
            object.__setattr__(
                self,
                "sync_async",
                _record(self.sync_async, SyncAsyncSpec, field_name="sync_async"),
            )
        object.__setattr__(
            self, "side_effects", _normalize_effects(self.side_effects)
        )
        object.__setattr__(
            self, "capabilities", _normalize_capabilities(self.capabilities)
        )
        for name, typ in (
            ("authorization", AuthorizationSpec),
            ("idempotence", IdempotenceSpec),
            ("ordering", OrderingSpec),
            ("atomicity", AtomicitySpec),
            ("consistency", ConsistencySpec),
            ("resource_bounds", ResourceBounds),
            ("fallback", FallbackSpec),
            ("applicability", Applicability),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _record(value, typ, field_name=name)
                )
        object.__setattr__(
            self,
            "assumptions",
            _records(
                self.assumptions,
                Assumption,
                field_name="assumptions",
                maximum=MAX_ASSUMPTIONS,
            ),
        )
        object.__setattr__(
            self,
            "unsupported",
            _records(
                self.unsupported,
                UnsupportedSemantics,
                field_name="unsupported",
                maximum=MAX_UNSUPPORTED,
            ),
        )
        object.__setattr__(
            self,
            "conflicts",
            _records(
                self.conflicts,
                ContractConflict,
                field_name="conflicts",
                maximum=MAX_CONFLICTS,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        _bounded(self, artifact_name="expected program contract")

    @property
    def role(self) -> ProgramContractRole:
        return ProgramContractRole.EXPECTED

    @property
    def expected_contract_id(self) -> str:
        return self.content_id

    @property
    def primary_source_kind(self) -> ContractSourceKind:
        return _primary_source_kind(self.sources)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    def aspect_support(self, aspect: SemanticAspect) -> SupportStatus:
        for item in self.unsupported:
            if item.aspect is aspect:
                return SupportStatus.UNSUPPORTED
        mapping: dict[SemanticAspect, Any] = {
            SemanticAspect.IDENTITY: self.symbol,
            SemanticAspect.SOURCE_PRECEDENCE: self.sources,
            SemanticAspect.INPUTS: self.inputs,
            SemanticAspect.OUTPUTS: self.returns,
            SemanticAspect.SYNC_ASYNC: self.sync_async,
            SemanticAspect.ERRORS: self.errors,
            SemanticAspect.SIDE_EFFECTS: self.side_effects,
            SemanticAspect.CAPABILITIES: self.capabilities,
            SemanticAspect.AUTHORIZATION: self.authorization,
            SemanticAspect.IDEMPOTENCE: self.idempotence,
            SemanticAspect.ORDERING: self.ordering,
            SemanticAspect.ATOMICITY: self.atomicity,
            SemanticAspect.CONSISTENCY: self.consistency,
            SemanticAspect.RESOURCE_BOUNDS: self.resource_bounds,
            SemanticAspect.FALLBACK_DEGRADATION: self.fallback,
        }
        value = mapping.get(aspect)
        if value is None or value == () or value is False:
            return SupportStatus.UNKNOWN
        return SupportStatus.SUPPORTED

    def is_refinement_of(self, other: "ExpectedProgramContract") -> bool:
        """Whether this expected contract refines ``other`` (compatible subtype)."""

        if not self.symbol.binds_same_subject(other.symbol):
            return False
        if not self.interface.binds_same_surface(other.interface):
            return False
        if self.policy_revision != other.policy_revision:
            return False
        # Inputs: contravariant — base required inputs must be accepted.
        other_inputs = {param.name: param for param in other.inputs}
        self_inputs = {param.name: param for param in self.inputs}
        for name, required in other_inputs.items():
            if required.optionality is Optionality.OPTIONAL:
                continue
            if name not in self_inputs:
                return False
            # Contravariance: self parameter type must accept required values,
            # i.e. required.type is subtype of self.type.
            if not required.type_shape.is_subtype_of(
                self_inputs[name].type_shape
            ):
                return False
        if other.returns is not None:
            if self.returns is None:
                return False
            if not self.returns.is_subtype_of(other.returns):
                return False
        if other.sync_async is not None and self.sync_async is not None:
            if not self.sync_async.is_compatible_with(other.sync_async):
                return False
        if other.idempotence is not None and self.idempotence is not None:
            if not self.idempotence.is_refinement_of(other.idempotence):
                return False
        if other.atomicity is not None and self.atomicity is not None:
            if not self.atomicity.is_refinement_of(other.atomicity):
                return False
        if other.consistency is not None and self.consistency is not None:
            if not self.consistency.is_refinement_of(other.consistency):
                return False
        if other.resource_bounds is not None and self.resource_bounds is not None:
            if not self.resource_bounds.is_refinement_of(other.resource_bounds):
                return False
        if other.authorization is not None and self.authorization is not None:
            if not self.authorization.is_refinement_of(other.authorization):
                return False
        # Expected effects: refinement may only remove allowed effects or
        # strengthen forbidden ones; required effects must remain.
        other_required = {
            effect.effect_kind
            for effect in other.side_effects
            if effect.polarity is EffectPolarity.REQUIRED
        }
        self_kinds = {effect.effect_kind for effect in self.side_effects}
        if not other_required.issubset(self_kinds | {EffectKind.NONE}):
            # If other requires an effect, self must still declare it.
            if other_required and not other_required.issubset(
                {
                    effect.effect_kind
                    for effect in self.side_effects
                    if effect.polarity
                    in {EffectPolarity.REQUIRED, EffectPolarity.ALLOWED}
                }
            ):
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "role": ProgramContractRole.EXPECTED.value,
            "symbol": self.symbol.to_dict(),
            "interface": self.interface.to_dict(),
            "policy_revision": self.policy_revision,
            "sources": [source.to_dict() for source in self.sources],
            "inputs": [item.to_dict() for item in self.inputs],
            "returns": None if self.returns is None else self.returns.to_dict(),
            "errors": [item.to_dict() for item in self.errors],
            "sync_async": (
                None if self.sync_async is None else self.sync_async.to_dict()
            ),
            "side_effects": [item.to_dict() for item in self.side_effects],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "authorization": (
                None
                if self.authorization is None
                else self.authorization.to_dict()
            ),
            "idempotence": (
                None if self.idempotence is None else self.idempotence.to_dict()
            ),
            "ordering": (
                None if self.ordering is None else self.ordering.to_dict()
            ),
            "atomicity": (
                None if self.atomicity is None else self.atomicity.to_dict()
            ),
            "consistency": (
                None if self.consistency is None else self.consistency.to_dict()
            ),
            "resource_bounds": (
                None
                if self.resource_bounds is None
                else self.resource_bounds.to_dict()
            ),
            "fallback": (
                None if self.fallback is None else self.fallback.to_dict()
            ),
            "applicability": (
                None
                if self.applicability is None
                else self.applicability.to_dict()
            ),
            "assumptions": [item.to_dict() for item in self.assumptions],
            "unsupported": [item.to_dict() for item in self.unsupported],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "summary": self.summary,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "expected_contract_id": self.expected_contract_id,
            "primary_source_kind": self.primary_source_kind.value,
            "has_conflicts": self.has_conflicts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpectedProgramContract":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "role",
            "symbol",
            "interface",
            "policy_revision",
            "sources",
            "inputs",
            "returns",
            "errors",
            "sync_async",
            "side_effects",
            "capabilities",
            "authorization",
            "idempotence",
            "ordering",
            "atomicity",
            "consistency",
            "resource_bounds",
            "fallback",
            "applicability",
            "assumptions",
            "unsupported",
            "conflicts",
            "summary",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "expected_contract_id",
                "primary_source_kind",
                "has_conflicts",
            },
            artifact_name="expected program contract",
        )
        role = payload.get("role")
        if role not in (None, "", ProgramContractRole.EXPECTED.value):
            raise ForgedSourceError(
                "expected program contract role must be 'expected'"
            )
        result = cls(
            symbol=payload.get("symbol"),
            interface=payload.get("interface"),
            policy_revision=payload.get("policy_revision", ""),
            sources=tuple(payload.get("sources") or ()),
            inputs=tuple(payload.get("inputs") or ()),
            returns=payload.get("returns"),
            errors=tuple(payload.get("errors") or ()),
            sync_async=payload.get("sync_async"),
            side_effects=tuple(payload.get("side_effects") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            authorization=payload.get("authorization"),
            idempotence=payload.get("idempotence"),
            ordering=payload.get("ordering"),
            atomicity=payload.get("atomicity"),
            consistency=payload.get("consistency"),
            resource_bounds=payload.get("resource_bounds"),
            fallback=payload.get("fallback"),
            applicability=payload.get("applicability"),
            assumptions=tuple(payload.get("assumptions") or ()),
            unsupported=tuple(payload.get("unsupported") or ()),
            conflicts=tuple(payload.get("conflicts") or ()),
            summary=payload.get("summary", ""),
        )
        _check_identity(
            payload,
            result.expected_contract_id,
            names=("expected_contract_id", "content_id"),
            artifact_name="expected program contract",
        )
        claimed_kind = payload.get("primary_source_kind")
        if (
            claimed_kind not in (None, "")
            and claimed_kind != result.primary_source_kind.value
        ):
            raise ForgedIdentityError(
                "primary_source_kind does not match derived state"
            )
        claimed_conflicts = payload.get("has_conflicts")
        if claimed_conflicts is not None and (
            not isinstance(claimed_conflicts, bool)
            or claimed_conflicts is not result.has_conflicts
        ):
            raise ForgedIdentityError(
                "has_conflicts does not match derived state"
            )
        return result


@dataclass(frozen=True)
class ObservedProgramContract(_ProgramContract):
    """Behavior observed at one repository observation.

    Observations never define expectations.  They bind to a repository
    observation id so they cannot be reused as self-validating oracles.
    """

    SCHEMA: ClassVar[str] = OBSERVED_PROGRAM_CONTRACT_SCHEMA

    symbol: SymbolIdentity
    interface: InterfaceIdentity
    policy_revision: str
    repository_observation_id: str
    sources: tuple[SourceReference, ...]
    inputs: tuple[ParameterSpec, ...] = ()
    returns: ReturnSpec | None = None
    errors: tuple[ErrorSpec, ...] = ()
    sync_async: SyncAsyncSpec | None = None
    side_effects: tuple[SideEffectSpec, ...] = ()
    capabilities: tuple[CapabilitySpec, ...] = ()
    authorization: AuthorizationSpec | None = None
    idempotence: IdempotenceSpec | None = None
    ordering: OrderingSpec | None = None
    atomicity: AtomicitySpec | None = None
    consistency: ConsistencySpec | None = None
    resource_bounds: ResourceBounds | None = None
    fallback: FallbackSpec | None = None
    applicability: Applicability | None = None
    assumptions: tuple[Assumption, ...] = ()
    unsupported: tuple[UnsupportedSemantics, ...] = ()
    summary: str = ""
    producer_id: str = ""
    producer_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            _record(self.symbol, SymbolIdentity, field_name="symbol"),
        )
        object.__setattr__(
            self,
            "interface",
            _record(self.interface, InterfaceIdentity, field_name="interface"),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, field_name="policy_revision"),
        )
        object.__setattr__(
            self,
            "repository_observation_id",
            _text(
                self.repository_observation_id,
                field_name="repository_observation_id",
            ),
        )
        object.__setattr__(
            self,
            "sources",
            _normalize_sources(
                self.sources, expected_role=ProgramContractRole.OBSERVED
            ),
        )
        if not self.sources:
            raise ProgramContractError(
                "observed contracts require at least one observation source"
            )
        for source in self.sources:
            if source.source_kind is not ContractSourceKind.IMPLEMENTATION_OBSERVATION:
                # Contextual sources may be attached but primary observation
                # semantics must include at least one implementation observation.
                pass
        if not any(
            source.source_kind
            is ContractSourceKind.IMPLEMENTATION_OBSERVATION
            for source in self.sources
        ):
            raise ForgedSourceError(
                "observed contracts require an implementation_observation source"
            )
        object.__setattr__(
            self, "inputs", _normalize_parameters(self.inputs, field_name="inputs")
        )
        if self.returns is not None:
            object.__setattr__(
                self,
                "returns",
                _record(self.returns, ReturnSpec, field_name="returns"),
            )
        object.__setattr__(self, "errors", _normalize_errors(self.errors))
        if self.sync_async is not None:
            object.__setattr__(
                self,
                "sync_async",
                _record(self.sync_async, SyncAsyncSpec, field_name="sync_async"),
            )
        object.__setattr__(
            self, "side_effects", _normalize_effects(self.side_effects)
        )
        object.__setattr__(
            self, "capabilities", _normalize_capabilities(self.capabilities)
        )
        for name, typ in (
            ("authorization", AuthorizationSpec),
            ("idempotence", IdempotenceSpec),
            ("ordering", OrderingSpec),
            ("atomicity", AtomicitySpec),
            ("consistency", ConsistencySpec),
            ("resource_bounds", ResourceBounds),
            ("fallback", FallbackSpec),
            ("applicability", Applicability),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _record(value, typ, field_name=name)
                )
        object.__setattr__(
            self,
            "assumptions",
            _records(
                self.assumptions,
                Assumption,
                field_name="assumptions",
                maximum=MAX_ASSUMPTIONS,
            ),
        )
        object.__setattr__(
            self,
            "unsupported",
            _records(
                self.unsupported,
                UnsupportedSemantics,
                field_name="unsupported",
                maximum=MAX_UNSUPPORTED,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, field_name="producer_id", required=False),
        )
        object.__setattr__(
            self,
            "producer_version",
            _text(
                self.producer_version,
                field_name="producer_version",
                required=False,
            ),
        )
        _bounded(self, artifact_name="observed program contract")

    @property
    def role(self) -> ProgramContractRole:
        return ProgramContractRole.OBSERVED

    @property
    def observed_contract_id(self) -> str:
        return self.content_id

    def binds_same_subject(self, expected: ExpectedProgramContract) -> bool:
        return (
            self.symbol.binds_same_subject(expected.symbol)
            and self.interface.binds_same_surface(expected.interface)
            and self.policy_revision == expected.policy_revision
        )

    def as_expectation_source(self) -> None:
        """Refuse conversion of observations into expectations."""

        raise CircularExpectationError(
            "observed program contracts cannot define expectations"
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "role": ProgramContractRole.OBSERVED.value,
            "symbol": self.symbol.to_dict(),
            "interface": self.interface.to_dict(),
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "sources": [source.to_dict() for source in self.sources],
            "inputs": [item.to_dict() for item in self.inputs],
            "returns": None if self.returns is None else self.returns.to_dict(),
            "errors": [item.to_dict() for item in self.errors],
            "sync_async": (
                None if self.sync_async is None else self.sync_async.to_dict()
            ),
            "side_effects": [item.to_dict() for item in self.side_effects],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "authorization": (
                None
                if self.authorization is None
                else self.authorization.to_dict()
            ),
            "idempotence": (
                None if self.idempotence is None else self.idempotence.to_dict()
            ),
            "ordering": (
                None if self.ordering is None else self.ordering.to_dict()
            ),
            "atomicity": (
                None if self.atomicity is None else self.atomicity.to_dict()
            ),
            "consistency": (
                None if self.consistency is None else self.consistency.to_dict()
            ),
            "resource_bounds": (
                None
                if self.resource_bounds is None
                else self.resource_bounds.to_dict()
            ),
            "fallback": (
                None if self.fallback is None else self.fallback.to_dict()
            ),
            "applicability": (
                None
                if self.applicability is None
                else self.applicability.to_dict()
            ),
            "assumptions": [item.to_dict() for item in self.assumptions],
            "unsupported": [item.to_dict() for item in self.unsupported],
            "summary": self.summary,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "observed_contract_id": self.observed_contract_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservedProgramContract":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "role",
            "symbol",
            "interface",
            "policy_revision",
            "repository_observation_id",
            "sources",
            "inputs",
            "returns",
            "errors",
            "sync_async",
            "side_effects",
            "capabilities",
            "authorization",
            "idempotence",
            "ordering",
            "atomicity",
            "consistency",
            "resource_bounds",
            "fallback",
            "applicability",
            "assumptions",
            "unsupported",
            "summary",
            "producer_id",
            "producer_version",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"observed_contract_id"},
            artifact_name="observed program contract",
        )
        role = payload.get("role")
        if role not in (None, "", ProgramContractRole.OBSERVED.value):
            raise ForgedSourceError(
                "observed program contract role must be 'observed'"
            )
        result = cls(
            symbol=payload.get("symbol"),
            interface=payload.get("interface"),
            policy_revision=payload.get("policy_revision", ""),
            repository_observation_id=payload.get(
                "repository_observation_id", ""
            ),
            sources=tuple(payload.get("sources") or ()),
            inputs=tuple(payload.get("inputs") or ()),
            returns=payload.get("returns"),
            errors=tuple(payload.get("errors") or ()),
            sync_async=payload.get("sync_async"),
            side_effects=tuple(payload.get("side_effects") or ()),
            capabilities=tuple(payload.get("capabilities") or ()),
            authorization=payload.get("authorization"),
            idempotence=payload.get("idempotence"),
            ordering=payload.get("ordering"),
            atomicity=payload.get("atomicity"),
            consistency=payload.get("consistency"),
            resource_bounds=payload.get("resource_bounds"),
            fallback=payload.get("fallback"),
            applicability=payload.get("applicability"),
            assumptions=tuple(payload.get("assumptions") or ()),
            unsupported=tuple(payload.get("unsupported") or ()),
            summary=payload.get("summary", ""),
            producer_id=payload.get("producer_id", ""),
            producer_version=payload.get("producer_version", ""),
        )
        _check_identity(
            payload,
            result.observed_contract_id,
            names=("observed_contract_id", "content_id"),
            artifact_name="observed program contract",
        )
        return result


@dataclass(frozen=True)
class ProgramContractBundle(_ProgramContract):
    """Paired expected/observed contracts with refinements and conflicts.

    The bundle keeps roles separate so satisfaction checking (VFS-016) can
    compare them without allowing either side to rewrite the other.
    """

    SCHEMA: ClassVar[str] = PROGRAM_CONTRACT_BUNDLE_SCHEMA

    repository_id: str
    tree_id: str
    policy_revision: str
    expected: tuple[ExpectedProgramContract, ...] = ()
    observed: tuple[ObservedProgramContract, ...] = ()
    refinements: tuple[ContractRefinement, ...] = ()
    conflicts: tuple[ContractConflict, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "policy_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "expected",
            _records(
                self.expected,
                ExpectedProgramContract,
                field_name="expected",
                maximum=MAX_COLLECTION_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "observed",
            _records(
                self.observed,
                ObservedProgramContract,
                field_name="observed",
                maximum=MAX_COLLECTION_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "refinements",
            _records(
                self.refinements,
                ContractRefinement,
                field_name="refinements",
                maximum=MAX_REFINEMENTS,
            ),
        )
        object.__setattr__(
            self,
            "conflicts",
            _records(
                self.conflicts,
                ContractConflict,
                field_name="conflicts",
                maximum=MAX_CONFLICTS,
            ),
        )
        object.__setattr__(
            self,
            "summary",
            _text(
                self.summary,
                field_name="summary",
                required=False,
                maximum=MAX_CLAUSE_BYTES,
            ),
        )
        for contract in self.expected:
            if contract.symbol.repository_id != self.repository_id:
                raise ProgramContractError(
                    "expected contract repository_id must match bundle"
                )
            if contract.symbol.tree_id != self.tree_id:
                raise ProgramContractError(
                    "expected contract tree_id must match bundle"
                )
            if contract.policy_revision != self.policy_revision:
                raise ProgramContractError(
                    "expected contract policy_revision must match bundle"
                )
        for contract in self.observed:
            if contract.symbol.repository_id != self.repository_id:
                raise ProgramContractError(
                    "observed contract repository_id must match bundle"
                )
            if contract.symbol.tree_id != self.tree_id:
                raise ProgramContractError(
                    "observed contract tree_id must match bundle"
                )
            if contract.policy_revision != self.policy_revision:
                raise ProgramContractError(
                    "observed contract policy_revision must match bundle"
                )
        # Refinements must reference expected contracts present in the bundle.
        expected_ids = {item.expected_contract_id for item in self.expected}
        for refinement in self.refinements:
            if refinement.base_contract_id not in expected_ids:
                raise ProgramContractError(
                    "refinement base_contract_id must reference a bundle expected contract"
                )
            if refinement.refined_contract_id not in expected_ids:
                raise ProgramContractError(
                    "refinement refined_contract_id must reference a bundle expected contract"
                )
        _bounded(
            self,
            maximum=MAX_BUNDLE_BYTES,
            artifact_name="program contract bundle",
        )

    @property
    def bundle_id(self) -> str:
        return self.content_id

    @property
    def has_conflicts(self) -> bool:
        if self.conflicts:
            return True
        return any(item.has_conflicts for item in self.expected)

    def expected_for(
        self, symbol: SymbolIdentity, interface: InterfaceIdentity
    ) -> tuple[ExpectedProgramContract, ...]:
        return tuple(
            item
            for item in self.expected
            if item.symbol.binds_same_subject(symbol)
            and item.interface.binds_same_surface(interface)
        )

    def observed_for(
        self, symbol: SymbolIdentity, interface: InterfaceIdentity
    ) -> tuple[ObservedProgramContract, ...]:
        return tuple(
            item
            for item in self.observed
            if item.symbol.binds_same_subject(symbol)
            and item.interface.binds_same_surface(interface)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "expected": [item.to_dict() for item in self.expected],
            "observed": [item.to_dict() for item in self.observed],
            "refinements": [item.to_dict() for item in self.refinements],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "summary": self.summary,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "bundle_id": self.bundle_id,
            "has_conflicts": self.has_conflicts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramContractBundle":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "repository_id",
            "tree_id",
            "policy_revision",
            "expected",
            "observed",
            "refinements",
            "conflicts",
            "summary",
        }
        _reject_unknown(
            payload,
            fields | _header_fields() | {"bundle_id", "has_conflicts"},
            artifact_name="program contract bundle",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            expected=tuple(payload.get("expected") or ()),
            observed=tuple(payload.get("observed") or ()),
            refinements=tuple(payload.get("refinements") or ()),
            conflicts=tuple(payload.get("conflicts") or ()),
            summary=payload.get("summary", ""),
        )
        _check_identity(
            payload,
            result.bundle_id,
            names=("bundle_id", "content_id"),
            artifact_name="program contract bundle",
        )
        claimed = payload.get("has_conflicts")
        if claimed is not None and (
            not isinstance(claimed, bool) or claimed is not result.has_conflicts
        ):
            raise ForgedIdentityError(
                "has_conflicts does not match derived state"
            )
        return result


# ---------------------------------------------------------------------------
# Helpers: precedence, conflict detection, serialization
# ---------------------------------------------------------------------------


def source_precedence_rank(kind: ContractSourceKind | str) -> int:
    """Return the closed precedence rank for a source kind (lower = stronger)."""

    return _enum(kind, ContractSourceKind, field_name="kind").rank


def may_define_expectation(kind: ContractSourceKind | str) -> bool:
    return _enum(kind, ContractSourceKind, field_name="kind").may_define_expectation


def select_dominant_sources(
    sources: Sequence[SourceReference],
) -> tuple[SourceReference, ...]:
    """Return sources sharing the strongest (lowest-rank) expectation kind."""

    expectation_sources = [
        source
        for source in sources
        if source.role is ProgramContractRole.EXPECTED
        and source.source_kind.may_define_expectation
    ]
    if not expectation_sources:
        return ()
    best = min(item.precedence_rank for item in expectation_sources)
    return tuple(
        item for item in expectation_sources if item.precedence_rank == best
    )


def detect_source_conflicts(
    sources: Sequence[SourceReference],
    *,
    aspect: SemanticAspect,
    left_summary: str,
    right_summary: str,
    disagree: bool,
) -> tuple[ContractConflict, ...]:
    """Emit conflicts when equal-precedence sources disagree."""

    if not disagree:
        return ()
    dominant = select_dominant_sources(sources)
    if len(dominant) < 2:
        return ()
    conflicts: list[ContractConflict] = []
    for index, left in enumerate(dominant):
        for right in dominant[index + 1 :]:
            conflicts.append(
                ContractConflict(
                    kind=ConflictKind.PRECEDENCE_COLLISION,
                    aspect=aspect,
                    left_source_id=left.source_id,
                    right_source_id=right.source_id,
                    summary=(
                        f"Equal-precedence sources disagree on {aspect.value}"
                    ),
                    left_summary=left_summary,
                    right_summary=right_summary,
                    resolved=False,
                )
            )
    return tuple(conflicts)


def compare_type_shapes(
    left: TypeShape, right: TypeShape
) -> RefinementRelation:
    """Compare two type shapes under structural subtyping."""

    left_sub = left.is_subtype_of(right)
    right_sub = right.is_subtype_of(left)
    if left_sub and right_sub:
        return RefinementRelation.EQUIVALENT
    if left_sub:
        return RefinementRelation.STRICT_SUBTYPE
    if right_sub:
        return RefinementRelation.STRICT_SUPERTYPE
    return RefinementRelation.INCOMPATIBLE


def compare_expected_contracts(
    refined: ExpectedProgramContract,
    base: ExpectedProgramContract,
) -> RefinementRelation:
    """Compare two expected contracts for refinement/subtyping."""

    if refined.expected_contract_id == base.expected_contract_id:
        return RefinementRelation.EQUIVALENT
    if refined.is_refinement_of(base) and base.is_refinement_of(refined):
        return RefinementRelation.EQUIVALENT
    if refined.is_refinement_of(base):
        return RefinementRelation.STRICT_SUBTYPE
    if base.is_refinement_of(refined):
        return RefinementRelation.STRICT_SUPERTYPE
    # Partial aspect compatibility without full refinement.
    if (
        refined.symbol.binds_same_subject(base.symbol)
        and refined.interface.binds_same_surface(base.interface)
    ):
        return RefinementRelation.INCOMPATIBLE
    return RefinementRelation.UNKNOWN


def reject_observation_as_expectation(
    observed: ObservedProgramContract,
) -> None:
    """Fail closed if a caller attempts to use observations as expectations."""

    observed.as_expectation_source()


def canonical_program_contract_json_bytes(payload: Any) -> bytes:
    """Encode deterministic DAG-JSON-compatible UTF-8 bytes for IR values."""

    return canonical_json_bytes(payload)


def program_contract_content_identity(payload: Any) -> str:
    """Return the content identity for a program-contract payload."""

    canonical_program_contract_json_bytes(payload)
    return content_identity(payload)


def all_semantic_aspects() -> tuple[SemanticAspect, ...]:
    return tuple(SemanticAspect)


def all_expectation_source_kinds() -> tuple[ContractSourceKind, ...]:
    return SOURCE_PRECEDENCE


__all__ = [
    "PROGRAM_CONTRACT_VERSION",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "MAX_TEXT_BYTES",
    "MAX_CLAUSE_BYTES",
    "MAX_COLLECTION_ITEMS",
    "MAX_RECORD_BYTES",
    "MAX_BUNDLE_BYTES",
    "SOURCE_PRECEDENCE",
    "ProgramContractError",
    "ContractBoundsError",
    "ForgedIdentityError",
    "ForgedSourceError",
    "ContractConflictError",
    "UnsupportedVersionError",
    "CircularExpectationError",
    "SubtypingError",
    "ProgramContractRole",
    "ContractSourceKind",
    "SemanticAspect",
    "SupportStatus",
    "ConfidenceClass",
    "SyncMode",
    "ParameterKind",
    "Optionality",
    "EffectKind",
    "EffectPolarity",
    "CapabilityMode",
    "AuthorizationMode",
    "IdempotenceMode",
    "OrderingMode",
    "AtomicityMode",
    "ConsistencyMode",
    "DegradationMode",
    "ConflictKind",
    "RefinementRelation",
    "TypeConstructor",
    "SymbolIdentity",
    "InterfaceIdentity",
    "SourceReference",
    "TypeShape",
    "ParameterSpec",
    "ReturnSpec",
    "ErrorSpec",
    "SideEffectSpec",
    "CapabilitySpec",
    "AuthorizationSpec",
    "IdempotenceSpec",
    "OrderingSpec",
    "AtomicitySpec",
    "ConsistencySpec",
    "ResourceBounds",
    "FallbackSpec",
    "SyncAsyncSpec",
    "Applicability",
    "Assumption",
    "UnsupportedSemantics",
    "ContractRefinement",
    "ContractConflict",
    "ExpectedProgramContract",
    "ObservedProgramContract",
    "ProgramContractBundle",
    "source_precedence_rank",
    "may_define_expectation",
    "select_dominant_sources",
    "detect_source_conflicts",
    "compare_type_shapes",
    "compare_expected_contracts",
    "reject_observation_as_expectation",
    "canonical_program_contract_json_bytes",
    "program_contract_content_identity",
    "all_semantic_aspects",
    "all_expectation_source_kinds",
]
