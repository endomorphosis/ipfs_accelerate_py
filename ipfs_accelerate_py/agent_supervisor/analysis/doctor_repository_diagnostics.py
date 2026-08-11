"""Compile a real-checkout AST and contract diagnostic evidence snapshot.

Interface: ``DoctorEvidenceCompiler@1`` / ``DoctorEvidenceSnapshot@1``

A bounded checkout (or in-memory source set) is parsed as **inert bytes**
through the existing program AST adapters and analysis AST index.  The
compiler never imports target modules, never writes source trees, and never
invokes an LLM or remote model provider.

Observed structural facts stay separate from expectation source/precedence.
Broken contract traces and structured validation failures are joined to
current AST facts deterministically.  Python-only or unsupported CFG,
reflection, exception, native/FFI, concurrency, and interprocedural analyses
are recorded as explicit open frontiers rather than silent completeness.
"""

from __future__ import annotations

import hashlib
import json
import os
import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .program_ast_adapters import (
    DEFAULT_MAX_SOURCE_BYTES,
    ProgramASTAdapterResult,
    ProgramEvidenceFact,
    SourceDocument,
    adapt_program_source,
    detect_program_language,
)
from ..proof.formal_verification_contracts import content_identity
from .analysis_ast_index import (
    AnalysisASTIndex,
    AnalysisASTIndexError,
    ASTEvidenceKind,
    ASTEvidenceReference,
    build_analysis_ast_index,
)
from .contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    EvidenceReference,
    SourceSpan,
    TraceDisposition,
)


# ---------------------------------------------------------------------------
# Schemas and bounds
# ---------------------------------------------------------------------------

DOCTOR_REPOSITORY_DIAGNOSTICS_INTERFACE: Final[str] = "DoctorEvidenceCompiler@1"
DOCTOR_REPOSITORY_DIAGNOSTICS_VERSION: Final[str] = "1"

DOCTOR_SNAPSHOT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-snapshot-policy@1"
)
DOCTOR_DIAGNOSTIC_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-diagnostic-input@1"
)
DOCTOR_DIAGNOSTIC_FINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-diagnostic-finding@1"
)
DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-evidence-snapshot@1"
)
DOCTOR_QUERY_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-query-result@1"
)
DOCTOR_AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-authority-roots@1"
)

DEFAULT_MAX_PATHS: Final[int] = 4_096
DEFAULT_MAX_FINDINGS: Final[int] = 4_096
DEFAULT_MAX_TOTAL_BYTES: Final[int] = 32 * 1024 * 1024
DEFAULT_MAX_TRACE_JOINS: Final[int] = 1_024
DEFAULT_MAX_VALIDATION_FAILURES: Final[int] = 1_024
DEFAULT_MAX_QUERY_RESULTS: Final[int] = 256
HARD_MAX_PATHS: Final[int] = 100_000
HARD_MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024
HARD_MAX_TOTAL_BYTES: Final[int] = 256 * 1024 * 1024

SUPPORTED_ADAPTER_LANGUAGES: Final[frozenset[str]] = frozenset(
    {
        "python",
        "javascript",
        "jsx",
        "typescript",
        "tsx",
        "json",
        "markdown",
    }
)

# Analyses that remain open frontiers for this deterministic doctor stage.
DEFAULT_OPEN_FRONTIERS: Final[tuple[str, ...]] = (
    "frontier:cfg_control_flow",
    "frontier:dynamic_dispatch",
    "frontier:reflection",
    "frontier:generated_code",
    "frontier:exception_propagation",
    "frontier:native_ffi",
    "frontier:concurrency",
    "frontier:interprocedural_dataflow",
    "frontier:python_only_full_type_inference",
    "frontier:type_analysis",
)

_EXPORT_KINDS: Final[frozenset[str]] = frozenset(
    {"export", "re_export", "function_definition", "async_function_definition", "class_definition"}
)
_ALIAS_RELATIONSHIPS: Final[frozenset[str]] = frozenset({"imports", "aliases", "reexports"})
_WRAPPER_KINDS: Final[frozenset[str]] = frozenset(
    {"decorator", "registration", "callback", "async_context_manager"}
)
_ENTRY_KINDS: Final[frozenset[str]] = frozenset(
    {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "method_definition",
        "export",
        "mcp_server",
        "mcp_manifest",
        "registration",
    }
)
_CALL_KINDS: Final[frozenset[str]] = frozenset(
    {"call", "await", "new_expression", "dynamic_import"}
)
_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
    }
)


# ---------------------------------------------------------------------------
# Errors and closed vocabularies
# ---------------------------------------------------------------------------


class DoctorDiagnosticsError(ValueError):
    """Malformed, unsafe, or out-of-policy doctor diagnostic input."""


class DoctorDiagnosticsBoundsError(DoctorDiagnosticsError):
    """A resource bound was exceeded."""


class DoctorDiagnosticsAuthorityError(DoctorDiagnosticsError):
    """Authority roots, paths, or snapshot identity failed closed."""


class DoctorDiagnosticsStaleError(DoctorDiagnosticsAuthorityError):
    """Stale root, tree, or index binding was rejected."""


class DoctorDiagnosticsSymlinkError(DoctorDiagnosticsAuthorityError):
    """A path resolved outside the admitted repository root."""


class DoctorDiagnosticsMixedRootError(DoctorDiagnosticsAuthorityError):
    """Sources spanned more than one admitted root without multi-root mode."""


class FindingKind(str, Enum):
    """Typed diagnostic finding categories produced by the compiler."""

    SYNTAX = "syntax"
    IMPORT = "import"
    NAME = "name"
    CALL_ARITY = "call_arity"
    TYPE = "type"
    CONTRACT = "contract"
    VALUE = "value"
    DATAFLOW = "dataflow"
    ERROR_FACET = "error_facet"
    EFFECT = "effect"
    RESOURCE = "resource"
    STATE = "state"
    SCHEMA = "schema"
    MEMORY = "memory"
    TRACE_JOIN = "trace_join"
    VALIDATION_JOIN = "validation_join"
    UNSUPPORTED = "unsupported"
    COMPLETENESS = "completeness"


class FindingDisposition(str, Enum):
    """Closed doctor dispositions for one finding (not write authority)."""

    SUPPORTED = "supported"
    ABSTAIN = "abstain"
    APPROVAL_REQUIRED = "approval_required"
    OBSERVED = "observed"
    UNKNOWN = "unknown"


class ExpectationSourceKind(str, Enum):
    """Where an expectation originated; never promoted from observations."""

    REVIEWED_CONTRACT = "reviewed_contract"
    REVIEWED_SCHEMA = "reviewed_schema"
    REVIEWED_IDL = "reviewed_idl"
    REVIEWED_SPECIFICATION = "reviewed_specification"
    DECLARED_INTERFACE = "declared_interface"
    STRUCTURED_VALIDATION = "structured_validation"
    BROKEN_TRACE = "broken_trace"
    NONE = "none"


class QuerySurface(str, Enum):
    """Queryable structural surfaces over the compiled snapshot."""

    IMPORTS = "imports"
    EXPORTS = "exports"
    ALIASES = "aliases"
    WRAPPERS = "wrappers"
    ENTRY_POINTS = "entry_points"
    CALL_SITES = "call_sites"
    FINDINGS = "findings"
    FRONTIERS = "frontiers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:" + _sha256_hex(_canonical_json_bytes(value))


def _text(value: Any, name: str, *, required: bool = True, limit: int = 4096) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise DoctorDiagnosticsError(f"{name} must be a string")
    if "\x00" in text:
        raise DoctorDiagnosticsError(f"{name} must not contain NUL")
    stripped = text.strip()
    if required and not stripped:
        raise DoctorDiagnosticsError(f"{name} is required")
    if len(stripped.encode("utf-8")) > limit:
        raise DoctorDiagnosticsBoundsError(f"{name} exceeds its byte bound")
    return stripped


def _optional_text(value: Any, name: str, *, limit: int = 4096) -> str:
    return _text(value, name, required=False, limit=limit)


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DoctorDiagnosticsError(f"{name} must be a positive integer")
    if value < 1 or value > maximum:
        raise DoctorDiagnosticsBoundsError(f"{name} is outside the hard bound")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DoctorDiagnosticsError(f"{name} must be a non-negative integer")
    if value < 0:
        raise DoctorDiagnosticsBoundsError(f"{name} must be non-negative")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorDiagnosticsError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise DoctorDiagnosticsError(f"{name} must be one of: {allowed}") from exc


def _repo_path(value: Any, name: str = "path") -> str:
    raw = _text(value, name, required=True, limit=1024).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw in {".", ""}:
        raise DoctorDiagnosticsAuthorityError(
            f"{name} must be a relative repository path without escape"
        )
    return path.as_posix()


def _string_tuple(
    value: Any,
    name: str,
    *,
    limit: int = 1024,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DoctorDiagnosticsError(f"{name} must be a sequence of strings")
    else:
        items = value
    if len(items) > limit:
        raise DoctorDiagnosticsBoundsError(f"{name} exceeds its item bound")
    result = tuple(sorted({_text(item, name, required=True) for item in items}))
    if required and not result:
        raise DoctorDiagnosticsError(f"{name} must not be empty")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DoctorDiagnosticsError(f"{name} must be a mapping")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise DoctorDiagnosticsBoundsError("nested structure exceeds depth bound")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:  # noqa: PLR0124
            raise DoctorDiagnosticsError("non-finite float is not allowed")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, depth=depth + 1) for item in value]
    raise DoctorDiagnosticsError("unsupported structured value type")


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text in _BODY_MARKERS:
                raise DoctorDiagnosticsError(
                    f"{name} must not carry source bodies via {key!r}"
                )
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, name)


def _decode_source_bytes(value: Any, name: str, *, max_bytes: int) -> bytes:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, bytearray):
        payload = bytes(value)
    elif isinstance(value, str):
        payload = value.encode("utf-8", errors="surrogatepass")
    else:
        raise DoctorDiagnosticsError(f"{name} must be bytes or text")
    if len(payload) > max_bytes:
        raise DoctorDiagnosticsBoundsError(
            f"{name} exceeds max_source_bytes ({max_bytes})"
        )
    return payload


def _source_text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="surrogatepass")


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / PurePosixPath(relative)).resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DoctorDiagnosticsSymlinkError(
            f"path escapes repository root: {relative}"
        ) from exc
    return candidate


# ---------------------------------------------------------------------------
# Policy, authority, and input records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorSnapshotPolicy:
    """Resource bounds and fail-closed admission policy for one snapshot build."""

    schema: str = DOCTOR_SNAPSHOT_POLICY_SCHEMA
    schema_version: int = 1
    policy_id: str = "doctor-snapshot-policy:default"
    max_paths: int = DEFAULT_MAX_PATHS
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_findings: int = DEFAULT_MAX_FINDINGS
    max_trace_joins: int = DEFAULT_MAX_TRACE_JOINS
    max_validation_failures: int = DEFAULT_MAX_VALIDATION_FAILURES
    max_query_results: int = DEFAULT_MAX_QUERY_RESULTS
    allow_mixed_roots: bool = False
    allow_dirty_analysis: bool = True
    supported_languages: tuple[str, ...] = tuple(sorted(SUPPORTED_ADAPTER_LANGUAGES))
    open_frontiers: tuple[str, ...] = DEFAULT_OPEN_FRONTIERS
    require_authority_roots: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        object.__setattr__(
            self,
            "max_paths",
            _positive_int(self.max_paths, "max_paths", maximum=HARD_MAX_PATHS),
        )
        object.__setattr__(
            self,
            "max_source_bytes",
            _positive_int(
                self.max_source_bytes, "max_source_bytes", maximum=HARD_MAX_SOURCE_BYTES
            ),
        )
        object.__setattr__(
            self,
            "max_total_bytes",
            _positive_int(
                self.max_total_bytes, "max_total_bytes", maximum=HARD_MAX_TOTAL_BYTES
            ),
        )
        object.__setattr__(
            self,
            "max_findings",
            _positive_int(self.max_findings, "max_findings", maximum=HARD_MAX_PATHS),
        )
        object.__setattr__(
            self,
            "max_trace_joins",
            _positive_int(
                self.max_trace_joins, "max_trace_joins", maximum=HARD_MAX_PATHS
            ),
        )
        object.__setattr__(
            self,
            "max_validation_failures",
            _positive_int(
                self.max_validation_failures,
                "max_validation_failures",
                maximum=HARD_MAX_PATHS,
            ),
        )
        object.__setattr__(
            self,
            "max_query_results",
            _positive_int(
                self.max_query_results, "max_query_results", maximum=10_000
            ),
        )
        object.__setattr__(
            self, "allow_mixed_roots", _bool(self.allow_mixed_roots, "allow_mixed_roots")
        )
        object.__setattr__(
            self,
            "allow_dirty_analysis",
            _bool(self.allow_dirty_analysis, "allow_dirty_analysis"),
        )
        languages = _string_tuple(self.supported_languages, "supported_languages")
        unknown = set(languages) - SUPPORTED_ADAPTER_LANGUAGES
        if unknown:
            raise DoctorDiagnosticsError(
                f"supported_languages contains unsupported adapters: {sorted(unknown)}"
            )
        object.__setattr__(self, "supported_languages", languages)
        object.__setattr__(
            self, "open_frontiers", _string_tuple(self.open_frontiers, "open_frontiers")
        )
        object.__setattr__(
            self,
            "require_authority_roots",
            _bool(self.require_authority_roots, "require_authority_roots"),
        )
        if int(self.schema_version) != 1:
            raise DoctorDiagnosticsError("unsupported doctor snapshot policy version")

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "max_paths": self.max_paths,
            "max_source_bytes": self.max_source_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_findings": self.max_findings,
            "max_trace_joins": self.max_trace_joins,
            "max_validation_failures": self.max_validation_failures,
            "max_query_results": self.max_query_results,
            "allow_mixed_roots": self.allow_mixed_roots,
            "allow_dirty_analysis": self.allow_dirty_analysis,
            "supported_languages": list(self.supported_languages),
            "open_frontiers": list(self.open_frontiers),
            "require_authority_roots": self.require_authority_roots,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DoctorSnapshotPolicy":
        if value is None:
            return cls()
        if isinstance(value, DoctorSnapshotPolicy):
            return value
        payload = dict(value)
        payload.pop("content_id", None)
        payload.pop("schema", None)
        return cls(**payload)


@dataclass(frozen=True)
class DoctorAuthorityRoots:
    """Parser/config/toolchain and derived index roots for one snapshot."""

    repository_id: str = ""
    forest_id: str = ""
    tree_id: str = ""
    overlay_id: str = ""
    file_root_id: str = ""
    blob_root_id: str = ""
    parser_id: str = "parser:program-ast-adapters@1"
    config_id: str = ""
    toolchain_id: str = "toolchain:deterministic-doctor@1"
    policy_id: str = ""
    ast_index_id: str = ""
    symbol_index_id: str = ""
    import_graph_id: str = ""
    dependency_graph_id: str = ""
    evidence_graph_id: str = ""
    impact_index_id: str = ""
    value_index_id: str = ""
    contract_root_id: str = ""
    corpus_root_id: str = ""
    vector_root_id: str = ""
    embedding_config_id: str = ""
    cache_generation_id: str = ""
    operator_registry_id: str = ""
    translator_id: str = ""
    solver_id: str = ""
    kernel_id: str = ""
    sandbox_id: str = ""
    environment_id: str = ""

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name, limit=512)
            )

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": DOCTOR_AUTHORITY_ROOTS_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    def with_updates(self, **updates: str) -> "DoctorAuthorityRoots":
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload.update(updates)
        return DoctorAuthorityRoots(**payload)

    @classmethod
    def from_mapping(cls, value: Any) -> "DoctorAuthorityRoots":
        if value is None:
            return cls()
        if isinstance(value, DoctorAuthorityRoots):
            return value
        if isinstance(value, AuthorityRoots):
            return cls(
                repository_id=value.repository_id,
                forest_id=value.forest_id,
                tree_id=value.tree_id,
                config_id=value.config_id,
                toolchain_id=value.toolchain_id,
                policy_id=value.policy_id,
                ast_index_id=value.index_id,
                dependency_graph_id=value.graph_id,
                translator_id=value.translator_id,
            )
        if not isinstance(value, Mapping):
            raise DoctorDiagnosticsError("authority roots must be a mapping")
        payload = dict(value)
        payload.pop("content_id", None)
        payload.pop("schema", None)
        # Map contract-repair AuthorityRoots field names when present.
        if "index_id" in payload and "ast_index_id" not in payload:
            payload["ast_index_id"] = payload.pop("index_id")
        if "graph_id" in payload and "dependency_graph_id" not in payload:
            payload["dependency_graph_id"] = payload.pop("graph_id")
        if "model_id" in payload:
            payload.pop("model_id")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: payload[key] for key in payload if key in allowed})


@dataclass(frozen=True)
class DoctorSourceUnit:
    """One inert source unit admitted for parsing (path + bytes)."""

    path: str
    source_bytes: bytes
    language: str = ""
    blob_identity: str = ""
    root_id: str = "root:primary"
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path, "path"))
        if not isinstance(self.source_bytes, (bytes, bytearray)):
            raise DoctorDiagnosticsError("source_bytes must be bytes")
        object.__setattr__(self, "source_bytes", bytes(self.source_bytes))
        language = _optional_text(self.language, "language", limit=64)
        if not language:
            language = detect_program_language(self.path, "")
        object.__setattr__(self, "language", language)
        digest = _sha256_hex(self.source_bytes)
        blob = _optional_text(self.blob_identity, "blob_identity", limit=128)
        object.__setattr__(self, "blob_identity", blob or f"blob:sha256:{digest}")
        object.__setattr__(
            self, "root_id", _text(self.root_id, "root_id", required=True, limit=256)
        )
        object.__setattr__(self, "generated", _bool(self.generated, "generated"))

    @property
    def source_sha256(self) -> str:
        return _sha256_hex(self.source_bytes)

    @property
    def byte_count(self) -> int:
        return len(self.source_bytes)

    def to_dict(self) -> dict[str, Any]:
        # Never embed the source body in durable findings/snapshot payloads.
        return {
            "path": self.path,
            "language": self.language,
            "blob_identity": self.blob_identity,
            "root_id": self.root_id,
            "generated": self.generated,
            "source_sha256": self.source_sha256,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class StructuredValidationFailure:
    """One structured validation failure joined to current AST facts."""

    failure_id: str
    kind: str
    path: str = ""
    symbol: str = ""
    message: str = ""
    expectation_source: ExpectationSourceKind = ExpectationSourceKind.STRUCTURED_VALIDATION
    expectation_ref: str = ""
    observed_ref: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "failure_id", _text(self.failure_id, "failure_id", required=True)
        )
        object.__setattr__(self, "kind", _text(self.kind, "kind", required=True))
        path = _optional_text(self.path, "path", limit=1024)
        object.__setattr__(self, "path", _repo_path(path, "path") if path else "")
        object.__setattr__(
            self, "symbol", _optional_text(self.symbol, "symbol", limit=512)
        )
        object.__setattr__(
            self, "message", _optional_text(self.message, "message", limit=2048)
        )
        object.__setattr__(
            self,
            "expectation_source",
            _enum(self.expectation_source, ExpectationSourceKind, "expectation_source"),
        )
        object.__setattr__(
            self,
            "expectation_ref",
            _optional_text(self.expectation_ref, "expectation_ref", limit=512),
        )
        object.__setattr__(
            self,
            "observed_ref",
            _optional_text(self.observed_ref, "observed_ref", limit=512),
        )
        details = MappingProxyType(dict(_plain(self.details or {})))
        _assert_body_free(details, "details")
        object.__setattr__(self, "details", details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "kind": self.kind,
            "path": self.path,
            "symbol": self.symbol,
            "message": self.message,
            "expectation_source": self.expectation_source.value,
            "expectation_ref": self.expectation_ref,
            "observed_ref": self.observed_ref,
            "details": dict(self.details),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "StructuredValidationFailure":
        if isinstance(value, StructuredValidationFailure):
            return value
        if not isinstance(value, Mapping):
            raise DoctorDiagnosticsError("validation failure must be a mapping")
        payload = dict(value)
        return cls(
            failure_id=str(payload.get("failure_id") or payload.get("id") or ""),
            kind=str(payload.get("kind") or payload.get("code") or "validation"),
            path=str(payload.get("path") or ""),
            symbol=str(payload.get("symbol") or ""),
            message=str(payload.get("message") or ""),
            expectation_source=payload.get(
                "expectation_source", ExpectationSourceKind.STRUCTURED_VALIDATION
            ),
            expectation_ref=str(payload.get("expectation_ref") or ""),
            observed_ref=str(payload.get("observed_ref") or ""),
            details=payload.get("details") or {},
        )


@dataclass(frozen=True)
class DoctorDiagnosticInput:
    """Bounded inputs for one doctor evidence compilation.

    Prefer in-memory ``sources`` for unit tests and leased checkouts.  When
    ``repository_root`` is set, paths are resolved under that root with
    symlink-escape rejection; the compiler still only reads admitted units.
    """

    sources: tuple[DoctorSourceUnit, ...] = ()
    authority_roots: DoctorAuthorityRoots = field(default_factory=DoctorAuthorityRoots)
    policy: DoctorSnapshotPolicy = field(default_factory=DoctorSnapshotPolicy)
    repository_root: str = ""
    broken_traces: tuple[BrokenContractTrace | Mapping[str, Any], ...] = ()
    validation_failures: tuple[StructuredValidationFailure | Mapping[str, Any], ...] = ()
    expectation_refs: tuple[str, ...] = ()
    previous_snapshot: "DoctorEvidenceSnapshot | Mapping[str, Any] | None" = None
    claimed_tree_id: str = ""
    claimed_snapshot_id: str = ""
    provider_call_count: int = 0
    source_write_count: int = 0

    def __post_init__(self) -> None:
        policy = DoctorSnapshotPolicy.from_mapping(self.policy)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(
            self, "authority_roots", DoctorAuthorityRoots.from_mapping(self.authority_roots)
        )
        sources = tuple(self.sources)
        if not isinstance(sources, tuple):
            raise DoctorDiagnosticsError("sources must be a sequence")
        normalized_sources: list[DoctorSourceUnit] = []
        for item in sources:
            if isinstance(item, DoctorSourceUnit):
                unit = item
            elif isinstance(item, SourceDocument):
                unit = DoctorSourceUnit(
                    path=item.path,
                    source_bytes=item.source.encode("utf-8", errors="surrogatepass"),
                    language=item.language,
                    blob_identity=item.blob_identity,
                    generated=item.generated,
                )
            elif isinstance(item, Mapping):
                unit = _source_unit_from_mapping(item, policy)
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) == 2
            ):
                path, body = item
                unit = DoctorSourceUnit(
                    path=str(path),
                    source_bytes=_decode_source_bytes(
                        body, "source", max_bytes=policy.max_source_bytes
                    ),
                )
            else:
                raise DoctorDiagnosticsError(
                    "sources must be DoctorSourceUnit, SourceDocument, mapping, or path/body pairs"
                )
            if unit.byte_count > policy.max_source_bytes:
                raise DoctorDiagnosticsBoundsError(
                    f"source {unit.path!r} exceeds max_source_bytes"
                )
            normalized_sources.append(unit)
        if len(normalized_sources) > policy.max_paths:
            raise DoctorDiagnosticsBoundsError("source path count exceeds max_paths")
        paths = [item.path for item in normalized_sources]
        if len(paths) != len(set(paths)):
            raise DoctorDiagnosticsError("source paths must be unique")
        total = sum(item.byte_count for item in normalized_sources)
        if total > policy.max_total_bytes:
            raise DoctorDiagnosticsBoundsError("total source bytes exceed max_total_bytes")
        object.__setattr__(self, "sources", tuple(normalized_sources))
        object.__setattr__(
            self,
            "repository_root",
            _optional_text(self.repository_root, "repository_root", limit=4096),
        )
        if len(self.broken_traces) > self.policy.max_trace_joins:
            raise DoctorDiagnosticsBoundsError("broken_traces exceeds max_trace_joins")
        object.__setattr__(self, "broken_traces", tuple(self.broken_traces))
        if len(self.validation_failures) > self.policy.max_validation_failures:
            raise DoctorDiagnosticsBoundsError(
                "validation_failures exceeds max_validation_failures"
            )
        object.__setattr__(
            self,
            "validation_failures",
            tuple(
                StructuredValidationFailure.from_mapping(item)
                for item in self.validation_failures
            ),
        )
        object.__setattr__(
            self,
            "expectation_refs",
            _string_tuple(self.expectation_refs, "expectation_refs"),
        )
        object.__setattr__(
            self,
            "claimed_tree_id",
            _optional_text(self.claimed_tree_id, "claimed_tree_id", limit=256),
        )
        object.__setattr__(
            self,
            "claimed_snapshot_id",
            _optional_text(self.claimed_snapshot_id, "claimed_snapshot_id", limit=256),
        )
        object.__setattr__(
            self,
            "provider_call_count",
            _nonneg_int(self.provider_call_count, "provider_call_count"),
        )
        object.__setattr__(
            self,
            "source_write_count",
            _nonneg_int(self.source_write_count, "source_write_count"),
        )
        if self.provider_call_count != 0:
            raise DoctorDiagnosticsError(
                "deterministic doctor diagnostics require zero provider calls"
            )
        if self.source_write_count != 0:
            raise DoctorDiagnosticsError(
                "deterministic doctor diagnostics require zero source writes"
            )


def _source_unit_from_mapping(
    value: Mapping[str, Any], policy: DoctorSnapshotPolicy
) -> DoctorSourceUnit:
    path = str(value.get("path") or value.get("file") or "")
    if "source_bytes" in value:
        body = value["source_bytes"]
    elif "source" in value:
        body = value["source"]
    elif "text" in value:
        body = value["text"]
    else:
        raise DoctorDiagnosticsError("source mapping requires source or source_bytes")
    return DoctorSourceUnit(
        path=path,
        source_bytes=_decode_source_bytes(
            body, "source", max_bytes=policy.max_source_bytes
        ),
        language=str(value.get("language") or ""),
        blob_identity=str(value.get("blob_identity") or value.get("blob_id") or ""),
        root_id=str(value.get("root_id") or "root:primary"),
        generated=bool(value.get("generated", False)),
    )


# ---------------------------------------------------------------------------
# Findings, query results, snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorDiagnosticFinding:
    """One typed finding with observations separated from expectations."""

    kind: FindingKind
    disposition: FindingDisposition
    path: str = ""
    symbol: str = ""
    message: str = ""
    observation_refs: tuple[str, ...] = ()
    expectation_source: ExpectationSourceKind = ExpectationSourceKind.NONE
    expectation_ref: str = ""
    expectation_precedence: int = 0
    open_frontier_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema: str = DOCTOR_DIAGNOSTIC_FINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, FindingKind, "kind"))
        object.__setattr__(
            self, "disposition", _enum(self.disposition, FindingDisposition, "disposition")
        )
        path = _optional_text(self.path, "path", limit=1024)
        object.__setattr__(self, "path", _repo_path(path, "path") if path else "")
        object.__setattr__(
            self, "symbol", _optional_text(self.symbol, "symbol", limit=512)
        )
        object.__setattr__(
            self, "message", _optional_text(self.message, "message", limit=2048)
        )
        object.__setattr__(
            self,
            "observation_refs",
            _string_tuple(self.observation_refs, "observation_refs"),
        )
        object.__setattr__(
            self,
            "expectation_source",
            _enum(self.expectation_source, ExpectationSourceKind, "expectation_source"),
        )
        object.__setattr__(
            self,
            "expectation_ref",
            _optional_text(self.expectation_ref, "expectation_ref", limit=512),
        )
        if isinstance(self.expectation_precedence, bool) or not isinstance(
            self.expectation_precedence, int
        ):
            raise DoctorDiagnosticsError("expectation_precedence must be an integer")
        if self.expectation_precedence < 0 or self.expectation_precedence > 1_000_000:
            raise DoctorDiagnosticsBoundsError("expectation_precedence out of range")
        object.__setattr__(
            self,
            "open_frontier_refs",
            _string_tuple(self.open_frontier_refs, "open_frontier_refs"),
        )
        object.__setattr__(
            self, "evidence_refs", _string_tuple(self.evidence_refs, "evidence_refs")
        )
        details = MappingProxyType(dict(_plain(self.details or {})))
        _assert_body_free(details, "finding.details")
        object.__setattr__(self, "details", details)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "path": self.path,
            "symbol": self.symbol,
            "message": self.message,
            "observation_refs": list(self.observation_refs),
            "expectation_source": self.expectation_source.value,
            "expectation_ref": self.expectation_ref,
            "expectation_precedence": self.expectation_precedence,
            "open_frontier_refs": list(self.open_frontier_refs),
            "evidence_refs": list(self.evidence_refs),
            "details": dict(self.details),
        }

    @property
    def finding_cid(self) -> str:
        """Canonical content identity for this finding (CIDv1 dag-json/sha2-256)."""

        return content_identity(self._payload())

    @property
    def content_id(self) -> str:
        return self.finding_cid

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "finding_cid": self.finding_cid}


@dataclass(frozen=True)
class DoctorQueryHit:
    """One query surface hit projected from the compiled snapshot."""

    surface: QuerySurface
    path: str
    name: str
    target: str = ""
    owner: str = ""
    language: str = ""
    fact_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", _enum(self.surface, QuerySurface, "surface"))
        object.__setattr__(self, "path", _repo_path(self.path, "path") if self.path else "")
        object.__setattr__(self, "name", _optional_text(self.name, "name", limit=512))
        object.__setattr__(
            self, "target", _optional_text(self.target, "target", limit=1024)
        )
        object.__setattr__(
            self, "owner", _optional_text(self.owner, "owner", limit=512)
        )
        object.__setattr__(
            self, "language", _optional_text(self.language, "language", limit=64)
        )
        object.__setattr__(
            self, "fact_id", _optional_text(self.fact_id, "fact_id", limit=256)
        )
        object.__setattr__(
            self, "details", MappingProxyType(dict(_plain(self.details or {})))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface.value,
            "path": self.path,
            "name": self.name,
            "target": self.target,
            "owner": self.owner,
            "language": self.language,
            "fact_id": self.fact_id,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DoctorQueryResult:
    """Bounded query response over a frozen evidence snapshot."""

    surface: QuerySurface
    hits: tuple[DoctorQueryHit, ...]
    truncated: bool = False
    schema: str = DOCTOR_QUERY_RESULT_SCHEMA

    @property
    def count(self) -> int:
        return len(self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "surface": self.surface.value,
            "hits": [item.to_dict() for item in self.hits],
            "truncated": self.truncated,
            "count": self.count,
        }


@dataclass(frozen=True)
class DoctorAdapterReceipt:
    """Per-path adapter admission receipt (no source body)."""

    path: str
    language: str
    status: str
    parser: str
    blob_identity: str
    source_sha256: str
    fact_count: int
    diagnostic_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "status": self.status,
            "parser": self.parser,
            "blob_identity": self.blob_identity,
            "source_sha256": self.source_sha256,
            "fact_count": self.fact_count,
            "diagnostic_codes": list(self.diagnostic_codes),
        }


@dataclass(frozen=True)
class DoctorEvidenceSnapshot:
    """Immutable doctor evidence snapshot over a real or synthetic checkout."""

    authority_roots: DoctorAuthorityRoots
    policy: DoctorSnapshotPolicy
    ast_index: AnalysisASTIndex
    adapter_receipts: tuple[DoctorAdapterReceipt, ...]
    findings: tuple[DoctorDiagnosticFinding, ...]
    open_frontiers: tuple[str, ...]
    query_index: Mapping[str, tuple[DoctorQueryHit, ...]]
    completeness: Mapping[str, Any]
    provider_call_count: int = 0
    source_write_count: int = 0
    schema: str = DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA
    schema_version: int = 1
    rebuild_mode: str = "clean"

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise DoctorDiagnosticsError("unsupported doctor evidence snapshot version")
        object.__setattr__(
            self,
            "authority_roots",
            DoctorAuthorityRoots.from_mapping(self.authority_roots),
        )
        object.__setattr__(
            self, "policy", DoctorSnapshotPolicy.from_mapping(self.policy)
        )
        if not isinstance(self.ast_index, AnalysisASTIndex):
            raise DoctorDiagnosticsError("ast_index must be an AnalysisASTIndex")
        object.__setattr__(
            self,
            "adapter_receipts",
            tuple(
                sorted(self.adapter_receipts, key=lambda item: (item.path, item.blob_identity))
            ),
        )
        object.__setattr__(
            self,
            "findings",
            tuple(sorted(self.findings, key=lambda item: item.finding_cid)),
        )
        object.__setattr__(
            self, "open_frontiers", tuple(sorted(set(self.open_frontiers)))
        )
        frozen_index = {
            key: tuple(hits)
            for key, hits in sorted((self.query_index or {}).items())
        }
        object.__setattr__(self, "query_index", MappingProxyType(frozen_index))
        object.__setattr__(
            self, "completeness", MappingProxyType(dict(_plain(self.completeness or {})))
        )
        object.__setattr__(
            self,
            "provider_call_count",
            _nonneg_int(self.provider_call_count, "provider_call_count"),
        )
        object.__setattr__(
            self,
            "source_write_count",
            _nonneg_int(self.source_write_count, "source_write_count"),
        )
        object.__setattr__(
            self,
            "rebuild_mode",
            _text(self.rebuild_mode, "rebuild_mode", required=True, limit=32),
        )
        if self.provider_call_count != 0 or self.source_write_count != 0:
            raise DoctorDiagnosticsError(
                "snapshot must record zero provider calls and zero source writes"
            )

    def _identity_payload(self) -> dict[str, Any]:
        # Rebuild mode and cache stats are excluded so incremental and clean
        # rebuilds that share the same evidence are identity-equivalent.
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "authority_roots": self.authority_roots._payload(),
            "policy": self.policy._payload(),
            "ast_index_id": self.ast_index.index_id,
            "adapter_receipts": [item.to_dict() for item in self.adapter_receipts],
            "findings": [item._payload() for item in self.findings],
            "open_frontiers": list(self.open_frontiers),
            "query_index": {
                key: [hit.to_dict() for hit in hits]
                for key, hits in sorted(self.query_index.items())
            },
            "completeness": dict(self.completeness),
            "provider_call_count": 0,
            "source_write_count": 0,
        }

    @property
    def snapshot_cid(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def snapshot_id(self) -> str:
        return _identity("doctor-evidence-snapshot", self._identity_payload())

    @property
    def finding_cids(self) -> tuple[str, ...]:
        return tuple(item.finding_cid for item in self.findings)

    def query(
        self,
        surface: QuerySurface | str,
        *,
        path: str = "",
        name: str = "",
        limit: int | None = None,
    ) -> DoctorQueryResult:
        surface_enum = _enum(surface, QuerySurface, "surface")
        max_results = (
            self.policy.max_query_results if limit is None else int(limit)
        )
        if max_results < 1:
            raise DoctorDiagnosticsBoundsError("query limit must be positive")
        hits = list(self.query_index.get(surface_enum.value, ()))
        if path:
            normalized = _repo_path(path, "path")
            hits = [item for item in hits if item.path == normalized]
        if name:
            needle = name.strip()
            hits = [
                item
                for item in hits
                if needle in {item.name, item.target, item.owner}
                or needle in item.name
                or needle in item.target
            ]
        truncated = len(hits) > max_results
        return DoctorQueryResult(
            surface=surface_enum,
            hits=tuple(hits[:max_results]),
            truncated=truncated,
        )

    def finding_for_cid(self, finding_cid: str) -> DoctorDiagnosticFinding | None:
        for item in self.findings:
            if item.finding_cid == finding_cid:
                return item
        return None

    def localize(
        self,
        finding: DoctorDiagnosticFinding | None = None,
        *,
        evidence: Sequence[Any] = (),
        impact_closure: Any = None,
        required_frontiers: Sequence[str] = (),
    ) -> Any:
        """Run PDR-042 causal localization over this immutable snapshot."""

        from .doctor_causal_localization import (
            DoctorCausalLocalizationRequest,
            localize_doctor_cause,
        )

        return localize_doctor_cause(
            DoctorCausalLocalizationRequest(
                snapshot=self,
                finding=finding,
                evidence=tuple(evidence),
                impact_closure=impact_closure,
                required_frontiers=tuple(required_frontiers),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "snapshot_cid": self.snapshot_cid,
            "snapshot_id": self.snapshot_id,
            "rebuild_mode": self.rebuild_mode,
            "finding_cids": list(self.finding_cids),
            "ast_index": self.ast_index.to_dict(),
        }


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def _coerce_trace(value: BrokenContractTrace | Mapping[str, Any]) -> BrokenContractTrace:
    if isinstance(value, BrokenContractTrace):
        return value
    if isinstance(value, Mapping):
        return (
            BrokenContractTrace.from_dict(value)
            if "schema" in value
            else BrokenContractTrace(**value)  # type: ignore[arg-type]
        )
    raise DoctorDiagnosticsError("broken_traces items must be BrokenContractTrace")


def _adapter_receipt(result: ProgramASTAdapterResult) -> DoctorAdapterReceipt:
    return DoctorAdapterReceipt(
        path=result.path or "",
        language=result.language,
        status=result.status,
        parser=result.parser,
        blob_identity=result.blob_identity,
        source_sha256=result.source_sha256,
        fact_count=len(result.facts),
        diagnostic_codes=tuple(
            sorted({item.code for item in result.diagnostics if item.code})
        ),
    )


def _hit_from_fact(
    surface: QuerySurface,
    path: str,
    language: str,
    fact: ProgramEvidenceFact,
) -> DoctorQueryHit:
    return DoctorQueryHit(
        surface=surface,
        path=path,
        name=fact.name,
        target=fact.target,
        owner=fact.owner,
        language=language,
        fact_id=fact.fact_id,
        details={
            "kind": fact.kind,
            "relationship": fact.relationship,
            "ambiguous": fact.ambiguous,
            "normative": fact.normative,
            "generated": fact.generated,
            **{
                key: value
                for key, value in fact.details.items()
                if key not in _BODY_MARKERS
            },
        },
    )


def _build_query_index(
    results: Sequence[ProgramASTAdapterResult],
    *,
    max_per_surface: int,
) -> dict[str, tuple[DoctorQueryHit, ...]]:
    buckets: dict[str, list[DoctorQueryHit]] = {
        surface.value: [] for surface in QuerySurface if surface is not QuerySurface.FINDINGS
    }
    for result in results:
        path = result.path or ""
        if not path:
            continue
        language = result.language
        for fact in result.facts:
            if fact.kind == "import":
                buckets[QuerySurface.IMPORTS.value].append(
                    _hit_from_fact(QuerySurface.IMPORTS, path, language, fact)
                )
                alias = str(fact.details.get("alias") or "")
                if alias or (fact.name and fact.target and fact.name != fact.target):
                    buckets[QuerySurface.ALIASES.value].append(
                        _hit_from_fact(QuerySurface.ALIASES, path, language, fact)
                    )
            if fact.kind in _EXPORT_KINDS and fact.relationship in {
                "defines",
                "exports",
                "reexports",
                "observed",
            }:
                if fact.kind in {"export", "re_export"} or (
                    fact.kind
                    in {
                        "function_definition",
                        "async_function_definition",
                        "class_definition",
                    }
                    and fact.owner in {"", "<module>"}
                ):
                    buckets[QuerySurface.EXPORTS.value].append(
                        _hit_from_fact(QuerySurface.EXPORTS, path, language, fact)
                    )
            if fact.kind in _WRAPPER_KINDS:
                buckets[QuerySurface.WRAPPERS.value].append(
                    _hit_from_fact(QuerySurface.WRAPPERS, path, language, fact)
                )
            if fact.kind in _ENTRY_KINDS and fact.owner in {"", "<module>"}:
                buckets[QuerySurface.ENTRY_POINTS.value].append(
                    _hit_from_fact(QuerySurface.ENTRY_POINTS, path, language, fact)
                )
            if fact.kind in _CALL_KINDS:
                buckets[QuerySurface.CALL_SITES.value].append(
                    _hit_from_fact(QuerySurface.CALL_SITES, path, language, fact)
                )
    # Also project AST index imports/calls for paths that only have blob records.
    for key, hits in buckets.items():
        # Stable order; drop exact duplicates by fact_id+surface+path+name+target.
        unique: dict[tuple[str, ...], DoctorQueryHit] = {}
        for hit in hits:
            unique[
                (
                    hit.surface.value,
                    hit.path,
                    hit.name,
                    hit.target,
                    hit.owner,
                    hit.fact_id,
                )
            ] = hit
        ordered = sorted(
            unique.values(),
            key=lambda item: (item.path, item.name, item.target, item.fact_id),
        )
        buckets[key] = ordered[:max_per_surface]
    return {key: tuple(value) for key, value in buckets.items()}


def _syntax_findings(
    result: ProgramASTAdapterResult,
) -> list[DoctorDiagnosticFinding]:
    findings: list[DoctorDiagnosticFinding] = []
    path = result.path or ""
    if result.status in {"malformed", "unsupported", "partial"}:
        for diagnostic in result.diagnostics:
            kind = (
                FindingKind.SYNTAX
                if diagnostic.code
                not in {"unsupported_language", "source_size_bound_exceeded"}
                else FindingKind.UNSUPPORTED
            )
            disposition = (
                FindingDisposition.ABSTAIN
                if kind is FindingKind.UNSUPPORTED
                else FindingDisposition.OBSERVED
            )
            findings.append(
                DoctorDiagnosticFinding(
                    kind=kind,
                    disposition=disposition,
                    path=path,
                    message=diagnostic.message,
                    observation_refs=(
                        f"adapter:{result.blob_identity}",
                        f"diagnostic:{diagnostic.code}",
                    ),
                    details={
                        "diagnostic_code": diagnostic.code,
                        "severity": diagnostic.severity,
                        "status": result.status,
                        "language": result.language,
                        "parser": result.parser,
                    },
                )
            )
        if not result.diagnostics:
            findings.append(
                DoctorDiagnosticFinding(
                    kind=FindingKind.SYNTAX
                    if result.status == "malformed"
                    else FindingKind.UNSUPPORTED,
                    disposition=FindingDisposition.OBSERVED,
                    path=path,
                    message=f"adapter status {result.status}",
                    observation_refs=(f"adapter:{result.blob_identity}",),
                    details={"status": result.status, "language": result.language},
                )
            )
    # Ambiguous/dynamic call sites remain observations, not repair authority.
    for fact in result.facts:
        if fact.kind == "monkey_patch":
            findings.append(
                DoctorDiagnosticFinding(
                    kind=FindingKind.UNSUPPORTED,
                    disposition=FindingDisposition.ABSTAIN,
                    path=path,
                    symbol=fact.name,
                    message="reflection/monkey-patch observation cannot close repair",
                    observation_refs=(fact.fact_id,),
                    open_frontier_refs=("frontier:reflection",),
                    details={"fact_kind": fact.kind, "target": fact.target},
                )
            )
        elif fact.kind in _CALL_KINDS and fact.ambiguous:
            resolution = str(fact.details.get("resolution") or "ambiguous")
            if resolution in {"dynamic_expression", "candidate_only"}:
                findings.append(
                    DoctorDiagnosticFinding(
                        kind=FindingKind.NAME,
                        disposition=FindingDisposition.UNKNOWN,
                        path=path,
                        symbol=fact.name or fact.target,
                        message=f"unresolved or candidate-only call ({resolution})",
                        observation_refs=(fact.fact_id,),
                        open_frontier_refs=(
                            ("frontier:interprocedural_dataflow",)
                            if resolution == "dynamic_expression"
                            else ()
                        ),
                        details={
                            "fact_kind": fact.kind,
                            "resolution": resolution,
                            "import_candidate": fact.details.get("import_candidate", ""),
                        },
                    )
                )
        elif fact.kind == "exception_handler":
            findings.append(
                DoctorDiagnosticFinding(
                    kind=FindingKind.ERROR_FACET,
                    disposition=FindingDisposition.OBSERVED,
                    path=path,
                    symbol=fact.name,
                    message="exception handler observed; propagation remains an open frontier",
                    observation_refs=(fact.fact_id,),
                    open_frontier_refs=("frontier:exception_propagation",),
                    details={"target": fact.target},
                )
            )
    return findings


def _python_signature_arity(
    signature: str,
) -> tuple[
    int,
    int | None,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    frozenset[str],
    bool,
] | None:
    """Return exact positional/keyword arity facts for an inert signature."""

    try:
        parsed = ast.parse(f"{signature}:\n    pass\n")
    except (SyntaxError, ValueError, TypeError):
        return None
    if not parsed.body or not isinstance(parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    args = parsed.body[0].args
    positional_nodes = (*args.posonlyargs, *args.args)
    positional_names = tuple(item.arg for item in positional_nodes)
    required_count = len(positional_nodes) - len(args.defaults)
    required_names = positional_names[:required_count]
    required_kwonly = tuple(
        item.arg
        for item, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    )
    positional_only = {item.arg for item in args.posonlyargs}
    accepted_keywords = frozenset(
        {item.arg for item in (*args.args, *args.kwonlyargs)} - positional_only
    )
    maximum = None if args.vararg is not None else len(positional_nodes)
    return (
        required_count,
        maximum,
        positional_names,
        required_names,
        required_kwonly,
        accepted_keywords,
        args.kwarg is not None,
    )


def _derived_call_contract_findings(
    query_index: Mapping[str, tuple[DoctorQueryHit, ...]],
) -> list[DoctorDiagnosticFinding]:
    """Derive exact local call/signature mismatches from current checkout facts.

    This pass never consumes an expected answer.  The declaration and call
    facts are independently parsed from the same admitted checkout and joined
    through an exact import candidate.  Ambiguous/dynamic targets remain in
    the frontier rather than being guessed.
    """

    exports = query_index.get(QuerySurface.EXPORTS.value, ())
    calls = query_index.get(QuerySurface.CALL_SITES.value, ())
    definitions: dict[str, list[DoctorQueryHit]] = {}
    for hit in exports:
        module = hit.path
        for suffix in (".py", ".pyi", ".js", ".ts", ".tsx", ".jsx"):
            if module.endswith(suffix):
                module = module[: -len(suffix)]
                break
        qualified = f"{module.replace('/', '.')}.{hit.name}"
        definitions.setdefault(qualified, []).append(hit)

    findings: list[DoctorDiagnosticFinding] = []
    for call in calls:
        candidate = str(call.details.get("import_candidate") or "")
        if not candidate:
            continue
        matches = definitions.get(candidate, ())
        if len(matches) != 1:
            continue
        definition = matches[0]
        signature = str(definition.details.get("signature") or definition.target or "")
        arity = _python_signature_arity(signature)
        observed = call.details.get("argument_count")
        if arity is None or isinstance(observed, bool) or not isinstance(observed, int):
            continue
        (
            minimum,
            maximum,
            positional_names,
            required_names,
            required_kwonly,
            accepted_keywords,
            has_var_keyword,
        ) = arity
        keywords = frozenset(
            str(item) for item in (call.details.get("keyword_names") or ())
        )
        positionally_filled = frozenset(positional_names[:observed])
        missing_required = (
            (set(required_names) - positionally_filled - keywords)
            | (set(required_kwonly) - keywords)
        )
        unexpected_keywords = (
            set() if has_var_keyword else set(keywords) - set(accepted_keywords)
        )
        duplicate_arguments = set(keywords) & set(positionally_filled)
        positional_overflow = maximum is not None and observed > maximum
        if not (
            missing_required
            or unexpected_keywords
            or duplicate_arguments
            or positional_overflow
        ):
            continue
        expected_text = str(minimum) if maximum == minimum else f"{minimum}..{maximum or '*'}"
        findings.append(
            DoctorDiagnosticFinding(
                kind=FindingKind.CALL_ARITY,
                disposition=FindingDisposition.SUPPORTED,
                path=call.path,
                symbol=call.owner or call.name,
                message=(
                    f"call supplies {observed} positional arguments; "
                    f"declared interface requires {expected_text}"
                ),
                observation_refs=(call.fact_id, definition.fact_id),
                expectation_source=ExpectationSourceKind.DECLARED_INTERFACE,
                expectation_ref=definition.fact_id,
                expectation_precedence=50,
                evidence_refs=(call.fact_id, definition.fact_id),
                details={
                    "call_fact_id": call.fact_id,
                    "definition_fact_id": definition.fact_id,
                    "declared_target": candidate,
                    "expected_argument_count": expected_text,
                    "observed_argument_count": observed,
                    "missing_required_parameters": sorted(missing_required),
                    "unexpected_keywords": sorted(unexpected_keywords),
                    "duplicate_arguments": sorted(duplicate_arguments),
                },
            )
        )
    return findings


def _join_trace(
    trace: BrokenContractTrace,
    *,
    path_facts: Mapping[str, tuple[ProgramEvidenceFact, ...]],
    index: AnalysisASTIndex,
    expected_roots: DoctorAuthorityRoots,
) -> DoctorDiagnosticFinding:
    if expected_roots.repository_id and trace.roots.repository_id != expected_roots.repository_id:
        raise DoctorDiagnosticsStaleError("broken trace repository_id does not match")
    if expected_roots.tree_id and trace.roots.tree_id != expected_roots.tree_id:
        raise DoctorDiagnosticsStaleError("broken trace tree_id does not match")
    path = trace.caller_span.path
    facts = path_facts.get(path, ())
    matching_calls = [
        fact
        for fact in facts
        if fact.kind in _CALL_KINDS
        and (
            trace.receiver_reference in {fact.name, fact.target}
            or trace.receiver_reference in fact.target
            or trace.receiver_reference in fact.name
            or str(fact.details.get("import_candidate") or "").endswith(
                trace.receiver_reference.split(".")[-1]
            )
        )
    ]
    indexed = index.record_for_path(path)
    observation_refs = [
        *(fact.fact_id for fact in matching_calls),
        *(ref.content_id for ref in trace.evidence_refs),
    ]
    if indexed is not None:
        observation_refs.append(indexed.record_id)
        for call in indexed.ast_record.calls:
            if trace.receiver_reference in call or call.endswith(
                trace.receiver_reference.split(".")[-1]
            ):
                observation_refs.append(f"ast-call:{path}:{call}")
    disposition_map = {
        TraceDisposition.RESOLVED_MISMATCH: FindingDisposition.SUPPORTED,
        TraceDisposition.MISSING_LOCAL: FindingDisposition.ABSTAIN,
        TraceDisposition.LIKELY_REFACTOR: FindingDisposition.APPROVAL_REQUIRED,
        TraceDisposition.ADAPTER_REQUIRED: FindingDisposition.APPROVAL_REQUIRED,
        TraceDisposition.EXTERNAL: FindingDisposition.ABSTAIN,
        TraceDisposition.DYNAMIC: FindingDisposition.ABSTAIN,
        TraceDisposition.AMBIGUOUS: FindingDisposition.ABSTAIN,
        TraceDisposition.UNSUPPORTED: FindingDisposition.ABSTAIN,
    }
    open_frontiers: list[str] = list(trace.graph_frontier_refs)
    if trace.disposition in {
        TraceDisposition.DYNAMIC,
        TraceDisposition.UNSUPPORTED,
        TraceDisposition.AMBIGUOUS,
    }:
        open_frontiers.append("frontier:interprocedural_dataflow")
    return DoctorDiagnosticFinding(
        kind=FindingKind.TRACE_JOIN,
        disposition=disposition_map.get(trace.disposition, FindingDisposition.ABSTAIN),
        path=path,
        symbol=trace.caller_symbol_id,
        message=(
            f"joined broken trace disposition={trace.disposition.value} "
            f"receiver={trace.receiver_reference}"
        ),
        observation_refs=tuple(observation_refs),
        expectation_source=ExpectationSourceKind.BROKEN_TRACE,
        expectation_ref=trace.content_id,
        expectation_precedence=100,
        open_frontier_refs=tuple(sorted(set(open_frontiers))),
        evidence_refs=tuple(ref.content_id for ref in trace.evidence_refs),
        details={
            "receiver_reference": trace.receiver_reference,
            "trace_disposition": trace.disposition.value,
            "matched_fact_count": len(matching_calls),
            "target_path": trace.target_span.path if trace.target_span else "",
            "excluded_refs": list(trace.excluded_refs),
        },
    )


def _join_validation(
    failure: StructuredValidationFailure,
    *,
    path_facts: Mapping[str, tuple[ProgramEvidenceFact, ...]],
    index: AnalysisASTIndex,
) -> DoctorDiagnosticFinding:
    facts = path_facts.get(failure.path, ()) if failure.path else ()
    observation_refs: list[str] = []
    if failure.observed_ref:
        observation_refs.append(failure.observed_ref)
    for fact in facts:
        if failure.symbol and failure.symbol in {
            fact.name,
            fact.target,
            fact.owner,
        }:
            observation_refs.append(fact.fact_id)
        elif not failure.symbol and fact.kind in {
            "function_definition",
            "async_function_definition",
            "class_definition",
            "import",
            "export",
            "call",
        }:
            # Path-level join without inventing a symbol match.
            observation_refs.append(fact.fact_id)
    if failure.path:
        indexed = index.record_for_path(failure.path)
        if indexed is not None:
            observation_refs.append(indexed.record_id)
    # De-dupe while preserving order.
    seen: set[str] = set()
    ordered_refs: list[str] = []
    for ref in observation_refs:
        if ref and ref not in seen:
            seen.add(ref)
            ordered_refs.append(ref)
    return DoctorDiagnosticFinding(
        kind=FindingKind.VALIDATION_JOIN,
        disposition=FindingDisposition.OBSERVED,
        path=failure.path,
        symbol=failure.symbol,
        message=failure.message or f"validation failure {failure.kind}",
        observation_refs=tuple(ordered_refs),
        expectation_source=failure.expectation_source,
        expectation_ref=failure.expectation_ref or failure.failure_id,
        expectation_precedence=200,
        details={
            "failure_id": failure.failure_id,
            "failure_kind": failure.kind,
            **dict(failure.details),
        },
    )


def _language_frontiers(
    results: Sequence[ProgramASTAdapterResult],
    policy: DoctorSnapshotPolicy,
) -> tuple[str, ...]:
    frontiers = set(policy.open_frontiers)
    languages = {item.language for item in results if item.language}
    non_python = languages - {"python", ""}
    if non_python:
        frontiers.add("frontier:python_only_full_type_inference")
        frontiers.add("frontier:non_python_cfg")
    if any(item.language in {"c", "cpp", "rust", "go"} for item in results):
        frontiers.add("frontier:native_ffi")
    if any(item.generated for item in results) or any(
        fact.generated for item in results for fact in item.facts
    ):
        frontiers.add("frontier:generated_code")
    if any(
        fact.kind in {"monkey_patch", "dynamic_import"}
        for item in results
        for fact in item.facts
    ):
        frontiers.add("frontier:reflection")
    if any(fact.kind == "exception_handler" for item in results for fact in item.facts):
        frontiers.add("frontier:exception_propagation")
    if any(
        fact.kind in _CALL_KINDS and fact.ambiguous
        for item in results
        for fact in item.facts
    ):
        frontiers.add("frontier:dynamic_dispatch")
    return tuple(sorted(frontiers))


def _load_sources_from_repository(
    repository_root: str,
    relative_paths: Sequence[str],
    *,
    policy: DoctorSnapshotPolicy,
    root_id: str = "root:primary",
) -> tuple[DoctorSourceUnit, ...]:
    root = Path(repository_root)
    if not root.is_dir():
        raise DoctorDiagnosticsError("repository_root must be an existing directory")
    units: list[DoctorSourceUnit] = []
    for relative in relative_paths:
        path = _repo_path(relative, "path")
        resolved = _resolve_under_root(root, path)
        if resolved.is_symlink():
            # Readlink target must still stay under root.
            target = resolved.resolve(strict=False)
            try:
                target.relative_to(root.resolve(strict=False))
            except ValueError as exc:
                raise DoctorDiagnosticsSymlinkError(
                    f"symlink escapes repository root: {path}"
                ) from exc
        if not resolved.is_file():
            raise DoctorDiagnosticsError(f"source path is not a file: {path}")
        payload = resolved.read_bytes()
        if len(payload) > policy.max_source_bytes:
            raise DoctorDiagnosticsBoundsError(
                f"source {path!r} exceeds max_source_bytes"
            )
        units.append(
            DoctorSourceUnit(
                path=path,
                source_bytes=payload,
                root_id=root_id,
            )
        )
    return tuple(units)


class DoctorEvidenceCompiler:
    """Compile a frozen doctor evidence snapshot from inert sources."""

    def __init__(self, policy: DoctorSnapshotPolicy | Mapping[str, Any] | None = None) -> None:
        self.policy = DoctorSnapshotPolicy.from_mapping(policy)

    def compile(
        self, diagnostic_input: DoctorDiagnosticInput | Mapping[str, Any]
    ) -> DoctorEvidenceSnapshot:
        return compile_doctor_evidence_snapshot(
            diagnostic_input,
            policy=self.policy if not isinstance(diagnostic_input, DoctorDiagnosticInput) else None,
        )

    def diagnose(
        self, diagnostic_input: DoctorDiagnosticInput | Mapping[str, Any]
    ) -> DoctorEvidenceSnapshot:
        return self.compile(diagnostic_input)

    def localize(self, request: Any) -> Any:
        """Localize one compiled finding without introducing an import cycle."""

        from .doctor_causal_localization import localize_doctor_cause

        return localize_doctor_cause(request)


def compile_doctor_evidence_snapshot(
    diagnostic_input: DoctorDiagnosticInput | Mapping[str, Any],
    *,
    policy: DoctorSnapshotPolicy | Mapping[str, Any] | None = None,
    previous: DoctorEvidenceSnapshot | AnalysisASTIndex | None = None,
) -> DoctorEvidenceSnapshot:
    """Compile one immutable doctor evidence snapshot.

    Incremental invalidation reuses the analysis AST index cache; the resulting
    snapshot identity excludes rebuild mode so incremental and clean rebuilds
    of the same admitted sources are identity-equivalent.
    """

    if isinstance(diagnostic_input, Mapping):
        if policy is not None and "policy" not in diagnostic_input:
            diagnostic_input = {**diagnostic_input, "policy": policy}
        diagnostic_input = DoctorDiagnosticInput(**diagnostic_input)  # type: ignore[arg-type]
    elif not isinstance(diagnostic_input, DoctorDiagnosticInput):
        raise DoctorDiagnosticsError("diagnostic_input is required")

    effective_policy = diagnostic_input.policy
    if policy is not None:
        effective_policy = DoctorSnapshotPolicy.from_mapping(policy)

    sources = list(diagnostic_input.sources)
    if diagnostic_input.repository_root and not sources:
        raise DoctorDiagnosticsError(
            "repository_root requires explicit relative source paths in sources"
        )
    if diagnostic_input.repository_root and sources:
        # Re-read bytes under the root for path-only units; otherwise keep admitted bytes.
        reloaded: list[DoctorSourceUnit] = []
        for unit in sources:
            if unit.byte_count == 0:
                loaded = _load_sources_from_repository(
                    diagnostic_input.repository_root,
                    [unit.path],
                    policy=effective_policy,
                    root_id=unit.root_id,
                )
                reloaded.extend(loaded)
            else:
                # Still verify the path cannot escape if a root is bound.
                _resolve_under_root(Path(diagnostic_input.repository_root), unit.path)
                reloaded.append(unit)
        sources = reloaded

    root_ids = {unit.root_id for unit in sources}
    if len(root_ids) > 1 and not effective_policy.allow_mixed_roots:
        raise DoctorDiagnosticsMixedRootError(
            "mixed-root sources require policy.allow_mixed_roots=true"
        )

    if diagnostic_input.claimed_tree_id and diagnostic_input.authority_roots.tree_id:
        if diagnostic_input.claimed_tree_id != diagnostic_input.authority_roots.tree_id:
            raise DoctorDiagnosticsStaleError("claimed_tree_id does not match authority tree_id")

    # Parse every admitted unit as inert text/bytes through existing adapters.
    adapter_results: list[ProgramASTAdapterResult] = []
    path_records: list[tuple[str, Any]] = []
    path_facts: dict[str, tuple[ProgramEvidenceFact, ...]] = {}
    for unit in sorted(sources, key=lambda item: item.path):
        if (
            unit.language
            and unit.language not in effective_policy.supported_languages
            and unit.language in SUPPORTED_ADAPTER_LANGUAGES
        ):
            # Policy may narrow supported languages.
            result = ProgramASTAdapterResult(
                path=unit.path,
                language=unit.language,
                status="unsupported",
                source_sha256=unit.source_sha256,
                blob_identity=unit.blob_identity,
                parser="none",
                diagnostics=(),
                generated=unit.generated,
            )
        else:
            result = adapt_program_source(
                _source_text(unit.source_bytes),
                path=unit.path,
                language=unit.language,
                blob_identity=unit.blob_identity,
                generated=unit.generated,
                max_source_bytes=effective_policy.max_source_bytes,
            )
        adapter_results.append(result)
        path_facts[unit.path] = result.facts
        if result.ast_record is not None:
            path_records.append((unit.path, result.ast_record))

    prior_index: AnalysisASTIndex | None = None
    rebuild_mode = "clean"
    prior_snapshot = diagnostic_input.previous_snapshot
    if previous is not None:
        if isinstance(previous, DoctorEvidenceSnapshot):
            prior_index = previous.ast_index
            rebuild_mode = "incremental"
        elif isinstance(previous, AnalysisASTIndex):
            prior_index = previous
            rebuild_mode = "incremental"
        else:
            raise DoctorDiagnosticsError("previous must be a snapshot or AST index")
    elif prior_snapshot is not None:
        if isinstance(prior_snapshot, DoctorEvidenceSnapshot):
            prior_index = prior_snapshot.ast_index
            rebuild_mode = "incremental"
        elif isinstance(prior_snapshot, Mapping) and "ast_index" in prior_snapshot:
            prior_index = AnalysisASTIndex.from_dict(prior_snapshot["ast_index"])
            rebuild_mode = "incremental"

    try:
        ast_index = build_analysis_ast_index(
            path_records,
            previous=prior_index,
        )
    except AnalysisASTIndexError as exc:
        raise DoctorDiagnosticsError(str(exc)) from exc

    receipts = tuple(_adapter_receipt(item) for item in adapter_results)
    query_index = _build_query_index(
        adapter_results, max_per_surface=effective_policy.max_query_results
    )

    findings: list[DoctorDiagnosticFinding] = []
    for result in adapter_results:
        findings.extend(_syntax_findings(result))
    findings.extend(_derived_call_contract_findings(query_index))

    for raw_trace in diagnostic_input.broken_traces:
        trace = _coerce_trace(raw_trace)
        findings.append(
            _join_trace(
                trace,
                path_facts=path_facts,
                index=ast_index,
                expected_roots=diagnostic_input.authority_roots,
            )
        )

    for failure in diagnostic_input.validation_failures:
        findings.append(
            _join_validation(failure, path_facts=path_facts, index=ast_index)
        )

    open_frontiers = _language_frontiers(adapter_results, effective_policy)
    if any(
        item.disposition
        in {FindingDisposition.ABSTAIN, FindingDisposition.APPROVAL_REQUIRED}
        for item in findings
    ):
        findings.append(
            DoctorDiagnosticFinding(
                kind=FindingKind.COMPLETENESS,
                disposition=FindingDisposition.ABSTAIN,
                message="open or unresolved findings prevent complete automatic repair authority",
                observation_refs=tuple(
                    item.finding_cid
                    for item in findings
                    if item.disposition
                    in {
                        FindingDisposition.ABSTAIN,
                        FindingDisposition.APPROVAL_REQUIRED,
                        FindingDisposition.UNKNOWN,
                    }
                ),
                open_frontier_refs=open_frontiers,
                details={"finding_count": len(findings)},
            )
        )

    if len(findings) > effective_policy.max_findings:
        raise DoctorDiagnosticsBoundsError("finding count exceeds max_findings")

    # Bind derived index roots onto authority metadata.  Path/blob order is
    # sorted so input permutation cannot change snapshot identity.
    ordered_units = tuple(sorted(sources, key=lambda item: item.path))
    authority = diagnostic_input.authority_roots.with_updates(
        policy_id=diagnostic_input.authority_roots.policy_id
        or effective_policy.policy_id,
        config_id=diagnostic_input.authority_roots.config_id
        or effective_policy.content_id,
        ast_index_id=ast_index.index_id,
        symbol_index_id=diagnostic_input.authority_roots.symbol_index_id
        or _identity(
            "doctor-symbol-index",
            {"paths": list(ast_index.paths), "index_id": ast_index.index_id},
        ),
        import_graph_id=diagnostic_input.authority_roots.import_graph_id
        or _identity(
            "doctor-import-graph",
            {
                "imports": [
                    hit.to_dict()
                    for hit in query_index.get(QuerySurface.IMPORTS.value, ())
                ]
            },
        ),
        parser_id=diagnostic_input.authority_roots.parser_id
        or "parser:program-ast-adapters@1",
        toolchain_id=diagnostic_input.authority_roots.toolchain_id
        or "toolchain:deterministic-doctor@1",
        file_root_id=diagnostic_input.authority_roots.file_root_id
        or _identity(
            "doctor-file-root",
            {
                "paths": [unit.path for unit in ordered_units],
                "blobs": [unit.blob_identity for unit in ordered_units],
            },
        ),
        blob_root_id=diagnostic_input.authority_roots.blob_root_id
        or _identity(
            "doctor-blob-root",
            [unit.blob_identity for unit in ordered_units],
        ),
    )
    if effective_policy.require_authority_roots:
        if not authority.parser_id or not authority.toolchain_id:
            raise DoctorDiagnosticsAuthorityError(
                "parser_id and toolchain_id must be bound"
            )

    completeness = {
        "path_count": len(sources),
        "indexed_path_count": len(ast_index.paths),
        "adapter_success_count": sum(
            1 for item in adapter_results if item.status == "success"
        ),
        "adapter_partial_count": sum(
            1 for item in adapter_results if item.status == "partial"
        ),
        "adapter_malformed_count": sum(
            1 for item in adapter_results if item.status == "malformed"
        ),
        "adapter_unsupported_count": sum(
            1 for item in adapter_results if item.status == "unsupported"
        ),
        "finding_count": len(findings),
        "open_frontier_count": len(open_frontiers),
        "import_count": len(query_index.get(QuerySurface.IMPORTS.value, ())),
        "export_count": len(query_index.get(QuerySurface.EXPORTS.value, ())),
        "alias_count": len(query_index.get(QuerySurface.ALIASES.value, ())),
        "wrapper_count": len(query_index.get(QuerySurface.WRAPPERS.value, ())),
        "entry_point_count": len(query_index.get(QuerySurface.ENTRY_POINTS.value, ())),
        "call_site_count": len(query_index.get(QuerySurface.CALL_SITES.value, ())),
        "expectation_ref_count": len(diagnostic_input.expectation_refs),
        "trace_join_count": len(diagnostic_input.broken_traces),
        "validation_join_count": len(diagnostic_input.validation_failures),
        "languages": sorted({item.language for item in adapter_results if item.language}),
        "root_ids": sorted(root_ids),
        "provider_call_count": 0,
        "source_write_count": 0,
        "complete_for_automatic_repair": not any(
            item.disposition
            in {
                FindingDisposition.ABSTAIN,
                FindingDisposition.APPROVAL_REQUIRED,
                FindingDisposition.UNKNOWN,
            }
            for item in findings
        )
        and not open_frontiers,
    }

    # Project finding surface into the query index for uniform queries.
    finding_hits = tuple(
        DoctorQueryHit(
            surface=QuerySurface.FINDINGS,
            path=item.path or "meta/findings",
            name=item.kind.value,
            target=item.finding_cid,
            owner=item.symbol,
            details={
                "disposition": item.disposition.value,
                "expectation_source": item.expectation_source.value,
                "message": item.message,
            },
        )
        for item in sorted(findings, key=lambda finding: finding.finding_cid)
    )
    frontier_hits = tuple(
        DoctorQueryHit(
            surface=QuerySurface.FRONTIERS,
            path="meta/frontiers",
            name=frontier,
            target=frontier,
        )
        for frontier in open_frontiers
    )
    full_query_index = {
        **query_index,
        QuerySurface.FINDINGS.value: finding_hits[: effective_policy.max_query_results],
        QuerySurface.FRONTIERS.value: frontier_hits[: effective_policy.max_query_results],
    }

    snapshot = DoctorEvidenceSnapshot(
        authority_roots=authority,
        policy=effective_policy,
        ast_index=ast_index,
        adapter_receipts=receipts,
        findings=tuple(findings),
        open_frontiers=open_frontiers,
        query_index=full_query_index,
        completeness=completeness,
        provider_call_count=0,
        source_write_count=0,
        rebuild_mode=rebuild_mode,
    )

    if diagnostic_input.claimed_snapshot_id:
        if diagnostic_input.claimed_snapshot_id not in {
            snapshot.snapshot_id,
            snapshot.snapshot_cid,
        }:
            raise DoctorDiagnosticsStaleError(
                "claimed_snapshot_id does not match compiled snapshot identity"
            )
    return snapshot


def diagnose_repository(
    sources: Iterable[Any] | None = None,
    *,
    repository_root: str = "",
    authority_roots: DoctorAuthorityRoots | Mapping[str, Any] | None = None,
    policy: DoctorSnapshotPolicy | Mapping[str, Any] | None = None,
    broken_traces: Sequence[BrokenContractTrace | Mapping[str, Any]] = (),
    validation_failures: Sequence[StructuredValidationFailure | Mapping[str, Any]] = (),
    expectation_refs: Sequence[str] = (),
    previous: DoctorEvidenceSnapshot | AnalysisASTIndex | None = None,
    claimed_tree_id: str = "",
) -> DoctorEvidenceSnapshot:
    """High-level entry point used by doctor inspect/explain/plan stages."""

    units: list[Any] = list(sources or ())
    diagnostic_input = DoctorDiagnosticInput(
        sources=tuple(units),
        authority_roots=DoctorAuthorityRoots.from_mapping(authority_roots),
        policy=DoctorSnapshotPolicy.from_mapping(policy),
        repository_root=repository_root,
        broken_traces=tuple(broken_traces),
        validation_failures=tuple(validation_failures),
        expectation_refs=tuple(expectation_refs),
        claimed_tree_id=claimed_tree_id,
        previous_snapshot=previous if isinstance(previous, DoctorEvidenceSnapshot) else None,
    )
    return compile_doctor_evidence_snapshot(diagnostic_input, previous=previous)


# Friendly aliases matching plan AST symbols and interface names.
compile_doctor_snapshot = compile_doctor_evidence_snapshot
DoctorRepositoryDiagnostics = DoctorEvidenceCompiler


__all__ = [
    "DEFAULT_OPEN_FRONTIERS",
    "DOCTOR_DIAGNOSTIC_FINDING_SCHEMA",
    "DOCTOR_DIAGNOSTIC_INPUT_SCHEMA",
    "DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA",
    "DOCTOR_REPOSITORY_DIAGNOSTICS_INTERFACE",
    "DOCTOR_REPOSITORY_DIAGNOSTICS_VERSION",
    "DOCTOR_SNAPSHOT_POLICY_SCHEMA",
    "SUPPORTED_ADAPTER_LANGUAGES",
    "DoctorAdapterReceipt",
    "DoctorAuthorityRoots",
    "DoctorDiagnosticFinding",
    "DoctorDiagnosticInput",
    "DoctorDiagnosticsAuthorityError",
    "DoctorDiagnosticsBoundsError",
    "DoctorDiagnosticsError",
    "DoctorDiagnosticsMixedRootError",
    "DoctorDiagnosticsStaleError",
    "DoctorDiagnosticsSymlinkError",
    "DoctorEvidenceCompiler",
    "DoctorEvidenceSnapshot",
    "DoctorQueryHit",
    "DoctorQueryResult",
    "DoctorRepositoryDiagnostics",
    "DoctorSnapshotPolicy",
    "DoctorSourceUnit",
    "ExpectationSourceKind",
    "FindingDisposition",
    "FindingKind",
    "QuerySurface",
    "StructuredValidationFailure",
    "compile_doctor_evidence_snapshot",
    "compile_doctor_snapshot",
    "diagnose_repository",
]
