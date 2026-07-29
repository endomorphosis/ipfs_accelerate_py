"""Cold static extraction of SwissKnife's expected MCP++ contracts.

The extractor reads source text; it never imports or executes SwissKnife.  It
normalizes the declarations that are useful to symbolic contract assurance:

* MCP++ interface and package descriptors;
* JSON schemas, capability registries, generated app bindings, and policies;
* connector calls, direct REST/fetch calls, and compatibility dispatch paths;
* executable contract-test expectations; and
* every relevant expression that cannot be resolved from local literals.

Literal declarations and local constant references are evaluated by a small,
deliberately incomplete JavaScript/TypeScript data parser.  Unsupported or
dynamic expressions are retained as :class:`UnresolvedContractValue` records
with exact source spans.  They are never guessed and never grant authority.
The normalized evidence is also projected into :class:`McpContractCatalog`,
where descriptor/test disagreements remain explicit contradictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .mcp_contract_catalog import (
    CATALOG_VERSION,
    CONTRACT_SCHEMA_VERSION,
    ContractRecord,
    ContractSourceKind,
    ContractSourceRecord,
    McpClaimFamily,
    McpContractCatalog,
    ReviewState,
    SourceAuthorityClass,
    build_default_mcp_contract_catalog,
    build_source_invalidators,
    detect_source_contradictions,
)


SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE: Final = "SwissKnifeContractExtractor@1"
SWISSKNIFE_CONTRACT_EXTRACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-contract-extraction@1"
)
SWISSKNIFE_SOURCE_SPAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/source-span@1"
)
SWISSKNIFE_UNRESOLVED_VALUE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/unresolved-contract-value@1"
)
SWISSKNIFE_DESCRIPTOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-mcp-descriptor@1"
)
SWISSKNIFE_INVOCATION_EDGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-invocation-edge@1"
)
SWISSKNIFE_EXPECTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-contract-expectation@1"
)
SWISSKNIFE_JSON_SCHEMA_RECORD: Final = (
    "ipfs_accelerate_py/agent-supervisor/swissknife-json-schema@1"
)
SWISSKNIFE_EXTRACTOR_VERSION: Final = "1"

DEFAULT_MAX_FILES: Final = 4096
DEFAULT_MAX_FILE_BYTES: Final = 4 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final = 64 * 1024 * 1024
HARD_MAX_FILES: Final = 100_000
HARD_MAX_FILE_BYTES: Final = 32 * 1024 * 1024
HARD_MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024

CANONICAL_SERVER_PACKAGES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
)

_DEFAULT_SOURCE_GLOBS: Final[tuple[str, ...]] = (
    "src/services/mcp/**/*",
    "src/services/apps/**/*",
    "contracts/**/*",
    "test/mcp-plus-plus/**/*",
    "tests/mcp-plus-plus/**/*",
    "scripts/**/*mcp*",
)
_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json"}
)
_TEST_MARKERS: Final[tuple[str, ...]] = (
    "/test/",
    "/tests/",
    ".test.",
    ".spec.",
    "_test.",
)
_POLICY_MARKERS: Final[tuple[str, ...]] = (
    "policy",
    "mediation",
    "authorization",
    "permission",
    "ucan",
    "deontic",
)
_COMPATIBILITY_MARKERS: Final[tuple[str, ...]] = (
    "compat",
    "legacy",
    "shim",
    "/api/v0/",
    "tools_dispatch",
    "tools_get_schema",
    "tools_list_categories",
    "tools_list_tools",
)


class SwissKnifeContractExtractorError(ValueError):
    """Malformed or unsafe extractor input."""


class SourceRole(str, Enum):
    """Closed source roles retained independently from catalog authority."""

    DESCRIPTOR = "descriptor"
    SCHEMA = "schema"
    CAPABILITY_REGISTRY = "capability_registry"
    CONNECTOR = "connector"
    POLICY_MEDIATOR = "policy_mediator"
    APP_BINDING = "app_binding"
    CONTRACT_TEST = "contract_test"
    MANIFEST = "manifest"
    OTHER = "other"


class ResolutionState(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class InvocationEdgeKind(str, Enum):
    """Invocation paths whose distinction matters to later bypass checks."""

    REGISTRATION = "registration"
    TOOLS_LIST = "tools_list"
    TOOLS_CALL = "tools_call"
    MCP_PLUS_PLUS = "mcp_plus_plus"
    HIERARCHICAL_DISPATCH = "hierarchical_dispatch"
    DIRECT_FETCH = "direct_fetch"
    DIRECT_REST = "direct_rest"
    COMPATIBILITY_ROUTE = "compatibility_route"
    APP_BINDING = "app_binding"
    POLICY_MEDIATION = "policy_mediation"
    TRANSPORT = "transport"


def _nonempty_text(value: Any, name: str, *, required: bool = False) -> str:
    if value is None:
        result = ""
    elif isinstance(value, (str, os.PathLike)):
        result = os.fspath(value).strip()
    else:
        raise SwissKnifeContractExtractorError(f"{name} must be text")
    if required and not result:
        raise SwissKnifeContractExtractorError(f"{name} is required")
    return result


def _json_value(value: Any) -> Any:
    """Return a deterministic, JSON-compatible copy."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _identity_value(value: Any) -> Any:
    """Adapt JSON values to the repository's float-free identity profile.

    Public normalized values retain JSON numbers.  Identity material tags
    finite floats by their shortest round-trippable decimal representation,
    avoiding both precision loss and the proof-contract canonicalizer's
    intentional rejection of bare IEEE-754 values.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise SwissKnifeContractExtractorError(
                "contract values cannot contain non-finite numbers"
            )
        return {"$number": repr(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    return content_identity({"value": _identity_value(_json_value(value))})


def _source_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(
        text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


@dataclass(frozen=True, order=True)
class SourceSpan:
    """One exact half-open source range (line and column are one-based)."""

    path: str
    start_offset: int
    end_offset: int
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __post_init__(self) -> None:
        path = _nonempty_text(self.path, "path", required=True).replace("\\", "/")
        object.__setattr__(self, "path", path)
        for name in (
            "start_offset",
            "end_offset",
            "start_line",
            "start_column",
            "end_line",
            "end_column",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SwissKnifeContractExtractorError(f"{name} must be an integer")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise SwissKnifeContractExtractorError("invalid source offsets")
        if min(
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column,
        ) < 1:
            raise SwissKnifeContractExtractorError(
                "source line and column values are one-based"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_SOURCE_SPAN_SCHEMA,
            "path": self.path,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class SwissKnifeSource:
    """One source body and explicit version binding."""

    path: str
    source: str | bytes
    source_version: str = ""
    role: SourceRole | str | None = None

    def __post_init__(self) -> None:
        path = _nonempty_text(self.path, "path", required=True).replace("\\", "/")
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise SwissKnifeContractExtractorError(
                "source paths must be relative and cannot traverse parents"
            )
        object.__setattr__(self, "path", path)
        if not isinstance(self.source, (str, bytes)):
            raise SwissKnifeContractExtractorError("source must be text or bytes")
        object.__setattr__(
            self,
            "source_version",
            _nonempty_text(self.source_version, "source_version"),
        )
        if self.role is not None and not isinstance(self.role, SourceRole):
            try:
                object.__setattr__(self, "role", SourceRole(str(self.role)))
            except ValueError as exc:
                raise SwissKnifeContractExtractorError(
                    f"unknown source role: {self.role!r}"
                ) from exc

    @property
    def text(self) -> str:
        if isinstance(self.source, str):
            return self.source
        try:
            return self.source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SwissKnifeContractExtractorError(
                f"{self.path}: source bytes must be valid UTF-8"
            ) from exc


@dataclass(frozen=True)
class UnresolvedContractValue:
    """A dynamic value retained as evidence, never replaced by a guess."""

    expression: str
    field_path: str
    reason_code: str
    source_span: SourceSpan
    declaration: str = ""
    unresolved_id: str = ""

    def __post_init__(self) -> None:
        expression = _nonempty_text(
            self.expression, "expression", required=True
        )
        field_path = _nonempty_text(
            self.field_path, "field_path", required=True
        )
        reason = _nonempty_text(
            self.reason_code, "reason_code", required=True
        )
        declaration = _nonempty_text(self.declaration, "declaration")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "field_path", field_path)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "declaration", declaration)
        derived = content_identity(
            {
                "schema": SWISSKNIFE_UNRESOLVED_VALUE_SCHEMA,
                "expression": expression,
                "field_path": field_path,
                "reason_code": reason,
                "source_span": self.source_span.to_dict(),
                "declaration": declaration,
            }
        )
        claimed = _nonempty_text(self.unresolved_id, "unresolved_id")
        if claimed and claimed != derived:
            raise SwissKnifeContractExtractorError(
                "unresolved_id does not match content"
            )
        object.__setattr__(self, "unresolved_id", derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_UNRESOLVED_VALUE_SCHEMA,
            "unresolved_id": self.unresolved_id,
            "expression": self.expression,
            "field_path": self.field_path,
            "reason_code": self.reason_code,
            "declaration": self.declaration,
            "source_span": self.source_span.to_dict(),
        }


@dataclass(frozen=True)
class MethodExpectation:
    name: str
    input_schema: str | None
    output_schema: str | None
    error_schemas: tuple[str, ...] = ()
    interaction_pattern: str = "request-response"
    streaming: bool = False
    defaults: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    policy_requirements: tuple[str, ...] = ()
    transport_expectations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _nonempty_text(self.name, "method name", required=True)
        )
        object.__setattr__(
            self,
            "input_schema",
            None
            if self.input_schema is None
            else _nonempty_text(self.input_schema, "input_schema"),
        )
        object.__setattr__(
            self,
            "output_schema",
            None
            if self.output_schema is None
            else _nonempty_text(self.output_schema, "output_schema"),
        )
        object.__setattr__(
            self,
            "error_schemas",
            tuple(sorted({str(item) for item in self.error_schemas if str(item)})),
        )
        pattern = _nonempty_text(self.interaction_pattern, "interaction_pattern")
        object.__setattr__(self, "interaction_pattern", pattern or "request-response")
        object.__setattr__(self, "streaming", bool(self.streaming or pattern == "stream"))
        object.__setattr__(
            self, "defaults", MappingProxyType(dict(_json_value(self.defaults)))
        )
        object.__setattr__(
            self,
            "policy_requirements",
            tuple(
                sorted({str(item) for item in self.policy_requirements if str(item)})
            ),
        )
        object.__setattr__(
            self,
            "transport_expectations",
            tuple(
                sorted(
                    {str(item) for item in self.transport_expectations if str(item)}
                )
            ),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_json_value(self.metadata)))
        )

    def semantic_value(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "error_schemas": list(self.error_schemas),
            "interaction_pattern": self.interaction_pattern,
            "streaming": self.streaming,
            "defaults": dict(self.defaults),
            "policy_requirements": list(self.policy_requirements),
            "transport_expectations": list(self.transport_expectations),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.semantic_value()
        result["source_span"] = (
            self.source_span.to_dict() if self.source_span is not None else None
        )
        return result


@dataclass(frozen=True)
class DescriptorExpectation:
    """Normalized MCP++ interface/descriptor declaration."""

    declaration: str
    package_id: str
    name: str
    namespace: str
    version: str
    interface_cid: str | None
    descriptor_id: str
    methods: tuple[MethodExpectation, ...]
    errors: tuple[Mapping[str, Any], ...] = ()
    requires: tuple[str, ...] = ()
    compatible_with: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    streaming: bool = False
    policy_requirements: tuple[str, ...] = ()
    transport_expectations: tuple[str, ...] = ()
    schema_refs: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    source_version: str = ""
    schema_version: str = ""
    source_role: SourceRole = SourceRole.DESCRIPTOR
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        for name in ("declaration", "name", "namespace", "version"):
            object.__setattr__(
                self,
                name,
                _nonempty_text(getattr(self, name), name, required=True),
            )
        package_id = _nonempty_text(self.package_id, "package_id")
        object.__setattr__(self, "package_id", package_id or "unknown")
        object.__setattr__(
            self,
            "interface_cid",
            None
            if self.interface_cid is None
            else _nonempty_text(self.interface_cid, "interface_cid"),
        )
        descriptor_id = _nonempty_text(self.descriptor_id, "descriptor_id")
        object.__setattr__(
            self, "descriptor_id", descriptor_id or f"{self.name}@{self.version}"
        )
        object.__setattr__(
            self, "methods", tuple(sorted(self.methods, key=lambda item: item.name))
        )
        object.__setattr__(
            self,
            "errors",
            tuple(
                MappingProxyType(dict(_json_value(item)))
                for item in sorted(
                    self.errors,
                    key=lambda item: (
                        str(item.get("name", "")),
                        str(item.get("code", "")),
                    ),
                )
            ),
        )
        for name in (
            "requires",
            "compatible_with",
            "supersedes",
            "policy_requirements",
            "transport_expectations",
        ):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        {
                            str(item)
                            for item in getattr(self, name)
                            if str(item)
                        }
                    )
                ),
            )
        object.__setattr__(
            self,
            "streaming",
            bool(self.streaming or any(method.streaming for method in self.methods)),
        )
        object.__setattr__(
            self, "schema_refs", MappingProxyType(dict(_json_value(self.schema_refs)))
        )
        object.__setattr__(
            self,
            "source_version",
            _nonempty_text(self.source_version, "source_version")
            or self.version,
        )
        object.__setattr__(
            self,
            "schema_version",
            _nonempty_text(self.schema_version, "schema_version")
            or self.version,
        )

    @property
    def subject(self) -> str:
        # The three base provider descriptors have a stable package-level
        # subject used by conformance tests.  Package-specific interop
        # descriptors are distinct interfaces and must not conflict merely
        # because they ultimately target the same provider repository.
        base_names = {
            "ipfs_accelerate_py": ("ipfs-accelerate", "com.ipfs.accelerate"),
            "ipfs_datasets_py": ("ipfs-datasets", "com.ipfs.datasets"),
            "ipfs_kit_py": ("ipfs-kit", "com.ipfs.kit"),
        }
        if base_names.get(self.package_id) == (self.name, self.namespace):
            return f"mcp-interface:{self.package_id}"
        return (
            f"mcp-interface:{self.package_id}:"
            f"{self.namespace}:{self.name}"
        )

    def semantic_value(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "name": self.name,
            "namespace": self.namespace,
            "version": self.version,
            "interface_cid": self.interface_cid,
            "descriptor_id": self.descriptor_id,
            "methods": [method.semantic_value() for method in self.methods],
            "errors": [dict(item) for item in self.errors],
            "requires": list(self.requires),
            "compatible_with": list(self.compatible_with),
            "supersedes": list(self.supersedes),
            "streaming": self.streaming,
            "policy_requirements": list(self.policy_requirements),
            "transport_expectations": list(self.transport_expectations),
            "schema_refs": dict(self.schema_refs),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_DESCRIPTOR_SCHEMA,
            "declaration": self.declaration,
            **self.semantic_value(),
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "source_role": self.source_role.value,
            "source_span": (
                self.source_span.to_dict() if self.source_span is not None else None
            ),
        }


@dataclass(frozen=True)
class InvocationEdge:
    source: str
    target: str | None
    kind: InvocationEdgeKind
    operation: str = ""
    http_method: str = ""
    transport: str = ""
    compatibility: bool = False
    bypass_candidate: bool = False
    policy_mediated: bool | None = None
    unresolved_id: str = ""
    source_span: SourceSpan | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    edge_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _nonempty_text(self.source, "source", required=True)
        )
        object.__setattr__(
            self,
            "target",
            None
            if self.target is None
            else _nonempty_text(self.target, "target", required=True),
        )
        if not isinstance(self.kind, InvocationEdgeKind):
            try:
                object.__setattr__(self, "kind", InvocationEdgeKind(str(self.kind)))
            except ValueError as exc:
                raise SwissKnifeContractExtractorError(
                    f"unknown invocation edge kind: {self.kind!r}"
                ) from exc
        for name in ("operation", "http_method", "transport", "unresolved_id"):
            object.__setattr__(
                self, name, _nonempty_text(getattr(self, name), name)
            )
        object.__setattr__(self, "compatibility", bool(self.compatibility))
        object.__setattr__(self, "bypass_candidate", bool(self.bypass_candidate))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_json_value(self.metadata)))
        )
        if self.target is None and not self.unresolved_id:
            raise SwissKnifeContractExtractorError(
                "an unresolved edge target must reference unresolved_id"
            )
        payload = {
            "schema": SWISSKNIFE_INVOCATION_EDGE_SCHEMA,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "operation": self.operation,
            "http_method": self.http_method,
            "transport": self.transport,
            "compatibility": self.compatibility,
            "bypass_candidate": self.bypass_candidate,
            "policy_mediated": self.policy_mediated,
            "unresolved_id": self.unresolved_id,
            "source_span": (
                self.source_span.to_dict() if self.source_span is not None else None
            ),
            "metadata": dict(self.metadata),
        }
        derived = content_identity(_identity_value(payload))
        claimed = _nonempty_text(self.edge_id, "edge_id")
        if claimed and claimed != derived:
            raise SwissKnifeContractExtractorError("edge_id does not match content")
        object.__setattr__(self, "edge_id", derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_INVOCATION_EDGE_SCHEMA,
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "operation": self.operation,
            "http_method": self.http_method,
            "transport": self.transport,
            "compatibility": self.compatibility,
            "bypass_candidate": self.bypass_candidate,
            "policy_mediated": self.policy_mediated,
            "unresolved_id": self.unresolved_id,
            "source_span": (
                self.source_span.to_dict() if self.source_span is not None else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContractExpectation:
    """One normalized field/default/test/policy/transport expectation."""

    subject: str
    field_path: str
    value: Any
    source_kind: ContractSourceKind
    source_version: str
    schema_version: str
    source_span: SourceSpan
    package_id: str = ""
    tool_name: str = ""
    claim_family: McpClaimFamily = McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for name in ("subject", "field_path", "source_version", "schema_version"):
            object.__setattr__(
                self,
                name,
                _nonempty_text(getattr(self, name), name, required=True),
            )
        if not isinstance(self.source_kind, ContractSourceKind):
            object.__setattr__(
                self, "source_kind", ContractSourceKind(str(self.source_kind))
            )
        if not isinstance(self.claim_family, McpClaimFamily):
            object.__setattr__(
                self, "claim_family", McpClaimFamily(str(self.claim_family))
            )
        object.__setattr__(self, "value", _json_value(self.value))
        object.__setattr__(
            self, "package_id", _nonempty_text(self.package_id, "package_id")
        )
        object.__setattr__(
            self, "tool_name", _nonempty_text(self.tool_name, "tool_name")
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_json_value(self.metadata)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_EXPECTATION_SCHEMA,
            "subject": self.subject,
            "field_path": self.field_path,
            "value": self.value,
            "source_kind": self.source_kind.value,
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "tool_name": self.tool_name,
            "claim_family": self.claim_family.value,
            "metadata": dict(self.metadata),
            "source_span": self.source_span.to_dict(),
        }


@dataclass(frozen=True)
class JsonSchemaExpectation:
    path: str
    schema_id: str
    schema_version: str
    title: str
    defaults: Mapping[str, Any]
    error_values: tuple[str, ...]
    source_span: SourceSpan
    source_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_JSON_SCHEMA_RECORD,
            "path": self.path,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "title": self.title,
            "defaults": dict(_json_value(self.defaults)),
            "error_values": list(self.error_values),
            "source_version": self.source_version,
            "source_span": self.source_span.to_dict(),
        }


@dataclass(frozen=True)
class SwissKnifeContractExtraction:
    descriptors: tuple[DescriptorExpectation, ...]
    invocation_edges: tuple[InvocationEdge, ...]
    expectations: tuple[ContractExpectation, ...]
    schemas: tuple[JsonSchemaExpectation, ...]
    unresolved_values: tuple[UnresolvedContractValue, ...]
    catalog: McpContractCatalog
    source_versions: Mapping[str, str]
    extractor_version: str = SWISSKNIFE_EXTRACTOR_VERSION
    extraction_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "descriptors",
            tuple(
                sorted(
                    self.descriptors,
                    key=lambda item: (
                        item.package_id,
                        item.namespace,
                        item.name,
                        item.source_span.path if item.source_span else "",
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "invocation_edges",
            tuple(sorted(self.invocation_edges, key=lambda item: item.edge_id)),
        )
        object.__setattr__(
            self,
            "expectations",
            tuple(
                sorted(
                    self.expectations,
                    key=lambda item: (
                        item.subject,
                        item.field_path,
                        item.source_span.path,
                        item.source_span.start_offset,
                    ),
                )
            ),
        )
        object.__setattr__(
            self, "schemas", tuple(sorted(self.schemas, key=lambda item: item.path))
        )
        object.__setattr__(
            self,
            "unresolved_values",
            tuple(
                sorted(
                    self.unresolved_values,
                    key=lambda item: (
                        item.source_span.path,
                        item.source_span.start_offset,
                        item.field_path,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "source_versions",
            MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in sorted(self.source_versions.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _nonempty_text(
                self.extractor_version, "extractor_version", required=True
            ),
        )
        payload = self._identity_payload()
        derived = content_identity(_identity_value(payload))
        claimed = _nonempty_text(self.extraction_id, "extraction_id")
        if claimed and claimed != derived:
            raise SwissKnifeContractExtractorError(
                "extraction_id does not match content"
            )
        object.__setattr__(self, "extraction_id", derived)

    @property
    def canonical_packages_present(self) -> tuple[str, ...]:
        present = {item.package_id for item in self.descriptors}
        return tuple(pkg for pkg in CANONICAL_SERVER_PACKAGES if pkg in present)

    @property
    def missing_canonical_packages(self) -> tuple[str, ...]:
        present = set(self.canonical_packages_present)
        return tuple(pkg for pkg in CANONICAL_SERVER_PACKAGES if pkg not in present)

    @property
    def contradictions(self):
        return self.catalog.contradictions

    def require_canonical_packages(self) -> "SwissKnifeContractExtraction":
        """Fail closed unless all three reviewed provider descriptors exist."""

        if self.missing_canonical_packages:
            raise SwissKnifeContractExtractorError(
                "missing canonical package descriptors: "
                + ", ".join(self.missing_canonical_packages)
            )
        return self

    def edges_of_kind(
        self, kind: InvocationEdgeKind | str
    ) -> tuple[InvocationEdge, ...]:
        kind_e = kind if isinstance(kind, InvocationEdgeKind) else InvocationEdgeKind(kind)
        return tuple(edge for edge in self.invocation_edges if edge.kind is kind_e)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SWISSKNIFE_CONTRACT_EXTRACTION_SCHEMA,
            "interface": SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE,
            "extractor_version": self.extractor_version,
            "source_versions": dict(self.source_versions),
            "descriptors": [item.to_dict() for item in self.descriptors],
            "invocation_edges": [item.to_dict() for item in self.invocation_edges],
            "expectations": [item.to_dict() for item in self.expectations],
            "schemas": [item.to_dict() for item in self.schemas],
            "unresolved_values": [
                item.to_dict() for item in self.unresolved_values
            ],
            "catalog_id": self.catalog.catalog_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "extraction_id": self.extraction_id,
            "canonical_packages": list(self.canonical_packages_present),
            "missing_canonical_packages": list(self.missing_canonical_packages),
            "catalog": self.catalog.to_dict(),
        }


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class _Dynamic:
    expression: str
    start: int
    end: int
    reason: str = "dynamic_expression"


@dataclass(frozen=True)
class _Declaration:
    name: str
    value: Any
    start: int
    end: int
    exported: bool
    type_name: str


def _tokenize(source: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    i = 0
    length = len(source)
    punctuation = set("{}[]():,;.=?!+-*/<>|&")
    while i < length:
        char = source[i]
        if char.isspace():
            i += 1
            continue
        if source.startswith("//", i):
            end = source.find("\n", i + 2)
            i = length if end < 0 else end + 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = length if end < 0 else end + 2
            continue
        if char in ("'", '"', "`"):
            quote = char
            start = i
            i += 1
            escaped = False
            interpolation = False
            while i < length:
                current = source[i]
                if escaped:
                    escaped = False
                    i += 1
                    continue
                if current == "\\":
                    escaped = True
                    i += 1
                    continue
                if quote == "`" and source.startswith("${", i):
                    interpolation = True
                if current == quote:
                    i += 1
                    break
                i += 1
            tokens.append(
                _Token(
                    "template_dynamic"
                    if quote == "`" and interpolation
                    else "string",
                    source[start:i],
                    start,
                    i,
                )
            )
            continue
        if char.isdigit() or (
            char == "."
            and i + 1 < length
            and source[i + 1].isdigit()
        ):
            start = i
            i += 1
            while i < length and (
                source[i].isalnum() or source[i] in "._xX+-"
            ):
                if source[i] in "+-" and source[i - 1] not in "eE":
                    break
                i += 1
            tokens.append(_Token("number", source[start:i], start, i))
            continue
        if char.isalpha() or char in "_$":
            start = i
            i += 1
            while i < length and (source[i].isalnum() or source[i] in "_$"):
                i += 1
            tokens.append(_Token("identifier", source[start:i], start, i))
            continue
        if source.startswith("...", i):
            tokens.append(_Token("punct", "...", i, i + 3))
            i += 3
            continue
        if source.startswith("=>", i):
            tokens.append(_Token("punct", "=>", i, i + 2))
            i += 2
            continue
        if source.startswith("?.", i):
            tokens.append(_Token("punct", "?.", i, i + 2))
            i += 2
            continue
        if char in punctuation:
            tokens.append(_Token("punct", char, i, i + 1))
            i += 1
            continue
        tokens.append(_Token("other", char, i, i + 1))
        i += 1
    return tuple(tokens)


def _decode_string(token: _Token) -> str:
    raw = token.value
    if len(raw) < 2:
        return raw
    if raw[0] == "`":
        body = raw[1:-1]
        return bytes(body, "utf-8").decode("unicode_escape")
    try:
        return json.loads(raw) if raw[0] == '"' else bytes(
            raw[1:-1], "utf-8"
        ).decode("unicode_escape")
    except (ValueError, UnicodeDecodeError):
        return raw[1:-1]


class _LiteralParser:
    def __init__(
        self,
        source: str,
        tokens: Sequence[_Token],
        environment: Mapping[str, Any],
        start: int,
    ) -> None:
        self.source = source
        self.tokens = tokens
        self.environment = environment
        self.index = start

    def current(self) -> _Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def parse(self) -> Any:
        token = self.current()
        if token is None:
            return _Dynamic("", len(self.source), len(self.source), "missing_expression")
        if token.value == "{":
            return self._parse_object()
        if token.value == "[":
            return self._parse_array()
        if token.kind == "string":
            self.index += 1
            return _decode_string(token)
        if token.kind == "template_dynamic":
            self.index += 1
            return _Dynamic(token.value, token.start, token.end, "template_expression")
        if token.kind == "number":
            self.index += 1
            cleaned = token.value.replace("_", "")
            try:
                if any(char in cleaned for char in ".eE"):
                    return float(cleaned)
                return int(cleaned, 0)
            except ValueError:
                return _Dynamic(token.value, token.start, token.end, "invalid_number")
        if token.kind == "identifier":
            if token.value == "true":
                self.index += 1
                return True
            if token.value == "false":
                self.index += 1
                return False
            if token.value in {"null", "undefined"}:
                self.index += 1
                return None if token.value == "null" else _Dynamic(
                    token.value, token.start, token.end, "undefined_value"
                )
            if token.value in self.environment:
                self.index += 1
                return self.environment[token.value]
        return self._consume_dynamic()

    def _parse_object(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        self.index += 1
        while self.current() is not None and self.current().value != "}":
            iteration_start = self.index
            token = self.current()
            assert token is not None
            if token.value == ",":
                self.index += 1
                continue
            if token.value == "...":
                spread_start = token.start
                self.index += 1
                value = self.parse()
                if isinstance(value, Mapping):
                    result.update(value)
                else:
                    end = self.tokens[self.index - 1].end if self.index else token.end
                    result[f"__unresolved_spread_{spread_start}"] = (
                        value
                        if isinstance(value, _Dynamic)
                        else _Dynamic(
                            self.source[spread_start:end],
                            spread_start,
                            end,
                            "dynamic_spread",
                        )
                    )
                continue
            if token.kind in {"string", "identifier", "number"}:
                key = (
                    _decode_string(token)
                    if token.kind == "string"
                    else token.value
                )
                key_start = token.start
                self.index += 1
                current = self.current()
                if current is not None and current.value == "?":
                    self.index += 1
                    current = self.current()
                if current is not None and current.value == ":":
                    self.index += 1
                    result[str(key)] = self.parse()
                elif token.kind == "identifier" and token.value in self.environment:
                    result[str(key)] = self.environment[token.value]
                else:
                    result[str(key)] = _Dynamic(
                        self.source[key_start : token.end],
                        key_start,
                        token.end,
                        "dynamic_shorthand",
                    )
            elif token.value == "[":
                dynamic = self._consume_dynamic(stop_values={":"})
                if self.current() is not None and self.current().value == ":":
                    self.index += 1
                    value = self.parse()
                else:
                    value = dynamic
                result[f"__computed_{dynamic.start}"] = value
            else:
                dynamic = self._consume_dynamic(stop_values={",", "}"})
                result[f"__unparsed_{dynamic.start}"] = dynamic
            if self.current() is not None and self.current().value == ",":
                self.index += 1
            if self.index == iteration_start:
                # Malformed or unsupported syntax must be bounded.  Retain the
                # token as unresolved and advance instead of looping forever.
                stuck = self.current()
                assert stuck is not None
                result[f"__unparsed_{stuck.start}"] = _Dynamic(
                    stuck.value, stuck.start, stuck.end, "unsupported_object_syntax"
                )
                self.index += 1
        if self.current() is not None and self.current().value == "}":
            self.index += 1
        return result

    def _parse_array(self) -> list[Any]:
        result: list[Any] = []
        self.index += 1
        while self.current() is not None and self.current().value != "]":
            iteration_start = self.index
            token = self.current()
            assert token is not None
            if token.value == ",":
                self.index += 1
                continue
            if token.value == "...":
                spread_start = token.start
                self.index += 1
                value = self.parse()
                if isinstance(value, (list, tuple)):
                    result.extend(value)
                else:
                    end = self.tokens[self.index - 1].end if self.index else token.end
                    result.append(
                        value
                        if isinstance(value, _Dynamic)
                        else _Dynamic(
                            self.source[spread_start:end],
                            spread_start,
                            end,
                            "dynamic_spread",
                        )
                    )
            else:
                result.append(self.parse())
            if self.current() is not None and self.current().value == ",":
                self.index += 1
            if self.index == iteration_start:
                stuck = self.current()
                assert stuck is not None
                result.append(
                    _Dynamic(
                        stuck.value,
                        stuck.start,
                        stuck.end,
                        "unsupported_array_syntax",
                    )
                )
                self.index += 1
        if self.current() is not None and self.current().value == "]":
            self.index += 1
        return result

    def _consume_dynamic(
        self, *, stop_values: set[str] | None = None
    ) -> _Dynamic:
        stop_values = stop_values or {",", "}", "]", ";"}
        start_token = self.current()
        if start_token is None:
            return _Dynamic("", len(self.source), len(self.source), "missing_expression")
        start = start_token.start
        end = start_token.end
        depth = 0
        while self.current() is not None:
            token = self.current()
            assert token is not None
            if depth == 0 and token.value in stop_values:
                break
            if token.value in {"(", "{", "["}:
                depth += 1
            elif token.value in {")", "}", "]"}:
                if depth == 0:
                    break
                depth -= 1
            end = token.end
            self.index += 1
        if self.index < len(self.tokens) and self.tokens[self.index].start == start:
            # An unexpected unmatched closing delimiter is itself the dynamic
            # expression.  Consuming it keeps callers progress-safe.
            end = self.tokens[self.index].end
            self.index += 1
        expression = self.source[start:end].strip()
        return _Dynamic(expression or start_token.value, start, end)


def _parse_declarations(source: str, tokens: Sequence[_Token]) -> tuple[_Declaration, ...]:
    environment: dict[str, Any] = {}
    declarations: list[_Declaration] = []
    i = 0
    while i < len(tokens):
        exported = False
        start_i = i
        if tokens[i].value == "export":
            exported = True
            i += 1
            if i >= len(tokens):
                break
        if tokens[i].value not in {"const", "let", "var"}:
            i = start_i + 1
            continue
        i += 1
        if i >= len(tokens) or tokens[i].kind != "identifier":
            i = start_i + 1
            continue
        name_token = tokens[i]
        name = name_token.value
        i += 1
        type_tokens: list[str] = []
        if i < len(tokens) and tokens[i].value == ":":
            i += 1
            depth = 0
            while i < len(tokens):
                token = tokens[i]
                if token.value in {"<", "[", "{", "("}:
                    depth += 1
                elif token.value in {">", "]", "}", ")"} and depth:
                    depth -= 1
                if token.value == "=" and depth == 0:
                    break
                type_tokens.append(token.value)
                i += 1
        if i >= len(tokens) or tokens[i].value != "=":
            i = start_i + 1
            continue
        i += 1
        parser = _LiteralParser(source, tokens, environment, i)
        value = parser.parse()
        end_i = parser.index
        # Strip TypeScript's trailing ``as const`` / ``satisfies Type`` from
        # the declaration range without changing the resolved literal.
        while end_i < len(tokens) and tokens[end_i].value in {"as", "const"}:
            end_i += 1
        declaration_end = (
            tokens[end_i - 1].end if end_i > i else name_token.end
        )
        environment[name] = value
        declarations.append(
            _Declaration(
                name=name,
                value=value,
                start=tokens[start_i].start,
                end=declaration_end,
                exported=exported,
                type_name="".join(type_tokens),
            )
        )
        i = max(end_i, start_i + 1)
    return tuple(declarations)


class _SpanFactory:
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.source = source
        self._line_starts = [0]
        self._line_starts.extend(
            match.end() for match in re.finditer(r"\n", source)
        )

    def _line_column(self, offset: int) -> tuple[int, int]:
        import bisect

        index = bisect.bisect_right(self._line_starts, offset) - 1
        return index + 1, offset - self._line_starts[index] + 1

    def make(self, start: int, end: int) -> SourceSpan:
        start = max(0, min(start, len(self.source)))
        end = max(start, min(end, len(self.source)))
        start_line, start_column = self._line_column(start)
        end_line, end_column = self._line_column(end)
        return SourceSpan(
            path=self.path,
            start_offset=start,
            end_offset=end,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
        )


def _role_for_path(path: str) -> SourceRole:
    normalized = "/" + path.replace("\\", "/").lower().lstrip("/")
    name = PurePosixPath(normalized).name
    if any(marker in normalized for marker in _TEST_MARKERS):
        return SourceRole.CONTRACT_TEST
    if normalized.endswith(".json"):
        if "manifest" in name:
            return SourceRole.MANIFEST
        return SourceRole.SCHEMA
    if "connector" in name or "transport" in name:
        return SourceRole.CONNECTOR
    if "registry" in name:
        return SourceRole.CAPABILITY_REGISTRY
    if "binding" in name or "manifest" in name or "/apps/" in normalized:
        return SourceRole.APP_BINDING
    if any(marker in name for marker in _POLICY_MARKERS):
        return SourceRole.POLICY_MEDIATOR
    if "descriptor" in name or normalized.endswith("/mcp-plus-plus.ts"):
        return SourceRole.DESCRIPTOR
    return SourceRole.OTHER


def _source_kind_for_role(role: SourceRole) -> ContractSourceKind:
    return {
        SourceRole.DESCRIPTOR: ContractSourceKind.MCP_IDL,
        SourceRole.SCHEMA: ContractSourceKind.JSON_SCHEMA,
        SourceRole.CAPABILITY_REGISTRY: ContractSourceKind.REGISTRATION,
        SourceRole.CONNECTOR: ContractSourceKind.TYPED_INTERFACE,
        SourceRole.POLICY_MEDIATOR: ContractSourceKind.POLICY_CONTRACT,
        SourceRole.APP_BINDING: ContractSourceKind.MANIFEST,
        SourceRole.CONTRACT_TEST: ContractSourceKind.CONFORMANCE_TEST,
        SourceRole.MANIFEST: ContractSourceKind.MANIFEST,
        SourceRole.OTHER: ContractSourceKind.TYPED_INTERFACE,
    }[role]


def _package_id(*values: Any) -> str:
    blob = " ".join(str(value or "") for value in values).lower()
    if "ipfs_accelerate" in blob or "ipfs-accelerate" in blob:
        return "ipfs_accelerate_py"
    if "ipfs_datasets" in blob or "ipfs-datasets" in blob:
        return "ipfs_datasets_py"
    if "ipfs_kit" in blob or "ipfs-kit" in blob or "com.ipfs.kit" in blob:
        return "ipfs_kit_py"
    if "meta_wearables_dat_android" in blob or "meta-wearables-dat-android" in blob:
        return "meta-wearables-dat-android"
    if "meta_wearables_dat_ios" in blob or "meta-wearables-dat-ios" in blob:
        return "meta-wearables-dat-ios"
    if "mcpplusplus" in blob or "mcp-plus-plus" in blob:
        return "Mcp-Plus-Plus"
    return "unknown"


def _literal(value: Any) -> Any:
    if isinstance(value, _Dynamic):
        return None
    if isinstance(value, Mapping):
        return {
            key: _literal(item)
            for key, item in value.items()
            if not str(key).startswith("__")
        }
    if isinstance(value, list):
        return [_literal(item) for item in value if not isinstance(item, _Dynamic)]
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _collect_unresolved(
    value: Any,
    *,
    field_path: str,
    declaration: str,
    source: str,
    spans: _SpanFactory,
    output: list[UnresolvedContractValue],
) -> None:
    if isinstance(value, _Dynamic):
        output.append(
            UnresolvedContractValue(
                expression=value.expression,
                field_path=field_path,
                reason_code=value.reason,
                declaration=declaration,
                source_span=spans.make(value.start, value.end),
            )
        )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{field_path}.{key}" if field_path else str(key)
            _collect_unresolved(
                item,
                field_path=child,
                declaration=declaration,
                source=source,
                spans=spans,
                output=output,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_unresolved(
                item,
                field_path=f"{field_path}[{index}]",
                declaration=declaration,
                source=source,
                spans=spans,
                output=output,
            )


def _method_expectations(
    methods: Any,
    *,
    span: SourceSpan,
    interface_requires: tuple[str, ...],
    transport_expectations: tuple[str, ...],
) -> tuple[MethodExpectation, ...]:
    if not isinstance(methods, list):
        return ()
    result: list[MethodExpectation] = []
    for method in methods:
        if not isinstance(method, Mapping):
            continue
        name = method.get("name")
        if not isinstance(name, str) or not name:
            continue
        pattern = (
            method.get("interaction_pattern")
            if isinstance(method.get("interaction_pattern"), str)
            else "request-response"
        )
        defaults = {
            key: _literal(value)
            for key, value in method.items()
            if "default" in str(key).lower() and not isinstance(value, _Dynamic)
        }
        metadata = {
            key: _literal(value)
            for key, value in method.items()
            if key
            not in {
                "name",
                "input_schema_cid",
                "inputSchema",
                "input_schema",
                "output_schema_cid",
                "outputSchema",
                "output_schema",
                "error_schema_cids",
                "errors",
                "interaction_pattern",
            }
            and not isinstance(value, _Dynamic)
        }
        result.append(
            MethodExpectation(
                name=name,
                input_schema=next(
                    (
                        value
                        for value in (
                            method.get("input_schema_cid"),
                            method.get("inputSchema"),
                            method.get("input_schema"),
                        )
                        if isinstance(value, str)
                    ),
                    None,
                ),
                output_schema=next(
                    (
                        value
                        for value in (
                            method.get("output_schema_cid"),
                            method.get("outputSchema"),
                            method.get("output_schema"),
                        )
                        if isinstance(value, str)
                    ),
                    None,
                ),
                error_schemas=_strings(
                    method.get("error_schema_cids", method.get("errors", ()))
                ),
                interaction_pattern=pattern,
                streaming=pattern in {"stream", "event"},
                defaults=defaults,
                policy_requirements=tuple(
                    value
                    for value in interface_requires
                    if any(marker in value.lower() for marker in _POLICY_MARKERS)
                ),
                transport_expectations=transport_expectations,
                metadata=metadata,
                source_span=span,
            )
        )
    return tuple(result)


def _descriptor_from_declaration(
    declaration: _Declaration,
    *,
    role: SourceRole,
    source_version: str,
    path: str,
    span: SourceSpan,
) -> DescriptorExpectation | None:
    if not isinstance(declaration.value, Mapping):
        return None
    raw = declaration.value
    interface: Mapping[str, Any] = raw
    descriptor_id = ""
    schema_refs: Mapping[str, Any] = {}
    if isinstance(raw.get("interface"), Mapping):
        interface = raw["interface"]
        descriptor_id = (
            raw.get("descriptor_id")
            if isinstance(raw.get("descriptor_id"), str)
            else ""
        )
        if isinstance(raw.get("schema_refs"), Mapping):
            schema_refs = _literal(raw["schema_refs"])
    has_shape = (
        isinstance(interface.get("name"), str)
        and isinstance(interface.get("namespace"), str)
        and isinstance(interface.get("version"), str)
        and isinstance(interface.get("methods"), list)
    )
    typed_shape = "MCPPPInterfaceDescriptor" in declaration.type_name
    if not has_shape and not typed_shape:
        return None
    if not has_shape:
        return None
    name = str(interface["name"])
    namespace = str(interface["namespace"])
    version = str(interface["version"])
    package_id = _package_id(
        declaration.name,
        name,
        namespace,
        descriptor_id,
        path,
        _literal(raw.get("metadata")),
    )
    requires = _strings(interface.get("requires", ()))
    compatibility = (
        interface.get("compatibility")
        if isinstance(interface.get("compatibility"), Mapping)
        else {}
    )
    transport_expectations: list[str] = []
    for requirement in requires:
        lowered = requirement.lower()
        if "p2p" in lowered or "transport" in lowered:
            transport_expectations.append(requirement)
    methods = _method_expectations(
        interface.get("methods"),
        span=span,
        interface_requires=requires,
        transport_expectations=tuple(transport_expectations),
    )
    errors = tuple(
        _literal(item)
        for item in interface.get("errors", ())
        if isinstance(item, Mapping)
    )
    return DescriptorExpectation(
        declaration=declaration.name,
        package_id=package_id,
        name=name,
        namespace=namespace,
        version=version,
        interface_cid=(
            interface.get("interface_cid")
            if isinstance(interface.get("interface_cid"), str)
            else None
        ),
        descriptor_id=descriptor_id or f"{name}@{version}",
        methods=methods,
        errors=errors,
        requires=requires,
        compatible_with=_strings(compatibility.get("compatible_with", ())),
        supersedes=_strings(compatibility.get("supersedes", ())),
        streaming=any(method.streaming for method in methods),
        policy_requirements=tuple(
            requirement
            for requirement in requires
            if any(marker in requirement.lower() for marker in _POLICY_MARKERS)
        ),
        transport_expectations=tuple(transport_expectations),
        schema_refs=schema_refs,
        source_version=source_version or version,
        schema_version=version,
        source_role=role,
        source_span=span,
    )


def _deduplicate_descriptors(
    descriptors: Iterable[DescriptorExpectation],
) -> tuple[DescriptorExpectation, ...]:
    result: dict[tuple[str, str, str, str], DescriptorExpectation] = {}
    for descriptor in descriptors:
        path = descriptor.source_span.path if descriptor.source_span else ""
        key = (path, descriptor.package_id, descriptor.namespace, descriptor.version)
        current = result.get(key)
        if current is None:
            result[key] = descriptor
            continue
        # Prefer the wrapper declaration because it supplies descriptor_id and
        # schema_refs; preserve interface methods from either declaration.
        if descriptor.schema_refs or (
            descriptor.descriptor_id != f"{descriptor.name}@{descriptor.version}"
        ):
            result[key] = replace(
                descriptor,
                methods=descriptor.methods or current.methods,
                errors=descriptor.errors or current.errors,
                requires=descriptor.requires or current.requires,
            )
    return tuple(result.values())


def _walk_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_objects(item)


def _is_compatibility(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _COMPATIBILITY_MARKERS)


def _app_binding_edges(
    declarations: Sequence[_Declaration],
    *,
    path: str,
    spans: _SpanFactory,
) -> tuple[InvocationEdge, ...]:
    edges: list[InvocationEdge] = []
    for declaration in declarations:
        for item in _walk_objects(declaration.value):
            tool_name = item.get("tool_name")
            upstream = item.get("upstream_function")
            if not isinstance(tool_name, str) or not isinstance(upstream, str):
                continue
            category = item.get("tool_category")
            source = f"tool:{tool_name}"
            operation = (
                f"{category}.{upstream}"
                if isinstance(category, str) and category
                else upstream
            )
            compatibility = _is_compatibility(tool_name + " " + upstream)
            direct = upstream.startswith("/") or upstream.startswith("http")
            kind = (
                InvocationEdgeKind.COMPATIBILITY_ROUTE
                if compatibility
                else (
                    InvocationEdgeKind.DIRECT_REST
                    if direct
                    else InvocationEdgeKind.APP_BINDING
                )
            )
            edges.append(
                InvocationEdge(
                    source=source,
                    target=upstream,
                    kind=kind,
                    operation=operation,
                    transport="http" if direct else "mcp-server",
                    compatibility=compatibility,
                    bypass_candidate=direct or compatibility,
                    policy_mediated=(
                        True
                        if item.get("payload_contracts")
                        and "mediation" in json.dumps(_literal(item)).lower()
                        else None
                    ),
                    source_span=spans.make(declaration.start, declaration.end),
                    metadata={
                        "declaration": declaration.name,
                        "tool_category": category if isinstance(category, str) else "",
                    },
                )
            )
    return tuple(edges)


def _nearest_symbol(source: str, offset: int, path: str) -> str:
    prefix = source[:offset]
    matches = list(
        re.finditer(
            r"(?:function\s+|(?:async\s+)?)([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?::[^={]+)?\{",
            prefix,
        )
    )
    symbol = matches[-1].group(1) if matches else "<module>"
    return f"{path}#{symbol}"


def _argument_expression(
    source: str, open_paren: int
) -> tuple[str, int, int]:
    i = open_paren + 1
    start = i
    depth = 0
    quote = ""
    escaped = False
    while i < len(source):
        char = source[i]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            i += 1
            continue
        if char in "'\"`":
            quote = char
            i += 1
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif char == "," and depth == 0:
            break
        i += 1
    return source[start:i].strip(), start, i


def _static_string_expression(expression: str) -> str | None:
    expression = expression.strip()
    if len(expression) >= 2 and expression[0] == expression[-1] and expression[0] in "'\"":
        token = _Token("string", expression, 0, len(expression))
        return _decode_string(token)
    if (
        len(expression) >= 2
        and expression[0] == expression[-1] == "`"
        and "${" not in expression
    ):
        return expression[1:-1]
    return None


def _call_edges(
    source: str,
    *,
    path: str,
    spans: _SpanFactory,
    unresolved: list[UnresolvedContractValue],
) -> tuple[InvocationEdge, ...]:
    edges: list[InvocationEdge] = []
    patterns = (
        (r"\b(?:this\.)?jsonRpc\s*\(", "json_rpc"),
        (r"\b(?:this\.)?callTool\s*\(", "call_tool"),
        (r"\b(?:globalThis\.)?fetch\s*\(", "fetch"),
        (r"\.registerInterface\s*\(", "register"),
    )
    for pattern, call_kind in patterns:
        for match in re.finditer(pattern, source):
            open_paren = source.find("(", match.start(), match.end())
            expression, start, end = _argument_expression(source, open_paren)
            target = _static_string_expression(expression)
            unresolved_id = ""
            if target is None:
                dynamic = UnresolvedContractValue(
                    expression=expression or "<missing>",
                    field_path=f"invocation.{call_kind}.target",
                    reason_code="dynamic_invocation_target",
                    declaration=_nearest_symbol(source, match.start(), path),
                    source_span=spans.make(start, end),
                )
                unresolved.append(dynamic)
                unresolved_id = dynamic.unresolved_id
            source_symbol = _nearest_symbol(source, match.start(), path)
            if call_kind == "json_rpc":
                if target == "tools/list":
                    kind = InvocationEdgeKind.TOOLS_LIST
                elif target == "tools/call":
                    kind = InvocationEdgeKind.TOOLS_CALL
                else:
                    kind = InvocationEdgeKind.MCP_PLUS_PLUS
                transport = "jsonrpc"
                bypass = False
            elif call_kind == "call_tool":
                kind = (
                    InvocationEdgeKind.HIERARCHICAL_DISPATCH
                    if target and _is_compatibility(target)
                    else InvocationEdgeKind.TOOLS_CALL
                )
                transport = "mcp-server"
                bypass = kind is InvocationEdgeKind.HIERARCHICAL_DISPATCH
            elif call_kind == "register":
                kind = InvocationEdgeKind.REGISTRATION
                transport = "in-process"
                bypass = False
            else:
                kind = (
                    InvocationEdgeKind.COMPATIBILITY_ROUTE
                    if target and _is_compatibility(target)
                    else InvocationEdgeKind.DIRECT_FETCH
                )
                transport = "http"
                bypass = True
            edges.append(
                InvocationEdge(
                    source=source_symbol,
                    target=target,
                    kind=kind,
                    operation=target or expression,
                    transport=transport,
                    compatibility=bool(target and _is_compatibility(target)),
                    bypass_candidate=bypass,
                    policy_mediated=None,
                    unresolved_id=unresolved_id,
                    source_span=spans.make(match.start(), end),
                    metadata={"callee": call_kind},
                )
            )
    return tuple(edges)


def _function_defaults(
    source: str,
    *,
    path: str,
    source_version: str,
    schema_version: str,
    source_kind: ContractSourceKind,
    spans: _SpanFactory,
    unresolved: list[UnresolvedContractValue],
) -> tuple[ContractExpectation, ...]:
    result: list[ContractExpectation] = []
    signature = re.compile(
        r"(?:export\s+)?(?:async\s+)?(?:function\s+)?"
        r"([A-Za-z_$][\w$]*)\s*\(([^()]{0,2000})\)"
        r"\s*(?::\s*[^={\n]+)?\s*\{"
    )
    for match in signature.finditer(source):
        symbol = match.group(1)
        params = match.group(2)
        params_offset = match.start(2)
        for param_match in re.finditer(
            r"([A-Za-z_$][\w$]*)\s*(?:\??\s*:\s*[^,=]+)?\s*=\s*([^,]+)",
            params,
        ):
            parameter = param_match.group(1)
            expression = param_match.group(2).strip()
            start = params_offset + param_match.start(2)
            end = params_offset + param_match.end(2)
            tokens = _tokenize(expression)
            parser = _LiteralParser(expression, tokens, {}, 0)
            value = parser.parse() if tokens else _Dynamic(expression, 0, len(expression))
            if isinstance(value, _Dynamic):
                dynamic = UnresolvedContractValue(
                    expression=expression,
                    field_path=f"{symbol}.parameters.{parameter}.default",
                    reason_code="dynamic_default",
                    declaration=symbol,
                    source_span=spans.make(start, end),
                )
                unresolved.append(dynamic)
                continue
            subject = f"function:{path}#{symbol}:{parameter}:default"
            result.append(
                ContractExpectation(
                    subject=subject,
                    field_path="default",
                    value=_literal(value),
                    source_kind=source_kind,
                    source_version=source_version,
                    schema_version=schema_version,
                    source_span=spans.make(start, end),
                    claim_family=McpClaimFamily.ARGUMENTS_PRESERVED,
                    metadata={"symbol": symbol, "parameter": parameter},
                )
            )
    return tuple(result)


_SEMANTIC_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "default",
    "error",
    "stream",
    "policy",
    "permission",
    "mediation",
    "receipt",
    "transport",
    "endpoint",
    "protocol",
    "rpc_path",
    "rpcpath",
    "toolspath",
    "healthpath",
    "mcppath",
    "p2p",
)
_SEMANTIC_EXACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "version",
        "requires",
        "operation",
        "method",
        "tool_name",
        "tool_category",
        "upstream_function",
        "payload_contracts",
        "required_fields",
        "route",
    }
)


def _contains_dynamic(value: Any) -> bool:
    if isinstance(value, _Dynamic):
        return True
    if isinstance(value, Mapping):
        return any(_contains_dynamic(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_dynamic(item) for item in value)
    return False


def _semantic_family(field_name: str) -> McpClaimFamily:
    lowered = field_name.lower()
    if "default" in lowered:
        return McpClaimFamily.ARGUMENTS_PRESERVED
    if "error" in lowered:
        return McpClaimFamily.FAILURE_PARITY
    if "stream" in lowered or "receipt" in lowered:
        return McpClaimFamily.RESULT_ENVELOPE_PRESERVED
    if any(marker in lowered for marker in _POLICY_MARKERS) or any(
        marker in lowered for marker in ("permission", "mediation")
    ):
        return McpClaimFamily.POLICY_BEFORE_EFFECT
    if any(
        marker in lowered
        for marker in (
            "transport",
            "endpoint",
            "protocol",
            "rpc",
            "path",
            "p2p",
        )
    ):
        return McpClaimFamily.TRANSPORT_PARITY
    if field_name == "version":
        return McpClaimFamily.SNAPSHOT_FRESHNESS
    if field_name in {"tool_name", "operation", "method"}:
        return McpClaimFamily.DECLARED_TOOL_EXISTS
    return McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES


def _semantic_declaration_expectations(
    declarations: Sequence[_Declaration],
    *,
    role: SourceRole,
    path: str,
    source_version: str,
    source_kind: ContractSourceKind,
    spans: _SpanFactory,
) -> tuple[ContractExpectation, ...]:
    """Retain registry/connector/policy fields not modeled as descriptors."""

    if role not in {
        SourceRole.CAPABILITY_REGISTRY,
        SourceRole.CONNECTOR,
        SourceRole.POLICY_MEDIATOR,
        SourceRole.APP_BINDING,
        SourceRole.MANIFEST,
    }:
        return ()
    result: list[ContractExpectation] = []
    for declaration in declarations:
        package_id = _package_id(declaration.name, path, _literal(declaration.value))
        span = spans.make(declaration.start, declaration.end)

        def walk(value: Any, field_path: str) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    key_text = str(key)
                    if key_text.startswith("__"):
                        continue
                    child = f"{field_path}.{key_text}" if field_path else key_text
                    lowered = key_text.lower()
                    selected = (
                        lowered in _SEMANTIC_EXACT_FIELDS
                        or any(marker in lowered for marker in _SEMANTIC_FIELD_MARKERS)
                    )
                    if selected and not _contains_dynamic(item):
                        result.append(
                            ContractExpectation(
                                subject=(
                                    f"swissknife:{role.value}:{path}#"
                                    f"{declaration.name}:{child}"
                                ),
                                field_path=child,
                                value=_literal(item),
                                source_kind=source_kind,
                                source_version=source_version,
                                schema_version=CONTRACT_SCHEMA_VERSION,
                                source_span=span,
                                package_id=(
                                    "" if package_id == "unknown" else package_id
                                ),
                                tool_name=(
                                    str(item)
                                    if lowered == "tool_name"
                                    and isinstance(item, str)
                                    else ""
                                ),
                                claim_family=_semantic_family(lowered),
                                metadata={
                                    "declaration": declaration.name,
                                    "source_role": role.value,
                                },
                            )
                        )
                    walk(item, child)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{field_path}[{index}]")

        walk(declaration.value, "")
    return tuple(result)


def _test_expectations(
    source: str,
    *,
    path: str,
    source_version: str,
    schema_version: str,
    spans: _SpanFactory,
    unresolved: list[UnresolvedContractValue],
) -> tuple[ContractExpectation, ...]:
    result: list[ContractExpectation] = []
    pattern = re.compile(
        r"expect\s*\((?P<actual>[^()\n]{1,600})\)\s*\."
        r"(?P<matcher>toBe|toEqual|toStrictEqual)\s*\("
        r"(?P<expected>[^()\n]{1,1200})\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(source):
        actual = match.group("actual").strip()
        expected_expression = match.group("expected").strip()
        tokens = _tokenize(expected_expression)
        parser = _LiteralParser(expected_expression, tokens, {}, 0)
        expected = parser.parse() if tokens else _Dynamic(
            expected_expression, 0, len(expected_expression)
        )
        if isinstance(expected, _Dynamic):
            start = match.start("expected")
            dynamic = UnresolvedContractValue(
                expression=expected_expression,
                field_path=f"test_expectation.{actual}",
                reason_code="dynamic_test_expectation",
                declaration="expect",
                source_span=spans.make(start, match.end("expected")),
            )
            unresolved.append(dynamic)
            continue
        package_id = _package_id(actual, path)
        field_name = actual.rsplit(".", 1)[-1]
        base = actual.split(".", 1)[0]
        canonical_test_package = {
            "IPFS_ACCELERATE_INTERFACE": "ipfs_accelerate_py",
            "IPFS_DATASETS_INTERFACE": "ipfs_datasets_py",
            "IPFS_KIT_INTERFACE": "ipfs_kit_py",
        }.get(base)
        if canonical_test_package is not None and field_name in {
            "version",
            "interface_cid",
            "name",
            "namespace",
            "descriptor_id",
        }:
            subject = f"mcp-interface:{canonical_test_package}:{field_name}"
        else:
            subject = (
                f"test-expectation:{path}:{match.start()}:"
                f"{base}:{field_name}"
            )
        result.append(
            ContractExpectation(
                subject=subject,
                field_path=field_name,
                value=_literal(expected),
                source_kind=ContractSourceKind.CONFORMANCE_TEST,
                source_version=source_version,
                schema_version=schema_version,
                source_span=spans.make(match.start(), match.end()),
                package_id="" if package_id == "unknown" else package_id,
                claim_family=McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
                metadata={
                    "actual_expression": actual,
                    "matcher": match.group("matcher"),
                },
            )
        )
    return tuple(result)


def _descriptor_expectations(
    descriptor: DescriptorExpectation,
    *,
    source_kind: ContractSourceKind,
) -> tuple[ContractExpectation, ...]:
    assert descriptor.source_span is not None
    result: list[ContractExpectation] = []
    fields: tuple[tuple[str, Any, McpClaimFamily], ...] = (
        ("version", descriptor.version, McpClaimFamily.SNAPSHOT_FRESHNESS),
        ("name", descriptor.name, McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES),
        ("namespace", descriptor.namespace, McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES),
        (
            "interface_cid",
            descriptor.interface_cid,
            McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
        ),
        (
            "requires",
            list(descriptor.requires),
            McpClaimFamily.POLICY_BEFORE_EFFECT,
        ),
        (
            "streaming",
            descriptor.streaming,
            McpClaimFamily.RESULT_ENVELOPE_PRESERVED,
        ),
        (
            "transport_expectations",
            list(descriptor.transport_expectations),
            McpClaimFamily.TRANSPORT_PARITY,
        ),
    )
    for field_name, value, family in fields:
        if value is None:
            continue
        result.append(
            ContractExpectation(
                subject=f"{descriptor.subject}:{field_name}",
                field_path=field_name,
                value=value,
                source_kind=source_kind,
                source_version=descriptor.source_version,
                schema_version=descriptor.schema_version,
                source_span=descriptor.source_span,
                package_id=descriptor.package_id,
                claim_family=family,
                metadata={"declaration": descriptor.declaration},
            )
        )
    for method in descriptor.methods:
        method_subject = (
            f"mcp-tool:{descriptor.package_id}:{method.name}"
            if descriptor.subject == f"mcp-interface:{descriptor.package_id}"
            else (
                f"mcp-tool:{descriptor.package_id}:{descriptor.namespace}:"
                f"{descriptor.name}:{method.name}"
            )
        )
        result.append(
            ContractExpectation(
                subject=method_subject,
                field_path="method",
                value=method.semantic_value(),
                source_kind=source_kind,
                source_version=descriptor.source_version,
                schema_version=descriptor.schema_version,
                source_span=method.source_span or descriptor.source_span,
                package_id=descriptor.package_id,
                tool_name=method.name,
                claim_family=McpClaimFamily.DECLARED_TOOL_EXISTS,
                metadata={"descriptor": descriptor.descriptor_id},
            )
        )
    return tuple(result)


def _json_schema_record(
    source: SwissKnifeSource,
    *,
    source_version: str,
    spans: _SpanFactory,
) -> tuple[JsonSchemaExpectation, tuple[ContractExpectation, ...]]:
    try:
        payload = json.loads(source.text)
    except json.JSONDecodeError as exc:
        raise SwissKnifeContractExtractorError(
            f"{source.path}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SwissKnifeContractExtractorError(
            f"{source.path}: schema/manifest JSON must be an object"
        )
    schema_id = str(payload.get("$id") or payload.get("id") or source.path)
    schema_version = str(
        payload.get("schema_version")
        or payload.get("version")
        or payload.get("$schema")
        or CONTRACT_SCHEMA_VERSION
    )
    title = str(payload.get("title") or "")
    defaults: dict[str, Any] = {}
    error_values: set[str] = set()

    def walk(value: Any, field_path: str) -> None:
        if isinstance(value, Mapping):
            if "default" in value:
                defaults[field_path or "$"] = _json_value(value["default"])
            lowered = field_path.lower()
            enum = value.get("enum")
            if isinstance(enum, list) and (
                "error" in lowered or "status" in lowered or "outcome" in lowered
            ):
                error_values.update(str(item) for item in enum)
            for key, item in value.items():
                walk(item, f"{field_path}.{key}" if field_path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{field_path}[{index}]")

    walk(payload, "")
    span = spans.make(0, len(source.text))
    record = JsonSchemaExpectation(
        path=source.path,
        schema_id=schema_id,
        schema_version=schema_version,
        title=title,
        defaults=MappingProxyType(defaults),
        error_values=tuple(sorted(error_values)),
        source_span=span,
        source_version=source_version,
    )
    expectations: list[ContractExpectation] = [
        ContractExpectation(
            subject=f"json-schema:{schema_id}",
            field_path="schema",
            value=payload,
            source_kind=ContractSourceKind.JSON_SCHEMA,
            source_version=source_version,
            schema_version=schema_version,
            source_span=span,
            claim_family=McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES,
            metadata={"path": source.path},
        )
    ]
    for field_path, value in sorted(defaults.items()):
        expectations.append(
            ContractExpectation(
                subject=f"json-schema:{schema_id}:default:{field_path}",
                field_path=field_path,
                value=value,
                source_kind=ContractSourceKind.JSON_SCHEMA,
                source_version=source_version,
                schema_version=schema_version,
                source_span=span,
                claim_family=McpClaimFamily.ARGUMENTS_PRESERVED,
                metadata={"path": source.path, "default": True},
            )
        )
    return record, tuple(expectations)


def _expectation_source(
    expectation: ContractExpectation,
    *,
    repository_tree_id: str,
) -> ContractSourceRecord:
    authority = {
        ContractSourceKind.MCP_IDL: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.JSON_SCHEMA: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.TYPED_INTERFACE: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.POLICY_CONTRACT: SourceAuthorityClass.AUTHORITATIVE,
        ContractSourceKind.CONFORMANCE_TEST: SourceAuthorityClass.CONFORMANCE,
        ContractSourceKind.REGISTRATION: SourceAuthorityClass.REGISTRATION,
        ContractSourceKind.MANIFEST: SourceAuthorityClass.MANIFEST,
        ContractSourceKind.DOCUMENTATION: SourceAuthorityClass.NOMINATING,
        ContractSourceKind.INFERRED_PROSE: SourceAuthorityClass.NONE,
    }[expectation.source_kind]
    review_state = (
        ReviewState.REVIEWED
        if authority.may_authorize_reviewed_contract
        else ReviewState.NOMINATED
    )
    fingerprint = _fingerprint(expectation.value)
    invalidators = build_source_invalidators(
        source_version=expectation.source_version,
        schema_version=expectation.schema_version,
        source_content_id=fingerprint,
        repository_tree_id=repository_tree_id,
        subject=expectation.subject,
        claim_family=expectation.claim_family.value,
        review_state=review_state.value,
        authority_class=authority.value,
    )
    return ContractSourceRecord(
        kind=expectation.source_kind,
        authority_class=authority,
        review_state=review_state,
        source_version=expectation.source_version,
        schema_version=expectation.schema_version,
        subject=expectation.subject,
        path=expectation.source_span.path,
        content_digest="",
        payload_fingerprint=fingerprint,
        metadata={
            "field_path": expectation.field_path,
            "package_id": expectation.package_id,
            "tool_name": expectation.tool_name,
            "source_span": expectation.source_span.to_dict(),
            **dict(expectation.metadata),
        },
        invalidators=invalidators,
    )


def _catalog_from_expectations(
    expectations: Sequence[ContractExpectation],
    *,
    repository_tree_id: str,
) -> McpContractCatalog:
    base = build_default_mcp_contract_catalog()
    sources = tuple(
        _expectation_source(item, repository_tree_id=repository_tree_id)
        for item in expectations
    )
    # Exact duplicate facts from the same source collapse deterministically.
    by_id: dict[str, ContractSourceRecord] = {
        source.source_id: source for source in sources
    }
    unique_sources = tuple(by_id[key] for key in sorted(by_id))
    by_subject: dict[str, list[ContractSourceRecord]] = {}
    expectation_by_subject: dict[str, ContractExpectation] = {}
    for expectation, source in zip(expectations, sources):
        by_subject.setdefault(expectation.subject, []).append(source)
        expectation_by_subject.setdefault(expectation.subject, expectation)
    contradictions = detect_source_contradictions(unique_sources)
    contradiction_by_subject: dict[str, tuple[Any, ...]] = {}
    for subject in by_subject:
        contradiction_by_subject[subject] = tuple(
            item for item in contradictions if item.subject == subject
        )
    contracts: list[ContractRecord] = []
    for subject, subject_sources in sorted(by_subject.items()):
        unique_subject_sources = {
            item.source_id: item for item in subject_sources
        }
        selected_sources = tuple(
            unique_subject_sources[key] for key in sorted(unique_subject_sources)
        )
        expectation = expectation_by_subject[subject]
        subject_contradictions = contradiction_by_subject[subject]
        best = min(selected_sources, key=lambda item: item.authority_class.rank)
        reviewed = any(item.may_authorize_contract for item in selected_sources)
        state = (
            ReviewState.CONTRADICTED
            if subject_contradictions
            else (ReviewState.REVIEWED if reviewed else ReviewState.NOMINATED)
        )
        contracts.append(
            ContractRecord(
                claim_family=expectation.claim_family,
                subject=subject,
                source_ids=tuple(item.source_id for item in selected_sources),
                authority_class=best.authority_class,
                review_state=state,
                source_version=best.source_version,
                schema_version=best.schema_version,
                tool_name=expectation.tool_name,
                package_id=expectation.package_id,
                contradiction_ids=tuple(
                    item.contradiction_id for item in subject_contradictions
                ),
                metadata={
                    "field_path": expectation.field_path,
                    "source_count": len(selected_sources),
                },
            )
        )
    return McpContractCatalog(
        claim_families=base.claim_families,
        sources=unique_sources,
        contracts=tuple(contracts),
        contradictions=contradictions,
        catalog_version=CATALOG_VERSION,
    )


class SwissKnifeContractExtractor:
    """Deterministic, bounded SwissKnife expected-contract extractor."""

    interface = SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE
    extractor_version = SWISSKNIFE_EXTRACTOR_VERSION

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        for name, value, maximum in (
            ("max_files", max_files, HARD_MAX_FILES),
            ("max_file_bytes", max_file_bytes, HARD_MAX_FILE_BYTES),
            ("max_total_bytes", max_total_bytes, HARD_MAX_TOTAL_BYTES),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise SwissKnifeContractExtractorError(
                    f"{name} must be between 1 and {maximum}"
                )
            setattr(self, name, value)
        if self.max_file_bytes > self.max_total_bytes:
            raise SwissKnifeContractExtractorError(
                "max_file_bytes cannot exceed max_total_bytes"
            )

    def extract(
        self,
        sources: Mapping[str, str | bytes] | Iterable[SwissKnifeSource],
        *,
        repository_tree_id: str = "",
        source_version: str = "",
    ) -> SwissKnifeContractExtraction:
        """Extract normalized contracts from in-memory source bodies."""

        normalized = self._normalize_sources(sources, source_version=source_version)
        repository_tree_id = _nonempty_text(
            repository_tree_id, "repository_tree_id"
        )
        descriptors: list[DescriptorExpectation] = []
        edges: list[InvocationEdge] = []
        expectations: list[ContractExpectation] = []
        schemas: list[JsonSchemaExpectation] = []
        unresolved: list[UnresolvedContractValue] = []
        source_versions: dict[str, str] = {}
        for item in normalized:
            text = item.text
            role = item.role or _role_for_path(item.path)
            source_kind = _source_kind_for_role(role)
            version = item.source_version or source_version or _source_digest(text)
            source_versions[item.path] = version
            spans = _SpanFactory(item.path, text)
            if item.path.lower().endswith(".json"):
                schema, schema_expectations = _json_schema_record(
                    item, source_version=version, spans=spans
                )
                schemas.append(schema)
                expectations.extend(schema_expectations)
                continue
            tokens = _tokenize(text)
            declarations = _parse_declarations(text, tokens)
            file_descriptors: list[DescriptorExpectation] = []
            for declaration in declarations:
                _collect_unresolved(
                    declaration.value,
                    field_path=declaration.name,
                    declaration=declaration.name,
                    source=text,
                    spans=spans,
                    output=unresolved,
                )
                if (
                    role is SourceRole.CONTRACT_TEST
                    and not declaration.exported
                ):
                    # Local descriptors in test bodies are usually independent
                    # scenario fixtures (often deliberately different
                    # versions), not package-wide conformance declarations.
                    # Explicit assertions below still join reviewed package
                    # constants to descriptor subjects.
                    continue
                descriptor = _descriptor_from_declaration(
                    declaration,
                    role=role,
                    source_version=version,
                    path=item.path,
                    span=spans.make(declaration.start, declaration.end),
                )
                if descriptor is not None:
                    file_descriptors.append(descriptor)
            file_descriptors = list(_deduplicate_descriptors(file_descriptors))
            descriptors.extend(file_descriptors)
            for descriptor in file_descriptors:
                expectations.extend(
                    _descriptor_expectations(
                        descriptor, source_kind=source_kind
                    )
                )
            edges.extend(
                _call_edges(
                    text,
                    path=item.path,
                    spans=spans,
                    unresolved=unresolved,
                )
            )
            if role in {
                SourceRole.CAPABILITY_REGISTRY,
                SourceRole.APP_BINDING,
                SourceRole.MANIFEST,
            }:
                edges.extend(
                    _app_binding_edges(
                        declarations, path=item.path, spans=spans
                    )
                )
            expectations.extend(
                _function_defaults(
                    text,
                    path=item.path,
                    source_version=version,
                    schema_version=CONTRACT_SCHEMA_VERSION,
                    source_kind=source_kind,
                    spans=spans,
                    unresolved=unresolved,
                )
            )
            expectations.extend(
                _semantic_declaration_expectations(
                    declarations,
                    role=role,
                    path=item.path,
                    source_version=version,
                    source_kind=source_kind,
                    spans=spans,
                )
            )
            if role is SourceRole.CONTRACT_TEST:
                expectations.extend(
                    _test_expectations(
                        text,
                        path=item.path,
                        source_version=version,
                        schema_version=CONTRACT_SCHEMA_VERSION,
                        spans=spans,
                        unresolved=unresolved,
                    )
                )
        # Edges are catalog evidence too, including direct and compatibility
        # paths that later NoCompatibilityBypass checks must inspect.
        for edge in edges:
            if edge.source_span is None:
                continue
            family = (
                McpClaimFamily.NO_COMPATIBILITY_BYPASS
                if edge.bypass_candidate or edge.compatibility
                else (
                    McpClaimFamily.TRANSPORT_PARITY
                    if edge.kind is InvocationEdgeKind.TRANSPORT
                    else McpClaimFamily.INVOCATION_REACHABLE
                )
            )
            expectations.append(
                ContractExpectation(
                    subject=f"invocation-edge:{edge.edge_id}",
                    field_path="edge",
                    value={
                        key: value
                        for key, value in edge.to_dict().items()
                        if key not in {"edge_id", "source_span"}
                    },
                    source_kind=_source_kind_for_role(
                        _role_for_path(edge.source_span.path)
                    ),
                    source_version=source_versions[edge.source_span.path],
                    schema_version=CONTRACT_SCHEMA_VERSION,
                    source_span=edge.source_span,
                    claim_family=family,
                    metadata={
                        "bypass_candidate": edge.bypass_candidate,
                        "compatibility": edge.compatibility,
                    },
                )
            )
        # Deduplicate syntax matches while preserving different source spans.
        edge_by_id = {edge.edge_id: edge for edge in edges}
        unresolved_by_id = {
            item.unresolved_id: item for item in unresolved
        }
        expectation_keys: dict[tuple[Any, ...], ContractExpectation] = {}
        for expectation in expectations:
            key = (
                expectation.subject,
                expectation.field_path,
                _fingerprint(expectation.value),
                expectation.source_span.path,
                expectation.source_span.start_offset,
                expectation.source_kind.value,
            )
            expectation_keys[key] = expectation
        final_expectations = tuple(expectation_keys.values())
        catalog = _catalog_from_expectations(
            final_expectations,
            repository_tree_id=repository_tree_id,
        )
        return SwissKnifeContractExtraction(
            descriptors=_deduplicate_descriptors(descriptors),
            invocation_edges=tuple(edge_by_id.values()),
            expectations=final_expectations,
            schemas=tuple(schemas),
            unresolved_values=tuple(unresolved_by_id.values()),
            catalog=catalog,
            source_versions=source_versions,
        )

    extract_sources = extract

    def extract_repository(
        self,
        root: str | os.PathLike[str],
        *,
        include_paths: Sequence[str] | None = None,
        repository_tree_id: str = "",
        source_version: str = "",
    ) -> SwissKnifeContractExtraction:
        """Read the reviewed SwissKnife scopes below ``root`` and extract them.

        ``include_paths`` may contain explicit relative files or glob patterns.
        Symlinks and paths escaping the resolved root are rejected.
        """

        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise SwissKnifeContractExtractorError(
                f"repository root does not exist: {root_path}"
            )
        patterns = tuple(include_paths or _DEFAULT_SOURCE_GLOBS)
        selected: dict[str, Path] = {}
        for pattern in patterns:
            pattern_text = _nonempty_text(pattern, "include path", required=True)
            candidate = root_path / pattern_text
            matches = (
                [candidate]
                if candidate.is_file()
                else list(root_path.glob(pattern_text))
            )
            for match in matches:
                if not match.is_file() or match.suffix.lower() not in _SOURCE_SUFFIXES:
                    continue
                resolved = match.resolve()
                try:
                    relative = resolved.relative_to(root_path)
                except ValueError as exc:
                    raise SwissKnifeContractExtractorError(
                        f"source path escapes repository root: {match}"
                    ) from exc
                if match.is_symlink():
                    raise SwissKnifeContractExtractorError(
                        f"source symlinks are not followed: {match}"
                    )
                selected[relative.as_posix()] = resolved
        source_items = [
            SwissKnifeSource(
                path=relative,
                source=path.read_bytes(),
                source_version=source_version,
            )
            for relative, path in sorted(selected.items())
        ]
        return self.extract(
            source_items,
            repository_tree_id=repository_tree_id,
            source_version=source_version,
        )

    def _normalize_sources(
        self,
        sources: Mapping[str, str | bytes] | Iterable[SwissKnifeSource],
        *,
        source_version: str,
    ) -> tuple[SwissKnifeSource, ...]:
        if isinstance(sources, Mapping):
            items = tuple(
                SwissKnifeSource(
                    path=str(path),
                    source=body,
                    source_version=source_version,
                )
                for path, body in sources.items()
            )
        else:
            items = tuple(sources)
            if not all(isinstance(item, SwissKnifeSource) for item in items):
                raise SwissKnifeContractExtractorError(
                    "sources must be a path mapping or SwissKnifeSource iterable"
                )
        if len(items) > self.max_files:
            raise SwissKnifeContractExtractorError(
                f"file limit exceeded: {len(items)} > {self.max_files}"
            )
        seen: set[str] = set()
        total = 0
        for item in items:
            if item.path in seen:
                raise SwissKnifeContractExtractorError(
                    f"duplicate source path: {item.path}"
                )
            seen.add(item.path)
            size = len(item.text.encode("utf-8", errors="surrogatepass"))
            if size > self.max_file_bytes:
                raise SwissKnifeContractExtractorError(
                    f"{item.path}: file byte limit exceeded"
                )
            total += size
            if total > self.max_total_bytes:
                raise SwissKnifeContractExtractorError(
                    "total source byte limit exceeded"
                )
        return tuple(sorted(items, key=lambda item: item.path))


# Compatibility spelling for callers that use the repository's lower-case
# "knife" convention in symbol names.
SwissknifeContractExtractor = SwissKnifeContractExtractor
SwissknifeContractExtraction = SwissKnifeContractExtraction


def extract_swissknife_contracts(
    sources: Mapping[str, str | bytes] | Iterable[SwissKnifeSource],
    *,
    repository_tree_id: str = "",
    source_version: str = "",
) -> SwissKnifeContractExtraction:
    """Convenience entry point using the bounded default extractor."""

    return SwissKnifeContractExtractor().extract(
        sources,
        repository_tree_id=repository_tree_id,
        source_version=source_version,
    )


__all__ = [
    "SWISSKNIFE_CONTRACT_EXTRACTOR_INTERFACE",
    "SWISSKNIFE_CONTRACT_EXTRACTION_SCHEMA",
    "SWISSKNIFE_SOURCE_SPAN_SCHEMA",
    "SWISSKNIFE_UNRESOLVED_VALUE_SCHEMA",
    "SWISSKNIFE_DESCRIPTOR_SCHEMA",
    "SWISSKNIFE_INVOCATION_EDGE_SCHEMA",
    "SWISSKNIFE_EXPECTATION_SCHEMA",
    "SWISSKNIFE_JSON_SCHEMA_RECORD",
    "SWISSKNIFE_EXTRACTOR_VERSION",
    "CANONICAL_SERVER_PACKAGES",
    "SwissKnifeContractExtractorError",
    "SourceRole",
    "ResolutionState",
    "InvocationEdgeKind",
    "SourceSpan",
    "SwissKnifeSource",
    "UnresolvedContractValue",
    "MethodExpectation",
    "DescriptorExpectation",
    "InvocationEdge",
    "ContractExpectation",
    "JsonSchemaExpectation",
    "SwissKnifeContractExtraction",
    "SwissKnifeContractExtractor",
    "SwissknifeContractExtractor",
    "SwissknifeContractExtraction",
    "extract_swissknife_contracts",
]
