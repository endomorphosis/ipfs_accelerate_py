"""Snapshot-bound typed program graph façades.

The program graph is a verification boundary over repository/AST indexes,
``CodeImpactIndex``, and ``SemanticDependencyGraph``.  It does not fork their
identity or trust models.  Every node and edge records exact source roots,
extractor identity, confidence, authority, and completeness.

GraphRAG, runtime, and vector witnesses may only *nominate* edges.  An edge
becomes authoritative only through an admitted extractor, reviewed manifest,
or conservative resolver result.  Reflection, plugins, FFI, remote services,
generated sources, and excluded roots remain explicit frontiers unless closed
by such evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

PROGRAM_GRAPH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph@1"
)
PROGRAM_NODE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-node@1"
)
PROGRAM_EDGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-edge@1"
)
PROGRAM_GRAPH_SNAPSHOT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-snapshot@1"
)
PROGRAM_GRAPH_ROOTS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-roots@1"
)
PROGRAM_GRAPH_VERSION = "program-graph@1"

DEFAULT_MAX_NODES = 100_000
DEFAULT_MAX_EDGES = 250_000
DEFAULT_MAX_FRONTIER = 4_096
DEFAULT_MAX_TOMBSTONES = 16_384
DEFAULT_MAX_FIELD_BYTES = 8_192


class ProgramGraphError(ValueError):
    """A program-graph record is malformed or violates its trust boundary."""


class ProgramGraphBoundsError(ProgramGraphError):
    """A graph or field exceeded a hard deterministic bound."""


class ProgramGraphIdentityError(ProgramGraphError):
    """A content identity claim does not match its canonical payload."""


class ProgramNodeKind(str, Enum):
    """Closed vocabulary of typed program-graph nodes."""

    # Structural / identity
    REPOSITORY = "repository"
    MODULE = "module"
    PACKAGE = "package"
    FILE = "file"
    BLOB = "blob"

    # Declarations
    SYMBOL = "symbol"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PARAMETER = "parameter"
    RETURN = "return"
    FIELD = "field"
    VARIABLE = "variable"
    INTERFACE = "interface"
    PROTOCOL = "protocol"
    TYPE_ALIAS = "type_alias"
    OVERLOAD = "overload"

    # Wiring / construction
    CONSTRUCTOR = "constructor"
    FACTORY = "factory"
    BUILDER = "builder"
    DI_BINDING = "di_binding"
    REGISTRY = "registry"
    CALLBACK = "callback"
    DECORATOR = "decorator"
    CONTEXT_MANAGER = "context_manager"

    # Imports / modules
    IMPORT = "import"
    EXPORT = "export"
    ALIAS = "alias"
    RE_EXPORT = "re_export"

    # Data / state / schemas
    SCHEMA = "schema"
    SERIALIZER = "serializer"
    DESERIALIZER = "deserializer"
    MIGRATION = "migration"
    MESSAGE = "message"
    DATABASE = "database"
    STATE = "state"
    EFFECT = "effect"
    RESOURCE = "resource"
    CAPABILITY = "capability"

    # Surfaces
    API_ENDPOINT = "api_endpoint"
    RPC_METHOD = "rpc_method"
    CLI_COMMAND = "cli_command"
    CONFIG_PROVIDER = "config_provider"
    FEATURE_FLAG = "feature_flag"
    IDL = "idl"

    # Tests / docs / ownership
    TEST = "test"
    MOCK = "mock"
    FIXTURE = "fixture"
    EXAMPLE = "example"
    BENCHMARK = "benchmark"
    DOCUMENTATION = "documentation"
    VALIDATION = "validation"
    OWNERSHIP = "ownership"

    # Boundaries
    BUILD_TARGET = "build_target"
    GENERATED = "generated"
    NATIVE_BOUNDARY = "native_boundary"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"
    FRONTIER = "frontier"


class ProgramEdgeKind(str, Enum):
    """Closed vocabulary of typed program-graph edges."""

    # Structural
    CONTAINS = "contains"
    DEFINES = "defines"
    DECLARES = "declares"
    OWNS = "owns"

    # Calls / dispatch
    CALLS = "calls"
    OVERRIDES = "overrides"
    IMPLEMENTS = "implements"
    OVERLOADS = "overloads"
    CONSTRUCTS = "constructs"
    FACTORY_CREATES = "factory_creates"
    BUILDER_BUILDS = "builder_builds"
    DECORATES = "decorates"
    REGISTERS = "registers"
    INJECTS = "injects"
    CALLBACK_TO = "callback_to"
    CONTEXT_MANAGES = "context_manages"

    # Imports / aliases
    IMPORTS = "imports"
    EXPORTS = "exports"
    RE_EXPORTS = "re_exports"
    ALIASES = "aliases"

    # Data / state flow
    PARAMETER_OF = "parameter_of"
    RETURNS = "returns"
    FIELD_OF = "field_of"
    DATA_FLOW = "data_flow"
    STATE_FLOW = "state_flow"
    REACHES = "reaches"
    DOMINATES = "dominates"
    PATH_CONDITION = "path_condition"
    EFFECT_OF = "effect_of"
    USES_RESOURCE = "uses_resource"
    REQUIRES_CAPABILITY = "requires_capability"

    # Schemas / surfaces
    SERIALIZES = "serializes"
    DESERIALIZES = "deserializes"
    MIGRATES = "migrates"
    SCHEMA_OF = "schema_of"
    SERVES = "serves"
    CONFIGURES = "configures"
    DOCUMENTS = "documents"

    # Tests / validation
    TESTS = "tests"
    MOCKS = "mocks"
    FIXTURES = "fixtures"
    VALIDATES = "validates"

    # Boundaries
    GENERATED_FROM = "generated_from"
    NATIVE_BOUND = "native_bound"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"  # nominated only (GraphRAG/vector/runtime)


class ProgramProvenance(str, Enum):
    """How a node/edge was observed."""

    AST = "ast"
    EXTRACTOR = "extractor"
    MANIFEST = "manifest"
    RESOLVER = "resolver"
    IMPACT_INDEX = "impact_index"
    SEMANTIC_GRAPH = "semantic_graph"
    REVIEWED = "reviewed"
    RUNTIME = "runtime"
    GRAPHRAG = "graphrag"
    VECTOR = "vector"
    HISTORY = "history"
    MODEL = "model"

    @property
    def trusted_channel(self) -> bool:
        return self not in {
            ProgramProvenance.RUNTIME,
            ProgramProvenance.GRAPHRAG,
            ProgramProvenance.VECTOR,
            ProgramProvenance.HISTORY,
            ProgramProvenance.MODEL,
        }

    @property
    def nominated_only(self) -> bool:
        return not self.trusted_channel


class ProgramTrust(str, Enum):
    TRUSTED = "trusted"
    VERIFIED = "verified"
    REVIEWED = "reviewed"
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"
    NOMINATED = "nominated"

    @property
    def accepted(self) -> bool:
        return self in {
            ProgramTrust.TRUSTED,
            ProgramTrust.VERIFIED,
            ProgramTrust.REVIEWED,
        }


class ProgramAuthority(str, Enum):
    AUTHORITATIVE = "authoritative"
    VERIFIED_INPUT = "verified_input"
    DESCRIPTIVE = "descriptive"
    NOMINATED = "nominated"
    PROPOSAL_ONLY = "proposal_only"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            ProgramAuthority.AUTHORITATIVE,
            ProgramAuthority.VERIFIED_INPUT,
            ProgramAuthority.DESCRIPTIVE,
        }


class Completeness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FRONTIER = "frontier"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


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


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise ProgramGraphError(f"invalid {name}: {value!r}") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise ProgramGraphError(f"{name} must be a string")
    if text != text.strip() or "\x00" in text:
        raise ProgramGraphError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not text:
        raise ProgramGraphError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_FIELD_BYTES:
        raise ProgramGraphBoundsError(f"{name} exceeds its byte bound")
    return text


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise ProgramGraphBoundsError("program record exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ProgramGraphError("floating values are not canonical graph data")
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise ProgramGraphBoundsError("program record mapping is invalid")
        return {key: _plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 16_384:
            raise ProgramGraphBoundsError("program record sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ProgramGraphError(
        f"unsupported program record value: {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise ProgramGraphError(f"{name} must be a mapping or typed record")
        value = to_dict()
    normalized = _plain(value)
    if not isinstance(normalized, dict):
        raise ProgramGraphError(f"{name} must normalize to a mapping")
    return MappingProxyType(normalized)


def _string_tuple(
    value: Any,
    name: str,
    *,
    limit: int = DEFAULT_MAX_FRONTIER,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray)):
        raise ProgramGraphError(f"{name} must be a sequence of strings")
    elif isinstance(value, Sequence):
        items = value
    else:
        raise ProgramGraphError(f"{name} must be a sequence of strings")
    if len(items) > limit:
        raise ProgramGraphBoundsError(f"{name} exceeds its item bound")
    result = tuple(
        sorted({_text(item, name, required=False) for item in items if str(item).strip()})
    )
    if required and not result:
        raise ProgramGraphError(f"{name} is required")
    return result


def _confidence(value: Any) -> int:
    if value is None:
        return 100
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramGraphError("confidence must be an integer in 0..100")
    if value < 0 or value > 100:
        raise ProgramGraphError("confidence must be an integer in 0..100")
    return value


@dataclass(frozen=True)
class ProgramGraphRoots:
    """Exact roots whose drift invalidates a program-graph snapshot.

    Binds forest/tree/overlay, coverage, included/excluded/generated/native
    roots, extractor/config/toolchain identities, and tombstones.
    """

    forest_id: str
    tree_id: str
    overlay_id: str = ""
    coverage_id: str = ""
    included_roots: tuple[str, ...] = ()
    excluded_roots: tuple[str, ...] = ()
    generated_roots: tuple[str, ...] = ()
    native_roots: tuple[str, ...] = ()
    extractor_id: str = PROGRAM_GRAPH_VERSION
    config_id: str = ""
    toolchain_id: str = ""
    tombstones: tuple[str, ...] = ()
    schema: str = PROGRAM_GRAPH_ROOTS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "overlay_id", _text(self.overlay_id, "overlay_id", required=False)
        )
        object.__setattr__(
            self,
            "coverage_id",
            _text(self.coverage_id, "coverage_id", required=False),
        )
        object.__setattr__(
            self,
            "included_roots",
            _string_tuple(self.included_roots, "included_roots"),
        )
        object.__setattr__(
            self,
            "excluded_roots",
            _string_tuple(self.excluded_roots, "excluded_roots"),
        )
        object.__setattr__(
            self,
            "generated_roots",
            _string_tuple(self.generated_roots, "generated_roots"),
        )
        object.__setattr__(
            self, "native_roots", _string_tuple(self.native_roots, "native_roots")
        )
        object.__setattr__(
            self,
            "extractor_id",
            _text(self.extractor_id or PROGRAM_GRAPH_VERSION, "extractor_id"),
        )
        object.__setattr__(
            self, "config_id", _text(self.config_id, "config_id", required=False)
        )
        object.__setattr__(
            self,
            "toolchain_id",
            _text(self.toolchain_id, "toolchain_id", required=False),
        )
        object.__setattr__(
            self,
            "tombstones",
            _string_tuple(
                self.tombstones, "tombstones", limit=DEFAULT_MAX_TOMBSTONES
            ),
        )
        object.__setattr__(
            self, "schema", _text(self.schema or PROGRAM_GRAPH_ROOTS_SCHEMA, "schema")
        )
        if self.schema != PROGRAM_GRAPH_ROOTS_SCHEMA:
            raise ProgramGraphError(f"unsupported program graph roots schema: {self.schema}")

    @property
    def roots_id(self) -> str:
        return _identity("program-graph-roots", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "overlay_id": self.overlay_id,
            "coverage_id": self.coverage_id,
            "included_roots": list(self.included_roots),
            "excluded_roots": list(self.excluded_roots),
            "generated_roots": list(self.generated_roots),
            "native_roots": list(self.native_roots),
            "extractor_id": self.extractor_id,
            "config_id": self.config_id,
            "toolchain_id": self.toolchain_id,
            "tombstones": list(self.tombstones),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "roots_id": self.roots_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphRoots":
        roots = cls(
            forest_id=str(payload.get("forest_id") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            overlay_id=str(payload.get("overlay_id") or ""),
            coverage_id=str(payload.get("coverage_id") or ""),
            included_roots=tuple(payload.get("included_roots") or ()),
            excluded_roots=tuple(payload.get("excluded_roots") or ()),
            generated_roots=tuple(payload.get("generated_roots") or ()),
            native_roots=tuple(payload.get("native_roots") or ()),
            extractor_id=str(payload.get("extractor_id") or PROGRAM_GRAPH_VERSION),
            config_id=str(payload.get("config_id") or ""),
            toolchain_id=str(payload.get("toolchain_id") or ""),
            tombstones=tuple(payload.get("tombstones") or ()),
            schema=str(payload.get("schema") or PROGRAM_GRAPH_ROOTS_SCHEMA),
        )
        claimed = str(payload.get("roots_id") or "")
        if claimed and claimed != roots.roots_id:
            raise ProgramGraphIdentityError(
                "program graph roots identity does not match payload"
            )
        return roots


@dataclass(frozen=True)
class ProgramNode:
    """One typed, root-bound program entity."""

    node_id: str
    kind: ProgramNodeKind
    name: str
    roots: ProgramGraphRoots
    path: str = ""
    qualified_name: str = ""
    language: str = ""
    blob_identity: str = ""
    source_sha256: str = ""
    span: Mapping[str, Any] = field(default_factory=dict)
    provenance: ProgramProvenance = ProgramProvenance.AST
    provenance_id: str = ""
    trust: ProgramTrust = ProgramTrust.TRUSTED
    authority: ProgramAuthority = ProgramAuthority.AUTHORITATIVE
    completeness: Completeness = Completeness.COMPLETE
    confidence: int = 100
    extractor_id: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PROGRAM_NODE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramNodeKind, "node kind")
        )
        object.__setattr__(self, "name", _text(self.name, "node name"))
        if not isinstance(self.roots, ProgramGraphRoots):
            if isinstance(self.roots, Mapping):
                object.__setattr__(
                    self, "roots", ProgramGraphRoots.from_dict(self.roots)
                )
            else:
                raise ProgramGraphError("node roots must be ProgramGraphRoots")
        object.__setattr__(
            self, "path", _text(self.path, "node path", required=False)
        )
        object.__setattr__(
            self,
            "qualified_name",
            _text(
                self.qualified_name or self.name,
                "node qualified_name",
                required=False,
            )
            or self.name,
        )
        object.__setattr__(
            self, "language", _text(self.language, "language", required=False)
        )
        object.__setattr__(
            self,
            "blob_identity",
            _text(self.blob_identity, "blob_identity", required=False),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _text(self.source_sha256, "source_sha256", required=False),
        )
        object.__setattr__(self, "span", _mapping(self.span, "node span"))
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ProgramProvenance, "node provenance"),
        )
        object.__setattr__(
            self,
            "provenance_id",
            _text(
                self.provenance_id or self.node_id,
                "node provenance_id",
            ),
        )
        object.__setattr__(
            self, "trust", _enum(self.trust, ProgramTrust, "node trust")
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ProgramAuthority, "node authority"),
        )
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, Completeness, "node completeness"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "extractor_id",
            _text(
                self.extractor_id or self.roots.extractor_id,
                "node extractor_id",
            ),
        )
        object.__setattr__(
            self, "attributes", _mapping(self.attributes, "node attributes")
        )
        object.__setattr__(
            self, "schema", _text(self.schema or PROGRAM_NODE_SCHEMA, "schema")
        )
        if self.schema != PROGRAM_NODE_SCHEMA:
            raise ProgramGraphError(f"unsupported program node schema: {self.schema}")
        if self.provenance.nominated_only or not self.trust.accepted:
            if self.authority.authority_bearing:
                raise ProgramGraphError(
                    "untrusted or nominated provenance cannot create authoritative nodes"
                )
            object.__setattr__(self, "authority", ProgramAuthority.NOMINATED)
            if self.trust.accepted:
                object.__setattr__(self, "trust", ProgramTrust.NOMINATED)

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.trust.accepted
            and self.authority.authority_bearing
        )

    @property
    def content_id(self) -> str:
        return _identity("program-node", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "language": self.language,
            "blob_identity": self.blob_identity,
            "source_sha256": self.source_sha256,
            "span": _plain(self.span),
            "roots_id": self.roots.roots_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "trust": self.trust.value,
            "authority": self.authority.value,
            "completeness": self.completeness.value,
            "confidence": self.confidence,
            "extractor_id": self.extractor_id,
            "attributes": _plain(self.attributes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "roots": self.roots.to_dict(),
            "content_id": self.content_id,
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramNode":
        roots_payload = payload.get("roots")
        if isinstance(roots_payload, Mapping):
            roots = ProgramGraphRoots.from_dict(roots_payload)
        else:
            raise ProgramGraphError("program node requires roots")
        node = cls(
            node_id=str(payload.get("node_id") or ""),
            kind=payload.get("kind", ""),
            name=str(payload.get("name") or ""),
            roots=roots,
            path=str(payload.get("path") or ""),
            qualified_name=str(payload.get("qualified_name") or ""),
            language=str(payload.get("language") or ""),
            blob_identity=str(payload.get("blob_identity") or ""),
            source_sha256=str(payload.get("source_sha256") or ""),
            span=payload.get("span") or {},
            provenance=payload.get("provenance", ProgramProvenance.AST),
            provenance_id=str(payload.get("provenance_id") or ""),
            trust=payload.get("trust", ProgramTrust.TRUSTED),
            authority=payload.get("authority", ProgramAuthority.AUTHORITATIVE),
            completeness=payload.get("completeness", Completeness.COMPLETE),
            confidence=payload.get("confidence", 100),
            extractor_id=str(payload.get("extractor_id") or ""),
            attributes=payload.get("attributes") or {},
            schema=str(payload.get("schema") or PROGRAM_NODE_SCHEMA),
        )
        claimed = str(payload.get("content_id") or "")
        if claimed and claimed != node.content_id:
            raise ProgramGraphIdentityError(
                "program node content identity does not match payload"
            )
        if "authoritative" in payload and bool(payload["authoritative"]) != node.authoritative:
            raise ProgramGraphIdentityError("program node authority claim is forged")
        return node


@dataclass(frozen=True)
class ProgramEdge:
    """One typed, root-bound program relationship."""

    source: str
    target: str
    kind: ProgramEdgeKind
    roots: ProgramGraphRoots
    edge_id: str = ""
    provenance: ProgramProvenance = ProgramProvenance.AST
    provenance_id: str = ""
    trust: ProgramTrust = ProgramTrust.TRUSTED
    authority: ProgramAuthority = ProgramAuthority.AUTHORITATIVE
    completeness: Completeness = Completeness.COMPLETE
    confidence: int = 100
    extractor_id: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PROGRAM_EDGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "edge source"))
        object.__setattr__(self, "target", _text(self.target, "edge target"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramEdgeKind, "edge kind")
        )
        if not isinstance(self.roots, ProgramGraphRoots):
            if isinstance(self.roots, Mapping):
                object.__setattr__(
                    self, "roots", ProgramGraphRoots.from_dict(self.roots)
                )
            else:
                raise ProgramGraphError("edge roots must be ProgramGraphRoots")
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, ProgramProvenance, "edge provenance"),
        )
        object.__setattr__(
            self,
            "provenance_id",
            _text(
                self.provenance_id
                or f"{self.source}:{self.kind.value}:{self.target}",
                "edge provenance_id",
            ),
        )
        object.__setattr__(
            self, "trust", _enum(self.trust, ProgramTrust, "edge trust")
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, ProgramAuthority, "edge authority"),
        )
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, Completeness, "edge completeness"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "extractor_id",
            _text(
                self.extractor_id or self.roots.extractor_id,
                "edge extractor_id",
            ),
        )
        object.__setattr__(
            self, "attributes", _mapping(self.attributes, "edge attributes")
        )
        object.__setattr__(
            self, "schema", _text(self.schema or PROGRAM_EDGE_SCHEMA, "schema")
        )
        if self.schema != PROGRAM_EDGE_SCHEMA:
            raise ProgramGraphError(f"unsupported program edge schema: {self.schema}")

        # GraphRAG/runtime/vector/history/model edges are nominated-only.
        if self.provenance.nominated_only or not self.trust.accepted:
            object.__setattr__(self, "authority", ProgramAuthority.NOMINATED)
            object.__setattr__(self, "trust", ProgramTrust.NOMINATED)
            if self.kind is ProgramEdgeKind.RELATED_TO:
                pass
            elif self.kind not in {
                ProgramEdgeKind.RELATED_TO,
                ProgramEdgeKind.DEPENDS_ON,
                ProgramEdgeKind.CALLS,
                ProgramEdgeKind.DATA_FLOW,
                ProgramEdgeKind.STATE_FLOW,
            }:
                # Non-authoritative nominated edges collapse to RELATED_TO
                # only when the declared kind cannot carry nomination.
                pass
        if (
            self.kind is ProgramEdgeKind.RELATED_TO
            and self.authority.authority_bearing
        ):
            raise ProgramGraphError(
                "related_to edges are nominated-only and cannot be authoritative"
            )
        if self.provenance.nominated_only and self.authority.authority_bearing:
            raise ProgramGraphError(
                "nominated provenance cannot create authoritative edges"
            )

        claimed = str(self.edge_id or "").strip()
        object.__setattr__(self, "edge_id", "")
        actual = _identity("program-edge", self._identity_payload())
        if claimed and claimed != actual:
            raise ProgramGraphIdentityError(
                "program edge identity does not match payload"
            )
        object.__setattr__(self, "edge_id", actual)

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.trust.accepted
            and self.authority.authority_bearing
            and self.kind is not ProgramEdgeKind.RELATED_TO
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "roots_id": self.roots.roots_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "trust": self.trust.value,
            "authority": self.authority.value,
            "completeness": self.completeness.value,
            "confidence": self.confidence,
            "extractor_id": self.extractor_id,
            "attributes": _plain(self.attributes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
            "roots": self.roots.to_dict(),
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramEdge":
        roots_payload = payload.get("roots")
        if isinstance(roots_payload, Mapping):
            roots = ProgramGraphRoots.from_dict(roots_payload)
        else:
            raise ProgramGraphError("program edge requires roots")
        return cls(
            source=str(payload.get("source") or ""),
            target=str(payload.get("target") or ""),
            kind=payload.get("kind", ""),
            roots=roots,
            edge_id=str(payload.get("edge_id") or ""),
            provenance=payload.get("provenance", ProgramProvenance.AST),
            provenance_id=str(payload.get("provenance_id") or ""),
            trust=payload.get("trust", ProgramTrust.TRUSTED),
            authority=payload.get("authority", ProgramAuthority.AUTHORITATIVE),
            completeness=payload.get("completeness", Completeness.COMPLETE),
            confidence=payload.get("confidence", 100),
            extractor_id=str(payload.get("extractor_id") or ""),
            attributes=payload.get("attributes") or {},
            schema=str(payload.get("schema") or PROGRAM_EDGE_SCHEMA),
        )


@dataclass(frozen=True)
class ProgramGraphSnapshot:
    """Content-addressed, root-bound program graph snapshot."""

    roots: ProgramGraphRoots
    nodes: tuple[ProgramNode, ...] = ()
    edges: tuple[ProgramEdge, ...] = ()
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    complete: bool = False
    schema: str = PROGRAM_GRAPH_SNAPSHOT_SCHEMA
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramGraphRoots):
            if isinstance(self.roots, Mapping):
                object.__setattr__(
                    self, "roots", ProgramGraphRoots.from_dict(self.roots)
                )
            else:
                raise ProgramGraphError("snapshot roots must be ProgramGraphRoots")
        nodes = tuple(self.nodes or ())
        edges = tuple(self.edges or ())
        if len(nodes) > DEFAULT_MAX_NODES:
            raise ProgramGraphBoundsError("node count exceeds hard bound")
        if len(edges) > DEFAULT_MAX_EDGES:
            raise ProgramGraphBoundsError("edge count exceeds hard bound")
        if not all(isinstance(node, ProgramNode) for node in nodes):
            raise ProgramGraphError("snapshot nodes must be ProgramNode values")
        if not all(isinstance(edge, ProgramEdge) for edge in edges):
            raise ProgramGraphError("snapshot edges must be ProgramEdge values")

        # Enforce shared root binding and deterministic ordering.
        for node in nodes:
            if node.roots.roots_id != self.roots.roots_id:
                raise ProgramGraphIdentityError(
                    f"node {node.node_id!r} is bound to a foreign roots identity"
                )
        for edge in edges:
            if edge.roots.roots_id != self.roots.roots_id:
                raise ProgramGraphIdentityError(
                    f"edge {edge.edge_id!r} is bound to a foreign roots identity"
                )

        by_id = {node.node_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise ProgramGraphError("snapshot node_ids must be unique")
        object.__setattr__(
            self,
            "nodes",
            tuple(sorted(by_id.values(), key=lambda item: item.node_id)),
        )

        edge_by_id = {edge.edge_id: edge for edge in edges}
        if len(edge_by_id) != len(edges):
            raise ProgramGraphError("snapshot edge_ids must be unique")
        for edge in edge_by_id.values():
            if edge.source not in by_id or edge.target not in by_id:
                raise ProgramGraphError(
                    f"edge {edge.edge_id!r} references missing nodes"
                )
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(edge_by_id.values(), key=lambda item: item.edge_id)),
        )
        object.__setattr__(
            self,
            "frontier_refs",
            _string_tuple(self.frontier_refs, "frontier_refs"),
        )
        object.__setattr__(
            self,
            "exclusion_refs",
            _string_tuple(self.exclusion_refs, "exclusion_refs"),
        )
        if not isinstance(self.complete, bool):
            raise ProgramGraphError("snapshot complete must be a boolean")
        # A non-empty frontier or exclusion means the graph is not complete.
        if (self.frontier_refs or self.exclusion_refs) and self.complete:
            object.__setattr__(self, "complete", False)
        object.__setattr__(
            self,
            "schema",
            _text(self.schema or PROGRAM_GRAPH_SNAPSHOT_SCHEMA, "schema"),
        )
        if self.schema != PROGRAM_GRAPH_SNAPSHOT_SCHEMA:
            raise ProgramGraphError(
                f"unsupported program graph snapshot schema: {self.schema}"
            )

        claimed = str(self.snapshot_id or "").strip()
        object.__setattr__(self, "snapshot_id", "")
        actual = _identity("program-graph-snapshot", self._identity_payload())
        if claimed and claimed != actual:
            raise ProgramGraphIdentityError(
                "program graph snapshot identity does not match payload"
            )
        object.__setattr__(self, "snapshot_id", actual)

    @property
    def graph_id(self) -> str:
        return self.snapshot_id

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "roots_id": self.roots.roots_id,
            "node_content_ids": [node.content_id for node in self.nodes],
            "edge_ids": [edge.edge_id for edge in self.edges],
            "frontier_refs": list(self.frontier_refs),
            "exclusion_refs": list(self.exclusion_refs),
            "complete": self.complete,
            "extractor_id": self.roots.extractor_id,
        }

    def node(self, node_id: str) -> ProgramNode | None:
        return next((item for item in self.nodes if item.node_id == node_id), None)

    def nodes_of_kind(self, kind: ProgramNodeKind | str) -> tuple[ProgramNode, ...]:
        kind = _enum(kind, ProgramNodeKind, "node kind")
        return tuple(item for item in self.nodes if item.kind is kind)

    def edges_of_kind(self, kind: ProgramEdgeKind | str) -> tuple[ProgramEdge, ...]:
        kind = _enum(kind, ProgramEdgeKind, "edge kind")
        return tuple(item for item in self.edges if item.kind is kind)

    def edges_from(self, node_id: str) -> tuple[ProgramEdge, ...]:
        return tuple(item for item in self.edges if item.source == node_id)

    def edges_to(self, node_id: str) -> tuple[ProgramEdge, ...]:
        return tuple(item for item in self.edges if item.target == node_id)

    def authoritative_edges(self) -> tuple[ProgramEdge, ...]:
        return tuple(item for item in self.edges if item.authoritative)

    def nominated_edges(self) -> tuple[ProgramEdge, ...]:
        return tuple(item for item in self.edges if not item.authoritative)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot_id": self.snapshot_id,
            "graph_id": self.graph_id,
            "roots": self.roots.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "frontier_refs": list(self.frontier_refs),
            "exclusion_refs": list(self.exclusion_refs),
            "complete": self.complete,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphSnapshot":
        schema = str(payload.get("schema") or PROGRAM_GRAPH_SNAPSHOT_SCHEMA)
        if schema != PROGRAM_GRAPH_SNAPSHOT_SCHEMA:
            raise ProgramGraphError(
                f"unsupported program graph snapshot schema: {schema}"
            )
        roots_payload = payload.get("roots")
        if not isinstance(roots_payload, Mapping):
            raise ProgramGraphError("snapshot requires roots")
        nodes = tuple(
            ProgramNode.from_dict(item)
            for item in (payload.get("nodes") or ())
        )
        edges = tuple(
            ProgramEdge.from_dict(item)
            for item in (payload.get("edges") or ())
        )
        return cls(
            roots=ProgramGraphRoots.from_dict(roots_payload),
            nodes=nodes,
            edges=edges,
            frontier_refs=tuple(payload.get("frontier_refs") or ()),
            exclusion_refs=tuple(payload.get("exclusion_refs") or ()),
            complete=bool(payload.get("complete", False)),
            schema=schema,
            snapshot_id=str(
                payload.get("snapshot_id") or payload.get("graph_id") or ""
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "ProgramGraphSnapshot":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))


class ProgramGraph:
    """Concrete root-bound whole-program call/dependency interface.

    Satisfies the existing capability probe (``vfs.program_graph``) and the
    narrow :class:`~analysis.broken_contract_trace.ProgramGraph` protocol via
    :meth:`trace_graph_evidence`.
    """

    def __init__(self, snapshot: ProgramGraphSnapshot) -> None:
        if not isinstance(snapshot, ProgramGraphSnapshot):
            if isinstance(snapshot, Mapping):
                snapshot = ProgramGraphSnapshot.from_dict(snapshot)
            else:
                raise ProgramGraphError("ProgramGraph requires a ProgramGraphSnapshot")
        self._snapshot = snapshot

    @property
    def snapshot(self) -> ProgramGraphSnapshot:
        return self._snapshot

    @property
    def roots(self) -> ProgramGraphRoots:
        return self._snapshot.roots

    @property
    def graph_id(self) -> str:
        return self._snapshot.graph_id

    @property
    def nodes(self) -> tuple[ProgramNode, ...]:
        return self._snapshot.nodes

    @property
    def edges(self) -> tuple[ProgramEdge, ...]:
        return self._snapshot.edges

    @property
    def frontier_refs(self) -> tuple[str, ...]:
        return self._snapshot.frontier_refs

    @property
    def exclusion_refs(self) -> tuple[str, ...]:
        return self._snapshot.exclusion_refs

    @property
    def complete(self) -> bool:
        return self._snapshot.complete

    def node(self, node_id: str) -> ProgramNode | None:
        return self._snapshot.node(node_id)

    def nodes_of_kind(self, kind: ProgramNodeKind | str) -> tuple[ProgramNode, ...]:
        return self._snapshot.nodes_of_kind(kind)

    def edges_of_kind(self, kind: ProgramEdgeKind | str) -> tuple[ProgramEdge, ...]:
        return self._snapshot.edges_of_kind(kind)

    def edges_from(self, node_id: str) -> tuple[ProgramEdge, ...]:
        return self._snapshot.edges_from(node_id)

    def edges_to(self, node_id: str) -> tuple[ProgramEdge, ...]:
        return self._snapshot.edges_to(node_id)

    def find_by_qualified_name(self, name: str) -> tuple[ProgramNode, ...]:
        target = str(name or "").strip()
        if not target:
            return ()
        return tuple(
            node
            for node in self._snapshot.nodes
            if node.qualified_name == target or node.name == target
        )

    def find_by_path(self, path: str) -> tuple[ProgramNode, ...]:
        normalized = str(path or "").strip().replace("\\", "/")
        if not normalized:
            return ()
        return tuple(
            node for node in self._snapshot.nodes if node.path == normalized
        )

    def out_neighbors(
        self,
        node_id: str,
        *,
        kinds: Iterable[ProgramEdgeKind | str] | None = None,
        authoritative_only: bool = False,
    ) -> tuple[ProgramNode, ...]:
        allowed = None
        if kinds is not None:
            allowed = {
                _enum(item, ProgramEdgeKind, "edge kind") for item in kinds
            }
        result: list[ProgramNode] = []
        for edge in self._snapshot.edges_from(node_id):
            if allowed is not None and edge.kind not in allowed:
                continue
            if authoritative_only and not edge.authoritative:
                continue
            node = self._snapshot.node(edge.target)
            if node is not None:
                result.append(node)
        return tuple(result)

    def in_neighbors(
        self,
        node_id: str,
        *,
        kinds: Iterable[ProgramEdgeKind | str] | None = None,
        authoritative_only: bool = False,
    ) -> tuple[ProgramNode, ...]:
        allowed = None
        if kinds is not None:
            allowed = {
                _enum(item, ProgramEdgeKind, "edge kind") for item in kinds
            }
        result: list[ProgramNode] = []
        for edge in self._snapshot.edges_to(node_id):
            if allowed is not None and edge.kind not in allowed:
                continue
            if authoritative_only and not edge.authoritative:
                continue
            node = self._snapshot.node(edge.source)
            if node is not None:
                result.append(node)
        return tuple(result)

    def trace_graph_evidence(self, roots: Any = None) -> Any:
        """Project snapshot coverage into the broken-trace GraphEvidence shape.

        Returns a lightweight mapping when the contract module is unavailable,
        or a real :class:`GraphEvidence` instance when it is importable.
        """

        graph_id = self.graph_id
        if roots is not None:
            claimed = getattr(roots, "graph_id", None)
            if claimed is not None and str(claimed) and str(claimed) != graph_id:
                # Root mismatch is represented as incomplete coverage rather
                # than raising; the classifier maps this to unsupported.
                try:
                    from .analysis.broken_contract_trace import GraphEvidence
                    from .analysis.contract_repair_contracts import (
                        EvidenceReference,
                    )

                    return GraphEvidence(
                        graph_id=str(claimed),
                        complete=False,
                        frontier_refs=("graph_root_mismatch",),
                        exclusion_refs=self.exclusion_refs,
                        evidence_refs=(
                            EvidenceReference(
                                "program_graph",
                                graph_id,
                                "graph_root_mismatch",
                                "ipfs_accelerate_py.agent_supervisor.program_graph",
                            ),
                        ),
                    )
                except Exception:
                    return {
                        "graph_id": str(claimed),
                        "complete": False,
                        "frontier_refs": ("graph_root_mismatch",),
                        "exclusion_refs": list(self.exclusion_refs),
                    }
        try:
            from .analysis.broken_contract_trace import GraphEvidence
            from .analysis.contract_repair_contracts import EvidenceReference

            return GraphEvidence(
                graph_id=graph_id,
                complete=self.complete and not self.frontier_refs,
                frontier_refs=self.frontier_refs,
                exclusion_refs=self.exclusion_refs,
                evidence_refs=(
                    EvidenceReference(
                        "program_graph",
                        graph_id,
                        "snapshot",
                        "ipfs_accelerate_py.agent_supervisor.program_graph",
                    ),
                ),
            )
        except Exception:
            return {
                "graph_id": graph_id,
                "complete": self.complete and not self.frontier_refs,
                "frontier_refs": list(self.frontier_refs),
                "exclusion_refs": list(self.exclusion_refs),
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_SCHEMA,
            "graph_id": self.graph_id,
            "snapshot": self._snapshot.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_snapshot(cls, snapshot: ProgramGraphSnapshot | Mapping[str, Any]) -> "ProgramGraph":
        if isinstance(snapshot, ProgramGraphSnapshot):
            return cls(snapshot)
        return cls(ProgramGraphSnapshot.from_dict(snapshot))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraph":
        snapshot_payload = payload.get("snapshot")
        if isinstance(snapshot_payload, Mapping):
            return cls(ProgramGraphSnapshot.from_dict(snapshot_payload))
        if "nodes" in payload and "roots" in payload:
            return cls(ProgramGraphSnapshot.from_dict(payload))
        raise ProgramGraphError("program graph payload requires a snapshot")

    @classmethod
    def from_json(cls, value: str | bytes) -> "ProgramGraph":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return cls.from_dict(json.loads(value))


__all__ = [
    "Completeness",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_FRONTIER",
    "DEFAULT_MAX_NODES",
    "PROGRAM_EDGE_SCHEMA",
    "PROGRAM_GRAPH_ROOTS_SCHEMA",
    "PROGRAM_GRAPH_SCHEMA",
    "PROGRAM_GRAPH_SNAPSHOT_SCHEMA",
    "PROGRAM_GRAPH_VERSION",
    "PROGRAM_NODE_SCHEMA",
    "ProgramAuthority",
    "ProgramEdge",
    "ProgramEdgeKind",
    "ProgramGraph",
    "ProgramGraphBoundsError",
    "ProgramGraphError",
    "ProgramGraphIdentityError",
    "ProgramGraphRoots",
    "ProgramGraphSnapshot",
    "ProgramNode",
    "ProgramNodeKind",
    "ProgramProvenance",
    "ProgramTrust",
]
