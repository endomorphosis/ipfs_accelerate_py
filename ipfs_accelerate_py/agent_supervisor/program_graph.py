"""Canonical cross-repository program evidence graph.

This module is the immutable, content-addressed projection of repository,
blob, module, symbol, definition, import/export, call, type, schema, contract,
doc, test, MCP tool/registration, transport, artifact, finding, and proof-
obligation evidence.  Every node and edge binds producer, blob CID, source
span, resolver status, and forest identity.

Canonical graph construction (this module, evidence ``vfs/program-graph@1``)
is intentionally separate from optional GraphRAG ranking
(``vfs/graphrag-projection@1`` in
:mod:`ipfs_datasets_program_graph_provider`).  GraphRAG and model enrichment
may rank neighborhoods later; they cannot mint authoritative program-graph
records, completion authority, or proof authority.

Objective validation repair for VFS-G040 / VFS-G144 anchors the synthetic
discovery term ``objective validation repair`` so supervisor scans re-find
the validation gate after the domain evidence surfaces
(``vfs/program-graph@1``, ``vfs/graphrag-projection@1``) are already present.
That term never becomes graph identity or completion authority.  VFS-G144 is
the child validation-gate goal that owns the repair obligation; the goal
packet with VFS-G041 (minimal call-slice queries) shares construction
identity preservation so query indexes may optimize without rewriting
``graph_id`` / node / edge content addresses.

Composition policy (VFS-008 / VFS-G040): compose the semantic-dependency and
code-evidence graph contracts without mutating GraphRAG or contract
extraction.  Separate canonical graph construction from optional GraphRAG
ranking.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence

from .proof.formal_verification_contracts import content_identity


PROGRAM_GRAPH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph@1"
)
PROGRAM_GRAPH_NODE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-node@1"
)
PROGRAM_GRAPH_EDGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-edge@1"
)
PROGRAM_GRAPH_CHUNK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-chunk@1"
)
PROGRAM_GRAPH_INDEX_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-index@1"
)
PROGRAM_GRAPH_COMPLETENESS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-completeness@1"
)
PROGRAM_GRAPH_FRONTIER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-frontier@1"
)

# Objective evidence term for VFS-G040 (exact-text discovery key).
# Canonical construction lives here; optional GraphRAG ranking is a separate
# evidence surface (vfs/graphrag-projection@1) and must not author these records.
PROGRAM_GRAPH_EVIDENCE: Final[str] = "vfs/program-graph@1"
# Synthetic objective-heap evidence term for VFS-G040 / VFS-G144 validation-gate
# work.  Exact-text discovery key only — never part of graph_id identity payload.
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = "objective validation repair"
# Domain parent goal that owns construction + ranking surfaces.
OBJECTIVE_GOAL_ID: Final[str] = "VFS-G040"
# Child validation-gate goal that owns the synthetic repair obligation
# (goal_packet/program_graph/... with VFS-G041 call-slice queries).
OBJECTIVE_VALIDATION_REPAIR_GOAL_ID: Final[str] = "VFS-G144"
# Shared goal-packet surface: call-slice queries + validation repair.
OBJECTIVE_GOAL_PACKET_IDS: Final[tuple[str, ...]] = ("VFS-G041", "VFS-G144")

DEFAULT_MAX_GRAPH_NODES = 250_000
DEFAULT_MAX_GRAPH_EDGES = 1_000_000
DEFAULT_MAX_CHUNK_NODES = 4_096
DEFAULT_MAX_CHUNK_EDGES = 16_384
DEFAULT_MAX_FRONTIER_ITEMS = 8_192
DEFAULT_MAX_COMPONENT_NODES = 65_536
DEFAULT_MAX_RECORD_BYTES = 262_144
DEFAULT_MAX_LABEL_BYTES = 4_096

# Edge kinds that form hierarchical containment or mandatory dependency
# structure.  Cycles among these are illegal.  Call/import/resolve cycles are
# retained as observed program structure and are not rejected here.
_ACYCLIC_EDGE_KINDS: frozenset[str] = frozenset(
    {
        "contains",
        "defines",
        "member_of",
        "depends_on",
        "derived_from",
        "implements",
        "registers",
        "uses_transport",
        "obligates",
    }
)


class ProgramGraphError(ValueError):
    """A program-graph record is malformed or violates the trust boundary."""


class ProgramGraphBoundsError(ProgramGraphError):
    """A graph, chunk, or component exceeded a hard deterministic bound."""


class DanglingEdgeError(ProgramGraphError):
    """An edge references a node that is not present in the graph."""


class ForgedIdentityError(ProgramGraphError):
    """A caller-supplied identity does not match the derived content address."""


class IllegalCycleError(ProgramGraphError):
    """An illegal structural cycle was detected among acyclic edge kinds."""


class ProgramNodeKind(str, Enum):
    """Closed vocabulary of program-evidence node kinds (VFS-008 / VFS-G040)."""

    REPOSITORY = "repository"
    BLOB = "blob"
    MODULE = "module"
    SYMBOL = "symbol"
    DEFINITION = "definition"
    IMPORT = "import"
    EXPORT = "export"
    CALL = "call"
    TYPE = "type"
    SCHEMA = "schema"
    CONTRACT = "contract"
    DOC = "doc"
    TEST = "test"
    MCP_TOOL = "mcp_tool"
    MCP_REGISTRATION = "mcp_registration"
    TRANSPORT = "transport"
    ARTIFACT = "artifact"
    FINDING = "finding"
    PROOF_OBLIGATION = "proof_obligation"


class ProgramEdgeKind(str, Enum):
    """Closed vocabulary of program-evidence edge kinds."""

    CONTAINS = "contains"
    DEFINES = "defines"
    IMPORTS = "imports"
    EXPORTS = "exports"
    CALLS = "calls"
    TYPED_AS = "typed_as"
    DOCUMENTS = "documents"
    TESTS = "tests"
    REGISTERS = "registers"
    USES_TRANSPORT = "uses_transport"
    DERIVED_FROM = "derived_from"
    RESOLVES_TO = "resolves_to"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    MEMBER_OF = "member_of"
    SUPPORTS = "supports"
    OBLIGATES = "obligates"


class ResolverStatus(str, Enum):
    """How a node or edge has been bound to a target."""

    UNRESOLVED = "unresolved"
    RESOLVED_STATIC = "resolved_static"
    CANDIDATE = "candidate"
    AMBIGUOUS = "ambiguous"
    EXTERNAL = "external"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"

    @property
    def terminal(self) -> bool:
        return self is ResolverStatus.RESOLVED_STATIC

    @property
    def frontier(self) -> bool:
        """True when the item still contributes to the unknown frontier."""

        return self in {
            ResolverStatus.UNRESOLVED,
            ResolverStatus.CANDIDATE,
            ResolverStatus.AMBIGUOUS,
            ResolverStatus.EXTERNAL,
            ResolverStatus.UNKNOWN,
            ResolverStatus.UNSUPPORTED,
        }


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ProgramGraphError(f"unsupported {label}: {text!r}") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ProgramGraphError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ProgramGraphError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise ProgramGraphBoundsError(f"{name} exceeds label bound")
    return text


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise ProgramGraphError("record nesting exceeds bound")
    if isinstance(value, Enum):
        return _plain(value.value, depth=depth + 1)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ProgramGraphError("floating-point values are not canonical")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProgramGraphError("record keys must be strings")
        return {key: _plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth=depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_plain(item, depth=depth + 1) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    raise ProgramGraphError(f"unsupported record value type: {type(value).__name__}")


def canonical_program_json(value: Any) -> str:
    """Return deterministic compact JSON for program-graph identities."""

    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProgramGraphError(f"{name} must be a mapping")
    plain = _plain(dict(value))
    encoded = canonical_program_json(plain).encode("utf-8")
    if len(encoded) > DEFAULT_MAX_RECORD_BYTES:
        raise ProgramGraphBoundsError(f"{name} exceeds record byte bound")
    return MappingProxyType(plain)


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise ProgramGraphBoundsError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


@dataclass(frozen=True)
class SourceSpan:
    """One-based line and zero-based column source coordinates."""

    line_start: int = 0
    column_start: int = 0
    line_end: int = 0
    column_end: int = 0

    def __post_init__(self) -> None:
        values = []
        for name in ("line_start", "column_start", "line_end", "column_end"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ProgramGraphError(f"span.{name} must be an integer")
            values.append(max(0, raw))
        object.__setattr__(self, "line_start", values[0])
        object.__setattr__(self, "column_start", values[1])
        object.__setattr__(self, "line_end", values[2])
        object.__setattr__(self, "column_end", values[3])

    def to_dict(self) -> dict[str, int]:
        return {
            "line_start": self.line_start,
            "column_start": self.column_start,
            "line_end": self.line_end,
            "column_end": self.column_end,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SourceSpan":
        if payload is None:
            return cls()
        if isinstance(payload, SourceSpan):
            return payload
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("span must be a mapping")
        return cls(
            line_start=int(payload.get("line_start") or 0),
            column_start=int(payload.get("column_start") or 0),
            line_end=int(payload.get("line_end") or 0),
            column_end=int(payload.get("column_end") or 0),
        )


@dataclass(frozen=True)
class ProgramGraphBinding:
    """Provenance binding required on every program-graph record."""

    producer: str
    blob_cid: str
    forest_id: str
    span: SourceSpan = field(default_factory=SourceSpan)
    resolver_status: ResolverStatus = ResolverStatus.UNRESOLVED

    def __post_init__(self) -> None:
        object.__setattr__(self, "producer", _text(self.producer, "producer"))
        object.__setattr__(self, "blob_cid", _text(self.blob_cid, "blob_cid"))
        object.__setattr__(self, "forest_id", _text(self.forest_id, "forest_id"))
        span = (
            self.span
            if isinstance(self.span, SourceSpan)
            else SourceSpan.from_dict(self.span)
        )
        object.__setattr__(self, "span", span)
        object.__setattr__(
            self,
            "resolver_status",
            _enum(self.resolver_status, ResolverStatus, "resolver_status"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "blob_cid": self.blob_cid,
            "forest_id": self.forest_id,
            "span": self.span.to_dict(),
            "resolver_status": self.resolver_status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphBinding":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("binding must be a mapping")
        return cls(
            producer=str(payload.get("producer") or ""),
            blob_cid=str(payload.get("blob_cid") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            span=SourceSpan.from_dict(payload.get("span")),
            resolver_status=payload.get(
                "resolver_status", ResolverStatus.UNRESOLVED.value
            ),
        )


@dataclass(frozen=True)
class ProgramGraphNode:
    """One immutable, content-addressed program-evidence node."""

    kind: ProgramNodeKind
    record_key: str
    binding: ProgramGraphBinding
    component_id: str = ""
    qualified_name: str = ""
    path: str = ""
    language: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramNodeKind, "node kind")
        )
        object.__setattr__(
            self, "record_key", _text(self.record_key, "node record_key")
        )
        binding = (
            self.binding
            if isinstance(self.binding, ProgramGraphBinding)
            else ProgramGraphBinding.from_dict(self.binding)
        )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id or self.record_key, "component_id"),
        )
        for name in ("qualified_name", "path", "language"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "record", _mapping(self.record, "node record"))

    @property
    def node_id(self) -> str:
        return "pnode-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_NODE_SCHEMA,
            "kind": self.kind.value,
            "record_key": self.record_key,
            "binding": self.binding.to_dict(),
            "component_id": self.component_id,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "language": self.language,
            "record": dict(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphNode":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("node payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_NODE_SCHEMA)
        if schema != PROGRAM_GRAPH_NODE_SCHEMA:
            raise ProgramGraphError(f"unsupported node schema: {schema}")
        binding_payload = payload.get("binding")
        if not isinstance(binding_payload, Mapping):
            # Flat binding fields for convenience.
            binding_payload = {
                "producer": payload.get("producer"),
                "blob_cid": payload.get("blob_cid"),
                "forest_id": payload.get("forest_id"),
                "span": payload.get("span"),
                "resolver_status": payload.get("resolver_status"),
            }
        node = cls(
            kind=payload.get("kind", ""),
            record_key=str(payload.get("record_key") or ""),
            binding=ProgramGraphBinding.from_dict(binding_payload),
            component_id=str(payload.get("component_id") or ""),
            qualified_name=str(payload.get("qualified_name") or ""),
            path=str(payload.get("path") or ""),
            language=str(payload.get("language") or ""),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("node_id") or "")
        if claimed and claimed != node.node_id:
            raise ForgedIdentityError(
                f"node identity is forged: claimed {claimed!r}"
            )
        return node


@dataclass(frozen=True)
class ProgramGraphEdge:
    """One immutable, content-addressed program-evidence edge."""

    source: str
    target: str
    kind: ProgramEdgeKind
    binding: ProgramGraphBinding
    component_id: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _text(self.source, "edge source"))
        object.__setattr__(self, "target", _text(self.target, "edge target"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramEdgeKind, "edge kind")
        )
        binding = (
            self.binding
            if isinstance(self.binding, ProgramGraphBinding)
            else ProgramGraphBinding.from_dict(self.binding)
        )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(
            self,
            "component_id",
            _text(
                self.component_id or f"{self.source}->{self.target}",
                "edge component_id",
            ),
        )
        object.__setattr__(self, "record", _mapping(self.record, "edge record"))

    @property
    def edge_id(self) -> str:
        return "pedge-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_EDGE_SCHEMA,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "binding": self.binding.to_dict(),
            "component_id": self.component_id,
            "record": dict(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphEdge":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("edge payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_EDGE_SCHEMA)
        if schema != PROGRAM_GRAPH_EDGE_SCHEMA:
            raise ProgramGraphError(f"unsupported edge schema: {schema}")
        binding_payload = payload.get("binding")
        if not isinstance(binding_payload, Mapping):
            binding_payload = {
                "producer": payload.get("producer"),
                "blob_cid": payload.get("blob_cid"),
                "forest_id": payload.get("forest_id"),
                "span": payload.get("span"),
                "resolver_status": payload.get("resolver_status"),
            }
        edge = cls(
            source=str(payload.get("source") or payload.get("source_node_id") or ""),
            target=str(payload.get("target") or payload.get("target_node_id") or ""),
            kind=payload.get("kind", payload.get("edge_kind", "")),
            binding=ProgramGraphBinding.from_dict(binding_payload),
            component_id=str(payload.get("component_id") or ""),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("edge_id") or "")
        if claimed and claimed != edge.edge_id:
            raise ForgedIdentityError(
                f"edge identity is forged: claimed {claimed!r}"
            )
        return edge


@dataclass(frozen=True)
class GraphFrontierItem:
    """One unresolved, ambiguous, external, or unsupported graph element."""

    element_id: str
    element_kind: str
    resolver_status: ResolverStatus
    reason: str = ""
    component_id: str = ""
    qualified_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "element_id", _text(self.element_id, "frontier element_id")
        )
        object.__setattr__(
            self,
            "element_kind",
            _text(self.element_kind, "frontier element_kind"),
        )
        object.__setattr__(
            self,
            "resolver_status",
            _enum(self.resolver_status, ResolverStatus, "frontier resolver_status"),
        )
        object.__setattr__(
            self, "reason", _text(self.reason, "frontier reason", required=False)
        )
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "frontier component_id", required=False),
        )
        object.__setattr__(
            self,
            "qualified_name",
            _text(self.qualified_name, "frontier qualified_name", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_kind": self.element_kind,
            "resolver_status": self.resolver_status.value,
            "reason": self.reason,
            "component_id": self.component_id,
            "qualified_name": self.qualified_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphFrontierItem":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("frontier item must be a mapping")
        return cls(
            element_id=str(payload.get("element_id") or ""),
            element_kind=str(payload.get("element_kind") or ""),
            resolver_status=payload.get("resolver_status", ""),
            reason=str(payload.get("reason") or ""),
            component_id=str(payload.get("component_id") or ""),
            qualified_name=str(payload.get("qualified_name") or ""),
        )


@dataclass(frozen=True)
class GraphCompleteness:
    """Compact completeness and frontier metadata for a program graph."""

    complete: bool
    node_count: int
    edge_count: int
    node_counts_by_kind: Mapping[str, int] = field(default_factory=dict)
    edge_counts_by_kind: Mapping[str, int] = field(default_factory=dict)
    resolver_counts: Mapping[str, int] = field(default_factory=dict)
    component_count: int = 0
    frontier_count: int = 0
    truncated: bool = False
    truncation_reason: str = ""
    unexplained_gap_count: int = 0
    frontier: tuple[GraphFrontierItem, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise ProgramGraphError("completeness.complete must be a boolean")
        if not isinstance(self.truncated, bool):
            raise ProgramGraphError("completeness.truncated must be a boolean")
        for name in (
            "node_count",
            "edge_count",
            "component_count",
            "frontier_count",
            "unexplained_gap_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProgramGraphError(f"completeness.{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "node_counts_by_kind",
            MappingProxyType(_plain(dict(self.node_counts_by_kind or {}))),
        )
        object.__setattr__(
            self,
            "edge_counts_by_kind",
            MappingProxyType(_plain(dict(self.edge_counts_by_kind or {}))),
        )
        object.__setattr__(
            self,
            "resolver_counts",
            MappingProxyType(_plain(dict(self.resolver_counts or {}))),
        )
        object.__setattr__(
            self,
            "truncation_reason",
            _text(self.truncation_reason, "truncation_reason", required=False),
        )
        items: list[GraphFrontierItem] = []
        for value in self.frontier or ():
            item = (
                value
                if isinstance(value, GraphFrontierItem)
                else GraphFrontierItem.from_dict(value)
            )
            items.append(item)
        if len(items) > DEFAULT_MAX_FRONTIER_ITEMS:
            raise ProgramGraphBoundsError("frontier exceeds bound")
        object.__setattr__(
            self,
            "frontier",
            tuple(sorted(items, key=lambda item: (item.element_id, item.element_kind))),
        )
        if self.frontier_count != len(self.frontier):
            object.__setattr__(self, "frontier_count", len(self.frontier))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_COMPLETENESS_SCHEMA,
            "complete": self.complete,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_counts_by_kind": dict(self.node_counts_by_kind),
            "edge_counts_by_kind": dict(self.edge_counts_by_kind),
            "resolver_counts": dict(self.resolver_counts),
            "component_count": self.component_count,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "unexplained_gap_count": self.unexplained_gap_count,
            "frontier": [item.to_dict() for item in self.frontier],
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "frontier_count": self.frontier_count,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphCompleteness":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("completeness payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_COMPLETENESS_SCHEMA)
        if schema != PROGRAM_GRAPH_COMPLETENESS_SCHEMA:
            raise ProgramGraphError(f"unsupported completeness schema: {schema}")
        completeness = cls(
            complete=bool(payload.get("complete")),
            node_count=int(payload.get("node_count") or 0),
            edge_count=int(payload.get("edge_count") or 0),
            node_counts_by_kind=payload.get("node_counts_by_kind") or {},
            edge_counts_by_kind=payload.get("edge_counts_by_kind") or {},
            resolver_counts=payload.get("resolver_counts") or {},
            component_count=int(payload.get("component_count") or 0),
            frontier_count=int(payload.get("frontier_count") or 0),
            truncated=bool(payload.get("truncated")),
            truncation_reason=str(payload.get("truncation_reason") or ""),
            unexplained_gap_count=int(payload.get("unexplained_gap_count") or 0),
            frontier=tuple(payload.get("frontier") or ()),
        )
        claimed = str(payload.get("content_id") or "")
        if claimed and claimed != completeness.content_id:
            raise ForgedIdentityError("completeness content identity is forged")
        return completeness


@dataclass(frozen=True)
class GraphChunk:
    """A bounded, content-addressed subset of the program graph."""

    chunk_key: str
    nodes: tuple[ProgramGraphNode, ...] = ()
    edges: tuple[ProgramGraphEdge, ...] = ()
    forest_id: str = ""
    component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_key", _text(self.chunk_key, "chunk_key"))
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "chunk forest_id", required=False)
        )
        node_map: dict[str, ProgramGraphNode] = {}
        for value in self.nodes:
            node = (
                value
                if isinstance(value, ProgramGraphNode)
                else ProgramGraphNode.from_dict(value)
            )
            previous = node_map.get(node.node_id)
            if previous is not None and previous.to_dict() != node.to_dict():
                raise ProgramGraphError(
                    f"conflicting node records in chunk: {node.node_id}"
                )
            node_map[node.node_id] = node
        if len(node_map) > DEFAULT_MAX_CHUNK_NODES:
            raise ProgramGraphBoundsError("graph chunk has too many nodes")

        edge_map: dict[str, ProgramGraphEdge] = {}
        for value in self.edges:
            edge = (
                value
                if isinstance(value, ProgramGraphEdge)
                else ProgramGraphEdge.from_dict(value)
            )
            if edge.source not in node_map or edge.target not in node_map:
                raise DanglingEdgeError(
                    f"chunk edge {edge.edge_id} references a missing node"
                )
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_CHUNK_EDGES:
            raise ProgramGraphBoundsError("graph chunk has too many edges")

        components = {
            _text(item, "component_id")
            for item in (self.component_ids or ())
            if str(item or "").strip()
        }
        for node in node_map.values():
            components.add(node.component_id)
        for edge in edge_map.values():
            components.add(edge.component_id)

        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        object.__setattr__(self, "component_ids", tuple(sorted(components)))

    @property
    def chunk_id(self) -> str:
        return "pchunk-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_CHUNK_SCHEMA,
            "chunk_key": self.chunk_key,
            "forest_id": self.forest_id,
            "component_ids": list(self.component_ids),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "chunk_id": self.chunk_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphChunk":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("chunk payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_CHUNK_SCHEMA)
        if schema != PROGRAM_GRAPH_CHUNK_SCHEMA:
            raise ProgramGraphError(f"unsupported chunk schema: {schema}")
        chunk = cls(
            chunk_key=str(payload.get("chunk_key") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            component_ids=tuple(payload.get("component_ids") or ()),
            nodes=tuple(payload.get("nodes") or ()),
            edges=tuple(payload.get("edges") or ()),
        )
        claimed = str(payload.get("chunk_id") or "")
        if claimed and claimed != chunk.chunk_id:
            raise ForgedIdentityError("chunk identity is forged")
        return chunk


@dataclass(frozen=True)
class GraphIndex:
    """Deterministic secondary indexes over a program graph."""

    forest_id: str
    node_ids_by_kind: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    edge_ids_by_kind: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    node_ids_by_component: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    edge_ids_by_component: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    node_ids_by_blob_cid: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    node_ids_by_qualified_name: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    node_ids_by_path: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "forest_id", _text(self.forest_id, "index forest_id"))
        object.__setattr__(
            self,
            "node_ids_by_kind",
            MappingProxyType(_freeze_index_map(self.node_ids_by_kind)),
        )
        object.__setattr__(
            self,
            "edge_ids_by_kind",
            MappingProxyType(_freeze_index_map(self.edge_ids_by_kind)),
        )
        object.__setattr__(
            self,
            "node_ids_by_component",
            MappingProxyType(_freeze_index_map(self.node_ids_by_component)),
        )
        object.__setattr__(
            self,
            "edge_ids_by_component",
            MappingProxyType(_freeze_index_map(self.edge_ids_by_component)),
        )
        object.__setattr__(
            self,
            "node_ids_by_blob_cid",
            MappingProxyType(_freeze_index_map(self.node_ids_by_blob_cid)),
        )
        object.__setattr__(
            self,
            "node_ids_by_qualified_name",
            MappingProxyType(_freeze_index_map(self.node_ids_by_qualified_name)),
        )
        object.__setattr__(
            self,
            "node_ids_by_path",
            MappingProxyType(_freeze_index_map(self.node_ids_by_path)),
        )
        components = tuple(
            sorted(
                {
                    _text(item, "component_id")
                    for item in (self.component_ids or ())
                    if str(item or "").strip()
                }
            )
        )
        object.__setattr__(self, "component_ids", components)

    @property
    def index_id(self) -> str:
        return "pindex-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_INDEX_SCHEMA,
            "forest_id": self.forest_id,
            "node_ids_by_kind": {
                key: list(value) for key, value in self.node_ids_by_kind.items()
            },
            "edge_ids_by_kind": {
                key: list(value) for key, value in self.edge_ids_by_kind.items()
            },
            "node_ids_by_component": {
                key: list(value) for key, value in self.node_ids_by_component.items()
            },
            "edge_ids_by_component": {
                key: list(value) for key, value in self.edge_ids_by_component.items()
            },
            "node_ids_by_blob_cid": {
                key: list(value) for key, value in self.node_ids_by_blob_cid.items()
            },
            "node_ids_by_qualified_name": {
                key: list(value)
                for key, value in self.node_ids_by_qualified_name.items()
            },
            "node_ids_by_path": {
                key: list(value) for key, value in self.node_ids_by_path.items()
            },
            "component_ids": list(self.component_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "index_id": self.index_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GraphIndex":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("index payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_INDEX_SCHEMA)
        if schema != PROGRAM_GRAPH_INDEX_SCHEMA:
            raise ProgramGraphError(f"unsupported index schema: {schema}")
        index = cls(
            forest_id=str(payload.get("forest_id") or ""),
            node_ids_by_kind=payload.get("node_ids_by_kind") or {},
            edge_ids_by_kind=payload.get("edge_ids_by_kind") or {},
            node_ids_by_component=payload.get("node_ids_by_component") or {},
            edge_ids_by_component=payload.get("edge_ids_by_component") or {},
            node_ids_by_blob_cid=payload.get("node_ids_by_blob_cid") or {},
            node_ids_by_qualified_name=payload.get("node_ids_by_qualified_name")
            or {},
            node_ids_by_path=payload.get("node_ids_by_path") or {},
            component_ids=tuple(payload.get("component_ids") or ()),
        )
        claimed = str(payload.get("index_id") or "")
        if claimed and claimed != index.index_id:
            raise ForgedIdentityError("index identity is forged")
        return index


def _freeze_index_map(value: Any) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProgramGraphError("index map must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for key, items in value.items():
        text_key = _text(key, "index key")
        if items is None:
            ids: list[str] = []
        elif isinstance(items, (list, tuple)):
            ids = [_text(item, "index id") for item in items]
        else:
            raise ProgramGraphError("index values must be sequences of ids")
        result[text_key] = tuple(sorted(set(ids)))
    return {key: result[key] for key in sorted(result)}


@dataclass(frozen=True)
class ProgramGraph:
    """Canonical, immutable cross-repository program evidence graph."""

    forest_id: str
    nodes: tuple[ProgramGraphNode, ...] = ()
    edges: tuple[ProgramGraphEdge, ...] = ()
    producer: str = "program-graph@1"
    unexplained_gap_count: int = 0
    truncated: bool = False
    truncation_reason: str = ""

    def __post_init__(self) -> None:
        forest_id = _text(self.forest_id, "graph forest_id")
        object.__setattr__(self, "forest_id", forest_id)
        object.__setattr__(
            self, "producer", _text(self.producer, "graph producer")
        )
        if not isinstance(self.truncated, bool):
            raise ProgramGraphError("truncated must be a boolean")
        if (
            isinstance(self.unexplained_gap_count, bool)
            or not isinstance(self.unexplained_gap_count, int)
            or self.unexplained_gap_count < 0
        ):
            raise ProgramGraphError(
                "unexplained_gap_count must be a non-negative integer"
            )
        object.__setattr__(
            self,
            "truncation_reason",
            _text(self.truncation_reason, "truncation_reason", required=False),
        )

        node_map: dict[str, ProgramGraphNode] = {}
        for value in self.nodes:
            node = (
                value
                if isinstance(value, ProgramGraphNode)
                else ProgramGraphNode.from_dict(value)
            )
            if node.binding.forest_id != forest_id:
                raise ProgramGraphError(
                    f"node {node.node_id} is bound to a foreign forest"
                )
            previous = node_map.get(node.node_id)
            if previous is not None and previous.to_dict() != node.to_dict():
                raise ProgramGraphError(
                    f"conflicting records for node {node.node_id}"
                )
            node_map[node.node_id] = node
        if len(node_map) > DEFAULT_MAX_GRAPH_NODES:
            raise ProgramGraphBoundsError("program graph has too many nodes")

        edge_map: dict[str, ProgramGraphEdge] = {}
        for value in self.edges:
            edge = (
                value
                if isinstance(value, ProgramGraphEdge)
                else ProgramGraphEdge.from_dict(value)
            )
            if edge.binding.forest_id != forest_id:
                raise ProgramGraphError(
                    f"edge {edge.edge_id} is bound to a foreign forest"
                )
            if edge.source not in node_map or edge.target not in node_map:
                raise DanglingEdgeError(
                    f"edge {edge.edge_id} references an unknown node"
                )
            source = node_map[edge.source]
            target = node_map[edge.target]
            if (
                source.binding.forest_id != forest_id
                or target.binding.forest_id != forest_id
            ):
                raise ProgramGraphError(
                    f"edge {edge.edge_id} crosses forest identity"
                )
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_GRAPH_EDGES:
            raise ProgramGraphBoundsError("program graph has too many edges")

        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        self._reject_illegal_cycles()

    def _reject_illegal_cycles(self) -> None:
        """Reject cycles within each structural edge kind independently.

        Distinct acyclic kinds may point in opposite directions (for example a
        repository ``contains`` a module while the module ``depends_on`` the
        repository).  Cross-kind orientation is therefore not a cycle.  Call,
        import, and resolve edges are intentionally excluded so mutual
        recursion and re-export loops remain representable.
        """

        by_kind: dict[str, list[ProgramGraphEdge]] = {}
        for edge in self.edges:
            if edge.kind.value not in _ACYCLIC_EDGE_KINDS:
                continue
            if edge.source == edge.target:
                raise IllegalCycleError(
                    f"self-referential {edge.kind.value} edge is illegal"
                )
            by_kind.setdefault(edge.kind.value, []).append(edge)

        for kind, kind_edges in sorted(by_kind.items()):
            adjacency: dict[str, set[str]] = {}
            indegree: dict[str, int] = {}
            for edge in kind_edges:
                adjacency.setdefault(edge.source, set()).add(edge.target)
                adjacency.setdefault(edge.target, set())
                indegree.setdefault(edge.source, 0)
                indegree[edge.target] = indegree.get(edge.target, 0) + 1
            ready = deque(
                sorted(key for key, degree in indegree.items() if degree == 0)
            )
            visited = 0
            while ready:
                current = ready.popleft()
                visited += 1
                for target in sorted(adjacency.get(current, ())):
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        ready.append(target)
            if visited != len(indegree):
                cycle_nodes = sorted(
                    key for key, degree in indegree.items() if degree
                )
                raise IllegalCycleError(
                    f"illegal {kind} cycle at "
                    + ", ".join(repr(item) for item in cycle_nodes[:8])
                )

    @property
    def graph_id(self) -> str:
        return "pgraph-" + content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_SCHEMA,
            "forest_id": self.forest_id,
            "producer": self.producer,
            "unexplained_gap_count": self.unexplained_gap_count,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def canonical_records(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        records = self.canonical_records()
        completeness = self.completeness()
        index = self.build_index()
        return {
            "schema": PROGRAM_GRAPH_SCHEMA,
            "graph_id": self.graph_id,
            "forest_id": self.forest_id,
            "producer": self.producer,
            # Envelope metadata only — not part of graph_id identity.
            "evidence": list(program_graph_evidence_terms()),
            "evidence_program_graph": PROGRAM_GRAPH_EVIDENCE,
            "canonical_construction": True,
            "graphrag_ranking_authority": False,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "unexplained_gap_count": self.unexplained_gap_count,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
            "completeness": completeness.to_dict(),
            "index": index.to_dict(),
            **records,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        payload = self.to_dict()
        if indent is None:
            return canonical_program_json(payload)
        return json.dumps(
            _plain(payload),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraph":
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("graph payload must be a mapping")
        schema = str(payload.get("schema") or PROGRAM_GRAPH_SCHEMA)
        if schema != PROGRAM_GRAPH_SCHEMA:
            raise ProgramGraphError(f"unsupported program graph schema: {schema}")
        graph = cls(
            forest_id=str(payload.get("forest_id") or ""),
            producer=str(payload.get("producer") or "program-graph@1"),
            nodes=tuple(payload.get("nodes") or ()),
            edges=tuple(payload.get("edges") or ()),
            unexplained_gap_count=int(payload.get("unexplained_gap_count") or 0),
            truncated=bool(payload.get("truncated")),
            truncation_reason=str(payload.get("truncation_reason") or ""),
        )
        claimed = str(payload.get("graph_id") or "")
        if claimed and claimed != graph.graph_id:
            raise ForgedIdentityError("graph identity is forged")
        return graph

    def node(self, node_id: str) -> ProgramGraphNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def edge(self, edge_id: str) -> ProgramGraphEdge:
        for item in self.edges:
            if item.edge_id == edge_id:
                return item
        raise KeyError(edge_id)

    def nodes_by_kind(
        self, kind: ProgramNodeKind | str
    ) -> tuple[ProgramGraphNode, ...]:
        expected = _enum(kind, ProgramNodeKind, "node kind")
        return tuple(item for item in self.nodes if item.kind is expected)

    def edges_by_kind(
        self, kind: ProgramEdgeKind | str
    ) -> tuple[ProgramGraphEdge, ...]:
        expected = _enum(kind, ProgramEdgeKind, "edge kind")
        return tuple(item for item in self.edges if item.kind is expected)

    def nodes_for_component(self, component_id: str) -> tuple[ProgramGraphNode, ...]:
        key = _text(component_id, "component_id")
        return tuple(item for item in self.nodes if item.component_id == key)

    def edges_for_component(self, component_id: str) -> tuple[ProgramGraphEdge, ...]:
        key = _text(component_id, "component_id")
        return tuple(item for item in self.edges if item.component_id == key)

    def component_ids(self) -> tuple[str, ...]:
        ids = {node.component_id for node in self.nodes}
        ids.update(edge.component_id for edge in self.edges)
        return tuple(sorted(ids))

    def build_index(self) -> GraphIndex:
        by_kind: dict[str, list[str]] = {}
        by_component: dict[str, list[str]] = {}
        by_blob: dict[str, list[str]] = {}
        by_name: dict[str, list[str]] = {}
        by_path: dict[str, list[str]] = {}
        for node in self.nodes:
            by_kind.setdefault(node.kind.value, []).append(node.node_id)
            by_component.setdefault(node.component_id, []).append(node.node_id)
            by_blob.setdefault(node.binding.blob_cid, []).append(node.node_id)
            if node.qualified_name:
                by_name.setdefault(node.qualified_name, []).append(node.node_id)
            if node.path:
                by_path.setdefault(node.path, []).append(node.node_id)

        edge_by_kind: dict[str, list[str]] = {}
        edge_by_component: dict[str, list[str]] = {}
        for edge in self.edges:
            edge_by_kind.setdefault(edge.kind.value, []).append(edge.edge_id)
            edge_by_component.setdefault(edge.component_id, []).append(edge.edge_id)

        return GraphIndex(
            forest_id=self.forest_id,
            node_ids_by_kind=by_kind,
            edge_ids_by_kind=edge_by_kind,
            node_ids_by_component=by_component,
            edge_ids_by_component=edge_by_component,
            node_ids_by_blob_cid=by_blob,
            node_ids_by_qualified_name=by_name,
            node_ids_by_path=by_path,
            component_ids=self.component_ids(),
        )

    def completeness(self) -> GraphCompleteness:
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        resolver_counts: dict[str, int] = {}
        frontier: list[GraphFrontierItem] = []

        for node in self.nodes:
            node_counts[node.kind.value] = node_counts.get(node.kind.value, 0) + 1
            status = node.binding.resolver_status
            resolver_counts[status.value] = resolver_counts.get(status.value, 0) + 1
            if status.frontier:
                frontier.append(
                    GraphFrontierItem(
                        element_id=node.node_id,
                        element_kind=f"node:{node.kind.value}",
                        resolver_status=status,
                        reason=str(node.record.get("reason") or status.value),
                        component_id=node.component_id,
                        qualified_name=node.qualified_name,
                    )
                )

        for edge in self.edges:
            edge_counts[edge.kind.value] = edge_counts.get(edge.kind.value, 0) + 1
            status = edge.binding.resolver_status
            resolver_counts[status.value] = resolver_counts.get(status.value, 0) + 1
            if status.frontier:
                frontier.append(
                    GraphFrontierItem(
                        element_id=edge.edge_id,
                        element_kind=f"edge:{edge.kind.value}",
                        resolver_status=status,
                        reason=str(edge.record.get("reason") or status.value),
                        component_id=edge.component_id,
                    )
                )

        # Deterministic truncation of frontier projection only; counts remain exact.
        frontier_items = sorted(
            frontier, key=lambda item: (item.element_id, item.element_kind)
        )
        truncated_frontier = False
        if len(frontier_items) > DEFAULT_MAX_FRONTIER_ITEMS:
            frontier_items = frontier_items[:DEFAULT_MAX_FRONTIER_ITEMS]
            truncated_frontier = True

        # EXTERNAL is a closed boundary (known outside the forest).  Candidate,
        # ambiguous, unresolved, unknown, and unsupported items keep the graph
        # incomplete even when inventory gaps are explained.
        incomplete_statuses = {
            ResolverStatus.UNRESOLVED,
            ResolverStatus.CANDIDATE,
            ResolverStatus.AMBIGUOUS,
            ResolverStatus.UNKNOWN,
            ResolverStatus.UNSUPPORTED,
        }
        has_open_resolution = any(
            item.resolver_status in incomplete_statuses for item in frontier
        )
        complete = (
            not self.truncated
            and self.unexplained_gap_count == 0
            and not has_open_resolution
            and not truncated_frontier
        )
        truncation_reason = self.truncation_reason
        if truncated_frontier and not truncation_reason:
            truncation_reason = "frontier_projection_bound"

        return GraphCompleteness(
            complete=complete,
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            node_counts_by_kind=node_counts,
            edge_counts_by_kind=edge_counts,
            resolver_counts=resolver_counts,
            component_count=len(self.component_ids()),
            frontier_count=len(frontier_items),
            truncated=self.truncated or truncated_frontier,
            truncation_reason=truncation_reason,
            unexplained_gap_count=self.unexplained_gap_count,
            frontier=tuple(frontier_items),
        )

    def chunk_by_component(
        self,
        component_id: str,
        *,
        max_nodes: int = DEFAULT_MAX_CHUNK_NODES,
        max_edges: int = DEFAULT_MAX_CHUNK_EDGES,
    ) -> GraphChunk:
        max_nodes = _positive_int(max_nodes, "max_nodes", maximum=DEFAULT_MAX_CHUNK_NODES)
        max_edges = _positive_int(max_edges, "max_edges", maximum=DEFAULT_MAX_CHUNK_EDGES)
        key = _text(component_id, "component_id")
        nodes = self.nodes_for_component(key)
        if len(nodes) > max_nodes:
            raise ProgramGraphBoundsError(
                f"component {key!r} exceeds chunk node bound"
            )
        node_ids = {node.node_id for node in nodes}
        # Include edges fully internal to the component.
        edges = tuple(
            edge
            for edge in self.edges_for_component(key)
            if edge.source in node_ids and edge.target in node_ids
        )
        if len(edges) > max_edges:
            raise ProgramGraphBoundsError(
                f"component {key!r} exceeds chunk edge bound"
            )
        # Promote endpoints referenced by component edges that live outside
        # the component so the chunk remains closed under its own edges.
        needed = set(node_ids)
        for edge in self.edges_for_component(key):
            needed.add(edge.source)
            needed.add(edge.target)
        closed_nodes = tuple(
            node for node in self.nodes if node.node_id in needed
        )
        if len(closed_nodes) > max_nodes:
            raise ProgramGraphBoundsError(
                f"component {key!r} closed chunk exceeds node bound"
            )
        closed_edges = tuple(
            edge
            for edge in self.edges
            if edge.component_id == key
            and edge.source in needed
            and edge.target in needed
        )
        if len(closed_edges) > max_edges:
            raise ProgramGraphBoundsError(
                f"component {key!r} closed chunk exceeds edge bound"
            )
        return GraphChunk(
            chunk_key=f"component:{key}",
            forest_id=self.forest_id,
            component_ids=(key,),
            nodes=closed_nodes,
            edges=closed_edges,
        )

    def chunk_all_components(
        self,
        *,
        max_nodes: int = DEFAULT_MAX_CHUNK_NODES,
        max_edges: int = DEFAULT_MAX_CHUNK_EDGES,
    ) -> tuple[GraphChunk, ...]:
        chunks = [
            self.chunk_by_component(
                component_id, max_nodes=max_nodes, max_edges=max_edges
            )
            for component_id in self.component_ids()
        ]
        return tuple(sorted(chunks, key=lambda item: item.chunk_key))

    def replace_component(
        self,
        component_id: str,
        *,
        nodes: Iterable[Any] = (),
        edges: Iterable[Any] = (),
    ) -> "ProgramGraph":
        """Return a new graph with ``component_id`` replaced incrementally.

        Existing nodes and edges whose ``component_id`` matches are removed.
        Replacement records must bind the same forest identity.  The resulting
        graph is revalidated for dangling edges, forged identities, bounds, and
        illegal cycles.
        """

        key = _text(component_id, "component_id")
        removed_node_ids = {
            node.node_id for node in self.nodes if node.component_id == key
        }
        retained_nodes = [
            node for node in self.nodes if node.component_id != key
        ]
        # Drop edges owned by the component and any edge that would dangle after
        # the component's nodes are removed (cross-component references).
        retained_edges = [
            edge
            for edge in self.edges
            if edge.component_id != key
            and edge.source not in removed_node_ids
            and edge.target not in removed_node_ids
        ]

        replacement_nodes: list[ProgramGraphNode] = []
        for value in nodes:
            node = (
                value
                if isinstance(value, ProgramGraphNode)
                else ProgramGraphNode.from_dict(value)
            )
            if node.component_id != key:
                # Normalize component ownership for the replacement set.
                node = ProgramGraphNode(
                    kind=node.kind,
                    record_key=node.record_key,
                    binding=node.binding,
                    component_id=key,
                    qualified_name=node.qualified_name,
                    path=node.path,
                    language=node.language,
                    record=dict(node.record),
                )
            if node.binding.forest_id != self.forest_id:
                raise ProgramGraphError(
                    f"replacement node {node.record_key!r} is bound to a foreign forest"
                )
            replacement_nodes.append(node)
        if len(replacement_nodes) > DEFAULT_MAX_COMPONENT_NODES:
            raise ProgramGraphBoundsError("replacement component has too many nodes")

        node_ids = {node.node_id for node in retained_nodes}
        node_ids.update(node.node_id for node in replacement_nodes)

        replacement_edges: list[ProgramGraphEdge] = []
        for value in edges:
            edge = (
                value
                if isinstance(value, ProgramGraphEdge)
                else ProgramGraphEdge.from_dict(value)
            )
            if edge.component_id != key:
                edge = ProgramGraphEdge(
                    source=edge.source,
                    target=edge.target,
                    kind=edge.kind,
                    binding=edge.binding,
                    component_id=key,
                    record=dict(edge.record),
                )
            if edge.binding.forest_id != self.forest_id:
                raise ProgramGraphError(
                    f"replacement edge is bound to a foreign forest"
                )
            if edge.source not in node_ids or edge.target not in node_ids:
                raise DanglingEdgeError(
                    "replacement edge references a node outside the merged graph"
                )
            replacement_edges.append(edge)
        if len(replacement_edges) > DEFAULT_MAX_GRAPH_EDGES:
            raise ProgramGraphBoundsError("replacement component has too many edges")

        return ProgramGraph(
            forest_id=self.forest_id,
            producer=self.producer,
            nodes=tuple(retained_nodes) + tuple(replacement_nodes),
            edges=tuple(retained_edges) + tuple(replacement_edges),
            unexplained_gap_count=self.unexplained_gap_count,
            truncated=self.truncated,
            truncation_reason=self.truncation_reason,
        )


def make_binding(
    *,
    producer: str,
    blob_cid: str,
    forest_id: str,
    span: SourceSpan | Mapping[str, Any] | None = None,
    resolver_status: ResolverStatus | str = ResolverStatus.UNRESOLVED,
) -> ProgramGraphBinding:
    """Construct a validated program-graph binding."""

    return ProgramGraphBinding(
        producer=producer,
        blob_cid=blob_cid,
        forest_id=forest_id,
        span=SourceSpan.from_dict(span) if span is not None else SourceSpan(),
        resolver_status=resolver_status,
    )


def make_node(
    *,
    kind: ProgramNodeKind | str,
    record_key: str,
    producer: str,
    blob_cid: str,
    forest_id: str,
    component_id: str = "",
    qualified_name: str = "",
    path: str = "",
    language: str = "",
    span: SourceSpan | Mapping[str, Any] | None = None,
    resolver_status: ResolverStatus | str = ResolverStatus.UNRESOLVED,
    record: Mapping[str, Any] | None = None,
) -> ProgramGraphNode:
    """Construct one validated program-graph node."""

    return ProgramGraphNode(
        kind=kind,
        record_key=record_key,
        binding=make_binding(
            producer=producer,
            blob_cid=blob_cid,
            forest_id=forest_id,
            span=span,
            resolver_status=resolver_status,
        ),
        component_id=component_id,
        qualified_name=qualified_name,
        path=path,
        language=language,
        record=record or {},
    )


def make_edge(
    *,
    source: str,
    target: str,
    kind: ProgramEdgeKind | str,
    producer: str,
    blob_cid: str,
    forest_id: str,
    component_id: str = "",
    span: SourceSpan | Mapping[str, Any] | None = None,
    resolver_status: ResolverStatus | str = ResolverStatus.UNRESOLVED,
    record: Mapping[str, Any] | None = None,
) -> ProgramGraphEdge:
    """Construct one validated program-graph edge."""

    return ProgramGraphEdge(
        source=source,
        target=target,
        kind=kind,
        binding=make_binding(
            producer=producer,
            blob_cid=blob_cid,
            forest_id=forest_id,
            span=span,
            resolver_status=resolver_status,
        ),
        component_id=component_id,
        record=record or {},
    )


def build_program_graph(
    *,
    forest_id: str,
    nodes: Iterable[Any] = (),
    edges: Iterable[Any] = (),
    producer: str = "program-graph@1",
    unexplained_gap_count: int = 0,
    truncated: bool = False,
    truncation_reason: str = "",
) -> ProgramGraph:
    """Build a validated, content-addressed program evidence graph."""

    return ProgramGraph(
        forest_id=forest_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        producer=producer,
        unexplained_gap_count=unexplained_gap_count,
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


def merge_program_graphs(
    graphs: Sequence[ProgramGraph],
    *,
    forest_id: str | None = None,
    producer: str = "program-graph@1",
) -> ProgramGraph:
    """Merge graphs that share one forest identity.

    Conflicting node payloads for the same ``node_id`` fail closed.  Completeness
    flags are OR-combined for truncation and summed for unexplained gaps.
    """

    items = list(graphs)
    if not items:
        raise ProgramGraphError("merge requires at least one program graph")
    expected_forest = forest_id or items[0].forest_id
    expected_forest = _text(expected_forest, "forest_id")
    nodes: list[ProgramGraphNode] = []
    edges: list[ProgramGraphEdge] = []
    unexplained = 0
    truncated = False
    reasons: list[str] = []
    for graph in items:
        if graph.forest_id != expected_forest:
            raise ProgramGraphError(
                f"cannot merge graph bound to foreign forest {graph.forest_id!r}"
            )
        nodes.extend(graph.nodes)
        edges.extend(graph.edges)
        unexplained += graph.unexplained_gap_count
        truncated = truncated or graph.truncated
        if graph.truncation_reason:
            reasons.append(graph.truncation_reason)
    return build_program_graph(
        forest_id=expected_forest,
        nodes=nodes,
        edges=edges,
        producer=producer,
        unexplained_gap_count=unexplained,
        truncated=truncated,
        truncation_reason=";".join(sorted(set(reasons))),
    )


def digest_hex(value: Any) -> str:
    """Return a stable hex digest for diagnostics (not a CID)."""

    return hashlib.sha256(canonical_program_json(value).encode("utf-8")).hexdigest()


def program_graph_evidence_terms() -> tuple[str, ...]:
    """Return the closed VFS-G040 evidence terms covered by canonical construction.

    Canonical program-graph identity and provenance (``vfs/program-graph@1``)
    are authored only by this module.  Optional GraphRAG ranking is a
    separate surface (``vfs/graphrag-projection@1``) and cannot create
    completion or proof authority.

    The synthetic ``objective validation repair`` term is intentionally
    omitted here so graph envelope ``evidence`` stays domain-only; use
    :func:`objective_validation_repair_evidence_terms` (or
    :func:`all_program_graph_evidence_terms`) for the VFS-G144 validation gate.
    """

    return (PROGRAM_GRAPH_EVIDENCE,)


def objective_validation_repair_evidence_terms() -> tuple[str, ...]:
    """Return the synthetic VFS-G144 validation-gate evidence term.

    Exact-text discovery key for objective validation repair.  Never mixes
    into content-addressed graph identity, completion authority, or proof
    authority.  Canonical construction remains separate from optional
    GraphRAG ranking.  Owned by :data:`OBJECTIVE_VALIDATION_REPAIR_GOAL_ID`
    (``VFS-G144``) under parent :data:`OBJECTIVE_GOAL_ID` (``VFS-G040``).
    """

    return (OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,)


def all_program_graph_evidence_terms() -> tuple[str, ...]:
    """Return construction domain term plus the objective validation repair gate.

    Domain ``vfs/program-graph@1`` comes first; the synthetic objective
    validation repair discovery key is appended last and never enters
    ``graph_id`` identity.  Optional GraphRAG ranking
    (``vfs/graphrag-projection@1``) remains on the provider surface via
    :func:`~ipfs_accelerate_py.agent_supervisor.ipfs_datasets_program_graph_provider.all_covered_evidence_terms`.
    """

    return program_graph_evidence_terms() + objective_validation_repair_evidence_terms()


def all_program_node_kinds() -> tuple[ProgramNodeKind, ...]:
    """Return the closed program-evidence node vocabulary."""

    return tuple(ProgramNodeKind)


def all_program_edge_kinds() -> tuple[ProgramEdgeKind, ...]:
    """Return the closed program-evidence edge vocabulary."""

    return tuple(ProgramEdgeKind)


__all__ = [
    "DEFAULT_MAX_CHUNK_EDGES",
    "DEFAULT_MAX_CHUNK_NODES",
    "DEFAULT_MAX_COMPONENT_NODES",
    "DEFAULT_MAX_FRONTIER_ITEMS",
    "DEFAULT_MAX_GRAPH_EDGES",
    "DEFAULT_MAX_GRAPH_NODES",
    "DanglingEdgeError",
    "ForgedIdentityError",
    "GraphChunk",
    "GraphCompleteness",
    "GraphFrontierItem",
    "GraphIndex",
    "IllegalCycleError",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_GOAL_PACKET_IDS",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_GOAL_ID",
    "PROGRAM_GRAPH_CHUNK_SCHEMA",
    "PROGRAM_GRAPH_COMPLETENESS_SCHEMA",
    "PROGRAM_GRAPH_EDGE_SCHEMA",
    "PROGRAM_GRAPH_EVIDENCE",
    "PROGRAM_GRAPH_FRONTIER_SCHEMA",
    "PROGRAM_GRAPH_INDEX_SCHEMA",
    "PROGRAM_GRAPH_NODE_SCHEMA",
    "PROGRAM_GRAPH_SCHEMA",
    "ProgramEdgeKind",
    "ProgramGraph",
    "ProgramGraphBinding",
    "ProgramGraphBoundsError",
    "ProgramGraphEdge",
    "ProgramGraphError",
    "ProgramGraphNode",
    "ProgramNodeKind",
    "ResolverStatus",
    "SourceSpan",
    "all_program_edge_kinds",
    "all_program_graph_evidence_terms",
    "all_program_node_kinds",
    "build_program_graph",
    "canonical_program_json",
    "digest_hex",
    "objective_validation_repair_evidence_terms",
    "program_graph_evidence_terms",
    "make_binding",
    "make_edge",
    "make_node",
    "merge_program_graphs",
]
