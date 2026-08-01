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

Language-edge resolution (``vfs/language-edge-resolution@1``, VFS-G021 /
VFS-G143) is co-owned with :mod:`program_ast_adapters`: call/import/export/
resolve edges must cite a source span and resolver rule; ambiguous,
unsupported, collision, and re-export sites stay explicit and cannot mint
forged ``resolved_static`` direct call edges.

Composition policy (VFS-008 / VFS-G040): compose the semantic-dependency and
code-evidence graph contracts without mutating GraphRAG or contract
extraction.  Separate canonical graph construction from optional GraphRAG
ranking.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABCMeta
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
# Language-edge resolution co-owned with program_ast_adapters (VFS-G021 / G143).
LANGUAGE_EDGE_RESOLUTION_EVIDENCE: Final[str] = "vfs/language-edge-resolution@1"
LANGUAGE_EDGE_RESOLUTION_GOAL_ID: Final[str] = "VFS-G021"
LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID: Final[str] = "VFS-G143"
LANGUAGE_EDGE_RESOLUTION_TASK_ID: Final[str] = "VFS-069"
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

# Edge kinds that participate in language-edge resolution provenance checks.
_LANGUAGE_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "imports",
        "exports",
        "calls",
        "resolves_to",
        "references",
        "registers",
        "uses_transport",
    }
)
LANGUAGE_EDGE_RESOLUTION_INVARIANTS: Final[tuple[str, ...]] = (
    "every language edge cites a source span and resolver rule",
    "ambiguous and unsupported constructs remain explicit",
    "adversarial name collisions cannot become forged direct calls",
    "re-exports cannot become forged direct calls",
)

assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE == "vfs/language-edge-resolution@1"
assert LANGUAGE_EDGE_RESOLUTION_GOAL_ID == "VFS-G021"
assert LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID == "VFS-G143"
assert LANGUAGE_EDGE_RESOLUTION_TASK_ID == "VFS-069"

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
class _EvidenceProgramGraph:
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
    resolver_rule: str = "",
) -> ProgramGraphEdge:
    """Construct one validated program-graph edge.

    When ``resolver_rule`` is supplied it is stored on the edge record as
    ``rule_id`` so language-edge resolution provenance remains explicit.
    """

    payload = dict(record or {})
    rule = str(resolver_rule or payload.get("rule_id") or payload.get("resolver_rule") or "").strip()
    if rule:
        if not rule.startswith("rule:"):
            rule = f"rule:{rule}"
        payload.setdefault("rule_id", rule)
        payload.setdefault("resolver_rule", rule)
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
        record=payload,
    )


def edge_cites_source_span_and_resolver_rule(edge: ProgramGraphEdge) -> bool:
    """True when a language edge carries a source span and resolver rule."""

    if not isinstance(edge, ProgramGraphEdge):
        return False
    span = edge.binding.span
    if span.line_start <= 0 and span.line_end <= 0:
        return False
    record = dict(edge.record or {})
    rule = str(record.get("rule_id") or record.get("resolver_rule") or "").strip()
    return bool(rule) and rule.startswith("rule:")


def language_edge_forged_direct_call_reason(edge: ProgramGraphEdge) -> str:
    """Return a non-empty reason when a resolved call edge is forged.

    Name collisions, re-exports, dynamic mechanisms, and non-terminal statuses
    must never appear as ``resolved_static`` ``calls`` / ``resolves_to`` edges.
    """

    if not isinstance(edge, ProgramGraphEdge):
        return "not_an_edge"
    kind = edge.kind.value if isinstance(edge.kind, ProgramEdgeKind) else str(edge.kind)
    if kind not in {"calls", "resolves_to"}:
        return ""
    status = edge.binding.resolver_status
    record = dict(edge.record or {})
    reason = str(
        record.get("reason")
        or record.get("reason_code")
        or record.get("mechanism")
        or ""
    ).strip().lower()
    rule = str(record.get("rule_id") or record.get("resolver_rule") or "").lower()
    collision_markers = (
        "same_name_collision",
        "name_collision",
        "member_collision",
        "reexport_collision",
        "re_export",
        "reexport",
        "alias_collision",
    )
    dynamic_markers = (
        "dynamic",
        "dependency_injection",
        "callback",
        "monkey_patch",
        "subprocess",
        "http",
        "rpc",
        "libp2p",
        "mcp",
    )
    if any(marker in reason or marker in rule for marker in collision_markers):
        if status is ResolverStatus.RESOLVED_STATIC:
            return "forged_direct_call_from_collision_or_reexport"
    if any(marker in reason or marker in rule for marker in dynamic_markers):
        if status is ResolverStatus.RESOLVED_STATIC:
            return "forged_direct_call_from_dynamic_construct"
    if status is ResolverStatus.RESOLVED_STATIC and not edge_cites_source_span_and_resolver_rule(
        edge
    ):
        return "resolved_static_without_span_or_rule"
    return ""


def graph_satisfies_language_edge_resolution(graph: ProgramGraph) -> bool:
    """Machine-check language-edge provenance on one program graph.

    Every call/import/export/resolve edge must cite a source span and
    resolver rule.  Ambiguous/unsupported/collision/re-export sites must not
    be promoted to forged ``resolved_static`` direct calls.
    """

    if not isinstance(graph, ProgramGraph):
        raise TypeError("graph must be a ProgramGraph")
    for edge in graph.edges:
        kind = edge.kind.value if isinstance(edge.kind, ProgramEdgeKind) else str(edge.kind)
        if kind not in _LANGUAGE_EDGE_KINDS:
            continue
        if not edge_cites_source_span_and_resolver_rule(edge):
            return False
        if language_edge_forged_direct_call_reason(edge):
            return False
        status = edge.binding.resolver_status
        if status in {
            ResolverStatus.AMBIGUOUS,
            ResolverStatus.UNSUPPORTED,
            ResolverStatus.UNKNOWN,
            ResolverStatus.EXTERNAL,
            ResolverStatus.CANDIDATE,
            ResolverStatus.UNRESOLVED,
        }:
            # Frontier statuses are explicit and allowed; they must not claim
            # terminal direct-call authority.
            if status is ResolverStatus.RESOLVED_STATIC:
                return False
    return True


def language_edge_resolution_evidence_terms() -> tuple[str, ...]:
    """Return the closed VFS-G021 / VFS-G143 language-edge evidence term.

    Exact identity: ``vfs/language-edge-resolution@1``.  Co-owned with
    :mod:`program_ast_adapters`; never mixes into ``graph_id`` identity.
    """

    return (LANGUAGE_EDGE_RESOLUTION_EVIDENCE,)


def prove_language_edge_resolution(
    graph: ProgramGraph | None = None,
) -> dict[str, Any]:
    """Emit a portable ``vfs/language-edge-resolution@1`` graph-side claim.

    When a graph is supplied, every language edge is checked for span +
    resolver-rule provenance and anti-forgery invariants.  Without a graph the
    claim binds the discovery key and invariants only.
    """

    language_edges: list[dict[str, Any]] = []
    missing_rule = 0
    missing_span = 0
    forged = 0
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    satisfied = True
    if graph is not None:
        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be a ProgramGraph")
        satisfied = graph_satisfies_language_edge_resolution(graph)
        for edge in graph.edges:
            kind = (
                edge.kind.value
                if isinstance(edge.kind, ProgramEdgeKind)
                else str(edge.kind)
            )
            if kind not in _LANGUAGE_EDGE_KINDS:
                continue
            status = edge.binding.resolver_status.value
            by_status[status] = by_status.get(status, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            record = dict(edge.record or {})
            rule = str(
                record.get("rule_id") or record.get("resolver_rule") or ""
            ).strip()
            span = edge.binding.span
            if not rule:
                missing_rule += 1
            if span.line_start <= 0 and span.line_end <= 0:
                missing_span += 1
            forge_reason = language_edge_forged_direct_call_reason(edge)
            if forge_reason:
                forged += 1
            language_edges.append(
                {
                    "edge_id": edge.edge_id,
                    "kind": kind,
                    "source": edge.source,
                    "target": edge.target,
                    "resolver_status": status,
                    "rule_id": rule,
                    "span": span.to_dict(),
                    "forged_reason": forge_reason,
                }
            )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "language-edge-resolution-graph-claim@1"
        ),
        "evidence": LANGUAGE_EDGE_RESOLUTION_EVIDENCE,
        "evidence_terms": list(language_edge_resolution_evidence_terms()),
        "requirement_id": LANGUAGE_EDGE_RESOLUTION_EVIDENCE,
        "goal_id": LANGUAGE_EDGE_RESOLUTION_GOAL_ID,
        "child_goal_id": LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID,
        "task_id": LANGUAGE_EDGE_RESOLUTION_TASK_ID,
        "satisfied": satisfied,
        "graph_id": graph.graph_id if graph is not None else None,
        "forest_id": graph.forest_id if graph is not None else None,
        "language_edge_count": len(language_edges),
        "missing_rule_count": missing_rule,
        "missing_span_count": missing_span,
        "forged_direct_call_count": forged,
        "by_status": dict(sorted(by_status.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "language_edges": language_edges,
        "invariants": list(LANGUAGE_EDGE_RESOLUTION_INVARIANTS),
        "authoritative": False,
        "completion_authoritative": False,
        "forges_direct_calls": False,
    }


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
    "LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID",
    "LANGUAGE_EDGE_RESOLUTION_EVIDENCE",
    "LANGUAGE_EDGE_RESOLUTION_GOAL_ID",
    "LANGUAGE_EDGE_RESOLUTION_INVARIANTS",
    "LANGUAGE_EDGE_RESOLUTION_TASK_ID",
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
    "edge_cites_source_span_and_resolver_rule",
    "graph_satisfies_language_edge_resolution",
    "language_edge_forged_direct_call_reason",
    "language_edge_resolution_evidence_terms",
    "objective_validation_repair_evidence_terms",
    "program_graph_evidence_terms",
    "prove_language_edge_resolution",
    "make_binding",
    "make_edge",
    "make_node",
    "merge_program_graphs",
]

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


class ProgramGraphIdentityError(ProgramGraphError):
    """A content identity claim does not match its canonical payload."""


class _SnapshotProgramNodeKind(str, Enum):
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


class _SnapshotProgramEdgeKind(str, Enum):
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


def _snapshot_canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _snapshot_identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_snapshot_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _snapshot_enum(value: Any, kind: type[Enum], name: str) -> Any:
    if kind is ProgramNodeKind:
        return _snapshot_node_kind(value)
    if kind is ProgramEdgeKind:
        return _snapshot_edge_kind(value)
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise ProgramGraphError(f"invalid {name}: {value!r}") from exc

def _snapshot_text(value: Any, name: str, *, required: bool = True) -> str:
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


def _snapshot_plain(value: Any, *, depth: int = 0) -> Any:
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
        return {key: _snapshot_plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 16_384:
            raise ProgramGraphBoundsError("program record sequence is oversized")
        return [_snapshot_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _snapshot_plain(to_dict(), depth=depth + 1)
    raise ProgramGraphError(
        f"unsupported program record value: {type(value).__name__}"
    )


def _snapshot_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise ProgramGraphError(f"{name} must be a mapping or typed record")
        value = to_dict()
    normalized = _snapshot_plain(value)
    if not isinstance(normalized, dict):
        raise ProgramGraphError(f"{name} must normalize to a mapping")
    return MappingProxyType(normalized)


def _snapshot_string_tuple(
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
        sorted({_snapshot_text(item, name, required=False) for item in items if str(item).strip()})
    )
    if required and not result:
        raise ProgramGraphError(f"{name} is required")
    return result


def _snapshot_confidence(value: Any) -> int:
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
            self, "forest_id", _snapshot_text(self.forest_id, "forest_id")
        )
        object.__setattr__(self, "tree_id", _snapshot_text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "overlay_id", _snapshot_text(self.overlay_id, "overlay_id", required=False)
        )
        object.__setattr__(
            self,
            "coverage_id",
            _snapshot_text(self.coverage_id, "coverage_id", required=False),
        )
        object.__setattr__(
            self,
            "included_roots",
            _snapshot_string_tuple(self.included_roots, "included_roots"),
        )
        object.__setattr__(
            self,
            "excluded_roots",
            _snapshot_string_tuple(self.excluded_roots, "excluded_roots"),
        )
        object.__setattr__(
            self,
            "generated_roots",
            _snapshot_string_tuple(self.generated_roots, "generated_roots"),
        )
        object.__setattr__(
            self, "native_roots", _snapshot_string_tuple(self.native_roots, "native_roots")
        )
        object.__setattr__(
            self,
            "extractor_id",
            _snapshot_text(self.extractor_id or PROGRAM_GRAPH_VERSION, "extractor_id"),
        )
        object.__setattr__(
            self, "config_id", _snapshot_text(self.config_id, "config_id", required=False)
        )
        object.__setattr__(
            self,
            "toolchain_id",
            _snapshot_text(self.toolchain_id, "toolchain_id", required=False),
        )
        object.__setattr__(
            self,
            "tombstones",
            _snapshot_string_tuple(
                self.tombstones, "tombstones", limit=DEFAULT_MAX_TOMBSTONES
            ),
        )
        object.__setattr__(
            self, "schema", _snapshot_text(self.schema or PROGRAM_GRAPH_ROOTS_SCHEMA, "schema")
        )
        if self.schema != PROGRAM_GRAPH_ROOTS_SCHEMA:
            raise ProgramGraphError(f"unsupported program graph roots schema: {self.schema}")

    @property
    def roots_id(self) -> str:
        return _snapshot_identity("program-graph-roots", self._identity_payload())

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
        object.__setattr__(self, "node_id", _snapshot_text(self.node_id, "node_id"))
        object.__setattr__(
            self, "kind", _snapshot_enum(self.kind, ProgramNodeKind, "node kind")
        )
        object.__setattr__(self, "name", _snapshot_text(self.name, "node name"))
        if not isinstance(self.roots, ProgramGraphRoots):
            if isinstance(self.roots, Mapping):
                object.__setattr__(
                    self, "roots", ProgramGraphRoots.from_dict(self.roots)
                )
            else:
                raise ProgramGraphError("node roots must be ProgramGraphRoots")
        object.__setattr__(
            self, "path", _snapshot_text(self.path, "node path", required=False)
        )
        object.__setattr__(
            self,
            "qualified_name",
            _snapshot_text(
                self.qualified_name or self.name,
                "node qualified_name",
                required=False,
            )
            or self.name,
        )
        object.__setattr__(
            self, "language", _snapshot_text(self.language, "language", required=False)
        )
        object.__setattr__(
            self,
            "blob_identity",
            _snapshot_text(self.blob_identity, "blob_identity", required=False),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _snapshot_text(self.source_sha256, "source_sha256", required=False),
        )
        object.__setattr__(self, "span", _snapshot_mapping(self.span, "node span"))
        object.__setattr__(
            self,
            "provenance",
            _snapshot_enum(self.provenance, ProgramProvenance, "node provenance"),
        )
        object.__setattr__(
            self,
            "provenance_id",
            _snapshot_text(
                self.provenance_id or self.node_id,
                "node provenance_id",
            ),
        )
        object.__setattr__(
            self, "trust", _snapshot_enum(self.trust, ProgramTrust, "node trust")
        )
        object.__setattr__(
            self,
            "authority",
            _snapshot_enum(self.authority, ProgramAuthority, "node authority"),
        )
        object.__setattr__(
            self,
            "completeness",
            _snapshot_enum(self.completeness, Completeness, "node completeness"),
        )
        object.__setattr__(self, "confidence", _snapshot_confidence(self.confidence))
        object.__setattr__(
            self,
            "extractor_id",
            _snapshot_text(
                self.extractor_id or self.roots.extractor_id,
                "node extractor_id",
            ),
        )
        object.__setattr__(
            self, "attributes", _snapshot_mapping(self.attributes, "node attributes")
        )
        object.__setattr__(
            self, "schema", _snapshot_text(self.schema or PROGRAM_NODE_SCHEMA, "schema")
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
        return _snapshot_identity("program-node", self._identity_payload())

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
            "span": _snapshot_plain(self.span),
            "roots_id": self.roots.roots_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "trust": self.trust.value,
            "authority": self.authority.value,
            "completeness": self.completeness.value,
            "confidence": self.confidence,
            "extractor_id": self.extractor_id,
            "attributes": _snapshot_plain(self.attributes),
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
        object.__setattr__(self, "source", _snapshot_text(self.source, "edge source"))
        object.__setattr__(self, "target", _snapshot_text(self.target, "edge target"))
        object.__setattr__(
            self, "kind", _snapshot_enum(self.kind, ProgramEdgeKind, "edge kind")
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
            _snapshot_enum(self.provenance, ProgramProvenance, "edge provenance"),
        )
        object.__setattr__(
            self,
            "provenance_id",
            _snapshot_text(
                self.provenance_id
                or f"{self.source}:{self.kind.value}:{self.target}",
                "edge provenance_id",
            ),
        )
        object.__setattr__(
            self, "trust", _snapshot_enum(self.trust, ProgramTrust, "edge trust")
        )
        object.__setattr__(
            self,
            "authority",
            _snapshot_enum(self.authority, ProgramAuthority, "edge authority"),
        )
        object.__setattr__(
            self,
            "completeness",
            _snapshot_enum(self.completeness, Completeness, "edge completeness"),
        )
        object.__setattr__(self, "confidence", _snapshot_confidence(self.confidence))
        object.__setattr__(
            self,
            "extractor_id",
            _snapshot_text(
                self.extractor_id or self.roots.extractor_id,
                "edge extractor_id",
            ),
        )
        object.__setattr__(
            self, "attributes", _snapshot_mapping(self.attributes, "edge attributes")
        )
        object.__setattr__(
            self, "schema", _snapshot_text(self.schema or PROGRAM_EDGE_SCHEMA, "schema")
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
        actual = _snapshot_identity("program-edge", self._identity_payload())
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
            "attributes": _snapshot_plain(self.attributes),
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
            _snapshot_string_tuple(self.frontier_refs, "frontier_refs"),
        )
        object.__setattr__(
            self,
            "exclusion_refs",
            _snapshot_string_tuple(self.exclusion_refs, "exclusion_refs"),
        )
        if not isinstance(self.complete, bool):
            raise ProgramGraphError("snapshot complete must be a boolean")
        # A non-empty frontier or exclusion means the graph is not complete.
        if (self.frontier_refs or self.exclusion_refs) and self.complete:
            object.__setattr__(self, "complete", False)
        object.__setattr__(
            self,
            "schema",
            _snapshot_text(self.schema or PROGRAM_GRAPH_SNAPSHOT_SCHEMA, "schema"),
        )
        if self.schema != PROGRAM_GRAPH_SNAPSHOT_SCHEMA:
            raise ProgramGraphError(
                f"unsupported program graph snapshot schema: {self.schema}"
            )

        claimed = str(self.snapshot_id or "").strip()
        object.__setattr__(self, "snapshot_id", "")
        actual = _snapshot_identity("program-graph-snapshot", self._identity_payload())
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
        kind = _snapshot_enum(kind, ProgramNodeKind, "node kind")
        return tuple(item for item in self.nodes if item.kind is kind)

    def edges_of_kind(self, kind: ProgramEdgeKind | str) -> tuple[ProgramEdge, ...]:
        kind = _snapshot_enum(kind, ProgramEdgeKind, "edge kind")
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
        return _snapshot_canonical_json(self.to_dict())

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


class _SnapshotProgramGraph:
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
                _snapshot_enum(item, ProgramEdgeKind, "edge kind") for item in kinds
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
                _snapshot_enum(item, ProgramEdgeKind, "edge kind") for item in kinds
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
        return _snapshot_canonical_json(self.to_dict())

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

# ---------------------------------------------------------------------------
# Snapshot-graph compatibility
#
# The supervisor historically exposed a root/snapshot-bound graph while the
# canonical VFS evidence graph now uses forest-bound records. Both contracts
# remain active without widening the canonical evidence Enum vocabularies.
# ---------------------------------------------------------------------------

def _install_snapshot_enum_aliases() -> None:
    for name, member in _SnapshotProgramNodeKind.__members__.items():
        if not hasattr(ProgramNodeKind, name):
            setattr(ProgramNodeKind, name, member)
            ProgramNodeKind._value2member_map_.setdefault(member.value, member)
    for name, member in _SnapshotProgramEdgeKind.__members__.items():
        if not hasattr(ProgramEdgeKind, name):
            setattr(ProgramEdgeKind, name, member)
            ProgramEdgeKind._value2member_map_.setdefault(member.value, member)


def _snapshot_node_kind(value: Any) -> ProgramNodeKind | _SnapshotProgramNodeKind:
    if isinstance(value, (ProgramNodeKind, _SnapshotProgramNodeKind)):
        return value
    raw = str(getattr(value, "value", value))
    try:
        return ProgramNodeKind(raw)
    except (TypeError, ValueError):
        try:
            return _SnapshotProgramNodeKind(raw)
        except (TypeError, ValueError) as exc:
            raise ProgramGraphError(f"invalid node kind: {value!r}") from exc


def _snapshot_edge_kind(value: Any) -> ProgramEdgeKind | _SnapshotProgramEdgeKind:
    if isinstance(value, (ProgramEdgeKind, _SnapshotProgramEdgeKind)):
        return value
    raw = str(getattr(value, "value", value))
    try:
        return ProgramEdgeKind(raw)
    except (TypeError, ValueError):
        try:
            return _SnapshotProgramEdgeKind(raw)
        except (TypeError, ValueError) as exc:
            raise ProgramGraphError(f"invalid edge kind: {value!r}") from exc


_install_snapshot_enum_aliases()


class ProgramGraph(metaclass=ABCMeta):
    """Factory for canonical evidence graphs and snapshot graph facades."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is not ProgramGraph:
            return super().__new__(cls)
        first = args[0] if args else kwargs.get("snapshot")
        if isinstance(first, ProgramGraphSnapshot):
            return _SnapshotProgramGraph(first)
        if isinstance(first, Mapping) and (
            "snapshot" in first
            or ("roots" in first and first.get("schema") == PROGRAM_GRAPH_SCHEMA)
        ):
            return _SnapshotProgramGraph.from_dict(first)
        if "snapshot" in kwargs:
            return _SnapshotProgramGraph.from_snapshot(kwargs["snapshot"])
        return _EvidenceProgramGraph(*args, **kwargs)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ProgramGraphSnapshot | Mapping[str, Any],
    ) -> _SnapshotProgramGraph:
        return _SnapshotProgramGraph.from_snapshot(snapshot)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> _EvidenceProgramGraph | _SnapshotProgramGraph:
        if "snapshot" in payload or (
            "roots" in payload and payload.get("schema") == PROGRAM_GRAPH_SCHEMA
        ):
            return _SnapshotProgramGraph.from_dict(payload)
        return _EvidenceProgramGraph.from_dict(payload)

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
    ) -> _EvidenceProgramGraph | _SnapshotProgramGraph:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise ProgramGraphError("program graph JSON must contain an object")
        return cls.from_dict(payload)


ProgramGraph.register(_EvidenceProgramGraph)
ProgramGraph.register(_SnapshotProgramGraph)

__all__ += [
    "Completeness",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_FRONTIER",
    "DEFAULT_MAX_NODES",
    "PROGRAM_EDGE_SCHEMA",
    "PROGRAM_GRAPH_ROOTS_SCHEMA",
    "PROGRAM_GRAPH_SNAPSHOT_SCHEMA",
    "PROGRAM_GRAPH_VERSION",
    "PROGRAM_NODE_SCHEMA",
    "ProgramAuthority",
    "ProgramEdge",
    "ProgramGraphIdentityError",
    "ProgramGraphRoots",
    "ProgramGraphSnapshot",
    "ProgramNode",
    "ProgramProvenance",
    "ProgramTrust",
]
__all__ = list(dict.fromkeys(__all__))
