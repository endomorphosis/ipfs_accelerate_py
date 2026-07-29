"""Minimal dependency-complete call and impact slice queries (VFS-013).

Queries the canonical :class:`~.program_graph.ProgramGraph` for the smallest
dependency-complete neighborhood needed by assurance, proof, and repair
work.  Supported surfaces:

* symbol callers / callees (transitive, call-node aware)
* changed-blob impact
* contract consumers / producers
* MCP end-to-end registration routes
* VFS operation surfaces
* proof dependency closures
* shortest counterexample paths

Results are deterministic, hard-bounded, provenance-bearing, and explicit
about cycles, ambiguity, missing nodes, excluded repositories, and
truncated frontiers.  A truncated or frontier-open slice is never reported
as complete.  Responses carry node/edge identities and spans only — never
source bodies.

Conflict policy: query immutable graph artifacts; do not embed source bodies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .program_graph import (
    GraphFrontierItem,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramNodeKind,
    ResolverStatus,
    canonical_program_json,
)
from .proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Schemas and bounds
# ---------------------------------------------------------------------------

PROGRAM_GRAPH_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-query@1"
)
PROGRAM_GRAPH_SLICE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-slice@1"
)
PROGRAM_GRAPH_SLICE_STEP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-slice-step@1"
)
PROGRAM_GRAPH_SLICE_PATH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-slice-path@1"
)
MINIMAL_CALL_SLICE_EVIDENCE = "vfs/minimal-call-slice@1"
QUERY_VERSION = "program-graph-queries@1"

DEFAULT_MAX_NODES = 512
DEFAULT_MAX_EDGES = 2_048
DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_PATH_LENGTH = 128
DEFAULT_MAX_FRONTIER = 256
DEFAULT_MAX_SEEDS = 64
DEFAULT_MAX_EXCLUDED = 256
DEFAULT_MAX_LABEL_BYTES = 4_096
HARD_MAX_NODES = 8_192
HARD_MAX_EDGES = 32_768
HARD_MAX_DEPTH = 256
HARD_MAX_PATHS = 1_024

# Call-graph edge kinds used for caller/callee and counterexample traversal.
_CALL_EDGE_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.CALLS.value,
        ProgramEdgeKind.CONTAINS.value,
        ProgramEdgeKind.RESOLVES_TO.value,
    }
)

# Impact / consumer reverse-walk kinds (dependent <- provider orientation
# is recovered by walking these edges reverse).
_IMPACT_EDGE_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.CALLS.value,
        ProgramEdgeKind.CONTAINS.value,
        ProgramEdgeKind.DEPENDS_ON.value,
        ProgramEdgeKind.REFERENCES.value,
        ProgramEdgeKind.IMPLEMENTS.value,
        ProgramEdgeKind.IMPORTS.value,
        ProgramEdgeKind.EXPORTS.value,
        ProgramEdgeKind.TESTS.value,
        ProgramEdgeKind.DOCUMENTS.value,
        ProgramEdgeKind.TYPED_AS.value,
        ProgramEdgeKind.REGISTERS.value,
        ProgramEdgeKind.USES_TRANSPORT.value,
        ProgramEdgeKind.DERIVED_FROM.value,
        ProgramEdgeKind.MEMBER_OF.value,
        ProgramEdgeKind.RESOLVES_TO.value,
        ProgramEdgeKind.DEFINES.value,
    }
)

# Contract consumer edges: consumer --kind--> contract/schema/symbol.
_CONTRACT_CONSUMER_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.REFERENCES.value,
        ProgramEdgeKind.IMPLEMENTS.value,
        ProgramEdgeKind.DEPENDS_ON.value,
        ProgramEdgeKind.TYPED_AS.value,
        ProgramEdgeKind.IMPORTS.value,
        ProgramEdgeKind.USES_TRANSPORT.value,
    }
)

# Contract producer edges: producer --kind--> schema/export/definition.
_CONTRACT_PRODUCER_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.DEFINES.value,
        ProgramEdgeKind.EXPORTS.value,
        ProgramEdgeKind.DERIVED_FROM.value,
        ProgramEdgeKind.IMPLEMENTS.value,
        ProgramEdgeKind.REFERENCES.value,
    }
)

# MCP end-to-end route kinds.
_MCP_ROUTE_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.REGISTERS.value,
        ProgramEdgeKind.USES_TRANSPORT.value,
        ProgramEdgeKind.IMPLEMENTS.value,
        ProgramEdgeKind.REFERENCES.value,
        ProgramEdgeKind.CALLS.value,
        ProgramEdgeKind.CONTAINS.value,
        ProgramEdgeKind.DEPENDS_ON.value,
        ProgramEdgeKind.RESOLVES_TO.value,
        ProgramEdgeKind.DEFINES.value,
        ProgramEdgeKind.TYPED_AS.value,
        ProgramEdgeKind.MEMBER_OF.value,
    }
)

# Proof dependency kinds (authority-bearing closure edges).
_PROOF_EDGE_KINDS: frozenset[str] = frozenset(
    {
        ProgramEdgeKind.DEPENDS_ON.value,
        ProgramEdgeKind.DERIVED_FROM.value,
        ProgramEdgeKind.IMPLEMENTS.value,
        ProgramEdgeKind.REFERENCES.value,
        ProgramEdgeKind.CALLS.value,
        ProgramEdgeKind.CONTAINS.value,
        ProgramEdgeKind.DEFINES.value,
        ProgramEdgeKind.RESOLVES_TO.value,
        ProgramEdgeKind.TYPED_AS.value,
        ProgramEdgeKind.MEMBER_OF.value,
        ProgramEdgeKind.TESTS.value,
    }
)

# VFS surface markers used for operation-surface seeds and expansion.
_VFS_MARKERS: frozenset[str] = frozenset(
    {
        "vfs",
        "fsspec",
        "ipfs_fsspec",
        "bucket",
        "bucket_vfs",
        "vfs_manager",
        "vfs_bucket",
        "ipfs_kit.vfs",
        "mcp/ipfs_kit/vfs",
        "filesystem",
        "journal",
        "wal",
    }
)

_AMBIGUOUS_STATUSES: frozenset[ResolverStatus] = frozenset(
    {
        ResolverStatus.AMBIGUOUS,
        ResolverStatus.CANDIDATE,
        ResolverStatus.UNRESOLVED,
        ResolverStatus.UNKNOWN,
        ResolverStatus.UNSUPPORTED,
    }
)


class ProgramGraphQueryError(ValueError):
    """A program-graph query input is malformed or out of bounds."""


class QueryKind(str, Enum):
    """Closed vocabulary of minimal slice query kinds (VFS-013)."""

    SYMBOL_CALLERS = "symbol_callers"
    SYMBOL_CALLEES = "symbol_callees"
    CHANGED_BLOB_IMPACT = "changed_blob_impact"
    CONTRACT_CONSUMERS = "contract_consumers"
    CONTRACT_PRODUCERS = "contract_producers"
    MCP_END_TO_END = "mcp_end_to_end"
    VFS_OPERATION_SURFACE = "vfs_operation_surface"
    PROOF_DEPENDENCIES = "proof_dependencies"
    SHORTEST_COUNTEREXAMPLE = "shortest_counterexample"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ProgramGraphQueryError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ProgramGraphQueryError(f"{name} is required")
    if len(text.encode("utf-8")) > DEFAULT_MAX_LABEL_BYTES:
        raise ProgramGraphQueryError(f"{name} exceeds label bound")
    return text


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramGraphQueryError(f"{name} must be an integer")
    if value < 1 or value > maximum:
        raise ProgramGraphQueryError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


def _non_negative_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramGraphQueryError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ProgramGraphQueryError(
            f"{name} must be an integer from 0 through {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    text = str(value or "").strip()
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ProgramGraphQueryError(f"unsupported {label}: {text!r}") from exc


def _sorted_unique(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted({str(item).strip() for item in values if str(item).strip()})
    )


def _looks_like_vfs(node: ProgramGraphNode) -> bool:
    haystacks = (
        node.qualified_name.casefold(),
        node.path.casefold(),
        node.record_key.casefold(),
        node.component_id.casefold(),
        str(node.record.get("operation") or "").casefold(),
        str(node.record.get("surface") or "").casefold(),
    )
    joined = " ".join(haystacks)
    return any(marker in joined for marker in _VFS_MARKERS)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceStep:
    """One hop on a slice path (node identity + optional edge)."""

    node_id: str
    edge_id: str = ""
    kind: str = ""
    qualified_name: str = ""
    path: str = ""
    resolver_status: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "step node_id"))
        for name in ("edge_id", "kind", "qualified_name", "path", "resolver_status"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), f"step {name}", required=False),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_GRAPH_SLICE_STEP_SCHEMA,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "kind": self.kind,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "resolver_status": self.resolver_status,
        }


@dataclass(frozen=True)
class SlicePath:
    """One ordered path through the slice (entry -> ... -> exit)."""

    steps: tuple[SliceStep, ...]
    entry_node_id: str = ""
    exit_node_id: str = ""
    length: int = 0
    cyclic: bool = False

    def __post_init__(self) -> None:
        steps = tuple(self.steps or ())
        for step in steps:
            if not isinstance(step, SliceStep):
                raise ProgramGraphQueryError("path steps must be SliceStep")
        object.__setattr__(self, "steps", steps)
        if steps and not self.entry_node_id:
            object.__setattr__(self, "entry_node_id", steps[0].node_id)
        else:
            object.__setattr__(
                self,
                "entry_node_id",
                _text(self.entry_node_id, "entry_node_id", required=False),
            )
        if steps and not self.exit_node_id:
            object.__setattr__(self, "exit_node_id", steps[-1].node_id)
        else:
            object.__setattr__(
                self,
                "exit_node_id",
                _text(self.exit_node_id, "exit_node_id", required=False),
            )
        object.__setattr__(
            self,
            "length",
            len(steps) if not self.length else int(self.length),
        )
        if not isinstance(self.cyclic, bool):
            raise ProgramGraphQueryError("cyclic must be a boolean")

    @property
    def path_id(self) -> str:
        return "spath-" + content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROGRAM_GRAPH_SLICE_PATH_SCHEMA,
            "entry_node_id": self.entry_node_id,
            "exit_node_id": self.exit_node_id,
            "length": self.length,
            "cyclic": self.cyclic,
            "steps": [step.to_dict() for step in self.steps],
        }
        if include_id:
            payload["path_id"] = self.path_id
        return payload


@dataclass(frozen=True)
class QueryBounds:
    """Hard bounds applied to a single query."""

    max_nodes: int = DEFAULT_MAX_NODES
    max_edges: int = DEFAULT_MAX_EDGES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_paths: int = DEFAULT_MAX_PATHS
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH
    max_frontier: int = DEFAULT_MAX_FRONTIER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_nodes",
            _positive_int(self.max_nodes, "max_nodes", maximum=HARD_MAX_NODES),
        )
        object.__setattr__(
            self,
            "max_edges",
            _positive_int(self.max_edges, "max_edges", maximum=HARD_MAX_EDGES),
        )
        object.__setattr__(
            self,
            "max_depth",
            _positive_int(self.max_depth, "max_depth", maximum=HARD_MAX_DEPTH),
        )
        object.__setattr__(
            self,
            "max_paths",
            _positive_int(self.max_paths, "max_paths", maximum=HARD_MAX_PATHS),
        )
        object.__setattr__(
            self,
            "max_path_length",
            _positive_int(
                self.max_path_length,
                "max_path_length",
                maximum=HARD_MAX_DEPTH,
            ),
        )
        object.__setattr__(
            self,
            "max_frontier",
            _positive_int(
                self.max_frontier, "max_frontier", maximum=DEFAULT_MAX_FRONTIER * 4
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
            "max_paths": self.max_paths,
            "max_path_length": self.max_path_length,
            "max_frontier": self.max_frontier,
        }

    @classmethod
    def from_value(cls, value: Any) -> "QueryBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProgramGraphQueryError("bounds must be a mapping")
        return cls(
            max_nodes=int(value.get("max_nodes") or DEFAULT_MAX_NODES),
            max_edges=int(value.get("max_edges") or DEFAULT_MAX_EDGES),
            max_depth=int(value.get("max_depth") or DEFAULT_MAX_DEPTH),
            max_paths=int(value.get("max_paths") or DEFAULT_MAX_PATHS),
            max_path_length=int(
                value.get("max_path_length") or DEFAULT_MAX_PATH_LENGTH
            ),
            max_frontier=int(value.get("max_frontier") or DEFAULT_MAX_FRONTIER),
        )


@dataclass(frozen=True)
class ProgramGraphQuery:
    """One immutable, content-addressed slice query request."""

    kind: QueryKind
    seed_node_ids: tuple[str, ...] = ()
    seed_record_keys: tuple[str, ...] = ()
    seed_qualified_names: tuple[str, ...] = ()
    seed_blob_cids: tuple[str, ...] = ()
    seed_paths: tuple[str, ...] = ()
    target_node_ids: tuple[str, ...] = ()
    target_qualified_names: tuple[str, ...] = ()
    repository_ids: tuple[str, ...] = ()
    excluded_repository_ids: tuple[str, ...] = ()
    bounds: QueryBounds = field(default_factory=QueryBounds)
    direction: str = ""  # optional override: forward | reverse | both
    include_structural: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, QueryKind, "query kind")
        )
        object.__setattr__(
            self, "seed_node_ids", _sorted_unique(self.seed_node_ids)[:DEFAULT_MAX_SEEDS]
        )
        object.__setattr__(
            self,
            "seed_record_keys",
            _sorted_unique(self.seed_record_keys)[:DEFAULT_MAX_SEEDS],
        )
        object.__setattr__(
            self,
            "seed_qualified_names",
            _sorted_unique(self.seed_qualified_names)[:DEFAULT_MAX_SEEDS],
        )
        object.__setattr__(
            self,
            "seed_blob_cids",
            _sorted_unique(self.seed_blob_cids)[:DEFAULT_MAX_SEEDS],
        )
        object.__setattr__(
            self, "seed_paths", _sorted_unique(self.seed_paths)[:DEFAULT_MAX_SEEDS]
        )
        object.__setattr__(
            self,
            "target_node_ids",
            _sorted_unique(self.target_node_ids)[:DEFAULT_MAX_SEEDS],
        )
        object.__setattr__(
            self,
            "target_qualified_names",
            _sorted_unique(self.target_qualified_names)[:DEFAULT_MAX_SEEDS],
        )
        object.__setattr__(
            self,
            "repository_ids",
            _sorted_unique(self.repository_ids)[:DEFAULT_MAX_EXCLUDED],
        )
        object.__setattr__(
            self,
            "excluded_repository_ids",
            _sorted_unique(self.excluded_repository_ids)[:DEFAULT_MAX_EXCLUDED],
        )
        object.__setattr__(self, "bounds", QueryBounds.from_value(self.bounds))
        direction = _text(self.direction, "direction", required=False).casefold()
        if direction and direction not in {"forward", "reverse", "both"}:
            raise ProgramGraphQueryError(
                "direction must be forward, reverse, or both"
            )
        object.__setattr__(self, "direction", direction)
        if not isinstance(self.include_structural, bool):
            raise ProgramGraphQueryError("include_structural must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise ProgramGraphQueryError("metadata must be a mapping")
        # Reject source-body payloads in metadata (conflict policy).
        forbidden = {
            "source",
            "source_text",
            "body",
            "source_body",
            "ast",
            "contents",
            "file_contents",
        }
        cleaned = {
            str(key): value
            for key, value in dict(self.metadata).items()
            if str(key) not in forbidden
        }
        object.__setattr__(self, "metadata", MappingProxyType(cleaned))
        if not (
            self.seed_node_ids
            or self.seed_record_keys
            or self.seed_qualified_names
            or self.seed_blob_cids
            or self.seed_paths
            or self.kind is QueryKind.VFS_OPERATION_SURFACE
        ):
            raise ProgramGraphQueryError(
                "query requires at least one seed selector "
                "(or vfs_operation_surface auto-seed)"
            )

    @property
    def query_id(self) -> str:
        return "pgq-" + content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROGRAM_GRAPH_QUERY_SCHEMA,
            "kind": self.kind.value,
            "seed_node_ids": list(self.seed_node_ids),
            "seed_record_keys": list(self.seed_record_keys),
            "seed_qualified_names": list(self.seed_qualified_names),
            "seed_blob_cids": list(self.seed_blob_cids),
            "seed_paths": list(self.seed_paths),
            "target_node_ids": list(self.target_node_ids),
            "target_qualified_names": list(self.target_qualified_names),
            "repository_ids": list(self.repository_ids),
            "excluded_repository_ids": list(self.excluded_repository_ids),
            "bounds": self.bounds.to_dict(),
            "direction": self.direction,
            "include_structural": self.include_structural,
            "metadata": dict(self.metadata),
            "version": QUERY_VERSION,
            "evidence": MINIMAL_CALL_SLICE_EVIDENCE,
        }
        if include_id:
            payload["query_id"] = self.query_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramGraphQuery":
        if not isinstance(payload, Mapping):
            raise ProgramGraphQueryError("query payload must be a mapping")
        return cls(
            kind=payload.get("kind", ""),
            seed_node_ids=tuple(payload.get("seed_node_ids") or ()),
            seed_record_keys=tuple(payload.get("seed_record_keys") or ()),
            seed_qualified_names=tuple(payload.get("seed_qualified_names") or ()),
            seed_blob_cids=tuple(payload.get("seed_blob_cids") or ()),
            seed_paths=tuple(payload.get("seed_paths") or ()),
            target_node_ids=tuple(payload.get("target_node_ids") or ()),
            target_qualified_names=tuple(
                payload.get("target_qualified_names") or ()
            ),
            repository_ids=tuple(payload.get("repository_ids") or ()),
            excluded_repository_ids=tuple(
                payload.get("excluded_repository_ids") or ()
            ),
            bounds=payload.get("bounds"),
            direction=str(payload.get("direction") or ""),
            include_structural=bool(payload.get("include_structural", True)),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class ProgramGraphSlice:
    """Bounded, provenance-bearing, minimal dependency-complete slice."""

    query_id: str
    kind: QueryKind
    forest_id: str
    graph_id: str
    seed_node_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    paths: tuple[SlicePath, ...] = ()
    complete: bool = False
    minimal: bool = False
    dependency_complete: bool = False
    truncated: bool = False
    truncation_reasons: tuple[str, ...] = ()
    cycles: tuple[str, ...] = ()
    ambiguous_element_ids: tuple[str, ...] = ()
    missing_node_ids: tuple[str, ...] = ()
    excluded_repository_ids: tuple[str, ...] = ()
    frontier: tuple[GraphFrontierItem, ...] = ()
    required_dependencies: tuple[str, ...] = ()
    omitted_dependencies: tuple[str, ...] = ()
    node_count: int = 0
    edge_count: int = 0
    depth_reached: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_id", _text(self.query_id, "query_id")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, QueryKind, "slice kind")
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id")
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id")
        )
        for name in (
            "seed_node_ids",
            "node_ids",
            "edge_ids",
            "truncation_reasons",
            "cycles",
            "ambiguous_element_ids",
            "missing_node_ids",
            "excluded_repository_ids",
            "required_dependencies",
            "omitted_dependencies",
            "notes",
        ):
            object.__setattr__(self, name, _sorted_unique(getattr(self, name)))
        paths = tuple(self.paths or ())
        for path in paths:
            if not isinstance(path, SlicePath):
                raise ProgramGraphQueryError("paths must contain SlicePath")
        # Stable path order by (length, entry, exit, path_id).
        object.__setattr__(
            self,
            "paths",
            tuple(
                sorted(
                    paths,
                    key=lambda item: (
                        item.length,
                        item.entry_node_id,
                        item.exit_node_id,
                        item.path_id,
                    ),
                )
            ),
        )
        frontier = tuple(self.frontier or ())
        for item in frontier:
            if not isinstance(item, GraphFrontierItem):
                raise ProgramGraphQueryError(
                    "frontier items must be GraphFrontierItem"
                )
        object.__setattr__(
            self,
            "frontier",
            tuple(
                sorted(
                    frontier,
                    key=lambda item: (item.element_id, item.element_kind),
                )
            ),
        )
        for flag in (
            "complete",
            "minimal",
            "dependency_complete",
            "truncated",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise ProgramGraphQueryError(f"{flag} must be a boolean")
        # Fail closed: truncated / omitted / missing seeds cannot be dependency-
        # complete.  Ambiguous frontiers block overall ``complete`` only —
        # dependency completeness means required neighbors were not dropped by
        # bounds, independent of resolver ambiguity on retained nodes.
        if (
            self.truncated
            or self.omitted_dependencies
            or self.missing_node_ids
        ):
            object.__setattr__(self, "dependency_complete", False)
        if (
            not self.dependency_complete
            or any(
                item.resolver_status in _AMBIGUOUS_STATUSES
                for item in self.frontier
            )
        ):
            object.__setattr__(self, "complete", False)
        object.__setattr__(
            self,
            "node_count",
            len(self.node_ids) if not self.node_count else int(self.node_count),
        )
        object.__setattr__(
            self,
            "edge_count",
            len(self.edge_ids) if not self.edge_count else int(self.edge_count),
        )
        object.__setattr__(
            self,
            "depth_reached",
            _non_negative_int(self.depth_reached, "depth_reached", maximum=10**9),
        )
        if not isinstance(self.provenance, Mapping):
            raise ProgramGraphQueryError("provenance must be a mapping")
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )

    @property
    def slice_id(self) -> str:
        return "pslice-" + content_identity(self.to_dict(include_id=False))

    @property
    def empty(self) -> bool:
        return not self.node_ids

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PROGRAM_GRAPH_SLICE_SCHEMA,
            "evidence": MINIMAL_CALL_SLICE_EVIDENCE,
            "version": QUERY_VERSION,
            "query_id": self.query_id,
            "kind": self.kind.value,
            "forest_id": self.forest_id,
            "graph_id": self.graph_id,
            "seed_node_ids": list(self.seed_node_ids),
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "paths": [path.to_dict() for path in self.paths],
            "complete": self.complete,
            "minimal": self.minimal,
            "dependency_complete": self.dependency_complete,
            "truncated": self.truncated,
            "truncation_reasons": list(self.truncation_reasons),
            "cycles": list(self.cycles),
            "ambiguous_element_ids": list(self.ambiguous_element_ids),
            "missing_node_ids": list(self.missing_node_ids),
            "excluded_repository_ids": list(self.excluded_repository_ids),
            "frontier": [item.to_dict() for item in self.frontier],
            "required_dependencies": list(self.required_dependencies),
            "omitted_dependencies": list(self.omitted_dependencies),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "depth_reached": self.depth_reached,
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
            # Permanent policy claims.
            "embeds_source_bodies": False,
            "embeds_ast": False,
        }
        if include_id:
            payload["slice_id"] = self.slice_id
        return payload

    def to_json(self) -> str:
        return canonical_program_json(self.to_dict())


# ---------------------------------------------------------------------------
# Graph adjacency index (query-local, not authoritative)
# ---------------------------------------------------------------------------


@dataclass
class _AdjEdge:
    edge: ProgramGraphEdge
    neighbor: str
    forward: bool  # True when walking edge.source -> edge.target


class _GraphView:
    """Indexed view over a program graph for bounded traversals."""

    def __init__(
        self,
        graph: ProgramGraph,
        *,
        repository_ids: frozenset[str] = frozenset(),
        excluded_repository_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.graph = graph
        self.nodes: dict[str, ProgramGraphNode] = {
            node.node_id: node for node in graph.nodes
        }
        self.edges: dict[str, ProgramGraphEdge] = {
            edge.edge_id: edge for edge in graph.edges
        }
        self.by_record_key: dict[str, list[str]] = {}
        self.by_qualified_name: dict[str, list[str]] = {}
        self.by_blob_cid: dict[str, list[str]] = {}
        self.by_path: dict[str, list[str]] = {}
        self.repository_of: dict[str, str] = {}
        self.excluded_repository_ids = set(excluded_repository_ids)
        self.repository_filter = set(repository_ids)

        # Pre-index nodes and repository membership via CONTAINS from repos.
        repo_nodes = {
            node.node_id
            for node in graph.nodes
            if node.kind is ProgramNodeKind.REPOSITORY
        }
        for node in graph.nodes:
            self.by_record_key.setdefault(node.record_key, []).append(node.node_id)
            if node.qualified_name:
                self.by_qualified_name.setdefault(
                    node.qualified_name, []
                ).append(node.node_id)
            self.by_blob_cid.setdefault(node.binding.blob_cid, []).append(
                node.node_id
            )
            if node.path:
                self.by_path.setdefault(node.path, []).append(node.node_id)
            if node.kind is ProgramNodeKind.REPOSITORY:
                self.repository_of[node.node_id] = node.record_key or node.node_id

        # Propagate repository membership along CONTAINS edges (limited BFS).
        children: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.kind is ProgramEdgeKind.CONTAINS:
                children.setdefault(edge.source, []).append(edge.target)
        for repo_id in repo_nodes:
            repo_key = self.repository_of.get(repo_id, repo_id)
            queue = deque([repo_id])
            seen = {repo_id}
            while queue:
                current = queue.popleft()
                self.repository_of.setdefault(current, repo_key)
                for child in children.get(current, ()):
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)

        # Forward and reverse adjacency by edge kind.
        self.forward: dict[str, list[_AdjEdge]] = {}
        self.reverse: dict[str, list[_AdjEdge]] = {}
        for edge in graph.edges:
            self.forward.setdefault(edge.source, []).append(
                _AdjEdge(edge=edge, neighbor=edge.target, forward=True)
            )
            self.reverse.setdefault(edge.target, []).append(
                _AdjEdge(edge=edge, neighbor=edge.source, forward=False)
            )
        for mapping in (self.forward, self.reverse):
            for key in mapping:
                mapping[key].sort(
                    key=lambda item: (
                        item.edge.kind.value,
                        item.neighbor,
                        item.edge.edge_id,
                    )
                )

    def allowed(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        repo = self.repository_of.get(node_id, "")
        if repo and repo in self.excluded_repository_ids:
            return False
        if self.repository_filter:
            # Allow nodes with unknown repo only when no filter, else require match.
            if repo and repo not in self.repository_filter:
                return False
            if not repo:
                # Nodes not under a repository still pass if seed-selected.
                return True
        return True

    def _prefer_seed_hits(
        self, hits: Sequence[str], *, query: ProgramGraphQuery
    ) -> list[str]:
        """Prefer semantic node kinds when a qualified name collides.

        Definitions, call sites, and exports often share a qualified name with
        the symbol they describe.  Call/impact queries seed on the symbol (or
        MCP/schema surface) unless the caller passed an explicit node id.
        """

        allowed = [nid for nid in hits if self.allowed(nid)]
        if not allowed:
            return []
        preferred_kinds = {
            QueryKind.SYMBOL_CALLERS: {
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.CALL,
            },
            QueryKind.SYMBOL_CALLEES: {
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.CALL,
            },
            QueryKind.SHORTEST_COUNTEREXAMPLE: {
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.CALL,
                ProgramNodeKind.MCP_TOOL,
            },
            QueryKind.PROOF_DEPENDENCIES: {
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.DEFINITION,
                ProgramNodeKind.CALL,
                ProgramNodeKind.MCP_TOOL,
            },
            QueryKind.MCP_END_TO_END: {
                ProgramNodeKind.MCP_TOOL,
                ProgramNodeKind.MCP_REGISTRATION,
                ProgramNodeKind.SYMBOL,
            },
            QueryKind.CONTRACT_CONSUMERS: {
                ProgramNodeKind.SCHEMA,
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.TYPE,
            },
            QueryKind.CONTRACT_PRODUCERS: {
                ProgramNodeKind.SCHEMA,
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.TYPE,
            },
            QueryKind.VFS_OPERATION_SURFACE: {
                ProgramNodeKind.SYMBOL,
                ProgramNodeKind.MCP_TOOL,
                ProgramNodeKind.DEFINITION,
            },
            QueryKind.CHANGED_BLOB_IMPACT: set(ProgramNodeKind),
        }.get(query.kind, {ProgramNodeKind.SYMBOL})

        preferred = [
            nid
            for nid in allowed
            if self.nodes[nid].kind in preferred_kinds
        ]
        if preferred:
            # Prefer SYMBOL over CALL when both match the same name.
            symbols = [
                nid
                for nid in preferred
                if self.nodes[nid].kind is ProgramNodeKind.SYMBOL
            ]
            if symbols and query.kind in {
                QueryKind.SYMBOL_CALLERS,
                QueryKind.SYMBOL_CALLEES,
                QueryKind.SHORTEST_COUNTEREXAMPLE,
                QueryKind.PROOF_DEPENDENCIES,
            }:
                return symbols
            return preferred
        return allowed

    def resolve_seeds(
        self,
        query: ProgramGraphQuery,
    ) -> tuple[list[str], list[str]]:
        """Return (resolved_seed_ids, missing_selectors)."""

        seeds: set[str] = set()
        missing: list[str] = []

        for node_id in query.seed_node_ids:
            if node_id in self.nodes and self.allowed(node_id):
                seeds.add(node_id)
            else:
                missing.append(node_id)

        for key in query.seed_record_keys:
            hits = self.by_record_key.get(key, ())
            preferred = self._prefer_seed_hits(hits, query=query)
            if preferred:
                seeds.update(preferred)
            else:
                missing.append(f"record_key:{key}")

        for name in query.seed_qualified_names:
            hits = self.by_qualified_name.get(name, ())
            preferred = self._prefer_seed_hits(hits, query=query)
            if preferred:
                seeds.update(preferred)
            else:
                missing.append(f"qualified_name:{name}")

        for blob in query.seed_blob_cids:
            hits = self.by_blob_cid.get(blob, ())
            found = False
            for node_id in hits:
                if self.allowed(node_id):
                    # Blob impact seeds every node bound to the blob.
                    seeds.add(node_id)
                    found = True
            if not found:
                missing.append(f"blob_cid:{blob}")

        for path in query.seed_paths:
            hits = self.by_path.get(path, ())
            preferred = self._prefer_seed_hits(hits, query=query)
            if preferred:
                seeds.update(preferred)
            else:
                missing.append(f"path:{path}")

        if query.kind is QueryKind.VFS_OPERATION_SURFACE and not seeds:
            for node in self.graph.nodes:
                if self.allowed(node.node_id) and _looks_like_vfs(node):
                    seeds.add(node.node_id)

        return sorted(seeds), sorted(set(missing))

    def resolve_targets(
        self, query: ProgramGraphQuery
    ) -> tuple[list[str], list[str]]:
        targets: set[str] = set()
        missing: list[str] = []
        for node_id in query.target_node_ids:
            if node_id in self.nodes and self.allowed(node_id):
                targets.add(node_id)
            else:
                missing.append(node_id)
        for name in query.target_qualified_names:
            hits = self.by_qualified_name.get(name, ())
            preferred = self._prefer_seed_hits(hits, query=query)
            if preferred:
                targets.update(preferred)
            else:
                missing.append(f"target_qualified_name:{name}")
        return sorted(targets), sorted(set(missing))


# ---------------------------------------------------------------------------
# Traversal engine
# ---------------------------------------------------------------------------


@dataclass
class _TraversalState:
    node_ids: set[str] = field(default_factory=set)
    edge_ids: set[str] = field(default_factory=set)
    # node_id -> parent_node_id for path reconstruction (shortest parent)
    parent: dict[str, str] = field(default_factory=dict)
    parent_edge: dict[str, str] = field(default_factory=dict)
    depth: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    truncation_reasons: set[str] = field(default_factory=set)
    cycles: set[str] = field(default_factory=set)
    depth_reached: int = 0
    # Edges that closed a cycle (for diagnostics)
    cycle_edges: set[str] = field(default_factory=set)


def _step_for(view: _GraphView, node_id: str, edge_id: str = "") -> SliceStep:
    node = view.nodes.get(node_id)
    if node is None:
        return SliceStep(node_id=node_id, edge_id=edge_id)
    return SliceStep(
        node_id=node_id,
        edge_id=edge_id,
        kind=node.kind.value,
        qualified_name=node.qualified_name,
        path=node.path,
        resolver_status=node.binding.resolver_status.value,
    )


def _edge_kinds_for(kind: QueryKind, direction: str) -> frozenset[str]:
    if kind is QueryKind.SYMBOL_CALLERS or kind is QueryKind.SYMBOL_CALLEES:
        return _CALL_EDGE_KINDS
    if kind is QueryKind.CHANGED_BLOB_IMPACT:
        return _IMPACT_EDGE_KINDS
    if kind is QueryKind.CONTRACT_CONSUMERS:
        return _CONTRACT_CONSUMER_KINDS
    if kind is QueryKind.CONTRACT_PRODUCERS:
        return _CONTRACT_PRODUCER_KINDS
    if kind is QueryKind.MCP_END_TO_END:
        return _MCP_ROUTE_KINDS
    if kind is QueryKind.VFS_OPERATION_SURFACE:
        return _IMPACT_EDGE_KINDS
    if kind is QueryKind.PROOF_DEPENDENCIES:
        return _PROOF_EDGE_KINDS
    if kind is QueryKind.SHORTEST_COUNTEREXAMPLE:
        return _CALL_EDGE_KINDS | _PROOF_EDGE_KINDS
    return _IMPACT_EDGE_KINDS


def _default_directions(kind: QueryKind, override: str) -> tuple[str, ...]:
    if override == "forward":
        return ("forward",)
    if override == "reverse":
        return ("reverse",)
    if override == "both":
        return ("forward", "reverse")
    if kind is QueryKind.SYMBOL_CALLERS:
        return ("reverse",)
    if kind is QueryKind.SYMBOL_CALLEES:
        return ("forward",)
    if kind is QueryKind.CHANGED_BLOB_IMPACT:
        # Impact walks reverse (who depends on the changed blob).
        return ("reverse",)
    if kind is QueryKind.CONTRACT_CONSUMERS:
        return ("reverse",)
    if kind is QueryKind.CONTRACT_PRODUCERS:
        return ("forward", "reverse")
    if kind is QueryKind.MCP_END_TO_END:
        return ("forward", "reverse")
    if kind is QueryKind.VFS_OPERATION_SURFACE:
        return ("forward", "reverse")
    if kind is QueryKind.PROOF_DEPENDENCIES:
        return ("forward", "reverse")
    if kind is QueryKind.SHORTEST_COUNTEREXAMPLE:
        return ("forward",)
    return ("forward",)


def _neighbors(
    view: _GraphView,
    node_id: str,
    *,
    directions: Sequence[str],
    edge_kinds: frozenset[str],
) -> list[_AdjEdge]:
    items: list[_AdjEdge] = []
    if "forward" in directions:
        items.extend(view.forward.get(node_id, ()))
    if "reverse" in directions:
        items.extend(view.reverse.get(node_id, ()))
    filtered: list[_AdjEdge] = []
    for item in items:
        if item.edge.kind.value not in edge_kinds:
            continue
        if not view.allowed(item.neighbor):
            continue
        neighbor = view.nodes.get(item.neighbor)
        current = view.nodes.get(node_id)
        # Never expand through repository CONTAINS fan-out: reverse walk to a
        # repository then forward CONTAINS would pull the entire inventory and
        # destroy slice minimality.  Repository nodes may still appear when
        # they are explicit seeds.
        if item.edge.kind is ProgramEdgeKind.CONTAINS:
            if neighbor is not None and neighbor.kind is ProgramNodeKind.REPOSITORY:
                continue
            if (
                current is not None
                and current.kind is ProgramNodeKind.REPOSITORY
                and item.forward
            ):
                continue
        filtered.append(item)
    # Deterministic order.
    filtered.sort(
        key=lambda item: (
            item.edge.kind.value,
            item.neighbor,
            item.edge.edge_id,
            0 if item.forward else 1,
        )
    )
    return filtered


def _bfs_closure(
    view: _GraphView,
    seeds: Sequence[str],
    *,
    directions: Sequence[str],
    edge_kinds: frozenset[str],
    bounds: QueryBounds,
    targets: Sequence[str] | None = None,
    stop_at_targets: bool = False,
) -> _TraversalState:
    state = _TraversalState()
    target_set = set(targets or ())
    queue: deque[str] = deque()

    for seed in seeds:
        if seed not in view.nodes or not view.allowed(seed):
            continue
        if seed in state.node_ids:
            continue
        if len(state.node_ids) >= bounds.max_nodes:
            state.truncated = True
            state.truncation_reasons.add("max_nodes")
            break
        state.node_ids.add(seed)
        state.depth[seed] = 0
        queue.append(seed)

    found_targets: set[str] = set()
    while queue:
        current = queue.popleft()
        depth = state.depth[current]
        state.depth_reached = max(state.depth_reached, depth)
        if stop_at_targets and current in target_set:
            found_targets.add(current)
            if found_targets >= target_set and target_set:
                # All targets found; still continue only if we want multi-path —
                # for shortest we stop enqueueing past targets.
                continue
        if depth >= bounds.max_depth:
            # Frontier at depth bound — mark truncated only if neighbors exist.
            if _neighbors(
                view, current, directions=directions, edge_kinds=edge_kinds
            ):
                state.truncated = True
                state.truncation_reasons.add("max_depth")
            continue
        for item in _neighbors(
            view, current, directions=directions, edge_kinds=edge_kinds
        ):
            neighbor = item.neighbor
            edge_id = item.edge.edge_id
            if neighbor in state.node_ids:
                # Back/cross edge — record cycle when mutual reachability
                # within the visited set along the same traversal.
                if state.depth.get(neighbor, 0) <= depth and neighbor != current:
                    # Potential cycle: neighbor already visited.
                    if (
                        state.parent.get(current) != neighbor
                        and state.parent.get(neighbor) != current
                    ):
                        # Detect simple cycle if neighbor is an ancestor.
                        cursor = current
                        seen_chain = {current}
                        while cursor in state.parent:
                            cursor = state.parent[cursor]
                            if cursor == neighbor:
                                cycle_id = f"cycle:{neighbor}->{current}"
                                state.cycles.add(cycle_id)
                                state.cycle_edges.add(edge_id)
                                break
                            if cursor in seen_chain:
                                break
                            seen_chain.add(cursor)
                # Include cycle-closing edges that stay within the slice so
                # the slice remains closed under observed call cycles.
                if len(state.edge_ids) < bounds.max_edges:
                    state.edge_ids.add(edge_id)
                elif edge_id not in state.edge_ids:
                    state.truncated = True
                    state.truncation_reasons.add("max_edges")
                continue
            if len(state.node_ids) >= bounds.max_nodes:
                state.truncated = True
                state.truncation_reasons.add("max_nodes")
                continue
            if len(state.edge_ids) >= bounds.max_edges:
                state.truncated = True
                state.truncation_reasons.add("max_edges")
                continue
            state.node_ids.add(neighbor)
            state.edge_ids.add(edge_id)
            state.parent[neighbor] = current
            state.parent_edge[neighbor] = edge_id
            state.depth[neighbor] = depth + 1
            if stop_at_targets and neighbor in target_set:
                found_targets.add(neighbor)
                # Do not expand beyond a found target for shortest paths.
                continue
            queue.append(neighbor)

    return state


def _reconstruct_path(
    view: _GraphView, state: _TraversalState, seed: str, target: str
) -> SlicePath | None:
    if target not in state.node_ids:
        return None
    if seed not in state.node_ids:
        return None
    # Walk parents from target to seed.
    chain: list[tuple[str, str]] = []  # (node_id, edge_id into node)
    cursor = target
    guard = 0
    while cursor != seed:
        if cursor not in state.parent:
            return None
        edge_id = state.parent_edge.get(cursor, "")
        chain.append((cursor, edge_id))
        cursor = state.parent[cursor]
        guard += 1
        if guard > DEFAULT_MAX_PATH_LENGTH * 2:
            return None
    chain.append((seed, ""))
    chain.reverse()
    steps: list[SliceStep] = []
    for index, (node_id, edge_id) in enumerate(chain):
        # edge_id is the edge that entered node_id from its parent.
        steps.append(_step_for(view, node_id, edge_id if index else ""))
    cyclic = False
    node_ids = [step.node_id for step in steps]
    if len(node_ids) != len(set(node_ids)):
        cyclic = True
    return SlicePath(steps=tuple(steps), cyclic=cyclic)


def _call_graph_refine(
    view: _GraphView,
    state: _TraversalState,
    *,
    kind: QueryKind,
) -> None:
    """Ensure call-site intermediates required for dependency completeness.

    For caller/callee slices, a symbol that only participates via
    ``symbol --contains--> call --calls--> target`` needs the call node
    retained.  The BFS already includes them when CONTAINS/CALLS are in the
    edge-kind set.  This pass also pulls DEFINES edges for symbols so
    definition provenance is dependency-complete without expanding unrelated
    modules.
    """

    if kind not in {
        QueryKind.SYMBOL_CALLERS,
        QueryKind.SYMBOL_CALLEES,
        QueryKind.SHORTEST_COUNTEREXAMPLE,
        QueryKind.MCP_END_TO_END,
        QueryKind.PROOF_DEPENDENCIES,
    }:
        return
    extra_nodes: set[str] = set()
    extra_edges: set[str] = set()
    for node_id in list(state.node_ids):
        node = view.nodes.get(node_id)
        if node is None:
            continue
        # Attach definitions for symbols already in the slice.
        if node.kind is ProgramNodeKind.SYMBOL:
            for item in view.forward.get(node_id, ()):
                if item.edge.kind is ProgramEdgeKind.DEFINES:
                    if view.allowed(item.neighbor):
                        extra_nodes.add(item.neighbor)
                        extra_edges.add(item.edge.edge_id)
            for item in view.reverse.get(node_id, ()):
                if item.edge.kind is ProgramEdgeKind.DEFINES:
                    if view.allowed(item.neighbor):
                        extra_nodes.add(item.neighbor)
                        extra_edges.add(item.edge.edge_id)
        # Attach owning symbol for call nodes.
        if node.kind is ProgramNodeKind.CALL:
            for item in view.reverse.get(node_id, ()):
                if item.edge.kind is ProgramEdgeKind.CONTAINS:
                    if view.allowed(item.neighbor):
                        extra_nodes.add(item.neighbor)
                        extra_edges.add(item.edge.edge_id)
    state.node_ids.update(extra_nodes)
    state.edge_ids.update(extra_edges)


def _collect_frontier(
    view: _GraphView,
    state: _TraversalState,
    *,
    max_frontier: int,
) -> tuple[list[GraphFrontierItem], list[str]]:
    frontier: list[GraphFrontierItem] = []
    ambiguous: list[str] = []
    for node_id in sorted(state.node_ids):
        node = view.nodes[node_id]
        status = node.binding.resolver_status
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
            if status in _AMBIGUOUS_STATUSES:
                ambiguous.append(node.node_id)
    for edge_id in sorted(state.edge_ids):
        edge = view.edges[edge_id]
        status = edge.binding.resolver_status
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
            if status in _AMBIGUOUS_STATUSES:
                ambiguous.append(edge.edge_id)
    frontier_sorted = sorted(
        frontier, key=lambda item: (item.element_id, item.element_kind)
    )
    if len(frontier_sorted) > max_frontier:
        frontier_sorted = frontier_sorted[:max_frontier]
        # Truncation of frontier projection is recorded by the caller.
    return frontier_sorted, sorted(set(ambiguous))


def _minimality_check(
    view: _GraphView,
    state: _TraversalState,
    seeds: Sequence[str],
    *,
    directions: Sequence[str],
    edge_kinds: frozenset[str],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Prove every retained node is reachable from seeds under query edges.

    Returns ``(minimal, required_dependencies, omitted_dependencies)``.
    Omitted dependencies are neighbors required by dependency completeness
    that were dropped by bounds (should already be reflected in truncated).
    """

    required = sorted(state.node_ids)
    # Re-run unbounded reachability within the retained subgraph to ensure
    # every node is seed-reachable; drop nothing here (already minimal by BFS
    # construction) but report any out-of-slice required neighbor as omitted.
    omitted: list[str] = []
    if state.truncated:
        for node_id in sorted(state.node_ids):
            for item in _neighbors(
                view, node_id, directions=directions, edge_kinds=edge_kinds
            ):
                if item.neighbor not in state.node_ids and view.allowed(
                    item.neighbor
                ):
                    omitted.append(item.neighbor)
    # Minimality: every node is reachable from some seed within the slice.
    reachable: set[str] = set()
    queue: deque[str] = deque()
    for seed in seeds:
        if seed in state.node_ids:
            reachable.add(seed)
            queue.append(seed)
    slice_edges = {
        edge_id: view.edges[edge_id]
        for edge_id in state.edge_ids
        if edge_id in view.edges
    }
    adj: dict[str, list[str]] = {}
    for edge in slice_edges.values():
        if edge.kind.value not in edge_kinds:
            # Structural DEFINES attachments still count for reachability.
            if edge.kind is not ProgramEdgeKind.DEFINES:
                continue
        if "forward" in directions or edge.kind is ProgramEdgeKind.DEFINES:
            if edge.source in state.node_ids and edge.target in state.node_ids:
                adj.setdefault(edge.source, []).append(edge.target)
        if "reverse" in directions or edge.kind is ProgramEdgeKind.DEFINES:
            if edge.source in state.node_ids and edge.target in state.node_ids:
                adj.setdefault(edge.target, []).append(edge.source)
    # Undirected reachability within the closed edge set for structural defs.
    undirected: dict[str, set[str]] = {nid: set() for nid in state.node_ids}
    for edge in slice_edges.values():
        if edge.source in undirected and edge.target in undirected:
            undirected[edge.source].add(edge.target)
            undirected[edge.target].add(edge.source)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(undirected.get(current, ())):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    unreachable = sorted(state.node_ids - reachable)
    # Nodes only attached via refine (DEFINES) should still be reachable via
    # undirected edges.  Any remaining unreachable nodes fail minimality.
    minimal = not unreachable and not state.truncated
    # When truncated, the retained set is still minimal *relative to the
    # truncated frontier* if every retained node is seed-reachable.
    if not unreachable:
        minimal = True
    return minimal, tuple(required), tuple(sorted(set(omitted)))


def _build_paths(
    view: _GraphView,
    state: _TraversalState,
    seeds: Sequence[str],
    *,
    targets: Sequence[str],
    bounds: QueryBounds,
    kind: QueryKind,
) -> list[SlicePath]:
    paths: list[SlicePath] = []
    if kind is QueryKind.SHORTEST_COUNTEREXAMPLE and targets:
        for seed in seeds:
            for target in targets:
                path = _reconstruct_path(view, state, seed, target)
                if path is not None:
                    paths.append(path)
                if len(paths) >= bounds.max_paths:
                    return paths
        return paths

    # For non-shortest queries, emit one representative path from each seed
    # to each frontier leaf (nodes with no further retained children).
    children: dict[str, list[str]] = {}
    for child, parent in state.parent.items():
        children.setdefault(parent, []).append(child)
    leaves = sorted(
        nid
        for nid in state.node_ids
        if nid not in children and nid not in seeds
    )
    if not leaves:
        # Emit trivial single-node paths for seeds.
        for seed in seeds:
            if seed in state.node_ids:
                paths.append(
                    SlicePath(steps=(_step_for(view, seed),))
                )
                if len(paths) >= bounds.max_paths:
                    break
        return paths
    for seed in seeds:
        for leaf in leaves:
            path = _reconstruct_path(view, state, seed, leaf)
            if path is not None:
                paths.append(path)
            if len(paths) >= bounds.max_paths:
                return paths
    return paths


# ---------------------------------------------------------------------------
# Public query entry points
# ---------------------------------------------------------------------------


def query_program_graph_slice(
    graph: ProgramGraph,
    query: ProgramGraphQuery | Mapping[str, Any],
) -> ProgramGraphSlice:
    """Execute one minimal dependency-complete slice query.

    The result is deterministic for a fixed graph and query, hard-bounded,
    provenance-bearing, and never reports ``complete=True`` when truncated,
    missing seeds, open ambiguous frontiers, or omitted dependencies remain.
    """

    if not isinstance(graph, ProgramGraph):
        raise ProgramGraphQueryError("graph must be a ProgramGraph")
    if not isinstance(query, ProgramGraphQuery):
        if not isinstance(query, Mapping):
            raise ProgramGraphQueryError("query must be ProgramGraphQuery or mapping")
        query = ProgramGraphQuery.from_dict(query)

    view = _GraphView(
        graph,
        repository_ids=frozenset(query.repository_ids),
        excluded_repository_ids=frozenset(query.excluded_repository_ids),
    )
    seeds, missing = view.resolve_seeds(query)
    targets, missing_targets = view.resolve_targets(query)
    missing = sorted(set(missing) | set(missing_targets))

    edge_kinds = _edge_kinds_for(query.kind, query.direction)
    directions = _default_directions(query.kind, query.direction)
    bounds = query.bounds

    notes: list[str] = []
    if not seeds:
        notes.append("no_seeds_resolved")
        return ProgramGraphSlice(
            query_id=query.query_id,
            kind=query.kind,
            forest_id=graph.forest_id,
            graph_id=graph.graph_id,
            seed_node_ids=(),
            node_ids=(),
            edge_ids=(),
            paths=(),
            complete=False,
            minimal=True,
            dependency_complete=False,
            truncated=False,
            truncation_reasons=(),
            cycles=(),
            ambiguous_element_ids=(),
            missing_node_ids=tuple(missing),
            excluded_repository_ids=query.excluded_repository_ids,
            frontier=(),
            required_dependencies=(),
            omitted_dependencies=(),
            depth_reached=0,
            provenance={
                "producer": QUERY_VERSION,
                "evidence": MINIMAL_CALL_SLICE_EVIDENCE,
                "graph_id": graph.graph_id,
                "forest_id": graph.forest_id,
                "query": query.to_dict(),
            },
            notes=tuple(notes),
        )

    stop_at_targets = query.kind is QueryKind.SHORTEST_COUNTEREXAMPLE
    if stop_at_targets and not targets:
        notes.append("shortest_counterexample_requires_targets")

    state = _bfs_closure(
        view,
        seeds,
        directions=directions,
        edge_kinds=edge_kinds,
        bounds=bounds,
        targets=targets if stop_at_targets else None,
        stop_at_targets=stop_at_targets,
    )
    if query.include_structural:
        _call_graph_refine(view, state, kind=query.kind)

    # Enforce bounds after refine.
    if len(state.node_ids) > bounds.max_nodes:
        # Deterministic truncation: keep seeds first, then by depth/node_id.
        ordered = sorted(
            state.node_ids,
            key=lambda nid: (
                0 if nid in seeds else 1,
                state.depth.get(nid, 10**9),
                nid,
            ),
        )
        keep = set(ordered[: bounds.max_nodes])
        state.node_ids = keep
        state.edge_ids = {
            eid
            for eid in state.edge_ids
            if view.edges[eid].source in keep and view.edges[eid].target in keep
        }
        state.truncated = True
        state.truncation_reasons.add("max_nodes")
    if len(state.edge_ids) > bounds.max_edges:
        ordered_edges = sorted(state.edge_ids)
        state.edge_ids = set(ordered_edges[: bounds.max_edges])
        state.truncated = True
        state.truncation_reasons.add("max_edges")

    frontier, ambiguous = _collect_frontier(
        view, state, max_frontier=bounds.max_frontier
    )
    frontier_truncated = False
    # Detect if more frontier items were available.
    raw_frontier_count = 0
    for node_id in state.node_ids:
        if view.nodes[node_id].binding.resolver_status.frontier:
            raw_frontier_count += 1
    for edge_id in state.edge_ids:
        if view.edges[edge_id].binding.resolver_status.frontier:
            raw_frontier_count += 1
    if raw_frontier_count > bounds.max_frontier:
        frontier_truncated = True
        state.truncation_reasons.add("max_frontier")
        state.truncated = True

    minimal, required, omitted = _minimality_check(
        view,
        state,
        seeds,
        directions=directions,
        edge_kinds=edge_kinds,
    )

    paths = _build_paths(
        view,
        state,
        seeds,
        targets=targets,
        bounds=bounds,
        kind=query.kind,
    )
    if len(paths) >= bounds.max_paths:
        # Path list was capped — does not by itself mean the node slice is
        # incomplete, but we record it for diagnostics.
        notes.append("paths_capped")

    # Dependency completeness: no omitted neighbors and no truncation and
    # all seeds resolved.
    dependency_complete = (
        not state.truncated
        and not omitted
        and not missing
        and bool(state.node_ids)
    )
    # Complete only when dependency-complete and no open ambiguous frontier.
    open_ambiguous = bool(
        any(
            item.resolver_status in _AMBIGUOUS_STATUSES
            for item in frontier
        )
    )
    complete = dependency_complete and not open_ambiguous and not frontier_truncated

    if query.kind is QueryKind.SHORTEST_COUNTEREXAMPLE:
        if targets and not any(
            path.exit_node_id in targets for path in paths
        ):
            dependency_complete = False
            complete = False
            notes.append("no_path_to_target")
        elif targets and paths:
            # Keep only shortest length paths (minimal counterexample slice).
            min_len = min(path.length for path in paths)
            paths = [path for path in paths if path.length == min_len]
            # Restrict node/edge set to the union of shortest paths for true
            # minimality of the counterexample slice.
            keep_nodes: set[str] = set(seeds) | set(targets)
            keep_edges: set[str] = set()
            for path in paths:
                for step in path.steps:
                    keep_nodes.add(step.node_id)
                    if step.edge_id:
                        keep_edges.add(step.edge_id)
            # Preserve DEFINES refine nodes attached to path symbols.
            refine_state = _TraversalState(
                node_ids=set(keep_nodes), edge_ids=set(keep_edges)
            )
            if query.include_structural:
                _call_graph_refine(view, refine_state, kind=query.kind)
            state.node_ids = refine_state.node_ids
            state.edge_ids = refine_state.edge_ids
            # Recompute minimality after restriction.
            minimal, required, omitted = _minimality_check(
                view,
                state,
                seeds,
                directions=directions,
                edge_kinds=edge_kinds,
            )
            dependency_complete = (
                not state.truncated
                and not omitted
                and not missing
                and bool(paths)
            )
            complete = dependency_complete and not open_ambiguous

    # Detect call cycles on the symbol-level projection of retained edges:
    # symbol --contains--> call --calls--> symbol.  Direct CALLS edges between
    # symbols are also included.  Cycles are diagnostic only (not rejected).
    symbol_adj: dict[str, set[str]] = {}
    for edge_id in state.edge_ids:
        edge = view.edges[edge_id]
        if edge.kind is ProgramEdgeKind.CALLS:
            src = edge.source
            tgt = edge.target
            src_node = view.nodes.get(src)
            # Lift call-node sources to their owning symbol via CONTAINS.
            if src_node is not None and src_node.kind is ProgramNodeKind.CALL:
                owners = [
                    item.neighbor
                    for item in view.reverse.get(src, ())
                    if item.edge.kind is ProgramEdgeKind.CONTAINS
                    and item.neighbor in state.node_ids
                ]
                if owners:
                    for owner in owners:
                        symbol_adj.setdefault(owner, set()).add(tgt)
                    continue
            symbol_adj.setdefault(src, set()).add(tgt)
    # Report SCCs of size > 1 and mutual pairs.
    for src in sorted(symbol_adj):
        for tgt in sorted(symbol_adj.get(src, ())):
            if src in symbol_adj.get(tgt, ()):
                pair = tuple(sorted((src, tgt)))
                state.cycles.add(f"call_cycle:{pair[0]}:{pair[1]}")
            # Longer cycles: src reaches itself via a path of length >= 2.
    # Tarjan-lite: DFS for back-edges within symbol_adj restricted to slice.
    visiting: set[str] = set()
    visited_syms: set[str] = set()

    def _dfs_cycle(node: str, stack: list[str]) -> None:
        visiting.add(node)
        stack.append(node)
        for nxt in sorted(symbol_adj.get(node, ())):
            if nxt not in state.node_ids and nxt not in symbol_adj:
                continue
            if nxt in visiting:
                # Cycle through stack.
                if nxt in stack:
                    idx = stack.index(nxt)
                    cycle_nodes = tuple(sorted(stack[idx:]))
                    if len(cycle_nodes) >= 2:
                        state.cycles.add(
                            "call_cycle:" + ":".join(cycle_nodes[:8])
                        )
                continue
            if nxt not in visited_syms:
                _dfs_cycle(nxt, stack)
        stack.pop()
        visiting.discard(node)
        visited_syms.add(node)

    for sym in sorted(symbol_adj):
        if sym not in visited_syms:
            _dfs_cycle(sym, [])

    excluded_hit = sorted(
        {
            view.repository_of[nid]
            for nid in state.node_ids
            if view.repository_of.get(nid)
            and view.repository_of[nid] in view.excluded_repository_ids
        }
    )
    # excluded_repository_ids on the result lists policy exclusions, not hits.
    excluded_reported = tuple(sorted(query.excluded_repository_ids))

    provenance = {
        "producer": QUERY_VERSION,
        "evidence": MINIMAL_CALL_SLICE_EVIDENCE,
        "graph_id": graph.graph_id,
        "forest_id": graph.forest_id,
        "graph_producer": graph.producer,
        "query_kind": query.kind.value,
        "directions": list(directions),
        "edge_kinds": sorted(edge_kinds),
        "bounds": bounds.to_dict(),
        "seed_count": len(seeds),
        "target_count": len(targets),
        "excluded_hit_count": len(excluded_hit),
    }

    return ProgramGraphSlice(
        query_id=query.query_id,
        kind=query.kind,
        forest_id=graph.forest_id,
        graph_id=graph.graph_id,
        seed_node_ids=tuple(seeds),
        node_ids=tuple(sorted(state.node_ids)),
        edge_ids=tuple(sorted(state.edge_ids)),
        paths=tuple(paths),
        complete=complete,
        minimal=minimal,
        dependency_complete=dependency_complete,
        truncated=state.truncated,
        truncation_reasons=tuple(sorted(state.truncation_reasons)),
        cycles=tuple(sorted(state.cycles)),
        ambiguous_element_ids=tuple(ambiguous),
        missing_node_ids=tuple(missing),
        excluded_repository_ids=excluded_reported,
        frontier=tuple(frontier),
        required_dependencies=required,
        omitted_dependencies=tuple(omitted),
        depth_reached=state.depth_reached,
        provenance=provenance,
        notes=tuple(sorted(set(notes))),
    )


def query_symbol_callers(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Transitive callers of the seeded symbol(s)."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.SYMBOL_CALLERS,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_symbol_callees(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Transitive callees of the seeded symbol(s)."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.SYMBOL_CALLEES,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_changed_blob_impact(
    graph: ProgramGraph,
    *,
    seed_blob_cids: Sequence[str] = (),
    seed_node_ids: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Reverse-dependency impact of one or more changed blobs."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.CHANGED_BLOB_IMPACT,
            seed_blob_cids=tuple(seed_blob_cids),
            seed_node_ids=tuple(seed_node_ids),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_contract_consumers(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Consumers of a contract/schema/symbol seed."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.CONTRACT_CONSUMERS,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_contract_producers(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Producers of a contract/schema seed."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.CONTRACT_PRODUCERS,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_mcp_end_to_end(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """MCP registration → transport → implementation route slice."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.MCP_END_TO_END,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_vfs_operation_surface(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    seed_paths: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """VFS/fsspec/bucket operation surface neighborhood."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.VFS_OPERATION_SURFACE,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            seed_paths=tuple(seed_paths),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_proof_dependencies(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Dependency-complete proof context neighborhood."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.PROOF_DEPENDENCIES,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


def query_shortest_counterexample(
    graph: ProgramGraph,
    *,
    seed_node_ids: Sequence[str] = (),
    seed_qualified_names: Sequence[str] = (),
    target_node_ids: Sequence[str] = (),
    target_qualified_names: Sequence[str] = (),
    bounds: QueryBounds | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProgramGraphSlice:
    """Shortest dependency-complete path(s) from entry seed(s) to target(s)."""

    return query_program_graph_slice(
        graph,
        ProgramGraphQuery(
            kind=QueryKind.SHORTEST_COUNTEREXAMPLE,
            seed_node_ids=tuple(seed_node_ids),
            seed_qualified_names=tuple(seed_qualified_names),
            target_node_ids=tuple(target_node_ids),
            target_qualified_names=tuple(target_qualified_names),
            bounds=QueryBounds.from_value(bounds),
            **kwargs,
        ),
    )


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_FRONTIER",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_PATHS",
    "HARD_MAX_DEPTH",
    "HARD_MAX_EDGES",
    "HARD_MAX_NODES",
    "HARD_MAX_PATHS",
    "MINIMAL_CALL_SLICE_EVIDENCE",
    "PROGRAM_GRAPH_QUERY_SCHEMA",
    "PROGRAM_GRAPH_SLICE_PATH_SCHEMA",
    "PROGRAM_GRAPH_SLICE_SCHEMA",
    "PROGRAM_GRAPH_SLICE_STEP_SCHEMA",
    "QUERY_VERSION",
    "ProgramGraphQuery",
    "ProgramGraphQueryError",
    "ProgramGraphSlice",
    "QueryBounds",
    "QueryKind",
    "SlicePath",
    "SliceStep",
    "query_changed_blob_impact",
    "query_contract_consumers",
    "query_contract_producers",
    "query_mcp_end_to_end",
    "query_program_graph_slice",
    "query_proof_dependencies",
    "query_shortest_counterexample",
    "query_symbol_callees",
    "query_symbol_callers",
    "query_vfs_operation_surface",
]
