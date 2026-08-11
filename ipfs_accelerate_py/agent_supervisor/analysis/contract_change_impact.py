"""Reverse transitive contract-change impact closure and SCCs (RPR-028).

Starting from each :class:`ProgramContractDelta`, compute a dependency-complete
reverse transitive consumer closure over a snapshot-bound
:class:`~ipfs_accelerate_py.agent_supervisor.program_graph.ProgramGraph`.

Authority rules
---------------
* Only trusted, authority-bearing program-graph edges expand the mandatory
  consumer worklist.
* GraphRAG / vector / runtime / history / model edges may only *nominate*
  frontier endpoints; they never close coverage or promote completeness.
* Truncation, stale graph/index roots, unresolved required routes, open graph
  frontiers, and forged completeness claims cannot yield
  :attr:`ImpactCompleteness.COMPLETE`.

Consumers are deduplicated by graph node while every supporting edge path is
retained. Strongly connected components are computed deterministically
(Tarjan, sorted adjacency) and ordered by topological condensation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .program_graph import (
    Completeness,
    ProgramEdge,
    ProgramEdgeKind,
    ProgramGraph,
    ProgramGraphSnapshot,
    ProgramNode,
    ProgramNodeKind,
    ProgramProvenance,
)
from .change_propagation_contracts import (
    CHANGE_PROPAGATION_VERSION,
    MAX_CONSUMER_COUNT,
    MAX_REFERENCE_COUNT,
    MAX_SCC_COUNT,
    ChangePropagationAuthorityError,
    ChangePropagationBoundsError,
    ChangePropagationError,
    GraphEdgeKind,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    ImpactSCC,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)

try:
    from .code_evidence_graph import CodeImpactIndex, CodeImpactResult
except Exception:  # pragma: no cover - optional co-location
    CodeImpactIndex = None  # type: ignore[misc, assignment]
    CodeImpactResult = None  # type: ignore[misc, assignment]


CONTRACT_CHANGE_IMPACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-change-impact@1"
)
CONTRACT_CHANGE_IMPACT_VERSION: Final[str] = "contract-change-impact@1"
DEFAULT_EXTRACTOR_ID: Final[str] = "contract-change-impact@1"

DEFAULT_MAX_CONSUMERS: Final[int] = min(MAX_CONSUMER_COUNT, 4_096)
DEFAULT_MAX_DEPTH: Final[int] = 256
DEFAULT_MAX_EDGES: Final[int] = 65_536
DEFAULT_MAX_SCCS: Final[int] = min(MAX_SCC_COUNT, 256)
HARD_MAX_DEPTH: Final[int] = 4_096
HARD_MAX_EDGES: Final[int] = 250_000

# Edge kinds treated as consumer → provider (source depends on target).
# Reverse impact of a changed provider walks *incoming* edges of these kinds.
_CONSUMER_TO_PROVIDER: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CALLS,
        ProgramEdgeKind.DEPENDS_ON,
        ProgramEdgeKind.TESTS,
        ProgramEdgeKind.MOCKS,
        ProgramEdgeKind.FIXTURES,
        ProgramEdgeKind.VALIDATES,
        ProgramEdgeKind.DOCUMENTS,
        ProgramEdgeKind.IMPORTS,
        ProgramEdgeKind.ALIASES,
        ProgramEdgeKind.RE_EXPORTS,
        ProgramEdgeKind.OVERRIDES,
        ProgramEdgeKind.IMPLEMENTS,
        ProgramEdgeKind.OVERLOADS,
        ProgramEdgeKind.PARAMETER_OF,
        ProgramEdgeKind.FIELD_OF,
        ProgramEdgeKind.GENERATED_FROM,
        ProgramEdgeKind.NATIVE_BOUND,
        ProgramEdgeKind.USES_RESOURCE,
        ProgramEdgeKind.REQUIRES_CAPABILITY,
        ProgramEdgeKind.EFFECT_OF,
        ProgramEdgeKind.PATH_CONDITION,
        ProgramEdgeKind.DOMINATES,
        ProgramEdgeKind.REACHES,
    }
)

# Edge kinds treated as provider → consumer (target depends on / is derived
# from source). Reverse impact of a changed provider also walks *outgoing*
# edges of these kinds so constructors, serializers, APIs, and data sinks
# surface as consumers.
_PROVIDER_TO_CONSUMER: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CONSTRUCTS,
        ProgramEdgeKind.FACTORY_CREATES,
        ProgramEdgeKind.BUILDER_BUILDS,
        ProgramEdgeKind.DECORATES,
        ProgramEdgeKind.REGISTERS,
        ProgramEdgeKind.INJECTS,
        ProgramEdgeKind.CALLBACK_TO,
        ProgramEdgeKind.CONTEXT_MANAGES,
        ProgramEdgeKind.DATA_FLOW,
        ProgramEdgeKind.STATE_FLOW,
        ProgramEdgeKind.RETURNS,
        ProgramEdgeKind.SERIALIZES,
        ProgramEdgeKind.DESERIALIZES,
        ProgramEdgeKind.MIGRATES,
        ProgramEdgeKind.SCHEMA_OF,
        ProgramEdgeKind.SERVES,
        ProgramEdgeKind.CONFIGURES,
        ProgramEdgeKind.DEFINES,
        ProgramEdgeKind.DECLARES,
        ProgramEdgeKind.OWNS,
    }
)

# Structural noise never expands impact. RELATED_TO is nominated-only and is
# still observed so it can surface as an explicit frontier endpoint.
_SKIP_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CONTAINS,
    }
)

# Node kinds that are themselves an unknown/unsupported frontier endpoint.
_FRONTIER_NODE_KINDS: Final[frozenset[ProgramNodeKind]] = frozenset(
    {
        ProgramNodeKind.FRONTIER,
        ProgramNodeKind.UNSUPPORTED,
        ProgramNodeKind.EXTERNAL,
        ProgramNodeKind.NATIVE_BOUNDARY,
    }
)

_NOMINATED_PROVENANCE: Final[frozenset[ProgramProvenance]] = frozenset(
    {
        ProgramProvenance.RUNTIME,
        ProgramProvenance.GRAPHRAG,
        ProgramProvenance.VECTOR,
        ProgramProvenance.HISTORY,
        ProgramProvenance.MODEL,
    }
)

_EDGE_KIND_TO_GRAPH_REF: Final[Mapping[ProgramEdgeKind, GraphEdgeKind]] = {
    ProgramEdgeKind.CALLS: GraphEdgeKind.CALL,
    ProgramEdgeKind.OVERRIDES: GraphEdgeKind.OVERRIDE,
    ProgramEdgeKind.IMPLEMENTS: GraphEdgeKind.OVERRIDE,
    ProgramEdgeKind.OVERLOADS: GraphEdgeKind.OVERRIDE,
    ProgramEdgeKind.DATA_FLOW: GraphEdgeKind.DATA_FLOW,
    ProgramEdgeKind.STATE_FLOW: GraphEdgeKind.STATE_FLOW,
    ProgramEdgeKind.REACHES: GraphEdgeKind.DATA_FLOW,
    ProgramEdgeKind.SCHEMA_OF: GraphEdgeKind.SCHEMA,
    ProgramEdgeKind.SERIALIZES: GraphEdgeKind.SCHEMA,
    ProgramEdgeKind.DESERIALIZES: GraphEdgeKind.SCHEMA,
    ProgramEdgeKind.MIGRATES: GraphEdgeKind.SCHEMA,
    ProgramEdgeKind.IMPORTS: GraphEdgeKind.IMPORT,
    ProgramEdgeKind.EXPORTS: GraphEdgeKind.IMPORT,
    ProgramEdgeKind.RE_EXPORTS: GraphEdgeKind.IMPORT,
    ProgramEdgeKind.ALIASES: GraphEdgeKind.IMPORT,
    ProgramEdgeKind.REGISTERS: GraphEdgeKind.REGISTRATION,
    ProgramEdgeKind.INJECTS: GraphEdgeKind.REGISTRATION,
    ProgramEdgeKind.OWNS: GraphEdgeKind.OWNERSHIP,
    ProgramEdgeKind.VALIDATES: GraphEdgeKind.VALIDATION,
    ProgramEdgeKind.TESTS: GraphEdgeKind.VALIDATION,
    ProgramEdgeKind.MOCKS: GraphEdgeKind.VALIDATION,
    ProgramEdgeKind.FIXTURES: GraphEdgeKind.VALIDATION,
    ProgramEdgeKind.CONSTRUCTS: GraphEdgeKind.WIRING,
    ProgramEdgeKind.FACTORY_CREATES: GraphEdgeKind.WIRING,
    ProgramEdgeKind.BUILDER_BUILDS: GraphEdgeKind.WIRING,
    ProgramEdgeKind.DECORATES: GraphEdgeKind.WIRING,
    ProgramEdgeKind.CALLBACK_TO: GraphEdgeKind.WIRING,
    ProgramEdgeKind.SERVES: GraphEdgeKind.WIRING,
    ProgramEdgeKind.CONFIGURES: GraphEdgeKind.WIRING,
    ProgramEdgeKind.DEPENDS_ON: GraphEdgeKind.WIRING,
}


class ContractChangeImpactError(ChangePropagationError):
    """Impact-closure construction failed a fail-closed invariant."""


class ContractChangeImpactBoundsError(
    ContractChangeImpactError, ChangePropagationBoundsError
):
    """Impact closure exceeded a declared resource bound."""


class ContractChangeImpactAuthorityError(
    ContractChangeImpactError, ChangePropagationAuthorityError
):
    """Impact closure attempted to promote untrusted or stale authority."""


@dataclass(frozen=True)
class ImpactClosureBounds:
    """Hard deterministic resource limits for reverse impact traversal."""

    max_consumers: int = DEFAULT_MAX_CONSUMERS
    max_depth: int = DEFAULT_MAX_DEPTH
    max_edges: int = DEFAULT_MAX_EDGES
    max_sccs: int = DEFAULT_MAX_SCCS

    def __post_init__(self) -> None:
        limits = {
            "max_consumers": (self.max_consumers, 1, MAX_CONSUMER_COUNT),
            "max_depth": (self.max_depth, 0, HARD_MAX_DEPTH),
            "max_edges": (self.max_edges, 1, HARD_MAX_EDGES),
            "max_sccs": (self.max_sccs, 1, MAX_SCC_COUNT),
        }
        for name, (value, minimum, maximum) in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or value > maximum
            ):
                raise ContractChangeImpactBoundsError(
                    f"{name} must be an integer from {minimum} through {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_consumers": self.max_consumers,
            "max_depth": self.max_depth,
            "max_edges": self.max_edges,
            "max_sccs": self.max_sccs,
        }

    @property
    def bound_ref(self) -> str:
        return (
            "bound:impact-closure:"
            f"c{self.max_consumers}:d{self.max_depth}:"
            f"e{self.max_edges}:s{self.max_sccs}"
        )


@dataclass(frozen=True)
class ImpactTraversalStep:
    """One retained reverse-edge path step for diagnostics / path retention."""

    edge_id: str
    kind: str
    source_node_id: str
    target_node_id: str
    authoritative: bool
    nominated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "authoritative": self.authoritative,
            "nominated": self.nominated,
        }


@dataclass
class _ConsumerState:
    node_id: str
    depth: int
    edge_refs: set[str] = field(default_factory=set)
    path_condition_refs: set[str] = field(default_factory=set)
    mandatory: bool = False
    frontier: bool = False


def _coerce_graph(
    graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any],
) -> ProgramGraph:
    if isinstance(graph, ProgramGraph):
        return graph
    if isinstance(graph, ProgramGraphSnapshot):
        return ProgramGraph(graph)
    if isinstance(graph, Mapping):
        if "snapshot" in graph or graph.get("schema", "").endswith("program-graph@1"):
            return ProgramGraph.from_dict(graph)
        return ProgramGraph(ProgramGraphSnapshot.from_dict(graph))
    raise ContractChangeImpactError(
        "impact analysis requires a ProgramGraph or ProgramGraphSnapshot"
    )


def _coerce_delta(delta: ProgramContractDelta | Mapping[str, Any]) -> ProgramContractDelta:
    if isinstance(delta, ProgramContractDelta):
        return delta
    if isinstance(delta, Mapping):
        return ProgramContractDelta.from_dict(delta)
    raise ContractChangeImpactError("impact analysis requires a ProgramContractDelta")


def _coerce_bounds(bounds: ImpactClosureBounds | Mapping[str, Any] | None) -> ImpactClosureBounds:
    if bounds is None:
        return ImpactClosureBounds()
    if isinstance(bounds, ImpactClosureBounds):
        return bounds
    if isinstance(bounds, Mapping):
        return ImpactClosureBounds(
            max_consumers=int(bounds.get("max_consumers", DEFAULT_MAX_CONSUMERS)),
            max_depth=int(bounds.get("max_depth", DEFAULT_MAX_DEPTH)),
            max_edges=int(bounds.get("max_edges", DEFAULT_MAX_EDGES)),
            max_sccs=int(bounds.get("max_sccs", DEFAULT_MAX_SCCS)),
        )
    raise ContractChangeImpactError("bounds must be ImpactClosureBounds or a mapping")


def _node_symbol_keys(node: ProgramNode) -> tuple[str, ...]:
    keys: list[str] = []
    for raw in (
        node.node_id,
        node.qualified_name,
        node.name,
        node.provenance_id,
        str(node.attributes.get("symbol_id") or ""),
        str(node.attributes.get("symbol") or ""),
        f"symbol:{node.name}" if node.name else "",
        f"symbol:{node.qualified_name}" if node.qualified_name else "",
    ):
        text = str(raw or "").strip()
        if text and text not in keys:
            keys.append(text)
    return tuple(keys)


def resolve_seed_nodes(
    graph: ProgramGraph,
    subject_symbol_id: str,
) -> tuple[ProgramNode, ...]:
    """Locate graph nodes that match a delta subject symbol identity."""

    subject = str(subject_symbol_id or "").strip()
    if not subject:
        raise ContractChangeImpactError("delta subject_symbol_id is required")

    exact: list[ProgramNode] = []
    fuzzy: list[ProgramNode] = []
    subject_leaf = subject.rsplit(".", 1)[-1]
    if subject_leaf.startswith("symbol:"):
        subject_leaf = subject_leaf[len("symbol:") :]

    for node in graph.nodes:
        keys = _node_symbol_keys(node)
        if subject in keys or node.node_id == subject:
            exact.append(node)
            continue
        # Conservative secondary match on simple name / qualified suffix.
        if node.qualified_name == subject or node.name == subject:
            fuzzy.append(node)
            continue
        if subject_leaf and (
            node.name == subject_leaf
            or node.qualified_name.endswith(f".{subject_leaf}")
            or node.qualified_name == subject_leaf
        ):
            fuzzy.append(node)

    if exact:
        return tuple(sorted(exact, key=lambda item: item.node_id))
    return tuple(sorted(fuzzy, key=lambda item: item.node_id))


def _edge_is_nominated(edge: ProgramEdge) -> bool:
    return (
        edge.provenance in _NOMINATED_PROVENANCE
        or edge.provenance.nominated_only
        or not edge.authoritative
        or edge.kind is ProgramEdgeKind.RELATED_TO
    )


def _node_is_frontier(node: ProgramNode) -> bool:
    if node.kind in _FRONTIER_NODE_KINDS:
        return True
    if node.completeness in {
        Completeness.FRONTIER,
        Completeness.UNSUPPORTED,
        Completeness.UNKNOWN,
    }:
        return True
    if not node.authoritative:
        return True
    return False


def _map_edge_kind(kind: ProgramEdgeKind) -> GraphEdgeKind:
    return _EDGE_KIND_TO_GRAPH_REF.get(kind, GraphEdgeKind.WIRING)


def _compact_identifier(value: str, *, fallback: str) -> str:
    """Collapse whitespace so GraphNodeRef symbol/artifact ids stay opaque."""

    text = str(value or "").strip()
    if not text:
        text = fallback
    # Identifiers must be compact (no whitespace) for change-propagation refs.
    compact = "".join(ch if not ch.isspace() else "_" for ch in text)
    compact = compact.strip("_") or fallback
    return compact


def _node_ref(node: ProgramNode, *, frontier: bool = False) -> GraphNodeRef:
    if frontier or _node_is_frontier(node) or not node.authoritative:
        provenance = GraphProvenance.FRONTIER
        extractor = node.extractor_id or ""
    else:
        provenance = GraphProvenance.TRUSTED
        extractor = node.extractor_id or node.roots.extractor_id or DEFAULT_EXTRACTOR_ID
    raw_symbol = str(
        node.attributes.get("symbol_id")
        or node.qualified_name
        or node.name
        or node.node_id
    )
    symbol_id = _compact_identifier(raw_symbol, fallback=node.node_id)
    artifact_id = _compact_identifier(
        str(node.blob_identity or node.source_sha256 or node.provenance_id or node.node_id),
        fallback=node.node_id,
    )
    path = node.path or "unknown"
    # GraphNodeRef.path rejects absolute / parent escapes; keep a relative token.
    if path in {".", ""}:
        path = "unknown"
    return GraphNodeRef(
        node_id=node.node_id,
        kind=node.kind.value,
        path=path,
        symbol_id=symbol_id,
        artifact_id=artifact_id,
        provenance=provenance,
        extractor_id=extractor if provenance is GraphProvenance.TRUSTED else extractor,
    )


def _consumer_id_for(node_id: str) -> str:
    return f"consumer:{node_id}"


def _scc_id_for(members: Sequence[str]) -> str:
    body = ",".join(sorted(members))
    return f"scc:{body}" if len(body) <= 200 else f"scc:{len(members)}:{hash(body) & 0xFFFFFFFF:08x}"


def _reverse_neighbors(
    graph: ProgramGraph,
    node_id: str,
) -> list[tuple[ProgramEdge, str]]:
    """Yield (edge, consumer_node_id) reverse-impact adjacencies for node_id."""

    results: list[tuple[ProgramEdge, str]] = []
    seen: set[tuple[str, str]] = set()

    for edge in graph.edges_to(node_id):
        if edge.kind in _SKIP_KINDS:
            continue
        # Incoming edges always contribute a reverse-impact neighbor
        # (consumer→provider orientation, plus RELATED_TO nominations).
        key = (edge.edge_id, edge.source)
        if key not in seen:
            seen.add(key)
            results.append((edge, edge.source))

    for edge in graph.edges_from(node_id):
        if edge.kind in _SKIP_KINDS:
            continue
        if edge.kind is ProgramEdgeKind.RELATED_TO:
            # Nominated association: target may be a nominated dependent.
            key = (edge.edge_id, edge.target)
            if key not in seen:
                seen.add(key)
                results.append((edge, edge.target))
            continue
        if edge.kind in _PROVIDER_TO_CONSUMER:
            key = (edge.edge_id, edge.target)
            if key not in seen:
                seen.add(key)
                results.append((edge, edge.target))

    results.sort(
        key=lambda item: (
            item[0].kind.value,
            item[1],
            item[0].edge_id,
        )
    )
    return results


def compute_sccs(
    node_ids: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Deterministic Tarjan SCCs plus topological condensation order.

    Returns
    -------
    sccs:
        Tuple of member-id tuples, each members sorted, SCCs ordered by
        topological condensation (dependencies before dependents in the
        reverse-impact orientation).
    topo_order:
        Flattened node order consistent with the condensation.
    """

    nodes = tuple(sorted({str(item) for item in node_ids if str(item)}))
    if not nodes:
        return (), ()

    adj: dict[str, list[str]] = {node: [] for node in nodes}
    for source in nodes:
        targets = {
            str(target)
            for target in adjacency.get(source, ())
            if str(target) in adj and str(target) != source
        }
        adj[source] = sorted(targets)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    raw_sccs: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adj[node]:
            if target not in indices:
                strongconnect(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            raw_sccs.append(sorted(component))

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    # Condensation adjacency: edge A→B between components when any member
    # edge goes A_member → B_member in the reverse-impact graph.
    member_to_scc: dict[str, int] = {}
    for scc_index, members in enumerate(raw_sccs):
        for member in members:
            member_to_scc[member] = scc_index

    condensation: dict[int, set[int]] = {i: set() for i in range(len(raw_sccs))}
    indegree = [0] * len(raw_sccs)
    for source, targets in adj.items():
        src_scc = member_to_scc[source]
        for target in targets:
            dst_scc = member_to_scc[target]
            if src_scc == dst_scc:
                continue
            if dst_scc not in condensation[src_scc]:
                condensation[src_scc].add(dst_scc)
                indegree[dst_scc] += 1

    # Kahn topological order with deterministic tie-break on first member id.
    ready = sorted(
        (i for i, degree in enumerate(indegree) if degree == 0),
        key=lambda i: (raw_sccs[i][0], raw_sccs[i]),
    )
    ordered_indices: list[int] = []
    while ready:
        current = ready.pop(0)
        ordered_indices.append(current)
        nxt: list[int] = []
        for dst in sorted(condensation[current], key=lambda i: (raw_sccs[i][0], raw_sccs[i])):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                nxt.append(dst)
        ready = sorted(
            ready + nxt,
            key=lambda i: (raw_sccs[i][0], raw_sccs[i]),
        )

    # Cycles among condensation (should not happen) — append remaining.
    if len(ordered_indices) != len(raw_sccs):
        remaining = [
            i for i in range(len(raw_sccs)) if i not in set(ordered_indices)
        ]
        remaining.sort(key=lambda i: (raw_sccs[i][0], raw_sccs[i]))
        ordered_indices.extend(remaining)

    ordered = tuple(tuple(raw_sccs[i]) for i in ordered_indices)
    topo = tuple(member for scc in ordered for member in scc)
    return ordered, topo


def _impact_index_validation_refs(
    impact_index: Any,
    seed_symbols: Sequence[str],
    seed_paths: Sequence[str],
) -> tuple[str, ...]:
    if impact_index is None or CodeImpactIndex is None:
        return ()
    if not isinstance(impact_index, CodeImpactIndex):
        return ()
    try:
        result = impact_index.impact(
            changed_symbols=seed_symbols,
            changed_paths=seed_paths,
        )
    except Exception:
        return ()
    return tuple(result.required_validation_ids)


def _impact_index_extra_consumers(
    impact_index: Any,
    seed_symbols: Sequence[str],
    seed_paths: Sequence[str],
    graph: ProgramGraph,
) -> tuple[tuple[str, str], ...]:
    """Return (node_id, via_ref) pairs nominated by CodeImpactIndex reverse closure.

    Index edges are reviewed and may expand the worklist, but only when the
    dependent resolves to a graph node. Unresolved index dependents become
    frontier refs (caller responsibility).
    """

    if impact_index is None or CodeImpactIndex is None:
        return ()
    if not isinstance(impact_index, CodeImpactIndex):
        return ()
    try:
        result = impact_index.impact(
            changed_symbols=seed_symbols,
            changed_paths=seed_paths,
        )
    except Exception:
        return ()

    extras: list[tuple[str, str]] = []
    affected = set(result.affected_symbols) | set(result.affected_paths)
    seeds = set(seed_symbols) | set(seed_paths)
    for item in sorted(affected - seeds):
        matches = resolve_seed_nodes(graph, item)
        for node in matches:
            extras.append((node.node_id, f"impact-index:{item}"))
        if not matches:
            # Path-level hit: map path nodes.
            for node in graph.find_by_path(item):
                extras.append((node.node_id, f"impact-index-path:{item}"))
    return tuple(extras)


@dataclass(frozen=True)
class ImpactClosureDiagnostics:
    """Non-authoritative diagnostics retained alongside the receipt."""

    seed_node_ids: tuple[str, ...] = ()
    truncated: bool = False
    stale_graph: bool = False
    stale_index: bool = False
    unresolved_subject: bool = False
    graph_complete: bool = False
    traversed_edge_ids: tuple[str, ...] = ()
    reverse_adjacency: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    topo_order: tuple[str, ...] = ()
    bound_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_node_ids": list(self.seed_node_ids),
            "truncated": self.truncated,
            "stale_graph": self.stale_graph,
            "stale_index": self.stale_index,
            "unresolved_subject": self.unresolved_subject,
            "graph_complete": self.graph_complete,
            "traversed_edge_ids": list(self.traversed_edge_ids),
            "reverse_adjacency": {
                key: list(value) for key, value in sorted(self.reverse_adjacency.items())
            },
            "topo_order": list(self.topo_order),
            "bound_ref": self.bound_ref,
        }


@dataclass(frozen=True)
class ImpactClosureResult:
    """Receipt plus diagnostics for a single reverse impact computation."""

    receipt: ImpactClosureReceipt
    diagnostics: ImpactClosureDiagnostics

    @property
    def completeness(self) -> ImpactCompleteness:
        return self.receipt.completeness

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_CHANGE_IMPACT_SCHEMA,
            "contract_version": CHANGE_PROPAGATION_VERSION,
            "receipt": self.receipt.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }


class ContractChangeImpactAnalyzer:
    """Compute reverse transitive impact closures and SCCs for contract deltas."""

    def __init__(
        self,
        *,
        bounds: ImpactClosureBounds | Mapping[str, Any] | None = None,
        extractor_id: str = DEFAULT_EXTRACTOR_ID,
    ) -> None:
        self._bounds = _coerce_bounds(bounds)
        self._extractor_id = str(extractor_id or DEFAULT_EXTRACTOR_ID).strip()

    @property
    def bounds(self) -> ImpactClosureBounds:
        return self._bounds

    def analyze(
        self,
        delta: ProgramContractDelta | Mapping[str, Any],
        graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any],
        *,
        impact_index: Any = None,
        bounds: ImpactClosureBounds | Mapping[str, Any] | None = None,
    ) -> ImpactClosureResult:
        """Compute one root-bound reverse impact closure for ``delta``."""

        contract_delta = _coerce_delta(delta)
        program_graph = _coerce_graph(graph)
        limits = _coerce_bounds(bounds) if bounds is not None else self._bounds
        roots = contract_delta.roots

        stale_graph = bool(roots.graph_id) and roots.graph_id != program_graph.graph_id
        stale_index = False
        if impact_index is not None and CodeImpactIndex is not None:
            if isinstance(impact_index, CodeImpactIndex):
                if roots.index_id and impact_index.index_id and roots.index_id != impact_index.index_id:
                    stale_index = True
                tree_id = str(getattr(impact_index, "repository_tree_id", "") or "")
                if tree_id and roots.candidate_tree_id and tree_id not in {
                    roots.candidate_tree_id,
                    roots.base_tree_id,
                }:
                    # Index bound to an unrelated tree is not authoritative.
                    stale_index = True

        seeds = resolve_seed_nodes(program_graph, contract_delta.subject_symbol_id)
        unresolved_subject = not seeds

        if unresolved_subject or stale_graph:
            diagnostics = ImpactClosureDiagnostics(
                seed_node_ids=(),
                truncated=False,
                stale_graph=stale_graph,
                stale_index=stale_index,
                unresolved_subject=unresolved_subject,
                graph_complete=bool(program_graph.complete and not program_graph.frontier_refs),
                bound_ref=limits.bound_ref,
            )
            frontier_nodes: list[str] = []
            if unresolved_subject:
                frontier_nodes.append(
                    f"unresolved_subject:{contract_delta.subject_symbol_id}"
                )
            if stale_graph:
                frontier_nodes.append(
                    f"stale_graph:claimed={roots.graph_id}:actual={program_graph.graph_id}"
                )
            if stale_index:
                frontier_nodes.append("stale_index")
            frontier_nodes.extend(program_graph.frontier_refs)
            # Fail closed: no mandatory consumers when the route cannot start.
            receipt = ImpactClosureReceipt(
                roots=roots,
                delta_id=contract_delta.content_id,
                completeness=ImpactCompleteness.ABSTAINED,
                consumers=(),
                sccs=(),
                frontier_node_ids=tuple(sorted(set(frontier_nodes)))[:MAX_CONSUMER_COUNT],
                frontier_edge_ids=(),
                excluded_refs=tuple(program_graph.exclusion_refs)[:MAX_REFERENCE_COUNT],
                validation_refs=(),
                resource_bound_refs=(limits.bound_ref,),
                evidence_refs=(
                    f"graph:{program_graph.graph_id}",
                    f"delta:{contract_delta.content_id}",
                ),
            )
            return ImpactClosureResult(receipt=receipt, diagnostics=diagnostics)

        return self._traverse(
            roots=roots,
            delta=contract_delta,
            graph=program_graph,
            seeds=seeds,
            impact_index=None if stale_index else impact_index,
            limits=limits,
            stale_graph=stale_graph,
            stale_index=stale_index,
        )

    def _traverse(
        self,
        *,
        roots: PropagationAuthorityRoots,
        delta: ProgramContractDelta,
        graph: ProgramGraph,
        seeds: Sequence[ProgramNode],
        impact_index: Any,
        limits: ImpactClosureBounds,
        stale_graph: bool,
        stale_index: bool,
    ) -> ImpactClosureResult:
        seed_ids = tuple(sorted({node.node_id for node in seeds}))
        seed_symbols = []
        seed_paths = []
        for node in seeds:
            seed_symbols.extend(_node_symbol_keys(node))
            if node.path:
                seed_paths.append(node.path)
        seed_symbols = sorted(set(seed_symbols))
        seed_paths = sorted(set(seed_paths))

        consumers: dict[str, _ConsumerState] = {}
        reverse_adj: dict[str, set[str]] = {node_id: set() for node_id in seed_ids}
        traversed_edges: set[str] = set()
        frontier_node_ids: set[str] = set()
        frontier_edge_ids: set[str] = set()
        truncated = False
        edge_budget = 0

        # Seeds themselves are the impact origin; they are not consumers unless
        # re-entered via a reverse edge (self-cycle through another node).
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in seed_ids)
        visited_expand: set[str] = set()

        def note_frontier_node(node_id: str) -> None:
            frontier_node_ids.add(node_id)

        def note_frontier_edge(edge_id: str) -> None:
            frontier_edge_ids.add(edge_id)

        def ensure_consumer(
            node_id: str,
            *,
            depth: int,
            mandatory: bool,
            frontier: bool,
            edge_id: str | None = None,
            path_condition: str = "",
        ) -> None:
            nonlocal truncated
            state = consumers.get(node_id)
            if state is None:
                if len(consumers) >= limits.max_consumers:
                    truncated = True
                    note_frontier_node(f"truncated:max_consumers:{node_id}")
                    return
                state = _ConsumerState(
                    node_id=node_id,
                    depth=depth,
                    mandatory=mandatory and not frontier,
                    frontier=frontier,
                )
                consumers[node_id] = state
                reverse_adj.setdefault(node_id, set())
            else:
                if depth < state.depth:
                    state.depth = depth
                if mandatory and not frontier:
                    state.mandatory = True
                if frontier:
                    state.frontier = True
            if edge_id:
                state.edge_refs.add(edge_id)
            if path_condition:
                state.path_condition_refs.add(path_condition)

        # Seed graph-level frontiers / exclusions into the unknown set.
        for ref in graph.frontier_refs:
            frontier_node_ids.add(ref)
        if not graph.complete:
            frontier_node_ids.add("graph_incomplete")

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited_expand:
                continue
            visited_expand.add(current_id)

            if depth >= limits.max_depth:
                # Neighbors beyond the depth bound become an explicit frontier.
                neighbors = _reverse_neighbors(graph, current_id)
                if neighbors:
                    truncated = True
                    note_frontier_node(f"truncated:max_depth:{current_id}")
                    for edge, neighbor_id in neighbors:
                        note_frontier_edge(edge.edge_id)
                continue

            for edge, neighbor_id in _reverse_neighbors(graph, current_id):
                if edge_budget >= limits.max_edges:
                    truncated = True
                    note_frontier_edge(f"truncated:max_edges:{edge.edge_id}")
                    note_frontier_node(f"truncated:max_edges:{neighbor_id}")
                    continue
                edge_budget += 1
                traversed_edges.add(edge.edge_id)
                reverse_adj.setdefault(current_id, set()).add(neighbor_id)
                reverse_adj.setdefault(neighbor_id, set())

                neighbor = graph.node(neighbor_id)
                if neighbor is None:
                    note_frontier_node(f"missing_node:{neighbor_id}")
                    note_frontier_edge(edge.edge_id)
                    continue

                nominated = _edge_is_nominated(edge)
                node_frontier = _node_is_frontier(neighbor)

                # Nominated GraphRAG/vector/runtime edges never expand coverage.
                if nominated:
                    note_frontier_edge(edge.edge_id)
                    note_frontier_node(neighbor_id)
                    ensure_consumer(
                        neighbor_id,
                        depth=depth + 1,
                        mandatory=False,
                        frontier=True,
                        edge_id=edge.edge_id,
                    )
                    continue

                if edge.completeness in {
                    Completeness.FRONTIER,
                    Completeness.UNSUPPORTED,
                    Completeness.UNKNOWN,
                    Completeness.PARTIAL,
                }:
                    # Partial but authoritative edges still expand, while also
                    # recording the incompleteness as a frontier signal.
                    if edge.completeness is not Completeness.PARTIAL:
                        note_frontier_edge(edge.edge_id)
                        note_frontier_node(neighbor_id)

                path_condition = ""
                if edge.kind is ProgramEdgeKind.PATH_CONDITION:
                    path_condition = edge.edge_id
                elif "path_condition" in edge.attributes:
                    path_condition = str(edge.attributes.get("path_condition") or "")

                ensure_consumer(
                    neighbor_id,
                    depth=depth + 1,
                    mandatory=edge.authoritative and neighbor.authoritative and not node_frontier,
                    frontier=node_frontier,
                    edge_id=edge.edge_id,
                    path_condition=path_condition,
                )
                if truncated and neighbor_id not in consumers:
                    continue
                if node_frontier:
                    note_frontier_node(neighbor_id)
                if neighbor_id not in visited_expand and not nominated:
                    queue.append((neighbor_id, depth + 1))

        # Reviewed CodeImpactIndex reverse edges (never nominated channels).
        if impact_index is not None and not stale_index:
            for node_id, via in _impact_index_extra_consumers(
                impact_index, seed_symbols, seed_paths, graph
            ):
                if node_id in seed_ids:
                    continue
                node = graph.node(node_id)
                if node is None:
                    frontier_node_ids.add(f"impact_index_unresolved:{via}")
                    continue
                for seed_id in seed_ids:
                    reverse_adj.setdefault(seed_id, set()).add(node_id)
                ensure_consumer(
                    node_id,
                    depth=1,
                    mandatory=node.authoritative and not _node_is_frontier(node),
                    frontier=_node_is_frontier(node),
                    edge_id=via,
                )
                if node_id not in visited_expand and not _node_is_frontier(node):
                    queue.append((node_id, 1))

            # Drain any index-enqueued work under the same rules.
            while queue:
                current_id, depth = queue.popleft()
                if current_id in visited_expand:
                    continue
                visited_expand.add(current_id)
                if depth >= limits.max_depth:
                    truncated = True
                    note_frontier_node(f"truncated:max_depth:{current_id}")
                    continue
                for edge, neighbor_id in _reverse_neighbors(graph, current_id):
                    if edge_budget >= limits.max_edges:
                        truncated = True
                        break
                    edge_budget += 1
                    traversed_edges.add(edge.edge_id)
                    reverse_adj.setdefault(current_id, set()).add(neighbor_id)
                    nominated = _edge_is_nominated(edge)
                    neighbor = graph.node(neighbor_id)
                    if neighbor is None or nominated:
                        if neighbor_id:
                            note_frontier_node(neighbor_id)
                        note_frontier_edge(edge.edge_id)
                        continue
                    ensure_consumer(
                        neighbor_id,
                        depth=depth + 1,
                        mandatory=edge.authoritative and neighbor.authoritative,
                        frontier=_node_is_frontier(neighbor),
                        edge_id=edge.edge_id,
                    )
                    if neighbor_id not in visited_expand:
                        queue.append((neighbor_id, depth + 1))

        # Build deterministic ImpactConsumer records.
        impact_consumers: list[ImpactConsumer] = []
        consumer_node_ids: list[str] = []
        for node_id in sorted(consumers):
            state = consumers[node_id]
            node = graph.node(node_id)
            if node is None:
                frontier_node_ids.add(f"missing_consumer:{node_id}")
                continue
            # Seeds that only appear as path intermediates without being
            # dependents are already excluded (they never enter consumers).
            ref = _node_ref(node, frontier=state.frontier or not state.mandatory)
            path_condition_ref = ""
            if state.path_condition_refs:
                path_condition_ref = sorted(state.path_condition_refs)[0]
            impact_consumers.append(
                ImpactConsumer(
                    consumer_id=_consumer_id_for(node_id),
                    node=ref,
                    depth=state.depth,
                    mandatory=bool(state.mandatory and not state.frontier),
                    edge_refs=tuple(sorted(state.edge_refs))[:MAX_REFERENCE_COUNT],
                    path_condition_ref=path_condition_ref,
                )
            )
            consumer_node_ids.append(node_id)
            if state.frontier or not state.mandatory:
                frontier_node_ids.add(node_id)

        # SCC over reverse-impact adjacency restricted to consumer nodes.
        # Include seed→consumer edges so cycles through the subject surface.
        scc_nodes = list(consumer_node_ids)
        scc_adj: dict[str, list[str]] = {node_id: [] for node_id in scc_nodes}
        consumer_set = set(consumer_node_ids)
        for source, targets in reverse_adj.items():
            if source not in consumer_set:
                # Edges from seeds into consumers are recorded on the consumer
                # only; seeds are not SCC members unless they are also consumers.
                continue
            scc_adj[source] = sorted(target for target in targets if target in consumer_set)

        # Also connect mutual reverse edges so A→B and B→A form an SCC even
        # when only consumers are present.
        scc_groups, topo_order = compute_sccs(scc_nodes, scc_adj)
        if len(scc_groups) > limits.max_sccs:
            truncated = True
            frontier_node_ids.add("truncated:max_sccs")
            scc_groups = scc_groups[: limits.max_sccs]

        impact_sccs = tuple(
            ImpactSCC(
                scc_id=_scc_id_for(members),
                member_consumer_ids=tuple(_consumer_id_for(member) for member in members),
            )
            for members in scc_groups
            if members
        )

        validation_refs: list[str] = []
        # Validation consumers discovered in the graph.
        for consumer in impact_consumers:
            node = graph.node(consumer.node.node_id)
            if node is not None and node.kind is ProgramNodeKind.VALIDATION:
                validation_refs.append(f"validation:{node.node_id}")
            if node is not None and node.kind is ProgramNodeKind.TEST:
                validation_refs.append(f"test:{node.node_id}")
        if impact_index is not None and not stale_index:
            validation_refs.extend(
                _impact_index_validation_refs(impact_index, seed_symbols, seed_paths)
            )
        # Stable unique validation refs.
        validation_refs = sorted(set(validation_refs))[:MAX_REFERENCE_COUNT]

        excluded_refs = list(graph.exclusion_refs)
        for root in graph.roots.excluded_roots:
            ref = f"excluded_root:{root}"
            if ref not in excluded_refs:
                excluded_refs.append(ref)
        excluded_refs = sorted(set(excluded_refs))[:MAX_REFERENCE_COUNT]

        graph_open = bool(graph.frontier_refs) or not graph.complete
        open_frontier = bool(frontier_node_ids or frontier_edge_ids)
        if stale_graph or stale_index:
            completeness = ImpactCompleteness.ABSTAINED
            # Authority failure: strip mandatory flags by rebuilding without them.
            if any(item.mandatory for item in impact_consumers):
                impact_consumers = [
                    ImpactConsumer(
                        consumer_id=item.consumer_id,
                        node=item.node,
                        depth=item.depth,
                        mandatory=False,
                        edge_refs=item.edge_refs,
                        path_condition_ref=item.path_condition_ref,
                    )
                    for item in impact_consumers
                ]
        elif truncated or open_frontier or graph_open:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            if not frontier_node_ids and not frontier_edge_ids:
                # Contract requires an explicit frontier for partial receipts.
                frontier_node_ids.add(
                    "partial_coverage"
                    if not truncated
                    else "truncated"
                )
        else:
            # Complete only when every consumer is mandatory, no frontier, no
            # truncation, and the underlying graph claims completeness.
            if any(not item.mandatory for item in impact_consumers):
                completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
                for item in impact_consumers:
                    if not item.mandatory:
                        frontier_node_ids.add(item.node.node_id)
                if not frontier_node_ids:
                    frontier_node_ids.add("non_mandatory_consumer")
            else:
                completeness = ImpactCompleteness.COMPLETE

        # Final hard rule: COMPLETE cannot retain a frontier.
        if completeness is ImpactCompleteness.COMPLETE and (
            frontier_node_ids or frontier_edge_ids
        ):
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
        if completeness is ImpactCompleteness.COMPLETE and truncated:
            completeness = ImpactCompleteness.PARTIAL_WITH_FRONTIER
            frontier_node_ids.add("truncated")

        # Sort consumers by content_id via ImpactConsumer ordering (node/depth).
        impact_consumers_sorted = tuple(
            sorted(impact_consumers, key=lambda item: (item.depth, item.consumer_id))
        )

        evidence_refs = sorted(
            {
                f"graph:{graph.graph_id}",
                f"delta:{delta.content_id}",
                f"extractor:{self._extractor_id}",
                f"subject:{delta.subject_symbol_id}",
                *[f"seed:{node_id}" for node_id in seed_ids],
            }
        )[:MAX_REFERENCE_COUNT]

        receipt = ImpactClosureReceipt(
            roots=roots,
            delta_id=delta.content_id,
            completeness=completeness,
            consumers=impact_consumers_sorted,
            sccs=impact_sccs,
            frontier_node_ids=tuple(sorted(frontier_node_ids))[:MAX_CONSUMER_COUNT],
            frontier_edge_ids=tuple(sorted(frontier_edge_ids))[:MAX_CONSUMER_COUNT],
            excluded_refs=tuple(excluded_refs),
            validation_refs=tuple(validation_refs),
            resource_bound_refs=(limits.bound_ref,),
            evidence_refs=tuple(evidence_refs),
        )

        diagnostics = ImpactClosureDiagnostics(
            seed_node_ids=seed_ids,
            truncated=truncated,
            stale_graph=stale_graph,
            stale_index=stale_index,
            unresolved_subject=False,
            graph_complete=bool(graph.complete and not graph.frontier_refs),
            traversed_edge_ids=tuple(sorted(traversed_edges)),
            reverse_adjacency={
                key: tuple(sorted(value)) for key, value in sorted(reverse_adj.items())
            },
            topo_order=topo_order,
            bound_ref=limits.bound_ref,
        )
        return ImpactClosureResult(receipt=receipt, diagnostics=diagnostics)


def compute_impact_closure(
    delta: ProgramContractDelta | Mapping[str, Any],
    graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any],
    *,
    impact_index: Any = None,
    bounds: ImpactClosureBounds | Mapping[str, Any] | None = None,
) -> ImpactClosureReceipt:
    """Module-level helper returning the canonical ImpactClosureReceipt@1."""

    result = ContractChangeImpactAnalyzer(bounds=bounds).analyze(
        delta,
        graph,
        impact_index=impact_index,
        bounds=bounds,
    )
    return result.receipt


def compute_impact_closure_result(
    delta: ProgramContractDelta | Mapping[str, Any],
    graph: ProgramGraph | ProgramGraphSnapshot | Mapping[str, Any],
    *,
    impact_index: Any = None,
    bounds: ImpactClosureBounds | Mapping[str, Any] | None = None,
) -> ImpactClosureResult:
    """Module-level helper returning receipt + diagnostics."""

    return ContractChangeImpactAnalyzer(bounds=bounds).analyze(
        delta,
        graph,
        impact_index=impact_index,
        bounds=bounds,
    )


# Public aliases matching the task AST surface.
ContractChangeImpact = ContractChangeImpactAnalyzer


__all__ = [
    "CONTRACT_CHANGE_IMPACT_SCHEMA",
    "CONTRACT_CHANGE_IMPACT_VERSION",
    "DEFAULT_MAX_CONSUMERS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_SCCS",
    "ContractChangeImpact",
    "ContractChangeImpactAnalyzer",
    "ContractChangeImpactAuthorityError",
    "ContractChangeImpactBoundsError",
    "ContractChangeImpactError",
    "ImpactClosureBounds",
    "ImpactClosureDiagnostics",
    "ImpactClosureResult",
    "ImpactTraversalStep",
    "compute_impact_closure",
    "compute_impact_closure_result",
    "compute_sccs",
    "resolve_seed_nodes",
    # Re-export canonical receipt types so consumers import one module.
    "ImpactClosureReceipt",
    "ImpactConsumer",
    "ImpactSCC",
    "ImpactCompleteness",
]
