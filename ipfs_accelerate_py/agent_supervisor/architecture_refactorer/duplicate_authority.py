"""Duplicate-authority detection for competing production authorities (PCAR-007).

`DuplicateAuthorityDetector` observes ArchitectureIR and an accepted ownership
graph. It emits hard findings for independent provider-capability and receipt
decisions, competing state owners, compatibility or control bypasses,
simulation-to-production flow, Python/CLI/MCP divergence, re-export
authorities, and tests that validate obsolete rather than canonical behavior.
Formally arbitrated competitors and classified adapters, projections, legacy
paths, and quarantined simulations are not collisions. Unknown production
ownership is a typed blocker. Heuristic or opaque evidence stays unknown and
is never promoted to a critical finding. The detector cannot select a
canonical owner, authorize a change, or execute remediation.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .architecture_ir import ArchitectureEdge, ArchitectureIR, ArchitectureNode
from .authority_graph import (
    AuthorityOwnershipGraph,
    ConcernKind,
    FormalArbitration,
    OwnerDisposition,
    OwnershipBlockerKind,
)
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    NON_PROBATIVE_CONFIDENCE,
)

DUPLICATE_AUTHORITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/duplicate-authority-finding@1"
)
DUPLICATE_AUTHORITY_VERSION = 1
DUPLICATE_AUTHORITY_EVIDENCE = "pcar/duplicate-authority-finding@1"
COLLISION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/authority-collision@1"
COLLISION_VERSION = 1
BYPASS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/authority-bypass-finding@1"
BYPASS_VERSION = 1
SURFACE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/surface-divergence-finding@1"
)
SURFACE_VERSION = 1
BLOCKER_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/duplicate-authority-blocker@1"
)
BLOCKER_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-007-duplicate-authority-detector"
TASK_ID = "PCAR-007"
DEFAULT_FRESHNESS = "pcar-007-duplicate-authority"
EFFECT_CLASS = "read_only_analysis"
DETECTOR_CAN_AUTHORIZE_CHANGES = False
DETECTOR_CAN_REMEDIATE = False
DETECTOR_CAN_SELECT_CANONICAL = False
CONTENT_IDENTITY_IS_NOT_AUTHORITY = True
REEXPORT_IS_NOT_AUTHORITY = True
SILENT_ARBITRATION_PROHIBITED = True
ONE_OWNER_INVARIANT = True
HEURISTIC_CRITICAL_PROMOTION_PROHIBITED = True

_UNKNOWN_FIELD_MESSAGE = "unknown duplicate-authority field"
_MISSING_FIELD_MESSAGE = "missing duplicate-authority field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_CONFIDENCE_RANK = {
    Confidence.EXACT: 0,
    Confidence.CONSERVATIVE: 1,
    Confidence.HEURISTIC: 2,
    Confidence.OPAQUE: 3,
}
_PROVIDER_DECISION_EDGES = frozenset(
    {EdgeKind.AUTHORIZES, EdgeKind.EVALUATES_POLICY, EdgeKind.EXECUTES}
)
_RECEIPT_DECISION_EDGES = frozenset(
    {EdgeKind.AUTHORIZES, EdgeKind.CONFIRMS, EdgeKind.PROVES}
)
_STATE_OWNER_EDGES = frozenset(
    {EdgeKind.PERSISTS, EdgeKind.WRITES, EdgeKind.MUTATES}
)
_BYPASS_EDGES = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.EXECUTES,
        EdgeKind.CONFIRMS,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.PERSISTS,
        EdgeKind.PROVES,
        EdgeKind.CALLS,
    }
)
_CONTROL_BYPASS_EDGES = frozenset(
    {EdgeKind.AUTHORIZES, EdgeKind.EXECUTES, EdgeKind.CONFIRMS}
)
_FLOW_EDGES = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.EXECUTES,
        EdgeKind.CONFIRMS,
        EdgeKind.AUTHORIZES,
        EdgeKind.PERSISTS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.PROVES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.FALLBACKS_TO,
    }
)
_EXPLAINING_TO_CANONICAL = frozenset({EdgeKind.ADAPTS, EdgeKind.REEXPORTS})
_EXPLAINING_FROM_CANONICAL = frozenset(
    {
        EdgeKind.SUPERSEDES,
        EdgeKind.DEPRECATES,
        EdgeKind.FALLBACKS_TO,
        EdgeKind.GENERATES,
        EdgeKind.ADAPTS,
    }
)
_PRODUCTION_NODE_KINDS = frozenset(
    {
        NodeKind.AUTHORITY,
        NodeKind.POLICY,
        NodeKind.STATE,
        NodeKind.RECEIPT,
        NodeKind.PROOF,
        NodeKind.PROVIDER,
    }
)
_NON_CANONICAL_OWNER_DISPOSITIONS = frozenset(
    {
        OwnerDisposition.ADAPTER,
        OwnerDisposition.PROJECTION,
        OwnerDisposition.LEGACY,
        OwnerDisposition.SIMULATION,
    }
)
_OWNERSHIP_UNKNOWN_BLOCKERS = frozenset(
    {
        OwnershipBlockerKind.UNKNOWN_OWNER,
        OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER,
    }
)
_CLI_MARKERS = ("/cli.py", "/cli/", "entrypoints/cli", "_cli.py", "/cli_")
_MCP_MARKERS = ("/mcp", "mcp_", "mcp.py", "mcp-server", "mcp_server")


class DuplicateAuthorityError(ArchitectureContractError):
    """Fail-closed duplicate-authority contract violation."""


class DuplicateAuthorityAuthorityError(DuplicateAuthorityError):
    """Raised when the detector is asked to remediate or select an owner."""


class CollisionKind(str, Enum):
    """Closed required duplicate-authority detection vocabulary."""

    INDEPENDENT_PROVIDER_CAPABILITY = "independent_provider_capability"
    INDEPENDENT_RECEIPT_DECISION = "independent_receipt_decision"
    COMPETING_STATE_OWNER = "competing_state_owner"
    COMPATIBILITY_BYPASS = "compatibility_bypass"
    CONTROL_BYPASS = "control_bypass"
    SIMULATION_TO_PRODUCTION_FLOW = "simulation_to_production_flow"
    PYTHON_CLI_MCP_DIVERGENCE = "python_cli_mcp_divergence"
    REEXPORT_AUTHORITY = "reexport_authority"
    OBSOLETE_AUTHORITY_TEST = "obsolete_authority_test"


REQUIRED_COLLISION_KINDS: tuple[CollisionKind, ...] = tuple(CollisionKind)
CLOSED_COLLISION_KINDS: frozenset[str] = frozenset(
    item.value for item in CollisionKind
)


class CollisionDisposition(str, Enum):
    """Closed disposition vocabulary for one candidate finding."""

    COLLISION = "collision"
    FORMALLY_ARBITRATED = "formally_arbitrated"
    FALSE_POSITIVE = "false_positive"
    UNKNOWN = "unknown"


CLOSED_COLLISION_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in CollisionDisposition
)


class DuplicateAuthorityBlockerKind(str, Enum):
    """Closed hard-blocker vocabulary emitted by the detector."""

    UNKNOWN_PRODUCTION_OWNER = "unknown_production_owner"
    UNKNOWN_OWNER = "unknown_owner"
    MULTIPLE_PRODUCTION_AUTHORITIES = "multiple_production_authorities"
    MISSING_ARBITRATION = "missing_arbitration"


CLOSED_DUPLICATE_AUTHORITY_BLOCKERS: frozenset[str] = frozenset(
    item.value for item in DuplicateAuthorityBlockerKind
)


class SurfaceKind(str, Enum):
    """Typed Python/CLI/MCP projections compared for authority parity."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


CLOSED_SURFACES: frozenset[str] = frozenset(item.value for item in SurfaceKind)
REQUIRED_SURFACES: tuple[SurfaceKind, ...] = tuple(SurfaceKind)

_KIND_CONCERN = {
    CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY: ConcernKind.PROVIDER_CAPABILITY,
    CollisionKind.INDEPENDENT_RECEIPT_DECISION: ConcernKind.COMPLETION_EVIDENCE,
    CollisionKind.COMPETING_STATE_OWNER: ConcernKind.STATE_PERSISTENCE,
    CollisionKind.COMPATIBILITY_BYPASS: ConcernKind.AUTHORIZATION,
    CollisionKind.CONTROL_BYPASS: ConcernKind.POLICY_DECISION,
    CollisionKind.SIMULATION_TO_PRODUCTION_FLOW: ConcernKind.PROVIDER_SELECTION,
    CollisionKind.PYTHON_CLI_MCP_DIVERGENCE: ConcernKind.OPERATION_IDENTITY,
    CollisionKind.REEXPORT_AUTHORITY: ConcernKind.OPERATION_IDENTITY,
    CollisionKind.OBSOLETE_AUTHORITY_TEST: ConcernKind.TEST_EVIDENCE,
}

_COLLISION_FIELDS = frozenset(
    {
        "concern",
        "confidence",
        "content_identity",
        "disposition",
        "edge_ids",
        "kind",
        "message",
        "node_ids",
        "provenance",
        "reachability",
        "schema",
        "version",
    }
)
_BYPASS_FIELDS = frozenset(
    {
        "bypass_node_id",
        "content_identity",
        "disposition",
        "edge_ids",
        "kind",
        "message",
        "production_node_id",
        "provenance",
        "reachability",
        "schema",
        "version",
    }
)
_SURFACE_FIELDS = frozenset(
    {
        "content_identity",
        "disposition",
        "edge_ids",
        "kind",
        "message",
        "missing_surfaces",
        "node_ids",
        "operation_node_id",
        "present_surfaces",
        "provenance",
        "schema",
        "version",
    }
)
_BLOCKER_FIELDS = frozenset(
    {
        "concern",
        "content_identity",
        "edge_ids",
        "kind",
        "message",
        "node_ids",
        "schema",
        "version",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "blockers",
        "bypasses",
        "can_authorize_changes",
        "can_remediate",
        "can_select_canonical",
        "collisions",
        "content_identity",
        "freshness",
        "one_owner_invariant",
        "ownership_graph_identity",
        "rejected",
        "repository_tree",
        "schema",
        "surface_divergences",
        "unknown",
        "version",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise DuplicateAuthorityError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise DuplicateAuthorityError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise DuplicateAuthorityError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DuplicateAuthorityError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=DuplicateAuthorityError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DuplicateAuthorityError(f"{name} must be a boolean")
    return value


def _require_architecture_ir(
    graph: ArchitectureIR | Mapping[str, Any],
) -> ArchitectureIR:
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise DuplicateAuthorityError(str(exc)) from exc


def _require_ownership_graph(
    graph: AuthorityOwnershipGraph | Mapping[str, Any] | None,
) -> AuthorityOwnershipGraph | None:
    if graph is None:
        return None
    if isinstance(graph, AuthorityOwnershipGraph):
        return graph
    try:
        return AuthorityOwnershipGraph.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise DuplicateAuthorityError(str(exc)) from exc


def _wrap_contract(exc: ArchitectureContractError) -> DuplicateAuthorityError:
    if isinstance(exc, DuplicateAuthorityError):
        return exc
    return DuplicateAuthorityError(str(exc))


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _worst_confidence(values: Iterable[Confidence]) -> Confidence:
    ranked = list(values)
    if not ranked:
        return Confidence.EXACT
    return max(ranked, key=lambda item: _CONFIDENCE_RANK[item])


@dataclass(frozen=True)
class _GraphView:
    architecture: ArchitectureIR
    nodes_by_id: dict[str, ArchitectureNode]
    edges_by_id: dict[str, ArchitectureEdge]
    outgoing: dict[str, tuple[ArchitectureEdge, ...]]
    incoming: dict[str, tuple[ArchitectureEdge, ...]]


def _build_view(architecture: ArchitectureIR) -> _GraphView:
    outgoing: dict[str, list[ArchitectureEdge]] = {
        node.node_id: [] for node in architecture.nodes
    }
    incoming: dict[str, list[ArchitectureEdge]] = {
        node.node_id: [] for node in architecture.nodes
    }
    for edge in architecture.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    return _GraphView(
        architecture=architecture,
        nodes_by_id={node.node_id: node for node in architecture.nodes},
        edges_by_id={edge.edge_id: edge for edge in architecture.edges},
        outgoing={key: tuple(value) for key, value in outgoing.items()},
        incoming={key: tuple(value) for key, value in incoming.items()},
    )


def _related_edges(view: _GraphView, node_id: str) -> tuple[ArchitectureEdge, ...]:
    return view.outgoing.get(node_id, ()) + view.incoming.get(node_id, ())


@dataclass(frozen=True)
class _OwnershipIndex:
    graph: AuthorityOwnershipGraph | None
    disposition_by_node: dict[str, OwnerDisposition]
    canonical_by_concern: dict[ConcernKind, str]
    arbitration_by_concern: dict[ConcernKind, FormalArbitration]
    classified_nodes: frozenset[str]


def _index_ownership(
    ownership: AuthorityOwnershipGraph | None,
    extra_arbitrations: Sequence[FormalArbitration] = (),
) -> _OwnershipIndex:
    disposition_by_node: dict[str, OwnerDisposition] = {}
    canonical_by_concern: dict[ConcernKind, str] = {}
    arbitration_by_concern: dict[ConcernKind, FormalArbitration] = {}
    classified: set[str] = set()
    if ownership is not None:
        for record in ownership.concerns:
            if record.canonical_owner is not None:
                disposition_by_node[record.canonical_owner.node_id] = (
                    OwnerDisposition.CANONICAL
                )
                canonical_by_concern[record.concern] = record.canonical_owner.node_id
                classified.add(record.canonical_owner.node_id)
            for owner, disposition in (
                *((item, OwnerDisposition.ADAPTER) for item in record.adapters),
                *((item, OwnerDisposition.PROJECTION) for item in record.projections),
                *((item, OwnerDisposition.LEGACY) for item in record.legacy_owners),
                *(
                    (item, OwnerDisposition.SIMULATION)
                    for item in record.simulation_owners
                ),
                *((item, OwnerDisposition.UNKNOWN) for item in record.unknown_owners),
            ):
                disposition_by_node.setdefault(owner.node_id, disposition)
                classified.add(owner.node_id)
            if record.arbitration is not None:
                arbitration_by_concern[record.concern] = record.arbitration
    for record in extra_arbitrations:
        if record.concern in arbitration_by_concern:
            raise DuplicateAuthorityError(
                f"duplicate arbitration for {record.concern.value}"
            )
        arbitration_by_concern[record.concern] = record
    return _OwnershipIndex(
        graph=ownership,
        disposition_by_node=disposition_by_node,
        canonical_by_concern=canonical_by_concern,
        arbitration_by_concern=arbitration_by_concern,
        classified_nodes=frozenset(classified),
    )


def _normalize_arbitrations(
    records: Sequence[FormalArbitration | Mapping[str, Any]] | None,
) -> tuple[FormalArbitration, ...]:
    if records is None:
        return ()
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise DuplicateAuthorityError("arbitrations must be a sequence")
    return tuple(
        item
        if isinstance(item, FormalArbitration)
        else FormalArbitration.from_mapping(item)
        for item in records
    )


def _node_disposition(
    view: _GraphView,
    index: _OwnershipIndex,
    node_id: str,
) -> OwnerDisposition | None:
    known = index.disposition_by_node.get(node_id)
    if known is not None:
        return known
    node = view.nodes_by_id.get(node_id)
    if node is None:
        return None
    if node.kind is NodeKind.SIMULATION:
        return OwnerDisposition.SIMULATION
    if node.kind is NodeKind.GENERATED:
        return OwnerDisposition.PROJECTION
    if node.kind is NodeKind.COMPATIBILITY:
        return OwnerDisposition.ADAPTER
    if node.kind is NodeKind.AUTHORITY:
        return OwnerDisposition.CANONICAL
    return None


def _has_explaining_edge(
    view: _GraphView, left: str, right: str
) -> bool:
    for edge in view.outgoing.get(left, ()):
        if edge.target != right:
            continue
        if edge.kind in _EXPLAINING_TO_CANONICAL | _EXPLAINING_FROM_CANONICAL:
            return True
    for edge in view.outgoing.get(right, ()):
        if edge.target != left:
            continue
        if edge.kind in _EXPLAINING_TO_CANONICAL | _EXPLAINING_FROM_CANONICAL:
            return True
    return False


def _competitor_ids(
    view: _GraphView, node_ids: Sequence[str]
) -> tuple[str, ...]:
    kinds = {
        NodeKind.AUTHORITY,
        NodeKind.PROVIDER,
        NodeKind.STATE,
        NodeKind.RECEIPT,
        NodeKind.POLICY,
        NodeKind.ENTRYPOINT,
        NodeKind.TEST,
        NodeKind.COMPATIBILITY,
        NodeKind.SIMULATION,
        NodeKind.INTERFACE,
        NodeKind.MODULE,
    }
    return tuple(
        node_id
        for node_id in node_ids
        if view.nodes_by_id.get(node_id) is not None
        and view.nodes_by_id[node_id].kind in kinds
    )


def _unexplained_production(
    view: _GraphView,
    index: _OwnershipIndex,
    node_ids: Sequence[str],
) -> tuple[str, ...]:
    production: list[str] = []
    for node_id in node_ids:
        disposition = _node_disposition(view, index, node_id)
        if disposition in _NON_CANONICAL_OWNER_DISPOSITIONS:
            continue
        node = view.nodes_by_id.get(node_id)
        if node is not None and node.kind in {
            NodeKind.COMPATIBILITY,
            NodeKind.GENERATED,
            NodeKind.SIMULATION,
        }:
            continue
        production.append(node_id)
    unexplained: list[str] = []
    for node_id in production:
        others = [other for other in production if other != node_id]
        if others and all(
            _has_explaining_edge(view, node_id, other) for other in others
        ):
            continue
        unexplained.append(node_id)
    return tuple(sorted(set(unexplained)))


def _formally_arbitrated(
    index: _OwnershipIndex,
    concern: ConcernKind,
    node_ids: Sequence[str],
) -> bool:
    arbitration = index.arbitration_by_concern.get(concern)
    if arbitration is None:
        return False
    present = set(node_ids)
    winner = arbitration.canonical_owner_node_id
    if winner not in present:
        return False
    covered = {winner} | set(arbitration.loser_ids())
    return present <= covered


def _bypass_is_classified_adapter(
    view: _GraphView, index: _OwnershipIndex, node_ids: Sequence[str]
) -> bool:
    if len(node_ids) < 2:
        return False
    bypass_id, production_id = node_ids[0], node_ids[1]
    if _adapter_of(view, bypass_id, production_id):
        return True
    if any(
        _adapter_of(view, bypass_id, node.node_id) and node.kind is NodeKind.AUTHORITY
        for node in view.architecture.nodes
    ):
        return True
    known = index.disposition_by_node.get(bypass_id)
    return known in _NON_CANONICAL_OWNER_DISPOSITIONS


def _classify_disposition(
    view: _GraphView,
    index: _OwnershipIndex,
    kind: CollisionKind,
    node_ids: Sequence[str],
    confidence: Confidence,
) -> CollisionDisposition:
    if confidence in NON_PROBATIVE_CONFIDENCE:
        return CollisionDisposition.UNKNOWN
    competitors = _competitor_ids(view, node_ids) or tuple(node_ids)
    concern = _KIND_CONCERN[kind]
    if _formally_arbitrated(index, concern, competitors) or _formally_arbitrated(
        index, concern, node_ids
    ):
        return CollisionDisposition.FORMALLY_ARBITRATED
    if kind in {
        CollisionKind.COMPATIBILITY_BYPASS,
        CollisionKind.CONTROL_BYPASS,
    }:
        if _bypass_is_classified_adapter(view, index, node_ids):
            return CollisionDisposition.FALSE_POSITIVE
        return CollisionDisposition.COLLISION
    if kind in {
        CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
        CollisionKind.REEXPORT_AUTHORITY,
        CollisionKind.OBSOLETE_AUTHORITY_TEST,
        CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
    }:
        return CollisionDisposition.COLLISION
    unexplained = _unexplained_production(view, index, competitors)
    if len(unexplained) <= 1:
        return CollisionDisposition.FALSE_POSITIVE
    return CollisionDisposition.COLLISION


def _facts_for(
    view: _GraphView,
    node_ids: Iterable[str],
    edge_ids: Iterable[str],
) -> tuple[SourceFactIdentity, ...]:
    facts: list[SourceFactIdentity] = []
    for node_id in node_ids:
        node = view.nodes_by_id.get(node_id)
        if node is not None:
            facts.append(node.provenance)
    for edge_id in edge_ids:
        edge = view.edges_by_id.get(edge_id)
        if edge is not None:
            facts.append(edge.provenance)
    return tuple(facts)


def _provenance_for(
    view: _GraphView,
    node_ids: Sequence[str],
    edge_ids: Sequence[str],
) -> SourceFactIdentity:
    for edge_id in edge_ids:
        edge = view.edges_by_id.get(edge_id)
        if edge is not None:
            return edge.provenance
    for node_id in node_ids:
        node = view.nodes_by_id.get(node_id)
        if node is not None:
            return node.provenance
    raise DuplicateAuthorityError("finding is missing reachability provenance")


@dataclass(frozen=True)
class AuthorityCollision:
    """One competing-authority finding bound to source reachability."""

    kind: CollisionKind
    concern: ConcernKind
    disposition: CollisionDisposition
    message: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...] = ()
    reachability: tuple[str, ...] = ()
    provenance: SourceFactIdentity | None = None
    confidence: Confidence = Confidence.EXACT
    schema: str = COLLISION_SCHEMA
    version: int = COLLISION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != COLLISION_SCHEMA:
            raise DuplicateAuthorityError("unexpected authority-collision schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != COLLISION_VERSION:
            raise DuplicateAuthorityError("unexpected authority-collision version")
        kind = _closed_enum(
            self.kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        concern = _closed_enum(
            self.concern, ConcernKind, "concern", error_type=DuplicateAuthorityError
        )
        disposition = _closed_enum(
            self.disposition,
            CollisionDisposition,
            "collision disposition",
            error_type=DuplicateAuthorityError,
        )
        message = _require_text(self.message, "message", error_type=DuplicateAuthorityError)
        node_ids = _require_text_tuple(self.node_ids, "node_ids")
        if len(node_ids) < 1:
            raise DuplicateAuthorityError("collision node_ids must be nonempty")
        if any(_looks_like_content_identity(item) for item in node_ids):
            raise DuplicateAuthorityError(
                "content identity is not inferred to be authority"
            )
        edge_ids = _require_text_tuple(self.edge_ids, "edge_ids")
        reachability = _require_text_tuple(self.reachability, "reachability")
        if self.provenance is None:
            raise DuplicateAuthorityError("collision provenance is required")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        confidence = _closed_enum(
            self.confidence, Confidence, "confidence", error_type=DuplicateAuthorityError
        )
        if (
            disposition is CollisionDisposition.COLLISION
            and confidence in NON_PROBATIVE_CONFIDENCE
        ):
            raise DuplicateAuthorityError(
                "heuristic or opaque facts cannot prove a critical collision"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "reachability", reachability)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "confidence", confidence)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
            if claimed != identity:
                raise DuplicateAuthorityError("collision content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "confidence": self.confidence.value,
            "disposition": self.disposition.value,
            "edge_ids": list(self.edge_ids),
            "kind": self.kind.value,
            "message": self.message,
            "node_ids": list(self.node_ids),
            "provenance": self.provenance.to_dict() if self.provenance is not None else {},
            "reachability": list(self.reachability),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError("collision content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AuthorityCollision":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _COLLISION_FIELDS)
        try:
            record = cls(
                kind=mapping["kind"],
                concern=mapping["concern"],
                disposition=mapping["disposition"],
                message=mapping["message"],
                node_ids=mapping["node_ids"],
                edge_ids=mapping["edge_ids"],
                reachability=mapping["reachability"],
                provenance=mapping["provenance"],
                confidence=mapping["confidence"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise DuplicateAuthorityError("collision content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class BypassFinding:
    """Compatibility or control-plane bypass of a production authority."""

    kind: CollisionKind
    bypass_node_id: str
    production_node_id: str
    disposition: CollisionDisposition
    message: str
    edge_ids: tuple[str, ...] = ()
    reachability: tuple[str, ...] = ()
    provenance: SourceFactIdentity | None = None
    schema: str = BYPASS_SCHEMA
    version: int = BYPASS_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != BYPASS_SCHEMA:
            raise DuplicateAuthorityError("unexpected authority-bypass schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != BYPASS_VERSION:
            raise DuplicateAuthorityError("unexpected authority-bypass version")
        kind = _closed_enum(
            self.kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        if kind not in {
            CollisionKind.COMPATIBILITY_BYPASS,
            CollisionKind.CONTROL_BYPASS,
        }:
            raise DuplicateAuthorityError("bypass kind must be a bypass collision")
        bypass_node_id = _require_text(
            self.bypass_node_id, "bypass_node_id", error_type=DuplicateAuthorityError
        )
        production_node_id = _require_text(
            self.production_node_id,
            "production_node_id",
            error_type=DuplicateAuthorityError,
        )
        if _looks_like_content_identity(bypass_node_id) or _looks_like_content_identity(
            production_node_id
        ):
            raise DuplicateAuthorityError(
                "content identity is not inferred to be authority"
            )
        disposition = _closed_enum(
            self.disposition,
            CollisionDisposition,
            "collision disposition",
            error_type=DuplicateAuthorityError,
        )
        message = _require_text(self.message, "message", error_type=DuplicateAuthorityError)
        edge_ids = _require_text_tuple(self.edge_ids, "edge_ids")
        reachability = _require_text_tuple(self.reachability, "reachability")
        if self.provenance is None:
            raise DuplicateAuthorityError("bypass provenance is required")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "bypass_node_id", bypass_node_id)
        object.__setattr__(self, "production_node_id", production_node_id)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "reachability", reachability)
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
            if claimed != identity:
                raise DuplicateAuthorityError("bypass content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "bypass_node_id": self.bypass_node_id,
            "disposition": self.disposition.value,
            "edge_ids": list(self.edge_ids),
            "kind": self.kind.value,
            "message": self.message,
            "production_node_id": self.production_node_id,
            "provenance": self.provenance.to_dict() if self.provenance is not None else {},
            "reachability": list(self.reachability),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError("bypass content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BypassFinding":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _BYPASS_FIELDS)
        try:
            record = cls(
                kind=mapping["kind"],
                bypass_node_id=mapping["bypass_node_id"],
                production_node_id=mapping["production_node_id"],
                disposition=mapping["disposition"],
                message=mapping["message"],
                edge_ids=mapping["edge_ids"],
                reachability=mapping["reachability"],
                provenance=mapping["provenance"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise DuplicateAuthorityError("bypass content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class SurfaceDivergenceFinding:
    """Python, CLI, and MCP projections of one operation that disagree."""

    operation_node_id: str
    present_surfaces: tuple[SurfaceKind, ...]
    missing_surfaces: tuple[SurfaceKind, ...]
    disposition: CollisionDisposition
    message: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    provenance: SourceFactIdentity | None = None
    kind: CollisionKind = CollisionKind.PYTHON_CLI_MCP_DIVERGENCE
    schema: str = SURFACE_SCHEMA
    version: int = SURFACE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != SURFACE_SCHEMA:
            raise DuplicateAuthorityError("unexpected surface-divergence schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != SURFACE_VERSION:
            raise DuplicateAuthorityError("unexpected surface-divergence version")
        kind = _closed_enum(
            self.kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        if kind is not CollisionKind.PYTHON_CLI_MCP_DIVERGENCE:
            raise DuplicateAuthorityError(
                "surface divergence kind must be python_cli_mcp_divergence"
            )
        operation_node_id = _require_text(
            self.operation_node_id,
            "operation_node_id",
            error_type=DuplicateAuthorityError,
        )
        if _looks_like_content_identity(operation_node_id):
            raise DuplicateAuthorityError(
                "content identity is not inferred to be authority"
            )
        if isinstance(self.present_surfaces, (str, bytes, bytearray)) or not isinstance(
            self.present_surfaces, Sequence
        ):
            raise DuplicateAuthorityError("present_surfaces must be a sequence")
        present = tuple(
            sorted(
                (
                    _closed_enum(
                        item,
                        SurfaceKind,
                        "surface",
                        error_type=DuplicateAuthorityError,
                    )
                    for item in self.present_surfaces
                ),
                key=lambda item: item.value,
            )
        )
        if len(present) != len(set(present)):
            raise DuplicateAuthorityError("present_surfaces must be unique")
        if isinstance(self.missing_surfaces, (str, bytes, bytearray)) or not isinstance(
            self.missing_surfaces, Sequence
        ):
            raise DuplicateAuthorityError("missing_surfaces must be a sequence")
        missing = tuple(
            sorted(
                (
                    _closed_enum(
                        item,
                        SurfaceKind,
                        "surface",
                        error_type=DuplicateAuthorityError,
                    )
                    for item in self.missing_surfaces
                ),
                key=lambda item: item.value,
            )
        )
        if len(missing) != len(set(missing)):
            raise DuplicateAuthorityError("missing_surfaces must be unique")
        if set(present) & set(missing):
            raise DuplicateAuthorityError(
                "a surface cannot be both present and missing"
            )
        disposition = _closed_enum(
            self.disposition,
            CollisionDisposition,
            "collision disposition",
            error_type=DuplicateAuthorityError,
        )
        message = _require_text(self.message, "message", error_type=DuplicateAuthorityError)
        node_ids = _require_text_tuple(self.node_ids, "node_ids")
        edge_ids = _require_text_tuple(self.edge_ids, "edge_ids")
        if self.provenance is None:
            raise DuplicateAuthorityError("surface-divergence provenance is required")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operation_node_id", operation_node_id)
        object.__setattr__(self, "present_surfaces", present)
        object.__setattr__(self, "missing_surfaces", missing)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
            if claimed != identity:
                raise DuplicateAuthorityError(
                    "surface-divergence content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "edge_ids": list(self.edge_ids),
            "kind": self.kind.value,
            "message": self.message,
            "missing_surfaces": [item.value for item in self.missing_surfaces],
            "node_ids": list(self.node_ids),
            "operation_node_id": self.operation_node_id,
            "present_surfaces": [item.value for item in self.present_surfaces],
            "provenance": self.provenance.to_dict() if self.provenance is not None else {},
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError(
                "surface-divergence content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SurfaceDivergenceFinding":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _SURFACE_FIELDS)
        try:
            record = cls(
                operation_node_id=mapping["operation_node_id"],
                present_surfaces=mapping["present_surfaces"],
                missing_surfaces=mapping["missing_surfaces"],
                disposition=mapping["disposition"],
                message=mapping["message"],
                node_ids=mapping["node_ids"],
                edge_ids=mapping["edge_ids"],
                provenance=mapping["provenance"],
                kind=mapping["kind"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != record.content_identity:
            raise DuplicateAuthorityError(
                "surface-divergence content identity mismatch"
            )
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class DuplicateAuthorityBlocker:
    """Typed hard blocker that prevents one-owner settlement."""

    kind: DuplicateAuthorityBlockerKind
    concern: ConcernKind
    message: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    schema: str = BLOCKER_SCHEMA
    version: int = BLOCKER_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != BLOCKER_SCHEMA:
            raise DuplicateAuthorityError("unexpected duplicate-authority-blocker schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != BLOCKER_VERSION:
            raise DuplicateAuthorityError(
                "unexpected duplicate-authority-blocker version"
            )
        kind = _closed_enum(
            self.kind,
            DuplicateAuthorityBlockerKind,
            "duplicate-authority blocker kind",
            error_type=DuplicateAuthorityError,
        )
        concern = _closed_enum(
            self.concern, ConcernKind, "concern", error_type=DuplicateAuthorityError
        )
        message = _require_text(self.message, "message", error_type=DuplicateAuthorityError)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self, "node_ids", _require_text_tuple(self.node_ids, "node_ids")
        )
        object.__setattr__(
            self, "edge_ids", _require_text_tuple(self.edge_ids, "edge_ids")
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
            if claimed != identity:
                raise DuplicateAuthorityError("blocker content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "edge_ids": list(self.edge_ids),
            "kind": self.kind.value,
            "message": self.message,
            "node_ids": list(self.node_ids),
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError("blocker content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DuplicateAuthorityBlocker":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _BLOCKER_FIELDS)
        record = cls(
            kind=mapping["kind"],
            concern=mapping["concern"],
            message=mapping["message"],
            node_ids=mapping["node_ids"],
            edge_ids=mapping["edge_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise DuplicateAuthorityError("blocker content identity mismatch")
        return record

    from_dict = from_mapping


def _record_tuple(
    value: Any,
    name: str,
    record_type: type[Any],
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DuplicateAuthorityError(f"{name} must be a list of objects")
    records = tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )
    return tuple(sorted(records, key=lambda item: item.content_identity))


@dataclass(frozen=True)
class DuplicateAuthorityReport:
    """Closed report of collisions, bypasses, divergences, and blockers."""

    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    collisions: tuple[AuthorityCollision, ...] = ()
    bypasses: tuple[BypassFinding, ...] = ()
    surface_divergences: tuple[SurfaceDivergenceFinding, ...] = ()
    blockers: tuple[DuplicateAuthorityBlocker, ...] = ()
    rejected: tuple[AuthorityCollision, ...] = ()
    unknown: tuple[AuthorityCollision, ...] = ()
    ownership_graph_identity: str = ""
    schema: str = DUPLICATE_AUTHORITY_SCHEMA
    version: int = DUPLICATE_AUTHORITY_VERSION
    can_authorize_changes: bool = DETECTOR_CAN_AUTHORIZE_CHANGES
    can_remediate: bool = DETECTOR_CAN_REMEDIATE
    can_select_canonical: bool = DETECTOR_CAN_SELECT_CANONICAL
    one_owner_invariant: bool = ONE_OWNER_INVARIANT
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != DUPLICATE_AUTHORITY_SCHEMA:
            raise DuplicateAuthorityError("unexpected duplicate-authority schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != DUPLICATE_AUTHORITY_VERSION:
            raise DuplicateAuthorityError("unexpected duplicate-authority version")
        if self.can_authorize_changes is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot authorize changes"
            )
        if self.can_remediate is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot remediate"
            )
        if self.can_select_canonical is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot select a canonical owner"
            )
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=DuplicateAuthorityError,
            )
        )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=DuplicateAuthorityError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=DuplicateAuthorityError
        )
        ownership_graph_identity = self.ownership_graph_identity
        if ownership_graph_identity:
            ownership_graph_identity = _validate_dag_json_cid(
                _require_text(
                    ownership_graph_identity,
                    "ownership_graph_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
        else:
            ownership_graph_identity = ""
        collisions = _record_tuple(self.collisions, "collisions", AuthorityCollision)
        bypasses = _record_tuple(self.bypasses, "bypasses", BypassFinding)
        surfaces = _record_tuple(
            self.surface_divergences, "surface_divergences", SurfaceDivergenceFinding
        )
        blockers = _record_tuple(
            self.blockers, "blockers", DuplicateAuthorityBlocker
        )
        rejected = _record_tuple(self.rejected, "rejected", AuthorityCollision)
        unknown = _record_tuple(self.unknown, "unknown", AuthorityCollision)
        for record in collisions:
            if record.disposition is not CollisionDisposition.COLLISION:
                raise DuplicateAuthorityError(
                    "collisions must use disposition collision"
                )
        for record in rejected:
            if record.disposition not in {
                CollisionDisposition.FALSE_POSITIVE,
                CollisionDisposition.FORMALLY_ARBITRATED,
            }:
                raise DuplicateAuthorityError(
                    "rejected findings must be false positives or formally arbitrated"
                )
        for record in unknown:
            if record.disposition is not CollisionDisposition.UNKNOWN:
                raise DuplicateAuthorityError(
                    "unknown findings must use disposition unknown"
                )
        _require_bool(self.one_owner_invariant, "one_owner_invariant")
        expected_one_owner = not collisions and not any(
            item.kind
            is DuplicateAuthorityBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES
            for item in blockers
        )
        if self.one_owner_invariant is not expected_one_owner:
            raise DuplicateAuthorityError(
                "one-owner invariant must match collision and blocker cardinality"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "ownership_graph_identity", ownership_graph_identity)
        object.__setattr__(self, "collisions", collisions)
        object.__setattr__(self, "bypasses", bypasses)
        object.__setattr__(self, "surface_divergences", surfaces)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_remediate", False)
        object.__setattr__(self, "can_select_canonical", False)
        object.__setattr__(self, "one_owner_invariant", expected_one_owner)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=DuplicateAuthorityError,
                )
            )
            if claimed != identity:
                raise DuplicateAuthorityError(
                    "duplicate-authority content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "blockers": [item.to_dict() for item in self.blockers],
            "bypasses": [item.to_dict() for item in self.bypasses],
            "can_authorize_changes": False,
            "can_remediate": False,
            "can_select_canonical": False,
            "collisions": [item.to_dict() for item in self.collisions],
            "freshness": self.freshness,
            "one_owner_invariant": self.one_owner_invariant,
            "ownership_graph_identity": self.ownership_graph_identity,
            "rejected": [item.to_dict() for item in self.rejected],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "surface_divergences": [
                item.to_dict() for item in self.surface_divergences
            ],
            "unknown": [item.to_dict() for item in self.unknown],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError(
                "duplicate-authority content identity mismatch"
            )
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def fails_closed(self) -> bool:
        return bool(self.collisions) or bool(self.blockers)

    @property
    def detected_kinds(self) -> frozenset[CollisionKind]:
        kinds = {item.kind for item in self.collisions}
        kinds.update(item.kind for item in self.bypasses)
        kinds.update(item.kind for item in self.surface_divergences)
        return frozenset(kinds)

    def collisions_of(self, kind: CollisionKind | str) -> tuple[AuthorityCollision, ...]:
        parsed = _closed_enum(
            kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        return tuple(item for item in self.collisions if item.kind is parsed)

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authorization("change")

    def remediate(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("remediation")

    def select_canonical(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_canonical_selection("canonical owner")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DuplicateAuthorityReport":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _REPORT_FIELDS)
        report = cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            collisions=mapping["collisions"],
            bypasses=mapping["bypasses"],
            surface_divergences=mapping["surface_divergences"],
            blockers=mapping["blockers"],
            rejected=mapping["rejected"],
            unknown=mapping["unknown"],
            ownership_graph_identity=mapping["ownership_graph_identity"],
            schema=mapping["schema"],
            version=mapping["version"],
            can_authorize_changes=mapping["can_authorize_changes"],
            can_remediate=mapping["can_remediate"],
            can_select_canonical=mapping["can_select_canonical"],
            one_owner_invariant=mapping["one_owner_invariant"],
        )
        if mapping["content_identity"] != report.content_identity:
            raise DuplicateAuthorityError(
                "duplicate-authority content identity mismatch"
            )
        return report

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "DuplicateAuthorityReport":
        if type(payload) is not str or not payload:
            raise DuplicateAuthorityError(
                "duplicate-authority JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DuplicateAuthorityError(
                "duplicate-authority JSON is malformed"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise DuplicateAuthorityError(
                "duplicate-authority JSON must contain an object"
            )
        return cls.from_mapping(decoded)


def refuse_authorization(action: str) -> None:
    """Reject attempts to treat findings as change authority."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityAuthorityError(
        f"duplicate-authority detector cannot authorize {name}"
    )


def refuse_remediation(action: str) -> None:
    """Reject attempts to consolidate or delete authorities from a finding."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityAuthorityError(
        f"duplicate-authority detector cannot execute {name}"
    )


def refuse_canonical_selection(action: str) -> None:
    """Reject attempts to select a canonical owner from a finding."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityAuthorityError(
        f"duplicate-authority detector cannot select a {name}"
    )


def lookup_owner_by_content_identity(*_args: Any, **_kwargs: Any) -> None:
    """Content identity never selects or proves a canonical owner."""

    raise DuplicateAuthorityError("content identity is not inferred to be authority")


def silently_select_canonical(*_args: Any, **_kwargs: Any) -> None:
    """Refuse first-listed, majority, or identity-sorted owner selection."""

    raise DuplicateAuthorityError("silent arbitration is prohibited")


def _collision(
    view: _GraphView,
    index: _OwnershipIndex,
    kind: CollisionKind,
    node_ids: Sequence[str],
    edge_ids: Sequence[str],
    message: str,
    reachability: Sequence[str] = (),
    competitors: Sequence[str] | None = None,
) -> AuthorityCollision:
    facts = _facts_for(view, node_ids, edge_ids)
    confidence = _worst_confidence(item.confidence for item in facts)
    classify_ids = tuple(competitors) if competitors is not None else tuple(node_ids)
    disposition = _classify_disposition(view, index, kind, classify_ids, confidence)
    if (
        disposition is CollisionDisposition.COLLISION
        and confidence in NON_PROBATIVE_CONFIDENCE
    ):
        disposition = CollisionDisposition.UNKNOWN
    return AuthorityCollision(
        kind=kind,
        concern=_KIND_CONCERN[kind],
        disposition=disposition,
        message=message,
        node_ids=tuple(node_ids),
        edge_ids=tuple(edge_ids),
        reachability=tuple(reachability) or tuple(node_ids),
        provenance=_provenance_for(view, node_ids, edge_ids),
        confidence=confidence,
    )


def _competing_sources(
    view: _GraphView,
    edge_kinds: frozenset[EdgeKind],
    *,
    source_kinds: frozenset[NodeKind] | None = None,
    target_kinds: frozenset[NodeKind] | None = None,
    require_source_or_target: frozenset[NodeKind] | None = None,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    grouped: dict[str, list[ArchitectureEdge]] = defaultdict(list)
    for edge in view.architecture.edges:
        if edge.kind not in edge_kinds or edge.source == edge.target:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        if source_kinds is not None and source.kind not in source_kinds:
            continue
        if target_kinds is not None and target.kind not in target_kinds:
            continue
        if require_source_or_target is not None:
            if (
                source.kind not in require_source_or_target
                and target.kind not in require_source_or_target
            ):
                continue
        grouped[edge.target].append(edge)
    results: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for target_id, edges in grouped.items():
        sources = tuple(sorted({edge.source for edge in edges}))
        if len(sources) < 2:
            continue
        edge_ids = tuple(sorted({edge.edge_id for edge in edges}))
        results.append((target_id, sources, edge_ids))
    return tuple(sorted(results, key=lambda item: item[0]))


def _explicit_duplicate_pairs(
    view: _GraphView,
    kinds: frozenset[NodeKind],
) -> tuple[tuple[str, str, str], ...]:
    pairs: list[tuple[str, str, str]] = []
    for edge in view.architecture.edges:
        if edge.kind not in {EdgeKind.DUPLICATES, EdgeKind.SHADOWS}:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        if source.kind not in kinds and target.kind not in kinds:
            continue
        left, right = sorted((edge.source, edge.target))
        pairs.append((left, right, edge.edge_id))
    return tuple(sorted(set(pairs)))


def _detect_independent_decisions(
    view: _GraphView,
    index: _OwnershipIndex,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    seen: set[tuple[CollisionKind, tuple[str, ...]]] = set()

    def _add(
        kind: CollisionKind,
        node_ids: Sequence[str],
        edge_ids: Sequence[str],
        message: str,
        reachability: Sequence[str] = (),
        competitors: Sequence[str] | None = None,
    ) -> None:
        key = (kind, tuple(sorted(set(node_ids))))
        if key in seen:
            return
        seen.add(key)
        findings.append(
            _collision(
                view,
                index,
                kind,
                node_ids,
                edge_ids,
                message,
                reachability,
                competitors=competitors,
            )
        )

    for target_id, sources, edge_ids in _competing_sources(
        view,
        _PROVIDER_DECISION_EDGES,
        source_kinds=frozenset({NodeKind.AUTHORITY, NodeKind.PROVIDER}),
        require_source_or_target=frozenset({NodeKind.PROVIDER}),
    ):
        _add(
            CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
            (*sources, target_id),
            edge_ids,
            "independent provider-capability decisions share one subject",
            (*sources, target_id),
            competitors=sources,
        )
    for left, right, edge_id in _explicit_duplicate_pairs(
        view, frozenset({NodeKind.PROVIDER})
    ):
        _add(
            CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
            (left, right),
            (edge_id,),
            "provider nodes are explicit duplicates or shadows",
            (left, right),
            competitors=(left, right),
        )
    for target_id, sources, edge_ids in _competing_sources(
        view,
        _RECEIPT_DECISION_EDGES,
        source_kinds=frozenset({NodeKind.AUTHORITY, NodeKind.RECEIPT}),
        require_source_or_target=frozenset({NodeKind.RECEIPT}),
    ):
        _add(
            CollisionKind.INDEPENDENT_RECEIPT_DECISION,
            (*sources, target_id),
            edge_ids,
            "independent receipt producers decide the same evidence",
            (*sources, target_id),
            competitors=sources,
        )
    for left, right, edge_id in _explicit_duplicate_pairs(
        view, frozenset({NodeKind.RECEIPT})
    ):
        _add(
            CollisionKind.INDEPENDENT_RECEIPT_DECISION,
            (left, right),
            (edge_id,),
            "receipt nodes are explicit duplicates or shadows",
            (left, right),
            competitors=(left, right),
        )
    for target_id, sources, edge_ids in _competing_sources(
        view,
        _STATE_OWNER_EDGES,
        source_kinds=frozenset({NodeKind.AUTHORITY, NodeKind.STATE}),
        require_source_or_target=frozenset({NodeKind.STATE}),
    ):
        _add(
            CollisionKind.COMPETING_STATE_OWNER,
            (*sources, target_id),
            edge_ids,
            "competing state owners persist the same mutable fact",
            (*sources, target_id),
            competitors=sources,
        )
    for left, right, edge_id in _explicit_duplicate_pairs(
        view, frozenset({NodeKind.STATE})
    ):
        _add(
            CollisionKind.COMPETING_STATE_OWNER,
            (left, right),
            (edge_id,),
            "state nodes are explicit duplicates or shadows",
            (left, right),
            competitors=(left, right),
        )
    return findings


def _adapter_of(view: _GraphView, source_id: str, target_id: str) -> bool:
    return any(
        edge.kind is EdgeKind.ADAPTS and edge.target == target_id
        for edge in view.outgoing.get(source_id, ())
    )


def _detect_bypasses(
    view: _GraphView,
    index: _OwnershipIndex,
) -> tuple[list[AuthorityCollision], list[BypassFinding]]:
    collisions: list[AuthorityCollision] = []
    bypasses: list[BypassFinding] = []
    policy_targets = {
        edge.target
        for edge in view.architecture.edges
        if edge.kind is EdgeKind.EVALUATES_POLICY
        and view.nodes_by_id[edge.source].kind is NodeKind.POLICY
    }
    seen: set[tuple[CollisionKind, str, str]] = set()
    for edge in view.architecture.edges:
        if edge.kind not in _BYPASS_EDGES or edge.source == edge.target:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        if target.kind not in _PRODUCTION_NODE_KINDS and target.kind not in {
            NodeKind.OPERATION,
            NodeKind.SYMBOL,
        }:
            continue
        if source.kind is NodeKind.COMPATIBILITY:
            kind = CollisionKind.COMPATIBILITY_BYPASS
            message = "compatibility path bypasses the canonical production authority"
        elif (
            edge.kind in _CONTROL_BYPASS_EDGES
            and edge.target in policy_targets
            and source.kind
            not in {NodeKind.POLICY, NodeKind.AUTHORITY, NodeKind.TEST, NodeKind.PROOF}
        ):
            policies = [
                incoming.source
                for incoming in view.incoming.get(edge.target, ())
                if incoming.kind is EdgeKind.EVALUATES_POLICY
                and view.nodes_by_id[incoming.source].kind is NodeKind.POLICY
            ]
            if any(_adapter_of(view, source.node_id, policy_id) for policy_id in policies):
                continue
            if any(
                policy_id == source.node_id
                or any(
                    rel.kind in {EdgeKind.AUTHORIZES, EdgeKind.ADAPTS}
                    and {rel.source, rel.target} == {source.node_id, policy_id}
                    for rel in _related_edges(view, source.node_id)
                )
                for policy_id in policies
            ):
                continue
            kind = CollisionKind.CONTROL_BYPASS
            message = "tool dispatch bypasses policy evaluation"
        else:
            continue
        key = (kind, source.node_id, target.node_id)
        if key in seen:
            continue
        seen.add(key)
        collision = _collision(
            view,
            index,
            kind,
            (source.node_id, target.node_id),
            (edge.edge_id,),
            message,
            (source.node_id, target.node_id),
        )
        collisions.append(collision)
        bypasses.append(
            BypassFinding(
                kind=kind,
                bypass_node_id=source.node_id,
                production_node_id=target.node_id,
                disposition=collision.disposition,
                message=message,
                edge_ids=(edge.edge_id,),
                reachability=(source.node_id, target.node_id),
                provenance=edge.provenance,
            )
        )
    return collisions, bypasses


def _detect_simulation_flow(
    view: _GraphView,
    index: _OwnershipIndex,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    simulations = [
        node
        for node in view.architecture.nodes
        if node.kind is NodeKind.SIMULATION
        or _node_disposition(view, index, node.node_id) is OwnerDisposition.SIMULATION
    ]
    for start in simulations:
        parent: dict[str, tuple[str, str]] = {}
        seen = {start.node_id}
        queue: deque[str] = deque([start.node_id])
        hit: tuple[str, ...] | None = None
        hit_edges: list[str] = []
        while queue:
            current = queue.popleft()
            for edge in view.outgoing.get(current, ()):
                if edge.kind not in _FLOW_EDGES:
                    continue
                if (
                    edge.kind is EdgeKind.FALLBACKS_TO
                    and view.nodes_by_id[edge.source].kind is not NodeKind.SIMULATION
                    and view.nodes_by_id[edge.target].kind is NodeKind.SIMULATION
                ):
                    continue
                nxt = edge.target
                if nxt in seen:
                    continue
                seen.add(nxt)
                parent[nxt] = (current, edge.edge_id)
                target = view.nodes_by_id[nxt]
                if (
                    nxt != start.node_id
                    and target.kind in _PRODUCTION_NODE_KINDS
                    and target.kind is not NodeKind.SIMULATION
                ):
                    path_nodes = [nxt]
                    path_edges = []
                    cursor = nxt
                    while cursor != start.node_id:
                        prev, edge_id = parent[cursor]
                        path_nodes.append(prev)
                        path_edges.append(edge_id)
                        cursor = prev
                    path_nodes.reverse()
                    path_edges.reverse()
                    hit = tuple(path_nodes)
                    hit_edges = list(dict.fromkeys(path_edges))
                    break
                queue.append(nxt)
            if hit is not None:
                break
        if hit is None:
            continue
        findings.append(
            _collision(
                view,
                index,
                CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
                hit,
                hit_edges,
                "simulation origin reaches a production authority or predicate",
                hit,
            )
        )
    return findings


def _surface_kind(node: ArchitectureNode) -> SurfaceKind | None:
    text = f"{node.provenance.span.path} {node.node_id}".lower().replace("\\", "/")
    if any(marker in text for marker in _MCP_MARKERS):
        return SurfaceKind.MCP
    if any(marker in text for marker in _CLI_MARKERS):
        return SurfaceKind.CLI
    if node.kind is NodeKind.ENTRYPOINT:
        return SurfaceKind.PYTHON
    if "entrypoints/" in text or text.endswith("/__init__.py"):
        return SurfaceKind.PYTHON
    return None


def _detect_surface_divergence(
    view: _GraphView,
    index: _OwnershipIndex,
) -> tuple[list[AuthorityCollision], list[SurfaceDivergenceFinding]]:
    collisions: list[AuthorityCollision] = []
    surfaces: list[SurfaceDivergenceFinding] = []
    by_operation: dict[str, dict[SurfaceKind, tuple[ArchitectureNode, list[str]]]] = {}
    for edge in view.architecture.edges:
        if edge.kind not in {
            EdgeKind.IMPLEMENTS,
            EdgeKind.EXECUTES,
            EdgeKind.CALLS,
        }:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        operation_id = None
        entry = None
        if target.kind is NodeKind.OPERATION:
            operation_id = target.node_id
            entry = source
        elif source.kind is NodeKind.OPERATION:
            operation_id = source.node_id
            entry = target
        if operation_id is None or entry is None:
            continue
        surface = _surface_kind(entry)
        if surface is None:
            continue
        slot = by_operation.setdefault(operation_id, {})
        current = slot.get(surface)
        if current is None:
            slot[surface] = (entry, [edge.edge_id])
        elif edge.edge_id not in current[1]:
            current[1].append(edge.edge_id)
    shadow_ops: set[str] = set()
    for edge in view.architecture.edges:
        if edge.kind not in {EdgeKind.SHADOWS, EdgeKind.DUPLICATES}:
            continue
        left = _surface_kind(view.nodes_by_id[edge.source])
        right = _surface_kind(view.nodes_by_id[edge.target])
        if left is None or right is None or left is right:
            continue
        for operation_id, present in by_operation.items():
            node_ids = {item[0].node_id for item in present.values()}
            if edge.source in node_ids and edge.target in node_ids:
                shadow_ops.add(operation_id)
    for operation_id, present in sorted(by_operation.items()):
        kinds = tuple(sorted(present, key=lambda item: item.value))
        missing = tuple(
            item for item in REQUIRED_SURFACES if item not in present
        )
        diverges = bool(missing) and len(present) >= 2
        diverges = diverges or operation_id in shadow_ops
        if not diverges:
            continue
        node_ids = [operation_id]
        edge_ids: list[str] = []
        for _kind, (node, edges) in sorted(
            present.items(), key=lambda item: item[0].value
        ):
            node_ids.append(node.node_id)
            edge_ids.extend(edges)
        if operation_id in shadow_ops:
            message = "Python/CLI/MCP projections shadow or duplicate one another"
        else:
            message = "Python/CLI/MCP projections of one operation diverge"
        collision = _collision(
            view,
            index,
            CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
            node_ids,
            edge_ids,
            message,
            node_ids,
        )
        collisions.append(collision)
        surfaces.append(
            SurfaceDivergenceFinding(
                operation_node_id=operation_id,
                present_surfaces=kinds,
                missing_surfaces=missing,
                disposition=collision.disposition,
                message=message,
                node_ids=tuple(node_ids),
                edge_ids=tuple(edge_ids),
                provenance=collision.provenance,
            )
        )
    return collisions, surfaces


def _detect_reexport_authorities(
    view: _GraphView,
    index: _OwnershipIndex,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    for edge in view.architecture.edges:
        if edge.kind is not EdgeKind.REEXPORTS:
            continue
        source = view.nodes_by_id[edge.source]
        claims_authority = any(
            outgoing.kind
            in {
                EdgeKind.AUTHORIZES,
                EdgeKind.CONFIRMS,
                EdgeKind.EVALUATES_POLICY,
                EdgeKind.PERSISTS,
                EdgeKind.PROVES,
                EdgeKind.EXECUTES,
            }
            for outgoing in view.outgoing.get(source.node_id, ())
        )
        if not claims_authority:
            continue
        if source.kind not in {NodeKind.AUTHORITY, NodeKind.INTERFACE, NodeKind.MODULE}:
            if _node_disposition(view, index, source.node_id) is not OwnerDisposition.CANONICAL:
                continue
        claimed = [
            outgoing.edge_id
            for outgoing in view.outgoing.get(source.node_id, ())
            if outgoing.kind
            in {
                EdgeKind.AUTHORIZES,
                EdgeKind.CONFIRMS,
                EdgeKind.EVALUATES_POLICY,
                EdgeKind.PERSISTS,
                EdgeKind.PROVES,
                EdgeKind.EXECUTES,
            }
        ]
        findings.append(
            _collision(
                view,
                index,
                CollisionKind.REEXPORT_AUTHORITY,
                (source.node_id, edge.target),
                (edge.edge_id, *claimed),
                "re-export is acting as a production authority",
                (source.node_id, edge.target),
            )
        )
    return findings


def _detect_obsolete_tests(
    view: _GraphView,
    index: _OwnershipIndex,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    superseded: dict[str, list[ArchitectureEdge]] = defaultdict(list)
    for edge in view.architecture.edges:
        if edge.kind in {EdgeKind.SUPERSEDES, EdgeKind.DEPRECATES}:
            superseded[edge.target].append(edge)
    for node in view.architecture.nodes:
        if node.kind is not NodeKind.TEST:
            continue
        tested = [
            edge
            for edge in view.outgoing.get(node.node_id, ())
            if edge.kind is EdgeKind.TESTS
        ]
        tested_ids = {edge.target for edge in tested}
        obsolete_targets = [
            target_id for target_id in tested_ids if target_id in superseded
        ]
        if not obsolete_targets:
            continue
        canonical_ids = {
            edge.source
            for target_id in obsolete_targets
            for edge in superseded[target_id]
        }
        if tested_ids & canonical_ids:
            continue
        edge_ids = [edge.edge_id for edge in tested if edge.target in obsolete_targets]
        edge_ids.extend(
            edge.edge_id
            for target_id in obsolete_targets
            for edge in superseded[target_id]
        )
        findings.append(
            _collision(
                view,
                index,
                CollisionKind.OBSOLETE_AUTHORITY_TEST,
                (node.node_id, *obsolete_targets, *sorted(canonical_ids)),
                edge_ids,
                "test validates obsolete rather than canonical authority",
                (node.node_id, *obsolete_targets),
            )
        )
    return findings


def _emit_blockers(
    view: _GraphView,
    index: _OwnershipIndex,
    collisions: Sequence[AuthorityCollision],
) -> list[DuplicateAuthorityBlocker]:
    blockers: list[DuplicateAuthorityBlocker] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def _add(
        kind: DuplicateAuthorityBlockerKind,
        concern: ConcernKind,
        message: str,
        node_ids: Sequence[str] = (),
        edge_ids: Sequence[str] = (),
    ) -> None:
        key = (kind.value, concern.value, tuple(sorted(set(node_ids))))
        if key in seen:
            return
        seen.add(key)
        blockers.append(
            DuplicateAuthorityBlocker(
                kind=kind,
                concern=concern,
                message=message,
                node_ids=tuple(node_ids),
                edge_ids=tuple(edge_ids),
            )
        )

    if index.graph is not None:
        for record in index.graph.blockers:
            if record.kind is OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER:
                _add(
                    DuplicateAuthorityBlockerKind.UNKNOWN_PRODUCTION_OWNER,
                    record.concern,
                    "unknown production ownership fails closed",
                    record.node_ids,
                    record.edge_ids,
                )
            elif record.kind is OwnershipBlockerKind.UNKNOWN_OWNER:
                if record.node_ids:
                    _add(
                        DuplicateAuthorityBlockerKind.UNKNOWN_OWNER,
                        record.concern,
                        "unknown production ownership fails closed",
                        record.node_ids,
                        record.edge_ids,
                    )
            elif record.kind is OwnershipBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES:
                _add(
                    DuplicateAuthorityBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES,
                    record.concern,
                    "multiple production authorities require formal arbitration",
                    record.node_ids,
                    record.edge_ids,
                )
            elif record.kind is OwnershipBlockerKind.MISSING_ARBITRATION:
                _add(
                    DuplicateAuthorityBlockerKind.MISSING_ARBITRATION,
                    record.concern,
                    "competing production authorities lack formal arbitration",
                    record.node_ids,
                    record.edge_ids,
                )
    for node in view.architecture.nodes:
        if node.kind is not NodeKind.AUTHORITY:
            continue
        authorizes_production = any(
            edge.kind
            in {
                EdgeKind.AUTHORIZES,
                EdgeKind.CONFIRMS,
                EdgeKind.EVALUATES_POLICY,
                EdgeKind.PERSISTS,
                EdgeKind.PROVES,
                EdgeKind.EXECUTES,
            }
            and view.nodes_by_id[edge.target].kind in _PRODUCTION_NODE_KINDS
            for edge in view.outgoing.get(node.node_id, ())
        )
        if not authorizes_production:
            continue
        disposition = index.disposition_by_node.get(node.node_id)
        if disposition is OwnerDisposition.UNKNOWN or (
            index.graph is not None
            and node.node_id not in index.classified_nodes
            and disposition is None
        ):
            _add(
                DuplicateAuthorityBlockerKind.UNKNOWN_PRODUCTION_OWNER,
                ConcernKind.AUTHORIZATION,
                "unclassified production authority fails closed",
                (node.node_id,),
                tuple(
                    edge.edge_id
                    for edge in view.outgoing.get(node.node_id, ())
                    if edge.kind is EdgeKind.AUTHORIZES
                ),
            )
    for collision in collisions:
        if collision.kind in {
            CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
            CollisionKind.INDEPENDENT_RECEIPT_DECISION,
            CollisionKind.COMPETING_STATE_OWNER,
        }:
            _add(
                DuplicateAuthorityBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES,
                collision.concern,
                "detected competing production authorities require arbitration",
                collision.node_ids,
                collision.edge_ids,
            )
    return blockers


def _partition(
    records: Sequence[AuthorityCollision],
) -> tuple[
    tuple[AuthorityCollision, ...],
    tuple[AuthorityCollision, ...],
    tuple[AuthorityCollision, ...],
]:
    collisions: list[AuthorityCollision] = []
    rejected: list[AuthorityCollision] = []
    unknown: list[AuthorityCollision] = []
    for record in records:
        if record.disposition is CollisionDisposition.COLLISION:
            collisions.append(record)
        elif record.disposition is CollisionDisposition.UNKNOWN:
            unknown.append(record)
        else:
            rejected.append(record)
    return tuple(collisions), tuple(rejected), tuple(unknown)


@dataclass(frozen=True)
class DuplicateAuthorityDetector:
    """Read-only detector for competing production authorities."""

    extractor_identity: str = EXTRACTOR_IDENTITY
    can_authorize_changes: bool = DETECTOR_CAN_AUTHORIZE_CHANGES
    can_remediate: bool = DETECTOR_CAN_REMEDIATE
    can_select_canonical: bool = DETECTOR_CAN_SELECT_CANONICAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extractor_identity",
            _require_text(
                self.extractor_identity,
                "extractor_identity",
                error_type=DuplicateAuthorityError,
            ),
        )
        if self.can_authorize_changes is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot authorize changes"
            )
        if self.can_remediate is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot remediate"
            )
        if self.can_select_canonical is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot select a canonical owner"
            )
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_remediate", False)
        object.__setattr__(self, "can_select_canonical", False)

    def detect(
        self,
        architecture: ArchitectureIR | Mapping[str, Any],
        ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
        *,
        arbitrations: Sequence[FormalArbitration | Mapping[str, Any]] | None = None,
    ) -> DuplicateAuthorityReport:
        return detect_duplicate_authorities(
            architecture, ownership, arbitrations=arbitrations
        )

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authorization("change")

    def remediate(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("remediation")

    def select_canonical(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_canonical_selection("canonical owner")


def detect_duplicate_authorities(
    architecture: ArchitectureIR | Mapping[str, Any],
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
    *,
    arbitrations: Sequence[FormalArbitration | Mapping[str, Any]] | None = None,
) -> DuplicateAuthorityReport:
    """Detect required duplicate-authority collisions without remediating them."""

    graph = _require_architecture_ir(architecture)
    ownership_graph = _require_ownership_graph(ownership)
    extra = _normalize_arbitrations(arbitrations)
    view = _build_view(graph)
    index = _index_ownership(ownership_graph, extra)
    decision_findings = _detect_independent_decisions(view, index)
    bypass_collisions, bypasses = _detect_bypasses(view, index)
    flow_findings = _detect_simulation_flow(view, index)
    surface_collisions, surfaces = _detect_surface_divergence(view, index)
    reexport_findings = _detect_reexport_authorities(view, index)
    obsolete_findings = _detect_obsolete_tests(view, index)
    combined = (
        *decision_findings,
        *bypass_collisions,
        *flow_findings,
        *surface_collisions,
        *reexport_findings,
        *obsolete_findings,
    )
    collisions, rejected, unknown = _partition(combined)
    collision_bypasses = tuple(
        item
        for item in bypasses
        if item.disposition is CollisionDisposition.COLLISION
    )
    collision_surfaces = tuple(
        item
        for item in surfaces
        if item.disposition is CollisionDisposition.COLLISION
    )
    blockers = tuple(
        sorted(
            _emit_blockers(view, index, collisions),
            key=lambda item: item.content_identity,
        )
    )
    one_owner = not collisions and not any(
        item.kind is DuplicateAuthorityBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES
        for item in blockers
    )
    return DuplicateAuthorityReport(
        architecture_ir_identity=graph.content_identity,
        repository_tree=graph.repository_tree,
        freshness=graph.freshness,
        collisions=collisions,
        bypasses=collision_bypasses,
        surface_divergences=collision_surfaces,
        blockers=blockers,
        rejected=rejected,
        unknown=unknown,
        ownership_graph_identity=(
            "" if ownership_graph is None else ownership_graph.content_identity
        ),
        one_owner_invariant=one_owner,
    )


build_duplicate_authority_report = detect_duplicate_authorities


__all__ = [
    "BLOCKER_SCHEMA",
    "BLOCKER_VERSION",
    "BYPASS_SCHEMA",
    "BYPASS_VERSION",
    "CLOSED_COLLISION_DISPOSITIONS",
    "CLOSED_COLLISION_KINDS",
    "CLOSED_DUPLICATE_AUTHORITY_BLOCKERS",
    "CLOSED_SURFACES",
    "COLLISION_SCHEMA",
    "COLLISION_VERSION",
    "CONTENT_IDENTITY_IS_NOT_AUTHORITY",
    "DEFAULT_FRESHNESS",
    "DETECTOR_CAN_AUTHORIZE_CHANGES",
    "DETECTOR_CAN_REMEDIATE",
    "DETECTOR_CAN_SELECT_CANONICAL",
    "DUPLICATE_AUTHORITY_EVIDENCE",
    "DUPLICATE_AUTHORITY_SCHEMA",
    "DUPLICATE_AUTHORITY_VERSION",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "HEURISTIC_CRITICAL_PROMOTION_PROHIBITED",
    "ONE_OWNER_INVARIANT",
    "REQUIRED_COLLISION_KINDS",
    "REQUIRED_SURFACES",
    "REEXPORT_IS_NOT_AUTHORITY",
    "SILENT_ARBITRATION_PROHIBITED",
    "SURFACE_SCHEMA",
    "SURFACE_VERSION",
    "TASK_ID",
    "AuthorityCollision",
    "BypassFinding",
    "CollisionDisposition",
    "CollisionKind",
    "DuplicateAuthorityAuthorityError",
    "DuplicateAuthorityBlocker",
    "DuplicateAuthorityBlockerKind",
    "DuplicateAuthorityDetector",
    "DuplicateAuthorityError",
    "DuplicateAuthorityReport",
    "SurfaceDivergenceFinding",
    "SurfaceKind",
    "build_duplicate_authority_report",
    "detect_duplicate_authorities",
    "lookup_owner_by_content_identity",
    "refuse_authorization",
    "refuse_canonical_selection",
    "refuse_remediation",
    "silently_select_canonical",
]
