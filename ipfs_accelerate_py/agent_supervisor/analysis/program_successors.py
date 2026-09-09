"""Conservative static successor generation for program-world planning.

Interface: ``StaticSuccessorPlanner@1``.  Evidence: ``sawm/static-successors@1``.

This module consumes datasets ``ProgramGraphSnapshot@1`` catalogs (SAWM-005/006)
and emits next-call / next-event candidate sets plus an explicit unresolved
dynamic frontier.  It does not scan source, does not mint graph meaning, and
does not run a solver.  Unknown dynamic behavior is never encoded as absence.

Authority rules (fail-closed):

* Static possibilities and explicit unknown are preserved unless a later
  authoritative symbolic reason removes them.
* Neural, ANN, and similarity signals cannot author candidates or completeness.
* Stale snapshot / environment / source bindings reject generation.
* Every selected / skipped / unavailable / rejected / escalated stage is
  recorded on a deterministic ``SuccessorDecisionReceipt@1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    DynamicFrontierReason,
    DynamicFrontierRecord,
    ProgramGraphEdge,
    ProgramGraphEdgeKind,
    ProgramGraphNode,
    ProgramGraphNodeKind,
    ProgramGraphSnapshot,
    ResolutionStatus,
    StaticSuccessorSet,
    is_unresolved_dynamic_edge,
    is_unresolved_dynamic_node,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_transition import (
    QueryFamily,
)

from ..proof.formal_verification_contracts import content_identity


# ---------------------------------------------------------------------------
# Interfaces / schemas / evidence
# ---------------------------------------------------------------------------

STATIC_SUCCESSOR_PLANNER_INTERFACE: Final[str] = "StaticSuccessorPlanner@1"
UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE: Final[str] = "UnresolvedDynamicFrontier@1"
SUCCESSOR_DECISION_RECEIPT_INTERFACE: Final[str] = "SuccessorDecisionReceipt@1"
STATIC_SUCCESSOR_PLAN_INTERFACE: Final[str] = "StaticSuccessorPlan@1"
STATIC_SUCCESSORS_EVIDENCE: Final[str] = "sawm/static-successors@1"
SYMBOLIC_PRUNING_EVIDENCE: Final[str] = "sawm/symbolic-pruning@1"

STATIC_SUCCESSOR_PLANNER_VERSION: Final[str] = "1"
PRODUCER_ID: Final[str] = "ipfs-accelerate.analysis.program-successors"

STATIC_SUCCESSOR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/static-successor-candidate@1"
)
UNRESOLVED_DYNAMIC_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/unresolved-dynamic-frontier@1"
)
SUCCESSOR_STAGE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/successor-stage-decision@1"
)
SUCCESSOR_DECISION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/successor-decision-receipt@1"
)
STATIC_SUCCESSOR_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/static-successor-plan@1"
)
PROGRAM_GRAPH_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-graph-catalog-view@1"
)

ADMITTED_LANGUAGE: Final[str] = "python"
IMPORT_SCAN_PERFORMED: Final[bool] = False
MAX_TEXT_CHARS: Final[int] = 512
MAX_DETAIL_CHARS: Final[int] = 256
MAX_COLLECTION_ITEMS: Final[int] = 10_000

FORBIDDEN_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "cosine",
        "distance",
        "embedding",
        "embedding_score",
        "embeddings",
        "knn",
        "nearest",
        "rank",
        "score",
        "scores",
        "similarity",
        "vector",
        "vectors",
    }
)

GENERATION_STAGE_ORDER: Final[tuple[str, ...]] = (
    "graph_bind",
    "snapshot_freshness",
    "subject_resolution",
    "static_recall",
    "call_successors",
    "event_successors",
    "cfg_successors",
    "dynamic_frontier",
    "completeness",
)

CALL_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphEdgeKind.CALLS.value,
        ProgramGraphEdgeKind.SUCCESSOR.value,
        ProgramGraphEdgeKind.MUTUAL_RECURSION.value,
    }
)
EVENT_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphEdgeKind.RAISES.value,
        ProgramGraphEdgeKind.WRITES_STATE.value,
        ProgramGraphEdgeKind.READS_STATE.value,
        ProgramGraphEdgeKind.EFFECT_OF.value,
        ProgramGraphEdgeKind.CATCHES.value,
    }
)
CFG_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphEdgeKind.CFG_NEXT.value,
        ProgramGraphEdgeKind.CFG_BRANCH.value,
        ProgramGraphEdgeKind.EXCEPTION_EDGE.value,
    }
)
UNKNOWN_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {ProgramGraphEdgeKind.UNRESOLVED_DYNAMIC.value}
)
SUBJECT_NODE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphNodeKind.FUNCTION.value,
        ProgramGraphNodeKind.TEST.value,
        ProgramGraphNodeKind.FIXTURE.value,
        ProgramGraphNodeKind.MODULE.value,
        ProgramGraphNodeKind.CALLSITE.value,
        ProgramGraphNodeKind.CFG_BLOCK.value,
        ProgramGraphNodeKind.CLASS.value,
    }
)
FRONTIER_REASON_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in DynamicFrontierReason
)
QUERY_FAMILY_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in QueryFamily
)
SUPPORTED_QUERY_FAMILIES: Final[frozenset[str]] = frozenset(
    {QueryFamily.NEXT_CALL.value, QueryFamily.NEXT_EVENT.value}
)


class StaticSuccessorError(ValueError):
    """Closed successor-generation contract violation."""


class StageStatus(str, Enum):
    SELECTED = "selected"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class SuccessorKind(str, Enum):
    CALL_TARGET = "call_target"
    NEXT_EVENT = "next_event"
    CFG_SUCCESSOR = "cfg_successor"
    UNKNOWN_DYNAMIC = "unknown_dynamic"


class SuccessorOrigin(str, Enum):
    SUCCESSOR_SET = "successor_set"
    CALL_EDGE = "call_edge"
    EVENT_EDGE = "event_edge"
    CFG_EDGE = "cfg_edge"
    DYNAMIC_FRONTIER = "dynamic_frontier"
    CALLSITE_HOP = "callsite_hop"


class CompletenessVerdict(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    IMPOSSIBLE = "impossible"


class FreshnessState(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


STAGE_STATUS_VALUES: Final[frozenset[str]] = frozenset(item.value for item in StageStatus)
SUCCESSOR_KIND_VALUES: Final[frozenset[str]] = frozenset(item.value for item in SuccessorKind)
SUCCESSOR_ORIGIN_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in SuccessorOrigin
)
COMPLETENESS_VALUES: Final[frozenset[str]] = frozenset(
    item.value for item in CompletenessVerdict
)
RESOLUTION_VALUES: Final[frozenset[str]] = frozenset(item.value for item in ResolutionStatus)

KIND_BY_EDGE: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{kind: SuccessorKind.CALL_TARGET.value for kind in CALL_EDGE_KINDS},
        **{kind: SuccessorKind.NEXT_EVENT.value for kind in EVENT_EDGE_KINDS},
        **{kind: SuccessorKind.CFG_SUCCESSOR.value for kind in CFG_EDGE_KINDS},
        **{kind: SuccessorKind.UNKNOWN_DYNAMIC.value for kind in UNKNOWN_EDGE_KINDS},
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounded_text(value: Any, name: str, *, empty: bool = False, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise StaticSuccessorError(f"{name} must be a string")
    text = value.strip()
    if not text and not empty:
        raise StaticSuccessorError(f"{name} must not be empty")
    if any(ch in text for ch in ("\x00", "\r", "\n")):
        raise StaticSuccessorError(f"{name} must not contain control characters")
    if len(text) > limit:
        raise StaticSuccessorError(f"{name} exceeds {limit} characters")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return _bounded_text(value, name)


def _enum_value(value: Any, allowed: frozenset[str], name: str) -> str:
    text = _bounded_text(getattr(value, "value", value), name)
    if text not in allowed:
        raise StaticSuccessorError(f"{name} must be one of {sorted(allowed)}")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise StaticSuccessorError(f"{name} must be a boolean")
    return value


def _unique_sorted(values: Iterable[Any], name: str, *, normalize) -> tuple[str, ...]:
    items = [normalize(item, name) for item in values]
    if len(items) > MAX_COLLECTION_ITEMS:
        raise StaticSuccessorError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(set(items)))
    return ordered


def _cid(value: Any, name: str) -> str:
    return _bounded_text(value, name, limit=MAX_TEXT_CHARS)


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return _cid(value, name)


def _reject_forbidden(mapping: Mapping[str, Any], name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise StaticSuccessorError(f"{name} must be an object")
    forbidden = set(mapping) & FORBIDDEN_IDENTITY_FIELDS
    if forbidden:
        raise StaticSuccessorError(
            f"{name} rejects ANN/score identity fields {sorted(forbidden)}"
        )


def _receipt_cid(payload: Mapping[str, Any]) -> str:
    _reject_forbidden(payload, "identity_payload")
    return content_identity(dict(payload))


def _attr(value: Any, name: str, default: Any = ()) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuccessorStageDecision:
    """One recorded planner/pruner stage and its closed status."""

    stage: str
    status: StageStatus | str
    reason_code: str
    detail: str = ""
    evidence_cids: Sequence[str] = ()

    SCHEMA: ClassVar[str] = SUCCESSOR_STAGE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _bounded_text(self.stage, "stage"))
        object.__setattr__(
            self, "status", _enum_value(self.status, STAGE_STATUS_VALUES, "status")
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "detail",
            _bounded_text(self.detail, "detail", empty=True, limit=MAX_DETAIL_CHARS),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _unique_sorted(self.evidence_cids, "evidence_cid", normalize=_cid),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stage": self.stage,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "evidence_cids": list(self.evidence_cids),
        }

    @property
    def stage_decision_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["stage_decision_cid"] = self.stage_decision_cid
        return payload


@dataclass(frozen=True, slots=True)
class SuccessorDecisionReceipt:
    """Deterministic record of every successor generation or prune stage."""

    snapshot_cid: str
    query_family: QueryFamily | str
    completeness: CompletenessVerdict | str
    stages: Sequence[SuccessorStageDecision]
    candidate_cids: Sequence[str] = ()
    pruned_cids: Sequence[str] = ()
    frontier_cid: str | None = None
    subject_node_cid: str | None = None
    parent_receipt_cid: str | None = None
    freshness: FreshnessState | str = FreshnessState.FRESH
    producer_id: str = PRODUCER_ID
    evidence_term: str = STATIC_SUCCESSORS_EVIDENCE

    SCHEMA: ClassVar[str] = SUCCESSOR_DECISION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = SUCCESSOR_DECISION_RECEIPT_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "snapshot_cid",
            "query_family",
            "completeness",
            "stages",
            "candidate_cids",
            "pruned_cids",
            "frontier_cid",
            "subject_node_cid",
            "parent_receipt_cid",
            "freshness",
            "producer_id",
            "evidence_term",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_cid", _cid(self.snapshot_cid, "snapshot_cid"))
        object.__setattr__(
            self,
            "query_family",
            _enum_value(self.query_family, QUERY_FAMILY_VALUES, "query_family"),
        )
        object.__setattr__(
            self,
            "completeness",
            _enum_value(self.completeness, COMPLETENESS_VALUES, "completeness"),
        )
        stages = tuple(self.stages)
        if not stages:
            raise StaticSuccessorError("receipts require at least one stage decision")
        names = [item.stage for item in stages]
        if len(names) != len(set(names)):
            raise StaticSuccessorError("stage names must be unique on a receipt")
        for item in stages:
            if not isinstance(item, SuccessorStageDecision):
                raise StaticSuccessorError("stages must contain SuccessorStageDecision values")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "candidate_cids",
            _unique_sorted(self.candidate_cids, "candidate_cid", normalize=_cid),
        )
        object.__setattr__(
            self,
            "pruned_cids",
            _unique_sorted(self.pruned_cids, "pruned_cid", normalize=_cid),
        )
        overlap = set(self.candidate_cids) & set(self.pruned_cids)
        if overlap:
            raise StaticSuccessorError(
                "retained candidate_cids and pruned_cids must be disjoint"
            )
        object.__setattr__(
            self, "frontier_cid", _optional_cid(self.frontier_cid, "frontier_cid")
        )
        object.__setattr__(
            self,
            "subject_node_cid",
            _optional_cid(self.subject_node_cid, "subject_node_cid"),
        )
        object.__setattr__(
            self,
            "parent_receipt_cid",
            _optional_cid(self.parent_receipt_cid, "parent_receipt_cid"),
        )
        object.__setattr__(
            self,
            "freshness",
            _enum_value(
                self.freshness,
                frozenset(item.value for item in FreshnessState),
                "freshness",
            ),
        )
        object.__setattr__(
            self, "producer_id", _bounded_text(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self, "evidence_term", _bounded_text(self.evidence_term, "evidence_term")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "snapshot_cid": self.snapshot_cid,
            "query_family": self.query_family,
            "completeness": self.completeness,
            "stages": [item.identity_payload() for item in self.stages],
            "candidate_cids": list(self.candidate_cids),
            "pruned_cids": list(self.pruned_cids),
            "frontier_cid": self.frontier_cid,
            "subject_node_cid": self.subject_node_cid,
            "parent_receipt_cid": self.parent_receipt_cid,
            "freshness": self.freshness,
            "producer_id": self.producer_id,
            "evidence_term": self.evidence_term,
        }

    @property
    def receipt_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def stage_map(self) -> dict[str, SuccessorStageDecision]:
        return {item.stage: item for item in self.stages}

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload


@dataclass(frozen=True, slots=True)
class StaticSuccessorCandidate:
    """One conservative next-call or next-event possibility."""

    kind: SuccessorKind | str
    subject_node_cid: str
    successor_node_cid: str
    successor_edge_cid: str
    resolution_status: ResolutionStatus | str
    origin: SuccessorOrigin | str
    unknown: bool = False
    reasons: Sequence[str] = ()
    unavailable_dimensions: Sequence[str] = ()

    SCHEMA: ClassVar[str] = STATIC_SUCCESSOR_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum_value(self.kind, SUCCESSOR_KIND_VALUES, "kind")
        )
        object.__setattr__(
            self, "subject_node_cid", _cid(self.subject_node_cid, "subject_node_cid")
        )
        object.__setattr__(
            self,
            "successor_node_cid",
            _cid(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(
            self,
            "successor_edge_cid",
            _cid(self.successor_edge_cid, "successor_edge_cid"),
        )
        object.__setattr__(
            self,
            "resolution_status",
            _enum_value(self.resolution_status, RESOLUTION_VALUES, "resolution_status"),
        )
        object.__setattr__(
            self, "origin", _enum_value(self.origin, SUCCESSOR_ORIGIN_VALUES, "origin")
        )
        unknown = _bool(self.unknown, "unknown")
        object.__setattr__(self, "unknown", unknown)
        object.__setattr__(
            self,
            "reasons",
            _unique_sorted(self.reasons, "reason", normalize=_bounded_text),
        )
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _unique_sorted(
                self.unavailable_dimensions, "unavailable_dimension", normalize=_bounded_text
            ),
        )
        if unknown and not self.reasons and not self.unavailable_dimensions:
            raise StaticSuccessorError(
                "unknown candidates require reasons or unavailable_dimensions"
            )
        if self.kind == SuccessorKind.UNKNOWN_DYNAMIC.value and not unknown:
            raise StaticSuccessorError("unknown_dynamic candidates must set unknown=true")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "subject_node_cid": self.subject_node_cid,
            "successor_node_cid": self.successor_node_cid,
            "successor_edge_cid": self.successor_edge_cid,
            "resolution_status": self.resolution_status,
            "origin": self.origin,
            "unknown": self.unknown,
            "reasons": list(self.reasons),
            "unavailable_dimensions": list(self.unavailable_dimensions),
        }

    @property
    def candidate_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["candidate_cid"] = self.candidate_cid
        return payload


@dataclass(frozen=True, slots=True)
class UnresolvedDynamicFrontier:
    """Accelerate-side explicit unknown; never encoded as absence."""

    unresolved_node_cids: Sequence[str] = ()
    unresolved_edge_cids: Sequence[str] = ()
    reasons: Sequence[str] = ()
    unavailable_dimensions: Sequence[str] = ()
    datasets_frontier_cids: Sequence[str] = ()
    widened: bool = False

    SCHEMA: ClassVar[str] = UNRESOLVED_DYNAMIC_FRONTIER_SCHEMA
    INTERFACE: ClassVar[str] = UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unresolved_node_cids",
            _unique_sorted(self.unresolved_node_cids, "unresolved_node_cid", normalize=_cid),
        )
        object.__setattr__(
            self,
            "unresolved_edge_cids",
            _unique_sorted(self.unresolved_edge_cids, "unresolved_edge_cid", normalize=_cid),
        )
        reasons = _unique_sorted(self.reasons, "reason", normalize=_bounded_text)
        illegal = [item for item in reasons if item not in FRONTIER_REASON_VALUES]
        if illegal:
            raise StaticSuccessorError(
                f"frontier reasons must be DynamicFrontierReason values, not {illegal}"
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _unique_sorted(
                self.unavailable_dimensions, "unavailable_dimension", normalize=_bounded_text
            ),
        )
        object.__setattr__(
            self,
            "datasets_frontier_cids",
            _unique_sorted(
                self.datasets_frontier_cids, "datasets_frontier_cid", normalize=_cid
            ),
        )
        object.__setattr__(self, "widened", _bool(self.widened, "widened"))
        open_frontier = bool(
            self.unresolved_node_cids
            or self.unresolved_edge_cids
            or self.reasons
            or self.unavailable_dimensions
        )
        if open_frontier and not self.reasons and not self.unavailable_dimensions:
            raise StaticSuccessorError(
                "open frontiers require reasons or unavailable_dimensions"
            )

    @property
    def open(self) -> bool:
        return bool(
            self.unresolved_node_cids
            or self.unresolved_edge_cids
            or self.reasons
            or self.unavailable_dimensions
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "unresolved_node_cids": list(self.unresolved_node_cids),
            "unresolved_edge_cids": list(self.unresolved_edge_cids),
            "reasons": list(self.reasons),
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "datasets_frontier_cids": list(self.datasets_frontier_cids),
            "widened": self.widened,
        }

    @property
    def frontier_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def union(
        self,
        *,
        node_cids: Iterable[str] = (),
        edge_cids: Iterable[str] = (),
        reasons: Iterable[str] = (),
        unavailable_dimensions: Iterable[str] = (),
        datasets_frontier_cids: Iterable[str] = (),
        mark_widened: bool = True,
    ) -> "UnresolvedDynamicFrontier":
        nodes = tuple(sorted(set(self.unresolved_node_cids) | set(node_cids)))
        edges = tuple(sorted(set(self.unresolved_edge_cids) | set(edge_cids)))
        next_reasons = tuple(sorted(set(self.reasons) | set(reasons)))
        dims = tuple(sorted(set(self.unavailable_dimensions) | set(unavailable_dimensions)))
        cited = tuple(sorted(set(self.datasets_frontier_cids) | set(datasets_frontier_cids)))
        grew = (
            nodes != self.unresolved_node_cids
            or edges != self.unresolved_edge_cids
            or next_reasons != self.reasons
            or dims != self.unavailable_dimensions
        )
        return UnresolvedDynamicFrontier(
            unresolved_node_cids=nodes,
            unresolved_edge_cids=edges,
            reasons=next_reasons,
            unavailable_dimensions=dims,
            datasets_frontier_cids=cited,
            widened=self.widened or (grew and mark_widened),
        )

    def subtract(
        self,
        *,
        node_cids: Iterable[str] = (),
        edge_cids: Iterable[str] = (),
        reasons: Iterable[str] = (),
        unavailable_dimensions: Iterable[str] = (),
    ) -> "UnresolvedDynamicFrontier":
        drop_nodes = set(node_cids)
        drop_edges = set(edge_cids)
        drop_reasons = set(reasons)
        drop_dims = set(unavailable_dimensions)
        remaining_nodes = tuple(
            item for item in self.unresolved_node_cids if item not in drop_nodes
        )
        remaining_edges = tuple(
            item for item in self.unresolved_edge_cids if item not in drop_edges
        )
        remaining_reasons = tuple(
            item for item in self.reasons if item not in drop_reasons
        )
        remaining_dims = tuple(
            item for item in self.unavailable_dimensions if item not in drop_dims
        )
        still_open = remaining_nodes or remaining_edges or remaining_reasons or remaining_dims
        if still_open and not remaining_reasons and not remaining_dims:
            remaining_reasons = (DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,)
        return UnresolvedDynamicFrontier(
            unresolved_node_cids=remaining_nodes,
            unresolved_edge_cids=remaining_edges,
            reasons=remaining_reasons,
            unavailable_dimensions=remaining_dims,
            datasets_frontier_cids=self.datasets_frontier_cids,
            widened=False,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["frontier_cid"] = self.frontier_cid
        payload["open"] = self.open
        return payload


@dataclass(frozen=True, slots=True)
class ProgramGraphCatalog:
    """Immutable view over a datasets program-graph snapshot and catalog."""

    snapshot: ProgramGraphSnapshot
    nodes: tuple[ProgramGraphNode, ...]
    edges: tuple[ProgramGraphEdge, ...]
    successor_sets: tuple[StaticSuccessorSet, ...] = ()
    frontiers: tuple[DynamicFrontierRecord, ...] = ()
    contract_states: tuple[Any, ...] = ()
    source_cids: tuple[tuple[str, str], ...] = ()
    environment_binding_cid: str | None = None
    _node_index: Mapping[str, ProgramGraphNode] = field(
        init=False, repr=False, default=MappingProxyType({})
    )
    _edge_index: Mapping[str, ProgramGraphEdge] = field(
        init=False, repr=False, default=MappingProxyType({})
    )
    _outgoing: Mapping[str, tuple[ProgramGraphEdge, ...]] = field(
        init=False, repr=False, default=MappingProxyType({})
    )
    _contained: Mapping[str, tuple[str, ...]] = field(
        init=False, repr=False, default=MappingProxyType({})
    )
    _successor_index: Mapping[str, StaticSuccessorSet] = field(
        init=False, repr=False, default=MappingProxyType({})
    )

    SCHEMA: ClassVar[str] = PROGRAM_GRAPH_CATALOG_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ProgramGraphSnapshot):
            raise StaticSuccessorError("catalog snapshot must be a ProgramGraphSnapshot")
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        if not nodes:
            raise StaticSuccessorError("catalog nodes must not be empty")
        node_index = {}
        for node in nodes:
            if not isinstance(node, ProgramGraphNode):
                raise StaticSuccessorError("catalog nodes must be ProgramGraphNode values")
            cid = node.program_graph_node_cid
            if cid in node_index:
                raise StaticSuccessorError("catalog node identities must be unique")
            node_index[cid] = node
        edge_index = {}
        for edge in edges:
            if not isinstance(edge, ProgramGraphEdge):
                raise StaticSuccessorError("catalog edges must be ProgramGraphEdge values")
            cid = edge.program_graph_edge_cid
            if cid in edge_index:
                raise StaticSuccessorError("catalog edge identities must be unique")
            if edge.source_node_cid not in node_index or edge.target_node_cid not in node_index:
                raise StaticSuccessorError("catalog edges must resolve against catalog nodes")
            edge_index[cid] = edge
        successor_index = {}
        for item in self.successor_sets:
            if not isinstance(item, StaticSuccessorSet):
                raise StaticSuccessorError(
                    "successor_sets must contain StaticSuccessorSet values"
                )
            successor_index[item.subject_node_cid] = item
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "successor_sets", tuple(self.successor_sets))
        object.__setattr__(self, "frontiers", tuple(self.frontiers))
        object.__setattr__(self, "contract_states", tuple(self.contract_states))
        object.__setattr__(self, "source_cids", tuple(self.source_cids))
        object.__setattr__(
            self,
            "environment_binding_cid",
            _optional_cid(self.environment_binding_cid, "environment_binding_cid")
            or self.snapshot.environment_binding_set_cid,
        )
        object.__setattr__(self, "_node_index", MappingProxyType(node_index))
        object.__setattr__(self, "_edge_index", MappingProxyType(edge_index))
        outgoing: dict[str, list[ProgramGraphEdge]] = {}
        incoming_contains: dict[str, list[str]] = {}
        for edge in edges:
            outgoing.setdefault(edge.source_node_cid, []).append(edge)
            if str(edge.edge_kind) == ProgramGraphEdgeKind.CONTAINS.value:
                incoming_contains.setdefault(edge.source_node_cid, []).append(
                    edge.target_node_cid
                )
        object.__setattr__(
            self,
            "_outgoing",
            MappingProxyType({key: tuple(value) for key, value in outgoing.items()}),
        )
        object.__setattr__(
            self,
            "_contained",
            MappingProxyType(
                {key: tuple(value) for key, value in incoming_contains.items()}
            ),
        )
        object.__setattr__(self, "_successor_index", MappingProxyType(successor_index))

    @classmethod
    def from_receipt(cls, receipt: Any) -> "ProgramGraphCatalog":
        return cls.from_parts(
            snapshot=_attr(receipt, "snapshot", None),
            nodes=tuple(_attr(receipt, "nodes", ())),
            edges=tuple(_attr(receipt, "edges", ())),
            successor_sets=tuple(_attr(receipt, "successor_sets", ())),
            frontiers=tuple(_attr(receipt, "frontiers", ())),
            contract_states=tuple(_attr(receipt, "contract_states", ())),
            source_cids=tuple(_attr(receipt, "source_cids", ())),
            environment_binding_cid=_attr(receipt, "environment_binding_cid", None),
        )

    @classmethod
    def from_parts(
        cls,
        *,
        snapshot: ProgramGraphSnapshot,
        nodes: Sequence[ProgramGraphNode],
        edges: Sequence[ProgramGraphEdge] = (),
        successor_sets: Sequence[StaticSuccessorSet] = (),
        frontiers: Sequence[DynamicFrontierRecord] = (),
        contract_states: Sequence[Any] = (),
        source_cids: Sequence[tuple[str, str]] = (),
        environment_binding_cid: str | None = None,
    ) -> "ProgramGraphCatalog":
        return cls(
            snapshot=snapshot,
            nodes=tuple(nodes),
            edges=tuple(edges),
            successor_sets=tuple(successor_sets),
            frontiers=tuple(frontiers),
            contract_states=tuple(contract_states),
            source_cids=tuple(source_cids),
            environment_binding_cid=environment_binding_cid,
        )

    @property
    def snapshot_cid(self) -> str:
        return self.snapshot.program_graph_snapshot_cid

    def node(self, node_cid: str) -> ProgramGraphNode:
        try:
            return self._node_index[node_cid]
        except KeyError as exc:
            raise StaticSuccessorError(f"unknown node {node_cid}") from exc

    def edge(self, edge_cid: str) -> ProgramGraphEdge:
        try:
            return self._edge_index[edge_cid]
        except KeyError as exc:
            raise StaticSuccessorError(f"unknown edge {edge_cid}") from exc

    def outgoing(self, node_cid: str) -> tuple[ProgramGraphEdge, ...]:
        return self._outgoing.get(node_cid, ())

    def contained(self, node_cid: str) -> tuple[str, ...]:
        return self._contained.get(node_cid, ())

    def successor_set_for(self, node_cid: str) -> StaticSuccessorSet | None:
        return self._successor_index.get(node_cid)

    def nodes_named(self, logical_name: str, *, kind: str | None = None) -> tuple[ProgramGraphNode, ...]:
        matches = [
            node
            for node in self.nodes
            if node.logical_name == logical_name
            and (kind is None or str(node.node_kind) == kind)
        ]
        return tuple(matches)

    def source_cid_map(self) -> dict[str, str]:
        return {path: cid for path, cid in self.source_cids}


@dataclass(frozen=True, slots=True)
class StaticSuccessorPlan:
    """Conservative successor set, frontier, and generation receipt."""

    catalog: ProgramGraphCatalog
    subject_node_cid: str | None
    query_family: str
    candidates: tuple[StaticSuccessorCandidate, ...]
    frontier: UnresolvedDynamicFrontier
    receipt: SuccessorDecisionReceipt
    completeness: str
    freshness: str

    SCHEMA: ClassVar[str] = STATIC_SUCCESSOR_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = STATIC_SUCCESSOR_PLAN_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ProgramGraphCatalog):
            raise StaticSuccessorError("plan catalog must be a ProgramGraphCatalog")
        if not isinstance(self.receipt, SuccessorDecisionReceipt):
            raise StaticSuccessorError("plan receipt must be a SuccessorDecisionReceipt")
        if not isinstance(self.frontier, UnresolvedDynamicFrontier):
            raise StaticSuccessorError("plan frontier must be UnresolvedDynamicFrontier")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "query_family", _enum_value(self.query_family, QUERY_FAMILY_VALUES, "query_family")
        )
        object.__setattr__(
            self,
            "completeness",
            _enum_value(self.completeness, COMPLETENESS_VALUES, "completeness"),
        )
        object.__setattr__(
            self,
            "freshness",
            _enum_value(
                self.freshness,
                frozenset(item.value for item in FreshnessState),
                "freshness",
            ),
        )

    @property
    def snapshot_cid(self) -> str:
        return self.catalog.snapshot_cid

    @property
    def known_candidates(self) -> tuple[StaticSuccessorCandidate, ...]:
        return tuple(item for item in self.candidates if not item.unknown)

    @property
    def unknown_candidates(self) -> tuple[StaticSuccessorCandidate, ...]:
        return tuple(item for item in self.candidates if item.unknown)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "snapshot_cid": self.snapshot_cid,
            "subject_node_cid": self.subject_node_cid,
            "query_family": self.query_family,
            "candidate_cids": [item.candidate_cid for item in self.candidates],
            "frontier_cid": self.frontier.frontier_cid,
            "receipt_cid": self.receipt.receipt_cid,
            "completeness": self.completeness,
            "freshness": self.freshness,
        }

    @property
    def plan_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_cid"] = self.plan_cid
        payload["candidates"] = [item.to_dict() for item in self.candidates]
        payload["frontier"] = self.frontier.to_dict()
        payload["receipt"] = self.receipt.to_dict()
        return payload


def as_program_graph_catalog(value: Any) -> ProgramGraphCatalog:
    """Accept a catalog or datasets ``ProgramGraphBuildReceipt``."""

    if isinstance(value, ProgramGraphCatalog):
        return value
    if isinstance(value, ProgramGraphSnapshot):
        raise StaticSuccessorError(
            "a snapshot CID catalog is insufficient; supply nodes and edges"
        )
    snapshot = _attr(value, "snapshot", None)
    nodes = _attr(value, "nodes", None)
    if snapshot is None or nodes is None:
        raise StaticSuccessorError(
            "successor generation requires a ProgramGraphCatalog or build receipt"
        )
    return ProgramGraphCatalog.from_receipt(value)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class StaticSuccessorPlanner:
    """Generate conservative next-call/event sets from a sealed program graph."""

    INTERFACE: ClassVar[str] = STATIC_SUCCESSOR_PLANNER_INTERFACE
    VERSION: ClassVar[str] = STATIC_SUCCESSOR_PLANNER_VERSION

    def generate(
        self,
        catalog: Any,
        *,
        subject_node_cid: str | None = None,
        subject_logical_name: str | None = None,
        subject_node_kind: str | None = None,
        query_family: QueryFamily | str = QueryFamily.NEXT_CALL,
        expected_snapshot_cid: str | None = None,
        expected_environment_binding_cid: str | None = None,
        expected_source_cids: Mapping[str, str] | None = None,
        policy_cid: str | None = None,
    ) -> StaticSuccessorPlan:
        del policy_cid  # cited by callers; generation is graph-bound, not policy-admitted
        stages: list[SuccessorStageDecision] = []
        family = _enum_value(query_family, QUERY_FAMILY_VALUES, "query_family")

        try:
            bound = as_program_graph_catalog(catalog)
            stages.append(
                _stage(
                    "graph_bind",
                    StageStatus.SELECTED,
                    "datasets_snapshot_bound",
                    detail="consumed datasets ProgramGraphSnapshot catalog",
                    evidence_cids=(bound.snapshot_cid,),
                )
            )
        except StaticSuccessorError as exc:
            empty_frontier = UnresolvedDynamicFrontier(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("graph",),
            )
            stages.append(
                _stage("graph_bind", StageStatus.REJECTED, "catalog_rejected", detail=str(exc)[:MAX_DETAIL_CHARS])
            )
            _fill_remaining_stages(stages, GENERATION_STAGE_ORDER, StageStatus.SKIPPED, "preceding_stage_rejected")
            receipt = _generation_receipt(
                snapshot_cid="unavailable",
                family=family,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                stages=stages,
                frontier=empty_frontier,
                freshness=FreshnessState.UNAVAILABLE.value,
            )
            raise StaticSuccessorError(str(exc)) from exc

        language = str(bound.snapshot.language)
        if language != ADMITTED_LANGUAGE:
            stages.append(
                _stage(
                    "snapshot_freshness",
                    StageStatus.UNAVAILABLE,
                    "language_unavailable",
                    detail=f"admitted language is {ADMITTED_LANGUAGE}",
                )
            )
            _fill_remaining_stages(stages, GENERATION_STAGE_ORDER, StageStatus.SKIPPED, "language_unavailable")
            frontier = UnresolvedDynamicFrontier(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("language",),
            )
            receipt = _generation_receipt(
                snapshot_cid=bound.snapshot_cid,
                family=family,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                stages=stages,
                frontier=frontier,
                freshness=FreshnessState.UNAVAILABLE.value,
            )
            return StaticSuccessorPlan(
                catalog=bound,
                subject_node_cid=None,
                query_family=family,
                candidates=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=FreshnessState.UNAVAILABLE.value,
            )

        freshness, freshness_stage = _evaluate_freshness(
            bound,
            expected_snapshot_cid=expected_snapshot_cid,
            expected_environment_binding_cid=expected_environment_binding_cid,
            expected_source_cids=expected_source_cids,
        )
        stages.append(freshness_stage)
        if freshness == FreshnessState.STALE.value:
            _fill_remaining_stages(stages, GENERATION_STAGE_ORDER, StageStatus.SKIPPED, "stale_graph")
            frontier = UnresolvedDynamicFrontier(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("stale_graph",),
            )
            receipt = _generation_receipt(
                snapshot_cid=bound.snapshot_cid,
                family=family,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                stages=stages,
                frontier=frontier,
                freshness=freshness,
            )
            return StaticSuccessorPlan(
                catalog=bound,
                subject_node_cid=None,
                query_family=family,
                candidates=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=freshness,
            )

        if family not in SUPPORTED_QUERY_FAMILIES:
            stages.append(
                _stage(
                    "subject_resolution",
                    StageStatus.REJECTED,
                    "query_family_unsupported",
                    detail=family,
                )
            )
            _fill_remaining_stages(stages, GENERATION_STAGE_ORDER, StageStatus.SKIPPED, "query_family_unsupported")
            frontier = UnresolvedDynamicFrontier(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("query_family",),
            )
            receipt = _generation_receipt(
                snapshot_cid=bound.snapshot_cid,
                family=family,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                stages=stages,
                frontier=frontier,
                freshness=freshness,
            )
            return StaticSuccessorPlan(
                catalog=bound,
                subject_node_cid=None,
                query_family=family,
                candidates=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=freshness,
            )

        subject, subject_stage = _resolve_subject(
            bound,
            subject_node_cid=subject_node_cid,
            subject_logical_name=subject_logical_name,
            subject_node_kind=subject_node_kind,
        )
        stages.append(subject_stage)
        if subject is None:
            _fill_remaining_stages(stages, GENERATION_STAGE_ORDER, StageStatus.SKIPPED, "subject_unresolved")
            frontier = UnresolvedDynamicFrontier(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("subject",),
            )
            receipt = _generation_receipt(
                snapshot_cid=bound.snapshot_cid,
                family=family,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                stages=stages,
                frontier=frontier,
                freshness=freshness,
            )
            return StaticSuccessorPlan(
                catalog=bound,
                subject_node_cid=None,
                query_family=family,
                candidates=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=freshness,
            )

        successor_set = bound.successor_set_for(subject.program_graph_node_cid)
        if successor_set is None:
            stages.append(
                _stage(
                    "static_recall",
                    StageStatus.UNAVAILABLE,
                    "successor_set_absent",
                    detail="walked catalog edges conservatively",
                )
            )
        else:
            stages.append(
                _stage(
                    "static_recall",
                    StageStatus.SELECTED,
                    "successor_set_recalled",
                    evidence_cids=(successor_set.static_successor_set_cid,),
                )
            )

        call_enabled = family == QueryFamily.NEXT_CALL.value
        event_enabled = family == QueryFamily.NEXT_EVENT.value
        collected: dict[str, StaticSuccessorCandidate] = {}
        unknown_nodes: set[str] = set()
        unknown_edges: set[str] = set()
        unknown_reasons: set[str] = set()
        unknown_dims: set[str] = set()

        call_count = 0
        event_count = 0
        cfg_count = 0
        seed_cids = _subject_seed_cids(bound, subject)
        for source_cid in seed_cids:
            for edge in bound.outgoing(source_cid):
                kind = KIND_BY_EDGE.get(str(edge.edge_kind))
                if kind is None:
                    continue
                origin = _origin_for(str(edge.edge_kind), source_cid, subject.program_graph_node_cid)
                candidate = _candidate_from_edge(
                    subject_node_cid=subject.program_graph_node_cid,
                    edge=edge,
                    target=bound.node(edge.target_node_cid),
                    kind=kind,
                    origin=origin,
                )
                if candidate.unknown:
                    unknown_nodes.add(candidate.successor_node_cid)
                    unknown_edges.add(candidate.successor_edge_cid)
                    unknown_reasons.update(candidate.reasons)
                    unknown_dims.update(candidate.unavailable_dimensions)
                    collected[candidate.candidate_cid] = candidate
                    continue
                if kind == SuccessorKind.CALL_TARGET.value:
                    if call_enabled:
                        collected[candidate.candidate_cid] = candidate
                        call_count += 1
                elif kind == SuccessorKind.NEXT_EVENT.value:
                    if event_enabled:
                        collected[candidate.candidate_cid] = candidate
                        event_count += 1
                elif kind == SuccessorKind.CFG_SUCCESSOR.value:
                    if call_enabled:
                        collected[candidate.candidate_cid] = candidate
                        cfg_count += 1
                else:
                    collected[candidate.candidate_cid] = candidate

        if call_enabled:
            stages.append(
                _stage(
                    "call_successors",
                    StageStatus.SELECTED if call_count else StageStatus.SKIPPED,
                    "call_edges_recalled" if call_count else "no_call_edges",
                )
            )
        else:
            stages.append(
                _stage("call_successors", StageStatus.SKIPPED, "query_family_excludes_calls")
            )
        if event_enabled:
            stages.append(
                _stage(
                    "event_successors",
                    StageStatus.SELECTED if event_count else StageStatus.SKIPPED,
                    "event_edges_recalled" if event_count else "no_event_edges",
                )
            )
        else:
            stages.append(
                _stage("event_successors", StageStatus.SKIPPED, "query_family_excludes_events")
            )
        if call_enabled:
            stages.append(
                _stage(
                    "cfg_successors",
                    StageStatus.SELECTED if cfg_count else StageStatus.SKIPPED,
                    "cfg_edges_recalled" if cfg_count else "no_cfg_edges",
                )
            )
        else:
            stages.append(
                _stage("cfg_successors", StageStatus.SKIPPED, "query_family_excludes_cfg")
            )

        frontier, frontier_stage = _collect_frontier(
            bound,
            subject=subject,
            seed_cids=seed_cids,
            unknown_nodes=unknown_nodes,
            unknown_edges=unknown_edges,
            unknown_reasons=unknown_reasons,
            unknown_dims=unknown_dims,
            successor_set=successor_set,
        )
        stages.append(frontier_stage)
        seed_set = set(seed_cids)
        for edge_cid in frontier.unresolved_edge_cids:
            try:
                edge = bound.edge(edge_cid)
            except StaticSuccessorError:
                continue
            if edge.source_node_cid not in seed_set:
                continue
            target = bound.node(edge.target_node_cid)
            candidate = _candidate_from_edge(
                subject_node_cid=subject.program_graph_node_cid,
                edge=edge,
                target=target,
                kind=SuccessorKind.UNKNOWN_DYNAMIC.value,
                origin=SuccessorOrigin.DYNAMIC_FRONTIER.value,
            )
            collected.setdefault(candidate.candidate_cid, candidate)

        candidates = tuple(sorted(collected.values(), key=lambda item: item.candidate_cid))
        snapshot_incomplete = bool(bound.snapshot.unavailable_dimensions)
        set_incomplete = successor_set is None or not successor_set.complete
        completeness = (
            CompletenessVerdict.INCOMPLETE.value
            if frontier.open or snapshot_incomplete or set_incomplete
            else CompletenessVerdict.COMPLETE.value
        )
        stages.append(
            _stage(
                "completeness",
                StageStatus.SELECTED,
                completeness,
                detail="unknown widens; absence is not completeness",
            )
        )
        receipt = _generation_receipt(
            snapshot_cid=bound.snapshot_cid,
            family=family,
            completeness=completeness,
            stages=stages,
            frontier=frontier,
            candidates=candidates,
            subject_node_cid=subject.program_graph_node_cid,
            freshness=freshness,
        )
        return StaticSuccessorPlan(
            catalog=bound,
            subject_node_cid=subject.program_graph_node_cid,
            query_family=family,
            candidates=candidates,
            frontier=frontier,
            receipt=receipt,
            completeness=completeness,
            freshness=freshness,
        )


def generate_static_successors(
    catalog: Any,
    *,
    subject_node_cid: str | None = None,
    subject_logical_name: str | None = None,
    subject_node_kind: str | None = None,
    query_family: QueryFamily | str = QueryFamily.NEXT_CALL,
    expected_snapshot_cid: str | None = None,
    expected_environment_binding_cid: str | None = None,
    expected_source_cids: Mapping[str, str] | None = None,
    policy_cid: str | None = None,
    planner: StaticSuccessorPlanner | None = None,
) -> StaticSuccessorPlan:
    """Public interface: conservative static successor generation."""

    active = planner or StaticSuccessorPlanner()
    return active.generate(
        catalog,
        subject_node_cid=subject_node_cid,
        subject_logical_name=subject_logical_name,
        subject_node_kind=subject_node_kind,
        query_family=query_family,
        expected_snapshot_cid=expected_snapshot_cid,
        expected_environment_binding_cid=expected_environment_binding_cid,
        expected_source_cids=expected_source_cids,
        policy_cid=policy_cid,
    )


# ---------------------------------------------------------------------------
# Internal generation helpers
# ---------------------------------------------------------------------------


def _stage(
    name: str,
    status: StageStatus | str,
    reason_code: str,
    *,
    detail: str = "",
    evidence_cids: Sequence[str] = (),
) -> SuccessorStageDecision:
    return SuccessorStageDecision(
        stage=name,
        status=status,
        reason_code=reason_code,
        detail=detail,
        evidence_cids=evidence_cids,
    )


def _fill_remaining_stages(
    stages: list[SuccessorStageDecision],
    order: Sequence[str],
    status: StageStatus,
    reason_code: str,
) -> None:
    present = {item.stage for item in stages}
    for name in order:
        if name not in present:
            stages.append(_stage(name, status, reason_code))


def _generation_receipt(
    *,
    snapshot_cid: str,
    family: str,
    completeness: str,
    stages: Sequence[SuccessorStageDecision],
    frontier: UnresolvedDynamicFrontier,
    candidates: Sequence[StaticSuccessorCandidate] = (),
    subject_node_cid: str | None = None,
    freshness: str = FreshnessState.FRESH.value,
) -> SuccessorDecisionReceipt:
    ordered = tuple(stages)
    present = {item.stage for item in ordered}
    missing = [name for name in GENERATION_STAGE_ORDER if name not in present]
    if missing:
        raise StaticSuccessorError(f"generation receipt missing stages {missing}")
    return SuccessorDecisionReceipt(
        snapshot_cid=snapshot_cid,
        query_family=family,
        completeness=completeness,
        stages=ordered,
        candidate_cids=tuple(item.candidate_cid for item in candidates),
        frontier_cid=frontier.frontier_cid,
        subject_node_cid=subject_node_cid,
        freshness=freshness,
        evidence_term=STATIC_SUCCESSORS_EVIDENCE,
    )


def _evaluate_freshness(
    catalog: ProgramGraphCatalog,
    *,
    expected_snapshot_cid: str | None,
    expected_environment_binding_cid: str | None,
    expected_source_cids: Mapping[str, str] | None,
) -> tuple[str, SuccessorStageDecision]:
    if (
        expected_snapshot_cid is None
        and expected_environment_binding_cid is None
        and not expected_source_cids
    ):
        return FreshnessState.UNKNOWN.value, _stage(
            "snapshot_freshness",
            StageStatus.SKIPPED,
            "no_expected_bindings",
            detail="freshness not asserted",
        )
    if expected_snapshot_cid is not None and expected_snapshot_cid != catalog.snapshot_cid:
        return FreshnessState.STALE.value, _stage(
            "snapshot_freshness",
            StageStatus.REJECTED,
            "stale_snapshot",
            detail="expected snapshot CID does not match catalog",
        )
    env_cid = catalog.environment_binding_cid or catalog.snapshot.environment_binding_set_cid
    if (
        expected_environment_binding_cid is not None
        and expected_environment_binding_cid != env_cid
    ):
        return FreshnessState.STALE.value, _stage(
            "snapshot_freshness",
            StageStatus.REJECTED,
            "stale_environment",
            detail="expected environment binding does not match catalog",
        )
    if expected_source_cids:
        current = catalog.source_cid_map()
        for path, cid in expected_source_cids.items():
            if current.get(path) != cid:
                return FreshnessState.STALE.value, _stage(
                    "snapshot_freshness",
                    StageStatus.REJECTED,
                    "stale_source",
                    detail=f"source binding drifted for {path[:80]}",
                )
    return FreshnessState.FRESH.value, _stage(
        "snapshot_freshness",
        StageStatus.SELECTED,
        "bindings_fresh",
        evidence_cids=(catalog.snapshot_cid,),
    )


def _resolve_subject(
    catalog: ProgramGraphCatalog,
    *,
    subject_node_cid: str | None,
    subject_logical_name: str | None,
    subject_node_kind: str | None,
) -> tuple[ProgramGraphNode | None, SuccessorStageDecision]:
    if subject_node_cid:
        try:
            node = catalog.node(subject_node_cid)
        except StaticSuccessorError:
            return None, _stage(
                "subject_resolution",
                StageStatus.REJECTED,
                "subject_node_missing",
            )
        return node, _stage(
            "subject_resolution",
            StageStatus.SELECTED,
            "subject_by_cid",
            evidence_cids=(node.program_graph_node_cid,),
        )
    if not subject_logical_name:
        return None, _stage(
            "subject_resolution",
            StageStatus.REJECTED,
            "subject_unspecified",
        )
    kind = _optional_text(subject_node_kind, "subject_node_kind")
    matches = catalog.nodes_named(subject_logical_name, kind=kind)
    if not matches and kind is None:
        preferred = [
            node
            for node in catalog.nodes_named(subject_logical_name)
            if str(node.node_kind) in SUBJECT_NODE_KINDS
        ]
        matches = tuple(preferred)
    if len(matches) != 1:
        return None, _stage(
            "subject_resolution",
            StageStatus.REJECTED,
            "subject_ambiguous" if len(matches) > 1 else "subject_missing",
            detail=subject_logical_name[:MAX_DETAIL_CHARS],
        )
    node = matches[0]
    return node, _stage(
        "subject_resolution",
        StageStatus.SELECTED,
        "subject_by_name",
        evidence_cids=(node.program_graph_node_cid,),
    )


def _subject_seed_cids(catalog: ProgramGraphCatalog, subject: ProgramGraphNode) -> tuple[str, ...]:
    seeds = [subject.program_graph_node_cid]
    for contained in catalog.contained(subject.program_graph_node_cid):
        node = catalog.node(contained)
        if str(node.node_kind) in {
            ProgramGraphNodeKind.CALLSITE.value,
            ProgramGraphNodeKind.CFG_BLOCK.value,
            ProgramGraphNodeKind.UNRESOLVED_DYNAMIC.value,
        }:
            seeds.append(contained)
    return tuple(dict.fromkeys(seeds))


def _origin_for(edge_kind: str, source_cid: str, subject_cid: str) -> str:
    if edge_kind in UNKNOWN_EDGE_KINDS:
        return SuccessorOrigin.DYNAMIC_FRONTIER.value
    if source_cid != subject_cid:
        if edge_kind in CALL_EDGE_KINDS:
            return SuccessorOrigin.CALLSITE_HOP.value
        if edge_kind in EVENT_EDGE_KINDS:
            return SuccessorOrigin.EVENT_EDGE.value
        if edge_kind in CFG_EDGE_KINDS:
            return SuccessorOrigin.CFG_EDGE.value
    if edge_kind in CALL_EDGE_KINDS:
        return SuccessorOrigin.CALL_EDGE.value
    if edge_kind in EVENT_EDGE_KINDS:
        return SuccessorOrigin.EVENT_EDGE.value
    if edge_kind in CFG_EDGE_KINDS:
        return SuccessorOrigin.CFG_EDGE.value
    return SuccessorOrigin.SUCCESSOR_SET.value


def _frontier_reasons_from(values: Iterable[str]) -> tuple[str, ...]:
    reasons = [item for item in values if item in FRONTIER_REASON_VALUES]
    return tuple(sorted(set(reasons)))


def _candidate_from_edge(
    *,
    subject_node_cid: str,
    edge: ProgramGraphEdge,
    target: ProgramGraphNode,
    kind: str,
    origin: str,
) -> StaticSuccessorCandidate:
    unresolved = (
        is_unresolved_dynamic_edge(edge)
        or is_unresolved_dynamic_node(target)
        or kind == SuccessorKind.UNKNOWN_DYNAMIC.value
    )
    reasons = _frontier_reasons_from(
        [
            *edge.unavailable_dimensions,
            *target.unavailable_dimensions,
        ]
    )
    dims = tuple(
        sorted(set(edge.unavailable_dimensions) | set(target.unavailable_dimensions))
    )
    if unresolved and not reasons:
        reasons = (DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,)
    if unresolved and not dims:
        dims = ("incomplete_analysis",)
    status = str(edge.resolution_status)
    if unresolved and status == ResolutionStatus.DEFINITE.value:
        status = ResolutionStatus.UNRESOLVED.value
    return StaticSuccessorCandidate(
        kind=SuccessorKind.UNKNOWN_DYNAMIC.value if unresolved else kind,
        subject_node_cid=subject_node_cid,
        successor_node_cid=target.program_graph_node_cid,
        successor_edge_cid=edge.program_graph_edge_cid,
        resolution_status=status,
        origin=origin,
        unknown=unresolved,
        reasons=reasons if unresolved else (),
        unavailable_dimensions=dims if unresolved else (),
    )


def _collect_frontier(
    catalog: ProgramGraphCatalog,
    *,
    subject: ProgramGraphNode,
    seed_cids: Sequence[str],
    unknown_nodes: set[str],
    unknown_edges: set[str],
    unknown_reasons: set[str],
    unknown_dims: set[str],
    successor_set: StaticSuccessorSet | None,
) -> tuple[UnresolvedDynamicFrontier, SuccessorStageDecision]:
    neighborhood = set(seed_cids) | set(unknown_nodes)
    for seed in seed_cids:
        node = catalog.node(seed)
        if is_unresolved_dynamic_node(node):
            unknown_nodes.add(node.program_graph_node_cid)
            unknown_dims.update(node.unavailable_dimensions)
            neighborhood.add(node.program_graph_node_cid)
        for edge in catalog.outgoing(seed):
            neighborhood.add(edge.target_node_cid)
            if is_unresolved_dynamic_edge(edge):
                unknown_edges.add(edge.program_graph_edge_cid)
                unknown_nodes.add(edge.target_node_cid)
                neighborhood.add(edge.target_node_cid)
                unknown_dims.update(edge.unavailable_dimensions)
                unknown_reasons.update(
                    item for item in edge.unavailable_dimensions if item in FRONTIER_REASON_VALUES
                )
    datasets_cids: list[str] = []
    for record in catalog.frontiers:
        datasets_cids.append(record.dynamic_frontier_record_cid)
        local_nodes = [cid for cid in record.unresolved_node_cids if cid in neighborhood]
        local_edges = []
        for edge_cid in record.unresolved_edge_cids:
            try:
                edge = catalog.edge(edge_cid)
            except StaticSuccessorError:
                continue
            if edge.source_node_cid in neighborhood or edge.target_node_cid in neighborhood:
                local_edges.append(edge_cid)
                unknown_nodes.add(edge.target_node_cid)
        if local_nodes or local_edges:
            unknown_nodes.update(local_nodes)
            unknown_edges.update(local_edges)
            unknown_reasons.update(record.reasons)
            unknown_dims.update(record.unavailable_dimensions)
    if successor_set is not None and not successor_set.complete:
        unknown_dims.update(successor_set.unavailable_dimensions)
    unknown_dims.update(subject.unavailable_dimensions)
    unknown_dims.update(catalog.snapshot.unavailable_dimensions)
    reasons = _frontier_reasons_from(unknown_reasons)
    if (unknown_nodes or unknown_edges or unknown_dims) and not reasons:
        reasons = (DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,)
    if (unknown_nodes or unknown_edges or reasons) and not unknown_dims:
        unknown_dims.add("incomplete_analysis")
    frontier = UnresolvedDynamicFrontier(
        unresolved_node_cids=tuple(unknown_nodes),
        unresolved_edge_cids=tuple(unknown_edges),
        reasons=reasons,
        unavailable_dimensions=tuple(unknown_dims),
        datasets_frontier_cids=tuple(datasets_cids),
        widened=False,
    )
    if frontier.open:
        return frontier, _stage(
            "dynamic_frontier",
            StageStatus.SELECTED,
            "unknown_preserved",
            evidence_cids=frontier.datasets_frontier_cids,
        )
    return frontier, _stage(
        "dynamic_frontier",
        StageStatus.SKIPPED,
        "no_unresolved_dynamic",
    )


__all__ = [
    "ADMITTED_LANGUAGE",
    "CALL_EDGE_KINDS",
    "CFG_EDGE_KINDS",
    "CompletenessVerdict",
    "EVENT_EDGE_KINDS",
    "FreshnessState",
    "GENERATION_STAGE_ORDER",
    "IMPORT_SCAN_PERFORMED",
    "ProgramGraphCatalog",
    "STATIC_SUCCESSOR_PLANNER_INTERFACE",
    "STATIC_SUCCESSORS_EVIDENCE",
    "SYMBOLIC_PRUNING_EVIDENCE",
    "SUCCESSOR_DECISION_RECEIPT_INTERFACE",
    "StageStatus",
    "StaticSuccessorCandidate",
    "StaticSuccessorError",
    "StaticSuccessorPlan",
    "StaticSuccessorPlanner",
    "SuccessorDecisionReceipt",
    "SuccessorKind",
    "SuccessorOrigin",
    "SuccessorStageDecision",
    "UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE",
    "UnresolvedDynamicFrontier",
    "as_program_graph_catalog",
    "generate_static_successors",
]
