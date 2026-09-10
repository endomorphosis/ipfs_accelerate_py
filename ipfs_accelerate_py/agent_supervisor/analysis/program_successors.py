"""Conservative static successor generation for program-world planning.

Interface: ``StaticSuccessorPlanner@1``
Evidence: ``sawm/static-successors@1``

This module is the accelerator operational planner for SAWM-017.  It consumes
datasets ``ProgramGraphSnapshot@1``, ``StaticSuccessorSet@1``, and
``DynamicFrontierRecord@1`` records and emits conservative next-call/event
candidate sets.  Datasets remains the semantic authority for graph meaning;
this planner never mints a second scanner, solver, or router and never mutates
operational state.

Normative constraints:

* Static possibilities are recalled exactly and may only grow.
* Explicit unknown (reflection, dynamic dispatch, plugins, native calls,
  solver unknowns) stays explicit; absence is not a closed frontier.
* Unknown widens.  It never shrinks a candidate set.
* Every generation stage is recorded as selected, skipped, unavailable,
  rejected, or escalated.
* Stale snapshot, graph, or environment bindings fail closed.
* Importing this module performs no I/O and starts no analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    ADMITTED_LANGUAGES,
    UNAVAILABLE_LANGUAGES,
    CallsiteRecord,
    ContractStateRecord,
    DynamicFrontierReason,
    DynamicFrontierRecord,
    FunctionSymbolRecord,
    ProgramGraphEdge,
    ProgramGraphEdgeKind,
    ProgramGraphError,
    ProgramGraphNode,
    ProgramGraphNodeKind,
    ProgramGraphSnapshot,
    ProofObligationGraph,
    ResolutionStatus,
    StaticSuccessorSet,
    is_unresolved_dynamic_edge,
    is_unresolved_dynamic_node,
    verify_program_graph_catalog,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_transition import (
    QueryFamily,
)

from ..proof.formal_verification_contracts import content_identity


STATIC_SUCCESSOR_PLANNER_INTERFACE: Final[str] = "StaticSuccessorPlanner@1"
UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE: Final[str] = "UnresolvedDynamicFrontier@1"
SUCCESSOR_DECISION_RECEIPT_INTERFACE: Final[str] = "SuccessorDecisionReceipt@1"
SAWM_STATIC_SUCCESSORS_EVIDENCE: Final[str] = "sawm/static-successors@1"

STATIC_SUCCESSOR_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/static-successor-plan@1"
)
SUCCESSOR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/successor-candidate@1"
)
UNRESOLVED_DYNAMIC_FRONTIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/unresolved-dynamic-frontier@1"
)
SUCCESSOR_DECISION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/successor-decision-receipt@1"
)

PRODUCER_ID: Final[str] = "program-successors@1"
PLANNER_VERSION: Final[str] = "1"

MAX_CANDIDATES: Final[int] = 4_096
MAX_RECEIPTS: Final[int] = 256
MAX_TEXT_CHARS: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 100_000

NEXT_CALL_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphEdgeKind.CALLS.value,
        ProgramGraphEdgeKind.SUCCESSOR.value,
        ProgramGraphEdgeKind.MUTUAL_RECURSION.value,
    }
)
NEXT_EVENT_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphEdgeKind.CFG_NEXT.value,
        ProgramGraphEdgeKind.CFG_BRANCH.value,
        ProgramGraphEdgeKind.EXCEPTION_EDGE.value,
        ProgramGraphEdgeKind.RAISES.value,
        ProgramGraphEdgeKind.CATCHES.value,
    }
)
SUBJECT_NODE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ProgramGraphNodeKind.FUNCTION.value,
        ProgramGraphNodeKind.TEST.value,
        ProgramGraphNodeKind.FIXTURE.value,
        ProgramGraphNodeKind.MODULE.value,
        ProgramGraphNodeKind.CALLSITE.value,
        ProgramGraphNodeKind.CFG_BLOCK.value,
    }
)
FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "cosine",
        "distance",
        "embedding",
        "embedding_score",
        "embeddings",
        "knn",
        "nearest",
        "rank_score",
        "score",
        "scores",
        "similarity",
        "vector",
        "vectors",
    }
)


class StaticSuccessorError(ValueError):
    """Frozen successor inputs cannot produce a trustworthy plan."""


class StaticSuccessorBoundsError(StaticSuccessorError):
    """A successor plan exceeded a deterministic compactness bound."""


class StaleProgramGraphError(StaticSuccessorError):
    """Snapshot, graph, or environment bindings are not current."""


class SuccessorStage(str, Enum):
    """Closed generation and pruning stage vocabulary."""

    FRESHNESS = "freshness"
    LANGUAGE = "language"
    CATALOG = "catalog"
    STATIC_RECALL = "static_recall"
    CALLSITE_EXPANSION = "callsite_expansion"
    CFG_EVENT_EXPANSION = "cfg_event_expansion"
    DYNAMIC_FRONTIER = "dynamic_frontier"
    TYPE = "type"
    EFFECT = "effect"
    PATH_CONDITION = "path_condition"
    CONTRACT = "contract"
    CAPABILITY = "capability"
    ABSTRACT_INTERPRETATION = "abstract_interpretation"
    SOLVER = "solver"
    RESIDUAL_ESCALATION = "residual_escalation"
    COMPLETENESS = "completeness"


class SuccessorDisposition(str, Enum):
    """Closed outcome for one recorded stage."""

    SELECTED = "selected"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class SuccessorKind(str, Enum):
    """Closed successor candidate families."""

    NEXT_CALL = "next_call"
    NEXT_EVENT = "next_event"
    UNRESOLVED = "unresolved"


class SuccessorOrigin(str, Enum):
    """Closed origin of one conservative successor candidate."""

    STATIC_SUCCESSOR_SET = "static_successor_set"
    CALLS_EDGE = "calls_edge"
    SUCCESSOR_EDGE = "successor_edge"
    MUTUAL_RECURSION_EDGE = "mutual_recursion_edge"
    CFG_NEXT_EDGE = "cfg_next_edge"
    CFG_BRANCH_EDGE = "cfg_branch_edge"
    EXCEPTION_EDGE = "exception_edge"
    RAISES_EDGE = "raises_edge"
    CATCHES_EDGE = "catches_edge"
    CALLSITE_RECORD = "callsite_record"
    UNRESOLVED_DYNAMIC_EDGE = "unresolved_dynamic_edge"
    UNRESOLVED_DYNAMIC_NODE = "unresolved_dynamic_node"


class SuccessorReasonCode(str, Enum):
    """Closed reason codes for stage receipts."""

    GRAPH_FRESH = "graph_fresh"
    STALE_GRAPH = "stale_graph"
    STALE_SNAPSHOT = "stale_snapshot"
    STALE_ENVIRONMENT = "stale_environment"
    LANGUAGE_ADMITTED = "language_admitted"
    LANGUAGE_UNAVAILABLE = "language_unavailable"
    CATALOG_VERIFIED = "catalog_verified"
    CATALOG_MISMATCH = "catalog_mismatch"
    SUBJECT_MISSING = "subject_missing"
    STATIC_RECALL = "static_recall"
    NO_SUCCESSOR_SET = "no_successor_set"
    CALLSITE_EXPANDED = "callsite_expanded"
    CFG_EXPANDED = "cfg_expanded"
    FAMILY_SKIPPED = "family_skipped"
    DYNAMIC_FRONTIER_WIDENED = "dynamic_frontier_widened"
    FRONTIER_EMPTY = "frontier_empty"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    IMPOSSIBLE = "impossible"
    INCOMPATIBLE_TYPE = "incompatible_type"
    INCOMPATIBLE_EFFECT = "incompatible_effect"
    UNSAT_PATH = "unsat_path"
    CONTRACT_UNSAT = "contract_unsat"
    CONTRACT_UNKNOWN = "contract_unknown"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    EMPTY_ABSTRACT_STATE = "empty_abstract_state"
    ABSTRACT_INCOMPLETE = "abstract_incomplete"
    SOLVER_UNSAT = "solver_unsat"
    SOLVER_UNKNOWN = "solver_unknown"
    SOLVER_TIMEOUT = "solver_timeout"
    NON_AUTHORITATIVE = "non_authoritative"
    NO_EVIDENCE = "no_evidence"
    RESIDUAL_UNKNOWN = "residual_unknown"
    NO_RESIDUAL = "no_residual"
    FRESHNESS_BLOCKED = "freshness_blocked"
    UNKNOWN_WIDENED = "unknown_widened"
    CATALOG_SKIPPED = "catalog_skipped"


GENERATION_STAGE_ORDER: Final[tuple[SuccessorStage, ...]] = (
    SuccessorStage.FRESHNESS,
    SuccessorStage.LANGUAGE,
    SuccessorStage.CATALOG,
    SuccessorStage.STATIC_RECALL,
    SuccessorStage.CALLSITE_EXPANSION,
    SuccessorStage.CFG_EVENT_EXPANSION,
    SuccessorStage.DYNAMIC_FRONTIER,
    SuccessorStage.RESIDUAL_ESCALATION,
    SuccessorStage.COMPLETENESS,
)

_EDGE_KIND_TO_ORIGIN: Final[Mapping[str, SuccessorOrigin]] = MappingProxyType(
    {
        ProgramGraphEdgeKind.CALLS.value: SuccessorOrigin.CALLS_EDGE,
        ProgramGraphEdgeKind.SUCCESSOR.value: SuccessorOrigin.SUCCESSOR_EDGE,
        ProgramGraphEdgeKind.MUTUAL_RECURSION.value: SuccessorOrigin.MUTUAL_RECURSION_EDGE,
        ProgramGraphEdgeKind.CFG_NEXT.value: SuccessorOrigin.CFG_NEXT_EDGE,
        ProgramGraphEdgeKind.CFG_BRANCH.value: SuccessorOrigin.CFG_BRANCH_EDGE,
        ProgramGraphEdgeKind.EXCEPTION_EDGE.value: SuccessorOrigin.EXCEPTION_EDGE,
        ProgramGraphEdgeKind.RAISES.value: SuccessorOrigin.RAISES_EDGE,
        ProgramGraphEdgeKind.CATCHES.value: SuccessorOrigin.CATCHES_EDGE,
        ProgramGraphEdgeKind.UNRESOLVED_DYNAMIC.value: (
            SuccessorOrigin.UNRESOLVED_DYNAMIC_EDGE
        ),
    }
)
_EDGE_KIND_TO_CANDIDATE_KIND: Final[Mapping[str, SuccessorKind]] = MappingProxyType(
    {
        **{kind: SuccessorKind.NEXT_CALL for kind in NEXT_CALL_EDGE_KINDS},
        **{kind: SuccessorKind.NEXT_EVENT for kind in NEXT_EVENT_EDGE_KINDS},
        ProgramGraphEdgeKind.UNRESOLVED_DYNAMIC.value: SuccessorKind.UNRESOLVED,
    }
)


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 16:
        raise StaticSuccessorError("successor payload exceeds depth bound")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise StaticSuccessorError("successor payloads reject floating-point values")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        forbidden = set(value) & FORBIDDEN_FIELD_MARKERS
        if forbidden:
            raise StaticSuccessorError(
                "successor payloads reject non-semantic fields "
                + ", ".join(sorted(forbidden))
            )
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise StaticSuccessorError(
        f"successor payload contains unsupported type {type(value).__name__}"
    )


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise StaticSuccessorError(f"{name} must be a string")
    text = value.strip()
    if not empty and not text:
        raise StaticSuccessorError(f"{name} must be a nonempty string")
    if len(text) > MAX_TEXT_CHARS:
        raise StaticSuccessorBoundsError(f"{name} exceeds text bound")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise StaticSuccessorError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise StaticSuccessorError(f"{name} has unsupported value {value!r}") from exc


def _sorted_unique(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StaticSuccessorError(f"{name} must be a sequence")
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = _text(item, name, empty=True)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) > MAX_COLLECTION_ITEMS:
            raise StaticSuccessorBoundsError(f"{name} exceeds collection bound")
    return tuple(sorted(result))


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _identity_cid(payload: Mapping[str, Any]) -> str:
    return content_identity(_plain(payload))


def _query_families(value: Any) -> tuple[str, ...]:
    if value is None:
        return (QueryFamily.NEXT_CALL.value, QueryFamily.NEXT_EVENT.value)
    if isinstance(value, (list, tuple)):
        families = tuple(_enum(item, QueryFamily, "query_family") for item in value)
        return tuple(sorted(set(families)))
    family = _enum(value, QueryFamily, "query_family")
    return (family,)


def _record_cid(record: Any) -> str:
    field_name = getattr(record, "CID_FIELD", None)
    if not isinstance(field_name, str):
        raise StaticSuccessorError("catalog record is missing a CID field")
    return str(getattr(record, field_name))


def _coerce_record(value: Any, cls: type, name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        cid_field = getattr(cls, "CID_FIELD", None)
        if "schema" in payload and cid_field in payload:
            return cls.from_dict(payload)
        payload.pop("schema", None)
        if cid_field:
            payload.pop(cid_field, None)
        return cls(**payload)
    raise StaticSuccessorError(f"{name} must be {cls.__name__} or a mapping")


def _coerce_sequence(values: Any, cls: type, name: str) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise StaticSuccessorError(f"{name} must be a sequence")
    return tuple(_coerce_record(item, cls, name) for item in values)


def _index_by_cid(records: Sequence[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in records:
        cid = _record_cid(record)
        if cid in indexed:
            raise StaticSuccessorError("catalog identities must be unique")
        indexed[cid] = record
    return indexed


def _empty_frontier(*, language: str) -> "UnresolvedDynamicFrontier":
    return UnresolvedDynamicFrontier(
        language=language,
        unresolved_node_cids=(),
        unresolved_edge_cids=(),
        reasons=(),
        unavailable_dimensions=(),
        source_frontier_cids=(),
        widened=False,
    )


@dataclass(frozen=True, slots=True)
class SuccessorDecisionReceipt:
    """Immutable record of one selected/skipped/unavailable/rejected/escalated stage."""

    stage: SuccessorStage | str
    disposition: SuccessorDisposition | str
    reason_code: SuccessorReasonCode | str
    candidate_cids_in: Sequence[str] = ()
    candidate_cids_out: Sequence[str] = ()
    removed_cids: Sequence[str] = ()
    unknown_cids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    unavailable_dimensions: Sequence[str] = ()
    widened: bool = False
    diagnostic: str = ""

    SCHEMA: ClassVar[str] = SUCCESSOR_DECISION_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = SUCCESSOR_DECISION_RECEIPT_INTERFACE
    CID_FIELD: ClassVar[str] = "receipt_cid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(self.stage, SuccessorStage, "stage"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SuccessorDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _enum(self.reason_code, SuccessorReasonCode, "reason_code"),
        )
        object.__setattr__(
            self,
            "candidate_cids_in",
            _sorted_unique(self.candidate_cids_in, "candidate_cids_in"),
        )
        object.__setattr__(
            self,
            "candidate_cids_out",
            _sorted_unique(self.candidate_cids_out, "candidate_cids_out"),
        )
        object.__setattr__(
            self, "removed_cids", _sorted_unique(self.removed_cids, "removed_cids")
        )
        object.__setattr__(
            self, "unknown_cids", _sorted_unique(self.unknown_cids, "unknown_cids")
        )
        object.__setattr__(
            self, "evidence_cids", _sorted_unique(self.evidence_cids, "evidence_cids")
        )
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _sorted_unique(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(self, "widened", _bool(self.widened, "widened"))
        object.__setattr__(
            self, "diagnostic", _text(self.diagnostic, "diagnostic", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stage": self.stage,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "candidate_cids_in": list(self.candidate_cids_in),
            "candidate_cids_out": list(self.candidate_cids_out),
            "removed_cids": list(self.removed_cids),
            "unknown_cids": list(self.unknown_cids),
            "evidence_cids": list(self.evidence_cids),
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "widened": self.widened,
            "diagnostic": self.diagnostic,
        }

    @property
    def receipt_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["receipt_cid"] = self.receipt_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuccessorDecisionReceipt":
        payload = dict(data)
        claimed = payload.pop("receipt_cid", None)
        payload.pop("schema", None)
        result = cls(**payload)
        if claimed is not None and claimed != result.receipt_cid:
            raise StaticSuccessorError("successor decision receipt identity does not match")
        return result


def record_stage(
    stage: SuccessorStage | str,
    disposition: SuccessorDisposition | str,
    reason_code: SuccessorReasonCode | str,
    *,
    candidate_cids_in: Sequence[str] = (),
    candidate_cids_out: Sequence[str] = (),
    removed_cids: Sequence[str] = (),
    unknown_cids: Sequence[str] = (),
    evidence_cids: Sequence[str] = (),
    unavailable_dimensions: Sequence[str] = (),
    widened: bool = False,
    diagnostic: str = "",
) -> SuccessorDecisionReceipt:
    """Construct one deterministic stage receipt."""

    return SuccessorDecisionReceipt(
        stage=stage,
        disposition=disposition,
        reason_code=reason_code,
        candidate_cids_in=candidate_cids_in,
        candidate_cids_out=candidate_cids_out,
        removed_cids=removed_cids,
        unknown_cids=unknown_cids,
        evidence_cids=evidence_cids,
        unavailable_dimensions=unavailable_dimensions,
        widened=widened,
        diagnostic=diagnostic,
    )


@dataclass(frozen=True, slots=True)
class SuccessorCandidate:
    """One conservative next-call, next-event, or unresolved successor."""

    kind: SuccessorKind | str
    subject_node_cid: str
    target_node_cid: str
    origin: SuccessorOrigin | str
    edge_cid: str | None = None
    resolution_status: str = ResolutionStatus.DEFINITE.value
    unknown: bool = False
    complete: bool = False
    unavailable_dimensions: Sequence[str] = ()

    SCHEMA: ClassVar[str] = SUCCESSOR_CANDIDATE_SCHEMA
    CID_FIELD: ClassVar[str] = "candidate_cid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, SuccessorKind, "kind"))
        object.__setattr__(
            self, "subject_node_cid", _text(self.subject_node_cid, "subject_node_cid")
        )
        object.__setattr__(
            self, "target_node_cid", _text(self.target_node_cid, "target_node_cid")
        )
        object.__setattr__(
            self, "origin", _enum(self.origin, SuccessorOrigin, "origin")
        )
        object.__setattr__(self, "edge_cid", _optional_text(self.edge_cid, "edge_cid"))
        object.__setattr__(
            self,
            "resolution_status",
            _enum(self.resolution_status, ResolutionStatus, "resolution_status"),
        )
        object.__setattr__(self, "unknown", _bool(self.unknown, "unknown"))
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _sorted_unique(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        if self.unknown and self.complete:
            raise StaticSuccessorError("unknown successors cannot be marked complete")
        if not self.complete and not self.unavailable_dimensions and not self.unknown:
            object.__setattr__(
                self, "unavailable_dimensions", ("incomplete_analysis",)
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "subject_node_cid": self.subject_node_cid,
            "target_node_cid": self.target_node_cid,
            "origin": self.origin,
            "edge_cid": self.edge_cid,
            "resolution_status": self.resolution_status,
            "unknown": self.unknown,
            "complete": self.complete,
            "unavailable_dimensions": list(self.unavailable_dimensions),
        }

    @property
    def candidate_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["candidate_cid"] = self.candidate_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuccessorCandidate":
        payload = dict(data)
        claimed = payload.pop("candidate_cid", None)
        payload.pop("schema", None)
        result = cls(**payload)
        if claimed is not None and claimed != result.candidate_cid:
            raise StaticSuccessorError("successor candidate identity does not match")
        return result


@dataclass(frozen=True, slots=True)
class UnresolvedDynamicFrontier:
    """Operational unresolved-dynamic frontier; unknown always stays explicit."""

    language: str
    unresolved_node_cids: Sequence[str] = ()
    unresolved_edge_cids: Sequence[str] = ()
    reasons: Sequence[str] = ()
    unavailable_dimensions: Sequence[str] = ()
    source_frontier_cids: Sequence[str] = ()
    widened: bool = False

    SCHEMA: ClassVar[str] = UNRESOLVED_DYNAMIC_FRONTIER_SCHEMA
    INTERFACE: ClassVar[str] = UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE
    CID_FIELD: ClassVar[str] = "frontier_cid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(
            self,
            "unresolved_node_cids",
            _sorted_unique(self.unresolved_node_cids, "unresolved_node_cids"),
        )
        object.__setattr__(
            self,
            "unresolved_edge_cids",
            _sorted_unique(self.unresolved_edge_cids, "unresolved_edge_cids"),
        )
        reasons = tuple(
            _enum(item, DynamicFrontierReason, "reason") for item in self.reasons
        )
        object.__setattr__(self, "reasons", _sorted_unique(reasons, "reasons"))
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _sorted_unique(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(
            self,
            "source_frontier_cids",
            _sorted_unique(self.source_frontier_cids, "source_frontier_cids"),
        )
        object.__setattr__(self, "widened", _bool(self.widened, "widened"))

    @property
    def empty(self) -> bool:
        return not (
            self.unresolved_node_cids
            or self.unresolved_edge_cids
            or self.reasons
            or self.unavailable_dimensions
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "language": self.language,
            "unresolved_node_cids": list(self.unresolved_node_cids),
            "unresolved_edge_cids": list(self.unresolved_edge_cids),
            "reasons": list(self.reasons),
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "source_frontier_cids": list(self.source_frontier_cids),
            "widened": self.widened,
        }

    @property
    def frontier_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["frontier_cid"] = self.frontier_cid
        return value

    def widen(
        self,
        *,
        reasons: Sequence[str] = (),
        unavailable_dimensions: Sequence[str] = (),
        unresolved_node_cids: Sequence[str] = (),
        unresolved_edge_cids: Sequence[str] = (),
        source_frontier_cids: Sequence[str] = (),
    ) -> "UnresolvedDynamicFrontier":
        """Return a frontier that only grows."""

        extra_reasons = tuple(
            _enum(item, DynamicFrontierReason, "reason") for item in reasons
        )
        return UnresolvedDynamicFrontier(
            language=self.language,
            unresolved_node_cids=(
                *self.unresolved_node_cids,
                *unresolved_node_cids,
            ),
            unresolved_edge_cids=(
                *self.unresolved_edge_cids,
                *unresolved_edge_cids,
            ),
            reasons=(*self.reasons, *extra_reasons),
            unavailable_dimensions=(
                *self.unavailable_dimensions,
                *unavailable_dimensions,
            ),
            source_frontier_cids=(*self.source_frontier_cids, *source_frontier_cids),
            widened=True,
        )

    def without_nodes(self, node_cids: Sequence[str]) -> "UnresolvedDynamicFrontier":
        """Drop frontier nodes only after an authoritative removal."""

        blocked = set(node_cids)
        remaining_nodes = tuple(
            cid for cid in self.unresolved_node_cids if cid not in blocked
        )
        if remaining_nodes == self.unresolved_node_cids:
            return self
        return UnresolvedDynamicFrontier(
            language=self.language,
            unresolved_node_cids=remaining_nodes,
            unresolved_edge_cids=self.unresolved_edge_cids,
            reasons=self.reasons,
            unavailable_dimensions=self.unavailable_dimensions,
            source_frontier_cids=self.source_frontier_cids,
            widened=self.widened,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnresolvedDynamicFrontier":
        payload = dict(data)
        claimed = payload.pop("frontier_cid", None)
        payload.pop("schema", None)
        result = cls(**payload)
        if claimed is not None and claimed != result.frontier_cid:
            raise StaticSuccessorError("unresolved dynamic frontier identity does not match")
        return result


@dataclass(frozen=True, slots=True)
class StaticSuccessorPlan:
    """Deterministic conservative successor plan plus stage receipts."""

    snapshot_cid: str
    graph_cid: str
    language: str
    query_families: Sequence[str]
    environment_binding_cid: str
    sealed_binding_cid: str
    subject_node_cid: str | None
    candidates: Sequence[SuccessorCandidate] = ()
    frontier: UnresolvedDynamicFrontier | None = None
    receipts: Sequence[SuccessorDecisionReceipt] = ()
    complete: bool = False
    unavailable_dimensions: Sequence[str] = ()
    policy_cid: str | None = None

    SCHEMA: ClassVar[str] = STATIC_SUCCESSOR_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = STATIC_SUCCESSOR_PLANNER_INTERFACE
    CID_FIELD: ClassVar[str] = "plan_cid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_cid", _text(self.snapshot_cid, "snapshot_cid"))
        object.__setattr__(self, "graph_cid", _text(self.graph_cid, "graph_cid"))
        object.__setattr__(self, "language", _text(self.language, "language"))
        families = _query_families(self.query_families)
        object.__setattr__(self, "query_families", families)
        object.__setattr__(
            self,
            "environment_binding_cid",
            _text(self.environment_binding_cid, "environment_binding_cid"),
        )
        object.__setattr__(
            self,
            "sealed_binding_cid",
            _text(self.sealed_binding_cid, "sealed_binding_cid"),
        )
        object.__setattr__(
            self,
            "subject_node_cid",
            _optional_text(self.subject_node_cid, "subject_node_cid"),
        )
        candidates = tuple(self.candidates)
        if len(candidates) > MAX_CANDIDATES:
            raise StaticSuccessorBoundsError("successor candidate bound exceeded")
        for item in candidates:
            if not isinstance(item, SuccessorCandidate):
                raise StaticSuccessorError("candidates must be SuccessorCandidate values")
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(candidates, key=lambda item: item.candidate_cid)),
        )
        frontier = self.frontier
        if frontier is None:
            frontier = _empty_frontier(language=self.language)
        elif not isinstance(frontier, UnresolvedDynamicFrontier):
            raise StaticSuccessorError("frontier must be UnresolvedDynamicFrontier")
        object.__setattr__(self, "frontier", frontier)
        receipts = tuple(self.receipts)
        if len(receipts) > MAX_RECEIPTS:
            raise StaticSuccessorBoundsError("successor receipt bound exceeded")
        for item in receipts:
            if not isinstance(item, SuccessorDecisionReceipt):
                raise StaticSuccessorError(
                    "receipts must be SuccessorDecisionReceipt values"
                )
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _sorted_unique(self.unavailable_dimensions, "unavailable_dimensions"),
        )
        object.__setattr__(
            self, "policy_cid", _optional_text(self.policy_cid, "policy_cid")
        )
        if self.complete and (
            not frontier.empty
            or self.unavailable_dimensions
            or any(item.unknown or not item.complete for item in self.candidates)
        ):
            raise StaticSuccessorError(
                "complete successor plans cannot retain unknown or incomplete residue"
            )
        if not self.complete and not self.unavailable_dimensions:
            object.__setattr__(
                self, "unavailable_dimensions", ("incomplete_analysis",)
            )

    @property
    def candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self.candidates)

    @property
    def unknown_cids(self) -> tuple[str, ...]:
        return tuple(
            item.candidate_cid for item in self.candidates if item.unknown
        )

    @property
    def target_node_cids(self) -> tuple[str, ...]:
        return _sorted_unique(
            (item.target_node_cid for item in self.candidates), "target_node_cids"
        )

    def identity_payload(self) -> dict[str, Any]:
        assert self.frontier is not None
        return {
            "schema": self.SCHEMA,
            "snapshot_cid": self.snapshot_cid,
            "graph_cid": self.graph_cid,
            "language": self.language,
            "query_families": list(self.query_families),
            "environment_binding_cid": self.environment_binding_cid,
            "sealed_binding_cid": self.sealed_binding_cid,
            "subject_node_cid": self.subject_node_cid,
            "candidates": [item.to_dict() for item in self.candidates],
            "frontier": self.frontier.to_dict(),
            "receipts": [item.to_dict() for item in self.receipts],
            "complete": self.complete,
            "unavailable_dimensions": list(self.unavailable_dimensions),
            "policy_cid": self.policy_cid,
        }

    @property
    def plan_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["plan_cid"] = self.plan_cid
        return value

    def receipt_for(self, stage: SuccessorStage | str) -> SuccessorDecisionReceipt | None:
        stage_value = _enum(stage, SuccessorStage, "stage")
        for receipt in self.receipts:
            if receipt.stage == stage_value:
                return receipt
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticSuccessorPlan":
        payload = dict(data)
        claimed = payload.pop("plan_cid", None)
        payload.pop("schema", None)
        payload["candidates"] = tuple(
            SuccessorCandidate.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("candidates", ())
        )
        frontier = payload.get("frontier")
        if isinstance(frontier, Mapping):
            payload["frontier"] = UnresolvedDynamicFrontier.from_dict(frontier)
        payload["receipts"] = tuple(
            SuccessorDecisionReceipt.from_dict(item)
            if isinstance(item, Mapping)
            else item
            for item in payload.get("receipts", ())
        )
        result = cls(**payload)
        if claimed is not None and claimed != result.plan_cid:
            raise StaticSuccessorError("static successor plan identity does not match")
        return result


@dataclass(frozen=True, slots=True)
class ProgramGraphCatalog:
    """Sealed datasets catalog consumed by successor generation."""

    snapshot: ProgramGraphSnapshot
    nodes: tuple[ProgramGraphNode, ...] = ()
    edges: tuple[ProgramGraphEdge, ...] = ()
    successor_sets: tuple[StaticSuccessorSet, ...] = ()
    frontiers: tuple[DynamicFrontierRecord, ...] = ()
    callsites: tuple[CallsiteRecord, ...] = ()
    function_symbols: tuple[FunctionSymbolRecord, ...] = ()
    contract_states: tuple[ContractStateRecord, ...] = ()
    proof_obligation_graphs: tuple[ProofObligationGraph, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ProgramGraphSnapshot):
            raise StaticSuccessorError("catalog snapshot must be a ProgramGraphSnapshot")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "successor_sets", tuple(self.successor_sets))
        object.__setattr__(self, "frontiers", tuple(self.frontiers))
        object.__setattr__(self, "callsites", tuple(self.callsites))
        object.__setattr__(self, "function_symbols", tuple(self.function_symbols))
        object.__setattr__(self, "contract_states", tuple(self.contract_states))
        object.__setattr__(
            self, "proof_obligation_graphs", tuple(self.proof_obligation_graphs)
        )


def _catalog_from_mapping(data: Mapping[str, Any]) -> ProgramGraphCatalog:
    snapshot = data.get("snapshot")
    if snapshot is None:
        raise StaticSuccessorError("catalog requires a snapshot")
    return ProgramGraphCatalog(
        snapshot=_coerce_record(snapshot, ProgramGraphSnapshot, "snapshot"),
        nodes=_coerce_sequence(data.get("nodes", ()), ProgramGraphNode, "nodes"),
        edges=_coerce_sequence(data.get("edges", ()), ProgramGraphEdge, "edges"),
        successor_sets=_coerce_sequence(
            data.get("successor_sets", ()), StaticSuccessorSet, "successor_sets"
        ),
        frontiers=_coerce_sequence(
            data.get("frontiers", ()), DynamicFrontierRecord, "frontiers"
        ),
        callsites=_coerce_sequence(
            data.get("callsites", ()), CallsiteRecord, "callsites"
        ),
        function_symbols=_coerce_sequence(
            data.get("function_symbols", ()), FunctionSymbolRecord, "function_symbols"
        ),
        contract_states=_coerce_sequence(
            data.get("contract_states", ()), ContractStateRecord, "contract_states"
        ),
        proof_obligation_graphs=_coerce_sequence(
            data.get("proof_obligation_graphs", ()),
            ProofObligationGraph,
            "proof_obligation_graphs",
        ),
    )


def _language_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _catalog_language(snapshot: Any, catalog: Any) -> str | None:
    """Read declared language before datasets record construction."""

    sources: list[Any] = []
    if isinstance(catalog, ProgramGraphCatalog):
        sources.append(catalog.snapshot)
    elif isinstance(catalog, Mapping):
        sources.append(catalog.get("snapshot"))
        sources.append(catalog.get("language"))
    sources.append(snapshot)
    for source in sources:
        if source is None:
            continue
        if isinstance(source, str):
            language = _language_text(source)
            if language is not None:
                return language
            continue
        language = _language_text(
            source.get("language")
            if isinstance(source, Mapping)
            else getattr(source, "language", None)
        )
        if language is not None:
            return language
    return None


def _snapshot_source(
    snapshot: Any, catalog: Any
) -> ProgramGraphSnapshot | Mapping[str, Any] | None:
    if isinstance(snapshot, (ProgramGraphSnapshot, Mapping)):
        return snapshot
    if isinstance(catalog, ProgramGraphCatalog):
        return catalog.snapshot
    if isinstance(catalog, Mapping):
        item = catalog.get("snapshot")
        if isinstance(item, (ProgramGraphSnapshot, Mapping)):
            return item
    return None


def _language_admitted(language: str) -> bool:
    return language in ADMITTED_LANGUAGES and language not in UNAVAILABLE_LANGUAGES


def _language_unavailable_plan(
    *,
    language: str,
    snapshot: ProgramGraphSnapshot | Mapping[str, Any] | None,
    query_families: Sequence[str],
    subject_node_cid: str | None,
    policy_cid: str | None,
    receipts: Sequence[SuccessorDecisionReceipt],
) -> StaticSuccessorPlan:
    dimension = f"language:{language}"
    recorded = list(receipts)
    recorded.append(
        record_stage(
            SuccessorStage.LANGUAGE,
            SuccessorDisposition.UNAVAILABLE,
            SuccessorReasonCode.LANGUAGE_UNAVAILABLE,
            unavailable_dimensions=(dimension,),
            diagnostic=(
                "static successors admit only "
                + ", ".join(sorted(ADMITTED_LANGUAGES))
                + f"; {language} is typed unavailable"
            ),
        )
    )
    snapshot_obj = snapshot if isinstance(snapshot, ProgramGraphSnapshot) else None
    snapshot_cid = None
    graph_cid = None
    environment_binding_cid = None
    sealed_binding_cid = None
    if isinstance(snapshot, Mapping):
        snapshot_cid = _optional_text(
            snapshot.get("program_graph_snapshot_cid")
            or snapshot.get("snapshot_cid"),
            "snapshot_cid",
        )
        graph_cid = _optional_text(
            snapshot.get("canonical_program_graph_cid")
            or snapshot.get("graph_cid"),
            "graph_cid",
        )
        environment_binding_cid = _optional_text(
            snapshot.get("environment_binding_set_cid"),
            "environment_binding_set_cid",
        )
        sealed_binding_cid = _optional_text(
            snapshot.get("sealed_binding_cid"),
            "sealed_binding_cid",
        )
    return _blocked_plan(
        snapshot=snapshot_obj,
        query_families=query_families,
        subject_node_cid=subject_node_cid,
        policy_cid=policy_cid,
        receipts=recorded,
        unavailable_dimensions=(dimension,),
        language=language,
        snapshot_cid=snapshot_cid,
        graph_cid=graph_cid,
        environment_binding_cid=environment_binding_cid,
        sealed_binding_cid=sealed_binding_cid,
    )


def _is_language_unavailable_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "typed unavailable" in message or (
        "language" in message and "unavailable" in message
    )


def _blocked_plan(
    *,
    snapshot: ProgramGraphSnapshot | None,
    query_families: Sequence[str],
    subject_node_cid: str | None,
    policy_cid: str | None,
    receipts: Sequence[SuccessorDecisionReceipt],
    unavailable_dimensions: Sequence[str],
    language: str = "python",
    snapshot_cid: str | None = None,
    graph_cid: str | None = None,
    environment_binding_cid: str | None = None,
    sealed_binding_cid: str | None = None,
) -> StaticSuccessorPlan:
    snapshot_obj = snapshot if isinstance(snapshot, ProgramGraphSnapshot) else None
    language_value = (
        str(snapshot_obj.language) if snapshot_obj is not None else language
    )
    return StaticSuccessorPlan(
        snapshot_cid=(
            snapshot_obj.program_graph_snapshot_cid
            if snapshot_obj is not None
            else snapshot_cid or "stale"
        ),
        graph_cid=(
            snapshot_obj.canonical_program_graph_cid
            if snapshot_obj is not None
            else graph_cid or "stale"
        ),
        language=language_value,
        query_families=query_families,
        environment_binding_cid=(
            snapshot_obj.environment_binding_set_cid
            if snapshot_obj is not None
            else environment_binding_cid or "stale"
        ),
        sealed_binding_cid=(
            snapshot_obj.sealed_binding_cid
            if snapshot_obj is not None
            else sealed_binding_cid or "stale"
        ),
        subject_node_cid=subject_node_cid,
        candidates=(),
        frontier=_empty_frontier(language=language_value).widen(
            reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
            unavailable_dimensions=unavailable_dimensions,
        ),
        receipts=_fill_remaining_receipts(receipts, blocked=True),
        complete=False,
        unavailable_dimensions=unavailable_dimensions,
        policy_cid=policy_cid,
    )


def _fill_remaining_receipts(
    receipts: Sequence[SuccessorDecisionReceipt],
    *,
    blocked: bool,
    candidate_cids: Sequence[str] = (),
) -> tuple[SuccessorDecisionReceipt, ...]:
    recorded = {item.stage: item for item in receipts}
    filled: list[SuccessorDecisionReceipt] = []
    for stage in GENERATION_STAGE_ORDER:
        existing = recorded.get(stage.value)
        if existing is not None:
            filled.append(existing)
            continue
        if blocked:
            filled.append(
                record_stage(
                    stage,
                    SuccessorDisposition.SKIPPED,
                    SuccessorReasonCode.FRESHNESS_BLOCKED,
                    candidate_cids_in=candidate_cids,
                    candidate_cids_out=candidate_cids,
                    diagnostic="prior stage failed closed",
                )
            )
        else:
            filled.append(
                record_stage(
                    stage,
                    SuccessorDisposition.SKIPPED,
                    SuccessorReasonCode.NO_EVIDENCE,
                    candidate_cids_in=candidate_cids,
                    candidate_cids_out=candidate_cids,
                )
            )
    return tuple(filled)


def _candidate_key(candidate: SuccessorCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.kind,
        candidate.subject_node_cid,
        candidate.target_node_cid,
        candidate.origin,
    )


def _merge_candidates(
    existing: Sequence[SuccessorCandidate],
    incoming: Sequence[SuccessorCandidate],
) -> list[SuccessorCandidate]:
    merged: dict[tuple[str, str, str, str], SuccessorCandidate] = {
        _candidate_key(item): item for item in existing
    }
    for item in incoming:
        key = _candidate_key(item)
        prior = merged.get(key)
        if prior is None:
            merged[key] = item
            continue
        merged[key] = SuccessorCandidate(
            kind=prior.kind,
            subject_node_cid=prior.subject_node_cid,
            target_node_cid=prior.target_node_cid,
            origin=prior.origin,
            edge_cid=prior.edge_cid or item.edge_cid,
            resolution_status=(
                prior.resolution_status
                if prior.resolution_status != ResolutionStatus.DEFINITE.value
                else item.resolution_status
            ),
            unknown=prior.unknown or item.unknown,
            complete=prior.complete and item.complete,
            unavailable_dimensions=(
                *prior.unavailable_dimensions,
                *item.unavailable_dimensions,
            ),
        )
    return list(merged.values())


def _candidate_from_edge(
    edge: ProgramGraphEdge,
    *,
    subject_node_cid: str,
    allowed_kinds: frozenset[str],
) -> SuccessorCandidate | None:
    kind_value = str(edge.edge_kind)
    if kind_value not in allowed_kinds and kind_value != (
        ProgramGraphEdgeKind.UNRESOLVED_DYNAMIC.value
    ):
        return None
    if edge.source_node_cid != subject_node_cid:
        return None
    unresolved = is_unresolved_dynamic_edge(edge)
    candidate_kind = (
        SuccessorKind.UNRESOLVED
        if unresolved
        else _EDGE_KIND_TO_CANDIDATE_KIND.get(kind_value)
    )
    origin = _EDGE_KIND_TO_ORIGIN.get(kind_value)
    if candidate_kind is None or origin is None:
        return None
    dimensions = tuple(edge.unavailable_dimensions)
    if unresolved and not dimensions:
        dimensions = ("incomplete_analysis",)
    return SuccessorCandidate(
        kind=candidate_kind,
        subject_node_cid=subject_node_cid,
        target_node_cid=edge.target_node_cid,
        origin=origin,
        edge_cid=edge.program_graph_edge_cid,
        resolution_status=str(edge.resolution_status),
        unknown=unresolved,
        complete=not unresolved and not dimensions,
        unavailable_dimensions=dimensions,
    )


def _subjects_for(
    catalog: ProgramGraphCatalog,
    subject_node_cid: str | None,
) -> tuple[ProgramGraphNode, ...]:
    nodes = catalog.nodes
    if subject_node_cid is not None:
        match = [node for node in nodes if node.program_graph_node_cid == subject_node_cid]
        return tuple(match)
    return tuple(
        node for node in nodes if str(node.node_kind) in SUBJECT_NODE_KINDS
    )


def _edge_to_target(
    edges: Sequence[ProgramGraphEdge],
    *,
    source_cid: str,
    target_cid: str,
) -> ProgramGraphEdge | None:
    matches = [
        edge
        for edge in edges
        if edge.source_node_cid == source_cid and edge.target_node_cid == target_cid
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def generate_static_successors(
    snapshot: ProgramGraphSnapshot | Mapping[str, Any] | None = None,
    *,
    catalog: Mapping[str, Any] | ProgramGraphCatalog | None = None,
    nodes: Sequence[ProgramGraphNode | Mapping[str, Any]] = (),
    edges: Sequence[ProgramGraphEdge | Mapping[str, Any]] = (),
    successor_sets: Sequence[StaticSuccessorSet | Mapping[str, Any]] = (),
    frontiers: Sequence[DynamicFrontierRecord | Mapping[str, Any]] = (),
    callsites: Sequence[CallsiteRecord | Mapping[str, Any]] = (),
    function_symbols: Sequence[FunctionSymbolRecord | Mapping[str, Any]] = (),
    contract_states: Sequence[ContractStateRecord | Mapping[str, Any]] = (),
    proof_obligation_graphs: Sequence[ProofObligationGraph | Mapping[str, Any]] = (),
    query_family: Any = None,
    subject_node_cid: str | None = None,
    expected_snapshot_cid: str | None = None,
    expected_graph_cid: str | None = None,
    environment_binding_cid: str | None = None,
    policy_cid: str | None = None,
) -> StaticSuccessorPlan:
    """Generate conservative next-call/event successors from a sealed graph catalog.

    Static possibilities and explicit unknown are preserved.  The returned plan
    records every generation stage.  Stale bindings fail closed with no
    candidates.
    """

    families = _query_families(query_family)
    receipts: list[SuccessorDecisionReceipt] = []
    subject = _optional_text(subject_node_cid, "subject_node_cid")
    peeked_language = _catalog_language(snapshot, catalog)
    if peeked_language is not None and not _language_admitted(peeked_language):
        return _language_unavailable_plan(
            language=peeked_language,
            snapshot=_snapshot_source(snapshot, catalog),
            query_families=families,
            subject_node_cid=subject,
            policy_cid=policy_cid,
            receipts=receipts,
        )

    try:
        if isinstance(catalog, ProgramGraphCatalog):
            bound = catalog
        elif isinstance(catalog, Mapping):
            bound = _catalog_from_mapping(catalog)
        elif snapshot is not None:
            bound = ProgramGraphCatalog(
                snapshot=_coerce_record(snapshot, ProgramGraphSnapshot, "snapshot"),
                nodes=_coerce_sequence(nodes, ProgramGraphNode, "nodes"),
                edges=_coerce_sequence(edges, ProgramGraphEdge, "edges"),
                successor_sets=_coerce_sequence(
                    successor_sets, StaticSuccessorSet, "successor_sets"
                ),
                frontiers=_coerce_sequence(frontiers, DynamicFrontierRecord, "frontiers"),
                callsites=_coerce_sequence(callsites, CallsiteRecord, "callsites"),
                function_symbols=_coerce_sequence(
                    function_symbols, FunctionSymbolRecord, "function_symbols"
                ),
                contract_states=_coerce_sequence(
                    contract_states, ContractStateRecord, "contract_states"
                ),
                proof_obligation_graphs=_coerce_sequence(
                    proof_obligation_graphs,
                    ProofObligationGraph,
                    "proof_obligation_graphs",
                ),
            )
        else:
            raise StaticSuccessorError("generate_static_successors requires a snapshot")
    except ProgramGraphError as exc:
        if _is_language_unavailable_error(exc):
            return _language_unavailable_plan(
                language=peeked_language or "unknown",
                snapshot=snapshot if isinstance(snapshot, Mapping) else None,
                query_families=families,
                subject_node_cid=subject,
                policy_cid=policy_cid,
                receipts=receipts,
            )
        raise

    snap = bound.snapshot

    stale_reason: SuccessorReasonCode | None = None
    stale_diagnostic = ""
    if expected_snapshot_cid is not None and expected_snapshot_cid != (
        snap.program_graph_snapshot_cid
    ):
        stale_reason = SuccessorReasonCode.STALE_SNAPSHOT
        stale_diagnostic = "expected snapshot CID does not match the sealed snapshot"
    elif expected_graph_cid is not None and expected_graph_cid != (
        snap.canonical_program_graph_cid
    ):
        stale_reason = SuccessorReasonCode.STALE_GRAPH
        stale_diagnostic = "expected graph CID does not match the sealed snapshot"
    elif environment_binding_cid is not None and environment_binding_cid not in {
        snap.environment_binding_set_cid,
        snap.sealed_binding_cid,
    }:
        stale_reason = SuccessorReasonCode.STALE_ENVIRONMENT
        stale_diagnostic = "environment binding is not current for this snapshot"

    if stale_reason is not None:
        receipts.append(
            record_stage(
                SuccessorStage.FRESHNESS,
                SuccessorDisposition.REJECTED,
                stale_reason,
                unavailable_dimensions=("stale_graph",),
                diagnostic=stale_diagnostic,
            )
        )
        return _blocked_plan(
            snapshot=snap,
            query_families=families,
            subject_node_cid=subject,
            policy_cid=policy_cid,
            receipts=receipts,
            unavailable_dimensions=("stale_graph",),
        )

    receipts.append(
        record_stage(
            SuccessorStage.FRESHNESS,
            SuccessorDisposition.SELECTED,
            SuccessorReasonCode.GRAPH_FRESH,
            evidence_cids=(snap.program_graph_snapshot_cid,),
        )
    )

    language = str(snap.language)
    if not _language_admitted(language):
        return _language_unavailable_plan(
            language=language,
            snapshot=snap,
            query_families=families,
            subject_node_cid=subject,
            policy_cid=policy_cid,
            receipts=receipts,
        )

    receipts.append(
        record_stage(
            SuccessorStage.LANGUAGE,
            SuccessorDisposition.SELECTED,
            SuccessorReasonCode.LANGUAGE_ADMITTED,
        )
    )

    try:
        verify_program_graph_catalog(
            snap,
            nodes=bound.nodes,
            edges=bound.edges,
            callsites=bound.callsites,
            function_symbols=bound.function_symbols,
            contract_states=bound.contract_states,
            proof_obligation_graphs=bound.proof_obligation_graphs,
            successor_sets=bound.successor_sets,
            frontiers=bound.frontiers,
        )
    except ProgramGraphError as exc:
        receipts.append(
            record_stage(
                SuccessorStage.CATALOG,
                SuccessorDisposition.REJECTED,
                SuccessorReasonCode.CATALOG_MISMATCH,
                unavailable_dimensions=("stale_graph",),
                diagnostic=str(exc)[:MAX_TEXT_CHARS],
            )
        )
        return _blocked_plan(
            snapshot=snap,
            query_families=families,
            subject_node_cid=subject,
            policy_cid=policy_cid,
            receipts=receipts,
            unavailable_dimensions=("stale_graph",),
        )

    subjects = _subjects_for(bound, subject)
    if subject is not None and not subjects:
        receipts.append(
            record_stage(
                SuccessorStage.CATALOG,
                SuccessorDisposition.REJECTED,
                SuccessorReasonCode.SUBJECT_MISSING,
                unavailable_dimensions=("stale_graph",),
                diagnostic="subject_node_cid is not present in the sealed catalog",
            )
        )
        return _blocked_plan(
            snapshot=snap,
            query_families=families,
            subject_node_cid=subject,
            policy_cid=policy_cid,
            receipts=receipts,
            unavailable_dimensions=("stale_graph",),
        )

    receipts.append(
        record_stage(
            SuccessorStage.CATALOG,
            SuccessorDisposition.SELECTED,
            SuccessorReasonCode.CATALOG_VERIFIED,
            evidence_cids=(snap.canonical_program_graph_cid,),
        )
    )

    node_index = _index_by_cid(bound.nodes)
    include_calls = QueryFamily.NEXT_CALL.value in families
    include_events = QueryFamily.NEXT_EVENT.value in families
    allowed_edge_kinds = frozenset(
        (*(NEXT_CALL_EDGE_KINDS if include_calls else ()),
         *(NEXT_EVENT_EDGE_KINDS if include_events else ()),
         ProgramGraphEdgeKind.UNRESOLVED_DYNAMIC.value)
    )

    candidates: list[SuccessorCandidate] = []
    recalled_cids: list[str] = []
    recalled_subjects = 0
    for node in subjects:
        subject_cid = node.program_graph_node_cid
        matching_sets = [
            item
            for item in bound.successor_sets
            if item.subject_node_cid == subject_cid
        ]
        if not matching_sets:
            continue
        recalled_subjects += 1
        for successor_set in matching_sets:
            for target_cid in successor_set.successor_node_cids:
                edge = _edge_to_target(
                    bound.edges, source_cid=subject_cid, target_cid=target_cid
                )
                target = node_index.get(target_cid)
                unresolved = target is not None and is_unresolved_dynamic_node(target)
                if edge is not None:
                    unresolved = unresolved or is_unresolved_dynamic_edge(edge)
                dimensions = tuple(successor_set.unavailable_dimensions)
                if unresolved and not dimensions:
                    dimensions = ("incomplete_analysis",)
                if target is not None:
                    dimensions = tuple(
                        sorted(set(dimensions) | set(target.unavailable_dimensions))
                    )
                candidate = SuccessorCandidate(
                    kind=(
                        SuccessorKind.UNRESOLVED
                        if unresolved
                        else (
                            SuccessorKind.NEXT_EVENT
                            if include_events and not include_calls
                            else SuccessorKind.NEXT_CALL
                        )
                    ),
                    subject_node_cid=subject_cid,
                    target_node_cid=target_cid,
                    origin=SuccessorOrigin.STATIC_SUCCESSOR_SET,
                    edge_cid=None if edge is None else edge.program_graph_edge_cid,
                    resolution_status=(
                        str(edge.resolution_status)
                        if edge is not None
                        else (
                            ResolutionStatus.UNRESOLVED.value
                            if unresolved
                            else ResolutionStatus.DEFINITE.value
                        )
                    ),
                    unknown=unresolved,
                    complete=not unresolved and not dimensions,
                    unavailable_dimensions=dimensions,
                )
                candidates = _merge_candidates(candidates, (candidate,))
                recalled_cids.append(candidate.candidate_cid)
            for edge_cid in successor_set.successor_edge_cids:
                edge_match = next(
                    (
                        item
                        for item in bound.edges
                        if item.program_graph_edge_cid == edge_cid
                    ),
                    None,
                )
                if edge_match is None:
                    continue
                derived = _candidate_from_edge(
                    edge_match,
                    subject_node_cid=subject_cid,
                    allowed_kinds=allowed_edge_kinds,
                )
                if derived is None:
                    derived = SuccessorCandidate(
                        kind=(
                            SuccessorKind.UNRESOLVED
                            if is_unresolved_dynamic_edge(edge_match)
                            else SuccessorKind.NEXT_CALL
                        ),
                        subject_node_cid=subject_cid,
                        target_node_cid=edge_match.target_node_cid,
                        origin=SuccessorOrigin.STATIC_SUCCESSOR_SET,
                        edge_cid=edge_match.program_graph_edge_cid,
                        resolution_status=str(edge_match.resolution_status),
                        unknown=is_unresolved_dynamic_edge(edge_match),
                        complete=not is_unresolved_dynamic_edge(edge_match)
                        and not successor_set.unavailable_dimensions,
                        unavailable_dimensions=successor_set.unavailable_dimensions,
                    )
                candidates = _merge_candidates(candidates, (derived,))
                recalled_cids.append(derived.candidate_cid)

    if recalled_subjects:
        receipts.append(
            record_stage(
                SuccessorStage.STATIC_RECALL,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.STATIC_RECALL,
                candidate_cids_out=_sorted_unique(recalled_cids, "recalled_cids"),
                evidence_cids=tuple(
                    item.static_successor_set_cid for item in bound.successor_sets
                ),
            )
        )
    else:
        receipts.append(
            record_stage(
                SuccessorStage.STATIC_RECALL,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.NO_SUCCESSOR_SET,
            )
        )

    callsite_added: list[str] = []
    if include_calls:
        functions_by_declaration = {
            item.declaration_cid: item for item in bound.function_symbols
        }
        nodes_by_record = {
            node.record_cid: node
            for node in bound.nodes
            if node.record_cid is not None
        }
        for node in subjects:
            for callsite in bound.callsites:
                if callsite.caller_logical_name != node.logical_name:
                    continue
                target_node = None
                if callsite.callee_declaration_cid is not None:
                    symbol = functions_by_declaration.get(callsite.callee_declaration_cid)
                    if symbol is not None:
                        target_node = nodes_by_record.get(symbol.function_symbol_record_cid)
                unresolved = str(callsite.resolution_status) in {
                    ResolutionStatus.UNRESOLVED.value,
                    ResolutionStatus.UNAVAILABLE.value,
                }
                if target_node is None and not unresolved:
                    continue
                target_cid = (
                    target_node.program_graph_node_cid
                    if target_node is not None
                    else callsite.callsite_record_cid
                )
                dimensions = tuple(callsite.unavailable_dimensions)
                if unresolved and not dimensions:
                    dimensions = tuple(callsite.unavailable_dimensions) or (
                        "incomplete_analysis",
                    )
                candidate = SuccessorCandidate(
                    kind=(
                        SuccessorKind.UNRESOLVED
                        if unresolved
                        else SuccessorKind.NEXT_CALL
                    ),
                    subject_node_cid=node.program_graph_node_cid,
                    target_node_cid=target_cid,
                    origin=SuccessorOrigin.CALLSITE_RECORD,
                    edge_cid=None,
                    resolution_status=str(callsite.resolution_status),
                    unknown=unresolved,
                    complete=not unresolved and not dimensions,
                    unavailable_dimensions=dimensions,
                )
                candidates = _merge_candidates(candidates, (candidate,))
                callsite_added.append(candidate.candidate_cid)
        receipts.append(
            record_stage(
                SuccessorStage.CALLSITE_EXPANSION,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.CALLSITE_EXPANDED,
                candidate_cids_out=_sorted_unique(callsite_added, "callsite_added"),
                evidence_cids=tuple(item.callsite_record_cid for item in bound.callsites),
            )
        )
    else:
        receipts.append(
            record_stage(
                SuccessorStage.CALLSITE_EXPANSION,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.FAMILY_SKIPPED,
                diagnostic="callsite expansion applies to next_call",
            )
        )

    cfg_added: list[str] = []
    if include_events or include_calls:
        for node in subjects:
            subject_cid = node.program_graph_node_cid
            for edge in bound.edges:
                derived = _candidate_from_edge(
                    edge,
                    subject_node_cid=subject_cid,
                    allowed_kinds=allowed_edge_kinds,
                )
                if derived is None:
                    continue
                if derived.kind == SuccessorKind.NEXT_EVENT and not include_events:
                    continue
                if derived.kind == SuccessorKind.NEXT_CALL and not include_calls:
                    continue
                candidates = _merge_candidates(candidates, (derived,))
                if derived.kind == SuccessorKind.NEXT_EVENT:
                    cfg_added.append(derived.candidate_cid)
        if include_events:
            receipts.append(
                record_stage(
                    SuccessorStage.CFG_EVENT_EXPANSION,
                    SuccessorDisposition.SELECTED,
                    SuccessorReasonCode.CFG_EXPANDED,
                    candidate_cids_out=_sorted_unique(cfg_added, "cfg_added"),
                )
            )
        else:
            receipts.append(
                record_stage(
                    SuccessorStage.CFG_EVENT_EXPANSION,
                    SuccessorDisposition.SKIPPED,
                    SuccessorReasonCode.FAMILY_SKIPPED,
                    diagnostic="cfg/event expansion applies to next_event",
                )
            )
    else:
        receipts.append(
            record_stage(
                SuccessorStage.CFG_EVENT_EXPANSION,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.FAMILY_SKIPPED,
            )
        )

    if len(candidates) > MAX_CANDIDATES:
        raise StaticSuccessorBoundsError("successor candidate bound exceeded")

    frontier = _empty_frontier(language=language)
    unresolved_nodes = [
        node for node in bound.nodes if is_unresolved_dynamic_node(node)
    ]
    unresolved_edges = [
        edge for edge in bound.edges if is_unresolved_dynamic_edge(edge)
    ]
    if subject is not None:
        subject_cids = {item.program_graph_node_cid for item in subjects}
        unresolved_edges = [
            edge
            for edge in unresolved_edges
            if edge.source_node_cid in subject_cids
            or edge.target_node_cid in subject_cids
        ]
        unresolved_target_cids = {edge.target_node_cid for edge in unresolved_edges}
        unresolved_nodes = [
            node
            for node in unresolved_nodes
            if node.program_graph_node_cid in unresolved_target_cids
            or any(
                candidate.target_node_cid == node.program_graph_node_cid
                for candidate in candidates
            )
        ]

    reasons: list[str] = []
    dimensions: list[str] = list(snap.unavailable_dimensions)
    for node in unresolved_nodes:
        reasons.extend(node.unavailable_dimensions)
        dimensions.extend(node.unavailable_dimensions)
        for candidate in candidates:
            if candidate.target_node_cid == node.program_graph_node_cid:
                candidate_unknown = SuccessorCandidate(
                    kind=SuccessorKind.UNRESOLVED,
                    subject_node_cid=candidate.subject_node_cid,
                    target_node_cid=candidate.target_node_cid,
                    origin=SuccessorOrigin.UNRESOLVED_DYNAMIC_NODE,
                    edge_cid=candidate.edge_cid,
                    resolution_status=ResolutionStatus.UNRESOLVED.value,
                    unknown=True,
                    complete=False,
                    unavailable_dimensions=(
                        *candidate.unavailable_dimensions,
                        *node.unavailable_dimensions,
                    ),
                )
                candidates = _merge_candidates(candidates, (candidate_unknown,))
    for edge in unresolved_edges:
        reasons.extend(edge.unavailable_dimensions)
        dimensions.extend(edge.unavailable_dimensions)
        derived = _candidate_from_edge(
            edge,
            subject_node_cid=edge.source_node_cid,
            allowed_kinds=allowed_edge_kinds,
        )
        if derived is not None and (
            subject is None
            or derived.subject_node_cid in {item.program_graph_node_cid for item in subjects}
        ):
            candidates = _merge_candidates(candidates, (derived,))

    allowed_reasons = {item.value for item in DynamicFrontierReason}
    frontier_reasons = tuple(sorted({item for item in reasons if item in allowed_reasons}))
    source_frontier_cids = tuple(item.dynamic_frontier_record_cid for item in bound.frontiers)
    for record in bound.frontiers:
        frontier_reasons = tuple(
            sorted(set(frontier_reasons) | set(record.reasons))
        )
        dimensions.extend(record.unavailable_dimensions)
        if subject is None:
            frontier = frontier.widen(
                unresolved_node_cids=record.unresolved_node_cids,
                unresolved_edge_cids=record.unresolved_edge_cids,
            )
        else:
            subject_cids = {item.program_graph_node_cid for item in subjects}
            related_nodes = [
                cid
                for cid in record.unresolved_node_cids
                if cid in {item.target_node_cid for item in candidates}
                or any(
                    edge.target_node_cid == cid and edge.source_node_cid in subject_cids
                    for edge in bound.edges
                )
            ]
            related_edges = [
                cid
                for cid in record.unresolved_edge_cids
                if any(
                    edge.program_graph_edge_cid == cid
                    and (
                        edge.source_node_cid in subject_cids
                        or edge.target_node_cid in subject_cids
                    )
                    for edge in bound.edges
                )
            ]
            frontier = frontier.widen(
                unresolved_node_cids=related_nodes,
                unresolved_edge_cids=related_edges,
            )

    if (
        unresolved_nodes
        or unresolved_edges
        or frontier_reasons
        or any(not item.complete or item.unknown for item in candidates)
        or snap.unavailable_dimensions
    ):
        if not frontier_reasons:
            frontier_reasons = (DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,)
        if not dimensions:
            dimensions = ["incomplete_analysis"]
        frontier = frontier.widen(
            reasons=frontier_reasons,
            unavailable_dimensions=dimensions,
            unresolved_node_cids=tuple(
                item.program_graph_node_cid for item in unresolved_nodes
            ),
            unresolved_edge_cids=tuple(
                item.program_graph_edge_cid for item in unresolved_edges
            ),
            source_frontier_cids=source_frontier_cids,
        )
        receipts.append(
            record_stage(
                SuccessorStage.DYNAMIC_FRONTIER,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.DYNAMIC_FRONTIER_WIDENED,
                unknown_cids=tuple(
                    item.candidate_cid for item in candidates if item.unknown
                ),
                unavailable_dimensions=frontier.unavailable_dimensions,
                widened=True,
                evidence_cids=source_frontier_cids,
            )
        )
    else:
        receipts.append(
            record_stage(
                SuccessorStage.DYNAMIC_FRONTIER,
                SuccessorDisposition.SKIPPED,
                SuccessorReasonCode.FRONTIER_EMPTY,
            )
        )

    receipts.append(
        record_stage(
            SuccessorStage.RESIDUAL_ESCALATION,
            SuccessorDisposition.SKIPPED,
            SuccessorReasonCode.NO_RESIDUAL,
            diagnostic="generation is deterministic-only; model fallback is not invoked",
        )
    )

    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_cid))
    unknown_cids = tuple(item.candidate_cid for item in ordered if item.unknown)
    unavailable = _sorted_unique(
        (
            *snap.unavailable_dimensions,
            *frontier.unavailable_dimensions,
            *(dim for item in ordered for dim in item.unavailable_dimensions),
        ),
        "unavailable_dimensions",
    )
    complete = (
        frontier.empty
        and not unknown_cids
        and not unavailable
        and all(item.complete for item in ordered)
    )
    if complete:
        receipts.append(
            record_stage(
                SuccessorStage.COMPLETENESS,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.COMPLETE,
                candidate_cids_out=tuple(item.candidate_cid for item in ordered),
            )
        )
        unavailable = ()
    else:
        if not unavailable:
            unavailable = ("incomplete_analysis",)
        receipts.append(
            record_stage(
                SuccessorStage.COMPLETENESS,
                SuccessorDisposition.SELECTED,
                SuccessorReasonCode.INCOMPLETE,
                candidate_cids_out=tuple(item.candidate_cid for item in ordered),
                unknown_cids=unknown_cids,
                unavailable_dimensions=unavailable,
                widened=frontier.widened,
            )
        )

    filled = _fill_remaining_receipts(receipts, blocked=False)
    return StaticSuccessorPlan(
        snapshot_cid=snap.program_graph_snapshot_cid,
        graph_cid=snap.canonical_program_graph_cid,
        language=language,
        query_families=families,
        environment_binding_cid=snap.environment_binding_set_cid,
        sealed_binding_cid=snap.sealed_binding_cid,
        subject_node_cid=subject,
        candidates=ordered,
        frontier=frontier,
        receipts=filled,
        complete=complete,
        unavailable_dimensions=unavailable,
        policy_cid=policy_cid,
    )


@dataclass(frozen=True, slots=True)
class StaticSuccessorPlanner:
    """Operational planner for conservative static successor generation."""

    expected_snapshot_cid: str | None = None
    expected_graph_cid: str | None = None
    environment_binding_cid: str | None = None
    policy_cid: str | None = None

    INTERFACE: ClassVar[str] = STATIC_SUCCESSOR_PLANNER_INTERFACE
    EVIDENCE: ClassVar[str] = SAWM_STATIC_SUCCESSORS_EVIDENCE

    def generate(
        self,
        snapshot: ProgramGraphSnapshot | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> StaticSuccessorPlan:
        """Generate successors, applying planner-bound freshness constraints."""

        kwargs.setdefault("expected_snapshot_cid", self.expected_snapshot_cid)
        kwargs.setdefault("expected_graph_cid", self.expected_graph_cid)
        kwargs.setdefault("environment_binding_cid", self.environment_binding_cid)
        kwargs.setdefault("policy_cid", self.policy_cid)
        return generate_static_successors(snapshot, **kwargs)


__all__ = [
    "FORBIDDEN_FIELD_MARKERS",
    "GENERATION_STAGE_ORDER",
    "SAWM_STATIC_SUCCESSORS_EVIDENCE",
    "STATIC_SUCCESSOR_PLANNER_INTERFACE",
    "SUCCESSOR_DECISION_RECEIPT_INTERFACE",
    "UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE",
    "ProgramGraphCatalog",
    "StaleProgramGraphError",
    "StaticSuccessorBoundsError",
    "StaticSuccessorError",
    "StaticSuccessorPlan",
    "StaticSuccessorPlanner",
    "SuccessorCandidate",
    "SuccessorDecisionReceipt",
    "SuccessorDisposition",
    "SuccessorKind",
    "SuccessorOrigin",
    "SuccessorReasonCode",
    "SuccessorStage",
    "UnresolvedDynamicFrontier",
    "generate_static_successors",
    "record_stage",
]
