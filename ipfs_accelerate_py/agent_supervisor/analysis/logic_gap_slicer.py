"""Conservative static logic-gap slicer (LPR-007).

``LogicGapSlicer`` builds a dependency-complete *minimal* static slice and an
explicit :class:`InformationDemand` for every supported :class:`ProgramLogicGoal`.

Authority rules (fail-closed):

* Slices carry **references** only — never source bodies, snippets, or secrets.
* Completeness is never forged: a bound hit, unsupported construct, open
  frontier, or partial analyzer coverage yields
  :attr:`StaticSliceCompleteness.INCOMPLETE` or
  :attr:`StaticSliceCompleteness.UNSUPPORTED`, never ``solved``.
* Cross-root facts are rejected; root drift marks the slice stale.
* Cycles are represented as finite SCC references, not unrolled.
* Selected and excluded facts, reaching/path/dominance requirements, and
  caller/constructor/schema boundaries are recorded explicitly.
* Unknown frontier, exclusions, analyzer coverage, and required next source
  types feed Tactician routing without claiming semantic authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from ..program_graph import Completeness, ProgramEdgeKind, ProgramNodeKind
from .change_propagation_contracts import ImpactClosureReceipt, ImpactCompleteness
from .program_logic_prediction_contracts import (
    GapDisposition,
    GapMissingClass,
    GoalDisposition,
    GoalFamily,
    LogicFacetKind,
    LogicGap,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    ProgramLogicPredictionError,
    SourceRouteKind,
)
from .program_logic_premise_corpus import (
    PremiseSourceClass,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from .value_provenance_graph import (
    Completeness as VpgCompleteness,
    ProvenanceStatus,
    UnknownReason,
    ValueProvenanceGraph,
)


# ---------------------------------------------------------------------------
# Schema / producer constants
# ---------------------------------------------------------------------------

LOGIC_GAP_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-gap-slice@1"
)
INFORMATION_DEMAND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/information-demand@1"
)
ANALYZER_COVERAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analyzer-coverage@1"
)
SLICE_FACT_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/slice-fact-selection@1"
)
SCC_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/slice-scc-reference@1"
)
LOGIC_GAP_SLICING_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-gap-slicing-inventory@1"
)

PRODUCER_ID: Final[str] = "logic-gap-slicer@1"
CONTRACT_VERSION: Final[int] = 1

MAX_GOALS: Final[int] = 512
MAX_SLICES: Final[int] = 512
MAX_FACTS_PER_SLICE: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_REF_BYTES: Final[int] = 512
MAX_SCC_MEMBERS: Final[int] = 256
MAX_BACKWARD_DEPTH: Final[int] = 64
DEFAULT_MAX_SLICE_NODES: Final[int] = 512


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicGapSlicerError(ContractValidationError):
    """Base error for logic-gap slicing failures."""


class LogicGapSlicerAuthorityError(LogicGapSlicerError):
    """Cross-root fact, forged completeness, or authority violation."""


class LogicGapSlicerBoundsError(LogicGapSlicerError):
    """A hard bound would be exceeded by the requested slice."""


# ---------------------------------------------------------------------------
# Closed taxonomies
# ---------------------------------------------------------------------------


class StaticSliceCompleteness(str, Enum):
    """Whether a static slice may be treated as dependency-complete.

    ``COMPLETE`` is allowed only when every required static fact is present and
    no bound, frontier, or unsupported construct remains.  Bounds and
    unsupported syntax never yield a "solved" disposition.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    FRONTIER = "frontier"
    STALE = "stale"
    ABSTAINED = "abstained"


class SliceFactClass(str, Enum):
    """Closed classes of facts that may appear in a static slice."""

    PREMISE = "premise"
    REACHING_DEFINITION = "reaching_definition"
    DEF_USE = "def_use"
    DOMINANCE = "dominance"
    PATH_CONDITION = "path_condition"
    GRAPH_NODE = "graph_node"
    GRAPH_EDGE = "graph_edge"
    CALLER_BOUNDARY = "caller_boundary"
    CONSTRUCTOR_BOUNDARY = "constructor_boundary"
    SCHEMA_BOUNDARY = "schema_boundary"
    CONSUMER_CLOSURE = "consumer_closure"
    SCC_REFERENCE = "scc_reference"
    UNKNOWN_FRONTIER = "unknown_frontier"
    EXCLUSION = "exclusion"
    ANALYZER_COVERAGE = "analyzer_coverage"
    INFORMATION_PROVENANCE = "information_provenance"
    INTERPROCEDURAL_THREAD = "interprocedural_thread"
    TYPE_REFINEMENT = "type_refinement"
    ASSUMPTION = "assumption"
    INVALIDATOR = "invalidator"


class SliceSelectionDisposition(str, Enum):
    """Whether a fact was admitted into the minimal slice or excluded."""

    SELECTED = "selected"
    EXCLUDED = "excluded"
    FRONTIER = "frontier"
    BOUND_EXCEEDED = "bound_exceeded"
    UNSUPPORTED = "unsupported"
    CROSS_ROOT = "cross_root"


class AnalyzerKind(str, Enum):
    """Closed analyzer surfaces whose coverage is recorded on every slice."""

    VALUE_PROVENANCE = "value_provenance"
    PROGRAM_DEPENDENCY_GRAPH = "program_dependency_graph"
    IMPACT_CLOSURE = "impact_closure"
    PREMISE_CORPUS = "premise_corpus"
    CONTROL_FLOW = "control_flow"
    REACHING_DEFINITIONS = "reaching_definitions"
    DOMINANCE = "dominance"
    PATH_CONDITIONS = "path_conditions"
    CALL_RESOLUTION = "call_resolution"
    SCHEMA_PROTOCOL = "schema_protocol"


class InventoryDisposition(str, Enum):
    """Closed outcomes for a multi-goal slicing inventory."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    ABSTAINED = "abstained"
    CONFLICT = "conflict"


class ExclusionReason(str, Enum):
    """Why a candidate fact was excluded from the minimal slice."""

    OUT_OF_SCOPE = "out_of_scope"
    NOT_REACHABLE = "not_reachable"
    NOMINATING_ONLY = "nominating_only"
    HYPOTHESIS_ONLY = "hypothesis_only"
    CROSS_ROOT = "cross_root"
    REDUNDANT = "redundant"
    BEYOND_BOUND = "beyond_bound"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    EXCLUDED_ROOT = "excluded_root"
    GENERATED_OR_NATIVE = "generated_or_native"
    SELF_VALIDATION = "self_validation"
    STALE = "stale"


# Edge kinds that define caller / constructor / schema boundaries.
_CALLER_EDGE_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CALLS,
        ProgramEdgeKind.OVERRIDES,
        ProgramEdgeKind.CALLBACK_TO,
        ProgramEdgeKind.INJECTS,
        ProgramEdgeKind.DECORATES,
    }
)
_CONSTRUCTOR_EDGE_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.CONSTRUCTS,
        ProgramEdgeKind.FACTORY_CREATES,
        ProgramEdgeKind.BUILDER_BUILDS,
        ProgramEdgeKind.REGISTERS,
    }
)
_SCHEMA_EDGE_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        ProgramEdgeKind.SERIALIZES,
        ProgramEdgeKind.DESERIALIZES,
        ProgramEdgeKind.MIGRATES,
        ProgramEdgeKind.SCHEMA_OF,
        ProgramEdgeKind.CONFIGURES,
        ProgramEdgeKind.VALIDATES,
    }
)
_BACKWARD_EDGE_KINDS: Final[frozenset[ProgramEdgeKind]] = frozenset(
    {
        *_CALLER_EDGE_KINDS,
        *_CONSTRUCTOR_EDGE_KINDS,
        *_SCHEMA_EDGE_KINDS,
        ProgramEdgeKind.DATA_FLOW,
        ProgramEdgeKind.STATE_FLOW,
        ProgramEdgeKind.REACHES,
        ProgramEdgeKind.DEPENDS_ON,
        ProgramEdgeKind.PARAMETER_OF,
        ProgramEdgeKind.RETURNS,
        ProgramEdgeKind.FIELD_OF,
        ProgramEdgeKind.IMPORTS,
        ProgramEdgeKind.RE_EXPORTS,
        ProgramEdgeKind.ALIASES,
        ProgramEdgeKind.IMPLEMENTS,
        ProgramEdgeKind.OVERLOADS,
        ProgramEdgeKind.EFFECT_OF,
        ProgramEdgeKind.USES_RESOURCE,
        ProgramEdgeKind.REQUIRES_CAPABILITY,
    }
)

_HYPOTHESIS_PREMISE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.CANDIDATE_IMPLEMENTATION,
        PremiseSourceClass.COMMENT,
        PremiseSourceClass.RUNTIME_WITNESS,
        PremiseSourceClass.HISTORY,
        PremiseSourceClass.VECTOR_ANALOGUE,
        PremiseSourceClass.KNOWLEDGE_GRAPH,
        PremiseSourceClass.MODEL_HYPOTHESIS,
        PremiseSourceClass.GIT_LINEAGE,
    }
)

_STATIC_PREMISE_CLASSES: Final[frozenset[PremiseSourceClass]] = frozenset(
    {
        PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
        PremiseSourceClass.VALUE_PROVENANCE,
        PremiseSourceClass.PROGRAM_GRAPH,
        PremiseSourceClass.SCHEMA_PROTOCOL,
        PremiseSourceClass.LOCAL_STATIC,
        PremiseSourceClass.THEOREM_CORPUS,
        PremiseSourceClass.REVIEWED_CONTRACT,
        PremiseSourceClass.NORMATIVE_SPEC,
        PremiseSourceClass.REVIEWED_CONFORMANCE_TEST,
    }
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
        "theorem_text",
        "proof_script",
        "prompt_body",
        "objective_text",
        "prose",
    }
)

_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "access_token",
        "refresh_token",
        "bearer",
        "credential",
        "ssh_key",
        "client_secret",
    }
)

# Facet kind → preferred next source routes when facts are missing.
_FACET_SOURCE_ROUTES: Final[Mapping[LogicFacetKind, tuple[SourceRouteKind, ...]]] = {
    LogicFacetKind.TYPE: (
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.DATAFLOW,
    ),
    LogicFacetKind.EFFECT: (
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
    ),
    LogicFacetKind.RESOURCE: (
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.RUNTIME_WITNESS,
    ),
    LogicFacetKind.MEMORY: (
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.SOLVER,
    ),
    LogicFacetKind.LIFETIME: (
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.SOLVER,
        SourceRouteKind.REVIEWED_CONTRACT,
    ),
    LogicFacetKind.AUTHORIZATION: (
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.NORMATIVE_SPEC,
        SourceRouteKind.LOCAL_STATIC,
    ),
    LogicFacetKind.STATE: (
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
        SourceRouteKind.LOCAL_STATIC,
    ),
    LogicFacetKind.SCHEMA: (
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.GRAPH,
        SourceRouteKind.LOCAL_STATIC,
    ),
    LogicFacetKind.PLACEMENT: (
        SourceRouteKind.GRAPH,
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.HISTORY,
    ),
    LogicFacetKind.INFORMATION: (
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.GRAPH,
    ),
    LogicFacetKind.ERROR: (
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_TEST,
    ),
    LogicFacetKind.TEMPORAL: (
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.SOLVER,
    ),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _text(
    value: Any, name: str, *, required: bool = False, limit: int = MAX_TEXT_BYTES
) -> str:
    if value is None:
        if required:
            raise LogicGapSlicerError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise LogicGapSlicerError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise LogicGapSlicerError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise LogicGapSlicerBoundsError(f"{name} exceeds its byte bound")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True, limit=MAX_REF_BYTES)
    if any(ch.isspace() for ch in text):
        raise LogicGapSlicerError(f"{name} must be a compact identifier")
    return text


def _enum(value: Any, enum: type[Enum], name: str) -> Enum:
    if isinstance(value, enum):
        return value
    try:
        return enum(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum)
        raise LogicGapSlicerError(f"{name} must be one of: {choices}") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicGapSlicerError(f"{name} must be a boolean")
    return value


def _ids(
    values: Iterable[Any] | None,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCE_COUNT,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values or ():
        text = _identifier(item, name)
        if text not in seen:
            seen.add(text)
            result.append(text)
    if len(result) > limit:
        raise LogicGapSlicerBoundsError(f"{name} exceeds its item bound")
    ordered = tuple(result if preserve_order else sorted(result))
    if required and not ordered:
        raise LogicGapSlicerError(f"{name} must not be empty")
    return ordered


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, float):
        raise LogicGapSlicerError(f"{name} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LogicGapSlicerError(f"{name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise LogicGapSlicerError(f"{name} may not contain source bodies")
            if normalized in _SECRET_KEY_MARKERS:
                raise LogicGapSlicerError(f"{name} may not contain secret material")
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, name)
    elif isinstance(value, (bytes, bytearray)):
        raise LogicGapSlicerError(f"{name} may not contain binary bodies")
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in ("-----begin ", "api_key=", "password="):
            if marker in lowered:
                raise LogicGapSlicerError(f"{name} may not contain secret material")


def _bounded(record: CanonicalContract, name: str) -> None:
    payload = record.to_dict()
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_RECORD_BYTES:
        raise LogicGapSlicerBoundsError(f"{name} exceeds its serialized byte bound")


def _roots(value: Any) -> ProgramLogicAuthorityRoots:
    if isinstance(value, ProgramLogicAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return (
            ProgramLogicAuthorityRoots.from_dict(value)
            if "schema" in value
            else ProgramLogicAuthorityRoots(**value)
        )
    raise LogicGapSlicerError("roots must be ProgramLogicAuthorityRoots")


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise LogicGapSlicerAuthorityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise LogicGapSlicerError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, CONTRACT_VERSION):
        raise LogicGapSlicerError(f"{name} has an unsupported contract version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise LogicGapSlicerError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    return {field_name: payload[field_name] for field_name in fields if field_name in payload}


def _decode_nested(
    value: Any, cls: type[CanonicalContract], field_name: str
) -> CanonicalContract:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        if "schema" in value:
            return cls.from_dict(value)  # type: ignore[attr-defined, return-value]
        return cls(**value)  # type: ignore[arg-type, call-arg, return-value]
    raise LogicGapSlicerError(f"{field_name} must be {cls.__name__}")


def _decode_sequence(
    values: Any,
    cls: type[CanonicalContract],
    field_name: str,
    *,
    limit: int,
    required: bool = False,
) -> tuple[CanonicalContract, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise LogicGapSlicerError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise LogicGapSlicerBoundsError(f"{field_name} exceeds its item bound")
    items: list[CanonicalContract] = []
    seen: set[str] = set()
    for item in raw:
        decoded = _decode_nested(item, cls, field_name)
        if decoded.content_id not in seen:
            seen.add(decoded.content_id)
            items.append(decoded)
    result = tuple(sorted(items, key=lambda record: record.content_id))
    if required and not result:
        raise LogicGapSlicerError(f"{field_name} must not be empty")
    return result


def _decode_enum_sequence(
    values: Any, enum: type[Enum], field_name: str, *, limit: int = MAX_REFERENCE_COUNT
) -> tuple[Enum, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise LogicGapSlicerError(f"{field_name} must be a sequence")
    if len(raw) > limit:
        raise LogicGapSlicerBoundsError(f"{field_name} exceeds its item bound")
    decoded = tuple(
        sorted(
            {_enum(item, enum, field_name) for item in raw},
            key=lambda item: item.value,
        )
    )
    return decoded


# ---------------------------------------------------------------------------
# Loose graph / impact protocols (query without manufacturing edges)
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgramGraphLike(Protocol):
    """Minimal program-graph surface consumed by the slicer."""

    @property
    def graph_id(self) -> str: ...

    def node(self, node_id: str) -> Any: ...

    def edges_from(self, node_id: str) -> Sequence[Any]: ...

    def edges_to(self, node_id: str) -> Sequence[Any]: ...

    def edges_of_kind(self, kind: Any) -> Sequence[Any]: ...

    def find_by_qualified_name(self, name: str) -> Sequence[Any]: ...


@runtime_checkable
class DependencyGraphLike(Protocol):
    """``ProgramDependencyGraph``-shaped façade (has ``.graph``)."""

    @property
    def graph(self) -> ProgramGraphLike | None: ...

    @property
    def roots(self) -> Any: ...


def _resolve_program_graph(
    dependency_graph: ProgramGraphLike | DependencyGraphLike | None,
) -> ProgramGraphLike | None:
    if dependency_graph is None:
        return None
    if isinstance(dependency_graph, ProgramGraphLike) and hasattr(
        dependency_graph, "edges_to"
    ):
        # Prefer concrete graph when both .graph and edge methods exist.
        graph_attr = getattr(dependency_graph, "graph", None)
        if graph_attr is not None and graph_attr is not dependency_graph:
            if isinstance(graph_attr, ProgramGraphLike):
                return graph_attr
        return dependency_graph  # type: ignore[return-value]
    graph = getattr(dependency_graph, "graph", None)
    if graph is not None and isinstance(graph, ProgramGraphLike):
        return graph
    return None


def _edge_kind(edge: Any) -> ProgramEdgeKind | None:
    kind = getattr(edge, "kind", None)
    if kind is None and isinstance(edge, Mapping):
        kind = edge.get("kind")
    if kind is None:
        return None
    try:
        return kind if isinstance(kind, ProgramEdgeKind) else ProgramEdgeKind(kind)
    except (TypeError, ValueError):
        return None


def _edge_source(edge: Any) -> str:
    value = getattr(edge, "source", None)
    if value is None and isinstance(edge, Mapping):
        value = edge.get("source")
    return str(value or "").strip()


def _edge_target(edge: Any) -> str:
    value = getattr(edge, "target", None)
    if value is None and isinstance(edge, Mapping):
        value = edge.get("target")
    return str(value or "").strip()


def _edge_id(edge: Any) -> str:
    value = getattr(edge, "edge_id", None) or getattr(edge, "content_id", None)
    if value is None and isinstance(edge, Mapping):
        value = edge.get("edge_id") or edge.get("content_id")
    if value:
        return str(value)
    return _identity(
        "slice-edge",
        {
            "source": _edge_source(edge),
            "target": _edge_target(edge),
            "kind": str(getattr(getattr(edge, "kind", None), "value", _edge_kind(edge))),
        },
    )


def _node_id(node: Any) -> str:
    if node is None:
        return ""
    value = getattr(node, "node_id", None)
    if value is None and isinstance(node, Mapping):
        value = node.get("node_id")
    return str(value or "").strip()


def _node_qualified_name(node: Any) -> str:
    if node is None:
        return ""
    for attr in ("qualified_name", "name", "symbol_id"):
        value = getattr(node, attr, None)
        if value is None and isinstance(node, Mapping):
            value = node.get(attr)
        if value:
            return str(value)
    return _node_id(node)


# ---------------------------------------------------------------------------
# Slice fact selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceFactSelection(CanonicalContract):
    """One selected or excluded fact reference in a static slice."""

    SCHEMA: ClassVar[str] = SLICE_FACT_SELECTION_SCHEMA

    fact_ref: str
    fact_class: SliceFactClass
    disposition: SliceSelectionDisposition
    reason: str = ""
    exclusion_reason: ExclusionReason | None = None
    supporting_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_ref", _identifier(self.fact_ref, "fact_ref"))
        object.__setattr__(
            self, "fact_class", _enum(self.fact_class, SliceFactClass, "fact_class")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, SliceSelectionDisposition, "disposition"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=False))
        if self.exclusion_reason is not None:
            object.__setattr__(
                self,
                "exclusion_reason",
                _enum(self.exclusion_reason, ExclusionReason, "exclusion_reason"),
            )
        object.__setattr__(
            self, "supporting_refs", _ids(self.supporting_refs, "supporting_refs")
        )
        if (
            self.disposition
            in {
                SliceSelectionDisposition.EXCLUDED,
                SliceSelectionDisposition.CROSS_ROOT,
                SliceSelectionDisposition.BOUND_EXCEEDED,
            }
            and self.exclusion_reason is None
        ):
            raise LogicGapSlicerError(
                "excluded/bound/cross-root facts require exclusion_reason"
            )
        _bounded(self, "slice fact selection")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "fact_ref": self.fact_ref,
            "fact_class": self.fact_class.value,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "exclusion_reason": (
                self.exclusion_reason.value if self.exclusion_reason is not None else ""
            ),
            "supporting_refs": list(self.supporting_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SliceFactSelection":
        fields = (
            "fact_ref",
            "fact_class",
            "disposition",
            "reason",
            "exclusion_reason",
            "supporting_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "slice fact selection")
        if not values.get("exclusion_reason"):
            values["exclusion_reason"] = None
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class AnalyzerCoverage(CanonicalContract):
    """Per-analyzer coverage declaration for one slice."""

    SCHEMA: ClassVar[str] = ANALYZER_COVERAGE_SCHEMA

    analyzer: AnalyzerKind
    completeness: Completeness
    language_refs: tuple[str, ...] = ()
    supported_construct_refs: tuple[str, ...] = ()
    unsupported_construct_refs: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    coverage_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "analyzer", _enum(self.analyzer, AnalyzerKind, "analyzer")
        )
        object.__setattr__(
            self, "completeness", _enum(self.completeness, Completeness, "completeness")
        )
        object.__setattr__(
            self, "language_refs", _ids(self.language_refs, "language_refs")
        )
        object.__setattr__(
            self,
            "supported_construct_refs",
            _ids(self.supported_construct_refs, "supported_construct_refs"),
        )
        object.__setattr__(
            self,
            "unsupported_construct_refs",
            _ids(self.unsupported_construct_refs, "unsupported_construct_refs"),
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        object.__setattr__(
            self, "coverage_ref", _text(self.coverage_ref, "coverage_ref", required=False)
        )
        if (
            self.completeness is Completeness.COMPLETE
            and self.unsupported_construct_refs
        ):
            raise LogicGapSlicerAuthorityError(
                "analyzer coverage cannot claim complete with unsupported constructs"
            )
        if not self.coverage_ref:
            object.__setattr__(
                self,
                "coverage_ref",
                _identity(
                    "analyzer-coverage",
                    {
                        "analyzer": self.analyzer.value,
                        "completeness": self.completeness.value,
                        "unsupported": list(self.unsupported_construct_refs),
                        "bounds": list(self.bound_refs),
                    },
                ),
            )
        _bounded(self, "analyzer coverage")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "analyzer": self.analyzer.value,
            "completeness": self.completeness.value,
            "language_refs": list(self.language_refs),
            "supported_construct_refs": list(self.supported_construct_refs),
            "unsupported_construct_refs": list(self.unsupported_construct_refs),
            "bound_refs": list(self.bound_refs),
            "coverage_ref": self.coverage_ref,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyzerCoverage":
        fields = (
            "analyzer",
            "completeness",
            "language_refs",
            "supported_construct_refs",
            "unsupported_construct_refs",
            "bound_refs",
            "coverage_ref",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "analyzer coverage")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class SccReference(CanonicalContract):
    """Finite SCC reference for cyclic dependency groups (never unrolled)."""

    SCHEMA: ClassVar[str] = SCC_REFERENCE_SCHEMA

    scc_id: str
    member_refs: tuple[str, ...]
    edge_refs: tuple[str, ...] = ()
    origin: str = "graph"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scc_id", _identifier(self.scc_id, "scc_id"))
        object.__setattr__(
            self,
            "member_refs",
            _ids(self.member_refs, "member_refs", required=True, limit=MAX_SCC_MEMBERS),
        )
        object.__setattr__(self, "edge_refs", _ids(self.edge_refs, "edge_refs"))
        object.__setattr__(self, "origin", _text(self.origin, "origin", required=True))
        _bounded(self, "scc reference")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "scc_id": self.scc_id,
            "member_refs": list(self.member_refs),
            "edge_refs": list(self.edge_refs),
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SccReference":
        fields = ("scc_id", "member_refs", "edge_refs", "origin")
        values = _decode_fields(payload, cls.SCHEMA, fields, "scc reference")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# LogicGapSlice
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicGapSlice(CanonicalContract):
    """Dependency-complete minimal static slice for one program-logic goal.

    Contains references only.  Completeness is explicit and never forged.
    """

    SCHEMA: ClassVar[str] = LOGIC_GAP_SLICE_SCHEMA

    roots: ProgramLogicAuthorityRoots
    slice_id: str
    goal_id: str
    completeness: StaticSliceCompleteness
    facts: tuple[SliceFactSelection, ...] = ()
    selected_fact_refs: tuple[str, ...] = ()
    excluded_fact_refs: tuple[str, ...] = ()
    reaching_definition_refs: tuple[str, ...] = ()
    dominance_refs: tuple[str, ...] = ()
    path_condition_refs: tuple[str, ...] = ()
    caller_boundary_refs: tuple[str, ...] = ()
    constructor_boundary_refs: tuple[str, ...] = ()
    schema_boundary_refs: tuple[str, ...] = ()
    consumer_closure_refs: tuple[str, ...] = ()
    scc_refs: tuple[SccReference, ...] = ()
    unknown_frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    analyzer_coverage: tuple[AnalyzerCoverage, ...] = ()
    required_next_source_types: tuple[SourceRouteKind, ...] = ()
    unsupported_construct_refs: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    dependency_slice_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "slice_id", _identifier(self.slice_id, "slice_id"))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(
            self,
            "completeness",
            _enum(self.completeness, StaticSliceCompleteness, "completeness"),
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))

        facts = _decode_sequence(
            self.facts, SliceFactSelection, "facts", limit=MAX_FACTS_PER_SLICE
        )
        object.__setattr__(self, "facts", facts)

        selected = _ids(self.selected_fact_refs, "selected_fact_refs", limit=MAX_FACTS_PER_SLICE)
        excluded = _ids(self.excluded_fact_refs, "excluded_fact_refs", limit=MAX_FACTS_PER_SLICE)
        # Derive from facts when callers omit the explicit lists.
        if not selected:
            selected = tuple(
                sorted(
                    item.fact_ref
                    for item in facts
                    if item.disposition is SliceSelectionDisposition.SELECTED
                )
            )
        if not excluded:
            excluded = tuple(
                sorted(
                    item.fact_ref
                    for item in facts
                    if item.disposition
                    in {
                        SliceSelectionDisposition.EXCLUDED,
                        SliceSelectionDisposition.CROSS_ROOT,
                        SliceSelectionDisposition.BOUND_EXCEEDED,
                        SliceSelectionDisposition.UNSUPPORTED,
                    }
                )
            )
        if set(selected) & set(excluded):
            raise LogicGapSlicerError(
                "selected and excluded fact refs must be disjoint"
            )
        object.__setattr__(self, "selected_fact_refs", selected)
        object.__setattr__(self, "excluded_fact_refs", excluded)

        object.__setattr__(
            self,
            "reaching_definition_refs",
            _ids(self.reaching_definition_refs, "reaching_definition_refs"),
        )
        object.__setattr__(
            self, "dominance_refs", _ids(self.dominance_refs, "dominance_refs")
        )
        object.__setattr__(
            self,
            "path_condition_refs",
            _ids(self.path_condition_refs, "path_condition_refs"),
        )
        object.__setattr__(
            self,
            "caller_boundary_refs",
            _ids(self.caller_boundary_refs, "caller_boundary_refs"),
        )
        object.__setattr__(
            self,
            "constructor_boundary_refs",
            _ids(self.constructor_boundary_refs, "constructor_boundary_refs"),
        )
        object.__setattr__(
            self,
            "schema_boundary_refs",
            _ids(self.schema_boundary_refs, "schema_boundary_refs"),
        )
        object.__setattr__(
            self,
            "consumer_closure_refs",
            _ids(self.consumer_closure_refs, "consumer_closure_refs"),
        )
        object.__setattr__(
            self,
            "scc_refs",
            _decode_sequence(self.scc_refs, SccReference, "scc_refs", limit=MAX_REFERENCE_COUNT),
        )
        object.__setattr__(
            self,
            "unknown_frontier_refs",
            _ids(self.unknown_frontier_refs, "unknown_frontier_refs"),
        )
        object.__setattr__(
            self, "exclusion_refs", _ids(self.exclusion_refs, "exclusion_refs")
        )
        object.__setattr__(
            self,
            "analyzer_coverage",
            _decode_sequence(
                self.analyzer_coverage,
                AnalyzerCoverage,
                "analyzer_coverage",
                limit=MAX_REFERENCE_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "required_next_source_types",
            _decode_enum_sequence(
                self.required_next_source_types, SourceRouteKind, "required_next_source_types"
            ),
        )
        object.__setattr__(
            self,
            "unsupported_construct_refs",
            _ids(self.unsupported_construct_refs, "unsupported_construct_refs"),
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        object.__setattr__(
            self,
            "dependency_slice_refs",
            _ids(self.dependency_slice_refs, "dependency_slice_refs"),
        )
        invalidation = _ids(
            self.invalidation_refs, "invalidation_refs", required=False
        )
        if not invalidation:
            invalidation = tuple(
                sorted(
                    {
                        self.roots.tree_id,
                        self.roots.graph_id,
                        self.roots.corpus_id,
                        self.roots.index_id,
                        self.roots.policy_id,
                        self.roots.overlay_id,
                    }
                )
            )
        object.__setattr__(self, "invalidation_refs", invalidation)

        if self.semantic_authority is not False:
            raise LogicGapSlicerAuthorityError(
                "logic gap slices cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

        # Completeness invariants — never claim complete when residual remains.
        if self.completeness is StaticSliceCompleteness.COMPLETE:
            if self.unknown_frontier_refs:
                raise LogicGapSlicerAuthorityError(
                    "complete slices cannot retain an unknown frontier"
                )
            if self.unsupported_construct_refs:
                raise LogicGapSlicerAuthorityError(
                    "complete slices cannot retain unsupported constructs"
                )
            if self.bound_refs:
                # Bound refs that participated without exhaustion may still be
                # listed; only *active* exhaustion is recorded on incompleteness.
                # Forged complete+bound_exhausted is rejected via bound_refs
                # containing the sentinel "bound:exhausted".
                if any(ref.startswith("bound:exhausted") for ref in self.bound_refs):
                    raise LogicGapSlicerAuthorityError(
                        "complete slices cannot claim completeness under bound exhaustion"
                    )
            for coverage in self.analyzer_coverage:
                # Absent optional analyzers may report UNKNOWN without forging
                # completeness; FRONTIER/UNSUPPORTED on a present analyzer
                # cannot coexist with a complete slice.
                if coverage.completeness is Completeness.UNSUPPORTED:
                    raise LogicGapSlicerAuthorityError(
                        "complete slices cannot include unsupported analyzer coverage"
                    )
                if coverage.completeness is Completeness.FRONTIER:
                    raise LogicGapSlicerAuthorityError(
                        "complete slices cannot include frontier analyzer coverage"
                    )
                if coverage.unsupported_construct_refs:
                    raise LogicGapSlicerAuthorityError(
                        "complete slices cannot include unsupported analyzer constructs"
                    )

        if (
            self.completeness is StaticSliceCompleteness.FRONTIER
            and not self.unknown_frontier_refs
        ):
            raise LogicGapSlicerError(
                "frontier slices require unknown_frontier_refs"
            )
        if (
            self.completeness is StaticSliceCompleteness.UNSUPPORTED
            and not self.unsupported_construct_refs
        ):
            raise LogicGapSlicerError(
                "unsupported slices require unsupported_construct_refs"
            )

        _bounded(self, "logic gap slice")

    @property
    def is_dependency_complete(self) -> bool:
        return self.completeness is StaticSliceCompleteness.COMPLETE

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "slice_id": self.slice_id,
            "goal_id": self.goal_id,
            "completeness": self.completeness.value,
            "facts": [item.to_dict() for item in self.facts],
            "selected_fact_refs": list(self.selected_fact_refs),
            "excluded_fact_refs": list(self.excluded_fact_refs),
            "reaching_definition_refs": list(self.reaching_definition_refs),
            "dominance_refs": list(self.dominance_refs),
            "path_condition_refs": list(self.path_condition_refs),
            "caller_boundary_refs": list(self.caller_boundary_refs),
            "constructor_boundary_refs": list(self.constructor_boundary_refs),
            "schema_boundary_refs": list(self.schema_boundary_refs),
            "consumer_closure_refs": list(self.consumer_closure_refs),
            "scc_refs": [item.to_dict() for item in self.scc_refs],
            "unknown_frontier_refs": list(self.unknown_frontier_refs),
            "exclusion_refs": list(self.exclusion_refs),
            "analyzer_coverage": [item.to_dict() for item in self.analyzer_coverage],
            "required_next_source_types": [
                item.value for item in self.required_next_source_types
            ],
            "unsupported_construct_refs": list(self.unsupported_construct_refs),
            "bound_refs": list(self.bound_refs),
            "dependency_slice_refs": list(self.dependency_slice_refs),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicGapSlice":
        fields = (
            "roots",
            "slice_id",
            "goal_id",
            "completeness",
            "facts",
            "selected_fact_refs",
            "excluded_fact_refs",
            "reaching_definition_refs",
            "dominance_refs",
            "path_condition_refs",
            "caller_boundary_refs",
            "constructor_boundary_refs",
            "schema_boundary_refs",
            "consumer_closure_refs",
            "scc_refs",
            "unknown_frontier_refs",
            "exclusion_refs",
            "analyzer_coverage",
            "required_next_source_types",
            "unsupported_construct_refs",
            "bound_refs",
            "dependency_slice_refs",
            "invalidation_refs",
            "producer_id",
            "semantic_authority",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "logic gap slice")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# InformationDemand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationDemand(CanonicalContract):
    """Explicit information demand for Tactician routing (body-free).

    Mirrors the structural content of :class:`LogicGap` while retaining the
    originating slice identity and required next source types.
    """

    SCHEMA: ClassVar[str] = INFORMATION_DEMAND_SCHEMA

    roots: ProgramLogicAuthorityRoots
    demand_id: str
    goal_id: str
    slice_id: str
    missing_class: GapMissingClass
    disposition: GapDisposition
    observed_fact_ref: str
    required_fact_ref: str
    discrepancy_ref: str
    dependency_slice_refs: tuple[str, ...] = ()
    candidate_source_routes: tuple[SourceRouteKind, ...] = ()
    unknown_frontier_refs: tuple[str, ...] = ()
    coverage_refs: tuple[str, ...] = ()
    severity: str = "mandatory"
    automation_eligible: bool = False
    semantic_authority: bool = False
    invalidation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(self, "demand_id", _identifier(self.demand_id, "demand_id"))
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        object.__setattr__(self, "slice_id", _identifier(self.slice_id, "slice_id"))
        object.__setattr__(
            self,
            "missing_class",
            _enum(self.missing_class, GapMissingClass, "missing_class"),
        )
        object.__setattr__(
            self, "disposition", _enum(self.disposition, GapDisposition, "disposition")
        )
        object.__setattr__(
            self,
            "observed_fact_ref",
            _identifier(self.observed_fact_ref, "observed_fact_ref"),
        )
        object.__setattr__(
            self,
            "required_fact_ref",
            _identifier(self.required_fact_ref, "required_fact_ref"),
        )
        object.__setattr__(
            self, "discrepancy_ref", _identifier(self.discrepancy_ref, "discrepancy_ref")
        )
        object.__setattr__(
            self,
            "dependency_slice_refs",
            _ids(self.dependency_slice_refs, "dependency_slice_refs"),
        )
        object.__setattr__(
            self,
            "candidate_source_routes",
            _decode_enum_sequence(
                self.candidate_source_routes, SourceRouteKind, "candidate_source_routes"
            ),
        )
        object.__setattr__(
            self,
            "unknown_frontier_refs",
            _ids(self.unknown_frontier_refs, "unknown_frontier_refs"),
        )
        object.__setattr__(
            self, "coverage_refs", _ids(self.coverage_refs, "coverage_refs")
        )
        object.__setattr__(
            self, "severity", _identifier(self.severity, "severity")
        )
        object.__setattr__(
            self,
            "automation_eligible",
            _bool(self.automation_eligible, "automation_eligible"),
        )
        if self.semantic_authority is not False:
            raise LogicGapSlicerAuthorityError(
                "information demands cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        invalidation = _ids(self.invalidation_refs, "invalidation_refs")
        if not invalidation:
            invalidation = tuple(
                sorted(
                    {
                        self.roots.tree_id,
                        self.roots.graph_id,
                        self.roots.corpus_id,
                        self.roots.policy_id,
                    }
                )
            )
        object.__setattr__(self, "invalidation_refs", invalidation)

        if self.disposition is GapDisposition.FRONTIER and not self.unknown_frontier_refs:
            raise LogicGapSlicerError("frontier demands require unknown_frontier_refs")
        if self.disposition is GapDisposition.COVERED and not self.coverage_refs:
            raise LogicGapSlicerError("covered demands require coverage_refs")
        if (
            self.disposition is GapDisposition.REQUIRED
            and self.automation_eligible
            and not self.candidate_source_routes
        ):
            raise LogicGapSlicerError(
                "automation-eligible required demands need candidate source routes"
            )
        _bounded(self, "information demand")

    def to_logic_gap(self) -> LogicGap:
        """Project this demand into a canonical :class:`LogicGap@1` record."""
        return LogicGap(
            roots=self.roots,
            gap_id=self.demand_id if self.demand_id.startswith("gap:") else f"gap:{self.demand_id}",
            goal_id=self.goal_id,
            missing_class=self.missing_class,
            disposition=self.disposition,
            observed_fact_ref=self.observed_fact_ref,
            required_fact_ref=self.required_fact_ref,
            discrepancy_ref=self.discrepancy_ref,
            dependency_slice_refs=self.dependency_slice_refs or (self.slice_id,),
            candidate_source_routes=self.candidate_source_routes,
            unknown_frontier_refs=self.unknown_frontier_refs,
            coverage_refs=self.coverage_refs,
            severity=self.severity,
            automation_eligible=self.automation_eligible,
            semantic_authority=False,
            invalidation_refs=self.invalidation_refs,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "demand_id": self.demand_id,
            "goal_id": self.goal_id,
            "slice_id": self.slice_id,
            "missing_class": self.missing_class.value,
            "disposition": self.disposition.value,
            "observed_fact_ref": self.observed_fact_ref,
            "required_fact_ref": self.required_fact_ref,
            "discrepancy_ref": self.discrepancy_ref,
            "dependency_slice_refs": list(self.dependency_slice_refs),
            "candidate_source_routes": [
                item.value for item in self.candidate_source_routes
            ],
            "unknown_frontier_refs": list(self.unknown_frontier_refs),
            "coverage_refs": list(self.coverage_refs),
            "severity": self.severity,
            "automation_eligible": self.automation_eligible,
            "semantic_authority": False,
            "invalidation_refs": list(self.invalidation_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InformationDemand":
        fields = (
            "roots",
            "demand_id",
            "goal_id",
            "slice_id",
            "missing_class",
            "disposition",
            "observed_fact_ref",
            "required_fact_ref",
            "discrepancy_ref",
            "dependency_slice_refs",
            "candidate_source_routes",
            "unknown_frontier_refs",
            "coverage_refs",
            "severity",
            "automation_eligible",
            "semantic_authority",
            "invalidation_refs",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, "information demand")
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Inventory result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicGapSlicingInventory(CanonicalContract):
    """Finite inventory of slices and information demands for a goal set."""

    SCHEMA: ClassVar[str] = LOGIC_GAP_SLICING_INVENTORY_SCHEMA

    roots: ProgramLogicAuthorityRoots
    inventory_id: str
    disposition: InventoryDisposition
    slices: tuple[LogicGapSlice, ...] = ()
    demands: tuple[InformationDemand, ...] = ()
    gaps: tuple[LogicGap, ...] = ()
    residual_goal_ids: tuple[str, ...] = ()
    unsupported_goal_ids: tuple[str, ...] = ()
    bound_refs: tuple[str, ...] = ()
    invalidation_refs: tuple[str, ...] = ()
    producer_id: str = PRODUCER_ID
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", _roots(self.roots))
        object.__setattr__(
            self, "inventory_id", _identifier(self.inventory_id, "inventory_id")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, InventoryDisposition, "disposition"),
        )
        object.__setattr__(self, "producer_id", _identifier(self.producer_id, "producer_id"))
        slices = _decode_sequence(
            self.slices, LogicGapSlice, "slices", limit=MAX_SLICES
        )
        object.__setattr__(
            self,
            "slices",
            tuple(sorted(slices, key=lambda item: item.slice_id)),
        )
        demands = _decode_sequence(
            self.demands, InformationDemand, "demands", limit=MAX_SLICES
        )
        object.__setattr__(
            self,
            "demands",
            tuple(sorted(demands, key=lambda item: item.demand_id)),
        )
        gaps = _decode_sequence(self.gaps, LogicGap, "gaps", limit=MAX_SLICES)
        object.__setattr__(
            self, "gaps", tuple(sorted(gaps, key=lambda item: item.gap_id))
        )
        object.__setattr__(
            self, "residual_goal_ids", _ids(self.residual_goal_ids, "residual_goal_ids")
        )
        object.__setattr__(
            self,
            "unsupported_goal_ids",
            _ids(self.unsupported_goal_ids, "unsupported_goal_ids"),
        )
        object.__setattr__(self, "bound_refs", _ids(self.bound_refs, "bound_refs"))
        invalidation = _ids(self.invalidation_refs, "invalidation_refs")
        if not invalidation:
            invalidation = tuple(
                sorted(
                    {
                        self.roots.tree_id,
                        self.roots.graph_id,
                        self.roots.corpus_id,
                        self.roots.policy_id,
                        self.roots.index_id,
                    }
                )
            )
        object.__setattr__(self, "invalidation_refs", invalidation)
        if self.semantic_authority is not False:
            raise LogicGapSlicerAuthorityError(
                "slicing inventory cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        # Root consistency across nested records.
        for item in self.slices:
            if item.roots.content_id != self.roots.content_id:
                raise LogicGapSlicerAuthorityError(
                    "slice roots must match inventory roots"
                )
        for item in self.demands:
            if item.roots.content_id != self.roots.content_id:
                raise LogicGapSlicerAuthorityError(
                    "demand roots must match inventory roots"
                )
        for item in self.gaps:
            if item.roots.content_id != self.roots.content_id:
                raise LogicGapSlicerAuthorityError(
                    "gap roots must match inventory roots"
                )
        _bounded(self, "logic gap slicing inventory")

    def slice_for_goal(self, goal_id: str) -> LogicGapSlice | None:
        for item in self.slices:
            if item.goal_id == goal_id:
                return item
        return None

    def demand_for_goal(self, goal_id: str) -> InformationDemand | None:
        for item in self.demands:
            if item.goal_id == goal_id:
                return item
        return None

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "roots": self.roots.to_dict(),
            "inventory_id": self.inventory_id,
            "disposition": self.disposition.value,
            "slices": [item.to_dict() for item in self.slices],
            "demands": [item.to_dict() for item in self.demands],
            "gaps": [item.to_dict() for item in self.gaps],
            "residual_goal_ids": list(self.residual_goal_ids),
            "unsupported_goal_ids": list(self.unsupported_goal_ids),
            "bound_refs": list(self.bound_refs),
            "invalidation_refs": list(self.invalidation_refs),
            "producer_id": self.producer_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicGapSlicingInventory":
        fields = (
            "roots",
            "inventory_id",
            "disposition",
            "slices",
            "demands",
            "gaps",
            "residual_goal_ids",
            "unsupported_goal_ids",
            "bound_refs",
            "invalidation_refs",
            "producer_id",
            "semantic_authority",
        )
        values = _decode_fields(
            payload, cls.SCHEMA, fields, "logic gap slicing inventory"
        )
        values["roots"] = _roots(values["roots"])
        value = cls(**values)
        _verify_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Internal draft state
# ---------------------------------------------------------------------------


@dataclass
class _SliceDraft:
    goal: ProgramLogicGoal
    facts: list[SliceFactSelection] = field(default_factory=list)
    selected: set[str] = field(default_factory=set)
    excluded: set[str] = field(default_factory=set)
    reaching: set[str] = field(default_factory=set)
    dominance: set[str] = field(default_factory=set)
    path_conditions: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    constructors: set[str] = field(default_factory=set)
    schemas: set[str] = field(default_factory=set)
    consumers: set[str] = field(default_factory=set)
    sccs: list[SccReference] = field(default_factory=list)
    frontier: set[str] = field(default_factory=set)
    exclusions: set[str] = field(default_factory=set)
    coverage: list[AnalyzerCoverage] = field(default_factory=list)
    next_sources: set[SourceRouteKind] = field(default_factory=set)
    unsupported: set[str] = field(default_factory=set)
    bounds: set[str] = field(default_factory=set)
    dependency_refs: set[str] = field(default_factory=set)
    bound_exhausted: bool = False
    root_mismatch: bool = False
    missing_required: set[str] = field(default_factory=set)

    def select(
        self,
        fact_ref: str,
        fact_class: SliceFactClass,
        *,
        reason: str = "",
        supporting: Sequence[str] = (),
    ) -> None:
        if fact_ref in self.selected or fact_ref in self.excluded:
            return
        if len(self.selected) >= MAX_FACTS_PER_SLICE:
            self.bound_exhausted = True
            self.bounds.add("bound:exhausted:facts")
            self.exclude(
                fact_ref,
                fact_class,
                ExclusionReason.BEYOND_BOUND,
                reason="fact selection bound exhausted",
            )
            return
        self.selected.add(fact_ref)
        self.facts.append(
            SliceFactSelection(
                fact_ref=fact_ref,
                fact_class=fact_class,
                disposition=SliceSelectionDisposition.SELECTED,
                reason=reason or f"selected {fact_class.value}",
                supporting_refs=tuple(supporting),
            )
        )

    def exclude(
        self,
        fact_ref: str,
        fact_class: SliceFactClass,
        exclusion_reason: ExclusionReason,
        *,
        reason: str = "",
        disposition: SliceSelectionDisposition = SliceSelectionDisposition.EXCLUDED,
    ) -> None:
        if fact_ref in self.excluded:
            return
        if fact_ref in self.selected:
            # Prefer selection once admitted; exclusions apply to non-selected only.
            return
        self.excluded.add(fact_ref)
        self.exclusions.add(f"exclusion:{exclusion_reason.value}:{fact_ref}")
        self.facts.append(
            SliceFactSelection(
                fact_ref=fact_ref,
                fact_class=fact_class,
                disposition=disposition,
                reason=reason or exclusion_reason.value,
                exclusion_reason=exclusion_reason,
            )
        )


# ---------------------------------------------------------------------------
# Slicer
# ---------------------------------------------------------------------------


class LogicGapSlicer:
    """Produce conservative static slices and information demands per goal.

    Queries existing graphs and corpora by reference.  Never manufactures
    missing edges, definitions, dominance, or completeness claims.
    """

    def __init__(
        self,
        roots: ProgramLogicAuthorityRoots,
        *,
        max_slice_nodes: int = DEFAULT_MAX_SLICE_NODES,
        max_backward_depth: int = MAX_BACKWARD_DEPTH,
    ) -> None:
        self.roots = _roots(roots)
        if max_slice_nodes < 1 or max_slice_nodes > MAX_FACTS_PER_SLICE:
            raise LogicGapSlicerBoundsError("max_slice_nodes out of supported range")
        if max_backward_depth < 1 or max_backward_depth > MAX_BACKWARD_DEPTH:
            raise LogicGapSlicerBoundsError("max_backward_depth out of supported range")
        self._max_slice_nodes = int(max_slice_nodes)
        self._max_backward_depth = int(max_backward_depth)

    def slice(
        self,
        goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] = (),
        *,
        corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None = None,
        dependency_graph: ProgramGraphLike | DependencyGraphLike | None = None,
        value_provenance: ValueProvenanceGraph | Mapping[str, Any] | None = None,
        impact_closure: ImpactClosureReceipt | Mapping[str, Any] | None = None,
        language_refs: Sequence[str] = ("language:python",),
    ) -> LogicGapSlicingInventory:
        """Slice every supported goal; residual/unsupported goals are recorded."""

        decoded_goals = self._decode_goals(goals)
        if len(decoded_goals) > MAX_GOALS:
            raise LogicGapSlicerBoundsError("goals exceed hard bound")

        corpus_obj = self._decode_corpus(corpus)
        vpg = self._decode_vpg(value_provenance)
        impact = self._decode_impact(impact_closure)
        graph = _resolve_program_graph(dependency_graph)

        # Cross-root checks (fail closed before any completeness claim).
        self._assert_root_alignment(corpus_obj, vpg, impact, graph)

        slices: list[LogicGapSlice] = []
        demands: list[InformationDemand] = []
        gaps: list[LogicGap] = []
        residual: list[str] = []
        unsupported: list[str] = []
        bound_refs: set[str] = {
            self.roots.tree_id,
            self.roots.graph_id,
            self.roots.corpus_id,
            f"bound:max_slice_nodes:{self._max_slice_nodes}",
            f"bound:max_backward_depth:{self._max_backward_depth}",
        }

        for goal in sorted(decoded_goals, key=lambda item: item.goal_id):
            if goal.roots.content_id != self.roots.content_id:
                raise LogicGapSlicerAuthorityError(
                    f"goal {goal.goal_id} roots do not match slicer roots"
                )
            if goal.disposition in {
                GoalDisposition.UNSUPPORTED,
                GoalDisposition.STALE,
            }:
                unsupported.append(goal.goal_id)
                continue
            if goal.disposition in {
                GoalDisposition.DISCHARGED,
                GoalDisposition.ABSTAINED,
            }:
                residual.append(goal.goal_id)
                continue

            draft = _SliceDraft(goal=goal)
            self._seed_goal_facts(draft)
            self._slice_value_provenance(draft, vpg, language_refs)
            self._slice_dependency_graph(draft, graph, language_refs)
            self._slice_impact_closure(draft, impact)
            self._slice_premise_corpus(draft, corpus_obj)
            self._record_required_next_sources(draft)

            gap_slice = self._materialize_slice(draft)
            slices.append(gap_slice)
            demand = self._materialize_demand(draft, gap_slice)
            demands.append(demand)
            gaps.append(demand.to_logic_gap())
            if gap_slice.bound_refs:
                bound_refs.update(gap_slice.bound_refs)
            if gap_slice.completeness in {
                StaticSliceCompleteness.UNSUPPORTED,
            }:
                unsupported.append(goal.goal_id)
            elif gap_slice.completeness in {
                StaticSliceCompleteness.INCOMPLETE,
                StaticSliceCompleteness.FRONTIER,
                StaticSliceCompleteness.STALE,
            }:
                residual.append(goal.goal_id)

        if not slices and not residual and not unsupported:
            disposition = InventoryDisposition.ABSTAINED
        elif any(
            item.completeness is StaticSliceCompleteness.STALE for item in slices
        ):
            disposition = InventoryDisposition.STALE
        elif unsupported and not any(
            item.completeness is StaticSliceCompleteness.COMPLETE for item in slices
        ):
            disposition = InventoryDisposition.UNSUPPORTED
        elif residual or any(
            item.completeness is not StaticSliceCompleteness.COMPLETE for item in slices
        ):
            disposition = InventoryDisposition.PARTIAL
        else:
            disposition = InventoryDisposition.COMPLETE

        inventory_id = _identity(
            "logic-gap-slicing-inventory",
            {
                "roots": self.roots.content_id,
                "slice_ids": [item.slice_id for item in slices],
                "demand_ids": [item.demand_id for item in demands],
                "disposition": disposition.value,
            },
        )
        return LogicGapSlicingInventory(
            roots=self.roots,
            inventory_id=inventory_id,
            disposition=disposition,
            slices=tuple(slices),
            demands=tuple(demands),
            gaps=tuple(gaps),
            residual_goal_ids=tuple(sorted(set(residual))),
            unsupported_goal_ids=tuple(sorted(set(unsupported))),
            bound_refs=tuple(sorted(bound_refs)),
        )

    # ------------------------------------------------------------------
    # Decoding / root alignment
    # ------------------------------------------------------------------

    def _decode_goals(
        self, goals: Sequence[ProgramLogicGoal | Mapping[str, Any]]
    ) -> list[ProgramLogicGoal]:
        result: list[ProgramLogicGoal] = []
        seen: set[str] = set()
        for item in goals or ():
            if isinstance(item, ProgramLogicGoal):
                goal = item
            elif isinstance(item, Mapping):
                goal = (
                    ProgramLogicGoal.from_dict(item)
                    if "schema" in item
                    else ProgramLogicGoal(**item)
                )
            else:
                raise LogicGapSlicerError("goals must be ProgramLogicGoal values")
            if goal.goal_id in seen:
                raise LogicGapSlicerError(f"duplicate goal_id: {goal.goal_id}")
            seen.add(goal.goal_id)
            result.append(goal)
        return result

    def _decode_corpus(
        self, corpus: ProgramLogicPremiseCorpus | Mapping[str, Any] | None
    ) -> ProgramLogicPremiseCorpus | None:
        if corpus is None:
            return None
        if isinstance(corpus, ProgramLogicPremiseCorpus):
            return corpus
        if isinstance(corpus, Mapping):
            return (
                ProgramLogicPremiseCorpus.from_dict(corpus)
                if "schema" in corpus
                else ProgramLogicPremiseCorpus(**corpus)
            )
        raise LogicGapSlicerError("corpus must be ProgramLogicPremiseCorpus")

    def _decode_vpg(
        self, value_provenance: ValueProvenanceGraph | Mapping[str, Any] | None
    ) -> ValueProvenanceGraph | None:
        if value_provenance is None:
            return None
        if isinstance(value_provenance, ValueProvenanceGraph):
            return value_provenance
        if isinstance(value_provenance, Mapping):
            return ValueProvenanceGraph.from_dict(value_provenance)
        raise LogicGapSlicerError("value_provenance must be ValueProvenanceGraph")

    def _decode_impact(
        self, impact_closure: ImpactClosureReceipt | Mapping[str, Any] | None
    ) -> ImpactClosureReceipt | None:
        if impact_closure is None:
            return None
        if isinstance(impact_closure, ImpactClosureReceipt):
            return impact_closure
        if isinstance(impact_closure, Mapping):
            return (
                ImpactClosureReceipt.from_dict(impact_closure)
                if "schema" in impact_closure
                else ImpactClosureReceipt(**impact_closure)
            )
        raise LogicGapSlicerError("impact_closure must be ImpactClosureReceipt")

    def _assert_root_alignment(
        self,
        corpus: ProgramLogicPremiseCorpus | None,
        vpg: ValueProvenanceGraph | None,
        impact: ImpactClosureReceipt | None,
        graph: ProgramGraphLike | None,
    ) -> None:
        if corpus is not None and corpus.roots.content_id != self.roots.content_id:
            raise LogicGapSlicerAuthorityError(
                "premise corpus roots do not match slicer roots"
            )
        if vpg is not None:
            # ValueProvenanceGraph uses ProgramGraphRoots; align tree/graph ids.
            vpg_tree = getattr(vpg.roots, "tree_id", "")
            if vpg_tree and vpg_tree != self.roots.tree_id:
                raise LogicGapSlicerAuthorityError(
                    "value provenance tree_id does not match slicer roots"
                )
        if impact is not None:
            impact_tree = getattr(impact.roots, "candidate_tree_id", None) or getattr(
                impact.roots, "tree_id", ""
            )
            impact_graph = getattr(impact.roots, "graph_id", "")
            if impact_tree and impact_tree != self.roots.tree_id:
                raise LogicGapSlicerAuthorityError(
                    "impact closure tree_id does not match slicer roots"
                )
            if impact_graph and impact_graph != self.roots.graph_id:
                raise LogicGapSlicerAuthorityError(
                    "impact closure graph_id does not match slicer roots"
                )
        if graph is not None:
            graph_id = getattr(graph, "graph_id", "")
            # graph_id is content-addressed; compare via roots if available.
            roots = getattr(graph, "roots", None)
            if roots is not None:
                tree_id = getattr(roots, "tree_id", "")
                if tree_id and tree_id != self.roots.tree_id:
                    raise LogicGapSlicerAuthorityError(
                        "dependency graph tree_id does not match slicer roots"
                    )

    # ------------------------------------------------------------------
    # Seed + per-analyzer slices
    # ------------------------------------------------------------------

    def _seed_goal_facts(self, draft: _SliceDraft) -> None:
        goal = draft.goal
        draft.select(
            goal.positive_statement_ref,
            SliceFactClass.ASSUMPTION,
            reason="goal positive statement",
        )
        for symbol_id in goal.affected_symbol_ids:
            draft.select(
                symbol_id,
                SliceFactClass.GRAPH_NODE,
                reason="goal affected symbol",
            )
        for source_ref in goal.source_refs:
            draft.select(
                source_ref,
                SliceFactClass.ASSUMPTION,
                reason="goal source reference",
            )
        for facet in goal.required_facets:
            draft.select(
                facet.facet_id,
                SliceFactClass.ASSUMPTION,
                reason=f"required facet {facet.kind.value}",
            )
            draft.missing_required.add(facet.facet_id)
        for facet in goal.unsupported_facets:
            draft.unsupported.add(facet.facet_id)
            draft.exclude(
                facet.facet_id,
                SliceFactClass.ASSUMPTION,
                ExclusionReason.UNSUPPORTED_CONSTRUCT,
                reason="goal unsupported facet",
                disposition=SliceSelectionDisposition.UNSUPPORTED,
            )
        for assumption in goal.assumption_refs:
            draft.select(
                assumption,
                SliceFactClass.ASSUMPTION,
                reason="goal assumption",
            )
        for invalidator in goal.invalidation_refs:
            draft.select(
                invalidator,
                SliceFactClass.INVALIDATOR,
                reason="goal invalidator",
            )
        for bound in goal.bound_refs:
            draft.bounds.add(bound)

    def _slice_value_provenance(
        self,
        draft: _SliceDraft,
        vpg: ValueProvenanceGraph | None,
        language_refs: Sequence[str],
    ) -> None:
        if vpg is None:
            draft.coverage.append(
                AnalyzerCoverage(
                    analyzer=AnalyzerKind.VALUE_PROVENANCE,
                    completeness=Completeness.UNKNOWN,
                    language_refs=tuple(language_refs),
                    bound_refs=(),
                )
            )
            draft.next_sources.add(SourceRouteKind.DATAFLOW)
            draft.frontier.add("frontier:value_provenance:absent")
            return

        # Completeness mapping from VPG.
        vpg_complete = vpg.completeness
        if isinstance(vpg_complete, Completeness):
            completeness = vpg_complete
        elif isinstance(vpg_complete, VpgCompleteness):
            completeness = Completeness(vpg_complete.value)
        else:
            completeness = Completeness(str(vpg_complete))

        unsupported: list[str] = []
        bounds: list[str] = []
        for unknown in vpg.unknown_frontier:
            reason = getattr(unknown, "reason", None)
            reason_value = (
                reason.value if isinstance(reason, UnknownReason) else str(reason or "")
            )
            ref = getattr(unknown, "fact_id", None) or getattr(
                unknown, "frontier_id", None
            )
            if ref is None:
                ref = _identity(
                    "unknown-frontier",
                    {
                        "reason": reason_value,
                        "detail": str(getattr(unknown, "detail", "") or ""),
                    },
                )
            ref = str(ref)
            draft.frontier.add(ref)
            if reason_value in {
                UnknownReason.UNSUPPORTED_AST.value,
                UnknownReason.UNSUPPORTED_CFG.value,
                UnknownReason.NATIVE_CALL.value,
                UnknownReason.REFLECTION.value,
                UnknownReason.CONCURRENCY.value,
            }:
                unsupported.append(ref)
                draft.unsupported.add(ref)
            if reason_value == UnknownReason.LOOP_BEYOND_BOUNDS.value:
                bounds.append(ref)
                draft.bound_exhausted = True
                draft.bounds.add("bound:exhausted:loop_unroll")
            if reason_value == UnknownReason.STALE_ROOTS.value:
                draft.root_mismatch = True

        # Select proved reaching definitions / dominance / path conditions
        # that touch goal symbols or procedures.
        symbol_tokens = self._symbol_tokens(draft.goal)
        selected_defs = 0
        for definition in vpg.definitions:
            def_id = definition.def_id
            status = definition.status
            if not self._touches_symbols(definition.variable, definition.procedure_id, symbol_tokens):
                continue
            if status is ProvenanceStatus.UNSUPPORTED:
                draft.exclude(
                    def_id,
                    SliceFactClass.REACHING_DEFINITION,
                    ExclusionReason.UNSUPPORTED_CONSTRUCT,
                    disposition=SliceSelectionDisposition.UNSUPPORTED,
                )
                draft.unsupported.add(def_id)
                continue
            if status is ProvenanceStatus.UNKNOWN:
                draft.frontier.add(def_id)
                draft.exclude(
                    def_id,
                    SliceFactClass.REACHING_DEFINITION,
                    ExclusionReason.NOT_REACHABLE,
                    reason="unknown reaching definition",
                    disposition=SliceSelectionDisposition.FRONTIER,
                )
                continue
            if selected_defs >= self._max_slice_nodes:
                draft.bound_exhausted = True
                draft.bounds.add("bound:exhausted:reaching_definitions")
                draft.exclude(
                    def_id,
                    SliceFactClass.REACHING_DEFINITION,
                    ExclusionReason.BEYOND_BOUND,
                    disposition=SliceSelectionDisposition.BOUND_EXCEEDED,
                )
                continue
            draft.select(
                def_id,
                SliceFactClass.REACHING_DEFINITION,
                reason="proved reaching definition",
            )
            draft.reaching.add(def_id)
            selected_defs += 1
            draft.missing_required.discard(def_id)

        for chain in vpg.def_use_chains:
            if chain.def_id in draft.reaching or chain.use_id in draft.selected:
                draft.select(
                    chain.chain_id if hasattr(chain, "chain_id") and chain.chain_id else _identity(
                        "def-use",
                        {"def": chain.def_id, "use": chain.use_id},
                    ),
                    SliceFactClass.DEF_USE,
                    reason="def-use chain for selected definition",
                    supporting=(chain.def_id, chain.use_id),
                )

        for fact in vpg.dominance_facts:
            if fact.status is not ProvenanceStatus.PROVED:
                if fact.status is ProvenanceStatus.UNSUPPORTED:
                    draft.unsupported.add(fact.fact_id if hasattr(fact, "fact_id") else str(fact))
                continue
            # Keep dominance involving selected definition blocks.
            if fact.dominator_block_id or fact.dominated_block_id:
                fact_id = getattr(fact, "fact_id", None) or _identity(
                    "dominance",
                    {
                        "dom": fact.dominator_block_id,
                        "sub": fact.dominated_block_id,
                        "kind": fact.kind.value,
                    },
                )
                # Only include if related to selected reaching defs' blocks.
                related = False
                for definition in vpg.definitions:
                    if definition.def_id not in draft.reaching:
                        continue
                    if definition.block_id in {
                        fact.dominator_block_id,
                        fact.dominated_block_id,
                    }:
                        related = True
                        break
                if related:
                    draft.select(
                        str(fact_id),
                        SliceFactClass.DOMINANCE,
                        reason="proved dominance for selected defs",
                    )
                    draft.dominance.add(str(fact_id))

        for path in vpg.path_conditions:
            if path.status is ProvenanceStatus.UNSUPPORTED:
                draft.unsupported.add(path.condition_id)
                continue
            # Path conditions for blocks of selected defs.
            related = any(
                definition.block_id == path.block_id
                for definition in vpg.definitions
                if definition.def_id in draft.reaching
            )
            if related:
                if path.status is ProvenanceStatus.PROVED:
                    draft.select(
                        path.condition_id,
                        SliceFactClass.PATH_CONDITION,
                        reason="path condition for selected def block",
                    )
                    draft.path_conditions.add(path.condition_id)
                else:
                    draft.frontier.add(path.condition_id)

        for info in vpg.information_provenances:
            if info.def_id in draft.reaching:
                draft.select(
                    info.provenance_id,
                    SliceFactClass.INFORMATION_PROVENANCE,
                    reason="information provenance for selected def",
                    supporting=(info.def_id,),
                )

        for thread in vpg.interprocedural_threads:
            thread_id = getattr(thread, "thread_id", None) or _identity(
                "iproc-thread", thread.to_dict() if hasattr(thread, "to_dict") else str(thread)
            )
            completeness = getattr(thread, "completeness", None)
            if completeness is not None and str(
                getattr(completeness, "value", completeness)
            ) in {"unsupported", "unknown"}:
                draft.frontier.add(str(thread_id))
                continue
            # Include when either endpoint touches selected symbols.
            symbols = {
                str(getattr(thread, "caller_procedure_id", "") or ""),
                str(getattr(thread, "callee_procedure_id", "") or ""),
            }
            if symbols & symbol_tokens or any(
                token in symbols for token in symbol_tokens
            ):
                draft.select(
                    str(thread_id),
                    SliceFactClass.INTERPROCEDURAL_THREAD,
                    reason="interprocedural thread for goal symbols",
                )

        # Reject forged completeness on the VPG itself.
        if completeness is Completeness.COMPLETE and (
            unsupported or draft.frontier or draft.bound_exhausted
        ):
            raise LogicGapSlicerAuthorityError(
                "value provenance claims complete despite frontier/unsupported/bounds"
            )

        mapped_completeness = completeness
        if draft.bound_exhausted:
            mapped_completeness = Completeness.PARTIAL
        if unsupported:
            mapped_completeness = Completeness.UNSUPPORTED
        elif draft.frontier and mapped_completeness is Completeness.COMPLETE:
            mapped_completeness = Completeness.FRONTIER

        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.VALUE_PROVENANCE,
                completeness=mapped_completeness,
                language_refs=tuple(language_refs),
                unsupported_construct_refs=tuple(sorted(set(unsupported))),
                bound_refs=tuple(sorted(set(bounds))),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.REACHING_DEFINITIONS,
                completeness=(
                    Completeness.COMPLETE
                    if draft.reaching and not draft.bound_exhausted and not unsupported
                    else Completeness.PARTIAL
                    if draft.reaching
                    else Completeness.UNKNOWN
                ),
                language_refs=tuple(language_refs),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.DOMINANCE,
                completeness=(
                    Completeness.COMPLETE
                    if draft.dominance
                    else Completeness.PARTIAL
                    if vpg.dominance_facts
                    else Completeness.UNKNOWN
                ),
                language_refs=tuple(language_refs),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.PATH_CONDITIONS,
                completeness=(
                    Completeness.COMPLETE
                    if draft.path_conditions
                    else Completeness.PARTIAL
                    if vpg.path_conditions
                    else Completeness.UNKNOWN
                ),
                language_refs=tuple(language_refs),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.CONTROL_FLOW,
                completeness=mapped_completeness,
                language_refs=tuple(language_refs),
                unsupported_construct_refs=tuple(sorted(set(unsupported))),
            )
        )

    def _slice_dependency_graph(
        self,
        draft: _SliceDraft,
        graph: ProgramGraphLike | None,
        language_refs: Sequence[str],
    ) -> None:
        if graph is None:
            draft.coverage.append(
                AnalyzerCoverage(
                    analyzer=AnalyzerKind.PROGRAM_DEPENDENCY_GRAPH,
                    completeness=Completeness.UNKNOWN,
                    language_refs=tuple(language_refs),
                )
            )
            draft.next_sources.add(SourceRouteKind.GRAPH)
            draft.frontier.add("frontier:program_dependency_graph:absent")
            return

        symbol_tokens = self._symbol_tokens(draft.goal)
        seed_nodes: list[str] = []
        for token in symbol_tokens:
            # Prefer explicit node ids; fall back to qualified-name lookup.
            node = None
            try:
                node = graph.node(token)
            except Exception:  # noqa: BLE001 — protocol may raise
                node = None
            if node is not None:
                seed_nodes.append(_node_id(node) or token)
                continue
            try:
                matches = graph.find_by_qualified_name(token)
            except Exception:  # noqa: BLE001
                matches = ()
            for match in matches or ():
                nid = _node_id(match)
                if nid:
                    seed_nodes.append(nid)

        if not seed_nodes:
            # Seed from affected_symbol_ids as opaque node ids.
            seed_nodes = list(draft.goal.affected_symbol_ids)

        visited: set[str] = set()
        scc_members_seen: set[str] = set()
        edge_back_links: dict[str, set[str]] = defaultdict(set)
        queue: deque[tuple[str, int]] = deque((nid, 0) for nid in seed_nodes if nid)
        nodes_selected = 0

        while queue:
            node_id, depth = queue.popleft()
            if not node_id or node_id in visited:
                continue
            visited.add(node_id)
            if nodes_selected >= self._max_slice_nodes:
                draft.bound_exhausted = True
                draft.bounds.add("bound:exhausted:slice_nodes")
                draft.exclude(
                    node_id,
                    SliceFactClass.GRAPH_NODE,
                    ExclusionReason.BEYOND_BOUND,
                    disposition=SliceSelectionDisposition.BOUND_EXCEEDED,
                )
                continue
            draft.select(
                node_id,
                SliceFactClass.GRAPH_NODE,
                reason="backward dependency slice node",
            )
            nodes_selected += 1
            draft.dependency_refs.add(node_id)

            if depth >= self._max_backward_depth:
                draft.bound_exhausted = True
                draft.bounds.add("bound:exhausted:backward_depth")
                draft.frontier.add(f"frontier:depth:{node_id}")
                continue

            try:
                inbound = list(graph.edges_to(node_id) or ())
            except Exception:  # noqa: BLE001
                inbound = []
            try:
                outbound = list(graph.edges_from(node_id) or ())
            except Exception:  # noqa: BLE001
                outbound = []

            for edge in inbound + outbound:
                kind = _edge_kind(edge)
                if kind is None:
                    continue
                eid = _edge_id(edge)
                src = _edge_source(edge)
                tgt = _edge_target(edge)
                if kind not in _BACKWARD_EDGE_KINDS and kind is not ProgramEdgeKind.RELATED_TO:
                    draft.exclude(
                        eid,
                        SliceFactClass.GRAPH_EDGE,
                        ExclusionReason.OUT_OF_SCOPE,
                        reason=f"edge kind {kind.value} outside slice vocabulary",
                    )
                    continue
                if kind is ProgramEdgeKind.RELATED_TO:
                    # Nominated-only edges never enter the authoritative slice.
                    draft.exclude(
                        eid,
                        SliceFactClass.GRAPH_EDGE,
                        ExclusionReason.NOMINATING_ONLY,
                        reason="related_to edges are nominating only",
                    )
                    continue

                draft.select(
                    eid,
                    SliceFactClass.GRAPH_EDGE,
                    reason=f"dependency edge {kind.value}",
                    supporting=(src, tgt),
                )
                if src and tgt:
                    edge_back_links[src].add(tgt)
                    edge_back_links[tgt].add(src)

                if kind in _CALLER_EDGE_KINDS:
                    caller = src if tgt == node_id else tgt
                    if caller:
                        draft.callers.add(caller)
                        draft.select(
                            caller,
                            SliceFactClass.CALLER_BOUNDARY,
                            reason="caller boundary",
                        )
                if kind in _CONSTRUCTOR_EDGE_KINDS:
                    ctor = src if tgt == node_id else tgt
                    if ctor:
                        draft.constructors.add(ctor)
                        draft.select(
                            ctor,
                            SliceFactClass.CONSTRUCTOR_BOUNDARY,
                            reason="constructor boundary",
                        )
                if kind in _SCHEMA_EDGE_KINDS:
                    schema = src if tgt == node_id else tgt
                    if schema:
                        draft.schemas.add(schema)
                        draft.select(
                            schema,
                            SliceFactClass.SCHEMA_BOUNDARY,
                            reason="schema boundary",
                        )

                # Walk predecessors (sources of inbound edges).
                predecessor = src if tgt == node_id else None
                if predecessor and predecessor not in visited:
                    queue.append((predecessor, depth + 1))
                # For constructor/schema also walk the other endpoint once.
                if kind in _CONSTRUCTOR_EDGE_KINDS | _SCHEMA_EDGE_KINDS:
                    other = tgt if src == node_id else src
                    if other and other not in visited:
                        queue.append((other, depth + 1))

        # Finite SCCs among visited nodes (Tarjan-lite via mutual reachability
        # approximation on undirected back-links of selected edges).
        sccs = self._finite_sccs(visited, edge_back_links)
        for index, members in enumerate(sccs):
            if len(members) < 2:
                continue
            scc_id = _identity(
                "slice-scc",
                {"goal": draft.goal.goal_id, "members": sorted(members), "i": index},
            )
            scc = SccReference(
                scc_id=scc_id,
                member_refs=tuple(sorted(members)),
                origin="program_dependency_graph",
            )
            draft.sccs.append(scc)
            draft.select(
                scc_id,
                SliceFactClass.SCC_REFERENCE,
                reason="finite SCC reference (cycle not unrolled)",
                supporting=tuple(sorted(members)),
            )
            scc_members_seen.update(members)

        # Graph-level frontier / exclusions when available.
        # Only graph-local frontier/bound state can invalidate the graph's own
        # completeness claim (VPG frontiers must not forge a graph forgery).
        graph_frontier: list[str] = []
        for ref in getattr(graph, "frontier_refs", ()) or ():
            text = str(ref)
            graph_frontier.append(text)
            draft.frontier.add(text)
        for ref in getattr(graph, "exclusion_refs", ()) or ():
            draft.exclusions.add(str(ref))
            draft.exclude(
                str(ref),
                SliceFactClass.EXCLUSION,
                ExclusionReason.EXCLUDED_ROOT,
                reason="graph exclusion ref",
            )

        graph_complete = bool(getattr(graph, "complete", False))
        # Forged completeness: an input graph that claims complete while
        # retaining its own open frontier is rejected.  Slice-budget exhaustion
        # during our walk demotes *our* coverage without accusing the graph.
        if graph_complete and graph_frontier:
            raise LogicGapSlicerAuthorityError(
                "dependency graph claims complete despite an open frontier"
            )
        graph_bound_exhausted = any(
            ref.startswith("bound:exhausted:slice_nodes")
            or ref.startswith("bound:exhausted:backward_depth")
            for ref in draft.bounds
        )

        completeness = Completeness.COMPLETE if graph_complete else Completeness.PARTIAL
        if graph_bound_exhausted:
            completeness = Completeness.PARTIAL
        if graph_frontier:
            completeness = Completeness.FRONTIER
        if any(ref.startswith("unsupported:") for ref in draft.unsupported):
            completeness = Completeness.UNSUPPORTED

        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.PROGRAM_DEPENDENCY_GRAPH,
                completeness=completeness,
                language_refs=tuple(language_refs),
                unsupported_construct_refs=tuple(
                    sorted(r for r in draft.unsupported if r.startswith("unsupported:"))
                ),
                bound_refs=tuple(
                    sorted(r for r in draft.bounds if r.startswith("bound:exhausted"))
                ),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.CALL_RESOLUTION,
                completeness=(
                    Completeness.COMPLETE
                    if draft.callers and not draft.bound_exhausted
                    else Completeness.PARTIAL
                    if draft.callers
                    else Completeness.UNKNOWN
                ),
                language_refs=tuple(language_refs),
            )
        )
        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.SCHEMA_PROTOCOL,
                completeness=(
                    Completeness.COMPLETE
                    if draft.schemas
                    else Completeness.PARTIAL
                    if draft.constructors
                    else Completeness.UNKNOWN
                ),
                language_refs=tuple(language_refs),
            )
        )

    def _slice_impact_closure(
        self, draft: _SliceDraft, impact: ImpactClosureReceipt | None
    ) -> None:
        if impact is None:
            draft.coverage.append(
                AnalyzerCoverage(
                    analyzer=AnalyzerKind.IMPACT_CLOSURE,
                    completeness=Completeness.UNKNOWN,
                )
            )
            return

        for consumer in impact.consumers:
            draft.consumers.add(consumer.consumer_id)
            draft.select(
                consumer.consumer_id,
                SliceFactClass.CONSUMER_CLOSURE,
                reason="impact consumer closure",
                supporting=tuple(consumer.edge_refs),
            )
            if consumer.path_condition_ref:
                draft.path_conditions.add(consumer.path_condition_ref)
                draft.select(
                    consumer.path_condition_ref,
                    SliceFactClass.PATH_CONDITION,
                    reason="consumer path condition",
                )

        for scc in impact.sccs:
            ref = SccReference(
                scc_id=scc.scc_id,
                member_refs=tuple(scc.member_consumer_ids),
                origin="impact_closure",
            )
            draft.sccs.append(ref)
            draft.select(
                scc.scc_id,
                SliceFactClass.SCC_REFERENCE,
                reason="impact SCC (finite cycle reference)",
                supporting=tuple(scc.member_consumer_ids),
            )

        for ref in impact.frontier_node_ids:
            draft.frontier.add(ref)
        for ref in impact.frontier_edge_ids:
            draft.frontier.add(ref)
        for ref in impact.excluded_refs:
            draft.exclusions.add(ref)
            draft.exclude(
                ref,
                SliceFactClass.EXCLUSION,
                ExclusionReason.EXCLUDED_ROOT,
                reason="impact exclusion",
            )
        for ref in impact.resource_bound_refs:
            draft.bounds.add(ref)

        completeness_map = {
            ImpactCompleteness.COMPLETE: Completeness.COMPLETE,
            ImpactCompleteness.PARTIAL_WITH_FRONTIER: Completeness.FRONTIER,
            ImpactCompleteness.ABSTAINED: Completeness.UNKNOWN,
        }
        mapped = completeness_map.get(impact.completeness, Completeness.PARTIAL)

        # Reject forged complete impact with open frontier.
        if (
            impact.completeness is ImpactCompleteness.COMPLETE
            and (impact.frontier_node_ids or impact.frontier_edge_ids)
        ):
            raise LogicGapSlicerAuthorityError(
                "impact closure claims complete despite open frontier"
            )

        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.IMPACT_CLOSURE,
                completeness=mapped,
                bound_refs=tuple(impact.resource_bound_refs),
            )
        )

    def _slice_premise_corpus(
        self, draft: _SliceDraft, corpus: ProgramLogicPremiseCorpus | None
    ) -> None:
        if corpus is None:
            draft.coverage.append(
                AnalyzerCoverage(
                    analyzer=AnalyzerKind.PREMISE_CORPUS,
                    completeness=Completeness.UNKNOWN,
                )
            )
            draft.next_sources.add(SourceRouteKind.LOCAL_STATIC)
            return

        symbol_tokens = self._symbol_tokens(draft.goal)
        selected_count = 0
        for premise in corpus.premises:
            if premise.self_validation:
                draft.exclude(
                    premise.premise_id,
                    SliceFactClass.PREMISE,
                    ExclusionReason.SELF_VALIDATION,
                    reason="self-validating premise excluded",
                )
                continue
            if premise.source_class in _HYPOTHESIS_PREMISE_CLASSES:
                draft.exclude(
                    premise.premise_id,
                    SliceFactClass.PREMISE,
                    ExclusionReason.HYPOTHESIS_ONLY,
                    reason=f"hypothesis premise {premise.source_class.value}",
                )
                continue
            if not self._premise_touches_goal(premise, draft.goal, symbol_tokens):
                draft.exclude(
                    premise.premise_id,
                    SliceFactClass.PREMISE,
                    ExclusionReason.OUT_OF_SCOPE,
                    reason="premise features do not touch goal symbols",
                )
                continue
            if selected_count >= self._max_slice_nodes:
                draft.bound_exhausted = True
                draft.bounds.add("bound:exhausted:premises")
                draft.exclude(
                    premise.premise_id,
                    SliceFactClass.PREMISE,
                    ExclusionReason.BEYOND_BOUND,
                    disposition=SliceSelectionDisposition.BOUND_EXCEEDED,
                )
                continue
            draft.select(
                premise.premise_id,
                SliceFactClass.PREMISE,
                reason=f"static premise {premise.source_class.value}",
                supporting=(premise.statement_ref,),
            )
            selected_count += 1
            # Close required facets when expectation premises cover them.
            if premise.expectation_authority:
                for facet_id in list(draft.missing_required):
                    if facet_id in premise.statement_ref or facet_id in premise.premise_id:
                        draft.missing_required.discard(facet_id)

        for tombstone in corpus.tombstones:
            draft.exclude(
                tombstone.premise_id,
                SliceFactClass.PREMISE,
                ExclusionReason.STALE,
                reason=f"tombstoned premise: {tombstone.reason}",
            )

        draft.coverage.append(
            AnalyzerCoverage(
                analyzer=AnalyzerKind.PREMISE_CORPUS,
                completeness=(
                    Completeness.COMPLETE
                    if selected_count and not draft.bound_exhausted
                    else Completeness.PARTIAL
                    if selected_count
                    else Completeness.UNKNOWN
                ),
            )
        )

    def _record_required_next_sources(self, draft: _SliceDraft) -> None:
        goal = draft.goal
        # Always prefer local static / reviewed contract first.
        draft.next_sources.add(SourceRouteKind.LOCAL_STATIC)
        for facet in goal.required_facets:
            for route in _FACET_SOURCE_ROUTES.get(facet.kind, ()):
                draft.next_sources.add(route)
        if draft.frontier:
            draft.next_sources.add(SourceRouteKind.GRAPH)
            draft.next_sources.add(SourceRouteKind.DATAFLOW)
        if draft.unsupported:
            draft.next_sources.add(SourceRouteKind.SOLVER)
            # Unsupported never routes to LLM by default.
        if draft.missing_required:
            draft.next_sources.add(SourceRouteKind.REVIEWED_CONTRACT)
            draft.next_sources.add(SourceRouteKind.NORMATIVE_SPEC)
        if not draft.reaching and not draft.callers:
            draft.next_sources.add(SourceRouteKind.DATAFLOW)
            draft.next_sources.add(SourceRouteKind.GRAPH)

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_slice(self, draft: _SliceDraft) -> LogicGapSlice:
        completeness = self._decide_completeness(draft)
        slice_id = _identity(
            "logic-gap-slice",
            {
                "goal_id": draft.goal.goal_id,
                "roots": self.roots.content_id,
                "selected": sorted(draft.selected),
                "excluded": sorted(draft.excluded),
                "completeness": completeness.value,
                "frontier": sorted(draft.frontier),
            },
        )
        # Sort facts deterministically.
        facts = tuple(
            sorted(
                draft.facts,
                key=lambda item: (item.disposition.value, item.fact_class.value, item.fact_ref),
            )
        )
        coverage = tuple(
            sorted(draft.coverage, key=lambda item: item.analyzer.value)
        )
        sccs = tuple(sorted(draft.sccs, key=lambda item: item.scc_id))
        next_sources = tuple(
            sorted(draft.next_sources, key=lambda item: item.value)
        )

        # dependency_slice_refs = selected structural refs (not bodies).
        dependency_refs = tuple(
            sorted(
                set(draft.dependency_refs)
                | set(draft.reaching)
                | set(draft.callers)
                | set(draft.constructors)
                | set(draft.schemas)
                | {item.scc_id for item in sccs}
            )
        )

        return LogicGapSlice(
            roots=self.roots,
            slice_id=slice_id,
            goal_id=draft.goal.goal_id,
            completeness=completeness,
            facts=facts,
            selected_fact_refs=tuple(sorted(draft.selected)),
            excluded_fact_refs=tuple(sorted(draft.excluded)),
            reaching_definition_refs=tuple(sorted(draft.reaching)),
            dominance_refs=tuple(sorted(draft.dominance)),
            path_condition_refs=tuple(sorted(draft.path_conditions)),
            caller_boundary_refs=tuple(sorted(draft.callers)),
            constructor_boundary_refs=tuple(sorted(draft.constructors)),
            schema_boundary_refs=tuple(sorted(draft.schemas)),
            consumer_closure_refs=tuple(sorted(draft.consumers)),
            scc_refs=sccs,
            unknown_frontier_refs=tuple(sorted(draft.frontier)),
            exclusion_refs=tuple(sorted(draft.exclusions)),
            analyzer_coverage=coverage,
            required_next_source_types=next_sources,
            unsupported_construct_refs=tuple(sorted(draft.unsupported)),
            bound_refs=tuple(sorted(draft.bounds)),
            dependency_slice_refs=dependency_refs,
        )

    def _decide_completeness(self, draft: _SliceDraft) -> StaticSliceCompleteness:
        if draft.root_mismatch:
            return StaticSliceCompleteness.STALE
        if draft.unsupported and not draft.selected:
            return StaticSliceCompleteness.UNSUPPORTED
        if draft.unsupported and draft.goal.unsupported_facets and not draft.reaching:
            # Goal declared unsupported facets and no static proof material.
            if all(
                facet.unsupported for facet in draft.goal.unsupported_facets
            ) and not draft.goal.required_facets:
                return StaticSliceCompleteness.UNSUPPORTED
        if draft.unsupported and draft.bound_exhausted:
            return StaticSliceCompleteness.UNSUPPORTED
        if draft.unsupported and not draft.reaching and not draft.callers:
            return StaticSliceCompleteness.UNSUPPORTED
        if draft.bound_exhausted:
            return StaticSliceCompleteness.INCOMPLETE
        if draft.frontier:
            return StaticSliceCompleteness.FRONTIER
        if draft.missing_required:
            return StaticSliceCompleteness.INCOMPLETE
        if draft.unsupported:
            return StaticSliceCompleteness.UNSUPPORTED
        # Require at least one analyzer to have non-unknown coverage when
        # we claim complete.
        analyzer_ok = False
        for coverage in draft.coverage:
            if coverage.completeness is Completeness.UNSUPPORTED:
                return StaticSliceCompleteness.UNSUPPORTED
            if coverage.completeness in {
                Completeness.COMPLETE,
                Completeness.PARTIAL,
            }:
                analyzer_ok = True
            if coverage.unsupported_construct_refs:
                return StaticSliceCompleteness.UNSUPPORTED
        if not analyzer_ok and not draft.selected:
            return StaticSliceCompleteness.ABSTAINED
        if not analyzer_ok:
            return StaticSliceCompleteness.INCOMPLETE
        # Complete only when required static material is present and closed.
        if draft.selected and not draft.frontier and not draft.unsupported:
            return StaticSliceCompleteness.COMPLETE
        return StaticSliceCompleteness.INCOMPLETE

    def _materialize_demand(
        self, draft: _SliceDraft, gap_slice: LogicGapSlice
    ) -> InformationDemand:
        completeness = gap_slice.completeness
        missing_class, disposition = self._gap_classification(draft, completeness)

        observed = (
            next(iter(sorted(draft.selected)), "")
            or draft.goal.positive_statement_ref
        )
        required = (
            next(iter(sorted(draft.missing_required)), "")
            or draft.goal.positive_statement_ref
        )
        discrepancy = _identity(
            "discrepancy",
            {
                "goal": draft.goal.goal_id,
                "observed": observed,
                "required": required,
                "completeness": completeness.value,
            },
        )
        coverage_refs = tuple(
            sorted({item.coverage_ref for item in gap_slice.analyzer_coverage})
        )
        automation = (
            disposition is GapDisposition.REQUIRED
            and completeness
            in {
                StaticSliceCompleteness.INCOMPLETE,
                StaticSliceCompleteness.FRONTIER,
            }
            and bool(gap_slice.required_next_source_types)
            and not draft.unsupported
        )
        demand_id = _identity(
            "information-demand",
            {
                "goal": draft.goal.goal_id,
                "slice": gap_slice.slice_id,
                "missing": missing_class.value,
                "disposition": disposition.value,
            },
        )
        # Use gap: prefix for stable LogicGap projection.
        demand_id = f"demand:{demand_id.split(':')[-1]}" if not demand_id.startswith("demand:") else demand_id

        return InformationDemand(
            roots=self.roots,
            demand_id=demand_id,
            goal_id=draft.goal.goal_id,
            slice_id=gap_slice.slice_id,
            missing_class=missing_class,
            disposition=disposition,
            observed_fact_ref=observed,
            required_fact_ref=required,
            discrepancy_ref=discrepancy,
            dependency_slice_refs=(gap_slice.slice_id, *gap_slice.dependency_slice_refs[:32]),
            candidate_source_routes=gap_slice.required_next_source_types,
            unknown_frontier_refs=gap_slice.unknown_frontier_refs,
            coverage_refs=coverage_refs if disposition is GapDisposition.COVERED else coverage_refs,
            severity=(
                "optional"
                if disposition is GapDisposition.OPTIONAL
                else "mandatory"
            ),
            automation_eligible=automation,
        )

    def _gap_classification(
        self, draft: _SliceDraft, completeness: StaticSliceCompleteness
    ) -> tuple[GapMissingClass, GapDisposition]:
        if completeness is StaticSliceCompleteness.STALE:
            return GapMissingClass.FRONTIER, GapDisposition.STALE
        if completeness is StaticSliceCompleteness.UNSUPPORTED:
            return GapMissingClass.UNSUPPORTED_CONSTRUCT, GapDisposition.UNSUPPORTED
        if completeness is StaticSliceCompleteness.FRONTIER:
            return GapMissingClass.FRONTIER, GapDisposition.FRONTIER
        if completeness is StaticSliceCompleteness.COMPLETE:
            return GapMissingClass.PREMISE, GapDisposition.COVERED
        if completeness is StaticSliceCompleteness.ABSTAINED:
            return GapMissingClass.PREMISE, GapDisposition.OPTIONAL

        # Incomplete — classify by dominant missing material.
        if draft.missing_required:
            facet_kinds = {
                facet.kind
                for facet in draft.goal.required_facets
                if facet.facet_id in draft.missing_required
            }
            if LogicFacetKind.INFORMATION in facet_kinds:
                return GapMissingClass.VALUE, GapDisposition.REQUIRED
            if LogicFacetKind.PLACEMENT in facet_kinds:
                return GapMissingClass.PLACEMENT, GapDisposition.REQUIRED
            if LogicFacetKind.SCHEMA in facet_kinds:
                return GapMissingClass.CONSTRUCTION, GapDisposition.REQUIRED
            return GapMissingClass.CONTRACT, GapDisposition.REQUIRED
        if not draft.reaching:
            return GapMissingClass.VALUE, GapDisposition.REQUIRED
        if not draft.callers and draft.goal.family in {
            GoalFamily.BEHAVIOR,
            GoalFamily.PLACEMENT,
        }:
            return GapMissingClass.BEHAVIOR, GapDisposition.REQUIRED
        return GapMissingClass.PREMISE, GapDisposition.REQUIRED

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symbol_tokens(goal: ProgramLogicGoal) -> set[str]:
        tokens: set[str] = set(goal.affected_symbol_ids)
        for ref in goal.source_refs:
            tokens.add(ref)
        tokens.add(goal.positive_statement_ref)
        for facet in goal.required_facets:
            if facet.subject_symbol_id:
                tokens.add(facet.subject_symbol_id)
        return {token for token in tokens if token}

    @staticmethod
    def _touches_symbols(variable: str, procedure_id: str, tokens: set[str]) -> bool:
        if not tokens:
            return True
        candidates = {variable, procedure_id}
        for token in tokens:
            simple = token.rsplit(".", 1)[-1]
            simple = simple.rsplit(":", 1)[-1]
            for candidate in candidates:
                if not candidate:
                    continue
                if candidate == token or candidate.endswith(simple) or simple in candidate:
                    return True
                if token in candidate or candidate in token:
                    return True
        return False

    @staticmethod
    def _premise_touches_goal(
        premise: ProgramLogicPremise,
        goal: ProgramLogicGoal,
        symbol_tokens: set[str],
    ) -> bool:
        if premise.source_class not in _STATIC_PREMISE_CLASSES | _HYPOTHESIS_PREMISE_CLASSES:
            return False
        if premise.source_class in _HYPOTHESIS_PREMISE_CLASSES:
            return False
        features = premise.features
        if features is None:
            # No feature filter → include reviewed/static premises conservatively
            # only when statement/source refs overlap goal sources.
            refs = {premise.statement_ref, premise.premise_id, premise.contract_identity}
            return bool(refs & set(goal.source_refs)) or bool(
                refs & {goal.positive_statement_ref}
            ) or bool(symbol_tokens & refs)
        feature_refs = set(features.symbol_feature_refs) | set(
            features.type_feature_refs
        ) | set(features.import_feature_refs)
        if feature_refs & symbol_tokens:
            return True
        for token in symbol_tokens:
            simple = token.rsplit(".", 1)[-1]
            for feature in feature_refs:
                if simple and simple in feature:
                    return True
        # Expectation premises with empty features still apply when contract
        # identity matches a goal source.
        if premise.expectation_authority:
            return premise.contract_identity in goal.source_refs or (
                premise.statement_ref in goal.source_refs
            )
        return False

    @staticmethod
    def _finite_sccs(
        nodes: set[str], adjacency: Mapping[str, set[str]]
    ) -> list[set[str]]:
        """Return SCCs among *nodes* using Tarjan's algorithm (finite)."""
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        result: list[set[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            for w in adjacency.get(v, ()):
                if w not in nodes:
                    continue
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                component: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.add(w)
                    if w == v:
                        break
                result.append(component)

        for node in sorted(nodes):
            if node not in indices:
                strongconnect(node)
        return result


# ---------------------------------------------------------------------------
# Module entry points
# ---------------------------------------------------------------------------


def slice_logic_gaps(
    roots: ProgramLogicAuthorityRoots | Mapping[str, Any],
    goals: Sequence[ProgramLogicGoal | Mapping[str, Any]] = (),
    **kwargs: Any,
) -> LogicGapSlicingInventory:
    """Module-level entry point for :class:`LogicGapSlicer.slice`."""
    return LogicGapSlicer(_roots(roots)).slice(goals, **kwargs)


def all_analyzer_kinds() -> tuple[AnalyzerKind, ...]:
    return tuple(AnalyzerKind)


def all_slice_fact_classes() -> tuple[SliceFactClass, ...]:
    return tuple(SliceFactClass)


def all_static_slice_completeness() -> tuple[StaticSliceCompleteness, ...]:
    return tuple(StaticSliceCompleteness)


def is_terminal_completeness(value: StaticSliceCompleteness | str) -> bool:
    """True when completeness is a closed residual (never 'solved')."""
    completeness = (
        value
        if isinstance(value, StaticSliceCompleteness)
        else StaticSliceCompleteness(value)
    )
    return completeness in {
        StaticSliceCompleteness.INCOMPLETE,
        StaticSliceCompleteness.UNSUPPORTED,
        StaticSliceCompleteness.FRONTIER,
        StaticSliceCompleteness.STALE,
        StaticSliceCompleteness.ABSTAINED,
    }


def required_next_source_types_for_facets(
    facets: Sequence[LogicFacetKind | str],
) -> tuple[SourceRouteKind, ...]:
    routes: set[SourceRouteKind] = {SourceRouteKind.LOCAL_STATIC}
    for facet in facets:
        kind = facet if isinstance(facet, LogicFacetKind) else LogicFacetKind(facet)
        routes.update(_FACET_SOURCE_ROUTES.get(kind, ()))
    return tuple(sorted(routes, key=lambda item: item.value))
