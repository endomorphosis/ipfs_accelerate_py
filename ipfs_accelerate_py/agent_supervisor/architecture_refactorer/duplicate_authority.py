"""Duplicate-authority detection for competing production owners (PCAR-007).

`DuplicateAuthorityDetector` inspects an accepted `AuthorityOwnershipGraph`
and source-bound `ArchitectureIR`. It emits hard findings for independent
provider-capability and receipt decisions, competing state owners,
compatibility/control bypasses, simulation-to-production flow, Python/CLI/MCP
divergence, re-export authorities, and obsolete-authority tests. Formal
arbitration is recognized rather than treated as a collision. Adapters,
projections, quarantined legacy/simulation paths, and import-only edges are
false positives. Heuristic or opaque signals stay unknown and cannot be
promoted to critical findings. No finding selects a canonical owner or
executes remediation.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
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
    INITIAL_CONCERNS,
    OwnershipBlockerKind,
    resolve_authority_ownership,
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
    "ipfs_accelerate_py/agent-supervisor/duplicate-authority-report@1"
)
DUPLICATE_AUTHORITY_VERSION = 1
DUPLICATE_AUTHORITY_EVIDENCE = "pcar/duplicate-authority-finding@1"
COLLISION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/duplicate-authority-finding@1"
)
COLLISION_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-007-duplicate-authority"
TASK_ID = "PCAR-007"
DEFAULT_FRESHNESS = "pcar-007-duplicate-authority"
EFFECT_CLASS = "read_only_analysis"
DETECTOR_CAN_AUTHORIZE_CHANGES = False
DETECTOR_CAN_SELECT_OWNER = False
DETECTOR_CAN_REMEDIATE = False
HEURISTIC_CRITICAL_PROMOTION_PROHIBITED = True
CONTENT_IDENTITY_IS_NOT_AUTHORITY = True
REEXPORT_IS_NOT_AUTHORITY = True
SILENT_ARBITRATION_PROHIBITED = True
UNKNOWN_PRODUCTION_OWNER_BLOCKS = True

_UNKNOWN_FIELD_MESSAGE = "unknown duplicate-authority field"
_MISSING_FIELD_MESSAGE = "missing duplicate-authority field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_CONFIDENCE_RANK = {
    Confidence.EXACT: 0,
    Confidence.CONSERVATIVE: 1,
    Confidence.HEURISTIC: 2,
    Confidence.OPAQUE: 3,
}

_PRODUCTION_KINDS = frozenset(
    {
        NodeKind.AUTHORITY,
        NodeKind.PROVIDER,
        NodeKind.POLICY,
        NodeKind.STATE,
        NodeKind.RECEIPT,
        NodeKind.OPERATION,
        NodeKind.ENTRYPOINT,
        NodeKind.PROOF,
    }
)
_DECISION_EDGE_KINDS = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.EXECUTES,
        EdgeKind.IMPLEMENTS,
        EdgeKind.CONFIRMS,
        EdgeKind.PROVES,
    }
)
_PROVIDER_EDGE_KINDS = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.EXECUTES,
        EdgeKind.IMPLEMENTS,
    }
)
_RECEIPT_EDGE_KINDS = frozenset(
    {
        EdgeKind.CONFIRMS,
        EdgeKind.PROVES,
        EdgeKind.AUTHORIZES,
        EdgeKind.SERIALIZES,
        EdgeKind.GENERATES,
    }
)
_STATE_EDGE_KINDS = frozenset(
    {EdgeKind.PERSISTS, EdgeKind.WRITES, EdgeKind.MUTATES}
)
_FLOW_EDGE_KINDS = frozenset(
    {
        EdgeKind.CALLS,
        EdgeKind.CONSTRUCTS,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.AUTHORIZES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.CONFIRMS,
        EdgeKind.EXECUTES,
        EdgeKind.PERSISTS,
        EdgeKind.IMPLEMENTS,
        EdgeKind.GENERATES,
        EdgeKind.FALLBACKS_TO,
    }
)
_EXPLAINING_EDGE_KINDS = frozenset(
    {
        EdgeKind.ADAPTS,
        EdgeKind.GENERATES,
        EdgeKind.SUPERSEDES,
        EdgeKind.DEPRECATES,
        EdgeKind.FALLBACKS_TO,
    }
)
_BYPASS_EDGE_KINDS = frozenset(
    {
        EdgeKind.AUTHORIZES,
        EdgeKind.EVALUATES_POLICY,
        EdgeKind.CONFIRMS,
        EdgeKind.EXECUTES,
        EdgeKind.WRITES,
        EdgeKind.MUTATES,
        EdgeKind.PERSISTS,
        EdgeKind.CALLS,
        EdgeKind.IMPLEMENTS,
    }
)
_SURFACE_EDGE_KINDS = frozenset(
    {
        EdgeKind.IMPLEMENTS,
        EdgeKind.EXECUTES,
        EdgeKind.CALLS,
        EdgeKind.AUTHORIZES,
    }
)


class DuplicateAuthorityError(ArchitectureContractError):
    """Fail-closed duplicate-authority contract violation."""


class DuplicateAuthorityAuthorityError(DuplicateAuthorityError):
    """Raised when the detector is asked to remediate or select an owner."""


class CollisionKind(str, Enum):
    """Closed duplicate-authority finding vocabulary (PCAR-PLAN-R1)."""

    INDEPENDENT_PROVIDER_CAPABILITY = "independent_provider_capability"
    INDEPENDENT_RECEIPT_PRODUCER = "independent_receipt_producer"
    COMPETING_STATE_OWNER = "competing_state_owner"
    COMPATIBILITY_BYPASS = "compatibility_bypass"
    CONTROL_BYPASS = "control_bypass"
    SIMULATION_TO_PRODUCTION_FLOW = "simulation_to_production_flow"
    PYTHON_CLI_MCP_DIVERGENCE = "python_cli_mcp_divergence"
    REEXPORT_AUTHORITY = "reexport_authority"
    OBSOLETE_AUTHORITY_TEST = "obsolete_authority_test"
    UNKNOWN_PRODUCTION_OWNER = "unknown_production_owner"
    MULTIPLE_PRODUCTION_AUTHORITIES = "multiple_production_authorities"


REQUIRED_DETECTIONS: tuple[CollisionKind, ...] = (
    CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
    CollisionKind.INDEPENDENT_RECEIPT_PRODUCER,
    CollisionKind.COMPETING_STATE_OWNER,
    CollisionKind.COMPATIBILITY_BYPASS,
    CollisionKind.CONTROL_BYPASS,
    CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
    CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
    CollisionKind.REEXPORT_AUTHORITY,
    CollisionKind.OBSOLETE_AUTHORITY_TEST,
)
CLOSED_COLLISION_KINDS: frozenset[str] = frozenset(
    item.value for item in CollisionKind
)
BYPASS_KINDS: frozenset[CollisionKind] = frozenset(
    {
        CollisionKind.COMPATIBILITY_BYPASS,
        CollisionKind.CONTROL_BYPASS,
        CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
    }
)
BLOCKER_KINDS: frozenset[CollisionKind] = frozenset(
    {
        CollisionKind.UNKNOWN_PRODUCTION_OWNER,
        CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES,
        CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
        CollisionKind.CONTROL_BYPASS,
        CollisionKind.COMPATIBILITY_BYPASS,
    }
)


class FindingDisposition(str, Enum):
    """Closed finding-disposition vocabulary."""

    COLLISION = "collision"
    FALSE_POSITIVE = "false_positive"
    UNKNOWN = "unknown"
    BLOCKER = "blocker"


CLOSED_FINDING_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in FindingDisposition
)


class SurfaceKind(str, Enum):
    """Closed Python/CLI/MCP surface vocabulary."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"
    UNKNOWN = "unknown"


CLOSED_SURFACES: frozenset[str] = frozenset(item.value for item in SurfaceKind)
REQUIRED_SURFACES: tuple[SurfaceKind, ...] = (
    SurfaceKind.PYTHON,
    SurfaceKind.CLI,
    SurfaceKind.MCP,
)

_OWNERSHIP_BLOCKER_TO_COLLISION = {
    OwnershipBlockerKind.UNKNOWN_OWNER: CollisionKind.UNKNOWN_PRODUCTION_OWNER,
    OwnershipBlockerKind.UNKNOWN_PRODUCTION_OWNER: (
        CollisionKind.UNKNOWN_PRODUCTION_OWNER
    ),
    OwnershipBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES: (
        CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES
    ),
    OwnershipBlockerKind.MISSING_ARBITRATION: (
        CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES
    ),
    OwnershipBlockerKind.UNCLASSIFIED_COMPETITOR: (
        CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES
    ),
    OwnershipBlockerKind.REEXPORT_CLAIMED_AUTHORITY: CollisionKind.REEXPORT_AUTHORITY,
    OwnershipBlockerKind.SIMULATED_AS_LIVE: (
        CollisionKind.SIMULATION_TO_PRODUCTION_FLOW
    ),
    OwnershipBlockerKind.SILENT_ARBITRATION: (
        CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES
    ),
}

_COLLISION_FIELDS = frozenset(
    {
        "concern",
        "content_identity",
        "disposition",
        "edge_ids",
        "formally_arbitrated",
        "kind",
        "message",
        "node_ids",
        "provenance",
        "reachability_path",
        "schema",
        "surfaces",
        "version",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "blockers",
        "can_authorize_changes",
        "can_remediate",
        "can_select_owner",
        "collisions",
        "content_identity",
        "false_positives",
        "findings",
        "freshness",
        "one_owner_invariant_holds",
        "ownership_graph_identity",
        "recognized_arbitrations",
        "repository_tree",
        "schema",
        "unknowns",
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


def _require_ordered_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DuplicateAuthorityError(f"{name} must be a list of strings")
    return tuple(
        _require_text(item, f"{name} item", error_type=DuplicateAuthorityError)
        for item in value
    )


def _require_architecture_ir(
    graph: ArchitectureIR | Mapping[str, Any],
) -> ArchitectureIR:
    if isinstance(graph, ArchitectureIR):
        return graph
    try:
        return ArchitectureIR.from_mapping(graph)
    except ArchitectureContractError as exc:
        raise DuplicateAuthorityError(str(exc)) from exc


def _require_ownership(
    graph: AuthorityOwnershipGraph | Mapping[str, Any] | None,
) -> AuthorityOwnershipGraph | None:
    if graph is None:
        return None
    if isinstance(graph, AuthorityOwnershipGraph):
        return graph
    if isinstance(graph, Mapping):
        try:
            return AuthorityOwnershipGraph.from_mapping(graph)
        except ArchitectureContractError as exc:
            raise DuplicateAuthorityError(str(exc)) from exc
    raise DuplicateAuthorityError("ownership graph must be an object or null")


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> DuplicateAuthorityError:
    if isinstance(exc, DuplicateAuthorityError):
        return exc
    return DuplicateAuthorityError(str(exc))


def _optional_cid(value: Any, name: str) -> str:
    text = value if type(value) is str else _require_text(
        value, name, error_type=DuplicateAuthorityError
    )
    if text == "":
        return ""
    return _validate_dag_json_cid(
        _require_text(text, name, error_type=DuplicateAuthorityError)
    )


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


def _weakest_confidence(
    nodes: Iterable[ArchitectureNode],
    edges: Iterable[ArchitectureEdge],
) -> Confidence:
    weakest = Confidence.EXACT
    rank = _CONFIDENCE_RANK[weakest]
    for fact in (*nodes, *edges):
        current = fact.provenance.confidence
        current_rank = _CONFIDENCE_RANK[current]
        if current_rank > rank:
            weakest = current
            rank = current_rank
    return weakest


def _representative_provenance(
    nodes: Sequence[ArchitectureNode],
    edges: Sequence[ArchitectureEdge],
) -> SourceFactIdentity:
    facts: tuple[ArchitectureNode | ArchitectureEdge, ...] = (*nodes, *edges)
    if not facts:
        raise DuplicateAuthorityError("finding provenance requires a source fact")
    ordered = sorted(
        facts,
        key=lambda item: (
            _CONFIDENCE_RANK[item.provenance.confidence],
            item.provenance.span.path,
            item.provenance.span.start_line,
        ),
    )
    return ordered[0].provenance


def _surface_from_path(path: str) -> SurfaceKind:
    lowered = path.replace("\\", "/").lower()
    name = PurePosixPath(lowered).stem
    parts = tuple(part for part in lowered.split("/") if part)
    if any("mcp" in part for part in (*parts, name)):
        return SurfaceKind.MCP
    if (
        any(part in {"cli", "bin"} or part.endswith("-cli") for part in parts)
        or "cli" in name
        or lowered.endswith("/cli.py")
    ):
        return SurfaceKind.CLI
    if PurePosixPath(lowered).suffix == ".py":
        return SurfaceKind.PYTHON
    return SurfaceKind.UNKNOWN


def _surface_of(node: ArchitectureNode) -> SurfaceKind:
    if node.kind is NodeKind.ENTRYPOINT:
        return _surface_from_path(node.provenance.span.path)
    path = node.provenance.span.path
    classified = _surface_from_path(path)
    if classified is SurfaceKind.CLI or classified is SurfaceKind.MCP:
        return classified
    return SurfaceKind.UNKNOWN


@dataclass(frozen=True)
class AuthorityCollision:
    """One evidence-bound duplicate-authority finding."""

    kind: CollisionKind
    disposition: FindingDisposition
    concern: ConcernKind
    message: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    provenance: SourceFactIdentity
    reachability_path: tuple[str, ...] = ()
    surfaces: tuple[SurfaceKind, ...] = ()
    formally_arbitrated: bool = False
    schema: str = COLLISION_SCHEMA
    version: int = COLLISION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != COLLISION_SCHEMA:
            raise DuplicateAuthorityError("unexpected duplicate-authority-finding schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != COLLISION_VERSION:
            raise DuplicateAuthorityError(
                "unexpected duplicate-authority-finding version"
            )
        kind = _closed_enum(
            self.kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        disposition = _closed_enum(
            self.disposition,
            FindingDisposition,
            "finding disposition",
            error_type=DuplicateAuthorityError,
        )
        concern = _closed_enum(
            self.concern, ConcernKind, "concern", error_type=DuplicateAuthorityError
        )
        message = _require_text(self.message, "message", error_type=DuplicateAuthorityError)
        node_ids = _require_text_tuple(self.node_ids, "node_ids")
        if not node_ids:
            raise DuplicateAuthorityError("finding node_ids must be nonempty")
        if any(_looks_like_content_identity(item) for item in node_ids):
            raise DuplicateAuthorityError(
                "content identity is not inferred to be authority"
            )
        edge_ids = _require_text_tuple(self.edge_ids, "edge_ids")
        reachability_path = _require_ordered_text_tuple(
            self.reachability_path, "reachability_path"
        )
        if isinstance(self.surfaces, SurfaceKind):
            surfaces = (self.surfaces,)
        elif isinstance(self.surfaces, str):
            surfaces = (
                _closed_enum(
                    self.surfaces,
                    SurfaceKind,
                    "surface",
                    error_type=DuplicateAuthorityError,
                ),
            )
        elif isinstance(self.surfaces, (bytes, bytearray)) or not isinstance(
            self.surfaces, Sequence
        ):
            raise DuplicateAuthorityError("surfaces must be a list of surface kinds")
        else:
            surfaces = tuple(
                _closed_enum(
                    item, SurfaceKind, "surface", error_type=DuplicateAuthorityError
                )
                for item in self.surfaces
            )
        surfaces = tuple(sorted(set(surfaces), key=lambda item: item.value))
        if type(self.formally_arbitrated) is not bool:
            raise DuplicateAuthorityError("formally_arbitrated must be a boolean")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        if (
            disposition in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
            and provenance.confidence in NON_PROBATIVE_CONFIDENCE
            and kind is not CollisionKind.UNKNOWN_PRODUCTION_OWNER
        ):
            raise DuplicateAuthorityError(
                "heuristic or opaque facts cannot prove a critical duplicate-authority finding"
            )
        if self.formally_arbitrated and disposition is FindingDisposition.COLLISION:
            raise DuplicateAuthorityError(
                "formally arbitrated competitors are not collisions"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "concern", concern)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "reachability_path", reachability_path)
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "formally_arbitrated", self.formally_arbitrated)
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
                raise DuplicateAuthorityError("finding content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "concern": self.concern.value,
            "disposition": self.disposition.value,
            "edge_ids": list(self.edge_ids),
            "formally_arbitrated": self.formally_arbitrated,
            "kind": self.kind.value,
            "message": self.message,
            "node_ids": list(self.node_ids),
            "provenance": self.provenance.to_dict(),
            "reachability_path": list(self.reachability_path),
            "schema": self.schema,
            "surfaces": [item.value for item in self.surfaces],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError("finding content identity mismatch")
        return {**payload, "content_identity": identity}

    @property
    def is_bypass(self) -> bool:
        return self.kind in BYPASS_KINDS

    @property
    def is_surface_divergence(self) -> bool:
        return self.kind is CollisionKind.PYTHON_CLI_MCP_DIVERGENCE

    @property
    def fails_closed(self) -> bool:
        return self.disposition is FindingDisposition.BLOCKER

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AuthorityCollision":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _COLLISION_FIELDS)
        try:
            finding = cls(
                kind=mapping["kind"],
                disposition=mapping["disposition"],
                concern=mapping["concern"],
                message=mapping["message"],
                node_ids=mapping["node_ids"],
                edge_ids=mapping["edge_ids"],
                provenance=mapping["provenance"],
                reachability_path=mapping["reachability_path"],
                surfaces=mapping["surfaces"],
                formally_arbitrated=mapping["formally_arbitrated"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != finding.content_identity:
            raise DuplicateAuthorityError("finding content identity mismatch")
        return finding

    from_dict = from_mapping


def _finding_tuple(value: Any, name: str) -> tuple[AuthorityCollision, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DuplicateAuthorityError(f"{name} must be a list of finding objects")
    findings = tuple(
        item if isinstance(item, AuthorityCollision) else AuthorityCollision.from_mapping(item)
        for item in value
    )
    ordered = tuple(
        sorted(
            findings,
            key=lambda item: (
                item.kind.value,
                item.concern.value,
                item.node_ids,
                item.edge_ids,
                item.content_identity,
            ),
        )
    )
    identities = tuple(item.content_identity for item in ordered)
    if len(identities) != len(set(identities)):
        raise DuplicateAuthorityError(f"{name} content identities must be unique")
    return ordered


def _filter_disposition(
    findings: Sequence[AuthorityCollision],
    disposition: FindingDisposition,
) -> tuple[AuthorityCollision, ...]:
    return tuple(item for item in findings if item.disposition is disposition)


@dataclass(frozen=True)
class DuplicateAuthorityReport:
    """Deterministic duplicate-authority findings for one ArchitectureIR."""

    architecture_ir_identity: str
    repository_tree: str
    freshness: str
    findings: tuple[AuthorityCollision, ...]
    ownership_graph_identity: str = ""
    recognized_arbitrations: tuple[str, ...] = ()
    schema: str = DUPLICATE_AUTHORITY_SCHEMA
    version: int = DUPLICATE_AUTHORITY_VERSION
    can_authorize_changes: bool = DETECTOR_CAN_AUTHORIZE_CHANGES
    can_select_owner: bool = DETECTOR_CAN_SELECT_OWNER
    can_remediate: bool = DETECTOR_CAN_REMEDIATE
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=DuplicateAuthorityError)
        if schema != DUPLICATE_AUTHORITY_SCHEMA:
            raise DuplicateAuthorityError("unexpected duplicate-authority-report schema")
        version = _require_int(self.version, "version", error_type=DuplicateAuthorityError)
        if version != DUPLICATE_AUTHORITY_VERSION:
            raise DuplicateAuthorityError(
                "unexpected duplicate-authority-report version"
            )
        if self.can_authorize_changes is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot authorize changes"
            )
        if self.can_select_owner is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot select a canonical owner"
            )
        if self.can_remediate is not False:
            raise DuplicateAuthorityError(
                "duplicate-authority detector cannot execute remediation"
            )
        architecture_ir_identity = _validate_dag_json_cid(
            _require_text(
                self.architecture_ir_identity,
                "architecture_ir_identity",
                error_type=DuplicateAuthorityError,
            )
        )
        ownership_graph_identity = _optional_cid(
            self.ownership_graph_identity, "ownership_graph_identity"
        )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=DuplicateAuthorityError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=DuplicateAuthorityError
        )
        findings = _finding_tuple(self.findings, "findings")
        recognized = _require_text_tuple(
            self.recognized_arbitrations, "recognized_arbitrations"
        )
        for item in recognized:
            _validate_dag_json_cid(item)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "ownership_graph_identity", ownership_graph_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "recognized_arbitrations", recognized)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_select_owner", False)
        object.__setattr__(self, "can_remediate", False)
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
                raise DuplicateAuthorityError("report content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "architecture_ir_identity": self.architecture_ir_identity,
            "blockers": [item.to_dict() for item in self.blockers],
            "can_authorize_changes": False,
            "can_remediate": False,
            "can_select_owner": False,
            "collisions": [item.to_dict() for item in self.collisions],
            "false_positives": [item.to_dict() for item in self.false_positives],
            "findings": [item.to_dict() for item in self.findings],
            "freshness": self.freshness,
            "one_owner_invariant_holds": self.one_owner_invariant_holds,
            "ownership_graph_identity": self.ownership_graph_identity,
            "recognized_arbitrations": list(self.recognized_arbitrations),
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "unknowns": [item.to_dict() for item in self.unknowns],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise DuplicateAuthorityError("report content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def collisions(self) -> tuple[AuthorityCollision, ...]:
        return _filter_disposition(self.findings, FindingDisposition.COLLISION)

    @property
    def blockers(self) -> tuple[AuthorityCollision, ...]:
        return _filter_disposition(self.findings, FindingDisposition.BLOCKER)

    @property
    def false_positives(self) -> tuple[AuthorityCollision, ...]:
        return _filter_disposition(self.findings, FindingDisposition.FALSE_POSITIVE)

    @property
    def unknowns(self) -> tuple[AuthorityCollision, ...]:
        return _filter_disposition(self.findings, FindingDisposition.UNKNOWN)

    @property
    def one_owner_invariant_holds(self) -> bool:
        return not self.collisions and not self.blockers

    @property
    def fails_closed(self) -> bool:
        return bool(self.blockers)

    @property
    def bypass_findings(self) -> tuple[AuthorityCollision, ...]:
        return tuple(item for item in self.findings if item.is_bypass)

    @property
    def surface_divergence_findings(self) -> tuple[AuthorityCollision, ...]:
        return tuple(item for item in self.findings if item.is_surface_divergence)

    def findings_of(self, kind: CollisionKind | str) -> tuple[AuthorityCollision, ...]:
        closed = _closed_enum(
            kind, CollisionKind, "collision kind", error_type=DuplicateAuthorityError
        )
        return tuple(item for item in self.findings if item.kind is closed)

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("change")

    def select_owner(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_owner_selection("select")

    def remediate(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("remediate")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DuplicateAuthorityReport":
        mapping = _require_mapping(payload, error_type=DuplicateAuthorityError)
        _require_fields(mapping, _REPORT_FIELDS)
        report = cls(
            architecture_ir_identity=mapping["architecture_ir_identity"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            findings=mapping["findings"],
            ownership_graph_identity=mapping["ownership_graph_identity"],
            recognized_arbitrations=mapping["recognized_arbitrations"],
            schema=mapping["schema"],
            version=mapping["version"],
            can_authorize_changes=mapping["can_authorize_changes"],
            can_select_owner=mapping["can_select_owner"],
            can_remediate=mapping["can_remediate"],
        )
        if mapping["content_identity"] != report.content_identity:
            raise DuplicateAuthorityError("report content identity mismatch")
        if mapping["collisions"] != [item.to_dict() for item in report.collisions]:
            raise DuplicateAuthorityError("collisions projection mismatch")
        if mapping["blockers"] != [item.to_dict() for item in report.blockers]:
            raise DuplicateAuthorityError("blockers projection mismatch")
        if mapping["false_positives"] != [
            item.to_dict() for item in report.false_positives
        ]:
            raise DuplicateAuthorityError("false_positives projection mismatch")
        if mapping["unknowns"] != [item.to_dict() for item in report.unknowns]:
            raise DuplicateAuthorityError("unknowns projection mismatch")
        if mapping["one_owner_invariant_holds"] is not report.one_owner_invariant_holds:
            raise DuplicateAuthorityError("one_owner_invariant_holds projection mismatch")
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


def refuse_remediation(action: str) -> None:
    """Reject attempts to treat a finding as change or consolidation authority."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityAuthorityError(
        f"duplicate-authority detector cannot {name}"
    )


def refuse_owner_selection(action: str) -> None:
    """Reject attempts to select a canonical owner from a finding."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityAuthorityError(
        f"duplicate-authority detector cannot {name} a canonical owner"
    )


def refuse_heuristic_promotion(action: str = "promote") -> None:
    """Reject heuristic-only promotion of a critical finding."""

    name = _require_text(action, "action", error_type=DuplicateAuthorityError)
    raise DuplicateAuthorityError(
        f"heuristic-only critical finding {name} is prohibited"
    )


def lookup_owner_by_content_identity(*_args: Any, **_kwargs: Any) -> None:
    """Content identity never selects or proves a canonical owner."""

    raise DuplicateAuthorityError("content identity is not inferred to be authority")


@dataclass(frozen=True)
class _OwnershipIndex:
    graph: AuthorityOwnershipGraph | None
    canonical_ids: frozenset[str]
    adapter_ids: frozenset[str]
    projection_ids: frozenset[str]
    legacy_ids: frozenset[str]
    simulation_ids: frozenset[str]
    unknown_ids: frozenset[str]
    arbitrated_groups: tuple[frozenset[str], ...]
    arbitration_identities: tuple[str, ...]
    explained_ids: frozenset[str]


def _index_ownership(
    ownership: AuthorityOwnershipGraph | None,
) -> _OwnershipIndex:
    if ownership is None:
        empty: frozenset[str] = frozenset()
        return _OwnershipIndex(
            graph=None,
            canonical_ids=empty,
            adapter_ids=empty,
            projection_ids=empty,
            legacy_ids=empty,
            simulation_ids=empty,
            unknown_ids=empty,
            arbitrated_groups=(),
            arbitration_identities=(),
            explained_ids=empty,
        )
    canonical: set[str] = set()
    adapters: set[str] = set()
    projections: set[str] = set()
    legacy: set[str] = set()
    simulation: set[str] = set()
    unknown: set[str] = set()
    groups: list[frozenset[str]] = []
    identities: list[str] = []
    for record in ownership.concerns:
        if record.canonical_owner is not None:
            canonical.add(record.canonical_owner.node_id)
        adapters.update(item.node_id for item in record.adapters)
        projections.update(item.node_id for item in record.projections)
        legacy.update(item.node_id for item in record.legacy_owners)
        simulation.update(item.node_id for item in record.simulation_owners)
        unknown.update(item.node_id for item in record.unknown_owners)
        if record.arbitration is not None:
            identities.append(record.arbitration.content_identity)
            members = {record.arbitration.canonical_owner_node_id}
            members.update(record.arbitration.loser_ids())
            groups.append(frozenset(members))
    explained = adapters | projections | legacy | simulation
    return _OwnershipIndex(
        graph=ownership,
        canonical_ids=frozenset(canonical),
        adapter_ids=frozenset(adapters),
        projection_ids=frozenset(projections),
        legacy_ids=frozenset(legacy),
        simulation_ids=frozenset(simulation),
        unknown_ids=frozenset(unknown),
        arbitrated_groups=tuple(groups),
        arbitration_identities=tuple(sorted(set(identities))),
        explained_ids=frozenset(explained),
    )


def recognize_formal_arbitration(
    node_ids: Iterable[str],
    arbitration: FormalArbitration | Mapping[str, Any] | None = None,
    *,
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
) -> bool:
    """Return True when competing nodes are covered by formal arbitration."""

    ids = frozenset(
        _require_text(item, "node_id", error_type=DuplicateAuthorityError)
        for item in node_ids
    )
    if len(ids) < 2:
        return False
    records: list[FormalArbitration] = []
    if arbitration is not None:
        if isinstance(arbitration, FormalArbitration):
            records.append(arbitration)
        elif isinstance(arbitration, Mapping):
            records.append(FormalArbitration.from_mapping(arbitration))
        else:
            raise DuplicateAuthorityError("arbitration must be an object")
    graph = _require_ownership(ownership)
    if graph is not None:
        records.extend(graph.arbitrations)
    for record in records:
        covered = {record.canonical_owner_node_id} | set(record.loser_ids())
        if ids <= covered:
            return True
    return False


def _arbitrated(index: _OwnershipIndex, node_ids: Iterable[str]) -> bool:
    ids = frozenset(node_ids)
    if len(ids) < 2:
        return False
    return any(ids <= group for group in index.arbitrated_groups)


def _explained_relationship(
    view: _GraphView,
    left: str,
    right: str,
) -> bool:
    if left == right:
        return True
    for edge in _related_edges(view, left):
        other = edge.target if edge.source == left else edge.source
        if other != right:
            continue
        if edge.kind in _EXPLAINING_EDGE_KINDS:
            return True
        if edge.kind is EdgeKind.ADAPTS:
            return True
        if edge.kind is EdgeKind.REEXPORTS:
            return True
    return False


def _pair_explained(
    view: _GraphView,
    index: _OwnershipIndex,
    left: str,
    right: str,
) -> bool:
    if _arbitrated(index, (left, right)):
        return True
    if _explained_relationship(view, left, right):
        return True
    if left in index.explained_ids and right in index.canonical_ids:
        return True
    if right in index.explained_ids and left in index.canonical_ids:
        return True
    return False


def _nodes_of(view: _GraphView, ids: Iterable[str]) -> tuple[ArchitectureNode, ...]:
    return tuple(view.nodes_by_id[item] for item in ids if item in view.nodes_by_id)


def _edges_of(view: _GraphView, ids: Iterable[str]) -> tuple[ArchitectureEdge, ...]:
    return tuple(view.edges_by_id[item] for item in ids if item in view.edges_by_id)


def _disposition_for(
    kind: CollisionKind,
    confidence: Confidence,
    *,
    explained: bool,
    arbitrated: bool,
    unknown_owner: bool,
) -> FindingDisposition:
    if arbitrated or explained:
        return FindingDisposition.FALSE_POSITIVE
    if unknown_owner:
        return FindingDisposition.BLOCKER
    if confidence in NON_PROBATIVE_CONFIDENCE:
        return FindingDisposition.UNKNOWN
    if kind in BLOCKER_KINDS:
        return FindingDisposition.BLOCKER
    return FindingDisposition.COLLISION


def _make_finding(
    view: _GraphView,
    *,
    kind: CollisionKind,
    concern: ConcernKind,
    message: str,
    node_ids: Iterable[str],
    edge_ids: Iterable[str] = (),
    reachability_path: Sequence[str] = (),
    surfaces: Iterable[SurfaceKind] = (),
    index: _OwnershipIndex,
    explained: bool = False,
    unknown_owner: bool = False,
) -> AuthorityCollision | None:
    nodes = _nodes_of(view, node_ids)
    if not nodes:
        return None
    ids = tuple(sorted({node.node_id for node in nodes}))
    edges = _edges_of(view, edge_ids)
    path = tuple(reachability_path)
    if path:
        edges = _edges_of(view, [*[edge.edge_id for edge in edges], *path])
    confidence = _weakest_confidence(nodes, edges)
    arbitrated = _arbitrated(index, ids)
    # REEXPORTS explains other collision kinds; it is the REEXPORT_AUTHORITY finding.
    if not explained and kind is not CollisionKind.REEXPORT_AUTHORITY:
        if len(ids) == 2:
            explained = _pair_explained(view, index, ids[0], ids[1])
        elif len(ids) > 2:
            explained = all(
                _pair_explained(view, index, ids[0], other) for other in ids[1:]
            )
    disposition = _disposition_for(
        kind,
        confidence,
        explained=explained,
        arbitrated=arbitrated,
        unknown_owner=unknown_owner,
    )
    provenance = _representative_provenance(nodes, edges)
    if (
        not unknown_owner
        and disposition in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
        and (
            confidence in NON_PROBATIVE_CONFIDENCE
            or provenance.confidence in NON_PROBATIVE_CONFIDENCE
        )
    ):
        disposition = FindingDisposition.UNKNOWN
    return AuthorityCollision(
        kind=kind,
        disposition=disposition,
        concern=concern,
        message=message,
        node_ids=ids,
        edge_ids=tuple(edge.edge_id for edge in edges),
        provenance=provenance,
        reachability_path=path,
        surfaces=tuple(surfaces),
        formally_arbitrated=arbitrated,
    )


def _group_sources(
    view: _GraphView,
    edge_kinds: frozenset[EdgeKind],
    *,
    source_kinds: frozenset[NodeKind] | None = None,
    target_kinds: frozenset[NodeKind] | None = None,
) -> dict[str, list[ArchitectureEdge]]:
    grouped: dict[str, list[ArchitectureEdge]] = defaultdict(list)
    for edge in view.architecture.edges:
        if edge.kind not in edge_kinds:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        if source_kinds is not None and source.kind not in source_kinds:
            continue
        if target_kinds is not None and target.kind not in target_kinds:
            continue
        grouped[edge.target].append(edge)
    return grouped


def _independent_group_findings(
    view: _GraphView,
    index: _OwnershipIndex,
    grouped: Mapping[str, Sequence[ArchitectureEdge]],
    *,
    kind: CollisionKind,
    concern: ConcernKind,
    message: str,
    source_filter: frozenset[NodeKind] | None = None,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    for target_id, edges in grouped.items():
        sources: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            source = view.nodes_by_id[edge.source]
            if source_filter is not None and source.kind not in source_filter:
                continue
            if source.kind not in _PRODUCTION_KINDS and source.kind is not NodeKind.AUTHORITY:
                continue
            sources[edge.source].append(edge.edge_id)
        source_ids = tuple(sorted(sources))
        if len(source_ids) < 2:
            continue
        independent = [
            item
            for item in source_ids
            if not all(
                _pair_explained(view, index, item, other)
                for other in source_ids
                if other != item
            )
        ]
        if len(independent) < 2:
            # Still record the explained group as a false positive.
            edge_ids = [edge_id for item in source_ids for edge_id in sources[item]]
            finding = _make_finding(
                view,
                kind=kind,
                concern=concern,
                message=message,
                node_ids=(*source_ids, target_id),
                edge_ids=edge_ids,
                index=index,
                explained=True,
            )
            if finding is not None:
                findings.append(finding)
            continue
        edge_ids = [edge_id for item in independent for edge_id in sources[item]]
        finding = _make_finding(
            view,
            kind=kind,
            concern=concern,
            message=message,
            node_ids=(*independent, target_id),
            edge_ids=edge_ids,
            index=index,
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _detect_independent_provider(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    grouped = _group_sources(
        view,
        _PROVIDER_EDGE_KINDS,
        source_kinds=frozenset({NodeKind.AUTHORITY, NodeKind.PROVIDER}),
        target_kinds=frozenset(
            {NodeKind.PROVIDER, NodeKind.OPERATION, NodeKind.AUTHORITY, NodeKind.POLICY}
        ),
    )
    provider_targets = {
        target: edges
        for target, edges in grouped.items()
        if view.nodes_by_id[target].kind is NodeKind.PROVIDER
        or any(view.nodes_by_id[edge.source].kind is NodeKind.PROVIDER for edge in edges)
    }
    return _independent_group_findings(
        view,
        index,
        provider_targets,
        kind=CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
        concern=ConcernKind.PROVIDER_CAPABILITY,
        message="independent provider-capability or selection authorities decide the same subject",
        source_filter=frozenset({NodeKind.AUTHORITY, NodeKind.PROVIDER}),
    )


def _detect_independent_receipts(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    grouped = _group_sources(
        view,
        _RECEIPT_EDGE_KINDS,
        source_kinds=frozenset({NodeKind.RECEIPT, NodeKind.AUTHORITY, NodeKind.PROOF}),
        target_kinds=frozenset(
            {
                NodeKind.RECEIPT,
                NodeKind.OPERATION,
                NodeKind.AUTHORITY,
                NodeKind.PROOF,
                NodeKind.STATE,
            }
        ),
    )
    receipt_targets = {
        target: edges
        for target, edges in grouped.items()
        if view.nodes_by_id[target].kind in {NodeKind.RECEIPT, NodeKind.OPERATION, NodeKind.PROOF}
        or any(
            view.nodes_by_id[edge.source].kind is NodeKind.RECEIPT for edge in edges
        )
    }
    return _independent_group_findings(
        view,
        index,
        receipt_targets,
        kind=CollisionKind.INDEPENDENT_RECEIPT_PRODUCER,
        concern=ConcernKind.COMPLETION_EVIDENCE,
        message="independent receipt producers decide the same subject",
        source_filter=frozenset({NodeKind.RECEIPT, NodeKind.AUTHORITY, NodeKind.PROOF}),
    )


def _detect_competing_state(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    grouped = _group_sources(
        view,
        _STATE_EDGE_KINDS,
        source_kinds=frozenset({NodeKind.STATE, NodeKind.AUTHORITY}),
        target_kinds=frozenset({NodeKind.STATE, NodeKind.AUTHORITY}),
    )
    return _independent_group_findings(
        view,
        index,
        grouped,
        kind=CollisionKind.COMPETING_STATE_OWNER,
        concern=ConcernKind.STATE_PERSISTENCE,
        message="competing production stores persist the same mutable fact",
        source_filter=frozenset({NodeKind.STATE, NodeKind.AUTHORITY}),
    )


def _canonical_for_target(view: _GraphView, target_id: str) -> frozenset[str]:
    owners = {
        edge.source
        for edge in view.incoming.get(target_id, ())
        if edge.kind in _DECISION_EDGE_KINDS
        and view.nodes_by_id[edge.source].kind is NodeKind.AUTHORITY
    }
    return frozenset(owners)


def _has_adapts(view: _GraphView, source_id: str, target_ids: Iterable[str]) -> bool:
    wanted = set(target_ids)
    if not wanted:
        return False
    for edge in view.outgoing.get(source_id, ()):
        if edge.kind is EdgeKind.ADAPTS and edge.target in wanted:
            return True
    for edge in view.incoming.get(source_id, ()):
        if edge.kind is EdgeKind.ADAPTS and edge.source in wanted:
            return True
    return False


def _detect_compatibility_bypass(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    for node in view.architecture.nodes:
        if node.kind is not NodeKind.COMPATIBILITY:
            continue
        for edge in view.outgoing.get(node.node_id, ()):
            if edge.kind not in _BYPASS_EDGE_KINDS:
                continue
            target = view.nodes_by_id[edge.target]
            if target.kind not in _PRODUCTION_KINDS:
                continue
            if edge.kind is EdgeKind.ADAPTS:
                continue
            owners = _canonical_for_target(view, target.node_id)
            if node.node_id in index.adapter_ids and (
                owners <= index.canonical_ids or _has_adapts(view, node.node_id, owners)
            ):
                finding = _make_finding(
                    view,
                    kind=CollisionKind.COMPATIBILITY_BYPASS,
                    concern=ConcernKind.AUTHORIZATION,
                    message="compatibility path is an explicit adapter rather than a bypass",
                    node_ids=(node.node_id, target.node_id, *owners),
                    edge_ids=(edge.edge_id,),
                    index=index,
                    explained=True,
                )
                if finding is not None:
                    findings.append(finding)
                continue
            if _has_adapts(view, node.node_id, owners) or _has_adapts(
                view, node.node_id, index.canonical_ids
            ):
                finding = _make_finding(
                    view,
                    kind=CollisionKind.COMPATIBILITY_BYPASS,
                    concern=ConcernKind.AUTHORIZATION,
                    message="compatibility path is an explicit adapter rather than a bypass",
                    node_ids=(node.node_id, target.node_id, *owners),
                    edge_ids=(edge.edge_id,),
                    index=index,
                    explained=True,
                )
                if finding is not None:
                    findings.append(finding)
                continue
            finding = _make_finding(
                view,
                kind=CollisionKind.COMPATIBILITY_BYPASS,
                concern=ConcernKind.AUTHORIZATION,
                message="compatibility path bypasses canonical control or authority",
                node_ids=(node.node_id, target.node_id, *owners),
                edge_ids=(edge.edge_id,),
                index=index,
            )
            if finding is not None:
                findings.append(finding)
    return findings


def _detect_control_bypass(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    control_targets = [
        node
        for node in view.architecture.nodes
        if node.kind in {NodeKind.POLICY, NodeKind.AUTHORITY}
    ]
    for target in control_targets:
        incoming = [
            edge
            for edge in view.incoming.get(target.node_id, ())
            if edge.kind in {EdgeKind.AUTHORIZES, EdgeKind.EVALUATES_POLICY, EdgeKind.CONFIRMS, EdgeKind.EXECUTES}
        ]
        canonical = {
            edge.source
            for edge in incoming
            if view.nodes_by_id[edge.source].kind is NodeKind.AUTHORITY
        }
        if index.canonical_ids:
            canonical |= {
                node_id
                for node_id in index.canonical_ids
                if any(
                    edge.source == node_id
                    for edge in incoming
                )
            }
        for edge in incoming:
            source = view.nodes_by_id[edge.source]
            if source.kind in {
                NodeKind.COMPATIBILITY,
                NodeKind.SIMULATION,
                NodeKind.TEST,
                NodeKind.GENERATED,
            }:
                continue
            if source.node_id in canonical and source.kind is NodeKind.AUTHORITY:
                continue
            if source.node_id in index.explained_ids:
                finding = _make_finding(
                    view,
                    kind=CollisionKind.CONTROL_BYPASS,
                    concern=ConcernKind.POLICY_DECISION,
                    message="noncanonical control path is classified as adapter, projection, or quarantine",
                    node_ids=(source.node_id, target.node_id, *canonical),
                    edge_ids=(edge.edge_id,),
                    index=index,
                    explained=True,
                )
                if finding is not None:
                    findings.append(finding)
                continue
            if canonical and source.node_id not in canonical:
                if _has_adapts(view, source.node_id, canonical):
                    finding = _make_finding(
                        view,
                        kind=CollisionKind.CONTROL_BYPASS,
                        concern=ConcernKind.POLICY_DECISION,
                        message="noncanonical control path adapts the canonical owner",
                        node_ids=(source.node_id, target.node_id, *canonical),
                        edge_ids=(edge.edge_id,),
                        index=index,
                        explained=True,
                    )
                    if finding is not None:
                        findings.append(finding)
                    continue
                finding = _make_finding(
                    view,
                    kind=CollisionKind.CONTROL_BYPASS,
                    concern=ConcernKind.POLICY_DECISION,
                    message="tool dispatch or control path bypasses the canonical authority",
                    node_ids=(source.node_id, target.node_id, *canonical),
                    edge_ids=(edge.edge_id,),
                    index=index,
                )
                if finding is not None:
                    findings.append(finding)
    return findings


def _detect_simulation_flow(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    seen_sinks: set[tuple[str, str]] = set()
    for start in view.architecture.nodes:
        if start.kind is not NodeKind.SIMULATION:
            continue
        queue: deque[tuple[str, tuple[str, ...]]] = deque(((start.node_id, ()),))
        visited = {start.node_id}
        while queue:
            current, path = queue.popleft()
            for edge in view.outgoing.get(current, ()):
                if edge.kind not in _FLOW_EDGE_KINDS:
                    continue
                target = view.nodes_by_id[edge.target]
                if (
                    edge.kind is EdgeKind.FALLBACKS_TO
                    and target.kind is NodeKind.SIMULATION
                ):
                    continue
                new_path = (*path, edge.edge_id)
                if target.kind in _PRODUCTION_KINDS and target.kind is not NodeKind.SIMULATION:
                    key = (start.node_id, target.node_id)
                    if key not in seen_sinks:
                        seen_sinks.add(key)
                        explained = (
                            start.node_id in index.simulation_ids
                            and target.node_id in index.canonical_ids
                            and edge.kind is EdgeKind.FALLBACKS_TO
                            and current == start.node_id
                        )
                        # Production fallbacks_to simulation is quarantine; the reverse is a flow.
                        # Outgoing FALLBACKS_TO from simulation to production is inverted.
                        inverted = edge.kind is EdgeKind.FALLBACKS_TO and current == start.node_id
                        finding = _make_finding(
                            view,
                            kind=CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
                            concern=ConcernKind.PROVIDER_SELECTION,
                            message=(
                                "simulation path reaches a production authority, store, or receipt"
                            ),
                            node_ids=(start.node_id, target.node_id),
                            edge_ids=new_path,
                            reachability_path=new_path,
                            index=index,
                            explained=explained and not inverted,
                        )
                        if finding is not None:
                            findings.append(finding)
                if target.node_id not in visited:
                    visited.add(target.node_id)
                    queue.append((target.node_id, new_path))
        # Quarantined production --FALLBACKS_TO--> simulation is a false positive.
        for edge in view.incoming.get(start.node_id, ()):
            if edge.kind is not EdgeKind.FALLBACKS_TO:
                continue
            source = view.nodes_by_id[edge.source]
            if source.kind not in _PRODUCTION_KINDS:
                continue
            finding = _make_finding(
                view,
                kind=CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
                concern=ConcernKind.PROVIDER_SELECTION,
                message="production fallbacks to a quarantined simulation path",
                node_ids=(source.node_id, start.node_id),
                edge_ids=(edge.edge_id,),
                reachability_path=(edge.edge_id,),
                index=index,
                explained=True,
            )
            if finding is not None:
                findings.append(finding)
    return findings


def _operation_for(view: _GraphView, node_id: str) -> set[str]:
    operations: set[str] = set()
    node = view.nodes_by_id[node_id]
    if node.kind is NodeKind.OPERATION:
        operations.add(node_id)
    for edge in view.outgoing.get(node_id, ()):
        if edge.kind in _SURFACE_EDGE_KINDS:
            target = view.nodes_by_id[edge.target]
            if target.kind is NodeKind.OPERATION:
                operations.add(target.node_id)
    for edge in view.incoming.get(node_id, ()):
        if edge.kind in _SURFACE_EDGE_KINDS:
            source = view.nodes_by_id[edge.source]
            if source.kind is NodeKind.OPERATION:
                operations.add(source.node_id)
    return operations


def _authority_targets(view: _GraphView, node_id: str) -> set[str]:
    targets: set[str] = set()
    for edge in view.outgoing.get(node_id, ()):
        if edge.kind in {EdgeKind.AUTHORIZES, EdgeKind.IMPLEMENTS, EdgeKind.EXECUTES}:
            if view.nodes_by_id[edge.target].kind in {
                NodeKind.AUTHORITY,
                NodeKind.POLICY,
                NodeKind.OPERATION,
            }:
                targets.add(edge.target)
    for edge in view.incoming.get(node_id, ()):
        if edge.kind is EdgeKind.AUTHORIZES and view.nodes_by_id[edge.source].kind is NodeKind.AUTHORITY:
            targets.add(edge.source)
    return targets


def _detect_surface_divergence(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    surfaces: dict[str, tuple[ArchitectureNode, SurfaceKind]] = {}
    for node in view.architecture.nodes:
        if node.kind is NodeKind.ENTRYPOINT:
            surfaces[node.node_id] = (node, _surface_of(node))
            continue
        classified = _surface_of(node)
        if classified in {SurfaceKind.CLI, SurfaceKind.MCP} and node.kind in {
            NodeKind.MODULE,
            NodeKind.SYMBOL,
            NodeKind.INTERFACE,
        }:
            surfaces[node.node_id] = (node, classified)
    by_operation: dict[str, list[tuple[ArchitectureNode, SurfaceKind]]] = defaultdict(list)
    for node, surface in surfaces.values():
        operations = _operation_for(view, node.node_id)
        if not operations:
            continue
        for operation in operations:
            by_operation[operation].append((node, surface))
    for operation_id, members in by_operation.items():
        present = {surface for _node, surface in members}
        python = [item for item in members if item[1] is SurfaceKind.PYTHON]
        cli = [item for item in members if item[1] is SurfaceKind.CLI]
        mcp = [item for item in members if item[1] is SurfaceKind.MCP]
        comparable = [group for group in (python, cli, mcp) if group]
        if len(comparable) < 2 and present <= {SurfaceKind.PYTHON, SurfaceKind.UNKNOWN}:
            continue
        authorities = [
            frozenset(_authority_targets(view, node.node_id)) for node, _surface in members
        ]
        node_ids = tuple(node.node_id for node, _surface in members)
        edge_ids = [
            edge.edge_id
            for node, _surface in members
            for edge in _related_edges(view, node.node_id)
            if edge.kind in _SURFACE_EDGE_KINDS
            and operation_id in {edge.source, edge.target}
        ]
        diverged = False
        if len(comparable) >= 2:
            unique_authorities = {item for item in authorities if item}
            if len(unique_authorities) > 1:
                diverged = True
            if any(not item for item in authorities) and any(authorities):
                diverged = True
        if SurfaceKind.PYTHON in present and (
            SurfaceKind.CLI in present or SurfaceKind.MCP in present
        ):
            if not diverged:
                # Shared authority across surfaces is a projection, not a collision.
                finding = _make_finding(
                    view,
                    kind=CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
                    concern=ConcernKind.OPERATION_IDENTITY,
                    message="Python, CLI, and MCP projections share one canonical operation authority",
                    node_ids=(*node_ids, operation_id),
                    edge_ids=edge_ids,
                    surfaces=present,
                    index=index,
                    explained=True,
                )
                if finding is not None:
                    findings.append(finding)
                continue
        if diverged or (
            SurfaceKind.CLI in present
            and SurfaceKind.MCP in present
            and SurfaceKind.PYTHON not in present
        ):
            finding = _make_finding(
                view,
                kind=CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
                concern=ConcernKind.OPERATION_IDENTITY,
                message="Python, CLI, and MCP surfaces diverge on operation authority or schema",
                node_ids=(*node_ids, operation_id),
                edge_ids=edge_ids,
                surfaces=present,
                index=index,
            )
            if finding is not None:
                findings.append(finding)
        # SHADOWS/DUPLICATES between different surfaces without shared authority.
        for left_node, left_surface in members:
            for right_node, right_surface in members:
                if left_node.node_id >= right_node.node_id:
                    continue
                if left_surface is right_surface:
                    continue
                for edge in _related_edges(view, left_node.node_id):
                    other = edge.target if edge.source == left_node.node_id else edge.source
                    if other != right_node.node_id:
                        continue
                    if edge.kind not in {EdgeKind.SHADOWS, EdgeKind.DUPLICATES}:
                        continue
                    if _pair_explained(view, index, left_node.node_id, right_node.node_id):
                        continue
                    finding = _make_finding(
                        view,
                        kind=CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
                        concern=ConcernKind.OPERATION_IDENTITY,
                        message="entrypoint surfaces shadow or duplicate without a shared authority",
                        node_ids=(left_node.node_id, right_node.node_id, operation_id),
                        edge_ids=(edge.edge_id,),
                        surfaces=(left_surface, right_surface),
                        index=index,
                    )
                    if finding is not None:
                        findings.append(finding)
    return findings


def _detect_reexport_authority(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    for edge in view.architecture.edges:
        if edge.kind is not EdgeKind.REEXPORTS:
            continue
        source = view.nodes_by_id[edge.source]
        target = view.nodes_by_id[edge.target]
        if source.kind is not NodeKind.AUTHORITY and target.kind is not NodeKind.AUTHORITY:
            continue
        authority = source if source.kind is NodeKind.AUTHORITY else target
        explained = authority.node_id in index.adapter_ids or authority.node_id in index.explained_ids
        other_kinds = {
            item.kind
            for item in view.outgoing.get(authority.node_id, ())
            if item.kind is not EdgeKind.REEXPORTS
        }
        claimed_independent = bool(other_kinds & _DECISION_EDGE_KINDS) or not explained
        finding = _make_finding(
            view,
            kind=CollisionKind.REEXPORT_AUTHORITY,
            concern=ConcernKind.OPERATION_IDENTITY,
            message="re-export is not authority",
            node_ids=(source.node_id, target.node_id),
            edge_ids=(edge.edge_id,),
            index=index,
            explained=explained and not claimed_independent,
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _superseded_targets(view: _GraphView) -> dict[str, set[str]]:
    superseded: dict[str, set[str]] = defaultdict(set)
    for edge in view.architecture.edges:
        if edge.kind in {EdgeKind.SUPERSEDES, EdgeKind.DEPRECATES}:
            superseded[edge.target].add(edge.source)
        if edge.kind is EdgeKind.SHADOWS:
            superseded[edge.target].add(edge.source)
    return superseded


def _detect_obsolete_tests(
    view: _GraphView, index: _OwnershipIndex
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    superseded = _superseded_targets(view)
    for node in view.architecture.nodes:
        if node.kind is not NodeKind.TEST:
            continue
        tested = [
            edge
            for edge in view.outgoing.get(node.node_id, ())
            if edge.kind is EdgeKind.TESTS
        ]
        tested_ids = {edge.target for edge in tested}
        canonical_tested = tested_ids & index.canonical_ids
        for edge in tested:
            target = view.nodes_by_id[edge.target]
            obsolete = (
                target.node_id in superseded
                or target.kind in {NodeKind.COMPATIBILITY, NodeKind.SIMULATION}
                or target.node_id in index.legacy_ids
                or target.node_id in index.simulation_ids
            )
            if not obsolete:
                continue
            replacements = superseded.get(target.node_id, set()) | set(index.canonical_ids)
            tests_canonical = bool(tested_ids & replacements) or bool(canonical_tested)
            finding = _make_finding(
                view,
                kind=CollisionKind.OBSOLETE_AUTHORITY_TEST,
                concern=ConcernKind.TEST_EVIDENCE,
                message=(
                    "test covers a superseded, compatibility, or simulation authority"
                    if not tests_canonical
                    else "test covers both canonical and superseded authorities"
                ),
                node_ids=(node.node_id, target.node_id, *sorted(replacements)),
                edge_ids=(edge.edge_id,),
                index=index,
                explained=tests_canonical,
            )
            if finding is not None:
                findings.append(finding)
    return findings


def _detect_ownership_findings(
    view: _GraphView,
    index: _OwnershipIndex,
) -> list[AuthorityCollision]:
    findings: list[AuthorityCollision] = []
    ownership = index.graph
    if ownership is None:
        return findings
    for record in ownership.concerns:
        if record.blocker is not None:
            mapped = _OWNERSHIP_BLOCKER_TO_COLLISION.get(record.blocker.kind)
            if mapped is None:
                if record.blocker.kind is OwnershipBlockerKind.UNKNOWN_OWNER:
                    mapped = CollisionKind.UNKNOWN_PRODUCTION_OWNER
                else:
                    mapped = CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES
            node_ids = record.blocker.node_ids
            if not node_ids and record.canonical_owner is not None:
                node_ids = (record.canonical_owner.node_id,)
            if not node_ids and record.unknown_owners:
                node_ids = tuple(item.node_id for item in record.unknown_owners)
            if not node_ids:
                # Bind the blocker to any architecture node so the finding remains
                # source-addressable; unknown ownership without a node still blocks.
                if view.architecture.nodes:
                    node_ids = (view.architecture.nodes[0].node_id,)
                else:
                    continue
            unknown_owner = mapped is CollisionKind.UNKNOWN_PRODUCTION_OWNER
            finding = _make_finding(
                view,
                kind=mapped,
                concern=record.concern,
                message=record.blocker.message,
                node_ids=node_ids,
                edge_ids=record.blocker.edge_ids,
                index=index,
                unknown_owner=unknown_owner,
            )
            if finding is not None:
                findings.append(finding)
            continue
        if record.arbitration is not None:
            members = (
                record.arbitration.canonical_owner_node_id,
                *sorted(record.arbitration.loser_ids()),
            )
            finding = _make_finding(
                view,
                kind=CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES,
                concern=record.concern,
                message="formal arbitration classifies competing production authorities",
                node_ids=members,
                edge_ids=record.arbitration.evidence_edge_ids,
                index=index,
                explained=True,
            )
            if finding is not None:
                findings.append(finding)
        if record.canonical_owner is None:
            continue
        for owner, kind, concern, message in (
            (
                record.adapters,
                CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY
                if record.concern
                in {ConcernKind.PROVIDER_CAPABILITY, ConcernKind.PROVIDER_SELECTION}
                else CollisionKind.CONTROL_BYPASS,
                record.concern,
                "adapter is an explicit non-canonical owner, not a competing authority",
            ),
            (
                record.projections,
                CollisionKind.PYTHON_CLI_MCP_DIVERGENCE,
                record.concern,
                "projection is generated from the canonical owner",
            ),
            (
                record.legacy_owners,
                CollisionKind.OBSOLETE_AUTHORITY_TEST
                if record.concern is ConcernKind.TEST_EVIDENCE
                else CollisionKind.CONTROL_BYPASS,
                record.concern,
                "legacy path is superseded or deprecated by the canonical owner",
            ),
            (
                record.simulation_owners,
                CollisionKind.SIMULATION_TO_PRODUCTION_FLOW,
                record.concern,
                "simulation path is quarantined and does not own production",
            ),
        ):
            for item in owner:
                finding = _make_finding(
                    view,
                    kind=kind,
                    concern=concern,
                    message=message,
                    node_ids=(record.canonical_owner.node_id, item.node_id),
                    edge_ids=item.evidence_edge_ids,
                    index=index,
                    explained=True,
                )
                if finding is not None:
                    findings.append(finding)
    return findings


def _dedupe_findings(
    findings: Sequence[AuthorityCollision],
) -> tuple[AuthorityCollision, ...]:
    by_key: dict[tuple[Any, ...], AuthorityCollision] = {}
    rank = {
        FindingDisposition.BLOCKER: 0,
        FindingDisposition.COLLISION: 1,
        FindingDisposition.UNKNOWN: 2,
        FindingDisposition.FALSE_POSITIVE: 3,
    }
    for item in findings:
        key = (item.kind, item.concern, item.node_ids, item.edge_ids, item.reachability_path)
        current = by_key.get(key)
        if current is None or rank[item.disposition] < rank[current.disposition]:
            by_key[key] = item
    return tuple(
        sorted(
            by_key.values(),
            key=lambda item: (
                item.kind.value,
                item.concern.value,
                item.node_ids,
                item.edge_ids,
                item.content_identity,
            ),
        )
    )


def detect_duplicate_authorities(
    architecture: ArchitectureIR | Mapping[str, Any],
    ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
    *,
    claims: Sequence[Any] | None = None,
    arbitrations: Sequence[Any] | None = None,
) -> DuplicateAuthorityReport:
    """Detect competing production authorities without remediating them."""

    graph = _require_architecture_ir(architecture)
    view = _build_view(graph)
    resolved = _require_ownership(ownership)
    if resolved is None and claims is not None:
        resolved = resolve_authority_ownership(graph, claims, arbitrations)
    index = _index_ownership(resolved)
    findings: list[AuthorityCollision] = []
    findings.extend(_detect_ownership_findings(view, index))
    findings.extend(_detect_independent_provider(view, index))
    findings.extend(_detect_independent_receipts(view, index))
    findings.extend(_detect_competing_state(view, index))
    findings.extend(_detect_compatibility_bypass(view, index))
    findings.extend(_detect_control_bypass(view, index))
    findings.extend(_detect_simulation_flow(view, index))
    findings.extend(_detect_surface_divergence(view, index))
    findings.extend(_detect_reexport_authority(view, index))
    findings.extend(_detect_obsolete_tests(view, index))
    return DuplicateAuthorityReport(
        architecture_ir_identity=graph.content_identity,
        repository_tree=graph.repository_tree,
        freshness=graph.freshness,
        findings=_dedupe_findings(findings),
        ownership_graph_identity="" if resolved is None else resolved.content_identity,
        recognized_arbitrations=index.arbitration_identities,
    )


build_duplicate_authority_report = detect_duplicate_authorities


class DuplicateAuthorityDetector:
    """Read-only detector for competing production authorities."""

    can_authorize_changes = DETECTOR_CAN_AUTHORIZE_CHANGES
    can_select_owner = DETECTOR_CAN_SELECT_OWNER
    can_remediate = DETECTOR_CAN_REMEDIATE
    extractor_identity = EXTRACTOR_IDENTITY
    task_id = TASK_ID
    effect_class = EFFECT_CLASS

    def detect(
        self,
        architecture: ArchitectureIR | Mapping[str, Any],
        ownership: AuthorityOwnershipGraph | Mapping[str, Any] | None = None,
        *,
        claims: Sequence[Any] | None = None,
        arbitrations: Sequence[Any] | None = None,
    ) -> DuplicateAuthorityReport:
        return detect_duplicate_authorities(
            architecture,
            ownership,
            claims=claims,
            arbitrations=arbitrations,
        )

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("authorize changes")

    def select_owner(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_owner_selection("select")

    def remediate(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("remediate")

    def consolidate(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_remediation("consolidate authorities")

    def promote_heuristic(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_heuristic_promotion("promotion")


BypassFinding = AuthorityCollision
SurfaceDivergenceFinding = AuthorityCollision


__all__ = [
    "BLOCKER_KINDS",
    "BYPASS_KINDS",
    "BypassFinding",
    "CLOSED_COLLISION_KINDS",
    "CLOSED_FINDING_DISPOSITIONS",
    "CLOSED_SURFACES",
    "COLLISION_SCHEMA",
    "COLLISION_VERSION",
    "CONTENT_IDENTITY_IS_NOT_AUTHORITY",
    "DEFAULT_FRESHNESS",
    "DETECTOR_CAN_AUTHORIZE_CHANGES",
    "DETECTOR_CAN_REMEDIATE",
    "DETECTOR_CAN_SELECT_OWNER",
    "DUPLICATE_AUTHORITY_EVIDENCE",
    "DUPLICATE_AUTHORITY_SCHEMA",
    "DUPLICATE_AUTHORITY_VERSION",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "HEURISTIC_CRITICAL_PROMOTION_PROHIBITED",
    "INITIAL_CONCERNS",
    "REEXPORT_IS_NOT_AUTHORITY",
    "REQUIRED_DETECTIONS",
    "REQUIRED_SURFACES",
    "SILENT_ARBITRATION_PROHIBITED",
    "SurfaceDivergenceFinding",
    "TASK_ID",
    "UNKNOWN_PRODUCTION_OWNER_BLOCKS",
    "AuthorityCollision",
    "CollisionKind",
    "DuplicateAuthorityAuthorityError",
    "DuplicateAuthorityDetector",
    "DuplicateAuthorityError",
    "DuplicateAuthorityReport",
    "FindingDisposition",
    "SurfaceKind",
    "build_duplicate_authority_report",
    "detect_duplicate_authorities",
    "lookup_owner_by_content_identity",
    "recognize_formal_arbitration",
    "refuse_heuristic_promotion",
    "refuse_owner_selection",
    "refuse_remediation",
]
