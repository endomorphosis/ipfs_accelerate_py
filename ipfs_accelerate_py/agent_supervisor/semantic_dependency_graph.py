"""Canonical cross-domain dependency graph for proof-directed decisions.

The graph is a verification boundary, not an inference engine.  Every node and
edge is bound to one decision root and to the exact source root, provenance,
trust state, authority class, and producer version from which it was derived.
Untrusted/model-produced annotations may be retained, but they can never enter
an authority-bearing mandatory closure.

Edges point *from a subject to a dependency*.  For example, a decision
``requires`` an obligation and an obligation is ``proven_by`` a proof.  This
makes the bounded forward closure the complete set of mandatory inputs for a
decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SEMANTIC_DEPENDENCY_GRAPH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-dependency-graph@1"
)
SEMANTIC_DEPENDENCY_NODE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-dependency-node@1"
)
SEMANTIC_DEPENDENCY_EDGE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/semantic-dependency-edge@1"
)
MANDATORY_CLOSURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mandatory-dependency-closure@1"
)

DEFAULT_MAX_GRAPH_NODES = 100_000
DEFAULT_MAX_GRAPH_EDGES = 250_000
DEFAULT_MAX_CLOSURE_NODES = 16_384
DEFAULT_MAX_CLOSURE_EDGES = 65_536
DEFAULT_MAX_CLOSURE_DEPTH = 256
DEFAULT_MAX_ANNOTATIONS = 4_096


class SemanticGraphError(ValueError):
    """A graph record is malformed or violates the authority boundary."""


class SemanticGraphBoundsError(SemanticGraphError):
    """A graph or mandatory closure exceeded a hard deterministic bound."""


class CrossRootEdgeError(SemanticGraphError):
    """An edge joins records from different decision roots."""


class UnsafeDependencyCycleError(SemanticGraphError):
    """Authority-bearing dependency edges contain an unsafe cycle."""


class SemanticNodeKind(str, Enum):
    DECISION = "decision"
    PLAN = "plan"
    ACTION = "action"
    EFFECT = "effect"
    TOOL = "tool"
    RESOURCE = "resource"

    INTENT_GOAL = "intent_goal"
    INTENT_DECLARATION = "intent_declaration"
    INTENT_ACTION = "intent_action"
    INTENT_CONTROL_FLOW = "intent_control_flow"
    INTENT_PRECONDITION = "intent_precondition"
    INTENT_GUARD = "intent_guard"
    INTENT_INVARIANT = "intent_invariant"
    INTENT_EFFECT = "intent_effect"
    INTENT_POSTCONDITION = "intent_postcondition"
    INTENT_ASSUMPTION = "intent_assumption"
    INTENT_FAILURE = "intent_failure"
    INTENT_RETRY = "intent_retry"
    INTENT_VERIFICATION = "intent_verification"
    INTENT_FORMAL_VIEW = "intent_formal_view"
    INTENT_CLAIM = "intent_claim"
    INTENT_OBLIGATION = "intent_obligation"
    INTENT_RESULT_AUTHORITY = "intent_result_authority"

    LEGAL_OBLIGATION = "legal_obligation"
    LEGAL_DECLARATION = "legal_declaration"
    LEGAL_PROHIBITION = "legal_prohibition"
    LEGAL_PERMISSION = "legal_permission"
    LEGAL_POWER = "legal_power"
    LEGAL_EXCEPTION = "legal_exception"
    LEGAL_FORMAL_VIEW = "legal_formal_view"
    LEGAL_CLAIM = "legal_claim"
    LEGAL_ASSUMPTION = "legal_assumption"
    LEGAL_PROOF_OBLIGATION = "legal_proof_obligation"
    LEGAL_RESULT_AUTHORITY = "legal_result_authority"

    SECURITY_PRINCIPAL = "security_principal"
    SECURITY_DECLARATION = "security_declaration"
    SECURITY_ASSET = "security_asset"
    SECURITY_RESOURCE = "security_resource"
    SECURITY_ZONE = "security_zone"
    SECURITY_CHANNEL = "security_channel"
    SECURITY_POLICY = "security_policy"
    SECURITY_STATE_MACHINE = "security_state_machine"
    SECURITY_THREAT_ASSUMPTION = "security_threat_assumption"
    SECURITY_FORMAL_VIEW = "security_formal_view"
    SECURITY_CLAIM = "security_claim"
    SECURITY_OBLIGATION = "security_obligation"
    SECURITY_RESULT_AUTHORITY = "security_result_authority"

    WORKTREE = "worktree"
    REPOSITORY_TREE = "repository_tree"
    FILE = "file"
    AST = "ast"
    SYMBOL = "symbol"
    INTERFACE = "interface"
    CALL = "call"
    DATA_FLOW = "data_flow"
    PROGRAM = "program"
    ENVIRONMENT = "environment"
    TOOLCHAIN = "toolchain"

    ASSUMPTION = "assumption"
    PREMISE = "premise"
    OBLIGATION = "obligation"
    PROOF = "proof"
    MONITOR = "monitor"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    MERGE_EVIDENCE = "merge_evidence"
    ANNOTATION = "annotation"


class SemanticEdgeKind(str, Enum):
    REQUIRES = "requires"
    CONSTRAINED_BY = "constrained_by"
    APPLIES_TO = "applies_to"
    EXCEPTION_TO = "exception_to"
    CONFLICTS_WITH = "conflicts_with"
    AUTHORIZES = "authorizes"
    DENIES = "denies"
    IMPLEMENTS = "implements"
    AFFECTS = "affects"
    DEPENDS_ON = "depends_on"
    PROVEN_BY = "proven_by"
    MONITORED_BY = "monitored_by"
    INVALIDATES = "invalidates"
    SOURCED_FROM = "sourced_from"


class SemanticProvenance(str, Enum):
    SOURCE = "source"
    DECISION = "decision"
    PLANNER = "planner"
    INTENT_IR = "intent_ir"
    LEGAL_IR = "legal_ir"
    SECURITY_IR = "security_ir"
    WORKTREE = "worktree"
    AST = "ast"
    PROGRAM = "program"
    TOOL = "tool"
    PROOF = "proof"
    MONITOR = "monitor"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    MERGE = "merge"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    GRAPHRAG = "graphrag"

    @property
    def trusted_channel(self) -> bool:
        return self not in {
            SemanticProvenance.MODEL,
            SemanticProvenance.RETRIEVAL,
            SemanticProvenance.GRAPHRAG,
        }


class SemanticTrust(str, Enum):
    TRUSTED = "trusted"
    VERIFIED = "verified"
    REVIEWED = "reviewed"
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"
    QUARANTINED = "quarantined"

    @property
    def accepted(self) -> bool:
        return self in {
            SemanticTrust.TRUSTED,
            SemanticTrust.VERIFIED,
            SemanticTrust.REVIEWED,
        }


class SemanticAuthority(str, Enum):
    AUTHORITATIVE = "authoritative"
    VERIFIED_INPUT = "verified_input"
    CONSTRAINT_INPUT = "constraint_input"
    POLICY_INPUT = "policy_input"
    DESCRIPTIVE_INPUT = "descriptive_input"
    PROPOSAL_ONLY = "proposal_only"
    CONTEXT_ONLY = "context_only"
    UNTRUSTED = "untrusted"
    NONE = "none"

    @property
    def authority_bearing(self) -> bool:
        return self in {
            SemanticAuthority.AUTHORITATIVE,
            SemanticAuthority.VERIFIED_INPUT,
            SemanticAuthority.CONSTRAINT_INPUT,
            SemanticAuthority.POLICY_INPUT,
            # Trusted intent descriptions and explicit assumptions do not
            # grant execution, but may still be mandatory decision premises.
            SemanticAuthority.DESCRIPTIVE_INPUT,
            SemanticAuthority.CONTEXT_ONLY,
        }


_UNSAFE_CYCLE_EDGE_KINDS = frozenset(
    {
        SemanticEdgeKind.REQUIRES,
        SemanticEdgeKind.CONSTRAINED_BY,
        SemanticEdgeKind.EXCEPTION_TO,
        SemanticEdgeKind.IMPLEMENTS,
        SemanticEdgeKind.DEPENDS_ON,
        SemanticEdgeKind.PROVEN_BY,
        SemanticEdgeKind.MONITORED_BY,
        SemanticEdgeKind.INVALIDATES,
        SemanticEdgeKind.SOURCED_FROM,
    }
)

_AUTHORIZATION_PROVENANCE = frozenset(
    {SemanticProvenance.SECURITY_IR, SemanticProvenance.AUTHORIZATION}
)


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw))
    except (TypeError, ValueError) as exc:
        raise SemanticGraphError(f"invalid {name}: {value!r}") from exc


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise SemanticGraphError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise SemanticGraphError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise SemanticGraphError(f"{name} is required")
    if len(value.encode("utf-8")) > 8_192:
        raise SemanticGraphBoundsError(f"{name} is oversized")
    return value


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise SemanticGraphBoundsError("semantic record exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise SemanticGraphError("floating values are not canonical graph data")
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise SemanticGraphBoundsError("semantic record mapping is invalid")
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 16_384:
            raise SemanticGraphBoundsError("semantic record sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise SemanticGraphError(
        f"unsupported semantic record value: {type(value).__name__}"
    )


def canonical_semantic_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(namespace: str, value: Any) -> str:
    digest = hashlib.sha256(
        canonical_semantic_json(value).encode("utf-8")
    ).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise SemanticGraphError(f"{name} must be a mapping or typed record")
        value = to_dict()
    normalized = _plain(value)
    if not isinstance(normalized, dict):
        raise SemanticGraphError(f"{name} must normalize to a mapping")
    return MappingProxyType(normalized)


def _binding_payload(value: Any) -> dict[str, Any]:
    return {
        "root_id": value.root_id,
        "source_root_id": value.source_root_id,
        "provenance": value.provenance.value,
        "provenance_id": value.provenance_id,
        "trust": value.trust.value,
        "authority": value.authority.value,
        "version": value.version,
    }


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    kind: SemanticNodeKind
    root_id: str
    provenance: SemanticProvenance
    trust: SemanticTrust
    authority: SemanticAuthority
    version: str
    source_root_id: str = ""
    provenance_id: str = ""
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("node_id", "root_id", "version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"node {name}")
            )
        object.__setattr__(self, "kind", _enum(self.kind, SemanticNodeKind, "node kind"))
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, SemanticProvenance, "node provenance"),
        )
        object.__setattr__(
            self, "trust", _enum(self.trust, SemanticTrust, "node trust")
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, SemanticAuthority, "node authority"),
        )
        source_root = self.source_root_id or self.root_id
        provenance_id = self.provenance_id or self.node_id
        object.__setattr__(
            self, "source_root_id", _text(source_root, "node source_root_id")
        )
        object.__setattr__(
            self, "provenance_id", _text(provenance_id, "node provenance_id")
        )
        object.__setattr__(self, "record", _mapping(self.record, "node record"))
        if (
            not self.provenance.trusted_channel
            or not self.trust.accepted
        ) and self.authority.authority_bearing:
            raise SemanticGraphError(
                "untrusted or model provenance cannot create authoritative nodes"
            )

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.trust.accepted
            and self.authority.authority_bearing
        )

    @property
    def content_id(self) -> str:
        return _identity("semantic-node", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_DEPENDENCY_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind.value,
            **_binding_payload(self),
            "record": _plain(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "content_id": self.content_id,
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticNode":
        schema = str(payload.get("schema") or SEMANTIC_DEPENDENCY_NODE_SCHEMA)
        if schema != SEMANTIC_DEPENDENCY_NODE_SCHEMA:
            raise SemanticGraphError(f"unsupported semantic node schema: {schema}")
        node = cls(
            node_id=str(payload.get("node_id") or ""),
            kind=payload.get("kind", ""),
            root_id=str(payload.get("root_id") or ""),
            source_root_id=str(payload.get("source_root_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            trust=payload.get("trust", ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("content_id") or "")
        if claimed and claimed != node.content_id:
            raise SemanticGraphError("semantic node content identity mismatch")
        if "authoritative" in payload and bool(payload["authoritative"]) != node.authoritative:
            raise SemanticGraphError("semantic node authority claim is forged")
        return node


@dataclass(frozen=True)
class SemanticEdge:
    source: str
    target: str
    kind: SemanticEdgeKind
    root_id: str
    provenance: SemanticProvenance
    trust: SemanticTrust
    authority: SemanticAuthority
    version: str
    provenance_id: str
    source_root_id: str = ""
    mandatory: bool = True
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source",
            "target",
            "root_id",
            "version",
            "provenance_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), f"edge {name}")
            )
        if self.source == self.target:
            raise UnsafeDependencyCycleError("self-referential semantic edge")
        object.__setattr__(
            self,
            "source_root_id",
            _text(
                self.source_root_id or self.root_id,
                "edge source_root_id",
            ),
        )
        object.__setattr__(self, "kind", _enum(self.kind, SemanticEdgeKind, "edge kind"))
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, SemanticProvenance, "edge provenance"),
        )
        object.__setattr__(
            self, "trust", _enum(self.trust, SemanticTrust, "edge trust")
        )
        object.__setattr__(
            self,
            "authority",
            _enum(self.authority, SemanticAuthority, "edge authority"),
        )
        if not isinstance(self.mandatory, bool):
            raise SemanticGraphError("edge mandatory must be a boolean")
        object.__setattr__(self, "record", _mapping(self.record, "edge record"))
        if not self.provenance.trusted_channel or not self.trust.accepted:
            if self.authority.authority_bearing:
                raise SemanticGraphError(
                    "untrusted or model provenance cannot create authoritative edges"
                )
            # Proposal/context annotations are retained, never mandatory.
            object.__setattr__(self, "mandatory", False)
        if self.authority in {
            SemanticAuthority.PROPOSAL_ONLY,
            SemanticAuthority.UNTRUSTED,
            SemanticAuthority.NONE,
        }:
            object.__setattr__(self, "mandatory", False)
        if (
            self.kind in {SemanticEdgeKind.AUTHORIZES, SemanticEdgeKind.DENIES}
            and self.provenance not in _AUTHORIZATION_PROVENANCE
        ):
            raise SemanticGraphError(
                f"{self.kind.value} edges require SecurityIR or authorization provenance"
            )
        if (
            self.kind is SemanticEdgeKind.PROVEN_BY
            and self.provenance is not SemanticProvenance.PROOF
        ):
            raise SemanticGraphError("proven_by edges require proof provenance")
        if (
            self.kind is SemanticEdgeKind.MONITORED_BY
            and self.provenance is not SemanticProvenance.MONITOR
        ):
            raise SemanticGraphError("monitored_by edges require monitor provenance")

    @property
    def authoritative(self) -> bool:
        return (
            self.provenance.trusted_channel
            and self.trust.accepted
            and self.authority.authority_bearing
        )

    @property
    def edge_id(self) -> str:
        return _identity("semantic-edge", self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_DEPENDENCY_EDGE_SCHEMA,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "root_id": self.root_id,
            "source_root_id": self.source_root_id,
            "provenance": self.provenance.value,
            "provenance_id": self.provenance_id,
            "trust": self.trust.value,
            "authority": self.authority.value,
            "version": self.version,
            "mandatory": self.mandatory,
            "record": _plain(self.record),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "edge_id": self.edge_id,
            "authoritative": self.authoritative,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticEdge":
        schema = str(payload.get("schema") or SEMANTIC_DEPENDENCY_EDGE_SCHEMA)
        if schema != SEMANTIC_DEPENDENCY_EDGE_SCHEMA:
            raise SemanticGraphError(f"unsupported semantic edge schema: {schema}")
        edge = cls(
            source=str(payload.get("source") or payload.get("source_node_id") or ""),
            target=str(payload.get("target") or payload.get("target_node_id") or ""),
            kind=payload.get("kind", payload.get("edge_kind", "")),
            root_id=str(payload.get("root_id") or ""),
            provenance=payload.get("provenance", ""),
            provenance_id=str(payload.get("provenance_id") or ""),
            source_root_id=str(payload.get("source_root_id") or ""),
            trust=payload.get("trust", ""),
            authority=payload.get("authority", ""),
            version=str(payload.get("version") or ""),
            mandatory=payload.get("mandatory", True),
            record=payload.get("record") or {},
        )
        claimed = str(payload.get("edge_id") or "")
        if claimed and claimed != edge.edge_id:
            raise SemanticGraphError("semantic edge identity mismatch")
        if "authoritative" in payload and bool(payload["authoritative"]) != edge.authoritative:
            raise SemanticGraphError("semantic edge authority claim is forged")
        return edge


@dataclass(frozen=True)
class ClosureBounds:
    max_nodes: int = DEFAULT_MAX_CLOSURE_NODES
    max_edges: int = DEFAULT_MAX_CLOSURE_EDGES
    max_depth: int = DEFAULT_MAX_CLOSURE_DEPTH
    max_annotations: int = DEFAULT_MAX_ANNOTATIONS

    def __post_init__(self) -> None:
        maxima = {
            "max_nodes": (self.max_nodes, DEFAULT_MAX_GRAPH_NODES),
            "max_edges": (self.max_edges, DEFAULT_MAX_GRAPH_EDGES),
            "max_depth": (self.max_depth, 4_096),
            "max_annotations": (self.max_annotations, DEFAULT_MAX_GRAPH_NODES),
        }
        for name, (value, maximum) in maxima.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise SemanticGraphBoundsError(
                    f"{name} must be an integer from 1 through {maximum}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_depth": self.max_depth,
            "max_annotations": self.max_annotations,
        }


@dataclass(frozen=True)
class MandatoryClosure:
    root_id: str
    decision_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    paths: Mapping[str, tuple[str, ...]]
    annotation_node_ids: tuple[str, ...] = ()
    annotation_edge_ids: tuple[str, ...] = ()
    bounds: ClosureBounds = field(default_factory=ClosureBounds)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", _text(self.root_id, "closure root_id"))
        object.__setattr__(
            self, "decision_id", _text(self.decision_id, "closure decision_id")
        )
        object.__setattr__(self, "node_ids", tuple(sorted(set(self.node_ids))))
        object.__setattr__(self, "edge_ids", tuple(sorted(set(self.edge_ids))))
        object.__setattr__(
            self,
            "annotation_node_ids",
            tuple(sorted(set(self.annotation_node_ids))),
        )
        object.__setattr__(
            self,
            "annotation_edge_ids",
            tuple(sorted(set(self.annotation_edge_ids))),
        )
        canonical_paths = {
            str(key): tuple(str(item) for item in value)
            for key, value in sorted(self.paths.items())
        }
        object.__setattr__(self, "paths", MappingProxyType(canonical_paths))
        if self.decision_id not in self.node_ids:
            raise SemanticGraphError("closure does not contain its decision")
        if set(canonical_paths) != set(self.node_ids):
            raise SemanticGraphError("closure paths do not cover exactly its nodes")
        for node_id, path in canonical_paths.items():
            if (
                not path
                or path[0] != self.decision_id
                or path[-1] != node_id
                or len(path) != len(set(path))
            ):
                raise SemanticGraphError(f"invalid closure path for {node_id!r}")
        if set(self.annotation_node_ids).intersection(self.node_ids):
            raise SemanticGraphError(
                "proposal annotations cannot be authority-closure nodes"
            )
        if len(self.node_ids) > self.bounds.max_nodes:
            raise SemanticGraphBoundsError("restored closure exceeds max_nodes")
        if len(self.edge_ids) > self.bounds.max_edges:
            raise SemanticGraphBoundsError("restored closure exceeds max_edges")
        if len(self.annotation_node_ids) > self.bounds.max_annotations:
            raise SemanticGraphBoundsError("restored closure exceeds max_annotations")
        restored_depth = max(
            (len(path) - 1 for path in canonical_paths.values()),
            default=0,
        )
        if restored_depth > self.bounds.max_depth:
            raise SemanticGraphBoundsError("restored closure exceeds max_depth")

    @property
    def closure_id(self) -> str:
        # Deliberately excludes graph identity and bounds: unrelated graph
        # growth and a looser unused budget cannot perturb a decision closure.
        return _identity(
            "mandatory-closure",
            {
                "schema": MANDATORY_CLOSURE_SCHEMA,
                "root_id": self.root_id,
                "decision_id": self.decision_id,
                "node_ids": list(self.node_ids),
                "edge_ids": list(self.edge_ids),
                "paths": {key: list(value) for key, value in self.paths.items()},
            },
        )

    @property
    def complete(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANDATORY_CLOSURE_SCHEMA,
            "closure_id": self.closure_id,
            "root_id": self.root_id,
            "decision_id": self.decision_id,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "paths": {key: list(value) for key, value in self.paths.items()},
            "annotation_node_ids": list(self.annotation_node_ids),
            "annotation_edge_ids": list(self.annotation_edge_ids),
            "bounds": self.bounds.to_dict(),
            "complete": True,
            "truncated": False,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_semantic_json(self.to_dict())
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MandatoryClosure":
        schema = str(payload.get("schema") or MANDATORY_CLOSURE_SCHEMA)
        if schema != MANDATORY_CLOSURE_SCHEMA:
            raise SemanticGraphError(f"unsupported closure schema: {schema}")
        bounds = payload.get("bounds") or {}
        if not isinstance(bounds, Mapping):
            raise SemanticGraphError("closure bounds must be a mapping")
        paths = payload.get("paths") or {}
        if not isinstance(paths, Mapping):
            raise SemanticGraphError("closure paths must be a mapping")
        closure = cls(
            root_id=str(payload.get("root_id") or ""),
            decision_id=str(payload.get("decision_id") or ""),
            node_ids=tuple(payload.get("node_ids") or ()),
            edge_ids=tuple(payload.get("edge_ids") or ()),
            paths={
                str(key): tuple(value)
                for key, value in paths.items()
            },
            annotation_node_ids=tuple(
                payload.get("annotation_node_ids") or ()
            ),
            annotation_edge_ids=tuple(
                payload.get("annotation_edge_ids") or ()
            ),
            bounds=ClosureBounds(
                max_nodes=bounds.get("max_nodes", DEFAULT_MAX_CLOSURE_NODES),
                max_edges=bounds.get("max_edges", DEFAULT_MAX_CLOSURE_EDGES),
                max_depth=bounds.get("max_depth", DEFAULT_MAX_CLOSURE_DEPTH),
                max_annotations=bounds.get(
                    "max_annotations", DEFAULT_MAX_ANNOTATIONS
                ),
            ),
        )
        claimed = str(payload.get("closure_id") or "")
        if claimed and claimed != closure.closure_id:
            raise SemanticGraphError("mandatory closure identity mismatch")
        if payload.get("complete") is False or payload.get("truncated") is True:
            raise SemanticGraphError("mandatory closure must be complete")
        return closure

    @classmethod
    def from_json(cls, payload: str) -> "MandatoryClosure":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SemanticGraphError("mandatory closure JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise SemanticGraphError("mandatory closure JSON must contain an object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class SemanticDependencyGraph:
    root_id: str
    nodes: tuple[SemanticNode, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()

    def __post_init__(self) -> None:
        root_id = _text(self.root_id, "graph root_id")
        node_map: dict[str, SemanticNode] = {}
        for value in self.nodes:
            node = value if isinstance(value, SemanticNode) else SemanticNode.from_dict(value)
            if node.root_id != root_id:
                raise CrossRootEdgeError(
                    f"node {node.node_id!r} is bound to a foreign root"
                )
            old = node_map.get(node.node_id)
            if old is not None and old.to_dict() != node.to_dict():
                raise SemanticGraphError(f"conflicting semantic node: {node.node_id}")
            node_map[node.node_id] = node
        if len(node_map) > DEFAULT_MAX_GRAPH_NODES:
            raise SemanticGraphBoundsError("semantic graph has too many nodes")

        edge_map: dict[str, SemanticEdge] = {}
        for value in self.edges:
            edge = value if isinstance(value, SemanticEdge) else SemanticEdge.from_dict(value)
            source = node_map.get(edge.source)
            target = node_map.get(edge.target)
            if source is None or target is None:
                raise SemanticGraphError(
                    f"edge {edge.edge_id} references an unknown node"
                )
            if (
                edge.root_id != root_id
                or source.root_id != edge.root_id
                or target.root_id != edge.root_id
            ):
                raise CrossRootEdgeError(
                    f"edge {edge.edge_id} crosses semantic roots"
                )
            if edge.authoritative and (
                not source.authoritative or not target.authoritative
            ):
                raise SemanticGraphError(
                    "authoritative edge cannot promote a non-authoritative endpoint"
                )
            if (
                edge.kind
                in {SemanticEdgeKind.AUTHORIZES, SemanticEdgeKind.DENIES}
                and source.kind
                not in {
                    SemanticNodeKind.AUTHORIZATION,
                    SemanticNodeKind.SECURITY_POLICY,
                    SemanticNodeKind.DECISION,
                }
            ):
                raise SemanticGraphError(
                    f"{edge.kind.value} source is not an authorization decision or policy"
                )
            if (
                edge.kind is SemanticEdgeKind.PROVEN_BY
                and target.kind is not SemanticNodeKind.PROOF
            ):
                raise SemanticGraphError("proven_by target must be a proof")
            if (
                edge.kind is SemanticEdgeKind.MONITORED_BY
                and target.kind is not SemanticNodeKind.MONITOR
            ):
                raise SemanticGraphError("monitored_by target must be a monitor")
            edge_map[edge.edge_id] = edge
        if len(edge_map) > DEFAULT_MAX_GRAPH_EDGES:
            raise SemanticGraphBoundsError("semantic graph has too many edges")

        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )
        self._reject_unsafe_cycles()

    @property
    def graph_id(self) -> str:
        return _identity(
            "semantic-graph",
            {
                "schema": SEMANTIC_DEPENDENCY_GRAPH_SCHEMA,
                "root_id": self.root_id,
                "nodes": [item.to_dict() for item in self.nodes],
                "edges": [item.to_dict() for item in self.edges],
            },
        )

    def _reject_unsafe_cycles(self) -> None:
        adjacency: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        for edge in self.edges:
            if (
                edge.authoritative
                and edge.mandatory
                and edge.kind in _UNSAFE_CYCLE_EDGE_KINDS
            ):
                adjacency.setdefault(edge.source, set()).add(edge.target)
                adjacency.setdefault(edge.target, set())
        for source, targets in adjacency.items():
            indegree.setdefault(source, 0)
            for target in targets:
                indegree[target] = indegree.get(target, 0) + 1
        ready = deque(sorted(key for key, degree in indegree.items() if degree == 0))
        visited = 0
        while ready:
            current = ready.popleft()
            visited += 1
            for target in sorted(adjacency.get(current, ())):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(indegree):
            cycle_nodes = sorted(key for key, degree in indegree.items() if degree)
            raise UnsafeDependencyCycleError(
                "unsafe mandatory dependency cycle at "
                + ", ".join(repr(item) for item in cycle_nodes[:8])
            )

    def node(self, node_id: str) -> SemanticNode:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def nodes_by_kind(
        self, kind: SemanticNodeKind | str
    ) -> tuple[SemanticNode, ...]:
        expected = _enum(kind, SemanticNodeKind, "node kind")
        return tuple(item for item in self.nodes if item.kind is expected)

    def edges_by_kind(
        self, kind: SemanticEdgeKind | str
    ) -> tuple[SemanticEdge, ...]:
        expected = _enum(kind, SemanticEdgeKind, "edge kind")
        return tuple(item for item in self.edges if item.kind is expected)

    def mandatory_closure(
        self,
        decision_id: str,
        *,
        bounds: ClosureBounds | None = None,
    ) -> MandatoryClosure:
        limits = bounds or ClosureBounds()
        node_by_id = {item.node_id: item for item in self.nodes}
        try:
            seed = node_by_id[decision_id]
        except KeyError as exc:
            raise KeyError(decision_id) from exc
        if seed.kind is not SemanticNodeKind.DECISION:
            raise SemanticGraphError("mandatory closure seed must be a decision")
        if not seed.authoritative:
            raise SemanticGraphError(
                "mandatory closure seed must be authority-bearing"
            )

        outgoing: dict[str, list[SemanticEdge]] = {}
        for edge in self.edges:
            outgoing.setdefault(edge.source, []).append(edge)
        for values in outgoing.values():
            values.sort(key=lambda item: (item.kind.value, item.target, item.edge_id))

        paths: dict[str, tuple[str, ...]] = {decision_id: (decision_id,)}
        depths = {decision_id: 0}
        included_edges: set[str] = set()
        queue: deque[str] = deque((decision_id,))
        while queue:
            current = queue.popleft()
            for edge in outgoing.get(current, ()):
                if not edge.authoritative or not edge.mandatory:
                    continue
                target = node_by_id[edge.target]
                if not target.authoritative:
                    raise SemanticGraphError(
                        "mandatory authority edge reached a non-authoritative node"
                    )
                depth = depths[current] + 1
                if depth > limits.max_depth:
                    raise SemanticGraphBoundsError(
                        "mandatory closure exceeds max_depth"
                    )
                included_edges.add(edge.edge_id)
                if len(included_edges) > limits.max_edges:
                    raise SemanticGraphBoundsError(
                        "mandatory closure exceeds max_edges"
                    )
                candidate = (*paths[current], edge.target)
                previous = paths.get(edge.target)
                if previous is None:
                    paths[edge.target] = candidate
                    depths[edge.target] = depth
                    if len(paths) > limits.max_nodes:
                        raise SemanticGraphBoundsError(
                            "mandatory closure exceeds max_nodes"
                        )
                    queue.append(edge.target)
                elif (len(candidate), candidate) < (len(previous), previous):
                    paths[edge.target] = candidate
                    depths[edge.target] = depth

        authority_nodes = set(paths)
        annotation_nodes: set[str] = set()
        annotation_edges: set[str] = set()
        for edge in self.edges:
            if edge.authoritative and edge.mandatory:
                continue
            attached = (
                edge.source in authority_nodes or edge.target in authority_nodes
            )
            if not attached:
                continue
            other_ids = {edge.source, edge.target} - authority_nodes
            if not other_ids:
                continue
            if all(not node_by_id[item].authoritative for item in other_ids):
                annotation_nodes.update(other_ids)
                annotation_edges.add(edge.edge_id)
                if len(annotation_nodes) > limits.max_annotations:
                    raise SemanticGraphBoundsError(
                        "closure annotations exceed max_annotations"
                    )

        return MandatoryClosure(
            root_id=self.root_id,
            decision_id=decision_id,
            node_ids=tuple(paths),
            edge_ids=tuple(included_edges),
            paths=paths,
            annotation_node_ids=tuple(annotation_nodes),
            annotation_edge_ids=tuple(annotation_edges),
            bounds=limits,
        )

    closure = mandatory_closure
    forward_mandatory_closure = mandatory_closure

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_DEPENDENCY_GRAPH_SCHEMA,
            "graph_id": self.graph_id,
            "root_id": self.root_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_semantic_json(self.to_dict())
        return json.dumps(
            _plain(self.to_dict()),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticDependencyGraph":
        schema = str(payload.get("schema") or SEMANTIC_DEPENDENCY_GRAPH_SCHEMA)
        if schema != SEMANTIC_DEPENDENCY_GRAPH_SCHEMA:
            raise SemanticGraphError(f"unsupported semantic graph schema: {schema}")
        raw_nodes = payload.get("nodes") or ()
        raw_edges = payload.get("edges") or ()
        if (
            isinstance(raw_nodes, (str, bytes))
            or not isinstance(raw_nodes, Sequence)
            or not all(isinstance(item, Mapping) for item in raw_nodes)
        ):
            raise SemanticGraphError("semantic graph nodes must be a sequence of mappings")
        if (
            isinstance(raw_edges, (str, bytes))
            or not isinstance(raw_edges, Sequence)
            or not all(isinstance(item, Mapping) for item in raw_edges)
        ):
            raise SemanticGraphError("semantic graph edges must be a sequence of mappings")
        graph = cls(
            root_id=str(payload.get("root_id") or ""),
            nodes=tuple(
                SemanticNode.from_dict(item) for item in raw_nodes
            ),
            edges=tuple(
                SemanticEdge.from_dict(item) for item in raw_edges
            ),
        )
        claimed = str(payload.get("graph_id") or "")
        if claimed and claimed != graph.graph_id:
            raise SemanticGraphError("semantic graph identity mismatch")
        if "node_count" in payload and int(payload["node_count"]) != len(graph.nodes):
            raise SemanticGraphError("semantic graph node_count mismatch")
        if "edge_count" in payload and int(payload["edge_count"]) != len(graph.edges):
            raise SemanticGraphError("semantic graph edge_count mismatch")
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "SemanticDependencyGraph":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SemanticGraphError("semantic graph JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise SemanticGraphError("semantic graph JSON must contain an object")
        return cls.from_dict(value)


_NORMALIZED_FAMILY_KIND: Mapping[tuple[str, str, str], SemanticNodeKind] = {
    # IntentIR constraint families.
    **{
        ("intent_ir", "declaration", name): SemanticNodeKind(f"intent_{name}")
        for name in (
            "goal",
            "action",
            "control_flow",
            "precondition",
            "guard",
            "invariant",
            "effect",
            "postcondition",
            "assumption",
            "failure",
            "retry",
            "verification",
        )
    },
    ("intent_ir", "formal_view", "*"): SemanticNodeKind.INTENT_FORMAL_VIEW,
    ("intent_ir", "claim", "*"): SemanticNodeKind.INTENT_CLAIM,
    ("intent_ir", "assumption", "*"): SemanticNodeKind.INTENT_ASSUMPTION,
    ("intent_ir", "obligation", "*"): SemanticNodeKind.INTENT_OBLIGATION,
    ("intent_ir", "result_authority", "*"): SemanticNodeKind.INTENT_RESULT_AUTHORITY,
    # LegalIR modality families and normalized supporting records.
    ("legal_ir", "declaration", "obligation"): SemanticNodeKind.LEGAL_OBLIGATION,
    ("legal_ir", "declaration", "prohibition"): SemanticNodeKind.LEGAL_PROHIBITION,
    ("legal_ir", "declaration", "permission"): SemanticNodeKind.LEGAL_PERMISSION,
    ("legal_ir", "declaration", "power"): SemanticNodeKind.LEGAL_POWER,
    ("legal_ir", "declaration", "exception"): SemanticNodeKind.LEGAL_EXCEPTION,
    ("legal_ir", "formal_view", "*"): SemanticNodeKind.LEGAL_FORMAL_VIEW,
    ("legal_ir", "claim", "*"): SemanticNodeKind.LEGAL_CLAIM,
    ("legal_ir", "assumption", "*"): SemanticNodeKind.LEGAL_ASSUMPTION,
    ("legal_ir", "obligation", "*"): SemanticNodeKind.LEGAL_PROOF_OBLIGATION,
    ("legal_ir", "result_authority", "*"): SemanticNodeKind.LEGAL_RESULT_AUTHORITY,
    # SecurityIR normalized declaration families.
    **{
        ("security_ir", "declaration", name): SemanticNodeKind(f"security_{name}")
        for name in (
            "principal",
            "asset",
            "resource",
            "zone",
            "channel",
            "policy",
            "state_machine",
        )
    },
    ("security_ir", "assumption", "*"): SemanticNodeKind.SECURITY_THREAT_ASSUMPTION,
    ("security_ir", "formal_view", "*"): SemanticNodeKind.SECURITY_FORMAL_VIEW,
    ("security_ir", "claim", "*"): SemanticNodeKind.SECURITY_CLAIM,
    ("security_ir", "obligation", "*"): SemanticNodeKind.SECURITY_OBLIGATION,
    ("security_ir", "result_authority", "*"): SemanticNodeKind.SECURITY_RESULT_AUTHORITY,
}


def _normalized_kind(record: Mapping[str, Any]) -> SemanticNodeKind:
    family = str(record.get("family") or "")
    node_kind = str(record.get("node_kind") or "")
    declaration = str(record.get("declaration_kind") or "").lower()
    aliases = {
        "objective": "goal",
        "desired_state": "goal",
        "task": "action",
        "step": "action",
        "operation": "action",
        "ordering": "control_flow",
        "sequence": "control_flow",
        "parallel": "control_flow",
        "pre_condition": "precondition",
        "post_condition": "postcondition",
        "duty": "obligation",
        "forbidden": "prohibition",
        "permitted": "permission",
        "right": "permission",
        "trust_zone": "zone",
        "authorization_policy": "policy",
        "security_state_machine": "state_machine",
        "threat": "assumption",
        "threat_assumption": "assumption",
    }
    declaration = aliases.get(declaration, declaration)
    key = (family, node_kind, declaration)
    wildcard = (family, node_kind, "*")
    if key in _NORMALIZED_FAMILY_KIND:
        return _NORMALIZED_FAMILY_KIND[key]
    if wildcard in _NORMALIZED_FAMILY_KIND:
        return _NORMALIZED_FAMILY_KIND[wildcard]
    if node_kind == "declaration":
        return {
            "intent_ir": SemanticNodeKind.INTENT_DECLARATION,
            "legal_ir": SemanticNodeKind.LEGAL_DECLARATION,
            "security_ir": SemanticNodeKind.SECURITY_DECLARATION,
        }[family]
    raise SemanticGraphError(
        "unsupported normalized IR constraint family: "
        f"{family}/{node_kind}/{declaration}"
    )


def _semantic_trust(value: Any) -> SemanticTrust:
    raw = str(getattr(value, "value", value) or "unknown")
    try:
        return SemanticTrust(raw)
    except ValueError as exc:
        raise SemanticGraphError(f"unsupported semantic trust: {raw}") from exc


def _semantic_authority(value: Any) -> SemanticAuthority:
    raw = str(getattr(value, "value", value) or "none")
    aliases = {
        "authoritative_input": SemanticAuthority.AUTHORITATIVE,
        "authoritative": SemanticAuthority.AUTHORITATIVE,
        "verified": SemanticAuthority.VERIFIED_INPUT,
        "advisory": SemanticAuthority.CONTEXT_ONLY,
        "proposal": SemanticAuthority.PROPOSAL_ONLY,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return SemanticAuthority(raw)
    except ValueError as exc:
        raise SemanticGraphError(f"unsupported semantic authority: {raw}") from exc


def nodes_from_normalized_ir(
    artifact: Any,
    *,
    root_id: str,
) -> tuple[SemanticNode, ...]:
    """Project every normalized IntentIR/LegalIR/SecurityIR family."""

    payload = artifact.to_dict() if callable(getattr(artifact, "to_dict", None)) else artifact
    if not isinstance(payload, Mapping):
        raise SemanticGraphError("normalized IR artifact must be a typed record")
    family = str(payload.get("family") or "")
    if family not in {"intent_ir", "legal_ir", "security_ir"}:
        raise SemanticGraphError("only normalized IntentIR/LegalIR/SecurityIR is supported")
    source_root = str(payload.get("root_artifact_id") or "")
    version = str(payload.get("artifact_schema_version") or "")
    if not source_root or not version:
        raise SemanticGraphError("normalized IR artifact root and version are required")
    artifact_trust = _semantic_trust(payload.get("trust_state", "unknown"))
    artifact_authority = _semantic_authority(
        payload.get("declared_authority", "none")
    )
    authority_rank = {
        SemanticAuthority.NONE: 0,
        SemanticAuthority.UNTRUSTED: 0,
        SemanticAuthority.PROPOSAL_ONLY: 1,
        SemanticAuthority.CONTEXT_ONLY: 2,
        SemanticAuthority.DESCRIPTIVE_INPUT: 3,
        SemanticAuthority.VERIFIED_INPUT: 4,
        SemanticAuthority.CONSTRAINT_INPUT: 4,
        SemanticAuthority.POLICY_INPUT: 4,
        SemanticAuthority.AUTHORITATIVE: 5,
    }
    allowed_domain_authority = {
        "intent_ir": {
            SemanticAuthority.DESCRIPTIVE_INPUT,
            SemanticAuthority.CONTEXT_ONLY,
            SemanticAuthority.PROPOSAL_ONLY,
            SemanticAuthority.VERIFIED_INPUT,
            SemanticAuthority.UNTRUSTED,
            SemanticAuthority.NONE,
        },
        "legal_ir": {
            SemanticAuthority.CONSTRAINT_INPUT,
            SemanticAuthority.CONTEXT_ONLY,
            SemanticAuthority.PROPOSAL_ONLY,
            SemanticAuthority.VERIFIED_INPUT,
            SemanticAuthority.UNTRUSTED,
            SemanticAuthority.NONE,
        },
        "security_ir": {
            SemanticAuthority.POLICY_INPUT,
            SemanticAuthority.CONTEXT_ONLY,
            SemanticAuthority.PROPOSAL_ONLY,
            SemanticAuthority.VERIFIED_INPUT,
            SemanticAuthority.UNTRUSTED,
            SemanticAuthority.NONE,
        },
    }[family]
    result: list[SemanticNode] = []
    for collection in (
        "declarations",
        "formal_views",
        "claims",
        "assumptions",
        "obligations",
        "result_authority",
    ):
        values = payload.get(collection) or ()
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise SemanticGraphError(f"normalized IR {collection} must be a sequence")
        for value in values:
            record = value.to_dict() if callable(getattr(value, "to_dict", None)) else value
            if not isinstance(record, Mapping):
                raise SemanticGraphError("normalized IR node must be a typed record")
            if str(record.get("family") or "") != family:
                raise SemanticGraphError(
                    "normalized IR node family differs from its artifact"
                )
            node_id = str(record.get("node_id") or "")
            trust = _semantic_trust(
                record.get("trust_state", payload.get("trust_state", "unknown"))
            )
            authority = _semantic_authority(
                record.get(
                    "result_authority",
                    record.get(
                        "declared_authority",
                        payload.get("declared_authority", "none"),
                    ),
                )
            )
            if trust.accepted and not artifact_trust.accepted:
                raise SemanticGraphError(
                    "normalized IR node trust exceeds its artifact"
                )
            if (
                authority not in allowed_domain_authority
                or authority_rank[authority] > authority_rank[artifact_authority]
            ):
                raise SemanticGraphError(
                    "normalized IR node authority exceeds its artifact or family"
                )
            result.append(
                SemanticNode(
                    node_id=node_id,
                    kind=_normalized_kind(record),
                    root_id=root_id,
                    source_root_id=source_root,
                    provenance=SemanticProvenance(family),
                    provenance_id=str(
                        payload.get("source_artifact_id") or source_root
                    ),
                    trust=trust,
                    authority=authority,
                    version=version,
                    record=record,
                )
            )
    return tuple(result)


def nodes_and_edges_from_normalized_ir(
    artifact: Any,
    *,
    root_id: str,
) -> tuple[tuple[SemanticNode, ...], tuple[SemanticEdge, ...]]:
    """Project normalized nodes plus exact, in-artifact semantic references."""

    payload = artifact.to_dict() if callable(getattr(artifact, "to_dict", None)) else artifact
    if not isinstance(payload, Mapping):
        raise SemanticGraphError("normalized IR artifact must be a typed record")
    nodes = nodes_from_normalized_ir(payload, root_id=root_id)
    by_id = {item.node_id: item for item in nodes}
    family = str(payload.get("family") or "")
    provenance = SemanticProvenance(family)
    version = str(payload.get("artifact_schema_version") or "")
    source_root = str(payload.get("root_artifact_id") or "")
    artifact_id = str(payload.get("source_artifact_id") or source_root)
    relationship_fields = {
        "requires": SemanticEdgeKind.REQUIRES,
        "required_ids": SemanticEdgeKind.REQUIRES,
        "required_evidence_ids": SemanticEdgeKind.REQUIRES,
        "precondition_ids": SemanticEdgeKind.REQUIRES,
        "guard_ids": SemanticEdgeKind.REQUIRES,
        "invariant_ids": SemanticEdgeKind.REQUIRES,
        "postcondition_ids": SemanticEdgeKind.REQUIRES,
        "verification_ids": SemanticEdgeKind.REQUIRES,
        "assumption_ids": SemanticEdgeKind.REQUIRES,
        "claim_ids": SemanticEdgeKind.REQUIRES,
        "obligation_ids": SemanticEdgeKind.REQUIRES,
        "proof_obligation_ids": SemanticEdgeKind.REQUIRES,
        "formal_view_ids": SemanticEdgeKind.REQUIRES,
        "provision_ids": SemanticEdgeKind.APPLIES_TO,
        "subject_constraint_ids": SemanticEdgeKind.APPLIES_TO,
        "constrained_by": SemanticEdgeKind.CONSTRAINED_BY,
        "policy_ids": SemanticEdgeKind.CONSTRAINED_BY,
        "state_machine_id": SemanticEdgeKind.CONSTRAINED_BY,
        "applies_to": SemanticEdgeKind.APPLIES_TO,
        "resource_id": SemanticEdgeKind.APPLIES_TO,
        "action_ids": SemanticEdgeKind.APPLIES_TO,
        "goal_ids": SemanticEdgeKind.APPLIES_TO,
        "exception_to": SemanticEdgeKind.EXCEPTION_TO,
        "conflicts_with": SemanticEdgeKind.CONFLICTS_WITH,
        "depends_on": SemanticEdgeKind.DEPENDS_ON,
        "premise_ids": SemanticEdgeKind.DEPENDS_ON,
        "implements": SemanticEdgeKind.IMPLEMENTS,
        "affects": SemanticEdgeKind.AFFECTS,
        "invalidates": SemanticEdgeKind.INVALIDATES,
    }
    edges: list[SemanticEdge] = []
    for node in nodes:
        fields = dict(node.record)
        attributes = fields.get("attributes")
        if isinstance(attributes, Mapping):
            fields.update(attributes)
        for field_name, edge_kind in relationship_fields.items():
            raw_targets = fields.get(field_name)
            if raw_targets is None:
                continue
            targets = (
                (raw_targets,)
                if isinstance(raw_targets, str)
                else raw_targets
            )
            if not isinstance(targets, Sequence):
                raise SemanticGraphError(
                    f"normalized IR relationship {field_name} must be a sequence"
                )
            for target_id in sorted({str(item) for item in targets}):
                target = by_id.get(target_id)
                if target is None:
                    raise SemanticGraphError(
                        f"normalized IR relationship references unknown node {target_id!r}"
                    )
                authoritative = node.authoritative and target.authoritative
                edges.append(
                    SemanticEdge(
                        source=node.node_id,
                        target=target_id,
                        kind=edge_kind,
                        root_id=root_id,
                        source_root_id=source_root,
                        provenance=provenance,
                        provenance_id=(
                            f"{artifact_id}:{node.node_id}:"
                            f"{field_name}:{target_id}"
                        ),
                        trust=(
                            SemanticTrust.VERIFIED
                            if authoritative
                            else SemanticTrust.UNTRUSTED
                        ),
                        authority=(
                            min(
                                (node.authority, target.authority),
                                key=lambda item: {
                                    SemanticAuthority.NONE: 0,
                                    SemanticAuthority.UNTRUSTED: 0,
                                    SemanticAuthority.PROPOSAL_ONLY: 1,
                                    SemanticAuthority.CONTEXT_ONLY: 2,
                                    SemanticAuthority.DESCRIPTIVE_INPUT: 3,
                                    SemanticAuthority.VERIFIED_INPUT: 4,
                                    SemanticAuthority.CONSTRAINT_INPUT: 4,
                                    SemanticAuthority.POLICY_INPUT: 4,
                                    SemanticAuthority.AUTHORITATIVE: 5,
                                }[item],
                            )
                            if authoritative
                            else SemanticAuthority.PROPOSAL_ONLY
                        ),
                        version=version,
                        mandatory=authoritative,
                        record={"relationship_field": field_name},
                    )
                )
    return nodes, tuple(edges)


def nodes_and_edges_from_code_evidence(
    evidence_graph: Any,
    *,
    root_id: str,
    version: str = "code-evidence-graph@1",
) -> tuple[tuple[SemanticNode, ...], tuple[SemanticEdge, ...]]:
    """Losslessly project legacy code/proof evidence into this graph.

    Legacy authority is recomputed from the typed records; claimed boolean
    authority fields are never trusted.
    """

    from .code_evidence_graph import EvidenceEdgeKind, EvidenceNodeKind

    node_kinds = {
        EvidenceNodeKind.TASK: SemanticNodeKind.PLAN,
        EvidenceNodeKind.TREE: SemanticNodeKind.REPOSITORY_TREE,
        EvidenceNodeKind.SYMBOL: SemanticNodeKind.SYMBOL,
        EvidenceNodeKind.AST_SCOPE: SemanticNodeKind.AST,
        EvidenceNodeKind.OBLIGATION: SemanticNodeKind.OBLIGATION,
        EvidenceNodeKind.ATTEMPT: SemanticNodeKind.PREMISE,
        EvidenceNodeKind.PROOF: SemanticNodeKind.PROOF,
        EvidenceNodeKind.VALIDATION: SemanticNodeKind.VALIDATION,
        EvidenceNodeKind.MERGE: SemanticNodeKind.MERGE_EVIDENCE,
        EvidenceNodeKind.EVIDENCE: SemanticNodeKind.PREMISE,
        EvidenceNodeKind.ENRICHMENT: SemanticNodeKind.ANNOTATION,
    }
    provenance = {
        "ast": SemanticProvenance.AST,
        "task": SemanticProvenance.PLANNER,
        "proof": SemanticProvenance.PROOF,
        "validation": SemanticProvenance.VALIDATION,
        "merge": SemanticProvenance.MERGE,
        "enrichment": SemanticProvenance.RETRIEVAL,
        "llm": SemanticProvenance.MODEL,
        "graphrag": SemanticProvenance.GRAPHRAG,
    }
    nodes = tuple(
        SemanticNode(
            node_id=item.node_id,
            kind=node_kinds[item.kind],
            root_id=root_id,
            source_root_id=item.tree_id or evidence_graph.graph_id,
            provenance=provenance[item.provenance.value],
            provenance_id=item.record_key,
            trust=(
                SemanticTrust.VERIFIED
                if item.authoritative
                else SemanticTrust.UNTRUSTED
            ),
            authority=(
                SemanticAuthority.VERIFIED_INPUT
                if item.authoritative
                else SemanticAuthority.PROPOSAL_ONLY
            ),
            version=version,
            record=item.record,
        )
        for item in evidence_graph.nodes
    )
    node_by_id = {item.node_id: item for item in nodes}
    edge_kinds = {
        EvidenceEdgeKind.DEPENDS_ON: (SemanticEdgeKind.DEPENDS_ON, False),
        EvidenceEdgeKind.TARGETS_TREE: (SemanticEdgeKind.SOURCED_FROM, False),
        EvidenceEdgeKind.DEFINES_SYMBOL: (SemanticEdgeKind.IMPLEMENTS, False),
        EvidenceEdgeKind.CONTAINS: (SemanticEdgeKind.IMPLEMENTS, False),
        EvidenceEdgeKind.HAS_OBLIGATION: (SemanticEdgeKind.REQUIRES, False),
        EvidenceEdgeKind.COVERS: (SemanticEdgeKind.APPLIES_TO, False),
        EvidenceEdgeKind.ATTEMPT_FOR: (SemanticEdgeKind.DEPENDS_ON, False),
        EvidenceEdgeKind.DERIVED_FROM: (SemanticEdgeKind.SOURCED_FROM, False),
        EvidenceEdgeKind.PROVES: (SemanticEdgeKind.PROVEN_BY, True),
        EvidenceEdgeKind.VALIDATES: (SemanticEdgeKind.SOURCED_FROM, True),
        EvidenceEdgeKind.MERGED: (SemanticEdgeKind.SOURCED_FROM, True),
        EvidenceEdgeKind.COMPLETES: (SemanticEdgeKind.SOURCED_FROM, True),
        EvidenceEdgeKind.RELATED_TO: (SemanticEdgeKind.AFFECTS, False),
        EvidenceEdgeKind.MENTIONS: (SemanticEdgeKind.AFFECTS, False),
        EvidenceEdgeKind.SUGGESTS: (SemanticEdgeKind.AFFECTS, False),
    }
    projected_edges: list[SemanticEdge] = []
    for item in evidence_graph.edges:
        kind, reverse = edge_kinds[item.kind]
        source, target = (
            (item.target, item.source) if reverse else (item.source, item.target)
        )
        semantic_provenance = provenance[item.provenance.value]
        # PROVES is reversed to read "obligation proven_by proof".
        if kind is SemanticEdgeKind.PROVEN_BY:
            semantic_provenance = SemanticProvenance.PROOF
        authoritative = item.authoritative
        projected_edges.append(
            SemanticEdge(
                source=source,
                target=target,
                kind=kind,
                root_id=root_id,
                provenance=semantic_provenance,
                provenance_id=item.provenance_record_id,
                source_root_id=(
                    node_by_id[source].source_root_id
                    if node_by_id[source].source_root_id
                    == node_by_id[target].source_root_id
                    else evidence_graph.graph_id
                ),
                trust=(
                    SemanticTrust.VERIFIED
                    if authoritative
                    else SemanticTrust.UNTRUSTED
                ),
                authority=(
                    SemanticAuthority.VERIFIED_INPUT
                    if authoritative
                    else SemanticAuthority.PROPOSAL_ONLY
                ),
                version=version,
                mandatory=authoritative,
                record={
                    "legacy_edge_id": item.edge_id,
                    "legacy_kind": item.kind.value,
                    "metadata": item.metadata,
                },
            )
        )
    # Make a defensive reference so static analysis catches accidental IDs
    # omitted from the node projection before graph construction does.
    if any(
        edge.source not in node_by_id or edge.target not in node_by_id
        for edge in projected_edges
    ):
        raise SemanticGraphError("legacy evidence projection has detached edges")
    return nodes, tuple(projected_edges)


def nodes_and_edges_from_program_behavior(
    behavior: Any,
    *,
    root_id: str,
) -> tuple[tuple[SemanticNode, ...], tuple[SemanticEdge, ...]]:
    """Project an ASI-129 behavior root and all of its bounded components."""

    payload = behavior.to_dict() if callable(getattr(behavior, "to_dict", None)) else behavior
    if not isinstance(payload, Mapping):
        raise SemanticGraphError("program behavior must be a typed record")
    behavior_root = str(payload.get("behavior_root") or "")
    schema_version = str(payload.get("schema_version") or "")
    if not behavior_root or not schema_version:
        raise SemanticGraphError("program behavior root and schema version are required")
    version = f"program-behavior@{schema_version}"
    node_values: list[SemanticNode] = []
    edge_values: list[SemanticEdge] = []

    def add_node(
        node_id: str,
        kind: SemanticNodeKind,
        source_root_id: str,
        provenance: SemanticProvenance,
        record: Mapping[str, Any],
    ) -> None:
        node_values.append(
            SemanticNode(
                node_id=node_id,
                kind=kind,
                root_id=root_id,
                source_root_id=source_root_id,
                provenance=provenance,
                provenance_id=node_id,
                trust=SemanticTrust.VERIFIED,
                authority=SemanticAuthority.VERIFIED_INPUT,
                version=version,
                record=record,
            )
        )

    def add_edge(
        source: str,
        target: str,
        kind: SemanticEdgeKind,
        provenance: SemanticProvenance,
        source_root_id: str,
    ) -> None:
        edge_values.append(
            SemanticEdge(
                source=source,
                target=target,
                kind=kind,
                root_id=root_id,
                source_root_id=source_root_id,
                provenance=provenance,
                provenance_id=f"{behavior_root}:{kind.value}:{source}:{target}",
                trust=SemanticTrust.VERIFIED,
                authority=SemanticAuthority.VERIFIED_INPUT,
                version=version,
                mandatory=True,
            )
        )

    add_node(
        behavior_root,
        SemanticNodeKind.PROGRAM,
        behavior_root,
        SemanticProvenance.PROGRAM,
        {
            key: payload[key]
            for key in (
                "schema",
                "schema_version",
                "repository_snapshot_id",
                "execution_tree_root",
                "program_root",
                "ast_root",
                "tool_catalog_root",
                "environment_root",
                "effect_manifest_root",
                "component_manifest_blob",
            )
            if key in payload
        },
    )

    repository = payload.get("repository") or {}
    if not isinstance(repository, Mapping):
        raise SemanticGraphError("program behavior repository must be a mapping")
    snapshot_id = str(
        repository.get("snapshot_id") or payload.get("repository_snapshot_id") or ""
    )
    execution_root = str(
        repository.get("execution_tree_root")
        or payload.get("execution_tree_root")
        or ""
    )
    if not snapshot_id or not execution_root:
        raise SemanticGraphError("program behavior repository roots are required")
    add_node(
        snapshot_id,
        SemanticNodeKind.WORKTREE,
        execution_root,
        SemanticProvenance.WORKTREE,
        repository,
    )
    add_edge(
        behavior_root,
        snapshot_id,
        SemanticEdgeKind.DEPENDS_ON,
        SemanticProvenance.PROGRAM,
        behavior_root,
    )
    head_tree_id = str(repository.get("head_tree_id") or "")
    if head_tree_id:
        add_node(
            head_tree_id,
            SemanticNodeKind.REPOSITORY_TREE,
            head_tree_id,
            SemanticProvenance.WORKTREE,
            {
                "head_commit_id": repository.get("head_commit_id", ""),
                "head_tree_id": head_tree_id,
            },
        )
        add_edge(
            snapshot_id,
            head_tree_id,
            SemanticEdgeKind.SOURCED_FROM,
            SemanticProvenance.WORKTREE,
            execution_root,
        )
    entries = repository.get("entries") or ()
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise SemanticGraphError("repository entries must be a sequence")
    if not all(isinstance(item, Mapping) for item in entries):
        raise SemanticGraphError("repository entry must be a mapping")
    for raw in sorted(entries, key=lambda item: str(item.get("entry_id") or "")):
        entry_id = str(raw.get("entry_id") or "")
        if not entry_id:
            raise SemanticGraphError("repository entry_id is required")
        add_node(
            entry_id,
            SemanticNodeKind.FILE,
            execution_root,
            SemanticProvenance.WORKTREE,
            raw,
        )
        add_edge(
            snapshot_id,
            entry_id,
            SemanticEdgeKind.IMPLEMENTS,
            SemanticProvenance.WORKTREE,
            execution_root,
        )

    analysis = payload.get("analysis") or {}
    if not isinstance(analysis, Mapping):
        raise SemanticGraphError("program behavior analysis must be a mapping")
    program_root = str(analysis.get("program_root") or payload.get("program_root") or "")
    ast_root = str(analysis.get("ast_root") or payload.get("ast_root") or "")
    if not program_root or not ast_root:
        raise SemanticGraphError("program and AST roots are required")
    add_node(
        ast_root,
        SemanticNodeKind.AST,
        ast_root,
        SemanticProvenance.AST,
        analysis,
    )
    add_edge(
        behavior_root,
        ast_root,
        SemanticEdgeKind.DEPENDS_ON,
        SemanticProvenance.PROGRAM,
        program_root,
    )
    observations = getattr(getattr(behavior, "analysis", None), "observations", ())
    if not observations and isinstance(analysis.get("observations"), Sequence):
        observations = analysis.get("observations") or ()
    observation_kind = {
        "ast": SemanticNodeKind.AST,
        "symbol": SemanticNodeKind.SYMBOL,
        "interface": SemanticNodeKind.INTERFACE,
        "call": SemanticNodeKind.CALL,
        "data_flow": SemanticNodeKind.DATA_FLOW,
    }
    normalized_observations = [
        raw.to_dict() if callable(getattr(raw, "to_dict", None)) else raw
        for raw in observations
    ]
    for record in sorted(
        normalized_observations,
        key=lambda item: canonical_semantic_json(item),
    ):
        if not isinstance(record, Mapping):
            raise SemanticGraphError("program observation must be a typed record")
        kind_text = str(record.get("kind") or "")
        if kind_text not in observation_kind:
            raise SemanticGraphError(f"unsupported program observation: {kind_text}")
        observation_id = _identity(
            "program-observation",
            {"ast_root": ast_root, "record": record},
        )
        add_node(
            observation_id,
            observation_kind[kind_text],
            ast_root,
            SemanticProvenance.AST,
            record,
        )
        add_edge(
            ast_root,
            observation_id,
            SemanticEdgeKind.IMPLEMENTS,
            SemanticProvenance.AST,
            ast_root,
        )

    tools = payload.get("tools") or {}
    if not isinstance(tools, Mapping):
        raise SemanticGraphError("program behavior tools must be a mapping")
    catalog_root = str(tools.get("catalog_root") or payload.get("tool_catalog_root") or "")
    tool_records = tools.get("tools") or ()
    if isinstance(tool_records, (str, bytes)) or not isinstance(
        tool_records, Sequence
    ):
        raise SemanticGraphError("tool descriptors must be a sequence")
    if not all(isinstance(item, Mapping) for item in tool_records):
        raise SemanticGraphError("tool descriptor must be a mapping")
    for raw in sorted(
        tool_records, key=lambda item: str(item.get("tool_id") or "")
    ):
        tool_id = "tool:" + str(raw.get("tool_id") or "")
        if tool_id == "tool:":
            raise SemanticGraphError("tool descriptor ID is required")
        add_node(
            tool_id,
            SemanticNodeKind.TOOL,
            catalog_root,
            SemanticProvenance.TOOL,
            raw,
        )
        add_edge(
            behavior_root,
            tool_id,
            SemanticEdgeKind.REQUIRES,
            SemanticProvenance.TOOL,
            catalog_root,
        )

    environment = payload.get("environment") or {}
    if not isinstance(environment, Mapping):
        raise SemanticGraphError("program behavior environment must be a mapping")
    environment_root = str(
        environment.get("environment_root") or payload.get("environment_root") or ""
    )
    if not environment_root:
        raise SemanticGraphError("program behavior environment root is required")
    add_node(
        environment_root,
        SemanticNodeKind.ENVIRONMENT,
        environment_root,
        SemanticProvenance.PROGRAM,
        environment,
    )
    add_edge(
        behavior_root,
        environment_root,
        SemanticEdgeKind.DEPENDS_ON,
        SemanticProvenance.PROGRAM,
        environment_root,
    )

    effects = payload.get("effects") or {}
    if not isinstance(effects, Mapping):
        raise SemanticGraphError("program behavior effects must be a mapping")
    manifest_root = str(
        effects.get("manifest_root") or payload.get("effect_manifest_root") or ""
    )
    effect_records = effects.get("effects") or ()
    if isinstance(effect_records, (str, bytes)) or not isinstance(
        effect_records, Sequence
    ):
        raise SemanticGraphError("proposed effects must be a sequence")
    if not all(isinstance(item, Mapping) for item in effect_records):
        raise SemanticGraphError("proposed effect must be a mapping")
    for raw in sorted(
        effect_records, key=lambda item: str(item.get("effect_id") or "")
    ):
        effect_id = str(raw.get("effect_id") or "")
        target = str(raw.get("target") or "")
        if not effect_id or not target:
            raise SemanticGraphError("proposed effect identity and target are required")
        add_node(
            effect_id,
            SemanticNodeKind.EFFECT,
            manifest_root,
            SemanticProvenance.PROGRAM,
            raw,
        )
        add_edge(
            behavior_root,
            effect_id,
            SemanticEdgeKind.REQUIRES,
            SemanticProvenance.PROGRAM,
            manifest_root,
        )
        resource_id = "resource:" + target
        if not any(item.node_id == resource_id for item in node_values):
            add_node(
                resource_id,
                SemanticNodeKind.RESOURCE,
                manifest_root,
                SemanticProvenance.PROGRAM,
                {"target": target},
            )
        add_edge(
            effect_id,
            resource_id,
            SemanticEdgeKind.AFFECTS,
            SemanticProvenance.PROGRAM,
            manifest_root,
        )

    return tuple(node_values), tuple(edge_values)


def build_semantic_dependency_graph(
    *,
    root_id: str,
    nodes: Iterable[SemanticNode | Mapping[str, Any]] = (),
    edges: Iterable[SemanticEdge | Mapping[str, Any]] = (),
    normalized_ir_artifacts: Iterable[Any] = (),
    code_evidence_graph: Any | None = None,
    program_behaviors: Iterable[Any] = (),
    program_behavior: Any | None = None,
) -> SemanticDependencyGraph:
    """Build one canonical graph from explicit and trusted typed channels."""

    projected_nodes = list(nodes)
    projected_edges = list(edges)
    for artifact in normalized_ir_artifacts:
        ir_nodes, ir_edges = nodes_and_edges_from_normalized_ir(
            artifact, root_id=root_id
        )
        projected_nodes.extend(ir_nodes)
        projected_edges.extend(ir_edges)
    behavior_values = list(program_behaviors)
    if program_behavior is not None:
        behavior_values.append(program_behavior)
    for behavior in behavior_values:
        behavior_nodes, behavior_edges = nodes_and_edges_from_program_behavior(
            behavior, root_id=root_id
        )
        projected_nodes.extend(behavior_nodes)
        projected_edges.extend(behavior_edges)
    if code_evidence_graph is not None:
        legacy_nodes, legacy_edges = nodes_and_edges_from_code_evidence(
            code_evidence_graph, root_id=root_id
        )
        projected_nodes.extend(legacy_nodes)
        projected_edges.extend(legacy_edges)
    return SemanticDependencyGraph(
        root_id=root_id,
        nodes=tuple(projected_nodes),
        edges=tuple(projected_edges),
    )


def compute_mandatory_closure(
    graph: SemanticDependencyGraph,
    decision_id: str,
    *,
    bounds: ClosureBounds | None = None,
) -> MandatoryClosure:
    return graph.mandatory_closure(decision_id, bounds=bounds)


DependencyNodeKind = SemanticNodeKind
DependencyEdgeKind = SemanticEdgeKind
DependencyNode = SemanticNode
DependencyEdge = SemanticEdge
DependencyGraph = SemanticDependencyGraph
SemanticDependencyNode = SemanticNode
SemanticDependencyEdge = SemanticEdge
GraphValidationError = SemanticGraphError
MandatoryDependencyClosure = MandatoryClosure


__all__ = [
    "DEFAULT_MAX_ANNOTATIONS",
    "DEFAULT_MAX_CLOSURE_DEPTH",
    "DEFAULT_MAX_CLOSURE_EDGES",
    "DEFAULT_MAX_CLOSURE_NODES",
    "MANDATORY_CLOSURE_SCHEMA",
    "SEMANTIC_DEPENDENCY_EDGE_SCHEMA",
    "SEMANTIC_DEPENDENCY_GRAPH_SCHEMA",
    "SEMANTIC_DEPENDENCY_NODE_SCHEMA",
    "ClosureBounds",
    "CrossRootEdgeError",
    "DependencyEdge",
    "DependencyEdgeKind",
    "DependencyGraph",
    "DependencyNode",
    "DependencyNodeKind",
    "GraphValidationError",
    "MandatoryClosure",
    "MandatoryDependencyClosure",
    "SemanticAuthority",
    "SemanticDependencyGraph",
    "SemanticDependencyEdge",
    "SemanticDependencyNode",
    "SemanticEdge",
    "SemanticEdgeKind",
    "SemanticGraphBoundsError",
    "SemanticGraphError",
    "SemanticNode",
    "SemanticNodeKind",
    "SemanticProvenance",
    "SemanticTrust",
    "UnsafeDependencyCycleError",
    "build_semantic_dependency_graph",
    "canonical_semantic_json",
    "compute_mandatory_closure",
    "nodes_and_edges_from_code_evidence",
    "nodes_and_edges_from_normalized_ir",
    "nodes_and_edges_from_program_behavior",
    "nodes_from_normalized_ir",
]
