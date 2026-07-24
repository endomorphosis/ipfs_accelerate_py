"""Deterministic, query-friendly graph of code and proof evidence.

The graph is deliberately a projection of records produced by trusted
supervisor boundaries.  It is not a knowledge-graph inference engine: model or
GraphRAG annotations may add descriptive ``related_to`` edges, but they cannot
manufacture proof, validation, coverage, merge, or completion facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

CODE_EVIDENCE_GRAPH_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.code-evidence-graph@1"
)
CODE_EVIDENCE_NODE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.code-evidence-node@1"
)
CODE_EVIDENCE_EDGE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.code-evidence-edge@1"
)
CODE_IMPACT_INDEX_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.code-impact-index@1"
)
CODE_IMPACT_RESULT_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.code-impact-result@1"
)


class EvidenceGraphValidationError(ValueError):
    """Raised when graph input is malformed or violates the trust boundary."""


class EvidenceNodeKind(str, Enum):
    TASK = "task"
    TREE = "tree"
    SYMBOL = "symbol"
    AST_SCOPE = "ast_scope"
    AST = "ast_scope"
    OBLIGATION = "obligation"
    ATTEMPT = "attempt"
    PROOF = "proof"
    PROOF_RECEIPT = "proof"
    VALIDATION = "validation"
    VALIDATION_RECEIPT = "validation"
    MERGE = "merge"
    MERGE_RECEIPT = "merge"
    EVIDENCE = "evidence"
    ENRICHMENT = "enrichment"


class EvidenceEdgeKind(str, Enum):
    DEPENDS_ON = "depends_on"
    TARGETS_TREE = "targets_tree"
    DEFINES_SYMBOL = "defines_symbol"
    CONTAINS = "contains"
    HAS_OBLIGATION = "has_obligation"
    COVERS = "covers"
    ATTEMPT_FOR = "attempt_for"
    DERIVED_FROM = "derived_from"
    PROVES = "proves"
    PROOF_FOR = "proves"
    VALIDATES = "validates"
    MERGED = "merged"
    COMPLETES = "completes"
    RELATED_TO = "related_to"
    MENTIONS = "mentions"
    SUGGESTS = "suggests"


class EvidenceProvenance(str, Enum):
    AST = "ast"
    TASK = "task"
    PROOF = "proof"
    VALIDATION = "validation"
    MERGE = "merge"
    ENRICHMENT = "enrichment"
    LLM = "llm"
    GRAPHRAG = "graphrag"

    @property
    def authoritative(self) -> bool:
        return self not in {
            EvidenceProvenance.ENRICHMENT,
            EvidenceProvenance.LLM,
            EvidenceProvenance.GRAPHRAG,
        }


# These relationships affect gates or claims of completion.  Only the stated
# durable record boundary is allowed to originate each one.
AUTHORITATIVE_EDGE_PROVENANCE: Mapping[EvidenceEdgeKind, frozenset[EvidenceProvenance]] = {
    EvidenceEdgeKind.DEPENDS_ON: frozenset({EvidenceProvenance.TASK, EvidenceProvenance.PROOF}),
    EvidenceEdgeKind.TARGETS_TREE: frozenset(
        {EvidenceProvenance.AST, EvidenceProvenance.TASK, EvidenceProvenance.PROOF}
    ),
    EvidenceEdgeKind.DEFINES_SYMBOL: frozenset({EvidenceProvenance.AST}),
    EvidenceEdgeKind.CONTAINS: frozenset({EvidenceProvenance.AST}),
    EvidenceEdgeKind.HAS_OBLIGATION: frozenset({EvidenceProvenance.PROOF}),
    EvidenceEdgeKind.COVERS: frozenset(
        {EvidenceProvenance.PROOF, EvidenceProvenance.VALIDATION}
    ),
    EvidenceEdgeKind.ATTEMPT_FOR: frozenset({EvidenceProvenance.PROOF}),
    EvidenceEdgeKind.DERIVED_FROM: frozenset(
        {
            EvidenceProvenance.AST,
            EvidenceProvenance.TASK,
            EvidenceProvenance.PROOF,
            EvidenceProvenance.VALIDATION,
            EvidenceProvenance.MERGE,
        }
    ),
    EvidenceEdgeKind.PROVES: frozenset({EvidenceProvenance.PROOF}),
    EvidenceEdgeKind.VALIDATES: frozenset({EvidenceProvenance.VALIDATION}),
    EvidenceEdgeKind.MERGED: frozenset({EvidenceProvenance.MERGE}),
    EvidenceEdgeKind.COMPLETES: frozenset(
        {EvidenceProvenance.VALIDATION, EvidenceProvenance.MERGE}
    ),
}
ENRICHMENT_EDGE_KINDS = frozenset(
    {
        EvidenceEdgeKind.RELATED_TO,
        EvidenceEdgeKind.MENTIONS,
        EvidenceEdgeKind.SUGGESTS,
    }
)
UNTRUSTED_PROVENANCE = frozenset(
    {
        EvidenceProvenance.ENRICHMENT,
        EvidenceProvenance.LLM,
        EvidenceProvenance.GRAPHRAG,
    }
)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise EvidenceGraphValidationError("non-finite numbers are not canonical")
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise EvidenceGraphValidationError("graph record keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=canonical_json)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if not isinstance(result, Mapping):
            raise EvidenceGraphValidationError("to_dict() must return a mapping")
        return _canonical_value(result)
    raise EvidenceGraphValidationError(
        f"unsupported graph record value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON text without insignificant whitespace."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _record(value: Any) -> dict[str, Any]:
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        value = to_record()
    normalized = _canonical_value(value)
    if not isinstance(normalized, dict):
        raise EvidenceGraphValidationError("evidence records must be mappings")
    return normalized


def _text(record: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        raise EvidenceGraphValidationError(f"invalid {label}: {value!r}") from exc


@dataclass(frozen=True)
class EvidenceNode:
    """One deterministic projection of a source record."""

    kind: EvidenceNodeKind
    record_key: str
    provenance: EvidenceProvenance
    record: Mapping[str, Any] = field(default_factory=dict)
    task_id: str = ""
    tree_id: str = ""
    symbol: str = ""
    obligation_id: str = ""
    assurance: str = ""
    freshness: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, EvidenceNodeKind, "node kind"))
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, EvidenceProvenance, "node provenance"),
        )
        key = str(self.record_key or "").strip()
        if not key:
            raise EvidenceGraphValidationError("node record_key is required")
        object.__setattr__(self, "record_key", key)
        object.__setattr__(self, "record", _record(self.record))
        for name in (
            "task_id",
            "tree_id",
            "symbol",
            "obligation_id",
            "assurance",
            "freshness",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if self.provenance in UNTRUSTED_PROVENANCE and self.kind is not EvidenceNodeKind.ENRICHMENT:
            raise EvidenceGraphValidationError(
                "enrichment provenance may only create enrichment nodes"
            )

    @property
    def node_id(self) -> str:
        return "node-" + _identity(
            {
                "schema": CODE_EVIDENCE_NODE_SCHEMA,
                "kind": self.kind.value,
                "record_key": self.record_key,
            }
        )

    @property
    def authoritative(self) -> bool:
        return self.provenance.authoritative

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_EVIDENCE_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "record_key": self.record_key,
            "provenance": self.provenance.value,
            "authoritative": self.authoritative,
            "task_id": self.task_id,
            "tree_id": self.tree_id,
            "symbol": self.symbol,
            "obligation_id": self.obligation_id,
            "assurance": self.assurance,
            "freshness": self.freshness,
            "record": dict(self.record),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceNode":
        schema = str(payload.get("schema") or CODE_EVIDENCE_NODE_SCHEMA)
        if schema != CODE_EVIDENCE_NODE_SCHEMA:
            raise EvidenceGraphValidationError(f"unsupported node schema: {schema}")
        node = cls(
            kind=payload.get("kind", ""),
            record_key=str(payload.get("record_key") or ""),
            provenance=payload.get("provenance", ""),
            record=payload.get("record") or {},
            task_id=str(payload.get("task_id") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            symbol=str(payload.get("symbol") or ""),
            obligation_id=str(payload.get("obligation_id") or ""),
            assurance=str(payload.get("assurance") or ""),
            freshness=str(payload.get("freshness") or ""),
        )
        claimed = str(payload.get("node_id") or "")
        if claimed and claimed != node.node_id:
            raise EvidenceGraphValidationError("node identity does not match payload")
        if "authoritative" in payload and bool(payload["authoritative"]) != node.authoritative:
            raise EvidenceGraphValidationError("node authority does not match provenance")
        return node


@dataclass(frozen=True)
class ProvenanceEdge:
    """A typed relationship with an immutable provenance record binding."""

    source: str
    target: str
    kind: EvidenceEdgeKind
    provenance: EvidenceProvenance
    provenance_record_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source", "target", "provenance_record_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise EvidenceGraphValidationError(f"edge {name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "kind", _enum(self.kind, EvidenceEdgeKind, "edge kind"))
        object.__setattr__(
            self,
            "provenance",
            _enum(self.provenance, EvidenceProvenance, "edge provenance"),
        )
        object.__setattr__(self, "metadata", _record(self.metadata))
        allowed = AUTHORITATIVE_EDGE_PROVENANCE.get(self.kind)
        if self.provenance in UNTRUSTED_PROVENANCE:
            if self.kind not in ENRICHMENT_EDGE_KINDS:
                raise EvidenceGraphValidationError(
                    f"enrichment cannot create {self.kind.value!r} edges"
                )
        elif allowed is not None and self.provenance not in allowed:
            raise EvidenceGraphValidationError(
                f"{self.provenance.value} records cannot create {self.kind.value!r} edges"
            )

    @property
    def authoritative(self) -> bool:
        return (
            self.kind in AUTHORITATIVE_EDGE_PROVENANCE
            and self.provenance
            in AUTHORITATIVE_EDGE_PROVENANCE.get(self.kind, frozenset())
        )

    @property
    def edge_id(self) -> str:
        return "edge-" + _identity(
            {
                "schema": CODE_EVIDENCE_EDGE_SCHEMA,
                "source": self.source,
                "target": self.target,
                "kind": self.kind.value,
                "provenance": self.provenance.value,
                "provenance_record_id": self.provenance_record_id,
                "metadata": self.metadata,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_EVIDENCE_EDGE_SCHEMA,
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "provenance": self.provenance.value,
            "provenance_record_id": self.provenance_record_id,
            "authoritative": self.authoritative,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceEdge":
        schema = str(payload.get("schema") or CODE_EVIDENCE_EDGE_SCHEMA)
        if schema != CODE_EVIDENCE_EDGE_SCHEMA:
            raise EvidenceGraphValidationError(f"unsupported edge schema: {schema}")
        edge = cls(
            source=str(payload.get("source") or payload.get("source_node_id") or ""),
            target=str(payload.get("target") or payload.get("target_node_id") or ""),
            kind=payload.get("kind", payload.get("edge_kind", "")),
            provenance=payload.get("provenance", ""),
            provenance_record_id=str(payload.get("provenance_record_id") or ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = str(payload.get("edge_id") or "")
        if claimed and claimed != edge.edge_id:
            raise EvidenceGraphValidationError("edge identity does not match payload")
        if "authoritative" in payload and bool(payload["authoritative"]) != edge.authoritative:
            raise EvidenceGraphValidationError("edge authority does not match provenance")
        return edge


@dataclass(frozen=True)
class CodeEvidenceGraph:
    """Canonical node and provenance-edge set."""

    nodes: tuple[EvidenceNode, ...] = ()
    edges: tuple[ProvenanceEdge, ...] = ()

    def __post_init__(self) -> None:
        node_map: dict[str, EvidenceNode] = {}
        for value in self.nodes:
            node = value if isinstance(value, EvidenceNode) else EvidenceNode.from_dict(value)
            previous = node_map.get(node.node_id)
            if previous is not None and previous.to_dict() != node.to_dict():
                raise EvidenceGraphValidationError(
                    f"conflicting records for node {node.node_id}"
                )
            node_map[node.node_id] = node
        edge_map: dict[str, ProvenanceEdge] = {}
        for value in self.edges:
            edge = (
                value
                if isinstance(value, ProvenanceEdge)
                else ProvenanceEdge.from_dict(value)
            )
            if edge.source not in node_map or edge.target not in node_map:
                raise EvidenceGraphValidationError(
                    f"edge {edge.edge_id} references an unknown node"
                )
            edge_map[edge.edge_id] = edge
        object.__setattr__(
            self, "nodes", tuple(node_map[key] for key in sorted(node_map))
        )
        object.__setattr__(
            self, "edges", tuple(edge_map[key] for key in sorted(edge_map))
        )

    @property
    def graph_id(self) -> str:
        return "graph-" + _identity(self.canonical_records())

    def canonical_records(self) -> dict[str, list[dict[str, Any]]]:
        """Return the representation shared exactly by JSON and DuckDB."""

        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        records = self.canonical_records()
        return {
            "schema": CODE_EVIDENCE_GRAPH_SCHEMA,
            "graph_id": self.graph_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            **records,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(
            _canonical_value(self.to_dict()),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def nodes_by_kind(self, kind: EvidenceNodeKind | str) -> tuple[EvidenceNode, ...]:
        expected = _enum(kind, EvidenceNodeKind, "node kind")
        return tuple(node for node in self.nodes if node.kind is expected)

    def edges_by_kind(self, kind: EvidenceEdgeKind | str) -> tuple[ProvenanceEdge, ...]:
        expected = _enum(kind, EvidenceEdgeKind, "edge kind")
        return tuple(edge for edge in self.edges if edge.kind is expected)

    def find_nodes(
        self,
        *,
        kind: EvidenceNodeKind | str | None = None,
        task_id: str = "",
        tree_id: str = "",
        symbol: str = "",
        obligation_id: str = "",
    ) -> tuple[EvidenceNode, ...]:
        expected = _enum(kind, EvidenceNodeKind, "node kind") if kind else None
        return tuple(
            node
            for node in self.nodes
            if (expected is None or node.kind is expected)
            and (not task_id or node.task_id == task_id)
            and (not tree_id or node.tree_id == tree_id)
            and (not symbol or node.symbol == symbol)
            and (not obligation_id or node.obligation_id == obligation_id)
        )

    @classmethod
    def from_records(cls, **records: Any) -> "CodeEvidenceGraph":
        """Materialize a graph through the same trusted record channels as the builder."""

        return materialize_code_evidence_graph(**records)

    materialize = from_records

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeEvidenceGraph":
        schema = str(payload.get("schema") or CODE_EVIDENCE_GRAPH_SCHEMA)
        if schema != CODE_EVIDENCE_GRAPH_SCHEMA:
            raise EvidenceGraphValidationError(f"unsupported graph schema: {schema}")
        nodes = payload.get("nodes") or ()
        edges = payload.get("edges") or ()
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise EvidenceGraphValidationError("graph nodes must be a sequence")
        if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
            raise EvidenceGraphValidationError("graph edges must be a sequence")
        if not all(isinstance(item, Mapping) for item in (*nodes, *edges)):
            raise EvidenceGraphValidationError("graph records must be mappings")
        graph = cls(
            nodes=tuple(EvidenceNode.from_dict(item) for item in nodes),
            edges=tuple(ProvenanceEdge.from_dict(item) for item in edges),
        )
        claimed = str(payload.get("graph_id") or "")
        if claimed and claimed != graph.graph_id:
            raise EvidenceGraphValidationError("graph identity does not match payload")
        if "node_count" in payload and int(payload["node_count"]) != len(graph.nodes):
            raise EvidenceGraphValidationError("graph node_count does not match records")
        if "edge_count" in payload and int(payload["edge_count"]) != len(graph.edges):
            raise EvidenceGraphValidationError("graph edge_count does not match records")
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "CodeEvidenceGraph":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvidenceGraphValidationError("graph JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise EvidenceGraphValidationError("graph JSON must contain an object")
        return cls.from_dict(value)


CodeEvidenceNode = EvidenceNode
CodeEvidenceEdge = ProvenanceEdge
EvidenceGraph = CodeEvidenceGraph


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: dict[str, ProvenanceEdge] = {}
        self.by_key: dict[tuple[EvidenceNodeKind, str], str] = {}

    def node(
        self,
        kind: EvidenceNodeKind,
        key: str,
        provenance: EvidenceProvenance,
        record: Mapping[str, Any],
        **indexes: str,
    ) -> EvidenceNode:
        node = EvidenceNode(kind, key, provenance, record, **indexes)
        old = self.nodes.get(node.node_id)
        if old is not None and old.to_dict() != node.to_dict():
            raise EvidenceGraphValidationError(f"conflicting record key: {kind.value}:{key}")
        self.nodes[node.node_id] = node
        self.by_key[(kind, key)] = node.node_id
        return node

    def reference(
        self,
        kind: EvidenceNodeKind,
        key: str,
        provenance: EvidenceProvenance,
        **indexes: str,
    ) -> EvidenceNode:
        existing = self.by_key.get((kind, key))
        if existing:
            return self.nodes[existing]
        # Source contracts often use a content CID as their record key while
        # downstream records refer to the same entity by task_id, tree_id, or
        # obligation_id.  Resolve those aliases before creating a reference
        # placeholder so the graph remains one-node-per-entity.
        for candidate in self.nodes.values():
            if candidate.kind is not kind:
                continue
            aliases = {
                candidate.record_key,
                candidate.task_id,
                candidate.tree_id,
                candidate.symbol,
                candidate.obligation_id,
                _text(
                    candidate.record,
                    "task_id",
                    "canonical_task_id",
                    "task_cid",
                    "canonical_task_cid",
                    "scope_id",
                    "record_id",
                    "receipt_id",
                    "attempt_id",
                    "obligation_id",
                    "repository_tree_id",
                    "tree_id",
                ),
            }
            if key in aliases:
                self.by_key[(kind, key)] = candidate.node_id
                return candidate
        return self.node(kind, key, provenance, {"reference": True}, **indexes)

    def edge(
        self,
        source: EvidenceNode,
        target: EvidenceNode,
        kind: EvidenceEdgeKind,
        provenance: EvidenceProvenance,
        record_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        edge = ProvenanceEdge(
            source.node_id,
            target.node_id,
            kind,
            provenance,
            record_id,
            metadata or {},
        )
        self.edges[edge.edge_id] = edge


def _record_key(
    record: Mapping[str, Any], names: Sequence[str], kind: EvidenceNodeKind
) -> str:
    value = _text(record, *names)
    if value:
        return value
    return f"{kind.value}-{_identity(record)}"


def _task_id(record: Mapping[str, Any]) -> str:
    return _text(record, "task_id", "canonical_task_id", "task_cid", "canonical_task_cid", "id")


def _tree_id(record: Mapping[str, Any]) -> str:
    return _text(
        record,
        "repository_tree_id",
        "tree_id",
        "candidate_tree_id",
        "commit_sha",
        "tree_sha",
    )


def _successful(record: Mapping[str, Any]) -> bool:
    if record.get("success") is True or record.get("passed") is True:
        return True
    status = _text(record, "status", "result", "verdict", "outcome").lower()
    return status in {
        "passed",
        "pass",
        "success",
        "succeeded",
        "successful",
        "proved",
        "merged",
        "completed",
        "complete",
    }


def _freshness(record: Mapping[str, Any]) -> str:
    value = record.get("freshness")
    if isinstance(value, Mapping):
        status = _text(value, "status", "freshness")
        if status:
            return status
        if value.get("fresh") is True:
            return "current"
        if value.get("fresh") is False:
            return "stale"
        return "unknown"
    return str(value or "").strip()


def _add_tree(builder: _GraphBuilder, tree_id: str, provenance: EvidenceProvenance) -> EvidenceNode | None:
    if not tree_id:
        return None
    return builder.reference(
        EvidenceNodeKind.TREE, tree_id, provenance, tree_id=tree_id
    )


def _ingest_tasks(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    pending: list[tuple[EvidenceNode, dict[str, Any]]] = []
    for value in records:
        record = _record(value)
        task_id = _task_id(record)
        key = _record_key(
            record,
            ("canonical_task_cid", "task_cid", "task_id", "canonical_task_id", "id"),
            EvidenceNodeKind.TASK,
        )
        tree_id = _tree_id(record)
        node = builder.node(
            EvidenceNodeKind.TASK,
            key,
            EvidenceProvenance.TASK,
            record,
            task_id=task_id,
            tree_id=tree_id,
        )
        pending.append((node, record))
        tree = _add_tree(builder, tree_id, EvidenceProvenance.TASK)
        if tree:
            builder.edge(
                node, tree, EvidenceEdgeKind.TARGETS_TREE,
                EvidenceProvenance.TASK, key,
            )
    aliases = {
        alias: node
        for node, record in pending
        for alias in {
            node.record_key,
            _task_id(record),
            _text(record, "task_cid", "canonical_task_cid"),
        }
        if alias
    }
    for node, record in pending:
        dependencies: set[str] = set()
        for name in (
            "depends_on",
            "dependencies",
            "dependency_task_ids",
            "dependency_task_cids",
            "blocking_task_cids",
        ):
            dependencies.update(_strings(record.get(name)))
        for dependency in sorted(dependencies):
            target = aliases.get(dependency)
            if target is None:
                target = builder.reference(
                    EvidenceNodeKind.TASK,
                    dependency,
                    EvidenceProvenance.TASK,
                    task_id=dependency,
                )
                aliases[dependency] = target
            builder.edge(
                node,
                target,
                EvidenceEdgeKind.DEPENDS_ON,
                EvidenceProvenance.TASK,
                node.record_key,
            )


def _ingest_ast(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        kind_text = _text(record, "kind", "scope_kind", "scope_type").lower()
        symbol = _text(
            record, "qualified_symbol", "symbol", "qualified_name", "name", "value"
        )
        is_symbol = bool(symbol) and kind_text in {
            "qualified_symbol", "symbol", "function", "class", "method", "interface"
        }
        kind = EvidenceNodeKind.SYMBOL if is_symbol else EvidenceNodeKind.AST_SCOPE
        key = _record_key(
            record, ("scope_id", "record_id", "content_id", "ast_id"), kind
        )
        tree_id = _tree_id(record)
        task_id = _task_id(record)
        node = builder.node(
            kind,
            key,
            EvidenceProvenance.AST,
            record,
            task_id=task_id,
            tree_id=tree_id,
            symbol=symbol if is_symbol else "",
        )
        tree = _add_tree(builder, tree_id, EvidenceProvenance.AST)
        if tree:
            builder.edge(
                tree,
                node,
                EvidenceEdgeKind.DEFINES_SYMBOL if is_symbol else EvidenceEdgeKind.CONTAINS,
                EvidenceProvenance.AST,
                key,
            )
        # ASTBlobRecord is path-independent and contains a tuple of symbols
        # instead of representing one changed symbol per record.  Expand a
        # deterministic symbol projection while retaining the blob node as its
        # provenance parent.
        for qualified_symbol in _strings(record.get("qualified_symbols")):
            symbol_key = f"{key}#symbol:{qualified_symbol}"
            symbol_record = {
                "ast_record_id": key,
                "qualified_symbol": qualified_symbol,
                "source_sha256": record.get("source_sha256", ""),
                "symbol_hash": (
                    record.get("symbol_hashes", {}).get(qualified_symbol, "")
                    if isinstance(record.get("symbol_hashes"), Mapping)
                    else ""
                ),
            }
            symbol_node = builder.node(
                EvidenceNodeKind.SYMBOL,
                symbol_key,
                EvidenceProvenance.AST,
                symbol_record,
                task_id=task_id,
                tree_id=tree_id,
                symbol=qualified_symbol,
            )
            builder.edge(
                node,
                symbol_node,
                EvidenceEdgeKind.DEFINES_SYMBOL,
                EvidenceProvenance.AST,
                key,
            )


def _ingest_trees(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        tree_id = _tree_id(record) or _text(record, "content_id", "id")
        if not tree_id:
            raise EvidenceGraphValidationError("tree record requires a tree identity")
        builder.node(
            EvidenceNodeKind.TREE,
            tree_id,
            EvidenceProvenance.AST,
            record,
            tree_id=tree_id,
        )


def _ingest_obligations(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        obligation_id = _text(record, "obligation_id", "content_id", "id")
        key = obligation_id or f"obligation-{_identity(record)}"
        task_id = _task_id(record)
        tree_id = _tree_id(record)
        node = builder.node(
            EvidenceNodeKind.OBLIGATION,
            key,
            EvidenceProvenance.PROOF,
            record,
            task_id=task_id,
            tree_id=tree_id,
            obligation_id=key,
            assurance=_text(record, "required_assurance"),
        )
        tree = _add_tree(builder, tree_id, EvidenceProvenance.PROOF)
        if tree:
            builder.edge(
                node, tree, EvidenceEdgeKind.TARGETS_TREE,
                EvidenceProvenance.PROOF, key,
            )
        if task_id:
            task = builder.reference(
                EvidenceNodeKind.TASK,
                task_id,
                EvidenceProvenance.TASK,
                task_id=task_id,
            )
            builder.edge(
                task, node, EvidenceEdgeKind.HAS_OBLIGATION,
                EvidenceProvenance.PROOF, key,
            )
        for scope_id in _strings(record.get("ast_scope_ids")):
            scope = (
                builder.nodes[builder.by_key[(EvidenceNodeKind.SYMBOL, scope_id)]]
                if (EvidenceNodeKind.SYMBOL, scope_id) in builder.by_key
                else builder.reference(
                    EvidenceNodeKind.AST_SCOPE,
                    scope_id,
                    EvidenceProvenance.AST,
                    tree_id=tree_id,
                )
            )
            builder.edge(
                node, scope, EvidenceEdgeKind.COVERS,
                EvidenceProvenance.PROOF, key,
            )
        for dependency in _strings(record.get("premise_ids")):
            premise = builder.reference(
                EvidenceNodeKind.OBLIGATION,
                dependency,
                EvidenceProvenance.PROOF,
                obligation_id=dependency,
            )
            builder.edge(
                node, premise, EvidenceEdgeKind.DEPENDS_ON,
                EvidenceProvenance.PROOF, key,
            )


def _ingest_attempts(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        key = _record_key(
            record, ("attempt_id", "content_id", "id"), EvidenceNodeKind.ATTEMPT
        )
        obligation_id = _text(record, "obligation_id")
        node = builder.node(
            EvidenceNodeKind.ATTEMPT,
            key,
            EvidenceProvenance.PROOF,
            record,
            task_id=_task_id(record),
            tree_id=_tree_id(record),
            obligation_id=obligation_id,
            assurance=_text(record, "authoritative_assurance", "assurance"),
            freshness=_text(record, "freshness"),
        )
        if obligation_id:
            obligation = builder.reference(
                EvidenceNodeKind.OBLIGATION,
                obligation_id,
                EvidenceProvenance.PROOF,
                obligation_id=obligation_id,
            )
            builder.edge(
                node, obligation, EvidenceEdgeKind.ATTEMPT_FOR,
                EvidenceProvenance.PROOF, key,
            )


def _ingest_proofs(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        key = _record_key(
            record,
            ("receipt_id", "proof_id", "content_id", "artifact_id", "id"),
            EvidenceNodeKind.PROOF,
        )
        obligation_id = _text(record, "obligation_id", "subject_id")
        assurance = _text(
            record, "authoritative_assurance", "assurance", "level"
        )
        freshness = _freshness(record)
        node = builder.node(
            EvidenceNodeKind.PROOF,
            key,
            EvidenceProvenance.PROOF,
            record,
            task_id=_task_id(record),
            tree_id=_tree_id(record),
            obligation_id=obligation_id,
            assurance=assurance,
            freshness=freshness,
        )
        if obligation_id:
            obligation = builder.reference(
                EvidenceNodeKind.OBLIGATION,
                obligation_id,
                EvidenceProvenance.PROOF,
                obligation_id=obligation_id,
            )
            verdict = _text(record, "verdict", "status", "result").lower()
            authoritative_proof = (
                verdict == "proved"
                and freshness in {"", "current", "fresh"}
                and assurance in {"solver_checked", "kernel_verified", "attested"}
            )
            if authoritative_proof:
                builder.edge(
                    node, obligation, EvidenceEdgeKind.PROVES,
                    EvidenceProvenance.PROOF, key,
                    {"assurance": assurance, "freshness": freshness},
                )
            else:
                builder.edge(
                    node, obligation, EvidenceEdgeKind.DERIVED_FROM,
                    EvidenceProvenance.PROOF, key,
                    {"verdict": verdict or "unknown"},
                )
        attempt_id = _text(record, "attempt_id")
        if attempt_id:
            attempt = builder.reference(
                EvidenceNodeKind.ATTEMPT, attempt_id, EvidenceProvenance.PROOF
            )
            builder.edge(
                node, attempt, EvidenceEdgeKind.DERIVED_FROM,
                EvidenceProvenance.PROOF, key,
            )
        evidence_values = record.get("evidence")
        if isinstance(evidence_values, Sequence) and not isinstance(
            evidence_values, (str, bytes, bytearray)
        ):
            for raw_evidence in evidence_values:
                if not isinstance(raw_evidence, Mapping):
                    continue
                evidence = _record(raw_evidence)
                evidence_key = _record_key(
                    evidence,
                    ("evidence_id", "content_id", "artifact_id", "id"),
                    EvidenceNodeKind.EVIDENCE,
                )
                evidence_node = builder.node(
                    EvidenceNodeKind.EVIDENCE,
                    evidence_key,
                    EvidenceProvenance.PROOF,
                    evidence,
                    tree_id=_tree_id(record),
                    obligation_id=obligation_id,
                    freshness=_freshness(evidence),
                )
                builder.edge(
                    node,
                    evidence_node,
                    EvidenceEdgeKind.DERIVED_FROM,
                    EvidenceProvenance.PROOF,
                    key,
                )


def _ingest_validations(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        key = _record_key(
            record,
            ("validation_receipt_id", "receipt_id", "provenance_cid", "content_id", "id"),
            EvidenceNodeKind.VALIDATION,
        )
        task_id = _task_id(record)
        node = builder.node(
            EvidenceNodeKind.VALIDATION,
            key,
            EvidenceProvenance.VALIDATION,
            record,
            task_id=task_id,
            tree_id=_tree_id(record),
            freshness=_freshness(record),
        )
        if task_id:
            task = builder.reference(
                EvidenceNodeKind.TASK,
                task_id,
                EvidenceProvenance.TASK,
                task_id=task_id,
            )
            if _successful(record):
                builder.edge(
                    node, task, EvidenceEdgeKind.VALIDATES,
                    EvidenceProvenance.VALIDATION, key,
                )
        for obligation_id in _strings(
            record.get("obligation_ids") or record.get("covered_obligation_ids")
        ):
            obligation = builder.reference(
                EvidenceNodeKind.OBLIGATION,
                obligation_id,
                EvidenceProvenance.PROOF,
                obligation_id=obligation_id,
            )
            if _successful(record):
                builder.edge(
                    node, obligation, EvidenceEdgeKind.COVERS,
                    EvidenceProvenance.VALIDATION, key,
                )


def _ingest_merges(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    for value in records:
        record = _record(value)
        key = _record_key(
            record,
            ("merge_receipt_id", "receipt_cid", "provenance_cid", "content_id", "id"),
            EvidenceNodeKind.MERGE,
        )
        task_id = _task_id(record)
        tree_id = _tree_id(record)
        node = builder.node(
            EvidenceNodeKind.MERGE,
            key,
            EvidenceProvenance.MERGE,
            record,
            task_id=task_id,
            tree_id=tree_id,
        )
        if task_id and _successful(record):
            task = builder.reference(
                EvidenceNodeKind.TASK,
                task_id,
                EvidenceProvenance.TASK,
                task_id=task_id,
            )
            builder.edge(
                node, task, EvidenceEdgeKind.MERGED,
                EvidenceProvenance.MERGE, key,
            )
            if _text(record, "completion_status", "task_status").lower() in {
                "complete", "completed", "done"
            }:
                builder.edge(
                    node, task, EvidenceEdgeKind.COMPLETES,
                    EvidenceProvenance.MERGE, key,
                )
        tree = _add_tree(builder, tree_id, EvidenceProvenance.MERGE)
        if tree:
            builder.edge(
                node, tree, EvidenceEdgeKind.DERIVED_FROM,
                EvidenceProvenance.MERGE, key,
            )


def _ingest_enrichments(builder: _GraphBuilder, records: Iterable[Any]) -> None:
    known_by_external = {
        external: node
        for node in builder.nodes.values()
        for external in (
            node.node_id,
            node.record_key,
            node.task_id,
            node.tree_id,
            node.symbol,
            node.obligation_id,
        )
        if external
    }
    for value in records:
        record = _record(value)
        requested = _text(record, "edge_kind", "kind").lower()
        if requested and requested not in {item.value for item in ENRICHMENT_EDGE_KINDS}:
            raise EvidenceGraphValidationError(
                f"enrichment cannot create {requested!r} edges"
            )
        key = _record_key(
            record, ("enrichment_id", "content_id", "id"), EvidenceNodeKind.ENRICHMENT
        )
        source = _text(record, "provenance", "source", "authority").lower()
        provenance = (
            EvidenceProvenance.GRAPHRAG
            if "graphrag" in source
            else EvidenceProvenance.LLM
            if source in {"llm", "model", "language_model"} or "llm" in source
            else EvidenceProvenance.ENRICHMENT
        )
        node = builder.node(
            EvidenceNodeKind.ENRICHMENT,
            key,
            provenance,
            record,
        )
        edge_kind = EvidenceEdgeKind(requested or EvidenceEdgeKind.RELATED_TO.value)
        targets = set()
        for name in ("target", "target_id", "targets", "target_ids", "mentions"):
            targets.update(_strings(record.get(name)))
        for target_id in sorted(targets):
            target = known_by_external.get(target_id)
            if target is None:
                continue
            builder.edge(
                node, target, edge_kind,
                provenance, key,
            )


def materialize_code_evidence_graph(
    *,
    ast_records: Iterable[Any] = (),
    tree_records: Iterable[Any] = (),
    repository_trees: Iterable[Any] = (),
    task_records: Iterable[Any] = (),
    tasks: Iterable[Any] = (),
    obligations: Iterable[Any] = (),
    proof_obligations: Iterable[Any] = (),
    attempts: Iterable[Any] = (),
    proof_attempts: Iterable[Any] = (),
    proof_records: Iterable[Any] = (),
    proof_receipts: Iterable[Any] = (),
    validation_records: Iterable[Any] = (),
    validation_receipts: Iterable[Any] = (),
    merge_records: Iterable[Any] = (),
    merge_receipts: Iterable[Any] = (),
    enrichments: Iterable[Any] = (),
) -> CodeEvidenceGraph:
    """Build a deterministic graph from authoritative record channels.

    Duplicate aliases are accepted for ergonomic integration, and all inputs
    are sorted by their derived identities in the resulting graph.
    """

    builder = _GraphBuilder()
    _ingest_trees(builder, (*tuple(tree_records), *tuple(repository_trees)))
    _ingest_tasks(builder, (*tuple(task_records), *tuple(tasks)))
    _ingest_ast(builder, ast_records)
    _ingest_obligations(builder, (*tuple(obligations), *tuple(proof_obligations)))
    _ingest_attempts(builder, (*tuple(attempts), *tuple(proof_attempts)))
    _ingest_proofs(builder, (*tuple(proof_records), *tuple(proof_receipts)))
    _ingest_validations(
        builder, (*tuple(validation_records), *tuple(validation_receipts))
    )
    _ingest_merges(builder, (*tuple(merge_records), *tuple(merge_receipts)))
    _ingest_enrichments(builder, enrichments)
    return CodeEvidenceGraph(
        nodes=tuple(builder.nodes.values()), edges=tuple(builder.edges.values())
    )


build_code_evidence_graph = materialize_code_evidence_graph


def canonical_graph_records(value: CodeEvidenceGraph | Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize a graph or graph mapping to canonical node and edge records."""

    graph = value if isinstance(value, CodeEvidenceGraph) else CodeEvidenceGraph.from_dict(value)
    return graph.canonical_records()


def _impact_path(value: Any) -> str:
    """Return a safe canonical repository-relative path.

    Impact evidence is persisted and later used to select commands.  It must
    therefore reject absolute and escaping paths instead of silently resolving
    them against the scheduler's current directory.
    """

    raw = str(value or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "\0" in raw:
        raise EvidenceGraphValidationError(
            f"unsafe repository impact path: {value!r}"
        )
    return path.as_posix()


def _impact_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else value
    try:
        return tuple(
            sorted(
                {
                    str(item).strip()
                    for item in values
                    if str(item).strip()
                }
            )
        )
    except TypeError as exc:
        raise EvidenceGraphValidationError(
            "impact graph collections must be iterable"
        ) from exc


@dataclass(frozen=True)
class ChangedASTSymbol:
    """One exact symbol changed by a candidate patch.

    ``interface_changed`` is explicit evidence from the before/after AST
    comparison.  A public-looking name is not enough to infer that a signature
    changed.
    """

    symbol: str
    path: str
    change_kind: str = "modified"
    interface_changed: bool = False

    def __post_init__(self) -> None:
        symbol = str(self.symbol or "").strip()
        if not symbol:
            raise EvidenceGraphValidationError("changed AST symbol is required")
        kind = str(self.change_kind or "modified").strip().lower()
        if kind not in {"added", "modified", "deleted", "renamed"}:
            raise EvidenceGraphValidationError(
                f"unsupported AST symbol change kind: {kind!r}"
            )
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "path", _impact_path(self.path))
        object.__setattr__(self, "change_kind", kind)
        object.__setattr__(self, "interface_changed", bool(self.interface_changed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "path": self.path,
            "change_kind": self.change_kind,
            "interface_changed": self.interface_changed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangedASTSymbol":
        return cls(
            symbol=str(
                value.get("symbol")
                or value.get("qualified_symbol")
                or value.get("qualified_name")
                or ""
            ),
            path=str(
                value.get("path")
                or value.get("new_path")
                or value.get("old_path")
                or ""
            ),
            change_kind=str(value.get("change_kind") or "modified"),
            interface_changed=bool(value.get("interface_changed", False)),
        )


@dataclass(frozen=True)
class CodeImpactResult:
    """Complete direct and transitive closure for one candidate change."""

    repository_tree_id: str
    index_id: str
    changed_symbols: tuple[str, ...]
    affected_symbols: tuple[str, ...]
    changed_paths: tuple[str, ...]
    affected_paths: tuple[str, ...]
    dependency_chains: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_validation_ids: tuple[str, ...] = ()
    validation_reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    uncovered_symbols: tuple[str, ...] = ()
    uncovered_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("repository_tree_id", "index_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise EvidenceGraphValidationError(
                    f"code impact result requires {name}"
                )
            object.__setattr__(self, name, value)
        for name in (
            "changed_symbols",
            "affected_symbols",
            "required_validation_ids",
            "uncovered_symbols",
        ):
            object.__setattr__(
                self, name, _impact_strings(getattr(self, name))
            )
        for name in ("changed_paths", "affected_paths", "uncovered_paths"):
            object.__setattr__(
                self,
                name,
                tuple(
                    sorted(
                        {
                            _impact_path(value)
                            for value in getattr(self, name)
                        }
                    )
                ),
            )
        object.__setattr__(
            self,
            "dependency_chains",
            {
                str(key): tuple(
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                )
                for key, value in sorted(
                    dict(self.dependency_chains or {}).items()
                )
            },
        )
        object.__setattr__(
            self,
            "validation_reasons",
            {
                str(key): _impact_strings(value)
                for key, value in sorted(
                    dict(self.validation_reasons or {}).items()
                )
            },
        )

    @property
    def uncovered_impact(self) -> bool:
        return bool(self.uncovered_symbols or self.uncovered_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_IMPACT_RESULT_SCHEMA,
            "repository_tree_id": self.repository_tree_id,
            "index_id": self.index_id,
            "changed_symbols": list(self.changed_symbols),
            "affected_symbols": list(self.affected_symbols),
            "changed_paths": list(self.changed_paths),
            "affected_paths": list(self.affected_paths),
            "dependency_chains": {
                key: list(value)
                for key, value in sorted(self.dependency_chains.items())
            },
            "required_validation_ids": list(self.required_validation_ids),
            "validation_reasons": {
                key: list(value)
                for key, value in sorted(self.validation_reasons.items())
            },
            "uncovered_symbols": list(self.uncovered_symbols),
            "uncovered_paths": list(self.uncovered_paths),
            "uncovered_impact": self.uncovered_impact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeImpactResult":
        schema = str(value.get("schema") or CODE_IMPACT_RESULT_SCHEMA)
        if schema != CODE_IMPACT_RESULT_SCHEMA:
            raise EvidenceGraphValidationError(
                f"unsupported code impact result schema: {schema}"
            )
        return cls(
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            index_id=str(value.get("index_id") or ""),
            changed_symbols=tuple(value.get("changed_symbols") or ()),
            affected_symbols=tuple(value.get("affected_symbols") or ()),
            changed_paths=tuple(value.get("changed_paths") or ()),
            affected_paths=tuple(value.get("affected_paths") or ()),
            dependency_chains={
                str(key): (
                    (items,) if isinstance(items, str) else tuple(items)
                )
                for key, items in dict(
                    value.get("dependency_chains") or {}
                ).items()
            },
            required_validation_ids=tuple(
                value.get("required_validation_ids") or ()
            ),
            validation_reasons={
                str(key): (
                    (items,) if isinstance(items, str) else tuple(items)
                )
                for key, items in dict(
                    value.get("validation_reasons") or {}
                ).items()
            },
            uncovered_symbols=tuple(
                value.get("uncovered_symbols") or ()
            ),
            uncovered_paths=tuple(value.get("uncovered_paths") or ()),
        )


@dataclass(frozen=True)
class CodeImpactIndex:
    """Tree-bound symbol and path dependency evidence used for selection.

    Dependency mappings use ``dependent -> providers`` orientation.  Impact
    selection walks the reverse edges, so changing a provider selects consumers
    and validations even when their files are outside the direct patch.
    """

    repository_tree_id: str
    symbol_paths: Mapping[str, str]
    symbol_dependencies: Mapping[str, Sequence[str]] = field(default_factory=dict)
    path_dependencies: Mapping[str, Sequence[str]] = field(default_factory=dict)
    validation_targets: Mapping[str, Sequence[str]] = field(default_factory=dict)
    index_version: str = "code-impact-index-v1"
    index_id: str = ""

    def __post_init__(self) -> None:
        tree_id = str(self.repository_tree_id or "").strip()
        if not tree_id:
            raise EvidenceGraphValidationError(
                "code impact index requires repository_tree_id"
            )
        object.__setattr__(self, "repository_tree_id", tree_id)

        symbol_paths: dict[str, str] = {}
        for raw_symbol, raw_path in dict(self.symbol_paths or {}).items():
            symbol = str(raw_symbol or "").strip()
            if not symbol:
                raise EvidenceGraphValidationError(
                    "code impact index contains an empty symbol"
                )
            symbol_paths[symbol] = _impact_path(raw_path)
        object.__setattr__(
            self, "symbol_paths", dict(sorted(symbol_paths.items()))
        )

        symbol_dependencies: dict[str, tuple[str, ...]] = {}
        known_symbols = set(symbol_paths)
        for raw_dependent, raw_dependencies in dict(
            self.symbol_dependencies or {}
        ).items():
            dependent = str(raw_dependent or "").strip()
            dependencies = _impact_strings(raw_dependencies)
            unknown = ({dependent, *dependencies} - known_symbols)
            if unknown:
                raise EvidenceGraphValidationError(
                    "symbol dependency references unknown symbols: "
                    + ", ".join(sorted(unknown))
                )
            if dependent in dependencies:
                raise EvidenceGraphValidationError(
                    "symbol dependency cannot reference itself"
                )
            symbol_dependencies[dependent] = dependencies
        object.__setattr__(
            self,
            "symbol_dependencies",
            dict(sorted(symbol_dependencies.items())),
        )

        path_dependencies: dict[str, tuple[str, ...]] = {}
        for raw_dependent, raw_dependencies in dict(
            self.path_dependencies or {}
        ).items():
            dependent = _impact_path(raw_dependent)
            dependencies = tuple(
                sorted(
                    {
                        _impact_path(value)
                        for value in (
                            (raw_dependencies,)
                            if isinstance(raw_dependencies, str)
                            else raw_dependencies
                        )
                    }
                )
            )
            if dependent in dependencies:
                raise EvidenceGraphValidationError(
                    "path dependency cannot reference itself"
                )
            path_dependencies[dependent] = dependencies
        object.__setattr__(
            self, "path_dependencies", dict(sorted(path_dependencies.items()))
        )

        targets: dict[str, tuple[str, ...]] = {}
        known_paths = set(symbol_paths.values())
        known_paths.update(path_dependencies)
        for dependencies in path_dependencies.values():
            known_paths.update(dependencies)
        known_targets = known_symbols | known_paths
        for raw_validation_id, raw_targets in dict(
            self.validation_targets or {}
        ).items():
            validation_id = str(raw_validation_id or "").strip()
            if not validation_id:
                raise EvidenceGraphValidationError(
                    "validation target requires an identity"
                )
            values = _impact_strings(raw_targets)
            if not values:
                raise EvidenceGraphValidationError(
                    f"validation {validation_id!r} has no impact targets"
                )
            unknown = set(values) - known_targets
            if unknown:
                raise EvidenceGraphValidationError(
                    "validation references unknown impact targets: "
                    + ", ".join(sorted(unknown))
                )
            targets[validation_id] = values
        object.__setattr__(
            self, "validation_targets", dict(sorted(targets.items()))
        )

        version = str(self.index_version or "").strip()
        if not version:
            raise EvidenceGraphValidationError(
                "code impact index version is required"
            )
        object.__setattr__(self, "index_version", version)
        claimed = str(self.index_id or "").strip()
        object.__setattr__(self, "index_id", "")
        actual = _identity(self._identity_payload())
        if claimed and claimed != actual:
            raise EvidenceGraphValidationError(
                "code impact index identity does not match payload"
            )
        object.__setattr__(self, "index_id", actual)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CODE_IMPACT_INDEX_SCHEMA,
            "repository_tree_id": self.repository_tree_id,
            "index_version": self.index_version,
            "symbol_paths": self.symbol_paths,
            "symbol_dependencies": self.symbol_dependencies,
            "path_dependencies": self.path_dependencies,
            "validation_targets": self.validation_targets,
        }

    @staticmethod
    def _reverse(
        dependencies: Mapping[str, Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        reverse: dict[str, set[str]] = {}
        for dependent, providers in dependencies.items():
            reverse.setdefault(dependent, set())
            for provider in providers:
                reverse.setdefault(provider, set()).add(dependent)
        return {
            key: tuple(sorted(value))
            for key, value in sorted(reverse.items())
        }

    @staticmethod
    def _closure_with_chains(
        roots: Iterable[str],
        reverse: Mapping[str, Sequence[str]],
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        from collections import deque

        normalized_roots = tuple(sorted(set(roots)))
        chains = {root: (root,) for root in normalized_roots}
        queue = deque(normalized_roots)
        while queue:
            current = queue.popleft()
            for dependent in reverse.get(current, ()):
                candidate = (*chains[current], dependent)
                existing = chains.get(dependent)
                if existing is None or (len(candidate), candidate) < (
                    len(existing),
                    existing,
                ):
                    chains[dependent] = candidate
                    queue.append(dependent)
        return tuple(sorted(chains)), chains

    def impact(
        self,
        *,
        changed_symbols: Iterable[str | ChangedASTSymbol | Mapping[str, Any]] = (),
        changed_paths: Iterable[str] = (),
    ) -> CodeImpactResult:
        """Compute a deterministic, explainable reverse dependency closure."""

        explicit_symbols: set[str] = set()
        explicit_paths: set[str] = set()
        for value in changed_symbols:
            if isinstance(value, ChangedASTSymbol):
                change = value
                explicit_symbols.add(change.symbol)
                explicit_paths.add(change.path)
            elif isinstance(value, Mapping):
                change = ChangedASTSymbol.from_dict(value)
                explicit_symbols.add(change.symbol)
                explicit_paths.add(change.path)
            else:
                symbol = str(value or "").strip()
                if symbol:
                    explicit_symbols.add(symbol)
        explicit_paths.update(_impact_path(value) for value in changed_paths)

        inferred_symbols = {
            symbol
            for symbol, path in self.symbol_paths.items()
            if path in explicit_paths
        }
        known_changed_symbols = (explicit_symbols | inferred_symbols) & set(
            self.symbol_paths
        )
        uncovered_symbols = tuple(
            sorted(explicit_symbols - set(self.symbol_paths))
        )
        known_paths = set(self.symbol_paths.values())
        known_paths.update(self.path_dependencies)
        for dependencies in self.path_dependencies.values():
            known_paths.update(dependencies)
        uncovered_paths = tuple(sorted(explicit_paths - known_paths))

        affected_symbols, symbol_chains = self._closure_with_chains(
            known_changed_symbols,
            self._reverse(self.symbol_dependencies),
        )
        symbol_affected_paths = {
            self.symbol_paths[symbol] for symbol in affected_symbols
        }
        path_roots = explicit_paths | symbol_affected_paths
        affected_paths, path_chains = self._closure_with_chains(
            path_roots,
            self._reverse(self.path_dependencies),
        )

        impacted_targets = set(affected_symbols) | set(affected_paths)
        validation_reasons = {
            validation_id: tuple(
                sorted(impacted_targets.intersection(targets))
            )
            for validation_id, targets in self.validation_targets.items()
            if impacted_targets.intersection(targets)
        }
        chains = dict(symbol_chains)
        for target, chain in path_chains.items():
            chains.setdefault(target, chain)
        return CodeImpactResult(
            repository_tree_id=self.repository_tree_id,
            index_id=self.index_id,
            changed_symbols=tuple(sorted(explicit_symbols | inferred_symbols)),
            affected_symbols=affected_symbols,
            changed_paths=tuple(sorted(explicit_paths)),
            affected_paths=affected_paths,
            dependency_chains=chains,
            required_validation_ids=tuple(sorted(validation_reasons)),
            validation_reasons=validation_reasons,
            uncovered_symbols=uncovered_symbols,
            uncovered_paths=uncovered_paths,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "index_id": self.index_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CodeImpactIndex":
        schema = str(value.get("schema") or CODE_IMPACT_INDEX_SCHEMA)
        if schema != CODE_IMPACT_INDEX_SCHEMA:
            raise EvidenceGraphValidationError(
                f"unsupported code impact index schema: {schema}"
            )
        return cls(
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            symbol_paths=value.get("symbol_paths") or {},
            symbol_dependencies=value.get("symbol_dependencies") or {},
            path_dependencies=value.get("path_dependencies") or {},
            validation_targets=value.get("validation_targets") or {},
            index_version=str(
                value.get("index_version") or "code-impact-index-v1"
            ),
            index_id=str(value.get("index_id") or ""),
        )

    @classmethod
    def from_ast_records(
        cls,
        *,
        repository_tree_id: str,
        ast_records: Iterable[Any],
        symbol_dependencies: Mapping[str, Sequence[str]] | None = None,
        path_dependencies: Mapping[str, Sequence[str]] | None = None,
        validation_targets: Mapping[str, Sequence[str]] | None = None,
    ) -> "CodeImpactIndex":
        """Build an index from canonical AST records plus reviewed edges.

        Records may expose either one ``qualified_symbol`` or the canonical
        ``qualified_symbols`` collection.  Explicit dependency facts on a
        record are merged with the reviewed arguments; no probabilistic
        relationship is admitted into the authority-bearing index.
        """

        symbols: dict[str, str] = {}
        dependencies: dict[str, set[str]] = {
            str(key): set(_impact_strings(value))
            for key, value in dict(symbol_dependencies or {}).items()
        }
        paths: dict[str, set[str]] = {
            _impact_path(key): {
                _impact_path(item)
                for item in ((value,) if isinstance(value, str) else value)
            }
            for key, value in dict(path_dependencies or {}).items()
        }
        for raw_record in ast_records:
            record = _record(raw_record)
            nested_ast = record.get("ast_record")
            if isinstance(nested_ast, Mapping):
                # ``IndexedASTPath`` deliberately keeps the reusable AST blob
                # path-independent.  Merge its facts with the authoritative
                # path wrapper for impact indexing.
                record = {**dict(nested_ast), **record}
            path_value = (
                record.get("path")
                or record.get("root_relative_path")
                or record.get("new_path")
                or record.get("file")
            )
            if not path_value:
                raise EvidenceGraphValidationError(
                    "AST impact records require a repository path"
                )
            path = _impact_path(path_value)
            record_symbols = set(_strings(record.get("qualified_symbols")))
            singular = _text(
                record,
                "qualified_symbol",
                "symbol",
                "qualified_name",
            )
            if singular:
                record_symbols.add(singular)
            for symbol in record_symbols:
                previous = symbols.get(symbol)
                if previous is not None and previous != path:
                    raise EvidenceGraphValidationError(
                        f"symbol {symbol!r} is defined by multiple paths"
                    )
                symbols[symbol] = path
            raw_symbol_dependencies = record.get("symbol_dependencies") or {}
            if isinstance(raw_symbol_dependencies, Mapping):
                for dependent, providers in raw_symbol_dependencies.items():
                    dependencies.setdefault(str(dependent), set()).update(
                        _impact_strings(providers)
                    )
            raw_path_dependencies = record.get("path_dependencies") or ()
            paths.setdefault(path, set()).update(
                _impact_path(item)
                for item in (
                    (raw_path_dependencies,)
                    if isinstance(raw_path_dependencies, str)
                    else raw_path_dependencies
                )
            )
        return cls(
            repository_tree_id=repository_tree_id,
            symbol_paths=symbols,
            symbol_dependencies={
                key: tuple(sorted(value))
                for key, value in dependencies.items()
            },
            path_dependencies={
                key: tuple(sorted(value)) for key, value in paths.items()
            },
            validation_targets=validation_targets or {},
        )


def build_code_impact_index(
    *,
    repository_tree_id: str,
    ast_records: Iterable[Any],
    symbol_dependencies: Mapping[str, Sequence[str]] | None = None,
    path_dependencies: Mapping[str, Sequence[str]] | None = None,
    validation_targets: Mapping[str, Sequence[str]] | None = None,
) -> CodeImpactIndex:
    """Compatibility-friendly functional constructor for impact evidence."""

    return CodeImpactIndex.from_ast_records(
        repository_tree_id=repository_tree_id,
        ast_records=ast_records,
        symbol_dependencies=symbol_dependencies,
        path_dependencies=path_dependencies,
        validation_targets=validation_targets,
    )


__all__ = [
    "AUTHORITATIVE_EDGE_PROVENANCE",
    "CODE_EVIDENCE_EDGE_SCHEMA",
    "CODE_EVIDENCE_GRAPH_SCHEMA",
    "CODE_EVIDENCE_NODE_SCHEMA",
    "CODE_IMPACT_INDEX_SCHEMA",
    "CODE_IMPACT_RESULT_SCHEMA",
    "ENRICHMENT_EDGE_KINDS",
    "UNTRUSTED_PROVENANCE",
    "CodeEvidenceEdge",
    "CodeEvidenceGraph",
    "CodeEvidenceNode",
    "ChangedASTSymbol",
    "CodeImpactIndex",
    "CodeImpactResult",
    "EvidenceEdgeKind",
    "EvidenceGraph",
    "EvidenceGraphValidationError",
    "EvidenceNode",
    "EvidenceNodeKind",
    "EvidenceProvenance",
    "ProvenanceEdge",
    "build_code_evidence_graph",
    "build_code_impact_index",
    "canonical_graph_records",
    "canonical_json",
    "materialize_code_evidence_graph",
]
