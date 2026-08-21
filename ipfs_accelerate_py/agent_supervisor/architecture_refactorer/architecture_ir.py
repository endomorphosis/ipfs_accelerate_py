"""Canonical ArchitectureIR graph values with strict content identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .contracts import (
    ARCHITECTURE_IR_SCHEMA,
    ARCHITECTURE_IR_VERSION,
    ArchitectureContractError,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    _closed_enum,
    _require_exact_fields,
    _require_mapping,
    _require_text,
    _require_int,
)

_NODE_FIELDS = frozenset({"node_id", "kind", "provenance", "content_identity"})
_EDGE_FIELDS = frozenset(
    {"edge_id", "kind", "source", "target", "provenance", "content_identity"}
)
_GRAPH_FIELDS = frozenset(
    {
        "schema",
        "version",
        "repository_tree",
        "freshness",
        "nodes",
        "edges",
        "content_identity",
    }
)


class ArchitectureIRError(ArchitectureContractError):
    """Fail-closed ArchitectureIR graph error."""


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise ArchitectureIRError("content identity must be a dag-json CIDv1") from exc


@dataclass(frozen=True)
class ArchitectureNode:
    """One closed ArchitectureIR node bound to source-fact provenance."""

    node_id: str
    kind: NodeKind
    provenance: SourceFactIdentity
    content_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            _require_text(self.node_id, "node_id", error_type=ArchitectureIRError),
        )
        object.__setattr__(
            self,
            "kind",
            _closed_enum(self.kind, NodeKind, "node kind", error_type=ArchitectureIRError),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ArchitectureIRError,
                )
            )
            if claimed != identity:
                raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "node_id": self.node_id,
            "provenance": self.provenance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureNode":
        mapping = _require_mapping(payload, error_type=ArchitectureIRError)
        try:
            _require_exact_fields(mapping, _NODE_FIELDS, error_type=ArchitectureIRError)
            provenance_payload = mapping["provenance"]
            if not isinstance(provenance_payload, Mapping):
                raise ArchitectureIRError("node provenance must be an object")
            node = cls(
                node_id=mapping["node_id"],
                kind=mapping["kind"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
            )
            if mapping["content_identity"] != node.content_identity:
                raise ArchitectureIRError("ArchitectureIR content identity mismatch")
            return node
        except ArchitectureContractError as exc:
            if isinstance(exc, ArchitectureIRError):
                raise
            raise ArchitectureIRError(str(exc)) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class ArchitectureEdge:
    """One closed ArchitectureIR edge bound to source-fact provenance."""

    edge_id: str
    kind: EdgeKind
    source: str
    target: str
    provenance: SourceFactIdentity
    content_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "edge_id",
            _require_text(self.edge_id, "edge_id", error_type=ArchitectureIRError),
        )
        object.__setattr__(
            self,
            "kind",
            _closed_enum(self.kind, EdgeKind, "edge kind", error_type=ArchitectureIRError),
        )
        object.__setattr__(
            self,
            "source",
            _require_text(self.source, "source", error_type=ArchitectureIRError),
        )
        object.__setattr__(
            self,
            "target",
            _require_text(self.target, "target", error_type=ArchitectureIRError),
        )
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        object.__setattr__(self, "provenance", provenance)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ArchitectureIRError,
                )
            )
            if claimed != identity:
                raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind.value,
            "provenance": self.provenance.to_dict(),
            "source": self.source,
            "target": self.target,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureEdge":
        mapping = _require_mapping(payload, error_type=ArchitectureIRError)
        try:
            _require_exact_fields(mapping, _EDGE_FIELDS, error_type=ArchitectureIRError)
            provenance_payload = mapping["provenance"]
            if not isinstance(provenance_payload, Mapping):
                raise ArchitectureIRError("edge provenance must be an object")
            edge = cls(
                edge_id=mapping["edge_id"],
                kind=mapping["kind"],
                source=mapping["source"],
                target=mapping["target"],
                provenance=SourceFactIdentity.from_mapping(provenance_payload),
            )
            if mapping["content_identity"] != edge.content_identity:
                raise ArchitectureIRError("ArchitectureIR content identity mismatch")
            return edge
        except ArchitectureContractError as exc:
            if isinstance(exc, ArchitectureIRError):
                raise
            raise ArchitectureIRError(str(exc)) from exc

    from_dict = from_mapping


@dataclass(frozen=True)
class ArchitectureIR:
    """Canonical closed architecture graph for one repository tree."""

    schema: str
    version: int
    repository_tree: str
    freshness: str
    nodes: tuple[ArchitectureNode, ...]
    edges: tuple[ArchitectureEdge, ...]
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=ArchitectureIRError)
        if schema != ARCHITECTURE_IR_SCHEMA:
            raise ArchitectureIRError("unexpected ArchitectureIR schema")
        version = _require_int(self.version, "version", error_type=ArchitectureIRError)
        if version != ARCHITECTURE_IR_VERSION:
            raise ArchitectureIRError("unexpected ArchitectureIR version")
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=ArchitectureIRError
        )
        freshness = _require_text(self.freshness, "freshness", error_type=ArchitectureIRError)
        if isinstance(self.nodes, (str, bytes, bytearray)) or not isinstance(self.nodes, Sequence):
            raise ArchitectureIRError("nodes must be a sequence")
        if isinstance(self.edges, (str, bytes, bytearray)) or not isinstance(self.edges, Sequence):
            raise ArchitectureIRError("edges must be a sequence")
        nodes = tuple(
            node if isinstance(node, ArchitectureNode) else ArchitectureNode.from_mapping(node)
            for node in self.nodes
        )
        edges = tuple(
            edge if isinstance(edge, ArchitectureEdge) else ArchitectureEdge.from_mapping(edge)
            for edge in self.edges
        )
        ordered_nodes = tuple(sorted(nodes, key=lambda item: item.node_id))
        ordered_edges = tuple(sorted(edges, key=lambda item: item.edge_id))
        node_ids = tuple(node.node_id for node in ordered_nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ArchitectureIRError("ArchitectureIR node ids must be unique")
        edge_ids = tuple(edge.edge_id for edge in ordered_edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ArchitectureIRError("ArchitectureIR edge ids must be unique")
        known = set(node_ids)
        for edge in ordered_edges:
            if edge.source not in known or edge.target not in known:
                raise ArchitectureIRError("ArchitectureIR edge references unknown node")
        for fact in (*ordered_nodes, *ordered_edges):
            if fact.provenance.repository_tree != repository_tree:
                raise ArchitectureIRError("graph fact repository_tree must match ArchitectureIR")
            if fact.provenance.freshness != freshness:
                raise ArchitectureIRError("graph fact freshness must match ArchitectureIR")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "nodes", ordered_nodes)
        object.__setattr__(self, "edges", ordered_edges)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=ArchitectureIRError,
                )
            )
            if claimed != identity:
                raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "edges": [edge.to_dict() for edge in self.edges],
            "freshness": self.freshness,
            "nodes": [node.to_dict() for node in self.nodes],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_parts(
        cls,
        *,
        repository_tree: str,
        freshness: str,
        nodes: Sequence[ArchitectureNode],
        edges: Sequence[ArchitectureEdge],
    ) -> "ArchitectureIR":
        return cls(
            schema=ARCHITECTURE_IR_SCHEMA,
            version=ARCHITECTURE_IR_VERSION,
            repository_tree=repository_tree,
            freshness=freshness,
            nodes=tuple(nodes),
            edges=tuple(edges),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureIR":
        mapping = _require_mapping(payload, error_type=ArchitectureIRError)
        try:
            _require_exact_fields(mapping, _GRAPH_FIELDS, error_type=ArchitectureIRError)
        except ArchitectureContractError as exc:
            if isinstance(exc, ArchitectureIRError):
                raise
            raise ArchitectureIRError(str(exc)) from exc
        nodes_payload = mapping["nodes"]
        edges_payload = mapping["edges"]
        if not isinstance(nodes_payload, list) or not isinstance(edges_payload, list):
            raise ArchitectureIRError("nodes and edges must be lists")
        try:
            nodes = tuple(ArchitectureNode.from_mapping(item) for item in nodes_payload)
            edges = tuple(ArchitectureEdge.from_mapping(item) for item in edges_payload)
        except ArchitectureContractError as exc:
            if isinstance(exc, ArchitectureIRError):
                raise
            raise ArchitectureIRError(str(exc)) from exc
        graph = cls(
            schema=mapping["schema"],
            version=mapping["version"],
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            nodes=nodes,
            edges=edges,
        )
        if mapping["content_identity"] != graph.content_identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return graph

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "ArchitectureIR":
        if type(payload) is not str or not payload:
            raise ArchitectureIRError("ArchitectureIR JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ArchitectureIRError("ArchitectureIR JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise ArchitectureIRError("ArchitectureIR JSON must contain an object")
        return cls.from_mapping(decoded)
