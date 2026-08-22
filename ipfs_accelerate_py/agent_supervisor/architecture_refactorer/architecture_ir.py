"""Canonical ArchitectureIR graph values with strict identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import canonical_dag_json_bytes

from .contracts import (
    ARCHITECTURE_IR_SCHEMA,
    ARCHITECTURE_IR_VERSION,
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceSpan,
    _reject_unknown,
)

SHA256_PREFIX = "sha256:"


class ArchitectureIRError(ArchitectureContractError):
    """Fail-closed ArchitectureIR graph error."""


def _sha256_identity(payload: Mapping[str, Any]) -> str:
    digest = __import__("hashlib").sha256(canonical_dag_json_bytes(payload)).hexdigest()
    return SHA256_PREFIX + digest


@dataclass(frozen=True)
class ArchitectureNode:
    """One closed ArchitectureIR node."""

    node_id: str
    kind: NodeKind
    span: SourceSpan
    confidence: Confidence
    provenance: str
    content_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "span": self.span.to_dict(),
            "confidence": self.confidence.value,
            "provenance": self.provenance,
            "content_identity": self.content_identity,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureNode":
        _reject_unknown(
            payload,
            {
                "node_id",
                "kind",
                "span",
                "confidence",
                "provenance",
                "content_identity",
            },
        )
        span_payload = payload["span"]
        if not isinstance(span_payload, Mapping):
            raise ArchitectureIRError("node span must be an object")
        return cls(
            node_id=str(payload["node_id"]),
            kind=NodeKind(payload["kind"]),
            span=SourceSpan.from_mapping(span_payload),
            confidence=Confidence(payload["confidence"]),
            provenance=str(payload["provenance"]),
            content_identity=str(payload["content_identity"]),
        )


@dataclass(frozen=True)
class ArchitectureEdge:
    """One closed ArchitectureIR edge."""

    edge_id: str
    kind: EdgeKind
    source: str
    target: str
    confidence: Confidence
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "confidence": self.confidence.value,
            "provenance": self.provenance,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureEdge":
        _reject_unknown(
            payload,
            {"edge_id", "kind", "source", "target", "confidence", "provenance"},
        )
        return cls(
            edge_id=str(payload["edge_id"]),
            kind=EdgeKind(payload["kind"]),
            source=str(payload["source"]),
            target=str(payload["target"]),
            confidence=Confidence(payload["confidence"]),
            provenance=str(payload["provenance"]),
        )


@dataclass(frozen=True)
class ArchitectureIR:
    """Canonical closed architecture graph for one repository tree."""

    schema: str
    version: int
    repository_tree: str
    freshness: str
    nodes: tuple[ArchitectureNode, ...]
    edges: tuple[ArchitectureEdge, ...]
    content_identity: str

    def __post_init__(self) -> None:
        if self.schema != ARCHITECTURE_IR_SCHEMA:
            raise ArchitectureIRError("unexpected ArchitectureIR schema")
        if self.version != ARCHITECTURE_IR_VERSION:
            raise ArchitectureIRError("unexpected ArchitectureIR version")
        if type(self.repository_tree) is not str or not self.repository_tree:
            raise ArchitectureIRError("repository_tree must be a nonempty string")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ArchitectureIRError("ArchitectureIR node ids must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ArchitectureIRError("ArchitectureIR edge references unknown node")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "repository_tree": self.repository_tree,
            "freshness": self.freshness,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        identity = _sha256_identity(payload)
        if self.content_identity != identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return {**payload, "content_identity": self.content_identity}

    @classmethod
    def from_parts(
        cls,
        *,
        repository_tree: str,
        freshness: str,
        nodes: Sequence[ArchitectureNode],
        edges: Sequence[ArchitectureEdge],
    ) -> "ArchitectureIR":
        ordered_nodes = tuple(sorted(nodes, key=lambda item: item.node_id))
        ordered_edges = tuple(sorted(edges, key=lambda item: item.edge_id))
        payload = {
            "schema": ARCHITECTURE_IR_SCHEMA,
            "version": ARCHITECTURE_IR_VERSION,
            "repository_tree": repository_tree,
            "freshness": freshness,
            "nodes": [node.to_dict() for node in ordered_nodes],
            "edges": [edge.to_dict() for edge in ordered_edges],
        }
        return cls(
            schema=ARCHITECTURE_IR_SCHEMA,
            version=ARCHITECTURE_IR_VERSION,
            repository_tree=repository_tree,
            freshness=freshness,
            nodes=ordered_nodes,
            edges=ordered_edges,
            content_identity=_sha256_identity(payload),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArchitectureIR":
        try:
            _reject_unknown(
                payload,
                {
                    "schema",
                    "version",
                    "repository_tree",
                    "freshness",
                    "nodes",
                    "edges",
                    "content_identity",
                },
            )
        except ArchitectureContractError as exc:
            raise ArchitectureIRError(str(exc)) from exc
        nodes_payload = payload.get("nodes")
        edges_payload = payload.get("edges")
        if not isinstance(nodes_payload, list) or not isinstance(edges_payload, list):
            raise ArchitectureIRError("nodes and edges must be lists")
        graph = cls.from_parts(
            repository_tree=str(payload["repository_tree"]),
            freshness=str(payload["freshness"]),
            nodes=tuple(ArchitectureNode.from_mapping(item) for item in nodes_payload),
            edges=tuple(ArchitectureEdge.from_mapping(item) for item in edges_payload),
        )
        claimed = str(payload.get("content_identity") or "")
        if claimed and claimed != graph.content_identity:
            raise ArchitectureIRError("ArchitectureIR content identity mismatch")
        return graph
