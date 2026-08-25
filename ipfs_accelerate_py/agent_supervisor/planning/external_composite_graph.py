"""Multi-authority composite dependency graph (EAAEF-080).

Edges keep distinct meanings: goal, task, AST, data, schema, contract, proof,
validation, scope, effect, merge, and resource.  Authority is not inferred
from graph membership.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


COMPOSITE_GRAPH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-composite-graph@1"
)
EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "goal",
        "task",
        "ast",
        "data",
        "schema",
        "contract",
        "proof",
        "validation",
        "scope",
        "effect",
        "merge",
        "resource",
    }
)


class CompositeGraphError(ValueError):
    """Malformed composite graph."""


@dataclass(frozen=True)
class CompositeEdge:
    kind: str
    source: str
    target: str
    meaning: str

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        if kind not in EDGE_KINDS:
            raise CompositeGraphError(f"unknown or conflated edge kind: {kind}")
        if not str(self.source).strip() or not str(self.target).strip():
            raise CompositeGraphError("edge endpoints are required")
        if str(self.source) == str(self.target):
            raise CompositeGraphError("self-edges are not admitted")
        if not str(self.meaning).strip():
            raise CompositeGraphError("edge meaning is required")
        if kind == "authority" or "authority" in str(self.meaning).lower():
            raise CompositeGraphError("graph membership is not authority")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "kind": self.kind,
                "source": self.source,
                "target": self.target,
                "meaning": self.meaning,
            }
        )


@dataclass(frozen=True)
class CompositeGraph:
    nodes: tuple[str, ...]
    edges: tuple[CompositeEdge, ...]

    def __post_init__(self) -> None:
        nodes = tuple(str(node).strip() for node in self.nodes if str(node).strip())
        if len(set(nodes)) != len(nodes):
            raise CompositeGraphError("duplicate nodes")
        object.__setattr__(self, "nodes", nodes)
        node_set = set(nodes)
        for edge in self.edges:
            if edge.source not in node_set or edge.target not in node_set:
                raise CompositeGraphError("edge references a missing node")
        kinds = {edge.kind for edge in self.edges}
        if "effect" in kinds and "merge" in kinds:
            # Distinct meanings may coexist; they must not share identity.
            for effect in (edge for edge in self.edges if edge.kind == "effect"):
                for merge in (edge for edge in self.edges if edge.kind == "merge"):
                    if (
                        effect.source == merge.source
                        and effect.target == merge.target
                        and effect.meaning == merge.meaning
                    ):
                        raise CompositeGraphError("effect and merge edges must not conflate")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": COMPOSITE_GRAPH_SCHEMA,
                "nodes": list(self.nodes),
                "edges": [dict(edge.to_dict()) for edge in self.edges],
            }
        )

    @property
    def content_id(self) -> str:
        return content_identity(dict(self.to_dict()))

    @classmethod
    def compose(cls, nodes: Sequence[str], edges: Sequence[Mapping[str, Any] | CompositeEdge]) -> "CompositeGraph":
        compiled = []
        for edge in edges:
            if isinstance(edge, CompositeEdge):
                compiled.append(edge)
            else:
                compiled.append(
                    CompositeEdge(
                        kind=str(edge.get("kind") or ""),
                        source=str(edge.get("source") or ""),
                        target=str(edge.get("target") or ""),
                        meaning=str(edge.get("meaning") or ""),
                    )
                )
        return cls(nodes=tuple(nodes), edges=tuple(compiled))
