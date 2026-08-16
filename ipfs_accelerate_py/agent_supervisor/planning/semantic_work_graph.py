"""SemanticWorkGraph@1 — distinct-authority composite work edges."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA = "lgswf/semantic-work-graph@1"
INTERFACE = "SemanticWorkGraph@1"

EDGE_KINDS = (
    "goal_parent",
    "goal_dependency",
    "task",
    "code",
    "data",
    "interface",
    "schema",
    "contract",
    "proof",
    "validation",
    "policy",
    "merge",
    "lifecycle",
    "scope_read",
    "scope_write",
    "scope_effect",
    "invalidation",
    "conflict",
    "supersession",
    "generation",
    "blocking",
    "unlocking",
)

REQUIRED_EDGE_FIELDS = (
    "source",
    "target",
    "kind",
    "authority",
    "evidence",
    "certainty",
    "source_root",
    "source_plan",
    "invalidation",
)


class SemanticWorkGraphError(ValueError):
    """SemanticWorkGraph@1 rejected an edge or graph."""


def _cid(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_edge(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("kind") not in EDGE_KINDS:
        raise SemanticWorkGraphError(f"unsupported or collapsed edge kind: {payload.get('kind')!r}")
    missing = [name for name in REQUIRED_EDGE_FIELDS if name not in payload]
    if missing:
        raise SemanticWorkGraphError(f"edge missing {missing}")
    if payload.get("authority") == "mixed":
        raise SemanticWorkGraphError("mixed-root/authority edges fail")
    if isinstance(payload.get("certainty"), float) and not isinstance(
        payload.get("certainty"), int
    ):
        raise SemanticWorkGraphError("binary floats are forbidden")
    edge = {name: payload[name] for name in REQUIRED_EDGE_FIELDS}
    edge["kind"] = payload["kind"]
    return MappingProxyType(edge)


def compose_work_graph(edges: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    parsed = [dict(parse_edge(edge)) for edge in edges]
    ordered = sorted(
        parsed,
        key=lambda item: (
            str(item["source"]),
            str(item["target"]),
            str(item["kind"]),
            str(item["authority"]),
        ),
    )
    graph = {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "edges": ordered,
        "edge_kinds": tuple(sorted({item["kind"] for item in ordered})),
    }
    graph["graph_cid"] = _cid({"edges": ordered, "schema": SCHEMA})
    return MappingProxyType(graph)
