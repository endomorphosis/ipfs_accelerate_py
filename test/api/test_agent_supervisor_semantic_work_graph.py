"""Focused SemanticWorkGraph@1 checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.semantic_work_graph import (
    EDGE_KINDS,
    SemanticWorkGraphError,
    compose_work_graph,
    parse_edge,
)


def _edge(kind: str = "task", source: str = "A", target: str = "B") -> dict:
    return {
        "source": source,
        "target": target,
        "kind": kind,
        "authority": "ipfs_accelerate_py",
        "evidence": "sha256:" + ("aa" * 32),
        "certainty": 1,
        "source_root": "sha256:" + ("bb" * 32),
        "source_plan": "sha256:" + ("cc" * 32),
        "invalidation": "root-change",
    }


def test_all_edge_kinds_and_order_independence() -> None:
    edges = [_edge(kind, source=kind, target="Z") for kind in EDGE_KINDS]
    first = compose_work_graph(edges)
    second = compose_work_graph(list(reversed(edges)))
    assert first["graph_cid"] == second["graph_cid"]
    assert set(first["edge_kinds"]) == set(EDGE_KINDS)


def test_mixed_root_and_unknown_kind_fail() -> None:
    with pytest.raises(SemanticWorkGraphError, match="mixed"):
        parse_edge({**_edge(), "authority": "mixed"})
    with pytest.raises(SemanticWorkGraphError, match="collapsed"):
        parse_edge({**_edge(), "kind": "everything"})
