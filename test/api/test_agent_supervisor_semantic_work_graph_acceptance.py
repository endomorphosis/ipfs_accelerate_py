"""W5 integration acceptance."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.planning.semantic_work_graph import EDGE_KINDS
from ipfs_accelerate_py.agent_supervisor.planning.semantic_work_graph_integration import (
    compose_w5,
    shared_read_is_not_a_conflict,
)


def test_w5_roots_are_reproducible_and_separate_conflict() -> None:
    edges = [
        {
            "source": kind,
            "target": "Z",
            "kind": kind,
            "authority": "ipfs_accelerate_py",
            "evidence": "sha256:" + ("aa" * 32),
            "certainty": 1,
            "source_root": "sha256:" + ("bb" * 32),
            "source_plan": "sha256:" + ("cc" * 32),
            "invalidation": "root-change",
        }
        for kind in EDGE_KINDS
    ]
    tasks = [
        {"task_id": "T1", "mode": "read", "symbol": "A"},
        {"task_id": "T2", "mode": "read", "symbol": "A"},
        {"task_id": "T3", "mode": "write", "symbol": "B"},
    ]
    graph = {
        "nodes": ["T1", "T2", "T3"],
        "edges": [{"source": "T1", "target": "T3"}],
        "predicted_cost": {"T1": 1, "T2": 1, "T3": 1},
    }
    first = compose_w5(edges, tasks, graph)
    second = compose_w5(list(reversed(edges)), tasks, graph)
    assert first["work_graph_cid"] == second["work_graph_cid"]
    assert first["dependency_conflict_separated"] is True
    assert shared_read_is_not_a_conflict() is True
    assert "T1" in first["conflict_admitted"]
