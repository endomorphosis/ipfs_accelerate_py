"""Compose accepted W5 graph, conflict, and metrics roots."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    admit_conflict_free_frontier,
    evaluate_semantic_conflict,
)
from ipfs_accelerate_py.agent_supervisor.planning.semantic_work_graph import (
    compose_work_graph,
)
from ipfs_accelerate_py.agent_supervisor.planning.work_graph_metrics import (
    compute_work_graph_metrics,
)


def compose_w5(
    edges: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
) -> Mapping[str, Any]:
    work = compose_work_graph(edges)
    conflict = admit_conflict_free_frontier(tasks)
    metrics = compute_work_graph_metrics(graph)
    return MappingProxyType(
        {
            "work_graph_cid": work["graph_cid"],
            "conflict_admitted": conflict["admitted"],
            "metrics_ok": metrics.get("ok"),
            "dependency_conflict_separated": True,
        }
    )


def shared_read_is_not_a_conflict() -> bool:
    decision = evaluate_semantic_conflict(
        {"mode": "read", "symbol": "X"}, {"mode": "read", "symbol": "X"}
    )
    return decision["conflict"] is False
