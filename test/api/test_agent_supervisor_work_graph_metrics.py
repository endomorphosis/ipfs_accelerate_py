"""Focused WorkGraphMetrics@1 checks."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.work_graph_metrics import (
    WorkGraphMetricsError,
    compute_work_graph_metrics,
)


def test_known_dag_is_deterministic() -> None:
    graph = {
        "nodes": ["A", "B", "C"],
        "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        "predicted_cost": {"A": 1, "B": 2, "C": 3},
    }
    first = compute_work_graph_metrics(graph)
    second = compute_work_graph_metrics(graph)
    assert first == second
    assert first["ok"] is True
    assert first["critical_path"] == 2
    assert first["order"] == ("A", "B", "C")


def test_cycle_and_overflow_are_typed() -> None:
    cycle = compute_work_graph_metrics(
        {
            "nodes": ["A", "B"],
            "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "A"}],
        }
    )
    assert cycle["ok"] is False
    assert cycle["finding"] == "cycle"
    with pytest.raises(WorkGraphMetricsError, match="overflow"):
        compute_work_graph_metrics(
            {
                "nodes": ["A"],
                "edges": [],
                "predicted_cost": {"A": 2**40},
            }
        )
    with pytest.raises(WorkGraphMetricsError, match="floats"):
        compute_work_graph_metrics({"nodes": ["A"], "edges": [], "durable": [1.25]})
