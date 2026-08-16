"""WorkGraphMetrics@1 — integer/fixed-point DAG metrics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA = "lgswf/work-graph-metrics@1"
MAX_METRIC = 2**31 - 1


class WorkGraphMetricsError(ValueError):
    """Metrics could not be computed from the supplied DAG."""


def _checked(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise WorkGraphMetricsError(f"{name} must be an integer")
    if value < 0 or value > MAX_METRIC:
        raise WorkGraphMetricsError(f"{name} overflow/bounds")
    return value


def compute_work_graph_metrics(graph: Mapping[str, Any]) -> Mapping[str, Any]:
    nodes = list(graph.get("nodes") or ())
    edges = list(graph.get("edges") or ())
    if any(isinstance(item, float) for item in graph.get("durable", ())):
        raise WorkGraphMetricsError("binary floats in durable records")
    incoming: dict[str, list[str]] = {str(node): [] for node in nodes}
    outgoing: dict[str, list[str]] = {str(node): [] for node in nodes}
    for edge in edges:
        src, dst = str(edge["source"]), str(edge["target"])
        incoming.setdefault(dst, []).append(src)
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(src, incoming.get(src, []))
        outgoing.setdefault(dst, outgoing.get(dst, []))
    remaining = {node: len(incoming.get(node, ())) for node in incoming}
    queue = sorted(node for node, count in remaining.items() if count == 0)
    order: list[str] = []
    depth = {node: 0 for node in remaining}
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in sorted(outgoing.get(node, ())):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
                queue.sort()
    if len(order) != len(remaining):
        return MappingProxyType(
            {
                "schema": SCHEMA,
                "ok": False,
                "finding": "cycle",
                "nodes": tuple(sorted(remaining)),
            }
        )
    unlocks = {node: len(outgoing.get(node, ())) for node in remaining}
    blocked = {node: len(incoming.get(node, ())) for node in remaining}
    predicted = {
        node: _checked(int(graph.get("predicted_cost", {}).get(node, 1)), "cost")
        for node in remaining
    }
    observed = {
        node: graph.get("observed_cost", {}).get(node)
        for node in remaining
    }
    critical = max(depth.values(), default=0)
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "depth": MappingProxyType(depth),
        "critical_path": critical,
        "unlocks": MappingProxyType(unlocks),
        "blocked_goals": MappingProxyType(blocked),
        "predicted_cost": MappingProxyType(predicted),
        "observed_cost": MappingProxyType(observed),
        "order": tuple(order),
    }
    return MappingProxyType(payload)
