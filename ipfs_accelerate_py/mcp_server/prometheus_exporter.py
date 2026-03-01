"""Prometheus exporter compatibility layer for unified MCP runtime."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import prometheus_client as prom  # type: ignore[import]

    PROMETHEUS_AVAILABLE = True
except ImportError:
    prom = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False


class _NoOpMetric:
    """Inert metric object that discards operations."""

    def labels(self, **_kw: Any) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        _ = amount

    def dec(self, amount: float = 1.0) -> None:
        _ = amount

    def set(self, value: float) -> None:
        _ = value

    def observe(self, value: float) -> None:
        _ = value


def _make_counter(name: str, documentation: str, labelnames: list[str] | None = None) -> Any:
    if PROMETHEUS_AVAILABLE:
        return prom.Counter(name, documentation, labelnames or [])
    return _NoOpMetric()


def _make_gauge(name: str, documentation: str, labelnames: list[str] | None = None) -> Any:
    if PROMETHEUS_AVAILABLE:
        return prom.Gauge(name, documentation, labelnames or [])
    return _NoOpMetric()


def _make_histogram(
    name: str,
    documentation: str,
    labelnames: list[str] | None = None,
    buckets: list[float] | None = None,
) -> Any:
    if PROMETHEUS_AVAILABLE:
        kwargs: Dict[str, Any] = {}
        if buckets:
            kwargs["buckets"] = buckets
        return prom.Histogram(name, documentation, labelnames or [], **kwargs)
    return _NoOpMetric()


class PrometheusExporter:
    """Bridge unified metrics collector snapshots into Prometheus metrics."""

    def __init__(self, collector: Any = None, *, port: int = 9090, namespace: str = "mcp") -> None:
        self.collector = collector
        self.port = int(port)
        self.namespace = str(namespace)
        self._http_server: Optional[Any] = None
        self._start_time = time.time()

        self.tool_calls_total = _make_counter(
            f"{namespace}_tool_calls_total",
            "Total number of MCP tool calls",
            ["category", "tool", "status"],
        )
        self.tool_latency_seconds = _make_histogram(
            f"{namespace}_tool_latency_seconds",
            "Tool call latency in seconds",
            ["category", "tool"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.active_connections = _make_gauge(f"{namespace}_active_connections", "Active MCP connections")
        self.error_rate = _make_gauge(f"{namespace}_error_rate", "MCP error rate")
        self.cpu_usage = _make_gauge(f"{namespace}_system_cpu_usage_percent", "System CPU percent")
        self.memory_usage = _make_gauge(f"{namespace}_system_memory_usage_percent", "System memory percent")
        self.uptime_seconds = _make_gauge(f"{namespace}_uptime_seconds", "MCP uptime seconds")

    def start_http_server(self) -> None:
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client is not installed")
        prom.start_http_server(self.port)
        self._http_server = True

    def stop_http_server(self) -> None:
        self._http_server = None

    def update(self) -> None:
        self.uptime_seconds.set(time.time() - self._start_time)

        if self.collector is None:
            return

        snapshot = self._safe_get_snapshot()
        if not isinstance(snapshot, dict):
            return

        if "error_rate" in snapshot:
            self.error_rate.set(float(snapshot["error_rate"]))
        if "active_connections" in snapshot:
            self.active_connections.set(float(snapshot["active_connections"]))

        system_metrics = snapshot.get("system_metrics", {})
        if isinstance(system_metrics, dict):
            if "cpu_percent" in system_metrics:
                self.cpu_usage.set(float(system_metrics["cpu_percent"]))
            if "memory_percent" in system_metrics:
                self.memory_usage.set(float(system_metrics["memory_percent"]))

    def _safe_get_snapshot(self) -> Dict[str, Any]:
        if hasattr(self.collector, "get_snapshot"):
            return self.collector.get_snapshot() or {}
        if hasattr(self.collector, "get_current_metrics"):
            return self.collector.get_current_metrics() or {}
        return {}

    def record_tool_call(self, category: str, tool: str, status: str, latency_seconds: float) -> None:
        self.tool_calls_total.labels(category=category, tool=tool, status=status).inc()
        self.tool_latency_seconds.labels(category=category, tool=tool).observe(float(latency_seconds))

    def get_info(self) -> Dict[str, Any]:
        return {
            "exporter": "prometheus",
            "namespace": self.namespace,
            "port": self.port,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "http_server_running": self._http_server is not None,
            "uptime_seconds": time.time() - self._start_time,
        }


__all__ = [
    "PrometheusExporter",
    "PROMETHEUS_AVAILABLE",
]
