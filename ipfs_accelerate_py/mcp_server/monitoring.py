"""Deterministic monitoring helpers for unified MCP runtime.

This module ports the key monitoring APIs used by the source MCP runtime while
keeping runtime behavior lightweight and dependency-free for deterministic tests.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional


def _utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class MetricData:
    """Container for a metric sample."""

    value: float
    timestamp: datetime = field(default_factory=_utc_now)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check callback."""

    component: str
    status: str
    message: str
    timestamp: datetime = field(default_factory=_utc_now)
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None


class EnhancedMetricsCollector:
    """In-memory metrics collector used by unified MCP bootstrap."""

    def __init__(self, enabled: bool = True, retention_hours: int = 24) -> None:
        self.enabled = bool(enabled)
        self.retention_hours = max(1, int(retention_hours))
        self.start_time = _utc_now()
        self._lock = threading.Lock()

        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2048))
        self.tool_metrics = {
            "call_counts": defaultdict(int),
            "error_counts": defaultdict(int),
            "execution_times": defaultdict(lambda: deque(maxlen=512)),
            "success_rates": defaultdict(float),
            "last_called": defaultdict(lambda: None),
        }
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.request_count = 0
        self.error_count = 0
        self.request_times_ms: deque[float] = deque(maxlen=2048)
        self.active_requests: Dict[str, datetime] = {}
        self._started_monitoring = False

    def start_monitoring(self) -> None:
        """Mark collector as active.

        Unified runtime keeps monitoring deterministic in-process; no background
        thread is required for the current parity scope.
        """
        self._started_monitoring = True

    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        if not self.enabled:
            return
        metric_name = self._metric_name_with_labels(name, labels)
        with self._lock:
            self.counters[metric_name] += float(value)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        if not self.enabled:
            return
        metric_name = self._metric_name_with_labels(name, labels)
        with self._lock:
            self.gauges[metric_name] = float(value)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        if not self.enabled:
            return
        metric_name = self._metric_name_with_labels(name, labels)
        with self._lock:
            self.histograms[metric_name].append(float(value))

    @asynccontextmanager
    async def track_request(self, endpoint: str) -> AsyncGenerator[None, None]:
        """Track request lifecycle for coarse request/error/latency metrics."""
        request_id = f"{endpoint}:{time.time_ns()}"
        started = time.perf_counter()
        with self._lock:
            self.request_count += 1
            self.active_requests[request_id] = _utc_now()

        try:
            yield
        except Exception:
            with self._lock:
                self.error_count += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self.active_requests.pop(request_id, None)
                self.request_times_ms.append(elapsed_ms)

    def track_tool_execution(self, tool_name: str, execution_time_ms: float, success: bool) -> None:
        if not self.enabled:
            return

        with self._lock:
            self.tool_metrics["call_counts"][tool_name] += 1
            self.tool_metrics["execution_times"][tool_name].append(float(execution_time_ms))
            self.tool_metrics["last_called"][tool_name] = _utc_now()
            if not success:
                self.tool_metrics["error_counts"][tool_name] += 1

            calls = self.tool_metrics["call_counts"][tool_name]
            errors = self.tool_metrics["error_counts"][tool_name]
            self.tool_metrics["success_rates"][tool_name] = 1.0 - (errors / calls if calls else 0.0)

    def register_health_check(self, name: str, result: HealthCheckResult) -> None:
        """Register a point-in-time health check result."""
        with self._lock:
            self.health_checks[name] = result

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a deterministic point-in-time metrics summary."""
        with self._lock:
            tool_summary: Dict[str, Any] = {}
            for tool in self.tool_metrics["call_counts"].keys():
                times = list(self.tool_metrics["execution_times"][tool])
                avg_ms = float(sum(times) / len(times)) if times else 0.0
                tool_summary[tool] = {
                    "total_calls": int(self.tool_metrics["call_counts"][tool]),
                    "error_count": int(self.tool_metrics["error_counts"][tool]),
                    "success_rate": float(self.tool_metrics["success_rates"][tool]),
                    "avg_execution_time_ms": avg_ms,
                    "last_called": self.tool_metrics["last_called"][tool],
                }

            request_total = int(self.request_count)
            error_total = int(self.error_count)
            avg_response_ms = float(sum(self.request_times_ms) / len(self.request_times_ms)) if self.request_times_ms else 0.0
            error_rate = float(error_total / request_total) if request_total else 0.0

            return {
                "uptime_seconds": (_utc_now() - self.start_time).total_seconds(),
                "request_metrics": {
                    "total_requests": request_total,
                    "total_errors": error_total,
                    "error_rate": error_rate,
                    "avg_response_time_ms": avg_response_ms,
                    "active_requests": len(self.active_requests),
                },
                "tool_metrics": tool_summary,
                "health_status": {
                    name: {
                        "status": item.status,
                        "message": item.message,
                        "last_check": item.timestamp,
                        "response_time_ms": item.response_time_ms,
                    }
                    for name, item in self.health_checks.items()
                },
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "monitoring_started": bool(self._started_monitoring),
            }

    def get_snapshot(self) -> Dict[str, Any]:
        """Compatibility alias used by Prometheus exporter integration."""
        summary = self.get_metrics_summary()
        return {
            "error_rate": summary["request_metrics"]["error_rate"],
            "active_connections": summary["request_metrics"]["active_requests"],
            "system_metrics": {
                "cpu_percent": float(self.gauges.get("system_cpu_percent", 0.0)),
                "memory_percent": float(self.gauges.get("system_memory_percent", 0.0)),
            },
            "tool_metrics": summary["tool_metrics"],
            "request_metrics": summary["request_metrics"],
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Compatibility alias with source runtime naming."""
        return self.get_snapshot()

    def get_performance_trends(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Return coarse trend points from retained request latencies."""
        _ = max(1, int(hours))
        now = _utc_now()
        cutoff = now - timedelta(hours=hours)
        with self._lock:
            samples: Iterable[float] = list(self.request_times_ms)
        points = [{"timestamp": cutoff.isoformat(), "value": float(sum(samples) / len(samples))}] if samples else []
        return {
            "response_time_trend": points,
            "cpu_trend": [],
            "memory_trend": [],
            "request_rate_trend": [],
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "collector": "enhanced_metrics",
            "enabled": self.enabled,
            "retention_hours": self.retention_hours,
            "monitoring_started": self._started_monitoring,
        }

    def _metric_name_with_labels(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return str(name)
        suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{suffix}]"


class P2PMetricsCollector:
    """Compatibility P2P metrics facade for source parity."""

    def __init__(self, base_collector: Optional[EnhancedMetricsCollector] = None) -> None:
        self.base_collector = base_collector or get_metrics_collector()
        self.peer_discoveries = 0
        self.workflow_events = 0
        self.bootstrap_events = 0

    def track_peer_discovery(self, source: str, peers_found: int, success: bool, duration_ms: Optional[float] = None) -> None:
        self.peer_discoveries += 1
        self.base_collector.increment_counter("p2p.peer_discovery.total", 1.0, {"source": source, "success": str(success).lower()})
        self.base_collector.increment_counter("p2p.peer_discovery.peers_found", float(max(0, int(peers_found))), {"source": source})
        if duration_ms is not None:
            self.base_collector.observe_histogram("p2p.peer_discovery.duration_ms", float(duration_ms), {"source": source})

    def track_workflow_execution(self, workflow_id: str, status: str, execution_time_ms: Optional[float] = None) -> None:
        _ = workflow_id
        self.workflow_events += 1
        self.base_collector.increment_counter("p2p.workflow.events", 1.0, {"status": status})
        if execution_time_ms is not None:
            self.base_collector.observe_histogram("p2p.workflow.execution_ms", float(execution_time_ms), {"status": status})

    def track_bootstrap_operation(self, method: str, success: bool, duration_ms: Optional[float] = None) -> None:
        self.bootstrap_events += 1
        self.base_collector.increment_counter("p2p.bootstrap.total", 1.0, {"method": method, "success": str(success).lower()})
        if duration_ms is not None:
            self.base_collector.observe_histogram("p2p.bootstrap.duration_ms", float(duration_ms), {"method": method})

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "peer_discovery": {"total_discoveries": self.peer_discoveries},
            "workflows": {"total_events": self.workflow_events},
            "bootstrap": {"total_events": self.bootstrap_events},
        }


_metrics_collector: Optional[EnhancedMetricsCollector] = None
_p2p_metrics_collector: Optional[P2PMetricsCollector] = None


def get_metrics_collector() -> EnhancedMetricsCollector:
    """Return process-global collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = EnhancedMetricsCollector()
    return _metrics_collector


def get_p2p_metrics_collector() -> P2PMetricsCollector:
    """Return process-global P2P collector singleton."""
    global _p2p_metrics_collector
    if _p2p_metrics_collector is None:
        _p2p_metrics_collector = P2PMetricsCollector(base_collector=get_metrics_collector())
    return _p2p_metrics_collector


__all__ = [
    "MetricData",
    "HealthCheckResult",
    "EnhancedMetricsCollector",
    "P2PMetricsCollector",
    "get_metrics_collector",
    "get_p2p_metrics_collector",
]
