#!/usr/bin/env python3
"""Unit tests for unified MCP observability components."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.monitoring import EnhancedMetricsCollector, P2PMetricsCollector
from ipfs_accelerate_py.mcp_server.otel_tracing import MCPTracer
from ipfs_accelerate_py.mcp_server.prometheus_exporter import PrometheusExporter


class TestObservabilityMonitoring(unittest.TestCase):
    """Validate deterministic monitoring behavior."""

    def test_metrics_collector_tracks_tool_execution(self) -> None:
        collector = EnhancedMetricsCollector(enabled=True)
        collector.start_monitoring()

        collector.track_tool_execution("smoke.echo", 12.5, True)
        collector.track_tool_execution("smoke.echo", 25.0, False)

        snapshot = collector.get_snapshot()
        tool = snapshot["tool_metrics"].get("smoke.echo") or {}

        self.assertEqual(tool.get("total_calls"), 2)
        self.assertEqual(tool.get("error_count"), 1)
        self.assertLess(tool.get("success_rate", 0.0), 1.0)

    def test_p2p_metrics_collector_updates_base_metrics(self) -> None:
        collector = EnhancedMetricsCollector(enabled=True)
        p2p = P2PMetricsCollector(base_collector=collector)

        p2p.track_peer_discovery("dht", peers_found=3, success=True, duration_ms=5.0)
        p2p.track_workflow_execution("wf-1", status="completed", execution_time_ms=11.0)
        p2p.track_bootstrap_operation("seed", success=True, duration_ms=3.0)

        self.assertGreaterEqual(collector.counters.get("p2p.peer_discovery.total[source=dht,success=true]", 0.0), 1.0)
        self.assertEqual(p2p.get_dashboard_data()["peer_discovery"]["total_discoveries"], 1)


class TestObservabilityTracing(unittest.TestCase):
    """Validate tracing fallback behavior without hard dependency requirements."""

    def test_tracer_span_context_is_safe(self) -> None:
        tracer = MCPTracer()
        with tracer.start_dispatch_span("smoke", "echo", {"value": "ok"}) as span:
            tracer.set_span_ok(span, {"status": "ok"})

        info = tracer.get_info()
        self.assertEqual(info.get("tracer"), "opentelemetry")
        self.assertIn("otel_available", info)


class TestObservabilityPrometheus(unittest.TestCase):
    """Validate exporter behavior with dependency-optional no-op backend."""

    def test_prometheus_exporter_updates_from_collector_snapshot(self) -> None:
        collector = EnhancedMetricsCollector(enabled=True)
        collector.set_gauge("system_cpu_percent", 12.0)
        collector.set_gauge("system_memory_percent", 33.0)
        collector.track_tool_execution("smoke.echo", 8.0, True)

        exporter = PrometheusExporter(collector=collector, namespace="mcp_test")
        exporter.record_tool_call("smoke", "echo", "success", 0.01)
        exporter.update()

        info = exporter.get_info()
        self.assertEqual(info.get("namespace"), "mcp_test")
        self.assertIn("prometheus_available", info)


if __name__ == "__main__":
    unittest.main()
