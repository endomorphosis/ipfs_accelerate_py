#!/usr/bin/env python3
"""Unit tests for audit-to-prometheus bridge wiring."""

from __future__ import annotations

import unittest

from ipfs_accelerate_py.mcp_server.audit_metrics_bridge import (
    AuditMetricsBridge,
    connect_audit_to_prometheus,
)
from ipfs_accelerate_py.mcp_server.policy_audit_log import PolicyAuditLog
from ipfs_accelerate_py.mcp_server.prometheus_exporter import PrometheusExporter


class TestAuditMetricsBridge(unittest.TestCase):
    """Validate policy-audit to metrics bridge behavior."""

    def test_bridge_forwards_allow_and_deny_decisions(self) -> None:
        audit = PolicyAuditLog(enabled=True)
        exporter = PrometheusExporter(namespace="mcp_test_bridge")

        bridge = connect_audit_to_prometheus(audit, exporter)
        self.assertTrue(bridge.is_attached)

        audit.record(decision="allow", tool="smoke.echo", actor="did:model:worker")
        audit.record(decision="deny", tool="smoke.echo", actor="did:model:worker")

        self.assertEqual(bridge.forwarded_count, 2)
        info = bridge.get_info()
        self.assertTrue(info["attached"])
        self.assertEqual(info["category"], "policy")
        self.assertEqual(info["forwarded_count"], 2)

    def test_bridge_detach_restores_previous_sink(self) -> None:
        sink_calls: list[str] = []

        def previous_sink(entry) -> None:
            sink_calls.append(entry.decision)

        audit = PolicyAuditLog(enabled=True, sink=previous_sink)
        exporter = PrometheusExporter(namespace="mcp_test_bridge")

        bridge = AuditMetricsBridge(audit, exporter)
        bridge.attach()
        audit.record(decision="allow", tool="smoke.echo", actor="did:model:worker")
        bridge.detach()
        audit.record(decision="deny", tool="smoke.echo", actor="did:model:worker")

        # First call goes through bridge and chained previous sink,
        # second call goes through restored previous sink only.
        self.assertEqual(sink_calls, ["allow", "deny"])
        self.assertEqual(bridge.forwarded_count, 1)


if __name__ == "__main__":
    unittest.main()
