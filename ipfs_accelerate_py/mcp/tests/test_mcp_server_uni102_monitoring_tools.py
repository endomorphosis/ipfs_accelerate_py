#!/usr/bin/env python3
"""UNI-102 monitoring tools parity and validation tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools import (
    generate_monitoring_report,
    get_performance_metrics,
    health_check,
    monitor_services,
    register_native_monitoring_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI102MonitoringTools(unittest.TestCase):
    def test_register_schema_includes_source_parity_fields(self) -> None:
        manager = _DummyManager()
        register_native_monitoring_tools(manager)
        self.assertEqual(len(manager.calls), 4)

        health_schema = manager.calls[0]["input_schema"]
        self.assertIn("check_type", health_schema["properties"])
        self.assertIn("components", health_schema["properties"])

        perf_schema = manager.calls[1]["input_schema"]
        self.assertIn("metric_types", perf_schema["properties"])
        self.assertIn("include_history", perf_schema["properties"])

        monitor_schema = manager.calls[2]["input_schema"]
        self.assertIn("services", monitor_schema["properties"])
        self.assertIn("check_interval", monitor_schema["properties"])

        report_schema = manager.calls[3]["input_schema"]
        self.assertIn("time_period", report_schema["properties"])

    def test_health_check_maps_service_name_to_source_custom_call(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools._API",
                {
                    "health_check": None,
                    "get_performance_metrics": None,
                    "monitor_services": None,
                    "generate_monitoring_report": None,
                },
            ):
                async def _health_impl(**kwargs):
                    return kwargs

                from ipfs_accelerate_py.mcp_server.tools.monitoring_tools import native_monitoring_tools as nmt

                nmt._API["health_check"] = _health_impl
                result = await health_check(service_name="vector_store")

            self.assertEqual(result.get("check_type"), "custom")
            self.assertEqual(result.get("components"), ["vector_store"])
            self.assertEqual(result.get("include_metrics"), True)

        anyio.run(_run)

    def test_get_performance_metrics_rejects_invalid_time_range(self) -> None:
        async def _run() -> None:
            result = await get_performance_metrics(time_range="2h")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("time_range must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_monitor_services_action_passthrough_for_non_status(self) -> None:
        async def _run() -> None:
            result = await monitor_services(action="restart", services=["mcp_server"], check_interval=5)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "restart")
            self.assertIn("not supported", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_monitoring_report_passes_time_period(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools._API",
                {
                    "health_check": None,
                    "get_performance_metrics": None,
                    "monitor_services": None,
                    "generate_monitoring_report": None,
                },
            ):
                async def _report_impl(**kwargs):
                    return kwargs

                from ipfs_accelerate_py.mcp_server.tools.monitoring_tools import native_monitoring_tools as nmt

                nmt._API["generate_monitoring_report"] = _report_impl
                result = await generate_monitoring_report(report_type="detailed", time_period="7d")

            self.assertEqual(result.get("report_type"), "detailed")
            self.assertEqual(result.get("time_period"), "7d")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
