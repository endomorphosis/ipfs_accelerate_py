#!/usr/bin/env python3
"""UNI-125 monitoring tools parity hardening extension tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools import (
    check_health,
    collect_metrics,
    generate_monitoring_report,
    get_performance_metrics,
    health_check,
    manage_alerts,
    monitor_services,
    register_native_monitoring_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI125MonitoringTools(unittest.TestCase):
    def test_register_schema_contracts_include_enums(self) -> None:
        manager = _DummyManager()
        register_native_monitoring_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        perf_schema = by_name["get_performance_metrics"]["input_schema"]
        self.assertIn("24h", perf_schema["properties"]["time_range"].get("enum", []))

        report_schema = by_name["generate_monitoring_report"]["input_schema"]
        self.assertIn("performance", report_schema["properties"]["report_type"].get("enum", []))

        health_schema = by_name["check_health"]["input_schema"]
        self.assertIn("comprehensive", health_schema["properties"]["check_depth"].get("enum", []))

        collect_schema = by_name["collect_metrics"]["input_schema"]
        self.assertIn("parquet", collect_schema["properties"]["export_format"].get("enum", []))

        alerts_schema = by_name["manage_alerts"]["input_schema"]
        self.assertIn("configure_thresholds", alerts_schema["properties"]["action"].get("enum", []))

    def test_health_check_rejects_invalid_check_type(self) -> None:
        async def _run() -> None:
            result = await health_check(check_type="full")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("check_type must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_get_performance_metrics_rejects_invalid_metric_types_shape(self) -> None:
        async def _run() -> None:
            result = await get_performance_metrics(metric_types="cpu")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("metric_types must be an array", str(result.get("error", "")))

        anyio.run(_run)

    def test_monitor_services_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await monitor_services(action="reload")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_generate_report_rejects_invalid_time_period(self) -> None:
        async def _run() -> None:
            result = await generate_monitoring_report(time_period="30d")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("time_period must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_health_check_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await health_check(check_type="basic")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("check_type"), "basic")

        anyio.run(_run)

    def test_success_defaults_applied_for_minimal_metrics_payload(self) -> None:
        async def _minimal_metrics(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools._API",
                {
                    "health_check": None,
                    "get_performance_metrics": _minimal_metrics,
                    "monitor_services": None,
                    "generate_monitoring_report": None,
                },
            ):
                result = await get_performance_metrics(time_range="1h", metric_types=["cpu"], include_history=False)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("time_range"), "1h")
            self.assertEqual(result.get("metric_types"), ["cpu"])
            self.assertEqual(result.get("include_history"), False)
            self.assertEqual(result.get("metrics"), {})

        anyio.run(_run)

    def test_success_defaults_applied_for_minimal_status_and_report_payloads(self) -> None:
        async def _minimal_services(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_report(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools._API",
                {
                    "health_check": None,
                    "get_performance_metrics": None,
                    "monitor_services": _minimal_services,
                    "generate_monitoring_report": _minimal_report,
                },
            ):
                status_result = await monitor_services(action="status", services=["mcp_server"], check_interval=5)
                report_result = await generate_monitoring_report(report_type="summary", time_period="24h")

            self.assertEqual(status_result.get("status"), "success")
            self.assertEqual(status_result.get("action"), "status")
            self.assertEqual(status_result.get("services"), ["mcp_server"])
            self.assertEqual(status_result.get("check_interval"), 5)

            self.assertEqual(report_result.get("status"), "success")
            self.assertEqual(report_result.get("report_type"), "summary")
            self.assertEqual(report_result.get("time_period"), "24h")
            self.assertEqual(report_result.get("report"), {})

        anyio.run(_run)

    def test_check_health_rejects_invalid_depth(self) -> None:
        async def _run() -> None:
            result = await check_health(check_depth="deep")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("check_depth must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_collect_metrics_rejects_invalid_export_format(self) -> None:
        async def _run() -> None:
            result = await collect_metrics(export_format="xml")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("export_format must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_alerts_rejects_missing_alert_id(self) -> None:
        async def _run() -> None:
            result = await manage_alerts(action="resolve")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("alert_id required", str(result.get("message", "")))

        anyio.run(_run)

    def test_enhanced_monitoring_success_shapes(self) -> None:
        async def _run() -> None:
            health_result = await check_health(check_depth="comprehensive")
            self.assertEqual(health_result.get("status"), "success")
            self.assertIn("health_check", health_result)
            self.assertIn("timestamp", health_result)
            self.assertIn("diagnostics", health_result)

            metrics_result = await collect_metrics(include_anomalies=True)
            self.assertEqual(metrics_result.get("status"), "success")
            self.assertIn("metrics_collection", metrics_result)
            self.assertIn("trend_analysis", metrics_result)
            self.assertIn("anomaly_detection", metrics_result)

            alerts_result = await manage_alerts(action="list", include_metrics=True)
            self.assertEqual(alerts_result.get("status"), "success")
            self.assertIn("alerts", alerts_result)
            self.assertIn("filters_applied", alerts_result)
            self.assertIn("alert_metrics", alerts_result)

        anyio.run(_run)

    def test_enhanced_success_defaults_applied_for_minimal_payloads(self) -> None:
        async def _minimal_check_health(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_collect_metrics(**_: object) -> dict:
            return {"status": "success"}

        async def _minimal_manage_alerts(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.monitoring_tools.native_monitoring_tools._API",
                {
                    "health_check": None,
                    "get_performance_metrics": None,
                    "monitor_services": None,
                    "generate_monitoring_report": None,
                    "check_health": _minimal_check_health,
                    "collect_metrics": _minimal_collect_metrics,
                    "manage_alerts": _minimal_manage_alerts,
                },
            ):
                health_result = await check_health(check_depth="comprehensive")
                metrics_result = await collect_metrics(include_anomalies=True, export_format="csv")
                alerts_list_result = await manage_alerts(action="list", include_metrics=True, time_range="1h")
                alerts_resolve_result = await manage_alerts(action="resolve", alert_id="alert-1")
                alerts_config_result = await manage_alerts(
                    action="configure_thresholds",
                    threshold_config={"cpu": 90},
                )

            self.assertEqual(health_result.get("status"), "success")
            self.assertIn("timestamp", health_result)
            self.assertEqual(health_result.get("diagnostics"), {})

            self.assertEqual(metrics_result.get("status"), "success")
            self.assertEqual(metrics_result.get("trend_analysis"), {})
            self.assertEqual(metrics_result.get("anomaly_detection"), {"anomalies_found": 0, "anomalies": []})
            self.assertEqual(metrics_result.get("export_info"), {"format": "csv"})

            self.assertEqual(alerts_list_result.get("status"), "success")
            self.assertEqual(alerts_list_result.get("filters_applied"), {"severity": None, "resolved": None, "time_range": "1h"})
            self.assertEqual(alerts_list_result.get("alert_metrics"), {})

            self.assertEqual(alerts_resolve_result.get("status"), "success")
            self.assertEqual(alerts_resolve_result.get("alert_id"), "alert-1")
            self.assertTrue(alerts_resolve_result.get("success"))
            self.assertIn("timestamp", alerts_resolve_result)

            self.assertEqual(alerts_config_result.get("status"), "success")
            self.assertEqual(alerts_config_result.get("updated_thresholds"), {"cpu": 90})
            self.assertEqual(alerts_config_result.get("current_thresholds"), {"cpu": 90})
            self.assertFalse(alerts_config_result.get("restart_required"))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
