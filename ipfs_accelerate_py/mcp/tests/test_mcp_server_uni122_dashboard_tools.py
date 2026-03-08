#!/usr/bin/env python3
"""UNI-122 dashboard tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.dashboard_tools.native_dashboard_tools import (
    check_tdfol_performance_regression,
    compare_tdfol_strategies,
    export_tdfol_statistics,
    generate_tdfol_dashboard,
    get_tdfol_metrics,
    get_tdfol_profiler_report,
    profile_tdfol_operation,
    reset_tdfol_metrics,
    register_native_dashboard_tools,
)
from ipfs_accelerate_py.mcp_server.tools.dashboard_tools import native_dashboard_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI122DashboardTools(unittest.TestCase):
    def test_register_includes_dashboard_tools(self) -> None:
        manager = _DummyManager()
        register_native_dashboard_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("profile_tdfol_operation", names)
        self.assertIn("export_tdfol_statistics", names)
        self.assertIn("check_tdfol_performance_regression", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_dashboard_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        export_schema = by_name["export_tdfol_statistics"]["input_schema"]
        self.assertIn("json", export_schema["properties"]["format"].get("enum", []))

        report_schema = by_name["get_tdfol_profiler_report"]["input_schema"]
        self.assertEqual(report_schema["properties"]["top_n"].get("minimum"), 1)

    def test_profile_rejects_empty_formula(self) -> None:
        async def _run() -> None:
            result = await profile_tdfol_operation(formula_str="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("formula_str is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_export_rejects_invalid_format(self) -> None:
        async def _run() -> None:
            result = await export_tdfol_statistics(format="csv")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("format must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_profiler_report_rejects_invalid_top_n(self) -> None:
        async def _run() -> None:
            result = await get_tdfol_profiler_report(top_n=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("top_n must be an integer >= 1", str(result.get("message", "")))

        anyio.run(_run)

    def test_compare_rejects_invalid_strategy(self) -> None:
        async def _run() -> None:
            result = await compare_tdfol_strategies(
                formula_str="P(a)",
                strategies=["forward", "beam"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("strategies must only include", str(result.get("message", "")))

        anyio.run(_run)

    def test_profile_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await profile_tdfol_operation(formula_str="P(a)", runs=1)
            self.assertIn(result.get("status"), ["success", "error", "warning"])
            self.assertEqual(result.get("formula"), "P(a)")
            self.assertEqual(result.get("runs"), 1)

        anyio.run(_run)

    def test_dashboard_and_metrics_success_defaults_with_minimal_payloads(self) -> None:
        def _minimal_metrics() -> dict:
            return {"status": "success"}

        def _minimal_profile(*_: object, **__: object) -> dict:
            return {"status": "success"}

        def _minimal_dashboard(*_: object, **__: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.dashboard_tools.native_dashboard_tools._API",
                {
                    "get_tdfol_metrics": _minimal_metrics,
                    "profile_tdfol_operation": _minimal_profile,
                    "generate_tdfol_dashboard": _minimal_dashboard,
                    "export_tdfol_statistics": None,
                    "get_tdfol_profiler_report": None,
                    "compare_tdfol_strategies": None,
                    "check_tdfol_performance_regression": None,
                    "reset_tdfol_metrics": None,
                },
            ):
                metrics = await get_tdfol_metrics()
                profiled = await profile_tdfol_operation(formula_str="P(a)", runs=2)
                dashboard = await generate_tdfol_dashboard(include_profiling=True)

            self.assertEqual(metrics.get("status"), "success")
            self.assertEqual(metrics.get("metrics"), {})

            self.assertEqual(profiled.get("status"), "success")
            self.assertEqual(profiled.get("formula"), "P(a)")
            self.assertEqual(profiled.get("runs"), 2)
            self.assertEqual(profiled.get("profile"), {})

            self.assertEqual(dashboard.get("status"), "success")
            self.assertEqual(dashboard.get("include_profiling"), True)
            self.assertEqual(dashboard.get("dashboard_generated"), False)

        anyio.run(_run)

    def test_export_report_compare_regression_reset_defaults_with_minimal_payloads(self) -> None:
        def _minimal_export(*_: object, **__: object) -> dict:
            return {"status": "success"}

        def _minimal_report(*_: object, **__: object) -> dict:
            return {"status": "success"}

        def _minimal_compare(*_: object, **__: object) -> dict:
            return {"status": "success"}

        def _minimal_regression(*_: object, **__: object) -> dict:
            return {"status": "success"}

        def _minimal_reset() -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.dashboard_tools.native_dashboard_tools._API",
                {
                    "get_tdfol_metrics": None,
                    "profile_tdfol_operation": None,
                    "generate_tdfol_dashboard": None,
                    "export_tdfol_statistics": _minimal_export,
                    "get_tdfol_profiler_report": _minimal_report,
                    "compare_tdfol_strategies": _minimal_compare,
                    "check_tdfol_performance_regression": _minimal_regression,
                    "reset_tdfol_metrics": _minimal_reset,
                },
            ):
                exported = await export_tdfol_statistics(format="json", include_raw_data=False)
                report = await get_tdfol_profiler_report(report_format="text", top_n=5)
                compared = await compare_tdfol_strategies(formula_str="P(a)", runs_per_strategy=3)
                regression = await check_tdfol_performance_regression(threshold_percent=12.5)
                reset = await reset_tdfol_metrics()

            self.assertEqual(exported.get("status"), "success")
            self.assertEqual(exported.get("statistics"), {})

            self.assertEqual(report.get("status"), "success")
            self.assertEqual(report.get("report"), {})

            self.assertEqual(compared.get("status"), "success")
            self.assertEqual(compared.get("comparison"), {})

            self.assertEqual(regression.get("status"), "success")
            self.assertEqual(regression.get("regression_detected"), False)

            self.assertEqual(reset.get("status"), "success")
            self.assertEqual(reset.get("reset"), True)

        anyio.run(_run)

    def test_dashboard_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        def _contradictory_failure(*_: object, **__: object) -> dict:
            return {"status": "success", "success": False, "error": "dashboard delegate failure"}

        async def _run() -> None:
            with patch.dict(
                native_dashboard_tools._API,
                {
                    "get_tdfol_metrics": _contradictory_failure,
                    "profile_tdfol_operation": _contradictory_failure,
                    "generate_tdfol_dashboard": _contradictory_failure,
                    "export_tdfol_statistics": _contradictory_failure,
                    "get_tdfol_profiler_report": _contradictory_failure,
                    "compare_tdfol_strategies": _contradictory_failure,
                    "check_tdfol_performance_regression": _contradictory_failure,
                    "reset_tdfol_metrics": _contradictory_failure,
                },
                clear=False,
            ):
                metrics = await get_tdfol_metrics()
                self.assertEqual(metrics.get("status"), "error")
                self.assertEqual(metrics.get("error"), "dashboard delegate failure")

                profiled = await profile_tdfol_operation(formula_str="P(a)", runs=1)
                self.assertEqual(profiled.get("status"), "error")
                self.assertEqual(profiled.get("error"), "dashboard delegate failure")

                dashboard = await generate_tdfol_dashboard(include_profiling=True)
                self.assertEqual(dashboard.get("status"), "error")
                self.assertEqual(dashboard.get("error"), "dashboard delegate failure")

                exported = await export_tdfol_statistics(format="json")
                self.assertEqual(exported.get("status"), "error")
                self.assertEqual(exported.get("error"), "dashboard delegate failure")

                report = await get_tdfol_profiler_report(report_format="text", top_n=5)
                self.assertEqual(report.get("status"), "error")
                self.assertEqual(report.get("error"), "dashboard delegate failure")

                compared = await compare_tdfol_strategies(formula_str="P(a)", runs_per_strategy=2)
                self.assertEqual(compared.get("status"), "error")
                self.assertEqual(compared.get("error"), "dashboard delegate failure")

                regression = await check_tdfol_performance_regression(threshold_percent=10.0)
                self.assertEqual(regression.get("status"), "error")
                self.assertEqual(regression.get("error"), "dashboard delegate failure")

                reset = await reset_tdfol_metrics()
                self.assertEqual(reset.get("status"), "error")
                self.assertEqual(reset.get("error"), "dashboard delegate failure")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
