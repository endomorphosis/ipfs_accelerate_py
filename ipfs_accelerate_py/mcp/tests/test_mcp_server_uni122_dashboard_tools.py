#!/usr/bin/env python3
"""UNI-122 dashboard tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.dashboard_tools.native_dashboard_tools import (
    compare_tdfol_strategies,
    export_tdfol_statistics,
    get_tdfol_profiler_report,
    profile_tdfol_operation,
    register_native_dashboard_tools,
)


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


if __name__ == "__main__":
    unittest.main()
