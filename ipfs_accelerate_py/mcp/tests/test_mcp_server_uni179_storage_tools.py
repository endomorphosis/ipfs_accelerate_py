#!/usr/bin/env python3
"""UNI-179 storage lifecycle-report parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    manage_collections,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI179StorageTools(unittest.TestCase):
    def test_storage_schema_includes_lifecycle_report_action(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        manage_schema = by_name["manage_collections"]["input_schema"]
        actions = (manage_schema.get("properties", {}).get("action", {}).get("enum") or [])
        self.assertIn("lifecycle_report", actions)

    def test_lifecycle_report_global_shape(self) -> None:
        async def _run() -> None:
            result = await manage_collections(action="lifecycle_report", report_format="summary")
            self.assertEqual(result.get("status"), "success")
            report = result.get("lifecycle_report") or {}
            self.assertEqual(report.get("scope"), "global")
            self.assertIn("collections_total", report)
            self.assertIn("collection_names", report)
            totals = report.get("totals") or {}
            self.assertIn("total_items", totals)
            self.assertIn("total_size_bytes", totals)

        anyio.run(_run)

    def test_lifecycle_report_collection_scope(self) -> None:
        async def _run() -> None:
            result = await manage_collections(
                action="lifecycle_report",
                collection_name="default",
                report_format="analytics",
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            report = result.get("lifecycle_report") or {}
            self.assertEqual(report.get("scope"), "collection")
            self.assertEqual(report.get("collection_name"), "default")
            self.assertIn("collections", report)
            self.assertIn("analytics", report)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
