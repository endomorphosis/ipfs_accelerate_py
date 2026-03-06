#!/usr/bin/env python3
"""UNI-173 storage analytics-reporting parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI173StorageTools(unittest.TestCase):
    def test_manage_collections_schema_exposes_analytics_report_format(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        report_schema = schemas["manage_collections"]["properties"]["report_format"]
        report_enum = report_schema.get("enum") or []
        self.assertIn("analytics", report_enum)

    def test_manage_collections_summary_handles_nested_basic_stats_shape(self) -> None:
        async def _fake_manage_collections(**_: object) -> dict:
            return {
                "action": "stats",
                "success": True,
                "global_stats": {
                    "basic_stats": {
                        "total_items": 12,
                        "total_size_bytes": 4096,
                        "storage_types": {"memory": 9, "ipfs": 3},
                    }
                },
            }

        async def _run() -> None:
            with patch.dict(
                native_storage_tools._API,
                {"manage_collections": _fake_manage_collections},
                clear=False,
            ):
                result = await native_storage_tools.manage_collections(
                    action="stats",
                    report_format="summary",
                    include_breakdown=True,
                )

            self.assertEqual(result.get("status"), "success")
            report = result.get("storage_report") or {}
            self.assertEqual((report.get("summary") or {}).get("total_items"), 12)
            self.assertEqual((report.get("summary") or {}).get("total_size_bytes"), 4096)
            self.assertEqual(
                ((report.get("breakdown") or {}).get("storage_distribution") or {}).get("ipfs"),
                3,
            )

        anyio.run(_run)

    def test_manage_collections_analytics_report_contains_expected_fields(self) -> None:
        async def _fake_manage_collections(**_: object) -> dict:
            return {
                "action": "stats",
                "success": True,
                "global_stats": {
                    "basic_stats": {
                        "total_items": 5,
                        "total_size_bytes": 1024,
                        "storage_types": {"memory": 5},
                    },
                    "average_item_size_bytes": 204.8,
                    "compression_usage_ratios": {"none": 1.0},
                    "largest_collection": "default",
                },
            }

        async def _run() -> None:
            with patch.dict(
                native_storage_tools._API,
                {"manage_collections": _fake_manage_collections},
                clear=False,
            ):
                result = await native_storage_tools.manage_collections(
                    action="stats",
                    report_format="analytics",
                )

            self.assertEqual(result.get("status"), "success")
            report = result.get("storage_report") or {}
            analytics = report.get("analytics") or {}
            self.assertEqual(analytics.get("scope"), "global")
            self.assertEqual((analytics.get("totals") or {}).get("total_items"), 5)
            self.assertEqual((analytics.get("storage_distribution") or {}).get("memory"), 5)
            self.assertEqual(analytics.get("largest_collection"), "default")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
