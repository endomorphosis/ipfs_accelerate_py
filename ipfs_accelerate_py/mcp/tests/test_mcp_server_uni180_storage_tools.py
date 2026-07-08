#!/usr/bin/env python3
"""UNI-180 storage lifecycle alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_lifecycle_report,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI180StorageTools(unittest.TestCase):
    def test_storage_schema_registers_lifecycle_alias_tool(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("get_storage_lifecycle_report", by_name)
        schema = by_name["get_storage_lifecycle_report"]["input_schema"]
        props = schema.get("properties", {})
        self.assertEqual((props.get("report_format") or {}).get("default"), "detailed")
        self.assertEqual((props.get("include_breakdown") or {}).get("default"), False)

    def test_lifecycle_alias_validates_report_format(self) -> None:
        async def _run() -> None:
            result = await get_storage_lifecycle_report(report_format="xml")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("report_format must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_lifecycle_alias_returns_normalized_shape(self) -> None:
        async def _run() -> None:
            result = await get_storage_lifecycle_report(
                collection_name="default",
                report_format="analytics",
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("scope"), "collection")
            self.assertEqual(result.get("collection_name"), "default")
            self.assertIn("collections_total", result)
            self.assertIn("collection_names", result)
            self.assertIn("totals", result)
            self.assertIn("lifecycle_report", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
