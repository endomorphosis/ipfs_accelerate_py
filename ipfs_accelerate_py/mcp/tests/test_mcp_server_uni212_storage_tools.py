#!/usr/bin/env python3
"""UNI-212 storage collection-stats alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_collection_stats,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI212StorageTools(unittest.TestCase):
    def test_schema_registers_collection_stats_alias_tool(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("get_storage_collection_stats", by_name)
        schema = by_name["get_storage_collection_stats"]["input_schema"]
        self.assertEqual(schema.get("required"), ["collection_name"])
        props = schema.get("properties", {})
        self.assertEqual((props.get("report_format") or {}).get("default"), "summary")

    def test_collection_stats_alias_requires_collection_name(self) -> None:
        async def _run() -> None:
            result = await get_storage_collection_stats(collection_name="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("collection_name is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_collection_stats_alias_returns_normalized_shape(self) -> None:
        async def _run() -> None:
            result = await get_storage_collection_stats(
                collection_name="default",
                report_format="summary",
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("scope"), "collection")
            self.assertEqual(result.get("collection_name"), "default")
            self.assertIn("total_objects", result)
            self.assertIn("total_bytes", result)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
