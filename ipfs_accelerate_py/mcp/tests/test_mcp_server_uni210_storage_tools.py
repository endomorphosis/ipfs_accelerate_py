#!/usr/bin/env python3
"""UNI-210 storage collection inventory alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    list_storage_collections,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI210StorageTools(unittest.TestCase):
    def test_schema_registers_collection_inventory_alias_tool(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("list_storage_collections", by_name)
        schema = by_name["list_storage_collections"]["input_schema"]
        props = schema.get("properties", {})
        self.assertEqual((props.get("include_metadata") or {}).get("default"), True)
        self.assertEqual((props.get("include_timestamps") or {}).get("default"), True)

    def test_collection_inventory_alias_validates_booleans(self) -> None:
        async def _run() -> None:
            result = await list_storage_collections(include_metadata="yes")  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("include_metadata must be a boolean", str(result.get("error", "")))

        anyio.run(_run)

    def test_collection_inventory_alias_returns_normalized_shape(self) -> None:
        async def _run() -> None:
            result = await list_storage_collections(include_metadata=False, include_timestamps=False)
            self.assertEqual(result.get("status"), "success")
            self.assertIn("collections", result)
            self.assertIn("total_count", result)
            collections = result.get("collections") or []
            self.assertIsInstance(collections, list)
            if collections:
                first = collections[0] or {}
                self.assertNotIn("metadata", first)
                self.assertNotIn("created_at", first)
                self.assertNotIn("updated_at", first)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
