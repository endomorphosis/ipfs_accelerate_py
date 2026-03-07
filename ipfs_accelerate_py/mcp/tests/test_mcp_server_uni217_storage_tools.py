#!/usr/bin/env python3
"""UNI-217 storage collection-delete alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    delete_storage_collection,
    manage_collections,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI217StorageTools(unittest.TestCase):
    def test_schema_registers_collection_delete_alias_tool(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("delete_storage_collection", by_name)
        schema = by_name["delete_storage_collection"]["input_schema"]
        self.assertEqual(schema.get("required"), ["collection_name"])
        props = schema.get("properties", {})
        self.assertEqual((props.get("delete_items") or {}).get("default"), False)

    def test_collection_delete_alias_requires_collection_name(self) -> None:
        async def _run() -> None:
            result = await delete_storage_collection(collection_name="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("collection_name is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_collection_delete_alias_deletes_collection(self) -> None:
        async def _run() -> None:
            await manage_collections(action="create", collection_name="uni217-temp")
            result = await delete_storage_collection(collection_name="uni217-temp", delete_items=True)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("collection_name"), "uni217-temp")
            self.assertTrue(result.get("deleted"))
            self.assertEqual(result.get("action"), "delete")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
