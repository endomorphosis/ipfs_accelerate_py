#!/usr/bin/env python3
"""UNI-104 storage tools parity and validation tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    manage_collections,
    query_storage,
    register_native_storage_tools,
    retrieve_data,
    store_data,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI104StorageTools(unittest.TestCase):
    def test_store_data_rejects_invalid_storage_type(self) -> None:
        async def _run() -> None:
            result = await store_data(data={"x": 1}, storage_type="tape")
            self.assertEqual(result.get("stored"), False)
            self.assertIn("Invalid storage type", str(result.get("error", "")))

        anyio.run(_run)

    def test_retrieve_data_requires_item_ids(self) -> None:
        async def _run() -> None:
            result = await retrieve_data(item_ids=[])
            self.assertEqual(result.get("retrieved_count"), 0)
            self.assertIn("At least one item ID", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_collections_rejects_unknown_action(self) -> None:
        async def _run() -> None:
            result = await manage_collections(action="rename", collection_name="a")
            self.assertEqual(result.get("success"), False)
            self.assertIn("Unknown action", str(result.get("error", "")))

        anyio.run(_run)

    def test_query_storage_rejects_invalid_storage_type(self) -> None:
        async def _run() -> None:
            result = await query_storage(storage_type="ssd")
            self.assertEqual(result.get("total_found"), 0)
            self.assertIn("Invalid storage type", str(result.get("error", "")))

        anyio.run(_run)

    def test_register_schema_includes_validation_enums(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        self.assertEqual(len(manager.calls), 4)

        store_schema = manager.calls[0]["input_schema"]["properties"]
        self.assertIn("enum", store_schema["storage_type"])
        self.assertIn("enum", store_schema["compression"])

        manage_schema = manager.calls[2]["input_schema"]["properties"]
        self.assertIn("enum", manage_schema["action"])


if __name__ == "__main__":
    unittest.main()
