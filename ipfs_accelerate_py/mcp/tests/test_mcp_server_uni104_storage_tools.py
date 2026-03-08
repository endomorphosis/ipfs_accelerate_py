#!/usr/bin/env python3
"""UNI-104 storage tools parity and validation tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_stats,
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

            result = await store_data(data={"x": 1}, collection="")
            self.assertEqual(result.get("stored"), False)
            self.assertIn("collection must be a non-empty string when provided", str(result.get("error", "")))

        anyio.run(_run)

    def test_retrieve_data_requires_item_ids(self) -> None:
        async def _run() -> None:
            result = await retrieve_data(item_ids=[])
            self.assertEqual(result.get("retrieved_count"), 0)
            self.assertIn("At least one item ID", str(result.get("error", "")))

            result = await retrieve_data(item_ids=["item-1"], format_type="")
            self.assertEqual(result.get("retrieved_count"), 0)
            self.assertIn("format_type must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_manage_collections_rejects_unknown_action(self) -> None:
        async def _run() -> None:
            result = await manage_collections(action="rename", collection_name="a")
            self.assertEqual(result.get("success"), False)
            self.assertIn("Unknown action", str(result.get("error", "")))

            result = await manage_collections(action="stats", collection_name="")
            self.assertEqual(result.get("success"), False)
            self.assertIn("collection_name must be a non-empty string when provided", str(result.get("error", "")))

        anyio.run(_run)

    def test_get_storage_stats_rejects_empty_collection_name(self) -> None:
        async def _run() -> None:
            result = await get_storage_stats(collection_name="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("collection_name must be a non-empty string when provided", str(result.get("error", "")))

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

        self.assertEqual(len(manager.calls), 14)

        schemas = {call["name"]: call["input_schema"]["properties"] for call in manager.calls}
        self.assertEqual(
            set(schemas),
            {
                "store_data",
                "retrieve_data",
                "manage_collections",
                "query_storage",
                "list_storage",
                "get_storage_stats",
                "get_storage_collection_stats",
                "get_storage_lifecycle_report",
                "get_storage_backend_status",
                "list_storage_collections",
                "create_storage_collection",
                "get_storage_collection",
                "delete_storage_collection",
                "delete_data",
            },
        )

        store_schema = schemas["store_data"]
        self.assertIn("enum", store_schema["storage_type"])
        self.assertIn("enum", store_schema["compression"])

        manage_schema = schemas["manage_collections"]
        self.assertIn("enum", manage_schema["action"])
        self.assertIn("enum", manage_schema["report_format"])

        self.assertIn("storage_type", schemas["query_storage"])
        self.assertIn("report_format", schemas["get_storage_stats"])
        self.assertIn("report_format", schemas["get_storage_collection_stats"])
        self.assertIn("collection_name", schemas["create_storage_collection"])
        self.assertIn("include_metadata", schemas["get_storage_collection"])
        self.assertIn("delete_items", schemas["delete_storage_collection"])
        self.assertIn("item_ids", schemas["delete_data"])


if __name__ == "__main__":
    unittest.main()
