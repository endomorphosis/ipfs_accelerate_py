#!/usr/bin/env python3
"""UNI-156 deterministic parity tests for native storage tools."""

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


class TestMCPServerUNI156StorageTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        store_props = schemas["store_data"]["properties"]
        self.assertEqual(store_props["collection"].get("minLength"), 1)

        query_props = schemas["query_storage"]["properties"]
        self.assertEqual(query_props["limit"].get("minimum"), 1)
        self.assertEqual(query_props["offset"].get("minimum"), 0)

    def test_store_data_validates_metadata_tags_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("store boom")

        async def _run() -> None:
            invalid_tags = await native_storage_tools.store_data(
                data={"x": 1},
                tags=["ok", ""],
            )
            self.assertEqual(invalid_tags.get("status"), "error")
            self.assertIn("tags must be an array of non-empty strings", str(invalid_tags.get("error", "")))

            with patch.dict(native_storage_tools._API, {"store_data": _boom}, clear=False):
                result = await native_storage_tools.store_data(data={"x": 1})
                self.assertEqual(result.get("status"), "error")
                self.assertIn("store_data failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_retrieve_and_manage_validate_inputs(self) -> None:
        async def _run() -> None:
            invalid_item_ids = await native_storage_tools.retrieve_data(item_ids=["id-1", ""])
            self.assertEqual(invalid_item_ids.get("status"), "error")
            self.assertIn("item_ids must be an array of non-empty strings", str(invalid_item_ids.get("error", "")))

            invalid_collection = await native_storage_tools.manage_collections(action="delete", collection_name="   ")
            self.assertEqual(invalid_collection.get("status"), "error")
            self.assertIn("collection_name required", str(invalid_collection.get("error", "")))

        anyio.run(_run)

    def test_query_storage_validates_ranges_and_paging(self) -> None:
        async def _run() -> None:
            invalid_size_range = await native_storage_tools.query_storage(size_range=[10, 1])
            self.assertEqual(invalid_size_range.get("status"), "error")
            self.assertIn("size_range must be non-negative and ordered", str(invalid_size_range.get("error", "")))

            invalid_limit = await native_storage_tools.query_storage(limit=0)
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("limit must be a positive integer", str(invalid_limit.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
