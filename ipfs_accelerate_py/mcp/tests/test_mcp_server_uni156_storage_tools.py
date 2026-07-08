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

        list_props = schemas["list_storage"]["properties"]
        self.assertEqual(list_props["limit"].get("minimum"), 1)
        self.assertEqual(list_props["offset"].get("minimum"), 0)

        stats_props = schemas["get_storage_stats"]["properties"]
        self.assertEqual(stats_props["report_format"].get("default"), "summary")

        delete_props = schemas["delete_data"]["properties"]
        self.assertEqual(delete_props["item_ids"].get("minItems"), 1)
        self.assertEqual(delete_props["missing_ok"].get("default"), False)

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
            self.assertIn(
                "collection_name must be a non-empty string when provided",
                str(invalid_collection.get("error", "")),
            )

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

    def test_storage_alias_tools_validate_and_normalize(self) -> None:
        async def _run() -> None:
            invalid_list_limit = await native_storage_tools.list_storage(limit=0)
            self.assertEqual(invalid_list_limit.get("status"), "error")
            self.assertIn("limit must be a positive integer", str(invalid_list_limit.get("error", "")))

            invalid_stats_format = await native_storage_tools.get_storage_stats(report_format="xml")
            self.assertEqual(invalid_stats_format.get("status"), "error")
            self.assertIn("report_format must be one of", str(invalid_stats_format.get("error", "")))

            invalid_delete_ids = await native_storage_tools.delete_data(item_ids=["id-1", ""])
            self.assertEqual(invalid_delete_ids.get("status"), "error")
            self.assertIn("item_ids must be an array of non-empty strings", str(invalid_delete_ids.get("error", "")))

            stored = await native_storage_tools.store_data(data={"x": 1}, collection="default")
            item_id = stored.get("item_id")
            self.assertIsInstance(item_id, str)

            listing = await native_storage_tools.list_storage(limit=5)
            self.assertEqual(listing.get("status"), "success")
            self.assertIn("objects", listing)

            stats = await native_storage_tools.get_storage_stats(report_format="summary")
            self.assertEqual(stats.get("status"), "success")
            self.assertIn("total_objects", stats)
            self.assertIn("total_bytes", stats)

            deleted = await native_storage_tools.delete_data(item_ids=[str(item_id)])
            self.assertEqual(deleted.get("status"), "success")
            self.assertEqual(deleted.get("deleted_count"), 1)

            missing = await native_storage_tools.delete_data(item_ids=[str(item_id)], missing_ok=True)
            self.assertEqual(missing.get("status"), "success")
            self.assertEqual(missing.get("missing_count"), 1)

        anyio.run(_run)

    def test_failed_delegate_payloads_infer_error_status(self) -> None:
        async def _failed(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_storage_tools._API,
                {
                    "store_data": _failed,
                    "retrieve_data": _failed,
                    "manage_collections": _failed,
                    "query_storage": _failed,
                },
                clear=False,
            ):
                stored = await native_storage_tools.store_data(data={"x": 1})
                self.assertEqual(stored.get("status"), "error")

                retrieved = await native_storage_tools.retrieve_data(item_ids=["item-1"])
                self.assertEqual(retrieved.get("status"), "error")

                managed = await native_storage_tools.manage_collections(action="list")
                self.assertEqual(managed.get("status"), "error")

                queried = await native_storage_tools.query_storage(limit=5)
                self.assertEqual(queried.get("status"), "error")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
