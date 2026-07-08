#!/usr/bin/env python3
"""UNI-168 storage schema non-empty string parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI168StorageTools(unittest.TestCase):
    def test_storage_schema_requires_non_empty_item_ids_and_tags(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        retrieve_props = schemas["retrieve_data"]["properties"]
        self.assertEqual((retrieve_props.get("item_ids") or {}).get("items", {}).get("minLength"), 1)

        store_props = schemas["store_data"]["properties"]
        self.assertEqual((store_props.get("tags") or {}).get("items", {}).get("minLength"), 1)

        query_props = schemas["query_storage"]["properties"]
        self.assertEqual((query_props.get("tags") or {}).get("items", {}).get("minLength"), 1)

    def test_retrieve_data_rejects_empty_item_id(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.retrieve_data(item_ids=["id-1", ""])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("item_ids must be an array of non-empty strings", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
