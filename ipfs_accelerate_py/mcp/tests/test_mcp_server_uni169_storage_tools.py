#!/usr/bin/env python3
"""UNI-169 storage schema cardinality/range parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI169StorageTools(unittest.TestCase):
    def test_storage_schema_enforces_item_id_count_and_non_negative_size_range(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        retrieve_props = schemas["retrieve_data"]["properties"]
        self.assertEqual((retrieve_props.get("item_ids") or {}).get("minItems"), 1)

        query_props = schemas["query_storage"]["properties"]
        size_range_items = (query_props.get("size_range") or {}).get("items", {})
        self.assertEqual(size_range_items.get("minimum"), 0)

    def test_retrieve_data_rejects_empty_item_id_list(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.retrieve_data(item_ids=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("At least one item ID must be provided", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
