#!/usr/bin/env python3
"""UNI-172 manage_collections conditional non-null schema parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI172StorageTools(unittest.TestCase):
    def test_manage_collections_schema_requires_non_null_collection_name_for_subset(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_schema = schemas["manage_collections"]
        all_of = manage_schema.get("allOf") or []
        self.assertGreaterEqual(len(all_of), 1)

        first_rule = all_of[0]
        then_props = (first_rule.get("then") or {}).get("properties") or {}
        collection_schema = then_props.get("collection_name") or {}
        self.assertEqual(collection_schema.get("type"), "string")
        self.assertEqual(collection_schema.get("minLength"), 1)

    def test_manage_collections_rejects_null_collection_name_for_create(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(action="create", collection_name=None)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("collection_name required for create action", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
