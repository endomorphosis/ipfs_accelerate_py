#!/usr/bin/env python3
"""UNI-218 storage collection create/get alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    create_storage_collection,
    get_storage_collection,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI218StorageTools(unittest.TestCase):
    def test_schema_registers_collection_create_get_alias_tools(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("create_storage_collection", by_name)
        self.assertIn("get_storage_collection", by_name)

        create_schema = by_name["create_storage_collection"]["input_schema"]
        self.assertEqual(create_schema.get("required"), ["collection_name"])

        get_schema = by_name["get_storage_collection"]["input_schema"]
        self.assertEqual(get_schema.get("required"), ["collection_name"])
        props = get_schema.get("properties", {})
        self.assertEqual((props.get("include_metadata") or {}).get("default"), True)
        self.assertEqual((props.get("include_timestamps") or {}).get("default"), True)

    def test_collection_create_alias_requires_collection_name(self) -> None:
        async def _run() -> None:
            result = await create_storage_collection(collection_name="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("collection_name is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_collection_create_get_alias_round_trip(self) -> None:
        async def _run() -> None:
            created = await create_storage_collection(
                collection_name="uni218-temp",
                description="uni218 temp collection",
                metadata={"owner": "uni218"},
            )
            self.assertEqual(created.get("status"), "success")
            self.assertEqual(created.get("collection_name"), "uni218-temp")
            self.assertTrue(created.get("created"))

            fetched = await get_storage_collection(
                collection_name="uni218-temp",
                include_metadata=False,
                include_timestamps=False,
            )
            self.assertEqual(fetched.get("status"), "success")
            self.assertEqual(fetched.get("collection_name"), "uni218-temp")
            collection_payload = fetched.get("collection") or {}
            self.assertNotIn("metadata", collection_payload)
            self.assertNotIn("created_at", collection_payload)
            self.assertNotIn("updated_at", collection_payload)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
