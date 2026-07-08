#!/usr/bin/env python3
"""UNI-223 manage_collections metadata-key validation parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    manage_collections,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI223StorageTools(unittest.TestCase):
    def test_manage_collections_schema_enforces_metadata_property_names(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        schema = by_name["manage_collections"]["input_schema"]
        metadata_schema = (schema.get("properties", {}).get("metadata") or {})
        property_names = metadata_schema.get("propertyNames") or {}
        self.assertEqual(property_names.get("minLength"), 1)

    def test_manage_collections_rejects_empty_metadata_key(self) -> None:
        async def _run() -> None:
            result = await manage_collections(
                action="create",
                collection_name="uni223-temp",
                metadata={"": "invalid"},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "metadata keys must be non-empty strings when provided",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_manage_collections_accepts_non_empty_metadata_key(self) -> None:
        async def _run() -> None:
            result = await manage_collections(
                action="create",
                collection_name="uni223-valid",
                metadata={"owner": "uni223"},
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "create")
            self.assertTrue(result.get("success"))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
