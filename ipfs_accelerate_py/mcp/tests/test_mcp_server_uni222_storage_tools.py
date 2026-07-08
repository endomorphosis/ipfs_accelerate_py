#!/usr/bin/env python3
"""UNI-222 manage_collections description validation parity tests."""

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


class TestMCPServerUNI222StorageTools(unittest.TestCase):
    def test_manage_collections_schema_enforces_non_empty_description(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        schema = by_name["manage_collections"]["input_schema"]
        props = schema.get("properties", {})
        self.assertEqual((props.get("description") or {}).get("minLength"), 1)

    def test_manage_collections_rejects_empty_description(self) -> None:
        async def _run() -> None:
            result = await manage_collections(
                action="create",
                collection_name="uni222-temp",
                description="",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "description must be a non-empty string when provided",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_manage_collections_accepts_non_empty_description(self) -> None:
        async def _run() -> None:
            result = await manage_collections(
                action="create",
                collection_name="uni222-valid",
                description="uni222 valid",
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "create")
            self.assertTrue(result.get("success"))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
