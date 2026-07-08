#!/usr/bin/env python3
"""UNI-219 storage collection alias validation hardening tests."""

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


class TestMCPServerUNI219StorageTools(unittest.TestCase):
    def test_create_collection_alias_schema_enforces_non_empty_description(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        schema = by_name["create_storage_collection"]["input_schema"]
        props = schema.get("properties", {})
        self.assertEqual((props.get("description") or {}).get("minLength"), 1)

    def test_create_collection_alias_rejects_empty_description(self) -> None:
        async def _run() -> None:
            result = await create_storage_collection(
                collection_name="uni219-temp",
                description="",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "description must be a non-empty string when provided",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_get_collection_alias_rejects_non_boolean_include_flags(self) -> None:
        async def _run() -> None:
            result = await get_storage_collection(
                collection_name="uni219-temp",
                include_metadata="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("include_metadata must be a boolean", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
