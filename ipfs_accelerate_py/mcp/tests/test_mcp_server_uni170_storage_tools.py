#!/usr/bin/env python3
"""UNI-170 storage date-range schema/string parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI170StorageTools(unittest.TestCase):
    def test_storage_schema_enforces_non_empty_date_range_strings(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        query_props = schemas["query_storage"]["properties"]
        date_range_items = (query_props.get("date_range") or {}).get("items", {})
        self.assertEqual(date_range_items.get("minLength"), 1)

    def test_query_storage_rejects_empty_date_range_value(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.query_storage(
                date_range=["", "2026-01-01T00:00:00Z"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("date_range values must be non-empty strings", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
