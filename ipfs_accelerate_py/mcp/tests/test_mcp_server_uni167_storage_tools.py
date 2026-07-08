#!/usr/bin/env python3
"""UNI-167 deterministic date-range parity tests for native storage tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI167StorageTools(unittest.TestCase):
    def test_query_storage_schema_includes_date_time_format(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        query_props = schemas["query_storage"]["properties"]
        date_range_schema = query_props["date_range"]
        self.assertEqual((date_range_schema.get("items") or {}).get("format"), "date-time")

    def test_query_storage_rejects_invalid_date_range_values(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.query_storage(
                date_range=["not-a-date", "2026-01-01T00:00:00Z"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "date_range values must be valid ISO-8601 datetime strings",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_query_storage_rejects_unordered_date_range(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.query_storage(
                date_range=["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("date_range must be ordered as [start, end]", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
