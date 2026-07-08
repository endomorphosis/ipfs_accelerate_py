#!/usr/bin/env python3
"""UNI-177 backend availability-filter parity tests for storage tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI177StorageTools(unittest.TestCase):
    def test_manage_collections_schema_exposes_availability_filter(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_props = (schemas["manage_collections"].get("properties") or {})
        self.assertIn("availability_filter", manage_props)
        filter_schema = manage_props["availability_filter"]
        self.assertEqual(filter_schema.get("default"), "all")
        self.assertIn("available", filter_schema.get("enum") or [])
        self.assertIn("unavailable", filter_schema.get("enum") or [])

    def test_backend_status_filters_to_unavailable_entries(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                backend_types=["memory", "ipfs", "s3"],
                unavailable_backends=["ipfs", "s3"],
                availability_filter="unavailable",
            )
            self.assertEqual(result.get("status"), "success")

            report = result.get("backend_report") or {}
            self.assertEqual(report.get("availability_filter"), "unavailable")
            backends = report.get("backends") or []
            self.assertEqual(len(backends), 2)
            self.assertTrue(all(not item.get("available") for item in backends))

        anyio.run(_run)

    def test_backend_status_rejects_invalid_availability_filter(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                availability_filter="partial",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("availability_filter must be one of", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
