#!/usr/bin/env python3
"""UNI-175 backend filter/availability parity tests for storage tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI175StorageTools(unittest.TestCase):
    def test_manage_collections_schema_exposes_backend_filter_fields(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_props = (schemas["manage_collections"].get("properties") or {})
        self.assertIn("backend_types", manage_props)
        self.assertIn("unavailable_backends", manage_props)

    def test_backend_status_supports_filters_and_unavailable_marking(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            report = result.get("backend_report") or {}
            self.assertEqual(report.get("backend_count"), 2)

            backends = report.get("backends") or []
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual(set(by_type.keys()), {"memory", "ipfs"})
            self.assertEqual((by_type.get("memory") or {}).get("available"), True)
            self.assertEqual((by_type.get("ipfs") or {}).get("available"), False)

            breakdown = report.get("breakdown") or {}
            self.assertEqual(breakdown.get("available_count"), 1)
            self.assertEqual(breakdown.get("unavailable_count"), 1)

        anyio.run(_run)

    def test_backend_status_rejects_unknown_backend_filter(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                backend_types=["memory", "tape"],
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("backend_types contains unknown storage backends", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
