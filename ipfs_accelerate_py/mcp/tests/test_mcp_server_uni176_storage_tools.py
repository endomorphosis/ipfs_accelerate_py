#!/usr/bin/env python3
"""UNI-176 backend unavailable-reason parity tests for storage tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI176StorageTools(unittest.TestCase):
    def test_manage_collections_schema_exposes_unavailable_reasons(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_props = (schemas["manage_collections"].get("properties") or {})
        self.assertIn("unavailable_reasons", manage_props)
        reason_schema = manage_props["unavailable_reasons"]
        self.assertIn("object", reason_schema.get("type") or [])

    def test_backend_status_includes_unavailable_reason_for_unavailable_backend(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                backend_types=["ipfs", "memory"],
                unavailable_backends=["ipfs"],
                unavailable_reasons={"ipfs": "dial timeout"},
            )
            self.assertEqual(result.get("status"), "success")
            backends = ((result.get("backend_report") or {}).get("backends") or [])
            by_type = {item.get("storage_type"): item for item in backends}
            self.assertEqual((by_type.get("ipfs") or {}).get("unavailable_reason"), "dial timeout")
            self.assertIsNone((by_type.get("memory") or {}).get("unavailable_reason"))

        anyio.run(_run)

    def test_backend_status_rejects_unknown_unavailable_reason_backend(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                unavailable_reasons={"tape": "unsupported backend"},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_reasons contains unknown storage backends",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
