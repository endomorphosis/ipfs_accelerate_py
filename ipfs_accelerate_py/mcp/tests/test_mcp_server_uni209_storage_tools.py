#!/usr/bin/env python3
"""UNI-209 storage backend-status alias parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI209StorageTools(unittest.TestCase):
    def test_schema_registers_backend_status_alias_tool(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}
        self.assertIn("get_storage_backend_status", by_name)
        schema = by_name["get_storage_backend_status"]["input_schema"]
        props = schema.get("properties", {})
        self.assertEqual((props.get("availability_filter") or {}).get("default"), "all")
        self.assertEqual((props.get("include_capabilities") or {}).get("default"), False)

    def test_backend_alias_validates_filter(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(availability_filter="partial")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("availability_filter must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_backend_alias_returns_normalized_shape(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory", "ipfs"],
                unavailable_backends=["ipfs"],
                unavailable_reasons={"ipfs": "dial timeout"},
                availability_filter="unavailable",
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("availability_filter"), "unavailable")
            self.assertEqual(result.get("backend_count"), 1)
            backends = result.get("backends") or []
            self.assertEqual(len(backends), 1)
            self.assertEqual((backends[0] or {}).get("storage_type"), "ipfs")
            self.assertEqual((backends[0] or {}).get("available"), False)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
