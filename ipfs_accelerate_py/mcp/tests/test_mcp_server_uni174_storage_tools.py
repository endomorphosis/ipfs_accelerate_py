#!/usr/bin/env python3
"""UNI-174 backend-management parity tests for storage tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools import native_storage_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI174StorageTools(unittest.TestCase):
    def test_manage_collections_schema_exposes_backend_controls(self) -> None:
        manager = _DummyManager()
        native_storage_tools.register_native_storage_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        manage_schema = schemas["manage_collections"]
        action_enum = (manage_schema.get("properties") or {}).get("action", {}).get("enum") or []
        self.assertIn("backend_status", action_enum)

        include_capabilities = (manage_schema.get("properties") or {}).get("include_capabilities") or {}
        self.assertEqual(include_capabilities.get("type"), "boolean")

    def test_manage_collections_backend_status_returns_report(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                include_capabilities=True,
                include_breakdown=True,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "backend_status")

            report = result.get("backend_report") or {}
            self.assertGreaterEqual(int(report.get("backend_count", 0)), 1)
            backends = report.get("backends") or []
            self.assertTrue(backends)
            self.assertIn("storage_type", backends[0])
            self.assertIn("capabilities", backends[0])
            self.assertIn("breakdown", report)

        anyio.run(_run)

    def test_manage_collections_rejects_invalid_include_capabilities_type(self) -> None:
        async def _run() -> None:
            result = await native_storage_tools.manage_collections(
                action="backend_status",
                include_capabilities="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("include_capabilities must be a boolean", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
