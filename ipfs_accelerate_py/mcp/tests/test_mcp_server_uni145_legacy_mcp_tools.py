#!/usr/bin/env python3
"""UNI-145 legacy_mcp_tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.legacy_mcp_tools import native_legacy_mcp_tools as legacy_mod


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI145LegacyMcpTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        legacy_mod.register_native_legacy_mcp_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        schema = by_name["legacy_tools_inventory"]["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], [])

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            result = await legacy_mod.legacy_tools_inventory()
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertIs(result.get("deprecated"), True)
            self.assertIn("temporal_deontic_tool_count", result)

        anyio.run(_run)

    def test_error_envelope_shape_on_api_failure(self) -> None:
        async def _run() -> None:
            class _ExplodingAPI:
                def get(self, *_args, **_kwargs):
                    raise RuntimeError("boom")

            with patch.object(legacy_mod, "_API", new=_ExplodingAPI()):
                result = await legacy_mod.legacy_tools_inventory()
            self.assertEqual(result.get("status"), "error")
            self.assertIn("boom", str(result.get("error", "")))
            self.assertIs(result.get("deprecated"), True)

        anyio.run(_run)

    def test_inventory_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch.object(legacy_mod, "_normalize_payload", return_value={"status": "success", "deprecated": True}):
                result = await legacy_mod.legacy_tools_inventory()

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("temporal_deontic_tool_count"), 0)
            self.assertIs(result.get("deprecated"), True)

        anyio.run(_run)

    def test_normalize_payload_infers_error_status_from_contradictory_delegate_payload(self) -> None:
        result = legacy_mod._normalize_payload(
            {"status": "success", "success": False, "error": "delegate failure", "deprecated": True}
        )

        self.assertEqual(result.get("status"), "error")
        self.assertEqual(result.get("error"), "delegate failure")
        self.assertIs(result.get("deprecated"), True)


if __name__ == "__main__":
    unittest.main()
