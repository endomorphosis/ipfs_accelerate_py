#!/usr/bin/env python3
"""UNI-135 p2p tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.p2p_tools.native_p2p_tools import (
    p2p_cache_get,
    p2p_remote_call_tool,
    p2p_remote_status,
    p2p_task_submit,
    register_native_p2p_tools_category,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI135P2PTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_p2p_tools_category(manager)
        by_name = {c["name"]: c for c in manager.calls}

        status_schema = by_name["p2p_service_status"]["input_schema"]
        self.assertEqual(status_schema["properties"]["peers_limit"].get("minimum"), 1)

        call_schema = by_name["p2p_remote_call_tool"]["input_schema"]
        props = call_schema["properties"]
        self.assertEqual(props["tool_name"].get("minLength"), 1)
        self.assertIn("exclusiveMinimum", props["timeout_s"])

    def test_p2p_cache_get_rejects_blank_key(self) -> None:
        async def _run() -> None:
            result = await p2p_cache_get(key="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("key must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_p2p_task_submit_rejects_non_object_payload(self) -> None:
        async def _run() -> None:
            result = await p2p_task_submit(task_type="demo", payload=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("payload must be an object", str(result.get("error", "")))

        anyio.run(_run)

    def test_p2p_remote_status_rejects_nonpositive_timeout(self) -> None:
        async def _run() -> None:
            result = await p2p_remote_status(timeout_s=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout_s must be a number > 0", str(result.get("error", "")))

        anyio.run(_run)

    def test_p2p_remote_call_tool_rejects_blank_tool_name(self) -> None:
        async def _run() -> None:
            result = await p2p_remote_call_tool(tool_name="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("tool_name must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_p2p_cache_get_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await p2p_cache_get(key="smoke")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("key"), "smoke")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
