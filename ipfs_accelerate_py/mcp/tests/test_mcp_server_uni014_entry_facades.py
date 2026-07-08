#!/usr/bin/env python3
"""Compatibility facade tests for UNI-014 entrypoint/client surfaces."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.client import IPFSDatasetsMCPClient
from ipfs_accelerate_py.mcp_server.simple_server import (
    SimpleCallResult,
    SimpleIPFSDatasetsMCPServer,
    start_simple_server,
)


class TestMCPServerUNI014EntryFacades(unittest.TestCase):
    @patch("ipfs_accelerate_py.mcp_server.simple_server.run_server")
    def test_start_simple_server_delegates(self, mock_run_server) -> None:
        start_simple_server()
        mock_run_server.assert_called_once_with()

    @patch("ipfs_accelerate_py.mcp_server.simple_server.run_server")
    def test_simple_server_class_run_delegates(self, mock_run_server) -> None:
        server = SimpleIPFSDatasetsMCPServer()
        server.run(host="127.0.0.1", port=8891)
        mock_run_server.assert_called_once_with(host="127.0.0.1", port=8891)

    def test_simple_call_result_contract(self) -> None:
        ok = SimpleCallResult(result={"ok": True})
        self.assertEqual(ok.to_dict(), {"success": True, "result": {"ok": True}})
        err = SimpleCallResult(result=None, error="boom")
        self.assertEqual(err.to_dict(), {"success": False, "error": "boom"})

    def test_client_call_tool_delegates_to_manifest(self) -> None:
        async def _run() -> None:
            client = IPFSDatasetsMCPClient("local://", mcp_like=object())
            with patch(
                "ipfs_accelerate_py.tool_manifest.invoke_mcp_tool",
                new=AsyncMock(return_value={"ok": True, "tool": "echo"}),
            ) as mock_invoke:
                result = await client.call_tool("echo", {"value": "ok"})

            self.assertEqual(result, {"ok": True, "tool": "echo"})
            mock_invoke.assert_awaited_once()

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
