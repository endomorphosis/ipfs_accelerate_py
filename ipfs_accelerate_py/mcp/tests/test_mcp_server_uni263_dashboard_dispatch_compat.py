#!/usr/bin/env python3
"""UNI-263 dashboard dispatch compatibility tests for focused parity coverage."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.dashboard_tools import native_dashboard_tools


class TestMCPServerUNI263DashboardDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_dashboard_dispatch_infers_error_status_from_contradictory_delegate_payloads(self, mock_wrapper) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        def _contradictory_failure(*_: object, **__: object) -> dict:
            return {"status": "success", "success": False, "error": "dashboard delegate failure"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_dashboard_tools._API,
                {
                    "get_tdfol_metrics": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="dashboard-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                metrics = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "dashboard_tools",
                        "get_tdfol_metrics",
                        {},
                    )
                )
                self.assertEqual(metrics.get("status"), "error")
                self.assertEqual(metrics.get("error"), "dashboard delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()