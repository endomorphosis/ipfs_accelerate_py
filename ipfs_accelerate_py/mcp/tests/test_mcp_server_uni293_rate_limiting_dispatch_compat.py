#!/usr/bin/env python3
"""UNI-293 native rate-limiting dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.rate_limiting import native_rate_limiting_tools


class TestMCPServerUNI293RateLimitingDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_rate_limiting_dispatch_infers_error_status_from_contradictory_delegate_payloads(
        self, mock_wrapper
    ) -> None:
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
        contradictory = {"status": "success", "success": False, "error": "delegate failure"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.object(
                native_rate_limiting_tools._rate_limiter,
                "check_rate_limit",
                return_value=contradictory,
            ), patch.object(
                native_rate_limiting_tools._rate_limiter,
                "get_stats",
                return_value=contradictory,
            ), patch.object(
                native_rate_limiting_tools._rate_limiter,
                "reset_limits",
                return_value=contradictory,
            ):
                server = create_mcp_server(name="rate-limiting-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                checked = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting",
                        "check_rate_limit",
                        {"limit_name": "api", "identifier": "client-a"},
                    )
                )
                self.assertEqual(checked.get("status"), "error")
                self.assertEqual(checked.get("error"), "delegate failure")

                stats = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting",
                        "manage_rate_limits",
                        {"action": "stats", "limit_name": "api"},
                    )
                )
                self.assertEqual(stats.get("status"), "error")
                self.assertEqual(stats.get("error"), "delegate failure")

                reset = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting",
                        "manage_rate_limits",
                        {"action": "reset", "limit_name": "api"},
                    )
                )
                self.assertEqual(reset.get("status"), "error")
                self.assertEqual(reset.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
