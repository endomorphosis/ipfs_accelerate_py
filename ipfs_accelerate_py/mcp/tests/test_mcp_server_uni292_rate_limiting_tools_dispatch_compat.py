#!/usr/bin/env python3
"""UNI-292 rate-limiting-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.rate_limiting_tools import native_rate_limiting_tools_category


class TestMCPServerUNI292RateLimitingToolsDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_rate_limiting_tools_dispatch_infers_error_status_from_contradictory_delegate_payloads(
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

        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failure"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_rate_limiting_tools_category._API,
                {
                    "configure_rate_limits": _contradictory_failure,
                    "check_rate_limit": _contradictory_failure,
                    "manage_rate_limits": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="rate-limiting-tools-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                configured = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting_tools",
                        "configure_rate_limits",
                        {"limits": [{"name": "api", "requests": 10}]},
                    )
                )
                self.assertEqual(configured.get("status"), "error")
                self.assertEqual(configured.get("error"), "delegate failure")

                checked = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting_tools",
                        "check_rate_limit",
                        {"limit_name": "api", "identifier": "client-a"},
                    )
                )
                self.assertEqual(checked.get("status"), "error")
                self.assertEqual(checked.get("error"), "delegate failure")

                managed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "rate_limiting_tools",
                        "manage_rate_limits",
                        {"action": "stats", "limit_name": "api"},
                    )
                )
                self.assertEqual(managed.get("status"), "error")
                self.assertEqual(managed.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
