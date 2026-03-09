#!/usr/bin/env python3
"""UNI-294 logic-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.logic_tools import native_logic_tools


class TestMCPServerUNI294LogicDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_logic_dispatch_infers_error_status_from_contradictory_delegate_payloads(self, mock_wrapper) -> None:
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
                native_logic_tools._API,
                {
                    "tdfol_parse": _contradictory_failure,
                    "tdfol_prove": _contradictory_failure,
                    "cec_prove": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="logic-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                parsed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "logic_tools",
                        "tdfol_parse",
                        {"text": "forall x P(x)"},
                    )
                )
                self.assertEqual(parsed.get("status"), "error")
                self.assertEqual(parsed.get("success"), False)
                self.assertEqual(parsed.get("error"), "delegate failure")

                proved = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "logic_tools",
                        "tdfol_prove",
                        {"formula": "forall x P(x)"},
                    )
                )
                self.assertEqual(proved.get("status"), "error")
                self.assertEqual(proved.get("success"), False)
                self.assertEqual(proved.get("error"), "delegate failure")

                cec_proved = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "logic_tools",
                        "cec_prove",
                        {"goal": "P(a)"},
                    )
                )
                self.assertEqual(cec_proved.get("status"), "error")
                self.assertEqual(cec_proved.get("success"), False)
                self.assertEqual(cec_proved.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
