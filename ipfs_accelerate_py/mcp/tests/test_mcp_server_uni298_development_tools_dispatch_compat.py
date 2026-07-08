#!/usr/bin/env python3
"""UNI-298 development-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.development_tools import native_development_tools


class TestMCPServerUNI298DevelopmentToolsDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_development_tools_dispatch_infers_error_status_from_contradictory_delegate_payloads(
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
                native_development_tools._API,
                {
                    "codebase_search": _contradictory_failure,
                    "documentation_generator": _contradictory_failure,
                    "run_comprehensive_tests": _contradictory_failure,
                    "vscode_cli_execute": _contradictory_failure,
                    "vscode_cli_status": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="development-tools-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                searched = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "development_tools",
                        "codebase_search",
                        {"pattern": "README", "path": "src"},
                    )
                )
                self.assertEqual(searched.get("status"), "error")
                self.assertEqual(searched.get("success"), False)
                self.assertEqual(searched.get("error"), "delegate failure")

                documented = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "development_tools",
                        "documentation_generator",
                        {"input_path": "src", "output_path": "docs"},
                    )
                )
                self.assertEqual(documented.get("status"), "error")
                self.assertEqual(documented.get("success"), False)
                self.assertEqual(documented.get("error"), "delegate failure")

                tested = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "development_tools",
                        "run_comprehensive_tests",
                        {"path": ".", "test_framework": "pytest"},
                    )
                )
                self.assertEqual(tested.get("status"), "error")
                self.assertEqual(tested.get("success"), False)
                self.assertEqual(tested.get("error"), "delegate failure")

                executed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "development_tools",
                        "vscode_cli_execute",
                        {"command": ["--version"], "timeout": 30},
                    )
                )
                self.assertEqual(executed.get("status"), "error")
                self.assertEqual(executed.get("success"), False)
                self.assertEqual(executed.get("error"), "delegate failure")

                status = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "development_tools",
                        "vscode_cli_status",
                        {"install_dir": "/opt/code"},
                    )
                )
                self.assertEqual(status.get("status"), "error")
                self.assertEqual(status.get("success"), False)
                self.assertEqual(status.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
