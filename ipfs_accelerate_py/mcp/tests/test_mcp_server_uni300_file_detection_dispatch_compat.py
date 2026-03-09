#!/usr/bin/env python3
"""UNI-300 file-detection dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.file_detection_tools import native_file_detection_tools


class TestMCPServerUNI300FileDetectionDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_file_detection_dispatch_infers_error_status_from_contradictory_delegate_payloads(
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

        def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_file_detection_tools._API,
                {
                    "detect_file_type": _contradictory_failure,
                    "batch_detect_file_types": _contradictory_failure,
                    "analyze_detection_accuracy": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="file-detection-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                detected = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "file_detection_tools",
                        "detect_file_type",
                        {"file_path": "/tmp/x.txt", "strategy": "accurate"},
                    )
                )
                self.assertEqual(detected.get("status"), "error")
                self.assertEqual(detected.get("success"), False)
                self.assertEqual(detected.get("error"), "delegate failed")

                batched = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "file_detection_tools",
                        "batch_detect_file_types",
                        {"file_paths": ["/tmp/x.txt"], "pattern": "*"},
                    )
                )
                self.assertEqual(batched.get("status"), "error")
                self.assertEqual(batched.get("success"), False)
                self.assertEqual(batched.get("error"), "delegate failed")

                analyzed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "file_detection_tools",
                        "analyze_detection_accuracy",
                        {"directory": "/tmp", "pattern": "*"},
                    )
                )
                self.assertEqual(analyzed.get("status"), "error")
                self.assertEqual(analyzed.get("success"), False)
                self.assertEqual(analyzed.get("error"), "delegate failed")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
