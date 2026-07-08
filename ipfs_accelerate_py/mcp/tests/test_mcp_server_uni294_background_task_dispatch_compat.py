#!/usr/bin/env python3
"""UNI-294 background-task dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.background_task_tools import native_background_task_tools


class TestMCPServerUNI294BackgroundTaskDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_background_task_dispatch_infers_error_status_from_contradictory_delegate_payloads(
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
                native_background_task_tools._API,
                {
                    "check_task_status": _contradictory_failure,
                    "manage_background_tasks": _contradictory_failure,
                    "manage_task_queue": _contradictory_failure,
                    "get_task_status": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="background-task-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                checked = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "background_task_tools",
                        "check_task_status",
                        {"task_id": "task-1", "task_type": "all", "status_filter": "all", "limit": 10},
                    )
                )
                self.assertEqual(checked.get("status"), "error")
                self.assertEqual(checked.get("error"), "delegate failure")

                managed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "background_task_tools",
                        "manage_background_tasks",
                        {"action": "list", "task_type": "create_embeddings", "priority": "high"},
                    )
                )
                self.assertEqual(managed.get("status"), "error")
                self.assertEqual(managed.get("error"), "delegate failure")

                queued = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "background_task_tools",
                        "manage_task_queue",
                        {"action": "get_stats"},
                    )
                )
                self.assertEqual(queued.get("status"), "error")
                self.assertEqual(queued.get("error"), "delegate failure")

                detailed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "background_task_tools",
                        "get_task_status",
                        {
                            "task_id": "task-1",
                            "include_logs": False,
                            "include_system_status": True,
                            "include_queue_status": True,
                            "log_limit": 5,
                        },
                    )
                )
                self.assertEqual(detailed.get("status"), "error")
                self.assertEqual(detailed.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
