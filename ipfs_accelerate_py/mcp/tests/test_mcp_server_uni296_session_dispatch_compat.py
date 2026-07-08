#!/usr/bin/env python3
"""UNI-296 session-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerUNI296SessionDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_session_dispatch_preserves_enhanced_validation_and_success_contracts(
        self, mock_wrapper
    ) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(
                self,
                name,
                function,
                description,
                input_schema,
                execution_context=None,
                tags=None,
            ):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                server = create_mcp_server(name="session-dispatch-compat")
                dispatch = server.tools["tools_dispatch"]["function"]

                created = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "create_session",
                        {
                            "session_name": "uni296-session",
                            "user_id": "uni296-user",
                            "session_type": "batch",
                            "metadata": {"purpose": "dispatch-compat"},
                            "tags": ["uni296", "session"],
                        },
                    )
                )
                self.assertEqual(created.get("status"), "success")
                session_id = created.get("session_id")
                self.assertIsInstance(session_id, str)
                self.assertEqual(created.get("session_type"), "batch")
                self.assertEqual(created.get("metadata", {}).get("purpose"), "dispatch-compat")

                managed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "manage_session",
                        {
                            "action": "get",
                            "session_id": session_id,
                        },
                    )
                )
                self.assertEqual(managed.get("status"), "success")
                self.assertEqual((managed.get("session") or {}).get("session_id"), session_id)

                state = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "get_session_state",
                        {
                            "session_id": session_id,
                            "include_metrics": True,
                            "include_resources": True,
                            "include_health": True,
                        },
                    )
                )
                self.assertEqual(state.get("status"), "success")
                session_state = state.get("session_state") or {}
                self.assertEqual(session_state.get("session_id"), session_id)
                self.assertIn("metrics", session_state)
                self.assertIn("resource_usage", session_state)
                self.assertIn("health_info", session_state)

                invalid_cleanup = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "manage_session",
                        {
                            "action": "cleanup",
                            "cleanup_options": {"max_age_hours": 6, "dry_run": "yes"},
                        },
                    )
                )
                self.assertEqual(invalid_cleanup.get("status"), "error")
                self.assertEqual(invalid_cleanup.get("code"), "INVALID_CLEANUP_OPTIONS")
                self.assertIn(
                    "cleanup_options.dry_run must be a boolean",
                    str(invalid_cleanup.get("error", "")),
                )

                invalid_state = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "get_session_state",
                        {
                            "session_id": "not-a-uuid",
                        },
                    )
                )
                self.assertEqual(invalid_state.get("status"), "error")
                self.assertEqual(invalid_state.get("code"), "INVALID_SESSION_ID")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_session_dispatch_infers_error_status_from_contradictory_delegate_payloads(
        self, mock_wrapper
    ) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(
                self,
                name,
                function,
                description,
                input_schema,
                execution_context=None,
                tags=None,
            ):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class _ContradictoryManager:
            async def create_session(self, **kwargs):
                return {"status": "success", "success": False, "error": "create failed"}

            async def get_session(self, session_id: str):
                return {"status": "success", "success": False, "error": "lookup failed"}

            async def update_session(self, session_id: str, **kwargs):
                return {"status": "success", "success": False, "error": "update failed"}

            async def delete_session(self, session_id: str):
                return {"status": "success", "success": False, "error": "delete failed"}

            async def list_sessions(self, **filters):
                return {"status": "success", "success": False, "error": "list failed"}

            async def cleanup_expired_sessions(self, max_age_hours: int = 24):
                return {"status": "success", "success": False, "error": "cleanup failed"}

        mock_wrapper.return_value = DummyServer()

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch(
                "ipfs_accelerate_py.mcp_server.tools.session_tools.native_session_tools._get_session_manager",
                return_value=_ContradictoryManager(),
            ):
                server = create_mcp_server(name="session-dispatch-contradictory")
                dispatch = server.tools["tools_dispatch"]["function"]

                created = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "create_session",
                        {"session_name": "uni296-contradictory"},
                    )
                )
                self.assertEqual(created.get("status"), "error")
                self.assertFalse(created.get("success"))
                self.assertEqual(created.get("error"), "create failed")

                managed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "manage_session",
                        {
                            "action": "get",
                            "session_id": "11111111-1111-1111-1111-111111111111",
                        },
                    )
                )
                self.assertEqual(managed.get("status"), "error")
                self.assertFalse(managed.get("success"))
                self.assertEqual(managed.get("error"), "lookup failed")

                listed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "manage_session",
                        {"action": "list"},
                    )
                )
                self.assertEqual(listed.get("status"), "error")
                self.assertFalse(listed.get("success"))
                self.assertEqual(listed.get("error"), "list failed")

                state = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "session_tools",
                        "get_session_state",
                        {"session_id": "11111111-1111-1111-1111-111111111111"},
                    )
                )
                self.assertEqual(state.get("status"), "error")
                self.assertFalse(state.get("success"))
                self.assertEqual(state.get("error"), "lookup failed")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
