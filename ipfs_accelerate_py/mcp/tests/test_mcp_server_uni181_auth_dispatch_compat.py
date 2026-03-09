#!/usr/bin/env python3
"""UNI-181 auth dispatch compatibility tests for enhanced payload shapes."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.auth_tools import native_auth_tools


class TestMCPServerUNI181AuthDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_auth_dispatch_preserves_enhanced_compatibility_fields(self, mock_wrapper) -> None:
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

        async def _authenticate(**_: object) -> dict:
            return {
                "status": "success",
                "username": "demo",
                "access_token": "token-123",
                "token_type": "bearer",
                "role": "user",
                "expires_in": 3600,
            }

        async def _validate(**kwargs: object) -> dict:
            action = kwargs.get("action")
            if action == "decode":
                return {
                    "status": "success",
                    "user_id": "user-1",
                    "username": "demo",
                    "exp": 12345,
                    "role": "admin",
                }
            return {
                "status": "success",
                "valid": True,
                "username": "demo",
                "role": "admin",
                "permissions": ["manage"],
                "expires_in": 7200,
                "has_required_permission": True,
            }

        async def _get_user_info(**_: object) -> dict:
            return {
                "status": "success",
                "username": "demo",
                "role": "admin",
                "permissions": ["manage"],
                "profile": {"team": "infra"},
            }

        async def _run_flow() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools._API",
                {
                    "authenticate_user": _authenticate,
                    "validate_token": _validate,
                    "get_user_info": _get_user_info,
                },
            ), patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                server = create_mcp_server(name="auth-dispatch-compat")

                get_schema = server.tools["tools_get_schema"]["function"]
                dispatch = server.tools["tools_dispatch"]["function"]

                validate_schema = await get_schema("auth_tools", "validate_token")
                props = (validate_schema.get("input_schema") or {}).get("properties", {})
                self.assertEqual((props.get("strict") or {}).get("default"), False)
                self.assertIn("decode", (props.get("action") or {}).get("enum", []))

                auth_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "authenticate_user",
                        {"username": "demo", "password": "pw", "remember_me": True},
                    )
                )
                self.assertEqual((auth_result.get("authentication") or {}).get("access_token"), "token-123")
                self.assertEqual((auth_result.get("authentication") or {}).get("expires_in"), 86400 * 7)

                validate_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "validate_token",
                        {"token": "tok", "required_permission": "manage", "strict": True},
                    )
                )
                self.assertEqual((validate_result.get("validation_result") or {}).get("username"), "demo")
                self.assertEqual(validate_result.get("strict"), True)

                decode_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "validate_token",
                        {"token": "tok", "action": "decode"},
                    )
                )
                self.assertEqual((decode_result.get("decoded_token") or {}).get("user_id"), "user-1")

                info_result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "get_user_info",
                        {"token": "tok", "include_permissions": True, "include_profile": True},
                    )
                )
                self.assertEqual((info_result.get("user_info") or {}).get("profile"), {"team": "infra"})
                self.assertEqual(info_result.get("message"), "User information retrieved successfully")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_auth_dispatch_infers_error_status_from_contradictory_delegate_payloads(self, mock_wrapper) -> None:
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
                native_auth_tools._API,
                {
                    "authenticate_user": _contradictory_failure,
                    "validate_token": _contradictory_failure,
                    "get_user_info": _contradictory_failure,
                },
                clear=False,
            ):
                server = create_mcp_server(name="auth-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                authenticated = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "authenticate_user",
                        {"username": "demo", "password": "pw"},
                    )
                )
                self.assertEqual(authenticated.get("status"), "error")
                self.assertEqual(authenticated.get("success"), False)
                self.assertEqual(authenticated.get("error"), "delegate failure")

                validated = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "validate_token",
                        {"token": "tok"},
                    )
                )
                self.assertEqual(validated.get("status"), "error")
                self.assertEqual(validated.get("success"), False)
                self.assertEqual(validated.get("valid"), False)
                self.assertEqual(validated.get("error"), "delegate failure")

                user_info = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "auth_tools",
                        "get_user_info",
                        {"token": "tok"},
                    )
                )
                self.assertEqual(user_info.get("status"), "error")
                self.assertEqual(user_info.get("success"), False)
                self.assertEqual(user_info.get("error"), "delegate failure")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
