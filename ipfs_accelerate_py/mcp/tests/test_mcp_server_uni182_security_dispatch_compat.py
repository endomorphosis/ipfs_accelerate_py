#!/usr/bin/env python3
"""UNI-182 security dispatch compatibility tests for focused parity coverage."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerUNI182SecurityDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_security_dispatch_preserves_batch_schema_and_error_counts(self, mock_wrapper) -> None:
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

        async def _check_access(**kwargs: object) -> dict:
            if kwargs.get("resource_id") == "resource-allow":
                return {"status": "success", "allowed": True, **kwargs}
            return {"status": "success", "allowed": False, **kwargs}

        async def _run_flow() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION",
                _check_access,
            ), patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                server = create_mcp_server(name="security-dispatch-compat")

                get_schema = server.tools["tools_get_schema"]["function"]
                dispatch = server.tools["tools_dispatch"]["function"]

                batch_schema = await get_schema("security_tools", "check_access_permissions_batch")
                props = (batch_schema.get("input_schema") or {}).get("properties", {})
                self.assertEqual((props.get("requests") or {}).get("minItems"), 1)
                self.assertEqual((props.get("fail_fast") or {}).get("default"), False)

                result = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "security_tools",
                        "check_access_permissions_batch",
                        {
                            "requests": [
                                {"resource_id": "resource-allow", "user_id": "user-1", "permission_type": "read"},
                                {"resource_id": "resource-deny", "user_id": "user-2", "permission_type": "read"},
                                "bad-entry",
                            ],
                            "fail_fast": False,
                        },
                    )
                )
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("processed"), 3)
                self.assertEqual(result.get("allowed_count"), 1)
                self.assertEqual(result.get("denied_count"), 1)
                self.assertEqual(result.get("error_count"), 1)

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()