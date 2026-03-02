#!/usr/bin/env python3
"""Tests for tools_dispatch control-flag coercion behavior."""

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerDispatchFlagCoercion(unittest.TestCase):
    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_rejects_invalid_boolean_flag(self, mock_wrapper) -> None:
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

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-invalid-flag")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": "definitely",
                },
            )

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "invalid_dispatch_parameter")
            self.assertIn("__enforce_ucan", response.get("details", ""))

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_parses_false_string_without_enforcing_ucan(self, mock_wrapper) -> None:
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

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-false-flag")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(
                "smoke",
                "echo",
                {
                    "value": "ok",
                    "__enforce_ucan": "false",
                    "__ucan_actor": "did:model:worker",
                    "__ucan_proof_chain": [],
                },
            )

            self.assertTrue(response["ok"])
            self.assertEqual(response["result"], {"echo": "ok"})

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
