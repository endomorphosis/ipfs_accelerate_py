#!/usr/bin/env python3
"""End-to-end transport matrix tests for unified MCP bootstrap."""

import os
import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class _DummyServer:
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


class TestMCPServerTransportE2EMatrix(unittest.TestCase):
    """Validate stdio/http/trio-p2p transport-style dispatch paths."""

    def _bootstrap_server(self) -> _DummyServer:
        with patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper", return_value=_DummyServer()):
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                server = create_mcp_server(name="transport-e2e")
        return server

    def test_stdio_style_direct_manager_dispatch(self) -> None:
        """Simulate stdio transport by dispatching directly through unified manager."""

        async def _run() -> None:
            server = self._bootstrap_server()
            manager = server._unified_tool_manager

            async def echo(value: str):
                return {"mode": "stdio", "value": value}

            manager.register_tool("transport", "echo_stdio", echo, description="stdio echo")
            result = await manager.dispatch("transport", "echo_stdio", {"value": "ok"})
            self.assertEqual(result, {"mode": "stdio", "value": "ok"})

        anyio.run(_run)

    def test_http_style_meta_tool_dispatch(self) -> None:
        """Simulate HTTP transport by dispatching through registered meta-tool."""

        async def _run() -> None:
            server = self._bootstrap_server()
            manager = server._unified_tool_manager

            async def echo(value: str):
                return {"mode": "http", "value": value}

            manager.register_tool("transport", "echo_http", echo, description="http echo")
            dispatch = server.tools["tools_dispatch"]["function"]
            result = await dispatch("transport", "echo_http", {"value": "ok"})
            self.assertEqual(result, {"mode": "http", "value": "ok"})

        anyio.run(_run)

    def test_trio_p2p_style_dispatch_uses_trio_runtime(self) -> None:
        """Simulate trio-p2p transport by forcing trio runtime and asserting trio executor path."""

        async def _run() -> None:
            server = self._bootstrap_server()
            manager = server._unified_tool_manager
            router = server._unified_runtime_router

            async def echo(value: str):
                return {"mode": "trio", "value": value}

            manager.register_tool("transport", "echo_trio", echo, runtime="trio", description="trio echo")

            dispatch = server.tools["tools_dispatch"]["function"]
            with patch.object(router, "_execute_trio", AsyncMock(return_value={"mode": "trio", "value": "ok"})) as mock_trio:
                result = await dispatch("transport", "echo_trio", {"value": "ok"})

            self.assertEqual(result, {"mode": "trio", "value": "ok"})
            self.assertEqual(mock_trio.await_count, 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
