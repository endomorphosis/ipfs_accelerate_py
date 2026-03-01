#!/usr/bin/env python3
"""End-to-end transport matrix tests for unified MCP bootstrap."""

import os
import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import decode_jsonrpc_frame, encode_jsonrpc_frame
from ipfs_accelerate_py.p2p_tasks.mcp_p2p import handle_mcp_p2p_stream


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


class _FakeStream:
    def __init__(self, incoming: bytes) -> None:
        self._incoming = incoming
        self._offset = 0
        self.written = bytearray()
        self.closed = False

    async def read(self, n: int) -> bytes:
        if self._offset >= len(self._incoming):
            return b""
        end = min(len(self._incoming), self._offset + max(0, int(n)))
        chunk = self._incoming[self._offset : end]
        self._offset = end
        return bytes(chunk)

    async def write(self, data: bytes) -> None:
        self.written.extend(bytes(data))

    async def close(self) -> None:
        self.closed = True


def _decode_all_frames(buffer: bytes) -> list[dict]:
    out: list[dict] = []
    idx = 0
    raw = bytes(buffer)
    while idx < len(raw):
        payload, consumed = decode_jsonrpc_frame(raw[idx:])
        out.append(payload)
        idx += consumed
    return out


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
            self.assertIsInstance(result, dict)
            self.assertEqual(result.get("mode"), "http")
            self.assertEqual(result.get("value"), "ok")

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

            self.assertIsInstance(result, dict)
            self.assertEqual(result.get("mode"), "trio")
            self.assertEqual(result.get("value"), "ok")
            self.assertEqual(mock_trio.await_count, 1)

        anyio.run(_run)

    def test_mcp_p2p_style_initialize_and_tools_list_parity(self) -> None:
        """Validate MCP+p2p handler can list tools from the unified registry contract."""

        async def _run() -> None:
            server = self._bootstrap_server()

            async def echo(value: str):
                return {"mode": "mcp+p2p", "value": value}

            server.register_tool(
                "echo_mcp_p2p",
                echo,
                "mcp+p2p echo",
                {"type": "object", "properties": {"value": {"type": "string"}}},
            )

            stream = _FakeStream(
                encode_jsonrpc_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
                + encode_jsonrpc_frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            )

            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-parity",
                registry=server,
                max_frame_bytes=1024 * 1024,
            )

            self.assertTrue(stream.closed)
            responses = _decode_all_frames(bytes(stream.written))
            self.assertEqual(len(responses), 2)
            self.assertEqual(responses[0].get("id"), 1)
            self.assertIn("result", responses[0])
            self.assertEqual(responses[1].get("id"), 2)

            tools = ((responses[1].get("result") or {}).get("tools") or [])
            self.assertTrue(any(t.get("name") == "echo_mcp_p2p" for t in tools if isinstance(t, dict)))

        anyio.run(_run)

    def test_mcp_p2p_style_tools_call_parity(self) -> None:
        """Validate MCP+p2p tools/call uses the same registry function descriptor contract."""

        async def _run() -> None:
            server = self._bootstrap_server()

            async def echo(value: str):
                return {"mode": "mcp+p2p", "value": value}

            server.register_tool(
                "echo_mcp_p2p",
                echo,
                "mcp+p2p echo",
                {"type": "object", "properties": {"value": {"type": "string"}}},
            )

            stream = _FakeStream(
                encode_jsonrpc_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
                + encode_jsonrpc_frame(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "echo_mcp_p2p",
                            "arguments": {"value": "ok"},
                        },
                    }
                )
            )

            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-parity",
                registry=server,
                max_frame_bytes=1024 * 1024,
            )

            responses = _decode_all_frames(bytes(stream.written))
            self.assertEqual(len(responses), 2)
            self.assertEqual(responses[1].get("id"), 2)
            content = ((responses[1].get("result") or {}).get("content") or {})
            self.assertEqual(content.get("mode"), "mcp+p2p")
            self.assertEqual(content.get("value"), "ok")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
