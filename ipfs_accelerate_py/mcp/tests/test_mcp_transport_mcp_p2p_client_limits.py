#!/usr/bin/env python3
"""Transport conformance tests for MCP+p2p client outbound frame limits."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.p2p_tasks.mcp_p2p_client import MCPFramingError, MCPP2PClient
from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import encode_jsonrpc_frame


class _FakeStream:
    def __init__(self, incoming: bytes = b"") -> None:
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


class TestMCPP2PClientLimits(unittest.TestCase):
    """Validate outbound frame-size enforcement in client request paths."""

    def test_request_rejects_oversized_outbound_frame(self) -> None:
        stream = _FakeStream()
        client = MCPP2PClient(stream, max_frame_bytes=1024 * 1024, max_outbound_frame_bytes=64)

        async def _run() -> None:
            with self.assertRaises(MCPFramingError):
                await client.request("tools/call", {"x": "y" * 1024}, id_value=1)

        anyio.run(_run)
        self.assertEqual(bytes(stream.written), b"")

    def test_notify_uses_compat_env_outbound_limit(self) -> None:
        stream = _FakeStream()
        payload = {"big": "z" * 1024}

        async def _run() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_DATASETS_PY_MCP_P2P_CLIENT_MAX_OUTBOUND_FRAME_BYTES": "32",
                },
                clear=False,
            ):
                client = MCPP2PClient(stream, max_frame_bytes=1024 * 1024)
                with self.assertRaises(MCPFramingError):
                    await client.notify("oversized", payload)

        anyio.run(_run)
        self.assertEqual(bytes(stream.written), b"")

    def test_request_succeeds_when_within_outbound_limit(self) -> None:
        response = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "result": {"ok": True},
            }
        )
        stream = _FakeStream(response)
        client = MCPP2PClient(stream, max_frame_bytes=1024 * 1024, max_outbound_frame_bytes=2048)

        async def _run() -> None:
            result = await client.request("initialize", {}, id_value=7)
            self.assertIn("result", result)

        anyio.run(_run)
        self.assertGreater(len(stream.written), 0)


if __name__ == "__main__":
    unittest.main()
