#!/usr/bin/env python3
"""Tests for canonical mcp_server mcp+p2p transport facades."""

import unittest
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server import mcp_p2p_transport


class TestMCPP2PTransportFacade(unittest.TestCase):
    def test_protocol_constant_contract(self) -> None:
        self.assertEqual(mcp_p2p_transport.PROTOCOL_MCP_P2P_V1, "/mcp+p2p/1.0.0")

    @patch("ipfs_accelerate_py.p2p_tasks.mcp_p2p.get_mcp_p2p_stats", return_value={"sessions_started": 3})
    def test_stats_delegate(self, mock_get) -> None:
        stats = mcp_p2p_transport.get_mcp_p2p_stats()
        self.assertEqual(stats, {"sessions_started": 3})
        mock_get.assert_called_once_with()

    @patch("ipfs_accelerate_py.p2p_tasks.mcp_p2p.reset_mcp_p2p_stats")
    def test_reset_delegate(self, mock_reset) -> None:
        mcp_p2p_transport.reset_mcp_p2p_stats()
        mock_reset.assert_called_once_with()

    def test_async_delegates(self) -> None:
        async def _run() -> None:
            stream = object()
            with patch(
                "ipfs_accelerate_py.p2p_tasks.mcp_p2p.read_u32_framed_json",
                new=AsyncMock(return_value=({"jsonrpc": "2.0"}, None)),
            ) as mock_read:
                result = await mcp_p2p_transport.read_u32_framed_json(stream, max_frame_bytes=128, chunk_size=32)
                self.assertEqual(result, ({"jsonrpc": "2.0"}, None))
                mock_read.assert_awaited_once_with(stream, max_frame_bytes=128, chunk_size=32)

            with patch(
                "ipfs_accelerate_py.p2p_tasks.mcp_p2p.write_u32_framed_json",
                new=AsyncMock(return_value=True),
            ) as mock_write:
                ok = await mcp_p2p_transport.write_u32_framed_json(stream, {"ok": True}, max_frame_bytes=256)
                self.assertTrue(ok)
                mock_write.assert_awaited_once_with(stream, {"ok": True}, max_frame_bytes=256)

            with patch(
                "ipfs_accelerate_py.p2p_tasks.mcp_p2p.handle_mcp_p2p_stream",
                new=AsyncMock(return_value=None),
            ) as mock_handle:
                registry = object()
                await mcp_p2p_transport.handle_mcp_p2p_stream(
                    stream,
                    registry=registry,
                    peer_id="peer-1",
                    max_frame_bytes=4096,
                )
                mock_handle.assert_awaited_once_with(
                    stream,
                    local_peer_id="peer-1",
                    registry=registry,
                    max_frame_bytes=4096,
                )

            with patch(
                "ipfs_accelerate_py.p2p_tasks.mcp_p2p.handle_mcp_p2p_stream",
                new=AsyncMock(return_value=None),
            ) as mock_handle:
                await mcp_p2p_transport.handle_mcp_p2p_stream(
                    stream,
                    registry=registry,
                    peer_id="peer-1",
                    local_peer_id="peer-2",
                )
                mock_handle.assert_awaited_once_with(
                    stream,
                    local_peer_id="peer-2",
                    registry=registry,
                    max_frame_bytes=1024 * 1024,
                )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
