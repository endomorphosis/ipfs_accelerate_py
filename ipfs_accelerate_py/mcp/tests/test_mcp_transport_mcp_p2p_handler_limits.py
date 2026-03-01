#!/usr/bin/env python3
"""Transport conformance tests for MCP+p2p stream handler enforcement."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import (
    decode_jsonrpc_frame,
    encode_jsonrpc_frame,
)
from ipfs_accelerate_py.p2p_tasks.mcp_p2p import handle_mcp_p2p_stream
from ipfs_accelerate_py.p2p_tasks.mcp_p2p import get_mcp_p2p_stats, reset_mcp_p2p_stats


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


class _DummyRegistry:
    def __init__(self) -> None:
        self.tools = {}


def _decode_all_frames(buffer: bytes) -> list[dict]:
    out: list[dict] = []
    idx = 0
    raw = bytes(buffer)
    while idx < len(raw):
        payload, consumed = decode_jsonrpc_frame(raw[idx:])
        out.append(payload)
        idx += consumed
    return out


class TestMCPP2PHandlerLimits(unittest.TestCase):
    """Validate session-level framing and abuse controls."""

    def setUp(self) -> None:
        reset_mcp_p2p_stats()

    def test_handler_rejects_declared_oversized_frame(self) -> None:
        # Declared length is over max_frame_bytes; no body is required for rejection.
        stream = _FakeStream((4097).to_bytes(4, byteorder="big", signed=False))

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=1024,
            )

        anyio.run(_run)

        self.assertTrue(stream.closed)
        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].get("error", {}).get("code"), -32003)
        self.assertEqual(responses[0].get("error", {}).get("message"), "frame_too_large")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)
        self.assertEqual(stats.get("frame_errors"), 1)

    def test_handler_enforces_token_bucket_rate_limit(self) -> None:
        init = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            }
        )
        tools_list = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
        )
        stream = _FakeStream(init + tools_list)

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=1024 * 1024,
            )

        with patch.dict(
            os.environ,
            {
                "IPFS_DATASETS_PY_MCP_P2P_MAX_FRAMES": "16",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_CAPACITY": "1",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_REFILL_PER_SEC": "0.0001",
            },
            clear=False,
        ):
            anyio.run(_run)

        self.assertTrue(stream.closed)
        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 2)

        self.assertEqual(responses[0].get("id"), 1)
        self.assertIn("result", responses[0])

        self.assertEqual(responses[1].get("id"), 2)
        self.assertEqual(responses[1].get("error", {}).get("code"), -32010)
        self.assertEqual(responses[1].get("error", {}).get("message"), "rate_limited")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)
        self.assertEqual(stats.get("initialized_sessions"), 1)
        self.assertEqual(stats.get("rate_limited"), 1)

    def test_initialize_advertises_effective_limits(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {},
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=4096,
            )

        with patch.dict(
            os.environ,
            {
                "IPFS_DATASETS_PY_MCP_P2P_MAX_FRAMES": "42",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_CAPACITY": "7",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_REFILL_PER_SEC": "3.5",
            },
            clear=False,
        ):
            anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].get("id"), 1)

        result = responses[0].get("result", {})
        self.assertEqual(result.get("transport"), "/mcp+p2p/1.0.0")
        limits = result.get("limits", {})
        self.assertEqual(limits.get("max_frame_bytes"), 4096)
        self.assertEqual(limits.get("max_frames"), 42)
        self.assertEqual(limits.get("rate_capacity"), 7)
        self.assertEqual(limits.get("rate_refill_per_sec"), 3.5)


if __name__ == "__main__":
    unittest.main()
