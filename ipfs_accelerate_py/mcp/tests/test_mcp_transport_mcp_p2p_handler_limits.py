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


class _NegotiatingRegistry:
    def __init__(self) -> None:
        self.tools = {}
        self._unified_supported_profiles = [
            "mcp++/profile-a-idl",
            "mcp++/profile-e-mcp-p2p",
        ]
        self._unified_profile_negotiation = {
            "supports_profile_negotiation": True,
            "mode": "optional_additive",
            "profiles": list(self._unified_supported_profiles),
        }


class _MalformedNegotiationRegistry:
    def __init__(self) -> None:
        self.tools = {}
        self._unified_supported_profiles = [
            "mcp++/profile-a-idl",
            "",
            "mcp++/profile-a-idl",
            "mcp++/profile-e-mcp-p2p",
        ]
        self._unified_profile_negotiation = {
            "supports_profile_negotiation": "false",
            "mode": "   ",
            "profiles": ["mcp++/profile-z-ignored"],
        }


class _RejectingRegistry:
    def __init__(self) -> None:
        self.tools = {}

    def validate_p2p_message(self, _msg: dict) -> bool:
        return False


class _ToolRegistry:
    def __init__(self) -> None:
        self._unified_supported_profiles = [
            "mcp++/profile-a-idl",
            "mcp++/profile-e-mcp-p2p",
        ]
        self.tools = {
            "echo": {
                "description": "Echo arguments",
                "input_schema": {"type": "object"},
                "function": self._echo,
            }
        }

    @staticmethod
    def _echo(**kwargs):
        return dict(kwargs)


class _FailingWriteStream(_FakeStream):
    async def write(self, _data: bytes) -> None:
        raise RuntimeError("write_failed")


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

    def test_handler_enforces_max_frames_limit_independent_of_token_bucket(self) -> None:
        init = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            }
        )
        tools_list_1 = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
        )
        tools_list_2 = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": {},
            }
        )
        stream = _FakeStream(init + tools_list_1 + tools_list_2)

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
                "IPFS_DATASETS_PY_MCP_P2P_MAX_FRAMES": "2",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_CAPACITY": "100",
                "IPFS_DATASETS_PY_MCP_P2P_RATE_REFILL_PER_SEC": "100",
            },
            clear=False,
        ):
            anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0].get("id"), 1)
        self.assertIn("result", responses[0])
        self.assertEqual(responses[1].get("id"), 2)
        self.assertIn("result", responses[1])
        self.assertEqual(responses[2].get("id"), 3)
        self.assertEqual(responses[2].get("error", {}).get("code"), -32010)
        self.assertEqual(responses[2].get("error", {}).get("message"), "rate_limited")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)
        self.assertEqual(stats.get("initialized_sessions"), 1)
        self.assertEqual(stats.get("rate_limited"), 1)

    def test_rate_limited_counter_accumulates_across_sessions(self) -> None:
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

        async def _run_session() -> None:
            stream = _FakeStream(init + tools_list)
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
            anyio.run(_run_session)
            anyio.run(_run_session)

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("sessions_started"), 2)
        self.assertEqual(stats.get("sessions_closed"), 2)
        self.assertEqual(stats.get("initialized_sessions"), 2)
        self.assertEqual(stats.get("rate_limited"), 2)

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

        negotiation = result.get("profile_negotiation", {})
        self.assertTrue(negotiation.get("supports_profile_negotiation"))
        self.assertEqual(negotiation.get("mode"), "optional_additive")
        profiles = negotiation.get("profiles", [])
        self.assertIsInstance(profiles, list)
        self.assertIn("mcp++/profile-e-mcp-p2p", profiles)
        self.assertEqual(result.get("active_profile"), profiles[0])

    def test_initialize_selects_requested_supported_profile(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"profile": "mcp++/profile-e-mcp-p2p"},
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_NegotiatingRegistry(),
                max_frame_bytes=4096,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        result = responses[0].get("result", {})
        self.assertEqual(result.get("active_profile"), "mcp++/profile-e-mcp-p2p")
        self.assertEqual(
            (result.get("profile_negotiation") or {}).get("profiles"),
            ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
        )

    def test_initialize_selects_from_profile_list_for_mixed_version_peer(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "profiles": [
                            "mcp++/profile-z-next",
                            "mcp++/profile-e-mcp-p2p",
                            "mcp++/profile-a-idl",
                        ]
                    },
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_NegotiatingRegistry(),
                max_frame_bytes=4096,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        result = responses[0].get("result", {})
        self.assertEqual(result.get("active_profile"), "mcp++/profile-e-mcp-p2p")
        self.assertEqual(
            (result.get("profile_negotiation") or {}).get("profiles"),
            ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
        )

    def test_initialize_falls_back_when_requested_profile_unsupported(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"profile": "mcp++/profile-z-unknown"},
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_NegotiatingRegistry(),
                max_frame_bytes=4096,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        result = responses[0].get("result", {})
        self.assertEqual(result.get("active_profile"), "mcp++/profile-a-idl")

    def test_initialize_normalizes_negotiation_payload_and_profile_list(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "profiles": [
                            "mcp++/profile-z-unknown",
                            {"invalid": "candidate"},
                            "",
                        ]
                    },
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_MalformedNegotiationRegistry(),
                max_frame_bytes=4096,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        result = responses[0].get("result", {})
        self.assertEqual(result.get("active_profile"), "mcp++/profile-a-idl")
        negotiation = result.get("profile_negotiation", {})
        self.assertFalse(negotiation.get("supports_profile_negotiation"))
        self.assertEqual(negotiation.get("mode"), "optional_additive")
        self.assertEqual(
            negotiation.get("profiles"),
            ["mcp++/profile-a-idl", "mcp++/profile-e-mcp-p2p"],
        )

    def test_handler_tracks_unauthorized_counter(self) -> None:
        request = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            }
        )
        stream = _FakeStream(request)

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_RejectingRegistry(),
                max_frame_bytes=1024 * 1024,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].get("error", {}).get("code"), -32001)
        self.assertEqual(responses[0].get("error", {}).get("message"), "unauthorized")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("unauthorized"), 1)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)

    def test_handler_tracks_internal_error_counter_when_write_fails(self) -> None:
        # Oversized declaration triggers deterministic error response path;
        # with write failure, outer exception accounting should increment internal_errors.
        stream = _FailingWriteStream((4097).to_bytes(4, byteorder="big", signed=False))

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=1024,
            )

        anyio.run(_run)

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("frame_errors"), 1)
        self.assertEqual(stats.get("internal_errors"), 1)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)

    def test_non_initialize_first_request_is_rejected(self) -> None:
        stream = _FakeStream(
            encode_jsonrpc_frame(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/list",
                    "params": {},
                }
            )
        )

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=1024 * 1024,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].get("id"), 99)
        self.assertEqual(responses[0].get("error", {}).get("code"), -32000)
        self.assertEqual(responses[0].get("error", {}).get("message"), "init_required")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("initialized_sessions"), 0)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)

    def test_initialize_notification_does_not_initialize_session(self) -> None:
        init_notification = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {},
            }
        )
        follow_up_request = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/list",
                "params": {},
            }
        )
        stream = _FakeStream(init_notification + follow_up_request)

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_DummyRegistry(),
                max_frame_bytes=1024 * 1024,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].get("id"), 5)
        self.assertEqual(responses[0].get("error", {}).get("code"), -32000)
        self.assertEqual(responses[0].get("error", {}).get("message"), "init_required")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("initialized_sessions"), 0)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)

    def test_mixed_version_initialize_and_call_flow_remains_deterministic(self) -> None:
        init = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocol_version": "/mcp+p2p/0.9.0",
                    "client_version": "0.9.3",
                    "profile": "mcp++/profile-z-next",
                    "profiles": [
                        "mcp++/profile-z-next",
                        "mcp++/profile-e-mcp-p2p",
                    ],
                },
            }
        )
        tools_list = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools.list",
                "params": {},
            }
        )
        tools_call = encode_jsonrpc_frame(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools.call",
                "params": {
                    "name": "echo",
                    "arguments": {"value": "ok", "count": 2},
                },
            }
        )
        stream = _FakeStream(init + tools_list + tools_call)

        async def _run() -> None:
            await handle_mcp_p2p_stream(
                stream,
                local_peer_id="peer-a",
                registry=_ToolRegistry(),
                max_frame_bytes=1024 * 1024,
            )

        anyio.run(_run)

        responses = _decode_all_frames(bytes(stream.written))
        self.assertEqual(len(responses), 3)

        init_result = responses[0].get("result", {})
        self.assertEqual(init_result.get("active_profile"), "mcp++/profile-e-mcp-p2p")
        self.assertEqual(responses[1].get("id"), 2)
        self.assertEqual((responses[1].get("result") or {}).get("tools")[0].get("name"), "echo")
        self.assertEqual(responses[2].get("id"), 3)
        self.assertEqual(
            (responses[2].get("result") or {}).get("content"),
            {"value": "ok", "count": 2},
        )

    def test_handler_rejects_non_object_json_payload_as_invalid_message(self) -> None:
        raw_payload = b'"not-an-object"'
        stream = _FakeStream(len(raw_payload).to_bytes(4, byteorder="big", signed=False) + raw_payload)

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
        self.assertEqual(responses[0].get("error", {}).get("message"), "invalid_message")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("frame_errors"), 1)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)

    def test_handler_rejects_invalid_utf8_payload_as_invalid_json(self) -> None:
        raw_payload = b"\xff\xfe\xfd"
        stream = _FakeStream(len(raw_payload).to_bytes(4, byteorder="big", signed=False) + raw_payload)

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
        self.assertEqual(responses[0].get("error", {}).get("message"), "invalid_json")

        stats = get_mcp_p2p_stats()
        self.assertEqual(stats.get("frame_errors"), 1)
        self.assertEqual(stats.get("sessions_started"), 1)
        self.assertEqual(stats.get("sessions_closed"), 1)


if __name__ == "__main__":
    unittest.main()
