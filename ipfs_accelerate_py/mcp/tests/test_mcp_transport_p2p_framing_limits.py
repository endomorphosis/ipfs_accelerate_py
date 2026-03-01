#!/usr/bin/env python3
"""Transport conformance tests for `mcp+p2p` framing and abuse limits."""

import unittest

from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import (
    FrameSizeExceededError,
    FramingError,
    TokenBucketLimiter,
    decode_jsonrpc_frame,
    encode_jsonrpc_frame,
)


class TestMCPP2PFraming(unittest.TestCase):
    """Validate deterministic framing and max-size enforcement."""

    def test_frame_round_trip(self) -> None:
        payload = {"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {"category": "ipfs"}}

        frame = encode_jsonrpc_frame(payload)
        decoded, consumed = decode_jsonrpc_frame(frame)

        self.assertEqual(decoded, payload)
        self.assertEqual(consumed, len(frame))

    def test_framing_is_stable_for_same_payload(self) -> None:
        payload = {"jsonrpc": "2.0", "id": "1", "method": "ping", "params": {"x": 1, "y": "z"}}

        frame_a = encode_jsonrpc_frame(payload)
        frame_b = encode_jsonrpc_frame(payload)

        self.assertEqual(frame_a, frame_b)

    def test_encode_rejects_oversized_payload(self) -> None:
        payload = {"data": "x" * 1024}
        with self.assertRaises(FrameSizeExceededError):
            encode_jsonrpc_frame(payload, max_frame_bytes=128)

    def test_decode_rejects_declared_oversized_frame(self) -> None:
        declared_size = (4096).to_bytes(4, byteorder="big", signed=False)
        frame = declared_size + b"{}"

        with self.assertRaises(FrameSizeExceededError):
            decode_jsonrpc_frame(frame, max_frame_bytes=512)

    def test_decode_rejects_incomplete_prefix(self) -> None:
        with self.assertRaises(FramingError):
            decode_jsonrpc_frame(b"\x00\x00")

    def test_decode_rejects_incomplete_body(self) -> None:
        frame = (10).to_bytes(4, byteorder="big", signed=False) + b"{}"
        with self.assertRaises(FramingError):
            decode_jsonrpc_frame(frame)


class TestMCPP2PAbuseLimits(unittest.TestCase):
    """Validate token-bucket rate limiting behavior."""

    def test_token_bucket_blocks_when_budget_exhausted(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_rate_per_sec=1)

        self.assertTrue(limiter.allow(cost=1, now=0.0))
        self.assertTrue(limiter.allow(cost=1, now=0.0))
        self.assertFalse(limiter.allow(cost=1, now=0.0))

    def test_token_bucket_refills_over_time(self) -> None:
        limiter = TokenBucketLimiter(capacity=2, refill_rate_per_sec=2)

        self.assertTrue(limiter.allow(cost=2, now=0.0))
        self.assertFalse(limiter.allow(cost=1, now=0.0))

        # 0.5s at 2 tokens/s refills 1 token.
        self.assertTrue(limiter.allow(cost=1, now=0.5))

        # 1s more refills back toward capacity.
        self.assertTrue(limiter.allow(cost=1, now=1.5))


if __name__ == "__main__":
    unittest.main()
