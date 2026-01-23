"""Tests for GitHub API cache peer exchange.

These tests focus on whether the GitHub API cache *mechanism* can exchange
information (broadcast + receive/ingest) without requiring a real libp2p
network in CI/dev environments.

They intentionally do NOT prove that two separate processes can communicate
over libp2p (that requires libp2p + networking). Instead they verify:
- A cache write triggers the broadcast scheduling hook when P2P is enabled.
- An incoming peer payload is accepted by the receive handler and becomes
  available via normal cache.get().
"""

import asyncio
import json
import time
from typing import List

import pytest

from ipfs_accelerate_py.github_cli.cache import configure_cache


class _FakeP2PStream:
    """Minimal async stream shim for exercising cache receive handler."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.writes: List[bytes] = []
        self.closed = False

    async def read(self) -> bytes:
        return self._payload

    async def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def close(self) -> None:
        self.closed = True


def test_peer_exchange_broadcast_attempt(tmp_path) -> None:
    cache = configure_cache(
        cache_dir=str(tmp_path),
        enable_p2p=False,
        enable_persistence=False,
    )

    # Force the codepath that calls _broadcast_in_background.
    cache.enable_p2p = True

    calls = {"n": 0}

    def _fake_broadcast_in_background(cache_key, entry) -> None:
        calls["n"] += 1

    cache._broadcast_in_background = _fake_broadcast_in_background  # type: ignore[attr-defined]

    cache.put("list_repos", [{"name": "repo1"}], ttl=60, owner="owner1", limit=1)

    assert calls["n"] == 1


def test_peer_exchange_receive_and_serve(tmp_path) -> None:
    cache = configure_cache(
        cache_dir=str(tmp_path),
        enable_p2p=False,
        enable_persistence=True,
    )

    operation = "list_repos"
    assert cache.get(operation, owner="peer-owner", limit=2) is None

    cache_key = cache._make_cache_key(operation, owner="peer-owner", limit=2)  # type: ignore[attr-defined]
    expected = [{"name": "from-peer", "full_name": "peer-owner/from-peer"}]

    message = {
        "key": cache_key,
        "entry": {
            "data": expected,
            "timestamp": time.time(),
            "ttl": 300,
            "content_hash": None,
            "validation_fields": None,
        },
    }

    stream = _FakeP2PStream(json.dumps(message).encode("utf-8"))
    asyncio.run(cache._handle_cache_stream(stream))  # type: ignore[attr-defined]

    assert cache.get(operation, owner="peer-owner", limit=2) == expected

    stats = cache.get_stats()
    assert stats.get("peer_hits", 0) >= 1

    # Sanity check for handler handshake
    assert stream.writes and stream.writes[-1] in {b"OK", b"ERROR: Invalid format", b"ERROR: Decryption failed"}
    assert stream.closed is True
