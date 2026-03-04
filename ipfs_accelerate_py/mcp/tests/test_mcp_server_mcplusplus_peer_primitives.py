#!/usr/bin/env python3
"""Unit tests for unified MCP++ peer primitives."""

import unittest
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.peer_discovery import PeerDiscoveryManager
from ipfs_accelerate_py.mcp_server.mcplusplus.peer_registry import create_peer_registry


class _FakeRegistry:
    async def discover_peers(self, max_peers: int = 50):
        del max_peers
        return [
            {
                "peer_id": "peer-a",
                "multiaddr": "/ip4/127.0.0.1/tcp/4001/p2p/peer-a",
                "capabilities": ["compute", "storage"],
                "ttl_seconds": 3600,
            },
            {
                "peer_id": "peer-b",
                "multiaddr": "/ip4/127.0.0.1/tcp/4002/p2p/peer-b",
                "capabilities": ["storage"],
                "ttl_seconds": 3600,
            },
        ]


class TestPeerPrimitives(unittest.TestCase):
    """Validate peer wrapper and discovery manager behavior."""

    def test_bootstrap_nodes_management(self) -> None:
        registry = create_peer_registry(bootstrap_nodes=["/ip4/1.2.3.4/tcp/4001/p2p/x"])
        registry.add_bootstrap_node("/ip4/1.2.3.4/tcp/4001/p2p/x")
        registry.add_bootstrap_node("/ip4/5.6.7.8/tcp/4001/p2p/y")
        self.assertEqual(len(registry.get_bootstrap_nodes()), 2)

    def test_registry_unavailable_paths(self) -> None:
        async def _run() -> None:
            with patch("ipfs_accelerate_py.mcp_server.mcplusplus.peer_registry.HAVE_PEER_REGISTRY", False):
                registry = create_peer_registry()
                self.assertEqual(await registry.discover_peers(), [])
                self.assertFalse(await registry.connect_to_peer("p", "m"))
                self.assertFalse(await registry.disconnect_peer("p"))
                self.assertEqual(await registry.list_connected_peers(), [])
                self.assertIsNone(await registry.get_peer_metrics("p"))

        anyio.run(_run)

    def test_discovery_manager_capability_filter(self) -> None:
        async def _run() -> None:
            manager = PeerDiscoveryManager(registry=cast(Any, _FakeRegistry()))
            peers = await manager.discover_peers(capability_filter=["compute"])
            self.assertEqual(len(peers), 1)
            self.assertEqual(peers[0].peer_id, "peer-a")

        anyio.run(_run)

    def test_registry_delegates_to_common_method_names(self) -> None:
        async def _run() -> None:
            registry = create_peer_registry()
            registry.available = True
            registry._registry = type("R", (), {})()
            registry._registry.discover_peers = AsyncMock(return_value=[{"peer_id": "p1", "multiaddr": "m1"}])
            peers = await registry.discover_peers()
            self.assertEqual(peers, [{"peer_id": "p1", "multiaddr": "m1"}])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
