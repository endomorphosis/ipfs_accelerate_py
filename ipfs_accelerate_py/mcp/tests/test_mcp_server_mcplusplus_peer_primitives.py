#!/usr/bin/env python3
"""Unit tests for unified MCP++ peer primitives."""

import unittest
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import anyio

from ipfs_accelerate_py.mcp_server.mcplusplus.peer_discovery import PeerDiscoveryManager
from ipfs_accelerate_py.mcp_server.mcplusplus.peer_bootstrap import create_peer_bootstrap
from ipfs_accelerate_py.mcp_server.mcplusplus.peer_discovery import create_peer_discovery
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


class _FakeBootstrap:
    async def discover_peers(self, max_peers: int = 10):
        del max_peers
        return [{"peer_id": "peer-a", "multiaddr": "/ip4/127.0.0.1/tcp/4001/p2p/peer-a"}]

    def get_bootstrap_addrs(self, max_peers: int = 5):
        del max_peers
        return ["/ip4/127.0.0.1/tcp/4001/p2p/peer-a"]

    def cleanup_stale_peers(self):
        return 2

    def register_peer(self, **kwargs):
        return kwargs.get("peer_id") == "peer-a"

    def heartbeat(self, **_kwargs):
        return None


class TestPeerPrimitives(unittest.TestCase):
    """Validate peer wrapper and discovery manager behavior."""

    def test_bootstrap_nodes_management(self) -> None:
        registry = create_peer_registry(bootstrap_nodes=["/ip4/1.2.3.4/tcp/4001/p2p/x"])
        registry.add_bootstrap_node("/ip4/1.2.3.4/tcp/4001/p2p/x")
        registry.add_bootstrap_node("/ip4/5.6.7.8/tcp/4001/p2p/y")
        self.assertEqual(len(registry.get_bootstrap_nodes()), 2)

        bootstrap = create_peer_bootstrap(bootstrap_nodes=["/ip4/1.2.3.4/tcp/4001/p2p/x"])
        bootstrap.add_bootstrap_node("/ip4/1.2.3.4/tcp/4001/p2p/x")
        bootstrap.add_bootstrap_node("/ip4/5.6.7.8/tcp/4001/p2p/y")
        self.assertEqual(len(bootstrap.get_bootstrap_nodes()), 2)

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

    def test_bootstrap_unavailable_paths(self) -> None:
        async def _run() -> None:
            with patch("ipfs_accelerate_py.mcp_server.mcplusplus.peer_bootstrap.HAVE_PEER_BOOTSTRAP", False):
                bootstrap = create_peer_bootstrap()
                self.assertEqual(await bootstrap.discover_peers(), [])
                self.assertEqual(await bootstrap.get_bootstrap_addrs(), [])
                self.assertEqual(await bootstrap.cleanup_stale_peers(), 0)
                self.assertFalse(
                    await bootstrap.register_peer(
                        peer_id="p",
                        listen_port=1,
                        multiaddr="m",
                    )
                )
                self.assertFalse(await bootstrap.heartbeat("p", 1, "m"))

        anyio.run(_run)

    def test_discovery_manager_capability_filter(self) -> None:
        async def _run() -> None:
            manager = PeerDiscoveryManager(registry=cast(Any, _FakeRegistry()))
            peers = await manager.discover_peers(capability_filter=["compute"])
            self.assertEqual(len(peers), 1)
            self.assertEqual(peers[0].peer_id, "peer-a")

        anyio.run(_run)

    def test_create_peer_discovery_uses_service_bundle_registry(self) -> None:
        async def _run() -> None:
            fake_registry = cast(Any, _FakeRegistry())
            with patch(
                "ipfs_accelerate_py.mcp_server.mcplusplus.peer_discovery.create_peer_service_bundle",
                return_value=cast(Any, type("Bundle", (), {"peer_registry": fake_registry})()),
            ):
                manager = create_peer_discovery()
                peers = await manager.discover_peers(max_peers=1)

            self.assertIs(manager.registry, fake_registry)
            self.assertEqual(len(peers), 1)
            self.assertEqual(peers[0].peer_id, "peer-a")

        anyio.run(_run)

    def test_create_peer_discovery_honors_bundle_registry_disable(self) -> None:
        async def _run() -> None:
            bundle = cast(Any, type("Bundle", (), {"peer_registry": None})())
            with patch(
                "ipfs_accelerate_py.mcp_server.mcplusplus.peer_discovery.create_peer_registry",
                side_effect=AssertionError("create_peer_registry should not run when bundle is supplied"),
            ):
                manager = create_peer_discovery(service_bundle=bundle)
                peers = await manager.discover_peers()

            self.assertIsNone(manager.registry)
            self.assertEqual(peers, [])

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

    def test_bootstrap_wrapper_delegates_to_helper_methods(self) -> None:
        async def _run() -> None:
            bootstrap = create_peer_bootstrap(bootstrap_nodes=["/ip4/1.2.3.4/tcp/4001/p2p/x"])
            bootstrap.available = True
            bootstrap._bootstrap = _FakeBootstrap()

            peers = await bootstrap.discover_peers()
            self.assertEqual(peers, [{"peer_id": "peer-a", "multiaddr": "/ip4/127.0.0.1/tcp/4001/p2p/peer-a"}])

            addrs = await bootstrap.get_bootstrap_addrs()
            self.assertEqual(
                addrs,
                [
                    "/ip4/1.2.3.4/tcp/4001/p2p/x",
                    "/ip4/127.0.0.1/tcp/4001/p2p/peer-a",
                ],
            )

            self.assertEqual(await bootstrap.cleanup_stale_peers(), 2)
            self.assertTrue(
                await bootstrap.register_peer(
                    peer_id="peer-a",
                    listen_port=4001,
                    multiaddr="/ip4/127.0.0.1/tcp/4001/p2p/peer-a",
                )
            )
            self.assertTrue(await bootstrap.heartbeat("peer-a", 4001, "/ip4/127.0.0.1/tcp/4001/p2p/peer-a"))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
