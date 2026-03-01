"""Peer registry primitive for MCP++ runtime integration."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

import anyio

logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py.mcplusplus_module.p2p.peer_registry import P2PPeerRegistry as _PeerRegistry

    HAVE_PEER_REGISTRY = True
except ImportError:
    HAVE_PEER_REGISTRY = False
    _PeerRegistry = None  # type: ignore[assignment]


class PeerRegistryWrapper:
    """Small async-friendly wrapper around MCP++ peer registry."""

    def __init__(self, repo: str = "endomorphosis/ipfs_accelerate_py", bootstrap_nodes: Optional[List[str]] = None):
        self.repo = str(repo)
        self.bootstrap_nodes = list(bootstrap_nodes or [])
        self.available = HAVE_PEER_REGISTRY
        self._registry: Any = None

        if self.available and _PeerRegistry is not None:
            try:
                # Prefer minimal constructor args to reduce coupling.
                self._registry = _PeerRegistry(repo=self.repo)
            except Exception as exc:
                logger.warning("Failed to initialize peer registry: %s", exc)
                self.available = False

    async def _call_registry(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if not self.available or self._registry is None:
            return None

        method = getattr(self._registry, method_name, None)
        if method is None:
            return None

        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await anyio.to_thread.run_sync(lambda: method(*args, **kwargs))

    async def discover_peers(self, max_peers: int = 50, timeout: float = 30.0) -> List[Dict[str, Any]]:
        del timeout
        # Try common method names from existing peer registry implementations.
        result = await self._call_registry("discover_peers")
        if result is None:
            result = await self._call_registry("list_peers")

        if isinstance(result, list):
            peers = [p for p in result if isinstance(p, dict)]
            return peers[: int(max_peers)]
        if isinstance(result, dict):
            peers = result.get("peers")
            if isinstance(peers, list):
                return [p for p in peers if isinstance(p, dict)][: int(max_peers)]
        return []

    async def connect_to_peer(self, peer_id: str, multiaddr: str) -> bool:
        result = await self._call_registry("connect_peer", peer_id=peer_id, multiaddr=multiaddr)
        if result is None:
            result = await self._call_registry("connect_to_peer", peer_id=peer_id, multiaddr=multiaddr)

        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("ok") or result.get("connected"))
        return False

    async def disconnect_peer(self, peer_id: str) -> bool:
        result = await self._call_registry("disconnect_peer", peer_id=peer_id)
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("ok") or result.get("disconnected"))
        return False

    async def list_connected_peers(self) -> List[Dict[str, Any]]:
        result = await self._call_registry("list_peers")
        if isinstance(result, list):
            return [p for p in result if isinstance(p, dict)]
        if isinstance(result, dict):
            peers = result.get("peers")
            if isinstance(peers, list):
                return [p for p in peers if isinstance(p, dict)]
        return []

    async def get_peer_metrics(self, peer_id: str) -> Optional[Dict[str, Any]]:
        result = await self._call_registry("get_peer_metrics", peer_id=peer_id)
        return result if isinstance(result, dict) else None

    def get_bootstrap_nodes(self) -> List[str]:
        return list(self.bootstrap_nodes)

    def add_bootstrap_node(self, multiaddr: str) -> None:
        value = str(multiaddr)
        if value not in self.bootstrap_nodes:
            self.bootstrap_nodes.append(value)


def create_peer_registry(
    repo: str = "endomorphosis/ipfs_accelerate_py",
    bootstrap_nodes: Optional[List[str]] = None,
) -> PeerRegistryWrapper:
    """Create peer registry wrapper instance."""
    return PeerRegistryWrapper(repo=repo, bootstrap_nodes=bootstrap_nodes)


__all__ = [
    "HAVE_PEER_REGISTRY",
    "PeerRegistryWrapper",
    "create_peer_registry",
]
