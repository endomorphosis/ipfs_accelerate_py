"""Peer bootstrap primitive for MCP++ runtime integration."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, List, Optional

import anyio

logger = logging.getLogger(__name__)

try:
    from ipfs_accelerate_py.mcplusplus_module.p2p.bootstrap import SimplePeerBootstrap as _PeerBootstrapImpl

    HAVE_PEER_BOOTSTRAP = True
    _PeerBootstrap: Any = _PeerBootstrapImpl
except ImportError:
    HAVE_PEER_BOOTSTRAP = False
    _PeerBootstrap = None


class PeerBootstrapWrapper:
    """Async-friendly wrapper around the MCP++ peer bootstrap helper."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        peer_ttl_minutes: int = 30,
        bootstrap_nodes: Optional[List[str]] = None,
    ):
        self.cache_dir = cache_dir
        self.peer_ttl_minutes = int(peer_ttl_minutes)
        self.bootstrap_nodes = [str(item) for item in (bootstrap_nodes or []) if str(item).strip()]
        self.available = HAVE_PEER_BOOTSTRAP
        self._bootstrap: Any = None

        if self.available and _PeerBootstrap is not None:
            try:
                self._bootstrap = _PeerBootstrap(
                    cache_dir=self.cache_dir,
                    peer_ttl_minutes=self.peer_ttl_minutes,
                )
            except Exception as exc:
                logger.warning("Failed to initialize peer bootstrap helper: %s", exc)
                self.available = False

    async def _call_bootstrap(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if not self.available or self._bootstrap is None:
            return None

        method = getattr(self._bootstrap, method_name, None)
        if method is None:
            return None

        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await anyio.to_thread.run_sync(lambda: method(*args, **kwargs))

    async def discover_peers(self, max_peers: int = 10) -> List[dict]:
        result = await self._call_bootstrap("discover_peers", max_peers=max_peers)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)][: int(max_peers)]
        return []

    async def get_bootstrap_addrs(self, max_peers: int = 5) -> List[str]:
        merged: List[str] = []
        for node in self.bootstrap_nodes:
            if node not in merged:
                merged.append(node)

        result = await self._call_bootstrap("get_bootstrap_addrs", max_peers=max_peers)
        if isinstance(result, list):
            for node in result:
                if isinstance(node, str) and node and node not in merged:
                    merged.append(node)
        return merged[: int(max_peers)]

    async def cleanup_stale_peers(self) -> int:
        result = await self._call_bootstrap("cleanup_stale_peers")
        if isinstance(result, int):
            return result
        return 0

    async def register_peer(
        self,
        peer_id: str,
        listen_port: int,
        multiaddr: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        result = await self._call_bootstrap(
            "register_peer",
            peer_id=peer_id,
            listen_port=listen_port,
            multiaddr=multiaddr,
            metadata=metadata,
        )
        return bool(result)

    async def heartbeat(self, peer_id: str, listen_port: int, multiaddr: str) -> bool:
        result = await self._call_bootstrap(
            "heartbeat",
            peer_id=peer_id,
            listen_port=listen_port,
            multiaddr=multiaddr,
        )
        if result is None and self.available:
            return True
        return bool(result)

    def get_bootstrap_nodes(self) -> List[str]:
        return list(self.bootstrap_nodes)

    def add_bootstrap_node(self, multiaddr: str) -> None:
        value = str(multiaddr)
        if value and value not in self.bootstrap_nodes:
            self.bootstrap_nodes.append(value)


def create_peer_bootstrap(
    cache_dir: Optional[Path] = None,
    peer_ttl_minutes: int = 30,
    bootstrap_nodes: Optional[List[str]] = None,
) -> PeerBootstrapWrapper:
    """Create peer bootstrap wrapper instance."""
    return PeerBootstrapWrapper(
        cache_dir=cache_dir,
        peer_ttl_minutes=peer_ttl_minutes,
        bootstrap_nodes=bootstrap_nodes,
    )


__all__ = [
    "HAVE_PEER_BOOTSTRAP",
    "PeerBootstrapWrapper",
    "create_peer_bootstrap",
]