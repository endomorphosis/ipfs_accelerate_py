"""Peer discovery primitive for MCP++ runtime integration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .peer_registry import PeerRegistryWrapper, create_peer_registry
from .peer_services import PeerServiceBundle, create_peer_service_bundle


@dataclass
class PeerInfo:
    """Information about a discovered peer."""

    peer_id: str
    multiaddr: str
    capabilities: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    source: str = "registry"
    ttl_seconds: int = 3600
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        return (time.time() - self.last_seen) > int(self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PeerInfo":
        return cls(**data)


class PeerDiscoveryManager:
    """Aggregate and normalize peers from registry backends."""

    def __init__(
        self,
        registry: Optional[PeerRegistryWrapper] = None,
        *,
        service_bundle: Optional[PeerServiceBundle] = None,
        repo: str = "endomorphosis/ipfs_accelerate_py",
        bootstrap_nodes: Optional[List[str]] = None,
    ):
        bundle = service_bundle
        if registry is None and bundle is None:
            bundle = create_peer_service_bundle(
                repo=repo,
                bootstrap_nodes=bootstrap_nodes,
                enable_bootstrap=False,
            )
        if registry is not None:
            self.registry = registry
        elif bundle is not None:
            self.registry = bundle.peer_registry
        else:
            self.registry = create_peer_registry(
                repo=repo,
                bootstrap_nodes=bootstrap_nodes,
            )

    async def discover_peers(
        self,
        capability_filter: Optional[List[str]] = None,
        max_peers: int = 50,
    ) -> List[PeerInfo]:
        if self.registry is None:
            return []

        raw = await self.registry.discover_peers(max_peers=max_peers)
        peers: List[PeerInfo] = []

        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                peer = PeerInfo(
                    peer_id=str(item.get("peer_id") or ""),
                    multiaddr=str(item.get("multiaddr") or ""),
                    capabilities=list(item.get("capabilities") or []),
                    first_seen=float(item.get("first_seen") or time.time()),
                    last_seen=float(item.get("last_seen") or time.time()),
                    source=str(item.get("source") or "registry"),
                    ttl_seconds=int(item.get("ttl_seconds") or 3600),
                    metadata=dict(item.get("metadata") or {}),
                )
            except Exception:
                continue

            if not peer.peer_id or not peer.multiaddr or peer.is_expired():
                continue

            if capability_filter:
                if not all(cap in peer.capabilities for cap in capability_filter):
                    continue

            peers.append(peer)
            if len(peers) >= int(max_peers):
                break

        return peers


def create_peer_discovery(
    registry: Optional[PeerRegistryWrapper] = None,
    *,
    service_bundle: Optional[PeerServiceBundle] = None,
    repo: str = "endomorphosis/ipfs_accelerate_py",
    bootstrap_nodes: Optional[List[str]] = None,
) -> PeerDiscoveryManager:
    """Create peer discovery manager through the canonical peer-service path."""
    return PeerDiscoveryManager(
        registry=registry,
        service_bundle=service_bundle,
        repo=repo,
        bootstrap_nodes=bootstrap_nodes,
    )


__all__ = [
    "PeerInfo",
    "PeerDiscoveryManager",
    "create_peer_discovery",
]
