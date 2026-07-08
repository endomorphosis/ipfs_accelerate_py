"""Shared canonical peer-service construction helpers for MCP++ runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from .peer_bootstrap import create_peer_bootstrap
from .peer_registry import create_peer_registry


@dataclass
class PeerServiceBundle:
    """Canonical peer-service bundle for registry/bootstrap consumers."""

    peer_registry: Any | None = None
    peer_bootstrap: Any | None = None


def create_peer_service_bundle(
    *,
    repo: str = "endomorphosis/ipfs_accelerate_py",
    cache_dir: Optional[Path] = None,
    peer_ttl_minutes: int = 30,
    bootstrap_nodes: Optional[List[str]] = None,
    enable_peer_registry: bool = True,
    enable_bootstrap: bool = True,
) -> PeerServiceBundle:
    """Create the canonical MCP++ peer-service bundle."""
    nodes = list(bootstrap_nodes or [])
    return PeerServiceBundle(
        peer_registry=(
            create_peer_registry(repo=repo, bootstrap_nodes=nodes)
            if enable_peer_registry
            else None
        ),
        peer_bootstrap=(
            create_peer_bootstrap(
                cache_dir=cache_dir,
                peer_ttl_minutes=peer_ttl_minutes,
                bootstrap_nodes=nodes,
            )
            if enable_bootstrap
            else None
        ),
    )


__all__ = ["PeerServiceBundle", "create_peer_service_bundle"]