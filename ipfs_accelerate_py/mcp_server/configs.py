"""Configuration surface for unified MCP server bootstrap and routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List


def env_enabled(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_preload_categories(value: str | None, allowed: List[str]) -> List[str]:
    """Parse preload categories from env/config string.

    Accepts comma-separated category names or the special value `all`.
    Unknown category names are ignored.
    """
    if not value:
        return []

    raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw:
        return []

    if "all" in raw:
        return list(allowed)

    allowed_set = set(allowed)
    return [name for name in raw if name in allowed_set]


@dataclass
class UnifiedMCPServerConfig:
    """Configuration for unified MCP bootstrap behavior."""

    enable_unified_bridge: bool = False
    enable_unified_bootstrap: bool = False
    preload_categories: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, *, allowed_preload_categories: List[str]) -> "UnifiedMCPServerConfig":
        """Create config from environment variables.

        Environment variables:
        - `IPFS_MCP_ENABLE_UNIFIED_BRIDGE`
        - `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP`
        - `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES`
        """
        return cls(
            enable_unified_bridge=env_enabled("IPFS_MCP_ENABLE_UNIFIED_BRIDGE", default=False),
            enable_unified_bootstrap=env_enabled("IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP", default=False),
            preload_categories=parse_preload_categories(
                os.environ.get("IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES", ""),
                allowed_preload_categories,
            ),
        )


__all__ = [
    "UnifiedMCPServerConfig",
    "env_enabled",
    "parse_preload_categories",
]
