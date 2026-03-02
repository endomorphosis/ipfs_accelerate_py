"""Canonical FastAPI configuration helpers for unified MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UnifiedFastAPIConfig:
    """Runtime config for canonical MCP FastAPI service wrappers."""

    host: str = "0.0.0.0"
    port: int = 8000
    mount_path: str = "/mcp"
    name: str = "ipfs-accelerate-mcp"
    description: str = "IPFS Accelerate MCP Server"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "UnifiedFastAPIConfig":
        """Build config from environment variables.

        Supported environment variables:
        - `IPFS_MCP_HOST`
        - `IPFS_MCP_PORT`
        - `IPFS_MCP_MOUNT_PATH`
        - `IPFS_MCP_NAME`
        - `IPFS_MCP_DESCRIPTION`
        - `IPFS_MCP_VERBOSE` (`1`, `true`, `yes`, `on`)
        """
        verbose_raw = str(os.getenv("IPFS_MCP_VERBOSE", "")).strip().lower()
        verbose = verbose_raw in {"1", "true", "yes", "on"}

        port_raw = str(os.getenv("IPFS_MCP_PORT", "")).strip()
        try:
            port = int(port_raw) if port_raw else cls.port
        except ValueError:
            port = cls.port

        return cls(
            host=str(os.getenv("IPFS_MCP_HOST", cls.host) or cls.host),
            port=port,
            mount_path=str(os.getenv("IPFS_MCP_MOUNT_PATH", cls.mount_path) or cls.mount_path),
            name=str(os.getenv("IPFS_MCP_NAME", cls.name) or cls.name),
            description=str(os.getenv("IPFS_MCP_DESCRIPTION", cls.description) or cls.description),
            verbose=verbose,
        )
