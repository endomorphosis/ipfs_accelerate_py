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

        Legacy compatibility fallbacks (when canonical keys are unset):
        - `HOST`, `PORT`, `MOUNT_PATH`, `APP_NAME`, `APP_DESCRIPTION`, `DEBUG`
        """
        def _env(primary: str, fallback: str, default: str) -> str:
            raw = os.getenv(primary)
            if raw is None or str(raw).strip() == "":
                raw = os.getenv(fallback)
            return str(raw if raw is not None else default)

        verbose_raw = _env("IPFS_MCP_VERBOSE", "DEBUG", "").strip().lower()
        verbose = verbose_raw in {"1", "true", "yes", "on"}

        port_raw = _env("IPFS_MCP_PORT", "PORT", "").strip()
        try:
            port = int(port_raw) if port_raw else cls.port
        except ValueError:
            port = cls.port

        return cls(
            host=_env("IPFS_MCP_HOST", "HOST", cls.host).strip() or cls.host,
            port=port,
            mount_path=_env("IPFS_MCP_MOUNT_PATH", "MOUNT_PATH", cls.mount_path).strip() or cls.mount_path,
            name=_env("IPFS_MCP_NAME", "APP_NAME", cls.name).strip() or cls.name,
            description=_env("IPFS_MCP_DESCRIPTION", "APP_DESCRIPTION", cls.description).strip() or cls.description,
            verbose=verbose,
        )
