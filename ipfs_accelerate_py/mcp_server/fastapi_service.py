"""Canonical FastAPI service facade for unified MCP runtime.

This module provides a stable HTTP entrypoint in ``ipfs_accelerate_py.mcp_server``
while delegating to proven integration helpers under ``ipfs_accelerate_py.mcp``.
"""

from __future__ import annotations

from typing import Any

from .fastapi_config import UnifiedFastAPIConfig


def create_fastapi_app(config: UnifiedFastAPIConfig | None = None) -> Any:
    """Create a standalone FastAPI-compatible app for MCP endpoints."""
    from ipfs_accelerate_py.mcp.integration import create_standalone_app

    resolved = config or UnifiedFastAPIConfig.from_env()
    return create_standalone_app(
        name=resolved.name,
        description=resolved.description,
        mount_path=resolved.mount_path,
    )


def run_fastapi_server(config: UnifiedFastAPIConfig | None = None) -> None:
    """Run canonical MCP FastAPI service using integration runner."""
    from ipfs_accelerate_py.mcp.integration import run_standalone_app

    resolved = config or UnifiedFastAPIConfig.from_env()
    app = create_fastapi_app(resolved)
    run_standalone_app(
        app,
        host=resolved.host,
        port=resolved.port,
        verbose=resolved.verbose,
    )


def main() -> None:
    """Entrypoint for `python -m ipfs_accelerate_py.mcp_server.fastapi_service`."""
    run_fastapi_server()


if __name__ == "__main__":
    main()
