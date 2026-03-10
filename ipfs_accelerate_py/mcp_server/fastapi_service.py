"""Canonical FastAPI service facade for unified MCP runtime.

This module provides a stable HTTP entrypoint in ``ipfs_accelerate_py.mcp_server``
while delegating to proven integration helpers under ``ipfs_accelerate_py.mcp``.
"""

from __future__ import annotations

from typing import Any

from .fastapi_config import UnifiedFastAPIConfig


_DEFAULT_CONFIG: UnifiedFastAPIConfig | None = None
_DEFAULT_APP: Any | None = None


def get_fastapi_config() -> UnifiedFastAPIConfig:
    """Return cached canonical FastAPI settings for import-compatible callers."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = UnifiedFastAPIConfig.from_env()
    return _DEFAULT_CONFIG


def get_fastapi_app() -> Any:
    """Return cached canonical FastAPI app for import-compatible callers."""
    global _DEFAULT_APP
    if _DEFAULT_APP is None:
        _DEFAULT_APP = create_fastapi_app(get_fastapi_config())
    return _DEFAULT_APP


def create_fastapi_app(config: UnifiedFastAPIConfig | None = None) -> Any:
    """Create a standalone FastAPI-compatible app for MCP endpoints."""
    from ipfs_accelerate_py.mcp.integration import create_standalone_app

    resolved = config or UnifiedFastAPIConfig.from_env()
    return create_standalone_app(
        name=resolved.name,
        description=resolved.description,
        mount_path=resolved.mount_path,
        verbose=resolved.verbose,
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


def __getattr__(name: str) -> Any:
    """Provide lazy import-compatible `settings` and `app` module attributes."""
    if name == "settings":
        return get_fastapi_config()
    if name == "app":
        return get_fastapi_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
