"""Canonical FastAPI service facade for unified MCP runtime.

This module provides a stable HTTP entrypoint in ``ipfs_accelerate_py.mcp_server``
without routing back through the legacy ``ipfs_accelerate_py.mcp`` facade.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .fastapi_config import UnifiedFastAPIConfig
from .server import create_server


logger = logging.getLogger(__name__)


class _FallbackStandaloneApp:
    """Minimal fallback app used when FastAPI is unavailable."""

    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.mounts: list[dict[str, Any]] = []
        self.routes: list[dict[str, Any]] = []

    def mount(self, path: str, app: Any, name: Optional[str] = None) -> None:
        self.mounts.append({"path": path, "app": app, "name": name})

    def add_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        methods: Optional[List[str]] = None,
    ) -> None:
        self.routes.append({"path": path, "endpoint": endpoint, "methods": methods or ["GET"]})


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
    resolved = config or UnifiedFastAPIConfig.from_env()
    title = "IPFS Accelerate MCP API"

    try:
        from fastapi import FastAPI

        app: Any = FastAPI(
            title=title,
            description=resolved.description,
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        @app.get("/healthz")
        async def _healthz() -> Dict[str, Any]:
            return {"status": "ok", "service": resolved.name}

    except ImportError:
        logger.warning("FastAPI is not installed; using fallback standalone app")
        app = _FallbackStandaloneApp(title=title, description=resolved.description)

        async def _healthz() -> Dict[str, Any]:
            return {"status": "ok", "service": resolved.name}

        app.add_route("/healthz", _healthz, methods=["GET"])

    mcp_server = create_server(
        name=resolved.name,
        description=resolved.description,
        mount_path="",
    )

    mountable = getattr(mcp_server, "app", None)
    app.mount(resolved.mount_path, mountable if mountable is not None else mcp_server, name="mcp_server")
    setattr(app, "_mcp_server", mcp_server)
    return app


def run_standalone_app(app: Any, host: str = "localhost", port: int = 8000, verbose: bool = False) -> None:
    """Run a standalone FastAPI app using uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the standalone MCP app") from exc

    uvicorn.run(app, host=host, port=int(port), log_level="debug" if verbose else "info")


def run_fastapi_server(config: UnifiedFastAPIConfig | None = None) -> None:
    """Run canonical MCP FastAPI service using integration runner."""
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
