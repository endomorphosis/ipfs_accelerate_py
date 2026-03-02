"""Canonical Trio transport adapter facade for unified MCP runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .server import create_server

try:
    import trio as _trio  # noqa: F401

    TRIO_AVAILABLE = True
except Exception:
    TRIO_AVAILABLE = False


@dataclass(slots=True)
class TrioServerConfig:
    """Runtime configuration for Trio adapter startup."""

    host: str = "127.0.0.1"
    port: int = 7000
    name: str = "ipfs-accelerate"
    description: str = "IPFS Accelerate MCP Server"


class TrioMCPServerAdapter:
    """Lightweight canonical facade for Trio transport lifecycle."""

    def __init__(
        self,
        config: TrioServerConfig | None = None,
        *,
        server_factory: Callable[..., Any] | None = None,
        serve_fn: Callable[[Any, TrioServerConfig], Any] | None = None,
    ) -> None:
        self.config = config or TrioServerConfig()
        self._server_factory = server_factory or create_server
        self._serve_fn = serve_fn
        self.server: Any | None = None
        self._running = False

    @property
    def running(self) -> bool:
        """Return whether adapter has an active server lifecycle."""
        return bool(self._running)

    async def start(self) -> Any:
        """Create the canonical server and invoke optional Trio serve hook."""
        if self._running and self.server is not None:
            return self.server

        self.server = self._server_factory(name=self.config.name, description=self.config.description)
        self._running = True

        if self._serve_fn is not None:
            result = self._serve_fn(self.server, self.config)
            if inspect.isawaitable(result):
                return await result
            return result

        return self.server

    async def stop(self) -> None:
        """Stop lifecycle and close wrapped server if close hooks exist."""
        if not self._running:
            return

        server = self.server
        self._running = False
        self.server = None
        if server is None:
            return

        for method_name in ("stop", "aclose", "close"):
            method = getattr(server, method_name, None)
            if callable(method):
                result = method()
                if inspect.isawaitable(result):
                    await result
                return


__all__ = ["TRIO_AVAILABLE", "TrioServerConfig", "TrioMCPServerAdapter"]
