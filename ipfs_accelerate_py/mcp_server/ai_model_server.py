"""Canonical ai_model_server shim for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.ai_model_server``.
Re-exports from the legacy module while it is present.
"""

from __future__ import annotations

from typing import Any

try:
    from ipfs_accelerate_py.mcp.ai_model_server import (  # type: ignore[import]
        create_ai_model_server,
    )
except Exception:
    def create_ai_model_server(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub: returns None when legacy module is unavailable."""
        return None


__all__ = ["create_ai_model_server"]
