"""Canonical inference_tools_compat shim for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.inference_tools``.
Re-exports from the legacy module while it is present.
"""

from __future__ import annotations

from typing import Any

try:
    from ipfs_accelerate_py.mcp.inference_tools import (  # type: ignore[import]
        create_inference_tools,
    )
except Exception:
    def create_inference_tools(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub: returns None when legacy module is unavailable."""
        return None


__all__ = ["create_inference_tools"]
