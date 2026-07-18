"""Canonical fastmcp_compat shims for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.fastmcp_compat``.
These helpers patch a FastMCP (or compatible) instance so it exposes the
``register_tool`` / ``tools`` / ``register_resource`` / ``register_prompt``
dict-based API expected by the MCP++ dispatch layer.
"""

from __future__ import annotations

from typing import Any


def ensure_register_tool_compat(mcp: Any) -> Any:
    """Ensure *mcp* has a working ``register_tool`` method and ``.tools`` dict.

    Canonical version: delegates to the legacy implementation while the legacy
    package is present; falls back to a no-op if the canonical StandaloneMCP
    already provides the interface.
    """
    # Canonical StandaloneMCP already provides register_tool / tools — nothing to do.
    from ipfs_accelerate_py.mcp_server.server import StandaloneMCP  # noqa: PLC0415
    if isinstance(mcp, StandaloneMCP):
        return mcp

    try:
        from ipfs_accelerate_py.mcp.fastmcp_compat import (  # type: ignore[import]  # noqa: PLC0415
            ensure_register_tool_compat as _legacy,
        )
        return _legacy(mcp)
    except Exception:
        return mcp


def ensure_register_resource_compat(mcp: Any) -> Any:
    """Ensure *mcp* has a working ``register_resource`` method and ``.resources`` dict."""
    from ipfs_accelerate_py.mcp_server.server import StandaloneMCP  # noqa: PLC0415
    if isinstance(mcp, StandaloneMCP):
        return mcp

    try:
        from ipfs_accelerate_py.mcp.fastmcp_compat import (  # type: ignore[import]  # noqa: PLC0415
            ensure_register_resource_compat as _legacy,
        )
        return _legacy(mcp)
    except Exception:
        return mcp


def ensure_register_prompt_compat(mcp: Any) -> Any:
    """Ensure *mcp* has a working ``register_prompt`` method."""
    from ipfs_accelerate_py.mcp_server.server import StandaloneMCP  # noqa: PLC0415
    if isinstance(mcp, StandaloneMCP):
        return mcp

    try:
        from ipfs_accelerate_py.mcp.fastmcp_compat import (  # type: ignore[import]  # noqa: PLC0415
            ensure_register_prompt_compat as _legacy,
        )
        return _legacy(mcp)
    except Exception:
        return mcp


__all__ = [
    "ensure_register_tool_compat",
    "ensure_register_resource_compat",
    "ensure_register_prompt_compat",
]
