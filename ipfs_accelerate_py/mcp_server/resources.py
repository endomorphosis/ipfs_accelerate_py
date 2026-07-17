"""Canonical resources registration for ipfs_accelerate_py.mcp_server.

Drop-in replacement for ``ipfs_accelerate_py.mcp.resources``.
Delegates to the legacy implementation while it is present; provides a
no-op fallback when it is not.
"""

from __future__ import annotations

from typing import Any


def register_all_resources(mcp: Any) -> None:
    """Register all canonical MCP resources on *mcp*.

    Falls back to the legacy ``ipfs_accelerate_py.mcp.resources`` while that
    package is available.
    """
    try:
        from ipfs_accelerate_py.mcp.resources import (  # type: ignore[import]  # noqa: PLC0415
            register_all_resources as _legacy_register,
        )
        _legacy_register(mcp)
        return
    except Exception:
        pass

    # Canonical stub — no resources registered when legacy is unavailable.


__all__ = ["register_all_resources"]
