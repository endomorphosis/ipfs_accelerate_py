"""Canonical mock MCP module for ipfs_accelerate_py.mcp_server.

Provides ``FastMCP`` and ``Context`` stubs for testing and development
without the full FastMCP package.  This is a canonical migration target
for code that previously imported from ``ipfs_accelerate_py.mcp.mock_mcp``.

Drop-in migration::

    # Old (deprecated):
    from ipfs_accelerate_py.mcp.mock_mcp import FastMCP, Context

    # New (canonical):
    from ipfs_accelerate_py.mcp_server.mock_mcp import FastMCP, Context
"""

from __future__ import annotations

# Always use the canonical StandaloneMCP so that register_tool() works
# consistently regardless of whether the legacy mcp package is present.
from .server import StandaloneMCP as FastMCP  # noqa: F401


class Context:
    """Minimal MCP context stub for testing."""

    def __init__(self) -> None:
        self.state = None


__all__ = ["FastMCP", "Context"]
