"""Shim module to support tests importing .github_tools."""

from ipfs_accelerate_py.mcp.tools.github_tools import register_github_tools

__all__ = ["register_github_tools"]
