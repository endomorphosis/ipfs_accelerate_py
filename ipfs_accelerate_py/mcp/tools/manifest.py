"""Manifest/introspection tools for the MCP server."""

from __future__ import annotations

from typing import Any, Dict


def register_tools(mcp: Any) -> None:
    @mcp.tool()
    def get_mcp_manifest(include_schemas: bool = True) -> Dict[str, Any]:
        """Return a JSON-friendly list of registered MCP tools/resources/prompts."""

        from ipfs_accelerate_py.tool_manifest import extract_mcp_manifest

        return extract_mcp_manifest(mcp, include_schemas=bool(include_schemas))
