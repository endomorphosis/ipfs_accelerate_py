"""Manifest/introspection tools for the MCP server.

.. deprecated::
    This module has been migrated to the canonical runtime at
    ``ipfs_accelerate_py.mcp_server.tools.monitoring_tools``.  Import from the canonical module instead.
    This file is preserved as a compatibility shim only.
"""

from __future__ import annotations

from typing import Any, Dict


def register_tools(mcp: Any) -> None:
    import warnings
    warnings.warn(
        "ipfs_accelerate_py.mcp.tools.manifest.register_tools is deprecated. "
        "Use ipfs_accelerate_py.mcp_server.tools.monitoring_tools instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    @mcp.tool()
    def get_mcp_manifest(include_schemas: bool = True) -> Dict[str, Any]:
        """Return a JSON-friendly list of registered MCP tools/resources/prompts."""

        from ipfs_accelerate_py.tool_manifest import extract_mcp_manifest

        return extract_mcp_manifest(mcp, include_schemas=bool(include_schemas))

    tools_dict = getattr(mcp, "tools", None)
    if isinstance(tools_dict, dict):
        entry = tools_dict.get("get_mcp_manifest")
        if isinstance(entry, dict):
            entry["execution_context"] = "server"
