"""Legacy MCP client facade for canonical runtime compatibility."""

from __future__ import annotations

from typing import Any, Dict, Optional


class IPFSDatasetsMCPClient:
    """Source-compatible client wrapper around an MCP-like local registry."""

    def __init__(self, server_url: str, mcp_like: Any = None):
        self.server_url = str(server_url or "")
        self._mcp_like = mcp_like

    async def get_available_tools(self):
        if self._mcp_like is None:
            return []
        from ipfs_accelerate_py.tool_manifest import extract_mcp_manifest

        manifest = extract_mcp_manifest(self._mcp_like, include_schemas=False)
        return list(manifest.get("tools", []))

    async def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._mcp_like is None:
            return {"ok": False, "error": "mcp_registry_unavailable", "tool": str(tool_name)}

        from ipfs_accelerate_py.tool_manifest import invoke_mcp_tool

        return await invoke_mcp_tool(
            self._mcp_like,
            tool_name=str(tool_name),
            args=dict(params or {}),
            accelerate_instance=None,
        )


__all__ = ["IPFSDatasetsMCPClient"]
