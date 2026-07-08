"""Legacy temporal deontic MCP server facade over canonical unified runtime."""

from __future__ import annotations

from typing import Any, Dict, List

from .server import create_server


class TemporalDeonticMCPServer:
    """Compatibility facade for the source temporal deontic MCP server API."""

    def __init__(self, port: int = 8765) -> None:
        self.port = int(port)
        self.server = None

    def setup_server(self) -> Any:
        """Create canonical server instance lazily and return it."""
        if self.server is None:
            self.server = create_server(name="temporal-deontic-logic")
        return self.server

    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Return name->schema mapping from canonical tool registry."""
        server = self.setup_server()
        schemas: Dict[str, Dict[str, Any]] = {}
        for name, record in getattr(server, "tools", {}).items():
            schema = record.get("input_schema") if isinstance(record, dict) else None
            schemas[str(name)] = {
                "description": str(record.get("description", "") if isinstance(record, dict) else ""),
                "input_schema": schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
            }
        return schemas

    async def call_tool_direct(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool by name through canonical meta-tool path."""
        server = self.setup_server()
        tools = getattr(server, "tools", {})
        list_tools = tools.get("tools_list_tools", {}).get("function") if isinstance(tools, dict) else None
        dispatch = tools.get("tools_dispatch", {}).get("function") if isinstance(tools, dict) else None

        if not callable(list_tools) or not callable(dispatch):
            return {
                "success": False,
                "error": "Canonical dispatch tools unavailable",
                "error_code": "DISPATCH_UNAVAILABLE",
            }

        try:
            discovered = await list_tools()
            category = ""
            for item in discovered if isinstance(discovered, list) else []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name", "")) == str(tool_name):
                    category = str(item.get("category", ""))
                    break

            if not category:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "error_code": "UNKNOWN_TOOL",
                }

            result = await dispatch(category, str(tool_name), dict(parameters or {}))
            if isinstance(result, dict) and "success" in result:
                return result
            return {"success": True, "result": result}
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "tool": str(tool_name),
                "error_code": "TOOL_EXECUTION_ERROR",
            }


temporal_deontic_mcp_server = TemporalDeonticMCPServer()


__all__ = ["TemporalDeonticMCPServer", "temporal_deontic_mcp_server"]
