"""Protocol definitions for canonical MCP server runtime interfaces.

These protocols provide a dependency-light contract for adapters that need
minimal access to MCP server, tool manager, and MCP-like client capabilities.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class MCPServerProtocol(Protocol):
    """Protocol for MCP server registries used by transport adapters."""

    tools: dict[str, Any]

    def register_tool(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        input_schema: dict[str, Any],
        execution_context: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register a callable tool in the server registry."""

    def validate_p2p_message(self, message: dict[str, Any]) -> bool:
        """Validate an incoming P2P JSON-RPC message payload."""


@runtime_checkable
class ToolManagerProtocol(Protocol):
    """Protocol for hierarchical category tool managers."""

    def list_categories(self) -> list[str]:
        """List available tool categories."""

    def list_tools(self, category: str) -> list[dict[str, str]]:
        """List tools for one category."""

    def get_tool_schema(self, category: str, tool_name: str) -> dict[str, Any]:
        """Return schema metadata for one category tool."""

    async def dispatch(self, category: str, tool_name: str, parameters: dict[str, Any]) -> Any:
        """Dispatch a category tool invocation."""


@runtime_checkable
class MCPClientProtocol(Protocol):
    """Protocol for minimal MCP client behavior used by wrappers."""

    def register_tool(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
        input_schema: dict[str, Any],
        execution_context: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register a local callable tool available to server-side adapters."""
