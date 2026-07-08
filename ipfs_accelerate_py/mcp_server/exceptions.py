"""Exception types for the unified MCP server package."""

from __future__ import annotations

from typing import Any


class MCPServerError(Exception):
    """Base error for mcp_server package exceptions."""


class ToolRegistrationError(MCPServerError):
    """Raised when a tool cannot be registered."""


class ToolNotFoundError(MCPServerError):
    """Raised when a tool lookup fails."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name


class ToolExecutionError(MCPServerError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, error: Exception | str):
        message = str(error)
        super().__init__(f"Tool execution failed [{tool_name}]: {message}")
        self.tool_name = tool_name
        self.error = error


class ValidationError(MCPServerError):
    """Raised when tool parameters fail validation."""

    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error in '{field}': {message}")
        self.field = field
        self.message = message


class ConfigurationError(MCPServerError):
    """Raised when registry/server configuration is invalid."""

    def __init__(self, message: str, details: Any | None = None):
        super().__init__(message)
        self.details = details


class RuntimeRoutingError(MCPServerError):
    """Raised when runtime selection/routing fails."""


class RuntimeNotFoundError(RuntimeRoutingError):
    """Raised when an unsupported runtime is requested."""

    def __init__(self, runtime: str):
        super().__init__(f"Runtime not found: {runtime}")
        self.runtime = runtime


class RuntimeExecutionError(RuntimeRoutingError):
    """Raised when runtime execution fails for a tool call."""

    def __init__(self, runtime: str, tool_name: str, error: Exception | str):
        message = str(error)
        super().__init__(f"Runtime execution failed [{runtime}] for '{tool_name}': {message}")
        self.runtime = runtime
        self.tool_name = tool_name
        self.error = error
