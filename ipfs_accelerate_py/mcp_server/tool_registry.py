"""Tool registry for the unified MCP server package.

This module ports the core registry behavior from
`ipfs_datasets_py.mcp_server.tool_registry` and adapts imports to the
`ipfs_accelerate_py.mcp_server` package.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .exceptions import (
    ConfigurationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ValidationError as MCPValidationError,
)
from .tools.tool_wrapper import BaseMCPTool, wrap_function_as_tool

logger = logging.getLogger(__name__)


class ClaudeMCPTool(ABC):
    """Base class for class-based MCP tools used by the registry."""

    def __init__(self) -> None:
        self.name: str = ""
        self.description: str = ""
        self.input_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        self.category: str = "general"
        self.tags: List[str] = []
        self.version: str = "1.0.0"
        self.created_at = datetime.now(tz=timezone.utc)
        self.last_used = None
        self.usage_count = 0

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the provided parameters."""

    def get_schema(self) -> Dict[str, Any]:
        """Return schema metadata for discovery APIs."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "category": self.category,
            "tags": self.tags,
            "version": self.version,
        }

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Convenience entrypoint that updates usage metadata."""
        self.usage_count += 1
        self.last_used = datetime.now(tz=timezone.utc)
        return await self.execute(kwargs)


class ToolRegistry:
    """Registry for MCP tools with category/tag indexing and execution helpers."""

    def __init__(self) -> None:
        self._tools: Dict[str, Union[ClaudeMCPTool, BaseMCPTool]] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        self.total_executions = 0
        logger.info("Tool registry initialized")

    def register_tool(self, tool: Union[ClaudeMCPTool, BaseMCPTool]) -> None:
        """Register a class-based tool object."""
        if not isinstance(tool, (ClaudeMCPTool, BaseMCPTool)):
            raise ToolRegistrationError("Tool must inherit from ClaudeMCPTool or BaseMCPTool")

        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            raise ToolRegistrationError("Tool name must be non-empty")

        if tool_name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting", tool_name)

        self._tools[tool_name] = tool

        category = str(getattr(tool, "category", "general") or "general")
        tags = [str(tag) for tag in (getattr(tool, "tags", []) or []) if str(tag).strip()]

        self._categories.setdefault(category, [])
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)

        for tag in tags:
            self._tags.setdefault(tag, [])
            if tool_name not in self._tags[tag]:
                self._tags[tag].append(tool_name)

    def register_function(
        self,
        function: Any,
        tool_name: Optional[str] = None,
        category: str = "general",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a plain function by wrapping it as a tool."""
        if not callable(function):
            raise ToolRegistrationError("function must be callable")

        effective_name = tool_name or getattr(function, "__name__", "")
        if not effective_name:
            raise ToolRegistrationError("Unable to infer a tool name from function")

        wrapped = wrap_function_as_tool(
            function,
            effective_name,
            category=category,
            description=description,
            tags=tags,
        )
        self.register_tool(wrapped)

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool and clean category/tag indexes."""
        if tool_name not in self._tools:
            return False

        tool = self._tools[tool_name]
        category = str(getattr(tool, "category", "general") or "general")
        tags = [str(tag) for tag in (getattr(tool, "tags", []) or []) if str(tag).strip()]

        if category in self._categories and tool_name in self._categories[category]:
            self._categories[category].remove(tool_name)
            if not self._categories[category]:
                del self._categories[category]

        for tag in tags:
            if tag in self._tags and tool_name in self._tags[tag]:
                self._tags[tag].remove(tool_name)
                if not self._tags[tag]:
                    del self._tags[tag]

        del self._tools[tool_name]
        return True

    def get_tool(self, tool_name: str) -> Optional[Union[ClaudeMCPTool, BaseMCPTool]]:
        """Return a registered tool by name."""
        return self._tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check whether a tool is registered."""
        return tool_name in self._tools

    def get_all_tools(self) -> List[Union[ClaudeMCPTool, BaseMCPTool]]:
        """Return all registered tool objects."""
        return list(self._tools.values())

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all tool schemas."""
        output: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            if hasattr(tool, "get_schema"):
                output.append(tool.get_schema())
            else:
                output.append(
                    {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "input_schema": getattr(tool, "input_schema", {}),
                        "category": getattr(tool, "category", "general"),
                        "tags": getattr(tool, "tags", []),
                        "version": getattr(tool, "version", "1.0.0"),
                    }
                )
        return output

    def get_tools_by_category(self, category: str) -> List[Union[ClaudeMCPTool, BaseMCPTool]]:
        """Return tools under a category."""
        names = self._categories.get(category, [])
        return [self._tools[name] for name in names if name in self._tools]

    def get_tools_by_tag(self, tag: str) -> List[Union[ClaudeMCPTool, BaseMCPTool]]:
        """Return tools carrying a tag."""
        names = self._tags.get(tag, [])
        return [self._tools[name] for name in names if name in self._tools]

    def get_categories(self) -> List[str]:
        """Return known category names."""
        return list(self._categories.keys())

    def get_tags(self) -> List[str]:
        """Return known tag names."""
        return list(self._tags.keys())

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolNotFoundError(tool_name)

        self.total_executions += 1

        try:
            execute = getattr(tool, "execute", None)
            if execute is None or not callable(execute):
                raise ConfigurationError(f"Tool '{tool_name}' has no callable execute()")

            if inspect.iscoroutinefunction(execute):
                result = await execute(parameters)
            else:
                result = execute(parameters)

            if isinstance(result, dict):
                return result
            return {"result": result}
        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.error("Tool '%s' execution failed: %s", tool_name, exc, exc_info=True)
            raise ToolExecutionError(tool_name, exc)

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Return usage and inventory metrics for registry state."""
        return {
            "total_tools": len(self._tools),
            "total_executions": self.total_executions,
            "categories": {name: len(tools) for name, tools in self._categories.items()},
            "tags": {name: len(tools) for name, tools in self._tags.items()},
            "tool_usage": {
                name: {
                    "usage_count": getattr(tool, "usage_count", 0),
                    "last_used": getattr(tool, "last_used", None).isoformat()
                    if getattr(tool, "last_used", None)
                    else None,
                    "category": getattr(tool, "category", "general"),
                }
                for name, tool in self._tools.items()
            },
        }

    def search_tools(self, query: str) -> List[Union[ClaudeMCPTool, BaseMCPTool]]:
        """Search tools by name, description, and tags."""
        needle = str(query or "").strip().lower()
        if not needle:
            return []

        matches: List[Union[ClaudeMCPTool, BaseMCPTool]] = []
        for tool in self._tools.values():
            name = str(getattr(tool, "name", "")).lower()
            description = str(getattr(tool, "description", "")).lower()
            tags = [str(tag).lower() for tag in (getattr(tool, "tags", []) or [])]
            if needle in name or needle in description or any(needle in tag for tag in tags):
                matches.append(tool)
        return matches

    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate required parameters based on a tool input schema."""
        tool = self.get_tool(tool_name)
        if not tool:
            return False

        schema = getattr(tool, "input_schema", {}) or {}
        required = schema.get("required", [])

        if not isinstance(parameters, dict):
            raise MCPValidationError("parameters", "Parameters must be a dictionary")

        for required_param in required:
            if required_param not in parameters:
                return False
        return True


_global_registry: Optional[ToolRegistry] = None


def get_global_registry() -> ToolRegistry:
    """Return a process-level singleton registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    tool: Union[ClaudeMCPTool, BaseMCPTool],
    registry: Optional[ToolRegistry] = None,
) -> None:
    """Register a tool in the provided or global registry."""
    (registry or get_global_registry()).register_tool(tool)


def register_function(
    function: Any,
    tool_name: Optional[str] = None,
    category: str = "general",
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> None:
    """Register a plain function in the provided or global registry."""
    (registry or get_global_registry()).register_function(
        function=function,
        tool_name=tool_name,
        category=category,
        description=description,
        tags=tags,
    )


__all__ = [
    "ClaudeMCPTool",
    "ToolRegistry",
    "get_global_registry",
    "register_tool",
    "register_function",
]
