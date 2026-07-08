"""Minimal tool wrapper utilities for mcp_server registry compatibility."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class BaseMCPTool(ABC):
    """Base class for class-based MCP tools."""

    def __init__(self) -> None:
        self.name: str = ""
        self.description: str = ""
        self.input_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        self.category: str = "general"
        self.tags: list[str] = []
        self.version: str = "1.0.0"
        self.created_at = datetime.now(tz=timezone.utc)
        self.last_used = None
        self.usage_count = 0

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with validated parameters."""

    def get_schema(self) -> Dict[str, Any]:
        """Return tool schema metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "category": self.category,
            "tags": self.tags,
            "version": self.version,
        }


class FunctionToolWrapper(BaseMCPTool):
    """Wrap a plain callable as a `BaseMCPTool`-compatible object."""

    def __init__(
        self,
        function: Callable[..., Any],
        tool_name: str,
        category: str = "general",
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self._function = function
        self.name = tool_name
        self.category = category
        self.description = description or (function.__doc__ or f"Execute {tool_name}")
        self.tags = tags or []
        self.input_schema = _signature_to_schema(function)

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wrapped sync/async function and normalize output."""
        if inspect.iscoroutinefunction(self._function):
            result = await self._function(**parameters)
        else:
            result = self._function(**parameters)

        if isinstance(result, dict):
            return result
        return {"result": result}


def wrap_function_as_tool(
    function: Callable[..., Any],
    tool_name: str,
    category: str = "general",
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> FunctionToolWrapper:
    """Create a `FunctionToolWrapper` from a plain callable."""
    return FunctionToolWrapper(
        function=function,
        tool_name=tool_name,
        category=category,
        description=description,
        tags=tags,
    )


def _signature_to_schema(function: Callable[..., Any]) -> Dict[str, Any]:
    """Build a lightweight JSON schema from a callable signature."""
    sig = inspect.signature(function)
    properties: Dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue

        prop: Dict[str, Any] = {"type": _annotation_to_json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _annotation_to_json_type(annotation: Any) -> str:
    """Translate Python annotations into a coarse JSON schema type."""
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation in {dict, Dict}:
        return "object"
    if annotation in {list, tuple}:
        return "array"
    return "string"
