"""Registration adapters for incremental MCP migration.

This module bridges legacy `ipfs_accelerate_py.mcp.tools` registrations into
`ipfs_accelerate_py.mcp_server.HierarchicalToolManager` so migration can proceed
without a big-bang rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .hierarchical_tool_manager import HierarchicalToolManager

logger = logging.getLogger(__name__)


@dataclass
class LegacyToolRecord:
    """Captured tool registration metadata from legacy mcp tool registrars."""

    name: str
    function: Callable[..., Any]
    description: str
    input_schema: Dict[str, Any]
    execution_context: Optional[str] = None
    tags: Optional[List[str]] = None


class LegacyCollectorMCP:
    """Collector that emulates enough MCP API for legacy register_* functions."""

    def __init__(self) -> None:
        self.tools: Dict[str, LegacyToolRecord] = {}

    def register_tool(
        self,
        name: str | None = None,
        function: Callable[..., Any] | None = None,
        description: str = "",
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        tags: List[str] | None = None,
        *,
        category: str | None = None,
        func: Callable[..., Any] | None = None,
        runtime: str | None = None,
    ) -> None:
        """Collect explicit register_tool calls from legacy or hierarchical callers."""
        del category

        resolved_name = str(name or "").strip()
        resolved_function = function or func
        resolved_schema = input_schema or {"type": "object", "properties": {}, "required": []}
        resolved_execution_context = execution_context
        if resolved_execution_context is None:
            if runtime == "trio":
                resolved_execution_context = "worker"
            elif runtime == "fastapi":
                resolved_execution_context = "server"

        if not resolved_name:
            raise ValueError("register_tool requires a non-empty name")
        if not callable(resolved_function):
            raise ValueError(f"register_tool requires a callable function for '{resolved_name}'")

        self.tools[resolved_name] = LegacyToolRecord(
            name=resolved_name,
            function=resolved_function,
            description=description,
            input_schema=resolved_schema,
            execution_context=resolved_execution_context,
            tags=tags,
        )

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: Dict[str, Any] | None = None,
        execution_context: str | None = None,
        tags: List[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator-compatible collector for `@mcp.tool()` style registrations."""

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = str(name or func.__name__)
            tool_description = description or (func.__doc__ or "")
            schema = input_schema or {"type": "object", "properties": {}, "required": []}
            self.register_tool(
                name=tool_name,
                function=func,
                description=tool_description,
                input_schema=schema,
                execution_context=execution_context,
                tags=tags,
            )
            return func

        return _decorator


def collect_legacy_mcp_tools(include_p2p_taskqueue_tools: bool = True) -> Dict[str, LegacyToolRecord]:
    """Collect tools from existing `ipfs_accelerate_py.mcp.tools` register flow."""
    from ipfs_accelerate_py.mcp.tools import register_all_tools

    collector = LegacyCollectorMCP()
    register_all_tools(collector, include_p2p_taskqueue_tools=include_p2p_taskqueue_tools)
    return collector.tools


def register_legacy_tools_into_manager(
    manager: HierarchicalToolManager,
    default_category: str = "legacy_mcp",
    include_p2p_taskqueue_tools: bool = True,
) -> int:
    """Register collected legacy tools into the hierarchical manager.

    Category mapping heuristic:
    - If a tool has a prefix like `github_foo`, category = `github`.
    - Else category = `default_category`.
    """
    records = collect_legacy_mcp_tools(include_p2p_taskqueue_tools=include_p2p_taskqueue_tools)

    count = 0
    for record in records.values():
        category = _category_from_tool_name(record.name, fallback=default_category)
        manager.register_tool(
            category=category,
            name=record.name,
            func=record.function,
            description=record.description,
            input_schema=record.input_schema,
            runtime=_runtime_from_execution_context(record.execution_context),
            tags=record.tags,
        )
        count += 1

    logger.info("Registered %d legacy tools into hierarchical manager", count)
    return count


def _category_from_tool_name(name: str, fallback: str) -> str:
    """Infer category from tool naming convention."""
    allowed_prefixes = {
        "github",
        "docker",
        "hardware",
        "runner",
        "ipfs",
        "network",
        "model",
        "models",
        "inference",
        "endpoint",
        "endpoints",
        "status",
        "workflow",
        "workflows",
        "dashboard",
        "manifest",
        "p2p",
    }
    if "_" in name:
        prefix = name.split("_", 1)[0].strip().lower()
        if prefix in allowed_prefixes:
            return prefix
    return fallback


def _runtime_from_execution_context(execution_context: Optional[str]) -> Optional[str]:
    """Map legacy execution context to runtime-router values."""
    if execution_context == "worker":
        return "trio"
    if execution_context == "server":
        return "fastapi"
    return None


__all__ = [
    "LegacyCollectorMCP",
    "LegacyToolRecord",
    "collect_legacy_mcp_tools",
    "register_legacy_tools_into_manager",
]
