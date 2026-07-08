"""Hierarchical tool manager for unified MCP server.

This module provides category-level tool discovery and dispatch so MCP clients
can use a compact meta-tool interface instead of loading all tool schemas.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .exceptions import ConfigurationError, ToolExecutionError, ToolNotFoundError
from .runtime_router import RuntimeRouter
from .tool_metadata import (
    RUNTIME_AUTO,
    ToolMetadata,
    register_tool_metadata,
)
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class CircuitState:
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple per-category circuit breaker for dispatch resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, name: str = "default") -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        """Return current state while auto-transitioning OPEN -> HALF_OPEN on timeout."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _on_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    async def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a callable under breaker protection."""
        if self.state == CircuitState.OPEN:
            raise ToolExecutionError(self.name, "Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise


@dataclass
class ToolDefinition:
    """Definition of one registered tool in a category."""

    category: str
    name: str
    func: Callable[..., Any]
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    runtime: Optional[str] = None


class HierarchicalToolManager:
    """Manage MCP tools by category with lazy category loading and dispatch."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        runtime_router: Optional[RuntimeRouter] = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.runtime_router = runtime_router
        self._categories: Dict[str, Dict[str, ToolDefinition]] = {}
        self._category_loaders: Dict[str, Callable[["HierarchicalToolManager"], None]] = {}
        self._category_loaded: Dict[str, bool] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def register_category_loader(self, category: str, loader: Callable[["HierarchicalToolManager"], None]) -> None:
        """Register a lazy loader for a category.

        Loader should register tools for that category when called.
        """
        if not callable(loader):
            raise ConfigurationError(f"Loader for category '{category}' must be callable")
        self._category_loaders[category] = loader
        self._category_loaded.setdefault(category, False)

    def register_tool(
        self,
        category: str,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        runtime: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a callable in a category and the shared registry."""
        if not callable(func):
            raise ConfigurationError(f"Tool '{category}.{name}' must be callable")

        self._categories.setdefault(category, {})
        self._categories[category][name] = ToolDefinition(
            category=category,
            name=name,
            func=func,
            description=description or (func.__doc__ or ""),
            input_schema=input_schema,
            runtime=runtime,
        )

        self.registry.register_function(
            function=func,
            tool_name=f"{category}.{name}",
            category=category,
            description=description,
            tags=tags,
        )

        if runtime and self.runtime_router:
            self.runtime_router.register_tool_runtime(f"{category}.{name}", runtime)

        register_tool_metadata(
            ToolMetadata(
                name=f"{category}.{name}",
                runtime=runtime or RUNTIME_AUTO,
                category=category,
                mcp_description=description or (func.__doc__ or ""),
                mcp_schema=input_schema,
                tags=tags or [],
            )
        )

        self._category_loaded[category] = True

    def list_categories(self) -> List[str]:
        """List known category names."""
        names = set(self._categories.keys()) | set(self._category_loaders.keys())
        return sorted(names)

    def _ensure_category_loaded(self, category: str) -> None:
        """Load category via its lazy loader exactly once."""
        if self._category_loaded.get(category):
            return
        loader = self._category_loaders.get(category)
        if loader is None:
            self._category_loaded[category] = True
            return

        logger.info("Loading tool category '%s'", category)
        loader(self)
        self._category_loaded[category] = True

    def list_tools(self, category: str) -> List[Dict[str, str]]:
        """List tool names and descriptions for a category."""
        self._ensure_category_loaded(category)
        tools = self._categories.get(category, {})
        return [
            {"name": name, "description": (tool.description or "").strip().split("\n")[0]}
            for name, tool in sorted(tools.items())
        ]

    def get_tool_schema(self, category: str, tool_name: str) -> Dict[str, Any]:
        """Return schema metadata for one tool."""
        self._ensure_category_loaded(category)
        tool = self._categories.get(category, {}).get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"{category}.{tool_name}")

        schema = tool.input_schema or _signature_schema(tool.func)
        return {
            "name": tool.name,
            "category": tool.category,
            "description": tool.description,
            "runtime": tool.runtime,
            "input_schema": schema,
        }

    async def dispatch(self, category: str, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Dispatch a category tool with optional runtime routing and breaker support."""
        self._ensure_category_loaded(category)
        tool = self._categories.get(category, {}).get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"{category}.{tool_name}")

        breaker = self._breakers.setdefault(
            category,
            CircuitBreaker(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
                name=category,
            ),
        )

        async def _invoke() -> Any:
            if self.runtime_router:
                full_name = f"{category}.{tool_name}"
                return await self.runtime_router.route_tool_call(full_name, tool.func, **parameters)

            result = tool.func(**parameters)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            return await breaker.run(_invoke)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{category}.{tool_name}", exc) from exc

    def get_breaker_state(self, category: str) -> Dict[str, Any]:
        """Return circuit-breaker state details for one category."""
        breaker = self._breakers.get(category)
        if breaker is None:
            return {"name": category, "state": CircuitState.CLOSED, "failure_count": 0}
        return {
            "name": category,
            "state": breaker.state,
            "failure_count": breaker._failure_count,
            "failure_threshold": breaker.failure_threshold,
            "recovery_timeout": breaker.recovery_timeout,
        }


def _signature_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Build a basic schema from a function signature."""
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue

        properties[name] = {
            "type": _annotation_type(param.annotation),
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = param.default

    return {"type": "object", "properties": properties, "required": required}


def _annotation_type(annotation: Any) -> str:
    """Map Python annotations into coarse JSON types."""
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


__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "HierarchicalToolManager",
    "ToolDefinition",
]
