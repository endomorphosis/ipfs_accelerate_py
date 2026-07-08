"""Tool metadata system for unified mcp_server runtime routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

RUNTIME_FASTAPI = "fastapi"
RUNTIME_TRIO = "trio"
RUNTIME_AUTO = "auto"


@dataclass(frozen=True)
class ToolMetadata:
    """Metadata for a single tool runtime contract."""

    name: str
    runtime: str = RUNTIME_AUTO
    category: str = "general"
    requires_p2p: bool = False
    priority: int = 5
    timeout_seconds: Optional[float] = 30.0
    mcp_description: Optional[str] = None
    mcp_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    _func: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.runtime not in {RUNTIME_FASTAPI, RUNTIME_TRIO, RUNTIME_AUTO}:
            raise ValueError(f"Invalid runtime: {self.runtime}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata excluding internal function handle."""
        payload = asdict(self)
        payload.pop("_func", None)
        return payload


class ToolMetadataRegistry:
    """Registry storing tool metadata by name/category/runtime."""

    def __init__(self) -> None:
        self._registry: Dict[str, ToolMetadata] = {}
        self._by_runtime: Dict[str, Set[str]] = {
            RUNTIME_FASTAPI: set(),
            RUNTIME_TRIO: set(),
            RUNTIME_AUTO: set(),
        }
        self._by_category: Dict[str, Set[str]] = {}

    def register(self, metadata: ToolMetadata) -> None:
        """Register or replace metadata for a tool name."""
        existing = self._registry.get(metadata.name)
        if existing:
            self._by_runtime.get(existing.runtime, set()).discard(existing.name)
            if existing.category in self._by_category:
                self._by_category[existing.category].discard(existing.name)

        self._registry[metadata.name] = metadata
        self._by_runtime[metadata.runtime].add(metadata.name)
        self._by_category.setdefault(metadata.category, set()).add(metadata.name)

    def get(self, tool_name: str) -> Optional[ToolMetadata]:
        """Return metadata for a tool name."""
        return self._registry.get(tool_name)

    def list_all(self) -> List[ToolMetadata]:
        """List all metadata records."""
        return list(self._registry.values())

    def list_by_runtime(self, runtime: str) -> List[ToolMetadata]:
        """List metadata records for a runtime."""
        names = self._by_runtime.get(runtime, set())
        return [self._registry[name] for name in names]

    def list_by_category(self, category: str) -> List[ToolMetadata]:
        """List metadata records for a category."""
        names = self._by_category.get(category, set())
        return [self._registry[name] for name in names]


_REGISTRY: Optional[ToolMetadataRegistry] = None


def get_registry() -> ToolMetadataRegistry:
    """Return global metadata registry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolMetadataRegistry()
    return _REGISTRY


def register_tool_metadata(metadata: ToolMetadata, registry: Optional[ToolMetadataRegistry] = None) -> None:
    """Register metadata in provided or global registry."""
    (registry or get_registry()).register(metadata)


def get_tool_metadata(tool_name: str, registry: Optional[ToolMetadataRegistry] = None) -> Optional[ToolMetadata]:
    """Retrieve metadata by tool name from provided or global registry."""
    return (registry or get_registry()).get(tool_name)


def tool_metadata(
    *,
    runtime: str = RUNTIME_AUTO,
    category: str = "general",
    requires_p2p: bool = False,
    priority: int = 5,
    timeout_seconds: Optional[float] = 30.0,
    description: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that attaches and registers metadata for a tool function."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        meta = ToolMetadata(
            name=func.__name__,
            runtime=runtime,
            category=category,
            requires_p2p=requires_p2p,
            priority=priority,
            timeout_seconds=timeout_seconds,
            mcp_description=description or (func.__doc__ or ""),
            mcp_schema=schema,
            tags=tags or [],
            _func=func,
        )
        register_tool_metadata(meta)
        setattr(func, "__mcp_metadata__", meta)
        setattr(func, "__mcp_runtime__", runtime)
        return func

    return _decorator


__all__ = [
    "RUNTIME_FASTAPI",
    "RUNTIME_TRIO",
    "RUNTIME_AUTO",
    "ToolMetadata",
    "ToolMetadataRegistry",
    "get_registry",
    "register_tool_metadata",
    "get_tool_metadata",
    "tool_metadata",
]
