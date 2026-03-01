"""Compatibility bootstrap server for MCP unification.

This module provides a stable import location for the new canonical MCP server
package while delegating runtime behavior to the existing
`ipfs_accelerate_py.mcp.server` implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from .configs import UnifiedMCPServerConfig, parse_preload_categories
from .hierarchical_tool_manager import HierarchicalToolManager
from .runtime_router import RuntimeRouter
from .wave_a_loaders import configure_wave_a_loaders

logger = logging.getLogger(__name__)


def get_unified_meta_tool_names() -> list[str]:
    """Return canonical control-plane meta-tool names for unified MCP."""
    return [
        "tools_list_categories",
        "tools_list_tools",
        "tools_get_schema",
        "tools_dispatch",
        "tools_runtime_metrics",
    ]


def get_unified_wave_a_categories() -> list[str]:
    """Return canonical Wave A categories supported by unified bootstrap."""
    return ["ipfs", "workflow", "p2p"]


def _parse_preload_categories(value: str | None) -> list[str]:
    """Parse preload categories from env var.

    Accepts comma-separated category names or the special value `all`.
    """
    return parse_preload_categories(value, get_unified_wave_a_categories())


def _preload_configured_categories(manager: HierarchicalToolManager, preload_categories: list[str]) -> list[str]:
    """Preload selected categories (if configured) by triggering list_tools()."""
    loaded: list[str] = []
    for category in preload_categories:
        try:
            manager.list_tools(category)
            loaded.append(category)
        except Exception as exc:
            logger.warning("Failed to preload unified category '%s': %s", category, exc)

    return loaded


def _build_unified_services() -> dict[str, Any]:
    """Build lazy MCP++ service factories for unified runtime composition.

    Services are attached as callables to avoid heavy startup side-effects from
    optional runtime dependencies.
    """
    return {
        "task_queue_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_task_queue"]
        ).create_task_queue(**kwargs),
        "workflow_scheduler_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_workflow_scheduler"]
        ).create_workflow_scheduler(**kwargs),
        "workflow_engine_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["WorkflowEngine"]
        ).WorkflowEngine(**kwargs),
        "workflow_dag_executor_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["WorkflowDAGExecutor"]
        ).WorkflowDAGExecutor(**kwargs),
        "peer_registry_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["create_peer_registry"]
        ).create_peer_registry(**kwargs),
        "peer_discovery_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["PeerDiscoveryManager"]
        ).PeerDiscoveryManager(**kwargs),
        "result_cache_factory": lambda **kwargs: __import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["ResultCache", "MemoryCacheBackend"]
        ).ResultCache(backend=__import__(
            "ipfs_accelerate_py.mcp_server.mcplusplus", fromlist=["MemoryCacheBackend"]
        ).MemoryCacheBackend(), **kwargs),
    }


def _attach_unified_bootstrap(server: Any, config: UnifiedMCPServerConfig) -> None:
    """Attach unified migration components to a legacy MCP server instance.

    This is intentionally non-invasive: no legacy registration paths are replaced.
    The unified components are attached as attributes for incremental integration.
    """
    runtime_router = RuntimeRouter(default_runtime="fastapi")
    manager = HierarchicalToolManager(runtime_router=runtime_router)
    configure_wave_a_loaders(manager)
    preloaded_categories = _preload_configured_categories(manager, config.preload_categories)

    async def tools_list_categories() -> dict[str, Any]:
        return {"categories": manager.list_categories()}

    async def tools_list_tools(category: str) -> dict[str, Any]:
        return {"category": category, "tools": manager.list_tools(category)}

    async def tools_get_schema(category: str, tool_name: str) -> dict[str, Any]:
        return manager.get_tool_schema(category, tool_name)

    async def tools_dispatch(category: str, tool_name: str, parameters: dict[str, Any]) -> Any:
        payload = parameters if isinstance(parameters, dict) else {}
        return await manager.dispatch(category, tool_name, payload)

    async def tools_runtime_metrics() -> dict[str, Any]:
        return {"runtimes": runtime_router.get_metrics()}

    # Attach migration components for callers that want the unified surface.
    setattr(server, "_unified_runtime_router", runtime_router)
    setattr(server, "_unified_tool_manager", manager)
    setattr(server, "_unified_bootstrap_enabled", True)
    setattr(server, "_unified_meta_tools", get_unified_meta_tool_names())
    setattr(server, "_unified_preloaded_categories", preloaded_categories)
    setattr(server, "_unified_services", _build_unified_services())

    # Register compact hierarchical meta-tools if legacy server supports it.
    if hasattr(server, "register_tool") and callable(getattr(server, "register_tool")):
        meta_tool_specs = {
            "tools_list_categories": {
                "function": tools_list_categories,
                "description": "List available unified MCP tool categories.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_list_tools": {
                "function": tools_list_tools,
                "description": "List tools in a unified MCP category.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                    },
                    "required": ["category"],
                },
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_get_schema": {
                "function": tools_get_schema,
                "description": "Get schema for a unified MCP tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "tool_name": {"type": "string"},
                    },
                    "required": ["category", "tool_name"],
                },
                "tags": ["unified", "discovery", "meta"],
            },
            "tools_dispatch": {
                "function": tools_dispatch,
                "description": "Dispatch a unified MCP tool call by category and name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "tool_name": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["category", "tool_name", "parameters"],
                },
                "tags": ["unified", "dispatch", "meta"],
            },
            "tools_runtime_metrics": {
                "function": tools_runtime_metrics,
                "description": "Get unified runtime router metrics (latency/error/timeout).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
                "tags": ["unified", "metrics", "meta"],
            },
        }

        for tool_name in get_unified_meta_tool_names():
            spec = meta_tool_specs[tool_name]
            server.register_tool(
                name=tool_name,
                function=spec["function"],
                description=spec["description"],
                input_schema=spec["input_schema"],
                execution_context="server",
                tags=spec["tags"],
            )


def create_server(*args: Any, **kwargs: Any) -> Any:
    """Create and return an MCP server instance.

    Delegates to the current stable implementation under
    `ipfs_accelerate_py.mcp.server`. This allows callers to migrate imports to
    `ipfs_accelerate_py.mcp_server.server` immediately while preserving behavior.
    """
    from ipfs_accelerate_py.mcp.server import create_mcp_server

    # Prevent recursive bridging if legacy create_mcp_server is configured to
    # route back through this unified package.
    kwargs.setdefault("_skip_unified_bridge", True)

    config = UnifiedMCPServerConfig.from_env(
        allowed_preload_categories=get_unified_wave_a_categories()
    )

    server = create_mcp_server(*args, **kwargs)

    if config.enable_unified_bootstrap:
        try:
            _attach_unified_bootstrap(server, config)
            logger.info("Attached unified MCP server bootstrap components")
        except Exception as exc:
            # Keep old behavior even if bootstrap init fails.
            logger.warning("Unified bootstrap initialization failed: %s", exc)

    return server


def main() -> None:
    """Start the MCP server using existing CLI behavior.

    Delegates to `ipfs_accelerate_py.mcp.server.main` until the unified server
    implementation is ported in place.
    """
    from ipfs_accelerate_py.mcp.server import main as mcp_main

    mcp_main()


if __name__ == "__main__":
    main()
