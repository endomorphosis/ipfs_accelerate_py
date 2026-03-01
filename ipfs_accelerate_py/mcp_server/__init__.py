"""Unified MCP server package for IPFS Accelerate.

This package is the canonical destination for MCP server unification work.
During migration, public entry points delegate to `ipfs_accelerate_py.mcp`
for compatibility while internals are progressively ported.
"""

from .server import (
    create_server,
    get_unified_meta_tool_names,
    get_unified_wave_a_categories,
    main,
)
from .runtime_router import RuntimeRouter
from .hierarchical_tool_manager import HierarchicalToolManager
from .registration_adapter import register_legacy_tools_into_manager
from .configs import UnifiedMCPServerConfig, parse_preload_categories
from .tool_metadata import (
    ToolMetadata,
    get_tool_metadata,
    register_tool_metadata,
    tool_metadata,
)
from .tool_registry import ToolRegistry, get_global_registry
from .wave_a_loaders import configure_wave_a_loaders

__all__ = [
    "create_server",
    "get_unified_meta_tool_names",
    "get_unified_wave_a_categories",
    "main",
    "RuntimeRouter",
    "HierarchicalToolManager",
    "register_legacy_tools_into_manager",
    "UnifiedMCPServerConfig",
    "parse_preload_categories",
    "ToolMetadata",
    "tool_metadata",
    "register_tool_metadata",
    "get_tool_metadata",
    "configure_wave_a_loaders",
    "ToolRegistry",
    "get_global_registry",
]
