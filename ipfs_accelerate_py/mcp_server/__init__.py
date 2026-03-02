"""Unified MCP server package for IPFS Accelerate.

This package is the canonical destination for MCP server unification work.
During migration, public entry points delegate to `ipfs_accelerate_py.mcp`
for compatibility while internals are progressively ported.
"""

from .server import (
    create_server,
    get_unified_meta_tool_names,
    get_unified_supported_profiles,
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
from .dispatch_pipeline import (
    coerce_dispatch_bool,
    coerce_dispatch_dict,
    coerce_dispatch_list,
    compute_dispatch_intent_cid,
    normalize_dispatch_parameters,
)
from .server_context import UnifiedServerContext
from .fastapi_config import UnifiedFastAPIConfig
from .fastapi_service import create_fastapi_app, run_fastapi_server
from .standalone_server import run_server as run_standalone_server
from .mcp_p2p_transport import (
    PROTOCOL_MCP_P2P_V1,
    get_mcp_p2p_stats,
    handle_mcp_p2p_stream,
    read_u32_framed_json,
    reset_mcp_p2p_stats,
    write_u32_framed_json,
)
from .trio_adapter import TRIO_AVAILABLE, TrioMCPServerAdapter, TrioServerConfig
from .register_p2p_tools import (
    P2P_TOOL_MODULES,
    discover_p2p_tool_modules,
    get_p2p_tool_summary,
    register_p2p_category_loaders,
)
from .monitoring import (
    EnhancedMetricsCollector,
    HealthCheckResult,
    MetricData,
    P2PMetricsCollector,
    get_metrics_collector,
    get_p2p_metrics_collector,
)
from .otel_tracing import MCPTracer, configure_tracing
from .prometheus_exporter import PrometheusExporter

__all__ = [
    "create_server",
    "get_unified_meta_tool_names",
    "get_unified_supported_profiles",
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
    "normalize_dispatch_parameters",
    "coerce_dispatch_bool",
    "coerce_dispatch_list",
    "coerce_dispatch_dict",
    "compute_dispatch_intent_cid",
    "UnifiedServerContext",
    "UnifiedFastAPIConfig",
    "create_fastapi_app",
    "run_fastapi_server",
    "run_standalone_server",
    "PROTOCOL_MCP_P2P_V1",
    "get_mcp_p2p_stats",
    "reset_mcp_p2p_stats",
    "read_u32_framed_json",
    "write_u32_framed_json",
    "handle_mcp_p2p_stream",
    "TRIO_AVAILABLE",
    "TrioServerConfig",
    "TrioMCPServerAdapter",
    "P2P_TOOL_MODULES",
    "discover_p2p_tool_modules",
    "register_p2p_category_loaders",
    "get_p2p_tool_summary",
    "MetricData",
    "HealthCheckResult",
    "EnhancedMetricsCollector",
    "P2PMetricsCollector",
    "get_metrics_collector",
    "get_p2p_metrics_collector",
    "MCPTracer",
    "configure_tracing",
    "PrometheusExporter",
]
