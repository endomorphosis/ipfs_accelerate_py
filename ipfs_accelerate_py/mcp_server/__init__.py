"""Unified MCP server package for IPFS Accelerate.

This package is the canonical destination for MCP server unification work.
During migration, public entry points delegate to `ipfs_accelerate_py.mcp`
for compatibility while internals are progressively ported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "create_server": (".server", "create_server"),
    "get_unified_meta_tool_names": (".server", "get_unified_meta_tool_names"),
    "get_unified_supported_profiles": (".server", "get_unified_supported_profiles"),
    "get_unified_wave_a_categories": (".server", "get_unified_wave_a_categories"),
    "main": (".server", "main"),
    "RuntimeRouter": (".runtime_router", "RuntimeRouter"),
    "HierarchicalToolManager": (".hierarchical_tool_manager", "HierarchicalToolManager"),
    "register_legacy_tools_into_manager": (".registration_adapter", "register_legacy_tools_into_manager"),
    "UnifiedMCPServerConfig": (".configs", "UnifiedMCPServerConfig"),
    "parse_preload_categories": (".configs", "parse_preload_categories"),
    "ToolMetadata": (".tool_metadata", "ToolMetadata"),
    "get_tool_metadata": (".tool_metadata", "get_tool_metadata"),
    "register_tool_metadata": (".tool_metadata", "register_tool_metadata"),
    "tool_metadata": (".tool_metadata", "tool_metadata"),
    "ToolRegistry": (".tool_registry", "ToolRegistry"),
    "get_global_registry": (".tool_registry", "get_global_registry"),
    "configure_wave_a_loaders": (".wave_a_loaders", "configure_wave_a_loaders"),
    "coerce_dispatch_bool": (".dispatch_pipeline", "coerce_dispatch_bool"),
    "coerce_dispatch_dict": (".dispatch_pipeline", "coerce_dispatch_dict"),
    "coerce_dispatch_list": (".dispatch_pipeline", "coerce_dispatch_list"),
    "compute_dispatch_intent_cid": (".dispatch_pipeline", "compute_dispatch_intent_cid"),
    "normalize_dispatch_parameters": (".dispatch_pipeline", "normalize_dispatch_parameters"),
    "UnifiedServerContext": (".server_context", "UnifiedServerContext"),
    "UnifiedFastAPIConfig": (".fastapi_config", "UnifiedFastAPIConfig"),
    "create_fastapi_app": (".fastapi_service", "create_fastapi_app"),
    "run_fastapi_server": (".fastapi_service", "run_fastapi_server"),
    "run_standalone_server": (".standalone_server", "run_server"),
    "PROTOCOL_MCP_P2P_V1": (".mcp_p2p_transport", "PROTOCOL_MCP_P2P_V1"),
    "get_mcp_p2p_stats": (".mcp_p2p_transport", "get_mcp_p2p_stats"),
    "handle_mcp_p2p_stream": (".mcp_p2p_transport", "handle_mcp_p2p_stream"),
    "read_u32_framed_json": (".mcp_p2p_transport", "read_u32_framed_json"),
    "reset_mcp_p2p_stats": (".mcp_p2p_transport", "reset_mcp_p2p_stats"),
    "write_u32_framed_json": (".mcp_p2p_transport", "write_u32_framed_json"),
    "TRIO_AVAILABLE": (".trio_adapter", "TRIO_AVAILABLE"),
    "TrioMCPServerAdapter": (".trio_adapter", "TrioMCPServerAdapter"),
    "TrioServerConfig": (".trio_adapter", "TrioServerConfig"),
    "P2P_TOOL_MODULES": (".register_p2p_tools", "P2P_TOOL_MODULES"),
    "discover_p2p_tool_modules": (".register_p2p_tools", "discover_p2p_tool_modules"),
    "get_p2p_tool_summary": (".register_p2p_tools", "get_p2p_tool_summary"),
    "register_p2p_category_loaders": (".register_p2p_tools", "register_p2p_category_loaders"),
    "EnhancedMetricsCollector": (".monitoring", "EnhancedMetricsCollector"),
    "HealthCheckResult": (".monitoring", "HealthCheckResult"),
    "MetricData": (".monitoring", "MetricData"),
    "P2PMetricsCollector": (".monitoring", "P2PMetricsCollector"),
    "get_metrics_collector": (".monitoring", "get_metrics_collector"),
    "get_p2p_metrics_collector": (".monitoring", "get_p2p_metrics_collector"),
    "MCPTracer": (".otel_tracing", "MCPTracer"),
    "configure_tracing": (".otel_tracing", "configure_tracing"),
    "PrometheusExporter": (".prometheus_exporter", "PrometheusExporter"),
    "Capability": (".ucan_delegation", "Capability"),
    "Delegation": (".ucan_delegation", "Delegation"),
    "DelegationEvaluator": (".ucan_delegation", "DelegationEvaluator"),
    "InvocationContext": (".ucan_delegation", "InvocationContext"),
    "add_delegation": (".ucan_delegation", "add_delegation"),
    "get_delegation": (".ucan_delegation", "get_delegation"),
    "get_delegation_evaluator": (".ucan_delegation", "get_delegation_evaluator"),
    "PolicyClause": (".temporal_policy", "PolicyClause"),
    "PolicyEvaluator": (".temporal_policy", "PolicyEvaluator"),
    "PolicyObject": (".temporal_policy", "PolicyObject"),
    "make_simple_permission_policy": (".temporal_policy", "make_simple_permission_policy"),
    "TemporalDeonticMCPServer": (".temporal_deontic_mcp_server", "TemporalDeonticMCPServer"),
    "artifact_cid": (".cid_artifacts", "artifact_cid"),
    "InterfaceRepository": (".interface_descriptor", "InterfaceRepository"),
    "get_interface_repository": (".interface_descriptor", "get_interface_repository"),
    "MCPClientProtocol": (".mcp_interfaces", "MCPClientProtocol"),
    "MCPServerProtocol": (".mcp_interfaces", "MCPServerProtocol"),
    "ToolManagerProtocol": (".mcp_interfaces", "ToolManagerProtocol"),
    "EventDAG": (".event_dag", "EventDAG"),
    "P2PServiceManager": (".p2p_service_manager", "P2PServiceManager"),
    "P2PServiceState": (".p2p_service_manager", "P2PServiceState"),
    "P2PMCPRegistryAdapter": (".p2p_mcp_registry_adapter", "P2PMCPRegistryAdapter"),
    "RUNTIME_FASTAPI": (".p2p_mcp_registry_adapter", "RUNTIME_FASTAPI"),
    "RUNTIME_TRIO": (".p2p_mcp_registry_adapter", "RUNTIME_TRIO"),
    "RUNTIME_UNKNOWN": (".p2p_mcp_registry_adapter", "RUNTIME_UNKNOWN"),
    "SimpleCallResult": (".simple_server", "SimpleCallResult"),
    "SimpleIPFSDatasetsMCPServer": (".simple_server", "SimpleIPFSDatasetsMCPServer"),
    "start_simple_server": (".simple_server", "start_simple_server"),
    "IPFSDatasetsMCPClient": (".client", "IPFSDatasetsMCPClient"),
    "DIDKeyManager": (".did_key_manager", "DIDKeyManager"),
    "get_did_key_manager": (".did_key_manager", "get_did_key_manager"),
    "EnhancedParameterValidator": (".validators", "EnhancedParameterValidator"),
    "validate_dispatch_inputs": (".validators", "validate_dispatch_inputs"),
    "validator": (".validators", "validator"),
    "configure_root_logging": (".logger", "configure_root_logging"),
    "get_logger": (".logger", "get_logger"),
    "logger": (".logger", "logger"),
    "mcp_logger": (".logger", "mcp_logger"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
