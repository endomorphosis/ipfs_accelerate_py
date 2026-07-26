"""Agent-supervisor tools for the canonical MCP server."""

from .native_agent_supervisor_tools import (
    AGENT_SUPERVISOR_MCP_DISPATCH_MODE,
    AGENT_SUPERVISOR_MCP_CATEGORY,
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV,
    AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV,
    AgentSupervisorMCPConfigurationError,
    agent_supervisor_control,
    agent_supervisor_discovery_manifest,
    agent_supervisor_v2_discovery_manifest,
    agent_supervisor_service_resolution_count,
    configure_agent_supervisor_control,
    execute_agent_supervisor_operation,
    mcp_control_surface_publication,
    mcp_v2_control_surface_publication,
    register_native_agent_supervisor_tools,
    validate_agent_supervisor_mcp_catalog,
)
from . import native_agent_supervisor_tools as _native

for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = getattr(_native, _tool.__name__)

__all__ = [
    "AGENT_SUPERVISOR_MCP_DISPATCH_MODE",
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "AgentSupervisorMCPConfigurationError",
    "agent_supervisor_control",
    "agent_supervisor_discovery_manifest",
    "agent_supervisor_v2_discovery_manifest",
    "agent_supervisor_service_resolution_count",
    "configure_agent_supervisor_control",
    "execute_agent_supervisor_operation",
    "mcp_control_surface_publication",
    "mcp_v2_control_surface_publication",
    "register_native_agent_supervisor_tools",
    "validate_agent_supervisor_mcp_catalog",
    *[_tool.__name__ for _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.values()],
]
