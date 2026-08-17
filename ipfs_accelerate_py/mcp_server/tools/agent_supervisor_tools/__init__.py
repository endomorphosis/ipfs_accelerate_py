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
from .prompt_entrypoints import (
    PROMPT_LIFECYCLE_TOOLS,
    agent_supervisor_doctor,
    agent_supervisor_explain,
    agent_supervisor_follow,
    agent_supervisor_preview,
    agent_supervisor_run,
    agent_supervisor_status,
    agent_supervisor_steer,
    configure_prompt_lifecycle_supervisor,
    prompt_lifecycle_discovery_manifest,
    register_prompt_lifecycle_tools,
)

for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = getattr(_native, _tool.__name__)


def register_all_agent_supervisor_tools(manager: object) -> None:
    """Register control-plane tools and prompt-lifecycle facade tools."""

    register_native_agent_supervisor_tools(manager)
    register_prompt_lifecycle_tools(manager)


__all__ = [
    "AGENT_SUPERVISOR_MCP_DISPATCH_MODE",
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "PROMPT_LIFECYCLE_TOOLS",
    "AgentSupervisorMCPConfigurationError",
    "agent_supervisor_control",
    "agent_supervisor_discovery_manifest",
    "agent_supervisor_doctor",
    "agent_supervisor_explain",
    "agent_supervisor_follow",
    "agent_supervisor_preview",
    "agent_supervisor_run",
    "agent_supervisor_status",
    "agent_supervisor_steer",
    "agent_supervisor_v2_discovery_manifest",
    "agent_supervisor_service_resolution_count",
    "configure_agent_supervisor_control",
    "configure_prompt_lifecycle_supervisor",
    "execute_agent_supervisor_operation",
    "mcp_control_surface_publication",
    "mcp_v2_control_surface_publication",
    "prompt_lifecycle_discovery_manifest",
    "register_all_agent_supervisor_tools",
    "register_native_agent_supervisor_tools",
    "register_prompt_lifecycle_tools",
    "validate_agent_supervisor_mcp_catalog",
    *[_tool.__name__ for _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.values()],
]
