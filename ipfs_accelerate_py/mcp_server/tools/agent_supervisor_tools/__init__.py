"""Agent-supervisor tools for the canonical MCP server."""

from .native_agent_supervisor_tools import (
    AGENT_SUPERVISOR_MCP_CATEGORY,
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV,
    AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV,
    AgentSupervisorMCPConfigurationError,
    agent_supervisor_control,
    configure_agent_supervisor_control,
    execute_agent_supervisor_operation,
    register_native_agent_supervisor_tools,
)
from . import native_agent_supervisor_tools as _native

for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = getattr(_native, _tool.__name__)

__all__ = [
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "AgentSupervisorMCPConfigurationError",
    "agent_supervisor_control",
    "configure_agent_supervisor_control",
    "execute_agent_supervisor_operation",
    "register_native_agent_supervisor_tools",
    *[_tool.__name__ for _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.values()],
]
