"""Declared PDR-032 MCP surface path for agent-supervisor control tools.

The package directory ``agent_supervisor_tools/`` remains the import root used
by the MCP server (Python prefers packages over sibling modules).  This module
is the taskboard-declared output path for plan create/steer transport exposure:

* Plan operations (``plan_create_preview``, ``plan_create_apply``,
  ``plan_steer_preview``, ``plan_steer_apply``) are members of the closed
  ``Operation`` catalog and are therefore published automatically by the
  package's catalog-driven tool factory — no transport-local policy.
* Workflow aliases (``workflow_preview``, ``workflow_materialize``) keep their
  catalog identity while dispatching through the shared
  ``PlanSupervisorService@1`` facade bound by the default control service.
* Help, import, and discovery remain provider-free: tools are static callables;
  a control service is resolved only at invocation.  Prompt and repository
  text never widen repository/state allowlists.

Importing this module re-exports the package public API for dual-layout and
admission tooling that addresses the declared ``.py`` path explicitly.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

# Load the package (directory) public API without circularly importing this
# sibling module file.  ``import_module`` of the package name resolves to the
# package directory when both layouts exist.
_package = import_module(
    "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools"
)

# Re-export the package surface so explicit file-path loaders and dual-layout
# checkers observe the same symbols as ``from ...agent_supervisor_tools import *``.
AGENT_SUPERVISOR_MCP_CATEGORY = _package.AGENT_SUPERVISOR_MCP_CATEGORY
AGENT_SUPERVISOR_MCP_DISPATCH_MODE = _package.AGENT_SUPERVISOR_MCP_DISPATCH_MODE
AGENT_SUPERVISOR_OPERATION_TOOLS = _package.AGENT_SUPERVISOR_OPERATION_TOOLS
AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV = (
    _package.AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV
)
AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV = _package.AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV
AgentSupervisorMCPConfigurationError = (
    _package.AgentSupervisorMCPConfigurationError
)
agent_supervisor_control = _package.agent_supervisor_control
agent_supervisor_discovery_manifest = _package.agent_supervisor_discovery_manifest
agent_supervisor_v2_discovery_manifest = (
    _package.agent_supervisor_v2_discovery_manifest
)
agent_supervisor_service_resolution_count = (
    _package.agent_supervisor_service_resolution_count
)
configure_agent_supervisor_control = _package.configure_agent_supervisor_control
execute_agent_supervisor_operation = _package.execute_agent_supervisor_operation
mcp_control_surface_publication = _package.mcp_control_surface_publication
mcp_v2_control_surface_publication = _package.mcp_v2_control_surface_publication
register_native_agent_supervisor_tools = (
    _package.register_native_agent_supervisor_tools
)
validate_agent_supervisor_mcp_catalog = (
    _package.validate_agent_supervisor_mcp_catalog
)

# Plan-control operations published from the closed catalog (PDR-032).
# Identity is catalog-authoritative; this mapping is documentation/evidence.
PLAN_CONTROL_MCP_OPERATIONS: frozenset[str] = frozenset(
    {
        "plan_create_preview",
        "plan_create_apply",
        "plan_steer_preview",
        "plan_steer_apply",
    }
)
PLAN_WORKFLOW_ALIAS_MCP_OPERATIONS: frozenset[str] = frozenset(
    {
        "workflow_preview",
        "workflow_materialize",
    }
)


def plan_control_mcp_tool_names() -> MappingProxyType:
    """Return catalog tool names for plan create/steer and workflow aliases."""

    names = {
        operation: f"agent_supervisor_{operation}"
        for operation in sorted(
            PLAN_CONTROL_MCP_OPERATIONS | PLAN_WORKFLOW_ALIAS_MCP_OPERATIONS
        )
    }
    return MappingProxyType(names)


def plan_control_operations_are_published() -> bool:
    """True when every plan control operation has a catalog MCP tool callable."""

    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
        PLAN_CONTROL_OPERATIONS,
        Operation,
    )

    tools = AGENT_SUPERVISOR_OPERATION_TOOLS
    for operation in PLAN_CONTROL_OPERATIONS:
        if operation not in tools or not callable(tools[operation]):
            return False
    for name in PLAN_WORKFLOW_ALIAS_MCP_OPERATIONS:
        operation = Operation(name)
        if operation not in tools or not callable(tools[operation]):
            return False
    return True


# Mirror per-operation tool callables into this module's globals for parity
# with the package ``__init__`` re-export pattern.
for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = _tool


__all__ = [
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_MCP_DISPATCH_MODE",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "AgentSupervisorMCPConfigurationError",
    "PLAN_CONTROL_MCP_OPERATIONS",
    "PLAN_WORKFLOW_ALIAS_MCP_OPERATIONS",
    "agent_supervisor_control",
    "agent_supervisor_discovery_manifest",
    "agent_supervisor_v2_discovery_manifest",
    "agent_supervisor_service_resolution_count",
    "configure_agent_supervisor_control",
    "execute_agent_supervisor_operation",
    "mcp_control_surface_publication",
    "mcp_v2_control_surface_publication",
    "plan_control_mcp_tool_names",
    "plan_control_operations_are_published",
    "register_native_agent_supervisor_tools",
    "validate_agent_supervisor_mcp_catalog",
    *[_tool.__name__ for _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.values()],
]
