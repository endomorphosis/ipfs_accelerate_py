"""
MCP++ Tools - MCP tools for P2P operations

This module provides MCP tool registrations for P2P taskqueue and workflow operations,
refactored from the original MCP implementation to work natively with Trio.
"""

from importlib import import_module
from typing import Any

from .taskqueue_tools import register_p2p_taskqueue_tools
from .workflow_tools import register_p2p_workflow_tools


def _resolve_p2p_registrars():
    """Resolve taskqueue/workflow registrars with canonical-first behavior."""
    try:
        taskqueue_module = import_module(
            "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools"
        )
        workflow_module = import_module(
            "ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools"
        )
        return (
            taskqueue_module.register_p2p_taskqueue_tools,
            workflow_module.register_p2p_workflow_tools,
        )
    except (ImportError, AttributeError):
        # Fallback to already-imported package-level symbols.
        return register_p2p_taskqueue_tools, register_p2p_workflow_tools


def register_all_p2p_tools(mcp: Any) -> None:
    """Register all P2P tools with the MCP server.
    
    This includes:
    - P2P TaskQueue tools (14 tools for task management)
    - P2P Workflow tools (6 tools for workflow scheduling)
    
    Args:
        mcp: MCP server instance to register tools with
    """
    taskqueue_registrar, workflow_registrar = _resolve_p2p_registrars()
    taskqueue_registrar(mcp)
    workflow_registrar(mcp)


__all__ = [
    "_resolve_p2p_registrars",
    "register_p2p_taskqueue_tools",
    "register_p2p_workflow_tools",
    "register_all_p2p_tools",
]
