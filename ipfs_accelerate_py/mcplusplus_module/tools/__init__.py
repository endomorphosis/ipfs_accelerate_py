"""
MCP++ Tools - MCP tools for P2P operations

This module provides MCP tool registrations for P2P taskqueue and workflow operations,
refactored from the original MCP implementation to work natively with Trio.
"""

from typing import Any

from .taskqueue_tools import register_p2p_taskqueue_tools
from .workflow_tools import register_p2p_workflow_tools


def register_all_p2p_tools(mcp: Any) -> None:
    """Register all P2P tools with the MCP server.
    
    This includes:
    - P2P TaskQueue tools (14 tools for task management)
    - P2P Workflow tools (6 tools for workflow scheduling)
    
    Args:
        mcp: MCP server instance to register tools with
    """
    register_p2p_taskqueue_tools(mcp)
    register_p2p_workflow_tools(mcp)


__all__ = [
    "register_p2p_taskqueue_tools",
    "register_p2p_workflow_tools",
    "register_all_p2p_tools",
]
