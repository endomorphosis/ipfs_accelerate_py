"""
MCP++ Tools - MCP tools for P2P operations

This module provides MCP tool registrations for P2P taskqueue and workflow operations,
refactored from the original MCP implementation to work natively with Trio.
"""

from .taskqueue_tools import register_p2p_taskqueue_tools

try:
    from .workflow_tools import register_p2p_workflow_tools
except ImportError:
    register_p2p_workflow_tools = None


def register_all_p2p_tools(mcp: Any) -> None:
    """Register all P2P tools with the MCP server.
    
    Args:
        mcp: MCP server instance to register tools with
    """
    register_p2p_taskqueue_tools(mcp)
    if register_p2p_workflow_tools is not None:
        register_p2p_workflow_tools(mcp)


__all__ = [
    "register_p2p_taskqueue_tools",
    "register_p2p_workflow_tools",
    "register_all_p2p_tools",
]
