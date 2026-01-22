"""
IPFS Accelerate MCP Module

This module implements a Model Context Protocol server to expose
IPFS Accelerate capabilities through a standardized interface.
"""
from .server import create_mcp_server, get_mcp_server_instance

__all__ = ["create_mcp_server", "get_mcp_server_instance"]
