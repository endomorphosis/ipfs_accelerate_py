"""
Type definitions for the IPFS Accelerate MCP server.

This module provides type definitions used across the MCP server implementation
to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class IPFSAccelerateContext:
    """Context object for the IPFS Accelerate MCP server.
    
    This contains initialized resources and connections used by MCP tools and resources.
    """
    # Add fields for IPFS connections, acceleration context, etc.
    config: Dict[str, Any]
    ipfs_client: Any = None  # Will be initialized in lifespan
