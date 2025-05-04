"""
IPFS Accelerate MCP (Model Context Protocol) Integration

This package provides integration between IPFS Accelerate and LLMs through MCP,
allowing LLMs to perform IPFS operations and leverage hardware acceleration.
"""

__version__ = "0.1.0"

# Re-export key components to make them accessible at package level
try:
    from .server import create_ipfs_mcp_server, default_server, IPFSAccelerateContext
except ImportError:
    # This allows the package to be imported even if all components aren't available
    # Individual imports will still fail if dependencies aren't met
    pass

# Define public API
__all__ = [
    "create_ipfs_mcp_server",
    "default_server",
    "IPFSAccelerateContext",
]
