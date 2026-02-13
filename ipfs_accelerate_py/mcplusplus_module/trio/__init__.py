"""
Trio-native MCP server and client implementations

This module provides Trio-first implementations of the Model Context Protocol,
designed to work natively with Trio's structured concurrency model.
"""

from .bridge import run_in_trio, is_trio_context, require_trio, TrioContext

# Server and client implementations
try:
    from .server import TrioMCPServer, ServerConfig, create_app
except ImportError as e:
    # Log but don't fail if optional dependencies are missing
    import logging
    logging.getLogger(__name__).debug(f"TrioMCPServer not available: {e}")
    TrioMCPServer = None
    ServerConfig = None
    create_app = None

try:
    from .client import TrioMCPClient, ClientConfig, call_tool
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"TrioMCPClient not available: {e}")
    TrioMCPClient = None
    ClientConfig = None
    call_tool = None

__all__ = [
    # Bridge utilities
    "run_in_trio",
    "is_trio_context",
    "require_trio",
    "TrioContext",
    # Server
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    # Client
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
]
