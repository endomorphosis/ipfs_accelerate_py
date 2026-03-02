"""
Trio-native MCP server and client implementations

This module provides Trio-first implementations of the Model Context Protocol,
designed to work natively with Trio's structured concurrency model.
"""

from .. import _missing_dependency_stub

try:
    from .bridge import run_in_trio, is_trio_context, require_trio, TrioContext
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Trio bridge utilities not available: {e}")
    run_in_trio = _missing_dependency_stub("run_in_trio")
    is_trio_context = _missing_dependency_stub("is_trio_context")
    require_trio = _missing_dependency_stub("require_trio")
    TrioContext = _missing_dependency_stub("TrioContext")

# Server and client implementations
try:
    from .server import TrioMCPServer, ServerConfig, create_app
except ImportError as e:
    # Log but don't fail if optional dependencies are missing
    import logging
    logging.getLogger(__name__).debug(f"TrioMCPServer not available: {e}")
    TrioMCPServer = _missing_dependency_stub("TrioMCPServer")
    ServerConfig = _missing_dependency_stub("ServerConfig")
    create_app = _missing_dependency_stub("create_app")

try:
    from .client import TrioMCPClient, ClientConfig, call_tool
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"TrioMCPClient not available: {e}")
    TrioMCPClient = _missing_dependency_stub("TrioMCPClient")
    ClientConfig = _missing_dependency_stub("ClientConfig")
    call_tool = _missing_dependency_stub("call_tool")

__all__ = [
    # Bridge utilities
    "run_in_trio",
    "is_trio_context",
    "require_trio",
    "TrioContext",
    "_missing_dependency_stub",
    # Server
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    # Client
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
]
