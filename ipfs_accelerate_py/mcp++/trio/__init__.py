"""
Trio-native MCP server and client implementations

This module provides Trio-first implementations of the Model Context Protocol,
designed to work natively with Trio's structured concurrency model.
"""

from .bridge import run_in_trio, is_trio_context, require_trio, TrioContext

# Server and client will be implemented in subsequent files
try:
    from .server import TrioMCPServer
except ImportError:
    TrioMCPServer = None

try:
    from .client import TrioMCPClient
except ImportError:
    TrioMCPClient = None

__all__ = [
    "run_in_trio",
    "is_trio_context",
    "require_trio",
    "TrioContext",
    "TrioMCPServer",
    "TrioMCPClient",
]
