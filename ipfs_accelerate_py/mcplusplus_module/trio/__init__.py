"""
Trio-native MCP server and client implementations

This module provides Trio-first implementations of the Model Context Protocol,
designed to work natively with Trio's structured concurrency model.
"""

from .bridge import run_in_trio, is_trio_context, require_trio, TrioContext


class _MissingDependencyStub:
    """Compatibility stub for optional symbols unavailable at import time."""

    def __init__(self, symbol_name: str):
        self._symbol_name = str(symbol_name)

    def __repr__(self) -> str:
        return f"<Unavailable {self._symbol_name}>"

    def __bool__(self) -> bool:
        return False

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        raise RuntimeError(f"{self._symbol_name} is unavailable in this environment")

    def __getattr__(self, _name: str):
        raise RuntimeError(f"{self._symbol_name} is unavailable in this environment")


def _missing_dependency_stub(symbol_name: str):
    """Create a consistent compatibility stub for an optional symbol."""
    return _MissingDependencyStub(symbol_name)

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
