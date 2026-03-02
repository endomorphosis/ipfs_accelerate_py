"""
MCP++ (MCP Plus Plus) - Trio-based MCP + P2P implementation

Module name: ipfs_accelerate_py.mcplusplus_module

This module provides a Trio-native implementation of the Model Context Protocol (MCP)
with P2P (peer-to-peer) capabilities, following the MCP++ blueprint from:
https://github.com/endomorphosis/Mcp-Plus-Plus

The MCP++ module implements:
- Content-addressed interface contracts (MCP-IDL)
- Immutable execution envelopes and receipts
- Capability delegation chains (UCAN)
- Temporal deontic policy evaluation
- Event DAG provenance and ordering
- P2P transport bindings (libp2p)

Architecture:
------------
The module is organized into the following submodules:

- trio/: Trio-native MCP server and client implementations
- p2p/: P2P networking layer using libp2p with Trio
- tools/: MCP tools for P2P taskqueue and workflow orchestration
- tests/: Test infrastructure for validating MCP++ implementation

Key differences from existing MCP implementation:
-------------------------------------------------
1. Trio-first: All async operations use Trio nurseries, cancel scopes, etc.
2. No bridging: Direct Trio execution without asyncio-to-Trio bridges
3. Unified P2P: libp2p operations run natively in the Trio event loop
4. Content-addressed: Uses CIDs for interface contracts and execution envelopes

Usage:
------
For running a Trio-backed MCP server with P2P capabilities:

    import trio
    from ipfs_accelerate_py.mcplusplus_module import TrioMCPServer
    
    async def main():
        server = TrioMCPServer(name="my-p2p-server")
        await server.run()
    
    if __name__ == "__main__":
        trio.run(main)

For more information, see:
- docs/MCP_TRIO_ROADMAP.md - Roadmap for Trio-based MCP implementation
- ipfs_accelerate_py/mcplusplus/README.md - MCP++ specification
- ipfs_accelerate_py/mcplusplus/docs/ARCHITECTURE.md - Architecture details
"""

__version__ = "0.1.0"
__author__ = "endomorphosis"

from importlib import import_module
from typing import Callable, Optional


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


def _resolve_storage_wrapper_factory() -> Optional[Callable[..., object]]:
    """Resolve a storage-wrapper factory across historical import locations."""
    module_candidates = (
        "ipfs_accelerate_py.common.storage_wrapper",
        "ipfs_accelerate_py.mcplusplus_module.common.storage_wrapper",
        "test.common.storage_wrapper",
    )
    for module_name in module_candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        have_wrapper = bool(getattr(module, "HAVE_STORAGE_WRAPPER", False))
        factory = getattr(module, "get_storage_wrapper", None)
        if have_wrapper and callable(factory):
            return factory
    return None

# Import key components
try:
    from .trio import (
        TrioMCPServer,
        ServerConfig,
        create_app,
        TrioMCPClient,
        ClientConfig,
        call_tool,
    )
except ImportError:
    TrioMCPServer = _missing_dependency_stub("TrioMCPServer")
    ServerConfig = _missing_dependency_stub("ServerConfig")
    create_app = _missing_dependency_stub("create_app")
    TrioMCPClient = _missing_dependency_stub("TrioMCPClient")
    ClientConfig = _missing_dependency_stub("ClientConfig")
    call_tool = _missing_dependency_stub("call_tool")

try:
    from .p2p import P2PTaskQueue, P2PWorkflowScheduler
except ImportError:
    P2PTaskQueue = _missing_dependency_stub("P2PTaskQueue")
    P2PWorkflowScheduler = _missing_dependency_stub("P2PWorkflowScheduler")

__all__ = [
    "__version__",
    "__author__",
    "_missing_dependency_stub",
    "_resolve_storage_wrapper_factory",
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
    "P2PTaskQueue",
    "P2PWorkflowScheduler",
]
