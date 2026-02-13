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
    TrioMCPServer = None
    ServerConfig = None
    create_app = None
    TrioMCPClient = None
    ClientConfig = None
    call_tool = None

try:
    from .p2p import P2PTaskQueue, P2PWorkflowScheduler
except ImportError:
    P2PTaskQueue = None
    P2PWorkflowScheduler = None

__all__ = [
    "__version__",
    "__author__",
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    "TrioMCPClient",
    "ClientConfig",
    "call_tool",
    "P2PTaskQueue",
    "P2PWorkflowScheduler",
]
