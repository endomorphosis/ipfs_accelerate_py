# MCP++ Module

This module implements the **MCP++ blueprint** - a Trio-native implementation of the Model Context Protocol (MCP) with peer-to-peer (P2P) capabilities.

## Overview

The `mcp++` module provides:

1. **Trio-first MCP implementation**: Native Trio async/await support without asyncio bridges
2. **P2P networking**: libp2p-based peer-to-peer communication layer
3. **Content-addressed contracts**: CID-native interface contracts and execution envelopes
4. **Event provenance**: Immutable event DAG for audit and replay
5. **Capability delegation**: UCAN-based capability chains and policy evaluation

## Architecture

```
ipfs_accelerate_py/mcp++/
├── __init__.py              # Module initialization and exports
├── README.md                # This file
├── trio/                    # Trio-native MCP server/client
│   ├── __init__.py
│   ├── server.py           # Trio MCP server implementation
│   ├── client.py           # Trio MCP client implementation
│   └── bridge.py           # Helper for running Trio code
├── p2p/                     # P2P networking layer
│   ├── __init__.py
│   ├── taskqueue.py        # P2P task queue (refactored from mcp/tools/)
│   ├── workflow.py         # P2P workflow scheduler
│   ├── peer_registry.py    # Peer discovery and registry
│   └── bootstrap.py        # Bootstrap helpers
├── tools/                   # MCP tools for P2P operations
│   ├── __init__.py
│   ├── taskqueue_tools.py  # MCP tools for P2P taskqueue
│   └── workflow_tools.py   # MCP tools for P2P workflows
└── tests/                   # Test infrastructure
    ├── __init__.py
    ├── test_trio_server.py
    ├── test_p2p_taskqueue.py
    └── test_integration.py
```

## Key Differences from Original MCP

The original MCP implementation in `ipfs_accelerate_py/mcp/` uses:
- FastAPI/Uvicorn (asyncio-based)
- Bridging layer for Trio-only libp2p code
- Thread hops for P2P operations

The `mcp++` module uses:
- Pure Trio event loop
- Direct libp2p integration
- No thread hops or bridges
- Optional Hypercorn for ASGI with Trio support

## Migration from Original MCP

This module refactors and consolidates P2P-related code from:
- `ipfs_accelerate_py/mcp/tools/p2p_taskqueue.py`
- `ipfs_accelerate_py/mcp/tools/p2p_workflow_tools.py`
- `ipfs_accelerate_py/github_cli/p2p_*.py`

The original implementations remain available for backward compatibility.

## Usage

### Starting a Trio-based MCP Server

```python
import trio
from ipfs_accelerate_py.mcplusplus.trio import TrioMCPServer

async def main():
    server = TrioMCPServer(
        name="ipfs-accelerate-p2p",
        version="0.1.0"
    )
    
    # Register P2P tools
    from ipfs_accelerate_py.mcplusplus.tools import register_p2p_tools
    register_p2p_tools(server)
    
    # Run the server
    async with trio.open_nursery() as nursery:
        await server.run(nursery)

if __name__ == "__main__":
    trio.run(main)
```

### Using P2P TaskQueue

```python
from ipfs_accelerate_py.mcplusplus.p2p import P2PTaskQueue

async def submit_task():
    queue = P2PTaskQueue(
        peer_id="QmExample...",
        multiaddr="/ip4/127.0.0.1/tcp/4001"
    )
    
    result = await queue.submit(
        task_type="inference",
        model_name="gpt2",
        payload={"prompt": "Hello, world!"}
    )
    
    return result
```

## Testing Infrastructure

The module includes comprehensive tests from the Mcp-Plus-Plus submodule:

```bash
# Run Trio-specific tests
pytest ipfs_accelerate_py/mcp++/tests/test_trio_server.py

# Run P2P integration tests
pytest ipfs_accelerate_py/mcp++/tests/test_p2p_taskqueue.py

# Run all mcp++ tests
pytest ipfs_accelerate_py/mcp++/tests/
```

## Roadmap

See [docs/MCP_TRIO_ROADMAP.md](../../docs/MCP_TRIO_ROADMAP.md) for detailed implementation roadmap.

Key milestones:
1. ✅ Create module structure and add Mcp-Plus-Plus submodule
2. ⏳ Implement Trio-native MCP server and client
3. ⏳ Refactor P2P code from original MCP module
4. ⏳ Add comprehensive tests
5. ⏳ Document migration path from original MCP

## References

- [MCP++ Specification](./../../mcplusplus/README.md)
- [MCP Trio Roadmap](../../docs/MCP_TRIO_ROADMAP.md)
- [MCP++ Architecture](./../../mcplusplus/docs/ARCHITECTURE.md)
- [Original MCP Implementation](../mcp/README.md)

## License

MIT License - See LICENSE file for details.
