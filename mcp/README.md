# IPFS Accelerate MCP Integration

This module provides [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/mcp) integration for IPFS Accelerate, allowing Language Models (LLMs) to interact directly with IPFS operations and leverage hardware acceleration capabilities.

## Overview

The MCP integration allows LLMs to:

1. Perform IPFS operations (add, pin, retrieve files, etc.)
2. Accelerate AI models stored on IPFS using available hardware
3. Access metadata and content hash verification
4. Utilize P2P networking capabilities

## Installation

To install the MCP integration with all dependencies:

```bash
# Install from the requirements file
pip install -r mcp/requirements-mcp.txt

# OR install the package with the MCP extra
pip install ipfs-accelerate-py[mcp]
```

## Usage

### Running the MCP Server

The MCP server can be run directly using the provided script:

```bash
# Run with stdio transport (default for direct LLM communication)
python run_mcp.py

# Run with WebSocket transport on port 8080
python run_mcp.py -t ws --port 8080

# Run with SSE transport on port 8000
python run_mcp.py -t sse --port 8000
```

### Core Components

The MCP integration consists of the following components:

- **Server**: `mcp/server.py` - Core MCP server implementation
- **Tools**: `mcp/tools/` - IPFS operation tool modules
- **Types**: `mcp/types.py` - Type definitions for MCP objects
- **Mock Implementation**: `mcp/mock_mcp.py` - Fallback implementation when dependencies aren't available

### Using in LLM Applications

To use the IPFS Accelerate MCP server in an LLM application:

```python
from ipfs_accelerate_py.mcp import create_ipfs_mcp_server

# Create and configure the server
mcp_server = create_ipfs_mcp_server(name="IPFS Accelerate")

# Register additional custom tools if needed
@mcp_server.tool()
async def custom_ipfs_tool(ctx):
    # Implementation here
    return {"result": "success"}

# Run the server
mcp_server.run(transport="stdio")  # or "sse", "ws"
```

For a complete example, see `examples/mcp_integration_example.py`.

## Available Tools

The MCP server provides the following tools:

### IPFS File Operations

- `ipfs_add`: Add a file to IPFS
- `ipfs_files_ls`: List files in IPFS MFS
- `ipfs_files_read`: Read a file from IPFS MFS
- `ipfs_files_write`: Write to a file in IPFS MFS
- `ipfs_files_cp`: Copy files in IPFS MFS
- `ipfs_files_rm`: Remove files from IPFS MFS

### IPFS Network Operations

- `ipfs_pubsub_publish`: Publish a message to a topic
- `ipfs_pubsub_subscribe`: Subscribe to a topic
- `ipfs_dht_findpeer`: Find a peer in the DHT
- `ipfs_dht_findprovs`: Find providers for a CID

### IPFS Acceleration Operations

- `ipfs_accelerate_model`: Accelerate an AI model stored on IPFS
- `ipfs_get_hardware_info`: Get available hardware for acceleration
- `ipfs_benchmark_model`: Benchmark a model's performance

### Status and Information

- `ipfs_status`: Get IPFS node status information
- `ipfs_version`: Get IPFS version information

## Fallback Behavior

If the required dependencies (like `fastmcp` or `ipfs-kit-py`) are not available, the integration will use mock implementations that provide simulated responses. This allows development and testing without all dependencies, but with limited functionality.

To use the full functionality, install all dependencies from the `requirements-mcp.txt` file.

## Adding New Tools

To add new tools to the MCP server, create a module in the `mcp/tools/` directory and register the tools in that module. The tools will be automatically discovered and registered with the server.

Example:

```python
# In mcp/tools/my_tools.py
from mcp.server import Context

async def register_tools(mcp_server):
    @mcp_server.tool()
    async def my_custom_tool(arg1: str, arg2: int, ctx: Context):
        """Tool description for documentation."""
        # Implementation here
        return {"result": arg1, "count": arg2}
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Install dependencies: `pip install -r mcp/requirements-mcp.txt`
   - If a specific package is not available, the mock implementation will be used

2. **Connection Issues**:
   - Ensure IPFS daemon is running: `ipfs daemon`
   - Check API endpoint configuration

3. **Transport Errors**:
   - For `sse` or `ws` transport, ensure the port is available
   - For `stdio` transport, ensure standard I/O is not redirected

### Logging

To enable debug logging:

```bash
python run_mcp.py --debug
```

## Testing

Run the MCP integration tests with:

```bash
python -m unittest test.test_mcp_integration
```
