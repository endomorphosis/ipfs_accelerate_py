# IPFS Accelerate Model Context Protocol (MCP) Integration

This directory contains all the components necessary to use IPFS Accelerate with AI assistants via the Model Context Protocol (MCP).

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a standardized way for AI models to interact with tools and resources. It allows AI assistants like Claude to access your IPFS Accelerate functionality in a secure, controlled manner.

## Quick Start

### Option 1: Full Installation for Claude

To install the IPFS Accelerate MCP server for use with Claude:

```bash
# From the project root
./install_mcp_server.py
```

This script:
1. Installs all required dependencies
2. Registers the MCP server with Claude
3. Sets up the server to run with SSE transport

After installation, open Claude and enable the IPFS Accelerate server in settings.

### Option 2: Manual Startup

To start the server manually:

```bash
# From the project root
./install_and_run_mcp.sh
```

## Features

The IPFS Accelerate MCP server provides:

### Tools

- **Hardware Detection**: Analyze available hardware for optimal model performance
- **Model Inference**: Run inference on models with accelerated performance
- **System Status**: Check the status of the IPFS Accelerate system

### Resources

- **Model Information**: Access details about supported models
- **Configuration**: View and access configuration settings

## Documentation

For more detailed information, see:

- [IPFS Accelerate MCP Integration Guide](../../IPFS_ACCELERATE_MCP_INTEGRATION_GUIDE.md): Comprehensive guide to using MCP with IPFS Accelerate
- [Examples](./examples/): Code examples showing how to use the MCP server programmatically

## Development

### Structure

- `server.py`: Core MCP server implementation
- `tools/`: Executable functions exposed via MCP
- `resources/`: Data sources exposed via MCP
- `prompts/`: Reusable prompt templates
- `examples/`: Code samples showing MCP usage
- `tests/`: Test suite for MCP integration

### Requirements

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Client Examples

To test the MCP server programmatically:

```python
# See detailed example in examples/sse_client_example.py
from ipfs_accelerate_py.mcp.examples.sse_client_example import IPFSAccelerateMCPClient

client = IPFSAccelerateMCPClient(base_url="http://localhost:8000")
hardware_info = client.call_tool("get_hardware_info")
```

## Troubleshooting

### Connection Issues (RESOLVED)

**Previous Issue**: The server was missing JSON-RPC 2.0 protocol support, causing VS Code MCP extension and other JSON-RPC clients to fail when calling `tools/list`.

**Resolution**: The server now fully supports JSON-RPC 2.0 protocol alongside HTTP REST endpoints. All major connection issues have been resolved.

### Current Status ✅

- ✅ JSON-RPC 2.0 protocol fully implemented
- ✅ `tools/list` method working correctly
- ✅ VS Code MCP extension compatibility confirmed
- ✅ All 8 tools properly registered and accessible
- ✅ Server runs on port 8004 as expected by clients

### Quick Diagnostics

If you encounter any issues:

1. **Verify server is running**:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"ping","id":1}' \
     http://127.0.0.1:8004/jsonrpc
   ```
   Expected: `{"jsonrpc":"2.0","result":"pong","id":1}`

2. **Check tool availability**:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":2}' \
     http://127.0.0.1:8004/jsonrpc
   ```
   Expected: Array of 8 tools

3. **Verify dependencies are installed**
4. **Check logs for detailed error messages**

For more help, see the [Integration Guide](../../IPFS_ACCELERATE_MCP_INTEGRATION_GUIDE.md#troubleshooting).
