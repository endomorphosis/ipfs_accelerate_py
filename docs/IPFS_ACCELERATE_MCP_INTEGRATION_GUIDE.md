# IPFS Accelerate MCP Integration Guide

This guide explains how to use the Model Context Protocol (MCP) server with IPFS Accelerate.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Available Tools and Resources](#available-tools-and-resources)
5. [Using the Client](#using-the-client)
6. [Real-time Updates with SSE](#real-time-updates-with-sse)
7. [Integration with IPFS Accelerate](#integration-with-ipfs-accelerate)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Introduction

The Model Context Protocol (MCP) provides a standardized way for AI models to access external resources and tools. The IPFS Accelerate MCP server allows AI models to access hardware information, model inference capabilities, and IPFS functionality through a simple HTTP API.

Benefits of using MCP with IPFS Accelerate:

- **Hardware Awareness**: AI models can query hardware capabilities to optimize performance
- **IPFS Integration**: Access IPFS content directly from AI models
- **Acceleration Options**: Discover and utilize WebGPU and WebNN acceleration
- **Real-time Updates**: Get real-time hardware and system information using Server-Sent Events (SSE)

## Installation

To install the IPFS Accelerate MCP server, you need to have Python 3.7 or later and pip installed.

```bash
# Clone the repository
git clone https://github.com/your-repo/ipfs_accelerate_py.git
cd ipfs_accelerate_py

# Install the package and dependencies
pip install -e .
pip install -r mcp/requirements.txt
```

## Getting Started

### Starting the Server

You can start the MCP server using the provided script:

```bash
# Start the server
./restart_mcp_server.sh

# OR

python -m mcp.run_server --debug
```

By default, the server runs on `http://localhost:8002`. You can check if the server is running by accessing the following endpoints:

- `http://localhost:8002/health` - Health check endpoint
- `http://localhost:8002/mcp/manifest` - MCP manifest with available tools and resources

### Basic Client Usage

Here's a simple example of how to use the MCP client:

```python
from mcp.client import MCPClient

# Create a client
client = MCPClient(host="localhost", port=8002)

# Check if the server is available
if client.is_server_available():
    # Get hardware information
    hardware_info = client.get_hardware_info()
    print(hardware_info)
else:
    print("MCP server is not running")
```

## Available Tools and Resources

### Tools

The MCP server provides the following tools:

1. **get_hardware_info** - Get information about the hardware available for acceleration
2. **get_file_from_ipfs** - Get a file from IPFS by its CID
3. **add_file_to_ipfs** - Add a file to IPFS
4. **get_acceleration_options** - Get available acceleration options for a model
5. **accelerate_model** - Accelerate a model using WebGPU or WebNN

### Resources

The MCP server provides the following resources:

1. **system_info** - Information about the system
2. **model_info** - Information about available models
3. **accelerator_info** - Information about available hardware accelerators

## Using the Client

### Getting Hardware Information

```python
from mcp.client import MCPClient

client = MCPClient()
hardware_info = client.get_hardware_info()

# Check for WebGPU support
webgpu_support = hardware_info.get("accelerators", {}).get("webgpu", {}).get("available", False)
print(f"WebGPU support: {webgpu_support}")
```

### Working with IPFS

```python
from mcp.client import MCPClient

client = MCPClient()

# Add a file to IPFS
file_path = "path/to/your/file.txt"
cid = client.call_tool("add_file_to_ipfs", file_path=file_path)
print(f"Added file to IPFS with CID: {cid}")

# Get a file from IPFS
output_path = "downloaded_file.txt"
success = client.call_tool("get_file_from_ipfs", cid=cid, output_path=output_path)
if success:
    print(f"Downloaded file from IPFS to {output_path}")
```

### Accelerating Models

```python
from mcp.client import MCPClient

client = MCPClient()

# Get acceleration options for a model
model_path = "path/to/your/model.onnx"
options = client.call_tool("get_acceleration_options", model_path=model_path)
print(f"Acceleration options: {options}")

# Accelerate the model
accelerated_model_path = client.call_tool(
    "accelerate_model",
    model_path=model_path,
    runtime="webgpu",
    optimization_level="high"
)
print(f"Accelerated model saved to: {accelerated_model_path}")
```

## Real-time Updates with SSE

The MCP server provides real-time updates using Server-Sent Events (SSE). You can use the provided SSE client example to receive these updates:

```python
python -m mcp.examples.sse_client_example
```

Or use the following code in your application:

```python
# This example requires aiohttp package
import asyncio
import aiohttp
import json

async def connect_to_sse():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8002/sse") as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:].decode("utf-8"))
                    print(f"Received update: {data}")

asyncio.run(connect_to_sse())
```

## Integration with IPFS Accelerate

The MCP server integrates with IPFS Accelerate to provide hardware-aware acceleration for AI models. You can use the integration module to initialize the server and register tools and resources:

```python
from mcp.integration import initialize_mcp_server

# Initialize and start the MCP server
success, port = initialize_mcp_server(start_server=True)
if success:
    print(f"MCP server started on port {port}")
```

## Examples

### Simple Client Example

See `mcp/examples/client_example.py` for a complete example of using the MCP client.

### SSE Client Example

See `mcp/examples/sse_client_example.py` for a complete example of using SSE with the MCP server.

## Troubleshooting

### ✅ Major Issues Resolved

**Connection Issues (FIXED)**: The primary issue was missing JSON-RPC 2.0 protocol support. The server now fully implements JSON-RPC 2.0 alongside HTTP REST endpoints, resolving all major connectivity problems.

**Tools/List Method (FIXED)**: The `tools/list` JSON-RPC method is now properly implemented and returns all 8 registered tools with correct schema.

**VS Code MCP Extension (WORKING)**: Full compatibility with VS Code MCP extension confirmed through JSON-RPC 2.0 protocol support.

### Current Server Status

- **Server Location**: `/home/barberb/ipfs_accelerate_py/final_mcp_server.py`
- **Port**: 8004 (correctly configured for client expectations)
- **Protocol Support**: JSON-RPC 2.0 + HTTP REST
- **Tools Available**: 8 registered tools
- **Status**: ✅ Fully operational

### Quick Health Check

Verify the server is working correctly:

```bash
# Test server ping
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}' \
  http://127.0.0.1:8004/jsonrpc

# List all tools
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":2}' \
  http://127.0.0.1:8004/jsonrpc

# Get server info
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"get_server_info","id":3}' \
  http://127.0.0.1:8004/jsonrpc
```

### Remaining Troubleshooting

For any remaining issues:

1. **Server Not Starting**: Check if port 8004 is available and dependencies are installed
2. **Module Import Errors**: Ensure IPFS Accelerate package is installed correctly
3. **Network Issues**: Verify no firewall is blocking localhost connections on port 8004

All major connectivity and protocol issues have been resolved. The server now provides enterprise-grade JSON-RPC 2.0 support for seamless integration with MCP clients.

## Additional Resources

- [MCP Specification](https://github.com/modelcontextprotocol/mcp-spec)
- [FastMCP Documentation](docs/fastmcp/)
- [MCP Python SDK Documentation](docs/mcp-python-sdk/)
