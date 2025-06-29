# IPFS Accelerate MCP Integration

This package integrates the IPFS Accelerate library with the Model Context Protocol (MCP), allowing AI models to access hardware information and model inference capabilities through a standardized API.

## Overview

The Model Context Protocol (MCP) enables AI assistants to access external tools and resources through a well-defined interface. This integration exposes IPFS Accelerate functionality via the MCP protocol, providing:

- Hardware detection capabilities
- System information access
- Potential model inference acceleration in future versions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/ipfs_accelerate_py.git
cd ipfs_accelerate_py
```

2. Install the package and its dependencies:

```bash
pip install -r mcp/requirements.txt
```

## Usage

### Starting the MCP Server

You can start the MCP server using the provided `run_ipfs_mcp.py` script:

```bash
python run_ipfs_mcp.py
```

Or using the module directly:

```bash
python -m mcp.run_server
```

The server provides the following command-line options:

- `--port PORT`: Port to bind to (default: 8004)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--debug`: Run in debug mode
- `--find-port`: Automatically find an available port if the default is in use

### Server Architecture

The MCP server supports both HTTP REST API endpoints and JSON-RPC 2.0 protocol:

**JSON-RPC 2.0 Endpoints (Primary)**:
- `POST /jsonrpc` - Main JSON-RPC endpoint supporting:
  - `ping` - Health check (returns "pong")
  - `tools/list` - Get all available tools
  - `get_tools` - Alias for tools/list
  - `get_server_info` - Server metadata
  - `use_tool` - Execute specific tools

**HTTP REST Endpoints (Legacy)**:
- `GET /tools` - List available tools
- `GET /health` - Health check
- `POST /mcp/tool/{tool_name}` - Execute specific tools

### Using the MCP Client

#### From Python Code

```python
from mcp import MCPClient

# Create a client
client = MCPClient(host="localhost", port=8004)

# Get hardware information
hardware_info = client.get_hardware_info()
print(hardware_info)

# Access a resource
system_info = client.access_resource("system_info")
print(system_info)
```

#### Using JSON-RPC 2.0 Protocol

The server now supports full JSON-RPC 2.0 protocol for compatibility with VS Code MCP extension and other JSON-RPC clients:

```python
import requests

# Test server connectivity
response = requests.post("http://localhost:8004/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "ping",
    "id": 1
})
print(response.json())  # {"jsonrpc": "2.0", "result": "pong", "id": 1}

# Get available tools
response = requests.post("http://localhost:8004/jsonrpc", json={
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 2
})
tools = response.json()["result"]
print(f"Available tools: {len(tools)}")
```

#### Using the Command-line Interface

The `run_ipfs_mcp.py` script can also be used as a client:

```bash
# Get hardware information
python run_ipfs_mcp.py --info

# List available tools
python run_ipfs_mcp.py --list-tools

# List available resources
python run_ipfs_mcp.py --list-resources

# Call a specific tool
python run_ipfs_mcp.py --tool get_hardware_info

# Access a specific resource
python run_ipfs_mcp.py --resource system_info
```

#### Testing JSON-RPC Connectivity

Verify the server is working with JSON-RPC protocol:

```bash
# Test ping endpoint
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"ping","id":1}' \
  http://localhost:8004/jsonrpc

# List available tools
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":2}' \
  http://localhost:8004/jsonrpc
```

## Directory Structure

```
mcp/
├── __init__.py              # Package initialization
├── client.py                # MCP client implementation
├── server.py                # MCP server implementation
├── run_server.py            # Server runner script
├── requirements.txt         # Package dependencies
├── examples/                # Usage examples
│   ├── __init__.py
│   └── client_example.py    # Client example
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_mcp_server.py   # Server tests
├── tools/                   # MCP tools
│   ├── __init__.py
│   └── hardware.py          # Hardware detection tool
└── resources/               # MCP resources
    └── __init__.py
```

## Available Tools

The MCP server currently provides **8 registered tools**:

| Tool Name | Description |
|-----------|-------------|
| get_hardware_info | Returns detailed information about available hardware accelerators |
| ipfs_add_file | Add a file to IPFS and return its hash |
| ipfs_cat | Retrieve content from IPFS by hash |
| ipfs_get | Download content from IPFS to local filesystem |
| process_data | Process data using available accelerators |
| init_endpoints | Initialize IPFS endpoints |
| vfs_list | List files in the virtual filesystem |
| create_storage | Create new storage configurations |

All tools are accessible via both JSON-RPC 2.0 (`tools/list`, `use_tool`) and HTTP REST endpoints (`/tools`, `/mcp/tool/{tool_name}`).

## Available Resources

| Resource Name | Description |
|--------------|-------------|
| system_info | Information about the host system |
| accelerator_info | Information about available accelerators |

## Development

### Adding a New Tool

You can add new tools to the MCP server by registering them using the `register_tool` function:

```python
from mcp.server import register_tool

def my_new_tool(parameter1: str, parameter2: int = 0) -> dict:
    """
    My new tool description
    
    Args:
        parameter1: Description of parameter1
        parameter2: Description of parameter2
    
    Returns:
        Dict with results
    """
    # Tool implementation
    return {"result": parameter1, "count": parameter2}

register_tool(
    name="my_new_tool",
    description="Description of my new tool",
    function=my_new_tool,
)
```

### Adding a New Resource

You can add new resources to the MCP server by registering them using the `register_resource` function:

```python
from mcp.server import register_resource

def get_my_resource() -> dict:
    """Get my resource data"""
    return {"key": "value", "status": "ok"}

register_resource(
    name="my_resource",
    description="Description of my resource",
    getter=get_my_resource,
)
```

## Testing

To run the tests:

```bash
python -m unittest discover mcp/tests
```

## Documentation

For more detailed documentation, refer to:

- [IPFS Accelerate MCP Integration Guide](../IPFS_ACCELERATE_MCP_INTEGRATION_GUIDE.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
