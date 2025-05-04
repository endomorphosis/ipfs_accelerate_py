# IPFS Accelerate MCP Integration

This package provides a Model Context Protocol (MCP) integration for IPFS Accelerate, allowing language models to interact with IPFS Accelerate capabilities.

## Overview

The Model Context Protocol (MCP) enables language models to access tools and resources through a standardized API. This integration allows language models to:

1. Query system and hardware information
2. Test available hardware accelerators
3. Get hardware recommendations for specific models
4. Access model information and compatibility data

## Installation

To install the IPFS Accelerate MCP integration, follow these steps:

```bash
# Install from requirements file
pip install -r ipfs_accelerate_py/mcp/requirements.txt

# Optional: Install torch for hardware acceleration
pip install torch
```

## Getting Started

### Starting the MCP Server

You can start the MCP server in several ways:

#### Method 1: From Python

```python
from ipfs_accelerate_py.mcp import start_server

# Start server with default settings
start_server()

# Or with custom settings
start_server(
    name="custom-ipfs-server",
    host="0.0.0.0",  # Allow external connections
    port=8080,
    mount_path="/ipfs-mcp",
    debug=True
)
```

#### Method 2: Using the CLI

```bash
python -m ipfs_accelerate_py.mcp.server --host 0.0.0.0 --port 8080 --debug
```

### Using the MCP Client

Once the server is running, you can access it using any MCP client or directly via HTTP requests:

```python
import requests

# Base URL for the MCP server
BASE_URL = "http://localhost:8000/mcp"

# Use a tool
response = requests.post(f"{BASE_URL}/tools/get_hardware_info", json={
    "include_detailed": True
})
hardware_info = response.json()
print(hardware_info)

# Access a resource
response = requests.get(f"{BASE_URL}/resources/ipfs_accelerate/version")
version_info = response.json()
print(version_info)
```

See the `examples/client_example.py` file for a complete example.

## Available Tools and Resources

### Tools

- `get_hardware_info`: Get information about available hardware accelerators
- `test_hardware`: Test available hardware accelerators (CPU, CUDA, etc.)
- `recommend_hardware`: Get hardware recommendations for a specific model

### Resources

- `ipfs_accelerate/version`: Get IPFS Accelerate version information
- `ipfs_accelerate/system_info`: Get system information (CPU, memory, OS, etc.)
- `ipfs_accelerate/config`: Get IPFS Accelerate configuration
- `ipfs_accelerate/supported_models`: Get information about supported models

## Integration with IPFS Accelerate

This MCP server seamlessly integrates with the main IPFS Accelerate package, allowing:

1. Hardware detection and optimization
2. Model compatibility checks
3. Inference acceleration via WebGPU/WebNN
4. IPFS-specific optimizations

## Developer Documentation

The MCP integration is built with a modular design:

- `ipfs_accelerate_py/mcp/server.py`: The main MCP server implementation
- `ipfs_accelerate_py/mcp/tools/`: Tools available to language models
- `ipfs_accelerate_py/mcp/resources/`: Resources available to language models
- `ipfs_accelerate_py/mcp/examples/`: Example code for using the MCP server

### Adding New Tools

To add a new tool, create a new module in the `tools` directory and implement the tool functions. Then, update the `tools/__init__.py` file to register your new tool.

### Adding New Resources

To add a new resource, create a new module in the `resources` directory and implement the resource functions. Then, update the `resources/__init__.py` file to register your new resource.

## License

This project is licensed under the same license as IPFS Accelerate.
