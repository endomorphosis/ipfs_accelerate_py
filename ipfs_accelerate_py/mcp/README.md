# IPFS Accelerate MCP

Model Context Protocol (MCP) integration for IPFS Accelerate.

## Overview

This module implements the Model Context Protocol for IPFS Accelerate, allowing AI assistants to interact with IPFS Accelerate's functionality through a standardized API. It provides:

* A FastAPI-based MCP server that exposes IPFS Accelerate tools and resources
* Both a standalone implementation and FastMCP integration options
* Tools for hardware detection, model inferencing, and IPFS operations
* Resources for model information and system status

## Quick Start

### Server Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Starting the Server

Start the MCP server:

```bash
python -m ipfs_accelerate_py.mcp.server --debug
```

By default, the server will run at http://localhost:8000/mcp.

### External Access

To expose the MCP API beyond localhost, bind to all interfaces and optionally configure CORS for browser clients:

```bash
python -m ipfs_accelerate_py.mcp.server --host 0.0.0.0 --port 8000 --debug
```

For browser-based apps calling this API from a different origin, set allowed origins (comma-separated) via `MCP_CORS_ORIGINS`:

```bash
export MCP_CORS_ORIGINS="https://yourapp.example.com,https://another.example"
python -m ipfs_accelerate_py.mcp.server --host 0.0.0.0 --port 8000
```

If using a cloud VM, also open the port in your firewall/security group.

### Using the Client

Connect to the server using Python requests:

```python
import requests

# Server URL
base_url = "http://localhost:8000"
mcp_path = "/mcp"

# Using a tool
response = requests.post(
    f"{base_url}{mcp_path}/tool/get_hardware_info",
    json={}
)
hardware_info = response.json()
print(f"CPU: {hardware_info.get('cpu', {}).get('model', 'Unknown')}")

# Accessing a resource
response = requests.get(
    f"{base_url}{mcp_path}/resource/ipfs_accelerate/supported_models"
)
model_info = response.json()
print(f"Total models: {model_info.get('count', 0)}")
```

## Server Options

The server supports the following command-line options:

* `--name`: Name of the server (default: "ipfs-accelerate")
* `--host`: Host to bind the server to (default: "localhost")
* `--port`: Port to bind the server to (default: 8000)
* `--mount-path`: Path to mount the server at (default: "/mcp")
* `--debug`: Enable debug logging

## Documentation

For more detailed information, see:

* [Integration Summary](./INTEGRATION_SUMMARY.md): Overview of the MCP integration
* [Getting Started](./GETTING_STARTED.md): Detailed setup instructions
* [API Documentation](http://localhost:8000/docs): Interactive API documentation (when server is running)

## Examples

See the [examples directory](./examples/) for example client code and usage patterns.
