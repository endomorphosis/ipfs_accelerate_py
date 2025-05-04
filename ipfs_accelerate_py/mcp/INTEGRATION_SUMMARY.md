# IPFS Accelerate MCP Integration

This document summarizes the integration of the Model Context Protocol (MCP) with the IPFS Accelerate Python package.

## Overview

The IPFS Accelerate MCP integration provides a standardized way for AI models to interact with IPFS Accelerate's capabilities through a well-defined API. This integration allows AI assistants to use IPFS Accelerate's tools and resources directly via API calls.

## Implementation Details

The implementation consists of the following components:

### Server Architecture

- **Standalone Implementation**: A fallback implementation when FastMCP is not available
- **FastAPI Integration**: Using FastAPI to provide a RESTful API for MCP
- **Tool & Resource Registration**: A system for registering and managing tools and resources

### API Endpoints

The MCP server exposes the following endpoints:

- **Tool Endpoint**: `POST /mcp/tool/{tool_name}`
  - Dynamically dispatches to the appropriate tool function based on the tool name
  - Accepts tool-specific parameters as JSON in the request body

- **Resource Endpoint**: `GET /mcp/resource/{resource_uri}`
  - Dynamically fetches the appropriate resource based on the URI
  - Returns resource data in a consistent format

### Available Tools

The current implementation provides the following tools:

- **Hardware Tools**:
  - `get_hardware_info`: Get information about the system hardware
  - `test_hardware`: Test hardware compatibility with IPFS Accelerate
  - `recommend_hardware`: Get hardware recommendations for IPFS Accelerate

### Available Resources

The current implementation provides the following resources:

- **Model Information**:
  - `ipfs_accelerate/supported_models`: Information about supported models

## Usage

### Starting the Server

To start the MCP server:

```bash
python -m ipfs_accelerate_py.mcp.server [--debug] [--host HOST] [--port PORT]
```

### Using the Client

Example client code:

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

# Accessing a resource
response = requests.get(
    f"{base_url}{mcp_path}/resource/ipfs_accelerate/supported_models"
)
model_info = response.json()
```

## Extension Points

The MCP implementation can be extended in the following ways:

1. **Adding new tools**: Create new tool modules in `ipfs_accelerate_py/mcp/tools/` and register them in `register_all_tools`
2. **Adding new resources**: Create new resource modules in `ipfs_accelerate_py/mcp/resources/` and register them in `register_all_resources`
3. **Adding new prompts**: Update the `_register_prompts` method in `IPFSAccelerateMCPServer`

## Future Improvements

- Full FastMCP integration when available
- Additional IPFS-specific tools and resources
- Performance optimizations for resource-intensive operations
- Extended documentation and examples
