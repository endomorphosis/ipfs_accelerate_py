# IPFS Accelerate MCP Integration

This module provides a Model Context Protocol (MCP) server integration for the IPFS Accelerate Python framework. The MCP server exposes IPFS Accelerate's hardware-accelerated machine learning inference capabilities to language models and other clients through a standardized interface.

## What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) is a standardized way to provide context and tools to language models. It allows servers to:

- Expose data through **Resources** (context information)
- Provide functionality through **Tools** (actions and computations)
- Define interaction patterns through **Prompts** (reusable templates)

The IPFS Accelerate MCP server makes the hardware acceleration and distributed inference capabilities available to language models like Claude, GPT, and others.

## Features

The MCP integration provides:

- **Hardware Detection Tools**: Query hardware capabilities of the system
- **Inference Tools**: Run ML models with optimized hardware utilization
- **System Resources**: Access information about available hardware and capabilities
- **Model Resources**: Discover and get details about available models
- **Prompt Templates**: Pre-defined interaction patterns for common tasks

## Getting Started

### Installation

This module is included with the IPFS Accelerate package. You may need to install additional dependencies:

```bash
pip install fastmcp mcp
```

### Running the MCP Server Standalone

To run the MCP server as a standalone application:

```bash
# From the project root
python -m mcp.cli --host 127.0.0.1 --port 8000
```

For development mode with auto-reload:

```bash
python -m mcp.cli --dev
```

### Integration with FastAPI Application

To integrate the MCP server with your existing FastAPI application:

```python
from fastapi import FastAPI
from mcp.fastapi_integration import integrate_mcp_with_fastapi

app = FastAPI()
model_server = ...  # Your model server with ipfs_accelerate_py instance

# Integrate MCP server
integrate_mcp_with_fastapi(app, model_server)
```

## Using the MCP Client

The module includes a client example that demonstrates how to connect to and use the MCP server:

```bash
# Get system information
python -m mcp.examples.client_example --action info

# Query hardware detection
python -m mcp.examples.client_example --action hardware

# List available models
python -m mcp.examples.client_example --action models

# Run inference
python -m mcp.examples.client_example --action inference
```

## Tools and Resources

### Tools

The server provides the following tools:

- `detect_hardware`: Detect available hardware accelerators
- `get_optimal_hardware`: Get the optimal hardware for a given model type
- `run_inference`: Run inference using a specified model
- `batch_inference`: Run batch inference on multiple inputs

### Resources

The server provides the following resources:

- `system://info`: System information
- `system://capabilities`: System capabilities
- `models://available`: List of available models
- `models://details/{model_id}`: Details for a specific model

## Development

### Running Tests

To run the test suite:

```bash
python -m unittest discover -s mcp/tests
```

### Adding New Components

To extend the MCP server with new tools, resources, or prompts:

1. Create a new module in the appropriate directory (`tools`, `resources`, or `prompts`)
2. Define your components using the FastMCP decorators
3. Create a registration function to register your components with the MCP server
4. Update the `_register_core_components` function in `server.py` to include your registration function

## License

This module is part of the IPFS Accelerate project and is released under the same license.
