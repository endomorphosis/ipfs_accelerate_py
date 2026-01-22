# IPFS Accelerate with Model Context Protocol

## Overview

The Model Context Protocol (MCP) integration enables language models and other AI systems to interact directly with IPFS Accelerate's hardware-accelerated inference capabilities through a standardized API. This integration allows AI systems to:

1. Discover available hardware accelerators
2. Find optimal models for specific tasks
3. Run inference on models with hardware acceleration
4. Manage model endpoints and configurations
5. Access distributed inference capabilities

## Key Components

### 1. MCP Server Integration

The MCP server can be integrated into your existing FastAPI application:

```python
from fastapi import FastAPI
from ipfs_accelerate_py.mcp import integrate_with_fastapi, get_ipfs_instance

# Create FastAPI app
app = FastAPI(title="IPFS Accelerate API")

# Get IPFS Accelerate instance
ipfs_instance = get_ipfs_instance()

# Integrate MCP server with FastAPI app
mcp_server = integrate_with_fastapi(
    app=app, 
    ipfs_instance=ipfs_instance, 
    mount_path="/mcp"
)
```

### 2. Standalone MCP Server

Alternatively, you can run a standalone MCP server:

```bash
# Run with default settings
python -m ipfs_accelerate_py.mcp.standalone

# Run with custom settings
python -m ipfs_accelerate_py.mcp.standalone \
    --host 0.0.0.0 \
    --port 8080 \
    --name "My IPFS Accelerate MCP Server" \
    --integrate \
    --verbose
```

### 3. Client Usage

Clients can interact with the MCP server using the FastMCP client:

```python
from fastmcp import FastMCPClient

# Connect to MCP server
client = FastMCPClient("http://localhost:8080")

# Get available hardware
hardware = client.use_tool("test_hardware")
print(f"Available accelerators: {hardware['available_accelerators']}")

# Run inference
result = client.use_tool(
    "run_inference", 
    model="mistralai/Mistral-7B-v0.1",
    inputs=["What is IPFS?"],
    device="cuda:0" if "cuda" in hardware['available_accelerators'] else "cpu"
)
print(result["outputs"][0])
```

## Available Tools

### Hardware Tools

- `test_hardware`: Tests available hardware accelerators
- `test_ipfs_hardware`: Tests hardware using IPFS Accelerate's native detection
- `get_hardware_info`: Gets detailed hardware information
- `get_hardware_recommendation`: Gets hardware recommendations for models

### Inference Tools

- `run_inference`: Runs inference on a model
- `batch_inference`: Runs batch inference on a model
- `get_available_models`: Lists available models for inference

### Endpoint Management Tools

- `get_endpoints`: Lists available inference endpoints
- `add_endpoint`: Adds a new inference endpoint
- `remove_endpoint`: Removes an existing inference endpoint
- `test_endpoint`: Tests an inference endpoint

## Available Resources

### Model Resources

- `models://available`: List of available models
- `models://recommended`: Recommended models for different tasks
- `models://info/{model_id}`: Information about a specific model

### Configuration Resources

- `config://system`: System configuration
- `config://hardware`: Hardware configuration
- `config://acceleration`: Acceleration configuration

## Distributed Inference

IPFS Accelerate MCP provides advanced distributed inference capabilities:

```python
# Get distributed inference capabilities
capabilities = client.use_tool("get_distributed_capabilities")
print(f"Maximum model size: {capabilities['max_model_size_b'] / 1e9}B parameters")

# Run distributed inference
result = client.use_tool(
    "run_distributed_inference",
    model="meta-llama/Llama-2-70b",
    inputs=["Explain distributed model inference"],
    sharding_strategy="tensor_parallel"
)
print(result["outputs"][0])
```

## Documentation

For more detailed information on using the MCP integration with IPFS Accelerate, see the following prompts:

- `hardware_guide`: Guide for hardware selection
- `inference_guide`: Guide for running inference
- `distributed_inference_guide`: Guide for distributed inference
- `model_recommendation`: Guide for model selection
- `getting_started`: Quick start guide

## Next Steps

1. **Install IPFS Accelerate MCP**: Follow the installation guide in MCP_INSTALLATION_GUIDE.md
2. **Run the Integration Test**: Use the mcp_fastapi_integration_test.py script to verify the integration
3. **Explore the Examples**: Check the examples directory for client usage examples
4. **Integrate with Your Application**: Add MCP support to your application using the FastAPI integration

For more information, see the [IPFS Accelerate documentation](https://ipfs-accelerate.docs.example.com).
