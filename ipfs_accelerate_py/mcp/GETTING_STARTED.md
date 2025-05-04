# Getting Started with IPFS Accelerate MCP

This guide provides step-by-step instructions for getting started with the IPFS Accelerate MCP integration.

## Installation

### Prerequisites

Before installing the MCP integration, ensure you have:

- Python 3.8 or higher
- IPFS Accelerate package installed
- Pip package manager

### Installing Required Packages

1. Install the core dependencies:

```bash
pip install fastmcp fastapi uvicorn requests pydantic
```

2. Install hardware information dependencies:

```bash
pip install psutil py-cpuinfo
```

3. (Optional) Install hardware acceleration libraries:

```bash
# For CUDA support
pip install torch

# For WebGPU support
pip install wgpu

# For OpenVINO support
pip install openvino
```

4. Clone and install the IPFS Accelerate package (if not already installed):

```bash
git clone https://github.com/your-org/ipfs_accelerate_py.git
cd ipfs_accelerate_py
pip install -e .
```

## Running the MCP Server

There are several ways to run the MCP server, depending on your needs.

### Option 1: Running the Standalone Server

The simplest way to get started is to run the standalone server:

```bash
# Navigate to the project directory
cd ipfs_accelerate_py

# Run the standalone server
python -m ipfs_accelerate_py.mcp.standalone
```

This will start the server on the default host (`localhost`) and port (`8080`). You should see output similar to:

```
INFO - ipfs_accelerate_mcp.standalone - Starting MCP server at http://localhost:8080
INFO - ipfs_accelerate_mcp.server - Created MCP server at http://localhost:8080
INFO - ipfs_accelerate_mcp.tools - Registering MCP tools
INFO - ipfs_accelerate_mcp.tools.hardware - Hardware tools registered successfully
INFO - ipfs_accelerate_mcp.tools - All MCP tools registered successfully
INFO - ipfs_accelerate_mcp.resources - Registering MCP resources
INFO - ipfs_accelerate_mcp.resources.config - Configuration resources registered successfully
INFO - ipfs_accelerate_mcp.resources - All MCP resources registered successfully
INFO - ipfs_accelerate_mcp.server - Registered all server components
INFO - ipfs_accelerate_mcp.standalone - Server created, registered components
INFO - ipfs_accelerate_mcp.standalone - Server URL: http://localhost:8080
INFO - ipfs_accelerate_mcp.standalone - Press Ctrl+C to stop the server
```

To customize the server configuration, you can use command-line arguments:

```bash
python -m ipfs_accelerate_py.mcp.standalone --host 0.0.0.0 --port 9090 --verbose
```

This will start the server on all network interfaces (`0.0.0.0`), on port `9090`, with verbose logging.

### Option 2: Integrating with Your Python Code

You can also integrate the MCP server directly into your Python code:

```python
from ipfs_accelerate_py.mcp import create_mcp_server, start_server

# Create server
server = create_mcp_server(
    host="localhost",
    port=8080,
    name="ipfs-accelerate",
    description="IPFS Accelerate MCP Server"
)

# Start server (this will block until the server is stopped)
start_server(server)
```

This is useful if you want to add custom logic or error handling.

### Option 3: Integrating with FastAPI

If you're already using FastAPI, you can integrate the MCP server with your application:

```python
from fastapi import FastAPI
from ipfs_accelerate_py.mcp import integrate_with_fastapi

# Create FastAPI app
app = FastAPI(title="My Application with IPFS Accelerate")

# Mount the MCP server at the /mcp path
mcp_server = integrate_with_fastapi(app, mount_path="/mcp")

# You can now add your own routes
@app.get("/")
def read_root():
    return {"message": "Welcome to my application with IPFS Accelerate"}

# Start the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

Start your FastAPI application as usual:

```bash
python your_app.py
```

Or using Uvicorn directly:

```bash
uvicorn your_app:app --reload
```

## Interacting with the MCP Server

Once the server is running, you can interact with it in various ways.

### Using the Client Example

The package includes a client example that you can use to interact with the server:

```bash
# Get server information
python -m ipfs_accelerate_py.mcp.examples.client_example --info

# Access a specific resource
python -m ipfs_accelerate_py.mcp.examples.client_example --resource ipfs_accelerate/system_info

# Call a tool
python -m ipfs_accelerate_py.mcp.examples.client_example --tool test_hardware

# Call a tool with arguments
python -m ipfs_accelerate_py.mcp.examples.client_example --tool recommend_hardware --args '{"model_name": "llama2-7b"}'
```

### Making API Requests Directly

You can also interact with the MCP server using HTTP requests:

#### Accessing Resources

Resources are accessed via GET requests:

```bash
# Get system information
curl http://localhost:8080/resources/ipfs_accelerate/system_info

# Get version information
curl http://localhost:8080/resources/ipfs_accelerate/version

# Get configuration information
curl http://localhost:8080/resources/ipfs_accelerate/config

# Get supported models information
curl http://localhost:8080/resources/ipfs_accelerate/supported_models
```

#### Calling Tools

Tools are called via POST requests with JSON payloads:

```bash
# Test hardware
curl -X POST http://localhost:8080/tools/test_hardware -H "Content-Type: application/json" -d '{}'

# Get hardware information
curl -X POST http://localhost:8080/tools/get_hardware_info -H "Content-Type: application/json" -d '{}'

# Recommend hardware for a specific model
curl -X POST http://localhost:8080/tools/recommend_hardware -H "Content-Type: application/json" -d '{"model_name": "llama2-7b", "parameter_size": 7}'

# Test IPFS hardware acceleration
curl -X POST http://localhost:8080/tools/test_ipfs_hardware -H "Content-Type: application/json" -d '{"tests": ["basic", "cuda"]}'
```

### Using from Python Code

You can also interact with the MCP server from your Python code:

```python
import requests

# Define server URL
server_url = "http://localhost:8080"

# Call a tool
response = requests.post(
    f"{server_url}/tools/test_hardware",
    json={}
)
hardware_info = response.json()
print(hardware_info)

# Get a resource
response = requests.get(f"{server_url}/resources/ipfs_accelerate/system_info")
system_info = response.json()
print(system_info)
```

For a more complete client implementation, see the client example:

```python
from ipfs_accelerate_py.mcp.examples.client_example import MCPClient

# Create client
client = MCPClient(url="http://localhost:8080")

# Get server information
server_info = client.get_server_info()
print(server_info)

# Call a tool
hardware_test = client.call_tool("test_hardware", {"include_benchmarks": True})
print(hardware_test)

# Access a resource
system_info = client.access_resource("ipfs_accelerate/system_info")
print(system_info)
```

## Integrating with Language Models

The MCP integration is designed to be used by language models through the Model Context Protocol. When a language model connects to the MCP server, it can:

1. Access the server's resources to get information about the system and IPFS Accelerate
2. Call the server's tools to interact with IPFS Accelerate

For example, a language model might want to get information about the hardware available for inference:

```
1. LM calls the `test_hardware` tool to get information about available hardware
2. LM accesses the `ipfs_accelerate/supported_models` resource to get information about supported models
3. LM calls the `recommend_hardware` tool to recommend hardware for a specific model
```

## Troubleshooting

### Server Won't Start

If the server won't start, check:

- You have the required dependencies installed
- The port is not in use by another application
- You have the correct permissions to bind to the specified port

### Cannot Connect to Server

If you cannot connect to the server, check:

- The server is running
- You're using the correct URL
- There are no firewall rules blocking the connection
- If you're using `localhost`, ensure you're connecting from the same machine

### Tool or Resource Not Found

If a tool or resource is not found, check:

- You're using the correct path (e.g., `ipfs_accelerate/system_info` for the system information resource)
- The tool or resource is registered with the server
- There are no typos in the path

### Server Errors

If the server returns an error, check:

- The error message for details
- The server logs for more information
- You're using the correct input format for tools

## Next Steps

Now that you have the MCP server running, you can:

- Explore the available resources and tools
- Integrate the MCP server with your applications
- Extend the MCP server with custom tools and resources

For more information, see the [README.md](./README.md) file.
