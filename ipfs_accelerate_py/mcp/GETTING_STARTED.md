# Getting Started with IPFS Accelerate MCP

This guide will help you get up and running with the IPFS Accelerate MCP integration.

## Prerequisites

- Python 3.8 or higher
- IPFS Accelerate Python package

## Quick Setup

1. **Install required dependencies:**
   ```bash
   cd ipfs_accelerate_py/mcp
   pip install -r requirements.txt
   ```

2. **Start the MCP server:**

   You can start the server directly:
   ```bash
   python -m ipfs_accelerate_py.mcp.server --debug
   ```

   Or use the helper script that automatically finds an available port:
   ```bash
   python run_ipfs_mcp.py --debug --find-port
   ```

3. **Access the API documentation:**
   
   Open your browser and navigate to http://localhost:8000/docs to see the interactive API documentation.

## Understanding MCP

The Model Context Protocol (MCP) is a standardized way for AI models to interact with external tools and resources. The IPFS Accelerate MCP integration provides:

- **Tools**: Executable functions that perform actions (hardware detection, model inference, etc.)
- **Resources**: Data sources that provide information (model details, system status, etc.)

## Basic API Usage

### Using Tools

Tools are accessed via POST requests to the `/mcp/tool/{tool_name}` endpoint:

```python
import requests

# Example: Get hardware information
response = requests.post(
    "http://localhost:8000/mcp/tool/get_hardware_info",
    json={}
)
print(response.json())

# Example: Test hardware compatibility
response = requests.post(
    "http://localhost:8000/mcp/tool/test_hardware",
    json={"accelerator": "all", "test_level": "basic"}
)
print(response.json())

# Example: Get hardware recommendations
response = requests.post(
    "http://localhost:8000/mcp/tool/recommend_hardware",
    json={"model_name": "llama-7b"}
)
print(response.json())
```

### Accessing Resources

Resources are accessed via GET requests to the `/mcp/resource/{resource_uri}` endpoint:

```python
import requests

# Example: Get supported models information
response = requests.get(
    "http://localhost:8000/mcp/resource/ipfs_accelerate/supported_models"
)
print(response.json())
```

## Example Code

For a complete working example, see the `examples/client_example.py` file:

```bash
cd ipfs_accelerate_py/mcp/examples
python client_example.py
```

This example demonstrates:
1. Connecting to the MCP server
2. Getting hardware information
3. Retrieving supported model information
4. Testing hardware compatibility
5. Getting hardware recommendations

## Next Steps

- Read the [Comprehensive MCP Guide](./COMPREHENSIVE_MCP_GUIDE.md) for detailed information on extending and customizing the MCP server
- Review the [Integration Summary](./INTEGRATION_SUMMARY.md) for an overview of the architecture and implementation
- Explore the API documentation at http://localhost:8000/docs

## Troubleshooting

### Server won't start
- Check if another process is using port 8000
- Try specifying a different port: `python -m ipfs_accelerate_py.mcp.server --port 8001`

### Cannot connect to server
- Ensure the server is running
- Check if the host and port are correct
- Try accessing the API documentation at http://localhost:8000/docs

### Errors when using tools
- Check the API documentation for the correct parameters
- Verify that required parameters are provided
- Look at the server logs for detailed error information
