# IPFS Accelerate MCP Server Installation Guide

This guide provides instructions for installing and setting up the IPFS Accelerate MCP (Model Context Protocol) server.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git (for cloning repositories)

## Installation Steps

### 1. Install the required dependencies

First, install the IPFS Accelerate MCP server dependencies:

```bash
# Navigate to the ipfs_accelerate_py directory
cd /path/to/ipfs_accelerate_py

# Install the requirements
pip install -r mcp/requirements.txt
```

### 2. Install the FastMCP framework

The IPFS Accelerate MCP server is built on the FastMCP framework. Install it using pip:

```bash
pip install fastmcp
```

If you want to install the development version from source:

```bash
git clone https://github.com/model-context-protocol/fastmcp.git
cd fastmcp
pip install -e .
```

### 3. Test the installation

You can verify that everything is installed correctly by running a simple test:

```bash
python -c "from ipfs_accelerate_py.mcp import create_mcp_server; print('MCP server package successfully imported')"
```

## Running the MCP Server

### Basic Usage

To start the MCP server with default settings:

```bash
python -m ipfs_accelerate_py.mcp.server
```

This will start the server on `localhost:8080`.

### Custom Configuration

You can customize the server configuration by passing command-line arguments:

```bash
python -m ipfs_accelerate_py.mcp.server --host 0.0.0.0 --port 9000 --verbose
```

### Using the Server in Your Application

To use the MCP server in your application, you can create a server instance:

```python
from ipfs_accelerate_py.mcp import create_mcp_server

# Create a server
mcp = create_mcp_server(host="localhost", port=8080, verbose=True)

# Start the server
mcp.start()
```

## Client Usage Examples

### Python Client

```python
from fastmcp import FastMCPClient

# Connect to the server
client = FastMCPClient("http://localhost:8080")

# Use a tool
hardware = client.use_tool("test_hardware")
print(f"Available accelerators: {hardware['available_accelerators']}")

# Get hardware recommendation for a model
model_size = 1_000_000_000  # 1B parameters
recommendation = client.use_tool(
    "get_hardware_recommendation",
    model_size=model_size,
    task_type="embedding"
)
print(f"Best hardware: {recommendation['best_recommendation']['device']}")

# Run inference
texts = ["This is a test sentence.", "Another example text."]
results = client.use_tool(
    "run_inference",
    model="BAAI/bge-small-en-v1.5",
    inputs=texts,
    device="cpu"
)
print(f"Generated {len(results['embeddings'])} embeddings")
```

For more detailed examples, see the `ipfs_accelerate_py/mcp/examples` directory.

## Common Issues and Troubleshooting

### Import Errors

If you encounter import errors, ensure that:

1. The `ipfs_accelerate_py` package is in your Python path
2. All dependencies are installed
3. You're using a compatible Python version (3.8+)

### Connection Issues

If you can't connect to the MCP server:

1. Check that the server is running
2. Verify that the host and port are correct
3. Ensure there are no firewall restrictions blocking the connection

### Hardware Detection Issues

If hardware detection is not working correctly:

1. Ensure you have the necessary drivers installed (e.g., CUDA for NVIDIA GPUs)
2. Check that your hardware is supported
3. Run the server with `--verbose` flag for more detailed logs

## Next Steps

- Explore the [Client Example](examples/client_example.py) for more advanced usage
- Read the [Getting Started Guide](GETTING_STARTED.md) for a more detailed introduction
- See the [README](README.md) for an overview of the project

## Support and Feedback

If you encounter any issues or have feedback, please open an issue on the [GitHub repository](https://github.com/ipfs/ipfs_accelerate_py).
