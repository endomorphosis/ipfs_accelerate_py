# Getting Started with IPFS Accelerate MCP

This guide provides step-by-step instructions for setting up and using the MCP (Model Context Protocol) server integration with IPFS Accelerate.

## Prerequisites

- Python 3.8 or higher
- IPFS Accelerate Python package installed
- `pip` package manager

## Installation

1. Install the required dependencies:

```bash
pip install -r mcp/requirements.txt
```

## Running the MCP Server

### Option 1: Standalone Mode

Run the MCP server directly using the CLI:

```bash
# From the project root
python -m mcp.cli --host 127.0.0.1 --port 8000
```

For development mode with auto-reload:

```bash
python -m mcp.cli --dev
```

### Option 2: Integration with FastAPI Application

The MCP server can be integrated with the main FastAPI application in `main.py`. This happens automatically if you're starting the server through the standard means.

If you need to manually integrate the MCP server with your own FastAPI application:

```python
from fastapi import FastAPI
from mcp.main_integration import add_mcp_to_main_app

app = FastAPI()
model_server = ...  # Your model server instance

# Add MCP server to the FastAPI app
add_mcp_to_main_app(app, model_server)
```

## Using the MCP Client Example

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

## Connecting to the MCP Server from External Applications

The MCP server implements the standard Model Context Protocol, allowing it to be used with any MCP-compatible client or language model. The server endpoints are:

- Server Info: `http://localhost:8000/mcp`
- Resources: `http://localhost:8000/mcp/resources/{resource_path}`
- Tools: `http://localhost:8000/mcp/tools/{tool_name}`

## Using with Anthropic Claude

You can use the MCP server with Anthropic Claude by:

1. Start the MCP server in development mode:
   ```bash
   python -m mcp.cli --dev
   ```

2. Access the MCP Inspector in your browser at:
   ```
   http://localhost:8000/
   ```

3. Start Claude Desktop and enable MCP

4. Point Claude to your local MCP server:
   ```
   http://localhost:8000/
   ```

5. Claude can now directly interact with the IPFS Accelerate system through the MCP server.

## Running Tests

To verify the MCP server is functioning correctly, run the test suite:

```bash
python -m unittest discover -s ipfs_accelerate_py/mcp/tests
```

## Troubleshooting

### Common Issues

#### Port Already in Use

If you see an error like "Address already in use", try a different port:

```bash
python -m mcp.cli --port 8001
```

#### Missing Dependencies

If you encounter ImportError or ModuleNotFoundError, install the missing dependencies:

```bash
pip install -r mcp/requirements.txt
```

#### Connection Refused

If clients can't connect to the server, make sure:
- The server is running
- You're using the correct host and port
- There's no firewall blocking the connection

## Next Steps

Once you have the MCP server running, you can:

1. Explore the available tools and resources using the client example
2. Integrate with language models for model-assisted exploration of IPFS Accelerate
3. Use the MCP server to facilitate hardware-accelerated inference from any application
