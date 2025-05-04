# IPFS Accelerate MCP Integration

This directory contains the Model Context Protocol (MCP) integration for IPFS Accelerate. The MCP server exposes IPFS and acceleration functionality to Large Language Models (LLMs) through a standardized API.

## Overview

The IPFS Accelerate MCP server provides tools for:

- IPFS file operations (add, get, pin, etc.)
- IPFS network operations (peer discovery, swarm management, etc.)
- Hardware acceleration of AI models
- Integration with Filecoin and other decentralized storage systems

## Architecture

The MCP integration is organized into the following components:

- **Server**: Core MCP server implementation and lifecycle management
- **Tools**: Individual tools exposed through the MCP API, organized by functionality
- **Types**: Shared type definitions and context objects
- **Mock**: Mock implementations for development and testing without dependencies

## Getting Started

### Installation

Install the required dependencies:

```bash
pip install -r mcp/requirements-mcp.txt
```

### Running the Server

The MCP server can be started using the provided script:

```bash
python run_mcp.py --debug
```

Additional options:

```bash
python run_mcp.py --help
```

## Configuration

The MCP server can be configured through command-line options:

- `--name`: Server name
- `--description`: Server description
- `--transport`: Transport type (stdio or sse)
- `--host`: Host to bind to for network transports
- `--port`: Port to bind to for network transports
- `--debug`: Enable debug logging

## Dependencies

- `fastmcp`: Model Context Protocol implementation
- `ipfs-kit-py`: Python bindings for IPFS
- Optional: WebSocket or SSE dependencies for different transport types

## Development

The MCP server includes fallback mock implementations that can be used when dependencies are not available, making it easier to develop and test without a full IPFS installation.

For local development:

1. Create a virtual environment
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run the server with debug logging: `python run_mcp.py --debug`

## Extension

To add new tools to the MCP server:

1. Create a new module in the `mcp/tools` directory
2. Implement the tool functions using the `@mcp.tool()` decorator
3. Create a registration function: `register_*_tools(mcp: FastMCP)`
4. Import and register your tools in `mcp/tools/__init__.py`

## License

This project is licensed under the same license as the IPFS Accelerate project.
