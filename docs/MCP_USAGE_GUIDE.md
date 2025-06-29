# IPFS Accelerate MCP Server Usage Guide

This guide provides detailed instructions for setting up and running the IPFS Accelerate MCP server for integration with Cline and other MCP clients.

## Overview

The IPFS Accelerate MCP server exposes IPFS and hardware acceleration functionality to Large Language Models through a standardized API. Two implementation options are available:

1. **Full MCP Server** (`run_mcp.py`): Using the FastMCP library with comprehensive features
2. **Direct Server** (`direct_mcp_server.py`): Simplified Flask-based implementation for quick testing

## Prerequisites

### Option 1: Full MCP Server

To use the full MCP server with FastMCP, install these dependencies:

```bash
pip install -r mcp/requirements-mcp.txt
```

### Option 2: Direct Flask-based Server

To use the simplified direct server:

```bash
pip install flask flask-cors
```

## Running the MCP Server

### Option 1: Full MCP Server

```bash
python run_mcp.py --debug
```

Additional options:
- `--name`: Server name (default: "direct-ipfs-kit-mcp")
- `--description`: Server description
- `--transport`: Transport type (stdio or sse, default: sse)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 3000)

### Option 2: Direct Flask-based Server

```bash
python direct_mcp_server.py --debug
```

Additional options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 3000)

## Cline Integration

### Configuration

Cline connects to the MCP server through its configuration. The default configuration is:

```json
{
  "mcpServers": {
    "direct-ipfs-kit-mcp": {
      "disabled": false,
      "timeout": 60,
      "url": "http://localhost:3000/sse",
      "transportType": "sse"
    }
  }
}
```

This should match the server configuration you're using. The default settings in `run_mcp.py` and `direct_mcp_server.py` are already configured to work with this Cline config.

### Using MCP Tools in Cline

To use the MCP tools from Cline, use the `use_mcp_tool` capability:

```
<use_mcp_tool>
<server_name>direct-ipfs-kit-mcp</server_name>
<tool_name>ipfs_add_file</tool_name>
<arguments>
{
  "path": "/path/to/file.txt"
}
</arguments>
</use_mcp_tool>
```

Available tools include:
- `ipfs_add_file`: Add a file to IPFS
- `ipfs_cat`: Retrieve content from IPFS
- `ipfs_files_write`: Write content to the IPFS MFS
- `ipfs_files_read`: Read content from the IPFS MFS
- `health_check`: Check the health of the MCP server

## Troubleshooting

### Server Won't Start

**Issue**: `ImportError: No module named 'fastmcp'` or similar missing dependencies.

**Solution**: Install the required dependencies:
```bash
pip install -r mcp/requirements-mcp.txt
```

If FastMCP is not available, use the direct Flask-based server instead:
```bash
pip install flask flask-cors
python direct_mcp_server.py
```

### Cline Can't Connect to Server

**Issue**: Cline reports it cannot connect to the MCP server.

**Solution**:
1. Check if the server is running: `curl http://localhost:3000/`
2. Ensure the port in Cline's configuration matches the server's port
3. Check if there are any firewall issues blocking the connection

### "Unknown Tool" Error

**Issue**: Cline reports "Unknown tool" when trying to use an MCP tool.

**Solution**: 
1. Check if the tool name is spelled correctly
2. Make sure you're using the correct server name in the `use_mcp_tool` capability
3. Restart the MCP server and try again

## Advanced Usage

### Custom Tool Implementation

To add a new tool to the MCP server:

1. Create a new module in the `mcp/tools` directory
2. Implement the tool functions using the `@mcp.tool()` decorator
3. Register the tool by adding its registration function to `mcp/tools/__init__.py`

### Debugging

For detailed logging, run the server with the `--debug` flag:

```bash
python run_mcp.py --debug
```

This will show detailed logs of all tool calls and server operations.

## Conclusion

The IPFS Accelerate MCP integration provides a powerful way to connect LLMs with IPFS and hardware acceleration functionality. By following this guide, you should be able to successfully set up and use the MCP server with Cline or other MCP clients.
