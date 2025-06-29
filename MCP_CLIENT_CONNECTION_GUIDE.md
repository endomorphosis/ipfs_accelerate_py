# MCP Client Connection Guide

This guide explains how to resolve connection issues between Claude and the IPFS Accelerate MCP server.

## Prerequisites

1. The IPFS Accelerate MCP server is running on port 8002
2. The MCP settings file has been updated correctly

## Steps to Resolve Connection Issues

### Step A: Verify MCP Server Functionality

1. Confirm the MCP server is running:
   ```bash
   pgrep -f "python.*start_mcp_with_ipfs.py"
   ```

2. Verify the server is responding to HTTP requests:
   ```bash
   curl -i http://localhost:8002/mcp/manifest
   ```
   This should return a 200 OK status and a JSON response with the server manifest.

3. Verify the SSE endpoint is accessible:
   ```bash
   curl http://localhost:8002/sse
   ```
   This should establish a connection that remains open (cancel with Ctrl+C).

### Step B: Restart VSCode and the MCP Connection

1. Save all open files in VSCode.

2. Close VSCode completely:
   - File > Exit or press Ctrl+Q
   - Verify VSCode has fully closed by checking the process list:
     ```bash
     pgrep -f code
     ```

3. Restart VSCode.

4. After VSCode restarts, reload the Claude extension:
   - Open the Command Palette (Ctrl+Shift+P)
   - Type and select "Developer: Reload Window"

### Step C: Verify the Connection

1. Start a new conversation with Claude.

2. Test the connection with:
   ```
   use_mcp_tool with server_name='ipfs-accelerate-mcp', tool_name='ipfs_node_info', arguments={}
   ```

3. If the connection is working, Claude should respond with information about the IPFS node.

### Step D: Additional Troubleshooting (if needed)

If the connection issues persist, try these additional steps:

1. **Check MCP Settings**: Verify the MCP settings file:
   ```bash
   cat ~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
   ```
   Confirm the file contains the correct server configuration.

2. **Disable and Re-enable the Server**: In the settings file, try setting "disabled" to true, save the file, then set it back to false and save again.

3. **Try a Different Server Name**: If "ipfs-accelerate-mcp" doesn't work, try one of the other server names in the settings file, such as "ipfs-accelerate-py".

4. **Check Firewall Settings**: Ensure no firewall is blocking the connection to localhost on port 8002.

5. **Clear Extension Cache**:
   - Close VSCode
   - Delete the extension cache directory:
     ```bash
     rm -rf ~/.config/Code/Cache/*
     ```
   - Restart VSCode

6. **Check Server Logs**: If the server is running in a terminal, check for any error messages when connection attempts are made.

## Understanding the Connection Process

The MCP client connection works as follows:

1. The MCP server registers tools and resources with the HTTP server
2. The server exposes an SSE (Server-Sent Events) endpoint at `/sse`
3. Claude's MCP client connects to this endpoint to receive updates
4. When a tool or resource is requested, the client sends a request to the server via the SSE connection
5. The server processes the request and sends the result back to the client

If any part of this process is interrupted, the connection will fail.

## Common Issues

- **Incorrect URL**: The server URL in the settings file must match the actual URL where the server is running.
- **Server Not Running**: The MCP server must be running for the connection to work.
- **SSE Connection Issues**: The SSE connection may be interrupted or blocked.
- **Extension Cache Issues**: The VSCode extension cache may need to be cleared.
- **Reload Required**: VSCode and/or the Claude extension may need to be reloaded to establish a fresh connection.

By following the steps in this guide, you should be able to resolve any connection issues between Claude and the IPFS Accelerate MCP server.
