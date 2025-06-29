#!/bin/bash
# Install IPFS Accelerate MCP dependencies and run the server with SSE transport

echo "Installing IPFS Accelerate MCP dependencies..."
pip install -r ipfs_accelerate_py/mcp/requirements.txt

echo "Starting IPFS Accelerate MCP server with SSE transport..."
python run_ipfs_mcp.py --debug --find-port --transport sse

# Exit with the status of the last command
exit $?
