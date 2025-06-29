#!/bin/bash

# Run IPFS Accelerate MCP Server
# This script activates the virtual environment and runs the MCP server

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source ipfs_env/bin/activate

# Run the simple MCP server
echo "Starting IPFS Accelerate MCP Server..."
echo "Server will be available at: http://localhost:8000/sse"
echo "Use Ctrl+C to stop the server"
echo ""

python simple_mcp_server.py --port 8000 --debug

# Deactivate the virtual environment when done
deactivate
