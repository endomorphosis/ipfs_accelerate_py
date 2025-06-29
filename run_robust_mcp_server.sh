#!/bin/bash

# Run the robust MCP server with error handling
echo "Starting robust MCP server for IPFS Accelerate..."

# Kill any existing MCP servers
pkill -f "python.*mcp.*server" || true
sleep 1

# Set the port (default: 8002)
PORT=${1:-8002}

# Run the server in the background
python robust_mcp_server.py --port $PORT &

# Store the PID
PID=$!
echo "Server started with PID: $PID"

# Create a PID file
echo $PID > robust_mcp_server.pid

# Wait a moment to ensure server starts
sleep 2

# Check if the server is running
if ps -p $PID > /dev/null; then
    echo "Server is running successfully on port $PORT"
    echo "Server logs are in robust_mcp_server_*.log"
    echo "To connect with Claude, use the ipfs-accelerate MCP tools"
else
    echo "Failed to start the server. Check the logs for errors."
    exit 1
fi

echo "Press Ctrl+C to stop the server"
wait $PID