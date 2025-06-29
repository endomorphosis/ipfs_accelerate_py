#!/bin/bash
#
# Run Enhanced MCP Server for IPFS Accelerate
#
# This script starts the enhanced MCP server on port 8002

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask..."
    pip install flask flask-cors
fi

# Kill any existing MCP servers on port 8002
echo "Checking for existing MCP servers on port 8002..."
PID=$(lsof -t -i:8002 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "Stopping existing MCP server (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 1
fi

# Start the MCP server
echo "Starting Enhanced MCP Server on port 8002..."
python3 enhanced_mcp_server.py --port 8002 &

# Wait for the server to start
sleep 2

# Check if the server started successfully
if ! curl -s http://localhost:8002/mcp/manifest > /dev/null; then
    echo "Failed to start MCP server"
    exit 1
fi

echo "Enhanced MCP Server is running"
echo "MCP Manifest: http://localhost:8002/mcp/manifest"
echo "SSE Endpoint: http://localhost:8002/sse"
echo ""
echo "To test the server, run:"
echo "  curl http://localhost:8002/tools"
echo "  curl http://localhost:8002/mcp/manifest"
echo ""
echo "To connect Claude to this server, ensure your MCP settings have:"
echo '  "ipfs-accelerate-mcp": {'
echo '    "disabled": false,'
echo '    "timeout": 60,'
echo '    "url": "http://localhost:8002/sse",'
echo '    "transportType": "sse"'
echo '  }'
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for the background process
wait
