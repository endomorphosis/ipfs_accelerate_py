#!/bin/bash

# Run the comprehensive MCP tool tests for ipfs_accelerate_py

echo "===== IPFS Accelerate MCP Tools Test Runner ====="
echo "Starting tests at $(date)"

# Set up Python environment
if [ -d "./ipfs_env" ]; then
  echo "Activating virtual environment..."
  source ./ipfs_env/bin/activate
  # Install required packages
  pip install websockets
fi

# Kill any running MCP server processes
echo "Stopping any running MCP servers..."
pkill -f "python.*mcp_server.py" || true
sleep 2

# Test simple_mcp_server.py first
if [ -f "simple_mcp_server.py" ]; then
  echo "===== Testing simple_mcp_server.py ====="
  echo "Starting simple MCP server..."
  python simple_mcp_server.py --port 8000 > simple_mcp_server.log 2>&1 &
  SIMPLE_SERVER_PID=$!
  sleep 5  # Wait for the server to start
  
  # Run tests against simple server
  echo "Running tests against simple server..."
  python test_mcp_tools.py --host localhost --port 8000
  python test_mcp_server_integration.py --host localhost --port 8000
  
  # Stop simple server
  echo "Stopping simple MCP server..."
  kill $SIMPLE_SERVER_PID || true
  sleep 2
else
  echo "simple_mcp_server.py not found, skipping tests"
fi

# Test unified_mcp_server.py
if [ -f "unified_mcp_server.py" ]; then
  echo "===== Testing unified_mcp_server.py ====="
  echo "Starting unified MCP server..."
  python unified_mcp_server.py --port 8001 > unified_mcp_server.log 2>&1 &
  UNIFIED_SERVER_PID=$!
  sleep 5  # Wait for the server to start
  
  # Run tests against unified server
  echo "Running tests against unified server..."
  python test_mcp_tools.py --host localhost --port 8001
  python test_mcp_server_integration.py --host localhost --port 8001
  
  # Stop unified server
  echo "Stopping unified MCP server..."
  kill $UNIFIED_SERVER_PID || true
  sleep 2
else
  echo "unified_mcp_server.py not found, skipping tests"
fi

# Run integration tests on port 8001
echo "Running integration tests..."
python test_ipfs_mcp_integration.py --host localhost --port 8001

# Capture the exit code
TEST_EXIT_CODE=$?

# Make sure all MCP servers are stopped
echo "Ensuring all MCP servers are stopped..."
pkill -f "python.*mcp_server.py" || true
sleep 2

echo "Tests completed at $(date)"
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "✅ All tests passed!"
else
  echo "❌ Some tests failed with exit code $TEST_EXIT_CODE"
fi

exit $TEST_EXIT_CODE
