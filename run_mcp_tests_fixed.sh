#!/bin/bash
# ===== IPFS Accelerate MCP Tools Test Runner =====
# This script runs the integration tests for MCP server

echo "===== IPFS Accelerate MCP Tools Test Runner ====="
echo "Starting tests at $(date)"

# Activate virtual environment if it exists
if [ -d "ipfs_env" ]; then
    echo "Activating virtual environment..."
    source ipfs_env/bin/activate
    pip install websockets
fi

# Stop any running servers
echo "Stopping any running MCP servers..."
pkill -f "python.*mcp_server.py" || true
sleep 2

# Run tests against minimal MCP server
echo "===== Testing minimal_mcp_server.py ====="
echo "Starting minimal MCP server..."
python minimal_mcp_server.py --port 8001 > minimal_mcp_server.log 2>&1 &
SERVER_PID=$!
sleep 5

echo "Running tests against minimal server..."
# Run the main test script
python test_mcp_tools.py --host localhost --port 8001 || true
# Run detailed integration tests
python test_mcp_server_integration.py --host localhost --port 8001 || true

echo "Stopping minimal MCP server..."
kill $SERVER_PID || true
sleep 2

# Run additional integration tests
echo "Running integration tests..."
python test_ipfs_mcp_integration.py --host localhost || true

# Make sure all servers are stopped
echo "Ensuring all MCP servers are stopped..."
pkill -f "python.*mcp_server.py" || true

echo "Tests completed at $(date)"

# Check for test result files
if [ -d "test_results" ]; then
    PASSED_TESTS=$(grep -l "\"passed\": true" test_results/*.json | wc -l)
    TOTAL_TESTS=$(ls -1 test_results/*.json | wc -l)
    if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
        echo "✅ All tests passed"
        exit 0
    else
        echo "❌ Some tests failed"
        exit 1
    fi
else
    echo "No test results found"
    exit 2
fi
