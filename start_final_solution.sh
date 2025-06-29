#!/bin/bash
#
# IPFS Accelerate MCP - Final Solution Startup and Test Script
# This script stops existing processes, starts the final MCP server,
# runs the comprehensive test suite, and stops the server.
#

set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define the target port
MCP_PORT=8004
MCP_HOST="127.0.0.1"
SERVER_PID_FILE="final_mcp_server.pid"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - Final Solution Test Script     ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Function to stop processes by name
stop_processes() {
    local process_name="$1"
    echo -e "${YELLOW}Attempting to stop processes matching: ${process_name}${NC}"
    # Use pgrep to find PIDs and pkill to terminate them
    pkill -f "${process_name}" || true
    # Give processes a moment to stop gracefully
    sleep 2
    # Force kill if still running
    if pgrep -f "${process_name}" > /dev/null; then
        echo -e "${YELLOW}Forcing termination of processes matching: ${process_name}${NC}"
        pkill -9 -f "${process_name}" || true
        sleep 1
    fi
}

# Stop any potentially running servers or test scripts
stop_processes "final_mcp_server.py"
stop_processes "clean_mcp_server.py"
stop_processes "mcp.run_server"
stop_processes "test_mcp_server_comprehensive.py"

# Clean up old PID file if it exists
if [ -f "$SERVER_PID_FILE" ]; then
    rm "$SERVER_PID_FILE" || true
fi

# Start the final_mcp_server.py in the background
echo -e "${YELLOW}Starting final_mcp_server.py on ${MCP_HOST}:${MCP_PORT} in the background...${NC}"
# Ensure python3 is used and redirect output to a log file
python3 final_mcp_server.py --host "$MCP_HOST" --port "$MCP_PORT" &> server_startup.log &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo -e "${GREEN}Server started with PID ${SERVER_PID}. Output redirected to server_startup.log${NC}"

# Give the server time to start and initialize
echo -e "${YELLOW}Giving the server 20 seconds to fully start and register tools...${NC}"
sleep 20

# Check if the server process is still running
if ! kill -0 "$SERVER_PID" > /dev/null 2>&1; then
    echo -e "${RED}Server process (PID ${SERVER_PID}) is not running. Check server_startup.log for errors.${NC}"
    exit 1
fi
echo -e "${GREEN}Server process (PID ${SERVER_PID}) is running.${NC}"

# Run the comprehensive test script targeting the specific port
echo -e "${YELLOW}Running comprehensive MCP server tests targeting ${MCP_HOST}:${MCP_PORT}...${NC}"
# Ensure python3 is used and run the test script (with auto-fix enabled)
echo -e "${YELLOW}Running comprehensive MCP server tests (initial run with auto-fix) targeting ${MCP_HOST}:${MCP_PORT}...${NC}"
python3 test_mcp_server_comprehensive.py --host "$MCP_HOST" --port "$MCP_PORT" --output post_startup_diagnostics.json --auto-fix --timeout 15

INITIAL_TEST_EXIT_CODE=$?

# Stop the background server process
echo -e "${YELLOW}Stopping the background server process (PID ${SERVER_PID})...${NC}"
kill "$SERVER_PID" || true # Attempt graceful shutdown first
sleep 5 # Give it a moment to shut down
if kill -0 "$SERVER_PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}Server process (PID ${SERVER_PID}) did not stop gracefully, forcing kill...${NC}"
    kill -9 "$SERVER_PID" || true # Force kill if still running
fi

# Clean up PID file
if [ -f "$SERVER_PID_FILE" ]; then
    rm "$SERVER_PID_FILE" || true
fi

# --- Restart server to load fixes ---
echo -e "${YELLOW}Restarting final_mcp_server.py on ${MCP_HOST}:${MCP_PORT} in the background to load fixes...${NC}"
# Ensure python3 is used and redirect output to a log file
python3 final_mcp_server.py --host "$MCP_HOST" --port "$MCP_PORT" &> server_restart.log &
SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo -e "${GREEN}Server restarted with PID ${SERVER_PID}. Output redirected to server_restart.log${NC}"

# Give the server time to start and initialize
echo -e "${YELLOW}Giving the server 20 seconds to fully start and register tools after restart...${NC}"
sleep 20

# Check if the server process is still running
if ! kill -0 "$SERVER_PID" > /dev/null 2>&1; then
    echo -e "${RED}Server process (PID ${SERVER_PID}) is not running after restart. Check server_restart.log for errors.${NC}"
    exit 1
fi
echo -e "${GREEN}Server process (PID ${SERVER_PID}) is running after restart.${NC}"

# --- Run comprehensive test script again to verify fixes ---
echo -e "${YELLOW}Running comprehensive MCP server tests (post-fix verification) targeting ${MCP_HOST}:${MCP_PORT}...${NC}"
# Ensure python3 is used and run the test script
python3 test_mcp_server_comprehensive.py --host "$MCP_HOST" --port "$MCP_PORT" --output post_fix_diagnostics.json --timeout 15

POST_FIX_TEST_EXIT_CODE=$?

# Stop the background server process (after the second test run)
echo -e "${YELLOW}Stopping the background server process (PID ${SERVER_PID}) after verification tests...${NC}"
kill "$SERVER_PID" || true # Attempt graceful shutdown first
sleep 5 # Give it a moment to shut down
if kill -0 "$SERVER_PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}Server process (PID ${SERVER_PID}) did not stop gracefully, forcing kill...${NC}"
    kill -9 "$SERVER_PID" || true # Force kill if still running
fi

# Clean up PID file
if [ -f "$SERVER_PID_FILE" ]; then
    rm "$SERVER_PID_FILE" || true
fi

# Report final test results based on the exit code of the second test script run
if [ $POST_FIX_TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Comprehensive tests completed successfully after applying fixes.${NC}"
    echo -e "${GREEN}Check post_startup_diagnostics.json (initial run) and post_fix_diagnostics.json (post-fix run) for detailed results.${NC}"
else
    echo -e "${RED}Comprehensive tests failed after applying fixes.${NC}"
    echo -e "${RED}Check post_startup_diagnostics.json, post_fix_diagnostics.json, server_startup.log, server_restart.log, and mcp_test_comprehensive.log for details.${NC}"
fi

exit $POST_FIX_TEST_EXIT_CODE
