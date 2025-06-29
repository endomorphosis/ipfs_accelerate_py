#!/bin/bash
#
# IPFS Accelerate MCP Server - Verification and Auto-fix Script
#
# This script runs comprehensive tests on the MCP server and automatically
# fixes issues with tool registration and endpoint configuration.
#

set -e  # Exit on error

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - Verification and Auto-Fix      ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Set default port and host
MCP_HOST="127.0.0.1"
MCP_PORT=8002
START_SERVER=false
RESTART=true
OUTPUT_FILE="mcp_verification_results.json"
PYTHON="python3"

# Process command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      MCP_PORT="$2"
      shift
      shift
      ;;
    --host)
      MCP_HOST="$2"
      shift
      shift
      ;;
    --start-server)
      START_SERVER=true
      shift
      ;;
    --no-restart)
      RESTART=false
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 [--port PORT] [--host HOST] [--start-server] [--no-restart] [--output FILE]"
      echo ""
      echo "Options:"
      echo "  --port PORT       MCP server port (default: 8002)"
      echo "  --host HOST       MCP server host (default: 127.0.0.1)"
      echo "  --start-server    Start the server if not running"
      echo "  --no-restart      Don't restart the server after applying fixes"
      echo "  --output FILE     Output file for test results (default: mcp_verification_results.json)"
      echo ""
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--port PORT] [--host HOST] [--start-server] [--no-restart] [--output FILE]"
      exit 1
      ;;
  esac
done

# Check Python version
if ! command -v $PYTHON &>/dev/null; then
  echo -e "${RED}Python not found. Please install Python 3.8 or newer.${NC}"
  exit 1
fi

# Build command line arguments for test script
TEST_ARGS=(
  "--host" "$MCP_HOST"
  "--port" "$MCP_PORT"
  "--output" "$OUTPUT_FILE"
  "--auto-fix"
)

# Add optional arguments based on flags
if [ "$START_SERVER" = true ]; then
  TEST_ARGS+=("--start-server")
fi

if [ "$RESTART" = false ]; then
  TEST_ARGS+=("--no-restart")
fi

# Check if the test script exists
if [ ! -f "test_mcp_server_comprehensive.py" ]; then
  echo -e "${RED}Test script not found: test_mcp_server_comprehensive.py${NC}"
  exit 1
fi

# Run the test script
echo -e "${YELLOW}Running comprehensive MCP server tests with auto-fix capability...${NC}"
if $PYTHON test_mcp_server_comprehensive.py "${TEST_ARGS[@]}"; then
  echo -e "${GREEN}Verification completed successfully!${NC}"
  echo -e "${GREEN}The server is properly configured and all tools are working.${NC}"
  echo -e "${BLUE}Results saved to $OUTPUT_FILE${NC}"
  exit 0
else
  echo -e "${YELLOW}Verification completed with some issues.${NC}"
  echo -e "${YELLOW}Some fixes were applied, but manual intervention may be needed.${NC}"
  echo -e "${BLUE}Check $OUTPUT_FILE for details.${NC}"
  exit 1
fi
