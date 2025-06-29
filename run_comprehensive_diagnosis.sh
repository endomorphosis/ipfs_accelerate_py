#!/bin/bash
#
# IPFS Accelerate MCP - Run Comprehensive Diagnosis
#
# This script runs a comprehensive diagnostic test, identifies problems,
# and suggests solutions for the IPFS Accelerate MCP server.
#

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - Comprehensive Diagnosis        ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Default values
HOST="127.0.0.1"
PORT=8002
OUTPUT="mcp_diagnostics.json"
TIMEOUT=5
VERBOSE=false

# Process command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --host)
      HOST="$2"
      shift
      shift
      ;;
    --port)
      PORT="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--host HOST] [--port PORT] [--output FILE] [--timeout SECONDS] [--verbose]"
      exit 1
      ;;
  esac
done

# Determine Python executable
if command -v python3 &>/dev/null; then
  PYTHON="python3"
elif command -v python &>/dev/null; then
  PYTHON="python"
else
  echo -e "${RED}Python not found. Please install Python 3.8 or newer.${NC}"
  exit 1
fi

# Check if the test script exists
if [[ ! -f "test_mcp_server_comprehensive.py" ]]; then
  echo -e "${RED}Test script not found: test_mcp_server_comprehensive.py${NC}"
  echo "Please run this script from the IPFS Accelerate project directory."
  exit 1
fi

# Run the diagnostic test
echo -e "${YELLOW}Running MCP server diagnostics...${NC}"
echo "Server: $HOST:$PORT"

# Execute the test script without auto-fix
$PYTHON test_mcp_server_comprehensive.py --host "$HOST" --port "$PORT" --output "$OUTPUT" --timeout "$TIMEOUT"
TEST_STATUS=$?

# Parse the diagnostics file
echo -e "\n${YELLOW}Analyzing diagnostic results...${NC}"

if [[ ! -f "$OUTPUT" ]]; then
  echo -e "${RED}Error: Diagnostic output file not found: $OUTPUT${NC}"
  exit 1
fi

# Extract key information using jq if available, otherwise use grep
if command -v jq &>/dev/null; then
  SERVER_REACHABLE=$(jq -r '.server.reachable' "$OUTPUT")
  AVAILABLE_TOOLS=$(jq -r '.tools.available | length' "$OUTPUT")
  AVAILABLE_TOOLS_LIST=$(jq -r '.tools.available | join(", ")' "$OUTPUT")
  
  # Create a diagnostic summary
  echo -e "\n${BLUE}DIAGNOSTIC SUMMARY:${NC}"
  echo -e "${BLUE}--------------------------------------------------${NC}"
  echo -e "Server Reachable: $(if [[ "$SERVER_REACHABLE" == "true" ]]; then echo -e "${GREEN}YES${NC}"; else echo -e "${RED}NO${NC}"; fi)"
  echo -e "Available Tools: $AVAILABLE_TOOLS ($AVAILABLE_TOOLS_LIST)"
  
  echo -e "\n${BLUE}TOOL STATUS:${NC}"
  echo -e "${BLUE}--------------------------------------------------${NC}"
  
  # Check each required tool
  jq -r '.tools.required_tools_working | to_entries[] | .key + " = " + (.value | tostring)' "$OUTPUT" | while read -r line; do
    TOOL_NAME="${line%% =*}"
    TOOL_STATUS="${line##*= }"
    
    if [[ "$TOOL_STATUS" == "true" ]]; then
      echo -e "${GREEN}✓ $TOOL_NAME${NC}"
    else
      echo -e "${RED}✗ $TOOL_NAME${NC}"
    fi
  done
else
  # Fallback to grep if jq is not available
  echo -e "${YELLOW}Note: Install 'jq' for better diagnostics output${NC}"
  
  if grep -q '"reachable": true' "$OUTPUT"; then
    echo -e "Server Reachable: ${GREEN}YES${NC}"
  else
    echo -e "Server Reachable: ${RED}NO${NC}"
    echo -e "${RED}Cannot proceed with further diagnostics. Server is not reachable.${NC}"
  fi
  
  echo -e "\n${BLUE}TOOL STATUS:${NC}"
  echo -e "${BLUE}--------------------------------------------------${NC}"
  
  # Check get_hardware_info specifically
  if grep -q '"get_hardware_info": true' "$OUTPUT"; then
    echo -e "${GREEN}✓ get_hardware_info${NC}"
  else
    echo -e "${RED}✗ get_hardware_info${NC}"
  fi
  
  # Check other tools
  for TOOL in health_check ipfs_add_file ipfs_cat ipfs_files_write ipfs_files_read list_models create_endpoint run_inference; do
    if grep -q "\"$TOOL\": true" "$OUTPUT"; then
      echo -e "${GREEN}✓ $TOOL${NC}"
    else
      echo -e "${RED}✗ $TOOL${NC}"
    fi
  done
fi

# Provide recommendations based on diagnostic results
echo -e "\n${BLUE}RECOMMENDATIONS:${NC}"
echo -e "${BLUE}--------------------------------------------------${NC}"

if [[ $TEST_STATUS -ne 0 ]]; then
  echo -e "${YELLOW}Issues detected with the MCP server. Follow these steps to fix:${NC}"
  echo ""
  echo -e "1. ${YELLOW}Run the auto-fix script to address common issues:${NC}"
  echo "   ./fix_and_verify_mcp_server.sh --auto-fix"
  echo ""
  echo -e "2. ${YELLOW}If issues persist, check server logs for errors:${NC}"
  echo "   tail -50 final_mcp_server.log"
  echo ""
  echo -e "3. ${YELLOW}Verify tool registration in final_mcp_server.py${NC}"
  echo -e "4. ${YELLOW}Check for API endpoint compatibility in mcp/server.py${NC}"
  echo -e "5. ${YELLOW}Restart the server after making changes:${NC}"
  echo "   ./restart_mcp_server.sh"
  echo ""
else
  echo -e "${GREEN}MCP server appears to be functioning correctly!${NC}"
  echo -e "To view more detailed hardware information, access the get_hardware_info tool directly:"
  echo "curl -X POST http://$HOST:$PORT/mcp/tool/get_hardware_info"
fi

# Print final status and exit
if [[ $TEST_STATUS -eq 0 ]]; then
  echo -e "\n${GREEN}Diagnostics completed successfully!${NC}"
  exit 0
else
  echo -e "\n${YELLOW}Diagnostics completed with issues. See recommendations above.${NC}"
  exit $TEST_STATUS
fi