#!/bin/bash
#
# IPFS Accelerate MCP - Fix and Verify Script
#
# This script runs the comprehensive test script with auto-fixing capabilities
# to ensure all required tools are properly registered and accessible.
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - Fix and Verify                 ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Default settings
HOST="127.0.0.1"
PORT="8002"
TIMEOUT=5
START_SERVER=false
NO_RESTART=false
OUTPUT_FILE="mcp_test_results.json"

# Process command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      PORT="$2"
      shift
      shift
      ;;
    --host)
      HOST="$2"
      shift
      shift
      ;;
    --start-server)
      START_SERVER=true
      shift
      ;;
    --no-restart)
      NO_RESTART=true
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--port PORT] [--host HOST] [--start-server] [--no-restart] [--output FILE] [--timeout SECONDS]"
      exit 1
      ;;
  esac
done

# Check if the comprehensive test script exists
if [[ ! -f "test_mcp_server_comprehensive.py" ]]; then
  echo -e "${RED}Error: test_mcp_server_comprehensive.py not found${NC}"
  echo "Make sure you are running this script from the IPFS Accelerate project directory."
  exit 1
fi

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if command -v python3 &>/dev/null; then
  PYTHON="python3"
elif command -v python &>/dev/null; then
  PYTHON="python"
else
  echo -e "${RED}Python not found. Please install Python 3.8 or newer.${NC}"
  exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Prepare the command
CMD="$PYTHON test_mcp_server_comprehensive.py --host $HOST --port $PORT --timeout $TIMEOUT --output $OUTPUT_FILE --auto-fix"

# Add start-server flag if requested
if [[ "$START_SERVER" == "true" ]]; then
  CMD="$CMD --start-server"
fi

# Add no-restart flag if requested
if [[ "$NO_RESTART" == "true" ]]; then
  CMD="$CMD --no-restart"
fi

# Run the test and auto-fix script
echo -e "${YELLOW}Running comprehensive MCP server test with auto-fix capabilities...${NC}"
echo "Command: $CMD"
eval "$CMD"

# Check the exit status
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Success: The MCP server has all required tools working correctly${NC}"
  
  # Check if specific tools are working
  if grep -q '"get_hardware_info": true' "$OUTPUT_FILE"; then
    echo -e "${GREEN}✓ get_hardware_info tool is working correctly${NC}"
  else
    echo -e "${RED}✗ get_hardware_info tool is still not working correctly${NC}"
    echo "  Check the server logs for more details."
  fi
  
  # Check other required tools
  for tool in health_check ipfs_add_file ipfs_cat ipfs_files_write ipfs_files_read list_models create_endpoint run_inference; do
    if grep -q "\"$tool\": true" "$OUTPUT_FILE"; then
      echo -e "${GREEN}✓ $tool tool is working correctly${NC}"
    else
      echo -e "${YELLOW}✗ $tool tool is not working correctly${NC}"
    fi
  done
  
  # If a post-fix results file exists, compare the before/after status
  POST_FIX_FILE="post_fix_${OUTPUT_FILE}"
  if [[ -f "$POST_FIX_FILE" ]]; then
    echo -e "\n${YELLOW}Comparing before and after fix results:${NC}"
    echo -e "${BLUE}--------------------------------------------------${NC}"
    echo "Before:"
    grep -A 20 "required_tools_working" "$OUTPUT_FILE" | head -20
    echo -e "${BLUE}--------------------------------------------------${NC}"
    echo "After:"
    grep -A 20 "required_tools_working" "$POST_FIX_FILE" | head -20
    echo -e "${BLUE}--------------------------------------------------${NC}"
  fi
  
  # Final status
  echo -e "\n${GREEN}MCP Server verification complete!${NC}"
  exit 0
else
  echo -e "${RED}Error: The MCP server still has issues with required tools${NC}"
  echo -e "Check ${YELLOW}$OUTPUT_FILE${NC} for detailed test results."
  echo -e "You can try running this script again with --start-server to restart the server."
  exit 1
fi
