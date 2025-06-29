#!/bin/bash
#
# IPFS Accelerate MCP - One-step Verification Script
#
# This script is a one-step verification tool for the IPFS Accelerate MCP server.
# It will:
# 1. Check server status
# 2. Apply fixes if needed
# 3. Generate a comprehensive report
#

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}    IPFS Accelerate MCP - One-step Verification          ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Default settings
HOST="127.0.0.1"
PORT="8002"
REPORT_FILE="ipfs_accelerate_mcp_verification_report.md"
SKIP_FIX=false
START_SERVER=false

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
    --report)
      REPORT_FILE="$2"
      shift
      shift
      ;;
    --skip-fix)
      SKIP_FIX=true
      shift
      ;;
    --start-server)
      START_SERVER=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--port PORT] [--host HOST] [--report FILE] [--skip-fix] [--start-server]"
      exit 1
      ;;
  esac
done

# Check Python
if command -v python3 &>/dev/null; then
  PYTHON="python3"
elif command -v python &>/dev/null; then
  PYTHON="python"
else
  echo -e "${RED}Python not found. Please install Python 3.8 or newer.${NC}"
  exit 1
fi

# Check if server is running
echo -e "${YELLOW}Checking for existing MCP server process on port $PORT...${NC}"
if pgrep -f "final_mcp_server.py.*$PORT" > /dev/null || pgrep -f "uvicorn.*$PORT" > /dev/null; then
    echo -e "${GREEN}MCP server is running on port $PORT${NC}"
    SERVER_RUNNING=true
else
    echo -e "${YELLOW}No MCP server detected on port $PORT${NC}"
    SERVER_RUNNING=false
    
    if [[ "$START_SERVER" == "true" ]]; then
        echo -e "${YELLOW}Starting MCP server as requested...${NC}"
        if [[ -f "run_final_solution.sh" ]]; then
            bash run_final_solution.sh --port "$PORT" --host "$HOST" &
            # Give it time to start
            echo "Waiting for server to start..."
            sleep 5
            SERVER_RUNNING=true
        else
            echo -e "${RED}Error: run_final_solution.sh not found${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Use --start-server to automatically start the server${NC}"
    fi
fi

# Run comprehensive test to detect issues
if [[ "$SERVER_RUNNING" == "true" ]]; then
    echo -e "${YELLOW}Running comprehensive MCP server test...${NC}"
    TEST_OUTPUT="mcp_verification_test.json"
    $PYTHON test_mcp_server_comprehensive.py --host "$HOST" --port "$PORT" --output "$TEST_OUTPUT"
    TEST_STATUS=$?
    
    # Apply fixes if needed
    if [[ "$TEST_STATUS" -ne 0 && "$SKIP_FIX" != "true" ]]; then
        echo -e "${YELLOW}Issues detected with MCP server, applying fixes...${NC}"
        
        if [[ -f "fix_and_verify_mcp_server.sh" ]]; then
            bash fix_and_verify_mcp_server.sh --host "$HOST" --port "$PORT" --auto-fix
            FIX_STATUS=$?
            
            if [[ "$FIX_STATUS" -eq 0 ]]; then
                echo -e "${GREEN}Fixes successfully applied${NC}"
            else
                echo -e "${YELLOW}Some issues could not be automatically fixed${NC}"
            fi
        else
            echo -e "${YELLOW}fix_and_verify_mcp_server.sh not found, skipping auto-fix${NC}"
        fi
    fi
    
    # Generate detailed verification report
    echo -e "${YELLOW}Generating verification report...${NC}"
    $PYTHON generate_mcp_verification_report.py --host "$HOST" --port "$PORT" --output "$REPORT_FILE"
    REPORT_STATUS=$?
    
    if [[ "$REPORT_STATUS" -eq 0 ]]; then
        echo -e "${GREEN}Verification complete! MCP server is fully operational with all required tools.${NC}"
        echo -e "${GREEN}Report generated: $REPORT_FILE${NC}"
        
        # Display hardware info summary
        if command -v jq &>/dev/null; then
            HARDWARE_INFO_FILE="hardware_info_report.json"
            curl -s -X POST "http://$HOST:$PORT/mcp/tool/get_hardware_info" > "$HARDWARE_INFO_FILE"
            
            echo -e "\n${BLUE}Hardware Acceleration Summary:${NC}"
            if [[ -f "$HARDWARE_INFO_FILE" ]]; then
                jq -r '.accelerators | to_entries[] | "\(.key): \(.value.available)"' "$HARDWARE_INFO_FILE" | 
                while read line; do
                    ACCEL_NAME="${line%:*}"
                    ACCEL_AVAIL="${line#*: }"
                    if [[ "$ACCEL_AVAIL" == "true" ]]; then
                        echo -e "${GREEN}✓ $ACCEL_NAME acceleration is available${NC}"
                    else
                        echo -e "${YELLOW}○ $ACCEL_NAME acceleration is not available${NC}"
                    fi
                done
            fi
        fi
        
        exit 0
    else
        echo -e "${YELLOW}Verification completed with some issues. See the report for details.${NC}"
        echo -e "${YELLOW}Report generated: $REPORT_FILE${NC}"
        exit 1
    fi
else
    echo -e "${RED}Cannot proceed with verification. No MCP server running on $HOST:$PORT${NC}"
    exit 1
fi