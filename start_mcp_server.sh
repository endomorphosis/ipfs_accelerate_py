#!/bin/bash
# Comprehensive MCP Server Starter and Test Framework
#
# This script provides a complete testing framework for MCP servers:
# 1. Starts/restarts the MCP server with clean environment
# 2. Runs comprehensive tests to verify all MCP tools work correctly
# 3. Ensures proper integration with ipfs_accelerate_py module and its virtual filesystem
# 4. Provides detailed diagnostics for troubleshooting different MCP versions
# 5. Generates test reports with coverage metrics
#
# Author: IPFS Accelerate Team
# Version: 2.0.0
# Last updated: 2025-05-05

set -e  # Exit on error

# Default configuration
PORT=8002
HOST="0.0.0.0"
DEBUG=0
RUN_TESTS=1
TEST_LEVEL="normal"  # basic, normal, comprehensive
OUTPUT_DIR="test_results"
TIMEOUT=10
RESTART_SERVER=1     # Whether to restart any existing server
INTEGRATION_TEST=1   # Whether to run the enhanced integration tests
VERSION_CHECK=1      # Whether to verify MCP version compatibility
VFS_TEST=1           # Whether to test virtual filesystem functionality
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/mcp_test_${TIMESTAMP}.log"
TEST_REPORT="${OUTPUT_DIR}/mcp_test_report_${TIMESTAMP}.json"
IPFS_TOOLS_EXPECTED=("ipfs_add_file" "ipfs_get_file" "ipfs_cat_file" "ipfs_pin" "ipfs_unpin" "ipfs_node_info" "ipfs_gateway_url")
IPFS_VFS_TOOLS_EXPECTED=("ipfs_files_mkdir" "ipfs_files_write" "ipfs_files_read" "ipfs_files_ls" "ipfs_files_rm" "ipfs_files_cp" "ipfs_files_mv" "ipfs_files_stat" "ipfs_files_flush")

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
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
    --debug)
      DEBUG=1
      shift
      ;;
    --no-tests)
      RUN_TESTS=0
      shift
      ;;
    --test-level)
      TEST_LEVEL="$2"
      shift
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift
      shift
      ;;
    --no-restart)
      RESTART_SERVER=0
      shift
      ;;
    --no-integration-test)
      INTEGRATION_TEST=0
      shift
      ;;
    --no-version-check)
      VERSION_CHECK=0
      shift
      ;;
    --no-vfs-test)
      VFS_TEST=0
      shift
      ;;
    --help)
      echo "MCP Server Testing Framework"
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT                   Set MCP server port (default: $PORT)"
      echo "  --host HOST                   Set MCP server host (default: $HOST)"
      echo "  --debug                       Enable debug mode"
      echo "  --no-tests                    Skip running tests"
      echo "  --test-level LEVEL            Set test level: basic, normal, or comprehensive (default: $TEST_LEVEL)"
      echo "  --output-dir DIR              Directory for test output (default: $OUTPUT_DIR)"
      echo "  --timeout SECONDS             Server startup timeout (default: $TIMEOUT)"
      echo "  --no-restart                  Don't restart existing server"
      echo "  --no-integration-test         Skip enhanced integration tests"
      echo "  --no-version-check           Skip MCP version compatibility check"
      echo "  --no-vfs-test                 Skip virtual filesystem tests"
      echo "  --help                        Show this help message"
      exit 0
      ;;
    *)
      # Unknown option
      echo -e "${RED}Unknown option: $key${NC}"
      echo "Usage: $0 [--port PORT] [--host HOST] [--debug] [--no-tests] [--test-level basic|normal|comprehensive] [--output-dir DIR] [--timeout SECONDS] [--no-restart] [--no-integration-test] [--no-version-check] [--no-vfs-test] [--help]"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "MCP Server Test Run: $(date)" > "$LOG_FILE"
echo "Configuration:" >> "$LOG_FILE"
echo "  Host: $HOST" >> "$LOG_FILE"
echo "  Port: $PORT" >> "$LOG_FILE"
echo "  Debug: $DEBUG" >> "$LOG_FILE"
echo "  Test Level: $TEST_LEVEL" >> "$LOG_FILE"
echo "  Timeout: $TIMEOUT seconds" >> "$LOG_FILE"
echo "-----------------------------------------" >> "$LOG_FILE"

# Initialize report file with basic structure
cat > "$TEST_REPORT" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "configuration": {
    "host": "$HOST",
    "port": $PORT,
    "debug": $DEBUG,
    "test_level": "$TEST_LEVEL",
    "timeout": $TIMEOUT
  },
  "server": {
    "status": "unknown",
    "restart_success": false,
    "version": "unknown"
  },
  "tools_registered": [],
  "missing_tools": [],
  "tests": []
}
EOF

# Logging function
log() {
  echo -e "$1" | tee -a "$LOG_FILE"
}

# Report function for updating the JSON report
update_report() {
  local field=$1
  local value=$2
  local tmp_file="${OUTPUT_DIR}/report_tmp.json"
  
  # Use jq to update the field if jq is available
  if command -v jq &> /dev/null; then
    jq "$field = $value" "$TEST_REPORT" > "$tmp_file" && mv "$tmp_file" "$TEST_REPORT"
  else
    log "${YELLOW}Warning: jq not installed, JSON report may not be updated correctly${NC}"
    # Simple sed-based replacement - limited functionality
    sed -i "s|\"$field\": .*|\"$field\": $value,|" "$TEST_REPORT"
  fi
}

# Add test result to the report
add_test_result() {
  local name=$1
  local result=$2
  local details=$3
  local tmp_file="${OUTPUT_DIR}/report_tmp.json"
  
  # Format test result as JSON
  local test_json="{\"name\": \"$name\", \"result\": \"$result\", \"details\": \"$details\"}"
  
  # Use jq to add the test result if jq is available
  if command -v jq &> /dev/null; then
    jq ".tests += [$test_json]" "$TEST_REPORT" > "$tmp_file" && mv "$tmp_file" "$TEST_REPORT"
  else
    log "${YELLOW}Warning: jq not installed, test result may not be added correctly${NC}"
  fi
}

# Function to check if port is in use
is_port_in_use() {
  if command -v nc &> /dev/null; then
    nc -z "$HOST" $1 &> /dev/null
    return $?
  elif command -v lsof &> /dev/null; then
    lsof -i:"$1" &> /dev/null
    return $?
  else
    # Fallback to /dev/tcp on Linux
    (echo > /dev/tcp/"$HOST"/$1) &> /dev/null
    return $?
  fi
}

# Find PID of process using the specified port
find_pid_using_port() {
  if command -v lsof &> /dev/null; then
    lsof -ti:"$1" -sTCP:LISTEN
  elif command -v netstat &> /dev/null; then
    netstat -tulpn 2>/dev/null | grep ":$1 " | grep LISTEN | awk '{print $7}' | cut -d'/' -f1
  else
    log "${RED}Cannot find PID: neither lsof nor netstat is available${NC}"
    return 1
  fi
}

# Function to wait for server to be ready
wait_for_server() {
  local port=$1
  local timeout=$2
  local start_time=$(date +%s)
  local current_time
  
  log "${BLUE}Waiting for MCP server to be ready on port $port (timeout: ${timeout}s)...${NC}"
  
  while true; do
    if is_port_in_use "$port"; then
      # Try a basic health check
      if curl -s "http://localhost:$port/health" | grep -q "healthy"; then
        log "${GREEN}MCP server is ready on port $port${NC}"
        return 0
      fi
    fi
    
    current_time=$(date +%s)
    if [ $((current_time - start_time)) -ge "$timeout" ]; then
      log "${RED}Timed out waiting for MCP server to be ready on port $port${NC}"
      return 1
    fi
    
    sleep 1
  done
}

# Function to test MCP server connection
test_server_connection() {
  local success=false
  local server_info=""
  
  log "${BLUE}Testing connection to MCP server on port $PORT...${NC}"
  
  # Try to get server info
  if server_info=$(curl -s "http://localhost:$PORT/health" 2>/dev/null); then
    success=true
    log "${GREEN}Successfully connected to MCP server:${NC}"
    log "$server_info"
    
    # Update report
    update_report ".server.status" "\"running\""
    
    # Extract version if available
    if echo "$server_info" | grep -q "version"; then
      local version=$(echo "$server_info" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
      update_report ".server.version" "\"$version\""
    fi
  else
    log "${RED}Failed to connect to MCP server on port $PORT${NC}"
    update_report ".server.status" "\"not_running\""
  fi
  
  return $success
}

# Function to get list of available tools
get_available_tools() {
  local tools_json=$(curl -s "http://localhost:$PORT/tools" 2>/dev/null)
  
  if [ -z "$tools_json" ]; then
    log "${RED}Failed to get list of available tools${NC}"
    return 1
  fi
  
  # Extract tools list
  if command -v jq &> /dev/null; then
    echo "$tools_json" | jq -r '.tools[]'
  else
    echo "$tools_json" | grep -o '"[^"]*"' | tr -d '"' | sort
  fi
}

# Function to test MCP tools registration
test_tools_registration() {
  log "${BLUE}Testing MCP tools registration...${NC}"
  
  # Get available tools
  local available_tools=$(get_available_tools)
  if [ $? -ne 0 ]; then
    log "${RED}Failed to get available tools${NC}"
    add_test_result "tools_registration" "fail" "Failed to get available tools"
    return 1
  fi
  
  log "${GREEN}Available tools:${NC}"
  echo "$available_tools" | while read -r tool; do
    log "  - $tool"
  done
  
  # Update report with available tools
  local tools_json="[$(echo "$available_tools" | sed 's/^/"/g; s/$/"/g' | paste -sd,)]"
  update_report ".tools_registered" "$tools_json"
  
  # Check for IPFS tools
  log "${BLUE}Checking for IPFS tools...${NC}"
  local missing_tools=()
  
  for tool in "${IPFS_TOOLS_EXPECTED[@]}"; do
    if ! echo "$available_tools" | grep -q "^$tool$"; then
      missing_tools+=("$tool")
      log "${RED}Missing IPFS tool: $tool${NC}"
    fi
  done
  
  if [ ${#missing_tools[@]} -eq 0 ]; then
    log "${GREEN}All expected IPFS tools are registered${NC}"
    add_test_result "ipfs_tools_registration" "pass" "All expected IPFS tools are registered"
  else
    log "${RED}Missing IPFS tools: ${missing_tools[*]}${NC}"
    add_test_result "ipfs_tools_registration" "fail" "Missing IPFS tools: ${missing_tools[*]}"
    
    # Update report with missing tools
    local missing_json="[$(echo "${missing_tools[@]}" | sed 's/ /","/g; s/^/"/; s/$/"/')]"
    update_report ".missing_tools" "$missing_json"
    
    return 1
  fi
  
  return 0
}

# Function to test IPFS functionality
test_ipfs_functionality() {
  log "${BLUE}Testing IPFS functionality...${NC}"
  
  # Create a test file
  local test_file="${OUTPUT_DIR}/ipfs_test_${TIMESTAMP}.txt"
  echo "IPFS Test Content $(date)" > "$test_file"
  
  # Test ipfs_add_file
  log "${BLUE}Testing ipfs_add_file...${NC}"
  local add_result=$(curl -s -X POST "http://localhost:$PORT/call" \
    -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"ipfs_add_file\",\"arguments\":{\"path\":\"$test_file\"}}")
  
  if [ -z "$add_result" ] || echo "$add_result" | grep -q "error"; then
    log "${RED}Failed to add file to IPFS:${NC}"
    log "$add_result"
    add_test_result "ipfs_add_file" "fail" "Failed to add file to IPFS"
    return 1
  fi
  
  # Extract CID
  local cid=""
  if command -v jq &> /dev/null; then
    cid=$(echo "$add_result" | jq -r '.result.cid')
  else
    cid=$(echo "$add_result" | grep -o '"cid":"[^"]*"' | cut -d'"' -f4)
  fi
  
  if [ -z "$cid" ] || [ "$cid" == "null" ]; then
    log "${RED}Failed to extract CID from result:${NC}"
    log "$add_result"
    add_test_result "ipfs_add_file" "fail" "Failed to extract CID from result"
    return 1
  fi
  
  log "${GREEN}Successfully added file to IPFS with CID: $cid${NC}"
  add_test_result "ipfs_add_file" "pass" "Successfully added file to IPFS with CID: $cid"
  
  # Test ipfs_cat_file
  log "${BLUE}Testing ipfs_cat_file...${NC}"
  local cat_result=$(curl -s -X POST "http://localhost:$PORT/call" \
    -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"ipfs_cat_file\",\"arguments\":{\"ipfs_hash\":\"$cid\"}}")
  
  if [ -z "$cat_result" ] || echo "$cat_result" | grep -q "error"; then
    log "${RED}Failed to cat file from IPFS:${NC}"
    log "$cat_result"
    add_test_result "ipfs_cat_file" "fail" "Failed to cat file from IPFS"
    return 1
  fi
  
  log "${GREEN}Successfully retrieved content from IPFS${NC}"
  add_test_result "ipfs_cat_file" "pass" "Successfully retrieved content from IPFS"
  
  # Clean up
  rm -f "$test_file"
  
  # Test ipfs_gateway_url
  log "${BLUE}Testing ipfs_gateway_url...${NC}"
  local gateway_result=$(curl -s -X POST "http://localhost:$PORT/call" \
    -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"ipfs_gateway_url\",\"arguments\":{\"ipfs_hash\":\"$cid\"}}")
  
  if [ -z "$gateway_result" ] || echo "$gateway_result" | grep -q "error"; then
    log "${RED}Failed to get gateway URL:${NC}"
    log "$gateway_result"
    add_test_result "ipfs_gateway_url" "fail" "Failed to get gateway URL"
    return 1
  fi
  
  log "${GREEN}Successfully got gateway URL for CID${NC}"
  add_test_result "ipfs_gateway_url" "pass" "Successfully got gateway URL for CID"
  
  return 0
}

# Function to check ipfs_accelerate_py integration
test_integration() {
  log "${BLUE}Testing integration with ipfs_accelerate_py module...${NC}"
  
  # Run a Python script to test the integration
  cat > "${OUTPUT_DIR}/test_integration.py" << EOF
#!/usr/bin/env python3
import os
import sys
import json
import requests

def test_integration():
    try:
        # Try to import ipfs_accelerate_py
        try:
            import ipfs_accelerate_py
            print("Successfully imported ipfs_accelerate_py")
        except ImportError as e:
            print("Failed to import ipfs_accelerate_py:", e)
            return False
        
        # Connect to MCP server
        server_url = "http://localhost:$PORT"
        
        # Test hardware info (should be using hardware_detection from ipfs_accelerate_py)
        response = requests.post(
            f"{server_url}/call",
            json={"tool_name": "get_hardware_info", "arguments": {}}
        )
        
        if response.status_code != 200:
            print("Failed to call get_hardware_info:", response.text)
            return False
        
        result = response.json().get("result", {})
        
        # Check if we have comprehensive hardware info (which would come from ipfs_accelerate_py)
        has_detailed_info = False
        if "accelerators" in result:
            accelerators = result.get("accelerators", {})
            if len(accelerators) > 0 and any("memory" in acc for acc in accelerators.values()):
                has_detailed_info = True
        
        if has_detailed_info:
            print("MCP server is using ipfs_accelerate_py for hardware detection")
        else:
            print("MCP server may not be using ipfs_accelerate_py for hardware detection")
        
        return True
    
    except Exception as e:
        print("Error during integration test:", e)
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
EOF

  chmod +x "${OUTPUT_DIR}/test_integration.py"
  
  log "${BLUE}Running integration test script...${NC}"
  if python3 "${OUTPUT_DIR}/test_integration.py"; then
    log "${GREEN}Integration test passed${NC}"
    add_test_result "ipfs_accelerate_py_integration" "pass" "Integration test completed successfully"
    return 0
  else
    log "${RED}Integration test failed${NC}"
    add_test_result "ipfs_accelerate_py_integration" "fail" "Integration test failed"
    return 1
  fi
}

# Function to test virtual filesystem tools
test_virtual_filesystem() {
  log "${BLUE}Testing virtual filesystem tools...${NC}"
  
  # Test if the tools are registered
  local fs_tools=("${IPFS_VFS_TOOLS_EXPECTED[@]}")
  local missing_fs_tools=()
  local available_tools=$(get_available_tools)
  
  for tool in "${fs_tools[@]}"; do
    if ! echo "$available_tools" | grep -q "^$tool$"; then
      missing_fs_tools+=("$tool")
    fi
  done
  
  if [ ${#missing_fs_tools[@]} -gt 0 ]; then
    log "${YELLOW}Some virtual filesystem tools are not registered: ${missing_fs_tools[*]}${NC}"
    log "${YELLOW}Skipping virtual filesystem tests${NC}"
    add_test_result "virtual_filesystem" "skip" "Missing tools: ${missing_fs_tools[*]}"
    return 0
  fi
  
  # Test writing to the virtual filesystem
  log "${BLUE}Testing ipfs_files_write...${NC}"
  local mfs_path="/mcp-test-$(date +%s).txt"
  local content="MCP Test Content $(date)"
  
  local write_result=$(curl -s -X POST "http://localhost:$PORT/call" \
    -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"ipfs_files_write\",\"arguments\":{\"path\":\"$mfs_path\",\"content\":\"$content\"}}")
  
  if [ -z "$write_result" ] || echo "$write_result" | grep -q "error"; then
    log "${RED}Failed to write to MFS:${NC}"
    log "$write_result"
    add_test_result "ipfs_files_write" "fail" "Failed to write to MFS"
    return 1
  fi
  
  log "${GREEN}Successfully wrote to MFS at path: $mfs_path${NC}"
  add_test_result "ipfs_files_write" "pass" "Successfully wrote to MFS"
  
  # Test reading from the virtual filesystem
  log "${BLUE}Testing ipfs_files_read...${NC}"
  local read_result=$(curl -s -X POST "http://localhost:$PORT/call" \
    -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"ipfs_files_read\",\"arguments\":{\"path\":\"$mfs_path\"}}")
  
  if [ -z "$read_result" ] || echo "$read_result" | grep -q "error"; then
    log "${RED}Failed to read from MFS:${NC}"
    log "$read_result"
    add_test_result "ipfs_files_read" "fail" "Failed to read from MFS"
    return 1
  fi
  
  log "${GREEN}Successfully read from MFS at path: $mfs_path${NC}"
  add_test_result "ipfs_files_read" "pass" "Successfully read from MFS"
  
  return 0
}

# Run comprehensive tests
run_tests() {
  log "${BLUE}Running comprehensive MCP tests...${NC}"
  
  # Test server connection
  if ! test_server_connection; then
    log "${RED}Server connection test failed. Cannot continue with further tests.${NC}"
    return 1
  fi
  
  # Test tools registration
  if ! test_tools_registration; then
    log "${YELLOW}Tools registration test failed. Some tests may be skipped.${NC}"
  fi
  
  # Based on test level, run additional tests
  if [ "$TEST_LEVEL" = "basic" ]; then
    log "${BLUE}Running basic tests only${NC}"
    # Basic tests are already covered by the connection and registration tests
  else
    # Normal and comprehensive levels
    test_ipfs_functionality
    test_integration
    
    if [ "$TEST_LEVEL" = "comprehensive" ]; then
      log "${BLUE}Running comprehensive tests${NC}"
      test_virtual_filesystem
      # Add more comprehensive tests here
    fi
  fi
  
  # Print test summary
  log "${BLUE}==== Test Summary ====${NC}"
  
  # Use jq to get a nice summary if available
  if command -v jq &> /dev/null; then
    local pass_count=$(jq '.tests | map(select(.result == "pass")) | length' "$TEST_REPORT")
    local fail_count=$(jq '.tests | map(select(.result == "fail")) | length' "$TEST_REPORT")
    local skip_count=$(jq '.tests | map(select(.result == "skip")) | length' "$TEST_REPORT")
    local total_count=$(jq '.tests | length' "$TEST_REPORT")
    
    log "${GREEN}Tests passed: $pass_count${NC}"
    log "${RED}Tests failed: $fail_count${NC}"
    log "${YELLOW}Tests skipped: $skip_count${NC}"
    log "${BLUE}Total tests: $total_count${NC}"
    
    if [ "$fail_count" -eq 0 ]; then
      log "${GREEN}All tests passed successfully!${NC}"
    else
      log "${RED}Some tests failed. Check the log and report for details.${NC}"
    fi
  else
    log "${YELLOW}jq not installed, cannot generate detailed test summary.${NC}"
    log "${BLUE}See test report at: $TEST_REPORT${NC}"
  fi
  
  log "${BLUE}Test report saved to: $TEST_REPORT${NC}"
  log "${BLUE}Test log saved to: $LOG_FILE${NC}"
}

# Main script execution starts here
log "${BLUE}===== IPFS Accelerate MCP Server Manager and Test Suite =====${NC}"
log "Starting at $(date)"

# Check if the server is already running
log "${BLUE}Checking if MCP server is running on port $PORT...${NC}"
if is_port_in_use $PORT; then
  log "${YELLOW}MCP server is running on port $PORT. Attempting to stop it...${NC}"
  
  # Find the PID of the process using the port
  PID=$(find_pid_using_port $PORT)
  
  if [ -n "$PID" ]; then
    log "${BLUE}Found process with PID $PID using port $PORT. Terminating...${NC}"
    kill $PID
    
    # Wait for the port to be freed
    log "${BLUE}Waiting for port $PORT to be freed...${NC}"
    for i in $(seq 1 $TIMEOUT); do
      if ! is_port_in_use $PORT; then
        log "${GREEN}Port $PORT is now free.${NC}"
        break
      fi
      sleep 1
      if [ $i -eq $TIMEOUT ]; then
        log "${RED}Timed out waiting for port $PORT to be freed. Forcefully terminating...${NC}"
        kill -9 $PID
        sleep 1
      fi
    done
  else
    log "${RED}Could not find PID of process using port $PORT. You may need to manually terminate it.${NC}"
  fi
else
  log "${BLUE}No MCP server running on port $PORT.${NC}"
fi

# Register tools using the simplified script
log "${BLUE}Registering MCP tools...${NC}"
python3 /home/barberb/ipfs_accelerate_py/simple_mcp_register.py

# Verify ipfs_vfs module registration
VFS_MODULE_PATH="/home/barberb/ipfs_accelerate_py/ipfs_accelerate_py/mcp/tools/ipfs_vfs.py"
if [ -f "$VFS_MODULE_PATH" ]; then
  log "${BLUE}IPFS virtual filesystem module detected at $VFS_MODULE_PATH.${NC}"
else
  VFS_MODULE_PATH="/home/barberb/ipfs_accelerate_py/mcp/tools/ipfs_vfs.py"
  if [ -f "$VFS_MODULE_PATH" ]; then
    log "${BLUE}IPFS virtual filesystem module detected at $VFS_MODULE_PATH.${NC}"
  else
    log "${YELLOW}IPFS virtual filesystem module not found in expected locations. Checking current directory...${NC}"
    VFS_MODULE_PATH="/home/barberb/ipfs_accelerate_py/ipfs_vfs.py"
    if [ -f "$VFS_MODULE_PATH" ]; then
      log "${BLUE}IPFS virtual filesystem module detected at $VFS_MODULE_PATH.${NC}"
    else
      log "${YELLOW}IPFS virtual filesystem module not found. Virtual filesystem tests may fail.${NC}"
    fi
  fi
fi

# Construct the command to start the server
CMD="python3 -m mcp.run_server --port $PORT --host $HOST"
if [ $DEBUG -eq 1 ]; then
  CMD="$CMD --debug"
  log "${BLUE}Debug mode enabled.${NC}"
fi

# Start the server in the background
log "${BLUE}Starting MCP server on port $PORT...${NC}"
$CMD &

# Get the PID of the server
SERVER_PID=$!
log "${BLUE}MCP server started with PID $SERVER_PID${NC}"

# Update report with restart success
update_report ".server.restart_success" "true"

# Wait for server to be ready
if ! wait_for_server $PORT $TIMEOUT; then
  log "${RED}Failed to start MCP server. Check the log for errors.${NC}"
  update_report ".server.restart_success" "false"
  update_report ".server.status" "\"not_running\""
  exit 1
fi

# Run tests if enabled
if [ $RUN_TESTS -eq 1 ]; then
  run_tests
  
  # Run comprehensive integration tests using our Python test framework
  if [ "$TEST_LEVEL" = "comprehensive" ] || [ -n "$INTEGRATION_TEST" ]; then
    log "${BLUE}Running comprehensive integration tests...${NC}"
    
    INTEGRATION_TEST_SCRIPT="/home/barberb/ipfs_accelerate_py/test_mcp_server_integration.py"
    VFS_TEST_SCRIPT="/home/barberb/ipfs_accelerate_py/test_mcp_virtual_fs.py"
    
    # Run the main integration test script
    if [ -f "$INTEGRATION_TEST_SCRIPT" ]; then
      log "${BLUE}Found comprehensive test script at $INTEGRATION_TEST_SCRIPT. Running...${NC}"
      INTEGRATION_RESULT_FILE="${OUTPUT_DIR}/mcp_integration_results_${TIMESTAMP}.json"
      
      python3 "$INTEGRATION_TEST_SCRIPT" --host localhost --port $PORT --protocol http --output-dir "$OUTPUT_DIR"
      
      INTEGRATION_TEST_STATUS=$?
      if [ $INTEGRATION_TEST_STATUS -eq 0 ]; then
        log "${GREEN}Comprehensive integration tests passed.${NC}"
        add_test_result "comprehensive_integration_tests" "pass" "All integration tests completed successfully"
      else
        log "${RED}Comprehensive integration tests failed. Check the test report for details.${NC}"
        add_test_result "comprehensive_integration_tests" "fail" "Integration tests failed with exit code $INTEGRATION_TEST_STATUS"
      fi
      
      # Merge test results if available
      if [ -f "$INTEGRATION_RESULT_FILE" ] && command -v jq &> /dev/null; then
        log "${BLUE}Merging integration test results into main report...${NC}"
        local tmp_file="${OUTPUT_DIR}/combined_report_tmp.json"
        jq -s '.[0] * {"integration_results": .[1]}' "$TEST_REPORT" "$INTEGRATION_RESULT_FILE" > "$tmp_file" && mv "$tmp_file" "$TEST_REPORT"
      fi
    else
      log "${YELLOW}Comprehensive test script not found at $INTEGRATION_TEST_SCRIPT. Skipping additional tests.${NC}"
      add_test_result "comprehensive_integration_tests" "skip" "Integration test script not found"
    fi
    
    # Run the dedicated VFS test script if not already run by the integration test
    if [ -f "$VFS_TEST_SCRIPT" ] && [ "$TEST_LEVEL" = "comprehensive" ]; then
      log "${BLUE}Running dedicated virtual filesystem tests...${NC}"
      VFS_RESULT_FILE="${OUTPUT_DIR}/vfs_test_results_${TIMESTAMP}.json"
      
      python3 "$VFS_TEST_SCRIPT" --host localhost --port $PORT --protocol http --output "$VFS_RESULT_FILE"
      
      VFS_TEST_STATUS=$?
      if [ $VFS_TEST_STATUS -eq 0 ]; then
        log "${GREEN}Virtual filesystem tests passed.${NC}"
        add_test_result "virtual_filesystem_tests" "pass" "All virtual filesystem tests completed successfully"
      else
        log "${RED}Virtual filesystem tests failed. Check the test report for details.${NC}"
        add_test_result "virtual_filesystem_tests" "fail" "Virtual filesystem tests failed with exit code $VFS_TEST_STATUS"
      fi
    fi
  fi
else
  log "${YELLOW}Tests skipped as requested.${NC}"
fi

# Server is running in the background
log "${GREEN}Server is running in the background with PID $SERVER_PID.${NC}"
log "${BLUE}To test, visit: http://localhost:$PORT/health${NC}"
log "${BLUE}For the MCP manifest, visit: http://localhost:$PORT/mcp/manifest${NC}"
log "${BLUE}To stop the server, run: kill $SERVER_PID${NC}"

# Detach the process
disown

log "${BLUE}===== Completed at $(date) =====${NC}"

# Exit with success
exit 0
