#!/bin/bash
# High Availability Cluster Example Script
#
# This script provides a convenient way to run the high availability cluster example
# with various configurations.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  High Availability Cluster Example  ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
NODES=3
BASE_PORT=8080
RUNTIME=120  # 2 minutes
FAULT_INJECTION=0
DEBUG=0

# Function to show help
show_help() {
  echo -e "${GREEN}Usage: $0 [options]${NC}"
  echo ""
  echo -e "${CYAN}Options:${NC}"
  echo "  --nodes N            Number of coordinator nodes to start (default: 3)"
  echo "  --base-port PORT     Base port number (default: 8080)"
  echo "  --runtime SECONDS    Runtime in seconds (default: 120)"
  echo "  --fault-injection    Enable fault injection to test automatic failover"
  echo "  --debug              Enable debug output"
  echo "  --help               Show this help message"
  echo ""
  echo -e "${CYAN}Examples:${NC}"
  echo "  $0 --nodes 3 --fault-injection"
  echo "  $0 --nodes 5 --runtime 300"
  echo "  $0 --base-port 9000 --nodes 2"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --runtime)
      RUNTIME="$2"
      shift 2
      ;;
    --fault-injection)
      FAULT_INJECTION=1
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create command
CMD="python duckdb_api/distributed_testing/examples/high_availability_cluster.py"
CMD+=" --nodes $NODES"
CMD+=" --base-port $BASE_PORT"
CMD+=" --runtime $RUNTIME"

if [ $FAULT_INJECTION -eq 1 ]; then
  CMD+=" --fault-injection"
fi

# Create log file
LOG_FILE="ha_cluster_$(date +%Y%m%d_%H%M%S).log"

# Print configuration
echo -e "${YELLOW}High Availability Cluster Configuration:${NC}"
echo -e "  Nodes:           ${GREEN}$NODES${NC}"
echo -e "  Base Port:       ${GREEN}$BASE_PORT${NC}"
echo -e "  Runtime:         ${GREEN}${RUNTIME}s${NC}"
echo -e "  Fault Injection: ${GREEN}$([ $FAULT_INJECTION -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Log File:        ${GREEN}$LOG_FILE${NC}"
echo ""

# Confirm execution
echo -e "${YELLOW}Starting High Availability Cluster...${NC}"
echo -e "Press Ctrl+C to stop all nodes"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run the command
if [ $DEBUG -eq 1 ]; then
  echo "$CMD"
fi

# Run in foreground and log to file
if [ $DEBUG -eq 1 ]; then
  $CMD 2>&1 | tee "$LOG_FILE"
else
  $CMD 2>&1 | tee "$LOG_FILE" | grep -v "DEBUG"
fi

# Capture exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "${GREEN}Cluster shutdown successfully${NC}"
else
  echo -e "${RED}Cluster exited with error code: $EXIT_STATUS${NC}"
fi

echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "${YELLOW}Log saved to: ${GREEN}$LOG_FILE${NC}"
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_STATUS