#!/bin/bash
# run_integrated_system.sh
#
# This script provides a convenient way to run the complete Distributed Testing Framework
# with Coordinator, Load Balancer, and Monitoring Dashboard integration.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Distributed Testing Framework - Integrated System Runner  ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
MOCK_WORKERS=5
DB_PATH="./distributed_testing_db.duckdb"
COORDINATOR_PORT=8080
DASHBOARD_PORT=5000
OPEN_BROWSER=0
STRESS_TEST=0
TEST_TASKS=20
TEST_DURATION=60
TERMINAL_DASHBOARD=0
ENABLE_WORK_STEALING=0
LOG_FILE="integrated_system_$(date +%Y%m%d_%H%M%S).log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mock-workers)
      MOCK_WORKERS="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --coordinator-port)
      COORDINATOR_PORT="$2"
      shift 2
      ;;
    --dashboard-port)
      DASHBOARD_PORT="$2"
      shift 2
      ;;
    --open-browser)
      OPEN_BROWSER=1
      shift
      ;;
    --stress-test)
      STRESS_TEST=1
      shift
      ;;
    --test-tasks)
      TEST_TASKS="$2"
      shift 2
      ;;
    --test-duration)
      TEST_DURATION="$2"
      shift 2
      ;;
    --terminal-dashboard)
      TERMINAL_DASHBOARD=1
      shift
      ;;
    --enable-work-stealing)
      ENABLE_WORK_STEALING=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo ""
      echo "Options:"
      echo "  --mock-workers N       Launch N mock workers (default: 5)"
      echo "  --db-path PATH         Path to database (default: ./distributed_testing_db.duckdb)"
      echo "  --coordinator-port N   Port for coordinator server (default: 8080)"
      echo "  --dashboard-port N     Port for dashboard server (default: 5000)"
      echo "  --open-browser         Open web browser to dashboard automatically"
      echo "  --stress-test          Run stress test after starting"
      echo "  --test-tasks N         Number of tasks for stress test (default: 20)"
      echo "  --test-duration N      Duration of stress test in seconds (default: 60)"
      echo "  --terminal-dashboard   Use terminal-based dashboard instead of web dashboard"
      echo "  --enable-work-stealing Enable work stealing between workers"
      echo "  --log-file FILE        Log output to specified file"
      echo "  --help                 Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --mock-workers 10 --open-browser"
      echo "  $0 --stress-test --test-tasks 50 --test-duration 120"
      echo "  $0 --terminal-dashboard --enable-work-stealing"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create command with options
CMD="python3 -m duckdb_api.distributed_testing.run_coordinator_with_dashboard"
CMD+=" --port $COORDINATOR_PORT"
CMD+=" --dashboard-port $DASHBOARD_PORT"
CMD+=" --db-path $DB_PATH"
CMD+=" --mock-workers $MOCK_WORKERS"

# Add optional arguments
if [ $STRESS_TEST -eq 1 ]; then
  CMD+=" --stress-test --test-tasks $TEST_TASKS --test-duration $TEST_DURATION"
fi

if [ $OPEN_BROWSER -eq 1 ]; then
  CMD+=" --open-browser"
fi

if [ $TERMINAL_DASHBOARD -eq 1 ]; then
  CMD+=" --terminal-dashboard"
fi

if [ $ENABLE_WORK_STEALING -eq 1 ]; then
  CMD+=" --enable-work-stealing"
fi

# Print summary
echo -e "${YELLOW}System Configuration:${NC}"
echo -e "  Coordinator Port: ${GREEN}$COORDINATOR_PORT${NC}"
echo -e "  Dashboard Port: ${GREEN}$DASHBOARD_PORT${NC}"
echo -e "  Database Path: ${GREEN}$DB_PATH${NC}"
echo -e "  Mock Workers: ${GREEN}$MOCK_WORKERS${NC}"
echo -e "  Dashboard Type: ${GREEN}$([ $TERMINAL_DASHBOARD -eq 1 ] && echo "Terminal" || echo "Web")${NC}"
echo -e "  Work Stealing: ${GREEN}$([ $ENABLE_WORK_STEALING -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Stress Test: ${GREEN}$([ $STRESS_TEST -eq 1 ] && echo "Enabled ($TEST_TASKS tasks, ${TEST_DURATION}s)" || echo "Disabled")${NC}"
echo -e "  Log File: ${GREEN}$LOG_FILE${NC}"
echo ""

# Confirm execution
echo -e "${YELLOW}Starting Distributed Testing Framework...${NC}"
echo -e "Press Ctrl+C to stop the server"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run the command
echo "$CMD" | tee "$LOG_FILE"
echo -e "${BLUE}------------------------------------------------------------${NC}"
$CMD 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "${GREEN}Server shutdown successfully${NC}"
else
  echo -e "${RED}Server exited with error code: $EXIT_STATUS${NC}"
fi

echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "${YELLOW}Log saved to: ${GREEN}$LOG_FILE${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

exit $EXIT_STATUS