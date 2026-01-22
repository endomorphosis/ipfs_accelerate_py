#!/bin/bash
# run_integrated_system.sh
#
# This script provides a convenient way to run the complete Distributed Testing Framework
# with enhanced features including Multi-Device Orchestrator, Fault Tolerance System,
# and Comprehensive Monitoring Dashboard.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Distributed Testing Framework - Integrated System Runner  ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
MOCK_WORKERS=3
DB_PATH="./benchmark_db.duckdb"
HOST="localhost"
COORDINATOR_PORT=8080
DASHBOARD_PORT=8888
OPEN_BROWSER=0
STRESS_TEST=0
TEST_TASKS=20
TEST_DURATION=60
TERMINAL_DASHBOARD=0
ENABLE_WORK_STEALING=0
FAULT_INJECTION=0
FAULT_RATE=0.1
HIGH_AVAILABILITY=0
WEB_INTEGRATION=0
PERFORMANCE_ANALYTICS=0
ENHANCED_HARDWARE=0
DYNAMIC_RESOURCE=0
RESULT_AGGREGATION=0
VISUALIZATION_PATH="./visualizations"
ORCHESTRATOR_STRATEGY="auto"
PREDICTION_MODEL=""
LOG_FILE="integrated_system_$(date +%Y%m%d_%H%M%S).log"
DEBUG=0
COMMAND=""

# Function to show help
show_help() {
  echo -e "${GREEN}Usage: $0 [options] [command]${NC}"
  echo ""
  echo -e "${CYAN}Commands:${NC}"
  echo "  basic               Run basic integrated system"
  echo "  comprehensive       Run with comprehensive feature set"
  echo "  fault-tolerant      Run with enhanced fault tolerance features"
  echo "  web-integration     Run with WebNN/WebGPU integration"
  echo "  high-availability   Run in high availability mode"
  echo "  performance         Run with performance analytics"
  echo "  full                Run with all features enabled"
  echo ""
  echo -e "${CYAN}Options:${NC}"
  echo "  --mock-workers N        Launch N mock workers (default: $MOCK_WORKERS)"
  echo "  --db-path PATH          Path to database (default: $DB_PATH)"
  echo "  --host HOST             Host to bind to (default: $HOST)"
  echo "  --coordinator-port N    Port for coordinator server (default: $COORDINATOR_PORT)"
  echo "  --dashboard-port N      Port for dashboard server (default: $DASHBOARD_PORT)"
  echo "  --open-browser          Open web browser to dashboard automatically"
  echo "  --stress-test           Run stress test after starting"
  echo "  --test-tasks N          Number of tasks for stress test (default: $TEST_TASKS)"
  echo "  --test-duration N       Duration of stress test in seconds (default: $TEST_DURATION)"
  echo "  --terminal-dashboard    Use terminal-based dashboard instead of web dashboard"
  echo "  --enable-work-stealing  Enable work stealing between workers"
  echo "  --fault-injection       Enable fault injection for testing fault tolerance"
  echo "  --fault-rate RATE       Rate of fault injection (0.0 to 1.0) (default: $FAULT_RATE)"
  echo "  --high-availability     Enable high availability mode"
  echo "  --web-integration       Enable WebNN/WebGPU integration"
  echo "  --performance-analytics Enable performance analytics"
  echo "  --enhanced-hardware     Enable enhanced hardware taxonomy"
  echo "  --dynamic-resource      Enable dynamic resource management"
  echo "  --result-aggregation    Enable advanced result aggregation"
  echo "  --visualization-path P  Path for visualizations (default: $VISUALIZATION_PATH)"
  echo "  --orchestrator-strategy S Orchestrator strategy (default: $ORCHESTRATOR_STRATEGY)"
  echo "  --prediction-model TYPE Performance prediction model (basic or advanced)"
  echo "  --log-file FILE         Log output to specified file (default: $LOG_FILE)"
  echo "  --debug                 Enable debug mode"
  echo "  --help                  Show this help message"
  echo ""
  echo -e "${CYAN}Examples:${NC}"
  echo "  $0 basic"
  echo "  $0 --mock-workers 5 --open-browser comprehensive"
  echo "  $0 --stress-test --test-tasks 50 fault-tolerant"
  echo "  $0 --web-integration --enhanced-hardware web-integration"
  echo "  $0 --performance-analytics --prediction-model advanced performance"
  echo "  $0 --visualization-path ./my_visualizations performance"
  echo "  $0 --high-availability high-availability"
  echo "  $0 --debug full"
}

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
    --host)
      HOST="$2"
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
    --fault-injection)
      FAULT_INJECTION=1
      shift
      ;;
    --fault-rate)
      FAULT_RATE="$2"
      shift 2
      ;;
    --high-availability)
      HIGH_AVAILABILITY=1
      shift
      ;;
    --web-integration)
      WEB_INTEGRATION=1
      shift
      ;;
    --performance-analytics)
      PERFORMANCE_ANALYTICS=1
      shift
      ;;
    --enhanced-hardware)
      ENHANCED_HARDWARE=1
      shift
      ;;
    --dynamic-resource)
      DYNAMIC_RESOURCE=1
      shift
      ;;
    --result-aggregation)
      RESULT_AGGREGATION=1
      shift
      ;;
    --visualization-path)
      VISUALIZATION_PATH="$2"
      shift 2
      ;;
    --orchestrator-strategy)
      ORCHESTRATOR_STRATEGY="$2"
      shift 2
      ;;
    --prediction-model)
      PREDICTION_MODEL="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --help)
      show_help
      exit 0
      ;;
    basic|comprehensive|fault-tolerant|web-integration|high-availability|performance|full)
      COMMAND="$1"
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create base command
CMD="python duckdb_api/distributed_testing/run_integrated_system.py"
CMD+=" --host $HOST"
CMD+=" --port $COORDINATOR_PORT"
CMD+=" --dashboard-port $DASHBOARD_PORT"
CMD+=" --db-path $DB_PATH"

# Apply command presets if specified
if [ -n "$COMMAND" ]; then
  case $COMMAND in
    basic)
      echo -e "${CYAN}Running with basic configuration${NC}"
      ;;
    comprehensive)
      echo -e "${CYAN}Running with comprehensive feature set${NC}"
      MOCK_WORKERS=3
      OPEN_BROWSER=1
      ENABLE_WORK_STEALING=1
      ENHANCED_HARDWARE=1
      RESULT_AGGREGATION=1
      ;;
    fault-tolerant)
      echo -e "${CYAN}Running with enhanced fault tolerance features${NC}"
      MOCK_WORKERS=3
      OPEN_BROWSER=1
      FAULT_INJECTION=1
      ENABLE_WORK_STEALING=1
      STRESS_TEST=1
      ;;
    web-integration)
      echo -e "${CYAN}Running with WebNN/WebGPU integration${NC}"
      MOCK_WORKERS=3
      OPEN_BROWSER=1
      WEB_INTEGRATION=1
      ENHANCED_HARDWARE=1
      ;;
    high-availability)
      echo -e "${CYAN}Running with high availability features${NC}"
      MOCK_WORKERS=3
      OPEN_BROWSER=1
      HIGH_AVAILABILITY=1
      DYNAMIC_RESOURCE=1
      ;;
    performance)
      echo -e "${CYAN}Running with performance analytics${NC}"
      MOCK_WORKERS=3
      OPEN_BROWSER=1
      PERFORMANCE_ANALYTICS=1
      if [ -z "$PREDICTION_MODEL" ]; then
        PREDICTION_MODEL="advanced"
      fi
      ;;
    full)
      echo -e "${CYAN}Running with all features enabled${NC}"
      MOCK_WORKERS=5
      OPEN_BROWSER=1
      ENABLE_WORK_STEALING=1
      HIGH_AVAILABILITY=1
      WEB_INTEGRATION=1
      PERFORMANCE_ANALYTICS=1
      ENHANCED_HARDWARE=1
      DYNAMIC_RESOURCE=1
      RESULT_AGGREGATION=1
      if [ -z "$PREDICTION_MODEL" ]; then
        PREDICTION_MODEL="advanced"
      fi
      ;;
  esac
fi

# Add optional arguments based on flags
if [ $MOCK_WORKERS -gt 0 ]; then
  CMD+=" --mock-workers $MOCK_WORKERS"
fi

if [ $OPEN_BROWSER -eq 1 ]; then
  CMD+=" --open-browser"
fi

if [ $TERMINAL_DASHBOARD -eq 1 ]; then
  CMD+=" --terminal-dashboard"
fi

if [ $STRESS_TEST -eq 1 ]; then
  CMD+=" --stress-test --test-tasks $TEST_TASKS --test-duration $TEST_DURATION"
fi

if [ $FAULT_INJECTION -eq 1 ]; then
  CMD+=" --fault-injection --fault-rate $FAULT_RATE"
fi

if [ $HIGH_AVAILABILITY -eq 1 ]; then
  CMD+=" --high-availability --coordinator-id coordinator-$(date +%s) --auto-leader-election"
fi

if [ $WEB_INTEGRATION -eq 1 ]; then
  CMD+=" --enable-web-integration"
fi

if [ $PERFORMANCE_ANALYTICS -eq 1 ]; then
  CMD+=" --performance-analytics --visualization-path $VISUALIZATION_PATH"
  
  # Add prediction model if specified
  if [ -n "$PREDICTION_MODEL" ]; then
    CMD+=" --prediction-model $PREDICTION_MODEL"
  fi
fi

if [ $ENHANCED_HARDWARE -eq 1 ]; then
  CMD+=" --enhanced-hardware-taxonomy"
fi

if [ $DYNAMIC_RESOURCE -eq 1 ]; then
  CMD+=" --dynamic-resource-management"
fi

if [ $RESULT_AGGREGATION -eq 1 ]; then
  CMD+=" --enable-result-aggregation"
fi

if [ $ENABLE_WORK_STEALING -eq 1 ]; then
  CMD+=" --enable-work-stealing"
fi

if [ "$ORCHESTRATOR_STRATEGY" != "auto" ]; then
  CMD+=" --orchestrator-strategy $ORCHESTRATOR_STRATEGY"
fi

if [ $DEBUG -eq 1 ]; then
  CMD+=" --debug"
fi

# Create visualization directory if needed
if [ $PERFORMANCE_ANALYTICS -eq 1 ] || [ $WEB_INTEGRATION -eq 1 ]; then
  mkdir -p "$VISUALIZATION_PATH"
fi

# Create database directory if needed
DB_DIR=$(dirname "$DB_PATH")
mkdir -p "$DB_DIR"

# Print summary
echo -e "${YELLOW}System Configuration:${NC}"
echo -e "  Host:               ${GREEN}$HOST${NC}"
echo -e "  Coordinator Port:   ${GREEN}$COORDINATOR_PORT${NC}"
echo -e "  Dashboard Port:     ${GREEN}$DASHBOARD_PORT${NC}"
echo -e "  Database Path:      ${GREEN}$DB_PATH${NC}"
echo -e "  Visualization Path: ${GREEN}$VISUALIZATION_PATH${NC}"
echo -e "  Mock Workers:       ${GREEN}$MOCK_WORKERS${NC}"
echo -e "  Dashboard Type:     ${GREEN}$([ $TERMINAL_DASHBOARD -eq 1 ] && echo "Terminal" || echo "Web")${NC}"
echo -e "  Work Stealing:      ${GREEN}$([ $ENABLE_WORK_STEALING -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Fault Injection:    ${GREEN}$([ $FAULT_INJECTION -eq 1 ] && echo "Enabled (rate: $FAULT_RATE)" || echo "Disabled")${NC}"
echo -e "  Stress Test:        ${GREEN}$([ $STRESS_TEST -eq 1 ] && echo "Enabled ($TEST_TASKS tasks, ${TEST_DURATION}s)" || echo "Disabled")${NC}"
echo -e "  High Availability:  ${GREEN}$([ $HIGH_AVAILABILITY -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Web Integration:    ${GREEN}$([ $WEB_INTEGRATION -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Performance Analytics: ${GREEN}$([ $PERFORMANCE_ANALYTICS -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Enhanced Hardware:  ${GREEN}$([ $ENHANCED_HARDWARE -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Dynamic Resource:   ${GREEN}$([ $DYNAMIC_RESOURCE -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Result Aggregation: ${GREEN}$([ $RESULT_AGGREGATION -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Orchestrator Strategy: ${GREEN}$ORCHESTRATOR_STRATEGY${NC}"
echo -e "  Prediction Model:   ${GREEN}$([ -n "$PREDICTION_MODEL" ] && echo "$PREDICTION_MODEL" || echo "None")${NC}"
echo -e "  Debug Mode:         ${GREEN}$([ $DEBUG -eq 1 ] && echo "Enabled" || echo "Disabled")${NC}"
echo -e "  Log File:           ${GREEN}$LOG_FILE${NC}"
echo ""

# Confirm execution
echo -e "${YELLOW}Starting Integrated Distributed Testing System...${NC}"
echo -e "Press Ctrl+C to stop the server"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run the command
echo "$CMD" | tee "$LOG_FILE"
echo -e "${BLUE}------------------------------------------------------------${NC}"
$CMD 2>&1 | tee -a "$LOG_FILE"

# Capture exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "${GREEN}System shutdown successfully${NC}"
else
  echo -e "${RED}System exited with error code: $EXIT_STATUS${NC}"
fi

echo -e "${BLUE}------------------------------------------------------------${NC}"
echo -e "${YELLOW}Log saved to: ${GREEN}$LOG_FILE${NC}"
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_STATUS