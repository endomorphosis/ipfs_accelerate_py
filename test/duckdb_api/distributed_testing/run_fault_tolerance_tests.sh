#!/bin/bash
# run_fault_tolerance_tests.sh
#
# This script runs the fault tolerance tests for the Distributed Testing Framework.
# It includes both load balancer fault tolerance tests and hardware-aware fault tolerance tests.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Distributed Testing Framework - Fault Tolerance Tests     ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
RUN_ALL=0
SPECIFIC_TEST=""
VERBOSE=0
LOG_FILE="fault_tolerance_tests_$(date +%Y%m%d_%H%M%S).log"
TEST_TYPE="all"  # Default to running all test types

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      RUN_ALL=1
      shift
      ;;
    --test)
      SPECIFIC_TEST="$2"
      shift 2
      ;;
    --verbose | -v)
      VERBOSE=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo ""
      echo "Options:"
      echo "  --all               Run all fault tolerance tests"
      echo "  --test TEST_NAME    Run a specific test (e.g., test_06_worker_recovery)"
      echo "  --verbose, -v       Enable verbose output"
      echo "  --log-file FILE     Log output to specified file"
      echo "  --type TYPE         Test type: all, load_balancer, hardware, coordinator"
      echo "  --help              Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --all"
      echo "  $0 --type hardware --verbose"
      echo "  $0 --test test_06_worker_recovery --type load_balancer"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set default if no options specified
if [ $RUN_ALL -eq 0 ] && [ -z "$SPECIFIC_TEST" ]; then
  RUN_ALL=1
fi

# Initialize commands array
declare -a COMMANDS
declare -a DESCRIPTIONS

# Add test commands based on type
if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "load_balancer" ]]; then
  if [ $RUN_ALL -eq 1 ]; then
    COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_load_balancer_fault_tolerance")
    DESCRIPTIONS+=("load balancer fault tolerance tests")
  else
    if [[ "$SPECIFIC_TEST" == "test_"* ]]; then
      COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_load_balancer_fault_tolerance LoadBalancerFaultToleranceTest.$SPECIFIC_TEST")
      DESCRIPTIONS+=("load balancer $SPECIFIC_TEST")
    fi
  fi
fi

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "hardware" ]]; then
  if [ $RUN_ALL -eq 1 ]; then
    COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_hardware_fault_tolerance")
    DESCRIPTIONS+=("hardware-aware fault tolerance tests")
  else
    if [[ "$SPECIFIC_TEST" == "test_"* ]]; then
      COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_hardware_fault_tolerance HardwareAwareFaultToleranceTest.$SPECIFIC_TEST")
      DESCRIPTIONS+=("hardware-aware $SPECIFIC_TEST")
    fi
  fi
fi

if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "coordinator" ]]; then
  if [ $RUN_ALL -eq 1 ]; then
    COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_auto_recovery")
    DESCRIPTIONS+=("coordinator auto recovery tests")
  else
    if [[ "$SPECIFIC_TEST" == "test_"* ]]; then
      COMMANDS+=("python -m duckdb_api.distributed_testing.tests.test_auto_recovery AutoRecoveryTest.$SPECIFIC_TEST")
      DESCRIPTIONS+=("coordinator recovery $SPECIFIC_TEST")
    fi
  fi
fi

# Add verbose flag if specified
if [ $VERBOSE -eq 1 ]; then
  for i in "${!COMMANDS[@]}"; do
    COMMANDS[$i]="${COMMANDS[$i]} -v"
  done
fi

# Check if we have any commands to run
if [ ${#COMMANDS[@]} -eq 0 ]; then
  echo -e "${RED}Error: No tests specified. Check your --type and --test options.${NC}"
  exit 1
fi

# Print test information
echo -e "${YELLOW}Running ${#COMMANDS[@]} test suites:${NC}"
for i in "${!COMMANDS[@]}"; do
  echo -e "${CYAN}[$((i+1))] ${DESCRIPTIONS[$i]}${NC}"
  echo -e "    ${GREEN}${COMMANDS[$i]}${NC}"
done
echo -e "Log file: ${GREEN}$LOG_FILE${NC}"
echo -e "${BLUE}------------------------------------------------------------${NC}"

# Run the tests and capture output
echo "Started: $(date)" | tee "$LOG_FILE"
echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$LOG_FILE"

# Initialize success tracker
ALL_PASSED=true
FAILED_TESTS=()

# Execute each command
for i in "${!COMMANDS[@]}"; do
  echo -e "${CYAN}[$((i+1))/${#COMMANDS[@]}] Running: ${DESCRIPTIONS[$i]}${NC}" | tee -a "$LOG_FILE"
  echo "${COMMANDS[$i]}" | tee -a "$LOG_FILE"
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$LOG_FILE"
  
  # Execute the command
  eval "${COMMANDS[$i]}" 2>&1 | tee -a "$LOG_FILE"
  
  # Capture exit status
  EXIT_STATUS=${PIPESTATUS[0]}
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$LOG_FILE"
  
  if [ $EXIT_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ ${DESCRIPTIONS[$i]} passed!${NC}" | tee -a "$LOG_FILE"
  else
    echo -e "${RED}❌ ${DESCRIPTIONS[$i]} failed with exit code $EXIT_STATUS${NC}" | tee -a "$LOG_FILE"
    ALL_PASSED=false
    FAILED_TESTS+=("${DESCRIPTIONS[$i]}")
  fi
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
done

echo "Finished: $(date)" | tee -a "$LOG_FILE"

# Print summary
echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${YELLOW}Test Summary:${NC}" | tee -a "$LOG_FILE"

if [ "$ALL_PASSED" = true ]; then
  echo -e "${GREEN}✅ All test suites passed!${NC}" | tee -a "$LOG_FILE"
else
  echo -e "${RED}❌ Some test suites failed:${NC}" | tee -a "$LOG_FILE"
  for test in "${FAILED_TESTS[@]}"; do
    echo -e "${RED}   - $test${NC}" | tee -a "$LOG_FILE"
  done
fi

echo -e "${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${YELLOW}Log saved to: ${GREEN}$LOG_FILE${NC}"

# Set exit status based on test results
if [ "$ALL_PASSED" = true ]; then
  exit 0
else
  exit 1
fi