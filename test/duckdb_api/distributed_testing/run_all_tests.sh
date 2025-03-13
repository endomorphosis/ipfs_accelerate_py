#!/bin/bash
# run_all_tests.sh
#
# This script provides a unified interface to run all tests for the Distributed Testing Framework,
# including integration tests, fault tolerance tests, monitoring tests, and stress tests.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Distributed Testing Framework - Complete Test Suite       ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
TEST_TYPE="all"
VERBOSE=0
TEST_FILTER=""
LOG_DIR="test_logs_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --filter | -f)
      TEST_FILTER="$2"
      shift 2
      ;;
    --verbose | -v)
      VERBOSE=1
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo ""
      echo "Options:"
      echo "  --type TYPE         Type of tests to run: all, integration, fault, monitoring, stress"
      echo "  --filter, -f FILTER Only run tests matching this filter"
      echo "  --verbose, -v       Enable verbose output"
      echo "  --log-dir DIR       Directory to store log files (default: test_logs_<timestamp>)"
      echo "  --help              Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --type all"
      echo "  $0 --type fault"
      echo "  $0 --type integration --filter load_balancer"
      echo "  $0 --type monitoring --verbose"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create the log directory
mkdir -p "$LOG_DIR"
echo -e "${YELLOW}Log directory created: ${GREEN}$LOG_DIR${NC}"

# Function to run integration tests
run_integration_tests() {
  local filter="$1"
  local log_file="$LOG_DIR/integration_tests.log"
  
  echo -e "${CYAN}Running integration tests...${NC}"
  
  # Build command
  local cmd="python -m duckdb_api.distributed_testing.tests.run_integration_tests"
  
  # Add test filter if specified
  if [ -n "$filter" ]; then
    cmd="$cmd --test $filter"
  fi
  
  # Add verbose flag if specified
  if [ $VERBOSE -eq 1 ]; then
    cmd="$cmd --verbose"
  fi
  
  # Execute command
  echo "Command: $cmd" | tee "$log_file"
  echo "Started: $(date)" | tee -a "$log_file"
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  
  $cmd 2>&1 | tee -a "$log_file"
  local exit_status=$?
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  echo "Finished: $(date)" | tee -a "$log_file"
  
  if [ $exit_status -eq 0 ]; then
    echo -e "${GREEN}✅ Integration tests passed!${NC}" | tee -a "$log_file"
  else
    echo -e "${RED}❌ Integration tests failed with exit code $exit_status${NC}" | tee -a "$log_file"
  fi
  
  echo -e "${YELLOW}Log saved to: ${GREEN}$log_file${NC}"
  echo ""
  
  return $exit_status
}

# Function to run fault tolerance tests
run_fault_tolerance_tests() {
  local filter="$1"
  local log_file="$LOG_DIR/fault_tolerance_tests.log"
  
  echo -e "${CYAN}Running fault tolerance tests...${NC}"
  
  # Build command
  local cmd="python -m duckdb_api.distributed_testing.tests.test_load_balancer_fault_tolerance"
  
  # Add test filter if specified
  if [ -n "$filter" ]; then
    cmd="$cmd LoadBalancerFaultToleranceTest.$filter"
  fi
  
  # Add verbose flag if specified
  if [ $VERBOSE -eq 1 ]; then
    cmd="$cmd -v"
  fi
  
  # Execute command
  echo "Command: $cmd" | tee "$log_file"
  echo "Started: $(date)" | tee -a "$log_file"
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  
  $cmd 2>&1 | tee -a "$log_file"
  local exit_status=$?
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  echo "Finished: $(date)" | tee -a "$log_file"
  
  if [ $exit_status -eq 0 ]; then
    echo -e "${GREEN}✅ Fault tolerance tests passed!${NC}" | tee -a "$log_file"
  else
    echo -e "${RED}❌ Fault tolerance tests failed with exit code $exit_status${NC}" | tee -a "$log_file"
  fi
  
  echo -e "${YELLOW}Log saved to: ${GREEN}$log_file${NC}"
  echo ""
  
  return $exit_status
}

# Function to run monitoring tests
run_monitoring_tests() {
  local filter="$1"
  local log_file="$LOG_DIR/monitoring_tests.log"
  
  echo -e "${CYAN}Running monitoring tests...${NC}"
  
  # Build command
  local cmd="python -m duckdb_api.distributed_testing.tests.test_load_balancer_monitoring"
  
  # Add test filter if specified
  if [ -n "$filter" ]; then
    cmd="$cmd LoadBalancerMonitoringIntegrationTest.$filter"
  fi
  
  # Add verbose flag if specified
  if [ $VERBOSE -eq 1 ]; then
    cmd="$cmd -v"
  fi
  
  # Execute command
  echo "Command: $cmd" | tee "$log_file"
  echo "Started: $(date)" | tee -a "$log_file"
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  
  $cmd 2>&1 | tee -a "$log_file"
  local exit_status=$?
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  echo "Finished: $(date)" | tee -a "$log_file"
  
  if [ $exit_status -eq 0 ]; then
    echo -e "${GREEN}✅ Monitoring tests passed!${NC}" | tee -a "$log_file"
  else
    echo -e "${RED}❌ Monitoring tests failed with exit code $exit_status${NC}" | tee -a "$log_file"
  fi
  
  echo -e "${YELLOW}Log saved to: ${GREEN}$log_file${NC}"
  echo ""
  
  return $exit_status
}

# Function to run stress tests
run_stress_tests() {
  local filter="$1"
  local log_file="$LOG_DIR/stress_tests.log"
  
  echo -e "${CYAN}Running stress tests...${NC}"
  
  # Build command
  local cmd="python -m duckdb_api.distributed_testing.test_load_balancer_stress"
  
  # Add test filter if specified
  if [ -n "$filter" ]; then
    cmd="$cmd --mode $filter"
  else
    cmd="$cmd --mode stress"
  fi
  
  # Add verbose flag if specified
  if [ $VERBOSE -eq 1 ]; then
    cmd="$cmd --verbose"
  fi
  
  # Execute command
  echo "Command: $cmd" | tee "$log_file"
  echo "Started: $(date)" | tee -a "$log_file"
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  
  $cmd 2>&1 | tee -a "$log_file"
  local exit_status=$?
  
  echo -e "${BLUE}------------------------------------------------------------${NC}" | tee -a "$log_file"
  echo "Finished: $(date)" | tee -a "$log_file"
  
  if [ $exit_status -eq 0 ]; then
    echo -e "${GREEN}✅ Stress tests passed!${NC}" | tee -a "$log_file"
  else
    echo -e "${RED}❌ Stress tests failed with exit code $exit_status${NC}" | tee -a "$log_file"
  fi
  
  echo -e "${YELLOW}Log saved to: ${GREEN}$log_file${NC}"
  echo ""
  
  return $exit_status
}

# Main function to run all tests
run_all_tests() {
  local status=0
  local num_failures=0
  
  # Create summary file
  local summary_file="$LOG_DIR/test_summary.log"
  echo "Distributed Testing Framework - Test Summary" > "$summary_file"
  echo "Date: $(date)" >> "$summary_file"
  echo "---------------------------------------------------" >> "$summary_file"
  
  if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "integration" ]]; then
    run_integration_tests "$TEST_FILTER"
    if [ $? -ne 0 ]; then
      status=1
      num_failures=$((num_failures + 1))
      echo "❌ Integration tests: FAILED" >> "$summary_file"
    else
      echo "✅ Integration tests: PASSED" >> "$summary_file"
    fi
  fi
  
  if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "fault" ]]; then
    run_fault_tolerance_tests "$TEST_FILTER"
    if [ $? -ne 0 ]; then
      status=1
      num_failures=$((num_failures + 1))
      echo "❌ Fault tolerance tests: FAILED" >> "$summary_file"
    else
      echo "✅ Fault tolerance tests: PASSED" >> "$summary_file"
    fi
  fi
  
  if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "monitoring" ]]; then
    run_monitoring_tests "$TEST_FILTER"
    if [ $? -ne 0 ]; then
      status=1
      num_failures=$((num_failures + 1))
      echo "❌ Monitoring tests: FAILED" >> "$summary_file"
    else
      echo "✅ Monitoring tests: PASSED" >> "$summary_file"
    fi
  fi
  
  if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "stress" ]]; then
    run_stress_tests "$TEST_FILTER"
    if [ $? -ne 0 ]; then
      status=1
      num_failures=$((num_failures + 1))
      echo "❌ Stress tests: FAILED" >> "$summary_file"
    else
      echo "✅ Stress tests: PASSED" >> "$summary_file"
    fi
  fi
  
  # Print summary
  echo "---------------------------------------------------" >> "$summary_file"
  if [ $status -eq 0 ]; then
    echo "✅ ALL TESTS PASSED" >> "$summary_file"
  else
    echo "❌ TESTS FAILED: $num_failures test suite(s) failed" >> "$summary_file"
  fi
  
  # Print summary to console
  echo -e "${BLUE}============================================================${NC}"
  echo -e "${YELLOW}Test Summary:${NC}"
  cat "$summary_file" | sed -e "s/❌/${RED}❌${NC}/g" -e "s/✅/${GREEN}✅${NC}/g"
  echo -e "${BLUE}============================================================${NC}"
  echo -e "${YELLOW}All logs saved to: ${GREEN}$LOG_DIR${NC}"
  
  return $status
}

# Run tests based on type
run_all_tests
exit $?