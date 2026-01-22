#!/bin/bash
# =============================================================================
# WebGPU Optimizer Benchmark Wrapper Script
# =============================================================================
#
# This script provides a convenient way to run WebGPU optimizer benchmarks,
# correctness tests, and generate performance dashboards. It supports different
# modes, benchmark types, browsers, and various configuration options.
#
# Author: IPFS Accelerate Team
# Version: 1.0.0
# Date: March 2025
#
# Usage:
#   ./run_webgpu_benchmarks.sh [options]
#
# For detailed usage information, run:
#   ./run_webgpu_benchmarks.sh --help

set -e

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEBGPU_BENCHMARKS_DIR="$PROJECT_ROOT/ipfs_accelerate_js/test/performance/webgpu_optimizer"

# Ensure the benchmark directory exists
if [ ! -d "$WEBGPU_BENCHMARKS_DIR" ]; then
  echo "Error: WebGPU benchmark directory not found: $WEBGPU_BENCHMARKS_DIR"
  exit 1
fi

# Define colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║               WebGPU Optimizer Benchmark Runner            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Default settings
BENCHMARK_TYPE="all"
BROWSER="simulate"
ITERATIONS=5
WARMUP=2
VERBOSE=false
MODE="standard"  # standard, correctness, comprehensive, dashboard, clean
DASHBOARD=true
DASHBOARD_ONLY=false
OPEN_DASHBOARD=true

# Process command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      # Display help
      echo -e "${GREEN}Usage:${NC} ./run_webgpu_benchmarks.sh [options]"
      echo
      echo "Options:"
      echo "  -h, --help                 Show this help message"
      echo "  -t, --type TYPE            Benchmark type to run (default: all)"
      echo "                             Options: all, general, memory-layout, browser-specific,"
      echo "                             operation-fusion, neural-pattern"
      echo "  -b, --browser BROWSER      Browser to use (default: simulate)"
      echo "                             Options: simulate, chrome, firefox, edge, all"
      echo "  -i, --iterations N         Number of iterations for each benchmark (default: 5)"
      echo "  -w, --warmup N             Number of warmup iterations (default: 2)"
      echo "  -v, --verbose              Enable verbose output"
      echo "  -m, --mode MODE            Run mode (default: standard)"
      echo "                             Options: standard, correctness, comprehensive, dashboard, clean"
      echo "  -nd, --no-dashboard        Don't generate dashboard"
      echo "  -do, --dashboard-only      Only generate dashboard (skip benchmarks)"
      echo "  -no, --no-open             Don't open dashboard after generation"
      echo
      echo "Examples:"
      echo "  # Run all benchmarks with default settings"
      echo "  ./run_webgpu_benchmarks.sh"
      echo
      echo "  # Run only memory layout benchmarks in Chrome"
      echo "  ./run_webgpu_benchmarks.sh --type memory-layout --browser chrome"
      echo
      echo "  # Run comprehensive benchmarks with more iterations"
      echo "  ./run_webgpu_benchmarks.sh --mode comprehensive --iterations 10 --warmup 5"
      echo
      echo "  # Only generate dashboard from existing results"
      echo "  ./run_webgpu_benchmarks.sh --dashboard-only"
      echo
      echo "  # Clean all benchmark results and dashboard files"
      echo "  ./run_webgpu_benchmarks.sh --mode clean"
      exit 0
      ;;
    -t|--type)
      BENCHMARK_TYPE="$2"
      shift
      shift
      ;;
    -b|--browser)
      BROWSER="$2"
      shift
      shift
      ;;
    -i|--iterations)
      ITERATIONS="$2"
      shift
      shift
      ;;
    -w|--warmup)
      WARMUP="$2"
      shift
      shift
      ;;
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -m|--mode)
      MODE="$2"
      shift
      shift
      ;;
    -nd|--no-dashboard)
      DASHBOARD=false
      shift
      ;;
    -do|--dashboard-only)
      DASHBOARD_ONLY=true
      shift
      ;;
    -no|--no-open)
      OPEN_DASHBOARD=false
      shift
      ;;
    *)
      echo -e "${RED}Error:${NC} Unknown option $1"
      exit 1
      ;;
  esac
done

# Check benchmark type
case $BENCHMARK_TYPE in
  all|general|memory-layout|browser-specific|operation-fusion|neural-pattern)
    # Valid type
    ;;
  *)
    echo -e "${RED}Error:${NC} Unknown benchmark type: $BENCHMARK_TYPE"
    exit 1
    ;;
esac

# Check browser
case $BROWSER in
  simulate|chrome|firefox|edge|all)
    # Valid browser
    ;;
  *)
    echo -e "${RED}Error:${NC} Unknown browser: $BROWSER"
    exit 1
    ;;
esac

# Check mode
case $MODE in
  standard|correctness|comprehensive|dashboard|clean)
    # Valid mode
    ;;
  *)
    echo -e "${RED}Error:${NC} Unknown mode: $MODE"
    exit 1
    ;;
esac

# Handle clean mode
if [[ $MODE == "clean" ]]; then
  echo -e "${YELLOW}Cleaning all benchmark results and dashboard files...${NC}"
  
  # Clean benchmark results
  rm -rf "$WEBGPU_BENCHMARKS_DIR/benchmark_results"
  rm -rf "$WEBGPU_BENCHMARKS_DIR/dashboard_output"
  
  echo -e "${GREEN}All benchmark results and dashboard files have been cleaned.${NC}"
  exit 0
fi

# Handle dashboard-only mode
if [[ $DASHBOARD_ONLY == true ]]; then
  echo -e "${YELLOW}Generating dashboard from existing results...${NC}"
  
  # Check if dashboard generation script exists
  if [ ! -f "$WEBGPU_BENCHMARKS_DIR/dashboard/generate_dashboard.js" ]; then
    echo -e "${RED}Error:${NC} Dashboard generation script not found: $WEBGPU_BENCHMARKS_DIR/dashboard/generate_dashboard.js"
    exit 1
  fi
  
  # Generate dashboard
  DASHBOARD_CMD="node $WEBGPU_BENCHMARKS_DIR/dashboard/generate_dashboard.js"
  if [[ $VERBOSE == true ]]; then
    DASHBOARD_CMD="$DASHBOARD_CMD --verbose"
  fi
  if [[ $OPEN_DASHBOARD == false ]]; then
    DASHBOARD_CMD="$DASHBOARD_CMD --no-open"
  fi
  
  echo -e "${CYAN}Executing:${NC} $DASHBOARD_CMD"
  eval "$DASHBOARD_CMD"
  
  echo -e "${GREEN}Dashboard generation completed.${NC}"
  exit 0
fi

# Handle correctness mode
if [[ $MODE == "correctness" ]]; then
  echo -e "${YELLOW}Running WebGPU optimizer correctness tests...${NC}"
  
  # Check if Jest is installed
  if ! command -v npx &> /dev/null; then
    echo -e "${RED}Error:${NC} npx command not found. Make sure Node.js and npm are installed."
    exit 1
  fi
  
  # Run correctness tests
  CORRECTNESS_CMD="cd $PROJECT_ROOT/ipfs_accelerate_js && npx jest test/performance/webgpu_optimizer/test_optimizer_correctness.ts"
  
  if [[ $VERBOSE == true ]]; then
    CORRECTNESS_CMD="$CORRECTNESS_CMD --verbose"
  fi
  
  echo -e "${CYAN}Executing:${NC} $CORRECTNESS_CMD"
  eval "$CORRECTNESS_CMD"
  
  echo -e "${GREEN}Correctness tests completed.${NC}"
  exit 0
fi

# Handle comprehensive mode
if [[ $MODE == "comprehensive" ]]; then
  echo -e "${YELLOW}Running comprehensive WebGPU optimizer benchmarks...${NC}"
  
  # Check if Node.js is installed
  if ! command -v node &> /dev/null; then
    echo -e "${RED}Error:${NC} node command not found. Make sure Node.js is installed."
    exit 1
  fi
  
  # Prepare benchmark command
  BENCHMARK_CMD="cd $PROJECT_ROOT/ipfs_accelerate_js && node test/performance/webgpu_optimizer/run_comprehensive_benchmarks.js"
  
  # Add options
  if [[ $VERBOSE == true ]]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --verbose"
  fi
  
  BENCHMARK_CMD="$BENCHMARK_CMD --iterations=$ITERATIONS --warmup=$WARMUP"
  
  if [[ $DASHBOARD == false ]]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --no-dashboard"
  fi
  
  if [[ $OPEN_DASHBOARD == false ]]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --no-open"
  fi
  
  if [[ $BROWSER != "simulate" ]]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --no-simulate"
    
    if [[ $BROWSER == "all" ]]; then
      BENCHMARK_CMD="$BENCHMARK_CMD --browsers=chrome,firefox,edge"
    else
      BENCHMARK_CMD="$BENCHMARK_CMD --browsers=$BROWSER"
    fi
  fi
  
  if [[ $BENCHMARK_TYPE != "all" ]]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --only=$BENCHMARK_TYPE"
  fi
  
  echo -e "${CYAN}Executing:${NC} $BENCHMARK_CMD"
  eval "$BENCHMARK_CMD"
  
  echo -e "${GREEN}Comprehensive benchmarks completed.${NC}"
  exit 0
fi

# Handle standard mode
echo -e "${YELLOW}Running WebGPU optimizer benchmarks...${NC}"

# Determine benchmark script
SCRIPT=""
case $BENCHMARK_TYPE in
  all)
    echo -e "${CYAN}Running all benchmark types${NC}"
    ;;
  general)
    SCRIPT="test_webgpu_optimizer_benchmark.ts"
    ;;
  memory-layout)
    SCRIPT="test_memory_layout_optimization.ts"
    ;;
  browser-specific)
    SCRIPT="test_browser_specific_optimizations.ts"
    ;;
  operation-fusion)
    SCRIPT="test_operation_fusion.ts"
    ;;
  neural-pattern)
    SCRIPT="test_neural_network_pattern_recognition.ts"
    ;;
esac

# Prepare environment variables
ENV_VARS="BENCHMARK_ITERATIONS=$ITERATIONS BENCHMARK_WARMUP_ITERATIONS=$WARMUP"

if [[ $VERBOSE == true ]]; then
  ENV_VARS="$ENV_VARS VERBOSE=true"
fi

# Determine browser settings
if [[ $BROWSER == "simulate" ]]; then
  ENV_VARS="$ENV_VARS BENCHMARK_SIMULATE_BROWSERS=true"
else
  ENV_VARS="$ENV_VARS BENCHMARK_SIMULATE_BROWSERS=false"
  
  if [[ $BROWSER == "all" ]]; then
    ENV_VARS="$ENV_VARS BENCHMARK_BROWSERS=chrome,firefox,edge"
  else
    ENV_VARS="$ENV_VARS BENCHMARK_BROWSERS=$BROWSER"
  fi
fi

# Output directory
OUTPUT_DIR="$WEBGPU_BENCHMARKS_DIR/benchmark_results"
mkdir -p "$OUTPUT_DIR"
ENV_VARS="$ENV_VARS BENCHMARK_OUTPUT_DIR=$OUTPUT_DIR"

# Run benchmark
if [[ $BENCHMARK_TYPE == "all" ]]; then
  # Run all benchmark types
  for type in general memory-layout browser-specific operation-fusion neural-pattern; do
    case $type in
      general)
        script="test_webgpu_optimizer_benchmark.ts"
        ;;
      memory-layout)
        script="test_memory_layout_optimization.ts"
        ;;
      browser-specific)
        script="test_browser_specific_optimizations.ts"
        ;;
      operation-fusion)
        script="test_operation_fusion.ts"
        ;;
      neural-pattern)
        script="test_neural_network_pattern_recognition.ts"
        ;;
    esac
    
    echo -e "${CYAN}Running $type benchmarks...${NC}"
    
    BENCHMARK_CMD="cd $PROJECT_ROOT/ipfs_accelerate_js && $ENV_VARS npx ts-node test/performance/webgpu_optimizer/$script"
    
    echo -e "${CYAN}Executing:${NC} $BENCHMARK_CMD"
    eval "$BENCHMARK_CMD"
    
    echo -e "${GREEN}$type benchmarks completed.${NC}"
  done
else
  # Run specific benchmark type
  BENCHMARK_CMD="cd $PROJECT_ROOT/ipfs_accelerate_js && $ENV_VARS npx ts-node test/performance/webgpu_optimizer/$SCRIPT"
  
  echo -e "${CYAN}Executing:${NC} $BENCHMARK_CMD"
  eval "$BENCHMARK_CMD"
  
  echo -e "${GREEN}$BENCHMARK_TYPE benchmarks completed.${NC}"
fi

# Generate dashboard if enabled
if [[ $DASHBOARD == true ]]; then
  echo -e "${YELLOW}Generating dashboard...${NC}"
  
  # Generate dashboard
  DASHBOARD_CMD="cd $PROJECT_ROOT/ipfs_accelerate_js && node test/performance/webgpu_optimizer/dashboard/generate_dashboard.js"
  
  if [[ $VERBOSE == true ]]; then
    DASHBOARD_CMD="$DASHBOARD_CMD --verbose"
  fi
  
  if [[ $OPEN_DASHBOARD == false ]]; then
    DASHBOARD_CMD="$DASHBOARD_CMD --no-open"
  fi
  
  echo -e "${CYAN}Executing:${NC} $DASHBOARD_CMD"
  eval "$DASHBOARD_CMD"
  
  echo -e "${GREEN}Dashboard generation completed.${NC}"
fi

echo -e "${GREEN}All tasks completed successfully.${NC}"