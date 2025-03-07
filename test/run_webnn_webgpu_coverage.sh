#!/bin/bash
# WebNN and WebGPU Coverage Testing Script
# Enhanced version with predefined test profiles and detailed reporting
# Version: 2.0 - Updated March 2025

# Default settings
TEST_TYPE="standard"
BROWSER="edge"
MODEL_TYPE="all"
OUTPUT_DIR="./webnn_coverage_results"
DB_PATH=${BENCHMARK_DB_PATH:-"./benchmark_db.duckdb"}
REPORT_FORMAT="markdown"
OPTIMIZATIONS="none"
PARALLEL="true"
MAX_WORKERS="2"
TIMEOUT="600"
USE_REAL_BROWSERS="auto"
VERBOSE="false"
QUICK_PROFILE=""

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to show the title header
show_header() {
  echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║                WebNN/WebGPU Coverage Test Tool                 ║${NC}"
  echo -e "${CYAN}║                      March 2025 Edition                        ║${NC}"
  echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

# Function to show usage information
show_usage() {
  show_header
  
  echo -e "${GREEN}Usage:${NC} ./run_webnn_webgpu_coverage.sh [options] [profile]"
  echo ""
  echo -e "${YELLOW}Profiles:${NC}"
  echo "  quick               - Quick capability check with minimal models"
  echo "  capabilities-only   - Only check browser WebNN/WebGPU capabilities"
  echo "  firefox-audio       - Focus on Firefox audio optimization tests"
  echo "  all-browsers        - Test across Chrome, Edge, and Firefox"
  echo "  full                - Full test suite with all models and optimizations"
  echo "  optimization-check  - Focus on measuring optimization impact"
  echo ""
  echo -e "${YELLOW}Options:${NC}"
  echo "  --test-type TYPE     - Test configuration type: quick, capabilities, standard,"
  echo "                         comprehensive, firefox-audio, browser-comparison,"
  echo "                         or optimization-impact (default: standard)"
  echo "  --browser BROWSER    - Browser to test: chrome, edge, firefox, safari"
  echo "                         (default: edge)"
  echo "  --model-type TYPE    - Model type: text, audio, multimodal, vision, all"
  echo "                         (default: all)"
  echo "  --output-dir DIR     - Output directory for results and reports"
  echo "                         (default: ./webnn_coverage_results)"
  echo "  --db-path PATH       - DuckDB database path"
  echo "                         (default: BENCHMARK_DB_PATH or ./benchmark_db.duckdb)"
  echo "  --report-format FMT  - Report format: markdown or html (default: markdown)"
  echo "  --optimizations OPT  - Optimizations: none, compute-shaders, parallel-loading,"
  echo "                         shader-precompile, all (default: none)"
  echo "  --parallel           - Run tests in parallel (default: true)"
  echo "  --sequential         - Run tests sequentially"
  echo "  --max-workers N      - Maximum number of parallel workers (default: 2)"
  echo "  --timeout N          - Timeout in seconds for each test (default: 600)"
  echo "  --real-browsers      - Force using real browser instances"
  echo "  --no-real-browsers   - Disable using real browser instances"
  echo "  --verbose            - Show more detailed output"
  echo "  --help               - Show this help message"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  ./run_webnn_webgpu_coverage.sh quick"
  echo "  ./run_webnn_webgpu_coverage.sh --browser chrome --model-type audio --optimizations compute-shaders"
  echo "  ./run_webnn_webgpu_coverage.sh firefox-audio --output-dir ./firefox_results"
  echo "  ./run_webnn_webgpu_coverage.sh full --report-format html"
  echo ""
  
  echo -e "${YELLOW}Recommended Use Cases:${NC}"
  echo "  • Checking browser compatibility:      ./run_webnn_webgpu_coverage.sh capabilities-only"
  echo "  • Testing audio models on Firefox:     ./run_webnn_webgpu_coverage.sh firefox-audio"
  echo "  • Generate complete HTML report:       ./run_webnn_webgpu_coverage.sh full --report-format html"
  echo "  • Quick check of a specific browser:   ./run_webnn_webgpu_coverage.sh quick --browser chrome"
  echo "  • Compare optimization impact:         ./run_webnn_webgpu_coverage.sh optimization-check"
  echo ""
}

# Parse command-line arguments
if [ $# -eq 0 ]; then
  show_usage
  exit 0
fi

# Check for profile argument (non-flag argument)
for arg in "$@"; do
  if [[ "$arg" != --* ]]; then
    QUICK_PROFILE="$arg"
  fi
done

# Parse regular arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --test-type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --browser)
      BROWSER="$2"
      shift 2
      ;;
    --model-type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --report-format)
      REPORT_FORMAT="$2"
      shift 2
      ;;
    --optimizations)
      OPTIMIZATIONS="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="true"
      shift
      ;;
    --sequential)
      PARALLEL="false"
      shift
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --real-browsers)
      USE_REAL_BROWSERS="true"
      shift
      ;;
    --no-real-browsers)
      USE_REAL_BROWSERS="false"
      shift
      ;;
    --verbose)
      VERBOSE="true"
      shift
      ;;
    --help)
      show_usage
      exit 0
      ;;
    quick|capabilities-only|firefox-audio|all-browsers|full|optimization-check)
      # Skip profile argument as we already processed it
      shift
      ;;
    *)
      if [[ "$1" == --* ]]; then
        echo -e "${RED}Error: Unknown option: $1${NC}"
        echo "Run with --help for usage information"
        exit 1
      else
        # Skip non-flag arguments (already processed)
        shift
      fi
      ;;
  esac
done

# Handle predefined profiles
if [ ! -z "$QUICK_PROFILE" ]; then
  case "$QUICK_PROFILE" in
    "quick")
      TEST_TYPE="quick"
      BROWSER="edge"
      MODEL_TYPE="text"
      OPTIMIZATIONS="none"
      ;;
    "capabilities-only")
      TEST_TYPE="capabilities"
      BROWSER="edge"
      # Also test chrome and firefox capabilities if available
      MODEL_TYPE="none"
      OPTIMIZATIONS="none"
      ;;
    "firefox-audio")
      TEST_TYPE="firefox-audio"
      BROWSER="firefox"
      MODEL_TYPE="audio"
      OPTIMIZATIONS="compute-shaders"
      ;;
    "all-browsers")
      TEST_TYPE="browser-comparison"
      MODEL_TYPE="all"
      OPTIMIZATIONS="none"
      ;;
    "full")
      TEST_TYPE="comprehensive"
      MODEL_TYPE="all"
      OPTIMIZATIONS="all"
      ;;
    "optimization-check")
      TEST_TYPE="optimization-impact"
      MODEL_TYPE="all"
      OPTIMIZATIONS="all"
      ;;
    *)
      echo -e "${RED}Error: Unknown profile: $QUICK_PROFILE${NC}"
      show_usage
      exit 1
      ;;
  esac
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set model arguments based on model type
case "$MODEL_TYPE" in
  "text")
    MODEL_ARGS="--models prajjwal1/bert-tiny t5-small"
    ;;
  "audio")
    MODEL_ARGS="--audio-models-only"
    ;;
  "multimodal")
    MODEL_ARGS="--multimodal-models-only"
    ;;
  "vision")
    MODEL_ARGS="--models google/vit-base-patch16-224"
    ;;
  "all")
    MODEL_ARGS="--models prajjwal1/bert-tiny whisper-tiny openai/clip-vit-base-patch32 google/vit-base-patch16-224 t5-small"
    ;;
  "none")
    MODEL_ARGS=""
    ;;
  *)
    echo -e "${RED}Error: Unknown model type: $MODEL_TYPE${NC}"
    exit 1
    ;;
esac

# Set optimization arguments
case "$OPTIMIZATIONS" in
  "none")
    OPTIMIZATION_ARGS=""
    ;;
  "compute-shaders")
    OPTIMIZATION_ARGS="--compute-shaders"
    ;;
  "parallel-loading")
    OPTIMIZATION_ARGS="--parallel-loading"
    ;;
  "shader-precompile")
    OPTIMIZATION_ARGS="--shader-precompile"
    ;;
  "all")
    OPTIMIZATION_ARGS="--all-optimizations"
    ;;
  *)
    echo -e "${RED}Error: Unknown optimization option: $OPTIMIZATIONS${NC}"
    exit 1
    ;;
esac

# Configure test based on test type
case "$TEST_TYPE" in
  "quick")
    # Quick test with minimal configuration
    TEST_ARGS="--quick"
    ;;
  "capabilities")
    # Only check browser capabilities
    TEST_ARGS="--capabilities-only"
    if [ "$BROWSER" != "edge" ]; then
      TEST_ARGS="$TEST_ARGS --browser $BROWSER"
    fi
    ;;
  "standard")
    # Standard test with specified browser and models
    TEST_ARGS="--browser $BROWSER $MODEL_ARGS $OPTIMIZATION_ARGS"
    ;;
  "comprehensive")
    # Comprehensive test across browsers and models
    TEST_ARGS="--all-browsers $MODEL_ARGS $OPTIMIZATION_ARGS"
    ;;
  "firefox-audio")
    # Firefox audio optimization test
    TEST_ARGS="--firefox-audio-only"
    ;;
  "browser-comparison")
    # Cross-browser comparison
    TEST_ARGS="--browsers chrome edge firefox $MODEL_ARGS $OPTIMIZATION_ARGS"
    ;;
  "optimization-impact")
    # Focus on optimization impact
    TEST_ARGS="--browser $BROWSER $MODEL_ARGS --all-optimizations"
    ;;
  *)
    echo -e "${RED}Error: Unknown test type: $TEST_TYPE${NC}"
    exit 1
    ;;
esac

# Add report format
REPORT_ARGS="--report-format $REPORT_FORMAT"

# Add database path if provided
if [ ! -z "$DB_PATH" ]; then
  DB_ARGS="--db-path $DB_PATH"
else
  DB_ARGS=""
fi

# Add parallel/sequential options
if [ "$PARALLEL" = "true" ]; then
  PARALLEL_ARGS="--max-concurrent $MAX_WORKERS"
else
  PARALLEL_ARGS="--sequential"
fi

# Add timeout
TIMEOUT_ARGS="--timeout $TIMEOUT"

# Add real browser options
if [ "$USE_REAL_BROWSERS" = "true" ]; then
  BROWSER_ARGS="--use-real-browsers"
elif [ "$USE_REAL_BROWSERS" = "false" ]; then
  BROWSER_ARGS="--no-real-browsers"
else
  BROWSER_ARGS=""
fi

# Add verbose flag if needed
if [ "$VERBOSE" = "true" ]; then
  VERBOSE_ARGS="--verbose"
else
  VERBOSE_ARGS=""
fi

# Run the test
show_header

echo -e "${MAGENTA}Running WebNN/WebGPU coverage test with configuration:${NC}"
echo -e "${YELLOW}Test type:${NC}       $TEST_TYPE"
echo -e "${YELLOW}Browser:${NC}         $BROWSER"
echo -e "${YELLOW}Model type:${NC}      $MODEL_TYPE"
echo -e "${YELLOW}Output directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}Database path:${NC}    ${DB_PATH:-None}"
echo -e "${YELLOW}Report format:${NC}    $REPORT_FORMAT"
echo -e "${YELLOW}Optimizations:${NC}    $OPTIMIZATIONS"
echo -e "${YELLOW}Parallel execution:${NC} $PARALLEL"
if [ "$PARALLEL" = "true" ]; then
echo -e "${YELLOW}Max workers:${NC}      $MAX_WORKERS"
fi
echo -e "${YELLOW}Timeout:${NC}          $TIMEOUT seconds"
echo -e "${YELLOW}Real browsers:${NC}    $USE_REAL_BROWSERS"
echo ""

echo -e "${BLUE}Command:${NC} python run_webnn_coverage_tests.py $TEST_ARGS $PARALLEL_ARGS $TIMEOUT_ARGS $BROWSER_ARGS $VERBOSE_ARGS --output-dir $OUTPUT_DIR $DB_ARGS $REPORT_ARGS"
echo ""
echo -e "${GREEN}Running test...${NC}"
echo ""

# Record the start time
START_TIME=$(date +%s)

# Run the actual command
python run_webnn_coverage_tests.py $TEST_ARGS $PARALLEL_ARGS $TIMEOUT_ARGS $BROWSER_ARGS $VERBOSE_ARGS --output-dir "$OUTPUT_DIR" $DB_ARGS $REPORT_ARGS

# Check exit code
RESULT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ $RESULT -eq 0 ]; then
  echo -e "${GREEN}✓ Test completed successfully!${NC}"
  
  # Look for the most recent report file
  REPORT_FILES=$(ls -t "$OUTPUT_DIR"/webnn_coverage_report_*.{md,html} 2>/dev/null || echo "")
  REPORT_FILE=$(echo "$REPORT_FILES" | head -n 1)
  
  if [ ! -z "$REPORT_FILE" ]; then
    echo -e "${GREEN}Report saved to:${NC} $REPORT_FILE"
  fi
  
  # Print execution time
  MINUTES=$((DURATION / 60))
  SECONDS=$((DURATION % 60))
  echo -e "${GREEN}Total execution time:${NC} ${MINUTES}m ${SECONDS}s"
  
  # Summarize what was tested
  echo -e "\n${GREEN}Test Summary:${NC}"
  if [ "$TEST_TYPE" = "capabilities" ]; then
    echo "- Checked WebNN and WebGPU capabilities"
  elif [ "$TEST_TYPE" = "firefox-audio" ]; then
    echo "- Tested Firefox with audio models and compute shader optimization"
  elif [ "$TEST_TYPE" = "browser-comparison" ]; then
    echo "- Compared WebNN/WebGPU performance across browsers"
  elif [ "$TEST_TYPE" = "optimization-impact" ]; then
    echo "- Measured impact of WebGPU optimizations"
  elif [ "$TEST_TYPE" = "comprehensive" ]; then
    echo "- Ran comprehensive tests across browsers and models"
  else
    echo "- Ran standard tests with the specified configuration"
  fi
  
  echo -e "\n${YELLOW}Next steps:${NC}"
  echo "- View the report for detailed results and recommendations"
  echo "- Run 'python query_benchmark_db.py --report webnn_webgpu' to see database results"
  echo "- Try different optimization combinations for your specific use case"
  
  exit 0
else
  echo -e "${RED}✗ Test failed with errors.${NC}"
  
  # Print execution time even for failures
  MINUTES=$((DURATION / 60))
  SECONDS=$((DURATION % 60))
  echo -e "${YELLOW}Execution time:${NC} ${MINUTES}m ${SECONDS}s"
  
  # Provide troubleshooting tips
  echo -e "\n${YELLOW}Troubleshooting:${NC}"
  echo "- Check browser availability and configuration"
  echo "- Verify browser flags are correctly set for WebNN/WebGPU"
  echo "- Consider using '--capabilities-only' to verify browser support"
  echo "- Try 'python test_webnn_minimal.py --browser $BROWSER' for a minimal test"
  
  exit 1
fi