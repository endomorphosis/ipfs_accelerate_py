#!/bin/bash
# Quick run script for WebNN and WebGPU coverage testing

# Default settings
TEST_TYPE="standard"
BROWSER="edge"
MODEL_TYPE="all"
OUTPUT_DIR="./webnn_coverage_results"
DB_PATH=${BENCHMARK_DB_PATH:-"./benchmark_db.duckdb"}
REPORT_FORMAT="markdown"
OPTIMIZATIONS="none"

# Parse command-line arguments
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
    --help)
      echo "WebNN and WebGPU Coverage Testing Quick Run Script"
      echo ""
      echo "Usage: ./run_webnn_webgpu_coverage.sh [options]"
      echo ""
      echo "Options:"
      echo "  --test-type TYPE     Test type: quick, capabilities, standard, comprehensive, firefox-audio,"
      echo "                       browser-comparison, or optimization-impact (default: standard)"
      echo "  --browser BROWSER    Browser to test: chrome, edge, firefox, or safari (default: edge)"
      echo "  --model-type TYPE    Model type: text, audio, multimodal, vision, or all (default: all)"
      echo "  --output-dir DIR     Output directory (default: ./webnn_coverage_results)"
      echo "  --db-path PATH       DuckDB database path (default: ./benchmark_db.duckdb or BENCHMARK_DB_PATH)"
      echo "  --report-format FMT  Report format: markdown or html (default: markdown)"
      echo "  --optimizations OPT  Optimizations: none, compute-shaders, parallel-loading,"
      echo "                       shader-precompile, or all (default: none)"
      echo "  --help               Show this help message"
      echo ""
      echo "Example:"
      echo "  ./run_webnn_webgpu_coverage.sh --test-type quick --browser chrome"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information"
      exit 1
      ;;
  esac
done

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
    MODEL_ARGS="--models prajjwal1/bert-tiny whisper-tiny openai/clip-vit-base-patch32"
    ;;
  *)
    echo "Unknown model type: $MODEL_TYPE"
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
    echo "Unknown optimization option: $OPTIMIZATIONS"
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
    echo "Unknown test type: $TEST_TYPE"
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

# Run the test
echo "Running WebNN/WebGPU coverage test with configuration:"
echo "Test type: $TEST_TYPE"
echo "Browser: $BROWSER"
echo "Model type: $MODEL_TYPE"
echo "Output directory: $OUTPUT_DIR"
echo "Database path: ${DB_PATH:-None}"
echo "Report format: $REPORT_FORMAT"
echo "Optimizations: $OPTIMIZATIONS"
echo ""
echo "Command: python run_webnn_coverage_tests.py $TEST_ARGS --output-dir $OUTPUT_DIR $DB_ARGS $REPORT_ARGS"
echo ""
echo "Running test..."

python run_webnn_coverage_tests.py $TEST_ARGS --output-dir "$OUTPUT_DIR" $DB_ARGS $REPORT_ARGS

# Check exit code
if [ $? -eq 0 ]; then
  echo "Test completed successfully!"
else
  echo "Test failed with errors."
  exit 1
fi