#!/bin/bash
# Run hardware benchmarking script with different configurations

set -e

# Create benchmark results directory if it doesn't exist
BENCHMARK_DIR="./benchmark_results"
mkdir -p $BENCHMARK_DIR

# Default values
MODEL_SET="quick"
HARDWARE_SET="quick"
BATCH_SIZES="1 4 16"
FORMAT="markdown"
DEBUG=""
OPENVINO_PRECISION="FP32"
WARMUP=2
RUNS=5
OUTPUT_DIR="$BENCHMARK_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-set)
      MODEL_SET="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --hardware-set)
      HARDWARE_SET="$2"
      shift 2
      ;;
    --hardware)
      HARDWARE="$2"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES="$2"
      shift 2
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --openvino-precision)
      OPENVINO_PRECISION="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model-set MODEL_SET          Set of models to benchmark (default: quick)"
      echo "                                 Available sets: text_embedding, text_generation, vision, audio, all, quick"
      echo "  --models \"MODEL1 MODEL2...\"    Space-separated list of specific models to benchmark"
      echo "  --hardware-set HARDWARE_SET    Set of hardware backends to test (default: quick)"
      echo "                                 Available sets: local, gpu, intel, all, web, quick"
      echo "  --hardware \"HW1 HW2...\"        Space-separated list of specific hardware backends to test"
      echo "  --batch-sizes \"SIZE1 SIZE2...\" Space-separated list of batch sizes to test (default: \"1 4 16\")"
      echo "  --format FORMAT                Output format: markdown, json, csv (default: markdown)"
      echo "  --debug                        Enable debug logging"
      echo "  --openvino-precision PRECISION Precision for OpenVINO models: FP32, FP16, INT8 (default: FP32)"
      echo "  --warmup COUNT                 Number of warmup runs (default: 2)"
      echo "  --runs COUNT                   Number of measurement runs (default: 5)"
      echo "  --output-dir DIR               Directory to save benchmark results (default: ./benchmark_results)"
      echo "  --help                         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Prepare command
CMD="python run_hardware_comparison.py"

if [ -n "$MODEL_SET" ] && [ -z "$MODELS" ]; then
  CMD="$CMD --model-set $MODEL_SET"
fi

if [ -n "$MODELS" ]; then
  CMD="$CMD --models $MODELS"
fi

if [ -n "$HARDWARE_SET" ] && [ -z "$HARDWARE" ]; then
  CMD="$CMD --hardware-set $HARDWARE_SET"
fi

if [ -n "$HARDWARE" ]; then
  CMD="$CMD --hardware $HARDWARE"
fi

if [ -n "$BATCH_SIZES" ]; then
  CMD="$CMD --batch-sizes $BATCH_SIZES"
fi

if [ -n "$FORMAT" ]; then
  CMD="$CMD --format $FORMAT"
fi

if [ -n "$DEBUG" ]; then
  CMD="$CMD $DEBUG"
fi

if [ -n "$OPENVINO_PRECISION" ]; then
  CMD="$CMD --openvino-precision $OPENVINO_PRECISION"
fi

if [ -n "$WARMUP" ]; then
  CMD="$CMD --warmup $WARMUP"
fi

if [ -n "$RUNS" ]; then
  CMD="$CMD --runs $RUNS"
fi

if [ -n "$OUTPUT_DIR" ]; then
  CMD="$CMD --output-dir $OUTPUT_DIR"
fi

# Print command
echo "Running: $CMD"

# Execute command
eval $CMD

# Print summary report location
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$FORMAT" == "markdown" ]; then
  echo "Benchmark report saved to: $OUTPUT_DIR/benchmark_report_*.md"
  if command -v cat > /dev/null; then
    if [ -f "$OUTPUT_DIR/benchmark_report_*.md" ]; then
      echo "Summary:"
      cat "$OUTPUT_DIR/benchmark_report_*.md" | grep -A 20 "## Summary" | head -n 20
    fi
  fi
else
  echo "Benchmark report saved to: $OUTPUT_DIR/benchmark_report_*.$FORMAT"
fi

echo "Hardware benchmark completed at $(date)"