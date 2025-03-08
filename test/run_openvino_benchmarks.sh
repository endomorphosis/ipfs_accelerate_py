#!/bin/bash
#
# Run OpenVINO benchmarks with various configurations
# This script provides examples of common benchmark scenarios.
#

# Set terminal colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure the database environment variable is set
if [ -z "$BENCHMARK_DB_PATH" ]; then
  export BENCHMARK_DB_PATH="./benchmark_db.duckdb"
  echo -e "${YELLOW}Setting default BENCHMARK_DB_PATH to ./benchmark_db.duckdb${NC}"
fi

# Check if required tools are available
if ! python -c "import openvino" &> /dev/null; then
  echo -e "${RED}Error: OpenVINO is not installed. Please install it first.${NC}"
  echo "You can install it with: pip install openvino openvino-dev"
  exit 1
fi

# Check if optimum.intel is available
if ! python -c "import optimum.intel" &> /dev/null; then
  echo -e "${YELLOW}Warning: optimum.intel is not installed. Some features may not work.${NC}"
  echo "You can install it with: pip install optimum[openvino]"
fi

# Parse command line arguments
MODELS=""
DEVICE="CPU"
PRECISION="FP32,FP16,INT8"
BATCH_SIZES="1,4,16"
DRY_RUN=0
MODEL_FAMILY=""
ITERATIONS=10
OUTPUT_DIR="./benchmark_results/openvino"
REPORT=0
REPORT_FORMAT="markdown"
DB_PATH=""

# Display help message
function show_help {
  echo -e "${BLUE}OpenVINO Benchmark Runner${NC}"
  echo ""
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -m, --models       Comma-separated list of models to benchmark"
  echo "  -f, --family       Model family to benchmark (text, vision, audio, multimodal)"
  echo "  -d, --device       OpenVINO device (CPU, GPU, AUTO, etc.)"
  echo "  -p, --precision    Comma-separated list of precisions (FP32, FP16, INT8)"
  echo "  -b, --batch-sizes  Comma-separated list of batch sizes"
  echo "  -i, --iterations   Number of iterations for each benchmark"
  echo "  -o, --output-dir   Directory to save results"
  echo "  -r, --report       Generate report (0=no, 1=yes)"
  echo "  --report-format    Report format (markdown, html, json)"
  echo "  --db-path          Path to the benchmark database"
  echo "  --dry-run          List benchmarks without running them"
  echo "  -h, --help         Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 --models bert-base-uncased --device CPU --precision FP32,FP16"
  echo "  $0 --family text --batch-sizes 1,2,4,8 --report 1"
  echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--models)
      MODELS="$2"
      shift 2
      ;;
    -f|--family)
      MODEL_FAMILY="$2"
      shift 2
      ;;
    -d|--device)
      DEVICE="$2"
      shift 2
      ;;
    -p|--precision)
      PRECISION="$2"
      shift 2
      ;;
    -b|--batch-sizes)
      BATCH_SIZES="$2"
      shift 2
      ;;
    -i|--iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -r|--report)
      REPORT="$2"
      shift 2
      ;;
    --report-format)
      REPORT_FORMAT="$2"
      shift 2
      ;;
    --db-path)
      DB_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      show_help
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build the command
CMD="python benchmark_openvino.py"

# Add database path if specified
if [ -n "$DB_PATH" ]; then
  CMD="$CMD --db-path $DB_PATH"
fi

# Add models or model family
if [ -n "$MODELS" ]; then
  CMD="$CMD --model $MODELS"
elif [ -n "$MODEL_FAMILY" ]; then
  CMD="$CMD --model-family $MODEL_FAMILY"
else
  # Default to text models if nothing specified
  CMD="$CMD --model-family text"
fi

# Add device
CMD="$CMD --device $DEVICE"

# Add precision
CMD="$CMD --precision $PRECISION"

# Add batch sizes
CMD="$CMD --batch-sizes $BATCH_SIZES"

# Add iterations
CMD="$CMD --iterations $ITERATIONS"

# Add output file
CMD="$CMD --output-file $OUTPUT_DIR/openvino_benchmark_${TIMESTAMP}.json"

# Add dry run flag if specified
if [ "$DRY_RUN" -eq 1 ]; then
  CMD="$CMD --dry-run"
fi

# Add report options if specified
if [ "$REPORT" -eq 1 ]; then
  CMD="$CMD --report --report-format $REPORT_FORMAT --report-file $OUTPUT_DIR/openvino_benchmark_${TIMESTAMP}.$REPORT_FORMAT"
fi

# Display the command
echo -e "${GREEN}Running benchmark with command:${NC}"
echo -e "${BLUE}$CMD${NC}"
echo ""

# Run the command
if [ "$DRY_RUN" -eq 1 ]; then
  echo -e "${YELLOW}Dry run mode - command will not be executed${NC}"
else
  echo -e "${GREEN}Starting benchmarks...${NC}"
  eval "$CMD"
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Benchmarks completed successfully!${NC}"
    echo -e "Results saved to: $OUTPUT_DIR/openvino_benchmark_${TIMESTAMP}.json"
    if [ "$REPORT" -eq 1 ]; then
      echo -e "Report saved to: $OUTPUT_DIR/openvino_benchmark_${TIMESTAMP}.$REPORT_FORMAT"
    fi
  else
    echo -e "${RED}Benchmarks failed!${NC}"
  fi
fi