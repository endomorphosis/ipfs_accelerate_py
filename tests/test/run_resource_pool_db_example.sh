#!/bin/bash
# Script to run the WebGPU/WebNN Resource Pool Database Integration Example
#
# This script runs the resource_pool_db_example.py script that demonstrates
# the DuckDB integration with the WebGPU/WebNN Resource Pool.
#
# Usage:
#   ./run_resource_pool_db_example.sh [options]
#
# Options:
#   --html       Generate HTML report (default)
#   --markdown   Generate Markdown report
#   --json       Generate JSON report
#   --visualize  Create visualizations
#   --model NAME Specific model to analyze
#   --browser BR Specific browser to analyze
#   --days DAYS  Number of days to include (default: 30)
#   --db PATH    Path to DuckDB database

set -e

# Set default values
FORMAT="html"
VISUALIZE=""
MODEL=""
BROWSER=""
DAYS="30"
DB_PATH="${BENCHMARK_DB_PATH:-benchmark_db.duckdb}"
OUTPUT_DIR="./reports"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --html)
      FORMAT="html"
      shift
      ;;
    --markdown)
      FORMAT="markdown"
      shift
      ;;
    --json)
      FORMAT="json"
      shift
      ;;
    --visualize)
      VISUALIZE="--visualize"
      shift
      ;;
    --model)
      MODEL="--model $2"
      shift
      shift
      ;;
    --browser)
      BROWSER="--browser $2"
      shift
      shift
      ;;
    --days)
      DAYS="$2"
      shift
      shift
      ;;
    --db)
      DB_PATH="$2"
      shift
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "┌───────────────────────────────────────────────────────────────┐"
echo "│ WebGPU/WebNN Resource Pool Database Integration Example       │"
echo "│ (March 12, 2025)                                             │"
echo "└───────────────────────────────────────────────────────────────┘"
echo ""
echo "Settings:"
echo "  - Report format: $FORMAT"
echo "  - Database path: $DB_PATH"
echo "  - Days to include: $DAYS"
echo "  - Output directory: $OUTPUT_DIR"
if [ -n "$MODEL" ]; then
  echo "  - Model filter: $(echo $MODEL | cut -d ' ' -f 2)"
fi
if [ -n "$BROWSER" ]; then
  echo "  - Browser filter: $(echo $BROWSER | cut -d ' ' -f 2)"
fi
if [ -n "$VISUALIZE" ]; then
  echo "  - Visualizations: Enabled"
fi
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set environment variable for database path
export BENCHMARK_DB_PATH="$DB_PATH"

# Run the example script
echo "Running resource pool database integration example..."
echo ""

python examples/resource_pool_db_example.py \
  --report-format "$FORMAT" \
  --days "$DAYS" \
  --output-dir "$OUTPUT_DIR" \
  $VISUALIZE $MODEL $BROWSER

echo ""
echo "Example completed!"
echo ""
echo "You can find the generated reports in: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review the generated reports to understand performance patterns"
echo "  2. Analyze browser-specific performance for different model types"
echo "  3. Look for optimization opportunities based on the data"
echo "  4. Explore trend analysis and regression detection"
echo ""
echo "For more information, see WEB_RESOURCE_POOL_DB_INTEGRATION.md"