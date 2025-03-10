#!/bin/bash
# Local benchmark runner that mimics the CI workflow
# This script allows testing the CI benchmark database integration locally

set -e

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Local Benchmark With CI Database Integration"
  echo "This script runs benchmarks locally and stores results in a DuckDB database,"
  echo "mimicking the CI workflow defined in benchmark_db_ci.yml."
  echo ""
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --model MODEL       Model to benchmark (default: bert-base-uncased)"
  echo "  --hardware HW       Hardware to use (cpu, cuda, rocm, etc.) (default: cpu)"
  echo "  --batch-sizes SIZES Comma-separated list of batch sizes (default: 1,2,4,8)"
  echo "  --iterations NUM    Number of iterations to run (default: 20)"
  echo "  --warmup NUM        Number of warmup iterations (default: 5)"
  echo "  --output-dir DIR    Output directory for database and reports (default: ./benchmark_results)"
  echo "  --simulate          Simulate benchmarks without actually running them"
  echo "  --help, -h          Display this help message"
  exit 0
fi

# Default values
MODEL="bert-base-uncased"
HARDWARE="cpu"
BATCH_SIZES="1,2,4,8"
ITERATIONS=20
WARMUP=5
OUTPUT_DIR="./benchmark_results"
SIMULATE=""

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --hardware)
      HARDWARE="$2"
      shift
      shift
      ;;
    --batch-sizes)
      BATCH_SIZES="$2"
      shift
      shift
      ;;
    --iterations)
      ITERATIONS="$2"
      shift
      shift
      ;;
    --warmup)
      WARMUP="$2"
      shift
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --simulate)
      SIMULATE="--simulate"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate run ID
RUN_ID=$(date +'%Y%m%d%H%M%S')
TODAY=$(date +'%Y%m%d')

echo "=============================="
echo "Local Benchmark Configuration"
echo "=============================="
echo "Model: $MODEL"
echo "Hardware: $HARDWARE"
echo "Batch sizes: $BATCH_SIZES"
echo "Iterations: $ITERATIONS"
echo "Warmup: $WARMUP"
echo "Output directory: $OUTPUT_DIR"
echo "Run ID: $RUN_ID"
if [ -n "$SIMULATE" ]; then
  echo "Mode: SIMULATION"
fi
echo "=============================="

# Stage 1: Setup database
echo "Setting up database..."
DB_PATH="$OUTPUT_DIR/benchmark_$RUN_ID.duckdb"
python scripts/create_benchmark_schema.py --output "$DB_PATH" --sample-data

# Generate CI metadata
echo "Generating CI metadata..."
cat > "$OUTPUT_DIR/ci_metadata.json" << EOF
{
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "local_run": true
}
EOF

# Get git info
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Stage 2: Run benchmark
echo "Running benchmark..."
python run_benchmark_with_db.py \
  --db "$DB_PATH" \
  --model "$MODEL" \
  --hardware "$HARDWARE" \
  --batch-sizes "$BATCH_SIZES" \
  --iterations "$ITERATIONS" \
  --warmup "$WARMUP" \
  --commit "$COMMIT" \
  --branch "$BRANCH" \
  $SIMULATE

# Stage 3: Generate reports
echo "Generating reports..."
python scripts/benchmark_db_query.py \
  --db "$DB_PATH" \
  --report performance \
  --format html \
  --output "$OUTPUT_DIR/performance_report_$TODAY.html"

python scripts/benchmark_db_query.py \
  --db "$DB_PATH" \
  --compatibility-matrix \
  --format html \
  --output "$OUTPUT_DIR/compatibility_matrix_$TODAY.html"

python scripts/benchmark_db_query.py \
  --db "$DB_PATH" \
  --hardware-comparison-chart \
  --metric throughput \
  --output "$OUTPUT_DIR/hardware_comparison_$TODAY.png"

python scripts/benchmark_db_query.py \
  --db "$DB_PATH" \
  --summary \
  --format json \
  --output "$OUTPUT_DIR/benchmark_summary_$TODAY.json"

# Create report index
echo "Creating report index..."
cat > "$OUTPUT_DIR/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Hardware Benchmark Reports</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Hardware Benchmark Reports</h1>
        <p>Benchmark run: $TODAY (Run ID: $RUN_ID)</p>
        
        <h2>Reports</h2>
        <ul>
            <li><a href="performance_report_$TODAY.html">Performance Report</a></li>
            <li><a href="compatibility_matrix_$TODAY.html">Compatibility Matrix</a></li>
            <li><a href="hardware_comparison_$TODAY.png">Hardware Comparison Chart</a></li>
            <li><a href="benchmark_summary_$TODAY.json">Benchmark Summary (JSON)</a></li>
        </ul>
    </div>
</body>
</html>
EOF

echo "=============================="
echo "Benchmark completed successfully!"
echo "=============================="
echo "Results stored in: $OUTPUT_DIR"
echo "Database file: $DB_PATH"
echo "Report index: $OUTPUT_DIR/index.html"
echo "=============================="