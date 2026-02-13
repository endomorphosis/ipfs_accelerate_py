#!/bin/bash

# Run Benchmark API Server
# This script starts the FastAPI server for benchmark monitoring and control

# Default values
PORT=8000
HOST="0.0.0.0"
DB_PATH="./benchmark_db.duckdb"
RESULTS_DIR="./benchmark_results"

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Start the Benchmark API Server with WebSocket support."
    echo
    echo "Options:"
    echo "  -p, --port PORT         Port to listen on (default: 8000)"
    echo "  -h, --host HOST         Host to bind to (default: 0.0.0.0)"
    echo "  -d, --db-path PATH      Path to DuckDB database (default: ./benchmark_db.duckdb)"
    echo "  -r, --results-dir DIR   Directory for benchmark results (default: ./benchmark_results)"
    echo "  --help                  Show this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --port 8888 --db-path /path/to/benchmarks.duckdb"
    echo
    echo "API Endpoints:"
    echo "  POST /api/benchmark/run                  Start a new benchmark run"
    echo "  GET  /api/benchmark/status/{run_id}      Get status of a benchmark run"
    echo "  GET  /api/benchmark/results/{run_id}     Get results of a completed benchmark"
    echo "  GET  /api/benchmark/models               List available models"
    echo "  GET  /api/benchmark/hardware             List available hardware platforms"
    echo "  GET  /api/benchmark/reports              List available benchmark reports"
    echo "  GET  /api/benchmark/query                Query benchmark results with filters"
    echo "  WS   /api/benchmark/ws/{run_id}          WebSocket for real-time updates"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -d|--db-path)
            DB_PATH="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make sure directories exist
mkdir -p "$(dirname "$DB_PATH")"
mkdir -p "$RESULTS_DIR"

# Start the server
echo "Starting Benchmark API Server on http://$HOST:$PORT"
echo "Database path: $DB_PATH"
echo "Results directory: $RESULTS_DIR"
echo 
echo "Press Ctrl+C to stop the server"
echo "Use 'curl' or any HTTP client to interact with the API"
echo "Example: curl -X GET http://localhost:$PORT/api/benchmark/models"
echo

exec python benchmark_api_server.py \
    --port "$PORT" \
    --host "$HOST" \
    --db-path "$DB_PATH" \
    --results-dir "$RESULTS_DIR"