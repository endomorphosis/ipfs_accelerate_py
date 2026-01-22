#!/bin/bash

# Run Benchmark Dashboard
# This script starts the interactive dashboard for benchmark visualization

# Default values
PORT=8050
API_URL="http://localhost:8000"
DB_PATH="./benchmark_db.duckdb"
DEBUG=false

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Start the Benchmark Dashboard for visualizing benchmark results."
    echo
    echo "Options:"
    echo "  -p, --port PORT         Port to listen on (default: 8050)"
    echo "  -a, --api-url URL       URL of the benchmark API server (default: http://localhost:8000)"
    echo "  -d, --db-path PATH      Path to DuckDB database (default: ./benchmark_db.duckdb)"
    echo "  --debug                 Enable debug mode"
    echo "  --help                  Show this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --port 8080 --api-url http://benchmarkserver:8000 --db-path /path/to/benchmarks.duckdb"
    echo
    echo "Note:"
    echo "  This dashboard requires the benchmark API server to be running."
    echo "  You can start the API server with: ./run_benchmark_api_server.sh"
    echo
    echo "Dependencies:"
    echo "  This script requires dash, dash-bootstrap-components, plotly, pandas to be installed."
    echo "  You can install them with: pip install dash dash-bootstrap-components plotly pandas"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -a|--api-url)
            API_URL="$2"
            shift 2
            ;;
        -d|--db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
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

# Check if required packages are installed
pip list | grep -q "dash" || {
    echo "Error: dash package not installed. Please install required packages:"
    echo "pip install dash dash-bootstrap-components plotly pandas"
    exit 1
}

# Check if the benchmark API server is running
if ! curl -s "$API_URL/api/benchmark/models" > /dev/null; then
    echo "Warning: Could not connect to the benchmark API server at $API_URL"
    echo "The dashboard may not display data properly."
    echo "Start the API server with: ./run_benchmark_api_server.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Construct debug flag if needed
DEBUG_FLAG=""
if [[ "$DEBUG" == true ]]; then
    DEBUG_FLAG="--debug"
fi

# Start the dashboard
echo "Starting Benchmark Dashboard on http://localhost:$PORT"
echo "Using API server: $API_URL"
echo "Using database: $DB_PATH"
echo 
echo "Press Ctrl+C to stop the dashboard"
echo

# Check if benchmark_db.duckdb exists, create directory if needed
mkdir -p "$(dirname "$DB_PATH")"

python benchmark_dashboard.py --port "$PORT" --api-url "$API_URL" --db-path "$DB_PATH" $DEBUG_FLAG