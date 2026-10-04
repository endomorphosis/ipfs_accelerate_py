#!/bin/bash
# End-to-End Web Dashboard Integration Test Runner
# This script runs the end-to-end integration test for the Web Dashboard

# Set default values
DB_PATH="./e2e_test_results.duckdb"
COORDINATOR_PORT=8081
DASHBOARD_PORT=8050
NUM_WORKERS=5
NUM_TASKS=50
DEBUG=""
GENERATE_ANOMALIES=""
GENERATE_TRENDS=""
OPEN_BROWSER=""
UPDATE_INTERVAL=5

# Display usage information
function show_usage {
    echo "Usage: $0 [OPTIONS]"
    echo "Run the end-to-end integration test for the Result Aggregator Web Dashboard"
    echo ""
    echo "Options:"
    echo "  --db-path PATH          Path to DuckDB database (default: ./e2e_test_results.duckdb)"
    echo "  --coordinator-port PORT Port for coordinator (default: 8081)"
    echo "  --dashboard-port PORT   Port for web dashboard (default: 8050)"
    echo "  --num-workers NUM       Number of simulated workers (default: 5)"
    echo "  --num-tasks NUM         Number of simulated tasks (default: 50)"
    echo "  --update-interval SEC   Interval in seconds for WebSocket updates (default: 5)"
    echo "  --generate-anomalies    Generate anomalous test results"
    echo "  --generate-trends       Generate performance trends"
    echo "  --quick                 Run a quick test with fewer workers and tasks"
    echo "  --debug                 Enable debug mode"
    echo "  --open-browser          Open web browser when dashboard is ready"
    echo "  --help                  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --generate-anomalies --generate-trends --open-browser"
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --coordinator-port)
            COORDINATOR_PORT="$2"
            shift 2
            ;;
        --dashboard-port)
            DASHBOARD_PORT="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --update-interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --generate-anomalies)
            GENERATE_ANOMALIES="--generate-anomalies"
            shift
            ;;
        --generate-trends)
            GENERATE_TRENDS="--generate-trends"
            shift
            ;;
        --quick)
            NUM_WORKERS=3
            NUM_TASKS=20
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --open-browser)
            OPEN_BROWSER="--open-browser"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if any ports are already in use
function check_port {
    if nc -z localhost $1 >/dev/null 2>&1; then
        echo "Error: Port $1 is already in use. Please choose a different port."
        exit 1
    fi
}

check_port $COORDINATOR_PORT
check_port $DASHBOARD_PORT

# Ensure the directory for the database exists
DB_DIR=$(dirname "$DB_PATH")
if [ ! -z "$DB_DIR" ] && [ "$DB_DIR" != "." ]; then
    mkdir -p "$DB_DIR"
fi

# Run the test
echo "Running end-to-end integration test with the following parameters:"
echo "  Database Path: $DB_PATH"
echo "  Coordinator Port: $COORDINATOR_PORT"
echo "  Dashboard Port: $DASHBOARD_PORT"
echo "  Number of Workers: $NUM_WORKERS"
echo "  Number of Tasks: $NUM_TASKS"
echo "  Generate Anomalies: $([ -z "$GENERATE_ANOMALIES" ] && echo "No" || echo "Yes")"
echo "  Generate Trends: $([ -z "$GENERATE_TRENDS" ] && echo "No" || echo "Yes")"
echo "  Debug Mode: $([ -z "$DEBUG" ] && echo "No" || echo "Yes")"
echo "  Open Browser: $([ -z "$OPEN_BROWSER" ] && echo "No" || echo "Yes")"
echo "  WebSocket Update Interval: $UPDATE_INTERVAL seconds"
echo ""

# Create reports directory if it doesn't exist
mkdir -p reports

# Run the Python script
python run_e2e_web_dashboard_integration.py \
    --db-path "$DB_PATH" \
    --coordinator-port "$COORDINATOR_PORT" \
    --dashboard-port "$DASHBOARD_PORT" \
    --num-workers "$NUM_WORKERS" \
    --num-tasks "$NUM_TASKS" \
    --update-interval "$UPDATE_INTERVAL" \
    $GENERATE_ANOMALIES \
    $GENERATE_TRENDS \
    $DEBUG \
    $OPEN_BROWSER