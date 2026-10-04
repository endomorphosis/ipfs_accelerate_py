#!/bin/bash
# End-to-End Test Runner for DRM External Monitoring Integration
#
# This script sets up the environment and runs the end-to-end tests for
# the DRM External Monitoring integration with Prometheus and Grafana.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check dependencies
echo "Checking dependencies..."
MISSING=0

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker is required but not installed."
    MISSING=1
fi

# Check Python dependencies
PYTHON_DEPS=("prometheus_client" "requests" "dash" "dash_bootstrap_components" "plotly")
for dep in "${PYTHON_DEPS[@]}"; do
    if ! python3 -c "import $dep" &> /dev/null; then
        echo "Python package '$dep' is required but not installed."
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "Missing dependencies. Please install required packages:"
    echo "pip install -r requirements_dashboard.txt"
    echo "And make sure Docker is installed."
    exit 1
fi

# Check port availability
PORTS=(9191 9292 9393 9494)
for port in "${PORTS[@]}"; do
    if nc -z localhost $port 2>/dev/null; then
        echo "Port $port is already in use. Please free this port before running the test."
        exit 1
    fi
done

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/e2e_test_output"
mkdir -p "$OUTPUT_DIR"

# Run the tests
echo "Running end-to-end tests..."
python3 -m duckdb_api.distributed_testing.tests.test_drm_external_monitoring_e2e -v 2>&1 | tee "$OUTPUT_DIR/test_output.log"

TEST_RESULT=$?

# Output test report
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n========================================="
    echo "✅ End-to-End Tests PASSED"
    echo "========================================="
else
    echo -e "\n========================================="
    echo "❌ End-to-End Tests FAILED"
    echo "See log for details: $OUTPUT_DIR/test_output.log"
    echo "========================================="
fi

echo "End-to-end test report generated at: $OUTPUT_DIR/test_output.log"

# Cleanup any lingering containers
echo "Cleaning up test containers..."
docker rm -f test_prometheus_drm test_grafana_drm 2>/dev/null || true

exit $TEST_RESULT