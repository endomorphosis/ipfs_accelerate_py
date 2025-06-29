#!/bin/bash
# Script to run end-to-end tests for CI/CD environments

set -e  # Exit on error

# Display help message
function show_help {
    echo "Usage: ./run_e2e_ci_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-family FAMILY    Model family to test (text-embedding, text-generation, vision, audio, multimodal, all)"
    echo "  --hardware HARDWARE      Hardware platforms to test, comma-separated (e.g., cpu,cuda)"
    echo "  --distributed            Use distributed testing"
    echo "  --no-distributed         Disable distributed testing"
    echo "  --update-expected        Update expected results"
    echo "  --ci                     Run in CI mode with enhanced reporting"
    echo "  --workers N              Number of worker threads (default: 4)"
    echo "  --db-path PATH           Custom database path"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_e2e_ci_tests.sh --model-family text-embedding --hardware cpu,cuda"
    echo "  ./run_e2e_ci_tests.sh --model-family all --priority-hardware --distributed"
    echo ""
}

# Default values
MODEL_FAMILY=""
HARDWARE=""
DISTRIBUTED="--distributed"
UPDATE_EXPECTED=""
CI="--ci"
WORKERS="--workers 4"
DB_PATH=""
USE_DB="--use-db"
SIMULATION_AWARE="--simulation-aware"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model-family)
            MODEL_FAMILY="--model-family $2"
            shift
            shift
            ;;
        --all-models)
            MODEL_FAMILY="--all-models"
            shift
            ;;
        --hardware)
            HARDWARE="--hardware $2"
            shift
            shift
            ;;
        --priority-hardware)
            HARDWARE="--priority-hardware"
            shift
            ;;
        --distributed)
            DISTRIBUTED="--distributed"
            shift
            ;;
        --no-distributed)
            DISTRIBUTED=""
            shift
            ;;
        --update-expected)
            UPDATE_EXPECTED="--update-expected"
            shift
            ;;
        --ci)
            CI="--ci"
            shift
            ;;
        --no-ci)
            CI=""
            shift
            ;;
        --workers)
            WORKERS="--workers $2"
            shift
            shift
            ;;
        --db-path)
            DB_PATH="--db-path $2"
            shift
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

# Validate arguments
if [ -z "$MODEL_FAMILY" ]; then
    echo "Error: Model family is required (--model-family or --all-models)"
    show_help
    exit 1
fi

if [ -z "$HARDWARE" ]; then
    echo "Error: Hardware specification is required (--hardware or --priority-hardware)"
    show_help
    exit 1
fi

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure the directories exist
mkdir -p generators/expected_results
mkdir -p generators/collected_results
mkdir -p generators/model_documentation

# Run the end-to-end tests
echo "Running end-to-end tests with: $MODEL_FAMILY $HARDWARE $DISTRIBUTED $UPDATE_EXPECTED $CI $WORKERS $DB_PATH $USE_DB $SIMULATION_AWARE"
python generators/runners/end_to_end/run_e2e_tests.py \
    $MODEL_FAMILY \
    $HARDWARE \
    $DISTRIBUTED \
    $UPDATE_EXPECTED \
    $CI \
    $WORKERS \
    $DB_PATH \
    $USE_DB \
    $SIMULATION_AWARE

# Get and display the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "End-to-end tests completed successfully."
else
    echo "End-to-end tests failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE