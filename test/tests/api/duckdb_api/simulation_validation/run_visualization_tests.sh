#!/bin/bash
# Script to run visualization tests for the Simulation Accuracy and Validation Framework

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR=""
HTML_REPORT="false"
GENERATE_EXAMPLES="false"
TEST_TYPE="all"
VERBOSE="false"
INTERACTIVE="false"
SPECIFIC_TEST=""

# Display usage
function show_usage {
    echo -e "${BLUE}Simulation Validation Framework - Visualization Test Runner${NC}"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -o, --output-dir DIR       Specify output directory for test results and visualizations"
    echo "  -H, --html-report          Generate HTML report"
    echo "  -g, --generate-examples    Generate example visualizations"
    echo "  -t, --test-type TYPE       Specify test type: [all, connector, e2e, mape, hardware, time, drift, calibration, comprehensive]"
    echo "  -v, --verbose              Enable verbose output"
    echo "  -i, --interactive          Generate interactive visualizations"
    echo "  -s, --specific-test TEST   Run a specific test method (e.g., test_e2e_mape_comparison_chart)"
    echo
    echo "Examples:"
    echo "  $0 --generate-examples --output-dir ./vis_examples"
    echo "  $0 --test-type calibration --interactive"
    echo "  $0 --specific-test test_e2e_drift_visualization"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -H|--html-report)
            HTML_REPORT="true"
            shift
            ;;
        -g|--generate-examples)
            GENERATE_EXAMPLES="true"
            shift
            ;;
        -t|--test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -i|--interactive)
            INTERACTIVE="true"
            shift
            ;;
        -s|--specific-test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Validate test type
VALID_TEST_TYPES=("all" "connector" "e2e" "mape" "hardware" "time" "drift" "calibration" "comprehensive")
if [[ ! " ${VALID_TEST_TYPES[*]} " =~ " ${TEST_TYPE} " ]]; then
    echo -e "${RED}Invalid test type: ${TEST_TYPE}${NC}"
    echo -e "${YELLOW}Valid test types are: ${VALID_TEST_TYPES[*]}${NC}"
    exit 1
fi

# Build command
CMD="python -m unittest"

# Determine the specific test to run based on test-type
if [ -n "$SPECIFIC_TEST" ]; then
    # Run a specific test method
    CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.$SPECIFIC_TEST"
else
    case $TEST_TYPE in
        all)
            CMD="$CMD duckdb_api.simulation_validation.test_visualization_db_connector duckdb_api.simulation_validation.test_e2e_visualization_db_integration"
            ;;
        connector)
            CMD="$CMD duckdb_api.simulation_validation.test_visualization_db_connector"
            ;;
        e2e)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration"
            ;;
        mape)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_mape_comparison_chart"
            ;;
        hardware)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_hardware_comparison_heatmap"
            ;;
        time)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_time_series_chart"
            ;;
        drift)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_drift_visualization"
            ;;
        calibration)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_calibration_improvement_chart"
            ;;
        comprehensive)
            CMD="$CMD duckdb_api.simulation_validation.test_e2e_visualization_db_integration.TestE2EVisualizationDBIntegration.test_e2e_comprehensive_dashboard"
            ;;
    esac
fi

# If we're generating examples, use run_e2e_tests.py
if [ "$GENERATE_EXAMPLES" = "true" ]; then
    CMD="python duckdb_api/simulation_validation/run_e2e_tests.py --generate-examples"
    
    if [ -n "$OUTPUT_DIR" ]; then
        CMD="$CMD --output-dir $OUTPUT_DIR"
    fi
    
    if [ "$HTML_REPORT" = "true" ]; then
        CMD="$CMD --html-report"
    fi
    
    if [ "$VERBOSE" = "true" ]; then
        CMD="$CMD --verbose"
    fi
else
    # Set environment variables for test settings
    if [ -n "$OUTPUT_DIR" ]; then
        export TEST_OUTPUT_DIR="$OUTPUT_DIR"
    fi
    
    if [ "$INTERACTIVE" = "true" ]; then
        export TEST_INTERACTIVE="true"
    fi
    
    # For verbose mode, add -v to unittest
    if [ "$VERBOSE" = "true" ]; then
        CMD="$CMD -v"
    fi
fi

# Create output dir if specified and doesn't exist
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Get the start time
START_TIME=$(date +%s)

# Display what we're doing
echo -e "${BLUE}Running Simulation Validation Framework visualization tests...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}"
echo

# Run the command
echo -e "${CYAN}Test Output:${NC}"
echo -e "${CYAN}$(printf '=%.0s' {1..80})${NC}"
$CMD
RESULT=$?
echo -e "${CYAN}$(printf '=%.0s' {1..80})${NC}"

# Get the end time
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Format runtime as minutes and seconds
MINUTES=$((RUNTIME / 60))
SECONDS=$((RUNTIME % 60))

# Display result
echo
if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}All visualization tests passed successfully!${NC}"
else
    echo -e "${RED}Tests failed with exit code $RESULT${NC}"
fi
echo -e "${BLUE}Total runtime: ${MINUTES}m ${SECONDS}s${NC}"

# Show output directory if specified
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Output files saved to: ${OUTPUT_DIR}${NC}"
    # List files if not too many
    FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
    if [ "$FILE_COUNT" -lt 20 ]; then
        echo -e "${YELLOW}Generated files:${NC}"
        find "$OUTPUT_DIR" -type f -name "*.html" | while read -r file; do
            echo "  - $file"
        done
    else
        echo -e "${YELLOW}Generated $FILE_COUNT files in output directory${NC}"
    fi
fi

# Return the result
exit $RESULT