#!/bin/bash
# Enhanced Test Runner for Simulation Accuracy and Validation Framework
# This script runs the comprehensive test suite with support for CI/CD integration

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="./validation_output"
RUN_HTML="false"
RUN_DB="false"
RUN_CONNECTOR="false"
RUN_E2E="false"
SKIP_LONG="false"
VERBOSE="false"
GENERATE_EXAMPLES="false"
TEST_TYPE="all"
HARDWARE_PROFILE="all"
RUN_ANALYSIS="false"
RUN_COVERAGE="false"
RUN_ISSUES="false"
RUN_DASHBOARD="false"

# Display usage
function show_usage {
    echo -e "${BLUE}Simulation Validation Framework - Enhanced Test Runner${NC}"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                    Show this help message"
    echo "  -o, --output-dir DIR          Specify output directory"
    echo "  --html-report                 Generate HTML report"
    echo "  -d, --database                Run database integration tests only"
    echo "  -c, --connector               Run visualization connector tests only"
    echo "  --run-e2e                     Run end-to-end tests only"
    echo "  -s, --skip-long               Skip long-running tests"
    echo "  -v, --verbose                 Enable verbose output"
    echo "  --generate-examples           Generate example visualizations"
    echo "  --test-type TYPE              Type of test to run (all, unit, e2e, calibration, drift, visualization)"
    echo "  --hardware-profile PROFILE    Hardware profile to validate (all, cpu, gpu, webgpu)"
    echo "  --run-analysis                Run validation results analysis"
    echo "  --run-coverage                Run test coverage analysis"
    echo "  --detect-issues               Run validation issue detection"
    echo "  --create-dashboard            Create comprehensive dashboard"
    echo
    echo "If no test suite is specified, all tests will be run."
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
        --html-report)
            RUN_HTML="true"
            shift
            ;;
        -d|--database)
            RUN_DB="true"
            shift
            ;;
        -c|--connector)
            RUN_CONNECTOR="true"
            shift
            ;;
        --run-e2e)
            RUN_E2E="true"
            shift
            ;;
        -s|--skip-long)
            SKIP_LONG="true"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        --generate-examples)
            GENERATE_EXAMPLES="true"
            shift
            ;;
        --test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --hardware-profile)
            HARDWARE_PROFILE="$2"
            shift 2
            ;;
        --run-analysis)
            RUN_ANALYSIS="true"
            shift
            ;;
        --run-coverage)
            RUN_COVERAGE="true"
            shift
            ;;
        --detect-issues)
            RUN_ISSUES="true"
            shift
            ;;
        --create-dashboard)
            RUN_DASHBOARD="true"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/reports"
mkdir -p "$OUTPUT_DIR/examples"

# Generate a run ID
RUN_ID=$(date +%Y%m%d%H%M%S)

# Print run information
echo -e "${BLUE}========================================"
echo -e "Simulation Validation Enhanced Test Runner"
echo -e "========================================${NC}"
echo -e "${YELLOW}Run ID:${NC} $RUN_ID"
echo -e "${YELLOW}Test Type:${NC} $TEST_TYPE"
echo -e "${YELLOW}Hardware Profile:${NC} $HARDWARE_PROFILE"
echo -e "${YELLOW}Output Directory:${NC} $OUTPUT_DIR"
echo -e "${BLUE}========================================${NC}"

# Get the start time
START_TIME=$(date +%s)

# Run unit tests if appropriate
if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "unit" ]]; then
    echo -e "${BLUE}Running unit tests...${NC}"
    
    # Build pytest command
    PYTEST_CMD="python -m pytest duckdb_api/simulation_validation/ -v"
    
    # Add coverage report options
    if [[ "$RUN_HTML" == "true" ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov=duckdb_api/simulation_validation --cov-report=xml:\"$OUTPUT_DIR/reports/coverage.xml\" --cov-report=html:\"$OUTPUT_DIR/reports/htmlcov\""
    else
        PYTEST_CMD="$PYTEST_CMD --cov=duckdb_api/simulation_validation --cov-report=xml:\"$OUTPUT_DIR/reports/coverage.xml\""
    fi
    
    # Add JUnit XML output
    PYTEST_CMD="$PYTEST_CMD --junitxml=\"$OUTPUT_DIR/reports/test-results.xml\""
    
    # Run pytest
    echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
    eval $PYTEST_CMD
    echo -e "${GREEN}Unit tests completed${NC}"
fi

# Build E2E test command
if [[ "$RUN_E2E" == "true" || "$TEST_TYPE" == "e2e" ]]; then
    echo -e "${BLUE}Running end-to-end tests...${NC}"
    
    # Build E2E command
    E2E_CMD="python -m duckdb_api.simulation_validation.run_e2e_tests"
    E2E_CMD="$E2E_CMD --hardware-profile $HARDWARE_PROFILE"
    E2E_CMD="$E2E_CMD --run-id $RUN_ID"
    E2E_CMD="$E2E_CMD --output-dir $OUTPUT_DIR"
    
    if [[ "$RUN_HTML" == "true" ]]; then
        E2E_CMD="$E2E_CMD --html-report"
    fi
    
    if [[ "$RUN_DB" == "true" ]]; then
        E2E_CMD="$E2E_CMD --run-db"
    fi
    
    if [[ "$RUN_CONNECTOR" == "true" ]]; then
        E2E_CMD="$E2E_CMD --run-connector"
    fi
    
    if [[ "$SKIP_LONG" == "true" ]]; then
        E2E_CMD="$E2E_CMD --skip-long-tests"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        E2E_CMD="$E2E_CMD --verbose"
    fi
    
    # Run E2E tests
    echo -e "${YELLOW}Command: $E2E_CMD${NC}"
    eval $E2E_CMD
    echo -e "${GREEN}End-to-end tests completed${NC}"
fi

# Generate examples if requested
if [[ "$GENERATE_EXAMPLES" == "true" ]]; then
    echo -e "${BLUE}Generating examples...${NC}"
    
    # Build example generator command
    GEN_CMD="python -m duckdb_api.simulation_validation.test.test_data_generator"
    GEN_CMD="$GEN_CMD --hardware-profile $HARDWARE_PROFILE"
    GEN_CMD="$GEN_CMD --output-dir $OUTPUT_DIR/examples"
    GEN_CMD="$GEN_CMD --run-id $RUN_ID"
    
    # Run example generator
    echo -e "${YELLOW}Command: $GEN_CMD${NC}"
    eval $GEN_CMD
    echo -e "${GREEN}Examples generated${NC}"
    
    # Generate visualizations from examples if dashboard is requested
    if [[ "$RUN_DASHBOARD" == "true" ]]; then
        echo -e "${BLUE}Generating examples dashboard...${NC}"
        
        DASH_CMD="python -m duckdb_api.simulation_validation.visualization.generate_dashboard"
        DASH_CMD="$DASH_CMD --input-dir $OUTPUT_DIR/examples"
        DASH_CMD="$DASH_CMD --output-dir $OUTPUT_DIR/examples/dashboard"
        DASH_CMD="$DASH_CMD --run-id $RUN_ID"
        DASH_CMD="$DASH_CMD --interactive"
        DASH_CMD="$DASH_CMD --title \"Simulation Validation Examples Dashboard\""
        
        echo -e "${YELLOW}Command: $DASH_CMD${NC}"
        eval $DASH_CMD
        echo -e "${GREEN}Examples dashboard created${NC}"
    fi
fi

# Analyze test coverage if available and requested
if [[ -f "$OUTPUT_DIR/reports/coverage.xml" && ("$RUN_COVERAGE" == "true" || "$RUN_ANALYSIS" == "true") ]]; then
    echo -e "${BLUE}Analyzing test coverage...${NC}"
    
    # Build coverage analyzer command
    COV_CMD="python -m duckdb_api.simulation_validation.analyze_test_coverage"
    COV_CMD="$COV_CMD --coverage-file $OUTPUT_DIR/reports/coverage.xml"
    
    # Generate HTML report
    echo -e "${YELLOW}Generating HTML coverage report...${NC}"
    eval "$COV_CMD --output-format html --output-file $OUTPUT_DIR/reports/coverage_report.html"
    
    # Generate Markdown report
    echo -e "${YELLOW}Generating Markdown coverage report...${NC}"
    eval "$COV_CMD --output-format markdown --output-file $OUTPUT_DIR/reports/coverage_report.md"
    
    echo -e "${GREEN}Coverage analysis completed${NC}"
fi

# Analyze validation results if available and requested
if [[ -d "$OUTPUT_DIR" && ("$RUN_ANALYSIS" == "true" || "$RUN_ISSUES" == "true") ]]; then
    echo -e "${BLUE}Analyzing validation results...${NC}"
    
    # Build validation analyzer command
    VAL_CMD="python -m duckdb_api.simulation_validation.analyze_validation_results"
    VAL_CMD="$VAL_CMD --results-dir $OUTPUT_DIR"
    VAL_CMD="$VAL_CMD --run-id $RUN_ID"
    
    # Generate HTML report
    echo -e "${YELLOW}Generating HTML validation analysis...${NC}"
    eval "$VAL_CMD --output-format html --output-file $OUTPUT_DIR/reports/validation_analysis.html"
    
    # Generate Markdown report
    echo -e "${YELLOW}Generating Markdown validation analysis...${NC}"
    eval "$VAL_CMD --output-format markdown --output-file $OUTPUT_DIR/reports/validation_analysis.md"
    
    echo -e "${GREEN}Validation analysis completed${NC}"
    
    # Detect validation issues if requested
    if [[ "$RUN_ISSUES" == "true" ]]; then
        echo -e "${BLUE}Detecting validation issues...${NC}"
        
        # Build issue detector command
        ISS_CMD="python -m duckdb_api.simulation_validation.detect_validation_issues"
        ISS_CMD="$ISS_CMD --results-dir $OUTPUT_DIR"
        ISS_CMD="$ISS_CMD --threshold 0.1"
        
        # Generate HTML report
        echo -e "${YELLOW}Generating HTML issues report...${NC}"
        eval "$ISS_CMD --output-format html --output-file $OUTPUT_DIR/reports/validation_issues.html"
        
        # Generate Markdown report
        echo -e "${YELLOW}Generating Markdown issues report...${NC}"
        eval "$ISS_CMD --output-format markdown --output-file $OUTPUT_DIR/reports/validation_issues.md"
        
        echo -e "${GREEN}Issue detection completed${NC}"
    fi
fi

# Create comprehensive dashboard if requested
if [[ "$RUN_DASHBOARD" == "true" ]]; then
    echo -e "${BLUE}Creating comprehensive dashboard...${NC}"
    
    # Build dashboard generator command
    DASH_CMD="python -m duckdb_api.simulation_validation.visualization.generate_dashboard"
    DASH_CMD="$DASH_CMD --input-dir $OUTPUT_DIR"
    DASH_CMD="$DASH_CMD --output-dir $OUTPUT_DIR/dashboard"
    DASH_CMD="$DASH_CMD --run-id $RUN_ID"
    DASH_CMD="$DASH_CMD --interactive"
    DASH_CMD="$DASH_CMD --title \"Simulation Validation Dashboard - $(date -Iseconds)\""
    
    echo -e "${YELLOW}Command: $DASH_CMD${NC}"
    eval $DASH_CMD
    
    # Create index.html for the dashboard
    echo -e "${YELLOW}Creating dashboard index...${NC}"
    cat > "$OUTPUT_DIR/dashboard/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Simulation Validation Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Simulation Validation Dashboard</h1>
        <p>Latest validation run: $(date -Iseconds)</p>
        <p>Run ID: $RUN_ID</p>
        
        <h2>Latest Reports</h2>
        <ul>
            <li><a href="validation_report.html">Validation Report</a></li>
            <li><a href="calibration_report.html">Calibration Report</a></li>
            <li><a href="drift_detection_report.html">Drift Detection Report</a></li>
            <li><a href="visualization_gallery.html">Visualization Gallery</a></li>
            <li><a href="hardware_profiles.html">Hardware Profiles</a></li>
        </ul>
        
        <h2>Performance Analysis</h2>
        <p>The simulation validation framework includes performance analysis for simulation vs hardware:</p>
        <ul>
            <li><a href="performance_analysis.html">Performance Analysis Dashboard</a></li>
            <li><a href="metrics_comparison.html">Metrics Comparison</a></li>
        </ul>
        
        <h2>View Reports</h2>
        <p>
            <a href="../reports/validation_analysis.html" class="btn btn-primary">
                Validation Analysis
            </a>
            <a href="../reports/validation_issues.html" class="btn btn-primary">
                Validation Issues
            </a>
            <a href="../reports/htmlcov/index.html" class="btn btn-primary">
                Coverage Report
            </a>
        </p>
    </div>
</body>
</html>
EOF
    
    echo -e "${GREEN}Dashboard created${NC}"
fi

# Create a summary report
echo -e "${BLUE}Creating summary report...${NC}"
cat > "$OUTPUT_DIR/summary.md" << EOF
# Simulation Validation Test Summary

## Run Information
- Run ID: $RUN_ID
- Test Type: $TEST_TYPE
- Hardware Profile: $HARDWARE_PROFILE
- Run Time: $(date -Iseconds)

## Test Results
EOF

if [[ -f "$OUTPUT_DIR/reports/test-results.xml" ]]; then
    # Extract test summary using xmllint if available
    if command -v xmllint > /dev/null; then
        TOTAL_TESTS=$(xmllint --xpath "string(/testsuites/@tests)" "$OUTPUT_DIR/reports/test-results.xml")
        FAILURES=$(xmllint --xpath "string(/testsuites/@failures)" "$OUTPUT_DIR/reports/test-results.xml")
        ERRORS=$(xmllint --xpath "string(/testsuites/@errors)" "$OUTPUT_DIR/reports/test-results.xml")
        SKIPPED=$(xmllint --xpath "string(/testsuites/@skipped)" "$OUTPUT_DIR/reports/test-results.xml")
        
        cat >> "$OUTPUT_DIR/summary.md" << EOF
- Total Tests: $TOTAL_TESTS
- Failures: $FAILURES
- Errors: $ERRORS
- Skipped: $SKIPPED
EOF
    else
        echo "- Test results available in test-results.xml" >> "$OUTPUT_DIR/summary.md"
    fi
fi

cat >> "$OUTPUT_DIR/summary.md" << EOF

## Available Reports
EOF

# List available reports
for report in "$OUTPUT_DIR/reports"/*.{html,md}; do
    if [[ -f "$report" ]]; then
        # Get filename without path
        filename=$(basename "$report")
        echo "- [$filename]($filename)" >> "$OUTPUT_DIR/summary.md"
    fi
done

# Add dashboard information if created
if [[ "$RUN_DASHBOARD" == "true" ]]; then
    cat >> "$OUTPUT_DIR/summary.md" << EOF

## Dashboard
- [Comprehensive Dashboard](dashboard/index.html)
EOF
fi

# Add examples information if generated
if [[ "$GENERATE_EXAMPLES" == "true" ]]; then
    cat >> "$OUTPUT_DIR/summary.md" << EOF

## Generated Examples
- [Examples Dashboard](examples/dashboard/index.html)
EOF
fi

# Get the end time
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Format runtime as minutes and seconds
MINUTES=$((RUNTIME / 60))
SECONDS=$((RUNTIME % 60))

# Add runtime information to summary
cat >> "$OUTPUT_DIR/summary.md" << EOF

## Runtime
- Total runtime: ${MINUTES}m ${SECONDS}s
EOF

# Display result
echo -e "${GREEN}All tasks completed successfully!${NC}"
echo -e "${BLUE}Total runtime: ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${YELLOW}Output available in: $OUTPUT_DIR${NC}"

# If running in GitHub Actions, add summary to step summary
if [[ -n "$GITHUB_STEP_SUMMARY" ]]; then
    cat "$OUTPUT_DIR/summary.md" >> "$GITHUB_STEP_SUMMARY"
fi

# Return success
exit 0