#!/bin/bash
# Circuit Breaker Benchmark Script
#
# This script runs the circuit breaker benchmark with various configurations.
# It provides an easy way to invoke the benchmark with common options.
# 
# Enhanced with CI/CD support and comprehensive benchmark profiles

# Default settings
BROWSERS=2
ITERATIONS=3
REPORT_DIR="./benchmark_reports"
VERBOSE=false
SIMULATE=false
FAILURE_TYPES=""
CI_MODE=false
SUMMARY_ONLY=false
COMPARE_WITH_PREVIOUS=false
EXPORT_METRICS=false
METRICS_FILE=""
BENCHMARK_TYPE="standard"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --browsers=*)
      BROWSERS="${1#*=}"
      shift
      ;;
    --iterations=*)
      ITERATIONS="${1#*=}"
      shift
      ;;
    --report-dir=*)
      REPORT_DIR="${1#*=}"
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --simulate)
      SIMULATE=true
      shift
      ;;
    --quick)
      # Quick mode: 1 browser, 2 iterations
      BROWSERS=1
      ITERATIONS=2
      BENCHMARK_TYPE="quick"
      shift
      ;;
    --comprehensive)
      # Comprehensive mode: 3 browsers, 5 iterations
      BROWSERS=3
      ITERATIONS=5
      BENCHMARK_TYPE="comprehensive"
      shift
      ;;
    --extreme)
      # Extreme mode: All browsers, 10 iterations, all failure types
      BROWSERS=3
      ITERATIONS=10
      BENCHMARK_TYPE="extreme"
      shift
      ;;
    --ci)
      # CI mode: Adjusted settings for CI environment
      CI_MODE=true
      SIMULATE=true  # Default to simulation in CI
      SUMMARY_ONLY=true
      BENCHMARK_TYPE="ci"
      shift
      ;;
    --summary-only)
      # Only print summary, not detailed logs
      SUMMARY_ONLY=true
      shift
      ;;
    --compare-with-previous)
      # Compare with previous benchmark results
      COMPARE_WITH_PREVIOUS=true
      shift
      ;;
    --export-metrics)
      # Export metrics in CI-friendly format
      EXPORT_METRICS=true
      shift
      ;;
    --metrics-file=*)
      # File to export metrics to
      METRICS_FILE="${1#*=}"
      EXPORT_METRICS=true
      shift
      ;;
    --failure-types=*)
      FAILURE_TYPES="${1#*=}"
      shift
      ;;
    --chrome-only)
      # Test with Chrome only
      BROWSERS=1
      shift
      ;;
    --firefox-only)
      # Test with Firefox only
      BROWSERS=1
      export CIRCUIT_BREAKER_TEST_BROWSER="firefox"
      shift
      ;;
    --edge-only)
      # Test with Edge only
      BROWSERS=1
      export CIRCUIT_BREAKER_TEST_BROWSER="edge"
      shift
      ;;
    --connection-failures-only)
      # Test with connection failures only
      FAILURE_TYPES="connection_failure"
      shift
      ;;
    --resource-failures-only)
      # Test with resource exhaustion failures only
      FAILURE_TYPES="resource_exhaustion"
      shift
      ;;
    --gpu-failures-only)
      # Test with GPU errors only
      FAILURE_TYPES="gpu_error"
      shift
      ;;
    --api-failures-only)
      # Test with API errors only
      FAILURE_TYPES="api_error"
      shift
      ;;
    --crash-failures-only)
      # Test with crash failures only
      FAILURE_TYPES="crash"
      shift
      ;;
    --timeout-failures-only)
      # Test with timeout failures only
      FAILURE_TYPES="timeout"
      shift
      ;;
    --weekday-schedule)
      # Weekday schedule (smaller scope for daily runs)
      if [ "$(date +%u)" -lt 6 ]; then
        # Monday-Friday: quick test
        BROWSERS=1
        ITERATIONS=2
        BENCHMARK_TYPE="weekday"
      else
        # Weekend: comprehensive test
        BROWSERS=3
        ITERATIONS=5
        BENCHMARK_TYPE="weekend"
      fi
      shift
      ;;
    --help)
      echo "Circuit Breaker Benchmark Script"
      echo ""
      echo "This script runs the circuit breaker benchmark with various configurations."
      echo ""
      echo "Usage:"
      echo "  ./run_circuit_breaker_benchmark.sh [options]"
      echo ""
      echo "Options:"
      echo "  --browsers=N          Number of browser types to include (default: 2)"
      echo "  --iterations=N        Number of iterations for each test case (default: 3)"
      echo "  --report-dir=DIR      Directory to save reports (default: ./benchmark_reports)"
      echo "  --verbose             Enable verbose logging"
      echo "  --simulate            Run in simulation mode (no real browsers)"
      echo ""
      echo "Benchmark Profiles:"
      echo "  --quick               Run quick test (1 browser, 2 iterations)"
      echo "  --comprehensive       Run comprehensive test (3 browsers, 5 iterations)"
      echo "  --extreme             Run extreme test (3 browsers, 10 iterations, all failures)"
      echo "  --weekday-schedule    Auto-select based on day of week (quick on weekdays, comprehensive on weekends)"
      echo ""
      echo "CI/CD Options:"
      echo "  --ci                  Run in CI mode (simulation mode, summary output)"
      echo "  --summary-only        Only print summary, not detailed logs"
      echo "  --compare-with-previous   Compare with previous benchmark results"
      echo "  --export-metrics      Export metrics in CI-friendly format"
      echo "  --metrics-file=FILE   File to export metrics to"
      echo ""
      echo "Failure Type Options:"
      echo "  --failure-types=TYPES Comma-separated list of failure types to test"
      echo "                        Valid types: connection_failure, resource_exhaustion,"
      echo "                                     gpu_error, api_error, timeout,"
      echo "                                     crash, internal_error"
      echo "  --chrome-only         Test with Chrome only"
      echo "  --firefox-only        Test with Firefox only"
      echo "  --edge-only           Test with Edge only"
      echo "  --connection-failures-only    Test with connection failures only"
      echo "  --resource-failures-only      Test with resource exhaustion failures only"
      echo "  --gpu-failures-only           Test with GPU errors only"
      echo "  --api-failures-only           Test with API errors only"
      echo "  --timeout-failures-only       Test with timeout failures only"
      echo "  --crash-failures-only         Test with crash failures only"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Special handling for extreme benchmark type
if [ "$BENCHMARK_TYPE" = "extreme" ]; then
  # If failure types not specified for extreme benchmark, use all
  if [ -z "$FAILURE_TYPES" ]; then
    FAILURE_TYPES="connection_failure,resource_exhaustion,gpu_error,api_error,timeout,crash,internal_error"
  fi
fi

# Create report directory if it doesn't exist
mkdir -p "$REPORT_DIR"

# Set timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build command
CMD="python benchmark_circuit_breaker.py --browsers $BROWSERS --iterations $ITERATIONS"

# Add report path
CMD="$CMD --report $REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json"

# Add failure types if specified
if [ -n "$FAILURE_TYPES" ]; then
  # Replace commas with spaces for argument list
  FAILURE_TYPES_ARG=$(echo "$FAILURE_TYPES" | tr ',' ' ')
  CMD="$CMD --failure-types $FAILURE_TYPES_ARG"
fi

# Add verbose flag if needed
if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

# Add simulate flag if needed
if [ "$SIMULATE" = true ]; then
  CMD="$CMD --simulate"
fi

# If summary only, redirect detailed output
if [ "$SUMMARY_ONLY" = true ]; then
  CMD="$CMD 2>&1 | grep -E '(Circuit Breaker Benchmark Results|Recovery time improvement|Success rate improvement|Overall rating)'"
fi

# Print header (unless in summary only mode)
if [ "$SUMMARY_ONLY" = false ]; then
  echo "================================================================================"
  echo "Circuit Breaker Benchmark"
  echo "================================================================================"
  echo "Date:          $(date +"%Y-%m-%d %H:%M:%S")"
  echo "Benchmark:     $BENCHMARK_TYPE"
  echo "Browsers:      $BROWSERS"
  echo "Iterations:    $ITERATIONS"
  echo "Failure Types: ${FAILURE_TYPES:-All}"
  echo "Simulate:      $SIMULATE"
  echo "Verbose:       $VERBOSE"
  echo "CI Mode:       $CI_MODE"
  echo "Report:        $REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json"
  echo "================================================================================"
  echo "Command: $CMD"
  echo "================================================================================"
else
  echo "Running $BENCHMARK_TYPE benchmark with $BROWSERS browsers, $ITERATIONS iterations..."
fi

# Run the command and capture both output and exit code
if [ "$SUMMARY_ONLY" = true ]; then
  # For summary mode, capture output
  OUTPUT=$(eval $CMD)
  EXIT_CODE=$?
  echo "$OUTPUT"
else
  # For normal mode, let output go to console
  eval $CMD
  EXIT_CODE=$?
fi

# Extract metrics if requested
if [ "$EXPORT_METRICS" = true ]; then
  if [ -f "$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json" ]; then
    # Choose metrics file if not specified
    if [ -z "$METRICS_FILE" ]; then
      METRICS_FILE="$REPORT_DIR/circuit_breaker_metrics_${TIMESTAMP}.json"
    fi
    
    # Extract key metrics using jq if available, otherwise use grep/sed
    if command -v jq &> /dev/null; then
      jq '{
        timestamp: .timestamp,
        recovery_time_improvement_pct: .summary.recovery_time_improvement_pct,
        success_rate_improvement_pct: .summary.success_rate_improvement_pct,
        overall_rating: .summary.overall_rating,
        with_circuit_breaker: {
          success_rate: .with_circuit_breaker.overall.success_rate,
          avg_recovery_time_ms: .with_circuit_breaker.overall.avg_recovery_time_ms
        },
        without_circuit_breaker: {
          success_rate: .without_circuit_breaker.overall.success_rate,
          avg_recovery_time_ms: .without_circuit_breaker.overall.avg_recovery_time_ms
        },
        benchmark_type: "'$BENCHMARK_TYPE'",
        browsers: '$BROWSERS',
        iterations: '$ITERATIONS'
      }' "$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json" > "$METRICS_FILE"
      echo "Metrics exported to $METRICS_FILE"
    else
      # Simplified extraction without jq
      echo "jq not found, using simplified metrics extraction"
      # Use Python to extract metrics if available
      if command -v python &> /dev/null; then
        python -c "
import json, sys
with open('$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json', 'r') as f:
    data = json.load(f)
metrics = {
    'timestamp': data.get('timestamp'),
    'recovery_time_improvement_pct': data.get('summary', {}).get('recovery_time_improvement_pct'),
    'success_rate_improvement_pct': data.get('summary', {}).get('success_rate_improvement_pct'),
    'overall_rating': data.get('summary', {}).get('overall_rating'),
    'benchmark_type': '$BENCHMARK_TYPE',
    'browsers': $BROWSERS,
    'iterations': $ITERATIONS
}
with open('$METRICS_FILE', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics exported to $METRICS_FILE')
        "
      else
        echo "Neither jq nor python found. Metrics export requires one of these tools."
      fi
    fi
  else
    echo "Error: Benchmark report not found, cannot export metrics"
  fi
fi

# Compare with previous if requested
if [ "$COMPARE_WITH_PREVIOUS" = true ]; then
  # Find the previous benchmark report
  PREV_REPORT=$(find "$REPORT_DIR" -name "circuit_breaker_benchmark_*.json" -not -name "circuit_breaker_benchmark_${TIMESTAMP}.json" | sort -r | head -n 1)
  
  if [ -n "$PREV_REPORT" ] && [ -f "$PREV_REPORT" ] && [ -f "$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json" ]; then
    echo ""
    echo "================================================================================"
    echo "Comparison with previous benchmark: $(basename "$PREV_REPORT")"
    echo "================================================================================"
    
    # Use python for comparison if available
    if command -v python &> /dev/null; then
      python -c "
import json, sys
from datetime import datetime

# Load current and previous results
with open('$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json', 'r') as f:
    current = json.load(f)
with open('$PREV_REPORT', 'r') as f:
    previous = json.load(f)

# Extract key metrics
curr_recovery_time = current.get('summary', {}).get('recovery_time_improvement_pct', 0)
prev_recovery_time = previous.get('summary', {}).get('recovery_time_improvement_pct', 0)
recovery_time_diff = curr_recovery_time - prev_recovery_time

curr_success_rate = current.get('summary', {}).get('success_rate_improvement_pct', 0)
prev_success_rate = previous.get('summary', {}).get('success_rate_improvement_pct', 0)
success_rate_diff = curr_success_rate - prev_success_rate

curr_rating = current.get('summary', {}).get('overall_rating', 'Unknown')
prev_rating = previous.get('summary', {}).get('overall_rating', 'Unknown')

# Format dates
curr_date = datetime.fromisoformat(current.get('timestamp', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
prev_date = datetime.fromisoformat(previous.get('timestamp', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')

# Print comparison
print(f'Current  ({curr_date}): Recovery improvement: {curr_recovery_time:.1f}%, Success rate improvement: {curr_success_rate:.1f}%, Rating: {curr_rating}')
print(f'Previous ({prev_date}): Recovery improvement: {prev_recovery_time:.1f}%, Success rate improvement: {prev_success_rate:.1f}%, Rating: {prev_rating}')
print('')
print(f'Recovery time improvement change: {recovery_time_diff:+.1f}%')
print(f'Success rate improvement change:  {success_rate_diff:+.1f}%')

# Overall assessment
if recovery_time_diff > 5 or success_rate_diff > 5:
    print('\nSIGNIFICANT IMPROVEMENT detected in circuit breaker performance!')
elif recovery_time_diff < -5 or success_rate_diff < -5:
    print('\nSIGNIFICANT REGRESSION detected in circuit breaker performance!')
elif abs(recovery_time_diff) <= 2 and abs(success_rate_diff) <= 2:
    print('\nPerformance is stable (within ±2% variation).')
else:
    print('\nMinor performance variation detected.')
        "
    else
      echo "Python not found. Comparison requires python."
    fi
  else
    echo "No previous benchmark report found for comparison."
  fi
fi

# Always output a GitHub-friendly summary if in CI mode
if [ "$CI_MODE" = true ]; then
  if [ -f "$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json" ]; then
    # Use Python to extract key metrics for GitHub
    if command -v python &> /dev/null; then
      python -c "
import json, sys
with open('$REPORT_DIR/circuit_breaker_benchmark_${TIMESTAMP}.json', 'r') as f:
    data = json.load(f)

recovery_time = data.get('summary', {}).get('recovery_time_improvement_pct', 0)
success_rate = data.get('summary', {}).get('success_rate_improvement_pct', 0)
rating = data.get('summary', {}).get('overall_rating', 'Unknown')

print('::group::Circuit Breaker Benchmark Results')
print(f'Recovery time improvement: {recovery_time:.1f}%')
print(f'Success rate improvement: {success_rate:.1f}%')
print(f'Overall rating: {rating}')
print('::endgroup::')

# Set output parameters for GitHub Actions
print(f'::set-output name=recovery_time_improvement::{recovery_time:.1f}')
print(f'::set-output name=success_rate_improvement::{success_rate:.1f}')
print(f'::set-output name=overall_rating::{rating}')
      "
    fi
  fi
fi

# Exit with the command's exit code
exit $EXIT_CODE