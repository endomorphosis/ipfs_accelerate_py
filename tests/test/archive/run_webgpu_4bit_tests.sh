#!/bin/bash
# WebGPU 4-bit LLM Inference Test Runner
#
# This script runs comprehensive tests for the WebGPU 4-bit inference 
# implementation, testing different model types, sizes, and configurations.
#
# May 2025 Update: Enhanced with adaptive precision, cross-browser support,
# and memory-efficient KV-cache optimizations.
#
# Usage:
#   ./run_webgpu_4bit_tests.sh [--quick] [--full] [--tiny] [--small] [--large] [--report-dir DIR]
#   ./run_webgpu_4bit_tests.sh [--browser chrome|firefox|edge] [--adaptive-precision] [--cross-platform]

# Set default values
QUICK_MODE=false
FULL_MODE=false
TINY_MODE=false
SMALL_MODE=false
LARGE_MODE=false
REPORT_DIR="./webgpu_4bit_results"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# May 2025 enhancement parameters
BROWSER="chrome"
BROWSER_SET=false
ADAPTIVE_PRECISION=false
CROSS_PLATFORM=false
ENABLE_KV_CACHE=true
SIMULATION=false
SPECIALIZED_COMPUTE_SHADERS=false
FIREFOX_OPTIMIZATIONS=false
SAFARI_COMPATIBILITY=false
REINFORCEMENT_LEARNING=false
ALL_OPTIMIZATIONS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      QUICK_MODE=true
      shift
      ;;
    --full)
      FULL_MODE=true
      shift
      ;;
    --tiny)
      TINY_MODE=true
      shift
      ;;
    --small)
      SMALL_MODE=true
      shift
      ;;
    --large)
      LARGE_MODE=true
      shift
      ;;
    --report-dir)
      REPORT_DIR="$2"
      shift
      shift
      ;;
    --browser)
      BROWSER="$2"
      BROWSER_SET=true
      shift
      shift
      ;;
    --adaptive-precision)
      ADAPTIVE_PRECISION=true
      shift
      ;;
    --cross-platform)
      CROSS_PLATFORM=true
      shift
      ;;
    --disable-kv-cache)
      ENABLE_KV_CACHE=false
      shift
      ;;
    --simulation)
      SIMULATION=true
      shift
      ;;
    --specialized-compute-shaders)
      SPECIALIZED_COMPUTE_SHADERS=true
      shift
      ;;
    --firefox-optimizations)
      FIREFOX_OPTIMIZATIONS=true
      shift
      ;;
    --safari-compatibility)
      SAFARI_COMPATIBILITY=true
      shift
      ;;
    --reinforcement-learning)
      REINFORCEMENT_LEARNING=true
      shift
      ;;
    --all-optimizations)
      SPECIALIZED_COMPUTE_SHADERS=true
      FIREFOX_OPTIMIZATIONS=true
      SAFARI_COMPATIBILITY=true
      REINFORCEMENT_LEARNING=true
      ADAPTIVE_PRECISION=true
      CROSS_PLATFORM=true
      ALL_OPTIMIZATIONS=true
      shift
      ;;
    --help)
      echo "WebGPU 4-bit LLM Inference Test Runner"
      echo "===================================="
      echo "Usage:"
      echo "  $0 [--quick] [--full] [--tiny] [--small] [--large] [--report-dir DIR]"
      echo "  $0 [--browser chrome|firefox|edge] [--adaptive-precision] [--cross-platform]"
      echo ""
      echo "Test Suite Options:"
      echo "  --quick                   Run quick test suite with tiny models"
      echo "  --full                    Run full test suite with all model sizes"
      echo "  --tiny                    Include tiny models in tests"
      echo "  --small                   Include small models in tests"
      echo "  --large                   Include 7B parameter models in tests"
      echo "  --report-dir DIR          Directory for test reports (default: ./webgpu_4bit_results)"
      echo ""
      echo "May 2025 Enhancement Options:"
      echo "  --browser BROWSER         Specify browser (chrome, firefox, edge)"
      echo "  --adaptive-precision      Enable adaptive precision testing"
      echo "  --cross-platform          Compare against CPU, GPU, and NPU implementations"
      echo "  --disable-kv-cache        Disable KV-cache optimizations"
      echo "  --simulation              Run in simulation mode instead of real browser"
      echo ""
      echo "Next Steps Options:"
      echo "  --specialized-compute-shaders  Test specialized compute shaders for adaptive precision"
      echo "  --firefox-optimizations        Test Firefox-specific optimizations"
      echo "  --safari-compatibility         Test Safari compatibility features"
      echo "  --reinforcement-learning       Test RL-based autotuning for precision params"
      echo "  --all-optimizations            Enable all next steps features and optimizations"
      echo ""
      echo "Examples:"
      echo "  # Run quick tests for tiny models"
      echo "  $0 --quick"
      echo ""
      echo "  # Run comprehensive tests with adaptive precision"
      echo "  $0 --full --adaptive-precision"
      echo ""
      echo "  # Compare across browsers"
      echo "  $0 --small --browser firefox --cross-platform"
      echo ""
      echo "  # Test Firefox-specific optimizations with adaptive precision"
      echo "  $0 --tiny --firefox-optimizations --adaptive-precision"
      echo ""
      echo "  # Test specialized compute shaders for adaptive precision"
      echo "  $0 --tiny --specialized-compute-shaders --adaptive-precision"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--quick] [--full] [--tiny] [--small] [--large] [--report-dir DIR]"
      echo "       $0 [--browser chrome|firefox|edge] [--adaptive-precision] [--cross-platform]"
      echo "Run with --help for more information."
      exit 1
      ;;
  esac
done

# Create report directory if it doesn't exist
mkdir -p "$REPORT_DIR"
echo "Results will be saved to $REPORT_DIR"

# Apply May 2025 enhancement settings
if [ "$BROWSER_SET" = true ]; then
  echo "Using browser: $BROWSER"
  export WEBGPU_BROWSER="$BROWSER"
fi

if [ "$ADAPTIVE_PRECISION" = true ]; then
  echo "Enabling adaptive precision testing"
  export WEBGPU_ENABLE_ADAPTIVE_PRECISION=1
fi

if [ "$ENABLE_KV_CACHE" = false ]; then
  echo "Disabling KV-cache optimizations"
  export WEBGPU_ENABLE_KV_CACHE_OPTIMIZATION=0
else
  export WEBGPU_ENABLE_KV_CACHE_OPTIMIZATION=1
fi

if [ "$SIMULATION" = true ]; then
  echo "Running in simulation mode"
  export WEBGPU_SIMULATION=1
  export WEBGPU_AVAILABLE=0
else
  export WEBGPU_SIMULATION=0
  export WEBGPU_AVAILABLE=1
fi

# Set cross-platform comparison settings
if [ "$CROSS_PLATFORM" = true ]; then
  echo "Enabling cross-platform comparison (CPU, GPU, NPU, WebGPU)"
  export WEBGPU_CROSS_PLATFORM_TEST=1
fi

# Apply next steps feature settings
if [ "$SPECIALIZED_COMPUTE_SHADERS" = true ]; then
  echo "Enabling specialized compute shaders for adaptive precision"
  export WEBGPU_SPECIALIZED_COMPUTE_SHADERS=1
fi

if [ "$FIREFOX_OPTIMIZATIONS" = true ]; then
  echo "Enabling Firefox-specific optimizations"
  export WEBGPU_FIREFOX_OPTIMIZATIONS=1
  # Default to Firefox browser when testing Firefox optimizations
  if [ "$BROWSER_SET" = false ]; then
    echo "Setting browser to Firefox for Firefox optimization tests"
    export WEBGPU_BROWSER="firefox"
  fi
fi

if [ "$SAFARI_COMPATIBILITY" = true ]; then
  echo "Enabling Safari compatibility features"
  export WEBGPU_SAFARI_COMPATIBILITY=1
  # Note: We still use simulation mode for Safari as it has limited WebGPU support
  export WEBGPU_SIMULATION=1
fi

if [ "$REINFORCEMENT_LEARNING" = true ]; then
  echo "Enabling reinforcement learning autotuning for precision parameters"
  export WEBGPU_RL_AUTOTUNING=1
fi

# Always enable database for results storage
export BENCHMARK_DB_PATH="./benchmark_db.duckdb"

# Set the modes
if [ "$QUICK_MODE" = true ]; then
  # Quick mode tests only tiny models with basic tests
  echo "Running quick mode tests..."
  MODEL_TYPES=("llama")
  MODEL_SIZES=("tiny")
  TESTS=("--all-tests")
  
elif [ "$FULL_MODE" = true ]; then
  # Full mode tests all models and configurations
  echo "Running full test suite..."
  MODEL_TYPES=("llama" "qwen2")
  MODEL_SIZES=("tiny" "small" "7b")
  TESTS=("--all-tests" "--compare-precision" "--disable-kv-cache --all-tests")
  
else
  # Default mode based on size options
  MODEL_TYPES=("llama" "qwen2")
  TESTS=("--all-tests")
  
  # Set model sizes based on flags
  MODEL_SIZES=()
  if [ "$TINY_MODE" = true ]; then
    MODEL_SIZES+=("tiny")
  fi
  if [ "$SMALL_MODE" = true ]; then
    MODEL_SIZES+=("small")
  fi
  if [ "$LARGE_MODE" = true ]; then
    MODEL_SIZES+=("7b")
  fi
  
  # If no size mode is specified, default to tiny
  if [ ${#MODEL_SIZES[@]} -eq 0 ]; then
    MODEL_SIZES=("tiny")
  fi
fi

# Display configuration
echo "Test Configuration:"
echo "- Model Types: ${MODEL_TYPES[*]}"
echo "- Model Sizes: ${MODEL_SIZES[*]}"
echo "- Test Configurations: ${#TESTS[@]}"
echo "- Report Directory: $REPORT_DIR"
echo ""

# Create a summary report file
SUMMARY_FILE="$REPORT_DIR/summary_$TIMESTAMP.md"
echo "# WebGPU 4-bit LLM Inference Test Summary" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "## Test Configuration" >> "$SUMMARY_FILE"
echo "- Model Types: ${MODEL_TYPES[*]}" >> "$SUMMARY_FILE"
echo "- Model Sizes: ${MODEL_SIZES[*]}" >> "$SUMMARY_FILE"
echo "- Test Configurations: ${#TESTS[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "## Results Summary" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| Model | Size | Test | Memory Reduction | Target Met | Speedup | Target Met | KV-Cache | Target Met |" >> "$SUMMARY_FILE"
echo "|-------|------|------|------------------|------------|---------|------------|----------|------------|" >> "$SUMMARY_FILE"

# Track overall success
ALL_TARGETS_MET=true
TOTAL_TESTS=0
PASSED_TESTS=0

# Run tests for each configuration
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
  for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    for TEST_CONFIG in "${TESTS[@]}"; do
      TOTAL_TESTS=$((TOTAL_TESTS+1))
      
      # Create test name
      if [[ "$TEST_CONFIG" == *"disable-kv-cache"* ]]; then
        TEST_NAME="no-kv-cache"
      elif [[ "$TEST_CONFIG" == *"compare-precision"* ]]; then
        TEST_NAME="precision-comparison"
      else
        TEST_NAME="standard"
      fi
      
      # Add adaptive precision suffix if enabled
      if [ "$ADAPTIVE_PRECISION" = true ]; then
        TEST_NAME="${TEST_NAME}-adaptive"
      fi
      
      # Add browser info if specified
      if [ "$BROWSER_SET" = true ]; then
        TEST_NAME="${TEST_NAME}-${BROWSER}"
      fi
      
      # Create file names
      BASE_NAME="${MODEL_TYPE}_${MODEL_SIZE}_${TEST_NAME}"
      REPORT_PATH="$REPORT_DIR/${BASE_NAME}_report_$TIMESTAMP.md" 
      VIS_PATH="$REPORT_DIR/${BASE_NAME}_chart_$TIMESTAMP.png"
      
      # Add cross-platform options if enabled
      CROSS_PLATFORM_ARGS=""
      if [ "$CROSS_PLATFORM" = true ]; then
        CROSS_PLATFORM_ARGS="--cross-platform --compare-hardware --output-comparison-chart $REPORT_DIR/${BASE_NAME}_platform_comparison_$TIMESTAMP.png"
      fi
      
      # Add adaptive precision args if enabled
      ADAPTIVE_PRECISION_ARGS=""
      if [ "$ADAPTIVE_PRECISION" = true ]; then
        ADAPTIVE_PRECISION_ARGS="--adaptive-precision --measure-accuracy --optimize-for-target-accuracy"
      fi
      
      # Add next steps feature args
      NEXT_STEPS_ARGS=""
      if [ "$SPECIALIZED_COMPUTE_SHADERS" = true ]; then
        NEXT_STEPS_ARGS="$NEXT_STEPS_ARGS --specialized-compute-shaders"
      fi
      if [ "$FIREFOX_OPTIMIZATIONS" = true ]; then
        NEXT_STEPS_ARGS="$NEXT_STEPS_ARGS --firefox-optimizations"
      fi
      if [ "$SAFARI_COMPATIBILITY" = true ]; then
        NEXT_STEPS_ARGS="$NEXT_STEPS_ARGS --safari-compatibility"
      fi
      if [ "$REINFORCEMENT_LEARNING" = true ]; then
        NEXT_STEPS_ARGS="$NEXT_STEPS_ARGS --reinforcement-learning"
      fi
      
      echo "Running test: $MODEL_TYPE $MODEL_SIZE ($TEST_NAME)"
      
      # First, test the specialized compute shaders implementation
      if [ "$SPECIALIZED_COMPUTE_SHADERS" = true ]; then
        echo "Testing specialized compute shaders implementation..."
        python test/test_webgpu_compute_shaders.py --all-operations --browser "$BROWSER" --benchmark --test-compilation --generate-shader-set --output-report "${REPORT_DIR}/${BASE_NAME}_shaders_report_$TIMESTAMP.md" --output-json "${REPORT_DIR}/${BASE_NAME}_shaders_results_$TIMESTAMP.json" --verbose --model-size "$MODEL_SIZE"
      fi
      
      # Then run the original LLM tests
      python test/test_webgpu_4bit_llm_inference.py --model "$MODEL_TYPE" --size "$MODEL_SIZE" $TEST_CONFIG $CROSS_PLATFORM_ARGS $ADAPTIVE_PRECISION_ARGS $NEXT_STEPS_ARGS --output-report "$REPORT_PATH" --output-visualization "$VIS_PATH" --use-db
      
      # Extract results from database using DuckDB API
      # Use python to extract and format key metrics from the database
      METRICS=$(python -c "
import sys
import duckdb
import datetime

try:
    # Connect to the database
    db_path = './benchmark_db.duckdb'
    conn = duckdb.connect(db_path)
    
    # Get most recent test results for this model and configuration
    query = \"\"\"
    SELECT 
        m.model_name,
        tr.test_name,
        tr.memory_reduction_percent,
        tr.memory_target_met,
        tr.inference_speedup,
        tr.speedup_target_met,
        tr.kv_cache_enabled,
        tr.kv_cache_improvement,
        tr.kv_cache_target_met
    FROM 
        test_runs tr
    JOIN 
        models m ON tr.model_id = m.model_id
    WHERE 
        m.model_name = ? AND
        tr.test_name LIKE ? AND
        tr.started_at >= ?
    ORDER BY 
        tr.started_at DESC
    LIMIT 1
    \"\"\"
    
    # Get current timestamp in the correct format and compute cutoff time (10 minutes ago)
    cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=10)
    
    # Execute the query
    result = conn.execute(query, ('$MODEL_TYPE', '%$TEST_NAME%', cutoff_time)).fetchone()
    
    if result:
        memory_reduction = result[2] or 0
        memory_target_met = result[3] or False
        memory_status = '✅' if memory_target_met else '❌'
        
        speedup = result[4] or 0
        speedup_target_met = result[5] or False
        speedup_status = '✅' if speedup_target_met else '❌'
        
        kv_cache_enabled = result[6] or False
        if kv_cache_enabled:
            kv_improvement = result[7] or 0
            kv_target_met = result[8] or False
            kv_status = '✅' if kv_target_met else '❌'
            kv_value = f'{kv_improvement:.1f}x'
        else:
            kv_value = 'Disabled'
            kv_status = 'N/A'
            kv_target_met = True  # Don't count against success if disabled
        
        all_targets_met = memory_target_met and speedup_target_met and (kv_status == 'N/A' or kv_target_met)
        
        print(f'{memory_reduction:.1f}% | {memory_status} | {speedup:.2f}x | {speedup_status} | {kv_value} | {kv_status}')
        print(f'{all_targets_met}')
    else:
        # If no results found in database, report error
        print(f'No results found in database | ❌ | 0.00x | ❌ | N/A | N/A')
        print('false')
        
except Exception as e:
    print(f'Database error: {str(e)}')
    print('false')
" || echo "Error extracting metrics | ❌ | Error | ❌ | Error | ❌
false")
      
      # Split the metrics and success status
      METRICS_LINE=$(echo "$METRICS" | head -n 1)
      SUCCESS_LINE=$(echo "$METRICS" | tail -n 1)
      
      # Update summary file
      echo "| $MODEL_TYPE | $MODEL_SIZE | $TEST_NAME | $METRICS_LINE" >> "$SUMMARY_FILE"
      
      # Update success tracking
      if [ "$SUCCESS_LINE" == "false" ]; then
        ALL_TARGETS_MET=false
      else
        PASSED_TESTS=$((PASSED_TESTS+1))
      fi
      
      echo ""
    done
  done
done

# Add overall summary
echo "" >> "$SUMMARY_FILE"
echo "## Overall Summary" >> "$SUMMARY_FILE"
echo "- Total Tests: $TOTAL_TESTS" >> "$SUMMARY_FILE"
echo "- Passed Tests: $PASSED_TESTS" >> "$SUMMARY_FILE"
echo "- Pass Rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%" >> "$SUMMARY_FILE"
echo "- All Targets Met: $([ "$ALL_TARGETS_MET" = true ] && echo "✅ Yes" || echo "❌ No")" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"

echo ""
echo "Testing completed!"
echo "Summary report saved to: $SUMMARY_FILE"
echo "Passed $PASSED_TESTS/$TOTAL_TESTS tests ($(( (PASSED_TESTS * 100) / TOTAL_TESTS ))% pass rate)"
echo "Overall Success: $([ "$ALL_TARGETS_MET" = true ] && echo "✅ Yes" || echo "❌ No")"

# Make script executable
chmod +x "$0"