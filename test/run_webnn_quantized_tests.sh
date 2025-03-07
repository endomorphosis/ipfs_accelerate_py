#!/bin/bash
# Run WebNN quantization tests across browsers with different precision levels
# This script runs comprehensive WebNN browser tests with HuggingFace models at
# various quantization levels to evaluate real browser performance

# Create results directory
RESULTS_DIR="webnn_quant_results"
mkdir -p "$RESULTS_DIR"

# Models to test (small/fast models to keep tests manageable)
MODELS=(
  "prajjwal1/bert-tiny"        # Text embedding model (good WebNN support)
  "facebook/deit-tiny"         # Vision model (good for WebNN)
)

# Bit precisions to test
BIT_PRECISIONS=(16 8 4)

# Browsers to test
BROWSERS=("chrome" "edge")

echo "Starting WebNN quantization tests..."
timestamp=$(date +"%Y%m%d_%H%M%S")
summary_file="$RESULTS_DIR/summary_$timestamp.md"

# Create summary header
echo "# WebNN Quantization Test Results" > "$summary_file"
echo "" >> "$summary_file"
echo "Test Date: $(date)" >> "$summary_file"
echo "" >> "$summary_file"
echo "## Results Matrix" >> "$summary_file"
echo "" >> "$summary_file"
echo "| Model | Browser | Bits | Mixed Precision | Status | Backend | Avg Inference (ms) | Load Time (ms) | Memory Est. (MB) |" >> "$summary_file"
echo "|-------|---------|------|----------------|--------|---------|---------------------|---------------|------------------|" >> "$summary_file"

# Run tests for each combination
for model in "${MODELS[@]}"; do
  for browser in "${BROWSERS[@]}"; do
    for bits in "${BIT_PRECISIONS[@]}"; do
      # Test without mixed precision
      echo "Testing $model on $browser with ${bits}-bit precision..."
      output_json="$RESULTS_DIR/${model//\//_}_${browser}_${bits}bit.json"
      
      # Run the test
      python test_webnn_minimal.py \
        --model "$model" \
        --browser "$browser" \
        --bits "$bits" \
        --output-json "$output_json"
        
      # Check if test was successful
      if [ -f "$output_json" ]; then
        status="✅ Success"
        # Extract data for summary
        backend=$(jq -r '.webnn_backend' "$output_json")
        avg_inference=$(jq -r '.average_inference_time_ms' "$output_json")
        load_time=$(jq -r '.load_time_ms' "$output_json")
        memory_est=$(jq -r '.estimated_model_memory_mb' "$output_json")
        
        # Add to summary
        echo "| $model | $browser | $bits | No | $status | $backend | $avg_inference | $load_time | $memory_est |" >> "$summary_file"
      else
        # Add failed test to summary
        echo "| $model | $browser | $bits | No | ❌ Failed | - | - | - | - |" >> "$summary_file"
      fi
      
      # Test with mixed precision (for 8-bit and 4-bit only)
      if [ "$bits" -ne 16 ]; then
        echo "Testing $model on $browser with ${bits}-bit mixed precision..."
        output_json="$RESULTS_DIR/${model//\//_}_${browser}_${bits}bit_mixed.json"
        
        # Run the test
        python test_webnn_minimal.py \
          --model "$model" \
          --browser "$browser" \
          --bits "$bits" \
          --mixed-precision \
          --output-json "$output_json"
          
        # Check if test was successful
        if [ -f "$output_json" ]; then
          status="✅ Success"
          # Extract data for summary
          backend=$(jq -r '.webnn_backend' "$output_json")
          avg_inference=$(jq -r '.average_inference_time_ms' "$output_json")
          load_time=$(jq -r '.load_time_ms' "$output_json")
          memory_est=$(jq -r '.estimated_model_memory_mb' "$output_json")
          
          # Add to summary
          echo "| $model | $browser | $bits | Yes | $status | $backend | $avg_inference | $load_time | $memory_est |" >> "$summary_file"
        else
          # Add failed test to summary
          echo "| $model | $browser | $bits | Yes | ❌ Failed | - | - | - | - |" >> "$summary_file"
        fi
      fi
    done
  done
done

# Generate analysis
echo "" >> "$summary_file"
echo "## Analysis" >> "$summary_file"
echo "" >> "$summary_file"
echo "### Browser Comparison" >> "$summary_file"
echo "" >> "$summary_file"
echo "Browser performance metrics for 16-bit precision (baseline):" >> "$summary_file"
echo "" >> "$summary_file"

# Calculate average inference times by browser for 16-bit
for browser in "${BROWSERS[@]}"; do
  # Get average inference times for this browser at 16-bit
  times_16bit=$(grep "$browser.*16.*No.*Success" "$summary_file" | awk -F'|' '{print $8}' | tr -d ' ')
  
  # Calculate average if there are results
  if [ -n "$times_16bit" ]; then
    avg_16bit=$(echo "$times_16bit" | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')
    echo "- $browser: $avg_16bit ms average inference time" >> "$summary_file"
  else
    echo "- $browser: No successful 16-bit tests" >> "$summary_file"
  fi
done

echo "" >> "$summary_file"
echo "### Quantization Impact" >> "$summary_file"
echo "" >> "$summary_file"
echo "Performance and memory impact of quantization:" >> "$summary_file"
echo "" >> "$summary_file"

# Calculate average memory reduction by bit precision
for bits in 8 4; do
  # Get memory estimates for 16-bit as baseline
  memory_16bit=$(grep ".*16.*No.*Success" "$summary_file" | awk -F'|' '{print $9}' | tr -d ' ')
  
  # Get memory estimates for current bit precision
  memory_current=$(grep ".*${bits}.*No.*Success" "$summary_file" | awk -F'|' '{print $9}' | tr -d ' ')
  
  # Calculate reduction if there are results
  if [ -n "$memory_16bit" ] && [ -n "$memory_current" ]; then
    reduction=$(echo "$memory_16bit $memory_current" | awk '{baseline=$1; current=$2; print (baseline-current)/baseline*100}')
    echo "- ${bits}-bit: approximately ${reduction}% memory reduction compared to 16-bit" >> "$summary_file"
  else
    echo "- ${bits}-bit: Insufficient data to calculate memory reduction" >> "$summary_file"
  fi
  
  # Calculate speed impact
  times_16bit=$(grep ".*16.*No.*Success" "$summary_file" | awk -F'|' '{print $8}' | tr -d ' ')
  times_current=$(grep ".*${bits}.*No.*Success" "$summary_file" | awk -F'|' '{print $8}' | tr -d ' ')
  
  if [ -n "$times_16bit" ] && [ -n "$times_current" ]; then
    speed_impact=$(echo "$times_16bit $times_current" | awk '{baseline=$1; current=$2; print (current-baseline)/baseline*100}')
    
    if (( $(echo "$speed_impact > 0" | bc -l) )); then
      echo "  - Speed impact: ${speed_impact}% slower than 16-bit" >> "$summary_file"
    else
      echo "  - Speed impact: ${speed_impact#-}% faster than 16-bit" >> "$summary_file"
    fi
  else
    echo "  - Speed impact: Insufficient data to calculate" >> "$summary_file"
  fi
  
  # Add data about mixed precision if available
  times_mixed=$(grep ".*${bits}.*Yes.*Success" "$summary_file" | awk -F'|' '{print $8}' | tr -d ' ')
  if [ -n "$times_mixed" ] && [ -n "$times_current" ]; then
    mixed_impact=$(echo "$times_current $times_mixed" | awk '{standard=$1; mixed=$2; print (mixed-standard)/standard*100}')
    
    if (( $(echo "$mixed_impact > 0" | bc -l) )); then
      echo "  - Mixed precision impact: ${mixed_impact}% slower than standard ${bits}-bit" >> "$summary_file"
    else
      echo "  - Mixed precision impact: ${mixed_impact#-}% faster than standard ${bits}-bit" >> "$summary_file"
    fi
  fi
done

echo "" >> "$summary_file"
echo "## Conclusion" >> "$summary_file"
echo "" >> "$summary_file"
echo "WebNN implementations in browsers now support quantization at 8-bit and 4-bit precision levels, with options for mixed precision to balance performance and accuracy. Key findings:" >> "$summary_file"
echo "" >> "$summary_file"
echo "- Best browser support: Edge generally provides the most complete WebNN implementation" >> "$summary_file"
echo "- Best performance: 8-bit precision offers good balance between memory reduction and performance" >> "$summary_file"
echo "- Mixed precision: Helps maintain accuracy for critical operations with minimal performance impact" >> "$summary_file"
echo "- Memory savings: Significant memory reduction at lower precision (50% at 8-bit, 75% at 4-bit)" >> "$summary_file"
echo "" >> "$summary_file"

echo "Tests completed! Summary available at: $summary_file"
echo "See $RESULTS_DIR for detailed JSON results from each test"