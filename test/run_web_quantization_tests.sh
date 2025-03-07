#!/bin/bash
# Master script to run comprehensive web browser quantization tests
# This script runs both WebNN and WebGPU tests, then generates a combined report

# Create output directories
WEBNN_DIR="webnn_quant_results"
WEBGPU_DIR="webgpu_results"
REPORT_DIR="web_quant_reports"

mkdir -p "$WEBNN_DIR"
mkdir -p "$WEBGPU_DIR"  
mkdir -p "$REPORT_DIR"

# Timestamp for reports
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check dependencies
echo "Checking dependencies..."
MISSING_DEPS=false

# Check for jq (used in result parsing)
if ! command -v jq &> /dev/null; then
    echo "⚠️ jq is not installed. Some result parsing may not work correctly."
    MISSING_DEPS=true
fi

# Check for Python dependencies
echo "Checking Python dependencies..."
python -c "import selenium" 2>/dev/null || { echo "⚠️ selenium not installed (pip install selenium)"; MISSING_DEPS=true; }
python -c "import pandas, matplotlib" 2>/dev/null || { echo "⚠️ pandas/matplotlib not installed (for visualization)"; }

if [ "$MISSING_DEPS" = true ]; then
    echo "Some dependencies are missing. Continue anyway? (y/n)"
    read -r continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        echo "Exiting. Please install the missing dependencies and try again."
        exit 1
    fi
fi

echo "=== Web Browser Quantization Test Suite ==="
echo "This script will run comprehensive WebNN and WebGPU quantization tests"
echo "across different browsers and precision levels."
echo ""
echo "1. Run WebNN tests in Chrome and Edge"
echo "2. Run WebGPU tests in Chrome, Firefox, and Edge"
echo "3. Generate comprehensive report and matrix"
echo ""
echo "Tests may take 20-30 minutes to complete depending on your hardware."
echo "Results will be saved in:"
echo "- $WEBNN_DIR (WebNN results)"
echo "- $WEBGPU_DIR (WebGPU results)"
echo "- $REPORT_DIR (Combined reports)"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read -r

# Part 1: Run WebNN tests
echo ""
echo "=== Running WebNN quantization tests ==="
./run_webnn_quantized_tests.sh
echo "WebNN tests completed!"

# Part 2: Run WebGPU tests
echo ""
echo "=== Running WebGPU quantization tests ==="
# Run comprehensive WebGPU tests
python test_webgpu_quantization.py --run-all --output-dir "$WEBGPU_DIR"
echo "WebGPU tests completed!"

# Part 3: Generate the quantization matrix
echo ""
echo "=== Generating quantization matrix ==="
python web_quantization_dashboard.py --create-matrix
cp web_quantization_matrix.md "$REPORT_DIR/web_quantization_matrix_$TIMESTAMP.md"
echo "Matrix generated!"

# Part 4: Generate combined report
echo ""
echo "=== Generating combined report ==="
REPORT_PATH="$REPORT_DIR/web_quant_report_$TIMESTAMP.html"
python web_quantization_dashboard.py --webnn-dir "$WEBNN_DIR" --webgpu-dir "$WEBGPU_DIR" --combined-report "$REPORT_PATH"

# Show summary
echo ""
echo "=== Testing Complete ==="
echo "Tests run: $(find "$WEBNN_DIR" "$WEBGPU_DIR" -name "*.json" | wc -l)"
echo "Reports generated:"
echo "- Quantization Matrix: $REPORT_DIR/web_quantization_matrix_$TIMESTAMP.md"
echo "- Combined Report: $REPORT_PATH"
echo ""
echo "You can view the generated reports to analyze the results."
echo "The matrix provides a summary of browser support for different precision levels."
echo "The combined report includes detailed performance comparisons across browsers."
echo ""
echo "For more detailed testing, you can run individual tests with:"
echo "- WebNN: python test_webnn_minimal.py --model <model_name> --browser <browser> --bits <bits>"
echo "- WebGPU: python test_webgpu_quantization.py --model <model_name> --browser <browser> --bits <bits>"
echo ""
echo "Thank you for running the web quantization tests!"