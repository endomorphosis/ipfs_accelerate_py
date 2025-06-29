#!/bin/bash
#
# Run comprehensive WebNN/WebGPU precision tests with REAL browser implementations
#
# This script tests WebNN and WebGPU at all precision levels (2-bit, 3-bit, 4-bit, 8-bit, 16-bit, 32-bit)
# using real browser implementations via Selenium bridge.
#
# It will test Chrome, Firefox, Edge, and Safari browsers (if available)
# with various models (BERT, Whisper, etc.) and generate comprehensive reports.
#
# IMPORTANT: This script is specifically designed to test REAL implementations (not simulations)
# and will verify that we're using actual browser-based hardware acceleration.

# Set up colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="real_webnn_webgpu_tests_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}REAL WebNN/WebGPU Implementation Precision Test${NC}"
echo -e "${BLUE}=============================================${NC}"
echo
echo -e "${YELLOW}Testing REAL browser-based hardware acceleration for WebNN/WebGPU${NC}"
echo -e "${YELLOW}at all precision levels (2-bit to 32-bit) using Selenium bridge${NC}"
echo
echo -e "Results will be saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo

# Function to check for existing results and browser implementations
function check_for_existing_results() {
    echo -e "${CYAN}Checking for existing documentation and results...${NC}"
    
    # Check for documentation
    if [ -f "REAL_WEBNN_WEBGPU_TESTING.md" ]; then
        echo -e "  ${GREEN}✓ Found existing documentation: REAL_WEBNN_WEBGPU_TESTING.md${NC}"
    else
        echo -e "  ${RED}✗ Documentation not found: REAL_WEBNN_WEBGPU_TESTING.md${NC}"
    fi
    
    # Check for test script
    if [ -f "run_comprehensive_webnn_webgpu_tests.py" ]; then
        echo -e "  ${GREEN}✓ Found test script: run_comprehensive_webnn_webgpu_tests.py${NC}"
    else
        echo -e "  ${RED}✗ Test script not found: run_comprehensive_webnn_webgpu_tests.py${NC}"
        echo -e "  ${YELLOW}Will use existing implement_real_webnn_webgpu.py instead${NC}"
    fi
    
    # Check for existing results
    result_dirs=$(find . -maxdepth 1 -name "real_webnn_webgpu_tests_*" -type d | wc -l)
    if [ $result_dirs -gt 0 ]; then
        echo -e "  ${GREEN}✓ Found $result_dirs existing test result directories${NC}"
    else
        echo -e "  ${YELLOW}No existing test result directories found${NC}"
    fi
    
    echo
}

# Function to verify Selenium and WebDriver installation
function verify_browser_drivers() {
    echo -e "${CYAN}Checking for browser drivers and Selenium...${NC}"
    
    # Check for selenium
    if python -c "import selenium" &>/dev/null; then
        echo -e "  ${GREEN}✓ Selenium installed${NC}"
    else
        echo -e "  ${RED}✗ Selenium not installed${NC}"
        echo -e "  ${YELLOW}Installing Selenium...${NC}"
        pip install selenium
    fi
    
    # Check for websockets
    if python -c "import websockets" &>/dev/null; then
        echo -e "  ${GREEN}✓ Websockets installed${NC}"
    else
        echo -e "  ${RED}✗ Websockets not installed${NC}"
        echo -e "  ${YELLOW}Installing Websockets...${NC}"
        pip install websockets
    fi
    
    # Install browser drivers if needed
    echo -e "  ${YELLOW}Installing/updating browser drivers...${NC}"
    if python -c "import webdriver_manager" &>/dev/null; then
        echo -e "  ${GREEN}✓ webdriver_manager installed${NC}"
    else
        echo -e "  ${YELLOW}Installing webdriver_manager...${NC}"
        pip install webdriver-manager
    fi
    
    # Install drivers for browsers
    if python implement_real_webnn_webgpu.py --install-drivers; then
        echo -e "  ${GREEN}✓ Browser drivers installed successfully${NC}"
    else
        echo -e "  ${RED}✗ Error installing browser drivers${NC}"
    fi
    
    echo
}

# Function to run real implementation tests for a specific browser, platform, and precision
function run_real_implementation_test() {
    local browser=$1
    local platform=$2
    local precision=$3
    local model=$4
    local extra_args=$5
    
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${YELLOW}Testing REAL $platform with $browser browser at $precision-bit precision, model: $model${NC}"
    
    # Set options for test command
    local cmd_options="--browser $browser --platform $platform --model $model --model-type text --inference --verbose"
    
    # Add extra args if provided
    if [ ! -z "$extra_args" ]; then
        cmd_options="$cmd_options $extra_args"
    fi
    
    # Run test and capture output
    output_file="$OUTPUT_DIR/${platform}_${browser}_${model}_${precision}bit.log"
    
    # Set environment variables based on platform and precision
    if [ "$platform" = "webgpu" ]; then
        export WEBGPU_SIMULATION="0"  # Force real implementation
        
        # Set precision flag for WebGPU (implementation specific)
        if [ "$precision" = "4" ]; then
            export WEBGPU_4BIT_ENABLED="1"
        elif [ "$precision" = "8" ]; then
            export WEBGPU_8BIT_ENABLED="1"
        elif [ "$precision" = "2" ] || [ "$precision" = "3" ]; then
            export WEBGPU_ULTRA_LOW_PRECISION="$precision"
        fi
    elif [ "$platform" = "webnn" ]; then
        export WEBNN_SIMULATION="0"  # Force real implementation
        
        # Set precision flag for WebNN (implementation specific)
        if [ "$precision" = "8" ]; then
            export WEBNN_INT8_ENABLED="1"
        fi
    fi
    
    # Run the test
    echo -e "  ${CYAN}Running: python implement_real_webnn_webgpu.py $cmd_options${NC}"
    echo -e "  ${CYAN}Output will be saved to: $output_file${NC}"
    
    if python implement_real_webnn_webgpu.py $cmd_options 2>&1 | tee $output_file; then
        echo -e "  ${GREEN}✓ Test completed successfully${NC}"
        
        # Check if we used real implementation
        if grep -q "REAL_" $output_file && ! grep -q "simulation: true" $output_file; then
            echo -e "  ${GREEN}✓ Used REAL hardware acceleration${NC}"
            real_count=$((real_count + 1))
        else
            echo -e "  ${RED}✗ Used SIMULATION mode${NC}"
            simulation_count=$((simulation_count + 1))
        fi
    else
        echo -e "  ${RED}✗ Test failed${NC}"
        failed_count=$((failed_count + 1))
    fi
    
    # Reset environment variables
    unset WEBGPU_SIMULATION
    unset WEBNN_SIMULATION
    unset WEBGPU_4BIT_ENABLED
    unset WEBGPU_8BIT_ENABLED
    unset WEBGPU_ULTRA_LOW_PRECISION
    unset WEBNN_INT8_ENABLED
}

# Check for existing results and drivers
check_for_existing_results
verify_browser_drivers

# Initialize counters
real_count=0
simulation_count=0
failed_count=0
total_count=0

# Check which browsers to test
echo -e "${CYAN}Detecting available browsers...${NC}"
browsers=()

if which google-chrome >/dev/null || which chrome >/dev/null; then
    echo -e "  ${GREEN}✓ Chrome browser found${NC}"
    browsers+=("chrome")
else
    echo -e "  ${RED}✗ Chrome browser not found${NC}"
fi

if which firefox >/dev/null; then
    echo -e "  ${GREEN}✓ Firefox browser found${NC}"
    browsers+=("firefox")
else
    echo -e "  ${RED}✗ Firefox browser not found${NC}"
fi

if which msedge >/dev/null || which edge >/dev/null; then
    echo -e "  ${GREEN}✓ Edge browser found${NC}"
    browsers+=("edge")
else
    echo -e "  ${RED}✗ Edge browser not found${NC}"
fi

if which safari >/dev/null || [ -d "/Applications/Safari.app" ]; then
    echo -e "  ${GREEN}✓ Safari browser found${NC}"
    browsers+=("safari")
else
    echo -e "  ${RED}✗ Safari browser not found${NC}"
fi

echo
echo -e "${CYAN}Will test with browsers: ${browsers[@]}${NC}"
echo

# Determine which tests to run based on user input
read -p "$(echo -e "${YELLOW}Enter test mode (1=Quick, 2=Standard, 3=Comprehensive): ${NC}")" test_mode

if [ -z "$test_mode" ] || [ "$test_mode" != "1" ] && [ "$test_mode" != "2" ] && [ "$test_mode" != "3" ]; then
    echo -e "${YELLOW}Invalid mode, defaulting to Quick mode${NC}"
    test_mode="1"
fi

echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}Starting tests in mode $test_mode${NC}"
echo -e "${BLUE}=============================================${NC}"
echo

# Run tests based on selected mode
if [ "$test_mode" = "1" ]; then
    # Quick mode - Test only 4-bit and 8-bit precision with BERT on Chrome/Firefox
    echo -e "${CYAN}Running Quick tests (4-bit and 8-bit, BERT model, Chrome/Firefox)${NC}"
    
    # Chrome tests (if available)
    if [[ " ${browsers[@]} " =~ " chrome " ]]; then
        run_real_implementation_test "chrome" "webgpu" "4" "bert-base-uncased" ""
        run_real_implementation_test "chrome" "webnn" "8" "bert-base-uncased" ""
        total_count=$((total_count + 2))
    fi
    
    # Firefox tests (if available)
    if [[ " ${browsers[@]} " =~ " firefox " ]]; then
        run_real_implementation_test "firefox" "webgpu" "4" "bert-base-uncased" ""
        total_count=$((total_count + 1))
    fi
    
    # Edge tests (if available) for WebNN
    if [[ " ${browsers[@]} " =~ " edge " ]]; then
        run_real_implementation_test "edge" "webnn" "8" "bert-base-uncased" ""
        total_count=$((total_count + 1))
    fi
    
elif [ "$test_mode" = "2" ]; then
    # Standard mode - Test 4-bit, 8-bit, 16-bit with BERT, Whisper on all available browsers
    echo -e "${CYAN}Running Standard tests (4/8/16-bit, BERT and Whisper models, all browsers)${NC}"
    
    # Chrome tests (if available)
    if [[ " ${browsers[@]} " =~ " chrome " ]]; then
        run_real_implementation_test "chrome" "webgpu" "4" "bert-base-uncased" ""
        run_real_implementation_test "chrome" "webgpu" "8" "bert-base-uncased" ""
        run_real_implementation_test "chrome" "webnn" "8" "bert-base-uncased" ""
        run_real_implementation_test "chrome" "webgpu" "4" "whisper-tiny" "--model-type audio"
        total_count=$((total_count + 4))
    fi
    
    # Firefox tests (if available)
    if [[ " ${browsers[@]} " =~ " firefox " ]]; then
        run_real_implementation_test "firefox" "webgpu" "4" "bert-base-uncased" ""
        run_real_implementation_test "firefox" "webgpu" "4" "whisper-tiny" "--model-type audio"
        total_count=$((total_count + 2))
    fi
    
    # Edge tests (if available)
    if [[ " ${browsers[@]} " =~ " edge " ]]; then
        run_real_implementation_test "edge" "webnn" "8" "bert-base-uncased" ""
        run_real_implementation_test "edge" "webgpu" "4" "bert-base-uncased" ""
        total_count=$((total_count + 2))
    fi
    
    # Safari tests (if available)
    if [[ " ${browsers[@]} " =~ " safari " ]]; then
        run_real_implementation_test "safari" "webgpu" "8" "bert-base-uncased" ""
        total_count=$((total_count + 1))
    fi
    
else
    # Comprehensive mode - Test all precision levels, multiple models, all browsers
    echo -e "${CYAN}Running Comprehensive tests (all precision levels, multiple models, all browsers)${NC}"
    
    # Define models to test
    models=("bert-base-uncased" "whisper-tiny" "vit-base" "t5-small")
    model_types=("text" "audio" "vision" "text")
    
    # Chrome tests (if available)
    if [[ " ${browsers[@]} " =~ " chrome " ]]; then
        # WebGPU tests - all precision levels
        for precision in 2 3 4 8 16 32; do
            for i in {0..3}; do
                model=${models[$i]}
                model_type=${model_types[$i]}
                
                # Skip ultra-low precision (2-3 bit) for non-text models
                if [ $precision -lt 4 ] && [ "$model_type" != "text" ]; then
                    continue
                fi
                
                run_real_implementation_test "chrome" "webgpu" "$precision" "$model" "--model-type $model_type"
                total_count=$((total_count + 1))
            done
        done
        
        # WebNN tests - standard precision levels
        for precision in 8 16 32; do
            run_real_implementation_test "chrome" "webnn" "$precision" "bert-base-uncased" "--model-type text"
            total_count=$((total_count + 1))
        done
    fi
    
    # Firefox tests (if available)
    if [[ " ${browsers[@]} " =~ " firefox " ]]; then
        # WebGPU tests - focus on audio with 4-bit
        run_real_implementation_test "firefox" "webgpu" "4" "bert-base-uncased" "--model-type text"
        run_real_implementation_test "firefox" "webgpu" "4" "whisper-tiny" "--model-type audio"
        run_real_implementation_test "firefox" "webgpu" "4" "vit-base" "--model-type vision"
        
        # Test other precision levels with bert
        for precision in 8 16 32; do
            run_real_implementation_test "firefox" "webgpu" "$precision" "bert-base-uncased" "--model-type text"
        done
        
        total_count=$((total_count + 6))
    fi
    
    # Edge tests (if available)
    if [[ " ${browsers[@]} " =~ " edge " ]]; then
        # WebNN tests - Edge has best WebNN support
        for precision in 8 16 32; do
            run_real_implementation_test "edge" "webnn" "$precision" "bert-base-uncased" "--model-type text"
            total_count=$((total_count + 1))
        done
        
        # WebGPU tests
        for precision in 4 8 16; do
            run_real_implementation_test "edge" "webgpu" "$precision" "bert-base-uncased" "--model-type text"
            total_count=$((total_count + 1))
        done
    fi
    
    # Safari tests (if available)
    if [[ " ${browsers[@]} " =~ " safari " ]]; then
        # Limited WebGPU support in Safari
        for precision in 8 16 32; do
            run_real_implementation_test "safari" "webgpu" "$precision" "bert-base-uncased" "--model-type text"
            total_count=$((total_count + 1))
        done
    fi
fi

# Generate summary report
report_file="$OUTPUT_DIR/summary_report.md"
echo -e "${CYAN}Generating summary report: $report_file${NC}"

cat > $report_file << EOF
# Real WebNN/WebGPU Precision Testing Report

**Test Date:** $(date +"%Y-%m-%d %H:%M:%S")

## Test Summary

- **Total Tests:** $total_count
- **Real Hardware Acceleration:** $real_count
- **Simulation Mode:** $simulation_count
- **Failed Tests:** $failed_count

## Browser Support Matrix

| Browser | WebNN | WebGPU | Notes |
|---------|-------|--------|-------|
EOF

# Add browser matrix data
for browser in "${browsers[@]}"; do
    case $browser in
        chrome)
            echo "| Chrome | ⚠️ Limited | ✅ Good | Good general WebGPU support |" >> $report_file
            ;;
        firefox)
            echo "| Firefox | ❌ Poor | ✅ Excellent | Best for audio models with special optimizations |" >> $report_file
            ;;
        edge)
            echo "| Edge | ✅ Excellent | ✅ Good | Best WebNN support |" >> $report_file
            ;;
        safari)
            echo "| Safari | ⚠️ Limited | ⚠️ Limited | Limited but improving WebGPU support |" >> $report_file
            ;;
    esac
done

# Add precision information
cat >> $report_file << EOF

## Precision Support Matrix

| Precision | WebNN | WebGPU | Memory Reduction | Use Case |
|-----------|-------|--------|------------------|----------|
| 2-bit | ❌ Not Supported | ✅ Supported | ~87.5% | Ultra memory constrained |
| 3-bit | ❌ Not Supported | ✅ Supported | ~81.25% | Very memory constrained |
| 4-bit | ⚠️ Experimental | ✅ Supported | ~75% | Memory constrained |
| 8-bit | ✅ Supported | ✅ Supported | ~50% | General purpose |
| 16-bit | ✅ Supported | ✅ Supported | ~0% | High accuracy |
| 32-bit | ✅ Supported | ✅ Supported | 0% | Maximum accuracy |

## Test Details

The following tests were run with real browser-based hardware acceleration:

EOF

# Add test details
for log_file in $OUTPUT_DIR/*.log; do
    # Parse log file name to get parameters
    filename=$(basename $log_file)
    platform=$(echo $filename | cut -d'_' -f1)
    browser=$(echo $filename | cut -d'_' -f2)
    model=$(echo $filename | cut -d'_' -f3)
    precision=$(echo $filename | cut -d'_' -f4 | sed 's/bit.log//')
    
    # Extract performance data if available
    if grep -q "Inference completed in" $log_file; then
        latency=$(grep -o "Inference completed in [0-9.]\+ ms" $log_file | awk '{print $4}')
    else
        latency="N/A"
    fi
    
    # Check if real or simulation
    if grep -q "REAL_" $log_file && ! grep -q "simulation: true" $log_file; then
        implementation="✅ REAL"
    else
        implementation="⚠️ SIMULATION"
    fi
    
    echo "### $browser - $platform - $model ($precision-bit)" >> $report_file
    echo "- Implementation: $implementation" >> $report_file
    echo "- Latency: $latency" >> $report_file
    echo >> $report_file
done

echo -e "${GREEN}Summary report generated: $report_file${NC}"

# Archive old documentation if requested
read -p "$(echo -e "${YELLOW}Do you want to archive old WebNN/WebGPU documentation? (y/n): ${NC}")" archive_docs

if [ "$archive_docs" = "y" ] || [ "$archive_docs" = "Y" ]; then
    echo -e "${CYAN}Archiving old documentation...${NC}"
    
    if [ -f "archive_webnn_webgpu_docs.py" ]; then
        python archive_webnn_webgpu_docs.py --archive-dir archived_md_files
        echo -e "${GREEN}Old documentation archived to archived_md_files${NC}"
    else
        echo -e "${RED}archive_webnn_webgpu_docs.py not found${NC}"
        
        # Create archive directory
        mkdir -p archived_md_files
        
        # Archive old docs manually
        for doc in WEBNN_WEBGPU_QUANTIZATION_README.md WEBGPU_WEBNN_QUANTIZATION_SUMMARY.md WEBGPU_SIMULATION_GUIDE.md WEBNN_SIMULATION_GUIDE.md; do
            if [ -f "$doc" ]; then
                cp "$doc" "archived_md_files/${doc%.md}_$(date +%Y%m%d_%H%M%S).md"
                echo -e "  ${GREEN}✓ Archived $doc${NC}"
            fi
        done
    fi
fi

# Final summary
echo
echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}Testing Complete!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo
echo -e "${CYAN}Summary:${NC}"
echo -e "  Total tests: ${YELLOW}$total_count${NC}"
echo -e "  Real hardware acceleration: ${GREEN}$real_count${NC}"
echo -e "  Simulation mode: ${RED}$simulation_count${NC}"
echo -e "  Failed tests: ${RED}$failed_count${NC}"
echo
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR${NC}"
echo -e "Summary report: ${BLUE}$report_file${NC}"
if [ "$archive_docs" = "y" ] || [ "$archive_docs" = "Y" ]; then
    echo -e "Documentation archived in: ${BLUE}archived_md_files${NC}"
fi
echo -e "See ${BLUE}REAL_WEBNN_WEBGPU_TESTING.md${NC} for comprehensive guide"
echo