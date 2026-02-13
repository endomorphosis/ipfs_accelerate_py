#!/bin/bash

# Run a test file with different mock configurations
#
# This script demonstrates how to control the mock detection system
# using environment variables.

# ANSI color codes for terminal output
GREEN="\033[32m"
BLUE="\033[34m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# Default settings
TEST_FILE=""
TORCH="False"
TRANSFORMERS="False"
TOKENIZERS="False"
SENTENCEPIECE="False"
OUTPUT_FILE=""
CAPTURE_OUTPUT=false

# Help message
show_help() {
    echo -e "${GREEN}Run Test with Mock Control${RESET}"
    echo
    echo "Usage: $0 -f TEST_FILE [options]"
    echo
    echo "Options:"
    echo "  -f, --file FILE        Test file to run (required)"
    echo "  --mock-torch           Mock torch (default: False)"
    echo "  --mock-transformers    Mock transformers (default: False)"
    echo "  --mock-tokenizers      Mock tokenizers (default: False)"
    echo "  --mock-sentencepiece   Mock sentencepiece (default: False)"
    echo "  --all-real             Run with all dependencies real (default)"
    echo "  --all-mock             Run with all dependencies mocked"
    echo "  -o, --output FILE      Save output to file"
    echo "  --capture              Capture output to auto-generated file"
    echo "  -h, --help             Show this help message"
    echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            TEST_FILE="$2"
            shift 2
            ;;
        --mock-torch)
            TORCH="True"
            shift
            ;;
        --mock-transformers)
            TRANSFORMERS="True"
            shift
            ;;
        --mock-tokenizers)
            TOKENIZERS="True"
            shift
            ;;
        --mock-sentencepiece)
            SENTENCEPIECE="True"
            shift
            ;;
        --all-real)
            TORCH="False"
            TRANSFORMERS="False"
            TOKENIZERS="False"
            SENTENCEPIECE="False"
            shift
            ;;
        --all-mock)
            TORCH="True"
            TRANSFORMERS="True"
            TOKENIZERS="True"
            SENTENCEPIECE="True"
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --capture)
            CAPTURE_OUTPUT=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Validate input
if [ -z "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file is required${RESET}"
    show_help
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file not found: $TEST_FILE${RESET}"
    exit 1
fi

# Generate output file name if requested but not specified
if [ "$CAPTURE_OUTPUT" = true ] && [ -z "$OUTPUT_FILE" ]; then
    # Extract model name from file path (e.g., test_hf_bert.py -> bert)
    MODEL_NAME=$(basename "$TEST_FILE" | sed 's/test_hf_\(.*\)\.py/\1/')
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CONFIG_STRING="torch${TORCH}_transformers${TRANSFORMERS}_tokenizers${TOKENIZERS}_sentencepiece${SENTENCEPIECE}"
    OUTPUT_FILE="test_output_${MODEL_NAME}_${CONFIG_STRING}_${TIMESTAMP}.txt"
fi

# Show configuration
echo -e "${GREEN}=== Running Test with Mock Control ===${RESET}"
echo -e "Test file: $TEST_FILE"
echo -e "Mock configuration:"
echo -e "  - MOCK_TORCH: $TORCH"
echo -e "  - MOCK_TRANSFORMERS: $TRANSFORMERS"
echo -e "  - MOCK_TOKENIZERS: $TOKENIZERS"
echo -e "  - MOCK_SENTENCEPIECE: $SENTENCEPIECE"
if [ -n "$OUTPUT_FILE" ]; then
    echo -e "Output file: $OUTPUT_FILE"
fi
echo

# Create test_output directory if it doesn't exist (when using output file)
if [ -n "$OUTPUT_FILE" ]; then
    mkdir -p test_output
    OUTPUT_FILE="test_output/$OUTPUT_FILE"
fi

# Run the test with the specified configuration
echo -e "${BLUE}Running test...${RESET}"

# Start capturing output if requested
if [ -n "$OUTPUT_FILE" ]; then
    # Save configuration information to the output file
    {
        echo "=== Test Configuration ==="
        echo "Test file: $TEST_FILE"
        echo "Mock configuration:"
        echo "  - MOCK_TORCH: $TORCH"
        echo "  - MOCK_TRANSFORMERS: $TRANSFORMERS"
        echo "  - MOCK_TOKENIZERS: $TOKENIZERS"
        echo "  - MOCK_SENTENCEPIECE: $SENTENCEPIECE"
        echo "Timestamp: $(date)"
        echo "Command: MOCK_TORCH=$TORCH MOCK_TRANSFORMERS=$TRANSFORMERS MOCK_TOKENIZERS=$TOKENIZERS MOCK_SENTENCEPIECE=$SENTENCEPIECE python \"$TEST_FILE\""
        echo "=== Test Output ==="
        echo
    } > "$OUTPUT_FILE"
    
    # Run test and capture output
    MOCK_TORCH=$TORCH MOCK_TRANSFORMERS=$TRANSFORMERS MOCK_TOKENIZERS=$TOKENIZERS MOCK_SENTENCEPIECE=$SENTENCEPIECE python "$TEST_FILE" 2>&1 | tee -a "$OUTPUT_FILE"
    TEST_EXIT_CODE=${PIPESTATUS[0]}
    
    # Log the completion and exit code
    {
        echo
        echo "=== Test Completion ==="
        echo "Exit code: $TEST_EXIT_CODE"
        echo "Completed at: $(date)"
    } >> "$OUTPUT_FILE"
    
    echo -e "\n${GREEN}Output saved to: $OUTPUT_FILE${RESET}"
else
    # Run test normally
    MOCK_TORCH=$TORCH MOCK_TRANSFORMERS=$TRANSFORMERS MOCK_TOKENIZERS=$TOKENIZERS MOCK_SENTENCEPIECE=$SENTENCEPIECE python "$TEST_FILE"
    TEST_EXIT_CODE=$?
fi

# Display the command
echo
echo -e "${YELLOW}Command used:${RESET}"
echo "MOCK_TORCH=$TORCH MOCK_TRANSFORMERS=$TRANSFORMERS MOCK_TOKENIZERS=$TOKENIZERS MOCK_SENTENCEPIECE=$SENTENCEPIECE python \"$TEST_FILE\""

# Exit with the same code as the test
exit $TEST_EXIT_CODE