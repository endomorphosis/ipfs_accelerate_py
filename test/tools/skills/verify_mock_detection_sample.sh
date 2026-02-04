#!/bin/bash

# Sample verification script to test the verify_all_mock_detection.py script
# on a small subset of files before running it on all files.

# Set up colors for better readability
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RESET="\033[0m"

echo -e "${GREEN}=== Mock Detection Sample Verification ===${RESET}"
echo -e "Testing mock detection verification script on sample files"
echo

# Define sample files to test
SAMPLE_FILES=(
    "fixed_tests/test_hf_bert.py"
    "fixed_tests/test_hf_gpt2.py"
    "fixed_tests/test_hf_t5.py"
    "fixed_tests/test_hf_vit.py"
)

# Run verification for each sample file
for file in "${SAMPLE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}Testing file: ${file}${RESET}"
        python verify_all_mock_detection.py --file "$file" --check-only
        echo
    else
        echo -e "${RED}File not found: ${file}${RESET}"
    fi
done

echo -e "${GREEN}=== Complete Sample Test ===${RESET}"
echo -e "Running a complete test on one file with check, fix, and verify"
echo

# Run a complete test on one file
python verify_all_mock_detection.py --file "fixed_tests/test_hf_bert.py" --fix --verify

echo -e "\n${GREEN}=== Sample Verification Complete ===${RESET}"
echo "If the sample verification looks good, you can now run the full verification:"
echo "python verify_all_mock_detection.py --fix --verify"