#\!/bin/bash

# Verify mock detection system with environment variable control
#
# This script tests the mock detection system with different environment 
# variable settings to verify that it correctly identifies when tests
# are using real inference vs. mock objects.

# Colors for better visibility
GREEN="\033[0;32m"
BLUE="\033[0;34m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

FIXED_TESTS_DIR="fixed_tests"
TEST_RESULTS_DIR="verify_results"

# Ensure directories exist
mkdir -p "$TEST_RESULTS_DIR"

echo -e "${GREEN}=== Mock Detection Verification Script ===${RESET}"
echo "Testing mock detection system with environment variables"
echo

# Model types to test
MODEL_TYPES=("bert" "gpt2" "t5" "vit")

# Verify a specific model with specified environment variables
verify_model() {
    local model=$1
    local torch_mock=$2
    local transformers_mock=$3
    local test_name="${model}_torch${torch_mock}_transformers${transformers_mock}"
    local test_file="${FIXED_TESTS_DIR}/test_hf_${model}.py"
    local result_file="${TEST_RESULTS_DIR}/${test_name}.txt"
    
    echo -e "\n${YELLOW}Testing ${model} with MOCK_TORCH=${torch_mock}, MOCK_TRANSFORMERS=${transformers_mock}${RESET}"
    
    # Set environment variables and run the test
    MOCK_TORCH=${torch_mock} MOCK_TRANSFORMERS=${transformers_mock} python ${test_file} 2>&1 | tee "${result_file}"
    
    # Check for mock indicator in results
    if grep -q "🚀 Using REAL INFERENCE" "${result_file}"; then
        echo -e "${GREEN}✅ Test detected REAL INFERENCE${RESET}"
        REAL_COUNT=$((REAL_COUNT+1))
    elif grep -q "🔷 Using MOCK OBJECTS" "${result_file}"; then
        echo -e "${BLUE}✅ Test detected MOCK OBJECTS${RESET}"
        MOCK_COUNT=$((MOCK_COUNT+1))
    else
        echo -e "${RED}❌ Could not determine test type${RESET}"
        UNKNOWN_COUNT=$((UNKNOWN_COUNT+1))
    fi
}

# Run verification for all model types with different mock combinations
run_all_verifications() {
    # Initialize counters
    REAL_COUNT=0
    MOCK_COUNT=0
    UNKNOWN_COUNT=0
    
    # Run tests with different mock combinations
    for model in "${MODEL_TYPES[@]}"; do
        # Both real (no mocking)
        verify_model "${model}" "False" "False"
        
        # Mock torch only
        verify_model "${model}" "True" "False"
        
        # Mock transformers only
        verify_model "${model}" "False" "True"
        
        # Mock both
        verify_model "${model}" "True" "True"
    done
    
    # Show summary
    echo -e "\n${GREEN}=== Verification Summary ===${RESET}"
    echo -e "Total tests run: $((REAL_COUNT + MOCK_COUNT + UNKNOWN_COUNT))"
    echo -e "${GREEN}✅ Tests with REAL INFERENCE: ${REAL_COUNT}${RESET}"
    echo -e "${BLUE}✅ Tests with MOCK OBJECTS: ${MOCK_COUNT}${RESET}"
    echo -e "${RED}❌ Tests with UNKNOWN detection: ${UNKNOWN_COUNT}${RESET}"
    
    if [ ${UNKNOWN_COUNT} -eq 0 ]; then
        echo -e "\n${GREEN}✅ All tests successfully detected inference type${RESET}"
    else
        echo -e "\n${RED}❌ Some tests failed to detect inference type${RESET}"
    fi
}

# Run all verifications
run_all_verifications
