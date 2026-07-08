#!/bin/bash

# Verify and fix all HuggingFace test files
# This script runs the comprehensive_test_fix.py script on all test files in the fixed_tests directory
# It provides detailed reporting and verification of fixes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

# Log file
LOG_FILE="verify_and_fix_$(date +%Y%m%d_%H%M%S).log"
echo "Starting verification and fix process at $(date)" | tee -a "$LOG_FILE"

# Check if comprehensive_test_fix.py exists
if [ ! -f "comprehensive_test_fix.py" ]; then
    echo -e "${RED}Error: comprehensive_test_fix.py not found in current directory${RESET}" | tee -a "$LOG_FILE"
    exit 1
fi

# Ensure the script is executable
chmod +x comprehensive_test_fix.py

# Step 1: Check all files to identify issues without fixing
echo -e "\n${BLUE}=== STEP 1: Checking all test files for issues ===${RESET}" | tee -a "$LOG_FILE"
python comprehensive_test_fix.py --check-only | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${YELLOW}Issues found in test files, proceeding to fix stage${RESET}" | tee -a "$LOG_FILE"
else
    echo -e "${GREEN}All test files have complete mock detection implementations${RESET}" | tee -a "$LOG_FILE"
    exit 0
fi

# Step 2: Fix all files
echo -e "\n${BLUE}=== STEP 2: Applying fixes to all test files ===${RESET}" | tee -a "$LOG_FILE"
python comprehensive_test_fix.py | tee -a "$LOG_FILE"
FIX_RESULT=$?

# Step 3: Verify files with different mock configurations
echo -e "\n${BLUE}=== STEP 3: Verifying test files with different mock configurations ===${RESET}" | tee -a "$LOG_FILE"

# Get list of files to verify - focus on previously fixed files if any failures
if [ $FIX_RESULT -ne 0 ]; then
    # Find files that were successfully fixed by checking the summary file
    LATEST_SUMMARY=$(ls -t fix_summary_*.txt | head -1)
    if [ -n "$LATEST_SUMMARY" ]; then
        echo -e "${YELLOW}Some files could not be fixed, verifying only successfully fixed files${RESET}" | tee -a "$LOG_FILE"
        FILES_TO_VERIFY=$(grep "^✅" "$LATEST_SUMMARY" | cut -d ":" -f 1 | sed 's/✅ /fixed_tests\//')
    else
        echo -e "${RED}No summary file found, cannot determine which files were fixed${RESET}" | tee -a "$LOG_FILE"
        FILES_TO_VERIFY=""
    fi
else
    # All files were fixed successfully, verify all of them
    FILES_TO_VERIFY=$(find fixed_tests -name "test_hf_*.py")
fi

# Count number of files to verify
NUM_FILES=$(echo "$FILES_TO_VERIFY" | wc -l)
echo -e "Verifying $NUM_FILES test files..." | tee -a "$LOG_FILE"

# Verify each file
VERIFY_SUCCESS=0
VERIFY_FAIL=0
for file in $FILES_TO_VERIFY; do
    echo -e "\n${YELLOW}Verifying $file...${RESET}" | tee -a "$LOG_FILE"
    python comprehensive_test_fix.py --file "$file" --verify | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        VERIFY_SUCCESS=$((VERIFY_SUCCESS + 1))
    else
        VERIFY_FAIL=$((VERIFY_FAIL + 1))
    fi
done

# Generate final report
echo -e "\n${BLUE}=== FINAL REPORT ===${RESET}" | tee -a "$LOG_FILE"
echo -e "Fix results:" | tee -a "$LOG_FILE"
if [ $FIX_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ All files were successfully fixed${RESET}" | tee -a "$LOG_FILE"
else
    echo -e "${YELLOW}⚠️ Some files could not be fixed${RESET}" | tee -a "$LOG_FILE"
fi

echo -e "\nVerification results:" | tee -a "$LOG_FILE"
echo -e "${GREEN}✅ Files verified successfully: $VERIFY_SUCCESS${RESET}" | tee -a "$LOG_FILE"
echo -e "${RED}❌ Files with verification issues: $VERIFY_FAIL${RESET}" | tee -a "$LOG_FILE"

echo -e "\nLog file: $LOG_FILE" | tee -a "$LOG_FILE"
echo -e "Completed at $(date)" | tee -a "$LOG_FILE"

if [ $FIX_RESULT -eq 0 ] && [ $VERIFY_FAIL -eq 0 ]; then
    echo -e "\n${GREEN}✅ SUCCESS: All files were fixed and verified successfully${RESET}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "\n${YELLOW}⚠️ PARTIAL SUCCESS: Some issues remain${RESET}" | tee -a "$LOG_FILE"
    exit 1
fi