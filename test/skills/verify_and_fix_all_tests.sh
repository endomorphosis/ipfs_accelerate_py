#!/bin/bash

# Verify and fix mock detection in all test files
#
# This script:
# 1. Checks all test files for proper mock detection implementation
# 2. Fixes any issues found using the enhanced fix script
# 3. Verifies that the fixes work properly with different environment variables
# 4. Generates a comprehensive report of the results

# ANSI color codes for terminal output
GREEN="\033[32m"
BLUE="\033[34m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# Default settings
TEST_DIR="fixed_tests"
MAX_WORKERS=4
VERIFY_TIMEOUT=120

# Help message
show_help() {
    echo -e "${GREEN}Verify and Fix Mock Detection in Test Files${RESET}"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -d, --dir DIR      Directory containing test files (default: fixed_tests)"
    echo "  -w, --workers N    Maximum number of parallel workers (default: 4)"
    echo "  -t, --timeout N    Verification timeout in seconds (default: 120)"
    echo "  -c, --check-only   Only check files without fixing"
    echo "  -h, --help         Show this help message"
    echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            TEST_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -t|--timeout)
            VERIFY_TIMEOUT="$2"
            shift 2
            ;;
        -c|--check-only)
            CHECK_ONLY=true
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

# Ensure test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $TEST_DIR${RESET}"
    exit 1
fi

# Create timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="mock_detection_${TIMESTAMP}.log"
REPORT_FILE="mock_detection_report_${TIMESTAMP}.txt"

echo -e "${GREEN}=== Mock Detection Verification and Fixing ===${RESET}" | tee -a "$LOG_FILE"
echo -e "Starting verification at $(date)" | tee -a "$LOG_FILE"
echo -e "Test directory: $TEST_DIR" | tee -a "$LOG_FILE"
echo -e "Max workers: $MAX_WORKERS" | tee -a "$LOG_FILE"
echo -e "Timeout: $VERIFY_TIMEOUT seconds" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# Step 1: Check all files
echo -e "${BLUE}Step 1: Checking all files for proper mock detection${RESET}" | tee -a "$LOG_FILE"
echo -e "Running: python verify_all_mock_detection.py --dir $TEST_DIR --check-only --max-workers $MAX_WORKERS --output $REPORT_FILE" | tee -a "$LOG_FILE"

python verify_all_mock_detection.py --dir "$TEST_DIR" --check-only --max-workers "$MAX_WORKERS" --output "$REPORT_FILE"
CHECK_STATUS=$?

if [ $CHECK_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ All files have proper mock detection${RESET}" | tee -a "$LOG_FILE"
    exit 0
fi

# If check-only flag is set, exit now
if [ "$CHECK_ONLY" = true ]; then
    echo -e "${YELLOW}Check-only flag set, exiting without fixing${RESET}" | tee -a "$LOG_FILE"
    exit $CHECK_STATUS
fi

# Step 2: Fix issues
echo -e "\n${BLUE}Step 2: Fixing files with missing or incorrect mock detection${RESET}" | tee -a "$LOG_FILE"

# Extract files needing fixes from the report
NEEDS_FIX=$(grep -B 1 "CHECK: ⚠️ Needs Fix" "$REPORT_FILE" | grep "---" | sed 's/--- //' | sed 's/ ---//')

if [ -z "$NEEDS_FIX" ]; then
    echo -e "${YELLOW}No files need fixing${RESET}" | tee -a "$LOG_FILE"
else
    echo -e "Found $(echo "$NEEDS_FIX" | wc -l) files needing fixes:" | tee -a "$LOG_FILE"
    echo "$NEEDS_FIX" | tee -a "$LOG_FILE"
    echo | tee -a "$LOG_FILE"

    # Fix each file
    for file in $NEEDS_FIX; do
        echo -e "${YELLOW}Fixing $file...${RESET}" | tee -a "$LOG_FILE"
        python fix_all_mock_checks.py --file "$TEST_DIR/$file" 2>&1 | tee -a "$LOG_FILE"
        FIX_STATUS=$?
        
        if [ $FIX_STATUS -eq 0 ]; then
            echo -e "${GREEN}✅ Fixed $file${RESET}" | tee -a "$LOG_FILE"
        else
            echo -e "${RED}❌ Failed to fix $file${RESET}" | tee -a "$LOG_FILE"
        fi
    done
fi

# Step 3: Verify fixes
echo -e "\n${BLUE}Step 3: Verifying all files with different environment variables${RESET}" | tee -a "$LOG_FILE"
echo -e "Running: python verify_all_mock_detection.py --dir $TEST_DIR --verify --max-workers $MAX_WORKERS --output $REPORT_FILE" | tee -a "$LOG_FILE"

python verify_all_mock_detection.py --dir "$TEST_DIR" --verify --max-workers "$MAX_WORKERS" --output "$REPORT_FILE"
VERIFY_STATUS=$?

if [ $VERIFY_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ All files passed verification${RESET}" | tee -a "$LOG_FILE"
else
    echo -e "${RED}❌ Some files failed verification${RESET}" | tee -a "$LOG_FILE"
    
    # Extract failed files from the report
    FAILED_VERIFY=$(grep -B 1 "VERIFY: ❌" "$REPORT_FILE" | grep "---" | sed 's/--- //' | sed 's/ ---//')
    
    if [ -n "$FAILED_VERIFY" ]; then
        echo -e "Files that failed verification:" | tee -a "$LOG_FILE"
        echo "$FAILED_VERIFY" | tee -a "$LOG_FILE"
    fi
fi

# Final summary
echo -e "\n${GREEN}=== Verification and Fixing Complete ===${RESET}" | tee -a "$LOG_FILE"
echo -e "Completed at $(date)" | tee -a "$LOG_FILE"
echo -e "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo -e "Report file: $REPORT_FILE" | tee -a "$LOG_FILE"

# Exit with appropriate status
if [ $VERIFY_STATUS -eq 0 ]; then
    exit 0
else
    exit 1
fi