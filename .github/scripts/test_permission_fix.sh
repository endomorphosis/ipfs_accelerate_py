#!/bin/bash

# Test script for runner permission fix
# This validates that the fix works correctly

set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Runner Permission Fix - Validation Test                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PASS=0
FAIL=0

test_result() {
    local name="$1"
    local result="$2"
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $name"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $name"
        ((FAIL++))
    fi
}

# Test 1: Check if fix script exists and is executable
echo "Testing fix script..."
if [ -x ".github/scripts/fix_runner_permissions.sh" ]; then
    test_result "Fix script exists and is executable" "PASS"
else
    test_result "Fix script exists and is executable" "FAIL"
fi

# Test 2: Check if cleanup workflow exists
echo "Testing cleanup workflow..."
if [ -f ".github/workflows/runner-cleanup.yml" ]; then
    test_result "Cleanup workflow exists" "PASS"
else
    test_result "Cleanup workflow exists" "FAIL"
fi

# Test 3: Check if cleanup action exists
echo "Testing cleanup action..."
if [ -f ".github/actions/cleanup-workspace/action.yml" ]; then
    test_result "Cleanup action exists" "PASS"
else
    test_result "Cleanup action exists" "FAIL"
fi

# Test 4: Validate cleanup action syntax
echo "Testing cleanup action syntax..."
if grep -q "uses: 'composite'" ".github/actions/cleanup-workspace/action.yml"; then
    test_result "Cleanup action has correct syntax" "PASS"
else
    test_result "Cleanup action has correct syntax" "FAIL"
fi

# Test 5: Check if cleanup workflow has correct schedule
echo "Testing cleanup workflow schedule..."
if grep -q "cron:.*0 \*/6 \* \* \*" ".github/workflows/runner-cleanup.yml"; then
    test_result "Cleanup workflow scheduled for every 6 hours" "PASS"
else
    test_result "Cleanup workflow scheduled for every 6 hours" "FAIL"
fi

# Test 6: Check documentation
echo "Testing documentation..."
if [ -f "RUNNER_PERMISSION_FIX_GUIDE.md" ]; then
    test_result "Main documentation exists" "PASS"
else
    test_result "Main documentation exists" "FAIL"
fi

# Test 7: Check quick reference
if [ -f "RUNNER_PERMISSION_FIX_QUICK_REF.md" ]; then
    test_result "Quick reference exists" "PASS"
else
    test_result "Quick reference exists" "FAIL"
fi

# Test 8: Check workflow examples
if [ -f ".github/workflows/WORKFLOW_UPDATE_EXAMPLES.md" ]; then
    test_result "Workflow examples exist" "PASS"
else
    test_result "Workflow examples exist" "FAIL"
fi

# Test 9: Check action documentation
if [ -f ".github/actions/cleanup-workspace/README.md" ]; then
    test_result "Action documentation exists" "PASS"
else
    test_result "Action documentation exists" "FAIL"
fi

# Test 10: Validate fix script has required functions
echo "Testing fix script functions..."
if grep -q "fix_runner_permissions()" ".github/scripts/fix_runner_permissions.sh"; then
    test_result "Fix script has required functions" "PASS"
else
    test_result "Fix script has required functions" "FAIL"
fi

# Test 11: Check if fix script handles multiple runners
if grep -q "RUNNER_LOCATIONS" ".github/scripts/fix_runner_permissions.sh"; then
    test_result "Fix script supports multiple runners" "PASS"
else
    test_result "Fix script supports multiple runners" "FAIL"
fi

# Test 12: Validate cleanup action removes lock files
if grep -q "find.*\.lock.*delete" ".github/actions/cleanup-workspace/action.yml"; then
    test_result "Cleanup action removes lock files" "PASS"
else
    test_result "Cleanup action removes lock files" "FAIL"
fi

# Test 13: Validate cleanup action fixes git permissions
if grep -q "chmod.*\.git" ".github/actions/cleanup-workspace/action.yml"; then
    test_result "Cleanup action fixes git permissions" "PASS"
else
    test_result "Cleanup action fixes git permissions" "FAIL"
fi

# Test 14: Check if cleanup action is idempotent (continue-on-error)
if grep -q "continue-on-error: true" ".github/actions/cleanup-workspace/action.yml"; then
    test_result "Cleanup action is idempotent" "PASS"
else
    test_result "Cleanup action is idempotent" "FAIL"
fi

# Test 15: Verify workflow has workflow_call trigger
if grep -q "workflow_call:" ".github/workflows/runner-cleanup.yml"; then
    test_result "Cleanup workflow can be called by other workflows" "PASS"
else
    test_result "Cleanup workflow can be called by other workflows" "FAIL"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Test Results Summary                                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC} $PASS"
echo -e "  ${RED}Failed:${NC} $FAIL"
echo -e "  ${BLUE}Total:${NC}  $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Run the fix script: ./.github/scripts/fix_runner_permissions.sh"
    echo "  2. Update your workflows to include pre-job cleanup"
    echo "  3. Monitor the automated cleanup runs"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
    echo ""
    exit 1
fi
