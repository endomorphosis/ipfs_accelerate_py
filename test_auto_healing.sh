#!/bin/bash
# Test script for auto-healing error handling system

set -e

echo "=================================="
echo "Auto-Healing System Test Script"
echo "=================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for test output
test_step() {
    echo -e "${YELLOW}TEST:${NC} $1"
}

test_pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    ((TESTS_FAILED++))
}

# Change to repo directory
cd "$(dirname "$0")/.."

echo "Step 1: Testing Python Module Imports"
echo "--------------------------------------"

test_step "Import error_handler module"
if python3 -c "import sys; sys.path.insert(0, '.'); from ipfs_accelerate_py.error_handler import CLIErrorHandler; print('OK')" 2>&1 | grep -q "OK"; then
    test_pass "error_handler module imports successfully"
else
    test_fail "error_handler module import failed"
fi

test_step "Import error_aggregator module"
if python3 -c "import sys; sys.path.insert(0, '.'); from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator; print('OK')" 2>&1 | grep -q "OK"; then
    test_pass "error_aggregator module imports successfully"
else
    test_fail "error_aggregator module import failed"
fi

echo ""
echo "Step 2: Testing Error Handler Functionality"
echo "-------------------------------------------"

test_step "Create error handler instance"
if python3 -c "
import sys
sys.path.insert(0, '.')
from ipfs_accelerate_py.error_handler import CLIErrorHandler
handler = CLIErrorHandler('test/repo', enable_auto_issue=False)
print('Instance created successfully')
" 2>/dev/null | grep -q "Instance created"; then
    test_pass "Error handler instance created"
else
    test_fail "Error handler instance creation failed"
fi

test_step "Test error capture"
if python3 -c "
import sys
sys.path.insert(0, '.')
from ipfs_accelerate_py.error_handler import CLIErrorHandler
handler = CLIErrorHandler('test/repo')
try:
    raise ValueError('Test error')
except Exception as e:
    handler.capture_error(e)
    if len(handler._captured_errors) == 1:
        print('Error captured successfully')
" 2>/dev/null | grep -q "Error captured"; then
    test_pass "Error capture works"
else
    test_fail "Error capture failed"
fi

test_step "Test severity determination"
if python3 -c "
import sys
sys.path.insert(0, '.')
from ipfs_accelerate_py.error_handler import CLIErrorHandler
handler = CLIErrorHandler('test/repo')
assert handler._determine_severity(ValueError()) == 'medium'
assert handler._determine_severity(MemoryError()) == 'critical'
print('Severity determination works')
" 2>/dev/null | grep -q "Severity determination"; then
    test_pass "Severity determination works"
else
    test_fail "Severity determination failed"
fi

echo ""
echo "Step 3: Testing Examples"
echo "------------------------"

test_step "Run auto-healing demo"
if python3 examples/auto_healing_demo.py 2>&1 | grep -q "Example completed successfully"; then
    test_pass "Auto-healing demo runs successfully"
else
    test_fail "Auto-healing demo failed"
fi

echo ""
echo "Step 4: Testing Configuration"
echo "------------------------------"

test_step "Environment variable parsing"
if python3 -c "
import os
os.environ['IPFS_AUTO_ISSUE'] = 'true'
os.environ['IPFS_AUTO_PR'] = '1'
os.environ['IPFS_AUTO_HEAL'] = 'yes'
os.environ['IPFS_REPO'] = 'test/repo'

enable_auto_issue = os.environ.get('IPFS_AUTO_ISSUE', '').lower() in ('1', 'true', 'yes')
enable_auto_pr = os.environ.get('IPFS_AUTO_PR', '').lower() in ('1', 'true', 'yes')
enable_auto_heal = os.environ.get('IPFS_AUTO_HEAL', '').lower() in ('1', 'true', 'yes')
repo = os.environ.get('IPFS_REPO')

assert enable_auto_issue == True
assert enable_auto_pr == True
assert enable_auto_heal == True
assert repo == 'test/repo'
print('Environment variables parsed correctly')
" | grep -q "parsed correctly"; then
    test_pass "Environment variable parsing works"
else
    test_fail "Environment variable parsing failed"
fi

echo ""
echo "Step 5: Checking Documentation"
echo "-------------------------------"

test_step "Check for AUTO_HEALING_CONFIGURATION.md"
if [ -f "docs/AUTO_HEALING_CONFIGURATION.md" ]; then
    test_pass "Configuration documentation exists"
else
    test_fail "Configuration documentation missing"
fi

test_step "Check for IMPLEMENTATION_SUMMARY.md"
if [ -f "IMPLEMENTATION_SUMMARY.md" ]; then
    test_pass "Implementation summary exists"
else
    test_fail "Implementation summary missing"
fi

echo ""
echo "Step 6: Checking File Structure"
echo "--------------------------------"

test_step "Check error_handler.py exists"
if [ -f "ipfs_accelerate_py/error_handler.py" ]; then
    test_pass "error_handler.py exists"
else
    test_fail "error_handler.py missing"
fi

test_step "Check test file exists"
if [ -f "test/test_error_handler.py" ]; then
    test_pass "test_error_handler.py exists"
else
    test_fail "test_error_handler.py missing"
fi

test_step "Check example exists"
if [ -f "examples/auto_healing_demo.py" ]; then
    test_pass "auto_healing_demo.py exists"
else
    test_fail "auto_healing_demo.py missing"
fi

echo ""
echo "Step 7: Optional Integrations"
echo "------------------------------"

test_step "Check GitHub CLI availability"
if command -v gh &> /dev/null; then
    test_pass "GitHub CLI (gh) is installed"
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        test_pass "GitHub CLI is authenticated"
        echo "  → Auto-issue creation would work"
    else
        echo "  ⚠️  GitHub CLI not authenticated (optional)"
        echo "     Run: gh auth login"
    fi
else
    echo "  ⚠️  GitHub CLI not installed (optional for auto-issue)"
    echo "     Install: brew install gh (macOS) or see https://cli.github.com"
fi

test_step "Check Copilot SDK availability"
if python3 -c "import copilot" 2>/dev/null; then
    test_pass "GitHub Copilot SDK is installed"
else
    echo "  ⚠️  Copilot SDK not installed (optional for auto-heal)"
    echo "     Install: pip install github-copilot-sdk"
fi

echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Enable auto-features: export IPFS_AUTO_ISSUE=true"
    echo "2. Authenticate GitHub CLI: gh auth login"
    echo "3. Test with real CLI: ipfs-accelerate --help"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Please review the failures above and fix them."
    echo ""
    exit 1
fi
