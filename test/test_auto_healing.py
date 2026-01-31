#!/usr/bin/env python3
"""
Simple test runner for auto-healing error handling system.
"""

import sys
import os
from pathlib import Path

# Determine the repository root
script_path = Path(__file__).resolve()
repo_root = script_path.parent

# Add repository root to path
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

tests_passed = 0
tests_failed = 0


def test_step(msg):
    print(f"{YELLOW}TEST:{NC} {msg}")


def test_pass(msg):
    global tests_passed
    print(f"{GREEN}✓ PASS:{NC} {msg}")
    tests_passed += 1


def test_fail(msg):
    global tests_failed
    print(f"{RED}✗ FAIL:{NC} {msg}")
    tests_failed += 1


def run_tests():
    print("=" * 60)
    print("Auto-Healing System Test Runner")
    print("=" * 60)
    print()
    
    # Test 1: Import error_handler
    print("Step 1: Testing Python Module Imports")
    print("-" * 60)
    
    test_step("Import error_handler module")
    try:
        from ipfs_accelerate_py.error_handler import CLIErrorHandler
        test_pass("error_handler module imports successfully")
    except ImportError as e:
        test_fail(f"error_handler module import failed: {e}")
    
    test_step("Import error_aggregator module")
    try:
        from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
        test_pass("error_aggregator module imports successfully")
    except ImportError as e:
        test_fail(f"error_aggregator module import failed: {e}")
    
    print()
    
    # Test 2: Error Handler Functionality
    print("Step 2: Testing Error Handler Functionality")
    print("-" * 60)
    
    test_step("Create error handler instance")
    try:
        from ipfs_accelerate_py.error_handler import CLIErrorHandler
        handler = CLIErrorHandler('test/repo', enable_auto_issue=False)
        test_pass("Error handler instance created")
    except Exception as e:
        test_fail(f"Error handler instance creation failed: {e}")
    
    test_step("Test error capture")
    try:
        from ipfs_accelerate_py.error_handler import CLIErrorHandler
        handler = CLIErrorHandler('test/repo')
        try:
            raise ValueError('Test error')
        except Exception as e:
            handler.capture_error(e)
            if len(handler._captured_errors) == 1:
                test_pass("Error capture works")
            else:
                test_fail(f"Error capture failed: expected 1 error, got {len(handler._captured_errors)}")
    except Exception as e:
        test_fail(f"Error capture test failed: {e}")
    
    test_step("Test severity determination")
    try:
        from ipfs_accelerate_py.error_handler import CLIErrorHandler
        handler = CLIErrorHandler('test/repo')
        
        if handler._determine_severity(ValueError()) != 'medium':
            test_fail(f"ValueError severity wrong: {handler._determine_severity(ValueError())}")
        elif handler._determine_severity(MemoryError()) != 'critical':
            test_fail(f"MemoryError severity wrong: {handler._determine_severity(MemoryError())}")
        else:
            test_pass("Severity determination works")
    except Exception as e:
        test_fail(f"Severity determination test failed: {e}")
    
    print()
    
    # Test 3: Examples
    print("Step 3: Testing Examples")
    print("-" * 60)
    
    test_step("Check auto-healing demo exists")
    demo_path = repo_root / 'examples' / 'auto_healing_demo.py'
    if demo_path.exists():
        test_pass("Auto-healing demo file exists")
        
        test_step("Run auto-healing demo")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(demo_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and 'Example completed successfully' in result.stdout:
                test_pass("Auto-healing demo runs successfully")
            else:
                test_fail(f"Auto-healing demo failed with code {result.returncode}")
        except Exception as e:
            test_fail(f"Auto-healing demo execution failed: {e}")
    else:
        test_fail("Auto-healing demo file missing")
    
    print()
    
    # Test 4: Documentation
    print("Step 4: Checking Documentation")
    print("-" * 60)
    
    test_step("Check AUTO_HEALING_CONFIGURATION.md")
    config_doc = repo_root / 'docs' / 'AUTO_HEALING_CONFIGURATION.md'
    if config_doc.exists():
        test_pass("Configuration documentation exists")
    else:
        test_fail("Configuration documentation missing")
    
    test_step("Check IMPLEMENTATION_SUMMARY.md")
    impl_doc = repo_root / 'IMPLEMENTATION_SUMMARY.md'
    if impl_doc.exists():
        test_pass("Implementation summary exists")
    else:
        test_fail("Implementation summary missing")
    
    print()
    
    # Test 5: File Structure
    print("Step 5: Checking File Structure")
    print("-" * 60)
    
    test_step("Check error_handler.py")
    if (repo_root / 'ipfs_accelerate_py' / 'error_handler.py').exists():
        test_pass("error_handler.py exists")
    else:
        test_fail("error_handler.py missing")
    
    test_step("Check test_error_handler.py")
    if (repo_root / 'test' / 'test_error_handler.py').exists():
        test_pass("test_error_handler.py exists")
    else:
        test_fail("test_error_handler.py missing")
    
    print()
    
    # Test 6: Optional Integrations
    print("Step 6: Optional Integrations")
    print("-" * 60)
    
    test_step("Check GitHub CLI availability")
    import subprocess
    try:
        result = subprocess.run(['gh', '--version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            test_pass("GitHub CLI (gh) is installed")
            
            # Check auth
            try:
                auth_result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, timeout=5)
                if auth_result.returncode == 0:
                    test_pass("GitHub CLI is authenticated")
                    print("  → Auto-issue creation would work")
                else:
                    print("  ⚠️  GitHub CLI not authenticated (optional)")
                    print("     Run: gh auth login")
            except:
                print("  ⚠️  Could not check GitHub CLI auth status")
        else:
            print("  ⚠️  GitHub CLI not installed (optional for auto-issue)")
    except:
        print("  ⚠️  GitHub CLI not installed (optional for auto-issue)")
    
    test_step("Check Copilot SDK availability")
    try:
        import copilot
        test_pass("GitHub Copilot SDK is installed")
    except ImportError:
        print("  ⚠️  Copilot SDK not installed (optional for auto-heal)")
        print("     Install: pip install github-copilot-sdk")
    
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests Passed: {GREEN}{tests_passed}{NC}")
    print(f"Tests Failed: {RED}{tests_failed}{NC}")
    print()
    
    if tests_failed == 0:
        print(f"{GREEN}✓ All tests passed!{NC}")
        print()
        print("Next steps:")
        print("1. Enable auto-features: export IPFS_AUTO_ISSUE=true")
        print("2. Authenticate GitHub CLI: gh auth login")
        print("3. Test with real CLI: ipfs-accelerate --help")
        print()
        return 0
    else:
        print(f"{RED}✗ Some tests failed{NC}")
        print()
        print("Please review the failures above and fix them.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
