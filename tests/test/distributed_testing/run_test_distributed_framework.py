#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Runner

This script runs all the tests for the distributed testing framework and provides
a summary of test coverage. It runs both unit tests and integration tests.

NOTE: Security and authentication features have been marked as OUT OF SCOPE.
      See SECURITY_DEPRECATED.md for details.

Usage:
    python run_test_distributed_framework.py [--unit] [--integration] [--coverage]
"""

import argparse
import os
import sys
import unittest
from typing import List, Optional

try:
    import pytest
    import coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False

# Configure paths
TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def discover_tests(test_type: Optional[str] = None) -> unittest.TestSuite:
    """
    Discover tests in the tests directory.
    
    Args:
        test_type: Type of tests to discover (unit or integration)
        
    Returns:
        Test suite with all discovered tests
    """
    pattern = "test_*.py"
    
    if test_type == "unit":
        # Only run unit tests (exclude integration tests)
        pattern = "test_[^i]*.py"  # Exclude files starting with "test_i"
    elif test_type == "integration":
        # Only run integration tests
        pattern = "test_i*.py"  # Only files starting with "test_i"
    
    # Create initial test suite
    test_suite = unittest.defaultTestLoader.discover(TESTS_DIR, pattern=pattern)
    
    # Filter out security-related tests (security features are out of scope)
    filtered_suite = unittest.TestSuite()
    for suite in test_suite:
        for test in suite:
            # Skip test_security.py
            if "test_security" not in str(test):
                filtered_suite.addTest(test)
    
    print("NOTE: Security-related tests have been skipped (security features are OUT OF SCOPE)")
    return filtered_suite


def run_unittest_tests(test_type: Optional[str] = None) -> int:
    """
    Run unittest tests.
    
    Args:
        test_type: Type of tests to run (unit or integration)
        
    Returns:
        Number of failures
    """
    # Discover tests
    test_suite = discover_tests(test_type)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Return number of failures
    return len(test_result.failures) + len(test_result.errors)


def run_pytest_tests(test_type: Optional[str] = None) -> int:
    """
    Run pytest tests.
    
    Args:
        test_type: Type of tests to run (unit or integration)
        
    Returns:
        Number of failures
    """
    # Determine test path based on test type
    if test_type == "unit":
        # Only run unit tests (exclude integration tests)
        test_path = os.path.join(TESTS_DIR, "test_[!i]*.py")
    elif test_type == "integration":
        # Only run integration tests
        test_path = os.path.join(TESTS_DIR, "test_i*.py")
    else:
        # Run all tests
        test_path = os.path.join(TESTS_DIR, "test_*.py")
    
    # Exclude security-related tests (security features are out of scope)
    ignore_patterns = ["test_security.py"]
    
    # Build pytest args
    pytest_args = ["-xvs", test_path]
    for pattern in ignore_patterns:
        pytest_args.extend(["--ignore", os.path.join(TESTS_DIR, pattern)])
    
    print("NOTE: Security-related tests have been skipped (security features are OUT OF SCOPE)")
    
    # Run pytest
    result = pytest.main(pytest_args)
    
    # Return number of failures
    return 1 if result != 0 else 0


def run_coverage(test_type: Optional[str] = None) -> None:
    """
    Run tests with coverage.
    
    Args:
        test_type: Type of tests to run (unit or integration)
    """
    if not HAS_COVERAGE:
        print("Coverage package not installed. Install with: pip install coverage")
        return
    
    # Create coverage object
    cov = coverage.Coverage(
        source=["."],
        omit=["tests/*", "run_test*.py", "*/__pycache__/*", "*/.*"]
    )
    
    # Start coverage
    cov.start()
    
    # Run tests
    run_unittest_tests(test_type)
    run_pytest_tests(test_type)
    
    # Stop coverage
    cov.stop()
    
    # Save coverage data
    cov.save()
    
    # Generate reports
    print("\nCoverage Summary:")
    cov.report()
    
    # Generate HTML report
    html_dir = "htmlcov"
    cov.html_report(directory=html_dir)
    print(f"\nHTML coverage report generated in {html_dir}/")


def run_tests(test_types: List[str], with_coverage: bool) -> int:
    """
    Run all specified tests.
    
    Args:
        test_types: List of test types to run (unit, integration, or both)
        with_coverage: Whether to run with coverage
        
    Returns:
        Number of failures
    """
    if with_coverage:
        # Run all tests with coverage
        run_coverage(None if "all" in test_types else test_types[0])
        return 0
    
    failures = 0
    
    if "all" in test_types or "unit" in test_types:
        print("\n=== Running Unit Tests (unittest) ===\n")
        failures += run_unittest_tests("unit")
        
        print("\n=== Running Unit Tests (pytest) ===\n")
        failures += run_pytest_tests("unit")
    
    if "all" in test_types or "integration" in test_types:
        print("\n=== Running Integration Tests (unittest) ===\n")
        failures += run_unittest_tests("integration")
        
        print("\n=== Running Integration Tests (pytest) ===\n")
        failures += run_pytest_tests("integration")
    
    return failures


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run distributed testing framework tests.")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    
    args = parser.parse_args()
    
    # Determine test types to run
    test_types = []
    if args.unit:
        test_types.append("unit")
    if args.integration:
        test_types.append("integration")
    if not test_types:
        test_types.append("all")
    
    # Run tests
    failures = run_tests(test_types, args.coverage)
    
    # Exit with failure if any tests failed
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()