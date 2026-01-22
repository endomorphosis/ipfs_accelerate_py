#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for the refactored generator suite.
Runs all test files and reports results.
"""

import os
import sys
import unittest
import argparse
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(test_pattern=None, verbose=False):
    """Run tests matching the given pattern."""
    # Set verbosity level
    verbosity = 2 if verbose else 1
    
    # Get all test files
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    # Filter test files if a pattern is provided
    if test_pattern:
        test_files = [f for f in test_files if test_pattern in f.name]
    
    # Calculate relative paths
    base_dir = Path(__file__).parent.parent
    rel_paths = [str(f.relative_to(base_dir)) for f in test_files]
    
    print(f"Running {len(test_files)} test files...")
    
    # Create a test suite from the filtered test files
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for file_path in test_files:
        # Convert file path to module name
        rel_path = file_path.relative_to(base_dir)
        module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        
        try:
            # Load tests from the module
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
        except Exception as e:
            print(f"Error loading tests from {module_name}: {e}")
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for the refactored generator suite.")
    parser.add_argument("--pattern", type=str, help="Pattern to filter test files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity.")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests.")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests.")
    
    args = parser.parse_args()
    
    # Determine test pattern
    pattern = args.pattern
    if args.integration and not pattern:
        pattern = "integration"
    elif args.unit and not pattern:
        pattern = ""  # Run all tests except integration
    
    # Run tests
    result = run_tests(pattern, args.verbose)
    
    # Calculate statistics
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = tests_run - failures - errors
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {tests_run} tests run.")
    print(f"  Passed:   {passed}")
    print(f"  Failures: {failures}")
    print(f"  Errors:   {errors}")
    print("=" * 70)
    
    # Return appropriate exit code
    return 0 if failures == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())