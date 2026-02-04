#!/usr/bin/env python3
"""
Test Runner for Refactored Test Suite

This script runs tests in the refactored test suite structure and generates
a report of the results.
"""

import os
import sys
import argparse
import unittest
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('refactored_test_runner')

# Path to refactored test suite
REFACTORED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refactored_test_suite')


def discover_tests(directory: str, pattern: str = 'test_*.py') -> unittest.TestSuite:
    """Discover tests in the specified directory."""
    logger.info(f"Discovering tests in {directory} with pattern {pattern}")
    return unittest.defaultTestLoader.discover(directory, pattern=pattern)


def run_tests(test_suite: unittest.TestSuite) -> unittest.TestResult:
    """Run the test suite and return the result."""
    logger.info(f"Running {test_suite.countTestCases()} tests")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(test_suite)


def find_test_directories(base_dir: str) -> List[str]:
    """Find all directories containing tests."""
    test_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains test files
        if any(f.startswith('test_') and f.endswith('.py') for f in files):
            test_dirs.append(root)
    return test_dirs


def generate_report(results: Dict[str, unittest.TestResult], output_path: str) -> None:
    """Generate a report of the test results."""
    with open(output_path, 'w') as f:
        f.write("# Refactored Test Suite Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        total_tests = sum(r.testsRun for r in results.values())
        total_failures = sum(len(r.failures) for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        total_skipped = sum(len(r.skipped) for r in results.values())
        
        f.write("## Summary\n\n")
        f.write(f"- Total tests run: {total_tests}\n")
        f.write(f"- Failures: {total_failures}\n")
        f.write(f"- Errors: {total_errors}\n")
        f.write(f"- Skipped: {total_skipped}\n")
        f.write(f"- Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0:.2f}%\n\n")
        
        # Details by directory
        f.write("## Results by Directory\n\n")
        for directory, result in results.items():
            f.write(f"### {directory}\n\n")
            f.write(f"- Tests run: {result.testsRun}\n")
            f.write(f"- Failures: {len(result.failures)}\n")
            f.write(f"- Errors: {len(result.errors)}\n")
            f.write(f"- Skipped: {len(result.skipped)}\n")
            
            if result.failures:
                f.write("\n#### Failures\n\n")
                for test, traceback in result.failures:
                    f.write(f"- {test}\n")
                    f.write("```python\n")
                    f.write(traceback)
                    f.write("\n```\n\n")
            
            if result.errors:
                f.write("\n#### Errors\n\n")
                for test, traceback in result.errors:
                    f.write(f"- {test}\n")
                    f.write("```python\n")
                    f.write(traceback)
                    f.write("\n```\n\n")


def main():
    parser = argparse.ArgumentParser(description='Run tests in the refactored test suite')
    parser.add_argument('--directory', type=str, default=REFACTORED_DIR,
                      help='Base directory for tests (default: refactored_test_suite)')
    parser.add_argument('--pattern', type=str, default='test_*.py',
                      help='Pattern for test files (default: test_*.py)')
    parser.add_argument('--output', type=str, default='refactored_test_results.md',
                      help='Output file for test results (default: refactored_test_results.md)')
    parser.add_argument('--subdirs', type=str, nargs='+',
                      help='Specific subdirectories to test')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Find test directories
    if args.subdirs:
        test_dirs = [os.path.join(args.directory, subdir) for subdir in args.subdirs]
    else:
        test_dirs = find_test_directories(args.directory)
    
    logger.info(f"Found {len(test_dirs)} test directories")
    
    # Run tests in each directory
    results = {}
    for test_dir in test_dirs:
        try:
            relative_dir = os.path.relpath(test_dir, args.directory)
            logger.info(f"Running tests in {relative_dir}")
            
            test_suite = discover_tests(test_dir, args.pattern)
            if test_suite.countTestCases() > 0:
                results[relative_dir] = run_tests(test_suite)
            else:
                logger.info(f"No tests found in {relative_dir}")
        except Exception as e:
            logger.error(f"Error running tests in {test_dir}: {str(e)}")
    
    # Generate report
    generate_report(results, args.output)
    
    # Print summary
    elapsed_time = time.time() - start_time
    total_tests = sum(r.testsRun for r in results.values())
    total_failures = sum(len(r.failures) for r in results.values())
    total_errors = sum(len(r.errors) for r in results.values())
    
    print(f"\nTest run completed in {elapsed_time:.2f} seconds")
    print(f"Total tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()