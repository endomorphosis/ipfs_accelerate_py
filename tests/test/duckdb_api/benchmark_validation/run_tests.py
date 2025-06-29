#!/usr/bin/env python3
"""
Test runner for the Benchmark Validation System.

This script runs all tests for the benchmark validation system.
"""

import os
import sys
import unittest
import argparse
import logging
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def run_tests(verbose=False, test_pattern=None):
    """
    Run benchmark validation tests.
    
    Args:
        verbose: Whether to output verbose test results
        test_pattern: Pattern to match test names (e.g., "test_outlier*")
    """
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Discover and run tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific tests matching pattern
        pattern = f"*{test_pattern}*" if not test_pattern.startswith("*") else test_pattern
        tests = loader.discover(test_dir, pattern=pattern)
    else:
        # Run all tests
        tests = loader.discover(test_dir)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(tests)
    
    # Return success status (0 for success, 1 for failure)
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run Benchmark Validation tests")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('-p', '--pattern', type=str, help="Test name pattern to match")
    args = parser.parse_args()
    
    sys.exit(run_tests(args.verbose, args.pattern))

if __name__ == "__main__":
    main()