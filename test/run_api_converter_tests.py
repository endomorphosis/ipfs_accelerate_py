#!/usr/bin/env python3
"""
Test Runner for API Backend Converter Tests

This script discovers and runs all tests for the API Backend Converter.
"""

import os
import sys
import unittest
import argparse


def run_tests(verbosity=1, test_pattern=None):
    """Discover and run tests with optional filter pattern"""
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Set the test pattern if provided
    if test_pattern:
        loader.testNamePattern = test_pattern
        
    # Find the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover tests in the current directory matching the pattern
    pattern = "test_api_backend_converter*.py"
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run API Backend Converter tests")
    parser.add_argument("-v", "--verbose", action="store_true", 
                      help="Increase output verbosity")
    parser.add_argument("-p", "--pattern", 
                      help="Only run tests matching the pattern")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set verbosity level based on arguments
    verbosity = 2 if args.verbose else 1
    
    # Run the tests
    success = run_tests(verbosity=verbosity, test_pattern=args.pattern)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)