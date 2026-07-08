#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run tests for the Resource Optimization component of the Dynamic Resource Management system.

This script runs the tests for the ResourceOptimizer component and provides a summary
of the test results.

Usage:
    python run_resource_optimization_tests.py [--verbose] [--quick]
"""

import os
import sys
import logging
import argparse
import unittest
import tempfile
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("resource_optimization_tests")

# Add parent directory to path to import modules correctly
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_tests(verbose=False, quick=False):
    """
    Run tests for the ResourceOptimizer component.
    
    Args:
        verbose (bool): Whether to show verbose output
        quick (bool): Whether to run a smaller subset of tests for faster execution
        
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Load test modules
    from test_resource_optimization import (
        TestResourceOptimizer,
        TestResourceOptimizerIntegration,
        TestResourceOptimizerPerformance
    )
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestResourceOptimizer))
    suite.addTest(unittest.makeSuite(TestResourceOptimizerIntegration))
    
    # Add performance tests only if not in quick mode
    if not quick:
        suite.addTest(unittest.makeSuite(TestResourceOptimizerPerformance))
    
    # Create a test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run the tests
    logger.info("Running ResourceOptimizer tests...")
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Calculate test duration
    duration = end_time - start_time
    
    # Generate test summary
    logger.info(f"Test run completed in {duration:.2f} seconds")
    logger.info(f"Ran {result.testsRun} tests")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main entry point for the test runner."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run ResourceOptimizer tests")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--quick", action="store_true", help="Run a smaller subset of tests")
    args = parser.parse_args()
    
    # Run tests
    success = run_tests(verbose=args.verbose, quick=args.quick)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()