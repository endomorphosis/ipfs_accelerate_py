#!/usr/bin/env python3
"""
Runner script for Worker Reconnection System tests.

This script runs the comprehensive test suite for the Worker Reconnection System,
verifying proper connection recovery, state synchronization, and task resumption.
"""

import os
import sys
import time
import argparse
import unittest
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directories to path
current_dir = Path(__file__).parent
parent_dir = str(current_dir.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_worker_reconnection_tests")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Worker Reconnection System tests")
    
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run (unit, integration, or all)"
    )
    
    parser.add_argument(
        "--test-name", 
        type=str,
        help="Run a specific test by name (e.g., 'TestConnectionStats.test_average_latency')"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    
    return parser.parse_args()


def run_tests(test_type: str = "all", test_name: Optional[str] = None, 
              verbose: bool = False, failfast: bool = False):
    """
    Run Worker Reconnection System tests.
    
    Args:
        test_type: Type of tests to run (unit, integration, or all)
        test_name: Run a specific test by name
        verbose: Enable verbose output
        failfast: Stop on first failure
    """
    # Import test module
    try:
        from tests.test_worker_reconnection import (
            TestConnectionStats, 
            TestWorkerReconnectionManager,
            TestWorkerReconnectionIntegration,
            TestWorkerReconnectionPlugin
        )
    except ImportError as e:
        logger.error(f"Failed to import test module: {e}")
        logger.error("Make sure you're running this script from the correct directory")
        sys.exit(1)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests based on test_type
    if test_name:
        # Run a specific test
        if "." in test_name:
            class_name, method_name = test_name.split(".")
            test_class = globals().get(class_name)
            if test_class:
                suite.addTest(test_class(method_name))
            else:
                logger.error(f"Test class '{class_name}' not found")
                sys.exit(1)
        else:
            logger.error(f"Invalid test name format: {test_name}. Use 'TestClass.test_method'")
            sys.exit(1)
    else:
        # Add test classes based on test_type
        if test_type in ["unit", "all"]:
            suite.addTest(unittest.makeSuite(TestConnectionStats))
            suite.addTest(unittest.makeSuite(TestWorkerReconnectionManager))
            suite.addTest(unittest.makeSuite(TestWorkerReconnectionPlugin))
        
        if test_type in ["integration", "all"]:
            suite.addTest(unittest.makeSuite(TestWorkerReconnectionIntegration))
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    
    # Print summary
    print("\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_tests(
        test_type=args.test_type,
        test_name=args.test_name,
        verbose=args.verbose,
        failfast=args.failfast
    )
    sys.exit(exit_code)