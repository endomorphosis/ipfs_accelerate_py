#!/usr/bin/env python3
"""
Run unit tests for browser recovery strategies.

This script runs the unit tests for the browser recovery strategies implementation,
providing a convenient way to test the functionality.

Usage:
    python distributed_testing/run_test_browser_recovery_strategies.py
"""

import os
import sys
import unittest
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_browser_recovery_tests")

def get_script_dir():
    """Get the directory of this script."""
    return os.path.dirname(os.path.abspath(__file__))

def setup_environment():
    """Setup the environment for running tests."""
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(get_script_dir())
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

def run_tests():
    """Run the browser recovery strategy tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Browser Recovery Strategy Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--test", type=str, help="Run specific test (e.g. TestSimpleRetryStrategy)")
    args = parser.parse_args()
    
    # Import test module
    from tests.test_browser_recovery_strategies import (
        TestBrowserRecoveryStrategy, TestSimpleRetryStrategy, TestBrowserRestartStrategy,
        TestSettingsAdjustmentStrategy, TestBrowserFallbackStrategy, TestSimulationFallbackStrategy,
        TestModelSpecificRecoveryStrategy, TestProgressiveRecoveryManager, TestUtilityFunctions
    )
    
    # Create test suite
    if args.test:
        # Run specific test
        test_class = globals().get(args.test)
        if not test_class:
            logger.error(f"Test class '{args.test}' not found")
            sys.exit(1)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    else:
        # Run all tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBrowserRecoveryStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSimpleRetryStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBrowserRestartStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSettingsAdjustmentStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBrowserFallbackStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSimulationFallbackStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestModelSpecificRecoveryStrategy))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestProgressiveRecoveryManager))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1

def main():
    """Main entry point."""
    setup_environment()
    
    print("=" * 80)
    print("Browser Recovery Strategy Tests")
    print("=" * 80)
    print("Running unit tests for the browser recovery strategies implementation.")
    print("This will test the functionality of each recovery strategy and the")
    print("progressive recovery system as a whole.")
    print()
    print("For comprehensive documentation, see:")
    print("  - ADVANCED_BROWSER_RECOVERY_STRATEGIES.md")
    print("  - ADVANCED_FAULT_TOLERANCE_BROWSER_INTEGRATION.md")
    print("=" * 80)
    print()
    
    # Run the tests
    sys.exit(run_tests())

if __name__ == "__main__":
    main()