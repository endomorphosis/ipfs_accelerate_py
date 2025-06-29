#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test runner for Dynamic Resource Management (DRM) system tests.
"""

import unittest
import sys
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests(test_pattern=None, verbose=False):
    """
    Run DRM system tests matching the specified pattern.
    
    Args:
        test_pattern (str, optional): Pattern to match test names. Defaults to None.
        verbose (bool, optional): Enable verbose output. Defaults to False.
    
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    # Find the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add parent directory to path to import modules correctly
    sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))
    
    # Build test suite
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific tests matching pattern
        suite = loader.discover(script_dir, pattern=f"test_{test_pattern}*.py")
    else:
        # Run all DRM tests
        suite = unittest.TestSuite()
        suite.addTest(loader.discover(script_dir, pattern="test_dynamic_resource_manager.py"))
        suite.addTest(loader.discover(script_dir, pattern="test_resource_performance_predictor.py"))
        suite.addTest(loader.discover(script_dir, pattern="test_cloud_provider_manager.py"))
        suite.addTest(loader.discover(script_dir, pattern="test_drm_integration.py"))
        suite.addTest(loader.discover(script_dir, pattern="test_resource_optimization.py"))
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Report results
    logger.info(f"Ran {result.testsRun} tests")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run Dynamic Resource Management (DRM) system tests")
    parser.add_argument("--pattern", type=str, help="Pattern to match test names")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    success = run_tests(test_pattern=args.pattern, verbose=args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()