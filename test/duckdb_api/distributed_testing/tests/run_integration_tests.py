#!/usr/bin/env python3
"""
Run integration tests for the distributed testing framework.

This script runs the integration tests for the distributed testing framework
to ensure all components work together seamlessly.

Usage:
    python run_integration_tests.py [--verbose]
"""

import os
import sys
import argparse
import unittest
import logging
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_integration_tests(verbose=False):
    """
    Run integration tests for the distributed testing framework.
    
    Args:
        verbose: Whether to output verbose test information
    
    Returns:
        True if all tests passed, False otherwise
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    
    # Discover and run tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    
    # Run tests with verbosity based on arguments
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for the distributed testing framework"
    )
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print(f"Running integration tests for the distributed testing framework...")
    success = run_integration_tests(args.verbose)
    
    if success:
        print("\n✅ All integration tests passed successfully!")
        return 0
    else:
        print("\n❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())