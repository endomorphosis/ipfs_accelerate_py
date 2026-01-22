#!/usr/bin/env python3
"""
Runner script for Worker Reconnection System integration tests.

This script runs the integration tests for the Worker Reconnection System,
verifying proper communication between worker and coordinator over WebSockets.
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
logger = logging.getLogger("run_worker_reconnection_integration_tests")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Worker Reconnection System integration tests")
    
    parser.add_argument(
        "--test-name", 
        type=str,
        help="Run a specific test by name (e.g., 'TestWorkerReconnectionWithRealCoordinator.test_basic_connection')"
    )
    
    parser.add_argument(
        "--no-websocket", 
        action="store_true",
        help="Skip tests requiring WebSocket communication"
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
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Test timeout in seconds"
    )
    
    return parser.parse_args()


def run_tests(test_name: Optional[str] = None, no_websocket: bool = False,
              verbose: bool = False, failfast: bool = False, log_level: str = "INFO",
              timeout: int = 30):
    """
    Run Worker Reconnection System integration tests.
    
    Args:
        test_name: Run a specific test by name
        no_websocket: Skip tests requiring WebSocket communication
        verbose: Enable verbose output
        failfast: Stop on first failure
        log_level: Logging level
        timeout: Test timeout in seconds
    """
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Import test module
    try:
        from tests.test_worker_reconnection_integration import (
            TestWorkerReconnectionWithRealCoordinator
        )
    except ImportError as e:
        logger.error(f"Failed to import test module: {e}")
        logger.error("Make sure you're running this script from the correct directory")
        sys.exit(1)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests based on parameters
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
        # Add all integration tests
        if not no_websocket:
            suite.addTest(unittest.makeSuite(TestWorkerReconnectionWithRealCoordinator))
        else:
            logger.warning("Skipping WebSocket tests due to --no-websocket flag")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    
    # Set timeout for tests if specified
    if timeout > 0:
        import signal
        
        def timeout_handler(signum, frame):
            logger.error(f"Tests timed out after {timeout} seconds")
            sys.exit(1)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    result = runner.run(suite)
    
    # Cancel timeout
    if timeout > 0:
        signal.alarm(0)
    
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
        test_name=args.test_name,
        no_websocket=args.no_websocket,
        verbose=args.verbose,
        failfast=args.failfast,
        log_level=args.log_level,
        timeout=args.timeout
    )
    sys.exit(exit_code)