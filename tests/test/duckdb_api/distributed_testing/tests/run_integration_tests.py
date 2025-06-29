#!/usr/bin/env python3
"""
Run integration tests for the Distributed Testing Framework.

This script runs all the integration tests in the distributed testing framework,
including:
1. Basic integration tests (test_integration.py)
2. Task scheduler tests (test_scheduler.py)
3. Health monitor tests (test_health_monitor.py)
4. Load balancer tests (test_load_balancer.py)
5. Load balancer monitoring tests (test_load_balancer_monitoring.py)
6. Benchmark tests (test_benchmark.py)

Usage:
    python run_integration_tests.py [--verbose]
    python run_integration_tests.py --test load_balancer_monitoring
    python run_integration_tests.py --case LoadBalancerMonitoringIntegrationTest.test_01_monitoring_initialization
    python run_integration_tests.py --list
"""

import os
import sys
import time
import argparse
import unittest
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def discover_tests(test_pattern: Optional[str] = None) -> unittest.TestSuite:
    """
    Discover all integration test modules.
    
    Args:
        test_pattern: Optional pattern to filter test modules
        
    Returns:
        TestSuite containing all discovered tests
    """
    # Get the directory containing test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific tests matching the pattern
        pattern = f"test_{test_pattern}.py"
        return loader.discover(test_dir, pattern=pattern)
    else:
        # Run all test files
        return loader.discover(test_dir, pattern="test_*.py")


def run_integration_tests(verbose=False, test_pattern=None) -> bool:
    """
    Run integration tests for the distributed testing framework.
    
    Args:
        verbose: Whether to output verbose test information
        test_pattern: Optional pattern to filter test modules
    
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
    test_suite = discover_tests(test_pattern)
    
    # Run tests with verbosity based on arguments
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


def run_specific_test_case(test_path: str, verbose: bool = False) -> bool:
    """
    Run a single test case by its path.
    
    Args:
        test_path: Path to the test case (e.g., 'LoadBalancerTest.test_worker_scoring')
        verbose: Whether to output verbose test information
        
    Returns:
        True if the test passes, False otherwise
    """
    # Parse the test path
    parts = test_path.split('.')
    if len(parts) < 2:
        print(f"Error: Invalid test path '{test_path}'. Format should be 'TestClass.test_method'")
        return False
    
    # Get the class name and test name
    test_class_name = parts[0]
    test_method_name = parts[1]
    
    # Find the test module that contains the class
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    module_name = None
    for test_file in test_files:
        # Read the file and check if it contains the class definition
        file_path = os.path.join(test_dir, test_file)
        with open(file_path, 'r') as f:
            content = f.read()
            if f"class {test_class_name}(" in content:
                module_name = test_file[:-3]  # Remove .py
                break
    
    if not module_name:
        print(f"Error: Could not find test class '{test_class_name}' in any test module")
        return False
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    
    # Create the test suite with just the specified method
    loader = unittest.TestLoader()
    
    # Need to import the module to load the test class
    module = __import__(f"duckdb_api.distributed_testing.tests.{module_name}", 
                        fromlist=[test_class_name])
    test_class = getattr(module, test_class_name)
    
    # Create a suite with just the specified test
    suite = unittest.TestSuite()
    suite.addTest(test_class(test_method_name))
    
    # Run the test
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def list_tests() -> None:
    """List all available tests without running them."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all test modules
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    print("Available test modules:")
    
    for test_file in sorted(test_files):
        module_name = test_file[:-3]  # Remove .py extension
        print(f"  - {module_name}")
        
        # Read the file to extract test class names and methods
        file_path = os.path.join(test_dir, test_file)
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract class definitions
            class_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith('class ') and '(unittest.TestCase)' in line]
            
            for class_line in class_lines:
                # Extract class name
                class_name = class_line.split('class ')[1].split('(')[0].strip()
                print(f"    - {class_name}")
                
                # Extract test methods
                test_methods = [line.strip() for line in content.split('\n') 
                               if line.strip().startswith('def test_') and class_name in content.split(line)[0].split('class ')[-1]]
                
                for method_line in test_methods:
                    # Extract method name
                    method_name = method_line.split('def ')[1].split('(')[0].strip()
                    print(f"      - {method_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for the distributed testing framework"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--test", type=str,
                       help="Run a specific test file (e.g., 'load_balancer' for test_load_balancer.py)")
    parser.add_argument("--case", type=str,
                       help="Run a specific test case (e.g., 'LoadBalancerTest.test_worker_scoring')")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available tests without running them")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("Distributed Testing Framework - Integration Tests")
    print("=" * 80)
    
    if args.list:
        # List available tests
        list_tests()
        return 0
    
    if args.case:
        # Run a specific test case
        print(f"\nRunning specific test case: {args.case}")
        success = run_specific_test_case(args.case, args.verbose)
    elif args.test:
        # Run tests from a specific module
        print(f"\nRunning tests from module: test_{args.test}.py")
        success = run_integration_tests(args.verbose, args.test)
    else:
        # Run all tests
        print("\nRunning all integration tests...")
        success = run_integration_tests(args.verbose)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    if success:
        print(f"✅ ALL TESTS PASSED in {elapsed_time:.2f} seconds!")
    else:
        print(f"❌ TESTS FAILED after {elapsed_time:.2f} seconds!")
    print("=" * 80 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())