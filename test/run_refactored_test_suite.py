#!/usr/bin/env python3
"""
Enhanced Test Runner for Refactored Test Suite

This script runs tests in the refactored test suite structure and generates
a report of the results. It properly handles import paths and test discovery.
"""

import os
import sys
import argparse
import unittest
import time
import logging
import importlib
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

# Ensure the refactored test suite directory is in the Python path
if REFACTORED_DIR not in sys.path:
    sys.path.insert(0, REFACTORED_DIR)

# Also add the parent directory to sys.path for absolute imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

def initialize_module_structure():
    """Initialize the module structure for import resolution."""
    # Create __init__.py files in any directories that don't have them
    for root, dirs, files in os.walk(REFACTORED_DIR):
        if os.path.relpath(root, REFACTORED_DIR) != '.':  # Skip the root directory
            init_file = os.path.join(root, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated __init__.py file for module structure\n')
                logger.info(f"Created {init_file}")

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
        total_skipped = sum(len(r.skipped) for r in results.values() if hasattr(r, 'skipped'))
        
        f.write("## Summary\n\n")
        f.write(f"- Total tests run: {total_tests}\n")
        f.write(f"- Failures: {total_failures}\n")
        f.write(f"- Errors: {total_errors}\n")
        f.write(f"- Skipped: {total_skipped}\n")
        
        if total_tests > 0:
            success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100)
            f.write(f"- Success rate: {success_rate:.2f}%\n\n")
        else:
            f.write("- Success rate: N/A\n\n")
        
        # Details by directory
        f.write("## Results by Directory\n\n")
        for directory, result in results.items():
            f.write(f"### {directory}\n\n")
            f.write(f"- Tests run: {result.testsRun}\n")
            f.write(f"- Failures: {len(result.failures)}\n")
            f.write(f"- Errors: {len(result.errors)}\n")
            
            if hasattr(result, 'skipped'):
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
            
            if hasattr(result, 'skipped') and result.skipped:
                f.write("\n#### Skipped\n\n")
                for test, reason in result.skipped:
                    f.write(f"- {test}: {reason}\n")

def run_directory_tests(directory: str, pattern: str = 'test_*.py') -> unittest.TestResult:
    """Run tests from a specific directory using proper import paths."""
    # Calculate the module name based on the directory
    module_dir = os.path.relpath(directory, REFACTORED_DIR).replace(os.sep, '.')
    if module_dir == '.':
        module_dir = 'refactored_test_suite'
    else:
        module_dir = 'refactored_test_suite.' + module_dir
    
    logger.info(f"Testing module: {module_dir}")
    
    # Create a test suite for this directory
    suite = unittest.TestSuite()
    
    # Find test files in the directory
    test_files = [f for f in os.listdir(directory) if f.startswith('test_') and f.endswith('.py')]
    
    for test_file in test_files:
        module_name = module_dir + '.' + test_file[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(module_name)
            tests = unittest.defaultTestLoader.loadTestsFromModule(module)
            suite.addTest(tests)
            logger.info(f"Added tests from {module_name}")
        except Exception as e:
            logger.error(f"Error loading tests from {module_name}: {str(e)}")
    
    if suite.countTestCases() > 0:
        logger.info(f"Running {suite.countTestCases()} tests from {module_dir}")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return result
    else:
        logger.info(f"No tests found in {module_dir}")
        return unittest.TestResult()

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
    parser.add_argument('--init', action='store_true',
                      help='Create missing __init__.py files for module structure')
    
    args = parser.parse_args()
    
    # Initialize module structure if requested
    if args.init:
        initialize_module_structure()
    
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
        relative_dir = os.path.relpath(test_dir, args.directory)
        logger.info(f"Running tests in {relative_dir}")
        
        if relative_dir == '.':
            # For the root directory, use the default discoverer
            suite = discover_tests(test_dir, args.pattern)
            if suite.countTestCases() > 0:
                results[relative_dir] = run_tests(suite)
            else:
                logger.info(f"No tests found in root directory")
        else:
            # For subdirectories, use the import-based approach
            results[relative_dir] = run_directory_tests(test_dir, args.pattern)
    
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