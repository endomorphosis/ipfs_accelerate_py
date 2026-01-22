#!/usr/bin/env python3
"""
Run script for CI/CD Integration tests and examples

This script demonstrates the CI/CD Integration capabilities of the
Distributed Testing Framework by running various CI integration examples
and tests.
"""

import argparse
import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path

# Ensure proper imports by adding parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available examples to run
EXAMPLES = {
    "github": "distributed_testing.examples.github_ci_integration_example",
    "gitlab": "distributed_testing.examples.gitlab_ci_integration_example",
    "generic": "distributed_testing.examples.generic_ci_integration_example",
    "batch": "distributed_testing.examples.ci_coordinator_batch_example",
    "worker_auto_discovery": "distributed_testing.examples.worker_auto_discovery_with_ci"
}

def run_tests(args):
    """Run CI/CD integration tests."""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Get the absolute path to the tests directory
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    
    # Discover all test files related to CI/CD integration
    if args.test_all:
        pattern = "test_ci_*.py"
        logger.info(f"Running all CI integration tests matching pattern: {pattern}")
        discovered_tests = test_loader.discover(tests_dir, pattern=pattern)
        test_suite.addTests(discovered_tests)
        
        pattern = "test_worker_auto_discovery_with_ci.py"
        logger.info(f"Running worker auto-discovery tests with CI integration: {pattern}")
        discovered_tests = test_loader.discover(tests_dir, pattern=pattern)
        test_suite.addTests(discovered_tests)
    else:
        # Run specific tests based on arguments
        if args.test_ci_providers:
            logger.info("Running CI provider interface tests")
            test_suite.addTests(test_loader.discover(tests_dir, pattern="test_ci_integration.py"))
        
        if args.test_worker_discovery:
            logger.info("Running worker auto-discovery with CI integration tests")
            test_suite.addTests(test_loader.discover(tests_dir, pattern="test_worker_auto_discovery_with_ci.py"))
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=args.verbosity)
    
    # Run the tests
    result = test_runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1

async def run_example(args):
    """Run the specified example."""
    example_name = args.example
    
    if example_name not in EXAMPLES:
        logger.error(f"Unknown example: {example_name}")
        return 1
    
    # Import the example dynamically
    module_path = EXAMPLES[example_name]
    
    try:
        module = __import__(module_path, fromlist=["run_example"])
        
        # Extract additional arguments if provided
        kwargs = {}
        
        if args.provider:
            kwargs["ci_provider_type"] = args.provider
        
        if args.config:
            # Load config from JSON file
            import json
            with open(args.config, "r") as f:
                config = json.load(f)
            
            # If a specific provider is selected, use its config
            if args.provider and args.provider in config:
                kwargs["ci_config"] = config[args.provider]
            else:
                kwargs["ci_config"] = config
        
        # Add any other specific arguments
        if example_name == "worker_auto_discovery" and args.workers:
            kwargs["num_workers"] = args.workers
        
        # Run the example
        logger.info(f"Running CI/CD Integration example: {example_name}")
        await module.run_example(**kwargs)
        logger.info(f"CI/CD Integration example {example_name} complete")
        
        return 0
    except ImportError as e:
        logger.error(f"Failed to import example {module_path}: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error running example {example_name}: {e}")
        return 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CI/CD integration tests and examples')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Example mode
    example_parser = subparsers.add_parser("example", help="Run a CI/CD integration example")
    example_parser.add_argument("example", choices=list(EXAMPLES.keys()), help="Example to run")
    example_parser.add_argument("--provider", help="CI provider type (github, gitlab, jenkins, etc.)")
    example_parser.add_argument("--config", help="Path to CI provider configuration file")
    example_parser.add_argument("--workers", type=int, help="Number of workers (for worker_auto_discovery example)")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Run CI/CD integration tests")
    test_parser.add_argument("--test-all", action="store_true", help="Run all CI/CD integration tests")
    test_parser.add_argument("--test-ci-providers", action="store_true", help="Run CI provider interface tests")
    test_parser.add_argument("--test-worker-discovery", action="store_true", help="Run worker auto-discovery with CI integration tests")
    test_parser.add_argument("-v", "--verbosity", type=int, default=2, help="Verbosity level (1-3)")
    
    args = parser.parse_args()
    
    # If no mode specified, default to example mode with basic example
    if not args.mode:
        args.mode = "example"
        args.example = "generic"
        args.provider = "local"
        args.config = None
        args.workers = None
    
    # In test mode, if no specific tests are selected, run all tests
    if args.mode == "test" and not (args.test_all or args.test_ci_providers or args.test_worker_discovery):
        args.test_all = True
    
    return args

async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == "example":
        return await run_example(args)
    elif args.mode == "test":
        return run_tests(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

# Run the script
if __name__ == "__main__":
    sys.exit(asyncio.run(main()))