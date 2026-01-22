#!/usr/bin/env python3
"""
Test runner for CI provider standardization and features.

This script runs all the tests and demos for the CI provider integration,
ensuring that all providers are correctly standardized and all features are working.
"""

import asyncio
import logging
import argparse
import subprocess
import sys
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_test(test_name: str, args: List[str] = None):
    """
    Run a specific test or demo.
    
    Args:
        test_name: Name of the test or demo to run
        args: Additional arguments to pass to the test
    """
    if args is None:
        args = []
    
    cmd = [sys.executable, test_name] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout:
            logger.info(f"Output from {test_name}:")
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        
        logger.info(f"Successfully ran {test_name}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {test_name}")
        logger.error(f"Exit code: {e.returncode}")
        
        if e.stdout:
            logger.info(f"Output from {test_name}:")
            for line in e.stdout.splitlines():
                logger.info(f"  {line}")
        
        if e.stderr:
            logger.error(f"Error output from {test_name}:")
            for line in e.stderr.splitlines():
                logger.error(f"  {line}")
        
        return False

async def run_all_tests():
    """Run all CI provider tests and demos."""
    logger.info("Running all CI provider tests and demos...")
    
    # List of tests to run
    tests = [
        "ci/test_provider_standardization.py",
        "ci/test_artifact_handling.py"
    ]
    
    # Run each test
    results = {}
    for test in tests:
        results[test] = await run_test(test)
    
    # Show summary
    logger.info("==== Test Results ====")
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
        all_passed = all_passed and passed
    
    return all_passed

async def run_demos(args):
    """Run the CI provider demos."""
    logger.info("Running CI provider demos...")
    
    # List of demos to run
    demos = [
        ("run_test_artifact_handling.py", ["--provider", args.provider])
    ]
    
    # Add provider-specific arguments
    provider_args = []
    
    if args.token:
        provider_args.extend(["--token", args.token])
    
    if args.repository:
        provider_args.extend(["--repository", args.repository])
    
    if args.project:
        provider_args.extend(["--project", args.project])
    
    if args.organization:
        provider_args.extend(["--organization", args.organization])
    
    # Run each demo
    results = {}
    for demo, demo_args in demos:
        results[demo] = await run_test(demo, demo_args + provider_args)
    
    # Show summary
    logger.info("==== Demo Results ====")
    all_passed = True
    for demo, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{demo}: {status}")
        all_passed = all_passed and passed
    
    return all_passed

async def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="CI Provider Tests")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, skip demos")
    parser.add_argument("--demo-only", action="store_true", help="Run demos only, skip tests")
    parser.add_argument("--provider", choices=["github", "gitlab", "jenkins", "azure"], default="github",
                    help="CI provider to use for demos")
    parser.add_argument("--token", help="API token for the CI provider")
    parser.add_argument("--repository", help="Repository name (for GitHub, GitLab)")
    parser.add_argument("--project", help="Project name (for Azure)")
    parser.add_argument("--organization", help="Organization name (for Azure)")
    args = parser.parse_args()
    
    results = []
    
    # Run tests if not skipped
    if not args.demo_only:
        test_result = await run_all_tests()
        results.append(("Tests", test_result))
    
    # Run demos if not skipped
    if not args.test_only:
        demo_result = await run_demos(args)
        results.append(("Demos", demo_result))
    
    # Show overall summary
    logger.info("==== Overall Results ====")
    all_passed = all(passed for _, passed in results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("All tests and demos PASSED!")
        return 0
    else:
        logger.error("Some tests or demos FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)