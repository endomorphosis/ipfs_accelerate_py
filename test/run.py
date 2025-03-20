#!/usr/bin/env python
"""
IPFS Accelerate Test Framework Runner

A unified entry point for running tests in the IPFS Accelerate framework.
This script provides a flexible way to run tests based on test type,
hardware platform, model type, and other parameters.

Usage:
    python run.py [options]

Examples:
    # Run all tests
    python run.py

    # Run all model tests
    python run.py --test-type model

    # Run specific model tests
    python run.py --test-type model --model bert

    # Run hardware-specific tests
    python run.py --test-type hardware --platform webgpu

    # Run tests with specific hardware platform
    python run.py --platform cuda

    # Run API tests
    python run.py --test-type api --api openai

    # Run tests with custom markers
    python run.py --markers "slow or webgpu"

    # Run distributed tests
    python run.py --distributed --worker-count 4
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate Test Runner")
    
    # Test selection options
    parser.add_argument('--test-type', choices=['model', 'hardware', 'api', 'integration', 'all'],
                        default='all', help='Type of tests to run')
    parser.add_argument('--model', type=str, help='Specific model to test (e.g., bert, t5, vit)')
    parser.add_argument('--platform', type=str, 
                        help='Hardware platform (e.g., webgpu, webnn, cuda, rocm, cpu)')
    parser.add_argument('--api', type=str, help='API to test (e.g., openai, hf, vllm)')
    
    # Test execution options
    parser.add_argument('--markers', type=str, help='pytest markers expression (e.g., "slow or webgpu")')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('--collect-only', action='store_true', help='Only collect tests, do not run')
    parser.add_argument('--report', action='store_true', help='Generate HTML test report')
    
    # Distributed testing options
    parser.add_argument('--distributed', action='store_true', help='Run tests in distributed mode')
    parser.add_argument('--worker-count', type=int, default=4, 
                        help='Number of workers for distributed testing')
    parser.add_argument('--coordinator', type=str, help='Coordinator address for distributed testing')
    
    # Paths and files
    parser.add_argument('--output-dir', type=str, help='Output directory for test artifacts')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('tests', nargs='*', help='Specific test files or directories to run')
    
    # CI integration
    parser.add_argument('--ci', action='store_true', help='Running in CI environment')
    parser.add_argument('--junit-xml', action='store_true', help='Generate JUnit XML report')
    
    return parser.parse_args()


def build_pytest_command(args: argparse.Namespace) -> List[str]:
    """Build the pytest command based on parsed arguments."""
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-" + "v" * args.verbose])
        
    # Add fail fast
    if args.fail_fast:
        cmd.append("-x")
        
    # Add collect only
    if args.collect_only:
        cmd.append("--collect-only")
        
    # Add markers
    if args.markers:
        cmd.extend(["-m", args.markers])
        
    # Test selection by type
    if args.test_type != 'all':
        if args.test_type == 'model':
            test_path = "models/"
            if args.model:
                # Try to find the model directory
                # Check in text, vision, audio, multimodal directories
                model_dirs = []
                for category in ["text", "vision", "audio", "multimodal"]:
                    potential_dir = os.path.join("models", category)
                    if os.path.exists(potential_dir):
                        for d in os.listdir(potential_dir):
                            if d.lower() == args.model.lower() or args.model.lower() in d.lower():
                                model_dirs.append(os.path.join(potential_dir, d))
                
                if model_dirs:
                    test_path = " ".join(model_dirs)
                else:
                    # Fall back to a generic path that matches the model name
                    test_path = f"models/*/{args.model}*"
        elif args.test_type == 'hardware':
            test_path = "hardware/"
            if args.platform:
                test_path = f"hardware/{args.platform}"
        elif args.test_type == 'api':
            test_path = "api/"
            if args.api:
                test_path = f"api/*{args.api}*"
        elif args.test_type == 'integration':
            test_path = "integration/"
    else:
        # If specific tests are provided, use those
        if args.tests:
            test_path = " ".join(args.tests)
        else:
            # Otherwise run everything except specific directories we want to exclude
            test_path = "."
    
    # Add platform filter if specified
    if args.platform and args.test_type != 'hardware':
        cmd.extend(["-k", args.platform])
    
    # Add report generation
    if args.report:
        cmd.extend(["--html=report.html", "--self-contained-html"])
    
    # Add JUnit XML for CI
    if args.junit_xml or args.ci:
        cmd.extend(["--junitxml=test-results.xml"])
    
    # Add distributed testing parameters
    if args.distributed:
        cmd.extend([
            f"--distributed",
            f"--worker-count={args.worker_count}"
        ])
        if args.coordinator:
            cmd.extend([f"--coordinator={args.coordinator}"])
    
    # Add the test path
    cmd.append(test_path)
    
    return cmd


def run_tests(cmd: List[str]) -> int:
    """Run the pytest command and return the exit code."""
    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False)
        exit_code = result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        exit_code = 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Test run completed in {duration:.2f} seconds with exit code {exit_code}")
    return exit_code


def main() -> int:
    """Main entry point for the test runner."""
    args = parse_args()
    
    # Change to the test directory to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Build and run the pytest command
    cmd = build_pytest_command(args)
    
    return run_tests(cmd)


if __name__ == "__main__":
    sys.exit(main())