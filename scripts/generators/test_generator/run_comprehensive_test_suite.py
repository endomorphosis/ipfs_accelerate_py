#!/usr/bin/env python3
"""
Comprehensive test suite runner.

This script ties together validation, generation of missing tests,
running integration tests, and benchmarking for a complete test workflow.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"comprehensive_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a command and log output.
    
    Args:
        cmd: Command to run
        description: Description of the command
        
    Returns:
        Tuple of (success, output)
    """
    logger.info(f"Running {description}: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully in {elapsed:.2f}s")
            return True, result.stdout
        else:
            logger.error(f"❌ {description} failed in {elapsed:.2f}s")
            logger.error(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ {description} failed with exception in {elapsed:.2f}s")
        logger.error(f"Exception: {e}")
        return False, str(e)

def run_validation(args):
    """Run validation on test files."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "run_validation.py"),
        "--test-dir", args.test_dir,
        "--report-dir", args.report_dir
    ]
    
    success, output = run_command(cmd, "Validation")
    return success

def run_test_generation(args):
    """Generate missing tests."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "run_test_generation.py"),
        "--priority", args.priority,
        "--output-dir", args.test_dir,
        "--force" if args.force else "",
        "--verify"
    ]
    
    # Remove empty args
    cmd = [arg for arg in cmd if arg]
    
    success, output = run_command(cmd, "Test generation")
    return success

def run_integration_tests(args):
    """Run integration tests."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "run_integration_tests.py"),
        "--test-dir", args.test_dir,
        "--output-dir", args.report_dir,
        "--architectures", "all",
        "--save-results" if args.save_results else "",
        "--mock" if args.mock else ""
    ]
    
    # Remove empty args
    cmd = [arg for arg in cmd if arg]
    
    success, output = run_command(cmd, "Integration tests")
    return success

def run_implementation_tracking(args):
    """Track implementation progress."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "track_implementation_progress.py"),
        "--dirs", args.test_dir,
        "--output", os.path.join(args.report_dir, "implementation_progress.md")
    ]
    
    success, output = run_command(cmd, "Implementation tracking")
    return success

def run_benchmark(args):
    """Run benchmarking."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "benchmarking/run_hardware_benchmark.py"),
        "--model-id", args.benchmark_model,
        "--device", args.benchmark_device if args.benchmark_device else "cpu",
        "--output-dir", os.path.join(args.report_dir, "benchmarks"),
        "--save"
    ]
    
    # Add optional arguments
    if args.benchmark_precision:
        cmd.extend(["--precision", args.benchmark_precision])
    
    # Remove empty args
    cmd = [arg for arg in cmd if arg]
    
    success, output = run_command(cmd, "Benchmarking")
    return success

def run_batch_benchmark(args):
    """Run batch benchmarking."""
    cmd = [
        sys.executable,
        os.path.join(args.scripts_dir, "benchmarking/batch_benchmark.py"),
        "--models", args.batch_benchmark_models,
        "--devices", args.batch_benchmark_devices,
        "--output-dir", os.path.join(args.report_dir, "benchmarks"),
        "--report", os.path.join(args.report_dir, "batch_benchmark_report.md")
    ]
    
    # Add optional arguments
    if args.db_path:
        cmd.extend(["--db-path", args.db_path])
    
    # Remove empty args
    cmd = [arg for arg in cmd if arg]
    
    success, output = run_command(cmd, "Batch Benchmarking")
    return success

def run_comprehensive_suite(args):
    """Run the comprehensive test suite."""
    # Create directories if they don't exist
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.join(args.report_dir, "benchmarks"), exist_ok=True)
    
    # Set base scripts directory if not specified
    if not args.scripts_dir:
        args.scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track timing
    start_time = time.time()
    
    # Run steps based on specified actions
    results = {}
    
    if args.validate or args.all:
        results["validation"] = run_validation(args)
    
    if args.generate or args.all:
        results["generation"] = run_test_generation(args)
    
    if args.track or args.all:
        results["tracking"] = run_implementation_tracking(args)
    
    if args.integrate or args.all:
        results["integration"] = run_integration_tests(args)
    
    if args.benchmark and args.benchmark_model:
        results["benchmark"] = run_benchmark(args)
    
    if args.batch_benchmark and args.batch_benchmark_models:
        results["batch_benchmark"] = run_batch_benchmark(args)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Print summary
    logger.info("\nComprehensive Test Suite Summary:")
    logger.info(f"- Total time: {elapsed:.2f}s")
    
    for step, success in results.items():
        status = "✅ Passed" if success else "❌ Failed"
        logger.info(f"- {step.capitalize()}: {status}")
    
    return all(results.values())

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    
    parser.add_argument(
        "--test-dir",
        default="./generated_tests",
        help="Directory containing/for test files"
    )
    
    parser.add_argument(
        "--report-dir",
        default="./reports",
        help="Directory to save reports"
    )
    
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help="Directory containing script files (default: this script's directory)"
    )
    
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low", "all"],
        default="high",
        help="Priority of models to generate tests for"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files when generating tests"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to files"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked dependencies for integration tests"
    )
    
    # Testing Actions
    test_group = parser.add_argument_group("Testing Actions")
    
    test_group.add_argument(
        "--all",
        action="store_true",
        help="Run all testing steps (validate, generate, integrate, track)"
    )
    
    test_group.add_argument(
        "--validate",
        action="store_true",
        help="Run validation step"
    )
    
    test_group.add_argument(
        "--generate",
        action="store_true",
        help="Run test generation step"
    )
    
    test_group.add_argument(
        "--integrate",
        action="store_true",
        help="Run integration tests step"
    )
    
    test_group.add_argument(
        "--track",
        action="store_true",
        help="Run implementation tracking step"
    )
    
    # Benchmarking Actions
    benchmark_group = parser.add_argument_group("Benchmarking Actions")
    
    benchmark_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking on a specific model"
    )
    
    benchmark_group.add_argument(
        "--benchmark-model",
        type=str,
        help="Model ID to benchmark"
    )
    
    benchmark_group.add_argument(
        "--benchmark-device",
        type=str,
        choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"],
        help="Device to benchmark on (default: auto-select)"
    )
    
    benchmark_group.add_argument(
        "--benchmark-precision",
        type=str,
        choices=["float32", "float16", "bfloat16", "int8"],
        help="Precision for benchmarking"
    )
    
    benchmark_group.add_argument(
        "--batch-benchmark",
        action="store_true",
        help="Run batch benchmarking on multiple models"
    )
    
    benchmark_group.add_argument(
        "--batch-benchmark-models",
        type=str,
        help="Comma-separated list of models to benchmark in batch"
    )
    
    benchmark_group.add_argument(
        "--batch-benchmark-devices",
        type=str,
        default="cpu",
        help="Comma-separated list of devices to benchmark on in batch"
    )
    
    benchmark_group.add_argument(
        "--db-path",
        type=str,
        help="Path to benchmark database"
    )
    
    args = parser.parse_args()
    
    # If no specific actions specified, run all testing steps
    if not (args.validate or args.generate or args.integrate or args.track or 
            args.all or args.benchmark or args.batch_benchmark):
        args.all = True
    
    return args

def main():
    """Command-line entry point."""
    args = parse_args()
    success = run_comprehensive_suite(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())