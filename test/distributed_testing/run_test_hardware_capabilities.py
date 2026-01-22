#!/usr/bin/env python3
"""
Run Hardware Capability Detection Example

This script runs the hardware capability detection example with various options.
It can be used to test the hardware capability detector, including WebGPU/WebNN detection.

Usage:
    python run_test_hardware_capabilities.py [--option]
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_test_hardware_capabilities")

def main():
    """Main function to run the hardware capability detection example."""
    parser = argparse.ArgumentParser(description="Run Hardware Capability Detection Example")
    parser.add_argument("--detect-only", action="store_true", help="Only detect hardware capabilities, don't run examples")
    parser.add_argument("--with-browser", action="store_true", help="Enable browser-based WebGPU/WebNN detection")
    parser.add_argument("--db-path", help="Path to DuckDB database for storing results")
    parser.add_argument("--task-scheduling", action="store_true", help="Run task scheduling simulation")
    parser.add_argument("--worker-compatibility", action="store_true", help="Run worker compatibility example")
    parser.add_argument("--worker-id", help="Worker ID (default: auto-generated)")
    parser.add_argument("--output-json", help="Path to output JSON file for capabilities")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--run-tests", action="store_true", help="Run tests for the hardware capability detector")
    
    options = parser.parse_args()
    
    # Define script paths
    script_dir = Path(__file__).resolve().parent
    example_script = script_dir / "examples" / "hardware_capability_example.py"
    test_script = script_dir / "tests" / "test_hardware_capability_detector.py"
    
    # Check if scripts exist
    if not example_script.exists():
        logger.error(f"Example script not found: {example_script}")
        return 1
    
    if options.run_tests and not test_script.exists():
        logger.error(f"Test script not found: {test_script}")
        return 1
    
    # If --all is specified, enable all examples
    if options.all:
        options.task_scheduling = True
        options.worker_compatibility = True
    
    # Default DB path if not specified
    if not options.db_path and not options.detect_only:
        options.db_path = script_dir / "hardware_capabilities.duckdb"
    
    # Run tests if requested
    if options.run_tests:
        logger.info("Running hardware capability detector tests...")
        test_result = subprocess.run([sys.executable, str(test_script)], check=False)
        
        if test_result.returncode != 0:
            logger.error("Tests failed with return code: %d", test_result.returncode)
            return test_result.returncode
        
        logger.info("All tests passed successfully")
    
    # Run example
    if not options.run_tests or options.all:
        # Build command
        cmd = [sys.executable, str(example_script)]
        
        if options.detect_only:
            cmd.append("--detect-only")
        
        if options.with_browser:
            cmd.append("--enable-browser-detection")
        
        if options.db_path:
            cmd.extend(["--db-path", str(options.db_path)])
        
        if options.task_scheduling:
            cmd.append("--task-scheduling")
        
        if options.worker_compatibility:
            cmd.append("--worker-compatibility")
        
        if options.worker_id:
            cmd.extend(["--worker-id", options.worker_id])
        
        if options.output_json:
            cmd.extend(["--output-json", options.output_json])
        
        if options.all:
            cmd.append("--all")
        
        # Run example
        logger.info("Running hardware capability example with: %s", " ".join(cmd))
        example_result = subprocess.run(cmd, check=False)
        
        if example_result.returncode != 0:
            logger.error("Example failed with return code: %d", example_result.returncode)
            return example_result.returncode
        
        logger.info("Example completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())