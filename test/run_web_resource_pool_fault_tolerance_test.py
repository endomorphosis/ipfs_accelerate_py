#!/usr/bin/env python3
"""
Run WebGPU/WebNN Resource Pool Fault Tolerance Integration Tests

This script provides a simple interface to run the fault tolerance integration
tests in different modes. It is designed to be used both in development and
CI/CD environments.

Usage:
    # Run basic test with mock implementation (no real browsers needed)
    python run_web_resource_pool_fault_tolerance_test.py --mock

    # Run comprehensive test with mock implementation
    python run_web_resource_pool_fault_tolerance_test.py --mock --comprehensive

    # Run real browser test with specific model
    python run_web_resource_pool_fault_tolerance_test.py --model bert-base-uncased

    # Run stress test with specific settings
    python run_web_resource_pool_fault_tolerance_test.py --stress-test --iterations 10
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the fault tolerance integration tests."""
    parser = argparse.ArgumentParser(
        description="Run WebGPU/WebNN Resource Pool Fault Tolerance Integration Tests"
    )
    
    # Basic options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                      help="Model name to test")
    parser.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    parser.add_argument("--mock", action="store_true",
                      help="Use mock implementation for testing without browsers")
    
    # Test modes
    parser.add_argument("--basic", action="store_true",
                      help="Run basic integration test")
    parser.add_argument("--comparative", action="store_true",
                      help="Run comparative integration test")
    parser.add_argument("--stress-test", action="store_true",
                      help="Run stress test integration")
    parser.add_argument("--resource-pool", action="store_true",
                      help="Test integration with resource pool")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run all test modes")
    
    # Output options
    parser.add_argument("--output-dir", type=str,
                      help="Directory for output files")
    
    # Stress test options
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations for stress testing")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"./fault_tolerance_test_results_{timestamp}"
    
    # Build command
    cmd = ["python", "test_web_resource_pool_fault_tolerance_integration.py"]
    
    # Add arguments
    cmd.extend(["--model", args.model])
    cmd.extend(["--browsers", args.browsers])
    cmd.extend(["--output-dir", output_dir])
    
    if args.mock:
        cmd.append("--mock")
    
    if args.basic:
        cmd.append("--basic")
    
    if args.comparative:
        cmd.append("--comparative")
        
    if args.stress_test:
        cmd.append("--stress-test")
        cmd.extend(["--iterations", str(args.iterations)])
        
    if args.resource_pool:
        cmd.append("--resource-pool")
        
    if args.comprehensive:
        cmd.append("--comprehensive")
        
    if args.verbose:
        cmd.append("--verbose")
    
    # Log the command
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Update the command to use the fixed mock implementation
    if args.mock:
        # Replace the file reference to use our fixed mock implementation
        fixed_cmd = []
        for item in cmd:
            if item == "test_web_resource_pool_fault_tolerance_integration.py":
                fixed_cmd.append("test_web_resource_pool_fault_tolerance_integration.py")
                # Add environment variable to use fixed mock implementation
                os.environ["USE_FIXED_MOCK"] = "1"
            else:
                fixed_cmd.append(item)
        cmd = fixed_cmd
    
    # Execute the integration test
    try:
        result = subprocess.run(cmd, check=True)
        
        logger.info(f"Test completed with exit code: {result.returncode}")
        logger.info(f"Results available in: {output_dir}")
        
        # Return the test exit code
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Test failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())