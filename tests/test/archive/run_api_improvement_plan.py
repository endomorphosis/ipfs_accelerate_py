#!/usr/bin/env python
"""
Script to run the complete API improvement plan.
This is a wrapper script that:
1. Runs the comprehensive implementation plan
2. Performs comprehensive testing
3. Generates implementation reports
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_api_improvement_plan")

# Get base paths
script_dir = Path(__file__).parent
project_root = script_dir.parent

def run_command(cmd, cwd=None):
    """Run a command and return the output"""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=True, 
                             capture_output=True, text=True)
        return True, proc.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr

def run_api_plan(api=None, skip_test=False, skip_backup=False, verbose=False):
    """Run the comprehensive API improvement plan"""
    cmd = [sys.executable, str(script_dir / "complete_api_improvement_plan.py")]
    
    if api:
        cmd.extend(["--api", api])
    if skip_test:
        cmd.append("--skip-test")
    if skip_backup:
        cmd.append("--skip-backup")
    if verbose:
        cmd.append("--verbose")
    
    success, output = run_command(cmd)
    return success

def run_queue_tests(api=None):
    """Run comprehensive queue and backoff tests"""
    cmd = [sys.executable, str(script_dir / "run_queue_backoff_tests.py")]
    
    if api:
        cmd.extend(["--apis", api])
    
    # Generate a timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success, output = run_command(cmd)
    return success, f"queue_backoff_tests_{timestamp}.json"

def main():
    parser = argparse.ArgumentParser(description="Run the complete API improvement plan")
    parser.add_argument("--api", help="Specific API to update (default: all)")
    parser.add_argument("--skip-test", action="store_true", 
                       help="Skip verification tests")
    parser.add_argument("--skip-backup", action="store_true", 
                       help="Skip file backups")
    parser.add_argument("--only-test", action="store_true", 
                       help="Only run tests, skip implementation")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose output")
    
    args = parser.parse_args()
    
    logger.info("=== API Improvement Plan Execution ===")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target API: {args.api or 'ALL'}")
    
    # Track success status
    success = True
    
    # Step 1: Run the implementation plan (unless only testing)
    if not args.only_test:
        logger.info("\n=== Running Comprehensive Implementation Plan ===")
        impl_success = run_api_plan(
            api=args.api,
            skip_test=args.skip_test,
            skip_backup=args.skip_backup,
            verbose=args.verbose
        )
        
        if not impl_success:
            logger.error("Implementation plan failed!")
            success = False
        else:
            logger.info("Implementation plan completed successfully!")
    
    # Step 2: Run comprehensive tests
    if args.only_test or not args.skip_test:
        logger.info("\n=== Running Comprehensive Queue Tests ===")
        test_success, output_file = run_queue_tests(args.api)
        
        if not test_success:
            logger.error("Comprehensive tests failed!")
            success = False
        else:
            logger.info(f"Comprehensive tests completed successfully! Results saved to {output_file}")
    
    # Print final status
    logger.info("\n=== API Improvement Plan Summary ===")
    logger.info(f"Target API: {args.api or 'ALL'}")
    logger.info(f"Status: {'SUCCESS' if success else 'FAILURE'}")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        logger.info("\nAll API backends have been successfully improved with:")
        logger.info("- Queue Management: Thread-safe request queuing with concurrency limits")
        logger.info("- Backoff System: Exponential backoff for failed requests")
        logger.info("- Circuit Breaker: Automatic service outage detection and recovery")
        logger.info("- Request Tracking: Unique request IDs and detailed error reporting")
        logger.info("- Priority Queue: Priority-based request scheduling")
        logger.info("- Monitoring: Comprehensive metrics collection and reporting")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())