#!/usr/bin/env python3
"""
Run tests for skillset implementations.

This script runs the tests for skillset implementations and generates a report.
"""

import os
import sys
import glob
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"skillset_tests_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def get_test_files(test_dir: str) -> List[str]:
    """Get a list of test files in the directory.
    
    Args:
        test_dir: Directory containing test files
        
    Returns:
        List of test files
    """
    pattern = os.path.join(test_dir, 'test_hf_*.py')
    return glob.glob(pattern)


def run_test(test_file: str, device: str = "cpu", save_results: bool = True) -> Dict[str, Any]:
    """Run a test and return the results.
    
    Args:
        test_file: Path to the test file
        device: Device to run on (cpu, cuda, etc.)
        save_results: Whether to save results to file
        
    Returns:
        Dictionary with test results
    """
    model_name = os.path.basename(test_file).replace('test_hf_', '').replace('.py', '')
    logger.info(f"Running test for {model_name} on {device}")
    
    cmd = [sys.executable, test_file, f"--device={device}"]
    if save_results:
        cmd.append("--save")
        cmd.append("--output-dir=skillset_test_results")
        
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        status = process.returncode == 0
        result = {
            "model": model_name,
            "device": device,
            "success": status,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode
        }
        
        if status:
            logger.info(f"✅ Test passed for {model_name} on {device}")
        else:
            logger.error(f"❌ Test failed for {model_name} on {device}")
            logger.error(process.stderr)
            
        return result
    except Exception as e:
        logger.error(f"❌ Error running test for {model_name}: {e}")
        return {
            "model": model_name,
            "device": device,
            "success": False,
            "error": str(e),
            "returncode": -1
        }


def run_all_tests(test_dir: str, device: str = "cpu", parallel: bool = False, max_workers: int = 4) -> List[Dict[str, Any]]:
    """Run all tests in the directory.
    
    Args:
        test_dir: Directory containing test files
        device: Device to run on (cpu, cuda, etc.)
        parallel: Whether to run tests in parallel
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of test results
    """
    test_files = get_test_files(test_dir)
    if not test_files:
        logger.error(f"No test files found in {test_dir}")
        return []
        
    logger.info(f"Found {len(test_files)} test files to run")
    
    results = []
    
    if parallel and len(test_files) > 1:
        logger.info(f"Running tests in parallel with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map test files to futures
            future_to_file = {
                executor.submit(run_test, file, device, True): file
                for file in test_files
            }
            
            # Collect results as they complete
            for future in future_to_file:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    file = future_to_file[future]
                    model_name = os.path.basename(file).replace('test_hf_', '').replace('.py', '')
                    logger.error(f"❌ Error in test for {model_name}: {e}")
                    results.append({
                        "model": model_name,
                        "device": device,
                        "success": False,
                        "error": str(e),
                        "returncode": -1
                    })
    else:
        logger.info("Running tests sequentially")
        for file in test_files:
            result = run_test(file, device, True)
            results.append(result)
            
    return results


def generate_report(results: List[Dict[str, Any]], output_file: str = "skillset_test_report.md") -> str:
    """Generate a Markdown report of test results.
    
    Args:
        results: List of test results
        output_file: File to write report to
        
    Returns:
        Path to the generated report
    """
    try:
        # Count successes and failures
        total_tests = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total_tests - successful
        
        # Generate report
        report = f"# Skillset Test Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## Summary\n\n"
        report += f"- **Total tests**: {total_tests}\n"
        report += f"- **Successful**: {successful} ({(successful/total_tests)*100:.1f}%)\n"
        report += f"- **Failed**: {failed} ({(failed/total_tests)*100:.1f}%)\n\n"
        
        report += f"## Test Results\n\n"
        report += f"| Model | Device | Status | Error |\n"
        report += f"|-------|--------|--------|-------|\n"
        
        for r in sorted(results, key=lambda x: x.get("model", "")):
            model = r.get("model", "unknown")
            device = r.get("device", "unknown")
            status = "✅ Pass" if r.get("success", False) else "❌ Fail"
            if r.get("success", False):
                error = ""
            else:
                stderr = r.get("stderr", "")
                if stderr:
                    # Extract the first error line
                    error_lines = [line for line in stderr.split('\n') if "Error" in line]
                    error = error_lines[0] if error_lines else "Unknown error"
                else:
                    error = r.get("error", "Unknown error")
            
            report += f"| {model} | {device} | {status} | {error} |\n"
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write(report)
            
        logger.info(f"Report written to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return ""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for skillset implementations")
    parser.add_argument(
        "--test-dir", 
        type=str, 
        default="../skillset",
        help="Directory containing test files"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "rocm", "mps", "openvino", "qnn"],
        default="cpu",
        help="Device to run tests on"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=4,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--report-file", 
        type=str, 
        default="skillset_test_report.md",
        help="File to write report to"
    )
    parser.add_argument(
        "--model", 
        type=str,
        help="Run test for specific model only"
    )
    parser.add_argument(
        "--sample", 
        type=int,
        default=0,
        help="Run tests for a random sample of N models"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("skillset_test_results", exist_ok=True)
    
    # Get test files
    if args.model:
        # Find the test file for the specified model
        test_file = os.path.join(args.test_dir, f"test_hf_{args.model}.py")
        if not os.path.exists(test_file):
            logger.error(f"Test file not found for model {args.model}")
            return 1
            
        result = run_test(test_file, args.device, True)
        results = [result]
    elif args.sample > 0:
        # Get all test files
        test_files = get_test_files(args.test_dir)
        if not test_files:
            logger.error(f"No test files found in {args.test_dir}")
            return 1
            
        # Randomly sample N test files
        import random
        sample_size = min(args.sample, len(test_files))
        sampled_files = random.sample(test_files, sample_size)
        logger.info(f"Running tests for {sample_size} randomly sampled models")
        
        # Run tests for sampled files
        results = []
        for file in sampled_files:
            result = run_test(file, args.device, True)
            results.append(result)
    else:
        # Run all tests
        results = run_all_tests(
            args.test_dir, 
            args.device, 
            args.parallel, 
            args.max_workers
        )
        
    # Generate report
    if results:
        generate_report(results, args.report_file)
        
    # Return success if all tests passed
    return 0 if all(r.get("success", False) for r in results) else 1
    

if __name__ == "__main__":
    sys.exit(main())