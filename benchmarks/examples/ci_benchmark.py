#!/usr/bin/env python3
"""
CI/CD Integration Example for Benchmark Framework

This script demonstrates how to integrate the benchmark framework into CI/CD pipelines.
It runs a benchmark suite and compares the results with a baseline to detect regressions.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import benchmark_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_core import BenchmarkRunner, BenchmarkRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CI_Benchmark")

def run_performance_suite(args):
    """Run performance benchmark suite and compare with baseline."""
    
    # Create benchmark runner
    runner = BenchmarkRunner(config={
        "output_dir": args.output_dir
    })
    
    # Run benchmark suite
    logger.info(f"Running benchmark suite: {args.suite}")
    suite_result = runner.execute_suite(args.suite)
    
    logger.info(f"Benchmark suite complete:")
    logger.info(f"  - Benchmarks executed: {suite_result['benchmarks_executed']}")
    logger.info(f"  - Successful: {suite_result['success_count']}")
    logger.info(f"  - Failed: {suite_result['failure_count']}")
    logger.info(f"  - Results saved to: {suite_result['results_path']}")
    logger.info(f"  - Report generated at: {suite_result['report_path']}")
    
    # Compare with baseline if provided
    if args.baseline:
        if not os.path.exists(args.baseline):
            logger.error(f"Baseline file not found: {args.baseline}")
            return 1
            
        logger.info(f"Comparing results with baseline: {args.baseline}")
        
        comparison_result = runner.compare_with_baseline(
            suite_result['results_path'],
            args.baseline,
            args.threshold
        )
        
        logger.info(f"Comparison complete:")
        logger.info(f"  - Comparison saved to: {comparison_result['comparison_path']}")
        logger.info(f"  - Report generated at: {comparison_result['report_path']}")
        
        # Check for significant regressions
        if comparison_result['has_regression']:
            logger.error("Significant performance regressions detected!")
            
            # Print regressions
            regressions = comparison_result['comparison']['regressions']
            for name, data in regressions.items():
                logger.error(f"Regression in {name}:")
                for metric, values in data['metrics'].items():
                    baseline = values['baseline']
                    current = values['current']
                    change = values['change'] * 100  # Convert to percentage
                    logger.error(f"  - {metric}: {baseline:.4f} -> {current:.4f} ({change:+.2f}%)")
                    
            # Exit with error code for CI/CD
            return 1
            
    return 0

def main():
    """Main entry point for CI/CD integration."""
    parser = argparse.ArgumentParser(description="CI/CD Benchmark Runner")
    
    parser.add_argument("--suite", type=str, required=True,
                       help="Path to benchmark suite configuration file")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--baseline", type=str,
                       help="Path to baseline results for comparison")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="Regression threshold (0.05 = 5%)")
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="List available benchmarks and exit")
                       
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List benchmarks if requested
    if args.list_benchmarks:
        print("Available benchmarks:")
        for name, metadata in BenchmarkRegistry.list_benchmarks().items():
            print(f"  - {name}:")
            for key, value in metadata.items():
                print(f"      {key}: {value}")
        return 0
        
    # Run benchmark suite
    return run_performance_suite(args)

if __name__ == "__main__":
    sys.exit(main())