#!/usr/bin/env python3
"""
Single Model Benchmark Runner

This script runs a benchmark for a specific model on a specific hardware platform.
It serves as a wrapper around benchmark_hardware_models.py to allow testing of
the fix for issue #10 from NEXT_STEPS.md.

Usage:
    python run_single_benchmark.py --model bert --hardware cpu --batch-sizes 1
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import the run_benchmark function from benchmark_hardware_models.py
try:
    from benchmark_hardware_models import (
        run_benchmark, save_benchmark_results, generate_markdown_report,
        BENCHMARK_CONFIG, KEY_MODELS, HARDWARE_PLATFORMS
    )
except ImportError:
    logger.error("Error importing from benchmark_hardware_models.py")
    sys.exit(1)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run a single model benchmark on a specific hardware platform")
    parser.add_argument("--model", required=True, help="Model to benchmark (e.g., bert, t5, vit)")
    parser.add_argument("--hardware", required=True, help="Hardware platform to benchmark on (e.g., cpu, cuda)")
    parser.add_argument("--batch-sizes", default="1", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--quick", action="store_true", help="Run a faster, less comprehensive benchmark")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    parser.add_argument("--db-path", type=str, default=None, help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true", help="Store results only in the database, not in JSON")
    
    args = parser.parse_args()
    
    # Validate model and hardware
    if args.model not in KEY_MODELS:
        logger.error(f"Unknown model: {args.model}")
        logger.info(f"Available models: {', '.join(KEY_MODELS.keys())}")
        return 1
    
    if args.hardware not in HARDWARE_PLATFORMS:
        logger.error(f"Unknown hardware platform: {args.hardware}")
        logger.info(f"Available hardware platforms: {', '.join(HARDWARE_PLATFORMS.keys())}")
        return 1
    
    # Process batch sizes
    batch_sizes = None
    if args.batch_sizes:
        try:
            batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
        except ValueError:
            logger.error("Error: Batch sizes must be comma-separated integers")
            return 1
    
    # Run benchmark
    logger.info(f"Running benchmark for {args.model} on {args.hardware} with batch sizes {batch_sizes}")
    
    result = run_benchmark(
        args.model,
        args.hardware,
        batch_sizes,
        quick_mode=args.quick,
        verbose=args.verbose
    )
    
    # Save result
    if result.get("batch_sizes"):
        output_path = save_benchmark_results(result, args.output_dir)
        logger.info(f"Benchmark results saved to: {output_path}")
    
    # Generate report
    report_file = os.path.join(args.output_dir, f"benchmark_report_{args.model}_{args.hardware}.md")
    report = generate_markdown_report([result], report_file)
    logger.info(f"Benchmark report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())