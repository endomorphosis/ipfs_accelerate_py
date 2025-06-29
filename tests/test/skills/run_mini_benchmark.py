#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini benchmark script to verify the refactored benchmark suite functionality.

This script runs a small benchmark on a minimal model to verify the basic functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from refactored_benchmark_suite import ModelBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_minimal_benchmark():
    """Run a minimal benchmark to verify the functionality."""
    # Create a very minimal configuration
    benchmark = ModelBenchmark(
        model_id="bert-base-uncased",  # Small model for quick testing
        batch_sizes=[1],  # Single batch size
        sequence_lengths=[8],  # Short sequence
        hardware=["cpu"],  # CPU only for compatibility
        metrics=["latency", "throughput", "memory"],  # Basic metrics
        warmup_iterations=2,  # Minimal warmup
        test_iterations=5,  # Minimal test iterations
        output_dir="mini_benchmark_results"
    )
    
    # Run the benchmark
    logger.info("Running minimal benchmark on bert-base-uncased...")
    results = benchmark.run()
    
    # Export results
    json_path = results.export_to_json()
    markdown_path = results.export_to_markdown()
    
    logger.info(f"Benchmark complete. Results saved to:")
    logger.info(f" - JSON: {json_path}")
    logger.info(f" - Markdown: {markdown_path}")
    
    # Print a summary of the results
    hardware_results = {}
    for result in results.results:
        hardware = result.hardware
        if hardware not in hardware_results:
            hardware_results[hardware] = result
    
    logger.info("\nBenchmark Results Summary:")
    for hw, result in hardware_results.items():
        logger.info(f" - Hardware: {hw.upper()}")
        if "latency_ms" in result.metrics:
            logger.info(f"   - Latency: {result.metrics['latency_ms']:.2f} ms")
        if "throughput_items_per_sec" in result.metrics:
            logger.info(f"   - Throughput: {result.metrics['throughput_items_per_sec']:.2f} items/sec")
        if "memory_peak_mb" in result.metrics:
            logger.info(f"   - Memory: {result.metrics['memory_peak_mb']:.2f} MB")
    
    # Try to generate a plot if matplotlib is available
    try:
        plot_path = results.plot_latency_comparison()
        if plot_path:
            logger.info(f" - Plot: {plot_path}")
    except:
        logger.info("Plotting not available (matplotlib required)")
    
    # Calculate CPU to GPU speedup if both results are available
    speedup = results.get_cpu_gpu_speedup()
    if speedup is not None:
        logger.info(f"\nCPU to GPU Speedup: {speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    run_minimal_benchmark()