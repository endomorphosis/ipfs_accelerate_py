#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware-Aware Benchmark Example.

This example demonstrates how to use the complete hardware-aware benchmarking suite,
focusing on power efficiency and memory bandwidth metrics across different hardware
platforms.

This script:
1. Detects available hardware platforms
2. Runs benchmarks with all metrics (latency, throughput, memory, FLOPs, power, bandwidth)
3. Generates visualizations for hardware efficiency (power and bandwidth)
4. Provides interpretation guidelines for the results
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import local modules
from benchmark import ModelBenchmark
from hardware import get_available_hardware

def run_hardware_aware_benchmark(model_id, output_dir, hardware=None, batch_sizes=None, 
                                sequence_lengths=None, publish_to_hub=False):
    """
    Run a hardware-aware benchmark on the specified model.
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save results
        hardware: List of hardware platforms to test on (auto-detected if None)
        batch_sizes: List of batch sizes to test (defaults to [1, 2, 4, 8])
        sequence_lengths: List of sequence lengths to test (defaults to [16, 32, 64])
        publish_to_hub: Whether to publish results to HuggingFace Hub
    """
    # Auto-detect hardware if not specified
    if hardware is None:
        hardware = get_available_hardware()
        logger.info(f"Auto-detected hardware: {hardware}")
    
    # Set default batch sizes and sequence lengths if not specified
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    if sequence_lengths is None:
        sequence_lengths = [16, 32, 64]
    
    # Create benchmark
    benchmark = ModelBenchmark(
        model_id=model_id,
        hardware=hardware,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
        output_dir=output_dir,
        warmup_iterations=5,
        test_iterations=20
    )
    
    # Run benchmark
    logger.info(f"Running hardware-aware benchmark for {model_id}...")
    logger.info(f"Hardware platforms: {hardware}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Sequence lengths: {sequence_lengths}")
    
    results = benchmark.run()
    
    # Export results
    logger.info("Exporting results...")
    json_path = results.export_to_json()
    csv_path = results.export_to_csv()
    md_path = results.export_to_markdown()
    
    logger.info(f"Results exported to:")
    logger.info(f"  - JSON: {json_path}")
    logger.info(f"  - CSV: {csv_path}")
    logger.info(f"  - Markdown: {md_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Standard metrics visualizations
    latency_plot = results.plot_latency_comparison()
    throughput_plot = results.plot_throughput_scaling()
    memory_plot = results.plot_memory_usage()
    flops_plot = results.plot_flops_comparison()
    
    # Hardware-aware metrics visualizations
    power_plot = results.plot_power_efficiency()
    bandwidth_plot = results.plot_bandwidth_utilization()
    
    logger.info("Visualizations generated:")
    logger.info(f"  - Latency comparison: {latency_plot}")
    logger.info(f"  - Throughput scaling: {throughput_plot}")
    logger.info(f"  - Memory usage: {memory_plot}")
    logger.info(f"  - FLOPs comparison: {flops_plot}")
    logger.info(f"  - Power efficiency: {power_plot}")
    logger.info(f"  - Bandwidth utilization: {bandwidth_plot}")
    
    # Optionally publish to Hub
    if publish_to_hub:
        logger.info("Publishing results to HuggingFace Hub...")
        success = results.publish_to_hub()
        if success:
            logger.info("Results published successfully!")
        else:
            logger.warning("Failed to publish results to Hub")
    
    # Print summary of hardware efficiency insights
    print_hardware_efficiency_insights(results)
    
    return results

def print_hardware_efficiency_insights(results):
    """
    Print hardware efficiency insights from the benchmark results.
    
    Args:
        results: BenchmarkResults object
    """
    logger.info("\n=== HARDWARE EFFICIENCY INSIGHTS ===\n")
    
    # Check if we have results for multiple hardware platforms
    hardware_platforms = set(result.hardware for result in results.results)
    
    if len(hardware_platforms) > 1:
        # Compare hardware platforms
        logger.info("Hardware Platform Comparison:")
        
        # Get CPU vs GPU speedup if available
        speedup = results.get_cpu_gpu_speedup()
        if speedup is not None:
            logger.info(f"  - CPU to GPU Speedup: {speedup:.1f}x")
    
    # Power efficiency insights
    power_metrics_available = any("power_avg_watts" in result.metrics for result in results.results)
    if power_metrics_available:
        logger.info("\nPower Efficiency Insights:")
        
        for hw in hardware_platforms:
            # Get power metrics for this hardware
            hw_results = [r for r in results.results if r.hardware == hw]
            if not hw_results:
                continue
                
            # Get the result with the highest efficiency
            power_results = [r for r in hw_results if "gflops_per_watt" in r.metrics]
            if power_results:
                max_efficiency_result = max(power_results, key=lambda r: r.metrics["gflops_per_watt"])
                
                logger.info(f"  {hw.upper()}:")
                logger.info(f"    - Average Power: {max_efficiency_result.metrics['power_avg_watts']:.2f} watts")
                logger.info(f"    - Computational Efficiency: {max_efficiency_result.metrics['gflops_per_watt']:.2f} GFLOPs/watt")
                logger.info(f"    - Best Batch Size for Efficiency: {max_efficiency_result.batch_size}")
    
    # Bandwidth utilization insights
    bandwidth_metrics_available = any("avg_bandwidth_gbps" in result.metrics for result in results.results)
    if bandwidth_metrics_available:
        logger.info("\nMemory Bandwidth Insights:")
        
        for hw in hardware_platforms:
            # Get bandwidth metrics for this hardware
            hw_results = [r for r in results.results if r.hardware == hw]
            if not hw_results:
                continue
                
            # Get results with bandwidth metrics
            bw_results = [r for r in hw_results if "bandwidth_utilization_percent" in r.metrics]
            if bw_results:
                max_util_result = max(bw_results, key=lambda r: r.metrics["bandwidth_utilization_percent"])
                
                logger.info(f"  {hw.upper()}:")
                logger.info(f"    - Average Bandwidth: {max_util_result.metrics['avg_bandwidth_gbps']:.2f} GB/s")
                logger.info(f"    - Peak Theoretical Bandwidth: {max_util_result.metrics['peak_theoretical_bandwidth_gbps']:.2f} GB/s")
                logger.info(f"    - Utilization: {max_util_result.metrics['bandwidth_utilization_percent']:.2f}%")
                
                # Check compute vs memory bound
                if "compute_bound" in max_util_result.metrics:
                    bound_type = "compute-bound" if max_util_result.metrics["compute_bound"] else "memory-bound"
                    logger.info(f"    - Performance Characteristic: {bound_type}")
                
                # Check arithmetic intensity
                if "arithmetic_intensity_flops_per_byte" in max_util_result.metrics:
                    ai = max_util_result.metrics["arithmetic_intensity_flops_per_byte"]
                    logger.info(f"    - Arithmetic Intensity: {ai:.2f} FLOPs/byte")
    
    # Roofline model interpretation
    roofline_data_available = any("roofline_data" in result.metrics for result in results.results)
    if roofline_data_available:
        logger.info("\nRoofline Model Interpretation:")
        
        for hw in hardware_platforms:
            # Get results with roofline data
            hw_results = [r for r in results.results if r.hardware == hw and "roofline_data" in r.metrics]
            if not hw_results:
                continue
                
            sample_result = hw_results[0]
            roofline_data = sample_result.metrics["roofline_data"]
            
            logger.info(f"  {hw.upper()}:")
            logger.info(f"    - Peak Compute: {roofline_data['peak_compute_flops']/1e12:.2f} TFLOPS")
            logger.info(f"    - Peak Memory Bandwidth: {roofline_data['peak_memory_bandwidth_bytes_per_sec']/1e9:.2f} GB/s")
            logger.info(f"    - Ridge Point: {roofline_data['ridge_point_flops_per_byte']:.2f} FLOPs/byte")
            
            # Add interpretation
            ai = roofline_data['arithmetic_intensity_flops_per_byte']
            ridge_point = roofline_data['ridge_point_flops_per_byte']
            
            if ai < ridge_point:
                logger.info("    - Interpretation: The model is MEMORY-BOUND. Performance can be improved by:")
                logger.info("      * Reducing memory transfers")
                logger.info("      * Increasing data reuse")
                logger.info("      * Optimizing memory access patterns")
            else:
                logger.info("    - Interpretation: The model is COMPUTE-BOUND. Performance can be improved by:")
                logger.info("      * Optimizing compute operations")
                logger.info("      * Leveraging hardware acceleration (Tensor Cores, SIMD)")
                logger.info("      * Model quantization or pruning")
    
    logger.info("\n=== END OF HARDWARE EFFICIENCY INSIGHTS ===\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hardware-aware benchmark example")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="HuggingFace model ID to benchmark")
    parser.add_argument("--hardware", type=str, nargs="+",
                        help="Hardware platforms to benchmark on (auto-detected if not specified)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[16, 32, 64],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--publish", action="store_true",
                        help="Publish results to HuggingFace Hub")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run hardware-aware benchmark
    results = run_hardware_aware_benchmark(
        model_id=args.model,
        output_dir=args.output_dir,
        hardware=args.hardware,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        publish_to_hub=args.publish
    )
    
    logger.info("\nHARDWARE-AWARE BENCHMARK COMPLETE")
    logger.info(f"Model: {args.model}")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()