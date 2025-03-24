#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware-Aware Vision Model Benchmark Example.

This example demonstrates how to use the hardware-aware benchmarking suite with modern
vision models like DETR, SAM, DINOv2, and Swin transformers, focusing on power efficiency 
and memory bandwidth metrics across different hardware platforms.

This script:
1. Detects available hardware platforms
2. Runs benchmarks on vision models with hardware optimizations
3. Measures power efficiency and memory bandwidth metrics
4. Generates visualizations for hardware efficiency
5. Provides interpretation of the results
"""

import os
import sys
import logging
import argparse
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

def run_vision_hardware_aware_benchmark(model_id, output_dir, hardware=None, batch_sizes=None, 
                                      sequence_lengths=None, use_flash_attention=False,
                                      use_torch_compile=False, use_4bit=False, use_8bit=False):
    """
    Run a hardware-aware benchmark on the specified vision model.
    
    Args:
        model_id: HuggingFace model ID
        output_dir: Directory to save results
        hardware: List of hardware platforms to test on (auto-detected if None)
        batch_sizes: List of batch sizes to test (defaults to [1, 2, 4])
        sequence_lengths: Not used for vision models but kept for API consistency
        use_flash_attention: Whether to use Flash Attention optimization
        use_torch_compile: Whether to use torch.compile for PyTorch 2.0+ optimizations
        use_4bit: Whether to use 4-bit quantization (for compatible models)
        use_8bit: Whether to use 8-bit quantization (for compatible models)
    """
    # Auto-detect hardware if not specified
    if hardware is None:
        hardware = get_available_hardware()
        logger.info(f"Auto-detected hardware: {hardware}")
    
    # Set default batch sizes if not specified
    if batch_sizes is None:
        # Use smaller batch sizes for large models like SAM
        if "sam" in model_id.lower():
            batch_sizes = [1, 2]
        else:
            batch_sizes = [1, 2, 4, 8]
    
    # Create benchmark
    benchmark = ModelBenchmark(
        model_id=model_id,
        hardware=hardware,
        batch_sizes=batch_sizes,
        metrics=["latency", "throughput", "memory", "flops", "power", "bandwidth"],
        output_dir=output_dir,
        warmup_iterations=3,
        test_iterations=10,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        flash_attention=use_flash_attention,
        torch_compile=use_torch_compile
    )
    
    # Run benchmark
    logger.info(f"Running hardware-aware benchmark for vision model {model_id}...")
    logger.info(f"Hardware platforms: {hardware}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Hardware optimizations:")
    logger.info(f"  - Flash Attention: {use_flash_attention}")
    logger.info(f"  - torch.compile: {use_torch_compile}")
    logger.info(f"  - 4-bit quantization: {use_4bit}")
    logger.info(f"  - 8-bit quantization: {use_8bit}")
    
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
    
    # Print hardware efficiency insights
    print_hardware_efficiency_insights(results)
    
    return results

def run_vision_model_family_comparison(output_dir, hardware=None, batch_sizes=None):
    """
    Run a comparison of different vision model families to analyze hardware efficiency.
    
    Args:
        output_dir: Directory to save results
        hardware: List of hardware platforms to test on (auto-detected if None)
        batch_sizes: List of batch sizes to test
    """
    # Auto-detect hardware if not specified
    if hardware is None:
        hardware = get_available_hardware()
    
    # Set default batch sizes if not specified
    if batch_sizes is None:
        batch_sizes = [1, 2]
    
    # Define representative models from different vision model families
    vision_models = {
        "ViT": "google/vit-base-patch16-224",
        "ConvNeXt": "facebook/convnext-tiny-224",
        "DINOv2": "facebook/dinov2-base",
        "DETR": "facebook/detr-resnet-50",
        "Swin": "microsoft/swin-base-patch4-window7-224",
    }
    
    # Optional: Add SAM if you want to include it (resource intensive)
    # vision_models["SAM"] = "facebook/sam-vit-base"
    
    # Create results directory for this comparison
    comparison_dir = os.path.join(output_dir, "vision_model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run benchmarks for each model
    results = {}
    for model_family, model_id in vision_models.items():
        logger.info(f"Benchmarking {model_family} ({model_id})...")
        
        # Create model-specific output directory
        model_dir = os.path.join(comparison_dir, model_family)
        os.makedirs(model_dir, exist_ok=True)
        
        # Run benchmark with hardware optimizations
        try:
            benchmark_results = run_vision_hardware_aware_benchmark(
                model_id=model_id,
                output_dir=model_dir,
                hardware=hardware,
                batch_sizes=batch_sizes,
                use_flash_attention=True,
                use_torch_compile=True
            )
            results[model_family] = benchmark_results
        except Exception as e:
            logger.error(f"Error benchmarking {model_family}: {e}")
    
    # Create comparison report
    create_model_family_comparison_report(results, comparison_dir)
    
    return results

def create_model_family_comparison_report(benchmark_results, output_dir):
    """
    Create a comparison report for different vision model families.
    
    Args:
        benchmark_results: Dictionary mapping model families to benchmark results
        output_dir: Directory to save the report
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("pandas and matplotlib are required for comparison reports")
        return
    
    # Create DataFrame to store comparison metrics
    all_metrics = []
    
    for family, results in benchmark_results.items():
        for result in results.results:
            # Only consider batch size 1 for fair comparison
            if result.batch_size != 1:
                continue
                
            metrics = {
                "Model Family": family,
                "Hardware": result.hardware,
                "Batch Size": result.batch_size
            }
            
            # Add performance metrics
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    metrics[metric_name] = metric_value
            
            all_metrics.append(metrics)
    
    # Create DataFrame
    if not all_metrics:
        logger.error("No metrics data collected for comparison")
        return
        
    df = pd.DataFrame(all_metrics)
    
    # Save the data
    csv_path = os.path.join(output_dir, "vision_model_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Create visualizations
    
    # 1. Latency comparison
    if "latency_ms" in df.columns:
        plt.figure(figsize=(12, 6))
        latency_pivot = df.pivot(index="Model Family", columns="Hardware", values="latency_ms")
        ax = latency_pivot.plot(kind="bar")
        plt.title("Latency Comparison Across Vision Model Families")
        plt.ylabel("Latency (ms)")
        plt.xlabel("Model Family")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vision_latency_comparison.png"))
    
    # 2. Power efficiency comparison
    if "gflops_per_watt" in df.columns:
        plt.figure(figsize=(12, 6))
        power_pivot = df.pivot(index="Model Family", columns="Hardware", values="gflops_per_watt")
        ax = power_pivot.plot(kind="bar")
        plt.title("Power Efficiency Comparison (GFLOPs/watt)")
        plt.ylabel("GFLOPs/watt")
        plt.xlabel("Model Family")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vision_power_efficiency_comparison.png"))
    
    # 3. Bandwidth utilization comparison
    if "bandwidth_utilization_percent" in df.columns:
        plt.figure(figsize=(12, 6))
        bw_pivot = df.pivot(index="Model Family", columns="Hardware", values="bandwidth_utilization_percent")
        ax = bw_pivot.plot(kind="bar")
        plt.title("Memory Bandwidth Utilization (%)")
        plt.ylabel("Utilization (%)")
        plt.xlabel("Model Family")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vision_bandwidth_utilization_comparison.png"))
    
    # 4. Arithmetic intensity comparison
    if "arithmetic_intensity_flops_per_byte" in df.columns:
        plt.figure(figsize=(12, 6))
        ai_pivot = df.pivot(index="Model Family", columns="Hardware", values="arithmetic_intensity_flops_per_byte")
        ax = ai_pivot.plot(kind="bar")
        plt.title("Arithmetic Intensity (FLOPs/byte)")
        plt.ylabel("FLOPs/byte")
        plt.xlabel("Model Family")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vision_arithmetic_intensity_comparison.png"))
    
    # Create a markdown report
    markdown = f"# Vision Model Family Hardware Efficiency Comparison\n\n"
    markdown += f"This report compares the hardware efficiency of different vision model families.\n\n"
    
    # Add hardware info
    hardware_platforms = df["Hardware"].unique()
    markdown += f"## Hardware Platforms\n\n"
    for hw in hardware_platforms:
        markdown += f"- {hw.upper()}\n"
    markdown += "\n"
    
    # Add latency comparison
    if "latency_ms" in df.columns:
        markdown += f"## Latency Comparison\n\n"
        markdown += "| Model Family | " + " | ".join(hw.upper() for hw in hardware_platforms) + " |\n"
        markdown += "| --- | " + " | ".join(["---"] * len(hardware_platforms)) + " |\n"
        
        for family in df["Model Family"].unique():
            family_data = df[df["Model Family"] == family]
            row_values = [family]
            
            for hw in hardware_platforms:
                hw_value = family_data[family_data["Hardware"] == hw]["latency_ms"].values
                value_str = f"{hw_value[0]:.2f}" if len(hw_value) > 0 else "N/A"
                row_values.append(value_str)
            
            markdown += "| " + " | ".join(row_values) + " |\n"
        
        markdown += "\n"
    
    # Add power efficiency comparison
    if "gflops_per_watt" in df.columns:
        markdown += f"## Power Efficiency Comparison (GFLOPs/watt)\n\n"
        markdown += "| Model Family | " + " | ".join(hw.upper() for hw in hardware_platforms) + " |\n"
        markdown += "| --- | " + " | ".join(["---"] * len(hardware_platforms)) + " |\n"
        
        for family in df["Model Family"].unique():
            family_data = df[df["Model Family"] == family]
            row_values = [family]
            
            for hw in hardware_platforms:
                hw_value = family_data[family_data["Hardware"] == hw]["gflops_per_watt"].values
                value_str = f"{hw_value[0]:.2f}" if len(hw_value) > 0 else "N/A"
                row_values.append(value_str)
            
            markdown += "| " + " | ".join(row_values) + " |\n"
        
        markdown += "\n"
    
    # Save markdown report
    with open(os.path.join(output_dir, "vision_model_comparison.md"), "w") as f:
        f.write(markdown)
    
    logger.info(f"Model family comparison report saved to {output_dir}")

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
    
    # Vision model specific insights
    logger.info("\nVision Model Specific Insights:")
    
    # Determine model type
    model_id_lower = results.config.model_id.lower()
    if "vit" in model_id_lower:
        logger.info("  - Model Type: Vision Transformer (ViT)")
        logger.info("  - Characteristics: Attention-heavy computation, moderate memory requirements")
        logger.info("  - Optimization Tips: Consider using Flash Attention for better performance and memory efficiency")
    elif "convnext" in model_id_lower:
        logger.info("  - Model Type: ConvNeXt (Modern CNN)")
        logger.info("  - Characteristics: Compute-bound, efficient memory usage")
        logger.info("  - Optimization Tips: Can benefit from tensor core acceleration and torch.compile")
    elif "detr" in model_id_lower:
        logger.info("  - Model Type: DETR (DEtection TRansformer)")
        logger.info("  - Characteristics: Mix of CNN and transformer computation, higher memory usage")
        logger.info("  - Optimization Tips: Balanced approach to memory and compute optimization")
    elif "sam" in model_id_lower:
        logger.info("  - Model Type: SAM (Segment Anything Model)")
        logger.info("  - Characteristics: Very memory intensive, requires larger VRAM for batch processing")
        logger.info("  - Optimization Tips: Consider quantization and gradient checkpointing")
    elif "dino" in model_id_lower:
        logger.info("  - Model Type: DINOv2")
        logger.info("  - Characteristics: Self-supervised vision transformer, compute intensive")
        logger.info("  - Optimization Tips: Benefits from hardware acceleration for attention operations")
    elif "swin" in model_id_lower:
        logger.info("  - Model Type: Swin Transformer")
        logger.info("  - Characteristics: Hierarchical transformer architecture, better locality than ViT")
        logger.info("  - Optimization Tips: Good candidate for mixed precision and Flash Attention")
    
    logger.info("\n=== END OF HARDWARE EFFICIENCY INSIGHTS ===\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hardware-aware vision benchmark example")
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224",
                        help="HuggingFace model ID to benchmark")
    parser.add_argument("--hardware", type=str, nargs="+",
                        help="Hardware platforms to benchmark on (auto-detected if not specified)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="Batch sizes to benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--flash-attention", action="store_true",
                        help="Use Flash Attention optimization")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Use torch.compile optimization")
    parser.add_argument("--comparison", action="store_true",
                        help="Run comparison across multiple vision model families")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.comparison:
        # Run comparison across multiple vision model families
        logger.info("Running vision model family comparison...")
        run_vision_model_family_comparison(
            output_dir=args.output_dir,
            hardware=args.hardware,
            batch_sizes=args.batch_sizes
        )
    else:
        # Run hardware-aware benchmark for single model
        results = run_vision_hardware_aware_benchmark(
            model_id=args.model,
            output_dir=args.output_dir,
            hardware=args.hardware,
            batch_sizes=args.batch_sizes,
            use_flash_attention=args.flash_attention,
            use_torch_compile=args.torch_compile
        )
    
    logger.info("\nVISION MODEL HARDWARE-AWARE BENCHMARK COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()