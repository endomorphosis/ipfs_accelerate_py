#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for benchmarking vision models with hardware-aware metrics.

This script demonstrates how to use the refactored benchmark suite
to benchmark different types of vision models with hardware-aware metrics.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from benchmark import ModelBenchmark, BenchmarkSuite, BenchmarkConfig
from visualizers.plots import plot_model_comparison

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def benchmark_modern_vision_models(use_power_metrics=True, use_bandwidth_metrics=True, 
                                   model_size="tiny", output_dir=None, use_gpu=True):
    """
    Benchmark multiple modern vision models for comparison.
    
    Args:
        use_power_metrics: Whether to collect power efficiency metrics
        use_bandwidth_metrics: Whether to collect memory bandwidth metrics
        model_size: Size of models to benchmark (tiny, small, base, large)
        output_dir: Directory to save results
        use_gpu: Whether to use GPU for benchmarking
    
    Returns:
        BenchmarkSuite results
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(parent_dir) / "benchmark_results"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select models based on size
    if model_size == "tiny":
        models = [
            {"id": "google/vit-base-patch16-224", "task": "image-classification"},  # ViT
            {"id": "facebook/convnext-tiny-224", "task": "image-classification"},   # ConvNeXt
        ]
    elif model_size == "small":
        models = [
            {"id": "google/vit-base-patch16-224", "task": "image-classification"},     # ViT
            {"id": "facebook/convnext-base", "task": "image-classification"},          # ConvNeXt
            {"id": "microsoft/resnet-50", "task": "image-classification"},             # ResNet
            {"id": "microsoft/swin-base-patch4-window7-224", "task": "image-classification"},  # Swin
        ]
    elif model_size == "base":
        models = [
            {"id": "google/vit-large-patch16-224", "task": "image-classification"},     # ViT
            {"id": "facebook/convnext-large", "task": "image-classification"},          # ConvNeXt
            {"id": "facebook/dinov2-base", "task": "image-classification"},             # DINOv2
            {"id": "facebook/detr-resnet-50", "task": "object-detection"},             # DETR
        ]
    else:  # large
        models = [
            {"id": "facebook/sam-vit-huge", "task": "image-segmentation"},             # SAM
            {"id": "facebook/dinov2-large", "task": "image-classification"},           # DINOv2
            {"id": "facebook/detr-resnet-101", "task": "object-detection"},            # DETR
            {"id": "microsoft/swin-large-patch4-window7-224", "task": "image-classification"},  # Swin
        ]
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory"]
    if use_power_metrics:
        metrics.append("power")
    if use_bandwidth_metrics:
        metrics.append("bandwidth")
    
    # Create configuration for benchmark suite
    config = BenchmarkConfig()
    config.models = models
    config.hardware = hardware
    config.batch_sizes = [1, 2, 4] if use_gpu else [1, 2]
    config.sequence_lengths = [1]  # Not relevant for most vision models
    config.metrics = metrics
    config.warmup_iterations = 2
    config.test_iterations = 5
    
    # For power and bandwidth metrics
    config.power_sampling_interval = 0.1  # 100ms sampling
    config.bandwidth_sampling_interval = 0.1  # 100ms sampling
    
    # Enable hardware optimizations
    config.use_flash_attention = use_gpu  # Flash attention for transformer-based vision models
    config.use_torch_compile = hasattr(torch, "compile")  # PyTorch 2.0+ compilation
    
    # Log configuration
    logger.info(f"Benchmarking {len(models)} vision models on {hardware}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Hardware optimizations: Flash Attention={config.use_flash_attention}, " 
               f"torch.compile={config.use_torch_compile}")
    
    # Create benchmark suite
    logger.info("Creating benchmark suite...")
    suite = BenchmarkSuite(config)
    
    # Run all benchmarks
    logger.info("Running benchmarks...")
    results = suite.run()
    
    # Export combined results
    results_file_prefix = f"vision_benchmark_{model_size}_{timestamp}"
    results.export_to_json(output_dir / f"{results_file_prefix}.json")
    results.export_to_markdown(output_dir / f"{results_file_prefix}.md")
    results.export_to_csv(output_dir / f"{results_file_prefix}.csv")
    
    # Generate comparison plots
    try:
        logger.info("Generating comparison plots...")
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_latency.png",
            metric="latency",
            title=f"Latency Comparison - {model_size.capitalize()} Vision Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_throughput.png",
            metric="throughput",
            title=f"Throughput Comparison - {model_size.capitalize()} Vision Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_memory.png",
            metric="memory",
            title=f"Memory Usage Comparison - {model_size.capitalize()} Vision Models"
        )
        
        # Plot power metrics if collected
        if use_power_metrics:
            plot_model_comparison(
                results, 
                output_dir / f"{results_file_prefix}_power.png",
                metric="power_efficiency",
                title=f"Power Efficiency Comparison - {model_size.capitalize()} Vision Models"
            )
        
        # Plot bandwidth metrics if collected
        if use_bandwidth_metrics:
            plot_model_comparison(
                results, 
                output_dir / f"{results_file_prefix}_bandwidth.png",
                metric="bandwidth_utilization",
                title=f"Bandwidth Utilization Comparison - {model_size.capitalize()} Vision Models"
            )
        
        logger.info(f"Plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return results

def run_vision_model_family_comparison(output_dir=None, hardware=None):
    """
    Run a comprehensive comparison of different vision model families.
    
    Args:
        output_dir: Directory to save results
        hardware: List of hardware devices to benchmark on
    
    Returns:
        Dictionary of BenchmarkResults by model family
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(parent_dir) / "benchmark_results"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model families to compare
    model_families = {
        "vit": {"id": "google/vit-base-patch16-224", "task": "image-classification"},
        "swin": {"id": "microsoft/swin-base-patch4-window7-224", "task": "image-classification"},
        "convnext": {"id": "facebook/convnext-base", "task": "image-classification"},
        "dinov2": {"id": "facebook/dinov2-base", "task": "image-classification"},
        "detr": {"id": "facebook/detr-resnet-50", "task": "object-detection"},
        "resnet": {"id": "microsoft/resnet-50", "task": "image-classification"}
    }
    
    # Determine hardware
    if hardware is None:
        hardware = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    
    # Run benchmarks for each model family
    results = {}
    
    for family_name, model_info in model_families.items():
        logger.info(f"Benchmarking {family_name} model family: {model_info['id']}")
        
        # Create benchmark
        benchmark = ModelBenchmark(
            model_id=model_info['id'],
            task=model_info['task'],
            batch_sizes=[1, 2, 4, 8] if "cuda" in hardware else [1, 2],
            sequence_lengths=[1],  # Not relevant for vision models
            hardware=hardware,
            metrics=["latency", "throughput", "memory", "power", "bandwidth"],
            warmup_iterations=2,
            test_iterations=5,
            power_sampling_interval=0.1,
            bandwidth_sampling_interval=0.1,
            use_flash_attention=True if "cuda" in hardware else False,
            use_torch_compile=hasattr(torch, "compile"),
            output_dir=str(output_dir)
        )
        
        # Run benchmark
        results[family_name] = benchmark.run()
        
        # Export individual results
        results_file_prefix = f"vision_{family_name}_{timestamp}"
        results[family_name].export_to_json(output_dir / f"{results_file_prefix}.json")
    
    # Create comparative visualization
    try:
        from visualizers.plots import plot_families_comparison
        
        # Compare latency across families and batch sizes
        plot_families_comparison(
            results,
            output_dir / f"vision_families_latency_{timestamp}.png",
            metric="latency",
            title="Latency Comparison Across Vision Model Families"
        )
        
        # Compare throughput across families
        plot_families_comparison(
            results,
            output_dir / f"vision_families_throughput_{timestamp}.png",
            metric="throughput",
            title="Throughput Comparison Across Vision Model Families"
        )
        
        # Compare memory usage 
        plot_families_comparison(
            results,
            output_dir / f"vision_families_memory_{timestamp}.png",
            metric="memory",
            title="Memory Usage Comparison Across Vision Model Families"
        )
        
        # Compare power efficiency
        plot_families_comparison(
            results,
            output_dir / f"vision_families_power_{timestamp}.png",
            metric="power_efficiency",
            title="Power Efficiency Comparison Across Vision Model Families"
        )
        
        logger.info(f"Family comparison plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping comparative plots.")
    except Exception as e:
        logger.error(f"Error generating comparative plots: {e}")
    
    return results

def run_vision_hardware_aware_benchmark(model_id, output_dir=None, flash_attention=False, torch_compile=False):
    """
    Run hardware-aware benchmark for a specific vision model.
    
    Args:
        model_id: HuggingFace model ID to benchmark
        output_dir: Directory to save results
        flash_attention: Whether to use Flash Attention optimization
        torch_compile: Whether to use torch.compile optimization
    
    Returns:
        BenchmarkResults for the model
    """
    # Set output directory
    if output_dir is None:
        output_dir = Path(parent_dir) / "benchmark_results"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine task based on model ID
    task = "image-classification"  # Default
    
    if "detr" in model_id.lower():
        task = "object-detection"
    elif "sam" in model_id.lower() or "segmentation" in model_id.lower() or "mask" in model_id.lower():
        task = "image-segmentation"
    
    # Determine hardware
    use_cuda = torch.cuda.is_available()
    hardware = ["cuda", "cpu"] if use_cuda else ["cpu"]
    
    logger.info(f"Benchmarking {model_id} on {hardware}")
    logger.info(f"Hardware optimizations: Flash Attention={flash_attention}, torch.compile={torch_compile}")
    
    # Create benchmark
    benchmark = ModelBenchmark(
        model_id=model_id,
        task=task,
        batch_sizes=[1, 2, 4, 8] if use_cuda else [1, 2],
        sequence_lengths=[1],  # Not relevant for vision models
        hardware=hardware,
        metrics=["latency", "throughput", "memory", "power", "bandwidth"],
        warmup_iterations=2,
        test_iterations=5,
        power_sampling_interval=0.1,
        bandwidth_sampling_interval=0.1,
        use_flash_attention=flash_attention and use_cuda,
        use_torch_compile=torch_compile and hasattr(torch, "compile"),
        output_dir=str(output_dir)
    )
    
    # Run benchmark
    logger.info("Running benchmark...")
    results = benchmark.run()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export results
    model_name = model_id.split("/")[-1]
    results_file_prefix = f"vision_{model_name}_{timestamp}"
    
    results.export_to_json(output_dir / f"{results_file_prefix}.json")
    results.export_to_markdown(output_dir / f"{results_file_prefix}.md")
    results.export_to_csv(output_dir / f"{results_file_prefix}.csv")
    
    # Create visualizations
    try:
        from visualizers.plots import (
            plot_latency_by_batch_size,
            plot_throughput_by_batch_size,
            plot_memory_by_batch_size,
            plot_power_efficiency,
            plot_bandwidth_roofline
        )
        
        # Latency comparison across hardware
        plot_latency_by_batch_size(
            results,
            output_dir / f"{results_file_prefix}_latency.png",
            title=f"{model_id} - Latency by Batch Size"
        )
        
        # Throughput scaling
        plot_throughput_by_batch_size(
            results,
            output_dir / f"{results_file_prefix}_throughput.png",
            title=f"{model_id} - Throughput by Batch Size"
        )
        
        # Memory usage
        plot_memory_by_batch_size(
            results,
            output_dir / f"{results_file_prefix}_memory.png",
            title=f"{model_id} - Memory Usage by Batch Size"
        )
        
        # Power efficiency (CUDA only)
        if use_cuda:
            plot_power_efficiency(
                results,
                output_dir / f"{results_file_prefix}_power.png",
                title=f"{model_id} - Power Efficiency"
            )
            
            # Bandwidth utilization and roofline model (CUDA only)
            plot_bandwidth_roofline(
                results,
                output_dir / f"{results_file_prefix}_roofline.png",
                title=f"{model_id} - Roofline Performance Model"
            )
        
        logger.info(f"Visualizations saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping visualizations.")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
    
    # Display summary
    logger.info(f"Benchmark results for {model_id}:")
    for hw in hardware:
        hw_results = [r for r in results.results if r.hardware == hw]
        if hw_results:
            # Get results for batch size 1
            bs1_result = next((r for r in hw_results if r.batch_size == 1), None)
            if bs1_result:
                latency = bs1_result.metrics.get("latency_ms", "N/A")
                logger.info(f"  {hw.upper()} - Latency (BS=1): {latency if isinstance(latency, str) else f'{latency:.2f} ms'}")
    
    logger.info(f"Results saved to {output_dir}")
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark vision models with hardware-aware metrics")
    
    # Main options
    parser.add_argument("--mode", choices=["multi", "single", "family"], default="multi",
                        help="Benchmark mode: multiple models, single model, or family comparison")
    
    # Model options
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID to benchmark (for single mode)")
    parser.add_argument("--size", choices=["tiny", "small", "base", "large"], default="tiny",
                        help="Size of models to benchmark (for multi mode)")
    
    # Hardware options
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU only, even if GPU is available")
    
    # Optimization options
    parser.add_argument("--flash-attention", action="store_true",
                        help="Use Flash Attention optimization for transformer models")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Use torch.compile optimization for PyTorch 2.0+")
    
    # Metric options
    parser.add_argument("--no-power", action="store_true",
                        help="Disable power metrics")
    parser.add_argument("--no-bandwidth", action="store_true",
                        help="Disable bandwidth metrics")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save benchmark results")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    if has_cuda and not args.cpu_only:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Running on CPU only.")
    
    # Run benchmarks based on mode
    if args.mode == "multi":
        # Benchmark multiple models
        benchmark_modern_vision_models(
            use_power_metrics=not args.no_power,
            use_bandwidth_metrics=not args.no_bandwidth,
            model_size=args.size,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    elif args.mode == "single":
        # Benchmark specific model
        if args.model is None:
            logger.error("Model ID is required for single mode")
            sys.exit(1)
        
        run_vision_hardware_aware_benchmark(
            model_id=args.model,
            output_dir=args.output_dir,
            flash_attention=args.flash_attention,
            torch_compile=args.torch_compile
        )
    elif args.mode == "family":
        # Run family comparison
        hardware = ["cpu"] if args.cpu_only else (["cuda", "cpu"] if has_cuda else ["cpu"])
        
        run_vision_model_family_comparison(
            output_dir=args.output_dir,
            hardware=hardware
        )
    
    logger.info("Benchmarking complete!")