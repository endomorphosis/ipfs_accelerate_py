#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for benchmarking multimodal models with hardware-aware metrics.

This script demonstrates how to use the refactored benchmark suite
to benchmark different types of multimodal models with hardware-aware metrics.
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

def benchmark_modern_multimodal_models(use_power_metrics=True, use_bandwidth_metrics=True, 
                                      model_size="tiny", output_dir=None, use_gpu=True):
    """
    Benchmark multiple modern multimodal models for comparison.
    
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
            {"id": "openai/clip-vit-base-patch32", "task": "image-to-text"},  # CLIP
            {"id": "Salesforce/blip-image-captioning-base", "task": "image-to-text"},  # BLIP
        ]
    elif model_size == "small":
        models = [
            {"id": "openai/clip-vit-base-patch16", "task": "image-to-text"},  # CLIP
            {"id": "Salesforce/blip-image-captioning-large", "task": "image-to-text"},  # BLIP
            {"id": "microsoft/git-base", "task": "image-to-text"},  # GIT
            {"id": "google/pix2struct-base", "task": "image-to-text"},  # Pix2Struct
        ]
    elif model_size == "base":
        models = [
            {"id": "openai/clip-vit-large-patch14", "task": "image-to-text"},  # CLIP
            {"id": "Salesforce/blip2-opt-2.7b", "task": "image-to-text"},  # BLIP-2
            {"id": "Salesforce/instructblip-flan-t5-xl", "task": "image-to-text"},  # InstructBLIP
            {"id": "facebook/flava-full", "task": "image-to-text"},  # FLAVA
        ]
    else:  # large
        models = [
            {"id": "openai/clip-vit-large-patch14-336", "task": "image-to-text"},  # CLIP
            {"id": "Salesforce/blip2-flan-t5-xl", "task": "image-to-text"},  # BLIP-2
            {"id": "Salesforce/instructblip-vicuna-7b", "task": "image-to-text"},  # InstructBLIP
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
    config.sequence_lengths = [32, 64] if use_gpu else [32]
    config.metrics = metrics
    config.warmup_iterations = 2
    config.test_iterations = 5
    
    # For power and bandwidth metrics
    config.power_sampling_interval = 0.1  # 100ms sampling
    config.bandwidth_sampling_interval = 0.1  # 100ms sampling
    
    # Enable hardware optimizations
    config.use_flash_attention = use_gpu  # Flash attention for transformer models
    config.use_torch_compile = hasattr(torch, "compile")  # PyTorch 2.0+ compilation
    
    # Log configuration
    logger.info(f"Benchmarking {len(models)} models on {hardware}")
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
    results_file_prefix = f"multimodal_benchmark_{model_size}_{timestamp}"
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
            title=f"Latency Comparison - {model_size.capitalize()} Multimodal Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_throughput.png",
            metric="throughput",
            title=f"Throughput Comparison - {model_size.capitalize()} Multimodal Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_memory.png",
            metric="memory",
            title=f"Memory Usage Comparison - {model_size.capitalize()} Multimodal Models"
        )
        
        # Plot power metrics if collected
        if use_power_metrics:
            plot_model_comparison(
                results, 
                output_dir / f"{results_file_prefix}_power.png",
                metric="power_efficiency",
                title=f"Power Efficiency Comparison - {model_size.capitalize()} Multimodal Models"
            )
        
        # Plot bandwidth metrics if collected
        if use_bandwidth_metrics:
            plot_model_comparison(
                results, 
                output_dir / f"{results_file_prefix}_bandwidth.png",
                metric="bandwidth_utilization",
                title=f"Bandwidth Utilization Comparison - {model_size.capitalize()} Multimodal Models"
            )
        
        logger.info(f"Plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return results

def benchmark_specific_multimodal_task(task="visual-question-answering", model_id=None, 
                                      use_power_metrics=True, output_dir=None, use_gpu=True):
    """
    Benchmark a specific multimodal model and task with hardware metrics.
    
    Args:
        task: Multimodal task to benchmark
        model_id: Specific model ID to benchmark (uses default if None)
        use_power_metrics: Whether to collect power efficiency metrics
        output_dir: Directory to save results
        use_gpu: Whether to use GPU for benchmarking
    
    Returns:
        ModelBenchmark results
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
    
    # Default models by task
    default_models = {
        "image-to-text": "Salesforce/blip-image-captioning-base",
        "visual-question-answering": "dandelin/vilt-b32-finetuned-vqa",
        "image-classification": "google/vit-base-patch16-224",
        "document-qa": "microsoft/layoutlm-base-uncased",
        "video-classification": "MCG-NJU/videomae-base",
    }
    
    # Use default model if not specified
    if model_id is None:
        if task in default_models:
            model_id = default_models[task]
        else:
            model_id = "openai/clip-vit-base-patch32"
            logger.warning(f"No default model for task '{task}', using {model_id}")
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory"]
    if use_power_metrics:
        metrics.append("power")
    
    # Log configuration
    logger.info(f"Benchmarking {model_id} for task '{task}' on {hardware}")
    logger.info(f"Metrics: {metrics}")
    
    # Create a benchmark for the specific model and task
    benchmark = ModelBenchmark(
        model_id=model_id, 
        task=task,
        batch_sizes=[1, 2, 4] if use_gpu else [1, 2],
        sequence_lengths=[32, 64] if use_gpu else [32],
        hardware=hardware,
        metrics=metrics,
        warmup_iterations=2,
        test_iterations=5,
        power_sampling_interval=0.1,  # 100ms sampling if power metrics enabled
        use_flash_attention=use_gpu,  # Flash attention for transformer models
        use_torch_compile=hasattr(torch, "compile")  # PyTorch 2.0+ compilation
    )
    
    # Run the benchmark
    logger.info("Running benchmark...")
    results = benchmark.run()
    
    # Extract model name for filenames
    model_name = model_id.split("/")[-1]
    
    # Export results
    results_file_prefix = f"benchmark_{model_name}_{timestamp}"
    results.export_to_json(output_dir / f"{results_file_prefix}.json")
    results.export_to_markdown(output_dir / f"{results_file_prefix}.md")
    results.export_to_csv(output_dir / f"{results_file_prefix}.csv")
    
    # Display summary
    logger.info(f"Benchmark results for {model_id} on {hardware}:")
    logger.info(f"Mean Latency: {results.get_mean_latency():.4f} ms")
    logger.info(f"Throughput: {results.get_mean_throughput():.2f} items/sec")
    logger.info(f"Peak Memory: {results.get_peak_memory_mb():.2f} MB")
    
    if use_power_metrics:
        logger.info(f"Power Efficiency: {results.get_power_efficiency():.4f} GFLOPs/watt")
    
    logger.info(f"Results saved to {output_dir}/{results_file_prefix}.*")
    
    return results

def benchmark_specific_multimodal_model(model_id, use_hardware_metrics=True, output_dir=None, use_gpu=True):
    """
    Benchmark a specific multimodal model with hardware-aware metrics.
    
    Args:
        model_id: HuggingFace model ID to benchmark
        use_hardware_metrics: Whether to collect hardware-specific metrics
        output_dir: Directory to save results
        use_gpu: Whether to use GPU for benchmarking
    
    Returns:
        ModelBenchmark results
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
    
    # Determine task based on model ID
    task = None
    if "vqa" in model_id.lower():
        task = "visual-question-answering"
    elif "caption" in model_id.lower():
        task = "image-to-text"
    elif "layoutlm" in model_id.lower() or "donut" in model_id.lower():
        task = "document-qa"
    elif "video" in model_id.lower():
        task = "video-classification"
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory"]
    if use_hardware_metrics:
        metrics.extend(["power", "bandwidth"])
    
    # Log configuration
    logger.info(f"Benchmarking {model_id} on {hardware}")
    logger.info(f"Metrics: {metrics}")
    
    # Create a benchmark for the specific model
    benchmark = ModelBenchmark(
        model_id=model_id, 
        task=task,
        batch_sizes=[1, 2, 4] if use_gpu else [1, 2],
        sequence_lengths=[32, 64] if use_gpu else [32],
        hardware=hardware,
        metrics=metrics,
        warmup_iterations=2,
        test_iterations=5,
        power_sampling_interval=0.1,  # 100ms sampling if power metrics enabled
        bandwidth_sampling_interval=0.1,  # 100ms sampling if bandwidth metrics enabled
        use_flash_attention=use_gpu,  # Flash attention for transformer models
        use_torch_compile=hasattr(torch, "compile")  # PyTorch 2.0+ compilation
    )
    
    # Run the benchmark
    logger.info("Running benchmark...")
    results = benchmark.run()
    
    # Extract model name for filenames
    model_name = model_id.split("/")[-1]
    
    # Export results
    results_file_prefix = f"benchmark_{model_name}_{timestamp}"
    results.export_to_json(output_dir / f"{results_file_prefix}.json")
    results.export_to_markdown(output_dir / f"{results_file_prefix}.md")
    results.export_to_csv(output_dir / f"{results_file_prefix}.csv")
    
    # Generate visualizations if possible
    try:
        from visualizers.plots import plot_metric_by_batch_size
        
        # Generate plots for each metric
        for metric in ["latency", "throughput", "memory"]:
            plot_metric_by_batch_size(
                results,
                output_dir / f"{results_file_prefix}_{metric}.png",
                metric=metric,
                title=f"{model_id} - {metric.capitalize()} by Batch Size"
            )
        
        # Generate hardware metrics plots if collected
        if "power" in metrics:
            plot_metric_by_batch_size(
                results,
                output_dir / f"{results_file_prefix}_power.png",
                metric="power_efficiency",
                title=f"{model_id} - Power Efficiency by Batch Size"
            )
        
        if "bandwidth" in metrics:
            plot_metric_by_batch_size(
                results,
                output_dir / f"{results_file_prefix}_bandwidth.png",
                metric="bandwidth_utilization",
                title=f"{model_id} - Bandwidth Utilization by Batch Size"
            )
        
        logger.info(f"Plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    # Display summary
    logger.info(f"Benchmark results for {model_id} on {hardware}:")
    logger.info(f"Mean Latency: {results.get_mean_latency():.4f} ms")
    logger.info(f"Throughput: {results.get_mean_throughput():.2f} items/sec")
    logger.info(f"Peak Memory: {results.get_peak_memory_mb():.2f} MB")
    
    if "power" in metrics:
        logger.info(f"Power Efficiency: {results.get_power_efficiency():.4f} GFLOPs/watt")
    
    if "bandwidth" in metrics:
        logger.info(f"Bandwidth Utilization: {results.get_bandwidth_utilization():.2f} GB/s")
    
    logger.info(f"Results saved to {output_dir}/{results_file_prefix}.*")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark multimodal models with hardware-aware metrics")
    
    # Main options
    parser.add_argument("--mode", choices=["multi", "task", "single"], default="multi",
                        help="Benchmark mode: multiple models, specific task, or single model")
    
    # Model options
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID to benchmark (for single mode)")
    parser.add_argument("--task", type=str, default="image-to-text",
                        help="Task for benchmarking (for task mode)")
    parser.add_argument("--size", choices=["tiny", "small", "base", "large"], default="tiny",
                        help="Size of models to benchmark (for multi mode)")
    
    # Hardware options
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU only, even if GPU is available")
    
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
        benchmark_modern_multimodal_models(
            use_power_metrics=not args.no_power,
            use_bandwidth_metrics=not args.no_bandwidth,
            model_size=args.size,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    elif args.mode == "task":
        # Benchmark specific task
        benchmark_specific_multimodal_task(
            task=args.task,
            model_id=args.model,
            use_power_metrics=not args.no_power,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    elif args.mode == "single":
        # Benchmark specific model
        if args.model is None:
            logger.error("Model ID is required for single mode")
            sys.exit(1)
        
        benchmark_specific_multimodal_model(
            model_id=args.model,
            use_hardware_metrics=not (args.no_power and args.no_bandwidth),
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    
    logger.info("Benchmarking complete!")