#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for benchmarking speech models with hardware-aware metrics.

This script demonstrates how to use the refactored benchmark suite
to benchmark different types of speech models with hardware-aware metrics.
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

def benchmark_speech_model(model_id, use_power_metrics=True, use_bandwidth_metrics=True, 
                         output_dir=None, use_gpu=True):
    """
    Benchmark a specific speech model with hardware-aware metrics.
    
    Args:
        model_id: HuggingFace model ID to benchmark
        use_power_metrics: Whether to collect power efficiency metrics
        use_bandwidth_metrics: Whether to collect memory bandwidth metrics
        output_dir: Directory to save results
        use_gpu: Whether to use GPU for benchmarking
    
    Returns:
        BenchmarkResults for the model
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
    task = "automatic-speech-recognition"  # Default
    if "classifier" in model_id.lower() or "xvector" in model_id.lower():
        task = "audio-classification"
    elif "tts" in model_id.lower() or "speecht5" in model_id.lower():
        task = "text-to-speech"
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory"]
    if use_power_metrics:
        metrics.append("power")
    if use_bandwidth_metrics:
        metrics.append("bandwidth")
    
    # Log configuration
    logger.info(f"Benchmarking speech model {model_id} on {hardware}")
    logger.info(f"Task: {task}")
    logger.info(f"Metrics: {metrics}")
    
    # Create benchmark
    benchmark = ModelBenchmark(
        model_id=model_id,
        task=task,
        batch_sizes=[1, 2, 4] if use_gpu else [1, 2],
        sequence_lengths=[16000, 32000] if task == "audio-classification" else [32000, 64000],
        hardware=hardware,
        metrics=metrics,
        warmup_iterations=2,
        test_iterations=5,
        power_sampling_interval=0.1,
        bandwidth_sampling_interval=0.1,
        use_flash_attention=use_gpu,
        use_torch_compile=hasattr(torch, "compile"),
        output_dir=str(output_dir)
    )
    
    # Run benchmark
    logger.info("Running benchmark...")
    results = benchmark.run()
    
    # Export results
    model_name = model_id.split("/")[-1]
    results_file_prefix = f"speech_{model_name}_{timestamp}"
    
    results.export_to_json(output_dir / f"{results_file_prefix}.json")
    results.export_to_markdown(output_dir / f"{results_file_prefix}.md")
    results.export_to_csv(output_dir / f"{results_file_prefix}.csv")
    
    # Create visualizations
    try:
        from visualizers.plots import (
            plot_latency_by_batch_size,
            plot_throughput_by_batch_size,
            plot_memory_by_batch_size,
            plot_latency_by_sequence_length,
            plot_power_efficiency,
            plot_bandwidth_roofline
        )
        
        # Latency by batch size
        plot_latency_by_batch_size(
            results,
            output_dir / f"{results_file_prefix}_latency_by_batch.png",
            title=f"{model_id} - Latency by Batch Size"
        )
        
        # Latency by sequence length
        plot_latency_by_sequence_length(
            results,
            output_dir / f"{results_file_prefix}_latency_by_seq.png",
            title=f"{model_id} - Latency by Sequence Length"
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
        if use_gpu and torch.cuda.is_available() and use_power_metrics:
            plot_power_efficiency(
                results,
                output_dir / f"{results_file_prefix}_power.png",
                title=f"{model_id} - Power Efficiency"
            )
        
        # Bandwidth utilization (CUDA only)
        if use_gpu and torch.cuda.is_available() and use_bandwidth_metrics:
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

def compare_speech_architectures(models=None, output_dir=None, use_gpu=True):
    """
    Compare different speech model architectures with hardware-aware metrics.
    
    Args:
        models: List of HuggingFace model IDs to benchmark (default=None, uses predefined models)
        output_dir: Directory to save results
        use_gpu: Whether to use GPU for benchmarking
    
    Returns:
        Dictionary of BenchmarkResults by model
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
    
    # Use predefined models if none provided
    if models is None:
        models = [
            "openai/whisper-tiny",
            "facebook/wav2vec2-base-960h",
            "facebook/hubert-base-ls960",
            "microsoft/wavlm-base",
            "facebook/encodec_24khz",
            "microsoft/speecht5_tts"
        ]
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory", "power", "bandwidth"]
    
    # Run benchmarks for each model
    results = {}
    
    # Create BenchmarkSuite configuration
    models_config = []
    for model_id in models:
        # Determine task based on model ID
        task = "automatic-speech-recognition"  # Default
        if "classifier" in model_id.lower() or "xvector" in model_id.lower():
            task = "audio-classification"
        elif "tts" in model_id.lower() or "speecht5" in model_id.lower():
            task = "text-to-speech"
            
        models_config.append({"id": model_id, "task": task})
    
    # Create and run benchmark suite
    config = BenchmarkConfig()
    config.models = models_config
    config.hardware = hardware
    config.batch_sizes = [1, 2, 4] if use_gpu else [1, 2]
    config.sequence_lengths = [16000, 32000]  # 1-2 seconds at 16kHz
    config.metrics = metrics
    config.warmup_iterations = 2
    config.test_iterations = 5
    config.power_sampling_interval = 0.1
    config.bandwidth_sampling_interval = 0.1
    config.use_flash_attention = use_gpu
    config.use_torch_compile = hasattr(torch, "compile")
    
    # Create and run suite
    logger.info(f"Benchmarking {len(models)} speech models on {hardware}")
    suite = BenchmarkSuite(config)
    suite_results = suite.run()
    
    # Export combined results
    suite_results_prefix = f"speech_architecture_comparison_{timestamp}"
    suite_results.export_to_json(output_dir / f"{suite_results_prefix}.json")
    suite_results.export_to_markdown(output_dir / f"{suite_results_prefix}.md")
    
    # Generate comparison plots
    try:
        logger.info("Generating comparison plots...")
        
        plot_model_comparison(
            suite_results, 
            output_dir / f"{suite_results_prefix}_latency.png",
            metric="latency",
            title="Latency Comparison - Speech Model Architectures"
        )
        
        plot_model_comparison(
            suite_results, 
            output_dir / f"{suite_results_prefix}_throughput.png",
            metric="throughput",
            title="Throughput Comparison - Speech Model Architectures"
        )
        
        plot_model_comparison(
            suite_results, 
            output_dir / f"{suite_results_prefix}_memory.png",
            metric="memory",
            title="Memory Usage Comparison - Speech Model Architectures"
        )
        
        # Plot power metrics
        plot_model_comparison(
            suite_results, 
            output_dir / f"{suite_results_prefix}_power.png",
            metric="power_efficiency",
            title="Power Efficiency Comparison - Speech Model Architectures"
        )
        
        # Plot bandwidth metrics
        plot_model_comparison(
            suite_results, 
            output_dir / f"{suite_results_prefix}_bandwidth.png",
            metric="bandwidth_utilization",
            title="Bandwidth Utilization Comparison - Speech Model Architectures"
        )
        
        logger.info(f"Comparison plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping comparison plots.")
    except Exception as e:
        logger.error(f"Error generating comparison plots: {e}")
    
    return suite_results

def benchmark_specific_speech_task(task="automatic-speech-recognition", output_dir=None, use_gpu=True):
    """
    Benchmark models for a specific speech task.
    
    Args:
        task: Speech task to benchmark
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
    
    # Select models based on task
    if task == "automatic-speech-recognition":
        models = [
            {"id": "openai/whisper-base", "task": task},
            {"id": "facebook/wav2vec2-base-960h", "task": task},
            {"id": "facebook/hubert-base-ls960", "task": task}
        ]
    elif task == "audio-classification":
        models = [
            {"id": "superb/wav2vec2-base-superb-ks", "task": task},
            {"id": "MIT/ast-finetuned-audioset-10-10-0.4593", "task": task}
        ]
    elif task == "text-to-speech":
        models = [
            {"id": "microsoft/speecht5_tts", "task": task},
            {"id": "facebook/fastspeech2-en-ljspeech", "task": task}
        ]
    else:
        logger.error(f"Unknown task: {task}")
        return None
    
    # Determine hardware devices
    hardware = ["cuda"] if use_gpu and torch.cuda.is_available() else ["cpu"]
    
    # Configure metrics
    metrics = ["latency", "throughput", "memory", "power", "bandwidth"]
    
    # Create benchmark suite
    config = BenchmarkConfig()
    config.models = models
    config.hardware = hardware
    config.batch_sizes = [1, 2, 4] if use_gpu else [1, 2]
    config.sequence_lengths = [16000, 32000]  # 1-2 seconds at 16kHz
    config.metrics = metrics
    config.warmup_iterations = 2
    config.test_iterations = 5
    config.power_sampling_interval = 0.1
    config.bandwidth_sampling_interval = 0.1
    config.use_flash_attention = use_gpu
    config.use_torch_compile = hasattr(torch, "compile")
    
    # Create and run suite
    logger.info(f"Benchmarking {len(models)} models for {task} task on {hardware}")
    suite = BenchmarkSuite(config)
    results = suite.run()
    
    # Export results
    results_file_prefix = f"speech_{task.replace('-', '_')}_{timestamp}"
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
            title=f"Latency Comparison - {task.replace('-', ' ').title()} Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_throughput.png",
            metric="throughput",
            title=f"Throughput Comparison - {task.replace('-', ' ').title()} Models"
        )
        
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_memory.png",
            metric="memory",
            title=f"Memory Usage Comparison - {task.replace('-', ' ').title()} Models"
        )
        
        # Plot power metrics if collected
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_power.png",
            metric="power_efficiency",
            title=f"Power Efficiency Comparison - {task.replace('-', ' ').title()} Models"
        )
        
        # Plot bandwidth metrics if collected
        plot_model_comparison(
            results, 
            output_dir / f"{results_file_prefix}_bandwidth.png",
            metric="bandwidth_utilization",
            title=f"Bandwidth Utilization Comparison - {task.replace('-', ' ').title()} Models"
        )
        
        logger.info(f"Plots saved to {output_dir}")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark speech models with hardware-aware metrics")
    
    # Main options
    parser.add_argument("--mode", choices=["single", "compare", "task"], default="single",
                        help="Benchmark mode: single model, compare architectures, or specific task")
    
    # Model options
    parser.add_argument("--model", type=str, default="openai/whisper-base",
                        help="HuggingFace model ID to benchmark (for single mode)")
    parser.add_argument("--models", type=str, nargs="+",
                        help="List of models to compare (for compare mode)")
    parser.add_argument("--task", choices=["automatic-speech-recognition", "audio-classification", "text-to-speech"], 
                        default="automatic-speech-recognition",
                        help="Speech task to benchmark (for task mode)")
    
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
    if args.mode == "single":
        # Benchmark single model
        benchmark_speech_model(
            model_id=args.model,
            use_power_metrics=not args.no_power,
            use_bandwidth_metrics=not args.no_bandwidth,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    elif args.mode == "compare":
        # Compare speech architectures
        compare_speech_architectures(
            models=args.models,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    elif args.mode == "task":
        # Benchmark specific task
        benchmark_specific_speech_task(
            task=args.task,
            output_dir=args.output_dir,
            use_gpu=has_cuda and not args.cpu_only
        )
    
    logger.info("Benchmarking complete!")