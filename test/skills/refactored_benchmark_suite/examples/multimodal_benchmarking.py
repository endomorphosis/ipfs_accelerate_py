#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for benchmarking multimodal models.

This script demonstrates how to use the refactored benchmark suite
to benchmark different types of multimodal models.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from benchmark import ModelBenchmark, BenchmarkSuite, BenchmarkConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def benchmark_single_multimodal_model():
    """Benchmark a single multimodal model."""
    # Create a benchmark for CLIP model
    benchmark = ModelBenchmark(
        model_id="openai/clip-vit-base-patch32", 
        task="image-to-text",
        batch_sizes=[1, 2, 4],
        sequence_lengths=[32, 64],
        hardware=["cpu"],
        metrics=["latency", "throughput", "memory"],
        warmup_iterations=2,
        test_iterations=10
    )
    
    # Run the benchmark
    results = benchmark.run()
    
    # Export results
    results.export_to_json("clip_benchmark_results.json")
    results.export_to_markdown("clip_benchmark_results.md")
    
    # Display summary
    logger.info("CLIP Model Benchmark Results:")
    logger.info(f"Mean Latency: {results.get_mean_latency():.4f} ms")
    logger.info(f"Throughput: {results.get_mean_throughput():.2f} items/sec")
    logger.info(f"Peak Memory: {results.get_peak_memory_mb():.2f} MB")
    
    return results

def benchmark_multiple_multimodal_models():
    """Benchmark multiple multimodal models for comparison."""
    # Create configuration for benchmark suite
    config = BenchmarkConfig()
    config.models = [
        {"id": "openai/clip-vit-base-patch32", "task": "image-to-text"},
        {"id": "Salesforce/blip-image-captioning-base", "task": "image-to-text"},
    ]
    config.hardware = ["cpu"]
    config.batch_sizes = [1, 2]
    config.sequence_lengths = [32]
    config.metrics = ["latency", "throughput", "memory"]
    config.warmup_iterations = 2
    config.test_iterations = 5
    
    # Create benchmark suite
    suite = BenchmarkSuite(config)
    
    # Run all benchmarks
    results = suite.run()
    
    # Export combined results
    results.export_to_json("multimodal_benchmark_results.json")
    results.export_to_markdown("multimodal_benchmark_results.md")
    
    # Generate comparison plots
    try:
        from visualizers.plots import (
            plot_latency_comparison,
            plot_throughput_comparison,
            plot_memory_comparison
        )
        
        plot_latency_comparison(results, "multimodal_latency_comparison.png")
        plot_throughput_comparison(results, "multimodal_throughput_comparison.png")
        plot_memory_comparison(results, "multimodal_memory_comparison.png")
        
        logger.info("Saved comparison plots")
    except ImportError:
        logger.warning("Visualization dependencies not available. Skipping plot generation.")
    
    return results

def benchmark_vqa_model():
    """Benchmark a visual question answering model."""
    # Create a benchmark for VQA model
    benchmark = ModelBenchmark(
        model_id="dandelin/vilt-b32-finetuned-vqa", 
        task="visual-question-answering",
        batch_sizes=[1, 2],
        sequence_lengths=[32],
        hardware=["cpu"],
        metrics=["latency", "throughput", "memory"],
        warmup_iterations=2,
        test_iterations=5
    )
    
    # Run the benchmark
    results = benchmark.run()
    
    # Export results
    results.export_to_json("vqa_benchmark_results.json")
    
    # Display summary
    logger.info("VQA Model Benchmark Results:")
    logger.info(f"Mean Latency: {results.get_mean_latency():.4f} ms")
    logger.info(f"Throughput: {results.get_mean_throughput():.2f} items/sec")
    logger.info(f"Peak Memory: {results.get_peak_memory_mb():.2f} MB")
    
    return results

def generate_multimodal_dashboard(results_list):
    """Generate an interactive dashboard for multimodal benchmark results."""
    try:
        from visualizers.dashboard import generate_dashboard
        
        # Combine all results
        dashboard_data = []
        for results in results_list:
            dashboard_data.extend(results.to_dict_list())
        
        # Generate dashboard
        generate_dashboard(
            dashboard_data, 
            "multimodal_benchmark_dashboard.html",
            title="Multimodal Models Performance Comparison"
        )
        
        logger.info("Generated interactive dashboard: multimodal_benchmark_dashboard.html")
    except ImportError:
        logger.warning("Dashboard dependencies not available. Skipping dashboard generation.")

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Running on CPU only.")
    
    # Run single model benchmark
    clip_results = benchmark_single_multimodal_model()
    
    # Run multiple model comparison
    multimodal_results = benchmark_multiple_multimodal_models()
    
    # Run VQA model benchmark
    vqa_results = benchmark_vqa_model()
    
    # Generate dashboard
    generate_multimodal_dashboard([clip_results, multimodal_results, vqa_results])