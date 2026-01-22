#!/usr/bin/env python3
"""
Comprehensive benchmark script for testing multiple models on different hardware backends.
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
import run_benchmark_updated as benchmark_runner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model sets
MODEL_SETS = {
    "text_embedding": [
        "prajjwal1/bert-tiny",
        "bert-base-uncased",
    ],
    "text_generation": [
        "google/t5-efficient-tiny",
        "t5-small",
    ],
    "all": [
        "prajjwal1/bert-tiny",
        "google/t5-efficient-tiny",
    ]
}

# Define hardware sets
HARDWARE_SETS = {
    "local": ["cpu"],
    "gpu": ["cuda"],
    "intel": ["cpu", "openvino"],
    "all": ["cpu", "cuda", "openvino"],
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    
    # Model selection
    parser.add_argument("--model-set", type=str, default="all",
                    choices=list(MODEL_SETS.keys()),
                    help="Set of models to benchmark")
    parser.add_argument("--models", type=str, nargs="+",
                    help="Specific models to benchmark (overrides model-set)")
    
    # Hardware selection
    parser.add_argument("--hardware-set", type=str, default="local",
                    choices=list(HARDWARE_SETS.keys()),
                    help="Set of hardware backends to test")
    parser.add_argument("--hardware", type=str, nargs="+",
                    help="Specific hardware backends to test (overrides hardware-set)")
    
    # Batch sizes
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="Batch sizes to test")
    
    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=2,
                    help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5,
                    help="Number of measurement runs")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                    help="Directory to save benchmark results")
    
    # Output format
    parser.add_argument("--format", type=str, default="markdown",
                    choices=["markdown", "json", "csv"],
                    help="Output format for the report")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                    help="Enable debug logging")
    
    return parser.parse_args()

def run_benchmark_for_model(model_name, hardware_backends, batch_sizes, warmup, runs, output_dir):
    """Run benchmark for a specific model across hardware backends."""
    logger.info(f"Benchmarking {model_name} across {len(hardware_backends)} hardware backends")
    
    # Results container
    results = {
        "model_name": model_name,
        "hardware_results": {},
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Run benchmarks for each hardware backend
    for hardware in hardware_backends:
        try:
            logger.info(f"Running benchmark for {model_name} on {hardware}")
            
            # Run the benchmark
            result = benchmark_runner.run_benchmark(
                model_name=model_name,
                hardware=hardware,
                batch_sizes=batch_sizes,
                warmup=warmup,
                runs=runs
            )
            
            # Store the result
            results["hardware_results"][hardware] = result
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name} on {hardware}: {e}")
            results["hardware_results"][hardware] = {
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Save the results
    model_name_safe = model_name.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"benchmark_{model_name_safe}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results for {model_name} saved to: {result_file}")
    
    return results

def generate_markdown_report(all_results, output_dir):
    """Generate a comprehensive markdown report."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"benchmark_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        # Header
        f.write("# Comprehensive Model Benchmark Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [Hardware Comparison](#hardware-comparison)\n")
        for model_name in all_results.keys():
            model_name_anchor = model_name.replace("/", "_").lower()
            f.write(f"3. [{model_name}](#{model_name_anchor})\n")
        f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("| Model | Hardware | Batch Size | Latency (ms) | Throughput (items/s) |\n")
        f.write("|-------|----------|------------|--------------|---------------------|\n")
        
        for model_name, model_results in all_results.items():
            for hardware, hw_result in model_results["hardware_results"].items():
                if "error" in hw_result:
                    continue
                
                model_type = hw_result.get("model_type", "unknown")
                
                if model_type == "bert":
                    # Get the result for batch size 1 (or the smallest batch size)
                    results = hw_result.get("results", [])
                    if results:
                        batch_result = sorted(results, key=lambda x: x.get("batch_size", 0))[0]
                        batch_size = batch_result.get("batch_size", "N/A")
                        latency = batch_result.get("latency_ms", "N/A")
                        throughput = batch_result.get("throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {model_name} | {hardware} | {batch_size} | {latency:.2f} | {throughput:.2f} |\n")
                
                elif model_type == "t5":
                    # Get the result for batch size 1 (or the smallest batch size)
                    results = hw_result.get("results", [])
                    if results:
                        batch_result = sorted(results, key=lambda x: x.get("batch_size", 0))[0]
                        batch_size = batch_result.get("batch_size", "N/A")
                        forward_latency = batch_result.get("forward_latency_ms", "N/A")
                        forward_throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {model_name} | {hardware} | {batch_size} | {forward_latency:.2f} | {forward_throughput:.2f} |\n")
        
        f.write("\n")
        
        # Hardware Comparison
        f.write("## Hardware Comparison\n\n")
        f.write("### Throughput Comparison\n\n")
        
        # Get unique hardware backends and models
        hardware_backends = set()
        for model_results in all_results.values():
            for hardware in model_results["hardware_results"].keys():
                hardware_backends.add(hardware)
        
        hardware_backends = sorted(list(hardware_backends))
        
        # Create throughput comparison table
        f.write("| Model | Batch Size |")
        for hardware in hardware_backends:
            f.write(f" {hardware} (items/s) |")
        f.write("\n")
        
        f.write("|-------|------------|")
        for _ in hardware_backends:
            f.write("--------------|")
        f.write("\n")
        
        # Batch sizes to include in the comparison
        batch_sizes_to_show = [1, 4, 16]
        
        for model_name, model_results in all_results.items():
            for batch_size in batch_sizes_to_show:
                f.write(f"| {model_name} | {batch_size} |")
                
                for hardware in hardware_backends:
                    hw_result = model_results["hardware_results"].get(hardware, {"error": "Not tested"})
                    
                    if "error" in hw_result:
                        f.write(" N/A |")
                        continue
                    
                    model_type = hw_result.get("model_type", "unknown")
                    results = hw_result.get("results", [])
                    
                    # Find the result for this batch size
                    batch_result = next((r for r in results if r.get("batch_size") == batch_size), None)
                    
                    if batch_result:
                        if model_type == "bert":
                            throughput = batch_result.get("throughput_items_per_sec", "N/A")
                            f.write(f" {throughput:.2f} |")
                        elif model_type == "t5":
                            throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                            f.write(f" {throughput:.2f} |")
                        else:
                            f.write(" N/A |")
                    else:
                        f.write(" N/A |")
                
                f.write("\n")
        
        f.write("\n")
        
        # Detailed results for each model
        for model_name, model_results in all_results.items():
            model_name_anchor = model_name.replace("/", "_").lower()
            f.write(f"## {model_name}\n\n")
            
            for hardware, hw_result in model_results["hardware_results"].items():
                f.write(f"### {hardware}\n\n")
                
                if "error" in hw_result:
                    f.write(f"Error: {hw_result['error']}\n\n")
                    continue
                
                model_type = hw_result.get("model_type", "unknown")
                
                if model_type == "bert":
                    f.write("| Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) |\n")
                    f.write("|------------|--------------|---------------------|------------|\n")
                    
                    for batch_result in hw_result.get("results", []):
                        batch_size = batch_result.get("batch_size", "N/A")
                        latency = batch_result.get("latency_ms", "N/A")
                        throughput = batch_result.get("throughput_items_per_sec", "N/A")
                        memory = batch_result.get("memory_usage_mb", "N/A")
                        
                        f.write(f"| {batch_size} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                
                elif model_type == "t5":
                    f.write("| Batch Size | Forward Latency (ms) | Forward Throughput (items/s) | Generation Latency (ms) | Generation Throughput (items/s) |\n")
                    f.write("|------------|----------------------|----------------------------|-------------------------|-------------------------------|\n")
                    
                    for batch_result in hw_result.get("results", []):
                        batch_size = batch_result.get("batch_size", "N/A")
                        forward_latency = batch_result.get("forward_latency_ms", "N/A")
                        forward_throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                        generation_latency = batch_result.get("generation_latency_ms", "N/A")
                        generation_throughput = batch_result.get("generation_throughput_items_per_sec", "N/A")
                        
                        f.write(f"| {batch_size} | {forward_latency:.2f} | {forward_throughput:.2f} | {generation_latency:.2f} | {generation_throughput:.2f} |\n")
                
                else:
                    f.write(f"Unknown model type: {model_type}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    logger.info(f"Markdown report saved to: {report_file}")
    return report_file

def main():
    """Main function."""
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine models to benchmark
    if args.models:
        models_to_benchmark = args.models
    else:
        models_to_benchmark = MODEL_SETS[args.model_set]
    
    # Determine hardware backends to test
    if args.hardware:
        hardware_to_test = args.hardware
    else:
        hardware_to_test = HARDWARE_SETS[args.hardware_set]
    
    logger.info(f"Models to benchmark: {models_to_benchmark}")
    logger.info(f"Hardware backends to test: {hardware_to_test}")
    
    # Run benchmarks for each model
    all_results = {}
    for model_name in models_to_benchmark:
        results = run_benchmark_for_model(
            model_name=model_name,
            hardware_backends=hardware_to_test,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            runs=args.runs,
            output_dir=output_dir
        )
        all_results[model_name] = results
    
    # Generate report
    if args.format == "markdown":
        report_file = generate_markdown_report(all_results, output_dir)
    elif args.format == "json":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"JSON report saved to: {report_file}")
    elif args.format == "csv":
        # TODO: Implement CSV report generation
        logger.warning("CSV report generation not implemented yet")
        report_file = None
    
    logger.info("Benchmark completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())