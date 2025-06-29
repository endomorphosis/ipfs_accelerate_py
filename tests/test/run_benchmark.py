#!/usr/bin/env python3
"""
Benchmark script for testing models on different hardware backends with multiple batch sizes.
This script manually fixes the syntax issues in the template files.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_benchmark(model_name="bert-base-uncased", hardware="cpu", batch_sizes=None, warmup=2, runs=5):
    """Run benchmarks for a model on a specific hardware."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    logger.info(f"Benchmarking {model_name} on {hardware}")
    
    # Set device
    if hardware == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA with {torch.cuda.device_count()} devices")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif hardware == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    elif hardware == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU")
    else:
        logger.warning(f"Hardware {hardware} not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Load BERT tokenizer and model
    try:
        from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
        
        # Load tokenizer and model
        logger.info(f"Loading {model_name} tokenizer and model...")
        
        if "bert" in model_name.lower():
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        
        model = model.to(device)
        model.eval()
        
        # Define text for benchmarking
        text = "This is a sample text for benchmarking models on different hardware backends."
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare input (replicate for batch size)
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.repeat(batch_size, 1).to(device) for k, v in encoded.items()}
            
            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(**inputs)
            
            # Measure inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**inputs)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
            
            # Calculate statistics
            latency = np.mean(timings) * 1000  # Convert to ms
            throughput = batch_size / np.mean(timings)  # Items per second
            latency_std = np.std(timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(**inputs)
                memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0
            
            # Record results
            results.append({
                "batch_size": batch_size,
                "latency_ms": latency,
                "latency_std_ms": latency_std,
                "throughput_items_per_sec": throughput,
                "memory_usage_mb": memory_usage,
                "input_shape": {k: v.shape for k, v in inputs.items()},
                "output_shape": {k: v.shape for k, v in outputs.items() if hasattr(v, "shape")}
            })
            
            logger.info(f"Batch size {batch_size}: Latency = {latency:.2f} ms, Throughput = {throughput:.2f} items/sec")
        
        return {
            "model_name": model_name,
            "hardware": hardware,
            "device": str(device),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error benchmarking {model_name} on {hardware}: {e}")
        return {
            "model_name": model_name,
            "hardware": hardware,
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark models on different hardware backends")
    
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Model to benchmark")
    parser.add_argument("--hardware", type=str, nargs="+", default=["cpu"],
                    choices=["cpu", "cuda", "mps", "rocm", "openvino"],
                    help="Hardware backends to test")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="Batch sizes to test")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                    help="Directory to save benchmark results")
    parser.add_argument("--warmup", type=int, default=2,
                    help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5,
                    help="Number of measurement runs")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Detect available hardware
    available_hardware = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "rocm": torch.cuda.is_available() and hasattr(torch.version, "hip"),
        "openvino": "openvino" in sys.modules
    }
    
    logger.info(f"Available hardware: {[hw for hw, available in available_hardware.items() if available]}")
    
    # Filter requested hardware by availability
    hardware_to_test = [hw for hw in args.hardware if available_hardware.get(hw, False)]
    if not hardware_to_test:
        logger.warning("None of the requested hardware is available, falling back to CPU")
        hardware_to_test = ["cpu"]
    
    logger.info(f"Hardware to test: {hardware_to_test}")
    
    # Run benchmarks
    benchmark_results = {}
    for hardware in hardware_to_test:
        benchmark_results[hardware] = run_benchmark(
            model_name=args.model,
            hardware=hardware,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            runs=args.runs
        )
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model.replace("/", "_")
    result_file = output_dir / f"benchmark_{model_name_safe}_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": args.model,
            "hardware_tested": hardware_to_test,
            "batch_sizes": args.batch_sizes,
            "available_hardware": {hw: available for hw, available in available_hardware.items()},
            "results": benchmark_results
        }, f, indent=2)
    
    logger.info(f"Benchmark results saved to: {result_file}")
    
    # Display summary
    logger.info("\n=== Benchmark Summary ===")
    for hardware, result in benchmark_results.items():
        if "error" in result:
            logger.info(f"{hardware}: ERROR - {result['error']}")
            continue
        
        logger.info(f"\n{hardware} results for {args.model}:")
        logger.info("-" * 80)
        logger.info(f"{'Batch Size':<10} {'Latency (ms)':<15} {'Throughput (items/s)':<25} {'Memory (MB)':<15}")
        logger.info("-" * 80)
        
        for batch_result in result.get("results", []):
            batch_size = batch_result.get("batch_size", "N/A")
            latency = batch_result.get("latency_ms", "N/A")
            throughput = batch_result.get("throughput_items_per_sec", "N/A")
            memory = batch_result.get("memory_usage_mb", "N/A")
            
            logger.info(f"{batch_size:<10} {latency:<15.2f} {throughput:<25.2f} {memory:<15.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())