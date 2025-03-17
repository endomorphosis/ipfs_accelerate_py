#!/usr/bin/env python3
"""
Updated benchmark script for testing models on different hardware backends with multiple batch sizes.
This script adds support for handling different model types properly.
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

def benchmark_bert(model_name, device, batch_sizes, warmup=2, runs=5):
    """Benchmark BERT-type models."""
    try:
        from transformers import BertModel, BertTokenizer
        
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
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
                "input_shape": {k: str(v.shape) for k, v in inputs.items()},
                "output_shape": {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            })
            
            logger.info(f"Batch size {batch_size}: Latency = {latency:.2f} ms, Throughput = {throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking BERT model {model_name}: {e}")
        raise e

def benchmark_t5(model_name, device, batch_sizes, warmup=2, runs=5):
    """Benchmark T5-type models."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        logger.info(f"Loading {model_name} tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Define text for benchmarking
        text = "translate English to German: The house is wonderful."
        
        # Benchmark results
        results = []
        
        # Run benchmarks for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking with batch size {batch_size}")
            
            # Prepare input (replicate for batch size)
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.repeat(batch_size, 1).to(device) for k, v in encoded.items()}
            
            # Create decoder_input_ids
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * model.config.decoder_start_token_id
            
            # Warmup for forward pass
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(**inputs, decoder_input_ids=decoder_input_ids)
            
            # Measure forward pass inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            forward_timings = []
            for _ in range(runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.perf_counter()
                forward_timings.append(end_time - start_time)
            
            # Warmup for generation
            for _ in range(1):  # Fewer warmup runs for generation as it's slower
                with torch.no_grad():
                    _ = model.generate(**inputs, max_length=20, num_beams=1)
            
            # Measure generation inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            generation_timings = []
            for _ in range(2):  # Fewer measurement runs for generation
                start_time = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(**inputs, max_length=20, num_beams=1)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.perf_counter()
                generation_timings.append(end_time - start_time)
            
            # Calculate statistics for forward pass
            forward_latency = np.mean(forward_timings) * 1000  # Convert to ms
            forward_throughput = batch_size / np.mean(forward_timings)  # Items per second
            forward_latency_std = np.std(forward_timings) * 1000  # Convert to ms
            
            # Calculate statistics for generation
            generation_latency = np.mean(generation_timings) * 1000  # Convert to ms
            generation_throughput = batch_size / np.mean(generation_timings)  # Items per second
            generation_latency_std = np.std(generation_timings) * 1000  # Convert to ms
            
            # Get memory usage if CUDA is used
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(**inputs, decoder_input_ids=decoder_input_ids)
                forward_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model.generate(**inputs, max_length=20, num_beams=1)
                generation_memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                forward_memory_usage = 0
                generation_memory_usage = 0
            
            # Record results
            results.append({
                "batch_size": batch_size,
                "forward_latency_ms": forward_latency,
                "forward_latency_std_ms": forward_latency_std,
                "forward_throughput_items_per_sec": forward_throughput,
                "forward_memory_usage_mb": forward_memory_usage,
                "generation_latency_ms": generation_latency,
                "generation_latency_std_ms": generation_latency_std,
                "generation_throughput_items_per_sec": generation_throughput,
                "generation_memory_usage_mb": generation_memory_usage,
                "input_shape": {k: str(v.shape) for k, v in inputs.items()},
                "output_shape": {k: str(v.shape) for k, v in outputs.items() if hasattr(v, "shape")}
            })
            
            logger.info(f"Batch size {batch_size} forward pass: Latency = {forward_latency:.2f} ms, "
                        f"Throughput = {forward_throughput:.2f} items/sec")
            logger.info(f"Batch size {batch_size} generation: Latency = {generation_latency:.2f} ms, "
                        f"Throughput = {generation_throughput:.2f} items/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking T5 model {model_name}: {e}")
        raise e

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
    
    try:
        # Determine model type
        model_type = "bert"  # Default
        
        if "t5" in model_name.lower():
            model_type = "t5"
            results = benchmark_t5(model_name, device, batch_sizes, warmup, runs)
        elif "bert" in model_name.lower():
            model_type = "bert"
            results = benchmark_bert(model_name, device, batch_sizes, warmup, runs)
        else:
            # Default to BERT for other models
            logger.warning(f"Defaulting to BERT benchmark for unknown model type: {model_name}")
            results = benchmark_bert(model_name, device, batch_sizes, warmup, runs)
        
        return {
            "model_name": model_name,
            "model_type": model_type,
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
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
        
        model_type = result.get("model_type", "unknown")
        
        if model_type == "bert":
            logger.info(f"{'Batch Size':<10} {'Latency (ms)':<15} {'Throughput (items/s)':<25} {'Memory (MB)':<15}")
            logger.info("-" * 80)
            
            for batch_result in result.get("results", []):
                batch_size = batch_result.get("batch_size", "N/A")
                latency = batch_result.get("latency_ms", "N/A")
                throughput = batch_result.get("throughput_items_per_sec", "N/A")
                memory = batch_result.get("memory_usage_mb", "N/A")
                
                logger.info(f"{batch_size:<10} {latency:<15.2f} {throughput:<25.2f} {memory:<15.2f}")
        
        elif model_type == "t5":
            logger.info(f"{'Batch Size':<10} {'Forward Latency':<15} {'Forward Throughput':<25} {'Generation Latency':<20} {'Generation Throughput':<25}")
            logger.info("-" * 100)
            
            for batch_result in result.get("results", []):
                batch_size = batch_result.get("batch_size", "N/A")
                forward_latency = batch_result.get("forward_latency_ms", "N/A")
                forward_throughput = batch_result.get("forward_throughput_items_per_sec", "N/A")
                generation_latency = batch_result.get("generation_latency_ms", "N/A")
                generation_throughput = batch_result.get("generation_throughput_items_per_sec", "N/A")
                
                logger.info(f"{batch_size:<10} {forward_latency:<15.2f} {forward_throughput:<25.2f} {generation_latency:<20.2f} {generation_throughput:<25.2f}")
        
        else:
            logger.info(f"Unknown model type: {model_type}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())