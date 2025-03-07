#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple BERT benchmark script to verify hardware detection and benchmarking
"""

import time
import torch
from transformers import BertModel, BertTokenizer
import json
from datetime import datetime

def benchmark_bert(model_name="prajjwal1/bert-tiny", hardware="cpu", batch_size=1, num_runs=10):
    """
    Run a simple benchmark for BERT model.
    
    Args:
        model_name: Name of the BERT model to benchmark
        hardware: Hardware to use (cpu, cuda)
        batch_size: Batch size to use
        num_runs: Number of runs to average over
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking {model_name} on {hardware} with batch size {batch_size}...")
    
    # Prepare result
    result = {
        "model_name": model_name,
        "hardware": hardware,
        "batch_size": batch_size,
        "timestamp": datetime.now().isoformat(),
        "is_simulated": False
    }
    
    # Load model and tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        # Move to correct device
        if hardware == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"
            if hardware == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available, falling back to CPU (simulated)")
                result["is_simulated"] = True
                result["simulation_reason"] = "CUDA not available"
                hardware = "cpu"
        
        # Create dummy input
        text = "This is a sample input for benchmarking BERT models with PyTorch"
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        # Benchmark
        print("Running benchmark...")
        latencies = []
        
        # Reset CUDA stats if available
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()
        
        # Run benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(**inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate memory usage for CUDA
        if device == "cuda":
            memory_used = (torch.cuda.max_memory_allocated() - memory_before) / (1024 * 1024)  # Convert to MB
        else:
            memory_used = None
        
        # Calculate results
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = 1000 / avg_latency  # items per second
        
        # Store results
        result["latency_ms"] = {
            "mean": avg_latency,
            "min": min_latency,
            "max": max_latency
        }
        result["throughput_items_per_second"] = throughput
        result["memory_mb"] = memory_used
        result["device"] = device
        result["success"] = True
        
        print(f"Results: Avg Latency = {avg_latency:.2f} ms, Throughput = {throughput:.2f} items/s")
        if memory_used:
            print(f"Memory used: {memory_used:.2f} MB")
        
        return result
    except Exception as e:
        print(f"Error benchmarking {model_name} on {hardware}: {e}")
        result["success"] = False
        result["error"] = str(e)
        return result

def main():
    """Main function"""
    # Test CPU
    cpu_result = benchmark_bert(hardware="cpu")
    
    # Test CUDA if available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        cuda_result = benchmark_bert(hardware="cuda")
    else:
        print("CUDA not available, skipping CUDA benchmark")
        cuda_result = {
            "model_name": "prajjwal1/bert-tiny",
            "hardware": "cuda",
            "is_simulated": True,
            "simulation_reason": "CUDA not available",
            "success": False
        }
    
    # Save results
    results = {
        "cpu": cpu_result,
        "cuda": cuda_result,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("bert_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to bert_benchmark_results.json")

if __name__ == "__main__":
    main()