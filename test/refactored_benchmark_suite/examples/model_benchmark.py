#!/usr/bin/env python3
"""
Example Model Benchmark Implementation

This example demonstrates how to implement a model benchmark using the unified framework.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from benchmark_core import BenchmarkBase, BenchmarkRegistry

logger = logging.getLogger(__name__)

@BenchmarkRegistry.register(
    name="model_inference",
    category="inference",
    models=["bert", "vit", "whisper"],
    hardware=["cpu", "cuda", "mps"]
)
class ModelInferenceBenchmark(BenchmarkBase):
    """Benchmark for model inference performance."""
    
    def setup(self):
        """Set up benchmark environment."""
        self.logger.info("Setting up model inference benchmark")
        
        # Extract configuration parameters
        self.model_name = self.config.get("model", "bert-base-uncased")
        self.batch_sizes = self.config.get("batch_sizes", [1, 2, 4, 8])
        self.warmup_runs = self.config.get("warmup_runs", 2)
        self.measurement_runs = self.config.get("measurement_runs", 5)
        
        try:
            # Import required libraries
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            self.torch = torch
            
            # Detect device
            if self.hardware.name == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif self.hardware.name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                
            self.logger.info(f"Using device: {self.device}")
                
            # Load tokenizer and model
            self.logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Prepare sample text for benchmarking
            self.sample_text = "This is a sample text for benchmarking the model inference performance."
            
            return True
            
        except ImportError as e:
            self.logger.error(f"Required libraries not available: {e}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error setting up benchmark: {e}")
            raise
            
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing model inference benchmark")
        
        results = {}
        
        for batch_size in self.batch_sizes:
            self.logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Prepare input (replicate for batch size)
            encoded = self.tokenizer(self.sample_text, return_tensors="pt", padding=True)
            inputs = {k: v.repeat(batch_size, 1).to(self.device) for k, v in encoded.items()}
            
            # Warmup runs
            for _ in range(self.warmup_runs):
                with self.torch.no_grad():
                    _ = self.model(**inputs)
                    
            # Synchronize if using CUDA
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
                
            # Measurement runs
            timings = []
            for _ in range(self.measurement_runs):
                start_time = time.perf_counter()
                
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Synchronize if using CUDA
                if self.device.type == "cuda":
                    self.torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
                
            # Get memory usage if CUDA is used
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
                self.torch.cuda.reset_peak_memory_stats()
                
                with self.torch.no_grad():
                    _ = self.model(**inputs)
                    
                memory_usage = self.torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_usage = 0
                
            # Store batch results
            batch_results = {
                "latency_ms": np.mean(timings) * 1000,  # Convert to ms
                "latency_std_ms": np.std(timings) * 1000,  # Convert to ms
                "throughput_items_per_sec": batch_size / np.mean(timings),
                "memory_usage_mb": memory_usage,
                "input_shape": {k: v.shape for k, v in inputs.items()},
                "output_shape": {k: v.shape for k, v in outputs.items() if hasattr(v, "shape")}
            }
            
            results[f"batch_size_{batch_size}"] = batch_results
            
        return results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing benchmark results")
        
        # Calculate aggregate metrics
        batch_sizes = []
        latencies = []
        throughputs = []
        
        for batch_key, batch_data in raw_results.items():
            batch_size = int(batch_key.split("_")[-1])
            batch_sizes.append(batch_size)
            latencies.append(batch_data["latency_ms"])
            throughputs.append(batch_data["throughput_items_per_sec"])
            
        # Find optimal batch size (highest throughput)
        optimal_idx = throughputs.index(max(throughputs))
        optimal_batch_size = batch_sizes[optimal_idx]
        
        # Create processed results
        processed_results = {
            "success": True,
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_results": raw_results,
            "metrics": {
                "min_latency_ms": min(latencies),
                "max_throughput_items_per_sec": max(throughputs),
                "optimal_batch_size": optimal_batch_size,
                "latency_at_batch_1_ms": raw_results.get("batch_size_1", {}).get("latency_ms", 0)
            },
            "hardware": self.hardware.get_info()
        }
        
        return processed_results
        
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Cleaning up resources")
        
        if hasattr(self, "model"):
            del self.model
            
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            
        if hasattr(self, "torch") and hasattr(self.torch, "cuda"):
            self.torch.cuda.empty_cache()


@BenchmarkRegistry.register(
    name="model_throughput",
    category="throughput",
    models=["bert", "vit", "whisper"],
    hardware=["cpu", "cuda", "mps"]
)
class ModelThroughputBenchmark(ModelInferenceBenchmark):
    """Benchmark for model throughput optimization."""
    
    def setup(self):
        """Set up benchmark environment."""
        # Use base class setup
        super().setup()
        
        # Additional setup for throughput benchmark
        self.concurrent_models = self.config.get("concurrent_models", 2)
        self.logger.info(f"Throughput benchmark with {self.concurrent_models} concurrent models")
        
        return True
        
    def execute(self):
        """Execute the benchmark."""
        self.logger.info("Executing model throughput benchmark")
        
        # Get base inference results
        inference_results = super().execute()
        
        # Additional throughput-specific measurements
        throughput_results = {
            "concurrent_models": self.concurrent_models,
            "inference_results": inference_results
        }
        
        # Simulate concurrent execution
        if self.concurrent_models > 1:
            # For demonstration, we'll just extrapolate from single-model results
            # In a real implementation, this would actually run concurrent models
            
            for batch_size in self.batch_sizes:
                batch_key = f"batch_size_{batch_size}"
                if batch_key in inference_results:
                    base_throughput = inference_results[batch_key]["throughput_items_per_sec"]
                    
                    # Simulate throughput scaling with concurrent models
                    # In reality, this is affected by hardware contention
                    scaling_factor = 0.8  # 80% scaling efficiency
                    concurrent_throughput = base_throughput * self.concurrent_models * scaling_factor
                    
                    throughput_results[f"concurrent_{batch_key}"] = {
                        "throughput_items_per_sec": concurrent_throughput,
                        "scaling_efficiency": scaling_factor
                    }
                    
        return throughput_results
        
    def process_results(self, raw_results):
        """Process raw benchmark results."""
        self.logger.info("Processing throughput benchmark results")
        
        # Process base inference results
        inference_results = raw_results.get("inference_results", {})
        base_processed = super().process_results(inference_results)
        
        # Extract throughput metrics
        concurrent_metrics = {}
        
        for key, value in raw_results.items():
            if key.startswith("concurrent_batch_size_"):
                batch_size = int(key.split("_")[-1])
                concurrent_metrics[f"concurrent_batch_{batch_size}"] = value
                
        # Calculate peak throughput across all configurations
        peak_concurrent_throughput = 0
        optimal_config = {}
        
        for config, metrics in concurrent_metrics.items():
            throughput = metrics.get("throughput_items_per_sec", 0)
            if throughput > peak_concurrent_throughput:
                peak_concurrent_throughput = throughput
                optimal_config = {
                    "config": config,
                    "throughput": throughput,
                    "scaling_efficiency": metrics.get("scaling_efficiency", 0)
                }
                
        # Create processed results
        processed_results = {
            "success": True,
            "model_name": self.model_name,
            "device": str(self.device),
            "concurrent_models": raw_results.get("concurrent_models", 1),
            "base_metrics": base_processed.get("metrics", {}),
            "metrics": {
                "peak_concurrent_throughput": peak_concurrent_throughput,
                "base_throughput": base_processed.get("metrics", {}).get("max_throughput_items_per_sec", 0),
                "scaling_efficiency": optimal_config.get("scaling_efficiency", 0),
                "optimal_configuration": optimal_config.get("config", "")
            },
            "hardware": self.hardware.get_info(),
            "throughput_results": concurrent_metrics
        }
        
        return processed_results


def main():
    """Example usage of the model benchmarks."""
    import sys
    import json
    from benchmark_core import BenchmarkRunner
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create runner
    runner = BenchmarkRunner(config={
        "output_dir": "./benchmark_results"
    })
    
    # Print available benchmarks
    print("Available benchmarks:")
    for name, metadata in BenchmarkRegistry.list_benchmarks().items():
        print(f"  - {name}: {metadata}")
        
    # Run model inference benchmark
    print("\nRunning model inference benchmark...")
    try:
        result = runner.execute("model_inference", {
            "hardware": "cpu",
            "model": "bert-base-uncased",
            "batch_sizes": [1, 2, 4],
            "warmup_runs": 1,
            "measurement_runs": 2
        })
        
        print("\nInference Results Summary:")
        print(f"Model: {result.get('model_name', 'unknown')}")
        print(f"Device: {result.get('device', 'unknown')}")
        print("\nMetrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  - {key}: {value}")
            
    except Exception as e:
        print(f"Error running inference benchmark: {e}")
        
    # Run model throughput benchmark
    print("\nRunning model throughput benchmark...")
    try:
        result = runner.execute("model_throughput", {
            "hardware": "cpu",
            "model": "bert-base-uncased",
            "batch_sizes": [1, 2],
            "concurrent_models": 2,
            "warmup_runs": 1,
            "measurement_runs": 2
        })
        
        print("\nThroughput Results Summary:")
        print(f"Model: {result.get('model_name', 'unknown')}")
        print(f"Device: {result.get('device', 'unknown')}")
        print(f"Concurrent Models: {result.get('concurrent_models', 0)}")
        print("\nMetrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  - {key}: {value}")
            
    except Exception as e:
        print(f"Error running throughput benchmark: {e}")
        
    # Save results and generate report
    results_path = runner.save_results()
    report_path = runner.generate_report()
    
    print(f"\nResults saved to: {results_path}")
    print(f"Report generated at: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())