#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple examples of how to use the refactored benchmark suite.

This script demonstrates common usage patterns for the benchmark suite.
"""

import os
from refactored_benchmark_suite import ModelBenchmark, BenchmarkSuite

def example_single_model():
    """Benchmark a single model with default settings."""
    # Initialize with a model ID
    benchmark = ModelBenchmark("bert-base-uncased")
    
    # Run the benchmark
    results = benchmark.run()
    
    # Export results
    results.export_to_json()
    
    print(f"Single model benchmark complete. Results saved to {results.config.output_dir}")

def example_custom_parameters():
    """Benchmark a model with custom parameters."""
    # Initialize with custom parameters
    benchmark = ModelBenchmark(
        model_id="gpt2",
        batch_sizes=[1, 4],
        sequence_lengths=[32, 128],
        hardware=["cpu"],  # Use CPU only
        metrics=["latency", "throughput", "memory"],
        warmup_iterations=3,
        test_iterations=10,
        output_dir="custom_results"
    )
    
    # Run the benchmark
    results = benchmark.run()
    
    # Export in multiple formats
    results.export_to_json()
    results.export_to_csv()
    results.export_to_markdown()
    
    # Generate plots
    results.plot_latency_comparison()
    results.plot_throughput_scaling()
    results.plot_memory_usage()
    
    print(f"Custom parameters benchmark complete. Results saved to {results.config.output_dir}")

def example_benchmark_suite():
    """Run a benchmark suite on multiple models."""
    # Create a custom suite
    suite = BenchmarkSuite(
        models=["bert-base-uncased", "gpt2", "t5-small"],
        batch_sizes=[1],
        sequence_lengths=[32],
        hardware=["cpu"],
        output_dir="suite_results"
    )
    
    # Run the suite
    suite_results = suite.run()
    
    # Process results
    for model_id, results in suite_results.items():
        results.export_to_json()
    
    print(f"Benchmark suite complete. Results saved to suite_results/")

def example_predefined_suite():
    """Run a predefined benchmark suite."""
    # Create a predefined suite
    suite = BenchmarkSuite.from_predefined_suite(
        "text-classification",
        hardware=["cpu"],
        output_dir="predefined_suite_results"
    )
    
    # Run the suite
    suite_results = suite.run()
    
    # Process results
    for model_id, results in suite_results.items():
        results.export_to_json()
    
    print(f"Predefined suite benchmark complete. Results saved to predefined_suite_results/")

def main():
    """Run all examples."""
    # Create output directories
    os.makedirs("benchmark_results", exist_ok=True)
    os.makedirs("custom_results", exist_ok=True)
    os.makedirs("suite_results", exist_ok=True)
    os.makedirs("predefined_suite_results", exist_ok=True)
    
    print("Running benchmark examples...\n")
    
    print("\n1. Single Model Example")
    print("-----------------------")
    example_single_model()
    
    print("\n2. Custom Parameters Example")
    print("---------------------------")
    example_custom_parameters()
    
    print("\n3. Benchmark Suite Example")
    print("-------------------------")
    example_benchmark_suite()
    
    print("\n4. Predefined Suite Example")
    print("--------------------------")
    example_predefined_suite()
    
    print("\nAll examples complete.")

if __name__ == "__main__":
    main()