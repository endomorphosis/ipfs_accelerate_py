#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the enhanced visualization capabilities of the benchmark suite.

This script showcases how to use the enhanced visualization system to create
interactive dashboards and detailed plots that leverage the hardware-aware metrics.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import ModelBenchmark
from visualizers.plots import (
    plot_latency_comparison,
    plot_throughput_scaling,
    plot_memory_usage,
    plot_flops_comparison
)
from visualizers.dashboard import generate_dashboard


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_benchmark():
    """Run a simple benchmark to generate data for visualization."""
    print("Running benchmark to generate data for visualization...")
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../benchmark_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmark on available hardware
    hardware = ["cpu"]
    if torch.cuda.is_available():
        hardware.append("cuda")
    
    # Create and run benchmark
    benchmark = ModelBenchmark(
        model_id="bert-base-uncased",
        batch_sizes=[1, 2, 4, 8],
        sequence_lengths=[16],
        hardware=hardware,
        metrics=["latency", "throughput", "memory", "flops"],
        warmup_iterations=2,
        test_iterations=10,
        output_dir=output_dir
    )
    
    results = benchmark.run()
    print(f"Benchmark completed. Results saved to {output_dir}")
    
    return results


def create_visualizations(results):
    """Create various visualizations from benchmark results."""
    print("Generating visualizations...")
    
    # Define output directory for visualizations
    viz_dir = os.path.join(os.path.dirname(__file__), "../visualization_results")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate standard plots
    print("Creating standard plots...")
    latency_plot = plot_latency_comparison(
        results, 
        output_path=os.path.join(viz_dir, "latency_comparison.png"),
        include_percentiles=True
    )
    
    throughput_plot = plot_throughput_scaling(
        results,
        output_path=os.path.join(viz_dir, "throughput_scaling.png")
    )
    
    memory_plot = plot_memory_usage(
        results,
        output_path=os.path.join(viz_dir, "memory_usage.png"),
        detailed=True
    )
    
    flops_plot = plot_flops_comparison(
        results,
        output_path=os.path.join(viz_dir, "flops_comparison.png"),
        detailed=True
    )
    
    print(f"Standard plots generated in {viz_dir}")
    
    # Generate interactive dashboard
    print("Creating interactive dashboard...")
    dashboard_path = generate_dashboard([results], viz_dir)
    
    if dashboard_path:
        print(f"Interactive dashboard generated at {dashboard_path}")
    else:
        print("Failed to generate dashboard. Make sure dash and plotly are installed.")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    try:
        # Import torch here to handle ImportError gracefully
        global torch
        import torch
        
        # Run benchmark
        results = run_benchmark()
        
        # Create visualizations
        create_visualizations(results)
        
        print("Visualization example completed successfully!")
        
    except ImportError as e:
        print(f"Error: Required package not found - {e}")
        print("Please install required packages with: pip install torch transformers matplotlib plotly dash pandas")
        return
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()