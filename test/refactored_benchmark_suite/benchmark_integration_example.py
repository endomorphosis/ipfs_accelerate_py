#!/usr/bin/env python3
"""
Benchmark Integration Example

This script demonstrates how to use the benchmark API client to interact with
the benchmark API server for running and monitoring benchmarks.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the benchmark API client
from benchmark_api_client import BenchmarkAPIClient

def print_progress_bar(progress: float, width: int = 60) -> None:
    """
    Print a progress bar to the console.
    
    Args:
        progress: Progress value (0.0 to 1.0)
        width: Width of the progress bar in characters
    """
    filled_width = int(width * progress)
    empty_width = width - filled_width
    
    bar = '█' * filled_width + '░' * empty_width
    percentage = int(progress * 100)
    
    print(f"\r|{bar}| {percentage}%", end='')
    
    if progress >= 1.0:
        print()

def progress_callback(status: Dict[str, Any]) -> None:
    """
    Callback function for progress updates.
    
    Args:
        status: Status data from the benchmark run
    """
    # Clear the line
    print("\r" + " " * 100, end="\r")
    
    # Print status information
    progress = status.get("progress", 0)
    current_step = status.get("current_step", "")
    completed = status.get("completed_models", 0)
    total = status.get("total_models", 0)
    
    print(f"Step: {current_step}")
    print(f"Models: {completed}/{total}")
    print_progress_bar(progress)
    
    # Print estimated time if available
    elapsed = status.get("elapsed_time", 0)
    remaining = status.get("estimated_remaining_time")
    
    if remaining:
        print(f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
    else:
        print(f"Elapsed: {elapsed:.1f}s")

def run_benchmark_example(api_url: str = "http://localhost:8000") -> None:
    """
    Run a benchmark example using the API client.
    
    Args:
        api_url: URL of the benchmark API server
    """
    client = BenchmarkAPIClient(api_url)
    
    # Check if server is running
    try:
        models = client.get_models()
        print(f"Connected to benchmark server. Found {len(models)} models.")
    except Exception as e:
        print(f"Error connecting to benchmark server: {e}")
        print(f"Please start the server with: ./run_benchmark_api_server.sh")
        return
    
    # Get available hardware
    hardware = client.get_hardware()
    hardware_types = [hw["name"] for hw in hardware]
    print(f"Available hardware: {', '.join(hardware_types)}")
    
    # Start a small benchmark run
    print("Starting benchmark run...")
    run_data = client.start_benchmark(
        priority="medium",
        hardware=["cpu"],
        models=["bert", "gpt2"],  # Limit to just two models for the example
        batch_sizes=[1, 8],
        precision="fp32",
        progressive_mode=True,
        incremental=True
    )
    
    run_id = run_data.get("run_id")
    print(f"Benchmark run started with ID: {run_id}")
    
    # Monitor progress with WebSocket (or polling fallback)
    print("Monitoring progress...")
    client.monitor_progress(run_id, progress_callback)
    
    # Get results
    print("Fetching results...")
    results = client.get_results(run_id)
    
    print(f"Benchmark complete!")
    print("Summary:")
    
    # Print a simple summary of results
    if "results" in results:
        for model, model_results in results["results"].items():
            print(f"\nModel: {model}")
            
            for hw, hw_results in model_results.items():
                print(f"  Hardware: {hw}")
                
                if "metrics" in hw_results:
                    metrics = hw_results["metrics"]
                    print(f"    Throughput: {metrics.get('throughput_items_per_second', 'N/A')} items/s")
                    print(f"    Latency: {metrics.get('average_latency_ms', 'N/A')} ms")
                    print(f"    Memory: {metrics.get('memory_peak_mb', 'N/A')} MB")
    
    # List available reports
    reports = client.get_reports()
    print("\nAvailable reports:")
    for report in reports:
        print(f"  Run ID: {report.get('run_id')}")
        for report_file in report.get("reports", []):
            print(f"    - {report_file}")
    
    print("\nTo visualize the results, start the dashboard:")
    print(f"./run_benchmark_dashboard.sh --api-url {api_url}")

def main():
    """Main entry point when run directly."""
    parser = argparse.ArgumentParser(description="Benchmark Integration Example")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of the benchmark API server")
    args = parser.parse_args()
    
    run_benchmark_example(args.api_url)

if __name__ == "__main__":
    main()