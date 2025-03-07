#!/usr/bin/env python3
"""
Batch Benchmark Runner

This script runs benchmarks for multiple models in parallel.
It uses multiprocessing to run benchmarks for different models concurrently.

Usage:
    python run_batch_benchmarks.py --models "prajjwal1/bert-tiny google/t5-efficient-tiny" --hardware cpu --output-dir benchmark_results
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default models to benchmark
DEFAULT_MODELS = [
    # Text embedding models
    "prajjwal1/bert-tiny",
    "bert-base-uncased",
    # Text generation models
    "google/t5-efficient-tiny",
    "google/t5-small",
    # Vision models
    "google/vit-base-patch16-224",
    "facebook/deit-tiny-patch16-224"
]

# Default hardware platforms
DEFAULT_HARDWARE = ["cpu"]  # Most widely available

# Default batch sizes
DEFAULT_BATCH_SIZES = "1,2,4"

# Database configuration
DB_PATH = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
# By default, we now prefer DuckDB over JSON files
USE_DB_ONLY = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

def run_single_benchmark(model: str, hardware: str, batch_sizes: str, 
                         output_dir: str, verbose: bool = False, 
                         db_path: str = DB_PATH, db_only: bool = USE_DB_ONLY) -> Dict[str, Any]:
    """Run benchmark for a single model on a specific hardware platform."""
    cmd = [
        "python", "run_direct_benchmark.py",
        "--model", model,
        "--hardware", hardware,
        "--batch-sizes", batch_sizes,
        "--output-dir", output_dir,
        "--db-path", db_path
    ]
    
    # Add database storage flags
    if db_only:
        cmd.append("--db-only")
    
    if verbose:
        cmd.append("--verbose")
    
    logger.info(f"Running benchmark: {' '.join(cmd)}")
    
    result = {
        "model": model,
        "hardware": hardware,
        "batch_sizes": batch_sizes,
        "command": " ".join(cmd),
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Run the benchmark command
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Process the output
        result["success"] = True
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        
        # Try to parse benchmark results from the output
        try:
            # Look for JSON files in the output directory that match this model
            model_safe = model.replace("/", "_")
            output_files = list(Path(output_dir).glob(f"direct_benchmark_{model_safe}_{hardware}_*.json"))
            
            if output_files:
                # Get the most recent file
                output_file = max(output_files, key=os.path.getctime)
                
                with open(output_file, "r") as f:
                    benchmark_data = json.load(f)
                
                result["benchmark_data"] = benchmark_data
                result["output_file"] = str(output_file)
                
                # Extract summary metrics for quick access
                if benchmark_data.get("success", False) and "batch_results" in benchmark_data:
                    summary = {}
                    for batch, batch_result in benchmark_data["batch_results"].items():
                        if batch_result.get("success", False):
                            summary[batch] = {
                                "latency_ms": batch_result.get("avg_latency_ms"),
                                "throughput": batch_result.get("throughput_items_per_second"),
                                "memory_mb": batch_result.get("memory_mb")
                            }
                    result["summary"] = summary
        except Exception as e:
            logger.warning(f"Error parsing benchmark results for {model}: {str(e)}")
            result["parse_error"] = str(e)
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark command failed for {model}: {e}")
        result["success"] = False
        result["error"] = f"Command failed with exit code {e.returncode}"
        result["stdout"] = e.stdout
        result["stderr"] = e.stderr
        return result
    except Exception as e:
        logger.error(f"Unexpected error benchmarking {model}: {str(e)}")
        result["success"] = False
        result["error"] = str(e)
        return result

def run_batch_benchmarks(models: List[str], hardware: str, batch_sizes: str, output_dir: str, 
                         max_workers: int = 3, verbose: bool = False,
                         db_path: str = DB_PATH, db_only: bool = USE_DB_ONLY) -> Dict[str, Any]:
    """
    Run benchmarks for multiple models in parallel.
    
    Args:
        models: List of models to benchmark
        hardware: Hardware platform to benchmark on
        batch_sizes: Comma-separated list of batch sizes
        output_dir: Directory to save results
        max_workers: Maximum number of concurrent benchmark processes
        verbose: Print detailed output
        
    Returns:
        Dictionary containing benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results
    results = {
        "models": models,
        "hardware": hardware,
        "batch_sizes": batch_sizes,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # Run benchmarks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all benchmark tasks
        future_to_model = {
            executor.submit(run_single_benchmark, model, hardware, batch_sizes, 
                           output_dir, verbose, db_path, db_only): model
            for model in models
        }
        
        # Process results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results["results"][model] = result
                
                if result.get("success", False):
                    logger.info(f"Benchmark completed for {model}")
                    
                    # Print summary if available
                    if "summary" in result:
                        print(f"\nBenchmark Summary for {model} on {hardware}:")
                        for batch, metrics in result["summary"].items():
                            print(f"  Batch {batch}: "
                                  f"Latency {metrics['latency_ms']:.2f}ms, "
                                  f"Throughput {metrics['throughput']:.2f} items/s, "
                                  f"Memory {metrics['memory_mb']:.2f}MB")
                else:
                    logger.error(f"Benchmark failed for {model}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error processing result for {model}: {str(e)}")
                results["results"][model] = {
                    "model": model,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"batch_benchmarks_{hardware}_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch benchmark results saved to {output_path}")
    
    # Generate summary
    successful = sum(1 for result in results["results"].values() if result.get("success", False))
    failed = len(models) - successful
    
    print(f"\nBatch Benchmark Summary:")
    print(f"Total models: {len(models)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to {output_path}")
    
    return results

def main():
    """Main function for the batch benchmark script."""
    parser = argparse.ArgumentParser(description="Run benchmarks for multiple models in parallel")
    parser.add_argument("--models", type=str, help="Space-separated list of models to benchmark (in quotes)")
    parser.add_argument("--hardware", choices=["cpu", "cuda", "mps"], default="cpu", help="Hardware to use")
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES, help="Comma-separated list of batch sizes to test")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory to save results")
    parser.add_argument("--db-path", help="Path to DuckDB database (defaults to BENCHMARK_DB_PATH env var)")
    parser.add_argument("--db-only", action="store_true", help="Store results only in database, not in JSON")
    parser.add_argument("--no-db", action="store_true", help="Don't store results in database")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of concurrent benchmark processes")
    parser.add_argument("--all-default-models", action="store_true", help="Benchmark all default models")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Determine models to benchmark
    models = []
    if args.all_default_models:
        models = DEFAULT_MODELS
    elif args.models:
        models = args.models.split()
    else:
        logger.error("No models specified. Use --models or --all-default-models")
        return 1
    
    if not models:
        logger.error("No models to benchmark")
        return 1
    
    # Get database path
    db_path = args.db_path or DB_PATH
    
    # Determine whether to use database only (default) or also JSON
    db_only = USE_DB_ONLY
    if args.db_only:
        db_only = True
    elif args.no_db:
        db_only = False
    
    # Run benchmarks
    logger.info(f"Running batch benchmarks for {len(models)} models on {args.hardware}")
    logger.info(f"Models: {', '.join(models)}")
    
    if db_only:
        logger.info("Using database storage only (JSON output deprecated)")
    elif args.no_db:
        logger.info("Using JSON output only (not recommended)")
    
    run_batch_benchmarks(
        models=models,
        hardware=args.hardware,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        verbose=args.verbose,
        db_path=db_path,
        db_only=db_only
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())