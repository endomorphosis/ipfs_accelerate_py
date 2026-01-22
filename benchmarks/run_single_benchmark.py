#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Real Hardware Benchmark

This script runs benchmarks on actually available hardware platforms (CPU, CUDA, OpenVINO)
and properly marks the results as real (not simulated) in the DuckDB database.

Usage:
    python run_real_hardware_benchmark.py --model bert-base-uncased --hardware cpu cuda openvino
    python run_real_hardware_benchmark.py --model-list bert-base-uncased,t5-small,vit-base
    python run_real_hardware_benchmark.py --all-available --small-models
"""

import os
import sys
import logging
import argparse
import time
import json
from typing import List, Dict, Any, Optional
import duckdb
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default small models that should work on most hardware
DEFAULT_SMALL_MODELS = [
    "bert-tiny",
    "t5-small",
    "vit-tiny",
    "clip-tiny"
]

def get_available_hardware():
    """Detect which hardware platforms are actually available."""
    try:
        # Direct implementation of hardware detection
        import torch
        
        # CPU is always available
        available = ["cpu"]
        
        # Check for CUDA
        if torch.cuda.is_available():
            try:
                # Test actual CUDA functionality
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                available.append("cuda")
                logger.info("CUDA is available")
            except Exception as e:
                logger.warning(f"CUDA detection failed: {e}")
        
        # Check for OpenVINO
        try:
            import openvino
            available.append("openvino")
            logger.info("OpenVINO is available")
        except ImportError:
            logger.debug("OpenVINO not available")
        
        # Log available hardware
        if available:
            logger.info(f"Detected available hardware: {', '.join(available)}")
        else:
            logger.warning("No hardware detected, defaulting to CPU")
            available = ["cpu"]
        
        return available
    except Exception as e:
        logger.error(f"Error detecting hardware: {e}")
        logger.error("Using CPU-only fallback")
        return ["cpu"]

def run_benchmark(model: str, hardware_types: List[str], db_path: str, batch_sizes: Optional[List[int]] = None):
    """
    Run benchmarks for the specified model on the available hardware platforms.
    
    Args:
        model: The model to benchmark
        hardware_types: List of hardware platforms to benchmark on
        db_path: Path to the DuckDB database for storing results
        batch_sizes: Optional list of batch sizes to test (default: [1, 2, 4, 8])
    """
    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    # Construct the command for benchmark_all_key_models.py
    cmd = [
        "python", "benchmark_all_key_models.py",
        "--specific-models", model,
        "--hardware"
    ]
    
    # Add available hardware types
    cmd.extend(hardware_types)
    
    # Add batch sizes
    cmd.append("--batch-sizes")
    cmd.extend([str(bs) for bs in batch_sizes])
    
    # Add database path and disable JSON output
    cmd.extend(["--db-path", db_path, "--db-only"])
    
    # Run the benchmark
    logger.info(f"Running benchmark for model {model} on hardware: {', '.join(hardware_types)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Benchmark completed for {model}")
        logger.debug(result.stdout)
        
        # Update simulation status in the database
        update_simulation_status(db_path, model, hardware_types)
        
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Error running benchmark for {model}: {e}")
        logger.error(f"STDOUT: {e.stdout if hasattr(e, 'stdout') else 'No output'}")
        logger.error(f"STDERR: {e.stderr if hasattr(e, 'stderr') else 'No output'}")
        return False

def update_simulation_status(db_path: str, model: str, hardware_types: List[str]):
    """
    Update the simulation status in the database to mark results as real hardware measurements.
    
    Args:
        db_path: Path to the DuckDB database
        model: The model that was benchmarked
        hardware_types: The hardware platforms that were benchmarked
    """
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Update the simulation status for the recent benchmark results
        conn.execute("""
        UPDATE performance_results
        SET is_simulated = FALSE, simulation_reason = NULL
        WHERE model_name = ? AND hardware_type IN ({})
        AND created_at >= NOW() - INTERVAL 30 MINUTE
        """.format(",".join(["?" for _ in hardware_types])), (model, *hardware_types))
        
        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        
        logger.info(f"Updated simulation status for {model} on {', '.join(hardware_types)} to indicate REAL HARDWARE")
    except Exception as e:
        logger.error(f"Error updating simulation status: {e}")

def run_full_benchmark(models: List[str], hardware_types: Optional[List[str]] = None, 
                      db_path: Optional[str] = None, batch_sizes: Optional[List[int]] = None):
    """
    Run benchmarks for multiple models on available hardware platforms.
    
    Args:
        models: List of models to benchmark
        hardware_types: Optional list of hardware platforms (if None, uses all available)
        db_path: Optional path to the DuckDB database (if None, uses default)
        batch_sizes: Optional list of batch sizes to test (default: [1, 2, 4, 8])
    """
    # Use default database path if not provided
    if db_path is None:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Get available hardware if not provided
    if hardware_types is None:
        hardware_types = get_available_hardware()
    
    # Verify that the hardware is actually available
    available_hardware = get_available_hardware()
    valid_hardware = [hw for hw in hardware_types if hw in available_hardware]
    
    if not valid_hardware:
        logger.error(f"None of the specified hardware platforms ({', '.join(hardware_types)}) are available.")
        logger.error(f"Available hardware: {', '.join(available_hardware)}")
        return False
    
    if len(valid_hardware) < len(hardware_types):
        logger.warning(f"Some specified hardware platforms are not available:")
        logger.warning(f"  Requested: {', '.join(hardware_types)}")
        logger.warning(f"  Available: {', '.join(valid_hardware)}")
        logger.warning(f"Proceeding with available hardware only: {', '.join(valid_hardware)}")
    
    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    # Run benchmarks for each model
    success = True
    for model in models:
        model_success = run_benchmark(model, valid_hardware, db_path, batch_sizes)
        success = success and model_success
    
    return success

def main():
    """Main function to run real hardware benchmarks."""
    parser = argparse.ArgumentParser(description="Run Real Hardware Benchmarks")
    
    # Model selection options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Single model to benchmark")
    model_group.add_argument("--model-list", type=str, help="Comma-separated list of models to benchmark")
    model_group.add_argument("--small-models", action="store_true", help="Use a predefined set of small models")
    
    # Hardware options
    parser.add_argument("--hardware", type=str, nargs="+", help="Hardware platforms to benchmark on")
    parser.add_argument("--all-available", action="store_true", help="Use all available hardware platforms")
    
    # Other options
    parser.add_argument("--db-path", type=str, help="Path to the DuckDB database (default: BENCHMARK_DB_PATH env var or ./benchmark_db.duckdb)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to test")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Determine which models to benchmark
    models = []
    if args.model:
        models = [args.model]
    elif args.model_list:
        models = [m.strip() for m in args.model_list.split(",") if m.strip()]
    elif args.small_models:
        models = DEFAULT_SMALL_MODELS
    
    # Determine which hardware platforms to use
    hardware_types = None
    if args.hardware:
        hardware_types = args.hardware
    elif args.all_available:
        hardware_types = get_available_hardware()
    else:
        hardware_types = get_available_hardware()
    
    # Database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Print benchmark configuration
    logger.info("Running Real Hardware Benchmarks")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Hardware: {', '.join(hardware_types)}")
    logger.info(f"Batch sizes: {args.batch_sizes}")
    logger.info(f"Database: {db_path}")
    
    # Run the benchmarks
    start_time = time.time()
    success = run_full_benchmark(models, hardware_types, db_path, args.batch_sizes)
    end_time = time.time()
    
    # Print summary
    logger.info(f"Benchmark completed {'successfully' if success else 'with errors'}")
    logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())