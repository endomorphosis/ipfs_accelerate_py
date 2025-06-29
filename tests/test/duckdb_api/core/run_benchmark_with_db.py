#!/usr/bin/env python3
"""
Benchmark runner with DuckDB integration.

This script provides a tool for running benchmarks on various models and hardware platforms,
with results stored directly in a DuckDB database.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

try:
    import duckdb
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import hardware detection utilities
try:
    from hardware_detection import detect_available_hardware, is_hardware_available
except ImportError:
    logger.warning("Could not import hardware detection utilities. Using simulation mode.")
    
    def detect_available_hardware():
        """Fallback function when hardware detection is not available."""
        return {"cpu": True}  # Only CPU is guaranteed
    
    def is_hardware_available(hardware_type):
        """Fallback function when hardware detection is not available."""
        return hardware_type == "cpu"  # Only CPU is guaranteed

class BenchmarkRunner:
    """
    Benchmark runner with DuckDB integration.
    
    This class runs benchmarks on various models and hardware platforms,
    with results stored directly in a DuckDB database.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark runner.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Initialized BenchmarkRunner with database: {db_path}")
        
        # Ensure database exists and has required tables
        self._ensure_database_setup()
    
    def _get_connection(self):
        """Get a connection to the database."""
        return duckdb.connect(self.db_path)
    
    def _ensure_database_setup(self):
        """Ensure database exists and has required tables."""
        conn = self._get_connection()
        
        try:
            # Check if models table exists
            result = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='models'
            """).fetchall()
            
            if not result:
                logger.warning("Database schema not found. Please run create_benchmark_schema.py first.")
                raise RuntimeError("Database schema not found. Please run create_benchmark_schema.py first.")
        finally:
            conn.close()
    
    def get_or_create_model_id(self, model_name: str) -> int:
        """
        Get model ID from database or create it if it doesn't exist.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model ID
        """
        conn = self._get_connection()
        
        try:
            # Try to get existing model
            result = conn.execute(
                "SELECT model_id FROM models WHERE model_name = ?", 
                [model_name]
            ).fetchone()
            
            if result:
                return result[0]
            
            # Model doesn't exist, create it
            logger.info(f"Adding model to database: {model_name}")
            max_id = conn.execute("SELECT COALESCE(MAX(model_id), 0) FROM models").fetchone()[0]
            next_id = max_id + 1
            
            conn.execute(
                """
                INSERT INTO models (model_id, model_name, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                [next_id, model_name]
            )
            
            return next_id
        finally:
            conn.close()
    
    def get_or_create_hardware_id(self, hardware_type: str) -> int:
        """
        Get hardware ID from database or create it if it doesn't exist.
        
        Args:
            hardware_type: Type of hardware
        
        Returns:
            Hardware ID
        """
        conn = self._get_connection()
        
        try:
            # Try to get existing hardware
            result = conn.execute(
                "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
                [hardware_type]
            ).fetchone()
            
            if result:
                return result[0]
            
            # Hardware doesn't exist, create it
            logger.info(f"Adding hardware to database: {hardware_type}")
            max_id = conn.execute("SELECT COALESCE(MAX(hardware_id), 0) FROM hardware_platforms").fetchone()[0]
            next_id = max_id + 1
            
            conn.execute(
                """
                INSERT INTO hardware_platforms (hardware_id, hardware_type, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                [next_id, hardware_type]
            )
            
            return next_id
        finally:
            conn.close()
    
    def run_single_benchmark(self, model_name: str, hardware_type: str, batch_size: int,
                           sequence_length: int = 128, use_simulation: bool = False) -> Dict[str, Any]:
        """
        Run a single benchmark with the specified configuration.
        
        Args:
            model_name: Name of the model to benchmark
            hardware_type: Type of hardware to use
            batch_size: Batch size for inference
            sequence_length: Sequence length for inference
            use_simulation: Use simulation mode if hardware not available
            
        Returns:
            Dictionary with benchmark results
        """
        # Check if hardware is available
        hardware_available = is_hardware_available(hardware_type)
        using_simulation = False
        
        if not hardware_available and not use_simulation:
            logger.error(f"Hardware '{hardware_type}' is not available and simulation is disabled.")
            return {
                "success": False,
                "error": f"Hardware '{hardware_type}' is not available and simulation is disabled.",
                "is_simulated": False
            }
        elif not hardware_available and use_simulation:
            logger.warning(f"Hardware '{hardware_type}' is not available. Using simulation mode.")
            using_simulation = True
        
        # In a real implementation, this would run the actual benchmark
        # For this example, we'll simulate the benchmark
        
        logger.info(f"Running benchmark for {model_name} on {hardware_type} with batch size {batch_size}")
        
        try:
            # Simulate benchmark execution
            start_time = time.time()
            
            # Simulate different performance based on hardware and model
            if hardware_type == "cpu":
                latency_base = 100.0  # ms
                throughput_base = 10.0  # items/sec
                memory_base = 1000.0  # MB
            elif hardware_type == "cuda":
                latency_base = 20.0  # ms
                throughput_base = 50.0  # items/sec
                memory_base = 2000.0  # MB
            elif hardware_type == "rocm":
                latency_base = 25.0  # ms
                throughput_base = 40.0  # items/sec
                memory_base = 1800.0  # MB
            else:
                latency_base = 50.0  # ms
                throughput_base = 20.0  # items/sec
                memory_base = 1500.0  # MB
            
            # Simulate benchmark duration
            time.sleep(0.5)
            
            # Simulate model size effect
            model_size_factor = 1.0
            if "base" in model_name:
                model_size_factor = 1.5
            elif "large" in model_name:
                model_size_factor = 2.5
            elif "tiny" in model_name:
                model_size_factor = 0.5
            
            # Simulate batch size effect
            batch_factor = batch_size / 4.0
            
            # Add some random variation
            import random
            variation = random.uniform(0.8, 1.2)
            
            # Calculate simulated metrics
            latency = latency_base * model_size_factor * batch_factor * variation
            throughput = throughput_base / model_size_factor * (1.0 / batch_factor) * variation
            memory = memory_base * model_size_factor * (batch_size / 1.0) * variation
            
            # Simulate lower performance for simulated hardware
            if using_simulation:
                latency *= 1.5
                throughput /= 1.5
                memory *= 0.8
            
            # Create result dict
            result = {
                "success": True,
                "model_name": model_name,
                "hardware_type": hardware_type,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "average_latency_ms": latency,
                "throughput_items_per_second": throughput,
                "memory_mb": memory,
                "execution_time_sec": time.time() - start_time,
                "is_simulated": using_simulation,
                "simulation_reason": "Hardware not available" if using_simulation else None
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "is_simulated": using_simulation
            }
    
    def store_benchmark_result(self, result: Dict[str, Any]) -> bool:
        """
        Store benchmark result in the database.
        
        Args:
            result: Dictionary with benchmark results
            
        Returns:
            True if successful, False otherwise
        """
        if not result.get("success", False):
            logger.error(f"Not storing failed benchmark result: {result.get('error', 'Unknown error')}")
            return False
        
        conn = self._get_connection()
        
        try:
            # Get model and hardware IDs
            model_id = self.get_or_create_model_id(result["model_name"])
            hardware_id = self.get_or_create_hardware_id(result["hardware_type"])
            
            # Store result in performance_results table
            conn.execute(
                """
                INSERT INTO performance_results (
                    model_id, hardware_id, batch_size, sequence_length,
                    average_latency_ms, throughput_items_per_second, memory_mb,
                    is_simulated, simulation_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    model_id, hardware_id, result["batch_size"], result["sequence_length"],
                    result["average_latency_ms"], result["throughput_items_per_second"], result["memory_mb"],
                    result["is_simulated"], result["simulation_reason"]
                ]
            )
            
            logger.info(f"Stored benchmark result for {result['model_name']} on {result['hardware_type']}")
            return True
        
        except Exception as e:
            logger.error(f"Error storing benchmark result: {e}")
            logger.error(traceback.format_exc())
            return False
        
        finally:
            conn.close()
    
    def run_benchmarks(self, model_names: List[str], hardware_types: List[str], 
                      batch_sizes: List[int], sequence_length: int = 128,
                      force_simulation: bool = False) -> Dict[str, Any]:
        """
        Run benchmarks for multiple configurations.
        
        Args:
            model_names: List of model names
            hardware_types: List of hardware types
            batch_sizes: List of batch sizes
            sequence_length: Sequence length for inference
            force_simulation: Force simulation mode even if hardware is available
            
        Returns:
            Dictionary with summary of benchmark results
        """
        results = []
        successful = 0
        failed = 0
        
        # Detect available hardware
        available_hardware = detect_available_hardware()
        logger.info(f"Available hardware: {available_hardware}")
        
        # Run benchmarks for all combinations
        for model_name in model_names:
            for hardware_type in hardware_types:
                for batch_size in batch_sizes:
                    # Determine if simulation should be used
                    use_simulation = force_simulation
                    
                    if not force_simulation and hardware_type not in available_hardware:
                        logger.warning(f"Hardware '{hardware_type}' not available. Enabling simulation mode.")
                        use_simulation = True
                    
                    # Run benchmark
                    result = self.run_single_benchmark(
                        model_name=model_name,
                        hardware_type=hardware_type,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        use_simulation=use_simulation
                    )
                    
                    # Store result in database if successful
                    if result.get("success", False):
                        self.store_benchmark_result(result)
                        successful += 1
                    else:
                        failed += 1
                    
                    results.append(result)
        
        # Create summary
        summary = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "started_at": str(datetime.datetime.now()),
            "results": results
        }
        
        return summary

def main():
    """Command-line interface for the benchmark runner with DuckDB integration."""
    parser = argparse.ArgumentParser(description="Benchmark Runner with DuckDB Integration")
    parser.add_argument("--db-path", 
                       help="Path to the DuckDB database (defaults to BENCHMARK_DB_PATH env variable)")
    parser.add_argument("--model", "--models", dest="models", required=True,
                       help="Comma-separated list of model names to benchmark")
    parser.add_argument("--hardware", required=True,
                       help="Comma-separated list of hardware types to benchmark")
    parser.add_argument("--batch-sizes", default="1,4,16",
                       help="Comma-separated list of batch sizes to benchmark")
    parser.add_argument("--sequence-length", type=int, default=128,
                       help="Sequence length for inference")
    parser.add_argument("--force-simulation", action="store_true",
                       help="Force simulation mode even if hardware is available")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--output-json",
                       help="Output file for benchmark results (JSON format)")
    args = parser.parse_args()
    
    # Set up database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Parse comma-separated lists
    models = [model.strip() for model in args.models.split(",")]
    hardware_types = [hw.strip() for hw in args.hardware.split(",")]
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Create runner
    runner = BenchmarkRunner(db_path=db_path, debug=args.debug)
    
    # Run benchmarks
    logger.info(f"Running {len(models)*len(hardware_types)*len(batch_sizes)} benchmarks")
    summary = runner.run_benchmarks(
        model_names=models,
        hardware_types=hardware_types,
        batch_sizes=batch_sizes,
        sequence_length=args.sequence_length,
        force_simulation=args.force_simulation
    )
    
    # Print summary
    logger.info(f"Completed {summary['total']} benchmarks: {summary['successful']} successful, {summary['failed']} failed")
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved benchmark results to {args.output_json}")

if __name__ == "__main__":
    main()