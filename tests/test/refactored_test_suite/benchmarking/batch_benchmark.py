#!/usr/bin/env python3
"""
Batch benchmarking script for IPFS Accelerate Python framework.

This script runs benchmarks on multiple models in batch.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"batch_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from benchmarking.run_hardware_benchmark import ModelBenchmark
    from hardware.hardware_detection import detect_available_hardware, get_optimal_device
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please make sure you are running from the refactored_test_suite directory")
    sys.exit(1)

# Try to import DuckDB for database integration
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not installed. Results will not be stored in database.")


class BatchBenchmarker:
    """Batch benchmarking class for running multiple model benchmarks."""
    
    def __init__(self, db_path: Optional[str] = None, output_dir: str = "benchmark_results"):
        """
        Initialize the batch benchmarker.
        
        Args:
            db_path: Path to benchmark database
            output_dir: Directory to save benchmark results
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.db_connection = None
        
        # Available hardware
        self.available_hardware = detect_available_hardware()
        logger.info(f"Available hardware: {', '.join([hw for hw, available in self.available_hardware.items() if available])}")
        
        # Connect to database if path provided and DuckDB is available
        if db_path and DUCKDB_AVAILABLE:
            try:
                self.db_connection = duckdb.connect(db_path)
                logger.info(f"Connected to database at {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.db_connection = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def store_in_database(self, results: Dict[str, Any]) -> bool:
        """
        Store benchmark results in database.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not DUCKDB_AVAILABLE or not self.db_connection:
            logger.warning("DuckDB not available or no database connection. Results not stored.")
            return False
            
        try:
            # Generate run ID
            run_id = f"run_{int(time.time())}_{results['model_id'].replace('/', '_')}_{results['device']}"
            
            # Insert benchmark run
            self.db_connection.execute("""
            INSERT INTO benchmark_runs (
                id, timestamp, model_id, device, precision, architecture_type, task, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, 
                datetime.datetime.now(), 
                results['model_id'], 
                results['device'], 
                results['precision'], 
                results['architecture_type'],
                results['task'],
                f"Benchmark run for {results['model_id']} on {results['device']}"
            ])
            
            # Insert hardware info
            hardware_id = f"hw_{run_id}"
            device_info = results['device_info']
            self.db_connection.execute("""
            INSERT INTO hardware_info (
                id, run_id, device_type, device_name, device_description, performance_tier, hardware_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                hardware_id,
                run_id,
                results['device'],
                device_info['settings'].get('name', results['device']),
                device_info['settings'].get('description', ''),
                device_info['settings'].get('performance_tier', 'medium'),
                json.dumps(device_info['settings'].get('specific_settings', {}))
            ])
            
            # Insert benchmark results for each batch configuration
            for batch_key, batch_result in results.get('batch_results', {}).items():
                if not batch_result.get('success', False):
                    continue
                    
                # Parse batch key to get batch size and sequence length
                # Format: b{batch_size}_s{seq_len}
                parts = batch_key.split("_")
                batch_size = int(parts[0][1:])  # Remove 'b' prefix
                seq_len = int(parts[1][1:])     # Remove 's' prefix
                
                result_id = f"res_{run_id}_{batch_key}"
                
                self.db_connection.execute("""
                INSERT INTO benchmark_results (
                    id, run_id, batch_size, sequence_length, iterations,
                    latency_mean_ms, latency_median_ms, latency_min_ms, latency_max_ms, 
                    latency_std_ms, latency_90p_ms, throughput_samples_per_sec,
                    memory_usage_mb, peak_memory_mb, first_token_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    result_id,
                    run_id,
                    batch_size,
                    seq_len,
                    batch_result.get('iterations', 0),
                    batch_result.get('latency_mean_ms', 0),
                    batch_result.get('latency_median_ms', 0),
                    batch_result.get('latency_min_ms', 0),
                    batch_result.get('latency_max_ms', 0),
                    batch_result.get('latency_std_ms', 0),
                    batch_result.get('latency_90p_ms', 0),
                    batch_result.get('throughput_samples_per_sec', 0),
                    batch_result.get('memory_usage_mb', 0),
                    batch_result.get('peak_memory_mb', 0),
                    batch_result.get('first_token_latency_ms', 0)
                ])
            
            # Insert raw data
            raw_id = f"raw_{run_id}"
            self.db_connection.execute("""
            INSERT INTO raw_benchmark_data (
                id, run_id, raw_data
            ) VALUES (?, ?, ?)
            """, [
                raw_id,
                run_id,
                json.dumps(results)
            ])
            
            logger.info(f"Benchmark results for {results['model_id']} on {results['device']} stored in database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing benchmark results in database: {e}")
            return False
    
    def run_model_benchmark(self, model_id: str, device: str, precision: str,
                           batch_sizes: List[int], sequence_lengths: List[int],
                           iterations: int) -> Dict[str, Any]:
        """
        Run benchmark for a single model on a specific device.
        
        Args:
            model_id: Model ID to benchmark
            device: Device to benchmark on
            precision: Precision mode
            batch_sizes: List of batch sizes
            sequence_lengths: List of sequence lengths
            iterations: Number of iterations
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking {model_id} on {device} with {precision} precision")
        
        # Check if device is available
        if device != "cpu" and not self.available_hardware.get(device, False):
            logger.warning(f"Device {device} not available, skipping")
            return {
                "model_id": model_id,
                "device": device,
                "error": f"Device {device} not available"
            }
        
        try:
            # Create benchmark
            benchmark = ModelBenchmark(model_id, device, precision)
            
            # Run benchmark
            results = benchmark.benchmark_model(
                batch_sizes=batch_sizes,
                sequence_lengths=sequence_lengths,
                iterations=iterations
            )
            
            # Save results to file
            output_path = benchmark.save_results(results, self.output_dir)
            
            # Store in database if available
            if DUCKDB_AVAILABLE and self.db_connection:
                self.store_in_database(results)
                
            logger.info(f"Benchmark for {model_id} on {device} completed")
            logger.info(f"Results saved to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_id} on {device}: {e}")
            return {
                "model_id": model_id,
                "device": device,
                "error": str(e)
            }
    
    def run_batch(self, models: List[str], devices: List[str], 
                 precisions: List[str] = ["float32"],
                 batch_sizes: List[int] = [1, 2, 4, 8],
                 sequence_lengths: List[int] = [128, 512],
                 iterations: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmarks for multiple models on multiple devices.
        
        Args:
            models: List of model IDs to benchmark
            devices: List of devices to benchmark on
            precisions: List of precision modes
            batch_sizes: List of batch sizes
            sequence_lengths: List of sequence lengths
            iterations: Number of iterations
            
        Returns:
            Dictionary mapping (model, device, precision) to benchmark results
        """
        results = {}
        total_benchmarks = len(models) * len(devices) * len(precisions)
        completed = 0
        
        logger.info(f"Running batch benchmark for {len(models)} models on {len(devices)} devices with {len(precisions)} precision modes")
        logger.info(f"Total benchmarks to run: {total_benchmarks}")
        
        # Check if database is available
        if not DUCKDB_AVAILABLE or not self.db_connection:
            logger.warning("DuckDB not available or no database connection. Results will not be stored in database.")
        
        # Run benchmarks
        start_time = time.time()
        
        for model_id, device, precision in itertools.product(models, devices, precisions):
            # Create key for results dictionary
            key = f"{model_id}_{device}_{precision}"
            
            # Run benchmark
            try:
                benchmark_results = self.run_model_benchmark(
                    model_id=model_id,
                    device=device,
                    precision=precision,
                    batch_sizes=batch_sizes,
                    sequence_lengths=sequence_lengths,
                    iterations=iterations
                )
                
                results[key] = benchmark_results
                
            except Exception as e:
                logger.error(f"Error in benchmark for {key}: {e}")
                results[key] = {
                    "model_id": model_id,
                    "device": device,
                    "precision": precision,
                    "error": str(e)
                }
            
            # Update progress
            completed += 1
            elapsed = time.time() - start_time
            remaining = (elapsed / completed) * (total_benchmarks - completed) if completed > 0 else 0
            
            logger.info(f"Progress: {completed}/{total_benchmarks} ({completed/total_benchmarks*100:.1f}%)")
            logger.info(f"Elapsed: {elapsed/60:.1f} minutes, Est. remaining: {remaining/60:.1f} minutes")
        
        total_time = time.time() - start_time
        logger.info(f"Batch benchmark completed in {total_time/60:.1f} minutes")
        
        return results
    
    def load_model_list(self, model_list_file: str) -> List[str]:
        """
        Load list of models from file.
        
        Args:
            model_list_file: Path to model list file
            
        Returns:
            List of model IDs
        """
        try:
            with open(model_list_file, 'r') as f:
                # Read lines and strip whitespace
                models = [line.strip() for line in f.readlines()]
                # Filter empty lines and comments
                models = [model for model in models if model and not model.startswith('#')]
                
            logger.info(f"Loaded {len(models)} models from {model_list_file}")
            return models
            
        except Exception as e:
            logger.error(f"Error loading model list: {e}")
            return []
    
    def generate_report(self, results: Dict[str, Dict[str, Any]], output_file: str):
        """
        Generate batch benchmark report.
        
        Args:
            results: Dictionary mapping (model, device, precision) to benchmark results
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                f.write("# Batch Benchmark Report\n\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summarize benchmarks
                successful = sum(1 for res in results.values() if "error" not in res)
                failed = sum(1 for res in results.values() if "error" in res)
                
                f.write(f"## Summary\n\n")
                f.write(f"Total benchmarks: {len(results)}\n")
                f.write(f"Successful: {successful}\n")
                f.write(f"Failed: {failed}\n\n")
                
                # Group by model
                models = {}
                for key, res in results.items():
                    model_id = res.get("model_id", "unknown")
                    if model_id not in models:
                        models[model_id] = []
                    models[model_id].append(res)
                
                f.write(f"## Models Benchmarked\n\n")
                for model_id, model_results in models.items():
                    successful_model = sum(1 for res in model_results if "error" not in res)
                    f.write(f"- {model_id}: {successful_model}/{len(model_results)} successful\n")
                f.write("\n")
                
                # Hardware comparison table for batch_size=1, sequence_length=128
                f.write(f"## Hardware Comparison (batch_size=1, sequence_length=128)\n\n")
                f.write(f"| Model | Architecture | Device | Precision | Latency (ms) | Throughput (samples/s) | Memory (MB) |\n")
                f.write(f"|-------|--------------|--------|-----------|--------------|------------------------|------------|\n")
                
                for model_id, model_results in models.items():
                    for res in model_results:
                        if "error" in res:
                            continue
                            
                        # Find batch_size=1, sequence_length=128 result
                        batch_key = "b1_s128"
                        if batch_key in res.get("batch_results", {}):
                            batch_result = res["batch_results"][batch_key]
                            if batch_result.get("success", False):
                                arch_type = res.get("architecture_type", "unknown")
                                device = res.get("device", "unknown")
                                precision = res.get("precision", "unknown")
                                latency = batch_result.get("latency_mean_ms", 0)
                                throughput = batch_result.get("throughput_samples_per_sec", 0)
                                memory = batch_result.get("memory_usage_mb", 0)
                                
                                f.write(f"| {model_id} | {arch_type} | {device} | {precision} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                
                f.write("\n")
                
                # Best throughput by model
                f.write(f"## Best Throughput by Model\n\n")
                f.write(f"| Model | Best Device | Batch Size | Seq Length | Throughput (samples/s) |\n")
                f.write(f"|-------|-------------|------------|------------|------------------------|\n")
                
                for model_id, model_results in models.items():
                    best_throughput = 0
                    best_config = None
                    
                    for res in model_results:
                        if "error" in res:
                            continue
                            
                        for batch_key, batch_result in res.get("batch_results", {}).items():
                            if not batch_result.get("success", False):
                                continue
                                
                            throughput = batch_result.get("throughput_samples_per_sec", 0)
                            if throughput > best_throughput:
                                best_throughput = throughput
                                
                                # Parse batch key to get batch size and sequence length
                                parts = batch_key.split("_")
                                batch_size = int(parts[0][1:])  # Remove 'b' prefix
                                seq_len = int(parts[1][1:])     # Remove 's' prefix
                                
                                best_config = {
                                    "device": res.get("device", "unknown"),
                                    "batch_size": batch_size,
                                    "seq_len": seq_len,
                                    "throughput": throughput
                                }
                    
                    if best_config:
                        f.write(f"| {model_id} | {best_config['device']} | {best_config['batch_size']} | {best_config['seq_len']} | {best_config['throughput']:.2f} |\n")
                    else:
                        f.write(f"| {model_id} | No successful benchmarks | | | |\n")
                
                f.write("\n")
                
                # Failed benchmarks
                if failed > 0:
                    f.write(f"## Failed Benchmarks\n\n")
                    f.write(f"| Model | Device | Precision | Error |\n")
                    f.write(f"|-------|--------|-----------|-------|\n")
                    
                    for key, res in results.items():
                        if "error" in res:
                            model_id = res.get("model_id", "unknown")
                            device = res.get("device", "unknown")
                            precision = res.get("precision", "unknown")
                            error = res.get("error", "Unknown error")
                            
                            f.write(f"| {model_id} | {device} | {precision} | {error} |\n")
            
            logger.info(f"Batch benchmark report generated at {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Batch benchmarking for models")
    
    parser.add_argument("--model-list", type=str,
                        help="File containing list of models to benchmark (one per line)")
    
    parser.add_argument("--models", type=str,
                        help="Comma-separated list of models to benchmark")
    
    parser.add_argument("--devices", type=str, default="cpu",
                        help="Comma-separated list of devices to benchmark on")
    
    parser.add_argument("--precisions", type=str, default="float32",
                        help="Comma-separated list of precision modes")
    
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated list of batch sizes")
    
    parser.add_argument("--sequence-lengths", type=str, default="128,512",
                        help="Comma-separated list of sequence lengths")
    
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations per benchmark")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    
    parser.add_argument("--db-path", type=str,
                        help="Path to benchmark database")
    
    parser.add_argument("--report", type=str, default="batch_benchmark_report.md",
                        help="Path to generate benchmark report")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_list is None and args.models is None:
        logger.error("Must provide either --model-list or --models")
        return 1
    
    # Parse arguments
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # Load models from file
        benchmarker = BatchBenchmarker(args.db_path, args.output_dir)
        models = benchmarker.load_model_list(args.model_list)
        if not models:
            logger.error("No models loaded from model list")
            return 1
    
    devices = [d.strip() for d in args.devices.split(",")]
    precisions = [p.strip() for p in args.precisions.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    sequence_lengths = [int(s) for s in args.sequence_lengths.split(",")]
    
    # Create batch benchmarker
    benchmarker = BatchBenchmarker(args.db_path, args.output_dir)
    
    # Run batch benchmarks
    results = benchmarker.run_batch(
        models=models,
        devices=devices,
        precisions=precisions,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        iterations=args.iterations
    )
    
    # Generate report
    benchmarker.generate_report(results, args.report)
    
    logger.info("Batch benchmarking completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())