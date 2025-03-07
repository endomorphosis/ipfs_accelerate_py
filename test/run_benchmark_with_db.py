#!/usr/bin/env python
"""
Benchmark Runner with Database Storage

This script runs benchmark tests and stores results directly in the DuckDB database
instead of generating JSON files. It serves as an example of how to integrate
the benchmark database with existing test runners.
"""

import os
import sys
import json
import argparse
import logging
import datetime
import time
import random
import uuid
import duckdb
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_runner")

# Database integration - use the standard BenchmarkDBAPI
import os
try:
    from benchmark_db_api import BenchmarkDBAPI
    HAS_DB_INTEGRATION = True
except ImportError:
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")
    HAS_DB_INTEGRATION = False

# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

# Improved hardware detection
try:
    from integrated_improvements.improved_hardware_detection import (
        detect_available_hardware,
        check_web_optimizations,
        HARDWARE_PLATFORMS,
        HAS_CUDA,
        HAS_ROCM,
        HAS_MPS,
        HAS_OPENVINO,
        HAS_WEBNN,
        HAS_WEBGPU
    )
    HAS_HARDWARE_MODULE = True
except ImportError:
    logger.warning("Improved hardware detection not available")
    HAS_HARDWARE_MODULE = False
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_runner")

# Already imported benchmark_db_api above

def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks and store results in database")
    
    parser.add_argument("--db", type=str, default=None, 
                        help="Path to DuckDB database. If not provided, uses BENCHMARK_DB_PATH environment variable or ./benchmark_db.duckdb")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to benchmark")
    parser.add_argument("--hardware", type=str, choices=['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu'],
                        required=True, help="Hardware platform to use")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                        help="Comma-separated list of batch sizes to test")
    parser.add_argument("--test-cases", type=str, default="embedding",
                        help="Comma-separated list of test cases to run")
    parser.add_argument("--precision", type=str, default="fp32",
                        help="Precision to use (fp32, fp16, int8)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations to run for each test")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--device-name", type=str,
                        help="Specific device name (if multiple devices of same type)")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate the benchmark instead of actually running it")
    parser.add_argument("--commit", type=str,
                        help="Git commit hash for this run")
    parser.add_argument("--branch", type=str,
                        help="Git branch for this run")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
    return parser.parse_args()

def store_benchmark_in_database(result, db_path=None):
    """Store benchmark results in database using the BenchmarkDBAPI"""
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot store benchmark")
        return False
    
    try:
        # Create API instance with specified database or use default
        db_api = BenchmarkDBAPI(db_path=db_path)
        
        # Prepare metadata
        metadata = {
            "benchmark_script": os.path.basename(__file__),
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "benchmark_runner"
        }
        # Add any additional information from result
        for key, value in result.items():
            if key not in ["model_name", "hardware_type", "batch_size", "precision", 
                          "throughput_items_per_second", "latency_ms", "memory_mb"]:
                metadata[key] = value
        
        # Create test run
        run_id = db_api.create_test_run(
            test_name=f"benchmark_{result.get('model_name', 'unknown_model')}",
            test_type="performance_benchmark",
            metadata=metadata
        )
        
        # Prepare model data
        model_name = result.get("model_name", "unknown_model")
        model_family = result.get("model_family", None)
        model_metadata = {
            "parameters": result.get("parameters", None),
            "source": result.get("source", "huggingface"),
            "modality": result.get("modality", None)
        }
        
        # Store performance result
        result_id = db_api.store_performance_result(
            model_name=model_name,
            hardware_type=result.get("hardware_type", "unknown"),
            device_name=result.get("device_name", None),
            run_id=run_id,
            test_case=result.get("test_case", "default"),
            batch_size=result.get("batch_size", 1),
            precision=result.get("precision", "fp32"),
            throughput=result.get("throughput_items_per_second", 0.0),
            latency_avg=result.get("latency_ms", 0.0),
            memory_peak=result.get("memory_mb", 0.0),
            total_time_seconds=result.get("total_time_seconds", None),
            iterations=result.get("iterations", None),
            warmup_iterations=result.get("warmup_iterations", None),
            metrics=metadata
        )
        
        # Complete test run
        db_api.complete_test_run(run_id)
        
        logger.info(f"Stored benchmark result in database for {model_name} (result_id: {result_id})")
        return True
    except Exception as e:
        logger.error(f"Error storing benchmark in database: {e}")
        import traceback
        traceback.print_exc()
        return False

def connect_to_db(db_path):
    """Connect to the DuckDB database using BenchmarkDBAPI.
    This function is deprecated and should be replaced with direct BenchmarkDBAPI usage.
    """
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot connect to database")
        return None
    
    try:
        # Create API instance with specified database path
        # The API will handle creating directory and initializing schema
        db_api = BenchmarkDBAPI(db_path=db_path)
        
        # Get a connection from the API
        conn = db_api._get_connection()
        
        logger.info(f"Connected to database at {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_or_create_model(conn, model_name):
    """Find a model in the database or create it if it doesn't exist.
    This function is deprecated and should be replaced with BenchmarkDBAPI._ensure_model_exists
    """
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot find/create model")
        return None, None
    
    try:
        # Create API instance with default database path
        db_api = BenchmarkDBAPI()
        
        # Use the API to find or create the model
        model_id = db_api._ensure_model_exists(conn, model_name)
        
        # Get model family
        model_info = conn.execute("""
        SELECT model_family FROM models WHERE model_id = ?
        """, [model_id]).fetchone()
        
        model_family = model_info[0] if model_info else None
        
        logger.info(f"Found or created model: {model_name} (ID: {model_id}, Family: {model_family})")
        return model_id, model_family
    except Exception as e:
        logger.error(f"Error finding or creating model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_or_create_hardware(conn, hardware_type, device_name=None):
    """Find a hardware platform in the database or create it if it doesn't exist.
    This function is deprecated and should be replaced with BenchmarkDBAPI._ensure_hardware_exists
    """
    if not HAS_DB_INTEGRATION:
        logger.warning("Database integration not available, cannot find/create hardware")
        return None
    
    try:
        # Create API instance with default database path
        db_api = BenchmarkDBAPI()
        
        # Use the API to find or create the hardware platform
        hardware_id = db_api._ensure_hardware_exists(
            conn, 
            hardware_type, 
            device_name=device_name
        )
        
        logger.info(f"Found or created hardware: {hardware_type} {device_name or ''} (ID: {hardware_id})")
        return hardware_id
    except Exception as e:
        logger.error(f"Error finding or creating hardware {hardware_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_test_run(conn, model_name, hardware_type, args):
    """Create a new test run entry in the database"""
    # Generate a test name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_name = f"benchmark_{model_name}_{hardware_type}_{timestamp}"
    
    # Get current time
    now = datetime.datetime.now()
    
    # Get command line
    command_line = f"python {' '.join(sys.argv)}"
    
    # Create metadata JSON
    metadata = {
        'model': model_name,
        'hardware': hardware_type,
        'batch_sizes': args.batch_sizes,
        'test_cases': args.test_cases,
        'precision': args.precision,
        'iterations': args.iterations,
        'warmup': args.warmup
    }
    
    # Get max run_id
    result = conn.execute("SELECT MAX(run_id) FROM test_runs").fetchone()
    max_id = result[0] if result[0] is not None else 0
    run_id = max_id + 1
    
    # Insert the test run
    conn.execute("""
    INSERT INTO test_runs (run_id, test_name, test_type, started_at, completed_at, 
                         execution_time_seconds, success, git_commit, git_branch, 
                         command_line, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [run_id, test_name, 'performance', now, now, 0, True, 
         args.commit, args.branch, command_line, json.dumps(metadata)])
    
    # Get the inserted ID
    run_id = conn.execute("""
    SELECT run_id FROM test_runs WHERE test_name = ? AND started_at = ?
    """, [test_name, now]).fetchone()[0]
    
    logger.info(f"Created new test run: {test_name} (ID: {run_id})")
    
    return run_id, now

def update_test_run_completion(conn, run_id, start_time):
    """Update the test run with completion information"""
    now = datetime.datetime.now()
    execution_time = (now - start_time).total_seconds()
    
    conn.execute("""
    UPDATE test_runs 
    SET completed_at = ?, execution_time_seconds = ?
    WHERE run_id = ?
    """, [now, execution_time, run_id])
    
    logger.info(f"Updated test run completion (ID: {run_id}, Execution time: {execution_time:.2f}s)")

def run_benchmark(model_name, hardware_type, test_case, batch_size, precision, iterations, warmup, simulate=False):
    """Run a benchmark test and return the results"""
    if simulate:
        logger.info(f"Simulating benchmark: {model_name} on {hardware_type}, {test_case}, batch_size={batch_size}")
        
        # Generate synthetic benchmark results
        latency_base = random.uniform(5, 100)  # Base latency in ms
        latency_factor = 1 + (batch_size / 32)  # Batch size effect on latency
        average_latency_ms = latency_base * latency_factor * (1 if precision == 'fp32' else 0.7)
        
        # Throughput is roughly inversely proportional to latency, but with batch effect
        throughput_base = random.uniform(10, 200)
        throughput_items_per_second = throughput_base * (batch_size / latency_factor)
        
        # Memory usage increases with batch size
        memory_base = random.uniform(1000, 5000)  # Base memory in MB
        memory_peak_mb = memory_base * (1 + (batch_size / 16))
        
        # Sleep to simulate benchmark running
        time.sleep(0.5)
        
        return {
            'test_case': test_case,
            'batch_size': batch_size,
            'precision': precision,
            'total_time_seconds': iterations * average_latency_ms / 1000,
            'average_latency_ms': average_latency_ms,
            'throughput_items_per_second': throughput_items_per_second,
            'memory_peak_mb': memory_peak_mb,
            'iterations': iterations,
            'warmup_iterations': warmup
        }
    
    # Actually run the benchmark
    logger.info(f"Running benchmark: {model_name} on {hardware_type}, {test_case}, batch_size={batch_size}")
    
    try:
        # This would normally be your actual benchmark code
        # For this example, we'll simulate it with random values
        
        # Simulate warmup
        logger.info(f"Running {warmup} warmup iterations")
        time.sleep(0.2)
        
        # Run the benchmark
        logger.info(f"Running {iterations} benchmark iterations")
        
        # Initialize result tracking
        latencies = []
        memory_usage = []
        
        # Simulate iterations
        for i in range(iterations):
            # Simulate a single iteration
            iteration_latency = random.uniform(10, 50) * (1 + (batch_size / 32))
            iteration_memory = random.uniform(1000, 3000) * (1 + (batch_size / 16))
            
            latencies.append(iteration_latency)
            memory_usage.append(iteration_memory)
            
            # Sleep a bit to simulate work
            time.sleep(0.01)
        
        # Calculate results
        average_latency_ms = sum(latencies) / len(latencies)
        throughput_items_per_second = (batch_size * 1000) / average_latency_ms
        memory_peak_mb = max(memory_usage)
        total_time_seconds = iterations * average_latency_ms / 1000
        
        return {
            'test_case': test_case,
            'batch_size': batch_size,
            'precision': precision,
            'total_time_seconds': total_time_seconds,
            'average_latency_ms': average_latency_ms,
            'throughput_items_per_second': throughput_items_per_second,
            'memory_peak_mb': memory_peak_mb,
            'iterations': iterations,
            'warmup_iterations': warmup
        }
    
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return None

def add_performance_result(conn, run_id, model_id, hardware_id, result):
    """Add a performance result to the database"""
    # Get max result_id
    result_data = conn.execute("SELECT MAX(result_id) FROM performance_results").fetchone()
    max_id = result_data[0] if result_data[0] is not None else 0
    result_id = max_id + 1
    
    # Insert the performance result
    conn.execute("""
    INSERT INTO performance_results (result_id, run_id, model_id, hardware_id, test_case, batch_size,
                                   precision, total_time_seconds, average_latency_ms,
                                   throughput_items_per_second, memory_peak_mb,
                                   iterations, warmup_iterations, metrics)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [result_id, run_id, model_id, hardware_id, result['test_case'], result['batch_size'],
         result['precision'], result['total_time_seconds'], result['average_latency_ms'],
         result['throughput_items_per_second'], result['memory_peak_mb'],
         result['iterations'], result['warmup_iterations'], json.dumps({})])
    
    logger.info(f"Added performance result (ID: {result_id})")
    logger.info(f"  - Test case: {result['test_case']}, Batch size: {result['batch_size']}")
    logger.info(f"  - Latency: {result['average_latency_ms']:.2f} ms, Throughput: {result['throughput_items_per_second']:.2f} items/s")
    logger.info(f"  - Memory: {result['memory_peak_mb']:.2f} MB")
    
    return result_id

def main():
    args = parse_args()
    conn = None
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Get database path from environment variable if not provided
    db_path = args.db
    if db_path is None:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        logger.info(f"Using database path from environment: {db_path}")
    
    try:
        # Connect to the database
        conn = connect_to_db(db_path)
        
        # Start a transaction for all database operations
        conn.execute("BEGIN TRANSACTION")
        
        # Parse batch sizes and test cases
        batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
        test_cases = [t.strip() for t in args.test_cases.split(',')]
        
        # Find or create model and hardware entries
        model_id, model_family = find_or_create_model(conn, args.model)
        hardware_id = find_or_create_hardware(conn, args.hardware, args.device_name)
        
        # Create test run
        run_id, start_time = create_test_run(conn, args.model, args.hardware, args)
        
        # Run benchmarks for each test case and batch size
        for test_case in test_cases:
            for batch_size in batch_sizes:
                # Run the benchmark
                result = run_benchmark(
                    args.model, args.hardware, test_case, batch_size, 
                    args.precision, args.iterations, args.warmup, args.simulate
                )
                
                if result:
                    # Add the result to the database
                    add_performance_result(conn, run_id, model_id, hardware_id, result)
                else:
                    logger.error(f"Benchmark failed for {test_case}, batch_size={batch_size}")
        
        # Update test run with completion information
        update_test_run_completion(conn, run_id, start_time)
        
        # Commit transaction when all operations succeed
        conn.execute("COMMIT")
        logger.info("All benchmark results saved to database")
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        # Rollback transaction on error
        if conn:
            try:
                conn.execute("ROLLBACK")
                logger.info("Transaction rolled back due to error")
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {rollback_error}")
    finally:
        # Ensure connection is closed properly
        if conn:
            try:
                conn.close()
                logger.debug("Database connection closed")
            except Exception as close_error:
                logger.error(f"Error closing database connection: {close_error}")

if __name__ == "__main__":
    main()