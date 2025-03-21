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
    
    return parser.parse_args()

def connect_to_db(db_path):
    """Connect to the DuckDB database"""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Try to connect with appropriate parameters - handle different DuckDB versions
        try:
            conn = duckdb.connect(db_path)
        except TypeError:
            # Fallback if parameters are different
            conn = duckdb.connect(database=db_path)
        
        # Check if required tables exist, if not try to create them
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0].lower() for t in tables]
        
        required_tables = ['hardware_platforms', 'models', 'test_runs', 'performance_results']
        missing_tables = [t for t in required_tables if t.lower() not in table_names]
        
        if missing_tables:
            logger.warning(f"Required tables missing from database: {', '.join(missing_tables)}")
            
            # Check multiple possible locations for the schema script
            schema_script = None
            possible_paths = [
                "scripts/create_benchmark_schema.py",
                "test/scripts/create_benchmark_schema.py",
                str(Path(__file__).parent / "scripts" / "create_benchmark_schema.py"),
                str(Path(__file__).parent / "scripts" / "benchmark_db" / "create_benchmark_schema.py"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "create_benchmark_schema.py")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    schema_script = path
                    logger.info(f"Found schema script at: {path}")
                    break
            
            if schema_script:
                logger.info(f"Creating schema using script: {schema_script}")
                try:
                    conn.close()  # Close connection before running script
                    import subprocess
                    subprocess.run([sys.executable, schema_script, "--output", db_path])
                    conn = duckdb.connect(db_path, read_only=False, access_mode='automatic')
                except Exception as e:
                    logger.error(f"Error running schema script: {e}")
                    logger.error("Please run scripts/create_benchmark_schema.py to initialize the database schema")
                    sys.exit(1)
            else:
                logger.error(f"Schema script not found. Checked paths: {possible_paths}")
                logger.error("Please run scripts/create_benchmark_schema.py to initialize the database schema")
                sys.exit(1)
        
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def find_or_create_model(conn, model_name):
    """Find a model in the database or create it if it doesn't exist"""
    # Check if model exists
    existing_model = conn.execute("""
    SELECT model_id, model_family FROM models WHERE model_name = ?
    """, [model_name]).fetchone()
    
    if existing_model:
        logger.info(f"Found existing model: {model_name} (ID: {existing_model[0]}, Family: {existing_model[1]})")
        return existing_model[0], existing_model[1]
    
    # Try to extract model family from name
    model_family = None
    if 'bert' in model_name.lower():
        model_family = 'bert'
    elif 't5' in model_name.lower():
        model_family = 't5'
    elif 'gpt' in model_name.lower():
        model_family = 'gpt'
    elif 'llama' in model_name.lower():
        model_family = 'llama'
    elif 'vit' in model_name.lower():
        model_family = 'vit'
    elif 'clip' in model_name.lower():
        model_family = 'clip'
    elif 'whisper' in model_name.lower():
        model_family = 'whisper'
    elif 'wav2vec' in model_name.lower():
        model_family = 'wav2vec'
    else:
        # If we can't determine family, use the first part of the name
        model_family = model_name.split('-')[0].lower()
    
    # Determine modality from family
    modality = 'text'  # default
    if model_family in ['vit', 'clip']:
        modality = 'image'
    elif model_family in ['whisper', 'wav2vec']:
        modality = 'audio'
    elif model_family in ['llava']:
        modality = 'multimodal'
    
    # Create a new model
    conn.execute("""
    INSERT INTO models (model_name, model_family, modality, source, metadata)
    VALUES (?, ?, ?, ?, ?)
    """, [model_name, model_family, modality, 'huggingface', '{}'])
    
    # Get the inserted ID
    new_model = conn.execute("""
    SELECT model_id FROM models WHERE model_name = ?
    """, [model_name]).fetchone()
    
    logger.info(f"Created new model entry: {model_name} (ID: {new_model[0]}, Family: {model_family})")
    
    return new_model[0], model_family

def find_or_create_hardware(conn, hardware_type, device_name=None):
    """Find a hardware platform in the database or create it if it doesn't exist"""
    # Build query based on available parameters
    query = "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?"
    params = [hardware_type]
    
    if device_name:
        query += " AND device_name = ?"
        params.append(device_name)
    
    # Check if hardware exists
    existing_hardware = conn.execute(query, params).fetchone()
    
    if existing_hardware:
        logger.info(f"Found existing hardware: {hardware_type} {device_name or ''} (ID: {existing_hardware[0]})")
        return existing_hardware[0]
    
    # If we're just checking, get any hardware of this type
    if not device_name:
        existing_any = conn.execute(
            "SELECT hardware_id, device_name FROM hardware_platforms WHERE hardware_type = ? LIMIT 1", 
            [hardware_type]
        ).fetchone()
        
        if existing_any:
            logger.info(f"Found existing hardware type: {hardware_type} Device: {existing_any[1]} (ID: {existing_any[0]})")
            return existing_any[0]
    
    # We need to create a new hardware entry
    # Try to detect hardware details
    platform = None
    platform_version = None
    driver_version = None
    memory_gb = None
    compute_units = None
    
    try:
        if hardware_type == 'cpu':
            import platform as plt
            import psutil
            platform = plt.system()
            platform_version = plt.version()
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            compute_units = psutil.cpu_count(logical=False)
            device_name = device_name or plt.processor()
        
        elif hardware_type == 'cuda':
            # Try to detect CUDA details
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = device_name or torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    platform = 'CUDA'
                    platform_version = torch.version.cuda
                    compute_units = 0  # Not easily available
            except (ImportError, AttributeError):
                pass
        
        elif hardware_type == 'rocm':
            # Try to detect ROCm details
            try:
                import torch
                if torch.cuda.is_available() and 'rocm' in torch.__version__.lower():
                    device_name = device_name or torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    platform = 'ROCm'
                    platform_version = "Unknown"  # Not easily available
                    compute_units = 0  # Not easily available
            except (ImportError, AttributeError):
                pass
    except Exception as e:
        logger.warning(f"Error detecting hardware details: {e}")
    
    # Use defaults if detection failed
    device_name = device_name or f"{hardware_type.upper()} Device"
    platform = platform or hardware_type.upper()
    platform_version = platform_version or "Unknown"
    memory_gb = memory_gb or 0
    compute_units = compute_units or 0
    
    # Create a new hardware entry
    conn.execute("""
    INSERT INTO hardware_platforms (hardware_type, device_name, platform, platform_version, 
                                   driver_version, memory_gb, compute_units, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [hardware_type, device_name, platform, platform_version, 
         driver_version, memory_gb, compute_units, '{}'])
    
    # Get the inserted ID
    new_hardware_id = conn.execute(query, params).fetchone()[0]
    
    logger.info(f"Created new hardware entry: {hardware_type} {device_name} (ID: {new_hardware_id})")
    
    return new_hardware_id

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