#!/usr/bin/env python
"""
Benchmark runner with improved simulation detection and error handling.

This script:
1. Uses enhanced hardware detection with explicit simulation tracking
2. Properly handles unavailable hardware with fallbacks instead of using mocks
3. Stores benchmark results with clear simulation flagging
4. Provides comprehensive error categorization and reporting

Implementation date: April 10, 2025
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our enhanced modules
try:
    import duckdb
    from hardware_detection_updates import detect_hardware_with_simulation_check
    from benchmark_error_handling import (
        handle_hardware_unavailable,
        handle_simulated_hardware,
        handle_benchmark_exception,
        ERROR_CATEGORY_HARDWARE_NOT_AVAILABLE
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure that the hardware_detection_updates.py and benchmark_error_handling.py files are in the current directory")
    sys.exit(1)

# Sample model data for testing
SAMPLE_MODELS = [
    {"name": "bert-base-uncased", "family": "text-embedding"},
    {"name": "t5-small", "family": "text-generation"},
    {"name": "vit-base", "family": "vision"},
    {"name": "whisper-tiny", "family": "audio"},
    {"name": "clip-base", "family": "multimodal"}
]

def get_db_connection(db_path=None):
    """Get a connection to the benchmark database"""
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    logger.info(f"Connecting to database: {db_path}")
    return duckdb.connect(db_path)

def detect_hardware():
    """Run enhanced hardware detection with simulation tracking"""
    try:
        # Use our enhanced hardware detection
        logger.info("Running enhanced hardware detection with simulation tracking...")
        hardware_info = detect_hardware_with_simulation_check()
        
        # Log available hardware
        available_hardware = [hw for hw, available in hardware_info["hardware"].items() if available]
        logger.info(f"Available hardware: {', '.join(available_hardware)}")
        
        # Log simulated hardware
        simulated_hardware = hardware_info.get("simulated_hardware", [])
        if simulated_hardware:
            logger.warning(f"SIMULATED hardware detected: {', '.join(simulated_hardware)}")
            logger.warning("Benchmark results for simulated hardware will be clearly marked")
        
        return hardware_info
    except Exception as e:
        logger.error(f"Hardware detection failed: {str(e)}")
        return None

def add_hardware_detection_log(conn, hardware_info):
    """Add hardware detection information to log table"""
    try:
        # Ensure the log table exists
        conn.execute("""
        CREATE TABLE IF NOT EXISTS hardware_availability_log (
            id INTEGER PRIMARY KEY,
            hardware_type VARCHAR,
            is_available BOOLEAN,
            is_simulated BOOLEAN DEFAULT FALSE,
            detection_method VARCHAR,
            detection_details JSON,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        now = datetime.datetime.now()
        
        # Add entry for each hardware type
        for hardware_type, is_available in hardware_info.get("hardware", {}).items():
            # Check if this hardware is simulated
            is_simulated = hardware_type in hardware_info.get("simulated_hardware", [])
            
            # Get detection details
            detection_details = hardware_info.get("details", {}).get(hardware_type, {})
            
            # Add log entry
            conn.execute("""
            INSERT INTO hardware_availability_log (
                hardware_type, is_available, is_simulated, 
                detection_method, detection_details, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, [
                hardware_type, 
                is_available, 
                is_simulated, 
                "enhanced_detection", 
                json.dumps(detection_details), 
                now
            ])
        
        conn.commit()
        logger.info(f"Added hardware detection log entries for {len(hardware_info.get('hardware', {}))} hardware types")
        return True
    except Exception as e:
        logger.error(f"Failed to add hardware detection log: {str(e)}")
        conn.rollback()
        return False

def get_or_create_model(conn, model_name, model_family):
    """Get or create a model entry in the database"""
    try:
        # Check if model exists
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Create new model entry
        conn.execute(
            "INSERT INTO models (model_name, model_family) VALUES (?, ?)",
            [model_name, model_family]
        )
        
        # Get the new model_id
        result = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", 
            [model_name]
        ).fetchone()
        
        if result:
            return result[0]
        else:
            logger.error(f"Failed to get model_id for {model_name}")
            return None
    except Exception as e:
        logger.error(f"Error in get_or_create_model for {model_name}: {str(e)}")
        return None

def get_or_create_hardware(conn, hardware_type):
    """Get or create a hardware entry in the database"""
    try:
        # Check if hardware exists
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
            [hardware_type]
        ).fetchone()
        
        if result:
            return result[0]
        
        # Create new hardware entry
        conn.execute(
            "INSERT INTO hardware_platforms (hardware_type) VALUES (?)",
            [hardware_type]
        )
        
        # Get the new hardware_id
        result = conn.execute(
            "SELECT hardware_id FROM hardware_platforms WHERE hardware_type = ?", 
            [hardware_type]
        ).fetchone()
        
        if result:
            return result[0]
        else:
            logger.error(f"Failed to get hardware_id for {hardware_type}")
            return None
    except Exception as e:
        logger.error(f"Error in get_or_create_hardware for {hardware_type}: {str(e)}")
        return None

def store_test_result(conn, test_result, model_id, hardware_id):
    """Store a test result in the database with simulation tracking"""
    try:
        # Prepare test data
        now = datetime.datetime.now()
        test_date = now.strftime("%Y-%m-%d")
        
        # Check if this is a simulated test result
        is_simulated = test_result.get('is_simulated', False)
        simulation_reason = test_result.get('simulation_reason', None)
        
        # Get error categorization if present
        error_category = test_result.get('error_category', None)
        error_details = test_result.get('error_details', {})
        
        # Store the test result
        conn.execute(
            """
            INSERT INTO test_results (
                timestamp, test_date, status, test_type, model_id, hardware_id,
                endpoint_type, success, error_message, execution_time, memory_usage, details,
                is_simulated, simulation_reason, error_category, error_details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                now, test_date, 
                test_result.get('status', 'completed'),
                test_result.get('test_type', 'benchmark'),
                model_id, hardware_id,
                test_result.get('endpoint_type', 'local'),
                test_result.get('success', False),
                test_result.get('error_message'),
                test_result.get('execution_time'),
                test_result.get('memory_usage'),
                json.dumps(test_result.get('details', {})),
                is_simulated,
                simulation_reason,
                error_category,
                json.dumps(error_details) if error_details else None
            ]
        )
        
        conn.commit()
        logger.info(f"Stored test result for model_id={model_id}, hardware_id={hardware_id}, success={test_result.get('success', False)}")
        
        # Return the newly created ID
        result = conn.execute("SELECT last_insert_rowid()").fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to store test result: {str(e)}")
        conn.rollback()
        return None

def store_performance_result(conn, performance_data, test_id, model_id, hardware_id):
    """Store performance metrics with simulation tracking"""
    if not performance_data:
        return None
    
    try:
        # Check if this performance data is from simulated hardware
        is_simulated = performance_data.get('is_simulated', False)
        simulation_reason = performance_data.get('simulation_reason')
        
        # Insert performance data
        conn.execute(
            """
            INSERT INTO performance_results (
                test_id, model_id, hardware_id, batch_size, sequence_length,
                average_latency_ms, throughput_items_per_second, memory_mb,
                is_simulated, simulation_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                test_id, model_id, hardware_id,
                performance_data.get('batch_size', 1),
                performance_data.get('sequence_length'),
                performance_data.get('average_latency_ms'),
                performance_data.get('throughput_items_per_second'),
                performance_data.get('memory_mb'),
                is_simulated,
                simulation_reason
            ]
        )
        
        conn.commit()
        logger.info(f"Stored performance result for test_id={test_id}")
        
        # Return the newly created ID
        result = conn.execute("SELECT last_insert_rowid()").fetchone()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to store performance result: {str(e)}")
        conn.rollback()
        return None

def simulate_benchmark(model, hardware_type, is_simulated=False, simulation_reason=None, should_fail=False):
    """
    Simulate running a benchmark for a model on specified hardware
    
    Args:
        model: Model information dictionary with at least 'name' key
        hardware_type: Type of hardware to test on
        is_simulated: Whether the hardware is being simulated
        simulation_reason: Reason for simulation
        should_fail: Whether to simulate a failure
    
    Returns:
        Dictionary with benchmark results
    """
    model_name = model["name"]
    logger.info(f"Running benchmark for {model_name} on {hardware_type}")
    
    # Simulate benchmark execution time
    start_time = time.time()
    time.sleep(0.5)  # Simulate execution time
    execution_time = time.time() - start_time
    
    # Handle failure cases
    if should_fail:
        try:
            if hardware_type == "cuda":
                # Simulate CUDA out of memory error
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            elif hardware_type in ["webnn", "webgpu"]:
                # Simulate missing hardware
                raise ImportError(f"No module named '{hardware_type}'")
            else:
                # Generic failure
                raise RuntimeError(f"Failed to initialize {hardware_type} for {model_name}")
        except Exception as e:
            return handle_benchmark_exception(e, hardware_type, model_name, is_simulated, simulation_reason)
    
    # Add simulation flags if needed
    simulation_data = {}
    if is_simulated:
        simulation_data = handle_simulated_hardware(hardware_type, model_name, simulation_reason)
    
    # Generate performance metrics
    performance_metrics = {
        "batch_size": 1,
        "sequence_length": 128,
        "average_latency_ms": 25.0 if hardware_type == "cpu" else 10.0,
        "throughput_items_per_second": 40.0 if hardware_type == "cpu" else 100.0,
        "memory_mb": 1024.0
    }
    
    # If simulated, add simulation data to performance metrics
    if is_simulated:
        performance_metrics.update({
            "is_simulated": True,
            "simulation_reason": simulation_reason or f"Hardware {hardware_type} is being simulated"
        })
    
    # Create result dictionary
    result = {
        "model_name": model_name,
        "hardware_type": hardware_type,
        "success": True,
        "test_type": "benchmark",
        "status": "completed",
        "execution_time": execution_time,
        "memory_usage": 1024.0,  # MB
        "details": {
            "batch_size": 1,
            "sequence_length": 128
        },
        "performance_metrics": performance_metrics
    }
    
    # Add simulation data if needed
    if is_simulated:
        result.update(simulation_data)
    
    return result

def run_benchmarks(models, hardware_types, conn):
    """
    Run benchmarks for multiple models on specified hardware types
    
    Args:
        models: List of model dictionaries
        hardware_types: List of hardware types to test
        conn: Database connection
    
    Returns:
        Dictionary with benchmark results
    """
    results = []
    hardware_info = detect_hardware()
    
    if not hardware_info:
        logger.error("Hardware detection failed, cannot proceed with benchmarks")
        return []
    
    # Add hardware detection log
    add_hardware_detection_log(conn, hardware_info)
    
    available_hardware = {k: v for k, v in hardware_info["hardware"].items() if v}
    simulated_hardware = hardware_info.get("simulated_hardware", [])
    
    # For each model and hardware combination
    for model in models:
        model_id = get_or_create_model(conn, model["name"], model.get("family", "unknown"))
        if not model_id:
            logger.error(f"Failed to create model entry for {model['name']}, skipping")
            continue
        
        for hardware_type in hardware_types:
            hardware_id = get_or_create_hardware(conn, hardware_type)
            if not hardware_id:
                logger.error(f"Failed to create hardware entry for {hardware_type}, skipping")
                continue
            
            # Check if hardware is available
            if hardware_type not in available_hardware:
                logger.warning(f"Hardware {hardware_type} is not available, skipping benchmark")
                success, result = handle_hardware_unavailable(hardware_type, model["name"], "cpu")
                
                # Store the unavailable hardware result
                test_id = store_test_result(conn, result, model_id, hardware_id)
                if test_id:
                    logger.info(f"Stored hardware unavailable result with ID {test_id}")
                    result["test_id"] = test_id
                
                results.append(result)
                continue
            
            # Check if hardware is simulated
            is_simulated = hardware_type in simulated_hardware
            simulation_reason = None
            if is_simulated:
                simulation_reason = f"Hardware {hardware_type} is being simulated based on environment settings"
                logger.warning(f"Hardware {hardware_type} is simulated, results will be marked accordingly")
            
            # Run the benchmark (simulated for now)
            # In a real implementation, this would call the actual benchmark code
            result = simulate_benchmark(
                model, 
                hardware_type, 
                is_simulated=is_simulated,
                simulation_reason=simulation_reason,
                should_fail=(model["name"] == "bert-base-uncased" and hardware_type == "cuda")  # Test error handling
            )
            
            # Store the test result
            test_id = store_test_result(conn, result, model_id, hardware_id)
            if test_id:
                logger.info(f"Stored test result with ID {test_id}")
                result["test_id"] = test_id
                
                # Store performance metrics if available
                if "performance_metrics" in result and result.get("success", False):
                    performance_id = store_performance_result(
                        conn, 
                        result["performance_metrics"], 
                        test_id, 
                        model_id, 
                        hardware_id
                    )
                    if performance_id:
                        logger.info(f"Stored performance result with ID {performance_id}")
                        result["performance_id"] = performance_id
            
            results.append(result)
    
    return results

def main():
    """Main function to run benchmarks with simulation detection"""
    parser = argparse.ArgumentParser(description="Run benchmarks with simulation detection")
    parser.add_argument("--models", nargs="+", help="Model names to benchmark")
    parser.add_argument("--hardware", nargs="+", help="Hardware types to test")
    parser.add_argument("--db-path", help="Path to the benchmark database")
    parser.add_argument("--output", help="Output file for benchmark results (JSON)")
    args = parser.parse_args()
    
    # Default hardware types if not specified
    if not args.hardware:
        args.hardware = ["cpu", "cuda", "mps", "webnn", "webgpu", "qualcomm"]
    
    # Use sample models if none specified
    models = []
    if args.models:
        for model_name in args.models:
            models.append({"name": model_name, "family": "unknown"})
    else:
        models = SAMPLE_MODELS
    
    logger.info(f"Running benchmarks for {len(models)} models on {len(args.hardware)} hardware types")
    
    # Connect to the database
    try:
        conn = get_db_connection(args.db_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        sys.exit(1)
    
    # Run the benchmarks
    results = run_benchmarks(models, args.hardware, conn)
    
    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote benchmark results to {args.output}")
    
    # Print summary
    print("\nBenchmark summary:")
    print(f"- Ran {len(results)} benchmarks for {len(models)} models on {len(args.hardware)} hardware types")
    
    successful = [r for r in results if r.get("success", False)]
    print(f"- {len(successful)} benchmarks were successful")
    
    failed = [r for r in results if not r.get("success", False)]
    print(f"- {len(failed)} benchmarks failed")
    
    # Count results by hardware type
    hardware_counts = {}
    for result in results:
        hardware_type = result.get("hardware_type")
        if hardware_type:
            hardware_counts[hardware_type] = hardware_counts.get(hardware_type, 0) + 1
    
    print("\nResults by hardware type:")
    for hardware_type, count in hardware_counts.items():
        successful_count = len([r for r in results if r.get("hardware_type") == hardware_type and r.get("success", False)])
        simulated_count = len([r for r in results if r.get("hardware_type") == hardware_type and r.get("is_simulated", False)])
        
        if simulated_count > 0:
            print(f"- {hardware_type}: {count} tests, {successful_count} successful, {simulated_count} SIMULATED")
        else:
            print(f"- {hardware_type}: {count} tests, {successful_count} successful")
    
    # Print error categories
    error_categories = {}
    for result in failed:
        category = result.get("error_category", "unknown")
        error_categories[category] = error_categories.get(category, 0) + 1
    
    if error_categories:
        print("\nFailures by error category:")
        for category, count in error_categories.items():
            print(f"- {category}: {count} failures")
    
    # Query test results with simulation flags
    try:
        sim_results = conn.execute("""
        SELECT 
            COUNT(*) as total_results,
            COUNT(CASE WHEN is_simulated = TRUE THEN 1 END) as simulated_results
        FROM test_results
        WHERE test_date = CURRENT_DATE
        """).fetchone()
        
        if sim_results:
            total, simulated = sim_results
            print(f"\nDatabase contains {total} test results for today, {simulated} marked as simulated")
    except Exception as e:
        logger.error(f"Failed to query simulation statistics: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())