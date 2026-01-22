#!/usr/bin/env python3
"""
Test script for database performance optimization.

This script demonstrates the performance improvements achieved by the database
performance optimization techniques implemented in the db_performance_optimizer module.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple
import random
import multiprocessing
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_db_performance")

# Import the database optimization module
try:
    from duckdb_api.simulation_validation.db_performance_optimizer import (
        DBPerformanceOptimizer,
        get_db_optimizer
    )
except ImportError:
    logger.error("Failed to import DBPerformanceOptimizer. Make sure the module is available.")
    sys.exit(1)

# Import the database integration module
try:
    from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
except ImportError:
    logger.error("Failed to import SimulationValidationDBIntegration. Make sure duckdb_api is properly installed.")
    sys.exit(1)

# Import base classes for creating test data
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)


def generate_test_data(
    num_records: int,
    hardware_types: List[str],
    model_types: List[str],
    batch_sizes: List[int],
    precision_types: List[str]
) -> List[ValidationResult]:
    """
    Generate test data for performance testing.
    
    Args:
        num_records: Number of validation results to generate
        hardware_types: List of hardware types to use
        model_types: List of model types to use
        batch_sizes: List of batch sizes to use
        precision_types: List of precision types to use
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    for _ in range(num_records):
        # Choose random hardware and model types
        hardware_id = random.choice(hardware_types)
        model_id = random.choice(model_types)
        batch_size = random.choice(batch_sizes)
        precision = random.choice(precision_types)
        
        # Generate random metrics
        throughput = random.uniform(100, 10000)
        latency = random.uniform(1, 100)
        memory = random.uniform(100, 10000)
        power = random.uniform(10, 500)
        init_time = random.uniform(10, 1000)
        warmup_time = random.uniform(10, 500)
        
        # Random adjustment factor for simulation vs hardware
        adjustment = random.uniform(0.85, 1.15)
        
        # Create simulation result
        sim_metrics = {
            "throughput_items_per_second": throughput * adjustment,
            "average_latency_ms": latency * adjustment,
            "memory_peak_mb": memory * adjustment,
            "power_consumption_w": power * adjustment,
            "initialization_time_ms": init_time * adjustment,
            "warmup_time_ms": warmup_time * adjustment
        }
        
        sim_metadata = {
            "simulation_version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            "parameters": {
                "batch_size": batch_size,
                "precision": precision,
                "threads": random.randint(1, 32)
            }
        }
        
        # Generate timestamp within last 90 days
        days_ago = random.randint(0, 90)
        timestamp = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).isoformat()
        
        sim_result = SimulationResult(
            model_id=model_id,
            hardware_id=hardware_id,
            metrics=sim_metrics,
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            simulation_version=sim_metadata["simulation_version"],
            additional_metadata=sim_metadata
        )
        
        # Create hardware result
        hw_metrics = {
            "throughput_items_per_second": throughput,
            "average_latency_ms": latency,
            "memory_peak_mb": memory,
            "power_consumption_w": power,
            "initialization_time_ms": init_time,
            "warmup_time_ms": warmup_time
        }
        
        hw_details = {
            "cpu": f"CPU-{random.randint(1, 5)}",
            "memory": f"{random.randint(8, 128)}GB",
            "gpu": f"GPU-{random.randint(1, 5)}",
            "os": f"OS-{random.randint(1, 3)}"
        }
        
        test_env = {
            "temperature": random.uniform(20, 30),
            "location": f"Location-{random.randint(1, 3)}",
            "run_id": str(uuid.uuid4())
        }
        
        hw_result = HardwareResult(
            model_id=model_id,
            hardware_id=hardware_id,
            metrics=hw_metrics,
            batch_size=batch_size,
            precision=precision,
            timestamp=timestamp,
            hardware_details=hw_details,
            test_environment=test_env
        )
        
        # Create validation result
        metrics_comparison = {}
        for metric in hw_metrics:
            sim_value = sim_metrics[metric]
            hw_value = hw_metrics[metric]
            absolute_error = abs(sim_value - hw_value)
            relative_error = absolute_error / hw_value if hw_value != 0 else 0
            mape = relative_error * 100  # Mean Absolute Percentage Error
            
            metrics_comparison[metric] = {
                "simulation_value": sim_value,
                "hardware_value": hw_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "mape": mape
            }
        
        # Compute overall accuracy metrics
        mape_values = [metrics_comparison[m]["mape"] for m in metrics_comparison]
        overall_accuracy = sum(mape_values) / len(mape_values) if mape_values else 0
        
        additional_metrics = {
            "overall_accuracy_score": 100 - overall_accuracy,
            "confidence_interval": random.uniform(0.5, 5.0),
            "validation_notes": f"Test validation for {model_id} on {hardware_id}"
        }
        
        validation_result = ValidationResult(
            simulation_result=sim_result,
            hardware_result=hw_result,
            metrics_comparison=metrics_comparison,
            validation_timestamp=timestamp,
            validation_version="v1.0",
            additional_metrics=additional_metrics
        )
        
        results.append(validation_result)
    
    return results


def test_baseline_performance(
    db_path: str,
    num_queries: int,
    hardware_types: List[str],
    model_types: List[str]
) -> Dict[str, float]:
    """
    Test baseline database performance without optimizations.
    
    Args:
        db_path: Path to the database
        num_queries: Number of queries to execute
        hardware_types: List of hardware types for filtering
        model_types: List of model types for filtering
        
    Returns:
        Dictionary with timing results
    """
    # Create database integration
    db_integration = SimulationValidationDBIntegration(db_path=db_path)
    
    # Measure query performance
    query_times = []
    
    for _ in range(num_queries):
        # Choose random hardware and model types for filtering
        hardware_id = random.choice(hardware_types)
        model_id = random.choice(model_types)
        
        # Measure query time
        start_time = time.time()
        results = db_integration.get_validation_results(
            hardware_id=hardware_id,
            model_id=model_id,
            limit=100
        )
        end_time = time.time()
        
        query_time = end_time - start_time
        query_times.append(query_time)
        
        # Log result
        logger.debug(f"Baseline query returned {len(results)} results in {query_time:.4f} seconds")
    
    # Calculate statistics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    min_query_time = min(query_times) if query_times else 0
    max_query_time = max(query_times) if query_times else 0
    
    logger.info(f"Baseline performance: Avg={avg_query_time:.4f}s, Min={min_query_time:.4f}s, Max={max_query_time:.4f}s")
    
    return {
        "avg_query_time": avg_query_time,
        "min_query_time": min_query_time,
        "max_query_time": max_query_time,
        "num_queries": len(query_times)
    }


def test_optimized_performance(
    db_path: str,
    num_queries: int,
    hardware_types: List[str],
    model_types: List[str],
    enable_caching: bool = True
) -> Dict[str, float]:
    """
    Test database performance with optimizations.
    
    Args:
        db_path: Path to the database
        num_queries: Number of queries to execute
        hardware_types: List of hardware types for filtering
        model_types: List of model types for filtering
        enable_caching: Whether to enable query caching
        
    Returns:
        Dictionary with timing results
    """
    # Create optimizer
    optimizer = get_db_optimizer(
        db_path=db_path,
        enable_caching=enable_caching
    )
    
    # Create indexes if needed
    optimizer.create_indexes()
    
    # Analyze tables for optimization
    optimizer.analyze_tables()
    
    # Measure query performance
    query_times = []
    cache_hits = 0
    
    for i in range(num_queries):
        # Choose random hardware and model types for filtering
        hardware_id = random.choice(hardware_types)
        model_id = random.choice(model_types)
        
        # For every 3rd query, reuse a previous combination to test caching
        if i % 3 == 0 and i > 0 and enable_caching:
            # Reuse previous combination
            pass
        
        # Measure query time
        start_time = time.time()
        results = optimizer.get_validation_results_optimized(
            hardware_id=hardware_id,
            model_id=model_id,
            limit=100,
            use_cache=enable_caching
        )
        end_time = time.time()
        
        query_time = end_time - start_time
        query_times.append(query_time)
        
        # Log result
        logger.debug(f"Optimized query returned {len(results)} results in {query_time:.4f} seconds")
    
    # Get cache statistics
    if enable_caching and optimizer.cache:
        cache_stats = optimizer.cache.get_stats()
        cache_hits = cache_stats.get("hits", 0)
    
    # Calculate statistics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    min_query_time = min(query_times) if query_times else 0
    max_query_time = max(query_times) if query_times else 0
    
    # Log results
    cache_info = f", Cache hits: {cache_hits}" if enable_caching else ""
    logger.info(f"Optimized performance: Avg={avg_query_time:.4f}s, Min={min_query_time:.4f}s, Max={max_query_time:.4f}s{cache_info}")
    
    return {
        "avg_query_time": avg_query_time,
        "min_query_time": min_query_time,
        "max_query_time": max_query_time,
        "num_queries": len(query_times),
        "cache_hits": cache_hits if enable_caching else 0
    }


def test_batch_insertion_performance(
    db_path: str,
    records: List[ValidationResult],
    batch_sizes: List[int]
) -> Dict[str, Any]:
    """
    Test batch insertion performance.
    
    Args:
        db_path: Path to the database
        records: List of ValidationResult objects to insert
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Create optimizer with specified batch size
        optimizer = get_db_optimizer(
            db_path=db_path,
            batch_size=batch_size
        )
        
        # Create a copy of the records
        test_records = records.copy()
        
        # Measure insertion time
        start_time = time.time()
        optimizer.batch_insert_validation_results(test_records)
        end_time = time.time()
        
        insertion_time = end_time - start_time
        
        # Log result
        logger.info(f"Batch insertion (size={batch_size}): {len(test_records)} records in {insertion_time:.4f} seconds")
        
        results[batch_size] = {
            "insertion_time": insertion_time,
            "records_per_second": len(test_records) / insertion_time if insertion_time > 0 else 0,
            "num_records": len(test_records)
        }
    
    return results


def test_backup_restore(db_path: str) -> Dict[str, float]:
    """
    Test database backup and restore performance.
    
    Args:
        db_path: Path to the database
        
    Returns:
        Dictionary with timing results
    """
    # Create optimizer
    optimizer = get_db_optimizer(db_path=db_path)
    
    # Measure backup time
    start_time = time.time()
    backup_path = optimizer.backup_database()
    end_time = time.time()
    
    backup_time = end_time - start_time
    
    # Log result
    logger.info(f"Database backup: {backup_time:.4f} seconds")
    
    # Measure restore time
    start_time = time.time()
    restore_success = optimizer.restore_database(backup_path)
    end_time = time.time()
    
    restore_time = end_time - start_time
    
    # Log result
    logger.info(f"Database restore: {restore_time:.4f} seconds, Success: {restore_success}")
    
    return {
        "backup_time": backup_time,
        "restore_time": restore_time,
        "backup_success": backup_path is not None,
        "restore_success": restore_success
    }


def create_test_database(
    db_path: str,
    num_records: int
) -> None:
    """
    Create a test database with sample data.
    
    Args:
        db_path: Path to the database
        num_records: Number of records to generate
    """
    # Define test data parameters
    hardware_types = [f"hw-{i}" for i in range(1, 6)]
    model_types = [f"model-{i}" for i in range(1, 11)]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    precision_types = ["fp32", "fp16", "int8", "int4"]
    
    # Generate test data
    logger.info(f"Generating {num_records} test records...")
    validation_results = generate_test_data(
        num_records=num_records,
        hardware_types=hardware_types,
        model_types=model_types,
        batch_sizes=batch_sizes,
        precision_types=precision_types
    )
    
    # Create database integration
    db_integration = SimulationValidationDBIntegration(db_path=db_path)
    
    # Insert test data
    logger.info("Inserting test data into database...")
    for i, result in enumerate(validation_results):
        if i % 100 == 0:
            logger.info(f"Inserted {i}/{len(validation_results)} records...")
        
        db_integration.store_validation_result(result)
    
    logger.info(f"Successfully created test database with {num_records} records")


def run_performance_tests(
    db_path: str,
    num_queries: int,
    num_insertion_records: int
) -> Dict[str, Any]:
    """
    Run comprehensive performance tests.
    
    Args:
        db_path: Path to the database
        num_queries: Number of queries to run in each test
        num_insertion_records: Number of records to insert in batch tests
        
    Returns:
        Dictionary with test results
    """
    # Define test data parameters
    hardware_types = [f"hw-{i}" for i in range(1, 6)]
    model_types = [f"model-{i}" for i in range(1, 11)]
    batch_sizes = [1, 10, 50, 100, 200]
    
    # Generate test data for batch insertion
    logger.info(f"Generating {num_insertion_records} test records for batch insertion...")
    batch_test_records = generate_test_data(
        num_records=num_insertion_records,
        hardware_types=hardware_types,
        model_types=model_types,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        precision_types=["fp32", "fp16", "int8"]
    )
    
    # Create optimizer
    optimizer = get_db_optimizer(db_path=db_path)
    
    # Get database statistics before tests
    stats_before = optimizer.get_database_stats()
    
    # Run baseline performance tests
    logger.info("\n--- Running baseline performance tests ---")
    baseline_results = test_baseline_performance(
        db_path=db_path,
        num_queries=num_queries,
        hardware_types=hardware_types,
        model_types=model_types
    )
    
    # Run optimized performance tests without caching
    logger.info("\n--- Running optimized performance tests (no caching) ---")
    optimized_no_cache_results = test_optimized_performance(
        db_path=db_path,
        num_queries=num_queries,
        hardware_types=hardware_types,
        model_types=model_types,
        enable_caching=False
    )
    
    # Run optimized performance tests with caching
    logger.info("\n--- Running optimized performance tests (with caching) ---")
    optimized_with_cache_results = test_optimized_performance(
        db_path=db_path,
        num_queries=num_queries,
        hardware_types=hardware_types,
        model_types=model_types,
        enable_caching=True
    )
    
    # Run batch insertion tests
    logger.info("\n--- Running batch insertion tests ---")
    batch_insertion_results = test_batch_insertion_performance(
        db_path=db_path,
        records=batch_test_records,
        batch_sizes=batch_sizes
    )
    
    # Run backup/restore tests
    logger.info("\n--- Running backup/restore tests ---")
    backup_restore_results = test_backup_restore(db_path)
    
    # Get database statistics after tests
    stats_after = optimizer.get_database_stats()
    
    # Combine and return results
    return {
        "baseline_performance": baseline_results,
        "optimized_no_cache_performance": optimized_no_cache_results,
        "optimized_with_cache_performance": optimized_with_cache_results,
        "batch_insertion_performance": batch_insertion_results,
        "backup_restore_performance": backup_restore_results,
        "database_stats_before": stats_before,
        "database_stats_after": stats_after
    }


def print_test_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the performance test results.
    
    Args:
        results: Results from run_performance_tests
    """
    print("\n=== Database Performance Test Summary ===\n")
    
    # Print database statistics
    stats_before = results.get("database_stats_before", {})
    stats_after = results.get("database_stats_after", {})
    
    print("Database Statistics:")
    print(f"- Path: {stats_after.get('database_path', 'unknown')}")
    print(f"- Size: {stats_after.get('file_size_mb', 0):.2f} MB")
    print(f"- Total Records: {stats_after.get('total_records', 0)}")
    print(f"- Number of Indexes: {stats_after.get('index_count', 0)}")
    
    # Print query performance comparison
    baseline = results.get("baseline_performance", {})
    no_cache = results.get("optimized_no_cache_performance", {})
    with_cache = results.get("optimized_with_cache_performance", {})
    
    baseline_avg = baseline.get("avg_query_time", 0)
    no_cache_avg = no_cache.get("avg_query_time", 0)
    with_cache_avg = with_cache.get("avg_query_time", 0)
    
    # Calculate improvement percentages
    if baseline_avg > 0:
        no_cache_improvement = ((baseline_avg - no_cache_avg) / baseline_avg) * 100
        with_cache_improvement = ((baseline_avg - with_cache_avg) / baseline_avg) * 100
    else:
        no_cache_improvement = 0
        with_cache_improvement = 0
    
    print("\nQuery Performance:")
    print(f"- Baseline: {baseline_avg:.4f} sec/query")
    print(f"- Optimized (no cache): {no_cache_avg:.4f} sec/query ({no_cache_improvement:.1f}% improvement)")
    print(f"- Optimized (with cache): {with_cache_avg:.4f} sec/query ({with_cache_improvement:.1f}% improvement)")
    print(f"- Cache hits: {with_cache.get('cache_hits', 0)}")
    
    # Print batch insertion performance
    batch_results = results.get("batch_insertion_performance", {})
    
    print("\nBatch Insertion Performance:")
    for batch_size, batch_data in sorted(batch_results.items()):
        print(f"- Batch size {batch_size}: {batch_data.get('insertion_time', 0):.4f} sec "
              f"({batch_data.get('records_per_second', 0):.1f} records/sec)")
    
    # Print backup/restore performance
    backup_restore = results.get("backup_restore_performance", {})
    
    print("\nBackup/Restore Performance:")
    print(f"- Backup: {backup_restore.get('backup_time', 0):.4f} sec")
    print(f"- Restore: {backup_restore.get('restore_time', 0):.4f} sec")
    
    # Print overall improvements
    print("\nOverall Improvements:")
    print(f"- Query performance improvement: {with_cache_improvement:.1f}%")
    print(f"- Query with index improvement: {no_cache_improvement:.1f}%")
    
    best_batch_size = max(batch_results.items(), key=lambda x: x[1].get('records_per_second', 0))[0]
    best_batch_rps = batch_results[best_batch_size].get('records_per_second', 0)
    
    print(f"- Best batch size: {best_batch_size} ({best_batch_rps:.1f} records/sec)")


def main() -> None:
    """Main function to run the performance tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Database Performance Test Script")
    parser.add_argument("--db-path", default="./benchmark_db_test.duckdb", help="Path to the test database")
    parser.add_argument("--create-db", action="store_true", help="Create a new test database")
    parser.add_argument("--num-records", type=int, default=1000, help="Number of records for database creation")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to run in each test")
    parser.add_argument("--num-batch-records", type=int, default=500, help="Number of records for batch insertion test")
    
    args = parser.parse_args()
    
    # Create test database if requested
    if args.create_db:
        logger.info(f"Creating test database at {args.db_path}...")
        
        # Remove existing database if it exists
        if os.path.exists(args.db_path):
            os.remove(args.db_path)
        
        create_test_database(args.db_path, args.num_records)
    
    # Run performance tests
    logger.info("Running performance tests...")
    test_results = run_performance_tests(
        db_path=args.db_path,
        num_queries=args.num_queries,
        num_insertion_records=args.num_batch_records
    )
    
    # Print summary
    print_test_summary(test_results)
    
    # Save results to file
    results_path = f"db_performance_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()