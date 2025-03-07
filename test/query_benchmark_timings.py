#!/usr/bin/env python3
"""
Query Benchmark Timings

A simple script to query benchmark timing data from the DuckDB database
and display it in a table format.
"""

import os
import sys
import duckdb
from pathlib import Path

def query_benchmark_timings():
    """Query benchmark timings from the database and print as a table."""
    # Connect to the database
    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        print(f"Connected to database: {db_path}")
        
        # Query the performance results
        query = """
        SELECT 
            m.model_name,
            hp.hardware_type,
            pr.batch_size,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        ORDER BY
            m.model_name, hp.hardware_type, pr.batch_size
        """
        
        results = conn.execute(query).fetchall()
        
        if not results:
            print("No benchmark timing data found in the database.")
            return
        
        # Query for error information as well
        error_query = """
        SELECT 
            m.model_name,
            hp.hardware_type,
            pr.batch_size,
            pr.status,
            pr.error_type,
            pr.error_message
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE
            pr.status = 'failed'
        ORDER BY
            m.model_name, hp.hardware_type, pr.batch_size
        """
        
        error_results = conn.execute(error_query).fetchall()
        
        # Count successful and failed benchmarks
        successful_count = sum(1 for row in results if row[3] is not None)
        failed_count = len(error_results)
        
        # Print the table header
        print("\nBenchmark Timing Results")
        print("=" * 110)
        print(f"{'Model Name':<30} {'Hardware':<10} {'Batch Size':<12} {'Latency (ms)':<15} {'Throughput (it/s)':<20} {'Memory (MB)':<12} {'Status':<10}")
        print("-" * 110)
        
        # Print the successful results
        for row in results:
            model_name = row[0]
            hardware_type = row[1]
            batch_size = row[2]
            latency = row[3]
            throughput = row[4]
            memory = row[5]
            
            # Check if this is a failed benchmark
            is_failed = any(error_row[0] == model_name and error_row[1] == hardware_type and error_row[2] == batch_size 
                           for error_row in error_results)
            
            if is_failed:
                status = "Failed"
                print(f"{model_name:<30} {hardware_type:<10} {batch_size:<12} {'---':<15} {'---':<20} {'---':<12} {'Failed':<10}")
            else:
                if latency is not None:
                    print(f"{model_name:<30} {hardware_type:<10} {batch_size:<12} {latency:<15.2f} {throughput:<20.2f} {memory:<12.2f} {'Success':<10}")
                else:
                    print(f"{model_name:<30} {hardware_type:<10} {batch_size:<12} {'N/A':<15} {'N/A':<20} {'N/A':<12} {'Unknown':<10}")
        
        print("-" * 110)
        print(f"Total results: {len(results)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        
        # Print details for failed benchmarks if any
        if failed_count > 0:
            print("\nFailed Benchmarks:")
            for i, row in enumerate(error_results, 1):
                model_name = row[0]
                hardware_type = row[1]
                batch_size = row[2]
                error_type = row[4] or "Unknown"
                error_message = row[5] or "No error message captured"
                
                print(f"{i}. {model_name} on {hardware_type} with batch size {batch_size}:")
                print(f"   Error type: {error_type}")
                print(f"   Error message: {error_message[:200]}{'...' if len(error_message) > 200 else ''}")
                print()
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return

if __name__ == "__main__":
    query_benchmark_timings()