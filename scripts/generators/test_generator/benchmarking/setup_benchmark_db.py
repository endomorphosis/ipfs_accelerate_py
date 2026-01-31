#!/usr/bin/env python3
"""
Setup benchmark database for IPFS Accelerate Python framework.

This script creates the database schema for storing benchmark results.
"""

import os
import sys
import time
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.error("DuckDB not found. Please install with: pip install duckdb")
    sys.exit(1)


def create_benchmark_schema(db_path: str, overwrite: bool = False) -> bool:
    """
    Create benchmark database schema.
    
    Args:
        db_path: Path to database file
        overwrite: Whether to overwrite existing database
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if database exists
        exists = os.path.exists(db_path)
        
        if exists and not overwrite:
            logger.warning(f"Database {db_path} already exists. Use --overwrite to recreate.")
            
            # Verify schema exists by connecting and checking tables
            con = duckdb.connect(db_path)
            tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]
            
            required_tables = ["benchmark_runs", "benchmark_results", "hardware_info"]
            missing_tables = [t for t in required_tables if t not in table_names]
            
            if missing_tables:
                logger.warning(f"Missing tables in existing database: {missing_tables}")
                logger.warning("Schema appears incomplete. Consider using --overwrite to recreate.")
            else:
                logger.info(f"Database schema verified. {len(tables)} tables found.")
                
            con.close()
            return True
            
        # Connect to database (creates if not exists)
        con = duckdb.connect(db_path)
        
        # Create tables
        
        # Table for benchmark runs (metadata)
        con.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP,
            model_id VARCHAR,
            device VARCHAR,
            precision VARCHAR,
            architecture_type VARCHAR,
            task VARCHAR,
            parameter_count BIGINT,
            description VARCHAR
        )
        """)
        
        # Table for benchmark results (performance data)
        con.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id VARCHAR PRIMARY KEY,
            run_id VARCHAR REFERENCES benchmark_runs(id),
            batch_size INTEGER,
            sequence_length INTEGER,
            iterations INTEGER,
            latency_mean_ms DOUBLE,
            latency_median_ms DOUBLE,
            latency_min_ms DOUBLE,
            latency_max_ms DOUBLE,
            latency_std_ms DOUBLE,
            latency_90p_ms DOUBLE,
            throughput_samples_per_sec DOUBLE,
            memory_usage_mb DOUBLE,
            peak_memory_mb DOUBLE,
            first_token_latency_ms DOUBLE,
            time_to_first_token_ms DOUBLE
        )
        """)
        
        # Table for hardware information
        con.execute("""
        CREATE TABLE IF NOT EXISTS hardware_info (
            id VARCHAR PRIMARY KEY,
            run_id VARCHAR REFERENCES benchmark_runs(id),
            device_type VARCHAR,
            device_name VARCHAR,
            device_description VARCHAR,
            performance_tier VARCHAR,
            hardware_details VARCHAR
        )
        """)
        
        # Table for raw benchmark data
        con.execute("""
        CREATE TABLE IF NOT EXISTS raw_benchmark_data (
            id VARCHAR PRIMARY KEY,
            run_id VARCHAR REFERENCES benchmark_runs(id),
            raw_data VARCHAR
        )
        """)
        
        # Create indexes for faster querying
        con.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model ON benchmark_runs(model_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_runs_device ON benchmark_runs(device)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_runs_arch ON benchmark_runs(architecture_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_benchmark_results_run ON benchmark_results(run_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hardware_info_run ON hardware_info(run_id)")
        
        # Create views for common queries
        
        # View for latest benchmark runs
        con.execute("""
        CREATE OR REPLACE VIEW v_latest_benchmarks AS
        SELECT 
            r.id, r.timestamp, r.model_id, r.device, r.precision, 
            r.architecture_type, r.task,
            b.batch_size, b.sequence_length, b.latency_mean_ms, 
            b.throughput_samples_per_sec, b.memory_usage_mb
        FROM 
            benchmark_runs r
        JOIN 
            benchmark_results b ON r.id = b.run_id
        WHERE 
            r.timestamp = (
                SELECT MAX(timestamp) 
                FROM benchmark_runs 
                WHERE model_id = r.model_id AND device = r.device
            )
        ORDER BY 
            r.timestamp DESC
        """)
        
        # View for hardware comparison
        con.execute("""
        CREATE OR REPLACE VIEW v_hardware_comparison AS
        SELECT 
            r.model_id, r.architecture_type, r.device, r.precision,
            b.batch_size, b.sequence_length,
            b.latency_mean_ms, b.throughput_samples_per_sec, b.memory_usage_mb,
            h.device_name, h.performance_tier
        FROM 
            benchmark_runs r
        JOIN 
            benchmark_results b ON r.id = b.run_id
        JOIN 
            hardware_info h ON r.id = h.run_id
        WHERE 
            r.timestamp = (
                SELECT MAX(timestamp) 
                FROM benchmark_runs 
                WHERE model_id = r.model_id AND device = r.device
            )
        ORDER BY 
            r.model_id, b.throughput_samples_per_sec DESC
        """)
        
        # View for throughput by architecture
        con.execute("""
        CREATE OR REPLACE VIEW v_throughput_by_architecture AS
        SELECT 
            r.architecture_type, r.device, r.precision,
            AVG(b.throughput_samples_per_sec) as avg_throughput,
            MIN(b.throughput_samples_per_sec) as min_throughput,
            MAX(b.throughput_samples_per_sec) as max_throughput,
            COUNT(DISTINCT r.model_id) as model_count
        FROM 
            benchmark_runs r
        JOIN 
            benchmark_results b ON r.id = b.run_id
        WHERE 
            b.batch_size = 1
        GROUP BY 
            r.architecture_type, r.device, r.precision
        ORDER BY 
            r.architecture_type, avg_throughput DESC
        """)
        
        # Finalize and close connection
        con.close()
        
        if exists and overwrite:
            logger.info(f"Database schema recreated at {db_path}")
        else:
            logger.info(f"Database schema created at {db_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error creating benchmark schema: {e}")
        return False


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Setup benchmark database")
    
    parser.add_argument("--db-path", type=str, default="benchmark_db.duckdb",
                        help="Path to database file")
    
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing database")
    
    args = parser.parse_args()
    
    success = create_benchmark_schema(args.db_path, args.overwrite)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())