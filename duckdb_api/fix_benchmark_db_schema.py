#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix benchmark database schema to allow storing real benchmark data.

This script updates the DuckDB database schema to add the missing 'test_timestamp' column
to the performance_results table, which is required for the update_benchmark_db.py script.
"""

import os
import sys
import argparse
import duckdb
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_database_schema(db_path):
    """
    Fix database schema to add missing columns and tables.
    
    Args:
        db_path: Path to the database file
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Fixing database schema for {db_path}")
    
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        
        # Create a backup of the database first
        backup_path = f"{db_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating backup of database at {backup_path}")
        
        # Create physical copy of the database file using Python's file operations
        # since DuckDB's COPY command might not work for the database file itself
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup successfully: {backup_path}")
        
        # List all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        logger.info(f"Existing tables: {table_names}")
        
        # Fix models table
        if 'models' in table_names:
            logger.info("Checking models table schema")
            
            # Get column names
            columns = conn.execute("PRAGMA table_info('models')").fetchall()
            column_names = [col[1] for col in columns]
            logger.info(f"Existing columns in models table: {column_names}")
            
            if 'model_type' not in column_names and 'modality' in column_names:
                logger.info("Adding model_type as an alias for modality")
                conn.execute("ALTER TABLE models ADD COLUMN model_type VARCHAR")
                conn.execute("UPDATE models SET model_type = modality")
        
        # Fix hardware_platforms table
        if 'hardware_platforms' in table_names:
            logger.info("Checking hardware_platforms table schema")
            
            # Get column names
            columns = conn.execute("PRAGMA table_info('hardware_platforms')").fetchall()
            column_names = [col[1] for col in columns]
            logger.info(f"Existing columns in hardware_platforms table: {column_names}")
            
            if 'memory_capacity' not in column_names and 'memory_gb' in column_names:
                logger.info("Adding memory_capacity as an alias for memory_gb")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN memory_capacity FLOAT")
                conn.execute("UPDATE hardware_platforms SET memory_capacity = memory_gb")
            
            if 'supported_precisions' not in column_names:
                logger.info("Adding supported_precisions column")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN supported_precisions VARCHAR")
                conn.execute("UPDATE hardware_platforms SET supported_precisions = 'fp32,fp16'")
                
            if 'max_batch_size' not in column_names:
                logger.info("Adding max_batch_size column")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN max_batch_size INTEGER DEFAULT 64")
            
            if 'detected_at' not in column_names:
                logger.info("Adding detected_at column")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            
            if 'is_simulated' not in column_names:
                logger.info("Adding is_simulated column")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE")
            
            if 'simulation_reason' not in column_names:
                logger.info("Adding simulation_reason column")
                conn.execute("ALTER TABLE hardware_platforms ADD COLUMN simulation_reason VARCHAR")
        
        # Fix performance_results table
        if 'performance_results' in table_names:
            logger.info("Checking performance_results table schema")
            
            # Get column names
            columns = conn.execute("PRAGMA table_info('performance_results')").fetchall()
            column_names = [col[1] for col in columns]
            logger.info(f"Existing columns in performance_results table: {column_names}")
            
            # Check if the primary key is called 'result_id' or 'id'
            if 'id' not in column_names and 'result_id' in column_names:
                logger.info("Adding id as an alias for result_id")
                # Create a view instead of a column
                conn.execute("CREATE OR REPLACE VIEW performance_results_view AS SELECT result_id as id, * FROM performance_results")
            
            # Check if test_timestamp column exists
            if 'test_timestamp' not in column_names:
                logger.info("Adding test_timestamp column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN test_timestamp TIMESTAMP")
            
            # Check if is_simulated column exists
            if 'is_simulated' not in column_names:
                logger.info("Adding is_simulated column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE")
            
            # Check if simulation_reason column exists
            if 'simulation_reason' not in column_names:
                logger.info("Adding simulation_reason column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN simulation_reason VARCHAR")

            # Check if p50_latency_ms column exists
            if 'p50_latency_ms' not in column_names:
                logger.info("Adding p50_latency_ms column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN p50_latency_ms FLOAT")
            
            # Check if p90_latency_ms column exists
            if 'p90_latency_ms' not in column_names:
                logger.info("Adding p90_latency_ms column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN p90_latency_ms FLOAT")
            
            # Check if p99_latency_ms column exists
            if 'p99_latency_ms' not in column_names:
                logger.info("Adding p99_latency_ms column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN p99_latency_ms FLOAT")
            
            # Check if power_watts column exists
            if 'power_watts' not in column_names:
                logger.info("Adding power_watts column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN power_watts FLOAT")
            
            # Check if energy_efficiency_items_per_joule column exists
            if 'energy_efficiency_items_per_joule' not in column_names:
                logger.info("Adding energy_efficiency_items_per_joule column to performance_results table")
                conn.execute("ALTER TABLE performance_results ADD COLUMN energy_efficiency_items_per_joule FLOAT")
        
        # Fix test_runs table
        if 'test_runs' in table_names:
            logger.info("Checking test_runs table schema")
            columns = conn.execute("PRAGMA table_info('test_runs')").fetchall()
            column_names = [col[1] for col in columns]
            primary_keys = [col[1] for col in columns if col[5] == 1]  # Get primary key columns
            logger.info(f"Existing columns in test_runs table: {column_names}")
            logger.info(f"Primary keys in test_runs table: {primary_keys}")
            
            # Check if run_id is a primary key and if it's TEXT or INTEGER
            if 'run_id' in primary_keys:
                try:
                    # Get column type information
                    run_id_type = None
                    for col in columns:
                        if col[1] == 'run_id':
                            run_id_type = col[2]
                            break
                    
                    logger.info(f"run_id column type: {run_id_type}")
                    
                    # If run_id is INTEGER but we need to store string IDs, create a new backup table
                    if run_id_type and 'INT' in run_id_type.upper():
                        logger.info("Creating a new test_runs_string table that accepts string run_ids")
                        conn.execute("""
                        CREATE TABLE IF NOT EXISTS test_runs_string (
                            id INTEGER,
                            run_id VARCHAR PRIMARY KEY,
                            test_name VARCHAR,
                            test_type VARCHAR,
                            success BOOLEAN,
                            started_at TIMESTAMP,
                            completed_at TIMESTAMP,
                            execution_time_seconds FLOAT,
                            metadata VARCHAR,
                            git_commit VARCHAR,
                            git_branch VARCHAR,
                            command_line VARCHAR
                        )
                        """)
                        
                        # Copy existing data if any
                        try:
                            conn.execute("""
                            INSERT INTO test_runs_string (
                                id, run_id, test_name, test_type, success, started_at, completed_at, 
                                execution_time_seconds, metadata, git_commit, git_branch, command_line
                            )
                            SELECT 
                                id, CAST(run_id AS VARCHAR), test_name, test_type, success, started_at, 
                                completed_at, execution_time_seconds, metadata, git_commit, git_branch, command_line 
                            FROM test_runs
                            """)
                            logger.info("Copied existing data to test_runs_string table")
                        except Exception as e:
                            logger.warning(f"Error copying test_runs data: {e}")
                except Exception as e:
                    logger.warning(f"Error checking test_runs run_id type: {e}")
        
        # Create test_results table if it doesn't exist
        if 'test_results' not in table_names:
            logger.info("Creating test_results table")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                test_date VARCHAR,
                status VARCHAR,
                test_type VARCHAR,
                model_id INTEGER,
                hardware_id INTEGER,
                endpoint_type VARCHAR,
                success BOOLEAN,
                execution_time FLOAT,
                memory_usage FLOAT,
                is_simulated BOOLEAN DEFAULT FALSE
            )
            """)
        else:
            logger.info("test_results table already exists")
            # Check if is_simulated column exists
            columns = conn.execute("PRAGMA table_info('test_results')").fetchall()
            column_names = [col[1] for col in columns]
            
            if 'is_simulated' not in column_names:
                logger.info("Adding is_simulated column to test_results table")
                conn.execute("ALTER TABLE test_results ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE")
        
        # Close connection
        conn.close()
        
        logger.info("Database schema fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing database schema: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fix benchmark database schema")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", 
                      help="Path to benchmark database")
    parser.add_argument("--verbose", action="store_true", 
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Fix database schema
    if fix_database_schema(args.db_path):
        print("\nSuccess: Database schema fixed successfully.")
        print("You can now run update_benchmark_db.py to add real benchmark data.")
        return 0
    else:
        print("\nError: Failed to fix database schema.")
        return 1

if __name__ == "__main__":
    sys.exit(main())