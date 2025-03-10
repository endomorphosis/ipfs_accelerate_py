#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Benchmark Database Schema

This script updates the DuckDB schema to add flags for simulated data and hardware availability tracking.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_database(db_path):
    """Create a backup of the database before making changes."""
    backup_path = f"{db_path}.bak_{int(time.time())}"
    logger.info(f"Creating backup of {db_path} to {backup_path}")
    try:
        import shutil
        
        shutil.copy(db_path, backup_path)
        logger.info(f"Backup created at {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return False

def update_schema(db_path):
    """Update the database schema to add flags for simulated data."""
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Check if the columns already exist
        result = conn.execute("PRAGMA table_info(performance_results)").fetchall()
        column_names = [col[1] for col in result]
        
        schema_updates = []
        
        # Add is_simulated column to performance_results if it doesn't exist
        if "is_simulated" not in column_names:
            logger.info("Adding is_simulated column to performance_results")
            conn.execute("ALTER TABLE performance_results ADD COLUMN is_simulated BOOLEAN DEFAULT TRUE")
            schema_updates.append("Added is_simulated column to performance_results")
        else:
            logger.info("is_simulated column already exists in performance_results")
            
        # Add simulation_reason column to performance_results if it doesn't exist
        if "simulation_reason" not in column_names:
            logger.info("Adding simulation_reason column to performance_results")
            conn.execute("ALTER TABLE performance_results ADD COLUMN simulation_reason VARCHAR")
            schema_updates.append("Added simulation_reason column to performance_results")
        else:
            logger.info("simulation_reason column already exists in performance_results")
        
        # Check if hardware_availability_log table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        if "hardware_availability_log" not in tables['name'].values:
            # Create hardware_availability_log if it doesn't exist
            logger.info("Creating hardware_availability_log table")
            conn.execute("""
            CREATE TABLE hardware_availability_log (
                id INTEGER PRIMARY KEY,
                hardware_type VARCHAR,
                is_available BOOLEAN,
                detection_method VARCHAR,
                detection_details JSON,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create a sequence for auto-increment
            conn.execute("CREATE SEQUENCE IF NOT EXISTS hardware_availability_log_id_seq")
            schema_updates.append("Created hardware_availability_log table and sequence")
        else:
            logger.info("hardware_availability_log table already exists")
            schema_updates.append("hardware_availability_log table already exists")
        
        # Mark existing data as simulated
        logger.info("Marking existing data as simulated")
        conn.execute("UPDATE performance_results SET is_simulated = TRUE, simulation_reason = 'Legacy data predating simulation tracking'")
        schema_updates.append("Marked all existing performance_results as simulated")
        
        # Close the connection
        conn.close()
        logger.info("Database schema update completed successfully")
        
        return True, schema_updates
    except Exception as e:
        logger.error(f"Failed to update schema: {e}")
        return False, [f"Error: {str(e)}"]

def main():
    """Main function to update the benchmark database schema."""
    parser = argparse.ArgumentParser(description="Update Benchmark Database Schema")
    parser.add_argument("--db-path", type=str, help="Path to the benchmark database (default: uses BENCHMARK_DB_PATH environment variable)")
    parser.add_argument("--no-backup", action="store_true", help="Skip database backup")
    
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    db_path = Path(db_path)
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
    
    # Create backup if requested
    if not args.no_backup:
        if not backup_database(db_path):
            logger.error("Failed to create backup, aborting")
            return 1
    
    # Update the schema
    success, updates = update_schema(db_path)
    
    if success:
        logger.info("Schema update completed successfully:")
        for update in updates:
            logger.info(f"- {update}")
            
        logger.info(f"\nNow all benchmark results in {db_path} have been marked as simulated.")
        logger.info("Future benchmarks will need to explicitly set is_simulated=False to indicate real hardware measurements.")
        
        return 0
    else:
        logger.error("Schema update failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())