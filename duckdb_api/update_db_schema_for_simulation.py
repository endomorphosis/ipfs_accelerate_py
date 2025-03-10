#!/usr/bin/env python3
"""
Database Schema Update for Simulation Flags

This script updates the database schema to add simulation flags to relevant tables.
These flags help clearly identify which benchmark results were generated using
simulated hardware versus real hardware.

April 2025 Update: Part of the benchmark system improvements from NEXT_STEPS.md
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.error("DuckDB not available. Please install it with: pip install duckdb")
    DUCKDB_AVAILABLE = False

def get_db_path():
    """Get database path from environment variable or default"""
    return os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")

def connect_to_db(db_path):
    """Connect to the database"""
    if not DUCKDB_AVAILABLE:
        logger.error("DuckDB not available. Please install it with: pip install duckdb")
        return None
    
    try:
        # Connect to the database
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def check_table_exists(conn, table_name):
    """Check if a table exists in the database"""
    try:
        # Query the information_schema.tables view
        result = conn.execute(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()
        return result is not None
    except Exception as e:
        logger.error(f"Error checking if table {table_name} exists: {e}")
        return False

def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        # Query the information_schema.columns view
        result = conn.execute(
            f"SELECT 1 FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{column_name}'"
        ).fetchone()
        return result is not None
    except Exception as e:
        logger.error(f"Error checking if column {column_name} exists in table {table_name}: {e}")
        return False

def backup_database(db_path):
    """Create a backup of the database before making changes"""
    try:
        backup_path = f"{db_path}.backup_{int(time.time())}"
        # Create a copy of the database file
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created database backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        return None

def update_schema(conn):
    """Update the database schema to add simulation flags"""
    try:
        # Tables to modify
        tables_to_update = [
            {
                "name": "test_results", 
                "columns": [
                    {"name": "is_simulated", "type": "BOOLEAN", "default": "FALSE"},
                    {"name": "simulation_reason", "type": "VARCHAR"}
                ]
            },
            {
                "name": "hardware_platforms", 
                "columns": [
                    {"name": "is_simulated", "type": "BOOLEAN", "default": "FALSE"},
                    {"name": "simulation_reason", "type": "VARCHAR"}
                ]
            },
            {
                "name": "performance_results", 
                "columns": [
                    {"name": "is_simulated", "type": "BOOLEAN", "default": "FALSE"},
                    {"name": "simulation_reason", "type": "VARCHAR"}
                ]
            }
        ]
        
        # Create hardware availability log table
        if not check_table_exists(conn, "hardware_availability_log"):
            logger.info("Creating hardware_availability_log table...")
            conn.execute("""
            CREATE TABLE hardware_availability_log (
                id INTEGER PRIMARY KEY,
                hardware_type VARCHAR,
                is_available BOOLEAN,
                is_simulated BOOLEAN DEFAULT FALSE,
                detection_method VARCHAR,
                detection_details JSON,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info("hardware_availability_log table created successfully")
        else:
            # Check if is_simulated column exists and add it if missing
            if not check_column_exists(conn, "hardware_availability_log", "is_simulated"):
                logger.info("Adding is_simulated column to hardware_availability_log table...")
                conn.execute("""
                ALTER TABLE hardware_availability_log 
                ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE
                """)
                logger.info("Added is_simulated column to hardware_availability_log table")
        
        # Add simulation flags to existing tables
        for table in tables_to_update:
            table_name = table["name"]
            
            if not check_table_exists(conn, table_name):
                logger.warning(f"Table {table_name} does not exist, skipping")
                continue
            
            for column in table["columns"]:
                column_name = column["name"]
                column_type = column["type"]
                column_default = column.get("default", "NULL")
                
                if not check_column_exists(conn, table_name, column_name):
                    logger.info(f"Adding column {column_name} to table {table_name}...")
                    
                    # Add the column
                    conn.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN {column_name} {column_type} DEFAULT {column_default}
                    """)
                    
                    logger.info(f"Column {column_name} added to {table_name}")
                else:
                    logger.info(f"Column {column_name} already exists in table {table_name}, skipping")
        
        # Commit changes
        conn.commit()
        logger.info("Schema update completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        return False

def main():
    """Main entry point for updating database schema"""
    parser = argparse.ArgumentParser(description="Update database schema to add simulation flags")
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database (defaults to BENCHMARK_DB_PATH environment variable)")
    parser.add_argument("--no-backup", action="store_true",
                      help="Skip creating a database backup before making changes")
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path or get_db_path()
    
    # Check if database file exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return 1
    
    # Create backup unless --no-backup is specified
    if not args.no_backup:
        backup_path = backup_database(db_path)
        if not backup_path:
            logger.error("Failed to create database backup, aborting")
            return 1
    
    # Connect to database
    conn = connect_to_db(db_path)
    if not conn:
        return 1
    
    # Update schema
    success = update_schema(conn)
    
    # Close connection
    conn.close()
    
    if success:
        logger.info("Database schema update completed successfully")
        return 0
    else:
        logger.error("Database schema update failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())