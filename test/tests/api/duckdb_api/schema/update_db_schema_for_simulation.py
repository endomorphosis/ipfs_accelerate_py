#!/usr/bin/env python3
"""
Database schema update script for simulation tracking and reporting.

This script adds simulation tracking columns and tables to the DuckDB benchmark database,
enabling transparent tracking of simulated vs real hardware results.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Error: Required package not installed. Please install with:")
    print("pip install duckdb")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def update_schema(db_path: str, force: bool = False) -> bool:
    """
    Update the database schema to add simulation tracking.
    
    Args:
        db_path: Path to the DuckDB database
        force: Force update even if columns already exist
        
    Returns:
        True if update was successful, False otherwise
    """
    logger.info(f"Updating schema for database: {db_path}")
    
    if not Path(db_path).exists():
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = duckdb.connect(db_path)
        
        # Check if performance_results table exists
        result = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='performance_results'
        """).fetchall()
        
        if not result:
            logger.error("performance_results table not found. Please initialize the database first.")
            return False
        
        # Check if simulation columns already exist
        has_simulation_columns = False
        if not force:
            result = conn.execute("""
                PRAGMA table_info(performance_results)
            """).fetchall()
            
            column_names = [col[1] for col in result]
            has_simulation_columns = 'is_simulated' in column_names and 'simulation_reason' in column_names
        
        if has_simulation_columns and not force:
            logger.info("Simulation columns already exist in performance_results table.")
        else:
            # Add is_simulated and simulation_reason columns to performance_results
            logger.info("Adding simulation columns to performance_results table...")
            
            try:
                conn.execute("""
                    ALTER TABLE performance_results 
                    ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE
                """)
                logger.info("Added is_simulated column to performance_results table.")
            except Exception as e:
                logger.warning(f"Error adding is_simulated column (may already exist): {e}")
            
            try:
                conn.execute("""
                    ALTER TABLE performance_results 
                    ADD COLUMN simulation_reason VARCHAR
                """)
                logger.info("Added simulation_reason column to performance_results table.")
            except Exception as e:
                logger.warning(f"Error adding simulation_reason column (may already exist): {e}")
        
        # Check if hardware_compatibility table exists
        result = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='hardware_compatibility'
        """).fetchall()
        
        if result:
            # Check if simulation columns already exist
            has_simulation_columns = False
            if not force:
                result = conn.execute("""
                    PRAGMA table_info(hardware_compatibility)
                """).fetchall()
                
                column_names = [col[1] for col in result]
                has_simulation_columns = 'is_simulated' in column_names and 'simulation_reason' in column_names
            
            if has_simulation_columns and not force:
                logger.info("Simulation columns already exist in hardware_compatibility table.")
            else:
                # Add is_simulated and simulation_reason columns to hardware_compatibility
                logger.info("Adding simulation columns to hardware_compatibility table...")
                
                try:
                    conn.execute("""
                        ALTER TABLE hardware_compatibility 
                        ADD COLUMN is_simulated BOOLEAN DEFAULT FALSE
                    """)
                    logger.info("Added is_simulated column to hardware_compatibility table.")
                except Exception as e:
                    logger.warning(f"Error adding is_simulated column (may already exist): {e}")
                
                try:
                    conn.execute("""
                        ALTER TABLE hardware_compatibility 
                        ADD COLUMN simulation_reason VARCHAR
                    """)
                    logger.info("Added simulation_reason column to hardware_compatibility table.")
                except Exception as e:
                    logger.warning(f"Error adding simulation_reason column (may already exist): {e}")
        
        # Create hardware_availability_log table if it doesn't exist
        result = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='hardware_availability_log'
        """).fetchall()
        
        if not result:
            logger.info("Creating hardware_availability_log table...")
            
            conn.execute("""
                CREATE TABLE hardware_availability_log (
                    log_id INTEGER PRIMARY KEY,
                    hardware_id INTEGER NOT NULL,
                    is_available BOOLEAN NOT NULL,
                    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason VARCHAR,
                    detected_on_host VARCHAR,
                    FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id)
                )
            """)
            logger.info("Created hardware_availability_log table.")
        else:
            logger.info("hardware_availability_log table already exists.")
        
        conn.close()
        
        logger.info("Schema update for simulation tracking completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        return False


def main():
    """Command-line interface for the schema update script."""
    parser = argparse.ArgumentParser(description="Update Database Schema for Simulation Tracking")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--force", action="store_true",
                       help="Force update even if columns already exist")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Use environment variable for database path if available
    db_path = args.db_path
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    # Update the schema
    success = update_schema(db_path, args.force)
    
    if not success:
        sys.exit(1)
    
    logger.info("Schema update completed successfully.")

if __name__ == "__main__":
    main()