#!/usr/bin/env python3
"""
Create Error Reporting Schema in DuckDB Database.

This script creates the error reporting schema in a DuckDB database.
It can be used to initialize a new database or update an existing one.

Usage:
    python -m duckdb_api.distributed_testing.schema.create_error_schema --db-path ./benchmark_db.duckdb
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_error_schema")

def create_error_schema(db_path: str) -> bool:
    """Create the error reporting schema in a DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        True if schema was created successfully, False otherwise
    """
    try:
        import duckdb
        
        # Get schema SQL
        schema_path = Path(__file__).parent / "error_reporting_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Check if any of the tables already exist
        result = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'worker_error_reports'
        """).fetchone()
        
        if result:
            logger.info(f"Error schema already exists in {db_path}")
            logger.info("Use --force to recreate the schema")
            return False
        
        # Execute schema SQL
        conn.execute(schema_sql)
        conn.close()
        
        logger.info(f"Error schema created successfully in {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        return False

def main():
    """Create error schema in DuckDB database."""
    parser = argparse.ArgumentParser(description="Create Error Reporting Schema in DuckDB Database")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database file")
    parser.add_argument("--force", action="store_true", help="Force recreation of schema if it already exists")
    
    args = parser.parse_args()
    
    # Ensure the directory exists
    db_dir = os.path.dirname(args.db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    if args.force:
        try:
            import duckdb
            
            # Drop existing tables if force flag is set
            conn = duckdb.connect(args.db_path)
            
            # Get schema SQL for recreating
            schema_path = Path(__file__).parent / "error_reporting_schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Drop existing tables
            tables_to_drop = [
                "worker_error_reports",
                "error_categories_summary",
                "error_patterns",
                "worker_error_statistics",
                "hardware_error_statistics"
            ]
            
            # Drop views first (they depend on tables)
            views_to_drop = [
                "recurring_errors",
                "critical_errors",
                "hardware_errors",
                "resource_errors",
                "network_errors",
                "worker_errors",
                "test_errors",
                "error_summary_statistics"
            ]
            
            for view in views_to_drop:
                try:
                    conn.execute(f"DROP VIEW IF EXISTS {view}")
                except:
                    pass
            
            for table in tables_to_drop:
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                except:
                    pass
            
            # Execute schema SQL
            conn.execute(schema_sql)
            conn.close()
            
            logger.info(f"Error schema recreated successfully in {args.db_path}")
            return 0
            
        except Exception as e:
            logger.error(f"Error recreating schema: {e}")
            return 1
    
    # Create schema
    success = create_error_schema(args.db_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())