#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify database schema to diagnose issues
"""

import duckdb
import sys
from pathlib import Path

def check_schema(db_path):
    """Check database schema and print tables and columns"""
    try:
        # Connect to database in read-only mode
        conn = duckdb.connect(db_path, read_only=True)
        
        # Get table list
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Found {len(tables)} tables in database {db_path}:")
        
        # Check each table schema
        for table in tables:
            table_name = table[0]
            print(f"\n=== TABLE: {table_name} ===")
            
            # Get column information
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print("Columns:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} (nullable: {col[3]})")
            
            # Count rows
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"Row count: {count}")
            
            # Sample data if available
            if count > 0:
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchone()
                print("Sample data:")
                for idx, col in enumerate(columns):
                    if idx < len(sample):
                        print(f"  - {col[0]}: {sample[idx]}")
            
        # Close connection
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False

if __name__ == "__main__":
    # Default to benchmark_db.duckdb in current directory, or use provided path
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_db.duckdb"
    check_schema(db_path)