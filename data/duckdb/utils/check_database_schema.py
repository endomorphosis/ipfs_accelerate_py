#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the database schema to understand column names
"""

import sys
import duckdb
import json

def check_schema(db_path):
    """Check database schema"""
    try:
        conn = duckdb.connect(db_path)
        
        # Get all tables
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        # Check schema for each table
        for table in tables:
            table_name = table[0]
            print(f"\nSchema for {table_name}:")
            
            schema = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = []
            for col in schema:
                print(f"  {col[0]}: {col[1]} ({col[2]}) {'PRIMARY KEY' if col[5] else ''}")
                columns.append(col[1])
            
            print(f"Columns: {columns}")
            
            # Sample data
            try:
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchall()
                if sample:
                    print(f"Sample data: {sample[0]}")
            except:
                print("No data or error fetching sample")
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error checking schema: {e}")
        return False

def main():
    """Main function"""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./benchmark_db.duckdb"
    check_schema(db_path)

if __name__ == "__main__":
    main()