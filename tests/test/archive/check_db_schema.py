#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Database Schema

This script checks the database schema to understand the structure of the tables
and columns available for proper querying.
"""

import os
import sys
import logging
import duckdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_tables(conn):
    """Get all tables in the database."""
    tables = conn.execute("SHOW TABLES").fetchdf()
    return tables

def get_table_schema(conn, table_name):
    """Get the schema for a specific table."""
    schema = conn.execute(f"PRAGMA table_info({table_name})").fetchdf()
    return schema

def get_table_data(conn, table_name, limit=5):
    """Get sample data from a table."""
    try:
        data = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchdf()
        return data
    except Exception as e:
        logger.error(f"Error querying table {table_name}: {e}")
        return None

def get_performance_results(conn):
    """Get latest performance results with model name."""
    try:
        # First, let's check if there's a join needed to get model names
        schema = get_table_schema(conn, "performance_results")
        has_model_name = "model_name" in schema["name"].values
        
        if has_model_name:
            # Direct query if model_name is in the table
            query = """
            SELECT *
            FROM performance_results
            ORDER BY created_at DESC
            LIMIT 10
            """
        else:
            # We need to join with models table to get model names
            query = """
            SELECT pr.*, m.model_name
            FROM performance_results pr
            JOIN models m ON pr.model_id = m.model_id
            ORDER BY pr.created_at DESC
            LIMIT 10
            """
        
        data = conn.execute(query).fetchdf()
        return data
    except Exception as e:
        logger.error(f"Error querying performance results: {e}")
        return None

def main():
    """Main function to check database schema."""
    # Connect to the database
    db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    try:
        conn = duckdb.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Get all tables
        tables = get_all_tables(conn)
        logger.info(f"Tables in database: {', '.join(tables['name'])}")
        
        # Check schema for each table
        for table_name in tables["name"]:
            schema = get_table_schema(conn, table_name)
            logger.info(f"Schema for {table_name}:")
            for _, row in schema.iterrows():
                logger.info(f"  {row['name']} ({row['type']})" + (" PRIMARY KEY" if row["pk"] else ""))
            
            # Get sample data
            data = get_table_data(conn, table_name)
            if data is not None and not data.empty:
                logger.info(f"Sample data for {table_name}:")
                logger.info(f"  {data.to_string(index=False)}")
            else:
                logger.info(f"No data in {table_name} or error occurred")
            
            logger.info("")
        
        # Get latest performance results
        logger.info("Checking performance results with model names:")
        results = get_performance_results(conn)
        if results is not None and not results.empty:
            logger.info(f"Latest performance results:")
            logger.info(f"  {results.to_string(index=False)}")
        else:
            logger.info("No performance results or error occurred")
        
        # Close the connection
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking database schema: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())