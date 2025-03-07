#!/usr/bin/env python3
"""
View Benchmark Results

This script queries the benchmark database and displays results in a simple format.
It provides a way to verify that benchmark data is being properly stored in the database
with the simulation flags implemented in item #10.

Usage:
    python view_benchmark_results.py
    python view_benchmark_results.py --output benchmark_summary.md
    python view_benchmark_results.py --format csv --output benchmark_data.csv
"""

import os
import sys
import argparse
import logging
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_path():
    """Get database path from environment variable or default"""
    return os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")

def connect_to_db(db_path):
    """Connect to the database"""
    try:
        # Connect to the database
        conn = duckdb.connect(db_path, read_only=True)
        logger.info(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_performance_results(conn):
    """Get performance results from the database"""
    try:
        query = """
        SELECT 
            m.model_name,
            m.model_family,
            hp.hardware_type,
            pr.batch_size,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb,
            pr.is_simulated,
            pr.simulation_reason,
            pr.test_timestamp
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        ORDER BY
            pr.test_timestamp DESC
        """
        
        result = conn.execute(query).fetchdf()
        return result
    except Exception as e:
        logger.error(f"Error fetching performance results: {e}")
        return pd.DataFrame()

def get_hardware_platforms(conn):
    """Get hardware platforms from the database"""
    try:
        # First check the schema to see what columns are available
        schema_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'hardware_platforms'
        """
        columns = conn.execute(schema_query).fetchall()
        column_names = [col[0] for col in columns]
        logger.info(f"Available columns in hardware_platforms: {column_names}")
        
        # Build query based on available columns
        has_description = "description" in column_names
        
        # Build the query accordingly
        if has_description:
            query = """
            SELECT 
                hardware_id,
                hardware_type,
                description,
                is_simulated,
                simulation_reason
            FROM 
                hardware_platforms
            """
        else:
            query = """
            SELECT 
                hardware_id,
                hardware_type,
                device_name as description,  -- Use device_name as fallback for description
                is_simulated,
                simulation_reason
            FROM 
                hardware_platforms
            """
        
        result = conn.execute(query).fetchdf()
        return result
    except Exception as e:
        logger.error(f"Error fetching hardware platforms: {e}")
        return pd.DataFrame()

def generate_markdown_report(performance_df, hardware_df, output_path):
    """Generate a markdown report with benchmark results"""
    with open(output_path, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Hardware Platforms Section
        f.write("## Hardware Platforms\n\n")
        f.write("| ID | Type | Description | Simulated? | Simulation Reason |\n")
        f.write("|---|---|---|---|---|\n")
        
        for _, row in hardware_df.iterrows():
            simulated = "✅ Yes" if row['is_simulated'] else "❌ No"
            reason = row['simulation_reason'] or "N/A"
            f.write(f"| {row['hardware_id']} | {row['hardware_type']} | {row['description'] or 'N/A'} | {simulated} | {reason} |\n")
        
        f.write("\n")
        
        # Performance Results Section
        f.write("## Recent Performance Results\n\n")
        f.write("| Model | Family | Hardware | Batch Size | Latency (ms) | Throughput (items/s) | Memory (MB) | Simulated? | Timestamp |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        
        # Limit to 20 most recent results for readability
        for _, row in performance_df.head(20).iterrows():
            simulated = "✅ Yes" if row['is_simulated'] else "❌ No"
            timestamp = row['test_timestamp'].strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(row['test_timestamp']) else "N/A"
            
            f.write(f"| {row['model_name']} | {row['model_family']} | {row['hardware_type']} | {row['batch_size']} | ")
            f.write(f"{row['average_latency_ms']:.2f} | {row['throughput_items_per_second']:.2f} | ")
            f.write(f"{row['memory_peak_mb']:.2f} | {simulated} | {timestamp} |\n")
        
        f.write("\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        
        # Count by Hardware Type
        f.write("### Results by Hardware Type\n\n")
        hardware_counts = performance_df['hardware_type'].value_counts()
        f.write("| Hardware Type | Result Count |\n")
        f.write("|---|---|\n")
        for hw, count in hardware_counts.items():
            f.write(f"| {hw} | {count} |\n")
        
        f.write("\n")
        
        # Count by Model Family
        f.write("### Results by Model Family\n\n")
        model_counts = performance_df['model_family'].value_counts()
        f.write("| Model Family | Result Count |\n")
        f.write("|---|---|\n")
        for model, count in model_counts.items():
            f.write(f"| {model} | {count} |\n")
        
        f.write("\n")
        
        # Simulation Status
        f.write("### Simulation Status\n\n")
        sim_counts = performance_df['is_simulated'].value_counts()
        f.write("| Simulation Status | Result Count |\n")
        f.write("|---|---|\n")
        for status, count in sim_counts.items():
            status_str = "Simulated" if status else "Real"
            f.write(f"| {status_str} | {count} |\n")
        
        logger.info(f"Markdown report generated: {output_path}")

def generate_csv_report(performance_df, output_path):
    """Generate a CSV report with benchmark results"""
    performance_df.to_csv(output_path, index=False)
    logger.info(f"CSV report generated: {output_path}")

def get_simulation_status(conn):
    """Query to check if simulation flags are set correctly"""
    try:
        # First check schema to find the ID column name
        columns_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'performance_results'
        """
        columns = conn.execute(columns_query).fetchall()
        column_names = [col[0] for col in columns]
        logger.info(f"Available columns in performance_results: {column_names}")
        
        # Use the correct ID column based on schema
        id_column = "result_id" if "result_id" in column_names else "id"
        
        # Check performance_results
        query1 = f"""
        SELECT 
            pr.{id_column}, 
            m.model_name, 
            hp.hardware_type, 
            pr.is_simulated, 
            pr.simulation_reason
        FROM 
            performance_results pr
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE 
            pr.is_simulated = TRUE
        LIMIT 10
        """
        
        # Check hardware_platforms
        query2 = """
        SELECT 
            hardware_id,
            hardware_type, 
            is_simulated, 
            simulation_reason
        FROM 
            hardware_platforms
        """
        
        result1 = conn.execute(query1).fetchdf()
        result2 = conn.execute(query2).fetchdf()
        
        return result1, result2
    except Exception as e:
        logger.error(f"Error checking simulation status: {e}")
        return pd.DataFrame(), pd.DataFrame()

def main():
    """Main entry point for viewing benchmark results"""
    parser = argparse.ArgumentParser(description="View benchmark results from database")
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database (defaults to BENCHMARK_DB_PATH environment variable)")
    parser.add_argument("--output", type=str, default="benchmark_summary.md",
                      help="Output file for the report")
    parser.add_argument("--format", choices=["markdown", "md", "csv"], default="markdown",
                      help="Output format for the report")
    parser.add_argument("--check-simulation", action="store_true",
                      help="Check if simulation flags are set correctly")
    args = parser.parse_args()
    
    # Get database path
    db_path = args.db_path or get_db_path()
    
    # Check if database file exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return 1
    
    # Connect to database
    conn = connect_to_db(db_path)
    if not conn:
        return 1
    
    # Check simulation status if requested
    if args.check_simulation:
        simulated_results, hardware_platforms = get_simulation_status(conn)
        
        if not simulated_results.empty:
            print("\nSimulated Performance Results:")
            print("-" * 80)
            print(simulated_results.to_string(index=False))
        else:
            print("\nNo simulated performance results found.")
        
        if not hardware_platforms.empty:
            print("\nHardware Platforms Simulation Status:")
            print("-" * 80)
            print(hardware_platforms.to_string(index=False))
        else:
            print("\nNo hardware platforms found.")
        
        print("\nTo fix simulation flags, you can run:")
        print("UPDATE performance_results SET is_simulated = TRUE, simulation_reason = 'Manual test' WHERE id = X;")
        print("UPDATE hardware_platforms SET is_simulated = TRUE, simulation_reason = 'Hardware not available' WHERE hardware_type = 'X';")
        
        # Close connection and exit
        conn.close()
        return 0
    
    # Get data for report
    performance_df = get_performance_results(conn)
    hardware_df = get_hardware_platforms(conn)
    
    # Close connection
    conn.close()
    
    # Generate report
    if performance_df.empty:
        logger.error("No performance results found in database")
        return 1
    
    if args.format in ["markdown", "md"]:
        generate_markdown_report(performance_df, hardware_df, args.output)
    elif args.format == "csv":
        generate_csv_report(performance_df, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())