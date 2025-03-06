#!/usr/bin/env python
"""
Implement the database schema enhancements outlined in NEXT_STEPS.md

This script adds:
1. Extended Model Metadata
2. Advanced Performance Metrics
3. Hardware Platform Relationships
4. Time-Series Performance Tracking
5. Mobile/Edge Device Metrics

Usage:
    python scripts/implement_db_schema_enhancements.py --db-path ./benchmark_db.duckdb
"""

import os
import sys
import argparse
import json
import datetime
import duckdb
import pandas as pd
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description="Implement database schema enhancements")
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb", 
                       help="Path to the DuckDB database")
    parser.add_argument("--force", action="store_true", 
                       help="Force recreate tables even if they exist")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed logging information")
    parser.add_argument("--backfill", action="store_true",
                       help="Backfill existing data to new schema where applicable")
    return parser.parse_args()

def connect_to_db(db_path):
    """Connect to DuckDB database"""
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Connect to the database
    return duckdb.connect(db_path)

def add_extended_model_metadata(conn, force=False):
    """
    Extend the models table with additional metadata columns
    """
    print("Adding extended model metadata...")
    
    # Check if columns already exist
    existing_columns = conn.execute("PRAGMA table_info(models)").fetchall()
    existing_column_names = [col[1] for col in existing_columns]
    
    # Columns to add
    new_columns = {
        "architecture": "VARCHAR",
        "parameter_efficiency_score": "FLOAT",
        "last_benchmark_date": "TIMESTAMP",
        "version_history": "JSON",
        "model_capabilities": "JSON",
        "licensing_info": "TEXT",
        "compatibility_matrix": "JSON"
    }
    
    # Add columns that don't exist yet
    for col_name, col_type in new_columns.items():
        if col_name not in existing_column_names:
            sql = f"ALTER TABLE models ADD COLUMN {col_name} {col_type}"
            print(f"  Adding column: {col_name} ({col_type})")
            conn.execute(sql)
        else:
            print(f"  Column already exists: {col_name}")
    
    print("Extended model metadata added successfully.")

def create_advanced_performance_metrics(conn, force=False):
    """
    Create the advanced performance metrics table
    """
    print("Creating advanced performance metrics table...")
    
    if force:
        conn.execute("DROP TABLE IF EXISTS performance_extended_metrics")
    
    sql = """
    CREATE TABLE IF NOT EXISTS performance_extended_metrics (
        id INTEGER PRIMARY KEY,
        performance_id INTEGER,
        memory_breakdown JSON,
        cpu_utilization_percent FLOAT,
        gpu_utilization_percent FLOAT,
        io_wait_ms FLOAT,
        inference_breakdown JSON,
        power_consumption_watts FLOAT,
        thermal_metrics JSON,
        memory_bandwidth_gbps FLOAT,
        compute_efficiency_percent FLOAT,
        FOREIGN KEY (performance_id) REFERENCES performance_results(id)
    )
    """
    
    conn.execute(sql)
    print("Advanced performance metrics table created successfully.")

def create_hardware_platform_relationships(conn, force=False):
    """
    Create the hardware platform relationships table
    """
    print("Creating hardware platform relationships table...")
    
    if force:
        conn.execute("DROP TABLE IF EXISTS hardware_platform_relationships")
    
    sql = """
    CREATE TABLE IF NOT EXISTS hardware_platform_relationships (
        id INTEGER PRIMARY KEY,
        source_hardware_id INTEGER,
        target_hardware_id INTEGER,
        performance_ratio FLOAT,
        relationship_type VARCHAR,
        confidence_score FLOAT,
        last_validated TIMESTAMP,
        validation_method VARCHAR,
        notes TEXT,
        FOREIGN KEY (source_hardware_id) REFERENCES hardware_platforms(hardware_id),
        FOREIGN KEY (target_hardware_id) REFERENCES hardware_platforms(hardware_id)
    )
    """
    
    conn.execute(sql)
    print("Hardware platform relationships table created successfully.")

def create_time_series_performance_tracking(conn, force=False):
    """
    Create the time-series performance tracking table
    """
    print("Creating time-series performance tracking table...")
    
    if force:
        conn.execute("DROP TABLE IF EXISTS performance_history")
    
    sql = """
    CREATE TABLE IF NOT EXISTS performance_history (
        id INTEGER PRIMARY KEY,
        model_id INTEGER,
        hardware_id INTEGER,
        batch_size INTEGER,
        timestamp TIMESTAMP,
        git_commit_hash VARCHAR,
        throughput_items_per_second FLOAT,
        latency_ms FLOAT,
        memory_mb FLOAT,
        power_watts FLOAT,
        baseline_performance_id INTEGER,
        regression_detected BOOLEAN,
        regression_severity VARCHAR,
        notes TEXT,
        FOREIGN KEY (model_id) REFERENCES models(model_id),
        FOREIGN KEY (hardware_id) REFERENCES hardware_platforms(hardware_id),
        FOREIGN KEY (baseline_performance_id) REFERENCES performance_history(id)
    )
    """
    
    conn.execute(sql)
    print("Time-series performance tracking table created successfully.")

def create_mobile_edge_metrics(conn, force=False):
    """
    Create the mobile/edge device metrics table
    """
    print("Creating mobile/edge device metrics table...")
    
    if force:
        conn.execute("DROP TABLE IF EXISTS mobile_edge_metrics")
    
    sql = """
    CREATE TABLE IF NOT EXISTS mobile_edge_metrics (
        id INTEGER PRIMARY KEY,
        performance_id INTEGER,
        device_model VARCHAR,
        battery_impact_percent FLOAT,
        thermal_throttling_detected BOOLEAN,
        thermal_throttling_duration_seconds INTEGER,
        battery_temperature_celsius FLOAT,
        soc_temperature_celsius FLOAT,
        power_efficiency_score FLOAT,
        startup_time_ms FLOAT,
        runtime_memory_profile JSON,
        FOREIGN KEY (performance_id) REFERENCES performance_results(id)
    )
    """
    
    conn.execute(sql)
    print("Mobile/edge device metrics table created successfully.")

def create_views(conn):
    """Create additional views for the new tables"""
    print("Creating views for new tables...")
    
    # Performance history view
    try:
        conn.execute("""
        CREATE OR REPLACE VIEW performance_history_summary AS
        SELECT 
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            ph.batch_size,
            AVG(ph.throughput_items_per_second) as avg_throughput,
            AVG(ph.latency_ms) as avg_latency,
            AVG(ph.memory_mb) as avg_memory,
            COUNT(CASE WHEN ph.regression_detected THEN 1 END) as regression_count,
            MIN(ph.timestamp) as first_benchmark,
            MAX(ph.timestamp) as last_benchmark,
            COUNT(*) as benchmark_count
        FROM 
            performance_history ph
        JOIN 
            models m ON ph.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON ph.hardware_id = hp.hardware_id
        GROUP BY 
            m.model_name, m.model_family, hp.hardware_type, hp.device_name, ph.batch_size
        """)
        print("  Created performance_history_summary view")
    except Exception as e:
        print(f"  Warning: Could not create performance_history_summary view: {e}")
    
    # Hardware relationship view
    try:
        conn.execute("""
        CREATE OR REPLACE VIEW hardware_relationship_matrix AS
        SELECT 
            hp_source.hardware_type as source_hardware,
            hp_source.device_name as source_device,
            hp_target.hardware_type as target_hardware,
            hp_target.device_name as target_device,
            hpr.performance_ratio,
            hpr.relationship_type,
            hpr.confidence_score,
            hpr.last_validated
        FROM 
            hardware_platform_relationships hpr
        JOIN 
            hardware_platforms hp_source ON hpr.source_hardware_id = hp_source.hardware_id
        JOIN 
            hardware_platforms hp_target ON hpr.target_hardware_id = hp_target.hardware_id
        """)
        print("  Created hardware_relationship_matrix view")
    except Exception as e:
        print(f"  Warning: Could not create hardware_relationship_matrix view: {e}")
    
    # Mobile/edge metrics view
    try:
        conn.execute("""
        CREATE OR REPLACE VIEW mobile_edge_performance AS
        SELECT 
            m.model_name,
            m.model_family,
            hp.hardware_type,
            hp.device_name,
            mem.device_model,
            pr.batch_size,
            pr.precision,
            pr.average_latency_ms,
            pr.throughput_items_per_second,
            pr.memory_peak_mb,
            mem.battery_impact_percent,
            mem.thermal_throttling_detected,
            mem.power_efficiency_score,
            mem.startup_time_ms
        FROM 
            mobile_edge_metrics mem
        JOIN 
            performance_results pr ON mem.performance_id = pr.result_id
        JOIN 
            models m ON pr.model_id = m.model_id
        JOIN 
            hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        """)
        print("  Created mobile_edge_performance view")
    except Exception as e:
        print(f"  Warning: Could not create mobile_edge_performance view: {e}")
    
    # Extended model view
    try:
        conn.execute("""
        CREATE OR REPLACE VIEW extended_model_info AS
        SELECT 
            model_id,
            model_name,
            model_family,
            modality,
            architecture,
            parameters_million,
            parameter_efficiency_score,
            last_benchmark_date,
            version,
            licensing_info,
            model_capabilities
        FROM 
            models
        """)
        print("  Created extended_model_info view")
    except Exception as e:
        print(f"  Warning: Could not create extended_model_info view: {e}")

def backfill_existing_data(conn):
    """Backfill existing data to the new tables where applicable"""
    print("Backfilling existing data to new tables...")
    
    # Check if we have any performance results to backfill
    try:
        perf_count = conn.execute("""
        SELECT COUNT(*) FROM performance_results
        """).fetchone()[0]
        
        if perf_count > 0:
            # Backfill performance history from performance_results
            conn.execute("""
            INSERT INTO performance_history (
                model_id, hardware_id, batch_size, timestamp, 
                throughput_items_per_second, latency_ms, memory_mb,
                regression_detected, notes
            )
            SELECT 
                model_id, hardware_id, batch_size, created_at,
                throughput_items_per_second, average_latency_ms, memory_peak_mb,
                FALSE, 'Backfilled from historical data'
            FROM 
                performance_results
            WHERE
                (model_id, hardware_id) NOT IN (
                    SELECT model_id, hardware_id FROM performance_history
                )
            """)
            
            # Get count of backfilled records
            backfill_count = conn.execute("""
            SELECT COUNT(*) FROM performance_history
            """).fetchone()[0]
            
            print(f"  Backfilled {backfill_count} records to performance_history")
            
            # Update models with last_benchmark_date
            conn.execute("""
            UPDATE models 
            SET last_benchmark_date = subquery.last_date
            FROM (
                SELECT model_id, MAX(created_at) as last_date 
                FROM performance_results 
                GROUP BY model_id
            ) as subquery
            WHERE models.model_id = subquery.model_id
            """)
            
            print(f"  Updated last_benchmark_date in models table")
            
        else:
            print("  No performance results to backfill")
            
    except Exception as e:
        print(f"  Warning: Error backfilling data: {e}")

def main():
    args = parse_args()
    
    print(f"Implementing database schema enhancements on: {args.db_path}")
    conn = connect_to_db(args.db_path)
    
    # Check if database exists
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        if not tables:
            print("Error: Database schema not initialized. Please run create_benchmark_schema.py first.")
            return
    except Exception as e:
        print(f"Error accessing database: {e}")
        print("Please make sure the database exists and run create_benchmark_schema.py first.")
        return
    
    # Implement the schema enhancements
    add_extended_model_metadata(conn, args.force)
    create_advanced_performance_metrics(conn, args.force)
    create_hardware_platform_relationships(conn, args.force)
    create_time_series_performance_tracking(conn, args.force)
    create_mobile_edge_metrics(conn, args.force)
    create_views(conn)
    
    # Backfill existing data if requested
    if args.backfill:
        backfill_existing_data(conn)
    
    # Display schema counts for verification
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"\nDatabase now has {len(tables)} tables and views:")
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
            print(f"  - {table[0]}: {count} rows")
        except Exception as e:
            print(f"  - {table[0]}: Error getting count: {e}")
    
    conn.close()
    print("\nDatabase schema enhancements implemented successfully.")

if __name__ == "__main__":
    main()