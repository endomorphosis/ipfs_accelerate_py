#!/usr/bin/env python
"""
Benchmark Database Query Tool

This script provides a command-line interface for querying the benchmark database,
generating reports, and extracting insights from the test results.
"""

import os
import sys
import json
import argparse
import datetime
import logging
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_query")

def parse_args():
    parser = argparse.ArgumentParser(description="Query benchmark database and generate reports")
    
    # Database connection
    parser.add_argument("--db", type=str, default="./benchmark_db.duckdb", 
                        help="Path to DuckDB database")
    
    # Query types
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--sql", type=str, 
                             help="Execute a custom SQL query")
    query_group.add_argument("--report", type=str, choices=['performance', 'hardware', 'integration', 'summary', 'web_platform', 'webgpu'],
                             help="Generate a predefined report")
    query_group.add_argument("--model", type=str,
                             help="Query data for a specific model")
    query_group.add_argument("--hardware", type=str,
                             help="Query data for a specific hardware type")
    query_group.add_argument("--compatibility-matrix", action="store_true",
                             help="Generate model-hardware compatibility matrix")
    query_group.add_argument("--trend", type=str, choices=['performance', 'compatibility'],
                             help="Show trends over time")
    
    # Filters
    parser.add_argument("--family", type=str,
                        help="Filter by model family (bert, t5, gpt, etc.)")
    parser.add_argument("--metric", type=str, 
                        help="Specific metric to query (throughput, latency, etc.)")
    parser.add_argument("--since", type=str,
                        help="Only include results since date (YYYY-MM-DD)")
    parser.add_argument("--compare-hardware", action="store_true",
                        help="Compare results across hardware platforms")
    
    # Output options
    parser.add_argument("--output", type=str,
                        help="Output file for reports")
    parser.add_argument("--format", type=str, choices=['csv', 'json', 'html', 'markdown', 'chart'],
                        default='markdown', help="Output format")
    parser.add_argument("--limit", type=int, default=100,
                        help="Limit number of results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    return parser.parse_args()

def connect_to_db(db_path):
    """Connect to the DuckDB database"""
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)
        
    try:
        conn = duckdb.connect(db_path)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        sys.exit(1)

def check_tables_exist(conn):
    """Verify that expected tables exist in the database"""
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0].lower() for t in tables]
    
    required_tables = ['hardware_platforms', 'models', 'test_runs']
    missing_tables = [t for t in required_tables if t.lower() not in table_names]
    
    if missing_tables:
        logger.error(f"Required tables missing from database: {', '.join(missing_tables)}")
        logger.error("Please run create_benchmark_schema.py to initialize the database schema")
        sys.exit(1)
    
    return table_names

def execute_sql_query(conn, sql, args=None):
    """Execute a SQL query and return the results"""
    try:
        if args:
            return conn.execute(sql, args).fetchdf()
        else:
            return conn.execute(sql).fetchdf()
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        logger.error(f"SQL: {sql}")
        sys.exit(1)

def generate_performance_report(conn, args):
    """Generate a performance benchmark report"""
    # Construct the WHERE clause based on filters
    where_clauses = []
    query_params = []
    
    if args.family:
        where_clauses.append("m.model_family = ?")
        query_params.append(args.family)
    
    if args.model:
        where_clauses.append("m.model_name LIKE ?")
        query_params.append(f"%{args.model}%")
    
    if args.hardware:
        where_clauses.append("hp.hardware_type = ?")
        query_params.append(args.hardware)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime(args.since, '%Y-%m-%d')
            where_clauses.append("pr.created_at >= ?")
            query_params.append(since_date)
        except ValueError:
            logger.warning(f"Invalid date format for --since: {args.since}. Expected YYYY-MM-DD")
    
    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Construct the query
    sql = f"""
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        pr.test_case,
        pr.batch_size,
        pr.precision,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    {where_clause}
    ORDER BY 
        pr.created_at DESC
    LIMIT ?
    """
    
    query_params.append(args.limit)
    
    # Execute the query
    df = execute_sql_query(conn, sql, query_params)
    
    if df.empty:
        logger.warning("No performance results found matching the criteria")
        return None
    
    # If comparing hardware, pivot the data
    if args.compare_hardware and args.metric:
        metric_col = None
        if args.metric.lower() in ['throughput', 'throughput_items_per_second']:
            metric_col = 'throughput_items_per_second'
        elif args.metric.lower() in ['latency', 'average_latency_ms']:
            metric_col = 'average_latency_ms'
        elif args.metric.lower() in ['memory', 'memory_peak_mb']:
            metric_col = 'memory_peak_mb'
        
        if metric_col:
            # Group and pivot
            pivot_df = df.pivot_table(
                index=['model_name', 'model_family', 'batch_size', 'precision'],
                columns='hardware_type',
                values=metric_col,
                aggfunc='mean'
            )
            
            # Reset index for better display
            pivot_df = pivot_df.reset_index()
            
            if args.format == 'chart':
                return create_hardware_comparison_chart(pivot_df, metric_col, args)
            
            return pivot_df
    
    return df

def generate_hardware_report(conn, args):
    """Generate a hardware compatibility report"""
    # Construct the WHERE clause based on filters
    where_clauses = []
    query_params = []
    
    if args.family:
        where_clauses.append("m.model_family = ?")
        query_params.append(args.family)
    
    if args.model:
        where_clauses.append("m.model_name LIKE ?")
        query_params.append(f"%{args.model}%")
    
    if args.hardware:
        where_clauses.append("hp.hardware_type = ?")
        query_params.append(args.hardware)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime(args.since, '%Y-%m-%d')
            where_clauses.append("hc.created_at >= ?")
            query_params.append(since_date)
        except ValueError:
            logger.warning(f"Invalid date format for --since: {args.since}. Expected YYYY-MM-DD")
    
    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Construct the query
    sql = f"""
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hp.device_name,
        hc.is_compatible,
        hc.compatibility_score,
        hc.error_message,
        hc.suggested_fix,
        hc.created_at
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    {where_clause}
    ORDER BY 
        hc.created_at DESC
    LIMIT ?
    """
    
    query_params.append(args.limit)
    
    # Execute the query
    df = execute_sql_query(conn, sql, query_params)
    
    if df.empty:
        logger.warning("No hardware compatibility results found matching the criteria")
        return None
    
    # If generating a compatibility matrix
    if args.compatibility_matrix or (args.format == 'chart' and not args.metric):
        # Create a pivot table with model families as rows and hardware types as columns
        pivot_df = df.pivot_table(
            index=['model_family'],
            columns='hardware_type',
            values='compatibility_score',
            aggfunc='mean'
        )
        
        # Fill NA values with 0
        pivot_df = pivot_df.fillna(0)
        
        # Reset index for better display
        pivot_df = pivot_df.reset_index()
        
        if args.format == 'chart':
            return create_compatibility_matrix_chart(pivot_df, args)
        
        return pivot_df
    
    return df

def generate_integration_report(conn, args):
    """Generate an integration test report"""
    # Construct the WHERE clause based on filters
    where_clauses = []
    query_params = []
    
    if args.model:
        where_clauses.append("m.model_name LIKE ?")
        query_params.append(f"%{args.model}%")
    
    if args.hardware:
        where_clauses.append("hp.hardware_type = ?")
        query_params.append(args.hardware)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime(args.since, '%Y-%m-%d')
            where_clauses.append("itr.created_at >= ?")
            query_params.append(since_date)
        except ValueError:
            logger.warning(f"Invalid date format for --since: {args.since}. Expected YYYY-MM-DD")
    
    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Construct the query
    sql = f"""
    SELECT 
        itr.test_module,
        itr.test_class,
        itr.test_name,
        itr.status,
        m.model_name,
        hp.hardware_type,
        itr.error_message,
        itr.created_at
    FROM 
        integration_test_results itr
    LEFT JOIN 
        models m ON itr.model_id = m.model_id
    LEFT JOIN 
        hardware_platforms hp ON itr.hardware_id = hp.hardware_id
    {where_clause}
    ORDER BY 
        itr.created_at DESC
    LIMIT ?
    """
    
    query_params.append(args.limit)
    
    # Execute the query
    df = execute_sql_query(conn, sql, query_params)
    
    if df.empty:
        logger.warning("No integration test results found matching the criteria")
        return None
    
    return df

def generate_summary_report(conn, args):
    """Generate a summary report across all result types"""
    # 1. Performance summary
    perf_summary_sql = """
    SELECT 
        m.model_family,
        hp.hardware_type,
        COUNT(*) as test_count,
        AVG(pr.average_latency_ms) as avg_latency_ms,
        AVG(pr.throughput_items_per_second) as avg_throughput,
        AVG(pr.memory_peak_mb) as avg_memory_mb,
        MAX(pr.created_at) as last_tested
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    GROUP BY 
        m.model_family, hp.hardware_type
    ORDER BY 
        m.model_family, hp.hardware_type
    """
    
    perf_summary = execute_sql_query(conn, perf_summary_sql)
    
    # 2. Hardware compatibility summary
    compat_summary_sql = """
    SELECT 
        m.model_family,
        hp.hardware_type,
        COUNT(*) as test_count,
        SUM(CASE WHEN hc.is_compatible THEN 1 ELSE 0 END) as compatible_count,
        AVG(hc.compatibility_score) as avg_compatibility_score,
        MAX(hc.created_at) as last_tested
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    GROUP BY 
        m.model_family, hp.hardware_type
    ORDER BY 
        m.model_family, hp.hardware_type
    """
    
    compat_summary = execute_sql_query(conn, compat_summary_sql)
    
    # 3. Integration test summary
    integration_summary_sql = """
    SELECT 
        itr.test_module,
        COUNT(*) as total_tests,
        SUM(CASE WHEN itr.status = 'pass' THEN 1 ELSE 0 END) as passed,
        SUM(CASE WHEN itr.status = 'fail' THEN 1 ELSE 0 END) as failed,
        SUM(CASE WHEN itr.status = 'error' THEN 1 ELSE 0 END) as errors,
        SUM(CASE WHEN itr.status = 'skip' THEN 1 ELSE 0 END) as skipped,
        MAX(itr.created_at) as last_run
    FROM 
        integration_test_results itr
    GROUP BY 
        itr.test_module
    ORDER BY 
        itr.test_module
    """
    
    integration_summary = execute_sql_query(conn, integration_summary_sql)
    
    # 4. Model summary
    model_summary_sql = """
    SELECT 
        m.model_family,
        COUNT(DISTINCT m.model_id) as model_count,
        AVG(m.parameters_million) as avg_parameters_million
    FROM 
        models m
    GROUP BY 
        m.model_family
    ORDER BY 
        m.model_family
    """
    
    model_summary = execute_sql_query(conn, model_summary_sql)
    
    # 5. Hardware platform summary
    hardware_summary_sql = """
    SELECT 
        hp.hardware_type,
        COUNT(DISTINCT hp.hardware_id) as device_count
    FROM 
        hardware_platforms hp
    GROUP BY 
        hp.hardware_type
    ORDER BY 
        hp.hardware_type
    """
    
    hardware_summary = execute_sql_query(conn, hardware_summary_sql)
    
    # Combined results
    summary = {
        'performance_summary': perf_summary,
        'compatibility_summary': compat_summary,
        'integration_summary': integration_summary,
        'model_summary': model_summary,
        'hardware_summary': hardware_summary
    }
    
    return summary

def query_model_data(conn, model_name, args):
    """Query all data for a specific model"""
    # 1. Get model info
    model_sql = """
    SELECT * FROM models WHERE model_name LIKE ?
    """
    
    model_info = execute_sql_query(conn, model_sql, [f"%{model_name}%"])
    
    if model_info.empty:
        logger.warning(f"No model found with name like: {model_name}")
        return None
    
    # Get the model ID for the first matching model
    model_id = model_info.iloc[0]['model_id']
    exact_model_name = model_info.iloc[0]['model_name']
    
    # 2. Get performance data
    perf_sql = """
    SELECT 
        hp.hardware_type,
        hp.device_name,
        pr.test_case,
        pr.batch_size,
        pr.precision,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at
    FROM 
        performance_results pr
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    WHERE 
        pr.model_id = ?
    ORDER BY 
        pr.created_at DESC
    """
    
    performance_data = execute_sql_query(conn, perf_sql, [model_id])
    
    # 3. Get hardware compatibility data
    compat_sql = """
    SELECT 
        hp.hardware_type,
        hp.device_name,
        hc.is_compatible,
        hc.compatibility_score,
        hc.error_message,
        hc.suggested_fix,
        hc.created_at
    FROM 
        hardware_compatibility hc
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    WHERE 
        hc.model_id = ?
    ORDER BY 
        hc.created_at DESC
    """
    
    compatibility_data = execute_sql_query(conn, compat_sql, [model_id])
    
    # 4. Get integration test data
    test_sql = """
    SELECT 
        itr.test_module,
        itr.test_class,
        itr.test_name,
        itr.status,
        itr.error_message,
        itr.created_at
    FROM 
        integration_test_results itr
    WHERE 
        itr.model_id = ?
    ORDER BY 
        itr.created_at DESC
    """
    
    test_data = execute_sql_query(conn, test_sql, [model_id])
    
    # Combined results
    model_data = {
        'model_info': model_info,
        'performance_data': performance_data,
        'compatibility_data': compatibility_data,
        'test_data': test_data
    }
    
    # If a specific metric is requested, focus on that
    if args.metric and not performance_data.empty:
        metric_col = None
        if args.metric.lower() in ['throughput', 'throughput_items_per_second']:
            metric_col = 'throughput_items_per_second'
        elif args.metric.lower() in ['latency', 'average_latency_ms']:
            metric_col = 'average_latency_ms'
        elif args.metric.lower() in ['memory', 'memory_peak_mb']:
            metric_col = 'memory_peak_mb'
        
        if metric_col and args.compare_hardware:
            # Create a comparison across hardware types
            pivot_df = performance_data.pivot_table(
                index=['test_case', 'batch_size', 'precision'],
                columns='hardware_type',
                values=metric_col,
                aggfunc='mean'
            )
            
            # Reset index for better display
            pivot_df = pivot_df.reset_index()
            
            model_data['metric_comparison'] = pivot_df
            
            if args.format == 'chart':
                title = f"{exact_model_name}: {args.metric} by Hardware"
                return create_model_comparison_chart(pivot_df, metric_col, title, args)
    
    return model_data

def query_hardware_data(conn, hardware_type, args):
    """Query all data for a specific hardware type"""
    # 1. Get hardware info
    hardware_sql = """
    SELECT * FROM hardware_platforms WHERE hardware_type = ?
    """
    
    hardware_info = execute_sql_query(conn, hardware_sql, [hardware_type])
    
    if hardware_info.empty:
        logger.warning(f"No hardware found with type: {hardware_type}")
        return None
    
    # Get a list of hardware IDs
    hardware_ids = hardware_info['hardware_id'].tolist()
    placeholders = ','.join(['?'] * len(hardware_ids))
    
    # 2. Get performance data
    perf_sql = f"""
    SELECT 
        m.model_name,
        m.model_family,
        pr.test_case,
        pr.batch_size,
        pr.precision,
        pr.average_latency_ms,
        pr.throughput_items_per_second,
        pr.memory_peak_mb,
        pr.created_at
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    WHERE 
        pr.hardware_id IN ({placeholders})
    ORDER BY 
        pr.created_at DESC
    """
    
    performance_data = execute_sql_query(conn, perf_sql, hardware_ids)
    
    # 3. Get hardware compatibility data
    compat_sql = f"""
    SELECT 
        m.model_name,
        m.model_family,
        hc.is_compatible,
        hc.compatibility_score,
        hc.error_message,
        hc.suggested_fix,
        hc.created_at
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    WHERE 
        hc.hardware_id IN ({placeholders})
    ORDER BY 
        hc.created_at DESC
    """
    
    compatibility_data = execute_sql_query(conn, compat_sql, hardware_ids)
    
    # 4. Get integration test data
    test_sql = f"""
    SELECT 
        itr.test_module,
        itr.test_class,
        itr.test_name,
        itr.status,
        m.model_name,
        itr.error_message,
        itr.created_at
    FROM 
        integration_test_results itr
    LEFT JOIN 
        models m ON itr.model_id = m.model_id
    WHERE 
        itr.hardware_id IN ({placeholders})
    ORDER BY 
        itr.created_at DESC
    """
    
    test_data = execute_sql_query(conn, test_sql, hardware_ids)
    
    # Combined results
    hardware_data = {
        'hardware_info': hardware_info,
        'performance_data': performance_data,
        'compatibility_data': compatibility_data,
        'test_data': test_data
    }
    
    # If a specific metric is requested, focus on that
    if args.metric and not performance_data.empty:
        metric_col = None
        if args.metric.lower() in ['throughput', 'throughput_items_per_second']:
            metric_col = 'throughput_items_per_second'
        elif args.metric.lower() in ['latency', 'average_latency_ms']:
            metric_col = 'average_latency_ms'
        elif args.metric.lower() in ['memory', 'memory_peak_mb']:
            metric_col = 'memory_peak_mb'
        
        if metric_col:
            # Create a comparison across model families
            pivot_df = performance_data.pivot_table(
                index=['model_family', 'batch_size', 'precision'],
                values=metric_col,
                aggfunc='mean'
            )
            
            # Reset index for better display
            pivot_df = pivot_df.reset_index()
            pivot_df = pivot_df.sort_values(by=metric_col, ascending=False)
            
            hardware_data['metric_comparison'] = pivot_df
            
            if args.format == 'chart':
                title = f"{hardware_type}: {args.metric} by Model Family"
                return create_hardware_model_chart(pivot_df, metric_col, title, args)
    
    return hardware_data

def analyze_performance_trend(conn, args):
    """Analyze performance trends over time"""
    # Construct the WHERE clause based on filters
    where_clauses = []
    query_params = []
    
    if args.family:
        where_clauses.append("m.model_family = ?")
        query_params.append(args.family)
    
    if args.model:
        where_clauses.append("m.model_name LIKE ?")
        query_params.append(f"%{args.model}%")
    
    if args.hardware:
        where_clauses.append("hp.hardware_type = ?")
        query_params.append(args.hardware)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime(args.since, '%Y-%m-%d')
            where_clauses.append("pr.created_at >= ?")
            query_params.append(since_date)
        except ValueError:
            logger.warning(f"Invalid date format for --since: {args.since}. Expected YYYY-MM-DD")
    
    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Determine which metric to analyze
    metric_col = 'throughput_items_per_second'  # default
    if args.metric:
        if args.metric.lower() in ['throughput', 'throughput_items_per_second']:
            metric_col = 'throughput_items_per_second'
        elif args.metric.lower() in ['latency', 'average_latency_ms']:
            metric_col = 'average_latency_ms'
        elif args.metric.lower() in ['memory', 'memory_peak_mb']:
            metric_col = 'memory_peak_mb'
    
    # Construct the query
    sql = f"""
    SELECT 
        DATE(pr.created_at) as test_date,
        m.model_family,
        hp.hardware_type,
        AVG(pr.{metric_col}) as avg_value
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    {where_clause}
    GROUP BY 
        DATE(pr.created_at), m.model_family, hp.hardware_type
    ORDER BY 
        test_date, m.model_family, hp.hardware_type
    """
    
    # Execute the query
    df = execute_sql_query(conn, sql, query_params)
    
    if df.empty:
        logger.warning("No performance data found for trend analysis")
        return None
    
    # Create a pivot for better trend visualization
    pivot_df = df.pivot_table(
        index='test_date',
        columns=['model_family', 'hardware_type'],
        values='avg_value'
    )
    
    # Reset index for better display
    pivot_df = pivot_df.reset_index()
    
    if args.format == 'chart':
        return create_trend_chart(pivot_df, metric_col, args)
    
    return pivot_df

def analyze_compatibility_trend(conn, args):
    """Analyze compatibility trends over time"""
    # Construct the WHERE clause based on filters
    where_clauses = []
    query_params = []
    
    if args.family:
        where_clauses.append("m.model_family = ?")
        query_params.append(args.family)
    
    if args.model:
        where_clauses.append("m.model_name LIKE ?")
        query_params.append(f"%{args.model}%")
    
    if args.hardware:
        where_clauses.append("hp.hardware_type = ?")
        query_params.append(args.hardware)
    
    if args.since:
        try:
            since_date = datetime.datetime.strptime(args.since, '%Y-%m-%d')
            where_clauses.append("hc.created_at >= ?")
            query_params.append(since_date)
        except ValueError:
            logger.warning(f"Invalid date format for --since: {args.since}. Expected YYYY-MM-DD")
    
    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Construct the query
    sql = f"""
    SELECT 
        DATE(hc.created_at) as test_date,
        m.model_family,
        hp.hardware_type,
        AVG(hc.compatibility_score) as avg_compatibility
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    {where_clause}
    GROUP BY 
        DATE(hc.created_at), m.model_family, hp.hardware_type
    ORDER BY 
        test_date, m.model_family, hp.hardware_type
    """
    
    # Execute the query
    df = execute_sql_query(conn, sql, query_params)
    
    if df.empty:
        logger.warning("No compatibility data found for trend analysis")
        return None
    
    # Create a pivot for better trend visualization
    pivot_df = df.pivot_table(
        index='test_date',
        columns=['model_family', 'hardware_type'],
        values='avg_compatibility'
    )
    
    # Reset index for better display
    pivot_df = pivot_df.reset_index()
    
    if args.format == 'chart':
        return create_trend_chart(pivot_df, 'compatibility_score', args)
    
    return pivot_df

def create_hardware_comparison_chart(df, metric_col, args):
    """Create a chart comparing performance across hardware types"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        x_labels = df['model_name'] + ' (' + df['model_family'] + ')'
        hardware_types = [col for col in df.columns if col not in ['model_name', 'model_family', 'batch_size', 'precision']]
        
        # Set up bar positions
        x = np.arange(len(x_labels))
        width = 0.8 / len(hardware_types)
        
        # Plot bars for each hardware type
        for i, hw_type in enumerate(hardware_types):
            offset = (i - len(hardware_types)/2 + 0.5) * width
            plt.bar(x + offset, df[hw_type], width, label=hw_type)
        
        # Add labels and title
        metric_name = args.metric if args.metric else 'Value'
        plt.xlabel('Models')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} by Model and Hardware')
        plt.xticks(x, x_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return df

def create_compatibility_matrix_chart(df, args):
    """Create a heatmap for model-hardware compatibility"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Get model families and hardware types
        model_families = df['model_family']
        hardware_types = [col for col in df.columns if col != 'model_family']
        
        # Prepare data matrix
        data_matrix = df[hardware_types].values
        
        # Create heatmap
        plt.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add labels
        plt.xticks(np.arange(len(hardware_types)), hardware_types, rotation=45, ha='right')
        plt.yticks(np.arange(len(model_families)), model_families)
        
        # Add colorbar
        plt.colorbar(label='Compatibility Score')
        
        # Add title
        plt.title('Model Family - Hardware Compatibility Matrix')
        
        # Add text annotations with the scores
        for i in range(len(model_families)):
            for j in range(len(hardware_types)):
                score = data_matrix[i, j]
                text_color = 'black' if 0.3 <= score <= 0.7 else 'white'
                plt.text(j, i, f"{score:.2f}", ha='center', va='center', color=text_color)
        
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return df

def create_model_comparison_chart(df, metric_col, title, args):
    """Create a chart comparing performance for a specific model across hardware types"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        indices = [f"{row['test_case']} (bs={row['batch_size']})" for _, row in df.iterrows()]
        hardware_types = [col for col in df.columns if col not in ['test_case', 'batch_size', 'precision']]
        
        # Set up bar positions
        x = np.arange(len(indices))
        width = 0.8 / len(hardware_types)
        
        # Plot bars for each hardware type
        for i, hw_type in enumerate(hardware_types):
            if hw_type in df.columns:
                offset = (i - len(hardware_types)/2 + 0.5) * width
                plt.bar(x + offset, df[hw_type], width, label=hw_type)
        
        # Add labels and title
        metric_name = args.metric if args.metric else 'Value'
        plt.xlabel('Test Case')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.xticks(x, indices, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return df

def create_hardware_model_chart(df, metric_col, title, args):
    """Create a chart comparing model families on a specific hardware"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        x_labels = df['model_family'] + f" (bs={df['batch_size']})"
        y_values = df[metric_col]
        
        # Create bar chart
        plt.bar(x_labels, y_values)
        
        # Add labels and title
        metric_name = args.metric if args.metric else 'Value'
        plt.xlabel('Model Family')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return df

def create_trend_chart(pivot_df, metric_col, args):
    """Create a line chart showing trends over time"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Get all combination columns (model_family, hardware_type)
        columns = [col for col in pivot_df.columns if col != 'test_date']
        
        # Plot a line for each column
        for col in columns:
            pivot_df.plot(x='test_date', y=col, ax=plt.gca(), label=str(col))
        
        # Add labels and title
        metric_name = args.metric if args.metric else metric_col
        plt.xlabel('Date')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Trend Over Time')
        plt.legend(title='Model Family / Hardware')
        plt.grid(True)
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return pivot_df

def format_output(result, args):
    """Format the query results according to the specified format"""
    if result is None:
        return "No results found."
    
    # Handle different result types
    if isinstance(result, pd.DataFrame):
        if args.format == 'csv':
            if args.output:
                result.to_csv(args.output, index=False)
                return f"Results saved to {args.output}"
            else:
                return result.to_csv(index=False)
        elif args.format == 'json':
            if args.output:
                result.to_json(args.output, orient='records', indent=2)
                return f"Results saved to {args.output}"
            else:
                return result.to_json(orient='records', indent=2)
        elif args.format == 'html':
            if args.output:
                result.to_html(args.output, index=False)
                return f"Results saved to {args.output}"
            else:
                return result.to_html(index=False)
        else:  # markdown or default
            return tabulate(result, headers='keys', tablefmt='pipe', showindex=False)
    
    elif isinstance(result, dict) and 'performance_summary' in result:
        # This is a summary report with multiple dataframes
        output = []
        
        output.append("# Benchmark Database Summary Report\n")
        
        output.append("## Model Families\n")
        output.append(tabulate(result['model_summary'], headers='keys', tablefmt='pipe', showindex=False))
        output.append("\n\n")
        
        output.append("## Hardware Platforms\n")
        output.append(tabulate(result['hardware_summary'], headers='keys', tablefmt='pipe', showindex=False))
        output.append("\n\n")
        
        output.append("## Performance Summary\n")
        output.append(tabulate(result['performance_summary'], headers='keys', tablefmt='pipe', showindex=False))
        output.append("\n\n")
        
        output.append("## Compatibility Summary\n")
        output.append(tabulate(result['compatibility_summary'], headers='keys', tablefmt='pipe', showindex=False))
        output.append("\n\n")
        
        output.append("## Integration Tests Summary\n")
        output.append(tabulate(result['integration_summary'], headers='keys', tablefmt='pipe', showindex=False))
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(output))
            return f"Summary report saved to {args.output}"
        else:
            return '\n'.join(output)
    
    elif isinstance(result, dict) and 'model_info' in result:
        # This is a model data report
        output = []
        
        model_name = result['model_info'].iloc[0]['model_name']
        model_family = result['model_info'].iloc[0]['model_family']
        
        output.append(f"# Model Report: {model_name}\n")
        output.append(f"- **Family:** {model_family}\n")
        
        if 'metric_comparison' in result and args.metric:
            output.append(f"\n## {args.metric} Comparison Across Hardware\n")
            output.append(tabulate(result['metric_comparison'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['performance_data'].empty:
            output.append("\n## Performance Results\n")
            output.append(tabulate(result['performance_data'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['compatibility_data'].empty:
            output.append("\n## Hardware Compatibility\n")
            output.append(tabulate(result['compatibility_data'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['test_data'].empty:
            output.append("\n## Integration Test Results\n")
            output.append(tabulate(result['test_data'], headers='keys', tablefmt='pipe', showindex=False))
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(output))
            return f"Model report saved to {args.output}"
        else:
            return '\n'.join(output)
    
    elif isinstance(result, dict) and 'hardware_info' in result:
        # This is a hardware data report
        output = []
        
        hardware_type = result['hardware_info'].iloc[0]['hardware_type']
        
        output.append(f"# Hardware Report: {hardware_type}\n")
        
        if 'metric_comparison' in result and args.metric:
            output.append(f"\n## {args.metric} Comparison Across Model Families\n")
            output.append(tabulate(result['metric_comparison'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['performance_data'].empty:
            output.append("\n## Performance Results\n")
            output.append(tabulate(result['performance_data'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['compatibility_data'].empty:
            output.append("\n## Model Compatibility\n")
            output.append(tabulate(result['compatibility_data'], headers='keys', tablefmt='pipe', showindex=False))
            output.append("\n")
        
        if not result['test_data'].empty:
            output.append("\n## Integration Test Results\n")
            output.append(tabulate(result['test_data'], headers='keys', tablefmt='pipe', showindex=False))
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(output))
            return f"Hardware report saved to {args.output}"
        else:
            return '\n'.join(output)
    
    elif isinstance(result, str) and result.startswith("Chart generated"):
        # This is a chart result
        return result
    
    else:
        # Default fallback
        return str(result)

def main():
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Connect to the database
    conn = connect_to_db(args.db)
    
    # Check if required tables exist
    table_names = check_tables_exist(conn)
    
    # Execute the query based on arguments
    result = None
    
    if args.sql:
        # Direct SQL query
        result = execute_sql_query(conn, args.sql)
    
    elif args.report:
        # Generate a predefined report
        if args.report == 'performance':
            result = generate_performance_report(conn, args)
        elif args.report == 'hardware':
            result = generate_hardware_report(conn, args)
        elif args.report == 'integration':
            result = generate_integration_report(conn, args)
        elif args.report == 'summary':
            result = generate_summary_report(conn, args)
        elif args.report == 'web_platform':
            result = generate_web_platform_report(conn, args)
        elif args.report == 'webgpu':
            result = generate_webgpu_features_report(conn, args)
    
    elif args.model:
        # Query data for a specific model
        result = query_model_data(conn, args.model, args)
    
    elif args.hardware:
        # Query data for a specific hardware type
        result = query_hardware_data(conn, args.hardware, args)
    
    elif args.compatibility_matrix:
        # Generate model-hardware compatibility matrix
        result = generate_hardware_report(conn, args)
    
    elif args.trend:
        # Show trends over time
        if args.trend == 'performance':
            result = analyze_performance_trend(conn, args)
        elif args.trend == 'compatibility':
            result = analyze_compatibility_trend(conn, args)
    
    # Format and display results
    if result is not None:
        print(format_output(result, args))
    else:
        logger.error("No results returned from query")
    
    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()