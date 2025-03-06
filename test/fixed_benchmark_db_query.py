#!/usr/bin/env python
"""
Enhanced Benchmark Database Query Tool for the IPFS Accelerate Python Framework.

This module provides a robust CLI tool for querying the benchmark database,
generating reports, and extracting insights with improved error handling
and NULL value processing.

Usage:
    python fixed_benchmark_db_query.py --sql "SELECT model_name, hardware_type, AVG(throughput_items_per_second) FROM performance_results JOIN models USING(model_id) JOIN hardware_platforms USING(hardware_id) GROUP BY model_name, hardware_type"
    python fixed_benchmark_db_query.py --report performance --format html --output benchmark_report.html
    python fixed_benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware --output bert_throughput.png --format chart
"""

import os
import sys
import json
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import duckdb
except ImportError:
    print("Error: DuckDB is required. Please install it with: pip install duckdb")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark_query')

class BenchmarkDBQuery:
    """
    Query tool for benchmark database.
    
    This class provides methods for querying the benchmark database and
    generating reports.
    """
    
    def __init__(self, db_path: Optional[str] = None, debug: bool = False):
        """
        Initialize the query tool.
        
        Args:
            db_path: Path to the benchmark database
            debug: Enable debug logging
        """
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Set benchmark database path
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        logger.debug(f"Using database at: {self.db_path}")
        
        # Connect to database
        try:
            self.con = duckdb.connect(self.db_path)
            logger.debug("Connected to database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query against the database.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        try:
            logger.debug(f"Executing SQL: {sql}")
            result = self.con.execute(sql).fetchdf()
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            logger.error(f"SQL: {sql}")
            return pd.DataFrame()
    
    def get_models(self) -> pd.DataFrame:
        """
        Get list of models in the database.
        
        Returns:
            DataFrame with model information
        """
        sql = """
        SELECT 
            model_id, 
            model_name, 
            model_family, 
            modality
        FROM 
            models
        ORDER BY 
            model_family, model_name
        """
        return self.execute_sql(sql)
    
    def get_hardware_platforms(self) -> pd.DataFrame:
        """
        Get list of hardware platforms in the database.
        
        Returns:
            DataFrame with hardware platform information
        """
        sql = """
        SELECT 
            hardware_id, 
            hardware_type, 
            device_name, 
            driver_version,
            compute_units,
            memory_gb
        FROM 
            hardware_platforms
        ORDER BY 
            hardware_type
        """
        return self.execute_sql(sql)
    
    def get_performance_results(self, 
                                model_id: Optional[int] = None,
                                model_name: Optional[str] = None,
                                hardware_id: Optional[int] = None,
                                hardware_type: Optional[str] = None,
                                limit: int = 1000) -> pd.DataFrame:
        """
        Get performance results from the database.
        
        Args:
            model_id: Filter by model ID
            model_name: Filter by model name
            hardware_id: Filter by hardware ID
            hardware_type: Filter by hardware type
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with performance results
        """
        # Build WHERE clause
        where_clauses = []
        if model_id is not None:
            where_clauses.append(f"p.model_id = {model_id}")
        if model_name is not None:
            where_clauses.append(f"m.model_name = '{model_name}'")
        if hardware_id is not None:
            where_clauses.append(f"p.hardware_id = {hardware_id}")
        if hardware_type is not None:
            where_clauses.append(f"h.hardware_type = '{hardware_type}'")
        
        # Construct WHERE clause string
        where_str = ""
        if where_clauses:
            where_str = "WHERE " + " AND ".join(where_clauses)
        
        sql = f"""
        SELECT 
            p.result_id,
            m.model_name,
            m.model_family,
            h.hardware_type,
            p.batch_size,
            p.precision,
            p.average_latency_ms,
            p.throughput_items_per_second,
            p.memory_peak_mb,
            p.created_at
        FROM 
            performance_results p
        JOIN 
            models m ON p.model_id = m.model_id
        JOIN 
            hardware_platforms h ON p.hardware_id = h.hardware_id
        {where_str}
        ORDER BY 
            p.created_at DESC
        LIMIT {limit}
        """
        return self.execute_sql(sql)
    
    def get_compatibility_matrix(self) -> pd.DataFrame:
        """
        Generate a hardware compatibility matrix.
        
        Returns:
            DataFrame with compatibility matrix
        """
        sql = """
        SELECT 
            m.model_name,
            m.model_family,
            MAX(CASE WHEN h.hardware_type = 'cpu' THEN 1 ELSE 0 END) as cpu_support,
            MAX(CASE WHEN h.hardware_type = 'cuda' THEN 1 ELSE 0 END) as cuda_support,
            MAX(CASE WHEN h.hardware_type = 'rocm' THEN 1 ELSE 0 END) as rocm_support,
            MAX(CASE WHEN h.hardware_type = 'mps' THEN 1 ELSE 0 END) as mps_support,
            MAX(CASE WHEN h.hardware_type = 'openvino' THEN 1 ELSE 0 END) as openvino_support,
            MAX(CASE WHEN h.hardware_type = 'webnn' THEN 1 ELSE 0 END) as webnn_support,
            MAX(CASE WHEN h.hardware_type = 'webgpu' THEN 1 ELSE 0 END) as webgpu_support,
            MAX(CASE WHEN h.hardware_type = 'qualcomm' THEN 1 ELSE 0 END) as qualcomm_support
        FROM models m
        LEFT JOIN performance_results p ON m.model_id = p.model_id
        LEFT JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        GROUP BY m.model_name, m.model_family
        ORDER BY m.model_family, m.model_name
        """
        df = self.execute_sql(sql)
        
        # Replace NaN values with 0
        for col in ['cpu_support', 'cuda_support', 'rocm_support', 'mps_support', 
                   'openvino_support', 'webnn_support', 'webgpu_support', 'qualcomm_support']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        return df
    
    def get_memory_analysis(self) -> pd.DataFrame:
        """
        Analyze memory usage across hardware platforms.
        
        Returns:
            DataFrame with memory analysis
        """
        sql = """
        SELECT 
            m.model_name, 
            h.hardware_type, 
            AVG(p.memory_peak_mb) as avg_memory,
            MIN(p.memory_peak_mb) as min_memory,
            MAX(p.memory_peak_mb) as max_memory
        FROM performance_results p
        JOIN models m ON p.model_id = m.model_id
        JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        GROUP BY m.model_name, h.hardware_type
        ORDER BY avg_memory DESC
        """
        return self.execute_sql(sql)
    
    def generate_performance_report(self, format: str = 'markdown') -> str:
        """
        Generate a performance report.
        
        Args:
            format: Output format (markdown, html)
            
        Returns:
            Report content
        """
        # Get performance results
        performance_df = self.get_performance_results(limit=100)
        
        # Get hardware platforms
        hardware_df = self.get_hardware_platforms()
        
        # Get compatibility matrix
        compatibility_df = self.get_compatibility_matrix()
        
        # Generate report based on format
        if format == 'html':
            report = f"""
            <html>
            <head>
                <title>Benchmark Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    table {{ border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>Benchmark Performance Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Hardware Platforms</h2>
                {hardware_df.to_html(index=False)}
                
                <h2>Performance Results</h2>
                {performance_df.to_html(index=False)}
                
                <h2>Hardware Compatibility Matrix</h2>
                {compatibility_df.to_html(index=False)}
            </body>
            </html>
            """
        else:  # Default to markdown
            report = f"""# Benchmark Performance Report

Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Hardware Platforms

{tabulate(hardware_df, headers='keys', tablefmt='pipe')}

## Performance Results

{tabulate(performance_df, headers='keys', tablefmt='pipe')}

## Hardware Compatibility Matrix

{tabulate(compatibility_df, headers='keys', tablefmt='pipe')}
"""
        
        return report
    
    def generate_hardware_report(self, format: str = 'markdown') -> str:
        """
        Generate a hardware report.
        
        Args:
            format: Output format (markdown, html)
            
        Returns:
            Report content
        """
        # Get hardware platforms
        hardware_df = self.get_hardware_platforms()
        
        # Get memory analysis
        memory_df = self.get_memory_analysis()
        
        # Generate report based on format
        if format == 'html':
            report = f"""
            <html>
            <head>
                <title>Hardware Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    table {{ border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                </style>
            </head>
            <body>
                <h1>Hardware Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Hardware Platforms</h2>
                {hardware_df.to_html(index=False)}
                
                <h2>Memory Analysis</h2>
                {memory_df.to_html(index=False)}
            </body>
            </html>
            """
        else:  # Default to markdown
            report = f"""# Hardware Report

Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Hardware Platforms

{tabulate(hardware_df, headers='keys', tablefmt='pipe')}

## Memory Analysis

{tabulate(memory_df, headers='keys', tablefmt='pipe')}
"""
        
        return report
    
    def generate_compatibility_matrix_report(self, format: str = 'markdown') -> str:
        """
        Generate a compatibility matrix report.
        
        Args:
            format: Output format (markdown, html)
            
        Returns:
            Report content
        """
        # Get compatibility matrix
        compatibility_df = self.get_compatibility_matrix()
        
        # Generate report based on format
        if format == 'html':
            report = f"""
            <html>
            <head>
                <title>Hardware Compatibility Matrix</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    table {{ border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .supported {{ color: green; font-weight: bold; }}
                    .unsupported {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Hardware Compatibility Matrix</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # Create a custom HTML table with green/red indicators
            report += "<table>\n<tr><th>Model</th><th>Model Family</th>"
            for hw in ['CPU', 'CUDA', 'ROCm', 'MPS', 'OpenVINO', 'WebNN', 'WebGPU', 'Qualcomm']:
                report += f"<th>{hw}</th>"
            report += "</tr>\n"
            
            for _, row in compatibility_df.iterrows():
                report += "<tr>"
                report += f"<td>{row['model_name']}</td>"
                report += f"<td>{row['model_family'] if not pd.isna(row['model_family']) else '-'}</td>"
                
                for hw, col in zip(
                    ['CPU', 'CUDA', 'ROCm', 'MPS', 'OpenVINO', 'WebNN', 'WebGPU', 'Qualcomm'],
                    ['cpu_support', 'cuda_support', 'rocm_support', 'mps_support', 'openvino_support', 'webnn_support', 'webgpu_support', 'qualcomm_support']
                ):
                    if col in row and row[col] == 1:
                        report += f'<td class="supported">✅</td>'
                    else:
                        report += f'<td class="unsupported">❌</td>'
                
                report += "</tr>\n"
            
            report += """
            </table>
            </body>
            </html>
            """
        else:  # Default to markdown
            # Convert binary values to ✅/❌ for better readability
            display_df = compatibility_df.copy()
            
            for col in ['cpu_support', 'cuda_support', 'rocm_support', 'mps_support', 
                       'openvino_support', 'webnn_support', 'webgpu_support', 'qualcomm_support']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: '✅' if x == 1 else '❌')
            
            # Rename columns for better readability
            column_map = {
                'cpu_support': 'CPU',
                'cuda_support': 'CUDA',
                'rocm_support': 'ROCm',
                'mps_support': 'MPS',
                'openvino_support': 'OpenVINO',
                'webnn_support': 'WebNN',
                'webgpu_support': 'WebGPU',
                'qualcomm_support': 'Qualcomm'
            }
            display_df = display_df.rename(columns=column_map)
            
            report = f"""# Hardware Compatibility Matrix

Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{tabulate(display_df, headers='keys', tablefmt='pipe')}
"""
        
        return report
    
    def compare_hardware_for_model(self, model_name: str, metric: str = 'throughput') -> pd.DataFrame:
        """
        Compare hardware platforms for a specific model.
        
        Args:
            model_name: Model name to compare
            metric: Metric to compare (latency, throughput, memory)
            
        Returns:
            DataFrame with comparison results
        """
        # Map metric name to database column
        metric_map = {
            'latency': 'average_latency_ms',
            'throughput': 'throughput_items_per_second',
            'memory': 'memory_peak_mb'
        }
        
        column = metric_map.get(metric.lower(), 'throughput_items_per_second')
        
        sql = f"""
        SELECT 
            h.hardware_type, 
            p.batch_size,
            AVG({column}) as avg_value,
            MIN({column}) as min_value,
            MAX({column}) as max_value
        FROM performance_results p
        JOIN models m ON p.model_id = m.model_id
        JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        WHERE m.model_name = '{model_name}'
        GROUP BY h.hardware_type, p.batch_size
        ORDER BY avg_value DESC
        """
        return self.execute_sql(sql)
    
    def plot_hardware_comparison(self, model_name: str, metric: str = 'throughput') -> str:
        """
        Plot hardware comparison for a specific model.
        
        Args:
            model_name: Model name to compare
            metric: Metric to compare (latency, throughput, memory)
            
        Returns:
            Path to saved plot
        """
        # Get comparison data
        df = self.compare_hardware_for_model(model_name, metric)
        
        if df.empty:
            logger.error(f"No data available for model: {model_name}")
            return ""
        
        # Map metric name to readable label
        metric_labels = {
            'latency': 'Latency (ms)',
            'throughput': 'Throughput (items/sec)',
            'memory': 'Memory Usage (MB)'
        }
        
        y_label = metric_labels.get(metric.lower(), metric)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='hardware_type', y='avg_value')
        plt.title(f"{y_label} Comparison for {model_name}")
        plt.xlabel("Hardware Platform")
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{model_name}_{metric}_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Plot saved to: {plot_path}")
        return plot_path
    
    def generate_summary_report(self, format: str = 'markdown') -> str:
        """
        Generate a summary report.
        
        Args:
            format: Output format (markdown, html)
            
        Returns:
            Report content
        """
        # Get model count
        model_count = self.execute_sql("SELECT COUNT(*) as count FROM models").iloc[0]['count']
        
        # Get hardware platform count
        hardware_count = self.execute_sql("SELECT COUNT(*) as count FROM hardware_platforms").iloc[0]['count']
        
        # Get result count
        result_count = self.execute_sql("SELECT COUNT(*) as count FROM performance_results").iloc[0]['count']
        
        # Get latest test date
        latest_date = self.execute_sql("SELECT MAX(created_at) as latest FROM performance_results").iloc[0]['latest']
        
        # Get average performance by hardware
        avg_perf = self.execute_sql("""
        SELECT 
            h.hardware_type, 
            AVG(p.throughput_items_per_second) as avg_throughput,
            AVG(p.average_latency_ms) as avg_latency
        FROM performance_results p
        JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
        GROUP BY h.hardware_type
        ORDER BY avg_throughput DESC
        """)
        
        if format == 'html':
            report = f"""
            <html>
            <head>
                <title>Benchmark Summary Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    table {{ border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .summary {{ display: flex; justify-content: space-between; margin: 20px 0; }}
                    .summary-item {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; width: 22%; text-align: center; }}
                    .summary-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                    .summary-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                </style>
            </head>
            <body>
                <h1>Benchmark Summary Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <div class="summary-item">
                        <div class="summary-value">{model_count}</div>
                        <div class="summary-label">Models</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{hardware_count}</div>
                        <div class="summary-label">Hardware Platforms</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{result_count}</div>
                        <div class="summary-label">Benchmark Results</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{latest_date}</div>
                        <div class="summary-label">Latest Test</div>
                    </div>
                </div>
                
                <h2>Average Performance by Hardware</h2>
                {avg_perf.to_html(index=False)}
            </body>
            </html>
            """
        else:  # Default to markdown
            report = f"""# Benchmark Summary Report

Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Models:** {model_count}
- **Hardware Platforms:** {hardware_count}
- **Benchmark Results:** {result_count}
- **Latest Test:** {latest_date}

## Average Performance by Hardware

{tabulate(avg_perf, headers='keys', tablefmt='pipe')}
"""
        
        return report
    
    def generate_report(self, report_type: str, format: str = 'markdown') -> str:
        """
        Generate a report.
        
        Args:
            report_type: Type of report (performance, hardware, compatibility, summary)
            format: Output format (markdown, html)
            
        Returns:
            Report content
        """
        if report_type == 'performance':
            return self.generate_performance_report(format)
        elif report_type == 'hardware':
            return self.generate_hardware_report(format)
        elif report_type == 'compatibility':
            return self.generate_compatibility_matrix_report(format)
        elif report_type == 'summary':
            return self.generate_summary_report(format)
        elif report_type == 'web_platform':
            try:
                result = generate_web_platform_report(self.conn, self.args)
                return format_output(result, self.args)
            except Exception as e:
                logger.error(f"Error generating web platform report: {e}")
                return f"Error generating web platform report: {e}"
        elif report_type == 'webgpu':
            try:
                result = generate_webgpu_features_report(self.conn, self.args)
                return format_output(result, self.args)
            except Exception as e:
                logger.error(f"Error generating WebGPU report: {e}")
                return f"Error generating WebGPU report: {e}"
        else:
            return f"Unknown report type: {report_type}"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark Database Query Tool')
    
    # Database connection
    parser.add_argument('--db', type=str, default=None,
                      help='Path to the benchmark database')
    
    # Query modes (mutually exclusive)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--sql', type=str,
                          help='Execute a SQL query against the database')
    query_group.add_argument('--report', type=str, choices=['performance', 'hardware', 'integration', 'summary', 'web_platform', 'webgpu'],
                          help='Generate a predefined report')
    query_group.add_argument('--model', type=str,
                          help='Show data for a specific model')
    query_group.add_argument('--hardware', type=str,
                          help='Show data for a specific hardware platform')
    query_group.add_argument('--compatibility-matrix', action='store_true',
                          help='Generate a hardware compatibility matrix')
    query_group.add_argument('--trend', type=str, choices=['performance', 'compatibility'],
                          help='Show trends over time')
    
    # Filters
    parser.add_argument('--family', type=str,
                      help='Filter by model family')
    parser.add_argument('--metric', type=str, default='throughput',
                      help='Metric to use for comparison (latency, throughput, memory)')
    parser.add_argument('--since', type=str,
                      help='Show data since this date (YYYY-MM-DD)')
    
    # Output options
    parser.add_argument('--compare-hardware', action='store_true',
                      help='Compare hardware platforms for a model')
    parser.add_argument('--output', type=str,
                      help='Output file path')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'html', 'markdown', 'chart'],
                      default='markdown', help='Output format')
    parser.add_argument('--limit', type=int, default=1000,
                      help='Maximum number of results to return')
    
    # Debug options
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()

    # Create query tool
    try:
        query_tool = BenchmarkDBQuery(db_path=args.db, debug=args.verbose)
    except Exception as e:
        logger.error(f"Error initializing query tool: {e}")
        return

    # Perform requested actions
    if args.sql:
        # Execute SQL query
        result = query_tool.execute_sql(args.sql)
        
        if result.empty:
            logger.error("Query returned no results")
        else:
            # Display or save result
            if args.output:
                if args.format == 'csv':
                    result.to_csv(args.output, index=False)
                elif args.format == 'json':
                    result.to_json(args.output, orient='records', indent=2)
                elif args.format == 'html':
                    result.to_html(args.output, index=False)
                elif args.format == 'markdown':
                    with open(args.output, 'w') as f:
                        f.write(tabulate(result, headers='keys', tablefmt='pipe'))
                else:
                    # Default to text
                    with open(args.output, 'w') as f:
                        f.write(result.to_string(index=False))
                logger.info(f"Query results saved to: {args.output}")
            else:
                # Display to console
                print(tabulate(result, headers='keys', tablefmt='psql'))
    
    elif args.report:
        # Generate predefined report
        report = query_tool.generate_report(args.report, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {args.output}")
        else:
            print(report)
    
    elif args.model:
        # Show data for a specific model
        if args.compare_hardware:
            # Compare hardware platforms for this model
            if args.format == 'chart':
                # Generate and save chart
                chart_path = query_tool.plot_hardware_comparison(args.model, args.metric)
                if chart_path:
                    if args.output:
                        import shutil
                        shutil.copy(chart_path, args.output)
                        logger.info(f"Chart saved to: {args.output}")
                    else:
                        logger.info(f"Chart saved to: {chart_path}")
            else:
                # Show tabular data
                result = query_tool.compare_hardware_for_model(args.model, args.metric)
                
                if result.empty:
                    logger.error(f"No data available for model: {args.model}")
                else:
                    if args.output:
                        if args.format == 'csv':
                            result.to_csv(args.output, index=False)
                        elif args.format == 'json':
                            result.to_json(args.output, orient='records', indent=2)
                        elif args.format == 'html':
                            result.to_html(args.output, index=False)
                        elif args.format == 'markdown':
                            with open(args.output, 'w') as f:
                                f.write(tabulate(result, headers='keys', tablefmt='pipe'))
                        logger.info(f"Results saved to: {args.output}")
                    else:
                        print(tabulate(result, headers='keys', tablefmt='psql'))
        else:
            # Show performance results for this model
            result = query_tool.get_performance_results(model_name=args.model, limit=args.limit)
            
            if result.empty:
                logger.error(f"No data available for model: {args.model}")
            else:
                if args.output:
                    if args.format == 'csv':
                        result.to_csv(args.output, index=False)
                    elif args.format == 'json':
                        result.to_json(args.output, orient='records', indent=2)
                    elif args.format == 'html':
                        result.to_html(args.output, index=False)
                    elif args.format == 'markdown':
                        with open(args.output, 'w') as f:
                            f.write(tabulate(result, headers='keys', tablefmt='pipe'))
                    logger.info(f"Results saved to: {args.output}")
                else:
                    print(tabulate(result, headers='keys', tablefmt='psql'))
    
    elif args.hardware:
        # Show data for a specific hardware platform
        result = query_tool.get_performance_results(hardware_type=args.hardware, limit=args.limit)
        
        if result.empty:
            logger.error(f"No data available for hardware: {args.hardware}")
        else:
            if args.output:
                if args.format == 'csv':
                    result.to_csv(args.output, index=False)
                elif args.format == 'json':
                    result.to_json(args.output, orient='records', indent=2)
                elif args.format == 'html':
                    result.to_html(args.output, index=False)
                elif args.format == 'markdown':
                    with open(args.output, 'w') as f:
                        f.write(tabulate(result, headers='keys', tablefmt='pipe'))
                logger.info(f"Results saved to: {args.output}")
            else:
                print(tabulate(result, headers='keys', tablefmt='psql'))
    
    elif args.compatibility_matrix:
        # Generate hardware compatibility matrix
        result = query_tool.get_compatibility_matrix()
        
        if result.empty:
            logger.error("No compatibility data available")
        else:
            # Convert binary values to ✅/❌ for better readability
            for col in ['cpu_support', 'cuda_support', 'rocm_support', 'mps_support', 
                       'openvino_support', 'webnn_support', 'webgpu_support', 'qualcomm_support']:
                if col in result.columns:
                    result[col] = result[col].apply(lambda x: '✅' if x == 1 else '❌')
            
            # Rename columns for better readability
            column_map = {
                'cpu_support': 'CPU',
                'cuda_support': 'CUDA',
                'rocm_support': 'ROCm',
                'mps_support': 'MPS',
                'openvino_support': 'OpenVINO',
                'webnn_support': 'WebNN',
                'webgpu_support': 'WebGPU',
                'qualcomm_support': 'Qualcomm'
            }
            result = result.rename(columns=column_map)
            
            if args.output:
                if args.format == 'csv':
                    result.to_csv(args.output, index=False)
                elif args.format == 'json':
                    result.to_json(args.output, orient='records', indent=2)
                elif args.format == 'html':
                    result.to_html(args.output, index=False, escape=False)
                elif args.format == 'markdown':
                    with open(args.output, 'w') as f:
                        f.write(tabulate(result, headers='keys', tablefmt='pipe'))
                logger.info(f"Compatibility matrix saved to: {args.output}")
            else:
                print(tabulate(result, headers='keys', tablefmt='psql'))
    
    elif args.trend:
        if args.trend == 'performance':
            result = analyze_performance_trend(conn, args)
        elif args.trend == 'compatibility':
            result = analyze_compatibility_trend(conn, args)
        else:
            logger.error(f"Unknown trend type: {args.trend}")
            
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

def generate_web_platform_report(conn, args):
    """Generate a report on web platform performance and compatibility"""
    # Construct query to get web platform performance data
    sql = """
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        pr.batch_size,
        pr.precision,
        AVG(pr.throughput_items_per_second) as avg_throughput,
        AVG(pr.average_latency_ms) as avg_latency,
        AVG(pr.memory_peak_mb) as avg_memory,
        COUNT(*) as test_count
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    WHERE 
        hp.hardware_type IN ('webgpu', 'webnn', 'cpu')
    GROUP BY 
        m.model_name, m.model_family, hp.hardware_type, pr.batch_size, pr.precision
    ORDER BY 
        m.model_family, m.model_name, hp.hardware_type
    """
    
    performance_data = execute_sql_query(conn, sql)
    
    # Get web platform compatibility data
    compat_sql = """
    SELECT 
        m.model_name,
        m.model_family,
        hp.hardware_type,
        hc.is_compatible,
        hc.compatibility_score
    FROM 
        hardware_compatibility hc
    JOIN 
        models m ON hc.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON hc.hardware_id = hp.hardware_id
    WHERE 
        hp.hardware_type IN ('webgpu', 'webnn')
    ORDER BY 
        m.model_family, m.model_name, hp.hardware_type
    """
    
    compatibility_data = execute_sql_query(conn, compat_sql)
    
    # Create a pivot table comparing web platforms to CPU performance
    if not performance_data.empty:
        try:
            # Create pivot table for throughput comparison
            throughput_pivot = performance_data.pivot_table(
                index=['model_name', 'model_family', 'batch_size'],
                columns='hardware_type',
                values='avg_throughput',
                aggfunc='mean'
            ).reset_index()
            
            # Calculate relative performance compared to CPU
            if 'cpu' in throughput_pivot.columns and 'webgpu' in throughput_pivot.columns:
                throughput_pivot['webgpu_vs_cpu'] = throughput_pivot['webgpu'] / throughput_pivot['cpu']
                
            if 'cpu' in throughput_pivot.columns and 'webnn' in throughput_pivot.columns:
                throughput_pivot['webnn_vs_cpu'] = throughput_pivot['webnn'] / throughput_pivot['cpu']
            
            # Return combined data
            return {
                'performance_data': performance_data,
                'compatibility_data': compatibility_data,
                'throughput_comparison': throughput_pivot
            }
        except Exception as e:
            logger.error(f"Error creating web platform performance comparison: {e}")
            return performance_data
    
    return performance_data

def generate_webgpu_features_report(conn, args):
    """Generate a report on WebGPU features usage and performance impact"""
    # Query for WebGPU features usage
    sql = """
    SELECT 
        m.model_name,
        m.model_family,
        pr.shader_precompilation_enabled,
        pr.compute_shaders_enabled,
        pr.parallel_loading_enabled,
        pr.batch_size,
        AVG(pr.throughput_items_per_second) as avg_throughput,
        AVG(pr.average_latency_ms) as avg_latency,
        COUNT(*) as test_count
    FROM 
        performance_results pr
    JOIN 
        models m ON pr.model_id = m.model_id
    JOIN 
        hardware_platforms hp ON pr.hardware_id = hp.hardware_id
    WHERE 
        hp.hardware_type = 'webgpu'
    GROUP BY 
        m.model_name, m.model_family, pr.shader_precompilation_enabled,
        pr.compute_shaders_enabled, pr.parallel_loading_enabled, pr.batch_size
    ORDER BY 
        m.model_family, m.model_name
    """
    
    try:
        # Execute query with a fallback to handle possibly missing feature columns
        features_data = execute_sql_query(conn, sql)
        
        if features_data.empty:
            # Try simplified query without feature columns that might not exist
            sql_simple = """
            SELECT 
                m.model_name,
                m.model_family,
                pr.batch_size,
                AVG(pr.throughput_items_per_second) as avg_throughput,
                AVG(pr.average_latency_ms) as avg_latency,
                COUNT(*) as test_count
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                hp.hardware_type = 'webgpu'
            GROUP BY 
                m.model_name, m.model_family, pr.batch_size
            ORDER BY 
                m.model_family, m.model_name
            """
            features_data = execute_sql_query(conn, sql_simple)
            
            # Add missing feature columns with default values
            if not features_data.empty:
                features_data['shader_precompilation_enabled'] = False
                features_data['compute_shaders_enabled'] = False
                features_data['parallel_loading_enabled'] = False
                
                # Add note about features not being tracked in database
                logger.warning("WebGPU feature flags not found in database schema. Using default values.")
    
        if not features_data.empty and args.format == 'chart':
            return create_webgpu_features_chart(features_data, args)
            
        return features_data
    except Exception as e:
        logger.error(f"Error generating WebGPU features report: {e}")
        return None

def create_webgpu_features_chart(df, args):
    """Create a chart showing the impact of WebGPU features on performance"""
    try:
        plt.figure(figsize=(14, 10))
        
        # Set up bar positions
        feature_cols = ['shader_precompilation_enabled', 'compute_shaders_enabled', 'parallel_loading_enabled']
        
        # Create feature combination column for grouping
        df['features'] = ''
        for col in feature_cols:
            if col in df.columns:
                df['features'] += df[col].map({True: '1', False: '0'})
        
        # Map binary feature strings to readable labels
        feature_map = {
            '000': 'No Features',
            '100': 'Shader Precompilation',
            '010': 'Compute Shaders',
            '001': 'Parallel Loading',
            '110': 'Shader+Compute',
            '101': 'Shader+Parallel',
            '011': 'Compute+Parallel',
            '111': 'All Features'
        }
        
        # Apply mapping
        df['feature_set'] = df['features'].map(feature_map)
        
        # Group by model and features to get average throughput
        perf_by_feature = df.groupby(['model_name', 'feature_set'])['avg_throughput'].mean().reset_index()
        
        # Pivot for plotting
        plot_data = perf_by_feature.pivot(index='model_name', columns='feature_set', values='avg_throughput')
        
        # Plot as stacked bar chart
        ax = plot_data.plot(kind='bar', figsize=(14, 8))
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel('Throughput (items/second)')
        plt.title('WebGPU Performance by Feature Combination')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Feature Set')
        plt.tight_layout()
        
        # Save or display the chart
        if args.output:
            plt.savefig(args.output)
            logger.info(f"Chart saved to {args.output}")
        else:
            plt.show()
        
        return "Chart generated successfully"
    except Exception as e:
        logger.error(f"Error creating WebGPU features chart: {e}")
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


if __name__ == '__main__':
    main()