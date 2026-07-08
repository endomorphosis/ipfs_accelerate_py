#!/usr/bin/env python3
"""
Distributed Testing Framework - Coordinator DuckDB Integration

This module provides integration between the coordinator and the DuckDB database system
for efficient test result storage and aggregation. It combines results from multiple
worker nodes and provides comprehensive reporting and visualization capabilities.

Core features:
- Centralized result storage and aggregation
- Automatic schema creation and management
- Real-time result visualization with interactive dashboards
- Comprehensive reporting capabilities
- Result caching and batch processing for efficiency
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_duckdb_integration")

# Conditional import for duckdb
try:
    from duckdb_result_processor import DuckDBResultProcessor
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.error("DuckDBResultProcessor not available. Integration cannot function.")
    DUCKDB_AVAILABLE = False

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class CoordinatorDuckDBIntegration:
    """Integrates the coordinator with DuckDB for result management."""
    
    def __init__(self, db_path, batch_size=20, cache_enabled=True, cache_ttl_seconds=3600,
                 auto_vacuum_interval_hours=24, enable_dashboard=False, dashboard_port=8050):
        """Initialize the coordinator DuckDB integration.
        
        Args:
            db_path: Path to the DuckDB database file
            batch_size: Maximum number of results to batch before database insertion
            cache_enabled: Whether to enable in-memory result caching
            cache_ttl_seconds: Time-to-live for cached results in seconds
            auto_vacuum_interval_hours: Interval for automatic database vacuuming (0 to disable)
            enable_dashboard: Whether to enable the Dash-based result dashboard
            dashboard_port: Port for the dashboard web server
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for CoordinatorDuckDBIntegration")
            
        self.db_path = db_path
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.auto_vacuum_interval = auto_vacuum_interval_hours * 3600
        self.enable_dashboard = enable_dashboard
        self.dashboard_port = dashboard_port
        
        # Initialize result processor
        self.result_processor = DuckDBResultProcessor(db_path)
        
        # Set up result cache
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        self.last_cache_cleanup = time.time()
        
        # Set up background threads
        self.vacuum_thread = None
        self.stop_event = threading.Event()
        
        # Start background processes
        if self.auto_vacuum_interval > 0:
            self.start_vacuum_thread()
            
        # Start dashboard if enabled
        self.dashboard_app = None
        self.dashboard_thread = None
        if self.enable_dashboard:
            self.start_dashboard()
            
        logger.info(f"Coordinator DuckDB Integration initialized with database at {db_path}")
    
    def start_vacuum_thread(self):
        """Start the background thread for database vacuuming."""
        if self.vacuum_thread is None or not self.vacuum_thread.is_alive():
            self.stop_event.clear()
            self.vacuum_thread = threading.Thread(target=self._vacuum_loop)
            self.vacuum_thread.daemon = True
            self.vacuum_thread.start()
            logger.info("Started database vacuum thread")
    
    def stop_vacuum_thread(self):
        """Stop the background thread for database vacuuming."""
        if self.vacuum_thread and self.vacuum_thread.is_alive():
            self.stop_event.set()
            self.vacuum_thread.join(timeout=5)
            logger.info("Stopped database vacuum thread")
    
    def _vacuum_loop(self):
        """Background loop for periodic database vacuuming."""
        while not self.stop_event.is_set():
            try:
                # Sleep for the vacuum interval
                for _ in range(min(3600, self.auto_vacuum_interval)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                
                if self.stop_event.is_set():
                    break
                
                # Perform vacuum
                logger.info("Starting database vacuum")
                self.vacuum_database()
                logger.info("Completed database vacuum")
                
            except Exception as e:
                logger.error(f"Error in vacuum loop: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def vacuum_database(self):
        """Vacuum the database to optimize storage and performance."""
        try:
            # Use a separate connection for vacuum
            conn = self.result_processor.get_connection()
            conn.execute("VACUUM")
            self.result_processor.release_connection(conn)
            
            logger.info("Database vacuum completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            return False
    
    def start_dashboard(self):
        """Start the result visualization dashboard."""
        try:
            # Conditional import for dash components
            import dash
            from dash import dcc, html
            import plotly.express as px
            import pandas as pd
            from threading import Thread
            
            # Create dashboard app
            app = dash.Dash(__name__)
            
            # Define dashboard layout
            app.layout = html.Div([
                html.H1("Distributed Testing Results Dashboard"),
                
                html.Div([
                    html.H2("Test Results Summary"),
                    dcc.Graph(id='success-rate-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=10*1000,  # in milliseconds (10s)
                        n_intervals=0
                    )
                ]),
                
                html.Div([
                    html.H2("Hardware Performance Comparison"),
                    dcc.Dropdown(
                        id='model-dropdown',
                        placeholder="Select a model"
                    ),
                    dcc.Graph(id='hardware-comparison-graph')
                ]),
                
                html.Div([
                    html.H2("Recent Test Results"),
                    html.Table(id='results-table')
                ])
            ])
            
            # Define callback for updating graphs
            @app.callback(
                [dash.dependencies.Output('success-rate-graph', 'figure'),
                 dash.dependencies.Output('model-dropdown', 'options'),
                 dash.dependencies.Output('results-table', 'children')],
                [dash.dependencies.Input('interval-component', 'n_intervals')]
            )
            def update_dashboard(n):
                # Query database for summary data
                summary_data = self.result_processor.get_summary_by_model_hardware()
                
                # Convert to DataFrame
                if summary_data:
                    df_summary = pd.DataFrame(summary_data, columns=[
                        'model_name', 'hardware_type', 'test_count', 'success_count',
                        'avg_execution_time', 'avg_memory_usage', 'avg_power_consumption'
                    ])
                    
                    # Calculate success rate
                    df_summary['success_rate'] = df_summary['success_count'] / df_summary['test_count'] * 100
                    
                    # Create success rate graph
                    fig = px.bar(
                        df_summary, 
                        x='model_name', 
                        y='success_rate', 
                        color='hardware_type',
                        title='Success Rate by Model and Hardware',
                        labels={'success_rate': 'Success Rate (%)', 'model_name': 'Model', 'hardware_type': 'Hardware'},
                        barmode='group'
                    )
                    
                    # Get model options for dropdown
                    model_options = [{'label': model, 'value': model} 
                                   for model in df_summary['model_name'].unique()]
                    
                    # Get recent test results
                    recent_results = self.result_processor.query_results(
                        "SELECT test_id, model_name, hardware_type, success, execution_time, timestamp " +
                        "FROM test_results ORDER BY timestamp DESC LIMIT 10"
                    )
                    
                    # Create table rows for recent results
                    table_header = [
                        html.Tr([
                            html.Th("Test ID"),
                            html.Th("Model"),
                            html.Th("Hardware"),
                            html.Th("Success"),
                            html.Th("Time (ms)"),
                            html.Th("Timestamp")
                        ])
                    ]
                    
                    table_rows = []
                    for result in recent_results:
                        table_rows.append(html.Tr([
                            html.Td(str(result[0])),
                            html.Td(str(result[1])),
                            html.Td(str(result[2])),
                            html.Td("✅" if result[3] else "❌"),
                            html.Td(f"{result[4]:.2f}"),
                            html.Td(str(result[5]))
                        ]))
                    
                    table = table_header + table_rows
                    
                    return fig, model_options, table
                else:
                    # Return empty data if no results
                    empty_fig = px.bar()
                    empty_fig.update_layout(title="No data available")
                    
                    return empty_fig, [], [html.Tr([html.Td("No data available")])]
            
            # Callback for model selection
            @app.callback(
                dash.dependencies.Output('hardware-comparison-graph', 'figure'),
                [dash.dependencies.Input('model-dropdown', 'value')]
            )
            def update_hardware_comparison(selected_model):
                if not selected_model:
                    empty_fig = px.bar()
                    empty_fig.update_layout(title="Select a model to see hardware comparison")
                    return empty_fig
                
                # Query database for model-specific data
                query = f"""
                    SELECT 
                        hardware_type, 
                        AVG(execution_time) as avg_time,
                        AVG(memory_usage) as avg_memory,
                        COUNT(*) as test_count
                    FROM test_results 
                    WHERE model_name = '{selected_model}'
                    GROUP BY hardware_type
                """
                
                results = self.result_processor.query_results(query)
                
                if results:
                    df = pd.DataFrame(results, columns=[
                        'hardware_type', 'avg_time', 'avg_memory', 'test_count'
                    ])
                    
                    # Create comparison figure
                    fig = px.bar(
                        df, 
                        x='hardware_type', 
                        y=['avg_time', 'avg_memory'],
                        title=f'Performance Metrics for {selected_model}',
                        labels={
                            'hardware_type': 'Hardware', 
                            'value': 'Value', 
                            'variable': 'Metric'
                        },
                        barmode='group'
                    )
                    
                    return fig
                else:
                    empty_fig = px.bar()
                    empty_fig.update_layout(title=f"No data available for {selected_model}")
                    return empty_fig
            
            # Save dashboard app
            self.dashboard_app = app
            
            # Start dashboard in a background thread
            def run_dashboard():
                app.run_server(debug=False, port=self.dashboard_port, host='0.0.0.0')
                
            self.dashboard_thread = Thread(target=run_dashboard)
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
            
            logger.info(f"Started dashboard on port {self.dashboard_port}")
            
        except ImportError as e:
            logger.error(f"Cannot start dashboard, required packages not installed: {e}")
            logger.error("Install dash, plotly, and pandas to enable the dashboard")
            self.enable_dashboard = False
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.enable_dashboard = False
    
    def store_result(self, result):
        """Store a test result.
        
        Args:
            result: Dictionary containing test result data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if caching is enabled
        if self.cache_enabled:
            with self.cache_lock:
                # Clean up cache if needed
                current_time = time.time()
                if current_time - self.last_cache_cleanup > 60:  # Clean every minute
                    self._cleanup_cache()
                    self.last_cache_cleanup = current_time
                
                # Add to cache
                self.result_cache[result.get('test_id')] = {
                    'result': result,
                    'timestamp': current_time
                }
                
                # Check if cache has reached batch size
                if len(self.result_cache) >= self.batch_size:
                    return self._flush_cache()
                
                return True
        else:
            # Direct insertion
            return self.result_processor.store_result(result)
    
    def _cleanup_cache(self):
        """Clean up expired entries from the result cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.result_cache.items():
            if current_time - entry['timestamp'] > self.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.result_cache.pop(key, None)
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _flush_cache(self):
        """Flush the result cache to the database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.cache_lock:
            if not self.result_cache:
                return True
                
            results = [entry['result'] for entry in self.result_cache.values()]
            success, failed = self.result_processor.store_batch_results(results)
            
            if success and not failed:
                # All succeeded, clear cache
                self.result_cache.clear()
                logger.debug(f"Successfully flushed {len(results)} results from cache")
                return True
            elif failed:
                # Some failed, remove successful ones
                for result in results:
                    if result not in failed:
                        self.result_cache.pop(result.get('test_id'), None)
                
                logger.warning(f"Partially flushed cache: {len(results) - len(failed)} succeeded, {len(failed)} failed")
                return False
            else:
                logger.error("Failed to flush cache")
                return False
    
    def store_batch_results(self, results):
        """Store multiple test results.
        
        Args:
            results: List of result dictionaries to store
            
        Returns:
            Tuple[bool, List]: (success, list of failed results)
        """
        # If caching is enabled, add to cache first
        if self.cache_enabled:
            for result in results:
                self.store_result(result)
            
            # Force a cache flush
            success = self._flush_cache()
            
            # Determine which results failed (still in cache)
            if not success:
                failed_ids = set(self.result_cache.keys())
                failed = [r for r in results if r.get('test_id') in failed_ids]
                return False, failed
            
            return True, []
        else:
            # Direct insertion
            return self.result_processor.store_batch_results(results)
    
    def get_result(self, test_id):
        """Get a specific test result.
        
        Args:
            test_id: ID of the test to retrieve
            
        Returns:
            Dict or None: Test result or None if not found
        """
        # Check cache first
        if self.cache_enabled:
            with self.cache_lock:
                if test_id in self.result_cache:
                    return self.result_cache[test_id]['result']
        
        # Query database
        return self.result_processor.get_result_by_id(test_id)
    
    def get_results_by_model(self, model_name, limit=100):
        """Get test results for a specific model.
        
        Args:
            model_name: Name of the model
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.get_results_by_model(model_name, limit)
    
    def get_results_by_hardware(self, hardware_type, limit=100):
        """Get test results for a specific hardware type.
        
        Args:
            hardware_type: Type of hardware
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.get_results_by_hardware(hardware_type, limit)
    
    def get_results_by_worker(self, worker_id, limit=100):
        """Get test results for a specific worker.
        
        Args:
            worker_id: ID of the worker
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.get_results_by_worker(worker_id, limit)
    
    def get_results_by_timerange(self, start_time, end_time=None, limit=100):
        """Get test results within a time range.
        
        Args:
            start_time: Start of the time range (ISO format)
            end_time: End of the time range (ISO format, defaults to now)
            limit: Maximum number of results to return
            
        Returns:
            List: Test results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.get_results_by_timerange(start_time, end_time, limit)
    
    def get_summary_by_model_hardware(self):
        """Get a summary of test results grouped by model and hardware type.
        
        Returns:
            List: Summary of test results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.get_summary_by_model_hardware()
    
    def run_custom_query(self, query, params=None):
        """Run a custom SQL query on the database.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List: Query results
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        return self.result_processor.query_results(query, params)
    
    def generate_report(self, format='markdown', output_file=None, filters=None):
        """Generate a comprehensive report of test results.
        
        Args:
            format: Output format ('markdown', 'html', or 'json')
            output_file: Path to output file (if None, returns the report as a string)
            filters: Dictionary of filters to apply (model, hardware, worker, timerange, etc.)
            
        Returns:
            str or None: Report content if output_file is None, otherwise None
        """
        # Need to flush cache to make sure all results are in database
        if self.cache_enabled:
            self._flush_cache()
            
        # Build query based on filters
        query_parts = ["SELECT * FROM test_results"]
        params = []
        
        if filters:
            conditions = []
            
            if 'model' in filters:
                conditions.append("model_name = ?")
                params.append(filters['model'])
                
            if 'hardware' in filters:
                conditions.append("hardware_type = ?")
                params.append(filters['hardware'])
                
            if 'worker' in filters:
                conditions.append("worker_id = ?")
                params.append(filters['worker'])
                
            if 'start_time' in filters:
                conditions.append("timestamp >= ?")
                params.append(filters['start_time'])
                
            if 'end_time' in filters:
                conditions.append("timestamp <= ?")
                params.append(filters['end_time'])
                
            if 'success' in filters and filters['success'] is not None:
                conditions.append("success = ?")
                params.append(filters['success'])
                
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
        
        query_parts.append("ORDER BY timestamp DESC")
        
        if 'limit' in filters:
            query_parts.append("LIMIT ?")
            params.append(filters['limit'])
        else:
            query_parts.append("LIMIT 1000")  # Default limit
            
        query = " ".join(query_parts)
        
        # Execute query
        results = self.result_processor.query_results(query, params)
        
        # Get summary
        summary_query = """
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                AVG(CASE WHEN success THEN execution_time ELSE NULL END) as avg_execution_time,
                AVG(CASE WHEN success THEN memory_usage ELSE NULL END) as avg_memory_usage,
                COUNT(DISTINCT model_name) as model_count,
                COUNT(DISTINCT hardware_type) as hardware_count,
                COUNT(DISTINCT worker_id) as worker_count
            FROM test_results
        """
        
        if 'WHERE' in query:
            where_clause = query.split('WHERE')[1].split('ORDER')[0]
            summary_query += " WHERE " + where_clause
            
        summary = self.result_processor.query_results(summary_query, params)[0]
        
        # Get model-hardware summary
        model_hardware_summary = self.get_summary_by_model_hardware()
        
        # Generate report
        if format == 'markdown':
            return self._generate_markdown_report(results, summary, model_hardware_summary, output_file)
        elif format == 'html':
            return self._generate_html_report(results, summary, model_hardware_summary, output_file)
        elif format == 'json':
            return self._generate_json_report(results, summary, model_hardware_summary, output_file)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_markdown_report(self, results, summary, model_hardware_summary, output_file):
        """Generate a Markdown report.
        
        Returns:
            str or None: Report content if output_file is None, otherwise None
        """
        report = []
        report.append("# Distributed Testing Framework Report")
        report.append(f"Generated on: {datetime.now().isoformat()}")
        report.append("")
        
        # Overall summary
        report.append("## Summary")
        report.append("")
        report.append(f"- Total tests: {summary[0]}")
        report.append(f"- Success rate: {summary[1] / summary[0] * 100:.2f}% ({summary[1]} / {summary[0]})")
        report.append(f"- Average execution time: {summary[2]:.2f} ms")
        report.append(f"- Average memory usage: {summary[3]:.2f} MB")
        report.append(f"- Unique models: {summary[4]}")
        report.append(f"- Unique hardware types: {summary[5]}")
        report.append(f"- Unique workers: {summary[6]}")
        report.append("")
        
        # Model-hardware summary
        report.append("## Model-Hardware Performance")
        report.append("")
        report.append("| Model | Hardware | Tests | Success Rate | Avg Time (ms) | Avg Memory (MB) | Avg Power (W) |")
        report.append("|-------|----------|-------|--------------|---------------|-----------------|---------------|")
        
        for row in model_hardware_summary:
            model_name = row[0]
            hardware_type = row[1]
            test_count = row[2]
            success_count = row[3]
            success_rate = success_count / test_count * 100 if test_count > 0 else 0
            avg_time = row[4] or 0
            avg_memory = row[5] or 0
            avg_power = row[6] or 0
            
            report.append(f"| {model_name} | {hardware_type} | {test_count} | {success_rate:.2f}% | {avg_time:.2f} | {avg_memory:.2f} | {avg_power:.2f} |")
            
        report.append("")
        
        # Recent tests
        report.append("## Recent Tests")
        report.append("")
        report.append("| Test ID | Model | Hardware | Worker | Success | Time (ms) | Memory (MB) | Timestamp |")
        report.append("|---------|-------|----------|--------|---------|-----------|-------------|-----------|")
        
        for row in results[:20]:  # Show only first 20 for brevity
            test_id = row[1]
            model_name = row[3]
            hardware_type = row[4]
            worker_id = row[2]
            success = "✅" if row[6] else "❌"
            execution_time = row[5] or 0
            memory_usage = row[9] or 0
            timestamp = row[8]
            
            report.append(f"| {test_id} | {model_name} | {hardware_type} | {worker_id} | {success} | {execution_time:.2f} | {memory_usage:.2f} | {timestamp} |")
            
        report.append("")
        report.append("---")
        report.append("Report generated by Distributed Testing Framework")
        
        report_content = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            return None
        else:
            return report_content
    
    def _generate_html_report(self, results, summary, model_hardware_summary, output_file):
        """Generate an HTML report.
        
        Returns:
            str or None: Report content if output_file is None, otherwise None
        """
        # Try to import pandas and plotly for enhanced reporting
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Convert results to DataFrame
            df_results = pd.DataFrame(results, columns=[
                'id', 'test_id', 'worker_id', 'model_name', 'hardware_type', 
                'execution_time', 'success', 'error_message', 'timestamp',
                'memory_usage', 'details', 'power_consumption', 'test_type'
            ])
            
            # Convert model-hardware summary to DataFrame
            df_summary = pd.DataFrame(model_hardware_summary, columns=[
                'model_name', 'hardware_type', 'test_count', 'success_count',
                'avg_execution_time', 'avg_memory_usage', 'avg_power_consumption'
            ])
            
            # Calculate success rate
            df_summary['success_rate'] = df_summary['success_count'] / df_summary['test_count'] * 100
            
            # Create figures
            fig1 = px.bar(
                df_summary, 
                x='model_name', 
                y='success_rate', 
                color='hardware_type',
                title='Success Rate by Model and Hardware',
                labels={'success_rate': 'Success Rate (%)', 'model_name': 'Model', 'hardware_type': 'Hardware'},
                barmode='group'
            )
            
            fig2 = px.bar(
                df_summary, 
                x='model_name', 
                y='avg_execution_time', 
                color='hardware_type',
                title='Average Execution Time by Model and Hardware',
                labels={'avg_execution_time': 'Avg Time (ms)', 'model_name': 'Model', 'hardware_type': 'Hardware'},
                barmode='group'
            )
            
            fig3 = px.bar(
                df_summary, 
                x='model_name', 
                y='avg_memory_usage', 
                color='hardware_type',
                title='Average Memory Usage by Model and Hardware',
                labels={'avg_memory_usage': 'Avg Memory (MB)', 'model_name': 'Model', 'hardware_type': 'Hardware'},
                barmode='group'
            )
            
            # Create HTML with embedded plots
            html_parts = []
            html_parts.append("<!DOCTYPE html>")
            html_parts.append("<html>")
            html_parts.append("<head>")
            html_parts.append("    <title>Distributed Testing Framework Report</title>")
            html_parts.append("    <style>")
            html_parts.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
            html_parts.append("        h1, h2 { color: #333; }")
            html_parts.append("        table { border-collapse: collapse; width: 100%; }")
            html_parts.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_parts.append("        th { background-color: #f2f2f2; }")
            html_parts.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
            html_parts.append("        .success { color: green; }")
            html_parts.append("        .failure { color: red; }")
            html_parts.append("        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }")
            html_parts.append("        .summary-item { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }")
            html_parts.append("        .plot-container { margin: 20px 0; }")
            html_parts.append("    </style>")
            html_parts.append("</head>")
            html_parts.append("<body>")
            html_parts.append("    <h1>Distributed Testing Framework Report</h1>")
            html_parts.append(f"    <p>Generated on: {datetime.now().isoformat()}</p>")
            
            # Summary section
            html_parts.append("    <h2>Summary</h2>")
            html_parts.append("    <div class='summary'>")
            html_parts.append(f"        <div class='summary-item'><strong>Total Tests:</strong> {summary[0]}</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Success Rate:</strong> {summary[1] / summary[0] * 100:.2f}% ({summary[1]} / {summary[0]})</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Avg Execution Time:</strong> {summary[2]:.2f} ms</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Avg Memory Usage:</strong> {summary[3]:.2f} MB</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Unique Models:</strong> {summary[4]}</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Unique Hardware:</strong> {summary[5]}</div>")
            html_parts.append(f"        <div class='summary-item'><strong>Unique Workers:</strong> {summary[6]}</div>")
            html_parts.append("    </div>")
            
            # Plots
            html_parts.append("    <h2>Performance Visualizations</h2>")
            html_parts.append("    <div class='plot-container'>")
            html_parts.append(f"        {pio.to_html(fig1, full_html=False)}")
            html_parts.append("    </div>")
            html_parts.append("    <div class='plot-container'>")
            html_parts.append(f"        {pio.to_html(fig2, full_html=False)}")
            html_parts.append("    </div>")
            html_parts.append("    <div class='plot-container'>")
            html_parts.append(f"        {pio.to_html(fig3, full_html=False)}")
            html_parts.append("    </div>")
            
            # Model-hardware summary table
            html_parts.append("    <h2>Model-Hardware Performance</h2>")
            html_parts.append("    <table>")
            html_parts.append("        <tr>")
            html_parts.append("            <th>Model</th>")
            html_parts.append("            <th>Hardware</th>")
            html_parts.append("            <th>Tests</th>")
            html_parts.append("            <th>Success Rate</th>")
            html_parts.append("            <th>Avg Time (ms)</th>")
            html_parts.append("            <th>Avg Memory (MB)</th>")
            html_parts.append("            <th>Avg Power (W)</th>")
            html_parts.append("        </tr>")
            
            for row in model_hardware_summary:
                model_name = row[0]
                hardware_type = row[1]
                test_count = row[2]
                success_count = row[3]
                success_rate = success_count / test_count * 100 if test_count > 0 else 0
                avg_time = row[4] or 0
                avg_memory = row[5] or 0
                avg_power = row[6] or 0
                
                html_parts.append("        <tr>")
                html_parts.append(f"            <td>{model_name}</td>")
                html_parts.append(f"            <td>{hardware_type}</td>")
                html_parts.append(f"            <td>{test_count}</td>")
                html_parts.append(f"            <td>{success_rate:.2f}%</td>")
                html_parts.append(f"            <td>{avg_time:.2f}</td>")
                html_parts.append(f"            <td>{avg_memory:.2f}</td>")
                html_parts.append(f"            <td>{avg_power:.2f}</td>")
                html_parts.append("        </tr>")
                
            html_parts.append("    </table>")
            
            # Recent tests table
            html_parts.append("    <h2>Recent Tests</h2>")
            html_parts.append("    <table>")
            html_parts.append("        <tr>")
            html_parts.append("            <th>Test ID</th>")
            html_parts.append("            <th>Model</th>")
            html_parts.append("            <th>Hardware</th>")
            html_parts.append("            <th>Worker</th>")
            html_parts.append("            <th>Success</th>")
            html_parts.append("            <th>Time (ms)</th>")
            html_parts.append("            <th>Memory (MB)</th>")
            html_parts.append("            <th>Timestamp</th>")
            html_parts.append("        </tr>")
            
            for row in results[:20]:  # Show only first 20 for brevity
                test_id = row[1]
                model_name = row[3]
                hardware_type = row[4]
                worker_id = row[2]
                success = row[6]
                execution_time = row[5] or 0
                memory_usage = row[9] or 0
                timestamp = row[8]
                
                html_parts.append("        <tr>")
                html_parts.append(f"            <td>{test_id}</td>")
                html_parts.append(f"            <td>{model_name}</td>")
                html_parts.append(f"            <td>{hardware_type}</td>")
                html_parts.append(f"            <td>{worker_id}</td>")
                if success:
                    html_parts.append(f"            <td class='success'>✅ Success</td>")
                else:
                    html_parts.append(f"            <td class='failure'>❌ Failed</td>")
                html_parts.append(f"            <td>{execution_time:.2f}</td>")
                html_parts.append(f"            <td>{memory_usage:.2f}</td>")
                html_parts.append(f"            <td>{timestamp}</td>")
                html_parts.append("        </tr>")
                
            html_parts.append("    </table>")
            
            html_parts.append("    <hr>")
            html_parts.append("    <p>Report generated by Distributed Testing Framework</p>")
            html_parts.append("</body>")
            html_parts.append("</html>")
            
            html_content = "\n".join(html_parts)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(html_content)
                return None
            else:
                return html_content
                
        except ImportError:
            # Fallback to basic HTML if pandas or plotly is not available
            logger.warning("pandas or plotly not available, generating basic HTML report")
            
            html_parts = []
            html_parts.append("<!DOCTYPE html>")
            html_parts.append("<html>")
            html_parts.append("<head>")
            html_parts.append("    <title>Distributed Testing Framework Report</title>")
            html_parts.append("    <style>")
            html_parts.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
            html_parts.append("        h1, h2 { color: #333; }")
            html_parts.append("        table { border-collapse: collapse; width: 100%; }")
            html_parts.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            html_parts.append("        th { background-color: #f2f2f2; }")
            html_parts.append("        tr:nth-child(even) { background-color: #f9f9f9; }")
            html_parts.append("        .success { color: green; }")
            html_parts.append("        .failure { color: red; }")
            html_parts.append("    </style>")
            html_parts.append("</head>")
            html_parts.append("<body>")
            html_parts.append("    <h1>Distributed Testing Framework Report</h1>")
            html_parts.append(f"    <p>Generated on: {datetime.now().isoformat()}</p>")
            
            # Summary section
            html_parts.append("    <h2>Summary</h2>")
            html_parts.append("    <ul>")
            html_parts.append(f"        <li><strong>Total Tests:</strong> {summary[0]}</li>")
            html_parts.append(f"        <li><strong>Success Rate:</strong> {summary[1] / summary[0] * 100:.2f}% ({summary[1]} / {summary[0]})</li>")
            html_parts.append(f"        <li><strong>Avg Execution Time:</strong> {summary[2]:.2f} ms</li>")
            html_parts.append(f"        <li><strong>Avg Memory Usage:</strong> {summary[3]:.2f} MB</li>")
            html_parts.append(f"        <li><strong>Unique Models:</strong> {summary[4]}</li>")
            html_parts.append(f"        <li><strong>Unique Hardware:</strong> {summary[5]}</li>")
            html_parts.append(f"        <li><strong>Unique Workers:</strong> {summary[6]}</li>")
            html_parts.append("    </ul>")
            
            # The rest of the report is the same as the full version
            # Model-hardware summary table
            html_parts.append("    <h2>Model-Hardware Performance</h2>")
            html_parts.append("    <table>")
            html_parts.append("        <tr>")
            html_parts.append("            <th>Model</th>")
            html_parts.append("            <th>Hardware</th>")
            html_parts.append("            <th>Tests</th>")
            html_parts.append("            <th>Success Rate</th>")
            html_parts.append("            <th>Avg Time (ms)</th>")
            html_parts.append("            <th>Avg Memory (MB)</th>")
            html_parts.append("            <th>Avg Power (W)</th>")
            html_parts.append("        </tr>")
            
            for row in model_hardware_summary:
                model_name = row[0]
                hardware_type = row[1]
                test_count = row[2]
                success_count = row[3]
                success_rate = success_count / test_count * 100 if test_count > 0 else 0
                avg_time = row[4] or 0
                avg_memory = row[5] or 0
                avg_power = row[6] or 0
                
                html_parts.append("        <tr>")
                html_parts.append(f"            <td>{model_name}</td>")
                html_parts.append(f"            <td>{hardware_type}</td>")
                html_parts.append(f"            <td>{test_count}</td>")
                html_parts.append(f"            <td>{success_rate:.2f}%</td>")
                html_parts.append(f"            <td>{avg_time:.2f}</td>")
                html_parts.append(f"            <td>{avg_memory:.2f}</td>")
                html_parts.append(f"            <td>{avg_power:.2f}</td>")
                html_parts.append("        </tr>")
                
            html_parts.append("    </table>")
            
            # Recent tests table
            html_parts.append("    <h2>Recent Tests</h2>")
            html_parts.append("    <table>")
            html_parts.append("        <tr>")
            html_parts.append("            <th>Test ID</th>")
            html_parts.append("            <th>Model</th>")
            html_parts.append("            <th>Hardware</th>")
            html_parts.append("            <th>Worker</th>")
            html_parts.append("            <th>Success</th>")
            html_parts.append("            <th>Time (ms)</th>")
            html_parts.append("            <th>Memory (MB)</th>")
            html_parts.append("            <th>Timestamp</th>")
            html_parts.append("        </tr>")
            
            for row in results[:20]:  # Show only first 20 for brevity
                test_id = row[1]
                model_name = row[3]
                hardware_type = row[4]
                worker_id = row[2]
                success = row[6]
                execution_time = row[5] or 0
                memory_usage = row[9] or 0
                timestamp = row[8]
                
                html_parts.append("        <tr>")
                html_parts.append(f"            <td>{test_id}</td>")
                html_parts.append(f"            <td>{model_name}</td>")
                html_parts.append(f"            <td>{hardware_type}</td>")
                html_parts.append(f"            <td>{worker_id}</td>")
                if success:
                    html_parts.append(f"            <td class='success'>✅ Success</td>")
                else:
                    html_parts.append(f"            <td class='failure'>❌ Failed</td>")
                html_parts.append(f"            <td>{execution_time:.2f}</td>")
                html_parts.append(f"            <td>{memory_usage:.2f}</td>")
                html_parts.append(f"            <td>{timestamp}</td>")
                html_parts.append("        </tr>")
                
            html_parts.append("    </table>")
            
            html_parts.append("    <hr>")
            html_parts.append("    <p>Report generated by Distributed Testing Framework</p>")
            html_parts.append("    <p><em>Note: Install pandas and plotly for enhanced visualizations</em></p>")
            html_parts.append("</body>")
            html_parts.append("</html>")
            
            html_content = "\n".join(html_parts)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(html_content)
                return None
            else:
                return html_content
    
    def _generate_json_report(self, results, summary, model_hardware_summary, output_file):
        """Generate a JSON report.
        
        Returns:
            str or None: Report content if output_file is None, otherwise None
        """
        # Convert results to list of dicts
        results_list = []
        for row in results:
            result = {
                'id': row[0],
                'test_id': row[1],
                'worker_id': row[2],
                'model_name': row[3],
                'hardware_type': row[4],
                'execution_time': row[5],
                'success': row[6],
                'error_message': row[7],
                'timestamp': row[8],
                'memory_usage': row[9],
                'details': json.loads(row[10]) if row[10] else {},
                'power_consumption': row[11],
                'test_type': row[12]
            }
            results_list.append(result)
            
        # Create summary dict
        summary_dict = {
            'total_count': summary[0],
            'success_count': summary[1],
            'avg_execution_time': summary[2],
            'avg_memory_usage': summary[3],
            'model_count': summary[4],
            'hardware_count': summary[5],
            'worker_count': summary[6]
        }
        
        # Convert model-hardware summary to list of dicts
        model_hardware_list = []
        for row in model_hardware_summary:
            model_hardware = {
                'model_name': row[0],
                'hardware_type': row[1],
                'test_count': row[2],
                'success_count': row[3],
                'avg_execution_time': row[4],
                'avg_memory_usage': row[5],
                'avg_power_consumption': row[6]
            }
            model_hardware_list.append(model_hardware)
            
        # Create report dict
        report_dict = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary_dict,
            'model_hardware_summary': model_hardware_list,
            'results': results_list[:100]  # Limit to 100 results for JSON format
        }
        
        # Convert to JSON
        json_content = json.dumps(report_dict, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_content)
            return None
        else:
            return json_content
    
    def close(self):
        """Close the integration and free resources."""
        # Flush cache
        if self.cache_enabled:
            self._flush_cache()
            
        # Stop threads
        self.stop_event.set()
        
        if self.vacuum_thread:
            self.vacuum_thread.join(timeout=5)
            
        if self.dashboard_thread:
            # Dashboard thread is daemon, no need to join
            self.dashboard_app = None
            self.dashboard_thread = None
            
        # Close database
        if self.result_processor:
            self.result_processor.close()
            
        logger.info("Closed coordinator DuckDB integration")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Coordinator DuckDB Integration")
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for result insertion")
    parser.add_argument("--disable-cache", action="store_true", help="Disable result caching")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")
    parser.add_argument("--vacuum-interval", type=int, default=24, help="Vacuum interval in hours (0 to disable)")
    parser.add_argument("--enable-dashboard", action="store_true", help="Enable result dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--generate-report", action="store_true", help="Generate a test report")
    parser.add_argument("--report-format", choices=["markdown", "html", "json"], default="markdown", help="Report format")
    parser.add_argument("--report-output", help="Report output file")
    parser.add_argument("--model", help="Filter by model name")
    parser.add_argument("--hardware", help="Filter by hardware type")
    parser.add_argument("--days", type=int, default=7, help="Days of data to include in report")
    
    args = parser.parse_args()
    
    integration = CoordinatorDuckDBIntegration(
        db_path=args.db_path,
        batch_size=args.batch_size,
        cache_enabled=not args.disable_cache,
        cache_ttl_seconds=args.cache_ttl,
        auto_vacuum_interval_hours=args.vacuum_interval,
        enable_dashboard=args.enable_dashboard,
        dashboard_port=args.dashboard_port
    )
    
    if args.generate_report:
        filters = {}
        
        if args.model:
            filters['model'] = args.model
            
        if args.hardware:
            filters['hardware'] = args.hardware
            
        if args.days:
            start_time = (datetime.now() - timedelta(days=args.days)).isoformat()
            filters['start_time'] = start_time
            
        report = integration.generate_report(
            format=args.report_format,
            output_file=args.report_output,
            filters=filters
        )
        
        if report:
            print(report)
    
    if args.enable_dashboard:
        print(f"Dashboard started on port {args.dashboard_port}. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping dashboard...")
    
    integration.close()