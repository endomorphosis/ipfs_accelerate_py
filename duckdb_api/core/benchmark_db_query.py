#!/usr/bin/env python
"""
Benchmark Database Query Tool for the IPFS Accelerate Python Framework.

This module provides a CLI tool for querying the benchmark database and
generating reports.

Usage:
    python benchmark_db_query.py --sql "SELECT model, hardware, AVG(throughput) FROM benchmarks GROUP BY model, hardware"
    python benchmark_db_query.py --report performance --format html --output benchmark_report.html
    python benchmark_db_query.py --model bert-base-uncased --metric throughput --compare-hardware
"""

import os
import sys
import json
import logging
# Import comprehensive benchmark timing report generator
try:
    from benchmark_timing_report import BenchmarkTimingReport
except ImportError:
    # Try relative import as fallback
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from benchmark_timing_report import BenchmarkTimingReport
    except ImportError:
        logger.warning("BenchmarkTimingReport could not be imported. Timing report generation will not be available.")

import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import matplotlib

# Add DuckDB database support
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import duckdb
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas numpy matplotlib seaborn")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkDBQuery:
    """
    Query tool for the benchmark database.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark database query tool.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Verify database exists
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            logger.info("Creating a new database file")
        
        # Connect to database
        try:
            self.conn = duckdb.connect(db_path, read_only=False)
            logger.info(f"Connected to database: {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
        
        # Check for tables
        self.available_tables = self._get_available_tables()
        logger.info(f"Available tables: {', '.join(self.available_tables)}")
    
    def _get_available_tables(self) -> List[str]:
        """Get list of available tables in the database"""
        tables = []
        try:
            # Query for tables
            result = self.conn.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
        
        return tables
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query on the database.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            logger.info(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            logger.error(f"Query was: {sql}")
            return pd.DataFrame()
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get a summary of performance data by model and hardware.
        
        Returns:
            DataFrame with performance summary
        """
        if "benchmark_performance" not in self.available_tables:
            logger.error("Performance data table not found in database")
            return pd.DataFrame()
        
        sql = """
            SELECT
                model,
                hardware,
                COUNT(*) as num_runs,
                AVG(throughput) as avg_throughput,
                MIN(throughput) as min_throughput,
                MAX(throughput) as max_throughput,
                AVG(latency_avg) as avg_latency,
                MIN(memory_peak) as min_memory,
                MAX(memory_peak) as max_memory,
                MAX(timestamp) as latest_run
            FROM benchmark_performance
            GROUP BY model, hardware
            ORDER BY model, avg_throughput DESC
        """
        
        return self.execute_sql(sql)
    
    def get_hardware_summary(self) -> pd.DataFrame:
        """
        Get a summary of hardware detection data.
        
        Returns:
            DataFrame with hardware summary
        """
        if "benchmark_hardware" not in self.available_tables:
            logger.error("Hardware data table not found in database")
            return pd.DataFrame()
        
        sql = """
            SELECT
                hardware_type,
                device_name,
                COUNT(*) as detection_count,
                SUM(CASE WHEN is_available THEN 1 ELSE 0 END) as available_count,
                MAX(memory_total) as max_memory,
                MAX(timestamp) as latest_detection
            FROM benchmark_hardware
            GROUP BY hardware_type, device_name
            ORDER BY hardware_type, device_name
        """
        
        return self.execute_sql(sql)
    
    def get_compatibility_summary(self) -> pd.DataFrame:
        """
        Get a summary of compatibility data by model and hardware.
        
        Returns:
            DataFrame with compatibility summary
        """
        if "benchmark_compatibility" not in self.available_tables:
            logger.error("Compatibility data table not found in database")
            return pd.DataFrame()
        
        sql = """
            SELECT
                model,
                hardware_type,
                COUNT(*) as test_count,
                SUM(CASE WHEN is_compatible THEN 1 ELSE 0 END) as compatible_count,
                (SUM(CASE WHEN is_compatible THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as compatibility_percentage,
                MAX(timestamp) as latest_test
            FROM benchmark_compatibility
            GROUP BY model, hardware_type
            ORDER BY model, hardware_type
        """
        
        return self.execute_sql(sql)
    
    def compare_hardware(self, model: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Compare hardware performance for a specific model.
        
        Args:
            model: Model name to compare
            metric: Metric to compare (throughput, latency_avg, memory_peak)
            
        Returns:
            DataFrame with hardware comparison
        """
        if "benchmark_performance" not in self.available_tables:
            logger.error("Performance data table not found in database")
            return pd.DataFrame()
        
        valid_metrics = ["throughput", "latency_avg", "memory_peak"]
        if metric not in valid_metrics:
            logger.error(f"Invalid metric: {metric}. Valid metrics are: {', '.join(valid_metrics)}")
            return pd.DataFrame()
        
        # First try the new schema with performance_results
        if "performance_results" in self.available_tables:
            try:
                # Try to query using the model_id from models table
                sql = f"""
                    SELECT 
                        h.hardware_type as hardware,
                        AVG(p.throughput_items_per_second) as avg_{metric},
                        MIN(p.throughput_items_per_second) as min_{metric},
                        MAX(p.throughput_items_per_second) as max_{metric},
                        stddev(p.throughput_items_per_second) as stddev_{metric},
                        COUNT(*) as run_count
                    FROM performance_results p
                    JOIN models m ON p.model_id = m.model_id
                    JOIN hardware_platforms h ON p.hardware_id = h.hardware_id
                    WHERE m.model_name = '{model}'
                    GROUP BY h.hardware_type
                    ORDER BY avg_{metric} {"DESC" if metric == "throughput" else "ASC"}
                """
                result = self.execute_sql(sql)
                if not result.empty:
                    return result
            except Exception as e:
                logger.warning(f"Error querying performance_results: {e}")
        
        # Fallback to the old schema
        sql = f"""
            SELECT
                hardware,
                AVG({metric}) as avg_{metric},
                MIN({metric}) as min_{metric},
                MAX({metric}) as max_{metric},
                stddev({metric}) as stddev_{metric},
                COUNT(*) as run_count
            FROM benchmark_performance
            WHERE model = '{model}'
            GROUP BY hardware
            ORDER BY avg_{metric} {"DESC" if metric == "throughput" else "ASC"}
        """
        
        return self.execute_sql(sql)
    
    def compare_models(self, hardware: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Compare model performance on a specific hardware.
        
        Args:
            hardware: Hardware type to compare
            metric: Metric to compare (throughput, latency_avg, memory_peak)
            
        Returns:
            DataFrame with model comparison
        """
        if "benchmark_performance" not in self.available_tables:
            logger.error("Performance data table not found in database")
            return pd.DataFrame()
        
        valid_metrics = ["throughput", "latency_avg", "memory_peak"]
        if metric not in valid_metrics:
            logger.error(f"Invalid metric: {metric}. Valid metrics are: {', '.join(valid_metrics)}")
            return pd.DataFrame()
        
        sql = f"""
            SELECT
                model,
                AVG({metric}) as avg_{metric},
                MIN({metric}) as min_{metric},
                MAX({metric}) as max_{metric},
                stddev({metric}) as stddev_{metric},
                COUNT(*) as run_count
            FROM benchmark_performance
            WHERE hardware = '{hardware}'
            GROUP BY model
            ORDER BY avg_{metric} {"DESC" if metric == "throughput" else "ASC"}
        """
        
        return self.execute_sql(sql)
    
    def get_performance_trends(self, model: str, hardware: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Get performance trends over time for a specific model and hardware.
        
        Args:
            model: Model name
            hardware: Hardware type
            metric: Metric to track (throughput, latency_avg, memory_peak)
            
        Returns:
            DataFrame with performance trends
        """
        if "benchmark_performance" not in self.available_tables:
            logger.error("Performance data table not found in database")
            return pd.DataFrame()
        
        valid_metrics = ["throughput", "latency_avg", "memory_peak"]
        if metric not in valid_metrics:
            logger.error(f"Invalid metric: {metric}. Valid metrics are: {', '.join(valid_metrics)}")
            return pd.DataFrame()
        
        sql = f"""
            SELECT
                model,
                hardware,
                {metric},
                timestamp,
                source_file
            FROM benchmark_performance
            WHERE model = '{model}' AND hardware = '{hardware}'
            ORDER BY timestamp
        """
        
        return self.execute_sql(sql)
    
    def get_compatibility_matrix(self) -> pd.DataFrame:
        """
        Generate a compatibility matrix for all models and hardware types.
        
        Returns:
            DataFrame with compatibility matrix
        """
        if "benchmark_compatibility" not in self.available_tables:
            logger.error("Compatibility data table not found in database")
            return pd.DataFrame()
        
        # Get latest compatibility status for each model and hardware
        sql = """
            SELECT
                model,
                hardware_type,
                is_compatible,
                compatibility_level
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY model, hardware_type
                        ORDER BY timestamp DESC
                    ) as row_num
                FROM benchmark_compatibility
            ) WHERE row_num = 1
        """
        
        df = self.execute_sql(sql)
        
        # Pivot to create matrix
        if not df.empty:
            pivot_df = df.pivot(index='model', columns='hardware_type', values='is_compatible')
            # Convert boolean to string for better readability
            pivot_df = pivot_df.astype(str).replace({'True': '✅', 'False': '❌', 'nan': '❓'})
            return pivot_df
        else:
            return pd.DataFrame()
    
    def export_data(self, category: str, output_file: str, format: str = "csv") -> bool:
        """
        Export data from a specific category to a file.
        
        Args:
            category: Data category (performance, hardware, compatibility)
            output_file: Output file path
            format: Output format (csv, json, html, xlsx)
            
        Returns:
            True if successful, False otherwise
        """
        table_name = f"benchmark_{category}"
        if table_name not in self.available_tables:
            logger.error(f"Table {table_name} not found in database")
            return False
        
        # Get data
        sql = f"SELECT * FROM {table_name}"
        df = self.execute_sql(sql)
        
        if df.empty:
            logger.error(f"No data found in table {table_name}")
            return False
        
        # Export based on format
        try:
            if format == "csv":
                df.to_csv(output_file, index=False)
            elif format == "json":
                df.to_json(output_file, orient="records", indent=2)
            elif format == "html":
                df.to_html(output_file, index=False)
            elif format == "xlsx":
                df.to_excel(output_file, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(df)} rows to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def generate_performance_report(self, output_file: str, format: str = "html") -> bool:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_file: Output file path
            format: Output format (html, md, txt)
            
        Returns:
            True if successful, False otherwise
        """
        if "benchmark_performance" not in self.available_tables:
            logger.error("Performance data table not found in database")
            return False
        
        # Get performance summary
        summary_df = self.get_performance_summary()
        
        if summary_df.empty:
            logger.error("No performance data found")
            return False
        
        # Get top models by throughput
        sql = """
            WITH ranked_models AS (
                SELECT
                    model,
                    hardware,
                    throughput,
                    ROW_NUMBER() OVER (PARTITION BY model ORDER BY throughput DESC) as row_num
                FROM benchmark_performance
            )
            SELECT
                model,
                throughput as max_throughput,
                hardware as best_hardware
            FROM ranked_models
            WHERE row_num = 1
            ORDER BY max_throughput DESC
            LIMIT 10
        """
        top_models = self.execute_sql(sql)
        
        # Get latest runs
        sql = """
            SELECT
                model,
                hardware,
                throughput,
                latency_avg,
                memory_peak,
                timestamp
            FROM benchmark_performance
            ORDER BY timestamp DESC
            LIMIT 20
        """
        latest_runs = self.execute_sql(sql)
        
        # Generate visualizations
        figures = []
        
        # 1. Performance by hardware type
        if not summary_df.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            hardware_perf = summary_df.groupby('hardware')['avg_throughput'].mean().reset_index()
            hardware_perf = hardware_perf.sort_values('avg_throughput', ascending=False)
            
            sns.barplot(x='hardware', y='avg_throughput', data=hardware_perf, ax=ax1)
            ax1.set_title('Average Throughput by Hardware Type')
            ax1.set_xlabel('Hardware')
            ax1.set_ylabel('Average Throughput')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            fig1_path = output_file.replace('.' + format, '_hardware_throughput.png')
            fig1.savefig(fig1_path)
            figures.append(fig1_path)
            
            # 2. Memory usage by model
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            model_memory = summary_df.groupby('model')['max_memory'].max().reset_index()
            model_memory = model_memory.sort_values('max_memory', ascending=False).head(10)
            
            sns.barplot(x='model', y='max_memory', data=model_memory, ax=ax2)
            ax2.set_title('Maximum Memory Usage by Model')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Maximum Memory (MB)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            fig2_path = output_file.replace('.' + format, '_model_memory.png')
            fig2.savefig(fig2_path)
            figures.append(fig2_path)
        
        # Generate report based on format
        try:
            if format == "html":
                # HTML Report
                html_content = f"""
                <html>
                <head>
                    <title>Performance Benchmark Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333366; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .visualization {{ margin: 20px 0; text-align: center; }}
                        .visualization img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Performance Benchmark Report</h1>
                    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Performance Summary</h2>
                    {summary_df.to_html(index=False)}
                    
                    <h2>Top Models by Throughput</h2>
                    {top_models.to_html(index=False)}
                    
                    <h2>Latest Benchmark Runs</h2>
                    {latest_runs.to_html(index=False)}
                    
                    <h2>Visualizations</h2>
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Visualization {i+1}</h3>
                        <img src="{fig_name}" alt="Visualization {i+1}">
                    </div>
                    """
                
                html_content += """
                </body>
                </html>
                """
                
                with open(output_file, 'w') as f:
                    f.write(html_content)
                
            elif format == "md":
                # Markdown Report
                md_content = f"""
                # Performance Benchmark Report
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Performance Summary
                
                {summary_df.to_markdown(index=False)}
                
                ## Top Models by Throughput
                
                {top_models.to_markdown(index=False)}
                
                ## Latest Benchmark Runs
                
                {latest_runs.to_markdown(index=False)}
                
                ## Visualizations
                
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    md_content += f"""
                ### Visualization {i+1}
                
                ![Visualization {i+1}]({fig_name})
                
                """
                
                with open(output_file, 'w') as f:
                    f.write(md_content)
                
            elif format == "txt":
                # Text Report
                txt_content = f"""
                Performance Benchmark Report
                ===========================
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Performance Summary
                ------------------
                
                {summary_df.to_string(index=False)}
                
                Top Models by Throughput
                -----------------------
                
                {top_models.to_string(index=False)}
                
                Latest Benchmark Runs
                --------------------
                
                {latest_runs.to_string(index=False)}
                
                Visualizations saved to:
                {os.path.dirname(output_file)}
                """
                
                with open(output_file, 'w') as f:
                    f.write(txt_content)
                
            else:
                logger.error(f"Unsupported report format: {format}")
                return False
            
            logger.info(f"Generated performance report: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return False
    
    def generate_hardware_report(self, output_file: str, format: str = "html") -> bool:
        """
        Generate a comprehensive hardware report.
        
        Args:
            output_file: Output file path
            format: Output format (html, md, txt)
            
        Returns:
            True if successful, False otherwise
        """
        if "benchmark_hardware" not in self.available_tables:
            logger.error("Hardware data table not found in database")
            return False
        
        # Get hardware summary
        summary_df = self.get_hardware_summary()
        
        if summary_df.empty:
            logger.error("No hardware data found")
            return False
        
        # Get latest hardware detection
        sql = """
            SELECT
                hardware_type,
                device_name,
                is_available,
                platform,
                driver_version,
                memory_total,
                memory_free,
                timestamp
            FROM benchmark_hardware
            ORDER BY timestamp DESC
            LIMIT 20
        """
        latest_detection = self.execute_sql(sql)
        
        # Get availability over time
        sql = """
            SELECT
                hardware_type,
                SUM(CASE WHEN is_available THEN 1 ELSE 0 END) as available_count,
                COUNT(*) as total_count,
                (SUM(CASE WHEN is_available THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as availability_percentage
            FROM benchmark_hardware
            GROUP BY hardware_type
            ORDER BY hardware_type
        """
        availability = self.execute_sql(sql)
        
        # Generate visualizations
        figures = []
        
        # 1. Hardware availability chart
        if not availability.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            sns.barplot(x='hardware_type', y='availability_percentage', data=availability, ax=ax1)
            ax1.set_title('Hardware Availability Percentage')
            ax1.set_xlabel('Hardware Type')
            ax1.set_ylabel('Availability (%)')
            ax1.set_ylim(0, 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            fig1_path = output_file.replace('.' + format, '_hardware_availability.png')
            fig1.savefig(fig1_path)
            figures.append(fig1_path)
            
            # 2. Memory by hardware type
            memory_data = latest_detection.copy()
            memory_data = memory_data[memory_data['memory_total'] > 0]
            
            if not memory_data.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                sns.barplot(x='device_name', y='memory_total', data=memory_data, ax=ax2)
                ax2.set_title('Total Memory by Device')
                ax2.set_xlabel('Device')
                ax2.set_ylabel('Total Memory (MB)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                fig2_path = output_file.replace('.' + format, '_device_memory.png')
                fig2.savefig(fig2_path)
                figures.append(fig2_path)
        
        # Generate report based on format
        try:
            if format == "html":
                # HTML Report
                html_content = f"""
                <html>
                <head>
                    <title>Hardware Detection Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333366; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .visualization {{ margin: 20px 0; text-align: center; }}
                        .visualization img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Hardware Detection Report</h1>
                    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Hardware Summary</h2>
                    {summary_df.to_html(index=False)}
                    
                    <h2>Hardware Availability</h2>
                    {availability.to_html(index=False)}
                    
                    <h2>Latest Hardware Detection</h2>
                    {latest_detection.to_html(index=False)}
                    
                    <h2>Visualizations</h2>
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Visualization {i+1}</h3>
                        <img src="{fig_name}" alt="Visualization {i+1}">
                    </div>
                    """
                
                html_content += """
                </body>
                </html>
                """
                
                with open(output_file, 'w') as f:
                    f.write(html_content)
                
            elif format == "md":
                # Markdown Report
                md_content = f"""
                # Hardware Detection Report
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Hardware Summary
                
                {summary_df.to_markdown(index=False)}
                
                ## Hardware Availability
                
                {availability.to_markdown(index=False)}
                
                ## Latest Hardware Detection
                
                {latest_detection.to_markdown(index=False)}
                
                ## Visualizations
                
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    md_content += f"""
                ### Visualization {i+1}
                
                ![Visualization {i+1}]({fig_name})
                
                """
                
                with open(output_file, 'w') as f:
                    f.write(md_content)
                
            elif format == "txt":
                # Text Report
                txt_content = f"""
                Hardware Detection Report
                ========================
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Hardware Summary
                ---------------
                
                {summary_df.to_string(index=False)}
                
                Hardware Availability
                --------------------
                
                {availability.to_string(index=False)}
                
                Latest Hardware Detection
                ------------------------
                
                {latest_detection.to_string(index=False)}
                
                Visualizations saved to:
                {os.path.dirname(output_file)}
                """
                
                with open(output_file, 'w') as f:
                    f.write(txt_content)
                
            else:
                logger.error(f"Unsupported report format: {format}")
                return False
            
            logger.info(f"Generated hardware report: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating hardware report: {e}")
            return False
    
    def generate_compatibility_report(self, output_file: str, format: str = "html") -> bool:
        """
        Generate a comprehensive compatibility report.
        
        Args:
            output_file: Output file path
            format: Output format (html, md, txt)
            
        Returns:
            True if successful, False otherwise
        """
        if "benchmark_compatibility" not in self.available_tables:
            logger.error("Compatibility data table not found in database")
            return False
        
        # Get compatibility summary
        summary_df = self.get_compatibility_summary()
        
        if summary_df.empty:
            logger.error("No compatibility data found")
            return False
        
        # Get compatibility matrix
        matrix_df = self.get_compatibility_matrix()
        
        # Get problem models/hardware
        sql = """
            SELECT
                model,
                hardware_type,
                error_message,
                error_type,
                timestamp
            FROM benchmark_compatibility
            WHERE NOT is_compatible
            ORDER BY timestamp DESC
            LIMIT 20
        """
        problems = self.execute_sql(sql)
        
        # Get overall compatibility by hardware type
        sql = """
            SELECT
                hardware_type,
                COUNT(*) as total_tests,
                SUM(CASE WHEN is_compatible THEN 1 ELSE 0 END) as compatible_tests,
                (SUM(CASE WHEN is_compatible THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as compatibility_percentage
            FROM benchmark_compatibility
            GROUP BY hardware_type
            ORDER BY compatibility_percentage DESC
        """
        hw_compat = self.execute_sql(sql)
        
        # Generate visualizations
        figures = []
        
        # 1. Hardware compatibility percentage
        if not hw_compat.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            sns.barplot(x='hardware_type', y='compatibility_percentage', data=hw_compat, ax=ax1)
            ax1.set_title('Compatibility Percentage by Hardware Type')
            ax1.set_xlabel('Hardware Type')
            ax1.set_ylabel('Compatibility (%)')
            ax1.set_ylim(0, 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            fig1_path = output_file.replace('.' + format, '_hardware_compatibility.png')
            fig1.savefig(fig1_path)
            figures.append(fig1_path)
            
            # 2. Model compatibility count
            model_compat = summary_df.groupby('model').agg({
                'test_count': 'sum',
                'compatible_count': 'sum'
            }).reset_index()
            
            model_compat['compatibility_percentage'] = (model_compat['compatible_count'] * 100.0 / model_compat['test_count'])
            model_compat = model_compat.sort_values('compatibility_percentage', ascending=False)
            
            if len(model_compat) > 10:
                model_compat = model_compat.head(10)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            sns.barplot(x='model', y='compatibility_percentage', data=model_compat, ax=ax2)
            ax2.set_title('Compatibility Percentage by Model (Top 10)')
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Compatibility (%)')
            ax2.set_ylim(0, 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            fig2_path = output_file.replace('.' + format, '_model_compatibility.png')
            fig2.savefig(fig2_path)
            figures.append(fig2_path)
        
        # Generate report based on format
        try:
            if format == "html":
                # HTML Report
                html_content = f"""
                <html>
                <head>
                    <title>Compatibility Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333366; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .visualization {{ margin: 20px 0; text-align: center; }}
                        .visualization img {{ max-width: 100%; height: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Compatibility Report</h1>
                    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Compatibility Summary</h2>
                    {summary_df.to_html(index=False)}
                    
                    <h2>Hardware Type Compatibility</h2>
                    {hw_compat.to_html(index=False)}
                    
                    <h2>Compatibility Matrix</h2>
                    {matrix_df.to_html()}
                    
                    <h2>Compatibility Issues</h2>
                    {problems.to_html(index=False)}
                    
                    <h2>Visualizations</h2>
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    html_content += f"""
                    <div class="visualization">
                        <h3>Visualization {i+1}</h3>
                        <img src="{fig_name}" alt="Visualization {i+1}">
                    </div>
                    """
                
                html_content += """
                </body>
                </html>
                """
                
                with open(output_file, 'w') as f:
                    f.write(html_content)
                
            elif format == "md":
                # Markdown Report
                md_content = f"""
                # Compatibility Report
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Compatibility Summary
                
                {summary_df.to_markdown(index=False)}
                
                ## Hardware Type Compatibility
                
                {hw_compat.to_markdown(index=False)}
                
                ## Compatibility Matrix
                
                {matrix_df.to_markdown()}
                
                ## Compatibility Issues
                
                {problems.to_markdown(index=False)}
                
                ## Visualizations
                
                """
                
                # Add visualization images
                for i, fig_path in enumerate(figures):
                    fig_name = os.path.basename(fig_path)
                    md_content += f"""
                ### Visualization {i+1}
                
                ![Visualization {i+1}]({fig_name})
                
                """
                
                with open(output_file, 'w') as f:
                    f.write(md_content)
                
            elif format == "txt":
                # Text Report
                txt_content = f"""
                Compatibility Report
                ===================
                
                Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Compatibility Summary
                --------------------
                
                {summary_df.to_string(index=False)}
                
                Hardware Type Compatibility
                --------------------------
                
                {hw_compat.to_string(index=False)}
                
                Compatibility Matrix
                -------------------
                
                {matrix_df.to_string()}
                
                Compatibility Issues
                -------------------
                
                {problems.to_string(index=False)}
                
                Visualizations saved to:
                {os.path.dirname(output_file)}
                """
                
                with open(output_file, 'w') as f:
                    f.write(txt_content)
                
            else:
                logger.error(f"Unsupported report format: {format}")
                return False
            
            logger.info(f"Generated compatibility report: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating compatibility report: {e}")
            return False
    
    def plot_hardware_comparison(self, model: str, metric: str = "throughput", output_file: str = None) -> str:
        """
        Plot a hardware comparison chart for a specific model.
        
        Args:
            model: Model name
            metric: Metric to compare (throughput, latency_avg, memory_peak)
            output_file: Output file path (or None to use default)
            
        Returns:
            Path to the generated plot
        """
        # Get comparison data
        df = self.compare_hardware(model, metric)
        
        if df.empty:
            logger.error(f"No data found for model: {model}")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean value as bar
        sns.barplot(x='hardware', y=f'avg_{metric}', data=df, ax=ax)
        
        # Add error bars if available
        if f'stddev_{metric}' in df.columns:
            # Add error bars
            x_pos = range(len(df))
            ax.errorbar(x=x_pos, y=df[f'avg_{metric}'], 
                       yerr=df[f'stddev_{metric}'], 
                       fmt='none', ecolor='black', capsize=5)
        
        # Set labels and title
        ax.set_title(f'{metric.capitalize()} Comparison by Hardware for {model}')
        ax.set_xlabel('Hardware Type')
        ax.set_ylabel(f'{metric.capitalize()}')
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            output_file = f"{model}_{metric}_hardware_comparison.png"
        
        fig.savefig(output_file)
        plt.close(fig)
        
        logger.info(f"Generated hardware comparison plot: {output_file}")
        return output_file
    
    def plot_model_comparison(self, hardware: str, metric: str = "throughput", output_file: str = None) -> str:
        """
        Plot a model comparison chart for a specific hardware.
        
        Args:
            hardware: Hardware type
            metric: Metric to compare (throughput, latency_avg, memory_peak)
            output_file: Output file path (or None to use default)
            
        Returns:
            Path to the generated plot
        """
        # Get comparison data
        df = self.compare_models(hardware, metric)
        
        if df.empty:
            logger.error(f"No data found for hardware: {hardware}")
            return None
        
        # Limit to top 10 models for readability
        if len(df) > 10:
            if metric == "throughput":
                df = df.sort_values(f'avg_{metric}', ascending=False).head(10)
            else:
                df = df.sort_values(f'avg_{metric}', ascending=True).head(10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean value as bar
        sns.barplot(x='model', y=f'avg_{metric}', data=df, ax=ax)
        
        # Add error bars if available
        if f'stddev_{metric}' in df.columns:
            # Add error bars
            x_pos = range(len(df))
            ax.errorbar(x=x_pos, y=df[f'avg_{metric}'], 
                       yerr=df[f'stddev_{metric}'], 
                       fmt='none', ecolor='black', capsize=5)
        
        # Set labels and title
        ax.set_title(f'{metric.capitalize()} Comparison by Model for {hardware}')
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric.capitalize()}')
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            output_file = f"{hardware}_{metric}_model_comparison.png"
        
        fig.savefig(output_file)
        plt.close(fig)
        
        logger.info(f"Generated model comparison plot: {output_file}")
        return output_file
    
    def plot_performance_trend(self, model: str, hardware: str, metric: str = "throughput", output_file: str = None) -> str:
        """
        Plot a performance trend chart for a specific model and hardware.
        
        Args:
            model: Model name
            hardware: Hardware type
            metric: Metric to plot (throughput, latency_avg, memory_peak)
            output_file: Output file path (or None to use default)
            
        Returns:
            Path to the generated plot
        """
        # Get trend data
        df = self.get_performance_trends(model, hardware, metric)
        
        if df.empty:
            logger.error(f"No data found for model: {model} on hardware: {hardware}")
            return None
        
        # Convert timestamp to datetime if it's string
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot trend
        sns.lineplot(x='timestamp', y=metric, data=df, ax=ax, marker='o')
        
        # Set labels and title
        ax.set_title(f'{metric.capitalize()} Trend for {model} on {hardware}')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{metric.capitalize()}')
        
        # Format x-axis as dates
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            output_file = f"{model}_{hardware}_{metric}_trend.png"
        
        fig.savefig(output_file)
        plt.close(fig)
        
        logger.info(f"Generated performance trend plot: {output_file}")
        return output_file


def generate_timing_report(conn, args):
    """Generate a comprehensive timing report for all models and hardware platforms."""
    logger.info("Generating comprehensive benchmark timing report...")
    
    # Create report generator with the same database connection
    try:
        report_gen = BenchmarkTimingReport(db_path=args.db_path)
        
        # Generate the report
        output_path = args.output or f"benchmark_timing_report.{args.format}"
        report_path = report_gen.generate_timing_report(
            output_format=args.format,
            output_path=output_path,
            days_lookback=args.days or 30
        )
        
        if report_path:
            logger.info(f"Timing report generated: {report_path}")
            return {"status": "success", "output": report_path}
        else:
            logger.error("Failed to generate timing report")
            return {"status": "error", "message": "Failed to generate timing report"}
    except Exception as e:
        logger.error(f"Error generating timing report: {str(e)}")
        return {"status": "error", "message": str(e)}

def main():
    """Command-line interface for the benchmark database query tool."""
    parser = argparse.ArgumentParser(description="Benchmark Database Query Tool")
    parser.add_argument("--db", default="./benchmark_db.duckdb",
                        help="Path to the DuckDB database")
    parser.add_argument("--sql", 
                        help="Execute a SQL query on the database")
    parser.add_argument("--report", choices=["performance", "hardware", "compatibility"],
                        help="Generate a report")
    parser.add_argument("--format", choices=["html", "md", "txt", "csv", "json", "xlsx"], default="html",
                        help="Output format for reports")
    parser.add_argument("--output", 
                        help="Output file path")
    parser.add_argument("--model", 
                        help="Model name for model-specific queries")
    parser.add_argument("--hardware", 
                        help="Hardware type for hardware-specific queries")
    parser.add_argument("--metric", choices=["throughput", "latency_avg", "memory_peak"], default="throughput",
                        help="Metric to use for comparisons")
    parser.add_argument("--compare-hardware", action="store_true",
                        help="Compare hardware performance for a specific model")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare model performance on a specific hardware")
    parser.add_argument("--plot-trend", action="store_true",
                        help="Plot performance trend for a model on a hardware")
    parser.add_argument("--export", choices=["performance", "hardware", "compatibility"],
                        help="Export data from a specific category")
    parser.add_argument("--matrix", action="store_true",
                        help="Generate a compatibility matrix")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()

# Create query tool
try:
    query_tool = BenchmarkDBQuery(db_path=args.db, debug=args.debug)
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
            if args.format == "csv":
                result.to_csv(args.output, index=False)
            elif args.format == "json":
                result.to_json(args.output, orient="records", indent=2)
            elif args.format == "html":
                result.to_html(args.output, index=False)
            elif args.format == "xlsx":
                result.to_excel(args.output, index=False)
                else:
                    # Default to text
                    with open(args.output, 'w') as f:
                        f.write(result.to_string(index=False))
                logger.info(f"Query results saved to: {args.output}")
            else:
                # Print to console
                print(result.to_string(index=False))
                
    elif args.report:
        # Generate report
        output_file = args.output or f"{args.report}_report.{args.format}"
        
        if args.report == "performance":
            success = query_tool.generate_performance_report(output_file, args.format)
        elif args.report == "hardware":
            success = query_tool.generate_hardware_report(output_file, args.format)
        elif args.report == "compatibility":
            success = query_tool.generate_compatibility_report(output_file, args.format)
        
        if success:
            logger.info(f"Report generated: {output_file}")
        else:
            logger.error("Failed to generate report")
            
    elif args.export:
        # Export data
        if not args.output:
            logger.error("Output file path is required for export")
            return
            
        success = query_tool.export_data(args.export, args.output, args.format)
        
        if success:
            logger.info(f"Data exported to: {args.output}")
        else:
            logger.error("Failed to export data")
            
    elif args.compare_hardware:
        # Compare hardware performance
        if not args.model:
            logger.error("Model name is required for hardware comparison")
            return
            
        if args.output:
            # Generate plot
            plot_path = query_tool.plot_hardware_comparison(args.model, args.metric, args.output)
            
            if plot_path:
                logger.info(f"Hardware comparison plot saved to: {plot_path}")
            else:
                logger.error("Failed to generate hardware comparison plot")
        else:
            # Display comparison data
            df = query_tool.compare_hardware(args.model, args.metric)
            
            if df.empty:
                logger.error(f"No data found for model: {args.model}")
            else:
                print(df.to_string(index=False))
                
    elif args.compare_models:
        # Compare model performance
        if not args.hardware:
            logger.error("Hardware type is required for model comparison")
            return
            
        if args.output:
            # Generate plot
            plot_path = query_tool.plot_model_comparison(args.hardware, args.metric, args.output)
            
            if plot_path:
                logger.info(f"Model comparison plot saved to: {plot_path}")
            else:
                logger.error("Failed to generate model comparison plot")
        else:
            # Display comparison data
            df = query_tool.compare_models(args.hardware, args.metric)
            
            if df.empty:
                logger.error(f"No data found for hardware: {args.hardware}")
            else:
                print(df.to_string(index=False))
                
    elif args.plot_trend:
        # Plot performance trend
        if not args.model or not args.hardware:
            logger.error("Both model and hardware are required for trend plot")
            return
            
        plot_path = query_tool.plot_performance_trend(args.model, args.hardware, args.metric, args.output)
        
        if plot_path:
            logger.info(f"Performance trend plot saved to: {plot_path}")
        else:
            logger.error("Failed to generate performance trend plot")
            
    elif args.matrix:
        # Generate compatibility matrix
        matrix = query_tool.get_compatibility_matrix()
        
        if matrix.empty:
            logger.error("No compatibility data found")
        else:
            if args.output:
                if args.format == "csv":
                    matrix.to_csv(args.output)
                elif args.format == "json":
                    matrix.to_json(args.output, orient="index", indent=2)
                elif args.format == "html":
                    matrix.to_html(args.output)
                elif args.format == "xlsx":
                    matrix.to_excel(args.output)
                else:
                    # Default to text
                    with open(args.output, 'w') as f:
                        f.write(matrix.to_string())
                logger.info(f"Compatibility matrix saved to: {args.output}")
            else:
                # Print to console
                print(matrix.to_string())
                
    else:
        # No specific action requested, print help
        parser.print_help()

if __name__ == "__main__":
    main()