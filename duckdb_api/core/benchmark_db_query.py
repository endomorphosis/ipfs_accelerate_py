#!/usr/bin/env python
"""
Query and visualization tool for the benchmark database.

This module provides a command-line interface for querying the benchmark database
and generating reports and visualizations from the results.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

try:
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas matplotlib seaborn")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

class BenchmarkDBQuery:
    """
    Query and visualization tool for the benchmark database.
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
        
        # Create API instance
        self.api = BenchmarkDBAPI(db_path=db_path, debug=debug)
        
        logger.info(f"Initialized BenchmarkDBQuery for database: {db_path}")
    
    def execute_query(self, sql: str, parameters: Dict = None) -> pd.DataFrame:
        """
        Execute a SQL query against the database.
        
        Args:
            sql: SQL query string
            parameters: Parameters for the query
            
        Returns:
            DataFrame with query results
        """
        return self.api.query(sql, parameters)
    
    def generate_performance_report(self, model_name: Optional[str] = None,
                                   hardware_type: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a performance report.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            
        Returns:
            DataFrame with performance report data
        """
        return self.api.get_performance_metrics(model_name, hardware_type)
    
    def generate_hardware_report(self) -> pd.DataFrame:
        """
        Generate a hardware report.
        
        Returns:
            DataFrame with hardware report data
        """
        return self.api.get_hardware_list()
    
    def generate_compatibility_report(self, model_name: Optional[str] = None,
                                     hardware_type: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a compatibility report.
        
        Args:
            model_name: Filter by model name (optional)
            hardware_type: Filter by hardware type (optional)
            
        Returns:
            DataFrame with compatibility report data
        """
        return self.api.get_model_hardware_compatibility(model_name, hardware_type)
    
    def generate_model_comparison(self, hardware_type: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Generate a model comparison on a specific hardware platform.
        
        Args:
            hardware_type: Hardware type to compare models on
            metric: Metric to compare ("throughput", "latency", "memory")
            
        Returns:
            DataFrame with model comparison data
        """
        sql = f"""
        WITH latest_results AS (
            SELECT 
                m.model_name,
                m.model_family,
                hp.hardware_type,
                pr.batch_size,
                pr.precision,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, pr.batch_size, pr.precision
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                hp.hardware_type = :hardware_type
        )
        SELECT
            model_name,
            model_family,
            hardware_type,
            batch_size,
            precision,
            average_latency_ms,
            throughput_items_per_second,
            memory_peak_mb
        FROM
            latest_results
        WHERE
            rn = 1
        """
        
        df = self.execute_query(sql, {"hardware_type": hardware_type})
        
        # Sort by the specified metric
        if metric.lower() == "throughput":
            df = df.sort_values("throughput_items_per_second", ascending=False)
        elif metric.lower() == "latency":
            df = df.sort_values("average_latency_ms", ascending=True)
        elif metric.lower() == "memory":
            df = df.sort_values("memory_peak_mb", ascending=True)
        
        return df
    
    def generate_hardware_comparison(self, model_name: str, metric: str = "throughput") -> pd.DataFrame:
        """
        Generate a hardware comparison for a specific model.
        
        Args:
            model_name: Model name to compare hardware platforms
            metric: Metric to compare ("throughput", "latency", "memory")
            
        Returns:
            DataFrame with hardware comparison data
        """
        return self.api.get_performance_comparison(model_name, metric)
    
    def plot_hardware_comparison(self, model_name: str, metric: str = "throughput", 
                               output_file: Optional[str] = None, 
                               figsize: tuple = (10, 6)) -> None:
        """
        Plot a hardware comparison for a specific model.
        
        Args:
            model_name: Model name to compare hardware platforms
            metric: Metric to compare ("throughput", "latency", "memory")
            output_file: Path to save the plot, or None to display
            figsize: Figure size (width, height) in inches
        """
        df = self.generate_hardware_comparison(model_name, metric)
        
        if df.empty:
            logger.warning(f"No data found for model: {model_name}")
            return
        
        # Configure plot
        plt.figure(figsize=figsize)
        
        # Get metric column and title
        if metric.lower() == "throughput":
            metric_col = "metric_value"
            title = f"Throughput Comparison for {model_name}"
            ylabel = "Throughput (items per second)"
        elif metric.lower() == "latency":
            metric_col = "metric_value"
            title = f"Latency Comparison for {model_name}"
            ylabel = "Latency (ms)"
        else:
            metric_col = "metric_value"
            title = f"{metric.capitalize()} Comparison for {model_name}"
            ylabel = metric.capitalize()
        
        # Create plot
        sns.barplot(x="hardware_type", y=metric_col, hue="batch_size", data=df)
        plt.title(title)
        plt.xlabel("Hardware Platform")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_file}")
        else:
            plt.show()
    
    def generate_batch_size_comparison(self, model_name: str, hardware_type: str, 
                                      metric: str = "throughput") -> pd.DataFrame:
        """
        Generate a batch size comparison for a specific model and hardware.
        
        Args:
            model_name: Model name
            hardware_type: Hardware type
            metric: Metric to compare ("throughput", "latency", "memory")
            
        Returns:
            DataFrame with batch size comparison data
        """
        sql = f"""
        WITH latest_results AS (
            SELECT 
                m.model_name,
                hp.hardware_type,
                pr.batch_size,
                pr.precision,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                ROW_NUMBER() OVER(PARTITION BY pr.batch_size, pr.precision
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_name = :model_name AND
                hp.hardware_type = :hardware_type
        )
        SELECT
            model_name,
            hardware_type,
            batch_size,
            precision,
            average_latency_ms,
            throughput_items_per_second,
            memory_peak_mb
        FROM
            latest_results
        WHERE
            rn = 1
        ORDER BY
            batch_size
        """
        
        return self.execute_query(sql, {
            "model_name": model_name,
            "hardware_type": hardware_type
        })
    
    def plot_batch_size_comparison(self, model_name: str, hardware_type: str, 
                                 metric: str = "throughput", 
                                 output_file: Optional[str] = None, 
                                 figsize: tuple = (10, 6)) -> None:
        """
        Plot a batch size comparison for a specific model and hardware.
        
        Args:
            model_name: Model name
            hardware_type: Hardware type
            metric: Metric to compare ("throughput", "latency", "memory")
            output_file: Path to save the plot, or None to display
            figsize: Figure size (width, height) in inches
        """
        df = self.generate_batch_size_comparison(model_name, hardware_type, metric)
        
        if df.empty:
            logger.warning(f"No data found for model: {model_name} on hardware: {hardware_type}")
            return
        
        # Configure plot
        plt.figure(figsize=figsize)
        
        # Get metric column and title
        if metric.lower() == "throughput":
            metric_col = "throughput_items_per_second"
            title = f"Throughput vs Batch Size for {model_name} on {hardware_type}"
            ylabel = "Throughput (items per second)"
        elif metric.lower() == "latency":
            metric_col = "average_latency_ms"
            title = f"Latency vs Batch Size for {model_name} on {hardware_type}"
            ylabel = "Latency (ms)"
        elif metric.lower() == "memory":
            metric_col = "memory_peak_mb"
            title = f"Memory Usage vs Batch Size for {model_name} on {hardware_type}"
            ylabel = "Memory Usage (MB)"
        else:
            logger.error(f"Unknown metric: {metric}")
            return
        
        # Create plot
        sns.lineplot(x="batch_size", y=metric_col, hue="precision", marker='o', data=df)
        plt.title(title)
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save or display plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {output_file}")
        else:
            plt.show()
    
    def generate_model_family_comparison(self, model_family: str, 
                                        hardware_type: Optional[str] = None,
                                        metric: str = "throughput") -> pd.DataFrame:
        """
        Generate a comparison of models within a family.
        
        Args:
            model_family: Model family name
            hardware_type: Hardware type (optional)
            metric: Metric to compare ("throughput", "latency", "memory")
            
        Returns:
            DataFrame with model family comparison data
        """
        sql = """
        WITH latest_results AS (
            SELECT 
                m.model_name,
                m.model_family,
                hp.hardware_type,
                pr.batch_size,
                pr.precision,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size, pr.precision
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_family = :model_family
        """
        
        params = {"model_family": model_family}
        
        if hardware_type:
            sql += " AND hp.hardware_type = :hardware_type"
            params["hardware_type"] = hardware_type
        
        sql += """
        )
        SELECT
            model_name,
            model_family,
            hardware_type,
            batch_size,
            precision,
            average_latency_ms,
            throughput_items_per_second,
            memory_peak_mb
        FROM
            latest_results
        WHERE
            rn = 1
        """
        
        return self.execute_query(sql, params)
    
    def export_to_format(self, df: pd.DataFrame, format: str, output: Optional[str] = None) -> Any:
        """
        Export a DataFrame to the specified format.
        
        Args:
            df: DataFrame to export
            format: Output format ('csv', 'json', 'markdown', 'html', 'chart')
            output: Output file path, or None for stdout
            
        Returns:
            Exported data as string or object, depending on format
        """
        if df.empty:
            logger.warning("DataFrame is empty, nothing to export")
            return None
        
        if format.lower() == 'csv':
            if output:
                df.to_csv(output, index=False)
                logger.info(f"Exported CSV to: {output}")
                return None
            return df.to_csv(index=False)
            
        elif format.lower() == 'json':
            if output:
                df.to_json(output, orient='records', indent=2)
                logger.info(f"Exported JSON to: {output}")
                return None
            return df.to_json(orient='records', indent=2)
            
        elif format.lower() == 'markdown':
            md = df.to_markdown(index=False)
            if output:
                with open(output, 'w') as f:
                    f.write(md)
                logger.info(f"Exported Markdown to: {output}")
                return None
            return md
            
        elif format.lower() == 'html':
            html = df.to_html(index=False)
            if output:
                with open(output, 'w') as f:
                    f.write(html)
                logger.info(f"Exported HTML to: {output}")
                return None
            return html
            
        elif format.lower() == 'chart':
            if not output:
                logger.warning("Chart format requires an output file")
                return None
            
            # Simple chart visualization
            plt.figure(figsize=(10, 6))
            
            # Determine if we should use bar or line chart based on data
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0 and len(df) <= 20:
                # Use bar chart for categorical comparison
                sns.barplot(data=df, x=df.columns[0], y=numeric_cols[0])
                plt.xticks(rotation=45)
            elif len(numeric_cols) > 0:
                # Use line chart for time series or continuous data
                sns.lineplot(data=df, x=df.columns[0], y=numeric_cols[0])
            
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            logger.info(f"Exported chart to: {output}")
            return None
            
        else:
            logger.error(f"Unsupported export format: {format}")
            return None

def main():
    """Command-line interface for the benchmark database query tool."""
    parser = argparse.ArgumentParser(description="Benchmark Database Query Tool")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--sql",
                       help="SQL query to execute")
    parser.add_argument("--report", choices=['performance', 'hardware', 'compatibility'],
                       help="Generate a predefined report")
    parser.add_argument("--model",
                       help="Filter by model name")
    parser.add_argument("--hardware",
                       help="Filter by hardware type")
    parser.add_argument("--metric", choices=['throughput', 'latency', 'memory'], default='throughput',
                       help="Metric to use for comparison")
    parser.add_argument("--model-family",
                       help="Filter by model family")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare models on a specific hardware platform")
    parser.add_argument("--compare-hardware", action="store_true",
                       help="Compare hardware platforms for a specific model")
    parser.add_argument("--batch-comparison", action="store_true",
                       help="Compare batch sizes for a specific model and hardware")
    parser.add_argument("--family-comparison", action="store_true",
                       help="Compare models within a family")
    parser.add_argument("--compatibility-matrix", action="store_true",
                       help="Generate a compatibility matrix")
    parser.add_argument("--format", choices=['csv', 'json', 'markdown', 'html', 'chart'], default='markdown',
                       help="Output format")
    parser.add_argument("--output",
                       help="Output file path")
    parser.add_argument("--plot", action="store_true",
                       help="Generate a plot")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create query tool
    query = BenchmarkDBQuery(db_path=args.db_path, debug=args.debug)
    
    # Execute query or generate report
    if args.sql:
        df = query.execute_query(args.sql)
    elif args.report == 'performance':
        df = query.generate_performance_report(args.model, args.hardware)
    elif args.report == 'hardware':
        df = query.generate_hardware_report()
    elif args.report == 'compatibility':
        df = query.generate_compatibility_report(args.model, args.hardware)
    elif args.compare_models and args.hardware:
        df = query.generate_model_comparison(args.hardware, args.metric)
    elif args.compare_hardware and args.model:
        df = query.generate_hardware_comparison(args.model, args.metric)
        if args.plot:
            query.plot_hardware_comparison(args.model, args.metric, args.output)
            return
    elif args.batch_comparison and args.model and args.hardware:
        df = query.generate_batch_size_comparison(args.model, args.hardware, args.metric)
        if args.plot:
            query.plot_batch_size_comparison(args.model, args.hardware, args.metric, args.output)
            return
    elif args.family_comparison and args.model_family:
        df = query.generate_model_family_comparison(args.model_family, args.hardware, args.metric)
    elif args.compatibility_matrix:
        df = query.generate_compatibility_report()
    else:
        parser.print_help()
        return
    
    # Export results
    result = query.export_to_format(df, args.format, args.output)
    
    # Print to console if no output file specified
    if result and not args.output:
        print(result)

if __name__ == "__main__":
    main()