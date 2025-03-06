#!/usr/bin/env python
"""
Benchmark Regression Analyzer

This script analyzes benchmark results stored in the DuckDB database to detect
performance regressions between runs. It generates reports with detailed
analysis of performance changes.

Usage:
    python benchmark_regression_analyzer.py --compare-runs 25 26 --models bert vit
    python benchmark_regression_analyzer.py --latest 2 --threshold 10 --output regression_report.md
"""

import os
import sys
import json
import logging
import argparse
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add DuckDB database support
try:
    from benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Try to import required packages
try:
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas numpy matplotlib")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("benchmark_regression")

class BenchmarkRegressionAnalyzer:
    """
    Analyzes benchmark results to detect performance regressions between runs.
    """
    
    def __init__(self, 
                 db_path: str = "./benchmark_db.duckdb",
                 output_dir: str = "./regression_reports",
                 threshold: float = 5.0,
                 debug: bool = False):
        """
        Initialize the benchmark regression analyzer.
        
        Args:
            db_path: Path to the DuckDB database
            output_dir: Directory for output reports
            threshold: Threshold percentage for regression detection
            debug: Enable debug logging
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        
        # Set debug logging if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check database exists
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        logger.info(f"Initialized regression analyzer with database: {db_path}")
        logger.info(f"Regression threshold: {threshold}%")
    
    def get_test_runs(self, limit: int = 10) -> pd.DataFrame:
        """
        Get test runs from the database.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            DataFrame with test runs
        """
        conn = duckdb.connect(self.db_path)
        try:
            query = """
            SELECT 
                run_id,
                test_name,
                test_type,
                started_at,
                completed_at,
                execution_time_seconds,
                success,
                metadata
            FROM 
                test_runs
            WHERE
                test_type = 'performance'
            ORDER BY 
                started_at DESC
            LIMIT ?
            """
            
            df = conn.execute(query, [limit]).fetch_df()
            return df
        finally:
            conn.close()
    
    def get_run_details(self, run_id: int) -> Dict[str, Any]:
        """
        Get details of a specific test run.
        
        Args:
            run_id: ID of the test run
            
        Returns:
            Dictionary with run details
        """
        conn = duckdb.connect(self.db_path)
        try:
            query = """
            SELECT 
                run_id,
                test_name,
                test_type,
                started_at,
                completed_at,
                execution_time_seconds,
                success,
                metadata
            FROM 
                test_runs
            WHERE
                run_id = ?
            """
            
            result = conn.execute(query, [run_id]).fetchone()
            
            if not result:
                raise ValueError(f"Test run not found: {run_id}")
            
            # Convert to dictionary
            run_details = {
                "run_id": result[0],
                "test_name": result[1],
                "test_type": result[2],
                "started_at": result[3],
                "completed_at": result[4],
                "execution_time_seconds": result[5],
                "success": result[6],
                "metadata": json.loads(result[7]) if result[7] else {}
            }
            
            return run_details
        finally:
            conn.close()
    
    def get_performance_data(self, run_id: int, models: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get performance data for a specific test run.
        
        Args:
            run_id: ID of the test run
            models: Optional list of model names to filter by
            
        Returns:
            DataFrame with performance data
        """
        conn = duckdb.connect(self.db_path)
        try:
            # Build query
            if models:
                placeholders = ", ".join(["?"] * len(models))
                query = f"""
                SELECT 
                    pr.result_id,
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    hp.device_name,
                    pr.test_case,
                    pr.batch_size,
                    pr.precision,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE
                    pr.run_id = ?
                    AND m.model_name IN ({placeholders})
                ORDER BY
                    m.model_name, hp.hardware_type, pr.batch_size
                """
                params = [run_id] + models
            else:
                query = """
                SELECT 
                    pr.result_id,
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    hp.device_name,
                    pr.test_case,
                    pr.batch_size,
                    pr.precision,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE
                    pr.run_id = ?
                ORDER BY
                    m.model_name, hp.hardware_type, pr.batch_size
                """
                params = [run_id]
            
            df = conn.execute(query, params).fetch_df()
            return df
        finally:
            conn.close()
    
    def compare_runs(self, 
                     base_run_id: int, 
                     compare_run_id: int,
                     models: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance data between two test runs.
        
        Args:
            base_run_id: ID of the base test run
            compare_run_id: ID of the test run to compare against
            models: Optional list of model names to filter by
            
        Returns:
            DataFrame with comparison results
        """
        # Get performance data for both runs
        base_data = self.get_performance_data(base_run_id, models)
        compare_data = self.get_performance_data(compare_run_id, models)
        
        # Check if we have data
        if base_data.empty or compare_data.empty:
            logger.warning("No performance data found for comparison")
            return pd.DataFrame()
        
        # Create unique identifiers for matching rows
        for df in [base_data, compare_data]:
            df['match_key'] = df.apply(
                lambda row: f"{row['model_name']}|{row['hardware_type']}|{row['batch_size']}|{row['precision']}|{row['test_case']}", 
                axis=1
            )
        
        # Merge data
        merged = pd.merge(
            base_data, compare_data,
            on='match_key',
            how='outer', 
            suffixes=('_base', '_compare')
        )
        
        # Calculate changes
        if not merged.empty:
            # Latency (lower is better)
            merged['latency_change'] = merged.apply(
                lambda row: (
                    (row['average_latency_ms_compare'] - row['average_latency_ms_base']) / 
                    row['average_latency_ms_base'] * 100
                ) if not pd.isnull(row.get('average_latency_ms_base')) and 
                   not pd.isnull(row.get('average_latency_ms_compare')) else np.nan,
                axis=1
            )
            
            # Throughput (higher is better)
            merged['throughput_change'] = merged.apply(
                lambda row: (
                    (row['throughput_items_per_second_compare'] - row['throughput_items_per_second_base']) / 
                    row['throughput_items_per_second_base'] * 100
                ) if not pd.isnull(row.get('throughput_items_per_second_base')) and 
                   not pd.isnull(row.get('throughput_items_per_second_compare')) else np.nan,
                axis=1
            )
            
            # Memory (lower is better)
            merged['memory_change'] = merged.apply(
                lambda row: (
                    (row['memory_peak_mb_compare'] - row['memory_peak_mb_base']) / 
                    row['memory_peak_mb_base'] * 100
                ) if not pd.isnull(row.get('memory_peak_mb_base')) and 
                   not pd.isnull(row.get('memory_peak_mb_compare')) else np.nan,
                axis=1
            )
            
            # Flag regressions
            merged['latency_regression'] = merged['latency_change'] > self.threshold
            merged['throughput_regression'] = merged['throughput_change'] < -self.threshold
            merged['memory_regression'] = merged['memory_change'] > self.threshold
            
            # Overall regression flag
            merged['has_regression'] = merged['latency_regression'] | merged['throughput_regression'] | merged['memory_regression']
        
        return merged
    
    def get_latest_runs(self, count: int = 2) -> List[int]:
        """
        Get the latest test runs.
        
        Args:
            count: Number of latest runs to retrieve
            
        Returns:
            List of run IDs
        """
        conn = duckdb.connect(self.db_path)
        try:
            query = """
            SELECT 
                run_id
            FROM 
                test_runs
            WHERE
                test_type = 'performance'
                AND success = TRUE
            ORDER BY 
                started_at DESC
            LIMIT ?
            """
            
            result = conn.execute(query, [count]).fetchall()
            return [r[0] for r in result]
        finally:
            conn.close()
    
    def plot_comparison(self, comparison_df: pd.DataFrame, metric: str, output_file: str) -> None:
        """
        Generate a plot comparing a specific metric between runs.
        
        Args:
            comparison_df: DataFrame with comparison data
            metric: Metric to plot ('latency', 'throughput', or 'memory')
            output_file: Path to save the plot
        """
        if comparison_df.empty:
            logger.warning("No data available for plotting")
            return
        
        # Map metric to column names
        metric_map = {
            'latency': {
                'base_col': 'average_latency_ms_base',
                'compare_col': 'average_latency_ms_compare',
                'change_col': 'latency_change',
                'title': 'Latency Comparison (ms)',
                'label': 'Latency (ms)',
                'lower_better': True
            },
            'throughput': {
                'base_col': 'throughput_items_per_second_base',
                'compare_col': 'throughput_items_per_second_compare',
                'change_col': 'throughput_change',
                'title': 'Throughput Comparison (items/s)',
                'label': 'Throughput (items/s)',
                'lower_better': False
            },
            'memory': {
                'base_col': 'memory_peak_mb_base',
                'compare_col': 'memory_peak_mb_compare',
                'change_col': 'memory_change',
                'title': 'Memory Usage Comparison (MB)',
                'label': 'Memory (MB)',
                'lower_better': True
            }
        }
        
        if metric not in metric_map:
            logger.error(f"Invalid metric: {metric}")
            return
        
        metric_info = metric_map[metric]
        
        # Filter out rows with missing data
        filtered_df = comparison_df.dropna(subset=[metric_info['base_col'], metric_info['compare_col']])
        
        if filtered_df.empty:
            logger.warning(f"No data available for metric: {metric}")
            return
        
        # Create labels for x-axis
        filtered_df['label'] = filtered_df.apply(
            lambda row: f"{row['model_name_base']}\n{row['hardware_type_base']}\nBatch {row['batch_size_base']}",
            axis=1
        )
        
        # Sort by the base metric value
        if metric_info['lower_better']:
            filtered_df = filtered_df.sort_values(by=metric_info['base_col'])
        else:
            filtered_df = filtered_df.sort_values(by=metric_info['base_col'], ascending=False)
        
        # Take at most 20 entries to keep the plot readable
        if len(filtered_df) > 20:
            # Get 10 best and 10 worst cases by change percentage
            sorted_by_change = filtered_df.sort_values(by=metric_info['change_col'], 
                                                     ascending=metric_info['lower_better'])
            plot_df = pd.concat([sorted_by_change.head(10), sorted_by_change.tail(10)])
        else:
            plot_df = filtered_df
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Bar plot with base and compare values
        x = range(len(plot_df))
        bar_width = 0.35
        
        plt.bar([i - bar_width/2 for i in x], plot_df[metric_info['base_col']], 
                width=bar_width, label='Base Run', color='blue', alpha=0.7)
        plt.bar([i + bar_width/2 for i in x], plot_df[metric_info['compare_col']], 
                width=bar_width, label='Compare Run', color='orange', alpha=0.7)
        
        # Add percentage change as text
        for i, row in enumerate(plot_df.itertuples()):
            change = getattr(row, metric_info['change_col'].replace('.', '_'))
            color = 'red' if (change > 0 and metric_info['lower_better']) or \
                             (change < 0 and not metric_info['lower_better']) else 'green'
            plt.annotate(f"{change:.1f}%", 
                        (i, max(getattr(row, metric_info['base_col'].replace('.', '_')), 
                               getattr(row, metric_info['compare_col'].replace('.', '_')))),
                        fontsize=8, color=color, ha='center', va='bottom')
        
        # Customize plot
        plt.title(metric_info['title'])
        plt.xlabel('Model / Hardware / Batch Size')
        plt.ylabel(metric_info['label'])
        plt.xticks(x, plot_df['label'], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Plot saved to: {output_file}")
    
    def generate_report(self, 
                       comparison_df: pd.DataFrame, 
                       base_run_id: int, 
                       compare_run_id: int,
                       output_file: Optional[str] = None) -> str:
        """
        Generate a markdown report comparing two runs.
        
        Args:
            comparison_df: DataFrame with comparison data
            base_run_id: ID of the base test run
            compare_run_id: ID of the test run to compare against
            output_file: Path to save the report (optional)
            
        Returns:
            Path to the generated report
        """
        if comparison_df.empty:
            logger.warning("No data available for report generation")
            if output_file:
                return output_file
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                return str(self.output_dir / f"empty_report_{timestamp}.md")
        
        # Get run details
        base_run = self.get_run_details(base_run_id)
        compare_run = self.get_run_details(compare_run_id)
        
        # Generate plots for the report
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        latency_plot = plots_dir / f"latency_comparison_{timestamp}.png"
        throughput_plot = plots_dir / f"throughput_comparison_{timestamp}.png"
        memory_plot = plots_dir / f"memory_comparison_{timestamp}.png"
        
        self.plot_comparison(comparison_df, 'latency', latency_plot)
        self.plot_comparison(comparison_df, 'throughput', throughput_plot)
        self.plot_comparison(comparison_df, 'memory', memory_plot)
        
        # Create output file path if not provided
        if not output_file:
            output_file = self.output_dir / f"regression_report_{base_run_id}_vs_{compare_run_id}_{timestamp}.md"
        
        # Count regressions
        regression_count = comparison_df['has_regression'].sum() if 'has_regression' in comparison_df.columns else 0
        total_comparisons = len(comparison_df)
        
        # Prepare markdown content
        with open(output_file, 'w') as f:
            f.write("# Benchmark Regression Analysis Report\n\n")
            
            # Run information
            f.write("## Run Information\n\n")
            f.write("### Base Run\n\n")
            f.write(f"- Run ID: {base_run['run_id']}\n")
            f.write(f"- Test Name: {base_run['test_name']}\n")
            f.write(f"- Started: {base_run['started_at']}\n")
            f.write(f"- Duration: {base_run['execution_time_seconds']:.2f} seconds\n")
            
            # Extract metadata
            if base_run['metadata']:
                models_tested = base_run['metadata'].get('models', [])
                hardware_tested = base_run['metadata'].get('hardware', [])
                
                if models_tested:
                    f.write(f"- Models: {', '.join(models_tested)}\n")
                
                if hardware_tested:
                    f.write(f"- Hardware: {', '.join(hardware_tested)}\n")
            
            f.write("\n### Compare Run\n\n")
            f.write(f"- Run ID: {compare_run['run_id']}\n")
            f.write(f"- Test Name: {compare_run['test_name']}\n")
            f.write(f"- Started: {compare_run['started_at']}\n")
            f.write(f"- Duration: {compare_run['execution_time_seconds']:.2f} seconds\n")
            
            # Extract metadata
            if compare_run['metadata']:
                models_tested = compare_run['metadata'].get('models', [])
                hardware_tested = compare_run['metadata'].get('hardware', [])
                
                if models_tested:
                    f.write(f"- Models: {', '.join(models_tested)}\n")
                
                if hardware_tested:
                    f.write(f"- Hardware: {', '.join(hardware_tested)}\n")
            
            # Summary
            f.write("\n## Summary\n\n")
            f.write(f"- Total comparisons: {total_comparisons}\n")
            f.write(f"- Regression threshold: {self.threshold}%\n")
            f.write(f"- Detected regressions: {regression_count}\n")
            f.write(f"- Regression rate: {(regression_count / total_comparisons * 100):.2f}% of comparisons\n\n")
            
            # Insert plots
            f.write("## Performance Comparison Plots\n\n")
            
            if latency_plot.exists():
                rel_path = os.path.relpath(latency_plot, os.path.dirname(output_file))
                f.write(f"### Latency Comparison\n\n")
                f.write(f"![Latency Comparison]({rel_path})\n\n")
            
            if throughput_plot.exists():
                rel_path = os.path.relpath(throughput_plot, os.path.dirname(output_file))
                f.write(f"### Throughput Comparison\n\n")
                f.write(f"![Throughput Comparison]({rel_path})\n\n")
            
            if memory_plot.exists():
                rel_path = os.path.relpath(memory_plot, os.path.dirname(output_file))
                f.write(f"### Memory Usage Comparison\n\n")
                f.write(f"![Memory Usage Comparison]({rel_path})\n\n")
            
            # Regression details
            f.write("## Regression Details\n\n")
            
            if regression_count > 0:
                # Filter for regressions
                regression_df = comparison_df[comparison_df['has_regression']]
                
                f.write("| Model | Hardware | Batch Size | Metric | Base Value | Compare Value | Change (%) |\n")
                f.write("|-------|----------|------------|--------|------------|---------------|------------|\n")
                
                for _, row in regression_df.iterrows():
                    model = row['model_name_base']
                    hardware = row['hardware_type_base']
                    batch_size = row['batch_size_base']
                    
                    # Check each metric for regression
                    if row['latency_regression']:
                        base_val = row['average_latency_ms_base']
                        comp_val = row['average_latency_ms_compare']
                        change = row['latency_change']
                        f.write(f"| {model} | {hardware} | {batch_size} | Latency | {base_val:.2f} ms | {comp_val:.2f} ms | {change:.2f}% |\n")
                    
                    if row['throughput_regression']:
                        base_val = row['throughput_items_per_second_base']
                        comp_val = row['throughput_items_per_second_compare']
                        change = row['throughput_change']
                        f.write(f"| {model} | {hardware} | {batch_size} | Throughput | {base_val:.2f} items/s | {comp_val:.2f} items/s | {change:.2f}% |\n")
                    
                    if row['memory_regression']:
                        base_val = row['memory_peak_mb_base']
                        comp_val = row['memory_peak_mb_compare']
                        change = row['memory_change']
                        f.write(f"| {model} | {hardware} | {batch_size} | Memory | {base_val:.2f} MB | {comp_val:.2f} MB | {change:.2f}% |\n")
            else:
                f.write("No regressions detected.\n\n")
            
            # Detailed comparison table
            f.write("\n## Detailed Comparison\n\n")
            
            # Sort by model, hardware, batch size
            sorted_df = comparison_df.sort_values(by=['model_name_base', 'hardware_type_base', 'batch_size_base'])
            
            f.write("| Model | Hardware | Batch | Latency Base | Latency Compare | Latency Δ | Throughput Base | Throughput Compare | Throughput Δ |\n")
            f.write("|-------|----------|-------|--------------|-----------------|-----------|-----------------|-------------------|-------------|\n")
            
            for _, row in sorted_df.iterrows():
                model = row['model_name_base'] if not pd.isnull(row.get('model_name_base', None)) else row['model_name_compare']
                hardware = row['hardware_type_base'] if not pd.isnull(row.get('hardware_type_base', None)) else row['hardware_type_compare']
                batch = row['batch_size_base'] if not pd.isnull(row.get('batch_size_base', None)) else row['batch_size_compare']
                
                latency_base = f"{row['average_latency_ms_base']:.2f}" if not pd.isnull(row.get('average_latency_ms_base', None)) else "N/A"
                latency_compare = f"{row['average_latency_ms_compare']:.2f}" if not pd.isnull(row.get('average_latency_ms_compare', None)) else "N/A"
                latency_change = f"{row['latency_change']:.2f}%" if not pd.isnull(row.get('latency_change', None)) else "N/A"
                
                throughput_base = f"{row['throughput_items_per_second_base']:.2f}" if not pd.isnull(row.get('throughput_items_per_second_base', None)) else "N/A"
                throughput_compare = f"{row['throughput_items_per_second_compare']:.2f}" if not pd.isnull(row.get('throughput_items_per_second_compare', None)) else "N/A"
                throughput_change = f"{row['throughput_change']:.2f}%" if not pd.isnull(row.get('throughput_change', None)) else "N/A"
                
                f.write(f"| {model} | {hardware} | {batch} | {latency_base} | {latency_compare} | {latency_change} | {throughput_base} | {throughput_compare} | {throughput_change} |\n")
            
            # Footer
            f.write("\n## Analysis Methodology\n\n")
            f.write(f"This report compares performance data between two benchmark runs with threshold {self.threshold}% for regression detection. ")
            f.write("A regression is flagged when:\n\n")
            f.write(f"- Latency increases by more than {self.threshold}%\n")
            f.write(f"- Throughput decreases by more than {self.threshold}%\n")
            f.write(f"- Memory usage increases by more than {self.threshold}%\n\n")
            f.write(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Report generated: {output_file}")
        return str(output_file)

def main():
    """Parse arguments and run the regression analyzer."""
    parser = argparse.ArgumentParser(description="Benchmark regression analyzer")
    
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--output-dir", default="./regression_reports",
                       help="Directory for output reports")
    
    # Run selection options
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--compare-runs", nargs=2, type=int, metavar=('BASE_RUN_ID', 'COMPARE_RUN_ID'),
                          help="IDs of two test runs to compare")
    run_group.add_argument("--latest", type=int, default=2, 
                          help="Compare the latest N runs (defaults to 2)")
    run_group.add_argument("--list-runs", action="store_true",
                          help="List available test runs")
    
    # Filtering options
    parser.add_argument("--models", nargs="+",
                       help="Filter comparison to specific models")
    
    # Reporting options
    parser.add_argument("--threshold", type=float, default=5.0,
                       help="Threshold percentage for regression detection")
    parser.add_argument("--output", type=str,
                       help="Output file for the report")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    try:
        # Create the regression analyzer
        analyzer = BenchmarkRegressionAnalyzer(
            db_path = args.db_path
    if db_path is None:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        logger.info(f"Using database path from environment: {db_path}"),
            output_dir=args.output_dir,
            threshold=args.threshold,
            debug=args.debug
        )
        
        # Handle command
        if args.list_runs:
            # List available runs
            runs_df = analyzer.get_test_runs(limit=20)
            print("\nAvailable test runs:\n")
            print("| Run ID | Test Name | Started At | Duration (s) |")
            print("|--------|-----------|------------|--------------|")
            
            for _, run in runs_df.iterrows():
                print(f"| {run['run_id']} | {run['test_name']} | {run['started_at']} | {run['execution_time_seconds']:.2f} |")
                
            print("\nUse --compare-runs to compare specific runs.")
            return 0
            
        elif args.compare_runs:
            # Compare specific runs
            base_run_id, compare_run_id = args.compare_runs
            
            comparison = analyzer.compare_runs(
                base_run_id=base_run_id,
                compare_run_id=compare_run_id,
                models=args.models
            )
            
            if comparison.empty:
                logger.error("No comparable data found between the runs")
                return 1
            
            # Generate report
            report_file = analyzer.generate_report(
                comparison_df=comparison,
                base_run_id=base_run_id,
                compare_run_id=compare_run_id,
                output_file=args.output
            )
            
            print(f"\nRegression analysis report generated: {report_file}")
            
            # Print basic summary
            regression_count = comparison['has_regression'].sum() if 'has_regression' in comparison.columns else 0
            total_comparisons = len(comparison)
            
            print(f"\nSummary:")
            print(f"- Regression threshold: {args.threshold}%")
            print(f"- Detected regressions: {regression_count} out of {total_comparisons} comparisons")
            print(f"- Regression rate: {(regression_count / total_comparisons * 100):.2f}%\n")
            
            return 0
            
        elif args.latest:
            # Compare latest runs
            if args.latest < 2:
                logger.error("Need at least 2 runs to compare")
                return 1
            
            latest_runs = analyzer.get_latest_runs(count=args.latest)
            
            if len(latest_runs) < 2:
                logger.error("Not enough runs in the database for comparison")
                return 1
            
            # Compare the latest two runs
            base_run_id = latest_runs[1]  # Second most recent
            compare_run_id = latest_runs[0]  # Most recent
            
            comparison = analyzer.compare_runs(
                base_run_id=base_run_id,
                compare_run_id=compare_run_id,
                models=args.models
            )
            
            if comparison.empty:
                logger.error("No comparable data found between the latest runs")
                return 1
            
            # Generate report
            report_file = analyzer.generate_report(
                comparison_df=comparison,
                base_run_id=base_run_id,
                compare_run_id=compare_run_id,
                output_file=args.output
            )
            
            print(f"\nRegression analysis report generated: {report_file}")
            
            # Print basic summary
            regression_count = comparison['has_regression'].sum() if 'has_regression' in comparison.columns else 0
            total_comparisons = len(comparison)
            
            print(f"\nSummary:")
            print(f"- Compared runs: {base_run_id} (base) vs {compare_run_id} (latest)")
            print(f"- Regression threshold: {args.threshold}%")
            print(f"- Detected regressions: {regression_count} out of {total_comparisons} comparisons")
            print(f"- Regression rate: {(regression_count / total_comparisons * 100):.2f}%\n")
            
            return 0
    
    except Exception as e:
        logger.error(f"Error during regression analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())