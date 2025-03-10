#\!/usr/bin/env python
"""
Benchmark Regression Detector

This script analyzes benchmark data from the DuckDB database to detect
performance regressions between test runs. It can be used to automatically
detect when changes to the codebase lead to performance degradation.

Usage:
    python benchmark_regression_detector.py --db benchmark_db.duckdb --run-id "123456789"
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Define pandas/duckdb import flag
HAS_DEPENDENCIES = False

# Try to import dependencies
try:
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    HAS_DEPENDENCIES = True
except ImportError:
    # Create minimal pd stub for function definitions
    class DataFrame:
        def empty():
            return True
        
        def iterrows():
            return []
    
    class DataFrameStub:
        def fetchdf(self):
            return DataFrame()
    
    pd = type('', (), {})()
    pd.DataFrame = DataFrame

# Define metrics to analyze
DEFAULT_METRICS = [
    "throughput_items_per_second",
    "average_latency_ms",
    "peak_memory_mb"
]

# Define regression thresholds (percent change)
DEFAULT_THRESHOLD = 0.1  # 10%
HIGH_THRESHOLD = 0.2     # 20%

class BenchmarkRegressionDetector:
    """Class to detect regressions in benchmark results."""
    
    def __init__(self, db_path: str, run_id: Optional[str] = None, 
                 threshold: float = DEFAULT_THRESHOLD,
                 window: int = 5,
                 metrics: List[str] = None):
        """
        Initialize the detector.
        
        Args:
            db_path (str): Path to the DuckDB database
            run_id (str, optional): Run ID to compare with previous runs
            threshold (float): Threshold for detecting regressions (percent change)
            window (int): Number of previous runs to include in baseline
            metrics (List[str]): Metrics to analyze for regressions
        """
        self.db_path = db_path
        self.run_id = run_id
        self.threshold = threshold
        self.window = window
        self.metrics = metrics or DEFAULT_METRICS
        
        # Check if the database exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Connect to the database
        self.conn = duckdb.connect(db_path)
        
        # Store detected regressions
        self.regressions = []
    
    def detect_regressions(self) -> List[Dict[str, Any]]:
        """
        Detect regressions in benchmark results.
        
        Returns:
            List[Dict[str, Any]]: List of detected regressions
        """
        if not HAS_DEPENDENCIES:
            print("Error: Required dependencies not found. Please install:")
            print("  pip install duckdb pandas numpy matplotlib scipy")
            return []
        
        # If we don't have a run ID, use the latest run
        if self.run_id is None:
            self.run_id = self._get_latest_run_id()
            if self.run_id is None:
                print("Error: No run ID provided and no runs found in database")
                return []
        
        # Get the current run's data
        current_run_data = self._get_run_data(self.run_id)
        if current_run_data.empty:
            print(f"Error: No data found for run ID: {self.run_id}")
            return []
        
        # Get previous runs for comparison
        previous_runs = self._get_previous_runs(self.run_id, self.window)
        if not previous_runs:
            print(f"Warning: No previous runs found for comparison with {self.run_id}")
            return []
        
        # Get baseline data from previous runs
        baseline_data = self._get_baseline_data(previous_runs)
        if baseline_data.empty:
            print("Warning: No baseline data found for comparison")
            return []
        
        # Compare current run with baseline
        self.regressions = self._compare_with_baseline(current_run_data, baseline_data)
        
        return self.regressions
    
    def _get_latest_run_id(self) -> Optional[str]:
        """
        Get the latest run ID from the database.
        
        Returns:
            Optional[str]: Latest run ID or None if no runs found
        """
        query = """
        SELECT run_id
        FROM performance_results
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        try:
            result = self.conn.execute(query).fetchone()
            if result:
                return result[0]
        except Exception as e:
            print(f"Error getting latest run ID: {e}")
        
        return None
    
    def _get_run_data(self, run_id: str) -> pd.DataFrame:
        """
        Get benchmark data for a specific run.
        
        Args:
            run_id (str): Run ID to get data for
            
        Returns:
            pd.DataFrame: Benchmark data for the run
        """
        # Build query dynamically to include the requested metrics
        metrics_str = ", ".join([f"pr.{metric}" for metric in self.metrics])
        
        query = f"""
        SELECT pr.result_id, pr.timestamp, pr.run_id, 
               m.model_id, m.model_name, m.model_type,
               hp.hardware_id, hp.hardware_type, hp.hardware_model,
               pr.batch_size, {metrics_str}
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE pr.run_id = ?
        """
        
        try:
            return self.conn.execute(query, [run_id]).fetchdf()
        except Exception as e:
            print(f"Error getting run data: {e}")
            return pd.DataFrame()
    
    def _get_previous_runs(self, current_run_id: str, window: int) -> List[str]:
        """
        Get previous run IDs for comparison.
        
        Args:
            current_run_id (str): Current run ID
            window (int): Number of previous runs to include
            
        Returns:
            List[str]: List of previous run IDs
        """
        query = """
        SELECT DISTINCT run_id
        FROM performance_results
        WHERE run_id \!= ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        try:
            results = self.conn.execute(query, [current_run_id, window]).fetchall()
            return [result[0] for result in results]
        except Exception as e:
            print(f"Error getting previous runs: {e}")
            return []
    
    def _get_baseline_data(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Get baseline data from previous runs.
        
        Args:
            run_ids (List[str]): List of run IDs to include in baseline
            
        Returns:
            pd.DataFrame: Baseline data
        """
        if not run_ids:
            return pd.DataFrame()
        
        # Build placeholders for SQL query
        placeholders = ", ".join(["?" for _ in run_ids])
        
        # Build query dynamically to include the requested metrics
        metrics_str = ", ".join([f"pr.{metric}" for metric in self.metrics])
        
        query = f"""
        SELECT pr.result_id, pr.timestamp, pr.run_id, 
               m.model_id, m.model_name, m.model_type,
               hp.hardware_id, hp.hardware_type, hp.hardware_model,
               pr.batch_size, {metrics_str}
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        WHERE pr.run_id IN ({placeholders})
        """
        
        try:
            return self.conn.execute(query, run_ids).fetchdf()
        except Exception as e:
            print(f"Error getting baseline data: {e}")
            return pd.DataFrame()
    
    def _compare_with_baseline(self, current_data: pd.DataFrame, 
                               baseline_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compare current run with baseline to detect regressions.
        
        Args:
            current_data (pd.DataFrame): Current run data
            baseline_data (pd.DataFrame): Baseline data from previous runs
            
        Returns:
            List[Dict[str, Any]]: List of detected regressions
        """
        regressions = []
        
        # Group data by model, hardware, and batch size
        for _, current_group in current_data.groupby(['model_id', 'hardware_id', 'batch_size']):
            if current_group.empty:
                continue
            
            # Extract key information
            model_id = current_group['model_id'].iloc[0]
            model_name = current_group['model_name'].iloc[0]
            hardware_id = current_group['hardware_id'].iloc[0]
            hardware_type = current_group['hardware_type'].iloc[0]
            batch_size = current_group['batch_size'].iloc[0]
            
            # Get corresponding baseline data
            baseline_group = baseline_data[
                (baseline_data['model_id'] == model_id) &
                (baseline_data['hardware_id'] == hardware_id) &
                (baseline_data['batch_size'] == batch_size)
            ]
            
            if baseline_group.empty:
                continue
            
            # Check each metric for regressions
            for metric in self.metrics:
                if metric not in current_group.columns or metric not in baseline_group.columns:
                    continue
                
                # Get current and baseline values
                current_value = current_group[metric].mean()
                baseline_values = baseline_group[metric].values
                baseline_mean = baseline_values.mean()
                baseline_std = baseline_values.std()
                
                # Skip if baseline has no variance (likely only one sample)
                if baseline_std == 0:
                    baseline_std = baseline_mean * 0.01  # Assume 1% standard deviation
                
                # Calculate percent change
                percent_change = (current_value - baseline_mean) / baseline_mean
                
                # For metrics where lower is better (like latency), invert the sign
                if metric in ['average_latency_ms', 'peak_memory_mb']:
                    percent_change = -percent_change
                
                # Calculate z-score for statistical significance
                z_score = (current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
                
                # Determine if this is a regression
                is_regression = percent_change < -self.threshold
                severity = "high" if percent_change < -HIGH_THRESHOLD else "medium" if is_regression else "low"
                
                if is_regression:
                    regressions.append({
                        "model_id": int(model_id),
                        "model_name": model_name,
                        "hardware_id": int(hardware_id),
                        "hardware_type": hardware_type,
                        "batch_size": int(batch_size),
                        "metric": metric,
                        "current_value": float(current_value),
                        "baseline_mean": float(baseline_mean),
                        "baseline_std": float(baseline_std),
                        "percent_change": float(percent_change),
                        "z_score": float(z_score),
                        "severity": severity,
                        "run_id": self.run_id
                    })
        
        # Sort regressions by severity and percent change
        regressions.sort(key=lambda x: (
            0 if x["severity"] == "high" else 1 if x["severity"] == "medium" else 2,
            x["percent_change"]
        ))
        
        return regressions
    
    def generate_report(self, format="text", output=None):
        """
        Generate a report of detected regressions.
        
        Args:
            format (str): Output format ("text", "json", "html", "markdown")
            output (str): Output file path (None for stdout)
            
        Returns:
            str: Report content
        """
        if not self.regressions:
            if not output:
                print("No regressions detected.")
            return "No regressions detected."
        
        if format == "json":
            report = json.dumps(self.regressions, indent=2)
        elif format == "html":
            report = self._generate_html_report()
        elif format == "markdown":
            report = self._generate_markdown_report()
        else:  # text
            report = self._generate_text_report()
        
        if output:
            with open(output, 'w') as f:
                f.write(report)
            print(f"Report saved to {output}")
        else:
            print(report)
        
        return report
    
    def _generate_text_report(self) -> str:
        """
        Generate a text report of detected regressions.
        
        Returns:
            str: Text report
        """
        lines = ["Performance Regression Report", "=" * 30, ""]
        
        lines.append(f"Run ID: {self.run_id}")
        lines.append(f"Threshold: {self.threshold * 100:.1f}%")
        lines.append(f"Window: {self.window} previous runs")
        lines.append(f"Metrics: {', '.join(self.metrics)}")
        lines.append("")
        
        lines.append(f"Detected {len(self.regressions)} regressions:")
        lines.append("")
        
        for i, reg in enumerate(self.regressions, 1):
            lines.append(f"Regression {i}:")
            lines.append(f"  Model: {reg['model_name']}")
            lines.append(f"  Hardware: {reg['hardware_type']}")
            lines.append(f"  Batch Size: {reg['batch_size']}")
            lines.append(f"  Metric: {reg['metric']}")
            lines.append(f"  Current Value: {reg['current_value']:.2f}")
            lines.append(f"  Baseline Mean: {reg['baseline_mean']:.2f}")
            lines.append(f"  Percent Change: {reg['percent_change'] * 100:.2f}%")
            lines.append(f"  Severity: {reg['severity']}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self) -> str:
        """
        Generate an HTML report of detected regressions.
        
        Returns:
            str: HTML report
        """
        html = """
        <\!DOCTYPE html>
        <html>
        <head>
            <title>Performance Regression Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .high { background-color: #ffdddd; }
                .medium { background-color: #ffffcc; }
                .low { background-color: #ddffdd; }
            </style>
        </head>
        <body>
            <h1>Performance Regression Report</h1>
            
            <div class="summary">
                <p><strong>Run ID:</strong> {run_id}</p>
                <p><strong>Threshold:</strong> {threshold:.1f}%</p>
                <p><strong>Window:</strong> {window} previous runs</p>
                <p><strong>Metrics:</strong> {metrics}</p>
                <p><strong>Total Regressions:</strong> {total_regressions}</p>
            </div>
            
            <table>
                <tr>
                    <th>Model</th>
                    <th>Hardware</th>
                    <th>Batch Size</th>
                    <th>Metric</th>
                    <th>Current Value</th>
                    <th>Baseline Mean</th>
                    <th>Percent Change</th>
                    <th>Z-Score</th>
                    <th>Severity</th>
                </tr>
        """.format(
            run_id=self.run_id,
            threshold=self.threshold * 100,
            window=self.window,
            metrics=', '.join(self.metrics),
            total_regressions=len(self.regressions)
        )
        
        for reg in self.regressions:
            html += """
                <tr class="{severity}">
                    <td>{model_name}</td>
                    <td>{hardware_type}</td>
                    <td>{batch_size}</td>
                    <td>{metric}</td>
                    <td>{current_value:.2f}</td>
                    <td>{baseline_mean:.2f}</td>
                    <td>{percent_change:.2f}%</td>
                    <td>{z_score:.2f}</td>
                    <td>{severity}</td>
                </tr>
            """.format(
                model_name=reg['model_name'],
                hardware_type=reg['hardware_type'],
                batch_size=reg['batch_size'],
                metric=reg['metric'],
                current_value=reg['current_value'],
                baseline_mean=reg['baseline_mean'],
                percent_change=reg['percent_change'] * 100,
                z_score=reg['z_score'],
                severity=reg['severity']
            )
        
        html += """
            </table>
            
            <p>Generated on: {date}</p>
        </body>
        </html>
        """.format(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return html
    
    def _generate_markdown_report(self) -> str:
        """
        Generate a markdown report of detected regressions.
        
        Returns:
            str: Markdown report
        """
        lines = ["# Performance Regression Report", ""]
        
        lines.append(f"**Run ID:** {self.run_id}")
        lines.append(f"**Threshold:** {self.threshold * 100:.1f}%")
        lines.append(f"**Window:** {self.window} previous runs")
        lines.append(f"**Metrics:** {', '.join(self.metrics)}")
        lines.append(f"**Total Regressions:** {len(self.regressions)}")
        lines.append("")
        
        lines.append("## Detected Regressions")
        lines.append("")
        
        lines.append("| Model | Hardware | Batch Size | Metric | Current Value | Baseline Mean | % Change | Severity |")
        lines.append("|-------|----------|------------|--------|---------------|---------------|----------|----------|")
        
        for reg in self.regressions:
            lines.append(f"| {reg['model_name']} | {reg['hardware_type']} | {reg['batch_size']} | " +
                         f"{reg['metric']} | {reg['current_value']:.2f} | {reg['baseline_mean']:.2f} | " +
                         f"{reg['percent_change'] * 100:.2f}% | {reg['severity']} |")
        
        lines.append("")
        lines.append(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)
    
    def visualize_regressions(self, output=None):
        """
        Visualize the detected regressions.
        
        Args:
            output (str): Output file path (None for display)
        """
        if not HAS_DEPENDENCIES:
            print("Error: Required dependencies not found. Please install:")
            print("  pip install matplotlib numpy pandas")
            return
        
        if not self.regressions:
            print("No regressions to visualize.")
            return
        
        # Group regressions by model and hardware
        groups = {}
        for reg in self.regressions:
            key = (reg['model_name'], reg['hardware_type'])
            if key not in groups:
                groups[key] = []
            groups[key].append(reg)
        
        # Create a figure for each group
        for (model_name, hardware_type), regs in groups.items():
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            metrics = []
            current_values = []
            baseline_means = []
            percent_changes = []
            
            for reg in regs:
                metrics.append(f"{reg['metric']} (bs={reg['batch_size']})")
                current_values.append(reg['current_value'])
                baseline_means.append(reg['baseline_mean'])
                percent_changes.append(reg['percent_change'] * 100)
            
            # Create bar chart
            x = range(len(metrics))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], baseline_means, width, label='Baseline Mean')
            ax.bar([i + width/2 for i in x], current_values, width, label='Current Value')
            
            # Add percent change as text
            for i, (pct, curr) in enumerate(zip(percent_changes, current_values)):
                ax.text(i, curr, f"{pct:.1f}%", ha='center', va='bottom', 
                        color='red' if pct < 0 else 'green')
            
            # Add labels and title
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Value')
            ax.set_title(f'Performance Regression: {model_name} on {hardware_type}')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            
            # Save or display
            if output:
                output_path = f"{os.path.splitext(output)[0]}_{model_name}_{hardware_type}.png"
                plt.savefig(output_path)
                print(f"Visualization saved to {output_path}")
            else:
                plt.show()
            
            plt.close(fig)

def main():
    """Main function."""
    if not HAS_DEPENDENCIES:
        print("Error: Required dependencies not found. Please install:")
        print("  pip install duckdb pandas numpy matplotlib scipy")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Detect performance regressions in benchmark results.")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--run-id", help="Run ID to analyze (default: latest run)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Regression detection threshold (default: {DEFAULT_THRESHOLD * 100}%%)")
    parser.add_argument("--window", type=int, default=5,
                        help="Number of previous runs to include in baseline (default: 5)")
    parser.add_argument("--metrics", nargs="+",
                        help=f"Metrics to analyze (default: {', '.join(DEFAULT_METRICS)})")
    parser.add_argument("--format", choices=["text", "json", "html", "markdown"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--visualize", action="store_true", help="Visualize regressions")
    
    args = parser.parse_args()
    
    detector = BenchmarkRegressionDetector(
        db_path=args.db,
        run_id=args.run_id,
        threshold=args.threshold,
        window=args.window,
        metrics=args.metrics
    )
    
    # Detect regressions
    detector.detect_regressions()
    
    # Generate report
    detector.generate_report(format=args.format, output=args.output)
    
    # Visualize regressions if requested
    if args.visualize:
        detector.visualize_regressions(output=args.output)

if __name__ == "__main__":
    main()
