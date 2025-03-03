#!/usr/bin/env python3
"""
Benchmark Query Interface for IPFS Accelerate Framework

This module provides a command-line and API interface for querying and analyzing
benchmark data from the benchmark database.

Features:
- Command-line interface for querying benchmark data
- Advanced filtering and aggregation capabilities
- Report generation in various formats
- Visualization of benchmark results
- Export functionality for further analysis
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Import the benchmark database
from benchmark_database import BenchmarkDatabase

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkQuery:
    """
    Interface for querying and analyzing benchmark data.
    """
    
    def __init__(self, database_path: str = "./benchmark_database"):
        """
        Initialize the benchmark query interface.
        
        Args:
            database_path: Path to the benchmark database
        """
        self.db = BenchmarkDatabase(database_path)
        logger.info(f"Connected to benchmark database at {database_path}")
        
        # Create output directory for reports and visualizations
        self.output_dir = Path("./benchmark_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set default report template
        self.report_template = {
            "title": "Benchmark Report",
            "timestamp": datetime.datetime.now().isoformat(),
            "sections": [
                {
                    "title": "Summary",
                    "type": "summary"
                },
                {
                    "title": "Hardware Comparison",
                    "type": "hardware_comparison"
                },
                {
                    "title": "Model Comparison",
                    "type": "model_comparison"
                },
                {
                    "title": "Batch Size Scaling",
                    "type": "batch_size_scaling"
                }
            ]
        }
    
    def get_latest_model_performance(self, 
                                    model_name: Optional[str] = None,
                                    model_family: Optional[str] = None,
                                    hardware_type: Optional[str] = None,
                                    batch_size: Optional[int] = None,
                                    test_type: str = "inference") -> pd.DataFrame:
        """
        Get the latest performance metrics for models.
        
        Args:
            model_name: Filter by specific model name
            model_family: Filter by model family
            hardware_type: Filter by hardware type
            batch_size: Filter by batch size
            test_type: Test type (inference or training)
            
        Returns:
            pd.DataFrame: Latest performance metrics
        """
        # Query the database for latest benchmarks
        results = self.db.get_latest_benchmarks(
            model_name=model_name,
            model_family=model_family,
            hardware_type=hardware_type,
            test_type=test_type
        )
        
        if len(results) == 0:
            logger.warning("No benchmark results found matching the criteria")
            return pd.DataFrame()
        
        # Apply batch size filter if provided
        if batch_size is not None:
            results = results[results["batch_size"] == batch_size]
        
        # Extract key performance metrics
        metrics = ["model_name", "model_family", "hardware_type", "batch_size", 
                  "avg_latency", "throughput", "peak_memory_usage"]
        
        if test_type == "training":
            metrics.extend(["samples_per_second", "gradient_computation_time"])
        
        # Extract the metrics (handling potential missing columns)
        available_metrics = [m for m in metrics if m in results.columns]
        performance_df = results[available_metrics].copy()
        
        # Calculate millisecond latency for easier reading
        if "avg_latency" in performance_df.columns:
            performance_df["latency_ms"] = performance_df["avg_latency"] * 1000
        
        # Format memory usage for easier reading
        if "peak_memory_usage" in performance_df.columns:
            performance_df["memory_mb"] = performance_df["peak_memory_usage"] / (1024 * 1024)
        
        logger.info(f"Retrieved performance metrics for {len(performance_df)} model-hardware combinations")
        return performance_df
    
    def compare_hardware_platforms(self,
                                  model_name: Optional[str] = None,
                                  model_family: Optional[str] = None,
                                  batch_size: int = 1,
                                  metrics: List[str] = ["throughput", "avg_latency", "peak_memory_usage"]) -> Dict[str, pd.DataFrame]:
        """
        Compare performance across hardware platforms.
        
        Args:
            model_name: Filter by specific model name
            model_family: Filter by model family
            batch_size: Batch size for comparison
            metrics: List of metrics to compare
            
        Returns:
            Dict[str, pd.DataFrame]: Comparison dataframes for each metric
        """
        if model_name is None and model_family is None:
            logger.error("Either model_name or model_family must be provided")
            return {}
        
        comparisons = {}
        
        for metric in metrics:
            comparison = self.db.get_hardware_comparison(
                model_name=model_name,
                model_family=model_family,
                batch_size=batch_size,
                metric=metric
            )
            
            if len(comparison) > 0:
                comparisons[metric] = comparison
                logger.info(f"Generated hardware comparison for {metric}")
            else:
                logger.warning(f"No data available for hardware comparison of {metric}")
        
        return comparisons
    
    def compare_models(self,
                      model_family: str,
                      hardware_type: str,
                      batch_size: int = 1,
                      metrics: List[str] = ["throughput", "avg_latency", "peak_memory_usage"]) -> Dict[str, pd.DataFrame]:
        """
        Compare performance across models within a family.
        
        Args:
            model_family: Model family to compare
            hardware_type: Hardware type for comparison
            batch_size: Batch size for comparison
            metrics: List of metrics to compare
            
        Returns:
            Dict[str, pd.DataFrame]: Comparison dataframes for each metric
        """
        comparisons = {}
        
        for metric in metrics:
            comparison = self.db.get_model_comparison(
                model_family=model_family,
                hardware_type=hardware_type,
                batch_size=batch_size,
                metric=metric
            )
            
            if len(comparison) > 0:
                comparisons[metric] = comparison
                logger.info(f"Generated model comparison for {metric}")
            else:
                logger.warning(f"No data available for model comparison of {metric}")
        
        return comparisons
    
    def analyze_batch_scaling(self,
                             model_name: str,
                             hardware_type: str,
                             metrics: List[str] = ["throughput", "avg_latency"]) -> Dict[str, pd.DataFrame]:
        """
        Analyze how performance scales with batch size.
        
        Args:
            model_name: Model name
            hardware_type: Hardware type
            metrics: List of metrics to analyze
            
        Returns:
            Dict[str, pd.DataFrame]: Batch scaling data for each metric
        """
        scaling_data = {}
        
        for metric in metrics:
            scaling = self.db.get_batch_size_scaling(
                model_name=model_name,
                hardware_type=hardware_type,
                metric=metric
            )
            
            if len(scaling) > 0:
                scaling_data[metric] = scaling
                logger.info(f"Generated batch scaling analysis for {metric}")
            else:
                logger.warning(f"No data available for batch scaling analysis of {metric}")
        
        return scaling_data
    
    def generate_report(self, 
                       title: str = "Benchmark Performance Report",
                       model_name: Optional[str] = None,
                       model_family: Optional[str] = None,
                       hardware_type: Optional[str] = None,
                       output_format: str = "markdown") -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            title: Report title
            model_name: Filter by model name
            model_family: Filter by model family
            hardware_type: Filter by hardware type
            output_format: Output format (markdown, html, json)
            
        Returns:
            str: Path to the generated report
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine filename suffix based on filters
        suffix = ""
        if model_name:
            suffix += f"_{model_name}"
        elif model_family:
            suffix += f"_{model_family}"
        if hardware_type:
            suffix += f"_{hardware_type}"
        
        # Create report data structure
        report_data = {
            "title": title,
            "timestamp": datetime.datetime.now().isoformat(),
            "filters": {
                "model_name": model_name,
                "model_family": model_family,
                "hardware_type": hardware_type
            },
            "database_stats": self.db.get_statistics(),
            "sections": []
        }
        
        # Add summary section
        summary_df = self.get_latest_model_performance(
            model_name=model_name,
            model_family=model_family,
            hardware_type=hardware_type
        )
        
        if len(summary_df) > 0:
            report_data["sections"].append({
                "title": "Performance Summary",
                "type": "summary",
                "data": summary_df.to_dict(orient="records")
            })
        
        # Add hardware comparison section
        if not hardware_type:
            hw_comparisons = self.compare_hardware_platforms(
                model_name=model_name,
                model_family=model_family
            )
            
            if hw_comparisons:
                report_data["sections"].append({
                    "title": "Hardware Platform Comparison",
                    "type": "hardware_comparison",
                    "data": {metric: df.reset_index().to_dict(orient="records") 
                            for metric, df in hw_comparisons.items()}
                })
        
        # Add model comparison section if model family is specified
        if model_family and hardware_type:
            model_comparisons = self.compare_models(
                model_family=model_family,
                hardware_type=hardware_type
            )
            
            if model_comparisons:
                report_data["sections"].append({
                    "title": "Model Comparison",
                    "type": "model_comparison",
                    "data": {metric: df.reset_index().to_dict(orient="records") 
                            for metric, df in model_comparisons.items()}
                })
        
        # Add batch size scaling section if model name is specified
        if model_name and hardware_type:
            batch_scaling = self.analyze_batch_scaling(
                model_name=model_name,
                hardware_type=hardware_type
            )
            
            if batch_scaling:
                report_data["sections"].append({
                    "title": "Batch Size Scaling Analysis",
                    "type": "batch_size_scaling",
                    "data": {metric: df.to_dict(orient="records") 
                            for metric, df in batch_scaling.items()}
                })
        
        # Generate report in the specified format
        if output_format == "json":
            report_path = self.output_dir / f"benchmark_report{suffix}_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
        elif output_format == "html":
            report_path = self.output_dir / f"benchmark_report{suffix}_{timestamp}.html"
            self._generate_html_report(report_data, report_path)
            
        else:  # default to markdown
            report_path = self.output_dir / f"benchmark_report{suffix}_{timestamp}.md"
            self._generate_markdown_report(report_data, report_path)
        
        logger.info(f"Generated benchmark report at {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self, report_data: Dict, output_path: Path) -> None:
        """Generate a markdown report from the report data"""
        with open(output_path, 'w') as f:
            # Write header
            f.write(f"# {report_data['title']}\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write filters
            f.write("## Report Filters\n\n")
            filters = report_data["filters"]
            for key, value in filters.items():
                if value:
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Write database stats
            f.write("## Database Statistics\n\n")
            stats = report_data["database_stats"]
            for key, value in stats.items():
                if not isinstance(value, dict):
                    f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Write each section
            for section in report_data["sections"]:
                f.write(f"## {section['title']}\n\n")
                
                if section["type"] == "summary":
                    df = pd.DataFrame(section["data"])
                    
                    # Write summary table
                    if len(df) > 0:
                        # Select columns to display
                        display_cols = ["model_name", "hardware_type", "batch_size", 
                                      "latency_ms", "throughput", "memory_mb"]
                        display_cols = [col for col in display_cols if col in df.columns]
                        
                        # Write table header
                        f.write("| " + " | ".join(display_cols) + " |\n")
                        f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
                        
                        # Write table rows
                        for _, row in df[display_cols].iterrows():
                            values = []
                            for col in display_cols:
                                if col in ["latency_ms", "throughput", "memory_mb"]:
                                    values.append(f"{row[col]:.2f}")
                                else:
                                    values.append(str(row[col]))
                            f.write("| " + " | ".join(values) + " |\n")
                        
                        f.write("\n")
                    else:
                        f.write("No summary data available.\n\n")
                
                elif section["type"] == "hardware_comparison":
                    for metric, data in section["data"].items():
                        f.write(f"### {metric.replace('_', ' ').title()} Comparison\n\n")
                        df = pd.DataFrame(data)
                        
                        if len(df) > 0:
                            # Create pivot table
                            model_col = "model_name"
                            hardware_cols = [col for col in df.columns if col not in ["model_name", metric]]
                            
                            # Write table header
                            f.write(f"| {model_col} |")
                            for hw_col in hardware_cols:
                                f.write(f" {hw_col} |")
                            f.write("\n")
                            
                            f.write(f"| --- |")
                            for _ in hardware_cols:
                                f.write(" --- |")
                            f.write("\n")
                            
                            # Write table rows
                            for _, row in df.iterrows():
                                f.write(f"| {row[model_col]} |")
                                for hw_col in hardware_cols:
                                    if hw_col in row and pd.notna(row[hw_col]):
                                        f.write(f" {row[hw_col]:.2f} |")
                                    else:
                                        f.write(" - |")
                                f.write("\n")
                            
                            f.write("\n")
                        else:
                            f.write("No comparison data available.\n\n")
                
                elif section["type"] == "model_comparison":
                    for metric, data in section["data"].items():
                        f.write(f"### {metric.replace('_', ' ').title()} Comparison\n\n")
                        
                        if data:
                            # Convert to dataframe
                            df = pd.DataFrame(data)
                            
                            # Write sorted table
                            f.write("| Model | Value |\n")
                            f.write("| --- | --- |\n")
                            
                            # Sort based on the metric
                            if metric in ["throughput", "samples_per_second"]:
                                # Higher is better
                                sorted_df = df.sort_values(by=metric, ascending=False)
                            else:
                                # Lower is better
                                sorted_df = df.sort_values(by=metric, ascending=True)
                            
                            for _, row in sorted_df.iterrows():
                                model_name = row["model_name"] if "model_name" in row else "Unknown"
                                value = row[metric] if metric in row else 0
                                f.write(f"| {model_name} | {value:.2f} |\n")
                            
                            f.write("\n")
                        else:
                            f.write("No comparison data available.\n\n")
                
                elif section["type"] == "batch_size_scaling":
                    for metric, data in section["data"].items():
                        f.write(f"### {metric.replace('_', ' ').title()} Scaling\n\n")
                        
                        if data:
                            # Convert to dataframe
                            df = pd.DataFrame(data)
                            
                            # Write table
                            f.write("| Batch Size | Value |\n")
                            f.write("| --- | --- |\n")
                            
                            for _, row in df.sort_values(by="batch_size").iterrows():
                                batch_size = row["batch_size"]
                                value = row[metric]
                                f.write(f"| {batch_size} | {value:.2f} |\n")
                            
                            f.write("\n")
                        else:
                            f.write("No scaling data available.\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write(f"Generated by IPFS Accelerate Benchmark Query on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def _generate_html_report(self, report_data: Dict, output_path: Path) -> None:
        """Generate an HTML report from the report data"""
        if not HAS_VISUALIZATION:
            logger.warning("Visualization libraries not available. Generating basic HTML report without plots.")
        
        # Create basic HTML
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report_data['title']}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".chart { width: 800px; height: 400px; margin: 20px 0; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report_data['title']}</h1>",
            f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Add filters
        html.append("<h2>Report Filters</h2>")
        html.append("<ul>")
        filters = report_data["filters"]
        for key, value in filters.items():
            if value:
                html.append(f"<li><strong>{key}</strong>: {value}</li>")
        html.append("</ul>")
        
        # Add database stats
        html.append("<h2>Database Statistics</h2>")
        html.append("<ul>")
        stats = report_data["database_stats"]
        for key, value in stats.items():
            if not isinstance(value, dict):
                html.append(f"<li><strong>{key}</strong>: {value}</li>")
        html.append("</ul>")
        
        # Add each section
        for section in report_data["sections"]:
            html.append(f"<h2>{section['title']}</h2>")
            
            if section["type"] == "summary":
                df = pd.DataFrame(section["data"])
                
                if len(df) > 0:
                    # Select columns to display
                    display_cols = ["model_name", "hardware_type", "batch_size", 
                                  "latency_ms", "throughput", "memory_mb"]
                    display_cols = [col for col in display_cols if col in df.columns]
                    
                    # Create HTML table
                    html.append("<table>")
                    html.append("<tr>")
                    for col in display_cols:
                        html.append(f"<th>{col}</th>")
                    html.append("</tr>")
                    
                    for _, row in df[display_cols].iterrows():
                        html.append("<tr>")
                        for col in display_cols:
                            if col in ["latency_ms", "throughput", "memory_mb"]:
                                html.append(f"<td>{row[col]:.2f}</td>")
                            else:
                                html.append(f"<td>{row[col]}</td>")
                        html.append("</tr>")
                    
                    html.append("</table>")
                    
                    # Add chart if visualization is available
                    if HAS_VISUALIZATION and len(df) > 1:
                        chart_path = self._create_performance_chart(df, "summary", self.output_dir)
                        if chart_path:
                            html.append(f'<img src="{os.path.basename(chart_path)}" class="chart" />')
                else:
                    html.append("<p>No summary data available.</p>")
            
            elif section["type"] == "hardware_comparison":
                for metric, data in section["data"].items():
                    html.append(f"<h3>{metric.replace('_', ' ').title()} Comparison</h3>")
                    
                    if data:
                        df = pd.DataFrame(data)
                        
                        if len(df) > 0:
                            # Create HTML table
                            model_col = "model_name"
                            hardware_cols = [col for col in df.columns if col not in ["model_name", metric]]
                            
                            html.append("<table>")
                            html.append("<tr>")
                            html.append(f"<th>{model_col}</th>")
                            for hw_col in hardware_cols:
                                html.append(f"<th>{hw_col}</th>")
                            html.append("</tr>")
                            
                            for _, row in df.iterrows():
                                html.append("<tr>")
                                html.append(f"<td>{row[model_col]}</td>")
                                for hw_col in hardware_cols:
                                    if hw_col in row and pd.notna(row[hw_col]):
                                        html.append(f"<td>{row[hw_col]:.2f}</td>")
                                    else:
                                        html.append("<td>-</td>")
                                html.append("</tr>")
                            
                            html.append("</table>")
                            
                            # Add chart if visualization is available
                            if HAS_VISUALIZATION:
                                chart_path = self._create_hardware_comparison_chart(df, metric, hardware_cols, self.output_dir)
                                if chart_path:
                                    html.append(f'<img src="{os.path.basename(chart_path)}" class="chart" />')
                        else:
                            html.append("<p>No comparison data available.</p>")
                    else:
                        html.append("<p>No comparison data available.</p>")
            
            elif section["type"] == "model_comparison":
                for metric, data in section["data"].items():
                    html.append(f"<h3>{metric.replace('_', ' ').title()} Comparison</h3>")
                    
                    if data:
                        # Convert to dataframe
                        df = pd.DataFrame(data)
                        
                        if len(df) > 0:
                            # Sort based on the metric
                            if metric in ["throughput", "samples_per_second"]:
                                # Higher is better
                                sorted_df = df.sort_values(by=metric, ascending=False)
                            else:
                                # Lower is better
                                sorted_df = df.sort_values(by=metric, ascending=True)
                            
                            # Create HTML table
                            html.append("<table>")
                            html.append("<tr><th>Model</th><th>Value</th></tr>")
                            
                            for _, row in sorted_df.iterrows():
                                model_name = row["model_name"] if "model_name" in row else "Unknown"
                                value = row[metric] if metric in row else 0
                                html.append(f"<tr><td>{model_name}</td><td>{value:.2f}</td></tr>")
                            
                            html.append("</table>")
                            
                            # Add chart if visualization is available
                            if HAS_VISUALIZATION:
                                chart_path = self._create_model_comparison_chart(sorted_df, metric, self.output_dir)
                                if chart_path:
                                    html.append(f'<img src="{os.path.basename(chart_path)}" class="chart" />')
                        else:
                            html.append("<p>No comparison data available.</p>")
                    else:
                        html.append("<p>No comparison data available.</p>")
            
            elif section["type"] == "batch_size_scaling":
                for metric, data in section["data"].items():
                    html.append(f"<h3>{metric.replace('_', ' ').title()} Scaling</h3>")
                    
                    if data:
                        # Convert to dataframe
                        df = pd.DataFrame(data)
                        
                        if len(df) > 0:
                            # Create HTML table
                            html.append("<table>")
                            html.append("<tr><th>Batch Size</th><th>Value</th></tr>")
                            
                            for _, row in df.sort_values(by="batch_size").iterrows():
                                batch_size = row["batch_size"]
                                value = row[metric]
                                html.append(f"<tr><td>{batch_size}</td><td>{value:.2f}</td></tr>")
                            
                            html.append("</table>")
                            
                            # Add chart if visualization is available
                            if HAS_VISUALIZATION:
                                chart_path = self._create_batch_scaling_chart(df, metric, self.output_dir)
                                if chart_path:
                                    html.append(f'<img src="{os.path.basename(chart_path)}" class="chart" />')
                        else:
                            html.append("<p>No scaling data available.</p>")
                    else:
                        html.append("<p>No scaling data available.</p>")
        
        # Add footer
        html.append("<hr />")
        html.append(f"<p>Generated by IPFS Accelerate Benchmark Query on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append("</body></html>")
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write("\n".join(html))
    
    def _create_performance_chart(self, df: pd.DataFrame, chart_type: str, output_dir: Path) -> Optional[str]:
        """Create a performance chart and save to file"""
        if not HAS_VISUALIZATION or len(df) == 0:
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = output_dir / f"chart_{chart_type}_{timestamp}.png"
            
            plt.figure(figsize=(10, 6))
            
            if "hardware_type" in df.columns and "latency_ms" in df.columns:
                # Create bar chart grouped by hardware type
                sns.barplot(x="model_name", y="latency_ms", hue="hardware_type", data=df)
                plt.title("Latency by Model and Hardware")
                plt.xlabel("Model")
                plt.ylabel("Latency (ms)")
                plt.xticks(rotation=45, ha="right")
                plt.legend(title="Hardware")
                plt.tight_layout()
                
                plt.savefig(chart_path)
                plt.close()
                
                return str(chart_path)
            
            return None
        
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return None
    
    def _create_hardware_comparison_chart(self, df: pd.DataFrame, metric: str, hardware_cols: List[str], output_dir: Path) -> Optional[str]:
        """Create a hardware comparison chart and save to file"""
        if not HAS_VISUALIZATION or len(df) == 0:
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = output_dir / f"chart_hw_comparison_{metric}_{timestamp}.png"
            
            # Prepare data for plotting
            plot_df = df.melt(
                id_vars=["model_name"],
                value_vars=hardware_cols,
                var_name="hardware_type",
                value_name=metric
            )
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x="model_name", y=metric, hue="hardware_type", data=plot_df)
            plt.title(f"{metric.replace('_', ' ').title()} by Model and Hardware")
            plt.xlabel("Model")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Hardware")
            plt.tight_layout()
            
            plt.savefig(chart_path)
            plt.close()
            
            return str(chart_path)
        
        except Exception as e:
            logger.error(f"Error creating hardware comparison chart: {e}")
            return None
    
    def _create_model_comparison_chart(self, df: pd.DataFrame, metric: str, output_dir: Path) -> Optional[str]:
        """Create a model comparison chart and save to file"""
        if not HAS_VISUALIZATION or len(df) == 0:
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = output_dir / f"chart_model_comparison_{metric}_{timestamp}.png"
            
            plt.figure(figsize=(10, 6))
            
            # Create horizontal bar chart for model comparison
            ax = sns.barplot(x=metric, y="model_name", data=df, orient="h")
            
            # Add value labels to the bars
            for i, v in enumerate(df[metric]):
                ax.text(v, i, f" {v:.2f}", va="center")
            
            plt.title(f"{metric.replace('_', ' ').title()} by Model")
            plt.xlabel(metric.replace('_', ' ').title())
            plt.ylabel("Model")
            plt.tight_layout()
            
            plt.savefig(chart_path)
            plt.close()
            
            return str(chart_path)
        
        except Exception as e:
            logger.error(f"Error creating model comparison chart: {e}")
            return None
    
    def _create_batch_scaling_chart(self, df: pd.DataFrame, metric: str, output_dir: Path) -> Optional[str]:
        """Create a batch scaling chart and save to file"""
        if not HAS_VISUALIZATION or len(df) == 0:
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = output_dir / f"chart_batch_scaling_{metric}_{timestamp}.png"
            
            plt.figure(figsize=(10, 6))
            
            # Sort by batch size and plot line chart
            sorted_df = df.sort_values(by="batch_size")
            plt.plot(sorted_df["batch_size"], sorted_df[metric], marker="o", linewidth=2)
            
            # Add value labels to the points
            for x, y in zip(sorted_df["batch_size"], sorted_df[metric]):
                plt.text(x, y, f" {y:.2f}", va="bottom")
            
            plt.title(f"{metric.replace('_', ' ').title()} Scaling with Batch Size")
            plt.xlabel("Batch Size")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(chart_path)
            plt.close()
            
            return str(chart_path)
        
        except Exception as e:
            logger.error(f"Error creating batch scaling chart: {e}")
            return None

def main():
    """Main function for the benchmark query command-line interface"""
    parser = argparse.ArgumentParser(description="Benchmark Query Interface")
    parser.add_argument("--database", type=str, default="./benchmark_database", help="Path to benchmark database")
    
    # Command-specific subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Performance command
    perf_parser = subparsers.add_parser("performance", help="Get latest performance metrics")
    perf_parser.add_argument("--model", type=str, help="Filter by model name")
    perf_parser.add_argument("--family", type=str, help="Filter by model family")
    perf_parser.add_argument("--hardware", type=str, help="Filter by hardware type")
    perf_parser.add_argument("--batch-size", type=int, help="Filter by batch size")
    perf_parser.add_argument("--test-type", type=str, default="inference", choices=["inference", "training"], help="Test type")
    perf_parser.add_argument("--output", type=str, help="Output file for results (CSV format)")
    
    # Hardware comparison command
    hw_parser = subparsers.add_parser("hardware", help="Compare hardware platforms")
    hw_parser.add_argument("--model", type=str, help="Filter by model name")
    hw_parser.add_argument("--family", type=str, help="Filter by model family")
    hw_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for comparison")
    hw_parser.add_argument("--metric", type=str, default="throughput", help="Metric to compare")
    hw_parser.add_argument("--output", type=str, help="Output file for results (CSV format)")
    
    # Model comparison command
    model_parser = subparsers.add_parser("models", help="Compare models within a family")
    model_parser.add_argument("--family", type=str, required=True, help="Model family to compare")
    model_parser.add_argument("--hardware", type=str, required=True, help="Hardware type for comparison")
    model_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for comparison")
    model_parser.add_argument("--metric", type=str, default="throughput", help="Metric to compare")
    model_parser.add_argument("--output", type=str, help="Output file for results (CSV format)")
    
    # Batch size scaling command
    batch_parser = subparsers.add_parser("batch", help="Analyze batch size scaling")
    batch_parser.add_argument("--model", type=str, required=True, help="Model name")
    batch_parser.add_argument("--hardware", type=str, required=True, help="Hardware type")
    batch_parser.add_argument("--metric", type=str, default="throughput", help="Metric to analyze")
    batch_parser.add_argument("--output", type=str, help="Output file for results (CSV format)")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument("--title", type=str, default="Benchmark Performance Report", help="Report title")
    report_parser.add_argument("--model", type=str, help="Filter by model name")
    report_parser.add_argument("--family", type=str, help="Filter by model family")
    report_parser.add_argument("--hardware", type=str, help="Filter by hardware type")
    report_parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "html", "json"], help="Output format")
    
    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Get database statistics")
    
    args = parser.parse_args()
    
    # Create the query interface
    query = BenchmarkQuery(database_path=args.database)
    
    # Execute the appropriate command
    if args.command == "performance":
        results = query.get_latest_model_performance(
            model_name=args.model,
            model_family=args.family,
            hardware_type=args.hardware,
            batch_size=args.batch_size,
            test_type=args.test_type
        )
        
        if len(results) == 0:
            print("No results found matching the criteria.")
            return
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print(results.to_string())
    
    elif args.command == "hardware":
        if not args.model and not args.family:
            print("Error: Either --model or --family must be provided.")
            return
        
        comparison = query.db.get_hardware_comparison(
            model_name=args.model,
            model_family=args.family,
            batch_size=args.batch_size,
            metric=args.metric
        )
        
        if len(comparison) == 0:
            print("No comparison data found matching the criteria.")
            return
        
        if args.output:
            comparison.to_csv(args.output)
            print(f"Comparison saved to {args.output}")
        else:
            print(comparison.to_string())
    
    elif args.command == "models":
        comparison = query.db.get_model_comparison(
            model_family=args.family,
            hardware_type=args.hardware,
            batch_size=args.batch_size,
            metric=args.metric
        )
        
        if len(comparison) == 0:
            print("No comparison data found matching the criteria.")
            return
        
        if args.output:
            comparison.to_csv(args.output)
            print(f"Comparison saved to {args.output}")
        else:
            print(comparison.to_string())
    
    elif args.command == "batch":
        scaling = query.db.get_batch_size_scaling(
            model_name=args.model,
            hardware_type=args.hardware,
            metric=args.metric
        )
        
        if len(scaling) == 0:
            print("No batch scaling data found matching the criteria.")
            return
        
        if args.output:
            scaling.to_csv(args.output, index=False)
            print(f"Batch scaling data saved to {args.output}")
        else:
            print(scaling.to_string())
    
    elif args.command == "report":
        report_path = query.generate_report(
            title=args.title,
            model_name=args.model,
            model_family=args.family,
            hardware_type=args.hardware,
            output_format=args.format
        )
        
        print(f"Report generated at {report_path}")
    
    elif args.command == "stats":
        stats = query.db.get_statistics()
        
        print("Database Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()