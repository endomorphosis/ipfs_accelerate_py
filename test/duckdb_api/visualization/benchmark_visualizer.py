#!/usr/bin/env python
"""
Benchmark visualization tool for creating performance charts.

This module provides tools for creating visual representations of benchmark data
to help understand performance across different models and hardware platforms.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

try:
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError:
    print("Error: Required packages not installed. Please install with:")
    print("pip install duckdb pandas matplotlib seaborn numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI

class BenchmarkVisualizer:
    """
    Benchmark visualization tool for creating performance charts.
    
    This class provides tools for creating visual representations of benchmark data
    to help understand performance across different models and hardware platforms.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the benchmark visualizer.
        
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
        
        # Configure plotting style
        sns.set_style("whitegrid")
        
        logger.info(f"Initialized BenchmarkVisualizer with database: {db_path}")
    
    def create_hardware_comparison_chart(self, model_name: str, metric: str = "throughput", 
                                      output_file: Optional[str] = None,
                                      title: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 6),
                                      batch_sizes: List[int] = None) -> None:
        """
        Create a chart comparing hardware platforms for a specific model.
        
        Args:
            model_name: Name of the model
            metric: Metric to compare ("throughput", "latency", "memory")
            output_file: Output file path (PNG, PDF, SVG, etc.)
            title: Custom title for the chart
            figsize: Figure size as (width, height) in inches
            batch_sizes: List of batch sizes to include (or None for all)
        """
        # Get comparison data
        df = self.api.get_performance_comparison(model_name, metric)
        
        if df.empty:
            logger.error(f"No data found for model: {model_name}")
            return
        
        # Filter by batch sizes if specified
        if batch_sizes:
            df = df[df['batch_size'].isin(batch_sizes)]
            
            if df.empty:
                logger.error(f"No data found for model: {model_name} with batch sizes: {batch_sizes}")
                return
        
        # Configure plot
        plt.figure(figsize=figsize)
        
        # Determine metric column and labels
        metric_col = "metric_value"
        if metric.lower() == "throughput":
            ylabel = "Throughput (items per second)"
            chart_title = title or f"Throughput Comparison for {model_name}"
            # Sort by throughput (descending)
            df = df.sort_values(metric_col, ascending=False)
        elif metric.lower() == "latency":
            ylabel = "Latency (ms)"
            chart_title = title or f"Latency Comparison for {model_name}"
            # Sort by latency (ascending)
            df = df.sort_values(metric_col, ascending=True)
        elif metric.lower() == "memory":
            ylabel = "Memory Usage (MB)"
            chart_title = title or f"Memory Usage Comparison for {model_name}"
            # Sort by memory (ascending)
            df = df.sort_values(metric_col, ascending=True)
        else:
            ylabel = metric.capitalize()
            chart_title = title or f"{metric.capitalize()} Comparison for {model_name}"
        
        # Create grouped bar chart
        ax = sns.barplot(
            x="hardware_type",
            y=metric_col,
            hue="batch_size",
            data=df,
            palette="viridis"
        )
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=8)
        
        # Set chart title and labels
        plt.title(chart_title)
        plt.xlabel("Hardware Platform")
        plt.ylabel(ylabel)
        
        # Adjust xtick rotation for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add simulation indicators if available
        if 'is_simulated' in df.columns:
            # Find simulated results
            simulated_indices = df[df['is_simulated'] == True].index
            
            # Extract bar patches
            bars = ax.patches
            
            # Add hatching to simulated result bars
            for i in simulated_indices:
                # Find all bars associated with this index (could be multiple due to hue)
                for j, bar in enumerate(bars):
                    if j // len(df['batch_size'].unique()) == i:
                        bar.set_hatch('///')
            
            # Add a legend entry for simulated results
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='gray', hatch='///', label='Simulated')]
            second_legend = plt.legend(handles=legend_elements, loc='upper right')
            plt.gca().add_artist(second_legend)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved hardware comparison chart to: {output_file}")
        else:
            plt.show()
    
    def create_model_comparison_chart(self, hardware_type: str, model_names: List[str] = None,
                                    metric: str = "throughput", 
                                    output_file: Optional[str] = None,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 6),
                                    batch_size: int = 1) -> None:
        """
        Create a chart comparing multiple models on a specific hardware platform.
        
        Args:
            hardware_type: Type of hardware platform
            model_names: List of model names to include (or None for all)
            metric: Metric to compare ("throughput", "latency", "memory")
            output_file: Output file path (PNG, PDF, SVG, etc.)
            title: Custom title for the chart
            figsize: Figure size as (width, height) in inches
            batch_size: Batch size to use for comparison
        """
        # Query the database for model comparison data
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
                pr.is_simulated,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, pr.batch_size, pr.precision
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                hp.hardware_type = :hardware_type AND
                pr.batch_size = :batch_size
        """
        
        params = {
            "hardware_type": hardware_type,
            "batch_size": batch_size
        }
        
        if model_names:
            model_list = ", ".join([f"'{model}'" for model in model_names])
            sql += f" AND m.model_name IN ({model_list})"
        
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
            memory_peak_mb,
            is_simulated
        FROM
            latest_results
        WHERE
            rn = 1
        """
        
        df = self.api.query(sql, params)
        
        if df.empty:
            logger.error(f"No data found for hardware type: {hardware_type}")
            return
        
        # Configure plot
        plt.figure(figsize=figsize)
        
        # Determine metric column and labels
        if metric.lower() == "throughput":
            metric_col = "throughput_items_per_second"
            ylabel = "Throughput (items per second)"
            chart_title = title or f"Model Throughput Comparison on {hardware_type}"
            # Sort by throughput (descending)
            df = df.sort_values(metric_col, ascending=False)
        elif metric.lower() == "latency":
            metric_col = "average_latency_ms"
            ylabel = "Latency (ms)"
            chart_title = title or f"Model Latency Comparison on {hardware_type}"
            # Sort by latency (ascending)
            df = df.sort_values(metric_col, ascending=True)
        elif metric.lower() == "memory":
            metric_col = "memory_peak_mb"
            ylabel = "Memory Usage (MB)"
            chart_title = title or f"Model Memory Usage Comparison on {hardware_type}"
            # Sort by memory (ascending)
            df = df.sort_values(metric_col, ascending=True)
        else:
            logger.error(f"Unsupported metric: {metric}")
            return
        
        # Limit to top N models if there are too many
        if len(df) > 20 and model_names is None:
            if metric.lower() == "throughput":
                df = df.nlargest(20, metric_col)
            else:
                df = df.nsmallest(20, metric_col)
        
        # Create color map based on model family
        if 'model_family' in df.columns and df['model_family'].notna().all():
            # Group by model family
            families = df['model_family'].unique()
            family_cmap = sns.color_palette("viridis", len(families))
            family_colors = {family: family_cmap[i] for i, family in enumerate(families)}
            
            # Map colors to each model
            colors = [family_colors[family] for family in df['model_family']]
        else:
            # Use default color palette
            colors = sns.color_palette("viridis", len(df))
        
        # Create bar chart
        ax = sns.barplot(
            x="model_name",
            y=metric_col,
            data=df,
            palette=colors
        )
        
        # Add value labels on bars
        for i, v in enumerate(df[metric_col]):
            ax.text(
                i, 
                v * 1.01,  # Position slightly above the bar
                f"{v:.1f}", 
                ha='center',
                fontsize=8
            )
        
        # Set chart title and labels
        plt.title(chart_title)
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        
        # Adjust xtick rotation for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add simulation indicators if available
        if 'is_simulated' in df.columns and df['is_simulated'].any():
            # Find simulated results
            simulated_indices = df[df['is_simulated'] == True].index
            
            # Add hatching to simulated result bars
            for i in simulated_indices:
                ax.patches[i].set_hatch('///')
            
            # Add a legend entry for simulated results
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='gray', hatch='///', label='Simulated')]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison chart to: {output_file}")
        else:
            plt.show()
    
    def create_batch_size_comparison_chart(self, model_name: str, hardware_type: str,
                                        metric: str = "throughput", 
                                        output_file: Optional[str] = None,
                                        title: Optional[str] = None,
                                        figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create a chart comparing performance across different batch sizes.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware platform
            metric: Metric to compare ("throughput", "latency", "memory")
            output_file: Output file path (PNG, PDF, SVG, etc.)
            title: Custom title for the chart
            figsize: Figure size as (width, height) in inches
        """
        # Query the database for batch size comparison data
        sql = """
        WITH latest_results AS (
            SELECT 
                m.model_name,
                hp.hardware_type,
                pr.batch_size,
                pr.precision,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                pr.is_simulated,
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
            memory_peak_mb,
            is_simulated
        FROM
            latest_results
        WHERE
            rn = 1
        ORDER BY
            batch_size
        """
        
        df = self.api.query(sql, {
            "model_name": model_name,
            "hardware_type": hardware_type
        })
        
        if df.empty:
            logger.error(f"No data found for model: {model_name} on hardware: {hardware_type}")
            return
        
        # Configure plot
        plt.figure(figsize=figsize)
        
        # Determine metric column and labels
        if metric.lower() == "throughput":
            metric_col = "throughput_items_per_second"
            ylabel = "Throughput (items per second)"
            chart_title = title or f"Throughput vs Batch Size for {model_name} on {hardware_type}"
        elif metric.lower() == "latency":
            metric_col = "average_latency_ms"
            ylabel = "Latency (ms)"
            chart_title = title or f"Latency vs Batch Size for {model_name} on {hardware_type}"
        elif metric.lower() == "memory":
            metric_col = "memory_peak_mb"
            ylabel = "Memory Usage (MB)"
            chart_title = title or f"Memory Usage vs Batch Size for {model_name} on {hardware_type}"
        else:
            logger.error(f"Unsupported metric: {metric}")
            return
        
        # Create line chart grouped by precision
        ax = sns.lineplot(
            x="batch_size",
            y=metric_col,
            hue="precision",
            style="precision",
            markers=True,
            data=df
        )
        
        # Add data labels
        for precision in df['precision'].unique():
            precision_df = df[df['precision'] == precision]
            for _, row in precision_df.iterrows():
                ax.text(
                    row['batch_size'],
                    row[metric_col] * 1.02,  # Position slightly above the line
                    f"{row[metric_col]:.1f}",
                    ha='center',
                    fontsize=8
                )
        
        # Set chart title and labels
        plt.title(chart_title)
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add simulation indicators if available
        if 'is_simulated' in df.columns and df['is_simulated'].any():
            # Find simulated results
            simulated_points = df[df['is_simulated'] == True]
            
            # Mark simulated results with different marker
            plt.scatter(
                simulated_points['batch_size'],
                simulated_points[metric_col],
                marker='x',
                s=100,
                color='red',
                zorder=10,
                label='Simulated'
            )
            
            # Add the legend back
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved batch size comparison chart to: {output_file}")
        else:
            plt.show()
    
    def create_hardware_heatmap(self, model_names: List[str] = None,
                             hardware_types: List[str] = None,
                             metric: str = "throughput",
                             output_file: Optional[str] = None,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 10),
                             batch_size: int = 1) -> None:
        """
        Create a heatmap showing performance across models and hardware platforms.
        
        Args:
            model_names: List of model names to include (or None for all)
            hardware_types: List of hardware types to include (or None for all)
            metric: Metric to visualize ("throughput", "latency", "memory")
            output_file: Output file path (PNG, PDF, SVG, etc.)
            title: Custom title for the chart
            figsize: Figure size as (width, height) in inches
            batch_size: Batch size to use for comparison
        """
        # Determine metric column
        if metric.lower() == "throughput":
            metric_col = "throughput_items_per_second"
            chart_title = title or f"Throughput Heatmap (Batch Size: {batch_size})"
            # Higher values are better for throughput (use reverse colormap)
            cmap = "viridis"
        elif metric.lower() == "latency":
            metric_col = "average_latency_ms"
            chart_title = title or f"Latency Heatmap (Batch Size: {batch_size})"
            # Lower values are better for latency
            cmap = "viridis_r"
        elif metric.lower() == "memory":
            metric_col = "memory_peak_mb"
            chart_title = title or f"Memory Usage Heatmap (Batch Size: {batch_size})"
            # Lower values are better for memory
            cmap = "viridis_r"
        else:
            logger.error(f"Unsupported metric: {metric}")
            return
        
        # Build SQL query
        sql = """
        WITH latest_results AS (
            SELECT 
                m.model_name,
                m.model_family,
                hp.hardware_type,
                pr.batch_size,
                pr.average_latency_ms,
                pr.throughput_items_per_second,
                pr.memory_peak_mb,
                pr.is_simulated,
                ROW_NUMBER() OVER(PARTITION BY m.model_id, hp.hardware_id, pr.batch_size
                ORDER BY pr.created_at DESC) as rn
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                pr.batch_size = :batch_size
        """
        
        params = {"batch_size": batch_size}
        
        if model_names:
            model_list = ", ".join([f"'{model}'" for model in model_names])
            sql += f" AND m.model_name IN ({model_list})"
        
        if hardware_types:
            hw_list = ", ".join([f"'{hw}'" for hw in hardware_types])
            sql += f" AND hp.hardware_type IN ({hw_list})"
        
        sql += """
        )
        SELECT
            model_name,
            hardware_type,
            average_latency_ms,
            throughput_items_per_second,
            memory_peak_mb,
            is_simulated
        FROM
            latest_results
        WHERE
            rn = 1
        """
        
        df = self.api.query(sql, params)
        
        if df.empty:
            logger.error("No data found for the specified parameters")
            return
        
        # Pivot the data for the heatmap
        pivot_df = df.pivot(index="model_name", columns="hardware_type", values=metric_col)
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={"label": metric.capitalize()}
        )
        
        # Set chart title and labels
        plt.title(chart_title)
        plt.xlabel("Hardware Platform")
        plt.ylabel("Model")
        
        # Adjust xtick rotation for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display the chart
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved hardware heatmap to: {output_file}")
        else:
            plt.show()
    
    def create_dashboard(self, output_dir: str, metrics: List[str] = None,
                      model_names: List[str] = None,
                      hardware_types: List[str] = None,
                      batch_sizes: List[int] = None) -> None:
        """
        Create a dashboard with multiple charts.
        
        Args:
            output_dir: Directory to save dashboard charts
            metrics: List of metrics to include (or None for all)
            model_names: List of model names to include (or None for all)
            hardware_types: List of hardware types to include (or None for all)
            batch_sizes: List of batch sizes to include (or None for [1, 4, 16])
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default metrics if not provided
        if not metrics:
            metrics = ["throughput", "latency", "memory"]
        
        # Default batch sizes if not provided
        if not batch_sizes:
            batch_sizes = [1, 4, 16]
        
        # Determine models and hardware types to include
        sql = """
        SELECT DISTINCT m.model_name
        FROM performance_results pr
        JOIN models m ON pr.model_id = m.model_id
        ORDER BY m.model_name
        """
        all_models_df = self.api.query(sql)
        
        if model_names:
            models_to_use = model_names
        else:
            models_to_use = all_models_df['model_name'].tolist()
        
        sql = """
        SELECT DISTINCT hp.hardware_type
        FROM performance_results pr
        JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
        ORDER BY hp.hardware_type
        """
        all_hardware_df = self.api.query(sql)
        
        if hardware_types:
            hardware_to_use = hardware_types
        else:
            hardware_to_use = all_hardware_df['hardware_type'].tolist()
        
        # Create index.html file
        with open(os.path.join(output_dir, "index.html"), "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Benchmark Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                    .container { width: 90%; margin: 0 auto; }
                    h1, h2, h3 { color: #333; }
                    .chart-container { margin-bottom: 30px; }
                    .chart-container img { max-width: 100%; border: 1px solid #ddd; }
                    .navigation { background: #f4f4f4; padding: 10px; position: sticky; top: 0; }
                    .navigation a { margin-right: 15px; color: #0066cc; }
                </style>
            </head>
            <body>
                <div class="navigation">
                    <a href="#overview">Overview</a>
                    <a href="#models">Model Comparisons</a>
                    <a href="#hardware">Hardware Comparisons</a>
                    <a href="#batch">Batch Size Comparisons</a>
                    <a href="#heatmaps">Heatmaps</a>
                </div>
                <div class="container">
                    <h1>Benchmark Dashboard</h1>
                    <p>Generated on """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                    
                    <h2 id="overview">Overview</h2>
                    <div class="chart-container">
                        <h3>Hardware Performance Overview</h3>
                        <img src="overview_hardware.png" alt="Hardware Performance Overview">
                    </div>
                    
                    <h2 id="models">Model Comparisons</h2>
            """)
            
            # Create overview chart (hardware performance heatmap)
            self.create_hardware_heatmap(
                model_names=models_to_use[:10],  # Limit to avoid too large a chart
                hardware_types=hardware_to_use,
                metric="throughput",
                output_file=os.path.join(output_dir, "overview_hardware.png"),
                title="Hardware Performance Overview (Throughput)",
                batch_size=1
            )
            
            # Create model comparison charts for each hardware type
            for hw_type in hardware_to_use:
                f.write(f"""
                <div class="chart-container">
                    <h3>Model Comparison on {hw_type}</h3>
                    <img src="model_comparison_{hw_type}.png" alt="Model Comparison on {hw_type}">
                </div>
                """)
                
                self.create_model_comparison_chart(
                    hardware_type=hw_type,
                    model_names=models_to_use,
                    metric="throughput",
                    output_file=os.path.join(output_dir, f"model_comparison_{hw_type}.png"),
                    title=f"Model Throughput Comparison on {hw_type}",
                    batch_size=1
                )
            
            # Hardware comparison section
            f.write("""
                <h2 id="hardware">Hardware Comparisons</h2>
            """)
            
            # Create hardware comparison charts for each model
            for model in models_to_use:
                f.write(f"""
                <div class="chart-container">
                    <h3>Hardware Comparison for {model}</h3>
                    <img src="hardware_comparison_{model.replace('/', '_')}.png" alt="Hardware Comparison for {model}">
                </div>
                """)
                
                self.create_hardware_comparison_chart(
                    model_name=model,
                    metric="throughput",
                    output_file=os.path.join(output_dir, f"hardware_comparison_{model.replace('/', '_')}.png"),
                    title=f"Throughput Comparison for {model}",
                    batch_sizes=batch_sizes
                )
            
            # Batch size comparison section
            f.write("""
                <h2 id="batch">Batch Size Comparisons</h2>
            """)
            
            # Create batch size comparison charts for selected models and hardware
            for model in models_to_use[:5]:  # Limit to avoid too many charts
                for hw_type in hardware_to_use[:3]:  # Limit to avoid too many charts
                    f.write(f"""
                    <div class="chart-container">
                        <h3>Batch Size Comparison for {model} on {hw_type}</h3>
                        <img src="batch_comparison_{model.replace('/', '_')}_{hw_type}.png" alt="Batch Size Comparison for {model} on {hw_type}">
                    </div>
                    """)
                    
                    self.create_batch_size_comparison_chart(
                        model_name=model,
                        hardware_type=hw_type,
                        metric="throughput",
                        output_file=os.path.join(output_dir, f"batch_comparison_{model.replace('/', '_')}_{hw_type}.png"),
                        title=f"Throughput vs Batch Size for {model} on {hw_type}"
                    )
            
            # Heatmap section
            f.write("""
                <h2 id="heatmaps">Heatmaps</h2>
            """)
            
            # Create heatmaps for each metric
            for metric in metrics:
                f.write(f"""
                <div class="chart-container">
                    <h3>{metric.capitalize()} Heatmap</h3>
                    <img src="heatmap_{metric}.png" alt="{metric.capitalize()} Heatmap">
                </div>
                """)
                
                self.create_hardware_heatmap(
                    model_names=models_to_use,
                    hardware_types=hardware_to_use,
                    metric=metric,
                    output_file=os.path.join(output_dir, f"heatmap_{metric}.png"),
                    title=f"{metric.capitalize()} Heatmap (Batch Size: 1)",
                    batch_size=1
                )
            
            # Close HTML file
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Created benchmark dashboard in directory: {output_dir}")
        logger.info(f"Open {os.path.join(output_dir, 'index.html')} in a web browser to view the dashboard")

def main():
    """Command-line interface for the benchmark visualizer."""
    parser = argparse.ArgumentParser(description="Benchmark Visualization Tool")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--chart-type", choices=[
                        'hardware-comparison', 'model-comparison', 
                        'batch-size-comparison', 'heatmap', 'dashboard'],
                       help="Type of chart to create")
    parser.add_argument("--model",
                       help="Model name for hardware or batch size comparison")
    parser.add_argument("--hardware",
                       help="Hardware type for model or batch size comparison")
    parser.add_argument("--models", 
                       help="Comma-separated list of models to include in the chart")
    parser.add_argument("--hardware-types",
                       help="Comma-separated list of hardware types to include in the chart")
    parser.add_argument("--metric", choices=['throughput', 'latency', 'memory'], default='throughput',
                       help="Metric to visualize")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size to use for comparison (for model comparison and heatmap)")
    parser.add_argument("--batch-sizes",
                       help="Comma-separated list of batch sizes to include in the chart")
    parser.add_argument("--output",
                       help="Output file or directory for the chart")
    parser.add_argument("--title",
                       help="Custom title for the chart")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(db_path=args.db_path, debug=args.debug)
    
    # Parse batch sizes if provided
    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    
    # Parse models if provided
    models = None
    if args.models:
        models = args.models.split(',')
    
    # Parse hardware types if provided
    hardware_types = None
    if args.hardware_types:
        hardware_types = args.hardware_types.split(',')
    
    # Create requested chart
    if args.chart_type == 'hardware-comparison':
        if not args.model:
            logger.error("--model is required for hardware comparison chart")
            sys.exit(1)
        
        visualizer.create_hardware_comparison_chart(
            model_name=args.model,
            metric=args.metric,
            output_file=args.output,
            title=args.title,
            batch_sizes=batch_sizes
        )
    
    elif args.chart_type == 'model-comparison':
        if not args.hardware:
            logger.error("--hardware is required for model comparison chart")
            sys.exit(1)
        
        visualizer.create_model_comparison_chart(
            hardware_type=args.hardware,
            model_names=models,
            metric=args.metric,
            output_file=args.output,
            title=args.title,
            batch_size=args.batch_size
        )
    
    elif args.chart_type == 'batch-size-comparison':
        if not args.model or not args.hardware:
            logger.error("--model and --hardware are required for batch size comparison chart")
            sys.exit(1)
        
        visualizer.create_batch_size_comparison_chart(
            model_name=args.model,
            hardware_type=args.hardware,
            metric=args.metric,
            output_file=args.output,
            title=args.title
        )
    
    elif args.chart_type == 'heatmap':
        visualizer.create_hardware_heatmap(
            model_names=models,
            hardware_types=hardware_types,
            metric=args.metric,
            output_file=args.output,
            title=args.title,
            batch_size=args.batch_size
        )
    
    elif args.chart_type == 'dashboard':
        if not args.output:
            logger.error("--output directory is required for dashboard")
            sys.exit(1)
        
        visualizer.create_dashboard(
            output_dir=args.output,
            metrics=[args.metric] if args.metric else ['throughput', 'latency', 'memory'],
            model_names=models,
            hardware_types=hardware_types,
            batch_sizes=batch_sizes
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()