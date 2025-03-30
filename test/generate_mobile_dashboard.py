#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobile Performance Dashboard Generator

This script generates an interactive HTML dashboard for visualizing mobile benchmark results.
It processes analysis data to create charts and tables comparing performance across
different mobile platforms, devices, and models.

Usage:
    python generate_mobile_dashboard.py --data-file DATA_FILE [--output OUTPUT]
    [--db-path DB_PATH] [--title TITLE] [--theme THEME] [--days DAYS] [--verbose]

Examples:
    # Generate dashboard from analysis data
    python generate_mobile_dashboard.py --data-file analysis_results.json
        --output mobile_dashboard.html
    
    # Include historical data from database with custom title
    python generate_mobile_dashboard.py --data-file analysis_results.json
        --db-path benchmark_results.duckdb --title "Mobile Edge Performance Dashboard"
    
    # Use dark theme and include 30 days of historical data
    python generate_mobile_dashboard.py --data-file analysis_results.json
        --theme dark --days 30

Date: April 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

# Try importing dashboard dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
except ImportError as e:
    logger.error(f"Dashboard dependency import error: {e}")
    logger.error("Install required packages: pip install plotly pandas kaleido")
    sys.exit(1)


class MobileDashboardGenerator:
    """
    Generates an interactive HTML dashboard for visualizing mobile benchmark results.
    
    This class processes analysis data to create charts and tables comparing
    performance across different mobile platforms, devices, and models.
    """
    
    # Chart color schemes
    COLOR_SCHEMES = {
        "light": {
            "background": "#ffffff",
            "text": "#333333",
            "grid": "#eeeeee",
            "platforms": {
                "android": "#3ddc84",
                "ios": "#007aff"
            },
            "metrics": {
                "throughput": "#4caf50",
                "latency": "#2196f3",
                "memory": "#ff9800",
                "battery": "#f44336"
            }
        },
        "dark": {
            "background": "#1e1e1e",
            "text": "#ffffff",
            "grid": "#333333",
            "platforms": {
                "android": "#3ddc84",
                "ios": "#007aff"
            },
            "metrics": {
                "throughput": "#4caf50",
                "latency": "#2196f3",
                "memory": "#ff9800",
                "battery": "#f44336"
            }
        }
    }
    
    def __init__(self, 
                 data_file: str,
                 output_path: str = "mobile_dashboard.html",
                 db_path: Optional[str] = None,
                 title: str = "Mobile Edge Performance Dashboard",
                 theme: str = "light",
                 days: int = 14,
                 verbose: bool = False):
        """
        Initialize the mobile dashboard generator.
        
        Args:
            data_file: Path to JSON data file with benchmark results
            output_path: Path to output HTML dashboard
            db_path: Optional path to DuckDB database for historical data
            title: Dashboard title
            theme: Dashboard theme (light or dark)
            days: Number of days to include in historical trends
            verbose: Enable verbose logging
        """
        self.data_file = data_file
        self.output_path = output_path
        self.db_path = db_path
        self.title = title
        self.theme = theme.lower()
        self.days_lookback = days
        self.verbose = verbose
        
        # Set logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize variables
        self.data = {}
        self.historical_data = {}
        self.db_api = None
        self.figures = []
        self.colors = self.COLOR_SCHEMES.get(self.theme, self.COLOR_SCHEMES["light"])
        
        # Calculate the date range for historical data
        self.end_date = datetime.datetime.now()
        self.start_date = self.end_date - datetime.timedelta(days=self.days_lookback)
    
    def load_data(self) -> bool:
        """
        Load benchmark data from JSON file.
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.data_file):
                logger.error(f"Data file not found: {self.data_file}")
                return False
            
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
            
            logger.info(f"Loaded data from: {self.data_file}")
            
            # Basic validation
            if not isinstance(self.data, dict):
                logger.error("Invalid data format: expected JSON object")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def connect_to_db(self) -> bool:
        """
        Connect to DuckDB database for historical data.
        
        Returns:
            Success status
        """
        # Skip if db_path not provided
        if not self.db_path:
            logger.info("No database path provided, skipping historical data")
            return False
        
        try:
            if not os.path.exists(self.db_path):
                logger.error(f"Database file not found: {self.db_path}")
                return False
            
            self.db_api = BenchmarkDBAPI(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def load_historical_data(self) -> bool:
        """
        Load historical benchmark data from database.
        
        Returns:
            Success status
        """
        if not self.db_api:
            if not self.connect_to_db():
                return False
        
        try:
            # Format date strings for database query
            start_date_str = self.start_date.strftime("%Y-%m-%d")
            end_date_str = self.end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Loading historical data from {start_date_str} to {end_date_str}")
            
            # Get data for trend charts
            query = f"""
                SELECT
                    r.id AS run_id,
                    r.model_name,
                    r.timestamp,
                    r.device_info->>'platform' AS platform,
                    r.device_info->>'model' AS device_model,
                    c.configuration->>'batch_size' AS batch_size,
                    res.throughput_items_per_second AS throughput,
                    res.latency_ms->>'mean' AS latency_mean,
                    res.memory_metrics->>'peak_mb' AS memory_peak,
                    res.battery_metrics->>'impact_percentage' AS battery_impact
                FROM
                    benchmark_runs r
                JOIN
                    benchmark_configurations c ON r.id = c.run_id
                JOIN
                    benchmark_results res ON c.id = res.config_id
                WHERE
                    r.timestamp >= '{start_date_str}'
                    AND r.timestamp <= '{end_date_str}'
                ORDER BY
                    r.timestamp ASC
            """
            
            results = self.db_api.query(query)
            
            # Convert to DataFrame for easier processing
            self.historical_data = pd.DataFrame(results)
            
            # Convert types
            if not self.historical_data.empty:
                self.historical_data["batch_size"] = self.historical_data["batch_size"].astype(int)
                self.historical_data["throughput"] = self.historical_data["throughput"].astype(float)
                self.historical_data["latency_mean"] = self.historical_data["latency_mean"].astype(float)
                self.historical_data["memory_peak"] = self.historical_data["memory_peak"].astype(float)
                self.historical_data["battery_impact"] = self.historical_data["battery_impact"].astype(float)
                self.historical_data["timestamp"] = pd.to_datetime(self.historical_data["timestamp"])
            
            logger.info(f"Loaded {len(self.historical_data)} historical data points")
            return True
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def _create_platform_comparison_chart(self) -> go.Figure:
        """
        Create a chart comparing performance across platforms.
        
        Returns:
            Plotly figure
        """
        # Extract platform data
        platforms_data = self.data.get("platforms", {})
        platforms = list(platforms_data.keys())
        
        # Collect model names across all platforms
        all_models: Set[str] = set()
        for platform, platform_data in platforms_data.items():
            models = platform_data.get("models", {})
            all_models.update(models.keys())
        
        # Prepare data for chart
        models_list = sorted(all_models)
        throughputs = {platform: [] for platform in platforms}
        latencies = {platform: [] for platform in platforms}
        
        for model in models_list:
            for platform in platforms:
                platform_data = platforms_data.get(platform, {})
                models = platform_data.get("models", {})
                model_data = models.get(model, {})
                
                # Get batch size 1 data (default)
                batch_data = model_data.get("batch_sizes", {}).get("1", {})
                
                throughput = batch_data.get("throughput", 0)
                latency = batch_data.get("latency", 0)
                
                throughputs[platform].append(throughput)
                latencies[platform].append(latency)
        
        # Create figure with 2 subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Throughput (items/s)", "Latency (ms)"),
            shared_yaxes=True
        )
        
        # Add throughput bars
        for i, platform in enumerate(platforms):
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=throughputs[platform],
                    name=f"{platform.capitalize()}",
                    marker_color=self.colors["platforms"].get(platform, f"hsl({(i*120) % 360}, 70%, 50%)"),
                    legendgroup=platform
                ),
                row=1, col=1
            )
        
        # Add latency bars
        for i, platform in enumerate(platforms):
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=latencies[platform],
                    name=f"{platform.capitalize()}",
                    marker_color=self.colors["platforms"].get(platform, f"hsl({(i*120) % 360}, 70%, 50%)"),
                    showlegend=False,
                    legendgroup=platform
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Cross-Platform Performance Comparison (Batch Size 1)",
            barmode="group",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=45,
            gridcolor=self.colors["grid"]
        )
        
        fig.update_yaxes(
            gridcolor=self.colors["grid"]
        )
        
        return fig
    
    def _create_model_comparison_chart(self) -> go.Figure:
        """
        Create a chart comparing performance across models.
        
        Returns:
            Plotly figure
        """
        # Extract model data
        models_data = self.data.get("models", {})
        
        # Get models with data for both platforms
        common_models = []
        for model_name, model_data in models_data.items():
            platforms = model_data.get("platforms", {})
            if "android" in platforms and "ios" in platforms:
                common_models.append(model_name)
        
        # Prepare data for chart
        batch_sizes = ["1", "4", "8"]  # Focus on these common batch sizes
        android_throughputs = {batch: [] for batch in batch_sizes}
        ios_throughputs = {batch: [] for batch in batch_sizes}
        
        common_models_with_data = []
        
        for model in common_models:
            model_data = models_data.get(model, {})
            platforms = model_data.get("platforms", {})
            
            android_data = platforms.get("android", {}).get("batch_sizes", {})
            ios_data = platforms.get("ios", {}).get("batch_sizes", {})
            
            # Check if model has data for at least one batch size on both platforms
            has_data = False
            for batch in batch_sizes:
                if batch in android_data and batch in ios_data:
                    has_data = True
                    break
            
            if not has_data:
                continue
            
            common_models_with_data.append(model)
            
            for batch in batch_sizes:
                android_throughput = android_data.get(batch, {}).get("throughput", 0)
                ios_throughput = ios_data.get(batch, {}).get("throughput", 0)
                
                android_throughputs[batch].append(android_throughput)
                ios_throughputs[batch].append(ios_throughput)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for Android
        for batch in batch_sizes:
            if any(android_throughputs[batch]):  # Only add if there's data
                fig.add_trace(
                    go.Bar(
                        x=common_models_with_data,
                        y=android_throughputs[batch],
                        name=f"Android (Batch {batch})",
                        marker_color=self.colors["platforms"]["android"],
                        opacity=0.6 + 0.2 * batch_sizes.index(batch)
                    )
                )
        
        # Add traces for iOS
        for batch in batch_sizes:
            if any(ios_throughputs[batch]):  # Only add if there's data
                fig.add_trace(
                    go.Bar(
                        x=common_models_with_data,
                        y=ios_throughputs[batch],
                        name=f"iOS (Batch {batch})",
                        marker_color=self.colors["platforms"]["ios"],
                        opacity=0.6 + 0.2 * batch_sizes.index(batch)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Model Performance by Platform and Batch Size (Throughput)",
            barmode="group",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                title="Throughput (items/s)"
            )
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=45,
            gridcolor=self.colors["grid"]
        )
        
        fig.update_yaxes(
            gridcolor=self.colors["grid"]
        )
        
        return fig
    
    def _create_battery_memory_chart(self) -> go.Figure:
        """
        Create a chart comparing battery and memory impact across platforms.
        
        Returns:
            Plotly figure
        """
        # Extract platform data
        platforms_data = self.data.get("platforms", {})
        platforms = list(platforms_data.keys())
        
        # Collect model names across all platforms
        all_models: Set[str] = set()
        for platform, platform_data in platforms_data.items():
            models = platform_data.get("models", {})
            all_models.update(models.keys())
        
        # Prepare data for chart
        models_list = sorted(all_models)
        memory_usage = {platform: [] for platform in platforms}
        battery_impact = {platform: [] for platform in platforms}
        
        for model in models_list:
            for platform in platforms:
                platform_data = platforms_data.get(platform, {})
                models = platform_data.get("models", {})
                model_data = models.get(model, {})
                
                # Get batch size 1 data (default)
                batch_data = model_data.get("batch_sizes", {}).get("1", {})
                
                memory = batch_data.get("memory_mb", 0)
                battery = batch_data.get("battery_impact", 0)
                
                memory_usage[platform].append(memory)
                battery_impact[platform].append(battery)
        
        # Create figure with 2 subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Memory Usage (MB)", "Battery Impact (%)"),
            shared_yaxes=True
        )
        
        # Add memory bars
        for i, platform in enumerate(platforms):
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=memory_usage[platform],
                    name=f"{platform.capitalize()}",
                    marker_color=self.colors["platforms"].get(platform, f"hsl({(i*120) % 360}, 70%, 50%)"),
                    legendgroup=platform
                ),
                row=1, col=1
            )
        
        # Add battery bars
        for i, platform in enumerate(platforms):
            fig.add_trace(
                go.Bar(
                    x=models_list,
                    y=battery_impact[platform],
                    name=f"{platform.capitalize()}",
                    marker_color=self.colors["platforms"].get(platform, f"hsl({(i*120) % 360}, 70%, 50%)"),
                    showlegend=False,
                    legendgroup=platform
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Memory and Battery Impact Comparison (Batch Size 1)",
            barmode="group",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=45,
            gridcolor=self.colors["grid"]
        )
        
        fig.update_yaxes(
            gridcolor=self.colors["grid"]
        )
        
        return fig
    
    def _create_historical_trends_chart(self) -> Optional[go.Figure]:
        """
        Create a chart showing historical performance trends.
        
        Returns:
            Plotly figure or None if no historical data
        """
        if self.historical_data is None or self.historical_data.empty:
            logger.warning("No historical data available for trends chart")
            return None
        
        # Select a few representative models for trend chart
        key_models = [
            "bert-base-uncased",
            "mobilenet-v2",
            "clip-vit-base-patch32",
            "whisper-tiny"
        ]
        
        # Filter data to batch size 1 for simplicity
        df = self.historical_data[self.historical_data["batch_size"] == 1]
        
        # Only include models in our key models list
        df = df[df["model_name"].isin(key_models)]
        
        # Group by date, platform, and model_name and calculate average throughput
        df["date"] = df["timestamp"].dt.date
        trend_data = df.groupby(["date", "platform", "model_name"])["throughput"].mean().reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each model and platform combination
        for model in key_models:
            for platform in ["android", "ios"]:
                model_platform_data = trend_data[(trend_data["model_name"] == model) & 
                                               (trend_data["platform"] == platform)]
                
                if not model_platform_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=model_platform_data["date"],
                            y=model_platform_data["throughput"],
                            mode="lines+markers",
                            name=f"{model} ({platform.capitalize()})",
                            line=dict(
                                color=self.colors["platforms"][platform],
                                width=2,
                                dash="solid" if platform == "android" else "dot"
                            ),
                            marker=dict(
                                size=6,
                                symbol="circle" if platform == "android" else "square"
                            )
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title="Performance Trends Over Time (Throughput, Batch Size 1)",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Date"
            ),
            yaxis=dict(
                title="Throughput (items/s)"
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.colors["grid"]
        )
        
        fig.update_yaxes(
            gridcolor=self.colors["grid"]
        )
        
        return fig
    
    def _create_batch_scaling_chart(self) -> go.Figure:
        """
        Create a chart showing performance scaling with batch size.
        
        Returns:
            Plotly figure
        """
        # Extract model data
        models_data = self.data.get("models", {})
        
        # Filter to models that have data for multiple batch sizes
        scaling_models = []
        for model_name, model_data in models_data.items():
            for platform, platform_data in model_data.get("platforms", {}).items():
                batch_sizes = platform_data.get("batch_sizes", {})
                # Check if model has at least 2 batch sizes
                if len(batch_sizes) >= 2:
                    scaling_models.append(model_name)
                    break
        
        # Prepare data for chart
        batch_sizes = ["1", "2", "4", "8", "16"]
        platforms = ["android", "ios"]
        
        # Create figure with a subplot for each model
        n_models = min(6, len(scaling_models))  # Limit to 6 models
        
        if n_models == 0:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No models with multiple batch sizes found",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=self.colors["text"])
            )
            return fig
        
        # Calculate subplot grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        # Create subplot figure
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[model[:20] for model in scaling_models[:n_models]],
            shared_yaxes=True,
            vertical_spacing=0.1
        )
        
        # Add traces for each model
        for i, model in enumerate(scaling_models[:n_models]):
            model_data = models_data.get(model, {})
            platforms_data = model_data.get("platforms", {})
            
            # Calculate row and column for this subplot
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            for platform in platforms:
                platform_data = platforms_data.get(platform, {})
                batch_data = platform_data.get("batch_sizes", {})
                
                x_values = []
                y_values = []
                
                for batch in batch_sizes:
                    if batch in batch_data:
                        x_values.append(int(batch))
                        y_values.append(batch_data[batch].get("throughput", 0))
                
                if x_values:
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode="lines+markers",
                            name=f"{platform.capitalize()} ({model})",
                            legendgroup=platform,
                            showlegend=i==0,  # Only show legend for first model
                            line=dict(
                                color=self.colors["platforms"].get(platform, "blue")
                            ),
                            marker=dict(
                                size=8,
                                symbol="circle" if platform == "android" else "square"
                            )
                        ),
                        row=row, col=col
                    )
                    
                    # Add efficiency line (throughput/batch_size)
                    efficiency_values = [y/x for x, y in zip(x_values, y_values)]
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=efficiency_values,
                            mode="lines",
                            name=f"{platform.capitalize()} Efficiency ({model})",
                            legendgroup=f"{platform}_eff",
                            showlegend=i==0,  # Only show legend for first model
                            line=dict(
                                color=self.colors["platforms"].get(platform, "blue"),
                                dash="dot"
                            )
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title="Performance Scaling with Batch Size",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor=self.colors["background"],
            font=dict(color=self.colors["text"]),
            height=200 * rows,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update all xaxes
        fig.update_xaxes(
            title="Batch Size",
            type="log",
            gridcolor=self.colors["grid"]
        )
        
        # Update all yaxes
        fig.update_yaxes(
            title="Throughput (items/s)",
            gridcolor=self.colors["grid"]
        )
        
        return fig
    
    def _create_summary_table(self) -> str:
        """
        Create an HTML table with summary statistics.
        
        Returns:
            HTML string with summary table
        """
        # Extract data
        platforms_data = self.data.get("platforms", {})
        models_data = self.data.get("models", {})
        
        # Calculate summary statistics
        total_platforms = len(platforms_data)
        total_models = len(models_data)
        
        devices = set()
        for platform, platform_data in platforms_data.items():
            platform_devices = platform_data.get("devices", {})
            devices.update(platform_devices.keys())
        total_devices = len(devices)
        
        # Average metrics by platform
        platform_stats = {}
        for platform, platform_data in platforms_data.items():
            throughputs = []
            latencies = []
            memory_usages = []
            battery_impacts = []
            
            for model_name, model_data in platform_data.get("models", {}).items():
                for batch_size, batch_data in model_data.get("batch_sizes", {}).items():
                    if batch_size == "1":  # Only consider batch size 1 for averages
                        throughputs.append(batch_data.get("throughput", 0))
                        latencies.append(batch_data.get("latency", 0))
                        memory_usages.append(batch_data.get("memory_mb", 0))
                        battery_impacts.append(batch_data.get("battery_impact", 0))
            
            # Calculate average metrics
            avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
            avg_battery = sum(battery_impacts) / len(battery_impacts) if battery_impacts else 0
            
            platform_stats[platform] = {
                "avg_throughput": avg_throughput,
                "avg_latency": avg_latency,
                "avg_memory": avg_memory,
                "avg_battery": avg_battery
            }
        
        # Build HTML table
        html = f"""
        <div class="summary-table">
            <h3>Performance Summary</h3>
            <table>
                <tr class="header">
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Platforms</td>
                    <td>{total_platforms}</td>
                </tr>
                <tr>
                    <td>Total Devices</td>
                    <td>{total_devices}</td>
                </tr>
                <tr>
                    <td>Total Models</td>
                    <td>{total_models}</td>
                </tr>
                <tr>
                    <td>Analysis Date</td>
                    <td>{datetime.datetime.now().strftime('%Y-%m-%d')}</td>
                </tr>
            </table>
            
            <h3>Platform Metrics (Batch Size 1)</h3>
            <table>
                <tr class="header">
                    <th>Platform</th>
                    <th>Throughput (items/s)</th>
                    <th>Latency (ms)</th>
                    <th>Memory (MB)</th>
                    <th>Battery Impact (%)</th>
                </tr>
        """
        
        for platform, stats in platform_stats.items():
            html += f"""
                <tr>
                    <td>{platform.capitalize()}</td>
                    <td>{stats['avg_throughput']:.2f}</td>
                    <td>{stats['avg_latency']:.2f}</td>
                    <td>{stats['avg_memory']:.1f}</td>
                    <td>{stats['avg_battery']:.1f}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def create_dashboard(self) -> bool:
        """
        Create the complete HTML dashboard.
        
        Returns:
            Success status
        """
        # Load data if not already loaded
        if not self.data:
            if not self.load_data():
                return False
        
        # Try to load historical data if database provided
        if self.db_path:
            self.load_historical_data()
        
        # Create figures
        logger.info("Creating dashboard visualizations")
        
        # Create platform comparison chart
        platform_comparison = self._create_platform_comparison_chart()
        self.figures.append(platform_comparison)
        
        # Create model comparison chart
        model_comparison = self._create_model_comparison_chart()
        self.figures.append(model_comparison)
        
        # Create battery and memory chart
        battery_memory = self._create_battery_memory_chart()
        self.figures.append(battery_memory)
        
        # Create batch scaling chart
        batch_scaling = self._create_batch_scaling_chart()
        self.figures.append(batch_scaling)
        
        # Create historical trends chart if data available
        if self.historical_data is not None and not self.historical_data.empty:
            historical_trends = self._create_historical_trends_chart()
            if historical_trends:
                self.figures.append(historical_trends)
        
        # Create summary table
        summary_table = self._create_summary_table()
        
        # Generate HTML
        logger.info("Generating HTML dashboard")
        
        # HTML header
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {self.colors["background"]};
                    color: {self.colors["text"]};
                }}
                .dashboard-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .dashboard-header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: {self.colors["background"]};
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                }}
                .summary-table {{
                    margin-bottom: 30px;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: {self.colors["background"]};
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid {self.colors["grid"]};
                }}
                tr.header {{
                    background-color: rgba(0, 0, 0, 0.1);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 0.8em;
                    color: rgba({int(self.colors["text"][1:3], 16)}, {int(self.colors["text"][3:5], 16)}, {int(self.colors["text"][5:7], 16)}, 0.7);
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>{self.title}</h1>
                    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
        """
        
        # Add summary table
        html += f"""
                <div class="summary-table">
                    {summary_table}
                </div>
        """
        
        # Add charts
        for i, fig in enumerate(self.figures):
            div_id = f"chart-{i+1}"
            html += f"""
                <div class="chart-container">
                    <div id="{div_id}"></div>
                </div>
            """
        
        # Add footer and JavaScript
        html += """
                <div class="footer">
                    <p>Generated by Mobile Performance Dashboard Generator</p>
                </div>
            </div>
            
            <script>
        """
        
        # Add Plotly figures
        for i, fig in enumerate(self.figures):
            div_id = f"chart-{i+1}"
            fig_json = fig.to_json()
            html += f"""
                var figure{i+1} = {fig_json};
                Plotly.newPlot('{div_id}', figure{i+1}.data, figure{i+1}.layout);
            """
        
        # Close HTML
        html += """
            </script>
        </body>
        </html>
        """
        
        # Write HTML to file
        try:
            with open(self.output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Dashboard saved to {self.output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
            return False
    
    def run(self) -> bool:
        """
        Run the complete dashboard generation process.
        
        Returns:
            Success status
        """
        # Load data
        if not self.load_data():
            return False
        
        # Attempt to load historical data if database provided
        if self.db_path:
            self.load_historical_data()
        
        # Create dashboard
        return self.create_dashboard()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Mobile Performance Dashboard Generator")
    
    parser.add_argument("--data-file", required=True, help="Path to JSON data file with benchmark results")
    parser.add_argument("--output", default="mobile_dashboard.html", help="Path to output HTML dashboard")
    parser.add_argument("--db-path", help="Optional path to DuckDB database for historical data")
    parser.add_argument("--title", default="Mobile Edge Performance Dashboard", help="Dashboard title")
    parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Dashboard theme")
    parser.add_argument("--days", type=int, default=14, 
                       help="Number of days to include in historical trends (default: 14)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = MobileDashboardGenerator(
            data_file=args.data_file,
            output_path=args.output,
            db_path=args.db_path,
            title=args.title,
            theme=args.theme,
            days=args.days,
            verbose=args.verbose
        )
        
        # Run dashboard generation
        if generator.run():
            print(f"Dashboard successfully generated at: {args.output}")
            return 0
        else:
            print("Error generating dashboard")
            return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())