#!/usr/bin/env python3
"""
Visualization Engine for Distributed Testing Framework

This module implements the visualization engine for the distributed testing framework.
It creates interactive visualizations for test results and performance metrics.

Features:
- Interactive charts and graphs for performance data
- Comparative visualizations across dimensions
- Regression detection highlights
- Time-series performance tracking
- Multi-dimensional analysis views
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("visualization_engine")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import optional dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Static visualization features will be limited.")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Interactive visualization features will be limited.")
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Data manipulation features will be limited.")
    PANDAS_AVAILABLE = False

class VisualizationEngine:
    """Visualization engine for the distributed testing framework."""
    
    def __init__(self, result_aggregator=None, output_dir: str = "./visualizations"):
        """Initialize the visualization engine.
        
        Args:
            result_aggregator: Result aggregator for accessing result data
            output_dir: Directory to save visualizations
        """
        self.result_aggregator = result_aggregator
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            "theme": "light",  # light or dark
            "color_palette": "default",  # default, vibrant, pastel, colorblind, etc.
            "interactive": PLOTLY_AVAILABLE,  # Use interactive visualizations if Plotly is available
            "static_format": "png",  # png, svg, pdf
            "width": 1200,  # Default width in pixels
            "height": 800,  # Default height in pixels
            "dpi": 100,  # Dots per inch for static images
            "include_annotations": True,  # Include annotations on charts
            "branding": True,  # Include branding in visualizations
            "save_data": True,  # Save data alongside visualizations
        }
        
        # Default colors for different visualizations
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d6b117",
            "danger": "#d62728",
            "regression": "#d62728",
            "improvement": "#2ca02c",
            "neutral": "#7f7f7f",
            "background": "#ffffff" if self.config["theme"] == "light" else "#222222",
            "text": "#000000" if self.config["theme"] == "light" else "#ffffff",
        }
        
        # Default color mapping for dimensions
        self.dimension_colors = {
            "hardware": ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"],
            "model": ["#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b"],
            "batch_size": ["#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22"],
            "precision": ["#17becf", "#9edae5", "#393b79", "#5254a3", "#6b6ecf"],
            "task_type": ["#9c9ede", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6"],
        }
        
        logger.info("Visualization engine initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """Update the visualization engine configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        
        # Update colors based on theme
        if "theme" in config_updates:
            self.colors["background"] = "#ffffff" if self.config["theme"] == "light" else "#222222"
            self.colors["text"] = "#000000" if self.config["theme"] == "light" else "#ffffff"
        
        logger.info(f"Visualization engine configuration updated: {config_updates}")
    
    def create_performance_dashboard(self, 
                                    data: Optional[Dict[str, Any]] = None,
                                    output_path: Optional[str] = None,
                                    dimensions: List[str] = ["hardware", "model", "task_type"],
                                    metrics: List[str] = ["throughput", "latency", "memory_usage"],
                                    time_range: int = 30,
                                    include_regressions: bool = True,
                                    ) -> Optional[str]:
        """Create a comprehensive performance dashboard.
        
        Args:
            data: Optional performance data (will use result_aggregator if None)
            output_path: Optional path for the dashboard (autogenerated if None)
            dimensions: Dimensions to include in dashboard
            metrics: Metrics to include in dashboard
            time_range: Number of days to include in time-series views
            include_regressions: Whether to highlight regressions
            
        Returns:
            Path to the generated dashboard, or None if generation failed
        """
        if not (MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE):
            logger.error("Neither Matplotlib nor Plotly is available. Cannot create dashboard.")
            return None
        
        # Get data if not provided
        if data is None and self.result_aggregator:
            data = {
                "test_analysis": self.result_aggregator.get_test_analysis(),
                "dimension_analysis": self.result_aggregator.get_dimension_analysis(),
                "overall_status": self.result_aggregator.get_overall_status(),
                "regressions": self.result_aggregator.get_regressions(),
                "historical_performance": self.result_aggregator.historical_performance
            }
        
        if not data:
            logger.error("No data provided and no result aggregator available.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"performance_dashboard_{timestamp}.html")
        
        # Create interactive dashboard with Plotly if available
        if PLOTLY_AVAILABLE and self.config["interactive"]:
            return self._create_interactive_dashboard(data, output_path, dimensions, metrics, time_range, include_regressions)
        
        # Fall back to static dashboard with Matplotlib
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_dashboard(data, output_path, dimensions, metrics, time_range, include_regressions)
        
        return None
    
    def _create_interactive_dashboard(self, data, output_path, dimensions, metrics, time_range, include_regressions):
        """Create an interactive dashboard using Plotly.
        
        Args:
            data: Performance data
            output_path: Path for the dashboard
            dimensions: Dimensions to include
            metrics: Metrics to include
            time_range: Days to include in time-series
            include_regressions: Whether to highlight regressions
            
        Returns:
            Path to the generated dashboard
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.error("Plotly or Pandas not available. Cannot create interactive dashboard.")
            return None
        
        # Prepare data
        try:
            # Create an empty dashboard with multiple sections
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Distributed Testing Performance Dashboard</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: """ + self.colors["background"] + """;
                        color: """ + self.colors["text"] + """;
                    }
                    .dashboard-container {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    .dashboard-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 10px;
                        background-color: """ + ("#f0f0f0" if self.config["theme"] == "light" else "#333333") + """;
                        border-radius: 5px;
                    }
                    .dashboard-title {
                        font-size: 24px;
                        font-weight: bold;
                    }
                    .dashboard-timestamp {
                        font-size: 14px;
                        color: """ + ("#666666" if self.config["theme"] == "light" else "#aaaaaa") + """;
                    }
                    .dashboard-summary {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                    }
                    .summary-card {
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        background-color: """ + ("#ffffff" if self.config["theme"] == "light" else "#333333") + """;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    }
                    .summary-card h3 {
                        margin-top: 0;
                        border-bottom: 1px solid """ + ("#dddddd" if self.config["theme"] == "light" else "#444444") + """;
                        padding-bottom: 10px;
                    }
                    .chart-container {
                        margin-top: 20px;
                        background-color: """ + ("#ffffff" if self.config["theme"] == "light" else "#333333") + """;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    }
                    .chart-title {
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .chart-description {
                        font-size: 14px;
                        color: """ + ("#666666" if self.config["theme"] == "light" else "#aaaaaa") + """;
                        margin-bottom: 15px;
                    }
                    .regression-marker {
                        color: """ + self.colors["regression"] + """;
                        font-weight: bold;
                    }
                    .tabs {
                        display: flex;
                        border-bottom: 1px solid """ + ("#dddddd" if self.config["theme"] == "light" else "#444444") + """;
                        margin-bottom: 15px;
                    }
                    .tab {
                        padding: 10px 15px;
                        cursor: pointer;
                        border-bottom: 2px solid transparent;
                    }
                    .tab.active {
                        border-bottom: 2px solid """ + self.colors["primary"] + """;
                        font-weight: bold;
                    }
                    .tab-content {
                        display: none;
                    }
                    .tab-content.active {
                        display: block;
                    }
                </style>
            </head>
            <body>
                <div class="dashboard-container">
                    <div class="dashboard-header">
                        <div class="dashboard-title">Distributed Testing Performance Dashboard</div>
                        <div class="dashboard-timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
                    </div>
            """
            
            # Add summary section
            overall_status = data.get("overall_status", {})
            dashboard_html += """
                <div class="dashboard-summary">
                    <div class="summary-card">
                        <h3>Test Summary</h3>
                        <p><strong>Test Count:</strong> """ + str(overall_status.get("test_count", 0)) + """</p>
                        <p><strong>Total Executions:</strong> """ + str(overall_status.get("total_executions", 0)) + """</p>
                        <p><strong>Worker Count:</strong> """ + str(overall_status.get("worker_count", 0)) + """</p>
                        <p><strong>Task Type Count:</strong> """ + str(overall_status.get("task_type_count", 0)) + """</p>
                    </div>
                    
                    <div class="summary-card">
                        <h3>Performance</h3>
            """
            
            # Add aggregate metrics if available
            if "aggregated_metrics" in overall_status:
                for metric in metrics:
                    mean_key = f"{metric}_mean"
                    if mean_key in overall_status.get("aggregated_metrics", {}):
                        value = overall_status["aggregated_metrics"][mean_key]
                        dashboard_html += f"<p><strong>{metric.replace('_', ' ').title()}:</strong> {value:.2f}</p>"
            
            dashboard_html += """
                    </div>
                    
                    <div class="summary-card">
                        <h3>Regressions</h3>
                        <p><strong>Regression Count:</strong> """ + str(overall_status.get("regression_count", 0)) + """</p>
                        <p><strong>Significant Regressions:</strong> """ + str(overall_status.get("significant_regression_count", 0)) + """</p>
            """
            
            # Add top regressions if available
            regressions = data.get("regressions", {})
            if regressions:
                dashboard_html += "<p><strong>Top Regressions:</strong></p><ul>"
                count = 0
                for test_id, regression_info in regressions.items():
                    if count >= 3:
                        break
                    if regression_info.get("has_significant_regression", False):
                        for metric, info in regression_info.get("metrics", {}).items():
                            if info.get("is_regression", False) and info.get("is_significant", False):
                                dashboard_html += f"<li class='regression-marker'>{test_id}: {metric} ({info.get('percent_change', 0):.2f}%)</li>"
                                count += 1
                                if count >= 3:
                                    break
                dashboard_html += "</ul>"
            
            dashboard_html += """
                    </div>
                </div>
            """
            
            # Add tabs for different views
            dashboard_html += """
                <div class="chart-container">
                    <div class="tabs">
                        <div class="tab active" onclick="showTab('performance-trends')">Performance Trends</div>
                        <div class="tab" onclick="showTab('dimension-comparison')">Dimension Comparison</div>
                        <div class="tab" onclick="showTab('regression-analysis')">Regression Analysis</div>
                    </div>
                    
                    <div id="performance-trends" class="tab-content active">
                        <div class="chart-title">Performance Trends Over Time</div>
                        <div class="chart-description">Shows performance metrics over time for major test categories</div>
                        <div id="time-series-chart" style="width: 100%; height: 500px;"></div>
                    </div>
                    
                    <div id="dimension-comparison" class="tab-content">
                        <div class="chart-title">Performance Comparison by Dimension</div>
                        <div class="chart-description">Compares performance across different dimensions</div>
                        <div id="dimension-chart" style="width: 100%; height: 500px;"></div>
                    </div>
                    
                    <div id="regression-analysis" class="tab-content">
                        <div class="chart-title">Regression Analysis</div>
                        <div class="chart-description">Highlights detected performance regressions</div>
                        <div id="regression-chart" style="width: 100%; height: 500px;"></div>
                    </div>
                </div>
            """
            
            # Add Plotly charts as JavaScript
            dashboard_html += """
                <script>
                    // Tab switching
                    function showTab(tabId) {
                        // Hide all tab contents
                        var tabContents = document.getElementsByClassName('tab-content');
                        for (var i = 0; i < tabContents.length; i++) {
                            tabContents[i].classList.remove('active');
                        }
                        
                        // Deactivate all tabs
                        var tabs = document.getElementsByClassName('tab');
                        for (var i = 0; i < tabs.length; i++) {
                            tabs[i].classList.remove('active');
                        }
                        
                        // Activate selected tab and content
                        document.getElementById(tabId).classList.add('active');
                        document.querySelector('.tab[onclick="showTab(\\''+tabId+'\\')"]').classList.add('active');
                    }
            """
            
            # Add time series chart
            if "historical_performance" in data and PANDAS_AVAILABLE:
                # Extract historical performance data
                history = data["historical_performance"]
                chart_data = []
                
                for test_id, dates in history.items():
                    for date, metrics_data in dates.items():
                        for metric_name in metrics:
                            if metric_name in metrics_data.get("metrics", {}) and isinstance(metrics_data["metrics"][metric_name], dict):
                                mean = metrics_data["metrics"][metric_name].get("mean", 0)
                                chart_data.append({
                                    "test_id": test_id,
                                    "date": date,
                                    "metric": metric_name,
                                    "value": mean
                                })
                
                if chart_data:
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(chart_data)
                    
                    # Create time-series chart for each metric
                    fig = make_subplots(rows=len(metrics), cols=1, 
                                        shared_xaxes=True, 
                                        vertical_spacing=0.05,
                                        subplot_titles=[metric.replace('_', ' ').title() for metric in metrics])
                    
                    for i, metric in enumerate(metrics):
                        metric_df = df[df["metric"] == metric]
                        if not metric_df.empty:
                            test_ids = metric_df["test_id"].unique()
                            
                            for j, test_id in enumerate(test_ids):
                                test_df = metric_df[metric_df["test_id"] == test_id]
                                test_df = test_df.sort_values("date")
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=test_df["date"],
                                        y=test_df["value"],
                                        mode="lines+markers",
                                        name=f"{test_id} - {metric}",
                                        line=dict(width=2),
                                    ),
                                    row=i+1, col=1
                                )
                    
                    fig.update_layout(
                        height=500 * len(metrics),
                        width=1000,
                        showlegend=True,
                        template="plotly_white" if self.config["theme"] == "light" else "plotly_dark"
                    )
                    
                    # Add figure to dashboard
                    dashboard_html += f"""
                        // Time Series Chart
                        var timeSeriesData = {fig.to_json()};
                        Plotly.newPlot('time-series-chart', timeSeriesData.data, timeSeriesData.layout);
                    """
            
            # Add dimension comparison chart
            if "dimension_analysis" in data and PANDAS_AVAILABLE:
                dimension_data = data["dimension_analysis"]
                chart_data = []
                
                for dimension in dimensions:
                    if dimension in dimension_data:
                        for value, aggregates in dimension_data[dimension].items():
                            for metric in metrics:
                                mean_key = f"{metric}_mean"
                                if mean_key in aggregates:
                                    chart_data.append({
                                        "dimension": dimension,
                                        "value": value,
                                        "metric": metric,
                                        "mean_value": aggregates[mean_key]
                                    })
                
                if chart_data:
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(chart_data)
                    
                    # Create a subplot for each metric
                    fig = make_subplots(rows=len(metrics), cols=1, 
                                        shared_xaxes=True, 
                                        vertical_spacing=0.05,
                                        subplot_titles=[metric.replace('_', ' ').title() for metric in metrics])
                    
                    for i, metric in enumerate(metrics):
                        metric_df = df[df["metric"] == metric]
                        
                        for dimension in dimensions:
                            dim_df = metric_df[metric_df["dimension"] == dimension]
                            if not dim_df.empty:
                                dim_df = dim_df.sort_values("mean_value", ascending=False)
                                
                                fig.add_trace(
                                    go.Bar(
                                        x=dim_df["value"],
                                        y=dim_df["mean_value"],
                                        name=dimension,
                                    ),
                                    row=i+1, col=1
                                )
                    
                    fig.update_layout(
                        height=400 * len(metrics),
                        width=1000,
                        barmode='group',
                        showlegend=True,
                        template="plotly_white" if self.config["theme"] == "light" else "plotly_dark"
                    )
                    
                    # Add figure to dashboard
                    dashboard_html += f"""
                        // Dimension Comparison Chart
                        var dimensionData = {fig.to_json()};
                        Plotly.newPlot('dimension-chart', dimensionData.data, dimensionData.layout);
                    """
            
            # Add regression analysis chart
            if "regressions" in data and include_regressions and PANDAS_AVAILABLE:
                regressions = data["regressions"]
                chart_data = []
                
                for test_id, regression_info in regressions.items():
                    for metric, info in regression_info.get("metrics", {}).items():
                        if info.get("is_regression", False):
                            chart_data.append({
                                "test_id": test_id,
                                "metric": metric,
                                "percent_change": info.get("percent_change", 0),
                                "is_significant": info.get("is_significant", False),
                                "baseline_mean": info.get("baseline_mean", 0),
                                "current_mean": info.get("current_mean", 0)
                            })
                
                if chart_data:
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(chart_data)
                    
                    # Sort by percent change
                    df = df.sort_values("percent_change", ascending=True)
                    
                    # Create regression chart
                    fig = go.Figure()
                    
                    # Add bars for percent change
                    fig.add_trace(
                        go.Bar(
                            x=df["test_id"] + " - " + df["metric"],
                            y=df["percent_change"],
                            name="Percent Change",
                            marker_color=[self.colors["danger"] if row["is_significant"] else self.colors["warning"] 
                                         for _, row in df.iterrows()]
                        )
                    )
                    
                    fig.update_layout(
                        title="Performance Regressions",
                        xaxis_title="Test and Metric",
                        yaxis_title="Percent Change (%)",
                        height=600,
                        width=1000,
                        template="plotly_white" if self.config["theme"] == "light" else "plotly_dark"
                    )
                    
                    # Add figure to dashboard
                    dashboard_html += f"""
                        // Regression Analysis Chart
                        var regressionData = {fig.to_json()};
                        Plotly.newPlot('regression-chart', regressionData.data, regressionData.layout);
                    """
            
            # Close script and HTML
            dashboard_html += """
                </script>
                </div>
            </body>
            </html>
            """
            
            # Write dashboard to file
            with open(output_path, "w") as f:
                f.write(dashboard_html)
            
            logger.info(f"Interactive dashboard created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_static_dashboard(self, data, output_path, dimensions, metrics, time_range, include_regressions):
        """Create a static dashboard using Matplotlib.
        
        Args:
            data: Performance data
            output_path: Path for the dashboard
            dimensions: Dimensions to include
            metrics: Metrics to include
            time_range: Days to include in time-series
            include_regressions: Whether to highlight regressions
            
        Returns:
            Path to the generated dashboard
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create static dashboard.")
            return None
        
        # Adjust output path for static format
        base_path = os.path.splitext(output_path)[0]
        output_path = f"{base_path}.{self.config['static_format']}"
        
        try:
            # Create a large figure with multiple subplots
            fig_width = self.config["width"] / 100  # Convert pixels to inches
            fig_height = self.config["height"] / 100
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.config["dpi"])
            
            # Adjust figure background color based on theme
            if self.config["theme"] == "dark":
                plt.style.use("dark_background")
                fig.patch.set_facecolor(self.colors["background"])
            
            # Create grid for subplots
            grid = plt.GridSpec(4, 2, height_ratios=[1, 2, 2, 2])
            
            # Create summary subplot
            summary_ax = fig.add_subplot(grid[0, :])
            summary_ax.axis("off")
            summary_ax.set_title("Distributed Testing Performance Dashboard", fontsize=16, fontweight="bold")
            
            # Add summary text
            overall_status = data.get("overall_status", {})
            summary_text = (
                f"Test Count: {overall_status.get('test_count', 0)} | "
                f"Total Executions: {overall_status.get('total_executions', 0)} | "
                f"Workers: {overall_status.get('worker_count', 0)} | "
                f"Regressions: {overall_status.get('regression_count', 0)}"
            )
            summary_ax.text(0.5, 0.5, summary_text, fontsize=12, ha="center", va="center")
            
            # Create time-series subplot
            time_series_ax = fig.add_subplot(grid[1, :])
            self._plot_time_series(time_series_ax, data, metrics, time_range)
            
            # Create dimension comparison subplot
            dimension_ax = fig.add_subplot(grid[2, 0])
            self._plot_dimension_comparison(dimension_ax, data, dimensions, metrics[0] if metrics else None)
            
            # Create regression analysis subplot
            regression_ax = fig.add_subplot(grid[2, 1])
            self._plot_regression_analysis(regression_ax, data, include_regressions)
            
            # Create metric correlation subplot
            correlation_ax = fig.add_subplot(grid[3, 0])
            self._plot_metric_correlation(correlation_ax, data, metrics)
            
            # Create performance heatmap subplot
            heatmap_ax = fig.add_subplot(grid[3, 1])
            self._plot_performance_heatmap(heatmap_ax, data, dimensions, metrics)
            
            # Add timestamp
            plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                     fontsize=8, ha="right", va="bottom")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                       bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            
            logger.info(f"Static dashboard created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating static dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _plot_time_series(self, ax, data, metrics, time_range):
        """Plot time-series data on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: Performance data
            metrics: Metrics to plot
            time_range: Days to include
        """
        ax.set_title("Performance Trends Over Time", fontsize=12)
        
        if "historical_performance" not in data:
            ax.text(0.5, 0.5, "No historical data available", ha="center", va="center")
            return
        
        history = data["historical_performance"]
        if not history:
            ax.text(0.5, 0.5, "No historical data available", ha="center", va="center")
            return
        
        # Extract historical performance data for the first metric
        metric = metrics[0] if metrics else None
        if not metric:
            ax.text(0.5, 0.5, "No metrics specified", ha="center", va="center")
            return
        
        # Collect data for plotting
        plot_data = {}
        
        for test_id, dates in history.items():
            test_data = []
            for date_str, metrics_data in sorted(dates.items()):
                if metric in metrics_data.get("metrics", {}):
                    metric_data = metrics_data["metrics"][metric]
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                        test_data.append((date, metric_data["mean"]))
            
            if test_data:
                plot_data[test_id] = test_data
        
        if not plot_data:
            ax.text(0.5, 0.5, f"No historical data found for metric: {metric}", ha="center", va="center")
            return
        
        # Plot each test's data
        for i, (test_id, test_data) in enumerate(plot_data.items()):
            dates, values = zip(*test_data)
            ax.plot(dates, values, label=test_id, marker="o", linestyle="-", linewidth=2)
        
        # Format axis
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add legend
        if len(plot_data) > 1:
            ax.legend(loc="best", fontsize=8)
    
    def _plot_dimension_comparison(self, ax, data, dimensions, metric):
        """Plot dimension comparison on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: Performance data
            dimensions: Dimensions to compare
            metric: Metric to use for comparison
        """
        ax.set_title("Performance by Dimension", fontsize=12)
        
        if "dimension_analysis" not in data or not dimensions or not metric:
            ax.text(0.5, 0.5, "No dimension data available", ha="center", va="center")
            return
        
        dimension_data = data["dimension_analysis"]
        if not dimension_data:
            ax.text(0.5, 0.5, "No dimension data available", ha="center", va="center")
            return
        
        # Select the first dimension with data
        plot_dimension = None
        for dimension in dimensions:
            if dimension in dimension_data and dimension_data[dimension]:
                plot_dimension = dimension
                break
        
        if not plot_dimension:
            ax.text(0.5, 0.5, "No dimension data available", ha="center", va="center")
            return
        
        # Collect data for the selected dimension and metric
        mean_key = f"{metric}_mean"
        
        values = []
        means = []
        
        for value, aggregates in dimension_data[plot_dimension].items():
            if mean_key in aggregates:
                values.append(str(value))
                means.append(aggregates[mean_key])
        
        if not values:
            ax.text(0.5, 0.5, f"No data for dimension: {plot_dimension}", ha="center", va="center")
            return
        
        # Create bar chart
        bars = ax.bar(values, means, color=self.colors["primary"])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha="center", va="bottom", fontsize=8)
        
        # Format axis
        ax.set_xlabel(plot_dimension.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Rotate x-axis labels if needed
        if len(values) > 5:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    def _plot_regression_analysis(self, ax, data, include_regressions):
        """Plot regression analysis on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: Performance data
            include_regressions: Whether to highlight regressions
        """
        ax.set_title("Regression Analysis", fontsize=12)
        
        if not include_regressions or "regressions" not in data:
            ax.text(0.5, 0.5, "No regression data available", ha="center", va="center")
            return
        
        regressions = data["regressions"]
        if not regressions:
            ax.text(0.5, 0.5, "No regressions detected", ha="center", va="center")
            return
        
        # Collect regression data
        test_ids = []
        percent_changes = []
        is_significant = []
        
        for test_id, regression_info in regressions.items():
            # Find the most significant regression metric
            max_percent_change = 0
            max_significant = False
            
            for metric, info in regression_info.get("metrics", {}).items():
                if info.get("is_regression", False):
                    percent_change = abs(info.get("percent_change", 0))
                    significant = info.get("is_significant", False)
                    
                    if significant and percent_change > max_percent_change:
                        max_percent_change = percent_change
                        max_significant = True
                    elif not significant and percent_change > max_percent_change and not max_significant:
                        max_percent_change = percent_change
                        max_significant = False
            
            if max_percent_change > 0:
                test_ids.append(test_id)
                percent_changes.append(max_percent_change)
                is_significant.append(max_significant)
        
        if not test_ids:
            ax.text(0.5, 0.5, "No regression data available", ha="center", va="center")
            return
        
        # Sort by percent change
        sorted_data = sorted(zip(test_ids, percent_changes, is_significant), key=lambda x: x[1], reverse=True)
        
        # Limit to top 5
        if len(sorted_data) > 5:
            sorted_data = sorted_data[:5]
        
        test_ids, percent_changes, is_significant = zip(*sorted_data)
        
        # Create bar chart with color coding
        colors = [self.colors["danger"] if sig else self.colors["warning"] for sig in is_significant]
        bars = ax.bar(test_ids, percent_changes, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha="center", va="bottom", fontsize=8)
        
        # Format axis
        ax.set_xlabel("Test ID")
        ax.set_ylabel("Percent Change (%)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=self.colors["danger"], label="Significant"),
            mpatches.Patch(color=self.colors["warning"], label="Not Significant")
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
    
    def _plot_metric_correlation(self, ax, data, metrics):
        """Plot metric correlation on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: Performance data
            metrics: Metrics to analyze
        """
        ax.set_title("Metric Correlation", fontsize=12)
        
        if len(metrics) < 2:
            ax.text(0.5, 0.5, "Need at least 2 metrics for correlation", ha="center", va="center")
            return
        
        # Use the first two metrics
        metric_x = metrics[0]
        metric_y = metrics[1]
        
        # Collect correlation data from test analysis
        test_analysis = data.get("test_analysis", {})
        
        x_values = []
        y_values = []
        test_ids = []
        
        for test_id, analysis in test_analysis.items():
            # Check if both metrics are available
            if f"{metric_x}_mean" in analysis and f"{metric_y}_mean" in analysis:
                x_values.append(analysis[f"{metric_x}_mean"])
                y_values.append(analysis[f"{metric_y}_mean"])
                test_ids.append(test_id)
        
        if not x_values or not y_values:
            ax.text(0.5, 0.5, "No correlation data available", ha="center", va="center")
            return
        
        # Create scatter plot
        scatter = ax.scatter(x_values, y_values, color=self.colors["primary"], alpha=0.7)
        
        # Add trend line if scipy is available
        try:
            import scipy.stats as stats
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            
            # Plot regression line
            x_line = np.linspace(min(x_values), max(x_values), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=self.colors["secondary"], linestyle="--")
            
            # Add R² value
            ax.annotate(f"R² = {r_value**2:.3f}", 
                      xy=(0.95, 0.05), 
                      xycoords="axes fraction",
                      ha="right", va="bottom", 
                      fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        except ImportError:
            pass
        
        # Format axis
        ax.set_xlabel(metric_x.replace("_", " ").title())
        ax.set_ylabel(metric_y.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add interactive tooltips if supported
        if self.config["interactive"] and PLOTLY_AVAILABLE:
            # This can't be done with Matplotlib directly - handle in Plotly version
            pass
    
    def _plot_performance_heatmap(self, ax, data, dimensions, metrics):
        """Plot performance heatmap on the given axis.
        
        Args:
            ax: Matplotlib axis
            data: Performance data
            dimensions: Dimensions to use
            metrics: Metrics to analyze
        """
        ax.set_title("Performance Heatmap", fontsize=12)
        
        if len(dimensions) < 2 or not metrics:
            ax.text(0.5, 0.5, "Need at least 2 dimensions for heatmap", ha="center", va="center")
            return
        
        # Use the first metric
        metric = metrics[0]
        
        # Use the first two dimensions
        dim_x = dimensions[0]
        dim_y = dimensions[1]
        
        # Extract dimension values
        dimension_analysis = data.get("dimension_analysis", {})
        
        if dim_x not in dimension_analysis or dim_y not in dimension_analysis:
            ax.text(0.5, 0.5, "Dimension data not available", ha="center", va="center")
            return
        
        # Get unique values for each dimension
        x_values = list(dimension_analysis[dim_x].keys())
        y_values = list(dimension_analysis[dim_y].keys())
        
        if not x_values or not y_values:
            ax.text(0.5, 0.5, "Dimension values not available", ha="center", va="center")
            return
        
        # For heatmap, we need to reconstruct the data from test results
        test_analysis = data.get("test_analysis", {})
        
        # Create a matrix for heatmap values
        heatmap_data = np.zeros((len(y_values), len(x_values)))
        heatmap_data.fill(np.nan)  # Fill with NaN to show missing data
        
        # Fill matrix with data where available
        for test_id, analysis in test_analysis.items():
            # Check if test has both dimensions
            if dim_x in analysis and dim_y in analysis:
                x_val = analysis[dim_x]
                y_val = analysis[dim_y]
                
                # Find indices
                try:
                    x_idx = x_values.index(x_val)
                    y_idx = y_values.index(y_val)
                    
                    # Get metric value
                    metric_key = f"{metric}_mean"
                    if metric_key in analysis:
                        heatmap_data[y_idx, x_idx] = analysis[metric_key]
                except ValueError:
                    # Value not found in dimension list
                    pass
        
        # Check if we have any valid data
        if np.isnan(heatmap_data).all():
            ax.text(0.5, 0.5, "No heatmap data available", ha="center", va="center")
            return
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap="viridis")
        
        # Format axis
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add axis labels
        ax.set_xlabel(dim_x.replace("_", " ").title())
        ax.set_ylabel(dim_y.replace("_", " ").title())
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label(metric.replace("_", " ").title())
        
        # Add values in cells
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                value = heatmap_data[i, j]
                if not np.isnan(value):
                    # Determine text color based on background
                    im_color = im.cmap(im.norm(value))
                    lightness = 0.299 * im_color[0] + 0.587 * im_color[1] + 0.114 * im_color[2]
                    text_color = "white" if lightness < 0.5 else "black"
                    
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    
    def create_regression_report(self, 
                               data: Optional[Dict[str, Any]] = None,
                               output_path: Optional[str] = None,
                               include_non_significant: bool = False,
                               detailed: bool = True
                               ) -> Optional[str]:
        """Create a regression report.
        
        Args:
            data: Optional regression data (will use result_aggregator if None)
            output_path: Optional path for the report (autogenerated if None)
            include_non_significant: Whether to include non-significant regressions
            detailed: Whether to include detailed information
            
        Returns:
            Path to the generated report, or None if generation failed
        """
        # Get data if not provided
        if data is None and self.result_aggregator:
            data = {
                "regressions": self.result_aggregator.get_regressions(significant_only=not include_non_significant),
                "overall_status": self.result_aggregator.get_overall_status()
            }
        
        if not data or "regressions" not in data:
            logger.error("No regression data provided and no result aggregator available.")
            return None
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"regression_report_{timestamp}.html")
        
        try:
            # Create HTML report
            report_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Performance Regression Report</title>
                <meta charset="UTF-8">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: """ + self.colors["background"] + """;
                        color: """ + self.colors["text"] + """;
                    }
                    .report-container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .report-header {
                        padding: 10px;
                        background-color: """ + ("#f0f0f0" if self.config["theme"] == "light" else "#333333") + """;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }
                    .report-title {
                        font-size: 24px;
                        font-weight: bold;
                        margin: 0;
                    }
                    .report-timestamp {
                        font-size: 14px;
                        color: """ + ("#666666" if self.config["theme"] == "light" else "#aaaaaa") + """;
                        margin-top: 5px;
                    }
                    .regression-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                    }
                    .regression-table th, .regression-table td {
                        padding: 8px;
                        text-align: left;
                        border-bottom: 1px solid """ + ("#dddddd" if self.config["theme"] == "light" else "#444444") + """;
                    }
                    .regression-table th {
                        background-color: """ + ("#f0f0f0" if self.config["theme"] == "light" else "#333333") + """;
                        font-weight: bold;
                    }
                    .regression-table tr:hover {
                        background-color: """ + ("#f5f5f5" if self.config["theme"] == "light" else "#3a3a3a") + """;
                    }
                    .significant {
                        color: """ + self.colors["danger"] + """;
                        font-weight: bold;
                    }
                    .not-significant {
                        color: """ + self.colors["warning"] + """;
                    }
                    .summary-card {
                        padding: 15px;
                        background-color: """ + ("#ffffff" if self.config["theme"] == "light" else "#333333") + """;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                    }
                    .regression-details {
                        margin-top: 10px;
                        padding: 10px;
                        background-color: """ + ("#f9f9f9" if self.config["theme"] == "light" else "#2a2a2a") + """;
                        border-radius: 5px;
                        display: none;
                    }
                    .details-toggle {
                        cursor: pointer;
                        color: """ + self.colors["primary"] + """;
                        text-decoration: underline;
                    }
                </style>
            </head>
            <body>
                <div class="report-container">
                    <div class="report-header">
                        <h1 class="report-title">Performance Regression Report</h1>
                        <div class="report-timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
                    </div>
                    
                    <div class="summary-card">
                        <h2>Regression Summary</h2>
            """
            
            # Add summary information
            overall_status = data.get("overall_status", {})
            regressions = data.get("regressions", {})
            
            regression_count = len(regressions)
            significant_count = sum(1 for r in regressions.values() if r.get("has_significant_regression", False))
            
            report_html += f"""
                <p><strong>Total Regressions:</strong> {regression_count}</p>
                <p><strong>Significant Regressions:</strong> {significant_count}</p>
                <p><strong>Tests Analyzed:</strong> {overall_status.get("test_count", 0)}</p>
                <p><strong>Total Executions:</strong> {overall_status.get("total_executions", 0)}</p>
            </div>
            """
            
            # Add regression table
            report_html += """
                <h2>Regression Details</h2>
                
                <table class="regression-table">
                    <thead>
                        <tr>
                            <th>Test ID</th>
                            <th>Metric</th>
                            <th>Percent Change</th>
                            <th>Significance</th>
                            <th>Baseline Value</th>
                            <th>Current Value</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Add rows for each regression
            for test_id, regression_info in sorted(regressions.items()):
                metrics = regression_info.get("metrics", {})
                
                for metric, info in metrics.items():
                    if info.get("is_regression", False):
                        # Skip non-significant if requested
                        if not include_non_significant and not info.get("is_significant", False):
                            continue
                            
                        # Determine CSS class based on significance
                        css_class = "significant" if info.get("is_significant", False) else "not-significant"
                        
                        # Format values
                        percent_change = info.get("percent_change", 0)
                        baseline_value = info.get("baseline_mean", 0)
                        current_value = info.get("current_mean", 0)
                        
                        # Add row
                        report_html += f"""
                            <tr class="{css_class}">
                                <td>{test_id}</td>
                                <td>{metric}</td>
                                <td>{percent_change:.2f}%</td>
                                <td>{"Significant" if info.get("is_significant", False) else "Not significant"}</td>
                                <td>{baseline_value:.2f}</td>
                                <td>{current_value:.2f}</td>
                            </tr>
                        """
                        
                        # Add detailed information if requested
                        if detailed:
                            # Extract additional details
                            report_html += f"""
                                <tr>
                                    <td colspan="6">
                                        <div class="details-toggle" onclick="toggleDetails('{test_id}_{metric}')">[Show Details]</div>
                                        <div id="{test_id}_{metric}" class="regression-details">
                                            <p><strong>Baseline Period:</strong> {', '.join(regression_info.get("baseline_period", []))}</p>
                                            <p><strong>Current Period:</strong> {', '.join(regression_info.get("current_period", []))}</p>
                                            <p><strong>Baseline Sample Size:</strong> {info.get("baseline_sample_size", 0)}</p>
                                            <p><strong>Current Sample Size:</strong> {info.get("current_sample_size", 0)}</p>
                                            <p><strong>P-value:</strong> {info.get("p_value", "N/A")}</p>
                                        </div>
                                    </td>
                                </tr>
                            """
            
            # Close table and add JavaScript for toggle functionality
            report_html += """
                    </tbody>
                </table>
                
                <script>
                    function toggleDetails(id) {
                        var details = document.getElementById(id);
                        var displayStyle = details.style.display;
                        
                        if (displayStyle === "block") {
                            details.style.display = "none";
                            var toggleLink = details.previousElementSibling;
                            toggleLink.textContent = "[Show Details]";
                        } else {
                            details.style.display = "block";
                            var toggleLink = details.previousElementSibling;
                            toggleLink.textContent = "[Hide Details]";
                        }
                    }
                </script>
                
                </div>
            </body>
            </html>
            """
            
            # Write report to file
            with open(output_path, "w") as f:
                f.write(report_html)
                
            logger.info(f"Regression report created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating regression report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_visualization(self, 
                           visualization_type: str,
                           data: Dict[str, Any],
                           output_path: Optional[str] = None
                           ) -> Optional[str]:
        """Create a specific visualization.
        
        Args:
            visualization_type: Type of visualization to create
            data: Data for the visualization
            output_path: Optional path for the visualization
            
        Returns:
            Path to the generated visualization, or None if generation failed
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                f"{visualization_type}_{timestamp}.{self.config['static_format']}"
            )
        
        try:
            # Create the visualization based on type
            if visualization_type == "time_series":
                return self._create_time_series_visualization(data, output_path)
            elif visualization_type == "dimension_comparison":
                return self._create_dimension_comparison_visualization(data, output_path)
            elif visualization_type == "regression_analysis":
                return self._create_regression_visualization(data, output_path)
            elif visualization_type == "correlation":
                return self._create_correlation_visualization(data, output_path)
            elif visualization_type == "heatmap":
                return self._create_heatmap_visualization(data, output_path)
            else:
                logger.error(f"Unknown visualization type: {visualization_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_time_series_visualization(self, data, output_path):
        """Create time series visualization.
        
        Args:
            data: Data for the visualization
            output_path: Path for the visualization
            
        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create visualization.")
            return None
        
        # Extract required data
        time_series_data = data.get("time_series", {})
        metric = data.get("metric", "unknown")
        title = data.get("title", f"{metric} over time")
        
        if not time_series_data:
            logger.error("No time series data provided.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["dpi"])
        
        # Set theme
        if self.config["theme"] == "dark":
            plt.style.use("dark_background")
            fig.patch.set_facecolor(self.colors["background"])
        
        # Plot each series
        for name, series in time_series_data.items():
            timestamps, values = zip(*series)
            ax.plot(timestamps, values, label=name, marker="o", linestyle="-", linewidth=2)
        
        # Format axis
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Format x-axis as dates if timestamps are datetime objects
        if all(isinstance(timestamp, datetime) for timestamp, _ in next(iter(time_series_data.values()))):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add legend
        if len(time_series_data) > 1:
            ax.legend(loc="best")
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fontsize=8, ha="right", va="bottom")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                   bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        logger.info(f"Time series visualization created: {output_path}")
        return output_path
    
    def _create_dimension_comparison_visualization(self, data, output_path):
        """Create dimension comparison visualization.
        
        Args:
            data: Data for the visualization
            output_path: Path for the visualization
            
        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create visualization.")
            return None
        
        # Extract required data
        dimension = data.get("dimension", "unknown")
        metric = data.get("metric", "unknown")
        values = data.get("values", {})
        title = data.get("title", f"{metric} by {dimension}")
        
        if not values:
            logger.error("No dimension comparison data provided.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["dpi"])
        
        # Set theme
        if self.config["theme"] == "dark":
            plt.style.use("dark_background")
            fig.patch.set_facecolor(self.colors["background"])
        
        # Extract labels and values
        labels = list(values.keys())
        heights = list(values.values())
        
        # Create bar chart
        bars = ax.bar(labels, heights, color=self.colors["primary"])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha="center", va="bottom")
        
        # Format axis
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(dimension.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        
        # Rotate x-axis labels if needed
        if len(labels) > 5:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fontsize=8, ha="right", va="bottom")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                   bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        logger.info(f"Dimension comparison visualization created: {output_path}")
        return output_path
    
    def _create_regression_visualization(self, data, output_path):
        """Create regression visualization.
        
        Args:
            data: Data for the visualization
            output_path: Path for the visualization
            
        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create visualization.")
            return None
        
        # Extract required data
        regressions = data.get("regressions", {})
        title = data.get("title", "Performance Regressions")
        
        if not regressions:
            logger.error("No regression data provided.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["dpi"])
        
        # Set theme
        if self.config["theme"] == "dark":
            plt.style.use("dark_background")
            fig.patch.set_facecolor(self.colors["background"])
        
        # Prepare data for plotting
        labels = []
        percent_changes = []
        colors = []
        
        for test_id, regression_info in regressions.items():
            for metric, info in regression_info.get("metrics", {}).items():
                if info.get("is_regression", False):
                    labels.append(f"{test_id}\n{metric}")
                    percent_changes.append(info.get("percent_change", 0))
                    colors.append(self.colors["danger"] if info.get("is_significant", False) else self.colors["warning"])
        
        # Sort by percent change
        sorted_data = sorted(zip(labels, percent_changes, colors), key=lambda x: x[1])
        
        if sorted_data:
            labels, percent_changes, colors = zip(*sorted_data)
        else:
            logger.error("No regression data to plot.")
            return None
        
        # Create bar chart
        bars = ax.barh(labels, percent_changes, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.2f}%",
                      xy=(width, bar.get_y() + bar.get_height() / 2),
                      xytext=(5, 0),  # 5 points horizontal offset
                      textcoords="offset points",
                      ha="left" if width >= 0 else "right", va="center")
        
        # Format axis
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Percent Change (%)")
        ax.set_ylabel("")
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=self.colors["danger"], label="Significant"),
            mpatches.Patch(color=self.colors["warning"], label="Not Significant")
        ]
        ax.legend(handles=legend_handles, loc="best")
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fontsize=8, ha="right", va="bottom")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                   bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        logger.info(f"Regression visualization created: {output_path}")
        return output_path
    
    def _create_correlation_visualization(self, data, output_path):
        """Create correlation visualization.
        
        Args:
            data: Data for the visualization
            output_path: Path for the visualization
            
        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create visualization.")
            return None
        
        # Extract required data
        metric_x = data.get("metric_x", "unknown")
        metric_y = data.get("metric_y", "unknown")
        points = data.get("points", [])
        labels = data.get("labels", [])
        title = data.get("title", f"{metric_y} vs {metric_x}")
        
        if not points:
            logger.error("No correlation data provided.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["dpi"])
        
        # Set theme
        if self.config["theme"] == "dark":
            plt.style.use("dark_background")
            fig.patch.set_facecolor(self.colors["background"])
        
        # Extract x and y values
        x_values, y_values = zip(*points)
        
        # Create scatter plot
        scatter = ax.scatter(x_values, y_values, color=self.colors["primary"], alpha=0.7)
        
        # Add labels if provided
        if labels and len(labels) == len(points):
            for i, (x, y) in enumerate(points):
                ax.annotate(labels[i],
                          xy=(x, y),
                          xytext=(5, 5),  # 5 points offset
                          textcoords="offset points",
                          fontsize=8)
        
        # Add trend line if scipy is available
        try:
            import scipy.stats as stats
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            
            # Plot regression line
            x_line = np.linspace(min(x_values), max(x_values), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=self.colors["secondary"], linestyle="--")
            
            # Add R² value
            ax.annotate(f"R² = {r_value**2:.3f}", 
                      xy=(0.95, 0.05), 
                      xycoords="axes fraction",
                      ha="right", va="bottom", 
                      fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white" if self.config["theme"] == "light" else "#333333", alpha=0.8))
        except ImportError:
            pass
        
        # Format axis
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(metric_x.replace("_", " ").title())
        ax.set_ylabel(metric_y.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fontsize=8, ha="right", va="bottom")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                   bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        logger.info(f"Correlation visualization created: {output_path}")
        return output_path
    
    def _create_heatmap_visualization(self, data, output_path):
        """Create heatmap visualization.
        
        Args:
            data: Data for the visualization
            output_path: Path for the visualization
            
        Returns:
            Path to the generated visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Cannot create visualization.")
            return None
        
        # Extract required data
        x_dimension = data.get("x_dimension", "unknown")
        y_dimension = data.get("y_dimension", "unknown")
        metric = data.get("metric", "unknown")
        values = data.get("values", [])
        x_labels = data.get("x_labels", [])
        y_labels = data.get("y_labels", [])
        title = data.get("title", f"{metric} by {x_dimension} and {y_dimension}")
        
        if not values or not x_labels or not y_labels:
            logger.error("No heatmap data provided or missing labels.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config["dpi"])
        
        # Set theme
        if self.config["theme"] == "dark":
            plt.style.use("dark_background")
            fig.patch.set_facecolor(self.colors["background"])
        
        # Create heatmap
        im = ax.imshow(values, cmap="viridis")
        
        # Format axis
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Add axis labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_dimension.replace("_", " ").title())
        ax.set_ylabel(y_dimension.replace("_", " ").title())
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace("_", " ").title())
        
        # Add values in cells
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                value = values[i][j]
                if not np.isnan(value):
                    # Determine text color based on background
                    im_color = im.cmap(im.norm(value))
                    lightness = 0.299 * im_color[0] + 0.587 * im_color[1] + 0.114 * im_color[2]
                    text_color = "white" if lightness < 0.5 else "black"
                    
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        
        # Add timestamp
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 fontsize=8, ha="right", va="bottom")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, format=self.config["static_format"], dpi=self.config["dpi"],
                   bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        
        logger.info(f"Heatmap visualization created: {output_path}")
        return output_path