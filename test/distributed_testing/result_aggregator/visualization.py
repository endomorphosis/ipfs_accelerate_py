#!/usr/bin/env python3
"""
Result Aggregator Visualization Module

This module provides visualization capabilities for the Result Aggregator service.
It creates various types of visualizations for test results, performance trends,
and anomaly detection.

Usage:
    from result_aggregator.visualization import ResultVisualizer
    
    # Create a visualizer
    visualizer = ResultVisualizer(service)
    
    # Generate visualizations
    visualizer.generate_performance_chart(metrics=["throughput", "latency"], output_path="performance.html")
    visualizer.generate_trend_analysis(output_path="trends.html")
    visualizer.generate_anomaly_dashboard(output_path="anomalies.html")
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("visualization.log")
    ]
)
logger = logging.getLogger(__name__)

# Try to import optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Basic visualization features will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive visualization features will be disabled.")

class ResultVisualizer:
    """Visualization tool for Result Aggregator service."""
    
    def __init__(self, service):
        """
        Initialize the result visualizer.
        
        Args:
            service: Result Aggregator service instance
        """
        self.service = service
        self.enable_matplotlib = MATPLOTLIB_AVAILABLE
        self.enable_plotly = PLOTLY_AVAILABLE
        
        # Set up default styling for matplotlib
        if self.enable_matplotlib:
            # Use a clean, modern style
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Set up default figure size
            plt.rcParams['figure.figsize'] = [10, 6]
            
            # Increase font sizes for better readability
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.labelsize'] = 14
            
            # Use a color palette that works well for data visualization
            self.color_palette = sns.color_palette("viridis", 10)
            sns.set_palette(self.color_palette)
    
    def _check_visualization_libraries(self, interactive: bool = False) -> bool:
        """
        Check if required visualization libraries are available.
        
        Args:
            interactive: Whether interactive visualizations are required
            
        Returns:
            True if libraries are available, False otherwise
        """
        if interactive and not self.enable_plotly:
            logger.warning("Interactive visualizations require Plotly. Install with 'pip install plotly'.")
            return False
        
        if not interactive and not self.enable_matplotlib:
            logger.warning("Static visualizations require Matplotlib. Install with 'pip install matplotlib seaborn'.")
            return False
        
        return True
    
    def generate_performance_chart(self, 
                                 metrics: List[str] = None, 
                                 filter_criteria: Dict[str, Any] = None,
                                 output_path: str = None,
                                 interactive: bool = True) -> Any:
        """
        Generate a performance chart for specified metrics.
        
        Args:
            metrics: List of metrics to include (default: all metrics)
            filter_criteria: Criteria to filter results
            output_path: Path to save the chart
            interactive: Whether to create an interactive chart
            
        Returns:
            Figure or None if visualization libraries not available
        """
        if not self._check_visualization_libraries(interactive):
            return None
        
        # Get results from service
        results = self.service.get_results(filter_criteria=filter_criteria)
        
        if not results:
            logger.warning("No results found for performance chart")
            return None
        
        # Convert results to DataFrame
        df_rows = []
        
        for result in results:
            timestamp = result["timestamp"]
            result_metrics = result["metrics"]
            task_type = result["type"]
            worker_id = result["worker_id"]
            
            # Extract all metrics
            row = {
                "timestamp": timestamp,
                "task_type": task_type,
                "worker_id": worker_id
            }
            
            # Add metrics from the result
            for metric_name, metric_value in result_metrics.items():
                if isinstance(metric_value, dict) and "value" in metric_value:
                    row[metric_name] = metric_value["value"]
                else:
                    row[metric_name] = metric_value
            
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Filter metrics if specified
        available_metrics = [col for col in df.columns if col not in ["timestamp", "task_type", "worker_id"]]
        
        if metrics:
            plot_metrics = [m for m in metrics if m in available_metrics]
            if not plot_metrics:
                logger.warning(f"None of the specified metrics {metrics} are available. Available metrics: {available_metrics}")
                return None
        else:
            # Use all available metrics
            plot_metrics = available_metrics
        
        if interactive:
            # Create interactive visualization with Plotly
            fig = self._generate_interactive_performance_chart(df, plot_metrics)
            
            # Save to file if output_path is specified
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Interactive performance chart saved to {output_path}")
            
            return fig
        else:
            # Create static visualization with Matplotlib
            fig = self._generate_static_performance_chart(df, plot_metrics)
            
            # Save to file if output_path is specified
            if output_path:
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                logger.info(f"Static performance chart saved to {output_path}")
            
            return fig
    
    def _generate_interactive_performance_chart(self, df: pd.DataFrame, metrics: List[str]) -> Any:
        """
        Generate an interactive performance chart using Plotly.
        
        Args:
            df: DataFrame with performance data
            metrics: List of metrics to include
            
        Returns:
            Plotly figure
        """
        if len(metrics) == 1:
            # Single metric line chart
            metric = metrics[0]
            fig = px.line(
                df, 
                x="timestamp", 
                y=metric,
                color="task_type",
                hover_data=["worker_id"],
                title=f"Performance Trend: {metric}"
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title=metric,
                legend_title="Task Type",
                hovermode="closest"
            )
        else:
            # Multiple metrics - create subplots
            fig = make_subplots(
                rows=len(metrics), 
                cols=1,
                shared_xaxes=True,
                subplot_titles=metrics,
                vertical_spacing=0.05
            )
            
            # Add each metric as a subplot
            for i, metric in enumerate(metrics):
                for task_type in df["task_type"].unique():
                    task_df = df[df["task_type"] == task_type]
                    fig.add_trace(
                        go.Scatter(
                            x=task_df["timestamp"],
                            y=task_df[metric],
                            mode="lines+markers",
                            name=f"{task_type} - {metric}",
                            legendgroup=task_type,
                            hovertemplate=f"{metric}: %{{y}}<br>Time: %{{x}}<br>Task Type: {task_type}<extra></extra>"
                        ),
                        row=i+1,
                        col=1
                    )
            
            # Improve layout
            fig.update_layout(
                title="Performance Trends",
                xaxis_title="Time",
                height=300 * len(metrics),
                legend_title="Task Type",
                hovermode="closest"
            )
            
            # Set y-axis titles for each subplot
            for i, metric in enumerate(metrics):
                fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        return fig
    
    def _generate_static_performance_chart(self, df: pd.DataFrame, metrics: List[str]) -> Any:
        """
        Generate a static performance chart using Matplotlib.
        
        Args:
            df: DataFrame with performance data
            metrics: List of metrics to include
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)), sharex=True)
        
        # Handle case of single metric
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot data for each task type
            for j, task_type in enumerate(df["task_type"].unique()):
                task_df = df[df["task_type"] == task_type]
                ax.plot(
                    task_df["timestamp"],
                    task_df[metric],
                    marker="o",
                    linestyle="-",
                    label=task_type,
                    color=self.color_palette[j % len(self.color_palette)]
                )
            
            # Set labels and title
            ax.set_ylabel(metric)
            ax.set_title(f"Performance Trend: {metric}")
            
            # Add legend
            ax.legend(title="Task Type")
            
            # Format x-axis to show dates nicely
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            
            # Add grid for better readability
            ax.grid(True, linestyle="--", alpha=0.7)
        
        # Set common x-axis label
        axes[-1].set_xlabel("Time")
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def generate_trend_analysis(self,
                              metrics: List[str] = None,
                              filter_criteria: Dict[str, Any] = None,
                              output_path: str = None,
                              interactive: bool = True) -> Any:
        """
        Generate a trend analysis visualization.
        
        Args:
            metrics: List of metrics to include (default: all metrics)
            filter_criteria: Criteria to filter results
            output_path: Path to save the visualization
            interactive: Whether to create an interactive visualization
            
        Returns:
            Figure or None if visualization libraries not available
        """
        if not self._check_visualization_libraries(interactive):
            return None
        
        # Analyze performance trends
        trends = self.service.analyze_performance_trends(filter_criteria=filter_criteria)
        
        if not trends:
            logger.warning("No trends found for analysis")
            return None
        
        # Filter metrics if specified
        if metrics:
            trend_metrics = {k: v for k, v in trends.items() if k in metrics and "trend" in v}
        else:
            trend_metrics = {k: v for k, v in trends.items() if "trend" in v}
        
        if not trend_metrics:
            logger.warning("No valid trend metrics found")
            return None
        
        if interactive:
            # Create interactive visualization with Plotly
            fig = self._generate_interactive_trend_analysis(trend_metrics)
            
            # Save to file if output_path is specified
            if output_path:
                fig.write_html(output_path)
                logger.info(f"Interactive trend analysis saved to {output_path}")
            
            return fig
        else:
            # Create static visualization with Matplotlib
            fig = self._generate_static_trend_analysis(trend_metrics)
            
            # Save to file if output_path is specified
            if output_path:
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                logger.info(f"Static trend analysis saved to {output_path}")
            
            return fig
    
    def _generate_interactive_trend_analysis(self, trends: Dict[str, Dict[str, Any]]) -> Any:
        """
        Generate an interactive trend analysis visualization using Plotly.
        
        Args:
            trends: Dictionary of trend data
            
        Returns:
            Plotly figure
        """
        # Create data for visualization
        metrics = []
        percent_changes = []
        trend_types = []
        mean_values = []
        
        for metric, trend_data in trends.items():
            metrics.append(metric)
            percent_changes.append(trend_data.get("percent_change", 0))
            trend_types.append(trend_data.get("trend", "unknown"))
            mean_values.append(trend_data.get("statistics", {}).get("mean", 0))
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            "metric": metrics,
            "percent_change": percent_changes,
            "trend": trend_types,
            "mean_value": mean_values
        })
        
        # Sort by absolute percent change
        df["abs_percent_change"] = df["percent_change"].abs()
        df = df.sort_values("abs_percent_change", ascending=False)
        
        # Define colors for trends
        colors = {
            "increasing": "green",
            "decreasing": "red",
            "stable": "blue",
            "unknown": "gray"
        }
        
        # Create bar chart
        fig = px.bar(
            df,
            x="metric",
            y="percent_change",
            color="trend",
            color_discrete_map=colors,
            title="Performance Trends by Metric",
            hover_data=["mean_value"]
        )
        
        # Add a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title="Percent Change (%)",
            legend_title="Trend",
            hovermode="closest"
        )
        
        return fig
    
    def _generate_static_trend_analysis(self, trends: Dict[str, Dict[str, Any]]) -> Any:
        """
        Generate a static trend analysis visualization using Matplotlib.
        
        Args:
            trends: Dictionary of trend data
            
        Returns:
            Matplotlib figure
        """
        # Create data for visualization
        metrics = []
        percent_changes = []
        trend_types = []
        
        for metric, trend_data in trends.items():
            metrics.append(metric)
            percent_changes.append(trend_data.get("percent_change", 0))
            trend_types.append(trend_data.get("trend", "unknown"))
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            "metric": metrics,
            "percent_change": percent_changes,
            "trend": trend_types
        })
        
        # Sort by absolute percent change
        df["abs_percent_change"] = df["percent_change"].abs()
        df = df.sort_values("abs_percent_change", ascending=False)
        
        # Define colors for trends
        colors = {
            "increasing": "green",
            "decreasing": "red",
            "stable": "blue",
            "unknown": "gray"
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        bars = ax.bar(
            df["metric"],
            df["percent_change"],
            color=[colors[t] for t in df["trend"]]
        )
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel("Metric")
        ax.set_ylabel("Percent Change (%)")
        ax.set_title("Performance Trends by Metric")
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=trend) for trend, color in colors.items() if trend in df["trend"].values]
        ax.legend(handles=legend_elements, title="Trend")
        
        # Adjust layout
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = "bottom"
                offset = 1
            else:
                va = "top"
                offset = -1
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.1f}%",
                ha="center",
                va=va,
                rotation=0,
                fontsize=10
            )
        
        return fig
    
    def generate_anomaly_dashboard(self,
                                 filter_criteria: Dict[str, Any] = None,
                                 output_path: str = None) -> Any:
        """
        Generate an anomaly detection dashboard.
        
        Args:
            filter_criteria: Criteria to filter results
            output_path: Path to save the dashboard
            
        Returns:
            Dashboard figure or None if visualization libraries not available
        """
        if not self.enable_plotly:
            logger.warning("Anomaly dashboard requires Plotly. Install with 'pip install plotly'.")
            return None
        
        # Detect anomalies
        anomalies = self.service.detect_anomalies(filter_criteria=filter_criteria)
        
        if not anomalies:
            logger.warning("No anomalies found for dashboard")
            return None
        
        # Create data for visualization
        result_ids = []
        scores = []
        types = []
        metrics = []
        values = []
        z_scores = []
        
        for anomaly in anomalies:
            result_id = anomaly.get("result_id", "unknown")
            anomaly_type = anomaly.get("type", "unknown")
            score = anomaly.get("score", 0)
            
            # Get anomalous features
            anomalous_features = anomaly.get("details", {}).get("anomalous_features", [])
            
            for feature in anomalous_features:
                result_ids.append(result_id)
                scores.append(score)
                types.append(anomaly_type)
                metrics.append(feature.get("feature", "unknown"))
                values.append(feature.get("value", 0))
                z_scores.append(feature.get("z_score", 0))
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            "result_id": result_ids,
            "anomaly_score": scores,
            "anomaly_type": types,
            "metric": metrics,
            "value": values,
            "z_score": z_scores
        })
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[
                "Anomaly Scores by Result",
                "Z-Scores by Metric",
                "Anomalous Values by Metric",
                "Anomaly Types Distribution"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "pie"}]
            ]
        )
        
        # 1. Anomaly Scores by Result (Bar chart)
        result_scores = df.groupby("result_id")["anomaly_score"].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=result_scores["result_id"],
                y=result_scores["anomaly_score"],
                name="Anomaly Score",
                marker_color="red"
            ),
            row=1, col=1
        )
        
        # 2. Z-Scores by Metric (Scatter plot)
        fig.add_trace(
            go.Scatter(
                x=df["metric"],
                y=df["z_score"],
                mode="markers",
                name="Z-Score",
                marker=dict(
                    size=df["z_score"].abs() * 3,
                    color=df["z_score"],
                    colorscale="RdBu_r",
                    colorbar=dict(title="Z-Score")
                )
            ),
            row=1, col=2
        )
        
        # 3. Anomalous Values by Metric (Box plot)
        fig.add_trace(
            go.Box(
                x=df["metric"],
                y=df["value"],
                name="Anomalous Values"
            ),
            row=2, col=1
        )
        
        # 4. Anomaly Types Distribution (Pie chart)
        type_counts = df["anomaly_type"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name="Anomaly Types"
            ),
            row=2, col=2
        )
        
        # Improve layout
        fig.update_layout(
            title="Anomaly Detection Dashboard",
            height=800,
            width=1200,
            showlegend=False,
            hovermode="closest"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Result ID", row=1, col=1)
        fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Z-Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        # Save to file if output_path is specified
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Anomaly dashboard saved to {output_path}")
        
        return fig
    
    def generate_summary_dashboard(self,
                                 filter_criteria: Dict[str, Any] = None,
                                 output_path: str = None) -> Any:
        """
        Generate a summary dashboard with key metrics and statistics.
        
        Args:
            filter_criteria: Criteria to filter results
            output_path: Path to save the dashboard
            
        Returns:
            Dashboard figure or None if visualization libraries not available
        """
        if not self.enable_plotly:
            logger.warning("Summary dashboard requires Plotly. Install with 'pip install plotly'.")
            return None
        
        # Get results
        results = self.service.get_results(filter_criteria=filter_criteria)
        
        if not results:
            logger.warning("No results found for summary dashboard")
            return None
        
        # Convert results to DataFrame
        df_rows = []
        
        for result in results:
            timestamp = result["timestamp"]
            result_metrics = result["metrics"]
            task_type = result["type"]
            worker_id = result["worker_id"]
            status = result["status"]
            
            # Extract all metrics
            row = {
                "timestamp": timestamp,
                "task_type": task_type,
                "worker_id": worker_id,
                "status": status
            }
            
            # Add metrics from the result
            for metric_name, metric_value in result_metrics.items():
                if isinstance(metric_value, dict) and "value" in metric_value:
                    row[metric_name] = metric_value["value"]
                else:
                    row[metric_name] = metric_value
            
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Identify numeric metrics for aggregation
        numeric_metrics = [col for col in df.columns if col not in ["timestamp", "task_type", "worker_id", "status"] and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_metrics:
            logger.warning("No numeric metrics found for summary dashboard")
            return None
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=[
                "Task Distribution by Type",
                "Task Distribution by Status",
                "Worker Distribution",
                "Time Series of Task Execution",
                "Metric Distribution",
                "Metric Correlations"
            ],
            specs=[
                [{"type": "pie"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"colspan": 2, "type": "heatmap"}, None]
            ]
        )
        
        # 1. Task Distribution by Type (Pie chart)
        type_counts = df["task_type"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name="Task Types"
            ),
            row=1, col=1
        )
        
        # 2. Task Distribution by Status (Pie chart)
        status_counts = df["status"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name="Task Status"
            ),
            row=1, col=2
        )
        
        # 3. Worker Distribution (Bar chart)
        worker_counts = df["worker_id"].value_counts().reset_index()
        worker_counts.columns = ["worker_id", "count"]
        fig.add_trace(
            go.Bar(
                x=worker_counts["worker_id"],
                y=worker_counts["count"],
                name="Worker Tasks"
            ),
            row=2, col=1
        )
        
        # 4. Time Series of Task Execution (Scatter plot)
        df_count = df.set_index("timestamp").resample("1H").size().reset_index()
        df_count.columns = ["timestamp", "count"]
        fig.add_trace(
            go.Scatter(
                x=df_count["timestamp"],
                y=df_count["count"],
                mode="lines+markers",
                name="Tasks over Time"
            ),
            row=2, col=2
        )
        
        # 5. Metric Correlations (Heatmap)
        if len(numeric_metrics) > 1:
            corr = df[numeric_metrics].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale="Viridis",
                    name="Metric Correlations"
                ),
                row=3, col=1
            )
        
        # Improve layout
        fig.update_layout(
            title="Results Summary Dashboard",
            height=1000,
            width=1200,
            showlegend=False,
            hovermode="closest"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Worker ID", row=2, col=1)
        fig.update_yaxes(title_text="Task Count", row=2, col=1)
        
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Tasks per Hour", row=2, col=2)
        
        # Save to file if output_path is specified
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Summary dashboard saved to {output_path}")
        
        return fig


# Example usage when run directly
if __name__ == "__main__":
    import argparse
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from result_aggregator.service import ResultAggregatorService
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Result Aggregator Visualization")
    parser.add_argument("--db-path", default="./test_results.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="./visualizations", help="Directory to save visualizations")
    parser.add_argument("--type", choices=["performance", "trend", "anomaly", "summary"], default="performance", help="Type of visualization to generate")
    parser.add_argument("--metrics", nargs="+", help="Metrics to visualize")
    parser.add_argument("--interactive", action="store_true", default=True, help="Generate interactive visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize service
    service = ResultAggregatorService(db_path=args.db_path)
    
    # Create visualizer
    visualizer = ResultVisualizer(service)
    
    # Generate visualization based on type
    if args.type == "performance":
        output_path = os.path.join(args.output_dir, "performance_chart.html" if args.interactive else "performance_chart.png")
        visualizer.generate_performance_chart(
            metrics=args.metrics,
            output_path=output_path,
            interactive=args.interactive
        )
        print(f"Generated performance chart: {output_path}")
    
    elif args.type == "trend":
        output_path = os.path.join(args.output_dir, "trend_analysis.html" if args.interactive else "trend_analysis.png")
        visualizer.generate_trend_analysis(
            metrics=args.metrics,
            output_path=output_path,
            interactive=args.interactive
        )
        print(f"Generated trend analysis: {output_path}")
    
    elif args.type == "anomaly":
        output_path = os.path.join(args.output_dir, "anomaly_dashboard.html")
        visualizer.generate_anomaly_dashboard(
            output_path=output_path
        )
        print(f"Generated anomaly dashboard: {output_path}")
    
    elif args.type == "summary":
        output_path = os.path.join(args.output_dir, "summary_dashboard.html")
        visualizer.generate_summary_dashboard(
            output_path=output_path
        )
        print(f"Generated summary dashboard: {output_path}")
    
    # Close service
    service.close()