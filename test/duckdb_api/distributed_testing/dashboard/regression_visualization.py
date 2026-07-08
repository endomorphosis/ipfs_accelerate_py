#!/usr/bin/env python3
"""
Advanced Regression Visualization for Distributed Testing Dashboard

This module enhances the regression detection visualization capabilities with:
- Improved visual representation of regressions
- Interactive time-series regression plots
- Comparative regression visualizations (before/after)
- Multi-metric regression correlation analysis
- Export capabilities for regression reports
"""

import os
import logging
import numpy as np
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("regression_visualization")

# Try to import optional dependencies with graceful fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Some regression visualization features will be limited.")
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Visualization features will be limited.")
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available. Statistical analysis features will be limited.")
    SCIPY_AVAILABLE = False


class RegressionVisualization:
    """Advanced visualization tools for regression detection results."""
    
    def __init__(self, output_dir: str = "./visualizations/regression"):
        """Initialize the regression visualization engine.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Theme configuration
        self.theme = "dark"
        self.color_scheme = {
            "dark": {
                "background": "#222222",
                "text": "#f8f9fa",
                "primary": "#375a7f",
                "secondary": "#444444",
                "success": "#00bc8c",
                "warning": "#f39c12",
                "danger": "#e74c3c",
                "info": "#3498db"
            },
            "light": {
                "background": "#ffffff",
                "text": "#333333",
                "primary": "#007bff",
                "secondary": "#6c757d",
                "success": "#28a745",
                "warning": "#ffc107",
                "danger": "#dc3545", 
                "info": "#17a2b8"
            }
        }
        
        logger.info("Regression visualization engine initialized")
    
    def set_theme(self, theme: str):
        """Set the visualization theme (light or dark).
        
        Args:
            theme: Theme name (light or dark)
        """
        if theme in ["light", "dark"]:
            self.theme = theme
            logger.info(f"Set visualization theme to {theme}")
        else:
            logger.warning(f"Invalid theme: {theme}. Using default theme.")
    
    def create_interactive_regression_figure(self, 
                                           time_series_data: Dict[str, Any], 
                                           regressions: List[Dict[str, Any]],
                                           metric: str,
                                           title: Optional[str] = None,
                                           include_annotations: bool = True,
                                           include_confidence_intervals: bool = True,
                                           include_trend_lines: bool = True) -> Optional[Dict[str, Any]]:
        """Create an enhanced interactive regression visualization with additional features.
        
        Args:
            time_series_data: Dictionary containing timestamps and values
            regressions: List of detected regressions
            metric: Metric name
            title: Optional title for the visualization
            include_annotations: Whether to include regression annotations
            include_confidence_intervals: Whether to include confidence intervals
            include_trend_lines: Whether to include trend lines
            
        Returns:
            Plotly figure as dictionary if Plotly is available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create regression visualization.")
            return None
            
        # Get color scheme based on theme
        colors = self.color_scheme[self.theme]
        
        # Extract data from time series
        timestamps = time_series_data.get("timestamps", [])
        values = time_series_data.get("values", [])
        
        if not timestamps or not values or len(timestamps) != len(values):
            logger.warning("Invalid time series data format")
            return None
        
        # Create figure with secondary y-axis for p-values
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add original values
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=f"{metric} (Raw)",
                line=dict(width=1, color=colors["primary"]),
                marker=dict(size=4),
                customdata=np.arange(len(timestamps)),  # Add index as custom data
                hovertemplate=(
                    f"<b>{metric}</b><br>" +
                    "Date: %{x}<br>" +
                    "Value: %{y:.2f}<br>" +
                    "Index: %{customdata}<extra></extra>"
                )
            ),
            secondary_y=False
        )
        
        # Apply smoothing if there's enough data
        if len(values) > 5:
            # Apply exponential smoothing
            alpha = 0.2  # Smoothing factor
            smoothed = [values[0]]
            for i in range(1, len(values)):
                smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
            
            # Add smoothed line
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=smoothed,
                    mode="lines",
                    name=f"{metric} (Smoothed)",
                    line=dict(width=2, color=colors["info"]),
                    hoverinfo="skip"
                ),
                secondary_y=False
            )
        
        # If there are regressions, add visualizations for them
        shapes = []
        annotations = []
        
        # If trend lines are requested and there are regressions
        if include_trend_lines and regressions:
            for regression in regressions:
                if not regression.get("is_significant", False):
                    continue
                
                change_point_time = regression.get("change_point_time")
                if change_point_time not in timestamps:
                    continue
                
                # Find index of change point
                change_idx = timestamps.index(change_point_time)
                
                # Calculate segment ranges with padding
                window_size = min(10, len(timestamps) // 5)  # Adaptive window size
                before_start = max(0, change_idx - window_size)
                before_end = change_idx
                after_start = change_idx
                after_end = min(len(timestamps), change_idx + window_size)
                
                # Before segment trend line
                before_x = np.array(range(before_start, before_end))
                before_y = np.array(values[before_start:before_end])
                if len(before_x) > 1:
                    slope, intercept, _, _, _ = stats.linregress(before_x, before_y)
                    before_fit = slope * before_x + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[timestamps[i] for i in before_x],
                            y=before_fit,
                            mode="lines",
                            name="Before Trend",
                            line=dict(width=2, color=colors["success"], dash="dash"),
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
                
                # After segment trend line
                after_x = np.array(range(after_start, after_end))
                after_y = np.array(values[after_start:after_end])
                if len(after_x) > 1:
                    slope, intercept, _, _, _ = stats.linregress(after_x, after_y)
                    after_fit = slope * after_x + intercept
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[timestamps[i] for i in after_x],
                            y=after_fit,
                            mode="lines",
                            name="After Trend",
                            line=dict(width=2, color=colors["danger"], dash="dash"),
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
        
        # If confidence intervals are requested and there are regressions
        if include_confidence_intervals and regressions and SCIPY_AVAILABLE:
            for regression in regressions:
                if not regression.get("is_significant", False):
                    continue
                
                change_point_time = regression.get("change_point_time")
                if change_point_time not in timestamps:
                    continue
                
                # Find index of change point
                change_idx = timestamps.index(change_point_time)
                
                # Calculate confidence intervals for before segment
                window_size = min(10, len(timestamps) // 5)  # Adaptive window size
                before_start = max(0, change_idx - window_size)
                before_end = change_idx
                before_values = values[before_start:before_end]
                
                if len(before_values) >= 3:  # Need at least 3 points for confidence interval
                    before_mean = np.mean(before_values)
                    before_ci = stats.t.interval(0.95, len(before_values)-1, loc=before_mean, scale=stats.sem(before_values))
                    
                    # Add confidence interval for before segment
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps[before_start:before_end],
                            y=[before_ci[1]] * len(before_values),
                            mode="lines",
                            name="CI Upper (Before)",
                            line=dict(width=0),
                            marker=dict(color=colors["info"]),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps[before_start:before_end],
                            y=[before_ci[0]] * len(before_values),
                            mode="lines",
                            name="CI Lower (Before)",
                            fill="tonexty",
                            fillcolor=f"rgba(52, 152, 219, 0.2)",
                            line=dict(width=0),
                            marker=dict(color=colors["info"]),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
                
                # Calculate confidence intervals for after segment
                after_start = change_idx
                after_end = min(len(timestamps), change_idx + window_size)
                after_values = values[after_start:after_end]
                
                if len(after_values) >= 3:  # Need at least 3 points for confidence interval
                    after_mean = np.mean(after_values)
                    after_ci = stats.t.interval(0.95, len(after_values)-1, loc=after_mean, scale=stats.sem(after_values))
                    
                    # Add confidence interval for after segment
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps[after_start:after_end],
                            y=[after_ci[1]] * len(after_values),
                            mode="lines",
                            name="CI Upper (After)",
                            line=dict(width=0),
                            marker=dict(color=colors["danger"]),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps[after_start:after_end],
                            y=[after_ci[0]] * len(after_values),
                            mode="lines",
                            name="CI Lower (After)",
                            fill="tonexty",
                            fillcolor=f"rgba(231, 76, 60, 0.2)",
                            line=dict(width=0),
                            marker=dict(color=colors["danger"]),
                            showlegend=False,
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )
        
        # Add p-value trace if available
        has_p_values = False
        for regression in regressions:
            if "p_value" in regression:
                has_p_values = True
                break
                
        if has_p_values:
            # Create a trace for p-values at change points
            p_values = [None] * len(timestamps)
            for regression in regressions:
                if "p_value" in regression and "change_point_time" in regression:
                    change_point_time = regression["change_point_time"]
                    if change_point_time in timestamps:
                        idx = timestamps.index(change_point_time)
                        p_values[idx] = regression["p_value"]
            
            # Add p-value trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=p_values,
                    mode="markers",
                    name="p-value",
                    marker=dict(
                        size=10,
                        color=colors["warning"],
                        symbol="diamond"
                    ),
                    hovertemplate="p-value: %{y:.4f}<extra></extra>"
                ),
                secondary_y=True
            )
            
            # Configure secondary y-axis for p-values
            fig.update_yaxes(
                title_text="p-value",
                range=[0, 0.1],  # Focus on the significant range of p-values
                showgrid=False,
                secondary_y=True
            )
        
        # Add annotations and shapes for change points
        if include_annotations:
            for regression in regressions:
                if not regression.get("is_significant", False):
                    continue
                
                change_point_time = regression.get("change_point_time")
                if change_point_time not in timestamps:
                    continue
                
                # Find index of change point
                change_idx = timestamps.index(change_point_time)
                
                # Add vertical line at change point
                shape_color = colors["danger"] if regression.get("is_regression", False) else colors["success"]
                shapes.append(
                    dict(
                        type="line",
                        x0=change_point_time,
                        y0=min(values),
                        x1=change_point_time,
                        y1=max(values),
                        line=dict(
                            color=shape_color,
                            width=2,
                            dash="dash",
                        )
                    )
                )
                
                # Add annotation
                direction = "▲" if regression.get("direction") == "increase" else "▼"
                color = colors["danger"] if regression.get("is_regression", False) else colors["success"]
                
                # Format percentage change with proper sign
                pct_change = regression.get('percentage_change', 0)
                pct_text = f"{direction} {abs(pct_change):.1f}%"
                
                # Add p-value if available
                p_value = regression.get('p_value', None)
                if p_value is not None:
                    significance = 1.0 - p_value if p_value < 1.0 else 0.0
                    pct_text += f"<br>p: {p_value:.4f}"
                    pct_text += f"<br>sig: {significance:.1%}"
                
                annotations.append(
                    dict(
                        x=change_point_time,
                        y=values[change_idx],
                        xref="x",
                        yref="y",
                        text=pct_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        font=dict(
                            size=12,
                            color=color
                        ),
                        bordercolor=color,
                        borderwidth=2,
                        borderpad=4,
                        bgcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
                        opacity=0.8
                    )
                )
        
        # Update figure with shapes and annotations
        fig.update_layout(
            shapes=shapes,
            annotations=annotations
        )
        
        # Add horizontal line at mean value
        mean_value = np.mean(values)
        fig.add_shape(
            type="line",
            x0=timestamps[0],
            y0=mean_value,
            x1=timestamps[-1],
            y1=mean_value,
            line=dict(
                color=colors["secondary"],
                width=1,
                dash="dot",
            )
        )
        
        # Update layout
        title = title or f"{metric} with Regression Detection"
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=f"{metric}",
            showlegend=True,
            hovermode="closest",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=500,
            width=900,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"])
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05
        )
        
        return fig.to_dict()
    
    def _dict_to_figure(self, fig_dict: Dict[str, Any]):
        """Convert a figure dictionary back to a plotly.graph_objects.Figure object.
        
        This helper method is primarily for testing and export functionality.
        
        Args:
            fig_dict: Dictionary representation of a Plotly figure
            
        Returns:
            plotly.graph_objects.Figure object
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot convert dict to Figure.")
            return None
            
        return go.Figure(fig_dict)
    
    def create_comparative_regression_visualization(self,
                                                 metrics_data: Dict[str, Dict[str, Any]],
                                                 regressions_by_metric: Dict[str, List[Dict[str, Any]]],
                                                 title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a comparative visualization showing multiple metrics with regressions.
        
        Args:
            metrics_data: Dictionary of time series data by metric
            regressions_by_metric: Dictionary of detected regressions by metric
            title: Optional title for the visualization
            
        Returns:
            Plotly figure as dictionary if Plotly is available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create comparative visualization.")
            return None
        
        if not metrics_data or not regressions_by_metric:
            logger.warning("No metrics data or regressions provided.")
            return None
        
        # Get color scheme based on theme
        colors = self.color_scheme[self.theme]
        
        # Create figure with subplots
        metrics = list(metrics_data.keys())
        num_metrics = len(metrics)
        
        if num_metrics == 0:
            logger.warning("No metrics to visualize.")
            return None
        
        # Create subplot grid layout
        if num_metrics == 1:
            rows, cols = 1, 1
        elif num_metrics == 2:
            rows, cols = 2, 1
        elif num_metrics <= 4:
            rows, cols = 2, 2
        else:
            # For more metrics, use 3 columns
            rows = (num_metrics + 2) // 3
            cols = 3
            
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[metric for metric in metrics]
        )
        
        # Add traces for each metric
        for i, metric in enumerate(metrics):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Extract data
            timestamps = metrics_data[metric].get("timestamps", [])
            values = metrics_data[metric].get("values", [])
            
            if not timestamps or not values:
                continue
                
            # Add trace for this metric
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    name=metric,
                    line=dict(width=1),
                    marker=dict(size=3)
                ),
                row=row, col=col
            )
            
            # Add regression markers and annotations
            regressions = regressions_by_metric.get(metric, [])
            
            for regression in regressions:
                if not regression.get("is_significant", False):
                    continue
                
                change_point_time = regression.get("change_point_time")
                if change_point_time not in timestamps:
                    continue
                
                # Find index of change point
                change_idx = timestamps.index(change_point_time)
                
                # Add marker for regression
                marker_color = colors["danger"] if regression.get("is_regression", False) else colors["success"]
                
                fig.add_trace(
                    go.Scatter(
                        x=[change_point_time],
                        y=[values[change_idx]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=marker_color,
                            symbol="star"
                        ),
                        name=f"Regression - {metric}",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{metric}</b><br>" +
                            "Date: %{x}<br>" +
                            "Value: %{y:.2f}<br>" +
                            f"Change: {regression.get('percentage_change', 0):.1f}%<br>" +
                            f"p-value: {regression.get('p_value', 'N/A')}<extra></extra>"
                        )
                    ),
                    row=row, col=col
                )
                
                # Add vertical line at change point
                fig.add_shape(
                    type="line",
                    x0=change_point_time,
                    y0=min(values),
                    x1=change_point_time,
                    y1=max(values),
                    line=dict(
                        color=marker_color,
                        width=1,
                        dash="dash",
                    ),
                    row=row, col=col
                )
        
        # Update layout
        title = title or "Comparative Regression Analysis"
        fig.update_layout(
            title=title,
            showlegend=True,
            height=300 * rows,
            width=400 * cols,
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"])
        )
        
        return fig.to_dict()
    
    def create_regression_heatmap(self,
                               time_ranges: List[str],
                               metrics: List[str],
                               regression_data: Dict[str, Dict[str, float]],
                               title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a heatmap visualization of regressions across time ranges and metrics.
        
        Args:
            time_ranges: List of time range labels
            metrics: List of metric names
            regression_data: Nested dictionary mapping time ranges to metrics to percentage changes
            title: Optional title for the visualization
            
        Returns:
            Plotly figure as dictionary if Plotly is available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create heatmap visualization.")
            return None
        
        if not time_ranges or not metrics or not regression_data:
            logger.warning("Insufficient data for heatmap visualization.")
            return None
        
        # Get color scheme based on theme
        colors = self.color_scheme[self.theme]
        
        # Create matrix of values for heatmap
        z = []
        hover_texts = []
        
        for time_range in time_ranges:
            row = []
            hover_row = []
            
            for metric in metrics:
                # Get percentage change if available
                metric_data = regression_data.get(time_range, {}).get(metric, None)
                
                if metric_data is None:
                    value = 0
                    hover_text = "No data"
                else:
                    value = metric_data
                    hover_text = f"Change: {value:.1f}%"
                    
                    # Add significance if available
                    p_value = regression_data.get(time_range, {}).get(f"{metric}_p_value", None)
                    if p_value is not None:
                        hover_text += f"<br>p-value: {p_value:.4f}"
                
                row.append(value)
                hover_row.append(hover_text)
            
            z.append(row)
            hover_texts.append(hover_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=metrics,
            y=time_ranges,
            colorscale='RdBu_r',  # Red for negative, blue for positive
            zmid=0,  # Center the colorscale at 0
            text=hover_texts,
            hovertemplate="Time: %{y}<br>Metric: %{x}<br>%{text}<extra></extra>"
        ))
        
        # Update layout
        title = title or "Regression Heatmap"
        fig.update_layout(
            title=title,
            height=600,
            width=800,
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"])
        )
        
        return fig.to_dict()
    
    def create_regression_summary_report(self,
                                      metrics_data: Dict[str, Dict[str, Any]],
                                      regressions_by_metric: Dict[str, List[Dict[str, Any]]],
                                      output_path: Optional[str] = None,
                                      include_plots: bool = True) -> str:
        """Generate a comprehensive HTML report of regression analysis results.
        
        Args:
            metrics_data: Dictionary of time series data by metric
            regressions_by_metric: Dictionary of detected regressions by metric
            output_path: Optional path to save the report
            include_plots: Whether to include interactive plots in the report
            
        Returns:
            Path to the generated report
        """
        if not metrics_data or not regressions_by_metric:
            logger.warning("No metrics data or regressions provided.")
            return None
        
        # Generate default output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"regression_report_{timestamp}.html")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get color scheme based on theme
        colors = self.color_scheme[self.theme]
        
        # Prepare HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regression Analysis Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: {colors["background"]};
                    color: {colors["text"]};
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: {colors["primary"]};
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .header h1 {{
                    margin: 0;
                    color: white;
                }}
                .header p {{
                    margin: 5px 0 0 0;
                    color: rgba(255, 255, 255, 0.8);
                }}
                .section {{
                    background-color: {colors["secondary"]};
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .section h2 {{
                    margin-top: 0;
                    color: {colors["text"]};
                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                    padding-bottom: 10px;
                }}
                .plot-container {{
                    margin: 20px 0;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 5px;
                    overflow: hidden;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }}
                th {{
                    background-color: rgba(0, 0, 0, 0.2);
                }}
                .regression-critical {{
                    color: {colors["danger"]};
                }}
                .regression-high {{
                    color: {colors["warning"]};
                }}
                .regression-medium {{
                    color: {colors["info"]};
                }}
                .regression-low {{
                    color: {colors["success"]};
                }}
                .regression-none {{
                    color: {colors["text"]};
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Regression Analysis Report</h1>
                    <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Metrics Analyzed</h4>
                            <p>{len(metrics_data)} metrics analyzed</p>
                            <ul>
        """
        
        # Add metric names to summary
        for metric in metrics_data.keys():
            html_content += f"<li>{metric}</li>\n"
        
        # Count regressions by severity
        total_regressions = 0
        significant_regressions = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "none": 0}
        
        for metric, regressions in regressions_by_metric.items():
            for regression in regressions:
                total_regressions += 1
                if regression.get("is_significant", False):
                    significant_regressions += 1
                    severity = regression.get("severity", "none")
                    severity_counts[severity] += 1
        
        # Continue with HTML content
        html_content += f"""
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h4>Regression Summary</h4>
                            <p>Total regressions detected: {total_regressions}</p>
                            <p>Statistically significant regressions: {significant_regressions}</p>
                            <ul>
                                <li class="regression-critical">Critical: {severity_counts["critical"]}</li>
                                <li class="regression-high">High: {severity_counts["high"]}</li>
                                <li class="regression-medium">Medium: {severity_counts["medium"]}</li>
                                <li class="regression-low">Low: {severity_counts["low"]}</li>
                            </ul>
                        </div>
                    </div>
                </div>
        """
        
        # Add detailed regression section
        html_content += """
                <div class="section">
                    <h2>Detailed Regression Analysis</h2>
        """
        
        # Add comparative visualization if plots are requested
        if include_plots and PLOTLY_AVAILABLE and len(metrics_data) > 1:
            comparative_fig = self.create_comparative_regression_visualization(
                metrics_data, 
                regressions_by_metric,
                title="Comparative Regression Analysis"
            )
            
            if comparative_fig:
                import plotly.io as pio
                comparative_html = pio.to_html(comparative_fig, full_html=False)
                
                html_content += f"""
                    <div class="plot-container">
                        {comparative_html}
                    </div>
                """
        
        # Add individual metric sections
        for metric, metric_data in metrics_data.items():
            regressions = regressions_by_metric.get(metric, [])
            
            # Skip metrics with no regressions
            if not regressions:
                continue
                
            html_content += f"""
                    <div class="subsection mt-4">
                        <h3>{metric}</h3>
            """
            
            # Add interactive visualization if requested
            if include_plots and PLOTLY_AVAILABLE:
                regression_fig = self.create_interactive_regression_figure(
                    metric_data,
                    regressions,
                    metric,
                    title=f"{metric} Analysis"
                )
                
                if regression_fig:
                    import plotly.io as pio
                    regression_html = pio.to_html(regression_fig, full_html=False)
                    
                    html_content += f"""
                        <div class="plot-container">
                            {regression_html}
                        </div>
                    """
            
            # Add regression details table
            html_content += """
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Change (%)</th>
                                    <th>Before</th>
                                    <th>After</th>
                                    <th>p-value</th>
                                    <th>Severity</th>
                                    <th>Type</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add rows for each regression
            for regression in regressions:
                if not regression.get("is_significant", False):
                    continue
                    
                # Format date
                change_point_time = regression.get("change_point_time", "Unknown")
                if isinstance(change_point_time, str):
                    date_str = change_point_time
                else:
                    # Attempt to format timestamp
                    try:
                        date_str = str(change_point_time)
                    except:
                        date_str = "Unknown"
                
                # Get regression details
                pct_change = regression.get("percentage_change", 0)
                before_mean = regression.get("before_mean", 0)
                after_mean = regression.get("after_mean", 0)
                p_value = regression.get("p_value", 1.0)
                severity = regression.get("severity", "none")
                is_regression = regression.get("is_regression", False)
                
                # Format values
                pct_change_str = f"{pct_change:.2f}%"
                before_str = f"{before_mean:.2f}"
                after_str = f"{after_mean:.2f}"
                p_value_str = f"{p_value:.4f}"
                type_str = "Regression" if is_regression else "Improvement"
                
                # Add table row
                html_content += f"""
                                <tr>
                                    <td>{date_str}</td>
                                    <td class="regression-{severity}">{pct_change_str}</td>
                                    <td>{before_str}</td>
                                    <td>{after_str}</td>
                                    <td>{p_value_str}</td>
                                    <td class="regression-{severity}">{severity.title()}</td>
                                    <td>{"Regression" if is_regression else "Improvement"}</td>
                                </tr>
                """
            
            # Close table and section
            html_content += """
                            </tbody>
                        </table>
                    </div>
            """
        
        # Close the detailed section
        html_content += """
                </div>
        """
        
        # Add correlation analysis section if applicable
        if len(metrics_data) > 1 and PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
            # Create correlation matrix
            metric_values = {}
            for metric, metric_data in metrics_data.items():
                values = metric_data.get("values", [])
                if values:
                    metric_values[metric] = values
            
            if metric_values and len(metric_values) > 1:
                # Convert to DataFrame
                import pandas as pd
                df = pd.DataFrame(metric_values)
                
                # Calculate correlation matrix
                corr_matrix = df.corr()
                
                # Create heatmap visualization
                corr_fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale="RdBu",
                        zmid=0,
                        text=[[f"{x:.2f}" for x in row] for row in corr_matrix.values],
                        texttemplate="%{text}",
                        hovertemplate="Correlation between %{y} and %{x}: %{z:.3f}<extra></extra>"
                    )
                )
                
                # Update layout
                corr_fig.update_layout(
                    title="Metric Correlation Analysis",
                    height=600,
                    width=800,
                    template="plotly_dark" if self.theme == "dark" else "plotly_white",
                    margin=dict(l=50, r=50, t=100, b=50),
                    plot_bgcolor=colors["background"],
                    paper_bgcolor=colors["background"],
                    font=dict(color=colors["text"])
                )
                
                # Convert to HTML
                import plotly.io as pio
                corr_html = pio.to_html(corr_fig, full_html=False)
                
                # Add correlation section
                html_content += f"""
                <div class="section">
                    <h2>Correlation Analysis</h2>
                    <div class="plot-container">
                        {corr_html}
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <h4>Correlation Insights</h4>
                            <ul>
                """
                
                # Add correlation insights
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr = corr_matrix.iloc[i, j]
                        
                        if abs(corr) > 0.5:
                            corr_type = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs(corr) > 0.7 else "moderate"
                            
                            html_content += f"""
                                <li>{strength.title()} {corr_type} correlation ({corr:.2f}) between {col1} and {col2}</li>
                            """
                
                # Close correlation section
                html_content += """
                            </ul>
                        </div>
                    </div>
                </div>
                """
        
        # Close HTML content
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Regression summary report created: {output_path}")
        return output_path
    
    def export_regression_visualization(self,
                                     figure_dict: Dict[str, Any],
                                     output_path: Optional[str] = None,
                                     format: str = "html") -> str:
        """Export a regression visualization to file.
        
        Args:
            figure_dict: Plotly figure dictionary
            output_path: Optional path to save the export
            format: Export format (html, png, svg, pdf, json)
            
        Returns:
            Path to the exported visualization
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot export visualization.")
            return None
        
        if not figure_dict:
            logger.warning("No figure provided for export.")
            return None
        
        # Generate default output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"regression_viz_{timestamp}.{format}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert figure dictionary to Figure object
        fig = go.Figure(figure_dict)
        
        # Export based on format
        if format == "html":
            fig.write_html(output_path)
        elif format == "png":
            fig.write_image(output_path)
        elif format == "svg":
            fig.write_image(output_path)
        elif format == "pdf":
            fig.write_image(output_path)
        elif format == "json":
            with open(output_path, "w") as f:
                import json
                json.dump(figure_dict, f)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return None
        
        logger.info(f"Visualization exported to: {output_path}")
        return output_path