"""
Time Series Visualization Component for the Advanced Visualization System.

This module provides specialized time series visualization capabilities for tracking
performance metrics over time. It extends the BaseVisualization class with 
time-series specific visualization methods.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger("time_series_visualization")

# Import base visualization class
from data.duckdb.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE

# Check for plotly
if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

# Check for matplotlib
if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator


class TimeSeriesVisualization(BaseVisualization):
    """
    Time Series Visualization Component.
    
    This component creates visualizations for tracking performance metrics over time
    with trend detection, anomaly highlighting, and comparative analysis.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False):
        """Initialize the time series visualization component."""
        super().__init__(db_connection, theme, debug)
        
        # Additional configuration specific to time series visualizations
        self.time_series_config = {
            "date_format": "%Y-%m-%d",   # Format for date display
            "show_trend": True,          # Show trend line
            "show_anomalies": True,      # Highlight anomalies
            "anomaly_threshold": 2.0,    # Z-score threshold for anomaly detection
            "trend_window": 5,           # Window size for moving average trend
            "include_regression": True,  # Include regression line
            "interpolate_missing": True, # Interpolate missing values
            "confidence_interval": 0.95, # Confidence interval for trend (0-1)
            "line_width": 2,             # Line width for main series
            "marker_size": 8,            # Marker size for data points
            "compare_period": None,      # Period to compare with (e.g., 'previous_month')
        }
        
        # Metric-specific configurations
        self.metric_config = {
            "throughput": {
                "color": "#1f77b4",      # Blue
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"  # Higher is better
            },
            "latency": {
                "color": "#ff7f0e",      # Orange
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "down"  # Lower is better
            },
            "memory": {
                "color": "#2ca02c",      # Green
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "down"  # Lower is better
            },
            "power": {
                "color": "#d62728",      # Red
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "down"  # Lower is better
            },
            "efficiency": {
                "color": "#9467bd",      # Purple
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"  # Higher is better
            }
        }
        
        logger.info("Time Series Visualization component initialized")
    
    def create_visualization(self, data=None, **kwargs):
        """
        Create a time series visualization.
        
        This is a wrapper for the more specific create_time_series method.
        
        Args:
            data: Performance data
            **kwargs: Additional arguments passed to create_time_series
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        return self.create_time_series(data, **kwargs)
    
    def create_time_series(self,
                          data=None,
                          metric="throughput",
                          model_name=None,
                          hardware_type=None,
                          batch_size=None,
                          time_column="timestamp",
                          value_column=None,
                          group_by=None,
                          start_date=None,
                          end_date=None,
                          output_path=None,
                          title=None,
                          **kwargs):
        """
        Create a time series visualization for tracking performance metrics over time.
        
        Args:
            data: Performance data (DataFrame, dict, or path to file)
            metric: Metric to visualize ("throughput", "latency", "memory", etc.)
            model_name: Optional model name to filter by
            hardware_type: Optional hardware type to filter by
            batch_size: Optional batch size to filter by
            time_column: Column name containing timestamps
            value_column: Column name containing values (defaults to metric-specific column)
            group_by: Optional columns to group by (for multiple series)
            start_date: Optional start date for filtering (format: YYYY-MM-DD)
            end_date: Optional end date for filtering (format: YYYY-MM-DD)
            output_path: Optional path for saving the visualization
            title: Custom title for the visualization
            **kwargs: Additional configuration parameters
            
        Returns:
            Path to the saved time series visualization, or None if creation failed
        """
        # Update configuration with any provided kwargs
        time_series_config = self.time_series_config.copy()
        time_series_config.update(kwargs)
        
        # Determine metric column if value_column not specified
        if value_column is None:
            if metric == "throughput":
                value_column = "throughput_items_per_second"
            elif metric == "latency":
                value_column = "average_latency_ms"
            elif metric == "memory":
                value_column = "memory_peak_mb"
            elif metric == "power":
                value_column = "power_consumption"
            elif metric == "efficiency":
                value_column = "efficiency"
            else:
                # Use the metric name directly
                value_column = metric
        
        # Determine visualization title
        if title is None:
            title_parts = []
            title_parts.append(f"{metric.capitalize()} Over Time")
            if model_name:
                title_parts.append(f"for {model_name}")
            if hardware_type:
                title_parts.append(f"on {hardware_type}")
            if batch_size:
                title_parts.append(f"(Batch Size: {batch_size})")
            title = " ".join(title_parts)
        
        # Create sample data if not provided (for demonstration purposes)
        if data is None:
            # In a real implementation, this would fetch from the database
            if self.debug:
                logger.info("No data provided, using sample data for demonstration")
            
            # Generate sample date range
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=90)
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            # Set up sample data parameters
            if model_name is None:
                model_name = "BERT" if metric == "throughput" else "ViT"
            
            if hardware_type is None:
                hardware_types = ["CPU", "GPU", "WebGPU"]
            else:
                hardware_types = [hardware_type]
            
            if batch_size is None:
                batch_size = 1
            
            # Generate sample data
            rows = []
            for date in date_range:
                for hw in hardware_types:
                    # Base value depends on hardware and metric
                    if metric == "throughput":
                        # Higher values for GPU
                        base_value = 100 if hw == "GPU" else 50
                        # Add some trend over time
                        trend_factor = 1.0 + (date - start_dt).days / 180.0
                        value = base_value * trend_factor
                        # Add noise
                        value *= (1 + np.random.normal(0, 0.1))
                    elif metric == "latency":
                        # Lower values for GPU
                        base_value = 20 if hw == "GPU" else 40
                        # Add some trend over time (slight decrease)
                        trend_factor = 1.0 - (date - start_dt).days / 400.0
                        value = base_value * trend_factor
                        # Add noise
                        value *= (1 + np.random.normal(0, 0.15))
                    else:
                        # Other metrics
                        base_value = 50
                        value = base_value * (1 + np.random.normal(0, 0.2))
                    
                    # Add some outliers
                    if np.random.random() < 0.03:  # 3% chance of outlier
                        value *= 1.5 if np.random.random() < 0.5 else 0.5
                    
                    rows.append({
                        "timestamp": date,
                        "model_name": model_name,
                        "hardware_type": hw,
                        "batch_size": batch_size,
                        "throughput_items_per_second": value if metric == "throughput" else np.random.random() * 100,
                        "average_latency_ms": value if metric == "latency" else np.random.random() * 100,
                        "memory_peak_mb": value if metric == "memory" else np.random.random() * 100,
                    })
            
            df = pd.DataFrame(rows)
        else:
            # Load data from the provided source
            df = self.load_data(data)
            
            if df.empty:
                logger.error("Failed to load data for visualization")
                return None
        
        # Apply filters
        if model_name and "model_name" in df.columns:
            df = df[df["model_name"] == model_name]
        
        if hardware_type and "hardware_type" in df.columns:
            df = df[df["hardware_type"] == hardware_type]
        
        if batch_size is not None and "batch_size" in df.columns:
            df = df[df["batch_size"] == batch_size]
        
        if start_date and time_column in df.columns:
            if pd.api.types.is_string_dtype(df[time_column]):
                df = df[pd.to_datetime(df[time_column]) >= pd.to_datetime(start_date)]
            else:
                df = df[df[time_column] >= pd.to_datetime(start_date)]
        
        if end_date and time_column in df.columns:
            if pd.api.types.is_string_dtype(df[time_column]):
                df = df[pd.to_datetime(df[time_column]) <= pd.to_datetime(end_date)]
            else:
                df = df[df[time_column] <= pd.to_datetime(end_date)]
        
        # Check if required columns are available
        required_columns = [time_column, value_column]
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            required_columns.extend(group_by)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Ensure timestamp column is in datetime format
        if pd.api.types.is_string_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by timestamp
        df = df.sort_values(time_column)
        
        # Create the time series visualization based on available libraries
        if PLOTLY_AVAILABLE:
            return self._create_interactive_time_series(
                df, time_column, value_column, metric, group_by,
                output_path, title, time_series_config
            )
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_time_series(
                df, time_column, value_column, metric, group_by,
                output_path, title, time_series_config
            )
        else:
            logger.error("Neither Plotly nor Matplotlib is available. Cannot create visualization.")
            return None
    
    def _create_interactive_time_series(self, df, time_column, value_column, metric, group_by,
                                       output_path, title, time_series_config):
        """
        Create an interactive time series visualization using Plotly.
        
        Args:
            df: DataFrame with time series data
            time_column: Column name containing timestamps
            value_column: Column name containing values
            metric: Metric being visualized
            group_by: Optional columns to group by
            output_path: Path to save the visualization
            title: Title for the visualization
            time_series_config: Configuration for the time series
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Get metric configuration
            metric_config = self.metric_config.get(metric.lower(), {
                "color": "#1f77b4",
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"
            })
            
            # Get color palette for groups
            color_palette = px.colors.qualitative.Plotly  # Default color palette
            
            # If grouping, create multiple series
            if group_by:
                if isinstance(group_by, str):
                    group_by = [group_by]
                
                # Create a unique identifier for each group
                if len(group_by) > 1:
                    df['_group'] = df[group_by].apply(lambda row: ' - '.join(str(val) for val in row), axis=1)
                    groups = df['_group'].unique()
                else:
                    groups = df[group_by[0]].unique()
                    df['_group'] = df[group_by[0]]
                
                # Ensure we have enough colors
                while len(color_palette) < len(groups):
                    color_palette = color_palette * 2
                
                # Create a trace for each group
                for i, group in enumerate(sorted(groups)):
                    group_df = df[df['_group'] == group]
                    
                    # Skip if empty
                    if group_df.empty:
                        continue
                    
                    # Add main trace
                    fig.add_trace(go.Scatter(
                        x=group_df[time_column],
                        y=group_df[value_column],
                        mode='lines+markers',
                        name=str(group),
                        line=dict(
                            width=time_series_config["line_width"],
                            color=color_palette[i % len(color_palette)]
                        ),
                        marker=dict(
                            size=time_series_config["marker_size"],
                            color=color_palette[i % len(color_palette)]
                        )
                    ))
                    
                    # Add trend line if requested
                    if time_series_config["show_trend"] and len(group_df) >= time_series_config["trend_window"]:
                        # Create moving average
                        window = min(time_series_config["trend_window"], len(group_df))
                        group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                        group_df['trend'] = group_df[value_column].rolling(window=window, center=True).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=group_df[time_column],
                            y=group_df['trend'],
                            mode='lines',
                            name=f"{group} (Trend)",
                            line=dict(
                                width=time_series_config["line_width"],
                                color=color_palette[i % len(color_palette)],
                                dash='dash'
                            ),
                            opacity=0.7
                        ))
                    
                    # Add regression line if requested
                    if time_series_config["include_regression"] and len(group_df) >= 3:
                        try:
                            # Convert to numeric index for regression
                            group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                            group_df['_numeric_index'] = np.arange(len(group_df))
                            
                            # Calculate regression
                            z = np.polyfit(group_df['_numeric_index'], group_df[value_column], 1)
                            p = np.poly1d(z)
                            
                            # Add regression line
                            fig.add_trace(go.Scatter(
                                x=group_df[time_column],
                                y=p(group_df['_numeric_index']),
                                mode='lines',
                                name=f"{group} (Regression)",
                                line=dict(
                                    width=1,
                                    color=color_palette[i % len(color_palette)],
                                    dash='dot'
                                ),
                                opacity=0.5
                            ))
                        except Exception as e:
                            logger.warning(f"Error calculating regression for group {group}: {e}")
                    
                    # Add anomaly detection if requested
                    if time_series_config["show_anomalies"] and len(group_df) >= 4:
                        try:
                            # Calculate z-scores
                            group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                            z_scores = np.abs((group_df[value_column] - group_df[value_column].mean()) / group_df[value_column].std())
                            anomalies = group_df[z_scores > time_series_config["anomaly_threshold"]]
                            
                            if not anomalies.empty:
                                fig.add_trace(go.Scatter(
                                    x=anomalies[time_column],
                                    y=anomalies[value_column],
                                    mode='markers',
                                    name=f"{group} (Anomalies)",
                                    marker=dict(
                                        size=time_series_config["marker_size"] + 4,
                                        color=color_palette[i % len(color_palette)],
                                        symbol='x',
                                        line=dict(
                                            width=2,
                                            color='black'
                                        )
                                    )
                                ))
                        except Exception as e:
                            logger.warning(f"Error detecting anomalies for group {group}: {e}")
            else:
                # Just one series for the entire dataset
                
                # Add main trace
                fig.add_trace(go.Scatter(
                    x=df[time_column],
                    y=df[value_column],
                    mode='lines+markers',
                    name=metric.capitalize(),
                    line=dict(
                        width=time_series_config["line_width"],
                        color=metric_config["color"]
                    ),
                    marker=dict(
                        size=time_series_config["marker_size"],
                        color=metric_config["color"]
                    )
                ))
                
                # Add trend line if requested
                if time_series_config["show_trend"] and len(df) >= time_series_config["trend_window"]:
                    # Create moving average
                    window = min(time_series_config["trend_window"], len(df))
                    df_trend = df.copy()  # Create copy to avoid SettingWithCopyWarning
                    df_trend['trend'] = df_trend[value_column].rolling(window=window, center=True).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df_trend[time_column],
                        y=df_trend['trend'],
                        mode='lines',
                        name='Trend',
                        line=dict(
                            width=time_series_config["line_width"],
                            color=metric_config["color"],
                            dash='dash'
                        ),
                        opacity=0.7
                    ))
                
                # Add regression line if requested
                if time_series_config["include_regression"] and len(df) >= 3:
                    try:
                        # Convert to numeric index for regression
                        df_reg = df.copy()  # Create copy to avoid SettingWithCopyWarning
                        df_reg['_numeric_index'] = np.arange(len(df_reg))
                        
                        # Calculate regression
                        z = np.polyfit(df_reg['_numeric_index'], df_reg[value_column], 1)
                        p = np.poly1d(z)
                        
                        # Add regression line
                        fig.add_trace(go.Scatter(
                            x=df_reg[time_column],
                            y=p(df_reg['_numeric_index']),
                            mode='lines',
                            name='Regression',
                            line=dict(
                                width=1,
                                color=metric_config["color"],
                                dash='dot'
                            ),
                            opacity=0.5
                        ))
                        
                        # Add regression info
                        slope = z[0]
                        direction = "improving" if (
                            (slope > 0 and metric_config["better_direction"] == "up") or
                            (slope < 0 and metric_config["better_direction"] == "down")
                        ) else "degrading"
                        
                        # Calculate slope per day
                        time_diff = (df_reg[time_column].max() - df_reg[time_column].min()).total_seconds()
                        days = time_diff / (60 * 60 * 24)
                        daily_slope = slope * len(df_reg) / max(1, days)
                        
                        fig.add_annotation(
                            x=0.05,
                            y=0.05,
                            xref="paper",
                            yref="paper",
                            text=f"Trend: {direction.capitalize()} ({daily_slope:.3f} per day)",
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255,255,255,0.8)" if self.theme == "light" else "rgba(0,0,0,0.8)",
                            bordercolor=self.theme_colors.get("accent1", "blue"),
                            borderwidth=1,
                            borderpad=4
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating regression: {e}")
                
                # Add anomaly detection if requested
                if time_series_config["show_anomalies"] and len(df) >= 4:
                    try:
                        # Calculate z-scores
                        mean = df[value_column].mean()
                        std = df[value_column].std()
                        if std > 0:  # Avoid division by zero
                            z_scores = np.abs((df[value_column] - mean) / std)
                            anomalies = df[z_scores > time_series_config["anomaly_threshold"]]
                            
                            if not anomalies.empty:
                                fig.add_trace(go.Scatter(
                                    x=anomalies[time_column],
                                    y=anomalies[value_column],
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(
                                        size=time_series_config["marker_size"] + 4,
                                        color=metric_config["color"],
                                        symbol='x',
                                        line=dict(
                                            width=2,
                                            color='black'
                                        )
                                    )
                                ))
                    except Exception as e:
                        logger.warning(f"Error detecting anomalies: {e}")
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=metric.capitalize(),
                xaxis=dict(
                    type='date',
                    tickformat="%Y-%m-%d"
                ),
                template="plotly_white" if self.theme == "light" else "plotly_dark",
                hovermode="closest",
                width=1000,
                height=600
            )
            
            # Add range slider for interactivity
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Export visualization or show it
            if output_path:
                self.figure = fig
                self.export(output_path, format="html")
                return output_path
            else:
                self.figure = fig
                return self.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive time series: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_static_time_series(self, df, time_column, value_column, metric, group_by,
                                  output_path, title, time_series_config):
        """
        Create a static time series visualization using Matplotlib.
        
        Args:
            df: DataFrame with time series data
            time_column: Column name containing timestamps
            value_column: Column name containing values
            metric: Metric being visualized
            group_by: Optional columns to group by
            output_path: Path to save the visualization
            title: Title for the visualization
            time_series_config: Configuration for the time series
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get metric configuration
            metric_config = self.metric_config.get(metric.lower(), {
                "color": "#1f77b4",
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"
            })
            
            # Prepare color palette for groups
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            # If grouping, create multiple series
            if group_by:
                if isinstance(group_by, str):
                    group_by = [group_by]
                
                # Create a unique identifier for each group
                if len(group_by) > 1:
                    df['_group'] = df[group_by].apply(lambda row: ' - '.join(str(val) for val in row), axis=1)
                    groups = df['_group'].unique()
                else:
                    groups = df[group_by[0]].unique()
                    df['_group'] = df[group_by[0]]
                
                # Ensure we have enough colors
                while len(colors) < len(groups):
                    colors = colors * 2
                
                # Create a line for each group
                for i, group in enumerate(sorted(groups)):
                    group_df = df[df['_group'] == group]
                    
                    # Skip if empty
                    if group_df.empty:
                        continue
                    
                    # Plot main line
                    ax.plot(
                        group_df[time_column], 
                        group_df[value_column],
                        label=str(group),
                        color=colors[i % len(colors)],
                        linewidth=time_series_config["line_width"],
                        marker='o', 
                        markersize=time_series_config["marker_size"],
                        alpha=0.8
                    )
                    
                    # Add trend line if requested
                    if time_series_config["show_trend"] and len(group_df) >= time_series_config["trend_window"]:
                        # Create moving average
                        window = min(time_series_config["trend_window"], len(group_df))
                        group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                        group_df['trend'] = group_df[value_column].rolling(window=window, center=True).mean()
                        
                        ax.plot(
                            group_df[time_column],
                            group_df['trend'],
                            label=f"{group} (Trend)",
                            color=colors[i % len(colors)],
                            linestyle='--',
                            linewidth=time_series_config["line_width"] * 0.8,
                            alpha=0.6
                        )
                    
                    # Add regression line if requested
                    if time_series_config["include_regression"] and len(group_df) >= 3:
                        try:
                            # Convert time to numeric values for regression
                            group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                            group_df['_numeric_time'] = np.arange(len(group_df))
                            
                            # Calculate regression
                            z = np.polyfit(group_df['_numeric_time'], group_df[value_column], 1)
                            p = np.poly1d(z)
                            
                            # Add regression line
                            ax.plot(
                                group_df[time_column],
                                p(group_df['_numeric_time']),
                                label=f"{group} (Regression)",
                                color=colors[i % len(colors)],
                                linestyle=':',
                                linewidth=time_series_config["line_width"] * 0.6,
                                alpha=0.5
                            )
                        except Exception as e:
                            logger.warning(f"Error calculating regression for group {group}: {e}")
                    
                    # Add anomaly detection if requested
                    if time_series_config["show_anomalies"] and len(group_df) >= 4:
                        try:
                            # Calculate z-scores
                            group_df = group_df.copy()  # Create copy to avoid SettingWithCopyWarning
                            z_scores = np.abs((group_df[value_column] - group_df[value_column].mean()) / group_df[value_column].std())
                            anomalies = group_df[z_scores > time_series_config["anomaly_threshold"]]
                            
                            if not anomalies.empty:
                                ax.scatter(
                                    anomalies[time_column],
                                    anomalies[value_column],
                                    label=f"{group} (Anomalies)" if i == 0 else "_nolegend_",
                                    color=colors[i % len(colors)],
                                    marker='x',
                                    s=(time_series_config["marker_size"] + 4) ** 2,
                                    linewidth=2,
                                    edgecolor='black',
                                    zorder=10
                                )
                        except Exception as e:
                            logger.warning(f"Error detecting anomalies for group {group}: {e}")
            else:
                # Just one series for the entire dataset
                
                # Plot main line
                ax.plot(
                    df[time_column], 
                    df[value_column],
                    label=metric.capitalize(),
                    color=metric_config["color"],
                    linewidth=time_series_config["line_width"],
                    marker='o', 
                    markersize=time_series_config["marker_size"],
                    alpha=0.8
                )
                
                # Add trend line if requested
                if time_series_config["show_trend"] and len(df) >= time_series_config["trend_window"]:
                    # Create moving average
                    window = min(time_series_config["trend_window"], len(df))
                    df_trend = df.copy()  # Create copy to avoid SettingWithCopyWarning
                    df_trend['trend'] = df_trend[value_column].rolling(window=window, center=True).mean()
                    
                    ax.plot(
                        df_trend[time_column],
                        df_trend['trend'],
                        label="Trend",
                        color=metric_config["color"],
                        linestyle='--',
                        linewidth=time_series_config["line_width"] * 0.8,
                        alpha=0.6
                    )
                
                # Add regression line if requested
                if time_series_config["include_regression"] and len(df) >= 3:
                    try:
                        # Convert time to numeric values for regression
                        df_reg = df.copy()  # Create copy to avoid SettingWithCopyWarning
                        df_reg['_numeric_time'] = np.arange(len(df_reg))
                        
                        # Calculate regression
                        z = np.polyfit(df_reg['_numeric_time'], df_reg[value_column], 1)
                        p = np.poly1d(z)
                        
                        # Add regression line
                        ax.plot(
                            df_reg[time_column],
                            p(df_reg['_numeric_time']),
                            label="Regression",
                            color=metric_config["color"],
                            linestyle=':',
                            linewidth=time_series_config["line_width"] * 0.6,
                            alpha=0.5
                        )
                        
                        # Add regression info
                        slope = z[0]
                        direction = "improving" if (
                            (slope > 0 and metric_config["better_direction"] == "up") or
                            (slope < 0 and metric_config["better_direction"] == "down")
                        ) else "degrading"
                        
                        # Calculate slope per day
                        time_diff = (df_reg[time_column].max() - df_reg[time_column].min()).total_seconds()
                        days = time_diff / (60 * 60 * 24)
                        daily_slope = slope * len(df_reg) / max(1, days)
                        
                        # Add text annotation
                        plt.text(
                            0.05, 0.05, 
                            f"Trend: {direction.capitalize()} ({daily_slope:.3f} per day)",
                            transform=ax.transAxes,
                            fontsize=10,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white" if self.theme == "light" else "black",
                                edgecolor=metric_config["color"],
                                alpha=0.8
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error calculating regression: {e}")
                
                # Add anomaly detection if requested
                if time_series_config["show_anomalies"] and len(df) >= 4:
                    try:
                        # Calculate z-scores
                        mean = df[value_column].mean()
                        std = df[value_column].std()
                        if std > 0:  # Avoid division by zero
                            z_scores = np.abs((df[value_column] - mean) / std)
                            anomalies = df[z_scores > time_series_config["anomaly_threshold"]]
                            
                            if not anomalies.empty:
                                ax.scatter(
                                    anomalies[time_column],
                                    anomalies[value_column],
                                    label="Anomalies",
                                    color=metric_config["color"],
                                    marker='x',
                                    s=(time_series_config["marker_size"] + 4) ** 2,
                                    linewidth=2,
                                    edgecolor='black',
                                    zorder=10
                                )
                    except Exception as e:
                        logger.warning(f"Error detecting anomalies: {e}")
            
            # Format x-axis as dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_series_config["date_format"]))
            
            # Set labels and title
            ax.set_xlabel("Date")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend()
            
            # Format plot based on theme
            if self.theme == "dark":
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#333333')
                ax.set_facecolor('#333333')
            
            # Rotate date labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Export visualization or show it
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                self.figure = fig
                plt.show()
                return True
            
        except Exception as e:
            logger.error(f"Error creating static time series: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None