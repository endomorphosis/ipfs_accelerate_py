"""
Animated Time Series Visualization Component for the Advanced Visualization System.

This module provides specialized animated time-series visualization capabilities for
tracking performance metrics over time with animation controls, trend analysis, and
interactive filtering.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logger = logging.getLogger("animated_time_series_visualization")

# Import base visualization class
from duckdb_api.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE

# Check for plotly
if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

# Check for matplotlib
if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.animation as animation
    from matplotlib.ticker import MaxNLocator


class AnimatedTimeSeriesVisualization(BaseVisualization):
    """
    Animated Time Series Visualization Component.
    
    This component creates animated visualizations for tracking performance metrics over time
    with interactive animation controls, trend analysis, and comparative filtering.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False):
        """Initialize the animated time series visualization component."""
        super().__init__(db_connection, theme, debug)
        
        # Additional configuration specific to animated time series visualizations
        self.animated_ts_config = {
            "date_format": "%Y-%m-%d",     # Format for date display
            "show_trend": True,            # Show trend line
            "show_anomalies": True,        # Highlight anomalies
            "anomaly_threshold": 2.0,      # Z-score threshold for anomaly detection
            "trend_window": 5,             # Window size for moving average trend
            "include_regression": True,    # Include regression line
            "interpolate_missing": True,   # Interpolate missing values
            "confidence_interval": 0.95,   # Confidence interval for trend (0-1)
            "line_width": 2,               # Line width for main series
            "marker_size": 8,              # Marker size for data points
            "animation_frame": "date",     # Column to use for animation frames
            "animation_speed": 1000,       # Animation speed in milliseconds
            "progressive_display": True,   # Show data progressively as it accumulates
            "show_timeline_slider": True,  # Show timeline slider control
            "control_buttons": True,       # Show play/pause/step buttons
            "speed_selector": True,        # Show speed selector for animation
            "transition_duration": 300,    # Transition duration between frames
            "show_history_traces": True,   # Show historical data as lighter traces
            "max_traces_history": 10,      # Maximum number of historical traces to show
            "highlight_last_point": True,  # Highlight the last point in each trace
            "events": [],                  # List of event markers to show on timeline
            "allow_comparative": True,     # Allow comparative visualization of metrics
            "export_formats": ["html", "mp4", "gif"], # Supported export formats
            "time_range": 90,              # Default time range in days
            "time_interval": "day",        # Default time aggregation interval
            "color_palette": None,         # Optional custom color palette
            "normalize_values": False,     # Normalize values for comparison
            "detect_anomalies": True,      # Auto-detect anomalies in data
            "comparative_dimension": None, # Dimension to use for comparative analysis
        }
        
        # Time interval options
        self.time_intervals = {
            "hour": {"pandas_freq": "H", "timedelta": timedelta(hours=1), "format": "%Y-%m-%d %H:00"},
            "day": {"pandas_freq": "D", "timedelta": timedelta(days=1), "format": "%Y-%m-%d"},
            "week": {"pandas_freq": "W", "timedelta": timedelta(weeks=1), "format": "%Y-%m-%d"},
            "month": {"pandas_freq": "M", "timedelta": timedelta(days=30), "format": "%Y-%m"}
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
        
        logger.info("Animated Time Series Visualization component initialized")
    
    def create_visualization(self, data=None, **kwargs):
        """
        Create an animated time series visualization.
        
        This is a wrapper for the more specific create_animated_time_series method.
        
        Args:
            data: Performance data
            **kwargs: Additional arguments passed to create_animated_time_series
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        return self.create_animated_time_series(data, **kwargs)
    
    def create_animated_time_series(self,
                                   data=None,
                                   metric="throughput",
                                   dimensions=None,
                                   time_column="timestamp",
                                   value_column=None,
                                   time_range=None,
                                   time_interval=None,
                                   start_date=None,
                                   end_date=None,
                                   filters=None,
                                   events=None,
                                   output_path=None,
                                   title=None,
                                   **kwargs):
        """
        Create an animated time series visualization for tracking performance metrics over time.
        
        Args:
            data: Performance data (DataFrame, dict, or path to file)
            metric: Metric to visualize ("throughput", "latency", "memory", etc.)
            dimensions: List of dimensions to group by (e.g., ["model_family", "hardware_type"])
            time_column: Column name containing timestamps
            value_column: Column name containing values (defaults to metric-specific column)
            time_range: Number of days to include (e.g., 90 for last 90 days)
            time_interval: Time aggregation interval ("hour", "day", "week", "month")
            start_date: Optional start date for filtering (format: YYYY-MM-DD)
            end_date: Optional end date for filtering (format: YYYY-MM-DD)
            filters: Dictionary of filters to apply, e.g., {"batch_size": [1, 4, 16]}
            events: List of events to mark on timeline, e.g., [{"date": "2025-06-01", "label": "v2.0 Release"}]
            output_path: Optional path for saving the visualization
            title: Custom title for the visualization
            **kwargs: Additional configuration parameters
            
        Returns:
            Path to the saved animated time series visualization, or None if creation failed
        """
        # Update configuration with any provided kwargs
        animated_ts_config = self.animated_ts_config.copy()
        animated_ts_config.update(kwargs)
        
        # Set default dimensions if not provided
        if dimensions is None:
            dimensions = ["model_family", "hardware_type"]
        elif isinstance(dimensions, str):
            dimensions = [dimensions]
        
        # Set default time interval if not provided
        if time_interval is None:
            time_interval = animated_ts_config.get("time_interval", "day")
        
        # Add events if provided
        if events is not None:
            animated_ts_config["events"] = events
        
        # Determine metric column if value_column not specified
        if value_column is None:
            if metric == "throughput":
                value_column = "throughput_items_per_second"
            elif metric == "latency":
                value_column = "average_latency_ms"
            elif metric == "memory":
                value_column = "memory_peak_mb"
            elif metric == "power":
                value_column = "power_consumption_w"
            elif metric == "efficiency":
                value_column = "efficiency"
            else:
                # Use the metric name directly
                value_column = metric
        
        # Determine visualization title
        if title is None:
            title_parts = []
            title_parts.append(f"{metric.capitalize()} Trends Over Time")
            if len(dimensions) > 0:
                title_parts.append(f"by {', '.join(dim.replace('_', ' ').title() for dim in dimensions)}")
            title = " ".join(title_parts)
        
        # Create sample data if not provided (for demonstration purposes)
        if data is None:
            # In a real implementation, this would fetch from the database
            if self.debug:
                logger.info("No data provided, using sample data for demonstration")
            
            # Generate sample date range
            end_dt = datetime.now()
            if time_range:
                start_dt = end_dt - timedelta(days=time_range)
            else:
                start_dt = end_dt - timedelta(days=animated_ts_config.get("time_range", 90))
            
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Generate dates according to the time interval
            interval_config = self.time_intervals.get(time_interval, self.time_intervals["day"])
            date_range = pd.date_range(start=start_dt, end=end_dt, freq=interval_config["pandas_freq"])
            
            # Set up sample data parameters
            model_families = ["Text", "Vision", "Audio"]
            hardware_types = ["CPU", "GPU", "WebGPU", "WebNN"]
            
            # Generate sample data
            rows = []
            
            # Progressive trend factors - allow values to change over time with trends
            hw_trend_factors = {
                "CPU": 1.0,
                "GPU": 1.2,
                "WebGPU": 1.15,
                "WebNN": 1.1
            }
            
            family_trend_factors = {
                "Text": 1.0,
                "Vision": 1.05,
                "Audio": 0.95
            }
            
            # Generate data for each date
            for date in date_range:
                # Calculate days from start for trend progression
                days_progression = (date - start_dt).days / max(1, (end_dt - start_dt).days)
                
                for family in model_families:
                    for hw in hardware_types:
                        # Base value depends on hardware and metric
                        if metric == "throughput":
                            # Higher values for GPU/WebGPU
                            base_value = 100 if hw in ["GPU", "WebGPU"] else 50
                            # Add progressive improvement over time (hardware gets faster)
                            improvement = 1.0 + days_progression * 0.5 * hw_trend_factors[hw] * family_trend_factors[family]
                            value = base_value * improvement
                            # Add noise and some weekly pattern
                            day_of_week_factor = 1.0 + 0.05 * (date.weekday() % 7) / 6.0
                            value *= day_of_week_factor * (1 + np.random.normal(0, 0.05))
                        elif metric == "latency":
                            # Lower values for GPU/WebGPU
                            base_value = 20 if hw in ["GPU", "WebGPU"] else 40
                            # Latency improves (decreases) over time
                            improvement = 1.0 - days_progression * 0.3 * hw_trend_factors[hw] * family_trend_factors[family]
                            value = base_value * max(0.5, improvement)
                            # Add noise and some weekly pattern
                            day_of_week_factor = 1.0 + 0.05 * (date.weekday() % 7) / 6.0
                            value *= day_of_week_factor * (1 + np.random.normal(0, 0.08))
                        elif metric == "memory":
                            # Memory usage increases slightly over time
                            base_value = 500 if hw in ["GPU", "WebGPU"] else 300
                            trend = 1.0 + days_progression * 0.2  # Memory usage tends to grow
                            value = base_value * trend * (1 + np.random.normal(0, 0.05))
                        else:
                            # Other metrics - create some basic trends
                            base_value = 50 
                            trend = 1.0 + days_progression * 0.3
                            value = base_value * trend * (1 + np.random.normal(0, 0.1))
                        
                        # Add some special events with significant changes
                        if len(animated_ts_config["events"]) > 0:
                            for event in animated_ts_config["events"]:
                                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                                # If this date is just after an event, add an effect
                                if date >= event_date and date < event_date + timedelta(days=7):
                                    days_after = (date - event_date).days
                                    if days_after < 3:
                                        # Immediate effect after event
                                        effect_magnitude = 0.2 * (3 - days_after) / 3.0
                                        if "effect" in event and event["effect"] == "negative":
                                            value *= (1.0 - effect_magnitude)
                                        else:
                                            value *= (1.0 + effect_magnitude)
                        
                        # Add occasional anomalies
                        if animated_ts_config["detect_anomalies"] and np.random.random() < 0.02:
                            # 2% chance of an anomaly
                            if metric in ["throughput", "efficiency"]:
                                # Throughput/efficiency can spike up or (more commonly) down
                                if np.random.random() < 0.3:
                                    # 30% chance of positive anomaly
                                    value *= 1.5 + np.random.random() * 0.5  # 1.5-2x increase
                                else:
                                    # 70% chance of negative anomaly
                                    value *= 0.4 + np.random.random() * 0.3  # 40-70% decrease
                            else:
                                # Latency/memory more commonly spike up
                                if np.random.random() < 0.7:
                                    # 70% chance of negative anomaly (higher value)
                                    value *= 1.5 + np.random.random() * 1.0  # 1.5-2.5x increase
                                else:
                                    # 30% chance of positive anomaly (lower value)
                                    value *= 0.5 + np.random.random() * 0.3  # 50-80% decrease
                        
                        # Add row
                        row = {
                            time_column: date,
                            "model_family": family,
                            "hardware_type": hw,
                            "batch_size": 1,
                            "throughput_items_per_second": value if metric == "throughput" else base_value * (1 + np.random.normal(0, 0.15)),
                            "average_latency_ms": value if metric == "latency" else base_value * (1 + np.random.normal(0, 0.15)),
                            "memory_peak_mb": value if metric == "memory" else base_value * (1 + np.random.normal(0, 0.1)),
                            "power_consumption_w": value if metric == "power" else base_value * (1 + np.random.normal(0, 0.1)),
                            "efficiency": value if metric == "efficiency" else base_value * (1 + np.random.normal(0, 0.15))
                        }
                        
                        # Add additional dimensions for richer data
                        if "model_name" not in row:
                            if family == "Text":
                                row["model_name"] = np.random.choice(["BERT", "T5", "LLAMA"])
                            elif family == "Vision":
                                row["model_name"] = np.random.choice(["ViT", "CLIP", "DETR"])
                            else:
                                row["model_name"] = np.random.choice(["Whisper", "CLAP"])
                        
                        rows.append(row)
            
            df = pd.DataFrame(rows)
        else:
            # Load data from the provided source
            df = self.load_data(data)
            
            if df.empty:
                logger.error("Failed to load data for visualization")
                return None
        
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Apply time range filter if specified
        if time_range:
            end_date_dt = df[time_column].max()
            start_date_dt = end_date_dt - timedelta(days=time_range)
            df = df[(df[time_column] >= start_date_dt) & (df[time_column] <= end_date_dt)]
        
        # Apply date filters if specified
        if start_date:
            start_date_dt = pd.to_datetime(start_date)
            df = df[df[time_column] >= start_date_dt]
        
        if end_date:
            end_date_dt = pd.to_datetime(end_date)
            df = df[df[time_column] <= end_date_dt]
        
        # Apply additional filters if provided
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    if isinstance(value, list):
                        df = df[df[key].isin(value)]
                    else:
                        df = df[df[key] == value]
        
        # Aggregate data by time interval if needed
        interval_config = self.time_intervals.get(time_interval, self.time_intervals["day"])
        
        # Add a truncated date column based on the time interval
        df['date'] = df[time_column].dt.to_period(interval_config["pandas_freq"]).dt.to_timestamp()
        
        # Check if required columns and dimensions are available
        required_columns = [time_column, 'date', value_column]
        required_columns.extend(dimensions)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Create the animated time series visualization based on available libraries
        if PLOTLY_AVAILABLE:
            return self._create_interactive_animated_time_series(
                df, time_column, value_column, 'date', metric, dimensions,
                output_path, title, animated_ts_config
            )
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_animated_time_series(
                df, time_column, value_column, 'date', metric, dimensions,
                output_path, title, animated_ts_config
            )
        else:
            logger.error("Neither Plotly nor Matplotlib is available. Cannot create visualization.")
            return None
    
    def _create_interactive_animated_time_series(self, df, time_column, value_column, date_column, 
                                                metric, dimensions, output_path, title, 
                                                animated_ts_config):
        """
        Create an interactive animated time series visualization using Plotly.
        
        Args:
            df: DataFrame with time series data
            time_column: Original timestamp column
            value_column: Column containing metric values
            date_column: Column with truncated dates for animation
            metric: Name of the metric being visualized
            dimensions: List of dimensions to group by
            output_path: Path to save the visualization
            title: Title for the visualization
            animated_ts_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Get metric configuration
            metric_config = self.metric_config.get(metric.lower(), {
                "color": "#1f77b4",
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"
            })
            
            # Group data by dimensions
            if dimensions and len(dimensions) > 0:
                # Create group identifier for better readability
                df['_group'] = df[dimensions].apply(
                    lambda row: ' - '.join(str(val) for val in row), 
                    axis=1
                )
                groups = df['_group'].unique()
            else:
                # No dimensions to group by
                df['_group'] = 'All Data'
                groups = ['All Data']
            
            # Sort dates
            all_dates = sorted(df['date'].unique())
            
            # Calculate trend lines and detect anomalies if configured
            group_dfs = {}
            for group in groups:
                group_df = df[df['_group'] == group].copy()
                group_df = group_df.sort_values('date')
                
                # Add trend calculation
                if animated_ts_config["show_trend"] and len(group_df) >= animated_ts_config["trend_window"]:
                    window = min(animated_ts_config["trend_window"], len(group_df))
                    group_df['trend'] = group_df[value_column].rolling(window=window, center=True).mean()
                
                # Add anomaly detection
                if animated_ts_config["show_anomalies"] and len(group_df) >= 4:
                    mean = group_df[value_column].mean()
                    std = group_df[value_column].std()
                    if std > 0:  # Avoid division by zero
                        z_scores = np.abs((group_df[value_column] - mean) / std)
                        group_df['is_anomaly'] = z_scores > animated_ts_config["anomaly_threshold"]
                    else:
                        group_df['is_anomaly'] = False
                else:
                    group_df['is_anomaly'] = False
                
                # Store processed group dataframe
                group_dfs[group] = group_df
            
            # Create figure with animation and traces
            fig = self._create_animated_figure(
                group_dfs, groups, all_dates, value_column, metric, 
                title, animated_ts_config
            )
            
            # Configure animation
            if animated_ts_config["control_buttons"]:
                self._configure_animation_controls(fig, animated_ts_config)
            
            # Add event markers if specified
            if animated_ts_config["events"]:
                self._add_event_markers(fig, animated_ts_config["events"])
            
            # Export visualization or show it
            if output_path:
                self.figure = fig
                self.export(output_path, format="html")
                return output_path
            else:
                self.figure = fig
                return self.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive animated time series: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_animated_figure(self, group_dfs, groups, all_dates, value_column, 
                               metric, title, animated_ts_config):
        """
        Create the animated figure with all traces and frames.
        
        Args:
            group_dfs: Dictionary of processed dataframes for each group
            groups: List of group names
            all_dates: Sorted list of all dates
            value_column: Column containing metric values
            metric: Name of the metric being visualized
            title: Title for the visualization
            animated_ts_config: Configuration for the visualization
            
        Returns:
            Plotly figure with animation
        """
        # Create figure
        fig = go.Figure()
        
        # Get color palette - either custom or default Plotly colors
        if animated_ts_config["color_palette"]:
            color_palette = animated_ts_config["color_palette"]
        else:
            color_palette = px.colors.qualitative.Plotly
            
        # Ensure enough colors
        while len(color_palette) < len(groups):
            color_palette = color_palette * 2
        
        # Add a trace for each group
        for i, group in enumerate(groups):
            group_df = group_dfs[group]
            color = color_palette[i % len(color_palette)]
            
            # Initial empty trace for each group (will be filled by frames)
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode='lines+markers',
                    name=group,
                    line=dict(
                        width=animated_ts_config["line_width"],
                        color=color
                    ),
                    marker=dict(
                        size=animated_ts_config["marker_size"],
                        color=color
                    ),
                    hovertemplate=f"<b>{group}</b><br>Date: %{{x|" + 
                                 animated_ts_config["date_format"] + 
                                 "}}<br>Value: %{y:.2f}<extra></extra>"
                )
            )
            
            # Add trend line if configured
            if animated_ts_config["show_trend"]:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        name=f"{group} (Trend)",
                        line=dict(
                            width=animated_ts_config["line_width"],
                            color=color,
                            dash='dash'
                        ),
                        opacity=0.7,
                        hovertemplate=f"<b>{group} (Trend)</b><br>Date: %{{x|" + 
                                     animated_ts_config["date_format"] + 
                                     "}}<br>Value: %{y:.2f}<extra></extra>"
                    )
                )
            
            # Add anomaly markers if configured
            if animated_ts_config["show_anomalies"]:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        name=f"{group} (Anomalies)",
                        marker=dict(
                            size=animated_ts_config["marker_size"] + 6,
                            color=color,
                            symbol='x',
                            line=dict(
                                width=2,
                                color='black'
                            )
                        ),
                        hovertemplate=f"<b>{group} (Anomaly)</b><br>Date: %{{x|" + 
                                     animated_ts_config["date_format"] + 
                                     "}}<br>Value: %{y:.2f}<extra></extra>"
                    )
                )
        
        # Create frames for animation
        frames = []
        
        # Create a frame for each date showing the data up to that date
        for i, date in enumerate(all_dates):
            frame_data = []
            
            # Frame data includes all points up to and including the current date
            for group in groups:
                group_df = group_dfs[group]
                
                # If progressive display, show all data up to this date
                if animated_ts_config["progressive_display"]:
                    visible_df = group_df[group_df['date'] <= date]
                else:
                    # Otherwise just show current date
                    visible_df = group_df[group_df['date'] == date]
                
                # Add main trace data
                frame_data.append(
                    go.Scatter(
                        x=visible_df['date'],
                        y=visible_df[value_column]
                    )
                )
                
                # Add trend trace data
                if animated_ts_config["show_trend"]:
                    frame_data.append(
                        go.Scatter(
                            x=visible_df['date'],
                            y=visible_df['trend'] if 'trend' in visible_df.columns else []
                        )
                    )
                
                # Add anomaly trace data
                if animated_ts_config["show_anomalies"]:
                    anomalies = visible_df[visible_df['is_anomaly']]
                    frame_data.append(
                        go.Scatter(
                            x=anomalies['date'],
                            y=anomalies[value_column]
                        )
                    )
            
            # Create frame for this date
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(date)
                )
            )
        
        # Add frames to figure
        fig.frames = frames
        
        # Configure animation settings
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None, 
                                {
                                    "frame": {"duration": animated_ts_config["animation_speed"], "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": animated_ts_config["transition_duration"]}
                                }
                            ]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None], 
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate"
                                }
                            ]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "x": 0.1,
                    "y": 0
                }
            ]
        )
        
        # Add slider if requested
        if animated_ts_config["show_timeline_slider"]:
            steps = []
            for i, date in enumerate(all_dates):
                date_str = pd.to_datetime(date).strftime(animated_ts_config["date_format"])
                step = {
                    "args": [
                        [str(date)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate"
                        }
                    ],
                    "label": date_str,
                    "method": "animate"
                }
                steps.append(step)
            
            sliders = [
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 14},
                        "prefix": "Date: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": animated_ts_config["transition_duration"]},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": steps
                }
            ]
            
            fig.update_layout(sliders=sliders)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric.capitalize(),
            xaxis=dict(
                type='date',
                tickformat=animated_ts_config["date_format"]
            ),
            template="plotly_white" if self.theme == "light" else "plotly_dark",
            hovermode="closest",
            width=1000,
            height=600
        )
        
        return fig
    
    def _configure_animation_controls(self, fig, animated_ts_config):
        """
        Configure animation controls including speed selector and step buttons.
        
        Args:
            fig: Plotly figure to configure
            animated_ts_config: Configuration for the visualization
        """
        # Update existing buttons
        buttons = fig.layout.updatemenus[0].buttons
        
        # Add step forward/backward buttons if there are frames
        if fig.frames:
            step_forward = {
                "label": "Next",
                "method": "animate",
                "args": [
                    [fig.frames[1].name if len(fig.frames) > 1 else fig.frames[0].name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate"
                    }
                ]
            }
            
            step_backward = {
                "label": "Prev",
                "method": "animate",
                "args": [
                    [fig.frames[-1].name if len(fig.frames) > 1 else fig.frames[0].name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate"
                    }
                ]
            }
            
            # Add buttons to menu
            updated_buttons = buttons + [step_backward, step_forward]
            fig.layout.updatemenus[0].buttons = updated_buttons
        
        # Add speed selector if configured
        if animated_ts_config["speed_selector"]:
            # Define speed options
            speeds = [
                {"label": "0.5x", "speed": animated_ts_config["animation_speed"] * 2},
                {"label": "1x", "speed": animated_ts_config["animation_speed"]},
                {"label": "2x", "speed": animated_ts_config["animation_speed"] / 2},
                {"label": "5x", "speed": animated_ts_config["animation_speed"] / 5}
            ]
            
            # Create speed menu buttons
            speed_buttons = []
            for speed_option in speeds:
                speed_button = {
                    "label": speed_option["label"],
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": speed_option["speed"], "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": animated_ts_config["transition_duration"]}
                        }
                    ]
                }
                speed_buttons.append(speed_button)
            
            # Add speed menu
            speed_menu = {
                "type": "buttons",
                "showactive": False,
                "buttons": speed_buttons,
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "x": 0.3,
                "y": 0,
                "title": {"text": "Speed:"}
            }
            
            # Update menus
            fig.layout.updatemenus = list(fig.layout.updatemenus) + [speed_menu]
    
    def _add_event_markers(self, fig, events):
        """
        Add event markers to the visualization.
        
        Args:
            fig: Plotly figure to add markers to
            events: List of event dictionaries with date, label, and optional color
        """
        if not events:
            return
        
        annotations = []
        shapes = []
        
        for event in events:
            event_date = pd.to_datetime(event["date"])
            label = event.get("label", "Event")
            color = event.get("color", "red")
            
            # Add vertical line
            shapes.append(
                {
                    "type": "line",
                    "x0": event_date,
                    "x1": event_date,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {
                        "color": color,
                        "width": 2,
                        "dash": "dash"
                    }
                }
            )
            
            # Add annotation
            annotations.append(
                {
                    "x": event_date,
                    "y": 1.05,
                    "yref": "paper",
                    "text": label,
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "arrowwidth": 1,
                    "arrowcolor": color,
                    "bgcolor": "rgba(255, 255, 255, 0.8)" if self.theme == "light" else "rgba(0, 0, 0, 0.8)",
                    "bordercolor": color,
                    "borderwidth": 1,
                    "borderpad": 4,
                    "font": {"size": 10}
                }
            )
        
        # Add shapes and annotations to figure
        fig.update_layout(
            shapes=shapes,
            annotations=annotations
        )
    
    def _create_static_animated_time_series(self, df, time_column, value_column, date_column, 
                                           metric, dimensions, output_path, title, 
                                           animated_ts_config):
        """
        Create a static animated time series visualization using Matplotlib.
        
        Since Matplotlib can't produce interactive animations for HTML output, 
        this function creates a static time series visualization with the option
        to save animations as GIF or MP4.
        
        Args:
            df: DataFrame with time series data
            time_column: Original timestamp column
            value_column: Column containing metric values
            date_column: Column with truncated dates for animation
            metric: Name of the metric being visualized
            dimensions: List of dimensions to group by
            output_path: Path to save the visualization
            title: Title for the visualization
            animated_ts_config: Configuration for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Get metric configuration
            metric_config = self.metric_config.get(metric.lower(), {
                "color": "#1f77b4",
                "line_style": "solid",
                "marker": "circle",
                "better_direction": "up"
            })
            
            # Group data by dimensions
            if dimensions and len(dimensions) > 0:
                # Create group identifier for better readability
                df['_group'] = df[dimensions].apply(
                    lambda row: ' - '.join(str(val) for val in row), 
                    axis=1
                )
                groups = df['_group'].unique()
            else:
                # No dimensions to group by
                df['_group'] = 'All Data'
                groups = ['All Data']
            
            # Sort dates
            all_dates = sorted(df['date'].unique())
            
            # Calculate trend lines and detect anomalies if configured
            group_dfs = {}
            for group in groups:
                group_df = df[df['_group'] == group].copy()
                group_df = group_df.sort_values('date')
                
                # Add trend calculation
                if animated_ts_config["show_trend"] and len(group_df) >= animated_ts_config["trend_window"]:
                    window = min(animated_ts_config["trend_window"], len(group_df))
                    group_df['trend'] = group_df[value_column].rolling(window=window, center=True).mean()
                
                # Add anomaly detection
                if animated_ts_config["show_anomalies"] and len(group_df) >= 4:
                    mean = group_df[value_column].mean()
                    std = group_df[value_column].std()
                    if std > 0:  # Avoid division by zero
                        z_scores = np.abs((group_df[value_column] - mean) / std)
                        group_df['is_anomaly'] = z_scores > animated_ts_config["anomaly_threshold"]
                    else:
                        group_df['is_anomaly'] = False
                else:
                    group_df['is_anomaly'] = False
                
                # Store processed group dataframe
                group_dfs[group] = group_df
            
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get color cycle
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            # Ensure enough colors
            while len(colors) < len(groups):
                colors = colors * 2
            
            # Define update function for animation
            def update(frame_idx):
                ax.clear()
                current_date = all_dates[frame_idx]
                
                # Draw data for each group up to current date
                for i, group in enumerate(groups):
                    group_df = group_dfs[group]
                    color = colors[i % len(colors)]
                    
                    # Filter data to current date
                    if animated_ts_config["progressive_display"]:
                        visible_df = group_df[group_df['date'] <= current_date]
                    else:
                        visible_df = group_df[group_df['date'] == current_date]
                    
                    # Skip if no data
                    if visible_df.empty:
                        continue
                    
                    # Plot main line
                    ax.plot(
                        visible_df['date'], 
                        visible_df[value_column],
                        label=group,
                        color=color,
                        linewidth=animated_ts_config["line_width"],
                        marker='o', 
                        markersize=animated_ts_config["marker_size"]
                    )
                    
                    # Add trend line if requested
                    if animated_ts_config["show_trend"] and 'trend' in visible_df.columns:
                        trend_df = visible_df.dropna(subset=['trend'])
                        if not trend_df.empty:
                            ax.plot(
                                trend_df['date'],
                                trend_df['trend'],
                                label=f"{group} (Trend)" if i == 0 else None,
                                color=color,
                                linestyle='--',
                                linewidth=animated_ts_config["line_width"] * 0.8,
                                alpha=0.7
                            )
                    
                    # Add anomaly markers if requested
                    if animated_ts_config["show_anomalies"]:
                        anomalies = visible_df[visible_df['is_anomaly']]
                        if not anomalies.empty:
                            ax.scatter(
                                anomalies['date'],
                                anomalies[value_column],
                                label=f"{group} (Anomalies)" if i == 0 else None,
                                color=color,
                                marker='x',
                                s=(animated_ts_config["marker_size"] + 6) ** 2,
                                linewidth=2,
                                edgecolor='black',
                                zorder=10
                            )
                
                # Add event markers if requested
                if animated_ts_config["events"]:
                    for event in animated_ts_config["events"]:
                        event_date = pd.to_datetime(event["date"])
                        if event_date <= current_date:
                            ax.axvline(
                                x=event_date,
                                color=event.get("color", "red"),
                                linestyle='--',
                                alpha=0.7
                            )
                            ax.text(
                                event_date, 
                                ax.get_ylim()[1] * 0.95,
                                event.get("label", "Event"),
                                ha='center',
                                va='top',
                                rotation=90,
                                bbox=dict(
                                    boxstyle="round,pad=0.3",
                                    fc='white',
                                    ec=event.get("color", "red"),
                                    alpha=0.8
                                )
                            )
                
                # Format x-axis as dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter(animated_ts_config["date_format"]))
                
                # Set labels and title
                ax.set_xlabel("Date")
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f"{title}\nDate: {current_date.strftime(animated_ts_config['date_format'])}")
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend
                ax.legend()
                
                # Format plot based on theme
                if self.theme == "dark":
                    plt.style.use('dark_background')
                    fig.patch.set_facecolor('#333333')
                    ax.set_facecolor('#333333')
                
                # Rotate date labels for better readability
                plt.xticks(rotation=45)
                
                # Adjust layout
                plt.tight_layout()
            
            # Create animation
            ani = animation.FuncAnimation(
                fig, 
                update, 
                frames=len(all_dates),
                interval=animated_ts_config["animation_speed"],
                repeat=True
            )
            
            # Save static image for the last frame
            update(len(all_dates) - 1)
            
            # Save animation if output format is specified
            if output_path:
                # Get file extension
                _, ext = os.path.splitext(output_path)
                
                if ext.lower() in ['.mp4', '.gif']:
                    # Save as animation
                    if ext.lower() == '.mp4':
                        # Save as MP4 video
                        writer = animation.FFMpegWriter(
                            fps=1000 / animated_ts_config["animation_speed"],
                            metadata=dict(title=title, artist='IPFS Accelerate Framework'),
                            bitrate=1800
                        )
                        ani.save(output_path, writer=writer)
                    else:
                        # Save as GIF
                        ani.save(output_path, writer='pillow', fps=1000 / animated_ts_config["animation_speed"])
                elif ext.lower() == '.html':
                    # Matplotlib doesn't support HTML output directly
                    # Save as PNG instead and return the PNG path
                    png_path = output_path.replace('.html', '.png')
                    plt.savefig(png_path, dpi=100, bbox_inches="tight")
                    logger.info(f"Matplotlib doesn't support HTML output. Saved static image as {png_path}")
                    return png_path
                else:
                    # Save as static image
                    plt.savefig(output_path, dpi=100, bbox_inches="tight")
                
                plt.close(fig)
                return output_path
            else:
                self.figure = fig
                plt.show()
                return True
            
        except Exception as e:
            logger.error(f"Error creating static animated time series: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def export_animation(self, output_format="mp4", output_path=None, fps=None):
        """
        Export the current animation to a file format.
        
        Args:
            output_format: Format to export to ("mp4", "gif")
            output_path: Path to save the exported file
            fps: Frames per second (default based on animation_speed)
            
        Returns:
            Path to the exported file, or None if export failed
        """
        if self.figure is None:
            logger.error("No visualization has been created")
            return None
        
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly is required for exporting animations")
            return None
        
        try:
            # Default output path if none provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"animated_time_series_{timestamp}.{output_format}"
            
            # Default FPS based on animation speed
            if fps is None:
                fps = int(1000 / self.animated_ts_config["animation_speed"])
            
            # Export based on format
            if output_format.lower() == "mp4":
                # Export as MP4 video
                self.figure.write_html(
                    output_path.replace(".mp4", ".html"),
                    include_plotlyjs="cdn",
                    auto_play=True
                )
                
                # For MP4, we would need additional libraries like moviepy or ImageMagick
                # For now, export HTML with auto-play enabled and inform user
                logger.info(f"MP4 export not directly supported. HTML with auto-play saved to {output_path.replace('.mp4', '.html')}")
                return output_path.replace(".mp4", ".html")
                
            elif output_format.lower() == "gif":
                # Export as GIF (requires additional libraries)
                try:
                    import kaleido
                    
                    # Create a temporary directory for frames
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    
                    # Export frames
                    frame_paths = []
                    for i, frame in enumerate(self.figure.frames):
                        frame_fig = go.Figure(
                            data=frame.data,
                            layout=self.figure.layout
                        )
                        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                        frame_fig.write_image(frame_path)
                        frame_paths.append(frame_path)
                    
                    # Combine frames into GIF
                    try:
                        from PIL import Image
                        
                        frames = [Image.open(frame) for frame in frame_paths]
                        frames[0].save(
                            output_path,
                            save_all=True,
                            append_images=frames[1:],
                            optimize=True,
                            duration=int(1000 / fps),
                            loop=0
                        )
                        
                        # Clean up temporary files
                        for frame_path in frame_paths:
                            os.remove(frame_path)
                        os.rmdir(temp_dir)
                        
                        return output_path
                    except ImportError:
                        logger.error("PIL is required for GIF export")
                        return None
                    
                except ImportError:
                    logger.error("Kaleido is required for frame export. Install with: pip install kaleido")
                    return None
            else:
                # HTML export (default)
                self.figure.write_html(
                    output_path.replace(f".{output_format}", ".html"),
                    include_plotlyjs="cdn"
                )
                return output_path.replace(f".{output_format}", ".html")
                
        except Exception as e:
            logger.error(f"Error exporting animation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None